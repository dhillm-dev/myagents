"""
WF Swarm Strategy - AI-driven trading strategy with swarm consensus
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..config import get_config
from ..bus import get_message_bus, publish_trade_signal
from ..agents.chart_agent import get_chart_agent
from ..agents.sentiment_agent import get_sentiment_agent
from ..agents.swarm_agent import get_swarm_agent
from ..agents.risk_agent import get_risk_agent

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass
class StrategySignal:
    """WF Swarm strategy signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    market_regime: MarketRegime
    swarm_consensus: str
    technical_score: float
    sentiment_score: float
    risk_score: float
    reasoning: str
    timestamp: datetime


class WFSwarmStrategy:
    """
    WF Swarm Strategy - Combines multiple AI agents with swarm intelligence
    
    This strategy uses:
    1. Technical analysis (chart patterns, indicators)
    2. Sentiment analysis (Perplexity Finance API)
    3. Swarm consensus (multiple AI perspectives)
    4. Risk management (position sizing, drawdown protection)
    """
    
    def __init__(self):
        self.config = get_config()
        self.name = "WF_SWARM"
        
        # Strategy parameters
        self.min_confidence = 0.7
        self.min_swarm_agreement = 0.65
        self.max_positions = 10
        self.position_timeout = timedelta(hours=24)
        
        # Market regime detection
        self.regime_cache = {}
        self.regime_cache_duration = timedelta(minutes=15)
        
        # Signal tracking
        self.active_signals = {}
        self.signal_history = []
        self.max_history = 1000
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
    
    async def analyze_symbol(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze a symbol and generate trading signal"""
        try:
            logger.info(f"WF Swarm analyzing {symbol}")
            
            # Get market regime
            market_regime = await self._detect_market_regime(symbol)
            
            # Gather analysis from all agents
            analysis_data = await self._gather_analysis(symbol)
            if not analysis_data:
                logger.warning(f"Failed to gather analysis for {symbol}")
                return None
            
            # Calculate component scores
            technical_score = await self._calculate_technical_score(analysis_data['chart'])
            sentiment_score = await self._calculate_sentiment_score(analysis_data['sentiment'])
            
            # Get swarm consensus
            swarm_result = await self._get_swarm_consensus(symbol, analysis_data)
            if not swarm_result:
                logger.warning(f"Failed to get swarm consensus for {symbol}")
                return None
            
            # Calculate overall confidence
            confidence = await self._calculate_confidence(
                technical_score, sentiment_score, swarm_result, market_regime
            )
            
            # Determine action
            action = await self._determine_action(
                technical_score, sentiment_score, swarm_result, confidence, market_regime
            )
            
            if action == 'HOLD' or confidence < self.min_confidence:
                logger.info(f"No signal for {symbol}: action={action}, confidence={confidence:.2f}")
                return None
            
            # Calculate entry, stop loss, and take profit
            entry_price, stop_loss, take_profit = await self._calculate_levels(
                symbol, action, analysis_data['chart'], market_regime
            )
            
            # Calculate position size
            quantity = await self._calculate_position_size(
                symbol, action, entry_price, stop_loss, confidence
            )
            
            if quantity <= 0:
                logger.warning(f"Invalid position size for {symbol}")
                return None
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(
                symbol, action, entry_price, stop_loss, quantity, market_regime
            )
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(
                action, technical_score, sentiment_score, swarm_result, market_regime
            )
            
            # Create signal
            signal = StrategySignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                market_regime=market_regime,
                swarm_consensus=swarm_result.consensus.value,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                risk_score=risk_score,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            # Store signal
            self.active_signals[symbol] = signal
            self._add_to_history(signal)
            self.stats['signals_generated'] += 1
            
            logger.info(f"WF Swarm signal: {symbol} {action} (confidence: {confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"WF Swarm analysis failed for {symbol}: {e}")
            return None
    
    async def _detect_market_regime(self, symbol: str) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Check cache
            if symbol in self.regime_cache:
                cached_time, cached_regime = self.regime_cache[symbol]
                if datetime.now() - cached_time < self.regime_cache_duration:
                    return cached_regime
            
            # Get chart analysis
            chart_agent = await get_chart_agent()
            chart_data = await chart_agent.analyze_symbol(symbol)
            
            if not chart_data:
                return MarketRegime.SIDEWAYS
            
            # Analyze trend and volatility
            trend = chart_data.get('trend', 'NEUTRAL')
            volatility = chart_data.get('volatility', 'MEDIUM')
            
            # Determine regime
            if trend == 'BULLISH' and volatility in ['LOW', 'MEDIUM']:
                regime = MarketRegime.BULL_TREND
            elif trend == 'BEARISH' and volatility in ['LOW', 'MEDIUM']:
                regime = MarketRegime.BEAR_TREND
            elif volatility == 'HIGH':
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility == 'LOW':
                regime = MarketRegime.LOW_VOLATILITY
            else:
                regime = MarketRegime.SIDEWAYS
            
            # Cache result
            self.regime_cache[symbol] = (datetime.now(), regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"Market regime detection failed for {symbol}: {e}")
            return MarketRegime.SIDEWAYS
    
    async def _gather_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gather analysis from all agents"""
        try:
            # Get agents
            chart_agent = await get_chart_agent()
            sentiment_agent = await get_sentiment_agent()
            
            # Gather analysis in parallel
            tasks = [
                chart_agent.analyze_symbol(symbol),
                sentiment_agent.analyze_sentiment(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            chart_data = results[0] if not isinstance(results[0], Exception) else None
            sentiment_data = results[1] if not isinstance(results[1], Exception) else None
            
            if not chart_data or not sentiment_data:
                logger.error(f"Missing analysis data for {symbol}")
                return None
            
            return {
                'chart': chart_data,
                'sentiment': sentiment_data
            }
            
        except Exception as e:
            logger.error(f"Failed to gather analysis for {symbol}: {e}")
            return None
    
    async def _calculate_technical_score(self, chart_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score (0-1)"""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # Signal strength
            signal = chart_data.get('signal', 'HOLD')
            confidence = chart_data.get('confidence', 0.0)
            
            if signal in ['BUY', 'SELL']:
                score += confidence * 0.4
                weight_sum += 0.4
            
            # Trend alignment
            trend = chart_data.get('trend', 'NEUTRAL')
            if trend == 'BULLISH':
                score += 0.8 * 0.2
            elif trend == 'BEARISH':
                score += 0.2 * 0.2
            else:
                score += 0.5 * 0.2
            weight_sum += 0.2
            
            # Pattern detection
            patterns = chart_data.get('patterns', [])
            if patterns:
                pattern_score = min(len(patterns) * 0.2, 1.0)
                score += pattern_score * 0.2
            weight_sum += 0.2
            
            # Volume confirmation
            volume_trend = chart_data.get('volume_trend', 'NEUTRAL')
            if volume_trend == 'INCREASING':
                score += 0.8 * 0.1
            elif volume_trend == 'DECREASING':
                score += 0.3 * 0.1
            else:
                score += 0.5 * 0.1
            weight_sum += 0.1
            
            # Support/resistance levels
            if chart_data.get('support_level') and chart_data.get('resistance_level'):
                score += 0.7 * 0.1
            weight_sum += 0.1
            
            return score / weight_sum if weight_sum > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Technical score calculation failed: {e}")
            return 0.5
    
    async def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate sentiment score (0-1)"""
        try:
            sentiment_score = sentiment_data.get('sentiment_score', 5.0) / 10.0
            tradability_score = sentiment_data.get('tradability_score', 5.0) / 10.0
            
            # Weighted combination
            combined_score = (sentiment_score * 0.7) + (tradability_score * 0.3)
            
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Sentiment score calculation failed: {e}")
            return 0.5
    
    async def _get_swarm_consensus(self, symbol: str, analysis_data: Dict[str, Any]):
        """Get swarm consensus"""
        try:
            swarm_agent = await get_swarm_agent()
            
            # Get market data (simplified)
            market_data = {
                'symbol': symbol,
                'price': 0.0,  # Would get from market connector
                'volume': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            consensus = await swarm_agent.get_consensus(
                symbol, analysis_data['chart'], analysis_data['sentiment'], market_data
            )
            
            return consensus
            
        except Exception as e:
            logger.error(f"Swarm consensus failed for {symbol}: {e}")
            return None
    
    async def _calculate_confidence(
        self,
        technical_score: float,
        sentiment_score: float,
        swarm_result,
        market_regime: MarketRegime
    ) -> float:
        """Calculate overall confidence"""
        try:
            # Base confidence from component scores
            base_confidence = (technical_score * 0.4) + (sentiment_score * 0.3)
            
            # Add swarm confidence
            if swarm_result:
                base_confidence += swarm_result.confidence * 0.3
            
            # Adjust for market regime
            regime_multiplier = 1.0
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                regime_multiplier = 0.8  # Reduce confidence in high volatility
            elif market_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                regime_multiplier = 1.1  # Increase confidence in trending markets
            
            final_confidence = base_confidence * regime_multiplier
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _determine_action(
        self,
        technical_score: float,
        sentiment_score: float,
        swarm_result,
        confidence: float,
        market_regime: MarketRegime
    ) -> str:
        """Determine trading action"""
        try:
            # Get swarm consensus action
            swarm_action = 'HOLD'
            if swarm_result:
                consensus_value = swarm_result.consensus.value
                if 'BUY' in consensus_value:
                    swarm_action = 'BUY'
                elif 'SELL' in consensus_value:
                    swarm_action = 'SELL'
            
            # Calculate buy/sell scores
            buy_score = 0.0
            sell_score = 0.0
            
            # Technical contribution
            if technical_score > 0.6:
                buy_score += technical_score * 0.4
            elif technical_score < 0.4:
                sell_score += (1 - technical_score) * 0.4
            
            # Sentiment contribution
            if sentiment_score > 0.6:
                buy_score += sentiment_score * 0.3
            elif sentiment_score < 0.4:
                sell_score += (1 - sentiment_score) * 0.3
            
            # Swarm contribution
            if swarm_action == 'BUY' and swarm_result:
                buy_score += swarm_result.confidence * 0.3
            elif swarm_action == 'SELL' and swarm_result:
                sell_score += swarm_result.confidence * 0.3
            
            # Market regime adjustment
            if market_regime == MarketRegime.BULL_TREND:
                buy_score *= 1.1
            elif market_regime == MarketRegime.BEAR_TREND:
                sell_score *= 1.1
            elif market_regime == MarketRegime.HIGH_VOLATILITY:
                # Reduce both scores in high volatility
                buy_score *= 0.8
                sell_score *= 0.8
            
            # Determine final action
            if buy_score > sell_score and buy_score > 0.6:
                return 'BUY'
            elif sell_score > buy_score and sell_score > 0.6:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Action determination failed: {e}")
            return 'HOLD'
    
    async def _calculate_levels(
        self,
        symbol: str,
        action: str,
        chart_data: Dict[str, Any],
        market_regime: MarketRegime
    ) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            # Get current price (simplified - would use market data)
            entry_price = 100.0  # Placeholder
            
            # Get ATR for volatility-based levels
            atr = chart_data.get('atr', entry_price * 0.02)
            
            # Base multipliers
            sl_multiplier = 1.5
            tp_multiplier = 2.0
            
            # Adjust for market regime
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                sl_multiplier = 2.0
                tp_multiplier = 1.5
            elif market_regime == MarketRegime.LOW_VOLATILITY:
                sl_multiplier = 1.0
                tp_multiplier = 2.5
            
            # Calculate levels
            if action == 'BUY':
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # SELL
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            # Use support/resistance if available
            support = chart_data.get('support_level')
            resistance = chart_data.get('resistance_level')
            
            if action == 'BUY' and support:
                stop_loss = max(stop_loss, support * 0.99)
            elif action == 'SELL' and resistance:
                stop_loss = min(stop_loss, resistance * 1.01)
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Level calculation failed: {e}")
            # Return default levels
            entry_price = 100.0
            if action == 'BUY':
                return entry_price, entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price, entry_price * 1.02, entry_price * 0.96
    
    async def _calculate_position_size(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        confidence: float
    ) -> float:
        """Calculate position size"""
        try:
            risk_agent = await get_risk_agent()
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            # Adjust risk based on confidence
            confidence_multiplier = min(confidence * 1.5, 1.0)
            
            # Get position size from risk management
            position_size = await risk_agent.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_per_share=risk_per_share
            )
            
            if position_size:
                # Apply confidence adjustment
                adjusted_quantity = position_size.quantity * confidence_multiplier
                return max(0.0, adjusted_quantity)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    async def _calculate_risk_score(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        quantity: float,
        market_regime: MarketRegime
    ) -> float:
        """Calculate risk score (0-1, higher = riskier)"""
        try:
            risk_score = 0.0
            
            # Risk from stop loss distance
            sl_distance = abs(entry_price - stop_loss) / entry_price
            risk_score += min(sl_distance * 10, 0.3)  # Max 0.3 from SL distance
            
            # Risk from market regime
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                risk_score += 0.3
            elif market_regime == MarketRegime.LOW_VOLATILITY:
                risk_score += 0.1
            else:
                risk_score += 0.2
            
            # Risk from position size (relative to account)
            # This would need account balance info
            risk_score += 0.2  # Placeholder
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 0.5
    
    async def _generate_reasoning(
        self,
        action: str,
        technical_score: float,
        sentiment_score: float,
        swarm_result,
        market_regime: MarketRegime
    ) -> str:
        """Generate reasoning for the signal"""
        try:
            reasoning_parts = []
            
            # Add action and scores
            reasoning_parts.append(f"{action} signal")
            reasoning_parts.append(f"Technical: {technical_score:.2f}")
            reasoning_parts.append(f"Sentiment: {sentiment_score:.2f}")
            
            # Add swarm consensus
            if swarm_result:
                reasoning_parts.append(f"Swarm: {swarm_result.consensus.value} ({swarm_result.confidence:.2f})")
            
            # Add market regime
            reasoning_parts.append(f"Regime: {market_regime.value}")
            
            # Add swarm reasoning if available
            if swarm_result and swarm_result.reasoning:
                reasoning_parts.append(f"AI: {swarm_result.reasoning}")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return f"{action} signal from WF Swarm strategy"
    
    def _add_to_history(self, signal: StrategySignal):
        """Add signal to history"""
        try:
            self.signal_history.append(signal)
            
            # Trim history if too long
            if len(self.signal_history) > self.max_history:
                self.signal_history = self.signal_history[-self.max_history:]
                
        except Exception as e:
            logger.error(f"Failed to add signal to history: {e}")
    
    async def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        try:
            return {
                'name': self.name,
                'stats': self.stats.copy(),
                'active_signals': len(self.active_signals),
                'signal_history_count': len(self.signal_history),
                'parameters': {
                    'min_confidence': self.min_confidence,
                    'min_swarm_agreement': self.min_swarm_agreement,
                    'max_positions': self.max_positions
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get strategy stats: {e}")
            return {}


# Global strategy instance
_wf_swarm_strategy: Optional[WFSwarmStrategy] = None


async def get_wf_swarm_strategy() -> WFSwarmStrategy:
    """Get the global WF Swarm strategy instance"""
    global _wf_swarm_strategy
    
    if _wf_swarm_strategy is None:
        _wf_swarm_strategy = WFSwarmStrategy()
    
    return _wf_swarm_strategy