"""
Strategy Agent - Combines chart analysis, sentiment, and swarm consensus
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..config import get_config
from ..bus import get_message_bus, publish_trade_signal
from .chart_agent import get_chart_agent
from .sentiment_agent import get_sentiment_agent
from .swarm_agent import get_swarm_agent
from .risk_agent import get_risk_agent
from .execution_agent import get_execution_agent

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    VERY_WEAK = "VERY_WEAK"


@dataclass
class TradingSignal:
    """Complete trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    reasoning: str
    risk_level: str
    components: Dict[str, Any]  # Individual analysis components
    timestamp: datetime


class StrategyAgent:
    """Strategy agent for combining all analysis"""
    
    def __init__(self):
        self.config = get_config()
        self.signal_cache: Dict[str, TradingSignal] = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Signal generation parameters
        self.min_confidence = 0.65
        self.min_swarm_agreement = 0.6
        self.max_risk_level = "MEDIUM"
        
        # Component weights
        self.weights = {
            'chart_analysis': 0.4,
            'sentiment': 0.3,
            'swarm_consensus': 0.3
        }
    
    async def generate_signal(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> Optional[TradingSignal]:
        """Generate comprehensive trading signal"""
        try:
            # Check cache first
            if not force_refresh and symbol in self.signal_cache:
                cached_signal = self.signal_cache[symbol]
                if datetime.now() - cached_signal.timestamp < self.cache_duration:
                    return cached_signal
            
            logger.info(f"Generating trading signal for {symbol}")
            
            # Get all analysis components
            components = await self._gather_analysis_components(symbol)
            if not components:
                logger.warning(f"Failed to gather analysis components for {symbol}")
                return None
            
            # Validate components
            if not await self._validate_components(components):
                logger.warning(f"Component validation failed for {symbol}")
                return None
            
            # Generate signal
            signal = await self._combine_analysis(symbol, components)
            if not signal:
                logger.warning(f"Failed to generate signal for {symbol}")
                return None
            
            # Validate signal with risk management
            if not await self._validate_signal_risk(signal):
                logger.warning(f"Signal failed risk validation for {symbol}")
                return None
            
            # Cache signal
            self.signal_cache[symbol] = signal
            
            # Publish signal
            await self._publish_signal(signal)
            
            logger.info(f"Generated signal for {symbol}: {signal.action} - {signal.strength.value} (confidence: {signal.confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate signal for {symbol}: {e}")
            return None
    
    async def _gather_analysis_components(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gather all analysis components"""
        try:
            # Get agents
            chart_agent = await get_chart_agent()
            sentiment_agent = await get_sentiment_agent()
            swarm_agent = await get_swarm_agent()
            
            # Gather analysis in parallel
            tasks = [
                chart_agent.analyze_symbol(symbol),
                sentiment_agent.analyze_sentiment(symbol),
                self._get_market_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            chart_analysis = results[0] if not isinstance(results[0], Exception) else None
            sentiment_data = results[1] if not isinstance(results[1], Exception) else None
            market_data = results[2] if not isinstance(results[2], Exception) else None
            
            if not chart_analysis or not sentiment_data or not market_data:
                logger.error(f"Missing analysis components for {symbol}")
                return None
            
            # Get swarm consensus
            swarm_consensus = await swarm_agent.get_consensus(
                symbol, chart_analysis, sentiment_data, market_data
            )
            
            return {
                'chart_analysis': chart_analysis,
                'sentiment_data': sentiment_data,
                'market_data': market_data,
                'swarm_consensus': swarm_consensus
            }
            
        except Exception as e:
            logger.error(f"Failed to gather analysis components: {e}")
            return None
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data"""
        try:
            # This would typically come from your market data connectors
            # For now, return a placeholder
            return {
                'symbol': symbol,
                'price': 0.0,
                'volume': 0.0,
                'change_24h': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _validate_components(self, components: Dict[str, Any]) -> bool:
        """Validate analysis components"""
        try:
            required_components = ['chart_analysis', 'sentiment_data', 'market_data']
            
            for component in required_components:
                if component not in components or not components[component]:
                    logger.error(f"Missing or invalid component: {component}")
                    return False
            
            # Validate chart analysis
            chart_analysis = components['chart_analysis']
            if 'signal' not in chart_analysis or 'confidence' not in chart_analysis:
                logger.error("Invalid chart analysis format")
                return False
            
            # Validate sentiment data
            sentiment_data = components['sentiment_data']
            if 'sentiment_score' not in sentiment_data or 'tradability_score' not in sentiment_data:
                logger.error("Invalid sentiment data format")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Component validation error: {e}")
            return False
    
    async def _combine_analysis(self, symbol: str, components: Dict[str, Any]) -> Optional[TradingSignal]:
        """Combine all analysis into final signal"""
        try:
            chart_analysis = components['chart_analysis']
            sentiment_data = components['sentiment_data']
            market_data = components['market_data']
            swarm_consensus = components.get('swarm_consensus')
            
            # Extract signals from components
            chart_signal = chart_analysis.get('signal', 'HOLD')
            chart_confidence = chart_analysis.get('confidence', 0.0)
            
            sentiment_score = sentiment_data.get('sentiment_score', 5.0) / 10.0  # Normalize to 0-1
            tradability_score = sentiment_data.get('tradability_score', 5.0) / 10.0
            
            swarm_signal = 'HOLD'
            swarm_confidence = 0.5
            if swarm_consensus:
                swarm_signal = self._consensus_to_signal(swarm_consensus.consensus.value)
                swarm_confidence = swarm_consensus.confidence
            
            # Calculate weighted scores
            buy_score = 0.0
            sell_score = 0.0
            
            # Chart analysis contribution
            if chart_signal == 'BUY':
                buy_score += self.weights['chart_analysis'] * chart_confidence
            elif chart_signal == 'SELL':
                sell_score += self.weights['chart_analysis'] * chart_confidence
            
            # Sentiment contribution
            if sentiment_score > 0.6:
                buy_score += self.weights['sentiment'] * sentiment_score
            elif sentiment_score < 0.4:
                sell_score += self.weights['sentiment'] * (1 - sentiment_score)
            
            # Swarm consensus contribution
            if swarm_signal == 'BUY':
                buy_score += self.weights['swarm_consensus'] * swarm_confidence
            elif swarm_signal == 'SELL':
                sell_score += self.weights['swarm_consensus'] * swarm_confidence
            
            # Determine final signal
            if buy_score > sell_score and buy_score > self.min_confidence:
                action = 'BUY'
                confidence = buy_score
            elif sell_score > buy_score and sell_score > self.min_confidence:
                action = 'SELL'
                confidence = sell_score
            else:
                action = 'HOLD'
                confidence = max(buy_score, sell_score)
            
            # Skip HOLD signals
            if action == 'HOLD':
                return None
            
            # Determine signal strength
            strength = self._calculate_signal_strength(confidence, components)
            
            # Calculate entry, stop loss, and take profit
            entry_price = market_data.get('price', 0.0)
            stop_loss, take_profit = await self._calculate_levels(
                symbol, action, entry_price, chart_analysis
            )
            
            # Calculate position size
            quantity = await self._calculate_position_size(
                symbol, action, entry_price, stop_loss
            )
            
            if quantity <= 0:
                logger.warning(f"Invalid position size calculated for {symbol}")
                return None
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(components, action, confidence)
            
            # Determine risk level
            risk_level = await self._assess_risk_level(components, action)
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                reasoning=reasoning,
                risk_level=risk_level,
                components=components,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to combine analysis: {e}")
            return None
    
    def _consensus_to_signal(self, consensus: str) -> str:
        """Convert swarm consensus to signal"""
        if 'BUY' in consensus:
            return 'BUY'
        elif 'SELL' in consensus:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_signal_strength(self, confidence: float, components: Dict[str, Any]) -> SignalStrength:
        """Calculate signal strength based on confidence and component agreement"""
        try:
            # Base strength from confidence
            if confidence >= 0.9:
                base_strength = SignalStrength.VERY_STRONG
            elif confidence >= 0.8:
                base_strength = SignalStrength.STRONG
            elif confidence >= 0.7:
                base_strength = SignalStrength.MODERATE
            elif confidence >= 0.6:
                base_strength = SignalStrength.WEAK
            else:
                base_strength = SignalStrength.VERY_WEAK
            
            # Adjust based on component agreement
            chart_analysis = components.get('chart_analysis', {})
            sentiment_data = components.get('sentiment_data', {})
            swarm_consensus = components.get('swarm_consensus')
            
            agreement_count = 0
            total_components = 0
            
            # Check chart analysis agreement
            if chart_analysis.get('confidence', 0) > 0.7:
                agreement_count += 1
            total_components += 1
            
            # Check sentiment agreement
            sentiment_score = sentiment_data.get('sentiment_score', 5.0)
            if sentiment_score > 7 or sentiment_score < 3:  # Strong sentiment
                agreement_count += 1
            total_components += 1
            
            # Check swarm agreement
            if swarm_consensus and swarm_consensus.confidence > 0.7:
                agreement_count += 1
            total_components += 1
            
            # Adjust strength based on agreement
            agreement_ratio = agreement_count / total_components if total_components > 0 else 0
            
            if agreement_ratio >= 0.8 and base_strength.value in ['MODERATE', 'WEAK']:
                # Upgrade strength for high agreement
                if base_strength == SignalStrength.MODERATE:
                    return SignalStrength.STRONG
                elif base_strength == SignalStrength.WEAK:
                    return SignalStrength.MODERATE
            elif agreement_ratio < 0.5 and base_strength.value in ['STRONG', 'VERY_STRONG']:
                # Downgrade strength for low agreement
                if base_strength == SignalStrength.VERY_STRONG:
                    return SignalStrength.STRONG
                elif base_strength == SignalStrength.STRONG:
                    return SignalStrength.MODERATE
            
            return base_strength
            
        except Exception as e:
            logger.error(f"Failed to calculate signal strength: {e}")
            return SignalStrength.WEAK
    
    async def _calculate_levels(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        chart_analysis: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            atr = chart_analysis.get('atr', entry_price * 0.02)  # Default 2% ATR
            
            if action == 'BUY':
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 2.0)
            else:  # SELL
                stop_loss = entry_price + (atr * 1.5)
                take_profit = entry_price - (atr * 2.0)
            
            # Use support/resistance levels if available
            support_level = chart_analysis.get('support_level')
            resistance_level = chart_analysis.get('resistance_level')
            
            if action == 'BUY' and support_level and support_level < entry_price:
                stop_loss = max(stop_loss, support_level * 0.99)  # Slightly below support
            elif action == 'SELL' and resistance_level and resistance_level > entry_price:
                stop_loss = min(stop_loss, resistance_level * 1.01)  # Slightly above resistance
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Failed to calculate levels: {e}")
            # Return default levels
            if action == 'BUY':
                return entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price * 1.02, entry_price * 0.96
    
    async def _calculate_position_size(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """Calculate position size using risk management"""
        try:
            risk_agent = await get_risk_agent()
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            # Get position size from risk management
            position_size = await risk_agent.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_per_share=risk_per_share
            )
            
            return position_size.quantity if position_size else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.0
    
    async def _generate_reasoning(
        self,
        components: Dict[str, Any],
        action: str,
        confidence: float
    ) -> str:
        """Generate reasoning for the signal"""
        try:
            reasoning_parts = []
            
            # Add action and confidence
            reasoning_parts.append(f"{action} signal with {confidence:.1%} confidence")
            
            # Add chart analysis reasoning
            chart_analysis = components.get('chart_analysis', {})
            if chart_analysis.get('signal') == action:
                chart_reasoning = chart_analysis.get('reasoning', '')
                if chart_reasoning:
                    reasoning_parts.append(f"Technical: {chart_reasoning}")
            
            # Add sentiment reasoning
            sentiment_data = components.get('sentiment_data', {})
            sentiment_score = sentiment_data.get('sentiment_score', 5.0)
            if (action == 'BUY' and sentiment_score > 6) or (action == 'SELL' and sentiment_score < 4):
                key_drivers = sentiment_data.get('key_drivers', [])
                if key_drivers:
                    reasoning_parts.append(f"Sentiment: {', '.join(key_drivers[:2])}")
            
            # Add swarm consensus reasoning
            swarm_consensus = components.get('swarm_consensus')
            if swarm_consensus and self._consensus_to_signal(swarm_consensus.consensus.value) == action:
                reasoning_parts.append(f"Swarm: {swarm_consensus.reasoning}")
            
            return '. '.join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return f"{action} signal generated"
    
    async def _assess_risk_level(self, components: Dict[str, Any], action: str) -> str:
        """Assess overall risk level"""
        try:
            risk_factors = []
            
            # Chart analysis risk
            chart_analysis = components.get('chart_analysis', {})
            volatility = chart_analysis.get('volatility', 'MEDIUM')
            if volatility == 'HIGH':
                risk_factors.append('high_volatility')
            
            # Sentiment risk
            sentiment_data = components.get('sentiment_data', {})
            tradability_score = sentiment_data.get('tradability_score', 5.0)
            if tradability_score < 4:
                risk_factors.append('low_tradability')
            
            # Swarm consensus risk
            swarm_consensus = components.get('swarm_consensus')
            if swarm_consensus and swarm_consensus.risk_factors:
                risk_factors.extend(swarm_consensus.risk_factors)
            
            # Determine overall risk level
            if len(risk_factors) >= 3:
                return 'HIGH'
            elif len(risk_factors) >= 1:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.error(f"Failed to assess risk level: {e}")
            return 'MEDIUM'
    
    async def _validate_signal_risk(self, signal: TradingSignal) -> bool:
        """Validate signal against risk management rules"""
        try:
            # Check minimum confidence
            if signal.confidence < self.min_confidence:
                logger.warning(f"Signal confidence too low: {signal.confidence}")
                return False
            
            # Check maximum risk level
            risk_levels = ['LOW', 'MEDIUM', 'HIGH']
            if risk_levels.index(signal.risk_level) > risk_levels.index(self.max_risk_level):
                logger.warning(f"Signal risk level too high: {signal.risk_level}")
                return False
            
            # Validate with risk agent
            risk_agent = await get_risk_agent()
            
            trade_data = {
                'symbol': signal.symbol,
                'quantity': signal.quantity,
                'entry_price': signal.entry_price,
                'side': signal.action,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            is_valid, validation_message = await risk_agent.validate_trade(trade_data)
            if not is_valid:
                logger.warning(f"Risk validation failed: {validation_message}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal risk validation error: {e}")
            return False
    
    async def _publish_signal(self, signal: TradingSignal) -> None:
        """Publish trading signal to message bus"""
        try:
            signal_data = {
                'symbol': signal.symbol,
                'action': signal.action,
                'strength': signal.strength.value,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'quantity': signal.quantity,
                'reasoning': signal.reasoning,
                'risk_level': signal.risk_level,
                'timestamp': signal.timestamp.isoformat()
            }
            
            await publish_trade_signal(
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                price=signal.entry_price,
                metadata=signal_data
            )
            
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
    
    async def execute_signal(self, signal: TradingSignal) -> Optional[str]:
        """Execute a trading signal"""
        try:
            execution_agent = await get_execution_agent()
            
            order_id = await execution_agent.execute_trade(
                symbol=signal.symbol,
                side=signal.action,
                quantity=signal.quantity,
                order_type="MARKET",
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata={
                    'signal_strength': signal.strength.value,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'risk_level': signal.risk_level
                }
            )
            
            if order_id:
                logger.info(f"Signal executed: {signal.symbol} {signal.action} - Order ID: {order_id}")
            else:
                logger.error(f"Failed to execute signal: {signal.symbol} {signal.action}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return None
    
    async def get_strategy_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get strategy summary for multiple symbols"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(symbols),
                'signals_generated': 0,
                'signal_distribution': {
                    'BUY': 0,
                    'SELL': 0,
                    'HOLD': 0
                },
                'strength_distribution': {
                    'VERY_STRONG': 0,
                    'STRONG': 0,
                    'MODERATE': 0,
                    'WEAK': 0,
                    'VERY_WEAK': 0
                },
                'average_confidence': 0.0,
                'active_signals': []
            }
            
            total_confidence = 0
            signal_count = 0
            
            for symbol in symbols:
                if symbol in self.signal_cache:
                    signal = self.signal_cache[symbol]
                    
                    # Update counts
                    summary['signals_generated'] += 1
                    summary['signal_distribution'][signal.action] += 1
                    summary['strength_distribution'][signal.strength.value] += 1
                    
                    # Update confidence
                    total_confidence += signal.confidence
                    signal_count += 1
                    
                    # Add to active signals
                    summary['active_signals'].append({
                        'symbol': symbol,
                        'action': signal.action,
                        'strength': signal.strength.value,
                        'confidence': signal.confidence,
                        'risk_level': signal.risk_level
                    })
            
            # Calculate average confidence
            if signal_count > 0:
                summary['average_confidence'] = total_confidence / signal_count
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get strategy summary: {e}")
            return {}


# Global strategy agent instance
_strategy_agent: Optional[StrategyAgent] = None


async def get_strategy_agent() -> StrategyAgent:
    """Get the global strategy agent"""
    global _strategy_agent
    
    if _strategy_agent is None:
        _strategy_agent = StrategyAgent()
    
    return _strategy_agent