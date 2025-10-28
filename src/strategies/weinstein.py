"""
Weinstein Stage Analysis Strategy
Based on Stan Weinstein's "Secrets for Profiting in Bull and Bear Markets"
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import get_config
from ..bus import get_message_bus, publish_trade_signal
from ..agents.chart_agent import get_chart_agent
from ..agents.risk_agent import get_risk_agent

logger = logging.getLogger(__name__)


class WeinsteinStage(Enum):
    """Weinstein market stages"""
    STAGE_1 = "STAGE_1"  # Accumulation/Base building
    STAGE_2 = "STAGE_2"  # Advancing/Uptrend
    STAGE_3 = "STAGE_3"  # Distribution/Top
    STAGE_4 = "STAGE_4"  # Declining/Downtrend


class TrendStrength(Enum):
    """Trend strength classification"""
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    VERY_WEAK = "VERY_WEAK"


@dataclass
class WeinsteinSignal:
    """Weinstein strategy signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    stage: WeinsteinStage
    trend_strength: TrendStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    volume_confirmation: bool
    ma_slope: float
    price_vs_ma: float
    relative_strength: float
    reasoning: str
    confidence: float
    timestamp: datetime


class WeinsteinStrategy:
    """
    Weinstein Stage Analysis Strategy
    
    Key principles:
    1. Only buy in Stage 2 (uptrend)
    2. Only sell short in Stage 4 (downtrend)
    3. Avoid Stage 1 (sideways) and Stage 3 (topping)
    4. Use 30-week (150-day) moving average as primary trend indicator
    5. Require volume confirmation for breakouts
    6. Focus on relative strength vs market
    """
    
    def __init__(self):
        self.config = get_config()
        self.name = "WEINSTEIN"
        
        # Strategy parameters
        self.primary_ma_period = 150  # 30-week equivalent
        self.secondary_ma_period = 50  # 10-week equivalent
        self.volume_ma_period = 50
        self.min_volume_ratio = 1.5  # Volume must be 1.5x average
        self.min_confidence = 0.65
        
        # Stage detection parameters
        self.stage_lookback = 20  # Days to confirm stage
        self.breakout_threshold = 0.02  # 2% breakout threshold
        self.ma_slope_threshold = 0.001  # Minimum MA slope for trend
        
        # Position management
        self.max_positions = 8
        self.position_timeout = timedelta(days=30)
        
        # Signal tracking
        self.active_signals = {}
        self.signal_history = []
        self.stage_cache = {}
        self.cache_duration = timedelta(hours=4)
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'stage_2_buys': 0,
            'stage_4_sells': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0
        }
    
    async def analyze_symbol(self, symbol: str) -> Optional[WeinsteinSignal]:
        """Analyze symbol using Weinstein methodology"""
        try:
            logger.info(f"Weinstein analyzing {symbol}")
            
            # Get chart data
            chart_agent = await get_chart_agent()
            chart_data = await chart_agent.analyze_symbol(symbol)
            
            if not chart_data:
                logger.warning(f"No chart data for {symbol}")
                return None
            
            # Get OHLCV data for detailed analysis
            ohlcv_data = await self._get_ohlcv_data(symbol)
            if not ohlcv_data:
                logger.warning(f"No OHLCV data for {symbol}")
                return None
            
            # Detect Weinstein stage
            stage = await self._detect_stage(symbol, ohlcv_data)
            
            # Calculate trend strength
            trend_strength = await self._calculate_trend_strength(ohlcv_data)
            
            # Check volume confirmation
            volume_confirmation = await self._check_volume_confirmation(ohlcv_data)
            
            # Calculate technical indicators
            ma_slope = await self._calculate_ma_slope(ohlcv_data)
            price_vs_ma = await self._calculate_price_vs_ma(ohlcv_data)
            relative_strength = await self._calculate_relative_strength(symbol, ohlcv_data)
            
            # Determine action based on stage
            action = await self._determine_action(
                stage, trend_strength, volume_confirmation, ma_slope, price_vs_ma
            )
            
            if action == 'HOLD':
                logger.info(f"No Weinstein signal for {symbol}: stage={stage.value}")
                return None
            
            # Calculate confidence
            confidence = await self._calculate_confidence(
                stage, trend_strength, volume_confirmation, ma_slope, relative_strength
            )
            
            if confidence < self.min_confidence:
                logger.info(f"Weinstein confidence too low for {symbol}: {confidence:.2f}")
                return None
            
            # Calculate entry, stop loss, and take profit
            entry_price, stop_loss, take_profit = await self._calculate_levels(
                symbol, action, ohlcv_data, stage
            )
            
            # Calculate position size
            quantity = await self._calculate_position_size(
                symbol, action, entry_price, stop_loss, confidence
            )
            
            if quantity <= 0:
                logger.warning(f"Invalid position size for {symbol}")
                return None
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(
                action, stage, trend_strength, volume_confirmation, ma_slope, relative_strength
            )
            
            # Create signal
            signal = WeinsteinSignal(
                symbol=symbol,
                action=action,
                stage=stage,
                trend_strength=trend_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                volume_confirmation=volume_confirmation,
                ma_slope=ma_slope,
                price_vs_ma=price_vs_ma,
                relative_strength=relative_strength,
                reasoning=reasoning,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Store signal
            self.active_signals[symbol] = signal
            self._add_to_history(signal)
            self.stats['signals_generated'] += 1
            
            if action == 'BUY':
                self.stats['stage_2_buys'] += 1
            elif action == 'SELL':
                self.stats['stage_4_sells'] += 1
            
            logger.info(f"Weinstein signal: {symbol} {action} - Stage {stage.value} (confidence: {confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Weinstein analysis failed for {symbol}: {e}")
            return None
    
    async def _get_ohlcv_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get OHLCV data for analysis"""
        try:
            # This would typically fetch from market data connectors
            # For now, return placeholder data structure
            return {
                'symbol': symbol,
                'timeframe': '1d',
                'data': [],  # Would contain OHLCV arrays
                'current_price': 100.0,
                'volume': 1000000,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return None
    
    async def _detect_stage(self, symbol: str, ohlcv_data: Dict[str, Any]) -> WeinsteinStage:
        """Detect Weinstein stage"""
        try:
            # Check cache
            if symbol in self.stage_cache:
                cached_time, cached_stage = self.stage_cache[symbol]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_stage
            
            # Simplified stage detection logic
            # In real implementation, this would analyze:
            # 1. Price vs 30-week MA
            # 2. MA slope
            # 3. Volume patterns
            # 4. Price action over time
            
            current_price = ohlcv_data.get('current_price', 100.0)
            
            # Placeholder logic - would be much more sophisticated
            # This is a simplified version for demonstration
            
            # Calculate moving averages (simplified)
            ma_150 = current_price * 0.95  # Placeholder
            ma_50 = current_price * 0.98   # Placeholder
            
            # Determine stage based on price vs MA and trend
            if current_price > ma_150 and ma_150 > ma_50:
                # Price above rising MA = Stage 2
                stage = WeinsteinStage.STAGE_2
            elif current_price < ma_150 and ma_150 < ma_50:
                # Price below falling MA = Stage 4
                stage = WeinsteinStage.STAGE_4
            elif current_price > ma_150 and ma_150 < ma_50:
                # Price above but MA falling = Stage 3
                stage = WeinsteinStage.STAGE_3
            else:
                # Sideways/base building = Stage 1
                stage = WeinsteinStage.STAGE_1
            
            # Cache result
            self.stage_cache[symbol] = (datetime.now(), stage)
            
            return stage
            
        except Exception as e:
            logger.error(f"Stage detection failed for {symbol}: {e}")
            return WeinsteinStage.STAGE_1
    
    async def _calculate_trend_strength(self, ohlcv_data: Dict[str, Any]) -> TrendStrength:
        """Calculate trend strength"""
        try:
            # Simplified trend strength calculation
            # In real implementation, this would analyze:
            # 1. MA slope steepness
            # 2. Price momentum
            # 3. Volume trends
            # 4. Consistency of direction
            
            # Placeholder logic
            current_price = ohlcv_data.get('current_price', 100.0)
            
            # Calculate momentum (simplified)
            momentum = 0.05  # Placeholder 5% momentum
            
            if momentum > 0.08:
                return TrendStrength.VERY_STRONG
            elif momentum > 0.05:
                return TrendStrength.STRONG
            elif momentum > 0.02:
                return TrendStrength.MODERATE
            elif momentum > 0.01:
                return TrendStrength.WEAK
            else:
                return TrendStrength.VERY_WEAK
                
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return TrendStrength.MODERATE
    
    async def _check_volume_confirmation(self, ohlcv_data: Dict[str, Any]) -> bool:
        """Check for volume confirmation"""
        try:
            current_volume = ohlcv_data.get('volume', 1000000)
            
            # Calculate average volume (simplified)
            avg_volume = current_volume * 0.8  # Placeholder
            
            # Check if current volume is above threshold
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return volume_ratio >= self.min_volume_ratio
            
        except Exception as e:
            logger.error(f"Volume confirmation check failed: {e}")
            return False
    
    async def _calculate_ma_slope(self, ohlcv_data: Dict[str, Any]) -> float:
        """Calculate moving average slope"""
        try:
            # Simplified MA slope calculation
            # In real implementation, this would calculate the actual slope
            # of the 150-day MA over the last 10-20 days
            
            # Placeholder: positive slope for uptrend
            return 0.002  # 0.2% daily slope
            
        except Exception as e:
            logger.error(f"MA slope calculation failed: {e}")
            return 0.0
    
    async def _calculate_price_vs_ma(self, ohlcv_data: Dict[str, Any]) -> float:
        """Calculate price position relative to MA"""
        try:
            current_price = ohlcv_data.get('current_price', 100.0)
            
            # Calculate 150-day MA (simplified)
            ma_150 = current_price * 0.95  # Placeholder
            
            # Return percentage above/below MA
            return (current_price - ma_150) / ma_150 if ma_150 > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Price vs MA calculation failed: {e}")
            return 0.0
    
    async def _calculate_relative_strength(self, symbol: str, ohlcv_data: Dict[str, Any]) -> float:
        """Calculate relative strength vs market"""
        try:
            # Simplified relative strength calculation
            # In real implementation, this would compare the symbol's performance
            # to a market index (S&P 500, etc.) over various timeframes
            
            # Placeholder: assume neutral relative strength
            return 0.0  # 0 = neutral, positive = outperforming, negative = underperforming
            
        except Exception as e:
            logger.error(f"Relative strength calculation failed: {e}")
            return 0.0
    
    async def _determine_action(
        self,
        stage: WeinsteinStage,
        trend_strength: TrendStrength,
        volume_confirmation: bool,
        ma_slope: float,
        price_vs_ma: float
    ) -> str:
        """Determine trading action based on Weinstein rules"""
        try:
            # Weinstein rules:
            # 1. Only buy in Stage 2 (uptrend)
            # 2. Only sell short in Stage 4 (downtrend)
            # 3. Avoid Stage 1 and Stage 3
            
            if stage == WeinsteinStage.STAGE_2:
                # Stage 2: Consider buying
                if (trend_strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG] and
                    volume_confirmation and
                    ma_slope > self.ma_slope_threshold and
                    price_vs_ma > 0):
                    return 'BUY'
            
            elif stage == WeinsteinStage.STAGE_4:
                # Stage 4: Consider selling/shorting
                if (trend_strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG] and
                    volume_confirmation and
                    ma_slope < -self.ma_slope_threshold and
                    price_vs_ma < 0):
                    return 'SELL'
            
            # Stage 1 and Stage 3: Hold/avoid
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Action determination failed: {e}")
            return 'HOLD'
    
    async def _calculate_confidence(
        self,
        stage: WeinsteinStage,
        trend_strength: TrendStrength,
        volume_confirmation: bool,
        ma_slope: float,
        relative_strength: float
    ) -> float:
        """Calculate signal confidence"""
        try:
            confidence = 0.0
            
            # Base confidence from stage
            if stage == WeinsteinStage.STAGE_2:
                confidence += 0.3
            elif stage == WeinsteinStage.STAGE_4:
                confidence += 0.3
            else:
                confidence += 0.1  # Low confidence for Stage 1 and 3
            
            # Trend strength contribution
            strength_scores = {
                TrendStrength.VERY_STRONG: 0.25,
                TrendStrength.STRONG: 0.2,
                TrendStrength.MODERATE: 0.15,
                TrendStrength.WEAK: 0.1,
                TrendStrength.VERY_WEAK: 0.05
            }
            confidence += strength_scores.get(trend_strength, 0.1)
            
            # Volume confirmation
            if volume_confirmation:
                confidence += 0.2
            else:
                confidence += 0.05
            
            # MA slope strength
            slope_strength = min(abs(ma_slope) * 100, 0.15)  # Max 0.15 contribution
            confidence += slope_strength
            
            # Relative strength
            if relative_strength > 0.02:  # Outperforming market
                confidence += 0.1
            elif relative_strength < -0.02:  # Underperforming market
                confidence += 0.05
            else:
                confidence += 0.075  # Neutral
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _calculate_levels(
        self,
        symbol: str,
        action: str,
        ohlcv_data: Dict[str, Any],
        stage: WeinsteinStage
    ) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            current_price = ohlcv_data.get('current_price', 100.0)
            entry_price = current_price
            
            # Weinstein stop loss rules:
            # - For Stage 2 buys: Stop below recent support or 150-day MA
            # - For Stage 4 sells: Stop above recent resistance or 150-day MA
            
            # Calculate ATR for volatility-based stops (simplified)
            atr = current_price * 0.02  # 2% ATR placeholder
            
            if action == 'BUY':
                # Stage 2 buy
                # Stop loss: Below 150-day MA or recent support
                ma_150 = current_price * 0.95  # Placeholder
                support_level = current_price * 0.97  # Placeholder
                
                stop_loss = min(ma_150 * 0.98, support_level)  # 2% buffer below MA/support
                
                # Take profit: 2-3x risk
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 2.5)
                
            else:  # SELL
                # Stage 4 sell
                # Stop loss: Above 150-day MA or recent resistance
                ma_150 = current_price * 1.05  # Placeholder
                resistance_level = current_price * 1.03  # Placeholder
                
                stop_loss = max(ma_150 * 1.02, resistance_level)  # 2% buffer above MA/resistance
                
                # Take profit: 2-3x risk
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2.5)
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Level calculation failed: {e}")
            # Return default levels
            entry_price = 100.0
            if action == 'BUY':
                return entry_price, entry_price * 0.95, entry_price * 1.10
            else:
                return entry_price, entry_price * 1.05, entry_price * 0.90
    
    async def _calculate_position_size(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        confidence: float
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
            
            if position_size:
                return position_size.quantity
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    async def _generate_reasoning(
        self,
        action: str,
        stage: WeinsteinStage,
        trend_strength: TrendStrength,
        volume_confirmation: bool,
        ma_slope: float,
        relative_strength: float
    ) -> str:
        """Generate reasoning for the signal"""
        try:
            reasoning_parts = []
            
            # Add action and stage
            reasoning_parts.append(f"{action} signal in {stage.value}")
            
            # Add trend strength
            reasoning_parts.append(f"Trend: {trend_strength.value}")
            
            # Add volume confirmation
            if volume_confirmation:
                reasoning_parts.append("Volume confirmed")
            else:
                reasoning_parts.append("Volume weak")
            
            # Add MA slope
            if ma_slope > 0.001:
                reasoning_parts.append("MA rising")
            elif ma_slope < -0.001:
                reasoning_parts.append("MA falling")
            else:
                reasoning_parts.append("MA flat")
            
            # Add relative strength
            if relative_strength > 0.02:
                reasoning_parts.append("Outperforming market")
            elif relative_strength < -0.02:
                reasoning_parts.append("Underperforming market")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return f"{action} signal from Weinstein strategy"
    
    def _add_to_history(self, signal: WeinsteinSignal):
        """Add signal to history"""
        try:
            self.signal_history.append(signal)
            
            # Trim history if too long
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
                
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
                    'primary_ma_period': self.primary_ma_period,
                    'min_confidence': self.min_confidence,
                    'min_volume_ratio': self.min_volume_ratio,
                    'breakout_threshold': self.breakout_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get strategy stats: {e}")
            return {}
    
    async def get_stage_distribution(self, symbols: List[str]) -> Dict[str, int]:
        """Get distribution of symbols across Weinstein stages"""
        try:
            distribution = {
                'STAGE_1': 0,
                'STAGE_2': 0,
                'STAGE_3': 0,
                'STAGE_4': 0
            }
            
            for symbol in symbols:
                try:
                    ohlcv_data = await self._get_ohlcv_data(symbol)
                    if ohlcv_data:
                        stage = await self._detect_stage(symbol, ohlcv_data)
                        distribution[stage.value] += 1
                except Exception as e:
                    logger.error(f"Failed to get stage for {symbol}: {e}")
            
            return distribution
            
        except Exception as e:
            logger.error(f"Failed to get stage distribution: {e}")
            return {}


# Global strategy instance
_weinstein_strategy: Optional[WeinsteinStrategy] = None


async def get_weinstein_strategy() -> WeinsteinStrategy:
    """Get the global Weinstein strategy instance"""
    global _weinstein_strategy
    
    if _weinstein_strategy is None:
        _weinstein_strategy = WeinsteinStrategy()
    
    return _weinstein_strategy