"""
Risk Management Agent with position sizing and drawdown protection
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum

from ..config import get_config
from ..bus import publish_analysis_result

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for position sizing"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class PositionSize:
    """Position size calculation result"""
    symbol: str
    quantity: float
    risk_amount: float
    risk_percentage: float
    stop_loss: float
    take_profit: float
    max_loss: float
    position_value: float
    leverage: float = 1.0


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    total_equity: float
    available_equity: float
    used_margin: float
    daily_pnl: float
    daily_drawdown: float
    max_drawdown: float
    risk_per_trade: float
    total_exposure: float
    open_positions: int
    risk_level: RiskLevel


class RiskManagementAgent:
    """Risk management agent with comprehensive risk controls"""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        
        # Risk tracking
        self.daily_trades: List[Dict[str, Any]] = []
        self.daily_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        
        # Risk limits
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.last_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Initialize with default equity if not set
        self.current_equity = self.config.trading.initial_equity
        self.peak_equity = self.current_equity
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_loss: float, 
        direction: str,
        risk_multiplier: float = 1.0
    ) -> Optional[PositionSize]:
        """Calculate position size based on risk management rules"""
        try:
            # Check if trading is halted
            if self.trading_halted:
                logger.warning(f"Trading halted: {self.halt_reason}")
                return None
            
            # Reset daily tracking if new day
            await self._check_daily_reset()
            
            # Get current risk metrics
            risk_metrics = await self.get_risk_metrics()
            
            # Check daily drawdown limit
            if risk_metrics.daily_drawdown >= self.config.trading.daily_max_drawdown_pct:
                await self._halt_trading("Daily drawdown limit exceeded")
                return None
            
            # Check maximum number of positions
            if len(self.open_positions) >= self.config.trading.max_positions:
                logger.warning(f"Maximum positions limit reached: {len(self.open_positions)}")
                return None
            
            # Calculate risk amount
            base_risk_pct = self.config.trading.risk_per_trade_pct / 100.0
            adjusted_risk_pct = base_risk_pct * risk_multiplier
            
            # Apply dynamic risk adjustment based on current drawdown
            if risk_metrics.daily_drawdown > 0.02:  # 2% drawdown
                adjusted_risk_pct *= 0.5  # Reduce risk by 50%
            elif risk_metrics.daily_drawdown > 0.01:  # 1% drawdown
                adjusted_risk_pct *= 0.75  # Reduce risk by 25%
            
            risk_amount = risk_metrics.available_equity * adjusted_risk_pct
            
            # Calculate position size
            price_diff = abs(entry_price - stop_loss)
            if price_diff == 0:
                logger.error(f"Invalid stop loss for {symbol}: entry={entry_price}, sl={stop_loss}")
                return None
            
            # Base quantity calculation
            base_quantity = risk_amount / price_diff
            
            # Apply leverage if configured
            leverage = self.config.trading.max_leverage
            if leverage > 1:
                base_quantity *= leverage
            
            # Round to appropriate precision
            quantity = self._round_quantity(symbol, base_quantity)
            
            # Calculate position value
            position_value = quantity * entry_price
            
            # Check if position value exceeds available equity
            max_position_value = risk_metrics.available_equity * 0.2  # Max 20% per position
            if position_value > max_position_value:
                quantity = max_position_value / entry_price
                quantity = self._round_quantity(symbol, quantity)
                position_value = quantity * entry_price
            
            # Calculate take profit (if not provided)
            take_profit = await self._calculate_take_profit(entry_price, stop_loss, direction)
            
            # Calculate maximum loss
            max_loss = quantity * price_diff
            
            position_size = PositionSize(
                symbol=symbol,
                quantity=quantity,
                risk_amount=risk_amount,
                risk_percentage=adjusted_risk_pct * 100,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_loss=max_loss,
                position_value=position_value,
                leverage=leverage
            )
            
            logger.info(f"Position size calculated for {symbol}: {quantity} units, risk: ${risk_amount:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            return None
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to appropriate precision based on symbol"""
        try:
            # Default rounding rules
            if 'BTC' in symbol.upper():
                return round(quantity, 6)
            elif 'ETH' in symbol.upper():
                return round(quantity, 4)
            elif any(crypto in symbol.upper() for crypto in ['USDT', 'USDC', 'USD']):
                return round(quantity, 2)
            else:
                # For stocks, round to whole shares or appropriate decimal
                if quantity >= 1:
                    return round(quantity, 0)
                else:
                    return round(quantity, 4)
        except Exception:
            return round(quantity, 4)
    
    async def _calculate_take_profit(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """Calculate take profit level"""
        try:
            risk_distance = abs(entry_price - stop_loss)
            reward_ratio = self.config.trading.reward_risk_ratio
            
            if direction.upper() == 'BUY':
                return entry_price + (risk_distance * reward_ratio)
            else:
                return entry_price - (risk_distance * reward_ratio)
                
        except Exception as e:
            logger.error(f"Failed to calculate take profit: {e}")
            return entry_price
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        try:
            # Calculate used margin from open positions
            used_margin = sum(
                pos.get('position_value', 0) / pos.get('leverage', 1)
                for pos in self.open_positions.values()
            )
            
            # Calculate available equity
            available_equity = max(0, self.current_equity - used_margin)
            
            # Calculate total exposure
            total_exposure = sum(
                pos.get('position_value', 0)
                for pos in self.open_positions.values()
            )
            
            # Calculate daily drawdown
            daily_start_equity = self.current_equity - self.daily_pnl
            daily_drawdown = max(0, (daily_start_equity - self.current_equity) / daily_start_equity) if daily_start_equity > 0 else 0
            
            # Update max drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Determine risk level
            risk_level = self._determine_risk_level(daily_drawdown, current_drawdown, len(self.open_positions))
            
            return RiskMetrics(
                total_equity=self.current_equity,
                available_equity=available_equity,
                used_margin=used_margin,
                daily_pnl=self.daily_pnl,
                daily_drawdown=daily_drawdown,
                max_drawdown=self.max_drawdown,
                risk_per_trade=self.config.trading.risk_per_trade_pct,
                total_exposure=total_exposure,
                open_positions=len(self.open_positions),
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            return RiskMetrics(
                total_equity=self.current_equity,
                available_equity=self.current_equity,
                used_margin=0,
                daily_pnl=0,
                daily_drawdown=0,
                max_drawdown=0,
                risk_per_trade=self.config.trading.risk_per_trade_pct,
                total_exposure=0,
                open_positions=0,
                risk_level=RiskLevel.MEDIUM
            )
    
    def _determine_risk_level(self, daily_drawdown: float, max_drawdown: float, open_positions: int) -> RiskLevel:
        """Determine current risk level"""
        try:
            risk_score = 0
            
            # Daily drawdown factor
            if daily_drawdown > 0.05:  # 5%
                risk_score += 3
            elif daily_drawdown > 0.03:  # 3%
                risk_score += 2
            elif daily_drawdown > 0.01:  # 1%
                risk_score += 1
            
            # Max drawdown factor
            if max_drawdown > 0.15:  # 15%
                risk_score += 3
            elif max_drawdown > 0.10:  # 10%
                risk_score += 2
            elif max_drawdown > 0.05:  # 5%
                risk_score += 1
            
            # Position concentration factor
            max_positions = self.config.trading.max_positions
            if open_positions >= max_positions:
                risk_score += 2
            elif open_positions >= max_positions * 0.8:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 6:
                return RiskLevel.EXTREME
            elif risk_score >= 4:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception:
            return RiskLevel.MEDIUM
    
    async def validate_trade(self, trade_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate if a trade meets risk management criteria"""
        try:
            # Check if trading is halted
            if self.trading_halted:
                return False, f"Trading halted: {self.halt_reason}"
            
            # Reset daily tracking if new day
            await self._check_daily_reset()
            
            symbol = trade_data.get('symbol')
            quantity = trade_data.get('quantity', 0)
            entry_price = trade_data.get('entry_price', 0)
            
            if not symbol or quantity <= 0 or entry_price <= 0:
                return False, "Invalid trade data"
            
            # Check if symbol is already at position limit
            if symbol in self.open_positions:
                return False, f"Position already exists for {symbol}"
            
            # Get risk metrics
            risk_metrics = await self.get_risk_metrics()
            
            # Check daily drawdown
            if risk_metrics.daily_drawdown >= self.config.trading.daily_max_drawdown_pct / 100:
                await self._halt_trading("Daily drawdown limit exceeded")
                return False, "Daily drawdown limit exceeded"
            
            # Check position limits
            if risk_metrics.open_positions >= self.config.trading.max_positions:
                return False, "Maximum positions limit reached"
            
            # Check position size
            position_value = quantity * entry_price
            max_position_value = risk_metrics.available_equity * 0.2  # Max 20% per position
            
            if position_value > max_position_value:
                return False, f"Position size too large: ${position_value:.2f} > ${max_position_value:.2f}"
            
            # Check available equity
            if position_value > risk_metrics.available_equity:
                return False, "Insufficient available equity"
            
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Failed to validate trade: {e}")
            return False, f"Validation error: {e}"
    
    async def add_position(self, position_data: Dict[str, Any]) -> bool:
        """Add a new position to tracking"""
        try:
            symbol = position_data.get('symbol')
            if not symbol:
                return False
            
            # Add position to tracking
            self.open_positions[symbol] = {
                'symbol': symbol,
                'quantity': position_data.get('quantity', 0),
                'entry_price': position_data.get('entry_price', 0),
                'stop_loss': position_data.get('stop_loss', 0),
                'take_profit': position_data.get('take_profit', 0),
                'direction': position_data.get('direction', 'BUY'),
                'position_value': position_data.get('position_value', 0),
                'leverage': position_data.get('leverage', 1),
                'timestamp': datetime.now().isoformat(),
                'unrealized_pnl': 0.0
            }
            
            # Add to daily trades
            self.daily_trades.append({
                'symbol': symbol,
                'action': 'OPEN',
                'quantity': position_data.get('quantity', 0),
                'price': position_data.get('entry_price', 0),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Added position for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False
    
    async def close_position(self, symbol: str, exit_price: float, reason: str = "manual") -> bool:
        """Close a position and update P&L"""
        try:
            if symbol not in self.open_positions:
                logger.warning(f"Position not found for {symbol}")
                return False
            
            position = self.open_positions[symbol]
            
            # Calculate P&L
            entry_price = position['entry_price']
            quantity = position['quantity']
            direction = position['direction']
            
            if direction.upper() == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            # Update equity and daily P&L
            self.current_equity += pnl
            self.daily_pnl += pnl
            
            # Add to daily trades
            self.daily_trades.append({
                'symbol': symbol,
                'action': 'CLOSE',
                'quantity': quantity,
                'price': exit_price,
                'pnl': pnl,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            })
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            logger.info(f"Closed position for {symbol}: P&L ${pnl:.2f}")
            
            # Publish risk update
            await self._publish_risk_update()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return False
    
    async def update_position_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized P&L for a position"""
        try:
            if symbol not in self.open_positions:
                return
            
            position = self.open_positions[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            direction = position['direction']
            
            if direction.upper() == 'BUY':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
        except Exception as e:
            logger.error(f"Failed to update P&L for {symbol}: {e}")
    
    async def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if stop loss or take profit should be triggered"""
        try:
            if symbol not in self.open_positions:
                return None
            
            position = self.open_positions[symbol]
            direction = position['direction']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if direction.upper() == 'BUY':
                if current_price <= stop_loss:
                    return 'STOP_LOSS'
                elif current_price >= take_profit:
                    return 'TAKE_PROFIT'
            else:
                if current_price >= stop_loss:
                    return 'STOP_LOSS'
                elif current_price <= take_profit:
                    return 'TAKE_PROFIT'
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check SL/TP for {symbol}: {e}")
            return None
    
    async def _check_daily_reset(self) -> None:
        """Check if we need to reset daily tracking"""
        try:
            now = datetime.now()
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if current_day > self.last_reset:
                # Reset daily tracking
                self.daily_trades = []
                self.daily_pnl = 0.0
                self.last_reset = current_day
                
                # Resume trading if halted due to daily limits
                if self.trading_halted and "daily" in self.halt_reason.lower():
                    self.trading_halted = False
                    self.halt_reason = None
                    logger.info("Trading resumed - daily reset")
                
        except Exception as e:
            logger.error(f"Failed to check daily reset: {e}")
    
    async def _halt_trading(self, reason: str) -> None:
        """Halt trading with reason"""
        self.trading_halted = True
        self.halt_reason = reason
        logger.error(f"TRADING HALTED: {reason}")
        
        # Publish halt notification
        await publish_analysis_result(
            symbol="SYSTEM",
            analysis_type="risk_management",
            result={
                "action": "TRADING_HALTED",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            },
            confidence=1.0
        )
    
    async def resume_trading(self) -> bool:
        """Resume trading if conditions allow"""
        try:
            # Check if conditions allow resuming
            risk_metrics = await self.get_risk_metrics()
            
            if risk_metrics.daily_drawdown < self.config.trading.daily_max_drawdown_pct / 100:
                self.trading_halted = False
                self.halt_reason = None
                logger.info("Trading resumed manually")
                return True
            else:
                logger.warning("Cannot resume trading - risk conditions not met")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resume trading: {e}")
            return False
    
    async def _publish_risk_update(self) -> None:
        """Publish risk metrics update"""
        try:
            risk_metrics = await self.get_risk_metrics()
            
            await publish_analysis_result(
                symbol="PORTFOLIO",
                analysis_type="risk_metrics",
                result={
                    "total_equity": risk_metrics.total_equity,
                    "available_equity": risk_metrics.available_equity,
                    "daily_pnl": risk_metrics.daily_pnl,
                    "daily_drawdown": risk_metrics.daily_drawdown,
                    "max_drawdown": risk_metrics.max_drawdown,
                    "open_positions": risk_metrics.open_positions,
                    "risk_level": risk_metrics.risk_level.value,
                    "trading_halted": self.trading_halted,
                    "timestamp": datetime.now().isoformat()
                },
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Failed to publish risk update: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            risk_metrics = await self.get_risk_metrics()
            
            # Calculate position summaries
            position_summaries = []
            total_unrealized_pnl = 0
            
            for symbol, position in self.open_positions.items():
                unrealized_pnl = position.get('unrealized_pnl', 0)
                total_unrealized_pnl += unrealized_pnl
                
                position_summaries.append({
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'current_price': position.get('current_price', position['entry_price']),
                    'unrealized_pnl': unrealized_pnl,
                    'direction': position['direction'],
                    'position_value': position['position_value']
                })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'equity': {
                    'total': risk_metrics.total_equity,
                    'available': risk_metrics.available_equity,
                    'used_margin': risk_metrics.used_margin
                },
                'pnl': {
                    'daily_realized': self.daily_pnl,
                    'total_unrealized': total_unrealized_pnl,
                    'daily_total': self.daily_pnl + total_unrealized_pnl
                },
                'drawdown': {
                    'daily': risk_metrics.daily_drawdown,
                    'maximum': risk_metrics.max_drawdown
                },
                'positions': {
                    'count': len(self.open_positions),
                    'max_allowed': self.config.trading.max_positions,
                    'details': position_summaries
                },
                'risk': {
                    'level': risk_metrics.risk_level.value,
                    'per_trade_pct': risk_metrics.risk_per_trade,
                    'total_exposure': risk_metrics.total_exposure
                },
                'trading_status': {
                    'halted': self.trading_halted,
                    'halt_reason': self.halt_reason
                },
                'daily_trades': len(self.daily_trades)
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {}


# Global risk agent instance
_risk_agent: Optional[RiskManagementAgent] = None


async def get_risk_agent() -> RiskManagementAgent:
    """Get the global risk management agent"""
    global _risk_agent
    
    if _risk_agent is None:
        _risk_agent = RiskManagementAgent()
    
    return _risk_agent