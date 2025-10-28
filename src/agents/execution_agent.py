"""
Execution Agent for publishing trade orders to Redis
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid
from enum import Enum

from ..config import get_config
from ..bus import get_message_bus, publish_trade_signal
from .risk_agent import get_risk_agent

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SENT = "SENT"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class ExecutionAgent:
    """Execution agent for managing trade orders"""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0
        }
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Execute a trade order"""
        try:
            # Validate inputs
            if not symbol or quantity <= 0:
                logger.error("Invalid trade parameters")
                return None
            
            # Normalize side
            side = side.upper()
            if side not in ['BUY', 'SELL']:
                logger.error(f"Invalid order side: {side}")
                return None
            
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Get risk agent for validation
            risk_agent = await get_risk_agent()
            
            # Prepare trade data for validation
            trade_data = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': price or 0,  # Will be filled by market price if not provided
                'side': side,
                'order_type': order_type
            }
            
            # Validate trade with risk management
            is_valid, validation_message = await risk_agent.validate_trade(trade_data)
            if not is_valid:
                logger.warning(f"Trade validation failed: {validation_message}")
                await self._record_failed_order(order_id, symbol, side, quantity, validation_message)
                return None
            
            # Create order
            order = await self._create_order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata or {}
            )
            
            # Send order to Redis
            success = await self._send_order_to_redis(order)
            if success:
                # Add to pending orders
                self.pending_orders[order_id] = order
                self.execution_stats['total_orders'] += 1
                
                logger.info(f"Order sent successfully: {order_id} - {symbol} {side} {quantity}")
                return order_id
            else:
                await self._record_failed_order(order_id, symbol, side, quantity, "Failed to send to Redis")
                return None
                
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return None
    
    async def _create_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create order dictionary"""
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': OrderStatus.PENDING.value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'retry_count': 0,
            'max_retries': 3
        }
        
        # Add execution parameters
        order['execution_params'] = {
            'timeout': self.config.trading.order_timeout,
            'slippage_tolerance': self.config.trading.slippage_tolerance,
            'partial_fill_allowed': True
        }
        
        return order
    
    async def _send_order_to_redis(self, order: Dict[str, Any]) -> bool:
        """Send order to Redis for MT5 execution"""
        try:
            # Get message bus
            bus = await get_message_bus()
            
            # Prepare order message
            order_message = {
                'type': 'TRADE_ORDER',
                'data': order,
                'timestamp': datetime.now().isoformat(),
                'source': 'execution_agent'
            }
            
            # Publish to trades.exec channel
            await bus.publish('trades.exec', json.dumps(order_message))
            
            # Update order status
            order['status'] = OrderStatus.SENT.value
            order['sent_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Order {order['order_id']} sent to Redis channel 'trades.exec'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send order to Redis: {e}")
            return False
    
    async def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """Cancel a pending order"""
        try:
            if order_id not in self.pending_orders:
                logger.warning(f"Order not found: {order_id}")
                return False
            
            order = self.pending_orders[order_id]
            
            # Create cancellation message
            cancel_message = {
                'type': 'CANCEL_ORDER',
                'data': {
                    'order_id': order_id,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                },
                'source': 'execution_agent'
            }
            
            # Send cancellation to Redis
            bus = await get_message_bus()
            await bus.publish('trades.exec', json.dumps(cancel_message))
            
            # Update order status
            order['status'] = OrderStatus.CANCELLED.value
            order['cancel_reason'] = reason
            order['cancel_timestamp'] = datetime.now().isoformat()
            
            # Move to history
            self.order_history.append(order)
            del self.pending_orders[order_id]
            
            self.execution_stats['cancelled_orders'] += 1
            
            logger.info(f"Order cancelled: {order_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """Close an existing position"""
        try:
            # Get risk agent to check if position exists
            risk_agent = await get_risk_agent()
            
            if symbol not in risk_agent.open_positions:
                logger.warning(f"No open position found for {symbol}")
                return False
            
            position = risk_agent.open_positions[symbol]
            
            # Determine close side (opposite of position direction)
            close_side = "SELL" if position['direction'].upper() == "BUY" else "BUY"
            
            # Create close order
            close_message = {
                'type': 'CLOSE_POSITION',
                'data': {
                    'symbol': symbol,
                    'side': close_side,
                    'quantity': position['quantity'],
                    'reason': reason,
                    'timestamp': datetime.now().isoformat(),
                    'order_id': str(uuid.uuid4())
                },
                'source': 'execution_agent'
            }
            
            # Send to Redis
            bus = await get_message_bus()
            await bus.publish('trades.exec', json.dumps(close_message))
            
            logger.info(f"Position close order sent for {symbol}: {close_side} {position['quantity']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return False
    
    async def close_all_positions(self, reason: str = "emergency") -> int:
        """Close all open positions"""
        try:
            risk_agent = await get_risk_agent()
            closed_count = 0
            
            for symbol in list(risk_agent.open_positions.keys()):
                success = await self.close_position(symbol, reason)
                if success:
                    closed_count += 1
            
            logger.info(f"Closed {closed_count} positions - reason: {reason}")
            return closed_count
            
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return 0
    
    async def handle_order_update(self, update_data: Dict[str, Any]) -> None:
        """Handle order status updates from MT5 worker"""
        try:
            order_id = update_data.get('order_id')
            if not order_id:
                return
            
            status = update_data.get('status')
            
            # Update pending order if exists
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                order.update(update_data)
                
                # Handle different status updates
                if status == OrderStatus.FILLED.value:
                    await self._handle_order_filled(order)
                elif status == OrderStatus.REJECTED.value:
                    await self._handle_order_rejected(order)
                elif status == OrderStatus.FAILED.value:
                    await self._handle_order_failed(order)
                
                # Move completed orders to history
                if status in [OrderStatus.FILLED.value, OrderStatus.REJECTED.value, OrderStatus.CANCELLED.value]:
                    self.order_history.append(order)
                    del self.pending_orders[order_id]
            
        except Exception as e:
            logger.error(f"Failed to handle order update: {e}")
    
    async def _handle_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle filled order"""
        try:
            self.execution_stats['successful_orders'] += 1
            
            # Add position to risk management
            risk_agent = await get_risk_agent()
            
            position_data = {
                'symbol': order['symbol'],
                'quantity': order['quantity'],
                'entry_price': order.get('fill_price', order.get('price', 0)),
                'stop_loss': order.get('stop_loss'),
                'take_profit': order.get('take_profit'),
                'direction': order['side'],
                'position_value': order['quantity'] * order.get('fill_price', order.get('price', 0)),
                'leverage': order.get('leverage', 1),
                'order_id': order['order_id']
            }
            
            await risk_agent.add_position(position_data)
            
            logger.info(f"Order filled: {order['order_id']} - {order['symbol']} {order['side']} {order['quantity']}")
            
        except Exception as e:
            logger.error(f"Failed to handle filled order: {e}")
    
    async def _handle_order_rejected(self, order: Dict[str, Any]) -> None:
        """Handle rejected order"""
        try:
            self.execution_stats['failed_orders'] += 1
            
            reject_reason = order.get('reject_reason', 'Unknown')
            logger.warning(f"Order rejected: {order['order_id']} - {reject_reason}")
            
        except Exception as e:
            logger.error(f"Failed to handle rejected order: {e}")
    
    async def _handle_order_failed(self, order: Dict[str, Any]) -> None:
        """Handle failed order with retry logic"""
        try:
            retry_count = order.get('retry_count', 0)
            max_retries = order.get('max_retries', 3)
            
            if retry_count < max_retries:
                # Retry the order
                order['retry_count'] = retry_count + 1
                order['status'] = OrderStatus.PENDING.value
                
                # Wait before retry
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
                # Resend order
                success = await self._send_order_to_redis(order)
                if not success:
                    order['status'] = OrderStatus.FAILED.value
                    self.execution_stats['failed_orders'] += 1
                
                logger.info(f"Retrying order {order['order_id']} (attempt {retry_count + 1}/{max_retries})")
            else:
                # Max retries reached
                self.execution_stats['failed_orders'] += 1
                logger.error(f"Order failed after {max_retries} retries: {order['order_id']}")
            
        except Exception as e:
            logger.error(f"Failed to handle failed order: {e}")
    
    async def _record_failed_order(self, order_id: str, symbol: str, side: str, quantity: float, reason: str) -> None:
        """Record a failed order"""
        try:
            failed_order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'status': OrderStatus.FAILED.value,
                'failure_reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            self.order_history.append(failed_order)
            self.execution_stats['failed_orders'] += 1
            
        except Exception as e:
            logger.error(f"Failed to record failed order: {e}")
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        try:
            success_rate = 0
            if self.execution_stats['total_orders'] > 0:
                success_rate = self.execution_stats['successful_orders'] / self.execution_stats['total_orders']
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_orders': self.execution_stats['total_orders'],
                'successful_orders': self.execution_stats['successful_orders'],
                'failed_orders': self.execution_stats['failed_orders'],
                'cancelled_orders': self.execution_stats['cancelled_orders'],
                'pending_orders': len(self.pending_orders),
                'success_rate': success_rate,
                'recent_orders': self.order_history[-10:] if self.order_history else []
            }
            
        except Exception as e:
            logger.error(f"Failed to get execution stats: {e}")
            return {}
    
    async def start_order_monitoring(self) -> None:
        """Start monitoring order status updates"""
        self.running = True
        
        try:
            # Subscribe to order updates channel
            bus = await get_message_bus()
            
            async def order_update_handler(message: str):
                try:
                    update_data = json.loads(message)
                    await self.handle_order_update(update_data)
                except Exception as e:
                    logger.error(f"Failed to process order update: {e}")
            
            # Subscribe to order updates
            await bus.subscribe('trades.updates', order_update_handler)
            
            logger.info("Started order monitoring")
            
            # Keep monitoring running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in order monitoring: {e}")
    
    async def stop_order_monitoring(self) -> None:
        """Stop order monitoring"""
        self.running = False
        logger.info("Stopped order monitoring")


# Global execution agent instance
_execution_agent: Optional[ExecutionAgent] = None


async def get_execution_agent() -> ExecutionAgent:
    """Get the global execution agent"""
    global _execution_agent
    
    if _execution_agent is None:
        _execution_agent = ExecutionAgent()
    
    return _execution_agent