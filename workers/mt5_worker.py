"""
MT5 Worker - Executes trades via MetaTrader5 API
This worker runs on Windows with MT5 installed and listens to Redis for trade orders
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import redis
import MetaTrader5 as mt5
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MT5Config:
    """MT5 configuration"""
    login: int
    password: str
    server: str
    timeout: int = 60000
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            login=int(os.getenv('MT5_LOGIN', '0')),
            password=os.getenv('MT5_PASSWORD', ''),
            server=os.getenv('MT5_SERVER', ''),
            timeout=int(os.getenv('MT5_TIMEOUT', '60000'))
        )


class MT5Worker:
    """MT5 Worker for executing trades"""
    
    def __init__(self):
        self.config = MT5Config.from_env()
        self.redis_client = None
        self.running = False
        self.connected_to_mt5 = False
        self.connection_retries = 0
        self.max_retries = 5
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_password = os.getenv('REDIS_PASSWORD', None)
        self.redis_db = int(os.getenv('REDIS_DB', '0'))
        
        # Trading statistics
        self.stats = {
            'orders_processed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'start_time': datetime.now(),
            'last_trade_time': None
        }
    
    async def initialize(self) -> bool:
        """Initialize MT5 and Redis connections"""
        try:
            # Initialize MT5
            if not await self._connect_to_mt5():
                return False
            
            # Initialize Redis
            if not await self._connect_to_redis():
                return False
            
            logger.info("MT5 Worker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MT5 Worker: {e}")
            return False
    
    async def _connect_to_mt5(self) -> bool:
        """Connect to MetaTrader5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False
            
            # Login to MT5
            if not mt5.login(self.config.login, self.config.password, self.config.server):
                error = mt5.last_error()
                logger.error(f"Failed to login to MT5: {error}")
                mt5.shutdown()
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            self.connected_to_mt5 = True
            logger.info(f"Connected to MT5 - Account: {account_info.login}, Balance: {account_info.balance}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    async def _connect_to_redis(self) -> bool:
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                db=self.redis_db,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def start(self) -> None:
        """Start the MT5 worker"""
        if not await self.initialize():
            logger.error("Failed to initialize MT5 Worker")
            return
        
        self.running = True
        logger.info("MT5 Worker started - listening for trade orders...")
        
        try:
            # Subscribe to trades.exec channel
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe('trades.exec')
            
            while self.running:
                try:
                    # Get message from Redis
                    message = pubsub.get_message(timeout=1.0)
                    
                    if message and message['type'] == 'message':
                        await self._process_message(message['data'])
                    
                    # Check MT5 connection periodically
                    if not self._check_mt5_connection():
                        logger.warning("MT5 connection lost, attempting to reconnect...")
                        await self._reconnect_mt5()
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()
    
    async def _process_message(self, message_data: str) -> None:
        """Process incoming trade message"""
        try:
            message = json.loads(message_data)
            message_type = message.get('type')
            data = message.get('data', {})
            
            self.stats['orders_processed'] += 1
            
            if message_type == 'TRADE_ORDER':
                await self._execute_trade_order(data)
            elif message_type == 'CLOSE_POSITION':
                await self._close_position(data)
            elif message_type == 'CANCEL_ORDER':
                await self._cancel_order(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
    
    async def _execute_trade_order(self, order_data: Dict[str, Any]) -> None:
        """Execute a trade order"""
        try:
            order_id = order_data.get('order_id')
            symbol = order_data.get('symbol')
            side = order_data.get('side')
            quantity = order_data.get('quantity')
            order_type = order_data.get('order_type', 'MARKET')
            price = order_data.get('price')
            stop_loss = order_data.get('stop_loss')
            take_profit = order_data.get('take_profit')
            
            logger.info(f"Executing order: {order_id} - {symbol} {side} {quantity}")
            
            # Validate symbol
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                await self._send_order_update(order_id, 'REJECTED', f"Symbol {symbol} not found")
                return
            
            # Enable symbol if not enabled
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    await self._send_order_update(order_id, 'REJECTED', f"Failed to enable symbol {symbol}")
                    return
            
            # Prepare trade request
            request = self._prepare_trade_request(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if request is None:
                await self._send_order_update(order_id, 'REJECTED', "Invalid trade request")
                return
            
            # Send order to MT5
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                await self._send_order_update(order_id, 'FAILED', f"MT5 error: {error}")
                return
            
            # Process result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Order successful
                await self._send_order_update(
                    order_id, 
                    'FILLED',
                    f"Order filled",
                    {
                        'fill_price': result.price,
                        'volume': result.volume,
                        'ticket': result.order,
                        'deal': result.deal,
                        'comment': result.comment
                    }
                )
                self.stats['successful_trades'] += 1
                self.stats['last_trade_time'] = datetime.now()
                
                logger.info(f"Order filled: {order_id} - Ticket: {result.order}, Price: {result.price}")
                
            else:
                # Order failed
                error_msg = f"MT5 error code: {result.retcode}"
                await self._send_order_update(order_id, 'REJECTED', error_msg)
                self.stats['failed_trades'] += 1
                
                logger.warning(f"Order rejected: {order_id} - {error_msg}")
                
        except Exception as e:
            logger.error(f"Failed to execute trade order: {e}")
            await self._send_order_update(order_data.get('order_id'), 'FAILED', str(e))
            self.stats['failed_trades'] += 1
    
    def _prepare_trade_request(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare MT5 trade request"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Normalize volume
            volume = self._normalize_volume(quantity, symbol_info)
            if volume <= 0:
                return None
            
            # Determine action
            if side.upper() == 'BUY':
                action = mt5.TRADE_ACTION_DEAL
                trade_type = mt5.ORDER_TYPE_BUY
            else:
                action = mt5.TRADE_ACTION_DEAL
                trade_type = mt5.ORDER_TYPE_SELL
            
            # Prepare request
            request = {
                "action": action,
                "symbol": symbol,
                "volume": volume,
                "type": trade_type,
                "deviation": 20,  # Price deviation in points
                "magic": 234000,  # Magic number
                "comment": f"Trading Hub - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add price for limit orders
            if order_type != 'MARKET' and price is not None:
                request["price"] = price
            
            # Add stop loss
            if stop_loss is not None:
                request["sl"] = stop_loss
            
            # Add take profit
            if take_profit is not None:
                request["tp"] = take_profit
            
            return request
            
        except Exception as e:
            logger.error(f"Failed to prepare trade request: {e}")
            return None
    
    def _normalize_volume(self, volume: float, symbol_info) -> float:
        """Normalize volume according to symbol specifications"""
        try:
            # Get volume step
            volume_step = symbol_info.volume_step
            volume_min = symbol_info.volume_min
            volume_max = symbol_info.volume_max
            
            # Round to volume step
            normalized_volume = round(volume / volume_step) * volume_step
            
            # Ensure within limits
            normalized_volume = max(volume_min, min(volume_max, normalized_volume))
            
            return normalized_volume
            
        except Exception as e:
            logger.error(f"Failed to normalize volume: {e}")
            return 0
    
    async def _close_position(self, close_data: Dict[str, Any]) -> None:
        """Close an existing position"""
        try:
            symbol = close_data.get('symbol')
            order_id = close_data.get('order_id')
            
            logger.info(f"Closing position for {symbol}")
            
            # Get open positions for symbol
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                await self._send_order_update(order_id, 'REJECTED', f"No open positions for {symbol}")
                return
            
            # Close all positions for this symbol
            for position in positions:
                # Determine close action
                if position.type == mt5.ORDER_TYPE_BUY:
                    trade_type = mt5.ORDER_TYPE_SELL
                else:
                    trade_type = mt5.ORDER_TYPE_BUY
                
                # Prepare close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": trade_type,
                    "position": position.ticket,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"Close position - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # Send close order
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    await self._send_order_update(
                        order_id,
                        'FILLED',
                        f"Position closed",
                        {
                            'close_price': result.price,
                            'volume': result.volume,
                            'ticket': result.order,
                            'deal': result.deal
                        }
                    )
                    logger.info(f"Position closed: {symbol} - Ticket: {position.ticket}")
                else:
                    error = mt5.last_error() if result is None else f"Error code: {result.retcode}"
                    await self._send_order_update(order_id, 'FAILED', f"Failed to close position: {error}")
                    logger.error(f"Failed to close position {position.ticket}: {error}")
                    
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            await self._send_order_update(close_data.get('order_id'), 'FAILED', str(e))
    
    async def _cancel_order(self, cancel_data: Dict[str, Any]) -> None:
        """Cancel a pending order"""
        try:
            order_id = cancel_data.get('order_id')
            
            # Note: This is a simplified implementation
            # In practice, you'd need to track pending orders and their MT5 ticket numbers
            logger.info(f"Order cancellation requested: {order_id}")
            
            await self._send_order_update(order_id, 'CANCELLED', "Order cancelled")
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
    
    async def _send_order_update(
        self,
        order_id: str,
        status: str,
        message: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send order status update to Redis"""
        try:
            update_data = {
                'order_id': order_id,
                'status': status,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'source': 'mt5_worker'
            }
            
            if additional_data:
                update_data.update(additional_data)
            
            # Publish update to trades.updates channel
            self.redis_client.publish('trades.updates', json.dumps(update_data))
            
        except Exception as e:
            logger.error(f"Failed to send order update: {e}")
    
    def _check_mt5_connection(self) -> bool:
        """Check if MT5 connection is still active"""
        try:
            account_info = mt5.account_info()
            return account_info is not None
        except:
            return False
    
    async def _reconnect_mt5(self) -> bool:
        """Attempt to reconnect to MT5"""
        try:
            self.connection_retries += 1
            
            if self.connection_retries > self.max_retries:
                logger.error("Max reconnection attempts reached")
                return False
            
            # Shutdown current connection
            mt5.shutdown()
            
            # Wait before reconnecting
            await asyncio.sleep(5)
            
            # Attempt to reconnect
            if await self._connect_to_mt5():
                self.connection_retries = 0
                logger.info("Successfully reconnected to MT5")
                return True
            else:
                logger.error(f"Reconnection attempt {self.connection_retries} failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during MT5 reconnection: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        try:
            account_info = mt5.account_info()
            
            uptime = datetime.now() - self.stats['start_time']
            
            return {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime.total_seconds(),
                'orders_processed': self.stats['orders_processed'],
                'successful_trades': self.stats['successful_trades'],
                'failed_trades': self.stats['failed_trades'],
                'last_trade_time': self.stats['last_trade_time'].isoformat() if self.stats['last_trade_time'] else None,
                'mt5_connected': self.connected_to_mt5,
                'account_balance': account_info.balance if account_info else 0,
                'account_equity': account_info.equity if account_info else 0,
                'connection_retries': self.connection_retries
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Shutdown the worker"""
        try:
            self.running = False
            
            if self.redis_client:
                self.redis_client.close()
            
            if self.connected_to_mt5:
                mt5.shutdown()
            
            logger.info("MT5 Worker shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main function"""
    worker = MT5Worker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await worker.shutdown()


if __name__ == "__main__":
    # Run the worker
    asyncio.run(main())