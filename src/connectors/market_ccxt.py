"""
CCXT-based cryptocurrency market data connector
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import ccxt.async_support as ccxt

from ..config import get_config
from ..bus import publish_market_data

logger = logging.getLogger(__name__)


class CCXTConnector:
    """CCXT-based cryptocurrency market data connector"""
    
    def __init__(self, exchange_name: str = "binance"):
        self.exchange_name = exchange_name
        self.exchange = None
        self.symbols: List[str] = []
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> None:
        """Initialize the exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': '',  # Read-only, no API key needed for public data
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': False,
            })
            
            await self.exchange.load_markets()
            logger.info(f"Initialized {self.exchange_name} exchange connector")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current ticker data for a symbol"""
        try:
            if not self.exchange:
                await self.initialize()
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
                'high': ticker['high'],
                'low': ticker['low'],
                'open': ticker['open'],
                'close': ticker['close'],
                'change': ticker['change'],
                'percentage': ticker['percentage']
            }
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1m', 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol"""
        try:
            if not self.exchange:
                await self.initialize()
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book for a symbol"""
        try:
            if not self.exchange:
                await self.initialize()
            
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            return {
                'symbol': symbol,
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': datetime.fromtimestamp(order_book['timestamp'] / 1000),
                'nonce': order_book['nonce']
            }
            
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None
    
    async def get_trades(self, symbol: str, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """Get recent trades for a symbol"""
        try:
            if not self.exchange:
                await self.initialize()
            
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            return [
                {
                    'id': trade['id'],
                    'symbol': symbol,
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side'],
                    'timestamp': datetime.fromtimestamp(trade['timestamp'] / 1000)
                }
                for trade in trades
            ]
            
        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            return None
    
    async def start_price_stream(self, symbols: List[str], interval: int = 5) -> None:
        """Start streaming price data for symbols"""
        self.symbols = symbols
        self.running = True
        
        # Create tasks for each symbol
        for symbol in symbols:
            task = asyncio.create_task(self._price_stream_worker(symbol, interval))
            self.tasks.append(task)
        
        logger.info(f"Started price streaming for {len(symbols)} symbols")
    
    async def _price_stream_worker(self, symbol: str, interval: int) -> None:
        """Worker task for streaming price data"""
        while self.running:
            try:
                ticker = await self.get_ticker(symbol)
                if ticker:
                    await publish_market_data(
                        symbol=symbol,
                        price=ticker['price'],
                        volume=ticker['volume'],
                        bid=ticker['bid'],
                        ask=ticker['ask']
                    )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in price stream for {symbol}: {e}")
                await asyncio.sleep(interval * 2)  # Back off on error
    
    async def stop_price_stream(self) -> None:
        """Stop price streaming"""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        logger.info("Stopped price streaming")
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        try:
            if not self.exchange:
                await self.initialize()
            
            markets = self.exchange.markets
            symbols = [symbol for symbol in markets.keys() if markets[symbol]['active']]
            
            # Filter for major pairs
            major_symbols = [
                symbol for symbol in symbols 
                if any(base in symbol for base in ['BTC', 'ETH', 'USDT', 'USDC', 'BNB'])
            ]
            
            return sorted(major_symbols)
            
        except Exception as e:
            logger.error(f"Failed to get supported symbols: {e}")
            return []
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a symbol"""
        try:
            if not self.exchange:
                await self.initialize()
            
            if symbol not in self.exchange.markets:
                return None
            
            market = self.exchange.markets[symbol]
            return {
                'symbol': symbol,
                'base': market['base'],
                'quote': market['quote'],
                'active': market['active'],
                'type': market['type'],
                'spot': market['spot'],
                'future': market['future'],
                'option': market['option'],
                'precision': market['precision'],
                'limits': market['limits'],
                'fees': market.get('fees', {}),
                'info': market.get('info', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        risk_amount: float, 
        entry_price: float, 
        stop_loss: float
    ) -> float:
        """Calculate position size based on risk"""
        try:
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                return 0.0
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit == 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            # Apply precision limits
            precision = symbol_info['precision']['amount']
            if precision:
                position_size = round(position_size, precision)
            
            # Apply minimum/maximum limits
            limits = symbol_info['limits']['amount']
            if limits['min'] and position_size < limits['min']:
                position_size = limits['min']
            if limits['max'] and position_size > limits['max']:
                position_size = limits['max']
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            return 0.0
    
    async def close(self) -> None:
        """Close the exchange connection"""
        await self.stop_price_stream()
        
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
        
        logger.info("Closed CCXT connector")


# Global connector instance
_ccxt_connector: Optional[CCXTConnector] = None


async def get_ccxt_connector() -> CCXTConnector:
    """Get the global CCXT connector instance"""
    global _ccxt_connector
    
    if _ccxt_connector is None:
        config = get_config()
        _ccxt_connector = CCXTConnector(config.market_data.crypto_exchange)
        await _ccxt_connector.initialize()
    
    return _ccxt_connector


async def close_ccxt_connector() -> None:
    """Close the global CCXT connector"""
    global _ccxt_connector
    
    if _ccxt_connector:
        await _ccxt_connector.close()
        _ccxt_connector = None