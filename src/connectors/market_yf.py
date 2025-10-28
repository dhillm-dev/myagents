"""
yfinance-based market data connector for stocks and forex
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

from ..config import get_config
from ..bus import publish_market_data

logger = logging.getLogger(__name__)


class YFinanceConnector:
    """yfinance-based market data connector"""
    
    def __init__(self):
        self.symbols: List[str] = []
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(seconds=30)  # 30 second cache
        
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current ticker data for a symbol"""
        try:
            # Check cache first
            if self._is_cached(symbol):
                return self._cache[symbol]
            
            # Fetch data in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ticker_data = await loop.run_in_executor(
                self.executor, self._fetch_ticker_sync, symbol
            )
            
            if ticker_data:
                # Cache the result
                self._cache[symbol] = ticker_data
                self._cache_expiry[symbol] = datetime.now() + self.cache_duration
            
            return ticker_data
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    def _fetch_ticker_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Synchronous ticker fetch for thread pool"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price from different sources
            current_price = (
                info.get('currentPrice') or 
                info.get('regularMarketPrice') or 
                info.get('previousClose')
            )
            
            if not current_price:
                return None
            
            return {
                'symbol': symbol,
                'price': current_price,
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('volume', 0),
                'timestamp': datetime.now(),
                'high': info.get('dayHigh'),
                'low': info.get('dayLow'),
                'open': info.get('regularMarketOpen'),
                'close': info.get('previousClose'),
                'change': info.get('regularMarketChange'),
                'percentage': info.get('regularMarketChangePercent'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow')
            }
            
        except Exception as e:
            logger.error(f"Sync fetch failed for {symbol}: {e}")
            return None
    
    def _is_cached(self, symbol: str) -> bool:
        """Check if symbol data is cached and not expired"""
        if symbol not in self._cache:
            return False
        
        if symbol not in self._cache_expiry:
            return False
        
        return datetime.now() < self._cache_expiry[symbol]
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        period: str = "1d", 
        interval: str = "1m",
        days: int = 30
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol"""
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                self.executor, self._fetch_ohlcv_sync, symbol, period, interval, days
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return None
    
    def _fetch_ohlcv_sync(
        self, 
        symbol: str, 
        period: str, 
        interval: str, 
        days: int
    ) -> Optional[pd.DataFrame]:
        """Synchronous OHLCV fetch for thread pool"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch historical data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if df.empty:
                return None
            
            # Rename columns to standard format
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for {symbol}")
                return None
            
            return df[required_cols]
            
        except Exception as e:
            logger.error(f"Sync OHLCV fetch failed for {symbol}: {e}")
            return None
    
    async def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information for a stock symbol"""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                self.executor, self._fetch_company_info_sync, symbol
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get company info for {symbol}: {e}")
            return None
    
    def _fetch_company_info_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Synchronous company info fetch"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'website': info.get('website'),
                'business_summary': info.get('longBusinessSummary'),
                'employees': info.get('fullTimeEmployees'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'currency': info.get('currency')
            }
            
        except Exception as e:
            logger.error(f"Sync company info fetch failed for {symbol}: {e}")
            return None
    
    async def start_price_stream(self, symbols: List[str], interval: int = 30) -> None:
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
                        bid=ticker.get('bid'),
                        ask=ticker.get('ask')
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
    
    def get_forex_symbols(self) -> List[str]:
        """Get list of major forex pairs"""
        return [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'EURGBP=X',
            'EURJPY=X', 'GBPJPY=X', 'CHFJPY=X', 'EURCHF=X',
            'AUDJPY=X', 'GBPCHF=X', 'NZDJPY=X', 'CADCHF=X'
        ]
    
    def get_major_stocks(self) -> List[str]:
        """Get list of major stock symbols"""
        return [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
            # Consumer
            'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE',
            # Industrial
            'BA', 'CAT', 'GE', 'MMM',
            # Energy
            'XOM', 'CVX', 'COP',
            # Indices
            'SPY', 'QQQ', 'IWM', 'DIA'
        ]
    
    def get_crypto_symbols(self) -> List[str]:
        """Get list of crypto symbols available on yfinance"""
        return [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'SHIB-USD',
            'MATIC-USD', 'LTC-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD'
        ]
    
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Search for symbols matching query"""
        try:
            # This is a simplified search - yfinance doesn't have a built-in search
            # In production, you might want to use a dedicated symbol search API
            
            all_symbols = (
                self.get_major_stocks() + 
                self.get_forex_symbols() + 
                self.get_crypto_symbols()
            )
            
            matching_symbols = [
                symbol for symbol in all_symbols 
                if query.upper() in symbol.upper()
            ]
            
            results = []
            for symbol in matching_symbols[:10]:  # Limit to 10 results
                info = await self.get_ticker(symbol)
                if info:
                    results.append({
                        'symbol': symbol,
                        'name': symbol,  # yfinance doesn't provide names in ticker
                        'price': info['price'],
                        'change': info.get('change'),
                        'percentage': info.get('percentage')
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search symbols for '{query}': {e}")
            return []
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        risk_amount: float, 
        entry_price: float, 
        stop_loss: float
    ) -> float:
        """Calculate position size based on risk"""
        try:
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share == 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / risk_per_share
            
            # For stocks, round to whole shares
            if not symbol.endswith('=X') and not symbol.endswith('-USD'):
                position_size = int(position_size)
            
            return max(position_size, 0)
            
        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            return 0.0
    
    async def close(self) -> None:
        """Close the connector"""
        await self.stop_price_stream()
        self.executor.shutdown(wait=True)
        self._cache.clear()
        self._cache_expiry.clear()
        logger.info("Closed yfinance connector")


# Global connector instance
_yf_connector: Optional[YFinanceConnector] = None


async def get_yf_connector() -> YFinanceConnector:
    """Get the global yfinance connector instance"""
    global _yf_connector
    
    if _yf_connector is None:
        _yf_connector = YFinanceConnector()
    
    return _yf_connector


async def close_yf_connector() -> None:
    """Close the global yfinance connector"""
    global _yf_connector
    
    if _yf_connector:
        await _yf_connector.close()
        _yf_connector = None