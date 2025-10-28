"""
Chart Analysis Agent with technical indicators and pattern detection
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib

from ..config import get_config
from ..connectors import get_ccxt_connector, get_yf_connector
from ..bus import publish_analysis_result

logger = logging.getLogger(__name__)


class ChartAnalysisAgent:
    """Chart analysis agent with technical indicators"""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)
    
    async def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> Optional[Dict[str, Any]]:
        """Perform comprehensive chart analysis for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if self._is_cached(cache_key):
                return self.analysis_cache[cache_key]
            
            # Get OHLCV data
            df = await self._get_ohlcv_data(symbol, timeframe)
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Perform analysis
            analysis = await self._perform_analysis(df, symbol, timeframe)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            # Publish analysis result
            await publish_analysis_result(
                symbol=symbol,
                analysis_type='chart_analysis',
                result=analysis,
                confidence=analysis.get('overall_confidence', 0.5)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {e}")
            return None
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if analysis is cached and not expired"""
        if cache_key not in self.analysis_cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[cache_key]
    
    async def _get_ohlcv_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get OHLCV data from appropriate connector"""
        try:
            # Determine if it's crypto or traditional asset
            if any(pair in symbol.upper() for pair in ['BTC', 'ETH', 'USDT', 'USDC']):
                # Use CCXT for crypto
                connector = await get_ccxt_connector()
                df = await connector.get_ohlcv(symbol, timeframe, limit=200)
            else:
                # Use yfinance for stocks/forex
                connector = await get_yf_connector()
                interval_map = {
                    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                    '1h': '1h', '4h': '1h', '1d': '1d'
                }
                yf_interval = interval_map.get(timeframe, '1h')
                df = await connector.get_ohlcv(symbol, interval=yf_interval, days=30)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return None
    
    async def _perform_analysis(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            # Ensure we have numpy arrays
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            open_prices = df['open'].values
            
            # Calculate technical indicators
            indicators = await self._calculate_indicators(high, low, close, volume, open_prices)
            
            # Detect patterns
            patterns = await self._detect_patterns(df, indicators)
            
            # Calculate support/resistance levels
            levels = await self._calculate_levels(df)
            
            # Determine trend
            trend = await self._determine_trend(indicators)
            
            # Calculate volatility
            volatility = await self._calculate_volatility(df)
            
            # Generate signals
            signals = await self._generate_signals(indicators, patterns, trend)
            
            # Calculate overall confidence
            confidence = await self._calculate_confidence(indicators, patterns, trend, signals)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'indicators': indicators,
                'patterns': patterns,
                'levels': levels,
                'trend': trend,
                'volatility': volatility,
                'signals': signals,
                'overall_confidence': confidence,
                'current_price': float(close[-1]),
                'recommendation': self._get_recommendation(signals, confidence)
            }
            
        except Exception as e:
            logger.error(f"Failed to perform analysis: {e}")
            return {}
    
    async def _calculate_indicators(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        volume: np.ndarray,
        open_prices: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            config = self.config.chart
            
            # Moving Averages
            ema_fast = talib.EMA(close, timeperiod=config.ema_fast)
            ema_slow = talib.EMA(close, timeperiod=config.ema_slow)
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
            
            # Momentum Indicators
            rsi = talib.RSI(close, timeperiod=config.rsi_period)
            macd, macd_signal, macd_hist = talib.MACD(close)
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            
            # Volatility Indicators
            atr = talib.ATR(high, low, close, timeperiod=config.atr_period)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            
            # Volume Indicators
            obv = talib.OBV(close, volume)
            ad = talib.AD(high, low, close, volume)
            
            # Trend Indicators
            adx = talib.ADX(high, low, close)
            cci = talib.CCI(high, low, close)
            
            # Price Action
            doji = talib.CDLDOJI(open_prices, high, low, close)
            hammer = talib.CDLHAMMER(open_prices, high, low, close)
            engulfing = talib.CDLENGULFING(open_prices, high, low, close)
            
            return {
                'ema_fast': float(ema_fast[-1]) if not np.isnan(ema_fast[-1]) else None,
                'ema_slow': float(ema_slow[-1]) if not np.isnan(ema_slow[-1]) else None,
                'sma_20': float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None,
                'sma_50': float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None,
                'sma_200': float(sma_200[-1]) if not np.isnan(sma_200[-1]) else None,
                'rsi': float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                'macd': float(macd[-1]) if not np.isnan(macd[-1]) else None,
                'macd_signal': float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None,
                'macd_histogram': float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None,
                'stoch_k': float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else None,
                'stoch_d': float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else None,
                'atr': float(atr[-1]) if not np.isnan(atr[-1]) else None,
                'bb_upper': float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None,
                'bb_middle': float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None,
                'bb_lower': float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None,
                'obv': float(obv[-1]) if not np.isnan(obv[-1]) else None,
                'ad': float(ad[-1]) if not np.isnan(ad[-1]) else None,
                'adx': float(adx[-1]) if not np.isnan(adx[-1]) else None,
                'cci': float(cci[-1]) if not np.isnan(cci[-1]) else None,
                'doji': int(doji[-1]) if not np.isnan(doji[-1]) else 0,
                'hammer': int(hammer[-1]) if not np.isnan(hammer[-1]) else 0,
                'engulfing': int(engulfing[-1]) if not np.isnan(engulfing[-1]) else 0,
                'current_price': float(close[-1])
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return {}
    
    async def _detect_patterns(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect chart patterns and structure breaks"""
        try:
            patterns = {
                'trend_break': False,
                'support_break': False,
                'resistance_break': False,
                'liquidity_sweep': False,
                'double_top': False,
                'double_bottom': False,
                'head_shoulders': False,
                'triangle': False
            }
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Trend break detection
            if indicators.get('ema_fast') and indicators.get('ema_slow'):
                if len(close) >= 2:
                    prev_fast = talib.EMA(close[:-1], timeperiod=self.config.chart.ema_fast)[-1]
                    prev_slow = talib.EMA(close[:-1], timeperiod=self.config.chart.ema_slow)[-1]
                    
                    # Check for EMA crossover
                    if (prev_fast <= prev_slow and indicators['ema_fast'] > indicators['ema_slow']) or \
                       (prev_fast >= prev_slow and indicators['ema_fast'] < indicators['ema_slow']):
                        patterns['trend_break'] = True
            
            # Support/Resistance break detection
            if len(df) >= 20:
                recent_highs = high[-20:]
                recent_lows = low[-20:]
                current_price = close[-1]
                
                # Find recent resistance (highest high in last 20 periods)
                resistance = np.max(recent_highs[:-1])  # Exclude current period
                if current_price > resistance * 1.001:  # 0.1% buffer
                    patterns['resistance_break'] = True
                
                # Find recent support (lowest low in last 20 periods)
                support = np.min(recent_lows[:-1])  # Exclude current period
                if current_price < support * 0.999:  # 0.1% buffer
                    patterns['support_break'] = True
            
            # Liquidity sweep detection (simplified)
            if len(df) >= 10:
                recent_data = df.tail(10)
                if (recent_data['low'].iloc[-1] < recent_data['low'].iloc[-5:-1].min() and
                    recent_data['close'].iloc[-1] > recent_data['low'].iloc[-1] * 1.005):
                    patterns['liquidity_sweep'] = True
            
            # Double top/bottom detection (simplified)
            if len(df) >= 50:
                highs = high[-50:]
                lows = low[-50:]
                
                # Find peaks and troughs
                peaks = []
                troughs = []
                
                for i in range(2, len(highs) - 2):
                    if highs[i] > highs[i-1] and highs[i] > highs[i+1] and \
                       highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                        peaks.append((i, highs[i]))
                    
                    if lows[i] < lows[i-1] and lows[i] < lows[i+1] and \
                       lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                        troughs.append((i, lows[i]))
                
                # Check for double top
                if len(peaks) >= 2:
                    last_two_peaks = peaks[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                        patterns['double_top'] = True
                
                # Check for double bottom
                if len(troughs) >= 2:
                    last_two_troughs = troughs[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                        patterns['double_bottom'] = True
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect patterns: {e}")
            return {}
    
    async def _calculate_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate pivot points
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            r1 = 2 * pivot - low[-1]
            s1 = 2 * pivot - high[-1]
            r2 = pivot + (high[-1] - low[-1])
            s2 = pivot - (high[-1] - low[-1])
            
            # Calculate recent support/resistance levels
            lookback = min(50, len(df))
            recent_highs = high[-lookback:]
            recent_lows = low[-lookback:]
            
            # Find significant levels (simplified)
            resistance_levels = []
            support_levels = []
            
            # Use percentile-based levels
            resistance_levels.extend([
                np.percentile(recent_highs, 95),
                np.percentile(recent_highs, 90),
                np.percentile(recent_highs, 85)
            ])
            
            support_levels.extend([
                np.percentile(recent_lows, 5),
                np.percentile(recent_lows, 10),
                np.percentile(recent_lows, 15)
            ])
            
            return {
                'pivot': float(pivot),
                'resistance_1': float(r1),
                'resistance_2': float(r2),
                'support_1': float(s1),
                'support_2': float(s2),
                'resistance_levels': [float(x) for x in resistance_levels],
                'support_levels': [float(x) for x in support_levels],
                'current_price': float(close[-1])
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate levels: {e}")
            return {}
    
    async def _determine_trend(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall trend direction"""
        try:
            trend_signals = []
            
            # EMA trend
            if indicators.get('ema_fast') and indicators.get('ema_slow'):
                if indicators['ema_fast'] > indicators['ema_slow']:
                    trend_signals.append('bullish')
                else:
                    trend_signals.append('bearish')
            
            # SMA trend
            current_price = indicators.get('current_price')
            if current_price and indicators.get('sma_20') and indicators.get('sma_50'):
                if current_price > indicators['sma_20'] > indicators['sma_50']:
                    trend_signals.append('bullish')
                elif current_price < indicators['sma_20'] < indicators['sma_50']:
                    trend_signals.append('bearish')
                else:
                    trend_signals.append('neutral')
            
            # ADX trend strength
            adx = indicators.get('adx')
            trend_strength = 'weak'
            if adx:
                if adx > 25:
                    trend_strength = 'strong'
                elif adx > 20:
                    trend_strength = 'moderate'
            
            # Overall trend
            bullish_count = trend_signals.count('bullish')
            bearish_count = trend_signals.count('bearish')
            
            if bullish_count > bearish_count:
                overall_trend = 'bullish'
            elif bearish_count > bullish_count:
                overall_trend = 'bearish'
            else:
                overall_trend = 'neutral'
            
            return {
                'overall': overall_trend,
                'strength': trend_strength,
                'signals': trend_signals,
                'confidence': max(bullish_count, bearish_count) / len(trend_signals) if trend_signals else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to determine trend: {e}")
            return {'overall': 'neutral', 'strength': 'weak', 'confidence': 0}
    
    async def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility metrics"""
        try:
            close = df['close'].values
            
            # Calculate returns
            returns = np.diff(np.log(close))
            
            # Volatility metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            avg_true_range = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            atr_pct = (avg_true_range / close[-len(avg_true_range):]).mean() * 100
            
            return {
                'annualized_volatility': float(volatility),
                'atr_percentage': float(atr_pct) if not np.isnan(atr_pct) else 0,
                'volatility_regime': 'high' if volatility > 0.3 else 'medium' if volatility > 0.15 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return {'annualized_volatility': 0, 'atr_percentage': 0, 'volatility_regime': 'unknown'}
    
    async def _generate_signals(
        self, 
        indicators: Dict[str, Any], 
        patterns: Dict[str, Any], 
        trend: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on analysis"""
        try:
            signals = {
                'buy': [],
                'sell': [],
                'strength': 0,
                'direction': 'neutral'
            }
            
            # RSI signals
            rsi = indicators.get('rsi')
            if rsi:
                if rsi < 30:
                    signals['buy'].append('rsi_oversold')
                elif rsi > 70:
                    signals['sell'].append('rsi_overbought')
            
            # MACD signals
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            if macd and macd_signal:
                if macd > macd_signal:
                    signals['buy'].append('macd_bullish')
                else:
                    signals['sell'].append('macd_bearish')
            
            # Trend signals
            if trend['overall'] == 'bullish' and trend['confidence'] > 0.6:
                signals['buy'].append('trend_bullish')
            elif trend['overall'] == 'bearish' and trend['confidence'] > 0.6:
                signals['sell'].append('trend_bearish')
            
            # Pattern signals
            if patterns.get('resistance_break'):
                signals['buy'].append('resistance_break')
            if patterns.get('support_break'):
                signals['sell'].append('support_break')
            if patterns.get('double_bottom'):
                signals['buy'].append('double_bottom')
            if patterns.get('double_top'):
                signals['sell'].append('double_top')
            
            # Calculate overall signal strength
            buy_strength = len(signals['buy'])
            sell_strength = len(signals['sell'])
            
            if buy_strength > sell_strength:
                signals['direction'] = 'bullish'
                signals['strength'] = min(buy_strength / 5.0, 1.0)  # Normalize to 0-1
            elif sell_strength > buy_strength:
                signals['direction'] = 'bearish'
                signals['strength'] = min(sell_strength / 5.0, 1.0)
            else:
                signals['direction'] = 'neutral'
                signals['strength'] = 0
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return {'buy': [], 'sell': [], 'strength': 0, 'direction': 'neutral'}
    
    async def _calculate_confidence(
        self, 
        indicators: Dict[str, Any], 
        patterns: Dict[str, Any], 
        trend: Dict[str, Any], 
        signals: Dict[str, Any]
    ) -> float:
        """Calculate overall analysis confidence"""
        try:
            confidence_factors = []
            
            # Trend confidence
            confidence_factors.append(trend.get('confidence', 0) * 0.3)
            
            # Signal strength
            confidence_factors.append(signals.get('strength', 0) * 0.3)
            
            # Indicator reliability
            indicator_count = sum(1 for v in indicators.values() if v is not None)
            indicator_confidence = min(indicator_count / 10.0, 1.0)
            confidence_factors.append(indicator_confidence * 0.2)
            
            # Pattern confirmation
            pattern_count = sum(1 for v in patterns.values() if v)
            pattern_confidence = min(pattern_count / 3.0, 1.0)
            confidence_factors.append(pattern_confidence * 0.2)
            
            return sum(confidence_factors)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _get_recommendation(self, signals: Dict[str, Any], confidence: float) -> str:
        """Get trading recommendation based on signals and confidence"""
        if confidence < 0.4:
            return 'HOLD'
        
        direction = signals.get('direction', 'neutral')
        strength = signals.get('strength', 0)
        
        if direction == 'bullish' and strength > 0.6:
            return 'STRONG_BUY'
        elif direction == 'bullish' and strength > 0.3:
            return 'BUY'
        elif direction == 'bearish' and strength > 0.6:
            return 'STRONG_SELL'
        elif direction == 'bearish' and strength > 0.3:
            return 'SELL'
        else:
            return 'HOLD'
    
    async def get_stop_loss_take_profit(
        self, 
        symbol: str, 
        entry_price: float, 
        direction: str
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            # Get ATR for the symbol
            analysis = await self.analyze_symbol(symbol)
            if not analysis or 'indicators' not in analysis:
                # Fallback to percentage-based levels
                if direction.upper() == 'BUY':
                    stop_loss = entry_price * 0.98  # 2% stop loss
                    take_profit = entry_price * 1.04  # 4% take profit
                else:
                    stop_loss = entry_price * 1.02
                    take_profit = entry_price * 0.96
                
                return stop_loss, take_profit
            
            atr = analysis['indicators'].get('atr')
            if not atr:
                # Fallback
                if direction.upper() == 'BUY':
                    return entry_price * 0.98, entry_price * 1.04
                else:
                    return entry_price * 1.02, entry_price * 0.96
            
            # Use ATR-based levels
            config = self.config.chart
            
            if direction.upper() == 'BUY':
                stop_loss = entry_price - (atr * config.atr_sl_multiplier)
                take_profit = entry_price + (atr * config.atr_tp_multiplier)
            else:
                stop_loss = entry_price + (atr * config.atr_sl_multiplier)
                take_profit = entry_price - (atr * config.atr_tp_multiplier)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Failed to calculate SL/TP for {symbol}: {e}")
            # Fallback levels
            if direction.upper() == 'BUY':
                return entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price * 1.02, entry_price * 0.96
    
    async def start_continuous_analysis(self, symbols: List[str], interval: int = 300) -> None:
        """Start continuous analysis for multiple symbols"""
        self.running = True
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._analysis_worker(symbol, interval))
            tasks.append(task)
        
        logger.info(f"Started continuous analysis for {len(symbols)} symbols")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in continuous analysis: {e}")
    
    async def _analysis_worker(self, symbol: str, interval: int) -> None:
        """Worker for continuous analysis"""
        while self.running:
            try:
                await self.analyze_symbol(symbol)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                await asyncio.sleep(interval * 2)
    
    async def stop_continuous_analysis(self) -> None:
        """Stop continuous analysis"""
        self.running = False
        logger.info("Stopped continuous analysis")


# Global chart agent instance
_chart_agent: Optional[ChartAnalysisAgent] = None


async def get_chart_agent() -> ChartAnalysisAgent:
    """Get the global chart analysis agent"""
    global _chart_agent
    
    if _chart_agent is None:
        _chart_agent = ChartAnalysisAgent()
    
    return _chart_agent