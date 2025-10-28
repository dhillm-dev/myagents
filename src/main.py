"""
Main Orchestrator - Trading Intelligence Hub
Runs FastAPI server + background analysis loop
"""
import asyncio
import logging
import signal
import sys
from typing import List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .config import get_config, setup_logging
from .bus import get_message_bus, close_message_bus
from .agents.chart_agent import get_chart_agent
from .agents.sentiment_agent import get_sentiment_agent
from .agents.swarm_agent import get_swarm_agent
from .agents.strategy_agent import get_strategy_agent
from .agents.risk_agent import get_risk_agent
from .agents.execution_agent import get_execution_agent

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Main orchestrator for the trading system"""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.analysis_task = None
        self.watchlist = self._load_watchlist()
        self.analysis_interval = 60  # seconds
        self.last_analysis = {}
        
        # Performance tracking
        self.stats = {
            'analysis_cycles': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'start_time': None
        }
    
    def _load_watchlist(self) -> List[str]:
        """Load trading watchlist from config"""
        try:
            # Default watchlist - can be configured via environment
            default_watchlist = [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT',  # Crypto
                'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN',     # Stocks
                'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'       # Forex
            ]
            
            # Get from config if available
            watchlist_str = getattr(self.config.trading, 'watchlist', '')
            if watchlist_str:
                return [s.strip().upper() for s in watchlist_str.split(',')]
            
            return default_watchlist
            
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'AAPL', 'TSLA']
    
    async def start(self):
        """Start the orchestrator"""
        try:
            logger.info("Starting Trading Intelligence Hub Orchestrator...")
            self.running = True
            self.stats['start_time'] = datetime.now()
            
            # Initialize components
            await self._initialize_components()
            
            # Start background analysis loop
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            
            logger.info(f"Orchestrator started with {len(self.watchlist)} symbols")
            logger.info(f"Watchlist: {', '.join(self.watchlist)}")
            
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            raise
    
    async def stop(self):
        """Stop the orchestrator"""
        try:
            logger.info("Stopping Trading Intelligence Hub Orchestrator...")
            self.running = False
            
            # Cancel analysis task
            if self.analysis_task and not self.analysis_task.done():
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            # Close components
            await self._cleanup_components()
            
            # Log final stats
            self._log_final_stats()
            
            logger.info("Orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
    
    async def _initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Initialize message bus
            bus = await get_message_bus()
            logger.info("Message bus initialized")
            
            # Initialize agents
            chart_agent = await get_chart_agent()
            sentiment_agent = await get_sentiment_agent()
            swarm_agent = await get_swarm_agent()
            strategy_agent = await get_strategy_agent()
            risk_agent = await get_risk_agent()
            execution_agent = await get_execution_agent()
            
            logger.info("All agents initialized")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    async def _cleanup_components(self):
        """Cleanup system components"""
        try:
            logger.info("Cleaning up system components...")
            
            # Close message bus
            await close_message_bus()
            
            logger.info("Components cleaned up")
            
        except Exception as e:
            logger.error(f"Component cleanup failed: {e}")
    
    async def _analysis_loop(self):
        """Main analysis loop"""
        logger.info("Starting analysis loop...")
        
        while self.running:
            try:
                cycle_start = datetime.now()
                
                # Run analysis cycle
                await self._run_analysis_cycle()
                
                # Update stats
                self.stats['analysis_cycles'] += 1
                
                # Calculate sleep time
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, self.analysis_interval - cycle_duration)
                
                logger.info(f"Analysis cycle completed in {cycle_duration:.2f}s, sleeping for {sleep_time:.2f}s")
                
                # Sleep until next cycle
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info("Analysis loop cancelled")
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                self.stats['errors'] += 1
                
                # Sleep before retrying
                await asyncio.sleep(30)
    
    async def _run_analysis_cycle(self):
        """Run a single analysis cycle"""
        try:
            logger.info(f"Running analysis cycle for {len(self.watchlist)} symbols...")
            
            # Get agents
            strategy_agent = await get_strategy_agent()
            risk_agent = await get_risk_agent()
            
            # Check risk status first
            risk_summary = await risk_agent.get_portfolio_summary()
            if risk_summary.get('trading_halted', False):
                logger.warning("Trading halted due to risk limits - skipping signal generation")
                return
            
            # Analyze symbols in batches to avoid overwhelming APIs
            batch_size = 5
            signals_generated = 0
            trades_executed = 0
            
            for i in range(0, len(self.watchlist), batch_size):
                batch = self.watchlist[i:i + batch_size]
                
                # Process batch
                batch_results = await self._process_symbol_batch(batch, strategy_agent)
                
                # Count results
                for result in batch_results:
                    if result.get('signal_generated'):
                        signals_generated += 1
                    if result.get('trade_executed'):
                        trades_executed += 1
                
                # Small delay between batches
                if i + batch_size < len(self.watchlist):
                    await asyncio.sleep(2)
            
            # Update stats
            self.stats['signals_generated'] += signals_generated
            self.stats['trades_executed'] += trades_executed
            
            # Log cycle summary
            logger.info(f"Cycle complete: {signals_generated} signals, {trades_executed} trades")
            
            # Update risk metrics
            await self._update_risk_metrics()
            
        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")
            raise
    
    async def _process_symbol_batch(
        self,
        symbols: List[str],
        strategy_agent
    ) -> List[Dict[str, Any]]:
        """Process a batch of symbols"""
        results = []
        
        for symbol in symbols:
            try:
                result = await self._analyze_symbol(symbol, strategy_agent)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'signal_generated': False,
                    'trade_executed': False
                })
        
        return results
    
    async def _analyze_symbol(self, symbol: str, strategy_agent) -> Dict[str, Any]:
        """Analyze a single symbol"""
        try:
            # Check if we need to analyze (avoid too frequent analysis)
            last_analysis_time = self.last_analysis.get(symbol)
            if last_analysis_time:
                time_since_last = datetime.now() - last_analysis_time
                if time_since_last < timedelta(minutes=5):
                    return {
                        'symbol': symbol,
                        'skipped': True,
                        'reason': 'too_recent',
                        'signal_generated': False,
                        'trade_executed': False
                    }
            
            # Generate signal
            signal = await strategy_agent.generate_signal(symbol)
            
            # Update last analysis time
            self.last_analysis[symbol] = datetime.now()
            
            if not signal:
                return {
                    'symbol': symbol,
                    'signal_generated': False,
                    'trade_executed': False,
                    'reason': 'no_signal'
                }
            
            logger.info(f"Signal generated for {symbol}: {signal.action} - {signal.strength.value}")
            
            # Execute signal if auto-execution is enabled
            trade_executed = False
            if self.config.trading.auto_execute:
                try:
                    order_id = await strategy_agent.execute_signal(signal)
                    if order_id:
                        trade_executed = True
                        logger.info(f"Signal executed for {symbol}: Order {order_id}")
                    else:
                        logger.warning(f"Signal execution failed for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Signal execution error for {symbol}: {e}")
            
            return {
                'symbol': symbol,
                'signal_generated': True,
                'trade_executed': trade_executed,
                'signal': {
                    'action': signal.action,
                    'strength': signal.strength.value,
                    'confidence': signal.confidence,
                    'risk_level': signal.risk_level
                }
            }
            
        except Exception as e:
            logger.error(f"Symbol analysis failed for {symbol}: {e}")
            raise
    
    async def _update_risk_metrics(self):
        """Update risk metrics and check limits"""
        try:
            risk_agent = await get_risk_agent()
            
            # Update portfolio metrics
            await risk_agent.update_portfolio_metrics()
            
            # Get current risk status
            risk_summary = await risk_agent.get_portfolio_summary()
            
            # Log risk status
            daily_pnl = risk_summary.get('daily_pnl', 0.0)
            total_pnl = risk_summary.get('total_pnl', 0.0)
            open_positions = risk_summary.get('open_positions', 0)
            
            logger.info(f"Risk Update - Daily P&L: {daily_pnl:.2f}, Total P&L: {total_pnl:.2f}, Positions: {open_positions}")
            
            # Check if trading is halted
            if risk_summary.get('trading_halted', False):
                logger.warning("Trading halted due to risk limits!")
            
        except Exception as e:
            logger.error(f"Risk metrics update failed: {e}")
    
    def _log_final_stats(self):
        """Log final statistics"""
        try:
            if self.stats['start_time']:
                runtime = datetime.now() - self.stats['start_time']
                runtime_hours = runtime.total_seconds() / 3600
                
                logger.info("=== FINAL STATISTICS ===")
                logger.info(f"Runtime: {runtime}")
                logger.info(f"Analysis cycles: {self.stats['analysis_cycles']}")
                logger.info(f"Signals generated: {self.stats['signals_generated']}")
                logger.info(f"Trades executed: {self.stats['trades_executed']}")
                logger.info(f"Errors: {self.stats['errors']}")
                
                if runtime_hours > 0:
                    logger.info(f"Signals per hour: {self.stats['signals_generated'] / runtime_hours:.2f}")
                    logger.info(f"Trades per hour: {self.stats['trades_executed'] / runtime_hours:.2f}")
                
                logger.info("========================")
                
        except Exception as e:
            logger.error(f"Failed to log final stats: {e}")


# Global orchestrator instance
orchestrator = None


async def start_orchestrator():
    """Start the orchestrator"""
    global orchestrator
    
    if orchestrator is None:
        orchestrator = TradingOrchestrator()
    
    await orchestrator.start()


async def stop_orchestrator():
    """Stop the orchestrator"""
    global orchestrator
    
    if orchestrator:
        await orchestrator.stop()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    
    # Create new event loop if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Stop orchestrator
    if orchestrator:
        loop.run_until_complete(stop_orchestrator())
    
    sys.exit(0)


async def main():
    """Main entry point"""
    try:
        # Setup logging
        setup_logging()
        
        # Get config
        config = get_config()
        
        logger.info("Starting Trading Intelligence Hub...")
        logger.info(f"Config loaded: API={config.api.host}:{config.api.port}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start orchestrator
        await start_orchestrator()
        
        # Import and run FastAPI app
        from .app import app
        
        # Run FastAPI server
        server_config = uvicorn.Config(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(server_config)
        
        logger.info(f"Starting FastAPI server on {config.api.host}:{config.api.port}")
        
        # Run server and orchestrator concurrently
        await asyncio.gather(
            server.serve(),
            return_exceptions=True
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise
    finally:
        # Cleanup
        await stop_orchestrator()
        logger.info("Trading Intelligence Hub stopped")


if __name__ == "__main__":
    asyncio.run(main())