"""
Trae Orchestrator
Async pub/sub coordination across agents with confidence weights and status.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..bus import get_message_bus, Message

logger = logging.getLogger(__name__)


class TraeOrchestrator:
    """Coordinates agents via async pub/sub and tracks dynamic weights."""

    CHANNELS = {
        'flow': '/flow',
        'alpha': '/alpha',
        'risk': '/risk',
        'sentiment': '/sentiment',
        'swarm': '/swarm',
        'exec': '/exec',
    }

    def __init__(self):
        self.bus = None
        self.running = False
        self._tasks: List[asyncio.Task] = []
        # Confidence weights per agent
        self.weights: Dict[str, float] = {
            'AlphaHunter': 0.5,
            'FlowGuard': 0.8,
            'RiskAgent': 0.9,
            'SwarmAgent': 0.7,
            'SentimentAgent': 0.6,
            'WhaleAgent': 0.65,
            'UniverseAgent': 0.6,
            'ChartAgent': 0.6,
        }
        self.metrics: Dict[str, Any] = {
            'rolling_sharpe': 0.0,
            'hit_rate': 0.0,
            'drawdown': 0.0,
            'last_updated': None,
        }

    async def start(self):
        if self.running:
            return
        self.bus = await get_message_bus()
        self.running = True
        # Subscribe basic listeners
        for name, channel in self.CHANNELS.items():
            task = asyncio.create_task(self._subscribe_channel(channel))
            self._tasks.append(task)
        logger.info("Trae Orchestrator started")

    async def stop(self):
        self.running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        logger.info("Trae Orchestrator stopped")

    async def _subscribe_channel(self, channel: str):
        async def handler(msg: Message):
            try:
                # Update simple heartbeat metrics
                self.metrics['last_updated'] = datetime.now().isoformat()
            except Exception as e:
                logger.error(f"Orchestrator handler error on {channel}: {e}")

        try:
            await self.bus.subscribe(channel, handler)
            while self.running:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Failed to subscribe {channel}: {e}")

    def adjust_weight(self, agent: str, delta: float):
        w = self.weights.get(agent, 0.5)
        w = max(0.0, min(1.0, w + delta))
        self.weights[agent] = w

    def get_status(self) -> Dict[str, Any]:
        return {
            'active_agents': list(self.weights.keys()),
            'confidence': self.weights,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat(),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            'key_signals': [],  # populated by agents in future iterations
            'pnl': {
                'daily': 0.0,
                'total': 0.0,
            },
            'confidence': self.weights,
            'timestamp': datetime.now().isoformat(),
        }

    async def reset(self):
        # Reset weights to neutral baseline
        for k in self.weights.keys():
            self.weights[k] = 0.5
        self.metrics['rolling_sharpe'] = 0.0
        self.metrics['hit_rate'] = 0.0
        self.metrics['drawdown'] = 0.0
        self.metrics['last_updated'] = datetime.now().isoformat()


# Global orchestrator instance
_trae_orchestrator: Optional[TraeOrchestrator] = None


async def get_trae_orchestrator() -> TraeOrchestrator:
    global _trae_orchestrator
    if _trae_orchestrator is None:
        _trae_orchestrator = TraeOrchestrator()
        await _trae_orchestrator.start()
    return _trae_orchestrator