"""
Message Bus System for Trading Intelligence Hub
Supports Redis pub/sub with in-memory fallback
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import redis.asyncio as redis
from redis.asyncio import Redis

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Message:
    """Base message structure"""
    type: str
    payload: Dict[str, Any]
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass(kw_only=True)
class TradeSignal(Message):
    """Trade signal message"""
    symbol: str = ""
    action: str = "BUY"  # BUY, SELL, CLOSE
    quantity: float = 0.0
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: Optional[str] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if self.type != 'trade_signal':
            self.type = 'trade_signal'


@dataclass(kw_only=True)
class MarketData(Message):
    """Market data message"""
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    def __post_init__(self):
        if self.type != 'market_data':
            self.type = 'market_data'


@dataclass(kw_only=True)
class AnalysisResult(Message):
    """Analysis result message"""
    symbol: str = ""
    analysis_type: str = ""
    result: Dict[str, Any] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.type != 'analysis_result':
            self.type = 'analysis_result'
        if self.result is None:
            self.result = {}


class MessageBus(ABC):
    """Abstract message bus interface"""
    
    @abstractmethod
    async def publish(self, channel: str, message: Union[Message, Dict[str, Any]]) -> None:
        """Publish a message to a channel"""
        pass
    
    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to a channel with callback"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the message bus"""
        pass


class RedisMessageBus(MessageBus):
    """Redis-based message bus implementation"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.pubsub = None
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def connect(self) -> None:
        """Connect to Redis"""
        try:
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def publish(self, channel: str, message: Union[Message, Dict[str, Any]]) -> None:
        """Publish a message to Redis channel"""
        if not self.redis:
            await self.connect()
        
        try:
            if isinstance(message, Message):
                data = message.to_json()
            else:
                data = json.dumps(message)
            
            await self.redis.publish(channel, data)
            logger.debug(f"Published message to channel {channel}")
            
        except Exception as e:
            logger.error(f"Failed to publish message to {channel}: {e}")
            raise
    
    async def subscribe(self, channel: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to Redis channel"""
        if not self.redis:
            await self.connect()
        
        self.subscribers[channel].append(callback)
        
        if not self._running:
            await self._start_subscriber()
    
    async def _start_subscriber(self) -> None:
        """Start the Redis subscriber"""
        if self._running:
            return
        
        self._running = True
        self.pubsub = self.redis.pubsub()
        
        # Subscribe to all channels
        for channel in self.subscribers.keys():
            await self.pubsub.subscribe(channel)
        
        # Start listening task
        task = asyncio.create_task(self._listen())
        self._tasks.append(task)
    
    async def _listen(self) -> None:
        """Listen for Redis messages"""
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel'].decode('utf-8')
                    data = message['data'].decode('utf-8')
                    
                    try:
                        # Try to parse as Message first
                        msg = Message.from_json(data)
                    except:
                        # Fallback to generic dict
                        msg_data = json.loads(data)
                        msg = Message(
                            type='generic',
                            payload=msg_data,
                            timestamp=datetime.now(),
                            source='unknown'
                        )
                    
                    # Call all callbacks for this channel
                    for callback in self.subscribers.get(channel, []):
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(msg)
                            else:
                                callback(msg)
                        except Exception as e:
                            logger.error(f"Error in callback for channel {channel}: {e}")
        
        except Exception as e:
            logger.error(f"Error in Redis listener: {e}")
        finally:
            self._running = False
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel"""
        if channel in self.subscribers:
            del self.subscribers[channel]
        
        if self.pubsub:
            await self.pubsub.unsubscribe(channel)
    
    async def close(self) -> None:
        """Close Redis connection"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        if self.pubsub:
            await self.pubsub.close()
        
        if self.redis:
            await self.redis.close()


class InMemoryMessageBus(MessageBus):
    """In-memory message bus for development/testing"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: Dict[str, deque] = defaultdict(deque)
        self._running = False
    
    async def publish(self, channel: str, message: Union[Message, Dict[str, Any]]) -> None:
        """Publish message to in-memory queue"""
        if isinstance(message, dict):
            msg = Message(
                type=message.get('type', 'generic'),
                payload=message,
                timestamp=datetime.now(),
                source='in_memory'
            )
        else:
            msg = message
        
        # Add to queue
        self.message_queue[channel].append(msg)
        
        # Immediately call callbacks
        for callback in self.subscribers.get(channel, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(msg)
                else:
                    callback(msg)
            except Exception as e:
                logger.error(f"Error in callback for channel {channel}: {e}")
    
    async def subscribe(self, channel: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to channel"""
        self.subscribers[channel].append(callback)
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel"""
        if channel in self.subscribers:
            del self.subscribers[channel]
    
    async def close(self) -> None:
        """Close in-memory bus"""
        self.subscribers.clear()
        self.message_queue.clear()


class MessageBusFactory:
    """Factory for creating message bus instances"""
    
    @staticmethod
    async def create() -> MessageBus:
        """Create appropriate message bus based on configuration"""
        config = get_config()
        
        try:
            # Try Redis first
            bus = RedisMessageBus(config.redis.url)
            await bus.connect()
            logger.info("Using Redis message bus")
            return bus
        
        except Exception as e:
            logger.warning(f"Redis unavailable, falling back to in-memory bus: {e}")
            return InMemoryMessageBus()


# Global message bus instance
_message_bus: Optional[MessageBus] = None


async def get_message_bus() -> MessageBus:
    """Get the global message bus instance"""
    global _message_bus
    
    if _message_bus is None:
        _message_bus = await MessageBusFactory.create()
    
    return _message_bus


async def publish_trade_signal(
    symbol: str,
    action: str,
    quantity: float,
    price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    strategy: Optional[str] = None,
    confidence: Optional[float] = None
) -> None:
    """Publish a trade signal"""
    config = get_config()
    bus = await get_message_bus()
    
    signal = TradeSignal(
        type='trade_signal',
        payload={},
        timestamp=datetime.now(),
        source='trading_hub',
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy=strategy,
        confidence=confidence
    )
    
    await bus.publish(config.redis.channel_trades, signal)


async def publish_market_data(
    symbol: str,
    price: float,
    volume: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None
) -> None:
    """Publish market data"""
    bus = await get_message_bus()
    
    data = MarketData(
        type='market_data',
        payload={},
        timestamp=datetime.now(),
        source='market_feed',
        symbol=symbol,
        price=price,
        volume=volume,
        bid=bid,
        ask=ask
    )
    
    await bus.publish('market.data', data)


async def publish_analysis_result(
    symbol: str,
    analysis_type: str,
    result: Dict[str, Any],
    confidence: float
) -> None:
    """Publish analysis result"""
    bus = await get_message_bus()
    
    analysis = AnalysisResult(
        type='analysis_result',
        payload={},
        timestamp=datetime.now(),
        source='analysis_agent',
        symbol=symbol,
        analysis_type=analysis_type,
        result=result,
        confidence=confidence
    )
    
    await bus.publish('analysis.results', analysis)


async def close_message_bus() -> None:
    """Close the global message bus"""
    global _message_bus
    
    if _message_bus:
        await _message_bus.close()
        _message_bus = None