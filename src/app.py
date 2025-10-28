"""
FastAPI Server - Trading Intelligence Hub API
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import os
from .config import get_config
from .bus import get_message_bus, close_message_bus
from .routers import brain as brain_router
from .connectors.market_yf import YFinanceConnector, get_yf_connector
from .agents.risk_agent import get_risk_agent
from .security import verify_api_key

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    timestamp: str
    version: str = "1.0.0"
    components: Dict[str, str] = {}


class TradabilityResponse(BaseModel):
    """Tradability research response"""
    symbol: str
    sentiment_score: float = Field(..., ge=0, le=10, description="Sentiment score 0-10")
    tradability_score: float = Field(..., ge=0, le=10, description="Tradability score 0-10")
    key_drivers: List[str] = Field(default_factory=list)
    analysis_summary: str = ""
    risk_factors: List[str] = Field(default_factory=list)
    timestamp: str


class OrderRequest(BaseModel):
    """Manual order request"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT, AAPL)")
    action: str = Field(..., description="BUY or SELL")
    quantity: Optional[float] = Field(None, description="Quantity (auto-calculated if not provided)")
    order_type: str = Field(default="MARKET", description="Order type")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    force_execute: bool = Field(default=False, description="Force execution bypassing some validations")


class OrderResponse(BaseModel):
    """Order execution response"""
    success: bool
    order_id: Optional[str] = None
    message: str
    symbol: str
    action: str
    quantity: float
    estimated_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: str


class SignalRequest(BaseModel):
    """Signal generation request"""
    symbol: str = Field(..., description="Trading symbol")
    force_refresh: bool = Field(default=False, description="Force refresh analysis")


class SignalResponse(BaseModel):
    """Signal generation response"""
    symbol: str
    signal: Optional[str] = None  # BUY, SELL, HOLD
    strength: Optional[str] = None
    confidence: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    reasoning: str = ""
    risk_level: Optional[str] = None
    timestamp: str



class PortfolioResponse(BaseModel):
    """Portfolio summary response"""
    total_equity: float
    available_balance: float
    used_margin: float
    open_positions: int
    daily_pnl: float
    total_pnl: float
    risk_metrics: Dict[str, Any] = {}
    timestamp: str


# Flow / Risk / Screening response models (placed before endpoint usage)
class FlowSnapshotResponse(BaseModel):
    symbol: str
    spread: Optional[float] = None
    imbalance: Optional[float] = None
    volatility: Optional[float] = None
    score: float
    timestamp: str


class RiskSizeRequest(BaseModel):
    symbol: str
    entry_price: float
    stop_loss: float
    direction: str = Field(..., description="BUY or SELL")
    risk_multiplier: float = Field(1.0, description="Adjust base risk percentage")


class RiskSizeResponse(BaseModel):
    symbol: str
    quantity: float
    risk_amount: float
    risk_percentage: float
    stop_loss: float
    take_profit: float
    max_loss: float
    position_value: float
    leverage: float
    timestamp: str


class MultibaggerScreenItem(BaseModel):
    symbol: str
    price: float
    rs_trend: float
    vol_contraction: float
    higher_low_score: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None


class MultibaggerScreenResponse(BaseModel):
    items: List[MultibaggerScreenItem]
    count: int
    timestamp: str


# Global app state
app_state = {
    "startup_time": None,
    "request_count": 0,
    "last_health_check": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Trading Intelligence Hub API...")
    app_state["startup_time"] = datetime.now()
    
    # Initialize message bus
    try:
        bus = await get_message_bus()
        logger.info("Message bus initialized")
    except Exception as e:
        logger.error(f"Failed to initialize message bus: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Trading Intelligence Hub API...")
    try:
        await close_message_bus()
        logger.info("Message bus closed")
    except Exception as e:
        logger.error(f"Error closing message bus: {e}")


# Create FastAPI app
app = FastAPI(
    title="Trading Intelligence Hub",
    description="AI-powered trading system with multi-agent analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboards
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include brain router
app.include_router(brain_router.router)


# Security dependency
# Moved to src/security.py to avoid circular imports


# Middleware to count requests
@app.middleware("http")
async def count_requests(request, call_next):
    """Count API requests"""
    app_state["request_count"] += 1
    response = await call_next(request)
    return response


# Health endpoint (no auth required)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        app_state["last_health_check"] = datetime.now()
        
        # Check component health
        components = {}
        
        # Check message bus
        try:
            bus = await get_message_bus()
            components["message_bus"] = "healthy"
        except Exception as e:
            components["message_bus"] = f"error: {str(e)}"
        
        # Check agents
        try:
            # Lazy import to avoid heavy deps during simple health checks
            from .agents.sentiment_agent import get_sentiment_agent
            sentiment_agent = await get_sentiment_agent()
            components["sentiment_agent"] = "healthy"
        except Exception as e:
            components["sentiment_agent"] = f"error: {str(e)}"
        
        try:
            # Lazy import to avoid importing heavy analysis stack on startup
            from .agents.strategy_agent import get_strategy_agent
            strategy_agent = await get_strategy_agent()
            components["strategy_agent"] = "healthy"
        except Exception as e:
            components["strategy_agent"] = f"error: {str(e)}"
        
        try:
            # Lazy import risk agent
            from .agents.risk_agent import get_risk_agent
            risk_agent = await get_risk_agent()
            components["risk_agent"] = "healthy"
        except Exception as e:
            components["risk_agent"] = f"error: {str(e)}"
        
        return HealthResponse(
            timestamp=datetime.now().isoformat(),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Research endpoint
@app.get("/research/tradability/{symbol}", response_model=TradabilityResponse)
async def get_tradability(
    symbol: str,
    _: bool = Depends(verify_api_key)
):
    """Get tradability analysis for a symbol"""
    try:
        logger.info(f"Tradability research requested for {symbol}")
        
        # Get sentiment agent (lazy import)
        from .agents.sentiment_agent import get_sentiment_agent
        sentiment_agent = await get_sentiment_agent()
        
        # Analyze sentiment and tradability
        analysis = await sentiment_agent.analyze_sentiment(symbol)
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to analyze symbol {symbol}"
            )
        
        return TradabilityResponse(
            symbol=symbol,
            sentiment_score=analysis.get('sentiment_score', 5.0),
            tradability_score=analysis.get('tradability_score', 5.0),
            key_drivers=analysis.get('key_drivers', []),
            analysis_summary=analysis.get('analysis_summary', ''),
            risk_factors=analysis.get('risk_factors', []),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tradability analysis failed for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# Order endpoint
@app.post("/order", response_model=OrderResponse)
async def execute_order(
    order: OrderRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    """Execute a manual trading order"""
    try:
        logger.info(f"Manual order requested: {order.symbol} {order.action}")
        
        # Get agents
        strategy_agent = await get_strategy_agent()
        risk_agent = await get_risk_agent()
        from .agents.execution_agent import get_execution_agent
        execution_agent = await get_execution_agent()
        
        # Generate signal if quantity not provided
        if order.quantity is None:
            signal = await strategy_agent.generate_signal(order.symbol, force_refresh=True)
            if not signal:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to generate trading signal for position sizing"
                )
            
            quantity = signal.quantity
            estimated_price = signal.entry_price
            stop_loss = order.stop_loss or signal.stop_loss
            take_profit = order.take_profit or signal.take_profit
        else:
            quantity = order.quantity
            # Get current market price (simplified)
            estimated_price = 0.0  # Would get from market data connector
            stop_loss = order.stop_loss
            take_profit = order.take_profit
        
        # Validate trade with risk management
        if not order.force_execute:
            trade_data = {
                'symbol': order.symbol,
                'quantity': quantity,
                'entry_price': estimated_price,
                'side': order.action,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            is_valid, validation_message = await risk_agent.validate_trade(trade_data)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Risk validation failed: {validation_message}"
                )
        
        # Execute the order
        order_id = await execution_agent.execute_trade(
            symbol=order.symbol,
            side=order.action,
            quantity=quantity,
            order_type=order.order_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'manual_order': True,
                'force_execute': order.force_execute
            }
        )
        
        if not order_id:
            raise HTTPException(
                status_code=500,
                detail="Order execution failed"
            )
        
        return OrderResponse(
            success=True,
            order_id=order_id,
            message="Order executed successfully",
            symbol=order.symbol,
            action=order.action,
            quantity=quantity,
            estimated_price=estimated_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Order execution failed: {str(e)}"
        )


# Signal generation endpoint
@app.post("/signal", response_model=SignalResponse)
async def generate_signal(
    request: SignalRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate trading signal for a symbol"""
    try:
        logger.info(f"Signal generation requested for {request.symbol}")
        
        # Get strategy agent
        from .agents.strategy_agent import get_strategy_agent
        strategy_agent = await get_strategy_agent()
        
        # Generate signal
        signal = await strategy_agent.generate_signal(
            request.symbol,
            force_refresh=request.force_refresh
        )
        
        if not signal:
            return SignalResponse(
                symbol=request.symbol,
                reasoning="No signal generated - insufficient confidence or HOLD signal",
                timestamp=datetime.now().isoformat()
            )
        
        return SignalResponse(
            symbol=signal.symbol,
            signal=signal.action,
            strength=signal.strength.value,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            quantity=signal.quantity,
            reasoning=signal.reasoning,
            risk_level=signal.risk_level,
            timestamp=signal.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Signal generation failed for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Signal generation failed: {str(e)}"
        )


# FlowGuard snapshot endpoint
@app.get("/flow/snapshot", response_model=FlowSnapshotResponse)
async def flow_snapshot(symbol: str = "BTC-USD", _: bool = Depends(verify_api_key)):
    """Instant FlowGuard-like snapshot using market data proxy"""
    try:
        yf = await get_yf_connector()
        ticker = await yf.get_ticker(symbol)
        ohlc = await yf.get_ohlcv(symbol, interval="1m", days=1)
        spread = None
        imbalance = None
        volatility = None
        score = 0.0

        if ticker:
            bid = ticker.get("bid") or 0
            ask = ticker.get("ask") or 0
            if bid and ask and ask > bid:
                spread = (ask - bid) / ((ask + bid) / 2)

        if ohlc is not None and not ohlc.empty:
            last = float(ohlc["close"].iloc[-1])
            vol_series = ohlc["close"].pct_change().rolling(60).std()
            volatility = float(vol_series.dropna().iloc[-1]) if not vol_series.dropna().empty else None
            # Simple imbalance proxy: last close vs 10-min mean
            mean_10 = float(ohlc["close"].tail(10).mean())
            imbalance = (last - mean_10) / mean_10 if mean_10 else None

        # Scoring: combine normalized metrics
        parts = []
        if spread is not None:
            parts.append(max(0.0, 1.0 - min(spread * 100, 5) / 5))  # tighter spreads score higher
        if volatility is not None:
            parts.append(max(0.0, 1.0 - min(volatility * 100, 5) / 5))  # lower vol scores higher
        if imbalance is not None:
            parts.append(max(0.0, 1.0 - min(abs(imbalance) * 100, 5) / 5))
        score = round(sum(parts) / len(parts), 3) if parts else 0.0

        return FlowSnapshotResponse(
            symbol=symbol,
            spread=spread,
            imbalance=imbalance,
            volatility=volatility,
            score=score,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Flow snapshot failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Flow snapshot failed: {e}")


# Risk blended position sizing endpoint
@app.post("/risk/size", response_model=RiskSizeResponse)
async def risk_size(request: RiskSizeRequest, _: bool = Depends(verify_api_key)):
    """Blended position sizing via Risk Agent (ATR + Kelly-Lite)."""
    try:
        risk_agent = await get_risk_agent()
        ps = await risk_agent.calculate_position_size(
            symbol=request.symbol,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            direction=request.direction,
            risk_multiplier=request.risk_multiplier
        )
        if not ps:
            raise HTTPException(status_code=400, detail="Unable to calculate position size")
        return RiskSizeResponse(
            symbol=ps.symbol,
            quantity=ps.quantity,
            risk_amount=ps.risk_amount,
            risk_percentage=ps.risk_percentage,
            stop_loss=ps.stop_loss,
            take_profit=ps.take_profit,
            max_loss=ps.max_loss,
            position_value=ps.position_value,
            leverage=ps.leverage,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk size failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk size failed: {e}")


# Universe multibagger screen endpoint
@app.get("/universe/screen/multibagger", response_model=MultibaggerScreenResponse)
async def screen_multibagger(
    symbols: str = "AAPL,TSLA,NVDA,AMD,PLTR,SOFI,ROKU,SHOP,ABNB,COIN",
    min_price: float = 1.0,
    max_price: float = 45.0,
    _: bool = Depends(verify_api_key)
):
    """Filter €1–€45 high-quality stocks with RS, volatility contraction, higher-lows."""
    try:
        yf = await get_yf_connector()
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        items: List[MultibaggerScreenItem] = []

        for sym in symbol_list:
            info = await yf.get_ticker(sym)
            if not info or not info.get('price'):
                continue
            price = float(info['price'])
            if price < min_price or price > max_price:
                continue

            df = await yf.get_ohlcv(sym, interval="1d", days=180)
            if df is None or df.empty:
                continue

            close = df['close']
            ma50 = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()
            rs_trend = 0.0
            if not ma50.dropna().empty and not ma200.dropna().empty:
                rs_trend = float((ma50.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1]) if ma200.iloc[-1] else 0.0

            vol20 = close.pct_change().rolling(20).std()
            vol60 = close.pct_change().rolling(60).std()
            vol_contraction = 0.0
            if not vol20.dropna().empty and not vol60.dropna().empty:
                v20 = float(vol20.dropna().iloc[-1])
                v60 = float(vol60.dropna().iloc[-1])
                vol_contraction = (v60 - v20) / v60 if v60 else 0.0

            lows = df['low']
            hl_score = 0.0
            if len(lows) >= 60:
                swing_lows = [float(lows.iloc[i]) for i in range(len(lows)-60, len(lows), 15)]
                if len(swing_lows) >= 3:
                    hl_score = 1.0 if swing_lows[0] < swing_lows[1] < swing_lows[2] else 0.0

            items.append(MultibaggerScreenItem(
                symbol=sym,
                price=price,
                rs_trend=round(rs_trend, 4),
                vol_contraction=round(vol_contraction, 4),
                higher_low_score=hl_score,
                market_cap=info.get('market_cap'),
                pe_ratio=info.get('pe_ratio')
            ))

        # Rank by combined score
        items.sort(key=lambda x: (x.rs_trend + x.vol_contraction + x.higher_low_score), reverse=True)
        return MultibaggerScreenResponse(
            items=items,
            count=len(items),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Multibagger screen failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multibagger screen failed: {e}")


# Portfolio summary endpoint
@app.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio_summary(_: bool = Depends(verify_api_key)):
    """Get portfolio summary and risk metrics"""
    try:
        logger.info("Portfolio summary requested")
        
        # Get risk agent
        from .agents.risk_agent import get_risk_agent
        risk_agent = await get_risk_agent()
        
        # Get portfolio summary
        summary = await risk_agent.get_portfolio_summary()
        
        return PortfolioResponse(
            total_equity=summary.get('total_equity', 0.0),
            available_balance=summary.get('available_balance', 0.0),
            used_margin=summary.get('used_margin', 0.0),
            open_positions=summary.get('open_positions', 0),
            daily_pnl=summary.get('daily_pnl', 0.0),
            total_pnl=summary.get('total_pnl', 0.0),
            risk_metrics=summary.get('risk_metrics', {}),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Portfolio summary failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio summary failed: {str(e)}"
        )


# Strategy summary endpoint
@app.get("/strategy/summary")
async def get_strategy_summary(
    symbols: str = "BTCUSDT,ETHUSDT,AAPL,TSLA",
    _: bool = Depends(verify_api_key)
):
    """Get strategy summary for multiple symbols"""
    try:
        logger.info("Strategy summary requested")
        
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Get strategy agent
        from .agents.strategy_agent import get_strategy_agent
        strategy_agent = await get_strategy_agent()
        
        # Get summary
        summary = await strategy_agent.get_strategy_summary(symbol_list)
        
        return summary
        
    except Exception as e:
        logger.error(f"Strategy summary failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Strategy summary failed: {str(e)}"
        )


# System stats endpoint
@app.get("/stats")
async def get_system_stats(_: bool = Depends(verify_api_key)):
    """Get system statistics"""
    try:
        uptime = None
        if app_state["startup_time"]:
            uptime = (datetime.now() - app_state["startup_time"]).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "request_count": app_state["request_count"],
            "last_health_check": app_state["last_health_check"].isoformat() if app_state["last_health_check"] else None,
            "startup_time": app_state["startup_time"].isoformat() if app_state["startup_time"] else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"System stats failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "src.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )