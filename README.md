# Trading Intelligence Hub

A production-ready AI trading system with built-in agents, chart analysis, Perplexity Finance integration, risk management, and MT5 trade execution via Redis bridge.

## 🚀 Features

- **AI-Powered Analysis**: Combines technical analysis with Perplexity Finance API for intelligent trading decisions
- **Multi-Agent Architecture**: Chart analysis, sentiment analysis, swarm consensus, risk management, and execution agents
- **Risk Management**: Built-in position sizing, drawdown protection, and risk validation
- **Multi-Market Support**: Crypto (CCXT), stocks/forex (yfinance), and MT5 pairs
- **Real-Time Execution**: Redis-based message bus for instant trade execution via MetaTrader5
- **Production Ready**: Docker deployment, 24/7 operation, comprehensive logging
- **RESTful API**: FastAPI server with health checks, research endpoints, and manual trading

## 📁 Project Structure

```
trading-intelligence-hub/
├── src/
│   ├── main.py                   # Main orchestrator (API + background loop)
│   ├── app.py                    # FastAPI routes (/health, /order, /research)
│   ├── config.py                 # Environment configuration loader
│   ├── bus.py                    # Redis pub/sub or in-memory fallback
│   ├── agents/
│   │   ├── chart_agent.py        # Technical analysis (EMA, RSI, ATR, patterns)
│   │   ├── sentiment_agent.py    # Perplexity Finance API sentiment scoring
│   │   ├── swarm_agent.py        # AI consensus validation
│   │   ├── strategy_agent.py     # Signal combination and generation
│   │   ├── risk_agent.py         # Position sizing and drawdown protection
│   │   └── execution_agent.py    # Trade execution via Redis
│   ├── connectors/
│   │   ├── market_ccxt.py        # Cryptocurrency data feed
│   │   └── market_yf.py          # Stock/forex data feed
│   └── strategies/
│       ├── wf_swarm.py           # Main AI-driven strategy
│       └── weinstein.py          # Weinstein Stage Analysis strategy
├── workers/
│   └── mt5_worker.py             # Windows MT5 listener for trade execution
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── docker-compose.yml            # Multi-service deployment
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- MetaTrader5 terminal (for live trading)
- Redis server (included in docker-compose)

### 1. Clone Repository

```bash
git clone <repository-url>
cd trading-intelligence-hub
```

### 2. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key-here

# Perplexity Finance API
PPLX_FINANCE_API_KEY=your-perplexity-api-key
PPLX_FINANCE_BASE=https://api.perplexity.ai
PPLX_TIMEOUT=30

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# MetaTrader5 Configuration
MT5_LOGIN=your-mt5-login
MT5_PASSWORD=your-mt5-password
MT5_SERVER=your-mt5-server

# Trading Configuration
RISK_PER_TRADE_PCT=1.0
DAILY_MAX_DRAWDOWN_PCT=5.0
MAX_POSITIONS=10
INITIAL_CAPITAL=10000

# Market Data
CCXT_EXCHANGE=binance
CCXT_SANDBOX=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_hub.log
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the System

### Local Development

Run the main orchestrator (includes FastAPI server and background analysis):

```bash
python -m src.main
```

The system will start:
- FastAPI server on `http://localhost:8000`
- Background analysis loop
- Redis message bus (if configured)

### Docker Deployment

Build and run with Docker Compose:

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f trading-hub

# Stop services
docker-compose down
```

### MT5 Worker (Windows Only)

On your Windows machine with MetaTrader5 installed:

```bash
# Install dependencies
pip install MetaTrader5 redis python-dotenv

# Run MT5 worker
python workers/mt5_worker.py
```

The MT5 worker will:
- Connect to MetaTrader5 terminal
- Subscribe to Redis trade orders
- Execute trades automatically
- Send status updates back to the system

## 📡 API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "redis": "connected",
    "agents": "active"
  }
}
```

### Research Tradability

```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/research/tradability/AAPL
```

Response:
```json
{
  "symbol": "AAPL",
  "tradability_score": 0.75,
  "sentiment_score": 0.65,
  "key_drivers": ["Strong earnings", "Market momentum"],
  "recommendation": "BUY",
  "confidence": 0.80,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Manual Trade Order

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key" \
     -d '{
       "symbol": "EURUSD",
       "action": "BUY",
       "quantity": 0.1,
       "order_type": "MARKET"
     }' \
     http://localhost:8000/order
```

Response:
```json
{
  "order_id": "ord_123456789",
  "status": "PENDING",
  "symbol": "EURUSD",
  "action": "BUY",
  "quantity": 0.1,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Generate Trading Signal

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key" \
     -d '{"symbol": "BTCUSDT"}' \
     http://localhost:8000/signal/generate
```

### Portfolio Summary

```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/portfolio/summary
```

## 🧠 AI Agents

### Chart Agent
- **Technical Indicators**: EMA, RSI, MACD, ATR, Bollinger Bands, ADX, CCI
- **Pattern Detection**: Trend breaks, support/resistance, liquidity sweeps, double tops/bottoms
- **Signal Generation**: Entry/exit signals with confidence scores

### Sentiment Agent
- **Perplexity Integration**: Real-time market sentiment analysis
- **Tradability Scoring**: AI-powered tradability assessment
- **Key Drivers**: Identification of market-moving factors

### Swarm Agent
- **AI Consensus**: Multiple AI perspectives (technical, fundamental, risk, momentum, contrarian)
- **Signal Validation**: Consensus-based signal filtering
- **Risk Assessment**: Collective risk evaluation

### Risk Agent
- **Position Sizing**: Kelly criterion and fixed percentage methods
- **Drawdown Protection**: Daily and maximum drawdown limits
- **Trade Validation**: Risk-based trade approval/rejection

### Execution Agent
- **Order Management**: Trade execution with retry logic
- **Status Tracking**: Real-time order status updates
- **Performance Metrics**: Execution statistics and analysis

## 📊 Trading Strategies

### WF Swarm Strategy
AI-driven strategy combining:
- Technical analysis signals
- Sentiment analysis scores
- Swarm consensus validation
- Risk management integration
- Market regime detection

### Weinstein Stage Analysis
Systematic trend-following based on Stan Weinstein's methodology:
- **Stage 1**: Accumulation/Base building (AVOID)
- **Stage 2**: Advancing/Uptrend (BUY)
- **Stage 3**: Distribution/Top (AVOID)
- **Stage 4**: Declining/Downtrend (SELL)

## 🔧 Configuration

### Risk Management

```env
# Risk per trade (1% = 1.0)
RISK_PER_TRADE_PCT=1.0

# Maximum daily drawdown (5% = 5.0)
DAILY_MAX_DRAWDOWN_PCT=5.0

# Maximum concurrent positions
MAX_POSITIONS=10

# Initial trading capital
INITIAL_CAPITAL=10000
```

### Market Data Sources

```env
# Cryptocurrency (CCXT)
CCXT_EXCHANGE=binance
CCXT_SANDBOX=true

# Stocks/Forex (yfinance)
YF_ENABLE_CACHE=true
YF_CACHE_DURATION=300
```

### AI Configuration

```env
# Perplexity Finance API
PPLX_FINANCE_API_KEY=your-key
PPLX_FINANCE_MODEL=sonar-pro
PPLX_TIMEOUT=30

# Analysis intervals
CHART_ANALYSIS_INTERVAL=300
SENTIMENT_ANALYSIS_INTERVAL=600
SWARM_ANALYSIS_INTERVAL=900
```

## 📈 Monitoring & Logging

### Log Files

- **Application Logs**: `logs/trading_hub.log`
- **Trade Logs**: `logs/trades.log`
- **Error Logs**: `logs/errors.log`

### Metrics Endpoints

```bash
# System statistics
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/stats

# Strategy performance
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/strategy/summary
```

## 🔒 Security

### API Security
- All endpoints protected by `X-API-Key` header
- Configurable API key in environment variables
- Request rate limiting and validation

### Data Security
- No secrets logged or exposed
- Secure Redis communication
- Environment-based configuration

### Trading Security
- Risk validation on all trades
- Position size limits
- Drawdown protection
- Emergency stop mechanisms

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   docker-compose ps redis
   
   # Restart Redis
   docker-compose restart redis
   ```

2. **MT5 Worker Not Connecting**
   - Verify MetaTrader5 terminal is running
   - Check MT5 credentials in `.env`
   - Ensure MT5 allows automated trading
   - Verify network connectivity to Redis

3. **Perplexity API Errors**
   - Verify API key is valid
   - Check API rate limits
   - Ensure sufficient API credits

4. **Market Data Issues**
   - Check internet connectivity
   - Verify exchange API status
   - Review symbol formatting

### Debug Mode

Enable debug logging:

```env
LOG_LEVEL=DEBUG
```

Run with verbose output:

```bash
python -m src.main --debug
```

## 📚 Development

### Adding New Strategies

1. Create strategy file in `src/strategies/`
2. Implement strategy interface
3. Register in strategy agent
4. Add configuration parameters

### Adding New Market Connectors

1. Create connector in `src/connectors/`
2. Implement connector interface
3. Add to connector factory
4. Update configuration

### Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**Happy Trading! 🚀📈**