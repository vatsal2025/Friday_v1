from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import ta
import joblib
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# OpenAI integration
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Create the main app without a prefix
app = FastAPI(title="Friday AI Trading System", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# F&O Symbols for Indian Markets
FNO_SYMBOLS = [
    # Large Cap Banking & Financial
    'ICICIBANK', 'HDFCBANK', 'AXISBANK', 'SBIN', 'KOTAKBANK',
    # IT Sector
    'INFY', 'TCS', 'HCLTECH', 'WIPRO', 'TECHM',
    # Pharma & Healthcare
    'SUNPHARMA', 'DRREDDY', 'DIVISLAB', 'CIPLA', 'APOLLOHOSP',
    # Manufacturing & Capital Goods
    'LTTS', 'HAVELLS', 'DIXON', 'TATAELXSI', 'BEL', 'HAL',
    # FMCG & Retail
    'HINDUNILVR', 'ITC', 'BRITANNIA', 'TATACONSUM', 'NESTLEIND',
    # Consumer Services  
    'ZOMATO', 'JUBLFOOD', 'DEVYANI', 'NAUKRI', 'IRCTC',
    # Chemicals & Materials
    'AARTIIND', 'SRF', 'PIIND', 'ASIANPAINT', 'TATACHEM'
]

# Pydantic Models
class TradingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str
    ai_analysis: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeframe: str = "intraday"
    sector: str
    risk_reward_ratio: Optional[float] = None

class TradingSignalCreate(BaseModel):
    symbols: Optional[List[str]] = None

class TradeExecution(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    executed_price: float
    status: str  # EXECUTED, PENDING, CANCELLED
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pnl: Optional[float] = None

class PortfolioMetrics(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class LearningInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trade_id: str
    symbol: str
    mistake_type: str
    lesson_learned: str
    ai_analysis: str
    confidence_adjustment: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# AI Trading Engine Class
class AITradingEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.sector_mapping = {
            'ICICIBANK': 'Banking', 'HDFCBANK': 'Banking', 'AXISBANK': 'Banking', 'SBIN': 'Banking', 'KOTAKBANK': 'Banking',
            'INFY': 'IT', 'TCS': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT',
            'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'DIVISLAB': 'Pharma', 'CIPLA': 'Pharma', 'APOLLOHOSP': 'Pharma',
            'LTTS': 'Manufacturing', 'HAVELLS': 'Manufacturing', 'DIXON': 'Manufacturing', 'TATAELXSI': 'Manufacturing', 'BEL': 'Manufacturing', 'HAL': 'Manufacturing',
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'BRITANNIA': 'FMCG', 'TATACONSUM': 'FMCG', 'NESTLEIND': 'FMCG',
            'ZOMATO': 'Consumer Services', 'JUBLFOOD': 'Consumer Services', 'DEVYANI': 'Consumer Services', 'NAUKRI': 'Consumer Services', 'IRCTC': 'Consumer Services',
            'AARTIIND': 'Chemicals', 'SRF': 'Chemicals', 'PIIND': 'Chemicals', 'ASIANPAINT': 'Chemicals', 'TATACHEM': 'Chemicals'
        }
        self.initialize_models()

    def initialize_models(self):
        """Initialize ML models for each sector"""
        sectors = list(set(self.sector_mapping.values()))
        for sector in sectors:
            self.models[sector] = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            self.scalers[sector] = StandardScaler()

    def get_market_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch market data for Indian stocks with multiple fallbacks"""
        try:
            # Try yfinance first (NSE then BSE)
            for suffix in [".NS", ".BO"]:
                try:
                    ticker = f"{symbol}{suffix}"
                    stock = yf.Ticker(ticker)
                    data = stock.history(period=period)
                    
                    if not data.empty and len(data) > 20:  # Ensure we have enough data
                        logger.info(f"Successfully fetched data for {ticker}")
                        return data
                except Exception as e:
                    logger.warning(f"yfinance failed for {ticker}: {e}")
                    continue
            
            # If yfinance fails, generate realistic demo data
            logger.warning(f"All data sources failed for {symbol}, generating demo data")
            return self.generate_demo_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self.generate_demo_data(symbol)

    def generate_demo_data(self, symbol: str) -> pd.DataFrame:
        """Generate realistic demo market data for testing"""
        try:
            # Base prices for different symbols (realistic NSE prices)
            base_prices = {
                'ICICIBANK': 1200, 'HDFCBANK': 1650, 'AXISBANK': 1100, 'SBIN': 800, 'KOTAKBANK': 1800,
                'INFY': 1850, 'TCS': 4200, 'HCLTECH': 1750, 'WIPRO': 650, 'TECHM': 1650,
                'SUNPHARMA': 1200, 'DRREDDY': 1300, 'DIVISLAB': 5200, 'CIPLA': 1500, 'APOLLOHOSP': 6800,
                'LTTS': 5500, 'HAVELLS': 1650, 'DIXON': 15000, 'TATAELXSI': 7200, 'BEL': 300, 'HAL': 4500,
                'HINDUNILVR': 2650, 'ITC': 480, 'BRITANNIA': 5200, 'TATACONSUM': 950, 'NESTLEIND': 22000,
                'ZOMATO': 280, 'JUBLFOOD': 650, 'DEVYANI': 200, 'NAUKRI': 8500, 'IRCTC': 850,
                'AARTIIND': 550, 'SRF': 2400, 'PIIND': 4800, 'ASIANPAINT': 2950, 'TATACHEM': 1100
            }
            
            base_price = base_prices.get(symbol, 1000)
            
            # Generate 6 months of daily data (approx 130 trading days)
            dates = pd.date_range(end=datetime.now(), periods=130, freq='D')
            dates = dates[dates.weekday < 5]  # Only weekdays
            
            # Generate realistic price movements
            np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol
            
            returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
            returns[0] = 0  # Start with no change
            
            # Add some trend and momentum
            trend = np.cumsum(np.random.normal(0, 0.001, len(dates)))
            momentum = np.zeros(len(dates))
            for i in range(1, len(dates)):
                momentum[i] = 0.1 * returns[i-1] + 0.9 * momentum[i-1]
            
            adjusted_returns = returns + trend + momentum
            
            # Calculate prices
            prices = base_price * np.exp(np.cumsum(adjusted_returns))
            
            # Generate OHLC data
            high_multiplier = np.random.uniform(1.005, 1.025, len(dates))
            low_multiplier = np.random.uniform(0.975, 0.995, len(dates))
            
            open_prices = np.roll(prices, 1)
            open_prices[0] = prices[0]
            
            high_prices = prices * high_multiplier
            low_prices = prices * low_multiplier
            close_prices = prices
            
            # Generate volume (higher volume on price moves)
            base_volume = 1000000
            volume_multiplier = 1 + np.abs(adjusted_returns) * 5
            volumes = np.random.poisson(base_volume * volume_multiplier)
            
            # Create DataFrame
            data = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes
            }, index=dates)
            
            logger.info(f"Generated demo data for {symbol}: {len(data)} days, price range ₹{data['Close'].min():.2f}-₹{data['Close'].max():.2f}")
            return data
            
        except Exception as e:
            logger.error(f"Error generating demo data for {symbol}: {e}")
            # Return minimal data as last resort
            return pd.DataFrame({
                'Open': [1000], 'High': [1020], 'Low': [980], 'Close': [1010], 'Volume': [100000]
            }, index=[datetime.now()])

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if data.empty:
            return pd.DataFrame()
        
        try:
            # Trend Indicators
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # MACD
            data['MACD'] = ta.trend.macd_diff(data['Close'])
            data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            data['BB_High'] = bb.bollinger_hband()
            data['BB_Low'] = bb.bollinger_lband()
            data['BB_Mid'] = bb.bollinger_mavg()
            
            # Volume Indicators
            data['Volume_SMA'] = ta.volume.volume_sma(data['Close'], data['Volume'], window=20)
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
            # Volatility
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Stochastic
            data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # Price action features
            data['Price_Change'] = data['Close'].pct_change()
            data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
            data['Close_Open_Pct'] = (data['Close'] - data['Open']) / data['Open']
            
            return data.dropna()
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for ML models"""
        if data.empty:
            return np.array([])
        
        feature_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
            'RSI', 'BB_High', 'BB_Low', 'BB_Mid', 'Volume_SMA', 'OBV',
            'ATR', 'Stoch_K', 'Stoch_D', 'Price_Change', 'High_Low_Pct', 'Close_Open_Pct'
        ]
        
        # Select only available columns
        available_columns = [col for col in feature_columns if col in data.columns]
        
        if not available_columns:
            return np.array([])
        
        features = data[available_columns].values
        return features[-1:] if len(features) > 0 else np.array([])

    async def generate_ai_analysis(self, symbol: str, signal: str, confidence: float, market_data: Dict) -> str:
        """Generate AI-powered analysis using OpenAI"""
        try:
            if not openai_api_key:
                return f"Technical analysis suggests {signal} signal for {symbol} with {confidence:.2%} confidence."
            
            chat = LlmChat(
                api_key=openai_api_key,
                session_id=f"trading_analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}",
                system_message="You are an expert Indian stock market trader with deep knowledge of F&O trading, technical analysis, and market dynamics."
            ).with_model("openai", "gpt-4o")
            
            analysis_prompt = f"""
            Analyze the trading signal for {symbol} (Indian F&O market):
            
            Signal: {signal}
            Confidence: {confidence:.2%}
            
            Market Data Summary:
            - Current Price: ₹{market_data.get('current_price', 'N/A')}
            - RSI: {market_data.get('rsi', 'N/A')}
            - MACD: {market_data.get('macd', 'N/A')}
            - Volume: {market_data.get('volume', 'N/A')}
            
            Provide a concise but insightful analysis covering:
            1. Technical rationale for the signal
            2. Key risk factors specific to this stock/sector
            3. Optimal entry/exit strategy for F&O trading
            4. Market timing considerations
            
            Keep the response under 200 words and focus on actionable insights.
            """
            
            user_message = UserMessage(text=analysis_prompt)
            response = await chat.send_message(user_message)
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
            return f"Technical analysis indicates {signal} signal for {symbol}. Consider market conditions and risk management."

    async def generate_signal(self, symbol: str) -> TradingSignal:
        """Generate trading signal for a symbol"""
        try:
            # Get market data
            data = self.get_market_data(symbol)
            if data.empty:
                raise Exception(f"No market data available for {symbol}")
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            if data.empty:
                raise Exception(f"Unable to calculate indicators for {symbol}")
            
            # Prepare features
            features = self.prepare_features(data)
            if len(features) == 0:
                raise Exception(f"Unable to prepare features for {symbol}")
            
            # Get current market metrics
            current_price = float(data['Close'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50.0
            macd = float(data['MACD'].iloc[-1]) if 'MACD' in data.columns else 0.0
            volume = int(data['Volume'].iloc[-1])
            
            # Simple rule-based signal generation (enhanced ML will come later)
            signal_score = 0
            confidence_factors = []
            
            # RSI analysis
            if rsi < 30:
                signal_score += 2
                confidence_factors.append("RSI oversold")
            elif rsi > 70:
                signal_score -= 2
                confidence_factors.append("RSI overbought")
            
            # MACD analysis
            if macd > 0:
                signal_score += 1
                confidence_factors.append("MACD bullish")
            else:
                signal_score -= 1
                confidence_factors.append("MACD bearish")
            
            # Price vs moving averages
            sma_20 = float(data['SMA_20'].iloc[-1]) if 'SMA_20' in data.columns else current_price
            if current_price > sma_20:
                signal_score += 1
                confidence_factors.append("Above SMA20")
            else:
                signal_score -= 1
                confidence_factors.append("Below SMA20")
            
            # Determine signal and confidence
            if signal_score >= 2:
                signal = "BUY"
                confidence = min(0.85, 0.5 + (signal_score * 0.1))
            elif signal_score <= -2:
                signal = "SELL" 
                confidence = min(0.85, 0.5 + (abs(signal_score) * 0.1))
            else:
                signal = "HOLD"
                confidence = 0.4 + (abs(signal_score) * 0.05)
            
            # Calculate targets and stop loss
            atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else current_price * 0.02
            
            if signal == "BUY":
                target_price = current_price + (atr * 2)
                stop_loss = current_price - (atr * 1.5)
            elif signal == "SELL":
                target_price = current_price - (atr * 2)
                stop_loss = current_price + (atr * 1.5)
            else:
                target_price = None
                stop_loss = None
            
            # Calculate risk-reward ratio
            risk_reward_ratio = None
            if target_price and stop_loss and signal != "HOLD":
                risk = abs(current_price - stop_loss)
                reward = abs(target_price - current_price)
                risk_reward_ratio = reward / risk if risk > 0 else None
            
            # Generate AI analysis
            market_data_summary = {
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd,
                'volume': volume
            }
            
            ai_analysis = await self.generate_ai_analysis(symbol, signal, confidence, market_data_summary)
            
            # Create trading signal
            sector = self.sector_mapping.get(symbol, "Unknown")
            reasoning = f"Technical factors: {', '.join(confidence_factors)}. Signal score: {signal_score}"
            
            trading_signal = TradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning,
                ai_analysis=ai_analysis,
                sector=sector,
                risk_reward_ratio=risk_reward_ratio
            )
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            # Return a default signal in case of error
            return TradingSignal(
                symbol=symbol,
                signal="HOLD",
                confidence=0.3,
                entry_price=0.0,
                reasoning=f"Unable to analyze {symbol}: {str(e)}",
                ai_analysis="Technical analysis unavailable due to data issues.",
                sector=self.sector_mapping.get(symbol, "Unknown")
            )

# Initialize AI Trading Engine
ai_engine = AITradingEngine()

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "Friday AI Trading System - Ready to Trade!", "version": "1.0.0"}

@api_router.post("/signals/generate", response_model=List[TradingSignal])
async def generate_signals(request: TradingSignalCreate, background_tasks: BackgroundTasks):
    """Generate trading signals for specified symbols or all F&O symbols"""
    try:
        symbols = request.symbols if request.symbols else FNO_SYMBOLS[:10]  # Limit to first 10 for demo
        
        signals = []
        for symbol in symbols:
            try:
                signal = await ai_engine.generate_signal(symbol)
                signals.append(signal)
                
                # Store signal in database
                await db.trading_signals.insert_one(signal.dict())
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/signals/latest", response_model=List[TradingSignal])
async def get_latest_signals(limit: int = 20):
    """Get latest trading signals"""
    try:
        signals = await db.trading_signals.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return [TradingSignal(**signal) for signal in signals]
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/portfolio/metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics():
    """Get portfolio performance metrics"""
    try:
        # Calculate metrics from trades (placeholder for now)
        total_trades = await db.trades.count_documents({})
        winning_trades = await db.trades.count_documents({"pnl": {"$gt": 0}})
        losing_trades = await db.trades.count_documents({"pnl": {"$lt": 0}})
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate total PnL
        pipeline = [
            {"$group": {"_id": None, "total_pnl": {"$sum": "$pnl"}}}
        ]
        pnl_result = await db.trades.aggregate(pipeline).to_list(1)
        total_pnl = pnl_result[0]["total_pnl"] if pnl_result else 0
        
        metrics = PortfolioMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl
        )
        
        # Store metrics
        await db.portfolio_metrics.insert_one(metrics.dict())
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/symbols")
async def get_symbols():
    """Get all F&O symbols grouped by sector"""
    try:
        symbols_by_sector = {}
        for symbol in FNO_SYMBOLS:
            sector = ai_engine.sector_mapping.get(symbol, "Unknown")
            if sector not in symbols_by_sector:
                symbols_by_sector[sector] = []
            symbols_by_sector[sector].append(symbol)
        
        return symbols_by_sector
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/learning/insight", response_model=LearningInsight)
async def create_learning_insight(insight: LearningInsight):
    """Create a learning insight from trading experience"""
    try:
        await db.learning_insights.insert_one(insight.dict())
        return insight
    except Exception as e:
        logger.error(f"Error creating learning insight: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/learning/insights", response_model=List[LearningInsight])
async def get_learning_insights(limit: int = 10):
    """Get recent learning insights"""
    try:
        insights = await db.learning_insights.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return [LearningInsight(**insight) for insight in insights]
    except Exception as e:
        logger.error(f"Error fetching learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
