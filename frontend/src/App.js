import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Dashboard Components
const TradingDashboard = () => {
  const [signals, setSignals] = useState([]);
  const [portfolioMetrics, setPortfolioMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedSymbols, setSelectedSymbols] = useState([]);
  const [symbolsBySection, setSymbolsBySection] = useState({});
  const [insights, setInsights] = useState([]);
  const [activeTab, setActiveTab] = useState('signals');

  useEffect(() => {
    fetchSymbols();
    fetchLatestSignals();
    fetchPortfolioMetrics();
    fetchLearningInsights();
  }, []);

  const fetchSymbols = async () => {
    try {
      const response = await axios.get(`${API}/symbols`);
      setSymbolsBySection(response.data);
    } catch (error) {
      console.error('Error fetching symbols:', error);
    }
  };

  const fetchLatestSignals = async () => {
    try {
      const response = await axios.get(`${API}/signals/latest`);
      setSignals(response.data);
    } catch (error) {
      console.error('Error fetching signals:', error);
    }
  };

  const fetchPortfolioMetrics = async () => {
    try {
      const response = await axios.get(`${API}/portfolio/metrics`);
      setPortfolioMetrics(response.data);
    } catch (error) {
      console.error('Error fetching portfolio metrics:', error);
    }
  };

  const fetchLearningInsights = async () => {
    try {
      const response = await axios.get(`${API}/learning/insights`);
      setInsights(response.data);
    } catch (error) {
      console.error('Error fetching learning insights:', error);
    }
  };

  const generateSignals = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/signals/generate`, {
        symbols: selectedSymbols.length > 0 ? selectedSymbols : null
      });
      setSignals(response.data);
      await fetchPortfolioMetrics();
    } catch (error) {
      console.error('Error generating signals:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'BUY': return 'text-green-600 bg-green-100';
      case 'SELL': return 'text-red-600 bg-red-100';
      case 'HOLD': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return 'text-green-600';
    if (confidence >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-lg border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-400 to-purple-500 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">F</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Friday AI Trading</h1>
                <p className="text-blue-300 text-sm">Intelligent F&O Trading System</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-white font-semibold">
                  {portfolioMetrics ? `₹${portfolioMetrics.total_pnl.toFixed(2)}` : '₹0.00'}
                </div>
                <div className="text-blue-300 text-sm">Total P&L</div>
              </div>
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <div className="relative h-64 bg-gradient-to-r from-blue-600/20 to-purple-600/20 backdrop-blur-sm">
        <div className="absolute inset-0 bg-black/30"></div>
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-20"
          style={{
            backgroundImage: 'url(https://images.pexels.com/photos/7789849/pexels-photo-7789849.jpeg)'
          }}
        ></div>
        <div className="relative max-w-7xl mx-auto px-6 py-16">
          <div className="text-center">
            <h2 className="text-4xl font-bold text-white mb-4">
              AI-Powered F&O Trading Intelligence
            </h2>
            <p className="text-xl text-blue-200 mb-8">
              Advanced machine learning algorithms analyzing 30+ Indian F&O symbols with real-time insights
            </p>
            <button
              onClick={generateSignals}
              disabled={loading}
              className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-8 py-3 rounded-xl font-semibold hover:from-blue-600 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Generating AI Signals...</span>
                </div>
              ) : (
                'Generate AI Trading Signals'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex space-x-1 bg-white/10 backdrop-blur-lg rounded-xl p-1">
          {[
            { id: 'signals', label: 'Trading Signals', icon: '📊' },
            { id: 'portfolio', label: 'Portfolio Metrics', icon: '💼' },
            { id: 'learning', label: 'AI Learning', icon: '🧠' },
            { id: 'symbols', label: 'F&O Symbols', icon: '📈' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all duration-300 ${
                activeTab === tab.id
                  ? 'bg-white text-blue-900 shadow-lg'
                  : 'text-white hover:bg-white/10'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="max-w-7xl mx-auto px-6 pb-12">
        {/* Trading Signals Tab */}
        {activeTab === 'signals' && (
          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">Latest AI Trading Signals</h3>
              
              {signals.length === 0 ? (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">🤖</span>
                  </div>
                  <p className="text-white/70">No signals available. Generate new signals to get started!</p>
                </div>
              ) : (
                <div className="grid gap-4">
                  {signals.slice(0, 8).map((signal, index) => (
                    <div key={index} className="bg-white/5 rounded-xl p-4 hover:bg-white/10 transition-all duration-300">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
                            <span className="text-white font-bold text-sm">{signal.symbol.slice(0, 2)}</span>
                          </div>
                          <div>
                            <h4 className="text-white font-semibold">{signal.symbol}</h4>
                            <p className="text-blue-300 text-sm">{signal.sector}</p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-4">
                          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getSignalColor(signal.signal)}`}>
                            {signal.signal}
                          </span>
                          <div className="text-right">
                            <div className={`font-bold ${getConfidenceColor(signal.confidence)}`}>
                              {(signal.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="text-xs text-white/70">Confidence</div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                        <div>
                          <div className="text-xs text-white/70">Entry Price</div>
                          <div className="text-white font-semibold">₹{signal.entry_price.toFixed(2)}</div>
                        </div>
                        {signal.target_price && (
                          <div>
                            <div className="text-xs text-white/70">Target</div>
                            <div className="text-green-400 font-semibold">₹{signal.target_price.toFixed(2)}</div>
                          </div>
                        )}
                        {signal.stop_loss && (
                          <div>
                            <div className="text-xs text-white/70">Stop Loss</div>
                            <div className="text-red-400 font-semibold">₹{signal.stop_loss.toFixed(2)}</div>
                          </div>
                        )}
                        {signal.risk_reward_ratio && (
                          <div>
                            <div className="text-xs text-white/70">Risk:Reward</div>
                            <div className="text-white font-semibold">1:{signal.risk_reward_ratio.toFixed(2)}</div>
                          </div>
                        )}
                      </div>
                      
                      <div className="bg-white/5 rounded-lg p-3 mb-3">
                        <div className="text-xs text-white/70 mb-1">Technical Analysis</div>
                        <p className="text-white/90 text-sm">{signal.reasoning}</p>
                      </div>
                      
                      <div className="bg-blue-500/10 rounded-lg p-3">
                        <div className="text-xs text-blue-300 mb-1">🤖 AI Analysis</div>
                        <p className="text-white/90 text-sm">{signal.ai_analysis}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Portfolio Metrics Tab */}
        {activeTab === 'portfolio' && portfolioMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Total Trades</h3>
                <span className="text-2xl">📊</span>
              </div>
              <div className="text-3xl font-bold text-white">{portfolioMetrics.total_trades}</div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Win Rate</h3>
                <span className="text-2xl">🎯</span>
              </div>
              <div className="text-3xl font-bold text-green-400">{portfolioMetrics.win_rate.toFixed(1)}%</div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Total P&L</h3>
                <span className="text-2xl">💰</span>
              </div>
              <div className={`text-3xl font-bold ${portfolioMetrics.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ₹{portfolioMetrics.total_pnl.toFixed(2)}
              </div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Active Positions</h3>
                <span className="text-2xl">📈</span>
              </div>
              <div className="text-3xl font-bold text-blue-400">0</div>
            </div>
          </div>
        )}

        {/* AI Learning Tab */}
        {activeTab === 'learning' && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6">
            <h3 className="text-xl font-semibold text-white mb-6">🧠 AI Learning Insights</h3>
            
            {insights.length === 0 ? (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">🧠</span>
                </div>
                <p className="text-white/70">No learning insights yet. The AI will learn from trading patterns and mistakes.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {insights.map((insight, index) => (
                  <div key={index} className="bg-white/5 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-blue-300">{insight.symbol}</span>
                      <span className="text-white/70 text-sm">{insight.mistake_type}</span>
                    </div>
                    <p className="text-white/90 mb-2">{insight.lesson_learned}</p>
                    <p className="text-blue-200 text-sm">{insight.ai_analysis}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* F&O Symbols Tab */}
        {activeTab === 'symbols' && (
          <div className="space-y-6">
            {Object.entries(symbolsBySection).map(([sector, symbols]) => (
              <div key={sector} className="bg-white/10 backdrop-blur-lg rounded-2xl p-6">
                <h3 className="text-xl font-semibold text-white mb-4">{sector}</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                  {symbols.map((symbol) => (
                    <div 
                      key={symbol}
                      className="bg-white/5 hover:bg-white/10 rounded-lg p-3 text-center cursor-pointer transition-all duration-300"
                      onClick={() => {
                        if (selectedSymbols.includes(symbol)) {
                          setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
                        } else {
                          setSelectedSymbols([...selectedSymbols, symbol]);
                        }
                      }}
                    >
                      <div className={`w-8 h-8 rounded-full mx-auto mb-2 flex items-center justify-center text-xs font-bold ${
                        selectedSymbols.includes(symbol) 
                          ? 'bg-blue-500 text-white' 
                          : 'bg-white/10 text-white/70'
                      }`}>
                        {symbol.slice(0, 2)}
                      </div>
                      <div className="text-white text-sm">{symbol}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <TradingDashboard />
    </div>
  );
}

export default App;
