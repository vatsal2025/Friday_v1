#!/usr/bin/env python3
import requests
import json
import os
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import unittest

# Get backend URL from frontend/.env
BACKEND_URL = None
try:
    with open('/app/frontend/.env', 'r') as f:
        for line in f:
            if line.startswith('REACT_APP_BACKEND_URL='):
                BACKEND_URL = line.strip().split('=')[1].strip('"\'')
                break
except Exception as e:
    print(f"Error reading frontend/.env: {e}")
    sys.exit(1)

if not BACKEND_URL:
    print("BACKEND_URL not found in frontend/.env")
    sys.exit(1)

# Ensure BACKEND_URL ends with /api
API_URL = f"{BACKEND_URL}/api" if not BACKEND_URL.endswith('/api') else BACKEND_URL

print(f"Using API URL: {API_URL}")

class FridayAITradingSystemTest(unittest.TestCase):
    """Test suite for Friday AI Trading System backend"""

    def setUp(self):
        """Set up test case"""
        self.api_url = API_URL
        self.test_symbols = ["INFY", "HDFCBANK", "SUNPHARMA"]  # Test with a few symbols for speed
        
        # Check if API is accessible
        try:
            response = requests.get(f"{self.api_url}/")
            if response.status_code != 200:
                print(f"API not accessible: {response.status_code}")
                print(response.text)
                sys.exit(1)
            print("API is accessible")
        except Exception as e:
            print(f"Error accessing API: {e}")
            sys.exit(1)

    def test_01_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{self.api_url}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        print(f"Root endpoint test passed: {data}")

    def test_02_symbols_endpoint(self):
        """Test F&O symbols endpoint"""
        response = requests.get(f"{self.api_url}/symbols")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify structure and content
        self.assertIsInstance(data, dict)
        
        # Check if we have the expected sectors
        expected_sectors = ["Banking", "IT", "Pharma", "Manufacturing", "FMCG", "Consumer Services", "Chemicals"]
        for sector in expected_sectors:
            self.assertIn(sector, data)
            self.assertIsInstance(data[sector], list)
            self.assertTrue(len(data[sector]) > 0)
        
        # Check if we have at least 30 symbols in total
        total_symbols = sum(len(symbols) for symbols in data.values())
        self.assertGreaterEqual(total_symbols, 30)
        
        print(f"Symbols endpoint test passed: {len(data)} sectors, {total_symbols} symbols")
        
        # Print a sample of symbols by sector
        for sector, symbols in data.items():
            print(f"  {sector}: {symbols[:3]}{'...' if len(symbols) > 3 else ''}")

    def test_03_generate_signals(self):
        """Test signal generation endpoint"""
        payload = {"symbols": self.test_symbols}
        
        print(f"Generating signals for: {self.test_symbols}")
        start_time = time.time()
        
        response = requests.post(f"{self.api_url}/signals/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        signals = response.json()
        self.assertIsInstance(signals, list)
        self.assertEqual(len(signals), len(self.test_symbols))
        
        print(f"Signal generation test passed: {len(signals)} signals generated in {elapsed:.2f} seconds")
        
        # Validate signal structure and content
        for signal in signals:
            self.assertIn("symbol", signal)
            self.assertIn("signal", signal)
            self.assertIn("confidence", signal)
            self.assertIn("entry_price", signal)
            self.assertIn("reasoning", signal)
            self.assertIn("ai_analysis", signal)
            self.assertIn("sector", signal)
            
            # Verify signal type
            self.assertIn(signal["signal"], ["BUY", "SELL", "HOLD"])
            
            # Verify confidence is in expected range (0.3-0.85)
            self.assertGreaterEqual(signal["confidence"], 0.3)
            self.assertLessEqual(signal["confidence"], 0.85)
            
            # Verify entry price is positive
            self.assertGreater(signal["entry_price"], 0)
            
            # Verify target and stop loss for BUY/SELL signals
            if signal["signal"] in ["BUY", "SELL"]:
                self.assertIn("target_price", signal)
                self.assertIn("stop_loss", signal)
                self.assertIsNotNone(signal["target_price"])
                self.assertIsNotNone(signal["stop_loss"])
                
                # Verify risk_reward_ratio
                self.assertIn("risk_reward_ratio", signal)
                if signal["risk_reward_ratio"] is not None:
                    self.assertGreater(signal["risk_reward_ratio"], 0)
            
            print(f"  {signal['symbol']}: {signal['signal']} (confidence: {signal['confidence']:.2f})")
            print(f"    Entry: ₹{signal['entry_price']:.2f}")
            if signal["signal"] in ["BUY", "SELL"] and "target_price" in signal and signal["target_price"]:
                print(f"    Target: ₹{signal['target_price']:.2f}")
                print(f"    Stop Loss: ₹{signal['stop_loss']:.2f}")
                if signal["risk_reward_ratio"]:
                    print(f"    Risk/Reward: {signal['risk_reward_ratio']:.2f}")
            
            # Print a snippet of the AI analysis
            ai_analysis = signal["ai_analysis"]
            print(f"    AI Analysis: {ai_analysis[:100]}..." if len(ai_analysis) > 100 else ai_analysis)
            print()

    def test_04_latest_signals(self):
        """Test latest signals endpoint"""
        response = requests.get(f"{self.api_url}/signals/latest")
        self.assertEqual(response.status_code, 200)
        
        signals = response.json()
        self.assertIsInstance(signals, list)
        
        # We should have at least the signals we just generated
        self.assertGreaterEqual(len(signals), len(self.test_symbols))
        
        print(f"Latest signals test passed: {len(signals)} signals retrieved")
        
        # Print a sample of the latest signals
        for i, signal in enumerate(signals[:3]):
            print(f"  Signal {i+1}: {signal['symbol']} - {signal['signal']} (confidence: {signal['confidence']:.2f})")

    def test_05_portfolio_metrics(self):
        """Test portfolio metrics endpoint"""
        response = requests.get(f"{self.api_url}/portfolio/metrics")
        self.assertEqual(response.status_code, 200)
        
        metrics = response.json()
        self.assertIsInstance(metrics, dict)
        
        # Verify structure
        self.assertIn("total_trades", metrics)
        self.assertIn("winning_trades", metrics)
        self.assertIn("losing_trades", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("total_pnl", metrics)
        
        print(f"Portfolio metrics test passed:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Winning Trades: {metrics['winning_trades']}")
        print(f"  Losing Trades: {metrics['losing_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Total PnL: ₹{metrics['total_pnl']:.2f}")

    def test_06_learning_insights(self):
        """Test learning insights endpoints"""
        # Create a test insight
        test_insight = {
            "trade_id": "test-trade-123",
            "symbol": "INFY",
            "mistake_type": "Early Exit",
            "lesson_learned": "Hold positions longer when trend is strong",
            "ai_analysis": "The trade was exited prematurely before the target was reached. The trend indicators were still positive.",
            "confidence_adjustment": -0.05
        }
        
        # Test POST endpoint
        post_response = requests.post(f"{self.api_url}/learning/insight", json=test_insight)
        self.assertEqual(post_response.status_code, 200)
        
        created_insight = post_response.json()
        self.assertIn("id", created_insight)
        self.assertEqual(created_insight["symbol"], test_insight["symbol"])
        self.assertEqual(created_insight["mistake_type"], test_insight["mistake_type"])
        
        print(f"Learning insight creation test passed: {created_insight['id']}")
        
        # Test GET endpoint
        get_response = requests.get(f"{self.api_url}/learning/insights")
        self.assertEqual(get_response.status_code, 200)
        
        insights = get_response.json()
        self.assertIsInstance(insights, list)
        self.assertGreaterEqual(len(insights), 1)
        
        print(f"Learning insights retrieval test passed: {len(insights)} insights retrieved")
        
        # Print a sample of the insights
        for i, insight in enumerate(insights[:2]):
            print(f"  Insight {i+1}: {insight['symbol']} - {insight['mistake_type']}")
            print(f"    Lesson: {insight['lesson_learned']}")

    def test_07_openai_integration(self):
        """Test OpenAI integration specifically"""
        # Generate a signal for a single symbol to test OpenAI integration
        test_symbol = "HDFCBANK"
        payload = {"symbols": [test_symbol]}
        
        print(f"Testing OpenAI integration with symbol: {test_symbol}")
        
        response = requests.post(f"{self.api_url}/signals/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        
        signals = response.json()
        self.assertEqual(len(signals), 1)
        
        signal = signals[0]
        ai_analysis = signal["ai_analysis"]
        
        # Verify that the AI analysis is not the default fallback message
        self.assertNotIn("Technical analysis suggests", ai_analysis)
        self.assertNotIn("Technical analysis indicates", ai_analysis)
        
        # Check for indicators of a proper OpenAI-generated analysis
        # The analysis should be detailed and contain specific insights
        self.assertGreater(len(ai_analysis), 100)  # Should be reasonably long
        
        # Look for specific technical terms that would indicate a proper analysis
        technical_terms = ["trend", "support", "resistance", "momentum", "volume", 
                          "market", "price", "level", "risk", "target"]
        
        term_count = sum(1 for term in technical_terms if term.lower() in ai_analysis.lower())
        self.assertGreaterEqual(term_count, 3)  # Should contain at least 3 technical terms
        
        print(f"OpenAI integration test passed")
        print(f"AI Analysis for {test_symbol}:")
        print(f"{ai_analysis}")
        
        # Check if the analysis contains stock-specific information
        self.assertIn(test_symbol, ai_analysis)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)