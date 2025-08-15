#!/usr/bin/env python3
"""
Daily progress tracker for rex.com 30-day plan
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cryptotrading.data.database import DatabaseClient

class DailyProgressTracker:
    def __init__(self):
        self.db = DatabaseClient(db_path='data/rex.db')
        self.start_date = datetime(2025, 8, 12)  # Today
        self.current_day = (datetime.now() - self.start_date).days + 1
        
    def get_current_week_tasks(self) -> Dict[str, List[str]]:
        """Get tasks for current week"""
        week_num = ((self.current_day - 1) // 7) + 1
        
        weeks = {
            1: {
                "theme": "Basic Infrastructure Setup",
                "days": {
                    1: ["Create Working Environment", "Project setup", "API keys"],
                    2: ["Market Research", "Top 20 cryptos", "Select pairs"],
                    3: ["Install Libraries", "CCXT setup", "API connections"],
                    4: ["Database Setup", "PostgreSQL", "Schema design"],
                    5: ["Historical Data Part 1", "Major pairs", "Rate limits"],
                    6: ["Historical Data Part 2", "Validation", "Cleanup"],
                    7: ["Analytics & Visualization", "Jupyter", "Correlations"]
                }
            },
            2: {
                "theme": "Technical Analysis and Indicators",
                "days": {
                    8: ["TA-Lib Setup", "SMA/EMA", "RSI"],
                    9: ["MACD", "Bollinger Bands", "ATR"],
                    10: ["Volume Indicators", "OBV", "VWAP"],
                    11: ["Support/Resistance", "Pivot Points", "Fibonacci"],
                    12: ["Chart Patterns", "Triangles", "Flags"],
                    13: ["Advanced Patterns", "Candlesticks", "H&S"],
                    14: ["Integrate Analytics", "Dashboard", "Performance"]
                }
            },
            3: {
                "theme": "Manual Trading and Intuition",
                "days": {
                    15: ["Paper Trading System", "Virtual Portfolio"],
                    16: ["Start Paper Trading", "Trading Diary"],
                    17: ["Market Cycles", "Session Analysis"],
                    18: ["Risk Management", "Position Sizing"],
                    19: ["Sentiment Analysis", "News Monitoring"],
                    20: ["Correlation Trading", "Portfolio Diversification"],
                    21: ["Weekly Review", "Performance Metrics"]
                }
            },
            4: {
                "theme": "Automation and ML Preparation",
                "days": {
                    22: ["Real-time Data", "WebSockets", "Monitoring"],
                    23: ["Alerts System", "Notifications", "Telegram"],
                    24: ["Advanced Metrics", "Market Structure", "Ranking"],
                    25: ["ML Data Prep", "Feature Engineering", "Normalization"],
                    26: ["ML Libraries", "Baseline Models", "Pipeline"],
                    27: ["First ML Models", "Price Prediction", "Validation"],
                    28: ["Model Testing", "Backtesting", "Walk-forward"],
                    29: ["System Integration", "Control Module", "Health Checks"],
                    30: ["Documentation", "Final Testing", "Month 2 Plan"]
                }
            }
        }
        
        if week_num <= 4:
            return weeks[week_num]
        return {"theme": "Continued Development", "days": {}}
    
    def display_progress(self):
        """Display current progress and today's tasks"""
        print(f"🚀 rex.com 30-Day Progress Tracker")
        print(f"{'='*50}")
        print(f"📅 Day {self.current_day} of 30")
        print(f"📆 Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        week_info = self.get_current_week_tasks()
        week_num = ((self.current_day - 1) // 7) + 1
        day_in_week = ((self.current_day - 1) % 7) + 1
        
        print(f"\n📌 Week {week_num}: {week_info['theme']}")
        
        # Show this week's progress
        print(f"\n📊 This Week's Progress:")
        for day_num in range(1, 8):
            actual_day = (week_num - 1) * 7 + day_num
            if actual_day > 30:
                break
                
            if actual_day < self.current_day:
                status = "✅"
            elif actual_day == self.current_day:
                status = "🔄"
            else:
                status = "⏳"
                
            tasks = week_info['days'].get(actual_day, ["No tasks"])
            print(f"{status} Day {actual_day}: {tasks[0]}")
        
        # Today's detailed tasks
        if self.current_day <= 30:
            print(f"\n📋 Today's Tasks (Day {self.current_day}):")
            today_tasks = week_info['days'].get(self.current_day, ["Review and planning"])
            for i, task in enumerate(today_tasks, 1):
                print(f"  {i}. {task}")
        
        # Progress summary
        print(f"\n📈 Overall Progress: {self.current_day}/30 days ({(self.current_day/30)*100:.1f}%)")
        
        # What's been completed
        print(f"\n✅ Completed Systems:")
        print("  • Basic infrastructure (rex.com platform)")
        print("  • Database system (SQLite)")
        print("  • AI integration (DeepSeek R1, Perplexity)")
        print("  • Blockchain integration (MetaMask)")
        print("  • A2A agent framework")
        
        # Next priorities
        if self.current_day <= 30:
            print(f"\n🎯 Next Priority Tasks:")
            if self.current_day <= 7:
                print("  • Complete historical data collection")
                print("  • Set up PostgreSQL/TimescaleDB")
            elif self.current_day <= 14:
                print("  • Implement technical indicators")
                print("  • Create analytics dashboard")
            elif self.current_day <= 21:
                print("  • Build paper trading system")
                print("  • Practice manual trading")
            else:
                print("  • Develop ML models")
                print("  • System integration")

def main():
    tracker = DailyProgressTracker()
    tracker.display_progress()

if __name__ == '__main__':
    main()