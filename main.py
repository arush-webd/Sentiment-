#!/usr/bin/env python3
"""
Main application script for Sentiment Trading System
"""

import yaml
import logging
import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

# Import our custom modules
from sentiment_analyzer import AdvancedFinancialSentimentAnalyzer
from trading_pipeline import SentimentTradingPipeline
from cmie_integration import CMIEDataIntegration, CMIEOptimizer

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentTradingSystem:
    def __init__(self, config_path='configs/config.yaml'):
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = self._get_default_config()
        
        # Initialize components
        logger.info("Initializing Sentiment Trading System...")
        
        try:
            self.sentiment_analyzer = AdvancedFinancialSentimentAnalyzer()
            logger.info("✓ Sentiment analyzer loaded")
        except Exception as e:
            logger.error(f"Failed to load sentiment analyzer: {e}")
            return
            
        try:
            self.trading_pipeline = SentimentTradingPipeline(self.config['database']['path'])
            logger.info("✓ Trading pipeline loaded")
        except Exception as e:
            logger.error(f"Failed to load trading pipeline: {e}")
            return
            
        try:
            self.cmie_integration = CMIEDataIntegration(self.config['database']['path'])
            self.cmie_integration.sentiment_analyzer = self.sentiment_analyzer
            logger.info("✓ CMIE integration loaded")
        except Exception as e:
            logger.error(f"Failed to load CMIE integration: {e}")
            return
        
        try:
            self.optimizer = CMIEOptimizer(self.cmie_integration)
            logger.info("✓ CMIE optimizer loaded")
        except Exception as e:
            logger.error(f"Failed to load optimizer: {e}")
            return
        
        logger.info("🚀 Sentiment Trading System initialized successfully!")
    
    def _get_default_config(self):
        """Return default configuration"""
        return {
            'database': {'path': 'data/sentiment_data.db'},
            'cmie': {
                'max_clicks_per_day': 50, 
                'rate_limit_seconds': 2,
                'priority_companies': 20
            },
            'trading': {
                'confidence_threshold': 0.65, 
                'max_position_size': 0.05,
                'risk_free_rate': 0.065,
                'transaction_cost': 0.002
            },
            'models': {
                'retrain_frequency': 'weekly', 
                'lookback_days': 252,
                'min_samples': 100
            }
        }
    
    def run_daily_pipeline(self):
        """Run the daily data collection and analysis pipeline"""
        logger.info("=== Starting Daily Pipeline ===")
        
        try:
            # Check CMIE click budget
            remaining_clicks = self.cmie_integration.track_usage(0)
            daily_budget = self.config['cmie']['max_clicks_per_day']
            
            if remaining_clicks < daily_budget:
                logger.warning(f"Low on CMIE clicks: {remaining_clicks} remaining")
                daily_budget = min(daily_budget, remaining_clicks)
            
            # Get priority companies for today
            priority_companies = self.optimizer.calculate_company_priority(
                self.cmie_integration.get_priority_companies()
            )
            
            logger.info(f"Today's budget: {daily_budget} clicks for {len(priority_companies)} companies")
            
            # In a real implementation, you would:
            # 1. Scrape CMIE for new articles using cmie_prowess_scraper.py
            # 2. Process sentiment for each article
            # 3. Update the database
            # 4. Generate new trading signals
            
            # For now, we'll use sample data
            sample_articles = self._get_sample_daily_data()
            
            if sample_articles:
                results = self.cmie_integration.batch_process_cmie_data(sample_articles)
                logger.info(f"Processed {results['processed']} articles")
                
                # Generate trading signals
                signals = self._generate_daily_signals()
                logger.info(f"Generated {len(signals)} trading signals")
                
                # Log top signals
                if not signals.empty:
                    top_signals = signals.nlargest(3, 'signal_strength')
                    logger.info("🔥 Top 3 Trading Signals:")
                    for _, signal in top_signals.iterrows():
                        logger.info(f"  - {signal['symbol']}: {signal['predicted_return']:.2%} "
                                  f"(confidence: {signal['confidence']:.2f})")
                
                return {
                    'articles_processed': results['processed'],
                    'signals_generated': len(signals),
                    'clicks_used': len(sample_articles),
                    'top_signals': top_signals.to_dict('records') if not signals.empty else []
                }
            else:
                logger.info("No new articles to process today")
                return {'articles_processed': 0, 'signals_generated': 0, 'clicks_used': 0}
            
        except Exception as e:
            logger.error(f"Daily pipeline error: {e}")
            return None
    
    def _get_sample_daily_data(self):
        """Get sample daily data (replace with actual CMIE scraping)"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Sample articles for testing
        sample_articles = [
            {
                'date': today,
                'headline': 'TCS demonstrates strong Q4 performance with robust margin expansion',
                'content': 'Tata Consultancy Services Ltd. reported exceptional quarterly results with margin expansion and strong client wins across key verticals.',
                'url': f'https://prowess.cmie.com/news/tcs-{today}',
                'category': 'Earnings'
            },
            {
                'date': today,
                'headline': 'Infosys announces strategic partnership with major European client',
                'content': 'Infosys Ltd. secured a multi-year digital transformation deal worth significant value, strengthening its European presence.',
                'url': f'https://prowess.cmie.com/news/infosys-{today}',
                'category': 'Business'
            },
            {
                'date': today,
                'headline': 'HDFC Bank maintains strong asset quality amid economic headwinds',
                'content': 'HDFC Bank Ltd. reported stable asset quality metrics and continued growth in retail banking segment.',
                'url': f'https://prowess.cmie.com/news/hdfc-{today}',
                'category': 'Financial'
            }
        ]
        
        return sample_articles
    
    def _generate_daily_signals(self):
        """Generate trading signals for today"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Create features from recent data
            feature_df = self.trading_pipeline.create_features(start_date, end_date)
            
            if not feature_df.empty:
                # Generate signals
                signals = self.trading_pipeline.generate_signals(
                    feature_df, 
                    confidence_threshold=self.config['trading']['confidence_threshold']
                )
                
                # Filter for today's signals only
                today = datetime.now().strftime('%Y-%m-%d')
                today_signals = signals[signals['date'] == today] if 'date' in signals.columns else signals
                
                return today_signals
            else:
                logger.warning("No feature data available for signal generation")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self, start_date, end_date, symbols=None):
        """Run comprehensive backtesting"""
        logger.info(f"=== Running Backtest: {start_date} to {end_date} ===")
        
        if symbols is None:
            symbols = self.cmie_integration.get_priority_companies()[:10]
        
        try:
            results = self.trading_pipeline.run_complete_pipeline(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols
            )
            
            if results:
                logger.info("=== Backtest Results ===")
                br = results['backtest_results']
                logger.info(f"📈 Annualized Return: {br['annualized_return']:.2%}")
                logger.info(f"📊 Sharpe Ratio: {br['sharpe_ratio']:.2f}")
                logger.info(f"📉 Max Drawdown: {br['max_drawdown']:.2%}")
                logger.info(f"🎯 Win Rate: {br['win_rate']:.2%}")
                logger.info(f"🔢 Total Trades: {br['total_trades']}")
                
                # Check if we're meeting our 40% target
                if br['annualized_return'] >= 0.40:
                    logger.info("🎯 TARGET ACHIEVED: 40%+ annual return!")
                    logger.info("🚀 Congratulations! Your sentiment trading system is working!")
                else:
                    progress = (br['annualized_return'] / 0.40) * 100
                    logger.info(f"📈 Progress: {br['annualized_return']:.1%} toward 40% target ({progress:.1f}% complete)")
                    
                    if br['annualized_return'] >= 0.20:
                        logger.info("💪 Strong performance! You're more than halfway to your goal.")
                    elif br['annualized_return'] >= 0.10:
                        logger.info("📊 Good progress! Consider optimizing parameters.")
                    else:
                        logger.info("🔧 Consider adjusting confidence thresholds or expanding data sources.")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            return None
    
    def get_system_status(self):
        """Get current system status"""
        status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cmie_clicks_remaining': self.cmie_integration.track_usage(0),
            'database_status': 'Connected',
            'models_loaded': len(self.sentiment_analyzer.models),
        }
        
        # Check database health
        try:
            conn = sqlite3.connect(self.config['database']['path'])
            cursor = conn.execute("SELECT COUNT(*) FROM news_sentiment")
            status['total_sentiment_records'] = cursor.fetchone()[0]
            
            # Check recent activity
            cursor = conn.execute("""
                SELECT COUNT(*) FROM news_sentiment 
                WHERE date >= date('now', '-7 days')
            """)
            status['recent_sentiment_records'] = cursor.fetchone()[0]
            
            conn.close()
        except Exception as e:
            status['database_status'] = f'Error: {e}'
            status['total_sentiment_records'] = 0
            status['recent_sentiment_records'] = 0
        
        return status
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        logger.info("=== Generating Daily Report ===")
        
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Generate sentiment report
            sentiment_report = self.cmie_integration.generate_sentiment_report(start_date, end_date)
            
            # Get system status
            system_status = self.get_system_status()
            
            # Generate signals for today
            signals = self._generate_daily_signals()
            
            # Create comprehensive report
            report = {
                'date': end_date,
                'system_status': system_status,
                'sentiment_summary': {
                    'total_companies': sentiment_report['total_companies'],
                    'total_articles': sentiment_report['total_articles'],
                    'avg_sentiment': sentiment_report['avg_sentiment_overall'],
                    'click_efficiency': sentiment_report['click_efficiency']
                },
                'signals': {
                    'count': len(signals),
                    'top_signals': signals.head(5).to_dict('records') if not signals.empty else [],
                    'avg_confidence': signals['confidence'].mean() if not signals.empty else 0
                }
            }
            
            # Log report summary
            logger.info(f"📊 Daily Report Summary:")
            logger.info(f"  - System Status: {system_status['database_status']}")
            logger.info(f"  - CMIE Clicks Remaining: {system_status['cmie_clicks_remaining']}")
            logger.info(f"  - Sentiment Records: {system_status['total_sentiment_records']} total, {system_status['recent_sentiment_records']} recent")
            logger.info(f"  - Today's Signals: {report['signals']['count']}")
            
            if report['signals']['top_signals']:
                logger.info(f"  - Top Signal: {report['signals']['top_signals'][0]['symbol']} "
                          f"({report['signals']['top_signals'][0]['predicted_return']:.2%})")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return None
    
    def run_weekly_analysis(self):
        """Run weekly comprehensive analysis"""
        logger.info("=== Running Weekly Analysis ===")
        
        try:
            # Run backtest for the past year
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            backtest_results = self.backtest_strategy(start_date, end_date)
            
            if backtest_results:
                # Generate optimization suggestions
                br = backtest_results['backtest_results']
                
                suggestions = []
                
                if br['sharpe_ratio'] < 1.0:
                    suggestions.append("Consider increasing confidence threshold to improve risk-adjusted returns")
                
                if br['win_rate'] < 0.55:
                    suggestions.append("Explore additional sentiment features or longer lookback periods")
                
                if br['max_drawdown'] > 0.20:
                    suggestions.append("Implement stronger position sizing or stop-loss mechanisms")
                
                if br['annualized_return'] < 0.15:
                    suggestions.append("Consider expanding to more companies or increasing update frequency")
                
                logger.info("💡 Weekly Optimization Suggestions:")
                for suggestion in suggestions:
                    logger.info(f"  - {suggestion}")
                
                return {
                    'backtest_results': br,
                    'suggestions': suggestions,
                    'feature_importance': backtest_results.get('feature_importance', {})
                }
            
        except Exception as e:
            logger.error(f"Weekly analysis error: {e}")
            return None

def main():
    """Main function"""
    logger.info("🚀 Starting Sentiment Trading System")
    logger.info("=" * 50)
    
    # Initialize system
    try:
        system = SentimentTradingSystem()
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return
    
    # Run daily pipeline
    logger.info("\n" + "=" * 20 + " DAILY PIPELINE " + "=" * 20)
    daily_results = system.run_daily_pipeline()
    
    if daily_results:
        logger.info(f"✅ Daily pipeline completed successfully!")
        logger.info(f"📊 Summary: {daily_results['articles_processed']} articles, "
                   f"{daily_results['signals_generated']} signals, "
                   f"{daily_results['clicks_used']} clicks used")
    else:
        logger.error("❌ Daily pipeline failed")
    
    # Generate daily report
    logger.info("\n" + "=" * 20 + " DAILY REPORT " + "=" * 20)
    report = system.generate_daily_report()
    
    # Run weekly analysis (if it's Monday)
    if datetime.now().weekday() == 0:  # Monday
        logger.info("\n" + "=" * 20 + " WEEKLY ANALYSIS " + "=" * 20)
        weekly_results = system.run_weekly_analysis()
        
        if weekly_results and weekly_results['backtest_results']['annualized_return'] >= 0.40:
            logger.info("🎉🎉🎉 CONGRATULATIONS! 🎉🎉🎉")
            logger.info("🎯 You've achieved your 40%+ return target!")
            logger.info("🚀 Your sentiment trading system is performing excellently!")
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ Sentiment Trading System execution completed")
    logger.info("📝 Check logs/trading_system.log for detailed information")
  if __name__ == "__main__":
    main()
