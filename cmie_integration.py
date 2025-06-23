import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import sqlite3
import json
import time
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CMIEDataIntegration:
    """
    Integration module for CMIE Prowess data source
    Handles data extraction, processing, and sentiment analysis integration
    """
    
    def __init__(self, db_path='data/sentiment_data.db'):
        self.db_path = db_path
        self.company_mapping = self._load_company_mapping()
        self.sentiment_analyzer = None  # Load from your advanced model
        
        # CMIE-specific configurations
        self.cmie_config = {
            'rate_limit': 2,  # Seconds between requests to respect your 1701 clicks
            'max_retries': 3,
            'timeout': 30
        }
        
        # Initialize database connection
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables for CMIE data"""
        conn = sqlite3.connect(self.db_path)
        
        # CMIE news articles table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cmie_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                headline TEXT NOT NULL,
                content TEXT,
                companies_mentioned TEXT,  -- JSON list of companies
                url TEXT,
                category TEXT,
                processed BOOLEAN DEFAULT FALSE,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Company name mapping table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS company_mapping (
                cmie_name TEXT PRIMARY KEY,
                nse_symbol TEXT,
                bse_symbol TEXT,
                company_full_name TEXT,
                sector TEXT,
                market_cap_category TEXT
            )
        ''')
        
        # CMIE usage tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cmie_usage (
                date TEXT PRIMARY KEY,
                clicks_used INTEGER DEFAULT 0,
                clicks_remaining INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_company_mapping(self):
        """Load company name mapping between CMIE and stock symbols"""
        # This would contain mappings like "Tata Consultancy Services Ltd." -> "TCS.NS"
        # You'll need to build this mapping based on your CMIE data
        
        default_mapping = {
            # Major IT companies
            "Tata Consultancy Services Ltd.": "TCS.NS",
            "Infosys Ltd.": "INFY.NS",
            "HCL Technologies Ltd.": "HCLTECH.NS",
            "Wipro Ltd.": "WIPRO.NS",
            "Tech Mahindra Ltd.": "TECHM.NS",
            
            # Banks
            "HDFC Bank Ltd.": "HDFCBANK.NS",
            "ICICI Bank Ltd.": "ICICIBANK.NS",
            "State Bank of India": "SBIN.NS",
            "Kotak Mahindra Bank Ltd.": "KOTAKBANK.NS",
            "Axis Bank Ltd.": "AXISBANK.NS",
            
            # Other major companies
            "Reliance Industries Ltd.": "RELIANCE.NS",
            "Tata Motors Ltd.": "TATAMOTORS.NS",
            "Hindustan Unilever Ltd.": "HINDUNILVR.NS",
            "ITC Ltd.": "ITC.NS",
            "Larsen & Toubro Ltd.": "LT.NS",
            "Asian Paints Ltd.": "ASIANPAINT.NS",
            "Maruti Suzuki India Ltd.": "MARUTI.NS",
            "Titan Company Ltd.": "TITAN.NS",
            "UltraTech Cement Ltd.": "ULTRACEMCO.NS",
            "Sun Pharmaceutical Industries Ltd.": "SUNPHARMA.NS"
        }
        
        return default_mapping
    
    def update_company_mapping(self, mapping_dict):
        """Update company mapping in database"""
        conn = sqlite3.connect(self.db_path)
        
        for cmie_name, symbol in mapping_dict.items():
            conn.execute('''
                INSERT OR REPLACE INTO company_mapping 
                (cmie_name, nse_symbol) VALUES (?, ?)
            ''', (cmie_name, symbol))
        
        conn.commit()
        conn.close()
        
        self.company_mapping.update(mapping_dict)
    
    def track_usage(self, clicks_used=1):
        """Track CMIE usage to manage your 1701 click limit"""
        today = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        
        # Get current usage
        cursor = conn.execute(
            'SELECT clicks_used, clicks_remaining FROM cmie_usage WHERE date = ?', 
            (today,)
        )
        result = cursor.fetchone()
        
        if result:
            new_used = result[0] + clicks_used
            new_remaining = max(0, result[1] - clicks_used)
        else:
            new_used = clicks_used
            new_remaining = 1701 - clicks_used  # Your total limit
        
        conn.execute('''
            INSERT OR REPLACE INTO cmie_usage 
            (date, clicks_used, clicks_remaining) 
            VALUES (?, ?, ?)
        ''', (today, new_used, new_remaining))
        
        conn.commit()
        conn.close()
        
        logger.info(f"CMIE usage: {new_used} clicks used today, {new_remaining} remaining")
        return new_remaining
    
    def extract_companies_from_text(self, text):
        """Extract company mentions from news text"""
        companies_found = []
        
        # Look for exact matches in company mapping
        for cmie_name in self.company_mapping:
            # Try both full name and shortened versions
            patterns = [
                cmie_name,
                cmie_name.replace(' Ltd.', '').replace(' Limited', ''),
                cmie_name.split()[0]  # First word only
            ]
            
            for pattern in patterns:
                if re.search(rf'\b{re.escape(pattern)}\b', text, re.IGNORECASE):
                    companies_found.append({
                        'cmie_name': cmie_name,
                        'symbol': self.company_mapping[cmie_name],
                        'match_type': 'exact' if pattern == cmie_name else 'partial'
                    })
                    break
        
        # Remove duplicates
        unique_companies = []
        seen_symbols = set()
        for company in companies_found:
            if company['symbol'] not in seen_symbols:
                unique_companies.append(company)
                seen_symbols.add(company['symbol'])
        
        return unique_companies
    
    def process_cmie_article(self, article_data):
        """Process a single CMIE article"""
        try:
            # Extract text content
            headline = article_data.get('headline', '')
            content = article_data.get('content', '')
            full_text = f"{headline}. {content}"
            
            # Extract company mentions
            companies = self.extract_companies_from_text(full_text)
            
            # Store article in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                INSERT INTO cmie_articles 
                (date, headline, content, companies_mentioned, url, category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                article_data.get('date'),
                headline,
                content,
                json.dumps(companies),
                article_data.get('url'),
                article_data.get('category')
            ))
            
            article_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Analyze sentiment for each company mentioned
            sentiment_results = []
            
            if self.sentiment_analyzer and companies:
                for company in companies:
                    try:
                        # Analyze sentiment with company context
                        sentiment = self.sentiment_analyzer.analyze_news_impact(
                            full_text, 
                            company['cmie_name']
                        )
                        
                        sentiment_results.append({
                            'article_id': article_id,
                            'date': article_data.get('date'),
                            'symbol': company['symbol'],
                            'text': full_text,
                            'sentiment': sentiment['sentiment'],
                            'confidence': sentiment['confidence'],
                            'impact': sentiment['impact_score'],
                            'method': 'ensemble',
                            'source': 'CMIE'
                        })
                        
                    except Exception as e:
                        logger.error(f"Sentiment analysis error for {company['symbol']}: {e}")
            
            return {
                'article_id': article_id,
                'companies_found': len(companies),
                'sentiment_results': sentiment_results
            }
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    def batch_process_cmie_data(self, articles_list):
        """Process multiple CMIE articles efficiently"""
        results = {
            'processed': 0,
            'failed': 0,
            'sentiment_records': [],
            'companies_coverage': {}
        }
        
        for article in articles_list:
            try:
                result = self.process_cmie_article(article)
                
                if result:
                    results['processed'] += 1
                    results['sentiment_records'].extend(result['sentiment_results'])
                    
                    # Track company coverage
                    for sentiment_record in result['sentiment_results']:
                        symbol = sentiment_record['symbol']
                        if symbol not in results['companies_coverage']:
                            results['companies_coverage'][symbol] = 0
                        results['companies_coverage'][symbol] += 1
                else:
                    results['failed'] += 1
                    
                # Rate limiting
                time.sleep(self.cmie_config['rate_limit'])
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                results['failed'] += 1
        
        # Store sentiment results in database
        if results['sentiment_records']:
            self._store_sentiment_batch(results['sentiment_records'])
        
        logger.info(f"Batch processing complete: {results['processed']} processed, {results['failed']} failed")
        logger.info(f"Company coverage: {results['companies_coverage']}")
        
        return results
    
    def _store_sentiment_batch(self, sentiment_records):
        """Store sentiment records in batch"""
        conn = sqlite3.connect(self.db_path)
        
        for record in sentiment_records:
            conn.execute('''
                INSERT OR REPLACE INTO news_sentiment 
                (date, company_symbol, news_text, sentiment_score, confidence_score, 
                 impact_score, method, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['date'], record['symbol'], record['text'],
                record['sentiment'], record['confidence'], record['impact'],
                record['method'], record['source']
            ))
        
        conn.commit()
        conn.close()
    
    def get_priority_companies(self, min_market_cap=1000):
        """Get priority companies for sentiment analysis based on market cap and liquidity"""
        # This would integrate with your stock selection criteria
        priority_companies = [
            'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS',
            'ITC.NS', 'LT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'TECHM.NS'
        ]
        
        return priority_companies
    
    def optimize_data_collection(self, target_companies, days_back=30):
        """Optimize CMIE data collection for your click budget"""
        remaining_clicks = self.track_usage(0)  # Check without using
        
        if remaining_clicks < 10:
            logger.warning("Low on CMIE clicks! Consider upgrading or waiting for reset.")
            return []
        
        # Prioritize recent news for active companies
        collection_plan = {
            'high_priority': target_companies[:10],  # Top 10 companies
            'medium_priority': target_companies[10:25],  # Next 15
            'clicks_per_company': max(1, remaining_clicks // len(target_companies)),
            'max_days_back': min(days_back, 30)
        }
        
        logger.info(f"Collection plan: {collection_plan}")
        return collection_plan
    
    def generate_sentiment_report(self, start_date, end_date):
        """Generate comprehensive sentiment report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get sentiment summary
        sentiment_summary = pd.read_sql_query('''
            SELECT 
                company_symbol,
                COUNT(*) as news_count,
                AVG(sentiment_score) as avg_sentiment,
                AVG(confidence_score) as avg_confidence,
                AVG(impact_score) as avg_impact,
                MIN(date) as first_news,
                MAX(date) as last_news
            FROM news_sentiment 
            WHERE date BETWEEN ? AND ? AND source = 'CMIE'
            GROUP BY company_symbol
            ORDER BY news_count DESC
        ''', conn, params=(start_date, end_date))
        
        # Get daily sentiment trends
        daily_trends = pd.read_sql_query('''
            SELECT 
                date,
                company_symbol,
                AVG(sentiment_score) as daily_sentiment,
                COUNT(*) as daily_news_count
            FROM news_sentiment 
            WHERE date BETWEEN ? AND ? AND source = 'CMIE'
            GROUP BY date, company_symbol
            ORDER BY date, company_symbol
        ''', conn, params=(start_date, end_date))
        
        # Get usage statistics
        usage_stats = pd.read_sql_query('''
            SELECT * FROM cmie_usage 
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', conn, params=(start_date, end_date))
        
        conn.close()
        
        # Calculate insights
        report = {
            'period': f"{start_date} to {end_date}",
            'total_companies': len(sentiment_summary),
            'total_articles': sentiment_summary['news_count'].sum() if not sentiment_summary.empty else 0,
            'avg_sentiment_overall': sentiment_summary['avg_sentiment'].mean() if not sentiment_summary.empty else 0,
            'most_covered_companies': sentiment_summary.head(10),
            'sentiment_distribution': {
                'positive': len(sentiment_summary[sentiment_summary['avg_sentiment'] > 0.1]) if not sentiment_summary.empty else 0,
                'neutral': len(sentiment_summary[abs(sentiment_summary['avg_sentiment']) <= 0.1]) if not sentiment_summary.empty else 0,
                'negative': len(sentiment_summary[sentiment_summary['avg_sentiment'] < -0.1]) if not sentiment_summary.empty else 0
            },
            'daily_trends': daily_trends,
            'usage_stats': usage_stats,
            'click_efficiency': sentiment_summary['news_count'].sum() / usage_stats['clicks_used'].sum() if not usage_stats.empty and usage_stats['clicks_used'].sum() > 0 else 0
        }
        
        return report
    
    def export_for_backtesting(self, start_date, end_date, output_file='cmie_sentiment_data.csv'):
        """Export processed sentiment data for backtesting"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                ns.date,
                ns.company_symbol,
                ns.sentiment_score,
                ns.confidence_score,
                ns.impact_score,
                ca.headline,
                ca.category
            FROM news_sentiment ns
            LEFT JOIN cmie_articles ca ON ns.news_text LIKE '%' || ca.headline || '%'
            WHERE ns.date BETWEEN ? AND ? 
            AND ns.source = 'CMIE'
            ORDER BY ns.date, ns.company_symbol
        '''
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Exported {len(df)} sentiment records to {output_file}")
        
        return df

class CMIEWebScraper:
    """
    Web scraper specifically for CMIE Prowess platform
    Note: Use responsibly and respect rate limits
    """
    
    def __init__(self, cmie_integration):
        self.cmie = cmie_integration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_news_links(self, company_name, max_articles=10):
        """
        Scrape news links for a specific company
        Note: This is a template - you'll need to adapt to CMIE's actual structure
        """
        try:
            # Check remaining clicks
            remaining = self.cmie.track_usage(0)
            if remaining < 5:
                logger.warning("Low on clicks, skipping scraping")
                return []
            
            # This is a placeholder - adapt to actual CMIE URLs and structure
            search_url = f"https://prowess.cmie.com/news/search?company={company_name}"
            
            response = self.session.get(search_url, timeout=30)
            self.cmie.track_usage(1)  # Count this as a click
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract news links (adapt selectors to actual CMIE structure)
                news_links = []
                news_items = soup.find_all('div', class_='news-item')  # Placeholder selector
                
                for item in news_items[:max_articles]:
                    link_elem = item.find('a')
                    date_elem = item.find('span', class_='date')
                    
                    if link_elem and date_elem:
                        news_links.append({
                            'url': link_elem.get('href'),
                            'headline': link_elem.get_text().strip(),
                            'date': date_elem.get_text().strip(),
                            'company': company_name
                        })
                
                return news_links
            
        except Exception as e:
            logger.error(f"Error scraping news for {company_name}: {e}")
            return []
    
    def scrape_article_content(self, article_url):
        """
        Scrape full article content from CMIE
        Note: Adapt to actual CMIE article structure
        """
        try:
            remaining = self.cmie.track_usage(0)
            if remaining < 1:
                return None
            
            response = self.session.get(article_url, timeout=30)
            self.cmie.track_usage(1)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract article content (adapt to actual structure)
                content_elem = soup.find('div', class_='article-content')  # Placeholder
                
                if content_elem:
                    return content_elem.get_text().strip()
            
        except Exception as e:
            logger.error(f"Error scraping article {article_url}: {e}")
            return None

def create_sample_cmie_data():
    """Create sample CMIE data for testing (replace with actual CMIE integration)"""
    sample_articles = [
        {
            'date': '2024-06-15',
            'headline': 'TCS reports strong Q1 FY25 results, revenue up 15%',
            'content': 'Tata Consultancy Services Ltd. reported robust financial results for Q1 FY25 with revenue growth of 15% year-on-year. The company exceeded analyst expectations and maintained strong margins despite global economic headwinds.',
            'url': 'https://prowess.cmie.com/news/tcs-q1-results',
            'category': 'Earnings'
        },
        {
            'date': '2024-06-16',
            'headline': 'Infosys wins major digital transformation deal worth $2B',
            'content': 'Infosys Ltd. secured a significant multi-year digital transformation contract valued at $2 billion from a Fortune 500 client. This deal reinforces the company\'s position in the digital services market.',
            'url': 'https://prowess.cmie.com/news/infosys-deal',
            'category': 'Business'
        },
        {
            'date': '2024-06-17',
            'headline': 'HDFC Bank faces regulatory scrutiny over lending practices',
            'content': 'HDFC Bank Ltd. is under regulatory review by RBI regarding its recent lending practices in the retail segment. The bank has assured full cooperation with the regulatory authorities.',
            'url': 'https://prowess.cmie.com/news/hdfc-regulatory',
            'category': 'Regulatory'
        },
        {
            'date': '2024-06-18',
            'headline': 'Reliance Industries announces major capex plan for green energy',
            'content': 'Reliance Industries Ltd. unveiled a ambitious capital expenditure plan worth ₹75,000 crore for renewable energy projects over the next three years. This includes solar, wind, and hydrogen initiatives.',
            'url': 'https://prowess.cmie.com/news/reliance-green-energy',
            'category': 'Investment'
        },
        {
            'date': '2024-06-19',
            'headline': 'Asian Paints stock tumbles on weak Q1 performance',
            'content': 'Asian Paints Ltd. shares declined 8% after the company reported disappointing Q1 results with flat revenue growth and margin compression due to raw material cost inflation.',
            'url': 'https://prowess.cmie.com/news/asian-paints-results',
            'category': 'Earnings'
        }
    ]
    
    return sample_articles

# Complete integration example
def run_complete_cmie_integration():
    """
    Complete example of CMIE integration with sentiment analysis
    """
    print("=== CMIE Data Integration Pipeline ===")
    
    # Initialize integration
    cmie_integration = CMIEDataIntegration()
    
    # Load sentiment analyzer (from your advanced model)
    try:
        from sentiment_analyzer import AdvancedFinancialSentimentAnalyzer
        cmie_integration.sentiment_analyzer = AdvancedFinancialSentimentAnalyzer()
        print("✓ Sentiment analyzer loaded")
    except ImportError:
        print("⚠ Sentiment analyzer not available - install first")
        return None, None
    
    # Get sample data (replace with actual CMIE data)
    sample_articles = create_sample_cmie_data()
    
    print(f"Processing {len(sample_articles)} articles from CMIE...")
    
    # Process articles
    results = cmie_integration.batch_process_cmie_data(sample_articles)
    
    print(f"Processing Results:")
    print(f"- Articles processed: {results['processed']}")
    print(f"- Articles failed: {results['failed']}")
    print(f"- Sentiment records created: {len(results['sentiment_records'])}")
    print(f"- Companies covered: {list(results['companies_coverage'].keys())}")
    
    # Generate sentiment report
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    report = cmie_integration.generate_sentiment_report(start_date, end_date)
    
    print(f"\n=== Sentiment Report ({report['period']}) ===")
    print(f"Total companies analyzed: {report['total_companies']}")
    print(f"Total articles processed: {report['total_articles']}")
    print(f"Overall average sentiment: {report['avg_sentiment_overall']:.3f}")
    print(f"Click efficiency: {report['click_efficiency']:.2f} articles per click")
    
    print(f"\nSentiment Distribution:")
    print(f"- Positive: {report['sentiment_distribution']['positive']} companies")
    print(f"- Neutral: {report['sentiment_distribution']['neutral']} companies")
    print(f"- Negative: {report['sentiment_distribution']['negative']} companies")
    
    if not report['most_covered_companies'].empty:
        print(f"\nTop 5 Most Covered Companies:")
        top_companies = report['most_covered_companies'].head()
        for _, row in top_companies.iterrows():
            print(f"- {row['company_symbol']}: {row['news_count']} articles, "
                  f"avg sentiment: {row['avg_sentiment']:.3f}")
    
    # Export data for backtesting
    export_df = cmie_integration.export_for_backtesting(start_date, end_date)
    print(f"\nExported {len(export_df)} records for backtesting")
    
    return cmie_integration, results

# Optimization strategies for your 1701 click limit
class CMIEOptimizer:
    """
    Optimize CMIE usage to maximize value from limited clicks
    """
    
    def __init__(self, cmie_integration):
        self.cmie = cmie_integration
    
    def calculate_company_priority(self, symbols):
        """Calculate priority scores for companies based on various factors"""
        priority_scores = {}
        
        for symbol in symbols:
            score = 0
            
            # Market cap weight (higher for large caps)
            if symbol in ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']:
                score += 10  # Large cap
            elif symbol in ['WIPRO.NS', 'TECHM.NS', 'AXISBANK.NS']:
                score += 7   # Mid cap
            else:
                score += 4   # Others
            
            # Sector weight (IT and Banking get higher priority)
            if any(it_symbol in symbol for it_symbol in ['TCS', 'INFY', 'WIPRO', 'TECHM', 'HCL']):
                score += 5  # IT sector
            elif any(bank_symbol in symbol for bank_symbol in ['HDFC', 'ICICI', 'SBIN', 'KOTAK', 'AXIS']):
                score += 5  # Banking sector
            
            # Volatility weight (more volatile stocks might have more sentiment impact)
            # This could be calculated from historical data
            score += 3  # Default volatility weight
            
            priority_scores[symbol] = score
        
        # Sort by priority
        return sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
    
    def optimal_collection_strategy(self, target_return=0.4, available_clicks=1701):
        """
        Design optimal data collection strategy given click constraints
        """
        # Get priority companies
        all_companies = self.cmie.get_priority_companies()
        priority_ranking = self.calculate_company_priority(all_companies)
        
        # Allocation strategy
        strategy = {
            'phase_1_companies': [symbol for symbol, score in priority_ranking[:15]],
            'phase_2_companies': [symbol for symbol, score in priority_ranking[15:30]],
            'clicks_allocation': {
                'phase_1': int(available_clicks * 0.6),  # 60% for top companies
                'phase_2': int(available_clicks * 0.3),  # 30% for second tier
                'reserve': int(available_clicks * 0.1)   # 10% reserve for opportunities
            },
            'collection_frequency': {
                'daily': priority_ranking[:5],    # Top 5 companies daily
                'weekly': priority_ranking[5:15], # Next 10 weekly
                'monthly': priority_ranking[15:]  # Others monthly
            }
        }
        
        print("=== Optimal CMIE Collection Strategy ===")
        print(f"Available clicks: {available_clicks}")
        print(f"Phase 1 companies (daily monitoring): {len(strategy['phase_1_companies'])}")
        print(f"Phase 2 companies (weekly monitoring): {len(strategy['phase_2_companies'])}")
        print(f"Click allocation: {strategy['clicks_allocation']}")
        
        return strategy

if __name__ == "__main__":
    # Run the complete integration
    cmie_integration, results = run_complete_cmie_integration()
    
    if cmie_integration:
        # Optimize collection strategy
        optimizer = CMIEOptimizer(cmie_integration)
        strategy = optimizer.optimal_collection_strategy()
        
        print("\n=== Integration Complete ===")
        print("Your CMIE data integration is ready!")
        print("Next steps:")
        print("1. Connect to actual CMIE Prowess API/scraper")
        print("2. Implement the optimal collection strategy")
        print("3. Run daily sentiment analysis pipeline")
        print("4. Integrate with your trading model")
