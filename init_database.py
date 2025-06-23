# Make sure you're in the project root
cd /root/sentiment-trading

# Create the database initialization script
cat > scripts/init_database.py << 'EOF'
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

def create_database():
    """Create and initialize the sentiment trading database"""
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Create database
    conn = sqlite3.connect('data/sentiment_data.db')
    
    print("Creating database tables...")
    
    # News sentiment table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            company_symbol TEXT NOT NULL,
            news_text TEXT NOT NULL,
            sentiment_score REAL,
            confidence_score REAL,
            impact_score REAL,
            method TEXT,
            source TEXT DEFAULT 'CMIE',
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Daily aggregated sentiment table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            date TEXT NOT NULL,
            company_symbol TEXT NOT NULL,
            avg_sentiment REAL,
            sentiment_volume INTEGER,
            weighted_sentiment REAL,
            sentiment_volatility REAL,
            positive_ratio REAL,
            negative_ratio REAL,
            PRIMARY KEY (date, company_symbol)
        )
    ''')
    
    # Stock price data table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            adj_close REAL,
            PRIMARY KEY (date, symbol)
        )
    ''')
    
    # Trading signals table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trading_signals (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_strength REAL,
            predicted_return REAL,
            confidence REAL,
            signal_type TEXT,
            features TEXT,
            model_version TEXT,
            PRIMARY KEY (date, symbol)
        )
    ''')
    
    # CMIE specific tables
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cmie_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            headline TEXT NOT NULL,
            content TEXT,
            companies_mentioned TEXT,
            url TEXT,
            category TEXT,
            processed BOOLEAN DEFAULT FALSE,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
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
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cmie_usage (
            date TEXT PRIMARY KEY,
            clicks_used INTEGER DEFAULT 0,
            clicks_remaining INTEGER
        )
    ''')
    
    print("✓ Database tables created")
    
    # Create sample data for testing
    print("Creating sample data...")
    
    companies = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    
    # Sample news sentiment data
    sample_sentiment = []
    start_date = datetime.now() - timedelta(days=30)
    
    # Create realistic sample data
    for i in range(150):  # 150 sample records
        date = (start_date + timedelta(days=i//5)).strftime('%Y-%m-%d')
        company = companies[i % len(companies)]
        
        # Create varied sentiment based on company
        if 'TCS' in company or 'INFY' in company:
            sentiment_base = 0.2  # IT companies generally positive
        elif 'RELIANCE' in company:
            sentiment_base = 0.1  # Mixed sentiment
        else:
            sentiment_base = 0.0  # Neutral
        
        sentiment_score = sentiment_base + np.random.normal(0, 0.3)
        sentiment_score = np.clip(sentiment_score, -1, 1)  # Keep in range
        
        sample_sentiment.append({
            'date': date,
            'company_symbol': company,
            'news_text': f'Sample news for {company} - {["positive", "neutral", "negative"][int((sentiment_score + 1) * 1.5)]} development in business operations',
            'sentiment_score': sentiment_score,
            'confidence_score': np.random.uniform(0.5, 0.9),
            'impact_score': np.random.uniform(0.8, 1.5),
            'method': 'ensemble',
            'source': 'SAMPLE'
        })
    
    # Insert sample data
    for record in sample_sentiment:
        conn.execute('''
            INSERT INTO news_sentiment 
            (date, company_symbol, news_text, sentiment_score, confidence_score, 
             impact_score, method, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record['date'], record['company_symbol'], record['news_text'],
            record['sentiment_score'], record['confidence_score'], record['impact_score'],
            record['method'], record['source']
        ))
    
    print(f"✓ Inserted {len(sample_sentiment)} sample sentiment records")
    
    # Initialize CMIE usage
    today = datetime.now().strftime('%Y-%m-%d')
    conn.execute('''
        INSERT OR REPLACE INTO cmie_usage (date, clicks_used, clicks_remaining)
        VALUES (?, ?, ?)
    ''', (today, 0, 1701))
    
    print("✓ Initialized CMIE usage tracking")
    
    # Insert company mapping
    company_mappings = [
        ('Tata Consultancy Services Ltd.', 'TCS.NS', 'TCS.BO', 'Tata Consultancy Services Limited', 'IT', 'Large Cap'),
        ('Infosys Ltd.', 'INFY.NS', 'INFY.BO', 'Infosys Limited', 'IT', 'Large Cap'),
        ('Reliance Industries Ltd.', 'RELIANCE.NS', 'RELIANCE.BO', 'Reliance Industries Limited', 'Oil & Gas', 'Large Cap'),
        ('HDFC Bank Ltd.', 'HDFCBANK.NS', 'HDFCBANK.BO', 'HDFC Bank Limited', 'Banking', 'Large Cap'),
        ('ICICI Bank Ltd.', 'ICICIBANK.NS', 'ICICIBANK.BO', 'ICICI Bank Limited', 'Banking', 'Large Cap'),
        ('Wipro Ltd.', 'WIPRO.NS', 'WIPRO.BO', 'Wipro Limited', 'IT', 'Large Cap'),
        ('Tech Mahindra Ltd.', 'TECHM.NS', 'TECHM.BO', 'Tech Mahindra Limited', 'IT', 'Large Cap'),
        ('HCL Technologies Ltd.', 'HCLTECH.NS', 'HCLTECH.BO', 'HCL Technologies Limited', 'IT', 'Large Cap'),
        ('State Bank of India', 'SBIN.NS', 'SBIN.BO', 'State Bank of India', 'Banking', 'Large Cap'),
        ('Kotak Mahindra Bank Ltd.', 'KOTAKBANK.NS', 'KOTAKBANK.BO', 'Kotak Mahindra Bank Limited', 'Banking', 'Large Cap')
    ]
    
    for mapping in company_mappings:
        conn.execute('''
            INSERT OR REPLACE INTO company_mapping 
            (cmie_name, nse_symbol, bse_symbol, company_full_name, sector, market_cap_category)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', mapping)
    
    print(f"✓ Inserted {len(company_mappings)} company mappings")
    
    # Create some sample daily sentiment aggregations
    daily_agg_data = []
    for date in pd.date_range(start=start_date, periods=20, freq='D'):
        for company in companies[:3]:  # Just a few companies
            daily_agg_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'company_symbol': company,
                'avg_sentiment': np.random.normal(0.1, 0.3),
                'sentiment_volume': np.random.randint(1, 8),
                'weighted_sentiment': np.random.normal(0.1, 0.2),
                'sentiment_volatility': np.random.uniform(0.1, 0.5),
                'positive_ratio': np.random.uniform(0.3, 0.7),
                'negative_ratio': np.random.uniform(0.1, 0.4)
            })
    
    for record in daily_agg_data:
        conn.execute('''
            INSERT OR REPLACE INTO daily_sentiment 
            (date, company_symbol, avg_sentiment, sentiment_volume, weighted_sentiment,
             sentiment_volatility, positive_ratio, negative_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record['date'], record['company_symbol'], record['avg_sentiment'],
            record['sentiment_volume'], record['weighted_sentiment'], record['sentiment_volatility'],
            record['positive_ratio'], record['negative_ratio']
        ))
    
    print(f"✓ Created {len(daily_agg_data)} daily sentiment aggregations")
    
    conn.commit()
    
    # Verify tables and data
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"✓ Created {len(tables)} database tables:")
    for table in tables:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} records")
    
    conn.close()
    
    print("\n🎉 Database initialization completed successfully!")
    print("📊 Your database is ready with sample data for testing")
    print("🚀 You can now run: python main.py")

def verify_database():
    """Verify database was created correctly"""
    try:
        conn = sqlite3.connect('data/sentiment_data.db')
        
        # Test basic queries
        cursor = conn.execute("SELECT COUNT(*) FROM news_sentiment")
        sentiment_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM company_mapping")
        mapping_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT clicks_remaining FROM cmie_usage ORDER BY date DESC LIMIT 1")
        clicks_result = cursor.fetchone()
        clicks_remaining = clicks_result[0] if clicks_result else 0
        
        conn.close()
        
        print("✅ Database Verification:")
        print(f"  - Sentiment records: {sentiment_count}")
        print(f"  - Company mappings: {mapping_count}")
        print(f"  - CMIE clicks remaining: {clicks_remaining}")
        
        if sentiment_count > 0 and mapping_count > 0:
            print("🎯 Database is ready for trading!")
            return True
        else:
            print("❌ Database setup incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🏗️ Initializing Sentiment Trading Database...")
    print("=" * 50)
    
    create_database()
    
    print("\n" + "=" * 50)
    verify_database()
    
    print("\n📋 Next Steps:")
    print("1. Run: python sentiment_analyzer.py")
    print("2. Run: python cmie_integration.py") 
    print("3. Run: python main.py")
    print("\nHappy Trading! 🚀📈")
EOF
