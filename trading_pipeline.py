import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sqlite3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class SentimentTradingPipeline:
    """
    Complete pipeline for sentiment-based trading signals
    Integrates sentiment analysis with stock price prediction
    """
    
    def __init__(self, db_path='data/sentiment_data.db'):
        self.db_path = db_path
        self.sentiment_analyzer = None  # Will be loaded from previous model
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Initialize database
        self._init_database()
        
        # Stock universes for different strategies
        self.nifty50_stocks = [
            'TCS.NS', 'RELIANCE.NS', 'HDFC.NS', 'INFY.NS', 'HDFCBANK.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS',
            'ITC.NS', 'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
            'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'BAJFINANCE.NS'
        ]
        
    def _init_database(self):
        """Initialize SQLite database for storing sentiment and price data"""
        conn = sqlite3.connect(self.db_path)
        
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
        
        conn.commit()
        conn.close()
    
    def store_news_sentiment(self, news_data):
        """Store news sentiment data in database"""
        conn = sqlite3.connect(self.db_path)
        
        for item in news_data:
            conn.execute('''
                INSERT OR REPLACE INTO news_sentiment 
                (date, company_symbol, news_text, sentiment_score, confidence_score, 
                 impact_score, method, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['date'], item['symbol'], item['text'], 
                item['sentiment'], item['confidence'], item['impact'],
                item['method'], item.get('source', 'CMIE')
            ))
        
        conn.commit()
        conn.close()
    
    def aggregate_daily_sentiment(self, start_date, end_date):
        """Aggregate news sentiment to daily level for each company"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                date,
                company_symbol,
                AVG(sentiment_score) as avg_sentiment,
                COUNT(*) as sentiment_volume,
                AVG(sentiment_score * confidence_score) as weighted_sentiment,
                STDEV(sentiment_score) as sentiment_volatility,
                AVG(CASE WHEN sentiment_score > 0.1 THEN 1.0 ELSE 0.0 END) as positive_ratio,
                AVG(CASE WHEN sentiment_score < -0.1 THEN 1.0 ELSE 0.0 END) as negative_ratio
            FROM news_sentiment 
            WHERE date BETWEEN ? AND ?
            GROUP BY date, company_symbol
        '''
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        # Store aggregated data
        df.to_sql('daily_sentiment', conn, if_exists='replace', index=False)
        
        conn.close()
        return df
    
    def fetch_stock_data(self, symbols, start_date, end_date):
        """Fetch and store stock price data"""
        conn = sqlite3.connect(self.db_path)
        
        all_data = []
        
        for symbol in symbols:
            try:
                stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not stock.empty:
                    stock_data = stock.reset_index()
                    stock_data['Symbol'] = symbol
                    stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
                    
                    # Rename columns to match database schema
                    stock_data.columns = [
                        'date', 'open_price', 'high_price', 'low_price', 
                        'close_price', 'adj_close', 'volume', 'symbol'
                    ]
                    
                    all_data.append(stock_data)
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data.to_sql('stock_prices', conn, if_exists='replace', index=False)
        
        conn.close()
        return all_data
    
    def create_features(self, start_date, end_date):
        """Create comprehensive feature set combining sentiment and price data"""
        conn = sqlite3.connect(self.db_path)
        
        # Fetch sentiment data
        sentiment_query = '''
            SELECT * FROM daily_sentiment 
            WHERE date BETWEEN ? AND ?
        '''
        sentiment_df = pd.read_sql_query(sentiment_query, conn, params=(start_date, end_date))
        
        # Fetch price data
        price_query = '''
            SELECT * FROM stock_prices 
            WHERE date BETWEEN ? AND ?
        '''
        price_df = pd.read_sql_query(price_query, conn, params=(start_date, end_date))
        
        conn.close()
        
        # Merge sentiment and price data
        merged_df = pd.merge(
            price_df, sentiment_df, 
            left_on=['date', 'symbol'], 
            right_on=['date', 'company_symbol'], 
            how='left'
        )
        
        # Fill missing sentiment values
        sentiment_cols = ['avg_sentiment', 'sentiment_volume', 'weighted_sentiment', 
                         'sentiment_volatility', 'positive_ratio', 'negative_ratio']
        merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
        
        # Create technical indicators
        merged_df = self._add_technical_indicators(merged_df)
        
        # Create sentiment features
        merged_df = self._add_sentiment_features(merged_df)
        
        # Create target variables
        merged_df = self._add_target_variables(merged_df)
        
        return merged_df
    
    def _add_technical_indicators(self, df):
        """Add technical indicators to the dataset"""
        df = df.sort_values(['symbol', 'date']).copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Price-based indicators
            symbol_data['returns'] = symbol_data['close_price'].pct_change()
            symbol_data['log_returns'] = np.log(symbol_data['close_price'] / symbol_data['close_price'].shift(1))
            
            # Moving averages
            symbol_data['sma_5'] = symbol_data['close_price'].rolling(5).mean()
            symbol_data['sma_20'] = symbol_data['close_price'].rolling(20).mean()
            symbol_data['sma_50'] = symbol_data['close_price'].rolling(50).mean()
            
            # Price relative to moving averages
            symbol_data['price_to_sma5'] = symbol_data['close_price'] / symbol_data['sma_5'] - 1
            symbol_data['price_to_sma20'] = symbol_data['close_price'] / symbol_data['sma_20'] - 1
            
            # Volatility
            symbol_data['volatility_5'] = symbol_data['returns'].rolling(5).std()
            symbol_data['volatility_20'] = symbol_data['returns'].rolling(20).std()
            
            # RSI
            symbol_data['rsi'] = self._calculate_rsi(symbol_data['close_price'])
            
            # Volume indicators
            symbol_data['volume_sma'] = symbol_data['volume'].rolling(20).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma']
            
            # Update main dataframe
            df.loc[mask, symbol_data.columns] = symbol_data
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_sentiment_features(self, df):
        """Create advanced sentiment-based features"""
        df = df.sort_values(['symbol', 'date']).copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Sentiment momentum
            symbol_data['sentiment_momentum_3'] = symbol_data['avg_sentiment'].rolling(3).mean()
            symbol_data['sentiment_momentum_7'] = symbol_data['avg_sentiment'].rolling(7).mean()
            
            # Sentiment change
            symbol_data['sentiment_change'] = symbol_data['avg_sentiment'].diff()
            symbol_data['sentiment_acceleration'] = symbol_data['sentiment_change'].diff()
            
            # Sentiment-volume interaction
            symbol_data['sentiment_volume_interaction'] = (
                symbol_data['avg_sentiment'] * np.log1p(symbol_data['sentiment_volume'])
            )
            
            # Sentiment divergence from moving average
            symbol_data['sentiment_sma_7'] = symbol_data['avg_sentiment'].rolling(7).mean()
            symbol_data['sentiment_divergence'] = (
                symbol_data['avg_sentiment'] - symbol_data['sentiment_sma_7']
            )
            
            # Update main dataframe
            df.loc[mask, symbol_data.columns] = symbol_data
        
        # Calculate cross-sectional sentiment features
        df = self._add_cross_sectional_features(df)
        
        return df
    
    def _add_cross_sectional_features(self, df):
        """Add cross-sectional sentiment features"""
        # Calculate market-wide sentiment by date
        market_sentiment = df.groupby('date')['avg_sentiment'].mean().rename('market_sentiment')
        df = df.merge(market_sentiment.to_frame(), left_on='date', right_index=True, how='left')
        
        # Relative sentiment
        df['relative_sentiment'] = df['avg_sentiment'] - df['market_sentiment']
        
        # Sentiment rank within date
        df['sentiment_rank'] = df.groupby('date')['avg_sentiment'].rank(pct=True)
        
        return df
    
    def _add_target_variables(self, df):
        """Create target variables for prediction"""
        df = df.sort_values(['symbol', 'date']).copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Forward returns (targets)
            symbol_data['target_1d'] = symbol_data['close_price'].shift(-1) / symbol_data['close_price'] - 1
            symbol_data['target_3d'] = symbol_data['close_price'].shift(-3) / symbol_data['close_price'] - 1
            symbol_data['target_7d'] = symbol_data['close_price'].shift(-7) / symbol_data['close_price'] - 1
            
            # Binary targets (outperformance)
            market_return_1d = symbol_data['target_1d'].mean()  # Simplified market proxy
            symbol_data['outperform_1d'] = (symbol_data['target_1d'] > market_return_1d).astype(int)
            
            # Update main dataframe
            df.loc[mask, symbol_data.columns] = symbol_data
        
        return df
    
    def prepare_ml_dataset(self, df, target_col='target_1d', min_periods=30):
        """Prepare dataset for machine learning"""
        # Feature columns
        feature_cols = [
            'avg_sentiment', 'sentiment_volume', 'weighted_sentiment', 'sentiment_volatility',
            'positive_ratio', 'negative_ratio', 'sentiment_momentum_3', 'sentiment_momentum_7',
            'sentiment_change', 'sentiment_acceleration', 'sentiment_volume_interaction',
            'sentiment_divergence', 'relative_sentiment', 'sentiment_rank',
            'returns', 'log_returns', 'price_to_sma5', 'price_to_sma20',
            'volatility_5', 'volatility_20', 'rsi', 'volume_ratio'
        ]
        
        # Filter valid data
        valid_data = df.dropna(subset=feature_cols + [target_col])
        
        # Remove stocks with insufficient data
        stock_counts = valid_data.groupby('symbol').size()
        valid_stocks = stock_counts[stock_counts >= min_periods].index
        valid_data = valid_data[valid_data['symbol'].isin(valid_stocks)]
        
        return valid_data, feature_cols
    
    def train_models(self, df, target_col='target_1d'):
        """Train ensemble of ML models for return prediction"""
        print(f"Training models for target: {target_col}")
        
        # Prepare dataset
        ml_data, feature_cols = self.prepare_ml_dataset(df, target_col)
        
        if ml_data.empty:
            print("No valid data for training")
            return None
        
        X = ml_data[feature_cols]
        y = ml_data[target_col]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            print("No valid samples after cleaning")
            return None
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[target_col] = scaler
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Model ensemble
        models = {
            'rf': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                min_samples_split=10, min_samples_leaf=5, random_state=42
            )
        }
        
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_scaled, y, cv=tscv, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            model_scores[name] = -cv_scores.mean()
            
            # Train on full dataset
            model.fit(X_scaled, y)
            trained_models[name] = model
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[f"{name}_{target_col}"] = importance_df
                print(f"Top 5 features for {name}:")
                print(importance_df.head())
        
        self.models[target_col] = trained_models
        
        print(f"Model scores (MSE): {model_scores}")
        return trained_models, model_scores
    
    def generate_signals(self, df, confidence_threshold=0.6):
        """Generate trading signals based on model predictions"""
        signals = []
        
        for target_col in self.models:
            if target_col not in self.models:
                continue
                
            # Prepare data
            ml_data, feature_cols = self.prepare_ml_dataset(df, target_col)
            
            if ml_data.empty:
                continue
            
            X = ml_data[feature_cols]
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Scale features
            if target_col in self.scalers:
                X_scaled = self.scalers[target_col].transform(X)
            else:
                continue
            
            # Get predictions from ensemble
            predictions = {}
            for model_name, model in self.models[target_col].items():
                predictions[model_name] = model.predict(X_scaled)
            
            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
            # Calculate prediction confidence (inverse of standard deviation)
            pred_std = np.std(list(predictions.values()), axis=0)
            confidence = 1 / (1 + pred_std)  # Higher confidence for lower std
            
            # Create signals dataframe
            signal_df = ml_data[['date', 'symbol']].copy()
            signal_df['predicted_return'] = ensemble_pred
            signal_df['confidence'] = confidence
            signal_df['signal_strength'] = ensemble_pred * confidence
            signal_df['target_period'] = target_col
            
            # Filter by confidence threshold
            signal_df = signal_df[signal_df['confidence'] >= confidence_threshold]
            
            signals.append(signal_df)
        
        if signals:
            all_signals = pd.concat(signals, ignore_index=True)
            return all_signals
        else:
            return pd.DataFrame()
    
    def backtest_strategy(self, signals_df, df, holding_period=1):
        """Backtest the sentiment-based trading strategy"""
        # Merge signals with actual returns
        backtest_df = signals_df.merge(
            df[['date', 'symbol', 'target_1d', 'close_price']], 
            on=['date', 'symbol'], 
            how='left'
        )
        
        # Calculate strategy returns
        backtest_df['strategy_return'] = (
            backtest_df['predicted_return'].apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0)) *
            backtest_df['target_1d']
        )
        
        # Portfolio-level performance
        daily_returns = backtest_df.groupby('date')['strategy_return'].mean()
        
        # Performance metrics
        total_return = (1 + daily_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(daily_returns)
        
        win_rate = (daily_returns > 0).mean()
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(backtest_df),
            'daily_returns': daily_returns
        }
        
        return results
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def run_complete_pipeline(self, start_date, end_date, symbols=None):
        """Run the complete sentiment trading pipeline"""
        if symbols is None:
            symbols = self.nifty50_stocks[:10]  # Start with subset for testing
        
        print("=== Sentiment Trading Pipeline ===")
        print(f"Period: {start_date} to {end_date}")
        print(f"Symbols: {len(symbols)} stocks")
        
        # Step 1: Fetch stock data
        print("\n1. Fetching stock price data...")
        self.fetch_stock_data(symbols, start_date, end_date)
        
        # Step 2: Create features (assuming sentiment data is already loaded)
        print("\n2. Creating feature set...")
        feature_df = self.create_features(start_date, end_date)
        
        if feature_df.empty:
            print("No data available for analysis")
            return None
        
        print(f"Created dataset with {len(feature_df)} rows and {len(feature_df.columns)} columns")
        
        # Step 3: Train models
        print("\n3. Training prediction models...")
        models_1d, scores_1d = self.train_models(feature_df, 'target_1d')
        
        # Step 4: Generate signals
        print("\n4. Generating trading signals...")
        signals = self.generate_signals(feature_df)
        
        if signals.empty:
            print("No signals generated")
            return None
        
        print(f"Generated {len(signals)} trading signals")
        
        # Step 5: Backtest strategy
        print("\n5. Backtesting strategy...")
        backtest_results = self.backtest_strategy(signals, feature_df)
        
        print("\n=== Backtest Results ===")
        print(f"Total Return: {backtest_results['total_return']:.2%}")
        print(f"Annualized Return: {backtest_results['annualized_return']:.2%}")
        print(f"Volatility: {backtest_results['volatility']:.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"Total Trades: {backtest_results['total_trades']}")
        
        return {
            'feature_df': feature_df,
            'signals': signals,
            'backtest_results': backtest_results,
            'models': self.models,
            'feature_importance': self.feature_importance
        }

# Usage example
def run_example_pipeline():
    """Example of how to use the sentiment trading pipeline"""
    
    # Initialize pipeline
    pipeline = SentimentTradingPipeline()
    
    # Define date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Example CMIE news data (you would replace this with actual CMIE data)
    example_news_data = [
        {
            'date': '2024-06-15',
            'symbol': 'TCS.NS',
            'text': 'TCS reports strong quarterly results with 12% revenue growth',
            'sentiment': 0.7,
            'confidence': 0.8,
            'impact': 1.2,
            'method': 'ensemble'
        },
        {
            'date': '2024-06-16',
            'symbol': 'RELIANCE.NS',
            'text': 'Reliance faces regulatory scrutiny over acquisition deal',
            'sentiment': -0.5,
            'confidence': 0.7,
            'impact': 1.1,
            'method': 'ensemble'
        }
    ]
    
    # Store sentiment data
    pipeline.store_news_sentiment(example_news_data)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        start_date=start_date,
        end_date=end_date,
        symbols=['TCS.NS', 'RELIANCE.NS', 'INFY.NS', 'HDFC.NS', 'HDFCBANK.NS']
    )
    
    return pipeline, results

if __name__ == "__main__":
    # Run example
    pipeline, results = run_example_pipeline()
    
    if results:
        print("\n=== Feature Importance Analysis ===")
        for model_name, importance_df in pipeline.feature_importance.items():
            print(f"\n{model_name}:")
            print(importance_df.head(10))
