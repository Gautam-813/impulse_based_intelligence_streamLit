#!/usr/bin/env python3
"""
Advanced Feature Engineering for ML-Enhanced Impulse Intelligence
Creates comprehensive features for machine learning model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta  # Technical Analysis library
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for impulse-reversal prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.session_encoder = LabelEncoder()
        self.direction_encoder = LabelEncoder()
        
    def engineer_features(self, df):
        """Create comprehensive feature set for ML training"""
        print("🔧 Starting advanced feature engineering...")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # 1. Basic Features (calculate from raw data)
        df_features = self._add_basic_features(df_features)
        
        # 2. Time-based Features
        df_features = self._add_time_features(df_features)
        
        # 3. Technical Indicator Features
        df_features = self._add_technical_features(df_features)
        
        # 4. Market Context Features
        df_features = self._add_market_context_features(df_features)
        
        # 5. Historical Performance Features
        df_features = self._add_historical_features(df_features)
        
        # 6. Interaction Features
        df_features = self._add_interaction_features(df_features)
        
        # 7. Encode categorical variables
        df_features = self._encode_categorical_features(df_features)
        
        # 8. Create target variable transformations
        df_features = self._create_target_transformations(df_features)
        
        print(f"🎯 Feature engineering complete! Created {len(df_features.columns)} features")
        return df_features
    
    def _add_basic_features(self, df):
        """Add basic engineered features (ATR-free)"""
        print("   ✅ Basic features: Impulse%, Price_Movement_Pct")
        
        # Impulse% (impulse relative to Entry Price)
        df['Impulse%'] = (df['Impulse'] / df['EntryPrice'].replace(0, 1)) * 100.0
        
        # Price Movement Pct (Peak relative to BasePrice)
        df['Price_Movement_Pct'] = ((df['Peak'] - df['BasePrice']) / df['BasePrice'].replace(0, 1)) * 100.0
        
        return df
    
    def _add_time_features(self, df):
        """Add comprehensive time-based features"""
        print("   🕐 Adding time-based features...")
        
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Basic time features
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Month'] = df['Time'].dt.month
        df['Quarter'] = df['Time'].dt.quarter
        
        # Market session timing
        df['Minutes_Since_Midnight'] = df['Hour'] * 60 + df['Time'].dt.minute
        
        # Session overlap indicators
        df['Tokyo_London_Overlap'] = ((df['Hour'] >= 7) & (df['Hour'] < 9)).astype(int)
        df['London_NY_Overlap'] = ((df['Hour'] >= 12) & (df['Hour'] < 16)).astype(int)
        df['NY_Sydney_Overlap'] = ((df['Hour'] >= 21) | (df['Hour'] < 1)).astype(int)
        
        # Market open/close proximity
        df['Minutes_After_London_Open'] = np.where(df['Hour'] >= 7, 
                                                  (df['Hour'] - 7) * 60 + df['Time'].dt.minute, 
                                                  np.nan)
        df['Minutes_Before_NY_Close'] = np.where(df['Hour'] <= 21,
                                                (21 - df['Hour']) * 60 - df['Time'].dt.minute,
                                                np.nan)
        
        # Cyclical encoding for time features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        return df
    
    def _add_technical_features(self, df):
        """Add technical analysis features"""
        print("   📈 Adding technical indicator features...")
        
        # Create price series for technical analysis
        # Using BasePrice as our main price series
        price_series = df['BasePrice'].copy()
        
        # Moving averages
        df['SMA_5'] = price_series.rolling(window=5).mean()
        df['SMA_20'] = price_series.rolling(window=20).mean()
        df['EMA_12'] = price_series.ewm(span=12).mean()
        df['EMA_26'] = price_series.ewm(span=26).mean()
        
        # Price position relative to moving averages
        df['Price_Above_SMA5'] = (price_series > df['SMA_5']).astype(int)
        df['Price_Above_SMA20'] = (price_series > df['SMA_20']).astype(int)
        df['Price_SMA5_Distance'] = (price_series - df['SMA_5']) / df['SMA_5']
        df['Price_SMA20_Distance'] = (price_series - df['SMA_20']) / df['SMA_20']
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI approximation (using price changes)
        price_change = price_series.diff()
        gain = price_change.where(price_change > 0, 0)
        loss = -price_change.where(price_change < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = price_series.rolling(window=20).mean()
        bb_std = price_series.rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (price_series - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility features
        df['Price_Volatility'] = price_series.rolling(window=20).std()
        
        return df
    
    def _add_market_context_features(self, df):
        """Add market context and regime features"""
        print("   🌍 Adding market context features...")
        
        # Trend strength
        df['Trend_Strength'] = abs(df['BasePrice'].rolling(window=20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        ))
        
        # Simple Volatility regime (based on price std deviation)
        price_std_ma = df['Price_Volatility'].rolling(window=50).mean()
        df['Volatility_Regime'] = np.where(df['Price_Volatility'] > price_std_ma * 1.5, 'HIGH',
                                  np.where(df['Price_Volatility'] < price_std_ma * 0.5, 'LOW', 'NORMAL'))
        
        # Market efficiency (how much price moves relative to volatility)
        df['Market_Efficiency'] = abs(df['BasePrice'].diff()) / df['Price_Volatility'].replace(0, 1)
        
        # Impulse momentum (rate of impulse development - simple)
        df['Impulse_Momentum'] = df['Impulse'] / df['Price_Volatility'].replace(0, 1)
        
        return df
    
    def _add_historical_features(self, df):
        """Add features based on historical performance"""
        print("   📊 Adding historical performance features...")
        
        # Sort by time to ensure proper order
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Recent reversal performance (Conditional)
        if 'Reversal%' in df.columns:
            df['Recent_Avg_Reversal'] = df['Reversal%'].rolling(window=10, min_periods=1).mean()
            df['Recent_Max_Reversal'] = df['Reversal%'].rolling(window=10, min_periods=1).max()
            df['Recent_Min_Reversal'] = df['Reversal%'].rolling(window=10, min_periods=1).min()
            df['Recent_Reversal_Std'] = df['Reversal%'].rolling(window=10, min_periods=1).std()
        else:
            df['Recent_Avg_Reversal'] = 0
            df['Recent_Max_Reversal'] = 0
            df['Recent_Min_Reversal'] = 0
            df['Recent_Reversal_Std'] = 0
        
        # Impulse strength zone recent performance
        for zone in ['SMALL', 'MEDIUM', 'LARGE', 'EXTREME']:
            # Define Impulse% zones
            if zone == 'SMALL':
                zone_mask = df['Impulse%'] < 0.1
            elif zone == 'MEDIUM':
                zone_mask = (df['Impulse%'] >= 0.1) & (df['Impulse%'] < 0.3)
            elif zone == 'LARGE':
                zone_mask = (df['Impulse%'] >= 0.3) & (df['Impulse%'] < 0.6)
            else:  # EXTREME
                zone_mask = df['Impulse%'] >= 0.6
            
            if 'Reversal%' in df.columns:
                zone_data = df[zone_mask]['Reversal%']
                df[f'Recent_Avg_Reversal_{zone}'] = zone_data.rolling(window=5, min_periods=1).mean()
            else:
                df[f'Recent_Avg_Reversal_{zone}'] = 0
        
        # Time since last similar event
        df['Time_Since_Last_Similar'] = 0
        for i in range(1, len(df)):
            similar_mask = (
                (abs(df.loc[:i-1, 'Impulse%'] - df.loc[i, 'Impulse%']) < 0.05) &
                (df.loc[:i-1, 'Session_Peak'] == df.loc[i, 'Session_Peak']) &
                (df.loc[:i-1, 'Direction'] == df.loc[i, 'Direction'])
            )
            if similar_mask.any():
                last_similar_idx = df.loc[:i-1][similar_mask].index[-1]
                time_diff = (df.loc[i, 'Time'] - df.loc[last_similar_idx, 'Time']).total_seconds() / 3600
                df.loc[i, 'Time_Since_Last_Similar'] = time_diff
        
        return df
    
    def _add_interaction_features(self, df):
        """Add interaction features between different variables (ATR-free)"""
        print("   🔗 Adding interaction features...")
        
        # Session and Hour synergy
        df['Session_Hour_Interaction'] = df['Hour'] * df['Session_Peak'].map({
            'TOKYO': 1, 'LONDON': 2, 'NEW YORK': 3, 'SYDNEY': 4
        }).fillna(0)
        
        # Impulse and Technical indicator interactions
        df['Impulse_RSI_Interaction'] = df['Impulse%'] * df['RSI']
        df['Impulse_BB_Position'] = df['Impulse%'] * df['BB_Position']
        
        # Direction and market context
        direction_numeric = df['Direction'].map({'BULLISH': 1, 'BEARISH': -1})
        df['Direction_Trend_Interaction'] = direction_numeric * df['Trend_Strength']
        
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical variables for ML"""
        print("   🏷️ Encoding categorical features...")
        
        # One-hot encode sessions
        session_dummies = pd.get_dummies(df['Session_Peak'], prefix='Session')
        df = pd.concat([df, session_dummies], axis=1)
        
        # One-hot encode volatility regime
        if 'Volatility_Regime' in df.columns:
            volatility_dummies = pd.get_dummies(df['Volatility_Regime'], prefix='VolRegime')
            df = pd.concat([df, volatility_dummies], axis=1)
        
        # Encode direction as numeric
        df['Direction_Numeric'] = df['Direction'].map({'BULLISH': 1, 'BEARISH': 0})
        
        return df
    
    def _create_target_transformations(self, df):
        """Create different target variable transformations"""
        print("   🎯 Creating target transformations...")
        
        if 'Reversal%' in df.columns:
            # Original target
            df['Target_Reversal_Percent'] = df['Reversal%']
            
            # Log-transformed target (for skewed distributions)
            df['Target_Log_Reversal'] = np.log1p(df['Reversal%'])
            
            # Binned target for classification
            df['Target_Reversal_Category'] = pd.cut(df['Reversal%'], 
                                                   bins=[0, 25, 50, 75, float('inf')],
                                                   labels=['LOW', 'MEDIUM', 'HIGH', 'EXTREME'])
            
            # Normalized target (0-1 scale)
            df['Target_Reversal_Normalized'] = df['Reversal%'] / 100.0
        else:
            # Fallback if no reversal% is present
            df['Target_Reversal_Percent'] = 0
            df['Target_Log_Reversal'] = 0
            df['Target_Reversal_Category'] = 'LOW'
            df['Target_Reversal_Normalized'] = 0
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns for ML training"""
        # Exclude non-feature columns
        exclude_cols = [
            'Time', 'BasePrice', 'Peak', 'TriggerPrice', 'Impulse', 'Pullback', 
            'Reversal%', 'Symbol', 'TF', 'MAPeriod', 'MAType', 'ScanStart', 'ScanEnd',
            'Target_Reversal_Percent', 'Target_Log_Reversal', 'Target_Reversal_Category',
            'Target_Reversal_Normalized', 'Direction', 'Session_Peak', 'Volatility_Regime',
            'Session_Base', 'Session_Trigger'  # Add session columns to exclusion
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any columns with all NaN values
        feature_cols = [col for col in feature_cols if not df[col].isna().all()]
        
        # Remove any non-numeric columns
        numeric_cols = []
        for col in feature_cols:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                print(f"   Excluding non-numeric column: {col}")
        
        return numeric_cols
    
    def prepare_ml_dataset(self, df):
        """Prepare final dataset for ML training"""
        print("🎯 Preparing final ML dataset...")
        
        # Engineer all features
        df_ml = self.engineer_features(df)
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df_ml)
        
        # Handle missing values - only for numeric columns
        numeric_feature_cols = [col for col in feature_cols if df_ml[col].dtype in ['int64', 'float64']]
        df_ml[numeric_feature_cols] = df_ml[numeric_feature_cols].fillna(df_ml[numeric_feature_cols].median())
        
        # Remove infinite values - only for numeric columns
        df_ml[numeric_feature_cols] = df_ml[numeric_feature_cols].replace([np.inf, -np.inf], np.nan)
        df_ml[numeric_feature_cols] = df_ml[numeric_feature_cols].fillna(df_ml[numeric_feature_cols].median())
        
        print(f"✅ ML dataset ready: {len(df_ml)} samples, {len(feature_cols)} features")
        
        return df_ml, feature_cols

def main():
    """Test the feature engineering"""
    print("🧪 Testing Advanced Feature Engineering")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv("data/Impulse_Reversal.csv")
    print(f"Loaded {len(df)} records")
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Prepare ML dataset
    df_ml, feature_cols = engineer.prepare_ml_dataset(df)
    
    print(f"\n📊 Feature Engineering Results:")
    print(f"   Original features: {len(df.columns)}")
    print(f"   Engineered features: {len(feature_cols)}")
    print(f"   Total columns: {len(df_ml.columns)}")
    
    print(f"\n🔧 Feature Categories:")
    time_features = [col for col in feature_cols if any(x in col.lower() for x in ['hour', 'day', 'time', 'minute'])]
    technical_features = [col for col in feature_cols if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb_'])]
    atr_features = [col for col in feature_cols if 'atr' in col.lower()]
    session_features = [col for col in feature_cols if 'session' in col.lower()]
    
    print(f"   Time features: {len(time_features)}")
    print(f"   Technical features: {len(technical_features)}")
    print(f"   ATR features: {len(atr_features)}")
    print(f"   Session features: {len(session_features)}")
    
    # Save enhanced dataset
    output_path = "data/ML_Enhanced_Dataset.csv"
    df_ml.to_csv(output_path, index=False)
    print(f"\n💾 Enhanced dataset saved to: {output_path}")
    
    return df_ml, feature_cols

if __name__ == "__main__":
    main()
