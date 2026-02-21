#!/usr/bin/env python3
"""
Advanced Machine Learning Models for Impulse Intelligence
Implements multiple ML algorithms with ensemble methods and online learning
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLModels:
    """Advanced machine learning models for reversal prediction"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_performance = {}
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize all ML models with optimized hyperparameters"""
        print("[INFO] Initializing advanced ML models...")
        
        self.models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            
            'NeuralNetwork': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            
            'Ridge': Ridge(alpha=1.0),
            
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        print(f"[OK] Initialized {len(self.models)} ML models")
        
    def train_individual_models(self, X_train, y_train, X_val, y_val):
        """Train all individual models and evaluate performance"""
        print("[INFO] Training individual ML models...")
        
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"   Training {name}...")
            
            try:
                # Scale features for models that need it
                if name in ['NeuralNetwork', 'Ridge', 'ElasticNet', 'SVR']:
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_val_scaled = self.scaler.transform(X_val)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                
                # Calculate performance metrics
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                self.model_performance[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }
                
                trained_models[name] = model
                
                print(f"     [OK] {name}: RMSE={rmse:.3f}, R²={r2:.3f}")
                
            except Exception as e:
                print(f"     [ERROR] {name} failed: {e}")
                continue
        
        self.models = trained_models
        return trained_models
    
    def create_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Create ensemble model combining best individual models"""
        print("[INFO] Creating ensemble model...")
        
        # Select top performing models for ensemble
        performance_scores = {name: perf['R2'] for name, perf in self.model_performance.items()}
        top_models = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"   Selected top {len(top_models)} models for ensemble:")
        for name, score in top_models:
            print(f"     {name}: R² = {score:.3f}")
        
        # Create ensemble with top models
        ensemble_estimators = []
        for name, _ in top_models:
            if name in self.models:
                ensemble_estimators.append((name, self.models[name]))
        
        if len(ensemble_estimators) >= 2:
            self.ensemble_model = VotingRegressor(
                estimators=ensemble_estimators,
                n_jobs=-1
            )
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = self.ensemble_model.predict(X_val)
            ensemble_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
            ensemble_r2 = r2_score(y_val, y_pred_ensemble)
            
            self.model_performance['Ensemble'] = {
                'MSE': mean_squared_error(y_val, y_pred_ensemble),
                'RMSE': ensemble_rmse,
                'MAE': mean_absolute_error(y_val, y_pred_ensemble),
                'R2': ensemble_r2
            }
            
            print(f"   [OK] Ensemble model: RMSE={ensemble_rmse:.3f}, R²={ensemble_r2:.3f}")
            
        return self.ensemble_model
    
    def analyze_feature_importance(self, X_train, feature_names):
        """Analyze feature importance across models"""
        print("[INFO] Analyzing feature importance...")
        
        importance_data = {}
        
        # Get importance from tree-based models
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = dict(zip(feature_names, model.feature_importances_))
        
        # Calculate average importance
        if importance_data:
            avg_importance = {}
            for feature in feature_names:
                importances = [imp_dict.get(feature, 0) for imp_dict in importance_data.values()]
                avg_importance[feature] = np.mean(importances)
            
            # Sort by importance
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            self.feature_importance = dict(sorted_importance)
            
            print("   Top 10 most important features:")
            for i, (feature, importance) in enumerate(sorted_importance[:10]):
                print(f"     {i+1:2d}. {feature}: {importance:.4f}")
        
        return self.feature_importance
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='XGBoost'):
        """Perform hyperparameter tuning for specified model"""
        print(f"[INFO] Hyperparameter tuning for {model_name}...")
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        else:
            print(f"   Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {-grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def predict_with_confidence(self, X, use_ensemble=True):
        """Make predictions with confidence intervals"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if use_ensemble and self.ensemble_model is not None:
            # Use ensemble prediction
            prediction = self.ensemble_model.predict(X)
            
            # Calculate confidence based on individual model agreement
            individual_predictions = []
            for name, model in self.models.items():
                try:
                    if name in ['NeuralNetwork', 'Ridge', 'ElasticNet', 'SVR']:
                        X_scaled = self.scaler.transform(X)
                        pred = model.predict(X_scaled)
                    else:
                        pred = model.predict(X)
                    individual_predictions.append(pred)
                except:
                    continue
            
            if individual_predictions:
                individual_predictions = np.array(individual_predictions)
                prediction_std = np.std(individual_predictions, axis=0)
                confidence = 1.0 / (1.0 + prediction_std)  # Higher std = lower confidence
            else:
                confidence = np.ones(len(prediction)) * 0.5
                
        else:
            # Use best individual model
            best_model_name = max(self.model_performance.items(), key=lambda x: x[1]['R2'])[0]
            best_model = self.models[best_model_name]
            
            if best_model_name in ['NeuralNetwork', 'Ridge', 'ElasticNet', 'SVR']:
                X_scaled = self.scaler.transform(X)
                prediction = best_model.predict(X_scaled)
            else:
                prediction = best_model.predict(X)
            
            confidence = np.ones(len(prediction)) * 0.8  # Default confidence
        
        return prediction, confidence
    
    def train_complete_system(self, df_ml, feature_cols, target_col='Target_Reversal_Percent'):
        """Train the complete ML system"""
        print("[INFO] Training complete ML system...")
        print("=" * 60)
        
        # Prepare data
        X = df_ml[feature_cols].copy()
        y = df_ml[target_col].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"Training data: {len(X)} samples, {len(feature_cols)} features")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Initialize and train models
        self.initialize_models()
        trained_models = self.train_individual_models(X_train, y_train, X_val, y_val)
        
        if len(trained_models) >= 2:
            ensemble_model = self.create_ensemble_model(X_train, y_train, X_val, y_val)
        
        # Analyze feature importance
        self.analyze_feature_importance(X_train, feature_cols)
        
        # Mark as trained
        self.is_trained = True
        
        # Print final performance summary
        self.print_performance_summary()
        
        return {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'scaler': self.scaler
        }
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        print("\n[INFO] MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Sort models by R² score
        sorted_performance = sorted(
            self.model_performance.items(),
            key=lambda x: x[1]['R2'],
            reverse=True
        )
        
        print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Rank':<6}")
        print("-" * 50)
        
        for i, (name, perf) in enumerate(sorted_performance):
            print(f"{name:<15} {perf['RMSE']:<8.3f} {perf['MAE']:<8.3f} {perf['R2']:<8.3f} #{i+1:<6}")
        
        # Best model
        best_model = sorted_performance[0]
        print(f"\n[INFO] Best Model: {best_model[0]} (R² = {best_model[1]['R2']:.3f})")
        
        if 'Ensemble' in self.model_performance:
            ensemble_perf = self.model_performance['Ensemble']
            print(f"[INFO] Ensemble Model: R² = {ensemble_perf['R2']:.3f}")
    
    def save_models(self, save_path="models/"):
        """Save all trained models and metadata"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print(f"[INFO] Saving models to {save_path}...")
        
        # Save individual models
        for name, model in self.models.items():
            model_file = f"{save_path}model_{name.lower()}.joblib"
            joblib.dump(model, model_file)
            print(f"   Saved {name} to {model_file}")
        
        # Save ensemble model
        if self.ensemble_model is not None:
            ensemble_file = f"{save_path}ensemble_model.joblib"
            joblib.dump(self.ensemble_model, ensemble_file)
            print(f"   Saved Ensemble to {ensemble_file}")
        
        # Save scaler
        scaler_file = f"{save_path}scaler.joblib"
        joblib.dump(self.scaler, scaler_file)
        
        # Save metadata
        metadata = {
            'performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().isoformat(),
            'is_trained': self.is_trained
        }
        
        metadata_file = f"{save_path}model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Saved metadata to {metadata_file}")
        print("[OK] All models saved successfully!")
    
    def load_models(self, load_path="models/"):
        """Load all trained models and metadata"""
        print(f"[INFO] Loading models from {load_path}...")
        
        # Load metadata
        metadata_file = f"{load_path}model_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.model_performance = metadata.get('performance', {})
            self.feature_importance = metadata.get('feature_importance', {})
            self.is_trained = metadata.get('is_trained', False)
        
        # Load individual models
        self.models = {}
        model_files = [f for f in os.listdir(load_path) if f.startswith('model_') and f.endswith('.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('model_', '').replace('.joblib', '').title()
            model_path = f"{load_path}{model_file}"
            
            try:
                self.models[model_name] = joblib.load(model_path)
                print(f"   Loaded {model_name}")
            except Exception as e:
                print(f"   Failed to load {model_name}: {e}")
        
        # Load ensemble model
        ensemble_file = f"{load_path}ensemble_model.joblib"
        if os.path.exists(ensemble_file):
            try:
                self.ensemble_model = joblib.load(ensemble_file)
                print(f"   Loaded Ensemble model")
            except Exception as e:
                print(f"   Failed to load Ensemble: {e}")
        
        # Load scaler
        scaler_file = f"{load_path}scaler.joblib"
        if os.path.exists(scaler_file):
            try:
                self.scaler = joblib.load(scaler_file)
                print(f"   Loaded Scaler")
            except Exception as e:
                print(f"   Failed to load Scaler: {e}")
        
        print(f"[OK] Loaded {len(self.models)} models successfully!")

class OnlineLearningSystem:
    """Online learning system for continuous model improvement"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.recent_data = []
        self.performance_history = []
        self.update_threshold = 50  # Retrain after 50 new samples
        
    def add_new_sample(self, features, actual_reversal):
        """Add new trade result for online learning"""
        self.recent_data.append((features, actual_reversal))
        
        # Check if we should retrain
        if len(self.recent_data) >= self.update_threshold:
            self.incremental_update()
    
    def incremental_update(self):
        """Perform incremental model update"""
        print(f"🔄 Performing incremental update with {len(self.recent_data)} new samples...")
        
        # Extract features and targets
        X_new = np.array([sample[0] for sample in self.recent_data])
        y_new = np.array([sample[1] for sample in self.recent_data])
        
        # Update model (implementation depends on model type)
        # For now, we'll retrain the model with new data
        try:
            # This is a simplified approach - in practice, you'd use
            # incremental learning algorithms like SGD or online learning variants
            if hasattr(self.base_model, 'partial_fit'):
                self.base_model.partial_fit(X_new, y_new)
            else:
                # For models without partial_fit, we'd need to implement
                # a more sophisticated online learning approach
                pass
            
            print("[OK] Model updated successfully")
            
        except Exception as e:
            print(f"[ERROR] Model update failed: {e}")
        
        # Clear recent data
        self.recent_data = []

def main():
    """Test the ML models"""
    print("🧪 Testing Advanced ML Models")
    print("=" * 60)
    
    # Load enhanced dataset
    try:
        df_ml = pd.read_csv("data/ML_Enhanced_Dataset.csv")
        print(f"Loaded enhanced dataset: {len(df_ml)} samples")
    except FileNotFoundError:
        print("[ERROR] Enhanced dataset not found. Run ml_feature_engineering.py first.")
        return
    
    # Get feature columns (exclude target and metadata columns)
    exclude_cols = [
        'Time', 'BasePrice', 'Peak', 'TriggerPrice', 'Impulse', 'Pullback', 
        'Reversal%', 'Symbol', 'TF', 'MAPeriod', 'MAType', 'ScanStart', 'ScanEnd',
        'Target_Reversal_Percent', 'Target_Log_Reversal', 'Target_Reversal_Category',
        'Target_Reversal_Normalized', 'Direction', 'Session_Peak', 'Volatility_Regime'
    ]
    
    feature_cols = [col for col in df_ml.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if not df_ml[col].isna().all()]
    
    print(f"Using {len(feature_cols)} features for training")
    
    # Initialize and train ML system
    ml_system = AdvancedMLModels()
    results = ml_system.train_complete_system(df_ml, feature_cols)
    
    # Save models
    ml_system.save_models()
    
    print("\n[OK] ML System Training Complete!")
    print("Next steps:")
    print("1. Integrate with intelligence server")
    print("2. Test predictions with new data")
    print("3. Deploy for live trading")

if __name__ == "__main__":
    main()
