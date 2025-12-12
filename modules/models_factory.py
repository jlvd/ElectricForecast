import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer

class ProphetModel:
    def __init__(self, holidays=True):
        self.holidays = holidays
        self.model = None
        
    def train(self, df):
        """
        Expects df with 'fecha' and 'valor'.
        Prophet needs 'ds' and 'y'.
        """
        data = df.rename(columns={'fecha': 'ds', 'valor': 'y'})
        
        self.model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        if self.holidays:
            self.model.add_country_holidays(country_name='EC') 
            
        self.model.fit(data)
        
    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        forecast = self.model.predict(future)
        return forecast

class XGBoostModel:
    def __init__(self, use_bayesian_opt=False):
        self.model = None
        self.feature_names = []
        self.use_bayesian_opt = use_bayesian_opt
        self.best_params = {}
        
    def create_features(self, df):
        df = df.copy()
        df['dayofweek'] = df['fecha'].dt.dayofweek
        df['quarter'] = df['fecha'].dt.quarter
        df['month'] = df['fecha'].dt.month
        df['year'] = df['fecha'].dt.year
        df['dayofyear'] = df['fecha'].dt.dayofyear
        return df
        
    def train(self, df):
        df_feats = self.create_features(df)
        
        X = df_feats.drop(columns=['fecha', 'valor'])
        y = df_feats['valor']
        
        self.feature_names = X.columns.tolist()
        
        if self.use_bayesian_opt:
            print("Iniciando Búsqueda Bayesiana de Hiperparámetros...")
            # Define search space
            search_space = {
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'n_estimators': Integer(100, 1000),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0)
            }
            
            xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
            
            # TimeSeriesSplit for CV to respect temporal order
            tscv = TimeSeriesSplit(n_splits=3)
            
            bayes_search = BayesSearchCV(
                estimator=xgb_reg,
                search_spaces=search_space,
                n_iter=20, # Number of parameter settings that are sampled
                cv=tscv,
                n_jobs=-1,
                verbose=0
            )
            
            bayes_search.fit(X, y)
            self.best_params = bayes_search.best_params_
            self.model = bayes_search.best_estimator_
            print(f"Mejores parámetros encontrados: {self.best_params}")
            
        else:
            # Default or simple manual params (could also accept args)
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                max_depth=3,
                learning_rate=0.1
            )
            self.model.fit(X, y)
        
    def predict(self, start_date, periods):
        future_dates = pd.date_range(start=start_date, periods=periods, freq='D')
        future_df = pd.DataFrame({'fecha': future_dates})
        
        future_feats = self.create_features(future_df)
        X_future = future_feats.drop(columns=['fecha'])
        X_future = X_future[self.feature_names]
        
        preds = self.model.predict(X_future)
        
        return pd.DataFrame({'fecha': future_dates, 'yhat': preds})

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}
