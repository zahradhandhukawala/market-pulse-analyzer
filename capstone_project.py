#!/usr/bin/env python
# coding: utf-8

import streamlit as st
ui = st  # alias: previous edits used 'ui', keep compatibility
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import subprocess
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import logging
import requests
import time

# optional Altair for richer charts
try:
    import altair as alt
    alt_available = True
except Exception:
    alt = None
    alt_available = False

# Configure logging early so it's available for the runtime-install logic below
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional pmdarima for ARIMA forecasting
try:
    import pmdarima as pm
    pmd_available = True
except Exception:
    pm = None
    pmd_available = False

# Optional XGBoost: try import, attempt pip install if missing, then re-import
def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier, True
    except Exception:
        return None, False

XGBClassifier, xgb_available = _try_import_xgboost()
if not xgb_available:
    try:
        logger.info("xgboost not found  -  attempting to install via pip")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        # try import again after install
        XGBClassifier, xgb_available = _try_import_xgboost()
        if xgb_available:
            logger.info("xgboost installed and imported successfully")
        else:
            logger.warning("xgboost install completed but import still failed")
    except Exception as e:
        logger.warning("Automatic xgboost install failed: %s", e)
        XGBClassifier = None
        xgb_available = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debugging function for ARIMA
def arima_debug_summary(endog, exog=None, exog_future=None, steps=None, name="series"):
    # Convert to pandas structures if possible so index info is available
    try:
        e = endog if isinstance(endog, (pd.Series, pd.DataFrame)) else pd.Series(endog)
    except Exception:
        e = pd.Series(list(endog))
    logger.info("%s endog length: %s; shape: %s", name, len(e), getattr(e, "shape", None))
    try:
        logger.info("endog index head: %s tail: %s", e.index[:3].tolist(), e.index[-3:].tolist())
    except Exception:
        pass

    if exog is not None:
        try:
            if isinstance(exog, (pd.Series, pd.DataFrame)):
                X = exog
            else:
                X = pd.DataFrame(exog)
        except Exception:
            X = pd.DataFrame(list(exog))
        logger.info("exog length: %s; shape: %s", len(X), getattr(X, "shape", None))
        try:
            logger.info("exog index head: %s tail: %s", X.index[:3].tolist(), X.index[-3:].tolist())
        except Exception:
            pass
        try:
            combined = pd.concat([e, X], axis=1)
            logger.info("combined shape: %s, dropna shape: %s", combined.shape, combined.dropna().shape)
        except Exception as err:
            logger.warning("Could not concat endog and exog for alignment check: %s", err)

    if exog_future is not None:
        try:
            if isinstance(exog_future, (pd.Series, pd.DataFrame)):
                XF = exog_future
            else:
                XF = pd.DataFrame(exog_future)
        except Exception:
            XF = pd.DataFrame(list(exog_future))
        logger.info("exog_future shape: %s", getattr(XF, "shape", None))
        if steps is not None:
            logger.info("forecast steps requested: %s; exog_future rows: %s", steps, XF.shape[0])


# Define helper functions for technical analysis
def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)  # avoid division by zero
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger(close, window=20, stds=2.0):
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + (std * stds)
    lower = ma - (std * stds)
    width = (upper - lower) / ma
    return ma, upper, lower, width

def stochastic(high, low, close, k=14, d=3):
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d_line = k_line.rolling(window=d).mean()
    return k_line, d_line

def atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def roc(close, window=12):
    return close.pct_change(window) * 100

def calculate_technical_indicators(stock_data):
    df = stock_data.copy()

    # normalize common column name cases: prefer Title-case keys used later
    # columns can be non-string (e.g. tuples from MultiIndex), so stringify names for matching
    col_strs = [str(c) for c in df.columns]

    def find_col(token: str):
        """Find a column in df whose stringified name matches token.
        Returns the actual column object (could be tuple) or None."""
        token = token.lower()
        # exact string match first
        for s, c in zip(col_strs, df.columns):
            if s.lower() == token:
                return c
        # then substring match
        for s, c in zip(col_strs, df.columns):
            if token in s.lower():
                return c
        return None

    close_col = find_col('close')
    high_col = find_col('high')
    low_col = find_col('low')
    vol_col = find_col('volume')

    if close_col is not None and 'Close' not in df.columns:
        df['Close'] = df[close_col]
    if high_col is not None and 'High' not in df.columns:
        df['High'] = df[high_col]
    if low_col is not None and 'Low' not in df.columns:
        df['Low'] = df[low_col]
    if vol_col is not None and 'Volume' not in df.columns:
        df['Volume'] = df[vol_col]

    # coerce numeric columns to numeric dtype to avoid unexpected DataFrame objects
    for col in ['Close', 'High', 'Low', 'Volume']:
        if col in df.columns:
            val = df[col]
            # If val is a DataFrame (e.g. MultiIndex where top-level matches), pick a sensible 1-D series
            if isinstance(val, pd.DataFrame):  # FIX: complete isinstance check
                # prefer a column whose name mentions 'adj' or 'close' or 'price'
                chosen = None
                for sub in val.columns:
                    name = str(sub).lower()
                    if 'adj' in name or 'close' in name or 'price' in name:
                        chosen = sub
                        break
                if chosen is None:
                    # fallback: first column
                    chosen = val.columns[0]
                series = val[chosen]
            else:
                series = val
            df[col] = pd.to_numeric(series, errors='coerce')

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (defensive: ensure results are Series)
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    # If rolling returned a DataFrame or 2D ndarray (happens with MultiIndex), pick the first column
    if isinstance(sma, pd.DataFrame):
        sma = sma.iloc[:, 0]
    if isinstance(std, pd.DataFrame):
        std = std.iloc[:, 0]
    # convert numpy 2D arrays to 1D
    try:
        # .squeeze() will turn shape (n,1) ->(n,)
        sma = np.squeeze(sma)
        std = np.squeeze(std)
    except Exception:
        pass
    # coerce to numeric Series
    sma = pd.to_numeric(sma, errors='coerce')
    std = pd.to_numeric(std, errors='coerce')
    df['SMA'] = sma.astype(float)
    df['Upper_Band'] = (sma + 2 * std).astype(float)
    df['Lower_Band'] = (sma - 2 * std).astype(float)

    # EMA indicators
    try:
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    except Exception:
        # defensive fallback in case Close is not numeric
        df['EMA_9'] = pd.to_numeric(df['Close'], errors='coerce').ewm(span=9, adjust=False).mean()
        df['EMA_50'] = pd.to_numeric(df['Close'], errors='coerce').ewm(span=50, adjust=False).mean()
        df['EMA_200'] = pd.to_numeric(df['Close'], errors='coerce').ewm(span=200, adjust=False).mean()

    # VWAP (running cumulative VWAP for daily bars)
    try:
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        v = df['Volume'].fillna(0).astype(float)
        cum_tp_v = (tp * v).fillna(0).cumsum()
        cum_v = v.cumsum().replace(0, np.nan)
        df['VWAP'] = (cum_tp_v / cum_v).fillna(method='ffill')
    except Exception:
        df['VWAP'] = np.nan

    # Additional Features
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Volatility_10d'] = df['Return_1d'].rolling(window=10).std()

    # Lagged returns
    df['ret_lag1'] = df['Return_1d'].shift(1)
    df['ret_lag3'] = df['Return_1d'].rolling(3).sum().shift(1)
    df['ret_lag5'] = df['Return_1d'].rolling(5).sum().shift(1)

    # Next day's return (target)
    df['next_ret'] = df['Close'].pct_change().shift(-1)
    df['next_up'] = (df['next_ret'] >0).astype(int)

    return df

# add helper used by plotting code to get a 1-D numeric Series
def _to_1d(col_name, out_name=None, src=None):
    """Return a 1-D pandas Series for plotting.
    - col_name: column key (or substring) to look up in src (defaults to global analysis_data)
    - out_name: optional name for the Series
    - src: optional DataFrame (if None, uses global analysis_data)
    """
    if src is None:
        src = globals().get('analysis_data', None)
    if src is None or not hasattr(src, 'columns'):
        # no source available ->empty series
        return pd.Series(dtype=float, name=out_name or col_name)

    # tolerant column lookup
    if col_name in src.columns:
        val = src[col_name]
    else:
        match = None
        for c in src.columns:
            s = str(c).lower()
            if s == str(col_name).lower() or str(col_name).lower() in s:
                match = c
                break
        if match is None:
            return pd.Series(dtype=float, index=src.index, name=out_name or col_name)
        val = src[match]

    # if DataFrame, pick first column
    if isinstance(val, pd.DataFrame):
        try:
            val = val.iloc[:, 0]
        except Exception:
            val = val.squeeze()

    # squeeze arrays and coerce to numeric series with datetime index when possible
    try:
        arr = np.squeeze(val)
    except Exception:
        arr = val

    try:
        idx = pd.to_datetime(src.index, errors='coerce')
        if idx.isna().all():
            idx = src.index
    except Exception:
        idx = src.index

    try:
        series = pd.Series(arr, index=idx)
    except Exception:
        series = pd.Series(arr)

    series = pd.to_numeric(series, errors='coerce')
    # drop NaT-index rows for plotting safety
    try:
        if isinstance(series.index, pd.DatetimeIndex):
            series = series.loc[~series.index.isna()]
    except Exception:
        pass

    # truncate very long series
    if len(series) >2000:
        series = series.iloc[-2000:]

    series.name = out_name or col_name
    return series

# --- New: ARIMA forecasting helper (pmdarima) ---
def forecast_arima_next_n(src_df, n_periods=30, seasonal=False):
    """Fit pmdarima.auto_arima to src_df['Close'] and forecast next n_periods business days.
    Returns (model, forecast_df) or (None, None) on failure."""
    if not pmd_available:
        ui.error("pmdarima not available")
        return None, None
    if src_df is None:
        ui.error("No data provided")
        return None, None
    if 'Close' not in src_df.columns:
        ui.error(f"Close column not found. Available: {list(src_df.columns)}")
        return None, None

    try:
        close_data = src_df['Close']
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
        y = pd.to_numeric(close_data, errors='coerce').dropna().astype(float)

        if y.empty:
            ui.error("No valid Close price data after cleaning")
            return None, None
        if len(y) < 50:
            ui.error(f"Insufficient data for ARIMA: {len(y)} (<50)")
            return None, None

        returns = y.pct_change().dropna()
        volatility = returns.std()
        ui.info(f"Volatility (daily return std): {volatility:.4f}")

        # Use returns-based modeling
        ui.info("Using returns-based ARIMA")
        y_transformed = returns * 100  # percent returns

        # Light clipping (preserve variation)
        q_low = y_transformed.quantile(0.10)
        q_high = y_transformed.quantile(0.90)
        pre_std = y_transformed.std()
        y_transformed = y_transformed.clip(q_low, q_high)
        post_std = y_transformed.std()
        ui.info(f"Returns clip range: {y_transformed.min():.3f}% to {y_transformed.max():.3f}% (std {pre_std:.3f}% ->{post_std:.3f}%)")

        # Stationarity test
        from statsmodels.tsa.stattools import adfuller
        try:
            adf_p = adfuller(y_transformed, autolag='AIC')[1]
            ui.info(f"ADF p-value: {adf_p:.4f} ({'Stationary' if adf_p <= 0.05 else 'Non-stationary'})")
        except Exception as e:
            ui.warning(f"ADF test failed: {e}")

        amodel = pm.auto_arima(
            y_transformed,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_d=2,
            trace=False,
            random_state=42,
            n_fits=100,
            information_criterion='aic'
        )
        ui.success(f"ARIMA fitted: {amodel.order}")
        try:
            ui.info(f"AIC: {amodel.aic():.2f}")
        except:
            pass

        preds_transformed, conf_int_transformed = amodel.predict(n_periods=n_periods, return_conf_int=True)

        # Optional simulation blending for variation
        try:
            sims = []
            for _ in range(10):
                sims.append(amodel.simulate(n_periods=n_periods, random_state=np.random.randint(0, 10_000)))
            sim_mean = np.mean(sims, axis=0)
            sim_std = np.std(sims, axis=0)
            preds_transformed = 0.7 * preds_transformed + 0.3 * sim_mean
            conf_int_transformed = np.column_stack([
                preds_transformed - 1.96 * sim_std,
                preds_transformed + 1.96 * sim_std
            ])
            ui.info("Applied simulation blending for forecast variation")
        except Exception as e:
            ui.info(f"Simulation blending skipped: {e}")

        preds_returns = np.asarray(preds_transformed).ravel() / 100.0
        conf_int_returns = np.asarray(conf_int_transformed) / 100.0

        if preds_returns.std() < 0.001:
            ui.warning("Flat forecast detected; injecting minimal noise based on volatility")
            noise_sd = volatility * 0.3
            noise = np.random.normal(0, noise_sd, size=len(preds_returns))
            preds_returns = preds_returns + noise
            conf_int_returns[:, 0] = preds_returns - 1.96 * noise_sd
            conf_int_returns[:, 1] = preds_returns + 1.96 * noise_sd

        ui.info(f"Return forecast range: {preds_returns.min()*100:.3f}% to {preds_returns.max()*100:.3f}% (std {preds_returns.std()*100:.3f}%)")

        last_price = float(y.iloc[-1])
        preds = np.zeros(n_periods)
        conf_int = np.zeros((n_periods, 2))
        for i in range(n_periods):
            base = last_price if i == 0 else preds[i-1]
            preds[i] = base * (1 + preds_returns[i])
            conf_int[i, 0] = (conf_int[i-1, 0] if i >0 else last_price) * (1 + conf_int_returns[i, 0])
            conf_int[i, 1] = (conf_int[i-1, 1] if i >0 else last_price) * (1 + conf_int_returns[i, 1])

        price_var_pct = (preds.max() - preds.min()) / last_price * 100
        # Use a single-line, clearly formatted message to avoid Streamlit
        # rendering parts on separate lines in the expander/log.
        price_msg = f"Price forecast range: ${preds.min():.2f} — ${preds.max():.2f} ({price_var_pct:.1f}% span)"
        ui.info(price_msg)

        if price_var_pct < 1.0:
            ui.warning(f"Low variation ({price_var_pct:.1f}%). Limited directional signal.")

        # Future index (business days)
        try:
            last_idx = src_df.index.max() if hasattr(src_df.index, 'max') else pd.to_datetime(y.index[-1])
            if not isinstance(last_idx, pd.Timestamp):
                last_idx = pd.to_datetime(last_idx)
            start = last_idx + pd.tseries.offsets.BDay(1)
            future_index = pd.bdate_range(start=start, periods=n_periods)
        except Exception as e:
            ui.warning(f"Business day index failed ({e}); using daily index")
            future_index = pd.date_range(start=pd.Timestamp.now().normalize(), periods=n_periods, freq='D')

        forecast_df = pd.DataFrame(
            {
                'Forecast': preds.astype(float),
                'Lower': conf_int[:, 0].astype(float),
                'Upper': conf_int[:, 1].astype(float)
            },
            index=future_index
        )
        forecast_df['Daily_Change'] = forecast_df['Forecast'].pct_change() * 100
        forecast_df['Cumulative_Change'] = ((forecast_df['Forecast'] / last_price) - 1) * 100

        return amodel, forecast_df

    except Exception as e:
        ui.error(f"ARIMA forecast failed: {e}")
        import traceback
        with ui.expander("ARIMA Traceback"):
            ui.text(traceback.format_exc())
        return None, None

def train_models(data, selected_features, selected_model_names, available_models):
    """Train selected models using specified features with hyperparameter tuning."""
    # Remove any rows with NaN values
    data = data.dropna()
    
    if len(data) < 100:  # Need sufficient data for training
        return None
    
    # Sidebar log: redirect info/warning/success while training runs
    _log_panel = st.sidebar.expander("Model Training Log", expanded=True)
    _st_info, _st_warn, _st_succ = st.info, st.warning, st.success
    _ui_info, _ui_warn, _ui_succ = ui.info, ui.warning, ui.success
    try:
        st.info = _log_panel.info
        st.warning = _log_panel.warning
        st.success = _log_panel.success
        ui.info = _log_panel.info
        ui.warning = _log_panel.warning
        ui.success = _log_panel.success

        ui.info(f"Training on {len(data)} samples after removing NaN values")
        ui.info(f"Target distribution - Up: {data['next_up'].sum()}, Down: {len(data) - data['next_up'].sum()}")
    
        # Check for potential data leakage
        if 'next_ret' in selected_features:
            ui.error("! DATA LEAKAGE DETECTED: 'next_ret' should not be used as a feature!")
            selected_features = [f for f in selected_features if f != 'next_ret']
            ui.info("Automatically removed 'next_ret' from features")
    
        # Feature selection to reduce overfitting - limit to top 10 most important features
        if len(selected_features) >10:
            ui.info(f"Too many features ({len(selected_features)}). Selecting top 10 to prevent overfitting.")
            from sklearn.feature_selection import SelectKBest, f_classif
            
            # Prepare data for feature selection
            X_temp = data[selected_features].fillna(0)
            y_temp = data['next_up']
            
            # Select top 10 features
            selector = SelectKBest(score_func=f_classif, k=10)
            X_temp_selected = selector.fit_transform(X_temp, y_temp)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_features = [feat for feat, selected in zip(selected_features, selected_mask) if selected]
            ui.info(f"Selected features: {selected_features}")
    
        # Split data into training and testing (80/20)
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Show split information
        ui.info(f"Train/Test split: {len(train_data)} train, {len(test_data)} test samples")
        train_up_pct = train_data['next_up'].mean()
        test_up_pct = test_data['next_up'].mean()
        ui.info(f"Target balance - Train: {train_up_pct:.1%} up, Test: {test_up_pct:.1%} up")
        
        # Prepare features and target
        X_train = train_data[selected_features]
        # target: whether next day's return is up
        y_train = train_data['next_up']
        X_test = test_data[selected_features]
        y_test = test_data['next_up']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter grids for each model - MORE REGULARIZED
        # Hyperparameter grids for each model - OPTIMIZED FOR SPEED
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'max_features': ['sqrt'],
                'class_weight': ['balanced']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5],
                'subsample': [0.9]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'class_weight': ['balanced']
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.9]
            }
        }
        
        # Train and evaluate selected models
        results = {}
        best_model = None
        best_auc = 0
        best_model_name = None

        # Train other selected models
        for name in selected_model_names:
            model = available_models.get(name)
            if model is None:
                st.warning(f'Model {name} not found in available_models; skipping')
                continue

            st.info(f"Training and tuning {name}...")
            
            # Get parameter grid for this model
            param_grid = param_grids.get(name, {})
            
            # Use hyperparameter tuning for models with parameter grids
            if param_grid and len(X_train) >200:  # Only tune for datasets with sufficient size
                try:
                    # Use RandomizedSearchCV for faster hyperparameter search
                    # Use RandomizedSearchCV for faster hyperparameter search
                    n_iter = min(10, len(X_train) // 20)  # REDUCED: Max 10 iterations
                    cv_folds = 3  # FIXED: Always 3 folds for speed
                    
                    st.info(f"Running hyperparameter search with {n_iter} iterations and {cv_folds}-fold CV...")
                    
                    search = RandomizedSearchCV(
                        model, 
                        param_grid, 
                        n_iter=n_iter,
                        cv=cv_folds,
                        scoring='roc_auc',
                        random_state=42,
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    search.fit(X_train_scaled, y_train)
                    model = search.best_estimator_
                    
                    # Display best parameters
                    best_params_str = ', '.join([f"{k}={v}" for k, v in search.best_params_.items()])
                    st.success(f"[OK] {name} tuned - Best params: {best_params_str}")
                    st.info(f"Best CV score: {search.best_score_:.3f}")
                    
                except Exception as e:
                    st.warning(f"! Hyperparameter tuning failed for {name}: {str(e)}. Using default parameters.")
                    # Fall back to training with default parameters
                    model.fit(X_train_scaled, y_train)
            else:
                # Train model with default parameters for small datasets or models without param grids
                if len(X_train) <= 200:
                    st.info(f"Dataset size ({len(X_train)}) too small for hyperparameter tuning. Using default parameters.")
                model.fit(X_train_scaled, y_train)

            # Cross-validation score (use stratified split)
            st.info(f"Evaluating {name} with cross-validation...")
            try:
                cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_splitter, scoring='roc_auc')
            except Exception:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')

            # Predictions on test set (probabilities)
            predictions = model.predict_proba(X_test_scaled)[:, 1]

            # Choose threshold that maximizes F1 on validation set
            threshold_used = 0.5
            pred_class = (predictions > threshold_used).astype(int)
            try:
                if y_test.nunique() == 2:
                    prec, rec, th = precision_recall_curve(y_test, predictions)
                    # precision_recall_curve returns len(th)+1 points; align f1 with thresholds
                    if th.size >0:
                        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
                        best_idx = int(np.argmax(f1s))
                        threshold_used = float(th[best_idx])
                        pred_class = (predictions > threshold_used).astype(int)
                        st.info(f"{name}: selected decision threshold for max F1 = {threshold_used:.3f}")
                else:
                    st.warning(f"{name}: test set has a single class; F1 is not informative.")
            except Exception as _e:
                # fall back to 0.5
                pass

            # Warn if model predicts a single class
            if pred_class.sum() == 0 or pred_class.sum() == len(pred_class):
                st.warning(f"{name}: predicts a single class at threshold {threshold_used:.3f} (F1 may be 0).")

            # Calculate metrics with selected threshold
            metrics = {
                'accuracy': accuracy_score(y_test, pred_class),
                'precision': precision_score(y_test, pred_class, zero_division=0),
                'recall': recall_score(y_test, pred_class, zero_division=0),
                'f1': f1_score(y_test, pred_class, zero_division=0),
                'auc': roc_auc_score(y_test, predictions),
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_test, pred_class).tolist(),
                'threshold_used': threshold_used
            }

            # Add diagnostic warnings for suspicious results
            if metrics['precision'] == 1.0:
                st.warning(f"! {name}: 100% precision detected - possible overfitting or data leakage!")
            if metrics['accuracy'] >0.95:
                st.warning(f"! {name}: Very high accuracy ({metrics['accuracy']:.1%}) - verify results!")
            
            # Show detailed confusion matrix breakdown
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            st.info(f"{name} detailed results:")
            st.info(f"  True Negatives: {tn}, False Positives: {fp}")
            st.info(f"  False Negatives: {fn}, True Positives: {tp}")
            if tp + fp >0:
                st.info(f"  Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {tp/(tp+fp):.3f}")

            results[name] = metrics
            
            # Display model performance
            st.success(f"[OK] {name} completed - AUC: {metrics['auc']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

            # Track best model
            if metrics['auc'] >best_auc:
                best_auc = metrics['auc']
                best_model = model
                best_model_name = name
        
        # Get feature importance for the best model if available
        feature_importance = {}
        if best_model is not None and hasattr(best_model, 'feature_importances_'):
            importance_dict = dict(zip(selected_features, best_model.feature_importances_))
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif best_model is not None and hasattr(best_model, 'coef_'):
            # For logistic regression, use coefficient magnitudes
            importance_dict = dict(zip(selected_features, abs(best_model.coef_[0])))
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Display best model summary
        if best_model_name:
            st.success(f"< Best model: {best_model_name} with AUC: {best_auc:.3f}")

        # include ARIMA info in return so UI can always show it
        return {
             'scaler': scaler,
             'best_model': best_model,
             'best_model_name': best_model_name,
             'all_results': results,
             'feature_importance': feature_importance,
             'selected_features': selected_features
         }
    finally:
        # Restore original Streamlit functions so non-training UI renders in main area
        st.info, st.warning, st.success = _st_info, _st_warn, _st_succ
        ui.info, ui.warning, ui.success = _ui_info, _ui_warn, _ui_succ

def validate_data_with_alphavantage(yf_data, ticker):
    """Validate yfinance data by comparing daily closes to Alpha Vantage.

    Returns True when data is considered valid (within threshold), False when
    the difference exceeds the configured threshold. External failures are
    treated as non-fatal and return True so the app can continue.
    """
  # Use Streamlit secrets if available, otherwise fallback to hardcoded key
    try:
        av_api_key = st.secrets.get("ALPHA_VANTAGE_KEY", "08DDSNK5LAM338H2")
    except Exception:
        # Secrets not available (local run without .streamlit/secrets.toml)
        av_api_key = "08DDSNK5LAM338H2"

    try:
        ts = TimeSeries(key=av_api_key, output_format='pandas')
        av_data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
    except Exception as e:
        st.warning(f"Could not retrieve data from Alpha Vantage for validation: {e}")
        return True

    if av_data is None or av_data.empty:
        st.warning(f"Alpha Vantage returned an empty dataset for {ticker}. Skipping validation.")
        return True

    # Normalize AV frame
    av_data.index = pd.to_datetime(av_data.index)
    av_data = av_data.rename(columns={
        '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
        '4. close': 'Close', '5. volume': 'Volume'
    }).sort_index()

    # Ensure yf_data index is datetime-like
    try:
        yf_index = pd.to_datetime(yf_data.index)
    except Exception:
        yf_index = yf_data.index

    # Align on common dates
    common_dates = yf_index.intersection(av_data.index)
    if common_dates.empty:
        st.warning("No common dates found between Yahoo Finance and Alpha Vantage data for comparison.")
        return True

    yf_subset = yf_data.loc[common_dates]
    av_subset = av_data.loc[common_dates]

    if 'Close' not in yf_subset.columns or 'Close' not in av_subset.columns:
        st.warning("Close price data not available in one or both datasets for validation.")
        return True

    # compute percent difference (in percent units)
    def _to_1d_series(x):
        # Accept Series, DataFrame, ndarray and return a 1-D Series
        if isinstance(x, pd.DataFrame):
            # prefer a column whose name mentions 'close' or 'adj'
            chosen = None
            for c in x.columns:
                name = str(c).lower()
                if 'close' in name or 'adj' in name:
                    chosen = c
                    break
            if chosen is None:
                chosen = x.columns[0]
            s = x[chosen]
        else:
            s = x

        # If it's a numpy array, squeeze to 1-D
        try:
            if hasattr(s, 'shape') and getattr(s, 'ndim', None) is not None and s.ndim == 2:
                s = np.squeeze(s)
        except Exception:
            pass

        # Finally coerce to pandas Series
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors='coerce').astype(float)
        else:
            return pd.to_numeric(pd.Series(s), errors='coerce').astype(float)

    yf_close = _to_1d_series(yf_subset['Close'])
    av_close = _to_1d_series(av_subset['Close'])
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        close_diff_pct = (yf_close - av_close) / av_close * 100

    # threshold in percent
    validation_threshold_pct = 1.0

    diff_mask = (close_diff_pct.abs() >validation_threshold_pct)
    exceeded = bool(diff_mask.any()) if hasattr(diff_mask, 'any') else bool(diff_mask)

    if exceeded:
        st.warning(f"Validation failed for {ticker}: close price differs from AlphaVantage by more than {validation_threshold_pct}%")
        return False

    st.success("Yahoo Finance data successfully validated against Alpha Vantage.")
    return True

# Set page title
st.title('Market Pulse Analyzer')

# Define a list of stock tickers and time period
# Robust loader: try GitHub raw CSV first, then a local `SP500.csv` next to this script,
# then fall back to a small built-in list if both fail.
try:
    import os
    # potential sources to try (GitHub raw then local file next to this script)
    github_raw = r'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    local_path = os.path.join(os.path.dirname(__file__), 'SP500.csv')

    _sp = None
    # Try GitHub first (network may fail in some environments)
    try:
        _sp = pd.read_csv(github_raw)
    except Exception:
        _sp = None

    # If GitHub failed or produced an empty frame, try local file
    if (_sp is None or getattr(_sp, 'empty', True)) and os.path.exists(local_path):
        try:
            _sp = pd.read_csv(local_path)
        except Exception:
            _sp = None

    if _sp is None or getattr(_sp, 'empty', True):
        st.warning('Could not load S&P 500 constituent list from GitHub or local file; using fallback list.')
        stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        stock_names = ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation', 'Amazon.com, Inc.', 'Meta Platforms, Inc.']
    else:
        # Normalize and pick appropriate columns
        if 'Symbol' in _sp.columns and 'Name' in _sp.columns:
            stock_tickers = _sp['Symbol'].astype(str).str.strip().tolist()
            stock_names = _sp['Name'].astype(str).str.strip().tolist()
        else:
            stock_tickers = _sp.iloc[:, 0].astype(str).str.strip().tolist()
            stock_names = _sp.iloc[:, 1].astype(str).str.strip().tolist()
except Exception:
    # final fallback
    stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    stock_names = ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation', 'Amazon.com, Inc.', 'Meta Platforms, Inc.']

# combined display strings
stock_display = [f"{s} | {n}" for s, n in zip(stock_tickers, stock_names)]
final_stock_tickers = dict(zip(stock_tickers, stock_names))

end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 2)  # 2 years of data

# Create a selectbox for ticker selection with a placeholder
# show combined display strings (e.g. "AAPL  -  Apple Inc.") in the UI
selected_ticker = st.selectbox('Select a Stock Ticker:', ['Select a ticker...'] + stock_display, key='ticker_selectbox')

# helper: extract SYMBOL from a display string like 'SYMBOL | Company Name'
def _extract_symbol(display_str: str):
    if not isinstance(display_str, str):
        return display_str
    if '|' in display_str:
        return display_str.split('|', 1)[0].strip()
    return display_str.strip()

# Only show content if a real ticker is selected
if selected_ticker != 'Select a ticker...':
    # Show controls in sidebar
    st.sidebar.header('Analysis Settings')
    # Indicate whether pmdarima is available in the running interpreter
    # Only show a note when pmdarima is not available
    pmdarima_available = pmd_available
    if not pmdarima_available:
        st.sidebar.info('pmdarima not detected  -  ARIMA will be skipped at training time')
    
    # Date range selection
    st.sidebar.subheader('Date Range')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # Default to 2 years of data for better model training
    
    # Define all available features
    all_features = {
        'Technical Indicators': {
            'RSI': 'Relative Strength Index',
            'MACD': 'Moving Average Convergence Divergence',
            'Signal': 'MACD Signal Line',
            'Upper_Band': 'Bollinger Upper Band',
            'Lower_Band': 'Bollinger Lower Band',
            'SMA': 'Simple Moving Average',
            'EMA_9': 'Exponential Moving Average (9)',
            'EMA_50': 'Exponential Moving Average (50)',
            'EMA_200': 'Exponential Moving Average (200)',
            'VWAP': 'Volume Weighted Average Price'
        },
        'Price Returns': {
            'Return_1d': '1-Day Return',
            'Return_5d': '5-Day Return',
            'Return_10d': '10-Day Return',
            'Volatility_10d': '10-Day Volatility'
        },
        'Lagged Features': {
            'ret_lag1': 'Previous Day Return',
            'ret_lag3': '3-Day Cumulative Return',
            'ret_lag5': '5-Day Cumulative Return'
        }
    }
    
    # Feature selection
    st.sidebar.subheader('Select Features')
    feature_group_options = ['All'] + list(all_features.keys())
    selected_feature_groups = st.sidebar.multiselect(
        'Feature Groups',
        options=feature_group_options,
        default=['All']
    )

    # If 'All' selected (or no selection), expand to all groups; otherwise use chosen groups
    if not selected_feature_groups or 'All' in selected_feature_groups:
        groups_to_use = list(all_features.keys())
    else:
        groups_to_use = selected_feature_groups

    # Flatten selected features
    selected_features = []
    for group in groups_to_use:
        selected_features.extend(list(all_features[group].keys()))
    
    # Model selection
    st.sidebar.subheader('Select Models')
    available_models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    # Add XGBoost if available; otherwise show as "XGBoost (missing)" in the options
    if xgb_available:
        available_models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        available_models['XGBoost (missing)'] = None

    # Create default selection with all ML models (excluding non-functional ones)
    default_models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
    if xgb_available:
        default_models.append('XGBoost')

    selected_models = st.sidebar.multiselect(
        'Models to Train',
        options=list(available_models.keys()),
        default=default_models  # Default to all available ML models
    )
    
    start_date = st.sidebar.date_input('Start Date', start_date)
    end_date = st.sidebar.date_input('End Date', end_date)

    # Model training status panel: shows last trained info without triggering training
    with st.sidebar.expander("Model Training Status", expanded=False):
        last_time = st.session_state.get('last_trained_time', None)
        last_ticker = st.session_state.get('last_trained_ticker', None)
        last_start = st.session_state.get('last_trained_start', None)
        last_end = st.session_state.get('last_trained_end', None)
        if last_time:
            st.write(f"**Last trained:** {last_time}")
            st.write(f"**Ticker:** {last_ticker}")
            st.write(f"**Range:** {last_start} — {last_end}")
        else:
            st.write("Models not trained yet.")

    # Chart overlay options (user can toggle indicators to show on the price chart)
    st.sidebar.subheader('Chart Overlays')
    show_bollinger = st.sidebar.checkbox('Bollinger Bands (Upper / Lower)', value=True)
    show_sma = st.sidebar.checkbox('SMA (20)', value=True)
    show_ema9 = st.sidebar.checkbox('EMA 9', value=False)
    show_ema50 = st.sidebar.checkbox('EMA 50', value=False)
    show_ema200 = st.sidebar.checkbox('EMA 200', value=True)
    show_vwap = st.sidebar.checkbox('VWAP', value=True)
    
    # Show loading message while fetching data
    # Use the display string for UI, but query yfinance with the extracted symbol
    selected_symbol = _extract_symbol(selected_ticker)
    with st.spinner(f'Loading data for {selected_ticker}...'):
        try:
            # Get stock data only for selected ticker
            stock_data = yf.download(selected_symbol, start=start_date, end=end_date)
            stock_data.index.name = 'date'  # Ensure the index is named 'date'
            
            if not stock_data.empty:
                # Reset index to make date a column
                # DISABLED: Alpha Vantage validation to avoid rate limits on cloud deployment
                # validation_success = validate_data_with_alphavantage(stock_data, selected_symbol)
                validation_success = True  # Skip validation - use Yahoo Finance data directly
                
                stock_data = stock_data.reset_index()
                # Calculate all technical indicators and prepare data
                analysis_data = calculate_technical_indicators(stock_data)
                model_results = None

                

                # Ensure analysis_data has a proper datetime index for plotting:
                # convert 'date' column to datetime, set as index, drop rows with NaT index,
                # sort and name the index so plotting libraries format x-axis as dates.
                if 'date' in analysis_data.columns:
                    try:
                        # coerce date column to datetime and drop rows without a valid date
                        analysis_data['date'] = pd.to_datetime(analysis_data['date'], errors='coerce')
                        analysis_data = analysis_data.loc[~analysis_data['date'].isna()].copy()
                        if not analysis_data.empty:
                            analysis_data.set_index('date', inplace=True)
                            # ensure index is datetime and drop NaT-index rows if any
                            analysis_data.index = pd.to_datetime(analysis_data.index, errors='coerce')
                            analysis_data = analysis_data.loc[~analysis_data.index.isna()]
                            # sort and name index for plotting consistency
                            analysis_data.sort_index(inplace=True)
                            analysis_data.index.name = 'date'
                    except Exception:
                        # non-fatal: leave analysis_data unchanged if something goes wrong
                        pass

                # Check if we have selections
                # MODEL TRAINING CONTROL
                # Do not retrain models on simple UI overlay toggles. Provide explicit
                # controls for training and cache results in session state.
                model_results = st.session_state.get('model_results', None)

                # Allow user to choose explicit training behavior. Remove the explicit
                # "Train Models" button so training only runs when auto-train is enabled
                # or when cached results are stale. This prevents overlay toggles from
                # retraining models.
                auto_train = st.sidebar.checkbox('Auto-train models on data load', value=False, key='auto_train')

                if not selected_models:
                    ui.warning("Please select at least one model from the sidebar.")
                    model_results = None
                elif not selected_features:
                    ui.warning("Please select at least one feature group from the sidebar.")
                    model_results = None
                else:
                    should_train = False
                    # Auto-train only when enabled and we haven't trained for this symbol/date range
                    if auto_train:
                        last_ticker = st.session_state.get('last_trained_ticker')
                        last_start = st.session_state.get('last_trained_start')
                        last_end = st.session_state.get('last_trained_end')
                        if (model_results is None or
                            last_ticker != selected_symbol or
                            last_start != str(start_date) or
                            last_end != str(end_date)):
                            should_train = True
                    if should_train:
                        with st.spinner('Training selected models...'):
                            model_results = train_models(
                                analysis_data,
                                selected_features,
                                selected_models,
                                available_models
                            )
                        # Cache results and metadata so overlay toggles won't retrain
                        st.session_state['model_results'] = model_results
                        st.session_state['last_trained_ticker'] = selected_symbol
                        st.session_state['last_trained_start'] = str(start_date)
                        st.session_state['last_trained_end'] = str(end_date)
                        st.session_state['last_trained_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Reuse any previously cached results
                        model_results = st.session_state.get('model_results', None)

                    # Provide a main-area Train button (not in sidebar) so users can
                    # manually trigger training without overlay toggles causing retraining.
                    train_now_main = st.button('Train Models Now')
                    if train_now_main:
                        with st.spinner('Training selected models...'):
                            model_results = train_models(
                                analysis_data,
                                selected_features,
                                selected_models,
                                available_models
                            )
                        # Cache results and metadata so overlay toggles won't retrain
                        st.session_state['model_results'] = model_results
                        st.session_state['last_trained_ticker'] = selected_symbol
                        st.session_state['last_trained_start'] = str(start_date)
                        st.session_state['last_trained_end'] = str(end_date)
                        st.session_state['last_trained_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Price Chart with Technical Overlays (overlay changes don't affect model training)
                # Build price chart series according to user-selected overlays
                components = []
                components.append(_to_1d('Close', 'Close'))
                if show_bollinger:
                    components.append(_to_1d('Upper_Band', 'Upper Band'))
                    components.append(_to_1d('Lower_Band', 'Lower Band'))
                if show_sma:
                    components.append(_to_1d('SMA', 'SMA'))
                if show_ema9:
                    components.append(_to_1d('EMA_9', 'EMA 9'))
                if show_ema50:
                    components.append(_to_1d('EMA_50', 'EMA 50'))
                if show_ema200:
                    components.append(_to_1d('EMA_200', 'EMA 200'))
                if show_vwap:
                    components.append(_to_1d('VWAP', 'VWAP'))

                # concatenate only the chosen components (safe if some series are all-NaN)
                try:
                    bb_chart_data = pd.concat(components, axis=1)
                except Exception:
                    # fallback: at least plot Close
                    bb_chart_data = pd.concat([_to_1d('Close', 'Close')], axis=1)

                st.subheader('Price Chart with Technical Overlays')
                # Create an interactive Altair chart with legend-based toggles when available.
                # Fallback to Streamlit's line_chart if Altair isn't available or fails.
                try:
                    if alt_available and alt is not None:
                        plot_df = bb_chart_data.copy()
                        # Ensure index is a named datetime index column for Altair
                        plot_df = plot_df.reset_index()
                        if plot_df.columns[0] != 'date':
                            plot_df = plot_df.rename(columns={plot_df.columns[0]: 'date'})

                        # Melt to long format for unified line plotting
                        long_df = plot_df.melt(id_vars=['date'], var_name='Series', value_name='Value')

                        # Remove rows with NaN values to avoid broken lines on Altair side
                        long_df = long_df.dropna(subset=['Value'])

                        # Create a multi-selection bound to the legend so users can toggle lines
                        selection = alt.selection_multi(fields=['Series'], bind='legend')

                        line = alt.Chart(long_df).mark_line().encode(
                            x=alt.X('date:T', title='Date'),
                            y=alt.Y('Value:Q', title='Price'),
                            color=alt.Color('Series:N', legend=alt.Legend(orient='bottom')),
                            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.12)),
                            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('Series:N', title='Series'), alt.Tooltip('Value:Q', title='Value')]
                        ).add_selection(selection).interactive()

                        # If Bollinger bands exist, add a soft area between Upper and Lower bands
                        band_layers = None
                        if 'Upper Band' in bb_chart_data.columns and 'Lower Band' in bb_chart_data.columns:
                            bands_df = bb_chart_data[['Upper Band', 'Lower Band']].reset_index()
                            if bands_df.columns[0] != 'date':
                                bands_df = bands_df.rename(columns={bands_df.columns[0]: 'date'})
                            # drop rows where either band is NaN
                            bands_df = bands_df.dropna(subset=['Upper Band', 'Lower Band'])
                            if not bands_df.empty:
                                band_layers = alt.Chart(bands_df).mark_area(color='#a6bddb', opacity=0.15).encode(
                                    x=alt.X('date:T', title='Date'),
                                    y=alt.Y('Upper Band:Q'),
                                    y2='Lower Band:Q',
                                    tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('Upper Band:Q', title='Upper'), alt.Tooltip('Lower Band:Q', title='Lower')]
                                )

                        # Compose final chart: bands (if any) under lines
                        final_chart = (band_layers + line) if band_layers is not None else line
                        st.altair_chart(final_chart, use_container_width=True)
                    else:
                        raise Exception('Altair not available')
                except Exception:
                    # Fallback: at least show Streamlit line chart
                    st.line_chart(bb_chart_data, use_container_width=True)

                # New UI: ARIMA 30-day forecast (only if pmdarima is installed)
                if pmd_available:
                    # place checkbox near charts so user can toggle forecasting on/off
                    show_arima_forecast = st.checkbox('Show 30-day ARIMA forecast', value=True)
                    if show_arima_forecast:
                        with st.spinner('Fitting ARIMA and forecasting next 30 business days...'):
                            # Route ARIMA informational output into a sidebar expander log
                            _arima_log = st.sidebar.expander("ARIMA Log", expanded=True)
                            _st_info, _st_warn, _st_succ = st.info, st.warning, st.success
                            _ui_info, _ui_warn, _ui_succ = ui.info, ui.warning, ui.success
                            try:
                                st.info = _arima_log.info
                                st.warning = _arima_log.warning
                                st.success = _arima_log.success
                                ui.info = _arima_log.info
                                ui.warning = _arima_log.warning
                                ui.success = _arima_log.success

                                amodel, forecast_df = forecast_arima_next_n(analysis_data, n_periods=30)
                            finally:
                                # Restore original functions so other UI renders normally
                                st.info, st.warning, st.success = _st_info, _st_warn, _st_succ
                                ui.info, ui.warning, ui.success = _ui_info, _ui_warn, _ui_succ
                        if amodel is None or forecast_df is None:
                            st.warning('ARIMA forecast unavailable (pmdarima not installed, not enough history, or fit failed).')
                        else:
                            st.subheader('ARIMA 30-day Forecast')
                            
                            # Show forecast summary
                            last_price = float(analysis_data['Close'].dropna().iloc[-1])
                            forecast_end = float(forecast_df['Forecast'].iloc[-1])
                            total_change = ((forecast_end / last_price) - 1) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Current Price", f"${last_price:.2f}")
                            col2.metric("30-Day Forecast", f"${forecast_end:.2f}")
                            col3.metric("Expected Change", f"{total_change:+.1f}%")
                            
                            # overlay historical Close and forecast on a single chart for context
                            try:
                                # Get recent historical data for context (last 60 days)
                                hist_close = _to_1d('Close').tail(60).rename('Historical Close')
                                fc_series = forecast_df['Forecast'].rename('ARIMA Forecast')
                                
                                # Create confidence interval series
                                upper_bound = forecast_df['Upper'].rename('Upper Confidence')
                                lower_bound = forecast_df['Lower'].rename('Lower Confidence')
                                
                                plot_df = pd.concat([hist_close, fc_series], axis=1)
                                
                                st.line_chart(plot_df, use_container_width=True)
                                
                                # Show confidence bands separately
                                st.subheader('Forecast with Confidence Intervals')
                                conf_df = pd.concat([fc_series, upper_bound, lower_bound], axis=1)
                                st.line_chart(conf_df, use_container_width=True)
                                
                            except Exception:
                                # fallback: show forecast alone
                                st.line_chart(forecast_df['Forecast'])
                                
                            # show numeric forecast with intervals
                            st.write("Forecast table (next 30 business days):")
                            # Show only every 5th day for readability, plus first and last
                            display_df = forecast_df.iloc[::5].copy()
                            if len(forecast_df) >5:
                                display_df = pd.concat([display_df, forecast_df.iloc[[-1]]])
                            
                            st.dataframe(display_df.style.format({
                                "Forecast": "${:.2f}", 
                                "Lower": "${:.2f}", 
                                "Upper": "${:.2f}",
                                "Daily_Change": "{:+.2f}%",
                                "Cumulative_Change": "{:+.2f}%"
                            }))
                else:
                    st.info('pmdarima not available  -  install pmdarima to enable ARIMA forecasting.')

                # Technical Indicators
                col1, col2 = st.columns(2)

                # Left column: RSI + Returns
                with col1:
                    st.subheader('RSI')
                    try:
                        # RSI chart always shows RSI with overbought/oversold lines regardless of overlay settings
                        rsi_series = _to_1d('RSI')
                        overbought = pd.Series(70.0, index=rsi_series.index, name='Overbought (70)')
                        oversold = pd.Series(30.0, index=rsi_series.index, name='Oversold (30)')
                        rsi_chart = pd.concat([rsi_series.rename('RSI'), overbought, oversold], axis=1)
                        if alt_available:
                            try:
                                df = rsi_chart.copy().reset_index().rename(columns={'index': 'date'})
                                df['date'] = pd.to_datetime(df['date'])
                                rsi_melt = df.melt('date', var_name='series', value_name='value')
                                chart = alt.Chart(rsi_melt).mark_line().encode(
                                    x=alt.X('date:T', title='Date', axis=alt.Axis(orient='bottom')),
                                    y=alt.Y('value:Q', title='Value'),
                                    color=alt.Color('series:N', legend=alt.Legend(orient='bottom')),
                                    tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('series:N', title='Series'), alt.Tooltip('value:Q', title='Value')]
                                ).interactive()
                                st.altair_chart(chart, use_container_width=True)
                            except Exception:
                                st.line_chart(rsi_chart)
                        else:
                            st.line_chart(rsi_chart)
                    except Exception:
                        # Fallback: just show RSI line
                        st.line_chart(_to_1d('RSI'))

                    st.subheader('Returns')
                    # Returns chart always shows all return periods regardless of overlay settings
                    returns_chart = pd.concat([
                        _to_1d('Return_1d', '1-Day Return'),
                        _to_1d('Return_5d', '5-Day Return'),
                        _to_1d('Return_10d', '10-Day Return')
                    ], axis=1)
                    if alt_available:
                        try:
                            df = returns_chart.copy().reset_index().rename(columns={'index': 'date'})
                            df['date'] = pd.to_datetime(df['date'])
                            returns_melt = df.melt('date', var_name='series', value_name='value')
                            chart = alt.Chart(returns_melt).mark_line().encode(
                                x=alt.X('date:T', title='Date', axis=alt.Axis(orient='bottom')),
                                y=alt.Y('value:Q', title='Return'),
                                color=alt.Color('series:N', legend=alt.Legend(orient='bottom')),
                                tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('series:N', title='Series'), alt.Tooltip('value:Q', title='Value')]
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            st.line_chart(returns_chart)
                    else:
                        st.line_chart(returns_chart)

                # Right column: MACD + Volatility
                with col2:
                    st.subheader('MACD')
                    # MACD chart always shows MACD and Signal lines regardless of overlay settings
                    macd_chart = pd.concat([
                        _to_1d('MACD', 'MACD'),
                        _to_1d('Signal', 'Signal')
                    ], axis=1)
                    if alt_available:
                        try:
                            df = macd_chart.copy().reset_index().rename(columns={'index': 'date'})
                            df['date'] = pd.to_datetime(df['date'])
                            macd_melt = df.melt('date', var_name='series', value_name='value')
                            chart = alt.Chart(macd_melt).mark_line().encode(
                                x=alt.X('date:T', title='Date', axis=alt.Axis(orient='bottom')),
                                y=alt.Y('value:Q', title='MACD'),
                                color=alt.Color('series:N', legend=alt.Legend(orient='bottom')),
                                tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('series:N', title='Series'), alt.Tooltip('value:Q', title='Value')]
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            st.line_chart(macd_chart)
                    else:
                        st.line_chart(macd_chart)

                    st.subheader('10-Day Volatility')
                    # Volatility chart always shows volatility regardless of overlay settings
                    vol_series = _to_1d('Volatility_10d')
                    if alt_available:
                        try:
                            vol_df = vol_series.reset_index().rename(columns={'index': 'date', vol_series.name: 'Volatility'})
                            vol_df['date'] = pd.to_datetime(vol_df['date'])
                            chart = alt.Chart(vol_df).mark_line().encode(
                                x=alt.X('date:T', title='Date', axis=alt.Axis(orient='bottom')),
                                y=alt.Y('Volatility:Q', title='10-Day Volatility'),
                                tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('Volatility:Q', title='Volatility')]
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            try:
                                st.line_chart(vol_series)
                            except Exception:
                                st.line_chart(analysis_data.get('Volatility_10d', pd.Series(dtype=float)))
                    else:
                        try:
                            st.line_chart(vol_series)
                        except Exception:
                            st.line_chart(analysis_data.get('Volatility_10d', pd.Series(dtype=float)))

                # Model Performance
                if model_results is not None:
                    st.subheader('Model Performance')

                    # Display results for each model
                    st.write("### Model Comparison")
                    for model_name, metrics in model_results['all_results'].items():
                        st.write(f"\n**{model_name}**")
                        cols = st.columns(4)
                        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
                        cols[1].metric("ROC-AUC", f"{metrics['auc']:.2%}")
                        cols[2].metric("Precision", f"{metrics['precision']:.2%}")
                        cols[3].metric("F1-Score", f"{metrics['f1']:.2%}")

                        # Cross-validation results
                        st.write(f"Cross-validation AUC: {metrics['cv_auc_mean']:.2%} ({metrics['cv_auc_std']:.2%})")
                        
                        # Confusion Matrix
                        if 'confusion_matrix' in metrics:
                            cm = metrics['confusion_matrix']
                            st.write(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

                    # Feature Importance
                    if model_results['feature_importance'] is not None:
                        st.write("### Feature Importance")
                        importance_df = pd.DataFrame(
                            model_results['feature_importance'].items(),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=True)

                        # Plot feature importance
                        st.bar_chart(importance_df.set_index('Feature'))

                    # Latest Prediction
                    st.write("### Tomorrow's Prediction")
                    
                    # Display which model was selected as best
                    if model_results.get('best_model_name'):
                        st.write(f"**Best performing model:** {model_results['best_model_name']}")
                    else:
                        st.write("**Best performing model:** None (using baseline)")
                    
                    latest_data = analysis_data.iloc[-1:]
                    best_model = model_results.get('best_model')
                    best_prob_up = 0.5

                    # Best ML model probability (if available)
                    if best_model is not None and hasattr(best_model, 'predict_proba'):
                        try:
                            X_latest = latest_data[model_results['selected_features']]
                            X_latest_scaled = model_results['scaler'].transform(X_latest)
                            
                            # Get raw probabilities
                            raw_probabilities = best_model.predict_proba(X_latest_scaled)[0]
                            
                            # Cap extreme probabilities to prevent overconfidence
                            prob_down = max(0.01, min(0.99, raw_probabilities[0]))  # Cap between 1% and 99%
                            prob_up = max(0.01, min(0.99, raw_probabilities[1]))
                            
                            # Normalize to ensure they sum to 1
                            total = prob_down + prob_up
                            prob_down = prob_down / total
                            prob_up = prob_up / total
                            
                            # DEBUG: Show raw probabilities and feature values
                            st.write("**Debug Information:**")
                            st.write(f"Raw probabilities: [Down: {raw_probabilities[0]:.6f}, Up: {raw_probabilities[1]:.6f}]")
                            st.write(f"Capped probabilities: [Down: {prob_down:.6f}, Up: {prob_up:.6f}]")
                            
                            # Warn about extreme predictions
                            if raw_probabilities[0] < 0.001 or raw_probabilities[1] < 0.001:
                                st.warning("! Model shows extreme confidence (>99.9%) - likely overfitting!")
                            
                            # Show the most important feature values for latest prediction
                            if model_results.get('feature_importance'):
                                st.write("**Top 5 feature values for latest prediction:**")
                                top_features = list(model_results['feature_importance'].keys())[:5]
                                for feature in top_features:
                                    if feature in X_latest.columns:
                                        value = X_latest[feature].iloc[0]
                                        scaled_value = X_latest_scaled[0][model_results['selected_features'].index(feature)]
                                        # Warn about extreme scaled values
                                        extreme_warning = " [EXTREME]" if abs(scaled_value) >3 else ""
                                        st.write(f"  {feature}: {value:.4f} (scaled: {scaled_value:.4f}){extreme_warning}")
                            
                            best_prob_up = float(prob_up)  # Use capped probability
                        except Exception as e:
                            st.write(f"Debug error: {str(e)}")
                            best_prob_up = 0.5
                    else:
                        best_prob_up = 0.5

                    # Confidence label for best model
                    confidence = "High" if abs(best_prob_up - 0.5) >0.2 else "Medium" if abs(best_prob_up - 0.5) >0.1 else "Low"
                    st.write(f"Prediction confidence (best model): **{confidence}**")

                    # Add warning for poor model performance
                    if model_results.get('all_results', {}).get(model_results.get('best_model_name', ''), {}).get('cv_auc_mean', 0.5) < 0.55:
                        st.error("! **MODEL WARNING**: Cross-validation AUC < 55% indicates the model has poor predictive power. Predictions may be unreliable!")

                    # Short textual summaries
                    if best_prob_up > 0.5:
                        st.write(f"Best model predicts price increase with {best_prob_up:.1%} probability")
                    else:
                        st.write(f"Best model predicts price decrease with {(1-best_prob_up):.1%} probability")

                # Summary Statistics
                st.subheader('Summary Statistics')
                st.write(analysis_data['Close'].describe())

                # Next-day predictions: show ARIMA and Best-ML side-by-side (numbers only)
                try:
                    ml_pred = None
                    ml_source = None
                    next_day_price = None
                    arima_pred = None
                    arima_next_day_price = None

                    # Best ML model: convert predicted probability ->signed expected return (heuristic)
                    if model_results is not None and model_results.get('best_model') is not None:
                        try:
                            bm = model_results['best_model']
                            if hasattr(bm, 'predict_proba'):
                                X_latest = analysis_data[model_results['selected_features']].iloc[-1:]
                                X_latest_scaled = model_results['scaler'].transform(X_latest)
                                prob_up = float(bm.predict_proba(X_latest_scaled)[0, 1])
                                recent_abs = analysis_data['Return_1d'].dropna().tail(20).abs()
                                avg_abs = float(recent_abs.mean()) if not recent_abs.empty else 0.01
                                ml_pred = (prob_up - 0.5) * 2.0 * avg_abs
                                ml_source = 'ML-prob-heuristic'
                        except Exception:
                            ml_pred = None

                    # Get ARIMA next-day prediction if forecast was generated
                    if pmd_available and show_arima_forecast and 'amodel' in locals() and 'forecast_df' in locals():
                        try:
                            current_price_arima = float(analysis_data['Close'].dropna().iloc[-1])
                            first_forecast = float(forecast_df['Forecast'].iloc[0])
                            arima_pred = (first_forecast / current_price_arima) - 1
                            arima_next_day_price = first_forecast
                        except Exception:
                            arima_pred = None

                    # Fallback to latest observed next_ret if either forecast missing
                    try:
                        last_obs = float(analysis_data['next_ret'].dropna().iloc[-1])
                    except Exception:
                        last_obs = None

                    if ml_pred is None and last_obs is not None:
                        ml_pred = last_obs
                        ml_source = 'historical_next_ret'

                    # Calculate next day predicted price
                    current_price = float(analysis_data['Close'].dropna().iloc[-1])
                    if ml_pred is not None:
                        next_day_price = current_price * (1 + ml_pred)

                    # Display side-by-side metrics
                    st.subheader("Next-Day Prediction Comparison")
                    
                    # Create comparison table
                    comparison_data = []
                    
                    # ML Model prediction
                    if ml_pred is not None and next_day_price is not None:
                        comparison_data.append({
                            'Model': f'Best ML ({model_results.get("best_model_name", "Unknown")})',
                            'Predicted_Return': f"{ml_pred:.2%}",
                            'Predicted_Price': f"${next_day_price:.2f}",
                            'Price_Change': f"${next_day_price - current_price:+.2f}",
                            'Direction': 'Up' if ml_pred > 0 else 'Down' if ml_pred < 0 else 'Flat'
                        })
                    
                    # ARIMA prediction
                    if arima_pred is not None and arima_next_day_price is not None:
                        comparison_data.append({
                            'Model': 'ARIMA Forecast',
                            'Predicted_Return': f"{arima_pred:.2%}",
                            'Predicted_Price': f"${arima_next_day_price:.2f}",
                            'Price_Change': f"${arima_next_day_price - current_price:+.2f}",
                            'Direction': 'Up' if arima_pred > 0 else 'Down' if arima_pred < 0 else 'Flat'
                        })
                    
                    if comparison_data:
                        # Display current price first
                        st.metric("Current Price", f"${current_price:.2f}")
                        
                        # Create comparison DataFrame
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Show detailed analysis
                        if len(comparison_data) == 2:  # Both predictions available
                            ml_return = float(comparison_data[0]['Predicted_Return'].strip('%')) / 100
                            arima_return = float(comparison_data[1]['Predicted_Return'].strip('%')) / 100
                            
                            agreement = "Agree" if (ml_return >0) == (arima_return >0) else "Disagree"
                            agreement_color = ">" if agreement == "Agree" else "!"
                            
                            st.info(f"{agreement_color} **Model Agreement**: {agreement} on direction")
                            
                            if agreement == "Agree":
                                direction = "bullish" if ml_return >0 else "bearish"
                                st.success(f"Both models are {direction} for tomorrow")
                            else:
                                st.warning("Models disagree - exercise caution in trading decisions")
                        
                        # Add source information
                        if ml_source and len(comparison_data) > 1:
                            st.caption(f"ML Model source: {ml_source}")
                    else:
                        st.write("Next-day prediction: N/A - No models available")
                except Exception:
                    st.write("Next-day prediction: N/A")

                # Quick Trade Pick removed per user request
                # (previously displayed a Bullish/Bearish/Neutral pick based on VWAP and EMA_200)
                
                # Raw Data
                if st.checkbox('Show Raw Data'):
                    st.subheader('Raw Data')
                    st.write(analysis_data)
                
                # Export Data Section
                st.subheader('Export Data')
                st.write("Export the calculated technical indicators and analysis data:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prepare export data
                    export_data = analysis_data.copy()
                    
                    # Add metadata columns
                    export_data.insert(0, 'Ticker', selected_symbol)
                    export_data.insert(1, 'Company_Name', final_stock_tickers.get(selected_symbol, 'N/A'))
                    export_data.insert(2, 'Export_Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    
                    # Convert to CSV

                    csv_data = export_data.to_csv(index=True)

                    st.download_button(
                        label="Download as CSV",
                        data=csv_data,
                        file_name=f"{selected_symbol}_technical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download all technical indicators and price data as CSV file"
                    )
                
                with col2:
                    # Feature importance export (if available)
                    if model_results is not None and model_results.get('feature_importance'):
                        importance_data = pd.DataFrame(
                            list(model_results['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        )
                        importance_data['Ticker'] = selected_symbol
                        importance_data['Export_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        importance_data['Best_Model'] = model_results.get('best_model_name', 'Unknown')
                        
                        importance_csv = importance_data.to_csv(index=False)
                        
                        st.download_button(
                            label="< Feature Importance CSV",
                            data=importance_csv,
                            file_name=f"{selected_symbol}_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download feature importance rankings from the best performing model"
                        )
                    else:
                        st.info("Feature importance data will be available after training models")
                
                # Export information
                with st.expander("Export Information"):
                    st.write("**CSV Export includes:**")
                    st.write("- All technical indicators (RSI, MACD, Bollinger Bands, EMAs, etc.)")
                    st.write("- Price data (Open, High, Low, Close, Volume)")
                    st.write("- Calculated returns and volatility metrics")
                    st.write("- Lagged features and target variables")
                    st.write("- Metadata (Ticker, Company Name, Export Date)")
                    
                    if len(analysis_data) >0:
                        st.write(f"**Current dataset:** {len(analysis_data)} data points from {analysis_data.index.min()} to {analysis_data.index.max()}")
                # END of handling when stock_data is not empty

            else:
                               # Added: handle empty DataFrame case
                ui.error(f"No data available for {selected_symbol} in the selected date range.")
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            # Move errors and tracebacks to the sidebar Run log
            ui.error(f'Error fetching or processing data for {selected_ticker}: {e}')
            with ui.expander("Traceback"):
                ui.text(tb)


# --- New: Route diagnostics to the Run log inside helper functions ---


