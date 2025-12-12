import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Color Palette for Dark Mode
COLOR_REAL = '#00FFFF'     # Cyan
COLOR_PROPHET = '#00FF00'  # Lime Green
COLOR_XGB = '#FF4500'      # Orange Red
COLOR_DIRTY = '#FF00FF'    # Magenta (for outliers/raw)
COLOR_CLEAN = '#00FFFF'    # Cyan
COLOR_CTX = '#808080'      # Gray

def plot_history(df):
    """
    Plots the entire historical demand.
    df columns: fecha, valor
    """
    fig = px.line(df, x='fecha', y='valor', title='Histórico de Demanda Energética')
    fig.update_traces(line_color=COLOR_REAL)
    fig.update_layout(xaxis_title='Fecha', yaxis_title='Demanda (MW)', template='plotly_dark')
    return fig

def plot_dirty_vs_clean(df_dirty, df_clean, outliers_indices=None):
    """
    Plots raw data vs cleaned data to show imputation.
    """
    fig = go.Figure()
    
    # Dirty Line (Raw)
    fig.add_trace(go.Scatter(
        x=df_dirty['fecha'], 
        y=df_dirty['valor'], 
        mode='lines', 
        name='Datos Originales (Sucio)', 
        line=dict(color=COLOR_DIRTY, width=1, dash='dot')
    ))
    
    # Clean Line (Imputed)
    fig.add_trace(go.Scatter(
        x=df_clean['fecha'], 
        y=df_clean['valor'], 
        mode='lines', 
        name='Datos Finales (Limpio)', 
        line=dict(color=COLOR_CLEAN, width=2)
    ))
    
    fig.update_layout(
        title="Calidad de Datos: Comparativa Original vs Limpio", 
        xaxis_title="Fecha", 
        yaxis_title="Demanda",
        template='plotly_dark'
    )
    return fig

def plot_comparison(y_true, prophet_pred, xgb_pred=None):
    """
    Plots Real vs Prophet vs XGBoost (if available).
    """
    fig = go.Figure()
    
    # Real Data
    fig.add_trace(go.Scatter(
        x=y_true['fecha'], 
        y=y_true['valor'], 
        mode='lines', 
        name='Real', 
        line=dict(color='white', width=1) # White for ground truth in dark mode
    ))
    
    # Prophet
    if prophet_pred is not None:
        fig.add_trace(go.Scatter(
            x=prophet_pred['ds'], 
            y=prophet_pred['yhat'], 
            mode='lines', 
            name='Prophet', 
            line=dict(color=COLOR_PROPHET)
        ))
        
    # XGBoost
    if xgb_pred is not None:
        fig.add_trace(go.Scatter(
            x=xgb_pred['fecha'], 
            y=xgb_pred['yhat'], 
            mode='lines', 
            name='XGBoost', 
            line=dict(color=COLOR_XGB)
        ))
        
    fig.update_layout(
        title="Backtesting: Comparativa de Modelos", 
        xaxis_title="Fecha", 
        yaxis_title="Demanda",
        template='plotly_dark'
    )
    return fig

def plot_forecast(history_df, forecast_df, model_name="Prophet"):
    """
    Plots historical context + future forecast.
    """
    fig = go.Figure()
    
    # Plot last 90 days of history for context
    subset_history = history_df.tail(90)
    
    fig.add_trace(go.Scatter(
        x=subset_history['fecha'], 
        y=subset_history['valor'], 
        mode='lines', 
        name='Historia Reciente', 
        line=dict(color=COLOR_CTX)
    ))
    
    # Forecast
    if 'yhat' in forecast_df.columns:
        # Prophet or XGBoost generic
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'] if 'ds' in forecast_df.columns else forecast_df['fecha'], 
            y=forecast_df['yhat'], 
            mode='lines', 
            name=f'Pronóstico {model_name}', 
            line=dict(color=COLOR_PROPHET if model_name=="Prophet" else COLOR_XGB, dash='dash')
        ))
        
        # Confidence Interval (only common for Prophet output structure usually)
        if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_upper'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_lower'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(255, 255, 255, 0.1)', # Lighter fill for dark mode
                fill='tonexty',
                showlegend=True,
                name='Intervalo Confianza'
            ))
            
    fig.update_layout(title=f"Proyección Futura ({model_name})", xaxis_title="Fecha", yaxis_title="Demanda", template='plotly_dark')
    return fig
