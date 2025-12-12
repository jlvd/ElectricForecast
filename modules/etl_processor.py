import pandas as pd
import numpy as np
import re
import os

def parse_date_from_filename(filename):
    """
    Extracts date from filename format 'rgd_DDMMYY.xlsx'.
    """
    # Regex to match 'rgd_DDMMYY'
    match = re.search(r'rgd_(\d{6})', filename, re.IGNORECASE)
    if match:
        date_str = match.group(1)
        try:
            # Parse DDMMYY
            date = pd.to_datetime(date_str, format='%d%m%y')
            return date
        except ValueError:
            return None
    return None

def read_excel_file(file):
    """
    Reads a specific format Excel file:
    - Range A12:A36 -> Hour (1-24) or 0-23
    - Range L12:L36 -> Demand
    Returns a DataFrame with 'fecha' (datetime specific hour) and 'valor'.
    """
    try:
        # Load workbook, no header, usecols A and L, skiprows first 11 rows
        # A is 0, L is 11.
        # Range A12:A36 is 25 rows (1 to 24 + 1 for 00:00 next day? or 24 hours?)
        # "iniciando 01:00 del día actual, hasta 00:00 del día siguiente" -> 24 hours.
        # 1, 2, ..., 23, 24 (where 24 is 00:00 next day)
        
        # We need the filename to get the base date.
        filename = file.name
        base_date = parse_date_from_filename(filename)
        
        if not base_date:
             raise ValueError(f"El nombre del archivo '{filename}' no cumple el formato 'rgd_DDMMYY.xlsx'.")
             
        # Read Excel
        # pd.read_excel header=None to read by index.
        # skiprows=11 to start at row 12.
        # nrows=24 for 24 hours.
        df = pd.read_excel(file, header=None, skiprows=11, nrows=25, usecols="A,L")
        
        df.columns = ['hora', 'valor']
        
        # Clean data
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        
        # Construct full datetime
        # Hours might be 1..24. 
        # 24:00:00 is usually represented as 00:00:00 of next day.
        
        timestamps = []
        for h in df['hora']:
            # Normalize h (hour) to integer
            h_int = None
            try:
                if isinstance(h, (int, float)):
                    h_int = int(h)
                elif isinstance(h, str):
                    # Handle "01:00" or "01"
                    parts = h.split(':')
                    h_int = int(parts[0])
                elif hasattr(h, 'hour'): # datetime.time or datetime
                    h_int = h.hour
                
                # Calculate Timestamp
                if h_int is not None:
                    # 24 or 0 represents midnight of the NEXT day per user specs
                    if h_int == 24 or h_int == 0:
                        ts = base_date + pd.Timedelta(days=1) 
                    else:
                        ts = base_date + pd.Timedelta(hours=h_int)
                    timestamps.append(ts)
                else:
                    timestamps.append(None)
            except:
                timestamps.append(None)
                
        df['fecha'] = timestamps
        df = df.dropna(subset=['fecha'])
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        return df[['fecha', 'valor']]
        
    except Exception as e:
        raise ValueError(f"Error al procesar excel: {str(e)}")

def detect_outliers_and_impute(df):
    """
    Detects outliers using Z-Score and imputes missing/outlier values.
    Returns: df_clean, df_dirty, stats
    """
    df_dirty = df.copy() # Keep original for comparison
    df_clean = df.copy()
    
    # 1. Handle Nans first? No, detect outliers on existing data first?
    # Actually, calculate Z-score on available data.
    
    # Z-Score
    mean = df_clean['valor'].mean()
    std = df_clean['valor'].std()
    
    if std == 0:
        df_clean['is_outlier'] = False
    else:
        df_clean['z_score'] = (df_clean['valor'] - mean) / std
        df_clean['is_outlier'] = df_clean['z_score'].abs() > 3
    
    # Mark Explicit Nuls
    df_clean['is_null'] = df_clean['valor'].isna()
    
    # Mask Outliers and Nulls as NaN for interpolation
    mask = df_clean['is_outlier'] | df_clean['is_null']
    df_clean.loc[mask, 'valor'] = np.nan
    
    # Interpolate
    df_clean['valor'] = df_clean['valor'].interpolate(method='linear', limit_direction='both')
    
    # Stats
    stats = {
        "total_rows": len(df),
        "nulls_inputs": int(df_dirty['valor'].isna().sum()),
        "outliers_detected": int(df_clean['is_outlier'].sum()),
        "interpolated_total": int(mask.sum())
    }
    
    return df_clean, df_dirty, stats

def process_data(df):
    """
    Main aggregator. 
    1. Agrega/Resample a Nivel Diario (Mean).
    2. Imputation on Daily level? Or Hourly then Resample?
    
    Propuesta says: "Resampling (Agregación) a nivel DIARIO. Motivo: Elimina ruido...".
    
    However, missing values handling is best done at high frequency (hourly) before aggregation if possible, 
    but if we have chunks of missing days, we do it at daily.
    
    The user feedback: "Control de nulos, rellenar valores ausentes... comparativa de datos sucios vs limpios".
    
    Let's clean at high frequency (Hourly) if the input is hourly, THEN resample.
    Or if input is the concatenated DF of multiple files.
    
    For the dashboard flow:
    1. User uploads N files.
    2. We parse all of them into one BIG hourly dataframe.
    3. We run cleaning on this BIG dataframe.
    4. We resample to Daily.
    
    Let's adjust logic.
    """
    
    # Clean Hourly First
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df_clean_hourly, df_dirty_hourly, stats_hourly = detect_outliers_and_impute(df.sort_values('fecha'))
    
    # Resample to Daily
    # Dirty Daily (just mean of raw, with nans propagating)
    daily_dirty = df_dirty_hourly.set_index('fecha').resample('D')['valor'].mean().reset_index()
    
    # Clean Daily (mean of clean hourly)
    daily_clean = df_clean_hourly.set_index('fecha')['valor'].resample('D').mean()
    
    # But wait, if we have missing HOURS in a day, the daily mean might be skewed if we don't fill holes first.
    # We already filled holes in df_clean_hourly.
    
    # Also check for MISSING DAYS (gaps in files).
    # Reindex daily to full range
    full_idx = pd.date_range(start=daily_clean.index.min(), end=daily_clean.index.max(), freq='D')
    daily_clean = daily_clean.reindex(full_idx)
    
    # Interpolate missing days if any (file gaps)
    daily_clean_imputed = daily_clean.interpolate(method='linear')
    
    # Stats update for missing days
    missing_days = daily_clean.isna().sum() 
    stats_hourly['missing_days_filled'] = int(missing_days)
    
    daily_clean_final = daily_clean_imputed.reset_index().rename(columns={'index': 'fecha', 'valor': 'valor'})
    
    return daily_clean_final, daily_dirty, stats_hourly
