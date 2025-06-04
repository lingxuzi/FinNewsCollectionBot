import pandas as pd
import numpy as np

def moving_avg(arr, w): 
    a = arr.cumsum() 
    ar = np.roll(a, w) 
    ar[:w-1] = np.nan 
    ar[w-1] = 0 
    return (a - ar) / w 

def val_ma(arr):
    a = arr.copy()
    a_ = arr[~np.isnan(arr)]
    a[~np.isnan(arr)] = moving_avg(a_, 3) - moving_avg(a_, 5)
    return a

def pnumpy_ma(df):
    non_numeric_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
    cols_to_scale = [col for col in df.columns if col not in non_numeric_cols]
    v = df[cols_to_scale].values 
    df[cols_to_scale] = pd.DataFrame(np.apply_along_axis(val_ma, 0, v), index = df.index, columns=cols_to_scale)
    return df