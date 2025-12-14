import yfinance as yf
import pandas as pd
import CV_ESN7
from CV_ESN7 import esn_rolling_forward

name = 'JNJ' # 주식 이름 
ticker = yf.Ticker(name)
ori_df = ticker.history(start='2015-07-22', end='2025-07-22', interval='1d', auto_adjust=False)
df = ori_df.copy()

if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
df.index = df.index.normalize()

best_params_cv, all_returns_cv = esn_rolling_forward(
    df=df, 
    n_splits=5,
    pop_size=30, 
    num_generations=30
)
