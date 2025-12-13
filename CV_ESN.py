import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from backtesting import Backtest, Strategy
from ESN_Signals import esn_signals
import warnings
import multiprocessing
import talib

import CPM as cpm
import MovingAverage as ma
import RSI as rsi
import ROC as roc
import importlib

warnings.filterwarnings('ignore')

# --- 워밍업을 위한 최대 룩백(Lookback) 기간 정의 ---
try:
    MAX_LOOKBACK_MA = ma.PARAM_BOUNDS['N'][1] 
    MAX_LOOKBACK_RSI = rsi.PARAM_BOUNDS['x'][1] 
    MAX_LOOKBACK_ROC = roc.PARAM_BOUNDS['p_long'][1] 
    
    # 모든 지표를 통틀어 가장 긴 룩백 + 버퍼(5일)
    GLOBAL_MAX_LOOKBACK = max(MAX_LOOKBACK_MA, MAX_LOOKBACK_RSI, MAX_LOOKBACK_ROC) + 5
    
except (AttributeError, KeyError) as e:
    print(f"--- [워밍업 설정 오류] ---")
    print(f"    {e}")
    print(f"    TA 모듈에 'PARAM_BOUNDS' 딕셔너리나 관련 키가 없습니다.")
    print(f"    최대 룩백을 200으로 임의 설정합니다.")
    print(f"--------------------------")
    GLOBAL_MAX_LOOKBACK = 200
    MAX_LOOKBACK_MA = 150
    MAX_LOOKBACK_RSI = 30
    MAX_LOOKBACK_ROC = 100

print(f"--- [워밍업 설정 완료] ---")
print(f"    최대 룩백 기간(Global Max Lookback) : {GLOBAL_MAX_LOOKBACK}일")
print(f"    (MA: {MAX_LOOKBACK_MA}, RSI: {MAX_LOOKBACK_RSI}, ROC: {MAX_LOOKBACK_ROC})")
print(f"--------------------------")


# 매수매도 전략
class PredictedSignalStrategy(Strategy):
    def init(self):
        self.signal = self.I(lambda x: x, self.data.Predicted_Signals, name='signal')
        self.consecutive_buys = 0
        
        if 'MA_Signals' in self.data.df.columns:
            self.ma_signal = self.I(lambda x: x, self.data.MA_Signals, name='MA Signals', overlay=False)
        
        if 'RSI_Signals' in self.data.df.columns:
            self.rsi_signal = self.I(lambda x: x, self.data.RSI_Signals, name='RSI Signals', overlay=False)
            
        if 'ROC_Signals' in self.data.df.columns:
            self.roc_signal = self.I(lambda x: x, self.data.ROC_Signals, name='ROC Signals', overlay=False)

    def next(self):
        current_signal = self.signal[-1]
        
        # 매도 신호가 나오고, 포지션을 보유 중일 때
        if current_signal == 1 and self.position:
            self.position.close()
            self.consecutive_buys = 0
            return

        # 매수 신호가 나왔을 때
        if current_signal == -1:
            price = self.data.Close[-1]
            if price <= 0:
                return
            
            self.consecutive_buys += 1

            # 첫 번째 매수: 포지션이 없을 때
            if self.consecutive_buys == 1 and not self.position:
                # 전체 자산(equity)의 33%
                amount_to_invest = self.equity * 0.33
                size_to_buy = int(amount_to_invest / price)
                self.buy(size=size_to_buy)

            # 두 번째 매수: 포지션을 보유 중일 때
            elif self.consecutive_buys == 2 and self.position:
                # 전체 자산의 33%
                amount_to_invest = self.equity * 0.33
                size_to_buy = int(amount_to_invest / price)
                self.buy(size=size_to_buy)

            # 세 번째 이후 매수: 포지션을 보유 중일 때
            elif self.consecutive_buys >= 3 and self.position:
                # 남은 현금을 모두 사용하여 매수
                self.buy()
        
        # 중립 또는 매도 신호지만 포지션이 없을 때
        else:
            self.consecutive_buys = 0
            
# 하이퍼파라미터 범위 설정
PARAM_RANGES = {
    'spectral_radius': {'min': 0.8, 'max': 0.99, 'type': float},
    'sparsity': {'min': 0.7, 'max': 0.9, 'type': float},
    'input_scaling': {'min': 0.5, 'max': 4.0, 'type': float},
    'buy_threshold': {'min': 0.3, 'max': 0.6, 'type': float},
    'sell_threshold': {'min': 0.3, 'max': 0.6, 'type': float}
}
# n_reservoir 고정값
N_RESERVOIR_FIXED = 300

def generate_individual():
    spec_rad = random.uniform(PARAM_RANGES['spectral_radius']['min'], PARAM_RANGES['spectral_radius']['max'])
    sp = random.uniform(PARAM_RANGES['sparsity']['min'], PARAM_RANGES['sparsity']['max'])
    inp_scale = random.uniform(PARAM_RANGES['input_scaling']['min'], PARAM_RANGES['input_scaling']['max'])
    buy_thresh = random.uniform(PARAM_RANGES['buy_threshold']['min'], PARAM_RANGES['buy_threshold']['max'])
    sell_thresh = random.uniform(PARAM_RANGES['sell_threshold']['min'], PARAM_RANGES['sell_threshold']['max'])
    return [spec_rad, sp, inp_scale, buy_thresh, sell_thresh]

def fitness_function_with_backtesting(params, train_df: pd.DataFrame, test_df: pd.DataFrame, Technical_Signals=None):
    spectral_radius, sparsity, input_scaling, buy_threshold, sell_threshold = params
    
    n_reservoir = N_RESERVOIR_FIXED
    spectral_radius = max(PARAM_RANGES['spectral_radius']['min'], min(spectral_radius, PARAM_RANGES['spectral_radius']['max']))
    sparsity = max(PARAM_RANGES['sparsity']['min'], min(sparsity, PARAM_RANGES['sparsity']['max']))
    input_scaling = max(PARAM_RANGES['input_scaling']['min'], min(input_scaling, PARAM_RANGES['input_scaling']['max']))
    buy_threshold = max(PARAM_RANGES['buy_threshold']['min'], min(buy_threshold, PARAM_RANGES['buy_threshold']['max']))
    sell_threshold = max(PARAM_RANGES['sell_threshold']['min'], min(sell_threshold, PARAM_RANGES['sell_threshold']['max']))

    try:
        backtest_signals_df = esn_signals(
            train_df=train_df,
            test_df=test_df,
            Technical_Signals=Technical_Signals,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scaling=input_scaling,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        if backtest_signals_df.empty or 'Predicted_Signals' not in backtest_signals_df.columns:
            return 0.0,
        
        backtest_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        backtest_data['Predicted_Signals'] = backtest_signals_df['Predicted_Signals'].reindex(backtest_data.index)
        backtest_data['Predicted_Signals'] = backtest_data['Predicted_Signals'].fillna(0)
        
        bt = Backtest(backtest_data, PredictedSignalStrategy,
                        cash=10000, commission=.002, exclusive_orders=False)
        stats = bt.run()
        return_percent = stats['Return [%]']
        max_drawdown = stats['Max. Drawdown [%]']

        fitness = (stats['Return [%]'] * stats['SQN']) / (abs(stats['Max. Drawdown [%]']) ** 0.25)

        if pd.isna(return_percent) or np.isinf(return_percent) or \
           pd.isna(max_drawdown) or np.isinf(max_drawdown) or \
           pd.isna(fitness) or np.isinf(fitness):
            return 0.0,
        return fitness, 
    except Exception as e:
        print(f"백테스팅 중 오류 발생: {e}")
        return 0.0,

def init_deap_creator():
    for name in ["FitnessMax", "Individual"]:
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def run_genetic_algorithm(train_df_ga: pd.DataFrame, test_df_ga: pd.DataFrame, technical_signals_list: list,
                          pop_size: int = 50, num_generations: int = 20, cxpb: float = 0.7, mutpb: float = 0.2,
                          random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    init_deap_creator()

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", fitness_function_with_backtesting,
                     train_df=train_df_ga,
                     test_df=test_df_ga,
                     Technical_Signals=technical_signals_list)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=[0, 0, 0, 0, 0],  # 5개
                     sigma=[
                         (PARAM_RANGES['spectral_radius']['max'] - PARAM_RANGES['spectral_radius']['min']) * 0.1,
                         (PARAM_RANGES['sparsity']['max'] - PARAM_RANGES['sparsity']['min']) * 0.1,
                         (PARAM_RANGES['input_scaling']['max'] - PARAM_RANGES['input_scaling']['min']) * 0.1,
                         (PARAM_RANGES['buy_threshold']['max'] - PARAM_RANGES['buy_threshold']['min']) * 0.1, 
                         (PARAM_RANGES['sell_threshold']['max'] - PARAM_RANGES['sell_threshold']['min']) * 0.1
                     ],
                     indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    population, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, num_generations,
                                          stats=stats, halloffame=hof, verbose=True)
    
    pool.close()
    pool.join()

    best_individual = hof[0]
    print(f"\nGA 최적화 완료 - 최적 하이퍼파라미터: {best_individual}")
    print(f"GA 최적화 완료 - 최고 Fitness: {best_individual.fitness.values[0]:.4f}")

    return best_individual, log
    
def perform_final_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame, best_params: list, technical_signals_list: list,
                         random_state: int = 42):
    spectral_radius, sparsity, input_scaling, buy_threshold, sell_threshold = best_params

    n_reservoir = N_RESERVOIR_FIXED
    spectral_radius = max(PARAM_RANGES['spectral_radius']['min'], min(spectral_radius, PARAM_RANGES['spectral_radius']['max']))
    sparsity = max(PARAM_RANGES['sparsity']['min'], min(sparsity, PARAM_RANGES['sparsity']['max']))
    input_scaling = max(PARAM_RANGES['input_scaling']['min'], min(input_scaling, PARAM_RANGES['input_scaling']['max']))
    buy_threshold = max(PARAM_RANGES['buy_threshold']['min'], min(buy_threshold, PARAM_RANGES['buy_threshold']['max']))
    sell_threshold = max(PARAM_RANGES['sell_threshold']['min'], min(sell_threshold, PARAM_RANGES['sell_threshold']['max']))

    print(f"\n--- 최적화된 파라미터로 최종 ESN 학습 및 백테스팅 ---")
    print(f"  n_reservoir: {n_reservoir}")
    print(f"  spectral_radius: {spectral_radius:.4f}")
    print(f"  sparsity: {sparsity:.4f}")
    print(f"  input_scaling: {input_scaling:.4f}")
    print(f"  buy_threshold: {buy_threshold:.4f}")
    print(f"  sell_threshold: {sell_threshold:.4f}")

    final_backtest_signals_df = esn_signals(
        train_df=train_df,
        test_df=test_df,
        Technical_Signals=technical_signals_list,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        random_state=random_state
    )

    if not isinstance(final_backtest_signals_df, pd.DataFrame) or final_backtest_signals_df.empty or 'Predicted_Signals' not in final_backtest_signals_df.columns:
        print("최종 ESN 모델에서 유효한 신호가 생성되지 않았습니다. 백테스팅을 건너뜀.")
        return None, None

    final_backtest_data = test_df.copy()
    final_backtest_data['Predicted_Signals'] = final_backtest_signals_df['Predicted_Signals']
    final_backtest_data['Predicted_Signals'] = final_backtest_data['Predicted_Signals'].fillna(0)

    bt_final = Backtest(final_backtest_data, PredictedSignalStrategy,
                        cash=10000, commission=.002, exclusive_orders=False)
    stats_final = bt_final.run()

    print("\n최종 백테스팅 결과 (최적화된 파라미터):")
    print(stats_final)
    
    bt_final.plot(filename='test_df_backtest_results', open_browser=True)
    
    return stats_final, final_backtest_signals_df

def rolling_forward_split_3way(df: pd.DataFrame, n_splits: int, initial_train_ratio: float = 0.5):
    """
    데이터를 Train / Validation / Test 3개로 분할하여 반환합니다.
    """
    total_len = len(df)
    initial_train_and_val_size = int(total_len * initial_train_ratio)
    remaining_len = total_len - initial_train_and_val_size
    
    if n_splits <= 0:
        print("n_splits는 1 이상이어야 합니다.")
        return

    test_size = remaining_len // n_splits
    if test_size == 0 and remaining_len > 0:
        test_size = remaining_len
        n_splits = 1
        print("경고: 데이터가 부족하여 n_splits=1로 강제 조정됩니다.")
    elif test_size == 0:
        print("오류: 분할할 테스트 데이터가 없습니다.")
        return

    val_test_ratio = 2
    # val_size = test_size
    val_size = int(test_size * val_test_ratio)
    initial_train_size = initial_train_and_val_size - val_size
    
    if initial_train_size <= 0:
        print("오류: 초기 Train 데이터 크기가 0 또는 음수입니다. initial_train_ratio를 늘리세요.")
        return

    print(f"--- 3-Way 분할 설정 ---")
    print(f"초기 Train 크기: {initial_train_size}, Validation 크기: {val_size}, Test 크기: {test_size}")
    print(f"총 {n_splits}개 폴드 생성")
    print(f"----------------------")

    for i in range(n_splits):
        train_end_idx = initial_train_size + i * test_size
        train_df = df.iloc[:train_end_idx].copy()
        
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_size
        val_df = df.iloc[val_start_idx:val_end_idx].copy()
        
        test_start_idx = val_end_idx
        test_end_idx = test_start_idx + test_size
        
        if i == n_splits - 1:
            test_end_idx = total_len
            
        test_df = df.iloc[test_start_idx:test_end_idx].copy()

        if val_df.empty or test_df.empty:
            continue
            
        yield train_df, val_df, test_df

def add_trend_features(df, p_long, p_short):
    """
    최적화된 N(p_long)과 n(p_short)을 받아 Slope를 계산합니다.
    """
    df = df.copy()
    
    # 1. Long-term Slope (N 사용)
    # 정규화: (Slope / Close) * 100
    slope_long = talib.LINEARREG_SLOPE(df['Close'], timeperiod=p_long)
    df['Slope_Long'] = (slope_long / df['Close']) * 100
    
    # 2. Short-term Slope (n 사용)
    slope_short = talib.LINEARREG_SLOPE(df['Close'], timeperiod=p_short)
    df['Slope_Short'] = (slope_short / df['Close']) * 100
    
    return df
    
def ts_optimization(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    1. 각 기술적 지표의 최적 파라미터를 'train_df'로 찾습니다.
    2. 워밍업 데이터를 포함하여 train/val/test 신호를 생성합니다.
    """
    
    # --- [0단계] 워밍업 데이터 결합 ---
    train_df_for_signals = train_df.copy()

    # Valid 신호 계산용 = Train 끝부분 + Valid 원본
    print(f"    [워밍업] Valid 신호용 데이터 생성 (Train {GLOBAL_MAX_LOOKBACK}일 + Valid {len(val_df)}일)")
    val_df_for_signals = pd.concat([
        train_df.tail(GLOBAL_MAX_LOOKBACK),
        val_df
    ])

    # Test 신호 계산용 = Valid 끝부분 + Test 원본
    print(f"    [워밍업] Test 신호용 데이터 생성 (Valid {GLOBAL_MAX_LOOKBACK}일 + Test {len(test_df)}일)")
    test_df_for_signals = pd.concat([
        val_df.tail(GLOBAL_MAX_LOOKBACK),
        test_df
    ])

    
    # --- [1단계] 각 기술적 지표의 최적 파라미터 찾기 ---
    print("- 이동평균(MA) 파라미터 최적화")
    ma_best_params, _, _ = ma.run_MA_ga_optimization(train_df)
    
    best_N = int(ma_best_params[0]) 
    best_n = int(ma_best_params[1])
    print(f"     - MA 최적 파라미터: {ma_best_params}")
    print(f"     - [Slope 적용] 최적 주기 N={best_N}, n={best_n}을 기울기 계산에 사용합니다.")

    print("- RSI 파라미터 최적화")
    rsi_best_params, _, _ = rsi.run_RSI_ga_optimization(train_df)
    print(f"     - RSI 최적 파라미터: {rsi_best_params}")

    print("- ROC 파라미터 최적화")
    roc_best_params, _, _ = roc.run_roc_ga_optimization(train_df)
    print(f"     - ROC 최적 파라미터: {roc_best_params}")


    # --- [2단계] 신호 및 추세 지표 생성 ---
    print("\n     - 모든 최적 파라미터로 신호 및 Slope 생성 중...")

    # 최종 반환용 원본 DF 복사
    train_df_with_signals = train_df.copy()
    val_df_with_signals = val_df.copy()
    test_df_with_signals = test_df.copy()
    
    # ---------------------------------------------------------
    # 1. MA 신호 생성
    # ---------------------------------------------------------
    N, n, a, b, c, a_bar, b_bar, c_bar = ma_best_params
    
    train_ma_signals = ma.generate_MA_signals_vectorized(train_df_for_signals, N, n, a, b, c, a_bar, b_bar, c_bar)
    val_ma_signals = ma.generate_MA_signals_vectorized(val_df_for_signals, N, n, a, b, c, a_bar, b_bar, c_bar)
    test_ma_signals = ma.generate_MA_signals_vectorized(test_df_for_signals, N, n, a, b, c, a_bar, b_bar, c_bar)
    
    train_df_with_signals['MA_Signals'] = train_ma_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(train_df_with_signals.index).fillna(0).astype(int)
    val_df_with_signals['MA_Signals'] = val_ma_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(val_df_with_signals.index).fillna(0).astype(int)
    test_df_with_signals['MA_Signals'] = test_ma_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(test_df_with_signals.index).fillna(0).astype(int)

    # ---------------------------------------------------------
    # 2. RSI 신호 생성
    # ---------------------------------------------------------
    x, overbought_level, oversold_level, p, q = rsi_best_params
    
    train_rsi_signals = rsi.generate_RSI_signals_vectorized(train_df_for_signals, x, overbought_level, oversold_level, p, q)
    val_rsi_signals = rsi.generate_RSI_signals_vectorized(val_df_for_signals, x, overbought_level, oversold_level, p, q)
    test_rsi_signals = rsi.generate_RSI_signals_vectorized(test_df_for_signals, x, overbought_level, oversold_level, p, q)
    
    train_df_with_signals['RSI_Signals'] = train_rsi_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(train_df_with_signals.index).fillna(0).astype(int)
    val_df_with_signals['RSI_Signals'] = val_rsi_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(val_df_with_signals.index).fillna(0).astype(int)
    test_df_with_signals['RSI_Signals'] = test_rsi_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(test_df_with_signals.index).fillna(0).astype(int)

    # ---------------------------------------------------------
    # 3. ROC 신호 생성
    # ---------------------------------------------------------
    p_long, p_short, high_level, low_level, eq_bw_upper, eq_bw_lower = roc_best_params

    train_roc_signals = roc.generate_roc_signals_divergence(train_df_for_signals, p_long, p_short, high_level, low_level, eq_bw_upper, eq_bw_lower)
    val_roc_signals = roc.generate_roc_signals_divergence(val_df_for_signals, p_long, p_short, high_level, low_level, eq_bw_upper, eq_bw_lower)
    test_roc_signals = roc.generate_roc_signals_divergence(test_df_for_signals, p_long, p_short, high_level, low_level, eq_bw_upper, eq_bw_lower)
    
    train_df_with_signals['ROC_Signals'] = train_roc_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(train_df_with_signals.index).fillna(0).astype(int)
    val_df_with_signals['ROC_Signals'] = val_roc_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(val_df_with_signals.index).fillna(0).astype(int)
    test_df_with_signals['ROC_Signals'] = test_roc_signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1}).reindex(test_df_with_signals.index).fillna(0).astype(int)

    # ---------------------------------------------------------
    # 4. Slope(추세) 생성
    # ---------------------------------------------------------
    
    train_slope_df = add_trend_features(train_df_for_signals, best_N, best_n)
    val_slope_df = add_trend_features(val_df_for_signals, best_N, best_n)
    test_slope_df = add_trend_features(test_df_for_signals, best_N, best_n)
    
    slope_cols = ['Slope_Long', 'Slope_Short']
    
    train_df_with_signals[slope_cols] = train_slope_df[slope_cols].reindex(train_df_with_signals.index)
    val_df_with_signals[slope_cols] = val_slope_df[slope_cols].reindex(val_df_with_signals.index)
    test_df_with_signals[slope_cols] = test_slope_df[slope_cols].reindex(test_df_with_signals.index)

    # ESN 입력 리스트 정의
    technical_signals_list = ['MA_Signals', 'RSI_Signals', 'ROC_Signals', 'Slope_Long', 'Slope_Short']
    
    return train_df_with_signals, val_df_with_signals, test_df_with_signals, technical_signals_list
    
def esn_rolling_forward(df: pd.DataFrame, n_splits: int = 5, initial_train_ratio: float = 0.5,
                        pop_size: int = 50, num_generations: int = 20):
    
    total_returns = []
    bh_returns = []
    total_mdd = []
    best_params_per_fold = []
    
    # --- 0단계: 전체 데이터에 대해 CPM 생성 ---
    print(f"--- 전체 데이터에 대해 CPM 생성 중 ---")
    _, df_with_cpm = cpm.cpm_model(df, column='Close', P=0.05, T=5)
    
    splits = list(rolling_forward_split_3way(df_with_cpm, n_splits, initial_train_ratio))
    
    if not splits:
        print("유효한 데이터 분할이 생성되지 않았습니다.")
        return None, None

    print(f"\n--- 롤링 포워드 교차 검증 시작 ---")
    
    for i, (train_df, validation_df, test_df) in enumerate(splits):
        print("\n" + "="*50)
        print(f"--- 폴드 {i+1} / {n_splits} ---")
        print(f"Train: {train_df.index.min()} ~ {train_df.index.max()} ({len(train_df)}일)")
        print(f"Valid: {validation_df.index.min()} ~ {validation_df.index.max()} ({len(validation_df)}일)")
        print(f"Test:  {test_df.index.min()} ~ {test_df.index.max()} ({len(test_df)}일)")
        print("="*50)
        
        train_df_with_cpm = train_df
        val_df_with_cpm = validation_df
        test_df_with_cpm = test_df
        
        try:
            # --- 1단계: TA 최적화 ---
            print(f"[{i+1}/{n_splits}] 1단계: 기술적 지표 파라미터 최적화 (Train/Valid 신호 생성)...")
            train_df_with_signals, val_df_with_signals, test_df_with_signals, technical_signals_for_esn = ts_optimization(
                train_df_with_cpm, val_df_with_cpm, test_df_with_cpm
            )

            # --- 2단계: ESN 하이퍼파라미터 최적화 (GA) ---
            print(f"\n[{i+1}/{n_splits}] 2단계: ESN 하이퍼파라미터 최적화 (Train->Valid)...")
            best_params, _ = run_genetic_algorithm(
                train_df_ga=train_df_with_signals,
                test_df_ga=val_df_with_signals, # ⬅️ Validation Set 전달
                technical_signals_list=technical_signals_for_esn,
                pop_size=pop_size,
                num_generations=num_generations
            )
            best_params_per_fold.append(best_params)
            print(f"[{i+1}/{n_splits}] ESN 최적 파라미터 (Valid 기준): {best_params}")

            # --- 3단계: 최종 성능 평가 ---
            final_train_signals = train_df_with_signals
            final_test_signals = test_df_with_signals
            final_signals_list = technical_signals_for_esn

            # Test 데이터로 최종 백테스트
            print(f"[{i+1}/{n_splits}] 3b. ESN 최종 평가 (Test 데이터)...")
            stats, _ = perform_final_backtest(
                train_df=final_train_signals,
                test_df=final_test_signals,
                best_params=best_params,
                technical_signals_list=final_signals_list
            )
            
            if stats is not None:
                print(f"\n--- 폴드 {i+1} 최종 성과 (Test Set) ---")
                print(f"Return [%]: {stats['Return [%]']:.2f}")
                print(f"Buy & Hold Return [%]: {stats['Buy & Hold Return [%]']:.2f}")
                print(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
                
                total_returns.append(stats['Return [%]'])
                bh_returns.append(stats['Buy & Hold Return [%]'])
                total_mdd.append(stats['Max. Drawdown [%]'])
        
        except Exception as e:
            print(f"폴드 {i+1} 처리 중 치명적 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("롤링 포워드 교차 검증 최종 결과 (Train/Validation/Test 분리):")
    if total_returns:
        print(f"총 {len(total_returns)}개 폴드 결과")
        print(f"각 폴드 Return [%] : {[round(r, 2) for r in total_returns]}")
        print(f"각 폴드 B&H Return [%]    : {[round(b, 2) for b in bh_returns]}")
        print(f"평균 Return [%]: {np.mean(total_returns):.4f}")
        print(f"Buy&Hold 평균 Return [%]: {np.mean(bh_returns):.4f}")
        print(f"평균 MDD [%]: {np.mean(total_mdd):.4f}")
    else:
        print("유효한 백테스팅 결과가 없습니다.")
    print("="*50)
    
    return best_params_per_fold, total_returns