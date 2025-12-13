import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal import calculate_total_fitness_optimized

PARAM_BOUNDS = {
    'N': (41, 150),       # Long-term MA
    'n': (5, 40),        # Short-term MA
    'a': (0.1, 10.0),      # BUY param
    'b': (0.01, 1.0),      # BUY param
    'c': (0.01, 1.0),    # BUY param
    'a_bar': (0.1, 10.0),  # SELL param
    'b_bar': (0.01, 1.0),  # SELL param
    'c_bar': (0.01, 1.0),# SELL param
}

BOUNDS_LIST = [
    (*PARAM_BOUNDS['N'], int),
    (*PARAM_BOUNDS['n'], int),
    (*PARAM_BOUNDS['a'], float),
    (*PARAM_BOUNDS['b'], float),
    (*PARAM_BOUNDS['c'], float),
    (*PARAM_BOUNDS['a_bar'], float),
    (*PARAM_BOUNDS['b_bar'], float),
    (*PARAM_BOUNDS['c_bar'], float)
]

def create_bounds_decorator(bounds_list):
    """
    개체(individual)의 값이 범위를 벗어나지 않고
    N > n 제약조건을 만족하도록 강제하는 데코레이터 함수를 생성합니다.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 1. 원본 변이/교배 함수를 실행
            offspring = func(*args, **kwargs)
            
            # 2. 반환된 모든 개체(자손)를 순회
            for ind in offspring:
                # 3. 각 파라미터 값을 기본 범위 내로 강제
                for i, (min_val, max_val, param_type) in enumerate(bounds_list):
                    # 정수형 파라미터(N, n)는 반올림
                    if param_type == int:
                        ind[i] = int(round(ind[i]))
                    
                    # 최소/최대 값 제한 (clamping)
                    ind[i] = max(min_val, min(ind[i], max_val))
                
                # 4. 특수 제약조건 강제: N > n
                # ind[0] == N, ind[1] == n
                if ind[0] <= ind[1]:
                    # N과 n이 범위 내에 있다는 것이 보장된 상태
                    ind[0] = ind[1] + 1
                    
                    # 만약 N이 n+1이 되면서 최대치를 넘으면, 대신 n을 N-1로 설정
                    if ind[0] > bounds_list[0][1]: 
                        ind[0] = bounds_list[0][1] 
                        ind[1] = ind[0] - 1 
                        
            # 5. 수정된 개체(자손) 튜플을 반환
            return offspring
        return wrapper
    return decorator


def init_creator():
    """DEAP creator 초기화"""
    for name in ["FitnessMin", "Individual"]:
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_MA_signals_vectorized(
    data: pd.DataFrame, 
    N: int, n: int, 
    a: float, b: float, c: float,         
    a_bar: float, b_bar: float, c_bar: float 
) -> pd.DataFrame:
    
    df = data.copy()
    close_values = df['Close'].values

    N, n = int(N), int(n)
    if N <= n: 
         n = N - 1
         if n <= 0:
              return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    ma_N_values = talib.SMA(close_values, timeperiod=N)
    ma_n_values = talib.SMA(close_values, timeperiod=n)

    df['MA_N'] = pd.Series(ma_N_values, index=df.index)
    df['MA_n'] = pd.Series(ma_n_values, index=df.index)
    
    df['Zt'] = np.where(
        df['MA_n'] != 0, 
        (df['MA_N'] - df['MA_n']) / df['MA_n'], 
        0.0  # MA_n이 0이면 비율도 0으로 처리
    )
    df['Wk'] = -df['Zt']              
    
    df = df.dropna(subset=['Zt'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    df['Sign'] = np.sign(df['Zt'])
    df['Cross'] = df['Sign'].diff().fillna(0)
    df['RegimeGroup'] = (df['Cross'] != 0).cumsum()

    df['Zt_buy'] = df['Zt'].where(df['Sign'] >= 0)
    df['Wk_sell'] = df['Wk'].where(df['Sign'] < 0)

    df['MEt_calc'] = df.groupby('RegimeGroup')['Zt_buy'].expanding().max().reset_index(level=0, drop=True)
    df['MWk_calc'] = df.groupby('RegimeGroup')['Wk_sell'].expanding().max().reset_index(level=0, drop=True)

    cond_buy_8 = (df['MEt_calc'] > (b * c))
    min_val_buy = (df['MEt_calc'] / a)
    cond_buy_9 = (df['Zt'] < np.minimum(min_val_buy, c))
    buy_signals = cond_buy_8 & cond_buy_9

    cond_sell_10 = (df['MWk_calc'] > (b_bar * c_bar))
    min_val_sell = (df['MWk_calc'] / a_bar)
    cond_sell_11 = (df['Wk'] < np.minimum(min_val_sell, c_bar))
    sell_signals = cond_sell_10 & cond_sell_11

    df['Type'] = np.nan
    df.loc[buy_signals, 'Type'] = 'BUY'
    df.loc[sell_signals, 'Type'] = 'SELL'

    final_signals_df = df.dropna(subset=['Type']).copy()

    if final_signals_df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    final_signals_df = final_signals_df[['Close', 'Type']].copy()
    final_signals_df.reset_index(inplace=True)
    index_column_name = final_signals_df.columns[0]
    final_signals_df.rename(columns={index_column_name: 'Index'}, inplace=True)
    
    return final_signals_df


def evaluate_MA_individual(
    individual, df_data, expected_trading_points_df
):  
    # 8개 파라미터 언패킹
    N, n, a, b, c, a_bar, b_bar, c_bar = individual

    suggested_signals_df = generate_MA_signals_vectorized(
        df_data, N, n, a, b, c, a_bar, b_bar, c_bar
    )
    
    fitness = calculate_total_fitness_optimized(
        df_data, expected_trading_points_df, suggested_signals_df
    )

    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)
    
    if fitness == float('inf'):
        return (1000000000.0,) 

    return (fitness,)

def run_MA_ga_optimization(
    df_input: pd.DataFrame, 
    generations: int = 50, 
    population_size: int = 50, 
    seed: int = None
):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    init_creator()
    
    df_data = df_input.copy()
    if isinstance(df_input, pd.Series):
        df_data = pd.DataFrame(df_input)
        df_data.columns = ['Close']
    elif 'Close' not in df_data.columns:
        raise ValueError("입력 DataFrame에 'Close' 컬럼이 반드시 포함되어야 합니다.")
    if 'cpm_point_type' not in df_data.columns:
        raise ValueError("입력 DataFrame에 'cpm_point_type' 컬럼이 반드시 포함되어야 합니다.")
        
    signal_rows = df_data.loc[df_data['cpm_point_type'] != 0].copy()
    signal_rows['Type'] = signal_rows['cpm_point_type'].map({-1: 'BUY', 1: 'SELL'})
    expected_trading_points_df = pd.DataFrame({
        'Index': signal_rows.index,
        'Type': signal_rows['Type'],
        'Close': signal_rows['Close']
    })
    
    toolbox = base.Toolbox()

    # 8개 파라미터 등록 (전역 PARAM_BOUNDS 사용)
    toolbox.register("attr_N", random.randint, *PARAM_BOUNDS['N'])
    toolbox.register("attr_n", random.randint, *PARAM_BOUNDS['n'])
    toolbox.register("attr_a", random.uniform, *PARAM_BOUNDS['a'])
    toolbox.register("attr_b", random.uniform, *PARAM_BOUNDS['b'])
    toolbox.register("attr_c", random.uniform, *PARAM_BOUNDS['c'])
    toolbox.register("attr_a_bar", random.uniform, *PARAM_BOUNDS['a_bar'])
    toolbox.register("attr_b_bar", random.uniform, *PARAM_BOUNDS['b_bar'])
    toolbox.register("attr_c_bar", random.uniform, *PARAM_BOUNDS['c_bar'])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_N, toolbox.attr_n, 
                      toolbox.attr_a, toolbox.attr_b, toolbox.attr_c,
                      toolbox.attr_a_bar, toolbox.attr_b_bar, toolbox.attr_c_bar), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluate 함수 등록
    toolbox.register("evaluate", evaluate_MA_individual, 
                     df_data=df_data, 
                     expected_trading_points_df=expected_trading_points_df)

    # 데코레이터를 적용하기 위해 원본 연산자 등록
    toolbox.register("mate_orig", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate_orig", tools.mutGaussian, 
                     mu=[0]*8, 
                     sigma=[3, 1, 0.5, 0.05, 0.01, 0.5, 0.05, 0.01], 
                     indpb=0.1)

    # 데코레이터 생성 및 적용
    bounds_decorator = create_bounds_decorator(BOUNDS_LIST)
    toolbox.register("mate", bounds_decorator(toolbox.mate_orig))
    toolbox.register("mutate", bounds_decorator(toolbox.mutate_orig))
    
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("--- MA 8-Params 벡터화 GA 최적화 시작 (순차 처리) ---")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- 이동평균 유전 알고리즘 결과 ---")
    print(f"최적의 8-파라미터: {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_N = best_individual[0]
    final_n = best_individual[1]
    final_a = best_individual[2]
    final_b = best_individual[3]
    final_c = best_individual[4]
    final_a_bar = best_individual[5]
    final_b_bar = best_individual[6]
    final_c_bar = best_individual[7]
    
    final_params = (final_N, final_n, final_a, final_b, final_c,
                    final_a_bar, final_b_bar, final_c_bar)
    
    print(f"반환될 파라미터 (N, n, a, b, c, ā, ̄b, ̄c): {final_params}")

    suggested_signals_from_best_params = generate_MA_signals_vectorized(
        df_data, *final_params
    )

    df_data['MA_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['MA_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return final_params, best_fitness, df_data