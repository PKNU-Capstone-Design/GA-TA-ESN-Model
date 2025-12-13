import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal import calculate_total_fitness_optimized

PARAM_BOUNDS = {
    'x': (5, 30),              # RSI 기간
    'overbought': (70.0, 99.9), # 과매수
    'oversold': (0.1, 30.0),    # 과매도
    'p': (-10.0, -0.1),         # 매도 사선 기울기 (음수)
    'q': (0.1, 10.0),           # 매수 사선 기울기 (양수)
}

BOUNDS_LIST = [
    (*PARAM_BOUNDS['x'], int),
    (*PARAM_BOUNDS['overbought'], float),
    (*PARAM_BOUNDS['oversold'], float),
    (*PARAM_BOUNDS['p'], float),
    (*PARAM_BOUNDS['q'], float)
]

def create_bounds_decorator(bounds_list):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for ind in offspring:
                # 1. 기본 범위 제한 (Clamping)
                for i, (min_val, max_val, param_type) in enumerate(bounds_list):
                    if param_type == int:
                        ind[i] = int(round(ind[i]))
                    ind[i] = max(min_val, min(ind[i], max_val))
                
                # 2. 특수 제약조건: Overbought > Oversold
                # ind[1]: overbought, ind[2]: oversold
                if ind[1] <= ind[2]:
                    # Overbought를 Oversold + 1.0으로 밀어 올림
                    ind[1] = ind[2] + 1.0
                    # 만약 Max를 넘어가면 Oversold를 깎음
                    if ind[1] > bounds_list[1][1]:
                        ind[1] = bounds_list[1][1]
                        ind[2] = ind[1] - 1.0
                        
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

def generate_RSI_signals_vectorized(
    data: pd.DataFrame, 
    x: int, 
    overbought_level: float, 
    oversold_level: float, 
    p: float, 
    q: float
) -> pd.DataFrame:

    df = data.copy()
    x = int(x)

    # 1. RSI 계산
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=x)
    df = df.dropna(subset=['RSI'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 2. 크로스(Cross) 및 리짐(Regime) 그룹 정의
    df['RSI_Shifted'] = df['RSI'].shift(1).fillna(50) # 50은 중립

    # (a) 과매수 진입/이탈 지점
    df['Overbought_Entry'] = (df['RSI_Shifted'] <= overbought_level) & (df['RSI'] > overbought_level)
    df['Overbought_Exit'] = (df['RSI_Shifted'] > overbought_level) & (df['RSI'] <= overbought_level)

    # (b) 과매도 진입/이탈 지점
    df['Oversold_Entry'] = (df['RSI_Shifted'] >= oversold_level) & (df['RSI'] < oversold_level)
    df['Oversold_Exit'] = (df['RSI_Shifted'] < oversold_level) & (df['RSI'] >= oversold_level)

    # (c) 모든 경계 교차 지점에서 새 그룹(Regime) 시작
    df['Cross'] = df['Overbought_Entry'] | df['Overbought_Exit'] | \
                  df['Oversold_Entry'] | df['Oversold_Exit']
    df['RegimeGroup'] = df['Cross'].cumsum()

    # 3. 그룹별 사선(Oblique Line) 계산
    
    # (a) 과매수 영역(RSI > level)에 있는 데이터만 필터링
    df_ob = df[df['RSI'] > overbought_level].copy()
    if not df_ob.empty:
        # TimeElapsed = (current_pos - start_pos)
        df_ob['TimeElapsed'] = df_ob.groupby('RegimeGroup').cumcount() 
        # StartRSI = df_filtered.loc[overbought_entry_idx, 'RSI']
        df_ob['StartRSI'] = df_ob.groupby('RegimeGroup')['RSI'].transform('first')
        # 사선 Y값 = StartRSI + p * TimeElapsed
        df_ob['ObliqueLine_Y'] = df_ob['StartRSI'] + p * df_ob['TimeElapsed']
        
        # 신호: RSI가 사선 아래로 하향 돌파
        df_ob['SellSignal'] = (df_ob['RSI'] < df_ob['ObliqueLine_Y'])
        
        sell_indices = df_ob[df_ob['SellSignal']].groupby('RegimeGroup').head(1).index
    else:
        sell_indices = pd.Index([])

    # (b) 과매도 영역(RSI < level)에 있는 데이터만 
    df_os = df[df['RSI'] < oversold_level].copy()
    if not df_os.empty:
        # TimeElapsed = (current_pos - start_pos)
        df_os['TimeElapsed'] = df_os.groupby('RegimeGroup').cumcount()
        # StartRSI = df_filtered.loc[oversold_entry_idx, 'RSI']
        df_os['StartRSI'] = df_os.groupby('RegimeGroup')['RSI'].transform('first')
        # 사선 Y값 = StartRSI + q * TimeElapsed
        df_os['ObliqueLine_Y'] = df_os['StartRSI'] + q * df_os['TimeElapsed']
        
        # 신호: RSI가 사선 위로 상향 돌파
        df_os['BuySignal'] = (df_os['RSI'] > df_os['ObliqueLine_Y'])
        
        buy_indices = df_os[df_os['BuySignal']].groupby('RegimeGroup').head(1).index
    else:
        buy_indices = pd.Index([])

    # 4. 신호 취합 및 반환
    buy_df = df.loc[buy_indices, ['Close']].copy()
    buy_df['Type'] = 'BUY'
    
    sell_df = df.loc[sell_indices, ['Close']].copy()
    sell_df['Type'] = 'SELL'

    signals_df = pd.concat([buy_df, sell_df])
    
    if signals_df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])
        
    signals_df = signals_df.sort_index()
    final_signals_df = signals_df[['Close', 'Type']].copy()
    final_signals_df.reset_index(inplace=True)
    index_column_name = final_signals_df.columns[0]
    final_signals_df.rename(columns={index_column_name: 'Index'}, inplace=True)
        
    return final_signals_df

def evaluate_RSI_individual(
    individual, df_data, expected_trading_points_df
):
    # 5개 파라미터 언패킹
    x, overbought_level, oversold_level, p, q = individual
    
    suggested_signals_df = generate_RSI_signals_vectorized(
        df_data, x, overbought_level, oversold_level, p, q
    )
    
    fitness = calculate_total_fitness_optimized(
        df_data, expected_trading_points_df, suggested_signals_df
    )
    
    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)
        
    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)

def run_RSI_ga_optimization(
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

    # 5개 파라미터 등록
    toolbox.register("attr_x", random.randint, *PARAM_BOUNDS['x'])
    toolbox.register("attr_overbought", random.uniform, *PARAM_BOUNDS['overbought'])
    toolbox.register("attr_oversold", random.uniform, *PARAM_BOUNDS['oversold'])
    toolbox.register("attr_p", random.uniform, *PARAM_BOUNDS['p'])
    toolbox.register("attr_q", random.uniform, *PARAM_BOUNDS['q'])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_x, toolbox.attr_overbought,
                      toolbox.attr_oversold, toolbox.attr_p, toolbox.attr_q), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluate 함수 등록
    toolbox.register("evaluate", evaluate_RSI_individual, 
                     df_data=df_data, 
                     expected_trading_points_df=expected_trading_points_df)

    # 데코레이터 적용을 위한 원본 연산자 등록
    toolbox.register("mate_orig", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate_orig", tools.mutGaussian, 
                     mu=[0]*5, sigma=[1, 1, 1, 0.2, 0.2], indpb=0.1)

    # [데코레이터 생성 및 적용
    bounds_decorator = create_bounds_decorator(BOUNDS_LIST)
    toolbox.register("mate", bounds_decorator(toolbox.mate_orig))
    toolbox.register("mutate", bounds_decorator(toolbox.mutate_orig))

    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("--- RSI 5-Params 벡터화 GA 최적화 시작 ---")
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

    print("\n--- RSI 유전 알고리즘 결과 ---")
    print(f"최적 5-파라미터: {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    # 최종 파라미터 추출
    final_params = tuple(best_individual)
    
    print(f"반환될 파라미터 (x, ob, os, p, q): {final_params}")

    suggested_signals_from_best_params = generate_RSI_signals_vectorized(
        df_data, *final_params
    )

    df_data['RSI_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['RSI_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return final_params, best_fitness, df_data