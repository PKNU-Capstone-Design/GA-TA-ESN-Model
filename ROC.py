import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal import calculate_total_fitness_optimized

PARAM_BOUNDS = {
    'p_long': (20, 100),        # 장기 기간
    'p_short': (5, 19),         # 단기 기간 (p_long > p_short 보장 필요)
    'high_level': (101.0, 130.0), # 고점 기준 (> 100)
    'low_level': (70.0, 99.0),    # 저점 기준 (< 100)
    'eq_bw_upper': (0.1, 5.0),    # 균형선 상단 밴드
    'eq_bw_lower': (0.1, 5.0),    # 균형선 하단 밴드
}

BOUNDS_LIST = [
    (*PARAM_BOUNDS['p_long'], int),
    (*PARAM_BOUNDS['p_short'], int),
    (*PARAM_BOUNDS['high_level'], float),
    (*PARAM_BOUNDS['low_level'], float),
    (*PARAM_BOUNDS['eq_bw_upper'], float),
    (*PARAM_BOUNDS['eq_bw_lower'], float)
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
                
                # 2. 특수 제약조건: p_long > p_short
                # ind[0]: p_long, ind[1]: p_short
                if ind[0] <= ind[1]:
                    # p_long을 p_short + 1로 설정
                    ind[0] = ind[1] + 1
                    # 만약 p_long이 최대값을 넘어가면, p_short를 깎음
                    if ind[0] > bounds_list[0][1]:
                        ind[0] = bounds_list[0][1]
                        ind[1] = ind[0] - 1
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

def generate_roc_signals_divergence(
    data: pd.DataFrame, 
    p_long: int, 
    p_short: int, 
    high_level: float, 
    low_level: float, 
    eq_bw_upper: float, 
    eq_bw_lower: float
) -> pd.DataFrame:

    df = data.copy()
    
    p_long = int(p_long)
    p_short = int(p_short)

    # 1. 장기/단기 ROCP 계산
    df['ROC_Long'] = talib.ROCR(df['Close'], timeperiod=p_long) * 100
    df['ROC_Short'] = talib.ROCR(df['Close'], timeperiod=p_short) * 100
    
    df = df.dropna(subset=['ROC_Long', 'ROC_Short'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 2. 조건 정의
    eq_line_upper = 100.0 + eq_bw_upper
    eq_line_lower = 100.0 - eq_bw_lower

    # 3. 불리언 마스크(Mask) 생성
    
    # 조건 1: 단기 ROC가 균형선(100) 근처에 있는가?
    is_short_near_eq = (df['ROC_Short'] >= eq_line_lower) & \
                       (df['ROC_Short'] <= eq_line_upper)
                        
    # 조건 2: 장기 ROC가 고점/저점에 도달했는가?
    is_long_at_high = (df['ROC_Long'] > high_level)
    is_long_at_low = (df['ROC_Long'] < low_level)

    # 4. 신호 조합
    # 매수: 장기 과매도(Low) 상태에서 + 단기 흐름이 안정을 찾았을 때
    buy_mask = is_long_at_low & is_short_near_eq

    # 매도: 장기 과매수(High) 상태에서 + 단기 흐름이 안정을 찾았을 때
    sell_mask = is_long_at_high & is_short_near_eq

    # 신호가 연속으로 발생하는 것을 방지
    buy_signals = buy_mask & ~buy_mask.shift(1).fillna(False)
    sell_signals = sell_mask & ~sell_mask.shift(1).fillna(False)

    # 5. DataFrame 생성
    buy_df = df.loc[buy_signals, ['Close']].copy()
    buy_df['Type'] = 'BUY'
    
    sell_df = df.loc[sell_signals, ['Close']].copy()
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

def evaluate_roc_individual(
    individual, df_data, expected_trading_points_df
):
    # 6개 파라미터 언패킹
    p_long, p_short, high_level, low_level, eq_bw_upper, eq_bw_lower = individual
    
    suggested_signals_df = generate_roc_signals_divergence(
        df_data, p_long, p_short, high_level, low_level, eq_bw_upper, eq_bw_lower
    )
    
    fitness = calculate_total_fitness_optimized(
        df_data, expected_trading_points_df, suggested_signals_df
    )
    
    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)

    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)

def run_roc_ga_optimization(
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
    
    # 기대 거래 지점 준비
    signal_rows = df_data.loc[df_data['cpm_point_type'] != 0].copy()
    signal_rows['Type'] = signal_rows['cpm_point_type'].map({-1: 'BUY', 1: 'SELL'})
    expected_trading_points_df = pd.DataFrame({
        'Index': signal_rows.index,
        'Type': signal_rows['Type'],
        'Close': signal_rows['Close']
    })

    toolbox = base.Toolbox()
    
    # 6-파라미터 등록
    toolbox.register("attr_p_long", random.randint, *PARAM_BOUNDS['p_long'])
    toolbox.register("attr_p_short", random.randint, *PARAM_BOUNDS['p_short'])
    toolbox.register("attr_high_level", random.uniform, *PARAM_BOUNDS['high_level'])
    toolbox.register("attr_low_level", random.uniform, *PARAM_BOUNDS['low_level'])
    toolbox.register("attr_eq_bw_upper", random.uniform, *PARAM_BOUNDS['eq_bw_upper'])
    toolbox.register("attr_eq_bw_lower", random.uniform, *PARAM_BOUNDS['eq_bw_lower'])

    # 6-파라미터 개체 생성
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_p_long, toolbox.attr_p_short, 
                      toolbox.attr_high_level, toolbox.attr_low_level,
                      toolbox.attr_eq_bw_upper, toolbox.attr_eq_bw_lower), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 평가 함수 등록
    toolbox.register("evaluate", evaluate_roc_individual, 
                     df_data=df_data, 
                     expected_trading_points_df=expected_trading_points_df)

    # 데코레이터 적용을 위한 원본 연산자 등록
    toolbox.register("mate_orig", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate_orig", tools.mutGaussian, 
                     mu=[0]*6, 
                     sigma=[5, 2, 2, 2, 0.5, 0.5], # 각 파라미터 범위에 맞춰 조정됨
                     indpb=0.1)
    
    # 데코레이터 생성 및 적용
    bounds_decorator = create_bounds_decorator(BOUNDS_LIST)
    toolbox.register("mate", bounds_decorator(toolbox.mate_orig))
    toolbox.register("mutate", bounds_decorator(toolbox.mutate_orig))
    
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("--- ROC 6-Params 다이버전스 GA 최적화 시작 ---")
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- ROC 유전 알고리즘 결과 ---")
    print(f"최적의 6-파라미터: {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    # 최종 파라미터 추출
    final_params = tuple(best_individual)
    
    print(f"반환될 파라미터 (p_long, p_short, high, low, eq_up, eq_low): {final_params}")

    # 벡터화된 함수 호출
    suggested_signals_from_best_params = generate_roc_signals_divergence(
        df_data, *final_params
    )
    
    df_data['ROC_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['ROC_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return final_params, best_fitness, df_data