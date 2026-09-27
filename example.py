"""GA-TA-ESN 실행 예제.

기본 실행은 최종 결과 산출용 실험 조건을 유지합니다.
환경과 전체 파이프라인만 빠르게 확인할 때는 ``--quick``을 사용하세요.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    n_splits: int
    initial_train_ratio: float
    pop_size: int
    num_generations: int
    ta_population_size: int
    ta_generations: int
    plot_results: bool


FULL_EXPERIMENT = ExperimentConfig(
    n_splits=5,
    initial_train_ratio=0.5,
    pop_size=30,
    num_generations=30,
    ta_population_size=50,
    ta_generations=50,
    plot_results=True,
)

QUICK_CHECK = ExperimentConfig(
    n_splits=2,
    initial_train_ratio=0.7,
    pop_size=4,
    num_generations=2,
    ta_population_size=4,
    ta_generations=2,
    plot_results=False,
)


def get_experiment_config(quick: bool = False) -> ExperimentConfig:
    """실행 목적에 맞는 실험 조건을 반환합니다."""
    return QUICK_CHECK if quick else FULL_EXPERIMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="기술적 지표 최적화와 ESN을 이용한 주가 예측 실험"
    )
    parser.add_argument("--ticker", default="JNJ", help="Yahoo Finance 종목 코드")
    parser.add_argument("--start", default="2015-07-22", help="조회 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-07-22", help="조회 종료일 (YYYY-MM-DD)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="전체 결과가 아닌 실행 환경과 파이프라인을 빠르게 검증합니다.",
    )
    return parser.parse_args()


def main() -> None:
    # 무거운 실험 의존성은 실제 실행 시점에 불러와 설정 테스트를 가볍게 유지합니다.
    import pandas as pd
    import yfinance as yf

    from CV_ESN import esn_rolling_forward

    args = parse_args()
    config = get_experiment_config(args.quick)
    mode = "빠른 검증" if args.quick else "전체 실험"

    print(
        f"[{mode}] ticker={args.ticker}, folds={config.n_splits}, "
        f"ESN(population={config.pop_size}, generations={config.num_generations}), "
        f"TA(population={config.ta_population_size}, generations={config.ta_generations})"
    )

    ticker = yf.Ticker(args.ticker)
    original_df = ticker.history(
        start=args.start,
        end=args.end,
        interval="1d",
        auto_adjust=False,
    )
    if original_df.empty:
        raise RuntimeError(
            f"{args.ticker}의 주가 데이터를 가져오지 못했습니다. 종목 코드와 기간을 확인하세요."
        )

    df = original_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()

    best_params_cv, all_returns_cv = esn_rolling_forward(
        df=df,
        n_splits=config.n_splits,
        initial_train_ratio=config.initial_train_ratio,
        pop_size=config.pop_size,
        num_generations=config.num_generations,
        ta_population_size=config.ta_population_size,
        ta_generations=config.ta_generations,
        plot_results=config.plot_results,
    )

    print("최적 파라미터:", best_params_cv)
    print("폴드별 결과:", all_returns_cv)


if __name__ == "__main__":
    main()
