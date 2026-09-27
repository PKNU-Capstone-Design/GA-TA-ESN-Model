# Results

NVDA, INTC, NFLX를 대상으로 수행한 5-Fold 롤링 포워드 백테스트 결과입니다.

## 요약

| 종목 | 모델 평균 수익률 | Buy & Hold 평균 | 평균 초과 수익률 | 모델 평균 MDD |
|---|---:|---:|---:|---:|
| INTC | -2.20% | -16.72% | +14.52%p | -17.14% |
| NFLX | +4.09% | +41.10% | -37.01%p | -12.54% |
| NVDA | +20.21% | +91.32% | -71.10%p | -13.58% |

## 파일

- `fold_metrics.csv`: 세 종목의 전체 폴드 수치
- `summary.json`: 종목별 평균과 폴드 집계
- `TICKER/fold_metrics.csv`: 개별 종목 폴드 수치
- `TICKER/summary.json`: 개별 종목 요약
- `figures/return_comparison.svg`: 평균 수익률 비교
- `figures/drawdown_comparison.svg`: 모델 평균 MDD 비교
- `TICKER/backtests/`: 로컬에서 생성된 인터랙티브 백테스트 HTML

빠른 검증 결과와 중간 로그는 포함하지 않았습니다. HTML 파일에서 거래가 생성되지 않은 폴드는 모델 수익률, MDD와 거래 횟수를 0으로 기록했습니다.
