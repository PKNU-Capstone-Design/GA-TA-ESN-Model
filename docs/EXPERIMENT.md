# 실험 설계 및 결과 기록 기준

이 문서는 README에 공개할 최종 결과의 재현 조건과 기록 형식을 정의합니다.

## 평가 목표

모델의 절대 수익률뿐 아니라 다음 항목을 함께 평가합니다.

- 모델 수익률
- Buy & Hold 수익률
- 초과 수익률
- 최대 낙폭(MDD)
- 거래 횟수와 승률
- 폴드 간 결과 편차

## 데이터 분할

미래 데이터가 과거 학습에 포함되지 않도록 시간 순서를 유지하는 Expanding Window 방식을 사용합니다.

```text
Fold 1: [Train][Validation][Test]
Fold 2: [------ Train ------][Validation][Test]
Fold 3: [------------- Train -------------][Validation][Test]
```

- Train: 기술적 지표 파라미터 최적화 및 ESN 학습
- Validation: ESN 하이퍼파라미터 선택
- Test: 최종 성능 평가

## 실행 프로필

### 빠른 검증

코드와 의존성의 정상 동작을 확인하기 위한 설정입니다.

| 항목 | 값 |
|---|---:|
| Fold | 2 |
| 초기 학습 비율 | 0.7 |
| ESN Population / Generation | 4 / 2 |
| TA Population / Generation | 4 / 2 |

이 결과는 최종 성능 비교에 사용하지 않습니다.

### 전체 실험

| 항목 | 값 |
|---|---:|
| Fold | 5 |
| 초기 학습 비율 | 0.5 |
| ESN Population / Generation | 30 / 30 |
| TA Population / Generation | 50 / 50 |
| 초기 자금 | 10,000 |
| 거래 수수료 | 0.2% |
| Random State | 42 |

## 결과 공개 원칙

1. 대표 그래프만 선택하더라도 실행한 폴드의 핵심 수치는 모두 공개합니다.
2. 빠른 검증 결과와 전체 실험 결과를 혼합하지 않습니다.
3. 종목은 결과를 확인하기 전에 선정하고 선정 근거를 기록합니다.
4. 실패한 폴드를 제외할 경우 제외 사유와 오류를 함께 기록합니다.
5. 백테스트 결과는 미래 수익을 보장하지 않는다는 점을 명시합니다.

## 결과 파일 형식

최종 결과는 다음 구조로 정리할 예정입니다.

```text
results/
└─ TICKER/
   ├─ fold_metrics.csv
   ├─ summary.json
   ├─ return_comparison.png
   ├─ drawdown_comparison.png
   └─ backtests/
      ├─ fold_1.html
      ├─ fold_2.html
      ├─ fold_3.html
      ├─ fold_4.html
      └─ fold_5.html
```

`fold_metrics.csv`에는 최소한 다음 열을 기록합니다.

```text
ticker, fold, test_start, test_end, model_return_pct,
buy_hold_return_pct, excess_return_pct, max_drawdown_pct,
trades, win_rate_pct
```

