# MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns

Kasun Bandara, Rob J Hyndman, Christoph Bergmeir

## 🧩 Problem to Solve

최근 고주파 샘플링(예: 일별, 시간별, 분별 데이터)으로 인해 시계열 데이터는 종종 여러 계절성 패턴을 포함합니다. 이러한 다중 계절성 시계열을 추세, 계절성, 잔차와 같은 구성 요소로 정확하고 효율적으로 분해하는 것은 데이터 이해 및 예측 정확도 향상에 중요합니다. 그러나 기존의 시계열 분해 방법들은 단일 계절성에만 초점을 맞추거나, 다중 계절성을 다룰 수 있더라도 계산 효율성이 떨어지거나 정확도가 낮은 한계가 있었습니다.

## ✨ Key Contributions

- **MSTL 알고리즘 제안**: 다중 계절성 패턴을 가진 시계열의 분해를 위해 전통적인 STL(Seasonal-Trend decomposition using Loess) 절차를 확장한 MSTL(Multiple Seasonal-Trend decomposition using Loess)을 제안합니다.
- **반복적 STL 적용**: STL 절차를 반복적으로 적용하여 시계열 내의 여러 계절성 구성 요소를 추정합니다. 이를 통해 각 계절성 주기에 대한 계절성 구성 요소의 변화 평활도를 제어하고, 결정론적/확률적 계절성 변동을 원활하게 분리합니다.
- **자동화되고 부가적인 분해**: MSTL은 완전 자동화된 부가적(additive) 시계열 분해 알고리즘입니다.
- **경쟁력 있는 성능과 낮은 계산 비용**: 시뮬레이션 데이터 및 실제 데이터셋 평가에서 다른 최신 분해 벤치마크(STR, TBATS, Prophet)에 비해 경쟁력 있는 정확도를 제공하며, 현저히 낮은 계산 비용을 보여줍니다.
- **R 패키지 구현**: MSTL의 구현은 R `forecast` 패키지에서 사용할 수 있어 접근성이 높습니다.

## 📎 Related Works

- **전통적인 단일 계절성 분해**:
  - STL(Seasonal-Trend decomposition using Loess)
  - X-13-ARIMA-SEATS
  - X-12-ARIMA
- **다중 계절성 분해 및 예측 기반 방법**:
  - STR(Seasonal-Trend decomposition by Regression): 회귀 기반의 부가적 분해 기법.
  - Fast-RobustSTL: 다중 계절성 및 노이즈를 처리하는 분해 기법.
  - TBATS(Trigonometric Exponential Smoothing State Space model with Box-Cox transformation, ARMA errors, Trend and Seasonal Components): 다중 계절성 시계열 예측 모델로, 예측 모델의 구성 요소를 분해에 활용할 수 있으나 예측 오류 최소화에 중점을 둡니다.
  - Prophet: Facebook에서 개발한 자동 예측 프레임워크로, 다중 계절성 패턴을 처리하며 부가적 분해 기법입니다.

## 🛠️ Methodology

MSTL은 시계열 $X_t$를 여러 계절성 구성 요소($\hat{S}_{1t}, \dots, \hat{S}_{nt}$), 추세($\hat{T}_t$), 잔차($\hat{R}_t$)로 분해하는 부가적 모델을 따릅니다:
$$X_t = \hat{S}_{1t} + \hat{S}_{2t} + \cdots + \hat{S}_{nt} + \hat{T}_t + \hat{R}_t$$
알고리즘의 주요 단계는 다음과 같습니다:

1. **계절성 패턴 식별**: 시계열에 존재하는 고유한 계절성 주기를 식별하고 오름차순으로 정렬합니다. (시계열 길이의 절반보다 큰 주기는 무시합니다.)
2. **결측치 처리 및 변환**: `na.interp` 함수를 사용하여 결측치를 보간하고, 주어진 $\lambda$ 값에 따라 Box-Cox 변환을 적용합니다.
3. **반복적 STL 적용 (내부/외부 루프)**:
   - **내부 루프**: 식별된 각 계절성 주기에 대해 STL 알고리즘을 반복적으로 적용하여 계절성 구성 요소를 추출합니다. 이 과정에서 `s.window` 매개변수를 통해 각 계절성 주기의 변동 평활도를 개별적으로 제어할 수 있습니다. 낮은 주기의 계절성 구성 요소가 높은 주기에 과도하게 흡수되는 '계절성 교란(seasonal confounding)'을 최소화하기 위해 주기를 오름차순으로 정렬하여 적용합니다.
   - **외부 루프**: 내부 루프를 여러 번 반복하여 추출된 계절성 구성 요소를 정제합니다.
4. **추세 구성 요소 계산**: STL의 최종 반복에서 얻은 결과를 사용하여 추세 구성 요소를 계산합니다. 시계열이 비계절성인 경우 R의 `supsmu` 함수를 사용하여 직접 추세를 추정합니다.
5. **잔차 구성 요소 계산**: 계절성이 조정된 시계열에서 추세 구성 요소를 빼서 잔차를 계산합니다. (비계절성 시계열의 경우 원본 시계열에서 추세를 뺌).

MSTL은 STL의 다른 매개변수(예: `t.window`, `l.window`)도 상속받아 사용할 수 있습니다.

## 📊 Results

- **시뮬레이션 데이터 (일별 및 시간별)**:
  - **확실론적(Deterministic) DGP**: MSTL은 대부분의 경우 TBATS보다 추세, 주간, 연간 계절성 구성 요소에서 더 나은 RMSE를 보였습니다.
  - **확률론적(Stochastic) DGP**: MSTL은 주간 RMSE에서 모든 벤치마크를 능가하며, TBATS에 비해 추세, 주간, 연간 계절성 구성 요소에서 더 좋은 결과를 얻었습니다. 시간별 데이터의 경우 Prophet보다 모든 구성 요소에서 더 나은 RMSE를 보였습니다.
- **교란된 실제 데이터 (호주 빅토리아 시간별 전력 수요)**:
  - MSTL은 모든 구성 요소(추세, 일별 계절성, 주간 계절성, 잔차) 추정에서 STR, TBATS, Prophet을 **현저하게 능가**하는 RMSE를 기록했습니다. (예: 추세 RMSE MSTL 207.6 vs STR 399.4, Prophet 243.7, TBATS 742.1)
  - **계산 효율성**: MSTL은 다른 벤치마크에 비해 압도적으로 낮은 계산 시간을 보여주었습니다. (MSTL 7초 vs STR 612초, Prophet 936초, TBATS 2521초).

## 🧠 Insights & Discussion

MSTL은 다중 계절성 시계열 분해에 있어 기존 방법들의 한계(낮은 효율성, 부정확성)를 효과적으로 해결하는 빠르고 정확한 알고리즘임을 입증했습니다. 특히, 고주파 시계열 데이터의 증가에 따라 그 중요성이 커지고 있는 상황에서 MSTL의 낮은 계산 비용은 대규모 데이터셋 처리 및 실시간 응용 분야에서 큰 장점입니다. STL의 강건하고 효율적인 특성을 계승하면서 다중 계절성을 유연하게 처리할 수 있도록 확장한 점이 MSTL의 핵심입니다. R의 `forecast` 패키지에 구현되어 있어 연구자와 실무자 모두에게 쉽게 활용될 수 있습니다.

## 📌 TL;DR

고주파 시계열 데이터에서 나타나는 다중 계절성 패턴을 효율적이고 정확하게 분해하는 문제를 해결하기 위해, 본 논문은 기존 STL 알고리즘을 확장한 MSTL(Multiple Seasonal-Trend decomposition using Loess)을 제안한다. MSTL은 여러 계절성 주기를 식별하고 이를 오름차순으로 정렬한 뒤, 각 주기에 대해 STL을 반복적으로 적용하여 추세, 다중 계절성, 잔차 구성 요소를 추출한다. 시뮬레이션 및 실제 데이터 평가 결과, MSTL은 다른 최신 분해 기법들(STR, TBATS, Prophet) 대비 경쟁력 있는 정확도를 보였을 뿐만 아니라, 특히 계산 비용 면에서 월등히 뛰어난 성능을 입증하며 다중 계절성 시계열 분해의 효율성과 확장성을 크게 개선했다.
