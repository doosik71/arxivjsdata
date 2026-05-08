# tsbootstrap: Enhancing Time Series Analysis with Advanced Bootstrapping Techniques

Sankalp Gilda, Benedikt Heidrich, Franz Kiraly (2024)

## 🧩 Problem to Solve

시계열 데이터 분석에서 가장 큰 도전 과제 중 하나는 데이터의 시간적 의존성(Temporal Dependency)을 처리하는 것이다. 기존의 전통적인 Bootstrapping 방법론은 데이터가 독립적이고 동일한 분포를 가진다(Independent and Identically Distributed, IID)는 가정을 전제로 한다. 하지만 시계열 데이터는 관측치 간의 순차적 관계와 복잡한 상관관계가 존재하므로, IID 가정을 적용한 단순 재표본 추출은 데이터의 내재적 구조를 파괴하여 부정확한 결과를 초래한다.

이러한 문제는 특히 미래 사건을 예측하는 Forecasting과 예측의 신뢰도를 평가하는 불확실성 정량화(Uncertainty Quantification) 단계에서 치명적이다. 금융의 리스크 평가, 기상학의 예측 정확도, 역학의 공중 보건 결정 등 정밀한 불확실성 분석이 필수적인 분야에서 기존 도구의 한계는 매우 크다. 따라서 본 논문은 시계열의 시간적 구조를 보존하면서도 강건한 통계적 추론을 가능하게 하는 소프트웨어 프레임워크를 제공하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 시계열 데이터의 특성을 반영한 다양한 Bootstrapping 기법을 통합 구현한 Python 패키지인 `tsbootstrap`의 제안이다.

중심적인 설계 아이디어는 데이터의 개별 포인트가 아닌 '블록(Block)' 단위의 샘플링이나 모델 기반의 잔차 샘플링을 통해 시계열의 연쇄적 의존성을 유지하는 것이다. 또한, 단순한 알고리즘 구현을 넘어 `scikit-learn`과 `sktime`과 같은 기존 Python 데이터 과학 생태계와 원활하게 통합될 수 있도록 모듈형 아키텍처를 설계하여, 연구자와 실무자가 복잡한 통계적 구현 없이도 불확실성 정량화 도구를 파이프라인에 쉽게 통합할 수 있도록 하였다.

## 📎 Related Works

전통적인 Bootstrapping은 Efron(1992) 등에 의해 정립되었으며, 데이터의 분포에 대한 엄격한 가정 없이 통계량의 변동성을 추정하는 비모수적 방법으로 널리 사용되었다. 하지만 시계열 데이터에 이를 적용하기 위해 Kunsch(1989)의 Block Bootstrap이나 Politis와 Romano(1994)의 Stationary Bootstrap과 같은 특화된 방법론들이 제안되었다.

기존 연구들의 한계는 이러한 고도화된 시계열 Bootstrapping 기법들이 이론적으로는 존재하나, 실제 분석 환경에서 쉽게 사용할 수 있는 통합된 소프트웨어 라이브러리 형태로 제공되지 않았다는 점이다. `tsbootstrap`은 이러한 소프트웨어적 공백을 메우기 위해 다양한 알고리즘을 하나의 일관된 API로 제공하며, 특히 `skbase`를 통해 인터페이스 표준화를 달성함으로써 기존 라이브러리들과의 상호운용성을 확보하였다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 시스템 구조 및 설계 원칙

`tsbootstrap`은 모듈성과 확장성을 최우선으로 설계되었다. Strategy 패턴과 Composition을 채택하여 새로운 Bootstrapping 알고리즘을 쉽게 추가할 수 있는 구조를 가진다. 특히 `scikit-learn`의 Cross-Validation 클래스 구조를 벤치마킹하여, 데이터를 분할하는 `split` 메서드 대신 시계열 구조를 보존하며 샘플링하는 `bootstrap` 메서드를 중심으로 API를 구성하였다.

### 2. 주요 Bootstrapping 알고리즘

본 패키지는 시계열 특성에 따라 다음과 같은 알고리즘들을 제공한다.

#### A. Block Bootstrap Methods (블록 기반 방법론)

데이터를 일정 길이의 블록으로 나누어 샘플링함으로써 인접 데이터 간의 상관관계를 유지한다.

- **Moving Block Bootstrap (MBB):** 겹치는(overlapping) 블록을 샘플링하며, 단기 의존성이 강한 금융 데이터 등에 적합하다.
- **Circular Block Bootstrap (CBB):** 시계열의 끝과 시작을 연결하여 원형으로 간주하고 샘플링한다. 계절성이나 주기성이 있는 데이터에 유리하다.
- **Stationary Block Bootstrap (SBB):** 블록의 길이를 고정하지 않고 확률 분포에 따라 가변적으로 설정하여, 단기와 장기 의존성을 동시에 포착한다.
- **Non-Overlapping Block Bootstrap (NBB):** 블록이 겹치지 않게 샘플링하여 중복성을 줄이며, 데이터가 명확한 세그먼트로 구분될 때 효과적이다.
- **Tapered Block Bootstrap (TBB):** 블록의 경계에서 발생하는 불연속성(edge effects)을 줄이기 위해 Hamming, Hanning 등의 윈도우 함수를 적용하여 가중치를 조절한다.

#### B. Advanced Bootstrap Methods (고도화된 방법론)

- **Residual Bootstrap:** 시계열 모델을 먼저 적합시킨 후, 실제 값과 예측 값의 차이인 잔차(Residuals)를 샘플링한다. 모델 기반의 불확실성을 평가하는 데 사용된다.
- **Statistic-Preserving Bootstrap:** 평균이나 분산 등 사용자가 정의한 핵심 통계적 특성을 유지하면서 샘플을 생성한다.
- **Distribution Bootstrap:** 데이터의 기반 분포가 알려져 있거나 추정 가능할 때, 해당 분포로부터 샘플을 생성한다.
- **Markov Bootstrap:** 미래 상태가 오직 현재 상태에만 의존하는 마르코프 성질을 가진 데이터에 적용하며, 주로 금융 시장의 확률적 프로세스 분석에 사용된다.
- **Sieve Bootstrap:** 자기회귀(Autoregressive) 구조를 가진 시계열을 위해 설계되었으며, AR 모델의 계수를 활용해 샘플링한다.

### 3. 통합 및 추론 절차

`tsbootstrap`은 `TSBootstrapAdapter`를 통해 `sktime` 라이브러리와 결합된다. 특히 `BaggingForecaster`와의 연동을 통해 다음과 같은 절차로 확률적 예측(Probabilistic Forecasting)을 수행한다.

1. `tsbootstrap` 알고리즘을 통해 원본 시계열 데이터로부터 여러 개의 Bootstrapped 샘플을 생성한다.
2. 각 샘플에 대해 베이스 모델(예: SARIMA)을 개별적으로 학습시킨다.
3. 추론 시, 학습된 모든 모델로부터 예측값을 얻어 이들의 분포를 통해 예측 구간(Prediction Interval)을 산출함으로써 불확실성을 정량화한다.

## 📊 Results

본 논문은 구체적인 수치적 벤치마크 테이블보다는 프레임워크의 유용성과 통합 가능성을 입증하는 사례 연구(Case Study) 중심으로 결과를 제시한다.

- **실험 설정:** `sktime`에서 제공하는 `airline` 데이터셋을 사용하여 확률적 예측 성능을 테스트하였다.
- **구현 파이프라인:** `Deseasonalizer` $\rightarrow$ `Detrender` $\rightarrow$ `BaggingForecaster` 순의 파이프라인을 구축하였으며, 이때 `BaggingForecaster`의 부트스트랩 엔진으로 `tsbootstrap`의 `BlockResidualBootstrap`을 사용하고 베이스 모델로 `ARIMA(1, 1, 0)`를 적용하였다.
- **결과:** 시뮬레이션 결과, 단순한 점 예측(Point Forecast)과 달리 `tsbootstrap`을 통해 생성된 예측 구간(Prediction Intervals)이 시계열의 변동성과 불확실성을 시각적으로 잘 나타냄을 확인하였다. 이는 사용자가 예측값의 신뢰 범위를 정량적으로 파악할 수 있게 함을 의미한다.

## 🧠 Insights & Discussion

### 강점

`tsbootstrap`은 단순한 알고리즘 구현체를 넘어, 현대적인 ML 파이프라인(`scikit-learn`, `sktime`)에 즉시 투입 가능한 소프트웨어 공학적 완성도를 갖추고 있다. 특히 $\text{skbase}$를 활용한 인터페이스 표준화는 커뮤니티의 확장성을 높여, 새로운 시계열 샘플링 기법이 등장했을 때 빠르게 라이브러리에 반영될 수 있는 구조를 제공한다.

### 한계 및 미해결 질문

논문에서 명시적으로 언급된 한계점은 현재 블록 크기(Block size) 결정이 사용자의 설정에 의존한다는 점이다. 시계열의 자기상관성(Autocorrelation)을 분석하여 블록 크기를 자동으로 최적화하는 적응형 알고리즘(Adaptive Algorithm)이 아직 구현되지 않았으며, 이는 향후 과제로 남아 있다. 또한, 대규모 데이터셋 처리를 위한 분산 컴퓨팅 백엔드(Dask, Spark 등)와의 통합이 아직 이루어지지 않아 확장성 측면의 개선이 필요하다.

### 비판적 해석

본 논문은 이론적 기여보다는 '도구적 기여'에 집중하고 있다. 따라서 기존의 Block Bootstrap 방법론들보다 통계적으로 얼마나 더 우수한지를 증명하는 정량적 성능 비교(예: Coverage probability 분석 등)가 부족한 점은 아쉽다. 그러나 시계열 분석 생태계에서 불확실성 정량화 도구의 접근성을 높였다는 점은 실무적으로 매우 높은 가치를 지닌다.

## 📌 TL;DR

`tsbootstrap`은 시계열 데이터의 시간적 의존성을 보존하며 샘플링할 수 있는 다양한 Bootstrapping 기법(Block, Residual, Markov, Sieve 등)을 제공하는 Python 패키지이다. 이 연구는 `sktime` 및 `scikit-learn`과의 매끄러운 통합을 통해 시계열 예측의 불확실성을 정량화하는 표준적인 프레임워크를 제공하며, 향후 자동 블록 크기 튜닝 및 분산 처리 기능을 통해 고도화될 가능성이 크다.
