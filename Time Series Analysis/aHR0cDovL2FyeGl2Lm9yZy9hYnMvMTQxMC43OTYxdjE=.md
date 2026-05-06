# Visualisation of financial time series by linear principal component analysis and nonlinear principal component analysis

HAO-CHE CHEN (2014)

## 🧩 Problem to Solve

본 연구의 주된 목적은 탐색적 분석을 위해 금융 시계열 데이터(Financial Time Series)를 시각화하는 다양한 접근 방식을 분석하는 것이다. 금융 시장의 시계열 데이터는 다차원적인 특성을 가지며, 이를 효과적으로 시각화하는 것은 기술적 분석(Technical Analysis)을 통한 거래 규칙(Trading Rules) 수립에 유용한 보조 도구가 될 수 있다.

특히, 저자는 시계열 데이터를 시각화할 때 데이터가 단 하루만 밀려도(Shifted one day) 시각화 결과에서 데이터 포인트의 위치가 급격하게 변하는 'Jump Gap' 현상이 발생한다는 점을 지적한다. 금융 데이터의 특성상 하루 정도의 시간 차이는 전체적인 구조에 큰 영향을 주지 않아야 함에도 불구하고, 기존의 시각화 방법론에서는 이로 인해 데이터의 일관성이 훼손되는 문제가 발생한다. 따라서 본 논문은 시각화의 품질을 높이는 동시에 이러한 시계열 밀림 현상에 강건한(Robust) 전처리 방법과 시각화 기법을 찾는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 금융 시계열 데이터의 시각화를 위해 선형 및 비선형 주성분 분석을 비교하고, 데이터의 안정성을 확보하기 위한 최적의 전처리 파이프라인을 제안한 것이다.

저자는 단순히 원본 데이터를 사용하는 대신, 이동 프레임(Moving Frame) 내에서 로그 수익률(Log-return)의 피어슨 상관계수(Pearson Correlation Coefficient) 벡터를 생성하여 전처리하는 방식이 가장 효과적임을 입증하였다. 이러한 접근 방식은 시계열 데이터가 하루 밀리더라도 시각화 결과의 변동성을 최소화하면서, 동시에 시장의 군집 구조(Cluster Structure)를 명확하게 유지한다는 점에서 중요한 설계 아이디어를 제공한다.

## 📎 Related Works

논문에서는 기술적 분석과 효율적 시장 가설(Efficient Market Hypothesis, EMH) 사이의 대립 관계를 설명한다. Fama(1970) 등이 주장한 EMH는 시장 가격이 모든 가용 정보를 반영하므로 과거의 가격 패턴을 통한 미래 예측이 불가능하다고 주장한다. 반면, Brock et al.(1992) 등의 연구는 이동 평균(Moving Average)과 같은 단순한 기술적 규칙으로도 초과 수익을 낼 수 있음을 보여주며 기술적 분석의 유효성을 지지한다.

시각화 측면에서는 고차원 데이터를 저차원으로 투영하여 구조를 분석하는 주성분 분석(Principal Component Analysis, PCA)의 이론적 배경과 함께, 비선형 구조를 포착하기 위한 Kernel PCA 및 Elastic Map과 같은 비선형 주성분 분석(Nonlinear PCA, NLPCA) 기법들이 소개된다. 기존의 선형 PCA는 데이터의 분산을 최대화하는 방향으로 투영하지만, 복잡한 금융 데이터의 비선형적 관계를 표현하는 데에는 한계가 있다.

## 🛠️ Methodology

### 1. 데이터 전처리: Log-return

시계열 데이터의 가격 변동을 단순화하기 위해 다음과 같은 로그 수익률(Log-return) 공식을 사용한다.

$$r_{t+1} = \ln\left(\frac{P_{t+1}}{P_t}\right)$$

여기서 $P$는 주가이며, 로그 수익률은 가격의 상승, 하락 및 유지 상태를 양수, 음수, 0으로 명확하게 나타낸다.

### 2. 시각화 방법론: Linear PCA vs NLPCA

- **Linear PCA**: 데이터의 공분산 행렬(Covariance Matrix) $A$를 생성하고, 고유값 방정식 $Av = \lambda v$를 통해 고유벡터(Eigenvector) $v$와 고유값(Eigenvalue) $\lambda$를 산출한다. 이를 통해 데이터의 차원을 축소하고 주요 성분을 추출한다.
- **Nonlinear PCA (Elastic Map)**: 데이터를 비선형 다양체(Manifold)에 투영하는 방식이다. Elastic Map은 정점(Vertices)과 에지(Edges)로 구성된 그래프를 사용하며, 탄성 에너지 함수(Elastic Energy Functional)를 최소화하는 최적화 알고리즘을 통해 고차원 데이터를 2차원 또는 3차원 평면으로 투영한다.

### 3. Jump Gap 해결을 위한 전처리 기법

시계열 밀림 현상을 해결하기 위해 저자는 세 가지 전처리 방식을 실험하였다.

- **Discrete Fourier Transform (DFT)**: 로그 수익률 데이터를 주파수 영역으로 변환하고 진폭(Amplitude)의 절대값을 사용한다.
  $$|F_k| = \sqrt{\text{real}(F_k)^2 + \text{imaginary}(F_k)^2}$$
  이 방법은 밀림 현상에 의한 위치 변화는 줄여주지만, 위상(Phase) 정보가 손실되어 군집 정보가 사라지는 문제가 있다.
- **Vector Projection**: 한 주식의 시계열 벡터를 다른 주식의 벡터로 투영하여 관계를 수치화한다.
- **Pearson Correlation Coefficient**: 이동 프레임 내에서 두 시계열 간의 상관계수를 계산하여 벡터화한다.

$$\text{corr}(x, y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

## 📊 Results

### 1. 기술적 분석 전략 테스트

NASDAQ 시장의 Google, Apple, Amazon 데이터를 사용하여 세 가지 전략을 테스트하였다.

- **Strategy 1 (Day-by-day)**: 가격이 7% 상승 시 매수, 7% 하락 시 매도. (수익률: 16.12%)
- **Strategy 2 (Trend Counters)**: 고점과 저점의 발생 횟수 차이를 이용해 추세 판단. (수익률: 42.86%)
- **Strategy 3 (Volume & MA)**: 5일/15일 이동평균선 교차와 거래량 급증을 결합. (수익률: 27.53%)
결과적으로 Strategy 2의 수익률이 가장 높았으나, 거래 횟수 대비 효율성을 고려할 때 Strategy 3가 가장 우수한 전략으로 평가되었다.

### 2. PCA vs NLPCA 시각화 결과

10개 NASDAQ 종목의 데이터를 시각화한 결과, Linear PCA보다 NLPCA(Elastic Map)가 데이터의 군집 구조를 훨씬 더 명확하게 보여주었다. 특히 시간 흐름에 따라 색상을 입혔을 때, 특정 위기 기간이나 시장 변화 시점의 군집화가 뚜렷하게 나타났다.

### 3. 전처리 기법에 따른 Jump Gap 분석

- **Log-return**: 시계열이 하루 밀릴 경우 데이터 포인트의 위치가 완전히 바뀌는 심각한 Jump Gap이 발생한다.
- **DFT**: Jump Gap은 현저히 줄어들지만, 군집 구조가 뭉쳐져 변별력이 사라진다.
- **Pearson Correlation**: Jump Gap을 효과적으로 억제하면서도, 시장의 군집 구조를 가장 명확하게 유지하였다.

### 4. 타 시장 적용 결과

제안한 '피어슨 상관계수 $\to$ NLPCA' 파이프라인을 FTSE(영국)와 대만 시장에 적용하였다. FTSE 시장에서는 NASDAQ와 유사하게 우수한 군집화 성능을 보였으나, 대만 시장에서는 군집 구조가 불분명하고 데이터 포인트들이 서로 겹치는 현상이 발견되었다. 이는 대만 시장의 시계열 관계가 상대적으로 덜 체계적임을 시사한다.

## 🧠 Insights & Discussion

본 연구는 금융 시계열 데이터의 시각화가 단순히 데이터를 그리는 것이 아니라, 데이터의 정적인 특성과 동적인 변화(시계열 밀림) 사이의 균형을 맞추는 전처리가 핵심임을 보여준다.

**강점**:

- 선형과 비선형 차원 축소 기법을 체계적으로 비교하여 NLPCA의 우수성을 입증하였다.
- 실무적인 문제인 'Jump Gap'을 정의하고, 이를 해결하기 위해 DFT부터 상관계수까지 단계적인 실험적 접근을 시도하였다.

**한계 및 비판적 해석**:

- NLPCA의 구현 도구인 Elastic Map의 설정 값(Bending, Expanding 계수)에 따라 결과가 크게 달라지는데, 이에 대한 객관적인 최적화 기준이 제시되지 않았다. 저자 스스로도 이 부분이 "unreliable" 할 수 있음을 언급하였다.
- 대만 시장에서의 낮은 성능 원인에 대해 단순히 "관계가 좋지 않다"고 결론지었으나, 시장의 규모나 변동성 특성 등 구체적인 분석이 부족하다.
- 제시된 기술적 분석 전략들이 수수료를 제외한 시뮬레이션 결과이므로, 실제 거래 환경에서의 유효성은 추가 검증이 필요하다.

## 📌 TL;DR

이 논문은 금융 시계열 데이터의 효과적인 시각화를 위해 **Log-return $\to$ Pearson Correlation Coefficient $\to$ Nonlinear PCA (Elastic Map)**로 이어지는 파이프라인을 제안한다. 특히, 시계열 데이터가 하루 밀릴 때 시각화 결과가 급변하는 'Jump Gap' 문제를 피어슨 상관계수 기반의 전처리를 통해 성공적으로 해결하였으며, 이를 통해 시장의 군집 구조를 안정적으로 분석할 수 있음을 입증하였다. 이 연구는 향후 금융 시장의 패턴 인식 및 탐색적 데이터 분석을 위한 강건한 시각화 도구로서 활용될 가능성이 높다.
