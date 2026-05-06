# Multiscale Entropy Analysis: A New Method to Detect Determinism in a Time Series

A. Sarkar and P. Barat (Year not explicitly stated, but references include 2005)

## 🧩 Problem to Solve

물리적 시스템에서 생성되는 시계열(time series) 데이터는 시스템 내부의 역학(dynamics)에 관한 정보를 포함하는 복잡한 변동성을 보인다. 이때 가장 중요한 분석 문제 중 하나는 해당 시계열이 단순히 확률적인 과정(stochastic process)에 의해 생성되었는지, 아니면 유한한 자유도를 가진 혼돈 역학(chaotic dynamics)에 의한 결정론적 성분(deterministic component)을 가지고 있는지를 판별하는 것이다.

시계열의 결정론적 성분 유무는 이후 해당 데이터를 분석하기 위해 어떤 방법론을 적용할지를 결정하는 기준이 되므로, 이를 정확하게 탐지하는 것은 매우 중요하다.

## ✨ Key Contributions

본 논문은 기존에 복잡성(complexity) 측정 도구로 제안되었던 Multiscale Entropy (MSE) 분석 방법이 시계열 데이터 내의 결정론(determinism)을 탐지하는 데 효과적으로 사용될 수 있음을 입증하였다. 핵심 아이디어는 단일 척도에서의 엔트로피 분석이 아닌, 여러 시간 척도(multiple time scales)에서 정규성(regularity)을 평가함으로써 결정론적 시스템과 확률적 시스템의 차별적인 복잡성 패턴을 찾아내는 것이다.

## 📎 Related Works

기존의 비선형 역학 분석 방법들은 결정론적 시계열의 상태 공간(state space)에서 재구성된 궤적이 시간이 흐름에 따라 인접한 궤적과 유사하게 행동한다는 가정을 바탕으로 결정론을 탐지하였다. 그러나 이러한 기존 방식들은 다음과 같은 한계점을 가진다.

- **데이터 요구량:** 인접 궤적의 미래 행동을 비교하기 위해 매우 많은 양의 데이터 포인트가 필요하다.
- **비정상성(Nonstationarity) 취약성:** 시계열이 비정상성(nonstationary)을 띨 경우 잘못된 결과(spurious results)를 도출할 가능성이 높다.

반면, 본 논문에서 활용하는 MSE 방법의 기반이 되는 Sample Entropy (SampEn)는 데이터 포인트가 약 750개 이상일 경우 시계열의 길이에 크게 영향을 받지 않는다는 장점이 있다.

## 🛠️ Methodology

MSE 분석은 크게 두 단계의 과정으로 이루어진다. 첫 번째는 시간 척도에 따른 coarse-grained 시계열을 생성하는 것이고, 두 번째는 각 척도에서 Sample Entropy를 계산하는 것이다.

### 1. Coarse-grained Time Series 생성

주어진 1차원 이산 시계열 $\{x_1, \dots, x_i, \dots, x_N\}$에 대해, 척도 인자(scale factor) $\tau$를 적용하여 다음과 같이 연속적인 coarse-grained 시계열 $\{y_i^{(\tau)}\}$를 구성한다.

$$y_i^{(\tau)} = \frac{1}{\tau} \sum_{j=(i-1)\tau+1}^{i\tau} x_j \quad (1)$$

여기서 $\tau$는 척도 인자이며, $1 \le \tau \le N$의 범위를 가진다. 결과적으로 생성된 시계열의 길이는 $N/\tau$가 되며, $\tau=1$일 때는 원본 시계열과 동일하다.

### 2. Sample Entropy (SampEn) 계산

각 척도 $\tau$에 대해 다음 절차를 통해 SampEn을 계산한다.

- 길이가 $N$인 시계열에서 길이 $m$인 벡터 $u_m(i)$를 정의한다.
- $u_m(i)$와 거리 $r$ 이내에 있는 다른 벡터 $u_m(j)$의 개수를 $n_m(r)$이라고 한다 (자기 자신은 제외).
- 임의의 $u_m(j)$가 $u_m(i)$의 거리 $r$ 이내에 존재할 확률 $C_m(r)$을 다음과 같이 정의한다.
$$C_m(r) = \frac{n_m(r)}{N-m}$$

최종적인 Sample Entropy는 $m$ 길이의 벡터와 $m+1$ 길이의 벡터 사이의 확률 비율을 통해 다음과 같이 정의된다.

$$\text{SampEn}(m, r, N) = -\ln \left( \frac{C_{m+1}(r)}{C_m(r)} \right) \quad (4)$$

본 연구에서는 모든 데이터셋에 대해 파라미터 $m=2$와 $r=0.15 \times SD$ (SD는 원본 시계열의 표준편차)를 사용하였다.

## 📊 Results

### 실험 설정

- **비교 데이터셋:**
  - **결정론적 혼돈 데이터 (Deterministic Chaotic Data):** Logistic Map, Henon Map, Ikeda Map, Quadratic Map, Rossler Equation, Lorenz Equation.
  - **확률적 노이즈 (Stochastic Noise):** White noise, $1/f$ noise, fractional Brownian noise (Hurst exponent 0.7).
- **데이터 규모:** 각 시계열당 20,000개의 데이터 포인트 사용.
- **측정 지표:** 척도 인자 $\tau$의 변화에 따른 Sample Entropy의 값.

### 분석 결과

실험 결과, 시계열의 성격에 따라 $\tau$에 따른 SampEn의 변화 양상이 뚜렷하게 구분되었다.

- **결정론적 혼돈 데이터:** 작은 척도(small scales)에서는 엔트로피 값이 증가하다가, 척도가 커짐에 따라 점차 감소하는 경향을 보인다. 이는 큰 척도에서 시스템의 복잡성이 감소함을 의미한다.
- **White Noise:** 척도가 증가함에 따라 엔트로피 값이 단조 감소(monotonically decrease)하며, 척도가 5보다 커지면 $1/f$ 노이즈보다 낮은 값을 가진다.
- **$1/f$ Noise:** 모든 척도에서 엔트로피 값이 거의 일정하게 유지(invariant)된다. 이는 $1/f$ 노이즈가 여러 척도에 걸쳐 복잡한 구조를 가지고 있음을 시사한다.
- **Fractional Brownian Noise:** White noise 및 $1/f$ noise와는 또 다른 독자적인 변화 패턴을 보인다.

결과적으로 모든 혼돈 데이터셋이 유사한 형태의 SampEn 변화 추세를 보였으며, 이는 MSE 분석이 시계열의 결정론 여부를 판별하는 유효한 도구가 될 수 있음을 입증한다.

## 🧠 Insights & Discussion

본 논문의 강점은 기존의 상태 공간 재구성 기반 방법들이 가졌던 데이터 요구량의 한계와 비정상성 데이터에 대한 취약성을 MSE라는 새로운 접근법으로 해결하려 했다는 점이다. 특히, 단일 척도의 엔트로피 분석으로는 구별하기 어려운 White noise와 $1/f$ noise의 복잡성 차이를 다중 척도 분석을 통해 명확히 구분해낸 점이 인상적이다.

다만, 논문에서 제시된 분석은 시뮬레이션된 합성 데이터(synthetic data)에 국한되어 있다. 실제 물리적 시스템에서 발생하는 노이즈가 섞인 결정론적 시계열(noisy deterministic time series)에서도 이 방법이 얼마나 강건하게(robust) 결정론을 탐지할 수 있는지에 대한 논의는 부족하다. 또한, $\tau$의 범위 설정이나 $m, r$ 파라미터 선정에 대한 이론적 근거보다는 기존 문헌의 설정을 따른 경향이 있다.

## 📌 TL;DR

본 연구는 다중 척도 엔트로피(MSE) 분석을 통해 시계열 데이터가 확률적인지 또는 결정론적인지를 판별할 수 있음을 보였다. 결정론적 혼돈 시스템은 척도가 증가함에 따라 엔트로피가 증가 후 감소하는 특유의 패턴을 보이는 반면, 확률적 노이즈는 이와 다른 양상을 보인다. 이 방법은 적은 양의 데이터로도 적용 가능하며 비정상성 데이터에 비교적 강건하여, 향후 복잡한 물리/생물학적 신호의 역학 구조를 분석하는 데 중요한 역할을 할 것으로 기대된다.
