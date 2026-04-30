# Real-time Anomaly Detection for Multivariate Data Streams

Kenneth Odoh(2018)

## 🧩 Problem to Solve

본 논문은 다변량 데이터 스트림(Multivariate Data Streams)에서 실시간으로 이상치(Anomaly)를 탐지하는 문제를 해결하고자 한다. 데이터 스트림 환경에서는 데이터의 통계적 특성이 시간에 따라 변하는 Concept Drift가 빈번하게 발생하며, 특히 다음과 같은 세 가지 형태의 신호 변화(Signal changes)가 나타날 수 있다:
1. **Abrupt Transient Shift**: 일시적이고 급격한 변화.
2. **Abrupt Distributional Shift**: 분포의 급격한 변화.
3. **Gradual Distributional Shift**: 분포의 점진적인 변화.

기존의 정적(Static) 분석 알고리즘은 모든 데이터를 메모리에 적재해야 하므로 실시간 스트림 처리에 부적합하며, 단순한 EWMA(Exponential Weighted Moving Average) 방식은 급격한 변화에 취약하거나 분포 변화를 제대로 반영하지 못하는 한계가 있다. 따라서 본 연구의 목표는 레이블이 없는 상태에서(Unsupervised) 다양한 데이터 시프트에 회복 탄력성을 가지며, 시간 및 공간 효율적인 실시간 다변량 이상 탐지 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단변량 시계열 데이터에 사용되던 **PEWMA(Probabilistic Exponentially Weighted Moving Average)** 알고리즘을 다변량 환경으로 확장하는 것이다. 이를 위해 단순히 평균을 추적하는 것을 넘어, 데이터의 상관관계를 나타내는 공분산 행렬(Covariance Matrix)과 그 역행렬(Inverse Covariance Matrix)을 실시간으로 업데이트하는 효율적인 온라인 메커니즘을 제안하였다. 특히, 계산 비용이 높은 행렬 역연산을 피하기 위해 **Sherman-Morrison 공식**을 도입하여 계산 효율성을 극대화한 점이 주요한 설계 특징이다.

## 📎 Related Works

논문에서는 다음과 같은 기존 접근 방식과 그 한계를 언급한다:
- **EWMA (Exponential Weighted Moving Average)**: 최근 데이터에 가중치를 두는 평활화 기법이지만, 급격한 과도 응답(Transient changes) 시 변동성이 크고 분포 시프트(Distribution shifts)에 적합하지 않다.
- **PEWMA (Probabilistic EWMA)**: 단변량 데이터에서 데이터의 확률 밀도를 반영한 파라미터를 추가하여 다양한 시프트에 대응 가능하도록 개선된 모델이다. 본 논문은 이 아이디어를 다변량으로 확장한다.
- **Diagnosis vs Accommodation**: 이상치를 제거하여 분포 왜곡을 막는 진단(Diagnosis) 방식과, Concept Drift를 반영하여 모델 파라미터를 업데이트하는 수용(Accommodation) 방식이 있다. 본 논문의 알고리즘은 후자인 Accommodation 모드로 동작하여 데이터 스트림의 변화를 모델에 반영한다.

## 🛠️ Methodology

제안된 시스템은 정적 단계에서 초기 파라미터를 추정하고, 이후 온라인 단계에서 데이터를 증분적으로 업데이트하는 'Static + Online' 하이브리드 방식으로 작동한다.

### 1. PEWMA의 기본 원리 (단변량 기반)
PEWMA는 데이터의 확률 밀도 함수(PDF)를 이용하여 가중치 $\alpha$를 동적으로 조절한다. 
$$ \mu_{t+1} = \alpha(1-\beta p_t)\mu_{t-1} + (1-\alpha(1-\beta p_t))x_t $$
여기서 $p_t$는 현재 데이터가 모델의 분포에 속할 확률이며, 이를 통해 Concept Drift가 발생했을 때 모델이 얼마나 빠르게 새로운 상태를 수용할지를 결정한다.

### 2. 온라인 공분산 행렬 업데이트 (Online Covariance Matrix)
다변량 데이터의 상관관계를 추적하기 위해 공분산 행렬 $C$를 실시간으로 업데이트한다.
- **초기화**: 초기 데이터 $X \in \mathbb{R}^{n \times m}$를 사용하여 $C = XX^T / N$으로 설정하고, Cholesky 분해 $C_t = A_t A_t^T$를 수행한다.
- **업데이트**: 새로운 데이터 $x_t$가 들어오면 다음과 같이 가중 평균 방식으로 업데이트한다.
$$ C_{t+1} = \alpha C_t + \beta m_t m_t^T $$
이때 $\alpha$와 $\beta$는 데이터 스트림의 통계적 특성을 반영하여 $\alpha = C_a^2, \beta = 1 - C_a^2$로 설정하며, $C_a$는 데이터 크기에 기반한 상수 값으로 결정된다.

### 3. 온라인 역공분산 행렬 (Online Inverse Covariance Matrix)
이상치 탐지를 위한 가우시안 확률 밀도 계산에는 역공분산 행렬 $C^{-1}$이 필요하다. 매번 역행렬을 계산하는 것은 비용이 매우 크므로, **Sherman-Morrison 공식**을 사용하여 다음과 같이 증분적으로 업데이트한다.
$$ C^{-1}_{t+1} = \frac{1}{\alpha} \left( C^{-1}_t - \frac{C^{-1}_t \hat{m}_t m_t^T C^{-1}_t}{1 + \hat{m}_t^T C^{-1}_t m_t} \right) $$
여기서 $\hat{m}_t = \frac{\beta m_t}{\alpha}$이며, 이 방식을 통해 행렬 역연산을 덧셈과 곱셈 수준의 연산으로 단순화하였다.

### 4. 다변량 이상 탐지 절차
최종적으로 새로운 데이터 $x$에 대해 다음과 같은 가우시안 PDF를 계산하여 이상 여부를 판단한다.
$$ p(x) = \frac{1}{\sqrt{(2\pi)^m |C|}} \exp \left( -\frac{1}{2}(x-\mu)^T C^{-1} (x-\mu) \right) $$
여기서 $\mu$는 이동 평균으로 업데이트되는 평균 벡터이다. 계산된 확률 값 $p(x)$가 미리 설정된 임계값(Acceptance region) 밖에 위치하면 이를 이상치로 간주한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 15차원의 랜덤 벡터 10,000,000개를 생성하여 시뮬레이션 수행.
- **평가 지표**: Absolute Average Deviation (AAD)을 사용한다.
$$ \text{AAD} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{\hat{y}_i - y_i}{y_i} \right| $$
- **실험 설계**: 데이터를 5개 세그먼트로 나누어, 정적 학습(Static phase)에 사용하는 세그먼트의 수를 변화시키며 온라인 업데이트 시의 오차를 측정하였다.

### 주요 결과
- **정적 윈도우 크기의 영향**: 초기 정적 학습 단계에서 더 많은 데이터를 사용할수록, 이후 온라인 업데이트 단계에서의 오차가 감소하는 경향을 보였다. 이는 초기 모델의 수렴도가 이후 온라인 추적 성능에 직접적인 영향을 미침을 시사한다.
- **초기 오차**: 학습 초기 단계에서는 두 실험 케이스 모두에서 유의미하게 높은 오차가 발생하였다. 저자는 이를 양정치(Positive-definite) 행렬을 생성하는 과정의 비용 문제로 분석하며, 처음부터 양정치인 랜덤 행렬을 사용하는 것이 더 효율적일 수 있다고 언급하였다.
- **강건성**: 알고리즘이 초기 시드 값이나 행렬의 차원에 민감하게 반응하지 않고 안정적으로 동작함을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 강점은 계산 복잡도가 높은 다변량 통계 업데이트 과정을 Sherman-Morrison 공식과 Cholesky 분해를 통해 실시간 처리가 가능하도록 최적화했다는 점이다. 또한, PEWMA의 확률적 가중치 메커니즘을 통해 Concept Drift에 유연하게 대응할 수 있는 구조를 갖추었다.

다만, 다음과 같은 한계와 논의 사항이 존재한다:
- **가정의 한계**: 본 연구는 특징량(Features)들이 시스템의 동적 특성을 충분히 캡처하고 있다는 가정하에 작동한다. 만약 특징량 선택이 잘못되었다면 알고리즘의 성능은 보장되지 않는다.
- **비정상 분포**: 현재 모델은 가우시안 분포를 기본으로 하고 있으나, 실제 데이터 스트림은 비정상(Non-stationary) 분포를 가질 수 있다. 저자 또한 이를 향후 연구 과제로 명시하였다.
- **임계값 설정**: 단변량에서는 $3\sigma$ 원칙을 언급하였으나, 다변량 환경에서 정확히 어떤 기준으로 Acceptance region의 임계값을 설정했는지에 대한 구체적인 수치적 근거가 부족하다.

## 📌 TL;DR

본 논문은 다변량 데이터 스트림에서 실시간으로 이상치를 탐지하기 위해 **PEWMA를 확장한 알고리즘**을 제안한다. **Sherman-Morrison 공식**을 이용해 역공분산 행렬을 효율적으로 업데이트함으로써 계산 비용을 줄였으며, 이를 통해 Concept Drift가 발생하는 환경에서도 레이블 없이(Unsupervised) 강건하게 이상치를 탐지할 수 있다. 이 연구는 실시간 시스템 모니터링 및 다변량 센서 데이터 분석 분야에 적용될 가능성이 높다.