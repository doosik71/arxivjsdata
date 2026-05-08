# WAFFLE: Weighted Averaging for Personalized Federated Learning

Martin Beaussart, Felix Grimberg, Mary-Anne Hartley, Martin Jaggi (2021)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 환경에서 클라이언트 간의 데이터 불균형, 즉 비독립 동일 분포(non-IID) 데이터로 인해 발생하는 성능 저하 문제를 해결하고자 한다. 일반적인 FL 알고리즘은 모든 클라이언트에게 적용되는 하나의 전역 모델(global model)을 학습시키지만, 실제 환경에서는 클라이언트마다 데이터 분포가 매우 다르기 때문에 '단일 모델' 방식은 개별 클라이언트의 특성을 반영하지 못하는 한계가 있다.

특히 비독립 동일 분포 데이터 환경에서는 클라이언트 드리프트(client drift) 현상이 발생하여 수렴 속도가 느려지고 최종 성능이 하락하는 문제가 나타난다. 따라서 본 연구의 목표는 모든 클라이언트에게 동일한 모델을 제공하는 대신, 협력적 학습의 이점을 누리면서도 개별 클라이언트의 특성에 최적화된 개인화 모델(personalized model)을 효율적으로 생성하는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 기존의 전역 모델 수렴 가속화 기법인 SCAFFOLD와 개인화 가중치 기법인 Weight Erosion(WE)을 결합한 **WAFFLE(Weighted Averaging For Federated LEarning)** 알고리즘을 제안한 점이다.

WAFFLE의 중심 아이디어는 다음과 같다. 첫째, Stochastic Control Variates(SCV)를 도입하여 비독립 동일 분포 데이터에서 발생하는 클라이언트 드리프트를 억제하고 수렴 속도를 높인다. 둘째, 클라이언트 간의 그래디언트 유클리드 거리(Euclidean distance)를 기반으로 가중치를 동적으로 할당하여, 특정 타겟 클라이언트(Alice)와 유사한 데이터를 가진 클라이언트의 기여도를 높임으로써 개인화 성능을 극대화한다. 이를 통해 학습 초기에는 전역적인 지식을 학습하고, 학습이 진행됨에 따라 점진적으로 개인화된 로컬 학습으로 전환하는 전략을 취한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 그 한계를 제시한다.

- **Federated Averaging (FedAvg):** 가장 기본적인 FL 방법론이나, 데이터가 non-IID일 때 수렴 속도와 성능이 크게 저하되는 문제가 있다.
- **SCAFFOLD:** SCV를 사용하여 클라이언트 드리프트를 수정함으로써 FedAvg보다 빠르게 전역 최적점에 수렴하게 한다. 하지만 이는 여전히 모든 클라이언트를 위한 '하나의 전역 모델'을 만드는 것에 집중한다.
- **Weight Erosion (WE):** 그래디언트 간의 거리를 이용해 가중 평균을 구함으로써 개인화 모델을 생성한다. 그러나 SCV와 같은 드리프트 보정 메커니즘이 없어 수렴 속도 면에서 한계가 있다.
- **APFL (Adaptive Personalized FL):** 전역 모델과 로컬 보정 모델 간의 보간(interpolation)을 통해 개인화를 달성한다.

WAFFLE은 SCAFFOLD의 빠른 수렴 성능과 WE의 개인화 전략을 동시에 채택함으로써, 기존 방법론들이 개별적으로 가졌던 수렴 속도 문제와 개인화 정밀도 문제를 동시에 해결하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 파이프라인

WAFFLE은 중앙 서버가 조율하는 중앙 집중형 연합 학습 구조를 따른다. 서버는 전역 모델 $x$와 제어 변수(control variate) $c$를 클라이언트들에게 전송하고, 각 클라이언트는 로컬 데이터를 통해 모델을 업데이트한 후 업데이트 값 $\Delta y_i$와 로컬 제어 변수 $\Delta c_i$를 서버로 반환한다. 서버는 이를 가중 평균하여 새로운 모델과 제어 변수를 생성한다.

### 2. 가중치 기반 업데이트 집계

WAFFLE의 핵심은 SCAFFOLD의 단순 평균 방식에서 벗어나, 타겟 클라이언트(Alice)를 기준으로 가중치 $\alpha_i$를 적용하는 것이다. 업데이트 식은 다음과 같다.

$$(\Delta x, \Delta c) \leftarrow \sum_{i \in \{1, \dots, N\}} \alpha_i (\Delta y_i, \Delta c_i)$$

여기서 모든 $\alpha_i = 1/N$이면 SCAFFOLD와 동일한 전역 학습이 되며, 타겟 클라이언트의 가중치만 1이고 나머지가 0이면 완전한 로컬 학습이 된다.

### 3. 가중치 계산 알고리즘 ($\text{CalcWeight}$)

가중치 $\alpha_i^r$은 라운드 $r$에서의 그래디언트 거리와 개인화 정도를 결정하는 하이퍼파라미터 $\Omega(r), \Psi(r)$에 의해 결정된다.

- **거리 측정:** 클라이언트 $i$와 타겟 클라이언트 $i?$ 사이의 유클리드 거리 $d_i = \|\Delta y_i - \Delta y_{i?}\|^2$를 계산한다.
- **타겟 클라이언트의 가상 거리 ($d_{i?}$):** 타겟 클라이언트 자신의 거리는 항상 0이므로, $\Omega(r)$를 이용하여 다음과 같이 가상 거리를 설정한다.
  $$d_{i?} \leftarrow d_m \cdot \left( 1 - \frac{d_M - d_m}{d_M}(1 - \Omega(r)) \right)$$
  여기서 $d_M$은 최대 거리, $d_m$은 최소 거리이다. $\Omega(r) \approx 1$이면 전역 학습에 가깝게, $0$에 가까우면 개인화 학습에 가깝게 설정된다.
- **최종 가중치 결정:**
  $$\alpha_i^r \leftarrow \max \left\{ \Psi(r) - \frac{d_i - d_{i?}}{d_M - d_{i?}}, 0 \right\}$$
  이후 가중치 합이 1이 되도록 정규화하며, 확률적 노이즈를 줄이기 위해 최근 3라운드의 가중치 평균을 사용한다.

### 4. 개인화 스케줄링

전역 학습에서 로컬 학습으로의 부드러운 전환을 위해 시그모이드 함수를 사용하여 $\Omega(r)$과 $\Psi(r)$을 정의한다.
$$\Omega(r) = \Psi(r) = \frac{1}{1 + e^{\Delta \Omega \cdot (r/(R/2)-1)}}$$
$\Delta \Omega$는 기울기를 조절하는 파라미터이며, 본 논문에서는 $\Delta \Omega = 3.2$를 기본값으로 사용하여 학습의 70~90% 지점에서 로컬 SGD로 전환되도록 설계하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** MNIST, CIFAR10.
- **데이터 분포 (Non-IID 설정):**
  - Label skew: 레이블 분포가 클라이언트마다 다름 (Distribution B, C).
  - Concept shift: 특징에서 레이블로의 매핑이 다름 (Distribution A*, B*).
- **비교 대상:** Local, FedAvg, SCAFFOLD, Weight Erosion (WE), APFL.
- **평가 지표:** 각 클라이언트의 로컬 테스트 정확도(Local test accuracy).

### 2. 주요 결과

- **MNIST 결과:**
  - Label skew 상황(B, C)에서 WAFFLE과 WE가 다른 방법론보다 우수한 성능을 보였다. 특히 WAFFLE은 SCV 덕분에 WE보다 수렴 속도가 훨씬 빨랐다.
  - Concept shift 상황(A*, B*)에서는 모든 개인화 방법론이 전역 방법론보다 성능이 좋았으며, WE가 약간 더 우세했으나 WAFFLE 역시 경쟁력 있는 성능을 보였다.
- **CIFAR10 결과:** MNIST와 유사한 경향을 보였다. IID 데이터에서는 전역 학습 방법론이 우세했지만, Label skew 및 Concept shift 상황에서는 WAFFLE과 WE가 Local이나 FedAvg보다 훨씬 높은 정확도를 달성하였다.
- **하이퍼파라미터 튜닝:** WE는 성능을 위해 정교한 튜닝이 필요했지만, WAFFLE은 단일 파라미터 $\Delta \Omega = 3.2$만으로도 모든 실험 케이스에서 경쟁력 있는 결과를 얻어 튜닝 편의성이 매우 높음을 입증하였다.

## 🧠 Insights & Discussion

### 강점

WAFFLE은 협력 학습의 전역적 지식과 개별 클라이언트의 로컬 특성을 조화롭게 통합하였다. 특히 SCAFFOLD의 제어 변수를 개인화 메커니즘에 접목함으로써, 기존의 개인화 FL 방법론들이 겪었던 느린 수렴 속도 문제를 효과적으로 해결하였다. 또한, 복잡한 튜닝 없이도 범용적으로 작동하는 하이퍼파라미터 설정을 제시한 점이 실용적이다.

### 한계 및 논의사항

- **이론적 분석 부족:** WAFFLE이 SCAFFOLD의 수렴 보장 특성을 그대로 유지하는지에 대한 이론적 증명이 제시되지 않았다.
- **실험 범위의 제한:** Concept shift와 Label skew 외에 Covariate shift 등 더 다양한 non-IID 시나리오에 대한 검증이 필요하다.
- **계산 비용:** CIFAR10 실험의 경우 계산량 문제로 인해 단일 시드(seed) 결과만 제시되었다는 점이 통계적 신뢰도 측면에서 아쉬운 부분이다.
- **신뢰성 가정:** 모든 클라이언트가 정직하게 참여한다는 가정을 전제로 한다. 다만, 가중치 기반 필터링 메커니즘이 잠재적으로 악의적인 참여자의 영향을 줄일 수 있을 것으로 기대된다.

## 📌 TL;DR

WAFFLE은 SCAFFOLD의 수렴 가속화 기법(SCV)과 Weight Erosion의 거리 기반 가중치 집계 방식을 결합하여, 비독립 동일 분포 데이터 환경에서 빠르게 수렴하는 개인화된 연합 학습 모델을 구축하는 알고리즘이다. 실험 결과, Label skew 및 Concept shift 환경에서 기존 개인화 방법론 대비 빠른 수렴 속도와 높은 정확도를 보였으며, 하이퍼파라미터 튜닝이 매우 간편하다는 장점이 있다. 이 연구는 향후 데이터 이질성이 심한 의료 데이터나 개인 맞춤형 서비스 등의 실제 FL 적용 분야에 중요한 기여를 할 가능성이 크다.
