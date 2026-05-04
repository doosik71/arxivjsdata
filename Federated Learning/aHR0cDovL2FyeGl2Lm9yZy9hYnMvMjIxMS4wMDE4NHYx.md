# FL Games: A Federated Learning Framework for Distribution Shifts

Sharut Gupta, Kartik Ahuja, Mohammad Havaei, Niladri Chatterjee, Yoshua Bengio (2022)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 환경에서 발생하는 **분포 변화(Distribution Shift)** 문제, 구체적으로는 각 클라이언트가 서로 다른 데이터 분포를 가진 Non-i.i.d. 상황에서의 일반화 성능 저하 문제를 해결하고자 한다.

일반적인 연합 학습 알고리즘(예: $\text{FedAvg}$)은 각 클라이언트의 데이터에서 나타나는 **가짜 상관관계(Spurious Correlation)**를 학습하는 경향이 있다. 이로 인해 훈련 과정에서는 높은 정확도를 보일 수 있으나, 훈련 데이터의 볼록 껍질(convex hull) 밖에 위치한 새로운 도메인, 즉 **외삽 도메인(Extrapolated Domain)** 또는 OOD(Out-of-Distribution) 클라이언트의 데이터에 대해서는 일반화 성능이 급격히 떨어지는 치명적인 문제가 발생한다.

따라서 본 연구의 목표는 클라이언트 간의 분포 변화에 강건하며, 모든 도메인에서 공통적으로 유효한 **인과적 특징(Causal Features)**만을 학습하여 OOD 일반화 성능을 극대화하는 연합 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **불변 위험 최소화(Invariant Risk Minimization, IRM)**의 원리를 게임 이론적 관점에서 연합 학습에 접목한 $\text{FL GAMES}$ 프레임워크를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1. **불변 표현 학습**: 모든 클라이언트에서 최적의 예측 성능을 유지하는 불변 예측기(Invariant Predictor)를 학습함으로써, 환경에 따라 변하는 가짜 상관관계가 아닌 인과적 관계를 포착한다.
2. **병렬 업데이트(Parallel Updates)**: 기존 $\text{IRM GAMES}$의 순차적 업데이트(Clockwise sequence) 방식이 가진 시간 복잡도 문제를 해결하기 위해, 모든 클라이언트가 동시에 업데이트를 수행하는 병렬화 구조를 도입하여 확장성을 높였다.
3. **메모리 앙상블(Memory Ensemble)**: $\text{Best Response Dynamics (BRD)}$ 학습 시 발생하는 고주파 진동(High-frequency oscillation) 문제를 해결하기 위해, 과거의 전략들을 저장하는 버퍼를 활용한 앙상블 기법($\text{SMOOTH}$ 변형)을 제안하여 학습 곡선을 안정화했다.
4. **통신 효율성 개선**: 표현 학습기 $\phi$의 업데이트 시 $\text{SGD}$ 대신 전체 배치 경사 하강법($\text{Full-batch GD}$)을 사용하여 통신 라운드 수를 획기적으로 줄인 $\text{FAST}$ 변형을 제안했다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계를 지적하며 차별점을 제시한다.

- **기존 FL 방법론**: $\text{FedAvg}$, $\text{FedBN}$, $\text{FedProx}$ 등은 제약 조건 최적화나 지식 증류(Knowledge Distillation)를 통해 클라이언트 간 이질성을 다루지만, 근본적인 분포 변화(Distribution Shift) 문제를 해결하지 못한다. 이들은 보간 도메인(Interpolated domains)에서는 어느 정도 작동하지만, 외삽 도메인(Extrapolated domains)으로의 일반화에는 실패한다.
- **인과적 ML 및 IRM**: $\text{IRM}$은 환경 간 불변성을 학습하여 OOD 일반화를 꾀하지만, $\text{IRM}$의 손실 함수는 최적화가 매우 어렵고 수렴 보장이 부족하다.
- **$\text{IRM GAMES}$**: 게임 이론을 도입하여 $\text{IRM}$ 목적 함수를 해결하려 했으나, 순차적 업데이트로 인해 클라이언트 수가 증가하면 계산 비용이 선형적으로 증가하며, 학습 과정에서 성능 지표가 심하게 진동하는 문제가 있다.

## 🛠️ Methodology

### 전체 구조 및 원리

$\text{FL GAMES}$는 각 클라이언트를 하나의 플레이어로 간주하고, 모든 플레이어가 합의하는 **내쉬 균형(Nash Equilibrium)**을 찾는 과정을 통해 불변 표현을 학습한다.

#### 1. 불변성 원리 (Invariance Principle)

데이터 표현 $\phi: X \to Z$가 불변하다는 것은, 모든 환경 $e \in E_{tr}$에 대해 동시에 최적인 분류기 $w$가 존재함을 의미한다. 즉, $\forall e \in E_{tr}, w \in \arg \min_{w'} R_e(w' \circ \phi)$를 만족해야 한다.

#### 2. 게임 이론적 정식화

각 클라이언트 $k$는 자신의 로컬 예측기 $w_k$를 최적화하는 플레이어이다. 전체 시스템은 앙상블 모델 $w_{av} = \frac{1}{|E_{tr}|} \sum w_k$가 다음의 목적 함수를 최소화하도록 한다.

$$\min_{w_{av}, \phi} \sum_k R_k(w_{av} \circ \phi) \quad \text{s.t.} \quad w_k \in \arg \min_{w'_k} R_k \left( \frac{1}{|E_{tr}|} \left( w'_k + \sum_{q \neq k} w_q \right) \circ \phi \right)$$

여기서 각 클라이언트는 다른 클라이언트들의 전략($w_q$)이 주어졌을 때, 자신의 손실 $R_k$를 최소화하는 **최적 대응(Best Response)**을 찾는다.

### 주요 구성 요소 및 변형

- **$\text{V-FL GAMES}$ vs $\text{F-FL GAMES}$**:
  - $\text{V-FL GAMES}$: 표현 학습기 $\phi$를 학습 가능한 파라미터로 둔다. 복잡한 데이터셋에서 유리하다.
  - $\text{F-FL GAMES}$: $\phi$를 항등 함수($\phi = I$)로 고정하여 예측기 $w$만 학습한다.
- **$\text{Parallelized BRD}$**: 기존의 순차적 턴제 업데이트 대신, 모든 클라이언트가 $t-1$ 시점의 상대방 전략을 바탕으로 $t$ 시점에 동시에 업데이트를 수행한다.
- **$\text{SMOOTH}$ (메모리 앙상블)**: 진동을 줄이기 위해 각 클라이언트는 상대방의 현재 모델뿐만 아니라 과거 모델들의 앙상블(버퍼 $\mathcal{B}_p$)에 대응한다. 수정된 로컬 목적 함수는 다음과 같다.

$$w_k \in \arg \min_{w'_k} R_k \left( \frac{1}{|S|} \left( w'_k + \sum_{q \neq k} w_q + \sum_{p \neq k} \frac{1}{|\mathcal{B}_p|} \sum_{j=1}^{|\mathcal{B}_p|} w_{p,j} \right) \circ \phi \right)$$

- **$\text{FAST}$ (Full-batch GD)**: $\phi$를 업데이트할 때 $\text{SGD}$ 대신 로컬 전체 데이터를 사용한 $\text{GD}$를 적용하여 수렴 속도를 높이고 통신 횟수를 줄인다.

### 학습 절차

1. **표현 업데이트 ($\phi$)**: (짝수 라운드) 각 클라이언트가 $\phi$에 대한 그래디언트를 계산하여 서버로 전송하면, 서버가 이를 합산하여 $\phi$를 업데이트한다 ($\text{FedSGD}$ 방식).
2. **예측기 업데이트 ($w_k$)**: (홀수 라운드) 각 클라이언트가 현재의 $\phi$와 다른 클라이언트들의 모델(및 메모리 버퍼)을 바탕으로 자신의 $w_k$를 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: $\text{Colored MNIST}$, $\text{Colored FashionMNIST}$, $\text{Spurious CIFAR10}$, $\text{Colored DSprites}$. (모두 훈련 세트에는 강한 가짜 상관관계를 부여하고, 테스트 세트에서는 이를 제거하거나 반전시켜 OOD 성능을 측정함)
- **비교 대상**: $\text{FedSGD}$, $\text{FedAvg}$, $\text{FedBN}$, $\text{FedProx}$ 및 $\text{IRM GAMES}$ 변형들.
- **지표**: 훈련 및 테스트 정확도, 수렴까지의 통신 라운드 수.

### 정량적 결과

- **OOD 일반화**: $\text{FedAvg}$ 등의 기존 FL 베이스라인은 훈련 정확도는 높지만 테스트 정확도가 매우 낮다(가짜 상관관계에 의존). 반면 $\text{FL GAMES}$는 모든 벤치마크에서 훨씬 높은 테스트 정확도를 기록했다.
- **$\text{V-FL GAMES}$의 효용성**: $\text{Colored DSprites}$와 같은 복잡한 데이터셋에서는 $\phi$를 학습하는 $\text{V-FL GAMES}$가 $\text{F-FL GAMES}$보다 우수한 성능을 보였다.
- **효율성**: $\text{Parallelized}$ 버전은 클라이언트 수가 증가해도 통신 라운드 수가 완만하게 증가하며, $\text{SMOOTH+FAST}$ 조합은 수렴 속도를 최대 5.4배까지 향상시켰다.

### 정성적 결과 (LIME 분석)

$\text{LIME}$을 이용한 특징 시각화 결과, $\text{FedAvg}$는 이미지의 배경(Background)을 보고 예측하는 반면, $\text{FL GAMES}$는 객체의 모양, 선, 곡선 등 **인과적 특징(Causal features)**을 기반으로 예측함이 확인되었다.

## 🧠 Insights & Discussion

### 강점

- **인과적 표현 학습**: 단순히 데이터 분포를 맞추는 것이 아니라, 환경에 무관한 불변 특징을 학습함으로써 진정한 의미의 OOD 일반화를 달성했다.
- **실용적 최적화**: 게임 이론적 접근의 고질적 문제인 진동(Oscillation)과 느린 수렴 속도를 메모리 버퍼와 $\text{Full-batch GD}$라는 실용적인 방법으로 해결했다.

### 한계 및 논의사항

- **이론적 분석**: $\text{SMOOTH}$ 기법(메모리 앙상블)이 실제로 내쉬 균형으로의 수렴을 보장하는지에 대한 이론적 분석은 아직 미비하며, 이는 향후 연구 과제로 남겨두었다.
- **계산 비용**: $\text{V-FL GAMES(FAST)}$의 경우 로컬에서 전체 배치 $\text{GD}$를 수행하므로, 클라이언트의 로컬 계산 부담이 증가한다. 다만 논문에서는 통신 비용이 계산 비용보다 훨씬 크기 때문에 이는 합리적인 트레이드-오프라고 주장한다.

## 📌 TL;DR

본 논문은 연합 학습에서 클라이언트 간 데이터 분포 차이로 인해 발생하는 가짜 상관관계 학습 문제를 해결하기 위해, 게임 이론 기반의 **$\text{FL GAMES}$** 프레임워크를 제안한다. 이 방법은 모든 클라이언트에게 공통적으로 적용되는 **불변 표현(Invariant Representation)**을 학습하여 새로운 환경(OOD)에서도 높은 일반화 성능을 보인다. 특히 병렬 업데이트, 메모리 앙상블, $\text{Full-batch GD}$를 도입하여 기존 $\text{IRM GAMES}$의 확장성 및 안정성 문제를 해결했으며, 이는 의료 영상과 같이 도메인 간 편차가 큰 실제 연합 학습 환경에 적용될 가능성이 매우 높다.
