# Fail-Safe Adversarial Generative Imitation Learning

Philipp Geiger, Christoph-Nikolas Straehle (2022)

## 🧩 Problem to Solve

본 논문은 로봇 제어나 자율주행 차량과 같이 안전성이 필수적인 도메인에서 Imitation Learning(IL)을 적용할 때 발생하는 안전성과 강건성(Robustness) 문제를 해결하고자 한다. 기존의 Generative Adversarial Imitation Learning(GAIL)과 같은 생성적 IL 방식은 유연한 정책 학습이 가능하지만, 학습 과정에서 발생하는 작은 오차가 누적되어(Compounding error) 테스트 단계에서 안전하지 않은 행동을 생성할 위험이 크다.

특히, 다중 에이전트 환경에서 타 에이전트의 불확실한 행동이 존재하는 상황에서, 하드 제약 조건(Hard constraints)을 만족하면서도 엔드-투-엔드(End-to-end) 학습이 가능한 생성적 정책을 설계하는 것은 매우 어렵다. 따라서 본 논문의 목표는 이론적으로 안전성이 보장되며, 동시에 미분 가능한 구조를 통해 전체 파이프라인을 효율적으로 학습시킬 수 있는 Fail-Safe Generative Imitation Learning 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **미분 가능한 Safety Layer**와 **샘플 기반의 안전 행동 집합(Safe Action Set) 추론**을 결합하여, 생성적 정책의 유연성과 제어 이론의 안전성 보장을 동시에 달성하는 것이다.

1.  **Piecewise Diffeomorphism 기반의 Safety Layer**: 잠재적으로 안전하지 않은 'Pre-safe' 행동을 안전한 행동 집합으로 매핑하는 미분 가능한 레이어를 제안한다. 특히, piecewise diffeomorphism 구조를 통해 전체 정책의 확률 밀도 함수(Probability Density)와 그래디언트를 닫힌 형태(Closed-form)로 계산할 수 있게 하여 엔드-투-엔드 학습을 가능하게 했다.
2.  **샘플 기반 안전 집합 추론**: 모든 행동을 전수 조사하는 대신, 유한한 샘플의 안전성을 확인한 뒤 Lipschitz 연속성(Lipschitz continuity) 또는 볼록성(Convexity)을 이용하여 해당 행동 주변의 영역까지 안전함을 보장하는 inner approximation $\tilde{A}_s^t$를 추론하는 방법을 제시한다.
3.  **학습 단계 Safety Layer 적용의 이론적 분석**: Safety Layer를 테스트 시에만 적용하는 경우보다 학습 단계부터 적용하는 것이 모방 오차(Imitation error)를 획기적으로 줄인다는 것을 이론적으로 증명하였다. 구체적으로, 테스트 시에만 적용할 경우 오차가 호라이즌 $T$에 대해 이차식($T^2$)으로 증가하는 반면, 학습 시부터 적용하면 선형($T$)으로 증가함을 보였다.

## 📎 Related Works

기존의 안전한 IL 연구는 크게 두 가지 방향으로 나뉜다. 첫째는 **보상 증강(Reward Augmentation)** 방식으로, 충돌과 같은 위험 상황에 페널티를 부여하는 방식(예: RAIL)이다. 그러나 이는 '소프트'한 제약 조건일 뿐이며, 실제 환경에서 하드한 안전 보장을 제공하지 못한다. 둘째는 **Safety Layer**를 사용하는 방식인데, 기존 연구들은 주로 결정론적(Deterministic) 정책에 국한되었거나, 테스트 시에만 레이어를 추가하여 학습된 정책과 실제 실행 정책 간의 괴리가 발생하는 문제가 있었다.

본 논문은 이러한 한계를 극복하기 위해 생성적(Generative) 정책에서도 적용 가능한 미분 가능한 Safety Layer를 제안하며, 이를 통해 확률 밀도 기반의 생성적 학습(GAIL 등)과 하드한 안전 제약을 통합하였다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
FAGIL의 전체 파이프라인은 다음과 같은 세 가지 모듈로 구성된다.
1.  **Pre-safe Generative Policy**: Gaussian 정책이나 Normalizing Flow와 같이 닫힌 형태의 밀도 함수를 제공하는 표준 생성 모델이다. 여기서 생성된 행동 $\hat{a}_t$는 아직 안전성이 보장되지 않은 상태이다.
2.  **Safe Set Inference Module**: 현재 상태 $s_t$에서 실행 가능한 안전한 행동들의 집합 $\tilde{A}_s^t$를 추론한다.
3.  **Safety Layer**: Pre-safe 행동 $\hat{a}_t$를 안전 집합 $\tilde{A}_s^t$ 내부의 행동 $\bar{a}_t$로 매핑한다.

### 상세 방법론 및 방정식

#### 1. 안전 행동 집합의 추론 (Safe Set Inference)
어떤 행동 $a$가 안전하려면, 그 이후의 최악의 상황(Worst-case others $\sigma$)에서도 적절한 대응 정책 $\pi$가 존재하여 안전 비용 $d(s) \leq 0$을 유지해야 한다. 이를 위해 총 안전 비용 $w_t(s, a)$를 다음과 같이 정의한다.
$$w_t(s, a) := \min_{\pi_{t+1:T}} \max_{\sigma_{t:T}} \max_{t' \in t+1:T} d(s_{t'})$$
안전 집합 $\bar{A}_s^t$는 $w_t(s, a) \leq 0$인 행동들의 집합이다. 본 논문은 이를 효율적으로 계산하기 위해 두 가지 접근법을 사용한다.
- **FAGIL-L (Lipschitz-based)**: $w_t$의 Lipschitz 상수를 이용하여, 샘플링된 지점의 안전성이 확인되면 그 주변 반경 내의 모든 행동이 안전함을 보장한다.
- **FAGIL-E (Extremality-based)**: 시스템 역학이 선형이고 안전 비용이 볼록(Convex)할 때, 다면체 영역의 꼭짓점(Corners)들만 확인하여 영역 전체의 안전성을 판별한다.

#### 2. Piecewise Diffeomorphism Safety Layer
Safety Layer $g: A \to \tilde{A}$는 $A$를 여러 구역 $A_k$로 나누고, 각 구역에서 미분 가능한 전단사 함수(Diffeomorphism) $g^k$를 적용하는 방식이다. 이를 통해 전체 정책의 확률 밀도 함수 $p_{\bar{a}}(\bar{a})$를 다음과 같이 계산할 수 있다.
$$p_{\bar{a}}(\bar{a}) = \sum_{k: \bar{a} \in g^k(A^k)} | \det(J_{g_k^{-1}}(\bar{a})) | p_{\hat{a}}(g_k^{-1}(\bar{a}))$$
여기서 $J_{g_k^{-1}}$는 $g^k$의 역함수에 대한 자코비안(Jacobian) 행렬이다. 이 수식을 통해 Safety Layer가 포함된 전체 모델의 그래디언트를 계산할 수 있어 엔드-투-엔드 학습이 가능해진다.

#### 3. 학습 절차
FAGIL은 GAIL 프레임워크를 사용하여 학습한다. 
- **Discriminator**: 모방자(Imitator)의 궤적과 전문가(Demonstrator)의 궤적을 구분하여 모방 비용 $c(s, a)$를 생성한다.
- **Generator (FAGIL Policy)**: Safety Layer를 포함한 정책 $\pi_{I, \theta}$를 통해 생성된 궤적이 Discriminator를 속이도록(즉, 전문가와 유사하도록) 학습한다. 이때 앞서 언급한 닫힌 형태의 밀도 함수와 그래디언트를 사용하여 정책 파라미터 $\theta$를 업데이트한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 실제 고속도로 차량 주행 데이터인 `highD` 데이터셋을 사용하였다.
- **작업**: 자율주행 차량의 종방향 및 횡방향 가속도를 제어하는 운전자 모방 학습.
- **기준선(Baselines)**: 표준 GAIL, 보상 증강 방식의 RAIL, 그리고 테스트 시에만 Safety Layer를 적용한 TTOS(Test-Time Only Safety)와 비교하였다.
- **지표**: 모방 성능은 ADE(Average Displacement Error)와 FDE(Final Displacement Error)로 측정하고, 안전성은 충돌 발생 확률(Probability of crash)로 측정하였다.

### 주요 결과
실험 결과, FAGIL(L 및 E 버전 모두)은 **충돌 확률 0%**를 달성하여 이론적인 안전 보장을 실증적으로 증명하였다. 

| Method | ADE (Gauss) | FDE (Gauss) | Crash Prob. |
| :--- | :---: | :---: | :---: |
| GAIL | 0.47 | 1.32 | 0.13 |
| RAIL | 0.48 | 1.35 | 0.22 |
| **FAGIL-L (Ours)** | **0.60** | **1.77** | **0.00** |
| TTOS | 0.60 | 1.78 | 0.00 |

모방 성능(ADE, FDE) 측면에서는 제약 조건이 없는 GAIL보다 약간 낮게 나타났으나, 이는 안전성을 보장하기 위한 필연적인 trade-off로 해석된다. 주목할 점은 FAGIL이 TTOS보다 모방 성능이 소폭 우수하다는 것이며, 이는 학습 단계부터 Safety Layer를 고려하여 정책을 최적화했기 때문에 발생하는 이점이다.

## 🧠 Insights & Discussion

본 논문은 생성적 IL 모델에 하드한 안전 제약을 통합하는 체계적인 방법을 제시하였다. 특히, Safety Layer를 학습 과정에 포함시켰을 때의 이점을 이론적으로 규명한 점이 매우 인상적이다. 그림 2에서 설명하듯, 테스트 시에만 안전 레이어를 적용하면 모델이 학습 과정에서 경험하지 못한 상태(Safe side strip)로 진입했을 때 그곳에서 회복(Recover)하는 방법을 배우지 못해 성능이 급격히 저하되는 현상이 발생한다. 반면, 학습 시부터 레이어를 포함하면 모델이 안전 레이어의 특성을 인지하고 이를 고려하여 계획(Planning)하게 된다.

**한계 및 논의 사항:**
1.  **차원의 저주**: 제안된 그리드 기반의 안전 집합 추론 및 Safety Layer 구현은 행동 공간의 차원이 낮을 때만 효율적이다. 고차원 행동 공간으로 확장하기 위해서는 더 효율적인 파티셔닝 방법이나 근사 기법이 필요하다.
2.  **보수적 성향**: 최악의 상황(Worst-case)을 가정한 안전 보장은 실제 상황에서 지나치게 보수적인 주행을 유발할 수 있다. 이를 해결하기 위해 RSS(Responsibility-Sensitive Safety)와 같이 현실적인 제약을 도입하는 방향으로 확장 가능하다.
3.  **학습 불안정성**: GAIL 기반의 적대적 학습은 본래 불안정한 특성이 있으며, Safety Layer라는 복잡한 구조가 추가됨에 따라 학습 수렴 속도나 안정성에 영향을 줄 수 있다.

## 📌 TL;DR

본 논문은 생성적 모방 학습(Generative IL)에서 안전성을 보장하기 위해 **미분 가능한 Piecewise Diffeomorphism Safety Layer**와 **샘플 기반 안전 집합 추론 모듈**을 제안하였다. 이를 통해 안전성을 이론적으로 보장하면서도 엔드-투-엔드 학습이 가능한 FAGIL 프레임워크를 구축하였으며, 실제 도로 데이터 실험을 통해 충돌률 0%와 준수한 모방 성능을 입증하였다. 이 연구는 안전이 최우선인 자율주행 및 로보틱스 분야에서 딥러닝 기반의 유연한 정책 학습과 제어 이론의 엄격한 안전 보장을 결합하는 중요한 가교 역할을 할 것으로 기대된다.