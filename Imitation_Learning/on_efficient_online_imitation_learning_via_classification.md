# On Efficient Online Imitation Learning via Classification

Yichen Li, Chicheng Zhang (2022)

## 🧩 Problem to Solve

본 논문은 연속적인 의사결정 문제(sequential decision-making problems)를 해결하기 위한 일반적인 패러다임인 모방 학습(Imitation Learning, IL), 그 중에서도 전문가에게 상호작용하며 쿼리를 보낼 수 있는 Interactive Imitation Learning의 효율성을 다룬다. 특히, 분류 기반의 온라인 모방 학습(Classification-based Online Imitation Learning, 이하 COIL) 설정에서 오라클 효율적인(oracle-efficient) 후회 최소화(regret-minimization) 알고리즘을 설계하는 것의 근본적인 가능성과 한계를 연구한다.

기존의 DAGGER와 같은 프레임워크는 Interactive IL을 온라인 학습의 후회 최소화 문제로 환원하여 해결해 왔다. 그러나 일반적인 비실현 가능(non-realizable) 설정에서 다음과 같은 문제들이 존재한다:

1. **적절한 학습(Proper Learning)의 한계**: 벤치마크 정책 클래스 $B$ 내에서만 정책을 선택하는 적절한 학습 방식으로는 일반적인 경우에 서브리니어(sublinear) 후회를 보장할 수 없다.
2. **대리 손실 함수(Surrogate Loss)의 불일치**: 많은 연구가 계산의 편의를 위해 볼록 대리 손실 함수(convex surrogate loss)를 사용하지만, 이는 비실현 가능 설정에서 원래의 zero-one 분류 손실을 최소화하는 모델과 매우 다른 결과를 초래할 수 있다.
3. **계산 효율성 문제**: 정책 클래스 $B$의 크기가 클 경우, 매 라운드 모든 정책에 대해 손실을 계산하고 업데이트하는 것은 계산적으로 불가능하다.

따라서 본 논문의 목표는 비실현 가능 설정에서 통계적 및 계산적 효율성을 동시에 갖춘 COIL 알고리즘 프레임워크를 제안하고, 다이내믹 후회(dynamic regret) 최소화의 계산적 난이도를 규명하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **불가능성 증명**: COIL 문제에서 적절한 온라인 학습 알고리즘(proper online learning algorithm)은 일반적인 경우 서브리니어 후회를 보장할 수 없음을 이론적으로 증명하였다.
2. **LOGGER 프레임워크 제안**: '혼합 정책 클래스(Mixed Policy Class)'라는 개념을 도입하여 COIL 문제를 온라인 선형 최적화(Online Linear Optimization) 문제로 환원하는 부적절한 학습(improper learning) 프레임워크인 LOGGER를 제안하였다.
3. **효율적인 알고리즘 설계**: LOGGER 프레임워크 내에서 계산 효율적인 두 가지 알고리즘인 LOGGER-M과 LOGGER-ME를 설계하였다.
    - **LOGGER-M**: 정적 후회(static regret) $O(\sqrt{N})$을 달성하며, 샘플 효율성과 계산 효율성을 균형 있게 제공한다.
    - **LOGGER-ME**: COIL의 예측 가능성(predictability)을 이용하여 $O(1)$의 정적 후회를 달성하며, 상호작용 라운드 수를 획기적으로 줄였다.
4. **계산 복잡도 분석**: LOGGER 프레임워크 내에서 효율적인 다이내믹 후회 최소화가 불가능함을 보였으며, 이는 PPAD-complete 문제의 난이도와 연결됨을 증명하였다.

## 📎 Related Works

본 논문은 기존의 Interactive IL 연구들을 다음과 같이 분석하고 차별점을 제시한다.

- **DAGGER 및 그 변형들**: DAGGER는 Interactive IL을 온라인 학습으로 환원하는 기본 틀을 제공했다. 하지만 많은 후속 연구들이 손실 함수의 볼록성(convexity)을 가정하거나 대리 손실 함수를 사용했다. 본 논문은 이러한 가정 없이 원래의 분류 손실(zero-one loss)을 직접 다루며, 비실현 가능 설정에서의 한계를 정면으로 다룬다.
- **온라인 학습 및 오라클 효율성**: 기존의 오라클 효율적인 온라인 분류 알고리즘(예: CFTPL)은 i.i.d. 설정이나 트랜스덕티브(transductive) 설정에서는 잘 작동하지만, COIL처럼 현재 선택한 정책 $\pi_n$에 의해 다음 상태 분포 $d^{\pi_n}$이 결정되는 설정에서는 리니어(linear) 후회가 발생한다. 본 논문은 이를 해결하기 위해 혼합 정책 클래스를 통한 부적절한 학습 방식을 제안한다.
- **다이내믹 후회 연구**: 일부 연구들이 연속적인 제어 문제에서 다이내믹 후회를 다루었으나, 이는 주로 손실 함수의 강볼록성(strong convexity)에 의존했다. 본 논문은 COIL 설정에서 이러한 가정 없이 다이내믹 후회 최소화의 계산적 불가능성을 PPAD-hardness를 통해 증명함으로써 이론적 토대를 강화했다.

## 🛠️ Methodology

### 1. 혼합 정책 클래스 (Mixed Policy Class)

적절한 학습의 한계를 극복하기 위해, 본 논문은 벤치마크 정책 클래스 $B$의 볼록 껍질(convex hull)에 해당하는 혼합 정책 클래스 $\Pi_B$를 정의한다.
$$\Pi_B := \left\{ \pi_u(\cdot|s) := \sum_{h \in B} u[h] \cdot h(\cdot|s) : u \in \Delta(B) \right\}$$
여기서 $u \in \Delta(B)$는 각 정책 $h \in B$에 할당된 확률 가중치 벡터이다. 정책을 선택할 때 $B$의 한 원소를 고르는 것이 아니라, 가중치 $u$를 학습함으로써 문제의 볼록성을 확보한다.

### 2. LOGGER 프레임워크

LOGGER는 COIL 문제를 온라인 선형 최적화 문제로 환원한다. 라운드 $n$에서의 손실 함수 $F_n(\pi_u)$는 $u$에 대한 선형 함수로 표현될 수 있다.
$$F_n(\pi_u) = \sum_{h \in B} u[h] \cdot \mathbb{E}_{s \sim d^{\pi_n}} \mathbb{E}_{a \sim h(\cdot|s)} [\zeta_E(s, a)] = \langle \theta(u_n), u \rangle$$
여기서 $\theta(u_n)$은 정책 $\pi_{u_n}$에 의해 유도된 상태 분포에서의 기대 비용 벡터이다. 이 환원을 통해 모든 표준적인 온라인 선형 최적화 알고리즘(OLOA)을 COIL에 적용할 수 있게 된다.

### 3. LOGGER-M (Mixed CFTPL 기반)

계산 효율성을 위해 비용 민감 분류(Cost-Sensitive Classification, CSC) 오라클 $O$를 사용한다. LOGGER-M은 FTRL(Follow-the-Regularized-Leader)을 오라클 효율적으로 근사하는 MFTPL 알고리즘을 사용한다.

- **핵심 아이디어**: laPlace-like 노이즈를 추가한 섭동 집합(perturbation set) $Z_j$를 생성하고, 이를 기존 데이터와 합쳐 CSC 오라클을 $T$번 호출한 뒤 그 결과들의 평균을 취한다.
- **결과**: 정적 후회 $O(\sqrt{N})$을 달성하며, 오라클 호출 횟수를 $B$의 크기에 관계없이 소규모 분리 집합(separator set) $X$의 크기에 의존하게 함으로써 계산 효율성을 확보했다.

### 4. LOGGER-ME (Extra-Gradient 기반)

COIL의 특성인 '분포 연속성(distributional continuity)'을 이용하여 후회를 더욱 줄인 알고리즘이다.

- **핵심 아이디어**: Extra-Gradient 방법을 적용하여, 현재 라운드에서 예측된 정책 $\hat{\pi}_n$으로 한 번 더 상호작용하여 $\theta(u_n)$에 대한 더 정확한 추정치 $\hat{g}_n$을 얻는다.
- **결과**: 정적 후회를 $O(1)$ 수준으로 낮추었으며, 이는 상호작용 라운드 수 $I(\epsilon)$를 $\tilde{O}(1/\epsilon^2)$에서 $\tilde{O}(1/\epsilon)$으로 크게 줄이는 효과를 가져온다.

## 📊 Results

본 논문은 제안한 알고리즘들을 Behavior Cloning(BC) 및 기존 baseline과 비교 분석하였다.

### 1. 정량적 분석 (Table 1 및 Table 3 기준)

- **근사 오차(ApproxErr)**: BC는 $H^2 \cdot \text{Bias}$의 오차를 가지는 반면, LOGGER 계열은 $\mu H \cdot \text{Bias}$의 오차를 가진다. 여기서 $\mu$는 복구 가능성 상수(recoverability constant)로, 일반적으로 $\mu \ll H$이므로 LOGGER가 더 정밀하다.
- **상호작용 라운드 수 $I(\epsilon)$**:
  - LOGGER-M: $\tilde{O}(\mu^2 H^2 / \epsilon^2)$
  - LOGGER-ME: $\tilde{O}(\mu H^2 / \epsilon)$ $\rightarrow$ **가장 효율적**
  - Behavior Cloning: $1$ (단, 오차가 매우 큼)
- **전문가 어노테이션 수 $A(\epsilon)$**:
  - LOGGER-M 및 LOGGER-ME: $\tilde{O}(\mu^2 H^2 / \epsilon^2)$
  - Behavior Cloning: $\tilde{O}(H^4 / \epsilon^2)$ $\rightarrow$ LOGGER가 훨씬 적은 샘플로 동일한 추정 오차 $\epsilon$ 달성 가능.

### 2. 주요 결론

LOGGER-ME는 적절한 수준의 샘플 복잡도를 유지하면서도 상호작용 횟수를 획기적으로 줄여, 실제 전문가 어노테이션이 배치 단위로 이루어지거나 지연이 발생하는 현실적인 환경에서 매우 유리함을 보였다.

## 🧠 Insights & Discussion

### 1. 적절한 학습 vs 부적절한 학습

본 논문은 COIL에서 적절한 학습이 실패하는 이유를 정확히 짚어냈다. 일반적인 온라인 분류와 달리, COIL에서는 현재의 선택 $\pi_n$이 미래의 손실 함수 $F_n$을 결정한다. 이는 일종의 적대적(adversarial) 환경을 조성하며, 이를 해결하기 위해서는 정책 공간을 볼록화(convexification)하는 '혼합 정책'이라는 부적절한 학습 방식이 필수적임을 시사한다.

### 2. 다이내믹 후회의 계산적 한계

다이내믹 후회 최소화는 이론적으로 더 낮은 근사 오차를 제공할 가능성이 크지만, 본 논문은 이것이 PPAD-complete 문제(예: 2인 일반 합 게임의 내쉬 균형 찾기)만큼 어렵다는 것을 증명했다. 이는 단순히 알고리즘의 개선으로 해결될 문제가 아니라, 계산 복잡도 측면에서의 근본적인 장벽이 존재함을 의미한다.

### 3. 한계 및 향후 과제

- **분리 집합 가정**: 본 논문은 소규모 분리 집합(small separator set)이 존재한다는 가정을 사용했다. 이는 많은 정책 클래스에서 성립하지만, 더 일반적인 클래스로 확장하기 위해서는 smoothed online learning 등의 기법을 도입한 추가 연구가 필요하다.
- **실험적 검증**: 본 논문은 이론적 분석에 집중하고 있으며, 실제 환경에서의 엠피리컬(empirical)한 성능 평가는 향후 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 분류 기반 온라인 모방 학습(COIL)에서 **적절한 학습 방식으로는 서브리니어 후회를 달성할 수 없음**을 증명하고, 이를 해결하기 위해 **혼합 정책 클래스를 이용한 LOGGER 프레임워크**를 제안했다. 이를 통해 계산 효율적인 **LOGGER-M($O(\sqrt{N})$ 후회)**과 상호작용 효율적인 **LOGGER-ME($O(1)$ 후회)** 알고리즘을 도출하였으며, 기존 Behavior Cloning 대비 더 적은 샘플과 더 낮은 근사 오차를 보장한다. 또한, 다이내믹 후회 최소화가 **PPAD-hard** 함을 보여 이론적 한계를 명확히 규명하였다.
