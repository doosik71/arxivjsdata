# A proof of imitation of Wasserstein inverse reinforcement learning for multi-objective optimization

Akira Kitaoka, Riki Eto (2023)

## 🧩 Problem to Solve

본 논문은 다목적 최적화(Multi-Objective Optimization, MOO) 환경에서 전문가의 의도(Intention)를 반영하는 보상 함수(Reward Function)를 자동으로 학습하는 문제를 다룬다. 

일반적으로 AI를 통한 자동화는 보상 함수를 설정하고 이를 최대화 또는 최소화하는 최적 해를 찾는 방식으로 이루어진다. 특히 근무 교대 스케줄링과 같은 조합 최적화나 다목적 최적화 문제에서는 "휴가 요청 반영도", "업무 부하 균등화" 등 여러 관점의 보상 함수를 설정해야 한다. 그러나 이러한 보상 함수를 수동으로 설계하는 것은 많은 시행착오를 필요로 하며, 이는 실제 수학적 최적화 적용에 큰 진입 장벽이 된다.

기존의 역강화학습(Inverse Reinforcement Learning, IRL) 방식들은 다목적 최적화에 적용할 때 다음과 같은 한계가 존재한다:
1. **계산 복잡도:** 행동 공간(Action Space)이 매우 넓어 보상 함수를 모든 상태와 행동에 대해 설정하는 비용이 매우 높다.
2. **계산 효율성 부족:** Maximum Entropy IRL(MEIRL)은 모든 궤적에 대한 보상 함수의 합을 계산해야 하므로 비용이 많이 든다.
3. **이산 값 처리의 어려움:** Guided Cost Learning(GCL)은 중요도 샘플링(Importance Sampling)을 사용하지만, 다목적 최적화의 이산적 특성으로 인해 보상 함수의 작은 변화가 결과의 큰 변화를 초래하여 확률 분포를 찾기 어렵다.

따라서 본 논문의 목표는 Wasserstein Inverse Reinforcement Learning(WIRL)이 다목적 최적화에서 수렴할 때, 학습자의 보상 값과 최적 해(Action)가 실제로 전문가의 것을 모방(Imitation)한다는 것을 수학적으로 증명하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 WIRL이 다목적 최적화 문제에서 전문가의 의도를 이론적으로 완벽하게 모방할 수 있음을 증명한 것이다.

중심적인 설계 아이디어는 Wasserstein GAN에서 영감을 받아 전문가의 궤적 분포 $P_E$와 학습자의 궤적 분포 $P_\phi$ 사이의 Wasserstein 거리(Wasserstein distance)를 최소화하는 것이다. 특히, 다목적 최적화의 특성을 반영하여 보상 함수를 선형 결합 형태로 정의하고, 사전식 순서(Lexicographic order)를 도입하여 학습자의 행동이 전문가의 행동과 일치함을 보장하는 이론적 토대를 마련하였다.

## 📎 Related Works

논문에서는 기존의 IRL 접근 방식과 WIRL의 차이점을 다음과 같이 설명한다.

1. **MEIRL 및 GAIL과의 차이:**
   - **보상 함수 설계:** MEIRL은 엔트로피 정규화된 가치 함수를 사용하지만, WIRL은 다목적 최적화의 목적 함수를 보상 함수로 사용한다.
   - **공간 정의:** MEIRL/GAIL은 상태 및 행동 공간이 유한 집합이라고 가정하는 경우가 많으나, WIRL은 유한 및 무한 집합 모두에 적용 가능하다. 이로 인해 기존의 점유 측정치(Occupancy measures) 기반 논리를 다목적 최적화에 그대로 적용할 수 없으므로 WIRL의 접근 방식이 필요하다.

2. **기존 WIRL 연구와의 차별점:**
   - 이전 연구([KE23])에서는 WIRL의 수렴성(Convergence)을 증명하였으나, 수렴했을 때 학습자가 실제로 전문가의 보상 함수와 행동을 모방하는지에 대한 이론적 증명은 부족했다. 본 논문은 바로 이 '모방'의 증명에 집중한다.

## 🛠️ Methodology

### 1. WIRL의 기본 구조
WIRL은 전문가 궤적 분포 $P_E$와 학습자 궤적 분포 $P_\phi$ 사이의 Wasserstein 거리를 최소화하는 파라미터 $\phi$를 찾는 것을 목표로 한다. Kantorovich-Rubinstein duality에 의해 Wasserstein 거리는 다음과 같이 정의된다:

$$W(P_E, P_\phi) = \sup_{\|r_\theta\|_L \le 1} \left\{ \frac{1}{N} \sum_{n=1}^N r_\theta(\tau_E^{(n)}) - \frac{1}{N} \sum_{n=1}^N r_\theta(g_\phi(s_{ini}^{(n)})) \right\}$$

여기서 $r_\theta$는 1-Lipschitz 함수이며, $g_\phi$는 학습자의 궤적 생성기(Generator)이다.

### 2. 다목적 최적화(MOO)로의 확장
다목적 최적화 문제에서 학습자의 행동 $a(\phi, s)$는 다음과 같이 정의된다:

$$a(\phi, s) \in \arg \max_{h(x) \in h(X(s))} \phi^\top h(x)$$

여기서 $\phi$는 각 목적 함수의 가중치를 나타내는 파라미터이며, $\phi^\top h(x)$가 보상 값이 된다. 

### 3. Inverse Multi-Objective Optimization Problem (IMOOP)
WIRL을 MOO에 적용하면, 다음과 같은 목적 함수 $F(\phi)$를 최소화하는 문제로 귀결된다:

$$\min_{\phi \in \Phi} F(\phi) := \frac{1}{N} \sum_{n=1}^N \phi^\top a(\phi, s^{(n)}) - \frac{1}{N} \sum_{n=1}^N \phi^\top a^{(n)}$$

이 함수 $F(\phi)$의 하위 그라디언트(Subgradient)는 다음과 같이 단순하게 계산된다:
$$\frac{1}{N} \sum_{n=1}^N (a(\phi, s^{(n)}) - a^{(n)})$$

### 4. 학습 절차 (Algorithm 1)
학습은 다음과 같은 반복적인 업데이트 과정을 통해 이루어진다:
1. 파라미터 $\phi_k$를 초기화한다.
2. 다음 식을 통해 $\phi$를 업데이트하고, 제약 조건 집합 $\Phi$로 투영(Projection)한다:
   $$\phi_{k+1} \leftarrow \phi_k - \alpha_k \frac{1}{N} \sum_{n=1}^N (a(\phi_k, s^{(n)}) - a^{(n)})$$
   여기서 $\alpha_k$는 0으로 수렴하지만 합은 무한대인 학습률(Nonsummable diminishing learning rate)을 사용한다.

## 📊 Results

본 논문은 수치적 실험 결과보다는 수학적 증명을 통한 이론적 결과를 제시한다.

### 1. 보상 값의 모방 증명 (Reward Value Imitation)
**Theorem 4.1**과 **Corollary 4.3**에 따르면, WIRL이 수렴하여 $F(\phi) < \epsilon$이 되면, 모든 $n$에 대해 학습자의 보상 값과 전문가의 보상 값의 차이가 $\epsilon N$보다 작음을 보였다. 즉, WIRL이 수렴하면 학습자의 보상 값이 전문가의 보상 값을 모방하게 된다.

### 2. 행동의 모방 증명 (Action Imitation)
행동의 유일성을 보장하기 위해, 본 논문은 **사전식 순서(Lexicographic order, $\le_{dic}$)**를 도입하여 학습자의 행동을 다음과 같이 재정의한다:
$$a(\phi, s) := \min_{dic} \arg \max_{h(x) \in h(X(s))} \phi^\top h(x)$$

이 설정을 바탕으로 **Theorem 5.2**에서는 다음 세 조건이 동치(Equivalent)임을 증명하였다:
1. $F(\phi)$의 하위 그라디언트가 $0$이다.
2. 모든 $n$에 대해 학습자의 행동과 전문가의 행동이 일치한다 ($a(\phi, s^{(n)}) = a(\phi^{(0)}, s^{(n)})$).
3. 학습자와 전문가의 궤적 분포 사이의 Wasserstein 거리가 $0$이다 ($W(P_\phi, P_{\phi^{(0)}}) = 0$).

결과적으로, WIRL이 수렴하여 하위 그라디언트가 0이 되면 학습자의 행동은 전문가의 행동을 완벽하게 모방하게 된다.

## 🧠 Insights & Discussion

### 강점 및 의의
본 논문은 WIRL이 단순한 수렴을 넘어, 실제로 전문가의 '의도(보상 값)'와 '결과(행동)'를 모두 정확하게 복원할 수 있음을 수학적으로 입증하였다. 이는 사용자가 일일이 보상 함수를 설계할 필요 없이, 전문가의 데이터만으로 다목적 최적화 시스템을 구축할 수 있는 이론적 근거를 제공한다.

### 한계 및 논의사항
1. **무한 반복의 가정:** 증명은 반복 횟수가 무한히 많을 때의 수렴성을 전제로 한다. 실제 적용에서는 유한한 반복 횟수 내에서 어느 정도의 오차 범위 내로 수렴하는지에 대한 분석이 추가로 필요하다.
2. **전문가 모델의 표현 가능성:** 본 논문은 전문가의 행동이 학습자의 생성기 $g_\phi$로 표현 가능하다는 가정을 전제로 한다. 하지만 실제 환경에서는 전문가의 의사결정 과정이 수학적 모델로 완전히 정의되지 않을 수 있으며, 이 경우의 모방 가능성은 여전히 미해결 과제로 남아 있다.
3. **사전식 순서의 실용성:** 행동의 일치성을 증명하기 위해 사용한 $\min_{dic}$ (사전식 순서의 최소값 선택)이 실제 도메인에서 전문가의 선택 기준과 일치하는지에 대한 추가적인 고찰이 필요하다.

## 📌 TL;DR

본 논문은 다목적 최적화 문제에서 Wasserstein Inverse Reinforcement Learning(WIRL)이 수렴할 경우, 학습자의 보상 값과 최적 행동이 전문가의 것을 이론적으로 완벽하게 모방한다는 것을 증명하였다. 특히 사전식 순서를 도입하여 행동의 일치성을 엄밀하게 입증함으로써, 수동 보상 함수 설계 없이 전문가의 데이터를 통해 최적화 파라미터를 학습할 수 있는 이론적 토대를 마련하였다. 이 연구는 향후 복잡한 조합 최적화나 다목적 스케줄링 문제에서 전문가의 의도를 자동으로 추출하는 시스템 구현에 중요한 역할을 할 것으로 기대된다.