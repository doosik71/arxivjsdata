# Going Beyond Expert Performance via Deep Implicit Imitation Reinforcement Learning

Iason Chrysomallis, Georgios Chalkiadakis (2025)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)의 두 가지 핵심적인 제약 사항을 해결하고자 한다.

첫째는 **데이터 접근성의 문제**이다. 전통적인 모방 학습은 전문가의 상태-행동 쌍(state-action pairs)으로 구성된 완전한 시연 데이터(demonstrations)를 요구한다. 그러나 실제 환경에서는 전문가의 행동(action) 정보 없이 상태 변화(state observations)만 기록된 데이터만 존재하는 경우가 많다.

둘째는 **전문가의 최적성(Optimality) 문제**이다. 기존의 많은 IL 알고리즘은 전문가가 최적이거나 거의 최적이라는 가정을 전제로 한다. 하지만 실제 전문가 데이터는 하위 최적(suboptimal)인 경우가 많으며, 단순히 이를 모방하기만 해서는 전문가의 성능 한계라는 '성능 천장(performance ceiling)'에 갇히게 되어, 에이전트가 전문가보다 더 나은 정책을 학습할 수 없다.

마지막으로, **행동 공간의 이질성(Heterogeneous Action Spaces)** 문제이다. 전문가와 학습 에이전트가 서로 다른 행동 집합이나 능력을 갖춘 경우(예: 숙련된 인간 외과 의사와 수술 로봇), 전문가의 행동을 그대로 복제하는 것이 물리적으로 불가능하여 기존의 IL 방식으로는 지식 전이가 어렵다.

결과적으로 본 논문의 목표는 행동 정보가 없는 하위 최적의 관찰 데이터만을 이용하여 학습하면서도, 환경과의 상호작용을 통해 전문가의 성능을 뛰어넘을 수 있는 Deep Implicit Imitation RL 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **암묵적 모방 학습(Implicit Imitation Learning)을 온라인 심층 강화학습(Deep RL) 프레임워크 내에 통합**하는 것이다.

1. **DIIQN (Deep Implicit Imitation Q-Network):** 행동 정보가 없는 상태 관찰 데이터로부터 전문가의 행동을 추론(Action Inference)하고, 이를 DQN의 학습 과정에 결합하여 학습 속도를 가속화함과 동시에 전문가의 성능을 초과 달성하게 한다.
2. **동적 신뢰 메커니즘 (Dynamic Confidence Mechanism):** 전문가의 가이드와 에이전트의 자기 주도적 학습(self-directed learning) 사이의 균형을 실시간으로 조절한다. 전문가의 행동이 유용하다고 판단될 때는 모방에 집중하고, 에이전트의 성능이 전문가를 앞서기 시작하면 자기 주도적 학습의 비중을 높인다.
3. **HA-DIIQN (Heterogeneous Actions DIIQN):** 전문가와 에이전트의 행동 공간이 서로 다른 경우를 처리한다. 전문가의 특정 전이가 불가능하다고 판단되면, 에이전트가 실행 가능한 대안 경로인 '브릿지(Bridge)'를 탐색하여 전문가의 전략적 의도를 유지하면서도 학습할 수 있게 한다.

## 📎 Related Works

논문은 암묵적 모방 학습의 기초가 된 Price와 Boutilier(2003)의 연구를 언급하며, 이를 현대적인 심층 학습 설정으로 확장하였다. 기존의 관련 연구들은 다음과 같은 한계를 가진다.

- **Behavioral Cloning (BC):** 행동 정보가 필수적이며, 데이터의 노이즈나 하위 최적성에 매우 취약하다.
- **Inverse RL (IRL) 및 GAIL 계열:** 전문가의 보상 함수를 추정하여 학습하지만, 대부분 명시적 모방(Explicit Imitation)에 기반하여 행동 정보가 필요하다.
- **Learning from Observation (LfO):** 행동 없이 상태만으로 학습하는 암묵적 모방을 다루지만, 대개 전문가의 성능을 복제하는 것에 그치며 전문가의 성능을 초과하는 메커니즘이 부족하다.
- **D-REX:** 하위 최적 전문가를 처리하여 성능을 높일 수 있지만, 여전히 명시적 모방 범주에 속해 행동 데이터가 필요하다.

본 논문은 암묵적 모방의 데이터 효율성과 RL의 최적성 추구 능력을 결합함으로써, 행동 정보 없이도 하위 최적 전문가를 넘어설 수 있는 차별점을 가진다.

## 🛠️ Methodology

### 1. DIIQN: Homogeneous Action Space

DIIQN은 전문가와 에이전트의 행동 공간이 동일하다는 가정하에 작동하며, 다음과 같은 파이프라인을 가진다.

**Action Inference (행동 추론)**
에이전트는 온라인 상호작용을 통해 얻은 경험 $\langle s_a, a_a, s'_a \rangle$를 사용하여 전문가의 상태 전이 $\langle s_e, s'_e \rangle$에서 전문가가 취했을 행동 $a_e$를 추론한다. 두 전이 사이의 거리 메트릭 $D$를 계산하여, 가장 유사한 전이를 만들어낸 에이전트의 행동을 전문가의 추론 행동으로 업데이트하고, 그 거리를 행동 오차 메트릭($err_{a_e}$)으로 저장한다.

**Expert Sampling (전문가 샘플링)**
에이전트의 현재 상태 $s_a$와 가장 유사한 전문가 상태 $s_e$를 찾기 위해 KNN(k-Nearest Neighbor) 검색을 수행한다. 이때, 단순 근접성뿐만 아니라 유사도 임계값 $\tau_{similarity}$를 적용하여 정말로 유사한 샘플만 선택한다. 선택된 샘플은 $\langle s_a, a_a, s'_a, r, s_e, a_e, s'_e \rangle$ 형태의 확장된 경험 튜플로 리플레이 버퍼에 저장된다.

**Confidence Mechanism (신뢰 메커니즘)**
최종 손실 함수를 결정하기 위해 다음 세 가지 요소를 결합한 신뢰도 $\Phi$를 계산한다.

1. **Q-value Divergence ($\Delta Q$):** 전문가 행동과 에이전트 행동의 Q-값 차이를 시그모이드 함수로 처리하여 전문가 행동의 유용성을 평가한다.
    $$\Delta Q(s_e, a_e, a_a) = \sigma([Q(s_e, a_e) - Q(s_e, a_a)] \cdot \beta)$$
2. **Training Frequency Weight ($\delta(s_e)$):** 해당 상태 영역이 충분히 학습되었는지 확인하여 Q-값 추정치의 신뢰도를 평가한다.
    $$\delta(s_e) = \frac{\log(1 + \min(c_{s_e}, c_{max}))}{\log(1 + c_{max})}$$
3. **Action Inference Confidence ($\zeta(s_e)$):** 추론된 행동의 오차 메트릭을 기반으로 행동 추론의 정확성을 평가한다.
    $$\zeta(s_e) = 1 - \frac{err_{a_e}}{err_{max}}$$

최종 신뢰도 $\Phi$는 $\Phi(s_e, a_e, a_a) = \min(\Delta Q \cdot \delta(s_e), \zeta(s_e))$로 정의된다.

**Loss Function (손실 함수)**
에이전트의 경험에서 배우는 손실 $\mathcal{L}(\theta^t)_a$와 전문가의 전이에서 배우는 손실 $\mathcal{L}(\theta^t)_e$를 $\Phi$를 통해 동적으로 결합한다.
$$\mathcal{L}(\theta^t) = \Phi \mathcal{L}(\theta^t)_e + (1 - \Phi) \mathcal{L}(\theta^t)_a$$

### 2. HA-DIIQN: Heterogeneous Action Space

전문가와 에이전트의 행동 능력이 다를 때, HA-DIIQN은 다음과 같은 절차를 추가한다.

**Infeasibility Identification (불가능성 식별)**
행동 추론 과정에서 행동 오차 메트릭 $err_{a_e}$가 매우 높으면, 에이전트의 능력으로는 전문가의 전이를 그대로 복제할 수 없는 '불가능한(infeasible)' 전이로 분류한다.

**Bridge Discovery (브릿지 탐색)**
불가능한 전이를 처리하기 위해, 에이전트의 리플레이 버퍼에서 전문가의 이후 경로(trajectory)와 다시 만날 수 있는 대안 경로인 '브릿지'를 탐색한다. 이는 에이전트의 가능한 경로와 전문가의 경로가 교차하는 지점을 찾는 과정이며, 최대 깊이 $k$(에이전트)와 $n$(전문가) 단계까지 탐색한다.

**Learning with Infeasibility (불가능성 기반 학습)**
브릿지가 발견되면, 전문가의 불가능한 행동 대신 브릿지의 첫 번째 가능한 행동 $a_{feas}$를 사용하여 학습한다. 이때, $a_{feas}$는 에이전트가 직접 경험한 행동이므로 행동 추론 신뢰도 $\zeta(s_e)$는 생략되며, 신뢰도는 다음과 같이 단순화된다.
$$\Phi(s_e, a_{feas}, a_a) = \Delta Q(s_e, a_{feas}, a_a) \cdot \delta(s_e)$$

## 📊 Results

### 실험 설정

- **DIIQN 평가:** MinAtar 환경 (Asterix, Breakout, Freeway, Seaquest, Space Invaders)에서 픽셀 기반 상태 표현을 사용하여 평가하였다.
- **HA-DIIQN 평가:** 2D Maze(부분 중첩 행동 공간) 및 Point Maze(완전 분리 행동 공간) 환경에서 평가하였다.
- **데이터셋:** DQN으로 학습시킨 하위 최적(suboptimal) 전문가들의 데이터를 수집하여 사용하였다.

### 주요 결과

1. **DIIQN vs DQN:** Freeway를 제외한 모든 환경에서 DIIQN이 DQN보다 24%~130% 높은 에피소드 보상을 달성하였다. Freeway의 경우 작업이 단순하여 DRL만으로도 최적 정책 발견이 가능하므로 큰 차이가 없었다.
2. **DIIQN vs 다른 IL 방법 (BCO, GAIfO, ORIL):** 기존 IL 방법들은 하위 최적 전문가의 성능 수준에서 보상이 정체되는 '성능 천장' 현상을 보였으나, DIIQN은 이를 훨씬 뛰어넘는 성능을 기록하였다.
3. **HA-DIIQN 성능:** 2D Maze에서 HA-DIIQN은 DIIQN보다 52%, DQN보다 64% 빠르게 수렴하였다. Point Maze(행동 공간 완전 분리)에서는 DIIQN과 DQN이 거의 동일하게 낮은 성능을 보인 반면, HA-DIIQN은 브릿지 메커니즘을 통해 전문가의 데이터를 유효하게 활용하여 훨씬 높은 최종 보상을 달성하였다.

### 파라미터 민감도 분석 (Space Invaders 기준)

- **데이터셋 크기:** 30K 샘플까지는 성능이 급격히 향상되나, 그 이후(100K까지)로는 한계 효용이 감소하는 plateau 현상이 관찰되었다.
- **신뢰 임계값 $c_{max}$:** 낮은 임계값(100K)은 초기 학습 속도를 높이지만 성능 편차가 커지며, 높은 값(150K-225K)이 학습 안정성을 보장한다.
- **유사도 임계값 $\tau_{similarity}$:** 임계값을 높일수록(예: 97%) 학습 데이터의 양은 줄어들지만 질이 높아져 최종 성능이 크게 향상되었다.

## 🧠 Insights & Discussion

**강점 및 기여**
본 논문은 암묵적 모방 학습과 심층 강화학습을 결합하여, 데이터 수집이 어려운 실제 환경(행동 정보 부재, 하위 최적 전문가, 이질적 행동 공간)에서도 유효하게 작동하는 프레임워크를 제시하였다. 특히 전문가의 성능에 얽매이지 않고 이를 도약대로 삼아 최적 정책을 찾아가는 동적 신뢰 메커니즘이 매우 효과적임을 입증하였다.

**한계 및 비판적 해석**

1. **이산 행동 공간의 제한:** 현재 프레임워크는 이산 행동 공간(Discrete Action Space)에 최적화되어 있다. 연속적 행동 공간으로 확장하려면 행동 추론 메커니즘을 밀도 추정(Density Estimation) 방식 등으로 대폭 수정해야 한다.
2. **브릿지 탐색의 비용:** 리플레이 버퍼를 전수 조사하는 브릿지 탐색 방식은 메모리 크기가 커질수록 계산 비용이 기하급수적으로 증가한다. 이를 해결하기 위해 그래프 기반의 상태 표현이나 포워드 다이내믹스 모델(Forward Dynamics Model) 도입이 필요해 보인다.
3. **거리 메트릭 의존성:** 상태 유사도를 판단하는 거리 메트릭을 수동으로 설정해야 한다는 점은 일반화 성능을 떨어뜨린다. Contrastive Learning 등을 이용한 Metric Learning을 통해 자동화할 필요가 있다.

## 📌 TL;DR

본 논문은 행동 정보가 없고 성능이 낮은 전문가의 관찰 데이터만으로도 학습이 가능하며, 심지어 전문가의 성능을 초과할 수 있는 **Deep Implicit Imitation RL 프레임워크(DIIQN, HA-DIIQN)**를 제안한다. 동적 신뢰 메커니즘을 통해 전문가 가이드와 자기 주도 학습의 비중을 조절하며, 브릿지 탐색 기법으로 전문가와 에이전트의 행동 능력이 서로 다른 상황에서도 지식 전이를 가능케 한다. 이 연구는 현실 세계의 불완전한 데이터셋을 활용한 로봇 학습 및 정책 전이 연구에 중요한 기여를 할 것으로 보인다.
