# CLIC: Curriculum Learning and Imitation for object Control in non-rewarding environments

Pierre Fournier, Cédric Colas, Mohamed Chetouani, Olivier Sigaud (2019)

## 🧩 Problem to Solve

본 논문은 외부 보상(external reward)이 존재하지 않는 환경에서 에이전트가 어떻게 주변 객체들을 제어하는 능력을 습득할 수 있는가에 대한 문제를 다룬다. 현실적인 환경은 단일 작업(single task)을 수행하는 것이 아니라, 제어 가능성이 각기 다른 여러 객체들이 존재하며, 이들 사이에 계층 구조가 있을 수 있다.

기존의 강화학습(RL)은 환경이 제공하는 명확한 보상 신호에 의존하여 최적의 정책을 학습하지만, 실제 환경에서는 사전 정의된 작업이나 외적 보상이 없는 경우가 많다. 또한, 단순히 무작위 탐색(random exploration)을 통해 여러 객체를 제어하려 할 경우 다음과 같은 문제점이 발생한다.

1. **샘플 효율성 저하**: 객체가 많을 때 무작위 탐색은 학습 속도가 매우 느리며 많은 샘플을 소모한다.
2. **환경 구조의 무시**: 어떤 객체는 다른 객체를 먼저 제어해야만 제어가 가능하거나(계층 구조), 아예 제어가 불가능할 수 있다. 무작위 선택은 이미 마스터한 객체나 제어 불가능한 객체에 시간을 낭비하게 만든다.

따라서 본 논문의 목표는 보상이 없는 환경에서 다른 에이전트(Bob)의 행동을 관찰하고, 스스로 학습 커리큘럼을 설계하여 효율적으로 객체 제어 능력을 획득하는 에이전트인 **CLIC(Curriculum Learning and Imitation for Control)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Goal-Conditioned Policies(GCP)**, **Imitation Learning**, 그리고 **Curriculum Learning**을 결합하여 보상이 없는 환경에서 객체 수준의 제어 능력을 효율적으로 습득하는 것이다.

구체적인 설계 직관은 다음과 같다.

- **객체 수준의 제어**: 전체 상태 공간이 아닌, 특정 객체의 상태만을 변경하는 목표를 설정하여 제어 능력을 세분화한다.
- **우연한 시연(Fortuitous Demonstrations)의 활용**: 다른 에이전트(Bob)가 명시적으로 가르치지 않더라도, 그가 환경과 상호작용하며 남긴 궤적에서 객체 제어에 도움이 되는 정보를 추출하여 모방한다.
- **학습 진전(Learning Progress, LP) 기반 커리큘럼**: 에이전트가 현재 어느 객체에서 가장 빠르게 성장하고 있는지를 측정하여, 학습 효율이 가장 높은 객체에 집중하고 이미 마스터했거나 제어 불가능한 객체는 무시하도록 설계한다.

## 📎 Related Works

### 1. 외적 보상이 없는 강화학습 (Unsupervised RL)

최근 스킬의 다양성을 극대화하거나, 상태 간의 상호 정보량(mutual information)을 최대화하는 등의 unsupervised RL 연구들이 진행되었다. 특히 Goal-Conditioned Policies(GCP)와 Universal Value Function Approximators(UVFA)를 결합하여 목표 상태에 도달했을 때 스스로 보상을 주는 방식이 제안되었다. 그러나 기존 GCP는 전체 상태를 목표로 하기 때문에, 여러 객체가 독립적으로 존재하는 환경에서 특정 객체만을 제어하고자 할 때 비효율적이라는 한계가 있다.

### 2. 다중 스킬 모방 학습 (Imitation Learning for Multiple Skills)

Learning from Demonstrations(LfD)는 보통 전문가가 특정 작업에 대해 명시적인 시연을 제공한다고 가정한다. 하지만 본 논문은 보조자(Bob)가 학습자와 독립적으로 행동하며 의도가 관찰되지 않는 상황을 가정한다는 점에서 차별점을 갖는다.

### 3. 커리큘럼 학습 (Curriculum Learning)

학습 진전(Learning Progress)을 최대화하여 연습할 스킬을 선택하는 방식이 제안된 바 있다. 본 논문은 이를 확장하여, 객체 제어라는 구체적인 맥락에서 절대적 학습 진전(absolute LP) 최대화를 통해 샘플 효율성을 높이고 환경의 구조(계층 구조 등)를 활용한다.

## 🛠️ Methodology

### 1. 환경 모델 (Environment Model)

환경은 $N \times N$ 그리드 월드로 모델링되며, $n$개의 객체 $O_i$가 존재한다. 각 객체는 그리드 상의 위치 리스트 $L_i = (p_1^i, \dots, p_{l_i}^i)$와 연결되어 있다. 에이전트가 특정 위치 $p_{k+1}^i$에서 `ACT` 액션을 수행하면 객체의 내부 상태 $o_i$가 $k$에서 $k+1$로 전이된다.

- **계층 구조 모델링**: 객체 $O_1$의 위치 리스트 $L_1$이 $L_2$에 포함($L_1 \subset L_2$)된다면, $O_2$를 제어하기 위해 $O_1$을 먼저 제어해야 하는 계층 관계가 형성된다.
- **상태 정의**: 전체 상태는 에이전트의 위치와 모든 객체의 내부 상태의 집합인 $s_A = (x_A, y_A, o_1, \dots, o_n)$으로 정의된다.

### 2. 객체별 독립 제어 (Separate Control of Objects)

특정 객체만을 제어하기 위해 보상 함수를 다음과 같이 수정한다.
$$R^{g,w}(s) = \begin{cases} 0, & \text{if } |w \cdot (g-s)| \le \epsilon \\ -1, & \text{otherwise} \end{cases}$$
여기서 $w$는 가중치 벡터이며, 특정 객체 $O_i$만을 제어하고자 할 때는 $O_i$의 상태 위치만 1이고 나머지는 0인 one-hot 벡터를 사용한다. 이를 통해 다른 객체의 상태와 관계없이 목표 객체의 상태 $g$를 달성했는지만 평가한다. 학습에는 Double DQN이 사용되며, 손실 함수 $J_{DQ}$는 다음과 같다.
$$J_{g,w_i}^{DQ}(Q) = [R_{g,w_i}(s') + \gamma \tilde{Q}_{g,w_i}(s', \text{argmax}_a Q_{g,w_i}(s', a)) - Q_{g,w_i}(s, a)]^2$$

### 3. 모방 학습 (Imitation)

에이전트는 Bob의 궤적을 관찰하고, Bob이 어떤 객체의 상태를 변화시킨 지점을 포착하여 이를 시연(demonstration)으로 간주한다. 모방 학습 단계에서는 DQNfD(DQN from Demonstrations)를 확장하여 다음과 같은 Large Margin Classification Loss를 사용한다.
$$J_{g,w_i}^{I}(Q) = \max_{a \in A} [Q_{g,w_i}(s_B, a) + l(a_B, a)] - Q_{g,w_i}(s_B, a_B)$$
여기서 $l(a_B, a)$는 $a=a_B$일 때 0, 그 외에는 1인 마진 함수이다. 이는 Bob이 선택한 행동의 $Q$값이 다른 행동보다 일정 마진 이상 높게 유지되도록 강제한다.

### 4. 커리큘럼 학습 (Curriculum Learning)

에이전트는 각 객체 $O_i$에 대한 역량(competence) $C_k(O_i)$를 윈도우 내 평균 성공률로 측정한다. 학습 진전(Learning Progress, LP)은 다음과 같이 정의된다.
$$LP_k(O_i) = |C_k(O_i) - C_{k-l}(O_i)|$$
객체를 샘플링할 확률 $p_{k,\epsilon}(O_i)$는 다음과 같다.
$$p_{k,\epsilon}(O_i) = \epsilon \times \frac{1}{N} + (1-\epsilon) \times \frac{LP_k(O_i)}{\sum_j LP_k(O_j)}$$
여기서 $\epsilon=0.2$를 사용하여 최소한의 무작위 탐색을 유지하면서, LP가 높은(즉, 현재 가장 빠르게 성장하고 있는) 객체에 우선적으로 집중한다.

## 📊 Results

### 1. 실험 설정

- **환경**: $11 \times 11$ 그리드 월드. $E_6$(독립 객체 6개), $E_1/E_3$(일부 제어 가능), $E_h$(계층적 구조) 총 4가지 시나리오.
- **지표**: 평균 정규화된 역량(Average normalized competence) $C(k)$.

### 2. 주요 결과

- **Bob의 영향**: Bob이 더 많은 객체를 제어할수록 CLIC의 학습 속도가 빨라졌다. 특히 $E_h$에서 Bob이 상위 객체를 제어할 때, CLIC은 그 과정에 포함된 하위 객체들의 제어 능력까지 함께 습득하는 효과를 보였다.
- **멘토링 효과**: Bob이 객체 $O_1 \to O_2 \to \dots \to O_6$ 순서로 시연을 제공했을 때, LP 최대화를 사용하는 CLIC은 Bob이 제시한 순서를 엄격히 따라 학습했다. 반면 무작위 샘플링을 하는 CLIC-RND는 Bob의 순서를 완전히 따르지 못했다.
- **비재현 가능 행동의 무시**: 에이전트가 제어할 수 없는 객체가 섞여 있는 $E_1, E_3$ 환경에서, CLIC은 LP가 낮은(성장이 없는) 제어 불가능 객체를 빠르게 식별하고 무시함으로써, 제어 가능한 객체에 더 빠르게 집중하여 CLIC-RND보다 높은 성능을 보였다.
- **마스터한 객체의 무시**: 계층 구조 $E_h$에서 Bob이 쉬운 객체들을 반복적으로 시연하더라도, CLIC은 이미 마스터하여 LP가 0이 된 쉬운 객체들에 대한 모방을 중단하고 어려운 객체 학습에 자원을 집중했다.

## 🧠 Insights & Discussion

본 논문은 단순히 데이터를 모방하는 것이 아니라, **'무엇을 학습하고 모방할 것인가'**를 결정하는 커리큘럼 전략이 보상이 없는 복잡한 환경에서 얼마나 중요한지를 입증하였다.

특히 인상적인 점은 **LP(Learning Progress)가 필터(filter)로서 작동**한다는 것이다.

1. **불가능한 것의 필터링**: 제어 불가능한 객체는 시간이 지나도 역량이 오르지 않으므로 $LP \approx 0$이 되어 자연스럽게 무시된다.
2. **이미 아는 것의 필터링**: 이미 마스터한 객체 역시 역량이 포화 상태에 이르러 $LP \approx 0$이 되므로, 더 이상 불필요한 모방 학습에 시간을 낭비하지 않는다.

한계점으로는 모든 실험이 이산 상태 및 이산 액션 환경에서 수행되었다는 점이 있다. 또한, 객체의 식별(identification)을 가정하고 시작했으므로, 실제 환경에서 raw sensor data로부터 객체 특징을 스스로 추출하는 representation learning 단계가 추가되어야 실용성이 높아질 것이다.

## 📌 TL;DR

본 논문은 보상이 없는 환경에서 다른 에이전트의 행동을 모방하고 스스로 학습 순서를 결정하는 **CLIC** 에이전트를 제안한다. CLIC은 Goal-Conditioned Policies를 통해 객체별 제어 능력을 학습하며, 학습 진전(Learning Progress)을 최대화하는 커리큘럼 전략을 사용하여 **제어 불가능하거나 이미 마스터한 객체를 효율적으로 걸러내고 학습 속도를 극대화**한다. 이는 로봇이 외부의 명시적 가이드 없이도 주변 환경의 구조를 파악하고 효율적으로 기술을 습득하는 방향으로 나아가는 중요한 연구이다.
