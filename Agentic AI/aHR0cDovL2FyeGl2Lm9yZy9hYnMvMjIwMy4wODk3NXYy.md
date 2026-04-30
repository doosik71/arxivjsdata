# A SURVEY OF MULTI-AGENT DEEP REINFORCEMENT LEARNING WITH COMMUNICATION

Changxi Zhu, Mehdi Dastani, Shihan Wang (2024)

## 🧩 Problem to Solve

다중 에이전트 강화학습(Multi-Agent Reinforcement Learning, MARL) 시스템에서 각 에이전트는 일반적으로 환경의 전체 상태가 아닌 국소적인 관찰(local observations)만을 수행할 수 있는 부분 관측 가능성(Partial Observability) 문제에 직면한다. 또한, 다른 에이전트들이 동시에 정책을 학습하고 변경함에 따라 각 에이전트가 느끼는 환경이 계속해서 변하는 비정상성(Non-stationarity) 문제가 발생한다.

통신(Communication)은 이러한 문제를 해결하고 에이전트 간의 협력을 촉진하며 환경에 대한 시야를 넓히는 효과적인 메커니즘이다. 최근 통신을 활용한 다중 에이전트 심층 강화학습(Comm-MADRL) 연구가 급증하고 있으나, 기존의 서베이 논문들은 통신을 단순히 미리 정의된 것으로 가정하거나, 매우 제한적인 기준(예: 메시지 수신 대상자 구분)으로만 분류하여 최신 연구 흐름을 체계적으로 분석하고 비교하는 데 한계가 있었다.

본 논문의 목표는 Comm-MADRL 시스템을 설계하고 분석할 수 있는 체계적이고 구조적인 분류 체계를 제안하는 것이며, 이를 위해 9가지 차원(Dimensions)의 분석 프레임워크를 제시한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Comm-MADRL 시스템의 설계 및 분석을 위한 **9가지 다차원 분석 프레임워크(9-dimensional framework)**를 제안한 것이다. 저자들은 에이전트가 '언제, 어떻게, 무엇을' 통신해야 하는가라는 근본적인 질문에서 시작하여, 시스템 전체를 구성하는 요소들을 다음과 같은 9가지 차원으로 정의하였다.

1. **Controlled Goals (제어 목표)**: 협력, 경쟁 또는 혼합된 목표 설정.
2. **Communication Constraints (통신 제약)**: 대역폭 제한, 메시지 손상 등 현실적 제약.
3. **Communicatee Type (통신 대상)**: 주변 에이전트, 전체 에이전트 또는 프록시(Proxy).
4. **Communication Policy (통신 정책)**: 통신 링크를 형성하는 방식(전체, 부분, 개별/전역 제어).
5. **Communicated Messages (통신 메시지)**: 기존 지식 공유 또는 미래 예측 지식 공유.
6. **Message Combination (메시지 결합)**: 수신된 여러 메시지를 통합하는 방법.
7. **Inner Integration (내부 통합)**: 결합된 메시지를 정책(Policy) 또는 가치 함수(Value function)에 통합하는 방식.
8. **Learning Methods (학습 방법)**: 미분 가능 방식, 지도 학습, 강화 학습 또는 정규화 방식.
9. **Training Schemes (훈련 체계)**: 중앙 집중식, 분산식 또는 CTDE(Centralized Training and Decentralized Execution).

이 프레임워크를 통해 기존의 41개 모델을 투영하여 분석함으로써 현재의 연구 트렌드를 발견하고, 향후 연구 방향을 제시하였다.

## 📎 Related Works

### MARL의 기본 정식화
본 논문은 MARL 환경을 부분 관측 가능 확률 게임(Partially Observable Stochastic Game, POSG)으로 정의한다. POSG는 다음과 같은 튜플로 표현된다.
$$\langle I, S, \rho_0, \{A_i\}, P, \{O_i\}, O, \{R_i\} \rangle$$
여기서 $I$는 에이전트 집합, $S$는 상태 공간, $A_i$는 액션 집합, $O_i$는 관찰 집합, $P$는 상태 전이 확률, $R_i$는 보상 함수를 의미한다. 특히 보상 함수가 모든 에이전트에게 동일한 경우 이는 Dec-POMDP로 축소된다.

### 기존 학습 방식의 한계
- **Value-based methods**: 개별 Q-함수를 학습하는 분산 Q-러닝은 비정상성 문제에 취약하다. 이를 해결하기 위해 공동 Q-함수를 분해하는 Value Decomposition 방식이 제안되었으며, 식은 다음과 같다.
$$Q_{\text{joint}}(\vec{\tau}, \vec{a}) = \sum_{i} w_i Q_i(\tau_i, a_i)$$
- **Policy-based methods**: 정책 경사(Policy Gradient) 정리를 통해 정책을 직접 최적화한다. MADDPG와 같은 Actor-Critic 구조는 중앙 집중식 Critic을 통해 전역 정보를 활용하고 분산된 Actor를 통해 실행하는 방식을 취한다.

### Emergent Language vs. Comm-MADRL
본 논문은 '발현 언어(Emergent Language)' 연구와 '통신을 활용한 학습 작업(Learning tasks with communication)'을 구분한다. 전자는 기호적 언어(symbolic language) 자체를 학습하는 것이 주 목적이며, 후자는 통신을 도구로 사용하여 도메인 특정 작업(예: 내비게이션, 게임)의 성능을 높이는 것이 목적이다.

## 🛠️ Methodology

본 논문은 새로운 알고리즘을 제안하는 대신, Comm-MADRL 시스템을 설계하는 가이드라인(Procedure 1)과 9가지 분석 차원을 상세히 설명한다.

### 1. 시스템 설계 프로세스 (Guideline)
에이전트가 환경과 상호작용하며 통신하는 과정은 다음과 같은 순서로 진행된다.
1. **목표 설정**: 협력/경쟁/혼합 보상 체계 설계 (Dim 1).
2. **제약 조건 설정**: 통신 비용 및 노이즈 고려 (Dim 2).
3. **대상 지정**: 누구와 통신할 것인지 결정 (Dim 3).
4. **통신 실행**: 
   - 통신 여부 및 대상 결정 (Dim 4) $\rightarrow$ 메시지 생성 및 공유 (Dim 5) $\rightarrow$ 수신 메시지 결합 (Dim 6) $\rightarrow$ 내부 모델 통합 (Dim 7).
5. **행동 결정**: 통신 결과를 반영하여 액션 선택 및 수행.
6. **학습 및 업데이트**: 경험을 바탕으로 정책, 가치 함수 및 통신 프로토콜 업데이트 (Dim 8, 9).

### 2. 주요 분석 차원의 상세 설명
- **Communication Policy (Dim 4)**: 통신 링크 형성 방식을 네 가지로 분류한다.
    - Full Communication: 모든 에이전트가 브로드캐스트 방식으로 연결.
    - Partial Structure: 미리 정의된 부분 그래프 구조를 사용.
    - Individual Control: 각 에이전트가 게이트(Gate) 메커니즘 등을 통해 스스로 통신 여부를 결정.
    - Global Control: 전역 스케줄러가 통신 링크를 제어.
- **Communicated Messages (Dim 5)**:
    - Existing Knowledge: 과거 관찰, 행동 이력 등을 인코딩하여 전달.
    - Imagined Future Knowledge: 의도(Intention)나 미래 계획(Future plans)을 예측하여 전달.
- **Learning Methods (Dim 8)**: 통신 프로토콜을 어떻게 학습하는가에 따라 분류한다.
    - Differentiable: 메시지 생성 함수를 미분 가능하게 설계하여 역전파(Backpropagation) 수행.
    - Supervised: 통신 여부나 내용에 대한 정답(Label)을 정의하여 지도 학습.
    - Reinforced: 통신 행위 자체에 보상을 부여하여 강화 학습.
    - Regularized: 상호 정보량(Mutual Information) 최소화 등 정규화 항을 추가.
- **Training Schemes (Dim 9)**: 
    - CTDE (Centralized Training and Decentralized Execution): 중앙에서 모든 에이전트의 경험을 모아 학습시키되, 실행 시에는 개별 정책만 사용하는 방식. 파라미터 공유(Parameter Sharing) 여부에 따라 further 구분된다.

## 📊 Results

본 논문은 실험적 결과 대신, 제안한 9차원 프레임워크를 사용하여 기존의 41개 Comm-MADRL 모델을 분류한 종합 표(Table 13)를 제시한다.

### 분석 결과 및 트렌드
- **목표(Goal)**: 대부분의 연구가 협력(Cooperative) 설정에 집중되어 있으며, 경쟁(Competitive)이나 혼합(Mixed) 설정에 대한 연구는 매우 부족하다.
- **제약(Constraints)**: 많은 연구가 통신 제약이 없는 이상적인 환경을 가정하고 있어, 실제 물리적 환경(대역폭 제한, 지연 시간 등)에 적용하기에는 한계가 있다.
- **통신 대상(Communicatee)**: 프록시(Proxy)를 이용한 중앙 집중식 조정 방식이 효율적으로 사용되고 있다.
- **학습 방법(Learning Methods)**: 미분 가능한(Differentiable) 방식이 가장 지배적이며, 특히 CTDE 구조와 파라미터 공유 방식이 널리 채택되고 있다.

### 평가 지표 분석
Comm-MADRL 연구에서 사용되는 주요 지표를 분석한 결과, 보상 기반(Reward-based) 지표가 가장 많이 사용되었으며, 통신 효율성(Communication Efficiency)이나 언어 발현 정도(Emergence Degree)를 측정하는 지표는 상대적으로 적게 사용되었음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문은 파편화되어 있던 Comm-MADRL 연구들을 9가지 차원이라는 통합된 관점에서 정리함으로써, 연구자들이 새로운 시스템을 설계할 때 고려해야 할 체크리스트를 제공하였다. 특히 통신을 단순한 정보 교환이 아닌, 설계 가능한 여러 구성 요소의 조합(Combinatorial Problem)으로 바라본 점이 매우 통찰력 있다.

### 한계 및 향후 방향
1. **비협력적 설정의 부족**: 에이전트 간의 신뢰(Trust) 문제나 기만적 통신(Deceptive communication)에 대한 연구가 필요하다.
2. **현실적 제약 반영**: 비동기 통신, 메시지 손실, 동적 네트워크 구조 등 실제 통신 환경의 불확실성을 모델링해야 한다.
3. **멀티모달 통신 (Multimodal Communication)**: 단순 벡터 형태의 메시지를 넘어 음성, 텍스트, 이미지 등 다양한 모달리티를 통합하는 통신 체계가 필요하다.
4. **구조적 통신 (Structural Communication)**: 대규모 시스템에서 계층적 구조나 브리지 에이전트(Bridge agent)를 통한 효율적인 정보 전달 경로 최적화 연구가 유망하다.
5. **강건한 중앙 유닛 (Robust Centralized Unit)**: 프록시나 중앙 Critic이 오염된 정보나 악의적인 메시지에 영향을 받지 않도록 하는 강건성(Robustness) 확보가 필수적이다.

## 📌 TL;DR

본 논문은 Comm-MADRL 분야의 복잡한 연구들을 체계적으로 분류하기 위해 **목표, 제약, 대상, 정책, 메시지, 결합, 통합, 학습 방법, 훈련 체계**라는 9가지 분석 차원을 제안하였다. 이를 통해 기존 41개 모델의 특성을 명확히 규명하였으며, 향후 연구가 단순한 협력적 설정을 넘어 멀티모달 통신, 강건한 중앙 유닛 설계, 비협력적 환경에서의 통신 프로토콜 학습 방향으로 나아가야 함을 제시하였다.