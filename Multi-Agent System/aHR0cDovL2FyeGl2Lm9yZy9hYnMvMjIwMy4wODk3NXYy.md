# A SURVEY OF MULTI-AGENT DEEP REINFORCEMENT LEARNING WITH COMMUNICATION

Changxi Zhu, Mehdi Dastani, Shihan Wang (2022/2024)

## 🧩 Problem to Solve

본 논문은 다중 에이전트 심층 강화학습(Multi-Agent Deep Reinforcement Learning, MADRL) 환경에서 에이전트 간의 통신(Communication)을 통해 성능을 향상시키려는 연구들을 체계적으로 분석하고 분류하는 것을 목표로 한다.

다중 에이전트 시스템은 다음과 같은 근본적인 문제들에 직면해 있다:
1. **부분 관측성(Partial Observability)**: 각 에이전트가 환경의 전체 상태가 아닌 국소적인 관측치만을 가질 수 있어 최적의 의사결정이 어렵다.
2. **비정상성(Non-stationarity)**: 모든 에이전트가 동시에 학습하며 정책을 변경하므로, 각 에이전트가 마주하는 환경이 동적으로 변화하여 학습의 불안정성이 초래된다.

통신은 이러한 문제를 해결하기 위해 에이전트들이 관측치, 의도, 경험 등을 공유함으로써 환경에 대한 더 넓은 시야를 확보하게 하는 핵심 기제이다. 그러나 최근 Comm-MADRL 분야의 연구가 급증했음에도 불구하고, 기존의 서베이 논문들은 분류 체계가 너무 단순하거나 최신 연구들을 충분히 포괄하지 못하는 한계가 있었다. 따라서 본 논문은 Comm-MADRL 시스템을 설계하고 분석할 수 있는 정밀하고 구조적인 9가지 차원의 분류 체계를 제안하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Comm-MADRL 시스템을 다각도로 분석할 수 있는 **9가지 분석 차원(9 Dimensions)**을 제안한 것이다. 이는 단순히 기존 논문을 나열하는 것이 아니라, 시스템 설계 시 고려해야 할 핵심 질문들(언제, 어떻게, 무엇을 통신할 것인가)을 기반으로 구조화된 프레임워크를 제공한다.

제안된 9가지 차원은 다음과 같다:
1. **제어 목표(Controlled Goals)**: 협력, 경쟁, 또는 혼합된 형태의 목표 설정.
2. **통신 제약(Communication Constraints)**: 대역폭 제한이나 메시지 손실/오염 등의 현실적 제약.
3. **통신 대상(Communicatee Type)**: 인접 에이전트, 모든 에이전트, 또는 프록시(Proxy)를 통한 통신.
4. **통신 정책(Communication Policy)**: 통신 연결을 설정하는 방식(전체, 부분, 개별 제어, 글로벌 제어).
5. **통신 메시지(Communicated Messages)**: 공유하는 정보의 종류(기존 지식 vs. 추론된 미래 지식).
6. **메시지 결합(Message Combination)**: 수신된 여러 메시지를 통합하는 방법(동일 가중치 vs. 차등 가중치).
7. **내부 통합(Inner Integration)**: 결합된 메시지를 정책이나 가치 함수에 통합하는 수준.
8. **학습 방법(Learning Methods)**: 통신 프로토콜을 학습시키는 기법(미분 가능, 지도 학습, 강화 학습, 정규화).
9. **훈련 체계(Training Schemes)**: 경험 데이터를 활용하는 방식(중앙 집중형, 분산형, CTDE).

## 📎 Related Works

본 논문은 기존 MARL 서베이들과의 차별점을 명확히 하기 위해 관련 연구를 분석한다.

1. **기존 서베이의 한계**: 초기 연구들은 통신이 미리 정의되었다고 가정했거나, 메시지 수신 대상(Broadcasting, Targeted 등)과 같은 매우 좁은 범위의 분류만을 제시하였다. 최신 Comm-MADRL의 동적인 특성과 학습 가능한 프로토콜을 충분히 반영하지 못했다.
2. **Emergent Language vs. Comm-MADRL**:
    - **Emergent Language**: 주요 목표가 에이전트 간의 '상징적 언어(Symbolic Language)' 자체를 학습하고 진화시키는 것에 있다.
    - **Comm-MADRL**: 통신을 도구로 활용하여 '도메인 특정 작업(Domain-specific Tasks)'의 성능을 극대화하는 것이 주 목적이다.
    - 본 논문은 이 두 영역의 교집합인 '학습 가능한 상징적 언어를 활용한 작업 수행'까지 포함하여 분석 범위를 설정하였다.

## 🛠️ Methodology

본 논문은 Comm-MADRL 시스템의 설계 가이드라인을 제공하기 위해 다음과 같은 이론적 배경과 분류 방법론을 제시한다.

### 1. MARL의 기본 정식화 (POSG)
다중 에이전트 환경은 부분 관측 확률 게임(Partially Observable Stochastic Game, POSG)으로 정의되며, 튜플 $\langle I, S, \rho_0, \{A_i\}, P, \{O_i\}, O, \{R_i\} \rangle$로 표현된다. 여기서 $I$는 에이전트 집합, $S$는 상태 공간, $A_i$는 액션 집합, $O_i$는 관측 집합을 의미한다.

### 2. 핵심 학습 알고리즘 설명
통신이 통합되는 기반이 되는 두 가지 주요 학습 방식을 설명한다.

- **가치 기반 방법 (Value-based)**: 각 에이전트가 로컬 Q-함수를 학습한다. 협력 설정에서의 업데이트 식은 다음과 같다:
$$Q^i(s, a^i) \leftarrow Q^i(s, a^i) + \alpha(r + \gamma \max_{a'^i} Q^i(s', a'^i) - Q^i(s, a^i))$$
또한, 공동 가치 함수를 분해하여 효율성을 높이는 가치 분해(Value Decomposition) 방식이 사용된다:
$$Q_{\text{joint}}(\vec{\tau}, \vec{a}) = \sum_{i=1}^n w_i Q^i(\tau^i, a^i)$$

- **정책 기반 방법 (Policy-based)**: 정책 $\pi$를 직접 최적화한다. 중앙 집중식 학습에서의 정책 경사(Policy Gradient) 식은 다음과 같다:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\vec{a} \sim \pi(\cdot|s), s \sim \rho^\pi} [\nabla_\theta \log \pi(\vec{a}|s; \theta) Q^\pi(s, \vec{a})]$$
특히 MADDPG와 같은 Actor-Critic 구조에서는 중앙 집중식 Critic이 글로벌 정보를 캡처하고, 로컬 Actor가 분산 실행을 담당하는 구조를 가진다.

### 3. Comm-MADRL 시스템 설계 절차 (Procedure 1)
논문은 시스템 설계를 위한 단계별 가이드라인을 제안한다:
- **설정 단계**: 목표 설정(Dim 1) $\rightarrow$ 통신 제약 설정(Dim 2) $\rightarrow$ 통신 대상 결정(Dim 3).
- **실행 단계**: 통신 여부 및 대상 결정(Dim 4) $\rightarrow$ 메시지 생성 및 공유(Dim 5) $\rightarrow$ 수신 메시지 결합(Dim 6) $\rightarrow$ 내부 모델 통합(Dim 7) $\rightarrow$ 액션 선택 및 수행.
- **훈련 단계**: 경험 데이터를 통한 정책 및 통신 프로세스 업데이트(Dim 8, 9).

## 📊 Results

본 논문은 41개의 Comm-MADRL 모델을 분석하여 다음과 같은 정량적/정성적 경향성을 도출하였다.

### 1. 분석 결과 (Findings)
- **목표(Goals)**: 대부분의 연구가 협력(Cooperative) 설정에 치중되어 있으며, 경쟁(Competitive)이나 혼합(Mixed) 시나리오에 대한 탐구는 부족하다.
- **제약(Constraints)**: 많은 연구가 통신 비용이나 노이즈가 없는 '제약 없는 통신(Unconstrained)'을 가정하고 있어, 실제 환경 적용 시 한계가 있다.
- **훈련 체계(Training Schemes)**: 중앙 집중식 훈련 및 분산 실행(CTDE) 방식과 파라미터 공유(Parameter Sharing) 기법이 가장 널리 채택되고 있다.
- **통신 정책(Policy)**: 단순한 전체 통신(Full Communication)에서 점차 개별 제어(Individual Control)나 글로벌 제어(Global Control) 방식으로 발전하는 추세이다.

### 2. 평가 지표 분석
Comm-MADRL 연구에서 주로 사용되는 5가지 지표를 분석하였다:
- **보상 기반(Reward-based)**: 누적 보상 및 평균 보상 측정.
- **성공률(Win or Fail Rate)**: 목표 달성 비율 측정.
- **소요 단계(Steps Taken)**: 목표 도달까지의 시간 효율성 측정.
- **통신 효율성(Communication Efficiency)**: 통신 리소스 사용량 측정 (가장 적게 사용되는 지표).
- **발현 정도(Emergence Degree)**: 메시지와 관측/행동 간의 상관관계를 통해 언어의 발현을 측정 (주로 Emergent Language 연구에서 사용).

## 🧠 Insights & Discussion

### 1. 강점 및 한계
- **강점**: 파편화되어 있던 Comm-MADRL 연구들을 9가지 차원이라는 통일된 프레임워크로 묶어내어, 새로운 시스템 설계 시 체크리스트로 활용할 수 있는 학술적 기반을 마련하였다.
- **한계**: 실제 현실의 통신 지연(Latency), 비동기적 통신, 그리고 에이전트 간의 신뢰(Trust) 및 기만(Deception) 문제에 대한 다각적인 분석이 부족하다.

### 2. 비판적 해석 및 향후 방향
저자는 현재의 Comm-MADRL 연구가 지나치게 '시뮬레이션상의 협력'에 매몰되어 있다고 해석한다. 이를 극복하기 위해 다음과 같은 연구 방향을 제시한다:
- **멀티모달 통신(Multimodal Communication)**: 텍스트, 음성, 영상 등 다양한 데이터 소스를 통합하여 통신하는 체계가 필요하다.
- **구조적 통신(Structural Communication)**: 단순한 그래프 연결을 넘어, 브릿지 에이전트(Bridge Agent)를 활용한 계층적/구조적 정보 전달 방식이 탐구되어야 한다.
- **강건한 중앙 유닛(Robust Centralized Unit)**: 프록시나 Critic이 오염된 메시지나 악의적인 공격에 영향을 받지 않도록 하는 강건성(Robustness) 확보가 필수적이다.
- **복잡한 상징 체계**: 단순 벡터 형태의 메시지가 아닌, 논리 표현식이나 그래프 형태의 복잡한 메시지 형식을 학습하는 방향으로 나아가야 한다.

## 📌 TL;DR

본 논문은 다중 에이전트 심층 강화학습에서 통신(Communication)을 다루는 최신 연구들을 체계적으로 분류하기 위해 **9가지 분석 차원(제어 목표, 통신 제약, 통신 대상, 통신 정책, 메시지 내용, 메시지 결합, 내부 통합, 학습 방법, 훈련 체계)**을 제안하였다. 

이 프레임워크를 통해 기존 연구들이 주로 협력적인 환경과 제약 없는 통신에 치중되어 있음을 발견하였으며, 향후 연구가 **멀티모달 통신, 구조적 통신 네트워크, 그리고 악의적 메시지에 강건한 시스템 설계**로 확장되어야 함을 시사한다. 이 연구는 향후 Comm-MADRL 시스템을 설계하는 연구자들에게 정밀한 설계 지침서 역할을 할 것으로 기대된다.