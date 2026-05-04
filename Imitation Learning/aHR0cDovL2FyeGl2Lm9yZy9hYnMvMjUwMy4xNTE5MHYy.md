# Learning Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach

Mohamed Hassouna, Clara Holzhüter, Malte Lehna, Matthijs de Jong, Jan Viebahn, Bernhard Sick, Christoph Scholz (2025)

## 🧩 Problem to Solve

본 논문은 재생 가능 에너지의 비중 증가로 인해 전력망 운영자가 직면한 전력망 혼잡 관리(Congestion Management) 문제를 해결하고자 한다. 전력망의 안정성을 유지하기 위해서는 동적인 상태 변화에 대응하여 전력망의 위상(Topology)을 적절히 변경하는 적응형 의사결정 전략이 필수적이다.

최근 이를 위해 딥러닝 기반의 접근 방식이 제안되었으나, 기존의 모방 학습(Imitation Learning, IL) 방법들은 전문가(Expert)의 단일 최적 행동만을 모방하는 Hard Label 방식을 사용한다. 하지만 실제 전력망 운영에서는 혼잡을 해소할 수 있는 유효한 행동이 여러 개 존재할 수 있다. 단일 행동만을 학습하는 Hard Label 방식은 솔루션 공간의 불확실성을 포착하지 못하며, 전문가의 잠재적인 편향(Bias)이나 하위 최적(Sub-optimal)의 결정까지 그대로 학습하게 되어 정책의 강건성과 일반화 성능을 저하시키는 문제를 야기한다.

따라서 본 논문의 목표는 여러 유효한 행동들을 동시에 학습할 수 있는 Soft-Label 기반의 모방 학습 프레임워크를 제안하고, 전력망의 구조적 특성을 반영하기 위해 Graph Neural Networks (GNNs)를 통합하여 보다 강건하고 적응력 있는 전력망 위상 제어 에이전트를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전문가의 행동을 단일 정답으로 처리하는 대신, 시뮬레이션된 결과의 효과성을 바탕으로 확률 분포 형태의 Soft Label을 생성하여 학습시키는 것이다. 이를 통해 에이전트는 특정 상태에서 어떤 행동들이 유효한지에 대한 풍부한 감독 신호(Supervisory Signal)를 제공받게 된다.

주요 기여 사항은 다음과 같다:

1. **Soft-Label Imitation Learning 도입**: 다수의 유효한 위상 변경 행동을 학습 신호에 포함함으로써, 전문가의 하위 최적 결정에 대한 오버피팅을 방지하고 모델의 일반화 능력을 향상시켰다.
2. **GNN 기반의 구조적 인코딩**: 전력망의 물리적 연결 구조를 반영하기 위해 GNN을 통합하여, 각 그리드 구성 요소의 문맥적으로 풍부한 표현(Representation)을 학습하고 의사결정 성능을 높였다.
3. **성능 입증**: 제안 방법이 기존의 Hard-Label IL 모델뿐만 아니라 최신 심층 강화학습(DRL) 베이스라인, 그리고 학습의 대상이 된 Greedy Expert 에이전트보다도 뛰어난 성능을 보임을 입증하였다.

## 📎 Related Works

전력망의 위상 최적화를 통한 혼잡 관리는 RTE의 L2RPN(Learning to Run a Power Network) 챌린지를 통해 활발히 연구되어 왔다. 기존 연구들은 주로 모델 프리(Model-free) 기반의 DRL 알고리즘을 사용하며, 많은 경우 Feed-forward Neural Network (FNN)를 기반으로 구축되었다. 최근에는 전력망의 그래프 특성을 활용하기 위해 GNN을 도입하는 추세이다.

모방 학습(IL)의 경우, 규칙 기반 전문가의 행동을 모방하여 DRL 학습의 초기 속도를 높이는 Pre-training 용도로 사용되거나, 추론 시간을 단축하기 위한 하이브리드 에이전트 형태로 연구되었다. 그러나 기존 IL 방식들은 결정론적(Deterministic) 정책에 의존하여 전문가의 편향을 그대로 이어받으며, 과부하 완화를 위한 다양한 대안적 행동들을 간과한다는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 레이블의 표현력을 높인 Soft-Label 방식을 제안하여 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. Greedy Expert ($\text{Greedy}_{90\%}$)

학습 데이터 생성을 위한 전문가 에이전트로, 현재 상태에서 가능한 모든 위상 변경 행동을 시뮬레이션하고, 결과적으로 최대 선로 부하(Maximum Line Loading, $\rho^{\max}$)를 가장 낮게 만드는 행동을 선택하는 반응형 에이전트이다. $\rho^{\max}$가 90%를 초과할 때만 활성화된다.

### 2. Soft Labels 생성

단일 최적 행동만을 저장하는 대신, 모든 가능한 행동 $a \in A$에 대해 효과성 점수(Effectiveness Score) $e_a$를 계산한다.
$$e_a = 1 - \rho^{\max}(s, a)$$
이 점수를 Temperature Softmax 함수에 적용하여 Soft Label $\Psi(a \mid s)$를 생성한다.
$$\Psi(a \mid s) = \frac{\exp(e_a / \tau)}{\sum_{a' \in A} \exp(e_{a'} / \tau)}$$
여기서 $\tau = 0.01$은 소프트맥스 분포를 날카롭게 만들어 효과적인 행동에 더 많은 확률 질량을 배분하기 위해 설정된 온도 파라미터이다. 이를 통해 모델은 행동 간의 상대적인 효과성을 학습하게 된다.

### 3. Graph-Based Encoding 및 GNN 아키텍처

전력망을 그래프로 표현하여 각 구성 요소(부하, 발전기, 선로 끝단, 저장 장치)를 노드로, 물리적 연결을 에지로 정의한다.

- **Graph Construction**: 노드 특성으로는 전압, 전력 주입량, 냉각 시간, 유지보수 정보 등이 포함된다. 선로의 특성(전력 흐름, 부하 등)은 해당 선로와 연결된 노드들에 직접 인코딩된다.
- **Architecture**: Graph Attention Network (GAT)를 사용하여 인접 노드 간의 관계를 모델링한다.
  - 4개의 GAT 레이어를 통해 노드 표현을 정제한다.
  - Global Max Pooling을 통해 그래프 전체의 표현을 추출한다.
  - 3개의 Linear 레이어를 거쳐 행동 공간 크기의 출력층으로 연결된다.
- **Loss Function**: 예측된 분포와 Soft Label 간의 차이를 최소화하기 위해 Kullback-Leibler Divergence ($\text{KLDivLoss}$)를 사용한다.

### 4. Agent 작동 메커니즘 및 최적화

- **활성화 조건**: $\rho^{\max} > 90\%$인 응급 상황에서만 모델이 예측한 행동 순위대로 실행한다. $\rho^{\max} < 90\%$인 경우, 기본 위상으로 복구하는 Topology Reversion을 시도한다.
- **Topology Reversion**: 모든 버스바 커플러를 닫아 기본 위상으로 되돌리는 작업으로, $\rho^{\max}$가 80%를 넘지 않는 안전한 범위 내에서 수행한다.
- **Feasibility 개선**: 연결이 끊긴 선로에 대해 불필요한 버스 할당 시도를 제거하는 전처리 단계를 도입하여 행동의 실행 가능성(Feasibility)을 높였다.
- **$\text{SoftGNN}_{90\%} N-1$**: 추론 시 상위 10개 행동에 대해 $N-1$ (단일 선로 고장) 시나리오를 시뮬레이션하여, 최악의 경우에도 가장 안전한(최저 $\rho^{\max}$를 가진) 행동을 선택하는 강건한 전략을 추가하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: IEEE 118-bus 송전 시스템 (WCCI 2022 환경, Grid2Op 플랫폼).
- **비교 대상**: $\text{DoNothing}$, $\text{Greedy}_{90\%}$, $\text{HardFNN}_{90\%}$, $\text{HardGNN}_{90\%}$, $\text{SoftFNN}_{90\%}$, $\text{SoftGNN}_{90\%}$, 그리고 SOTA RL 에이전트인 $\text{Senior}_{95\%}$, $\text{TopoAgent}_{85-95\%}$.
- **지표**: L2RPN Score (운영 비용 및 안정성 종합 점수), Median Survival Time, MSTCM (Median Survival Time across Chronic Medians).

### 주요 결과

- **Soft Label의 효과**: $\text{SoftGNN}_{90\%}$는 동일한 구조의 $\text{HardGNN}_{90\%}$보다 L2RPN 점수가 약 15% 향상되었다. 이는 Hard Label이 전문가의 편향을 그대로 학습하는 반면, Soft Label은 더 일반적인 효과성을 학습하기 때문이다.
- **GNN의 효과**: $\text{SoftGNN}_{90\%}$는 $\text{SoftFNN}_{90\%}$보다 우수한 성능을 보였으며, 이는 GNN이 전력망의 공간적 의존성과 물리적 구조를 효과적으로 캡처했음을 의미한다.
- **SOTA 대비 성능**: $\text{SoftGNN}_{90\%} N-1$ 에이전트는 평균 L2RPN 점수 $44.43$과 MSTCM $1566$을 기록하며, 최신 RL 기반 에이전트들보다 더 높은 성능을 달성하였다. 특히 학습 대상이었던 $\text{Greedy}_{90\%}$ 전문가보다 17% 더 나은 성능을 보였다.
- **통계적 유의성**: Welch's t-test 결과, $\text{SoftGNN}_{90\%} N-1$과 다른 베이스라인 간의 차이는 통계적으로 유의미함($p < 0.05$)이 확인되었다.

## 🧠 Insights & Discussion

본 연구는 전문가의 데이터를 단순히 모방하는 것을 넘어, 데이터에 내재된 불확실성과 대안적 가능성을 학습하는 것이 성능 향상의 핵심임을 보여주었다. Hard Label 방식은 전문가가 선택한 '단 하나의 정답'에 집착하게 만들어, 특정 상태에서는 최적일지 모르나 장기적인 관점에서는 불안정한 위상을 선택하게 하는 전문가의 편향을 전이시킨다.

반면, Soft Label은 일종의 신뢰도 점수(Confidence Score) 역할을 하여, 여러 행동이 유사하게 효과적일 때 모델이 특정 행동에 과도하게 확신하는 것을 방지한다. 이를 통해 모델은 개별 결정의 암기보다는 행동 공간의 구조적 패턴을 학습하게 되며, 결과적으로 학습 데이터에 없던 새로운 상태에 대해서도 더 강건한 일반화 능력을 갖게 된다.

또한, GNN의 도입은 전력망의 물리적 제약 조건을 모델이 이해하도록 돕는다. 특히 $N-1$ 보안 기준을 추론 단계에 통합함으로써, 단순한 혼잡 해소를 넘어 전력망의 회복 탄력성(Resilience)까지 고려한 의사결정이 가능함을 확인하였다. 이러한 결과는 실제 전력망 운영실에서 AI 기반 의사결정 지원 도구로 활용될 때, 단순한 추천을 넘어 안전성이 검증된 다수의 대안을 제시할 수 있는 가능성을 시사한다.

## 📌 TL;DR

본 논문은 전력망 혼잡 관리를 위해 **Soft-Label 모방 학습**과 **GNN**을 결합한 새로운 접근 방식을 제안하였다. 전문가의 단일 행동만 모방하는 기존 방식의 한계를 극복하기 위해, 행동의 효과성을 확률 분포로 표현한 Soft Label을 사용하여 학습시켰으며, GNN을 통해 전력망의 구조적 특성을 반영하였다. 실험 결과, 제안 방법은 기존의 Hard-Label 모델은 물론 최신 강화학습 에이전트와 Greedy 전문가보다 우수한 성능을 보였으며, 특히 $N-1$ 보안 분석을 결합했을 때 가장 높은 안정성을 달성하였다. 이는 불완전한 전문가 데이터로부터도 더 우수한 정책을 도출할 수 있음을 보여주며, 실제 전력망 운영의 의사결정 지원 시스템으로 응용될 가능성이 매우 높다.
