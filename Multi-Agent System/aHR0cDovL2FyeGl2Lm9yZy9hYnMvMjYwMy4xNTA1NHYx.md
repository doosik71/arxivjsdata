# Interference-Aware K-Step Reachable Communication in Multi-Agent Reinforcement Learning

Ziyu Cheng, Jinsheng Ren, Zhouxian Jiang, Chenzhihang Li, Rongye Shi, Bin Liang, Jun Yang (2026)

## 🧩 Problem to Solve

Multi-Agent Reinforcement Learning (MARL)에서 에이전트 간의 효율적인 통신은 복잡한 협업 과제를 해결하는 데 핵심적인 요소이다. 하지만 실제 환경에서는 통신 대역폭의 제한과 동적이고 복잡한 환경 토폴로지로 인해, 어떤 에이전트가 가장 가치 있는 통신 파트너인지 식별하는 것이 매우 어렵다.

기존의 통신 파트너 선택 방식은 주로 유클리드 거리(Euclidean distance)나 시야(Line-of-Sight) 기반의 근접성 제약에 의존한다. 그러나 이러한 방식은 다음과 같은 한계가 있다.
1. **실제 도달 가능성 오판**: 장애물이 있는 환경에서 유클리드 거리가 가깝더라도 실제 이동 경로는 매우 길 수 있어, 통신 효율이 떨어진다.
2. **시야의 한계**: 시야 기반 방식은 장애물에 가려진 도달 가능한 파트너를 식별하지 못한다.
3. **동적 간섭 무시**: 적군의 공격이나 다른 에이전트와의 상호작용으로 발생하는 '간섭'을 고려하지 않아, 물리적으로는 가깝지만 협력 비용이 지나치게 높은 파트너를 선택하는 문제가 발생한다.

본 논문의 목표는 물리적 도달 가능성(Physical Reachability)과 동적 간섭(Dynamic Interference)을 동시에 고려하여, 가장 효율적이고 지속 가능한 협력 파트너를 선택하는 통신 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Interference-Aware K-Step Reachable Communication (IA-KRC)** 프레임워크를 통해 통신 범위를 물리적으로 도달 가능한 영역으로 제한하고, 그 안에서 간섭을 최소화하는 파트너를 선택하는 것이다.

주요 기여 사항은 다음과 같다.
- **K-Step Reachability Protocol**: 단순 거리가 아닌, 실제 MDP(Markov Decision Process) 상에서 $K$ 스텝 이내에 도달 가능한 에이전트로 통신 대상을 제한하여 물리적 제약 조건을 충실히 반영하였다.
- **Interference-Prediction Module**: 적군의 공격 의도와 방향성을 예측하는 Potential Field를 도입하여, 협력 비용이 낮은 최적의 파트너를 식별한다.
- **Multi-layer Map 구조**: 정적 지형, 환경 규칙, 동적 간섭 정보를 서로 다른 시간 척도로 분리하여 관리함으로써, 비정상성(Non-stationarity)이 강한 환경에서도 효율적으로 거리와 간섭을 계산한다.

## 📎 Related Works

기존의 MARL 통신 연구들은 주로 다음과 같은 접근 방식을 취해왔다.
- **근접성 기반 제약**: 유클리드 거리나 시야 기반 방식을 사용하였으나, 앞서 언급한 대로 복잡한 토폴로지에서 실제 도달 가능성을 반영하지 못하는 한계가 있다.
- **End-to-End 학습 기반**: Attention 메커니즘이나 GNN(Graph Neural Networks)을 통해 통신 토폴로지를 학습하는 방식(예: CommFormer)이 제안되었다. 이러한 방식은 소규모 설정에서는 강력하지만, 물리적 공간 제약(Spatial Priors)이 부족하고 에이전트 수가 증가함에 따라 확장성(Scalability)이 떨어지는 경향이 있다.
- **K-Step Reachability**: 단일 에이전트 RL에서는 하위 목표(Subgoal) 설정에 사용되었으나, 이를 MARL의 통신 파트너 선택에 적용하고 특히 '간섭' 개념과 결합한 시도는 본 논문이 처음이다.

## 🛠️ Methodology

### 1. Interference-Aware K-Step Reachability
본 논문은 단순히 물리적 거리를 측정하는 것이 아니라, **Shortest Transition Distance**와 **Cooperation Cost**를 결합한 새로운 거리 개념을 제안한다.

**Shortest Transition Distance ($d^{st}$)**:
상태 $x_1$에서 $x_2$로 이동하는 데 걸리는 기대 시간의 최솟값으로 정의한다.
$$d^{st}(x_1, x_2) := \min_{\pi \in \Pi} \mathbb{E}[T_{x_1, x_2} | \pi]$$

**Interference-Aware Shortest Transition Distance ($d^{IA}$)**:
이동 시간과 협력 비용 $C$를 곱하여, 간섭이 심한 경로의 비용을 높게 평가한다.
$$d^{IA}(x_1, x_2) := \min_{\pi \in \Pi} \sum_{t=0}^{\infty} t P(T_{x_1, x_2} = t | \pi) \times C(T_{x_1, x_2} = t | \pi)$$

이에 따라 **Interference-Aware K-Step Reachable Region** $S^{IA}(x_1, K)$는 $d^{IA}(x_1, x_2) \le K$를 만족하는 에이전트들의 집합으로 정의된다.

### 2. Multi-layer Map (MLM)
환경의 변화 속도가 서로 다르다는 점에 착안하여 세 가지 레이어로 정보를 분리하여 관리한다.
- **Geometric Layer**: 정적 장애물 및 매우 느리게 변하는 지형 정보 저장.
- **Regulation Layer**: 문(Door)의 개폐 상태와 같은 환경 규칙 기반의 연결성 정보 저장.
- **Interference Layer**: 적군의 위치 및 공격 의도 등 가장 빠르게 변하는 간섭 정보 저장.

이 레이어들은 비동기적으로 업데이트되며, Dijkstra 알고리즘을 통해 통합된 그래프 $G^{(t)}$ 상에서 최단 경로와 도달 가능성을 계산한다.

### 3. Directional Interference Potential Field
협력 비용 $C$를 구체화하기 위해 방향성 간섭 포텐셜 필드를 도입한다.
개별 엔티티 $e_i$에 의한 상태 $x$에서의 간섭 강도 $I(x | e_i)$는 다음과 같이 모델링된다.
$$I(x | e_i) := I^{base} e^{-d^{eff}/\lambda^{base}}$$
이때, 유효 거리 $d^{eff}$는 적군의 공격 의도 방향 $\theta$를 반영하여 계산한다.
$$d^{eff} = d^{actual}(1 + \alpha(1 - \cos\theta))$$
$\theta$는 신경망을 통해 예측된 공격 의도 벡터와 실제 위치 사이의 각도이다. 공격 방향이 일치할수록($\theta \to 0$) $d^{eff}$가 감소하여 간섭 강도가 강해진다.

최종적인 협력 비용 $C$는 경로 상의 모든 상태에서 합산된 간섭 강도의 평균으로 정의된다.
$$C = \frac{1}{t} \sum_{x \in S} [1 + I(x)]$$

### 4. Grouping 및 학습 절차
- **Leader Election**: K-step 도달 가능 범위 내의 이웃 수 $N^{(K)}_i$가 가장 많은 상위 $M$명의 에이전트를 리더로 선정한다.
- **Follower Assignment**: 각 팔로워는 $d^{IA} \le K$를 만족하는 리더 후보 중, 현재 그룹 규모가 가장 작은 리더에게 소속되어 부하를 분산한다.
- **Policy Learning**: 각 그룹 $g$는 QMIX 가치 분해 프레임워크를 사용하여 공동의 행동-가치 함수 $Q^{g}_{tot}$를 학습하며, 전체 그룹의 TD loss를 최소화하는 방향으로 최적화한다.
$$\mathcal{L}(\theta) = \sum_{g \in G} \mathbb{E} \left[ (y^{tot}_g - Q^{g}_{tot}(\tau^g, a^g; \theta))^2 \right]$$

## 📊 Results

### 실험 설정
- **환경**: SMACv2 기반의 커스텀 맵(Dense-Obstacle, Maze-Structure) 및 표준 8m 맵.
- **비교 대상**: MAPPO, QMIX, CommFormer, Euclid, SOG(Vision), SOG(RL-Vision), DPP.
- **평가 지표**: 최종 승률(FW), 최고 승률(HW), 최종 패배율(FL).
- **프레임워크**: 알고리즘 간의 상대적 성능을 측정하기 위해 Self-play 프레임워크를 구축하여 평가하였다.

### 주요 결과
- **복잡한 토폴로지에서의 성능**: Dense-Obstacle 및 Maze-Structure 맵 모두에서 IA-KRC가 baseline 대비 압도적인 승률 우위를 보였다. 특히 MAPPO, QMIX 대비 최소 4.58배에서 최대 31.56배 높은 승률 이점을 기록하였다.
- **확장성(Scalability)**: 팀 규모가 $3v3$에서 $18v18$으로 증가할수록 IA-KRC의 성능 우위가 더 뚜렷해졌다. 이는 팀 규모가 커질수록 그룹 구성의 조합 공간이 폭발적으로 증가하는데, IA-KRC의 reachability 필터링이 효율적인 구조를 빠르게 찾도록 돕기 때문이다. 계산 복잡도는 에이전트 수 $N$에 대해 거의 선형적으로 증가하여 확장성이 입증되었다.
- **그룹 구조 분석**: IA-KRC는 고립된 에이전트 비율(Iso Rate)이 가장 낮았으며, 대수적 연결성($\lambda_2$)이 높고 분산이 작았다. 이는 통신 그래프가 더 견고하고 정보 흐름이 원활함을 의미한다.
- **일반화 능력**: 장애물이 없는 표준 8m 맵에서도 CommFormer를 제외한 대부분의 알고리즘보다 빠른 수렴 속도와 높은 승률을 보였으며, 특히 CommFormer 대비 학습 시간이 훨씬 짧아 효율적임을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문은 MARL 통신에서 '물리적 도달 가능성'과 '동적 간섭'이라는 두 가지 실제적인 제약 조건을 명시적으로 모델링하였다. 이를 통해 단순히 거리가 가까운 파트너가 아닌, 실제로 협력 가능한 최적의 파트너를 선택함으로써 협력의 지속성을 높였다. 특히 'Avalanche Effect'(일부 에이전트의 고립 $\to$ 순차적 제거 $\to$ 팀 전체 붕괴)를 효과적으로 방지한 점이 인상적이다.

### 한계 및 분석
- **K-step Horizon ($K$)의 민감도**: Ablation Study 결과, $K=9$에서 최적의 성능을 보였으며, $K$가 너무 작으면 협력 범위가 제한되고, 너무 크면 예측 불확실성과 노이즈가 누적되어 성능이 저하되는 비단조적(Non-monotonic) 관계를 보였다. 이는 하이퍼파라미터 $K$ 설정이 환경의 크기와 동역학에 따라 매우 중요함을 시사한다.
- **모듈별 기여도**: interference prediction을 제거했을 때보다 K-step reachability를 제거(유클리드 거리로 대체)했을 때 성능 하락이 더 컸다($\approx 18$pt 하락). 이는 복잡한 환경에서 '실제 도달 가능성'을 파악하는 것이 '간섭 예측'보다 더 근본적인 성능 요인임을 의미한다.

### 비판적 해석
본 연구는 물리적 기반의 prior를 강하게 부여함으로써 성능을 올렸으나, 이는 환경의 맵 정보나 이동 비용에 대한 사전 지식이 어느 정도 필요함을 전제로 한다. 완전히 미지의 환경에서 이러한 reachability를 스스로 학습해야 하는 상황에서의 작동 여부는 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 MARL에서 물리적 도달 가능성과 적군의 간섭을 모두 고려한 통신 파트너 선택 프레임워크인 **IA-KRC**를 제안한다. **K-step Reachability**와 **방향성 간섭 포텐셜 필드**를 통해 최적의 협력 그룹을 동적으로 형성하며, 이를 통해 장애물이 많고 적군이 존재하는 복잡한 환경에서 기존 SOTA 모델(CommFormer, MAPPO 등)보다 월등한 승률과 확장성을 달성하였다. 이 연구는 물리적 제약 조건이 강한 실제 로봇 군집 제어나 전술 시뮬레이션 분야에 중요한 기여를 할 가능성이 높다.