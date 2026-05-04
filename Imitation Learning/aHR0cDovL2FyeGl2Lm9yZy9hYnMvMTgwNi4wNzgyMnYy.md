# Learning Neural Parsers with Deterministic Differentiable Imitation Learning

Tanmay Shankar, Nicholas Rhinehart, Katharina Muelling, Kris M. Kitani (2018)

## 🧩 Problem to Solve

본 논문은 대형 객체를 도색하는 로봇의 예시와 같이, 복잡한 공간적 작업(spatial tasks)을 효율적인 세그먼트(segments)로 분할하는 문제를 해결하고자 한다. 로봇이 단 한 번의 붓질로 전체 객체를 덮을 수 없는 물리적 제약이 있을 때, 전체 작업을 처리 가능한 작은 단위로 나누는 '분할' 과정이 필수적이다.

이러한 공간 분할 문제의 중요성은 다음과 같다. 첫째, 분할된 세그먼트들이 전체 도색 범위(coverage)를 충분히 만족해야 하며, 둘째, 불필요한 도색 낭비(paint wasted)를 최소화해야 한다. 또한, 객체의 형태에 따라 가로 또는 세로 방향으로 분할하는 등 다양한 분할 방법이 존재하므로 최적의 분할 전략을 찾는 것이 매우 까다롭다.

따라서 본 연구의 목표는 원본 이미지(raw object images)만을 입력으로 받아, 전문가(expert)의 분할 방식을 모방하여 객체를 최적으로 분할할 수 있는 **Neural Parser**를 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체의 계층적 분할 과정이 전통적인 결정 트리(Decision Tree) 알고리즘인 ID3의 입력 공간 분할 방식과 매우 유사하다는 직관에서 출발한다.

1.  **IGM Oracle 정의**: 정답 라벨(ground-truth)을 사용하여 정보 이득(Information Gain)을 최대화하는 분할 방식을 수행하는 $\pi_{IGM}$ 알고리즘을 전문가 오라클로 정의하였다.
2.  **Imitation Learning 도입**: 정답 라벨이 없는 새로운 이미지에 대해서도 분할을 수행하기 위해, IGM 오라클의 결정 과정을 모방하는 모방 학습(Imitation Learning) 프레임워크를 제안하였다.
3.  **DRAG 알고리즘 제안**: 결정론적 정책(Deterministic Policy)을 효과적으로 학습시키기 위해 **DRAG(DeteRministically AGgrevate)**라는 새로운 정책 경사 업데이트 방법을 제안하였다. 이는 AggreVaTeD의 결정론적 액터-크리틱(Actor-Critic) 변형 버전으로, 특히 분할 위치(split location)와 같이 연속적인 값을 결정론적으로 예측해야 하는 설정에 최적화되어 있다.

## 📎 Related Works

-   **Facade Parsing**: 건물의 외벽 구조를 분석하여 구성 요소로 분할하는 연구들이 존재한다. 기존 연구들은 주로 강화 학습(RL)을 통해 문법 규칙을 적용하거나 결과물에 대한 라벨을 사용하는 방식이었으나, 본 논문은 전문가의 '결정 과정' 자체를 모방한다는 점에서 차별점을 갖는다.
-   **Policy Gradient RL**: 확률적 정책 경사(Stochastic Policy Gradient) 방법들이 널리 사용되어 왔으며, 이후 Deterministic Policy Gradient (DPG)가 제안되었다. 본 논문은 DPG를 모방 학습 설정으로 확장하여 적용한다.
-   **Imitation Learning**: DAgger나 AggreVaTe와 같이 전문가의 궤적이나 비용-전이(cost-to-go) 추정치를 사용하는 방법들이 있다. 본 연구는 AggreVaTeD의 확률적 업데이트를 결정론적 업데이트로 변형하여 적용하였다.
-   **Semantic Segmentation**: 픽셀 단위로 의미론적 라벨을 부여하는 연구들과 유사해 보일 수 있으나, 본 논문의 접근 방식은 최종 결과물로 계층적 분할 구조(Hierarchical Decomposition)를 생성하며, 도색 범위와 같은 물리적 제약 조건을 고려한다는 점에서 차이가 있다.

## 🛠️ Methodology

### 1. Shape Parsing 및 문법 정의
객체를 분할하기 위해 확률적 문맥 자유 문법(Probabilistic Context-Free Grammar, PCFG) $G = (V, T, R, V_0, P)$를 사용한다.
-   **Non-terminal ($V$)**: 이미지 상의 축 정렬 직사각형(axis-aligned rectangle)으로, $(x, y, w, h)$ 속성을 가진다.
-   **Terminal ($T$)**: 최종 분할된 세그먼트로, 도색 여부($b$) 속성을 가진다.
-   **Production Rules ($R$)**:
    -   가로 분할: $V \xrightarrow{h:l} (V_{left}, V_{right})$
    -   세로 분할: $V \xrightarrow{v:l} (V_{top}, V_{bottom})$
    -   라벨 할당: $V \rightarrow b_p$ (도색함) 또는 $V \rightarrow b_{np}$ (도색 안 함)
여기서 $l$은 분할 위치(split location)를 의미한다.

### 2. MDP(마르코프 결정 과정) 정식화
분할 과정을 순차적 의사결정 문제로 정의한다.
-   **상태($s$)**: 현재 확장해야 할 이미지 세그먼트(노드 $N$).
-   **행동($a$)**: 적용할 생성 규칙 $r$과 분할 위치 $l$의 선택.
-   **보상 및 리턴**: 터미널 노드 $T$에서 예측 라벨 $P$와 실제 라벨 $L$ 사이의 상관관계를 계산하며, 비터미널 노드 $V$의 리턴 $G(V)$는 자식 노드들의 리턴 합으로 정의된다 (Eq. 1).
$$G(N) = \begin{cases} \sum_{(x,y) \in N} L(x,y)P(x,y) & \text{if } N \in T \\ \sum_{c \in \text{Children}(N)} G(c) & \text{if } N \in V \end{cases}$$

### 3. DRAG (DeteRministically AGgrevate)
정책은 두 가지 구성 요소로 이루어진다: 규칙 선택을 위한 확률적 정책 $\pi(r|s, \theta)$와 분할 위치를 위한 결정론적 정책 $\mu(s|\theta)$.

**DRAG**는 오라클의 cost-to-go $Q^*$를 최소화하는 것을 목표로 한다. 하지만 $Q^*$는 미분 불가능하므로, 이를 근사하는 크리틱 네트워크 $Q(s, a|\omega)$를 학습시킨다.
$$\min_{\omega} \mathbb{E}_{s_t, a} [(Q(s_t, a|\omega) - G_t)^2]$$
여기서 $G_t$는 오라클이 해당 상태 이후로 얻은 실제 리턴(Monte Carlo sample)이다.

최종적인 혼합 정책 경사(Mixed Policy Gradient) 업데이트 식은 다음과 같다 (Eq. 18):
$$\nabla_{\theta} J_n(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log \pi(r_t|s_t, \theta) \cdot Q_t(s_t, r_t, \mu(s_t|\theta)|\omega) + \nabla_l Q_t(s_t, r_t, l|\omega)|_{l=\mu(s_t|\theta)} \cdot \nabla_{\theta} \mu(s_t|\theta) \right]$$
-   **첫 번째 항**: 규칙 선택($r$)에 대한 확률적 정책 경사.
-   **두 번째 항**: 분할 위치($l$)에 대한 결정론적 정책 경사.

이 방식은 크리틱이 '학습자의 정책'이 아닌 '전문가 오라클의 cost-to-go'를 근사하므로, $Q$가 $\theta$에 의존하지 않아 DPG의 기존 근사 단계 없이 직접적인 경사 하강이 가능하다.

## 📊 Results

### 실험 설정
-   **데이터셋**: 362장의 RGB 객체 이미지 (256x256), 3-fold 교차 검증 수행.
-   **지표**: 예측된 라벨과 정답 라벨 사이의 픽셀 정확도(Pixel Accuracy).
-   **비교 대상**: 
    -   RL: MCPG, DDPG
    -   IL: Behavior Cloning, DAgger
    -   Hybrid IL+RL: AggreVaTeD, AC-AggreVaTeD, Off-MCPG, Off-ACPG

### 정량적 결과
| 모델 | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: |
| IGM Oracle (GT access) | 98.50% | — |
| MCPG (RL) | 53.54% | 51.23% |
| DDPG (RL) | 51.94% | 48.78% |
| Behavior Cloning (IL) | 75.11% | 75.10% |
| DAgger (IL) | 84.01% | 84.03% |
| DRAG (Ours) | **88.05%** | **86.86%** |

### 주요 분석 결과
1.  **RL의 실패**: 분할 과정의 재귀적 특성상 무작위 탐색만으로는 유의미한 분할 규칙과 위치를 찾기 어려워 성능이 매우 낮게 나타났다.
2.  **IL의 효과**: IGM 오라클이 가이드라인을 제공함으로써 RL보다 훨씬 높은 성능을 보였다. 단순한 Behavior Cloning조차 RL보다 월등했다.
3.  **DRAG의 우수성**: DRAG는 다른 모든 베이스라인을 능가했다. 특히 분할 위치를 결정론적으로 예측함으로써 객체의 경계선에 더 정교하게 일치하는 분할(regular parses)을 생성할 수 있었다. 또한, 전문가가 제한된 트리 깊이로 인해 놓친 부분까지도 비용 민감형 모방(cost-sensitive imitation)을 통해 더 정확하게 라벨링하는 일반화 능력을 보였다.

## 🧠 Insights & Discussion

**강점**:
-   공간 분할 문제를 결정 트리 구조와 연결하여 IGM 오라클이라는 강력한 전문가를 정의한 점이 탁월하다.
-   확률적 요소(규칙 선택)와 결정론적 요소(분할 위치)가 혼합된 정책을 위해 DRAG라는 효율적인 업데이트 메커니즘을 제안하여 학습 안정성과 정확도를 높였다.
-   크리틱이 오라클의 $Q$값을 근사하게 함으로써 정책 파라미터 $\theta$와의 의존성을 제거하고 정확한 경사(true gradient)를 계산할 수 있게 한 수학적 통찰이 돋보인다.

**한계 및 논의**:
-   본 연구는 축 정렬(axis-aligned) 분할만을 다루고 있어, 대각선 분할이나 더 복잡한 비선형 분할이 필요한 객체에 대해서는 한계가 있을 수 있다.
-   사용된 데이터셋의 규모(362장)가 딥러닝 모델을 학습시키기에 상대적으로 작으므로, 더 대규모의 다양한 데이터셋에서 일반화 성능을 검증할 필요가 있다.
-   실험에서 트리 깊이를 7로 제한하였는데, 이 깊이 설정이 최종 성능에 미치는 영향에 대한 추가 분석이 있었다면 더 좋았을 것이다.

## 📌 TL;DR

본 논문은 로봇의 공간 작업 분할 문제를 해결하기 위해 **IGM(Information Gain Maximization) 오라클을 모방하는 Neural Parser**를 제안한다. 특히 분할 위치를 정교하게 학습시키기 위해 결정론적 액터-크리틱 구조의 **DRAG** 알고리즘을 도입하였으며, 이를 통해 기존의 강화 학습 및 모방 학습 방식보다 훨씬 정확하고 경계선에 부합하는 객체 분할 성능을 달성하였다. 이 연구는 복잡한 공간 분할 작업이 필요한 로보틱스 제어 및 컴퓨터 비전 분야에 중요한 방법론을 제공한다.