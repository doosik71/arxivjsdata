# Active Imitation Learning from Multiple Non-Deterministic Teachers: Formulation, Challenges, and Algorithms

Khanh Nguyen, Hal Daumé III (2020)

## 🧩 Problem to Solve

본 논문은 상호작용 비용(interaction cost)을 최소화하면서, **비결정론적(non-deterministic)이며 다수의 교사(teachers)로부터 학습하는 능동 모방 학습(Active Imitation Learning, AIL)** 문제를 정의하고 해결하고자 한다.

기존의 능동 모방 학습은 주로 단일한 결정론적 교사와의 상호작용에 집중해 왔다. 그러나 실제 환경에서는 다음과 같은 이유로 다수의 비결정론적 교사로부터 학습하는 능력이 필요하다.
- 하나의 과업을 수행하는 다양한 방법(multiple ways)을 익힘으로써 일반화 성능을 높일 수 있다.
- 확률적인 환경(stochastic environments)에서는 비결정론적인 행동 방식을 학습하는 것이 유리하다.
- 환경 곳곳에 흩어져 있는 다양한 소스로부터 지식을 습득하는 것이 더 효율적일 수 있다.

이때 발생하는 핵심 문제는 **교사의 행동 불확실성(behavioral uncertainty)**이다. 교사 개개인의 행동이 비결정론적이거나(intrinsic), 교사들 간의 의견 차이가 존재할 때(extrinsic), 에이전트는 이를 자신의 무능함으로 오인하여 불필요하게 많은 쿼리를 요청하는 경향이 있다. 본 논문의 목표는 이러한 불확실성 속에서도 효율적으로 쿼리를 결정하여 상호작용 비용을 줄이는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 다음과 같다.

1. **다수 교사 모방을 위한 일반적 프레임워크 제안**: 단순한 정책 학습이 아니라, 정책 공간에 대한 분포(distribution over a policy space)를 학습하기 위해 **정책 페르소나(policy persona)**라는 연속적인 표현(continuous representation) 방식을 도입하였다.
2. **불확실성 기반 AIL의 한계 분석**: 기존의 불확실성(uncertainty)이나 불일치(disagreement) 기반 AIL 알고리즘이 교사의 행동 불확실성을 에이전트의 지식 부족으로 오해하여 중복 쿼리를 생성하는 메커니즘을 이론적, 실험적으로 분석하였다.
3. **APIL(Active Performance-Based Imitation Learning) 알고리즘 개발**: 불확실성 대신 **미래의 과업 진행 상황(future task progress)에 대한 예측**을 기반으로 쿼리 여부를 결정하는 새로운 AIL 알고리즘을 제안하여, 성능 저하 없이 상호작용 비용을 획기적으로 낮추었다.

## 📎 Related Works

기존의 상호작용 모방 학습(Interactive Imitation Learning, I2L)과 능동 모방 학습(AIL) 연구들은 다음과 같은 한계를 가진다.

- **표준 I2L (예: DAgger)**: 에이전트가 방문하는 모든 상태에서 교사에게 쿼리를 요청하므로 쿼리 효율성이 매우 낮다.
- **기존 AIL**: 주로 단일 결정론적 교사를 가정한다. 불확실성 기반 방법론들은 에이전트의 불확실성이 높을 때 쿼리를 요청하는데, 이는 교사가 결정론적일 때는 유효하지만, 교사 자체가 비결정론적이거나 다수의 교사가 서로 다른 정답을 제시하는 상황에서는 무한 루프에 빠지거나 불필요한 쿼리를 남발하게 된다.
- **다수 교사 학습**: 일부 연구에서 잠재 변수(latent variable)를 도입하여 다수 정책을 학습하려 했으나, 본 논문은 교사의 정체성(identity)에 접근 가능하다는 가정하에 정책 공간의 분포를 직접 모델링하는 보다 일반적인 프레임워크를 제시하여 차별점을 둔다.

## 🛠️ Methodology

### 1. Teacher Persona-Aware Framework
에이전트는 단일 정책이 아닌, 상태에 따른 정책의 분포 $\hat{p}_\omega(\pi|s)$를 학습한다. 이를 위해 세 가지 구성 요소를 도입한다.
- **Identity Distribution ($\rho_\psi(k|s)$)**: 현재 상태 $s$에서 어떤 교사 $k$가 선택될지에 대한 확률 분포를 추정한다.
- **Persona Model ($h_\phi(k)$)**: 각 교사 $k$의 고유한 특성을 나타내는 벡터 표현인 페르소나(persona)를 계산한다.
- **Persona-Conditioned Policy ($\hat{\pi}_{\theta,h}(a|s)$)**: 페르소나 $h$가 주어졌을 때의 행동 분포를 정의하는 정책 모델이다.

전체 과정은 $\text{상태 } s \rightarrow \text{교사 식별 } k \rightarrow \text{페르소나 추출 } h \rightarrow \text{행동 결정 } a$ 순으로 진행된다. 학습은 다음의 손실 함수를 최소화함으로써 이루어진다.

$$\min_{\phi,\theta} \mathbb{E}_{s \sim P_{\hat{\pi}}} [\ell_{\text{NLL}}(s, \hat{\pi}_{\theta,h^?_s}, a^?_s)] + \min_{\psi} \mathbb{E}_{s \sim P_{\hat{\pi}}} [\ell_{\text{NLL}}(s, \rho_\psi, k^?_s)]$$

여기서 $\ell_{\text{NLL}}$은 Negative Log-Likelihood 손실 함수이며, 첫 번째 항은 페르소나에 따른 행동을 학습하고, 두 번째 항은 교사의 정체성을 정확히 예측하도록 학습한다.

### 2. Behavioral Uncertainty Analysis
논문은 교사의 불확실성을 두 가지로 구분하여 정의한다.
- **Intrinsic Behavioral Uncertainty**: 개별 교사 내부의 비결정론적 특성으로 인한 불확실성.
- **Extrinsic Behavioral Uncertainty**: 서로 다른 교사들 간의 정책 차이로 인한 불확실성.

이산 분포의 경우, 전체 행동 불확실성(Shannon Entropy)은 다음과 같이 분해된다.
$$H[a^?|s] = \mathbb{E}_{\pi^? \sim p^?}[H[a^?|\pi^?,s]] + I_{\pi^? \sim p^?}[a^?,\pi^?|s]$$
(전체 불확실성 = 내재적 불확실성의 기대값 + 외재적 불확실성을 나타내는 상호 정보량)

### 3. APIL (Active Performance-Based Imitation Learning)
APIL은 불확실성이 아닌 **성능 격차(performance gap)**를 기준으로 쿼리를 결정한다.
- **Performance Gap**: $g(s) = d(s) - d^?_T$로 정의하며, 여기서 $d(s)$는 현재 상태에서 목표까지의 거리, $d^?_T$는 교사를 항상 따라갔을 때의 기대 최종 거리이다.
- **Substantial Progress (실질적 진전)**: 에이전트가 스스로 행동하여 다음 조건 중 하나를 만족하면 '진전이 있다'고 판단한다.
    1. 최종 성능 격차가 매우 작음: $\mathbb{E}[g(s_{T})] \le \epsilon$
    2. 미래의 특정 시점에 성능 격차가 $\sigma$배 이상 감소함: $\exists i < j : \mathbb{E}[g(s_i)] \ge \sigma \cdot \mathbb{E}[g(s_j)]$

**쿼리 정책 학습**:
에이전트는 다음의 전략을 따르는 가상의 쿼리 교사를 모방하도록 학습된다.
- 상태 $s$에서 실질적 진전이 가능하면 $\rightarrow \text{continue}$ (쿼리 하지 않음)
- 실질적 진전이 불가능하면 $\rightarrow \text{query}$ (교사에게 요청)

이 메커니즘은 학습 초기에는 $\text{query}$ 비중을 높여 빠르게 성능을 올리고, 성능이 궤도에 오르면 $\text{continue}$ 비중을 높여 쿼리 비용을 줄이는 평형 상태를 찾아가게 한다.

## 📊 Results

### 실험 설정
- **작업**: GRID (5x5 격자 세계 내비게이션), R2R (실사 환경 내 언어 지시 기반 내비게이션)
- **교사 모델**:
    - `Detm`: 결정론적 단일 교사
    - `Rand`: 무작위 행동 교사
    - `TwoRand`: 동일한 무작위 교사 2명 (외재적 불확실성 0)
    - `TwoDifDetm`: 서로 다른 결정론적 교사 2명 (외재적 불확실성 높음)
- **비교 대상**: BC, DAgger, $\text{IntrUn}$(내재적 불확실성 기반), $\text{BehvUn}$(전체 행동 불확실성 기반), $\text{ErrPred}$(오류 예측 기반)

### 주요 결과
1. **페르소나 모델의 유효성**: 학습된 내재적/외재적 불확실성 수치가 실제 교사 모델의 설정값과 밀접하게 일치함을 확인하였다.
2. **불확실성 기반 AIL의 실패**: $\text{IntrUn}$이나 $\text{BehvUn}$ 방식은 교사가 비결정론적일 때, 에이전트가 이미 충분히 학습했음에도 불구하고 높은 불확실성 값 때문에 계속해서 불필요한 쿼리를 요청하는 현상이 발견되었다.
3. **APIL의 효율성 및 강건성**:
    - **GRID**: APIL은 교사의 성격과 관계없이 쿼리 횟수를 획기적으로 줄였으며, 최적의 상태에서는 거의 쿼리를 하지 않는 행동을 학습하였다.
    - **R2R**: APIL은 BC나 DAgger보다 높은 성공률을 기록하면서도 쿼리 횟수를 약 3배 가까이 줄였다. 특히 교사의 불확실성이 높을 때 다른 AIL 방법론들이 쿼리 횟수를 늘리는 것과 달리, APIL은 일정한 쿼리율을 유지하며 강건한 모습을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 AIL에서 단순히 "내가 얼마나 불확실한가"를 묻는 것이 위험할 수 있음을 지적하였다. 특히 다수의 비결정론적 교사가 존재하는 환경에서는 **에이전트의 불확실성이 교사의 행동 특성에서 기인한 것인지, 아니면 지식의 부족에서 기인한 것인지** 구분하기 어렵다. APIL은 이를 '과업 진행도(progress)'라는 외부 신호를 통해 해결함으로써, 모델의 내부 상태가 아닌 실제 성능 기반의 의사결정을 가능하게 하였다.

### 한계 및 비판적 해석
1. **거리 측정치의 의존성**: APIL은 $d(s)$라는 거리 정보에 의존한다. 하지만 실제 복잡한 환경에서 목표까지의 정확한 거리를 측정하는 것은 매우 어렵거나 불가능할 수 있다. 거리 정보에 노이즈가 섞일 경우 쿼리 정책이 오작동할 가능성이 크다.
2. **학습 용량의 판단 불가**: 현재 알고리즘은 에피소드 내에서의 진전은 판단할 수 있으나, 전체 학습 과정에서 에이전트가 최대 학습 용량(maximal learning capacity)에 도달했는지는 판단하지 못한다.
3. **계산 비용**: 모델 불확실성(Epistemic Uncertainty)을 정확히 추정하기 위해서는 매우 많은 샘플링이 필요하며, 이는 실시간 시스템에서 큰 오버헤드가 될 수 있음을 실험적으로 보여주었다.

## 📌 TL;DR

본 논문은 비결정론적인 다수 교사로부터 학습할 때, 기존의 불확실성 기반 쿼리 전략이 교사의 무작위성을 지식 부족으로 오해하여 쿼리를 남발하는 문제를 제기한다. 이를 해결하기 위해 **정책 페르소나 기반의 모델링**과 **미래 과업 진행도 예측 기반의 쿼리 알고리즘(APIL)**을 제안하였다. 실험 결과, APIL은 교사의 불확실성에 영향을 받지 않고 상호작용 비용을 크게 줄이면서도 높은 작업 수행 성능을 유지함을 입증하였다. 이는 향후 인간-AI 상호작용 시스템에서 효율적인 지식 전수 메커니즘을 구축하는 데 중요한 기초 연구가 될 것으로 보인다.