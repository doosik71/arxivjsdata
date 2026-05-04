# SocialGFs: Learning Social Gradient Fields for Adaptive Multi-Agent Systems

Qian Long, Fangwei Zhong, Mingdong Wu, Yizhou Wang, Song-Chun Zhu (2024)

## 🧩 Problem to Solve

멀티 에이전트 시스템(Multi-Agent Systems, MAS)은 동적인 환경, 변화하는 에이전트의 수, 그리고 다양한 작업(Task)에 적응적으로 대응해야 한다. 그러나 대부분의 MAS는 상태 공간(State Space)과 작업 공간의 복잡성으로 인해 이러한 변화에 쉽게 대처하지 못하는 한계가 있다.

특히 기존의 멀티 에이전트 강화학습(MARL) 방법론들은 특정 작업이나 환경에 과적합(Tailored)되는 경향이 있어, 학습된 지식을 새로운 설정으로 전이(Transfer)하거나 재사용(Reuse)하는 능력이 부족하다. 따라서 복잡한 환경을 일반적이고 효율적으로 표현하여 다양한 시나리오에 적응할 수 있는 표현 학습(Representation Learning) 방법이 절실히 요구되는 상황이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사회학의 **Social Force(사회적 힘)** 개념을 딥러닝의 **Gradient Field(그래디언트 필드)**로 구현하여 에이전트의 상태 표현으로 사용하는 것이다.

주요 기여 사항은 다음과 같다:

1. **학습 가능한 Gradient Field 제안**: 오프라인 예제 데이터로부터 **Denoising Score Matching (DSM)** 기법을 통해 사회적 힘을 모사하는 그래디언트 필드를 학습한다.
2. **적응형 MARL 에이전트 개발**: 학습된 그래디언트 필드를 상태 표현으로 사용하여, 환경이나 작업이 바뀌더라도 그래디언트 필드만 교체함으로써 빠르게 적응할 수 있는 에이전트를 구현한다.
3. **실증적 효과 입증**: 협력-경쟁 게임(Grassland)과 완전 협력 게임(Cooperative Navigation) 환경에서 제안 방법론이 기존 베이스라인보다 우수한 성능과 일반화 능력을 보임을 입증하였다.

## 📎 Related Works

### 1. 적응형 멀티 에이전트 시스템 (Adaptive MAS)

기존 연구들은 Graph Neural Networks(GNN), Attention mechanism, Meta-learning, Curriculum learning 등을 통해 확장성(Scalability)과 일반화 성능을 높이려 했다. 하지만 이러한 방법들은 작업 간의 유사성이 높을 때만 효과적이며, 완전히 새로운 환경으로의 전이는 어렵다는 한계가 있다.

### 2. 사회적 힘 모델 (Social Force Model, SFM)

SFM은 보행자 흐름 시뮬레이션 등에서 인력(Attraction)과 척력(Repulsion)을 이용하여 개별 행동을 모델링하는 데 사용되었다. 그러나 SFM은 수동으로 파라미터를 설계해야 하므로 캘리브레이션이 어렵고, 특정 환경에 고정되어 일반화 능력이 떨어진다.

### 3. 의사결정을 위한 Gradient Field

Artificial Potential Fields 등을 이용한 경로 계획(Path Planning) 연구가 있었으나, 대부분 타겟 함수가 명시적으로 정의되어야 하는 planning 기반 방식이었다. 최근에는 TarGF와 같이 데이터로부터 score function을 학습하는 방식이 등장하였으며, 본 논문은 이를 MAS의 상태 표현으로 확장하여 적용하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 시스템은 **[오프라인 예제 수집 $\rightarrow$ SocialGFs 학습 $\rightarrow$ GF 기반 상태 표현 생성 $\rightarrow$ MARL 정책 학습]**의 순서로 진행된다.

### 2. Social Gradient Fields (SocialGFs) 학습

에이전트가 지향해야 할 상태(Attractive)와 피해야 할 상태(Repulsive)를 정의한 오프라인 예제 세트 $S$를 구성한다. 각 예제 분포 $p_{data}(x)$에 대해, 로그 밀도 함수의 그래디언트인 score function $\nabla_x \log p_{data}(x)$를 학습한다.

실제 $p_{data}(x)$를 알 수 없으므로, **Denoising Score Matching (DSM)**을 사용하여 노이즈가 섞인 데이터 $\tilde{x}$를 원래 데이터 $x$로 복원하는 방향의 score network $s_\theta(x, t)$를 학습한다. 손실 함수는 다음과 같다:

$$L(\theta) = \mathbb{E}_{t \sim U(\epsilon, T)} \left( \mathbb{E}_{\tilde{x} \sim q_\sigma(t)(\tilde{x}|x), x \sim p_{data}(x)} \lambda(t) \left\| s_\theta(\tilde{x}, t) - \frac{1}{\sigma^2(t)}(x - \tilde{x}) \right\|^2 \right)$$

여기서 $s_\theta$는 학습된 그래디언트 필드 $\text{gf}$가 되며, 이는 에이전트에게 유리한 방향(인력) 또는 불리한 방향(척력)의 벡터 정보를 제공한다.

### 3. GF 기반 상태 표현 및 적응 (Adaptation)

에이전트의 관측값 $O_E$는 더 이상 단순한 좌표값이 아니라, 학습된 여러 $\text{gf}$ 벡터들의 결합(Concatenation)으로 대체된다:
$$O_{GF} = \{ \text{gf}_1, \text{gf}_2, \dots, \text{gf}_n \}$$

새로운 환경 $E_2$로 전이할 때, 정책 $\pi_\phi$는 그대로 유지하고 입력되는 $\text{gf}$ 표현만 $E_2$에서 학습된 것으로 교체함으로써 적응을 수행한다.

### 4. 희소 보상 문제 해결을 위한 Credit Assignment

보상이 매우 희소한(Sparse) 환경에서는 탐색이 어렵다. 이를 해결하기 위해 인력 그래디언트 $\text{gf}^+$의 크기 $|\text{gf}^+|$를 보상 함수에서 차감하는 방식의 reward shaping을 적용한다:
$$R_E \leftarrow R_E - \lambda |\text{gf}^+_{E}|$$
이는 에이전트가 보상을 최대화하기 위해 $|\text{gf}^+|$를 최소화, 즉 타겟 분포(Attractive examples)에 더 가까워지도록 유도한다.

### 5. 네트워크 구조

- **Score Network ($\phi_i$)**: Graph Neural Network(GNN)를 사용하여 에이전트와 랜드마크 간의 복잡한 관계를 캡처한다.
- **Policy Network**: GNN을 통해 생성된 $\text{gf}$ 표현들을 concatenate 하여 Fully Connected(FC) 레이어에 입력하고 최종 액션을 생성한다.

## 📊 Results

### 1. 실험 설정

- **환경**: Particle-world 기반의 2D 연속 공간.
  - **Grassland**: 양(Sheep)은 풀을 먹고 늑대(Wolf)를 피해야 하며, 늑대는 양을 잡아야 함.
  - **Cooperative Navigation**: Vanilla, Color, Team 세 가지 난이도로 구성되며, 에이전트들이 협력하여 랜드마크를 점유해야 함.
- **비교 대상**: Original Sparse Reward (MAPPO), Reward Engineering (Human-shaped dense reward).
- **지표**: 정규화된 보상(Grassland), 최종 단계 성공률(Navigation).

### 2. 주요 결과

- **Grassland Game**: SocialGFs가 모든 에이전트 규모(Scale 2-2부터 8-8까지)에서 베이스라인을 압도하였다. 특히 $4-4$ 규모에서 학습된 모델을 다른 규모로 전이시킨 $\text{SocialGFs}^*$ 모델 역시 매우 높은 적응력을 보였다.
- **Cooperative Navigation**:
  - Original 및 Reward Engineering 방식은 가장 간단한 Vanilla Navigation의 소규모 케이스를 제외하고는 거의 성공하지 못했다.
  - **$\text{SocialGFs}^+$**(Credit assignment 적용 모델)는 모든 난이도의 내비게이션 작업에서 압도적인 성공률을 기록하였다.
  - $\text{SocialGFs}^*$ (Grassland에서 학습된 모델을 내비게이션으로 전이) 역시 성공적인 결과를 보여, GF 표현의 강력한 전이 가능성을 증명하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

1. **표현의 추상화와 일반화**: 단순 좌표가 아닌 '힘의 흐름'인 그래디언트 필드를 사용함으로써, 에이전트 수나 구체적인 환경 설정이 바뀌어도 변하지 않는 추상적인 사회적 규칙을 학습할 수 있었다.
2. **희소 보상 문제 해결**: $\text{gf}^+$를 이용한 보상 수정은 수동으로 설계된 reward shaping보다 훨씬 유연하며, 데이터 기반으로 타겟 상태로의 유도를 가능하게 하여 credit assignment 문제를 효과적으로 해결하였다.
3. **확장성**: GNN을 기반으로 GF를 생성하므로, 입력 엔티티의 수가 변하더라도 네트워크 구조를 변경할 필요 없이 대응 가능하다.

### 한계 및 향후 과제

- 새로운 환경으로 전이할 때, 여러 GF 중 어떤 것이 더 중요한지 우선순위를 정하는(Ranking) 메커니즘이 부족하다.
- 현재는 단순한 2D 입자 환경에서 검증되었으므로, UnrealCV와 같은 고해상도 3D 환경으로의 확장이 필요하다.

## 📌 TL;DR

본 논문은 멀티 에이전트 시스템의 적응성 문제를 해결하기 위해, 오프라인 데이터로부터 **Denoising Score Matching**을 통해 **Social Gradient Fields (SocialGFs)**를 학습하고 이를 상태 표현으로 사용하는 방법을 제안한다. 이를 통해 에이전트는 복잡한 환경에서도 효율적으로 정책을 학습할 수 있으며, 환경 변화 시 GF 표현만 교체함으로써 뛰어난 전이 성능(Transferability)과 확장성(Scalability)을 보인다. 특히 희소 보상 환경에서 데이터 기반의 가이드라인을 제공함으로써 기존 MARL의 한계를 극복하였다.
