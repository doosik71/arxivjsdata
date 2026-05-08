# Quantifying the effects of environment and population diversity in multi-agent reinforcement learning

Kevin R. McKee, Joel Z. Leibo, Charlie Beattie, Richard Everett (2022)

## 🧩 Problem to Solve

본 논문은 Multi-Agent Reinforcement Learning (MARL)에서 발생하는 **일반화(Generalization)** 문제, 특히 에이전트가 학습 과정에서 경험하지 못한 새로운 환경(novel environments)과 새로운 협력자 또는 경쟁자(new co-players)와 상호작용할 때 성능이 급격히 저하되는 현상을 해결하고자 한다.

강화학습 에이전트는 일반적으로 단일 레벨에서 학습하고 테스트되는 경향이 있으며, 이는 학습 데이터에 과적합(overfitting)되어 환경의 상태와 특정 행동 간의 매핑을 단순히 암기하는 결과를 초래한다. MARL 환경에서는 이러한 환경적 과적합뿐만 아니라, 함께 학습한 특정 co-player들의 행동 패턴에 과적합되는 'co-player overfitting' 문제가 추가로 발생한다. 따라서 본 연구의 목표는 환경의 다양성(Environment Diversity)과 인구 집단의 다양성(Population Diversity)이 에이전트의 일반화 성능과 전체적인 성능에 어떠한 정량적 영향을 미치는지 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **환경 다양성과 일반화의 관계 규명**: 절차적 생성(Procedural Generation)을 통해 학습 레벨의 수를 늘리면 새로운 레벨에 대한 일반화 성능이 유의미하게 향상되지만, 일부 환경에서는 학습 세트 자체의 성능이 감소하는 트레이드오프가 발생함을 정량적으로 분석하였다.
2. **Expected Action Variation (EAV) 지표 제안**: 특정 환경에 종속되지 않고 인구 집단의 행동 다양성을 측정할 수 있는 새로운 도메인 불가지론적(domain-agnostic) 지표인 EAV를 도입하였다.
3. **인구 다양성의 영향 분석**: 인구 집단의 크기(Population Size)와 내재적 동기 부여(Intrinsic Motivation)가 행동 다양성을 증가시키며, 이러한 다양성이 특정 환경(예: Overcooked, Capture the Flag)에서는 성능 및 일반화 능력을 향상시킨다는 점을 확인하였다.

## 📎 Related Works

기존의 단일 에이전트 강화학습 연구에서는 절차적 생성을 통해 학습 환경의 다양성을 높임으로써 과적합을 방지하고 정책의 일반성을 높이려는 시도가 있었다. MARL 분야에서도 Population-based training, Policy ensembles, League training 등을 통해 co-player의 다양성을 확보하여 일반화 성능을 높이려는 연구들이 진행되었다.

그러나 기존 연구들은 주로 두 명의 플레이어가 참여하는 제로섬 게임(zero-sum games)에 집중되어 있었으며, 환경적 변동성과 인구 다양성이 동시에 MARL 성능에 미치는 영향을 엄격하게 정량적으로 분석한 사례는 부족했다. 본 논문은 다양한 성격의 4가지 마르코프 게임(Markov games)을 통해 이 관계를 구체적으로 조사함으로써 기존 연구의 공백을 메우고자 한다.

## 🛠️ Methodology

### 1. 시스템 구조 및 학습 알고리즘

본 연구는 분산 비동기 프레임워크를 사용하여 여러 '아레나(arenas)'에서 에이전트 인구 집단을 학습시킨다.

- **학습 알고리즘**: On-policy 변형인 V-MPO (Maximum a Posteriori Policy Optimization)를 사용한다.
- **신경망 구조**: 시각적 관측값은 3개 섹션의 ResNet을 통해 처리되며, 이후 이전 행동과 보상 값이 결합되어 LSTM(256 units)과 MLP를 거쳐 최종 행동 분포를 생성한다.
- **정규화**: PopArt normalization을 적용하여 성능을 개선하였다.
- **내재적 동기**: 일부 실험에서는 사회적 가치 지향성(Social Value Orientation, SVO) 컴포넌트를 추가하여 보상 분포에 대한 내재적 동기를 부여하였다.

### 2. 실험 환경 (4가지 Markov Games)

에이전트의 일반화 능력을 평가하기 위해 다음과 같은 서로 다른 성격의 환경을 구축하였다.

- **HarvestPatch**: 혼합 동기(mixed-motive) 게임. 사과를 채집하되, 과도한 채집은 재생성률을 낮추는 사회적 딜레마 상황을 제공한다.
- **Traffic Navigation**: 협력(coordination) 게임. 충돌을 피하며 각자의 목표 지점에 빠르게 도달해야 한다.
- **Overcooked**: 공동 보상(common-payoff) 게임. 두 플레이어가 협력하여 토마토 수프를 만들어 배달해야 한다.
- **Capture the Flag**: 경쟁(competitive) 게임. 상대 팀의 깃발을 탈취하여 자신의 기지로 가져와야 한다.

### 3. 환경 다양성 조사 방법

학습 레벨의 수 $L \in \{1, 10, 100, 1000, 10000\}$를 변화시키며 학습시킨 후, 학습에 사용되지 않은 100개의 테스트 레벨(held-out levels)에서 성능을 측정한다. 이때 **Generalization Gap**을 다음과 같이 정의하여 분석한다.
$$\text{Generalization Gap} = | \text{Performance on Test Set} - \text{Performance on Training Set} |$$

### 4. 인구 다양성 측정: Expected Action Variation (EAV)

본 논문은 행동 다양성을 측정하기 위해 EAV라는 지표를 제안한다. EAV는 인구 집단에서 무작위로 추출된 두 에이전트가 동일한 상태 $s$에서 서로 다른 행동을 선택할 확률을 나타낸다.

- **계산 절차**:
    1. 대표 상태 풀(pool) $S$를 생성한다.
    2. 각 에이전트 $\pi_A$에 대해 상태 $s \in S$에서의 행동 확률 분포 $\text{policy\_dists}_{A, (s,l)}$를 근사한다.
    3. 모든 에이전트 쌍 $(A_1, A_2)$에 대해 행동 분포 간의 **Total Variation Distance (TVD)**를 계산한다.
    $$ \text{tvd} = \sum_{a} | \pi_{A_1}(a|s) - \pi_{A_2}(a|s) | $$
    4. 모든 상태와 에이전트 쌍에 대해 평균을 내어 0에서 1 사이의 값으로 정규화한다.

## 📊 Results

### 1. 환경 다양성의 효과

- **일반화 성능**: 모든 환경에서 학습 레벨 수 $L$이 증가함에 따라 테스트 세트에서의 성능이 향상되었으며, Generalization Gap은 현저히 감소하였다. 특히 $L=1000$ 부근에서 Gap이 거의 0에 수렴하였다.
- **학습 세트 성능 저하**: 흥미롭게도 $L$이 매우 커질 때, 일부 환경(Overcooked, Capture the Flag)에서는 학습 세트 내에서의 성능이 오히려 감소하는 경향이 발견되었다. 이는 일반화 능력을 얻는 대신 특정 레벨에 최적화되는 능력이 희생됨을 시사한다.
- **Cross-Play 결과**: 환경마다 양상이 달랐다. Traffic Navigation과 Overcooked에서는 다양하게 학습된 에이전트와 함께 플레이할 때 성능이 일관되게 향상되었다.

### 2. 인구 다양성의 효과

- **EAV와 인구 크기**: 인구 집단의 크기 $N$이 증가할수록 모든 환경에서 EAV 값이 증가하였다. 즉, 별도의 최적화 없이 인구 수만 늘려도 행동 다양성이 확보된다.
- **내재적 동기의 영향**: SVO를 통해 이질적인(heterogeneous) 내재적 동기를 부여한 집단이 동일한 동기나 동기가 없는 집단보다 유의미하게 높은 EAV를 보였다.
- **성능에 미치는 영향**:
  - HarvestPatch와 Traffic Navigation에서는 인구 다양성이 보상에 유의미한 영향을 주지 않았다.
  - **Overcooked와 Capture the Flag**에서는 인구 다양성이 증가함에 따라 에이전트의 성능이 크게 향상되었으며, 특히 $N=1$에서 $N=2$로 증가할 때 가장 극적인 성능 향상이 나타났다.

## 🧠 Insights & Discussion

본 연구는 MARL에서 환경 및 인구 다양성이 일반화에 기여하는 바를 정량적으로 입증하였다. 특히, 적은 양의 환경 다양성만으로도 일반화 능력을 크게 개선할 수 있음을 보여주었다.

**비판적 해석 및 논의사항**:

- **행동 다양성 vs 전략 다양성**: 제안된 EAV 지표는 '행동 수준(action-level)'의 다양성을 측정한다. 하지만 서로 다른 행동이 동일한 전략적 결과(strategic outcome)를 낳을 수 있으므로, 향후 연구에서는 실제 '전략 세트'의 변동성을 직접 측정하는 방법이 필요하다.
- **환경 다양성의 정의**: 본 논문은 '유니크한 레벨의 수'를 다양성의 척도로 사용했으나, 모든 레벨이 에이전트에게 유의미한 차이를 주는 것은 아니다. 어떤 특성이 실질적인 다양성을 만드는지에 대한 심층 분석이 요구된다.
- **실제 적용 가능성**: 인간은 행동 변동성이 매우 크기 때문에, 본 연구에서 제시한 인구 다양성 확보 방법은 인간-AI 협동(human-AI cooperation) 시스템에서 AI가 인간의 다양한 행동 패턴에 적응하도록 만드는 데 중요한 기초가 될 수 있다.

## 📌 TL;DR

본 논문은 MARL에서 **환경 다양성(절차적 생성)**과 **인구 다양성(에이전트 수 및 내재적 동기)**이 일반화 성능에 미치는 영향을 분석하였다. 연구 결과, 환경 다양성은 테스트 세트의 일반화 성능을 높이지만 학습 세트 성능을 일부 희생시키며, 인구 다양성은 환경에 따라(특히 협력/경쟁이 강한 게임에서) 성능을 유의미하게 향상시킨다. 또한, 행동 다양성을 정량화하기 위한 새로운 지표인 **Expected Action Variation (EAV)**를 제안하여 인구 규모와 다양성 간의 상관관계를 입증하였다. 이 연구는 향후 인간-AI 협업 모델의 강건성을 높이는 데 기여할 가능성이 크다.
