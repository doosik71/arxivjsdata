# Genetic Imitation Learning by Reward Extrapolation

Boyuan Zheng, Jianlong Zhou and Fang Chen (2023)

## 🧩 Problem to Solve

본 논문은 Imitation Learning(모방 학습)에서 발생하는 **Suboptimal Demonstration(최적이 아닌 시연 데이터)**의 의존성 문제를 해결하고자 한다. 기존의 모방 학습은 주로 입력 데이터가 최적이거나 최적에 가깝다는 가정하에 설계되었으며, 이 경우 에이전트의 성능은 시연자의 수준이라는 상한선(Upper bound)에 갇히게 된다. 특히 최적의 시연 데이터를 수집하는 것은 비용이 많이 들고 비현실적인 경우가 많으며, 실제로는 최적이 아닌 데이터가 훨씬 더 풍부하게 존재한다.

이러한 문제를 해결하기 위해 기존 연구들은 Reward Extrapolation(보상 외삽) 방식을 통해 시연자보다 더 나은 성능을 내는 Better-than-demonstrator behavior를 구현하려 하였다. 그러나 T-REX와 같은 기존의 외삽 방법론들은 전문가가 직접 데이터에 순위를 매기는(Manual ranking) 노동 집약적인 과정이 필요하거나, D-REX처럼 노이즈를 주입하는 방식이 항상 유효하지 않다는 한계가 있다. 따라서 본 논문의 목표는 사람이 직접 개입하여 순위를 매기지 않고도, 효율적으로 보상 함수를 추론하여 최적이 아닌 데이터로부터 최적의 정책을 유도하는 GenIL(Genetic Imitation Learning) 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Genetic Algorithm(유전 알고리즘, GA)**을 Imitation Learning에 결합하여, 데이터 효율성을 높이고 보상 함수 추론을 위한 학습 데이터를 자동으로 생성하는 것이다.

주요 기여 사항은 다음과 같다.

1. **자동화된 데이터 랭킹 생성**: 유전 알고리즘의 Crossover(교차)와 Mutation(변이) 연산을 통해 기존의 suboptimal trajectory들을 조합하여 다양한 수준의 품질을 가진 "Fake" trajectory들을 생성한다. 이를 통해 전문가의 수동 랭킹 없이도 학습에 필요한 랭킹 데이터를 확보한다.
2. **데이터 효율성 극대화**: 단 두 개의 trajectory(하나의 좋은 데이터와 하나의 좋지 않은 데이터)만으로도 모델을 학습시킬 수 있는 구조를 제안한다.
3. **안정적인 보상 외삽**: 생성된 가짜 데이터셋을 통해 보상 함수의 파라미터를 더 정밀하고 콤팩트하게 추정함으로써, 보지 못한(unseen) trajectory에 대해서도 정확한 보상을 예측하고 이를 통해 높은 성능의 정책을 도출한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개하며 GenIL의 차별점을 설명한다.

- **Imitation Learning 및 Suboptimal Data**: 전통적인 Behavior Cloning(BC)이나 Inverse Reinforcement Learning(IRL)은 최적의 데이터를 가정한다. GAIL과 같은 적대적 학습 방식이나 HER(Hindsight Experience Replay)을 결합한 HGAIL, GoalGAIL 등은 suboptimal 데이터 문제를 해결하려 했으나, 구조적 취약성이나 환경 상호작용의 필요성 등의 한계가 있다.
- **Imitation from Observation (IfO)**: 액션 라벨(Action labels) 없이 오직 관찰 값(Observations)만을 이용해 학습하는 방식으로, 데이터 수집의 제약을 완화한다. 본 논문은 이 IfO 설정을 채택하여 액션 정보 없이 학습을 진행한다.
- **Reward Extrapolation**: T-REX는 수동 랭킹을 통해, D-REX는 노이즈 주입을 통해 시연자 이상의 성능을 내고자 했다. 하지만 수동 랭킹은 비용이 많이 들고, 노이즈 주입은 항상 효과적인 것은 아니라는 한계가 있다.
- **Genetic Algorithm (GA)**: 진화 연산(교차, 변이)을 통해 최적의 솔루션을 찾는 알고리즘이다. 기존에는 주로 하이퍼파라미터 최적화나 데이터 확장 등에 사용되었으나, 본 논문은 이를 **보상 추론을 위한 랭킹 데이터 생성**이라는 새로운 관점으로 적용하였다.

## 🛠️ Methodology

GenIL은 크게 유전 알고리즘(GA)을 통한 데이터 생성 단계와 Inverse Reinforcement Learning(IRL)을 통한 보상 추론 단계로 구성된다.

### 1. Genetic Algorithm을 통한 랭킹 데이터 생성

두 개의 서로 다른 성능을 가진 trajectory $\mathcal{D}_{original} = \{\tau_{good}, \tau_{bad}\}$를 입력으로 받는다. GA는 다음과 같은 과정을 통해 $\mathcal{D}_{fake}$를 생성한다.

- **Crossover (교차)**: 두 trajectory에서 무작위 구간(크기 10 미만)을 선택하여 서로 교체한다. 이를 통해 $\tau_{good}$의 비율이 높은 데이터부터 $\tau_{bad}$의 비율이 높은 데이터까지 다양한 수준의 가짜 trajectory가 만들어진다.
- **Mutation (변이)**: 방문했던 모든 상태들의 집합에서 무작위로 상태를 선택하여 삽입하거나 변경하며 무작위 랭크 값을 부여한다.
- **Selection (선택)**: 생성된 offspring의 평균 랭크가 미리 정의된 구간 내에 있을 경우에만 최종 랭킹 데이터셋 $\mathcal{D}_{ranked}$에 추가한다.

최종적으로 학습에 사용되는 데이터셋은 $\mathcal{D}_{ranked} = \mathcal{D}_{original} + \mathcal{D}_{fake}$가 된다.

### 2. Reward Inference (보상 추론)

생성된 랭킹 데이터셋을 이용하여 보상 함수 $R_\theta$를 학습한다.

- **아키텍처**: Atari 도메인에서는 4층의 Convolutional Neural Network(CNN)를 통해 특징을 추출하고, 이후 Multi-Layer Perceptron(MLP)을 통해 최종 보상 값을 출력한다. MuJoCo 도메인에서는 상태 정보를 직접 MLP로 입력한다.
- **손실 함수**: Plackett-Luce 모델에 기반한 Pairwise Ranking Loss를 사용한다. 두 trajectory $\tau_i$와 $\tau_j$에 대해 $\tau_j$의 랭크가 더 높을 때, 다음과 같은 손실 함수를 최소화한다.
$$L(\theta) \approx -\sum_{\tau_i, \tau_j} \log \frac{\exp \sum_{s \in \tau_j} R_\theta(s)}{\exp \sum_{s \in \tau_i} R_\theta(s) + \exp \sum_{s \in \tau_j} R_\theta(s)}$$
여기서 $\sum R'(\tau_j) > \sum R'(\tau_i)$ 관계가 성립해야 한다.

### 3. 정책 도출

학습된 보상 함수 $R_\theta$를 환경의 보상으로 사용하여 PPO(Proximal Policy Optimization)와 같은 강화학습 알고리즘을 통해 최종 정책 $\pi_\theta$를 학습한다.

## 📊 Results

### 실험 설정

- **환경**: MuJoCo(Hopper, HalfCheetah), Atari(Breakout, Beamrider, SpaceInvaders)
- **비교 대상**: BC, T-REX, D-REX
- **평가 지표**:
    1. **Extrapolation Accuracy**: 예측된 보상과 실제 보상의 비율 $\text{average}(\frac{R_\theta(\tau_t)}{R^*(\tau_t)})$를 통해 측정.
    2. **Overall Policy Performance**: 시뮬레이션에서 얻은 실제 Return 값으로 측정.

### 정량적 결과

Table 1의 결과에 따르면, GenIL은 모든 테스트 태스크에서 기존 방법론보다 우수한 성능을 보였다.

- **HalfCheetah**: GenIL이 다른 모든 방법론을 큰 차이로 압도하였으며, 특히 시연 데이터의 수준을 뛰어넘는 성능을 보였다.
- **Atari 태스크**: BC와 D-REX는 suboptimal 데이터의 영향으로 인해 낮은 성능을 보인 반면, GenIL과 T-REX는 상대적으로 높은 성능을 유지하였다.
- **안정성**: D-REX가 표준 편차는 가장 낮아 안정적이었으나, 절대적인 성능은 GenIL이 가장 높았다.

### 정성적 분석 (Extrapolation)

그림 3의 외삽 비교 그래프에서, GenIL은 동일한 실제 보상(Ground-truth reward)을 가진 unseen trajectory들에 대해 예측값의 수직 편차가 가장 작았다. 이는 GenIL이 생성한 가짜 데이터들이 보상 함수를 더 정밀하고 콤팩트하게 학습시키는 데 기여했음을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **데이터 효율성**: 단 2개의 trajectory만으로도 GA를 통해 풍부한 랭킹 데이터를 생성함으로써, 인간 전문가의 개입 없이도 효과적인 보상 외삽이 가능함을 입증하였다.
- **Crossover Step Size의 영향**: 교차 구간의 크기가 너무 크면 랭크 간의 차이가 모호해져 성능의 일관성이 떨어지며, 너무 작으면 동적 전이 일관성(Dynamic transition consistency)이 깨져 trajectory가 파편화된다. 이를 통해 적절한 하이퍼파라미터 설정의 중요성을 확인하였다.

### 한계점 및 향후 과제

- **하이퍼파라미터 의존성**: GA의 돌연변이율, 교차율, offspring 수 등이 Trial and Error 방식으로 설정되었다. 이를 동적으로 최적화하는 메커니즘이 필요하다.
- **단순한 랭킹 부여 방식**: 현재는 trajectory 전체에 동일한 랭크를 부여하고 있으나, 향후 상태 방문 빈도나 액션 빈도 등의 통계량을 활용하여 세그먼트별로 가중치를 두는 방식이 제안되었다.
- **파편화 문제**: GA 연산이 과도할 경우 trajectory가 너무 무작위하게 변해 학습에 방해가 될 수 있는 균형점(Generalization vs Fragmentation)을 찾는 연구가 필요하다.

## 📌 TL;DR

본 논문은 유전 알고리즘(GA)의 교차 및 변이 연산을 활용해 최적이 아닌 시연 데이터로부터 자동으로 랭킹이 매겨진 가짜 데이터를 생성하고, 이를 통해 정밀한 보상 함수를 추론하는 **GenIL** 방법론을 제안한다. 이를 통해 사람이 직접 순위를 매겨야 했던 기존 Reward Extrapolation의 번거로움을 제거하였으며, 매우 적은 양의 suboptimal 데이터만으로도 시연자보다 뛰어난 성능의 정책을 학습시킬 수 있음을 보였다. 이 연구는 특히 전문가 데이터 수집이 어려운 로보틱스나 복잡한 제어 분야에서 데이터 효율적인 모방 학습을 가능하게 하는 중요한 방향성을 제시한다.
