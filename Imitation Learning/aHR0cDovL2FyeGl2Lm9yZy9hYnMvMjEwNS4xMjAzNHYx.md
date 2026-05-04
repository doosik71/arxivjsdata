# Hyperparameter Selection for Imitation Learning

Léonard Hussenot, Marcin Andrychowicz, Damien Vincent, Robert Dadashi, Anton Raichuk, Lukasz Stafiniak, Sertan Girgin, Raphael Marinier, Nikola Momchev, Sabela Ramos, Manu Orsini, Olivier Bachem, Matthieu Geist, Olivier Pietquin (2021)

## 🧩 Problem to Solve

본 논문은 연속 제어(continuous-control) 환경에서 모방 학습(Imitation Learning, IL) 알고리즘의 하이퍼파라미터(Hyperparameter, HP) 선정 문제를 다룬다.

일반적으로 모방 학습의 목적은 전문가의 시연(demonstrations)만을 이용하여 정책을 학습하는 것이며, 이는 보상 함수(reward function)를 설계하기 어렵거나 알 수 없는 상황을 전제로 한다. 그러나 기존의 많은 IL 연구들은 역설적이게도 하이퍼파라미터를 최적화하는 과정에서 환경의 외부 보상 함수를 사용하여 가장 성능이 좋은 모델을 선택해 왔다.

만약 보상 함수를 사용할 수 있는 상황이라면, 굳이 모방 학습을 사용할 필요 없이 강화학습(RL)을 통해 직접 정책을 학습시키면 된다. 따라서 외부 보상에 의존한 하이퍼파라미터 선정은 실제 IL 설정과 맞지 않으며, 이는 알고리즘의 실제 성능을 과대평가하게 만들고 실용성을 저하시키는 요인이 된다. 본 논문의 목표는 외부 보상 함수 없이도 하이퍼파라미터를 효과적으로 선정할 수 있는 대리 지표(proxy metrics)를 제안하고 그 효용성을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 외부 보상 신호 없이 IL 알고리즘을 튜닝하기 위한 체계적인 평가 프로토콜과 대리 지표를 제안한 점이다. 주요 기여 사항은 다음과 같다.

1. **보상 함수 없는 HP 선정 문제 제기**: IL 연구에서 관습적으로 사용해 온 보상 기반 HP 선정 방식의 논리적 모순과 실제 적용 시의 한계를 지적하였다.
2. **다양한 대리 지표(Proxy Metrics) 제안**: 전문가의 행동 및 상태 분포와 에이전트의 행동 및 상태 분포를 비교하는 다양한 지표를 제안하였다.
3. **대규모 실험적 검증**: 9개의 환경(OpenAI Gym, Adroit)에서 10,000개 이상의 에이전트를 학습시켜, 제안한 지표들이 실제 보상 함수와 얼마나 높은 상관관계를 가지며 HP 선정에 유용한지 분석하였다.
4. **HP 전이 가능성 분석**: 보상 함수가 알려진 유사 작업에서 최적화된 HP를 새로운 작업으로 전이(transfer)했을 때의 성능을 평가하였다.
5. **실무적 권장사항 제공**: IL 알고리즘의 성능을 극대화하기 위한 HP 선정 및 조기 종료(early stopping) 전략에 대한 가이드를 제시하였다.

## 📎 Related Works

기존의 강화학습(RL) 분야에서는 하이퍼파라미터 민감도가 매우 높으며, 이를 해결하기 위해 그리드 서치, 랜덤 서치 또는 인구 기반 전략(population-based strategies) 등이 사용되었다. 또한, 최근의 오프라인 강화학습(Offline RL) 연구에서도 환경과의 상호작용 없이 하이퍼파라미터를 선정해야 한다는 점이 강조된 바 있다.

모방 학습에서는 전문가의 행동을 직접 복제하는 Behavioral Cloning(BC)부터, 전문가와 에이전트의 상태-행동 분포 차이를 줄이는 Adversarial Imitation Learning(AIL), Primal Wasserstein Imitation Learning(PWIL) 등 다양한 접근 방식이 존재한다. 기존 연구들에서 분포 유사도 측정 지표들이 제안되긴 하였으나, 이를 하이퍼파라미터 선정이라는 구체적인 최적화 프로세스에 적용하여 체계적으로 분석한 연구는 본 논문이 처음이라고 명시하고 있다.

## 🛠️ Methodology

### 1. 하이퍼파라미터 선정을 위한 대리 지표 (Proxy Metrics)

논문에서는 외부 보상을 대체하여 모델의 성능을 측정할 수 있는 5가지 지표를 제안한다.

* **Action MSE**: 전문가의 상태에서 에이전트가 생성한 행동과 전문가의 행동 간의 평균 제곱 오차(Mean Squared Error)를 계산한다. 이는 오프라인에서 계산 가능하며, BC 알고리즘의 손실 함수와 직접적으로 연결된다.
* **State Distribution Divergence**: 전문가가 방문한 상태 분포와 에이전트가 생성한 궤적의 상태 분포 사이의 $\text{Wasserstein distance}$를 계산한다. 이때 계산 효율과 안정성을 위해 $\text{entropy-regularized Sinkhorn algorithm}$을 사용하며, 상태 좌표는 표준편차가 1이 되도록 정규화한다.
* **Random Network Distillation (RND)**: 고정된 랜덤 네트워크 $f$와 이를 예측하도록 학습된 네트워크 $\hat{f}$를 구성한다. 전문가 데이터로 $\hat{f}$를 학습시킨 후, 에이전트의 궤적에 대해 다음의 MSE를 측정하여 상태 분포의 유사도를 평가한다.
    $$\text{Metric} = \exp(-\|f(\text{obs}) - \hat{f}(\text{obs})\|^2)$$
* **Imitation Return**: IRL(역강화학습) 알고리즘이 학습한 보상 함수를 사용하여 계산한 누적 보상이다.
* **Environment Return (Oracle)**: 실제 환경의 보상 함수를 사용하여 계산한 값으로, 모든 지표의 성능을 비교하는 기준점(Gold Standard)이 된다.

### 2. 대상 알고리즘 및 실험 설정

* **알고리즘**:
  * **Behavioral Cloning (BC)**: 전문가 데이터를 지도 학습 방식으로 학습하는 오프라인 알고리즘.
  * **Adversarial Imitation Learning (AIL)**: 판별자(Discriminator)를 통해 전문가와 에이전트를 구분하고, 이를 속이도록 정책을 학습하는 방식.
  * **Primal Wasserstein Imitation Learning (PWIL)**: 전문가와 에이전트의 상태-행동 분포 간 $\text{Wasserstein distance}$의 상한선을 최소화하는 방식.
  * $\text{AIL}$과 $\text{PWIL}$의 정책 학습에는 $\text{Soft Actor-Critic (SAC)}$을 사용하였다.
* **환경**: OpenAI Gym의 locomotion 작업 5종과 Adroit의 manipulation 작업 4종을 사용하였다.

### 3. 실험 절차

1. 각 알고리즘과 환경 조합에 대해 100개의 HP 조합을 샘플링하여 학습시킨다.
2. 실무자가 25개의 조합을 무작위로 선택해 테스트한다고 가정하고, 제안한 각 대리 지표를 기준으로 최적의 HP를 선정한다.
3. 선정된 HP로 학습된 정책의 실제 환경 보상(Environment Return)을 측정하여 지표의 유효성을 평가한다.
4. 동일한 지표를 사용하여 학습 도중 최적의 시점에서 학습을 멈추는 조기 종료(Early Stopping)의 효과를 분석한다.

## 📊 Results

### 1. 대리 지표를 통한 HP 선정 성능

실험 결과, **State Divergence (상태 분포 발산)** 지표가 모든 알고리즘에서 가장 우수한 성능을 보였다. 이 지표를 통해 선정된 HP는 실제 보상 함수를 사용하여 선정한 경우와 유사한 수준의 환경 보상을 달성하였다. 반면 Action MSE는 오프라인 지표로서 시스템 역학(dynamics)을 반영하지 못해 성능이 낮았으며, 특히 Adroit 환경에서 그 격차가 컸다.

### 2. 조기 종료(Early Stopping)의 중요성

신뢰할 수 있는 지표(특히 State Divergence)를 사용하여 조기 종료를 수행했을 때, 거의 모든 경우에서 최종 성능이 향상되었다. 이는 IL 알고리즘이 학습 과정에서 오버피팅되거나 성능이 하락하는 구간이 존재함을 시사하며, 적절한 대리 지표를 통한 모델 선택이 필수적임을 보여준다.

### 3. 알고리즘 선택으로의 확장

특정 지표를 기준으로 어떤 알고리즘(BC, AIL, PWIL 중 하나)을 사용할지 결정했을 때, State Divergence는 실제로 가장 성능이 좋은 알고리즘을 정확하게 선택하는 경향을 보였다. 반면 Action MSE는 항상 BC를 선택하는 경향이 있었는데, 이는 BC가 Action MSE 자체를 최적화하는 알고리즘이기 때문이다.

### 4. 하이퍼파라미터 전이(Transfer)

보상 함수가 알려진 유사 환경에서 최적의 HP를 찾아 전이하는 방식은 어느 정도 효과가 있었으나, 타겟 환경에서 직접 State Divergence를 통해 HP를 선정하는 것보다 성능이 낮았다. 이는 각 환경마다 최적의 HP가 다르며 알고리즘의 확률적 특성이 강하기 때문이다. 다만, BC는 구조가 단순하여 AIL이나 PWIL보다 HP 전이 성능이 더 좋게 나타났다.

## 🧠 Insights & Discussion

본 논문은 IL 연구 커뮤니티가 간과해 온 '보상 함수의 가용성' 문제를 정면으로 다루었다. 실험 결과는 다음과 같은 중요한 통찰을 제공한다.

첫째, 기존 연구들이 보상 함수를 이용해 HP를 튜닝함으로써 IL 알고리즘의 실제 성능을 심각하게 과대평가했을 가능성이 높다. 특히 AIL과 같은 알고리즘은 보상 함수 기반 튜닝과 대리 지표 기반 튜닝 사이의 성능 격차가 매우 컸다.

둘째, 상태 분포의 유사도를 측정하는 $\text{Wasserstein distance}$ 기반의 지표가 보상 함수 없이도 정책의 질을 평가하는 매우 강력한 도구가 될 수 있음을 입증하였다.

셋째, IL 알고리즘은 하이퍼파라미터에 매우 민감하며, 단순히 "최적의 HP"를 찾는 것뿐만 아니라 "어떤 HP가 다양한 환경에서 강건하게 작동하는가"와 "어떻게 효율적으로 튜닝할 것인가"에 대한 논의가 필요하다.

본 연구의 한계로는 모든 환경이 완전히 관측 가능한(fully observable) 상태였다는 점이 있으며, 시각적 입력(visual-based inputs)이나 부분 관측 가능성(partial observability)이 있는 더 복잡한 환경에서의 지표 유효성은 추가적인 연구가 필요하다.

## 📌 TL;DR

이 논문은 모방 학습(IL)에서 외부 보상 함수 없이 하이퍼파라미터를 선정할 수 있는 대리 지표들을 제안하고 검증하였다. 실험을 통해 **상태 분포의 발산(State Divergence)**을 측정하는 지표가 보상 함수를 대체하여 최적의 하이퍼파라미터를 선정하고 조기 종료 시점을 결정하는 데 가장 효과적임을 밝혔다. 이 연구는 IL 알고리즘의 평가 프로토콜을 현실적으로 재정의함으로써, 향후 더 실용적이고 강건한 모방 학습 알고리즘 개발의 토대를 마련하였다.
