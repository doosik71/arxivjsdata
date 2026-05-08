# Meta-SGD: Learning to Learn Quickly for Few-Shot Learning

Zhenguo Li, Fengwei Zhou, Fei Chen, Hang Li

## 🧩 Problem to Solve

기존 딥러닝 모델들은 개별 태스크를 고립된 방식으로, 그리고 방대한 양의 데이터에 의존하여 처음부터 학습하기 때문에 소수 데이터 학습(Few-Shot Learning) 환경에서는 효율성과 정확성 면에서 큰 어려움을 겪습니다. 사람처럼 적은 수의 예시만으로도 빠르게 학습하고 적응하는 능력을 기계에 부여하는 것이 목표이며, 이를 위해 데이터 효율성과 빠른 적응력을 갖춘 원칙적인(principled) 학습 방법론이 필요합니다. 메타 학습(Meta-learning)은 이러한 문제를 해결하기 위한 유망한 접근 방식이지만, 높은 용량을 가지면서도 효과적으로 학습 가능한 메타 학습기(meta-learner)를 설계하는 것이 핵심 과제입니다.

## ✨ Key Contributions

- **새로운 메타 학습기 Meta-SGD 개발**: SGD와 유사하며 쉽게 훈련될 수 있는 메타 학습기를 제안합니다.
- **높은 학습 유연성 및 용량**: 학습기의 초기화($\theta$), 업데이트 방향, 그리고 학습률($\alpha$)을 단일 메타 학습 과정에서 모두 학습하여 MAML(Model-Agnostic Meta-Learning)보다 훨씬 높은 용량을 가집니다.
- **단 한 번의 적응 단계(One-step Adaptation)**: 단 한 번의 업데이트를 통해 어떠한 미분 가능한(differentiable) 학습기도 초기화하고 적응시킬 수 있습니다.
- **광범위한 적용성**: 지도 학습(회귀, 분류) 및 강화 학습 분야 모두에 적용 가능합니다.
- **효율성**: 개념적으로 Meta-LSTM보다 간단하고 구현하기 쉬우며, 더욱 효율적으로 학습될 수 있습니다.
- **경쟁력 있는 성능**: 회귀, 분류, 강화 학습의 소수 데이터 학습 태스크에서 기존 최첨단(state-of-the-art) 방법론들을 능가하는 경쟁력 있는 성능을 달성합니다.

## 📎 Related Works

- **생성 모델(Generative Models)**: [14]는 확률적 프로그램(probabilistic programs)을 사용하여 필기체 문자 개념을 표현하고, 관련된 개념에 대한 지식을 활용하여 새로운 개념을 소수의 예시로 학습하는 방법을 보여주었습니다.
- **메트릭 기반 메타 학습기(Metric-based Meta-learners)**: [25]는 k-최근접 이웃(k-NN) 분류기와 같은 비모수적(non-parametric) 학습기를 위한 메타 학습기로 메트릭(metric)을 제안했으며, 훈련 및 테스트 조건을 일치시키는 방식을 활용했습니다.
- **RNN 기반 옵티마이저(RNN-based Optimizers)**: 초기 연구 [5, 29]는 RNN이 적응형 최적화 알고리즘을 모델링할 수 있음을 보여주었으며, 특히 LSTM [10]이 메타 학습기로서 좋은 성능을 보였습니다.
  - **Meta-LSTM** [2, 18]: LSTM을 SGD와 유사한 범용 옵티마이저로 사용하여 모델 업데이트 과정을 모방하고, 초기화와 업데이트 전략을 공동으로 학습했습니다. 그러나 구현 및 훈련 복잡도가 높고 각 매개변수가 독립적으로 업데이트되는 한계가 있었습니다.
- **MAML (Model-Agnostic Meta-Learning)** [7]: SGD를 메타 학습기로 사용하지만, 학습기(learner)의 초기화($\theta$)만 메타 학습을 통해 학습하는 방식입니다. 단순하지만 실용적이며 효과적입니다.

## 🛠️ Methodology

Meta-SGD는 학습기의 초기화, 업데이트 방향, 학습률을 모두 학습하는 새로운 옵티마이저 형태의 메타 학습기입니다.

1. **메타 학습기 정의**:
   - 기존 SGD의 업데이트 식 $\theta^t = \theta^{t-1} - \alpha \nabla \mathcal{L}_\mathcal{T}(\theta^{t-1})$와 달리, Meta-SGD는 한 번의 업데이트로 학습기를 적응시킵니다:
     $$ \theta' = \theta - \alpha \circ \nabla \mathcal{L}_\mathcal{T}(\theta) \tag{2}$$
     여기서,
     - $\theta$: 메타 학습을 통해 학습되는 학습기의 초기 매개변수입니다. 새로운 태스크에 대한 학습기를 초기화하는 데 사용됩니다.
     - $\alpha$: $\theta$와 동일한 차원의 벡터로, 메타 학습을 통해 학습되는 매개변수입니다. 이 $\alpha$는 업데이트의 **방향**과 **학습률**을 동시에 결정합니다.
     - $\circ$: 원소별 곱셈(element-wise product)을 나타냅니다.
   - $\alpha \circ \nabla \mathcal{L}_\mathcal{T}(\theta)$ 항은 단순히 기울기 $\nabla \mathcal{L}_\mathcal{T}(\theta)$를 따르지 않고, 학습된 $\alpha$에 의해 조정된 업데이트 방향과 암시적인 학습률을 갖게 됩니다.
2. **메타 훈련 목표**:
   - 메타 학습기의 목표는 주어진 태스크 분포 $p(\mathcal{T})$에서 학습기가 새로운 태스크에 대해 가장 잘 일반화되도록 초기 매개변수 $\theta$와 학습률/방향 매개변수 $\alpha$를 학습하는 것입니다.
   - **지도 학습 (Supervised Learning)**:
     $$ \min_{\theta, \alpha} E_{\mathcal{T} \sim p(\mathcal{T})} [\mathcal{L}_{test(\mathcal{T})}(\theta')] = E_{\mathcal{T} \sim p(\mathcal{T})} [\mathcal{L}_{test(\mathcal{T})}(\theta - \alpha \circ \nabla \mathcal{L}_{train(\mathcal{T})}(\theta))] \tag{3} $$
   - **강화 학습 (Reinforcement Learning)**: 태스크 $\mathcal{T}$를 MDP(Markov Decision Process)로 간주하며, 손실 $\mathcal{L}_\mathcal{T}(\theta)$는 음의 기대 할인 보상(negative expected discounted reward)입니다.
     $$ \min_{\theta, \alpha} E_{\mathcal{T} \sim p(T)} [\mathcal{L}_\mathcal{T}(\theta')] = E_{\mathcal{T} \sim p(\mathcal{T})} [\mathcal{L}_\mathcal{T}(\theta - \alpha \circ \nabla \mathcal{L}_\mathcal{T}(\theta))] \tag{5} $$
3. **최적화**:
   - 위의 목표 함수는 $\theta$와 $\alpha$에 대해 미분 가능하므로, 바깥 루프(outer loop)에서 SGD를 사용하여 $\theta$와 $\alpha$를 업데이트합니다 (알고리즘 1, 2 참조).
   - **알고리즘 1 (지도 학습)**: 각 배치 태스크 $\mathcal{T}_i$에 대해 $\theta$를 사용하여 $\nabla \mathcal{L}_{train(\mathcal{T}_i)}(\theta)$를 계산하고, 이를 통해 $\theta'_i$를 얻은 후, $\mathcal{L}_{test(\mathcal{T}_i)}(\theta'_i)$를 기준으로 $\theta, \alpha$를 업데이트합니다.
   - **알고리즘 2 (강화 학습)**: 정책 $f_\theta$를 사용하여 N1개의 궤적(trajectories)을 샘플링하고 정책 기울기(policy gradient) $\nabla \mathcal{L}_\mathcal{T}(\theta)$를 계산합니다. $\theta'_i$를 얻은 후, N2개의 궤적을 샘플링하여 일반화 손실을 계산하고, 이 손실의 기울기를 사용하여 $\theta, \alpha$를 업데이트합니다.

## 📊 Results

- **회귀 (K-shot Regression)**:
  - 사인(sine) 곡선 피팅 태스크에서 Meta-SGD는 MAML보다 모든 K-shot 설정(5, 10, 20샷)에서 **일관적으로 더 낮은 MSE(Mean Squared Error)**를 달성하며 크게 우수했습니다.
  - 이는 Meta-SGD가 초기화, 업데이트 방향, 학습률을 동시에 학습하여 MAML보다 높은 용량을 가짐을 시사합니다.
  - 특히 5-샷 회귀에서 MAML보다 훨씬 빠르게 사인 곡선 형태에 적응하는 능력을 보였습니다.
- **분류 (Few-shot Classification)**:
  - **Omniglot 데이터셋**: 5-way 및 20-way 분류(1-샷, 5-샷) 모든 태스크에서 Siamese Nets, Matching Nets, Meta-LSTM, MAML 등 기존 최첨단 모델들보다 **약간 더 높은 분류 정확도**를 달성했습니다.
  - **MiniImagenet 데이터셋**: 모든 경우(5-way/20-way, 1-샷/5-샷)에서 다른 모든 모델을 **능가하는 가장 높은 정확도**를 보였습니다.
  - Meta-SGD는 단 한 번의 업데이트로 높은 정확도를 달성하여 모델 훈련 및 새로운 태스크 적응 속도 면에서도 우수함을 입증했습니다.
- **강화 학습 (2D Navigation)**:
  - 고정된 시작 위치/랜덤 목표 위치, 랜덤 시작 위치/랜덤 목표 위치 두 가지 2D 네비게이션 태스크 모두에서 MAML보다 **상대적으로 높은 보상(낮은 음수값)**을 달성했습니다.
  - Meta-SGD에 의해 업데이트된 정책은 목표에 대한 더 강력한 인식을 보여주며, 학습된 최적화 전략이 기존 기울기 하강법보다 우수함을 나타냅니다.

## 🧠 Insights & Discussion

- **높은 용량의 장점**: Meta-SGD는 학습기의 초기화, 업데이트 방향, 학습률이라는 옵티마이저의 모든 핵심 요소를 엔드투엔드 방식으로 메타 학습함으로써, 기존에 초기화만 학습하는 MAML이나 독립적인 파라미터 업데이트에 의존하는 Meta-LSTM보다 훨씬 높은 용량을 가집니다.
- **빠른 적응력**: 단 한 번의 적응 단계만으로도 회귀, 분류, 강화 학습의 다양한 소수 데이터 학습 태스크에서 최첨단 성능을 달성하는 놀라운 결과를 보여주었습니다. 이는 소수 데이터 환경에서 중요한 빠른 적응 능력의 우수성을 입증합니다.
- **학습된 최적화 전략의 우수성**: Meta-SGD가 학습한 최적화 전략은 태스크 구조를 효과적으로 포착하여, 충분한 양의 데이터가 주어지는 경우에도 기존의 수동으로 설계된 기울기 하강법보다 더 나은 성능을 보였습니다.
- **한계점 및 미래 연구**:
  - **대규모 메타 학습**: 메타 학습기는 다수의 학습기를 훈련시키므로, 기존 학습 방식보다 훨씬 많은 계산 자원을 요구합니다. 특히 태스크당 데이터 양이 '소수'를 넘어설 경우 큰 학습기가 필요해져 계산 복잡도가 더욱 증가할 수 있습니다.
  - **메타 학습기의 다용성/일반화 능력**: 새로운 문제 설정이나 새로운 태스크 도메인, 심지어 다중 태스크 메타 학습기와 같은 예상치 못한 상황에 대처하는 능력 향상이 중요합니다. 이러한 문제 해결은 메타 학습의 실용적 가치를 크게 확장할 것입니다.

## 📌 TL;DR

Meta-SGD는 소수 데이터 학습(Few-Shot Learning)을 위한 새로운 메타 학습기로, 학습기(learner)의 초기화, 업데이트 방향, 학습률까지 모두 엔드투엔드(end-to-end) 방식으로 학습합니다. 이를 통해 기존 MAML보다 높은 용량을 가지며, Meta-LSTM보다 구현 및 학습이 효율적입니다. 회귀, 분류, 강화 학습 등 다양한 영역에서 단 한 번의 업데이트(one-step adaptation)만으로 기존 최신 메타 학습 기법들을 능가하는 경쟁력 있는 성능을 달성하여, 적은 수의 예시로도 빠르고 정확하게 학습하는 능력을 보여줍니다.
