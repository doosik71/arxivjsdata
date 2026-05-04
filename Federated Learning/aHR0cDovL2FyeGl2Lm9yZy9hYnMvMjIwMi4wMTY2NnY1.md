# Proportional Fairness in Federated Learning

Guojun Zhang, Saber Malekmohammadi, Xi Chen, Yaoliang Yu (2023)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 환경에서 발생하는 **공정성(Fairness)** 문제를 해결하고자 한다. 연합 학습 시스템이 실제 환경에 광범위하게 배포됨에 따라, 다양하고 이질적인 클라이언트들이 모두 합리적으로 만족스러운 성능을 얻도록 보장하는 것이 매우 중요해졌다.

일반적으로 FL에서의 공정성은 크게 두 가지 관점으로 나뉜다. 첫째는 전체 사회의 효용을 최대화하려는 **공리주의(Utilitarianism)** 관점이며, 이는 대표적으로 FedAvg 알고리즘이 이에 해당한다. 둘째는 가장 불리한 조건에 있는 클라이언트의 이득을 보장하려는 **평등주의(Egalitarianism)** 관점이며, 이는 Agnostic Federated Learning (AFL)과 같은 알고리즘이 추구하는 방향이다.

문제는 이 두 관점이 서로 충돌한다는 점이다. 최악의 성능을 내는 클라이언트를 개선하려 하면, 이미 좋은 성능을 내고 있는 클라이언트의 성능이 크게 저하될 수 있다. 따라서 본 논문의 목표는 공리주의적 효율성과 평등주의적 형평성 사이의 적절한 균형을 맞추는 새로운 공정성 개념인 **비례적 공정성(Proportional Fairness, PF)**을 FL에 도입하고, 이를 구현하는 알고리즘인 **PropFair**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 절대적인 성능 변화가 아닌 **상대적인 성능 변화(Relative change)**에 기반하여 공정성을 정의하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **비례적 공정성(PF)의 도입**: 협동 게임 이론(Cooperative Game Theory)의 **내쉬 협상 솔루션(Nash Bargaining Solution, NBS)**을 FL에 접목하여, 모든 클라이언트의 효용 곱을 최대화하는 방향으로 공정성을 정의하였다.
2. **PropFair 알고리즘 제안**: PF를 달성하기 위해 설계된 새로운 surrogate loss 함수를 제안하였으며, 이는 기존의 loss 관점의 공정성 접근 방식과 차별화되는 utility 관점의 접근 방식이다.
3. **이론적 수렴성 증명**: 완만한 가정 하에서 PropFair가 목적 함수의 stationary point로 수렴함을 수학적으로 증명하였다.
4. **실험적 검증**: 시각 및 언어 데이터셋을 통해 PropFair가 평균 성능과 최악의 10% 클라이언트 성능 사이에서 우수한 균형을 이룬다는 것을 입증하였다.

## 📎 Related Works

논문은 기존의 fair FL 알고리즘들을 다음과 같이 분류하고 한계를 지적한다.

- **FedAvg (Utilitarianism)**: 평균 손실을 최소화하지만, 특정 클라이언트가 매우 낮은 성능을 갖게 되는 불공정성 문제가 발생한다.
- **AFL (Egalitarianism)**: 최악의 경우의 손실을 최소화하는 Maximin 기준을 따른다. 그러나 일부 클라이언트의 샘플 수가 매우 적을 경우, empirical estimate가 실제 분포를 반영하지 못해 **일반화 성능(Generalization)**이 크게 떨어지는 문제가 있다.
- **q-FFL 및 TERM**: $\alpha$-fairness나 softmax 기반의 가중치 조절을 통해 공리주의와 평등주의 사이의 절충안을 제시한다. 하지만 이들이 정확히 어떤 종류의 균형을 추구하는지에 대한 직관적인 설명이 부족하다.

PropFair는 이러한 기존 방식들과 달리, "상대적 이득"이라는 직관적인 개념을 도입하여 NBS를 통해 수학적으로 정교한 균형점을 찾는다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 목적 함수

PropFair는 각 클라이언트 $i$의 유틸리티(Utility) $u_i$를 정의하고, 다음과 같은 내쉬 곱(Nash product)을 최대화하는 것을 목표로 한다.
$$\max_{\theta} \sum_{i} p_i \log u_i(\theta)$$
여기서 $p_i$는 클라이언트의 샘플 수에 비례하는 가중치이다. 본 논문에서는 유틸리티를 $u_i = M - f_i(\theta)$로 정의하며, 여기서 $f_i(\theta)$는 클라이언트 $i$의 손실 함수이고 $M$은 하이퍼파라미터(Baseline)이다. 따라서 최종적인 최적화 목적 함수 $\pi(\theta)$는 다음과 같다.
$$\min_{\theta} \pi(\theta) := -\sum_{i} p_i \log(M - f_i(\theta))$$

### 2. Huberization (안정화 장치)

$M - f_i$ 값이 매우 작아져 그래디언트가 폭발하거나, 음수가 되어 로그 함수를 사용할 수 없는 문제를 해결하기 위해 **Huberized** 버전을 도입한다.
$$\log^{[\epsilon]}(M-t) := \begin{cases} \log(M-t), & \text{if } t \le M-\epsilon \\ \log\epsilon - \frac{1}{\epsilon}(t-M+\epsilon), & \text{if } t > M-\epsilon \end{cases}$$
이 함수는 $t = M-\epsilon$ 지점에서 값과 미분값이 연속이 되도록 설계된 robust $C^1$ 확장 함수이다.

### 3. Dual View (이중 관점)

본 논문은 Kolmogorov의 일반화 평균(generalized mean)을 통해 PropFair를 해석한다. PropFair는 결국 각 클라이언트의 손실 함수에 가중치를 곱한 합을 최소화하는 문제로 환원될 수 있으며, 이때 최적 가중치 $\lambda_i$는 다음과 같이 결정된다.
$$\lambda_i \propto p_i (M - f_i)$$
즉, **손실이 큰(성능이 나쁜) 클라이언트에게 더 많은 가중치를 부여**하여 학습을 진행하는 구조이다.

### 4. 학습 절차 (Algorithm 1)

PropFair는 FedAvg의 구조를 그대로 따르되, 로컬 업데이트 단계에서 각 배치의 손실 함수 $\ell_{S_i}(\theta)$를 $\log^{[\epsilon]}(M - \ell_{S_i}(\theta))$로 대체하여 계산한다. 이후 서버에서 이를 가중 평균하여 글로벌 모델을 업데이트한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: CIFAR-10, CIFAR-100, TinyImageNet, Shakespeare
- **모델**: ResNet-18 (Group Normalization 적용), LSTM
- **비교 대상**: FedAvg, AFL, q-FFL, TERM
- **평가 지표**: 전체 클라이언트의 평균 정확도(Average Accuracy), 최악의 10% 클라이언트의 평균 정확도(Worst 10% Accuracy)

### 2. 주요 결과

- **비례적 공정성 검증**: CIFAR-10 및 CIFAR-100 데이터셋에서 PropFair로 학습된 모델을 다른 알고리즘으로 fine-tuning 했을 때, 대부분의 클라이언트 성능이 상대적으로 저하됨을 확인하였다. 이는 PropFair가 정의한 PF 솔루션이 다른 알고리즘보다 더 공정한 지점에 위치함을 시사한다.
- **성능 균형**:
  - **평균 성능**: FedAvg나 q-FFL에 비해 약간 낮거나 유사한 수준을 유지하며 경쟁력을 보였다.
  - **최악 성능**: 모든 데이터셋에서 **Worst 10% 정확도가 기존 알고리즘들보다 높게 나타나 SOTA(State-of-the-art)를 달성**하였다. 특히 AFL이 이론적으로는 최악의 경우를 개선해야 함에도 불구하고, 실제 시각 데이터셋에서는 PropFair보다 성능이 낮았는데, 이는 AFL의 일반화 문제 때문으로 분석된다.

## 🧠 Insights & Discussion

### 1. AFL의 일반화 문제와 PropFair의 해결책

논문은 AFL이 일부 클라이언트의 샘플 수가 매우 적을 때(Outlier), 훈련 오차는 낮으나 테스트 오차가 매우 높아지는 일반화 실패 사례를 제시한다. 반면 PropFair는 목적 함수에 샘플 수 기반 가중치 $p_i$가 포함되어 있어, 샘플이 적은 클라이언트가 전체 모델을 과도하게 지배하는 것을 방지함으로써 이 문제를 자연스럽게 해결한다.

### 2. FedAvg와의 관계

$M \to \infty$인 극한 상황에서 PropFair의 목적 함수는 1차 근사화를 통해 FedAvg의 목적 함수와 유사해진다. 즉, FedAvg는 PropFair의 특수한(혹은 근사적인) 형태로 해석될 수 있다.

### 3. 비판적 해석

PropFair는 매우 직관적이고 구현이 간단하며 이론적 수렴성까지 갖춘 우수한 알고리즘이다. 다만, 하이퍼파라미터 $M$과 $\epsilon$의 선택이 성능에 영향을 줄 수 있으며, 데이터셋마다 최적의 $M$ 값이 다르다는 점은 실제 적용 시 튜닝 비용을 발생시킬 수 있는 요인이다.

## 📌 TL;DR

본 논문은 연합 학습에서 공리주의(평균 최적화)와 평등주의(최악 최적화) 사이의 균형을 잡기 위해 게임 이론의 **내쉬 협상 솔루션(NBS)**에 기반한 **비례적 공정성(Proportional Fairness)** 개념을 도입하였다. 이를 구현한 **PropFair** 알고리즘은 로그 기반의 surrogate loss를 사용하여 성능이 낮은 클라이언트에게 더 높은 가중치를 부여하며, 실험을 통해 평균 성능을 크게 희생하지 않으면서도 최악의 성능을 겪는 클라이언트들의 정확도를 획기적으로 높일 수 있음을 증명하였다. 이 연구는 향후 더 신뢰할 수 있고 공정한 분산 학습 시스템을 구축하는 데 중요한 이론적/실천적 토대를 제공한다.
