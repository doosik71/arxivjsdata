# MetaGAD: Meta Representation Adaptation for Few-Shot Graph Anomaly Detection

Xiongxiao Xu, Kaize Ding, Canyu Chen, Kai Shu (2023/2024)

## 🧩 Problem to Solve

본 논문은 **Few-shot Graph Anomaly Detection (Few-shot GAD)** 문제를 해결하고자 한다. 그래프 상에서 이상치(Anomaly)를 탐지하는 작업은 금융 사기 탐지, 소셜 스팸 및 네트워크 침입 탐지 등 보안 분야에서 매우 중요하다. 기존의 GAD 방법론들은 대부분 Unsupervised 방식으로 수행되는데, 이는 대규모의 레이블링된 이상치 데이터를 확보하는 비용이 매우 높기 때문이다. 그러나 Unsupervised 방식은 사전 지식(Prior knowledge)의 부재로 인해 실제로는 중요하지 않은 데이터 인스턴스를 이상치로 판별하는 한계가 있다.

현실적인 시나리오에서는 도메인 전문가나 사용자 피드백을 통해 매우 제한적인 수의 레이블링된 이상치 데이터를 얻는 것이 가능하다. 따라서 본 논문은 소수의 레이블링된 이상치와 대량의 레이블링되지 않은 노드를 모두 활용하여 탐지 성능을 높이는 것을 목표로 한다. 특히, Few-shot 설정에서 발생하는 다음의 세 가지 핵심 난제를 해결하고자 한다:

1. **정보 활용의 불균형**: 대량의 Unlabeled 노드와 소수의 Labeled anomalies 사이의 정보를 원칙적으로 통합하는 방법의 부재.
2. **Representation Gap**: Self-supervised learning(SSL)을 통해 학습된 표현(Representation)이 반드시 이상치 탐지라는 특정 목적(Supervised learning)에 적합하지는 않다는 점.
3. **Overfitting 문제**: 매우 적은 수의 레이블 데이터로 딥러닝 모델을 학습시킬 때 발생하는 전형적인 과적합 문제.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Self-supervised learning으로 학습된 노드 표현을 Few-shot supervised learning에 최적화된 형태로 적응(Adaptation)시키는 Meta-learning 프레임워크**를 설계하는 것이다.

주요 기여 사항은 다음과 같다:

- **Representation Adaptation Network (RAN)**: Raw representation을 이상치 인지 표현(Anomaly-aware representation)으로 변환하는 Meta-learner를 도입하여 SSL과 Supervised learning 간의 표현 간극(Representation gap)을 해소한다.
- **Bi-level Optimization**: RAN과 Anomaly Detector의 학습을 이단계 최적화 구조로 설계하여, 최종적으로 Validation loss를 최소화하는 방향으로 수렴하게 함으로써 Few-shot 설정에서의 과적합 문제를 완화한다.
- **Cost-Sensitive Learning**: 클래스 불균형 문제를 해결하기 위해 비용 민감형 손실 함수를 도입하여 소수의 이상치 데이터가 학습에 효과적으로 반영되도록 한다.

## 📎 Related Works

### 1. Graph Anomaly Detection (GAD)

초기 연구들은 주로 Unsupervised 방식으로 진행되었으며, 최근에는 GNN(Graph Neural Networks)을 활용한 방법론들이 등장하였다. 예를 들어, DOMINANT는 GCN 기반의 Deep Autoencoder를 사용하여 구조와 속성을 재구성함으로써 이상치를 탐지한다. GAAN은 GAN(Generative Adversarial Networks)을 통해 가짜 노드를 생성하고 이를 구분하는 방식으로 학습한다. 하지만 이러한 방식들은 레이블 정보를 활용하지 못해 성능에 한계가 있다. 일부 Semi-supervised 연구가 존재하나, Meta-GDN과 같이 여러 그래프를 사용하는 경우나 SAD와 같이 동적 그래프를 대상으로 하는 경우가 많아, 단일 동종 그래프(Single homogeneous graph)에서의 Few-shot GAD 연구는 상대적으로 부족한 실정이다.

### 2. Few-Shot Graph Learning

제한된 레이블 데이터로 학습하는 Few-shot Graph Learning은 크게 Gradient-based 방법(최적의 파라미터 초기화를 학습)과 Metric-based 방법(일반화된 거리 측정 함수를 학습)으로 나뉜다. 본 논문은 이러한 Few-shot 학습의 아이디어를 GAD에 접목하되, 특히 SSL 표현을 타겟 태스크에 맞게 변환하는 Adaptation 관점에서 접근한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

MetaGAD는 크게 세 가지 구성 요소로 이루어져 있다: **Graph Encoder $\rightarrow$ Representation Adaptation Network (RAN) $\rightarrow$ Anomaly Detector**.

#### (1) Graph Encoder

그래프의 구조적 정보와 속성 정보를 추출하기 위해 GNN을 사용한다. 본 논문에서는 생성 모델 기반의 DOMINANT 인코더를 채택하여 데이터의 원래 분포를 더 잘 특성화하도록 하였다.
$$Z = \text{Encoder}(A, X)$$
여기서 $Z$는 인코더의 마지막 레이어에서 출력된 Raw representation이다.

#### (2) Representation Adaptation Network (RAN, Meta-Learner)

Raw representation $Z$를 이상치 탐지에 적합한 $Z'$로 변환하는 역할을 한다. RAN은 파라미터 $\Phi$를 가진 2층 Feed-forward 네트워크로 구성된다.
$$Z' = g_\Phi(Z) = W_2 \text{ReLU}(W_1 Z + b_1) + b_2$$
RAN은 Validation loss의 피드백을 통해 업데이트되며, 이는 강화학습의 Reward와 유사하게 일반화 성능을 가이드하는 역할을 수행한다.

#### (3) Anomaly Detector (Target Model)

변환된 $Z'$를 입력으로 받아 각 노드의 이상치 점수(Anomaly score) $s_i$를 계산하는 MLP(Multi-Layer Perceptron)이다. 파라미터 $\Theta$를 가지며, Training loss를 통해 학습된다.
$$s_i = f_\Theta(Z'_{i,:})$$

### 2. Cost-Sensitive Loss Function

이상치($y_i=1$)와 정상 노드($y_i=0$) 간의 극심한 불균형을 해결하기 위해 비용 가중치 $w$를 도입한 손실 함수를 사용한다.
$$L = -\frac{1}{n} \sum_{i=1}^n [w \cdot y_i \log(\sigma(s_i)) + (1-y_i) \log(1-\sigma(s_i))]$$
여기서 $\sigma$는 Sigmoid 함수이며, $w$는 양성 샘플(이상치)의 가중치를 조절하여 클래스 불균형의 영향을 제어한다.

### 3. Meta-Learning 및 Bi-level Optimization

MetaGAD의 핵심은 RAN($\Phi$)과 Anomaly Detector($\Theta$)를 상호 보완적으로 학습시키는 Bi-level Optimization 구조에 있다.

**목적 함수:**
$$\min_\Phi L_{val}(\Theta^*(\Phi), \Phi)$$
$$\text{s.t. } \Theta^*(\Phi) = \arg \min_\Theta L_{train}(\Theta, \Phi)$$

이는 $\Theta$를 training set으로 최적화한 후, 그 결과가 validation set에서 최소 loss를 갖도록 $\Phi$를 최적화하는 구조이다. 이를 통해 모델이 Training set에 과적합되는 것을 방지하고 일반화 능력을 향상시킨다.

**근사 최적화 알고리즘 (Approximate Optimization):**
계산 복잡도를 줄이기 위해 One-step SGD를 사용하여 근사적으로 업데이트한다.

1. **Target Model 업데이트**: $\Theta' = \Theta - \alpha \nabla_\Theta L_{train}(\Theta, \Phi)$
2. **Meta-Learner 업데이트**: $\Phi' = \Phi - \beta \nabla_\Phi L_{val}(\Theta', \Phi)$

이때, $\nabla_\Phi L_{val}$ 계산 시 발생하는 고차 미분(Hessian) 항은 Finite difference approximation 기법을 사용하여 효율적으로 계산한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Synthetic anomalies를 생성한 3개 데이터셋(Cora, Citeseer, Amazon Photo)과 Organic anomalies가 포함된 3개 데이터셋(Wiki, Amazon Review, YelpChi)을 사용하였다.
- **지표**: AUC-ROC 및 AUC-PR을 사용하였다.
- **비교 대상**: Shallow methods (Radar, ANOMALOUS), Unsupervised Deep Learning (DOMINANT, CoLA 등), Semi-supervised Deep Learning (SemiGNN, GDN 등)과 비교하였다.

### 2. 주요 결과

- **전반적 성능 (RQ1)**: MetaGAD는 모든 데이터셋에서 기존 baseline들을 유의미하게 앞섰다. 특히 YelpChi 데이터셋에서는 GDN 대비 AUC-ROC가 13.76% 향상되었다.
- **Few-shot 효율성**: 단 1개의 레이블링된 이상치만 제공된 **1-shot 설정에서도 baseline들을 압도하는 성능**을 보였으며, 레이블 수의 증가에 따라 성능이 선형적으로 증가하지는 않았으나, 매우 적은 데이터만으로도 충분한 성능을 낼 수 있음을 입증하였다.
- **Ablation Study (RQ2)**: RAN을 제거하거나($\text{META}^-$), 단순히 Fine-tuning만 수행한 경우($\text{FINETUNE}$)보다 Meta-learning 기반의 MetaGAD 성능이 월등히 높았다. 또한, Encoder를 함께 학습시키는 것($\text{JOINT}$)보다 pretrained encoder를 사용하는 것이 더 안정적이었다.
- **과적합 완화 (RQ3)**: $\text{FINETUNE}$ 방식은 epoch이 진행됨에 따라 validation loss가 다시 상승하는 전형적인 과적합 양상을 보였으나, MetaGAD는 validation loss가 지속적으로 감소하여 일반화 성능이 유지됨을 확인하였다.
- **클래스 불균형 및 오염도 (RQ4, RQ5)**:
  - 가중치 $w$를 완전히 균형 있게 설정하는 것보다, 어느 정도 불균형한 상태를 유지하는 것이 실제 환경의 분포와 유사하여 성능이 더 좋게 나타났다.
  - Unlabeled 노드 속에 숨어있는 이상치(Contamination)의 비율이 높아져도 성능 하락 폭이 매우 적어, 노이즈에 대해 강건함(Robustness)을 보였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 **"학습된 표현의 적응(Representation Adaptation)"**이라는 관점에서 Few-shot GAD를 접근했다는 점이다. 단순히 적은 데이터로 모델을 학습시키는 것이 아니라, 이미 존재하는 강력한 SSL 표현을 타겟 태스크에 맞게 변형하는 RAN을 도입함으로써 데이터 부족 문제를 효과적으로 해결하였다.

특히, **Bi-level Optimization**의 도입은 단순한 하이퍼파라미터 튜닝이 아니라, 최적화 과정 자체에 validation loss를 편입시킴으로써 딥러닝의 고질적인 문제인 과적합을 구조적으로 방지하였다는 점에서 학술적 가치가 높다.

다만, 본 논문에서는 단일 동종 그래프(Single homogeneous graph)만을 다루었으므로, 현실 세계의 더 복잡한 형태인 이종 그래프(Heterogeneous graphs)나 동적 그래프(Dynamic graphs)에서의 일반화 가능성에 대해서는 추가적인 연구가 필요하다. 또한, DOMINANT 외의 다른 SSL 인코더(예: Contrastive models)를 사용했을 때의 성능 변화에 대한 분석이 보완된다면 더욱 견고한 프레임워크가 될 것으로 보인다.

## 📌 TL;DR

MetaGAD는 소수의 레이블링된 이상치만으로 그래프 이상치를 탐지하는 **Few-shot GAD** 프레임워크이다. Self-supervised learning으로 얻은 Raw representation을 **Representation Adaptation Network (RAN)**를 통해 이상치 탐지에 최적화된 표현으로 변환하며, **Bi-level Optimization**을 통해 Few-shot 학습의 치명적인 문제인 과적합을 해결한다. 실험 결과, 1-shot 수준의 극소수 데이터만으로도 기존 Unsupervised 및 Semi-supervised 모델보다 뛰어난 성능을 보였으며, 이는 향후 데이터 확보가 어려운 실제 보안 및 금융 시스템의 이상 탐지 분야에 매우 유용하게 적용될 가능성이 높다.
