# Bi-Adversarial Auto-Encoder for Zero-Shot Learning

Yunlong Yu, Zhong Ji, Yanwei Pang, Jichang Guo, Zhongfei Zhang, and Fei Wu (2018)

## 🧩 Problem to Solve

본 논문은 데이터가 부족하거나 수집하기 어려운 상황에서 학습하지 않은 클래스를 분류해야 하는 Zero-Shot Learning (ZSL) 문제를 해결하고자 한다. ZSL의 핵심은 학습 단계에서 본 적 없는 unseen classes를 위해, seen classes에서 학습한 지식을 semantic information(예: attributes, word vectors)을 통해 전이하는 것이다.

최근의 생성 모델 기반 ZSL 방식들은 주로 클래스 의미론(class semantics)에서 시각적 특징(visual features)으로 향하는 단방향 정렬(unidirectional alignment)에만 집중하였다. 이러한 방식은 생성된 시각적 특징이 실제 시각적 분포를 따르도록 유도하지만, 생성된 특징이 원래의 의미론적 정보와 충분히 연관되어 있는지, 혹은 클래스를 구분할 수 있을 만큼 변별력(discriminative power)이 있는지를 보장하지 못한다는 한계가 있다. 따라서 본 논문의 목표는 시각적 모달리티와 의미론적 모달리티 간의 양방향 정렬(bidirectional alignment)을 통해, 실제 분포에 부합하면서도 의미론적으로 밀접하고 변별력이 높은 시각적 특징을 생성하는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Bi-Adversarial Auto-Encoder (BAAE)** 프레임워크를 제안하여 시각적-의미론적 상호작용을 강화하는 것이다. 주요 기여 사항은 다음과 같다.

1. **양방향 정렬(Bidirectional Alignment) 구조**: Encoder-Decoder 구조를 기반으로, 의미론$\rightarrow$시각적 특징 생성과 시각적 특징$\rightarrow$의미론 추론이라는 두 방향의 정렬을 동시에 수행하여 상호작용을 극대화한다.
2. **이중 적대적 네트워크(Bi-Adversarial Networks)**: 시각적 모달리티와 의미론적 모달리티 각각에 대해 독립적인 적대적 학습(adversarial learning)을 적용한다. 이를 통해 생성된 특징이 실제 시각 분포를 따르게 함과 동시에, 추론된 의미론적 특징이 실제 클래스 프로토타입과 일치하도록 강제한다.
3. **변별력 강화를 위한 분류 네트워크(Classification Network)**: 생성된 가짜(pseudo) 특징과 실제 특징 모두를 올바른 클래스로 분류하도록 하는 네트워크를 추가하여, 생성된 특징이 실제 특징만큼의 클래스 변별력을 갖도록 정규화한다.

## 📎 Related Works

ZSL 연구는 크게 두 가지 접근 방식으로 나뉜다.

1. **Discriminative Models**: 시각적 특징을 의미론적 공간으로 투영하거나, 두 공간 사이의 호환성 점수(compatibility score)를 최대화하는 방식이다. 하지만 이러한 방식은 시각적 모달리티와 의미론적 모달리티 사이의 '이질성 간극(heterogeneity gap)'으로 인해 정보 손실이 발생하며, 의미론적 일관성을 유지하기 어렵다는 한계가 있다.
2. **Generative Models**: 클래스 의미론 프로토타입을 입력으로 하여 가짜 시각적 특징을 생성함으로써 unseen classes의 데이터 부족 문제를 해결하려 한다. 특히 GAN이나 VAE 기반 모델들이 제안되었으나, 대부분 단방향 정렬에 치중하여 생성된 특징의 의미론적 연관성과 변별력이 부족한 경우가 많다.

본 논문의 BAAE는 기존 생성 모델들이 간과한 '시각적 특징 $\rightarrow$ 의미론' 방향의 정렬을 추가하고, 두 모달리티 모두에 적대적 학습을 적용함으로써 기존의 단방향 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

BAAE는 Encoder, Decoder, 그리고 Classification Network로 구성된 폐쇄 루프(closed loop) 구조를 가진다. 전체 목적 함수는 다음과 같이 정의된다.

$$\text{Obj} = F_{\text{align}} + F'_{\text{align}} + F_{\text{adv}} + F'_{\text{adv}} + \lambda F_{\text{cls}} + \mu R(\theta, \upsilon)$$

### 1. Encoder (Visual Feature Generator)

Encoder는 클래스 의미론 프로토타입 $a$와 가우시안 노이즈 $z$를 입력받아 가짜 시각적 특징 $\tilde{x}$를 생성한다.

* **정렬 손실 ($F_{\text{align}}$)**: 생성된 $\tilde{x}$와 실제 특징 $x$ 사이의 $\ell_2$-norm 거리로 계산하여 유사성을 높인다.
    $$F_{\text{align}} = \min_{\theta} \sum_{i} \|x_i - \tilde{x}_i\|_2^2$$
* **시각적 적대적 학습 ($F_{\text{adv}}$)**: 생성기(Encoder)와 판별기(Discriminator $D_\phi$)가 서로 경쟁하며, $\tilde{x}$가 실제 시각적 분포 $p(x)$를 따르도록 학습한다. Wasserstein GAN의 Gradient Penalty ($L_{GP}$)를 적용하여 학습 안정성을 높였다.

### 2. Decoder (Semantics Inference)

Decoder는 실제 또는 생성된 시각적 특징 $x$를 입력받아 이를 의미론적 벡터 $\tilde{a}$와 노이즈 벡터 $\tilde{z}$로 분해(decomposition)한다.

* **정렬 손실 ($F'_{\text{align}}$)**: 추론된 $\tilde{a}$가 실제 클래스 프로토타입 $a$와 일치하도록 강제한다.
    $$F'_{\text{align}} = \min_{\theta, \upsilon} \sum_{i} \|a_i - \tilde{a}_i\|_2^2$$
* **의미론적 적대적 학습 ($F'_{\text{adv}}$)**: 판별기 $D_\omega$를 통해 추론된 $[ \tilde{a}; \tilde{z} ]$가 실제의 $[ a; z ]$ 분포와 구별되지 않도록 학습하여 의미론적 정렬을 강화한다.

### 3. Classification Network

생성된 특징의 변별력을 높이기 위해, 실제 특징 $x$와 생성된 특징 $\tilde{x}$를 입력받아 정답 클래스를 예측하는 네트워크를 도입한다.

* **분류 손실 ($F_{\text{cls}}$)**: Cross-entropy 기반의 손실 함수를 사용하여, 생성된 특징이 실제 특징만큼 명확하게 클래스를 구분할 수 있도록 한다.
* **호환성 점수**: 시각적 특징과 의미론 프로토타입 사이의 내적(inner product)을 통해 점수를 계산하며, 이는 seen-unseen bias 문제를 완화하는 효과를 가진다.

## 📊 Results

### 실험 설정

* **데이터셋**: AwA1, AwA2, aPY, SUN 총 4개의 벤치마크 데이터셋을 사용하였다.
* **특징 추출**: ResNet-101의 top layer pooling units에서 추출된 2048차원 특징을 사용하였다.
* **평가 지표**: Traditional ZSL에서는 Top-1 accuracy를, Generalized ZSL(GZSL)에서는 seen 클래스 정확도($s$)와 unseen 클래스 정확도($u$)의 조화 평균(Harmonic Mean, $H = \frac{2su}{s+u}$)을 사용하였다.

### 주요 결과

1. **Traditional ZSL**: 모든 데이터셋에서 BAAE가 가장 높은 성능을 기록하였다. 특히 기존 SOTA 모델 대비 AwA1에서 3.0%, AwA2에서 2.1%의 향상을 보였다.
2. **Generalized ZSL**: GZSL 환경에서도 매우 경쟁력 있는 성능을 보였으며, 특히 AwA1, AwA2, aPY 데이터셋에서 타 모델 대비 큰 마진으로 우위를 점하였다. SUN 데이터셋에서는 CLSWGAN+SM에 약간 뒤처졌으나 전반적으로 우수한 성능을 나타냈다.
3. **분류기 영향 분석**: Nearest Neighbor (NN) 분류기가 Softmax 분류기보다 GZSL의 조화 평균($H$) 측면에서 더 안정적인 결과를 보였다. 이는 생성된 특징의 분포 정보보다 변별력 정보가 seen-unseen bias를 줄이는 데 더 기여함을 시사한다.
4. **샘플 수 영향**: 생성하는 가짜 샘플의 수가 일정 수준까지는 성능을 높이지만, 과도하게 많아지면 오히려 변별력이 저하되어 성능이 감소하는 경향을 보였다. 특히 fine-grained 데이터셋인 SUN에서 이러한 민감도가 높게 나타났다.

## 🧠 Insights & Discussion

본 연구는 ZSL에서 생성 모델을 사용할 때 단순히 분포를 맞추는 것보다 **양방향 정렬**과 **변별력 정규화**가 훨씬 중요하다는 점을 시사한다.

* **강점**: 적대적 학습을 두 모달리티 모두에 적용함으로써 '의미론 $\rightarrow$ 시각'뿐만 아니라 '시각 $\rightarrow$ 의미론'의 일관성을 확보하였다. 또한, 분류 네트워크를 통해 생성된 특징이 단순히 실제 분포를 흉내 내는 것에 그치지 않고 클래스 간 경계를 명확히 하도록 유도하였다.
* **한계 및 해석**: 실험 결과에서 seen 클래스의 정확도가 unseen 클래스보다 훨씬 높게 나타나는데, 이는 생성된 가짜 특징이 실제 특징의 분포를 완벽하게 대체하기에는 여전히 한계가 있음을 의미한다.
* **비판적 논의**: 샘플 수에 따른 성능 변화 그래프는 생성 모델의 '모드 붕괴(mode collapse)' 가능성이나, 너무 많은 가짜 샘플이 오히려 결정 경계를 모호하게 만들 수 있다는 점을 보여준다. 특히 클래스 간 차이가 작은 fine-grained 분류에서는 생성된 데이터의 질이 성능에 결정적인 영향을 미친다는 것을 알 수 있다.

## 📌 TL;DR

본 논문은 ZSL을 위해 **Bi-Adversarial Auto-Encoder (BAAE)**를 제안하여 시각적 특징과 의미론적 프로토타입 간의 **양방향 정렬**을 구현하였다. 두 개의 적대적 네트워크와 하나의 분류 네트워크를 통해 생성된 특징의 **분포 적합성**, **의미론적 연관성**, 그리고 **클래스 변별력**을 동시에 확보하였다. 실험 결과, Traditional 및 Generalized ZSL 모두에서 SOTA 수준의 성능을 달성하였으며, 특히 GZSL의 고질적인 문제인 seen-unseen bias를 효과적으로 완화하였다. 이 연구는 향후 생성 모델 기반의 ZSL 연구에서 단순한 특징 생성을 넘어 모달리티 간의 상호 제약 조건을 설계하는 것이 중요함을 시사한다.
