# A Meta-Learning Framework for Generalized Zero-Shot Learning

Vinay Kumar Verma, Dhanajit Brahma, and Piyush Rai (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Zero-Shot Learning (ZSL)과 그 확장 형태인 Generalized Zero-Shot Learning (GZSL)에서 발생하는 성능 저하 문제이다. ZSL은 학습 단계에서 보지 못한 클래스(unseen classes)에 속하는 샘플을 테스트 시점에 분류하는 작업이며, GZSL은 테스트 샘플이 학습된 클래스(seen classes)와 보지 못한 클래스 모두에서 나올 수 있는 더욱 어려운 설정이다. 특히 GZSL에서는 모델이 학습 데이터에 존재하는 seen 클래스에 대해 강한 편향(bias)을 갖게 되어, unseen 클래스를 제대로 분류하지 못하는 문제가 발생한다.

최근 VAE나 GAN과 같은 생성 모델을 통해 unseen 클래스의 가상 샘플을 합성하여 이 문제를 해결하려는 시도가 있었으나, 다음과 같은 한계가 존재한다:

1. 생성자가 seen 클래스 데이터로만 학습되어, unseen 클래스 샘플을 생성하는 능력을 명시적으로 학습하지 않는다.
2. seen과 unseen 클래스 생성 모두에 일반화될 수 있는 최적의 파라미터를 학습하지 못한다.
3. seen 클래스당 가용한 샘플 수가 매우 적은 경우(Few-shot setting), 모델의 성능이 급격히 떨어진다.

따라서 본 논문의 목표는 Meta-learning 프레임워크를 생성 모델에 통합하여, 소량의 seen 클래스 데이터만으로도 고품질의 unseen 클래스 샘플을 생성할 수 있는 일반화된 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Model-Agnostic Meta-Learning (MAML)을 Wasserstein GAN (WGAN)에 결합하여 생성자의 일반화 성능을 극대화하는 것이다. 구체적인 기여 사항은 다음과 같다:

- **Meta-learning 기반 생성 모델 제안**: MAML을 통해 소수의 샘플만으로도 새로운 클래스 분포에 빠르게 적응할 수 있는 최적의 파라미터를 학습함으로써, 기존 생성 모델이 가진 데이터 의존성 및 일반화 부족 문제를 해결하였다.
- **Zero-Shot Task Distribution 설계**: MAML의 에피소드 학습 과정에서 메타-학습 세트($T_{tr}$)와 메타-검증 세트($T_{val}$)의 클래스를 서로 겹치지 않게(disjoint) 구성하는 새로운 태스크 분포를 제안하였다. 이를 통해 학습 단계에서부터 '보지 못한 클래스'를 생성하는 능력을 명시적으로 학습하도록 유도하였다.
- **Few-shot ZSL/GZSL 대응**: Meta-learning의 특성을 활용하여 seen 클래스의 데이터가 극소수(예: 클래스당 5~10개)만 존재하는 극한의 환경에서도 강건한 성능을 낼 수 있음을 입증하였다.

## 📎 Related Works

기존의 ZSL 접근 방식은 크게 두 가지 흐름으로 나뉜다. 첫 번째는 입력 데이터를 클래스 속성(attribute) 공간으로 매핑하여 최근접 이웃 탐색을 통해 클래스를 예측하는 방식이다. 두 번째는 시각적 공간과 의미적 공간 사이의 bilinear compatibility 함수를 학습하는 방식이다. 하지만 이러한 방식들은 대량의 seen 클래스 데이터가 필요하며, GZSL 설정에서 seen 클래스 편향 문제를 해결하지 못한다는 한계가 있다.

최근에는 VAE나 GAN을 사용하여 unseen 클래스의 특징(feature)을 합성하고, 이를 통해 ZSL 문제를 일반적인 지도 학습(supervised learning) 문제로 변환하는 생성 기반 접근법이 주목받고 있다. 그러나 기존 생성 모델들은 여전히 seen 클래스에 과적합되는 경향이 있으며, unseen 클래스 생성 품질과 실제 샘플 간의 간극이 크다. 본 논문은 이러한 한계를 극복하기 위해 MAML을 도입하여 파라미터 수준에서의 일반화를 꾀했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인

제안된 ZSML(Zero-Shot Meta-Learning) 모델은 Generator($G$), Discriminator($D$), 그리고 Classifier($C$)의 세 가지 모듈로 구성되며, 각 모듈은 MAML 기반의 메타-러너(meta-learner)를 가지고 있다. 전체적인 흐름은 다음과 같다:

1. **에피소드 구성**: 태스크 $T_i$를 구성할 때, 학습 세트 $T_{tr}$(seen)과 검증 세트 $T_{val}$(unseen)의 클래스를 완전히 분리한다.
2. **내부 루프(Inner Loop)**: $T_{tr}$ 데이터를 사용하여 $G, D, C$의 파라미터를 빠르게 업데이트한다.
3. **외부 루프(Outer Loop)**: 업데이트된 파라미터를 사용하여 $T_{val}$ 데이터에 대한 손실을 계산하고, 이를 바탕으로 전역 파라미터를 업데이트한다.
4. **추론 단계**: 학습된 $G$를 이용해 unseen 클래스의 속성 벡터 $a_c$와 노이즈 $z$를 입력하여 가상 샘플 $\hat{x}$를 생성하고, 이를 이용해 최종 분류기(SVM 또는 Softmax)를 학습시킨다.

### 주요 구성 요소 및 역할

- **Generator ($G$)**: 랜덤 노이즈 $z$와 클래스 속성 $a_c$를 입력받아 해당 클래스의 특징 벡터 $\hat{x}$를 생성한다.
- **Discriminator ($D$)**: 입력된 샘플이 실제 데이터인지 생성된 데이터인지 판별한다. WGAN 구조를 채택하여 학습 안정성을 높였다.
- **Classifier ($C$)**: 생성된 샘플 $\hat{x}$가 원래 의도한 클래스 $c$로 올바르게 분류되는지 확인하여, 생성된 샘플의 식별 가능성(discriminability)을 보장한다.

### 손실 함수 및 방정식

Discriminator의 목적 함수 $L_{D}^{T_i}$는 실제 샘플에 대해서는 높은 값을, 생성 샘플에 대해서는 낮은 값을 갖도록 최대화하는 것이다:
$$L_{D}^{T_i}(\theta_d) = \mathbb{E}_{T_i} D(x, a_c | \theta_d) - \mathbb{E}_{a_c, \hat{x} \sim P_{\theta_g}} D(\hat{x}, a_c | \theta_d)$$

Generator와 Classifier의 공동 목적 함수 $L_{GC}^{T_i}$는 Discriminator를 속이고(즉, $D$의 값을 높이고) 분류기가 정답을 맞히도록 최소화하는 것이다:
$$L_{GC}^{T_i}(\theta_{gc}) = -\mathbb{E}_{a_c, z \sim N(0, I)} [D(G(a_c, z | \theta_g), a_c | \theta_d) + C(y | \hat{x}, \theta_c)]$$

### 학습 절차

1. **Inner Update**: 각 태스크 $T_i$에 대해 gradient ascent/descent를 수행하여 적응된 파라미터 $\theta'_d, \theta'_{gc}$를 구한다.
   $$\theta'_d = \theta_d + \eta_1 \nabla_{\theta_d} l_{D}^{T_{tr}}(\theta_d)$$
   $$\theta'_{gc} = \theta_{gc} - \eta_2 \nabla_{\theta_{gc}} l_{GC}^{T_{tr}}(\theta_{gc})$$
2. **Meta Update**: $T_{val}$에서의 손실을 기반으로 전역 파라미터를 업데이트한다.
   $$\theta_d \leftarrow \theta_d + \beta_1 \nabla_{\theta_d} \sum_{T_{val} \in T_i} l_{D}^{T_{val}}(\theta'_d)$$
   $$\theta_{gc} \leftarrow \theta_{gc} - \beta_2 \nabla_{\theta_{gc}} \sum_{T_{val} \in T_i} l_{GC}^{T_{val}}(\theta'_{gc})$$

## 📊 Results

### 실험 설정

- **데이터셋**: SUN, CUB, AWA1, AWA2, aPY의 5개 벤치마크 데이터셋을 사용하였다.
- **특징 추출**: ResNet-101의 사전 학습된 모델을 사용하여 이미지 특징 벡터를 추출하였다.
- **평가 지표**: ZSL에서는 클래스별 평균 정확도(per-class mean accuracy)를, GZSL에서는 seen/unseen 정확도의 조화 평균(Harmonic Mean, HM)을 사용하였다.

### 주요 결과

- **ZSL 성능**: 기존 SOTA 모델 대비 CUB(4.5%), AWA1(6.0%), AWA2(9.8%), aPY(27.9%)의 상대적 성능 향상을 보였다.
- **GZSL 성능**: 모든 데이터셋에서 조화 평균(HM) 지표가 크게 개선되었으며, 특히 CUB(3.9%), aPY(11.8%), AWA1(3.3%), AWA2(3.6%)의 상대적 향상을 기록하였다.
- **Few-shot 환경**: seen 클래스당 샘플을 5개 또는 10개만 사용했을 때도, 전체 데이터를 사용한 기존 SOTA 모델보다 우수한 성능을 보였으며, 특히 GZSL 설정에서 매우 강력한 성능을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구의 가장 큰 성과는 MAML의 '빠른 적응' 능력을 생성 모델에 접목하여, seen 클래스에 편향되지 않은 **일반적인 최적 파라미터 $\Theta$**를 찾아냈다는 점이다. 특히 $T_{tr}$과 $T_{val}$을 disjoint하게 구성한 전략은 모델이 학습 단계에서부터 "알지 못하는 클래스를 어떻게 생성해야 하는가"라는 ZSL의 본질적인 문제를 직접적으로 학습하게 만들었다. 또한, Discriminator와 Generator 모두에 메타-러너를 적용함으로써, 더 강력한 판별자가 더 정교한 생성자를 유도하는 상호 보완적 피드백 루프가 형성되었음을 확인하였다.

### 한계 및 논의

- **데이터셋별 편차**: SUN 데이터셋에서는 다른 데이터셋에 비해 상대적으로 낮은 성능 향상을 보였는데, 이는 SUN 데이터셋의 클래스 수가 매우 많아(717개) GAN의 고질적인 문제인 모드 붕괴(mode collapse)가 발생했을 가능성이 크다고 분석된다.
- **가정**: 본 모델은 클래스 속성(attribute) 벡터가 정확하게 제공된다는 가정 하에 작동한다. 속성 벡터의 품질이 낮을 경우 생성되는 샘플의 질이 떨어질 수 있다.
- **비판적 해석**: 제안된 방식은 생성된 샘플을 이용해 다시 분류기를 학습시키는 2단계 구조를 가진다. 이 과정에서 생성된 샘플의 분포가 실제 unseen 데이터의 분포와 완벽히 일치하지 않을 경우, 여전히 domain shift 문제가 발생할 수 있다.

## 📌 TL;DR

본 논문은 **MAML(Model-Agnostic Meta-Learning)과 WGAN을 결합**하여, 소량의 데이터만으로도 고품질의 unseen 클래스 샘플을 생성할 수 있는 **ZSML 프레임워크**를 제안한다. 특히 학습/검증 클래스를 분리한 **Zero-Shot Task Distribution**을 통해 seen 클래스 편향 문제를 효과적으로 해결하였으며, ZSL과 GZSL 모두에서 SOTA 성능을 달성하였다. 이 연구는 데이터가 극히 적은 Few-shot ZSL 환경에서도 강력한 성능을 보여, 향후 실제 데이터 수집이 어려운 도메인의 제로샷 학습 연구에 중요한 기여를 할 것으로 보인다.
