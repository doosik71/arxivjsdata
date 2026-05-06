# End-to-end Generative Zero-shot Learning via Few-shot Learning

Georgios Chochlakis, Efthymios Georgiou, Alexandros Potamianos (2021)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning(ZSL)과 Generalized Zero-Shot Learning(GZSL)에서 발생하는 데이터 부족 문제를 해결하고자 한다. ZSL은 학습 단계에서 전혀 보지 못한(unseen) 클래스에 대해 분류를 수행해야 하는 과제로, 일반적으로 클래스의 메타데이터(속성 벡터 등)를 이용해 해결한다. 최근의 주류 접근 방식인 Generative ZSL은 생성 모델을 통해 메타데이터 기반의 가상 샘플을 생성하고, 이를 이용해 지도 학습 방식의 분류기를 학습시킨다.

하지만 기존의 Generative ZSL 방식은 생성 모델 학습과 분류기 학습이 분리된 다단계(multi-stage) 파이프라인으로 구성되어 있어, 분류기의 성능 향상이 생성 모델의 학습 과정에 직접적으로 반영되지 않는 한계가 있다. 즉, 생성된 샘플이 분류 관점에서 얼마나 변별력(discriminative) 있는지를 최적화하는 end-to-end 학습 구조가 부족하다는 점이 핵심 문제이다.

## ✨ Key Contributions

본 논문은 ZSL 문제를 Few-Shot Learning(FSL) 문제로 환원시키는 새로운 프레임워크인 **Z2FSL**을 제안한다. 핵심 아이디어는 생성 모델(Generative ZSL backbone)이 생성한 가상 샘플들을 FSL 분류기의 **Support Set**으로 활용하고, 생성 모델과 FSL 분류기를 동시에 학습시키는 end-to-end 구조를 구축하는 것이다.

이를 통해 얻는 주요 이점은 다음과 같다:

1. **변별적 생성(Discriminative Generation):** FSL 분류기의 손실 함수가 생성 모델로 역전파됨으로써, 분류기가 더 잘 구분할 수 있는 형태의 샘플이 생성되도록 유도한다.
2. **유연한 통합:** 특정 알고리즘에 종속되지 않고, 다양한 Generative ZSL 백본과 FSL 분류기 알고리즘을 결합할 수 있는 범용적인 프레임워크를 제공한다.
3. **사전 학습 활용:** FSL 분류기는 실제 데이터셋(예: ImageNet)에서 사전 학습될 수 있어, 생성된 가상 데이터에만 의존하지 않고 기초적인 분류 능력을 갖춘 상태에서 학습을 시작할 수 있다.

## 📎 Related Works

기존의 ZSL 연구는 크게 세 가지 단계로 발전해 왔다. 초기에는 이미지의 속성(attribute)을 추론하거나 semantic space로 매핑하는 방식(DAP, IAP, ALE, DEVISE 등)이 주를 이루었다. 이후 클래스별 프로토타입을 계산하여 분류하는 Prototypical 접근 방식(SYNC, CVCZSL 등)이 제안되었다.

가장 최근의 State-of-the-art(SOTA)는 Generative ZSL이다. 이는 f-VAEGAN, LisGAN과 같이 조건부 생성 모델을 사용하여 unseen 클래스의 특징(feature)을 합성하고, 이를 통해 ZSL을 일반적인 지도 학습 문제로 변환한다. 그러나 이러한 방법들은 생성 모델을 먼저 학습시키고, 이후 고정된 샘플들로 별도의 분류기를 학습시키는 방식이기에 전체 파이프라인의 최적화가 어렵다는 한계가 있었다.

## 🛠️ Methodology

### 전체 시스템 구조

Z2FSL 프레임워크는 크게 **Generative ZSL Backbone**과 **FSL Classifier** 두 가지 모듈로 구성된다.

- **Backbone:** 클래스 메타데이터 $a$와 노이즈 $z$를 입력받아 가상 샘플 $G(a, z)$를 생성한다.
- **Classifier:** 생성된 가상 샘플들로 구성된 Support Set과 실제 샘플들로 구성된 Query Set을 입력받아 분류를 수행한다.

### 주요 구성 요소 및 알고리즘

#### 1. f-VAEGAN (백본 예시)

본 논문에서는 VAE(Variational Autoencoder)와 WGAN(Wasserstein GAN)을 결합한 f-VAEGAN을 백본으로 사용한다. VAE의 디코더와 WGAN의 생성자가 가중치를 공유하며, 전체 손실 함수는 다음과 같이 정의된다.
$$\mathcal{L}_{VAEGAN} = \mathcal{L}_{VAE} + \beta \cdot \mathcal{L}_{WGAN}$$
여기서 $\mathcal{L}_{VAE}$는 재구성 오차와 KL-divergence를 최소화하며, $\mathcal{L}_{WGAN}$은 실제 데이터 분포와 생성 데이터 분포 사이의 Wasserstein 거리를 줄이는 역할을 한다.

#### 2. Prototypical Networks (분류기 예시)

FSL 분류기로는 Prototypical Networks(PN)를 사용한다. PN은 입력 샘플을 임베딩 공간으로 매핑한 후, 각 클래스의 평균 벡터인 프로토타입 $c_k$를 계산하여 유클리드 거리 기반으로 분류한다.
$$c_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} f_\phi(x_i)$$
분류 확률 $p_\phi(y=k|x)$는 다음과 같이 softmax 형태로 계산된다.
$$p_\phi(y=k|x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_{k'} \exp(-d(f_\phi(x), c_{k'}))}$$

### 학습 절차 및 손실 함수

Z2FSL의 핵심은 생성 모델과 분류기를 동시에 학습시키는 것이다.

1. **FSL 분류기 학습:** 생성 모델이 만든 가상 샘플을 Support Set으로, 실제 샘플을 Query Set으로 사용하여 $\mathcal{L}_{FSL}$을 계산하고 분류기를 업데이트한다.
2. **생성 모델 학습:** 백본 자체의 생성 손실 $\mathcal{L}_{ZSL}$과 FSL 분류기에서 오는 손실 $\mathcal{L}_{FSL}$을 함께 사용하여 생성자를 업데이트한다.
   $$\mathcal{L}_{Z2FSL} = \mathcal{L}_{ZSL} + \gamma \mathcal{L}_{FSL}$$
   여기서 $\gamma$는 FSL 손실의 비중을 조절하는 하이퍼파라미터이다.

이 과정은 end-to-end로 이루어지므로, 분류기가 겪는 어려움이 생성자에게 직접 전달되어 더욱 변별력 있는 샘플을 생성하게 만든다.

## 📊 Results

### 실험 설정

- **데이터셋:** CUB (새), AwA2 (동물), SUN (장면)
- **특징 추출기:** ImageNet으로 사전 학습된 ResNet-101의 2048차원 특징 벡터 사용.
- **평가 지표:**
  - ZSL: 각 클래스별 Top-1 정확도의 평균.
  - GZSL: seen 클래스 정확도($s$)와 unseen 클래스 정확도($u$)의 조화 평균($H = \frac{2us}{u+s}$).

### 정량적 결과

1. **ZSL 성능:** Z2FSL(f-VAEGAN, PN)은 SUN 데이터셋에서 SOTA 성능을 달성하였으며, 다른 벤치마크에서도 f-VAEGAN 대비 우수한 성능을 보였다. 특히 단순한 VAE나 WGAN 백본을 사용하더라도 Z2FSL 프레임워크를 적용하면 기존의 복잡한 f-VAEGAN 단독 모델보다 높은 성능을 내는 것이 확인되었다.
2. **GZSL 성능:** AwA2 데이터셋에서 기존 SOTA 대비 소폭 향상된 성능을 보였다. 다만 CUB와 SUN에서는 seen/unseen 클래스 간의 균형을 맞추는 것이 어려워 상대적으로 성능 향폭이 적었다.

### 절제 연구 (Ablation Study)

- **End-to-End 학습의 효과:** FSL 분류기를 테스트 시에만 사용하거나 학습 시에만 사용한 경우보다, 두 단계 모두에서 결합하여 학습시켰을 때 성능 향상이 가장 컸다. 이는 생성 과정이 분류기에 최적화되는 '변별적 생성'의 효과를 입증한다.
- **사전 학습의 중요성:** FSL 분류기를 실제 데이터로 사전 학습시키지 않았을 때 SUN에서 5.2%, CUB에서 4.5%의 성능 저하가 발생하여, Overfitting 방지를 위한 사전 학습이 필수적임을 보였다.

## 🧠 Insights & Discussion

본 논문은 ZSL의 고질적인 문제인 '데이터 부재'를 FSL의 '소량 데이터 학습 능력'으로 해결하려는 독창적인 접근 방식을 취했다. 특히 생성 모델의 출력을 FSL의 Support Set으로 정의함으로써 ZSL을 FSL로 환원시킨 점이 매우 인상적이다.

**강점:**

- 생성 모델과 분류기를 단일 목적 함수로 연결하여 전체 파이프라인을 최적화하였다.
- 특정 알고리즘에 국한되지 않는 프레임워크 형태(Z2FSL(z, f))로 제안되어 확장성이 높다.

**한계 및 논의사항:**

- **GZSL의 Seen-class Bias:** GZSL 결과에서 나타나듯, 모델이 여전히 seen 클래스로 예측하려는 경향이 강하다. 저자들은 이를 해결하기 위해 seen 클래스의 가상 샘플 수를 조절하는 $m_S$ 파라미터를 도입했으나, 근본적인 편향 제거를 위한 추가 연구가 필요해 보인다.
- **계산 복잡도:** 생성 모델과 FSL 분류기를 동시에 학습시켜야 하므로, 단순한 다단계 학습보다 학습 시간이 더 소요될 가능성이 크다.

## 📌 TL;DR

Z2FSL은 Generative ZSL의 생성 모델과 FSL 분류기를 결합하여, **ZSL 문제를 FSL 문제로 변환해 end-to-end로 학습**하는 프레임워크이다. 생성 모델이 만든 가상 데이터를 FSL의 Support Set으로 활용하며, 분류기의 손실 함수를 생성 모델에 역전파함으로써 **분류에 최적화된 변별력 있는 샘플을 생성**하게 한다. 실험적으로 SUN 데이터셋의 ZSL에서 SOTA를 달성했으며, 사전 학습된 FSL 분류기의 결합이 성능 향상의 핵심임을 밝혔다. 향후 GZSL의 클래스 편향 문제를 해결하는 연구로 확장될 가능성이 높다.
