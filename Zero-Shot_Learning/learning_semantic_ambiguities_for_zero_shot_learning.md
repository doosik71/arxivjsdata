# Learning Semantic Ambiguities for Zero-Shot Learning

Celina Hanouti and Hervé Le Borgne (2022)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL) 및 Generalized Zero-Shot Learning (GZSL)에서 발생하는 **Seen 클래스에 대한 편향(Bias)** 문제를 해결하고자 한다. ZSL의 목적은 학습 단계에서 시각적 샘플이 전혀 없는 Unseen 클래스를 인식하는 것이며, 이를 위해 일반적으로 각 클래스의 시각적 특징과 시맨틱 설명(Semantic description/prototype) 간의 매핑을 학습한다.

최근의 state-of-the-art 접근 방식들은 생성 모델(Generative models)을 사용하여 클래스 프로토타입으로부터 시각적 특징을 합성하고, 이를 통해 감독 학습(Supervised learning) 방식으로 분류기를 학습시킨다. 그러나 이러한 방식들은 학습 과정에서 Seen 클래스의 시각적 인스턴스만을 활용하기 때문에, 테스트 단계에서 모델이 예측 결과를 Seen 클래스로 치우치게 만드는 강한 편향성을 보인다. 특히 GZSL 환경에서는 Seen 클래스와 Unseen 클래스를 동시에 식별해야 하므로 이러한 편향 문제가 더욱 심각하게 나타난다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **시맨틱 공간에서의 모호성(Semantic Ambiguities)을 학습**함으로써 생성 모델의 정규화(Regularization)를 수행하는 것이다.

기존의 Mixup 기법이 이미지와 라벨을 동시에 보간(Interpolation)하는 것과 달리, 본 연구는 오직 **시맨틱 프로토타입(Semantic prototype)만을 보간하여 '가상 모호 클래스(Virtual ambiguous classes)'를 생성**한다. 생성 모델이 이러한 가상 클래스에 대한 특징을 합성하고 이를 올바르게 분류하도록 학습함으로써, 클래스 간의 경계를 더 명확히 구분하는 능력을 배양하고 결과적으로 Unseen 클래스에 대한 판별력을 높여 Seen 클래스에 대한 편향을 완화한다.

## 📎 Related Works

**1. 생성 기반 ZSL 접근 방식**
최근 ZSL 연구들은 VAE(Variational Autoencoders)나 GAN(Generative Adversarial Networks)을 사용하여 시맨틱 특징으로부터 시각적 특징을 생성한다. 대표적으로 f-VAEGAN-D2는 VAE의 디코더와 GAN의 생성자 가중치를 공유하며, 전이적 설정(Transductive setting)에서 라벨이 없는 Unseen 샘플을 활용하여 성능을 높인다. TF-VAEGAN은 여기에 시맨틱 프로토타입을 재구성하는 디코더와 피드백 루프를 추가하여 생성된 특징을 정교화한다.

**2. Mixup 및 보간 기반 정규화**
Zhang et al. [34]이 제안한 Mixup은 훈련 데이터의 쌍을 볼록 조합(Convex combination)하여 가상 샘플을 만드는 데이터 증강 기법이다. Chou et al. [7]은 이를 ZSL에 적용하여 시각적 샘플과 시맨틱 프로토타입을 모두 보간하였다.

**3. 기존 방식과의 차별점**
본 논문은 Chou et al. [7]과 달리 시각적 샘플의 보간을 제외하고 **시맨틱 공간에서의 보간**에만 집중한다. 저자들은 픽셀 수준에서의 이미지 보간은 현실적인 의미를 갖기 어렵지만, 시맨틱 속성의 보간은 의미론적으로 타당할 수 있다고 주장한다. 또한, 가상 샘플을 단순히 데이터 증강으로 사용하는 것이 아니라, 생성 모델이 모호한 클래스를 구분하도록 만드는 정규화 손실 함수로 활용한다는 점에서 차이가 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

본 제안 방법은 기존의 조건부 생성 모델(Conditional generative model)에 추가적인 정규화 손실 함수를 통합하는 형태이다. 기본 구조는 $f\text{-VAEGAN-D2}$ 또는 $TF\text{-VAEGAN}$과 같은 모델을 기반으로 하며, 생성자 $G$가 시맨틱 프로토타입 $c$와 잠재 코드 $z$를 입력받아 시각적 특징 $\hat{x}$를 생성하는 구조를 가진다.

### 가상 모호 클래스의 생성

학습 과정에서 두 개의 실제 시맨틱 프로토타입 $c(y_i)$와 $c(y_j)$를 선택하여 다음과 같이 보간된 가상 프로토타입 $\tilde{c}$와 가상 라벨 $\tilde{y}$를 생성한다.

$$\tilde{c} = \lambda c(y_i) + (1 - \lambda)c(y_j)$$
$$\tilde{y} = \lambda y_i + (1 - \lambda)y_j$$

여기서 $\lambda$는 보간 비율을 결정하는 하이퍼파라미터이며, 고정된 값 또는 특정 분포에서 추출된 랜덤 변수일 수 있다.

### 학습 목표 및 손실 함수

생성자 $G$는 위에서 생성된 가상 프로토타입 $\tilde{c}$와 노이즈 $z \sim \mathcal{N}(0, 1)$를 입력받아 가상 특징 $\hat{x}$를 합성한다.

$$\hat{x} = G([z; \tilde{c}])$$

이후 합성된 특징 $\hat{x}$를 분류기 $f$에 입력하여 가상 라벨 $\tilde{y}$와의 교차 엔트로피(Cross-entropy)를 계산하는 정규화 손실 함수 $\mathcal{L}^I$를 정의한다.

$$\mathcal{L}^I = \mathbb{E}_{z, \lambda} [l(f(\hat{x}), \tilde{y})]$$

$f\text{-VAEGAN-D2}$에 이를 통합할 경우, 전체 손실 함수는 다음과 같이 구성된다.

$$\mathcal{L} = \mathcal{L}_{BCE} + \gamma \mathcal{L}^s_{WGAN} + \mathcal{L}^u_{WGAN} + \mathcal{L}^I$$

여기서 $\gamma$는 $\mathcal{L}^s_{WGAN}$의 가중치를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: CUB, SUN, FLO, AwA2 등 4개의 벤치마크 데이터셋을 사용하였다.
- **지표**: ZSL에서는 Top-1 정확도($T1$)를, GZSL에서는 Seen($s$)과 Unseen($u$) 클래스 정확도의 조화 평균(Harmonic Mean, $H$)을 측정하였다. 또한, 여러 데이터셋의 성능을 통합 평가하기 위해 $\text{mNRG}$ (median normalized relative gain) 지표를 도입하였다.
- **기준 모델**: $\text{CLSWGAN}$, $f\text{-VAEGAN-D2}$, $TF\text{-VAEGAN}$, $\text{LisGAN}$ 등과 비교하였다.

### 주요 결과

1. **ZSL 및 GZSL 성능 향상**: 제안 방법은 Inductive 및 Transductive 설정 모두에서 기존 모델들보다 우수한 성능을 보였다. 특히 Inductive GZSL 설정의 CUB 데이터셋에서 기존 $56.9\%$에서 $67.2\%$로 정확도를 크게 향상시켰다.
2. **편향 완화**: GZSL 결과에서 Unseen 클래스의 정확도가 크게 상승한 것을 확인할 수 있으며, 이는 본 방법이 Seen 클래스로의 편향을 효과적으로 줄였음을 시사한다.
3. **보간 비율 $\lambda$의 영향**: 실험 결과 $\lambda = 0.5$ 또는 $\lambda \sim \mathcal{N}(0.5, 0.25)$일 때 가장 좋은 성능을 보였다. 이는 기존 Mixup( $\lambda$가 0이나 1에 가까울 때 유리)과 대조적이며, 실제 클래스와 완전히 구별되는 '모호한' 영역을 학습하는 것이 정규화에 더 효과적임을 의미한다.
4. **프로토타입 구성의 영향**: Transductive 설정에서 Unseen 프로토타입만을 사용하여 정규화했을 때보다 Seen과 Unseen 프로토타입을 모두 사용하여 학습했을 때 성능이 더 높게 나타났다.

## 🧠 Insights & Discussion

**강점**
본 논문은 시맨틱 공간에서의 보간이라는 단순하면서도 강력한 정규화 방법을 제안하였다. 특히 이미지 수준의 보간이 갖는 비현실성을 제거하고, 생성 모델이 클래스 간의 전이 영역(Transition regions)을 학습하게 함으로써 결정 경계를 명확히 했다는 점이 긍정적이다. 또한, 특정 모델에 종속되지 않고 다양한 조건부 생성 모델에 적용 가능한 범용적인 손실 함수라는 점이 큰 장점이다.

**한계 및 해석**
실험 결과에서 Unseen 클래스의 성능은 크게 향상되었으나, GZSL에서 여전히 Seen 클래스의 정확도가 Unseen보다 높게 나타나는 경향이 있다. 이는 제안 방법이 편향을 완화시키기는 하지만, 완전히 제거하지는 못했음을 보여준다. 또한, 본 연구는 선형 보간(Linear interpolation)만을 사용하였는데, 저자들은 향후 비선형 보간이나 세 개 이상의 클래스를 조합하는 방식의 탐색이 필요함을 언급하고 있다.

**비판적 논의**
본 방법은 생성 모델의 정규화를 통해 간접적으로 분류 성능을 높이는 방식이다. 따라서 생성 모델 자체의 품질(예: 생성된 특징의 다양성)에 의존적일 수밖에 없다. 하지만 시맨틱 공간의 '빈 공간(Empty parts)'을 가상 클래스로 채움으로써 잠재 공간(Latent space)을 더 효율적으로 활용하게 만들었다는 점은 이론적으로 타당한 접근으로 판단된다.

## 📌 TL;DR

본 논문은 ZSL/GZSL의 고질적인 문제인 Seen 클래스 편향을 해결하기 위해, 시맨틱 프로토타입을 보간하여 만든 **가상 모호 클래스(Virtual ambiguous classes)를 학습하는 정규화 방법**을 제안한다. 생성 모델이 이 가상 클래스들을 구분하도록 학습함으로써 클래스 간 판별력을 높였으며, 그 결과 GZSL에서 Unseen 클래스의 인식 성능을 유의미하게 향상시켰다. 이 연구는 향후 멀티모달 엔티티 링킹이나 교차 모달 검색 등 모호한 시맨틱-시각 정보 관계를 다루는 다양한 분야에 응용될 가능성이 높다.
