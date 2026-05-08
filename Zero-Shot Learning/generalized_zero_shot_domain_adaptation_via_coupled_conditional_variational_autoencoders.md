# Generalized Zero-Shot Domain Adaptation via Coupled Conditional Variational Autoencoders

Qian Wang, Toby P. Breckon (2020)

## 🧩 Problem to Solve

본 논문은 **Generalized Zero-Shot Domain Adaptation (GZSDA)**라는 새로운 문제 정의와 이를 해결하기 위한 방법론을 제시한다.

일반적인 Domain Adaptation (DA)은 소스 도메인(Source Domain)의 풍부한 레이블 데이터를 활용하여 타겟 도메인(Target Domain)의 학습 문제를 해결하는 것을 목표로 한다. 하지만 기존의 DA 연구들은 타겟 도메인의 모든 클래스에 대해 데이터가 존재한다고 가정하거나, 일부 클래스에만 레이블이 있는 경우(Semi-supervised)를 다루었다. 반면, 실제 환경에서는 타겟 도메인에서 일부 클래스의 데이터는 아예 존재하지 않는 **Unseen Classes** 상황이 빈번하게 발생한다.

Zero-Shot Learning (ZSL) 역시 Unseen Classes를 다루지만, 소스 도메인에서 이미지 자체가 아닌 클래스 수준의 시맨틱 표현(Semantic Representation, 예: 속성 벡터나 워드 벡터)만을 제공한다는 점에서 차이가 있다. 따라서 본 논문은 다음과 같은 구체적인 제약 조건을 가진 GZSDA 문제를 해결하고자 한다.

1. **소스 도메인($S$):** 모든 클래스($Y$)에 대해 레이블이 지정된 샘플이 존재한다.
2. **타겟 도메인($T$):** 클래스의 부분 집합인 Seen Classes($Y_{seen} \subset Y$)에 대해서만 레이블이 지정된 샘플이 존재하며, Unseen Classes($Y_{unseen}$)에 대한 샘플은 전혀 없다.
3. **목표:** 타겟 도메인에서 입력된 샘플이 Seen Class인지 Unseen Class인지 구분하여 정확하게 분류하는 추론 모델 $y = f(x) \in Y$를 학습하는 것이다.

이 문제는 데이터의 불균형(Imbalance)과 도메인 간 분포 차이(Domain Shift)로 인해 모델이 Seen Classes나 소스 도메인에 편향(Bias)되기 쉬운 매우 도전적인 과제이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Coupled Conditional Variational Autoencoder (CCVAE)**를 통해 타겟 도메인의 Unseen Classes에 해당하는 **합성 특징(Synthetic Features)**을 생성하여 학습 데이터의 불균형을 해소하는 것이다.

단순히 클래스 평균값과 같은 시맨틱 벡터에서 특징을 생성하는 기존 ZSL 방식과 달리, CCVAE는 소스 도메인의 실제 개별 샘플들을 타겟 도메인의 특징 공간으로 투영(Projection)하여 생성한다. 이를 통해 클래스 내 분산(Intra-class variability)을 보존하면서도 도메인 특성이 반영된 특징을 만들어낼 수 있으며, 이렇게 생성된 데이터를 포함하여 일반적인 지도 학습(Supervised Learning) 방식으로 분류기를 학습시킨다.

## 📎 Related Works

본 논문은 기존 연구들의 한계를 다음과 같이 지적하며 차별성을 강조한다.

- **Domain Adaptation (UDA, SDA):** 주로 마진 분포(Marginal Distribution)를 정렬하거나 도메인 불변 표현(Domain-invariant Representation)을 학습한다. 그러나 타겟 도메인에 특정 클래스의 샘플이 완전히 결여된 ZSDA 상황에서는 클래스별 정렬(Class-wise alignment)이 불가능하므로 직접 적용할 수 없다.
- **Zero-Shot Learning (ZSL, GZSL):** GAN이나 VAE 기반의 생성 모델을 사용하여 시맨틱 속성으로부터 특징을 생성한다. 하지만 시맨틱 표현(속성 벡터 등)은 클래스 내의 다양성을 충분히 담아내지 못하는 한계가 있으며, 소스 도메인이 이미지 형태의 데이터인 본 문제의 설정과는 modality가 다르다.
- **Zero-Shot Domain Adaptation (ZSDA):** 일부 연구가 진행되었으나, 타겟 도메인의 Unseen Classes에 대해서만 인식을 허용하거나, 소스-타겟 도메인 간에 쌍을 이루는(Paired) 데이터가 필요하다는 제약이 있었다. 본 논문은 이러한 제약을 제거하고 Seen/Unseen 클래스를 모두 인식하는 'Generalized' 설정을 도입하였다.

## 🛠️ Methodology

본 논문이 제안하는 전체 파이프라인은 총 3단계로 구성된다.

### 1. Feature Extraction (특징 추출)

이미지 픽셀 공간에서 직접 생성 모델을 학습시키는 것은 복잡도가 너무 높기 때문에, 특징 공간(Feature Space)에서 작업을 수행한다. 공유된 CNN 모델(ResNet50 또는 AlexNet)을 사용하여 소스 및 타겟 도메인의 이미지에서 특징 벡터 $x$를 추출한다.

### 2. Coupled Conditional Variational Autoencoder (CCVAE)

CCVAE는 소스 도메인과 타겟 도메인의 특징을 상호 변환하고 생성하기 위한 모델이다.

- **구조:** 두 도메인을 위한 CVAE가 쌍(Coupled)을 이루고 있으며, 가중치를 공유(Weight Sharing)하여 하나의 통합된 모델로 작동한다.
- **입력:** 특징 벡터 $x$와 해당 샘플의 도메인 레이블 $c(x) \in \{s, t\}$ (one-hot vector)를 결합하여 인코더에 입력한다.
- **인코더 및 디코더:** 인코더는 잠재 공간(Latent Space)의 분포 $q(z|x, c) = \mathcal{N}(\mu_x, \Sigma_x)$를 추정하고, 여기서 샘플링된 $z$는 디코더를 통해 재구성된다.
- **상호 생성(Cross-generation):** 잠재 코드 $z$를 디코더에 넣을 때, 원래 입력과 **반대되는 도메인 레이블**을 조건으로 주면 타겟 도메인 샘플 $x_s$로부터 합성 타겟 특징 $\tilde{x}_{st}$를, 또는 그 반대 방향의 특징 $\tilde{x}_{ts}$를 생성할 수 있다.

**손실 함수 (Loss Function):**
CCVAE의 학습 목표는 다음과 같은 손실 함수 $L_{CCVAE}$를 최소화하는 것이다.

$$L_{CCVAE}(\Phi, \theta; x_s, x_t) = (L_{recon}(x_s, \tilde{x}_s) + L_{recon}(x_t, \tilde{x}_t)) + (L_{cross\_recon}(x_s, \tilde{x}_{ts}) + L_{cross\_recon}(x_t, \tilde{x}_{st})) + \lambda D_{KL}(\mathcal{N}(\mu_x, \Sigma_x) || \mathcal{N}(0, I))$$

- $L_{recon}$: 동일 도메인 내에서의 재구성 오차이다.
- $L_{cross\_recon}$: 서로 다른 도메인 간의 재구성 오차이다. 이는 인코더가 도메인에 관계없이 클래스 정보를 잠재 공간 $z$에 보존하도록 강제하여, 클래스 판별력이 높은 특징을 생성하게 한다.
- $D_{KL}$: 잠재 분포를 표준 정규 분포와 가깝게 유지하는 정규화 항이다.

### 3. Target Image Classification (타겟 이미지 분류)

학습된 CCVAE를 사용하여 소스 도메인의 Unseen Class 샘플들로부터 합성 타겟 특징 $\tilde{x}_{st}$를 생성한다. 최종적으로 다음과 같은 데이터를 모두 합쳐 통합 분류기(Classifier)를 학습시킨다.

- 실제 소스 도메인 데이터 ($D_s$)
- 실제 타겟 도메인 데이터 ($D_t$)
- CCVAE로 생성된 합성 데이터

## 📊 Results

### 실험 설정

- **데이터셋:**
  - **BaggageXray-20:** 항공 보안 검색 X-ray 이미지와 일반 RGB 이미지로 구성된 신규 데이터셋. (20개 클래스)
  - **Office-Home:** Art, Clipart, Product, Real-World의 4개 도메인. (65개 클래스)
  - **XMNIST:** MNIST, Fashion-MNIST, EMNIST를 활용하여 Gray, Color, Negative 도메인을 구축.
- **지표:** Seen Class 정확도($Acc_{seen}$), Unseen Class 정확도($Acc_{unseen}$), 그리고 두 값의 조화 평균인 **Harmony Mean ($H$)**을 측정한다. $H$ 값은 모델이 특정 클래스에 편향되지 않고 균형 있게 성능을 내는지를 평가하는 핵심 지표이다.
- **비교 대상:** Source Only, Baseline(1NN/NN), BiDiLEL, CADA-VAE, LPP.

### 주요 결과

1. **BaggageXray-20:** 소스 전용(Source Only) 방식은 도메인 간 차이가 커서 성능이 매우 낮았다. Baseline 방식들은 Seen 클래스 성능은 높으나 Unseen 클래스 성능이 극도로 낮아 $H$ 값이 낮게 나타났다. CCVAE는 $Acc_{unseen}$을 크게 향상시키며 가장 높은 $H$ 값(Regular $\to$ X-ray에서 34.5%, X-ray $\to$ Regular에서 58.6%)을 달성했다.
2. **Office-Home:** CCVAE가 $Acc_{unseen}$과 $H$ 지표에서 다른 비교 모델들을 일관되게 앞섰다.
3. **XMNIST:** 특히 Unseen 클래스가 많은 EMNIST 설정에서 CCVAE의 성능 우위가 뚜렷하게 나타났다.

결론적으로, CCVAE는 다양한 데이터셋과 도메인 전이 설정에서 Seen/Unseen 클래스 간의 균형을 맞추며 가장 우수한 일반화 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 차별점

본 연구의 가장 큰 강점은 **특징 수준의 생성 모델을 통해 GZSDA의 데이터 불균형 문제를 해결**했다는 점이다. 특히 CADA-VAE와 같은 기존 생성 모델과의 차별점은 다음과 같다.

- **개별 샘플 기반 생성:** 클래스 속성 벡터가 아닌 개별 샘플을 입력으로 사용하여 클래스 내 변동성을 보존한다.
- **통합 VAE 구조:** 도메인별 VAE를 따로 두지 않고 가중치를 공유함으로써 잠재 공간에서 클래스 정보가 더 잘 보존되도록 설계하였다.
- **상호 보완적 학습:** 타겟뿐만 아니라 소스 도메인의 특징도 함께 생성하여 전체 데이터셋을 증강함으로써 분류기의 강건성을 높였다.

### 한계 및 해석

논문에서 명시적으로 언급되지는 않았으나, 본 방법론은 기본적으로 고성능의 Feature Extractor(Pre-trained CNN)에 의존하고 있다. 만약 소스와 타겟 도메인의 시각적 차이가 너무 극심하여 사전 학습된 모델이 유의미한 특징을 추출하지 못할 경우, CCVAE의 생성 성능 또한 저하될 가능성이 있다. 또한, $L_{cross\_recon}$ 학습 시 Unseen 클래스에 대해 더미(Dummy) 특징을 사용한 점은 잠재적인 최적화의 한계로 작용할 수 있다.

## 📌 TL;DR

본 논문은 소스 도메인에는 모든 클래스 데이터가 있고, 타겟 도메인에는 일부 클래스 데이터만 존재하는 **Generalized Zero-Shot Domain Adaptation (GZSDA)** 문제를 정의하고, 이를 해결하기 위한 **CCVAE (Coupled Conditional Variational Autoencoder)** 모델을 제안한다. CCVAE는 소스 도메인의 샘플을 타겟 도메인의 특징으로 변환하여 Unseen Classes의 합성 데이터를 생성함으로써 학습 데이터의 불균형을 해소한다. 제안 방법은 신규 구축한 X-ray 보안 검색 데이터셋을 포함한 여러 벤치마크에서 기존 ZSL/DA 방법론보다 뛰어난 성능(특히 Harmony Mean)을 입증하였으며, 이는 항공 보안 및 자율 주행과 같이 새로운 객체가 지속적으로 등장하고 환경 변화가 심한 실제 응용 분야에 유용하게 적용될 수 있다.
