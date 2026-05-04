# AGAD: Adversarial Generative Anomaly Detection

Jian Shi, Ni Zhang (2023)

## 🧩 Problem to Solve

본 논문은 시각적 이상치 탐지(Visual Anomaly Detection) 분야에서 발생하는 **이상치 데이터의 희소성과 다양성 문제**를 해결하고자 한다. 일반적으로 이상치 데이터는 획득 비용이 높고 그 형태가 매우 다양하여, 대규모의 학습 데이터를 구축하는 것이 매우 어렵다.

기존의 준지도 학습(Semi-supervised) 방식은 정상 데이터만을 사용하여 정상 분포를 학습하고, 여기서 벗어난 데이터를 이상치로 간주한다. 그러나 이러한 방식은 정상 데이터를 너무 완벽하게 재구성하도록 학습되어, 정교한 이상치까지 정상으로 오인하는 낮은 재현율(Recall) 문제를 보인다. 반면, 지도 학습(Supervised) 방식은 이상치 데이터가 필요하지만, 수집된 소량의 데이터만으로는 모든 종류의 이상치를 커버할 수 없다는 한계가 있다.

따라서 본 논문의 목표는 **정상 데이터로부터 문맥적 적대적 정보(Contextual Adversarial Information)를 생성하여, 소량의 이상치 데이터만으로도(또는 없이도) 강건한 이상치 탐지가 가능한 데이터 효율적인 프레임워크를 제안하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 생성적 적대 신경망(GAN) 구조를 활용하여 **가상 이상치(Pseudo-anomaly) 데이터를 생성하고, 이를 통해 정상과 이상의 차이를 극대화하는 판별적 특징을 학습**하는 것이다.

주요 기여 사항은 다음과 같다:

- **문맥적 적대적 정보(Contextual Adversarial Information) 도입**: 정상 데이터를 재구성한 결과물을 가상 이상치로 간주하고, 이를 다시 재구성할 때는 실패하도록 학습시킴으로써 정상과 이상의 경계를 명확히 하는 판별적 특징을 학습한다.
- **통합된 이상치 탐지 패러다임 제안**: 준지도 학습(정상 데이터만 사용)과 지도 학습(소량의 이상치 데이터 포함) 시나리오 모두에 적용 가능한 단일 프레임워크를 구축하였다.
- **높은 데이터 효율성 입증**: 5% 이하의 매우 적은 양의 이상치 데이터만으로도 탐지 정확도를 대폭 향상시킬 수 있음을 보였다.

## 📎 Related Works

이상치 탐지 연구는 크게 다음과 같은 흐름으로 발전해 왔다:

- **재구성 기반 방법 (Reconstruction-based)**: AnoGAN, GANomaly, Skip-GANomaly 등이 있으며, 정상 데이터의 재구성 능력을 높여 재구성 오차로 이상치를 판단한다. 하지만 이들은 '정상'의 재구성에만 집중할 뿐, '이상치'가 무엇인지에 대한 인식(Anomaly-awareness)이 부족하다는 한계가 있다.
- **지도 학습 기반 방법 (Supervised)**: Deep SAD, TLSAD, ESAD 등이 있으며, 불균형 분류 문제로 접근한다. 하지만 실제 환경에서는 라벨링된 이상치 데이터를 충분히 확보하기 어렵다.
- **대조 학습 (Contrastive Learning)**: SimCLR, MoCo 등 최근의 자가 지도 학습 성과를 통해 판별적 특징 학습의 중요성이 대두되었으며, 본 논문은 이를 GAN과 결합하여 가상 이상치를 생성하는 방향으로 발전시켰다.

## 🛠️ Methodology

### 전체 시스템 구조

AGAD는 Generator $G$와 Discriminator $D$로 구성된 GAN 구조를 따른다.

- **Generator ($G$)**: 입력을 받아 재구성된 이미지를 생성한다. (UNet++ 또는 Naive Encoder-Decoder 사용)
- **Discriminator ($D$)**: 입력 이미지와 재구성된 이미지의 도메인을 구분하며, 동시에 이미지 데이터를 잠재 공간(Latent space) $z$로 압축하는 특징 추출기 역할을 수행한다.

### 핵심 메커니즘: 가상 이상치 생성

AGAD의 핵심은 Generator가 정상 데이터 $x$를 $\hat{x}$로 잘 재구성하되, **재구성된 $\hat{x}$를 다시 재구성하는 과정($\hat{x} \to \hat{x}'$)은 실패하게 만드는 것**이다. 즉, $\hat{x}$를 가상 이상치로 정의하고, 정상-가상 이상치 간의 거리를 멀게 하여 모델이 이상치의 특징을 스스로 학습하게 한다.

### 손실 함수 (Loss Functions)

모델은 다음과 같은 네 가지 손실 함수의 가중 합으로 학습된다.

1. **Adversarial Loss ($\mathcal{L}_{adv}$)**: $G$가 $D$를 속일 수 있도록 사실적인 이미지를 생성하게 한다.
   $$\mathcal{L}_{adv} = \mathbb{E}_{x \sim p_x} [\log D(x)] + \mathbb{E}_{x \sim p_x} [\log(1 - D(\hat{x}))]$$
2. **Contextual Loss ($\mathcal{L}_{con}$)**: 픽셀 수준에서 입력과 재구성 이미지의 유사도를 높이기 위한 $L_1$ 손실이다.
   $$\mathcal{L}_{con} = \mathbb{E}_{x \sim p_x} |x - \hat{x}|_1$$
3. **Latent Loss ($\mathcal{L}_{lat}$)**: 잠재 공간에서의 재구성 능력을 보장하기 위한 $L_2$ 손실이다.
   $$\mathcal{L}_{lat} = \mathbb{E}_{x \sim p_x} \|D_z(x) - D_z(\hat{x})\|_2^2$$
4. **Contextual Adversarial Loss ($\mathcal{L}_{adcon}$)**: 가상 이상치 $\hat{x}$의 재구성 오차를 최대화하여 판별력을 높인다. (음수 $L_1$ 손실로 구현)
   $$\mathcal{L}_{adcon} = -\mathbb{E}_{x \sim p_x} |\hat{x} - G(\hat{x})|_1$$

### 학습 절차 및 최종 손실

입력 데이터가 정상($y=0$)인지 이상($y=1$)인지에 따라 서로 다른 손실 함수를 적용한다.

- **Normality Loss ($\mathcal{L}_n$)**: 정상 데이터의 재구성 품질을 높이고, 가상 이상치의 재구성은 실패하도록 유도한다.
  $$\mathcal{L}_n = \lambda_{adv}\mathcal{L}_{adv} + \lambda_{con}\mathcal{L}_{con} + \lambda_{adcon}\mathcal{L}_{adcon} + \lambda_{lat}\mathcal{L}_{lat}$$
- **Anomaly Loss ($\mathcal{L}_a$)**: 실제 이상치 데이터의 재구성 오차를 최대화하여 재구성을 실패하게 만든다.
- **Final Loss**: $$\mathcal{L} = y\mathcal{L}_a + (1-y)\mathcal{L}_n$$

### 추론 단계 (Inference)

추론 시에는 입력 이미지 $x$와 재구성된 이미지 $\hat{x}$ 사이의 $L_2$ norm을 기반으로 이상치 점수(Anomaly Score)를 계산한다.
$$S(x) = \|x - \hat{x}\|_2$$

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 및 실제 의료 데이터셋(Alzheimer's MRI, Chest X-ray, Lung Histopathology, Retinal OCT).
- **지표**: AUROC (%)를 사용하여 성능을 측정하였다.
- **비교 대상**: AnoGAN, OCGAN, GeoTrans, ARNet, ADGAN, Deep SAD, TLSAD, ESAD 등.

### 주요 결과

1. **준지도 학습 시나리오**: 모든 벤치마크 데이터셋에서 기존 SOTA 모델보다 높은 성능을 보였다. 특히 CIFAR-100에서 가장 큰 향상(12.8%p)이 관찰되었다.
2. **제한적 지도 학습 시나리오**: 이상치 데이터의 비율 $\gamma$를 높일수록 성능이 향상되었으며, 특히 **$\gamma \le 5\%$의 소량 데이터만으로도 성능이 급격히 상승**하여 매우 효율적인 학습이 가능함을 입증하였다.
3. **의료 데이터셋**: 실제 의료 영상에서도 강건한 성능을 보였으며, 특히 정상과 이상의 차이가 미세한 Retinal OCT 데이터셋에서는 문맥적 적대적 정보(Contextual Adversarial Information)가 학습 성공의 핵심적인 역할을 수행하였다.
4. **정성적 분석**: GANomaly는 이상치까지 너무 잘 재구성하는 경향이 있는 반면, AGAD는 정상 데이터는 잘 재구성하고 이상치 데이터는 의도적으로 재구성을 실패하게 만들어 명확한 구분 능력을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구의 가장 큰 강점은 **'이상치가 부족하다면, 정상 데이터를 이용해 가상의 이상치를 만들어 학습시키면 된다'**는 직관을 성공적으로 구현한 점이다. 이를 통해 준지도 학습의 한계인 '이상치 인식 부족' 문제를 해결하고, 지도 학습의 한계인 '데이터 희소성' 문제를 동시에 극복하였다.

### 한계 및 논의 사항

- **시각적 설명력의 한계**: 재구성 기반 방법론의 특성상 시각적 설명력이 기대되지만, 실제 결과물에서 세부적인 이상치 특징이 이미지로 아주 명확하게 드러나지는 않았다. 향후 더 정밀한 이상치 특징을 시각화하는 연구가 필요하다.
- **백본 네트워크의 영향**: 단순한 데이터셋(MNIST)에서는 Naive Encoder-Decoder가 UNet++보다 좋은 성능을 보였는데, 이는 UNet의 Skip-connection이 너무 강력하여 단순 데이터의 이상치까지 쉽게 재구성해버리는 '시맨틱 갭(Semantic gap)' 문제 때문으로 분석된다.

## 📌 TL;DR

AGAD는 정상 데이터를 이용해 **가상 이상치(Pseudo-anomaly)를 생성**하고, 이를 정상 데이터와 대조하여 학습하는 **문맥적 적대적 학습 프레임워크**이다. 이 방법은 이상치 데이터가 거의 없는 환경에서도 매우 효율적으로 작동하며, 특히 **5% 이하의 적은 이상치 데이터만으로도 탐지 성능을 극대화**할 수 있어 데이터 수집이 어려운 의료 및 산업 현장에 적용 가능성이 매우 높다.
