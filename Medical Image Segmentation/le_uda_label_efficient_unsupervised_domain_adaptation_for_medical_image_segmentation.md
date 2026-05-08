# LE-UDA: Label-efficient unsupervised domain adaptation for medical image segmentation

Ziyuan Zhao, Fangcheng Zhou, Kaixin Xu, Zeng Zeng, Cuntai Guan, S. Kevin Zhou (2022)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델을 실무에 적용할 때 직면하는 두 가지 핵심적인 문제는 대규모의 정밀한 라벨링 데이터셋 확보의 어려움과 도메인 간의 전이 성능 저하(Domain Shift)이다. 특히 픽셀 수준의 주석(Pixel-level annotation)은 전문의의 많은 시간과 비용을 요구하며, MRI에서 CT로 또는 그 반대로 다른 모달리티(Modality) 간의 데이터를 사용할 때 발생하는 심각한 도메인 차이는 모델의 일반화 성능을 크게 떨어뜨린다.

기존의 Unsupervised Domain Adaptation (UDA) 기법들은 타겟 도메인의 라벨이 없는 상황을 해결하기 위해 풍부한 소스 도메인의 라벨링 데이터를 활용하여 도메인 간의 간극을 줄이는 방식을 사용한다. 그러나 실제 임상 환경에서는 소스 도메인조차 라벨이 부족한 **소스 라벨 희소성(Source label scarcity)** 문제가 발생할 수 있다. 기존 UDA 방법들은 풍부한 소스 라벨이 있다는 가정하에 설계되었기 때문에, 소스 라벨이 부족할 경우 결정 경계(Decision boundary)가 모호해지며 도메인 정렬(Domain alignment) 성능이 급격히 저하되는 한계가 있다. 따라서 본 논문은 소스 라벨이 제한적인 상황에서 타겟 도메인으로의 효과적인 전이를 달성하는 Label-Efficient UDA라는 새로운 과제를 해결하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 수준의 적응(Image adaptation)과 특징 수준의 적응(Feature adaptation)을 결합하고, 이를 준지도 학습(Semi-supervised learning)의 Self-ensembling 구조와 통합하여 소스 라벨의 부족함을 극복하는 것이다.

주요 기여 사항은 다음과 같다:

1. **소스 라벨 희소성 시나리오 정의**: 소스 데이터의 라벨이 제한적인 실제적인 UDA 상황을 정의하고, 이를 해결하기 위한 LE-UDA 프레임워크를 제안하였다.
2. **Dual-Domain Consistency 패러다임**: Self-ensembling 기반의 Dual-teacher 네트워크를 구축하여, 도메인 내(Intra-domain) 지식과 도메인 간(Inter-domain) 지식을 동시에 추출함으로써 라벨 효율성을 극대화하였다.
3. **Dual Self-Ensembling Adversarial Learning**: 특징 공간에서의 정렬을 강화하기 위해 Discriminator를 Teacher-Student 구조에 통합하여, 학생 네트워크가 보다 일반화된 표현(Generalizable representations)을 학습하도록 유도하였다.

## 📎 Related Works

### Unsupervised Domain Adaptation (UDA)

UDA는 타겟 도메인의 라벨 없이 소스 도메인의 지식을 전이하는 기술이다. 주로 두 가지 접근 방식이 사용된다:

- **Image Adaptation**: CycleGAN 등을 이용하여 소스 이미지를 타겟 스타일로 변환하여 학습한다.
- **Feature Adaptation**: Domain Adversarial Neural Network (DANN) 등과 같이 특징 공간에서 도메인 구분 불가능한 특징을 추출하도록 학습한다.
- **SIFA**와 같은 최신 기법은 이미지와 특징 적응을 동시에 수행하여 성능을 높였으나, 여전히 소스 도메인에 많은 양의 라벨이 필요하다는 한계가 있다.

### Semi-Supervised Learning (SSL)

라벨이 부족한 상황을 해결하기 위해 Unlabeled data를 활용하는 SSL 연구가 활발하다. 특히 **Mean Teacher (MT)** 프레임워크는 학생 모델의 가중치를 지수 이동 평균(EMA)으로 업데이트한 교사 모델을 두고, 두 모델 간의 예측 일관성(Consistency)을 강제함으로써 성능을 높이는 방식이다. 하지만 대부분의 SSL 연구는 단일 도메인 상황만을 가정하며, 도메인 전이(Domain shift) 상황은 고려하지 않았다.

## 🛠️ Methodology

LE-UDA는 크게 세 가지 단계의 파이프라인으로 구성된다.

### 1. Dual Cycle Alignment Module (DCAM)

이미지 수준의 도메인 간극을 줄이기 위해 CycleGAN 기반의 DCAM을 사용한다.

- 두 개의 생성자 $G_s, G_t$와 판별자 $D_s, D_t$를 사용하여 소스 도메인($D_s$)과 타겟 도메인($D_t$) 간의 양방향 이미지 변환을 수행한다.
- **Adversarial Loss**: 생성된 이미지가 실제 도메인의 이미지와 유사하도록 학습한다.
  $$L_{t}^{gan}(G_t, D_t) = \mathbb{E}_{x_t \sim D_t}[\log D_t(x_t)] + \mathbb{E}_{x_s \sim D_s}[\log (1 - D_t(G_t(x_s)))]$$
- **Cycle-consistency Loss**: 변환된 이미지를 다시 원래 도메인으로 되돌렸을 때 원래 이미지와 동일해야 함을 강제하는 $L_1$ 손실을 사용한다.
  $$L_{cyc}(G_s, G_t) = \mathbb{E}_{x_s \sim D_s}[\|G_s(G_t(x_s)) - x_s\|_1] + \mathbb{E}_{x_t \sim D_t}[\|G_t(G_s(x_t)) - x_t\|_1]$$
이를 통해 소스 유사 도메인($D_s^s$)과 타겟 유사 도메인($D_s^t$)이라는 보완적 데이터를 생성한다.

### 2. Dual-Domain Knowledge Transfer

두 개의 Mean-teacher 네트워크를 구축하여 지식을 전이한다. 교사 모델의 가중치 $\theta'_t$는 학생 모델 $\theta_t$의 EMA로 업데이트된다: $\theta'_t = \alpha\theta'_{t-1} + (1-\alpha)\theta_t$.

- **Intra-domain Transfer**: 소스 데이터와 소스 유사 데이터를 사용하여 도메인 내부의 일관성을 학습한다. 학생과 교사 모델의 예측 결과 간의 MSE 손실을 최소화한다.
  $$L_{intra}^{con} = \frac{1}{N} \sum_{i=1}^{N} \|f(x_i; \theta'_t, \xi') - f(x_i; \theta_t, \xi)\|^2$$
- **Inter-domain Transfer**: 서로 다른 모달리티 간의 구조적 유사성을 활용한다. 섀넌 엔트로피(Shannon entropy)를 이용하여 생성된 엔트로피 맵 간의 일관성을 강제한다.
  $$L_{inter}^{con} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{H \times W} \sum_{v=1}^{V} \| I_{s,i,v} - I_{t,i,v} \|^2$$
  여기서 $I_{s,i,v}$는 학생 모델의 $v$번째 픽셀에 대한 가중 자기 정보(Weighted self-information)이다.

### 3. Self-Ensembling Adversarial Learning

특징 수준의 정렬을 위해 $\text{Discriminator}$를 도입한다.

- **Intra-domain Discriminator ($D_{intra}$)**: 학생 모델 $G$와 내부 교사 모델 $G_{intra}$가 생성한 특징을 구분하려 하며, 학생 모델은 이를 속이도록 학습하여 $G_{intra}$의 특징 분포를 따라가게 된다.
- **Inter-domain Discriminator ($D_{inter}$)**: 학생 모델 $G$와 외부 교사 모델 $G_{inter}$ 사이의 특징을 정렬한다.
- **Multi-level Alignment**: U-Net의 디코더 단계의 서로 다른 레이어(예: 1번째, 3번째 블록)에 보조 판별자를 연결하여 저수준 특징부터 고수준 특징까지 다단계로 정렬을 수행한다.

### 전체 학습 목표

최종적으로 학생 네트워크는 다음의 통합 손실 함수를 통해 학습된다:
$$L_{stu} = L_{seg}^{stu} + \lambda_{intra}^{con}L_{intra}^{con} + \lambda_{intra}^{adv}L_{intra}^{adv} + \lambda_{inter}^{con}L_{inter}^{con} + \lambda_{inter}^{adv}L_{inter}^{adv}$$
여기서 $L_{seg}^{stu}$는 라벨링된 소스 데이터에 대한 Cross-Entropy 및 Dice Loss의 합이다.

## 📊 Results

### 실험 설정

- **데이터셋**: 심장 하부 구조 분할(MM-WHS 2017) 및 복부 다기관 분할(MICCAI 2015, ISBI 2019) 데이터셋을 사용하였다.
- **시나리오**: MRI $\rightarrow$ CT 및 CT $\rightarrow$ MRI 양방향 전이를 수행하였으며, 소스 라벨 비율을 $25\%$로 설정하여 라벨 희소성 상황을 시뮬레이션하였다.
- **지표**: Dice coefficient (높을수록 좋음)와 ASD (Average Symmetric Surface Distance, 낮을수록 좋음)를 사용하였다.

### 주요 결과

- **심장 분할 (MRI $\rightarrow$ CT)**: 소스 라벨 $25\%$ 상황에서 LE-UDA는 평균 Dice $70.8\%$를 기록하여, 기존 UDA 방법들(ADDA, CycleGAN, SIFA)보다 월등히 높은 성능을 보였다. 특히 이전 연구인 MT-UDA보다 약 $3\%$ 향상된 성능을 보였다.
- **복부 기관 분할 (CT $\rightarrow$ MRI)**: LE-UDA는 $87.7\%$의 Dice 점수를 달성하였으며, 이는 타겟 라벨 $25\%$만 사용하여 학습한 "Supervised-only" 상한선($88.5\%$)에 근접한 수치이다.
- **라벨 비율에 따른 분석**: 소스 라벨이 극도로 적은 'One-shot UDA' 상황에서도 LE-UDA는 다른 방법들에 비해 강건하게 구조적 형태를 유지하는 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

LE-UDA의 우수한 성능은 **이미지 적응과 특징 적응의 상호 보완적 작용**에서 기인한다. 이미지 적응(CycleGAN)은 도메인 간극을 좁혀주지만 라벨이 적을 때 오버피팅되기 쉽고, 특징 적응(Adversarial)은 특징 공간을 다양화하여 일반화 성능을 높인다. 여기에 Self-ensembling 기반의 Dual-teacher 구조가 unlabeled 데이터를 효율적으로 활용하여 소스 라벨 부족으로 인한 결정 경계의 모호함을 보정하였다.

### 한계 및 향후 과제

- **GAN의 불확실성**: CycleGAN을 통해 생성된 일부 이미지에서 아티팩트(Artifacts)나 해부학적 구조 누락(예: 상행 대동맥 소실)이 발견되었다. 이러한 불일치하는 이미지 쌍이 구조적 지식 전이에 부정적인 영향을 줄 수 있다.
- **계산 복잡도**: 다수의 교사 네트워크와 판별자를 사용하므로 메모리 소모가 크다. 현재는 2D U-Net을 사용하였으나, 3D 네트워크로 확장할 경우 심각한 메모리 문제가 예상되며 이에 대한 최적화가 필요하다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 **소스 도메인의 라벨이 부족한 상황에서의 비지도 도메인 적응(UDA)** 문제를 해결하기 위한 **LE-UDA** 프레임워크를 제안한다. CycleGAN을 이용한 이미지 생성, Dual-teacher 기반의 도메인 내/간 지식 전이, 그리고 특징 수준의 Adversarial Learning을 결합하여 소스 라벨이 $25\%$만 있는 상황에서도 타겟 도메인에서 높은 분할 성능을 달성하였다. 이 연구는 데이터 확보가 어려운 실제 의료 환경에서 소량의 라벨만으로도 효과적인 교차 모달리티 모델을 구축할 수 있는 가능성을 제시한다.
