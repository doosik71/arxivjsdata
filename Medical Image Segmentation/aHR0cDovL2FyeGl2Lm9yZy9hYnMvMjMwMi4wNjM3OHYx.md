# Semantic Image Segmentation: Two Decades of Research

Gabriela Csurka, Riccardo Volpi and Boris Chidlovskii (2023)

## 🧩 Problem to Solve

본 논문은 지난 20년간의 시맨틱 이미지 세그멘테이션(Semantic Image Segmentation, SiS) 연구와 특히 최근 급격히 성장한 시맨틱 세그멘테이션을 위한 도메인 적응(Domain Adaptation for Semantic Image Segmentation, DASiS) 분야를 종합적으로 분석하는 것을 목표로 한다.

시맨틱 세그멘테이션은 이미지의 모든 픽셀에 대해 해당 픽셀이 어떤 시맨틱 클래스(예: 자동차, 보행자, 도로, 하늘 등)에 속하는지 레이블을 지정하는 작업이다. 이는 자율주행 자동차의 주변 환경 인식, 지능형 로봇의 내비게이션, 의료 영상 분석 등 광범위한 컴퓨터 비전 응용 분야에서 핵심적인 역할을 수행한다.

그러나 SiS 모델의 성능을 높이기 위해서는 대량의 픽셀 수준 주석(pixel-wise annotation) 데이터가 필요하며, 이는 이미지 분류(Image Classification) 작업에 비해 비용과 시간이 매우 많이 소요된다는 치명적인 한계가 있다. 이러한 데이터 부족 문제를 해결하기 위해 게임 엔진 등을 이용한 합성 데이터(synthetic data) 생성과 이를 실제 데이터(real data)에 적용하기 위한 도메인 적응(Domain Adaptation) 기술의 중요성이 대두되었다.

## ✨ Key Contributions

본 논문의 주요 기여는 다음과 같이 두 가지 핵심 축으로 요약된다.

1.  **SiS 연구의 포괄적 리뷰**: 초기 역사적 방법론부터 최신 딥러닝 기반 방법론, 그리고 최근의 트렌드인 Transformer 기반 모델까지 20년 간의 기술적 진화를 체계적으로 정리하였다. 특히 인코더/디코더 구조, 어텐션 메커니즘, 풀링 레이어 등의 특성에 따라 모델을 분류한 상세 표(Table 1.1)를 제공한다.
2.  **DASiS 분야의 심층 분석**: 최근 5년간 급성장한 DASiS 분야를 집중 분석하였다. 합성-실제(Sim-to-Real) 적응을 중심으로 이미지 수준, 특징 수준, 출력 수준에서의 도메인 정렬(Domain Alignment) 기법을 분류하고, 이를 보완하는 다양한 기계학습 전략(Self-training, Entropy minimization 등)을 정리하였다. 또한, 다중 소스/타겟, 소스-프리(Source-free), 도메인 일반화(Domain Generalization)와 같은 확장된 시나리오를 다룬다.

## 📎 Related Works

본 논문은 기존의 SiS 및 DA 관련 서베이 논문들과의 차별점을 명시하고 있다.

- **SiS 관련 기존 연구**: Thoma (2016), Li et al. (2018a), Zhou et al. (2018), Minaee et al. (2020) 등이 초기 방법론과 딥러닝 기반 솔루션을 다루었으나, 본 논문은 최신 Transformer 및 Attention 메커니즘을 포함한 모델들을 추가로 분석하여 범위를 확장하였다.
- **DASiS 관련 기존 연구**: Toldo et al. (2020a)가 가장 유사한 최신 서베이로 언급되지만, 본 논문은 더 최신 연구들을 포함하며, 단순한 정렬 방식뿐만 아니라 백본 네트워크, 보완 기법 등을 기준으로 한 다각도 분류 체계(Table 2.1)를 제안한다. 또한, 다중 소스/타겟 적응 및 소스-프리 적응과 같은 최신 트렌드를 보다 상세히 다루고 있다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 특정 알고리즘을 제안하기보다 기존 방법론들을 체계적으로 분류하는 프레임워크를 제시한다.

### 1. Semantic Image Segmentation (SiS) 방법론

#### 1.1 역사적 방법론 (Pre-Deep Learning)
딥러닝 이전의 방법론은 크게 세 가지 방향에 집중하였다.
- **지역 외관 모델링 (Local Appearance)**: SIFT, Textons, Fisher Vectors 등을 사용하여 픽셀 또는 패치 수준의 특징을 추출하였다.
- **일관성 강화 (Consistency)**: Markov Random Field (MRF) 또는 Conditional Random Field (CRF)를 사용하여 인접 픽셀 간의 레이블 일관성을 보장하였다.
- **사전 지식 활용 (Prior Knowledge)**: 이미지 수준의 분류 결과나 객체 형상 사전 지식을 활용하여 세그멘테이션 품질을 높였다.

#### 1.2 딥러닝 기반 방법론
- **FCN (Fully Convolutional Networks)**: 전결합층을 컨볼루션 층으로 대체하여 임의의 크기의 입력 이미지에 대해 밀집한 예측 맵을 생성한다.
- **Encoder-Decoder 구조**: 인코더가 특징을 압축하고 디코더가 이를 다시 업샘플링하여 해상도를 복원한다. UNet, SegNet, DeConvNet 등이 대표적이며, Skip-connection을 통해 세부 공간 정보를 보존한다.
- **Pyramidal Architectures**: PSPNet과 같이 다양한 스케일의 컨텍스트 정보를 수집하는 피라미드 풀링 모듈을 사용하여 지역적 모호성을 제거한다.
- **Dilated Convolutions**: 해상도 손실 없이 수용 영역(Receptive Field)을 확장하기 위해 Atrous Convolution을 사용한다 (예: DeepLab 시리즈).
- **Attention & Transformers**: 전역적 컨텍스트를 캡처하기 위해 Self-attention 및 Vision Transformer (ViT) 기반 모델(예: SegFormer, Swin Transformer)이 도입되었다.

#### 1.3 SiS 손실 함수
가장 일반적인 손실 함수는 픽셀 단위의 Cross-Entropy Loss이다.
$$L_{ce} = -\mathbb{E}_{(X,Y)} \left[ \sum_{h,w} y^{(h,w)} \cdot \log(p(F(x^{(h,w)}))) \right]$$
여기서 $F$는 모델, $p$는 클래스 확률 벡터, $y$는 정답 원-핫 벡터이다. 클래스 불균형 문제를 해결하기 위해 가중치 기반 학습이나 IoU(Intersection over Union)를 직접 최적화하는 손실 함수가 사용된다.

### 2. Domain Adaptation for SiS (DASiS) 방법론

#### 2.1 기본 원리
소스 도메인($D_S$, 레이블 있음)과 타겟 도메인($D_T$, 레이블 없음) 간의 분포 차이(Domain Shift)를 줄이는 것이 목표이다.

#### 2.2 도메인 정렬 레벨 (Alignment Levels)
- **이미지 수준 (Image-level)**: GAN 기반의 스타일 전이(Style Transfer)를 통해 소스 이미지의 외관을 타겟과 유사하게 변환한다. Cycle-consistency loss 등을 사용하여 구조적 특징을 보존한다.
- **특징 수준 (Feature-level)**: 잠재 공간(Latent Space)에서 분포 간의 거리(MMD 등)를 최소화하거나, 도메인 판별자(Discriminator)를 속이는 적대적 학습(Adversarial Training)을 통해 도메인 불변 특징(Domain-invariant features)을 학습한다.
- **출력 수준 (Output-level)**: 최종 예측 맵(Class-likelihood maps)의 분포를 정렬하여 도메인 간의 격차를 줄인다.

#### 2.3 보완 기법 (Complementary Techniques)
- **Self-training**: 타겟 데이터에 대해 신뢰도가 높은 예측값을 의사 레이블(Pseudo-labels)로 사용하여 모델을 재학습시킨다.
- **Entropy Minimization**: 타겟 예측의 엔트로피를 최소화하여 예측 결과가 더 확신을 갖도록 유도한다.
- **Co-training**: 서로 다른 두 모델의 예측 일치성을 높여 판별력을 강화한다.

## 📊 Results

본 논문은 특정 모델의 성능을 측정하는 실험 논문이 아니라 서베이 논문이므로, 기존 연구들에서 사용된 데이터셋과 평가지표를 정리하여 제시한다.

### 1. 주요 데이터셋
- **Object Segmentation**: PASCAL VOC, MS COCO
- **Image Parsing**: ADE20K
- **Autonomous Driving (AD)**: Cityscapes (실제), GTA-5 (합성), SYNTHIA (합성), BDD100K, ACDC (다양한 기상 조건)

### 2. 평가 지표
- **mIoU (mean Intersection over Union)**: 각 클래스별 IoU의 평균으로, SiS의 표준 지표이다.
- **Pixel Accuracy**: 전체 픽셀 중 정답을 맞춘 비율이다.
- **BCM Score**: 경계선 세그멘테이션의 품질을 측정하는 지표이다.

### 3. DASiS 벤치마크
가장 널리 사용되는 설정은 **GTA-5 $\rightarrow$ Cityscapes** (Sim-to-Real) 적응 작업이다. 최근에는 실제 환경의 다양한 기상 조건을 포함한 ACDC 데이터셋 등이 중요하게 다뤄지고 있다.

## 🧠 Insights & Discussion

### 1. 강점 및 기술적 흐름
본 보고서는 SiS 기술이 **'지역적 특징 추출 $\rightarrow$ 전역적 컨텍스트 통합 $\rightarrow$ 도메인 간 일반화'** 순으로 발전해 왔음을 명확히 보여준다. 특히 CNN의 수용 영역 한계를 극복하기 위해 Dilated Convolution에서 시작하여, 최종적으로 Transformer의 전역 어텐션으로 이행한 흐름이 인상적이다.

### 2. 한계 및 미해결 과제
- **데이터 의존성**: 딥러닝 모델은 여전히 대량의 데이터에 의존하며, DASiS 기술이 발전했음에도 불구하고 합성 데이터와 실제 데이터 사이의 근본적인 '현실성 격차(Reality Gap)'를 완벽히 메우는 것은 여전히 어렵다.
- **계산 효율성**: Transformer 기반 모델은 높은 정확도를 보이지만, 픽셀 단위의 밀집 예측 작업 특성상 계산 복잡도가 매우 높아 실시간 자율주행 시스템에 적용하기 위한 경량화 연구가 필수적이다.
- **OOD(Out-of-Distribution) 문제**: 훈련 단계에서 보지 못한 새로운 클래스가 등장했을 때 이를 탐지하고 처리하는 능력(Unknown class detection)은 여전히 도전적인 과제이다.

### 3. 비판적 해석
논문은 광범위한 방법론을 체계적으로 분류하여 훌륭한 가이드를 제공하지만, 개별 방법론들의 정량적 성능 비교(Benchmark Table)보다는 분류(Taxonomy)에 치중되어 있다. 향후 연구에서는 각 분류군별 대표 모델들의 성능 지표를 통합 비교함으로써, 어떤 아키텍처가 특정 도메인 적응 시나리오에서 가장 효율적인지를 제시할 필요가 있다.

## 📌 TL;DR

본 논문은 지난 20년간의 **시맨틱 이미지 세그멘테이션(SiS)**과 **도메인 적응(DASiS)** 연구를 집대성한 종합 서베이이다. 전통적인 CRF/MRF 방식에서 최신 Transformer 모델까지의 진화 과정을 정리하였으며, 특히 합성 데이터를 실제 환경에 적용하기 위한 **이미지-특징-출력 수준의 정렬 기법**을 체계적으로 분류하였다. 이 연구는 자율주행 및 의료 영상 분석과 같이 고비용의 데이터 레이블링이 필요한 분야에서 **Sim-to-Real 적응 전략**을 수립하는 데 중요한 이론적 토대를 제공한다.