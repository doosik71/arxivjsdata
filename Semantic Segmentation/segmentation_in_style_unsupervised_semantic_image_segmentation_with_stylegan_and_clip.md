# Segmentation in Style: Unsupervised Semantic Image Segmentation with Stylegan and CLIP

Daniil Pakhomov, Sanchit Hira, Narayani Wagle, Kemar E. Green, Nassir Navab (2021)

## 🧩 Problem to Solve

본 논문은 인간의 감독(human supervision) 없이 이미지 내의 의미론적으로 유의미한 영역을 자동으로 분할(segmentation)하는 문제를 해결하고자 한다. 일반적으로 Semantic Image Segmentation을 수행하기 위해서는 방대한 양의 픽셀 단위 어노테이션(pixel-level annotation) 데이터셋이 필요하며, 이는 매우 느리고 비용이 많이 드는 작업이다. 특히 의료 영상 분야의 경우 도메인 전문가의 지식이 필수적이어서 레이블링 과정이 더욱 어렵다. 또한, 동일한 데이터셋에 대해 여러 작업자가 레이블링을 수행하더라도 시각적 경계가 불분명한 클래스의 경우 일관성이 떨어지는 문제가 발생한다. 따라서 본 연구의 목표는 어떠한 수동 레이블링 없이도 이미지 전반에 걸쳐 일관된 의미론적 영역을 발견하고, 이를 통해 실물 이미지에 적용 가능한 segmentation 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 학습된 생성 모델인 StyleGAN2의 특성 공간(feature space)을 활용하여 의미론적 클래스를 발견하는 것이다. 생성 모델의 중간 레이어 feature map들이 이미지의 의미론적 구조를 내포하고 있다는 점에 착안하여, 이를 clustering함으로써 레이블 없는 가상 데이터셋을 생성한다. 또한, CLIP(Contrastive Language-Image Pre-training)을 결합하여 자연어 프롬프트를 통해 희귀한 semantic class를 발견하고, 발견된 클러스터에 구체적인 클래스 이름을 할당하는 메커니즘을 제안한다. 최종적으로는 이렇게 생성된 합성 데이터(synthetic data)를 이용해 segmentation 네트워크를 학습시키고, 이를 실제 이미지로 일반화(generalization)하는 Knowledge Distillation 과정을 거친다.

## 📎 Related Works

논문에서는 소량의 어노테이션을 사용하는 준지도 학습(Semi-Supervised Learning, SSL) 접근 방식들을 주요 비교 대상으로 삼는다. 기존의 SSL 방식들은 일부 레이블을 사용하여 성능을 높이려 하지만, 본 연구는 어떠한 수동 레이블도 사용하지 않음에도 불구하고 최신 SSL 방법론들(예: DatasetGAN, semanticGAN)보다 우수한 성능을 보임을 강조한다. 또한, StyleGAN의 latent space를 조작하여 이미지를 편집하는 기존 연구들을 활용하여, clustering 과정에서 누락될 수 있는 희귀 클래스를 강제로 생성해내는 차별점을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인

전체 시스템은 다음의 단계로 구성된다: $\text{StyleGAN2 Feature Extraction} \rightarrow \text{Clustering} \rightarrow \text{Rare Class Discovery via CLIP} \rightarrow \text{Cluster Classification} \rightarrow \text{Knowledge Distillation}$.

### 1. StyleGAN2 및 Clustering

StyleGAN2는 고품질 이미지를 생성하며, 생성 과정에서 발생하는 중간 feature map에 접근할 수 있다. 연구진은 주로 7번째 또는 9번째 레이어의 feature map을 사용하며, 이전 레이어일수록 coarse(거친)하지만 의미론적으로 유의미한 클러스터가 형성되고, 이후 레이어일수록 fine(세밀)하지만 의미론적 의미는 적어진다는 것을 확인하였다.

추출된 feature map을 평탄화(flatten)한 후, 다음과 같은 손실 함수를 최소화하는 K-means clustering을 수행한다:

$$J = \sum_{n=1}^{\bar{N}} \sum_{k=1}^{K} r_{nk} \|x_n - \mu_k\|^2$$

여기서 $x_n$은 단일 데이터 포인트(특성 벡터), $\mu_k$는 $k$번째 클러스터의 중심점, $r_{nk} \in \{0, 1\}$은 데이터 포인트 $x_n$이 클러스터 $k$에 할당되었는지를 나타내는 이진 지표 변수이다. $\bar{N}$은 전체 샘플 수에 공간 해상도를 곱한 값이다. 이 과정을 통해 이미지 내의 픽셀들이 의미론적 그룹으로 묶이게 되며, 새로운 샘플에 대해서도 가장 가까운 클러스터 중심점에 할당함으로써 segmentation mask를 생성할 수 있다.

### 2. CLIP을 이용한 희귀 클래스 발견 (Class Discovery)

수염(beard), 안경(glasses), 모자(hat)와 같이 데이터셋에서 드물게 나타나는 클래스는 일반적인 clustering만으로는 발견하기 어렵다. 이를 해결하기 위해 CLIP을 이용한 latent space 조작 기법을 사용한다.

- 자연어 프롬프트(예: "a person with a beard")를 통해 CLIP으로 이미지를 분류하고, 해당 속성이 포함된 latent vector $w$들의 집합을 수집한다.
- 이를 통해 특정 속성을 강화하는 조작 방향(manipulation direction) $G$를 학습한다.
- 새로운 이미지 생성 시 latent code를 $w + \alpha G$로 수정하여 생성하면, 거의 모든 샘플에 해당 속성이 나타나게 되어 clustering 단계에서 해당 클래스가 명확하게 분리된다.

### 3. 클러스터 분류 (Cluster Classification)

발견된 각 클러스터가 실제로 어떤 클래스인지 정의하기 위해 CLIP의 text encoder와 image encoder를 사용한다.

- **Text Encoder**: "hair", "forehead" 등의 프롬프트를 입력하여 텍스트 임베딩을 생성한다.
- **Image Encoder**: 기존 CLIP의 image encoder에서 downsampling 레이어를 제거하고 dilated convolution을 추가하여 spatial resolution을 높인 수정된 인코더를 사용한다.
- 각 클러스터 영역의 임베딩을 여러 이미지에 대해 평균 낸 후, 텍스트 임베딩과의 내적(dot product)을 계산한다. 내적 값이 가장 큰 텍스트 프롬프트를 해당 클러스터의 클래스로 할당한다.

### 4. 지식 증류 (Knowledge Distillation)

StyleGAN2를 통해 생성된 합성 이미지와 앞서 구축한 segmentation mask 쌍을 이용하여 segmentation 네트워크(ResNet-18 기반의 dilated convolution 모델)를 학습시킨다. 이 모델은 합성 데이터로 학습되었지만, 실제 이미지(real images)에 대해서도 일반화된 segmentation 성능을 보인다.

## 📊 Results

### 실험 설정

- **데이터셋**: CelebA-Mask8, CelebAMask-HQ, OpenEDS(안구 분할), 그리고 자체 제작한 수염 분할 데이터셋을 사용하였다.
- **지표**: Mean IoU (Intersection over Union)를 사용하여 정량적으로 평가하였다.

### 주요 결과

1. **CelebA-Mask8**: 수동 어노테이션을 전혀 사용하지 않았음에도 Mean IoU 73.1을 달성하여, 소량의 레이블을 사용한 SSL 모델들(GAN [20]: 70.0, semanticGAN [10]: 69.02)보다 우수한 성능을 보였다.
2. **CelebAMask-HQ**: 19개의 클래스를 가진 복잡한 데이터셋에서 fully supervised 모델(DRN [19]: 70.5)보다는 낮은 62.5의 IoU를 기록했으나, 전반적으로 준수한 성능을 보였다.
3. **OpenEDS (Eye Segmentation)**: StyleGAN2를 처음부터 학습시켜 적용한 결과, Mean IoU 82.39를 기록하여 fully supervised 모델(SegNet [13]: 84.1)에 매우 근접한 성능을 나타냈다. 이는 안구 구조의 의미론적 클래스가 상대적으로 단순하기 때문으로 분석된다.
4. **수염 분할 (Beard)**: 자체 수집 데이터셋에서 fully supervised 모델(88.8) 대비 84.29의 IoU를 기록하며 경쟁력 있는 성능을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

본 연구는 생성 모델의 내부 feature space가 이미지의 semantic layout을 이미 학습하고 있다는 점을 성공적으로 이용하였다. 특히, 인간 작업자가 일관되게 레이블링하기 어려운 경계 영역에서도 생성 모델 기반의 clustering은 일관된 영역을 제안할 수 있다는 점이 큰 강점이다. 또한 CLIP과의 결합을 통해 "텍스트 $\rightarrow$ latent 조작 $\rightarrow$ clustering $\rightarrow$ segmentation"으로 이어지는 파이프라인을 구축함으로써, 데이터셋에 없는 희귀 클래스까지 unsupervised 방식으로 추출할 수 있음을 보여주었다.

### 한계 및 비판적 해석

StyleGAN2가 모든 세부 요소를 완벽하게 생성하지는 못한다는 점이 한계로 작용한다. 예를 들어 귀걸이나 목걸이 같은 작은 액세서리는 생성 퀄리티가 낮아 segmentation 성능이 떨어졌다. 이는 결국 unsupervised segmentation의 성능이 기반이 되는 generative model의 표현 능력에 강하게 의존하고 있음을 시사한다. 또한, fully supervised 모델과의 성능 격차는 클래스의 복잡도가 증가할수록 벌어지는 경향이 있어, 매우 정밀한 분할이 필요한 작업에서는 여전히 한계가 있을 수 있다.

## 📌 TL;DR

본 논문은 **StyleGAN2의 feature space clustering과 CLIP의 텍스트 가이던스를 결합하여, 인간의 레이블링 없이도 의미론적 이미지 분할을 수행하는 방법론**을 제안한다. 생성 모델을 통해 합성 데이터셋을 구축하고 이를 segmentation 모델에 증류(distillation)함으로써 실제 이미지에 적용 가능한 모델을 만들어냈으며, 특정 도메인에서는 준지도 학습(SSL)을 능가하고 완전 지도 학습(Supervised)에 근접하는 성능을 보였다. 이 연구는 특히 레이블링 비용이 극도로 높은 의료 영상이나 희귀 속성 추출 분야에서 중요한 역할을 할 가능성이 높다.
