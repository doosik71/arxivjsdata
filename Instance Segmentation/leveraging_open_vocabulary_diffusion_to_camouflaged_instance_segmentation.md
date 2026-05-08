# Leveraging Open-Vocabulary Diffusion to Camouflaged Instance Segmentation

Tuan-Anh Vu, Duc Thanh Nguyen, Qing Guo, Binh-Son Hua, Nhat Minh Chung, Ivor W. Tsang, Sai-Kit Yeung (2023)

## 🧩 Problem to Solve

본 논문은 **위장 객체 인스턴스 분할(Camouflaged Instance Segmentation, CIS)** 문제를 해결하고자 한다. 위장(Camouflage)은 생물학적으로 포식자나 피식자로부터 자신을 숨기기 위해 주변 환경과 색상 및 패턴을 일치시키는 메커니즘이다. 이러한 특성 때문에 컴퓨터 비전 관점에서는 객체와 배경 사이의 시각적 단서가 매우 미세하여, 객체를 배경으로부터 분리해내는 것이 매우 어렵다.

기존의 위장 객체 탐지(Camouflaged Object Detection, COD) 연구들은 주로 영역 단위(Bounding Box)의 거친 식별에 그쳤으며, 개별 인스턴스를 정밀하게 분리하고 세분화하는 능력이 부족했다. 특히, 학습 단계에서 보지 못한 새로운 범주의 객체(Novel Objects)를 분할해야 하는 상황에서 시각적 정보만으로는 한계가 명확하다. 따라서 본 연구의 목표는 **Open-Vocabulary Diffusion** 모델과 **Vision-Language 모델(CLIP)**을 활용하여, 시각적 단서가 부족한 상황에서도 텍스트-시각 교차 도메인 표현(Cross-domain representations)을 통해 위장 객체를 정밀하게 분할하는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **텍스트-이미지 확산 모델(Text-to-Image Diffusion Model)의 강력한 표현 학습 능력**과 **Open-Vocabulary 모델의 광범위한 개념 지식**을 결합하는 것이다.

1. **Open-Vocabulary 기반 CIS 프레임워크 제안**: Stable Diffusion과 CLIP을 결합하여, 시각적 정보가 부족한 위장 객체를 식별하기 위해 텍스트 도메인의 풍부한 정보를 시각 도메인으로 전이시킨다.
2. **위장 특화 모듈 설계**: 단순한 특징 추출을 넘어 위장 객체의 특성을 반영한 네 가지 핵심 모듈(MSFF, Mask Generator, TVA, CIN)을 제안하여 객체-배경 간의 변별력을 높였다.
3. **범용성 확보**: 학습 시 보지 못한 새로운 객체 카테고리에 대해서도 분할이 가능한 Open-Vocabulary 능력을 갖추어, 실제 환경에서의 적용 가능성을 높였다.

## 📎 Related Works

### 1. 위장 객체 이해 (Camouflaged Object Understanding)

기존의 COD 및 CIS 연구들은 주로 이미지 데이터만을 사용하여 객체와 배경을 구분하는 변별적 특징을 학습하려 했다. 하지만 위장의 본질 자체가 시각적 유사성을 이용하는 것이므로, 이미지 데이터만으로는 한계가 있다. 본 논문은 이에 대한 해결책으로 텍스트 데이터를 추가적인 단서로 활용하는 방안을 제시하며 기존 연구와 차별화한다.

### 2. 텍스트-이미지 확산 모델 (Text-to-Image Diffusion)

Stable Diffusion과 같은 모델들은 텍스트 설명으로부터 고품질 이미지를 생성하기 위해 방대한 양의 데이터를 학습했다. 본 연구는 이 모델을 '생성' 목적이 아닌, 이미지 내의 핵심 객체 특징을 추출하는 '표현 학습(Representation Learning)' 도구로 활용한다.

### 3. Open-Vocabulary 탐지 및 분할

CLIP과 같은 Vision-Language Model(VLM)을 활용해 제한된 사전 정의 클래스를 넘어 새로운 객체를 인식하려는 시도가 있었다. 하지만 기존 방식들은 일반적인 객체 클래스에 집중되어 있어, 배경과 극도로 유사한 위장 객체를 분할하는 데에는 한계가 있었다.

## 🛠️ Methodology

### 전체 파이프라인

본 모델은 입력 이미지와 해당 이미지 내 객체에 대한 텍스트 프롬프트를 입력으로 받아, 최종적으로 인스턴스 마스크와 카테고리를 출력한다. 전체 구조는 크게 특징 추출 $\rightarrow$ 마스크 생성 $\rightarrow$ 텍스트-시각 통합 $\rightarrow$ 정규화 및 분류 단계로 이어진다.

### 주요 구성 요소 및 역할

**1. Multi-scale Feature Fusion (MSFF)**
Stable Diffusion(SD)의 UNet 구조에서 인코더의 다중 스케일 특징과 디코더의 마지막 레이어 특징을 융합한다.

- **절차**: 다중 스케일 특징들을 Concatenation 한 후 $1\times1$ Convolution을 적용하고, 이를 다시 원본 특징과 Element-wise Multiplication 및 Addition 연산을 통해 융합한다. 이를 통해 저수준의 세부 정보와 고수준의 시맨틱 정보를 모두 확보한다.

**2. Mask Generator**
Mask2Former의 아키텍처를 기반으로 하며, MSFF에서 생성된 융합 특징을 입력으로 받아 클래스 구분 없는(Class-agnostic) 이진 마스크 $\{m_{pred_i}\}$와 마스크 임베딩 $\{z_{pred_i}\}$를 생성한다.

- **구조**: Pixel Decoder가 해상도를 높여 세밀한 임베딩을 생성하고, Transformer Decoder가 Object Query를 처리하여 다양한 크기의 객체를 효과적으로 포착한다.

**3. Textual-Visual Aggregation (TVA)**
마스크 임베딩(시각 정보)과 CLIP의 텍스트 임베딩(텍스트 정보)을 결합하여 전경 객체의 특징을 강조한다.

- **절차**: 마스크 풀링을 통해 추출된 시각 특징과 텍스트 특징 간의 내적(Dot product)을 계산한다. 이때 단순히 내적만 하는 것이 아니라, Softmax 연산과 평균 정규화(Mean-normalisation)를 적용하여 불필요한 노이즈를 제거하고 유의미한 특징만을 채널 방향으로 합산한다.

**4. Camouflaged Instance Normalisation (CIN)**
TVA의 결과물인 텍스트-시각 특징 맵과 마스크 제너레이터의 마스크를 입력으로 받아 최종 마스크를 예측한다.

- **절차**: 특징 맵을 선형 레이어로 투영한 후, 아핀 가중치(Affine weights)와 편향(Biases)을 생성하여 입력 마스크와 결합함으로써 최종 인스턴스 마스크를 정교화한다.

### 학습 절차 및 손실 함수

모델은 Supervised learning 방식으로 학습되며, SD와 CLIP 모델은 Frozen 상태로 유지하고 제안된 특화 모듈들만 학습시킨다.

**손실 함수**
전체 손실 함수 $\mathcal{L}$은 다음과 같이 정의된다:
$$\mathcal{L} = \alpha \mathcal{L}_{bce} + \mathcal{L}_{dice} + \mathcal{L}_{ce}$$

- $\mathcal{L}_{bce}$: 마스크 예측의 정확도를 위한 Binary Cross-Entropy Loss이다.
- $\mathcal{L}_{dice}$: 클래스 불균형 문제를 해결하기 위한 Dice Loss이다.
- $\mathcal{L}_{ce}$: 마스크 임베딩 $z_{pred_i}$를 정답 카테고리 $y_{cate_i}$와 연결하기 위한 Cross-Entropy Loss이며, 다음과 같이 계산된다:
$$\mathcal{L}_{ce} = \frac{1}{N} \sum_{i=1}^{N} \text{CE} \left( \text{Softmax} \left( \frac{z_{pred_i} T(C_{train})^\top}{\tau} \right), y_{cate_i} \right)$$
여기서 $T(C_{train})$은 CLIP 텍스트 인코더로 생성된 학습 카테고리들의 임베딩 집합이며, $\tau$는 학습 가능한 온도 파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: MS-COCO로 사전 학습 후, COD10K-v3의 학습셋으로 파인튜닝을 진행하였다. 테스트는 COD10K-v3와 NC4K 데이터셋에서 수행하였다. 범용 성능 평가를 위해 ADE20K와 Cityscapes 데이터셋도 사용하였다.
- **지표**: Intersection-over-Union (IoU) 임계값 $[50\%, 95\%]$ 범위의 Average Precision (AP)을 측정하였다.

### 주요 결과

1. **위장 객체 데이터셋 (COD10K-v3, NC4K)**:
   - **Ours (task-specific)** 버전은 기존의 Open-Vocabulary 방식인 ODISE를 압도하며 새로운 SOTA를 달성하였다.
   - 폐쇄 집합(Closed-set) 방식의 최상위 모델인 DCNet과 대등한 성능을 보이면서도, 학습 가능한 파라미터 수는 훨씬 적어 효율적임을 입증하였다.
   - 파인튜닝을 하지 않은 기본 "Ours" 모델보다 파인튜닝을 거친 "Ours (task-specific)" 모델의 성능이 대폭 향상되어, 도메인 적응의 중요성을 확인하였다.

2. **범용 Open-Vocabulary 데이터셋 (ADE20K, Cityscapes)**:
   - OpenSeeD 등 최신 모델들과 비교했을 때 AP 기준 2위를 기록하였다. 특히 OpenSeeD보다 파라미터 수를 약 4배 적게 사용하면서도 성능 하락은 매우 적어 효율성이 매우 뛰어났다.

3. **Ablation Study**:
   - **텍스트 임베딩의 중요성**: 텍스트 임베딩을 0으로 설정했을 때 AP가 19.3에서 12.2로 급격히 하락하여, 위장 객체 식별에 텍스트 정보가 필수적임을 보였다.
   - **모듈별 기여**: CIN 모듈을 제거했을 때 성능 하락이 컸으며, MSFF와 TVA 역시 성능 유지에 기여함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 시각적 정보가 극도로 제한된 위장 상황에서 **텍스트 도메인의 지식을 시각적 가이드로 활용**함으로써 문제를 해결했다는 점이 매우 독창적이다. 특히 Stable Diffusion의 중간 특징과 CLIP의 텍스트 임베딩을 정교하게 융합하는 구조를 통해, 새로운 객체에 대해서도 강건한 분할 성능을 보여주었다.

### 한계 및 비판적 해석

- **인스턴스 분리 문제**: 논문에서도 언급되었듯이, 서로 매우 유사한 외형을 가진 객체들이 겹쳐 있거나 맞닿아 있을 때 이를 개별 인스턴스로 분리하는 능력은 부족하다. 이는 텍스트 정보가 '무엇'인지에 대한 정보는 주지만, '어디서부터 어디까지가 개별 객체인가'에 대한 기하학적 경계 정보는 제공하지 못하기 때문으로 해석된다.
- **폐쇄 영역(Occlusion)**: 객체가 심하게 가려져 파편화된 형태로 나타날 경우, 이를 하나의 객체로 인식하지 못하고 오분류하는 경향이 있다.

### 향후 방향

저자들은 전경뿐만 아니라 배경 정보까지 포함된 프롬프트(예: "나무 위에 있는 도마뱀")를 사용하여 배경 인식 능력을 높이는 방향을 제시하고 있다. 이는 단순한 객체 탐지를 넘어 '관계' 중심의 표현 학습으로 확장될 필요가 있음을 시사한다.

## 📌 TL;DR

본 연구는 **Stable Diffusion과 CLIP을 결합하여 위장 객체 인스턴스 분할(CIS) 문제를 해결하는 Open-Vocabulary 프레임워크**를 제안하였다. 텍스트-시각 교차 도메인 특징을 융합하는 특화 모듈(MSFF, TVA, CIN)을 통해, 학습 시 보지 못한 위장 객체에 대해서도 정밀한 분할이 가능함을 입증하였다. 특히 기존 SOTA 모델 대비 적은 파라미터로 대등하거나 더 뛰어난 성능을 보였으며, 이는 향후 야생 동물 모니터링, 군사 정찰 등 극한의 위장 환경에서의 객체 인식 기술 발전에 크게 기여할 것으로 기대된다.
