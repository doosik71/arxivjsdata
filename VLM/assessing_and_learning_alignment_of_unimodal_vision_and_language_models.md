# Assessing and Learning Alignment of Unimodal Vision and Language Models

Le Zhang, Qian Yang, Aishwarya Agrawal (2025)

## 🧩 Problem to Solve

본 논문은 사전 학습된 단일 모달(unimodal) 시각 모델과 언어 모델이 서로 얼마나 잘 정렬(alignment)되어 있는지를 평가하고, 이를 효율적으로 학습시키는 방법을 다룬다. 기존의 정렬 평가 방법들은 상호 최근접 이웃(mutual nearest-neighbor) 지표와 같은 대리 지표(proxy)에 의존하여, 실제 시각-언어 작업(Vision-Language tasks)에서 모델이 사용되는 방식인 '교차 모달 간 거리 측정'을 직접적으로 반영하지 못한다는 한계가 있다.

따라서 본 연구의 목표는 첫째, 시각-언어 정렬 가능성을 직접적으로 측정할 수 있는 평가 체계를 구축하는 것이며, 둘째, 사전 학습된 단일 모달 모델들의 강점을 활용하여 매우 적은 데이터와 계산 자원만으로도 고성능의 시각-언어 정렬을 달성하는 효율적인 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 요약할 수 있다.

1. **Alignment Probing 제안**: 사전 학습된 시각 및 언어 백본을 동결하고 가벼운 선형 레이어만 학습시켜 두 모달리티 간의 정렬 잠재력을 직접 측정하는 방법을 제안하였다.
2. **정렬 결정 요인 분석**: SSL(Self-Supervised Learning) 시각 모델의 정렬 성능은 선형 분리 가능성(linear separability)보다 표현형의 클러스터링 품질(clustering quality, k-NN 성능으로 측정)에 더 큰 영향을 받는다는 것을 발견하였다.
3. **SAIL(Swift Alignment of Image and Language) 프레임워크**: 비선형 정렬 레이어(GLU), 최적화된 Sigmoid 손실 함수, MLLM 생성 고품질 캡션을 결합한 효율적인 전이 학습 프레임워크를 제안하였다.
4. **효율성 및 성능 입증**: SAIL은 CLIP과 같은 모델 대비 약 6%의 데이터만으로도 ImageNet 제로샷 정확도 73.4%를 달성하였으며, 단일 A100 GPU로 5시간 내에 학습이 가능하다는 점을 보였다.

## 📎 Related Works

기존 연구들은 단일 모달 모델들이 명시적인 정렬 학습 없이도 공유된 통계적 실체에 정렬될 수 있다는 '플라토닉 표현 가설(Platonic Representation Hypothesis)' 등을 제시하였다. 하지만 이러한 연구들은 개별 이미지-텍스트 쌍의 교차 모달 거리를 직접 측정하지 않고 대리 지표를 사용했다는 한계가 있다.

또한, LiT(Locked-image Text Tuning)나 ShareLock과 같은 효율적인 튜닝 방법들이 존재하지만, 이들은 주로 언어 모델을 튜닝하여 동결된 시각 인코더에 맞추는 방식이다. 반면 SAIL은 정렬 레이어를 통해 시각 인코더의 '언어 호환성(language-compatibility)' 자체를 개선함으로써, 학습된 시각 표현이 이후 MLLM(Multimodal Large Language Models)으로 전이될 때 더 높은 성능을 낼 수 있도록 설계되었다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Alignment Probing (정렬 평가)

두 단일 모달 모델의 정렬 잠재력을 평가하기 위해, 두 백본을 동결하고 그 사이에 가벼운 선형 정렬 레이어(linear alignment layer)만을 배치하여 학습시킨다. 이를 통해 모델 자체의 표현 능력이 교차 모달 정렬에 얼마나 유리한지를 정량적으로 측정한다.

### 2. SAIL 프레임워크

SAIL은 고성능의 사전 학습된 시각 모델(예: DINOv2)과 언어 모델(예: NV-Embed-v2)을 효율적으로 결합하는 전이 학습 프레임워크이다.

**가. 아키텍처 및 구성 요소**

- **Alignment Layer**: 단순 선형 레이어 대신 비선형 활성화 함수가 포함된 **GLU(Gated Linear Unit)** 레이어를 사용하여 정렬 성능을 높였다.
- **High-Quality Data**: 짧은 웹 캡션과 MLLM(ShareGPT4)이 생성한 상세하고 긴 캡션을 함께 사용하여 객체 인식 능력과 복잡한 추론 능력을 동시에 확보하였다.

**나. 손실 함수 (Loss Function)**
계산 효율성과 하드 네거티브(hard negatives)에 대한 민감도를 높이기 위해 CLIP의 InfoNCE 대신 이진 분류 기반의 **Sigmoid Loss**를 사용한다.

$$L(I,T) = -\frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{j=1}^{|B|} \log \frac{1}{1 + e^{z_{ij}(-\hat{x}_i \cdot \hat{y}_j + b)}}$$

여기서 $\hat{x}_i$와 $\hat{y}_j$는 각각 시각 및 언어 인코더와 정렬 레이어를 통과한 후 $L2$-정규화된 벡터이며, $z_{ij}$는 정답 쌍일 때 1, 아닐 때 -1의 값을 가진다. $t$는 온도 스케일링, $b$는 편향(bias)을 의미한다. 또한, 여러 개의 긍정 캡션을 활용하기 위해 다음과 같은 Multi-Pos Loss를 정의한다.

$$L^{\text{Multi-Pos}} = L(I, T) + L(I, T^{HQ})$$

**다. 학습 절차 (Cheap Training Recipe)**
학습 효율을 극대화하기 위해 **2단계 파이프라인**을 거친다.

1. **Pre-encoding**: 모든 이미지-텍스트 쌍을 사전 학습된 인코더로 미리 임베딩하여 저장한다.
2. **Alignment Tuning**: GPU에는 인코더를 올리지 않고, 미리 계산된 임베딩과 가벼운 정렬 레이어만 올려 학습한다. 이를 통해 단일 GPU에서도 32,768이라는 매우 큰 배치 사이즈를 사용할 수 있으며 학습 시간을 획기적으로 단축하였다.

## 📊 Results

### 1. 정량적 성능 평가

SAIL-L-NV2 모델(DINOv2-L 및 NV-Embed-v2 기반)은 2,300만 개의 이미지-텍스트 쌍으로 학습되었으며, 다음과 같은 성과를 거두었다.

- **ImageNet 제로샷 분류**: 73.4%의 정확도를 기록하여, 4억 개의 데이터로 학습된 CLIP-L(72.7%)을 능가하였다.
- **이미지-텍스트 검색 (COCO)**: T2I(Text-to-Image) 및 I2T(Image-to-Text) 검색 모두에서 CLIP-L 및 다른 베이스라인(DreamLIP, LiT, ShareLock)보다 우수한 성능을 보였다.
- **복잡한 추론 (Winoground)**: 단순 스케일업된 CLIP 모델들보다 훨씬 높은 성능을 보였으며, 특히 강력한 언어 모델(NV2)을 사용할 때 성능 향상이 두드러졌다.
- **Open-Vocabulary Segmentation**: ADE20K 등에서 CLIP 기반 방법들(MaskCLIP, SCLIP)보다 높은 mIOU를 기록하며 DINOv2의 세밀한 시각 표현력을 유지하고 있음을 입증하였다.

### 2. MLLM으로의 전이 성능

SAIL의 시각 인코더를 LLaVA-1.5에 통합하여 평가한 결과, DINOv2-L 단독 사용 시보다 성능이 크게 향상되었으며, 7개 벤치마크 중 5개 작업에서 CLIP 시각 인코더의 성능을 앞질렀다.

## 🧠 Insights & Discussion

본 논문은 시각-언어 정렬에 있어 다음과 같은 중요한 통찰을 제공한다.

첫째, SSL 시각 표현의 정렬 가능성은 **클러스터링 품질(clustering quality)**과 매우 강한 선형 상관관계를 가진다. 이는 단순히 특징들이 선형적으로 분리 가능한가보다, 같은 개념들이 얼마나 잘 뭉쳐 있는가가 정렬 학습에 더 중요함을 시사한다.

둘째, 복잡한 시각-언어 추론 능력은 시각 모델의 크기보다 **언어 모델의 이해 능력**에 더 크게 의존한다. CLIP의 텍스트 인코더는 웹 데이터의 특성상 풍부한 시맨틱 정보가 부족하여, 이를 강력한 사전 학습 언어 모델로 대체하는 것이 성능 향상의 핵심이다.

셋째, SAIL은 SSL 모델의 특징을 언어 모델과 호환 가능하게 변환함으로써, SSL 모델이 가진 세밀한 시각적 정확도(fine-grained visual acuity)를 유지하면서도 MLLM과의 통합 효율성을 높였다.

**한계점**: SAIL은 TextVQA 및 MMB와 같은 OCR(광학 문자 인식) 능력이 필요한 작업에서는 낮은 성능을 보였다. 이는 SAIL의 기반이 되는 DINOv2 모델 자체가 OCR 능력이 부족하기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 단일 모달 모델 간의 정렬 가능성을 측정하는 **Alignment Probing** 방법론을 제안하고, 이를 통해 SSL 표현의 클러스터링 품질이 정렬의 핵심임을 밝혀냈다. 또한, 매우 적은 데이터(CLIP의 $\sim 6\%$)와 단일 GPU만으로 학습 가능한 **SAIL** 프레임워크를 통해, 기존의 거대 모델들을 능가하는 제로샷 분류 및 복잡한 추론 성능을 달성하였다. 이 연구는 자원이 제한된 환경에서도 고성능 VLM을 구축할 수 있는 효율적인 경로를 제시하며, 특히 SSL 시각 인코더를 언어 호환적으로 만들어 MLLM의 성능을 높일 수 있음을 보여주었다.
