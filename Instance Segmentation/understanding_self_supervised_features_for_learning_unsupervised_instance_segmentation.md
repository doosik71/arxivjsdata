# Understanding Self-Supervised Features for Learning Unsupervised Instance Segmentation

Paul Engstler, Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina (2023)

## 🧩 Problem to Solve

본 논문은 사람이 작성한 레이블(manual annotations) 없이 이미지 내의 개별 객체들을 분리하고 세그멘테이션하는 **Unsupervised Instance Segmentation(비지도 인스턴스 세그멘테이션)** 문제를 다룬다.

일반적인 Unsupervised Semantic Segmentation은 같은 범주에 속하는 여러 객체를 하나의 영역으로 처리하지만, Instance Segmentation은 동일한 범주의 객체라 할지라도 각각의 개별 인스턴스를 구분하여 인식해야 하므로 훨씬 더 도전적인 과제이다. 이를 위해서는 이미지 내에서 다양한 크기와 외형을 가진 객체라는 개념, 즉 **Objectness**를 획득하는 것이 필수적이다.

논문의 주된 목표는 다양한 자기지도학습(Self-Supervised Learning, SSL) 기반의 특징 추출기(Feature Extractor)들이 인스턴스를 구분하는 능력인 **Instance-awareness** 측면에서 어떤 차이를 보이는지 분석하고, 이를 활용해 효과적인 비지도 인스턴스 세그멘테이션 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 서로 다른 SSL 학습 목표(Training Objective)가 생성되는 특징(Feature)의 성격과 인스턴스 구분 능력에 결정적인 영향을 미친다는 점을 밝혀낸 것이다.

핵심 직관은 다음과 같다. DINO와 같은 **Discriminative(판별적)** 또는 **Self-distillation(자기 증류)** 방식의 모델은 매우 강력한 시맨틱 정보를 인코딩하여 객체의 위치를 찾는 데는 뛰어나지만, 동일 클래스의 서로 다른 인스턴스를 하나로 묶어버리는 경향이 있다. 반면, MAE와 같은 **Generative(생성적)** 또는 **Image Reconstruction(이미지 재구성)** 방식의 모델은 픽셀 수준의 재구성 목표로 인해 상대적으로 낮은 시맨틱 수준의 특징을 학습하며, 이는 오히려 동일 클래스 내의 서로 다른 인스턴스를 공간적으로 분리해내는 능력(Instance-awareness)을 높여준다는 것이다.

## 📎 Related Works

### SSL 및 비지도 객체 발견

최근 SSL 연구는 Contrastive learning, Masked Image Modeling(MIM) 등을 통해 레이블 없는 데이터에서 강력한 표현력을 학습하는 방향으로 발전했다. 특히 Vision Transformers(ViTs)의 도입으로 성능이 크게 향상되었으며, DINO-ViT의 self-attention 맵이 전경-배경 분리 및 salient object discovery에 유용하다는 점이 이미 알려져 있다.

### 비지도 시맨틱 및 인스턴스 세그멘테이션

기존의 비지도 시맨틱 세그멘테이션 연구들은 주로 SSL 특징을 직접 클러스터링하여 pseudo-mask를 생성하는 방식을 사용했다. 하지만 인스턴스 세그멘테이션의 경우, 단순히 시맨틱 특징을 사용하는 것만으로는 부족하며, 대다수의 기존 연구(FreeSOLO, CutLER 등)는 coarse한 mask proposal을 먼저 생성하고 이를 통해 네트워크를 부트스트랩(bootstrap)하는 방식을 취하고 있다. 본 논문은 이러한 mask proposal을 생성하는 단계에서 어떤 SSL 특징을 사용하는 것이 최적인지를 분석함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 1. 특징 분석을 위한 마스크 생성 절차

특징 추출기가 인스턴스를 얼마나 잘 구분하는지 측정하기 위해, 저자들은 다음과 같은 절차로 마스크를 생성한다.

- ViT의 마지막 self-attention 블록의 key에서 특징을 추출한다.
- 특징 공간에서 **Cosine Similarity**를 사용하여 어피니티 행렬(Affinity Matrix)을 계산한다.
- 이 행렬에 대해 **k-way Spectral Clustering**을 공간적으로 적용하여 마스크를 생성한다.
- 다양한 객체 수와 세분화 수준을 고려하기 위해 $k \in K$ (예: $\{2, 3, 4, 5, 6\}$) 값들을 변경하며 여러 번 클러스터링을 수행하고, 결과 마스크들을 누적한다.
- 생성된 마스크 중 연결되지 않은 컴포넌트(non-connected components)가 있다면 이를 별도의 마스크로 분리한다.

### 2. 인스턴스 세그멘테이션을 위한 셀프 트레이닝 파이프라인

분석 결과를 바탕으로, 저자들은 다음과 같은 2단계 마스크 제안(Mask Proposal) 및 학습 파이프라인을 제안한다.

**단계 1: 후보 마스크 생성 (Candidate Mask Generation)**

- 선택한 SSL 특징 추출기를 사용하여 $K=\{2, 3, 4, 5\}$ 값으로 multi-k-way clustering을 수행한다. 이 과정에서 전체 이미지 영역을 대상으로 하므로 배경 영역이 포함된 마스크가 생성될 가능성이 높다.

**단계 2: Saliency 기반 필터링 (Saliency-based Masking)**

- 배경 마스크를 제거하고 객체일 가능성이 높은 마스크만 남기기 위해 DINO 특징을 사용한다.
- DINO 특징으로 $k=2$ spectral clustering을 수행하여 전경-배경을 분리하고, 이미지 경계와 겹치는 픽셀이 적은 쪽을 foreground(saliency map)로 선택한다.
- 단계 1에서 생성된 후보 마스크 중 이 saliency map과 강하게 교차(intersect)하는 마스크만 최종 제안(proposals)으로 선택한다.

**단계 3: 세그멘테이션 네트워크 학습**

- 최종 선택된 마스크들에 대해 **Non-Maximum Suppression(NMS)**을 적용하여 중복을 제거한다.
- 이렇게 얻은 pseudo-mask들을 타겟으로 하여 **SOLOv2** 아키텍처를 학습시킨다. 이때 backbone은 frozen 상태로 유지하며, BoxInst loss와 copy & paste augmentation을 사용한다.

## 📊 Results

### SSL 특징별 인스턴스 구분 능력 분석

Table 1의 결과에 따르면, 동일한 시맨틱 클래스의 인스턴스가 2개 이상 존재하는 경우 **MAE**가 다른 모든 모델보다 월등히 높은 Mean Average Recall(mAR)을 기록했다. 반면, 단일 객체를 찾는 능력은 **DINO**, MSN, MoCo-v3와 같은 판별적 모델들이 더 뛰어났다. 이는 MAE의 재구성 목표가 낮은 시맨틱 수준의 특징을 학습하게 하여 동일 클래스 객체들을 하나로 묶지 않고 공간적으로 분리하는 경향이 있음을 시사한다.

### 비지도 인스턴스 세그멘테이션 성능

Table 2에서 MAE 특징으로 생성한 마스크를 통해 학습시킨 모델이 $AP_{50}=12.1$, $AP=5.2$로 가장 좋은 성능을 보였다. DINO 기반 모델은 학습 후 성능 격차를 많이 좁혔는데, 이는 DINO 마스크가 시맨틱하게는 덜 정교할지라도 객체의 경계(boundary)를 더 깨끗하게 포착하여 학습에 유리했기 때문으로 분석된다.

또한, SOTA 방법론들과 비교한 Table 3에서 제안 방법(MAE 기반)은 FreeSOLO보다 우수한 성능을 보였으며, CutLER와 같은 최신 기법들과도 경쟁 가능한 수준의 성능을 나타냈다.

## 🧠 Insights & Discussion

본 논문은 SSL 모델의 학습 목표가 특징의 **Semanticity(시맨틱 수준)**와 **Instance-awareness(인스턴스 인식 능력)** 사이의 트레이드-오프를 만든다는 점을 시사한다.

- **MAE의 강점:** 픽셀 재구성이라는 공간적 제약 조건 덕분에 인스턴스 간의 경계를 더 잘 인식한다. 이는 인스턴스 분리가 핵심인 과제에서 강력한 이점이 된다.
- **DINO의 강점:** 높은 시맨틱 추상화 능력을 갖추고 있어, 객체가 무엇인지 파악하고 전경을 추출하는 Saliency map 생성에 최적이다.
- **비판적 해석:** 본 연구는 MAE가 인스턴스 구분 능력이 좋다는 것을 밝혔지만, 절대적인 마스크의 퀄리티(경계의 정교함 등)는 여전히 DINO가 우세하다는 점을 언급한다. 따라서 단일 모델을 사용하기보다 MAE의 인스턴스 분리 능력과 DINO의 Saliency 추출 능력을 결합한 하이브리드 접근 방식이 실질적인 성능 향상을 이끌어냈음을 알 수 있다.

## 📌 TL;DR

이 논문은 비지도 인스턴스 세그멘테이션을 위해 어떤 SSL 특징이 유리한지 분석하였다. 분석 결과, **MAE(생성적 모델)는 동일 클래스의 서로 다른 인스턴스를 구분하는 능력이 뛰어나고, DINO(판별적 모델)는 전경 객체를 찾는 능력이 뛰어남**을 발견했다. 이를 결합하여 MAE로 후보 마스크를 만들고 DINO로 필터링하여 SOLOv2를 학습시킨 결과, 기존 비지도 인스턴스 세그멘테이션 방법론들보다 향상된 성능을 달성했다. 이 연구는 향후 인스턴스 단위의 정밀한 비지도 학습을 위한 특징 선택 가이드라인을 제시한다.
