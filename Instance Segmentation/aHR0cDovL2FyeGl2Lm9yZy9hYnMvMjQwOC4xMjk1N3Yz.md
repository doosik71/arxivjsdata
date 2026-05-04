# Image Segmentation in Foundation Model Era: A Survey

Tianfei Zhou, Wang Xia, Fei Zhang, Boyu Chang, Wenguan Wang, Ye Yuan, Ender Konukoglu, Daniel Cremers (2024)

## 🧩 Problem to Solve

이미지 세그멘테이션(Image Segmentation)은 수십 년간 컴퓨터 비전 분야의 핵심 과제였으며, N-Cut, FCN, MaskFormer와 같은 기념비적인 알고리즘들이 지속적으로 연구되어 왔다. 그러나 최근 Foundation Model(FM)의 등장으로 인해 세그멘테이션 방법론은 새로운 시대에 진입하였다. CLIP, Stable Diffusion, DINO와 같은 범용 모델을 세그멘테이션에 적응시키거나, SAM(Segment Anything Model) 및 SAM2와 같은 전용 세그멘테이션 FM이 개발되면서 이전의 딥러닝 체계에서는 볼 수 없었던 새로운 능력이 나타나고 있다.

그럼에도 불구하고, 현재의 연구들은 이러한 발전이 가져온 구체적인 특성, 도전 과제 및 해결책에 대한 상세한 분석이 부족한 상태이다. 기존의 서베이 논문들은 대부분 2021년 이전에 발행되어 최신 FM 기반 접근 방식을 충분히 담아내지 못하고 있다. 따라서 본 논문의 목표는 FM 중심의 이미지 세그멘테이션 연구를 체계적으로 리뷰하여, 범용 이미지 세그멘테이션(Generic Image Segmentation)과 프롬프트 가능 이미지 세그멘테이션(Promptable Image Segmentation)의 두 가지 주요 흐름을 분석하고 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Foundation Model이 이미지 세그멘테이션 분야를 어떻게 변화시키고 있는지를 분석한 최초의 포괄적인 서베이 보고서를 제공한다는 점이다. 주요 설계 아이디어 및 기여 사항은 다음과 같다.

- **분류 체계의 정립**: 이미지 세그멘테이션을 프롬프트 $P$의 유무에 따라 Generic Image Segmentation(GIS)과 Promptable Image Segmentation(PIS)으로 구분하는 통합 수식 체계를 제안하였다.
- **지식의 창발성(Emergence) 분석**: CLIP, Diffusion Models(DMs), DINO와 같이 원래 세그멘테이션을 위해 설계되지 않은 모델들에서 어떻게 세그멘테이션 지식이 창발되는지를 분석하고, 이를 활용한 Training-free 세그멘테이션 기법을 설명하였다.
- **광범위한 문헌 검토**: 300개 이상의 세그멘테이션 접근 방식을 검토하여, 최신 FM 기반의 방법론들을 체계적으로 정리하였다.
- **미래 연구 방향 제시**: In-context segmentation, 객체 환각(Object Hallucination) 완화, 확장 가능한 데이터 엔진 구축 등 앞으로 해결해야 할 개방형 문제들을 정의하였다.

## 📎 Related Works

이미지 세그멘테이션 연구는 크게 세 단계로 발전해 왔다. 첫째는 임계값 처리(Thresholding), Region Growing, Normalized Cuts와 같은 전통적인 비-딥러닝 방식이다. 둘째는 FCN, DeepLab, Mask R-CNN, Transformer 기반 모델 등으로 대표되는 딥러닝 시대로, 이들은 Semantic, Instance, Panoptic 세그멘테이션에서 뛰어난 성능을 보였으나 대부분 폐쇄된 어휘 집합(Closed-vocabulary) 내에서 작동한다는 한계가 있었다.

최근의 서베이들은 특정 작업(예: Open-vocabulary segmentation)이나 특정 아키텍처(예: Transformer)에만 집중하거나, SAM과 같은 특정 모델 하나만을 다루는 경향이 있다. 본 논문은 특정 모델에 국한되지 않고 CLIP, Stable Diffusion, DINO, SAM, LLM/MLLM 등 다양한 Foundation Model들이 세그멘테이션 전반에 미치는 영향을 종합적으로 다룬다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. 통합 수식 체계 (Unified Formulation)
본 논문은 다양한 세그멘테이션 작업을 다음과 같은 통합 매핑 함수 $f$로 정의한다.

$$f: X \to Y, \text{ where } X = I \times P, Y = M \times C$$

여기서 $I$는 입력 이미지 도메인, $P$는 프롬프트 집합(특정 작업에서만 사용), $M$은 세그멘테이션 마스크 집합, $C$는 마스크와 연결된 시맨틱 카테고리의 어휘 집합을 의미한다. 이 수식을 기반으로 작업을 두 가지로 분류한다.

- **Generic Image Segmentation (GIS)**: $P = \emptyset$인 경우로, 이미지 $I$만으로 영역을 분할한다. 여기에는 Semantic, Instance, Panoptic 세그멘테이션이 포함된다.
- **Promptable Image Segmentation (PIS)**: $P \neq \emptyset$인 경우로, 사용자 정의 프롬프트(클릭, 박스, 텍스트, 지원 이미지 등)를 통해 특정 대상을 분할한다. Interactive, Referring, Few-shot 세그멘테이션이 이에 해당한다.

### 2. FM에서의 세그멘테이션 지식 창발 (Emergence of Knowledge)
본 논문은 학습 없이 FM의 내부 표현만으로 마스크를 추출하는 방법을 설명한다.

- **CLIP**: CLIP은 본래 공간 불변(Spatial-invariant) 특징을 학습하므로 위치 정보가 부족하다. 이를 해결하기 위해 Attention Pooling 모듈을 수정하여 공간 공변(Spatial-covariant) 특징을 추출함으로써 세그멘테이션 지식을 얻어낸다.
- **Diffusion Models (DMs)**: 텍스트-이미지 생성 과정의 Cross-attention map이 픽셀과 단어 간의 밀접한 상관관계를 담고 있음을 이용한다.

$$m = \text{CrossAttention}(q, k) = \text{softmax}(qk^\top / \sqrt{d})$$

여기서 $q$는 UNet의 잠재 공간 특징이고 $k$는 텍스트 토큰 임베딩이다. 여기에 Self-attention map을 결합하여 경계선을 정교화한다.
- **DINO**: 클래스 토큰 $[\text{CLS}]$와 패치 토큰 간의 Self-attention map이 자연스럽게 객체의 형태를 담고 있음을 활용한다.

### 3. 학습 패러다임의 변화
기존의 지도(Supervised), 비지도(Unsupervised), 약지도(Weakly-supervised) 학습 외에, **Training-free**라는 새로운 패러다임이 등장하였다. 이는 사전 학습된 FM에서 직접 지식을 추출하며 추가적인 모델 학습이나 파인튜닝을 거치지 않는 방식이다.

## 📊 Results

본 논문은 정량적 실험 결과보다는 수백 개의 기존 방법론을 체계적으로 분류하고 분석하는 데 집중하고 있다. 주요 분석 결과는 다음과 같다.

- **GIS에서의 FM 활용**:
    - **CLIP 기반**: 제로샷 분류기로 활용하거나, 파라미터 효율적 튜닝(PEFT)을 통해 Open-vocabulary 세그멘테이션을 구현한다.
    - **DM 기반**: 생성 모델의 잠재 표현(Latent representation)이 RGB 이미지보다 세그멘테이션에 더 유리한 입력값이 됨을 확인하였다. 특히 UNet의 중간 레이어가 가장 많은 시맨틱 정보를 포함하고 있다.
    - **DINO 기반**: 비지도 학습 환경에서 DINO의 특징을 클러스터링하여 의사 라벨(Pseudo-label)을 생성하고, 이를 통해 모델을 자가 학습(Self-training)시키는 방식이 효과적이다.
    - **SAM 기반**: SAM의 강력한 일반화 능력을 활용해 약지도 학습 상황에서 마스크의 품질을 높이는 후처리 도구로 사용된다.

- **PIS에서의 FM 활용**:
    - **Interactive**: SAM이 표준이 되었으며, HQ-SAM과 같이 출력 토큰을 개선하여 정밀도를 높이거나 MedSAM처럼 의료 영상에 맞게 파인튜닝하는 방향으로 발전하고 있다.
    - **Referring**: LLM/MLLM을 결합하여 "경주에서 누가 이길까?"와 같은 복잡한 추론(Reasoning)이 필요한 쿼리를 픽셀 영역으로 그라운딩(Grounding)하는 Reasoning Segmentation으로 확장되었다.
    - **Few-shot**: 지원 이미지(Support image)와 쿼리 이미지 간의 상관관계를 CLIP이나 DINOv2 특징으로 계산하여 세그멘테이션을 수행한다.

## 🧠 Insights & Discussion

### 강점 및 기회
본 논문은 FM의 등장으로 인해 세그멘테이션 모델이 더 이상 특정 작업에 고정되지 않고, 프롬프트를 통해 다양한 작업에 적응할 수 있는 **Segmentation Generalist**로 진화했음을 보여준다. 특히 MLLM의 추론 능력을 결합함으로써 단순한 객체 인식을 넘어 시각적 상식과 추론을 기반으로 한 세그멘테이션이 가능해졌다.

### 한계 및 미해결 과제
- **창발성의 원인 불명**: FM이 어떻게 픽셀 수준의 이해도를 갖게 되었는지에 대한 이론적 설명이 부족하다.
- **In-context Segmentation의 성능**: 언어 모델의 ICL(In-context Learning)만큼 시각 모델에서의 ICS 성능이 높지 않으며, 특히 Panoptic 세그멘테이션과 같은 복잡한 작업에서는 한계가 있다.
- **MLLM의 환각(Hallucination)**: MLLM 기반 모델들이 이미지에 없는 객체를 있다고 판단하여 잘못된 영역을 세그멘테이션하는 문제가 발생한다.
- **계산 비용**: FM의 거대한 파라미터 사이즈로 인해 추론 비용이 매우 높으며, 이는 실시간 적용의 걸림돌이 된다.

## 📌 TL;DR

본 논문은 Foundation Model(CLIP, DMs, DINO, SAM 등)이 이미지 세그멘테이션 패러다임을 '작업 전용 모델'에서 '프롬프트 기반 범용 모델'로 어떻게 전환시켰는지를 다룬 포괄적인 서베이이다. 특히 학습 없이 지식을 추출하는 Training-free 방식과 LLM을 결합한 Reasoning Segmentation의 부상을 강조한다. 이 연구는 향후 더 효율적이고 추론 능력이 뛰어난 시각 일반 모델(Large Vision Models) 개발 및 합성 데이터를 활용한 확장 가능한 데이터 엔진 구축의 필요성을 제시함으로써 후속 연구에 중요한 이정표 역할을 할 것으로 보인다.