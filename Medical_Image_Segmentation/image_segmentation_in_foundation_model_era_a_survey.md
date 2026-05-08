# Image Segmentation in Foundation Model Era: A Survey

Tianfei Zhou, Wang Xia, Fei Zhang, Boyu Chang, Wenguan Wang, Ye Yuan, Ender Konukoglu, Daniel Cremers (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 핵심 과제인 이미지 세그멘테이션(Image Segmentation)이 파운데이션 모델(Foundation Models, FMs)의 등장으로 인해 어떻게 변화하고 있는지 분석하는 것을 목표로 한다.

기존의 세그멘테이션 연구는 FCN, MaskFormer와 같은 특정 작업 중심의 아키텍처를 통해 발전해 왔으며, 대부분 닫힌 어휘집(Closed-vocabulary) 설정에서 작동하는 한계가 있었다. 그러나 최근 CLIP, Stable Diffusion, DINO, SAM과 같은 대규모 파운데이션 모델들이 등장하면서, 세그멘테이션은 단순한 픽셀 분류를 넘어 제로샷(Zero-shot) 학습, 프롬프트 기반 제어, 그리고 학습이 필요 없는(Training-free) 지식 추출이 가능한 새로운 시대로 진입하였다.

그럼에도 불구하고, 이러한 급격한 변화를 체계적으로 분석하고, 각 파운데이션 모델이 세그멘테이션에 기여하는 방식과 그로 인해 발생하는 새로운 도전 과제들을 정리한 종합적인 리뷰 논문이 부재한 상황이다. 따라서 본 논문은 파운데이션 모델 기반의 이미지 세그멘테이션 연구를 포괄적으로 조사하여 학계에 체계적인 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 파운데이션 모델이 이미지 세그멘테이션 분야에 가져온 패러다임의 전환을 세 가지 관점에서 정의하고 분석한 것이다.

첫째, **세그멘테이션 제너럴리스트(Segmentation Generalists)의 등장**이다. 기존 모델과 달리 현대의 모델들은 프롬프트(Prompt)를 통해 무엇을 세그멘테이션할지 지정할 수 있는 Promptable 인터페이스를 갖추게 되었으며, 이를 통해 제로샷 또는 퓨샷(Few-shot) 방식으로 새로운 작업에 빠르게 적응할 수 있게 되었다.

둘째, **학습 없는 세그멘테이션(Training-free Segmentation)의 가능성**이다. CLIP, Stable Diffusion, DINO와 같이 본래 세그멘테이션을 목적으로 설계되지 않은 모델들의 내부 표현(Internal Representation)이나 어텐션 맵(Attention Map)에서 세그멘테이션 지식을 직접 추출할 수 있음을 확인하였다.

셋째, **추론 능력의 결합**이다. LLM 및 MLLM을 세그멘테이션 시스템에 통합함으로써, "경주에서 누가 이길 것인가?"와 같은 복잡한 추론이 필요한 쿼리를 픽셀 영역으로 그라운딩(Grounding)할 수 있는 Reasoning Segmentation 능력을 확보하게 되었다.

## 📎 Related Works

기존의 세그멘테이션 관련 서베이들은 주로 다음과 같은 한계를 가지고 있다.

- **범위의 제한:** 2021년 이전의 서베이들은 주로 시맨틱(Semantic) 및 인스턴스(Instance) 세그멘테이션에만 집중하였으며, 최신 파운데이션 모델의 영향을 반영하지 못했다.
- **특정 모델 중심:** 최근 일부 연구는 SAM(Segment Anything Model)과 같은 특정 모델만을 집중적으로 다루었으나, 이는 파운데이션 모델 전체의 생태계를 포괄하기에는 부족하다.
- **단일 작업 중심:** 오픈 보캐블러리(Open-vocabulary)나 트랜스포머 기반 모델만을 다룬 연구들이 있었으나, 제너릭 세그멘테이션(GIS)과 프롬프트 기반 세그멘테이션(PIS)을 통합적으로 분석한 사례는 드물다.

본 논문은 이러한 한계를 극복하고, CLIP, Stable Diffusion, DINO, SAM, LLM/MLLM 등 다양한 파운데이션 모델이 세그멘테이션의 전 분야에 어떻게 적용되고 있는지를 통합적으로 분석함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 통합 수식화 (Unified Formulation)

논문은 다양한 세그멘테이션 작업을 다음과 같은 통합 수식으로 정의한다.
$$f: X \to Y, \text{ where } X = I \times P, Y = M \times C$$

- $f$: 신경망으로 구현된 매핑 함수이다.
- $X$: 입력 공간으로, 이미지 $I$와 프롬프트 집합 $P$의 곱으로 구성된다.
- $Y$: 출력 공간으로, 세그멘테이션 마스크 $M$과 시맨틱 카테고리 어휘집 $C$의 곱으로 구성된다.

이 수식을 바탕으로 프롬프트 $P$의 존재 여부에 따라 작업을 두 가지로 분류한다.

- **Generic Image Segmentation (GIS):** $P = \emptyset$인 경우로, 이미지 자체만으로 영역을 분할한다. (Semantic, Instance, Panoptic Segmentation 포함)
- **Promptable Image Segmentation (PIS):** $P \neq \emptyset$인 경우로, 사용자 입력(클릭, 텍스트, 이미지 등)에 따라 영역을 분할한다. (Interactive, Referring, Few-shot Segmentation 포함)

### 2. 파운데이션 모델에서의 세그멘테이션 지식 추출 (Emergence of Knowledge)

본 논문은 세그멘테이션 목적이 아닌 모델에서 어떻게 마스크가 추출되는지 상세히 설명한다.

**A. CLIP 기반 추출**
CLIP은 본래 공간 불변(Spatial-invariant) 특징을 학습하여 위치 정보가 부족하다. 이를 해결하기 위해 Self-attention의 어텐션 풀링(Attention Pooling) 구조를 수정하여 공간 공변(Spatial-covariant) 특징을 얻는다. 예를 들어, 어텐션 행렬을 단위 행렬(Identity Matrix)로 설정하여 각 토큰이 자신의 위치 정보만 유지하게 함으로써 세그멘테이션 성능을 높일 수 있다.

**B. Diffusion Models (DMs) 기반 추출**
텍스트-이미지 확산 모델의 Cross-attention 맵에서 픽셀과 단어 간의 밀집된 상관관계를 추출한다.
$$m = \text{CrossAttention}(q, k) = \text{softmax}(qk^\top / \sqrt{d})$$
여기서 $q$는 이미지 특징, $k$는 텍스트 토큰 특징이다. 이를 통해 특정 클래스 토큰 $[CLS]$에 대응하는 마스크 $m_{CLS}$를 얻으며, 경계선을 명확히 하기 위해 Self-attention 맵과 결합하여 최종 마스크 $\hat{m}_{CLS} = a_{SA} m_{CLS}$를 생성한다.

**C. DINO 기반 추출**
DINO와 DINOv2의 마지막 레이어에 있는 $[CLS]$ 토큰의 self-attention에서 객체 분할 정보가 자연스럽게 나타난다.
$$\alpha_{CLS} = q_{CLS} \cdot k_I^\top$$
이 어피니티 벡터(Affinity Vector) $\alpha_{CLS}$를 이진화함으로써 별도의 학습 없이도 객체 마스크를 얻을 수 있다.

### 3. 모델별 적용 전략

- **CLIP:** 제로샷 분류기로 사용하거나, 픽셀-텍스트 정렬을 위한 파인튜닝(Fine-tuning) 및 지식 증류(Knowledge Distillation)를 통해 적용한다.
- **Diffusion Models:** 어텐션 맵을 이용한 학습 없는 세그멘테이션, 잠재 표현(Latent Representation) 추출, 또는 세그멘테이션 자체를 노이즈 제거 과정(Denoising process)으로 재정의하여 해결한다.
- **SAM:** 강력한 제로샷 마스크 생성 능력을 활용하여 약지도 학습(Weakly Supervised)의 포스트 프로세싱이나 데이터 엔진으로 활용한다.
- **LLM/MLLM:** 복잡한 쿼리를 이해하여 $[seg]$ 토큰과 같은 특수 토큰을 생성하고, 이를 SAM 디코더 등에 전달하여 Reasoning Segmentation을 수행한다.

## 📊 Results

본 논문은 300개 이상의 세그멘테이션 접근 방식을 리뷰하며 다음과 같은 정성적/정량적 경향성을 분석하였다.

- **GIS 분야:** CLIP 기반 모델들은 오픈 보캐블러리(Open-vocabulary) 성능을 비약적으로 향상시켰으며, DINO 기반의 비지도 학습 방법들은 인간의 레이블 없이도 의미 있는 객체 군집화를 가능케 하였다.
- **PIS 분야:** SAM의 등장으로 인터랙티브 세그멘테이션의 범용성이 극대화되었으며, 특히 의료 영상 분야(MedSAM 등)에서 도메인 적응(Domain Adaptation)을 통한 성능 향상이 두드러졌다.
- **Reasoning Segmentation:** LISA와 같은 MLLM 기반 모델들은 기존의 Referring Segmentation이 해결하지 못했던 "추론이 필요한 쿼리"에 대해 성공적으로 마스크를 생성할 수 있음을 보여주었다.
- **In-Context Segmentation (ICS):** SegGPT와 같은 모델들은 몇 개의 예시(Support-Query pair)만으로 새로운 객체를 세그멘테이션하는 능력을 보여, 시각 지능의 'In-context learning' 가능성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 성과

파운데이션 모델의 도입은 세그멘테이션 연구를 '특정 데이터셋에 최적화된 모델 개발'에서 '범용적인 시각 이해 시스템 구축'으로 전환시켰다. 특히, 서로 다른 강점을 가진 모델들(시맨틱 이해의 CLIP, 공간 이해의 SAM/DINO)을 결합하여 상호 보완적인 시스템을 구축하는 경향이 뚜렷하다.

### 한계 및 미해결 과제

1. **지식 발현의 원인 불명:** 왜 서로 다른 목적(생성, 대조 학습 등)으로 학습된 모델들에서 공통적으로 세그멘테이션 지식이 나타나는지에 대한 이론적 설명이 부족하다.
2. **객체 환각(Object Hallucination):** MLLM 기반 모델들은 텍스트 생성 모델과 마찬가지로 이미지에 없는 객체를 있다고 판단하고 마스크를 생성하는 환각 현상이 발생한다.
3. **계산 효율성 문제:** 파운데이션 모델의 거대한 크기로 인해 실시간 추론이 어렵다. 지식 증류(Distillation)나 모델 압축 기술의 적용이 시급하다.
4. **데이터 엔진의 한계:** SAM의 SA-1B와 같은 거대 데이터셋이 구축되었으나, 여전히 시맨틱 정보가 부족하며 의료/위성 영상과 같은 특수 도메인 데이터는 여전히 희소하다.

## 📌 TL;DR

본 논문은 파운데이션 모델(CLIP, Diffusion Models, DINO, SAM, LLM)이 이미지 세그멘테이션에 미친 영향을 종합적으로 분석한 최초의 서베이 논문이다. 특히 **제너릭 세그멘테이션(GIS)**과 **프롬프트 기반 세그멘테이션(PIS)**으로 체계를 나누고, 학습 없이도 모델 내부에서 세그멘테이션 지식을 추출하는 방법론을 상세히 다루었다. 이 연구는 향후 시각 지능 모델이 단순한 픽셀 분류기를 넘어, 복잡한 추론과 제로샷 적응 능력을 갖춘 '시각 제너럴리스트'로 진화하는 방향을 제시하고 있다.
