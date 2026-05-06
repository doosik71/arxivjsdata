# Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models

Siddharth Karamcheti et al. (2024)

## 🧩 Problem to Solve

최근 LLaVa, InstructBLIP, PaLI-3와 같이 시각 정보를 조건으로 하는 언어 모델(Visually-Conditioned Language Models, VLMs)이 빠르게 등장하고 있으나, 정작 모델의 성능을 결정짓는 핵심 설계 결정 요소들에 대한 체계적인 연구는 부족한 실정이다. 특히 이미지 전처리 방법, 아키텍처 구성, 최적화 절차 등이 성능에 어떤 영향을 미치는지에 대한 분석이 미비하며, 이를 객관적으로 측정할 수 있는 일관된 평가 체계 또한 부족하다.

본 논문의 목표는 VLM의 설계 공간(Design Space)을 엄밀하게 조사하여 성능에 영향을 주는 핵심 요소들을 식별하고, 이를 바탕으로 최적의 설계를 갖춘 PRISM 모델 제품군을 제안하는 것이다. 또한, 미래의 VLM 연구자들이 활용할 수 있도록 표준화된 평가 세트와 유연한 학습 프레임워크를 제공하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 VLM을 구성하는 다양한 설계 축(Design Axes)을 독립적으로 제어하며 실험함으로써, 각 구성 요소가 하위 작업(downstream tasks) 성능에 미치는 영향을 정량적으로 분석하는 것이다. 주요 기여 사항은 다음과 같다.

1. **표준화된 평가 세트 구축**: VQA, 객체 로컬라이제이션(Object Localization), 그리고 환각(Hallucination) 및 공간 추론 등을 측정하는 챌린지 세트를 포함한 총 12개의 벤치마크를 통합하여 세밀한 분석이 가능하게 하였다.
2. **유연한 학습 코드베이스 제공**: 시각 백본, 언어 모델, 최적화 절차를 쉽게 교체할 수 있는 모듈형 오픈소스 프레임워크를 개발하여 재현성과 실험 효율성을 높였다.
3. **VLM 설계 공간의 엄밀한 조사**: 최적화 절차, 시각적 표현, 언어 모델 선택, 스케일링 특성이라는 4가지 설계 축을 중심으로 실험을 수행하여 핵심 인사이트를 도출하였다.
4. **PRISM 모델 제안**: 도출된 인사이트를 결합하여 7B 및 13B 규모의 PRISM 모델을 학습시켰으며, 이는 LLaVa v1.5 및 InstructBLIP과 같은 기존 최신 오픈 VLM의 성능을 상회한다.

## 📎 Related Works

기존의 VLM들은 복잡한 아키텍처 대신, 사전 학습된 시각 백본(예: CLIP)에서 추출한 패치 특징(patch features)을 언어 모델의 입력 공간으로 투영하는 'patch-as-token' 방식을 채택하는 추세이다. LLaVa v1.5나 PaLI-3가 이러한 레시피를 따르고 있다.

그러나 기존 연구들은 특정 구성 요소(데이터, 백본 등)만을 변경하여 결과를 보고할 뿐, 전체 설계 공간에 대한 포괄적인 조사를 수행하지 않았다. 본 논문은 단순히 더 좋은 모델을 만드는 것이 아니라, 어떤 설계 선택이 왜 성능 향상으로 이어지는지를 분석함으로써 기존 접근 방식의 한계를 극복하고 설계 가이드를 제시한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 논문은 일반적인 VLM 아키텍처를 따르며, 크게 세 가지 구성 요소로 이루어진다.

1. **Visual Representation Backbone ($V_\omega$)**: 입력 이미지 $x_{img} \in \mathbb{R}^{H \times W}$를 받아 패치 특징 시퀀스 $p_{img} \in \mathbb{R}^{L \times h_{vision}}$를 출력한다.
2. **Vision-Language Projector ($F_\psi$)**: 시각 특징 $p_{img}$를 언어 모델의 임베딩 공간인 $e_{img} \in \mathbb{R}^{L \times h_{text}}$로 매핑한다. 본 연구에서는 2층 구조의 GELU MLP를 사용하였다.
3. **Language Model ($LM_\theta$)**: 투영된 시각 임베딩 $e_{img}$와 텍스트 프롬프트 임베딩 $e_{prompt}$를 결합하여 최종 텍스트 $u_{gen}$을 생성한다.

전체 모델의 구성은 다음과 같은 수식으로 정의된다.
$$u_{gen} = LM_\theta([F_\psi(V_\omega(x_{img})); \text{embed}(u_{prompt})])$$

### 학습 목표 및 손실 함수

학습 단계에서는 이미지-텍스트 쌍 $(x_{img}, u_{prompt}, \hat{u}_{gen})$이 주어졌을 때, 정답 토큰의 생성 확률을 최대화하는 next-token prediction 목적 함수를 사용한다.
$$L(\omega, \psi, \theta) = -\log p(\hat{u}_{gen} | x_{img}, u_{prompt})$$
이 손실 함수를 경사 하강법(gradient descent)을 통해 최소화한다.

### 학습 절차 및 구현 세부사항

- **구현**: PyTorch 및 Fully Sharded Data Parallel (FSDP)를 사용하여 BF16 혼합 정밀도로 학습하였다.
- **데이터**: LLaVa v1.5 데이터 믹스처를 사용하였으며, 이는 캡셔닝 데이터(558K)와 멀티모달 인스트럭트 튜닝 데이터(665K)로 구성된다.
- **최적화**: AdamW 옵티마이저, Warmup 및 Cosine Decay 스케줄러를 사용하였으며, 학습률은 $2\text{e-5}$로 설정하였다.

## 📊 Results

### 실험 설정 및 지표

- **데이터셋 및 작업**: VQA(VQAv2, GQA 등), 로컬라이제이션(RefCOCO 등), 챌린지 세트(VSR, POPE 등) 등 12개 벤치마크를 사용하였다.
- **기준선**: LLaVa v1.5, InstructBLIP 등이 비교 대상으로 사용되었다.
- **측정 방법**: 각 모델의 성능을 정규화된 Z-score로 계산하여 통계적 유의성($p < 0.01$)을 검증하였다.

### 주요 설계 축별 결과

1. **최적화 절차 (Optimization Procedure)**:
    - **단일 단계 학습 (Single-stage)**: 기존의 2단계 학습(정렬 $\rightarrow$ 튜닝)에서 첫 번째 단계를 생략해도 성능이 유지되거나 오히려 향상됨을 확인하였다. 이를 통해 학습 비용을 20-25% 절감하였다.
    - **시각 백본 파인튜닝**: 시각 백본을 고정하지 않고 함께 학습시켰을 때, 오히려 대부분의 벤치마크(특히 로컬라이제이션)에서 성능이 크게 저하되었다.

2. **이미지 처리 및 시각적 표현 (Image Processing & Visual Representations)**:
    - **백본 비교**: CLIP과 SigLIP 같은 시각-언어 대조 학습(contrastive objective) 기반 모델이 DINOv2나 ImageNet 기반 모델보다 월등히 성능이 좋았다.
    - **전처리 방법**: 단순 리사이즈(Naive Resize) 방식이 Letterbox 패딩이나 Resize & Crop보다 CLIP 모델에서 더 좋은 성능을 보였다.
    - **해상도**: 입력 해상도를 336px 또는 384px로 높였을 때 성능이 유의미하게 향상되었다.
    - **특징 융합 (Ensembling)**: **SigLIP(시맨틱 특징)과 DINOv2(공간적 특징)의 특징을 결합**했을 때, 로컬라이제이션 및 챌린지 작업에서 5-10%의 큰 성능 향상이 나타났다.

3. **언어 모델 (Language Models)**:
    - **Base vs. Instruct-tuned**: 정량적 성능은 비슷하나, Base LM(Llama-2) 기반 모델이 Instruct-tuned LM(Vicuna) 기반 모델보다 덜 장황하고 환각(hallucination) 증상이 적어 정성적으로 더 우수하였다.
    - **안전성**: 언어 전용 안전 데이터(ShareGPT)를 함께 학습시키는 것이 Base LM 기반 모델의 유해한 출력을 방지하는 데 필수적임을 확인하였다.

4. **스케일링 특성 (Scaling Properties)**:
    - **학습 시간**: 1 epoch 학습은 심각한 언더피팅(underfitting) 상태이며, 2 epoch까지 학습했을 때 성능이 정점에 도달하였다.
    - **데이터 다양성**: LRV-Instruct와 같이 이미지 다양성이 높은 데이터를 추가했을 때 성능이 향상되었다.

### PRISM 모델 결과

위의 인사이트를 모두 적용한 PRISM 모델은 LLaVa v1.5 및 InstructBLIP을 모든 지표에서 압도하였다. 특히 동일한 데이터와 학습 예산을 사용한 'Controlled' 실험에서도 PRISM의 설계 최적화만으로 상당한 성능 향상을 이루어냈다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문의 가장 큰 강점은 VLM 설계의 불확실성을 제거하기 위해 매우 엄격한 ablation study를 수행했다는 점이다. 특히 단순히 모델의 크기를 키우는 것이 아니라, 시각 백본의 융합(SigLIP + DINOv2)이나 학습 epoch 수의 조정과 같은 작은 설계 변경이 실질적으로 큰 성능 차이를 만든다는 것을 입증하였다.

### 한계 및 미해결 질문

- **아키텍처의 일반성**: 본 연구는 '백본-프로젝터-LM'이라는 3단 구조에 집중하였다. 따라서 Flamingo나 IDEFICS에서 사용하는 Perceiver-resampler와 같이 패치를 다운샘플링하는 구조에 대한 영향은 분석하지 못하였다.
- **평가의 범위**: 객관적인 벤치마크 위주로 평가하였기에, 실제 사용자와의 긴 대화나 복잡한 멀티턴 인터랙션에서의 성능은 충분히 검증되지 않았다.
- **규모의 확장성**: 7B-13B 규모에서의 결과가 70B 이상의 초거대 모델에서도 동일하게 적용될지는 미지수이다.

### 비판적 해석

Base LM이 Instruct-tuned LM보다 VLM의 기반 모델로서 더 적합하다는 발견은 흥미롭다. 이는 인스트럭트 튜닝 과정에서 발생하는 언어적 편향(bias)이 시각 정보와의 정렬을 방해할 수 있음을 시사하며, VLM을 구축할 때는 언어 모델의 '채팅 능력'보다 '기초 언어 능력'과 '유연성'이 더 중요함을 보여준다.

## 📌 TL;DR

본 논문은 VLM 성능에 영향을 주는 4가지 설계 축(최적화, 시각 표현, 언어 모델, 스케일링)을 체계적으로 분석하여, **단일 단계 학습, SigLIP+DINOv2 특징 융합, Base LM 사용, 2 epoch 학습**이 최적의 조합임을 밝혀냈다. 이를 통해 구현된 **PRISM** 모델은 기존 SOTA 오픈 VLM들을 능가하는 성능을 보였으며, 함께 공개된 표준 평가 세트와 모듈형 학습 코드는 향후 VLM 연구의 중요한 기초 자산이 될 것으로 기대된다.
