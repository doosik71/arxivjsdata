# Heterogeneous Contrastive Learning for Foundation Models and Beyond

Lecheng Zheng, Baoyu Jing, Zihao Li, Hanghang Tong, and Jingrui He (2024)

## 🧩 Problem to Solve

현대 빅데이터의 핵심적인 특징 중 하나는 **Heterogeneity(이질성)**이다. 데이터는 다양한 소스에서 수집되며 서로 다른 작업(Task)과 연관되어 있는데, 이는 크게 **View Heterogeneity(뷰 이질성)**와 **Task Heterogeneity(작업 이질성)**로 나타난다. 예를 들어, 소셜 미디어의 게시물은 텍스트, 이미지, 비디오가 혼합된 뷰 이질성을 가지며, 금융 데이터는 수치 데이터와 텍스트 뉴스가 공존하는 형태를 띤다.

이러한 이질적 데이터를 효과적으로 모델링하기 위해 Contrastive Self-supervised Learning(대조 자기지도 학습)을 활용한 Foundation Model(기반 모델)의 중요성이 커지고 있다. 하지만 기존의 서베이 논문들은 단일 이질성(뷰 또는 작업 중 하나)에만 집중하거나, 전통적인 대조 학습 방법론에 국한되어 최신 기반 모델의 기술적 흐름을 체계적으로 통합하여 분석하지 못했다는 한계가 있다. 따라서 본 논문은 뷰와 작업 이질성을 모두 포괄하는 대조 학습 기반의 기반 모델들에 대한 종합적인 분석 보고서를 제공하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 대조 학습 기반의 기반 모델들을 **View Heterogeneity**와 **Task Heterogeneity**라는 두 가지 핵심 축으로 분류하여 체계적인 Taxonomy(분류 체계)를 정립한 것이다.

1.  **분류 체계 제안**: 대조 기반 모델을 뷰 이질성 해결 모델과 작업 이질성 해결 모델로 나누어 분석함으로써, 데이터의 형태적 차이와 목적의 차이를 동시에 다루었다.
2.  **기술적 리뷰**: Large Vision Model(LVM), Large Language Model(LLM), Multi-modal Foundation Model 등 각 도메인에서 대조 학습이 어떻게 적용되었는지 상세히 분석하고 비교하였다.
3.  **학습 전략 분석**: Pre-training(사전 학습) 단계의 Pretext task, Supervised task, Preference task, Auxiliary task와 Downstream task를 연결하는 전략(AutoML, Prompt Learning 등)을 체계화하였다.
4.  **미래 방향 제시**: 표현의 중복성, 계산 효율성, 벤치마크 데이터셋의 필요성, 신뢰성(Trustworthy CL) 등 향후 연구 방향을 제시하였다.

## 📎 Related Works

논문에서는 기존 서베이 연구들의 한계를 다음과 같이 지적하며 차별점을 제시한다.

-   **단일 이질성 집중 연구**: 일부 연구들은 Multi-view learning이나 Multi-label learning과 같은 특정 이질성만 다루었으며, 대조 학습이나 기반 모델과의 연계성을 다루지 않았다.
-   **전통적 대조 학습 연구**: 최신 대조 학습 기법을 다룬 연구들이 있으나, 이는 주로 전통적인 방식에 머물러 있으며 최신 Foundation Model의 규모와 특성을 반영하지 못했다.
-   **멀티모달 기반 모델 연구**: 최근의 멀티모달 기반 모델 서베이들은 주로 Large Language Model(LLM)의 확장판에 집중되어 있어, 다양한 데이터 타입(그래프, 시계열 등)에 대한 대조 학습 적용 사례를 포괄하지 못했다.

본 논문은 이러한 공백을 메우기 위해 뷰와 작업의 이질성을 동시에 고려하며, CV, NLP를 넘어 그래프 및 시계열 데이터까지 확장된 분석을 수행한다.

## 🛠️ Methodology

본 논문은 서베이 논문으로서 특정 알고리즘을 제안하기보다, 대조 학습의 일반적인 프레임워크와 이를 기반 모델에 적용하는 구조를 분석한다.

### 1. 대조 학습의 기본 파이프라인
대조 학습은 기본적으로 다음의 세 단계로 구성된다.
-   **Augmentation(증강)**: 원본 데이터나 임베딩에 변형을 가해 증강된 샘플을 생성한다.
-   **Contrastive Pair Construction(대조 쌍 구성)**: 긍정 쌍(Positive pair, 유사한 샘플)과 부정 쌍(Negative pair, 서로 다른 샘플)을 정의한다.
-   **Loss Function Formulation(손실 함수 정의)**: 긍정 쌍의 유사도는 높이고 부정 쌍의 유사도는 낮추도록 학습한다.

### 2. 핵심 방정식 (Contrastive Loss)
본문에서는 Noise Contrastive Estimation (NCE) 아이디어를 확장한 손실 함수를 정의한다.

$$
\mathcal{L} = -\mathbb{E}_{x_i \in \mathcal{D}} \log \frac{\exp(\text{sim}(z_i, z_i^+))}{\exp(\text{sim}(z_i, z_i^+)) + \sum_{k \neq i} \exp(\text{sim}(z_i, z_k^-))}
$$

-   $z_i$: 입력 샘플 $x_i$의 학습된 표현(Representation)이다.
-   $z_i^+$: $x_i$와 유사한 긍정 샘플의 표현이다.
-   $z_k^-$: $x_i$와 서로 다른 부정 샘플의 표현이다.
-   $\text{sim}(\cdot)$: 유사도 측정 함수(예: Cosine similarity)이며, 일반적으로 온도 하이퍼파라미터 $g$로 스케일링된다.

또한, 앵커-긍정-부정 샘플 간의 거리를 직접 비교하는 Triplet Loss 또한 언급된다.
$$
\mathcal{L} = \sum_{i} \max(0, \|z_i - z_i^+\|_2^2 - \|z_i - z_i^-\|_2^2 + \alpha)
$$
여기서 $\alpha$는 긍정 쌍과 부정 쌍 사이의 최소 거리 차이를 강제하는 마진(Margin) 값이다.

### 3. 뷰 이질성(View Heterogeneity) 처리 방식
-   **Global View**: 이미지의 회전, 색상 변형이나 그래프 전체의 구조 변경과 같이 샘플 전체를 변형하는 방식이다.
-   **Local View**: 이미지 크롭(Crop)이나 그래프의 노드/엣지 제거와 같이 부분적인 영역을 변형하는 방식이다.
-   **Cross-modal View**: 이미지-텍스트 쌍과 같이 서로 다른 모달리티 자체를 서로 다른 뷰로 간주하여 정렬(Alignment)하는 방식이다.

## 📊 Results

본 논문은 수많은 기존 연구들을 분류하여 다음과 같은 결과적 통찰을 제공한다.

### 1. 도메인별 기반 모델 적용 현황
-   **Large Vision Models (LVM)**: SimCLR, MoCo, BYOL 등이 대표적이다. 특히 SimCLR는 강력한 데이터 증강과 Projection head 도입을 통해 성능을 높였으며, MoCo는 Momentum-based dictionary를 사용하여 GPU 메모리 제약을 극복했다.
-   **Large Language Models (LLM)**: SimCSE, ContraCLM 등이 있으며, 주로 Dropout 기반의 증강이나 역번역(Back-translation)을 통해 뷰를 생성하여 문장 임베딩을 학습한다.
-   **Multi-modal Foundation Models**: CLIP이 가장 대표적이며, 이미지와 텍스트 뷰 사이의 상호 대조 학습을 통해 Zero-shot 전이 능력을 확보했다. 이후 CLAP(오디오-텍스트), ConGraT(그래프-텍스트) 등으로 확장되었다.

### 2. 작업 이질성(Task Heterogeneity) 해결 전략
사전 학습(Pre-training)과 다운스트림 작업(Downstream Task)을 연결하는 네 가지 주요 전략이 분석되었다.
-   **AutoML**: 최적의 대조 학습 전략(증강 방법, 손실 함수 등)을 자동으로 탐색한다.
-   **Prompt Learning**: Few-shot 또는 Zero-shot 설정에서 대조 학습을 통해 프롬프트를 최적화하여 미세 조정한다.
-   **Multi-Task Learning**: 대조 학습 손실과 특정 작업의 손실(예: 캡션 생성 손실)을 동시에 최적화하여 전역적 표현과 세부 특징을 모두 학습한다.
-   **Task Reformulation**: 분류나 클러스터링 같은 다운스트림 작업을 본질적으로 대조 학습 문제로 재정의하여 직접 해결한다.

## 🧠 Insights & Discussion

### 강점 및 분석 결과
본 논문은 파편화되어 있던 대조 학습 연구들을 '이질성(Heterogeneity)'이라는 관점에서 통합함으로써, 기반 모델이 단순한 규모의 확장을 넘어 데이터의 다양한 형태와 목적을 어떻게 수용해야 하는지에 대한 이론적 틀을 제공한다. 특히 뷰 이질성과 작업 이질성을 분리하여 분석함으로써, 연구자가 자신의 문제 상황에 맞는 대조 학습 전략을 선택할 수 있는 가이드를 제시했다.

### 한계 및 비판적 해석
1.  **계산 효율성 문제**: 논문에서도 언급되었듯, 대조 학습은 성능 향상을 위해 큰 배치 사이즈(Batch size)가 필수적이며 이는 막대한 GPU 메모리 비용을 초래한다. Zeroth-order optimization 같은 대안이 제시되었으나, 이는 다시 최적화 효율 저하라는 Trade-off를 가진다.
2.  **데이터 편향(Bias)**: 대규모 벤치마크 데이터셋을 사용한 기반 모델들이 인구통계학적, 고정관념적 편향을 내포하고 있을 가능성이 크며, 이를 해결하기 위한 Trustworthy CL 연구는 아직 초기 단계에 머물러 있다.
3.  **메커니즘의 불투명성**: 특정 대조 학습 전략이 왜 특정 다운스트림 작업에 더 효과적인지에 대한 근본적인 메커니즘 분석이 여전히 부족하며, 이는 향후 정교한 이론적 연구가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 빅데이터의 **뷰 이질성(View Heterogeneity)**과 **작업 이질성(Task Heterogeneity)**을 해결하기 위한 **대조 학습 기반의 Foundation Model**들에 대한 포괄적인 서베이이다. 대조 학습의 기본 원리부터 LVM, LLM, 멀티모달 모델로 이어지는 기술 흐름을 체계화하였으며, 특히 사전 학습 작업과 다운스트림 작업을 연결하는 전략들을 분류하였다. 이 연구는 향후 더 효율적이고 신뢰할 수 있는 멀티모달 기반 모델을 설계하는 데 있어 중요한 참조 지도가 될 가능성이 높다.