# VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, Lidong Bing (2024)

## 🧩 Problem to Solve

비디오 이해를 위한 대규모 언어 모델(Video-LLMs)은 정적 이미지 처리 모델(Image-LLMs)에 비해 발전 속도가 느리며, 이는 비디오 데이터가 가진 고유한 복잡성 때문이다. 본 논문이 해결하고자 하는 핵심 문제는 크게 두 가지이다.

첫째, **시공간적 동역학(Spatial-Temporal Dynamics) 모델링의 한계**이다. 기존의 Video-LLM들은 프레임 간의 특징을 융합하는 능력이 부족하여, 시간에 따른 시각적 패턴의 변화를 충분히 포착하지 못한다. 이는 미래 상태 예측이나 복잡한 상호작용을 이해하는 능력을 저하시키는 원인이 된다.

둘째, **오디오 스트림 통합의 부족**이다. 오디오는 비디오 장면을 완전히 이해하는 데 필수적인 풍부한 문맥적 단서를 제공함에도 불구하고, 많은 모델이 이를 간과하거나 제대로 통합하지 못해 다중 모달 분석 능력이 제한되는 문제가 있다.

따라서 본 연구의 목표는 시공간 모델링 능력을 강화하고 오디오 이해도를 높인 일반ist Video-LLM인 VideoLLaMA 2를 개발하여, 비디오 및 오디오 기반 태스크에서 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 비디오의 시공간적 세부 사항을 효율적으로 압축하면서도 순서를 보존하는 전용 커넥터를 도입하고, 오디오 브랜치를 공동 학습(Joint Training)을 통해 통합하는 것이다.

1. **Spatial-Temporal Convolution (STC) Connector**: 기존의 Q-former나 resampler 구조가 시공간적 순서를 보장하지 못해 LLM의 자기회귀(Autoregressive) 특성과 충돌한다는 점에 착안하여, 3D Convolution 기반의 STC 커넥터를 제안한다. 이를 통해 토큰 수를 줄이면서도 시공간적 국소 세부 사항을 효과적으로 유지한다.
2. **Joint-trained Audio Branch**: BEATs 오디오 인코더와 MLP 프로젝터를 통해 오디오 신호를 LLM에 정렬시키고, 시각 정보와 오디오 정보를 함께 학습시키는 공동 학습 단계를 통해 오디오-비주얼 통합 이해 능력을 향상시킨다.

## 📎 Related Works

기존의 Video-LLM들은 주로 사전 학습된 시각 인코더(CLIP, DINO 등)와 언어 디코더(LLaMA-2, Mistral 등)를 연결하는 어댑터(Adapter) 구조를 사용한다.

- **기존 접근 방식의 한계**: Cross-Attention, Q-Former, Linear Projection 등의 어댑터들은 주로 정적 이미지-텍스트 정렬에 최적화되어 있다. 이러한 방식들은 시간적 집계(Temporal Aggregation)를 완전히 무시하고 모든 시공간 모델링 책임을 언어 디코더(LLM)에게 전가한다. 이는 LLM이 처리해야 할 토큰 수를 과도하게 늘려 계산 효율성을 떨어뜨릴 뿐만 아니라, 비디오 이해 성능 자체를 저하시키는 결과를 초래한다.
- **차별점**: VideoLLaMA 2는 STC 커넥터를 통해 LLM으로 전달되기 전 단계에서 명시적으로 시공간적 특징을 추출하고 압축함으로써, 효율성과 효과성을 동시에 달성한다. 또한, 단순한 모달리티 추가를 넘어 오디오-비주얼 데이터를 함께 학습시키는 다단계 훈련 전략을 사용한다.

## 🛠️ Methodology

### 전체 시스템 구조

VideoLLaMA 2는 **Vision-Language Branch**와 **Audio-Language Branch**라는 두 개의 독립적인 브랜치로 구성된 듀얼 브랜치 프레임워크를 따른다. 각 브랜치는 전용 인코더를 통해 특징을 추출하고, 커넥터(또는 프로젝터)를 통해 LLM의 차원으로 정렬된 후 LLM으로 입력된다.

### 주요 구성 요소

#### 1. Vision-Language Branch

- **Vision Encoder**: CLIP (ViT-L/14)을 사용한다. 정적 이미지 인코더를 사용함으로써 다양한 프레임 샘플링 전략에 유연하게 대응할 수 있다.
- **STC Connector**: 시공간적 표현 학습을 위한 핵심 모듈이다. 다음의 세 가지 설계 원칙을 따른다.
  - 시공간적 토큰의 순서 유지 (Resampler 지양, Convolution/Pooling 지향)
  - 시공간 토큰 수의 감소 (3D Downsampling 적용)
  - 다운샘플링 과정에서의 정보 손실 완화 (RegStage 블록 배치)

STC 커넥터의 구조는 $\text{RegStage} \rightarrow \text{3D Convolution (Downsampler)} \rightarrow \text{RegStage} \rightarrow \text{MLP Projection}$ 순으로 구성된다. 여기서 RegStage는 강력한 컨볼루션 블록으로, 공간적 이해도를 높이는 역할을 한다.

#### 2. Audio-Language Branch

- **Audio Encoder**: 오디오 신호를 fbank spectrogram으로 변환한 후, 사전 학습된 BEATs 인코더를 통해 특징을 추출한다.
- **Audio Projector**: 두 개의 선형 레이어로 구성된 MLP 블록을 사용하여 BEATs의 출력 특징을 LLM의 임베딩 차원과 정렬시킨다.

#### 3. LLM Backbone

Mistral-Instruct (7B), Mixtral-Instruct (8x7B), Qwen2-Instruct (7B, 72B) 등을 디코더로 사용한다.

### 학습 절차 (Training Pipeline)

학습은 총 세 단계로 진행된다.

1. **Video-Language Training**:
    - **Pre-training**: 대규모 웹 크롤링 데이터(Panda-70M 등)를 사용하며, 인코더와 LLM은 고정(Frozen)하고 STC 커넥터만 최적화한다.
    - **Multi-task Fine-tuning**: 캡셔닝, 분류, VQA 및 지시어 튜닝 데이터를 사용하여 LLM과 STC 커넥터를 함께 최적화한다.

2. **Audio-Language Training**:
    - **Pre-training**: WavCaps 데이터를 사용하여 오디오 프로젝터를 최적화한다.
    - **Multi-task Fine-tuning**: 다양한 오디오 태스크 데이터셋을 통해 오디오 인코더와 프로젝터를 최적화한다.

3. **Audio-Video Joint Training**:
    - 오디오-비주얼 쌍 데이터(AVQA 등)를 활용하여 시각과 청각 정보의 상호작용을 학습한다.
    - 데이터 샘플링 비율을 $\text{Audio-Visual} : \text{Visual} : \text{Audio} = 2 : 1 : 1$로 설정하여 균형 있게 학습하며, LLM, 오디오 인코더, 프로젝터를 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**: MC-VQA (EgoSchema, MV-Bench, VideoMME), OE-VQA (MSVD-QA, ActivityNet-QA), Video Captioning (MSVC), 그리고 오디오 관련 태스크(Clotho-AQA, TUT2017 등)를 평가한다.
- **지표**: Top-1 Accuracy 및 GPT-3.5를 활용한 정성적 평가(Correctness, Detailedness 등)를 사용한다.

### 주요 결과

1. **비디오 이해 (Video Understanding)**:
    - **MC-VQA**: VideoLLaMA 2-7B는 EgoSchema에서 51.7%의 정확도를 기록하며 기존 SOTA인 LLaVA-NeXT-Video(43.9%)를 크게 상회한다. 특히 MV-Bench에서는 유료 모델인 GPT4-V(43.7%)보다 높은 성능(53.9%)을 보였다.
    - **Video Captioning**: MSVC 벤치마크에서 오픈소스 모델 중 가장 높은 점수를 기록하며 GPT4-V에 근접한 성능을 보였다.
    - **OE-VQA**: MSVD-QA에서 오픈소스 모델 중 최상위 성능을 보였으나, ActivityNet-QA에서는 LLaVA-NeXT-Video에 비해 다소 낮은 성능을 보였다.

2. **오디오 이해 (Audio Understanding)**:
    - **AQA (Audio-only QA)**: VideoLLaMA 2-AV (7B)는 훨씬 더 많은 데이터로 학습된 Qwen-Audio보다 Clotho-AQA와 TUT2017에서 더 높은 정확도를 달성하여 학습 효율성을 입증했다.
    - **OE-AVQA (Audio-Video QA)**: MUSIC-QA, VGGSound 등에서 기존 오픈소스 모델들을 압도하는 성능을 보였으며, 특히 VGGSound(70.9%)에서 매우 강력한 성능을 나타냈다.

## 🧠 Insights & Discussion

### 강점

VideoLLaMA 2는 STC 커넥터를 통해 시공간적 특징을 효율적으로 압축하고 보존함으로써, 토큰 수의 증가 없이도 비디오의 동적인 변화를 잘 포착한다. 또한, 오디오 브랜치의 통합 학습을 통해 시청각 정보가 상호 보완적으로 작용하도록 하여, 단순한 시각 정보만으로는 파악하기 어려운 맥락을 오디오를 통해 보완하는 능력을 갖추었다.

### 한계 및 비판적 해석

OE-VQA의 일부 벤치마크(ActivityNet-QA 등)에서 LLaVA-NeXT-Video보다 성능이 낮게 나타난 점은 주목할 만하다. 저자들은 이에 대해 LLaVA-NeXT-Video가 방대한 양의 정적 이미지 데이터로 학습되었기 때문에, 비디오 작업 중에서도 정적인 시각 정보가 중요한 태스크에서는 더 유리할 수 있다는 가설을 제시한다. 이는 비디오 모델이라 할지라도 고품질의 정적 이미지 데이터 학습이 여전히 중요하다는 것을 시사한다.

또한, STC 커넥터의 3D Convolution 구조가 효과적임은 입증되었으나, 매우 긴 비디오(Long-form Video)에 대한 처리 효율성과 메모리 제한 문제에 대해서는 구체적인 해결책이나 분석이 부족하다.

## 📌 TL;DR

VideoLLaMA 2는 **3D Convolution 기반의 STC 커넥터**를 도입하여 비디오의 시공간적 동역학 모델링 능력을 강화하고, **공동 학습된 오디오 브랜치**를 통해 시청각 통합 이해도를 높인 Video-LLM이다. 이 모델은 다수의 비디오-언어 벤치마크에서 기존 오픈소스 모델을 압도하며 일부 유료 모델에 근접하는 성능을 보였으며, 특히 오디오-비주얼 통합 QA 작업에서 탁월한 성능을 입증했다. 향후 자율 주행, 로봇 조작, 긴 비디오 이해 등 더 복잡한 시공간 추론이 필요한 분야로 확장될 가능성이 매우 높다.
