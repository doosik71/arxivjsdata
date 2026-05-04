# Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities

Zhifei Xie, Changqiao Wu (2024)

## 🧩 Problem to Solve

본 논문은 OpenAI의 GPT-4o와 같이 시각(Vision), 청각(Speech), 텍스트(Text) 모든 모달리티를 통합적으로 이해하고 생성하며, 특히 실시간 Duplex(양방향) 상호작용이 가능한 오픈소스 모델을 구현하는 것을 목표로 한다.

기존의 오픈소스 다중 모달 모델들은 특정 기능(예: 시각적 이해 또는 음성 채팅)에만 집중하거나, ASR(Automatic Speech Recognition) $\rightarrow$ LLM $\rightarrow$ TTS(Text-to-Speech)와 같은 계단식(Cascading) 구조를 사용하여 응답 지연 시간이 길고 자연스러운 상호작용에 한계가 있었다. 특히, 사용자의 말을 실시간으로 인식하여 자신의 출력을 중단하는 Interruption(방해/중단) 메커니즘을 통합한 엔드투엔드(End-to-End) 모델을 구축하는 것은 다중 모달 데이터의 복잡성과 모델 아키텍처의 정교함으로 인해 매우 도전적인 과제이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 GPT-4o의 기능을 가장 유사하게 재현한 오픈소스 엔드투엔드 다중 모달 모델인 Mini-Omni2를 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1.  **통합 아키텍처 구현**: 시각, 청각, 텍스트 입력을 동시에 처리하고 텍스트와 오디오를 병렬로 출력하는 단일 모델 구조를 설계하였다.
2.  **3단계 학습 파이프라인**: 제한된 데이터셋으로도 효율적으로 모달리티를 확장하고 정렬하기 위해 '인코더 적응 $\rightarrow$ 모달리티 정렬 $\rightarrow$ 사후 학습'으로 이어지는 단계적 학습 방법을 제안하였다.
3.  **명령어 기반 중단 메커니즘(Command-based Interruption)**: "Stop Omni"와 같은 특정 세맨틱 큐(Semantic cue)를 인식하여 모델의 오디오 출력을 실시간으로 제어하는 `irq` 및 `n-irq` 토큰 기반의 중단 시스템을 도입하였다.
4.  **지연 병렬 디코딩(Delayed Parallel Decoding)**: 텍스트와 오디오 토큰을 동시에 생성함으로써 GPT-4o와 유사한 실시간 음성 응답 능력을 확보하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구 흐름을 분석하고 차별점을 제시한다.

-   **Large Vision Language Models (LVLMs)**: CLIP, Llava, Qwen-VL 등은 시각적 입력을 텍스트로 변환하여 이해하는 데 집중하였다. Mini-Omni2는 이러한 시각 이해 능력을 유지하면서 출력 모달리티를 오디오까지 확장하였다.
-   **Audio Language Modeling**: VALL-E나 AudioPaLM과 같은 연구들이 음성 신호를 토큰화하여 처리하기 시작했다. 최근 Moshi, Llama-Omni 등이 음성-음성 상호작용을 탐구했으나, Mini-Omni2는 여기에 시각 모달리티를 통합한 진정한 Omni-model을 지향한다.
-   **Multi-modal Interaction Model**: 기존의 A-T-T-A(Audio-Text-Text-Audio) 방식은 지연 시간이 길다. Mini-Omni2는 이전 연구인 Mini-Omni의 병렬 생성 방식을 계승하여 텍스트와 오디오를 동시에 생성함으로써 응답 속도를 획기적으로 줄였다.

## 🛠️ Methodology

### 전체 시스템 구조
Mini-Omni2는 Qwen2-0.5B를 기반 모델(Foundational LM)로 사용하며, 각 모달리티의 특징을 추출하기 위해 사전 학습된 인코더를 통합한 구조를 가진다.

-   **Vision Encoder**: CLIP의 ViT-B/32를 사용하여 이미지를 50개 길이의 특징 시퀀스로 변환하며, 단층 LlamaMLP 기반의 Vision Adapter를 통해 LLM의 임베딩 공간으로 투영한다.
-   **Audio Encoder**: Whisper-small 모델을 사용하여 오디오 특징을 추출한다. 저자들은 토큰 기반의 입력(Token-in) 방식보다 Whisper의 세맨틱 특징(Semantic features)을 사용하는 것이 언어 정렬 및 강건성 측면에서 유리함을 확인하였다.
-   **Language Model**: Qwen2-0.5B를 사용하며, 오디오 출력을 위해 SNAC 토크나이저의 7개 계층을 반영한 다층 LM-head(Multi-layer LM-heads) 구조를 채택하여 어휘 사전 크기를 181,120개로 확장하였다.

### 다중 모달 언어 모델링 및 손실 함수
모델은 텍스트 $Y$, 이산 음성 토큰 $D$, 그리고 연속적인 시각 특징 $V$를 통합하여 처리한다. 텍스트와 오디오가 동시에 생성되는 경우의 확률 모델링은 다음과 같이 정의된다.

$$p(Z) = \prod_{i=1}^t p(z_i | z_{<i}, V)$$

여기서 $Z$는 텍스트, 음성 토큰 또는 이들의 조합을 의미한다. 학습을 위한 음성-텍스트 병렬 출력의 Negative Log-Likelihood (NLL) 손실 함수는 다음과 같다.

$$\mathcal{L}(T, A, V|C) = \sum_{j=1}^m \sum_{i=1}^{n_j} \log P(T_{i,j}, A_{i,j} | T_{<i,j}, A_{<i,j}, V_j; X_j)$$

- $T, A$: 훈련 코퍼스 $C$ 내의 텍스트-오디오 출력 쌍
- $X_j, V_j$: $j$번째 샘플의 입력 조건 및 시각 특징
- $m$: 훈련 샘플 수, $n_j$: 샘플 $j$의 최대 토큰 수

### 훈련 절차 (Three-stage Training)
1.  **Stage 1: Multimodal Encoder Adaptation**: LLM과 인코더를 연결하는 선형 레이어(Adapter)의 가중치만 학습시킨다. 다중 모달 특징이 LLM의 텍스트 임베딩 특성과 유사해지도록 하여, 이후 단계에서 논리적 추론에 집중할 수 있게 한다.
2.  **Stage 2: Modality Alignment**: 어댑터를 고정하고 LLM의 가중치를 학습시킨다. 이미지나 오디오 입력에 대해 텍스트 응답을 생성하는 QA 작업을 수행하여 기초적인 논리 능력을 정렬한다.
3.  **Stage 3: Post-training**: 출력 모달리티를 오디오까지 확장한다. 모든 QA 작업에 대해 오디오 토큰 출력을 학습시키며, 동시에 아래의 중단 메커니즘을 학습시킨다.

### Duplex 상호작용 및 중단 메커니즘
단순한 VAD(Voice Activity Detection)가 아닌, 사용자의 의도를 파악하는 세맨틱 중단 방식을 구현하였다.
-   **데이터 구성**: 다양한 배경 소음(MUSAN 데이터셋 등)이 섞인 환경에서 "Stop Omni"라는 문구를 무작위 음색으로 합성하여 학습 데이터를 구축하였다.
-   **동작 원리**: 모델은 실시간으로 입력되는 오디오를 인코딩하며 `irq`(중단)와 `n-irq`(비중단) 상태 토큰을 생성한다. 추론 과정에서 `irq` 토큰이 생성되면 모델은 즉시 오디오 생성을 멈추고 다시 듣기 상태로 전환된다.

## 📊 Results

### 실험 설정
-   **데이터셋**: Open-Orca(텍스트 QA), LibriTTS/VCTK(음성 인식), ALLaVA-4V(시각 QA), VoiceAssistant-400K(음성 비서 스타일) 등을 사용하였다.
-   **학습 환경**: 8장의 A100 GPU를 사용하였으며, Cosine Scheduler와 Llama-MLP 어댑터를 적용하였다.

### 주요 결과
1.  **음성 인식 성능 (ASR)**: LibriSpeech-other 데이터셋에서 Mini-Omni2는 기반 모델인 Whisper-small보다 우수한 성능을 보였다. 이는 다중 모달 학습 과정이 오히려 음성 인식의 강건성을 향상시켰음을 시사한다. (다만, 시각 모달리티 추가로 인해 데이터 비율이 변하면서 Mini-Omni 1버전보다는 약간의 성능 하락이 있었다.)
2.  **정성적 분석 (Case Study)**:
    -   오디오 질문("What can you see?")에 대해 이미지 속 골든 리트리버 두 마리의 상태와 배경을 상세히 설명하는 능력을 보였다.
    -   표지판 이미지와 오디오 질문("What does the sign say?")에 대해 "STOP"이라는 문구와 그에 따른 행동 지침을 정확히 응답하였다.

## 🧠 Insights & Discussion

### 강점
-   **효율적인 확장성**: 대규모 데이터를 무작정 늘리는 대신, 3단계 정렬 과정을 통해 적은 데이터로도 효과적으로 모달리티를 확장하였다.
-   **낮은 지연 시간**: 병렬 디코딩을 통해 텍스트 생성을 기다리지 않고 오디오를 즉시 출력함으로써 실제 대화형 AI로서의 사용성을 높였다.

### 한계 및 비판적 해석
-   **모델 규모의 한계**: Qwen2-0.5B라는 매우 작은 모델을 사용했기 때문에, 복잡한 추론이나 고차원적인 다중 모달 이해에는 한계가 있을 수 있다. 저자들 또한 데이터와 모델 규모의 확장이 필요함을 명시하였다.
-   **단순한 중단 메커니즘**: 현재의 중단 방식은 "Stop Omni"라는 특정 명령어에 의존하는 방식이다. 실제 GPT-4o처럼 맥락에 따라 자연스럽게 말을 끊거나 끼어드는 완전한 Semantic Interruption을 구현하기 위해서는 더 복잡한 데이터셋과 학습 전략이 필요할 것으로 보인다.
-   **출력 다양성 부족**: 음성의 감정, 억양, 노래하는 능력 등 오디오 출력의 스타일 제어 능력이 부족하다는 점이 한계로 지적되었다.

## 📌 TL;DR

Mini-Omni2는 시각, 청각, 텍스트를 모두 처리하는 **엔드투엔드 오픈소스 다중 모달 LLM**이다. 3단계 학습 전략과 병렬 디코딩을 통해 GPT-4o와 유사한 실시간 음성 응답 능력을 구현했으며, 특히 특정 명령어를 통해 출력을 제어하는 **Duplex 상호작용 메커니즘**을 도입한 것이 특징이다. 이 연구는 적은 자원으로도 효율적으로 다중 모달 모델을 구축하는 방법론을 제시하여, 향후 실시간 AI 비서 연구에 중요한 기준점이 될 가능성이 크다.