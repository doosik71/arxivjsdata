# Acoustic Prompt Tuning: Empowering Large Language Models with Audition Capabilities

Jinhua Liang, Xubo Liu, Wenwu Wang, Mark D. Plumbley, Huy Phan, Emmanouil Benetos (2023/2025)

## 🧩 Problem to Solve

현대의 대규모 언어 모델(LLM)과 시각-언어 모델(VLM)은 텍스트와 이미지 이해 분야에서 탁월한 성능을 보이고 있으나, 오디오 도메인으로의 확장성은 여전히 제한적이다. 기존의 오디오-언어 모델들은 주로 `[audio, question, answer]`라는 고정된 입력 형식을 사용하며, 이는 모델이 풍부한 문맥 정보나 구조화된 정보를 활용하는 것을 방해한다. 특히, 여러 개의 오디오 클립을 동시에 분석하거나, 소수의 예시만을 이용해 분류하는 few-shot audio classification과 같은 유연한 작업 수행이 어렵다는 문제가 있다.

본 논문의 목표는 오디오 클립을 'Acoustic Prompt(음향 프롬프트)'로 인코딩하여 LLM/VLM에 주입함으로써, 기존 모델의 도메인 특화 능력을 유지하면서도 오디오 이해 및 추론 능력을 부여하는 유연한 어댑터 구조인 Acoustic Prompt Tuning (APT)을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 오디오 데이터를 LLM이 이해할 수 있는 **soft prompts(가변적인 임베딩 벡터)** 형태로 변환하여, 텍스트 토큰과 오디오 토큰이 교차로 배치되는 **interleaved audio-text embeddings** 구조를 구축하는 것이다.

주요 기여 사항은 다음과 같다:

- **Acoustic Adapter 제안**: LLM/VLM을 오디오 도메인으로 확장하기 위한 soft prompting 기반의 어댑터를 도입하였다.
- **Interleaved 입력 포맷**: 오디오와 텍스트를 순서 제약 없이 교차 배치함으로써, 단일 피드포워드 단계에서 여러 개의 오디오 클립을 입력받고 이들 간의 상관관계를 분석할 수 있게 하였다.
- **다단계 커리큘럼 학습(Curriculum Learning)**: 오디오 데이터의 희소성 문제를 해결하기 위해 '오디오-텍스트 정렬 $\rightarrow$ 단일 클립 학습 $\rightarrow$ 다중 클립 학습'으로 이어지는 단계적 학습 전략을 제안하였다.
- **Natural Language Audio Reasoning (NLAR) 태스크 제안**: 두 개의 오디오 클립을 비교하고 요약하는 새로운 추론 태스크를 정의하여 모델의 고도화된 분석 능력을 평가하였다.
- **VLM 확장성 검증**: frozen 상태의 BLIP-2 모델에 APT를 결합하여 추가적인 파인튜닝 없이 오디오-시각 질의응답(AVQA) 성능을 향상시켰다.

## 📎 Related Works

기존의 멀티모달 언어 모델들은 주로 시각 도메인에 집중되어 왔으며, UniVAL나 ImageBind-LLM과 같은 모델들이 오디오를 포함한 다양한 모달리티를 통합하려 시도하였다. 그러나 이러한 방식은 모델 전체를 처음부터 학습시키기 위해 막대한 양의 멀티모달 데이터가 필요하다는 한계가 있다.

오디오 전용 언어 모델인 LTU, Pengi, Qwen-Audio 등은 오디오-텍스트 정렬을 통해 성능을 높였으나, 대부분 `[audio, question, answer]`라는 고정된 튜플 형식을 사용한다. 이로 인해 문맥 학습(In-context learning) 능력을 충분히 활용하지 못하며, 단일 오디오 클립 이상의 입력을 처리하는 능력이 부족하다. 반면, APT는 특정 아키텍처에 종속되지 않는 도메인 특화 어댑터 방식을 채택하여 기존의 어떤 LLM/VLM에도 적용 가능하며, 입력 포맷의 제약을 없애 다중 오디오 분석이 가능하다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

APT-LLM은 크게 **Audio Encoder**, **Audio Aligner**, 그리고 **Large Language Model (LLM)**의 세 가지 구성 요소로 이루어져 있다. 전체 파이프라인은 입력 오디오 스펙트로그램을 특징 맵으로 추출하고, 이를 텍스트 지시어와 함께 정렬하여 LLM의 입력 토큰으로 변환하는 과정을 거친다.

### 주요 구성 요소 및 역할

1. **Audio Encoder**: Audio-MAE를 사용하며, 10초 단위로 세그먼트화된 오디오 입력을 처리한다. 분류를 위한 마지막 레이어 대신, 세밀한 오디오 패턴을 보존하기 위해 **penultimate block(끝에서 두 번째 블록)**의 출력 특징 맵(shape: $(1024, 512)$)을 추출한다.
2. **Audio Aligner**: Audio-MAE의 가변적인 특징 맵을 고정된 수의 음향 임베딩(32개)으로 리샘플링하는 역할을 한다. Q-former 아키텍처(4개 트랜스포머 레이어)를 기반으로 하며, BERT 토크나이저를 통해 입력된 텍스트 프롬프트와 학습 가능한 쿼리 토큰(learnable query tokens)을 사용하여 오디오 특징에서 필요한 정보를 추출한다.
3. **Large Language Model**: Vicuna 7B v1.1과 같은 LLM을 사용하며, 학습 과정에서 모든 파라미터는 **frozen(동결)** 상태로 유지된다. 오디오 모달리티를 식별하기 위해 학습 가능한 특수 토큰인 `[AUDIO]`를 각 오디오 클립 앞에 추가한다.

### 학습 목표 및 방정식

기존 모델들이 오디오와 텍스트를 단순히 연결($C$)하여 사용했다면, APT-LLM은 교차 배치 함수 $I$를 사용하여 입력 시퀀스를 구성한다.

기존의 방식:
$$X_{audio;text} = C(M_\theta(A_\phi(a), t), W_\psi(t))$$

APT-LLM의 인터리브 방식:
$$X_{audio;text} = [ (S(a_1, t_1)), T_\psi(t_1), \dots, (S(a_N, t_N)), W_\psi(t_N) ]$$
여기서 $S(\cdot, \cdot) = M_\theta(A_\phi(\cdot), \cdot)$는 오디오 정렬기를 통해 생성된 음향 임베딩이며, $T_\psi$와 $W_\psi$는 텍스트 임베딩을 나타낸다. 최종적으로 LLM은 이전 토큰들을 조건으로 다음 토큰의 확률 분포를 최대화하는 방향으로 학습된다.

### 커리큘럼 학습 절차

데이터 부족 문제를 해결하기 위해 3단계 학습 전략을 수행한다:

- **Stage 0 (Audio-Text Alignment)**: LLM 결합 전, Audio-MAE는 고정하고 Aligner만 학습시킨다. Audio-Text Matching(ATM), Audio-Grounded Text Generation(AGTG), Audio-Text Contrasting(ATC)의 세 가지 목적 함수를 사용하여 모달리티 간의 기초적인 정렬을 수행한다.
- **Stage 1 (Single Audio Clip Learning)**: 단일 오디오 기반의 Audio Tagging, Captioning, QA 및 새롭게 정의한 Query-based SED, Temporal Event Retrieval, Sound Event Counting 작업을 통해 세밀한 오디오 특징을 학습한다.
- **Stage 2 (Multiple Audio Clips Learning)**: Few-shot audio classification과 NLAR 작업을 추가하여, 모델이 여러 오디오 클립 간의 상관관계를 분석하고 추론할 수 있도록 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: AudioSet, ESC-50 (분류), Clotho, AudioCaps (캡셔닝), NLAR, AVQA.
- **지표**: Accuracy, mAP, SPICE, SPIDEr.
- **비교 대상**: AudioCLIP, CLAP, Qwen-Audio, LTU, Pengi 등.

### 주요 결과

1. **오디오 태깅 및 캡셔닝**: APT-LLM은 AudioSet에서 경쟁력 있는 mAP를 기록하였으며, 특히 자동 오디오 캡셔닝 작업(AudioCaps, Clotho)에서 가중 평균 SPICE 점수 $0.172$를 달성하며 전문가 모델(task-specific systems)에 필적하는 성능을 보였다.
2. **Few-shot 오디오 분류**: 5-way 5-shot 설정에서 기존의 전문 분류 모델들을 능가하였으며, CLAP 어댑터와 경쟁 가능한 수준의 성능을 보였다. 다만 12-way 설정에서는 성능 저하가 관찰되었는데, 이는 입력 시퀀스가 길어짐에 따라 LLM의 어텐션 메커니즘에 부하가 걸리기 때문으로 분석된다.
3. **Natural Language Audio Reasoning (NLAR)**: AAC 모델과 ChatGPT-4o를 결합한 베이스라인(27.9%) 대비, APT-LLM은 **63.8%**라는 압도적인 정확도를 기록하며 다중 오디오 추론 능력을 입증하였다.
4. **Audio-Visual QA (AVQA)**: APT-BLIP-2는 비디오 전용(42.9%) 또는 오디오 전용(27.7%) 모델보다 높은 **59.7%**의 정확도를 기록하여, 오디오 모달리티의 추가가 시각적 이해를 보완함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 오디오 데이터를 단순한 입력값이 아닌, LLM의 문맥을 제어하는 '프롬프트'로 취급함으로써 LLM의 In-context learning 능력을 오디오 도메인으로 성공적으로 전이시켰다. 특히 인터리브 구조를 통해 단일 모델이 여러 오디오 클립을 동시에 처리하게 함으로써, 기존 오디오-언어 모델들이 해결하지 못했던 '비교'와 '요약'이라는 고차원 추론 능력을 구현했다는 점이 매우 고무적이다.

### 한계 및 비판적 해석

- **모델 종속성**: APT Aligner는 LLM의 워드 임베딩 공간에 맞춰 학습되므로, LLM 아키텍처가 동일하더라도 다른 가중치를 가진 모델을 사용할 경우 Aligner를 다시 학습시켜야 하는 제약이 있다.
- **지시어 튜닝의 부재**: Instruction-based 데이터셋으로 학습되지 않았기 때문에, 학습 데이터 범위 밖의 질문(out-of-domain)에 대해서는 대응 능력이 제한적일 수 있다.
- **도메인 확장성**: 현재는 일반적인 환경음(General Audio)에 집중되어 있으며, 언어적 특성이 강한 Speech나 예술적 구조를 가진 Music 도메인으로의 확장 검증은 이루어지지 않았다.

## 📌 TL;DR

본 연구는 LLM/VLM을 오디오 도메인으로 확장하기 위한 **Acoustic Prompt Tuning (APT)** 어댑터를 제안하였다. 오디오를 고정된 길이의 soft prompts로 변환하고 이를 텍스트와 교차 배치하는 구조를 통해, 모델이 단일 클립 분석을 넘어 **다중 오디오 비교 및 추론(NLAR)**과 **few-shot 분류**를 수행할 수 있게 하였다. 특히 3단계 커리큘럼 학습을 통해 데이터 희소성을 극복하였으며, frozen VLM에 결합하여 오디오-시각 통합 이해 능력을 향상시켰다. 이 연구는 향후 LLM 기반의 범용 오디오 분석 및 멀티모달 시스템 구축에 있어 중요한 방법론적 기초를 제공한다.
