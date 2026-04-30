# AudioPaLM: A Large Language Model That Can Speak and Listen

Paul K. Rubenstein et al. (2023)

## 🧩 Problem to Solve

기존의 음성-음성 번역(Speech-to-Speech Translation, S2ST) 시스템은 주로 자동 음성 인식(ASR), 텍스트 기계 번역(MT), 그리고 텍스트-음성 합성(TTS)의 세 단계가 결합된 캐스케이드(Cascade) 방식으로 구현되었다. 이러한 방식은 각 단계에서 발생하는 오류가 누적되는 compound errors 문제가 있으며, 화자의 정체성, 억양과 같은 부가 언어적 정보(paralinguistic information)가 손실된다는 치명적인 단점이 있다.

직접적인(Direct) S2ST 시스템이 제안되었으나, 이들은 주로 음성 도메인에 집중되어 있어 텍스트 기반 거대 언어 모델(LLM)이 보유한 방대한 언어적 지식과 상식적 추론 능력을 충분히 활용하지 못했다. 본 논문의 목표는 텍스트와 음성을 동시에 처리하고 생성할 수 있는 단일 멀티모달 아키텍처를 구축하여, 언어 모델의 지식과 음성 모델의 생성 능력을 결합한 통합 시스템을 구현하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 텍스트 기반 LLM인 PaLM-2와 음성 생성 모델인 AudioLM을 단일 Decoder-only Transformer 구조로 통합하는 것이다.

1.  **통합 어휘집(Joint Vocabulary) 구축**: 텍스트 토큰과 이산화된 음성 토큰(discrete audio tokens)을 하나의 어휘집으로 합쳐, 모델이 텍스트와 음성을 동일한 시퀀스 데이터로 처리하게 하였다.
2.  **사전 학습 지식의 전이**: 텍스트 전용으로 사전 학습된 PaLM-2의 가중치로 모델을 초기화함으로써, 음성 작업 수행 시 LLM이 가진 강력한 언어적 능력을 그대로 계승하도록 설계하였다.
3.  **멀티태스크 학습**: ASR, AST(Automatic Speech Translation), S2ST, TTS, MT 등 다양한 작업을 단일 모델에서 학습시켜, 모달리티 간의 자유로운 변환이 가능하게 하였다.
4.  **제로샷(Zero-shot) 능력 확보**: 학습 과정에서 보지 못한 언어 쌍에 대해서도 텍스트 모델의 번역 능력을 바탕으로 음성-텍스트 번역(AST)을 수행하는 능력을 입증하였다.

## 📎 Related Works

### 멀티모달 융합 (Multimodal Fusion)
기존의 Whisper나 Flamingo와 같은 모델들은 주로 인코더-디코더(Encoder-Decoder) 구조를 사용하며, 비텍스트 입력을 텍스트 디코더에 전달하는 어댑터(Adapter) 층을 활용한다. 그러나 이러한 모델들은 출력 결과가 텍스트로 제한된다는 한계가 있다. 반면, AudioPaLM은 Decoder-only 구조를 채택하여 텍스트와 음성 토큰 모두를 출력할 수 있다.

### 음성 생성 언어 모델 (Generating Audio with LMs)
AudioLM은 계층적 접근 방식을 통해 의미적 토큰(semantic tokens)과 음향적 토큰(acoustic tokens)을 순차적으로 생성하여 고품질 음성을 구현했다. SPEAR-TTS 등은 이를 확장해 TTS를 구현했지만, 텍스트와 음성 어휘집이 분리되어 있어 상호 교환 가능한 멀티모달 학습이 어려웠다. AudioPaLM은 이 두 어휘집을 통합함으로써 이 간극을 메웠다.

### 음성-음성 번역 (S2ST)
전통적인 캐스케이드 방식의 효율성 저하와 직접 방식의 언어적 지식 부족 문제를 해결하기 위해, 본 논문은 이산화된 음성 표현 공간에서 LLM의 추론 능력을 활용하는 방식을 제안한다.

## 🛠️ Methodology

### 전체 파이프라인
AudioPaLM은 입력된 텍스트 또는 음성을 토큰 시퀀스로 변환하여 처리하고, 다시 이를 토큰 시퀀스로 출력하는 구조이다. 음성 출력의 경우, 생성된 이산 토큰을 다시 raw audio로 변환하는 후처리 단계가 추가된다.

### 주요 구성 요소 및 절차

**1. 음성 토큰화 (Audio Tokenization)**
Raw waveform을 이산 토큰으로 변환하기 위해 다음과 같은 인코더를 실험하였다.
- **w2v-BERT**: 다국어 데이터로 학습된 모델을 사용하여 $25\text{Hz}$ 비율로 토큰을 추출하며, 어휘집 크기는 $1024$이다.
- **USM (Universal Speech Model)**: 더 강력한 성능의 USM 인코더를 사용하며, 특히 보조 ASR 손실 함수로 학습된 **USM-v2**가 가장 우수한 성능을 보였다.

**2. 모델 아키텍처 및 수정 (Model Modification)**
기존 PaLM-2의 Decoder-only Transformer 구조를 유지하되, 임베딩 행렬을 확장하였다.
- 기존 텍스트 어휘집 크기를 $t$, 임베딩 차원을 $m$이라고 할 때, 기존 임베딩 행렬 $E$의 크기는 $t \times m$이다.
- 여기에 음성 토큰의 개수 $a$만큼 행을 추가하여 $(t+a) \times m$ 크기의 확장된 행렬을 구축한다.
- 텍스트 임베딩은 PaLM-2의 가중치를 그대로 사용하고, 새로 추가된 음성 임베딩은 무작위 초기화 후 학습시킨다.

**3. 음성 복원 (Decoding Audio Tokens)**
모델이 출력한 음성 토큰을 실제 소리로 바꾸기 위해 두 가지 디코딩 방식을 사용한다.
- **AudioLM 방식**: Autoregressive하게 SoundStream 토큰을 생성한 후 컨볼루션 디코더를 통해 복원한다.
- **SoundStorm 방식**: Non-autoregressive 방식으로 병렬 생성하여 속도를 획기적으로 높이면서도 일관된 음질을 유지한다.

**4. 학습 절차 및 목표**
- **태스크 지정**: 입력 앞에 `[ASR French]`, `[S2ST English French]`와 같은 텍스트 태그를 붙여 수행할 작업을 명시한다.
- **결합 작업 (Combined Tasks)**: 복잡한 작업을 단계별로 수행하도록 유도한다. 예를 들어 S2ST 작업 시 `[ASR AST S2ST English French]` 태그를 사용하여 $\text{음성}_{\text{en}} \rightarrow \text{텍스트}_{\text{en}} \rightarrow \text{텍스트}_{\text{fr}} \rightarrow \text{음성}_{\text{fr}}$ 순으로 한 번에 생성하게 함으로써 성능을 향상시켰다(Chain-of-thought prompting과 유사한 원리).

## 📊 Results

### 실험 설정
- **데이터셋**: CoVoST2, VoxPopuli, CommonVoice 11 등 다국어 음성-텍스트 데이터셋을 사용하였다.
- **평가 지표**: AST 및 S2ST는 $\text{BLEU}$ 점수(ASR을 통한 텍스트 변환 후 측정)를 사용하고, ASR은 $\text{WER}$ (Word Error Rate)을 사용하였다.

### 주요 결과
1.  **정량적 성능**: AudioPaLM-2 8B 모델은 AST 및 S2ST 벤치마크에서 기존 시스템(Whisper, Translatron 2 등)을 유의미하게 능가하였다. 특히 AST에서는 $\text{BLEU}$ 점수에서 압도적인 성능 향상을 보였다.
2.  **제로샷 능력**: 학습 데이터에 포함되지 않은 언어 쌍에 대해서도 AST를 수행할 수 있었다. 이는 PaLM-2가 가진 텍스트 번역 능력이 음성 도메인으로 전이되었음을 의미한다.
3.  **음성 품질 및 화자 보존**: 주관적 평가(MOS) 및 객관적 지표(Cosine Similarity) 분석 결과, AudioPaLM은 화자의 정체성을 유지하며 고품질의 음성을 생성하였다. 이는 기존 Translatron 2보다 우수한 결과이다.

### Ablation Study 결과
- **ASR 데이터의 영향**: AST 학습 시 ASR 작업을 함께 학습시키면 AST 성능이 향상됨을 확인하였다.
- **초기화의 중요성**: 무작위 초기화 모델보다 PaLM-2 사전 학습 모델을 파인튜닝한 모델의 성능이 월등히 높았다.
- **토큰나이저 비교**: $\text{w2v-BERT} < \text{USM-v1} < \text{USM-v2}$ 순으로 성능이 좋았으며, 토큰의 질이 전체 모델 성능에 결정적인 영향을 미침을 확인하였다.
- **모델 크기**: $128\text{M} \rightarrow 1\text{B} \rightarrow 8\text{B}$로 모델 크기가 커질수록 $\text{WER}$ 감소와 $\text{BLEU}$ 점수 상승이 뚜렷하게 나타났다.

## 🧠 Insights & Discussion

**강점**
AudioPaLM은 단순한 모달리티 결합을 넘어, 거대 언어 모델의 고차원적 언어 지식을 음성 처리 작업에 성공적으로 주입하였다. 특히 '결합 작업(Combined Tasks)' 설계를 통해 복잡한 S2ST 과정을 단순화하고 성능을 끌어올린 점이 돋보인다. 또한, 화자의 음색을 유지하는 Voice Transfer 능력을 통해 실제 적용 가능성을 높였다.

**한계 및 비판적 해석**
- **토큰나이저 의존성**: 모델의 성능이 USM-v2와 같은 이산 토큰나이저의 품질에 매우 크게 의존한다. 이는 음성-텍스트 통합 모델의 병목 지점이 모델 아키텍처보다는 음성을 어떻게 이산화하느냐에 있을 수 있음을 시사한다.
- **전체 파인튜닝 필요성**: 일부 가중치를 고정(Freeze)하는 방식으로는 성능이 나지 않아 전체 파라미터를 학습시켜야 했다. 이는 텍스트 기반 가중치와 새로 추가된 음성 임베딩 간의 정렬(Alignment)을 위해 상당한 학습량이 필요함을 의미하며, 계산 비용의 증가를 초래한다.
- **데이터 불균형**: 저자원 언어의 경우 여전히 성능 향상 폭이 적으며, 이는 LLM의 지식이 있더라도 해당 언어의 음성-텍스트 정렬 데이터가 부족하면 한계가 있음을 보여준다.

## 📌 TL;DR

AudioPaLM은 **PaLM-2(텍스트 LLM)**와 **AudioLM(음성 생성 모델)**을 통합하여 텍스트와 음성을 동시에 이해하고 생성할 수 있는 단일 Decoder-only 모델이다. **이산화된 음성 토큰**을 사용하여 텍스트와 음성을 동일한 시퀀스로 처리하며, LLM의 사전 학습 지식을 계승하여 **SOTA 수준의 음성 번역(AST, S2ST)** 성능과 **제로샷 번역 능력**을 달성하였다. 이 연구는 LLM의 추론 능력을 음성 도메인으로 확장하는 효과적인 방법을 제시하였으며, 향후 범용 음성-텍스트 AI 에이전트 구현에 핵심적인 역할을 할 것으로 기대된다.