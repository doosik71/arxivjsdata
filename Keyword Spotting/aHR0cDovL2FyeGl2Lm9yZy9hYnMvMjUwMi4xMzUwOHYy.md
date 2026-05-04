# VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation

Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, Donglin Wang (2025)

## 🧩 Problem to Solve

본 논문은 로봇 조작(Robot Manipulation)을 위한 Vision-Language-Action(VLA) 모델이 주로 텍스트 기반의 지시어에 의존하고 있다는 점을 지적한다. 인간과 로봇의 상호작용에서 음성(Speech)은 가장 자연스러운 모달리티임에도 불구하고, 기존의 접근 방식은 외부 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템을 사용하여 음성을 텍스트로 변환한 뒤 VLA 모델에 입력하는 캐스케이딩(Cascading) 파이프라인을 사용했다.

이러한 방식은 두 가지 핵심적인 문제를 야기한다. 첫째, 시스템의 복잡도가 증가하여 계산 자원과 메모리 소모가 늘어난다. 둘째, 텍스트 변환 과정에서 화자의 정체성(Identity), 감정, 억양과 같은 비언어적 정보(Non-semantic information)가 손실된다. 특히 "내 컵을 집어줘"와 같은 개인화된 작업(Customized tasks)의 경우, 누가 말했느냐에 따라 대상이 달라지므로 음성 자체에 포함된 화자 정보(Voiceprint)가 필수적이다. 따라서 본 연구의 목표는 외부 ASR 없이 음성 지시를 직접 처리하여 개인화된 로봇 조작을 수행할 수 있는 엔드 투 엔드(End-to-End) VLA 모델인 VLAS를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 음성 모달리티를 VLA 모델에 직접 통합하여 자연스러운 상호작용과 개인화된 작업 수행을 가능하게 한 점이다. 이를 위한 주요 설계 아이디어는 다음과 같다.

1. **엔드 투 엔드 음성-행동 모델**: 외부 ASR 시스템 없이 raw speech를 직접 입력받아 로봇 행동을 생성하는 VLAS 아키텍처를 제안하였다.
2. **Voice RAG (Retrieval-Augmented Generation)**: 화자의 음성 지문(Voiceprint)을 통해 외부 데이터베이스에서 해당 사용자의 개인적 지식(예: 소유물, 선호도)을 검색하여 모델에 제공하는 패러다임을 설계하였다.
3. **새로운 데이터셋 및 기반 모델**: 음성 지시 튜닝을 위해 SQA(Speech Question Answering)와 CSI(CALVIN with Speech Instructions) 데이터셋을 구축하였으며, 범용적으로 사용 가능한 VLAS-Base 모델을 공개하였다.

## 📎 Related Works

기존의 관련 연구는 크게 두 가지 방향으로 나뉜다. 하나는 LLaVA와 같은 Vision-Language Model(VLM)을 통해 시각-언어 이해 능력을 높이는 것이고, 다른 하나는 RT-2나 OpenVLA처럼 VLM을 로봇 조작 데이터로 미세 조정하여 행동을 직접 생성하는 VLA 모델이다.

기존 VLA 모델들의 한계는 음성 모달리티를 거의 고려하지 않았거나, 고려하더라도 ASR을 통한 텍스트 변환 방식에 의존했다는 점이다. 이는 전술한 바와 같이 시스템 복잡도를 높이고 화자 식별과 같은 중요한 보조 정보를 소실시킨다. 본 연구는 음성 인코더를 LLM의 임베딩 공간에 직접 정렬(Alignment)함으로써 이러한 한계를 극복하고, 단순한 텍스트 명령 수행을 넘어 화자 맞춤형 동작을 수행할 수 있도록 차별화하였다.

## 🛠️ Methodology

### 전체 시스템 구조

VLAS는 시각적 관측치 $O$와 음성 지시 $s$를 입력받아 로봇 행동 $a$를 직접 생성한다. 기본 구조는 LLaVA를 기반으로 하며, 시각 인코더(CLIP)와 음성 인코더(Whisper)를 통해 추출된 특징을 MLP(Multi-Layer Perceptron) 프로젝트를 통해 LLM(Vicuna/LLaMA)의 언어 공간으로 매핑한다.

최종 입력 임베딩 $\text{Emb}(s, O)$는 다음과 같이 구성된다.
$$\text{Emb}(s, O) = \text{concat}(\text{MLP}_s(\text{Emb}_s(s)), \text{Tok}_l(\text{RAG}(s)), \text{MLP}_v(\text{Emb}_v(O)))$$
여기서 $\text{Emb}_s$와 $\text{Emb}_v$는 각각 음성과 시각 인코더이며, $\text{Tok}_l$은 Voice RAG를 통해 검색된 텍스트 지식을 토큰화하는 텍스트 토크나이저이다. 모델은 이 임베딩을 바탕으로 다음의 확률 분포에 따라 행동 토큰을 자기회귀(Autoregressive) 방식으로 생성한다.
$$p(a|\text{Emb}(s, O)) = \prod_{i=1}^{N} p(a_i|\text{Emb}(s, O), a_{<i})$$

### 주요 구성 요소

- **Speech Encoder**: Whisper 인코더를 사용하여 음성 신호를 80-bin mel-spectrogram으로 변환 후 1500개의 hidden representation을 생성한다. 계산 효율성을 위해 시간 축으로 5배 다운샘플링(Reshape)을 수행한 뒤 MLP를 통해 언어 공간으로 투영한다.
- **Voice RAG**: 화자 식별 모듈이 음성에서 voiceprint를 추출하면, 이를 키(Key)로 사용하여 외부 DB에서 해당 사용자의 개인 지식을 검색한다. 이 정보는 텍스트 토큰 형태로 LLM에 입력되어 "내 컵"이 어떤 색상인지 등의 구체적인 맥락을 제공한다.
- **Action Tokenization**: 연속적인 행동 값(7차원: 위치 $x, y, z$, 회전 $\phi, \theta, \psi$, 그리퍼 상태 $g$)을 256개의 균일한 빈(Bin)으로 이산화(Discretization)하여 LLM의 기존 어휘집 중 사용 빈도가 낮은 토큰들로 매핑한다.

### 학습 절차 (Three-Stage Tuning)

1. **Stage I (Speech Alignment)**: LibriSpeech 데이터셋을 사용하여 음성 인코더와 LLM 사이의 MLP 레이어만 미세 조정한다. 이는 음성을 텍스트 공간에 거칠게 정렬하는 단계이다.
2. **Stage II (Speech QA Fine-tuning)**: SQA 및 기존 VQA 데이터셋을 사용하여 시각-음성-텍스트 간의 다중 모달리티 이해 능력을 학습시킨다. 이 결과물이 VLAS-Base 모델이다.
3. **Stage III (Robot Manipulation Fine-tuning)**: CSI 데이터셋을 사용하여 로봇의 조작 궤적(Trajectory)을 학습시킨다. 행동 복제(Behavior Cloning) 방식을 통해 음성 및 텍스트 지시에 따른 실제 로봇 행동을 생성하도록 학습한다.

## 📊 Results

### 정량적 평가 (CALVIN Benchmark)

CALVIN 벤치마크 실험 결과, VLAS는 텍스트 지시와 음성 지시 모두에서 기존의 MCIL, RT-1 및 텍스트 기반 VLA 모델과 대등하거나 이를 상회하는 성능을 보였다. 특히 ASR 시스템을 결합한 VLA나 Roboflamingo-ASR 조합보다 VLAS(음성 직접 입력)의 성능이 더 높게 나타났는데, 이는 로봇 조작에 특화된 음성 데이터로 학습되어 ASR에서 발생하는 오류 전파(Error propagation)를 방지했기 때문이다.

### 개인화 작업 평가 (Customization Benchmark)

본 논문에서 새롭게 제안한 소유권(Ownership), 선호도(Preference), 복합 작업(Compound Tasks) 벤치마크에서 VLAS의 진가가 드러났다.

- **VLA(텍스트 기반)**: 배경 지식이 없어 평균 성공률이 20% 미만으로 매우 저조했다.
- **VLAS(Voice RAG 포함)**: 화자의 정체성을 파악하고 개인 지식을 검색함으로써 평균 86% 이상의 성공률을 기록했다.
- **Ablation Study**: Voice RAG를 제거했을 때 성능이 급격히 하락(16% 수준)하여, 개인화 작업에서 Voice RAG의 필수성이 입증되었다.

### 실물 로봇 실험 및 기반 모델 분석

실제 UR5 로봇 팔에 적용한 결과, 화자의 음성에 따라 서로 다른 컵을 집어 올리는 개인화 동작에 성공하였다. 또한 VLAS-Base 모델은 일반적인 VLM 벤치마크(VQAv2, GQA 등)에서 LLaVA v1.5와 거의 동일한 성능을 보였으며, LibriSpeech WER(Word Error Rate)에서도 Whisper와 대등한 수준의 음성 인식 능력을 보여주었다.

## 🧠 Insights & Discussion

본 연구는 음성이라는 모달리티가 단순히 텍스트의 대체제가 아니라, 화자의 정체성과 같은 중요한 보조 정보를 담고 있는 데이터 소스임을 입증하였다. 특히 Voice RAG를 통해 LLM의 외부 지식 확장 능력을 로봇 제어 영역으로 가져와, "내 물건"과 같은 모호한 지시어를 구체적인 행동으로 연결한 점이 매우 인상적이다.

**강점**:

- ASR-VLA로 이어지는 복잡한 파이프라인을 제거하여 지연 시간을 줄이고 효율성을 높였다.
- 음성 지문을 활용한 개인화 메커니즘을 통해 실제 가정 환경에서 필요한 맞춤형 서비스 가능성을 제시했다.

**한계 및 비판적 해석**:

- 실패 사례 분석에서 사용자의 '선호도(Preference)' 작업이나 '복합 작업의 두 번째 단계'에서 오류가 빈번하게 발생했다. 이는 모델이 지시어는 이해했으나 실제 정밀한 제어(Execution) 단계에서 한계가 있음을 시사한다.
- 음성 데이터 생성 시 TTS(Text-to-Speech) 모델을 사용하여 SQA와 CSI 데이터셋을 구축했는데, 이는 실제 인간 음성의 다양성을 완전히 반영하지 못했을 가능성이 있다. 비록 실제 음성으로도 테스트를 진행했으나, 학습 데이터 자체의 합성 특성이 모델의 강건성에 영향을 줄 수 있다.

## 📌 TL;DR

본 논문은 외부 ASR 없이 raw speech를 직접 처리하여 로봇을 조작하는 엔드 투 엔드 VLA 모델인 **VLAS**를 제안한다. 특히 **Voice RAG**를 통해 화자의 음성 지문으로부터 개인 맞춤형 지식을 검색하여 적용함으로써, 기존 텍스트 기반 모델이 해결하지 못한 "개인화된 로봇 조작" 문제를 성공적으로 해결하였다. 이 연구는 향후 로봇이 사용자의 정체성과 환경적 맥락을 더 깊이 이해하고 상호작용하는 맞춤형 서비스 로봇으로 발전하는 데 중요한 기반이 될 것으로 보인다.
