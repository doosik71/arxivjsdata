# SPEECH-MAMBA: LONG-CONTEXT SPEECH RECOGNITION WITH SELECTIVE STATE SPACES MODELS

Xiaoxue Gao and Nancy F. Chen (2024)

## 🧩 Problem to Solve

기존의 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템은 주로 Transformer 기반 모델을 사용하여 음향 모델링을 수행한다. Transformer는 어텐션(Attention) 메커니즘을 통해 입력 시퀀스의 정보를 밀집하게 캡처하는 능력이 뛰어나지만, 시퀀스 길이에 따라 연산 복잡도가 제곱으로 증가하는 Quadratic Complexity 문제를 가지고 있다. 이로 인해 매우 긴 음성 시퀀스를 처리할 때 모델의 효율성과 인식 정확도가 저하되는 한계가 발생한다.

본 논문의 목표는 이러한 Transformer의 한계를 극복하기 위해 Selective State Space Model인 Mamba를 Transformer 아키텍처에 통합한 **Speech-Mamba**를 제안하는 것이다. 이를 통해 긴 문맥(Long-context)의 음성 데이터를 효과적으로 처리하고, 연산 복잡도를 선형 수준($\text{near-linear}$)으로 낮추면서도 전사 정확도를 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Transformer의 국소적/저수준 표현 학습 능력과 Mamba의 전역적/고수준 롱-레인지 의존성(Long-range dependencies) 모델링 능력을 결합하는 하이브리드 설계를 도입하는 것이다.

구체적으로, Transformer는 음성과 텍스트의 하위 수준(lower-level) 표현과 시간적 특성을 모델링하는 데 집중하고, Mamba는 선택적 상태 공간 모델(Selective State Space Model)을 통해 긴 시퀀스에서 중요한 정보를 선택적으로 유지하며 전역적인 문맥 정보를 캡처하도록 설계되었다. 이러한 결합을 통해 모델은 긴 음성 데이터에서도 효율적인 연산 속도를 유지하면서 깊은 수준의 음성 및 텍스트 지식을 학습할 수 있다.

## 📎 Related Works

전통적인 ASR은 음향, 어휘, 언어 모델을 별도로 구현하는 하이브리드 구조를 사용했으나, 최근에는 이를 통합하여 처리하는 End-to-End(E2E) 방식인 CTC(Connectionist Temporal Classification), S2S(Sequence-to-Sequence), 그리고 이 둘을 결합한 Joint CTC-S2S 모델이 주류를 이루고 있다. 특히 Transformer 기반 모델은 RNN 기반 모델보다 우수한 성능을 보였지만, 앞서 언급한 복잡도 문제로 인해 긴 시퀀스 처리에 취약하다.

최근 NLP와 컴퓨터 비전 분야에서는 State Space Models(SSMs), 특히 S4나 Mamba와 같은 모델들이 등장하여 긴 시퀀스 모델링에서 효율성을 입증했다. Mamba는 입력 데이터에 따라 정보를 선택적으로 수용하는 Selective Mechanism을 통해 Transformer를 능가하는 성능과 선형적인 연산 효율성을 보여주었다. 본 논문은 이러한 Mamba의 가능성을 음성 인식 분야에 적용하여 Transformer의 보완재로 활용하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

Speech-Mamba는 Joint Encoder-Decoder 프레임워크를 기반으로 하며, CTC 손실 함수를 결합하여 음성 입력을 텍스트 출력으로 변환한다. 전체 구조는 Mamba Encoder와 Mamba Decoder로 구성된다.

### 주요 구성 요소 및 역할

1. **Mamba Encoder**: 음성 입력의 음향 특징(Acoustic features)을 받아 중간 숨겨진 표현(Intermediate hidden representations)으로 변환한다.
    * 구조: $M$개의 Mamba Encoder Block으로 구성되며, 각 블록은 `Mamba block` $\rightarrow$ `RMSNorm` $\rightarrow$ `Multi-head Attention (RMS-ATT)` $\rightarrow$ `Mamba block` 순서로 배치된다.
    * 역할: 첫 번째 Mamba 블록이 고수준 음성 정보를 캡처하고, RMS-ATT와 두 번째 Mamba 블록이 시간적 및 전역적 문맥을 동시에 모델링한다.

2. **Mamba Decoder**: 텍스트 임베딩과 인코더의 출력을 받아 텍스트 시퀀스를 순차적으로 예측한다.
    * 구조: $N$개의 Mamba Decoder Block으로 구성되며, 각 블록은 `Mamba block` $\rightarrow$ `RMSNorm` $\rightarrow$ `Source-target Multi-head Attention (RMS-STA)` $\rightarrow$ `Mamba block` 순서로 배치된다.
    * 역할: 텍스트 임베딩을 Mamba 블록으로 먼저 처리한 후, RMS-STA를 통해 음성 표현과 텍스트 임베딩 간의 관계를 캡처하고, 마지막 Mamba 블록이 전역적인 텍스트-음성 지식을 학습한다.

3. **Mamba Block 세부 구조**:
    * 입력 데이터는 먼저 `RMSNorm`을 거치며, 이후 `MLP`와 `Gated MLP` (SiLU/Swish 활성화 함수 포함)를 통과한다.
    * 그 후 `1D Convolution`, `SiLU activation`, 그리고 핵심인 `Selective State Space Model (Selective SSM)`을 거쳐 숨겨진 특징을 생성한다.
    * 마지막으로 Non-linearity, MLP, Dropout 및 Residual Connection이 적용되어 최종 특징을 도출한다.

### 학습 목표 및 손실 함수

모델은 CTC 손실과 S2S 손실을 동시에 최소화하는 다중 목표 학습(Multi-objective learning) 방식을 채택한다. 전체 손실 함수는 다음과 같이 정의된다.

$$L_{\text{Speech-Mamba}} = \alpha L_{\text{CTC}} + (1-\alpha) L_{\text{S2S}}$$

여기서 $\alpha \in [0, 1]$는 두 손실 함수의 가중치를 조절하는 하이퍼파라미터이다. CTC 손실은 인코더 출력과 타겟 텍스트 간의 단조 정렬(Monotonic alignment)을 보장하며, S2S 손실은 디코더의 출력과 타겟 텍스트 간의 Cross-entropy로 계산된다.

## 📊 Results

### 실험 설정

* **데이터셋**: LibriSpeech 데이터셋을 사용하였다. 특히 긴 문맥 성능 평가를 위해 화자별로 발화를 병합하여 45초 이상 60초 미만의 길이를 가진 **Long-context 데이터셋**(`dev-clean-L`, `test-clean-L` 등)을 자체 구축하였다.
* **지표**: 단어 에러율(Word Error Rate, WER)을 사용하여 성능을 측정하였다.
* **기준선(Baseline)**: Joint CTC-S2S 손실을 사용하는 Transformer ASR 모델을 사용하였다.

### 주요 결과

1. **일반 인식 성능**: 100시간의 데이터로 학습시킨 결과, Speech-Mamba가 Transformer baseline보다 전반적으로 우수한 WER을 기록하였다.
2. **긴 문맥 처리 능력**: 45~60초 길이의 데이터셋에서 Speech-Mamba는 Transformer 대비 65% 이상의 상대적 성능 향상을 보였으며, 특히 `test-clean-L`과 `dev-clean-L`에서는 최대 84%의 향상을 달성하였다.
3. **시퀀스 길이 확장 실험**: 발화 길이를 70초, 80초, 90초, 100초까지 확장하여 테스트한 결과, Transformer는 길이가 길어질수록 WER이 급격히 증가(67.87% @ 100s)한 반면, Speech-Mamba는 완만하게 증가(10.74% @ 100s)하며 압도적인 성능 차이를 보였다.
4. **SOTA 모델과의 비교**: 960시간의 데이터로 확장 학습하여 비교했을 때, Speech-Mamba(67.6M 파라미터)는 훨씬 더 많은 파라미터를 가진 Whisper-Large-V3(1550M)보다 모든 테스트셋에서 우수한 성능을 보였으며, Gemini 1.5 Pro와 비교해서도 긴 문맥 데이터셋에서 경쟁력 있거나 더 우수한 성능을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 Mamba의 Selective SSM이 긴 시퀀스의 핵심 정보를 효율적으로 압축하고 유지하는 능력이 탁월함을 보여주었다. 특히 Ablation Study를 통해 Mamba Encoder를 제거했을 때 긴 시퀀스 성능이 가장 크게 하락함을 확인하였으며, 이는 인코더 단계에서의 전역적 문맥 캡처가 긴 음성 인식의 핵심임을 시사한다. 또한, 파라미터 수가 훨씬 적음에도 불구하고 대규모 모델(Whisper 등)보다 높은 성능을 낸 것은 Mamba 아키텍처의 효율성이 매우 높음을 입증한다.

### 한계 및 논의사항

논문에서는 주로 영어 데이터셋인 LibriSpeech에 집중하였으므로, 다른 언어나 더 다양한 소음 환경에서의 일반화 성능에 대한 검증이 추가적으로 필요하다. 또한, Transformer의 Attention과 Mamba의 SSM을 결합하는 구체적인 배치 순서나 하이퍼파라미터 $\alpha$의 최적값에 대한 심층적인 분석은 부족한 편이다.

## 📌 TL;DR

Speech-Mamba는 Transformer의 시간적 표현 학습 능력과 Mamba(Selective SSM)의 선형 복잡도 기반 롱-레인지 모델링 능력을 결합한 새로운 ASR 아키텍처이다. 실험 결과, 100초에 달하는 매우 긴 음성 시퀀스에서도 Transformer 대비 압도적인 전사 정확도를 보였으며, Whisper-Large-V3와 같은 거대 모델보다 적은 파라미터로 더 높은 효율성과 성능을 달성하였다. 이 연구는 향후 차세대 긴 문맥 음성 인식 시스템의 기반 모델로서 높은 잠재력을 가진다.
