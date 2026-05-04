# Open Vocabulary Keyword Spotting through Transfer Learning from Speech Synthesis

Kesavaraj V, Anil Kumar Vuppala (2024)

## 🧩 Problem to Solve

본 논문은 사용자가 정의한 임의의 키워드를 인식해야 하는 Open Vocabulary Keyword Spotting (KWS), 또는 Custom Keyword Spotting 문제를 해결하고자 한다. 기존의 Closed Vocabulary KWS는 미리 정해진 키워드 집합만을 인식하지만, 실제 스마트 기기 환경에서는 사용자 맞춤형 개인화 인터랙션을 위해 학습 단계에서 보지 못한 임의의 키워드를 인식할 수 있는 능력이 필수적이다.

기존의 Open Vocabulary KWS 접근 방식들은 주로 오디오 인코더와 텍스트 인코더를 통해 두 모달리티를 공통의 임베딩 공간(shared embedding space)으로 투영하여 일치 여부를 판단하는 방식을 사용한다. 그러나 오디오와 텍스트라는 서로 다른 성질의 데이터 표현 방식(heterogeneous modality representations)으로 인해 발생하는 모달리티 간 불일치(audio-text mismatch) 문제가 존재하며, 특히 발음이 유사한 단어들을 구분하는 데 어려움이 있다는 점이 핵심적인 해결 과제이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 사전 학습된 Text-to-Speech (TTS) 시스템이 가지고 있는 지식을 전이 학습(Transfer Learning)하여 텍스트 표현에 '음향적 정보'를 주입하는 것이다. TTS 모델은 텍스트를 오디오로 변환하는 과정을 학습하므로, 그 내부의 중간 표현(intermediate representations)에는 텍스트가 실제로 어떻게 소리 나는지에 대한 오디오 투영 정보가 포함되어 있다. 이를 통해 텍스트 인코더가 음향적 인식을 갖게 함으로써, 오디오 임베딩과 텍스트 임베딩을 공통 잠재 공간으로 투영할 때 발생하는 간극을 줄이고 변별력을 높이는 전략을 제안한다.

## 📎 Related Works

논문에서는 Custom KWS를 위한 기존 접근 방식을 세 가지로 분류하여 설명한다.

1.  **Query-by-Example (QbyE):** 미리 등록된 오디오 예시와 입력 쿼리를 비교하는 방식이다. 하지만 등록 시의 음성과 평가 시의 음성 간의 일관성이 낮거나, 화자의 특성 및 배경 소음 등의 환경 변화에 매우 취약하다는 한계가 있다.
2.  **ASR(자동 음성 인식) 기반 방식:** 입력 스트림에서 음소 패턴을 검출하고 이를 등록된 키워드의 표현과 비교한다. 이 방식은 사용된 Acoustic Model의 정확도에 성능이 완전히 의존한다는 단점이 있다.
3.  **End-to-End 공통 잠재 공간 방식:** 오디오와 텍스트를 하나의 공유 공간으로 투영하여 매칭하는 최근의 기법들이다. CMCD(Cross-Modality Correspondence Detector) 등이 이에 해당하며, 유망한 결과를 보이지만 발음이 매우 유사한 쌍을 구분하는 정밀한 변별력으로는 여전히 한계가 있다.

본 논문은 이러한 End-to-End 방식의 틀을 유지하되, 텍스트 인코더의 초기화 단계에서 TTS의 지식을 활용함으로써 기존 방식들의 모달리티 불일치 문제를 해결하고자 한다.

## 🛠️ Methodology

제안된 프레임워크는 텍스트 인코더, 오디오 인코더, 패턴 추출기, 패턴 판별기의 네 가지 서브모듈로 구성된다.

### 1. Text Encoder
사전 학습된 Tacotron 2 모델을 기반으로 한다. 입력된 키워드의 문자 시퀀스가 Tacotron 2를 통과하며 생성되는 중간 표현을 추출한다.
- **구조:** Tacotron 2 $\rightarrow$ Bi-GRU (dim: 64) $\rightarrow$ Dense Layer (dim: 128).
- **결과물:** $T_E \in \mathbb{R}^{d \times m}$ (여기서 $d$는 임베딩 차원, $m$은 텍스트 길이).
- **목적:** 단순한 텍스트 임베딩이 아닌, TTS의 내부 지식을 통해 음향적 특성이 반영된 텍스트 표현을 생성하여 오디오 임베딩과의 정렬을 용이하게 한다.

### 2. Audio Encoder
입력 오디오에서 특징을 추출하여 텍스트와 비교 가능한 차원으로 투영한다.
- **입력:** 80차원의 Mel-filterbank coefficients (window: 25ms, stride: 10ms).
- **구조:** 
    - 2D Convolution layers $\times 2$ (filters: 32, 64 / kernel size: 3). 첫 번째 레이어는 stride 2를 사용하여 프레임 수를 줄인다.
    - Batch Normalization $\rightarrow$ Bi-GRU $\times 2$ (dim: 64) $\rightarrow$ Dense Layer (dim: 128).
- **결과물:** $A_E \in \mathbb{R}^{d \times n}$ (여기서 $d$는 임베딩 차원, $n$은 오디오 프레임 수).

### 3. Pattern Extractor
오디오와 텍스트 임베딩 간의 시간적 상관관계를 캡처하기 위해 Cross-Attention 메커니즘을 사용한다.
- **작동 방식:** 텍스트 임베딩 $T_E$가 Query($Q$) 역할을 수행하고, 오디오 임베딩 $A_E$가 Key($K$)와 Value($V$) 역할을 수행한다.
- **결과물:** 오디오와 텍스트의 일치 정보가 담긴 Context Vector를 생성한다.

### 4. Pattern Discriminator
추출된 패턴이 실제로 일치하는지 여부를 최종 판별한다.
- **구조:** Bi-GRU (dim: 128) $\rightarrow$ Dense Layer $\rightarrow$ Sigmoid activation.
- **목표 변수:** 오디오와 텍스트 입력이 동일한 키워드인지 여부를 결정하는 이진 분류(Binary Classification) 값이다.

### 5. 학습 절차
- **손실 함수:** 이진 교차 엔트로피 손실(Binary Cross-Entropy Loss)을 사용하여 학습한다.
- **최적화:** Adam Optimizer를 사용하며, 학습률은 $10^{-4}$, 배치 크기는 128로 설정하였다.

## 📊 Results

### 실험 설정
- **데이터셋:** LibriPhrase (Easy $LP_E$, Hard $LP_H$), Google Speech Commands V1 (G), Qualcomm Keyword Speech (Q).
- **지표:** Equal Error Rate (EER), Area Under the Curve (AUC), F1 score.
- **비교 대상:** CTC, Attention, Triplet, CMCD 등 기존 KWS 기법들.

### 주요 결과
1.  **전반적 성능:** 제안된 방법은 G 데이터셋을 제외한 모든 데이터셋에서 베이스라인 모델들을 능가하였다. 특히, 발음이 유사한 단어들이 포함된 $LP_H$ 데이터셋에서 CMCD 대비 AUC는 8.22% 상승, EER은 12.56% 감소하는 괄목할만한 성능 향상을 보였다.
2.  **TTS 중간 표현 분석 (Ablation Study):** Tacotron 2의 다양한 레이어($E1 \sim E7$)에서 추출한 표현을 실험한 결과, Bi-LSTM 블록의 출력인 $E3$가 모든 데이터셋에서 가장 낮은 EER과 가장 높은 AUC/F1 score를 기록하였다. 이는 $E3$가 음향적 정보와 언어적 정보를 가장 효과적으로 캡처하고 있음을 시사한다.
3.  **단어 길이 및 OOV 강건성:** 단어 길이가 길어질수록 EER이 상승하는 경향이 있으나, 모든 길이(1~4단어)에서 일관된 성능을 유지하였다. 또한, 학습 시 보지 못한 단어를 인식하는 OOV(Out-of-Vocabulary) 시나리오에서도 CMCD 대비 F1 score가 7.25% 절대적으로 향상되어 강건함을 입증하였다.
4.  **수렴 속도:** $E3$ 표현을 사용한 네트워크가 CMCD 모델보다 학습 손실(training loss)이 더 빠르게 감소하며 수렴 속도가 빠름을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 단순히 텍스트를 벡터화하는 것이 아니라, TTS 모델의 "텍스트 $\rightarrow$ 음성" 변환 지식을 이용해 텍스트 임베딩을 '음성 친화적'으로 만든 것이 핵심이다. 

특히 주목할 점은 TTS 모델의 Encoder 단계 표현($E2, E3$)이 Decoder 단계 표현($E5, E6, E7$)보다 KWS 작업에 훨씬 적합했다는 것이다. 이는 멜-스펙트로그램이 실제로 생성되기 전의 추상적인 음향 표현을 캡처하는 것이 키워드 검출에 더 중요하다는 점을 시사한다. 

또한, 발음이 매우 유사한 단어들(예: "madame" vs "modem")을 구분하는 능력이 크게 향상된 것은, 사전 학습된 TTS 모델이 이미 정교한 음소-음향 매핑 정보를 가지고 있기 때문에 이를 전이받은 텍스트 인코더가 더 세밀한 변별력을 갖게 되었음을 의미한다.

## 📌 TL;DR

본 연구는 사전 학습된 Tacotron 2 (TTS) 모델의 중간 표현을 전이 학습하여, 텍스트 임베딩에 음향적 정보를 주입한 Open Vocabulary Keyword Spotting 프레임워크를 제안한다. 실험 결과, 특히 발음이 유사한 어려운 사례($LP_H$)와 학습되지 않은 단어(OOV) 인식에서 기존 CMCD 방식보다 뛰어난 성능을 보였으며, TTS 모델의 Bi-LSTM 블록 출력($E3$)이 가장 효과적인 표현임이 밝혀졌다. 이 연구는 TTS 지식을 활용해 오디오-텍스트 간의 모달리티 간극을 줄이는 새로운 방향성을 제시하였다.