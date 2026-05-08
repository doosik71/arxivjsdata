# Keyword Spotting with Hyper-Matched Filters for Small Footprint Devices

Yael Segal-Feldman, Ann R. Bradlow, Matthew Goldrick, and Joseph Keshet (2025)

## 🧩 Problem to Solve

본 논문은 소형 기기(small-footprint devices)에서 효율적으로 동작하면서도, 학습 단계에서 보지 못한 새로운 단어를 인식할 수 있는 **Open-vocabulary Keyword Spotting (KWS)** 문제를 해결하고자 한다.

기존의 KWS 시스템은 사전에 정의된 키워드 세트만을 인식하는 폐쇄형 구조이거나, 새로운 키워드를 추가하기 위해 전체 모델을 재학습시켜야 하는 한계가 있었다. 또한, 기존의 Open-vocabulary 접근 방식들은 성능을 높이기 위해 모델 크기를 키움으로써 소형 기기에 탑재하기 어렵거나, 반대로 크기를 줄였을 때 인식 성능이 급격히 저하되는 트레이드-오프(trade-off) 문제가 존재했다. 따라서 본 연구의 목표는 모델의 파라미터 수를 최소화하면서도, 다양한 환경과 언어(특히 비원어민의 L2 음성)에 대해 높은 일반화 성능을 보이는 효율적인 KWS 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Hyper-network를 이용해 키워드별 맞춤형 필터(Matched Filter)의 가중치를 동적으로 생성**하고, 이를 **Perceiver 아키텍처**와 결합하여 연산 효율성을 극대화하는 것이다.

구체적인 기여 사항은 다음과 같다.

1. **Hyper-Matched Filter 설계**: 타겟 키워드의 문자열을 입력받아 해당 키워드에 최적화된 합성곱(Convolution) 층의 가중치를 생성하는 Hyper-network를 제안하였다. 이는 신호 처리의 Matched Filter 개념을 딥러닝에 접목하여, 입력 음성에서 특정 키워드의 특성을 효과적으로 추출한다.
2. **효율적인 Detection Network**: 고차원 입력을 고정된 크기의 잠재 공간(latent bottleneck)으로 매핑하는 Perceiver 구조를 채택하여, 트랜스포머의 성능을 유지하면서도 파라미터 수를 획기적으로 줄였다.
3. **강력한 일반화 성능 입증**: 원어민뿐만 아니라 비원어민(L2) 화자의 음성, 그리고 학습에 사용되지 않은 저자원 언어(low-resource languages)에 대해서도 높은 검출 성능을 보임을 실험적으로 증명하였다.

## 📎 Related Works

논문에서는 KWS 연구의 흐름을 세 가지 방향으로 설명한다.

- **소형 기기용 KWS**: 초기에는 CNN과 RNN 기반의 인코더를 사용하였으며, 이후 Residual layers나 Depthwise convolutions 등을 통해 효율성을 높이려 하였다. 그러나 이러한 방식들은 대부분 미리 정의된 키워드만을 인식할 수 있다는 제약이 있다.
- **Open-vocabulary KWS**:
  - **Query-by-Example (QbE)**: 타겟 키워드의 오디오 샘플을 참조로 사용하는 방식이나, 녹음 환경의 변화에 매우 민감하다는 단점이 있다.
  - **Embedding-based**: 오디오와 텍스트를 공통 임베딩 공간으로 매핑하여 거리를 측정하는 방식이다.
  - **CTC-based**: CTC loss를 이용해 음성을 텍스트 시퀀스로 변환하고 편집 거리(edit distance) 등을 통해 키워드를 찾는 방식이다.
- **Hyper-networks**: 특정 입력에 따라 다른 네트워크의 가중치를 생성하는 구조이다. 기존 연구에서 단일 CNN 층이나 트랜스포머의 정규화 층(normalization layer) 가중치를 생성하는 시도가 있었으나, 본 논문은 이를 Perceiver의 Cross-attention 메커니즘과 결합하여 소형 기기에 최적화하였다.

## 🛠️ Methodology

제안된 모델인 **HyperSpotter**는 크게 세 가지 모듈(Speech Encoder, Target Keyword Encoder, Detection Network)로 구성된다.

### 1. Speech Encoder

음성 신호 $x$를 입력받아 표현 벡터 시퀀스 $z$로 변환한다. 본 연구에서는 두 가지 인코더를 실험하였다.

- **tiny Whisper**: 4개의 트랜스포머 레이어로 구성되며, 특징 크기 $M=384$이다. (약 7.6M 파라미터)
- **tiny Conformer**: 6개의 인코더 레이어로 구성되며, 특징 크기 $M=144$이다. (약 3.7M 파라미터)

### 2. Target Keyword Encoder (Hyper-network)

타겟 키워드 $k$(문자열 시퀀스)를 입력받아 합성곱 층의 가중치 $W_k$를 생성한다.

- 구조: 문자 임베딩 층 $\rightarrow$ 4개의 LSTM 레이어(hidden size 256) $\rightarrow$ 2개의 선형 층(Linear layers).
- 역할: 생성된 $W_k$는 키워드 전용 Matched Filter 역할을 수행하며, 타겟 신호의 신호 대 잡음비(SNR)를 최대화하도록 학습된다. 이 모듈은 오프라인에서 가중치를 미리 생성할 수 있어 기기 자체에 탑재될 필요가 없다.

### 3. Detection Network

음성 임베딩 $z$와 키워드 가중치 $W_k$를 입력으로 받아 키워드 존재 여부를 판별한다. 효율성을 위해 **Perceiver** 아키텍처를 사용한다.

- **처리 단계**:
    1. 고차원 $M$ 공간을 $N=64$ 차원으로 투영한다.
    2. **Keyword-specific Convolution**: Hyper-network가 생성한 $W_k$를 사용하여 Depth-wise convolution을 수행한다. 이는 Perceiver의 Cross-attention 가중치를 가이드하여 오디오 인코더의 출력을 타겟 키워드 방향으로 유도한다.
    3. 가변 길이 $B$의 요소를 고정 길이 $S=16$의 벡터로 매핑한다.
    4. 이후 Latent Transformer 레이어를 통해 최종 판단을 내린다.

### 4. 학습 목표 및 손실 함수

모델은 이진 분류 문제로 정의되며, **Binary Cross-Entropy (BCE) Loss**를 사용하여 학습한다.

$$L_{BCE} = -\frac{1}{E} \sum_{i=1}^{E} [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

여기서 $E$는 배치 크기, $y_i$는 키워드 존재 여부(0 또는 1), $\hat{y}_i$는 모델이 예측한 확률값이다.

## 📊 Results

### 실험 설정

- **데이터셋**: VoxPopuli(다국어), LibriPhrase(영어, Hard/Easy split), Speech Commands V1, FLEURS(저자원 언어), Wildcat Diapix(L1/L2 영어 화자).
- **평가 지표**: AUC, F1 score, Equal Error Rate (EER), FRR@FAR5%.
- **비교 대상**: CED, EMKWS, CMCD 및 최신 기법인 AdaKWS-tiny.

### 주요 결과

1. **In-domain 성능 (VoxPopuli)**:
    - HyperSpotter-c (Conformer 기반) 모델이 가장 우수한 성능을 보였다. 특히 HyperSpotter-c (4) 모델이 AUC, F1, EER 모든 지표에서 최상위 성능을 기록하였다.
    - 모든 HyperSpotter 모델이 AdaKWS-tiny*보다 우수한 성능을 보였다.
2. **Fine-tuning 결과 (LibriPhrase)**:
    - HyperSpotter-c (3, 4) 모델이 모든 베이스라인 모델을 압도하였다.
    - 특히 4.2M 파라미터의 가장 작은 모델(HyperSpotter-c (1))이 EMKWS나 CED보다 높은 성능을 기록하며 효율성을 입증하였다.
3. **Out-of-domain 및 일반화 성능**:
    - **L2 화자 (Wildcat)**: Whisper 기반의 HyperSpotter-w 모델이 Conformer 기반 모델보다 더 강력한 강건성을 보였으며, 비원어민 화자에 대해서도 높은 정확도를 유지하였다.
    - **저자원 언어 (FLEURS)**: HyperSpotter-c (4)가 가장 좋은 성능을 보였으며, 전반적으로 HyperSpotter 모델들이 경쟁력을 유지하였다.
    - **소음 강건성**: 화이트 노이즈와 Pub 노이즈 환경에서도 모든 모델이 안정적인 성능을 보였으며, 특히 HyperSpotter-c (4)가 가장 우수하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **효율성과 성능의 조화**: Perceiver 구조와 Hyper-network를 결합함으로써 4.2M~10M 정도의 매우 작은 파라미터 수로도 기존의 대형 모델들과 대등하거나 더 뛰어난 성능을 낼 수 있음을 보여주었다.
- **Matched Filter의 작동 확인**: Cross-attention 맵의 히트맵 분석 결과, 모델이 실제로 음성 신호 내에서 타겟 키워드가 발화된 정확한 위치에 높은 확률 값을 할당하고 있음이 확인되었다. 이는 Hyper-network가 생성한 가중치가 성공적으로 Matched Filter 역할을 수행하고 있음을 시사한다.
- **인코더별 특성**: Conformer 기반 모델은 학습 데이터와 유사한 환경(In-domain)에서 매우 강력한 성능을 보이지만, Whisper 기반 모델은 사전 학습 데이터의 양이 방대하여 미지의 환경이나 외국인 억양(Out-of-domain, L2)에 대해 더 높은 강건성을 보인다는 트레이드-오프를 발견하였다.

### 한계 및 논의사항

- **IV vs OOV 구분 모호성**: 학습 시 키워드를 동적으로 생성하고 무작위로 선택했기 때문에, 평가 시 사용된 키워드가 학습 과정에서 한 번이라도 등장했는지(In-Vocabulary) 아니면 완전히 처음 보는 것인지(Out-of-Vocabulary) 명확히 구분하여 분석하지 못했다. 이는 Open-vocabulary 모델의 순수한 제로샷(Zero-shot) 능력을 정밀하게 측정하는 데 한계로 작용한다.

## 📌 TL;DR

본 논문은 소형 기기에서 구동 가능한 **Open-vocabulary KWS 모델인 HyperSpotter**를 제안한다. 핵심은 **Hyper-network를 통해 키워드 맞춤형 합성곱 필터(Matched Filter) 가중치를 생성**하고, 이를 **Perceiver** 구조의 Cross-attention에 적용하여 연산량을 줄이면서도 검출 정밀도를 높인 것이다. 실험 결과, 4.2M 파라미터의 초경량 모델로도 SOTA 수준의 성능을 달성하였으며, 특히 비원어민(L2)의 음성 인식에서도 뛰어난 일반화 능력을 보여 향후 다양한 실제 환경의 음성 인터페이스 적용 가능성을 높였다.
