# Adversarial Deep Metric Learning for Cross-Modal Audio-Text Alignment in Open-Vocabulary Keyword Spotting

Youngmoon Jung, Yong-Hyeok Lee, Myunghun Jung, Jaeyoung Roh, Chang Woo Han, Hoon-Young Cho (2025)

## 🧩 Problem to Solve

본 논문은 텍스트 등록 기반의 **Open-Vocabulary Keyword Spotting (KWS)**에서 발생하는 **교차 모달리티 이질성(Cross-modal heterogeneity)** 문제를 해결하고자 한다. 

일반적인 Open-Vocabulary KWS 시스템은 사용자가 텍스트로 키워드를 등록하면, 시스템이 이를 오디오 신호와 매칭하여 탐지하는 방식을 취한다. 이를 위해 Deep Metric Learning (DML)을 사용하여 오디오 임베딩(Acoustic Embedding, AE)과 텍스트 임베딩(Text Embedding, TE)을 공통의 임베딩 공간(Shared embedding space)으로 투영하여 비교한다. 그러나 오디오와 텍스트는 본질적으로 서로 다른 성질을 가진 데이터 형태(Modality)이기 때문에, 두 모달리티 간의 표상 차이(Domain gap)가 크게 발생하여 정확한 정렬(Alignment)이 어렵다는 문제가 있다.

따라서 본 연구의 목표는 오디오와 텍스트 임베딩 간의 모달리티 격차를 줄여, 텍스트로 등록된 키워드를 오디오 스트림에서 보다 정확하게 탐지할 수 있는 **Adversarial Deep Metric Learning (ADML)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **Modality Adversarial Learning (MAL) 도입**: 오디오와 텍스트 인코더가 모달리티에 구애받지 않는(Modality-invariant) 임베딩을 생성하도록 모달리티 분류기를 적대적으로 학습시키는 MAL을 제안하였다. 이를 통해 두 모달리티 간의 이질성을 완화한다.
2.  **음소 수준 정렬을 위한 DML 최적화**: Cross-attention 메커니즘을 통해 오디오와 텍스트의 음소 수준(Phoneme-level) 정렬을 수행하고, 여기에 Adaptive Margins and Scaling (AdaMS)이 적용된 Asymmetric-Proxy (AsyP) 손실 함수를 도입하여 성능을 극대화하였다.
3.  **SphereFace2 기반 키워드 분류 손실 적용**: 오디오 인코더의 내부 모달리티 판별력을 높이기 위해, 얼굴 인식 및 화자 확인 분야에서 검증된 SphereFace2 손실 함수를 KWS 태스크에 확장 적용하였다.

## 📎 Related Works

기존의 텍스트 기반 KWS 연구들은 주로 다음과 같은 접근 방식을 취했다.

*   **음소 수준 매칭**: Dynamic Sequence Partitioning (DSP) 알고리즘을 통해 오디오 시퀀스를 텍스트 길이에 맞게 정렬한 후 대조 손실(Contrastive loss)을 적용하는 방식이 사용되었다.
*   **모달리티 이질성 해결**: 일부 연구에서는 사전 학습된 오디오 인코더의 프레임 임베딩을 평균 내어 텍스트 임베딩을 생성하는 'Audio-compliant text encoder' 방식을 제안하였다. 그러나 이 방식은 대규모 음성 데이터셋으로 학습된 매우 강력한 오디오 인코더가 필요하며, 텍스트 임베딩 추출 과정이 복잡하다는 한계가 있다.

본 논문은 이러한 기존 방식과 달리, 학습 과정에서만 사용되는 **모달리티 분류기(Modality Classifier)**를 통해 적대적 학습을 수행함으로써, 추가적인 복잡한 전처리 없이도 유연하게 모달리티 불일치 문제를 해결한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
본 시스템은 오디오 입력을 처리하는 **Acoustic Encoder (ECAPA-TDNN)**와 텍스트 입력을 처리하는 **Text Encoder (bi-LSTM)**로 구성된다. 각 인코더는 음소 수준 임베딩($E_{phn}^a, E_{phn}^t$)과 발화 수준 임베딩($E_{utt}^a, E_{utt}^t$)을 생성한다.

### 2. 음소 수준 매칭 (Phoneme-Level Matching)
오디오와 텍스트 시퀀스의 길이 차이를 해결하기 위해 **Cross-attention** 메커니즘을 사용한다. 텍스트 임베딩 $E_{phn}^t$를 Query($Q$)로, 오디오 임베딩 $E_{phn}^a$를 Key($K$)와 Value($V$)로 설정하여 어피니티 행렬(Affinity matrix) $A$를 계산하고, 정렬된 오디오 임베딩 $\tilde{E}_{phn}^a$를 다음과 같이 산출한다.

$$\tilde{E}_{phn}^a = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \times V = A \times V$$

이후, $\tilde{E}_{phn}^a$와 $E_{phn}^t$ 사이의 거리를 최적화하기 위해 **AsyP loss**($L_{phn}$)를 적용한다. 이 손실 함수는 앵커-포지티브 쌍은 가깝게, 앵커-네거티브 쌍은 멀게 밀어내며, 하이퍼파라미터 $\alpha, \beta, \lambda$를 각 음소 클래스별로 학습 가능한 파라미터로 설정하는 **AdaMS** 프레임워크를 적용하여 자동 최적화한다.

### 3. 발화 수준 매칭 및 키워드 분류
*   **발화 수준 매칭**: 오디오는 CCSP(Channel- and Context-dependent Statistics Pooling)를, 텍스트는 Global Average Pooling을 통해 발화 수준 임베딩을 생성하고, **Relational Proxy (RP) loss**($L_{utt}$)를 통해 텍스트의 구조적 정보를 오디오 임베딩으로 전이시킨다.
*   **키워드 분류**: 오디오 인코더의 판별력을 높이기 위해 **SphereFace2** 손실($L_{key}$)을 보조 목표로 사용한다. 이는 다중 클래스 분류 대신 다수의 이진 분류기를 사용하는 방식으로 학습된다.

### 4. 모달리티 적대적 학습 (Modality Adversarial Learning, MAL)
두 모달리티의 표상을 유사하게 만들기 위해, 입력 임베딩이 오디오인지 텍스트인지를 판별하는 **모달리티 분류기 $M$**을 도입한다. 

*   **손실 함수**: 음소 수준($L_{phn}^{adv}$)과 발화 수준($L_{utt}^{adv}$)에서 각각 교차 엔트로피 손실을 계산하여 합산한 $L_{adv}$를 정의한다.
*   **학습 전략**: **Gradient Reversal Layer (GRL)**를 사용하여, 순전파 시에는 입력을 그대로 전달하지만 역전파 시에는 기울기에 $-1$을 곱한다. 결과적으로 인코더들은 분류기가 모달리티를 구분하지 못하도록(즉, 모달리티 불변 표상을 생성하도록) 학습된다.

최종 임베딩 학습 손실 함수는 다음과 같다.
$$L_{emb} = L_{utt} + L_{key} + L_{MM} + \lambda_{phn} L_{phn}$$
여기서 $L_{MM}$은 정렬 행렬이 대각 성분을 갖도록 강제하는 Monotonic Matching loss이다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: King-ASR-066(학습), WSJ 및 LibriPhrase(평가).
*   **지표**: WSJ에서는 Average Precision (AP), LibriPhrase에서는 Equal-Error Rate (EER) 및 Area Under the ROC Curve (AUC)를 사용하였다.
*   **모델**: ECAPA-TDNN (Acoustic), bi-LSTM (Text).

### 2. 주요 결과
*   **음소 수준 매칭 손실 비교**: AsyP loss에 AdaMS를 적용했을 때 85.54% AP를 기록하여, 기존의 Proxy-BD나 Proxy-MS 방식보다 우수한 성능을 보였다.
*   **MAL의 효과**: 발화 수준 MAL만 적용했을 때 AP가 75.10%에서 81.41%로 상승했으며, 음소 및 발화 수준 모두에 MAL을 적용했을 때 86.23% AP로 가장 높은 성능을 기록하였다. 특히 CMCD, PhonMatchNet 같은 기존 베이스라인 모델에 MAL을 추가했을 때도 성능이 향상됨을 확인하여 MAL의 범용성을 입증하였다.
*   **키워드 분류 손실 비교**: SphereFace2를 적용했을 때 AP가 90.24%까지 상승하여, Triplet loss나 AAM-Softmax보다 훨씬 효과적임을 확인하였다.
*   **LibriPhrase 결과**: ADML+SF2 모델이 EER 1.33%(Easy), 20.09%(Hard)를 기록하며 기존 SOTA 모델들보다 뛰어난 일반화 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 단순히 임베딩 공간을 공유하는 것을 넘어, 적대적 학습을 통해 **모달리티 간의 도메인 갭을 명시적으로 줄여야 한다**는 점을 시사한다. 특히 GRL을 이용한 MAL 방식은 추가적인 추론 비용 없이 학습 단계에서만 작동하므로, 실시간성이 중요한 KWS 시스템에 매우 적합한 구조이다.

또한, 음소 수준의 세밀한 정렬과 발화 수준의 전역적 특징 추출을 동시에 수행하고, 여기에 강력한 분류 손실 함수(SphereFace2)를 결합함으로써 상호 보완적인 효과를 얻었음을 알 수 있다.

한계점으로는, 현재의 적대적 학습이 단순한 이진 분류 기반이라는 점이다. 향후 더 정교한 적대적 학습 기법을 도입한다면 모달리티 불변 표상을 더욱 정밀하게 학습할 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 텍스트 등록 기반 Open-Vocabulary KWS에서 오디오와 텍스트 간의 표상 차이를 줄이기 위해 **모달리티 적대적 학습(MAL)**, **AdaMS-AsyP 기반 음소 정렬**, 그리고 **SphereFace2 분류 손실**을 결합한 **ADML** 프레임워크를 제안하였다. 실험 결과, 적대적 학습이 두 모달리티의 정렬 성능을 유의미하게 향상시켰으며, 특히 기존의 다른 KWS 모델들에도 적용 가능한 범용적인 성능 향상 도구임을 확인하였다. 이 연구는 향후 텍스트-오디오 교차 모달리티 학습의 효율적인 기준을 제시할 가능성이 높다.