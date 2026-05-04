# Speech Augmentation Based Unsupervised Learning for Keyword Spotting

Jian Luo, Jianzong Wang, Ning Cheng, Haobin Tang, Jing Xiao (2022)

## 🧩 Problem to Solve

본 논문은 키워드 검출(Keyword Spotting, KWS) 작업에서 발생하는 레이블링 된 데이터에 대한 과도한 의존성 문제를 해결하고자 한다. KWS는 오디오 스트림에서 미리 정의된 소수의 키워드를 탐지하는 기술로, 인터랙티브 에이전트의 시작 명령어나 프라이버시 보호를 위한 민감 단어 탐지에 널리 사용된다.

하지만 높은 정확도와 강건성을 갖춘 KWS 시스템을 구축하기 위해서는 방대한 양의 레이블링 된 데이터가 필요하다. 특히 새로운 키워드를 추가하거나 변경할 때마다 긍정 샘플(positive samples)을 다시 수집하고 레이블링 하는 데 많은 시간과 비용이 소요된다는 점이 주요한 한계이다. 따라서 본 연구의 목표는 레이블이 없는 데이터를 활용하는 비지도 학습(Unsupervised Learning) 기법을 도입하여, 제한된 레이블 데이터만으로도 KWS 모델의 성능과 강건성을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 음성 데이터의 스타일 변화(속도 및 음량)에 관계없이 동일한 키워드는 유사한 고수준 특징 표현(high-level feature representation)을 가져야 한다는 직관에 기반한다. 이를 위해 다음과 같은 기여를 제시한다.

1.  **CNN-Attention 아키텍처 설계**: 국소적인 음향 특징을 추출하는 CNN과 장기 의존성(long-time dependency)을 모델링하는 Transformer의 Self-Attention 메커니즘을 결합한 구조를 제안하여 KWS 작업에서 경쟁력 있는 성능을 확보하였다.
2.  **증강 기반 비지도 학습 방법론**: 속도(Speed)와 강도(Intensity/Volume) 변화를 주는 데이터 증강을 적용하고, 원본 음성과 증강된 음성 간의 특징 유사도를 극대화하는 비지도 학습 방식을 제안하였다.
3.  **다중 손실 함수 기반의 사전 학습**: Bottleneck layer의 특징 유사성을 측정하는 MSE 손실과 더불어, 입력 음성의 평균 특징을 복원하는 보조 학습 과제를 설계하여 네트워크가 음성의 본질적인 특징을 학습하도록 유도하였다.

## 📎 Related Works

KWS를 위한 기존 접근 방식은 다음과 같이 발전해 왔다.
-   **전통적 방식**: 키워드/필러 Hidden Markov Model(HMM)이 널리 사용되었으나, 추론 시 Viterbi 디코딩으로 인한 계산 비용 문제가 존재한다.
-   **딥러닝 기반 방식**: DNN, CNN, RNN(CTC loss 포함) 등이 도입되었다. CNN은 시간/주파수 도메인의 강한 의존성을 잘 포착하지만 문맥 정보 모델링에 한계가 있으며, RNN은 국소적 구조 학습 능력이 부족하다. 이를 보완하기 위해 CRNN이나 Gated Convolutional LSTM 등이 제안되었다.
-   **Transformer 기반 방식**: Self-attention 메커니즘을 통해 음성 표현 능력을 높인 BERT 등의 모델이 NLP 및 ASR 분야에서 성공을 거두었으며, 본 논문 역시 이를 KWS에 적용하였다.
-   **비지도 학습 기반 방식**: Contrastive Predictive Coding(CPC), Autoregressive Predictive Coding(APC), Masked Predictive Coding(MPC) 등이 일반적인 음성 표현 학습을 위해 제안되었다.

본 논문은 기존 비지도 학습이 일반적인 오디오 표현 학습에 집중한 것과 달리, KWS 작업의 특성에 맞춰 **속도와 음량 변화에 대한 강건성**을 직접적으로 학습하는 증강 기반 접근 방식을 취한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. KWS 모델 아키텍처
제안된 CNN-Attention 모델은 입력 오디오 시퀀스 $X$를 키워드 클래스 $Y$로 매핑하며, 총 5개의 단계로 구성된다.

1.  **CNN Block**: $N$개의 2D-convolutional layer를 통해 시간 및 스펙트럼 축의 국소적 변화를 처리한다.
    $$E_{cnn} = 2\text{DConv}^{\times N}(X)$$
2.  **Transformer Block**: $M$개의 Self-attention layer를 통해 장기적인 문맥 정보를 캡처한다.
    $$E_{tran} = \text{SelfAttention}^{\times M}(E_{cnn})$$
3.  **Feature Selecting Layer**: 시퀀스의 마지막 $r$개 프레임을 선택하여 하나의 특징 벡터로 결합(Concatenate)한다.
    $$E_{feat} = \text{Concat}(E_{tran}[T-r, T])$$
4.  **Bottleneck Layer**: 전결합층(FC layer)을 통해 특징을 압축한다.
    $$E_{bn} = \text{FC}_{bn}(E_{feat})$$
5.  **Project Layer**: 최종적으로 키워드 클래스를 예측하는 소프트맥스 층으로 매핑한다.
    $$\tilde{Y} = \text{FC}_{proj}(E_{bn})$$

지도 학습 및 미세 조정(Fine-tuning) 시에는 예측값 $\tilde{Y}$와 정답 $Y$ 사이의 Cross-Entropy(CE) 손실 함수 $L_{ce} = \text{CE}(Y, \tilde{Y})$를 사용한다.

### 2. 데이터 증강 방법 (Augmentation Method)
입력 음성 $X$를 진폭 $A$와 시간 인덱스 $t$의 함수 $X=A(t)$로 정의하고, 두 가지 증강을 수행한다.
-   **속도 증강(Speed Augmentation)**: 속도 비율 $\lambda_{speed}$를 적용하여 시간축을 조정한다.
    $$X_{aug} = A(\lambda_{speed}t)$$
-   **음량 증강(Volume Augmentation)**: 강도 비율 $\lambda_{volume}$을 적용하여 진폭을 조정한다.
    $$X_{aug} = \lambda_{volume}A(t)$$

### 3. 비지도 학습 손실 함수 (Unsupervised Learning Loss)
사전 학습 단계에서는 동일한 파라미터를 공유하는 네트워크에 원본 $X$와 증강된 $X_{aug}$를 입력한다.

**1) 특징 유사성 손실 ($L_{sim}$)**
원본과 증강된 음성의 Bottleneck feature가 유사해야 한다는 가정하에 MSE를 사용하여 거리를 측정한다.
$$L_{sim} = \frac{1}{U_{bn}} \sum_{u=0}^{U_{bn}} |E_{bn}(u) - E_{aug\_bn}(u)|^2$$

**2) 음성 복원 손실 ($L_x$ 및 $L_{aug\_x}$)**
네트워크가 음성의 내재적 특징을 학습하도록 돕는 보조 과제이다. 입력 Fbank 벡터 $X$의 시간축 평균 벡터 $\bar{X}$를 구하고, Bottleneck feature로부터 이를 복원한 $\tilde{X}$와의 MSE를 계산한다.
$$\bar{X} = \frac{1}{T} \sum_{T}(X), \quad \tilde{X} = \text{FC}_{reconstruct}(E_{bn})$$
$$L_x = \frac{1}{U_x} \sum_{u=0}^{U_x} |\bar{X}(u) - \tilde{X}(u)|^2$$
증강된 데이터에 대해서도 동일하게 $L_{aug\_x}$를 정의한다.

**3) 최종 비지도 학습 손실 ($L_{ul}$)**
위의 세 가지 손실을 가중치 $\lambda_1, \lambda_2, \lambda_3$로 결합하여 최종 손실을 구성한다.
$$L_{ul} = \lambda_1 L_{sim} + \lambda_2 L_x + \lambda_3 L_{aug\_x}$$

## 📊 Results

### 실험 설정
-   **데이터셋**: Google Speech Commands V2 (12개 클래스).
-   **사전 학습 데이터**: Speech Commands 및 100시간의 Librispeech 데이터셋.
-   **입력 특징**: 40-dimensional log-mel filter-bank (30ms frame length, 10ms frame shift).
-   **비교 대상**: Google의 Sainath and Parada 모델(지도 학습), CPC, APC, MPC(비지도 학습).

### 주요 결과
1.  **아키텍처 성능**: 제안된 CNN-Attention 모델은 단순 지도 학습만으로도 Google의 모델(84.7%)보다 높은 85.3%의 정확도를 기록하여 아키텍처의 유효성을 입증하였다.
2.  **증강의 효과**: 지도 학습 시 속도 및 음량 증강을 추가했을 때 정확도가 85.7%로 향상되었다.
3.  **비지도 학습의 성능 향상**: 
    -   **Ablation Study**: 음량 증강보다 속도 증강 기반의 사전 학습이 더 효과적이었으며, 두 방법을 모두 사용했을 때 가장 높은 성능을 보였다. 특히 대규모 데이터셋(Librispeech-100)으로 사전 학습했을 때 Eval 정확도가 88.1%까지 상승하였다.
    -   **타 모델 비교**: 제안 방법은 CPC, APC, MPC 등 기존 비지도 학습 모델보다 우수한 성능을 보였으며, 특히 Librispeech-100 기반 사전 학습 시 가장 높은 성능(Eval 88.1%)을 달성하였다.
4.  **사전 학습 단계 분석**: 사전 학습 단계가 증가할수록(5K $\rightarrow$ 30K) 최종 정확도가 향상되었으며, 다운스트림 KWS 작업의 수렴 속도가 빨라지는 것을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 비지도 학습과 데이터 증강을 결합하여 KWS 모델의 강건성을 높이는 효과적인 방법을 제시하였다. 특히, 단순히 일반적인 음성 표현을 학습하는 것이 아니라, KWS 작업에서 발생할 수 있는 실제적인 변이(속도, 음량)를 직접적으로 다룸으로써 실용적인 성능 향상을 이끌어냈다.

**강점**:
-   레이블이 없는 대규모 데이터를 활용하여 데이터 부족 문제를 완화하였다.
-   속도 및 음량 변화에 강건한 특징 표현을 학습함으로써 실제 환경에서의 적용 가능성을 높였다.
-   CNN과 Transformer를 결합하여 국소 특징과 전역 문맥을 모두 효과적으로 포착하였다.

**한계 및 논의**:
-   본 논문에서 사용한 증강 기법은 속도와 음량에 국한되어 있다. 배경 소음이나 채널 왜곡과 같은 더 복잡한 환경 변화에 대한 대응책이 추가된다면 더욱 강건한 모델이 될 수 있을 것이다.
-   사전 학습 단계가 성능에 영향을 미치지만, 20K와 30K 단계 사이의 차이가 적다는 점은 학습 효율성 측면에서 최적의 pre-training step을 찾는 기준이 될 수 있다.

## 📌 TL;DR

본 논문은 레이블 데이터 부족 문제를 해결하기 위해 **속도 및 음량 증강 기반의 비지도 학습**을 적용한 **CNN-Attention KWS 모델**을 제안한다. 원본과 증강 음성 간의 특징 유사성을 극대화하는 $L_{sim}$ 손실과 음성 복원 보조 손실을 통해 사전 학습을 수행하며, 실험 결과 기존의 CPC, APC, MPC 등 비지도 학습 방식보다 우수한 성능을 보였으며 특히 대규모 데이터셋을 활용한 사전 학습 시 KWS 정확도와 수렴 속도가 크게 향상됨을 입증하였다. 이 연구는 향후 화자 인증이나 음성 인식 등 다른 음성 처리 작업으로 확장될 가능성이 크다.