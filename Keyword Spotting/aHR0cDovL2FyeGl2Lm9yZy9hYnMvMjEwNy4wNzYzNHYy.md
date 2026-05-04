# MULTI-TASK LEARNING WITH CROSS ATTENTION FOR KEYWORD SPOTTING

Takuya Higuchi, Anmol Gupta, Chandra Dhir (2021)

## 🧩 Problem to Solve

본 논문은 음성 기반 디바이스 활성화에 핵심적인 Keyword Spotting (KWS) 시스템의 정확도와 효율성을 높이는 문제를 다룬다. KWS 모델을 학습시키기 위해서는 대량의 레이블링 된 데이터가 필요하지만, 실제 도메인에서의 KWS 데이터셋을 구축하는 것은 개인정보 보호 문제와 False Trigger의 희소성으로 인해 현실적으로 매우 어렵다.

반면, 자동 음성 인식(Automatic Speech Recognition, ASR) 데이터셋은 대량으로 존재하지만, 이를 KWS에 그대로 사용할 경우 학습 기준(음소 인식, Phoneme Recognition)과 실제 목표 작업(키워드 분류, KWS) 사이의 불일치(Mismatch)가 발생한다. 따라서 본 연구의 목표는 ASR 데이터와 KWS 데이터를 모두 활용하는 Multi-task Learning (MTL) 프레임워크 내에 **Cross Attention Decoder**를 도입하여, 이러한 불일치를 해결하고 KWS 성능을 최적화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단순히 출력층을 분리하는 기존의 MTL 방식에서 벗어나, **학습 가능한 쿼리 시퀀스(Trainable Query Sequence)**를 이용한 Cross Attention 메커니즘을 통해 음성 인코더의 정보를 효율적으로 요약하는 디코더를 설계한 것이다.

기존 방식이 인코더의 출력을 단순하게 분기하여 사용했다면, 제안된 방법은 인코더가 추출한 음성적 표현(Phonetic representations)과 고정된 길이의 쿼리 벡터 간의 Cross Attention을 수행함으로써, 오디오 시퀀스 전체에서 키워드 판단에 필요한 핵심 정보를 응축하여 단일 confidence score를 예측하도록 설계되었다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 차별점을 제시한다:

1. **BLSTM 기반 디코더:** Bluche 등은 음소 분류기 위에 BLSTM 키워드 인코더를 사용하여 confidence score를 예측하는 방식을 제안하였다. 그러나 해당 연구는 음소 분류기를 사전 학습 후 고정(Fixed)시킨 반면, 본 논문은 인코더와 디코더를 MTL 프레임워크 내에서 처음부터 함께 학습(Jointly trained)시킨다.
2. **Vanilla Transformer:** Adya 등은 Transformer를 음소 분류기로 사용하여 KWS에 적용하였다. 본 논문의 Cross Attention Decoder는 일반적인 Transformer 디코더와 달리 Auto-regressive 모델이 아니며, 고정된 길이의 쿼리 벡터를 사용하여 스칼라 값의 confidence score만을 예측한다는 점에서 구조적 차이가 있다.
3. **RNN Transducer 기반 MTL:** Tian 등은 ASR과 KWS 데이터를 모두 사용하여 RNN Transducer 모델을 학습시켰다. 하지만 그들은 주로 음소(Phoneme) 또는 음절(Syllable) 수준의 레이블을 사용하여 인코더를 학습시킨 반면, 본 논문은 ASR 데이터에는 음소 수준 레이블을, KWS 데이터에는 구절(Phrase) 수준 레이블을 사용하여 구절 단위의 직접적인 예측을 수행함으로써 더 높은 성능을 달성하였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 시스템은 **Phonetic Encoder**와 **Cross Attention Decoder**로 구성되며, 두 모듈은 MTL 프레임워크를 통해 공동 학습된다.

### 훈련 목표 및 손실 함수

모델은 음소 인식 손실($L^{(phone)}$)과 구절 분류 손실($L^{(phrase)}$)을 동시에 최소화하도록 학습된다. 전체 손실 함수 $L$은 다음과 같이 정의된다:

$$L = L^{(phone)} + \alpha L^{(phrase)}$$

여기서 $\alpha$는 두 손실 간의 균형을 맞추기 위한 스케일링 인자이며, $L^{(phone)}$은 ASR 데이터셋에 대해 CTC(Connectionist Temporal Classification) 손실을 사용하고, $L^{(phrase)}$는 KWS 데이터셋에 대해 Cross Entropy 손실을 사용한다.

### Cross Attention Decoder 상세 설계

1. **입력 데이터:** Phonetic Encoder가 입력 오디오 $X$를 처리하여 생성한 은닉 표현 시퀀스 $H = \{h_{n,t} | t=1, \dots, T\}$와 학습 가능한 쿼리 벡터 시퀀스 $Q = \{q_m | m=1, \dots, M\}$를 입력으로 받는다.
2. **Self-Attention:** 먼저 쿼리 시퀀스 $Q$에 대해 Multi-head Attention (MHA)을 수행하여 쿼리 간의 관계를 학습한다:
    $$H_q = \text{MHA}(Q, K, V)$$
3. **Cross-Attention:** Self-attention의 결과물인 $H_q$를 Query로, 인코더의 출력인 $H$를 Key와 Value로 사용하여 Cross Attention을 수행한다. 이때 Attention 연산은 다음과 같다:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
4. **최종 예측:** Transformer 블록($P$번 반복)을 통과한 후, 출력 벡터를 리쉐이핑(Reshape)하여 선형 레이어(Linear Layer)에 입력하고, 최종적으로 키워드 존재 여부에 대한 스칼라 logit을 출력한다.
    - 본 디코더는 Auto-regressive 방식이 아니며, 쿼리 시퀀스의 길이 $M$이 고정되어 있어 효율적인 연산이 가능하다.
    - 쿼리 벡터는 모델 파라미터와 함께 최적화되므로 별도의 Positional Encoding이 필요하지 않다.

## 📊 Results

### 실험 설정

- **데이터셋:**
  - ASR 데이터: 약 300만 개의 발화 $\to$ RIR 및 에코 추가를 통해 900만 개로 증강.
  - KWS 데이터: False Trigger 6.5만 개, True Trigger 30만 개.
- **평가 데이터셋:**
  - Structured set: 통제된 환경에서 수집된 1.3만 개의 positive 샘플과 2,000시간의 negative 샘플.
  - Take-home set: 실제 가정 환경에서 수집된 7,896개의 positive 및 20,919개의 negative 샘플.
- **두 단계 접근법 (Two-stage approach):** 계산 비용 절감을 위해 가벼운 1단계 모델(DNN-HMM)이 후보 구간을 검출하면, 제안하는 무거운 모델(Checker)이 최종 판단을 내리는 구조를 사용하였다.

### 주요 결과

- **정량적 성과:** 제안된 Cross Attention Decoder는 기존의 MTL(분기 방식) 및 BLSTM 디코더와 비교하여 **False Reject Ratio (FRR)를 평균 12% 상대적으로 감소**시켰다.
- **비교 분석 (Table 1 기준):**
  - 단순히 ASR 데이터로만 학습한 Phoneme classifier보다 MTL을 적용했을 때 FRR이 크게 개선되었다.
  - MTL 내에서도 Phonetic branch(음소 기반)보다 Phrase branch(구절 기반)의 성능이 더 높았으며, 그중에서도 Cross Attention Decoder가 가장 우수한 성능을 보였다.
  - Take-home set에서 Cross Attention Decoder의 FRR은 6.00%로, BLSTM(6.83%) 및 Conventional MTL(6.80%)보다 낮게 나타났다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **MTL의 유효성:** 대규모 ASR 데이터와 소규모 KWS 데이터를 함께 학습시키는 것이 모델의 일반화 성능을 크게 향상시킨다는 점을 입증하였다.
- **구조적 효율성:** Cross Attention 기반의 디코더는 BLSTM 디코더보다 학습 시간이 짧고 런타임 비용이 적으면서도 더 높은 정확도를 제공한다.
- **구절 수준 최적화:** 음소 수준의 예측보다 구절 수준의 직접적인 예측을 수행하는 것이 KWS 작업에 훨씬 효과적임을 보여주었다.

### 한계 및 논의

- **분포 불일치 문제:** Structured evaluation set에서는 제안된 방법이 기존 MTL 대비 성능 향상이 뚜렷하지 않았는데, 이는 KWS 학습 데이터와 통제된 환경의 평가 데이터 간의 분포 불일치(Distribution Mismatch) 때문으로 분석된다.
- **고정된 쿼리:** 현재 모델은 특정 키워드에 최적화된 학습 가능한 쿼리를 사용하므로, 새로운 키워드를 추가하려면 모델을 재학습하거나 쿼리를 새로 학습시켜야 하는 제약이 있을 수 있다.

## 📌 TL;DR

본 논문은 ASR 데이터와 KWS 데이터를 동시에 활용하는 Multi-task Learning 프레임워크에 **학습 가능한 쿼리 시퀀스를 이용한 Cross Attention Decoder**를 도입하여 KWS 성능을 높였다. 제안된 방법은 기존의 단순 출력 분기 방식이나 BLSTM 디코더보다 효율적이며, 특히 실제 가정 환경(Take-home set)에서 **FRR을 상대적으로 12% 감소**시키는 성과를 거두었다. 이 연구는 향후 Open Vocabulary KWS로 확장될 가능성이 크며, 온디바이스 음성 인식 시스템의 정확도를 높이는 데 중요한 기여를 할 것으로 보인다.
