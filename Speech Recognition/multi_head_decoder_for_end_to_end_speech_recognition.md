# Multi-Head Decoder for End-to-End Speech Recognition

Tomoki Hayashi, Shinji Watanabe, Tomoki Toda, Kazuya Takeda (2018)

## 🧩 Problem to Solve

본 논문은 End-to-End 음성 인식(Automatic Speech Recognition, ASR) 시스템에서 attention 메커니즘의 효율성을 극대화하여 인식 성능을 향상시키는 것을 목표로 한다.

전통적인 Attention 기반 Encoder-Decoder 모델은 입력 음성 신호와 출력 텍스트 사이의 정렬(alignment)을 유연하게 학습할 수 있다는 장점이 있다. 하지만 Attention 메커니즘이 지나치게 유연하여 발생하는 비인과적 정렬(non-causal alignment) 문제와 같은 한계가 존재한다. 기존의 Multi-Head Attention(MHA) 모델은 여러 개의 attention을 계산한 뒤 이를 하나의 attention으로 통합하여 사용함으로써 다양한 표현 공간의 정보를 동시에 포착하려 했으나, 통합이 attention 레벨에서 이루어진다는 점에 한계가 있다. 따라서 본 연구는 통합 단계를 decoder 레벨로 옮겨 각 head가 서로 다른 모달리티(modality)를 포착하게 함으로써 인식 성능을 높이고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Multi-Head Decoder (MHD)** 및 **Heterogeneous Multi-Head Decoder (HMHD)** 아키텍처의 제안이다.

중심적인 아이디어는 기존 MHA처럼 attention 레벨에서 벡터를 통합하는 것이 아니라, 각 attention head마다 개별적인 decoder를 할당하고 최종 출력 단계에서 이들의 결과를 통합하는 것이다. 이를 통해 각 head가 서로 다른 음성 및 언어적 맥락을 독립적으로 학습하게 하며, 결과적으로 앙상블 효과(ensemble effect)를 통해 전체적인 인식 성능을 향상시킨다. 특히, 각 head에 서로 다른 attention 함수를 적용하는 Heterogeneous 방식을 통해 더욱 다양한 맥락을 포착할 수 있음을 입증하였다.

## 📎 Related Works

본 논문은 End-to-End ASR 접근 방식을 크게 두 가지로 분류하여 설명한다.

첫째는 **CTC(Connectionist Temporal Classification)** 기반 방식이다. 이는 입력과 출력 시퀀스의 길이 차이를 동적 계획법으로 해결하지만, 마르코프 가정(Markov assumption)을 사용하여 각 프레임을 독립적으로 예측하기 때문에 대규모 학습 데이터가 없는 경우 별도의 언어 모델과 그래프 기반 디코딩이 필수적이라는 한계가 있다.

둘째는 **Attention 기반 방식**이다. Encoder-Decoder 구조를 사용하여 입력 특징을 텍스트로 직접 매핑하며, 마르코프 가정이나 외부 언어 모델 없이도 학습이 가능하다. 그러나 앞서 언급한 비인과적 정렬 문제가 발생할 수 있으며, 이를 해결하기 위해 CTC 손실 함수를 결합하거나 MHA를 사용하는 연구들이 진행되었다.

기존의 MHA는 여러 attention head가 서로 다른 위치의 정보를 포착하게 하지만, 결국 이들을 하나의 벡터로 선형 통합하여 하나의 decoder에 입력한다. 반면, 본 논문에서 제안하는 MHD는 통합 시점을 decoder 이후로 늦춤으로써 각 head의 독립성을 더 보장한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

전체 파이프라인은 음성 특징을 입력으로 받아 텍스트 시퀀스를 생성하는 Encoder-Decoder 구조를 따른다. Encoder는 입력 특징 $X$를 프레임별 은닉 상태 $h_t$로 변환하며, 일반적으로 BLSTM(Bidirectional LSTM)을 사용한다. Decoder는 Attention 메커니즘을 통해 $h_t$에서 필요한 정보를 추출하여 다음 문자 $c_l$을 예측한다.

### Multi-Head Decoder (MHD)

MHD는 기존 MHA의 통합 방식을 변경한 구조이다.

1. **Multi-Head Attention (MHA)의 기존 방식**:
   각 head $n$에 대해 attention weight $a_{lt}^{(n)}$와 letter-wise hidden vector $r_l^{(n)}$를 계산한 후, 이를 다음과 같이 선형 통합하여 하나의 벡터 $r_l$을 생성하고 단일 decoder에 입력한다.
   $$r_l = W_O [r^{(1)}_l, r^{(2)}_l, \dots, r^{(N)}_l]^\top$$

2. **Proposed MHD 방식**:
   통합 레벨을 decoder로 옮겨, 각 head $n$이 생성한 $r_l^{(n)}$을 각각 독립적인 $n$번째 decoder LSTM에 입력한다.
   $$q^{(n)}_l = \text{LSTM}^{(n)}(c_{l-1}, q^{(n)}_{l-1}, r^{(n)}_l)$$
   여기서 $q^{(n)}_l$은 각 head의 독립적인 decoder 은닉 상태이다. 최종 출력 확률은 모든 decoder의 출력을 합산하여 계산한다.
   $$p(c_l | c_{1:l-1}, X) = \text{Softmax} \left( \sum_{n=1}^N W^{(n)} q^{(n)}_l + b \right)$$

### Heterogeneous Multi-Head Decoder (HMHD)

HMHD는 MHD의 확장판으로, 각 head에 동일한 attention 함수를 사용하는 대신 서로 다른 attention 함수를 적용한다. 예를 들어, 어떤 head에는 Dot-product attention을, 다른 head에는 Location-based attention이나 Coverage mechanism attention을 적용함으로써, 단일 모델 내에서 다양한 음성/언어적 맥락을 동시에 포착하도록 설계하였다.

### 주요 Attention 함수 설명

본 논문에서 사용된 attention 함수들은 다음과 같다.

- **Dot-Product Attention**: 쿼리와 키의 내적을 통해 에너지를 계산하는 가장 단순한 형태이다.
- **Additive Attention**: 학습 가능한 파라미터를 통해 쿼리와 키의 합을 계산한다.
- **Location Attention**: 이전 시점의 attention weight를 convolution 필터로 처리하여 현재 위치 정보를 반영한다.
- **Coverage Attention**: 지금까지 생성된 attention weight의 누적 합을 이용하여 이미 처리된 부분을 파악한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Corpus of Spontaneous Japanese (CSJ), 학습 데이터 581시간 사용.
- **입력 특징**: 80차원 Log Mel filter bank 및 3차원 Pitch feature.
- **모델 구성**: Encoder는 6층 BLSTMP, Decoder는 1층 LSTM. MHA/MHD의 head 수는 4개로 설정.
- **평가 지표**: CER (Character Error Rate).
- **디코딩**: Beam size 20의 Beam search 사용.

### 정량적 결과

실험 결과, 제안된 HMHD 방식이 모든 테스트 세트에서 가장 낮은 CER을 기록하며 성능 우위를 보였다.

| 모델 | Task 1 (CER %) | Task 2 (CER %) | Task 3 (CER %) |
| :--- | :---: | :---: | :---: |
| Dot | 12.7 | 9.8 | 10.7 |
| Loc | 11.7 | 8.8 | 10.2 |
| MHA-Loc | 11.5 | 8.6 | 9.0 |
| MHD-Loc | 11.0 | 8.4 | 9.5 |
| **HMHD (2$\times$Loc + 2$\times$Cov)** | **10.4** | **7.7** | **8.9** |

특히, 단순한 MHD-Loc보다 서로 다른 attention을 섞어 사용한 HMHD의 성능이 더 높게 나타났다.

### 정성적 결과

HMHD (2$\times$Loc + 2$\times$Cov)의 attention weight를 시각화한 결과, 일부 head는 일반적인 attention과 유사하게 동작하는 반면, 다른 head는 음성의 더 추상적인 다이내믹스를 포착하는 경향을 보였다. 이는 서로 다른 attention 함수를 사용하는 것이 실제로 다양한 맥락을 포착하는 데 기여한다는 가설을 뒷받침한다.

## 🧠 Insights & Discussion

본 연구의 강점은 attention의 통합 시점을 변경함으로써 각 head의 전문성을 극대화했다는 점이다. 기존 MHA가 특징 공간(feature space)에서의 통합에 집중했다면, MHD는 결정 단계(decision level)에서의 앙상블을 구현하여 더 강력한 표현력을 확보하였다.

실험을 통해 일본어 문장은 영어보다 길이가 짧아 Location-based attention의 효과가 상대적으로 적을 수 있음이 확인되었으나, Multi-head 구조를 도입하면 전반적인 성능이 향상됨을 알 수 있다. 특히 Heterogeneous 구성을 통해 서로 다른 성격의 attention(예: Location과 Coverage)을 결합했을 때 시너지 효과가 가장 컸다는 점은 매우 유의미하다.

다만, 단순 MHD-Loc 모델이 일부 Task에서 성능 저하를 보였다는 점은, 단순히 head 수를 늘리는 것보다 각 head가 서로 다른 역할을 수행하도록 유도하는 설계(Heterogeneity)가 필수적임을 시사한다.

## 📌 TL;DR

본 논문은 End-to-End 음성 인식에서 attention head들을 통합하는 시점을 attention 레벨에서 decoder 출력 레벨로 변경한 **Multi-Head Decoder (MHD)** 구조를 제안하였다. 더 나아가 각 head에 서로 다른 attention 함수를 적용한 **Heterogeneous Multi-Head Decoder (HMHD)**를 통해 인식 성능(CER)을 유의미하게 향상시켰다. 이 연구는 다양한 attention 메커니즘을 병렬적으로 배치하여 음성 신호의 다각적인 맥락을 포착하는 앙상블 기법의 유효성을 입증하였으며, 향후 Joint CTC/Attention 구조와의 결합을 통해 더욱 발전할 가능성이 크다.
