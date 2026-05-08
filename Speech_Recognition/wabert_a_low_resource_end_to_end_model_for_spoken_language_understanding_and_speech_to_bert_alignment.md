# WaBERT: A Low-resource End-to-end Model for Spoken Language Understanding and Speech-to-BERT Alignment

Lin Yao, Jianfei Song, Ruizhuo Xu, Yingfang Yang, Zijian Chen and Yafeng Deng (2022)

## 🧩 Problem to Solve

본 논문은 구어 자연어 이해(Spoken Language Understanding, SLU) task, 특히 감성 분석(Sentiment Analysis, SA)의 성능을 향상시키는 것을 목표로 한다. 기존의 SLU 접근 방식은 크게 두 가지로 나뉘는데, 각각 명확한 한계를 가지고 있다.

첫째, 2단계 방식(Two-stage method)은 자동 음성 인식(Automatic Speech Recognition, ASR)을 통해 음성을 텍스트로 변환한 후, 언어 모델(Language Model)을 통해 하위 task를 수행한다. 이 방식은 ASR 과정에서 발생하는 인식 오류가 최종 결과에 전이되는 '오류 전파' 문제가 있으며, 억양이나 톤과 같은 음성 고유의 정서적 단서(emotional cues)가 소실된다는 단점이 있다.

둘째, 1단계 방식(One-stage method)은 사전 학습된 음성 모델을 직접 하위 task에 맞게 미세 조정(fine-tuning)한다. 그러나 음성 데이터는 텍스트 데이터에 비해 가용량이 매우 적어 대규모 언어 모델 수준의 풍부한 언어적 지식을 학습하기 어려우며, 사전 학습된 음성 모델들이 주로 ASR과 같은 저수준(lower-level) task에 최적화되어 있어 고수준의 의미론적 정보(semantic information)가 부족하다는 한계가 있다.

결과적으로 본 논문은 음성 모델의 오디오 특성 정보와 언어 모델의 풍부한 의미론적 지식을 모두 활용할 수 있는 효율적인 end-to-end 모델을 구축하여 SLU 성능을 높이고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 학습된 음성 인코더와 언어 인코더를 결합하여, 음성 신호를 언어 모델의 중간 레이어 표현으로 직접 정렬(alignment)시키는 것이다. 주요 기여 사항은 다음과 같다.

1. **WaBERT 모델 제안**: 음성 모델과 언어 모델을 결합한 새로운 end-to-end 모델을 통해 감성 분석(SA) 성능을 개선하였다.
2. **개선된 CIF Aligner**: 음성과 텍스트라는 서로 다른 모달리티 간의 단조 정렬(monotonic alignment)을 달성하기 위해 Continuous Integrate-and-Fire (CIF) 메커니즘을 수정하고, 새로운 정렬 손실 함수를 도입하였다.
3. **저자원 효율적 학습**: 모델 전체를 처음부터 학습시키는 대신, 사전 학습된 모델들을 사용하고 대부분의 파라미터를 고정(frozen)함으로써 학습 시간과 자원 소모를 최소화하였다.

## 📎 Related Works

### 사전 학습된 음성 및 언어 모델

최근 wav2vec 2.0, HuBERT, data2vec와 같은 자기지도학습(Self-Supervised Learning, SSL) 기반의 음성 모델들이 ASR 등에서 뛰어난 성능을 보였으나, 앞서 언급했듯 고수준 SLU task에서는 의미론적 정보 부족과 데이터 희소성 문제로 인해 한계가 있다. 반면, BERT, RoBERTa와 같은 NLP 모델들은 방대한 텍스트 데이터를 통해 강력한 성능을 보이지만, 오디오 입력 데이터를 직접 처리할 수 없다는 구조적 차이가 있다.

### 강제 정렬(Forced Alignment) 전략

텍스트와 음성 시퀀스 간의 양방향 매핑을 생성하는 강제 정렬 기술은 HMM 기반의 MFA나 신경망 기반의 NeuFA, CIF 등이 있다. 하지만 HMM이나 CTC 기반 방식은 상태(state)가 이산적(discrete)이어야 한다는 제약이 있어, 연속적인 텐서를 다루는 모달리티 간 정렬에는 부적합하다. NeuFA는 ASR과 TTS의 공동 학습이 필요하여 가볍지 않다는 단점이 있다. 반면 CIF는 입력값의 이산성 제약이 없어 모달리티 정렬에 잠재력이 있으며, 본 논문은 이를 발전시켜 사용한다.

## 🛠️ Methodology

### 전체 시스템 구조

WaBERT는 사전 학습된 **Wave Encoder(data2vec)**와 **NLP Encoder(BERT)**를 기반으로 하며, 그 사이에 **CIF Aligner**를 배치하여 두 모달리티를 연결한다. 전체적인 흐름은 오디오 입력 $\rightarrow$ data2vec $\rightarrow$ CIF Aligner $\rightarrow$ BERT (상위 레이어) $\rightarrow$ 최종 분류기로 이어진다.

### 모달리티 간 정렬 (Alignment)

음성 표현 벡터 $A = (a_1, a_2, \dots, a_M)$와 BERT의 $i$번째 레이어 출력인 언어 표현 벡터 $L^i = (l^i_1, l^i_2, \dots, l^i_N)$가 있을 때, $M > N$인 특성을 가진다. 이를 정렬하기 위해 CIF 메커니즘을 사용하여 $A$를 $L^i$와 동일한 길이 $N$을 가진 $\hat{A} = (\hat{a}_1, \hat{a}_2, \dots, \hat{a}_N)$로 리사이징한다.

이때, 단순히 코사인 유사도만을 사용하는 기존 방식($L_{cos}$)은 서로 다른 토큰 간의 변별력을 무시하는 문제가 있다. 이를 해결하기 위해 본 논문은 **InfoNCE (Information Noise Contrastive Estimation)** 손실 함수를 도입하여 정렬의 정밀도를 높였다.

$$ \text{InfoNCE}(X, Y) = \frac{1}{N} \sum_{i=0}^{N} \left( -\log \frac{\exp(\cos(x_i, y_i)/\tau)}{\sum_{j=0}^{N} \exp(\cos(x_i, y_j)/\tau)} \right) $$

최종 정렬 손실 함수 $L_{InfoNCE}$는 음성에서 언어로, 언어에서 음성으로의 양방향 유사도를 모두 고려하여 계산된다.

$$ L_{InfoNCE} = \frac{1}{2} \text{InfoNCE}(\hat{A}, L) + \frac{1}{2} \text{InfoNCE}(L, \hat{A}) $$

또한, 예측된 길이와 실제 길이의 차이를 줄이기 위한 길이 손실 함수 $L_{quantity} = N_{predicted} - N$을 함께 사용한다.

### 모델 이식 (Grafting)

정렬이 완료되면, BERT의 하위 레이어(1층부터 $i$층까지)를 data2vec 모델 및 CIF Aligner로 대체하는 'Grafting'을 수행한다. 이렇게 하면 BERT의 상위 레이어는 텍스트 입력 대신 정렬된 음성 표현을 입력으로 받게 되며, BERT의 파라미터를 고정한 상태에서 하위 단어(sub-word) 예측 손실 $L_{subword}$를 통해 학습한다.

### 학습 및 추론 절차

전체 학습 목표 함수는 다음과 같이 정의된다.

$$ L_{total} = L_{InfoNCE} + L_{quantity} + L_{subword} $$

학습 시 BERT의 파라미터는 고정하고, LibriSpeech ASR 코퍼스를 사용하여 학습을 진행한다. 추론 시에는 BERT의 하위 레이어를 제거하고 data2vec와 CIF Aligner가 그 역할을 대신하며, 최종적으로 downstream task(예: SA)를 위한 미세 조정이 이루어진다.

## 📊 Results

### CIF 정렬 성능 평가

TIMIT 데이터셋을 사용하여 정렬 정확도를 측정한 결과, 제안한 $L_{InfoNCE}$ 방식이 기존의 $L_{cos}$ 방식보다 월등히 뛰어난 성능을 보였다.

- **MAE (Mean Absolute Error)**: $L_{cos}$의 426.44ms에서 $L_{InfoNCE}$의 126.05ms로 크게 감소하였다.
- **정확도 (Tolerance별)**: 100ms 오차 범위 내에서의 정확도가 0.0559에서 0.6091로 비약적으로 상승하였다.
- **시각화**: 유사도 히트맵 분석 결과, $L_{InfoNCE}$는 명확한 대각선 형태를 보이며 정확한 정렬이 이루어짐을 증명하였다.

### SLUE SA Task 성능

감성 분석 작업에서 WaBERT는 기존의 2단계(Pipeline) 및 1단계(E2E) 방식보다 우수한 성능을 보였다.

- **vs 2단계 방식**: ASR 오류의 영향을 받지 않으므로, Ground Truth 텍스트를 사용한 BERT 모델보다도 더 높은 성능을 기록하였다. (Recall score 4.39% $\uparrow$, F1 score 1.12% $\uparrow$)
- **vs 1단계 방식**: 순수 음성 모델만 사용한 E2E 접근법보다 Recall 1.19%, F1 0.72% 향상된 결과를 보였다. 이는 NLP 인코더의 결합이 구어 이해 능력을 실질적으로 높였음을 의미한다.

### Ablation Study

BERT의 어느 레이어에 정렬시킬 것인가에 대한 실험 결과, 3번째 레이어와 12번째 레이어에 정렬했을 때 성능이 좋았다. 특히 3번째 레이어(임베딩 층에 가까움)에 정렬했을 때 가장 높은 성능을 보였는데, 이는 더 많은 BERT의 구조적 이점을 활용할 수 있기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 연구는 음성과 텍스트라는 서로 다른 모달리티를 정렬하기 위해 CIF 메커니즘과 InfoNCE 손실 함수를 결합함으로써, 특징 분포(feature distribution)와 공간 분포(spatial distribution)의 불일치 문제를 효과적으로 해결하였다.

특히 주목할 점은, WaBERT가 ASR의 전사(transcription) 과정 없이도 NLP 모델이 이해할 수 있는 수준의 특징을 생성한다는 것이다. 논문의 에러 분석(Table 4)에 따르면, 비록 ASR 관점에서는 오타가 발생하더라도, 생성된 특징 벡터는 NLP 모델이 정답을 도출하기에 충분한 정보를 담고 있었다. 이는 텍스트로의 완전한 변환보다 '의미적 정렬'이 SLU task에 더 중요하다는 통찰을 제공한다.

다만, 본 논문은 BERT와 data2vec라는 특정 조합만을 실험하였으며, 대부분의 파라미터를 고정하여 학습하였다. 모든 파라미터를 학습 가능하게 설정하거나, 여러 레이어를 동시에 정렬하는 방식 등을 도입한다면 추가적인 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

WaBERT는 사전 학습된 음성 인코더(data2vec)와 언어 모델(BERT)을 CIF Aligner를 통해 결합한 end-to-end SLU 모델이다. 특히 InfoNCE 기반의 새로운 정렬 손실 함수를 도입하여 음성-텍스트 간의 정밀한 단조 정렬을 구현하였으며, 이를 통해 ASR 오류 전파 문제를 해결하고 저자원 환경에서도 높은 감성 분석 성능을 달성하였다. 이 연구는 향후 음성-언어 통합 모델 설계 및 고수준 구어 이해 연구에 중요한 방법론적 기초를 제공한다.
