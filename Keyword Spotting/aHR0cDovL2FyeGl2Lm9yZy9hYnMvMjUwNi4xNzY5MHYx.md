# Low-resource keyword spotting using contrastively trained transformer acoustic word embeddings

Julian Herreilers, Christiaan Jacobs, Thomas Niesler (2025)

## 🧩 Problem to Solve

본 연구는 자원이 매우 부족한(low-resource) 언어 환경에서 효율적인 키워드 스포팅(Keyword Spotting, KWS) 시스템을 구축하는 것을 목표로 한다. 특히 인도주의적 구호 활동을 위해 라디오 방송을 모니터링해야 하는 상황에서, 인터넷 인프라가 부족하고 대상 언어의 학습 데이터가 거의 없는 환경을 상정한다.

일반적인 KWS 시스템은 자동 음성 인식(Automatic Speech Recognition, ASR)을 거쳐 텍스트 기반 검색을 수행하지만, 저자원 언어의 경우 ASR 모델을 학습시킬 충분한 데이터가 없으므로 이 방식은 불가능하다. 따라서 본 논문은 단일 음성 템플릿만으로 검색이 가능한 Query-by-Example (QbE) 방식을 채택하여, 데이터 부족 문제를 해결하고 빠르게 배포 가능한 KWS 시스템을 개발하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **ContrastiveTransformer**라는 새로운 아키텍처를 제안하여 고성능의 Acoustic Word Embeddings (AWEs)를 생성한 것이다. 주요 설계 아이디어는 다음과 같다.

1.  **Contrastive Learning의 도입**: 기존의 재구성(reconstruction) 기반 방식 대신, 임베딩 공간에서 동일한 단어는 가깝게, 서로 다른 단어는 멀게 배치하는 대비 학습(Contrastive Learning)을 Transformer Encoder에 적용하였다.
2.  **Fixed-dimensional Representation 추출**: 가변 길이의 음성 시퀀스를 고정 차원 벡터로 변환하기 위해, 입력 시퀀스 앞에 학습 가능한 벡터(trainable vector of ones)를 추가하고 최종 레이어의 첫 번째 출력 벡터를 AWE로 사용하는 방식을 제안하였다.
3.  **저자원 언어 KWS 성능 향상**: 매우 적은 양의 키워드 템플릿만으로도 Luganda 및 Bambara와 같은 극저자원 언어에서 기존 DTW 및 RNN 기반 임베딩 방식보다 뛰어난 성능을 입증하였다.

## 📎 Related Works

논문에서는 기존의 QbE 및 AWE 접근 방식들의 한계를 다음과 같이 설명한다.

-   **Dynamic Time Warping (DTW)**: 전통적인 방식이지만 계산 비용이 매우 높아 대규모 검색에 부적합하다.
-   **CAE-RNN (Correspondence Autoencoder RNN)**: 인코더-디코더 구조를 통해 한 단어의 인스턴스를 다른 인스턴스로 재구성하도록 학습한다. 하지만 본 논문에서는 이러한 재구성 기반 방식보다 대비 학습 기반 방식이 더 효과적임을 보여준다.
-   **Self-Supervised Learning (SSL) Features**: wav2vec 2.0이나 HuBERT와 같은 대규모 사전 학습 모델의 특징(feature)을 단순 평균(meanpooling)하거나 일부를 추출(subsampling)하여 사용하는 방식이다. 이는 데이터가 전혀 없는 환경에서 유망하지만, 본 연구의 제안 모델보다 성능이 낮게 나타났다.
-   **ContrastiveRNN**: RNN 인코더에 대비 학습을 적용한 방식이다. 본 연구는 이를 Transformer 구조로 확장하여 성능을 더욱 개선하였다.

## 🛠️ Methodology

### 전체 시스템 구조
제안된 시스템은 **ContrastiveTransformer**를 통해 음성 세그먼트를 고정 차원의 벡터인 AWE로 변환하고, 쿼리 템플릿과 검색 대상 세그먼트 간의 코사인 유사도를 측정하여 키워드를 탐색하는 구조이다.

### 상세 구성 요소 및 학습 절차
1.  **아키텍처**: 3개 층의 Transformer Encoder를 사용하며, 각 층은 16개의 Attention Head를 가진다. 최종적으로 Linear Layer를 통해 256차원의 AWE를 생성한다.
2.  **입력 처리**: 가변 길이 시퀀스 $X$ 앞에 학습 가능한 벡터(vector of ones)를 추가하여 입력한다. Transformer의 최종 레이어에서 출력된 첫 번째 벡터 $w$를 해당 단어의 AWE로 정의한다.
3.  **손실 함수 (NT-Xent Loss)**:
    임베딩 공간에서 긍정 쌍(positive pair, 동일 단어의 다른 발화) 간의 거리는 좁히고, 부정 쌍(negative pair, 서로 다른 단어) 간의 거리는 넓히기 위해 Normalized Temperature-scaled Cross Entropy (NT-Xent) 손실 함수를 사용한다.
    $$L_i = -\log \frac{\exp(\text{sim}(w^{(a,i)}, w^{(p,i)})/\tau)}{\sum_{\forall w \in W} \exp(\text{sim}(w^{(a,i)}, w)/\tau)}$$
    여기서 $\text{sim}(\cdot)$은 코사인 유사도, $\tau$는 온도 계수이며, $w^{(a,i)}$는 앵커, $w^{(p,i)}$는 긍정 쌍의 임베딩이다. $W$는 배치 내의 모든 임베딩 집합이다.
4.  **추론 절차 (KWS)**:
    -   검색 대상 오디오에 가변 길이 윈도우(10~65 frames)를 슬라이딩하며 각 세그먼트의 AWE를 추출한다.
    -   추출된 AWE와 키워드 템플릿의 AWE 간 코사인 유사도를 계산한다.
    -   설정된 임계값(threshold)을 적용하여 키워드 존재 여부를 결정한다.

### 학습 데이터 및 전략
-   **다국어 학습 (Multilingual Training)**: 대상 언어(Luganda, Bambara)의 데이터가 부족하므로, NCHLT 데이터셋의 남아프리카 공화국 11개 언어와 Swahili를 사용하여 모델을 사전 학습시킨 후 대상 언어에 적용하는 전이 학습 방식을 사용하였다.
-   **특징 추출**: mHuBERT-147 모델의 10번째 레이어 출력값을 입력 특징으로 사용하였다.

## 📊 Results

### 실험 설정
-   **대상 언어**: English (개발용), Luganda (저자원), Bambara (극저자원).
-   **비교 대상**: DTW with BNF (Baseline), mHuBERT (Meanpooling/Subsampling), CAE-RNN, ContrastiveRNN.
-   **평가 지표**: Mean Average Precision (MAP), Precision at 10 (P@10), Precision at N (P@N).

### 주요 결과
1.  **정량적 성능**:
    -   개발 세트 결과(Table 3)에서 **ContrastiveTransformer**가 모든 언어에서 가장 높은 MAP를 기록하였다.
    -   특히 저자원 언어인 Luganda와 Bambara에서 DTW baseline 대비 MAP가 각각 47%, 34% 절대적으로 향상되었다.
    -   테스트 세트(Table 4)에서도 ContrastiveTransformer가 ContrastiveRNN보다 우수한 성능을 보였으며, 특히 P@N 지표에서 더 일관된 검색 능력을 입증하였다.

2.  **SSL 모델 분석**:
    -   단순 특징 추출 방식 중에서는 mHuBERT의 Layer 8에서 Meanpooling을 적용했을 때 가장 좋은 성능이 나왔으며, 이는 DTW보다 우수하였다. 하지만 대비 학습을 적용한 AWE 모델들이 이를 다시 상회하였다.

3.  **언어 조합 분석**:
    -   대상 언어와 계통적으로 가까운 언어(예: Luganda를 위해 Swahili)로 학습했을 때 성능이 가장 좋았다.
    -   관련 언어가 없는 경우(Bambara), 유사한 언어 여러 개를 쓰는 것보다 서로 다른 특성을 가진 다양한 언어(예: sw+afr)를 조합하는 것이 더 유리함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 유효성
본 연구는 대규모 사전 학습 모델(mHuBERT)의 특징 추출 능력과 Transformer의 문맥 파악 능력, 그리고 대비 학습의 변별력을 결합하여 극저자원 환경에서도 작동하는 KWS 시스템을 구축하였다. 특히 ASR 없이 소수의 템플릿만으로 높은 정밀도를 달성했다는 점이 실용적 가치가 높다.

### 한계 및 논의사항
-   **언어 의존성**: 관련 언어가 있을 때는 성능 향상이 뚜렷하지만, 관련 언어가 전혀 없는 경우(Bambara)에는 상대적으로 성능 향상 폭이 낮다.
-   **가정**: 본 연구는 고정된 윈도우 크기를 통해 세그먼트를 나누는 방식을 사용하는데, 실제 발화의 속도 변화나 경계 모호성 문제를 완전히 해결했는지는 명시되지 않았다.
-   **비판적 해석**: 영어와 같은 고자원 언어에서는 mHuBERT 기반의 단순 접근법과 제안 모델 간의 차이가 크지 않은데, 이는 이미 사전 학습 모델이 영어에 대해 충분한 정보를 가지고 있기 때문으로 풀이된다. 따라서 본 모델의 진정한 가치는 '데이터가 극도로 부족한 언어'에서 발휘된다.

## 📌 TL;DR

본 논문은 극저자원 언어의 키워드 스포팅을 위해 **ContrastiveTransformer** 기반의 Acoustic Word Embeddings(AWE) 생성 방식을 제안한다. 이 모델은 mHuBERT의 특징을 입력으로 받아 NT-Xent 손실 함수를 통해 대비 학습을 수행하며, 가변 길이 음성을 고정 차원 벡터로 효율적으로 매핑한다. 실험 결과, 제안 방식은 기존의 DTW 및 RNN 기반 AWE 모델보다 뛰어난 성능을 보였으며, 특히 관련 언어를 활용한 다국어 학습이 저자원 언어의 KWS 성능을 크게 향상시킴을 입증하였다. 이는 데이터가 부족한 지역의 라디오 방송 모니터링 등 실제 인도주의적 구호 활동에 즉각적으로 적용될 가능성이 높다.