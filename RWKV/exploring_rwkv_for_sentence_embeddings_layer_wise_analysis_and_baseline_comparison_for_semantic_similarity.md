# Exploring RWKV for Sentence Embeddings: Layer-wise Analysis and Baseline Comparison for Semantic Similarity

Xinghan Pan (2025)

## 🧩 Problem to Solve

본 논문은 문장의 의미적 의미를 벡터로 표현하는 Sentence Embeddings 생성에 있어, 선형 어텐션(Linear Attention) 메커니즘을 가진 RWKV 아키텍처의 효용성을 탐구한다. 기존의 Transformer 기반 모델(예: BERT, Sentence-BERT)은 고품질의 문장 임베딩을 생성하여 다양한 NLP 작업에서 우수한 성능을 보였으나, 어텐션 메커니즘의 연산 복잡도가 시퀀스 길이의 제곱에 비례하는 $O(n^2)$ 수준이라는 치명적인 단점이 있다. 이는 특히 긴 시퀀스를 처리할 때 계산 비용을 급격히 증가시켜 자원이 제한된 환경에서의 확장성과 효율성을 저해한다.

반면 RWKV는 RNN과 Transformer의 장점을 결합하여 선형 시간 복잡도 $O(n)$를 달성함으로써 계산 효율성을 획기적으로 높인 모델이다. 하지만 RWKV가 언어 모델링 작업에서는 뛰어난 성능을 보였음에도 불구하고, 별도의 미세 조정이 없는 Zero-shot 설정에서 문장 임베딩을 생성하고 이를 통해 의미적 유사도(Semantic Similarity)를 측정하는 능력에 대해서는 아직 충분히 연구되지 않았다. 따라서 본 논문의 목표는 사전 학습된 RWKV 모델의 서로 다른 레이어에서 추출한 임베딩이 의미적 유사도를 얼마나 잘 포착하는지 분석하고, 이를 전통적인 baseline과 비교하여 RWKV의 잠재력과 한계를 규명하는 것이다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 사전 학습된 RWKV 모델을 문장 임베딩 생성에 적용하여 그 성능을 정량적으로 분석하고, 아키텍처 관점에서의 이론적 해석을 제공했다는 점이다. 구체적인 설계 아이디어 및 기여 사항은 다음과 같다.

- **레이어별 분석(Layer-wise Analysis):** RWKV 모델의 다양한 은닉 레이어(Hidden Layer)에서 임베딩을 추출하여, 어떤 깊이의 레이어가 문장의 의미적 유사성을 가장 잘 포착하는지 분석하였다.
- **Zero-shot 성능 평가:** 미세 조정 없이 사전 학습된 모델만을 사용하여 MRPC 데이터셋에서 Spearman 상관계수를 통해 성능을 측정하고, 이를 GloVe 기반의 단순 평균 임베딩 baseline과 비교하였다.
- **계산 효율성 측정:** 추론 시간(Inference Time)과 GPU 메모리 사용량을 측정하여 RWKV의 선형 복잡도가 실제 문장 임베딩 생성 단계에서 어떤 계산적 이점과 트레이드오프를 가지는지 분석하였다.
- **이론적 분석 제공:** Linear Attention의 수학적 구조, 정보 전파 특성, 엔트로피 진화 및 풀링 전략의 한계를 정보 이론적 관점에서 분석하여 성능 저하의 원인을 고찰하였다.

## 📎 Related Works

문장 임베딩 연구는 크게 세 단계의 발전 과정을 거쳤다. 초기에는 Word2Vec나 GloVe와 같은 사전 학습된 단어 벡터들의 단순 평균을 사용하는 방식이 쓰였으나, 이는 단어 순서나 복잡한 문장 구조를 반영하지 못한다는 한계가 있었다. 이후 RNN 기반의 Skip-Thought나 FastSent가 등장하여 문장 수준의 의미를 포착하기 시작했다.

가장 큰 도약은 BERT와 같은 Transformer 모델의 등장이었으며, 특히 Sentence-BERT(SBERT)는 Siamese 네트워크 구조를 통해 코사인 유사도로 빠르게 비교 가능한 고품질 임베딩을 생성하며 SOTA 성능을 달성했다. 그러나 이러한 Transformer 모델들은 앞서 언급한 $O(n^2)$의 연산 복잡도로 인해 긴 시퀀스 처리 시 병목 현상이 발생한다.

RWKV는 이러한 Transformer의 효율성 문제를 해결하기 위해 제안된 모델로, RNN처럼 선형적인 스케일링을 지원하면서도 Transformer에 근접한 언어 모델링 성능을 보여준다. 본 논문은 기존 연구들이 주로 언어 생성(Generation)에 집중했던 것과 달리, RWKV를 문장 표현 학습(Representation Learning) 영역으로 확장하여 그 가능성을 타진한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

본 연구는 사전 학습된 `RWKV-v6-Finch-1B6-HF` 모델을 사용한다. 전체적인 분석 파이프라인은 다음과 같다.

1. **임베딩 추출:** RWKV 모델의 특정 레이어($1, 3, 5, 7, 9, 11$)에서 은닉 상태(Hidden States)를 추출한다.
2. **풀링(Pooling):** 추출된 토큰 수준의 벡터들을 문장 전체에 대해 단순 평균(Average Pooling)하여 하나의 문장 벡터를 생성한다.
3. **유사도 계산:** 두 문장 벡터 간의 코사인 유사도(Cosine Similarity)를 계산한다.
4. **상관관계 분석:** 계산된 유사도 점수와 MRPC 데이터셋의 실제 정답 레이블(Paraphrase 여부) 간의 Spearman 상관계수를 측정한다.

### 핵심 방정식 및 메커니즘

**1. RWKV의 Linear Attention:**
전통적인 Transformer의 어텐션이 $O(n^2)$인 것과 달리, RWKV는 다음과 같은 재귀적 메커니즘을 통해 복잡도를 $O(nd)$로 낮춘다.

$$o_t = \frac{\sum_{i=1}^{t-1} e^{w_{t-i}}(k_i v_i) + e^u k_t v_t}{\sum_{i=1}^{t-1} e^{w_{t-i}} r_i + e^u r_t}$$

여기서 $w_{t-i}$는 학습 가능한 상대적 위치 가중치이며, $r_t = \sigma(W_r x_t)$는 게이팅 함수이다. 이는 누적 합(cumulative terms)을 캐싱함으로써 선형 시간 내에 계산이 가능하다.

**2. 평가 지표 (Spearman Correlation):**
단순한 정확도 대신 rank correlation을 측정하는 Spearman 상관계수를 사용하여, 모델이 예측한 유사도 순위가 인간이 판단한 유사도 순위와 얼마나 일치하는지를 평가한다.

**3. Baseline (GloVe):**
비교 대상인 GloVe는 50차원의 사전 학습된 단어 벡터를 사용하며, 문장 내 모든 단어 벡터의 단순 평균을 통해 문장 임베딩을 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋:** MRPC (Microsoft Research Paraphrase Corpus). 훈련 세트 중 1,000개 샘플과 전체 검증 세트(408개)를 사용하였다.
- **하드웨어:** Google Colab Tesla T4 GPU.
- **지표:** Spearman correlation, 추론 시간, 피크 GPU 메모리 사용량.

### 정량적 결과

**1. 의미적 유사도 성능 (Spearman Correlation):**

- **GloVe Baseline**이 검증 세트에서 $0.4326$으로 가장 높은 성능을 보였다.
- **RWKV**는 레이어 1이 $0.3498$로 가장 높았으며, 레이어 깊이가 깊어질수록 성능이 감소하는 경향($\text{Layer 1} \rightarrow \text{Layer 11}$)을 보였다.
- 결과적으로 Zero-shot 설정에서 RWKV는 단순한 GloVe baseline보다 낮은 성능을 기록하였다.

**2. 계산 효율성:**

- **추론 시간:** RWKV(마지막 레이어)의 추론 시간은 훈련 세트 기준 $0.4141$초로, GloVe($0.0006$초)보다 약 2~3 orders of magnitude 더 느렸다.
- **메모리 사용량:** 두 모델 모두 약 $3,000\text{MB}$ 수준의 피크 메모리를 사용하여 큰 차이가 없었다.

### 정성적 분석

분석 결과, RWKV와 GloVe 모두 실제로는 paraphrase가 아닌 문장 쌍에 대해서도 높은 코사인 유사도 점수를 부여하는 경향이 있었다. 이는 두 방식 모두 일반적인 의미적 연관성은 포착하지만, paraphrase 판별에 필요한 미세한 의미적 차이(subtle semantic nuances)를 구분하는 능력은 부족함을 시사한다.

## 🧠 Insights & Discussion

### 분석 및 강점

본 연구는 RWKV의 구조적 특성이 문장 임베딩에 미치는 영향을 이론적으로 고찰하였다. 특히 **엔트로피(Entropy) 분석**을 통해 얕은 레이어($S(H_3) \approx 5.2$)보다 깊은 레이어($S(H_{12}) \approx 3.8$)의 엔트로피가 낮음을 발견했다. 이는 깊은 레이어로 갈수록 더 추상적이고 변동성이 적은 특징을 추출하지만, 동시에 너무 낮은 엔트로피는 정보 손실을 초래하여 Zero-shot 유사도 측정 성능을 떨어뜨릴 수 있음을 의미한다.

### 한계 및 비판적 해석

1. **풀링 전략의 한계:** 본 논문은 단순 평균 풀링(Average Pooling)을 사용하였다. 이론적 분석에서 저자는 신호 대 잡음비(SNR) 관점에서 단순 평균은 중요 정보가 적은 토큰들에 의해 신호가 희석될 수 있음을 지적하며, 학습 가능한 쿼리 벡터를 사용하는 **Adaptive Pooling** $\left(h_{\text{pool}} = \sum \alpha_i h_i\right)$의 필요성을 제시하였다.
2. **Zero-shot의 한계:** RWKV는 기본적으로 다음 토큰 예측(Language Modeling)을 위해 학습된 모델이다. 따라서 문장 간의 유사도를 비교하는 Task에 맞게 정렬(Alignment)되지 않은 상태이므로, 단순한 GloVe보다 성능이 낮게 나오는 것은 당연한 결과일 수 있다.
3. **효율성의 역설:** 이론적으로는 $O(n)$의 복잡도를 가져 효율적이어야 하지만, 실제 짧은 문장을 처리하는 MRPC 데이터셋에서는 단순 벡터 룩업 및 평균을 수행하는 GloVe보다 추론 시간이 훨씬 느렸다. 이는 선형 어텐션의 이점이 매우 긴 시퀀스에서만 실현되며, 작은 규모의 작업에서는 모델 자체의 오버헤드가 더 크다는 것을 보여준다.

## 📌 TL;DR

본 논문은 효율적인 선형 어텐션 구조를 가진 **RWKV 모델을 Zero-shot 문장 임베딩 생성에 적용**하여 분석하였다. 실험 결과, **RWKV는 단순한 GloVe baseline보다 성능이 낮았으며, 특히 레이어가 깊어질수록 유사도 측정 능력이 떨어지는 경향**을 보였다. 또한 이론적 분석을 통해 단순 평균 풀링의 한계와 깊은 레이어에서의 정보 손실 가능성을 제기하였다. 결론적으로 RWKV가 실무적인 문장 임베딩 모델로 활용되기 위해서는 **Task-specific fine-tuning(특히 Contrastive Learning)과 고도화된 Adaptive Pooling 전략이 필수적**임을 시사한다.
