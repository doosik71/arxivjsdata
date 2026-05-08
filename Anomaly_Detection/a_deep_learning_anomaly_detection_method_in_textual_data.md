# A Deep Learning Anomaly Detection Method in Textual Data

Amir Jafari (2022)

## 🧩 Problem to Solve

본 논문은 텍스트 데이터 내에서 일반적인 패턴과 다른 이상치(Anomaly) 또는 특이값(Outlier)을 탐지하는 문제를 해결하고자 한다. 텍스트 마이닝에서 이상한 감성 패턴이나 고유한 텍스트 특성을 식별하는 것은 매우 중요하다. 만약 이러한 이상치가 적절히 처리되지 않은 채 텍스트 분류 시스템에 입력될 경우, 분류 성능의 심각한 저하를 초래할 수 있기 때문이다.

따라서 본 연구의 목표는 딥러닝 기반의 Transformer 아키텍처와 고전적인 머신러닝 알고리즘을 결합하여, 텍스트 데이터의 맥락(Context) 정보를 효과적으로 수치화하고 이를 통해 이상치를 정밀하게 탐지하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer 모델을 통해 텍스트의 풍부한 의미론적 정보를 추출하고, 이를 Convolutional Autoencoder(AE)를 통해 압축함으로써 정상 데이터의 특징을 학습하는 것이다.

특히, 단순히 Autoencoder의 재구성 오차(Reconstruction Error)만을 사용하는 것이 아니라, AE가 학습한 저차원의 잠재 벡터(Latent Vector)와 재구성 오차를 함께 Logistic Regression 분류기에 입력으로 제공함으로써 이상치 판별 능력을 극대화하였다. 이는 텍스트의 고차원 문맥 정보를 효율적으로 압축함과 동시에, 정상 데이터의 분포에서 벗어난 정도를 수치적으로 결합하여 판단하는 구조이다.

## 📎 Related Works

논문에서는 기존의 이상치 탐지 방식과 그 한계를 다음과 같이 설명한다.

1. **거리 기반 접근 방식 (Distance-based approach):**
    - 가우시안 분포를 가정하고 평균과 표준편차로부터의 거리를 측정하는 방식이다.
    - 하지만 실제 데이터가 정규 분포를 따르지 않거나 혼합 분포인 경우 적용이 어렵고, 고차원 데이터에서 성능이 저하되는 '차원의 저주' 문제가 발생한다. 또한, 평균과 분산 자체가 이상치에 매우 민감하여 거짓 음성(False Negative)이 발생할 가능성이 높다.
2. **주성분 분석 (PCA) 기반 방식:**
    - 데이터의 분산을 최대한 보존하는 주성분을 찾아 차원을 축소하고 이상치를 탐지한다.
3. **딥 오토인코더 (Deep Autoencoder):**
    - 입력 데이터를 재구성하도록 학습하며, 정상 데이터에 대해 낮은 재구성 오차를 갖는 특성을 이용해 이상치를 탐지한다.
4. **SBERT (Sentence-BERT):**
    - 기존 BERT는 문장 임베딩 생성 시 연산 비용이 매우 높거나(Cross-encoder), 단순히 $[CLS]$ 토큰을 사용하는 방식은 정확도가 낮았다. SBERT는 Siamese 네트워크 구조를 통해 문장 임베딩을 효율적으로 생성하며, 의미적 텍스트 유사도(STS) 작업에서 우수한 성능을 보인다.

## 🛠️ Methodology

본 논문에서 제안하는 시스템은 크게 **차원 축소(Dimension Reduction)**와 **이상치 식별(Identification of Outlier)**의 두 단계로 구성된다.

### 1. 특성 추출: Sentence-BERT (SBERT)

텍스트 데이터를 수치화하기 위해 SBERT를 사용한다. SBERT는 BERT 아키텍처에서 분류 헤드를 제거하고 Mean Pooling을 적용하여 문장 전체를 대표하는 고정 크기의 벡터를 생성한다. 이를 통해 각 문장은 $768$ 차원의 밀집 벡터(Dense Vector)로 변환된다.

### 2. 차원 축소 및 특징 학습: Convolutional Autoencoder (CAE)

SBERT로 생성된 벡터들을 입력으로 받는 Autoencoder를 구축한다.

- **입력 데이터:** $(1, \text{max\_sent}, 768)$ 형태의 3차원 배열을 입력으로 사용한다.
- **구조:** Encoder와 Decoder 모두 2D Convolutional layer를 사용한다.
- **잠재 공간 (Latent Space):** Encoder의 마지막 층인 Fully Connected layer를 통해 입력 데이터를 $32$ 차원의 context vector로 압축한다.
- **학습 목표:** 입력 데이터를 동일하게 복원하는 것을 목표로 하며, 입력값과 출력값 사이의 평균 제곱 오차(MSE)를 최소화하도록 학습한다.

### 3. 이상치 판별: Logistic Regression

최종적으로 이상치를 분류하기 위해 다음의 절차를 따른다.

- **특징 결합:** AE가 생성한 $32$ 차원의 잠재 벡터와, 입력-출력 간의 재구성 오차(Reconstruction Error, $1$ 차원)를 결합하여 총 $33$ 차원의 특징 벡터를 생성한다.
- **분류:** 이 벡터를 Logistic Regression 모델의 입력으로 넣어 해당 샘플이 정상(0)인지 이상치(1)인지를 예측한다.
- **데이터 불균형 처리:** 데이터셋의 클래스 불균형 문제를 해결하기 위해 오버샘플링(Over-sampling) 기법을 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋:** Cross-lingual Natural Language Inference (XNLI) 코퍼스를 정상 데이터로 사용하고, Stanford Sentiment Treebank (SST)에서 추출한 1,000개의 샘플을 이상치로 주입하여 합성 데이터셋(XNLI + SST)을 구축하였다.
- **평가 지표:** $F1\text{-score}$, $\text{Precision}$, $\text{Recall}$을 사용하였다.

### 결과 분석

검증 세트(Validation Sample) 3,490개에 대한 실험 결과는 다음과 같다.

| 지표 | 결과 값 |
| :--- | :--- |
| **F1 Score** | $0.8644$ |
| **Precision** | $0.918$ |
| **Recall** | $0.8167$ |

**Confusion Matrix:**
$$
\begin{bmatrix}
2284 & 82 \\
206 & 918
\end{bmatrix}
$$

- 정상 샘플 중 82개가 이상치로 오분류되었으며, 이상치 샘플 중 206개가 정상으로 오분류되었다. 전체적으로 $F1\text{-score}$ $86\%$의 높은 정확도를 보였으며, 특히 Precision이 높아 False Positive(정상을 이상치로 판단) 비율을 낮게 유지하고 있음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 Transformer 기반의 SBERT가 제공하는 풍부한 문맥 정보와 Autoencoder의 압축 능력을 결합함으로써, 단순한 거리 기반 방식보다 텍스트 이상치 탐지 성능을 향상시킬 수 있음을 보여주었다. 특히 텍스트를 $768$ 차원에서 $32$ 차원으로 압축하면서도 유의미한 특징을 유지하고, 여기에 재구성 오차라는 물리적 지표를 추가하여 분류기의 판단 근거를 강화한 점이 유효하였다.

다만, 몇 가지 한계점이 존재한다. 첫째, 실험에 사용된 이상치가 서로 다른 데이터셋(XNLI vs SST)을 단순히 섞어서 만든 인위적인 이상치라는 점이다. 실제 환경에서 발생하는 미묘한 텍스트 이상치에 대해서도 동일한 성능을 낼지는 검증되지 않았다. 둘째, 최종 분류기로 단순한 Logistic Regression을 사용하였는데, 더 복잡한 분류기나 비지도 학습 기반의 임계값(Threshold) 설정 방식을 도입했다면 더 나은 결과를 얻었을 가능성이 있다.

## 📌 TL;DR

본 논문은 SBERT를 통해 텍스트를 고차원 벡터로 변환하고, Convolutional Autoencoder를 이용해 이를 $32$ 차원으로 압축한 뒤, 잠재 벡터와 재구성 오차를 결합하여 이상치를 탐지하는 방법론을 제안한다. 실험 결과 $F1\text{-score}$ $0.86$의 성능을 기록하였으며, 이는 텍스트 분류 엔진의 전처리 필터로서 데이터의 품질을 높이는 데 기여할 수 있을 것으로 기대된다.
