# APPLYING SOFTTRIPLE LOSS FOR SUPERVISED LANGUAGE MODEL FINETUNING

Witold Sosnowski, Anna Wróblewska, Piotr Gawrysiak (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 사전 학습된 언어 모델(Pre-trained Language Models, PLMs)을 특정 분류 작업에 맞춰 미세 조정(Fine-tuning)할 때, 표준적으로 사용되는 Cross-Entropy loss가 가진 한계를 극복하는 것이다.

일반적으로 Cross-Entropy loss는 클래스 간의 선형 분리 가능성(Linear Separability)에만 집중하며, 클래스 내부의 응집도나 클래스 간의 거리와 같은 표현 학습(Representation Learning) 관점의 최적화에는 부족함이 있다. 이는 결과적으로 모델이 클래스별 고유한 특징보다는 단순히 클래스를 구분 짓는 경계선에만 집중하게 만들어, 새로운 데이터에 대한 일반화 성능을 떨어뜨리고 이상치(Outlier)에 취약하게 만드는 원인이 된다.

따라서 본 연구의 목표는 Distance Metric Learning(DML) 기법인 SoftTriple loss를 Cross-Entropy loss와 결합한 새로운 손실 함수인 **TripleEntropy**를 제안하여, 분류 성능을 향상시키고 더 견고한 임베딩 공간을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 분류를 위한 예측 성능(Cross-Entropy)과 임베딩 공간의 구조적 최적화(SoftTriple loss)를 동시에 달성하는 것이다.

1. **TripleEntropy 손실 함수 제안**: 기존의 Multinomial Cross-Entropy(MCE) loss와 SoftTriple loss를 가중 결합하여, 클래스 내 거리는 최소화하고 클래스 간 거리는 최대화하도록 유도한다.
2. **Proxy 기반의 효율적 학습**: 모든 데이터 쌍(Pair)이나 삼조(Triplet)를 직접 계산하는 대신, 각 클래스를 대표하는 가상의 점인 Proxy를 도입함으로써 계산 복잡도를 획기적으로 낮추면서도 DML의 이점을 취했다.
3. **토큰 수준의 임베딩 최적화**: 기존 연구들이 주로 `[CLS]` 토큰의 임베딩만을 사용하여 Contrastive Learning을 수행했던 것과 달리, 입력 문장의 모든 토큰 임베딩에 SoftTriple loss를 적용함으로써 모델의 일반화 능력을 극대화하였다.

## 📎 Related Works

### 문장 임베딩 및 언어 모델

초기의 Bag-of-Words(BOW)나 Word2Vec과 같은 방식은 문맥을 충분히 반영하지 못하는 한계가 있었다. 이후 BERT와 그 발전 형태인 RoBERTa가 등장하며 문맥 기반의 고품질 문장 임베딩 생성이 가능해졌으며, 현재 텍스트 분류 작업의 표준 베이스라인으로 자리 잡았다.

### Distance Metric Learning (DML)

DML은 동일 클래스 샘플 간의 거리는 가깝게, 서로 다른 클래스 간의 거리는 멀게 학습하는 방법론이다.

- **Contrastive Loss**: 유사한 쌍은 가깝게, 다른 쌍은 멀게 학습한다.
- **Triplet Loss**: Anchor, Positive, Negative 세 개의 샘플을 이용하여 상대적인 거리 차이를 학습한다. 하지만 배치 크기가 커질수록 계산량이 급격히 증가하고, 구분이 쉬운 샘플(Easy triplets)만 학습될 경우 성능 향상이 제한적인 문제가 있다.
- **ProxyNCA Loss**: 클래스당 하나의 Proxy를 도입하여 계산 효율성을 높인 방법이다.
- **SoftTriple Loss**: 클래스당 여러 개의 Proxy를 도입하여 실제 데이터의 복잡한 구조를 더 잘 반영하도록 개선된 DML 손실 함수이다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구는 RoBERTa-base 및 RoBERTa-large 모델을 인코더 $E(\cdot)$로 사용한다. 입력 텍스트는 WordPiece 토큰화 과정을 거쳐 모델에 입력되며, 모델의 출력값은 두 가지 경로로 활용된다. 하나는 풀링(Pooling) 후 Fully Connected 레이어를 통해 클래스 확률을 예측하는 MCE 경로이고, 다른 하나는 모든 토큰 임베딩을 직접 사용하여 거리 기반으로 학습하는 SoftTriple 경로이다.

### 손실 함수 (TripleEntropy)

최종 목적 함수 $L$은 다음과 같이 정의된다.

$$L = \beta L_{MCE} + (1-\beta) L_{SoftTriple}$$

여기서 $\beta$는 두 손실 함수의 영향력을 조절하는 스케일링 인자이다.

#### 1. Multinomial Cross-Entropy (MCE) Loss

표준적인 분류 손실 함수로, 모델이 예측한 확률 $p_{ic}$와 실제 정답 확률 $y_{ic}$ 사이의 차이를 계산한다.

$$L_{MCE} = -\frac{1}{N} \sum_{i}^{N} \sum_{c}^{C} y_{ic} \log(p_{ic})$$

#### 2. SoftTriple Loss

SoftTriple loss는 클래스별로 $k$개의 Proxy(학습 가능한 가중치 $w_c^k$)를 둔다. 샘플 $x_i$와 각 클래스 Proxy 간의 유사도를 계산하여 다음과 같이 정의된다.

$$\text{SoftTriple} = -\log \frac{\exp(\lambda(S'_{i,y_i} - \delta))}{\exp(\lambda(S'_{i,y_i} - \delta)) + \sum_{j \neq y_i} \exp(\lambda S'_{i,j})}$$

여기서 $S'_{i,c}$는 샘플 $x_i$와 클래스 $c$의 Proxy들 간의 관계를 나타내며, 다음 식을 통해 계산된다.

$$S'_{i,c} = \frac{\sum_{k} \exp(\frac{1}{\gamma} E(x_i) \cdot w_c^k)}{\sum_{k} \exp(\frac{1}{\gamma} E(x_i) \cdot w_c^k) E(x_i) \cdot w_c^k}$$

- $\delta$: 클래스 간 마진(Margin)
- $\lambda$: 이상치(Outlier)의 영향을 줄여 강건성을 높이는 파라미터
- $\gamma$: 엔트로피 정규화(Entropy regularizer)를 위한 스케일링 인자

### 학습 절차

- **최적화**: AdamW 옵티마이저를 사용하며, Linear Warmup 스케줄러를 적용하였다.
- **입력 처리**: 모든 토큰 임베딩 $B \times |x_i|$를 SoftTriple loss의 입력으로 넣어 Proxy들이 클래스 표현을 더 정밀하게 학습하도록 하였다.
- **하이퍼파라미터**: Grid Search를 통해 $k, \gamma, \lambda, \delta, \beta$ 등의 최적 조합을 탐색하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: SST2, IMDb, MR, MPQA, SUBJ, TREC, CR, MRPC 등 SentEval과 IMDb의 다양한 데이터셋을 사용하였다.
- **데이터 규모 분류**: 샘플 수에 따라 Small(1k), Medium(4k~5k), Large(10k~11k), Extra-Large(50k 이상)로 구분하여 실험하였다.
- **비교 대상**: RoBERTa-base (RB) 및 RoBERTa-large (RL)의 표준 Cross-Entropy 학습 모델을 베이스라인으로 설정하였다.

### 주요 결과

TripleEntropy를 적용한 모델은 모든 데이터 규모에서 베이스라인 대비 성능 향상을 보였으며, 특히 데이터셋의 크기가 작을수록 성능 이득이 크게 나타났다.

- **Small Dataset**: 평균 **0.78%** 향상 (최대 2.29% 증가, TREC-1k 기준).
- **Medium Dataset**: 평균 **0.86%** 향상.
- **Large Dataset**: 평균 **0.20%** 향상.
- **Extra-Large Dataset**: 평균 **0.04%** 향상 (비유의미한 수준).

또한, RoBERTa-large 모델을 이용한 소량 데이터(1k) 실험에서 기존의 Supervised Contrastive Learning 기반 연구(Gunel et al., 2020)의 성능 향상폭(0.27%)보다 높은 **0.48%**의 향상을 달성하였다.

## 🧠 Insights & Discussion

본 연구의 결과는 **데이터의 양이 적을수록 Distance Metric Learning(DML)의 효과가 극대화된다**는 점을 시사한다. 데이터가 충분한 경우(Extra-Large)에는 표준 Cross-Entropy만으로도 충분한 결정 경계를 찾을 수 있지만, 데이터가 부족한 상황에서는 단순한 분류 경계 학습보다는 클래스 자체의 강건한 표현(Representation)을 학습하는 것이 일반화 성능 향상에 훨씬 유리하기 때문이다.

**강점**:

- Proxy 기반 방식을 통해 Triplet Loss의 고질적인 계산 비용 문제를 해결하면서도 성능을 높였다.
- `[CLS]` 토큰에 국한되지 않고 모든 토큰 임베딩을 최적화함으로써 문장 전체의 의미론적 구조를 더 잘 반영하였다.

**한계 및 논의**:

- 모든 토큰 임베딩을 SoftTriple loss에 통과시키기 때문에, 베이스라인 대비 더 많은 계산 자원이 필요하다.
- 데이터셋이 매우 큰 경우에는 성능 향상 폭이 미미하므로, 대규모 데이터셋에서는 굳이 복잡한 DML을 도입할 실익이 적을 수 있다.

## 📌 TL;DR

본 논문은 사전 학습된 언어 모델의 미세 조정 단계에서 분류 성능을 높이기 위해 **Multinomial Cross-Entropy와 SoftTriple loss를 결합한 TripleEntropy 손실 함수**를 제안하였다. 특히 클래스별 Proxy를 활용하여 계산 효율성을 높였으며, 모든 토큰 임베딩을 학습에 활용함으로써 임베딩 공간의 구조를 최적화하였다. 실험 결과, 데이터셋의 크기가 작을수록 성능 향상 폭이 뚜렷하게 나타났으며, 이는 데이터 부족 상황에서 모델의 일반화 능력을 높이는 데 매우 효과적인 전략임을 입증하였다. 이 연구는 향후 적은 양의 데이터로 고성능 분류기를 구축해야 하는 Few-shot learning 분야 등에 기여할 가능성이 크다.
