# Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention

Zhihao Zhan, Jianan Zhao, Zhaocheng Zhu, Jian Tang (2025)

## 🧩 Problem to Solve

본 논문은 자연어 처리(NLP)에서 긴 문맥(Long-context)을 효율적으로 모델링하는 문제를 다룬다. 현재 주류인 Transformer 아키텍처는 시퀀스 길이에 따라 연산 복잡도가 이차적으로 증가하는 $\mathcal{O}(l^2)$ 문제를 가지고 있다. 이를 해결하기 위해 sub-quadratic 복잡도를 가진 State-Space Models(SSMs)가 대안으로 제시되었으나, SSMs는 Transformer에 비해 장거리 의존성(long-range dependencies)을 포착하는 능력이 떨어진다는 한계가 있다.

특히, 기존 연구들이 사용한 'Associative Recall'(하나의 키에 하나의 값이 고정적으로 연결된 형태) 작업은 실제 자연어의 복잡한 문맥 의존성을 충분히 반영하지 못한다. 실제 언어에서는 동일한 키라도 주변 문맥에 따라 서로 다른 값으로 매핑되는 경우가 많다. 따라서 본 논문의 목표는 SSMs의 표현 능력 한계를 분석하고, 이를 극복하여 긴 문맥 모델링 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Joint Recall 작업 제안**: 기존의 Associative Recall을 확장하여, 특정 문맥(Context)이 주어졌을 때 그에 맞는 값을 회상해야 하는 'Joint Recall'이라는 새로운 합성 작업을 정의하였다.
2. **이론적 분석**: 표준 SSMs가 sub-quadratic 시간 복잡도 내에서 multi-query joint recall 문제를 해결할 수 있는 표현 능력이 부족함을 이론적으로 증명하였다. 또한, Context-Dependent Sparse Attention(CDSA)을 결합했을 때만 이 문제가 효율적으로 해결될 수 있음을 보였다.
3. **HAX 아키텍처 제안**: 이론적 통찰을 바탕으로 Locality-Sensitive Hashing(LSH) Attention과 새롭게 제안한 Key Selection(KS) Attention을 결합한 **HAX** 아키텍처를 제안하였다. 이를 Mamba 및 Mamba2와 통합하여 실제 성능 향상을 입증하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 언급한다.

- **State-Space Models (SSMs)**: HiPPO, LSSL, H3를 거쳐 최근의 Mamba와 Mamba2에 이르기까지, SSMs는 입력 의존적인 파라미터를 통해 Transformer와 유사한 성능을 내면서도 효율적인 연산을 가능하게 했다.
- **Sparse Attention**: 연산량을 줄이기 위해 전체 attention map 대신 일부만 계산하는 방식이다. LSH Attention과 같은 CDSA(입력 내용에 따라 패턴이 변함)와 Sliding Window, Dilated Attention과 같은 CISA(패턴이 고정됨)로 구분된다.
- **SSMs의 한계 분석**: Jelassi et al. (2024) 등은 SSMs가 단순한 Copying 작업조차 수행하려면 상태 차원(state dimension)이 시퀀스 길이에 따라 선형적으로 증가해야 한다는 점을 지적하며 표현 능력의 한계를 보였다.

## 🛠️ Methodology

### 1. Joint Recall Formulation

Joint Recall은 모델이 $n_c$개의 문맥과 각 문맥당 $n_k$개의 키-값 쌍으로 구성된 테이블을 기억하도록 한다. 즉, 총 $n = n_c \times n_k$개의 항목을 학습해야 하며, 쿼리 시에는 `(문맥, 키)` 쌍이 주어졌을 때 올바른 `값`을 출력해야 한다.

### 2. Theoretical Analysis

- **SSMs의 한계**: 상태 공간의 크기를 $|U|$라고 할 때, multi-query joint recall에서 오류율은 최소 $1 - \frac{|U|}{|V|^n}$이 된다. 즉, 오류를 없애려면 상태 차원이 항목 수 $n$에 비례하여 선형적으로 증가해야 하며, 이는 고정된 상태 차원을 갖는 SSMs의 근본적인 한계이다.
- **CDSA의 필요성**: 이론적으로 SSM과 LSH Attention(CDSA의 일종)을 결합한 2계층 하이브리드 모델은 $\mathcal{O}(\log n)$의 SSM 상태 차원만으로 $\mathcal{O}(n \log^2 n)$ 시간 복잡도 내에 문제를 해결할 수 있다. 반면, CISA를 결합한 모델은 $\mathcal{O}(n^2)$의 복잡도가 필요함을 증명하였다.

### 3. HAX (Hashing Attention with sparse Key Selection)

HAX는 LSH Attention의 한계(특정 토큰에 집중하는 "Vertical-stripe" 패턴 포착 어려움)를 해결하기 위해 **Key Selection (KS) Attention**을 도입하여 결합한 구조이다.

#### (1) Key Selection (KS) Attention

KS Attention은 모든 쿼리가 공통적으로 주목하는 전역적으로 중요한 키(예: 지시어, 포맷 마커)를 포착하도록 설계되었다.

- **Key Scoring**: MLP를 통해 각 키 $K_i$와 이전 쿼리들의 평균 $\bar{Q}$를 입력받아 중요도 점수 $x_i$를 계산한다.
  $$x_i = \text{MLP}(K_i, \text{normalize}(\sum_{p=1}^i Q_p))$$
- **Key Selection**: 각 쿼리는 점수가 가장 높은 상위 $k$개의 키에만 주의를 기울인다.
  $$S_{KS_{ij}} = 1[x_j \in \text{Top-k}\{x_1, \dots, x_i\}]$$
- **Training**: 실제 attention weight와 예측 점수 간의 순위가 일치하도록 Pairwise Ranking Loss $\mathcal{L}_{score}$를 사용하여 학습한다.

#### (2) HAX 통합 및 전체 구조

LSH Attention의 동적 라우팅 능력과 KS Attention의 전역 제어 능력을 결합한다.
$$S_{HAX} = \max\{S_{LSH}, S_{KS}\}$$
이 결과로 생성된 sparse attention pattern은 Mamba나 Mamba2와 같은 SSM 계층과 병렬 또는 순차적으로 결합된다. 그림 3에 묘사된 바와 같이, SSM의 출력과 HAX의 출력을 게이트(gate)를 통해 융합하여 최종 출력을 생성한다.

## 📊 Results

### 1. 합성 데이터 (Multi-Query Joint Recall)

Mamba 및 Mamba2 기반의 다양한 하이브리드 모델을 비교한 결과, HAX가 가장 높은 정확도를 보였다. 특히 SSM 단독 모델이나 CISA(Dilated, Sliding Window 등)를 결합한 모델보다 월등히 높은 성능을 기록하여, CDSA의 이론적 우위가 실증적으로 확인되었다.

### 2. 실제 자연어 데이터 (Continual Pre-training)

Mamba 130M/790M 모델을 기반으로 The Pile 및 TxT360 데이터셋에서 추가 사전 학습을 진행하였다.

- **학습 안정성**: Validation Loss 측정 결과, Mamba 베이스라인과 CISA 결합 모델들은 학습 중 불안정하거나 조기에 정체되는 모습을 보였으나, HAX 모델은 지속적으로 loss가 감소하며 안정적인 학습 곡선을 그렸다.
- **Ruler & LongBench**: 2K 문맥 길이에서 평가한 결과, HAX는 retrieval, multi-hop reasoning 등 다양한 long-context 작업에서 Mamba 베이스라인을 유의미하게 앞질렀다.
- **외삽(Extrapolation) 능력**: 2K 길이로 학습된 모델을 4K 길이에 적용했을 때도 HAX가 가장 강건한 성능을 유지하였다.

## 🧠 Insights & Discussion

본 연구는 SSMs가 긴 문맥에서 취약한 이유가 단순히 메모리 용량의 문제가 아니라, **입력 내용에 따라 동적으로 주의 집중 대상을 변경하는 표현 능력(expressiveness)**의 부족에 있음을 밝혀냈다.

- **강점**: 이론적 분석을 통해 'Joint Recall'이라는 정교한 벤치마크를 제시하고, 이를 해결하기 위한 최적의 구조(CDSA)를 도출하여 아키텍처 설계에 반영한 점이 매우 논리적이다.
- **비판적 해석**: HAX는 LSH와 KS라는 두 가지 sparse attention을 결합하여 복잡도를 $\mathcal{O}(l \log l)$ 수준으로 유지하면서도 성능을 높였다. 하지만 LSH의 해시 충돌 문제나 KS Attention의 MLP 스코어링 오버헤드가 실제 대규모 모델 배포 시 어느 정도의 실질적 지연 시간을 유발할지는 명시되지 않았다.
- **결론**: SSMs와 CDSA의 결합은 Transformer의 연산 효율성과 장거리 의존성 포착 능력을 동시에 잡을 수 있는 유망한 방향임을 시사한다.

## 📌 TL;DR

이 논문은 SSMs가 긴 문맥의 문맥 의존적 정보(Joint Recall)를 효율적으로 처리하지 못한다는 이론적 한계를 증명하고, 이를 해결하기 위해 **LSH Attention**과 **Key Selection Attention**을 결합한 **HAX** 아키텍처를 제안하였다. HAX는 sub-quadratic 복잡도를 유지하면서도 실제 long-context 벤치마크에서 SSM 및 기존 sparse attention 결합 모델들을 압도하는 성능을 보였으며, 이는 향후 효율적인 초거대 문맥 모델링 연구에 중요한 설계 지침을 제공한다.
