# Mamba for Scalable and Efficient Personalized Recommendations

Andrew Starnes, Clayton Webster (2024)

## 🧩 Problem to Solve

본 논문은 개인화 추천 시스템(Personalized Recommendation Systems)에서 대규모 정형 데이터(Tabular Data)를 효율적으로 처리하는 문제를 해결하고자 한다. 기존의 추천 시스템은 시퀀스 데이터 처리 능력이 뛰어난 Transformer 아키텍처를 많이 채택하고 있으나, Transformer는 입력 시퀀스 길이에 대해 이차 복잡도($O(L^2)$)를 가지는 연산 비용 문제로 인해 데이터셋의 규모가 커질수록 확장성(Scalability)이 제한되는 치명적인 단점이 있다.

따라서 본 연구의 목표는 Transformer의 성능적 이점은 유지하면서 연산 복잡도를 선형 복잡도($O(L)$)로 낮추어, 대규모 정형 데이터를 처리할 때 더 효율적이고 확장 가능한 추천 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 정형 데이터를 위한 Feature Tokenizer와 Mamba 레이어를 결합한 하이브리드 모델인 **FT-Mamba**를 제안한 것이다.

가장 중심적인 아이디어는 기존의 FT-Transformer 구조에서 Transformer 레이어를 Mamba 레이어로 교체하는 것이다. Mamba는 State Space Models (SSMs)의 능력을 강화하여 연산 복잡도를 선형 수준으로 낮추면서도 Transformer 수준의 성능을 제공한다. 또한, Mamba의 특성상 마지막 토큰이 이전의 모든 정보를 집약한다는 점을 고려하여, 기존 FT-Transformer에서 시퀀스 시작 부분에 배치하던 `[CLS]` 토큰을 시퀀스의 끝으로 이동 배치하는 아키텍처 수정을 제안하였다.

## 📎 Related Works

논문에서는 정형 데이터를 Mamba에 적용하려는 최근의 시도들로 MambaTab과 Mambular를 언급한다. MambaTab은 특징을 인코딩한 후 피드포워드 신경망을 통해 임베딩하는 방식을 사용하며, Mambular는 FT-Transformer와 Mamba를 결합하되 수치형 특징에 대해 주기적 인코딩(Periodic Encodings)을 사용한다는 점에서 본 연구와 유사하나 차이가 있다.

추천 시스템 분야에서는 Mamba4Rec, MaTrRec, SIGMA와 같은 모델들이 제안되었다. 이들은 주로 사용자-아이템 상호작용 시퀀스를 처리하는 순차 추천(Sequential Recommendation)에 집중하고 있다. 반면, 본 논문은 인구통계학적 정보와 같은 정형(Tabular) 사용자 데이터를 처리하는 개인화 추천에 초점을 맞춘다는 점에서 기존 연구들과 차별점을 가진다.

## 🛠️ Methodology

### 1. Feature Tokenizer
정형 데이터를 Mamba 레이어가 처리할 수 있는 시퀀스 형태로 변환하기 위해 Feature Tokenizer를 사용한다.

- **수치형 특징 (Numerical Features):** 실수 값 $x_j$를 학습 가능한 파라미터 $w_j, b_j \in \mathbb{R}^d$를 사용하여 선형적으로 스케일링한다.
  $$T_j = x_j w_j + b_j$$
- **범주형 특징 (Categorical Features):** 각 특징의 카테고리 인덱스를 사용하여 룩업 테이블(Lookup Table) $W_j \in \mathbb{R}^{K_j \times d}$에서 임베딩을 추출하고 바이어스 $b_j$를 더한다.
  $$T_j = e_{x_j}^T W_j + b_j$$
  여기서 $e_{x_j}$는 $x_j$ 위치만 1이고 나머지는 0인 원-핫 벡터이다.
- **[CLS] 토큰:** 전체 특성을 대표하는 임베딩을 얻기 위해 수치 값 1을 입력으로 하는 특수 토큰 $T_{[CLS]} = w_{k+1} + b_{k+1}$을 생성하여 시퀀스의 마지막에 추가한다. 최종 토큰 시퀀스는 $T = \text{stack}\{T_1, \dots, T_k, T_{[CLS]}\}$가 된다.

### 2. Mamba Layer
FT-Mamba의 핵심인 Mamba 레이어는 다음과 같은 구성 요소로 이루어진다.
- 두 개의 선형 투영 레이어(Linear Projection Layers)
- 1D 컨볼루션 레이어(Convolutional Layer)
- 선택적 상태 공간 모델(Selective State Space Model, SSM)
- 비선형 활성화 함수(Nonlinear Activation Function)

Mamba의 SSM은 기본적으로 선형 RNN과 유사하게 $h_t = Ah_{t-1} + Bx_t$ 형태로 동작하지만, 입력에 따라 $A, B$ 등의 파라미터를 동적으로 조정하는 'Selective' 메커니즘을 통해 이전 입력을 잊어버리는 능력을 갖추며, 스캐닝 기법(Scanning Technique)을 통해 $O(L)$의 선형 시간 복잡도를 유지한다. 본 구현에서는 임베딩의 발산을 막기 위해 최종 선형 투영 이후에 선택적으로 정규화 레이어(Normalization Layer)를 배치하였다.

### 3. Two-Tower Architecture
추천 시스템의 효율적인 추론을 위해 Two-Tower 구조를 채택하였다.
- **User Tower:** 사용자의 정형 데이터를 입력받아 FT-Mamba(또는 FT-Transformer)를 통해 사용자 임베딩을 생성한다.
- **Content Tower:** 콘텐츠의 정형 데이터를 입력받아 동일한 구조를 통해 콘텐츠 임베딩을 생성한다.
- **Similarity Score:** 두 타워에서 생성된 임베딩의 내적(Inner Product)을 통해 최종 예측 점수를 산출한다.

## 📊 Results

### 실험 설정
- **데이터셋:** Spotify 음악 추천, H&M 패션 추천, Vaccine 메시징 추천(합성 데이터).
- **훈련 데이터:** 각 데이터셋당 160,000개의 사용자-액션 쌍을 사용하였으며, 손실 함수로는 MSE(Mean Squared Error)를 사용하였다.
- **지표:** Precision (P), Recall (R), Mean Reciprocal Rank (MRR), Hit Ratio (HR)를 $k=1, 5, 10$ 기준에서 측정하였다.
- **모델 파라미터:** Mamba 임베더는 약 479,232개의 파라미터를 가지며, 이는 Transformer 임베더(약 1,186,048개)의 약 40% 수준이다.

### 주요 결과
1. **Spotify 음악 추천:** FT-Mamba가 모든 지표에서 Transformer를 압도하였다. 특히 $\text{HR@1}$에서 Mamba는 97.7%의 성능을 보인 반면, Transformer는 84.11%에 그쳤다.
2. **H&M 패션 추천:** 두 모델의 성능이 비슷하게 나타났다. $\text{P@1}, \text{MRR@1}, \text{HR@1}$과 같은 단일 추천 지표에서는 Transformer가 근소하게 우세했으나, 전반적인 성능 차이는 크지 않았다.
3. **Vaccine 메시징 추천:** FT-Mamba가 Transformer보다 월등한 성능을 보였다. 대부분의 설정에서 $\text{HR@1}$이 Transformer 대비 거의 두 배 가까이 높게 나타났다.

결과적으로 FT-Mamba는 훨씬 적은 파라미터 수와 낮은 연산 복잡도를 가지면서도, 대부분의 추천 작업에서 Transformer와 비슷하거나 더 우수한 성능을 기록하였다.

## 🧠 Insights & Discussion

본 연구는 Mamba 아키텍처가 정형 데이터를 이용한 개인화 추천 시스템에서 Transformer의 효율적인 대안이 될 수 있음을 입증하였다. 특히 모델 파라미터 수를 60% 가량 줄였음에도 불구하고 성능이 유지되거나 향상되었다는 점은 실시간 추천 서비스와 같은 대규모 시스템에서 매우 강력한 강점이 된다.

다만, H&M 데이터셋 실험에서 나타났듯이 특정 데이터셋에서는 성능 향상이 미미할 수 있으며, 이는 무작위 네거티브 샘플링(Random Negative Sampling) 방식의 한계일 가능성이 크다. 더 정교한 네거티브 샘플링 전략이나 더 많은 학습 데이터가 확보된다면 Mamba의 효율성이 더 극대화될 수 있을 것으로 보인다.

또한, Mamba의 인과적(Causal) 특성 때문에 `[CLS]` 토큰의 위치를 끝으로 옮긴 결정은 아키텍처 설계 시 매우 중요한 포인트이다. 이는 정형 데이터를 시퀀스로 처리할 때 모델의 정보 집약 방식을 정확히 이해하고 적용해야 함을 시사한다.

## 📌 TL;DR

본 논문은 정형 데이터 기반 추천 시스템의 확장성 문제를 해결하기 위해 Transformer를 Mamba로 대체한 **FT-Mamba**를 제안한다. FT-Mamba는 연산 복잡도를 $O(L^2)$에서 $O(L)$로 낮추고 파라미터 수를 획기적으로 줄였음에도 불구하고, 세 가지 실제/합성 데이터셋 실험에서 Transformer와 대등하거나 더 우수한 추천 성능을 보였다. 이 연구는 대규모 실시간 개인화 추천 시스템을 구축하는 데 있어 Mamba 아키텍처가 매우 유망한 해결책이 될 수 있음을 보여준다.