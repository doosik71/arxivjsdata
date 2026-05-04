# Is Mamba Effective for Time Series Forecasting?

Zihan Wang, Fanheng Kong, Shi Feng, Ming Wang, Xiaocui Yang, Han Zhao, Daling Wang, Yifei Zhang (2024)

## 🧩 Problem to Solve

시계열 예측(Time Series Forecasting, TSF)의 핵심은 과거 데이터에 숨겨진 패턴, 즉 시간적 의존성(Temporal Dependency, TD)과 변수 간 상관관계(Inter-variate Correlation, VC)를 정확하게 파악하여 미래 상태를 예측하는 것이다.

기존의 Transformer 기반 모델들은 이러한 패턴을 포착하는 능력이 뛰어나 매우 강력한 성능을 보여주었다. 그러나 Transformer의 Self-attention 메커니즘은 연산 복잡도가 입력 길이에 대해 제곱 비례($O(N^2)$)하는 특성이 있어, 변수의 개수가 많아지거나 룩백 윈도우(lookback window)가 길어질 때 계산 비용과 GPU 메모리 사용량이 급격히 증가한다. 이는 실시간성이 중요하거나 대규모 데이터를 처리해야 하는 실제 환경에서의 배포를 어렵게 만든다.

반면, 선형(Linear) 모델들은 계산 복잡도가 낮아 효율적이지만, 컨텍스트 정보를 충분히 활용하지 못해 Transformer보다 성능이 떨어지는 경향이 있다. 따라서 본 논문의 목표는 Transformer 수준의 성능을 유지하면서도 계산 효율성을 획기적으로 높인 Mamba 기반의 시계열 예측 모델을 제안하고, Mamba가 TSF 작업에서 실제로 효과적인지 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 선택적 상태 공간 모델(Selective State Space Model, SSM)이 가진 선형 복잡도와 문맥 파악 능력을 TSF에 접목하는 것이다. 이를 위해 저자들은 **Simple-Mamba (S-Mamba)**라는 구조를 제안하며, 다음과 같은 설계를 도입하였다.

1. **변수 간 상관관계(VC) 추출을 위한 Bidirectional Mamba**: Mamba는 기본적으로 단방향으로 정보를 처리한다. 변수 간의 전역적인 상호작용을 포착하기 위해 정방향과 역방향의 Mamba 블록을 결합한 양방향 구조를 설계하여 Transformer의 Global Attention과 유사한 효과를 내면서도 연산 비용을 낮추었다.
2. **시간적 의존성(TD) 추출을 위한 FFN**: 복잡한 구조 대신 단순한 Feed-Forward Network(FFN)를 통해 각 변수의 시간적 순서 정보를 학습하도록 설계하여 모델의 경량화를 달성하였다.
3. **효율성과 성능의 균형**: S-Mamba는 GPU 메모리 사용량과 학습 시간을 크게 줄이면서도, 특히 변수 개수가 많고 주기성이 강한 데이터셋에서 최신 SOTA 모델들과 대등하거나 더 뛰어난 성능을 보임을 입증하였다.

## 📎 Related Works

### 1. 시계열 예측 모델의 기존 접근 방식

- **Transformer-based Models**: iTransformer, PatchTST, Autoformer 등이 있으며, 장기 의존성 포착 능력이 뛰어나지만 $O(N^2)$의 복잡도와 포지셔널 인코딩(Position Encoding) 문제가 존재한다.
- **Linear Models**: DLinear, RLinear, TiDE 등이 있으며, 구조가 단순하고 효율적이지만 변동성이 크거나 비주기적인 패턴을 가진 데이터에서는 성능이 낮으며, Transformer보다 더 긴 입력 시퀀스가 필요하다는 단점이 있다.

### 2. Mamba 및 SSM의 응용

최근 Mamba는 NLP, CV 등 다양한 분야에서 Transformer를 대체할 대안으로 주목받고 있다. SSM은 합성곱(Convolution) 계산을 통해 시퀀스 정보를 캡처하고 은닉 상태(hidden states)를 제거함으로써 병렬 계산이 가능하며, 거의 선형에 가까운 복잡도를 가진다. 이전 연구에서 SSM을 TSF에 적용하려는 시도가 있었으나, 콘텐츠를 효과적으로 필터링하지 못하고 거리 기반의 의존성만 캡처하여 성능이 만족스럽지 못했다. Mamba는 여기에 '선택 메커니즘(selective mechanism)'을 추가하여 필요한 정보를 선별적으로 수용함으로써 이 문제를 해결하였다.

## 🛠️ Methodology

S-Mamba는 크게 네 개의 레이어로 구성된 파이프라인을 가진다.

### 1. 전체 시스템 구조

S-Mamba의 전체 흐름은 다음과 같다:
$$\text{Input} \rightarrow \text{Linear Tokenization} \rightarrow \text{Mamba VC Encoding} \rightarrow \text{FFN TD Encoding} \rightarrow \text{Projection} \rightarrow \text{Forecast}$$

### 2. 주요 구성 요소 및 역할

#### (1) Linear Tokenization Layer

입력 시계열 데이터를 표준화된 토큰 형식으로 변환한다. iTransformer와 유사하게 각 변수의 시계열을 독립적으로 토큰화하며, 단순한 선형 레이어를 통해 수행된다.
$$\text{Token} = \text{Linear}(\text{Batch}(\text{Input}))$$

#### (2) Mamba VC Encoding Layer

변수 간의 상관관계(Inter-variate Correlation)를 추출하는 단계이다. Mamba의 단방향성을 극복하기 위해 양방향 Mamba 블록을 사용한다.

- **정방향 Mamba**: $\vec{y} = \text{Mamba\_Block}(\text{Token})$
- **역방향 Mamba**: $\overleftarrow{y} = \text{Mamba\_Block}(\text{Token}_{\text{reversed}})$
- **결합**: $\text{Fusion} = \vec{y} + \overleftarrow{y}$
이후 잔차 연결(Residual Connection)을 통해 $\text{Output}_{\text{VC}} = \text{Fusion} + \text{Token}$으로 최종 출력한다.

#### (3) FFN TD Encoding Layer

시간적 의존성(Temporal Dependency)을 학습하는 단계이다.

- 먼저 Layer Normalization을 통해 데이터를 가우시안 분포로 표준화하여 학습 안정성을 높인다.
- 이후 Feed-Forward Network(FFN)를 통해 각 변수의 시퀀스 관계를 암시적으로 인코딩하고 미래 표현을 디코딩한다.
- 마지막으로 다시 한번 Normalization을 적용한다.

#### (4) Projection Layer

FFN을 통해 처리된 토큰 정보를 다시 시계열 형태로 복원하여 최종 예측값을 생성하는 매핑 레이어이다.

### 3. Mamba Block의 수학적 배경

Mamba의 기반이 되는 SSM은 다음과 같은 연속 시간 상태 방정식으로 표현된다:
$$\begin{aligned} h'(t) &= Ah(t) + Bx(t) \\ y(t) &= Ch(t) \end{aligned}$$
여기서 $A, B, C$는 학습 가능한 행렬이다. 이를 스텝 사이즈 $\Delta$를 이용하여 이산화(Discretization)하면 다음과 같은 재귀적 형태로 변환되어 효율적인 계산이 가능해진다:
$$\begin{aligned} h_t &= \bar{A}h_{t-aligned-1} + \bar{B}x_t \\ y_t &= Ch_t \end{aligned}$$
Mamba는 여기서 $\bar{A}, \bar{B}, C$ 등을 입력값 $x$에 따라 동적으로 결정하는 선택 메커니즘을 도입하여, Transformer의 Attention과 유사하게 중요한 정보만을 선택적으로 기억하도록 한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Traffic, PEMS(03, 04, 07, 08), ETT(m1, m2, h1, h2), Electricity, Exchange, Weather, Solar-Energy 등 총 13개의 공개 데이터셋 사용.
- **비교 대상 (Baselines)**:
  - Transformer 기반: iTransformer, PatchTST, Crossformer, FEDformer, Autoformer
  - Linear 기반: RLinear, TiDE, DLinear
  - TCN 기반: TimesNet
- **측정 지표**: Mean Squared Error (MSE), Mean Absolute Error (MAE).

### 2. 주요 결과

- **정량적 성능**: S-Mamba는 Traffic, Electricity, Solar-Energy와 같이 **변수가 많고 주기성이 강한 데이터셋**에서 최상위 성능을 기록하였다. 반면, ETT나 Exchange 같이 변수 수가 적고 비주기적인 데이터셋에서는 상대적으로 성능이 낮거나 SOTA 대비 열세를 보였다. 이는 VC Encoding 레이어가 변수 간 관계가 약한 데이터에서는 오히려 노이즈로 작용할 수 있음을 시사한다.
- **계산 효율성**: RTX 3090 GPU 실험 결과, S-Mamba는 Transformer 기반 모델들에 비해 학습 시간과 GPU 메모리 사용량을 현저히 낮추면서도 MSE 성능을 최적화하는 'Sweet Spot'에 위치함을 확인하였다. (RLinear는 더 빠르고 가볍지만 정확도가 낮음)
- **일반화 능력**: 전체 변수의 40%만 사용하여 학습시킨 후 100% 변수를 예측하게 하는 실험에서, S-Mamba가 Transformer와 유사한 수준의 일반화 능력을 갖추고 있음을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 기여

S-Mamba는 Mamba의 선형 복잡도를 활용하여 대규모 변수를 가진 시계열 데이터에서도 효율적으로 작동함을 보여주었다. 특히 Bidirectional Mamba 구조가 Transformer의 Global Attention을 효과적으로 대체하여, 연산 비용 증가 없이 변수 간의 복잡한 상관관계를 잘 포착한다는 점이 입증되었다.

### 2. 한계 및 분석

- **데이터 특성에 따른 성능 편차**: 본 모델은 변수 간의 상관관계(VC)가 뚜렷한 주기적 데이터에서 강점을 보이지만, 비주기적 데이터에서는 그 효과가 미미하다. 이는 Mamba가 모든 종류의 시계열 데이터에서 Transformer보다 우월한 것은 아니며, 데이터의 통계적 특성에 따라 선택적으로 적용되어야 함을 의미한다.
- **TD 인코딩의 단순성**: Ablation Study 결과, 시간적 의존성(TD)을 추출하는 데에는 Mamba나 Attention보다 단순한 FFN이 더 효과적이라는 점이 발견되었다. 이는 TSF에서 시간적 순서 정보가 생각보다 단순한 구조로도 충분히 학습될 수 있음을 시사한다.

### 3. 비판적 해석

저자들은 Mamba가 Transformer의 성능을 유지하면서 비용을 낮춘다고 주장하지만, 일부 데이터셋에서는 여전히 Transformer 기반 모델(특히 iTransformer)이 우위에 있다. 또한, 룩백 윈도우 길이를 늘렸을 때 성능이 정체되는 현상이 Mamba 블록 추가만으로는 완전히 해결되지 않았다는 점은, 단순한 아키텍처 교체보다는 시계열 데이터 자체의 특성을 반영한 인코딩 방식의 근본적인 개선이 필요함을 보여준다.

## 📌 TL;DR

본 논문은 Mamba의 Selective SSM을 시계열 예측에 적용한 **S-Mamba**를 제안한다. **Bidirectional Mamba로 변수 간 상관관계(VC)를, FFN으로 시간적 의존성(TD)을 포착**함으로써 Transformer 수준의 예측 성능을 내면서도 연산 복잡도를 선형 수준으로 낮추었다. 특히 변수가 많은 대규모 시계열 데이터에서 탁월한 효율성과 성능을 보이며, 이는 향후 대규모 센서 데이터나 복잡한 시스템의 실시간 예측 모델 설계에 중요한 기반이 될 것으로 기대된다.
