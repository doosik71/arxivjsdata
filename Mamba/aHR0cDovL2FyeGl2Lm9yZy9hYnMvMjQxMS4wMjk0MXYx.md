# A MAMBA FOUNDATION MODEL FOR TIME SERIES FORECASTING

Haoyu Ma, Yushu Chen, Wenlai Zhao, Jinzhe Yang, Yingsheng Ji, Xinghua Xu, Xiaozhu Liu, Hao Jing, Shengzhuo Liu, Guangwen Yang (2024)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 예측(Multivariate Time Series Forecasting)을 위한 효율적인 기초 모델(Foundation Model)을 구축하는 것을 목표로 한다. 기존의 시계열 기초 모델들은 주로 Transformer 아키텍처에 의존하고 있으며, 이는 입력 길이(Input length)가 증가함에 따라 계산 복잡도가 이차적으로 증가하는 Quadratic Complexity $O(L^2)$ 문제를 야기한다.

또한, 시계열 데이터는 도메인별 이질성이 매우 크고, NLP 분야의 텍스트 데이터에 비해 대규모 데이터를 수집하는 것이 훨씬 어렵다는 한계가 있다. 따라서 대규모 데이터셋에 대한 의존도를 낮추면서도, 긴 시퀀스를 효율적으로 처리하고 일반화 성능이 뛰어난 시계열 기초 모델의 개발이 절실한 상황이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **TSMamba 제안**: Mamba 아키텍처를 기반으로 하여 선형 복잡도(Linear Complexity)를 가지는 시계열 예측 기초 모델을 제안한다. 이는 다양한 도메인과 주파수의 예측 작업에 적용 가능하다.
2. **2단계 전이 학습(Two-stage Transfer Learning)**: 대규모로 사전 학습된 Mamba LLM의 지식을 활용하는 2단계 학습 프로세스를 도입하여, 상대적으로 적은 양의 시계열 데이터와 낮은 학습 비용으로도 효과적인 모델 구축을 가능하게 한다.
3. **압축된 채널 간 어텐션(Compressed Channel-wise Attention) 모듈**: 기본적으로 채널 독립적(Channel-Independent, CI)인 구조를 유지하면서도, 파인튜닝 단계에서 채널 간의 의존성을 효과적으로 추출할 수 있는 정규화된 어텐션 메커니즘을 도입하여 성능을 향상시켰다.
4. **SOTA 성능 달성**: 제로샷(Zero-shot) 및 풀샷(Full-shot) 예측 시나리오 모두에서 기존의 최신 모델들과 경쟁하거나 이를 능가하는 성능을 입증하였다.

## 📎 Related Works

시계열 예측은 전통적인 통계 방식(ARIMA 등)에서 RNN, CNN, GNN, MLP, 그리고 Transformer 기반의 딥러닝 모델로 발전해 왔다.

- **Transformer 기반 모델**: Long-term dependency를 포착하는 데 뛰어나지만, 앞서 언급한 계산 복잡도 문제와 시계열의 시간적 순서라는 귀납적 편향(Inductive Bias)이 부족하다는 단점이 있다. PatchTST와 같은 모델이 패칭(Patching) 기법으로 계산량을 줄이려 했으나, 이론적인 이차 복잡도는 여전히 존재한다.
- **시계열 기초 모델**: 최근 LLM을 시계열 작업에 적응시키거나 대규모 데이터셋으로 직접 사전 학습시키는 시도가 이어지고 있다. 그러나 대부분 Transformer 기반이어서 효율성 문제가 여전하다.
- **State Space Models (SSMs) 및 Mamba**: SSM은 RNN과 CNN의 장점을 결합하여 선형 복잡도를 제공한다. 특히 Mamba는 입력 값에 따라 매개변수가 결정되는 선택적 메커니즘(Selective Mechanism)을 통해 콘텐츠 기반 추론(Content-based reasoning)을 가능하게 하며, 하드웨어 최적화 알고리즘을 통해 실제 계산 효율을 극대화하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

TSMamba는 입력 데이터를 전처리한 후, **Forward Mamba Encoder**와 **Backward Mamba Encoder**를 통해 시간적 의존성을 양방향으로 추출한다. 추출된 표현(Representation)은 결합된 후 예측 헤드(Prediction Head)를 통해 최종 예측값으로 매핑된다.

### 2. 주요 구성 요소 및 절차

**가. 전처리 및 임베딩 (Preprocessing & Embedding)**

- **Normalization**: 데이터의 분포 변화에 대응하기 위해 Reverse Instance Normalization을 적용한다.
- **Input Embedding**: 1D Convolution 레이어를 사용하여 시계열 데이터를 패치(Patch) 단위로 임베딩한다. 이는 개별 시점의 값보다 지역적 세밀함(Local semantic information)을 더 잘 포착하며, 인코더로 들어가는 입력 길이를 줄여준다.

**나. Mamba 백본 (Backbone)**

- **구조**: Forward/Backward Mamba 엔코더는 Mamba 블록, RMSNorm, 잔차 연결(Residual connection)로 구성된다.
- **수학적 원리**: Mamba는 다음과 같은 연속 시스템에서 시작한다.
    $$h'(t) = Ah(t) + Bx(t)$$
    $$y(t) = Ch(t)$$
    이를 Zero-Order Hold (ZOH) 규칙을 통해 이산화하여 계산하며, $\bar{A} = \exp(\Delta A)$와 같은 형태로 변환된다. Mamba의 핵심은 $B, C, \Delta$를 입력 $x$의 함수로 만들어 데이터에 따라 정보를 선택적으로 유지하거나 버리는 것이다.
- **양방향 처리**: Forward 엔코더는 인과적(Causal) 관계를, Backward 엔코더는 역방향 시간 관계를 추출한다. Backward의 출력은 다시 뒤집힌(Flip) 후 temporal convolution을 통해 Forward 표현과 정렬된다.

**다. 예측 헤드 (Prediction Head)**

- 모델 차원이 매우 크기 때문에 모든 패치를 직접 매핑하면 오버피팅 위험이 있다. 따라서 선형 투영과 GELU 활성화 함수를 통해 차원을 먼저 압축한 후, 작은 선형 헤드를 통해 타겟 윈도우를 한 번에 예측하는 방식을 사용한다.
- **손실 함수**: 이상치(Outlier)에 강건한 **Huber Loss**를 사용한다.

**라. 2단계 전이 학습 (Two-stage Transfer Learning)**

1. **1단계 (Refining Backbone)**: Mamba-130M 사전 학습 모델을 초기값으로 사용한다. 자기회귀(Autoregressive) 예측 또는 백캐스팅(Backcasting) 작업을 통해 입력 임베딩과 백본을 시계열 도메인에 맞게 조정한다.
2. **2단계 (Training Prediction Head)**: 원래의 TSMamba 구조를 복원하고, 랜덤 초기화된 예측 헤드와 정렬 모듈을 학습시킨다. 이때 기존 백본은 작은 학습률로 업데이트하여 사전 학습된 지식을 보존한다.

**마. 압축된 채널 간 어텐션 (Compressed Channel-wise Attention)**
다변량 데이터에서 채널 간 관계를 추출하기 위해 도입된 모듈이다.

- **절차**: $\text{Temporal Conv (정렬)} \rightarrow \text{Linear Projection (채널 수 } D \text{를 } \lceil \log_2(D) \rceil \text{로 압축)} \rightarrow \text{Attention} \rightarrow \text{Linear Expansion (원래 채널 수 } D \text{로 복구)} \rightarrow \text{Residual Addition}$
- 이러한 압축 과정은 채널 간 관계 모델링 시 발생할 수 있는 노이즈를 필터링하고 오버피팅을 방지하는 정규화 역할을 수행한다.

## 📊 Results

### 1. 실험 설정

- **비교 대상**: Zero-shot(Time-MoE, Moirai, TimesFM, Chronos 등), Full-shot(PatchTST, Autoformer, GPT4TS, MambaTS 등) 총 16개 모델과 비교하였다.
- **데이터셋 및 지표**: ETTm2, Weather, ILI 데이터셋을 사용하였으며, MSE(Mean Squared Error)와 MAE(Mean Absolute Error)를 평가지표로 사용하였다.
- **설정**: Encoder layers 3개, Embedding size 768, Input length 512로 설정하였다.

### 2. 주요 결과

- **Zero-shot Forecasting**: TSMamba는 매우 적은 학습 데이터를 사용했음에도 불구하고 SOTA 기초 모델들과 대등한 성능을 보였다. 특히 예측 길이(Horizon)가 긴 경우(336, 720)에 더 강점을 보였다.
- **Full-shot Forecasting**: 파인튜닝을 통한 예측에서 TSMamba는 대다수 데이터셋과 예측 길이에서 최상위 성능을 기록하였다. 특히 GPT2 기반의 GPT4TS 대비 평균 15%의 성능 향상을 보였으며, 작업 특화 모델인 PatchTST보다도 우수한 성능을 나타냈다.

## 🧠 Insights & Discussion

**강점 및 통찰**

- **데이터 효율성**: 거대한 데이터셋으로 학습된 다른 기초 모델들에 비해 훨씬 적은 데이터를 사용하고도 경쟁력 있는 성능을 냈다는 점은 Mamba 기반의 전이 학습이 매우 효율적임을 시사한다.
- **채널 독립성 vs 의존성**: 시계열 모델에서 채널 독립적(CI) 접근 방식이 강력한 베이스라인이 되는 이유는 데이터 부족으로 인한 오버피팅 위험 때문이다. 본 논문은 채널 차원을 $\log_2$ 수준으로 압축하여 어텐션을 적용함으로써, 계산 비용을 줄이면서도 효과적으로 채널 간 관계를 학습할 수 있음을 보여주었다.

**한계 및 비판적 해석**

- **평가 데이터셋의 제한**: 현재 보고서에서는 ETTm2, Weather, ILI 등 일부 데이터셋에 대해서만 결과가 제시되어 있다. 기초 모델로서의 범용성을 완전히 입증하기 위해서는 더 다양하고 방대한 벤치마크 데이터셋에 대한 검증이 필요하다.
- **Work in Progress**: 논문 스스로가 '진행 중인 작업'임을 명시하고 있어, 최종적인 하이퍼파라미터 최적화나 추가적인 아키텍처 개선 가능성이 남아 있다.

## 📌 TL;DR

본 연구는 Transformer의 이차 복잡도 문제를 해결하기 위해 **Mamba 아키텍처 기반의 시계열 기초 모델 TSMamba**를 제안한다. LLM의 지식을 활용하는 **2단계 전이 학습**과 오버피팅을 방지하는 **압축된 채널 어텐션** 메커니즘을 통해, 적은 데이터로도 높은 효율성과 예측 정확도를 달성하였다. 특히 제로샷과 풀샷 상황 모두에서 SOTA급 성능을 보여, 향후 다양한 도메인의 시계열 예측 작업에 효율적으로 적용될 가능성이 높다.
