# ms-Mamba: Multi-scale Mamba for Time-Series Forecasting

Yusuf Meric Karadag, Ipek Gursel Dino, and Sinan Kalkan (2025)

## 🧩 Problem to Solve

시계열 예측(Time-series Forecasting, TSF)의 핵심 문제는 과거의 데이터를 바탕으로 미래의 값을 정확하게 예측하는 것이다. 기존의 recurrent 구조, Transformer 기반 모델, 그리고 최근 제안된 Mamba 기반 아키텍처들은 대부분 입력을 단일 시간 척도(single temporal scale)에서 처리하는 경향이 있다.

하지만 실제 시계열 데이터는 여러 시간 척도에 걸쳐 정보가 변화하는 다중 척도(multi-scale) 특성을 가지고 있다. 단일 척도 처리 방식은 이러한 복잡한 신호를 포착하는 데 한계가 있으며, 이는 많은 예측 작업에서 최적의 성능을 내지 못하는 원인이 된다. 따라서 본 논문의 목표는 Mamba 아키텍처를 확장하여 다양한 시간 척도에서 신호를 동시에 처리할 수 있는 Multi-scale Mamba(ms-Mamba)를 제안함으로써 예측 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 State Space Models(SSMs)의 학습 가능한 샘플링 레이트(sampling rate, $\Delta$)의 유연성을 활용하는 것이다. 

단일 Mamba 블록 대신, 서로 다른 샘플링 레이트를 가진 여러 개의 Mamba 블록을 병렬로 배치하여 입력 신호를 다양한 시간 해상도에서 동시에 처리하도록 설계하였다. 이를 통해 모델이 단기적인 변동성과 장기적인 추세를 동시에 효과적으로 캡처할 수 있도록 하였다. 또한, 이러한 샘플링 레이트를 결정하기 위한 세 가지 전략(고정 하이퍼파라미터 기반, 개별 학습 기반, 입력 기반 동적 추정)을 제시하고 비교 분석하였다.

## 📎 Related Works

### 1. Transformer-based Models
self-attention 메커니즘을 통해 장기 의존성을 잘 포착하며, Informer, Autoformer, PatchTST 등 다양한 변형 모델이 제안되었다. 그러나 attention 메커니즘의 이차 복잡도($O(L^2)$)로 인해 긴 시퀀스 처리 시 계산 비용과 메모리 사용량이 급증하며, 특히 강한 계절성 패턴이 있는 경우 시간적 의존성을 감지하는 데 어려움이 있다.

### 2. Linear Models
MLP 기반의 단순한 구조로 효율성이 높으며, DLinear나 RLinear 같은 모델들이 제안되었다. 일부 데이터셋에서는 Transformer보다 높은 성능을 보이기도 하지만, 비선형 의존성 처리 능력이 부족하고 글로벌 의존성을 캡처하는 능력이 떨어져 매우 긴 입력 시퀀스가 필요하다는 단점이 있다.

### 3. Mamba Models
최근 제안된 Mamba는 SSM을 기반으로 하여 Transformer 수준의 성능을 유지하면서도 선형 복잡도($O(L)$)로 동작한다. TSF 분야에서는 S-Mamba가 제안되어 양방향 Mamba 레이어를 통해 변수 간 상관관계를 캡처함으로써 SOTA 성능을 달성한 바 있다. ms-Mamba는 이러한 S-Mamba의 단일 척도 처리 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
ms-Mamba의 전체 파이프라인은 **Embedding $\rightarrow$ Multi-scale Mamba Layers (Bidirectional) $\rightarrow$ Normalization $\rightarrow$ Feed-Forward Network $\rightarrow$ Linear Projection** 순으로 구성된다.

### 2. 상세 구성 요소 및 절차

**A. Embedding Layer**
입력 시퀀스 $X \in \mathbb{R}^{L \times D}$ (L: 시간 단계, D: 변수 수)를 선형 변환을 통해 고정된 임베딩 차원 $D_e$로 매핑한다.
$$E = \text{Embedding}(X) \in \mathbb{R}^{D_e \times D}$$
이를 통해 가변적인 입력 길이 $L$ 대신 고정된 길이의 토큰을 처리할 수 있게 한다.

**B. Multi-scale Mamba Layer**
본 방법론의 핵심으로, 서로 다른 샘플링 레이트 $\Delta_1, \Delta_2, \dots, \Delta_n$을 가진 $n$개의 Mamba 블록을 병렬로 운용한다. 각 블록의 출력은 평균(Average) 연산을 통해 통합된다.
$$E^l_m = \text{Avg}(\text{Mamba}(E^l; \Delta_1), \dots, \text{Mamba}(E^l; \Delta_n))$$
여기서 $\Delta_i$를 결정하는 세 가지 전략은 다음과 같다:
1. **Fixed temporal scales**: $\Delta_1$만 학습하고, 나머지는 하이퍼파라미터 $\alpha_i$를 곱해 결정한다. ($\Delta_i = \alpha_i \times \Delta_1$)
2. **Learnable temporal scales**: 모든 $\Delta_i$를 독립적인 학습 가능 변수로 설정한다.
3. **Dynamic temporal scales**: MLP를 통해 입력 임베딩 $E^l$로부터 $\Delta_i$를 직접 추정한다.
   $$\Delta_i = \text{MLP}(\text{Flatten}(E^l))$$

**C. 후처리 및 예측**
Multi-scale Mamba 레이어의 출력 $E^l_m$은 Layer Normalization과 ReLU 활성화 함수가 포함된 MLP를 거쳐 다음 레이어로 전달된다. 최종 레이어의 출력 $E^L$은 선형 투영(Linear Projection)을 통해 최종 예측값 $\hat{y} \in \mathbb{R}^{F \times D}$로 변환된다.
$$\hat{y} = \text{Linear}(E^L)$$

### 3. 훈련 목표 및 손실 함수
모델은 예측값 $\hat{y}$와 실제값 $y$ 사이의 평균 제곱 오차(Mean Squared Error, MSE)를 최소화하도록 학습된다.
$$\mathcal{L} = \frac{1}{F \times D} \sum_{i=1}^{F} \sum_{j=1}^{D} (\hat{y}_{i,j} - y_{i,j})^2$$
최적화 알고리즘으로는 Adam optimizer가 사용된다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Traffic, PEMS(03, 04, 07, 08), ETT(m1, m2, h1, h2), Electricity, Exchange, Weather, Solar-Energy 등 총 13개의 벤치마크 데이터셋을 사용하였다.
- **비교 대상**: S-Mamba, iTransformer, PatchTST, Crossformer, FEDformer, Autoformer, TiDE, DLinear, RLinear, TimesNet 등 10개의 SOTA 모델과 비교하였다.
- **평가 지표**: MSE를 사용하였다.

### 2. 주요 결과
- **정량적 성능**: Traffic 관련 데이터셋과 ETT 데이터셋 전반에서 ms-Mamba(고정 및 학습 가능 척도 버전)가 대부분 최상위 또는 차상위 성능을 기록하였다. 특히 S-Mamba 대비 유의미한 성능 향상을 보였다.
- **특이 사항**: Solar-Energy 데이터셋에서 ms-Mamba는 MSE 0.229를 기록하여, 가장 강력한 경쟁 모델인 S-Mamba(0.240)를 능가하였다.
- **Ablation Study**: $\alpha = (1, 2, 4, 8)$ 조합의 고정 척도 방식과 학습 가능 척도 방식이 우수한 성능을 보였으며, 특히 학습 가능 척도 방식이 하이퍼파라미터 튜닝 부담이 적어 더 선호될 수 있음을 확인하였다.
- **효율성 분석**: ETTh2 및 Solar Energy 데이터셋에서 ms-Mamba는 S-Mamba보다 더 적은 파라미터 수, 메모리 점유율, 그리고 연산량(MACs)으로 더 높은 성능을 달성하였다. 다만, 변수 수가 매우 많은 Traffic 데이터셋에서는 성능은 향상되었으나 파라미터 및 연산량 감소 효과는 나타나지 않았다.

## 🧠 Insights & Discussion

### 1. 강점
ms-Mamba는 SSM의 샘플링 레이트라는 특성을 다중 척도 분석 관점에서 재해석하여, 복잡한 시계열 데이터의 다중 해상도 특성을 효과적으로 캡처하였다. 특히 기존 Mamba 기반 모델보다 효율적인 자원 사용으로 더 나은 예측 성능을 낼 수 있음을 입증하였다.

### 2. 한계 및 논의사항
- **하이퍼파라미터 의존성**: 고정 척도 방식의 경우 $\alpha$ 계수 선정에 따라 성능 차이가 발생하므로 적절한 하이퍼파라미터 탐색이 필요하다.
- **데이터 차원에 따른 효율성**: 변수의 개수($D$)가 매우 많은 데이터셋에서는 파라미터 효율성이 떨어지는 모습이 관찰되었다. 이는 다중 척도 블록의 병렬 배치가 변수 차원이 높을 때 메모리 및 연산 오버헤드를 증가시키기 때문으로 해석된다.
- **범용성**: 본 연구는 TSF에 집중하였으나, 저자들은 이러한 다중 척도 Mamba 구조가 텍스트나 이미지 등 다른 모달리티에도 적용 가능할 것으로 전망하고 있다.

## 📌 TL;DR

본 논문은 시계열 데이터의 다중 척도 특성을 포착하기 위해, 서로 다른 샘플링 레이트($\Delta$)를 가진 Mamba 블록들을 병렬로 구성한 **ms-Mamba**를 제안한다. 실험 결과, ms-Mamba는 기존의 Transformer 및 Mamba 기반 SOTA 모델들보다 우수한 예측 성능을 보였으며, 많은 경우 더 적은 파라미터와 연산량으로 효율적인 예측이 가능함을 증명하였다. 이는 향후 효율적인 시퀀스 모델링 연구 및 타 도메인 확장 연구에 중요한 기초가 될 것으로 보인다.