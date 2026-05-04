# DTMamba : Dual Twin Mamba for Time Series Forecasting

Zexue Wu, Yifeng Gong, and Aoqian Zhang (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 시계열 데이터의 장기 예측(Long-term Time Series Forecasting, LTSF)에서 발생하는 연산 효율성과 예측 정확도 사이의 트레이드오프이다. 

전통적인 통계 기반 방법이나 RNN 기반 방법은 복잡한 비선형 관계를 포착하는 데 한계가 있으며, 특히 RNN은 기울기 소실(vanishing gradient) 문제와 순차적 처리로 인한 병렬화의 어려움이 있다. Transformer 기반 모델은 self-attention 메커니즘을 통해 장기 의존성(long-term dependencies)을 효과적으로 포착하여 높은 성능을 보이지만, 입력 시퀀스 길이에 대해 이차 복잡도(quadratic complexity)를 가지므로 연산 비용이 매우 크다는 치명적인 단점이 있다. 반면 DLinear와 같은 선형 모델은 효율적이지만, 변동성이 크거나 비주기적인 패턴을 가진 데이터에서는 Transformer 기반 모델보다 성능이 떨어진다.

따라서 본 논문의 목표는 Transformer의 높은 예측 성능과 선형 모델의 효율성을 동시에 달성할 수 있는 새로운 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 시퀀스 모델링에서 주목받는 Mamba(Selective State Space Model)를 시계열 예측에 도입하고, 이를 확장한 **Dual Twin Mamba (DTMamba)** 구조를 설계한 것이다.

가장 중점적인 설계 아이디어는 **TMamba Block**이라 명명된 구조를 두 개 배치하는 것이다. 각 TMamba Block 내부에는 두 개의 Mamba 모델이 병렬로 연결된 'Twin Mamba' 구조가 포함되어 있다. 이는 서로 다른 수준의 특성(저수준의 시간적 특성과 고수준의 시간적 패턴 및 관계)을 동시에 학습함으로써 모델의 표현 능력을 극대화하려는 의도이다. 또한, 채널 독립성(Channel Independence) 전략과 Reversible Instance Normalization(RevIN)을 결합하여 시계열 데이터 특유의 분포 변화(distribution shift) 문제와 과적합 문제를 해결하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 한계를 지적한다.
- **RNN-based methods**: 기울기 소실 문제로 인해 장기 의존성 포착이 어렵고 병렬 연산이 불가능하여 효율성이 낮다.
- **Transformer-based methods**: 성능은 우수하지만, self-attention의 이차 복잡도로 인해 긴 시퀀스를 처리할 때 메모리와 계산 시간이 기하급수적으로 증가한다.
- **Linear-based methods**: 연산 효율은 매우 높으나, 고변동성 데이터나 비주기적 패턴을 모델링하는 능력이 부족하여 성능적 한계가 존재한다.
- **TCN-based methods**: Dilated convolution 등을 사용하여 수용 영역을 넓혔음에도 불구하고, 여전히 장기 의존성을 모델링하는 능력이 제한적이다.

본 논문은 이러한 한계를 극복하기 위해 RNN의 순차 처리 능력과 CNN의 전역 정보 처리 능력을 결합하고, 선택적 메커니즘(selection mechanism)을 통해 불필요한 정보를 필터링하는 Mamba 아키텍처를 LTSF 분야에 적용하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 파이프라인
DTMamba의 전체 구조는 크게 세 단계의 층으로 구성된다: **Channel Independence Layer $\rightarrow$ TMamba Blocks $\rightarrow$ Projection Layer**. 데이터는 입력 전 RevIN을 통해 정규화되며, 예측 후에는 역정규화 과정을 거친다.

### 주요 구성 요소 및 역할
1.  **Normalization (RevIN)**: 입력 데이터 $X$를 $X^0 = \text{RevIN}(X)$로 정규화하여 시계열 데이터의 분포 변화 문제를 완화한다.
2.  **Channel Independence**: 다변량 시계열 데이터를 각 채널별로 독립적으로 처리하기 위해 형태를 변환한다. 입력 형태 $(B, T, D)$를 $(B \times D, 1, T)$로 변경하여 각 채널이 개별적인 시퀀스로 취급되게 함으로써 과적합을 방지한다.
3.  **TMamba Block**: 모델의 핵심 연산 단위로, 다음의 순서로 구성된다.
    - **Embedding**: 선형 층을 사용하여 시계열 데이터를 차원 $n_i$로 임베딩한다.
    - **Residual Connection**: FC(Fully Connected) 층을 통해 차원을 맞춘 후 잔차 연결을 추가하여 기울기 소실을 방지하고 학습 안정성을 높인다.
    - **Dropout**: 과적합 방지를 위해 Mamba 층 진입 전 적용한다.
    - **Twin Mambas**: 두 개의 Mamba 모델이 병렬로 배치되어 저수준 및 고수준의 시간적 특징을 동시에 추출하고, 그 결과를 합산한다.
4.  **Projection Layer**: TMamba Block을 통해 얻은 은닉 정보를 FC 층에 통과시켜 최종 예측 길이 $H$에 맞는 예측값 $X^P$를 생성한다.

### 수학적 배경 (Mamba/SSM)
Mamba는 상태 공간 모델(State Space Model, SSM)을 기반으로 한다. 연속 시간 영역에서 입력 $x(t)$를 출력 $y(t)$로 매핑하는 과정은 다음과 같다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

실제 이산 데이터 처리를 위해 $\Delta$라는 파라미터를 사용하여 이산화(discretization)를 수행하며, 이산화된 파라미터 $\bar{A}, \bar{B}$는 다음과 같이 정의된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\exp(\Delta A) - I)\Delta A^{-1}B$$

이산화된 시스템의 재귀 식은 다음과 같으며, 이는 RNN과 유사한 형태로 동작하여 효율적인 추론을 가능하게 한다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

## 📊 Results

### 실험 설정
- **데이터셋**: Weather, Traffic, Electricity, Exchange 및 ETT 시리즈(ETTh1, ETTh2, ETTm1, ETTm2) 등 6종의 실제 데이터셋을 사용하였다.
- **지표**: MSE(Mean Squared Error)와 MAE(Mean Absolute Error)를 사용하여 성능을 측정하였다.
- **비교 대상**: Autoformer, PatchTST, iTransformer 등 Transformer 계열, DLinear, TiDE 등 선형 계열, TimesNet 등 TCN 계열을 포함한 11개의 SOTA 모델과 비교하였다.

### 주요 결과
- **정량적 결과**: 대부분의 데이터셋과 예측 길이($H \in \{96, 192, 336, 720\}$)에서 DTMamba가 가장 낮은 MSE와 MAE를 기록하며 최우수 성능을 보였다. 특히 iTransformer가 두 번째로 좋은 성능을 보였으며, Traffic 및 Electricity와 같은 고차원 데이터셋에서는 iTransformer와 매우 근소한 차이를 보였다.
- **정성적 결과**: 시각화 분석 결과, DTMamba의 예측 곡선이 ground-truth에 가장 근접하게 추종하는 것을 확인하였다.
- **소거 실험(Ablation Study)**: 
    - 잔차 연결(Residual Connection)과 채널 독립성(Channel Independence)을 제거했을 때 예측 오차가 유의미하게 증가함을 확인하여 두 모듈의 필요성을 입증하였다.
    - 단일 Mamba나 단일 TMamba Block을 사용했을 때보다 두 개의 TMamba Block을 사용한 DTMamba 구조가 가장 뛰어난 성능을 보였다.
- **민감도 분석**: Dropout 비율, 선형 층의 차원, Mamba의 확장 계수($e\_fact$), 상태 확장 계수($d\_state$) 등에 대해 실험한 결과, 제안된 기본 설정값이 전반적으로 안정적인 성능을 제공함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 Mamba 아키텍처가 시계열 예측에서도 Transformer를 대체할 수 있는 강력한 후보임을 입증하였다. 특히 병렬 구조의 Twin Mamba 설계를 통해 다각적인 시간적 특징을 추출한 점이 성능 향상의 핵심으로 판단된다.

**강점**: 
- Transformer의 이차 복잡도 문제를 해결하면서도 SOTA 수준의 정확도를 달성하였다.
- 채널 독립성과 RevIN을 적절히 결합하여 시계열 데이터의 특성을 잘 반영하였다.

**한계 및 논의**:
- 실험 결과에서 Traffic과 Electricity 데이터셋(상대적으로 차원이 높은 데이터)에서는 iTransformer와 성능 차이가 크지 않거나 때로는 밀리는 경향이 있다. 이는 Mamba 구조가 매우 높은 차원의 변수 간 상호작용을 포착하는 데 있어 iTransformer의 Inverted 구조보다 효율성이 떨어질 수 있음을 시사한다.
- 논문 내에서 Mamba의 파라미터(예: $d\_state$)에 대한 민감도가 낮다고 언급하였으나, 특정 예측 길이($S=720$)에서는 일부 변동이 관찰되었다. 이는 데이터셋의 특성과 예측 길이에 따라 최적의 SSM 파라미터가 달라질 수 있음을 의미한다.

## 📌 TL;DR

본 논문은 시계열 예측의 연산 효율성과 정확도를 동시에 잡기 위해 **Selective State Space Model(Mamba)** 기반의 **DTMamba**를 제안한다. 핵심은 두 개의 **TMamba Block**을 배치하고, 각 블록 내에 두 개의 Mamba를 병렬로 구성하여 저수준/고수준 특징을 동시에 학습하는 것이다. 실험 결과, 기존의 Transformer 및 선형 기반 모델들보다 우수한 성능을 보였으며, 특히 긴 시퀀스 처리에서 선형 복잡도의 효율성을 확보하였다. 이 연구는 향후 거대 시계열 데이터셋의 효율적인 학습 및 예측을 위한 새로운 아키텍처 방향성을 제시한다.