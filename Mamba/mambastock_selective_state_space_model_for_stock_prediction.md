# MambaStock: Selective state space model for stock prediction

Zhuangwei Shi (2024)

## 🧩 Problem to Solve

본 논문은 주식 시장의 복잡한 변동성으로 인해 발생하는 주가 예측의 어려움을 해결하고자 한다. 주식 시장은 경제 발전에 핵심적인 역할을 하지만, 그 가격 움직임이 매우 불규칙하고 비선형적(nonlinearity)이기 때문에 투자자들이 리스크를 관리하는 데 큰 어려움을 겪는다.

기존의 전통적인 시계열 모델인 $\text{ARIMA}$($\text{Autoregressive Integrated Moving Average}$)는 비선형 시계열 데이터를 효과적으로 묘사하지 못하며, 모델링 전 많은 전제 조건이 필요하다는 한계가 있다. 이를 극복하기 위해 신경망 기반의 모델들이 도입되었으나, 여전히 시계열 데이터의 복잡한 의존성을 효율적으로 캡처하는 것은 도전적인 과제이다. 따라서 본 연구의 목표는 최신 시퀀스 모델링 기법인 $\text{Mamba}$를 주가 예측에 적용하여, 복잡한 전처리나 수작업 특징 추출(handcrafted features) 없이도 높은 정확도로 미래 주가를 예측하는 $\text{MambaStock}$ 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 선택적 상태 공간 모델(Selective State Space Model)인 $\text{Mamba}$의 $\text{S6}$ 구조를 주가 예측 시계열 데이터에 적용하는 것이다. $\text{Mamba}$의 중심적인 직관은 다음과 같다.

1. **선택 메커니즘(Selection Mechanism)**: 입력 데이터에 따라 모델의 파라미터가 동적으로 변하게 함으로써, 시퀀스 내에서 예측에 중요한 정보는 집중적으로 반영하고 불필요한 노이즈는 무시할 수 있게 한다.
2. **선형 시간 복잡도(Linear-time Complexity)**: $\text{Transformer}$의 $\text{Attention}$ 메커니즘이 가진 이차 시간 복잡도 문제를 해결하여, 대규모 시퀀스 데이터를 효율적으로 처리하면서도 긴 의존성(long-term dependency)을 캡처한다.
3. **스캔 모듈(Scan Module)**: 상태 공간을 효율적으로 스캔하여 예측에 필요한 관련 정보를 빠르게 식별한다.

## 📎 Related Works

논문에서는 주가 예측을 위해 사용된 기존의 접근 방식들을 다음과 같이 설명한다.

- **전통적 모델 및 하이브리드 모델**: $\text{ARIMA}$는 선형성 가정으로 인해 한계가 있으며, 이를 보완하기 위해 $\text{ARIMA-NN}$과 같은 신경망 결합 모델이 제안되었다.
- **순환 신경망(RNN) 및 $\text{LSTM}$**: $\text{LSTM}$은 게이트 메커니즘을 통해 유용한 정보를 기억하고 불필요한 정보를 망각함으로써 시퀀스 데이터를 처리하지만, 매우 긴 시퀀스의 의존성을 처리하는 데 한계가 있다.
- **$\text{Attention}$ 및 $\text{Transformer}$**: $\text{Attention}$ 메커니즘과 $\text{Transformer}$는 전역적인 의존성을 캡처하는 데 탁월하며, $\text{BERT}$와 같은 사전 학습 모델을 통해 성능을 높였다.
- **상태 공간 모델(SSM)**: $\text{Kalman Filter}$와 같은 전통적 $\text{SSM}$이 사용되었으며, 최근에는 구조화된 상태 공간 모델인 $\text{S4}$가 등장하여 효율적인 시퀀스 모델링의 가능성을 열었다.

$\text{MambaStock}$은 이러한 기존 모델들이 가진 비선형성 처리 능력의 부족이나 계산 복잡도 문제를 $\text{Mamba}$의 선택적 $\text{SSM}$ 구조를 통해 해결함으로써 차별점을 갖는다.

## 🛠️ Methodology

### 1. Structured State Space Sequence Model ($\text{S4}$)

$\text{S4}$는 상미분 방정식(ODE)의 풀이 방식에서 영감을 얻었으며, 시퀀스의 역학을 상태 공간 표현으로 나타낸다. 기본 방정식은 다음과 같다.

$$h_t = Ah_{t-1} + Bx_t$$
$$y_t = Ch_t$$

여기서 $x_t$는 입력, $h_t$는 은닉 상태, $y_t$는 출력이다. 연속 시간 모델을 이산화(discretization)하기 위해 샘플 시간 $\Delta$를 도입하며, 이산화된 파라미터 $\bar{A}$와 $\bar{B}$는 다음과 같이 계산된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

최종적인 이산 상태 방정식은 다음과 같다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$

### 2. Mamba ($\text{S6}$)

$\text{Mamba}$는 $\text{S4}$의 파라미터 $A, B, C, \Delta$가 시간에 따라 불변(time-invariant)이라는 점을 개선하여, 입력 $X$에 따라 동적으로 변하는 **선택 메커니즘**을 도입했다.

- **동적 파라미터**: $B, C, \Delta$가 입력 $X$로부터 완전 연결 층(fully-connected layers)을 통해 학습된다.
- **이산 상태 방정식**:
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
$$y_t = C_t h_t$$

### 3. MambaStock 아키텍처 및 학습 절차

$\text{MambaStock}$은 $\text{Mamba}$ 프레임워크를 사용하여 미래의 주가 변동률(movement rate)을 예측한다.

- **입력 데이터**: 시가, 고가, 저가, 거래량, 거래액, 회전율, 거래비율, $\text{PE}$, $\text{PB}$, $\text{PS}$, 총 주식 수, 유통 주식 수, 자유 유통 주식 수, 총 시가총액, 유통 시가총액 등 총 15가지의 금융 지표를 입력으로 사용한다.
- **모델 구조**:
  - 입력 데이터 $\rightarrow \text{Mamba}$ 모델 ($N=16$) $\rightarrow$ 1차원 표현으로 축소 $\rightarrow \tanh$ 활성화 함수 $\rightarrow$ 예측 변동률 출력.
  - $\tanh$ 함수를 사용하는 이유는 주가 변동률의 범위를 $(-1, 1)$ 사이로 제한하기 위함이다.
- **손실 함수**: 예측값과 실제값 사이의 평균 제곱 오차인 $\text{MSE}$($\text{Mean Squared Error}$)를 사용한다.
$$\text{MSE} = \frac{1}{n} \sum_{t=1}^{n} (\hat{X}_t - X_t)^2$$
- **학습 설정**: $\text{Adam}$ 옵티마이저를 사용하며, 학습 횟수는 100 $\text{epoch}$, 학습률(learning rate)은 0.01로 설정하였다. 하드웨어는 $\text{NVIDIA GTX 3060 (12GB)}$를 사용하였다.

## 📊 Results

### 1. 실험 설정

- **대상 데이터**: 중국 주식 시장의 4개 은행 주식(중국 상업은행 600036.SH, 중국 농업은행 601288.SH, 교통은행 601328.SH, 중국은행 601988.SH)을 대상으로 하였다.
- **데이터셋**: $\text{Tushare}$ 오픈 데이터셋을 사용하였으며, 테스트 세트의 크기는 300으로 고정하였다.
- **비교 모델 (Baselines)**: $\text{KF}$, $\text{ARIMA}$, $\text{ARIMA-NN}$, $\text{XGBoost}$, $\text{LSTM}$, $\text{BiLSTM}$, $\text{Transformer}$, $\text{TL-KF}$, $\text{AttCLX}$.
- **평가 지표**: $\text{MSE}$, $\text{RMSE}$, $\text{MAE}$, $R^2$.

### 2. 정량적 결과 분석

실험 결과, $\text{MambaStock}$은 대부분의 지표에서 기존 모델들보다 우수한 성능을 보였다.

- **$R^2$ 지표**: 특히 $R^2$ 값에서 $\text{MambaStock}$이 가장 높은 수치를 기록하며, 실제 주가 움직임을 가장 잘 설명하고 있음을 보여준다. (예: 601988.SH에서 $\text{MambaStock}$의 $R^2$는 0.9590으로 최상위 수준임)
- **오차 지표**: $\text{MSE}$, $\text{RMSE}$, $\text{MAE}$ 모두에서 $\text{MambaStock}$이 경쟁 모델들 대비 낮거나 매우 경쟁력 있는 수치를 기록하였다.
- **비교 분석**:
  - **전통적 모델 대비**: $\text{ARIMA}$나 $\text{KF}$보다 비선형 패턴 캡처 능력이 뛰어나 월등한 성능을 보였다.
  - **딥러닝 모델 대비**: $\text{LSTM}$, $\text{Transformer}$보다 시계열의 복잡한 의존성을 더 잘 추출하여 더 높은 정확도를 달성했다.
  - **하이브리드 모델 대비**: $\text{TL-KF}$나 $\text{AttCLX}$보다 시간적 의존성과 복잡한 패턴을 동시에 더 효과적으로 포착하였다.

## 🧠 Insights & Discussion

### 1. 강점

$\text{MambaStock}$의 가장 큰 강점은 $\text{Mamba}$의 **선택적 상태 공간 모델(S6)** 구조를 통해 주가 데이터의 극심한 비선형성과 변동성을 효과적으로 다루었다는 점이다. 별도의 복잡한 특징 공학(feature engineering) 없이도 원시 금융 지표만으로 높은 예측력을 보인 것은 모델의 표현 학습 능력이 매우 뛰어남을 시사한다. 또한, $\text{Transformer}$의 연산 효율성 문제를 해결하면서도 유사하거나 더 나은 성능을 냈다는 점이 고무적이다.

### 2. 한계 및 비판적 해석

- **데이터셋의 규모**: 실험이 중국의 4개 은행 주식이라는 매우 제한적인 데이터셋에서 이루어졌다. 주식 시장의 다양한 섹터(IT, 바이오 등)나 다른 국가의 시장에서도 동일한 일반화 성능이 나타날지는 명시되지 않았다.
- **하이퍼파라미터 분석 부재**: $N=16$이라는 설정이나 학습률 $0.01$ 등이 어떤 근거로 선택되었는지에 대한 Ablation Study가 부족하다.
- **예측 대상의 단순함**: 주가 '변동률'을 예측하는 과제에 집중되어 있는데, 실제 투자에서는 변동률뿐만 아니라 절대 가격이나 추세의 전환점을 맞추는 것이 중요하므로 이에 대한 분석이 추가될 필요가 있다.

## 📌 TL;DR

본 논문은 최신 선택적 상태 공간 모델인 $\text{Mamba}$를 주가 예측에 도입한 $\text{MambaStock}$을 제안한다. 이 모델은 $\text{S6}$ 메커니즘을 통해 시계열 데이터의 비선형성을 효과적으로 학습하며, $\text{Transformer}$나 $\text{LSTM}$ 기반 모델보다 뛰어난 예측 정확도($R^2$ 향상 및 오차 감소)를 달성하였다. 이 연구는 금융 시계열 예측 분야에서 $\text{Mamba}$ 아키텍처가 기존의 $\text{Attention}$이나 $\text{RNN}$ 기반 모델의 대안이 될 수 있음을 시사하며, 향후 실시간 고빈도 매매 시스템이나 복잡한 포트폴리오 최적화 연구에 중요한 기초가 될 가능성이 높다.
