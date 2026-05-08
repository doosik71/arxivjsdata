# Cross-Modal Deep Metric Learning for Time Series Anomaly Detection

Wei Li and Zheze Yang (2025)

**주의:** 본 논문은 초록(Abstract)에서 '교차 모달 딥 메트릭 러닝(Cross-Modal Deep Metric Learning)'과 'vMF 분포'를 이용한 이상치 탐지를 언급하고 있으나, 본문의 실제 내용은 'LSTM-RV-EVT' 모델을 이용한 금융 리스크 측정 및 Value at Risk(VaR) 예측에 집중되어 있다. 따라서 본 보고서는 논문의 실질적인 구현 내용인 금융 리스크 예측 모델을 중심으로 분석한다.

## 🧩 Problem to Solve

본 논문은 금융 시장의 변동성을 예측하고 이를 통해 금융 자산이나 포트폴리오의 최대 잠재 손실액을 측정하는 Value at Risk(VaR) 산출의 정확도를 높이는 문제를 해결하고자 한다.

전통적인 리스크 측정 방식은 크게 세 가지(모수적, 비모수적, 준모수적 방법)로 나뉘나, 실제 금융 데이터는 정규 분포에서 벗어나 두꺼운 꼬리(fat tails)와 높은 첨도(kurtosis)를 가지는 특성이 있어 기존의 단순한 가정으로는 극단적인 시장 상황에서의 리스크를 정확히 예측하기 어렵다. 또한, 시계열 데이터의 비정상성(nonstationarity)과 장기 기억 특성(long memory)으로 인해 예측 오차가 발생하는 문제가 존재한다.

따라서 본 연구의 목표는 고주파 데이터(high-frequency data)를 활용하여 변동성을 정확히 예측하는 LSTM-RV 모델을 구축하고, 이를 극단치 이론(Extreme Value Theory, EVT)과 결합하여 비정상적인 시장 상황에서도 강건한 리스크 측정 모델인 LSTM-RV-EVT를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 딥러닝의 시계열 예측 능력과 통계학의 극단치 분석 능력을 결합한 하이브리드 구조를 설계한 것이다.

1. **가격과 거래량의 다각적 활용:** 단순히 가격 데이터뿐만 아니라 거래량(volume) 기반의 Realized Volatility(RV)를 함께 사용하여 변동성 예측의 정확도를 높였다.
2. **LSTM을 통한 장기 기억 모델링:** 시계열 데이터의 long-memory 특성을 캡처하기 위해 LSTM(Long Short-Term Memory) 네트워크를 도입하여 변동성 예측의 오차를 줄였다.
3. **EVT 기반의 꼬리 리스크(Tail Risk) 보정:** 정규 분포 가정을 탈피하여, 일반 파레토 분포(Generalized Pareto Distribution, GPD)를 통해 수익률 분포의 꼬리 부분을 정밀하게 모델링함으로써 극단적 손실 가능성을 정확히 측정하였다.

## 📎 Related Works

논문은 시계열 분석을 위한 다양한 딥러닝 구조와 통계적 모델을 소개한다.

- **재귀형 신경망(RNN) 및 변형 모델:** LSTM과 GRU는 긴 의존성(long-range dependencies)을 처리하는 데 강점이 있으며, 금융 도메인의 대출 부도 예측이나 기업 재무 지표 예측에 활용되어 왔다.
- **Transformer 및 Attention:** 셀프 어텐션 메커니즘을 통해 주가 예측에서 장단기 변동성을 동시에 포착하는 능력이 입증되었다.
- **통계적 변동성 모델:** HAR(Heterogeneous Autoregressive) 모델은 일별, 주별, 월별 RV를 사용하여 다음 날의 RV를 예측하는 단순하고 강력한 모델로 알려져 있으며, 이를 확장한 HARQ, HARF 등이 제안되었다.
- **기존 접근 방식의 한계:** 전통적인 GARCH 계열 모델이나 단순 LSTM 모델은 금융 데이터의 극단적인 꼬리 부분(extreme tail)을 효과적으로 처리하지 못해 리스크 과소평가 문제가 발생할 수 있다.

## 🛠️ Methodology

### 1. Realized Volatility (RV) 산출

고주파 데이터를 이용하여 변동성을 측정하기 위해 다음과 같은 RV 식을 사용한다.
$$RV(p_t) = \sum_{i=1}^{N(\Delta)} (\ln P_{t,i} - \ln P_{t,i-1})^2$$
여기서 $P_{t,i}$는 $t$일의 $i$번째 고주파 시점의 가격이며, $N(\Delta)$는 일일 관측치 수이다.

### 2. Wavelet-LSTM-RV 변동성 예측 모델

가격($P$)과 거래량($V$) 각각에 대해 univariate LSTM 모델을 적용하여 로그 변동성을 예측한다.
$$\ln RV(P_t) = \text{LSTM}(\ln RV(P_{t-1}), \dots, \ln RV(P_{t-p}))$$
$$\ln RV(V_t) = \text{LSTM}(\ln RV(V_{t-1}), \dots, \ln RV(V_{t-p}))$$
이 구조는 가격 정보와 거래량 정보를 동시에 활용함으로써 변동성 예측의 정밀도를 향상시킨다.

### 3. LSTM-RV-EVT 하이브리드 모델 구축

예측된 변동성을 바탕으로 VaR를 계산하기 위해 극단치 이론(EVT)의 일반 파레토 분포(GPD)를 도입한다. GPD의 누적 분포 함수(CDF)는 다음과 같다.
$$F_{\xi, \beta}(x) = \begin{cases} 1 - (1 + \frac{\xi x}{\beta})^{-1/\xi}, & \xi \neq 0 \\ 1 - \exp(-\frac{x}{\beta}), & \xi = 0 \end{cases}$$
여기서 $\xi$는 형태 매개변수(shape parameter), $\beta$는 척도 매개변수(scale parameter)이다. 이를 통해 $(1-p_0)$ 분위수(quantile)를 다음과 같이 산출한다.
$$F^{-1}(1-p_0) = u + \frac{\beta}{\xi} \left[ \left( \frac{p_0 n}{n_u} \right)^{-\xi} - 1 \right]$$
최종적으로 예측된 변동성 $\sqrt{RV_{t+1|t}}$와 EVT의 분위수 값을 결합하여 VaR를 결정한다.
$$\text{VaR}_{t+1|t} = F^{-1}(1-p_0) \sqrt{RV_{t+1|t}}$$

### 4. LSTM 셀 구조

본 모델의 예측 엔진인 LSTM은 다음과 같은 게이트 메커니즘을 통해 그래디언트 소실 문제를 해결하고 장기 의존성을 학습한다.

- **Forget gate:** $f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$
- **Input gate:** $i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)$
- **Cell state update:** $c_t = f_t \cdot c_{t-1} + i_t \cdot \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)$
- **Output gate:** $o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)$
- **Hidden state:** $h_t = o_t \cdot \tanh(c_t)$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** 2000년 1월 4일부터 2017년 10월 31일까지의 CSI 300 지수 5분 단위 고주파 데이터 (총 4,239 거래일).
- **비교 대상 모델:** LSTM-RV, LSTM, HAR, HAR-QF, ARFIMA.
- **평가 지표:** MSE, MAE, MAPE, QLIKE (변동성 예측 성능 측정), 및 UC/IND/CC 테스트 (VaR 백테스팅 측정).

### 2. 변동성 예측 결과

LSTM-RV 모델이 모든 지표에서 가장 낮은 오차를 기록하였다. 특히 HAR 모델과 비교했을 때 MSE는 15.16%, MAE는 23.79% 감소하는 큰 개선을 보였다. 이는 가격과 거래량 데이터를 모두 활용한 딥러닝 모델이 전통적인 통계 모델보다 변동성 예측에 훨씬 유리함을 입증한다.

### 3. VaR 백테스팅 결과

18개의 모델(분위수 추정 방법 $\times$ 변동성 모델)을 비교한 결과, **LSTM-RV-EVT** 모델이 가장 우수한 성능을 보였다.

- **위반 비율(Violation Ratio):** 이론적 수치인 0.01에 가장 근접한 결과를 보였다.
- **통계적 유의성:** UC(Unconditional Coverage) 및 CC(Conditional Coverage) 테스트를 모두 통과하여 리스크 측정의 정확성을 검증받았다.
- **특이사항:** 롱 포지션(Long positions)의 결과가 숏 포지션(Short positions)보다 일반적으로 더 좋게 나타났는데, 이는 수익률 분포의 왼쪽 꼬리(손실 방향) 변동성이 오른쪽 꼬리보다 더 높기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 딥러닝의 예측력과 EVT의 통계적 정밀함을 결합하여 금융 리스크 측정의 실용적인 프레임워크를 제시하였다.

**강점:**

- 단순한 가격 시계열을 넘어 거래량(Volume)이라는 추가 모달리티를 활용함으로써 예측 성능을 유의미하게 끌어올렸다.
- 금융 데이터의 특성인 '두꺼운 꼬리' 문제를 EVT의 GPD 분포로 해결하여, 극단적인 시장 상황에서의 VaR 예측력을 확보하였다.

**한계 및 비판적 해석:**

- **내용의 불일치:** 논문의 제목과 초록에서는 'Cross-Modal Deep Metric Learning'과 'vMF 분포'를 언급하고 있으나, 정작 본문에서는 이에 대한 구체적인 아키텍처나 수식, 실험 결과가 전혀 제시되지 않았다. 본문은 전형적인 LSTM-EVT 결합 모델을 다루고 있어, 초록과 본문의 내용이 서로 다른 논문인 것처럼 보일 정도로 불일치가 심하다.
- **모델 복잡도:** LSTM-RV-EVT는 높은 성능을 보이지만, 실시간 리스크 관리를 위해 필요한 계산 복잡도와 하이퍼파라미터 튜닝 비용에 대한 분석이 부족하다.

## 📌 TL;DR

본 논문은 고주파 금융 데이터를 활용해 변동성을 예측하는 **LSTM-RV 모델**과 극단적 손실을 측정하는 **EVT(극단치 이론)**를 결합한 **LSTM-RV-EVT 리스크 측정 모델**을 제안한다. 실험 결과, 제안 모델은 전통적인 HAR, ARFIMA 모델보다 변동성 예측 오차가 적고, VaR 백테스팅에서 가장 높은 정확도를 보였다. 특히 가격뿐만 아니라 거래량 정보를 통합한 것이 성능 향상의 주요 요인이다. 이 연구는 딥러닝 기반의 정밀한 리스크 관리 시스템 구축에 중요한 기초가 될 수 있다.
