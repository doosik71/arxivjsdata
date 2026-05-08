# State-space Models with Layer-wise Nonlinearity are Universal Approximators with Exponential Decaying Memory

Shida Wang, Beichen Xue (2023)

## 🧩 Problem to Solve

본 논문은 최근 시퀀스 모델링 분야에서 주목받고 있는 State-space models(SSM)의 표현 능력(expressive capacity)과 메모리 특성에 대해 수학적으로 분석한다.

SSM은 Attention 기반의 Transformer가 가진 $O(T^2)$의 계산 복잡도를 $O(T \log T)$ 수준으로 낮추어 매우 효율적인 연산이 가능하다는 장점이 있다. 그러나 SSM은 기본적으로 시간 축(temporal direction)을 따라 비선형 활성화 함수(nonlinear activation)가 없는 선형 RNN과 유사한 구조를 가지고 있다. 이로 인해 다음과 같은 핵심적인 문제 제기가 가능하다.

1. **표현 능력의 한계**: 시간 축의 비선형성이 결여된 SSM이 레이어 간(layer-wise) 비선형성만으로 임의의 연속적인 sequence-to-sequence 관계를 근사할 수 있는 보편적 근사치(Universal Approximation) 능력을 갖추고 있는가?
2. **메모리 유지 능력**: SSM이 기존 RNN의 고질적인 문제인 지수적 메모리 감쇠(exponential decaying memory) 문제를 근본적으로 해결했는가?

논문의 목표는 multi-layer SSM이 보편적 근사 능력을 가짐을 구성적 증명(constructive proof)을 통해 보여주는 동시에, 이론적·실험적으로 SSM 역시 지수적 메모리 감쇠 특성을 가짐을 입증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **보편적 근사 특성 증명**: 레이어별 비선형 활성화 함수가 추가된 multi-layer SSM이 임의의 연속적인 sequence-to-sequence 매핑을 근사할 수 있음을 수학적으로 증명하였다.
2. **메모리 감쇠 특성 규명**: SSM이 구조적으로 지수적으로 감쇠하는 메모리 특성을 가지고 있음을 보였다. 이는 SSM이 기존 RNN과 유사한 메모리 한계를 공유함을 시사한다.
3. **구성적 접근법 제시**: Kolmogorov-Arnold 표현 정리와 Volterra 시리즈를 활용하여 SSM이 어떻게 임의의 함수를 근사할 수 있는지에 대한 구체적인 네트워크 구조를 제시하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들을 바탕으로 차별점을 둔다.

- **Recurrent Neural Networks (RNN)**: 비선형 활성화 함수를 가진 RNN이 보편적 근사 능력을 갖춘다는 점은 이미 알려져 있으나, 장기 의존성 학습 시 지수적 메모리 감쇠 문제가 발생한다.
- **State-space Models (SSM)**: HiPPO 행렬 등을 이용해 메모리 감쇠를 늦추려는 시도가 있었으며, 일부 연구에서 보편적 근사 가능성을 휴리스틱하게 제시하였다. 그러나 본 논문은 이를 수학적으로 엄밀하게 증명하는 구성적 증명을 제공한다.
- **기존 접근 방식과의 차별점**: 기존 SSM 연구들이 주로 효율성과 성능 향상에 집중했다면, 본 연구는 "표현 능력의 한계"와 "메모리 감쇠의 이론적 필연성"이라는 근본적인 수학적 성질을 분석했다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. State-space Model의 기본 구조

이산 시간 버전의 단일 레이어 SSM은 다음과 같이 정의된다.

$$h_{k+1} = Wh_k + Ux_{k+1}, \quad h_0 = 0$$
$$y_k = Ch_k + Dx_k$$

여기서 $x \in \mathbb{R}^{d_{in}}$, $y \in \mathbb{R}^{d_{out}}$, $h \in \mathbb{R}^m$이며 $W, U, C, D$는 학습 가능한 가중치 행렬이다. Multi-layer SSM의 경우, 각 레이어 사이에 비선형 활성화 함수 $\sigma$가 추가된다.

### 2. 보편적 근사 능력을 위한 증명 단계

논문은 세 가지 단계의 증명을 통해 multi-layer SSM의 보편성을 입증한다.

- **단계 1: 요소별 함수(Element-wise function) 근사**
  두 개의 레이어로 구성된 SSM에서 $W=0, b_2=0$으로 설정하면, 이는 단순한 피드포워드 네트워크가 되어 임의의 연속 함수 $f(x)$를 $\hat{y} = U_2 \sigma(U_1 x + b_1)$ 형태로 근사할 수 있다.
  
- **단계 2: 시간적 합성곱(Temporal Convolution) 근사**
  단일 레이어 SSM의 출력은 커널 함수 $\rho(t) = Ce^{Wt}U$와 입력의 합성곱으로 표현된다. 논문은 임의의 합성곱 커널 $\rho_k$를 지수 함수들의 합으로 근사할 수 있음을 보여, SSM이 임의의 시간적 합성곱을 수행할 수 있음을 증명한다.

- **단계 3: 일반적인 Sequence-to-Sequence 관계 근사**
  **Kolmogorov-Arnold 표현 정리**를 이용하여, 임의의 다변수 연속 함수를 단변수 함수들의 합과 합성으로 표현할 수 있다. 이를 위해 $\text{Element-wise function} \rightarrow \text{Convolution} \rightarrow \text{Element-wise function}$ 순으로 레이어를 쌓은 5-layer SSM 구조를 제안하여 보편적 근사 능력을 완성한다.

또한, 시퀀스 길이에 따라 뉴런 수가 증가하는 문제를 해결하기 위해, **Volterra Series** 기반의 구성을 제안하여 시퀀스 길이와 독립적인 뉴런 수로도 근사가 가능함을 보였다.

### 3. 메모리 감쇠 분석

메모리 특성을 측정하기 위해 **메모리 함수(Memory Function)** $\hat{\rho}(t)$를 다음과 같이 정의한다.

$$\hat{\rho}(t) = \left\| \frac{d\hat{y}_t}{dt} \right\|^2, \quad \hat{y}_t = H_t(x_{test})$$

여기서 $x_{test}$는 $t \ge 0$일 때 1, $t < 0$일 때 0인 스텝 함수이다. 논문은 Lipschitz 연속인 활성화 함수를 사용하는 multi-layer SSM의 경우, 다음의 지수적 감쇠 특성을 가짐을 수학적으로 증명하였다.

$$\lim_{t \to \infty} e^{c_0 t} \hat{\rho}(t) \to 0$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 임의로 생성된 합성 데이터셋(Synthetic datasets)을 사용하였다.
- **비교 대상**: Vanilla RNN, GRU, LSTM, Naive SSM, S4 모델.
- **측정 지표**: 시간에 따른 메모리 함수 $\hat{\rho}(t)$의 값의 변화.

### 2. 주요 결과

- **지수적 감쇠 확인**: Figure 5에서 보듯, 랜덤하게 초기화된 RNN, GRU, LSTM 및 Naive SSM 모두 시간이 경과함에 따라 메모리가 지수적으로 빠르게 감소하는 패턴을 보였다.
- **S4 모델의 특성**: S4 모델(Figure 6)은 정교한 초기화 기법(HiPPO 등) 덕분에 Naive SSM보다 메모리 감쇠 속도가 훨씬 느리다. 그러나 점근적(asymptotic) 관점에서는 여전히 직선 형태(로그 스케일 기준)의 지수적 감쇠를 따르는 것으로 나타났다.

## 🧠 Insights & Discussion

### 1. 강점 및 의의

본 연구는 SSM이 단순한 선형 구조임에도 불구하고, 레이어 간 비선형성만으로 RNN과 동일한 수준의 이론적 표현 능력을 갖추었음을 증명하였다. 이는 SSM이 Transformer의 효율적인 대안이 될 수 있는 이론적 토대를 제공한다.

### 2. 한계 및 비판적 해석

- **메모리의 근본적 한계**: S4와 같은 최신 모델들이 실무적으로는 긴 시퀀스를 잘 처리하는 것처럼 보이지만, 이론적으로는 여전히 지수적 감쇠라는 '메모리의 저주'에서 벗어나지 못했다는 점을 명시하였다.
- **정량적 분석의 부재**: 보편적 근사 가능성은 증명하였으나, 특정 태스크에서 얼마나 많은 레이어와 뉴런이 필요한지에 대한 정량적인 근사 속도(approximation rate) 분석은 이루어지지 않았다.

### 3. 종합 논의

SSM은 연산 효율성과 표현 능력 사이의 훌륭한 타협점이다. 특히 Hyena와 같이 합성곱 기반의 비재귀적 모델을 SSM의 재귀적 구조로 변환하더라도 표현 능력을 유지하면서 추론 시 메모리 비용을 낮출 수 있다는 가능성을 시사한다.

## 📌 TL;DR

본 논문은 **레이어별 비선형 활성화 함수가 추가된 multi-layer SSM이 임의의 시퀀스 관계를 근사할 수 있는 보편적 근사치(Universal Approximator)임을 수학적으로 증명**하였다. 하지만 동시에 **SSM 역시 기존 RNN과 마찬가지로 시간이 흐름에 따라 메모리가 지수적으로 감쇠하는 근본적인 한계를 가지고 있음**을 이론적·실험적으로 밝혔다. 이 연구는 SSM의 이론적 한계를 명확히 함으로써 향후 더 효율적인 장기 메모리 구조 설계의 지침을 제공한다.
