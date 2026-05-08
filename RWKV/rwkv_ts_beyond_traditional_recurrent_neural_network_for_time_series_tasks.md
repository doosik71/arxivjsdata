# RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks

Haowen Hou, F. Richard Yu (2024)

## 🧩 Problem to Solve

본 논문은 시계열(Time Series) 작업에서 전통적인 Recurrent Neural Network(RNN) 아키텍처, 특히 LSTM과 GRU가 겪고 있는 한계를 해결하고자 한다. 과거에 RNN은 순차적 데이터 모델링의 핵심이었으나, 최근에는 Transformer, MLP, CNN 기반 모델들에 밀려 그 지배적인 위치를 잃었다.

전통적인 RNN이 시계열 작업에서 외면받는 주요 이유는 다음과 같다:

1. **기울기 소실 및 폭주(Vanishing/Exploding Gradient):** 시퀀스 길이가 100을 초과할 경우 장기 의존성(Long-term dependencies)을 포착하는 능력이 급격히 저하된다.
2. **병렬 계산 불가능:** 순차적인 구조로 인해 계산 효율성이 낮으며, 모델의 규모를 확장(Scale-up)하는 데 어려움이 있다.
3. **오차 누적:** 단계별 예측(Step-by-step prediction) 방식은 추론 속도를 늦출 뿐만 아니라 예측 오차가 누적되는 문제를 야기한다.

따라서 본 논문의 목표는 전통적인 RNN의 효율성과 장점을 유지하면서도, Transformer 수준의 성능과 확장성을 갖춘 새로운 RNN 기반 시계열 모델인 **RWKV-TS**를 설계하는 것이다.

## ✨ Key Contributions

RWKV-TS의 핵심 아이디어는 **Linear RNN** 설계를 통해 RNN의 추론 효율성과 Transformer의 병렬 학습 능력을 동시에 확보하는 것이다. 주요 기여 사항은 다음과 같다:

- **선형 복잡도 구현:** 시간 및 공간 복잡도를 $O(L)$로 유지하여 매우 효율적인 계산이 가능하게 한다.
- **장기 의존성 포착 능력 강화:** Token Shift와 Time Decay 메커니즘을 통해 전통적인 RNN보다 훨씬 긴 시퀀스 정보(최대 4096 토큰 이상)를 효과적으로 처리한다.
- **병렬 학습 및 단일 단계 추론:** Encoder-only 구조를 채택하여 학습 시에는 병렬 처리가 가능하며, 추론 시에는 오차 누적 없이 단 한 번의 단계로 예측을 수행한다.

## 📎 Related Works

본 논문은 시계열 분석의 네 가지 주요 접근 방식을 검토한다:

1. **RNN 기반 모델:** 순차 데이터 처리에 능숙하지만, Long-Term Sequence Forecasting(LTSF)에서 장기 의존성 포착 실패로 인해 최근 기피되는 경향이 있다. 다만, 최근 LLM 분야에서 RWKV, Mamba와 같은 Linear RNN이 성공을 거두고 있다.
2. **Transformer 기반 모델:** Attention 메커니즘을 통해 병렬화와 장기 의존성 포착이 가능하지만, $O(L^2)$의 복잡도로 인해 계산 비용이 높다. (예: Informer, Autoformer, PatchTST)
3. **MLP 기반 모델:** DLinear와 같이 단순한 선형 층을 사용하여 Transformer보다 효율적이면서도 강력한 성능을 보여주어, 복잡한 모델의 필요성에 의문을 제기했다.
4. **CNN 기반 모델:** TimesNet과 같이 지역적 패턴 추출에 강점을 보이며 LTSF 영역에서 인상적인 결과를 냈다.

RWKV-TS는 이러한 기존 모델들의 장점을 결합하여, MLP의 효율성과 Transformer의 성능, 그리고 RNN의 순차적 특성을 모두 잡고자 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

RWKV-TS는 **입력 모듈 $\rightarrow$ RWKV 백본 $\rightarrow$ 출력 모듈**의 구조를 가진다.

- **Instance Normalization:** 훈련과 테스트 데이터 간의 분포 변화(Distribution shift)를 완화하기 위해 각 단변량 시계열 인스턴스를 평균 0, 표준편차 1로 정규화한다.
- **Patching:** 긴 시계열 데이터를 작은 조각(Patch)으로 나누어 토큰화한다. 패치 길이 $P$와 스트라이드 $S$를 사용하여 입력 토큰의 수를 $L$에서 $N$으로 줄여 계산 효율을 높인다. 이때 토큰 수는 다음과 같이 계산된다:
  $$N = \lfloor \frac{L-P}{S} \rfloor + 2$$
- 이후 학습 가능한 투영 행렬을 통해 패치를 모델 차원 $D$의 입력 토큰 $x_t \in \mathbb{R}^D$로 변환한다.

### 2. RWKV Blocks

백본은 Time-mixing 블록과 Channel-mixing 블록이 적층된 잔차 블록(Residual blocks)으로 구성된다.

#### (1) Time Mixing Sub-block

- **Token Shift:** 현재 시점 $x_t$와 이전 시점 $x_{t-1}$을 학습 가능한 변수 $\mu$를 이용해 선형 결합하여 단순한 시간 믹싱을 수행한다.
  $$g_t = W_g \cdot (\mu_g \odot x_t + (1-\mu_g) \odot x_{t-1})$$
  (동일한 구조가 $r_t, k_t, v_t$에 대해서도 적용된다.)
- **Multi-head WKV Operator:** RWKV의 핵심으로, Self-attention과 유사하지만 선형 복잡도를 가진다. 단일 헤드의 WKV 연산은 다음과 같다:
  $$wkv_t = \text{diag}(u) \cdot k_t^T \cdot v_t + \sum_{i=1}^{t-1} \text{diag}(w)^{t-1-i} \cdot k_i^T \cdot v_i$$
  여기서 $w$는 채널별 시간 감쇠(Time decay) 벡터이며, $w = \exp(-\exp(w))$로 변환되어 $(0, 1)$ 범위를 갖는 수축 행렬이 된다. $u$는 현재 토큰에 더 많은 주의를 기울이게 하는 보너스 파라미터이다.
- **Output Gating:** SiLU 활성화 함수와 Receptance를 사용하여 최종 출력을 생성한다.
  $$o_t = (\text{SiLU}(g_t) \odot \text{LayerNorm}(r_t \cdot wkv_t))W_o$$

#### (2) Channel Mixing Sub-block

채널 간의 정보를 믹싱하며, 강한 비선형 연산을 사용한다. 특히 $\text{ReLU}^2$ (Squared ReLU)를 사용하여 비선형성을 강화한다.
$$v'_t = \text{ReLU}^2(k'_t) \cdot W'_v, \quad o'_t = \text{Sigmoid}(r'_t) \odot v'_t$$

### 3. Recurrent Mode 및 추론

RWKV-TS는 병렬 모드와 순환 모드가 수학적으로 동일하다. 추론 시에는 다음과 같이 상태 $s_t$를 업데이트하는 RNN 형태로 동작한다:
$$wkv_t = s_{t-1} + \text{diag}(u) \cdot k_t^T \cdot v_t$$
$$s_t = \text{diag}(w) \cdot s_{t-1} + k_t^T \cdot v_t$$
이 구조 덕분에 테스트 단계(Test step)가 1이며, 전통적인 RNN의 오차 누적 문제를 피할 수 있다.

### 4. 손실 함수 및 복잡도

- **손실 함수:** Mean Squared Error (MSE)를 사용한다.
  $$\text{MSE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- **복잡도:** 시간 및 공간 복잡도 모두 $O(L)$로, Transformer의 $O(L^2)$보다 월등히 효율적이다.

## 📊 Results

### 1. 실험 설정 및 지표

- **데이터셋:** Weather, Traffic, Electricity, ILI, ETT(h1, h2, m1, m2) 등 8개 이상의 벤치마크 사용.
- **비교 모델:** TimesNet(CNN), DLinear/LightTS(MLP), PatchTST/Informer/Autoformer(Transformer) 등.
- **지표:** MSE, MAE, SMAPE, MASE, F1-score(이상치 탐지).

### 2. 주요 결과

- **효율성 분석:** 동일한 모델 차원(768)에서 RWKV-TS는 PatchTST나 TimesNet보다 파라미터 수, 학습 시간, 추론 시간 및 메모리 사용량 면에서 압도적으로 유리하다.
- **장기 예측(Long-term Forecasting):** PatchTST와 대등한 수준의 성능을 보이며, 최신 SOTA 모델인 TimesNet보다 평균 MSE를 약 12.58% 감소시켰다.
- **단기 예측(Short-term Forecasting):** M4 데이터셋에서 대부분의 Transformer 및 MLP 기반 모델보다 우수한 성능을 보였으며, TimesNet/N-BEATS와는 근소한 차이를 보였다.
- **Few-shot 학습:** 학습 데이터의 10%만 사용했을 때, TimesNet(MSE 28.95% 감소)과 DLinear(MSE 7.92% 감소)보다 월등한 성능을 보여 강력한 표현 학습 능력을 입증했다.
- **시계열 분류 및 이상치 탐지:** 분류 작업에서 평균 정확도 73.10%를 기록하여 TimesNet(73.60%)에 근접했으며, 이상치 탐지에서도 SOTA 모델들과 유사한 F1-score를 달성했다.
- **결측치 보간(Imputation):** MLP 및 일부 Transformer보다는 우수하지만, PatchTST 등 SOTA 모델보다는 성능이 낮았다. 이는 RWKV-TS가 단방향(Unidirectional) 모델이기 때문이며, 양방향 모델(Bidirectional)인 Transformer가 전후 맥락을 모두 파악할 수 있어 발생하는 차이이다.

## 🧠 Insights & Discussion

### 강점

RWKV-TS는 **"효율성과 성능의 최적의 트레이드-오프"**를 달성했다. 특히 메모리 사용량이 적고 추론 속도가 매우 빨라 자원이 제한된 엣지 디바이스(End-devices)에 배포하기에 매우 적합하다. 또한, Linear RNN이 단순한 효율성 도구를 넘어 고차원적인 시계열 표현 학습(Representation Learning)이 가능함을 증명했다.

### 한계 및 향후 과제

- **단방향성의 한계:** Imputation 작업에서 나타난 성능 저하는 모델이 과거 정보만 참조할 수 있기 때문이다. 저자들은 향후 **Bi-RWKV-TS**와 같은 양방향 구조를 도입함으로써 이 문제를 해결할 수 있을 것이라 제안한다.
- **가정:** 본 연구는 Encoder-only 구조를 통해 RNN의 고질적인 문제인 오차 누적을 해결했으나, 이는 전통적인 RNN의 '순차적 생성' 방식과는 다른 접근이다.

### 비판적 해석

본 논문은 RNN이 더 이상 시계열 작업에 적합하지 않다는 기존 학계의 통념에 정면으로 도전한다. 하지만 결과적으로 성능이 Transformer SOTA와 '비슷'한 수준이지, 모든 지표에서 '압도'하는 것은 아니다. 다만, 계산 비용을 획기적으로 줄이면서 동등한 성능을 낸다는 점이 실무적 관점에서 매우 큰 가치를 지닌다.

## 📌 TL;DR

RWKV-TS는 Linear RNN 아키텍처를 시계열 작업에 적용하여 **$O(L)$의 시간/공간 복잡도**와 **Transformer급 성능**을 동시에 달성한 모델이다. 장기 예측, 분류, 이상치 탐지 등 다양한 작업에서 SOTA 모델들과 경쟁 가능한 성능을 보였으며, 특히 **추론 속도와 메모리 효율성**에서 압도적인 우위를 점한다. 이 연구는 RNN이 현대적인 설계(Linear RNN)를 통해 시계열 분석 분야에서 다시금 강력한 도구가 될 수 있음을 시사한다.
