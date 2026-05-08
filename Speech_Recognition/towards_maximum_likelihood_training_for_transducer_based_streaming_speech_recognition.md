# Towards Maximum Likelihood Training for Transducer-based Streaming Speech Recognition

Hyeonseung Lee, Ji Won Yoon, Sungsoo Kim, and Nam Soo Kim (2024)

## 🧩 Problem to Solve

본 논문은 스트리밍 자동 음성 인식(Streaming Automatic Speech Recognition, ASR) 시스템, 특히 Transducer 기반 모델에서 발생하는 정확도 저하 문제를 해결하고자 한다. 일반적으로 스트리밍 ASR은 낮은 지연 시간(latency)을 유지해야 하므로, 전체 입력 문맥을 사용하는 비스트리밍(non-streaming) 모델에 비해 성능이 떨어진다.

저자들은 이러한 성능 저하의 원인을 두 가지로 분석한다. 첫째는 제한된 입력 문맥으로 인해 발생하는 '정보 부족(information deficiency)'이며, 이는 스트리밍 시스템의 본질적인 특성상 피하기 어렵다. 둘째는 '변형된 우도(deformed likelihood)' 문제이다. 기존의 스트리밍 Transducer 모델들은 비스트리밍 모델을 위해 설계된 재귀 규칙(recursion rules)을 기반으로 우도 함수를 최대화하도록 학습된다. 그러나 스트리밍 환경에서는 인코더가 인과적(causal)으로 동작하여 입력 문맥이 제한되므로, 학습 시 사용되는 우도 함수와 실제 추론 시의 우도 사이에 불일치(mismatch)가 발생하며, 이는 결과적으로 ASR 정확도의 최적화를 방해한다. 따라서 본 논문의 목표는 이 변형된 우도 문제를 수학적으로 정의하고, 이를 보정하여 실제 우도(actual likelihood)에 가깝게 학습시키는 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 스트리밍 Transducer 학습 시 발생하는 실제 우도와 변형된 우도 사이의 간극을 수학적으로 정량화하고, 이를 보정하기 위한 추정기(estimator)를 도입하는 것이다.

가장 중점적인 기여는 **Forward Variable Causal Compensation (FoCC)**이라는 개념을 도입한 것이다. FoCC는 인코더의 덩어리(chunk) 경계에서 발생하는 조건부 확률의 차이를 보상하는 확률 비율로 정의된다. 또한, 이 FoCC 값을 직접적으로 계산하는 것이 불가능하므로, 이를 근사적으로 추정하는 별도의 신경망인 **FoCCE (FoCC Estimator)** 네트워크를 제안하였다. 이를 통해 스트리밍 모델이 변형된 우도가 아닌, 실제 우도를 최대화하도록 학습시킴으로써 비스트리밍 모델과의 성능 격차를 줄였다.

## 📎 Related Works

기존의 스트리밍 ASR 연구들은 주로 지연 시간을 줄이면서 정확도를 유지하는 아키텍처 설계에 집중해 왔다. 일부 연구에서는 변형된 우도 문제를 피하기 위해 **전역 정규화(globally normalized likelihood)** 방식을 채택하기도 하였다. 전역 정규화는 수용 가능한 경로(accepting paths)의 점수 합과 모든 가능한 경로의 점수 합의 비율로 우도를 정의하여 국소적 확률(local probability) 항을 배제함으로써 문제를 우회한다.

하지만 이러한 전역 정규화 방식은 스트리밍과 비스트리밍 시나리오 모두에서 국소 확률 기반의 주류 ASR 방법들보다 낮은 정확도를 보인다는 한계가 있다. 본 논문은 전역 정규화처럼 방식을 완전히 바꾸는 것이 아니라, 주류 방식인 국소 확률 기반의 Transducer 구조를 유지하면서도 수학적 보정(FoCC)을 통해 스트리밍 환경에서의 우도 불일치 문제를 해결하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Transducer의 기본 구조 및 문제점

비스트리밍 Transducer는 입력 $x_{1:T}$와 타겟 $y_{0:U}$에 대해 다음과 같은 조건부 우도를 최대화한다.
$$L_\theta(x_{1:T}, y_{0:U}) := \log P_\theta(y_{0:U}, z_U \le T < z_{U+1} | x_{1:T})$$
여기서 $z_u$는 정렬 변수(alignment variable)이며, 전방 변수(forward variable) $\alpha_\theta(t, u)$를 통해 동적 계획법으로 계산된다. 재귀식은 다음과 같다.
$$\alpha_\theta(t, u) = \alpha_\theta(t-1, u) \phi_\theta(t-1, u) + \alpha_\theta(t, u-1) Y_\theta(t, u-1)$$
여기서 $\phi_\theta$는 blank 확률, $Y_\theta$는 레이블 확률이다.

스트리밍 Transducer는 $t$ 시점에서 제한된 문맥 $x_{1:e(t)}$만을 사용하여 국소 확률 $\tilde{\phi}_\theta, \tilde{Y}_\theta$를 추정한다. 기존 방식은 위 비스트리밍 재귀식에 스트리밍 국소 확률을 그대로 대입하여 학습하는데, 이는 베이즈 규칙(Bayes' rule)을 위반하며 '변형된 우도' 문제를 야기한다.

### 2. Forward Variable Causal Compensation (FoCC)

실제 우도를 얻기 위해 저자들은 다음과 같은 확률 비율인 $\gamma_\theta(t, u)$를 도입한다.
$$\gamma_\theta(t, u) := \frac{P_\theta(y_{0:u}, z_u \le t < z_{u+1} | x_{1:e(t+1)})}{P_\theta(y_{0:u}, z_u \le t < z_{u+1} | x_{1:e(t)})}$$
이 값은 인코더의 덩어리 경계($e(t) < e(t+1)$)에서만 유효하며, 그 외에는 1이 된다. 이를 적용한 수정된 재귀식은 다음과 같다.
$$\tilde{\alpha}_\theta(t, u) = \tilde{\alpha}_\theta(t-1, u) \tilde{\phi}_\theta(t-1, u) \gamma_\theta(t-1, u) + \tilde{\alpha}_\theta(t, u-1) \tilde{Y}_\theta(t, u-1)$$

### 3. FoCCE 네트워크 설계

$\gamma_\theta$를 직접 계산할 수 없으므로, 별도의 파라미터 $\omega$를 가진 **FoCCE 네트워크**를 통해 $\gamma_\omega(t, u)$를 추정한다.
$$\gamma_\omega(t, u) := \left( \frac{\chi_\omega(t, u)}{\bar{\chi}_\omega(t)} \right)^{\lambda_\gamma}$$
여기서 $\chi_\omega(t, u)$는 타겟 시퀀스 이력까지 고려한 다음 덩어리 입력 특징의 확률 밀도이며, $\bar{\chi}_\omega(t)$는 입력 이력만을 고려한 확률 밀도이다. 이를 구현하기 위해 **Masked Autoregressive Flow (MAF)** 기반의 $\text{DensityEstimator}$를 사용하였으며, 이는 연속적인 공간에서 임의의 확률 밀도를 모델링하기 위함이다.

### 4. 학습 절차 및 손실 함수

전체 학습 목표는 수정된 우도 $L_{mod}$와 FoCCE의 밀도 추정 손실 $L_\chi$의 가중 합으로 정의된다.
$$L_{tot} = \lambda_{mod} L_{mod} + \lambda_\chi L_\chi$$
수정된 우도 계산 시 $\gamma_\omega(t, u)$에는 **stop-gradient** 연산자 $\text{sg}(\cdot)$를 적용하여, FoCCE 네트워크의 그래디언트가 Transducer 네트워크에 직접적인 영향을 주어 발산하는 것을 방지한다.

## 📊 Results

### 실험 설정

- **데이터셋**: LibriSpeech 및 TED-LIUM3.
- **모델 아키텍처**: Zipformer를 기반으로 하며, 스트리밍 모델의 경우 8-chunk size (160ms latency)를 사용하였다.
- **FoCCE 구성**: 8개 층의 causal convolution 모듈, 128-D LSTM Predictor, MAF 기반 Density Estimator로 구성되었다.
- **평가 지표**: Word Error Rate (WER)를 사용하였으며, Beam size 4의 빔 서치 알고리즘을 적용하였다.

### 주요 결과

실험 결과, FoCCE 학습을 적용한 스트리밍 Transducer가 기존 스트리밍 모델보다 일관되게 낮은 WER을 기록하였다.

- **LibriSpeech**: 비스트리밍 모델과 스트리밍 베이스라인 간의 WER 격차를 `test-clean`에서 26.3%, `test-other`에서 12.3%까지 줄였다.
- **TED-LIUM3**: 스트리밍과 비스트리밍 모델 간의 WER 격차를 17.7% 감소시켰다.
- **하이퍼파라미터 분석**: $\lambda_\gamma$ 값에 따라 성능 변화가 민감하게 나타났다. 이는 $\lambda_\gamma$가 이산 공간이 아닌 연속 특징 공간의 두 확률 밀도 나눗셈을 기반으로 계산되기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 스트리밍 ASR의 성능 저하가 단순히 정보의 부족 때문만이 아니라, 학습 시 사용하는 우도 함수의 수학적 불일치라는 구조적인 문제에서 기인함을 밝혀냈다는 점에서 큰 의의가 있다. 특히 Normalizing Flows(MAF)를 이용하여 복잡한 확률 밀도 비율을 추정함으로써, 기존의 전역 정규화 방식이 가졌던 성능 저하 문제를 해결하고 국소 확률 기반 모델의 이점을 유지하였다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, FoCCE라는 별도의 보조 네트워크를 학습시켜야 하므로 학습 복잡도가 증가한다. 둘째, $\lambda_\gamma$와 같은 하이퍼파라미터에 성능이 민감하게 반응하므로, 최적의 값을 찾기 위한 추가적인 실험적 비용이 발생한다. 마지막으로, 본 연구는 Zipformer 아키텍처에 한정하여 검증되었으므로, 다른 인코더 구조에서도 동일한 효과가 나타날지에 대한 추가 연구가 필요하다.

## 📌 TL;DR

이 논문은 스트리밍 Transducer 학습 시 발생하는 '변형된 우도(deformed likelihood)' 문제를 수학적으로 규명하고, 이를 보정하기 위한 **Forward Variable Causal Compensation (FoCC)** 개념과 이를 추정하는 **FoCCE 네트워크**를 제안하였다. MAF 기반의 밀도 추정기를 통해 실제 우도에 가깝게 학습시킨 결과, LibriSpeech와 TED-LIUM3 데이터셋에서 스트리밍 모델의 WER을 유의미하게 낮추어 비스트리밍 모델과의 성능 격차를 줄였다. 이는 향후 고성능 실시간 온디바이스 ASR 시스템 구현에 중요한 기여를 할 것으로 보인다.
