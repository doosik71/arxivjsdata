# Overcoming Overfitting and Large Weight Update Problem in Linear Rectifiers: Thresholded Exponential Rectified Linear Units

Vijay Pandey (2019/Extract)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델, 특히 심층 신경망(Deep Complex Networks)에서 널리 사용되는 Linear Rectifier 계열의 활성화 함수들이 가진 두 가지 핵심적인 문제를 해결하고자 한다.

첫째는 **Bias Shift 문제**이다. ReLU를 포함한 많은 선형 정류기들은 출력값의 평균이 0이 아니기 때문에, 층이 깊어질수록 편향(bias)이 한쪽으로 쏠리는 현상이 발생하며 이는 학습 속도를 저하시키는 원인이 된다.

둘째는 **큰 가중치 업데이트(Large Weight Update) 문제**이다. 기존의 선형 정류기들은 양수 입력 영역에서 기울기(gradient)가 $1$로 일정하다. 이로 인해 가중치 업데이트가 특정 방향으로 계속 누적될 수 있으며, 특히 심층 네트워크에서는 saturation point(포화 지점)의 부재로 인해 가중치가 과도하게 커지게 된다. 이는 결과적으로 모델의 불안정성을 초래하고, 훈련 데이터에 과하게 적합되는 **Overfitting(과적합)** 현상을 유발한다.

따라서 본 논문의 목표는 양수 영역에서도 적절한 포화 지점을 제공하여 가중치 업데이트를 제어하고, 출력 평균을 0에 가깝게 유지하여 과적합을 방지하는 새로운 활성화 함수인 **TERELU(Thresholded Exponential Rectified Linear Unit)**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Linear Rectifier의 효율성과 Sigmoid 계열의 포화 특성을 결합**하는 것이다.

구체적으로, TERELU는 입력값의 범위에 따라 세 가지 영역으로 동작하도록 설계되었다. 0 이하의 영역에서는 ELU와 유사하게 지수 함수를 사용하여 saturation을 제공하고, $0$과 특정 임계값 $\mu$ 사이에서는 Linear Rectifier처럼 동작하여 vanishing gradient 문제를 방지하며, $\mu$보다 큰 양수 영역에서는 다시 지수 함수 기반의 메커니즘을 도입하여 가중치 업데이트를 억제하는 **Contracting Gradient(수축 기울기)**를 형성한다.

이러한 설계를 통해 양수 영역에서도 saturation point를 가짐으로써 가중치 붕괴나 폭주를 막고, 정규화 효과(Regularization)를 얻어 모델의 일반화 성능을 높이는 것이 본 연구의 중심적인 직관이다.

## 📎 Related Works

논문에서는 다음과 같은 기존 활성화 함수들을 소개하며 그 한계를 지적한다.

- **ReLU (Rectified Linear Unit):** 양수 영역에서 기울기가 1이므로 vanishing gradient를 해결하지만, 음수 영역에서 기울기가 0이 되는 Dead Neuron 문제와 non-zero mean으로 인한 bias shift 문제가 있다.
- **LReLU / ELU / PReLU:** Dead Neuron 문제를 완화하기 위해 음수 영역에 작은 기울기나 지수 함수를 도입하였다. 하지만 여전히 모든 양수 입력에 대해 선형적으로 증가하므로, 양수 영역에서의 saturation point가 없어 가중치 업데이트가 과도해지는 문제를 해결하지 못한다.
- **SRELU / APL / Maxout:** 학습 가능한 파라미터를 통해 유연성을 높였으나, TERELU가 지향하는 부드러운 포화 특성이나 노이즈 강건성(noise-robustness) 측면에서 부족함이 있다.
- **Tanh / Sigmoid:** 양방향 saturation을 제공하여 zero-mean 특성을 가지지만, 극단적인 값에서 기울기가 거의 0이 되는 Vanishing Gradient 문제로 인해 심층 네트워크 학습이 어렵다.

TERELU는 위 연구들의 장점을 취합하여, $\mu$라는 임계값을 통해 vanishing gradient를 피하면서도 양 끝단에서의 saturation을 통해 안정성을 확보한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 구조 및 수식

TERELU 활성화 함수 $f(x)$는 다음과 같이 정의된다.

$$
f(x) = 
\begin{cases} 
\alpha(e^x - 1), & x \le 0 \\
x, & 0 < x < \mu \\
\beta\mu(e^{(x-\mu)/\mu} - 1) + \mu, & x \ge \mu 
\end{cases}
$$

이 함수의 도함수 $f'(x)$는 다음과 같다.

$$
f'(x) = 
\begin{cases} 
\alpha e^x, & x \le 0 \\
1, & 0 < x < \mu \\
\beta e^{(x-\mu)/\mu}, & x \ge \mu 
\end{cases}
$$

여기서 $\alpha, \mu$는 하이퍼파라미터이며, $\beta$는 학습 가능한(trainable) 파라미터이다. 기본 설정값은 모두 $1$로 지정된다.

### 주요 구성 요소의 역할

1.  **$\mu$ (Threshold):** 활성화 함수가 선형적으로 동작하는 구간의 범위를 결정한다. $\mu$ 값을 적절히 설정함으로써 Tanh와 달리 saturation이 발생하는 시점을 늦출 수 있으며, 이는 vanishing gradient 문제를 효과적으로 회피하게 한다.
2.  **$\beta$ (Regularizer):** 학습 가능한 파라미터로서 가중치 업데이트의 강도를 조절한다. $\beta$가 감소하면 weight decay와 같은 정규화 효과를 내어 과적합을 방지하고, $\beta$가 증가하면 학습 속도를 높이는 가속기 역할을 한다.
3.  **Exponential Function (양 끝단):** 음수 영역($x \le 0$)과 양수 임계 영역($x \ge \mu$)에 지수 함수를 배치하여 출력의 평균을 0으로 유도(zero-mean)하고, 데이터의 희소성(sparsity)을 높여 노이즈에 강건한 표현을 생성한다.

## 📊 Results

### 실험 설정
- **데이터셋:** MNIST 데이터셋 사용.
- **모델 구조:** Fully Connected Neural Network (FCNN) 및 Batch Normalization (BN) 적용.
- **비교 대상:** ELU 등 기존 Linear Rectifier 기반 활성화 함수.
- **네트워크 깊이:** 8층, 20층, 그리고 매우 깊은 56층(hidden layers) 구조를 사용하여 깊이에 따른 안정성을 측정하였다. 모든 은닉층의 유닛 수는 64개로 고정하였다.

### 주요 결과
- **얕은 네트워크 (8층, 20층):** ELU와 TERELU의 성능 차이가 거의 없으며 두 함수 모두 유사하게 동작하였다.
- **깊은 네트워크 (56층):** ELU의 경우 훈련이 진행됨에 따라 Validation Error가 심하게 요동치며 전형적인 **Overfitting** 현상이 관찰되었다. 반면, TERELU는 Validation Accuracy와 Training Accuracy가 함께 매끄럽게 수렴하며 매우 안정적인 학습 곡선을 보였다.
- **손실 함수 분석:** ELU는 일정 반복 이후 Validation Loss가 더 이상 감소하지 않고 정체되거나 상승하는 반면, TERELU는 반복 횟수가 늘어남에 따라 Validation Loss가 지속적으로 감소하는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 선형 정류기가 가진 '무제한적인 양수 증가' 특성이 심층 네트워크에서 가중치 폭주와 과적합의 주범이 될 수 있음을 지적하였다. TERELU는 양수 영역에 임계값 $\mu$를 도입하여 이를 제어함으로써 다음과 같은 이점을 얻었다.

첫째, **수축 기울기(Contracting Gradient)**의 도입이다. 모든 입력에 대해 기울기가 $1$인 ReLU 계열과 달리, 고입력 영역에서 기울기를 제어함으로써 가중치 업데이트의 크기를 적절히 제한하고 이는 곧 일반화 성능 향상으로 이어진다.

둘째, **Zero-mean 특성의 회복**이다. 양방향 saturation을 통해 출력 평균을 0에 가깝게 유지함으로써 Bias Shift 문제를 완화하고, 이는 결과적으로 Natural Gradient와 Normal Gradient 사이의 간극을 줄여 학습 속도를 향상시킨다.

다만, 본 논문에서 제시한 $\beta$의 학습 가능성과 $\mu$의 최적값 설정에 대한 구체적인 가이드라인이나 다양한 데이터셋(예: CIFAR-10, ImageNet)에서의 검증이 부족하다는 점은 한계로 보인다. 또한, 수식상 $x \ge \mu$ 영역에서 지수 함수가 사용되었으나, 저자는 이를 'saturation'과 'contracting'이라고 표현하고 있다. 일반적인 $e^x$는 발산하는 함수이므로, 실제 구현에서 $\beta$가 매우 작게 유지되거나 혹은 수식의 부호가 반대($e^{-(x-\mu)/\mu}$)여야 진정한 의미의 saturation이 가능할 것이라는 비판적 해석이 가능하다.

## 📌 TL;DR

본 연구는 ReLU 계열 활성화 함수가 심층 네트워크에서 유발하는 과적합 및 가중치 폭주 문제를 해결하기 위해, 양수 영역에 임계값 기반의 포화 지점을 도입한 **TERELU**를 제안하였다. 실험 결과, 매우 깊은 FCNN 구조에서 ELU 대비 탁월한 학습 안정성과 일반화 성능을 보였으며, 이는 양방향 saturation을 통한 zero-mean 특성 확보와 가중치 업데이트 제어 덕분인 것으로 분석된다. 이 연구는 향후 초심층 신경망의 학습 안정화 및 정규화 기법 연구에 기여할 가능성이 크다.