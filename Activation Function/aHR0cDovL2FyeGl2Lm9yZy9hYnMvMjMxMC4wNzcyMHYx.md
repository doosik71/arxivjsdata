# Parametric Leaky Tanh: A New Hybrid Activation Function for Deep Learning

Stamatis Mastromichalakis (2021)

## 🧩 Problem to Solve

본 논문은 심층 신경망(Deep Neural Networks, DNNs)의 성능에 결정적인 영향을 미치는 활성화 함수(Activation Function, AF)의 고질적인 문제들을 해결하고자 한다. 구체적으로는 다음과 같은 문제들에 집중한다.

- **Dying ReLU 문제**: 전통적인 ReLU 함수는 입력값이 0보다 작을 때 기울기(gradient)가 0이 되어 뉴런이 더 이상 학습되지 않고 '죽어버리는' 현상이 발생한다.
- **Vanishing Gradient 문제**: Tanh나 Sigmoid와 같은 전통적인 활성화 함수는 입력값이 매우 크거나 작을 때 기울기가 0에 수렴하여, 역전파 과정에서 기울기가 사라지는 문제가 발생한다.
- **학습의 불안정성**: 기존의 ReLU 변형 함수들(LReLU 등)이 일부 개선을 이루었으나, 여전히 복잡한 분류 작업에서 수렴 속도가 느리거나 지역 최솟값(local minima)에 빠지는 등의 강건성(robustness) 문제가 존재한다.

따라서 본 연구의 목표는 Tanh의 부드러운 비선형성과 Leaky ReLU(LReLU)의 기울기 유지 특성을 결합한 새로운 하이브리드 활성화 함수인 **Parametric Leaky Tanh (PLTanh)**를 제안하여 DNN의 성능과 일반화 능력을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Tanh와 LReLU의 장점을 단일 함수로 통합하는 것이다. 

- **Tanh의 장점 활용**: Tanh는 출력이 $[-1, 1]$ 범위로 제한되어 있어 이상치(outlier)에 강건하며, 원점을 중심으로 대칭적인 특성을 가져 데이터의 중심화(centering) 효과를 제공한다.
- **LReLU의 장점 활용**: LReLU는 음수 입력 영역에서도 작은 기울기를 유지함으로써 'Dying ReLU' 문제를 방지하고 모든 뉴런이 학습 과정에 참여하도록 한다.
- **파라미터 $\alpha$ 도입**: 하이퍼파라미터 $\alpha$를 통해 다양한 데이터 분포에 맞춰 함수의 형태를 조정할 수 있는 유연성을 제공한다.

## 📎 Related Works

논문에서는 ReLU 및 그 변형 함수들에 대해 설명하며 기존 접근 방식의 한계를 지적한다.

- **ReLU 및 변형들**: ReLU는 계산 효율성이 높지만 Dying ReLU 문제가 있으며, 이를 해결하기 위해 Leaky ReLU(LReLU), Parametric ReLU(PReLU), Randomised ReLU(RReLU), Concatenated ReLU(CReLU) 등이 제안되었다. 특히 LReLU는 음수 영역에 작은 기울기를 추가하여 개선을 시도했다.
- **기타 최신 함수**: QReLU/m-QReLU, ALReLU, SigmoReLU 등이 언급되었으나, 여전히 일부 작업에서는 수렴 속도 저하나 지역 최솟값 함몰 문제가 발생한다고 설명한다.
- **차별점**: 제안된 PLTanh는 단순히 기울기를 추가하는 것을 넘어, Tanh의 유계(bounded) 특성과 LReLU의 지속적 기울기 특성을 결합하여 더 복잡한 비선형 관계를 학습할 수 있도록 설계되었다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 PLTanh 정의
PLTanh는 $\tanh(x)$와 $\alpha \cdot |x|$ 중 더 큰 값을 선택하는 방식으로 정의된다. 수학적 정의는 다음과 같다.

$$f(x) = \max(\tanh(x), \alpha \cdot |x|)$$

여기서 $\alpha$는 학습 가능한 파라미터 혹은 최적화된 하이퍼파라미터이다. 이 구조는 $x$가 0 근처일 때는 $\tanh(x)$의 비선형적 특성을 따르고, $x$의 절대값이 커질 때는 $\alpha \cdot |x|$의 선형적 특성을 따라 기울기가 소멸되는 것을 방지한다.

### 미분 함수 (Derivative)
PLTanh의 미분 함수 $\frac{dy}{dx}$는 입력값 $x$와 $\alpha$의 값에 따라 다음과 같이 조건부로 정의된다 (여기서 $\alpha = 0.01$ 기준).

$$
\frac{dy}{dx} = 
\begin{cases} 
0.01 & \text{if } x > 0 \text{ and } 0.01x \ge \tanh(x) \\
-0.01 & \text{if } x < 0 \text{ and } 0.01x + \tanh(x) \le 0 \\
\text{sech}^2(x) & \text{otherwise}
\end{cases}
$$

논문에서는 'intermediate scenarios' (예: $x \ge 0$ 이면서 $0.01x < \tanh(x)$ 인 경우 등)에 대해 구체적인 미분 정의가 명시되지 않았음을 언급하며, 그 외의 일반적인 경우에는 Tanh의 미분 형태인 $\text{sech}^2(x)$를 따른다고 설명한다.

### 학습 절차 및 구현
- **최적화**: 모든 모델은 Adam Optimizer를 사용하여 학습되었다.
- **하이퍼파라미터 튜닝**: PLTanh의 $\alpha$ 값은 각 데이터셋에 대해 Bayesian Optimization을 통해 최적의 값을 탐색하여 설정하였다.
- **구현**: Keras 프레임워크를 사용하여 구현되었다.

## 📊 Results

### 실험 환경
- **데이터셋**: MNIST, Fashion MNIST, TensorFlow Flowers, CIFAR-10, Histopathologic Cancer Detection 등 5가지의 다양한 데이터셋을 사용하였다.
- **모델 아키텍처**: 데이터셋의 특성에 따라 서로 다른 CNN 구조를 사용하였다. (예: MNIST는 단순 CNN, CIFAR-10 및 Cancer Detection은 Batch Normalization과 Dropout이 포함된 더 깊은 CNN 사용)
- **비교 대상**: ReLU, LReLU, ALReLU.
- **평가 지표**: Accuracy, Macro Precision, Macro Recall, Macro F1-score, AUC.
- **검증 방법**: 5-Fold Cross-validation을 통해 결과의 신뢰성을 확보하였다.

### 주요 결과
정량적 결과는 다음과 같다 (표 1 기반).

- **이미지 분류 성능**: MNIST, Fashion MNIST, TF Flowers, CIFAR-10 데이터셋에서 PLTanh는 비교 대상인 ReLU, LReLU, ALReLU보다 전반적으로 높은 Accuracy와 F1-score를 기록하였다. 
    - 특히 CIFAR-10에서는 Accuracy $85.87\%$를 기록하며 타 함수들을 앞섰다.
    - $\alpha$ 값은 데이터셋마다 다르게 설정되었으며, CIFAR-10의 경우 $\alpha=0.4$에서 최적의 성능을 보였다.
- **특이 사항**: Histopathologic Cancer Detection 데이터셋에서는 PLTanh의 성능(Accuracy $86.68\%$)이 ReLU나 LReLU보다 약간 낮게 측정되었다.
- **판별력**: 모든 활성화 함수에서 AUC 점수가 매우 높게 나타나, 양성 및 음성 클래스에 대한 판별력은 전반적으로 우수함을 확인하였다.

## 🧠 Insights & Discussion

### 강점
PLTanh는 Tanh의 중심화 특성과 LReLU의 기울기 유지 특성을 결합함으로써, 다양한 이미지 분류 작업에서 기존 함수들보다 우수한 성능을 보였다. 특히 파라미터 $\alpha$를 통해 데이터셋의 특성에 맞게 조정할 수 있다는 점이 성능 향상에 기여한 것으로 분석된다.

### 한계 및 비판적 해석
- **일반화의 한계**: 암 진단 데이터셋(Histopathologic Cancer Detection)에서는 오히려 성능이 소폭 하락하였다. 이는 PLTanh가 모든 도메인의 데이터에서 절대적인 우위를 가진다고 보기 어려우며, 데이터의 특성에 따라 최적의 AF가 다를 수 있음을 시사한다.
- **수학적 엄밀성 부족**: 논문 내에서 미분 함수의 '중간 시나리오(intermediate scenarios)'에 대한 정의가 명시적으로 제공되지 않았다. 활성화 함수의 미분 가능성과 연속성은 학습 안정성에 직결되는 문제이므로, 이 부분에 대한 명확한 수학적 정의가 누락된 점은 한계로 지적될 수 있다.
- **계산 복잡도**: $\max$ 연산과 $\tanh$, 절대값 연산이 동시에 들어가므로, 단순 ReLU에 비해 연산 비용이 증가했을 가능성이 있으나 이에 대한 분석은 제시되지 않았다.

## 📌 TL;DR

본 논문은 Tanh의 유계 특성과 Leaky ReLU의 기울기 보존 특성을 결합한 하이브리드 활성화 함수 **PLTanh**를 제안하였다. 실험 결과, 대부분의 이미지 분류 데이터셋에서 ReLU 및 그 변형들보다 우수한 성능을 보였으며, 특히 Dying ReLU와 Vanishing Gradient 문제를 동시에 완화할 수 있음을 입증하였다. 이 연구는 향후 더 넓은 범위의 작업에서 일반화 성능을 높이기 위한 AF 파라미터 최적화 연구에 기여할 가능성이 크다.