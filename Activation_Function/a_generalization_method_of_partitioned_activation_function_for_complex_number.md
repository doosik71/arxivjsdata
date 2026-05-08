# A Generalization Method of Partitioned Activation Function for Complex Number

HyeonSeok Lee, Hyo Seon Park (2018)

## 🧩 Problem to Solve

본 논문은 복소수 인공신경망(Complex Number Artificial Neural Net, 이하 CANN)에서 사용되는 복소수 활성화 함수(Complex Activation Function)의 체계적인 설계 방법론을 제시하는 것을 목표로 한다.

복소수는 MRI 이미지와 같은 전자기 신호 처리, 푸리에 변환(Fourier Transform), 라플라스 변환(Laplace Transform) 기반의 방법론 등 다양한 공학적 문제에서 필수적으로 등장한다. 이러한 문제를 효과적으로 처리하기 위해 CANN의 필요성이 제기되었으며, 합성곱(Convolution)이나 완전 연결(Fully-connection) 층과 같은 기본 구성 요소들은 이미 복소수 범위에서 수학적으로 잘 정의되어 있다. 그러나 활성화 함수(Activation Function)의 경우, 실수 영역에서 정의된 함수를 복소수 영역으로 어떻게 확장할 것인가에 대한 표준화된 방법론이 부족한 상태이다. 특히, 입력값의 범위에 따라 서로 다른 함수를 적용하는 분할 활성화 함수(Partitioned Activation Function)를 복소수 영역으로 일반화하는 문제는 여전히 해결해야 할 과제로 남아 있다.

## ✨ Key Contributions

본 논문의 핵심 기여는 실수 기반의 분할 활성화 함수를 복소수 활성화 함수로 변환할 수 있는 일반화 방법론(Generalization Method)을 제안한 것이다. 제안된 방법론은 사용자의 목적에 따라 선택할 수 있도록 총 4가지 변형(Variation)을 제공한다.

1. **Holomorphic 특성 확보**: 역전파(Back-propagation)를 단순화하여 학습 속도를 높일 수 있는 정칙 함수(Holomorphic function) 형태의 활성화 함수를 생성한다.
2. **복소수 위상 보존**: 입력 신호의 위상각(Phase angle)을 유지해야 하는 경우를 위한 설계를 제공한다.
3. **실수부와 허수부의 상호작용**: 실수부와 허수부가 서로 독립적으로 작동하지 않고 상호 영향을 주고받도록 보장하는 구조를 제공한다.

이러한 일반화 방법론을 실제 널리 쓰이는 LReLU(Leaky ReLU)와 SELU(Scaled Exponential Linear Unit)에 적용함으로써 그 범용성을 입증하였다.

## 📎 Related Works

기존의 복소수 활성화 함수 접근 방식은 크게 세 가지로 분류된다.

1. **No change**: $\tanh$나 $\text{logistic}$ 함수와 같이 이미 복소수 영역에서 정의된 함수를 그대로 사용하는 방식이다. 하지만 이러한 함수들은 원점 근처에서 극점(Pole)이 발생하는 문제가 있다.
2. **Modification**: 실수 활성화 함수의 아이디어를 차용하여 새로운 복소수 함수를 만드는 방식이다. 대표적으로 $\text{modReLU}$가 있으며, 이는 입력의 크기($|z|$)를 기준으로 활성화 여부를 결정하여 위상을 보존하지만, 정칙 함수(Holomorphic)가 아니라는 한계가 있다.
3. **Generalization**: 실수축 상에서의 값과 일치하도록 확장하는 방식이다. 실수부와 허수부에 각각 활성화 함수를 적용하는 $\text{SCReLU}$나, 두 성분이 모두 양수일 때만 활성화하는 $\text{zReLU}$ 등이 있다. 하지만 이러한 방식들은 본질적으로 두 개의 독립적인 실수 활성화 함수를 사용하는 것과 같아 복소수 특성을 충분히 활용하지 못한다는 비판이 있다. $\text{Complex Cardioid (CC)}$는 위상을 보존하지만 정칙 함수가 아니다.

본 논문은 $\text{Complex Cardioid}$에서 영감을 얻어, 분할 활성화 함수를 체계적으로 일반화하는 수식을 제안함으로써 기존 연구의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 일반화 프로세스

본 논문은 실수 기반의 분할 활성화 함수 $f(x)$가 다음과 같은 형태를 가진다고 가정한다.
$$f(x) = \begin{cases} f_0(x) & x < 0 \\ f_1(x) & x \geq 0 \end{cases}$$
이를 헤비사이드 계단 함수(Heaviside unit-step function) $H(x)$를 이용하여 다음과 같이 표현할 수 있다.
$$f(x) = f_0(x)H(-x) + f_1(x)H(x)$$

일반화의 핵심은 실수 영역의 $H(x)$와 $H(-x)$를 복소수 영역의 함수로 대체하는 것이다. 복소수 $z$의 위상각을 $\theta$라고 할 때, 다음과 같은 대체 함수를 제안한다.
$$H(x) \rightarrow \frac{1}{2}(1 + e^{i(2n+1)\theta}), \quad H(-x) \rightarrow \frac{1}{2}(1 - e^{i(2n+1)\theta})$$
여기서 $n$은 정수 파라미터이다. 이를 적용한 일반화된 복소수 활성화 함수 $f(z)$는 다음과 같다.
$$f(z) = \frac{1}{2}(1 - e^{i(2n+1)\theta})f_0(z) + \frac{1}{2}(1 + e^{i(2n+1)\theta})f_1(z)$$

### 4가지 변형 모델

논문은 목적에 따라 위 수식을 다음과 같이 변형하여 제시한다.

1. **Complex-coefficient (Variation 1)**: 위 식 (8)을 그대로 사용한다. 실수부와 허수부의 상호작용이 발생하며 위상과 크기가 모두 변한다.
2. **Cos-coefficient (Variation 2)**: 위상각 보존을 위해 실수 값 스케일을 사용한다.
    $$f(z) = \frac{1}{2}[1 - \cos((2n+1)\theta)]f_0(z) + \frac{1}{2}[1 + \cos((2n+1)\theta)]f_1(z)$$
3. **Abs-coefficient (Variation 3)**: 절댓값을 이용하여 실수 값 스케일을 적용한다.
    $$f(z) = \frac{1}{2}|1 - e^{i(2n+1)\theta}|f_0(z) + \frac{1}{2}|1 + e^{i(2n+1)\theta}|f_1(z)$$
4. **Approximate Generalization (Variation 4 - Holomorphic)**: 정칙 함수를 만들기 위해 $\text{erf}$(오차 함수) 기반의 시그모이드 함수 $S(z)$를 도입한다.
    $$S(z) = \frac{1}{2} \left( 1 + \text{erf}\left(\frac{z}{\sqrt{2\sigma}}\right) \right)$$
    이를 통해 구성된 함수 $\tilde{f}(z) = S(-z)f_0(z) + S(z)f_1(z)$는 $f_0, f_1$이 정칙 함수일 경우 전체 함수 또한 정칙 함수가 된다.

### 적용 사례

- **LReLU**: $\alpha z$와 $z$를 각각 $f_0, f_1$으로 설정하여 $\text{CLReLU}, \text{cLReLU}, \text{aLReLU}, \text{HLReLU}$를 유도하였다.
- **SELU**: $\lambda \alpha(e^z - 1)$와 $\lambda z$를 설정하여 $\text{CSELU}, \text{cSELU}, \text{aSELU}, \text{HSELU}$를 유도하였다.

## 📊 Results

본 논문은 정량적인 벤치마크 실험 결과(정확도, 손실 값 등)를 제시하지 않는다. 대신, 제안한 일반화 방법론을 통해 생성된 각 활성화 함수의 수학적 특성을 비교 분석한 결과를 표(Table 1) 형태로 제공한다.

| 활성화 함수 | Holomorphic | Real-Complex Interaction | Phase Preserving |
| :--- | :---: | :---: | :---: |
| **CLReLU** | X | O | X |
| **cLReLU** | X | X | O |
| **aLReLU** | X | X | O |
| **HLReLU** | O | O | X |
| **CSELU** | X | O | X |
| **cSELU** | X | O | X |
| **aSELU** | X | O | X |
| **HSELU** | O | O | X |

분석 결과, $\text{HLReLU}$와 $\text{HSELU}$는 정칙 함수(Holomorphic)의 특성을 가지며, $\text{cLReLU}$와 $\text{aLReLU}$는 위상 보존 특성을 가짐을 확인할 수 있다.

## 🧠 Insights & Discussion

본 논문은 복소수 활성화 함수를 설계할 때 **정칙성(Holomorphicity)**과 **위상 보존(Phase preservation)** 사이의 트레이드오프가 존재함을 시사한다. 정칙 함수는 역전파 계산을 단순화하여 학습 효율을 높일 수 있다는 강력한 강점이 있지만, 이 과정에서 입력의 위상각이 변하게 된다.

또한, 정칙 함수 기반의 일반화 방법($\text{HLReLU}, \text{HSELU}$)를 사용할 경우, 허수축 방향($\pm i\infty$)으로 특성 값이 발산할 위험이 있다. 이를 방지하기 위해 저자는 $\text{Batch-Renormalization}$과 같은 정규화 기법의 사용을 권장하고 있다.

비판적으로 보자면, 제안된 방법론이 수학적으로는 매우 정교하게 설계되었으나, 실제 데이터셋(예: MRI 데이터 또는 RF 신호 데이터)에 적용했을 때 기존의 $\text{modReLU}$나 $\text{SCReLU}$ 대비 어느 정도의 성능 향상이 있는지에 대한 실증적 증거가 부족하다는 점이 아쉬운 부분이다.

## 📌 TL;DR

본 논문은 실수 영역의 분할 활성화 함수(LReLU, SELU 등)를 복소수 영역으로 확장하기 위한 일반화 프레임워크를 제안하였다. 정칙성 확보, 위상 보존, 실수-허수 상호작용이라는 세 가지 서로 다른 목적에 따라 선택 가능한 4가지 수학적 변형 모델을 제시하였으며, 이는 향후 CANN의 다양한 빌딩 블록으로 활용되어 복소수 기반 딥러닝 모델의 설계 유연성을 높이는 데 기여할 수 있을 것으로 보인다.
