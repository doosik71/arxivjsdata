# EXPLORING THE RELATIONSHIP: TRANSFORMATIVE ADAPTIVE ACTIVATION FUNCTIONS IN COMPARISON TO OTHER ACTIVATION FUNCTIONS

Vladimír Kunc (2024)

## 🧩 Problem to Solve

신경망(Neural Networks)의 성능을 결정짓는 핵심 구성 요소 중 하나는 비선형성을 도입하는 활성화 함수(Activation Function, AF)이다. 현재 문헌에는 400개가 넘는 다양한 활성화 함수가 제안되어 있으며, 데이터에 따라 특성이 변하는 적응형 활성화 함수(Adaptive Activation Function, AAF)는 네트워크 성능을 향상시키는 유망한 방법으로 주목받고 있다.

본 논문이 해결하고자 하는 문제는 수많은 활성화 함수들 사이의 관계를 체계적으로 정리하고, 최근 제안된 Transformative Adaptive Activation Function (TAAF)이 기존의 다양한 활성화 함수들을 얼마나 포괄할 수 있는지를 분석하는 것이다. 즉, TAAF가 단순한 새로운 함수 하나가 아니라, 기존의 수많은 활성화 함수들을 특수 사례(Special Case)로 포함하는 일반화된 프레임워크임을 입증하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 TAAF가 기존 문헌에 제안된 50개 이상의 활성화 함수를 일반화(Generalize)할 수 있음을 수학적으로 증명하고, 추가적으로 70개 이상의 활성화 함수가 TAAF와 유사한 설계 개념을 공유하고 있음을 밝힌 것이다.

중심적인 직관은 임의의 내부 활성화 함수 $f$에 대해 수평/수직 방향의 스케일링(Scaling)과 평행 이동(Translation)을 가능하게 하는 네 가지 파라미터를 도입함으로써, 거의 모든 형태의 변형된 활성화 함수를 하나의 수식으로 표현할 수 있다는 점이다.

## 📎 Related Works

논문은 400개 이상의 활성화 함수를 조사한 기존 서베이 연구[70]를 바탕으로 관련 연구를 분석한다. 기존의 적응형 활성화 함수들은 대개 특정 파라미터 하나만을 도입하거나, 매우 제한적인 형태의 변형(예: 단순한 기울기 조절)만을 제공했다는 한계가 있다.

TAAF는 이러한 개별적인 접근 방식과 달리, 수평 및 수직의 이동과 확장을 동시에 제어하는 통합된 파라미터 세트를 제공함으로써 기존의 단편적인 적응형 함수들보다 더 넓은 표현력과 유연성을 가진다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. TAAF의 정의 및 구조
TAAF는 임의의 내부 활성화 함수 $f$를 감싸는 형태의 함수로, 다음과 같은 방정식으로 정의된다.

$$g(f, z_i) = \alpha_i \cdot f(\beta_i \cdot z_i + \gamma_i) + \delta_i$$

여기서 $z_i$는 활성화 함수의 입력이며, $\alpha_i, \beta_i, \gamma_i, \delta_i$는 각 뉴런마다 독립적으로 학습 가능한(trainable) 파라미터이다. 각 파라미터의 역할은 다음과 같다.

- $\alpha_i$: 수직 스케일링 (Vertical Scaling) - 함수의 출력값 크기를 조절한다.
- $\beta_i$: 수평 스케일링 (Horizontal Scaling) - 입력값의 변화 속도(기울기)를 조절한다.
- $\gamma_i$: 수평 평행 이동 (Horizontal Translation) - 함수를 좌우로 이동시킨다.
- $\delta_i$: 수직 평행 이동 (Vertical Translation) - 함수를 상하로 이동시킨다.

### 2. 뉴런 내에서의 연산 절차
실제 뉴런에서 TAAF가 적용될 때, 입력 $x_i$와 가중치 $w_i$에 의한 사전 활성화 값(pre-activation)이 TAAF의 입력 $z_i$가 된다. 전체 연산 과정은 다음과 같다.

$$\alpha_i \cdot f\left(\beta_i \cdot \sum_{i=1}^{n} w_i x_i + \gamma_i\right) + \delta_i$$

### 3. 기존 함수의 TAAF화 (Generalization)
저자는 기존의 수많은 활성화 함수들이 TAAF의 파라미터 $\alpha, \beta, \gamma, \delta$에 특정 값을 할당하고 내부 함수 $f$를 선택함으로써 구현될 수 있음을 보여준다. 예를 들어:
- **Scaled Hyperbolic Tangent**: $f(z) = \tanh(z)$이며, $\alpha=a, \beta=b, \gamma=0, \delta=0$으로 설정한 경우이다.
- **Shifted and Scaled Sigmoid (SSS)**: $f(z) = \sigma(z)$이며, $\alpha=1, \beta=a, \gamma=-ab, \delta=0$으로 설정한 경우이다.
- **FReLU**: $f(z) = \text{ReLU}(z)$이며, $\alpha=1, \beta=1, \gamma=a_i, \delta=b_i$로 설정하여 수평/수직 이동을 학습하는 경우이다.

## 📊 Results

본 논문은 새로운 실험 데이터를 제시하는 대신, 기존 문헌에 존재하는 활성화 함수들을 TAAF 프레임워크로 매핑한 정성적 분석 결과를 제시한다.

### 1. 특수 사례 분석 (Table 1)
논문은 50개 이상의 활성화 함수가 TAAF의 특수 사례임을 입증하였다. 분석 대상에는 $\tanh, \sigma, \text{ReLU}, \text{ELU}, \text{Swish}$ 등의 기본 함수와 이를 변형한 수십 가지의 변체들이 포함되었다. 이들은 $\alpha, \beta, \gamma, \delta$ 중 일부를 상수로 고정하거나 특정 값으로 설정함으로써 TAAF의 일부분으로 편입된다.

### 2. 관련 개념 분석 (Table 2)
TAAF의 완전한 특수 사례는 아니지만, TAAF가 추구하는 '파라미터를 통한 함수 변형'이라는 개념을 공유하는 70개 이상의 함수를 식별하였다. 
- **부분적 제어**: LReLU나 PReLU처럼 특정 구간(음수 영역)의 기울기만 조절하는 경우 ($\alpha$와 유사한 개념).
- **확률적 변형**: NReLU처럼 평행 이동 파라미터 $\gamma$에 노이즈를 추가하는 경우.
- **복합 구성**: ABU(Adaptive Blending Units)나 MoGU처럼 여러 개의 TAAF 기반 함수를 가중합(Weighted Sum) 형태로 결합하여 사용하는 경우.

## 🧠 Insights & Discussion

### 강점 및 의의
본 연구는 파편화되어 있던 수백 개의 활성화 함수들을 TAAF라는 하나의 통합된 수학적 틀 안에서 해석했다는 점에서 학술적 가치가 크다. TAAF의 네 가지 파라미터($\alpha, \beta, \gamma, \delta$)가 각각 수직 스케일, 수평 스케일, 수평 이동, 수직 이동이라는 기하학적 의미를 갖는다는 점은, 왜 이 파라미터들이 신경망의 표현력을 높이는지에 대한 이론적 근거를 제공한다.

### 한계 및 논의사항
논문에서는 TAAF가 많은 함수를 일반화할 수 있음을 보였으나, 모든 함수를 포함할 수는 없음을 명시하였다. 예를 들어, 함수의 특정 부분만 기울기를 조절하는 Piecewise 함수(예: Improved Logistic Sigmoid)나, 입력값에 따라 파라미터가 동적으로 결정되는 일부 복잡한 구조는 TAAF의 단순한 선형 변환 식만으로는 표현이 불가능하다.

또한, 본 보고서는 관계 분석에 집중하고 있어, TAAF를 사용했을 때 실제로 어떤 작업에서 어떤 기존 함수보다 성능이 우월한지에 대한 최신 벤치마크 결과는 포함되어 있지 않다. (해당 내용은 인용된 이전 연구 $[70, 71, 72]$에서 다루어지고 있다고 언급된다.)

## 📌 TL;DR

본 논문은 Transformative Adaptive Activation Function (TAAF)이 단순한 하나의 활성화 함수가 아니라, 기존의 수많은 활성화 함수들을 포함하는 **일반화된 프레임워크**임을 입증한 기술 보고서이다. TAAF는 네 개의 학습 가능한 파라미터($\alpha, \beta, \gamma, \delta$)를 통해 임의의 내부 함수를 수평/수직으로 이동 및 스케일링하며, 이를 통해 50개 이상의 기존 함수를 완벽히 재현하고 70개 이상의 함수와 개념적 유사성을 공유한다. 이 연구는 향후 신경망 설계 시 활성화 함수를 개별적으로 선택하는 대신, TAAF와 같은 유연한 프레임워크를 통해 데이터에 최적화된 함수 형태를 직접 학습시키는 방향의 중요성을 시사한다.