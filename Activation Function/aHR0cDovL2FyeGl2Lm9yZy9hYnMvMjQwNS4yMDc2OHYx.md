# Expanded Gating Ranges Improve Activation Functions

Allen Hao Huang (2024)

## 🧩 Problem to Solve

딥러닝 아키텍처의 핵심 구성 요소인 활성화 함수(Activation Function)는 모델에 비선형성을 도입하여 복잡한 데이터 패턴을 학습할 수 있게 한다. 현재 가장 널리 사용되는 GELU와 SiLU 같은 함수들은 'Self-gated' 구조를 가지고 있으며, 이들의 게이팅 함수(Gating function)는 출력 범위가 $0$과 $1$ 사이로 제한되어 있다.

본 논문은 기존의 활성화 함수들이 관습적으로 따랐던 $0$에서 $1$ 사이의 게이팅 범위 제한이 반드시 최적인가라는 의문에서 출발한다. 특히, 기존의 ReLU-like 특성(입력이 $-\infty$일 때 $0$으로 수렴하고, $+\infty$일 때 $x$로 수렴하는 성질)이 성능 향상을 위해 필수적인 요소인지 검증하고, 게이팅 범위를 확장함으로써 활성화 함수의 성능을 개선하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 게이팅 함수의 출력 범위를 $[0, 1]$에서 $[-\alpha, 1 + \alpha]$로 확장하는 학습 가능한 파라미터 $\alpha$를 도입하는 것이다. 주요 기여 사항은 다음과 같다.

1. **$\arctan$의 게이팅 가능성 증명**: $\arctan$ 함수를 게이팅 메커니즘으로 사용하는 ArcTan Linear Unit(ATLU)을 제안하고, 범위 확장 기술을 통해 이것이 기존의 SOTA 활성화 함수들과 경쟁하거나 이를 능가할 수 있음을 보였다.
2. **Expanded Gating Ranges 제안**: 기존의 GELU와 SiLU에도 범위 확장 개념을 적용한 xGELU, xSiLU를 제안하여 성능 향상을 이끌어냈다.
3. **1차 GLU의 성능 개선**: 게이팅 범위 확장이 1차 Gated Linear Units(GLU)의 성능을 높여, 상대적으로 복잡한 2차 GLU와의 성능 격차를 줄일 수 있음을 입증했다.
4. **ReLU-like 특성에 대한 재고**: 실험을 통해 활성화 함수가 반드시 ReLU와 유사한 수렴 특성을 가질 필요가 없으며, 오히려 적절한 '음수 기울기 흐름(Negative gradient flow)'을 허용하는 것이 더 중요함을 시사했다.

## 📎 Related Works

기존의 활성화 함수 연구는 주로 ReLU의 한계를 극복하는 방향으로 진행되었다. ReLU는 양수 입력에 대해 일정한 기울기를 유지하여 기울기 소실(Vanishing gradient) 문제를 해결했지만, 음수 영역에서 기울기가 $0$이 되는 문제가 있었다. 이를 해결하기 위해 GELU와 SiLU 같은 Smooth ReLU 변형들이 등장했으며, 이들은 연속적으로 미분 가능한 게이팅 함수를 사용하여 성능을 개선했다.

기존 연구들은 대부분 게이팅 함수의 범위를 $[0, 1]$로 제한하는 ReLU-like 특성을 유지하는 데 집중했다. 본 논문은 이러한 고정된 범위가 오히려 최적의 학습을 방해할 수 있다고 보며, 학습 가능한 파라미터를 통해 범위를 유연하게 조정하는 접근 방식을 취함으로써 기존 방식과 차별화를 둔다.

## 🛠️ Methodology

### 1. Self-Gated Activation Functions 기본 구조

Self-gated 활성화 함수는 다음과 같은 일반식으로 표현된다:
$$a(x) = g(x) \times x$$
여기서 $x$는 입력값, $g(x)$는 게이팅 함수이다. ReLU, GELU, SiLU 모두 $g(x)$의 범위가 $[0, 1]$인 구조를 가진다.

### 2. ArcTan Linear Unit (ATLU)

저자는 $\arctan$ 함수가 연속 미분 가능하고 단조 증가하며 범위가 $(-\pi/2, \pi/2)$라는 점에 주목하여, 이를 $(0, 1)$ 범위로 스케일링한 ATLU를 정의했다:
$$\text{ATLU}(x) = x \times \left( \frac{\arctan(x) + \pi/2}{\pi} \right)$$
하지만 기본 ATLU는 ReLU-like 특성을 가지지 않아 단독으로는 성능이 낮았다.

### 3. Expanded Gating Ranges (xATLU, xGELU, xSiLU)

범위 제한 문제를 해결하기 위해, 각 MLP 블록마다 학습 가능한 스칼라 파라미터 $\alpha$ (초기값 $0$)를 도입하여 게이팅 범위를 $[-\alpha, 1 + \alpha]$로 확장한다. xATLU의 수식은 다음과 같다:
$$\text{xATLU}(x, \alpha) = x \times \left( \frac{\arctan(x) + \pi/2}{\pi} \times (1 + 2\alpha) - \alpha \right)$$
동일한 원리를 GELU와 SiLU에 적용하여 xGELU와 xSiLU를 구성한다. $\alpha$가 커질수록 게이팅 범위가 양방향으로 확장되며, 이는 기울기의 범위를 넓히는 효과를 준다.

### 4. Gated Linear Units (GLU) 확장

GLU는 두 입력의 요소별 곱으로 정의되며, 본 논문에서는 다음 두 가지 형태를 실험했다:

- **1차 GLU**: $a(x, y) = g(x) \times y$
- **2차 GLU**: $a(x, y) = g(x) \times x \times y$
제안한 범위 확장 기법을 1차 GLU에 적용하여 xATGLU, xGEGLU, xSwiGLU를 구현했으며, 이를 통해 2차 GLU 수준의 성능을 목표로 했다.

## 📊 Results

### 실험 설정

- **데이터셋**: OpenWebText2
- **모델**: Transformer 기반 autoregressive language model (nanoGPT 기반)
- **하드웨어**: 단일 NVIDIA A100 GPU
- **지표**: Perplexity (낮을수록 우수)

### 주요 결과

1. **Self-Gated 함수 성능**:
    - $\alpha$를 학습 가능하게 설정했을 때, **xATLU $\rightarrow$ xSiLU $\approx$ xGELU $\rightarrow$ GELU/SiLU** 순으로 성능이 좋았다.
    - 특히 xATLU는 기존의 GELU와 SiLU보다 낮은 Perplexity를 기록하며 가장 우수한 성능을 보였다.
2. **GLU 성능**:
    - 범위 확장은 1차 GLU의 성능을 크게 향상시켰다.
    - 실험 결과, **1차 xATGLU, xGEGLU, xSwiGLU**의 성능이 **2차 GEGLU, SwiGLU**와 대등한 수준까지 올라왔음을 확인했다.
3. **Ablation Study (xATLU 중심)**:
    - **학습 가능 $\alpha$ vs 고정 $\alpha$**: 학습 가능한 $\alpha$가 더 우수한 성능을 보였다.
    - **음수 범위 확장($-\alpha$) vs 양수 범위 확장($1+\alpha$)**: 게이팅 범위의 하한선을 낮추는 것($-\alpha$)이 성능 향상에 결정적인 영향을 미쳤다. 상한선만 높이는 것은 효과가 미미했다. 이는 **음수 기울기의 흐름(Negative gradient flow)**을 확보하는 것이 매우 중요하다는 것을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 활성화 함수의 설계가 단순히 수학적 직관이나 trial-and-error에 의존하는 것이 아니라, 게이팅 범위라는 구체적인 하이퍼파라미터 제어를 통해 최적화될 수 있음을 보였다. 특히 xATLU가 가장 좋은 성능을 낸 이유는 ATLU의 1차 도함수가 단조 증가하기 때문이며, 이것이 더 유리한 학습 역학(training dynamics)을 제공한다고 분석된다.

### 한계 및 비판적 해석

- **실험 규모의 제한**: 단일 GPU 환경에서 상대적으로 작은 규모의 모델로 실험이 진행되었다. LLM 급의 초대형 모델에서도 동일한 경향성이 나타날지는 추가 검증이 필요하다.
- **희소성(Sparsity) 문제**: ReLU-like 함수들은 출력이 $0$이 되는 구간이 있어 활성화 희소성을 가지는데, 범위를 확장하면 이러한 특성이 사라진다. 이는 메모리 효율성이나 계산 효율성 측면에서 손해를 볼 수 있으며, 'ReLUfication' 같은 기법 적용이 어려워질 수 있다는 점이 한계로 지적된다.

### 결론적 논의

결과적으로 본 연구는 "활성화 함수는 $0$과 $1$ 사이의 게이트를 가져야 한다"는 기존의 믿음을 깨뜨렸다. $\alpha$를 통한 범위 확장은 모델이 데이터에 맞게 기울기의 흐름을 스스로 조절하게 함으로써, 더 효율적인 학습을 가능하게 한다.

## 📌 TL;DR

이 논문은 활성화 함수의 게이팅 범위를 $[0, 1]$에서 $[-\alpha, 1+\alpha]$로 확장하는 학습 가능한 파라미터 $\alpha$를 도입하여 성능을 개선했다. 이를 통해 제안된 **xATLU**는 기존의 GELU와 SiLU를 능가하는 성능을 보였으며, 1차 GLU의 성능을 2차 GLU 수준으로 끌어올렸다. 연구의 핵심은 **음수 기울기의 흐름을 허용하는 것이 모델 성능 향상에 필수적**이라는 점을 밝힌 것이며, 이는 향후 더 효율적인 활성화 함수 설계 및 하이퍼파라미터 최적화 연구에 중요한 실마리를 제공한다.
