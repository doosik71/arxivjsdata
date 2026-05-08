# Activation Functions in Artificial Neural Networks: A Systematic Overview

Johannes Lederer

## 🧩 Problem to Solve

인공신경망, 특히 딥러닝에서 활성화 함수는 신경망의 출력 형태를 결정하는 핵심 요소입니다. logistic이나 relu와 같은 일부 활성화 함수는 오랫동안 사용되어 왔지만, 딥러닝이 주류 연구 분야가 되면서 수많은 새로운 활성화 함수들이 등장하여 이론과 실제 모두에서 혼란을 야기하고 있습니다. 이 논문은 이러한 혼란을 해소하고, 인기 있는 활성화 함수들의 특성을 체계적으로 분석하여 연구자와 실무자에게 유용한 최신 정보를 제공하는 것을 목표로 합니다.

## ✨ Key Contributions

- **체계적인 개요 제공:** 인기 있는 활성화 함수들에 대한 분석적이고 최신 정보를 담은 포괄적인 개요를 제공합니다.
- **수학적 특성 분석:** 각 활성화 함수의 수학적 특성(예: 미분 가능성, 도함수, 출력 범위)을 면밀히 검토하고 설명합니다.
- **실제적 영향 논의:** 각 함수의 계산 복잡성, 표현 능력, 소실/폭발 기울기 문제 해결 능력 등 실제 적용 시의 효과를 논의합니다.
- **주요 활성화 함수 분류 및 비교:** 시그모이드(sigmoid), 조각별 선형(piecewise-linear), 기타 함수로 분류하여 장단점을 비교합니다.
- **권장 활성화 함수 제시:** 계산 효율성과 성능을 고려하여 softsign 및 relu를 현재의 표준 활성화 함수로 제안합니다.
- **향후 연구 방향 제시:** 이론적 고려 사항, 자동화된 탐색, 데이터 적응적 선택 방식 등 활성화 함수 개선을 위한 유망한 접근 방식을 강조합니다.

## 📎 Related Works

이 논문은 인공 신경망의 생물학적 동기(McCulloch and Pitts, 1943; Rosenblatt, 1958)에서부터 시작하여 현대 딥러닝(Goodfellow et al., 2016; LeCun et al., 2015; Schmidhuber, 2015)에 이르기까지 광범위한 선행 연구를 참조합니다. 특히, perceptron(Rosenblatt, 1958), softsign(Elliott, 1993), relu(Maas et al., 2013), softplus(Dugas et al., 2001), elu(Clevert et al., 2016), selu(Klambauer et al., 2017), swish(Ramachandran et al., 2017; Elfwing et al., 2018), maxout(Goodfellow et al., 2013) 등 다양한 활성화 함수 및 관련 개념을 소개한 주요 연구들을 인용하고 있습니다. 또한, 활성화 함수 학습(He et al., 2015; Trottier et al., 2017) 및 최적화 알고리즘(Bubeck, 2015)에 대한 논의도 포함합니다.

## 🛠️ Methodology

이 논문은 활성화 함수에 대한 체계적이고 분석적인 검토 방법론을 따릅니다.

1. **분류:** 활성화 함수를 세 가지 주요 범주(시그모이드 함수, 조각별 선형 함수, 기타 함수)로 분류합니다.
2. **수학적 분석:**
   - 각 함수의 정의와 출력 범위를 명시합니다.
   - 첫 번째 및 두 번째 도함수를 계산하고 이들의 속성(예: 미분 가능성, 도함수의 범위)을 분석합니다. 미분 불가능한 지점에서는 방향 도함수(directional derivatives) 개념을 활용합니다.
   - logistic ($f_{log}[z] = 1 / (1 + e^{-z})$), tanh ($f_{tanh}[z] = (e^z - e^{-z}) / (e^z + e^{-z})$), arctan ($f_{arctan}[z] = \arctan[z]$), softsign ($f_{soft}[z] = z / (1 + |z|)$)과 같은 시그모이드 함수들의 S-자형 곡선과 미분 가능성을 중점적으로 다룹니다.
   - linear ($f_{linear}[z] = z$), relu ($f_{relu}[z] = \max\{0, z\}$), leakyrelu ($f_{lrelu, a}[z] = \max\{0, z\} + \min\{0, az\}$)와 같은 조각별 선형 함수들의 비선형성 및 불연속 미분 지점을 분석합니다.
   - softplus ($f_{soft+}[z] = \log[1 + e^z]$), elu, selu, swish 등 최신 함수들의 고유한 수학적 특성을 탐구합니다.
3. **실제적 영향 논의:**
   - 계산 비용(exponential 또는 trigonometric 함수 포함 여부)을 평가합니다.
   - 네트워크의 표현 능력(예: linear 네트워크의 한계)에 대한 영향을 논의합니다.
   - 소실/폭발 기울기 문제(vanishing-gradient problem, dying-relu phenomenon)와의 관련성을 탐구하고, 이를 완화하기 위한 각 함수의 특성을 분석합니다.
   - selu의 자가 정규화(self-normalization) 특성 및 swish의 비단조성(non-monotonicity)과 같은 특정 기능의 의미를 설명합니다.

## 📊 Results

논문은 여러 활성화 함수들을 세 가지 주요 범주로 나누어 분석하고 다음과 같은 결과를 제시합니다.

- **시그모이드 함수 (Sigmoid Functions):**

  - logistic, arctan, tanh, softsign으로 구성되며, 모두 유계(bounded)이고 미분 가능하며 S자형 곡선을 가집니다.
  - tanh는 logistic의 확장 버전이며, arctan과 유사한 대칭성을 가집니다.
  - softsign은 지수 함수를 포함하지 않아 계산 비용이 저렴하며, 도함수가 원본 함수로 간단하게 표현되어 효율적입니다.
  - 주로 소실 기울기(vanishing gradient) 문제에 취약하다는 공통적인 단점이 있습니다.

- **조각별 선형 함수 (Piecewise-Linear Functions):**

  - linear, relu, leakyrelu로 구성되며, 계산 비용이 매우 저렴하며, 도함수가 0으로 수렴하지 않아 소실 기울기 문제를 완화할 수 있습니다.
  - linear는 표현 능력이 낮아 단독 사용 시 선형 모델만 학습할 수 있습니다.
  - relu는 딥러닝에서 널리 사용되지만, "죽은 ReLU(dying ReLU)" 현상($z \le 0$일 때 기울기가 0이 되어 업데이트가 멈추는 현상)에 취약할 수 있습니다.
  - leakyrelu는 음수 입력에 대해 작은 기울기 $a$ ($0 < a < 1$)를 부여하여 dying ReLU 문제를 완화합니다.

- **기타 함수 (Other Functions):**
  - **softplus ($f_{soft+}[z] = \log[1 + e^z]$):** relu의 매끄러운(smooth) 버전이며, 모든 구간에서 미분 가능합니다. 하지만 relu보다 계산 비용이 높습니다.
  - **elu ($f_{elu, a}[z] = z$ for $z \ge 0$, $a(e^z - 1)$ for $z < 0$):** relu와 유사하지만 음수 입력에 대해 0이 아닌 출력을 제공하여 dying ReLU를 방지합니다. 특정 파라미터 $a=1$에서 한 번 미분 가능합니다.
  - **selu (scaled exponential linear unit):** elu의 변형으로, 특정 파라미터 값($a_0 \approx 1.05, b_0 \approx 1.76$)을 통해 입력 분포를 정규화하는 self-normalizing 속성을 가집니다.
  - **swish ($f_{swish, a}[z] = z \cdot f_{log}[az]$):** relu와 linear 사이를 보간하는 비단조(non-monotonic) 함수입니다. 자동화된 탐색으로 발견된 첫 번째 활성화 함수 중 하나입니다.
  - **학습 가능한 활성화 함수 (Learning Activation Functions):** prelu, pelu, maxout과 같이 활성화 함수의 파라미터를 훈련 중 학습하거나 여러 선형 함수 중 최댓값을 취하는 방식 등, 데이터에 따라 활성화 패턴을 조절하려는 시도들입니다. maxout은 relu의 일반화로 볼 수 있습니다.

## 🧠 Insights & Discussion

- **현재의 표준:** softsign과 relu는 그들의 계산적 단순성과 합리적인 성능으로 인해 현재 딥러닝 실무에서 표준 활성화 함수로 권장됩니다. softsign은 시그모이드 계열 중 계산 효율성이 뛰어나고, relu는 조각별 선형 함수 계열 중 가장 널리 사용됩니다.
- **이론적 정당화의 부족:** 많은 새로운 활성화 함수들이 제안되었지만, 대부분은 relu나 softsign에 비해 명확한 경험적 또는 수학적 이점을 입증하지 못하고 있습니다. 특히 softplus, elu 등 relu의 매끄러운(smooth) 버전을 포함한 함수들은 추가적인 계산 비용이 드는 반면 실질적인 성능 향상 증거는 불분명합니다.
- **유망한 개선 접근 방식:**
  - **이론적 고려 사항:** selu의 자가 정규화 속성처럼 이론적으로 견고한 기반을 가진 함수 개발.
  - **자동화된 탐색:** swish와 같이 기계 학습을 통해 최적의 활성화 함수를 탐색하는 방법.
  - **데이터 적응적 선택:** 훈련 중 활성화 함수의 매개변수를 학습하거나 여러 함수 중에서 선택하는 방식(예: prelu, pelu, maxout). 이러한 접근 방식은 잠재력이 크지만, 파라미터 공간 증가 및 과적합 위험과 같은 실질적인 과제를 안고 있습니다.
- **향후 연구의 필요성:** 새로운 활성화 함수들의 실질적인 이점, 특히 배치 정규화(batch normalization)와 같은 다른 정규화 기법과의 상호작용에 대한 추가적인 경험적 및 이론적 탐구가 필요합니다.

## 📌 TL;DR

이 논문은 급증하는 인공신경망 활성화 함수들에 대한 체계적인 분석을 제공하여 이론적 혼란을 해소하고 실용적인 가이드라인을 제시합니다. 시그모이드, 조각별 선형, 기타 함수로 분류하여 각 함수의 수학적 특성과 실제적 영향을 검토합니다. 결론적으로 softsign과 relu를 계산 효율성과 성능 면에서 현재 표준 활성화 함수로 권장하며, selu의 이론적 기반, swish의 자동 탐색, prelu와 같은 학습 가능한 파라미터를 통한 데이터 적응적 방식이 향후 활성화 함수 개선의 유망한 방향이 될 수 있음을 시사합니다.
