# A Method on Searching Better Activation Functions

Haoyuan Sun, Zihao Wu, Bo Xia, Pu Chang, Zibin Dong, Yifu Yuan, Yongzhe Chang, Xueqian Wang (2024)

## 🧩 Problem to Solve

인공신경망(ANN)의 성공은 비선형성을 도입하여 데이터의 복잡한 관계를 모델링할 수 있게 하는 활성화 함수(Activation Function, AF)의 선택에 크게 의존한다. 그러나 지금까지 새로운 활성화 함수들의 탐색은 대부분 이론적 근거보다는 경험적인 지식(empirical knowledge)에 의존하여 진행되어 왔다. 이러한 접근 방식은 성능 향상을 가져왔음에도 불구하고, 왜 특정 함수가 더 나은 성능을 보이는지에 대한 수학적 설명이 부족하며, 이는 더 효율적인 활성화 함수를 체계적으로 찾는 것을 방해하는 요소가 된다.

본 논문의 목표는 활성화 함수 탐색을 위한 이론적인 프레임워크를 제공하는 것이다. 구체적으로는 정보 엔트로피(Information Entropy)의 관점에서 최악의 활성화 함수가 존재함을 이론적으로 증명하고, 이를 바탕으로 활성화 함수를 최적화할 수 있는 방법론을 제시하여 실제 성능이 향상된 새로운 활성화 함수를 도출하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1.  **최악의 활성화 함수(WAFBC) 증명**: 경계 조건이 주어진 상황에서 정보 엔트로피를 최대화하는 '최악의 활성화 함수(Worst Activation Function with Boundary Conditions, WAFBC)'가 존재함을 이론적으로 증명하였다. 이는 활성화 함수 성능 개선의 이론적 출발점을 제공한다.
2.  **EAFO 방법론 제안**: 엔트로피 기반 활성화 함수 최적화(Entropy-based Activation Function Optimization, EAFO) 방법론을 제안하였다. 이는 정적인 활성화 함수 설계뿐만 아니라, 반복적인 학습 과정에서 활성화 함수를 동적으로 최적화할 수 있는 가능성을 제시한다.
3.  **CRReLU 도출 및 검증**: EAFO 방법론을 ReLU에 적용하여 새로운 활성화 함수인 Correction Regularized ReLU (CRReLU)를 도출하였다. 이를 Vision Transformer(ViT) 및 대규모 언어 모델(LLM) 파인튜닝 실험을 통해 기존의 ReLU 변형 함수들보다 우수한 성능을 가짐을 입증하였다.

## 📎 Related Works

기존의 활성화 함수 연구는 크게 두 단계로 나뉜다. 초기에는 Sigmoid나 Tanh와 같은 Squashing 함수들이 사용되었으나, 이는 층이 깊어질수록 Gradient Vanishing(기울기 소실) 문제를 야기했다. 이를 해결하기 위해 ReLU가 등장하며 딥러닝의 비약적인 발전을 이끌었으나, ReLU 역시 음수 입력에 대해 출력이 0이 되어 뉴런이 죽어버리는 Dying ReLU 문제와 출력의 양수 편향(positive bias)으로 인한 다음 층의 시프트 문제가 존재한다.

이를 개선하기 위해 Leaky ReLU, PReLU, ELU, CELU, Swish, Mish 등 다양한 변형 함수들이 제안되었다. 특히 GELU는 최근 BERT, ViT, GPT-4 등 최신 아키텍처에서 광범위하게 사용되며 뛰어난 성능을 보이고 있다. 하지만 논문은 이러한 함수들의 성공이 대부분 경험적 증거에 기반하고 있으며, 수학적인 해석이나 체계적인 설계 방법론이 부족하다는 점을 한계로 지적하며 본 연구의 차별성을 강조한다.

## 🛠️ Methodology

### 전체 파이프라인 및 이론적 배경
본 연구는 정보 엔트로피와 베이즈 오류율(Bayesian Error Rate) 사이의 상관관계에서 출발한다. 이진 분류 문제에서 두 클래스의 분포가 더 많이 겹칠수록 정보 엔트로피가 증가하며, 이는 곧 분류 성능의 저하(베이즈 오류율 증가)로 이어진다. 따라서 활성화 함수를 통해 데이터 분포의 정보 엔트로피를 최소화하는 것이 최적의 활성화 함수를 찾는 방향이 된다.

### 정보 엔트로피 functional과 WAFBC
활성화 함수의 역함수를 $y(x)$라고 할 때, 활성화 함수 통과 후의 데이터 분포 $q(x)$는 $q(x) = p(y(x))y'(x)$로 표현된다. 여기서 $p(x)$는 통과 전의 분포이다. 이때 정보 엔트로피 $H(y(x))$는 다음과 같은 functional로 정의된다.

$$H(y(x)) = -\int q(x) \log q(x) dx = \int G(y'(x), y(x)) dx$$

저자들은 오일러-라그랑주 방정식(Euler-Lagrange equation)을 사용하여 이 functional의 극값을 찾았으며, 그 결과 다음과 같은 WAFBC의 분석적 형태를 도출하였다.

$$f(x) = C_1 \int_{-\infty}^x p(t) dt + C_2$$

이 함수는 정보 엔트로피를 최대화하는 전역 최댓값(global maximum)이며, 이는 곧 가장 성능이 낮은 활성화 함수임을 의미한다. 흥ging하게도 Sigmoid나 Tanh 같은 유계(bounded) 함수들이 이 WAFBC의 형태와 유사하며, 이것이 이들이 ReLU 같은 무계(unbounded) 함수보다 성능이 낮은 이유를 이론적으로 설명해 준다.

### EAFO (Entropy-based Activation Function Optimization)
최적의 활성화 함수(전역 최솟값)는 존재하지 않지만, WAFBC로부터 멀어질수록 성능이 향상된다는 점에 착안하여, 기존의 고성능 함수에서 엔트로피를 더 낮추는 방향으로 수정하는 EAFO를 제안한다. 

테일러 전개(Taylor expansion)를 통해 정보 엔트로피의 1차 항을 줄이기 위한 수정 항(correction term) $\eta(x)$를 다음과 같이 정의한다.

$$\eta(x) = -\left( \frac{p(y(x))}{y''(x)} y'(x) + p'(y(x))y'(x) \right)$$

최적화된 활성화 함수의 역함수는 $g(x) = y(x) + \eta(x)$가 되며, 최종적으로 $g(x)$의 역함수를 구함으로써 최적화된 활성화 함수를 얻는다.

### CRReLU: ReLU에서 도출된 개선 함수
실제 적용을 위해 ReLU를 시작점으로 설정하고 EAFO를 적용하였다. 이때 네트워크가 충분히 크다면 중심극한정리에 의해 데이터 분포 $p(y)$를 가우시안 분포($p(y) = C \cdot e^{-y^2/2}$)로 가정한다.

1.  ReLU의 양수 영역($x \ge 0$)에서 $y(x)=x, y'(x)=1, y''(x)=0$을 적용하여 $\eta(x)$를 계산하면 $\eta(x) = -C \cdot x e^{-x^2/2}$가 된다.
2.  상수 $C$를 학습 가능한 파라미터 $\epsilon$으로 설정하여 모델이 스스로 최적화하게 한다.
3.  최적화된 역함수 $g(x) = x - \epsilon x e^{-x^2/2}$의 근사 역함수를 도출하여 최종적인 CRReLU 식을 완성한다.

최종적인 **CRReLU**의 형태는 다음과 같다.

$$f(x) = \max(0, x) + \epsilon x e^{-x^2/2}$$

## 📊 Results

### 실험 설정
- **데이터셋**: 이미지 분류(CIFAR-10, CIFAR-100, ImageNet-1K), LLM 파인튜닝(SHP, HH).
- **모델**: ViT-Tiny, DeiT-Tiny, TNT-Small, GPT-2.
- **비교 대상(Baselines)**: PReLU, ELU, CELU, GELU, Swish, Mish.
- **하이퍼파라미터**: $\epsilon$은 기본적으로 0.01로 설정하였다.

### 주요 결과
1.  **이미지 분류**: ViT 및 그 변형 모델들을 사용하여 테스트한 결과, CRReLU가 CIFAR-10, CIFAR-100에서 기존 모든 ReLU 변형 함수들보다 높은 Top-1 정확도를 기록하였다. ImageNet-1K에서도 ViT-Tiny 모델에서 가장 높은 성능을 보였다. (단, DeiT-Tiny에서는 GELU가 소폭 높았는데, 이는 GELU로 학습된 교사 모델로부터 지식을 전수받은 distillation 구조의 특성으로 해석된다.)
2.  **LLM 파인튜닝**: GPT-2 모델을 DPO(Direct Preference Optimization) 방법으로 파인튜닝한 결과, 다양한 페널티 계수 $\beta$ 값에서 CRReLU가 GELU보다 Evaluation Margin Reward 및 Accuracy 면에서 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 활성화 함수의 탐색을 단순한 '시행착오'에서 '이론적 최적화'의 영역으로 확장했다는 점에서 큰 의미가 있다. 특히 정보 엔트로피라는 물리적 개념을 도입하여, 왜 유계 함수들이 무계 함수보다 성능이 낮은지를 수학적으로 규명한 점이 인상적이다.

**강점 및 해석**:
- CRReLU는 $\epsilon$이라는 학습 가능 파라미터를 도입함으로써, 가우시안 분포 가정의 한계를 극복하고 실제 데이터 분포에 맞게 스스로 적응할 수 있는 유연성을 갖는다.
- 또한, $x < 0$ 영역에서도 약간의 기울기를 가짐으로써 Dying ReLU 문제를 완화하는 동시에, $x \to -\infty$일 때 0으로 수렴하여 모델의 희소성(sparsity)을 유지하는 두 가지 이점을 동시에 챙겼다.

**한계 및 향후 과제**:
- **비가역 함수로의 확장**: EAFO 방법론은 기본적으로 역함수가 존재해야 함을 전제로 한다. ReLU의 경우 근사적인 방법을 통해 해결했지만, 일반적인 비가역 함수에 대한 체계적인 일반화 방법은 여전히 과제로 남아 있다.
- **동적 최적화의 비용**: 학습 도중 활성화 함수를 동적으로 최적화하는 아이디어는 흥미로우나, 대규모 네트워크에서 계산 복잡도가 폭발적으로 증가하는 문제가 있어 실용적인 알고리즘 개발이 필요하다.

## 📌 TL;DR

본 논문은 정보 엔트로피 이론을 바탕으로 '최악의 활성화 함수'를 정의하고, 이를 피하는 방향으로 함수를 설계하는 **EAFO 방법론**을 제안한다. 이 방법론을 ReLU에 적용하여 도출한 **CRReLU**($f(x) = \max(0, x) + \epsilon x e^{-x^2/2}$)는 ViT 기반 이미지 분류와 LLM 파인튜닝 작업에서 GELU를 포함한 기존 함수들을 능가하는 성능을 보였다. 이는 향후 데이터 분포에 최적화된 활성화 함수를 이론적으로 설계하고 동적으로 학습시키는 연구에 중요한 토대가 될 것이다.