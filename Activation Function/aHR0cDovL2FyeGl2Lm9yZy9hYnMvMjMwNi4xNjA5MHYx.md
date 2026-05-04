# Empirical Loss Landscape Analysis of Neural Network Activation Functions

Anna Sergeevna Bosman, Andries Engelbrecht, Mardé Helbig (2023)

## 🧩 Problem to Solve

인공신경망(Neural Networks, NNs)의 핵심은 은닉층의 비선형 활성화 함수(Activation Function)를 통해 복잡한 비선형 함수를 모델링하는 능력에 있다. 활성화 함수의 선택은 결과적으로 손실 함수가 형성하는 Loss Landscape의 성질에 지대한 영향을 미치며, 이는 신경망의 최적화 효율성과 일반화 성능으로 이어진다.

그동안 활성화 함수와 Loss Landscape의 관계에 대한 일부 이론적 연구들이 진행되었으나, 대다수의 연구가 엄격한 제한적 가정에 의존하거나 실증적인 증거를 충분히 제시하지 못했다. 특히 최신 활성화 함수 중 하나인 Exponential Linear Unit (ELU)에 대한 Loss Landscape 분석은 거의 이루어지지 않았다. 따라서 본 논문의 목표는 Hyperbolic Tangent (TanH), Rectified Linear Unit (ReLU), 그리고 ELU 활성화 함수가 신경망의 Loss Landscape 특성(볼록성, 평탄도, modality 등)에 어떠한 영향을 미치는지 실증적으로 분석하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Loss-Gradient Cloud (LGC) 시각화 기법과 Progressive Gradient Walk (PGW) 샘플링 방법을 통해 활성화 함수별 Loss Landscape의 기하학적 특성을 정량적으로 분석한 것이다. 주요 발견 사항은 다음과 같다.

1. **계곡의 존재 확인**: 모든 활성화 함수에서 Wide valley와 Narrow valley가 공통적으로 존재함을 확인하였다. 특히 Narrow valley는 뉴런의 포화(Saturation) 및 암묵적 정규화(Implicit Regularisation)된 네트워크 설정과 상관관계가 있음을 밝혀냈다.
2. **Modality의 불변성**: 활성화 함수의 선택이 Loss Landscape의 Modality, 즉 유일한 지역 최솟값(Local Minima)의 총 개수에는 영향을 주지 않는다는 점을 발견하였다.
3. **활성화 함수별 특성**:
    - **ReLU**는 가장 높은 수준의 볼록성(Convexity)을 보인다.
    - **ELU**는 가장 낮은 수준의 평탄도(Flatness)를 보이며, 오버피팅에 가장 강건한 Loss Landscape를 생성하여 우수한 일반화 성능을 나타낸다.

## 📎 Related Works

기존 연구들에 따르면 활성화 함수의 특성이 Loss Landscape의 복잡도를 결정한다. Kordos와 Duch는 비단조(Non-monotone) 함수가 단조(Monotone) 함수보다 더 복잡한 Landscape를 생성하며, 비매끄러운(Non-smooth) 함수는 더 많은 평탄 지역(Plateaus)을 만든다고 주장하였다.

ReLU에 대해서는 일부 이론적 연구가 존재한다. Liang 등은 ReLU가 0이 아닌 오차를 가진 지역 최적점(Local Optima)을 생성할 수 있음을 보였고, Cao 등은 2층 ReLU 네트워크의 임계점 주변에 평탄한 지역이 존재한다고 분석하였다. 또한, ReLU의 최적점은 일반적으로 미분 불가능하며, Sharp minima는 suboptimal한 경향이 있다는 연구 결과가 있다. 하지만 이러한 연구들은 주로 이론적 분석에 그쳤으며, 실제 데이터셋을 통한 실증적 분석이나 ELU와 같은 최신 함수에 대한 비교 분석은 부족한 실정이다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 연구는 단일 은닉층을 가진 Feed-forward NN을 사용하며, 출력층에는 Sigmoid 활성화 함수를, 은닉층에는 TanH, ReLU, ELU를 각각 적용하여 비교 분석한다. 손실 함수로는 Cross-entropy loss를 사용하였다.

### 샘플링 및 분석 방법

연속적인 가중치 공간을 효율적으로 탐색하기 위해 **Progressive Gradient Walk (PGW)** 알고리즘을 사용한다. PGW는 단순 무작위 샘플링과 달리 경사면(Gradient) 정보를 활용하여 오차가 낮은 영역을 집중적으로 탐색한다.

1. **PGW 절차**:
   - 현재 지점 $x_k$에서 손실 함수의 기울기 벡터 $\nabla \mathcal{L}$을 계산한다.
   - 기울기의 부호를 기반으로 이진 마스크 $b$를 생성한다:
     $$b_{kj} = \begin{cases} 0 & \text{if } \nabla \mathcal{L}_j > 0 \\ 1 & \text{otherwise} \end{cases}$$
   - 다음 지점 $x_{k+1}$은 현재 지점에 무작위 크기의 스텝 $\Delta x_k$를 더하여 결정하며, 이때 방향은 마스크 $b$에 의해 결정된다:
     $$x_{k+1} := x_k + \Delta x_k$$
     $$\Delta x_{kj} := \begin{cases} -\Delta x_{kj} & \text{if } b_{kj} = 0 \\ \Delta x_{kj} & \text{otherwise} \end{cases}$$
     여기서 $\Delta x_{kj} \in [0, \epsilon]$ 범위의 무작위 값이다.

2. **Loss-Gradient Clouds (LGC)**:
   샘플링된 가중치들의 손실 값(Loss)을 x축으로, 기울기의 크기(Gradient magnitude, $\lVert \nabla \mathcal{L} \rVert$)를 y축으로 하는 산점도를 그려 최적점의 특성을 시각화한다.
   - $\text{Loss} = 0, \lVert \nabla \mathcal{L} \rVert = 0 \implies$ Global Minima
   - $\text{Loss} \neq 0, \lVert \lVert \nabla \mathcal{L} \rVert = 0 \implies$ Local Minima 또는 Saddle points

3. **곡률 분석 (Curvature Analysis)**:
   Hessian 행렬의 고윳값(Eigenvalues)을 분석하여 해당 지점이 Convex local minima인지, Saddle point인지, 혹은 평탄한(Singular/Flat) 지역인지를 판별한다.

### 포화도 측정 (Saturation Measure)

- **TanH**: bounded 활성화 함수를 위한 $\sigma_h$ 측정법을 사용하여 [0, 1] 범위로 측정한다 (1에 가까울수록 포화).
- **ReLU**: 모든 은닉 뉴런 중 출력이 0인 뉴런의 비율을 포화도의 척도로 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: XOR, Iris, Heart, MNIST의 4가지 문제를 통해 차원과 복잡도별 영향을 분석하였다.
- **파라미터**: 초기화 범위 ( $[-1, 1]$, $[-10, 10]$ ), 스텝 크기 (Micro: 초기화 범위의 1%, Macro: 10%)를 조합하여 실험하였다.

### 주요 결과

1. **XOR 문제**:
   - 세 함수 모두 4개의 정지점(Stationary attractors)을 가졌으며, 그중 3개가 지역 최솟값이었다.
   - ReLU는 지역 최솟값 주변에 Saddle point와 Flat point가 많아, 경사 하강법 기반 알고리즘이 지역 최솟값에 갇히지 않고 탈출할 가능성이 높음을 확인하였다.
   - ELU는 TanH보다 표면이 더 매끄럽고 연결된 계곡(Valleys)이 더 넓어 탐색 효율성이 가장 좋았다.

2. **Iris 및 Heart 문제**:
   - 모든 활성화 함수에서 두 개의 클러스터(High error/Low gradient vs Low error/High gradient)로 나뉘는 현상이 관찰되었다.
   - **Flatness와 포화도의 관계**: Hessian 분석 결과, 평탄한 지역(Indefinite curvature)은 뉴런의 포화(Saturation)와 강한 상관관계가 있음이 입증되었다. 즉, 뉴런이 포화되면 가중치 변화가 결과에 영향을 주지 않아 평탄한 지형이 형성된다.
   - **일반화 성능**: ELU가 가장 우수한 일반화 성능을 보였으며, 특히 Narrow valley(High gradient)에 위치한 솔루션 중 일부가 암묵적 정규화 효과로 인해 일반화 성능이 좋게 나타났다.

3. **MNIST 문제 (고차원)**:
   - 고차원에서도 모든 함수가 단일 global attractor로 수렴하는 경향을 보였다.
   - ELU가 가장 일관되게 우수한 일반화 성능을 보였다.
   - ReLU와 ELU는 Unbounded 특성 덕분에 TanH보다 훨씬 강한 기울기를 생성하여, 고차원 공간에서도 더 탐색 가능한(Searchable) Landscape를 제공하였다.

## 🧠 Insights & Discussion

본 연구는 활성화 함수가 Loss Landscape의 구체적인 기하학적 구조에 어떻게 기여하는지를 실증적으로 규명하였다.

첫째, **포화(Saturation) $\rightarrow$ 평탄도(Flatness) $\rightarrow$ 좁은 계곡(Narrow Valley)**으로 이어지는 연결 고리를 확인하였다. 뉴런이 포화되면 특정 가중치의 영향력이 사라져 landscape가 평탄해지며, 이는 시각적으로 좁고 가파른 계곡의 형태로 나타난다.

둘째, 평탄한 지역이 반드시 나쁜 것은 아니라는 점이다. 일부 평탄한 영역은 불필요한 가중치가 0에 가깝게 조정된 '암묵적 정규화' 상태를 의미하며, 이는 모델의 복잡도를 줄여 오히려 일반화 성능을 높이는 결과(Embedded minima)를 가져올 수 있다.

셋째, **ELU의 우수성**이다. ELU는 ReLU의 볼록성과 TanH의 매끄러움을 적절히 절충하며, 특히 평탄한 지역을 최소화함으로써 경사 기반 최적화 알고리즘이 가장 효율적으로 작동할 수 있는 지형을 형성한다. 또한 오버피팅에 가장 강건한 모습을 보여 실무적인 적용 가치가 높음을 시사한다.

## 📌 TL;DR

본 논문은 TanH, ReLU, ELU 활성화 함수가 신경망의 손실 지형(Loss Landscape)에 미치는 영향을 LGC와 PGW 기법으로 분석하였다. 분석 결과, 활성화 함수가 최솟값의 개수(Modality)를 바꾸지는 않지만, **ReLU는 가장 높은 볼록성**을, **ELU는 가장 낮은 평탄도와 최상의 일반화 성능**을 제공함을 밝혀냈다. 특히 지형의 평탄함과 좁은 계곡의 존재가 뉴런의 포화 및 암묵적 정규화와 밀접하게 연관되어 있음을 실증적으로 입증하였다. 이 연구는 향후 일반화 성능이 뛰어난 특정 지형 영역으로 수렴하는 최적화 알고리즘 설계에 중요한 기초 자료가 될 것으로 보인다.
