# Activation Function Optimization Scheme for Image Classification

Abdur Rahman, Lu He, Haifeng Wang (2024)

## 🧩 Problem to Solve

딥러닝 모델의 성능, 수렴 속도 및 학습 역학은 활성화 함수(Activation Function)의 선택에 의해 결정적인 영향을 받는다. 하지만 Swish와 같이 강화학습 기반의 탐색 전략으로 개발된 일부 사례를 제외하고, 현재 널리 사용되는 대부분의 최신 활성화 함수들은 전문가의 직관에 의존하여 수동으로 설계(Hand-crafted)되었다는 한계가 있다.

본 논문은 이미지 분류 작업에 특화되어 기존의 최신 활성화 함수들보다 더 뛰어난 성능을 보이는 함수를 자동으로 발견하는 것을 목표로 한다. 이를 위해 수동 설계의 한계를 극복하고 진화 알고리즘(Evolutionary Approach)을 통한 최적화 프레임워크를 제안한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 유전 알고리즘(Genetic Algorithm, GA)을 사용하여 활성화 함수의 수학적 구조를 진화시키는 **Activation Function Optimization Scheme (AFOS)**를 구축하는 것이다. 주요 기여 사항은 다음과 같다.

1. **확장된 탐색 공간(Enhanced Search Space)**: 단순히 기존 함수들의 조합에 그치지 않고, 다양한 단항 함수(Unary functions)와 이항 함수(Binary functions)를 포함하여 약 1억 7천만 개 이상의 조합이 가능한 방대한 탐색 공간을 구성하였다. 특히 $x_1 \cdot \text{erf}(x_2)$와 같은 변형된 이항 함수를 추가하여 성능 향상을 도모하였다.
2. **효율적인 적합도 함수(Efficient Fitness Function)**: 검증 정확도($V_a$)와 검증 손실($V_l$)의 비율을 사용하는 적합도 함수 $f = V_a / V_l$를 제안하여, 우수한 염색체(Chromosome)가 다음 세대에 선택될 확률을 극대화하였다.
3. **EELU 시리즈의 발견**: AFOS를 통해 **Exponential Error Linear Unit (EELU)**라고 명명된 고성능 활성화 함수 시리즈를 발견하였으며, 특히 $-x \cdot \text{erf}(e^{-x})$ 형태의 함수가 가장 뛰어난 성능을 보임을 입증하였다.
4. **광범위한 일반화 검증**: 5가지 신경망 아키텍처와 8가지 다양한 이미지 데이터셋을 통해 제안된 함수의 범용성을 통계적으로 검증하였다.

## 📎 Related Works

기존의 활성화 함수 연구는 크게 세 가지 방향으로 진행되었다. 첫째, ReLU의 Dying ReLU 문제를 해결하기 위한 LReLU, ELU, SELU 등의 단조 증가 함수 개발이다. 둘째, Swish, GeLU, Mish와 같이 비단조성(Non-monotonicity)을 가진 함수들이 복잡한 데이터셋에서 더 좋은 성능을 보임이 밝혀졌다. 셋째, NeuroEvolution(NE)이나 강화학습(RL)을 이용해 구조를 탐색하는 시도가 있었다.

기존의 진화 알고리즘 기반 탐색 연구(Basirat and Roth, 2018 등)들은 탐색 공간을 기존에 존재하는 활성화 함수들로만 제한하는 경향이 있어 최적해를 찾지 못할 가능성이 컸다. 또한, 적합도 함수로 단순히 검증 정확도만을 사용한 경우, 탐색(Exploration)과 활용(Exploitation) 사이의 균형을 맞추는 데 한계가 있었다. 본 논문은 이러한 탐색 공간의 제약을 풀고, 손실 값을 함께 고려한 비율 기반 적합도 함수를 도입함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

AFOS는 유전 알고리즘(GA)을 기반으로 하며, 다음과 같은 절차로 진행된다:
$\text{초기 모집단 생성} \rightarrow \text{염색체 디코딩} \rightarrow \text{베이스 네트워크 학습 및 평가} \rightarrow \text{적합도 계산} \rightarrow \text{선택(Selection)} \rightarrow \text{교차(Crossover)} \rightarrow \text{변이(Mutation)} \rightarrow \text{다음 세대 생성}$

### 핵심 구성 요소

1. **염색체 표현 (Chromosome Representation)**:
   활성화 함수를 트리 기반 인코딩(Tree-based encoding)으로 표현한다. 트리의 노드는 단항 함수($u$)와 이항 함수($b$)로 구성된다.
   - **단항 함수**: $x, -x, e^x, |x|, e^{-x}, \min(x, 0), \max(x, 0), \sin(x), \cos(x), \text{erf}(x), \sigma(x), \ln(1+e^x)$ 등.
   - **이항 함수**: $x_1 + x_2, x_1 \cdot x_2, \max(x_1, x_2), \min(x_1, x_2), x_1 \cdot e^{x_2}, x_1 \cdot \sigma(x_2), x_1 \cdot \text{erf}(x_2)$ 등.

2. **적합도 함수 (Fitness Function)**:
   각 염색체를 디코딩하여 베이스 네트워크 $\phi$의 ReLU를 대체하고 CIFAR-10 데이터셋으로 15 epoch 동안 학습시킨다. 이후 다음 식을 통해 적합도 $f$를 계산한다.
   $$f = \frac{V_a}{V_l}$$
   여기서 $V_a$는 검증 정확도(Validation Accuracy), $V_l$은 검증 손실(Validation Loss)이다.

3. **GA 연산자**:
   - **Selection**: 적합도 점수가 높은 순으로 랭킹을 매겨 선택하는 Ranking Selection을 사용한다.
   - **Crossover**: 두 부모 염색체에서 무작위 교차 지점 $\tau$를 선택하여 부분 구조를 교환하는 Single-point crossover를 수행한다.
   - **Mutation**: 무작위로 유전자를 선택하여 동일 유형의 다른 함수(단항 $\rightarrow$ 단항, 이항 $\rightarrow$ 이항)로 교체하는 Single-point mutation을 적용한다.

4. **베이스 네트워크 ($\phi$)**:
   Conv $\rightarrow$ Max-pool $\rightarrow$ Dropout 층이 반복되는 전형적인 CNN 구조를 사용하며, 최종적으로 Dense 레이어를 통해 분류를 수행한다.

## 📊 Results

### EELU 시리즈 발견

AFOS를 통해 발견된 최상위 4개의 함수를 **EELU 시리즈**로 정의하였다.

| 이름 | 수학적 표현 | 특징 |
| :--- | :--- | :--- |
| **EELU-1** | $x \cdot \text{erf}(e^x)$ | 비단조성, 미분 가능, 하한 존재 |
| **EELU-2** | $-x \cdot \text{erf}(e^{-x})$ | **최고 성능**, 비단조성, 하한 존재 |
| **EELU-3** | $x \cdot (\text{erf}(e^x))^2$ | 계산 효율성 높음, 비단조성 |
| **EELU-4** | $\text{erf}(x) \cdot \ln(1+e^x)$ | 미분 가능, 부드러운 곡선 |

### 실험 결과 및 분석

- **네트워크 및 데이터셋 일반화**:
  - VGG16, AlexNet 및 베이스 네트워크 $\phi$에서는 EELU 시리즈가 기존 SOTA 함수(Swish, Mish, GeLU 등)보다 뛰어난 성능을 보였다.
  - 반면 ResNet50과 MobileNet에서는 Swish나 ELU가 더 나은 성능을 보였는데, 이는 Skip-connection이나 Depth-wise separable convolution과 같은 아키텍처 특성 때문으로 분석된다.
- **통계적 유의성**: Friedman Test 결과, $\chi^2$ 통계량이 임계값보다 훨씬 높게 나타나($p < 0.05$), 제안된 활성화 함수들이 기존 함수들과 통계적으로 유의미하게 다른(더 우수한) 성능을 가짐이 입증되었다. 특히 EELU-2가 가장 높은 평균 랭킹을 기록하였다.
- **대규모 데이터셋**: TinyImageNet과 CottonWeedID15 데이터셋에서도 ReLU 대비 EELU 시리즈의 성능 향상이 확인되었다.
- **계산 비용**: $\text{erf}$ 연산이 추가됨에 따라 약간의 오버헤드가 발생하지만, TensorFlow의 최적화된 구현을 사용할 경우 전체 학습 시간에 미치는 영향은 무시할 만한 수준이다. 특히 EELU-3는 ReLU와 거의 유사한 계산 효율성을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

EELU 시리즈의 고성능 원인은 다음과 같은 수학적 성질에서 기인한다.

1. **비단조성(Non-monotonicity)**: 입력값이 특정 구간에서 증가하다 감소하는 형태가 정보의 흐름을 개선한다.
2. **하한 존재(Lower-boundedness)**: 강한 정규화(Regularization) 효과를 제공한다.
3. **미분 가능성 및 부드러움(Smoothness)**: EELU 시리즈는 전 구간에서 연속적으로 미분 가능하다. Output Landscape 분석 결과, ReLU보다 훨씬 부드러운 손실 평면(Loss Landscape)을 형성하여 지역 최솟값(Local minima) 문제를 줄이고 수렴 속도를 높이는 것으로 보인다.

### 한계 및 비판적 해석

본 연구의 가장 큰 한계는 AFOS의 베이스 네트워크 $\phi$가 VGG-like 구조라는 점이다. 이로 인해 발견된 함수들이 VGG나 AlexNet에서는 매우 강력하지만, ResNet과 같은 현대적 구조에서는 성능 향상이 뚜렷하지 않았다. 이는 활성화 함수가 네트워크 아키텍처와 강하게 결합(Coupled)되어 있음을 시사한다. 따라서 "모든 네트워크에 적용 가능한 유일한 최적 함수"를 찾기보다는 "아키텍처별 최적 함수"를 찾는 방향으로 연구가 확장되어야 한다.

## 📌 TL;DR

본 논문은 유전 알고리즘을 활용해 이미지 분류에 최적화된 활성화 함수를 자동으로 탐색하는 AFOS 프레임워크를 제안하였다. 이를 통해 **EELU(Exponential Error Linear Unit)**라는 새로운 함수군을 발견하였으며, 특히 **EELU-2 ($-x \cdot \text{erf}(e^{-x})$)**가 VGG-like 아키텍처와 다양한 데이터셋에서 기존 SOTA 함수들을 압도하는 성능을 보임을 입증하였다. 이 연구는 활성화 함수의 자동 설계 가능성을 보여주었으며, 특히 비단조성과 부드러운 곡선이 딥러닝 모델의 일반화 성능에 핵심적인 역할을 한다는 통찰을 제공한다.
