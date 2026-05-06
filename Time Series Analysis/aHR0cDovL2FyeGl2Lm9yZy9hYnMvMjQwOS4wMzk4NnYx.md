# An Efficient and Generalizable Symbolic Regression Method for Time Series Analysis

Yi Xie, Tianyu Qiu, Yun Xiong, Xiuqi Huang, Xiaofeng Gao, and Chao Chen (2024)

## 🧩 Problem to Solve

전통적인 시계열 분석 및 예측 방법론(ARIMA, RNN, Transformer 등)은 정량적 분석과 미래 예측에서는 뛰어난 성능을 보이지만, 시계열 데이터의 기저에 흐르는 진화 패턴(evolution patterns)을 명시적으로 설명하는 데는 한계가 있다. 즉, 데이터가 '어떻게' 변하는지는 보여주지만, '무엇'이 이러한 변화를 일으키며 '왜' 특정 패턴이 발생하는지에 대한 정성적 통찰을 제공하지 못한다.

Symbolic Regression(SR)은 수학적 표현식을 통해 입력과 출력의 관계를 명시적으로 도출함으로써 이러한 문제를 해결할 수 있는 강력한 도구이다. 그러나 기존의 SR 기법들은 다음과 같은 심각한 문제점을 가지고 있다:

1. **계산 효율성 저하**: 조합 최적화(combinatorial optimization) 기반의 탐색으로 인해 탐색 공간이 기하급수적으로 증가하며, 계산 복잡도가 매우 높다.
2. **일반화 능력 부족**: 특정 샘플에 맞춤화된 휴리스틱 설계에 의존하여, 다양한 실제 데이터셋에 걸쳐 공통된 패턴을 학습하고 적용하는 능력이 떨어진다.

본 논문의 목표는 이러한 효율성과 일반화 문제를 해결하여, 대규모 실제 시계열 데이터에서도 효율적으로 해석 가능한 분석 표현식을 도출할 수 있는 **NEMoTS(Neural-Enhanced Monte-Carlo Tree Search)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Monte-Carlo Tree Search(MCTS)**의 탐색 구조에 **신경망(Neural Networks)**을 결합하여 탐색 공간을 획기적으로 줄이고 시뮬레이션 비용을 제거하는 것이다.

주요 기여 사항은 다음과 같다:

- **Neural-Enhanced MCTS**: MCTS의 선택(Selection)과 시뮬레이션(Simulation) 단계에 신경망을 통합하여, 유망한 노드를 우선적으로 탐색하고 시간 소모가 큰 무작위 시뮬레이션을 신경망의 보상 예측값으로 대체함으로써 효율성을 극대화하였다.
- **Symbolic Augmentation Strategy (SAS)**: 빈번 패턴 마이닝(Frequent Pattern Mining) 개념을 도입하여, 학습 과정에서 보상이 높게 나타난 복합 함수 패턴을 식별하고 이를 함수 라이브러리에 추가함으로써 복잡한 비선형 시스템의 표현 능력을 높였다.
- **해석 가능한 시계열 분석**: 단순한 수치 예측을 넘어, 실제 물리적 의미를 내포할 수 있는 명시적 수학식을 도출함으로써 시계열 데이터의 내재적 역학(intrinsic dynamics)을 분석할 수 있는 경로를 제시하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들의 한계를 지적한다:

- **전통적 SR (GP, BSR 등)**: 유전 알고리즘(Genetic Programming)이나 베이지안 접근법은 데이터 적합도는 높을 수 있으나, 조합 최적화의 특성상 계산 비용이 매우 크고 대규모 데이터셋에 적용하기 어렵다.
- **MCTS 기반 SR (SPL 등)**: MCTS를 사용하여 탐색 효율을 높이려는 시도가 있었으나, 여전히 많은 수의 무작위 시뮬레이션이 필요하며, 인스턴스 수준의 탐색에 그쳐 대규모 데이터로부터의 유도 학습(inductive learning) 능력이 부족하다.

NEMoTS는 MCTS의 구조적 장점(유효한 식 생성 보장)을 유지하면서, 신경망을 통해 '학습 가능한' 탐색 엔진을 구축함으로써 기존 방식들의 효율성 및 일반화 한계를 극복하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

NEMoTS는 크게 네 가지 구성 요소로 이루어진다:

- **Basic Function Library**: 덧셈, 뺄셈, 로그, 지수, 삼각함수 등 기본 연산자의 집합이다.
- **MCTS**: 표현식의 구조적 '뼈대(backbone)'를 생성하는 핵심 엔진이다.
- **Policy-Value Network**: MCTS의 선택 및 시뮬레이션 단계를 가이드하는 신경망이다.
- **Coefficient Optimizer**: 생성된 뼈대에 최적의 수치 계수를 할당하여 최종 표현식을 완성한다.

### 2. MCTS 파이프라인 및 주요 방정식

MCTS는 다음의 4단계 프로세스를 반복하여 표현식의 뼈대를 생성한다.

**① 선택 (Selection)**:
루트 노드에서 시작하여 **PUCT (Polynomial Upper Confidence Tree)** 스코어가 가장 높은 자식 노드를 선택한다.
$$Score(s, a) = Q(s, a) + c \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

- $Q(s, a)$: 액션 $a$의 평균 기대 보상
- $P(s, a)$: Policy-Value Network가 예측한 사전 확률 (Prior Probability)
- $N(s, a)$: 해당 노드의 방문 횟수
- $c$: 탐색(Exploration)과 이용(Exploitation)의 균형을 조절하는 상수

**② 확장 (Expansion)**:
선택된 리프 노드에 함수 라이브러리에서 무작위로 선택된 새로운 연산자 노드를 추가한다.

**③ 시뮬레이션 (Simulation)**:
기존 MCTS는 무작위 시뮬레이션을 통해 보상을 추정하지만, NEMoTS는 **Reward Estimator(신경망)**를 통해 현재 상태의 보상을 즉각적으로 예측하여 계산 시간을 단축한다.

**④ 역전파 (Back-propagation)**:
시뮬레이션 결과(보상)를 경로상의 모든 노드에 전달하여 $Q$값과 방문 횟수 $N$을 업데이트한다.

### 3. 보상 함수 (Reward Function)

도출된 표현식 $f(\cdot)$의 성능은 다음 식을 통해 평가된다:
$$R = \frac{\eta}{S} \cdot \frac{1}{1 + \sum_{i=0}^{N-1} \sqrt{(y_i - f(t_i))^2}}$$
여기서 $\eta$는 1보다 약간 작은 상수이며, $S$는 생성된 표현식의 크기(Complexity)이다. 이 함수는 적합도(Fitting accuracy)와 단순성(Simplicity) 사이의 균형을 맞춘다.

### 4. 신경망 설계 및 학습

- **구조**: 표현식 경로 시퀀스는 **LSTM**으로, 입력 시계열 신호는 **TCN (Temporal Convolutional Networks)**으로 인코딩한 후, 이를 결합하여 MLP를 통해 Policy와 Value를 출력한다.
- **손실 함수**:
  - **Policy Selector**: 예측 분포 $P$와 실제 보상 기반 분포 $Score$ 사이의 KL-Divergence를 최소화한다.
  - **Reward Estimator**: 예측 보상 $R'$과 실제 시뮬레이션 보상 $R$ 사이의 MSE (Mean Squared Error)를 최소화한다.
    $$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{policy} + \lambda_2 \mathcal{L}_{reward}$$

### 5. Symbolic Augmentation Strategy (SAS)

기본 함수만으로는 복잡한 시스템을 표현하기 어렵기 때문에, 학습 과정에서 높은 보상을 기록한 빈번한 표현식 경로(Composite functions)를 추출하여 라이브러리에 'Augmented Symbols'로 추가한다. 이를 통해 모델은 데이터셋에 특화된 고수준의 함수 패턴을 빠르게 사용할 수 있다.

### 6. 계수 최적화 (Coefficient Optimization)

MCTS가 생성한 뼈대에는 수치 계수가 없다. 이를 위해 기울기를 사용하지 않는 최적화 방법인 **Powell's method**를 사용하여 최적의 계수를 찾아 최종 식을 완성한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: WILI (독감 유사 질환 비율), ACER (호주 환율), AP (기압)
- **평가 지표**: 결정계수 ($R^2$), 상관계수 (CORR), 샘플당 평균 시간 비용 (ATC)
- **비교 대상**: GP, MRGP, BSR, PhySO, SPL (MCTS 기반 SR)

### 2. 주요 결과

- **적합 성능**: NEMoTS는 $R^2$와 CORR 지표에서 대부분의 베이스라인을 압도하였다. 특히 SPL과 유사하거나 더 높은 적합도를 보였으며, GP나 BSR 대비 월등한 성능 향상을 기록하였다.
- **계산 효율성**: ATC 측정 결과, NEMoTS는 다른 방법들보다 평균적으로 **약 68.06%의 시간 비용을 절감**하였다. 이는 신경망이 무작위 시뮬레이션을 대체했기 때문이다.
- **외삽(Extrapolation) 능력**: 도출된 식을 이용해 미래 6단계 값을 예측한 결과, ARIMA, SVR, RNN, TCN, NBeats 등 기존 예측 모델들보다 높은 $R^2$를 기록하였다. 이는 NEMoTS가 단순 오버피팅이 아니라 데이터의 내재적 역학을 정확히 포착했음을 시사한다.

## 🧠 Insights & Discussion

### 강점

NEMoTS는 MCTS의 정교한 탐색 능력과 신경망의 효율적인 패턴 인식 능력을 성공적으로 결합하였다. 특히, 시뮬레이션 단계를 신경망의 예측으로 대체한 점은 SR의 최대 난제인 계산 복잡도 문제를 실질적으로 해결한 지점으로 평가된다. 또한 SAS를 통해 데이터 특성에 맞는 복합 함수를 스스로 학습하여 라이브러리를 확장한 점이 돋보인다.

### 한계 및 비판적 해석

- **보상 예측의 정확도**: Ablation Study 결과, Reward Estimator를 제거하고 실제 시뮬레이션을 수행했을 때 적합 성능이 약간 향상되는 경향이 있었다. 이는 신경망의 보상 예측이 실제 값과 완전히 일치하지 않아 발생하는 오차이며, 효율성과 정확도 사이의 Trade-off가 존재함을 보여준다.
- **데이터 의존성**: 신경망 기반의 가이드 방식이므로, 학습 데이터의 양과 질이 탐색의 효율성에 직접적인 영향을 미친다. 본 논문에서는 10%의 데이터만으로 학습을 진행했음에도 좋은 결과가 나왔으나, 극도로 희소한 데이터셋에서의 동작 여부는 명시되지 않았다.

## 📌 TL;DR

본 논문은 시계열 데이터의 내재적 패턴을 명시적 수학식으로 도출하는 **NEMoTS**라는 Symbolic Regression 방법론을 제안한다. MCTS의 탐색 구조에 Policy-Value Network를 결합하여 탐색 공간을 효율적으로 좁히고, 시간 소모가 큰 시뮬레이션을 신경망 예측으로 대체하여 **계산 효율성을 68% 이상 향상**시켰다. 또한 빈번 패턴 마이닝 기반의 함수 확장 전략(SAS)을 통해 복잡한 비선형 시계열 데이터에 대한 적합도를 높였으며, 외삽 실험을 통해 도출된 식의 신뢰성과 해석 가능성을 입증하였다. 이 연구는 대규모 시계열 데이터에서 '설명 가능한' 모델을 빠르게 찾고자 하는 분야에 중요한 기여를 할 것으로 보인다.
