# Efficient Active Learning of Halfspaces: an Aggressive Approach

Alon Gonen, Sivan Sabato, Shai Shalev-Shwartz (2013)

## 🧩 Problem to Solve

본 논문은 유클리드 공간에서의 **Half-spaces(반공간)** 학습을 위한 **Pool-based Active Learning** 문제를 다룬다. 

일반적인 Active Learning의 목표는 레이블이 없는 데이터 풀(pool)에서 가장 정보량이 많은 데이터를 선택적으로 쿼리하여, 최소한의 레이블 사용량(Label Complexity)으로 낮은 오차를 가진 예측 규칙을 학습하는 것이다. 특히 레이블링 비용이 매우 높은 실제 응용 분야에서 이 문제는 매우 중요하다.

논문의 핵심 목표는 기존의 **Aggressive Approach(공격적 접근 방식)**를 효율적이고 실용적으로 구현하고, 합리적인 가정(예: 마진 가정) 하에서 이론적 보장(Theoretical Guarantees)을 제공하는 것이다. 또한, 이를 통해 기존의 **Mellow Approach(완만한 접근 방식)**보다 공격적인 방식이 더 우수할 수 있음을 이론적, 실험적으로 증명하고자 한다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **Version Space(버전 공간)**의 부피를 최대한 균등하게 분할하는 데이터를 쿼리하는 Greedy 전략을 효율적으로 구현하는 것이다.

1. **ALuMA 알고리즘 제안**: Randomized Volume Approximation 기술을 사용하여 버전 공간의 부피를 추정함으로써, 계산적으로 불가능했던 Greedy 쿼리 선택을 효율적으로 수행하는 ALuMA 알고리즘을 설계하였다.
2. **이론적 보장 제공**: 데이터 풀이 마진(Margin) $\gamma$를 가지고 분리 가능하다는 가정하에, ALuMA의 레이블 복잡도가 최적의 복잡도($\text{OPT}_{\max}$)에 근사함을 증명하였다.
3. **다수결 투표(Majority Vote) 도입**: 버전 공간이 완전히 순수해질 때까지 쿼리를 계속하는 대신, 적절한 시점에서 멈추고 버전 공간 내의 가설들에 대해 근사적인 다수결 투표를 수행함으로써 레이블 복잡도를 더욱 낮추는 방법을 제시하였다.
4. **비분리 데이터 처리**: 데이터가 완전히 분리되지 않는 경우(Non-separable), 데이터를 고차원으로 매핑하고 Johnson-Lindenstrauss Random Projection을 적용하는 전처리 과정을 통해 ALuMA를 적용할 수 있는 프레임워크를 제안하였다.

## 📎 Related Works

기존의 Active Learning 접근 방식은 크게 두 가지로 나뉜다.

- **Mellow Approach (예: CAL 알고리즘)**: 학습자가 아직 추론하지 못한 거의 모든 레이블을 쿼리하는 방식이다. Realizable case(가설 클래스가 정답을 포함하는 경우)에서 레이블 복잡도를 지수적으로 개선할 수 있음이 알려져 있으며, Agnostic setting(임의의 에러가 존재하는 경우)에서도 작동한다는 장점이 있다. 그러나 Uniform distribution 하에서도 항상 최적은 아니라는 한계가 있다.
- **Aggressive Approach (예: Tong and Koller, Dasgupta)**: 매우 정보량이 많은 쿼리만을 요청하는 방식이다. 대표적으로 버전 공간을 가장 균등하게 나누는 데이터를 선택하는 Greedy 전략이 있다. 하지만 유클리드 공간에서 Convex Body의 부피를 계산하는 것은 $\#\text{P-hard}$ 문제로 알려져 있어, 기존 연구들은 증명되지 않은 휴리스틱에 의존해 왔다.

본 논문은 이러한 계산적 난제를 Randomized 알고리즘으로 해결하여, 공격적 접근 방식의 이론적 보장과 실용성을 동시에 확보함으로써 기존의 Mellow 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 Greedy 전략
학습자는 현재까지 쿼리한 레이블과 일치하는 모든 가설의 집합인 **Version Space** $V^t$를 유지한다. 새로운 쿼리 $x$를 선택할 때, $x$의 레이블이 $+1$일 때의 버전 공간 $V_{+1}$와 $-1$일 때의 버전 공간 $V_{-1}$의 확률 질량(부피)을 최대한 균등하게 나누는 데이터를 선택한다.

목표 함수는 다음과 같다:
$$\text{argmax}_{x \in X} P(V_{+1}^{t,x}) \cdot P(V_{-1}^{t,x})$$

### 2. ALuMA 알고리즘의 핵심 구성 요소
부피 계산의 계산 복잡도를 해결하기 위해 다음과 같은 기법을 사용한다.

- **Volume Estimation**: Kannan et al. (1997)의 무작위 알고리즘을 사용하여 convex body의 부피를 근사한다. 이를 통해 $\alpha$-approximately greedy한 선택을 수행한다.
- **Approximate Majority Vote**: 모든 데이터를 정확히 레이블링할 때까지 쿼리하는 대신, 버전 공간의 순도(Purity)가 일정 수준에 도달하면 멈춘다. 이후 버전 공간에서 무작위로 추출한 가설들의 다수결 투표를 통해 나머지 풀의 레이블을 결정한다.
- **Hit-and-Run Sampling**: 버전 공간 내에서 가설 $w$를 균등하게 샘플링하기 위해 사용된다.

### 3. 이론적 레이블 복잡도
타겟 가설 $h$가 마진 $\gamma$를 가지고 데이터를 분리할 때, ALuMA의 레이블 복잡도는 다음과 같은 상한을 가진다:
$$\text{Label Complexity} = O(d \log(1/\gamma) \cdot \text{OPT}_{\max})$$
여기서 $d$는 공간의 차원이며, $\text{OPT}_{\max}$는 해당 풀에 대한 최적의 레이블 복잡도이다.

### 4. 비분리 데이터 처리 (Preprocessing)
데이터가 분리 가능하지 않을 때, 다음과 같은 변환을 거쳐 ALuMA를 적용한다:
1. **고차원 매핑**: $x_i \in \mathbb{R}^d$를 $x'_i = (ax_i; \sqrt{1-a^2}e_i) \in \mathbb{R}^{d+m}$으로 매핑하여 강제로 분리 가능하게 만든다. 이때 $a$는 힌지 손실(Hinge loss)의 상한 $H$에 따라 결정된다.
2. **Random Projection**: Johnson-Lindenstrauss 정리를 이용하여 차원을 $k$로 축소하면서도 마진 특성을 유지한다.
3. **ALuMA 적용**: 변환된 데이터 $\bar{X}$에 대해 ALuMA를 수행하여 레이블을 얻은 후, 원래 공간에서 SVM 등을 통해 최종 분류기를 학습한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: MNIST (숫자 3 vs 5, 4 vs 7), PCMAC (웹 포스트 분류), 합성 데이터 (Uniform distribution, Octahedron 구조).
- **비교 대상**: CAL (Mellow), QBC (Middle-ground), TK (Tong & Koller 휴리스틱), ERM (Passive learning).
- **지표**: 레이블 예산(Label Budget)에 따른 Train/Test Error.

### 2. 주요 결과
- **실용적 성능**: MNIST와 PCMAC 데이터셋에서 ALuMA와 TK(공격적 방식)가 CAL(완만한 방식)보다 훨씬 빠르게 에러를 감소시켰다. 특히 CAL은 초기 레이블 예산 구간에서 Passive ERM과 거의 차이가 없는 모습을 보였다.
- **고차원에서의 효율성**: Uniform distribution 실험에서 차원 $d$가 10에서 100으로 증가할 때, 공격적 방식의 우위가 더 뚜렷하게 나타났다.
- **특수 구조(Octahedron)**: 특정 합성 데이터셋에서 ALuMA는 매우 적은 쿼리로 정답을 찾은 반면, CAL과 QBC는 지수적으로 많은 쿼리가 필요함을 확인하였다.
- **비분리 데이터**: MNIST 데이터를 PCA로 축소하여 비분리 상태로 만들었을 때, ALuMA가 IWAL(Agnostic active learning의 SOTA)보다 더 빠른 에러 감소율을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰
- **공격적 전략의 재발견**: 그동안 계산 복잡성 때문에 외면받았던 Greedy 부피 분할 전략이 Randomized 알고리즘을 통해 실용적으로 구현 가능함을 보였으며, 많은 경우 Mellow 방식보다 훨씬 효율적임을 증명하였다.
- **마진의 역할**: 본 연구에서 마진 $\gamma$는 단순히 일반화 성능을 위한 것이 아니라, 계산 가능한 범위 내에서 최적의 쿼리를 찾기 위한 이론적 도구로 사용되었다.

### 2. 한계 및 가정
- **마진 의존성**: ALuMA의 성능 보장은 마진 $\gamma$가 존재한다는 가정에 기반한다. 만약 마진이 극도로 작다면 레이블 복잡도가 증가한다.
- **계산 비용**: 무작위 부피 추정과 Hit-and-Run 샘플링을 사용함에도 불구하고, 차원이 매우 높고 쿼리 횟수가 많아지면 여전히 계산 비용이 발생한다.

### 3. 비판적 해석
논문은 공격적 방식이 일반적으로 우수함을 주장하지만, 동시에 Mellow 방식(CAL)이 에러 레벨이 매우 높은 Agnostic setting에서는 더 안정적일 수 있음을 인정한다. 따라서 모든 상황에서 ALuMA가 정답이 아니라, 데이터의 노이즈 수준(Error level)에 따라 알고리즘을 선택하는 전략이 필요하다는 점이 향후 연구의 핵심이 될 것으로 보인다.

## 📌 TL;DR

본 논문은 Half-spaces 학습을 위해 버전 공간의 부피를 최적으로 분할하는 **공격적(Aggressive) Active Learning 알고리즘인 ALuMA**를 제안한다. 부피 계산의 난제를 Randomized Approximation으로 해결하여 실용성을 확보했으며, 마진 가정이 있을 때 최적의 레이블 복잡도에 근사함을 이론적으로 증명하였다. 실험 결과, 실무적인 데이터셋과 고차원 설정에서 기존의 CAL과 같은 완만한(Mellow) 방식보다 훨씬 적은 레이블로 높은 성능을 달성함을 확인하였다. 이는 향후 고효율 레이블링 시스템 구축에 중요한 기여를 할 수 있다.