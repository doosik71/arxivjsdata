# Active Nearest-Neighbor Learning in Metric Spaces

Aryeh Kontorovich, Sivan Sabato, and Ruth Urner (2018)

## 🧩 Problem to Solve

본 논문은 일반적인 Metric Space(거리 공간)에서 데이터의 레이블링 비용을 줄이기 위한 Pool-based Non-parametric Active Learning 알고리즘을 제안한다. 일반적인 머신러닝 설정에서 대량의 데이터를 레이블링하는 것은 비용과 시간이 많이 소요되지만, 레이블이 없는 데이터(unlabeled data)는 상대적으로 쉽게 얻을 수 있다.

연구의 핵심 목표는 레이블링된 샘플의 수, 즉 Label Complexity를 최소화하면서도, 기존의 Passive Learner(수동 학습기)에 필적하는 예측 정확도를 달성하는 Nearest-Neighbor(최근접 이웃) 분류기를 구축하는 것이다. 특히, 기존의 Active Learning 연구들이 주로 고정된 가설 공간을 갖는 Parametric 설정이나 특정 분포 가정이 필요한 Non-parametric 설정에 치중했던 것과 달리, 본 논문은 분포 가정 없이 일반적인 Metric Space에서 작동하며 이론적 보장을 제공하는 알고리즘을 개발하고자 한다.

## ✨ Key Contributions

본 논문의 가장 큰 기여는 **MARMANN(MArgin Regularized Metric Active Nearest Neighbor)**이라는 새로운 알고리즘을 제안하고 이에 대한 이론적 분석을 수행한 것이다. MARMANN의 핵심 직관은 다음과 같다.

1. **Generalized Sample Compression**: Margin 기반의 정규화를 통해 전체 데이터셋을 아주 작은 크기의 Compression Set(압축 집합)으로 줄여 표현하고, 이를 통해 일반화 성능을 보장한다.
2. **Label-Efficient Model Selection**: 최적의 Margin Scale $t$를 찾기 위해 모든 가능한 $t$를 테스트하는 대신, Binary Search와 유사한 효율적인 탐색 절차와 적응형 Bernoulli 추정법(`EstBer`)을 사용하여 레이블 요청 수를 획기적으로 줄인다.
3. **Distribution-Free Guarantees**: 데이터의 생성 분포에 대한 가정 없이, 입력 샘플의 Noisy-margin 속성에 기반하여 예측 오류의 상한선을 제공한다.
4. **Theoretical Lower Bounds**: 제안한 알고리즘의 Label Complexity가 동일한 오류 보장을 갖는 그 어떤 Passive Learner보다 현저히 낮음을 증명하였으며, Active Learning 자체의 하한선(Lower bound) 또한 제시하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **Parametric Active Learning**: 고정된 가설 클래스 내에서 학습하는 연구가 주를 이루었으며, 이는 Metric Space의 일반적인 특성을 반영하기 어렵다.
- **Non-parametric Active Learning**: 일부 연구가 분포 가정(예: Euclidean 공간의 특정 정규성) 하에서 minimax rate를 분석했으나, 분포 가정 없이 일반적인 Metric Space에서 작동하는 알고리즘에 대한 분석은 부족했다.
- **Passive Nearest-Neighbor**: Fix와 Hodges(1951) 이후 널리 사용되었으며, 특히 Margin-based regularization을 통해 Bayes-consistent한 결과가 도출될 수 있음이 알려져 있다.

### MARMANN의 차별점

MARMANN은 기존의 Passive Learner들이 사용하던 Sample Compression 기법을 Active 설정으로 확장하였다. 특히, 단순한 레이블 요청이 아니라 모델 선택(Model Selection) 단계에서 레이블 효율적인 절차를 도입하여, 데이터의 분포를 몰라도 일반적인 거리 공간에서 레이블 비용을 줄일 수 있음을 이론적으로 증명했다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인

MARMANN 알고리즘은 크게 두 단계로 구성된다.

1. **`SelectScale`**: 레이블링되지 않은 샘플 $U_{in}$에서 최적의 마진 척도 $\hat{t}$를 선택한다.
2. **`GenerateNNSet`**: 선택된 $\hat{t}$를 사용하여 1-NN 분류기의 기반이 될 Compression Set $\hat{S}$를 생성한다.

최종 출력은 $\hat{S}$를 이용한 1-Nearest-Neighbor 분류기 $h_{nn}^{\hat{S}}$이다.

### 2. 상세 구성 요소

#### (1) 모델 선택 절차 (`SelectScale`)

최적의 $\hat{t}$를 찾기 위해 $Dist_{mon}$(넷 크기가 단조 감소하는 거리 집합) 내에서 Binary Search를 수행한다. 각 단계에서 `EstimateErr` 함수를 통해 해당 scale에서의 경험적 오류 $\epsilon(t)$를 추정한다.

- **`EstimateErr` 및 `EstBer`**: Bernoulli 변수의 기댓값을 적응적으로 추정하는 `EstBer` 알고리즘을 사용한다. 이는 임계값 $\theta$를 기준으로, 실제 오류가 매우 작을 경우 적은 수의 샘플만으로 이를 빠르게 판단하여 레이블 낭비를 막는다.

#### (2) 압축 집합 생성 (`GenerateNNSet`)

선택된 $\hat{t}$에 대해 $t/2$-net을 구축하고, 각 Voronoi 영역 $P_i$에서 무작위로 $Q(m)$개의 점을 샘플링하여 다수결(Majority Vote)로 레이블 $\hat{y}_i$를 결정한다.

- 이때 $Q(m) = \lceil 18 \log(4m^3/\delta) \rceil$로 정의된다.
- 이 과정은 모든 데이터를 레이블링하는 대신, 각 영역의 대표 레이블만 효율적으로 추정하여 Compression Set을 구성하는 방식이다.

#### (3) 주요 방정식 및 이론적 보장

- **Noisy-Margin $(\nu, t)$-separation**: 데이터셋 $S$에서 $\nu$ 비율의 데이터를 제거했을 때, 서로 다른 레이블을 가진 모든 점 사이의 거리가 $t$ 이상인 상태를 말한다.
- **일반화 오차 상한 (Generalization Bound)**:
  $$GB(\epsilon, N, \delta, m, k) := \alpha\epsilon + \frac{2}{3} \frac{(N+1)\log(mk) + \log(1/\delta)}{m-N} + \frac{3\sqrt{2}\sqrt{\alpha\epsilon((N+1)\log(mk) + \log(1/\delta))}}{m-N}$$
  여기서 $\alpha = m/(m-N)$이며, $N$은 $\hat{t}$-net의 크기이다.
- **Label Complexity**: MARMANN이 요청하는 총 레이블 수는 다음과 같은 상한을 갖는다.
  $$O \left( \log^3 \left( \frac{m}{\delta} \right) \left( \frac{1}{\hat{G}} \log \left( \frac{1}{\hat{G}} \right) + m\hat{G} \right) \right)$$
  ($\hat{G}$는 선택된 scale에서의 일반화 오차 상한)

## 📊 Results

### 실험 설정 및 지표

본 논문은 실제 데이터셋 실험보다는 이론적 분석(Theoretical Analysis)에 집중한다. 분석의 핵심 지표는 **Prediction Error(예측 오류)**와 **Label Complexity(레이블 복잡도)**이다.

### 주요 이론적 결과

1. **오류 보장 (Theorem 4)**: MARMANN의 예측 오류 $\text{err}(\hat{h}, D)$는 최적의 Passive Learner가 달성하는 오류 $G_{min}(m, \delta)$의 상수 배(constant factor) 이내로 유지된다.
2. **레이블 효율성**: 특정 regime에서 Passive Learner가 $m$개의 레이블을 필요로 하는 반면, MARMANN은 $\tilde{\Theta}(\sqrt{m})$개의 레이블만으로도 유사한 성능을 낼 수 있다.
3. **Passive Lower Bound (Theorem 5)**: $\tilde{\Theta}(\sqrt{m})$개의 무작위 레이블만 사용하는 Passive Learner는 특정 분포에서 예측 오류가 $\tilde{\Omega}(m^{-1/4})$로 매우 느리게 감소함을 증명하여, MARMANN의 효율성을 입증하였다.
4. **Active Lower Bound (Theorem 6)**: MARMANN과 유사한 성능 보장을 갖는 그 어떤 Active Learner라도 $\tilde{\Omega}(m G_{min}(m, \delta))$의 레이블 복잡도를 가져야 함을 증명하여, 제안한 알고리즘의 하한선이 이론적으로 타이트함을 보였다.

## 🧠 Insights & Discussion

### 강점

- **분포 독립성**: 데이터의 생성 분포에 대한 가정이 전혀 없으므로 매우 일반적인 환경에서 적용 가능하다.
- **이론적 완결성**: 상한선(Upper bound)뿐만 아니라 Passive 및 Active Lower bound를 모두 제시하여 알고리즘의 효율성을 수학적으로 엄밀하게 증명하였다.
- **계산 효율성**: Vertex Cover와 같은 NP-hard 문제를 푸는 대신, 다수결 기반의 레이블링과 이진 탐색을 통해 계산 복잡도를 낮추었다.

### 한계 및 미해결 질문

- **상수 값의 크기**: 이론적 분석에서 도출된 상수(constant factor)가 약 2000 정도로 매우 크다. 저자들은 이것이 증명 과정에서의 산물(artifact)일 가능성이 높다고 언급하며, 향후 이를 줄이는 연구가 필요함을 시사한다.
- **Bayes-consistency**: Passive 버전의 알고리즘은 Bayes-consistent함이 증명되었으나, Active 버전인 MARMANN의 Bayes-consistency는 여전히 열린 문제로 남아 있다.
- **구현의 복잡성**: $t$-net을 효율적으로 구축하는 것은 실제 구현 단계에서 상당히 까다로운 작업이다.

### 비판적 해석

본 논문은 이론적으로 매우 견고한 분석을 제공하지만, 실무적인 관점에서는 매우 큰 상수 값과 $\tilde{\Theta}(\sqrt{m})$이라는 복잡도가 실제 데이터셋에서 얼마나 유의미한 레이블 감소를 가져오는지에 대한 실증적 데이터가 부족하다. 그럼에도 불구하고, Non-parametric Active Learning의 이론적 토대를 마련했다는 점에서 학술적 가치가 매우 높다.

## 📌 TL;DR

본 논문은 일반적인 Metric Space에서 분포 가정 없이 작동하는 Non-parametric Active Learning 알고리즘인 **MARMANN**을 제안한다. 이 알고리즘은 적응형 모델 선택 절차와 샘플 압축 기법을 통해, 기존 Passive Learner와 대등한 예측 정확도를 유지하면서도 레이블 요청 수를 획기적으로(예: $m \to \tilde{\Theta}(\sqrt{m})$) 줄일 수 있음을 이론적으로 증명하였다. 이는 향후 레이블링 비용이 매우 높은 고차원 데이터 분석 및 일반 거리 기반 분류 시스템의 효율성을 높이는 데 중요한 이론적 근거가 될 것으로 기대된다.
