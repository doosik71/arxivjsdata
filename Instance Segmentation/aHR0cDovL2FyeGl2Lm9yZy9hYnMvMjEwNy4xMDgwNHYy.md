# Active Learning in Incomplete Label Multiple Instance Multiple Label Learning

Tam Nguyen and Raviv Raich (2021)

## 🧩 Problem to Solve

본 논문은 **Multiple Instance Multiple Label (MIML)** 학습 환경에서 레이블링 비용을 최소화하기 위한 **Active Learning (AL)** 프레임워크를 제안한다. MIML 설정에서 각 샘플(bag)은 여러 개의 인스턴스(instance)로 구성되며, 각 bag은 여러 개의 클래스 레이블을 가질 수 있다. 일반적으로 bag 수준의 레이블만 존재하고 내부 인스턴스의 레이블은 알 수 없는 상태이다.

기존의 MIML-AL 접근 방식은 크게 두 가지였다. 첫째는 선택된 bag의 모든 클래스 레이블을 한 번에 획득하는 방식이고, 둘째는 bag을 먼저 선택한 후 특정 클래스 레이블을 획득하며 추가적으로 인스턴스 수준의 레이블링을 수행하는 방식이다. 하지만 전자는 불필요한 클래스까지 레이블링하게 되어 비용이 높고, 후자는 인스턴스 식별 과정에서 추가 비용이 발생한다.

따라서 본 연구의 목표는 **단일 bag-class pair(bag과 특정 클래스의 쌍)**만을 쿼리하여 레이블링 비용을 획기적으로 줄이는 새로운 AL 전략을 개발하는 것이다. 이 과정에서 일부 클래스 레이블만 존재하게 되는 **Incomplete-Label (ILL)** 상황이 발생하며, 이를 효율적으로 학습하고 최적의 쌍을 선택하는 것이 핵심 문제이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 MIML-ILL 설정에 최적화된 **bag-class pair 기반의 Active Learning 프레임워크**를 설계한 것이다. 주요 설계 아이디어는 다음과 같다.

1. **Bag-Class Pair 선택 전략**: 전체 bag이 아닌, 모델의 불확실성이 가장 높거나 모델 업데이트에 가장 큰 영향을 줄 것으로 예상되는 특정 'bag-class pair'만을 선택하여 쿼리함으로써 레이블링 중복을 제거하고 비용을 낮춘다.
2. **EGL 및 Uncertainty Sampling의 확장**: 단일 인스턴스-단일 레이블 환경에서 사용되던 Expected Gradient Length (EGL)와 Uncertainty Sampling 기준을 MIML-ILL의 bag-class pair 선택 수준으로 확장하여 적용하였다.
3. **온라인 모델 업데이트**: 매 쿼리마다 전체 데이터를 재학습하는 대신, Stochastic Gradient Descent (SGD)를 이용한 온라인 업데이트 방식을 도입하여 계산 복잡도를 줄이고 확장성을 확보하였다.

## 📎 Related Works

기존의 Active Learning 연구는 주로 Single-Instance Single-Label (SISL) 설정에 집중되어 있었으며, 이후 Multi-Instance Learning (MIL)이나 Multi-Label Learning (MLL)으로 확장되었다.

- **MIL-AL**: bag의 레이블을 쿼리하거나, 양성(positive) bag 내의 특정 인스턴스 레이블을 쿼리하는 방식이 제안되었다.
- **MLL-AL**: 인스턴스 하나에 대해 모든 클래스 레이블을 쿼리하거나, 인스턴스-클래스 쌍을 쿼리하는 방식이 존재한다.
- **MIML-AL**: 기존 연구([34], [35])에서는 bag을 선택한 후 모든 클래스를 레이블링하거나, bag 선택 후 클래스를 선택하는 방식을 취했다.

본 논문은 기존 MIML-AL 방식들이 여전히 높은 레이블링 비용을 초래한다는 한계를 지적하며, **단일 bag-class pair만을 독립적으로 쿼리**하는 방식이 가장 효율적임을 주장하며 차별성을 둔다.

## 🛠️ Methodology

### 1. 문제 정의 및 표기법

전체 데이터셋은 $B$개의 bag으로 구성되며, 각 bag $X_b$는 인스턴스 집합 $\{x_{b1}, \dots, x_{bn_b}\}$이다. bag 레이블 $Y_b \in \{-1, 0, 1\}^C$에서 $-1$은 레이블이 아직 제공되지 않은(missing) 상태를 의미하며, $0$은 음성, $1$은 양성을 의미한다.

### 2. 기본 모델 (MIML-ILL)

본 연구는 인스턴스 레이블 $y_{bi}$가 다항 로지스틱 회귀(multinomial logistic regression)를 따른다고 가정한다.
$$P(y_{bi}=c|x_{bi}, w) = \frac{e^{w_c^T x_{bi}}}{\sum_{k=1}^C e^{w_k^T x_{bi}}}$$
bag 레이블 $Y_{bc}$는 **OR 규칙**을 따른다. 즉, bag 내의 인스턴스 중 적어도 하나가 클래스 $c$이면 $Y_{bc}=1$이고, 모든 인스턴스가 $c$가 아니면 $Y_{bc}=0$이다.

계산 복잡도를 줄이기 위해 **Marginal Maximum Likelihood (MML)** 접근법을 사용하여 각 클래스별로 로그-우도(log-likelihood)의 합을 최대화한다.
$$\mathcal{L}^{MML}(w) = \sum_{b=1}^B \sum_{t \in S_b} \log P(Y_{bt} | X_b, w)$$
여기서 $S_b$는 bag $b$에서 사용 가능한 레이블의 집합이다.

### 3. 인스턴스(Bag-Class Pair) 선택 기준

모델이 가장 정보 가치가 높은 $(b, c) \in U$ (unlabeled set)를 선택하기 위해 두 가지 기준을 제안한다.

**A. Expected Gradient Length (EGL)**
새로운 레이블이 추가되었을 때 모델 파라미터 $w$의 그래디언트 변화량이 가장 클 것으로 예상되는 쌍을 선택한다.
$$\text{EGL}_{bc} = \frac{1}{|L|+1} \left( P(Y_{bc}=1|X_b, w) \|\nabla \log P(Y_{bc}=1|X_b, w)\| + P(Y_{bc}=0|X_b, w) \|\nabla \log P(Y_{bc}=0|X_b, w)\| \right)$$

**B. Uncertainty Sampling**
클래스 존재 확률이 결정 경계($0.5$)에 가장 가까운, 즉 가장 불확실한 쌍을 선택한다.
$$S_{bc} = 2 P(Y_{bc}=1|X_b, w) (1 - P(Y_{bc}=1|X_b, w))$$

### 4. 모델 업데이트 및 학습 절차

매 쿼리 후 모델을 업데이트하기 위해 SGD 방식을 사용한다.
$$w^k = P(w^{k-1} - \eta^k g^k)$$
여기서 $g^k$는 새로 획득한 레이블에 대한 그래디언트이며, $P$는 파라미터 $w$가 특정 반지름 $\tau$ 내의 구(sphere) 영역에 머물도록 하는 투영(projection) 연산자이다. $\tau$는 모델 파라미터의 $l_2$-norm 상한선으로 정의된다.

업데이트 방식은 두 가지로 나뉜다.

- **Pair-SGD**: 쿼리된 단일 bag-class pair의 그래디언트만 사용하여 업데이트한다.
- **Bag-SGD**: 쿼리된 pair가 속한 bag의 모든 사용 가능한 클래스 레이블 정보를 사용하여 업데이트한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: HJA (조류 노래 오디오), Letter Carol, Letter Frost (시의 단어/철자 데이터셋).
- **비교 대상**: Offline GD, MIMLSVM-AL [34], MIML-AL [35], Random Selection.
- **평가 지표**: Bag Accuracy, Average Precision, Hamming Loss, One-error.

### 2. 주요 결과

- **업데이트 효율성**: 온라인 SGD 방식(Bag-SGD, Pair-SGD)이 오프라인 GD 방식과 비교했을 때 수렴 속도가 더 빠르며, 최종 성능 또한 매우 유사함을 확인하였다.
- **선택 전략의 유효성**: 제안된 **Bag-Class Pair 선택 전략(특히 EGL)**이 기존의 Bag-only 선택[34]이나 Bag-then-label 선택[35]보다 훨씬 적은 쿼리 횟수로 더 높은 정확도에 도달하였다.
- **SOTA 비교**: MIML-AL [35] 및 MIMLSVM-AL [34]과 비교했을 때, 대부분의 데이터셋에서 Hamming Loss를 효과적으로 낮추고 Bag Accuracy를 빠르게 높이는 성능 우위를 보였다. 특히 HJA 데이터셋의 경우, 초기에는 SVM 기반 방식이 우세했으나 쿼리 횟수가 증가함에 따라 제안 방법의 성능 향상 속도가 더 빨라 역전하는 양상을 보였다.

## 🧠 Insights & Discussion

본 논문은 MIML-AL에서 쿼리 단위를 'bag'에서 'bag-class pair'로 세분화함으로써 레이블링의 중복성을 제거하고 효율성을 극대화하였다. 특히, MML(Marginal Maximum Likelihood)을 통해 클래스 간의 의존성을 일부 희생하더라도 계산 복잡도를 선형적으로 낮춘 점이 실용적인 온라인 학습을 가능하게 했다.

**강점**:

- 실제 레이블링 비용과 직결되는 쿼리 단위를 최적화하여 실질적인 비용 절감 효과를 입증하였다.
- EGL과 Uncertainty Sampling이라는 검증된 AL 기준을 MIML-ILL 환경으로 성공적으로 확장하였다.

**한계 및 논의**:

- MML 접근법이 클래스 간의 상관관계를 무시하고 개별적으로 처리하므로, 클래스 간 강한 의존성이 존재하는 데이터셋에서는 성능 저하가 있을 가능성이 있다.
- HJA 데이터셋에서 초기에 SVM 기반 방식이 강했던 이유는 데이터셋 규모와 클래스 수의 특성상 초기 모델의 일반화 능력이 더 좋았기 때문으로 추측되며, 이는 기본 분류기(base classifier)의 선택이 AL 성능에 초기 영향을 줄 수 있음을 시사한다.

## 📌 TL;DR

이 논문은 MIML 학습에서 레이블링 비용을 줄이기 위해 **단일 bag-class pair를 쿼리하는 Active Learning 프레임워크**를 제안한다. EGL 및 Uncertainty Sampling 기준을 확장하여 최적의 쌍을 선택하고, 온라인 SGD를 통해 효율적으로 모델을 업데이트한다. 실험 결과, 기존의 bag 단위 쿼리 방식보다 훨씬 적은 비용으로 빠르게 높은 성능에 도달함을 확인하였으며, 이는 대규모 MIML 데이터셋의 효율적인 구축에 기여할 수 있다.
