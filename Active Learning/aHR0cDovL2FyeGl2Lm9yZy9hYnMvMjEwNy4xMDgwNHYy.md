# Active Learning in Incomplete Label Multiple Instance Multiple Label Learning

Tam Nguyen and Raviv Raich (2021)

## 🧩 Problem to Solve

본 논문은 Multiple Instance Multiple Label (MIML) 학습 환경에서 발생하는 과도한 레이블링 비용 문제를 해결하고자 한다. MIML 설정에서 각 샘플은 여러 인스턴스로 구성된 '백(bag)' 형태이며, 각 백은 여러 개의 클래스 레이블을 가질 수 있다. 일반적인 단일 인스턴스 단일 레이블(SISL) 학습에 비해 MIML은 복잡한 객체를 표현하기에 유리하지만, 데이터셋의 규모가 커질수록 모든 백의 모든 클래스에 대해 레이블을 지정하는 비용은 매우 높다.

특히, 기존의 MIML Active Learning (AL) 접근 방식들은 (1) 선택된 백의 모든 클래스 레이블을 한 번에 요청하거나, (2) 백을 먼저 선택한 후 특정 클래스를 선택하는 계층적 방식을 사용했다. 전자는 불필요한 클래스 레이블까지 요청하게 되어 낭비가 심하고, 후자는 추가적인 인스턴스 레벨의 레이블링 비용이 발생할 수 있다. 따라서 본 연구의 목표는 **백-클래스 쌍(bag-class pair)** 단위로 쿼리를 수행하여 레이블링 비용을 최소화하고, 일부 레이블만 존재하는 **Incomplete-label MIML** 설정에서도 효과적으로 동작하는 Active Learning 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 쿼리의 단위를 '백' 전체가 아닌 **'백-클래스 쌍'**으로 세분화하는 것이다. 이를 통해 모델 성능 향상에 가장 기여도가 높은 특정 클래스의 레이블만을 선택적으로 요청함으로써 중복 레이블링을 방지하고 비용을 절감한다. 

주요 기여 사항은 다음과 같다:
1. **백-클래스 쌍 기반의 쿼리 전략**: MIML-ILL(Incomplete Label) 설정에서 가장 정보 가치가 높은 백-클래스 쌍을 선택하는 새로운 AL 프레임워크를 제안하였다.
2. **AL 기준의 확장**: 기존의 Expected Gradient Length (EGL)와 Uncertainty Sampling 기준을 백-클래스 쌍 선택 전략에 맞게 수정하여 적용하였다.
3. **효율적인 모델 업데이트**: 계산 복잡도를 줄여 확장성을 확보하기 위해, Marginal Log-Likelihood를 최대화하는 온라인 SGD(Stochastic Gradient Descent) 기반의 업데이트 방식을 도입하였다.

## 📎 Related Works

전통적인 Active Learning은 주로 SISL 설정에서 연구되었으며, 이후 Multi-Instance Learning (MIL)과 Multi-Label Learning (MLL)으로 확장되었다. 

- **MIL AL**: 백 전체의 레이블을 요청하거나, 양성(positive) 백 내의 특정 인스턴스 레이블을 요청하는 방식이 주를 이룬다.
- **MLL AL**: 인스턴스 하나에 대해 모든 클래스 레이블을 요청하거나, 인스턴스-클래스 쌍을 선택하여 요청하는 방식이 사용된다.
- **MIML AL**: 기존 연구들은 백을 선택한 뒤 모든 클래스를 레이블링하거나, 백 선택 후 클래스를 선택하는 방식을 취했다. 하지만 이러한 방식들은 클래스 수가 많아질수록 레이블링 비용이 선형적으로 증가하는 한계가 있다.

본 논문은 이러한 기존 방식들과 달리, 처음부터 **백-클래스 쌍**을 직접 선택하여 쿼리함으로써 불필요한 레이블 획득을 억제하고, 특히 레이블이 불완전하게 주어진(incomplete-label) 상황에서도 학습이 가능하도록 설계되었다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 문제 정의 및 데이터 표현
데이터셋은 $B$개의 백 $\{(X_b, Y_b)\}_{b=1}^B$로 구성된다. 여기서 $X_b$는 인스턴스들의 집합이며, $Y_b \in \{-1, 0, 1\}^C$는 백-레벨 레이블 벡터이다.
- $1$: Positive label
- $0$: Negative label
- $-1$: Label absence (알 수 없음/미제공)

학습 목표는 사용 가능한 레이블 집합 $L = \{(b, c) | Y_{bc} \neq -1\}$를 효율적으로 확장하여 최적의 분류기를 학습시키는 것이다.

### 2. 기본 모델 및 확률적 추론
본 연구는 Discriminative Graphical Model을 기반으로 하며, 인스턴스 레벨의 레이블 $y_{bi}$는 Multinomial Logistic Regression을 따른다고 가정한다:
$$P(y_{bi}=c|x_{bi}, w) = \frac{e^{w_c^T x_{bi}}}{\sum_{k=1}^C e^{w_k^T x_{bi}}}$$

백 레이블 $Y_{bc}$는 **OR rule**을 따른다. 즉, 백 내의 인스턴스 중 적어도 하나가 클래스 $c$이면 $Y_{bc}=1$이고, 모든 인스턴스가 $c$가 아니면 $Y_{bc}=0$이다.
$$P(Y_{bc}=Y|y_b) = Y(1 - \prod_{i=1}^{n_b} I(y_{bi} \neq c)) + (1-Y) \prod_{i=1}^{n_b} I(y_{bi} \neq c)$$

계산 복잡도를 줄이기 위해 EM 알고리즘 대신 **Marginal Maximum Likelihood (MML)** 접근 방식을 사용하여, 각 클래스별 로그 가능도의 합을 최대화하는 목적 함수 $L_{MML}(w)$를 정의하고 이를 최적화한다.

### 3. 백-클래스 쌍 선택 전략 (Instance Selection)
미제공 레이블 집합 $U = \{(b, c) | Y_{bc} = -1\}$에서 다음 두 가지 기준을 통해 쿼리 대상 $(b^*, c^*)$를 선택한다.

#### (1) Expected Gradient Length (EGL)
특정 백-클래스 쌍의 레이블을 알게 되었을 때 모델의 그라디언트 변화량이 가장 클 것으로 예상되는 쌍을 선택한다.
$$EGL_{bc} = \frac{1}{|L|+1} \left( P(Y_{bc}=1|X_b, w) \|\nabla \log P(Y_{bc}=1|X_b, w)\| + P(Y_{bc}=0|X_b, w) \|\nabla \log P(Y_{bc}=0|X_b, w)\| \right)$$

#### (2) Uncertainty Sampling
백-클래스 쌍의 예측 확률이 결정 경계(0.5)에 가장 가까운, 즉 불확실성이 가장 높은 쌍을 선택한다.
$$S_{bc} = 2 P(Y_{bc}=1|X_b, w) (1 - P(Y_{bc}=1|X_b, w))$$

### 4. 모델 업데이트 (Model Update)
매 쿼리 이후 모델을 업데이트하기 위해 온라인 SGD 방식을 사용한다.
- **Pair-SGD**: 새로 획득한 단일 백-클래스 쌍의 그라디언트만을 사용하여 파라미터를 업데이트한다.
- **Bag-SGD**: 쿼리된 쌍이 속한 백 전체의 가용한 모든 클래스 레이블을 사용하여 업데이트함으로써 안정성을 높인다.

파라미터 $w$는 발산 방지를 위해 반지름 $\tau$를 가진 구(sphere) 영역 내로 투영(projection)된다:
$$P(w) = \begin{cases} w, & \|w\| \le \tau \\ \tau \frac{w}{\|w\|}, & \|w\| > \tau \end{cases}$$

## 📊 Results

### 실험 설정
- **데이터셋**: HJA (조류 노래 오디오), Letter Carol, Letter Frost (문학 텍스트 기반 MIML 데이터).
- **비교 대상**: Random selection, Bag-only selection [34], Bag-then-label selection [35], MIMLSVM-AL [34], MIML-AL [35].
- **평가 지표**: Bag Accuracy, Average Precision, Hamming Loss, One-error.

### 주요 결과
1. **추론 방식 비교**: 온라인 SGD 방식(Bag-SGD, Pair-SGD)이 오프라인 GD 방식과 비교했을 때 수렴 속도가 빠르며, 최종 성능 면에서도 유사한 수준에 도달함을 확인하였다.
2. **선택 전략 비교**: 제안된 **백-클래스 쌍 기반의 EGL 및 Uncertainty sampling**이 기존의 백 단위 선택이나 계층적 선택 방식보다 적은 쿼리 횟수로도 더 빠르게 높은 성능(Bag Accuracy 등)에 도달하였다.
3. **SOTA 모델 비교**: 
   - MIML-AL [35] 대비 초기 성능과 쿼리 과정 전반에서 우월한 성능을 보였다.
   - MIMLSVM-AL [34]과의 비교에서는 HJA 데이터셋의 경우 초기에는 SVM 기반 방식이 유리했으나, 쿼리 횟수가 증가함에 따라 제안 기법의 성능 향상 속도가 더 가팔라 결국 추월하는 양상을 보였다.

## 🧠 Insights & Discussion

본 논문은 MIML 학습에서 레이블링 비용을 결정짓는 핵심 요소가 '쿼리의 단위'임을 입증하였다. 기존의 백 단위 쿼리는 클래스 수가 많을 때 심각한 중복 비용을 발생시키지만, 이를 백-클래스 쌍 단위로 쪼개어 접근함으로써 정보 획득 효율을 극대화할 수 있었다.

**강점**: 
- 현실적인 '불완전한 레이블' 상황을 가정하여 실용성을 높였다.
- MML과 온라인 SGD를 결합하여 MIML의 고질적인 계산 복잡도 문제를 해결하고 확장성을 확보하였다.

**한계 및 논의**:
- 성능이 기반 모델인 MIML-ILL 분류기의 성능에 의존적이다.
- HJA 데이터셋에서 SVM 기반 방식이 초기에 강세를 보인 것은, 해당 데이터셋의 클래스 수가 상대적으로 적고 데이터 규모가 충분하여 SVM의 일반화 성능이 초기에 잘 발휘되었기 때문으로 추측된다.

## 📌 TL;DR

이 논문은 MIML 학습의 레이블링 비용을 줄이기 위해 **백-클래스 쌍(bag-class pair)** 단위로 최적의 샘플을 선택하는 Active Learning 프레임워크를 제안한다. EGL과 Uncertainty Sampling을 백-클래스 단위로 확장 적용하고, 온라인 SGD를 통해 효율적인 모델 업데이트를 구현함으로써, 기존의 백 단위 쿼리 방식보다 훨씬 적은 비용으로도 빠르게 높은 분류 성능에 도달할 수 있음을 실험적으로 증명하였다. 이는 특히 클래스 수가 많은 복잡한 MIML 데이터셋의 레이블링 효율을 획기적으로 높일 수 있는 방안이 될 것이다.