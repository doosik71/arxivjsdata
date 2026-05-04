# Address Instance-level Label Prediction in Multiple Instance Learning

Minlong Peng, Qi Zhang (2019)

## 🧩 Problem to Solve

Multiple Instance Learning (MIL)은 개별 인스턴스(instance)의 레이블은 알 수 없고, 오직 인스턴스들의 집합인 백(bag)의 레이블만 주어진 상태에서 학습하는 방법론이다. 기존의 MIL 연구들은 주로 백 수준의 레이블 예측(bag-level label prediction)에 집중해 왔으며, 이에 따라 손실 함수(loss function) 역시 백 수준에서 정의되었다.

하지만 이미지 세그멘테이션(image segmentation)이나 세밀한 감성 분류(fine-grained sentiment classification)와 같은 많은 실제 작업에서는 인스턴스 수준의 레이블 예측(instance-level label prediction)이 훨씬 더 중요하다. 기존의 인스턴스 수준 패러다임 방법론들조차 인스턴스 예측을 백 레이블 예측을 위한 중간 단계로만 활용할 뿐, 인스턴스 수준에서 직접적으로 손실을 정의하여 최적화하지 않는다. 이로 인해 인스턴스 수준의 예측 성능이 저하되는 문제가 발생하며, 본 논문은 인스턴스 레이블이 없는 상황에서 인스턴스 수준의 레이블 예측 성능을 극대화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 레이블을 직접 사용하지 않고도 인스턴스 수준의 손실을 편향 없이(unbiasedly) 추정하여 최적화하는 것이다.

주요 기여 사항은 다음과 같다.

1. 인스턴스 수준의 레이블 예측을 위해 손실 함수가 인스턴스 수준에서 구체적으로 정의된 새로운 MIL 알고리즘을 제안하였다.
2. i.i.d 가정을 바탕으로, 인스턴스 레이블 없이도 인스턴스 수준의 손실을 편향 없이, 그리고 일관되게(consistently) 추정할 수 있음을 이론적으로 증명하였다.
3. 유계(bounded)된 Bayes consistent 손실 함수를 사용할 경우, 제안 방법이 인스턴스 레이블을 모두 알고 있는 완전 지도 학습(fully supervised) 모델과 유사한 성능을 낼 수 있음을 보였다.
4. 이미지 및 텍스트 데이터셋 실험을 통해 제안한 알고리즘이 백 수준과 인스턴스 수준 모두에서 우수한 성능을 보임을 검증하였다.

## 📎 Related Works

기존 MIL 알고리즘은 크게 두 가지 패러다임으로 나뉜다.

1. **Instance-level paradigm**: 인스턴스를 개별적으로 처리하여 인스턴스 레이블을 먼저 예측한 후, 이를 통해 백 레이블을 추론한다. Representative한 방법론으로는 Diverse Density (DD), Multiple Instance Logistic Regression (MILR), 그리고 최근의 miNet 등이 있다. 하지만 이들은 인스턴스 예측이 최종 목적이 아닌 중간 단계이므로 인스턴스 수준의 정확도가 낮다는 한계가 있다.
2. **Bag-level paradigm**: 백을 하나의 전체 단위로 취급하여 백 표현(bag representation)을 직접 얻는다. MI-kernel, miFV, 그리고 Attention 기반의 MIGA 등이 이에 해당한다. 이 방법들은 백 수준 예측에는 강점이 있으나 인스턴스 수준의 정보를 복원하는 데는 취약하다.

본 논문은 기존 방법들이 인스턴스 수준의 손실을 직접 최적화하지 않았다는 점을 지적하며, 이론적 근거를 바탕으로 인스턴스 수준의 리스크 최소화(risk minimization)를 수행한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 기본 가정 및 정의

본 논문은 인스턴스들이 독립 동일 분포(i.i.d)를 따르며, 인스턴스 레이블이 주어졌을 때 인스턴스 분포가 백 레이블과 독립이라는 가정을 세운다.

### 2. 인스턴스 수준 리스크 추정 ($\hat{R}^u_L$)

인스턴스 레이블 $y_x$를 모르는 상태에서 인스턴스 수준의 $L$-리스크 $R_L(f) = E_{X, Y_X} L(f(x), y_x)$를 추정하기 위해 다음과 같은 추정식 $\hat{R}^u_L(f)$를 제안한다.

$$\hat{R}^u_L(f) = \frac{1}{\sum_{b \in S} |b|} \sum_{b \in S} \sum_{x \in b} L(f(x), 1) + P(Y_X = 0) \sum_{b \in S^-} \frac{1}{|b|} \sum_{x \in b} (L(f(x), 0) - L(f(x), 1))$$

여기서 $S^-$는 음성 백(negative bags)의 집합이며, $P(Y_X = 0)$는 전체 인스턴스 중 음성 인스턴스의 비율이다. Theorem 1을 통해 이 추정치가 $R_L(f)$의 편향 없는 추정치(unbiased estimation)임을 증명하였다.

### 3. 유계된 Bayes Consistent 손실 함수의 중요성

논문은 손실 함수 $L$이 Bayes consistent해야 하며, 특히 **유계(bounded)**되어야 함을 강조한다. 만약 $L$이 상한선이 없는(unbounded) 함수(예: Cross Entropy)라면, 모델이 $L(f(x), 1) \to \infty$가 되도록 하여 리스크 값을 인위적으로 낮추는 과적합(overfitting) 문제가 발생할 수 있다. 따라서 본 연구에서는 유계된 손실 함수인 Mean Square Error (MSE)를 주로 사용한다.

### 4. MIL 백 제약 조건의 정식화 (BIMIL)

위의 리스크 추정치만으로는 MIL의 기본 제약 조건(음성 백은 모든 인스턴스가 음성, 양성 백은 최소 하나가 양성)을 만족시키지 못할 수 있다. 이를 해결하기 위해 다음과 같은 최적화 문제를 정의하여 **BIMIL (Bag- and Instance-level MIL)** 알고리즘을 구성한다.

$$\min_f \hat{R}^u_L + C \left( \sum_{i=1}^{|S^-|} \xi_i + \sum_{j=1}^{|S^+|} \xi_j \right)$$
$$\text{s.t. } \max_{x \in b_i} L(f(x), 0) - L(f(x), 1) < \xi_i, \forall b_i \in S^-$$
$$\max_{x \in b_j} L(f(x), 0) - L(f(x), 1) > -\xi_j, \forall b_j \in S^+$$
$$\xi_i, \xi_j > 0$$

여기서 $C$는 하이퍼파라미터이며, $\xi$는 제약 조건을 완화하기 위한 슬랙 변수(slack variable)이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MNIST, SVHN, CIFAR10 (이미지), 20Newsgroup (텍스트)
- **지표**: Test Instance-level / Bag-level Bayes Risk ($\times 100$)
- **비교 대상**:
  - 완전 지도 학습 모델 (Sup)
  - 인스턴스 수준 방법론: DD, MILR, miNet
  - 백 수준 방법론: miFV, MIGA
  - 제안 방법론의 변형: IMIL (백 제약 조건 없이 $\hat{R}^u_L$만 최소화) 및 BIMIL (제안 방법 전체)

### 2. 주요 결과

- **인스턴스 수준 성능**: Table 1 결과, BIMIL은 DD, MILR 등의 기존 인스턴스 수준 방법론보다 월등히 낮은 Bayes Risk를 기록했으며, 완전 지도 학습(Sup) 모델과 매우 유사한 성능을 보였다.
- **백 수준 성능**: Table 2 결과, BIMIL은 최신 MIL 방법론(MIGA 등)과 대등하거나 더 우수한 성능을 보였으며, 특히 IMIL보다 BIMIL이 더 좋은 성능을 냄으로써 백 제약 조건 추가의 효과를 입증하였다.
- **손실 함수 영향**: Figure 3에서 Cross Entropy (CE)를 사용했을 때 학습 리스크는 급격히 감소하지만 테스트 리스크는 증가하는 심각한 과적합이 관찰되었다. 반면 MSE를 사용했을 때는 과적합이 억제되고 안정적인 성능을 보였다.
- **$P(Y_X=0)$ 값의 민감도**: Figure 2를 통해 실제 값 주변에서 모델이 상당히 강건(robust)하게 작동함을 확인하였으며, 약간 과대평가된 값에서 오히려 더 좋은 성능을 내는 경향이 있음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 MIL 환경에서 인스턴스 수준의 레이블을 예측하기 위한 이론적 토대와 실용적인 알고리즘을 성공적으로 제시하였다.

**강점 및 통찰**:

- **이론적 정당성**: 단순한 휴리스틱이 아니라, 리스크 추정의 편향 없음(unbiasedness)과 일관성(consistency)을 수학적으로 증명하여 신뢰도를 높였다.
- **손실 함수의 선택**: 딥러닝에서 흔히 쓰이는 Cross Entropy가 MIL의 인스턴스 리스크 추정 시에는 유계되지 않은 특성 때문에 치명적인 과적합을 유발할 수 있다는 점을 밝혀낸 것은 매우 중요한 학술적 기여이다.

**한계 및 논의사항**:

- **i.i.d 가정**: 인스턴스가 독립 동일 분포를 따른다는 가정을 전제로 한다. 하지만 실제 데이터(예: 의료 영상의 인접 픽셀)에서는 인스턴스 간 강한 상관관계가 존재할 가능성이 높으며, 이 경우 성능이 어떻게 변할지에 대한 분석이 부족하다.
- **$P(Y_X=0)$의 획득**: 이론적으로는 이 비율을 알아야 하지만, 실제 환경에서 이를 정확히 어떻게 추정하거나 설정할 것인지에 대한 구체적인 가이드라인이 부족하다. (실험에서는 알고 있다고 가정함)

## 📌 TL;DR

본 논문은 인스턴스 레이블이 없는 MIL 환경에서 **인스턴스 수준의 손실을 편향 없이 추정하여 최적화하는 BIMIL 알고리즘**을 제안하였다. 특히 **유계된(bounded) Bayes consistent 손실 함수**를 사용함으로써 완전 지도 학습 모델에 근접하는 인스턴스 예측 성능을 달성하였으며, 이는 백 수준의 예측 성능 향상으로도 이어졌다. 이 연구는 인스턴스 단위의 정밀한 분석이 필요한 의료 영상 분석 등의 분야에 직접적으로 적용될 가능성이 매우 높다.
