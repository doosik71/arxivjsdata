# Distribution-Free Federated Learning with Conformal Predictions

Charles Lu, Jayashree Kalpathy-Cramer (2022)

## 🧩 Problem to Solve

본 논문은 의료분야의 협력적 기계 학습을 위해 사용되는 연합 학습(Federated Learning, FL) 환경에서 모델의 불확실성 추정(Uncertainty Estimation)과 보정(Calibration) 문제를 해결하고자 한다.

의료 영상 데이터는 개인정보 보호 문제로 인해 각 기관의 데이터를 외부로 공유하기 어렵기 때문에 연합 학습이 매력적인 대안으로 제시되어 왔다. 그러나 일반적인 합성곱 신경망(CNN) 모델은 예측 결과에 대해 과잉 확신(Overconfidence)하는 경향이 있으며, 보정이 제대로 이루어지지 않아 실제 임상 현장에서 의료진이 모델의 결과를 신뢰하고 의사결정에 활용하기에는 위험 요소가 크다.

따라서 본 연구의 목표는 모델의 아키텍처를 수정하지 않고도, 통계적으로 보장된 범위 내에서 예측 결과의 신뢰도를 제공할 수 있는 Distribution-Free(분포 무관) 방식의 Conformal Prediction(적합 예측) 프레임워크를 연합 학습에 통합하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 연합 학습으로 훈련된 모델의 추론 단계에 adaptive conformal calibration을 도입하여, 정해진 신뢰 수준(Confidence level)에서 정답 클래스가 반드시 포함되도록 보장하는 '예측 집합(Prediction Sets)'을 생성하는 것이다.

중심적인 설계 포인트는 다음과 같다:
1. **Distribution-Free Guarantee**: 데이터의 분포에 대한 엄격한 가정 없이도, 주어진 신뢰 수준 $1-\alpha$에 대해 정답이 집합 내에 포함될 확률(Coverage)을 수학적으로 보장한다.
2. **Federated Quantile Estimation**: 개별 기관의 로컬 데이터로만 보정하는 대신, 여러 기관에서 추정된 분위수(Quantile)를 연합하여 더 정밀하고 효율적인(크기가 작은) 예측 집합을 구축한다.
3. **Model Agnostic**: 모델의 손실 함수나 구조를 변경할 필요 없이, Softmax 확률값과 같은 스코어링 함수만 있다면 어떤 모델에도 적용 가능한 post-processing 방식으로 설계되었다.

## 📎 Related Works

### 관련 연구 및 한계
1. **Conformal Inference**: 임의의 머신러닝 모델에 엄격한 불확실성을 통합하는 프레임워크로, 분산 없이 marginal coverage bound를 제공한다. 하지만 이를 의료 영상의 연합 학습 환경에 구체적으로 적용한 연구는 부족했다.
2. **Bayesian Approximation & Post-hoc Calibration**: MC Dropout과 같은 베이지안 접근법은 가우시안 분포와 같은 엄격한 가정(Prior)이 필요하며, Temperature Scaling과 같은 사후 보정 기법은 유용하지만 Conformal Prediction이 제공하는 것과 같은 통계적 보장(Statistical guarantee)을 제공하지 못한다.
3. **Federated Learning (FL)**: 데이터 프라이버시를 유지하며 협력 학습을 가능하게 하지만, 비-IID(non-IID) 데이터 분포로 인한 보정 문제와 모델의 불확실성 측정 문제는 여전히 해결해야 할 과제로 남아 있다.

### 차별점
본 논문은 기존의 불확실성 추정 방식들이 가진 '가정 의존성'이나 '모델 수정 필요성'을 제거하고, 연합 학습 환경에서 모든 참여 기관이 합의된 분위수를 공유함으로써 로컬 보정보다 더 효율적인(Tighter) 예측 집합을 생성한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
본 시스템은 $\text{FedAvg}$를 통해 전역 모델을 훈련한 후, 추론 단계에서 Conformal Prediction을 통해 단일 클래스가 아닌 '클래스 집합'을 출력한다.

### 핵심 구성 요소 및 절차
1. **Scoring Function**: CNN 분류기의 Softmax 확률값 $S(X)$를 적합성 점수(Conformality score)로 사용한다.
2. **Prediction Set 생성**: 예측 집합 $C(X)$는 다음과 같이 정의된다.
   $$C(X) = \{y \in Y \mid S(X)_y > 1 - \hat{q}\}$$
   여기서 $S(X)_y$는 $y$번째 클래스의 점수이며, $\hat{q}$는 보정 데이터셋을 통해 추정된 점수 분위수(Score quantile)이다.
3. **Quantile $\hat{q}$의 추정**: 신뢰 수준 $1-\alpha$를 달성하기 위한 분위수 $\hat{q}$는 다음과 같이 계산된다.
   $$\hat{q} = \frac{\lceil (N+1)(1-\alpha) \rceil}{N}$$
   ($N$은 보정 데이터셋의 샘플 수)
4. **Federated Conformal Prediction**:
   - 각 기관 $k \in \{1, \dots, K\}$는 자신의 로컬 검증 셋을 사용하여 로컬 분위수 $\hat{q}_k$를 계산한다.
   - 서버는 이 값들을 평균 내어 전역 분위수를 도출한다: $\hat{q}_{fed} = \frac{1}{K} \sum_{k=1}^K \hat{q}_k$.
   - 최종적으로 이 $\hat{q}_{fed}$를 임계값으로 사용하여 예측 집합을 형성한다.

### 알고리즘 흐름 (Algorithm 1 요약)
- 입력: 연합 모델 $S$, $K$개의 검증 셋, 테스트 샘플 $X_m$, 신뢰 수준 $1-\alpha$.
- 과정: 각 기관에서 정답 클래스의 점수 $s_i$를 추출 $\rightarrow$ 각 기관의 분위수를 계산 $\rightarrow$ 이를 평균 내어 $\hat{q}$ 결정 $\rightarrow$ 테스트 샘플의 점수가 $\hat{q}/K$보다 큰 모든 클래스를 집합에 포함.
- 출력: 신뢰 구간이 보장된 예측 집합 $C(X_m)$.

## 📊 Results

### 실험 설정
- **데이터셋**: MedM-NIST 벤치마크 중 6개 데이터셋 (Blood, Derma, Path, Tissue, Retina, Organ3D).
- **모델**: ResNet-18, Cross Entropy Loss 사용.
- **비교 대상**:
    - **Naive**: 단순히 점수 순으로 정렬하여 $1-\alpha$ 임계값까지 클래스를 추가 (보정 과정 없음).
    - **Local Conformal**: 단일 기관의 검증 셋으로만 $\hat{q}$를 추정.
    - **Federated Conformal**: 모든 기관의 $\hat{q}$를 평균 내어 사용.
- **지표**: Coverage(정답이 집합에 포함될 확률) 및 Cardinality(집합의 평균 크기).

### 주요 결과
1. **Coverage**: Naive 방식은 거의 모든 클래스를 포함시키는 극도로 보수적인 결과를 보였으나, Local 및 Federated 방식은 설정한 $1-\alpha$ (예: 90%) 수준의 Coverage를 적절히 달성하였다.
2. **Cardinality (효율성)**: 동일한 Coverage를 유지할 때, 예측 집합의 크기는 $\text{Federated} < \text{Local} \ll \text{Naive}$ 순으로 나타났다. 특히 Federated 방식은 로컬 방식보다 더 작은 집합으로도 정답을 포함할 수 있어 훨씬 더 유용한 정보를 제공한다.
3. **강건성**: 일부 기관의 검증 데이터 레이블을 30% 무작위로 섞어 노이즈를 주었음에도 불구하고, 연합 방식의 예측 집합 크기는 로컬 방식보다 작게 유지되며 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **Softmax의 한계 증명**: Naive 방식이 100% Coverage를 위해 모든 클래스를 출력해야 했다는 점은, 딥러닝 모델의 Softmax 확률값이 실제 확률과 매우 다르게 보정되어 있다는(Overconfident) 기존 연구를 뒷받침한다.
- **연합 학습의 이점**: 단순한 모델 파라미터의 평균뿐만 아니라, 보정을 위한 분위수(Quantile)를 연합함으로써 단일 기관이 가진 데이터의 한계를 극복하고 더 정밀한(Tighter) 불확실성 추정이 가능함을 보였다.

### 한계 및 논의사항
- **데이터 규모의 영향**: RetinaMNIST나 OrganMNIST3D와 같이 데이터 샘플 수가 매우 적은 데이터셋에서는 Federated 방식과 Local 방식의 차이가 크지 않았으며, 분산(Variance)이 크게 나타났다. 이는 충분한 양의 보정 데이터가 확보되어야 연합 보정의 이점이 극대화됨을 시사한다.
- **에피스테믹 불확실성과의 관계**: 클래스 엔트로피(Class Entropy)와 예측 집합의 크기(Cardinality) 사이에 양의 상관관계가 있음을 확인하였다. 이는 Conformal Prediction이 모델의 내재적 불확실성(Epistemic uncertainty)을 효과적으로 반영하고 있음을 의미한다.

## 📌 TL;DR

본 논문은 의료 영상 분석을 위한 연합 학습(FL) 모델에 **Conformal Prediction** 프레임워크를 도입하여, 데이터 분포에 관계없이 통계적으로 보장된 **예측 집합(Prediction Sets)**을 생성하는 방법을 제안한다. 모델 수정 없이 사후 처리만으로 구현 가능하며, 특히 여러 기관의 분위수를 연합하여 계산함으로써 단일 기관 보정보다 더 작고 효율적인(Tighter) 예측 집합을 얻을 수 있음을 입증하였다. 이 연구는 높은 신뢰도가 요구되는 의료 AI 분야에서 모델의 불확실성을 정량화하고 임상적 신뢰도를 높이는 데 중요한 기여를 할 것으로 기대된다.