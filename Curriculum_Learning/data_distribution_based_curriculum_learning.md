# Data Distribution-based Curriculum Learning

Shonal Chaudhry, Anuraganand Sharma (2024)

## 🧩 Problem to Solve

본 논문은 지도 학습(Supervised Learning) 기반의 분류 작업에서 학습 샘플의 제시 순서가 모델의 최종 성능에 큰 영향을 미친다는 점에 주목한다. 일반적으로 Curriculum Learning은 '쉬운' 샘플부터 '어려운' 샘플 순으로 모델을 학습시켜 성능을 높이는 방법론이지만, 무엇이 '쉬운' 샘플인지 정의하는 Scoring Function을 설계하는 것은 매우 까다로운 문제이다.

특히 의료 데이터셋과 같이 데이터의 양이 적거나 다양성이 부족한 소규모 및 중규모 데이터셋의 경우, 데이터의 편향(Bias)이나 불균형이 학습의 불안정성을 초래하고 예측 성능을 저하시킬 가능성이 높다. 따라서 본 논문의 목표는 데이터의 고유한 분포(Inherent Data Distribution)를 활용하여 샘플의 난이도를 결정하고, 이를 통해 분류 모델의 정확도와 수렴 속도를 향상시키는 Data Distribution-based Curriculum Learning (DDCL) 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 데이터셋의 공간적 분포 특성을 활용하여 학습 순서를 결정하는 것이다. 연구진은 샘플의 난이도를 측정하기 위해 두 가지 서로 다른 scoring 방식인 **DDCL-Density**와 **DDCL-Point**를 제안한다.

1. **DDCL-Density**: 데이터의 밀도가 높은 지역에 위치한 샘플을 '쉬운' 샘플로 간주한다. 많은 샘플이 밀집된 영역은 클래스의 전형적인 특징을 잘 나타내므로 초기 학습 단계에서 모델이 빠르게 기초 개념을 잡는 데 유리하다는 직관에 기반한다.
2. **DDCL-Point**: 클래스의 중심점(Centroid)으로부터의 유클리드 거리(Euclidean distance)를 기준으로 난이도를 측정한다. 중심점에 가까운 샘플일수록 전형적(Easy)이며, 멀리 떨어진 샘플일수록 예외적이거나 복잡한(Hard) 샘플로 정의한다.

이 두 가지 방식은 각각 그룹 단위의 통계적 특성과 개별 샘플의 기하학적 위치라는 서로 다른 관점에서 커리큘럼을 구성한다.

## 📎 Related Works

기존의 Curriculum Learning은 주로 사람이 정의한 사전 지식(Prior Knowledge)에 의존하여 고정된 커리큘럼을 생성하거나, 학습 진행 상황에 따라 동적으로 샘플을 선택하는 Self-paced Learning (SPL) 방식으로 발전해 왔다. 특히 SPL은 학습자의 피드백을 반영하지만, 사전 지식을 활용하지 못해 과적합(Overfitting)에 취약하다는 단점이 있다. 이를 보완하기 위해 Self-paced Curriculum Learning (SPCL)이 제안되기도 하였다.

또한, 데이터 밀도를 활용한 기존 연구들이 존재하지만, 대부분 대규모 데이터셋을 전제로 하거나 가짜 라벨(Pseudo-label)을 활용한 도메인 적응(Domain Adaptation) 문제에 집중되어 있었다. 본 논문의 DDCL은 이러한 기존 방식과 달리 **소규모 및 중규모 의료 데이터셋**에서도 효과적으로 작동하도록 설계되었으며, 단순히 밀도를 측정하는 것을 넘어 클래스 중심점(Centroid)과 분위수(Quantile) 분할을 통해 커리큘럼을 체계화했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

DDCL의 전체 파이프라인은 데이터 분포 분석부터 학습 데이터 재배치까지 여러 단계로 구성된다.

### 1. 전체 프로세스 및 단계

전체 시스템 구조는 다음과 같은 순서로 진행된다.

1. **클래스별 그룹화**: 입력 데이터를 타겟 클래스별로 나눈다.
2. **중심점 결정 (Centroid Determination)**: 각 클래스 그룹 내에서 클러스터링을 수행하여 중심점 $O_s$를 계산한다.
3. **거리 계산 및 분포 분석**: 각 샘플 $x$와 해당 클래스 중심점 $O_s$ 사이의 유클리드 거리 $E_s$를 계산하고, 이를 정규화하여 데이터 분포 $D_s$를 확인한다.
4. **분위수 분할 (Quantile Division)**: 계산된 거리 분포를 바탕으로 데이터를 여러 개의 분위수($Q_1, Q_2, Q_3, \dots$)로 나눈다.
5. **오버샘플링 (Optional)**: 특정 분위수에 샘플 수가 너무 적어 불균형이 심한 경우, SMOTE(Synthetic Minority Over-sampling Technique)를 사용하여 합성 샘플을 생성함으로써 데이터 균형을 맞춘다.
6. **Scoring 및 재배치**: 선택한 scoring 방식(Density 또는 Point)에 따라 샘플의 순서를 결정하여 최종 학습 셋 $T$를 구성한다.

### 2. 상세 Scoring 방법론

#### (1) DDCL-Density (밀도 기반)

분위수별 샘플의 개수(Cardinality, $|q|$)를 기반으로 점수를 부여한다.

- **로직**: 샘플 수가 가장 많은 분위수를 가장 '쉬운' 단계로 설정하고, 샘플 수가 가장 적은 분위수를 가장 '어려운' 단계로 설정한다.
- **순서**: $\text{Highest Density} \rightarrow \text{Lowest Density}$ 순으로 데이터를 배치하여 학습시킨다.

#### (2) DDCL-Point (지점 기반)

개별 샘플의 정규화된 유클리드 거리 $\hat{E}_s$를 직접적으로 사용한다.

- **로직**: 중심점으로부터의 거리가 짧은 샘플일수록 높은 점수(Easy)를 부여하고, 거리가 먼 샘플일수록 낮은 점수(Hard)를 부여한다.
- **순서**: $\text{Shortest Distance} \rightarrow \text{Longest Distance}$ 순으로 개별 샘플을 정렬하여 배치한다. 이는 분위수 단위로 묶는 Density 방식과 달리 샘플 단위의 세밀한 정렬이 이루어지므로 순서가 보다 무작위적(Randomized)으로 보일 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋**: UCI Machine Learning Repository에서 수집한 7종의 의료 데이터셋 (Breast Cancer, Cancer, Haberman's Survival, Liver Disorder, Pima Indians Diabetes, New-Thyroid, Diabetes 130)을 사용하였다.
- **비교 모델 (Classifiers)**:
  - **Neural Network**: 입력층, 가변적 은닉층, 출력층으로 구성되었으며, Bayesian optimization으로 하이퍼파라미터를 튜닝하였다.
  - **SVM**: RBF 커널을 사용하였으며 $C=1.0$으로 설정하였다.
  - **Random Forest**: 100개의 Estimator와 노드 분할을 위한 2개의 Feature를 사용하였다.
- **평가 지표**: 정확도(Accuracy), Precision-Recall Curve, Confusion Matrix, Epoch별 Error Loss를 측정하였다.
- **비교군**: 커리큘럼을 적용하지 않은 'No Curriculum' 방식 및 기존 SOTA 연구 결과와 비교하였다.

### 주요 결과

- **정확도 향상**: DDCL을 적용했을 때 모든 데이터셋에서 baseline 대비 성능 향상이 관찰되었으며, 증가 폭은 약 $2\%$에서 $10\%$ 사이였다.
- **모델별 특성**:
  - Neural Network는 Haberman's Survival, Liver Disorder 등에서 높은 최고 성능을 보였다.
  - SVM은 Breast Cancer (Diagnostic) 및 Pima Indians Diabetes에서 높은 평균 정확도를 기록하였다.
  - Random Forest는 Cancer, Liver Disorder, New-Thyroid 등에서 우수한 성능을 나타냈다.
- **수렴 속도**: Neural Network의 학습 곡선을 분석한 결과, DDCL-Density와 DDCL-Point 모두 'No Curriculum' 방식보다 초기 Epoch에서 Error Loss가 더 빠르게 감소하여 수렴 속도가 향상됨을 확인하였다.
- **SOTA 비교**: Table 5에 따르면, DDCL은 기존의 Balanced Stratified Reduction (BSR) 방식이나 표준 NN/RF 모델보다 높은 정확도를 달성하였다. 특히 Diabetes 130 데이터셋에서 RF(DDCL)는 $66.95\%$의 정확도를 보여 기존 $55.97\%$ 대비 큰 폭의 향상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 데이터의 기하학적 분포라는 단순하면서도 강력한 지표를 활용하여 커리큘럼을 자동 생성했다는 점에서 강점이 있다. 특히 의료 데이터와 같이 데이터 획득이 어렵고 크기가 작은 데이터셋에서 SMOTE와 분위수 분할을 결합하여 데이터 부족 문제를 완화함과 동시에 효과적인 학습 경로를 제시하였다. 또한, 단일 모델이 아닌 NN, SVM, RF라는 서로 다른 메커니즘의 분류기 모두에서 성능 향상을 입증함으로써 방법론의 범용성을 증명하였다.

### 한계 및 비판적 해석

1. **고정적 커리큘럼**: 제안된 DDCL은 학습 시작 전 정적으로 결정되는 'Fixed Curriculum'이다. 학습 과정 중에 모델의 현재 상태(Loss, Gradient 등)를 반영하여 샘플 순서를 바꾸는 동적 메커니즘이 결여되어 있다.
2. **Scoring 방식의 단일 적용**: 두 가지 scoring 방식 중 하나만 선택하여 사용하며, 두 방식의 장점을 결합한 앙상블 전략이나 가중치 기반의 통합 scoring 함수에 대한 논의가 부족하다.
3. **클러스터링 세부사항**: 중심점(Centroid)을 결정할 때 구체적으로 어떤 클러스터링 알고리즘을 사용했는지 명시되지 않아 재현성에 제약이 있을 수 있다.

## 📌 TL;DR

본 논문은 의료 데이터와 같은 소규모 데이터셋의 분류 성능을 높이기 위해 데이터 분포 기반의 커리큘럼 학습 전략인 **DDCL**을 제안한다. 클래스 중심점과의 거리 및 밀도를 기준으로 샘플의 난이도를 측정하여 '쉬운' 샘플부터 학습시키는 이 방식은, 기존 baseline 대비 **정확도를 $2 \sim 10\%$ 향상**시켰으며 **학습 수렴 속도를 가속화**하였다. 이 연구는 데이터 확보가 어려운 의료 AI 분야에서 데이터 순서 최적화만으로도 성능을 끌어올릴 수 있음을 시사하며, 향후 동적 커리큘럼 및 이미지/텍스트 데이터로의 확장 가능성을 제시한다.
