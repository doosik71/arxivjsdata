# Detection of Anomalies in Multivariate Time Series Using Ensemble Techniques

Anastasios Iliopoulos, John Violos, Christos Diou, Iraklis Varlamis (2023)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series) 데이터에서 이상치(Anomaly)를 탐지하는 문제를 다룬다. 다변량 시계열 데이터의 특성상 이상치는 매우 드물게 발생하며, 이로 인해 데이터 불균형(Imbalanced data) 문제가 발생하여 일반적인 분류 알고리즘으로 해결하기 어렵다.

특히, 고차원 데이터에서 이상치는 전체 피처 세트가 아닌 매우 작은 부분 집합(Subset)에서만 나타나는 경우가 많다. 따라서 모든 피처를 한꺼번에 고려하는 기존의 딥러닝 모델들은 차원의 저주(Curse of Dimensionality)나 노이즈로 인해 이상치를 효과적으로 포착하지 못하는 한계가 있다. 본 연구의 목표는 Feature Bagging과 Nested Rotation 기술을 결합한 앙상블 기법을 통해 이러한 문제를 해결하고 이상치 탐지 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 개별 모델의 다양성을 극대화하여 고차원 데이터 속에 숨겨진 이상치를 효과적으로 찾아내는 것이다. 이를 위해 다음과 같은 설계 전략을 제안한다.

1. **Feature Bagging 적용**: 전체 피처 중 무작위로 선택된 부분 집합만을 사용하여 여러 개의 기본 모델(Base models)을 학습시킴으로써, 특정 피처 부분 집합에 숨겨진 이상치를 포착할 가능성을 높인다.
2. **Nested Rotation 도입**: PCA(주성분 분석)를 기반으로 피처 부분 집합을 다시 분할하고 회전(Rotation)시키는 변환을 적용하여 데이터의 다양성을 주입하고 모델의 일반화 성능을 개선한다.
3. **앙상블 전략 구축**: 비지도 학습(Unsupervised) 환경에서는 다수결 투표(Majority Voting)를, 준지도 학습(Semi-supervised) 환경에서는 Logistic Regressor를 사용하는 Stacking 방식을 통해 최종 결정을 내린다.

## 📎 Related Works

논문에서는 이상치 탐지 기법을 통계적 방법, 머신러닝 방법, 딥러닝 방법으로 구분한다. 최근에는 LSTM, Autoencoder, CNN 기반의 딥러닝 모델들이 사용되고 있으나, 여전히 다변량 시계열 데이터의 복잡성으로 인해 성능 개선의 여지가 많다고 지적한다.

기존 연구 중 Feature Bagging은 고차원 데이터에서 거리 기반 메트릭이 의미를 잃는 문제를 해결하기 위해 제안되었으며, Rotation Forest는 PCA를 통해 피처 공간을 회전시켜 분류기의 다양성을 높이는 방식을 제안하였다. 본 논문은 이러한 아이디어들을 이상치 탐지 분야에 접목하여, 단순한 모델 결합을 넘어 피처 수준의 변환과 앙상블을 동시에 수행한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

본 제안 방법론은 다변량 시계열 데이터를 입력받아 최종적으로 이상치 여부(Binary output)를 판별하는 파이프라인을 가진다. 전체 과정은 **Feature Bagging $\rightarrow$ Nested Rotations $\rightarrow$ Base Model Training $\rightarrow$ Ensemble Aggregation** 순으로 진행된다.

### 주요 구성 요소 및 절차

#### 1. Feature Bagging

고차원 공간에서의 노이즈를 줄이고 이상치가 존재할 가능성이 높은 하위 공간(Subspace)을 생성하는 단계이다.

- 전체 피처 수 $d$에 대하여, 각 모델 $m$마다 $\lfloor d/2 \rfloor$와 $d-1$ 사이의 무작위 정수 $N_m$을 선택한다.
- 원본 데이터 $X$에서 중복 없이 $N_m$개의 피처를 무작위로 추출하여 부분 집합 $F^m$을 구성한다.

#### 2. Nested Rotations (PCA 기반 변환)

Feature Bagging으로 선택된 $F^m$ 내에서도 다양성을 더 확보하기 위해 적용하는 변환 기법이다.

- 부분 집합 $F^m$을 $K$개의 파티션($S_1, S_2, \dots, S_K$)으로 나눈다.
- 각 파티션에 대해 무작위 샘플링(Subsampling)을 수행한 후 PCA를 적용하여 회전 행렬 $R^m_k$를 계산한다.
- 최종 회전 행렬 $R^m$은 각 파티션의 회전 행렬을 대각 성분으로 가지는 블록 대각 행렬(Block Diagonal Matrix) 형태로 구성된다.
  $$R^m = \begin{pmatrix} R^m_1 & 0 & \dots & 0 \\ 0 & R^m_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & R^m_K \end{pmatrix}$$
- 변환된 데이터는 $\text{new data} = R^m \times F^m$으로 계산되어 기본 모델의 입력으로 사용된다.

#### 3. 기본 모델(Base Models) 및 학습

다양한 특성을 포착하기 위해 다음의 5가지 아키텍처를 기본 모델로 사용한다:

- Autoencoder, Convolutional Autoencoder, LSTM, LSTM Autoencoder, LSTM Variational Autoencoder (VAE).

#### 4. 앙상블 및 추론 절차

각 모델 $i$가 산출한 이상치 점수 $ASc_i(X_t)$를 기반으로 이진 결과 $ASc_{binary}^i(X_t)$를 도출한다. 이때 임계값 $\delta$는 $1.5 \times IQR$ (Interquartile Range)로 설정된다.

- **Unsupervised setup**: 다수결 투표(Majority Voting)를 통해 최종 결과를 결정한다.
  $$ASc_{binary}(X_t) = \text{agg}_{\forall i}(ASc_{binary}^i(X_t))$$
- **Semi-supervised setup**: 학습 데이터를 Subset A(기본 모델 학습용)와 Subset B(Logistic Regressor 학습용)로 분리한다. Subset B에 대해 기본 모델들이 예측한 값들을 입력으로 하고 실제 라벨을 타겟으로 하여 Logistic Regressor를 학습시켜 최적의 결합 방식을 찾는다.

## 📊 Results

### 실험 설정

- **데이터셋**: Skoltech Anomaly Benchmark (SKAB). 수중 펌프 장치의 센서 데이터(8개 피처)를 포함하며, 정상 상태에서 시작하여 밸브 전환을 통해 이상치가 주입된 시계열 데이터이다.
- **평가 지표**: 데이터 불균형을 고려하여 F1-Score와 AUC (Area Under the ROC Curve)를 사용한다.
- **비교 대상**: Plain 모델(단일 모델), Feature Bagging 적용 모델, Feature Bagging + Nested Rotations 적용 모델.

### 주요 결과

- **정량적 성과**:
  - **비지도 학습**: Convolutional Autoencoder에 Feature Bagging과 Nested Rotations를 결합했을 때 가장 높은 성능($F1=0.7873, AUC=0.8315$)을 보였다. 일반 모델 대비 약 2~4%의 성능 향상이 관찰되었다.
  - **준지도 학습**: Logistic Regressor를 이용한 Stacking 모델이 가장 우수한 성과($F1=0.85, AUC=0.88$)를 기록하였으며, 이는 기본 모델들보다 최소 10% 이상 향상된 수치이다.
- **정성적 분석**: Feature Bagging 단독으로는 일부 모델에서만 성능 향상이 있었으나, Nested Rotations와 결합했을 때 대부분의 아키텍처에서 일관된 성능 향상이 나타났다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 Feature Bagging과 Nested Rotations가 서로 보완적인 역할을 수행함을 입증하였다. **Feature Bagging**은 이상치가 영향을 미치는 핵심 피처 부분 집합을 드러내는 역할을 하며, **Nested Rotations**는 PCA를 통해 데이터에 다양성을 주입하여 앙상블의 전체적인 성능을 끌어올린다. 특히 준지도 학습 방식을 통해 각 모델의 예측값을 최적으로 조합함으로써 탐지 정확도를 극대화할 수 있음을 보여주었다.

### 한계 및 비판적 논의

1. **계산 복잡도**: 앙상블 모델의 특성상 다수의 기본 모델을 학습시켜야 하며, 특히 각 모델마다 PCA 연산이 반복적으로 수행되므로 학습 및 추론 시간이 상당히 증가한다. 저자는 병렬 처리를 해결책으로 제시하였으나, 실시간 탐지 시스템에 적용하기에는 오버헤드가 클 수 있다.
2. **하이퍼파라미터 의존성**: 모델의 수 $M$, 파티션 수 $K$, 샘플링 비율 등 설정해야 할 파라미터가 많으며, 이는 Grid Search를 통해 결정되었다. 이러한 설정이 다른 데이터셋에서도 일반화될 수 있을지에 대한 분석이 부족하다.
3. **데이터셋의 한정성**: SKAB라는 단일 벤치마크 데이터셋에서만 검증되었으므로, 더 다양한 도메인의 다변량 시계열 데이터에 대한 검증이 필요하다.

## 📌 TL;DR

본 논문은 다변량 시계열의 고차원 데이터에서 이상치가 소수 피처에 숨어 있다는 점에 착안하여, **Feature Bagging**과 **PCA 기반의 Nested Rotations**를 결합한 앙상블 모델을 제안하였다. 비지도 학습에서는 다수결 투표를, 준지도 학습에서는 Logistic Regression 기반의 Stacking을 적용하여 성능을 높였으며, 실험 결과 준지도 학습 모델이 가장 높은 성능 향상을 보였다. 이 연구는 고차원 시계열 데이터의 이상치 탐지에서 피처 수준의 다양성 확보가 성능 향상의 핵심임을 시사하며, 향후 고차원 데이터 분석 및 산업 설비 이상 탐지 시스템에 적용될 가능성이 높다.
