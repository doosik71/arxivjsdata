# MARIA: a Multimodal Transformer Model for Incomplete Healthcare Data

Camillo Maria Caruso, Paolo Soda, Valerio Guarrasi (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 헬스케어 분야의 멀티모달 데이터셋에서 빈번하게 발생하는 **데이터 결측(Missing Data)** 문제이다. 의료 데이터는 센서 고장, 환자의 비순응, 기술적 제한 또는 개인정보 보호 제한 등으로 인해 특정 피처(Feature)가 누락되거나 아예 특정 모달리티(Modality) 전체가 부재하는 경우가 많다.

이러한 결측치는 머신러닝 모델의 성능을 심각하게 저하시킨다. 기존의 많은 접근 방식은 결측치를 인위적으로 채워넣는 **Imputation(대체)** 기법에 의존하는데, 이는 데이터에 편향(Bias)을 도입하거나 정보 손실을 초래할 위험이 있다. 따라서 본 연구의 목표는 데이터 대체 과정 없이도 결측치에 강건하게(Resilient) 작동하며, 가용한 데이터만을 효율적으로 활용하여 진단 및 예후 예측 정확도를 높이는 멀티모달 딥러닝 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Imputation 없이 가용한 정보만을 선택적으로 처리하는 modified masked self-attention 메커니즘**을 적용한 **Intermediate Fusion(중간 융합)** 전략의 도입이다.

1.  **Modified Masked Self-Attention**: 결측된 데이터에 대해 합성 값을 생성하는 대신, 마스킹 행렬을 통해 어텐션 계산 과정에서 결측치의 기여도를 완전히 제거함으로써 모델의 강건성을 확보한다.
2.  **Intermediate Fusion Architecture**: 모달리티별 인코더(Modality-specific encoders)와 공유 인코더(Shared encoder)를 계층적으로 배치하여, 각 모달리티의 특성을 유지하면서도 모달리티 간의 복잡한 상호작용을 효과적으로 캡처한다.
3.  **Stochastic Regularization**: 훈련 단계에서 Modality Dropout과 Feature Dropout을 통해 의도적으로 결측 상황을 시뮬레이션함으로써, 테스트 단계에서 마주할 다양한 결측 수준에 대해 모델이 일반화될 수 있도록 한다.

## 📎 Related Works

논문에서는 멀티모달 융합 전략을 세 가지로 구분하여 설명하며 각각의 한계를 지적한다.

-   **Early Fusion**: 원시 데이터 수준에서 특징을 결합한다. 구조가 단순하지만, 단 하나의 모달리티만 누락되어도 전체 입력 벡터에 영향을 주어 성능이 급격히 저하되는 취약점이 있다.
-   **Late Fusion**: 각 모달리티별로 독립적인 모델을 학습시킨 후 최종 결정 단계에서 결과를 합산한다. 모달리티 누락에는 유연하지만, 서로 다른 모달리티 간의 정교한 상호작용(Cross-modal interactions)을 학습할 수 없다는 치명적인 단점이 있다.
-   **Intermediate Fusion**: 각 모달리티의 특징을 먼저 추출한 후 중간 단계에서 융합한다. 앞선 두 방식의 균형을 맞춘 형태로, 모달리티 간 관계를 동적으로 학습할 수 있어 가장 유망하지만 계산 복잡도가 높다는 특징이 있다.

또한, 기존의 결측치 처리 방식인 k-Nearest Neighbors (kNN) imputer나 트리 기반 모델의 Missing in Attributes (MIA) 전략, 그리고 GAN이나 Bayesian meta-learning을 이용한 최신 DL 기법들이 언급된다. 하지만 저자들은 이러한 방법들이 공통적으로 **인위적인 데이터 채우기**에 의존하여 편향을 유발한다는 점을 한계로 제시하며, MARIA의 '대체 없는 처리' 방식과의 차별점을 강조한다.

## 🛠️ Methodology

### 전체 파이프라인
MARIA는 모달리티별 인코더 $E_i$들과 하나의 공유 인코더 $E_{sh}$로 구성된 중간 융합 구조를 가진다. 탭 데이터(Tabular data)를 입력으로 받으며, 각 모달리티는 독립적인 인코더를 거쳐 잠재 표현(Latent representation) $r_i$로 변환된 후, 이들이 결합되어 $r_{sh}$가 되고 최종적으로 예측값 $y$를 출력한다.

### 상세 구성 요소 및 동작 원리

**1. 모달리티별 인코더 ($E_i$)**
각 모달리티 $X_i$는 룩업 테이블을 통해 임베딩된 후, 선형 변환을 통해 Query($Q_i$), Key($K_i$), Value($V_i$) 행렬을 생성한다.
$$\begin{cases} Q_i = X_i \cdot W_{Q_i} \\ K_i = X_i \cdot W_{K_i} \\ V_i = X_i \cdot W_{V_i} \end{cases}$$
여기서 $W_{Q_i}, W_{K_i}, W_{V_i}$는 학습 가능한 가중치 행렬이다.

**2. Modified Masked Self-Attention (MSA)**
가장 핵심적인 부분으로, 결측치가 어텐션 가중치에 영향을 주지 않도록 다음과 같은 수식을 사용한다.
$$MSA(Q_i, K_i, V_i) = \text{ReLU}(\text{softmax}(\frac{Q_i K_i^T}{\sqrt{d_h}} + M_i) + M_i^T) V_i$$
여기서 $M_i$는 마스킹 행렬로, 데이터가 누락된 위치에는 $-\infty$를, 존재하는 위치에는 $0$을 할당한다. $\text{softmax}$를 거치면 $-\infty$ 값은 $0$에 가까워지며, 이후 $\text{ReLU}$와 $M_i^T$의 합산 과정을 통해 결측치와 관련된 모든 가중치가 완전히 $0$이 되도록 설계되었다.

**3. 공유 인코더 ($E_{sh}$)**
모든 모달리티 인코더에서 나온 결과물 $r_1, \dots, r_n$을 연결(Concatenation)하여 $r_{sh}$를 구성한다. 공유 인코더는 이 $r_{sh}$를 입력으로 받아 위와 동일한 방식의 Modified MSA를 한 번 더 적용함으로써 모달리티 간의 전역적인 관계를 학습한다. 이때의 마스킹 행렬 $M_{sh}$는 특정 모달리티 전체가 누락되었을 때 해당 모달리티의 기여도를 완전히 제거하는 역할을 한다.

### 훈련 절차 및 정규화
모델의 일반화 성능을 높이기 위해 훈련 과정에서 두 가지 Dropout 전략을 사용한다.
-   **Modality Dropout**: 무작위로 선택된 일부 모달리티 전체를 마스킹하여, 일부 데이터 소스가 완전히 부재한 상황을 학습시킨다. (최소 하나 이상의 모달리티는 남겨둠)
-   **Feature Dropout**: 특정 모달리티 내에서 무작위로 일부 피처를 마스킹하여 피처 수준의 결측 상황에 대비한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: ADNI (알츠하이머병 진단 및 예후 예측), AIforCOVID (코로나19 중증도 및 사망 예측)
-   **작업**: 이진 분류, 다중 클래스 분류, 시점별(12, 24, 36, 48개월) 예후 예측
-   **비교 대상**: 총 10개의 베이스 모델(AdaBoost, Random Forest, XGBoost, MLP, TabNet, FTTransformer 등) $\times$ Imputation(kNN) $\times$ 융합 전략(Early, Late, Intermediate)으로 구성된 총 32가지 설정
-   **지표**: 데이터 균형에 따라 AUC(Area Under the ROC Curve)와 MCC(Matthews Correlation Coefficient)를 사용함

### 주요 결과
1.  **종합 성능**: MARIA는 8가지 모든 작업에서 기존 ML 및 DL 모델들보다 일관되게 높은 성능을 보였다.
2.  **결측치 강건성**: 훈련 및 테스트 데이터의 결측률이 증가할수록 MARIA와 경쟁 모델 간의 성능 격차가 더 벌어졌다. 이는 MARIA의 정규화 전략과 마스킹 메커니즘이 극한의 결측 상황에서도 효과적임을 입증한다.
3.  **융합 전략 비교**: 
    -   ML 모델의 경우 Early Fusion이 Late Fusion보다 우수했는데, 이는 탭 데이터에서 피처 간 상호작용을 조기에 캡처하는 것이 중요하기 때문이다.
    -   반면, 기존 DL 모델들을 이용한 Intermediate Fusion은 오히려 Early Fusion보다 성능이 낮은 경향을 보였다. 이는 일반적인 중간 융합 구조만으로는 탭 데이터의 특성을 충분히 활용하기 어려움을 시사한다.
4.  **시나리오별 분석**: MARIA의 강점은 특히 특정 모달리티가 통째로 누락되는 'Missing Modalities' 시나리오에서 가장 두드러지게 나타났다.

## 🧠 Insights & Discussion

### 강점
MARIA는 데이터 대체(Imputation)라는 전처리 단계 없이 모델 아키텍처 자체에서 결측치를 처리함으로써, 대체 기법으로 인해 발생할 수 있는 인위적인 편향을 제거했다. 또한, Transformer의 어텐션 구조를 수정하여 결측치를 '무시'하는 방식을 택함으로써, 가용한 정보의 가치를 극대화했다는 점이 높게 평가된다.

### 한계 및 비판적 해석
1.  **계산 복잡도**: Modified MSA와 중간 융합 구조는 계산 자원을 많이 소모한다. 이는 자원이 제한된 의료 현장이나 초거대 데이터셋에 적용할 때 확장성(Scalability) 문제가 발생할 수 있다.
2.  **데이터 타입의 제한**: 본 연구는 탭 데이터(Tabular data)에 집중하였다. 의료 데이터의 핵심인 의료 영상(Imaging)이나 텍스트(Clinical notes)와 같은 비정형 데이터와의 결합 가능성은 아직 검증되지 않았다.
3.  **중간 융합의 효용성**: 실험 결과에서 나타나듯, 단순히 구조만 Intermediate Fusion으로 가져간다고 성능이 오르는 것이 아니라, MARIA와 같은 '특수하게 설계된 마스킹 어텐션'이 결합되어야만 의미가 있음이 드러났다. 이는 탭 데이터에서는 단순한 Early Fusion이 효율적일 수 있다는 점을 시사하며, 복잡한 아키텍처 도입의 비용-편익 분석이 필요함을 보여준다.

## 📌 TL;DR

본 논문은 헬스케어 데이터의 고질적인 문제인 결측치를 해결하기 위해, **데이터 대체 없이 가용한 정보만을 선택적으로 처리하는 Transformer 기반 모델 MARIA**를 제안한다. Modified Masked Self-Attention과 Intermediate Fusion, 그리고 강건한 Dropout 전략을 통해 모달리티나 피처가 심하게 누락된 상황에서도 기존 SOTA 모델들을 압도하는 성능을 보였다. 특히 데이터 보간으로 인한 편향을 제거했다는 점에서 실제 임상 적용 가능성이 높으며, 향후 비정형 데이터로의 확장 연구가 기대된다.