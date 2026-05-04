# Learning Abstract Task Representations

Mikhail M. Meskhi, Adriano Rivolli, Rafael G. Mantovani, Ricardo Vilalta (2021)

## 🧩 Problem to Solve

본 논문은 Meta-learning(MtL)에서 데이터 세트나 태스크의 특성을 정의하는 **Data Characterization** 과정의 한계를 해결하고자 한다. Meta-learning의 핵심 목표는 새로운 태스크에 대해 학습 알고리즘을 자동으로 선택하거나 하이퍼파라미터를 튜닝하는 self-adaptive 시스템을 구축하는 것이며, 이를 위해 태스크의 속성을 나타내는 **Meta-features**가 사용된다.

그러나 기존의 전통적인 Meta-features 방식은 다음과 같은 세 가지 주요 문제점을 가진다:

1. **표현력의 한계**: 통계적 Meta-features는 직관적이지 않으며, 서로 다른 데이터 분포를 가진 데이터 세트가 동일한 통계적 특성을 공유할 수 있어 표현력이 부족하다.
2. **계산 비용**: 특히 Topological Meta-features와 같은 복잡도 측정 지표는 대규모 데이터 세트에서 계산 비용이 매우 높다.
3. **임의적 선택**: 어떤 Meta-feature를 사용할지가 도메인 지식에 의존하는 ad hoc한 과정으로 결정된다.

따라서 본 연구의 목표는 전통적인 Meta-features로부터 고수준의 태스크 속성을 포착할 수 있는 **Abstract Meta-features**를 학습하여, 태스크의 일반화 성능 예측(Generalization Performance Estimation) 능력을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deep Neural Network(DNN)를 사용하여 전통적인 Meta-features를 추상적인 잠재 변수(Latent Variables)로 변환하는 것**이다.

단순히 차원을 축소하는 것이 아니라, 특정 타겟(여기서는 알고리즘의 성능 지표)을 예측하도록 네트워크를 학습시킴으로써, 네트워크의 은닉층이 데이터의 핵심적인 특성을 포착하도록 유도한다. 학습된 DNN의 마지막 은닉층에서 추출된 벡터를 **Abstract Meta-features**로 정의하며, 이를 통해 수동으로 설계된 특징의 한계를 극복하고 비선형적인 태스크 표현력을 확보하고자 한다.

## 📎 Related Works

기존의 데이터 특성 추출 방식은 크게 다음과 같이 분류된다:

- **전통적 Meta-features**: 통계적(Statistical), 정보 이론적(Information-theoretic), 모델 기반(Model-based), 랜드마킹(Landmarking) 및 복잡도(Complexity) 기반 특징들이 있다. 이들은 구현이 명확하지만, 앞서 언급한 계산 비용과 표현력 문제가 존재한다.
- **차원 축소 기법**: PCA(Principal Component Analysis)와 같은 선형 결합 방식을 통해 Meta-feature의 차원을 줄여 요약하는 시도가 있었다. 하지만 PCA는 메타 타겟(Meta-target)을 고려하지 않고 분산만을 기준으로 하기 때문에 예측 성능 향상에는 한계가 있다.
- **딥러닝 기반 접근**: 최근 Task2Vec과 같이 태스크를 임베딩(Embedding)으로 표현하려는 시도가 있으나, 본 논문은 기존의 풍부한 전통적 Meta-features를 입력값으로 하여 이를 추상화하는 중간 단계의 접근 방식을 제안한다.

## 🛠️ Methodology

### 전체 파이프라인

본 연구에서 제안하는 **AbstractNet**은 전통적인 Meta-features를 입력으로 받아 해당 태스크에서의 알고리즘 성능을 예측하도록 학습되며, 학습 완료 후 이 네트워크를 **Feature Extractor**로 활용한다.

### AbstractNet 아키텍처

- **구조**: 총 5개의 Fully Connected Layer로 구성된다.
- **레이어 구성**: 처음 4개의 은닉층은 각각 64개의 뉴런을 가지며, 마지막 5번째 층은 16개의 뉴런을 가진 **Latent Layer**로 구성된다.
- **활성화 함수**: 각 층 사이에는 비선형성을 부여하기 위해 $\text{ReLU}$ 함수를 사용한다.
$$\phi(q) = \max(0, q)$$
- **정규화**: 과적합 방지를 위해 두 번째와 네 번째 레이어 이후에 각각 $p=0.1, 0.05$ 확률의 Dropout을 적용한다.
- **학습 목표**: 세 가지 학습 알고리즘(SVM, RF, MLP)의 일반화 성능(AUC)을 예측하는 3차원 출력을 생성한다.

### 손실 함수 (Loss Function)

이상치(Outlier)에 덜 민감하고 기울기 폭주(Exploding Gradient)를 방지하기 위해 **Smooth $L_1$ Loss**를 사용한다.
$$J(\hat{y}, y) = \frac{1}{n} \sum_{i} \nu(\hat{y}_i - y_i)$$
여기서 $\nu(u)$는 다음과 같이 정의된다:
$$\nu(u) = \begin{cases} (0.5 \cdot u^2) / \lambda & \text{if } |u| < \lambda \\ |u| - 0.5 \cdot \lambda & \text{otherwise} \end{cases}$$
$\lambda$는 스텝 함수를 정의하는 임계값이다.

### Abstract Meta-features 추출 절차

1. 전통적인 Meta-features 공간 $\mathcal{F}_t$를 입력으로 하고 성능 지표를 타겟으로 하여 AbstractNet을 학습시킨다.
2. 학습이 완료된 후, 검증 데이터 세트를 네트워크에 통과시킨다.
3. 마지막 은닉층(16개 뉴런)에서 출력되는 잠재 변수 $\{z_i\}$를 추출한다. 이 $\{z_i\}$가 바로 **Abstract Meta-features** $\mathcal{F}_a$가 된다.

## 📊 Results

### 실험 설정

- **데이터셋**: OpenML에서 수집한 517개의 분류 데이터 세트.
- **기준 알고리즘**: SVM, RF, MLP (성능 지표는 AUC 사용).
- **Meta-inducers (메타 모델)**: Decision Trees(DT), Random Forest(RF), SVM.
- **비교 대상**:
  - **Abstract**: 제안 방법 (16개 특징)
  - **Traditional**: 전통적 특징만 사용 (154개 특징)
  - **Hybrid**: 전통적 특징 + 추상적 특징 (170개 특징)
  - **PCA**: PCA로 변환된 특징 (60개 특징)
- **평가 지표**: $\text{RMSE}$ (Root Mean Squared Error), $R^2$ (결정계수).

### 주요 결과

1. **예측 성능 향상**: Abstract Meta-features를 사용했을 때 모든 설정에서 가장 낮은 $\text{RMSE}$와 가장 높은 $R^2$를 기록하였다. 특히 Decision Tree 기반 메타 모델에서 전통적 방식 대비 평균적으로 약 18%의 성능 향상이 관찰되었다.
2. **차원 효율성**: 전통적 방식은 154개의 특징을 사용했지만, Abstract 방식은 단 16개의 특징만으로 훨씬 뛰어난 성능을 보였다. 이는 추상화된 특징이 훨씬 높은 일반화 능력을 가짐을 시사한다.
3. **특징 중요도**: Hybrid 설정에서 Gini Index를 통해 중요도를 분석한 결과, 상위 15개 중요 특징 중 7개가 학습된 Abstract Meta-features였다. 이는 추상 특징이 전통적 특징보다 개별적으로 더 높은 변별력을 가짐을 의미한다.
4. **통계적 유의성**: Hierarchical Bayesian t-test 결과, Abstract 방식이 Traditional 및 PCA 방식보다 유의미하게 우수함이 입증되었다 (Prob $\approx 1.000$).

## 🧠 Insights & Discussion

본 논문은 딥러닝의 잠재 표현 학습(Latent Representation Learning) 능력을 Meta-learning의 데이터 특성 추출 단계에 성공적으로 적용하였다.

**강점 및 해석**:

- **비선형성 확보**: PCA가 낮은 성능을 보인 이유는 단순 선형 결합만으로는 메타 특징과 타겟 성능 사이의 복잡한 관계를 포착할 수 없기 때문이다. 반면, AbstractNet은 $\text{ReLU}$와 다층 구조를 통해 고도로 비선형적인 추상화를 수행하였다.
- **정보의 압축**: 154차원의 공간을 16차원으로 압축했음에도 성능이 향상된 것은, DNN이 불필요한 노이즈를 제거하고 타겟 예측에 가장 결정적인 '핵심 속성'만을 추출했음을 보여준다.

**한계 및 미해결 질문**:

- **데이터 의존성**: 본 연구는 517개의 데이터 세트를 사용하였으나, DNN의 특성상 더 방대한 메타 데이터 세트가 있을 때 성능이 더욱 향상될 가능성이 있다.
- **해석 가능성**: Abstract Meta-features는 잠재 변수 형태이므로, 이 16개의 변수가 구체적으로 어떤 물리적/통계적 의미(예: 데이터의 희소성, 클래스 불균형 등)를 가지는지에 대한 분석은 부족하다.

## 📌 TL;DR

본 논문은 전통적인 Meta-features의 한계(높은 계산 비용, 낮은 표현력)를 극복하기 위해, **DNN을 이용하여 전통적 특징을 고차원적으로 추상화한 'Abstract Meta-features'를 학습하는 방법**을 제안한다. 실험 결과, 단 16개의 추상 특징만으로도 기존의 수백 개 전통 특징이나 PCA 변환 특징보다 훨씬 정확하게 알고리즘의 일반화 성능을 예측할 수 있음을 보였다. 이 연구는 향후 효율적인 태스크 표현 학습 및 자동화된 머신러닝(AutoML) 파이프라인 구축에 중요한 기여를 할 것으로 기대된다.
