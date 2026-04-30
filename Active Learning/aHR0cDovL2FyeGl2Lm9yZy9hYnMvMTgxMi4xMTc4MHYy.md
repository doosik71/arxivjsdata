# Weakly Supervised Active Learning with Cluster Annotation

Fábio Perez, Rémi Lebret, Karl Aberer (2019)

## 🧩 Problem to Solve

딥러닝 모델을 학습시키기 위해서는 막대한 양의 레이블링된 데이터가 필요하지만, 모든 데이터를 사람이 직접 레이블링하는 것은 비용과 시간 측면에서 매우 비효율적이다. 이를 해결하기 위해 Active Learning(능동 학습) 기법이 사용되며, 이는 모델이 가장 정보 가치가 높다고 판단하는 샘플만을 선택적으로 레이블링하여 효율성을 높이는 방식이다.

그러나 기존의 Active Learning은 여전히 샘플을 개별적으로(individually) 레이블링해야 한다는 한계가 있다. 따라서 본 논문의 목표는 개별 샘플이 아닌 샘플들의 집합인 Cluster(군집) 단위로 레이블을 부여하는 **Cluster Annotation** 방식을 도입하여, 인간의 상호작용(human interaction) 횟수를 획기적으로 줄이면서도 모델의 성능을 유지하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Semi-supervised Learning과 Weakly Supervised Learning을 Active Learning 프레임워크에 결합하는 것이다. 

전문가가 개별 이미지 하나하나를 확인하는 대신, 유사한 특성을 가진 샘플들의 군집을 확인하고 해당 군집이 하나의 클래스로 일관성(class consistency)을 가진다고 판단되면 군집 전체에 동일한 레이블을 부여하게 한다. 이를 통해 한 번의 상호작용으로 다수의 샘플을 레이블링할 수 있어, 전체적인 레이블링 비용을 크게 낮출 수 있다.

## 📎 Related Works

기존의 Convolutional Neural Networks(CNNs)를 위한 Active Learning 연구들은 주로 Uncertainty(불확실성) 측정에 기반한다.
- **Uncertainty-based methods**: Softmax 계층의 Maximum Entropy를 이용하거나, Monte Carlo Dropout을 통해 Bayesian 불확실성을 측정하는 방법, 또는 여러 모델의 예측값을 이용하는 Ensemble 방법 등이 있다. 하지만 불확실성 측정 자체가 어렵거나 계산 비용이 높다는 단점이 있다.
- **Geometric approach**: Sener와 Savarese는 Core-set이라는 기법을 통해 데이터의 기하학적 분포를 고려하여 샘플을 선택하는 방식을 제안하였다.
- **Clustering-based AL**: Berardo 등은 MNIST 데이터셋에 대해 특징 추출 후 군집화를 수행하고, 각 군집의 중심점에 가장 가까운 샘플만 레이블링하여 군집 전체에 확장하는 방식을 사용하였다.

본 연구는 기하학적 정보(Clustering)를 사용한다는 점에서 Core-set 방식과 유사하지만, 목적이 '가장 정보가 많은 샘플을 찾는 것'이 아니라 '한 번에 최대한 많은 데이터를 레이블링하는 것'에 있다는 점에서 차별점이 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
본 프레임워크는 표준적인 Pool-based Active Learning 구조에 **Data Clustering**과 **Cluster Annotation** 단계를 추가한 형태이다. 전체 프로세스는 다음과 같은 흐름으로 진행된다.

1.  **특징 추출(Feature Extraction)**: ImageNet으로 사전 학습된 ResNet-18 모델을 사용하며, Average Pooling 계층에서 512차원의 특징 벡터를 추출한다.
2.  **군집화(Clustering)**: 추출된 특징 벡터를 기반으로 Faiss 라이브러리의 k-means 알고리즘을 사용하여 unlabeled pool 내의 샘플들을 여러 개의 군집으로 나눈다.
3.  **레이블링(Annotation)**:
    -   **Individual Annotation**: 불확실성이 높은 개별 샘플을 선택하여 레이블링한다.
    -   **Cluster Annotation**: 군집 내 샘플들을 전문가에게 보여주고, 전문가가 해당 군집의 클래스 일관성이 높다고 판단하면 군집 전체에 레이블을 부여한다.

### 2. 상세 학습 절차 및 알고리즘
학습은 여러 번의 반복(Iteration)을 통해 이루어지며, 각 반복마다 모델을 초기화하고 새로운 데이터를 추가하여 학습시킨다.

**군집 레이블링의 시뮬레이션 조건:**
실제 전문가의 판단을 시뮬레이션하기 위해, 군집 내에서 최빈값(modal class)을 가진 클래스의 비율이 $80\%$ 이상일 때만 해당 군집에 레이블을 부여하고, 그렇지 않으면 레이블링하지 않은 채 unlabeled pool에 남겨둔다.

**레이블링 전략(Scenarios):**
- `random`: 무작위 샘플 선택
- `uncertain-only`: Maximum Entropy 기반의 불확실한 샘플만 선택
- `cluster-only`: 군집 단위 레이블링만 수행
- `uncertain+cluster`: 불확실한 샘플을 먼저 레이블링한 후, 남은 데이터로 군집 레이블링 수행
- `cluster+uncertain`: 군집 레이블링을 먼저 수행한 후, 불확실한 샘플을 레이블링 수행

### 3. 주요 방정식 및 설정
- **특징 추출**: ResNet-18의 Average Pooling layer output $f \in \mathbb{R}^{512}$
- **군집화 알고리즘**: k-means clustering
- **최적화**: SGD ($\text{learning rate}=1e^{-3}$ for CIFAR-10, $1e^{-4}$ for EuroSAT), $\text{momentum}=0.9$, $\text{weight decay}=5e^{-4}$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: CIFAR-10 (50,000 train / 10,000 test), EuroSAT (27,000 images, 80/20 split)
- **비교 지표**: 인간의 상호작용 횟수(Human Interactions) 대비 테스트 정확도(Test Accuracy). 여기서 개별 샘플 1개 레이블링과 군집 1개 레이블링은 동일하게 1회의 상호작용으로 간주한다.

### 2. 정량적 결과
- **효율성**: 군집 기반 시나리오(`cluster-only`, `uncertain+cluster`, `cluster+uncertain`)가 군집을 사용하지 않는 방식(`random`, `uncertain-only`)보다 정확도가 훨씬 빠르게 상승한다.
- **최적 조합**: `uncertain+cluster` 시나리오가 가장 우수한 성능을 보였다. 특히 불확실한 샘플을 먼저 제거함으로써 이후 군집화 단계에서의 레이블 오류(label error)가 감소하는 'Cluster Cleaner' 효과가 관찰되었다.
- **상호작용 감소량**: Fully-supervised 학습과 유사한 성능을 달성하기 위해 필요한 인간의 상호작용 횟수를 비교했을 때:
    -   **CIFAR-10**: 전체 데이터(50,000개) 대비 약 $18\%$인 9,000회의 상호작용만으로 달성 (82% 감소).
    -   **EuroSAT**: 전체 데이터(21,600개) 대비 약 $13\%$인 2,800회의 상호작용만으로 달성 (87% 감소).

## 🧠 Insights & Discussion

### 1. 강점 및 해석
본 연구는 Active Learning의 병목 현상인 '개별 레이블링 비용'을 군집 단위의 Weak Supervision으로 해결하였다. 특히 `uncertain+cluster` 순서가 효과적이었다는 점은, 모델이 확신하지 못하는 경계선상의 샘플들을 먼저 명확히 레이블링하여 제거하면, 남은 데이터들의 군집 응집도가 높아져 군집 레이블링의 정확도가 상승한다는 것을 시사한다.

### 2. 한계 및 가정
- **레이블 노이즈**: 군집 단위로 레이블을 부여하기 때문에 필연적으로 일부 샘플에 잘못된 레이블이 부여되는 label noise가 발생한다.
- **시각화 의존성**: 본 방법은 인간이 군집 내 이미지들을 빠르게 훑어보고(skim through) 일관성을 판단할 수 있다는 가정에 기반한다. 따라서 이미지가 아닌 텍스트나 오디오 데이터의 경우, 군집을 어떻게 시각화하여 전문가에게 보여줄 것인가에 대한 추가적인 연구가 필요하다.

### 3. 비판적 해석
논문에서는 `uncertain+cluster`가 `cluster+uncertain`보다 성능이 약간 높다고 언급하며, 이는 레이블 에러율의 차이 때문이라고 분석한다. 하지만 실제 환경에서 전문가가 군집을 레이블링할 때 느끼는 피로도나 주관적 판단의 일관성이 결과에 어떤 영향을 미치는지에 대한 분석은 부족하다. 또한, $80\%$라는 일관성 임계값(threshold)이 실제 인간의 판단 기준과 얼마나 일치하는지에 대한 실증적 근거가 제시되지 않았다.

## 📌 TL;DR

본 논문은 Active Learning 과정에서 개별 샘플이 아닌 **Cluster 단위로 레이블을 부여**함으로써 인간의 상호작용 횟수를 획기적으로 줄이는 프레임워크를 제안한다. 특히 불확실한 샘플을 먼저 레이블링한 후 군집 레이블링을 수행하는 방식이 가장 효율적임을 보였으며, 이를 통해 CIFAR-10과 EuroSAT 데이터셋에서 각각 82%, 87%의 상호작용 비용을 절감하면서도 전체 지도 학습에 근접하는 성능을 달성하였다. 이 연구는 대규모 데이터셋의 레이블링 비용 문제를 해결하려는 실무적인 딥러닝 학습 파이프라인 구축에 중요한 시사점을 제공한다.