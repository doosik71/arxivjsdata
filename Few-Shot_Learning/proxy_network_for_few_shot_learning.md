# Proxy Network for Few Shot Learning

Bin Xiao, Chien-Liang Liu, Wen-Hoar Hsaio (2020)

## 🧩 Problem to Solve

본 논문은 매우 적은 수의 학습 데이터만으로 새로운 클래스(novel classes)를 인식해야 하는 Few-Shot Learning (FSL) 문제를 해결하고자 한다. 일반적인 딥러닝 모델은 높은 성능을 내기 위해 클래스당 수천 개의 레이블링된 데이터가 필요하지만, 이는 소수의 사례만으로도 새로운 사물을 인식하는 인간의 학습 방식과 대조된다.

특히, 기존의 Metric-learning 기반 접근 방식들은 임베딩 공간에서 같은 클래스의 데이터는 가깝게, 서로 다른 클래스의 데이터는 멀게 배치하는 것을 목표로 한다. 저자들은 이러한 Metric-learning의 성공 여부가 다음의 세 가지 핵심 요소에 달려 있다고 분석한다:

1. 데이터 임베딩(Data embedding)
2. 클래스 대표값(Representative of each class)
3. 거리 측정 지표(Distance metric)

따라서 본 논문의 목표는 위 세 가지 구성 요소를 고정된 함수나 단순 연산이 아닌, 데이터로부터 직접 학습하는 엔드-투-엔드(end-to-end) 모델인 ProxyNet을 제안하여 FSL 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **임베딩, 클래스 대표값(Proxy), 그리고 거리 측정 지표를 모두 데이터로부터 동시에 학습**하게 만드는 것이다. 구체적인 기여점은 다음과 같다.

- **학습 가능한 클래스 대표값(Learnable Class Proxy):** 기존의 Prototypical Networks가 단순히 평균(mean)을 사용했던 것과 달리, 데이터의 특성과 전역적 경향을 고려하여 가중치를 학습하는 방식을 제안함으로써 이상치(outlier)의 영향을 줄였다.
- **학습 가능한 거리 지표(Learnable Distance Metric):** 유클리드 거리나 코사인 유사도 같은 고정된 지표 대신, 3D Convolutional Network를 통해 쿼리 임베딩과 클래스 프록시 사이의 관계를 학습하도록 설계하였다.
- **효율적인 아키텍처:** 모델의 복잡도를 낮추면서도 최신 기법들(SOTA)보다 우수한 성능을 보였으며, 특히 파라미터 수 대비 효율성이 높음을 입증하였다.

## 📎 Related Works

본 논문은 FSL 접근 방식을 크게 세 가지로 분류하여 설명한다.

1. **Metric-based approaches:** 임베딩 공간 내 거리 측정을 통해 분류하는 방식이다.
    - **Matching Networks:** 코사인 유사도를 사용하여 서포트 셋과 쿼리 셋을 비교한다.
    - **Prototypical Networks:** 클래스의 평균 임베딩(Prototype)을 생성하고 유클리드 거리를 측정한다.
    - **Relation Networks:** 학습 가능한 Relation module을 통해 거리 지표를 생성한다.
2. **Optimization-based methods:** 모델 파라미터의 좋은 초기값을 찾거나 업데이트 규칙을 학습하는 방식이다. (예: MAML, Reptile)
3. **Augmentation-based methods:** 가상의 데이터를 생성하여 데이터 부족 문제를 해결하려는 방식이다.

**기존 방식과의 차별점:**
기존의 Metric-based 방식들은 거리 지표를 고정하거나(MatchingNet, ProtoNet), 클래스 대표값을 단순 평균/합산으로 구하는 한계가 있었다. ProxyNet은 이 모든 과정을 신경망을 통해 학습함으로써 데이터 기반의 유연한 최적화를 가능하게 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

ProxyNet은 메타 러닝(Meta-learning) 프레임워크를 따르며, **임베딩 $\rightarrow$ 클래스 프록시 생성 $\rightarrow$ 거리 측정 및 분류**의 단계로 구성된다.

### 2. 주요 구성 요소 및 역할

#### (1) 임베딩 함수 (Embedding Function)

서포트 셋과 쿼리 셋을 동일한 임베딩 공간으로 매핑하기 위해 동일한 함수 $f_\theta$를 사용한다. 기본적으로 Conv-4 네트워크를 사용하였으나, 실험을 통해 ResNet 등 더 깊은 네트워크로 확장 가능하다.

#### (2) 학습 가능한 클래스 프록시 (Learnable Class Proxy)

$K$-shot- $N$-way 작업에서 $n$번째 클래스의 서포트 샘플들을 $\{x_1^n, \dots, x_K^n\}$이라고 할 때, 클래스 대표값 $\mu_n$을 다음과 같이 가중 합(weighted sum)으로 정의한다.

$$\mu_n = h_\theta(x_1^n, \dots, x_K^n) = \sum_{k=1}^K w_{nk} x_k^n$$

여기서 가중치 $w_{nk}$는 단순 상수가 아니라, 각 샘플 $x_k^n$과 해당 클래스의 전체 합 $s_n$을 입력으로 받는 신경망을 통해 학습된다. 이는 특정 샘플이 클래스의 전역적 경향에서 얼마나 벗어나 있는지를 판단하여, 이상치에는 낮은 가중치를 부여하고 대표성 있는 샘플에는 높은 가중치를 부여하기 위함이다.

#### (3) 거리 측정 모듈 (Distance Metric Proxy)

쿼리 임베딩과 각 클래스 프록시 $\mu_n$ 사이의 유사도를 측정하기 위해 3D Convolutional Network를 사용한다.

- **입력:** 쿼리 임베딩과 각 클래스 프록시 임베딩을 새로운 차원으로 쌓아(stacking) 4차원 텐서 형태로 구성한다.
- **구조:** 두 개의 3D Convolution 블록(3D Conv $\rightarrow$ 3D Batch Norm $\rightarrow$ ReLU)을 거친 후, Global Average Pooling (GAP)을 통해 각 클래스에 대한 할당 확률을 계산한다.
- **목적:** 3D Conv를 통해 채널 간의 관계와 공간적 특성을 동시에 학습하며, GAP를 사용하여 모델 복잡도를 줄이고 과적합(overfitting)을 방지한다.

### 3. 학습 절차

모델은 엔드-투-엔드(end-to-end) 방식으로 학습되며, 최종 출력층의 Softmax 결과와 실제 레이블 간의 Cross-Entropy Loss를 최소화하는 방향으로 최적화된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** CUB (세밀한 분류), mini-ImageNet (일반 분류)
- **시나리오:** 1-shot-5-way 및 5-shot-5-way
- **비교 대상:** MatchingNet, ProtoNet, RelationNet, CTM, DN4, Baseline++
- **지표:** Accuracy (정확도) 및 $95\%$ 신뢰 구간

### 2. 정량적 결과

Table 1에 따르면, ProxyNet은 대부분의 시나리오에서 기존 SOTA 모델들보다 우수한 성능을 보였다.

- **CUB 데이터셋:** 1-shot-5-way에서 $67.52\%$의 정확도를 기록하여 Baseline++($60.53\%$)나 ProtoNet($50.46\%$)을 크게 상회하였다.
- **mini-ImageNet 데이터셋:** 1-shot-5-way에서 $52.95\%$, 5-shot-5-way에서 $71.02\%$를 기록하며 타 모델 대비 경쟁 우위를 보였다.
- **파라미터 효율성:** Table 2에서 ProxyNet의 학습 가능 파라미터 수는 $165,171$개로, CTM($307,921$개)이나 Baseline++($433,288$개)보다 훨씬 적음에도 불구하고 더 높은 성능을 달성하였다.

## 🧠 Insights & Discussion

### 1. 임베딩 네트워크의 깊이와 과적합

저자들은 임베딩 네트워크의 깊이가 성능에 미치는 영향을 분석하였다.

- **데이터 증강(Data Augmentation) 적용 시:** ResNet-10, 18, 34와 같은 깊은 네트워크를 사용할수록 성능이 향상되는 경향을 보였다.
- **데이터 증강 미적용 시:** 오히려 Conv-4와 같은 단순한 네트워크가 더 좋은 성능을 냈다. 이는 ProxyNet이 임베딩, 프록시, 거리 지표를 모두 학습하는 유연한 구조이기 때문에, 데이터가 부족한 상황에서 깊은 네트워크를 사용하면 과적합(overfitting)이 발생하기 쉽다는 것을 시사한다.

### 2. 클래스 프록시의 유효성

실험을 통해 단순 평균(Mean)이나 합산(Sum)을 사용하는 것보다 학습 가능한 가중치 기반의 프록시를 사용하는 것이 성능이 더 높음을 확인하였다. 이는 고차원 공간에서 단순한 유클리드 거리 기반의 중심점 찾기가 항상 유의미하지 않으며, 데이터 기반의 대표값 학습이 더 효과적임을 보여준다.

### 3. 거리 지표의 설계

3D Convolution과 GAP를 결합한 거리 측정 방식이 유클리드 거리나 단순 FC 레이어 기반의 RelationNet보다 우수했다. 이는 채널 간의 관계를 보존하면서 공간적 특성을 학습하는 구조가 FSL의 분류 작업에 더 적합함을 의미한다.

## 📌 TL;DR

본 논문은 Few-Shot Learning의 핵심 요소인 **임베딩, 클래스 대표값(Proxy), 거리 지표**를 모두 고정하지 않고 **데이터로부터 직접 학습**하는 **ProxyNet**을 제안하였다. 특히 이상치에 강건한 가중치 기반 클래스 프록시와 3D Conv 기반의 거리 측정 모듈을 통해, 적은 파라미터만으로도 CUB 및 mini-ImageNet 데이터셋에서 기존 SOTA 모델들을 뛰어넘는 성능을 입증하였다. 이 연구는 FSL에서 단순한 연산보다 데이터 기반의 학습 가능한 모듈이 더 강력한 일반화 성능을 제공할 수 있음을 보여주며, 향후 크로스 도메인 FSL 연구로 확장될 가능성을 제시한다.
