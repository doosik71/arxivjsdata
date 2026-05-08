# ONE-CLASS META-LEARNING: TOWARDS GENERALIZABLE FEW-SHOT OPEN-SET CLASSIFICATION

Jedrzej Kozerawski, Matthew Turk (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Few-Shot Open-Set Classification (FSOSC)**이다. 일반적인 Few-Shot Learning은 테스트 단계의 쿼리 집합($Q$)이 학습 단계의 서포트 집합($S$)에 포함된 클래스들로만 구성된다고 가정하는 Closed-Set 설정에서 작동한다. 그러나 실제 환경에서는 학습 시 보지 못한 새로운 클래스(Unknown categories)가 포함된 Open-Set 상황이 빈번하게 발생한다.

이 문제의 중요성은 기존의 Open-Set 인식 방법들이 각 클래스당 다량의 데이터를 필요로 하여 클래스 내 분포를 모델링하고 꼬리 부분(tail)을 분석하는 방식(예: Extreme Value Theory)을 사용한다는 점에 있다. 하지만 Few-Shot 설정에서는 클래스당 샘플 수가 매우 적기 때문에(예: 1-shot), 기존의 Open-Set 방법론들을 그대로 적용하는 것이 불가능하거나 성능이 매우 낮다.

따라서 본 논문의 목표는 매우 적은 수의 학습 샘플만으로도 알려진 클래스(Known)와 알려지지 않은 클래스(Unknown)를 효과적으로 구분할 수 있는 일반화 가능한 Meta-Learning 기반의 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'분할 정복(Divide-and-Conquer)'** 접근 방식이다. 다중 클래스 Open-Set 문제를 $n$개의 독립적인 **One-Class Classification** 문제로 분해하여 해결하는 것이다. 즉, "이 샘플이 $n$개의 클래스 중 어디에 속하는가"를 묻는 대신, 각 클래스에 대해 "이 샘플이 해당 클래스에 속하는가"를 개별적으로 판단하고 이를 앙상블하는 방식이다.

이를 위해 저자들은 다음 두 가지 독립적인 Few-Shot One-Class 분류 방법을 제안한다.

1. **Meta Binary Cross-Entropy (Meta-BCE)**: One-Class 분류만을 위한 별도의 특징 표현(feature representation) 공간을 학습한다.
2. **One-Class Meta-Learning (OCML)**: 표준 다중 클래스 특징 표현을 입력받아, 해당 클래스에 최적화된 One-Class 분류기를 동적으로 생성하는 전이 학습 모듈을 학습한다.

이 두 방법론의 가장 큰 장점은 기존의 어떤 Closed-Set Few-Shot 모델(예: FEAT, PEELER)과도 결합 가능하며, 기존 모델을 재학습시킬 필요 없이 Open-Set 기능을 추가할 수 있다는 점이다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들의 한계를 지적한다.

- **Few-Shot Classification**: Prototypical Networks, Matching Networks 등은 대부분 Closed-Set을 가정하며, 최근 제안된 Few-Shot Open-Set 방법(Liu et al.)은 상대적인 랭킹 점수만을 제공할 뿐, 알려진 클래스와 알려지지 않은 클래스를 명확히 구분하는 기준을 제시하지 못하며 $n=1$인 상황(One-Class)을 처리할 수 없다.
- **One-Class Classification**: OC-SVM, Deep-SVDD 등은 긍정 클래스의 분포를 모델링하기 위해 많은 양의 데이터가 필요하므로, 샘플 수가 극히 적은 Few-Shot 설정에서는 과적합(Overfitting) 문제가 발생한다.
- **Open-Set Classification**: OpenMax나 Entropic Open-Set Loss 등은 소프트맥스(Softmax) 점수에 의존하므로 클래스가 하나뿐인 상황에서는 작동하지 않으며, 학습 단계에서 배경(Background/Unknown) 클래스 데이터가 필요하다는 제약이 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

제안된 방법론은 기존의 Feature Extractor $f_\phi$를 그대로 사용하면서, 그 위에 One-Class 판별을 위한 모듈을 추가하는 형태이다. 이 모듈들은 독립적으로 학습되어 기존 모델의 Closed-Set 성능을 저하시키지 않고 Open-Set 인식 능력을 부여한다.

### 2. Meta Binary Cross-Entropy (Meta-BCE)

Meta-BCE는 다중 클래스 분류를 위한 특징 공간과 One-Class 분류를 위한 특징 공간이 서로 달라야 한다는 가설에서 출발한다. 따라서 별도의 특징 추출 브랜치 $f'_\phi$를 구성하여 One-Class 분류 전용 표현을 학습한다.

- **학습 목표**: Binary Cross-Entropy (BCE) 손실 함수를 사용하여 샘플이 특정 클래스에 속하는지 여부를 이진 분류하도록 학습한다.
- **손실 함수**:
$$L_{\text{Meta-BCE}} = -\frac{1}{N} \sum_{(x_i, y_i) \in D_{\text{meta-train}}} \sum_{c=1}^{N} (y_{i,c} \log p(y_c | x_i) + (1-y_{i,c})(1-p(y_c | x_i)))$$
- **판별 확률**:
$$p(y=c | x_i) = \frac{1}{1 + \exp(d(f'_\phi(x_i), \bar{x}_c) + t)}$$
여기서 $d(\cdot, \cdot)$는 거리 함수이며, $\bar{x}_c$는 클래스 $c$의 프로토타입, $t$는 학습 가능한 파라미터(임계값)이다.

### 3. One-Class Meta-Learning (OCML)

OCML은 적은 샘플로도 클래스에 적응할 수 있도록, 클래스 표현 $\bar{x}_c$를 입력받아 해당 클래스의 분류기 가중치 $w_c$를 생성하는 전이 학습 모듈 $g_\theta$를 학습한다.

- **가중치 생성**:
$$w_c = g_\theta(f_\phi(x_c))$$
- **판별 확률**:
$$p(y=c | x_i) = \frac{1}{1 + \exp(-w_c \cdot f_\phi(x_i))}$$
여기서 $w_c \cdot f_\phi(x_i)$는 가중치 벡터와 특징 벡터의 내적(dot product)이다. OCML은 $f_\phi$와 $g_\theta$를 동시에 학습시킨다.

### 4. Few-Shot Open-Set (Multiclass) 확장

One-Class 모듈들을 다중 클래스 환경($n \ge 2$)으로 확장하기 위해, 모든 알려진 클래스 $c \in n$에 대해 "알려지지 않은 클래스($U$)일 확률"을 다음과 같이 정의한다. 즉, 모든 알려진 클래스에 속할 확률이 낮을수록 Unknown일 확률이 높다고 판단하는 방식이다.

- **Meta-BCE의 Open-Set 판별**:
$$p(y \in U | x_i) = \max_{c \in n} (1 - p(y=c | x_i)) = \min_{c \in n} \left( \frac{1}{1 + \exp(d(f'_\phi(x_i), \bar{x}_c) + t)} \right)$$
- **OCML의 Open-Set 판별**:
$$p(y \in U | x_i) = \max_{c \in n} (1 - p(y=c | x_i)) = \min_{c \in n} \left( \frac{1}{1 + \exp(-g_\theta(f_\phi(\bar{x}_c)) \cdot f_\phi(x_i))} \right)$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: miniImageNet, CUB-200-2011, tieredImageNet.
- **비교 대상**: DeepSVDD, OC-SVM, PEELER, FEAT 및 기존 Open-Set 방법론(OpenMax, Entropic Loss 등).
- **지표**: Accuracy, F1-score, AUROC, Normalized Accuracy (NA), F1-open.

### 2. 주요 결과

- **Few-Shot One-Class 성능**: miniImageNet 데이터셋 기준, 1-shot에서는 OCML이 가장 높은 정확도($68.05\%$)를 보였으며, 5-shot에서는 Meta-BCE가 가장 높은 정확도($77.38\%$)를 기록하였다. 기존의 Many-shot One-Class 방법들(DeepSVDD 등)은 Few-Shot 설정에서 제대로 작동하지 않음을 확인하였다.
- **Few-Shot Open-Set 성능**:
  - **NA (Normalized Accuracy)**: ResNet-18 기반 FEAT 모델에 결합했을 때, 1-shot에서는 OCML($59.79\%$), 5-shot에서는 Meta-BCE($73.32\%$)가 SOTA 성능을 달성하였다.
  - **Closed-Set 영향**: 제안된 방법들은 별도의 모듈로 작동하므로, 기존 Closed-Set 분류 성능을 전혀 떨어뜨리지 않으면서 Open-Set 인식 능력을 향상시켰다. 이는 Entropic Loss나 Objectosphere Loss 같은 방법들이 Closed-Set 정확도를 낮추는 것과 대조적이다.
  - **데이터 효율성**: 학습 시 별도의 Background 클래스가 필요 없다는 점이 기존 Open-Set 방법론 대비 강력한 강점으로 나타났다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **특징 공간의 분리**: Meta-BCE 실험(Meta-BCE vs $\text{Meta-BCE}^C$)을 통해, 다중 클래스 분류를 위한 특징 표현과 One-Class 판별을 위한 특징 표현이 서로 달라야 함을 입증하였다. 별도의 브랜치($f'_\phi$)를 사용했을 때 NA와 F1-open 점수가 현저히 높았다.
- **K-shot에 따른 방법론 선택**: OCML은 샘플 수가 극히 적은(1-shot) 상황에서 강점을 보이며, Meta-BCE는 샘플 수가 어느 정도 확보된(5-shot 이상) 상황에서 성능 향상 폭이 매우 크다.

### 2. 한계 및 논의사항

- **Upper-bound와의 간극**: 지도 학습 기반의 상한선(Supervised Upper-bound)과 비교했을 때 여전히 성능 차이가 존재하며, 이는 향후 개선의 여지가 있음을 시사한다.
- **가정**: 본 모델은 One-Class 분류기들의 앙상블을 통해 Open-Set을 판별하는데, 이는 각 클래스의 독립적인 경계가 명확하다는 가정을 내포하고 있다.

## 📌 TL;DR

본 논문은 Few-Shot 설정에서 알려지지 않은 클래스를 탐지하기 위해 **다중 클래스 Open-Set 문제를 $n$개의 One-Class 문제로 분해하여 해결하는 Meta-Learning 프레임워크**를 제안한다. **Meta-BCE**와 **OCML**이라는 두 가지 모듈을 통해 기존의 어떤 Few-Shot 모델과도 결합 가능하며, 특히 1-shot에서는 OCML이, 5-shot에서는 Meta-BCE가 탁월한 성능을 보인다. 이 연구는 데이터가 극소수인 환경에서도 재학습 없이 Open-Set 인식 기능을 추가할 수 있는 실용적인 방법을 제시했다는 점에서 향후 자율주행이나 희귀 질병 진단 등 실시간 적응이 필요한 분야에 중요하게 적용될 가능성이 높다.
