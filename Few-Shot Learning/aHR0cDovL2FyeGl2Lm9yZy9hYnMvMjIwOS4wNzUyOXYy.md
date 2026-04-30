# ON THE SOFT-SUBNETWORK FOR FEW-SHOT CLASS INCREMENTAL LEARNING

Haeyong Kang, Jaehong Yoon, Sultan Rizky Madjid, Sung Ju Hwang, and Chang D. Yoo (2023)

## 🧩 Problem to Solve

본 논문은 Few-Shot Class-Incremental Learning (FSCIL)에서 발생하는 두 가지 핵심 문제인 **Catastrophic Forgetting (치명적 망각)**과 **Overfitting (과적합)**을 해결하고자 한다.

FSCIL은 대량의 데이터가 주어진 Base session 이후, 새로운 클래스들이 매우 적은 수의 샘플(Few-shot)과 함께 순차적으로 등장하는 시나리오이다. 이 과정에서 모델은 새로운 클래스를 학습하기 위해 가중치를 업데이트하지만, 이로 인해 기존에 학습했던 클래스의 지식을 잃어버리는 Catastrophic Forgetting이 발생한다. 동시에, 새로운 클래스의 데이터가 극소수이기 때문에 모델이 해당 샘플들에 과하게 최적화되어 일반화 성능이 떨어지는 Overfitting 문제가 심각하게 나타난다. 

따라서 본 논문의 목표는 기존 지식을 보존하면서도 적은 수의 샘플만으로 새로운 클래스를 효과적으로 학습할 수 있는 효율적인 모델 업데이트 메커니즘을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Regularized Lottery Ticket Hypothesis (RLTH)**이다. 이는 무작위로 초기화된 밀집 신경망(Dense Network) 내에, 이전 클래스의 지식을 유지하면서도 새로운 클래스 지식을 학습할 수 있는 공간을 제공하는 '정규화된 서브네트워크(Regularized Subnetwork)'가 존재한다는 가설이다.

이를 구현하기 위해 제안된 **Soft-SubNetworks (SoftNet)**는 다음과 같은 설계 특징을 갖는다:
1. **비이진 소프트 마스크(Non-binary Soft Mask)**: 단순히 가중치를 사용하거나 버리는 이진 마스크(Binary Mask) 대신, $0$과 $1$ 사이의 값을 갖는 소프트 마스크를 도입하여 정규화 효과를 얻는다.
2. **Major 및 Minor 서브네트워크의 분리**: 서브네트워크를 지식 보존을 위한 Major 부분($m=1$)과 새로운 지식 학습 및 과적합 방지를 위한 Minor 부분($m < 1$)으로 나누어 관리한다.

## 📎 Related Works

기존의 Continual Learning 연구들은 다음과 같은 방향으로 발전해 왔다:
- **Constraint-based**: 가중치 변화에 제약을 두어 망각을 방지한다.
- **Memory-based**: 과거 데이터의 일부를 저장(Replay)하여 재학습한다.
- **Architecture-based**: 네트워크 용량을 확장하거나 파라미터를 격리하여 간섭을 줄인다.

특히 FSLL(Few-Shot Lifelong Learning)과 같은 기존 서브네트워크 기반 접근 방식은 세션별로 서브네트워크를 검색하고 가중치를 선택하는 반복적인 과정이 필요하여 계산 비용이 매우 높다는 한계가 있다. 반면, SoftNet은 Base session에서 모델 가중치와 소프트 마스크를 공동으로 학습함으로써 계산 효율성을 높이고, 소프트 마스크 자체를 정규화 도구로 활용하여 FSCIL 특유의 과적합 문제를 해결한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인
SoftNet은 Base session에서 학습된 지식을 고정된 Major 서브네트워크에 저장하고, 이후 등장하는 새로운 세션에서는 가변적인 Minor 서브네트워크만을 업데이트하는 구조를 가진다.

### 주요 구성 요소 및 학습 절차

**1. 소프트 마스크(Soft-mask) 정의**
소프트 마스크 $m_{soft}$는 다음과 같이 Major 마스크와 Minor 마스크의 합으로 정의된다:
$$m_{soft} = m_{major} \oplus m_{minor}$$
여기서 $m_{major}$는 가중치 중요도(Weight scores) 기준 상위 $c\%$에 해당하는 파라미터를 선택하는 이진 마스크이며, $m_{minor}$는 균등 분포 $U(0, 1)$에서 샘플링된 값으로 구성된다.

**2. Base Session 학습 ($t=1$)**
모델 가중치 $\theta$와 가중치 점수 $s$를 공동으로 최적화한다. 이때 이진 마스크의 미분 불가능성을 해결하기 위해 **Straight-through Estimator (STE)**를 사용하여 gradient를 전달한다.
- 업데이트 식: $\theta \leftarrow \theta - \alpha \left( \frac{\partial L}{\partial \theta} \odot m_{soft} \right)$
- 가중치 점수 업데이트: $s \leftarrow s - \alpha \left( \frac{\partial L}{\partial s} \odot m_{soft} \right)$

**3. Incremental Session 학습 ($t \ge 2$)**
Base session에서 학습된 Major 서브네트워크($m_{major}$)의 가중치는 완전히 고정하여 기존 지식을 보존한다. 오직 Minor 서브네트워크($m_{minor}$)에 해당하는 가중치만을 업데이트하여 새로운 클래스를 학습한다.
- 업데이트 식: $\theta \leftarrow \theta - \beta \left( \frac{\partial L}{\partial \theta} \odot m_{minor} \right)$

### 손실 함수 및 추론
본 논문은 Euclidean distance 대신 **Cosine distance**를 기반으로 한 메트릭 분류 알고리즘을 사용하여 과적합을 방지하고 정규화된 측정치를 얻는다. 손실 함수 $L_m$은 다음과 같다:
$$L_m(z; \theta \odot m_{soft}) = -\sum_{z \in D} \sum_{o \in O} \mathbb{1}(y=o) \log \left( \frac{e^{-d(p_o, f(x; \theta \odot m_{soft}))}}{\sum_{o_k \in O} e^{-d(p_{o_k}, f(x; \theta \odot m_{soft}))}} \right)$$
여기서 $d(\cdot, \cdot)$는 Cosine distance이며, $p_o$는 클래스 $o$의 프로토타입(Prototype)이다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-100, miniImageNet, CUB-200-2011.
- **설정**: 60개 base 클래스 $\rightarrow$ 5-way 5-shot incremental sessions (CIFAR-100, miniImageNet 기준).
- **백본**: ResNet18.
- **비교 대상**: iCaRL, Rebalance, FSLL, F2M 및 binary 서브네트워크를 사용하는 HardNet.

### 주요 결과
- **정량적 성과**: SoftNet은 CIFAR-100과 miniImageNet에서 SOTA 방법론들을 능가하였으며, 일부 설정에서는 상한선(Upper bound)으로 간주되는 cRT(Classifier Re-training)보다 높은 성능을 보였다.
- **Sparsity 영향**: 파라미터 사용량 $c$가 증가함에 따라 전반적인 성능이 향상되는 경향을 보였으며, 특히 miniImageNet과 같이 복잡한 데이터셋에서 그 효과가 더 뚜렷했다.
- **HardNet과의 비교**: 이진 마스크를 사용하는 HardNet보다 소프트 마스크를 사용하는 SoftNet의 성능이 일관되게 높았다. 이는 소프트 마스크가 더 유연한 최적화 공간을 제공함을 시사한다.
- **레이어별 분석**: 높은 층(Higher layers)의 가중치를 fine-tuning 하는 것이 가장 성능이 좋았으며, 이는 하위 층의 특징은 범용적이고 상위 층의 특징은 클래스 특이적이라는 점을 뒷받침한다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 Soft-subnetwork가 Dense network나 Hard-subnetwork보다 **더 평평한 손실 지형(Flatter loss landscape)**을 형성함을 이론적(Lipschitz-continuous objective gradients 분석) 및 실험적으로 증명하였다. 손실 지형이 평평할수록 모델은 작은 섭동에 강건하며, 이는 FSCIL 환경에서 적은 데이터로 학습할 때 과적합을 방지하고 일반화 성능을 높이는 핵심 요인이 된다.

### 한계 및 비판적 논의
- **Major Subnetwork의 고정**: 본 연구는 Major 부분을 완전히 고정함으로써 망각을 방지하지만, 이는 새로운 클래스가 매우 많아질 경우 모델의 용량 제한으로 인해 성능 저하가 발생할 수 있다.
- **보안 취약성**: 가중치 크기(Magnitude) 기반으로 서브네트워크를 분리하므로, 모델 파라미터가 노출될 경우 어떤 파라미터가 중요한 지식이 저장된 곳인지 쉽게 파악될 수 있어 공격에 취약할 가능성이 있다.
- **하이퍼파라미터 의존성**: 최적의 성능을 내기 위한 용량 $c$ 값이 데이터셋과 아키텍처에 따라 다르므로, 이에 대한 자동화된 탐색 방법이 필요하다.

## 📌 TL;DR

본 논문은 Few-Shot Class-Incremental Learning에서 발생하는 치명적 망각과 과적합 문제를 해결하기 위해 **소프트 마스크 기반의 서브네트워크(SoftNet)**를 제안한다. 모델을 지식 보존을 위한 **Major** 부분과 지식 습득을 위한 **Minor** 부분으로 나누어, Major는 고정하고 Minor만 업데이트하는 전략을 취한다. 이를 통해 Dense network보다 더 평평한 손실 지형을 확보하여 일반화 성능을 극대화하였으며, 다양한 벤치마크 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 향후 태스크 특화 아키텍처 탐색 및 희소 모델(Sparse model) 활용 연구에 중요한 기초를 제공한다.