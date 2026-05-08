# Shot in the Dark: Few-Shot Learning with No Base-Class Labels

Zitian Chen, Subhransu Maji, Erik Learned-Miller (2021)

## 🧩 Problem to Solve

Few-Shot Learning (FSL)의 핵심은 매우 적은 수의 레이블된 예제만을 사용하여 새로운 클래스(novel classes)에 대한 분류기를 구축하는 것이다. 일반적으로 FSL은 base classes라고 불리는 별도의 데이터셋에서 학습된 inductive bias를 활용하여 이 문제를 해결한다. 그러나 base classes와 novel classes 사이의 데이터 분포 차이(distribution shift)로 인해, base classes에서 학습된 지식이 novel classes로 전이될 때 일반화 성능이 떨어지는 문제가 발생한다.

기존의 Transductive Few-Shot Learning (TFSL) 연구들은 이 문제를 해결하기 위해 novel classes의 레이블되지 않은(unlabeled) 예제들을 함께 사용하였다. 하지만 대부분의 TFSL 방법론 역시 base classes의 레이블된 데이터를 통해 기초적인 inductive bias를 먼저 학습해야 한다는 전제 조건이 있다. 본 논문은 **base classes의 레이블이 전혀 없는 상황에서도 self-supervised learning (SSL)을 통해 강력한 inductive bias를 구축할 수 있는가**라는 질문을 던지며, 이를 통해 TFSL의 새로운 패러다임을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 novel classes의 unlabeled 데이터가 단순히 기존 모델을 미세 조정(fine-tuning)하는 용도가 아니라, **처음부터 모델을 학습시켜 더 나은 일반화 성능을 가진 표현(representation)을 학습**시키는 데 사용될 수 있다는 점이다.

특히, base classes의 레이블을 전혀 사용하지 않고 오직 unlabeled 데이터(base 및 novel 모두)만을 이용해 self-supervised learning을 수행하는 것이, 레이블된 base classes 데이터를 사용하는 기존의 SOTA TFSL 방법론들보다 더 뛰어난 성능을 보일 수 있음을 실험적으로 증명하였다. 또한, supervised features와 self-supervised features의 상호 보완성을 분석하여 두 특징을 결합했을 때 성능이 향상됨을 보였다.

## 📎 Related Works

### 1. Few-Shot Learning (FSL)

기존 FSL 연구는 크게 데이터 증강(Data Augmentation), 메타 학습(Meta-learning), 그리고 메트릭 학습(Metric Learning)으로 나뉜다. 최근에는 base classes에 대해 지도 학습(supervised learning)을 수행하여 얻은 특징 추출기 위에 단순한 분류기를 학습시키는 방식이 매우 경쟁력 있는 성능을 보이고 있다.

### 2. Transductive Few-Shot Learning (TFSL)

TFSL은 novel classes의 unlabeled instances의 분포 정보를 활용한다. 기존 방식들은 신뢰도가 높은 샘플을 선택해 self-training을 하거나, 보조 손실 함수(auxiliary loss)를 정규화 도구로 사용하여 base classes에서 얻은 inductive bias를 조정하는 방식을 취했다. 즉, 여전히 base classes의 레이블된 데이터에 의존적이다.

### 3. Self-supervised Learning (SSL)

레이블 없이 데이터 자체의 내부 구조를 학습하는 방식으로, 이미지 회전 예측, jigsaw puzzle 풀기, 그리고 최근의 instance discrimination (예: MoCo, SimCLR) 등이 있다. 이전 연구들이 SSL을 FSL의 보조 작업으로 사용하거나 non-transductive 설정에서 탐구한 것과 달리, 본 논문은 **transductive 설정에서 base class 레이블 없이 SSL만으로 학습하는 설정**의 효과를 분석하였다.

## 🛠️ Methodology

### 1. 학습 설정 정의

본 논문은 다음과 같이 네 가지 학습 설정을 정의한다.

- **FSL**: 레이블된 base 데이터($D_{base}$)만 사용.
- **TFSL**: 레이블된 base 데이터($D_{base}$)와 레이블되지 않은 novel 데이터($U_{novel}$)를 함께 사용.
- **UBC-FSL** (Unlabeled-Base-Class FSL): 레이블되지 않은 base 데이터($U_{base}$)만 사용.
- **UBC-TFSL** (Unlabeled-Base-Class TFSL): 레이블되지 않은 base 데이터($U_{base}$)와 레이블되지 않은 novel 데이터($U_{novel}$)를 모두 사용.

### 2. Self-supervised Learning (SSL)

효율성을 위해 **instance discrimination** 작업을 수행하며, MoCo-v2 프레임워크를 기반으로 한다. 각 입력 이미지 $x_i$는 두 가지 서로 다른 뷰 $x^q_i$와 $x^k_i$로 증강되며, 두 인코더 $f_q$와 $f_k$를 통해 임베딩 $q_i$와 $k_i$를 생성한다.

학습 목표는 동일한 이미지에서 생성된 positive pair는 가깝게, 서로 다른 이미지에서 생성된 negative pair는 멀게 만드는 것이며, 다음과 같은 contrastive loss를 사용한다.

$$L(q_i, k_i) = -\log \left( \frac{\exp(q_i^T k_i / \tau)}{\exp(q_i^T k_i / \tau) + \sum_{j \neq i} \exp(q_i^T k_j / \tau)} \right)$$

여기서 $\tau$는 temperature 하이퍼파라미터이다.

### 3. 평가 및 분류 절차

학습된 임베딩 네트워크를 고정한 상태에서, 주어진 $N$-way $m$-shot 태스크의 $N \times m$개 학습 예제들을 이용하여 그 위에 단순한 **logistic regression classifier**를 학습시킨 후 테스트 예제들을 분류한다.

### 4. 특징 결합 (Combined Method)

지도 학습으로 얻은 supervised features와 SSL로 얻은 self-supervised features의 상호 보완성을 확인하기 위해, 두 특징 벡터를 각각 정규화(normalization)한 뒤 단순 연결(concatenation)하고 다시 정규화하여 최종 임베딩으로 사용한다.

## 📊 Results

### 1. 실험 환경

- **데이터셋**: miniImageNet, tieredImageNet (single-domain), Caltech-256, miniImageNet&CUB (cross-domain).
- **백본 아키텍처**: ResNet-12, ResNet-12*, ResNet-50, ResNet-101, WRN-28-10.
- **비교 대상**: MetaOptNet, Distill, Neg-Cosine (FSL), ICI, TAFSSL, EPNet (TFSL).

### 2. 주요 결과

- **UBC-TFSL의 압도적 성능**: base class 레이블 없이 unlabeled 데이터만 사용한 UBC-TFSL이 base class 레이블을 사용하는 기존 SOTA TFSL 방법론들을 능가하였다. miniImageNet과 tieredImageNet의 5-shot 정확도에서 각각 3.5%, 3.9% 더 높은 성능을 기록하였다.
- **특징 결합의 효과**: supervised features와 self-supervised features를 결합한 "Combined" 방식이 FSL baseline보다 일관되게 높은 성능을 보였으며, 특히 ResNet-101 사용 시 5-shot 정확도가 최대 4% 향상되었다.
- **네트워크 깊이의 영향**: 지도 학습 기반의 FSL baseline은 네트워크가 깊어져도 성능 향상이 미미했으나, SSL 기반의 UBC-FSL 및 UBC-TFSL은 네트워크가 깊어질수록(예: ResNet-101) 성능이 비약적으로 향상되었다.
- **Cross-domain FSL**: 완전히 새로운 도메인으로 전이될 때, 테스트 데이터의 unlabeled 샘플에 접근할 수 없는 경우에는 supervised features가 더 유리했다. 하지만 테스트 데이터의 unlabeled 샘플을 사용할 수 있는 UBC-TFSL 설정에서는 다시 SSL 기반 방식이 우위를 점했다.
- **Shot 수에 따른 변화**: 1-shot에서는 supervised features가 우세하지만, shot 수가 증가할수록(예: 100-shot) self-supervised features의 성능이 이를 추월하는 경향을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 발견

본 연구는 FSL에서 반드시 base class의 레이블이 필요한 것은 아니며, 적절한 SSL 전략과 unlabeled 데이터(특히 target domain의 데이터)가 있다면 훨씬 더 일반화 능력이 뛰어난 inductive bias를 형성할 수 있음을 보였다. 이는 TFSL이 단순히 기존 모델을 조정하는 것이 아니라, 데이터의 분포 자체를 학습하는 방향으로 나아가야 함을 시사한다.

### 2. 한계 및 해석

- **데이터셋 크기 의존성**: SSL 특징은 데이터셋의 규모가 클수록 강력해지지만, 매우 작은 데이터셋(예: Caltech-256)에서는 오히려 supervised features보다 성능이 떨어진다. 이는 SSL이 유의미한 표현을 학습하기 위해 더 많은 양의 데이터가 필요함을 의미한다.
- **전이 가능성(Transferability)**: Supervised features는 고수준의 시맨틱 개념을 포착하여 적은 데이터로도 빠르게 적응하는 능력이 좋은 반면, self-supervised features는 더 풍부한 데이터가 주어졌을 때 더 나은 일반화 성능을 제공한다.

### 3. 비판적 논의

논문은 UBC-TFSL이 SOTA TFSL을 능가한다고 주장하지만, 이는 테스트 셋의 unlabeled 데이터를 학습에 사용했기 때문이다. 실제 환경에서 테스트 데이터의 unlabeled 샘플을 미리 확보할 수 있는지에 대한 현실적인 제약 조건을 고려해야 한다. 또한, SSL 방법론(MoCo, SimCLR 등) 간의 성능 차이가 크지 않다는 점은 특정 알고리즘보다 '데이터의 구성'과 'transductive 설정' 자체가 더 핵심적인 요인임을 보여준다.

## 📌 TL;DR

본 논문은 **base class의 레이블 없이 오직 unlabeled 데이터만을 이용한 Self-Supervised Learning (SSL)이 기존의 레이블 기반 Transductive Few-Shot Learning 방법론들보다 더 우수한 성능을 낼 수 있음**을 증명하였다. 특히, 깊은 네트워크 아키텍처와 대규모 unlabeled 데이터가 결합될 때 SSL의 일반화 능력이 극대화되며, 지도 학습 특징과 자가 지도 학습 특징을 결합하는 것이 성능 향상에 효과적임을 보였다. 이 연구는 향후 레이블 비용을 최소화하면서도 강력한 FSL 모델을 구축하는 새로운 방향성을 제시한다.
