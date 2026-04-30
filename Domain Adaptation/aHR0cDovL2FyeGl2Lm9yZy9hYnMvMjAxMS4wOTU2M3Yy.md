# Robustified Domain Adaptation

Jiajin Zhang, Hanqing Chao, Pingkun Yan (2021)

## 🧩 Problem to Solve

본 논문은 Unsupervised Domain Adaptation (UDA) 환경에서 모델의 Adversarial Robustness(적대적 강건성)를 확보하는 문제를 다룬다. UDA는 레이블이 있는 Source Domain의 지식을 레이블이 없는 Target Domain으로 전이하여 데이터 분포의 차이(Domain Shift)를 줄이는 것을 목표로 한다. 

기존의 딥러닝 모델들이 적대적 공격(Adversarial Attacks)에 취약하다는 점은 잘 알려져 있으나, UDA 모델의 강건성에 대한 연구는 상대적으로 부족했다. 저자들은 UDA에서 발생하는 불가피한 도메인 분포의 편차가 Target Domain에서의 모델 강건성을 저해하는 결정적인 장벽이 된다는 점을 지적한다. 특히, Source Domain에서 수행한 Adversarial Training (AT)의 효과가 Target Domain으로 그대로 전이되지 않으며, Target Domain에서 레이블 없이 강건성을 높이려 할 경우 오히려 깨끗한 샘플(Clean Samples)에 대한 분류 성능이 크게 떨어지는 '분류 혼란(Classification Confusion)' 현상이 발생한다는 문제를 제기한다. 따라서 본 논문의 목표는 Target Domain의 깨끗한 데이터 성능을 유지하면서도 적대적 공격에 강건한 UDA 모델을 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Source Domain에서 구축한 판별력 있는 강건한 특징 공간을 앵커(Anchor)로 활용하여, Target Domain의 깨끗한 샘플과 적대적 샘플을 동시에 정렬하는 것이다. 이를 위해 **Class-consistent Unsupervised Robust Domain Adaptation (CURDA)** 프레임워크를 제안한다.

CURDA의 중심 설계는 크게 두 가지이다. 첫째, **Contrastive Robust Training (CORTrain)**을 통해 클래스 내 거리는 가깝고 클래스 간 거리는 먼 판별력 있는 Source-Robust 모델을 먼저 학습시킨다. 둘째, 학습된 Source-Robust 모델의 표현력을 고정된 앵커로 사용하는 **Source Anchored Adversarial (SAA) Contrastive Loss**를 도입한다. 이를 통해 Target Domain의 샘플들이 잘못된 클래스로 매핑되는 것을 방지하면서, 깨끗한 샘플과 적대적 샘플이 동일한 클래스 앵커로 모이도록 유도하여 강건성을 확보한다.

## 📎 Related Works

### Unsupervised Domain Adaptation (UDA)
기존 UDA 연구들은 주로 도메인 불변 특징(Domain-invariant feature)을 학습하여 도메인 간의 간극을 줄이는 방식에 집중했다. 대표적으로 DANN이나 ADDA와 같은 적대적 학습 기반 방법과 MMD를 이용한 불일치 기반 방법들이 있다. 최근에는 SRDC와 같이 도메인 간의 구조적 유사성을 활용하는 방식들이 제안되어 높은 성능을 보였으나, 이러한 방법들은 모델의 적대적 강건성 문제는 고려하지 않았다.

### Adversarial Attack and Defense
FGSM, PGD와 같은 적대적 공격 방법과 이를 방어하기 위한 Adversarial Training 기법들이 연구되어 왔다. 특히 레이블이 없는 데이터를 활용하는 Label-free robust training 방법들이 제안되었으나, 이는 주로 동일 도메인 내에서의 반지도 학습(Semi-supervised learning) 상황을 가정한다. UDA와 같이 소스와 타겟의 도메인이 다른 경우에는 이러한 방법들이 타겟 도메인의 데이터 분포를 잘못 학습하게 하여 깨끗한 샘플의 정확도를 떨어뜨리는 한계가 있다.

## 🛠️ Methodology

CURDA 프레임워크는 크게 두 단계의 파이프라인으로 구성된다.

### 1. Contrastive Robust Training (CORTrain)
먼저 Source Domain에서 판별력이 높은 강건한 모델 $F_{sr}(\cdot)$을 학습시킨다. PGD 공격을 통해 생성된 적대적 샘플 $x'_s$를 사용하여 다음의 두 손실 함수를 최소화한다.

- **Cross Entropy Loss ($L_{ce}$)**: 적대적 샘플이 올바른 클래스로 분류되도록 가이드한다.
- **Contrastive Loss ($L_{con}$)**: 특징 공간에서의 클래스 내 응집도와 클래스 간 분리도를 높인다.
$$L_{con} = \frac{1}{2} [ \mathbb{1}[y_{si}=y_{sj}] D_r^2 + \mathbb{1}[y_{si} \neq y_{sj}] \max(0, m_s - D_r)^2 ]$$
여기서 $D_r = \|F_{sr}^f(x'_i) - F_{sr}^f(x'_j)\|_2$는 두 샘플 간의 유클리드 거리이며, $m_s$는 과적합 방지를 위한 마진이다.

최종적으로 $L_{cort} = L_{ce} + \lambda_{con} L_{con}$을 통해 Source-Robust 모델을 생성한다.

### 2. Source Anchored Adversarial (SAA) Contrastive Loss
다음으로, 고정된 $F_{sr}(\cdot)$을 앵커로 사용하여 UDA 모델 $F_{uda}(\cdot)$를 학습시킨다.

- **SAA-contrastive Loss**: Target Domain의 깨끗한 샘플 $x_t$와 적대적 샘플 $x'_t$를 소스 도메인의 고정된 앵커 표현력에 정렬시킨다. 
- **동작 원리**: 동일 클래스의 소스 앵커로는 끌어당기고(Pull), 다른 클래스의 앵커로는 밀어낸다(Push). 이는 Target Domain의 깨끗한 샘플과 적대적 샘플 사이의 거리를 간접적으로 줄이면서도, 고정된 앵커를 사용함으로써 적대적 샘플이 깨끗한 샘플을 잘못된 클래스로 끌고 가는 현상을 방지한다.
- **수식**: 
$$L_{saa} = \mathbb{E}_{\{(x_s, y_s), x_t\} \sim \{S, T\}} [ l(F_{sr}^f(x_s), F_{uda}^f(x_t), y_s, \hat{y}_t) + l(F_{sr}^f(x_s), F_{uda}^f(x'_t), y_s, \hat{y}_t) ]$$
여기서 $l(\cdot)$은 표준 Contrastive Loss 형태를 따르며, $\hat{y}_t$는 Target Domain의 의사 레이블(Pseudo label)이다.

### 3. Pseudo Label Generation 및 기타 손실 함수
- **의사 레이블 생성**: 초기 $\tau$ 반복 횟수 동안은 고정된 $F_{sr}(\cdot)$을 사용하고, 모델이 안정화된 이후에는 $F_{uda}(\cdot)$를 사용하여 $\hat{y}_t$를 생성한다. 이때 신뢰도 임계값 $P_{pseudo}$를 넘는 샘플만 사용한다.
- **Label-free Robust Consistency Loss ($L_{trade}$)**: TRADES에서 제안된 KL-divergence 기반 손실 함수를 추가하여, 의사 레이블이 불확실한 샘플들에 대해서도 깨끗한 샘플과 적대적 샘플 간의 일관성을 강제한다.
$$L_{trade} = -\mathbb{E}_{x_t \sim T} KL(C(E_{uda}(x_t)), C(E_{uda}(x'_t)))$$

- **전체 손실 함수**: 
$$L_{curda} = \lambda_{saa} L_{saa} + L_{uda} + L_{trade}$$
여기서 $L_{uda}$는 기존 UDA 모델(ADDA, SRDC 등)의 고유 손실 함수이다.

## 📊 Results

### 실험 설정
- **데이터셋**: DIGITS (MNIST $\rightarrow$ USPS, USPS $\rightarrow$ MNIST, MNIST $\rightarrow$ MNIST-M), Office-31 (6개 조합)
- **비교 대상**: Vanilla UDA, UDA + AT in Source Domain (SD), UDA + AT in SD & Target Domain (TD)
- **측정 지표**: 깨끗한 데이터에 대한 정확도(Clean Accuracy)와 PGD 공격 시의 강건성(Robustness)
- **백본**: LeNet (DIGITS), ResNet-50 (Office-31)

### 주요 결과
- **강건성 향상**: DIGITS 벤치마크에서 CURDA는 기존 UDA 모델 대비 강건성을 60% 이상 크게 향상시켰다. 
- **정확도 유지**: Source 및 Target 모두에서 AT를 수행한 모델(UDA+AT in SD & TD)은 강건성은 높았으나 Clean Accuracy가 급격히 하락하는 경향을 보였다. 반면, CURDA는 매우 경쟁력 있는 정확도를 유지하면서 높은 강건성을 달성하였다.
- **정성적 분석**: t-SNE 시각화 결과, CURDA는 Target Domain의 적대적 샘플들을 해당 클래스의 Source Domain 클러스터에 정확하게 정렬시키는 반면, 단순 AT 방식은 적대적 샘플들이 엉뚱한 클래스로 뭉치거나 분포가 흐트러지는 모습을 보였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 통찰은 **"단순히 Target Domain에서 적대적 샘플과 깨끗한 샘플의 거리를 좁히는 것만으로는 부족하며, 신뢰할 수 있는 기준점(Anchor)이 필요하다"**는 점이다. 기존의 Label-free robust training은 타겟 도메인 내에서만 일관성을 찾으려 하기에, 모델이 잘못된 방향으로 수렴할 경우 깨끗한 데이터까지 함께 오염되는 결과(Classification Confusion)를 초래한다.

CURDA는 Source Domain에서 미리 학습된 견고한 특징 공간을 앵커로 활용함으로써, 타겟 도메인의 데이터를 정렬할 때의 가이드라인을 제공한다. 이는 도메인 간의 분포 차이가 존재하더라도, 클래스 간의 상대적인 구조를 유지하며 강건성을 확보할 수 있게 한다. 다만, 정확도와 강건성 사이의 Trade-off가 존재하여, 강건성을 높이는 과정에서 Clean Accuracy가 소폭 하락하는 현상은 불가피한 것으로 보인다.

## 📌 TL;DR

본 논문은 UDA 모델이 적대적 공격에 취약하며, 단순한 타겟 도메인 강건성 학습이 오히려 일반 성능을 떨어뜨리는 문제를 해결하기 위해 **CURDA** 프레임워크를 제안한다. Source Domain에서 판별력 있는 강건한 공간을 먼저 학습시키고, 이를 고정 앵커로 삼아 Target Domain의 깨끗한/적대적 샘플을 클래스 일관성 있게 정렬함으로써, **정확도 손실을 최소화하면서 Target Domain의 적대적 강건성을 획기적으로 향상**시켰다. 이 연구는 향후 실세계의 다양한 도메인 환경에서 보안성이 강화된 AI 모델을 배포하는 데 중요한 기반이 될 수 있다.