# Comparing Contrastive and Triplet Loss: Variance Analysis and Optimization Behavior

Donghuo Zeng (2025)

## 🧩 Problem to Solve

본 논문은 Deep Metric Learning(DML) 분야에서 널리 사용되는 두 가지 손실 함수인 Contrastive Loss와 Triplet Loss가 학습된 표현(Representation)의 품질과 최적화 동작(Optimization Behavior)에 구체적으로 어떤 영향을 미치는지 분석하고자 한다.

두 손실 함수 모두 클래스 간의 분리도를 높이는 것을 목표로 하지만, 그 수식적 구성의 차이가 임베딩 공간의 기하학적 구조와 수렴 과정에 서로 다른 영향을 미친다. 특히, 클래스 내부의 응집도(Intra-class compactness)와 클래스 간의 분리도(Inter-class separation) 사이의 트레이드-오프를 어떻게 관리하는지, 그리고 학습 과정에서 그래디언트가 어떻게 할당되는지에 대한 이론적·경험적 이해가 부족한 상황이다. 따라서 본 연구의 목표는 Variance Analysis와 Optimization Greediness라는 관점에서 두 손실 함수를 비교하여, 특정 작업(예: 세밀한 검색 vs. 광범위한 정제)에 어떤 손실 함수가 더 적합한지에 대한 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Contrastive Loss의 'Greedy'한 최적화 특성과 Triplet Loss의 'Selective'한 업데이트 특성을 정량적으로 분석하고, 이것이 임베딩 공간의 분산 구조에 미치는 영향을 밝혀낸 것이다.

중심적인 아이디어는 Contrastive Loss가 마진(Margin)을 충족한 이후에도 계속해서 샘플들을 끌어당기거나 밀어내는 경향이 있어 클래스 내부 표현을 과도하게 압축(Over-compacting)하는 반면, Triplet Loss는 상대적 거리 제약 조건을 만족하면 그래디언트 업데이트를 중단함으로써 클래스 내부의 다양성(Diversity)을 더 잘 보존한다는 점이다. 이를 통해 Triplet Loss가 세밀한 구분(Fine-grained distinction)이 필요한 작업에서 더 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

논문은 Deep Metric Learning의 기초가 되는 Hadsell et al.의 Contrastive Loss와 FaceNet(Schroff et al.)의 Triplet Loss를 주요 배경으로 다룬다. 기존 연구들은 두 손실 함수가 임베딩 공간을 어떻게 형성하는지 언급해 왔으나, 본 논문은 다음과 같은 차별점을 가진다.

첫째, 단순히 성능 수치만을 비교하는 것이 아니라, Intra-class 및 Inter-class Variance라는 정량적 지표를 통해 임베딩 공간의 구조적 특성을 분석한다. 둘째, 'Greediness'라는 개념을 도입하여 Loss-decay rate, Active-sample ratio, Gradient norm이라는 세 가지 지표로 최적화 역학(Optimization dynamics)을 상세히 비교한다. 이는 기존 연구들이 간과했던 "학습 과정에서 그래디언트가 어떻게 분배되는가"에 대한 심층적인 분석을 제공한다.

## 🛠️ Methodology

### 1. 손실 함수 정의

본 연구에서 비교하는 두 손실 함수는 다음과 같다.

**Contrastive Loss:**
포지티브 쌍($P$)은 서로 끌어당기고, 네거티브 쌍($N$)은 마진 $m$ 이상으로 밀어낸다.
$$L_{con} = \sum_{(x,y) \in P} \|f(x) - f(y)\|^2 + \sum_{(x,y) \in N} [m - \|f(x) - f(y)\|]_+^2$$
여기서 $[z]_+ = \max(0, z)$이며, 마진을 충족한 이후에도 계속 업데이트가 발생하므로 'Greedy'한 특성을 보인다.

**Triplet Loss:**
앵커($a$), 포지티브($p$), 네거티브($n$)의 삼중조를 사용하여, 앵커-포지티브 거리가 앵커-네거티브 거리보다 최소 $m$만큼 더 가깝도록 강제한다.
$$L_{tri} = \sum_{(a,p,n)} [\|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + m]_+$$
이 방식은 상대적 순위(Ranking)를 중시하며, 제약 조건을 만족하면 그래디언트가 0이 되어 하드 샘플(Hard samples)에 집중하는 특성이 있다.

### 2. 분석 지표 (Metrics)

임베딩 공간의 구조와 최적화 동작을 분석하기 위해 다음 지표들을 정의한다.

**분산 분석 (Variance Analysis):**

- **Intra-class variance ($\sigma^2_{intra}$):** 각 클래스 내 샘플들이 중심 $\mu_c$로부터 얼마나 떨어져 있는지를 측정하여 클래스 내부의 다양성을 평가한다.
$$\sigma^2_{intra} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_c} \sum_{i \in I_c} \|z_i - \mu_c\|^2$$
- **Inter-class variance ($\sigma^2_{inter}$):** 서로 다른 클래스 중심들 간의 평균 거리를 측정하여 클래스 간 분리도를 평가한다.
$$\sigma^2_{inter} = \frac{1}{C(C-1)} \sum_{c \neq c'} \|\mu_c - \mu_{c'}\|^2$$

**최적화 탐욕성 (Optimization Greediness):**

- **Loss-decay rate:** 초기 손실 값의 10% 수준까지 떨어지는 데 걸리는 에폭(Epoch) 수이다.
- **Active Ratio:** 배치 내에서 손실 값이 0보다 커서 실제로 그래디언트 업데이트에 참여하는 샘플의 비율이다.
- **Gradient Norm:** 파라미터 업데이트의 크기를 결정하는 그래디언트의 $L_2$ 노름($\|\nabla L\|_2$)이다.

### 3. 실험 설정

- **데이터셋:** Synthetic data(128차원), MNIST, CIFAR-10, CARS196, CUB-200을 사용하였다.
- **아키텍처:** Synthetic 데이터는 MLP, MNIST/CIFAR-10은 CNN, 검색 작업(CARS196, CUB-200 등)은 Frozen ViT-B/32 백본을 사용하였다.
- **학습 설정:** Adam 옵티마이저, 학습률 $1e-3$, 배치 크기 64, 마진 $m=1.0$으로 설정하여 50에폭 동안 학습하였다.

## 📊 Results

### 1. 분산 구조 분석

Synthetic 데이터와 MNIST 실험 결과, Triplet Loss가 Contrastive Loss보다 더 높은 $\sigma^2_{intra}$를 유지함을 확인하였다. 특히 Synthetic 데이터에서 Triplet Loss의 클래스 내 분산은 Contrastive Loss보다 약 2.4배 높게 나타났다. 이는 Contrastive Loss가 클래스 내부를 과도하게 압축하여 세밀한 의미적 차이를 소멸시키는 반면, Triplet Loss는 데이터의 자연스러운 구조를 더 잘 보존함을 의미한다.

### 2. 최적화 동작 분석

최적화 지표 분석 결과, 두 손실 함수의 상반된 동작이 드러났다.

- **Contrastive Loss:** Loss-decay rate가 27에폭으로 짧고, Active ratio가 65%로 높으며, Gradient norm은 0.12로 낮다. 즉, 많은 샘플에 대해 작고 광범위한 업데이트를 빈번하게 수행하여 빠르게 수렴하지만, 과하게 압축되는 경향이 있다.
- **Triplet Loss:** Loss-decay rate가 43에폭으로 더 길고, Active ratio는 38%로 낮으나, Gradient norm은 0.27로 훨씬 높다. 이는 소수의 하드 샘플에 대해 강한 업데이트를 수행하며 학습을 더 오래 지속함으로써 임베딩의 다양성을 보존한다.

### 3. 실제 작업 성능

- **분류(Classification):** MNIST와 CIFAR-10에서 Triplet Loss가 더 높은 정확도를 보였다. (CIFAR-10 기준: Triplet 0.9371 vs. Contrastive 0.8998)
- **검색(Retrieval):** Recall@1(r@1) 지표에서 Triplet Loss가 일관되게 우수한 성능을 보였다. 특히 CIFAR-10(0.9192 vs 0.8433), CARS196(0.2982 vs 0.2542), CUB-200(0.3421 vs 0.3154)에서 확연한 차이를 보였다.

## 🧠 Insights & Discussion

본 논문의 결과는 임베딩 공간의 '적절한 분산'이 모델의 일반화 성능과 세밀한 구별 능력에 핵심적이라는 점을 시사한다.

**강점 및 해석:**
Contrastive Loss는 모든 쌍에 대해 절대적인 거리 제약을 강제하기 때문에 최적화 과정이 매우 '탐욕적'이다. 이로 인해 빠르게 수렴하지만, 클래스 내부의 샘플들이 하나의 점으로 붕괴(Collapse)되는 현상이 발생하여 세밀한 특징(Fine-grained details)이 사라지게 된다. 반면, Triplet Loss는 상대적인 거리 관계만을 정의하므로, 마진이 충족된 샘플은 학습에서 제외하고 어려운 샘플에만 집중한다. 이러한 '선택적' 업데이트 방식이 임베딩 공간에 적절한 여유를 주어, 검색 작업에서 필수적인 정교한 랭킹 능력을 향상시킨다.

**한계 및 논의:**
본 연구는 고정된 마진 $m=1.0$을 사용하였으나, 마진 값의 변화가 Greediness와 Variance에 어떤 영향을 미치는지에 대한 분석은 부족하다. 또한, Frozen ViT 백본을 사용한 실험이 많아, 전체 네트워크를 Fine-tuning 했을 때의 최적화 역학이 동일하게 나타날지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 Contrastive Loss와 Triplet Loss의 최적화 동작과 임베딩 구조를 정량적으로 비교 분석하였다. **Contrastive Loss**는 많은 샘플에 대해 작고 빈번한 업데이트를 수행하여 클래스 내부를 과도하게 압축하는 'Greedy'한 특성을 보이며, **Triplet Loss**는 소수의 하드 샘플에 집중하여 강한 업데이트를 수행함으로써 클래스 내 다양성을 보존한다. 결과적으로 Triplet Loss가 분류 및 검색 작업 모두에서 더 우수한 성능을 보였으며, 이는 세밀한 특징 보존이 필요한 작업에는 Triplet Loss가, 전반적인 임베딩 정제가 필요한 작업에는 Contrastive Loss가 적합함을 시사한다.
