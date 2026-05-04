# Incremental Meta-Learning via Indirect Discriminant Alignment

Qing Liu, Orchid Majumder, Alessandro Achille, Avinash Ravichandran, Rahul Bhotika, Stefano Soatto (2020)

## 🧩 Problem to Solve

본 논문은 새로운 분류 작업(classification tasks)을 학습하면서, 동시에 각 작업을 해결할 때마다 기본 모델(base model)이 점진적으로 개선될 수 있도록 하는 **Incremental Meta-Learning (IML)** 문제를 다룬다.

일반적인 Meta-learning 방법론들은 새로운 작업을 학습하는 과정에서 얻은 경험을 기본 모델에 다시 반영하여 모델 자체를 진화시키는 메커니즘이 부족하다. 반면, 점진적 학습(Incremental Learning)은 새로운 작업을 추가할 때 기존 지식을 잊어버리는 **Catastrophic Forgetting** 현상을 해결하는 데 집중한다. 

특히 Few-shot learning 환경에서의 IML은 다음과 같은 기술적 난제가 존재한다:
1. **Disjoint Classes**: 서로 다른 작업들이 서로 겹치지 않는 클래스 집합을 가질 수 있으므로, 모델 증류(Model Distillation)에서 사용하는 방식처럼 분류기(Classifier)를 직접 정렬(Align)하는 것이 불가능하다.
2. **Flexibility vs. Stability**: 단순히 모든 클래스가 공유하는 특징(Feature)을 정렬하는 방식은 모델이 새로운 작업을 해결하기 위해 진화해야 하는 유연성을 과도하게 제한한다.
3. **Data Accessibility**: 이전 단계의 학습 데이터에 다시 접근하는 것은 비용이 많이 들거나 개인정보 보호 등의 이유로 불가능할 수 있다.

따라서 본 논문의 목표는 이전 데이터를 재처리하지 않고도 새로운 데이터를 통해 기본 모델을 지속적으로 개선하며, 기존 지식을 보존하는 효율적인 IML 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Indirect Discriminant Alignment (IDA)**이다. 

기존의 방식들이 특징(Feature) 자체를 직접 정렬하거나 분류기의 출력을 직접 맞추려 했던 것과 달리, IDA는 최소한의 **"Anchor Classes"**(이전 작업의 클래스 대표값들)를 기준으로 두 모델의 **Discriminant(판별식)**를 간접적으로 정렬한다. 

즉, 새로운 데이터가 입력되었을 때, 이 데이터를 이전 모델의 백본으로 처리했을 때의 판별 결과와 새로운 모델의 백본으로 처리했을 때의 판별 결과가 (이전 클래스 앵커들에 대해) 유사하도록 강제하는 것이다. 이를 통해 모델은 이전 클래스들에 대한 변별력을 유지하면서도, 나머지 자유도(degrees of freedom)를 활용하여 새로운 작업에 적응할 수 있는 유연성을 확보한다.

## 📎 Related Works

- **Incremental / Continual Learning**: Catastrophic Forgetting을 막기 위해 EWC(Elastic Weight Consolidation)와 같이 가중치 변화를 제한하거나, Knowledge Distillation을 통해 모델의 거동을 보존하는 방식이 연구되었다. 하지만 이들은 주로 단일 모델의 유지에 집중하며, Meta-learning의 '학습하는 법을 학습하는' 관점을 통합하지 않았다.
- **Few-shot Meta-Learning**: Prototypical Networks (PN)나 Embedded Class Models (ECM)와 같이 클래스별 프로토타입을 생성하고 거리를 측정하는 Metric-based approach가 주를 이룬다.
- **기존 접근 방식과의 차별점**: 기존의 Incremental Few-shot Learning 연구들은 주로 새로운 클래스를 인식하는 데 집중했으나, 본 논문은 새로운 작업을 수행할 때마다 **기본 메타 학습 모델(Base Meta-Learner) 자체가 점진적으로 성능이 향상되는 virtuous cycle**을 구축하려 한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
본 논문은 모델을 특징 추출기인 **Backbone** $\phi$와 특징을 포스터리어 확률로 매핑하는 **Discriminant head** $f$로 정의한다. 전체 학습 목표는 새로운 데이터 $E$에 대해 다음의 손실 함수를 최소화하는 것이다.

$$w_{t+1} = \arg \min_{w_{t+1}} L(w_{t+1}; E) + \lambda \text{IDA}(\phi_{w_{t+1}} | \phi_{w_t}; C_t)$$

여기서 $L(w_{t+1}; E)$는 새로운 데이터에 대한 fine-tuning 손실이며, $\text{IDA}$는 이전 모델 $\phi_{w_t}$와의 간접적 정렬을 강제하는 정규화 항이다.

### Indirect Discriminant Alignment (IDA)
IDA는 이전 모델의 클래스 집합 $C_{old}$를 앵커로 사용하여, 새로운 데이터 $x \in E$가 들어왔을 때 이전 백본과 현재 백본이 생성하는 판별 결과의 차이를 줄인다.

$$\text{IDA} = \mathbb{E}_{x \sim E, \tau'} [KL(f_{old}(y | \phi_{old}(x)) \parallel f_{old}(y | \phi_{new}(x)))]$$

이 식의 의미는 다음과 같다:
1. $x$를 이전 모델 $\phi_{old}$에 통과시켜 $C_{old}$에 대한 판별 벡터를 얻는다.
2. 동일한 $x$를 새로운 모델 $\phi_{new}$에 통과시킨 후, **이전 모델의 판별 함수 $f_{old}$**를 사용하여 $C_{old}$에 대한 판별 벡터를 얻는다.
3. 두 분포 사이의 KL-divergence를 최소화함으로써, 새로운 모델이 이전 모델의 판별 능력을 유지하도록 만든다.

### 구체적 구현 (Metric Classifier 사례)
본 논문은 Prototypical Networks (PN)를 기본 모델로 사용한다.
- **Class Representative (Prototype)**: 각 클래스의 평균 벡터 $c_k = \psi(D_\tau)_k$로 정의한다.
- **Metric**: 유클리드 거리의 음수 값 $\chi(z, c) = -\|z - c\|^2$을 사용한다.
- **Discriminant (Posterior)**: Softmax 함수를 통해 다음과 같이 계산된다.
$$p(y=k | z) = \frac{e^{\chi(z, c_k)}}{\sum_j e^{\chi(z, c_j)}}$$

학습 절차는 다음과 같다:
1. 이전 단계에서 학습된 클래스 앵커 $C_t$를 저장한다.
2. 새로운 에피소드에서 쿼리 샘플을 통해 기본 분류 손실을 계산한다.
3. 동시에, 저장된 $C_t$를 이용하여 IDA 손실을 계산하고 이를 합산하여 가중치를 업데이트한다.

## 📊 Results

### 실험 설정
- **데이터셋**: MiniImageNet, TieredImageNet, 그리고 도메인 시프트를 측정하기 위한 DomainImageNet (Natural vs Man-made)을 사용하였다.
- **비교 대상 (Baselines)**:
    - **NU (No Update)**: 이전 데이터로만 학습된 모델.
    - **FT (Fine-Tuning)**: 제약 없이 새로운 데이터로만 학습.
    - **DFA (Direct Feature Alignment)**: 특징 벡터 $\phi(x)$를 직접 정렬.
    - **EIML (Exemplar-based IML)**: 이전 데이터의 일부(exemplars)를 보관하여 정렬.
    - **PAR (Paragon)**: 모든 데이터를 한꺼번에 학습시킨 Oracle 모델 (상한선).
- **지표**: 1-shot 5-way 및 5-shot 5-way 분류 정확도.

### 주요 결과
1. **Catastrophic Forgetting 방지**: FT나 DFA 방식은 새로운 클래스 성능은 올라가지만 이전 클래스 성능이 급격히 떨어진다. 반면 IDA는 이전 클래스의 성능을 상당히 잘 유지하였다.
2. **Generalization 성능**: IDA는 새로운 클래스뿐만 아니라, 학습에 사용되지 않은 **Unseen classes**에 대해서도 타 베이스라인보다 높은 성능을 보였다. 이는 IML을 통해 기본 모델의 범용적 표현 능력이 향상되었음을 시사한다.
3. **EIML과의 비교**: 이전 데이터를 일부 사용하는 EIML과 비교했을 때, IDA는 데이터를 저장하지 않음에도 불구하고 매우 유사한 성능을 보였다. 일부 케이스에서는 IDA가 더 우수하였다.
4. **DomainImageNet 결과**: 서로 다른 도메인(자연물 $\rightarrow$ 인공물)으로 학습을 확장했을 때, IDA는 도메인 시프트 상황에서도 효과적으로 지식을 보존하며 성능을 높였다. 특히 다양한 통계적 특성을 가진 클래스를 추가할 때 IML의 이점이 극대화되었다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **데이터 효율성**: 이전 데이터를 저장하거나 재처리하지 않고도 클래스 앵커($C_t$)라는 최소한의 정보만으로 지식 보존과 모델 진화를 동시에 달성하였다.
- **유연한 정렬**: 특징 공간 전체를 고정하는 DFA와 달리, 앵커 클래스에 대한 상대적 거리(Discriminant)만 유지함으로써 모델이 새로운 작업에 맞게 특징 공간을 재구성할 수 있는 여유를 주었다.

### 한계 및 비판적 논의
- **Way/Shot 의존성**: PN 기반 구현의 경우, 학습 시와 테스트 시의 way(클래스 수)가 동일해야 한다는 제약이 있다. 저자들은 이를 ECM(Embedded Class Models)으로 해결할 수 있음을 보였으나, 기본 PN의 한계는 명확하다.
- **Paragon과의 간극**: Unseen classes에 대해서는 PAR 모델에 근접하지만, 현재 학습 중인 새로운 클래스들에 대해서는 여전히 PAR 모델보다 낮은 성능을 보인다. 이는 점진적 학습이 전체 데이터를 한 번에 학습하는 것보다 최적점에 도달하는 효율이 낮을 수 있음을 의미한다.
- **Hard Task Mining의 부재**: 무작위 샘플링을 통해 에피소드를 구성하므로, 모델에게 충분한 자극을 줄 수 있는 '어려운 작업(Hard tasks)'을 선별하여 학습하는 메커니즘이 추가된다면 더 큰 성능 향상이 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 Meta-learning과 Incremental learning을 결합하여, 새로운 작업을 배울 때마다 기본 모델이 계속해서 진화하는 **Incremental Meta-Learning (IML)** 프레임워크를 제안한다. 핵심 기법인 **Indirect Discriminant Alignment (IDA)**는 이전 클래스들의 대표값(Anchor)을 기준으로 판별 결과만을 정렬함으로써, **Catastrophic Forgetting을 억제하는 동시에 새로운 작업에 적응할 수 있는 유연성을 유지**한다. 실험 결과, IDA는 이전 데이터를 저장하지 않고도 데이터 보관 방식(EIML)에 근접하는 성능을 보였으며, 특히 도메인이 다른 데이터를 순차적으로 학습할 때 모델의 일반화 성능이 크게 향상됨을 입증하였다. 이 연구는 실시간으로 새로운 태스크가 추가되는 환경에서 효율적인 Few-shot Learner를 구축하는 데 중요한 기여를 한다.