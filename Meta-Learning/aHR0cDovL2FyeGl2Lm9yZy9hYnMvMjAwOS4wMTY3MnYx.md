# Yet Meta Learning Can Adapt Fast, It Can Also Break Easily

Han Xu, Yaxin Li, Xiaorui Liu, Hui Liu, Jiliang Tang (2020)

## 🧩 Problem to Solve

본 논문은 Meta Learning(메타 학습) 알고리즘의 신뢰성과 강건성(Robustness) 문제를 다룬다. Meta Learning은 적은 수의 학습 샘플만으로도 새로운 작업에 빠르게 적응할 수 있는 능력을 갖추고 있어, 얼굴 인식, 객체 탐지, 로보틱스 모방 학습 등 안전이 중요한(safety-critical) 응용 분야에 널리 적용되고 있다.

그러나 저자들은 Meta Learning이 적대적 공격(Adversarial Attacks)에 얼마나 취약한지에 대해 의문을 제기한다. 특히, 공격자가 Meta Learner에게 제공되는 소수의 학습 데이터(Support set)를 미세하게 조작했을 때, Meta Learner가 학습된 경험을 잘못 이용하여 잘못된 지식을 구축하거나 무용지물인 Adapted Learner(적응된 모델)를 생성할 가능성이 크다는 점을 지적한다. 따라서 본 연구의 목표는 Few-shot Classification 문제 환경에서 Meta Learning에 대한 적대적 공격을 정형화하고, 이를 통해 Meta Learning의 취약성을 체계적으로 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Meta Learning의 적응 과정 자체를 공격 대상으로 삼는 새로운 적대적 공격 프레임워크를 제안한 것이다. 주요 기여 사항은 다음과 같다.

- **메타 공격의 정형화**: Meta Learning 환경에서 적대적 목표(Adversarial Goal)와 인지 불가능한 섭동(Unnoticeable Perturbation)의 개념을 최초로 공식 정의하였다.
- **목표 함수 설계**: Targeted attack과 Non-targeted attack 모두에 적용 가능한 새로운 메타 공격 목적 함수를 수립하였다.
- **MetaAttacker 알고리즘 제안**: 다양한 구조의 Victim Model(피해 모델)에 대해 효율적으로 적대적 입력 세트를 계산할 수 있는 MetaAttacker 알고리즘을 제안하였다.
- **체계적인 강건성 평가**: Optimization-based(MAML), Model-based(SNAIL), Metric-based(Prototypical Networks) 등 대표적인 메타 학습 프레임워크들을 대상으로 실험을 수행하여, 메타 학습이 적대적 공격에 매우 취약함을 입증하였다.

## 📎 Related Works

### 관련 연구 및 한계
기존의 적대적 공격 연구는 주로 이미 학습이 완료된 딥러닝 모델(DNN)의 테스트 입력값을 조작하여 잘못된 예측을 유도하는 방식에 집중해 왔다. 최근 일부 연구에서 메타 학습의 결과물인 Adapted model이 적대적 예제에 취약하다는 점을 밝혀내기도 하였다.

### 기존 접근 방식과의 차별점
본 논문은 Adapted model을 공격하는 것이 아니라, **Meta Learner 그 자체를 공격**한다는 점에서 차별점을 가진다. 즉, 테스트 샘플을 수정하는 것이 아니라 Meta Learner가 Adapted model을 생성하기 위해 사용하는 '가이드 데이터(teaching data)'를 오염시키는 방식이다. 이는 공격자가 테스트 샘플을 보지 않고도 생성될 모델의 전체적인 성능을 파괴하거나 특정 클래스에 대한 개념을 왜곡시킬 수 있다는 점에서 훨씬 더 치명적인 위험을 내포한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 정의
메타 학습의 적응 과정은 다음과 같이 정의된다. 파라미터 $\theta$를 가진 Meta Learner $f_\theta$가 작업 $T$의 학습 데이터 $D_T^{train}$을 입력받아 파라미터 $\phi$를 가진 Adapted model $F(\cdot; \phi)$를 생성한다.
$$\phi = f_\theta(D_T^{train})$$

### 적대적 공격 모델 (Threat Model)
공격자는 White-box 설정 하에 Meta Learner의 모든 파라미터와 적응 프로세스를 알고 있다고 가정한다. 공격자는 $D_T^{train}$의 일부를 조작하여 적대적 학습 세트 $D_T^{adv}$를 구축한다.

#### 1. 적대적 목표 (Adversarial Goals)
- **Non-targeted Attack (DoS Attack)**: 적응된 모델이 테스트 데이터 전체에서 낮은 정확도를 갖도록 하여 전체적인 성능을 파괴하는 것이 목표이다.
- **Targeted Attack**: 특정 클래스 $t$에 대한 샘플들을 오분류하도록 유도하여 해당 클래스에 대한 모델의 지식을 파괴하는 것이 목표이다.

#### 2. 대리 테스트 손실 (Surrogate Test Loss)
실제 환경에서 공격자는 테스트 세트 $D_T^{test}$를 알 수 없으므로, 학습 세트 $D_T^{train}$에서의 경험적 손실을 대리 지표로 사용하여 목적 함수를 다음과 같이 통합한다.
$$\text{maximize}_{D_T^{adv}} \sum_{x,y \in D_T^{train}} [L(F(x; \phi'), y)] \quad \text{s.t. } \phi' = f_\theta(D_T^{adv})$$

#### 3. 인지 불가능한 섭동 (Unnoticeable Perturbation)
공격이 탐지되지 않도록 두 가지 제약 조건을 둔다.
- **Perturbed Samples Budget**: 조작하는 샘플의 개수를 $k$개 이하로 제한한다.
- **Perceptual Similarity**: 각 샘플의 섭동 크기를 $l_\infty$ norm 기준 $\epsilon$ 이하로 제한한다.
$$\|x^{adv} - clean(x^{adv})\| \le \epsilon, \forall x^{adv} \in D_T^{adv}$$

### MetaAttacker 알고리즘
MetaAttacker는 두 단계의 최적화 과정을 거친다.

**Step 1: 주어진 선택 세트에 대한 적대적 샘플 생성 (Alg 1)**
선택된 샘플 세트 $D_{select}$에 대해 Projected Gradient Ascent를 사용하여 손실을 최대화하는 섭동을 계산한다. 이때 그래디언트는 체인 룰(Chain Rule)을 통해 계산된다.
$$\nabla_{x_i^k} L_{total}(\phi^k) = \frac{\partial L_{total}(\phi^k)}{\partial \phi^k} \frac{\partial f_\theta(D_T^{adv})}{\partial x_i^k}$$
특히 MAML과 같이 여러 단계의 그래디언트 업데이트를 수행하는 모델의 경우, 고차 미분(Higher-order derivatives) 계산이 필수적이다.

**Step 2: 최적의 적대적 세트 탐색 (Alg 2)**
모든 조합을 탐색하는 것은 계산 비용이 너무 크므로, Greedy 알고리즘을 사용한다. 한 번에 하나의 샘플을 추가하며 가장 큰 손실 증가를 유발하는 샘플을 선택하여 $k$개의 세트를 구성한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Omniglot, Mini-ImageNet (5-way 5-shot 설정).
- **기준선**: Random Noise(무작위 섭동), Random F.T.(MAML의 가이드 없이 무작위 초기값에서 파인튜닝).
- **지표**: 평균 테스트 정확도(Average Test Accuracy).

### 주요 결과
1. **MAML에 대한 공격**:
   - Mini-ImageNet에서 MAML의 1-step 적응 모델의 경우, 10개의 샘플만 조작해도 정확도가 $63.3\% \to 16.2\%$로 급락하였다. 이는 Random F.T.($23.4\%$)보다 더 낮은 수치로, 메타 학습의 가이드 자체가 독이 되었음을 의미한다.
   - 파인튜닝 단계($m$)가 증가할수록 강건성이 다소 향상되는 경향을 보였다.
2. **Targeted Attack (Direct vs Influence)**:
   - **Direct Attack**: 타겟 클래스의 샘플을 직접 조작하는 방식이며 가장 강력했다.
   - **Influence Attack**: 타겟 클래스가 아닌 다른 클래스의 샘플을 조작하여 타겟 클래스의 성능을 떨어뜨리는 방식이다. 실험 결과, 타겟과 무관한 클래스를 조작해도 타겟 클래스의 인지 능력이 저하됨을 확인하였다.
3. **타 모델 분석**:
   - **SNAIL (Model-based)**: Non-targeted 및 Targeted 공격 모두에 매우 취약하였다.
   - **Prototypical Networks (Metric-based)**: Direct attack에는 취약하지만, Influence attack에는 상대적으로 강건한 모습을 보였다.

## 🧠 Insights & Discussion

본 논문은 Meta Learning이 기존의 일반적인 딥러닝 모델보다 데이터 오염(Data Poisoning)에 더 취약할 수 있다는 중요한 통찰을 제공한다. 그 이유는 다음과 같다.
첫째, 메타 학습은 극히 적은 수의 샘플에 의존하여 빠르게 적응해야 하므로, 단 몇 개의 샘플만 오염되어도 모델의 방향성이 완전히 틀어질 가능성이 크다. 
둘째, 메타 학습의 아키텍처 자체가 공격자에게 효율적인 섭동을 삽입할 수 있는 명확한 가이드를 제공하는 구조를 가지고 있다.

특히 MAML에서 고차 미분을 통한 공격이 효과적이었다는 점은, 메타 학습의 최적화 경로(Optimization path) 자체가 공격의 통로가 될 수 있음을 시사한다. 결과적으로, 안전이 중요한 시스템에 메타 학습을 적용하기 위해서는 단순한 성능 향상뿐만 아니라, 적응 단계에서의 강건성을 확보하는 방어 기제 연구가 반드시 선행되어야 한다.

## 📌 TL;DR

본 논문은 Meta Learning이 소수의 학습 데이터 조작만으로도 매우 쉽게 파괴될 수 있음을 증명한 연구이다. 저자들은 메타 학습의 특성을 반영한 적대적 공격 알고리즘인 **MetaAttacker**를 제안하고, 이를 MAML, SNAIL, ProtoNet 등에 적용하여 성능을 급격히 저하시킬 수 있음을 보였다. 특히 타겟 클래스가 아닌 다른 클래스를 조작해도 해당 클래스의 성능이 떨어지는 'Influence Attack'의 가능성을 확인하여, 메타 학습의 신뢰성 문제에 경종을 울렸다. 이 연구는 향후 강건한 메타 학습(Robust Meta Learning) 모델 설계의 필요성을 제시하는 중요한 기초 연구가 될 것이다.