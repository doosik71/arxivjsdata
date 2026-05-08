# Gradual Domain Adaptation without Indexed Intermediate Domains

Hong-You Chen, Wei-Lun Chao (2021)

## 🧩 Problem to Solve

Unsupervised Domain Adaptation (UDA)는 레이블이 있는 소스 도메인($S$)에서 학습된 모델을 레이블이 없는 타겟 도메인($T$)에 적응시키는 것을 목표로 한다. 그러나 소스 도메인과 타겟 도메인 사이의 분포 차이(domain discrepancy)가 매우 클 경우, 기존의 UDA 방식(예: self-training)은 부정확한 pseudo-label을 생성하게 되어 성능이 급격히 저하되는 문제가 발생한다.

이를 해결하기 위해 소스와 타겟 사이를 점진적으로 연결하는 추가적인 unlabeled 데이터(intermediate domains)를 활용하는 Gradual Domain Adaptation (GDA)이 제안되었다. GDA는 도메인 간의 간극을 여러 개의 작은 단계로 쪼개어 적응시킴으로써 전체적인 적응 성능을 향상시킨다. 하지만 기존 GDA는 중간 도메인들이 이미 그룹화되어 있고, 소스에서 타겟으로 향하는 순서(index)가 정의되어 있다는 강한 가정을 전제로 한다(예: 시간 태그 활용). 실제 환경에서는 이러한 index 정보가 없거나 부정확한 경우가 많아 GDA의 적용 가능성이 제한된다는 문제가 있다.

본 논문의 목표는 **중간 도메인에 대한 index 정보가 전혀 없는 상황에서, 데이터를 이용해 스스로 최적의 도메인 시퀀스를 발견하고 이를 통해 GDA를 수행하는 프레임워크를 구축하는 것**이다.

## ✨ Key Contributions

본 논문은 중간 도메인 레이블러인 **IDOL (IntermediateDOmainLabeler)**이라는 coarse-to-fine 프레임워크를 제안한다. 핵심 아이디어는 다음과 같다.

1. **Coarse-to-Fine 구조**: 먼저 도메인 판별기를 통해 데이터의 대략적인 위치를 파악하는 coarse stage를 거친 후, 지식 보존 능력을 측정하는 cycle-consistency loss를 통해 이를 정밀하게 조정하는 fine stage를 수행한다.
2. **Progressive Domain Discriminator**: 단순히 소스와 타겟을 구분하는 것을 넘어, 판별기가 예측한 점수가 높거나 낮은 데이터를 점진적으로 학습 데이터에 추가하며 훈련함으로써 데이터 매니폴드(manifold)를 더 정확하게 캡처한다.
3. **Cycle-Consistency 기반의 정밀 조정**: 현재 도메인에서 다음 도메인으로 적응했다가 다시 현재 도메인으로 돌아왔을 때, 원래의 예측값과 일치해야 한다는 cycle-consistency 원리를 도입하여, 판별적 지식(discriminative knowledge)을 가장 잘 보존하는 데이터 그룹을 다음 도메인으로 선택한다.
4. **Meta-Reweightng 적용**: 이산적인 도메인 선택 문제를 연속적인 가중치 최적화 문제로 완화하여 미분 가능한 형태로 해결함으로써 효율적인 도메인 시퀀스 발견을 가능하게 한다.

## 📎 Related Works

- **Unsupervised Domain Adaptation (UDA)**: 도메인 간의 divergence를 최소화하거나, domain-invariant feature를 학습하는 방식이 주를 이룬다. 최근에는 pseudo-label을 이용한 self-training 방식이 효과적임이 입증되었으나, 도메인 간극이 클 때 취약하다는 한계가 있다.
- **Gradual Domain Adaptation (GDA)**: 소스와 타겟 사이의 가교 역할을 하는 중간 도메인을 활용한다. 기존 연구들은 주로 합성 데이터를 생성하거나, 시간(time)과 같은 부가 정보(side information)를 통해 도메인 순서를 정의했다. 본 논문은 이러한 부가 정보 없이 데이터 자체에서 순서를 찾아낸다는 점에서 차별점을 가진다.
- **Cycle Consistency**: CycleGAN 등에서 사용된 개념으로, 변환 후 다시 역변환했을 때 원래 상태로 돌아와야 한다는 제약 조건을 통해 비지도 학습의 안정성을 높인다. 본 논문은 이를 도메인 시퀀스 발견이라는 새로운 문제에 적용하였다.

## 🛠️ Methodology

IDOL은 크게 두 단계(Coarse stage $\rightarrow$ Fine stage)로 구성된다.

### 1. Coarse Stage: Domain Scores 할당

각 중간 데이터 $x_i \in U$에 대해 소스에 가까운지 타겟에 가까운지를 나타내는 점수 $q_i$를 부여한다.

- **Progressive Domain Discriminator**: 이진 분류기 $g(\cdot; \phi)$를 학습시켜 소스(class 1)와 타겟(class 0)을 구분한다. 손실 함수는 Binary Cross Entropy (BCE)를 사용한다.
    $$L(\phi) = -\frac{1}{|S|}\sum_{x^S \in S} \log(\sigma(g(x^S; \phi))) - \frac{1}{|T|}\sum_{x^T \in T} \log(1-\sigma(g(x^T; \phi)))$$
- **점진적 학습 절차**:
    1. $S$와 $T$로 $g$를 학습시킨다.
    2. $U$의 데이터들에 대해 점수 $\hat{q}_i = g(x_i, \phi)$를 예측한다.
    3. 점수가 가장 높은 상위 그룹을 $S$에, 가장 낮은 하위 그룹을 $T$에 추가하고 다시 학습한다.
    4. 이 과정을 $K$번 반복하여, 데이터 매니폴드를 따라 점진적으로 확장된 신뢰도 높은 점수를 획득한다.

### 2. Fine Stage: Cycle-Consistency를 통한 정밀 조정

단순한 분포 일치만으로는 부족하며, 이전 도메인의 판별적 지식을 보존하는 것이 중요하다.

- **핵심 직관**: 모델 $\theta_m$이 도메인 $U_{m+1}$로 적응하여 $\theta_{m+1}$이 되었을 때, 다시 $U_m$으로 역적응시킨 모델 $\theta'_m$이 원래의 $\theta_m$과 유사한 예측을 한다면, $U_{m+1}$이 지식을 잘 보존하고 있다고 판단한다.
- **최적화 목표**:
    $$\arg \min_{U_{m+1} \subset U \setminus \cup_{j=1}^m U_j} \frac{1}{|U_m|} \sum_{x \in U_m} \ell(f(x; \theta'_m), \text{sharpen}(f(x; \theta_m)))$$
    여기서 $\theta_{m+1} = ST(\theta_m, U_{m+1})$ 이고, $\theta'_m = ST(\theta_{m+1}, U_m)$ 이다 ($ST$는 self-training 프로세스).

- **Meta-Reweightng을 이용한 구현**:
    데이터 선택을 위한 이진 벡터 $q \in \{0,1\}^N$을 실수 벡터 $q \in \mathbb{R}^N$으로 완화(relaxation)하여 학습한다.
    1. **Forward**: $q$를 가중치로 사용하여 $\theta_m$에서 $\theta(q)$로 적응시킨다.
    2. **Backward**: $\theta(q)$에서 다시 $\theta'$로 $U_m$ 데이터를 통해 역적응시킨다.
    3. **Update**: $\theta'$와 $\theta_m$ 사이의 오차에 대한 $\nabla_q$를 계산하여 $q$를 업데이트한다.
    4. 최종적으로 $q$값이 높은 상위 데이터를 $U_{m+1}$로 선택한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Rotated MNIST (회전 각도에 따른 shift), Portraits (연도별 인물 사진의 스타일 shift), CIFAR10-STL (Open-domain 데이터 활용).
- **비교 대상**: Source only, UDA (Direct), Random sequence GDA, Pre-defined index GDA.
- **평가 지표**: 타겟 도메인에서의 정확도 (Target Accuracy).

### 주요 결과

1. **Pre-defined Sequence와의 경쟁력**: IDOL은 index 정보 없이도 pre-defined index를 사용한 GDA와 대등하거나 오히려 더 높은 성능을 보였다. 특히 Rotated MNIST에서 pre-defined index보다 약간 더 높은 성능을 보였는데, 이는 실제 이미지의 미세한 회전 차이를 IDOL이 더 잘 포착했기 때문으로 분석된다.
2. **Refinement의 효과**: Coarse score만 사용했을 때보다 cycle-consistency refinement를 거쳤을 때 성능이 일관되게 향상되었다.
3. **강건성 (Robustness)**:
    - 중간 데이터가 부족(30%만 존재)하거나, 일부 데이터에 noise/outlier가 섞여 있는 경우에도 IDOL은 타 방식보다 안정적으로 성능을 향상시켰다.
    - CIFAR10-STL 실험에서 ImageNet의 unlabeled 데이터를 활용했을 때, 단순 UDA보다 IDOL 기반의 GDA가 월등한 성능을 보였다.

## 🧠 Insights & Discussion

- **시간 순서 vs 데이터 순서**: Portraits 데이터셋 분석 결과, 단순히 '연도(year)' 순으로 정렬한 것보다 IDOL이 발견한 시퀀스가 더 매끄러운 성능 향상을 보였다. 이는 패션, 헤어스타일 등 실제 도메인 변화가 연도라는 단순 지표와 완벽히 일치하지 않으며, IDOL이 데이터 내부의 실제 특징 변화(smoother transition)를 더 잘 포착했음을 시사한다.
- **클래스 균형**: Fine-grained index로 조정할수록 각 도메인 내의 클래스 분포가 더 균형 있게(class-balanced) 형성되는 경향이 확인되었다.
- **한계점 및 가정**: 본 방법론은 추가 데이터가 소스와 타겟 사이를 '점진적으로' 연결하고 있다는 가정을 전제로 한다. 만약 추가 데이터가 소스/타겟과 완전히 동떨어진 outlier이거나 bridge 역할을 하지 못한다면 성능 저하가 발생할 수 있다.

## 📌 TL;DR

본 논문은 index 정보가 없는 unlabeled 데이터들로부터 최적의 도메인 적응 순서를 스스로 찾아내는 **IDOL** 프레임워크를 제안한다. **Progressive Domain Discriminator**를 통해 대략적인 순서를 정하고, **Cycle-Consistency**와 **Meta-Reweightng**을 통해 이를 정밀하게 조정함으로써, 기존의 pre-defined index 기반 GDA에 필적하는 성능을 달성했다. 이 연구는 실제 환경에서 index 정보 없이도 GDA를 적용할 수 있는 길을 열어, UDA의 적용 범위를 크게 확장시켰다는 점에서 중요한 의미를 가진다.
