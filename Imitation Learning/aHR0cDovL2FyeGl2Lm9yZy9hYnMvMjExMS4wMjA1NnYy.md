# Curriculum Offline Imitation Learning

Minghuan Liu, Hanye Zhao, Zhengyu Yang, Jian Shen, Weinan Zhang, Li Zhao, Tie-Yan Liu (2021)

## 🧩 Problem to Solve

본 논문은 환경과의 추가적인 상호작용 없이 사전 수집된 데이터셋만으로 학습해야 하는 Offline Reinforcement Learning (RL)에서 발생하는 효율성과 안정성 문제를 해결하고자 한다. 

기존의 Offline RL 접근 방식은 크게 두 가지 갈래로 나뉘는데, 각각 뚜렷한 한계를 가진다. 첫째, RL 기반 방법론(예: Q-learning)은 행동 정책(behavioral policy)보다 더 나은 성능을 낼 잠재력이 있지만, 가치 함수를 추정하는 과정에서 발생하는 Bootstrapping error와 Extrapolation error로 인해 학습이 매우 불안정하며, 최적의 하이퍼파라미터를 찾기 위해 잦은 Online evaluation이 필요하다는 실무적 단점이 있다. 둘째, Offline Imitation Learning(IL), 특히 Behavior Cloning (BC)은 가치 함수를 추정하지 않으므로 학습이 안정적이지만, 데이터셋이 여러 정책의 혼합물(mixture of policies)일 때 평균적인 성능의 '어중간한(mediocre)' 행동만을 학습하는 경향이 있다.

특히 저자들은 혼합 데이터셋에서 **양과 질의 딜레마(Quantity-Quality Dilemma)**를 지적한다. 고품질의 데이터만 선택하면 데이터의 양이 부족하여 Compounding error 문제가 발생하고, 데이터의 양을 늘리면 평균적인 품질이 낮아져 결국 성능이 저하되는 현상이다. 따라서 본 논문의 목표는 BC의 안정성을 유지하면서도, 혼합 데이터셋 내에서 최적의 정책을 효과적으로 추출하여 학습할 수 있는 실용적인 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **초기 정책과 타겟 정책 사이의 거리(discrepancy)가 가까울수록 BC가 더 적은 데이터로도 효과적으로 학습할 수 있다**는 관찰에서 시작된다. 이를 바탕으로 제안된 **Curriculum Offline Imitation Learning (COIL)**의 중심 설계 아이디어는 다음과 같다.

1.  **적응형 이웃 정책 모방(Adaptive Neighboring Policy Imitation):** 한 번에 최적의 정책을 모방하려 하지 않고, 현재 정책 $\pi$와 유사하면서도 더 높은 보상을 주는 '이웃 정책'을 단계적으로 모방하며 성능을 개선하는 커리큘럼 구조를 설계한다.
2.  **경험 선택 전략(Experience Picking Strategy):** 현재 정책이 생성할 법한 확률이 높은 궤적(trajectory)을 데이터셋에서 선택하여 학습 타겟으로 삼는다.
3.  **리턴 필터(Return Filter):** 학습 과정에서 현재 수준보다 낮은 보상을 가진 궤적들을 제거함으로써, 정책의 성능이 퇴보하는 것을 방지하고 점진적으로 더 높은 보상을 가진 데이터로 학습 범위를 좁혀나간다.

## 📎 Related Works

논문에서는 Offline RL을 Model-free 방식의 RL 기반 방법과 IL 기반 방법으로 나누어 설명한다.

-   **RL-based Methods:** BCQ, BEAR, CQL 등이 대표적이다. 이들은 Q-함수에 제약을 가하거나 보수적인 추정치를 사용하여 Out-of-distribution (OOD) 액션으로 인한 과대평가 문제를 해결하려 한다. 하지만 여전히 Bootstrapping error와 학습 불안정성 문제가 존재하며, 하이퍼파라미터 튜닝에 대한 의존도가 높다.
-   **Imitation-based Methods:** BC를 기본으로 하며, BAIL, ABM, MARWIL, AWR 등이 있다. 이들은 데이터에 가중치를 부여하거나 가치 함수(Value function)를 회귀 분석하여 최적의 액션을 선택하려 한다. 그러나 가치 함수 추정 과정에서 불안정성이 발생하며, 이 또한 많은 하이퍼파라미터 튜닝을 요구한다.

**COIL과의 차별점:** COIL은 가치 함수를 직접 회귀 분석하거나 추정하지 않는다. 대신 정책 간의 거리와 보상 기반의 필터링이라는 단순한 커리큘럼 전략을 사용하여, 가치 함수 추정으로 인한 불안정성을 원천적으로 제거하면서도 BC의 한계를 극복한다.

## 🛠️ Methodology

### 전체 파이프라인
COIL은 데이터셋 $D$에서 현재 정책 $\pi_i$와 유사한 궤적들을 선택해 학습하고, 정책을 $\pi_{i+1}$로 업데이트하는 과정을 반복하는 커리큘럼 학습 구조를 가진다. 전체 흐름은 [이웃 궤적 선택 $\to$ BC 학습 $\to$ 리턴 필터 업데이트 $\to$ 데이터셋 정제] 순으로 이루어진다.

### 주요 구성 요소 및 절차

#### 1. 이웃 정책 평가 (Neighboring Policy Assessment)
현재 정책 $\pi$가 특정 궤적 $\tau_{\tilde{\pi}}$를 생성했을 가능성이 높은지 판단한다. 이론적으로는 $\text{KL}$ divergence를 사용해야 하지만, 실무적 효율성을 위해 다음과 같은 확률 임계치 기반의 지표를 사용한다.

궤적 $\tau_{\tilde{\pi}}$ 내의 상태-액션 쌍 $(s, a)$들에 대해, 현재 정책 $\pi$가 해당 액션을 선택할 확률 $\pi(a|s)$가 특정 임계값 $\epsilon_c$보다 큰 비율이 $1-\beta$ 이상인지를 확인한다.
$$E_{(s,a) \in \tau_{\tilde{\pi}}} [I(\pi(a|s) \ge \epsilon_c)] \ge 1-\beta$$
여기서 $\beta$는 탐색 노이즈(exploration ratio)를 고려한 값(기본값 0.05)이며, $\epsilon_c$는 현재 정책과 가장 유사한 $N$개의 궤적을 찾음으로써 적응적으로 결정된다.

#### 2. 리턴 필터 (Return Filter)
현재 정책의 성능보다 낮은 품질의 데이터를 학습하여 성능이 하락하는 것을 방지하기 위해 리턴 필터 $V$를 운영한다. 필터 값 $V$는 현재 커리큘럼 단계에서 선택된 궤적들의 최소 보상을 사용하여 지수 이동 평균(Moving Average) 방식으로 업데이트된다.
$$V^k = (1-\alpha) \cdot V^{k-1} + \alpha \cdot \min \{R(\tau)\}_{n=1}^N$$
여기서 $\alpha$는 필터링 속도를 결정하는 윈도우 크기이다. 업데이트 후, 데이터셋 $D$에서 $R(\tau) < V$인 모든 궤적을 제거한다.
$$D = \{\tau \in D \mid R(\tau) \ge V\}$$

#### 3. 학습 절차 (Training Procedure)
1.  데이터셋 $D$가 빌 때까지 반복한다.
2.  현재 정책 $\pi$를 기준으로 위 식의 조건을 만족하는 상위 $N$개의 궤적을 선택하여 커리큘럼 세트를 구성한다.
3.  선택된 궤적들을 사용하여 표준 Behavior Cloning (BC) 손실 함수를 최소화하도록 정책을 업데이트한다.
    $$\min_{\theta} \sum_{(s,a) \in B} [-\log \pi_\theta(a|s)]$$
4.  리턴 필터 $V$를 업데이트하고 데이터셋 $D$에서 저품질 궤적을 제거한다.
5.  더 이상 선택할 수 있는 궤적이 없으면 학습을 종료한다.

## 📊 Results

### 실험 설정
-   **데이터셋:** Online agent가 학습하며 수집한 `final-buffer` 데이터셋과 D4RL 벤치마크 데이터셋을 사용하였다.
-   **환경:** Hopper, Walker2d, HalfCheetah (Continuous Control tasks).
-   **비교 대상:** BC, CQL(SOTA RL-based), AWR, BAIL(Imitation-based).
-   **지표:** 평균 누적 보상(Average Return).

### 주요 결과
1.  **Final Buffer 데이터셋:** COIL은 BC보다 압도적으로 높은 성능을 보였으며, 특히 CQL, AWR, BAIL과 비교했을 때도 매우 경쟁력 있거나 더 높은 성능을 기록하였다 (Table 1 참조). 
2.  **학습 경로 분석:** COIL이 선택하는 궤적의 순서가 실제 Online agent가 학습하며 성능을 올린 순서와 매우 유사함을 확인하였다. 이는 COIL이 데이터셋 내에서 적절한 난이도의 커리큘럼을 스스로 구축하고 있음을 시사한다.
3.  **D4RL 벤치마크:** 대부분의 데이터셋에서 최적의 행동 정책(Best 1%)의 성능에 근접하는 결과를 보였다. 특히 HalfCheetah의 경우, 매우 정교한 튜닝을 거친 RL 기반 방법론과 경쟁 가능한 수준의 성능을 달성하였다 (Table 2 참조).
4.  **효율성:** COIL은 다른 베이스라인들에 비해 훨씬 적은 Gradient Step만으로도 최적 정책에 도달하여 학습 효율성이 매우 높음을 보였다.

## 🧠 Insights & Discussion

### 강점
-   **안정성과 단순성:** 가치 함수 추정 과정이 없으므로 RL 기반 방법론의 고질적인 문제인 학습 불안정성과 Bootstrapping error에서 자유롭다.
-   **자동 종료 조건:** 데이터셋 내에서 더 이상 모방할 더 나은 이웃 정책이 없을 때 자동으로 학습이 종료되므로, Online evaluation을 통해 최적의 체크포인트를 일일이 찾을 필요가 없다. 이는 실제 배포 환경에서 매우 큰 이점이다.
-   **데이터 효율성:** 적응형 이웃 선택 전략을 통해 데이터의 '양'과 '질' 사이의 딜레마를 효과적으로 해결하였다.

### 한계 및 논의
-   **데이터 의존성:** 본 방법론은 결국 데이터셋 내에 존재하는 최적 정책을 모방하는 것이 목표이므로, 데이터셋 자체의 최대 성능(Optimal behavior policy)을 초과하는 성능을 내기는 어렵다. 이는 데이터셋의 한계를 넘어서려는 RL 기반 방법론과의 근본적인 차이점이다.
-   **하이퍼파라미터:** $\alpha$(필터 윈도우)와 $N$(선택 궤적 수)이 성능에 중요한 영향을 미치며, 이는 데이터셋의 분포 특성(궤적 간 보상 차이 등)에 따라 조정되어야 한다.

## 📌 TL;DR

본 논문은 Offline RL에서 BC의 안정성과 RL의 성능 향상 잠재력을 결합한 **COIL (Curriculum Offline Imitation Learning)**을 제안한다. 정책 간의 거리를 이용한 **적응형 경험 선택(Adaptive Experience Picking)**과 **리턴 필터(Return Filter)**를 통해, 데이터셋 내의 저품질 데이터를 배제하고 단계적으로 더 나은 정책을 모방하는 커리큘럼 학습을 수행한다. 실험 결과, COIL은 복잡한 가치 함수 추정 없이도 SOTA Offline RL 방법론들에 필적하는 성능을 보였으며, 매우 안정적인 학습 프로세스를 제공한다. 이 연구는 특히 데이터셋의 품질이 혼재되어 있는 실제 환경에서 Offline RL을 실용적으로 적용할 수 있는 새로운 방향을 제시한다.