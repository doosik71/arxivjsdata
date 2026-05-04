# Learning to Search via Retrospective Imitation

Jialin Song, Ravi Lanka, Albert Zhao, Aadyot Bhatnagar, Yisong Yue, Masahiro Ono (2019)

## 🧩 Problem to Solve

본 논문은 조합 탐색 공간(Combinatorial Search Spaces)에서 효율적인 탐색 정책(Search Policy)을 학습하는 문제를 다룬다. 조합 탐색 문제(예: 경로 계획, 정수 계획법)는 문제의 규모가 커짐에 따라 탐색 공간이 지수적으로 증가하므로, 어떤 노드를 우선적으로 탐색할지를 결정하는 휴리스틱(Heuristic)의 성능이 매우 중요하다.

전통적으로는 인간 전문가가 설계한 수동 휴리스틱을 사용해 왔으나, 이는 노동 집약적이며 문제의 구조적 특성에 대한 깊은 이해를 요구한다. 이를 해결하기 위해 강화학습(Reinforcement Learning)을 적용할 수 있으나, 조합 탐색 문제에서는 유효한 해(Feasible Solution)에 도달했을 때만 보상이 주어지는 Sparse Reward 문제가 발생하여 학습이 매우 어렵다. 모방 학습(Imitation Learning)은 기존 솔버(Expert)의 데이터를 사용할 수 있어 유망하지만, 대규모 문제에 대해 전문가의 데이터를 생성하는 비용이 매우 비싸다는 한계가 있다.

따라서 본 논문의 목표는 전문가에게 반복적으로 의존하지 않고도, 정책이 자신의 실수로부터 스스로 학습하여 성능을 개선하고, 특히 초기 전문가 데이터가 제공된 문제 규모보다 더 큰 규모의 문제로 확장(Scale-up)할 수 있는 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Retrospective Imitation (회고적 모방 학습)**이다. 이는 정책이 생성한 Roll-out(탐색 경로)을 사후에 분석하여 더 나은 경로를 추출하고, 이를 다시 학습 데이터로 사용하는 방식이다.

핵심 직관은 다음과 같다. 정책이 탐색 과정에서 시행착오를 겪으며 백트래킹(Backtracking)을 반복하다가 결국 유효한 해에 도달했다면, 그 경로에서 백트래킹 부분을 제거함으로써 해에 도달하는 가장 짧은 경로(Retrospective Optimal Trace)를 찾을 수 있다. 정책은 이 '정제된 경로'를 모방함으로써 스스로 더 효율적인 탐색 방법을 학습한다.

특히, 이 방식은 작은 규모의 문제에서 학습된 정책을 기반으로 점진적으로 더 큰 규모의 문제에 적용하고 다시 학습하는 과정을 반복함으로써, 초기 전문가 데이터의 규모를 뛰어넘는 확장성(Scalability)을 제공한다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들과 차별점을 가진다.

1.  **Imitation Learning (모방 학습):** DAgger나 SMILe 같은 기존 기법들은 학습 과정에서 전문가(Teacher)에게 지속적으로 쿼리를 보내 피드백을 받아야 한다. 반면, Retrospective Imitation은 환경(Environment)에 대한 쿼리만으로 스스로 피드백을 생성하므로 전문가 의존도를 획기적으로 낮춘다.
2.  **Reinforcement Learning (강화학습):** RL은 Sparse Reward 환경에서 학습이 불안정하지만, 본 논문은 모방 학습의 구조를 사용하여 보상 신호를 밀집화(Densify)함으로써 이 문제를 해결한다.
3.  **Learning to Optimize:** 기존의 최적화 학습 연구들은 주로 고정된 크기의 문제나 특정 도메인에 집중했다. 본 논문은 Retrospective Imitation을 일종의 전이 학습(Transfer Learning)으로 활용하여 문제의 규모를 점진적으로 키워나가는 Scale-up 메커니즘을 제안했다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
Retrospective Imitation은 일반적인 프레임워크이며, DAgger나 SMILe 같은 다양한 모방 학습 알고리즘과 결합될 수 있다. 논문에서는 이를 **Retrospective DAgger**로 구체화하여 설명한다.

전체 프로세스는 다음과 같은 단계로 진행된다:
1.  **초기 학습:** 소규모 문제에 대해 전문가가 생성한 데이터($D_0$)로 초기 정책 $\pi_1$을 학습시킨다.
2.  **Roll-out 생성:** 현재 정책 $\pi_i$를 사용하여 훈련 문제 집합 $\{P_j\}$에 대해 탐색을 수행하고 search trace $\{\tau_j\}$를 생성한다.
3.  **Retrospective Oracle 적용:** 생성된 trace에서 유효한 해(Terminal state)에 도달한 경우, Retrospective Oracle을 통해 최적의 경로를 추출한다.
4.  **데이터셋 업데이트 및 재학습:** 추출된 최적 경로를 새로운 학습 데이터 $D_i$로 추가하고 $\pi_{i+1}$을 학습시킨다.

### Retrospective Oracle
Retrospective Oracle은 주어진 탐색 트리 $\tau$와 도달한 종단 상태 $s$를 입력으로 받아, 루트 노드에서 $s$까지의 최단 경로를 반환한다. 트리 구조의 탐색에서 이는 단순히 종단 상태에서 부모 포인터(Parent pointer)를 따라 루트까지 거슬러 올라가는 것과 같다. 이를 통해 백트래킹이 모두 제거된 $\pi^*(\tau, s)$를 얻게 된다.

### Scaling Up (규모 확장)
더 큰 규모의 문제를 해결하기 위해 다음과 같은 절차(Algorithm 3)를 따른다.
- 초기 규모 $S_1$에서 학습된 정책 $\pi_{S_1}$을 시작점으로 설정한다.
- 문제 규모를 $S_1+1, S_1+2, \dots, S_2$까지 점진적으로 증가시킨다.
- 각 단계 $s$에서, 이전 단계 $\pi_{s-1}$을 이용해 초기 trace를 생성하고, 이를 Retrospective DAgger로 학습시켜 $\pi_s$를 갱신한다.

### 이론적 분석
논문은 **Trace Inclusion Assumption**(학습된 정책의 trace가 전문가의 trace를 포함한다는 가정) 하에 다음을 증명한다.

1.  **에러율 ($\le$):** 정책이 Retrospective Oracle의 결정과 일치하지 않는 비율로 정의한다.
    $$\le = \frac{\text{Non-optimal actions compared to retrospective optimal trace}}{\text{Actions to reach a terminal state in retrospective optimal trace}}$$
2.  **기대 탐색 시간:** 결정 공간이 2가지(자식 노드로 이동 또는 부모로 백트래킹)인 경우, 종단 상태에 도달하기까지의 기대 액션 수 $E[T]$는 다음과 같다.
    $$E[T] = \frac{N}{1-2\le}$$
    여기서 $N$은 최적 경로의 길이이다. 이는 에러율 $\le$이 낮아질수록 탐색 시간이 지수적으로 빠르게 감소함을 시사한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업:**
    - **Maze Solving:** A* Search를 사용하여 $11 \times 11$에서 $31 \times 31$ 크기의 미로 해결.
    - **Risk-aware Path Planning:** 혼합 정수 선형 계획법(MILP)을 통해 장애물을 피하는 경로 탐색.
    - **Minimum Vertex Cover (MVC):** NP-hard 문제인 최소 정점 커버 문제를 ILP로 변환하여 해결.
- **비교 대상 (Baselines):**
    - **Extrapolation:** 소규모 데이터로만 학습 후 큰 문제에 바로 적용.
    - **Cheating:** 타겟 규모의 전문가 데이터를 모두 제공한 모방 학습.
    - **Commercial Solvers:** Gurobi, SCIP.
- **측정 지표:** 탐색한 노드 수(Explored nodes), 최적성 갭(Optimality Gap).

### 주요 결과
1.  **모방 학습 대비 성능:** 모든 설정에서 Retrospective Imitation이 Extrapolation보다 압도적으로 우수한 성능을 보였다. 특히 미로 찾기에서는 'Cheating' 모델보다도 더 적은 노드를 탐색하며 해를 찾았다.
2.  **확장성 (Scaling Up):** 전문가 데이터가 없는 대규모 문제에서도 Retrospective Imitation은 최적성 갭의 증가 폭이 매우 완만하게 유지되며 효과적인 전이 학습이 가능함을 보였다.
3.  **상용 솔버 대비 성능:** Risk-aware Path Planning과 MVC 문제에서, 동일한 노드 탐색 예산(Node budget)을 주었을 때 Retrospective DAgger가 Gurobi나 SCIP보다 낮은 최적성 갭을 기록했다.
4.  **정책 구조의 영향:** 'Select & Pruner'(노드 선택 및 가지치기 모두 학습)보다 'Select only'(노드 선택만 학습) 정책이 더 좋은 성능을 보였다. 이는 조합 탐색 문제에서 단순한 정책 구조가 학습에 더 유리할 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 모방 학습의 치명적인 약점인 '전문가 데이터 확보 비용'과 '확장성 부족' 문제를 **사후 분석(Retrospection)**이라는 간단하지만 강력한 아이디어로 해결하였다.

**강점:**
- 전문가의 지속적인 개입 없이도 환경과의 상호작용만으로 성능을 개선할 수 있다.
- 소규모 데이터셋만으로도 대규모 문제에 적용 가능한 정책을 학습시키는 Scale-up 경로를 제시하였다.
- 이론적 분석을 통해 에러율 감소가 실제 탐색 시간 감소로 이어짐을 수학적으로 뒷받침하였다.

**한계 및 논의사항:**
- **Trace Inclusion Assumption:** 이론적 증명을 위해 사용된 가정이 실제 환경에서 항상 성립하는지는 명확하지 않다. 정책이 완전히 잘못된 방향으로 탐색하여 해를 전혀 찾지 못할 경우, Retrospective Oracle이 작동할 수 없다.
- **탐색 전략의 의존성:** 논문에서도 언급되었듯, Scaling up 과정에서 $\epsilon$-greedy나 노이즈 주입 같은 탐색(Exploration) 전략이 필수적이다. 효율적인 탐색 전략이 없으면 로컬 옵티마에 빠지거나 새로운 해를 발견하지 못할 위험이 있다.
- **컴퓨팅 비용:** Retrospective Imitation 과정에서 많은 양의 Roll-out을 생성해야 하므로, 학습 단계에서의 계산 비용이 증가할 수 있다.

## 📌 TL;DR

본 논문은 조합 탐색 문제에서 전문가 데이터 없이 스스로 학습하는 **Retrospective Imitation** 프레임워크를 제안한다. 이 방법은 정책이 생성한 경로에서 백트래킹을 제거하여 최적 경로를 추출하고 이를 다시 학습에 이용함으로써, 전문가의 도움 없이도 성능을 개선하고 학습 데이터가 없는 더 큰 규모의 문제로 확장할 수 있게 한다. 실험 결과, 미로 찾기 및 정수 계획법 문제에서 상용 솔버(Gurobi, SCIP) 및 기존 모방 학습 기법보다 뛰어난 효율성을 입증하였다. 이는 향후 고비용의 전문가 데이터가 필요한 복잡한 최적화 문제의 학습 자동화에 중요한 기여를 할 것으로 보인다.