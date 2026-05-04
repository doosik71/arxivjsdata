# A gray-box approach for curriculum learning

Francesco Foglino, Matteo Leonetti, Simone Sagratella, and Ruggiero Seccia (2019)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)에서 에이전트가 더 빠르게 최적의 행동을 학습하도록 돕는 Curriculum Learning(CL)의 자동화 문제를 다룬다. Curriculum Learning은 덜 복잡한 작업에서 시작하여 점진적으로 난이도를 높여가는 방식으로, 탐색 정책의 효율성을 높이고 전이 학습(Transfer Learning) 및 일반화 능력을 향상시킬 수 있다.

그러나 현재 대부분의 Curriculum은 인간의 직관에 의존하여 수동으로 설계(hand-designed)된다. 자동화된 작업 시퀀싱 방법들이 제안되었으나, 이들은 주로 휴리스틱한 솔루션을 제공할 뿐 결과의 품질에 대한 보장이 부족하다. 본 논문의 목표는 Curriculum Learning 문제를 수학적으로 재정의하고, 이를 효율적으로 해결하기 위한 Gray-box 접근 방식의 수치적 최적화 방법을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 조합 최적화 문제인 Curriculum 설계 문제를 **Gray-box 함수**를 통한 매개변수 최적화 문제로 변환하는 것이다.

기존 방식이 가능한 모든 작업 순서(Curriculum)를 직접 탐색했다면, 본 논문은 각 작업의 효용성(Utility, $u$)과 작업 간의 선행 관계에 따른 페널티(Penalty, $p$)라는 연속적인 매개변수 공간을 정의한다. 이 매개변수들을 입력으로 하여 정수 선형 계획법(Integer Linear Program, ILP) 기반의 스케줄링 문제를 해결하면 최적의 Curriculum이 도출된다. 즉, "매개변수 $\to$ 스케줄링 문제 해결 $\to$ Curriculum 도출 $\to$ 최종 성능 측정"으로 이어지는 구조를 통해, 불연속적인 조합 문제를 상대적으로 다루기 쉬운 최적화 문제로 재구성한 것이 핵심 기여이다.

## 📎 Related Works

논문에서는 Curriculum Learning을 위한 기존의 수치적 방법들이 주로 휴리스틱에 의존하며, 품질 보장이 어렵다는 점을 지적한다. 특히 다음과 같은 한계점을 가진 기존 접근 방식들을 언급한다:

1. **Mixed-Integer NonLinear Programs (MINLP):** 목적 함수가 명시적이지 않고 비매끄러우며(non-smooth), 비볼록(non-convex)하고 불연속적인 Black-box 특성을 가지므로 표준적인 MINLP 방법론을 적용할 수 없다.
2. **Derivative-Free (DF) methods:** 가능한 Curriculum의 집합이 조합론적(combinatorial) 구조를 가지므로, 일반적인 DF 알고리즘을 직접 적용하기 어렵다.

본 논문은 이러한 한계를 극복하기 위해, 내부적으로는 ILP(정수 선형 계획법)를 사용하여 구조적 제약을 해결하고 외부적으로는 DF 최적화를 사용하는 Gray-box 방식을 제안하여 기존 연구들과 차별화를 둔다.

## 🛠️ Methodology

### 1. 강화학습 배경 및 Regret 정의

에이전트는 MDP 환경 $m$에서 정책 $\pi$에 따라 행동한다. 가치 함수 $q^\pi(s, a)$는 선형 근사 $\hat{q}^\pi(s, a; \theta) = \sum_{k=1}^K \theta_k \phi_k(s, a)$를 사용하며, 여기서 $\theta$는 최적화 대상인 파라미터이다.

최종 작업 $m_L$에서의 성능을 측정하기 위해 **Regret 함수** $Pr(c)$를 도입한다. Curriculum $c$를 통해 학습된 초기점 $\theta_L(c)$에서 시작하여 $N_{m_L}$ 에피소드 동안 학습했을 때, 목표 성능 임계값 $g$와 실제 달성 성능 $\psi_{m_L}$ 사이의 차이를 합산한 값이다.

$$Pr(c) = \sum_{i=1}^{N_{m_L}} (g - \psi_{m_L}(\theta_{L+(i/N_{m_L})}(c)))$$

최종 목표는 $Pr(c)$를 최소화하는 최적의 Curriculum $c$를 찾는 것이다.

### 2. 스케줄링 문제로의 재구성 (The Scheduling Problem)

Curriculum 선택 문제를 해결하기 위해 다음과 같은 변수를 도입한다:

- $\delta_i \in \{0, 1\}$: $i$번째 작업이 Curriculum에 포함되면 1, 아니면 0.
- $\gamma_{ij} \in \{0, 1\}$: $i$번째 작업이 $j$번째 작업보다 먼저 스케줄링되면 1, 아니면 0.

이때 각 작업의 개별 효용 $u_i$와 작업 간 순서에 따른 페널티 $p_{ij}$를 사용하여 merit 함수 $\hat{U}$를 정의한다:

$$\hat{U}(\delta, \gamma; u, p) = \sum_{i=1}^n u_i \delta_i - \sum_{i=1}^n \sum_{i \neq j=1}^n p_{ij} \gamma_{ij}$$

이를 최대화하는 정수 선형 계획법(ILP) 문제는 다음과 같다:
$$\text{maximize}_{x, \delta, \gamma} \quad \hat{U}(\delta, \gamma; u, p)$$
$$\text{subject to} \quad x_i \geq (L-1)(1-\delta_i), \quad i=1, \dots, n$$
$$x_i + \delta_j \leq x_j + L\gamma_{ji}, \quad i, j=1, \dots, n, i \neq j$$
$$\gamma_{ij} + \gamma_{ji} \leq 1, \quad x \in [0, L-1]^n \cap \mathbb{Z}^n, \delta \in \{0, 1\}^n, \gamma \in \{0, 1\}^{n \times (n-1)}$$

여기서 $x_i$는 작업의 순서를 나타내는 변수이다.

### 3. Gray-box 함수 및 최적화 전략

본 논문은 매개변수 $(u, p)$를 입력받아 위 ILP 문제를 풀어 Curriculum $c$를 생성하고, 그 결과로 얻은 Regret $Pr(c)$를 반환하는 **Gray-box 함수 $\Psi(u, p)$**를 정의한다. 따라서 문제는 다음과 같이 단순화된다:

$$\min_{(u, p) \in \mathbb{R}^n_+ \times \mathbb{R}^{n \times (n-1)}_+} \Psi(u, p)$$

이 함수를 최적화하기 위해 세 가지 수치적 방법을 제안한다:

1. **SMBO (Sequential Model-Based Optimization):** Gaussian Process (GP)를 사용하여 $\Psi$의 대리 모델을 구축하고, Expected Improvement (EI)를 최대화하는 지점을 탐색한다.
2. **Heuristic Estimate:** 가정 (A1)을 바탕으로 개별 작업 및 쌍별 작업의 성능 데이터를 이용해 $u$와 $p$의 초기 추정치를 계산한다.
3. **TPE (Tree-structured Parzen Estimator):** 휴리스틱으로 얻은 추정치를 중심으로 가우시안 분포를 설정하고, 이 영역 내에서 지역 탐색을 수행하는 SMBO 방법이다.

## 📊 Results

### 실험 설정

- **환경:** GridWorld (보물 찾기, 불과 구덩이 회피).
- **비교 대상:**
  - $C_0$: Curriculum 없이 직접 학습.
  - GREEDY Par: 성능 향상이 큰 작업을 점진적으로 추가하는 기존 벤치마크 알고리즘.
  - GP: 사전 지식 없이 Gaussian Process 기반 최적화 수행.
  - Heuristic: 제안된 수식 (6), (7)을 이용한 추정치 사용.
  - TPE: 휴리스틱 추정치를 중심으로 지역 탐색 수행.
- **평가 지표:** Regret ($Pr$) 및 전체 가능한 Curriculum 중의 순위(Rank).

### 주요 결과

두 가지 시나리오($n=12, L=4$ 및 $n=7, L=7$)에서 실험을 진행한 결과는 다음과 같다.

| 알고리즘 | $n=12, L=4$ ($Pr$) | Rank | $n=7, L=7$ ($Pr$) | Rank |
| :--- | :---: | :---: | :---: | :---: |
| $C_0$ | -0.6389 | 11499 | -0.5051 | 4535 |
| GREEDY Par | -0.7765 | 144 | -0.6113 | 260 |
| GP | -0.7882 | 32 | -0.6511 | 38 |
| Heuristic | -0.7773 | 121 | -0.5966 | 417 |
| **TPE** | **-0.8025** | **4** | **-0.6697** | **14** |
| $Pr^*$ (Optimal) | -0.8149 | 1 | -0.7224 | 1 |

- 모든 제안 방법이 $C_0$보다 우수한 성능을 보였다.
- 휴리스틱 방법은 GREEDY Par와 유사하거나 더 나은 성능을 보이며 효율적인 초기점을 제공한다.
- 특히 **TPE**는 휴리스틱 추정치를 기반으로 최적화를 수행했을 때, 수만 개의 가능한 조합 중 상위 15위 이내의 매우 뛰어난 Curriculum을 찾아내어 가장 우수한 성능을 기록했다.

## 🧠 Insights & Discussion

본 논문은 조합 최적화 문제인 Curriculum Learning을 연속적인 매개변수 공간의 최적화 문제로 변환함으로써, 복잡한 탐색 공간을 효율적으로 다룰 수 있음을 보여주었다.

특히 주목할 점은 **Gray-box 구조**의 효율성이다. 내부의 ILP는 제약 조건을 엄격하게 준수하는 Curriculum을 생성하고, 외부의 SMBO(GP, TPE)는 매개변수 공간을 효율적으로 탐색함으로써 두 방식의 장점을 모두 취했다. 또한, 단순한 휴리스틱 추정치만으로도 상당한 성능 향상을 얻을 수 있으며, 이를 기반으로 TPE와 같은 지역 탐색을 결합했을 때 최적해에 매우 근접한 결과를 얻을 수 있다는 점이 입증되었다.

다만, 본 연구는 GridWorld라는 비교적 단순한 벤치마크 환경에서 검증되었다. 실제 매우 복잡한 Deep RL 환경에서도 휴리스틱 기반의 $u, p$ 추정이 유효할지, 그리고 작업의 수가 급격히 증가할 때 ILP 해결 시간이 어떻게 변할지에 대한 분석은 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

이 논문은 Curriculum Learning의 작업 순서 결정 문제를 **'효용(Utility)과 페널티(Penalty) 매개변수 최적화 $\to$ ILP 스케줄링 $\to$ Curriculum 도출'**이라는 Gray-box 프레임워크로 재정의했다. 이를 통해 조합론적 탐색 문제를 연속 공간의 최적화 문제로 바꾸었으며, 특히 휴리스틱 추정치와 TPE(Tree-structured Parzen Estimator)를 결합했을 때 매우 효율적으로 최적의 학습 경로를 찾을 수 있음을 증명했다. 이 연구는 향후 복잡한 RL 작업에서 인간의 개입 없이 최적의 학습 커리큘럼을 자동으로 생성하는 시스템 구축에 중요한 기여를 할 수 있다.
