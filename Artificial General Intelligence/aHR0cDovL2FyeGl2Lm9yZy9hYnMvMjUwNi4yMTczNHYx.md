# Hierarchical Reasoning Model

Guan Wang et al. (2025)

## 🧩 Problem to Solve

본 논문은 현대 거대 언어 모델(LLM)이 가진 근본적인 구조적 한계인 '얕은 계산 깊이(shallow computational depth)' 문제를 해결하고자 한다. 표준 Transformer 아키텍처는 고정된 층(layer) 수를 가지며, 이는 계산 복잡도 측면에서 $AC^0$ 또는 $TC^0$ 클래스에 속해 다항 시간(polynomial time)이 필요한 복잡한 알고리즘적 추론 문제를 해결하는 데 한계가 있다.

현재 LLM은 이를 극복하기 위해 Chain-of-Thought(CoT) 기법을 사용하여 추론 과정을 외부 토큰으로 명시화하지만, 이는 다음과 같은 문제를 야기한다:
1. **취약한 작업 분해**: 인간이 정의한 단계에 의존하며, 단 한 번의 단계 오류가 전체 추론을 실패로 이끈다.
2. **데이터 및 지연 시간**: 방대한 양의 학습 데이터가 필요하며, 추론 시 많은 수의 토큰을 생성해야 하므로 응답 시간이 길어진다.

따라서 본 연구의 목표는 인간 뇌의 계층적 및 다중 시간 척도(multi-timescale) 처리 방식에서 영감을 받아, 외부 토큰 생성 없이 모델 내부의 잠재 공간(latent space)에서 효율적으로 깊은 추론을 수행할 수 있는 **Hierarchical Reasoning Model(HRM)**을 구축하는 것이다.

## ✨ Key Contributions

HRM의 핵심 아이디어는 **계층적 수렴(Hierarchical Convergence)**과 **잠재 추론(Latent Reasoning)**을 통해 실질적인 계산 깊이를 획기적으로 늘리는 것이다.

- **이중 재귀 모듈 구조**: 추상적 계획을 담당하는 고수준(High-level, H) 모듈과 세부 계산을 수행하는 저수준(Low-level, L) 모듈을 결합하여, H-모듈이 가이드를 제공하고 L-모듈이 이를 실행하는 구조를 설계하였다.
- **계층적 수렴 메커니즘**: L-모듈이 국소 평형(local equilibrium)에 도달한 후 H-모듈이 업데이트되고, 다시 L-모듈이 초기화되어 새로운 계산 단계로 진입하는 과정을 통해 표준 RNN의 조기 수렴 문제를 해결하고 계산 깊이를 확장하였다.
- **효율적인 학습 및 추론**: BPTT(Backpropagation Through Time)를 대체하는 1단계 그래디언트 근사(one-step gradient approximation)를 도입하여 메모리 사용량을 $O(T)$에서 $O(1)$로 줄였으며, Q-learning 기반의 적응적 계산 시간(ACT)을 통해 문제 난이도에 따라 계산 자원을 유연하게 할당하도록 하였다.

## 📎 Related Works

기존의 알고리즘 학습 연구들은 다음과 같은 접근 방식을 취해왔다:
- **신경망 기반 알고리즘 학습**: Neural Turing Machines(NTM), Differentiable Neural Computer(DNC) 등이 하드웨어 구조를 모방하여 알고리즘 실행을 시도하였다.
- **Transformer 확장**: Universal Transformers는 층 간의 재귀 루프와 적응적 중단 메커니즘을 도입하여 계산 깊이를 늘리려 하였다.
- **CoT 기반 미세 조정**: A* 알고리즘과 같은 탐색 알고리즘의 경로를 SFT 타겟으로 사용하는 방식이 제안되었다.

그러나 이러한 방식들은 여전히 방대한 데이터에 의존하거나, 재귀 구조의 경우 BPTT로 인한 메모리 효율성 저하 및 학습 불안정성 문제를 겪는다. HRM은 뇌의 생물학적 원리를 차용하여 매우 적은 데이터(약 1,000개 샘플)만으로도 복잡한 상징적 추론(symbolic reasoning)을 수행할 수 있다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
HRM은 입력 네트워크 $f_I$, 저수준 재귀 모듈 $f_L$, 고수준 재귀 모듈 $f_H$, 그리고 출력 네트워크 $f_O$의 네 가지 구성 요소로 이루어진 Sequence-to-Sequence 아키텍처이다.

전체 프로세스는 $N$번의 고수준 사이클과 각 사이클당 $T$번의 저수준 타임스텝으로 구성된다. 전체 타임스텝 $i = 1, \dots, N \times T$ 동안의 상태 업데이트는 다음과 같다:

1. **입력 투영**: 입력 $x$를 $ \tilde{x} = f_I(x; \theta_I)$로 변환한다.
2. **상태 업데이트**:
   - **L-모듈**: 매 타임스텝 $i$마다 자신의 이전 상태, 현재 H-모듈 상태, 입력 표현을 사용하여 업데이트한다.
     $$z^i_L = f_L(z^{i-1}_L, z^{i-1}_H, \tilde{x}; \theta_L)$$
   - **H-모듈**: $T$ 타임스텝마다 한 번씩, L-모듈의 최종 상태를 사용하여 업데이트한다.
     $$z^i_H = \begin{cases} f_H(z^{i-1}_H, z^{i-1}_1; \theta_H) & \text{if } i \equiv 0 \pmod T \\ z^{i-1}_H & \text{otherwise} \end{cases}$$
3. **최종 예측**: 모든 사이클 종료 후 H-모듈의 상태에서 예측값 $\hat{y}$를 추출한다.
     $$\hat{y} = f_O(z^{NT}_H; \theta_O)$$

### 학습 절차 및 손실 함수

**1. 1단계 그래디언트 근사 (Approximate Gradient)**
BPTT의 메모리 부담을 줄이기 위해 Implicit Function Theorem(IFT)과 Neumann series 확장을 기반으로 그래디언트를 근사한다. $(I - J_F)^{-1} \approx I$라고 가정하여, 각 모듈의 마지막 상태에서의 그래디언트만을 사용하는 $O(1)$ 메모리 방식의 1단계 근사를 적용한다.

**2. 깊은 지도 학습 (Deep Supervision)**
모델이 여러 세그먼트(segment)를 거치며 예측을 수행할 때, 각 세그먼트 종료 시점마다 손실을 계산하고 가중치를 업데이트한다. 이때 이전 세그먼트의 상태를 `detach`하여 그래디언트가 이전 세그먼트로 역전파되지 않게 함으로써 학습 안정성을 높인다.

**3. 적응적 계산 시간 (ACT)**
Q-learning을 도입하여 현재 상태 $z^{NT}_H$에서 '중단(halt)'할지 '계속(continue)'할지를 결정하는 Q-head를 학습시킨다.
- **보상**: 'halt' 선택 시 예측의 정답 여부에 따라 binary reward($1$ 또는 $0$)를 부여한다.
- **손실 함수**: 예측 손실과 Q-head의 binary cross-entropy 손실을 결합하여 최적화한다.
  $$L_m^{ACT} = \text{LOSS}(\hat{y}_m, y) + \text{BINARYCROSSENTROPY}(\hat{Q}_m, \hat{G}_m)$$

## 📊 Results

### 실험 설정
- **데이터셋**: 
  - **ARC-AGI-1 & 2**: 귀납적 추론 능력을 측정하는 벤치마크.
  - **Sudoku-Extreme**: 기존 데이터셋보다 훨씬 어려운(평균 22회 백트래킹 필요) 스도쿠 퍼즐.
  - **Maze-Hard**: 30x30 크기의 미로에서 최단 경로를 찾는 작업.
- **비교 대상**: Deepseek R1, Claude 3.7, o3-mini-high 등 최신 CoT 모델 및 단순 Transformer 기반 Direct prediction 모델.
- **학습 조건**: 사전 학습 없이 각 작업당 약 1,000개의 샘플만 사용하여 학습하였으며, 모델 파라미터 수는 약 27M이다.

### 주요 결과
- **ARC-AGI**: HRM은 40.3%의 정확도를 기록하며, 훨씬 거대한 파라미터와 긴 컨텍스트 윈도우를 가진 o3-mini-high(34.5%)와 Claude 3.7(21.2%)를 능가하였다.
- **Sudoku 및 Maze**: CoT 모델들이 거의 0%에 가까운 성능을 보인 반면, HRM은 거의 완벽한(near-perfect) 정확도를 달성하였다. 이는 HRM이 잠재 공간에서 깊은 탐색과 백트래킹(backtracking)을 효과적으로 수행함을 시사한다.
- **추론 시간 확장성 (Inference-time scaling)**: 추가 학습 없이 추론 시 최대 계산 단계($M_{max}$)를 늘리는 것만으로도 특히 스도쿠와 같은 복잡한 작업에서 성능이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 뇌 과학적 대응 (Brain Correspondence)
본 논문은 HRM의 내부 표현이 실제 생물학적 뇌의 계층적 구조와 유사함을 **Participation Ratio (PR)** 분석을 통해 입증하였다.
- **차원 계층 구조**: 분석 결과, 고수준 모듈($z_H$)의 PR(89.95)이 저수준 모듈($z_L$)의 PR(30.22)보다 훨씬 높게 나타났다. 이는 고수준 영역이 더 유연하고 복잡한 표현 공간을 사용한다는 쥐(mouse)의 대뇌 피질 분석 결과와 일치한다.
- **학습된 특성**: 이러한 차원 계층 구조는 모델 아키텍처의 결과가 아니라, 학습을 통해 창발(emergent)된 특성임을 확인하였다(학습되지 않은 모델에서는 PR 차이가 없음).

### 비판적 해석 및 논의
HRM은 고정된 깊이의 Transformer가 해결하지 못하는 튜링 완전(Turing-complete)한 계산 능력을 실질적으로 구현할 수 있는 가능성을 보여주었다. 특히 CoT가 토큰 생성이라는 외부 도구에 의존하는 '보조 장치'라면, HRM은 내부 잠재 공간에서 직접 추론을 수행하는 '내재적 능력'을 지향한다. 다만, PR 분석을 통한 뇌 과학적 유사성은 상관관계일 뿐이며, 이것이 실제 성능 향상의 직접적인 원인인지에 대한 인과적 분석은 향후 과제로 남아 있다.

## 📌 TL;DR

본 연구는 뇌의 계층적 처리 구조를 모방하여, 고수준 계획 모듈과 저수준 실행 모듈이 상호작용하는 **Hierarchical Reasoning Model (HRM)**을 제안하였다. 27M이라는 매우 작은 파라미터와 1,000개의 적은 데이터만으로도 ARC-AGI, 고난도 스도쿠, 미로 찾기 등에서 최신 거대 CoT 모델들을 압도하는 성능을 보였으며, 이는 LLM의 '얕은 구조' 문제를 해결하고 일반 목적의 추론 시스템으로 나아가는 새로운 방향성을 제시한다.