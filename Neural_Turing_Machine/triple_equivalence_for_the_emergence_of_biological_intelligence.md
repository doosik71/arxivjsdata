# Triple equivalence for the emergence of biological intelligence

Takuya Isomura (2023/2024)

## 🧩 Problem to Solve

본 논문은 생물학적 유기체의 지능을 어떻게 정량적으로 특성 짓고(characterize), 진화 과정에서 지능적인 알고리즘이 어떻게 발현되는지를 규명하고자 한다. 생물학적 지능은 뇌의 신경 회로라는 **Dynamical Systems**, 환경을 추론하는 **Bayesian Inference**, 그리고 특정 계산을 수행하는 **Algorithms**라는 세 가지 관점에서 해석될 수 있으나, 이를 통합적으로 설명하는 이론적 프레임워크는 부족한 상태였다. 특히, 기존의 신경망 모델들은 특정 작업에 특화되어 설계된 경우가 많아 유연성이 떨어지며, 생물학적 에이전트가 진화를 통해 어떻게 범용적인 계산 능력(Turing completeness)을 획득하는지에 대한 메커니즘이 명확히 설명되지 않았다. 따라서 본 연구의 목표는 이 세 가지 관점을 통합하여, 진화에 의해 형성되는 생물학적 지능의 보편적 특성을 수식적으로 정의하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Canonical Neural Networks**, **Variational Bayesian Inference**, 그리고 **Differentiable Turing Machines** 사이의 자연스러운 등가성(natural equivalence)을 밝혀낸 이른바 **'Variational Trinity'** 개념을 제안한 것이다.

중심적인 직관은 이 세 가지 시스템이 모두 동일한 **Helmholtz energy**를 최소화한다는 점이다. 이를 통해 다음과 같은 결론을 도출한다.

1. **알고리즘의 신경 구현:** Turing Machine의 상태 전이와 메모리 조작이 신경망의 활동 및 시냅스 가소성으로 구현될 수 있음을 보였다.
2. **진화의 베이즈 해석:** 종(species) 수준에서의 Helmholtz energy 최소화가 자연 선택(Natural Selection)과 동일하며, 이는 결국 환경에 최적화된 생성 모델(Generative Model)을 선택하는 **Active Bayesian Model Selection** 과정임을 증명하였다.
3. **범용성 확보:** 적절한 정신적 행동(mental actions)과 가소성 규칙을 갖춘 신경망이 여러 외부 알고리즘을 모방하고 실행할 수 있는 **Universal Machine**으로 기능할 수 있음을 시뮬레이션을 통해 입증하였다.

## 📎 Related Works

논문은 다음과 같은 기존 연구들을 토대로 한다.

- **Free-energy principle (Friston):** 생물학적 유기체가 변분 자유 에너지(Variational Free Energy)를 최소화함으로써 지각, 학습, 행동을 수행한다는 이론이다.
- **Neural Turing Machines (Graves):** 신경망에 외부 메모리를 결합하여 계산 능력을 확장하려는 시도이다. 하지만 본 논문은 외부 메모리의 생물학적 기질(substrates)과 그것이 진화적으로 어떻게 자가 조직화(self-organize)되는지에 대한 설명이 부족하다는 점을 한계로 지적한다.
- **POMDP (Partially Observable Markov Decision Processes):** 환경의 상태를 완전히 알 수 없는 상황에서 최적의 의사결정을 내리는 프레임워크이다.

본 연구는 기존 연구들이 특정 아키텍처를 사전에 가정(a priori)했던 것과 달리, 일반적인 알고리즘 클래스인 Turing Machine을 도입하고, 이를 Helmholtz energy 최소화라는 단일 원리로 통합함으로써 진화적 관점에서의 유연한 지능 발현을 설명한다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. Canonical Neural Networks와 Helmholtz Energy

생물학적 에이전트를 두 개의 층(middle, output)으로 구성된 재귀 신경망(Recurrent Neural Network)으로 정의한다. 신경 활동 $u_t = \{x_t, y_t\}$의 동역학은 다음과 같은 미분 방정식으로 표현된다.
$$\dot{u} \propto -\text{sig}^{-1}(u) + f(u_{t-1}, s_t)$$
여기서 $\text{sig}$는 시그모이드 함수이며, $f$는 네트워크의 파라미터(시냅스 가중치 $\omega$, 임계값 $\theta$)에 의해 결정되는 함수이다. 이 시스템은 다음과 같은 **Helmholtz energy** $\mathcal{V}$의 경사 하강법(gradient descent)으로 해석될 수 있다.
$$\mathcal{V}[\pi(\phi), s_{0:t}, \xi] = \langle \mathcal{H}_\xi(s_{0:t}, \phi) + \ln \pi(\phi) \rangle_{-\beta(\cdot)}$$
여기서 $\mathcal{H}_\xi$는 유전자 $\xi$에 의해 결정되는 Hamiltonian이다.

### 2. Variational Bayesian Inference와의 등가성

**Complete Class Theorem**에 따라, $\mathcal{V}$를 최소화하는 결정 규칙은 특정 생성 모델 하에서의 베이즈 추론과 등가이다. 즉, $\mathcal{V} \equiv \mathcal{F}$ (Variational Free Energy)가 성립하며, 이때의 자유 에너지 $\mathcal{F}$는 다음과 같다.
$$\mathcal{F}[q(\theta), s_{0:t}, m] = \langle -\ln p_m(s_{0:t}, \theta) + \ln q(\theta) \rangle$$
이 관계를 통해 신경망의 활동($x_t, y_t$)은 외부 환경 상태에 대한 사후 믿음(posterior belief) $\mathbf{s}_t, \mathbf{\delta}_t$를 인코딩하는 것으로 해석된다.

### 3. Differentiable Turing Machines 구현

Turing Machine(TM)의 구성 요소인 오토마톤(Automaton)과 테이프(Tape)를 신경망에 다음과 같이 매핑한다.

- **오토마톤 상태 $\to$ 중간층 활동 $x_t$**
- **메모리 판독 $\to$ 출력층 활동 $y_t$**
- **전이 매핑/메모리 $\to$ 시냅스 가중치 $\mathcal{W}, \mathcal{V}$**

특히, 메모리 쓰기 작업은 신경조절물질(neuromodulator) $\rho_t$에 의해 변조되는 **Three-factor Hebbian plasticity** 규칙을 통해 수행된다. 리스크 $\rho_t$가 높을 때만 기존 값을 잊고 새로운 값을 쓰는 방식이다.

### 4. 진화와 Active Bayesian Model Selection

종 수준에서의 앙상블 Helmholtz energy $\mathcal{V}$를 정의함으로써 자연 선택 과정을 수식화한다.
$$\mathcal{V}[\pi(s_{0:T}, \xi)] = \langle \mathcal{V}[\pi(\phi), s_{0:T}, \xi] - \ln r(s_{0:T}, \xi) - \ln n(\xi) + \ln \pi(s_{0:T}, \xi) \rangle$$
여기서 $r$은 재생산율, $n$은 유전자 분포이다. 이 에너지를 최소화하는 것은 결과적으로 환경의 실제 생성 프로세스 $p(s_{0:T}, \theta)$와 가장 잘 일치하는 유전자(생성 모델)를 선택하는 **Hamiltonian matching** 과정이 되며, 이는 진화가 곧 최적의 생성 모델을 찾는 베이즈 모델 선택 과정임을 의미한다.

## 📊 Results

### 1. 가산기(Adder) 구현 시뮬레이션

외부 환경에 이진수 가산기(Adder) 알고리즘을 설정하고, MNIST 손글씨 숫자 입력을 통해 신경망이 이를 학습하는지 확인하였다.

- **결과:** 신경망은 노이즈가 섞인 입력으로부터 외부 TM의 숨겨진 상태 $\mathbf{s}_t$를 정확히 추론하였으며, 시냅스 가중치 $V$에 가산 결과(메모리 $\mathbf{C}$)를 성공적으로 저장하였다. 시간이 흐름에 따라 외부 TM의 메모리 동역학을 모방하는 오차가 지속적으로 감소함을 확인하였다.

### 2. 진화적 발현 시뮬레이션

임의의 유전자 $\xi$를 가진 개체군이 가산 성능에 따른 재생산율 차이로 진화하는 과정을 시뮬레이션하였다.

- **결과:** 초기에는 무작위 유전자로 인해 성능이 낮았으나, 세대를 거듭하며 최적의 가산 알고리즘을 인코딩하는 특정 유전자 $\xi = (1,0,0,1,0,1,0,1)$로 분포가 수렴(sharp peak)하였다. 이는 자연 선택이 최적의 생성 모델을 선택하는 Bayesian model selection으로 작동함을 보여준다.

### 3. Universal Machine 구현

10개의 서로 다른 외부 TM이 무작위로 교체되는 환경에서, 두 개의 정신적 행동(mental actions)을 가진 신경망의 성능을 측정하였다.

- **결과:** 신경망은 10개의 서로 다른 전이 행렬(transition matrices)을 시냅스 가중치 $V$에 각각 개별적으로 저장하였으며, 입력 데이터에 따라 어떤 TM이 작동 중인지 정확히 판별하여 적절한 메모리 헤더 위치로 이동(attentional switch)함으로써 다음 상태를 정확히 예측하였다.

## 🧠 Insights & Discussion

본 연구는 생물학적 지능을 '진화적으로 획득한 베이즈 추론 알고리즘'으로 정의함으로써, 신경과학의 세 가지 관점을 통합하는 강력한 이론적 틀을 제공한다.

**강점 및 시사점:**

- **범용성:** 특정 작업에 국한되지 않고 Turing completeness를 가진 신경망 아키텍처를 제시하여, 생물학적 지능의 유연성을 설명하였다.
- **생물학적 타당성:** Three-factor Hebbian plasticity와 같은 실제 신경과학적 기제를 수식에 통합하였으며, 해마(hippocampus)와 같은 영역이 메모리 $V$의 역할을 수행할 가능성을 제시하였다.
- **진화의 수학적 정의:** 자연 선택을 Helmholtz energy 최소화와 베이즈 모델 선택으로 연결하여, 지능의 발현 과정을 물리적/통계적 원리로 설명하였다.

**한계 및 논의사항:**

- **공진화(Coevolution)의 복잡성:** 논문 후반부에 여러 에이전트 간의 상호작용과 공진화를 언급하며, 이 경우 상태 전이가 혼돈(chaotic) 상태로 빠질 수 있음을 지적한다. 하지만 이에 대한 상세한 분석과 해결책은 본 논문의 범위를 벗어난 것으로 남겨두었다.
- **가정의 단순화:** 외부 환경을 POMDP와 Turing Machine으로 단순화하여 모델링하였으므로, 실제 복잡한 생태계에서의 적용 가능성에 대해서는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 **Canonical Neural Networks $\approx$ Variational Bayesian Inference $\approx$ Differentiable Turing Machines**라는 'Variational Trinity'를 제안하여, 이들이 모두 **Helmholtz energy**를 최소화한다는 점을 수학적으로 증명하였다. 또한, 자연 선택 과정이 환경에 최적화된 생성 모델을 선택하는 **Active Bayesian Model Selection**임을 밝히고, 이를 통해 신경망이 스스로 범용 계산기(Universal Machine)로 진화할 수 있음을 시뮬레이션으로 입증하였다. 이 연구는 생물학적 지능의 기원을 물리적 에너지 최소화와 통계적 추론으로 통합 설명함으로써, 향후 범용 인공지능(AGI) 설계에 중요한 이론적 기초를 제공할 가능성이 크다.
