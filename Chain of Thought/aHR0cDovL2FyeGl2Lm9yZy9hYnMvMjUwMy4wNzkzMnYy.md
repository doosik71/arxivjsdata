# A Theory of Learning with Autoregressive Chain of Thought

Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, Nathan Srebro (2025)

## 🧩 Problem to Solve

본 논문은 단순한 '다음 토큰 생성기(next-token generator)'를 반복적으로 적용하여 복잡한 함수를 학습하는 Autoregressive Chain-of-Thought (CoT) 생성 및 학습 과정에 대한 이론적 프레임워크를 제안한다. 구체적으로, 입력으로부터 최종 정답만을 관찰하여 학습하는 End-to-End (e2e) 학습과, 정답에 이르는 중간 추론 단계인 CoT 전체를 관찰하며 학습하는 CoT 학습 간의 통계적 및 계산적 복잡성 차이를 분석하는 것을 목표로 한다.

현대 대규모 언어 모델(LLM)이 복잡한 추론 작업을 위해 CoT를 사용하는 경향이 있으나, 이에 대한 이론적 분석은 부족한 상태이다. 특히, 생성 단계마다 동일한 파라미터를 사용하는 '시간 불변성(time-invariance)'이 학습 표본 복잡도(sample complexity)와 계산 효율성에 어떤 영향을 미치는지 규명하고자 한다.

## ✨ Key Contributions

본 연구의 핵심적인 직관은 **시간 불변성(Time-Invariance)**, 즉 생성의 모든 단계에서 동일한 다음 토큰 생성기 $f$를 반복 사용하는 구조가 학습 효율성을 극대화한다는 점이다. 주요 기여 사항은 다음과 같다.

1.  **표본 복잡도의 분리**: 시간 불변성을 통해 생성 길이 $T$에 관계없이(또는 $\log T$ 수준의 낮은 의존성으로) 학습이 가능한 표본 복잡도 경계를 제시하였다.
2.  **계산적 격차 증명**: CoT 데이터가 제공될 경우 계산적으로 다루기 쉬운(tractable) 학습이 가능하지만, 최종 정답만 제공되는 e2e 학습의 경우 계산적으로 매우 어렵다는(hard) 것을 증명하였다.
3.  **범용 학습 가능성 및 Attention의 필연성**: 튜링 머신(Turing Machine)을 시뮬레이션할 수 있는 범용 기본 클래스 $\mathcal{F}_{TM,S}$를 설계하고, 이러한 범용성을 구현하기 위해서는 Attention 메커니즘이 자연스럽게 도출됨을 이론적으로 보였다.

## 📎 Related Works

본 논문은 Malach (2023)의 연구와 직접적으로 대비된다. Malach는 생성 단계마다 서로 다른 생성기 $f_t$를 사용하는 '시간 의존적(time-dependent)' autoregressive 학습을 다루었으며, 이 경우 표본 복잡도가 $T$에 선형적으로 비례하는 한계가 있었다.

반면, 본 논문은 파라미터 공유(parameter sharing) 개념인 시간 불변성을 도입하여, 학습해야 할 대상이 단일 함수 $f$로 축소됨에 따라 $T$에 대한 의존성을 획기적으로 낮출 수 있음을 보였다. 또한, 행동 복제(Behavior Cloning, BC) 연구들과의 연결성을 언급하며, 전문가의 경로(CoT)를 관찰하는 것이 학습의 horizon 문제를 해결하는 핵심임을 시사한다.

## 🛠️ Methodology

### 1. 시스템 구조 및 정의
토큰 집합 $\Sigma$에 대해, 다음 토큰 생성기는 $f: \Sigma^* \to \Sigma$로 정의된다. 입력 $x$에 대해 $f$를 $T$번 반복 적용하여 생성된 문자열을 Chain-of-Thought $\text{f}^{CoT-T}(x)$라 하며, 이 중 마지막 토큰을 최종 정답으로 하는 매핑을 $\text{f}^{e2e-T}(x)$라고 정의한다.

$$f^{CoT-T}(x) = \underbrace{f \circ f \circ \dots \circ f}_{T \text{ times}}(x)$$
$$f^{e2e-T}(x) = f^{CoT-T}(x)[-1]$$

### 2. 학습 설정
본 논문은 다음 두 가지 학습 시나리오를 고려한다.
- **e2e-learnable**: $(x, y)$ 쌍(입력과 최종 정답)만으로 $f$를 학습하는 경우.
- **CoT-learnable**: $(x, z)$ 쌍(입력과 전체 CoT 경로)을 통해 $f$를 학습하는 경우.

### 3. 통계적 복잡도 분석
기본 클래스 $\mathcal{F}$의 성질에 따른 표본 복잡도를 분석하였다.
- **VC-Dimension 기반**: CoT 학습의 표본 복잡도는 $O(\epsilon^{-1}(VCdim(\mathcal{F}) \log T \log \epsilon^{-1} + \log \delta^{-1}))$로, $T$에 거의 독립적이다. 반면, e2e 학습은 $O(T \cdot VCdim(\mathcal{F}))$로 $T$에 선형적으로 비례한다.
- **Littlestone Dimension 기반**: e2e 학습에서도 $Ldim(\mathcal{F})$를 사용하면 $O(\log T)$ 수준의 의존성으로 학습 가능함을 보였다.

### 4. 범용 기본 클래스 $\mathcal{F}_{TM,S}$ 및 Attention 구현
튜링 머신 $\langle S, T, \tau \rangle$를 시뮬레이션하기 위해 토큰 $\Sigma := [S] \times A \times \{-1, 0, +1\}$를 정의하고, 상태 전환 표 $\tau$를 구현하는 $\mathcal{F}_{TM,S}$를 제안하였다. 이 과정에서 다음 두 연산이 필수적으로 요구된다.
- **포지션 계산**: 이전 모든 이동량의 합을 구하는 연산 $\to$ Uniform Attention으로 구현 가능.
- **테이프 룩업(Lookup)**: 현재 헤드 위치의 심볼을 찾는 연산 $\to$ Query(현재 위치)와 Key(쓰기 위치)를 매칭하는 Hard Attention으로 구현 가능.

## 📊 Results

### 1. 선형 임계값(Linear Thresholds) 실험
이진 알파벳 $\Sigma=\{0, 1\}$ 상의 $d$-차원 선형 임계값 클래스 $\mathcal{F}_{d,lin}$를 분석하였다.
- **통계적 결과**: e2e 학습과 CoT 학습 모두 표본 복잡도는 $O(d^2)$ 또는 $O(d \log T)$ 수준으로 나타났으나, $T$가 클수록 CoT 학습의 효율성이 높다.
- **계산적 결과**: CoT 학습은 선형 계획법(LP feasibility)을 통해 다항 시간 내에 해결 가능하지만, e2e 학습은 상수 깊이 임계값 회로($TC^0$) 학습의 어려움에 기반하여 다항 시간 내 학습이 불가능함을 증명하였다(Theorem 4.4).

### 2. 범용성 및 학습 가능성
$\mathcal{F}_{TM,S}$ 클래스는 튜링 머신으로 계산 가능한 모든 함수를 표현할 수 있으며, 다음의 특성을 만족한다.
- **표본 복잡도**: $m_{CoT-T} = O(\epsilon^{-1}(S \log S + \log \delta^{-1}))$로, 런타임 $T$가 아닌 프로그램 길이 $S$에만 의존한다.
- **계산 복잡도**: $\text{Cons}_{\mathcal{F}_{TM,S}}$ 알고리즘을 통해 $Poly(n, T, \epsilon^{-1}, \log \delta^{-1})$ 시간 내에 학습이 가능하다.

## 🧠 Insights & Discussion

본 논문은 CoT 학습이 단순한 성능 향상 도구가 아니라, **계산 불가능한 문제를 가능하게 만드는 필수적인 장치**임을 이론적으로 입증하였다. 특히, e2e 학습이 이론적으로 'Hard' 하다는 결과는 최근 DeepSeek-R1 등 RL을 통해 CoT를 학습시키는 모델들이 왜 사전 학습(Pre-training)된 모델을 기반으로 하는지 설명해 준다. 즉, 완전히 빈 상태에서 정답만으로 CoT를 배우는 것은 어렵지만, 어느 정도의 전이 학습(Transfer)이 가능하다면 tractable한 영역으로 들어올 수 있음을 시사한다.

또한, Transformer의 핵심인 Attention 메커니즘이 단순히 경험적인 선택이 아니라, 시간 불변성을 유지하면서 튜링 완전한(Turing-complete) 계산을 수행하기 위해 반드시 필요한 '비국소적 데이터 참조(non-local lookup)' 수단임을 보인 점이 매우 인상적이다.

한계점으로는, 본 연구가 실현 가능한(realizable) 설정, 즉 정답을 생성하는 완벽한 $f^*$가 클래스 내에 존재한다는 가정하에 진행되었다는 점이다. 실제 환경에서의 Agnostic 설정이나 모델 미지정(misspecification) 상황에서의 분석은 향후 과제로 남아 있다.

## 📌 TL;DR

이 논문은 autoregressive CoT 생성 모델에서 **시간 불변성(Time-Invariance)**이 표본 복잡도를 $T$로부터 독립시켜 효율적인 학습을 가능하게 함을 증명하였다. 특히, CoT 데이터의 유무가 학습의 계산 복잡도를 '불가능(Hard)'에서 '가능(Tractable)'으로 바꾸는 결정적 요인임을 보였으며, 이러한 범용 계산 능력을 구현하기 위해 Attention 메커니즘이 이론적으로 필연적임을 입증하였다.