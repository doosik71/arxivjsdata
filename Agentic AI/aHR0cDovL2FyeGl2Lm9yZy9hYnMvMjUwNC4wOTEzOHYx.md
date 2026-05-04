# White-Box AI Model: Next Frontier of Wireless Communications

Jiayao Yang, Jiayi Zhang, Bokai Xu, Jiakang Zheng, Zhilong Liu, Ziheng Liu, Dusit Niyato, Merouane Debbah, Zhu Han, and Bo Ai (2025)

## 🧩 Problem to Solve

본 논문은 6G 무선 통신 시스템이 요구하는 초고속 데이터 전송률, 대규모 연결성, 그리고 향상된 스펙트럼 및 에너지 효율성을 달성하기 위해 AI의 도입이 필수적이지만, 기존의 **Black-box AI** 모델이 가진 치명적인 한계점들을 해결하고자 한다.

기존의 Black-box 모델(DNN, Transformer 등)은 다음과 같은 문제점을 지닌다:

- **해석 가능성 및 이론적 기반 부족:** 엔드-투-엔드(end-to-end) 학습 특성으로 인해 의사결정 과정이 불투명하며, 이는 통신 시스템의 성능 분석, 공학적 구현 및 실제 검증을 어렵게 만든다.
- **높은 계산 복잡도:** 대규모 신호 처리 및 자원 할당과 같은 고차원 비볼록(non-convex) 최적화 문제를 해결하기 위해 방대한 계산 자원이 필요하며, 이는 시간 민감형 시나리오에서 성능 저하를 야기한다.
- **제한적인 일반화 능력:** 동적인 채널 변동, 비정상적(non-stationary) 간섭, 예측 불가능한 사용자 행동 등 변화가 심한 환경에서 적응력이 떨어진다.

따라서 본 논문의 목표는 투명성과 수학적 검증 가능성을 제공하는 **White-box AI (WAI)** 모델을 제안하고, 이를 무선 통신 시스템에 통합하여 해석 가능하고 신뢰할 수 있는 최적화 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 순수하게 데이터에 의존하는 학습 방식에서 벗어나, **정보 이론(Information Theory)**과 **최적화 이론(Optimization Theory)**을 결합하여 모델의 내부 메커니즘을 명시적으로 설계하는 것이다.

WAI의 중심적인 설계 직관은 다음과 같다:

- **이론 기반의 투명성:** 최적화 경로를 수학적으로 정의하여 결정 과정을 추적 가능하게 함으로써 Black-box의 불투명성을 제거한다.
- **신뢰성 및 강건성 향상:** 정보 병목(Information Bottleneck, IB) 원리와 같은 이론적 제약을 도입하여, 고차원 데이터에서 불필요한 노이즈를 제거하고 핵심 특징만을 추출함으로써 동적 환경에서의 모델 강건성을 높인다.
- **지속 가능한 효율성:** 적응형 최적화 전략과 효율적인 알고리즘 구조를 통해 계산 복잡도와 성능 사이의 최적의 트레이드오프(trade-off)를 달성한다.

## 📎 Related Works

논문에서는 기존의 AI 접근 방식과 WAI의 차별점을 다음과 같이 설명한다:

- **Traditional Black-box Models:** DNN, ResNet, Transformer 등이 무선 네트워크 최적화에 사용되어 왔으나, 앞서 언급한 해석 가능성 및 일반화 문제로 인해 한계가 있다.
- **Physics-Informed Machine Learning (PIML):** 물리 법칙과 수학적 원리를 통합하여 해석 가능성을 높이려는 시도가 있었으나, PIML은 주로 유체 역학이나 기상 예측과 같은 물리적 모델링에 집중한다.
- **WAI와의 차별점:** 제안된 WAI는 PIML과 달리 **정보 이론**과 **최적화 이론**을 통합하여 수학적으로 해석 가능한 최적화 프레임워크를 구축하며, 특히 무선 통신 환경의 특성에 최적화된 투명한 경로를 제공한다는 점에서 근본적으로 다르다.

## 🛠️ Methodology

WAI 모델은 크게 네 가지의 수학적/이론적 기초 위에 구축된다.

### 1. 확률 통계 및 추론 (Probability Statistics and Inference)

- **Bayesian Inference:** 사전 분포(prior), 가능도 함수(likelihood), 사후 업데이트(posterior update)를 통해 추론 과정을 구조화한다. 이를 통해 각 업데이트 단계에 명확한 통계적 해석을 부여하고 추론의 투명성을 확보한다.
- **Message Passing:** 팩터 그래프(factor graph)에서 국부적인 정보를 반복적으로 전달하여 결합 사후 분포를 근사하는 방식(예: Belief Propagation)을 사용하여 정보 흐름을 추적 가능하게 한다.

### 2. 특징 추출 및 표현 (Feature Extraction and Representation)

- **Information Bottleneck (IB):** 입력 $X$에서 타겟 $Y$에 대한 관련 정보는 유지하면서 불필요한 정보는 압축하는 최적의 표현 $Z$를 찾는 프레임워크이다.
  - 목표는 $I(Z;Y)$(관련성)를 최대화하고 $I(X;Z)$(압축률)를 최소화하는 것이며, 매개변수 $\beta$를 통해 이 둘 사이의 균형을 조절한다.
- **Coding Rate Reduction:** 특징 서브셋 내의 콤팩트함은 높이고 서브셋 간의 분리도는 최대화하는 방식이다.
  - 전체 코딩 전송률 $R(Z)$와 조건부 코딩 전송률 $R^c(Z|\Pi)$의 차이를 최대화하여 데이터 중복성을 줄이고 해석 가능성을 높인다. (예: ReduNet 모델)

### 3. 모델 기반 최적화 및 결정 (Model-Driven Optimization and Decision)

- **Deep Unfolding:** 반복적 최적화 알고리즘의 각 단계를 신경망의 레이어로 펼치는 기법이다. 각 업데이트 규칙이 수학적 공식으로 정의되므로 최적화 과정이 투명하며, 데이터 기반 학습을 통해 수렴 속도를 높일 수 있다.
- **Finite Horizon Optimization:** 제한된 반복 횟수 내에서 최악의 오차를 최소화하는 전략이다. 비볼록(non-convex) 문제를 반양정치 계획법(Semidefinite Programming, SDP)으로 재구성하여 볼록(convex) 형태로 변환함으로써 수학적으로 검증 가능한 최적화를 수행한다.

### 4. 대규모 AI 모델 및 아키텍처 (Large AI Model and Architecture)

WAI 기반의 대규모 모델은 다음과 같은 모듈형 구조를 가진다:

1. **Pre-process:** 정규화, 특징 추출, 노이즈 억제 등을 수학적 근거에 따라 수행한다.
2. **Multi-task Adapter:** 채널 추정, 빔포밍, 전력 할당 등 다양한 작업에 맞게 중간 표현을 동적으로 조정한다.
3. **White-box Training:** 정보 이론 및 최적화 이론에 기반하여 수학적으로 검증 가능한 훈련 프레임워크를 적용한다.
4. **Multi-task Output:** 각 작업의 요구사항에 맞는 최적 제어 전략을 생성한다.

- **대표 사례 (CRATE):** 기존 Transformer를 WAI 버전으로 구현한 모델로, Multi-Head Subspace Self-Attention (MSSA)와 Iterative Shrinkage-Thresholding Algorithm (ISTA)을 결합하여 투명한 최적화 경로를 제공한다.

## 📊 Results

논문은 Cell-free massive MIMO 시스템에서의 프리코딩(precoding) 최적화를 통해 WAI의 성능을 검증한다.

### Case 1: EGIB-MDGNN을 이용한 스펙트럼 효율 개선

- **설정:** 10개의 Access Points(APs), 각 AP당 4개 안테나, 4명의 단일 안테나 사용자(UEs), Rayleigh fading 채널, 대역폭 20MHz.
- **방법:** Edge Graph Information Bottleneck (EGIB) 기반의 다차원 그래프 신경망(EIB-MDGNN)을 적용하여 정보 흐름을 명시적으로 최적화하고 불필요한 데이터를 억제한다.
- **결과:** 전통적인 WMMSE 및 일반 GNN 프레임워크보다 높은 스펙트럼 효율(SE)을 보였으며, 특히 간섭 노이즈가 증가할수록 WAI 모델의 강건성이 더욱 두드러지게 나타났다.

### Case 2: Deep Unfolding PGD 알고리즘 적용

- **설정:** 4개의 기지국(BSs), 각 4개 안테나, 8명의 사용자(UEs), Rayleigh fading 채널.
- **방법:** Projected Gradient Descent (PGD) 알고리즘을 계층적 구조로 펼치고(unfolding), 각 레이어에 학습 가능한 파라미터와 적응형 스텝 사이즈(adaptive step size)를 도입하였다.
- **결과:** 고정 스텝 사이즈 방식보다 수렴 속도가 훨씬 빠르며, 불완전한 채널 상태 정보(imperfect CSI) 환경에서도 높은 시스템 안정성과 강건성을 유지함을 확인하였다.

## 🧠 Insights & Discussion

**강점:**
본 논문은 딥러닝의 강력한 성능과 전통적인 통신 이론의 엄밀함을 성공적으로 결합하였다. 특히 Black-box AI가 가진 '설명 불가능성'이라는 문제를 수학적 제약 조건(IB, SDP, Unfolding)을 통해 해결함으로써, 통신 시스템과 같이 신뢰성이 최우선인 분야에 AI를 적용할 수 있는 이론적 토대를 마련하였다.

**한계 및 논의사항:**

- **보안 및 개인정보 위험:** 모델이 투명해진다는 것은 역설적으로 모델의 내부 구조와 최적화 경로가 드러남을 의미한다. 이는 모델 역전 공격(model inversion attacks)이나 멤버십 추론 공격(membership inference attacks)에 더 취약해질 수 있는 보안 리스크를 내포한다.
- **복잡도와 해석력의 균형:** 이론적 제약을 많이 추가할수록 해석력은 높아지지만, 실제 복잡한 데이터 패턴을 학습하는 능력(표현력)이 일부 제한될 가능성이 있으며 이에 대한 정량적 분석이 추가로 필요하다.

## 📌 TL;DR

본 논문은 6G 무선 통신을 위해 기존 Black-box AI의 불투명성을 해결한 **White-box AI (WAI)** 프레임워크를 제안한다. 정보 이론과 최적화 이론을 결합한 Bayesian 추론, Information Bottleneck, Deep Unfolding 등의 기법을 통해 **해석 가능하고 수학적으로 검증 가능한** 모델을 설계하였다. 실험 결과, Cell-free massive MIMO 시스템의 프리코딩 최적화에서 기존 방식보다 높은 스펙트럼 효율과 강건성을 입증하였다. 이 연구는 향후 신뢰성이 필수적인 차세대 통신 네트워크의 AI 도입 및 엣지 지능(Edge Intelligence) 구현에 핵심적인 역할을 할 것으로 기대된다.
