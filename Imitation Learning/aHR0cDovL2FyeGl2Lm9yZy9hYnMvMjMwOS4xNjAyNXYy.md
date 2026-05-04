# Symbolic Imitation Learning: From Black-Box to Explainable Driving Policies

Iman Sharifi, Mustafa Yildirim, and Saber Fallah (2025)

## 🧩 Problem to Solve

본 논문은 자율 주행 시스템에서 널리 사용되는 심층 신경망 기반 모방 학습(Deep Neural Network-based Imitation Learning, DNNIL)의 치명적인 한계점을 해결하고자 한다. DNNIL은 실제 데이터셋으로부터 복잡한 주행 정책을 효율적으로 학습할 수 있지만, 다음과 같은 세 가지 주요 문제를 가지고 있다.

첫째, **해석 가능성의 부재(Lack of Interpretability)**이다. 신경망의 블랙박스(Black-box) 특성으로 인해 자율 주행 차량이 왜 특정 행동을 결정했는지 검증하거나 이해하기 어려우며, 이는 안전이 최우선인 자율 주행 분야에서 사회적 신뢰와 규제 준수를 어렵게 만든다.
둘째, **일반화 성능의 한계(Limited Generalizability)**이다. DNNIL 모델은 학습 데이터 분포와 다른 새로운 상황(Unseen situations)에 직면했을 때 부적절하거나 안전하지 않은 행동을 보일 가능성이 크다.
셋째, **데이터 효율성 저하(Data Inefficiency)**이다. 효과적인 학습을 위해 수백만 개의 상태-행동 쌍(state-action pairs)이 필요하며, 이는 학습 비용과 시간의 증가로 이어진다.

따라서 본 논문의 목표는 Inductive Logic Programming(ILP)을 활용하여 인간이 이해할 수 있는 심볼릭 규칙(Symbolic rules) 형태의 주행 정책을 추출하는 **Symbolic Imitation Learning (SIL)** 프레임워크를 제안함으로써, 투명성, 일반화 가능성, 그리고 데이터 효율성을 동시에 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전문가의 시연 데이터에서 단순한 패턴 매칭을 넘어, **일차 논리(First-Order Logic, FOL) 기반의 명시적인 규칙을 유도(Induce)**하는 것이다. 

중심적인 설계 방향은 다음과 같다.
1. **화이트박스 모델로의 전환**: 블랙박스 형태의 신경망 대신, 인간이 읽고 수정할 수 있는 논리 규칙($head :- body$)을 생성하여 의사결정 과정을 완전히 투명하게 만든다.
2. **ILP를 통한 규칙 추출**: 소량의 긍정/부정 예제와 배경 지식(Background Knowledge)만으로도 일반화 가능한 추상적 규칙을 효율적으로 검색하고 추출한다.
3. **계층적 규칙 집합 구성**: 안전(Fatal, Risky) $\rightarrow$ 효율성(Efficient) $\rightarrow$ 부드러움(Smooth) 순으로 규칙을 계층화하여, 치명적인 행동을 먼저 제거하고 남은 선택지 중 최적의 행동을 선택하는 안전 우선 구조를 설계한다.

## 📎 Related Works

기존의 설명 가능한 AI(XAI) 접근 방식은 크게 세 가지로 나뉜다.
- **화이트박스(Symbolic) 모델**: 유한 상태 머신(FSM)이나 결정 트리(Decision Tree)를 사용한다. 하지만 이러한 방식은 문제의 복잡도가 증가함에 따라 확장성(Scalability)이 떨어진다는 한계가 있다.
- **설명 가능한 신경망**: 히트맵 등을 통해 신경망의 내부 동작을 시각화하지만, 이는 사후 해석(Post-hoc)일 뿐 모델 자체가 투명한 것은 아니다.
- **뉴로심볼릭(Neurosymbolic) 프레임워크**: DNN의 학습 능력과 심볼릭 추론을 결합한 형태이다. 최신 기법인 Differentiable Logic Machine(DLM) 등이 이에 해당하지만, 여전히 대규모 레이블 데이터가 필요하거나 학습된 술어(Predicate)의 잠재적 특성으로 인해 완전한 투명성을 보장하지 못하는 경우가 많다.

본 논문이 제안하는 SIL은 **순수 ILP 기반의 모방 학습**으로, 배경 지식이 제공될 때 매우 적은 수의 예제만으로도 추상적인 규칙을 추출할 수 있어 기존 방식들보다 샘플 효율성이 높고 완전한 해석 가능성을 제공한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
SIL 프레임워크는 **지식 획득(Knowledge Acquisition) $\rightarrow$ 규칙 유도(Rule Induction) $\rightarrow$ 규칙 집합(Rule Aggregation)**의 세 단계로 구성된다.

### 1. 지식 획득 (Knowledge Acquisition)
먼저 자율 주행 환경을 심볼릭하게 정의한다. 차량 주변을 8개의 섹터(Front, FrontRight, Right, BackRight, Back, BackLeft, Left, FrontLeft)로 나누고, 각 섹터의 상태를 다음과 같은 술어(Predicate)로 표현한다.
- `sector_isBusy`: 해당 섹터에 차량이 존재하는지 여부 (True/False)
- `sectorVel_isBigger/Lower/Equal`: 상대 속도가 임계값 $\eta_v$ (5 km/h)보다 큰지, 작은지, 혹은 비슷한지 여부
- `sector_isValid`: 해당 도로 섹션이 주행 가능한 영역인지 여부

이러한 술어들을 바탕으로 편향 세트(Bias set, $\mathcal{B}$), 배경 지식($BK$), 그리고 긍정($E^+$) 및 부정($E^-$) 예제 세트를 정의하여 ILP 시스템에 입력한다.

### 2. 규칙 유도 (Rule Induction)
Popper라는 ILP 시스템을 사용하여 학습한다. Popper는 Answer Set Programming(ASP)과 Prolog를 결합하여 '실패로부터 학습(Learning from Failures, LFF)'하는 방식을 취한다.

추출되는 규칙은 크게 세 가지 카테고리로 나뉜다.
- **안전 규칙 (Safe Lane Changing)**: 
    - **Fatal**: 충돌 가능성이 매우 높은 행동. 예: $rlc\_isFatal :- right\_isBusy; not(right\_isValid)$. (우측 차선 변경 시 우측에 차가 있거나 도로가 유효하지 않으면 치명적임)
    - **Risky**: 법규 위반이나 잠재적 위험이 있는 행동. 예: $rlc\_isRisky :- backRight\_isBusy, backRightVel\_isBigger$. (우측 후방 차량이 더 빠르게 접근 중이면 위험함)
- **효율성 규칙 (Efficient Lane Changing)**: 안전한 선택지 중 최적의 경로를 선택. 예: 앞차가 막혀 있고 좌측이 비어 있다면 우측보다 좌측 변경($llc\_isBetter$)을 우선시한다.
- **종단 속도 제어 규칙 (Smooth Longitudinal Velocity)**:
    - **Catch-up**: 앞이 비어 있을 때 목표 속도 $V^d_x$까지 가속.
    - **Follow-up**: 앞차와의 거리가 안전할 때 앞차 속도에 맞춤.
    - **Brake**: 거리가 위험 수준 $C$ 미만이고 상대 속도가 빠를 때 급제동.

종단 가속도 $a_x$는 다음과 같이 계산된다.
$$
a_x = \begin{cases} 
\frac{V^d_x - v_{x,AV}}{\Delta t} & \text{(catch-up)} \\
\frac{v_{x,TV}^2 - v_{x,AV}^2}{2(D-C)} & \text{(follow-up)} \\
\frac{-v_{x,AV}^2}{2D} & \text{(brake)} 
\end{cases}
$$

### 3. 규칙 집합 및 제어 (Rule Aggregation)
유도된 규칙들을 우선순위에 따라 적용하여 최종 행동 $a_t$를 결정한다.
1. **필터링**: Fatal 규칙 $\rightarrow$ Risky 규칙 순으로 적용하여 실행 불가능한 행동을 제거한다.
2. **최적화**: 남은 행동 중 Efficient 규칙을 통해 최적의 행동(LK, LLC, RLC)을 선택한다.
3. **저수준 제어**: 선택된 행동을 목표 횡방향 위치 $y^{NL}_d$로 매핑하고, PID 제어기를 통해 실제 가속도 $a_y$와 $a_x$를 생성한다.

## 📊 Results

### 실험 설정
- **데이터셋**: HighD (독일 고속도로) 및 NGSim (미국 고속도로)
- **비교 대상 (Baselines)**: 
    - **DNNIL**: 3층 MLP 기반 모방 학습
    - **BCMDN**: Mixture Density Networks를 결합한 행동 복제(Behavior Cloning)
    - **GAIL**: 생성적 적대 모방 학습 (GAN 기반)
- **평가 지표**: 
    - **충돌 기반 성공률 ($SR_c$)**: $\left(1 - \frac{N_C}{N}\right) \times 100$
    - **거리 기반 성공률 ($SR_d$)**: $\frac{\bar{D}}{L} \times 100$
    - **평균 속도 ($\bar{V}$)** 및 **차선 변경 횟수 ($N_{LC}$)**

### 주요 결과
1. **안전성**: SIL은 HighD L2R 시나리오에서 **$SR_c$ 100%**를 달성하였다. 반면 DNNIL, BCMDN, GAIL은 현저히 낮은 성공률을 보였다. 이는 명시적인 안전 규칙이 치명적인 행동을 사전에 차단했기 때문이다.
2. **효율성**: SIL의 평균 속도 $\bar{V}$는 $116.06\text{ km/h}$로 비교 모델 중 가장 높았다. 이는 효율성 규칙을 통해 불필요한 정체를 피하고 빠르게 추월했음을 의미한다.
3. **일반화 능력**: 학습하지 않은 방향(R2L) 및 완전히 다른 데이터셋(NGSim)에서도 SIL은 높은 $SR_c$ (각각 98%, 96%)를 유지하며 강건함을 입증하였다.
4. **학습 효율성**: DNNIL 등은 수십만 개의 데이터와 13~20시간의 GPU 학습 시간이 필요했지만, SIL은 소량의 레이블링 된 예제만으로 **초 단위($T_i$)**의 빠른 규칙 유도가 가능했다.

## 🧠 Insights & Discussion

### 강점
- **완벽한 투명성**: 추출된 모든 주행 정책이 일차 논리 형태로 존재하므로, ISO 26262와 같은 자동차 기능 안전 표준을 충족시키기에 매우 유리하다.
- **높은 일반화 성능**: 특정 데이터 분포에 오버피팅되는 신경망과 달리, 추상적인 논리 규칙을 학습하므로 환경 변화에 강건하다.
- **극도의 샘플 효율성**: 전문가의 지식을 배경 지식으로 활용함으로써 학습에 필요한 데이터 양을 획기적으로 줄였다.

### 한계 및 비판적 해석
- **노이즈 취약성**: 부록(Table A4)의 실험에서 보듯, 데이터 레이블에 노이즈가 2%~20% 섞일 경우 정밀도(Precision)와 재현율(Recall)이 급격히 하락한다. 이는 고전적 ILP 시스템이 완벽한 데이터(Perfect data)를 가정하는 경향이 있기 때문이다.
- **설계 복잡도**: 각 규칙마다 적절한 편향 세트($\mathcal{B}$)와 예제 세트를 인간이 직접 정의해야 한다. 이는 시스템 설계자의 도메인 지식에 크게 의존하며, 규칙의 수가 늘어날수록 설정 비용이 증가한다.
- **일관성 문제**: 실제 인간 운전자는 동일한 상황에서도 개인 성향에 따라 다르게 행동하므로, 실제 데이터에서 일관된 심볼릭 규칙을 추출하는 것은 매우 어려울 수 있다.

## 📌 TL;DR

본 논문은 블랙박스 형태의 신경망 모방 학습 대신, **Inductive Logic Programming(ILP)을 통해 인간이 이해할 수 있는 논리 규칙을 추출하는 SIL(Symbolic Imitation Learning) 프레임워크**를 제안한다. SIL은 소량의 데이터만으로도 안전하고 효율적인 주행 정책을 생성하며, 특히 **충돌률 제로(0%)**에 가까운 안전성과 높은 일반화 성능을 보였다. 이 연구는 자율 주행의 핵심 과제인 **'해석 가능성'과 '안전 보장'**을 동시에 해결할 수 있는 실질적인 방향성을 제시하며, 향후 뉴로심볼릭 AI의 실용적 적용 가능성을 높인 연구라고 평가할 수 있다.