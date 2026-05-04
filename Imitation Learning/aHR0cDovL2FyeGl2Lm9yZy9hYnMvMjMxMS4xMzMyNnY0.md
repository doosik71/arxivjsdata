# Curriculum Learning and Imitation Learning for Model-free Control on Financial Time-series

Woosung Koh, Insu Choi, Yuntae Jang, Gimin Kang, Woo Chang Kim (2024)

## 🧩 Problem to Solve

본 논문은 금융 시계열 데이터 기반의 제어 작업(Control Task)에서 발생하는 높은 확률적 변동성(Stochasticity)과 그로 인한 높은 **Noise-to-Signal Ratio** 문제를 해결하고자 한다. 

로보틱스와 같은 물리적 제어 분야에서는 시뮬레이터를 통해 데이터 생성 과정($p_{\text{data}}$)을 모사하여 데이터를 무한히 생성할 수 있으나, 금융 시장에서는 데이터가 시간의 흐름에 따라 고정적으로 생성되므로 추가적인 샘플링이 불가능하다. 이러한 제약 조건 하에서 제한적이고 노이즈가 심한 데이터만으로 모델-프리 강화학습(Model-free RL) 에이전트가 유의미한 신호를 학습하고 일반화 성능을 확보하는 것은 매우 어려운 과제이다. 따라서 본 연구의 목표는 **Curriculum Learning(CL)**과 **Imitation Learning(IL)** 기법을 금융 제어 작업에 도입하여, 고정된 데이터 샘플을 최대한 효율적으로 활용하고 제어 성능을 향상시키는 방법을 탐구하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 데이터의 노이즈를 전략적으로 제어하여 학습 효율을 높이는 것이다.

1. **금융 제어를 위한 Curriculum Learning 도입**: 데이터 증강(Data Augmentation)의 일환으로 데이터 평활화(Smoothing)를 적용한다. 특히, 학습 초기에는 많이 평활화된 '쉬운' 데이터로 시작하여 점진적으로 원본 데이터에 가깝게 '어려운' 단계로 넘어가는 **Inverse-Smoothing** 방식을 제안한다.
2. **Direct Policy Distillation (DPD) 제안**: 미래 데이터를 알고 있는 Oracle(교사)의 정책을 학생 모델이 직접 모방하게 함으로써, 레이블 공간의 노이즈를 제거하고 최적 정책에 빠르게 수렴하도록 유도한다.
3. **신호-노이즈 분해 프레임워크**: 금융 데이터의 움직임을 $\Delta \text{signal}$과 $\Delta \text{noise}$로 이론적으로 분해하고, CL과 IL이 각각 이 구성 요소들에 어떤 영향을 미치는지 분석하여 실험 결과를 이론적으로 뒷받침한다.

## 📎 Related Works

- **Curriculum Learning (CL)**: 인간의 학습 방식처럼 쉬운 예제부터 어려운 예제로 순차적으로 학습시키는 방법이다. 컴퓨터 비전이나 로보틱스에서는 널리 쓰였으나, 금융 시장과 같이 노이즈가 매우 심한 end-to-end 제어 작업에 적용된 사례는 거의 없었다.
- **Imitation Learning (IL)**: 전문가(Expert)나 Oracle의 행동을 모방하여 학습하는 방법으로, 보상 함수 설계가 어렵거나 샘플 효율성이 중요할 때 유리하다. 금융 분야에서도 일부 연구(OPD 등)가 있었으나, 금융 제어의 특성에 맞춘 깊이 있는 탐구는 부족한 실정이다.
- **RL for Financial Control**: 최근 모델-프리 RL을 이용한 포트폴리오 최적화 연구가 증가하고 있으나, 금융 데이터 특유의 비정상성(Non-stationarity)과 높은 노이즈로 인해 일반화 성능을 확보하는 데 어려움을 겪고 있다.

## 🛠️ Methodology

### 1. Portfolio Control as a MDP
본 연구는 포트폴리오 제어 문제를 마르코프 결정 과정(MDP)으로 정의한다.
- **상태(State, $S$)**: 자산의 과거 수익률(Lagged values) 및 거시 경제 변수($M$)를 포함한다.
- **행동(Action, $A$)**: 각 자산에 대해 $\{-1, 0, 1\}$ (Short, None, Long)의 이산적 행동을 취하며, 이를 가중치 $W$로 변환하여 포트폴리오를 구성한다.
- **보상(Reward, $r$)**: 포트폴리오의 로그 수익률 $\ln(\text{portfolio}_t / \text{portfolio}_{t-1})$을 사용한다.
- **제약 조건**: 실제 투자 환경을 모사하여 총 노출도(Gross Exposure)를 $[-1, 1]$ 또는 $[-2, 2]$ 범위로 제한하는 하드 제약 조건을 적용하고, 이를 위해 선형 정규화(Linear Normalization)를 사용한다.

### 2. Curriculum Learning (CL)
데이터의 $\Delta \text{noise}$를 줄이기 위해 두 가지 평활화 방법을 제안한다.
- **EMA (Exponential Moving Average)**: 다음과 같이 재귀적으로 정의된다.
  $$\Delta \text{sec}^{\text{EMA}}_t \leftarrow \alpha \cdot \Delta \text{sec}_t + (1-\alpha) \cdot \Delta \text{sec}^{\text{EMA}}_{t-1}$$
  여기서 $\alpha = 2/(w_l + 1)$이며, $w_l$은 윈도우 크기 하이퍼파라미터이다.
- **Inverse-Smoothing (IS)**: CL의 핵심 아이디어로, $S$개의 단계로 나누어 학습한다. 첫 단계에서는 $w_l=S$로 매우 강하게 평활화하여 학습하고, 단계가 진행됨에 따라 $w_l$을 점차 줄여 마지막 단계에서는 $w_l=1$(평활화 없음)이 되도록 구성한다.

### 3. Imitation Learning (IL)
- **Oracle ($\phi$)**: 미래 데이터를 모두 알고 있는 특수 모델로, 각 시점의 최적 행동 $A^*$를 추출한다.
- **Student ($\psi$)**: Oracle의 행동을 모방하도록 학습한다. 본 논문에서 제안하는 **DPD(Direct Policy Distillation)**는 보상 함수 대신 Oracle과의 행동 거리(L2 distance)를 최소화하는 것을 목표로 한다.
  $$r^\psi := -\text{distance}(A^\psi, A^\phi = A^*)$$

## 📊 Results

### 실험 설정
- **데이터셋**: Macro ETFs (자산군 간 투자, Inter-asset) 및 Commodity Futures (단일 자산군 내 투자, Intra-asset) 두 가지 환경을 사용하였다.
- **알고리즘**: TRPO, PPO, A2C 세 가지 모델-프리 RL 알고리즘을 기반으로 비교하였다.
- **기준선(Baseline)**: Rebalanced Portfolio (RP, 휴리스틱), Vanilla RL, Oracle Policy Distillation (OPD)을 사용하였다.

### 주요 결과
- **CL의 우수성**: 모든 환경(2개 데이터셋 $\times$ 3개 알고리즘)에서 EMA 기반의 CL 방법(EMA5, IS8)이 Vanilla RL 및 휴리스틱 RP보다 뛰어난 성능을 보였다. 특히 EMA5는 6개 환경 모두에서 통계적 유의성($P\text{-value} < 0.05$)을 확보하였다.
- **IL의 한계**: DPD 및 OPD와 같은 모방 학습 접근법은 오히려 성능이 크게 하락하였다. 일부 케이스에서는 에이전트가 아무런 포지션을 잡지 않는(Flat performance) 현상이 관찰되었다.
- **정량적 지표**: 테스트 셋의 누적 수익률(Cumulative Return)을 통해 측정하였으며, CL 적용 시 변동성 대비 수익률이 유의미하게 개선됨을 확인하였다.

## 🧠 Insights & Discussion

### 1. 왜 CL은 성공하고 IL은 실패했는가?
논문은 이를 **신호-노이즈 분해(Signal-Noise Decomposition)** 관점에서 설명한다.
- **CL의 관점**: 적절한 평활화는 $\Delta \text{signal}$보다 $\Delta \text{noise}$를 더 많이 감소시킨다. 이는 모델이 데이터의 본질적인 패턴을 더 쉽게 학습하게 하여 일반화 성능을 높인다.
- **IL의 관점**: Oracle을 통해 레이블 공간의 노이즈를 제거하면, 레이블이 확률적(Stochastic) 성격에서 결정론적(Deterministic) 성격으로 변한다. 이 과정에서 노이즈뿐만 아니라 학습에 필요한 핵심 신호($\Delta \text{signal}$)까지 함께 제거되어, 에이전트가 유의미한 패턴을 인식하지 못하게 된다.

### 2. Degree of Noise-tolerance (DNT)
IL이 실패하는 또 다른 이유로 **DNT** 개념을 제시한다. 제약 조건이 있는 환경에서 Oracle의 최적 행동 $A^*$는 어느 정도의 노이즈가 추가되어도 변하지 않는 '내성'을 가진다. 하지만 이 내성이 너무 강하면 모델이 입력 데이터의 미세한 변화(신호)에 반응하지 않는 둔감한 모델이 되어버린다.

### 3. 비정상성 (Non-stationarity)
실험 중 TIS(Tuned Inverse-Smoothing)가 항상 최적은 아니라는 점을 통해, 신호와 노이즈의 분해 비율이 시간축에 따라 변하는 비정상적 성질을 가질 가능성을 시사한다. 이는 향후 동적으로 평활화 정도를 조절하는 적응형 CL 연구의 필요성을 제기한다.

## 📌 TL;DR

본 논문은 노이즈가 극심한 금융 시계열 제어 문제에서 **Curriculum Learning(데이터 평활화 $\rightarrow$ 원본 데이터)**이 일반화 성능을 획기적으로 향상시킨다는 것을 입증하였다. 반면, **Imitation Learning**은 노이즈를 과도하게 제거하여 오히려 핵심 신호까지 손실시키는 부작용이 있음을 밝혔다. 이 연구는 금융 RL에서 단순히 복잡한 모델을 쓰는 것보다, **데이터의 신호-노이즈 비율을 전략적으로 제어하는 학습 커리큘럼**이 훨씬 중요하다는 인사이트를 제공한다.