# ON THE TENSOR REPRESENTATION AND ALGEBRAIC HOMOMORPHISM OF THE NEURAL STATE TURING MACHINE

Ankur Mali, Alexander Ororbia, Daniel Kifer, Lee Giles (2023)

## 🧩 Problem to Solve

전통적으로 Recurrent Neural Networks (RNNs)와 Transformer는 튜링 완전성(Turing-completeness)을 가진다고 알려져 있다. 그러나 이러한 이론적 결과들은 대부분 가중치의 무한 정밀도(infinite precision), Transformer의 경우 위치 인코딩(positional encodings), 그리고 일반적으로 제한 없는 계산 시간(unbounded computation time)을 가정한다.

실제 응용 분야에서는 단 한 번의 패스(single pass)만으로 튜링 완전한 문법을 인식할 수 있는 실시간(real-time) 모델이 필요하다. 기존의 Neural Turing Machine (NTM)이나 Differentiable Neural Computer (DNC) 같은 모델들은 고정된 크기의 메모리 모듈을 사용하기 때문에 공간 제한적 튜링 머신(space-bounded TMs)만을 처리할 수 있어 진정한 의미의 튜링 완전성을 달성하지 못했다.

따라서 본 논문의 목표는 유한한 정밀도의 연결성과 제한된 가중치를 가지면서도, 임의의 튜링 머신(TM)을 실시간으로 시뮬레이션할 수 있는 새로운 클래스의 재귀 모델인 Neural State Turing Machine (NSTM)을 제안하고 그 이론적 근거를 증명하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 튜링 머신의 전이 규칙(transition rules)을 RNN의 고차 텐서 가중치(higher-order tensor weights)에 직접 매핑하는 것이다.

주요 기여 사항은 다음과 같다.

- 유한한 정밀도의 가중치와 제한된 시간 내에서 NSTM이 튜링 머신과 동등(Turing equivalence)함을 이론적으로 증명하였다.
- 텐서 연결을 통해 재귀 구조 없이도 튜링 머신을 시뮬레이션할 수 있는 유한 정밀도 ANN의 상한선을 제시하였다.
- 튜링 머신의 전이 테이블(transition table)과 심볼을 NSTM의 가중치에 인코딩하는 방법을 공식화하였다.
- 이론적으로 6개의 뉴런으로 범용 튜링 머신(UTM)을 시뮬레이션할 수 있으며, 유한 정밀도 환경에서는 13개의 뉴런만으로 모든 TM 클래스를 실시간으로 모델링할 수 있음을 보였다.
- Dyck 언어와 같은 복잡한 문법 추론 실험을 통해 NSTM이 기존의 메모리 증강 모델이나 Transformer보다 긴 시퀀스에서 월등한 성능을 보임을 입증하였다.

## 📎 Related Works

기존 연구들은 ANN의 튜링 완전성을 증명하기 위해 주로 무한 정밀도나 무제한 시간을 가정하였다. Siegelman과 Sontag는 임의의 정밀도와 무제한 가중치를 가진 RNN이 특정 조건을 만족할 때 튜링 완전함을 보였다. 최근에는 Transformer나 Neural GPU 같은 비재귀 모델로 이러한 증명이 확장되었으나, 이 역시 가중치의 무한 정밀도와 원-핫 인코딩 기반의 위치 인코딩을 전제로 한다. 이는 입력 알파벳의 크기가 커질 경우 차원이 기하급수적으로 증가하는 계산적 불가능성을 초래한다.

반면, NTM이나 DNC는 외부 메모리를 도입하여 한계를 극복하려 했으나, 고정된 메모리 크기로 인해 범용적인 튜링 머신을 완전히 대체하지 못했다. 본 논문은 이러한 기존 접근 방식과 달리, 텐서 기반의 고차 시냅스 연결을 통해 유한한 정밀도와 제한된 뉴런 수만으로도 실시간 튜링 동등성을 확보함으로써 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 구성 요소

NSTM은 완전히 연결된 재귀 뉴런 집합으로 구성되며, 튜링 머신의 상태 수와 동일한 수의 상태 뉴런(state neurons)을 가진다. 시스템은 크게 세 가지 뉴런 타입으로 나뉜다.

1. **상태 뉴런 ($s$):** TM의 제어 상태를 나타낸다.
2. **액션 뉴런 ($a$):** 테이프 심볼을 저장하거나 로컬 정보를 처리하며, TM의 전이 특성을 저장하는 텐서로 작동한다.
3. **결과 뉴런 ($z$):** 상태 뉴런과 액션 뉴런의 결합을 통해 TM의 한 단계 진화(evolution)를 모델링한다.

### 핵심 방정식 및 작동 원리

NSTM의 상태 업데이트는 다음과 같은 텐서 곱과 비선형 활성화 함수 $\sigma$를 통해 이루어진다.

$$s_{t+1}^i = \sigma\left(\sum_{j,k,l} W_{ijkl}^s (z_t^j, r_t^k, x_t^l) + b_i^s\right)$$
$$a_{t+1}^i = \sigma\left(\sum_{j,k,l} W_{ijkl}^a (z_t^j, r_t^k, x_t^l) + b_i^s\right)$$
$$z_{t+1}^i = \sigma\left(\sum_{j,k,l} W_{ijkl}^s (z_t^j, r_t^k, x_t^l) a_{t+1}^i + b_i^z\right)$$

여기서 $W^s$와 $W^a$는 4차원 가중치 텐서이며, $\sigma$는 다음과 같이 정의된 포화 선형 함수(saturated-linear function)이다.
$$\sigma(z) := \begin{cases} 0 & \text{if } z < 0 \\ z & \text{if } 0 \le z \le 1 \\ 1 & \text{if } z > 1 \end{cases}$$

### 튜링 머신의 인코딩

TM의 전이 함수 $\delta: \Gamma \times Q \to \Gamma \times Q \times \{-1, 0, 1\}$는 NSTM의 가중치 텐서에 직접 매핑된다. 특히 '정지 상태(halting state, $q_0$)'를 도입하여 유효하지 않은 설정(illegal configurations)을 처리함으로써 유한 정밀도 환경에서도 안정성을 확보하였다.

본 논문은 두 가지 타입의 텐서 곱을 정의한다.

- **Type 1 Tensor Product:** 상태 뉴런과 액션 뉴런의 상호작용을 통해 로컬 정보와 글로벌 정보를 캡처하여 TM의 다음 상태를 결정한다.
- **Type 2 Tensor Product:** 재귀 구조가 없는 피드포워드 네트워크에서 고차 텐서(n-th order tensor)를 사용하여 전체 상태 전이 이력을 인코딩함으로써 튜링 동등성을 달성한다.

## 📊 Results

### 실험 설정

- **작업:** Dyck 언어($D_2, D_3, D_4$) 인식. Dyck 언어는 괄호의 짝이 맞는지 확인하는 문법으로, 재귀적 구조를 인식해야 하므로 ANN의 메모리 능력을 테스트하기에 적합하다.
- **데이터셋:** 훈련 샘플 5,000개(길이 $T \le 52$), 검증 500개, 테스트 3,000개(길이 $52 < T \le 120$). 추가로 매우 긴 문자열($n=500, 1000$)에 대한 테스트 셋을 구성하였다.
- **비교 대상:** LSTM, Stack-RNN, NTM, Transformer (SA+ 변형 모델).
- **지표:** 정확도(Accuracy).

### 주요 결과

1. **긴 시퀀스에서의 성능:** Table 1에 따르면, $D_4$ 언어에 대해 NSTM은 단 8개의 뉴런만으로도 매우 긴 문자열($n=1000$)에서 99.4%의 정확도를 보였다. 이는 Transformer(4.0%)나 NTM(65.0%)보다 월등히 높은 수치이다.
2. **뉴런 효율성:** NSTM은 매우 적은 수의 뉴런(8개)으로 작동하지만, 다른 모델들은 하이든 유닛 수를 1,024개까지 늘려도 NSTM의 성능에 미치지 못했다(Table 2).
3. **일반화 능력:** 훈련 데이터보다 훨씬 긴 시퀀스에서도 성능 저하가 거의 없었으며, 이는 NSTM이 단순한 패턴 암기가 아니라 TM의 전이 규칙을 가중치에 성공적으로 학습했음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 의의

NSTM의 가장 큰 강점은 **해석 가능성(Interpretability)**이다. TM의 전이 규칙이 가중치 텐서에 직접 인코딩되므로, 학습된 모델에서 상태 머신을 역으로 추출하여 모델이 어떤 규칙으로 작동하는지 설명할 수 있다. 또한, 이론적으로 증명된 최소 뉴런 수(유한 정밀도 재귀 모델의 경우 13개)를 통해 ANN의 계산 능력에 대한 새로운 상한선을 제시하였다.

### 한계 및 비판적 해석

첫째, **계산 복잡도**의 문제이다. 고차 텐서 시냅스를 사용하기 때문에 가중치의 수가 급격히 증가하며, 이는 모델의 확장성(scalability)을 저해한다.
둘째, **학습의 어려움**이다. 표준적인 BPTT(Backpropagation Through Time)로는 고차 텐서 가중치를 최적화하기 어렵다. 이 때문에 본 논문에서는 계산 비용이 매우 높은 RTRL(Real-Time Recurrent Learning) 알고리즘을 사용하였는데, 이는 대규모 네트워크에 적용하기에 비현실적이다.
셋째, **최적화 도구의 부재**이다. Adam이나 RMSprop 같은 현대적인 최적화 도구가 NSTM의 텐서 구조와 잘 맞지 않아 단순 SGD만을 사용했다는 점은 실용적인 관점에서 한계로 작용한다.

## 📌 TL;DR

본 논문은 고차 텐서 가중치를 활용하여 튜링 머신의 전이 규칙을 직접 모델링하는 **Neural State Turing Machine (NSTM)**을 제안하였다. NSTM은 유한 정밀도의 가중치와 제한된 뉴런 수만으로도 임의의 튜링 머신을 실시간으로 시뮬레이션할 수 있음을 이론적으로 증명하였으며, 특히 13개의 뉴런만으로 튜링 동등성을 달성할 수 있음을 보였다. 실험적으로는 Dyck 언어 인식 작업에서 기존의 LSTM, Transformer, NTM보다 훨씬 적은 파라미터로 긴 시퀀스에 대해 압도적인 일반화 성능을 보였다. 이 연구는 향후 해석 가능한 뉴로-심볼릭(neuro-symbolic) AI 시스템 구축을 위한 중요한 이론적 토대를 제공한다.
