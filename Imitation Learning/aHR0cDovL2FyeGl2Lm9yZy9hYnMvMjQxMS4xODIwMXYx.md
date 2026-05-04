# Learning for Long-Horizon Planning via Neuro-Symbolic Abductive Imitation

Jie-Jing Shao, Hao-Ran Hao, Xiao-Wen Yang, Yu-Feng Li (2025)

## 🧩 Problem to Solve

본 논문은 로봇 조작이나 가사 노동과 같이 매우 긴 단계의 의사결정이 필요한 **Long-Horizon Planning** 문제에서 기존 모방 학습(Imitation Learning)과 심볼릭 플래닝(Symbolic Planning)이 가지는 한계를 해결하고자 한다.

먼저, 데이터 기반의 모방 학습 방식은 전문가의 시연 데이터를 학습하여 동작을 수행하지만, 시연 데이터가 제한적일 때 발생하는 **Covariate Shift**(학습 시의 상태 분포와 실제 실행 시 마주하는 상태 분포의 차이) 문제로 인해 작업의 길이가 길어질수록 성능이 급격히 저하되는 경향이 있다. 

반면, 전통적인 심볼릭 플래닝은 인간이 정의한 논리적 공간 위에서 추론하므로 Long-Horizon 작업에 매우 강하며 일반화 능력이 뛰어나다. 그러나 이러한 방식은 고차원의 시각적 입력(Raw Observation)을 심볼릭 상태(Symbolic State)로 매핑하는 지각(Perception) 과정에서 많은 수의 정교한 심볼릭 주석(Predicate-level annotations)을 필요로 하며, 실제 환경의 복잡한 관측값을 처리하는 데 어려움이 있다.

따라서 본 논문의 목표는 데이터 기반 학습의 유연성과 심볼릭 추론의 논리적 강점을 결합하여, **방대한 양의 심볼릭 주석 없이도 고차원 관측값으로부터 Long-Horizon 플래닝을 수행할 수 있는 프레임워크를 구축**하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **ABductive Imitation Learning (ABIL)**이라는 새로운 뉴로-심볼릭(Neuro-Symbolic) 프레임워크를 제안한 것이다. 

핵심 아이디어는 **가추법적 학습(Abductive Learning)**을 도입하여, 전문가의 시연 궤적과 전문가가 제공한 지식 베이스(Knowledge Base) 사이의 **순차적 일관성(Sequential Consistency)**을 이용해 지각 모델을 학습시키는 것이다. 즉, 명시적인 레이블 없이도 "이 시연 궤적이 지식 베이스의 논리적 흐름을 따랐다면, 특정 시점의 관측값은 어떤 심볼릭 상태여야만 한다"라는 가설을 세워 의사 레이블(Pseudo-labels)을 생성하고 이를 통해 지각 모델을 최적화한다. 

또한, 학습된 심볼릭 지각을 바탕으로 각 논리적 연산자(Logical Operator)에 대응하는 **정책 앙상블(Policy Ensemble)**을 구축하여, 고차원 추론과 저차원 제어를 효율적으로 연결한다.

## 📎 Related Works

본 연구는 크게 세 가지 분야의 관련 연구를 바탕으로 한다.

1.  **Imitation Learning**: 전문가의 시연을 통해 정책을 학습하는 방식이다. 하지만 결정 단계(Horizon)가 길어질수록 일반화 능력이 떨어지고 분포 변화에 취약하다는 한계가 있다.
2.  **Neuro-Symbolic Planning**: 신경망의 지각 능력과 심볼릭의 추론 능력을 결합하려는 시도이다. 기존 연구(예: Regression Planning Networks, PDSketch)는 많은 양의 심볼릭 주석이 필요하거나, 모델 기반 플래닝 시 오류가 누적되어 Long-Horizon 작업에서 실패하는 경우가 많다. 특히 PDSketch는 원시 관측-동작 공간에서의 플래닝으로 인해 오류 누적 문제가 심각하다.
3.  **Abductive Learning**: 머신러닝과 논리 추론을 통합하는 프레임워크로, 주로 분류 작업에 사용되어 왔다. 본 논문은 이를 플래닝 문제, 특히 Long-Horizon 의사결정 과정에 적용하여 차별점을 둔다.

## 🛠️ Methodology

### 1. 문제 정의 및 지식 베이스 (Knowledge Base)
환경은 상태 공간 $\mathcal{S}$, 동작 공간 $\mathcal{A}$, 전이 함수 $\mathcal{T}$, 그리고 작업 관련 객체 집합 $\mathcal{O}$와 술어(Predicate) 집합 $\mathcal{P}$로 정의된다. 전문가가 제공하는 지식 베이스는 유한 상태 머신(Finite-State Machine) 형태의 유향 그래프 $\mathcal{K} = \langle V, E \rangle$로 정식화된다.
- **노드($v \in V$):** 서브 태스크의 조건이 되는 ground atoms의 집합이다.
- **엣지($e \in E$):** 심볼릭 동작 $\text{op}$, 추가 효과 $\text{EFF}^{+}$, 삭제 효과 $\text{EFF}^{-}$로 구성된다.

### 2. 가추법적 추론을 통한 지각 학습 (Abductive Reasoning)
심볼릭 주석이 없는 상태에서 관측값 $s$를 심볼릭 상태로 변환하는 지각 함수 $f$를 학습하기 위해 가추법을 사용한다. 최적화 목표는 다음과 같다.

$$ \min_{f} \sum_{s_i \in \mathcal{D}} \sum_{t=1}^{T} L(f(s_t^i), b_t^i) $$
$$ \text{subject to } \{b_t^i\}_{t=1}^{T} \models \mathcal{K} $$

여기서 $b_t^i$는 지식 베이스 $\mathcal{K}$를 만족하는 최적의 심볼릭 시퀀스(의사 레이블)이다. **순차적 일관성(Sequential Consistency)** 원칙에 따라 다음과 같이 레이블을 생성한다.
- **최종 목표:** 시연의 마지막 상태 $b_T^i$는 최종 목표 $\mathcal{G}$를 만족해야 한다.
- **순서 제약:** 태스크 $a$가 $b$보다 먼저 수행되어야 한다면, 시퀀스는 $[1, j]$ 구간에서 $a$를, $[j+1, T]$ 구간에서 $b$를 만족해야 한다.
- **선택 제약:** $a$ 또는 $b$ 중 하나만 완료하면 된다면, 시퀀스는 $a$ 혹은 $b$ 중 하나를 만족해야 한다.
- **병렬 제약:** $a$와 $b$ 모두 완료해야 한다면, 순서에 상관없이 두 태스크가 모두 시퀀스에 포함되어야 한다.

지각 모델 $f$는 객체 수준에서 학습되며, 각 술어 $\phi$에 대해 객체 특징을 입력으로 받는 이진 분류기로 구현된다.

### 3. 심볼릭 기반 모방 학습 (Symbolic-grounded Imitation)
지각 모델이 학습되면, 이를 이용해 고수준 논리 연산자에 기반한 행동 정책을 학습한다.
1.  **연산자 선택:** 현재 관측값 $s_t$에 대해 지각 함수 $f(s_t)$가 어떤 노드 $v_k$에 해당하는지 판단하여, 해당 상태에서 수행해야 할 논리적 연산자 $\overline{\text{op}}_t$를 결정한다.
    $$ \text{op}_t = \text{op}_k, \text{ s.t. } f(s_t) \models v_k $$
2.  **동작 파라미터 추론:** 연산자 $\text{op}_t$에 필요한 대상 객체 $\theta_t$를 추론한다.
3.  **정책 앙상블 학습:** 각 논리적 연산자 $\text{op}$마다 별도의 행동 액터 $h_{\text{op}}$를 구축하고, 시연 데이터의 동작 $a_t^i$를 모방하도록 학습한다.
    $$ \min_{h} \sum_{s_i, a_i \in \mathcal{D}} \sum_{t=1}^{T} L(h_{\text{op}_t^i}(s_t^i, \theta_t), a_t^i) $$

## 📊 Results

### 실험 설정
- **데이터셋:** BabyAI, Mini-BEHAVIOR, CLIPort 세 가지 벤치마크 사용.
- **비교 대상:** Behavior Cloning (BC), Decision Transformer (DT), PDSketch.
- **지표:** 작업 성공률(Success Rate), 데이터 효율성(Data Efficiency), 제로샷 일반화(Zero-Shot Generalization).

### 주요 결과
1.  **Long-Horizon 작업 성능:** BabyAI의 `Put`(평균 9단계), `Unlock`(평균 10단계) 및 Mini-BEHAVIOR의 긴 작업들에서 ABIL은 BC, DT, PDSketch보다 월등한 성공률을 보였다. 특히 PDSketch는 작업 길이가 길어질수록 탐색 오류가 누적되어 성능이 급락했다.
2.  **데이터 효율성:** ABIL은 PDSketch보다 훨씬 적은 양의 데이터(약 20% 미만)만으로도 더 정확한 뉴로-심볼릭 그라운딩(Neuro-symbolic grounding)을 달성했다.
3.  **일반화 및 제로샷 전이:**
    - **일반화:** 학습 때 보지 못한 객체 수가 늘어난 환경에서도 ABIL은 안정적인 성능을 유지했다.
    - **제로샷:** `Pickup`과 `Open`을 학습한 후, 이를 조합한 `Unlock` 작업을 수행하는 제로샷 테스트에서 ABIL은 성공적으로 동작했으나, BC와 DT는 실패했다. 이는 심볼릭 추론을 통해 하위 목표를 순차적으로 달성하는 능력이 탑재되었음을 의미한다.
4.  **추론 효율성:** PDSketch와 같은 모델 기반 플래닝은 추론 시 탐색 시간이 오래 걸리지만, ABIL은 학습된 정책 앙상블을 사용하여 데이터 기반 방식과 유사한 빠른 추론 속도를 유지하면서도 논리적 강점을 가진다.

## 🧠 Insights & Discussion

### 강점
- **주석 비용 제거:** 가추법적 추론을 통해 명시적인 심볼릭 레이블 없이도 지각 모델을 학습시킴으로써 데이터 구축 비용을 획기적으로 줄였다.
- **오류 누적 방지:** 전체 궤적을 탐색하는 대신, 고수준 심볼릭 상태에 따라 저수준 정책을 선택하는 구조를 취함으로써 Long-Horizon 작업에서의 오류 누적 문제를 해결했다.
- **인간 인지 모델 모사:** "무엇을 해야 할지"를 먼저 결정하고 "어떻게 행동할지"를 수행하는 인간의 인지 과정을 신경망과 심볼릭의 결합으로 적절히 구현했다.

### 한계 및 비판적 해석
- **지식 베이스 의존성:** 본 모델은 전문가가 설계한 정확한 지식 베이스 $\mathcal{K}$가 존재한다는 가정 하에 작동한다. 만약 지식 베이스에 오류가 있거나 불충분하다면 성능이 저하될 가능성이 크다.
- **환경 가정:** 결정론적(Deterministic)이고 완전 관측 가능한(Fully Observable) 환경을 가정하고 있다. 실제 환경의 확률적 특성(Stochasticity)이나 부분 관측 가능성(Partial Observability)을 처리하기 위해서는 POMDP 등의 기법 도입이 필요해 보인다.

## 📌 TL;DR

본 논문은 Long-Horizon 플래닝을 위해 가추법적 학습(Abductive Learning)과 모방 학습을 결합한 **ABIL** 프레임워크를 제안한다. 이 연구는 **심볼릭 주석 없이도** 전문가의 시연과 논리적 지식 베이스 간의 일관성을 이용해 지각 모델을 학습시키고, 이를 기반으로 한 정책 앙상블을 통해 복잡한 작업을 수행한다. 실험 결과, 기존의 단순 모방 학습이나 모델 기반 플래닝보다 **데이터 효율성, 일반화 성능, 그리고 제로샷 조합 능력**에서 압도적인 우위를 보였으며, 이는 실제 로봇 제어 및 가사 지원 에이전트 구현에 중요한 이정표가 될 수 있다.