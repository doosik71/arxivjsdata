# DINGO: Constrained Inference for Diffusion LLMs

Tarun Suresh, Debangshu Banerjee, Shubham Ugare, Sasa Misailovic, Gagandeep Singh (2025)

## 🧩 Problem to Solve

본 논문은 Diffusion LLM(Large Language Models)에서 사용자 정의 제약 조건(Formal Constraints), 특히 정규 표현식(Regular Expressions)을 엄격하게 준수하면서도 모델의 출력 분포를 왜곡하지 않는 추론 방법을 해결하고자 한다.

기존의 Autoregressive(AR) LLM은 토큰을 순차적으로 생성하므로, 각 단계에서 유효하지 않은 토큰을 제거하는 방식의 Constrained Decoding 알고리즘이 잘 작동한다. 그러나 Diffusion LLM은 토큰 블록을 병렬로 예측하는 특성을 가지고 있어, 기존의 순차적 제약 디코딩 방식을 그대로 적용할 수 없다. 또한, AR 모델에서 사용되는 Greedy token selection 방식의 제약 디코딩은 국소적인 확률은 최대화할 수 있으나 전체 시퀀스 관점에서는 최적이 아니며, 이는 출력 분포의 왜곡(Distribution Distortion)을 초래하여 생성물의 품질을 저하시키는 문제가 있다.

따라서 본 연구의 목표는 Diffusion LLM의 병렬 생성 특성을 유지하면서도, 정규 표현식으로 정의된 제약 조건을 완벽하게 준수하고, 동시에 모델이 예측한 확률 분포 내에서 가장 가능성이 높은(Maximum Probability) 문자열을 효율적으로 샘플링할 수 있는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Diffusion LLM을 위한 최초의 Constrained Decoding 알고리즘인 **DINGO**를 제안한 것이다. DINGO의 중심 설계 아이디어는 다음과 같다.

1. **DP 기반의 최적 경로 탐색**: 정규 표현식을 결정적 유한 오토마톤(Deterministic Finite Automaton, DFA)으로 변환하고, 동적 계획법(Dynamic Programming, DP)을 사용하여 제약 조건을 만족하는 모든 가능한 문자열 중 모델의 출력 확률을 최대화하는 최적의 경로를 효율적으로 찾아낸다.
2. **분포 보존(Distribution-Preserving)**: 단순히 유효한 토큰을 선택하는 Greedy 방식이 아니라, 전체 블록 범위에서 확률을 최대화하는 문자열을 선택함으로써 모델의 본래 출력 분포를 보존한다.
3. **마스크 토큰($\bot$)의 처리**: Diffusion LLM의 핵심인 마스크 토큰을 DFA 전이 함수에 통합하여, 마스크된 상태에서도 향후 유효한 문자열로 확장 가능한지(Valid Prefix)를 판별할 수 있도록 설계하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Diffusion LLMs**: LLaDA, Dream-7B와 같은 최신 모델들은 AR 모델의 순차적 생성 속도 한계를 극복하기 위해 병렬 블록 생성 방식을 도입하였다. 하지만 이들은 구조화된 출력(예: JSON, 수학식)이 필요한 작업에서 구문 오류를 자주 범하며, 이를 강제할 방법이 부재했다.
2. **AR LLM Constrained Decoding**: 기존의 AR 모델들은 DFA나 CFG(Context-Free Grammar)를 이용해 생성 단계에서 유효하지 않은 토큰을 마스킹하는 방식을 사용한다. 그러나 이러한 방식은 Diffusion 모델의 병렬 예측 구조와 호환되지 않으며, 앞서 언급한 분포 왜곡 문제가 보고된 바 있다.

### 차별점

DINGO는 기존의 Greedy 방식이나 반복적인 Resampling 방식(계산 비용이 매우 높음)과 달리, DP를 통해 단 한 번의 최적화 과정으로 제약 조건을 만족하는 최적의 문자열을 도출한다. 또한, 병렬 생성 구조를 가진 Diffusion LLM에 특화되어 설계된 최초의 알고리즘이라는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

DINGO는 크게 **전처리(Precomputation)** 단계와 **추론 시 DP(Dynamic Programming)** 단계로 구성된다.

### 1. 전처리 (Precomputation)

사용자가 제공한 정규 표현식 $R$을 바탕으로 다음과 같은 과정을 거친다.

- **Token-level DFA 구축**: 캐릭터 단위의 DFA를 모델의 토큰 집합 $V$에 맞는 토큰 단위 DFA로 변환한다.
- **마스크 전이 함수 $\delta_\bot$ 정의**: 마스크 토큰 $\bot$이 나타났을 때, 이 위치에 어떤 토큰이 오더라도 전이 가능한 모든 상태의 집합을 정의한다.
- **Live States ($Q_l$) 식별**: 현재 상태에서 시작하여 최종 수락 상태(Accepting State)에 도달할 수 있는 '살아있는' 상태들의 집합을 미리 계산한다.

### 2. DINGO Dynamic Programming

모델이 예측한 각 위치의 토큰 확률 분포 $v_1, \dots, v_d$가 주어졌을 때, 다음의 DP를 수행한다.

**DP 상태 정의**:

- $W[i, q]$: 시작 상태 $q_0$에서 $i$번째 위치까지 도달했을 때, 상태 $q$에 도달하는 최대 확률.
- $Pr[q, i]$: $W[i, q]$를 달성하기 위해 거쳐온 이전 상태와 선택된 토큰을 저장하는 백포인터.

**상태 업데이트 (DP Update)**:
$$W[i+1, q] = \max_{q' \in Q} \left( W[i, q'] \times V_{i+1}(q, q') \right)$$
여기서 $V_{i+1}(q, q')$는 상태 $q'$에서 $q$로 전이하게 만드는 토큰 중 가장 확률이 높은 토큰의 확률값이다.

**경로 재구성 (Path Construction)**:
마지막 단계 $d$에서 $W[d, q]$ 값이 가장 크면서 동시에 $q \in Q_l$ (Live State)인 상태 $q_{max}$를 선택한다. 이후 $Pr$에 저장된 백포인터를 따라 역추적하여 최적의 토큰 시퀀스를 복원한다.

### 3. 복잡도 분석

전체 시간 복잡도는 $O(d \cdot |Q| \cdot (|Q| + |V|))$이다. 여기서 $d$는 블록 길이, $|Q|$는 DFA 상태 수, $|V|$는 어휘 사전 크기이다. 정규 표현식의 경우 $|Q|$가 작은 상수인 경우가 많으므로 실질적으로 매우 효율적이다.

## 📊 Results

### 실험 설정

- **데이터셋**: GSM-Symbolic(심볼릭 수학 추론), JSON-Mode-Eval(스키마 기반 JSON 생성).
- **모델**: LLaDA-8B (Base/Instruct), Dream-7B (Base/Instruct).
- **기준선(Baselines)**: Unconstrained(제약 없음), Greedy Constrained(기존 AR 방식의 그리디 적용), Best of Greedy + Unconstrained.

### 주요 결과

1. **구문 정확도 (Parse %)**: DINGO는 모든 모델과 작업에서 **100%의 구문 정확도**를 달성하였다. 반면, Greedy Constrained 방식은 유효한 접두사(Prefix)만 생성하고 전체 문자열을 완성하지 못하는 경우가 많아 낮은 Parse %를 보였다.
2. **기능적 정확도 (Acc %)**:
    - **GSM-Symbolic**: LLaDA-8B-I 모델 기준, Unconstrained 대비 약 13%p, Greedy Constrained 대비 5%p 성능 향상을 보였다.
    - **JSON Generation**: Dream-B-7B 모델의 경우, Unconstrained(15%) 대비 DINGO(100%)라는 극적인 성능 향상을 달성하였다.
3. **추론 효율성**: DINGO는 제약을 추가했음에도 불구하고 Unconstrained 추론 대비 오버헤드가 매우 미미하여 실제 적용 가능성이 높음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

DINGO는 Diffusion LLM의 병렬 생성이라는 특성과 정규 언어의 제약 조건을 수학적으로 결합하여, **'구문적 완벽함'**과 **'확률적 최적성'**을 동시에 잡았다. 특히, 단순히 가능한 토큰을 고르는 것이 아니라 DP를 통해 전체 경로의 확률을 최대화함으로써, 제약 조건 강제로 인한 생성 품질 저하 문제를 효과적으로 해결하였다.

### 한계 및 향후 과제

1. **언어 클래스의 제한**: 본 연구는 정규 언어(Regular Language)에 한정되어 있다. 실제 프로그래밍 언어와 같은 문맥 자유 언어(Context-Free Language)나 문맥 민감 언어를 강제하기 위해서는 더 확장된 DP 프레임워크가 필요하다.
2. **Semi-autoregressive 설정의 최적성**: 단일 블록 내에서는 최적의 해를 찾지만, 여러 블록을 순차적으로 생성하는 Semi-autoregressive 설정에서는 전체 시퀀스 관점에서의 전역 최적성(Global Optimality)이 보장되지 않을 수 있다.
3. **비정형 제약**: 독성 완화(Toxicity Mitigation)와 같이 형식 언어로 정의할 수 없는 제약 조건은 본 방법론으로 해결할 수 없다.

## 📌 TL;DR

본 논문은 Diffusion LLM에서 정규 표현식 제약 조건을 엄격하게 준수하면서도 모델의 출력 분포를 보존하는 **DINGO** 알고리즘을 제안한다. DINGO는 DFA와 동적 계획법(DP)을 활용하여 제약 조건을 만족하는 가장 확률 높은 문자열을 효율적으로 찾아내며, 실험 결과 수학식 및 JSON 생성 작업에서 구문 정확도 100%와 상당한 성능 향상을 달성하였다. 이 연구는 Diffusion LLM을 구조화된 데이터 생성과 같은 신뢰성이 중요한 실제 응용 분야에 적용하는 데 있어 중요한 기술적 토대를 제공한다.
