# Imitator Learning: Achieve Out-of-the-Box Imitation Ability in Variable Environments

Xiong-Hui Chen, Junyin Ye, Hang Zhao, Yi-Chen Li, Haoran Shi, Yu-Yan Xu, Zhihao Ye, Si-Hang Yang, Anqi Huang, Kai Xu, Zongzhang Zhang, and Yang Yu (2023)

## 🧩 Problem to Solve

본 논문은 기존의 모방 학습(Imitation Learning, IL)이 가진 한계점을 극복하기 위해 **Imitator Learning (ItorL)**이라는 새로운 연구 주제를 제안한다. 

기존의 IL 기술들은 대량의 전문가 시연(demonstrations)을 통해 하나의 특정 정책을 정밀하게 모방하는 데 집중해 왔다. 그러나 실제 응용 환경에서는 다음과 같은 문제들이 발생한다:
1. **제한된 시연 데이터**: 모든 작업에 대해 대량의 시연 데이터를 수집하는 것은 비용 효율적이지 않으며, 실제로는 매우 적은 수의 시연(예: 단 한 개의 시연)만으로 작업을 수행해야 하는 경우가 많다.
2. **환경의 가변성**: 배포 환경에서 전문가가 시연했을 때와는 다른 예상치 못한 환경 변화(Unexpected changes)가 발생할 수 있으며, 에이전트는 이러한 상황에서도 유연하게 적응하여 작업을 완수해야 한다.
3. **추가 조정의 부재**: 실시간 적용을 위해 배포 단계에서 추가적인 미세 조정(fine-tuning) 없이 즉각적으로 모방 능력을 발휘하는 'Out-of-the-box' 능력이 요구된다.

따라서 본 논문의 목표는 매우 제한된 전문가 시연만을 바탕으로, 추가 조정 없이 새로운 작업에 대한 모방 정책을 즉석에서 재구성할 수 있는 일반화된 'Imitator' 모듈을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단 한 개의 전문가 시연만으로도 다양한 환경에서 작업을 수행할 수 있는 **Demo-Attention Actor-Critic (DAAC)** 프레임워크를 제안한 것이다. 주요 설계 아이디어는 다음과 같다:

- **Imitator Learning (ItorL) 정의**: 단일 시연을 통해 새로운 작업의 정책을 즉각적으로 재구성하는 문제 정의 및 이론적 가능성($\tau_{\Omega}$-tracebackable MDP set) 제시.
- **Demo-Attention (DA) 아키텍처**: 현재 상태와 전문가 시연 상태 간의 유사도를 기반으로 어떤 상태를 따라갈지 결정하고, 그에 대응하는 전문가의 행동을 취하도록 유도하는 어텐션 메커니즘 설계.
- **Imitator Reward ($R_{Itor}$) 설계**: 시연에 포함되지 않은 상태(unvisited states)에서도 에이전트가 합리적인 행동을 할 수 있도록 유도하는 보상 함수를 설계하여 RL 패러다임 내에서 학습.
- **벤치마크 구축 및 검증**: ItorL 능력을 측정하기 위한 새로운 내비게이션 벤치마크(Demo-Navigation)와 로봇 조작 환경을 구축하여 제안 방법론의 우수성을 입증.

## 📎 Related Works

논문에서는 다음과 같은 기존 접근 방식들을 소개하며 DAAC와의 차별점을 설명한다:

1. **Behavior Cloning (BC) 및 Meta-IL**:
   - Transformer 등을 이용해 시연을 입력으로 받는 문맥 기반(context-based) 정책 모델들이 존재한다. 그러나 이들은 시연 데이터가 제한적일 때 예측 오류가 누적되는 compounding errors 문제에 취약하며, 새로운 작업으로의 일반화 능력이 떨어진다.
2. **Few-shot/One-shot Meta-IL (MAML 등)**:
   - MAML 계열의 방법들은 초기 파라미터를 학습한 후 배포 시 몇 단계의 경사 하강법(gradient descent)을 통한 미세 조정이 필요하다. 반면 DAAC는 추가적인 미세 조정 없이 즉각적으로 동작한다.
3. **Context-based Meta-RL (CbMRL)**:
   - RNN이나 GRU를 사용하여 환경 파라미터를 추출하고 적응형 정책을 학습하는 방식이다. DAAC는 이러한 프레임워크를 계승하되, 단순히 문맥 벡터를 사용하는 대신 DA 아키텍처를 통해 시연 데이터 내의 지식을 더 효율적으로 마이닝한다.

## 🛠️ Methodology

### 1. 전체 프레임워크: Context-based Meta-RL
DAAC는 문맥 기반 메타 강화학습(Context-based Meta-RL) 구조를 따른다. 전체 정책 $\Pi$는 문맥 추출기 $\phi$와 문맥 기반 정책 $\pi$로 분해되어 $\Pi(a|s, \phi(T_\omega))$ 형태로 정의된다. 여기서 $T_\omega$는 전문가 시연 데이터이며, 추출기 $\phi$는 이를 잠재 변수 $z$로 변환하여 정책 $\pi$에 전달한다.

### 2. Demo-Attention (DA) 아키텍처
에이전트가 시연 데이터를 효율적으로 활용하도록 하기 위해, 본 논문은 다음의 두 단계 의사결정 과정을 네트워크 구조로 내재화한다.

- **Phase 1: 어떤 상태를 따라갈 것인가? (Determine the state to follow)**
  - 현재 에이전트의 상태 $s_j$를 쿼리(Query, $q$)로, 전문가 시연의 상태 집합 $[s_{e0}, \dots, s_{et}]$를 키(Key, $k$)로 사용하여 어텐션 가중치 $w$를 계산한다.
  - $w = \text{softmax}(qk^\top / \sqrt{d_k})$를 통해 현재 상태와 가장 유사한 전문가 상태에 높은 가중치를 부여한다.
- **Phase 2: 어떤 행동을 취할 것인가? (Determine the action to take)**
  - 계산된 가중치 $w$를 전문가의 행동 표현인 밸류(Value, $v$)에 곱하여 최종 행동 $a_j$를 도출한다.
  - $v'' = \sum_i v_i w_i$

이 구조는 모델이 명시적인 손실 함수 없이도 "현재 상태와 가장 비슷한 전문가 상태의 행동을 모방하라"는 제약을 가지게 하여 학습 효율과 일반화 능력을 높인다.

### 3. Imitator Reward ($R_{Itor}$)
시연에 없는 상태에서도 안정적인 학습을 위해, 다음과 같은 정교한 보상 함수를 설계하여 Soft Actor-Critic (SAC) 알고리즘에 적용한다.

$$R_{Itor}(s, a) := 1 - \min_{n} \left( d(\bar{s}, s)^2 + \frac{d(\bar{a}, a)^2}{\exp(d(\bar{s}, s)^2)} \right), \eta \text{ or } \alpha R_\omega(s, a)$$

여기서 $(\bar{s}, \bar{a})$는 현재 상태 $s$와 가장 가까운 전문가의 상태-행동 쌍이다.
- **Clipping ($\eta$)**: 너무 먼 상태 쌍에 의한 보상이 에이전트를 오도하는 것을 방지하기 위해 페널티를 일정 상수로 클리핑한다.
- **Reweighting ($1/\exp(d^2)$)**: 현재 상태가 시연 상태와 멀 때, 전문가의 행동을 엄격하게 따르려다 발생하는 위험을 줄이고 다시 시연 경로로 복귀하는 것을 유도한다.
- **Task Reward ($\alpha R_\omega$)**: 최종 목표 달성에 대한 보상을 크게 설정하여, 단순한 모방을 넘어 실제 작업 완수에 집중하게 한다.

## 📊 Results

### 1. 실험 설정
- **Demo-Navigation Benchmark**: 24x24 크기의 미로에서 목표 지점까지 도달하는 작업. 전역 맵 정보 없이 지역 뷰(local view)만 제공된다.
- **Robot Manipulation**: 물체 잡기(Grasping), 쌓기(Stacking), 수집하기(Collecting) 등 복잡한 조작 작업.
- **Complex Control**: Reacher, Pusher 환경.
- **비교 대상**: DCRL, TRANS-BC, CbMRL.

### 2. 주요 결과
- **성공률(Success Rate)**: 표 1과 표 2에서 확인할 수 있듯이, DAAC는 학습된 시연(seen)뿐만 아니라 새로운 시연(new\_demo), 심지어 새로운 맵(new\_map)에서도 다른 베이스라인들을 압도하는 성능을 보였다.
- **일반화 능력**: 특히 coordinates(좌표) 정보가 제공될 때 매우 높은 성능을 보였으며, 좌표가 없을 때(Non-Coord)는 성능이 하락했으나 여전히 베이스라인보다는 우수했다.
- **Scaling Up**: 데이터의 양(demonstrations quantity)과 모델 파라미터 수를 늘렸을 때, 성능이 로그-선형(log-linear)적으로 증가하는 경향을 확인하였다. 특히 파라미터 수를 늘렸을 때 좌표 없는 환경에서의 성능이 약 2배 향상되었다.
- **Ablation Study**: $R_{Itor}$ 보상을 제거하거나 DA 구조를 일반 Transformer로 교체했을 때 학습 효율과 최종 성능이 크게 저하됨을 확인하여, 제안한 구성 요소들의 유효성을 입증했다.

## 🧠 Insights & Discussion

### 강점
- **즉각적인 적응력**: 미세 조정 없이 단 하나의 시연만으로 새로운 작업을 수행할 수 있는 능력을 입증했다.
- **구조적 정규화**: 어텐션 메커니즘을 통해 "상태 매칭 $\rightarrow$ 행동 선택"이라는 직관적인 프로세스를 네트워크 아키텍처에 내재화함으로써 학습 효율을 극대화했다.
- **강건성**: $R_{Itor}$와 RL의 결합을 통해 시연에 없는 예상치 못한 장애물이 나타나도 이를 우회하여 목표를 달성하는 능력을 보여주었다.

### 한계 및 논의사항
- **POMDP 문제**: 좌표 정보가 없는 경우, 동일한 지역 뷰를 가졌지만 실제 위치가 다른 상태들이 존재하여 성능이 저하되는 문제가 발생했다. 이는 부분 관측 마르코프 결정 과정(POMDP)의 특성으로, 향후 연구 과제로 남겨두었다.
- **계산 비용**: 시연 데이터의 길이가 길어질수록 self-attention 메커니즘으로 인해 추론 시 계산 자원 요구량이 증가한다.
- **원거리 상태**: 현재 상태가 전문가 시연의 모든 상태와 너무 멀리 떨어져 있을 경우, 어텐션 메커니즘이 적절한 상태를 매칭하지 못해 성능이 떨어지는 한계가 있다.

## 📌 TL;DR

본 논문은 단 한 개의 전문가 시연만으로 새로운 환경과 작업에 즉각 대응하는 **Imitator Learning (ItorL)**이라는 새로운 패러다임을 제시하고, 이를 해결하기 위한 **DAAC (Demo-Attention Actor-Critic)** 알고리즘을 제안했다. DAAC는 현재 상태와 시연 상태를 매칭하는 **Demo-Attention 아키텍처**와 시연 외 상태에서도 학습을 가능케 하는 **Imitator Reward**를 결합하여, 기존의 모방 학습 및 메타 학습 방법론보다 월등한 일반화 성능을 보여주었다. 이 연구는 자율 주행이나 로봇 조작과 같이 시연 데이터 수집이 어렵고 환경 변화가 심한 실제 시스템에 적용될 가능성이 매우 높다.