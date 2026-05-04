# CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning

Duo Wu, Jinghe Wang, Yuan Meng, Yanning Zhang, Le Sun, Zhi Wang (2024)

## 🧩 Problem to Solve

최근 대규모 언어 모델(LLM)을 활용하여 복잡한 작업을 해결하기 위해 외부 도구(Vision 모델, Python 프로그램 등)를 스케줄링하는 Tool Planning 연구가 활발히 진행되고 있다. 그러나 기존의 접근 방식들은 도구 실행 과정에서 필연적으로 발생하는 연산 비용, 즉 실행 시간(Execution Time)과 메모리 소비(Memory Consumption)를 고려하지 않는다는 치명적인 한계가 있다.

이로 인해 LLM은 작업 성능은 높을 수 있으나 실행 비용이 과도하게 높은 '비효율적인 계획'을 생성하게 되며, 이는 실제 환경에서 시스템의 실용성을 떨어뜨리는 결과를 초래한다. 따라서 본 논문은 도구 실행 비용을 인식하여 성능과 비용 사이의 최적의 균형(Trade-off)을 맞춘 도구 계획을 생성하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LLM이 도구의 실행 비용을 인지하고, 병렬 실행이 가능한 비순차적 계획(Non-sequential Plan)을 수립할 수 있도록 하는 프레임워크를 설계하는 것이다.

1. **Tool Planning Language (TPL)**: 도구와 그들 간의 의존성을 학습 가능한 토큰(Token)으로 정의하여, LLM이 복잡한 비순차적 구조의 계획을 단순한 토큰 시퀀스 예측 문제로 처리할 수 있도록 설계하였다.
2. **Cost-Aware Offline Reinforcement Learning (CAORL)**: 입력 데이터의 특성이 도구 비용에 미치는 영향을 반영하는 Context Augmentation과, 성능-비용 최적화를 위한 오프라인 강화학습 알고리즘을 통해 LLM을 미세 조정하였다.
3. **OpenCATP 데이터셋**: 비순차적 계획과 실행 비용을 측정할 수 있는 최초의 데이터셋을 구축하였으며, 성능과 비용을 통합적으로 평가하는 $Quality\ of\ Plan\ (QoP)$ 지표를 제안하였다.

## 📎 Related Works

기존의 Tool Planning 연구는 크게 프롬프트 엔지니어링(Prompt Engineering)과 미세 조정(Fine-tuning) 두 가지 패러다임으로 나뉜다. HuggingGPT나 VisProg 같은 프롬프트 기반 방식은 유연하지만, 논리적으로 비선형적인 비순차적 계획을 생성하는 능력이 부족하며 실행 비용을 전혀 고려하지 않는다.

최근 등장한 강화학습 기반 방식(RLTF 등)은 실행 피드백을 통해 모델을 개선하려 하지만, 대부분 계획이 완전히 끝난 후에만 피드백을 주는 Coarse-grained 방식을 사용하여 중간 과정에서의 비용 최적화가 어렵다. CATP-LLM은 TPL을 통해 구조적 제약을 해결하고, 세밀한 중간 피드백을 제공하는 Offline RL을 도입함으로써 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. Tool Planning Language (TPL)

LLM이 복잡한 방향성 비순차 그래프(DAG) 구조를 직접 생성하는 것은 어렵기 때문에, 본 논문은 이를 시퀀스 형태로 변환하는 TPL을 제안한다.

- **토큰 정의**: 각 도구($[t_i]$)와 의존성($\langle d_i \rangle$)을 학습 가능한 임베딩 벡터로 변환한다.
- **구조적 토큰**: 계획의 시작과 끝([SoP], [EoP]), 의존성의 시작과 끝($\langle SoD \rangle, \langle EoD \rangle$)을 알리는 특수 토큰을 도입하여 구조적 정보를 명시한다.
- **표현식**: 도구 계획 $p$는 다음과 같은 시퀀스로 표현된다.
$$p = \{[SoP], \dots, [t_i], \langle SoD \rangle, \langle d_{i1} \rangle, \dots, \langle d_{ij} \rangle, \langle EoD \rangle, \dots, [EoP]\}$$
이를 통해 LLM은 단순한 토큰 예측만으로 병렬 실행이 가능한 복잡한 계획을 생성할 수 있다.

### 2. Cost-Aware Offline RL (CAORL)

TPL을 기반으로 성능과 비용의 균형을 맞추기 위해 다음의 구성 요소를 포함한 CAORL 알고리즘을 사용한다.

**A. Context Augmentation (문맥 증강)**
입력 데이터의 크기(예: 이미지 해상도)에 따라 도구 비용이 달라지므로, 이를 반영하기 위해 다음과 같은 설계를 도입한다.

- 입력 크기를 $k$개 레벨로 분류하고, 각 도구에 대해 레벨별 비용 속성 벡터 $c(t_i)$를 할당한다.
- 현재 입력 레벨 $l$과의 거리에 따라 중요도를 부여하는 importance vector $v_{i-1}$를 계산한다.
$$v_{i-1} = \cos \frac{\pi(i-l)}{2k}, \quad i \in \{1, \dots, k\}$$
- 이후 Multi-head Self-Attention을 통해 최종적인 비용 인식 특징(Cost-aware features)을 생성하여 프롬프트에 삽입한다.

**B. Offline RL 기반 미세 조정**
도구 선택의 연쇄적 효과를 고려하여 Planning을 순차적 의사결정 문제로 정의하고 Decision Transformer(DT)를 사용하여 학습한다.

- **상태(State)**: 현재까지 생성된 도구 계획의 토큰 시퀀스 $s_i = \{t_{k1}, \dots, t_{ki}\}$.
- **액션(Action)**: 다음으로 생성할 도구 토큰 또는 의존성 토큰. (Tool Head와 Dependency Head로 분리하여 예측)
- **보상 함수(Reward)**: 계획의 성능($P$)과 비용($C$)을 가중치 $\alpha$로 조절하여 정의한다.
$$r_i = \begin{cases} -(1-\alpha)C(p_i), & \text{if } a_i \neq [EoP] \\ \alpha P(p_i) - (1-\alpha)C(p_i), & \text{if } a_i = [EoP] \end{cases}$$
특히, 계획이 완료되기 전이라도 중간 단계의 실행 비용을 피드백으로 제공하여 모델이 비용을 최소화하도록 유도한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: OpenCATP (순차적 작업 87종, 비순차적 작업 24종, 총 11,100개 샘플).
- **평가 지표**: $Quality\ of\ Plan\ (QoP)$를 핵심 지표로 사용하며, 다음과 같이 정의한다.
$$\text{QoP} = \alpha P_{task}(p) - (1-\alpha)C_{price}(p)$$
- **비교 대상**: GPT-3.5, GPT-4 기반의 Zero-shot, Few-shot, HuggingGPT, HYDRA 및 Llama2-7B 기반의 IFT, RLTF.

### 2. 주요 결과

- **순차적 계획(Sequential Planning)**: CATP-LLM(Llama2-7B)은 GPT-4 기반 모델보다 낮은 비용으로 유사한 성능을 내며, 평균적으로 QoP가 1.5%~40.6% 향상되었다.
- **비순차적 계획(Non-sequential Planning)**: CATP-LLM의 우위가 더욱 극명하게 나타났다. QoP 기준 타 모델 대비 92.0%~375.4% 높은 성능을 보였으며, 특히 GPT-3.5는 비순차 작업에서 많은 무효 계획(Invalid plans)을 생성한 반면, CATP-LLM은 100%의 유효성을 보장했다.
- **효율성**: 비순차적 계획을 통해 도구들을 병렬로 실행함으로써, 순차적 계획만 생성하는 기존 모델들보다 실행 시간(Runtime)을 획기적으로 단축시켰다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **비용 인식의 실효성**: Context Augmentation을 통해 입력 데이터의 크기와 도구 비용의 상관관계를 학습시킴으로써, 불필요하게 고비용 도구(예: 이미지 크기가 충분할 때의 Super Resolution)를 배제하는 합리적인 판단이 가능해졌다.
- **구조적 유연성**: TPL은 LLM이 자연어 지시만으로는 생성하기 어려워하는 비선형적 DAG 구조를 효과적으로 생성하게 함으로써, 병렬 처리의 이점을 극대화하였다.
- **강화학습의 효율성**: 중간 피드백(Intermediate feedback)을 제공하는 방식이 최종 결과만 보는 방식(RLTF)보다 비용 최적화에 훨씬 효과적임이 입증되었다.

### 한계 및 논의사항

- **LLM 크기의 영향**: 실험 결과, 단순한 순차 작업은 작은 모델(Qwen-3B 등)로도 가능하지만, 복잡한 비순차 작업에서는 Llama2-7B 이상의 모델이 필요하다는 점이 확인되었다. 이는 비용 인식 계획 능력이 모델의 기본 추론 능력에 의존함을 시사한다.
- **하드웨어 의존성**: 비용 측정치(런타임, 메모리)가 하드웨어 사양에 따라 달라질 수 있으나, RTX 3090/4090 등 다양한 환경에서도 CATP-LLM이 일관되게 높은 QoP를 유지함을 확인하여 범용성을 입증하였다.

## 📌 TL;DR

본 논문은 LLM 기반 도구 계획에서 무시되었던 **'실행 비용'** 문제를 해결하기 위해, 비순차적 계획 생성을 가능케 하는 **TPL(Tool Planning Language)**과 비용-성능 최적화를 위한 **CAORL(Cost-Aware Offline RL)** 프레임워크를 제안한다. 또한 이를 평가하기 위한 **OpenCATP 데이터셋**과 **QoP 지표**를 구축하였다. 실험 결과, Llama2-7B와 같은 상대적으로 작은 모델로도 GPT-4보다 비용 효율적인 최적의 도구 계획을 수립할 수 있음을 보였으며, 이는 향후 실용적인 AI 에이전트 시스템 구축에 중요한 기여를 할 것으로 평가된다.
