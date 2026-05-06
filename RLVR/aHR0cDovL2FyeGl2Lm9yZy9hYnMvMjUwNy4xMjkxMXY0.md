# LaViPlan: Language-Guided Visual Path Planning with RLVR

Hayeon Oh (2025)

## 🧩 Problem to Solve

자율주행 시스템에서 학습 데이터 분포를 벗어난 Out-of-distribution (OOD) 시나리오는 매우 치명적인 도전 과제이다. 기존의 플래너들은 학습 경험 외부의 상황에서 일반화 능력이 떨어져, 안전하지 않거나 예상치 못한 동작을 수행하는 경향이 있다.

최근 Vision-Language Models (VLMs)가 고차원의 장면 이해와 사용자 정렬된 의사결정 능력을 바탕으로 이러한 OOD 시나리오를 해결할 대안으로 제시되었다. 그러나 기존 VLM들은 언어 기반의 추론(Reasoning)과 실제 행동 단계에서 필요한 저수준 궤적(Low-level trajectories) 생성 사이에 불일치가 발생하는 'vision-language-action misalignment' 문제를 겪고 있다. 즉, 모델이 상황을 올바르게 설명하더라도 실제 예측하는 경로가 그 추론 결과와 일치하지 않는 현상이 발생한다.

본 논문의 목표는 Reinforcement Learning with Verifiable Rewards (RLVR)를 활용하여 VLM을 미세 조정함으로써, 언어 기반 추론과 행동 수준의 경로 계획을 정렬하고 OOD 시나리오에서의 일반화 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 경로 계획과 관련된 객관적인 지표(Planning-oriented metrics)를 검증 가능한 보상(Verifiable Rewards)으로 사용하여 VLM을 강화 학습시키는 것이다.

1. **RLVR 기반 정렬 프레임워크 제안**: 단순히 정답 궤적을 모방하는 Supervised Fine-Tuning (SFT)을 넘어, ADE(Average Displacement Error)와 FDE(Final Displacement Error) 같은 플래닝 지표를 직접 최적화하는 강화 학습 프레임워크를 제안하였다.
2. **기능적 추론(Functional Reasoning)으로의 전환**: RLVR을 통한 학습이 모델의 출력을 단순히 언어적으로 충실한(Linguistically faithful) 묘사에서, 작업 수행에 필수적인 핵심 요소에 집중하는 기능적 추론으로 변화시킨다는 점을 정성적/정량적으로 분석하였다.
3. **데이터 효율성 및 일반화 성능 입증**: RLVR이 SFT보다 훨씬 적은 양의 학습 샘플로도 더 높은 성능 향상을 이끌어낼 수 있으며, 특히 어려운 케이스(Hard cases)를 포함한 강화 학습이 OOD 일반화에 효과적임을 보였다.

## 📎 Related Works

### VLMs for Autonomous Driving

종단간(End-to-end) 자율주행은 단순성과 효율성이 장점이지만, 복잡한 장면 정보를 과도하게 단순화하여 중요한 단서를 놓치거나 롱테일(Long-tail) 시나리오에서 일반화 능력이 부족한 한계가 있다. 이를 해결하기 위해 LLM과 Visual Encoder를 결합한 VLM이 도입되었으며, 최근에는 3D 위치 임베딩이나 반사실적 학습(Counterfactual learning)을 통해 맥락 인식 의사결정을 내리려는 시도가 이어지고 있다.

### Preference Learning for Alignment

VLM의 출력을 하위 작업(예: 궤적 예측)과 정렬하기 위해 RLHF(Reinforcement Learning from Human Feedback)와 PPO(Proximal Policy Optimization)가 사용되어 왔으나, 이는 막대한 계산 비용과 인간의 피드백에 의존한다는 단점이 있다.

이에 대한 대안으로 Group Relative Policy Optimization (GRPO)가 제안되었다. GRPO는 별도의 리워드 모델이나 크리틱(Critic) 없이, 정책 모델과 참조 모델 간의 비교 및 규칙 기반의 검증 가능한 보상(Rule-based verifiable rewards)을 통해 최적화를 수행한다. 본 논문은 이러한 RLVR과 GRPO 개념을 자율주행의 경로 계획 문제에 적용하여, 인간의 주관적 피드백 대신 ADE/FDE라는 객관적 지표를 보상으로 사용함으로써 계산 비용을 줄이고 정렬 성능을 높였다.

## 🛠️ Methodology

### 전체 파이프라인 (Two-Phased Fine-tuning)

본 방법론은 두 단계의 학습 과정을 거친다.

1. **Phase 1: Supervised Fine-Tuning (SFT)**: 이미지, 지시어, 궤적 데이터 쌍을 사용하여 VLM을 기본적으로 미세 조정한다. 이는 모델이 경로 계획 동작의 기초를 다지는 단계이다.
2. **Phase 2: Reinforcement Learning with Verifiable Reward (RLVR)**: SFT 모델을 참조 모델(Reference model)로 삼고, GRPO 알고리즘을 통해 검증 가능한 보상을 기반으로 정책 모델(Policy model)을 최적화한다.

### GRPO 및 RLVR 최적화

GRPO는 그룹 내 상대적 이점(Group-relative advantage)을 사용하여 분산을 줄인다. 그룹 크기 $G$만큼의 궤적을 생성하고, 각 궤적의 보상 $R_i$를 그룹 내 평균과 표준편차로 정규화하여 Advantage $\hat{A}_{i,t}$를 계산한다.

$$ \hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)} $$

최종 목적 함수 $J_{GRPO}$는 PPO의 Clipped surrogate objective에 KL 발산(KL divergence) 규제 항을 추가하여, 정책 모델이 참조 모델에서 너무 멀어지지 않도록 제어한다.

### 보상 함수 설계 (Reward Function)

전체 보상 $R$은 출력 형식에 대한 보상($R_{format}$)과 플래닝 정확도에 대한 보상($R_{planning}$)의 합으로 정의된다.

$$ R = R_{format} + R_{planning} $$

특히 $R_{planning}$은 ADE와 FDE를 활용하여 다음과 같이 정의된다.

$$ R_{planning} = -\log \left( 1 + \frac{1}{N} \sum_{i=1}^N \|\hat{p}_i - p_i\|^2 \right) - \log (1 + \|\hat{p}_N - p_N\|^2) $$

여기서 $\hat{p}_i$는 예측된 좌표, $p_i$는 정답 좌표이며, 로그 스무딩(Logarithmic smoothing)을 적용하여 수치적 안정성을 확보하였다.

### 추론 절차 및 프롬프트 구조

모델은 반드시 `<think>` 태그 내에 시각적 단서를 바탕으로 한 추론 과정을 작성하고, `<answer>` 태그 내에 20개의 $(x, y)$ 좌표 리스트를 출력하도록 강제된다.

## 📊 Results

### 실험 설정

- **데이터셋**: ROADWork (도로 공사 시나리오, In-domain), CODA-LM (다양한 코너 케이스, OOD)
- **베이스라인 모델**: Qwen2VL-2B-Instruct
- **평가 지표**: ADE, FDE (In-domain) / Safety Score (OOD: Fail Rate, Collision Count, Penetration Length의 가중 합)

### 주요 결과

1. **In-domain 성능**: ROADWork 데이터셋에서 LaViPlan은 SFT 모델보다 낮은 ADE/FDE를 기록하며 가장 우수한 성능을 보였다. 특히 어려운(Hard) 궤적 시나리오에서 성능 향상이 뚜렷했다.
2. **OOD 안전성**: CODA-LM 데이터셋 평가 결과, LaViPlan은 모든 가중치 설정(Balanced, Safety-Focused 등)에서 SFT 및 베이스라인보다 높은 Safety Score를 획득하였다. 특히 물리적 충돌 깊이를 측정하는 Penetration Length를 최소화하는 경향을 보였다.
3. **추론 스타일의 변화**: BERTScore와 NLI 분석 결과, LaViPlan은 SFT 모델에 비해 언어적 유사도는 떨어졌으나, 정성적으로는 불필요한 묘사를 배제하고 '콘', '배리어' 등 주행에 필수적인 위험 요소에 집중하는 '기능적 추론' 양상을 보였다.

### Ablation Study

- **SFT vs RLVR**: SFT 이후 RLVR을 적용했을 때 ADE/FDE가 추가로 감소하여, 보상 최적화가 단순 모방 학습 이상의 성능 향상을 가져옴을 확인하였다.
- **샘플 구성**: RLVR 단계에서 어려운 샘플(Hard samples)의 비중을 높였을 때(최적 비율 6:4 또는 7:3), OOD 일반화 성능이 향상되었다.
- **추론 과정의 영향**: `<think>` 과정을 포함하여 학습했을 때, 포함하지 않았을 때보다 일관되게 성능이 높았으며, 특히 복잡한 시나리오에서 그 효과가 컸다.

## 🧠 Insights & Discussion

### 보상의 희소성 (Sparse Reward)

본 연구에서 사용된 ADE/FDE 보상은 전체 궤적이 생성된 후에만 계산되는 희소 보상(Sparse reward)의 성격을 띤다. 이는 정책 최적화 과정에서 학습 속도를 늦추고 분산을 증가시킬 수 있으며, 향후 단계별(Step-wise) 피드백이나 보조 보상을 도입한다면 더 효율적인 학습이 가능할 것으로 보인다.

### GRPO와 Affordance

GRPO 기반의 미세 조정은 강력한 SFT 모델이 선행되어야 한다는 전제가 있다. 또한, 설계된 보상 함수가 모델의 'Affordance(행동 가능성)'를 정의하게 된다. 단순히 정답 궤적을 모방하는 지표를 넘어, 안전성이나 성공률을 명시적으로 반영한 보상 설계가 필요하다.

### 비판적 해석

본 논문은 VLM의 언어적 충실도(Linguistic fidelity)와 기능적 정확도(Functional accuracy) 사이의 트레이드-오프를 발견했다는 점에서 학술적 가치가 크다. 하지만 여전히 2D 이미지 평면상의 좌표 예측에 의존하고 있어, 실제 3D 환경에서의 물리적 정밀도나 연속적인 의사결정 과정에서의 반사실적 추론(Counterfactual reasoning) 능력에 대해서는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 자율주행 VLM의 고질적인 문제인 '추론과 행동의 불일치'를 해결하기 위해, ADE/FDE와 같은 플래닝 지표를 보상으로 사용하는 **RLVR(Reinforcement Learning with Verifiable Rewards)** 프레임워크를 제안하였다. 실험 결과, LaViPlan은 SFT보다 적은 데이터로도 OOD 시나리오에서 더 안전하고 정확한 경로를 생성하였으며, 모델의 추론 스타일을 언어적 묘사 중심에서 기능적 위험 요소 중심으로 변화시켰다. 이는 VLM을 실제 제어 시스템에 정렬시키기 위한 효과적인 포스트 트레이닝 패러다임을 제시한 연구이다.
