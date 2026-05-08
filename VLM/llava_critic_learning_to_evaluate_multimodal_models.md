# LLaVA-Critic: Learning to Evaluate Multimodal Models

Tianyi Xiong et al. (2025)

## 🧩 Problem to Solve

본 논문은 Large Multimodal Models (LMMs)의 성능을 평가하는 과정에서 발생하는 높은 비용과 폐쇄적인 생태계 문제를 해결하고자 한다. 현재 복잡한 멀티모달 작업의 평가는 숙련된 인간 평가자의 노동력에 의존하거나, GPT-4V나 GPT-4o와 같은 고비용의 상용 폐쇄형 모델을 Judge로 사용하는 방식에 크게 의존하고 있다.

이러한 접근 방식은 다음과 같은 한계를 가진다. 첫째, 상용 모델의 API 비용이 매우 높으며, 둘째, 평가 기준을 모델에 맞게 세밀하게 조정(customization)하는 데 제약이 있다. 따라서 본 연구의 목표는 상용 모델에 필적하는 평가 능력을 갖춘 첫 번째 오픈소스 일반ist 평가 모델인 LLaVA-Critic을 개발하여, LMM-as-a-Judge 및 Preference Learning을 위한 효율적인 대안을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LMM이 단순한 답변 생성을 넘어, 정량적인 점수 부여와 논리적인 근거 제시라는 '비판적 평가(Critic)' 능력을 학습하도록 하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Critic Instruction-Following Dataset 구축**: 포인트와 페어 방식의 평가 시나리오를 모두 포함하며, 다양한 평가 기준과 도메인을 아우르는 고품질의 데이터셋(이미지 46k, 샘플 113k)을 구축하였다.
2. **LLaVA-Critic 모델 개발**: LLaVA-OneVision을 기반으로 하여, 오픈소스 모델이 상용 모델 수준의 평가 및 피드백 능력을 갖추도록 파인튜닝하였다.
3. **범용 평가자 및 보상 모델로서의 활용**: LLaVA-Critic이 단순한 성능 측정을 위한 Judge 모델뿐만 아니라, DPO(Direct Preference Optimization)와 같은 정렬(Alignment) 학습을 위한 효과적인 Reward Signal 생성기로 활용될 수 있음을 입증하였다.

## 📎 Related Works

### LMM-as-a-Judge

GPT-4V/-4o와 같은 강력한 상용 모델들이 이미 비전-언어 작업의 일반 평가자로 사용되고 있다. 오픈소스 진영에서는 Prometheus-Vision이 사용자 정의 스코어링 기준에 따라 훈련된 평가 모델로 제시되었으나, LLaVA-Critic은 보다 광범위한 시나리오에 대응하는 '일반ist 평가자'라는 점에서 차별점을 가진다.

### Preference Learning for LMMs

RLHF(Reinforcement Learning from Human Feedback)와 DPO는 LLM의 정렬을 위해 널리 사용되어 왔다. LMM 영역에서도 LLaVA-RLHF, RLHF-V 등이 인간의 선호도를 학습시키려 시도하였으나, 이는 여전히 값비싼 인간 피드백 수집 과정이 필요하다는 한계가 있다. LLaVA-Critic은 AI-generated feedback을 통해 이러한 인간 의존도를 낮추고 확장 가능한(scalable) 정렬 메커니즘을 제공하고자 한다.

## 🛠️ Methodology

### 1. 데이터 수집 (Data Collection)

LLaVA-Critic의 학습 데이터는 GPT-4o를 이용해 생성된 두 가지 설정으로 구성된다.

- **Pointwise Scoring**: 단일 응답에 대해 미리 정의된 기준에 따라 점수를 부여하는 방식이다. 데이터 구조는 다음과 같다:
    $$\text{(Image, Question, Response, Reference, Evaluation Criteria, Score, Reason)}$$
    8개의 멀티모달 지시어 데이터셋과 7개의 벤치마크 프롬프트 풀을 사용하여 72,782개의 샘플을 생성하였다.
- **Pairwise Ranking**: 두 개의 응답 중 어느 것이 더 우수한지 결정하는 방식이다. VLFeedback, RLHF, RLHF-V 데이터셋에서 선호도 관계를 수집하고, GPT-4o를 통해 그 이유(Reason)를 생성하여 40.1k개의 샘플을 구축하였다.

### 2. 모델 아키텍처 및 학습

LLaVA-Critic은 LLaVA-OneVision (7B/72B) 체크포인트를 기반으로 하며, 구축된 LLaVA-Critic-113k 데이터셋을 사용하여 1 epoch 동안 파인튜닝되었다. 학습 시에는 정량적 판단(Score/Ranking)과 정성적 근거(Justification) 모두에 대해 표준 Cross-Entropy Loss를 적용하였다.

### 3. 활용 시나리오

#### Scenario 1: LMM-as-a-Judge

LLaVA-Critic을 자동 평가 도구로 사용하여 시각적 채팅, 통합 역량, 선호도, 상세 묘사, 환각(Hallucination) 여부를 평가한다.

#### Scenario 2: Preference Learning (Iterative DPO)

LLaVA-Critic을 Reward Model로 사용하여 모델의 성능을 반복적으로 개선하는 파이프라인을 제안한다.

1. **응답 생성**: 초기 모델 $\pi_0$가 하나의 질문-이미지 쌍에 대해 $K$개의 응답 $\{y_1, \dots, y_K\}$를 생성한다.
2. **스코어링**: 모든 가능한 쌍 $(y_i, y_j)$에 대해 LLaVA-Critic이 상대적 점수 $a_{ij}$를 생성한다.
3. **보상 계산**: 각 응답 $y_i$에 대한 최종 보상 점수 $r_i$를 다음과 같이 계산한다.
    $$r_i = \sum_{k \neq i} a_{ki} - \sum_{l \neq i} a_{il}$$
4. **DPO 학습**: 가장 높은 보상을 받은 $y^+$와 가장 낮은 보상을 받은 $y^-$를 선택하여 DPO 학습을 진행한다. 이 과정을 $M$번 반복하여 모델을 점진적으로 개선한다.

## 📊 Results

### 1. LMM-as-a-Judge 성능

- **In-domain Pointwise**: LLaVA-Critic-72B는 GPT-4o와의 Pearson 상관계수 평균 0.754를 기록하며, 베이스라인 모델(0.634)을 크게 상회하였다. 7B 모델 또한 0.732로 매우 강력한 성능을 보였다.
- **In-domain Pairwise**: LLaVA-Critic-72B는 Tie를 제외한 정확도 73.6%를 달성하여 GPT-4o(73.4%) 및 GPT-4V(73.3%)와 대등하거나 이를 능가하는 성능을 보였다.
- **Out-of-domain (MLLM-as-a-Judge)**: 다양한 벤치마크에서 인간 평가자와의 정렬도를 측정한 결과, LLaVA-Critic-72B의 Pearson 유사도는 0.393으로, 오픈소스 모델 중 가장 높았으며 상용 모델과의 격차를 크게 좁혔다.

### 2. Preference Learning 성능

Iterative DPO를 통해 LLaVA-OneVision을 학습시킨 결과, LLaVA-Critic을 Reward Model로 사용했을 때 visual chat 능력이 크게 향상되었다.

- **결과**: LLaVA-W, LLaVA-Wilder, WildVision-Bench 등 6개 벤치마크 중 5~6개에서 기존의 인간 피드백 기반 LLaVA-RLHF 보상 모델보다 우수한 성능을 보였다.
- **일반화**: 이미지 데이터로만 정렬 학습을 진행했음에도 불구하고, Video Detailed Captioning 성능이 향상되어 비디오 컨텍스트로의 일반화 가능성을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **Scaling의 효과**: 데이터의 다양성과 모델의 크기가 증가함에 따라 평가 성능이 일관되게 향상됨을 확인하였다. 특히 LLaVA-Critic-7B가 상용 모델에 상당히 근접한 성능을 낸다는 점은 자원 제한적인 환경에서의 배포 가능성을 시사한다.
- **상호 보완적 학습**: Ablation Study를 통해 Pointwise 데이터와 Pairwise 데이터가 서로의 성능을 향상시키는 상호 보완적 관계임을 밝혀냈으며, 특히 Tie(동점) 데이터가 미세한 품질 차이에 과적합되는 것을 방지하여 성능을 높였음을 확인하였다.

### 한계 및 논의

- **GPT 의존적 데이터 생성**: 학습 데이터 구축 과정에서 GPT-4o를 Teacher 모델로 사용했기 때문에, LLaVA-Critic의 상한선이 GPT-4o의 평가 능력에 종속될 가능성이 있다.
- **평가 일관성**: 정성적 분석에서 LLaVA-Critic이 LLaVA-OV보다 훨씬 구체적이고 이미지에 근거한(grounded) 근거를 제시함을 확인하였으나, 여전히 복잡한 추론이 필요한 작업에서는 상용 모델과의 미세한 간극이 존재한다.

## 📌 TL;DR

본 논문은 오픈소스 LMM을 일반적인 평가자로 활용하기 위해 113k 규모의 평가 특화 데이터셋을 구축하고, 이를 통해 **LLaVA-Critic** 모델을 제안하였다. LLaVA-Critic은 상용 모델(GPT-4o) 수준의 포인트 및 페어 평가 능력을 갖추어 **LMM-as-a-Judge**로서 비용 효율적인 대안을 제시하며, 동시에 고품질의 보상 신호를 생성하여 **Iterative DPO**를 통한 LMM의 자가 개선(Self-improvement)을 가능하게 한다. 이는 향후 인간의 개입 없이도 확장 가능한 superhuman alignment 메커니즘 구축의 토대가 될 것으로 기대된다.
