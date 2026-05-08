# Towards Trustworthy Foundation Models for Medical Image Analysis

Congzhen Shi, Ryan Rezai, Jiaxi Yang, Qi Dou, Xiaoxiao Li (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석 분야에서 급격히 발전하고 있는 Foundation Model(기반 모델)의 도입 과정에서 발생하는 **신뢰성(Trustworthiness)** 문제에 주목한다. 의료 분야의 AI 모델은 진단 정확도와 환자의 치료 결과에 직접적인 영향을 미치므로, 단순한 성능 향상을 넘어 프라이버시, 강건성, 신뢰도, 설명 가능성, 공정성이라는 다섯 가지 핵심 요소가 반드시 보장되어야 한다.

현재 의료 영상 분석을 위한 Foundation Model에 대한 여러 서베이 논문들이 존재하지만, 대다수가 모델의 구조나 성능에 집중할 뿐 신뢰성 측면의 심층적인 분석은 부족한 실정이다. 또한, 일반적인 AI 모델의 신뢰성을 다룬 연구들은 의료 영상 도메인만이 가지는 특수성(예: 민감한 환자 데이터, 전문적인 임상 지식의 필요성 등)을 충분히 반영하지 못하고 있다. 따라서 본 논문의 목표는 의료 영상 분석에 사용되는 Foundation Model의 체계적인 분류 체계(Taxonomy)를 제시하고, 각 응용 분야별로 어떤 신뢰성 문제가 발생하는지 분석하며, 이를 해결하기 위한 전략과 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 의료 영상 Foundation Model의 활용 방식과 신뢰성 이슈를 통합적으로 연결하여 분석한 최초의 서베이 연구라는 점에 있다. 주요 기여 사항은 다음과 같다.

1. **신뢰성 프레임워크 정의**: 의료 영상 Foundation Model에서 고려해야 할 신뢰성 요소를 프라이버시(Privacy), 강건성(Robustness), 신뢰도(Reliability), 설명 가능성(Explainability), 공정성(Fairness)의 다섯 가지 관점으로 정의하였다.
2. **응용 분야별 분석**: 의료 영상 분석의 4대 핵심 작업인 분할(Segmentation), 보고서 생성(Report Generation), 의료 질의응답(Medical Q&A), 질환 진단(Disease Diagnosis)을 중심으로 최신 모델들을 검토하고 분류하였다.
3. **취약점 및 전략 규명**: 각 작업별로 어떤 신뢰성 문제가 두드러지게 나타나는지(예: 보고서 생성에서의 Hallucination, 진단에서의 인종/성별 편향 등)를 분석하고, 현재 제안된 해결 전략들의 한계를 지적하였다.
4. **미래 방향성 제시**: 데이터셋 및 벤치마크 구축, Hallucination 억제, 임상 워크플로우와의 정렬(Alignment), 윤리적 가이드라인 수립 등 향후 연구가 나아가야 할 방향을 구체적으로 제안하였다.

## 📎 Related Works

본 논문은 기존의 서베이 연구들과의 차별점을 명확히 하기 위해 비교 분석을 수행하였다. 기존 연구들은 크게 세 가지 방향으로 나뉜다.

- **의료 영상 Foundation Model 중심**: 모델의 구조나 적용 사례는 상세히 다루지만, 신뢰성(Trustworthiness)에 대한 심층 분석이 부족하다.
- **일반 Foundation Model의 신뢰성 중심**: 대규모 언어 모델(LLM) 등의 신뢰성을 다루지만, 의료 영상이라는 특수 도메인의 응용 사례나 제약 사항을 반영하지 않는다.
- **의료 영상의 신뢰성 중심**: 전통적인 딥러닝 모델의 신뢰성은 다루고 있으나, 최근의 Foundation Model(예: SAM, CLIP, GPT-4 등)이 가지는 특성(예: Zero-shot 능력, Prompting 등)을 반영하지 못한다.

본 연구는 이 세 가지 영역을 통합하여, Foundation Model의 적응 방식(Adaptation)과 의료 영상의 특수성, 그리고 신뢰성 지표를 동시에 고려함으로써 기존 문헌의 공백을 메우고 있다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하는 논문이 아닌 서베이 논문이므로, 분석을 위해 사용한 체계적인 프레임워크와 분류 기준을 Methodology로 정의할 수 있다.

### 1. Foundation Model의 적응 방식(Adaptation Methods)

의료 영상 분야에서 기반 모델을 사용하는 방식은 다음과 같이 네 가지로 분류된다.

- **Training From Scratch**: Transformer 아키텍처 등을 사용하여 대규모 의료 데이터로 처음부터 학습시키는 방식이다. MAE(Masked Autoencoders)와 같은 Self-Supervised Learning(SSL) 기법이 주로 사용된다.
- **Fine-Tuning**: 사전 학습된 모델의 파라미터를 의료 데이터에 맞게 조정하는 방식이다. 계산 비용을 줄이기 위해 LoRA(Low-Rank Adaptation)와 같은 Parameter-Efficient Tuning이 활용되며, 전문가의 피드백을 반영하는 RLHF(Reinforcement Learning with Human Feedback)가 적용되기도 한다.
- **Prompt-Tuning**: 모델의 백본은 고정하고 입력 프롬프트를 최적화하는 방식이다. 텍스트 설명이나 포인트, 박스 형태의 가이드를 통해 모델이 특정 의료 영상 특징에 집중하게 만든다.
- **Off-the-Shelf**: 수정 없이 사전 학습된 모델을 그대로 사용하는 방식이며, 주로 In-context learning을 통해 추론 시 예시를 제공하여 성능을 높인다.

### 2. 신뢰성 평가 차원(Trustworthiness Perspectives)

모델의 신뢰성을 다음 다섯 가지 지표로 분석한다.

- **Privacy ($\text{T}_1$)**: 환자의 민감 정보 보호 및 데이터 누출 방지.
- **Robustness ($\text{T}_2$)**: 노이즈, 적대적 공격, 데이터 분포 변화에도 일관된 성능 유지.
- **Reliability ($\text{T}_3$)**: 거짓 정보를 생성하는 Hallucination 방지 및 결과의 일관성 확보.
- **Explainability ($\text{T}_4$)**: 모델이 왜 그런 결정을 내렸는지 의료 전문가가 이해할 수 있도록 시각화하거나 설명하는 능력.
- **Fairness ($\text{T}_5$)**: 인종, 성별, 연령 등 인구통계학적 그룹 간의 성능 격차 해소.

## 📊 Results

논문은 2019년부터 2024년까지 발표된 76편의 논문에서 31개의 Foundation Model을 분석하였으며, 주요 결과는 다음과 같다.

### 1. 영상 분할 (Segmentation)

- **주요 모델**: SAM(Segment Anything Model) 및 그 변형들(MedSAM, MedLSAM 등).
- **신뢰성 이슈**:
  - **프라이버시**: Federated Learning(연합 학습)을 통해 데이터 공유 없이 모델을 튜닝하여 보호하려는 시도가 있다.
  - **강건성**: 의료 영상의 모호한 경계로 인해 Zero-shot 성능이 불안정하며, 수술 중 영상의 블러(Blur)나 반사광 등에 취약하다. 이를 해결하기 위해 불확실성 추정(Uncertainty Estimation) 기법이 제안되었다.
  - **공정성**: 환자의 BMI나 성별에 따라 분할 성능의 격차가 발생함이 확인되었다.

### 2. 보고서 생성 (Report Generation)

- **주요 모델**: LLaMA-2, GPT-4, MedCLIP 등.
- **신뢰성 이슈**:
  - **신뢰도**: 실제 환자 보고서에 포함된 '이전 기록 참조' 문구가 학습되어, 이미지 정보가 없음에도 허구의 이전 기록을 언급하는 Hallucination이 발생한다.
  - **설명 가능성**: 텍스트 보고서의 특정 단어를 영상의 특정 영역(Bounding Box)과 매칭시키는 Adaptive Patch-Word Matching 기법 등이 제안되어 해석력을 높이고 있다.

### 3. 의료 질의응답 (Medical Q&A)

- **주요 모델**: ChatGPT, Mistral-7B, Med-Flamingo 등.
- **신뢰성 이슈**:
  - **강건성**: 프롬프트의 단어 순서나 구조 변화에 성능이 민감하게 반응하며, CLIP 기반 모델은 Backdoor Attack에 취약한 모습을 보인다.
  - **신뢰도**: 전문 지식의 부재로 인한 오답 생성이 심각하며, 이를 방지하기 위해 RAG(Retrieval-Augmented Generation)를 통해 최신 의학 지식을 외부에서 검색하여 주입하는 방식이 사용된다.
  - **공정성**: 학습 데이터셋(예: SLAKE, CheXpert)의 인구통계학적 불균형으로 인해 특정 인종이나 연령대에서 성능이 저하되는 현상이 관찰되었다.

### 4. 질환 진단 (Disease Diagnosis)

- **주요 모델**: GPT-4, CheXzero, MedCLIP 등.
- **신뢰성 이슈**:
  - **프라이버시**: Differential Privacy(차분 프라이버시)를 적용한 파인튜닝이 성능 저하를 최소화하면서 프라이버시를 보호할 수 있음이 확인되었다.
  - **설명 가능성**: 단순 라벨링이 아닌 질환의 시각적 특징을 텍스트로 기술하는 Concept-Bottleneck Model 등이 제안되어 진단 근거를 명확히 하고 있다.
  - **공정성**: 특히 소수자 그룹(Marginalized groups)에 대해 과소 진단(Underdiagnose)하는 경향이 있으며, 이를 해결하기 위한 FairCLIP과 같은 공정성 강화 모델이 연구되고 있다.

## 🧠 Insights & Discussion

본 논문은 Foundation Model이 의료 영상 분석의 패러다임을 바꿀 잠재력이 크지만, 실전 배치를 위해서는 '성능'보다 '신뢰성'이 더 큰 진입장벽이 될 것임을 시사한다.

**강점 및 통찰**:

- 단순히 모델을 나열한 것이 아니라, **[적응 방식 $\rightarrow$ 응용 작업 $\rightarrow$ 신뢰성 요소]**로 이어지는 다층적 분석 구조를 통해 각 단계에서 발생할 수 있는 위험 요소를 체계적으로 짚어냈다.
- 특히 RLHF의 필요성을 주장하면서도, 의료 분야에서는 전문가의 시간 비용(Annotation Cost)과 주관적 진단 차이로 인해 일반 분야보다 RLHF 적용이 훨씬 어렵다는 실무적 한계를 정확히 지적하였다.

**한계 및 논의 사항**:

- 분석 대상이 된 31개의 모델 중 상당수가 여전히 실험실 단계의 벤치마크 성능에 의존하고 있으며, 실제 임상 환경에서의 신뢰성을 검증한 사례는 부족하다.
- 신뢰성 지표(T1~T5) 간의 트레이드-오프(Trade-off) 관계(예: 프라이버시 강화 $\rightarrow$ 성능 저하)에 대한 정량적인 분석은 충분히 다뤄지지 않았다.

## 📌 TL;DR

본 논문은 의료 영상 분석을 위한 Foundation Model의 **신뢰성(프라이버시, 강건성, 신뢰도, 설명 가능성, 공정성)**을 분석한 최초의 종합 서베이 보고서이다. 분할, 보고서 생성, Q&A, 진단이라는 4대 작업별로 발생하는 신뢰성 취약점과 해결 전략을 분류 체계(Taxonomy) 형식으로 제시하였다. 결론적으로, 의료 AI의 실질적인 임상 적용을 위해서는 단순 성능 향상이 아닌, 임상 워크플로우와의 정렬(Alignment)과 엄격한 윤리적/기술적 신뢰성 검증 체계 구축이 필수적임을 강조한다.
