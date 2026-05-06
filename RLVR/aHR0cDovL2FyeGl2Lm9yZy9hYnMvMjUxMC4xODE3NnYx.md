# LOCAL COHERENCE OR GLOBAL VALIDITY? INVESTIGATING RLVR TRACES IN MATH DOMAINS

Soumya Rani Samineni, Durgesh Kalwar, Vardaan Gangal, Siddhant Bhambri, Subbarao Kambhampati (2025)

## 🧩 Problem to Solve

최근 DeepSeek R1과 같은 모델의 등장으로, 정답 가능 여부를 확인할 수 있는 보상을 사용하는 Reinforcement Learning with Verifiable Rewards (RLVR) 기반의 포스트 트레이닝이 LLM의 추론 능력을 향상시키는 방법으로 큰 주목을 받고 있다. 그러나 기존의 RLVR 방법론들은 대부분 최종 정답의 정답 여부나 Pass@K 정확도에만 의존하여 성능을 평가하며, 보상을 모든 토큰에 균등하게 배분하는 특성이 있다.

이로 인해 RLVR이 실제로 모델의 '추론 과정(Reasoning Traces)' 자체를 개선하는지에 대한 엄밀한 분석이 부족한 상태이다. 즉, 최종 정답이 맞았다고 해서 그 과정이 논리적으로 타당한지, 아니면 단순히 운 좋게 정답에 도달했는지가 불분명하다는 점이 이 연구가 해결하고자 하는 핵심 문제이다. 본 논문의 목표는 RL 포스트 트레이닝이 직접적인 보상을 받지 않는 중간 토큰(intermediate tokens), 즉 추론 경로의 질에 어떤 영향을 미치는지 조사하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 RLVR이 추론의 '논리적 타당성(Validity)'이 아닌 '국소적 일관성(Local Coherence)'을 개선시킨다는 점을 밝혀낸 것이다.

저자들은 추론 경로의 일관성을 측정하기 위해 일차 논리(First-Order Logic, FOL)에 기반한 **Trace Coherence**라는 새로운 지표를 제안하였다. 이를 통해 RLVR이 추론 단계 간의 오류를 줄여 외견상 일관된 경로를 생성하게 만들지만, 이것이 반드시 전체적인 논리적 정당성이나 최종 정답의 보장으로 이어지지는 않는다는 점을 실험적으로 증명하였다.

## 📎 Related Works

기존 연구들은 주로 추론 경로의 길이, 구조, 혹은 사용자 해석 가능성(Interpretability)에 집중하였다. 일부 연구에서는 SFT(Supervised Fine-Tuning) 과정에서 추론 경로의 타당성과 최종 정답의 정답률 사이에 상관관계가 낮다는 점을 지적하였다.

또한, 수학적 추론의 오류를 분석하기 위한 다양한 분류 체계(Minerva CoT, Examiner 등)가 제안되었으나, 저자들은 이러한 기존 분류들이 서로 중복되거나 상호 배타적이지 않아 공식적인 일관성 측정에 한계가 있다고 주장한다. 본 논문은 이를 해결하기 위해 FOL 기반의 엄격하고 상호 배타적인 오류 분류 체계를 도입하여 기존 연구와 차별화를 두었다.

## 🛠️ Methodology

### 1. FOL 기반 오류 분류 체계 (Error Taxonomy)

저자들은 수학적 추론 과정을 '가정/사실 $\rightarrow$ 수식/공식 $\rightarrow$ 중간 계산 단계'의 구조로 파악하고, 이를 FOL로 표현하여 다음과 같이 상호 배타적인 오류 카테고리를 정의하였다.

- **False Premise**: 문제의 조건을 잘못 이해하거나 잘못된 가정을 세운 경우 (예: 단위 오해, 잘못된 데이터 사용).
- **False Rule**: 수학적 논리나 공식 적용이 틀린 경우 (예: 잘못된 공식 사용, 필수 계산 단계 누락).
- **Calculator Error**: 단순한 산술 계산 실수 (예: $5 \times 6 = 10$).
- **Format Error**: 최종 정답을 지정된 형식(예: $\boxed{}$)으로 작성하지 않은 경우.

### 2. Trace Coherence 측정 방법

추론 경로의 타당성(Validity)을 대규모로 검증하는 것은 어렵기 때문에, 저자들은 **Trace Coherence**를 대리 지표(Proxy metric)로 사용한다.

- **측정 프로세스**: $\text{Response} \rightarrow \text{GPT-4o (FOL 변환 및 오류 분류)} \rightarrow \text{Error Tagging}$
- **Pass@K Accuracy**: $k$개의 응답 중 하나라도 정답이면 정답으로 간주한다.
- **Pass@K Trace Coherence**: 정답을 맞힌 $c$개의 응답 중에서, 적어도 하나 이상의 응답이 오류가 없는(error-free) 경우에만 '일관성이 있다(coherent)'고 정의한다.

### 3. 실험 설정

- **데이터셋**: GSM8K (초등 수학 문제)
- **모델**: Qwen-2.5-0.5B (Base model)
- **알고리즘**: GRPO (Group Relative Policy Optimization)
- **평가 도구**: GPT-4o (LLM-as-a-Judge)

## 📊 Results

실험 결과, RLVR 포스트 트레이닝은 전반적으로 Trace Coherence를 향상시키는 것으로 나타났다. 특히 다음과 같은 패턴에서 유의미한 결과가 관찰되었다.

- **Pattern 01 (Base model 실패 $\rightarrow$ RL model 성공)**: RL 모델이 정답을 맞히게 되면서 Trace Coherence가 0%에서 약 85%까지 급격히 상승하였다.
- **Pattern 11 (둘 다 성공)**: RL 모델이 Base 모델보다 더 높은 Trace Coherence(최대 96%)를 보였다.
- **정량적 분석**: LLM-as-a-Judge의 오류 분류 정확도는 약 57.8%로 측정되었으며, 이는 FOL 변환 과정에서 Format Error의 재현율(Recall)이 낮았기 때문으로 분석된다.

결과적으로 RLVR은 모델이 정답을 맞히는 확률을 높일 뿐만 아니라, 그 과정에서 나타나는 국소적인 오류들을 제거하여 추론 경로를 더 "그럴듯하게(coherent)" 만드는 효과가 있음이 확인되었다.

## 🧠 Insights & Discussion

본 연구의 가장 중요한 통찰은 **"국소적 일관성(Local Coherence)의 향상이 곧 글로벌한 논리적 타당성(Global Validity)을 의미하지 않는다"**는 점이다.

RLVR은 최종 보상을 극대화하는 방향으로 학습되기 때문에, 모델은 정답에 도달하기 위해 논리적으로 무결한 경로를 설계하기보다, 오류가 적어 보이고 일관성 있어 보이는 경로를 생성하는 법을 배울 가능성이 크다. 이는 RLVR이 추론 능력을 근본적으로 향상시켰다는 주장보다는, '정답으로 향하는 경로의 외견상 품질'을 개선했다는 해석이 더 적절함을 시사한다.

따라서 RL 포스트 트레이닝을 통한 추론 능력 향상을 주장할 때는 단순히 최종 정답률의 상승뿐만 아니라, 제안된 Trace Coherence와 같은 세밀한 분석을 통해 실제 논리적 추론 과정이 개선되었는지 검증해야 한다는 비판적 시각을 제시한다.

## 📌 TL;DR

본 논문은 RLVR 기반의 학습이 LLM의 수학적 추론 경로에 미치는 영향을 분석하였다. FOL 기반의 오류 분류 체계를 통해 분석한 결과, RLVR은 추론 단계 간의 오류를 줄여 **국소적 일관성(Local Coherence)**을 크게 향상시키지만, 이것이 반드시 전체적인 **논리적 타당성(Global Validity)**이나 정답 보장으로 이어지는 것은 아님을 밝혔다. 이는 향후 RL 기반 추론 모델 평가 시, 최종 정답률 외에 추론 경로의 실제 논리적 무결성을 검증하는 것이 필수적임을 시사한다.
