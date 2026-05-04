# Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents

Renxi Wang, Haonan Li, Xudong Han, Yixuan Zhang, Timothy Baldwin (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)을 에이전트(Agent)로 활용하여 도구 사용 및 환경과의 상호작용 능력을 향상시키는 파인튜닝 과정에서의 데이터 효율성 문제를 다룬다. 기존의 에이전트 튜닝 방식은 주로 강력한 모델(예: GPT-4)을 통해 생성된 상호작용 궤적(Trajectory) 중 작업 완수에 성공한 '긍정적 예시(Positive Examples)'만을 사용하여 작은 모델을 학습시켰다.

이러한 접근 방식은 두 가지 주요 문제를 야기한다. 첫째, 성공한 궤적만을 선택함으로써 전체 데이터의 상당 부분(복잡한 작업의 경우 60% 이상)을 폐기하게 되어 데이터 부족 현상이 발생하며, 데이터 수집 비용이 증가한다. 둘째, 실패한 궤적(Negative Examples)이 제공하는 '무엇이 잘못되었는가'에 대한 잠재적인 통찰력을 활용하지 못함으로써 모델의 최적화 경로를 제한한다. 따라서 본 연구의 목표는 실패한 궤적을 적절한 품질 제어 및 학습 전략을 통해 통합함으로써, 에이전트의 성능을 높이고 데이터 효율성을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델이 성공과 실패의 차이를 명확히 인식할 수 있도록 돕는 **Negative-Aware Training (NAT)** 패러다임을 제안한 것이다. 단순히 긍정적 예시와 부정적 예시를 섞어서 학습시키는 것이 아니라, 쿼리에 접두사(Prefix)나 접미사(Suffix)를 추가하여 모델에게 "이 궤적은 정답을 맞힌 경로이다" 혹은 "이 궤적은 오답을 낸 경로이다"라고 명시적으로 알려주는 방식이다. 이를 통해 모델은 부정적 예시에서 발생한 오류를 그대로 학습하는 것이 아니라, 성공적인 경로와 실패한 경로의 차이를 대조하며 더 정교한 추론 및 계획 능력을 습득하게 된다.

## 📎 Related Works

기존의 LLM 에이전트 연구는 크게 두 가지 방향으로 진행되었다. 하나는 파인튜닝 없이 프롬프팅(Few-shot)에 의존하는 방식이고, 다른 하나는 GPT-4와 같은 고성능 모델이 생성한 성공적인 궤적만을 수집하여 작은 모델을 학습시키는 방식이다. 하지만 후자의 경우, 성공한 데이터만을 사용하기 때문에 데이터 희소성 문제가 발생하며 실패 사례로부터 배우는 메커니즘이 결여되어 있다.

실패로부터 배우는 연구(Learning from negative results)는 주로 프롬프트 기반의 반복적 정제(Iterative Refinement)나 외부 평가자의 피드백을 이용하는 방식이 주를 이루었다. 파인튜닝 기반의 접근법으로는 Chain-of-Thought(CoT) 프롬프트에 집중한 연구가 있었으나, 에이전트의 도구 사용 및 상호작용 시나리오에서 부정적 궤적을 직접적으로 활용하여 파인튜닝한 사례는 본 논문이 처음임을 강조하고 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 논문에서 제안하는 NAT의 전체 파이프라인은 **데이터 수집 $\rightarrow$ 부정-인식 재구성(Negative-Aware Reformatting) $\rightarrow$ 파인튜닝 $\rightarrow$ 추론**의 단계로 구성된다.

1. **에이전트 프레임워크**: ReAct (Reasoning and Acting) 형식을 따른다. 모델은 `Thought`(추론) $\rightarrow$ `Action`(도구 호출) $\rightarrow$ `Observation`(결과 관찰)의 순서로 상호작용하며, 최종적으로 `finish[Answer]` 액션을 통해 작업을 종료한다.
2. **데이터 수집**: GPT-3.5를 사용하여 각 쿼리에 대해 서로 다른 온도(Temperature: 0.2, 0.5, 0.7)로 3번의 궤적을 생성한다. 정답(Ground Truth)과 비교하여 각 궤적을 긍정적(Positive) 또는 부정적(Negative)으로 라벨링한다.
3. **부정-인식 재구성 (Negative-Aware Reformatting)**: 학습 데이터의 쿼리 끝에 다음과 같은 문자열을 추가하여 데이터의 성격을 구분한다.
    * **Positive Sample**: `"Please generate a solution that **correctly** answers the question."`
    * **Negative Sample**: `"Please generate a solution that **incorrectly** answers the question."`
4. **파인튜닝 및 추론**:
    * **학습**: 재구성된 데이터를 사용하여 LLM을 파인튜닝한다. 이때 손실 함수는 모델이 생성한 텍스트 부분에 대해서만 계산된다.
    * **추론**: 실제 서비스 시에는 오직 **Positive Sample**을 위한 프롬프트만을 사용하여 모델이 정답을 생성하도록 유도한다.

## 📊 Results

### 실험 설정

* **데이터셋**: 수학적 추론(GSM8K, ASDiv, SVAMP, MultiArith), 다중 홉 질의응답(HotpotQA), 전략적 질의응답(StrategyQA).
* **모델**: LLaMA-2-Chat 7B 및 13B.
* **비교 대상 (Baselines)**:
  * `Vanilla`: 긍정적 예시만으로 학습.
  * `NUT (Negative-Unaware Training)`: 긍정/부정 예시를 구분 없이 섞어서 학습.
  * `NAT`: 제안 방법론.

### 주요 결과

* **수학 과제**: NAT는 Vanilla 및 NUT보다 일관되게 높은 성능을 보였다. 특히 7B 모델에서 긍정적 예시가 2k개로 적은 저자원 상황일 때, NAT는 Vanilla 대비 8.74%의 성능 향상을 보이며 데이터 희소성 문제 해결에 효과적임을 입증했다.
* **질의응답 과제**: HotpotQA와 StrategyQA에서도 NAT가 가장 우수한 성능을 보였다. StrategyQA의 경우 Vanilla 대비 8% 이상의 성능 향상이 관찰되었다.
* **정량적 지표**: LLaMA-2-13B 모델 기준, GSM8K에서 NAT는 Vanilla(54.21%)보다 높은 53.75%(5k positive 기준, 다만 2k positive 설정에서는 NAT가 50.64%로 Vanilla 44.4%보다 월등히 높음)의 성능을 기록하며, 데이터 양이 적을수록 NAT의 효용성이 극대화됨을 보여주었다.

## 🧠 Insights & Discussion

### 1. 추론 능력과 액션 오류의 트레이드-오프

분석 결과, 부정적 예시를 학습에 포함하면 `Action Error`(도구 호출 오류)는 증가하는 경향이 있으나, 동시에 `Thought`(추론 및 계획) 능력은 향상된다. `NUT`는 추론 능력은 얻었지만 액션 오류가 너무 많아 성능이 낮았으나, `NAT`는 명시적 구분을 통해 액션 오류를 억제하면서 추론 능력을 효과적으로 습득함으로써 최적의 트레이드-오프를 달성했다.

### 2. 데이터 품질의 중요성

부정적 데이터의 품질이 성능에 결정적인 영향을 미친다. GPT-3.5가 생성한 고품질의 부정적 데이터는 성능을 향상시키지만, 작은 모델이 생성한 저품질의 부정적 데이터를 사용했을 때는 오히려 성능이 하락하는 결과가 나타났다.

### 3. 프롬프트의 해석 가능성

연구진은 "Correct/Incorrect" 대신 무작위 문자열을 사용하여 데이터를 구분하는 실험을 진행했다. 그 결과, 프롬프트의 구체적인 의미보다는 단순히 **긍정적 데이터와 부정적 데이터를 서로 다른 레이블로 구분해주는 것 자체**가 모델의 성능 향상에 핵심적이라는 사실을 발견했다.

### 4. 확장 가능성

* **Fine-grained NAT**: 부정적 궤적을 품질(예: F1 score)에 따라 여러 등급으로 나누어 서로 다른 프롬프트를 부여하는 `NAT-k` 방식이 일반 NAT보다 더 높은 성능을 보였다.
* **CoT 적용**: ReAct뿐만 아니라 Chain-of-Thought(CoT) 추론 과정에도 NAT를 적용했을 때 성능 향상이 확인되어, 다양한 추론 전략에 범용적으로 적용 가능함을 보여주었다.

## 📌 TL;DR

본 논문은 LLM 에이전트 튜닝 시 버려지던 **실패한 궤적(Negative Trajectories)**을 학습에 활용하는 **Negative-Aware Training (NAT)** 방법을 제안한다. 단순히 데이터를 섞는 것이 아니라 긍정/부정 예시를 명시적인 프롬프트로 구분하여 학습시킴으로써, 모델이 오류를 복제하지 않고 성공과 실패의 차이를 통해 추론 능력을 학습하게 한다. 특히 데이터가 부족한 저자원 환경이나 작은 모델에서 성능 향상 폭이 크며, 이는 향후 저비용 고효율의 에이전트 튜닝 방법론 개발에 중요한 가이드라인을 제공한다.
