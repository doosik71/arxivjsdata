# Contrastive Chain-of-Thought Prompting

Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, Lidong Bing (2023)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)의 추론 능력을 향상시키기 위해 널리 사용되는 Chain-of-Thought(CoT) 프롬프팅의 작동 원리에 대한 이해 부족과 그로 인한 한계점을 해결하고자 한다.

기존의 CoT는 모델에게 정답으로 가는 중간 추론 단계(intermediate reasoning steps)를 제공함으로써 복잡한 문제를 해결하도록 유도한다. 그러나 기존 방식은 모델에게 '어떤 실수를 피해야 하는지'에 대한 정보를 제공하지 않는다. 이로 인해 추론 과정에서 발생하는 작은 오류가 누적되어 최종 결과가 틀리는 error propagation 문제가 발생하며, 이는 모델의 신뢰성을 저하시킨다. 또한, 일부 선행 연구에서는 유효하지 않은(invalid) 논증을 사용하더라도 CoT의 성능에 큰 영향이 없다는 상반된 결과가 보고되는 등, CoT의 내재적 메커니즘이 명확히 규명되지 않은 상태이다.

따라서 본 연구의 목표는 인간이 정답 사례뿐만 아니라 오답 사례를 통해서도 학습한다는 점에 착안하여, 긍정적 예시와 부정적 예시를 동시에 제공하는 Contrastive Chain-of-Thought 방식을 제안하고 그 효과를 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델에게 올바른 추론 경로($T^+$)와 잘못된 추론 경로($T^-$)를 함께 제시하여, 모델이 무엇이 정답이고 무엇이 오류인지 대비(contrast)하며 학습하게 하는 것이다.

주요 기여 사항은 다음과 같다:

1. **부정적 예시의 영향 분석**: 다양한 유형의 잘못된 추론(invalid reasoning) 사례가 CoT 성능에 미치는 영향을 분석하여, 긍정-부정 예시의 조합이 추론 성능을 향상시킨다는 점을 확인하였다.
2. **Contrastive CoT 프롬프팅 제안**: 정답 추론 과정과 오답 추론 과정을 동시에 제공하는 새로운 프롬프팅 구조를 설계하였다.
3. **자동화된 대조 예시 생성 방법**: 수동 구축의 한계를 극복하기 위해, 기존의 정답 추론 체인에서 핵심 객체들을 무작위로 섞어 잘못된 추론 체인을 자동으로 생성하는 방법을 제안하였다.
4. **범용적 성능 향상 입증**: 산술 추론 및 사실적 질의응답 등 다양한 벤치마크에서 기존 CoT 대비 유의미한 성능 향상을 입증하였으며, 특히 Self-consistency 기법과 결합했을 때 시너지 효과가 나타남을 보였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 바탕으로 차별성을 갖는다.

- **Large Language Models (LLMs)**: 모델 크기의 확장이 일반화 성능을 높였으나, 산술 추론이나 사실적 QA와 같은 복잡한 논리 작업에서는 여전히 한계가 있다.
- **Chain-of-Thought (CoT)**: 중간 단계의 추론 과정을 생성하여 LLM의 능력을 끌어올리는 기법이다. 최근에는 Zero-shot CoT("Let's think step-by-step")나 프로그램을 활용한 PAL 등이 제안되었으나, 추론 과정의 신뢰성 문제와 내부 작동 원리에 대한 이해는 여전히 부족하다.
- **Learning from Negative Examples**: 딥러닝의 Contrastive Learning이나 RLHF(Reinforcement Learning from Human Feedback)는 긍정/부정 샘플을 구분함으로써 더 나은 표현이나 정렬을 학습한다. 본 논문은 이러한 직관을 프롬프팅 단계인 CoT에 적용하여, 기존의 긍정 예시 중심 CoT와 차별화하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

Contrastive CoT는 모델에게 질문과 함께 정답 설명(Positive demonstration)과 오답 설명(Negative demonstration)을 동시에 제공하는 구조이다.

- **Standard Prompting**: $E_j = (Q_j, A_j)$ 형태의 (질문, 정답) 쌍을 제공한다.
- **Conventional CoT**: $E_j = (Q_j, T_j, A_j)$ 형태로 (질문, 추론 과정, 정답)을 제공한다.
- **Contrastive CoT**: $E_j = (Q_j, T_j^+, A_j^+, T_j^-, A_j^-)$ 형태로 (질문, 정답 추론 과정, 정답, 오답 추론 과정, 오답)을 모두 제공한다.

### 부정적 추론 과정($T^-$)의 자동 생성 방법

수동으로 오답 사례를 만드는 비용을 줄이기 위해, 본 논문은 'Incoherent Objects'라는 개념을 활용하여 $T^-$를 자동으로 생성한다.

1. **객체 추출**: Entity recognition 모델(spaCy의 `en_core_web_trf`)을 사용하여 정답 추론 과정($T^+$) 내에서 숫자, 방정식, 인물 이름과 같은 핵심 상징적 아이템인 **Bridging objects**를 추출한다.
2. **무작위 셔플링**: 추출된 Bridging objects의 위치를 추론 과정 내에서 무작위로 섞는다.
3. **결과**: 논리적 순서가 파괴된, 즉 **Coherence(일관성)**가 결여된 잘못된 추론 경로($T^-$)가 생성된다.

### 추론 절차

테스트 시에는 쿼리 질문 $Q$와 함께 위에서 구성된 Contrastive demonstrations 세트가 프롬프트로 입력된다. 모델은 제공된 대조 사례를 참고하여 자신의 추론 단계($T$)를 먼저 생성한 후, 최종 정답($A$)을 도출한다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - 산술 추론(Arithmetic Reasoning): GSM8K, AQuA, GSM-Hard, SVAMP, ASDIV
  - 사실적 QA(Factual QA): Bamboogle, StrategyQA
- **모델**: GPT-3.5-Turbo (0301)
- **평가 지표**: 정답 정확도(Accuracy)
- **비교 대상**: Standard Prompting, Conventional CoT, 그리고 Contrastive CoT (단독 및 Self-consistency 결합)

### 주요 결과

1. **일관된 성능 향상**: Contrastive CoT는 모든 데이터셋에서 기존 CoT보다 높은 성능을 보였다. 특히 GSM-Hard(+10.4), Bamboogle(+16.0), StrategyQA(+10.4)에서 큰 폭의 상승이 있었다.
2. **Self-consistency(SC)와의 시너지**: 여러 추론 경로 중 다수결로 답을 정하는 SC 기법을 적용했을 때, Contrastive CoT-SC는 더욱 압도적인 성능을 기록하였다. (예: AQuA 데이터셋에서 Contrastive CoT 단독은 +4.0% 향상이었으나, SC 결합 시 +15.7% 향상)
3. **예비 연구 결과**: 다양한 오답 유형(Incoherent Objects, Incoherent Language, Irrelevant Objects 등) 중 특히 'Incoherent Objects'를 활용한 대조 예시가 가장 높은 성능 향상을 보였으며, 이는 자동 생성 방법의 근거가 되었다.

## 🧠 Insights & Discussion

### 강점

본 연구는 단순히 '더 많은 데이터'를 주는 것이 아니라, '무엇이 틀렸는가'에 대한 정보를 제공함으로써 모델의 추론 정밀도를 높였다. 특히, 복잡한 수동 어노테이션 없이 기존 정답 셋에서 객체 셔플링이라는 단순한 방법만으로도 효과적인 부정적 예시를 만들 수 있음을 보였다는 점이 실용적이다.

### 한계 및 논의사항

- **오답 유형의 제한**: 자동 생성 방식이 'Incoherent Objects' 유형에 치중되어 있어, 논리적 오류(Logical mistakes)나 관련 없는 정보(Irrelevance)와 같은 다른 유형의 오답 사례를 함께 제공했을 때의 효과는 충분히 탐구되지 않았다.
- **모델 의존성**: GPT-3.5-Turbo라는 특정 모델에서 검증되었으므로, 모델의 크기나 학습 방식에 따라 Contrastive prompt에 반응하는 정도가 다를 수 있다.
- **추론 길이 증가**: 긍정과 부정 예시를 모두 제공하므로 프롬프트의 길이가 길어지며, 이는 토큰 비용 증가 및 컨텍스트 윈도우 제한 문제를 야기할 수 있다.

## 📌 TL;DR

본 논문은 LLM의 추론 능력을 높이기 위해 정답과 오답 추론 과정을 동시에 제공하는 **Contrastive Chain-of-Thought Prompting**을 제안한다. 정답 추론 과정에서 핵심 객체들을 무작위로 섞어 오답 사례를 자동으로 생성하는 효율적인 방법을 도입하였으며, 이를 통해 산술 및 사실적 QA 작업에서 기존 CoT 대비 유의미한 성능 향상을 거두었다. 이 연구는 모델에게 '피해야 할 실수'를 알려주는 것이 추론 정확도 향상에 핵심적인 역할을 함을 시사하며, 향후 LLM의 논리적 정렬 연구에 중요한 방향성을 제시한다.
