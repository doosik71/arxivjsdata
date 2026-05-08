# A Survey on Explainable Deep Reinforcement Learning

Zelei Cheng, Jiahao Yu, Xinyu Xing

## Problem to Solve

심층 강화 학습(**DRL**(Deep Reinforcement Learning))은 순차적 의사결정 작업에서 뛰어난 성능을 보였지만, 블랙박스 신경 아키텍처에 의존하여 **해석 가능성**(interpretability), 신뢰, 그리고 고위험 애플리케이션에서의 배포를 어렵게 합니다. 이러한 불투명성은 헬스케어, 자율 주행, 금융 위험 평가와 같이 **해석 가능성**(explainability)이 필수적인 분야에서 **DRL**의 채택을 제한합니다. 또한, 인간 피드백 기반 강화 학습(**RLHF**(Reinforcement Learning from Human Feedback))을 통해 거대 언어 모델(**LLM**(Large Language Model))과 강화 학습(**RL**(Reinforcement Learning))이 통합되면서 **LLM**의 인간 선호도 정렬에 중요한 역할을 하지만, **RL** 업데이트와 신경 표현 간의 복잡한 상호 작용이 제대로 이해되지 않아 **해석 가능성**(explainability) 문제가 더욱 증폭되고 있습니다.

## Key Contributions

이 설문 조사는 **XRL**(Explainable Deep Reinforcement Learning) 분야의 최신 발전을 종합적으로 검토하며 다음을 포함합니다.

- **XRL** 방법론의 체계적인 분류 및 분석(**특징 수준**(feature-level), **상태 수준**(state-level), **데이터셋 수준**(dataset-level), **모델 수준**(model-level) 설명 기술 포함).
- **XRL** 기술 평가를 위한 정성적 및 정량적 평가 프레임워크 제시.
- 정책 개선, 적대적 견고성, 보안, 그리고 **LLM**과의 통합(**RLHF** 포함)에서 **XRL**의 역할 탐구.
- **해석 가능**(interpretable)하고 신뢰할 수 있으며 책임 있는 **DRL** 시스템 개발을 위한 미해결 연구 과제 및 향후 방향 제시.

## Methodology

이 설문 조사는 **DRL** 및 **LLM**에서 **해석 가능성**(interpretability)을 제공하는 기존 방법들을 체계적으로 검토하고 분석합니다.

- **설명 기술 분류**: **DRL** 에이전트의 의사결정 과정을 설명하는 기술을 다음 네 가지 주요 범주로 분류합니다.
  - **특징 수준 설명 방법**: 에이전트 관찰 공간에서 의사결정에 영향을 미치는 가장 중요한 특징을 식별합니다. **섭동 기반**(perturbation-based), **그래디언트 기반**(gradient-based), **어텐션 기반**(attention-based) 방법이 포함됩니다.
  - **상태 수준 설명 방법**: 에이전트의 궤적에서 성능에 중요한 영향을 미치는 상태를 식별합니다. **오프라인 궤적**(offline trajectories) 분석과 **온라인 상호 작용**(online interactions)을 통한 방법으로 나뉩니다.
  - **데이터셋 수준 설명 방법**: **RL** 에이전트의 학습된 정책에 특정 훈련 예제가 어떻게 영향을 미치는지 이해하는 데 중점을 둡니다. **영향 함수**(influence functions), **데이터 셰플리 값**(Data Shapley Values), **데이터 마스킹**(Data Masking)과 같은 접근 방식이 있습니다.
  - **모델 수준 설명 방법**: **RL** 정책 모델의 **자체 설명 가능성**(self-explainability)에 초점을 맞춥니다. **투명한 아키텍처**(transparent architectures) 설계(예: 의사결정 트리) 또는 인간이 이해할 수 있는 규칙 추출이 포함됩니다.
- **XRL 평가 방법**: 설명의 품질을 정성적 및 정량적 관점에서 평가하는 방법을 설명합니다.
  - **정성적 평가**: **해석 가능성**(interpretability) 및 **명료성**(clarity)을 사용자 연구(**user studies**)를 통해 평가합니다.
  - **정량적 평가**: **충실도**(fidelity) 및 **신뢰성**(faithfulness)을 측정합니다. 특히 **섭동 기반 접근 방식**(perturbation-based approach)을 사용하여 중요한 특징/상태/데이터 포인트 제거 시 에이전트 성능 변화를 측정하고, 설명이 에이전트 성능 개선에 기여하는지 **다운스트림 성능 영향**(downstream performance impact)으로 평가합니다.
- **설명의 응용**: **RL** 설명이 어떻게 활용될 수 있는지 탐구합니다.
  - **적대적 공격**: 설명이 에이전트의 취약점을 노출하고 공격을 유도하는 데 사용될 수 있음을 보여줍니다.
  - **적대적 공격 완화**: 설명이 에이전트의 취약한 상태나 행동을 식별하여 견고한 정책을 설계하는 데 도움을 줍니다(예: **관찰 블라인딩**(blinding observations), **백도어 트리거 차단**(shielding backdoor triggers)).
  - **정책 개선**: **DRL** 에이전트의 정책 오류를 수정하는 데 설명을 사용하며, **인간 참여형 수정**(human-in-the-loop correction)과 **자동화된 정책 개선**(automated policy refinement) 두 가지 범주를 포함합니다.

## Results

이 설문 조사를 통해 **XRL** 방법의 효과와 적용에 대한 다음과 같은 주요 결과가 도출되었습니다.

- **DRL** 에이전트의 의사결정 과정에 대한 **특징 수준**(feature-level), **상태 수준**(state-level), **데이터셋 수준**(dataset-level), **모델 수준**(model-level)의 다양한 설명 기법이 존재하며, 각 기법은 특정 통찰력을 제공합니다.
- **XRL** 시스템은 사용자 연구를 통한 **해석 가능성**(interpretability) 및 **명료성**(clarity)과 같은 **정성적 측면**(qualitative aspects)과, **섭동 기반 접근 방식**(perturbation-based approaches)을 통한 **충실도**(fidelity) 및 **신뢰성**(faithfulness)과 같은 **정량적 측면**(quantitative aspects)을 모두 사용하여 평가될 수 있습니다. 특히, 높은 **충실도**(fidelity)의 설명 방법은 더 효과적인 공격으로 이어질 수 있음을 보여줍니다.
- **XRL**은 양날의 검과 같이 작용합니다. 한편으로는 **정책 개선**(policy refinement) 및 **디버깅**(debugging)을 위한 강력한 도구로 활용될 수 있으며, 중요한 상태나 행동을 식별함으로써 정책의 오류를 수정하고 성능을 향상시킬 수 있습니다.
- 다른 한편으로는 **적대적 공격**(adversarial attacks)을 유도하는 데 사용될 수 있습니다. 에이전트의 의사결정 과정에 대한 통찰력은 공격자가 최소한의 개입으로 최대의 피해를 입히는 데 활용될 수 있는 취약점을 드러냅니다.
- 미래 연구는 비전문가 사용자를 위한 **전략 수준**(strategy-level) 또는 **내러티브 기반**(narrative-based) 설명과, 개발자를 위한 **실용적**(actionable)이고 **성능 향상**(performance-enhancing)에 기여하는 **메커니즘적 해석 방법**(mechanistic interpretation methods)에 중점을 두어, 사용자 및 개발자 지향적인 **해석 가능성**(explainability)을 제공해야 합니다.
