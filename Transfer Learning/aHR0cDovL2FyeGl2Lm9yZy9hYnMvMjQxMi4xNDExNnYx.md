# Trustworthy Transfer Learning: A Survey
Jun Wu, Jingrui He

## Problem to Solve

일반적인 기계 학습은 훈련 및 테스트 샘플이 독립적이고 동일하게 분포(**IID**)되어 있다고 가정하지만, 실제 시나리오에서는 이러한 가정이 종종 위배됩니다. 전이 학습(**Transfer Learning**, **TL**)은 훈련(소스 도메인) 데이터와 테스트(타겟 도메인) 데이터 간의 분포 변화를 해결하기 위해 도입되었지만, 전이 학습(**TL**) 과정에서 전이된 지식이 신뢰할 수 있는지에 대한 새로운 질문이 제기됩니다. 이 논문은 **지식 전이 가능성**(**Knowledge Transferability**)과 **지식 신뢰성**(**Knowledge Trustworthiness**)의 두 가지 관점에서 신뢰할 수 있는 전이 학습(**Trustworthy Transfer Learning**)을 포괄적으로 이해하는 것을 목표로 합니다. 주요 연구 질문은 다음과 같습니다:

1.  도메인 간 **지식 전이 가능성**(**knowledge transferability**)은 어떻게 정량적으로 측정되고 향상될 수 있는가?
2.  전이 학습(**TL**) 과정에서 전이된 지식을 신뢰할 수 있는가? (즉, 전이된 지식이 **적대적으로 강건**(**adversarially robust**)한지, **공정한**(**fair**)지, **프라이버시**(**privacy**)를 보존하는지 등)

## Key Contributions

*   **신뢰할 수 있는 전이 학습**(**Trustworthy Transfer Learning**)에 대한 포괄적인 검토를 제공합니다.
*   **지식 전이 가능성**(**knowledge transferability**)을 이해하기 위한 최신 이론과 알고리즘을 **IID** 및 **비-IID**(**non-IID**) 가설 하에서 요약합니다.
*   **프라이버시**(**privacy**), **적대적 강건성**(**adversarial robustness**), **공정성**(**fairness**), **투명성**(**transparency**) 등 **신뢰성**(**trustworthiness**) 속성이 전이 학습(**transfer learning**)에 미치는 영향을 검토합니다.
*   현재의 발전 사항을 논의할 뿐만 아니라, 신뢰할 수 있는 방식으로 전이 학습(**transfer learning**)을 이해하기 위한 미해결 과제와 미래 방향을 제시합니다.

## Methodology

본 논문은 신뢰할 수 있는 전이 학습(**Trustworthy Transfer Learning**) 분야를 체계적으로 조사합니다.

*   **초기 설정 및 문제 정의**: 입력 공간 $X$와 출력 공간 $Y$, 소스 도메인 $D_S$와 타겟 도메인 $D_T$에 대한 표기법을 설정하고, **신뢰할 수 있는 전이 학습**(**Trustworthy Transfer Learning**)을 타겟 도메인의 일반화 성능과 신뢰성을 향상시키는 것으로 정의합니다.
*   **지식 전이 가능성**(**Knowledge Transferability**) 분석:
    *   **IID 전이 가능성**(**IID Transferability**):
        *   **분포 불일치**(**Distribution Discrepancy**): 소스 및 타겟 도메인 간의 분포 차이($d_{IPM}(D_S, D_T) = \sup_{h \in \mathcal{H}} |\int_M h dP_S - \int_M h dP_T|$)를 정량화하는 다양한 지표 (예: `Total Variation Distance`, `HΔH-divergence`, `Discrepancy Distance`, `Y-discrepancy`, `Margin Disparity Discrepancy (MDD)`, `f-divergence`, `Generalized Discrepancy`, `Rényi Divergence`, `Wasserstein Distance`, `Maximum Mean Discrepancy (MMD)`, `Cauchy-Schwarz Divergence`)를 검토합니다.
        *   **태스크 다양성**(**Task Diversity**): 공유 표현 학습의 이점을 설명하기 위해 태스크 다양성(**Definition 3**)을 사용하여 전이 학습(**transfer learning**)의 초과 위험 경계(**excess risk bounds**)를 분석합니다.
        *   **전이 가능성 추정**(**Transferability Estimation**): 사전 학습된 소스 모델의 전이 효율성(`Trf(S→T) = E_{(x,y) \sim P_T} [acc(x,y;f_T)]`)을 측정하는 지표 (예: `NCE`, `LEEP`, `H-score`, `LogME`, `PACTran`, `TransRate`)를 탐색하고, 소스 모델 선택 및 앙상블 선택 방법을 논의합니다.
    *   **비-IID 전이 가능성**(**Non-IID Transferability**):
        *   **그래프 데이터 전이 가능성**(**Transferability on Graph Data**): 그래프 신경망(**GNNs**)의 스펙트럼(**spectral**) 및 공간 기반(**spatial-based**) 전이 가능성을 다루며, 분포 변화를 측정하고 **불변 노드 표현**(**invariant node representation**), **구조 재가중**(**structure reweighting**), **그래프 가우시안 프로세스**(**graph Gaussian process**)를 통해 전이 가능성을 향상시키는 방법을 제시합니다.
        *   **텍스트 데이터 전이 가능성**(**Transferability on Textual Data**): 대규모 언어 모델(**LLMs**)의 매개변수 효율적(**parameter-efficient**) 미세 조정(`PEFT`) 전략 (예: `Adapters`, `Low-rank Decomposition (LoRA)`, `Selective Masking (BitFit)`) 및 프롬프트 튜닝(**prompt tuning**) (예: **소프트 프롬프트 튜닝**(**soft prompt tuning**), **하드 프롬프트 튜닝**(**hard prompt tuning**), **전이 가능한 프롬프트 튜닝**(**transferable prompt tuning**))을 검토합니다.
        *   **시계열 데이터 전이 가능성**(**Transferability on Time Series Data**): 시계열 예측(**forecasting**) 및 분류(**classification**)를 위한 **시간적 분포 변화**(**temporal distribution shifts**)를 처리하는 방법 (예: `RevIN`, `AdaRNN`, `DAF`)을 분석합니다.
*   **지식 신뢰성**(**Knowledge Trustworthiness**) 분석:
    *   **프라이버시**(**Privacy**): **가설 전이**(**Hypothesis Transfer**) (예: **가설 전이 학습**(**hypothesis transfer learning**), **소스 없는 적응**(**source-free adaptation**), **테스트 시점 적응**(**test-time adaptation**)) 및 **연합 전이**(**Federated Transfer**) (예: **개인화된 연합 학습**(**personalized federated learning**), **연합 도메인 적응**(**federated domain adaptation**))와 같은 프라이버시 보존 전이 학습 프레임워크를 다룹니다.
    *   **적대적 강건성**(**Adversarial Robustness**): 전이 학습 모델에 대한 **우회 공격**(**evasion attacks**) 및 **오염 공격**(**poisoning attacks**) (예: `I2Attack`, `AdaptAttack`, **백도어 공격**(**backdoor attacks**))을 설명하고, **강건한 미세 조정**(**robust fine-tuning**) 및 **백도어 공격**(**backdoor attacks**) 방어 전략을 검토합니다. 또한 **전이 가능성**(**transferability**)과 **강건성**(**robustness**) 간의 상충 관계를 탐색합니다.
    *   **공정성**(**Fairness**): **그룹 공정성**(**Group Fairness**) (예: **인구 통계학적 동등성**(**Demographic Parity**), **기회 균등**(**Equality of Opportunity**), **균등화된 배당**(**Equalized Odds**)) 및 **개인 공정성**(**Individual Fairness**)의 전이 가능성을 논의하며, 타겟 도메인에서 민감 속성 정보의 가용성 여부에 따른 시나리오를 고려합니다.
    *   **투명성**(**Transparency**): **해석 가능성/설명 가능성**(**Interpretability/Explanability**) (예: `Grad-CAM`, `GSCLIP`) 및 **불확실성 정량화**(**Uncertainty Quantification**) (예: **온도 스케일링**(**Temperature Scaling**), **적합 예측**(**Conformal Prediction**), **베이즈 학습**(**Bayesian Learning**))을 통해 전이 학습 모델의 투명성을 향상시키는 방법을 제시합니다.
    *   **기타 신뢰성 문제**(**Other Trustworthiness Concerns**): **책임성 및 감사 가능성**(**Accountability and Auditability**)과 **지속 가능성 및 환경 복지**(**Sustainability and Environmental Well-being**) 문제를 간략하게 다룹니다.
*   **응용 분야 및 미래 동향**: 다양한 실제 응용 분야에서의 활용 사례를 제시하고, **부정적 전이 벤치마킹**(**benchmarking negative transfer**), **교차 모달 전이 가능성**(**cross-modal transferability**), **물리학 정보 기반 전이 학습**(**physics-informed transfer learning**), **전이 가능성**(**transferability**)과 **신뢰성**(**trustworthiness**) 간의 **상충 관계**(**trade-off**)와 같은 미해결 과제 및 미래 연구 방향을 제시합니다.

## Results

이 설문조사는 **신뢰할 수 있는 전이 학습**(**Trustworthy Transfer Learning**)에 대한 포괄적인 분석을 통해 다음을 밝힙니다:

*   **지식 전이 가능성**(**knowledge transferability**)은 **IID** 환경에서 **분포 불일치**(**distribution discrepancy**), **태스크 다양성**(**task diversity**), **전이 가능성 추정**(**transferability estimation**)을 통해, 그리고 **비-IID**(**non-IID**) 환경에서 그래프, 텍스트, 시계열 데이터의 특성을 고려하여 정량화하고 향상시킬 수 있는 다양한 이론적 및 알고리즘적 접근 방식이 존재합니다.
*   **지식 신뢰성**(**knowledge trustworthiness**)은 전이 학습(**transfer learning**) 과정에서 **프라이버시**(**privacy**), **적대적 강건성**(**adversarial robustness**), **공정성**(**fairness**), **투명성**(**transparency**)과 같은 핵심 속성을 보장하는 데 중요합니다. 사전 학습된 모델 활용(`Hypothesis Transfer`)이나 연합 학습(`Federated Transfer`)을 통해 데이터 프라이버시를 보호하고, **공격**(**attacks**)과 **방어**(**defenses**) 메커니즘을 통해 모델의 강건성을 확보하며, **그룹 공정성**(**group fairness**)과 **개인 공정성**(**individual fairness**)을 달성하고, **해석 가능성**(**interpretability**)과 **불확실성 정량화**(**uncertainty quantification**)를 통해 모델의 의사결정 과정을 이해할 수 있음을 보여줍니다.
*   신뢰할 수 있는 전이 학습(**Trustworthy Transfer Learning**)은 농업, 생명정보학, 헬스케어, 교육, 로봇 공학, 전자상거래 등 광범위한 실제 응용 분야에서 이미 상당한 진전을 이루었음을 입증합니다.
*   이 분야는 **부정적 전이**(**negative transfer**) 벤치마킹, **교차 모달 전이 가능성**(**cross-modal transferability**)에 대한 이론적 이해, **물리학 정보 기반 전이 학습**(**physics-informed transfer learning**)의 발전, **전이 가능성**(**transferability**)과 **신뢰성**(**trustworthiness**) 간의 **상충 관계**(**trade-off**)에 대한 심층적인 탐구를 포함하여 여전히 많은 미해결 과제를 안고 있습니다. 이러한 과제 해결은 신뢰할 수 있는 AI 시스템의 개발에 필수적입니다.