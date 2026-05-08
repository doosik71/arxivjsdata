# Explainable Deep Learning in Healthcare: A Methodological Survey from an Attribution View

Di Jin, Elena Sergeeva, Wei-Hung Weng, Geeticka Chauhan, and Peter Szolovits

## 🧩 Problem to Solve

딥러닝(DL) 모델은 방대한 전자의무기록(EHR) 데이터와 기술 발전 덕분에 의료 진단, 예후 예측, 치료 등 임상 의사결정 지원 시스템에서 탁월한 성능을 보이고 있습니다. 그러나 DL의 **"블랙박스" 특성**은 모델의 의사결정 과정을 이해하기 어렵게 만들어, 실제 의료 환경에서의 폭넓은 채택을 가로막는 주요 장애물이 되고 있습니다. 따라서 최종 사용자가 모델의 예측 및 권고사항을 수용할지 여부를 평가할 수 있도록 **해석 가능한 딥러닝(Interpretable Deep Learning, IDL)**에 대한 필요성이 대두되고 있습니다. 기존의 관련 연구들은 개념적 논의에 집중한 반면, 의료 분야에 특화된 심층적인 방법론적 가이드라인은 부족합니다.

## ✨ Key Contributions

- **포괄적인 방법론 소개**: 의료 분야 딥러닝 모델의 해석 가능성을 높이는 다양한 귀인(attribution) 기반 방법론들을 심층적으로 분류하고 소개합니다.
- **장단점 및 시나리오 논의**: 각 해석 방법론의 장점과 단점을 분석하고, 각각이 어떤 시나리오에 적합한지 논의하여 연구자 및 임상 실무자가 최적의 방법을 선택할 수 있도록 돕습니다.
- **의료 분야 적용 사례 분석**: 일반 도메인에서 개발된 방법들이 의료 문제에 어떻게 적용되고 맞춤화되었는지, 그리고 이러한 기술이 의사들의 이해를 돕는 방안을 탐구합니다.
- **핵심 평가 기준 제시**: 해석 가능성 방법론의 모델 의존성, 설명 범위, 신뢰성(faithfulness), 그리고 전문가 사용자 관점의 타당성(plausibility) 등 중요한 측면들을 논의하며 평가 기준을 제시합니다.
- **벤치마킹 및 미래 방향 제시**: 해석 방법론에 대한 현재의 벤치마킹 노력과 향후 연구 방향을 제시하여, 고성능이면서도 설명 가능한 의료 AI 모델 구축에 기여합니다.

## 📎 Related Works

이 서베이 논문은 의료 분야 설명 가능한 기계 학습(Explainable Machine Learning, XAI)에 대한 기존의 여러 서베이 논문들과 차별점을 가집니다. 이전 연구들([Ahmad et al., 2018], [Holzinger et al., 2019], [Wiens et al., 2019], [Tonekaboni et al., 2019a], [Vellido, 2019], [Payrovnaziri et al., 2020])은 주로 해석 가능성의 정의, 개념, 중요성 및 응용에 대한 고수준의 개요를 다루었습니다. 반면, 본 논문은 방법론에 대한 심층적인 설명과 의료 분야에서의 실제 적용 사례, 그리고 각 방법의 장단점 및 적합성을 다루어 실질적인 가이드를 제공합니다. 특히 LIME [Ribeiro et al., 2016a], RETAIN [Choi et al., 2016], SHAP [Lundberg and Lee, 2017]과 같이 성능을 유지하면서 좋은 설명을 제공하는 최신 방법들을 심층적으로 분석합니다.

## 🛠️ Methodology

본 논문은 의료 분야 딥러닝 모델의 해석 가능성을 위한 방법론을 **귀인(attribution)** 관점에서 체계적으로 분류하고 분석합니다.

**1. 논문 선정:**

- MEDLINE, IEEE Xplore, ACM, ACL Anthology 데이터베이스 및 주요 임상/AI 학술지 및 학회(Nature, JAMA, NeurIPS, ICML 등)를 대상으로 체계적인 검색을 수행했습니다.
- 키워드는 `(explainable OR explainability OR interpretable OR interpretability OR understandable OR understandability OR comprehensible OR comprehensibility) AND (machine learning OR artificial intelligence OR deep learning OR AI OR neural network)`를 사용했습니다.
- 검색된 논문들을 수동으로 필터링하여 다음 세 가지 유형의 연구를 선정했습니다: 일반 도메인 해석 방법, 의료 특화 해석 방법, 해석 가능성을 포함하는 의료 응용 분야. DL 모델을 해석할 수 있는 방법만 다루었습니다.

**2. 해석 방법론 분류 (귀인 관점):**
입력 특징 $x = [x_1, \dots, x_N]$이 주어졌을 때, 딥러닝 모델 $S(x) = [S_1(x), \dots, S_C(x)]$의 특정 출력 $S_c$에 대한 각 입력 특징 $x_i$의 기여도 $R_c = [R_c_1, \dots, R_c_N]$를 결정하는 것을 목표로 합니다. 이러한 방법들을 다음 범주로 나눕니다:

- **역전파 기반 (Back-propagation based)**:

  - 모델의 출력 $S_c$에 대한 입력 특징 $x_i$의 편도함수 $|\frac{\partial S_c(x)}{\partial x_i}|$의 절대값을 취하는 Saliency Map [Simonyan et al., 2014]으로 시작합니다.
  - ReLU 비선형성을 처리하는 방식에 따라 Deconvolution [Zeiler and Fergus, 2014a] 및 Guided Back-propagation [Springenberg et al., 2015]이 있습니다.
  - **Gradient \* Input** [Shrikumar et al., 2017b]은 기울기와 입력을 곱합니다.
  - **Integrated Gradients** [Sundararajan et al., 2017]는 베이스라인부터 입력까지 선형 경로를 따라 기울기를 평균화합니다.
  - **CAM (Class Activation Mapping)** [Zhou et al., 2016] 및 이를 일반화한 **Grad-CAM** [Selvaraju et al., 2017]은 컨볼루션 레이어의 기울기 정보를 활용하여 클래스별 특징 맵을 생성합니다.
  - **LRP (Layer-wise Relevance Propagation)** [Bach et al., 2015] 및 **DeepLIFT** [Shrikumar et al., 2017a]는 활성화 값의 역전파를 통해 관련성 점수를 할당합니다.

- **특징 섭동 기반 (Feature Perturbation based)**:

  - 입력의 일부를 가리거나 변경했을 때 모델의 예측 신뢰도 변화를 명시적으로 측정합니다.
  - 모델에 구애받지 않는 Prediction Difference Analysis [Zintgraf et al., 2017] 및 Representation Erasure [Li et al., 2016] 등이 있습니다.
  - **적대적 섭동 (Adversarial Perturbation)** 기법들은 입력에 노이즈를 추가하여 모델의 오분류를 유도하며, 이는 모델의 민감도를 파악하는 데 활용될 수 있습니다 (FGSM [Goodfellow et al., 2015], JSMA [Papernot et al., 2016a] 등).

- **어텐션 기반 (Attention based)**:

  - 어텐션 메커니즘의 가중치를 모델 의사결정에 대한 입력 특징의 기여도나 중요도로 해석합니다 [Xu et al., 2015].
  - **RETAIN (Reverse Time Attention Model)** [Choi et al., 2016]과 같은 모델이 EHR 기반 시계열 데이터에서 특정 시간 단계와 개별 특징의 중요도를 계산하는 데 사용됩니다.
  - Transformer [Vaswani et al., 2017]와 같은 완전 어텐션 기반 아키텍처도 NLP에서 인기가 많습니다.

- **모델 증류 기반 (Model Distillation based)**:

  - 복잡한 "선생님(teacher)" 모델의 동작을 설명 가능한 "학생(student)" 모델이 모방하도록 학습시키는 모델 압축 기법입니다.
  - **LIME (Local Interpretable Model-agnostic Explanations)** [Ribeiro et al., 2016a]은 특정 입력 사례에 대한 국소적 설명을 생성하는 가장 영향력 있는 방법 중 하나입니다.
  - Anchors [Ribeiro et al., 2018]는 고정된 설명을 생성하여 모델 예측이 특정 특징에 의해 강하게 결정되는 부분을 강조합니다.

- **게임 이론 기반 (Game Theory based)**:

  - 협력 게임의 Shapley 값 [Shapley, 1953] 개념을 사용하여 각 특징이 모델 예측에 기여하는 정도를 공정하게 분배합니다.
  - **SHAP (SHapley Additive exPlanations)** [Lundberg and Lee, 2017]은 Shapley 값을 근사하는 통합 프레임워크로, 선형 모델, 트리 모델, 심층 신경망 등 다양한 ML 모델에 적용됩니다.
  - DASP (Deep Approximate Shapley Propagation) [Ancona et al., 2019]는 비선형 모델에서 Shapley 값을 더 잘 근사하는 다항 시간 알고리즘입니다.

- **예시 기반 (Example based)**:

  - 모델 예측에 대표적이거나 영향력 있는 특정 훈련 데이터 포인트를 사용하여 모델 동작을 해석합니다.
  - Influence Function [Koh and Liang, 2017]은 특정 훈련 인스턴스의 제거 또는 섭동이 손실 함수에 미치는 영향을 측정합니다.
  - L2X (Learning to Explain) [Chen et al., 2018]는 각 인스턴스에 대한 특징 중요도를 국소적으로 측정합니다.
  - **Contextual Decomposition (CD)** [Murdoch et al., 2018]는 LSTM의 출력을 단어 또는 변수 조합의 기여도로 분해합니다.
  - MMD-critic [Kim et al., 2016]은 프로토타입 예시와 비판 샘플을 사용하여 설명을 제공합니다.

- **생성 기반 (Generative based)**:
  - 외부 지식 소스, 인과 모델 또는 설명 가능한 확률 모델에서 파생된 정보를 사용하여 모델의 동작에 대한 설명을 생성합니다.
  - CAGE (Commonsense Auto-Generated Explanations) [Rajani et al., 2019]는 언어 모델을 사용하여 상식 추론 설명을 생성합니다.
  - GEF (Generative Explanation Framework) [Liu et al., 2019]는 추상적이고 미세한 설명을 생성하면서 분류 작업을 수행합니다.
  - Action Influence Models [Madumal et al., 2020]는 인과 모델을 활용하여 강화 학습 에이전트의 동작 설명을 생성합니다.

## 📊 Results

각 해석 방법론은 의료 분야의 다양한 문제에 적용되어 다음과 같은 결과를 도출했습니다:

- **역전파 기반**:

  - **의료 영상**: 피부암 조직(CAM), COVID-19 흉부 CT 이미지(Grad-CAM), 유방 MRI(Integrated Gradients) 등에서 질병 영역을 강조하는 히트맵을 생성하여 진단에 중요한 형태학적 특징을 시각화하고 모델이 어떤 영역에 집중하는지 보여주었습니다. 뇌 MRI, 망막 영상, 유방 영상, 피부 영상, CT 스캔, 흉부 X-ray 등 다양한 의료 영상에서 활용되었습니다.
  - **특징 기반 예측 모델**: 고정된 특징(예: 치료 권고) 또는 시계열 데이터(예: 질병 진행)에서 DeepLIFT, LRP 등을 통해 어떤 특징이 예측에 더 중요한지, 어떤 시간적 패턴이 최종 모델 결정에 영향을 미치는지 분석하는 데 사용되었습니다.

- **특징 섭동 기반**:

  - 주로 **적대적 공격(adversarial attacks)** 연구에서 딥러닝 모델의 취약성을 밝히고, 이를 통해 더욱 견고하고 해석 가능한 모델을 구축하는 데 기여했습니다.
  - 당뇨병성 망막증, 기흉, 흑색종 진단 모델에 대한 섭동 공격이 수행되어 의료 딥러닝 시스템의 잠재적 위험을 보여주었습니다 [Finlayson et al., 2019b].
  - 스마트 헬스케어 시스템(SHS)에서 환자 상태를 조작하기 위해 장치 판독값을 섭동하는 연구 [Iqtidar Newaz et al., 2020]와 심전도(ECG) [Chen et al., 2020] 및 EHR 데이터 [Sun et al., 2018]에 대한 적대적 예시 생성 연구가 진행되었습니다.

- **어텐션 기반**:

  - 심부전, 패혈증, 중환자실(ICU) 사망률 예측, 자동 진단 등 EHR 기반 종단 예측 작업에서 **RETAIN** [Choi et al., 2016]과 같은 모델이 특정 시점의 데이터나 개별 특징이 예측에 기여하는 정도를 해석하는 데 널리 사용되었습니다.
  - NLP 기반 모델(Transformer 등)은 대규모 비정형 의료 텍스트를 사전 학습하는 데 혁신적인 역할을 했으며 [Lee et al., 2020], 이러한 어텐션 메커니즘도 해석에 활용됩니다.

- **모델 증류 기반**:

  - **LIME** [Ribeiro et al., 2016a]은 의료 AI에서 가장 인기 있는 사례 수준의 블랙박스 모델 설명 기법 중 하나입니다.
  - 심부전 예측 [Khedkar et al., 2020], 암 유형 및 중증도 추론 [Moreira et al., 2020], 유방암 생존 예측 [Hendriks et al., 2020], 고혈압 발병 예측 [Elshawi et al., 2019] 등 다양한 EHR 기반 예측 작업에 적용되어 개별 사례에 대한 직관적인 설명을 제공했습니다.

- **게임 이론 기반**:

  - **SHAP** [Lundberg and Lee, 2017]은 개별 예측뿐만 아니라 Shapley 값의 집계를 통해 전역 모델 동작까지 설명할 수 있어 의료 분야에서 특징 기여도 분석에 널리 적용되었습니다.
  - 당뇨병성 망막증 진행 진단에서 주요 영역 식별 [Arcadu et al., 2019], 흑색종 및 파킨슨병 예측을 위한 심층 신경망의 saliency map 생성 [Young et al., 2019], [Pianpanit et al., 2019]에 사용되었습니다.
  - 뇌전도(EEG) 신호를 이용한 뇌진탕 식별에서 특징 영향 조사 [Boshra et al., 2019]에도 활용되었습니다.
  - **DASP** [Ancona et al., 2019]는 파킨슨병 등 비선형 모델의 Shapley 값 근사에 적용되었습니다.
  - 마취과 의사들의 임상적 검증을 통해 SHAP 설명의 임상적 의미를 확인하는 연구도 진행되었습니다 [Lundberg et al., 2018b].

- **예시 기반**:

  - **CDEP (Contextual Decomposition Explanation Penalization)** [Rieger et al., 2020]는 피부암 진단에서 학습 모델이 허위 상관관계(예: 이미지의 패치)를 사용하지 않도록 하여 모델의 신뢰성을 높이는 데 활용되었습니다.
  - GnnExplainer [Ying et al., 2019]는 그래프 신경망의 예측 관련 엣지와 중요한 특징 하위 집합을 강조하여 그래프 기반 작업에 대한 해석 가능성을 제공했습니다.
  - MMD-critic [Kim et al., 2016]은 프로토타입 예시와 모델이 잘 맞지 않는 '비판 샘플'을 함께 활용하여 인간의 추론과 모델 이해를 돕는 설명을 제공했습니다.

- **생성 기반**:
  - 유방암 진단을 위한 **BIRADS (Breast Imaging Reporting And Data System)** 가이드 진단 네트워크 [Kim et al., 2018]는 인간이 이해할 수 있는 BIRADS 용어와 관련된 주요 영역에 모델이 집중하도록 시각적으로 해석 가능한 가이드 맵을 생성했습니다.
  - 폐결절 악성도 예측을 위한 **HSCNN (Hierarchical Semantic Convolutional Neural Network)** [Shen et al., 2019]은 구형도, 경계, 석회화 등 방사선 전문의가 일반적으로 사용하는 저수준의 전문가 해석 가능 진단 의미 특징들을 생성하여 고수준 분류 모델의 입력으로 활용했습니다. 이러한 방법들은 의료 도메인 지식 통합의 중요성을 보여주었습니다.

## 🧠 Insights & Discussion

**1. 해석 방법론의 차원:**

- **모델 의존성 (Model Dependence)**: 설명 모델이 해석하려는 모델의 내부 구조에 의존하는가 아니면 어떤 "블랙박스" 모델에도 사용될 수 있는가?
  - `의존적 (Dependent)`: 역전파 기반 (예: Grad-CAM), 어텐션 기반 (예: RETAIN)
  - `독립적 (Independent)`: 특징 섭동 기반 (예: LIME, SHAP), 모델 증류 기반, 게임 이론 기반
- **설명 범위 (Explanation Scope)**: 설명 모델이 특정 입력-예측 쌍에 대한 `국소적 (Local)` 설명을 생성하는가, 아니면 모델 동작에 대한 통합된 `전역적 (Global)` 설명을 생성하려는가?
  - 대부분의 방법은 국소적이며, 일부는 국소적 설명을 패턴으로 집계하려는 시도를 합니다.

**2. 해석 가능성 방법론의 신뢰성 및 타당성:**

- **해석의 충실도 (Faithfulness)**: 생성된 해석이 기반이 되는 의사결정 모델의 동작을 얼마나 정확하게 나타내는가.
  - 해석 모델과 원본 모델 간의 예측 일치도를 확인해야 합니다.
  - 최근 제안된 충실도 측정 지표들을 계산하는 것을 고려해야 합니다 [Yeh et al., 2019].
  - "특징 가림(feature occlusion)" 건전성 검사를 통해 설명에 따라 모델 요소들을 변경했을 때 원본 예측이 변화하는지 확인해야 합니다 [Hooker et al., 2018].
  - 일부 해석 방법은 동일한 입력에 대해 여러 번 실행 시 다른 설명을 생성할 수 있습니다.
- **전문 사용자 관점의 타당성 (Plausibility as defined by the expert user)**: 해석이 임상 지식 및 실제 임상 실습과 일치하며, 인간 전문가(의사)가 이해하기 쉬운가.
  - 임상 의사들은 종종 모델 성능보다 임상적으로 관련성 있는 특징과 설명 가능성을 더 중요하게 여깁니다 [Caruana et al., 2015].
  - 모델 개발자들은 시각화에 과도하게 의존하여 방법론을 오용할 수 있으며, 시스템의 전체 동작을 완전히 이해하지 못하고 잘못 해석할 수 있습니다 [Kaur et al., 2020].
  - 서로 다른 해석 방법론은 다양한 관점에서 통찰력을 제공하지만, 임상적으로 관련성 있는 중요한 특징들의 다른 부분 집합을 산출할 수 있습니다 [Elshawi et al., 2019].
  - 국소적 해석 방법(LIME)은 유사한 패턴의 환자들에게도 매우 다른 해석을 제공할 수 있는 불안정성 문제가 있습니다 [Elshawi et al., 2019]. 반면 SHAP은 계산 비용이 높고 학습 데이터 접근이 필요할 수 있습니다 [Lundberg and Lee, 2017].
  - 해석 가능성은 주관적이며, 컴퓨터 기반 기술은 인간 전문가 간의 의사소통에 필요한 상호작용이 부족합니다 [Lahav et al., 2018].

**3. 해석 방법 벤치마킹:**

- 다양한 해석 방법들 중 어떤 것을 선택해야 하는지에 대한 명확한 답은 아직 없습니다. 특정 모델 유형에 대한 상세하고 포괄적인 가이드라인이 부족합니다.
- 최근 여러 연구들이 CNN, RNN, Transformer 등 신경망 모델에 적용된 인기 있는 해석 방법들을 벤치마킹하기 시작했습니다 [Arras et al., 2019], [Ismail et al., 2020].
- 벤치마킹은 정확도 손실, 정밀도, 재현율 등 정량적 지표와 인간 평가를 통해 이루어집니다.
- 다양한 해석 방법들의 장단점을 활용하기 위해 두 가지 종류의 해석 방법을 결합하여 서로 보완하도록 하는 연구들이 제안되고 있습니다 [Ismail et al., 2020], [Bhatt et al., 2020a].
- 궁극적으로는 인간-컴퓨터 상호작용(HCI) 관점에서 사용자 중심의 설명 가능성 도구 설계와 임상 실무자의 참여를 통해 사용자(임상의)의 기술과 실제 요구 사항을 이해하고 모델 출력을 활용하는 방안을 모색해야 합니다 [Ahmad et al., 2018].

## 📌 TL;DR

의료 분야에서 딥러닝 모델의 **"블랙박스" 문제**는 임상적 채택을 저해합니다. 이 문제를 해결하고 모델의 의사결정 과정을 투명하게 이해하기 위해 **설명 가능한 딥러닝(XDL)**이 필수적입니다. 본 서베이는 역전파, 특징 섭동, 어텐션, 모델 증류, 게임 이론, 예시 기반, 생성 기반 등 **7가지 주요 XDL 방법론**을 심층적으로 분석하고, 각 방법론의 **의료 분야 적용 사례, 장단점, 적합성**을 포괄적으로 다룹니다. 또한, XDL 방법론의 **모델 의존성, 설명 범위, 충실도, 전문가 관점의 타당성** 등 핵심 평가 측면을 논의하며, 현재의 한계와 **벤치마킹, 방법론 결합, 인간-컴퓨터 상호작용(HCI) 통합** 등의 향후 연구 방향을 제시하여 의료 연구자 및 임상 실무자가 고성능이면서도 신뢰할 수 있는 XDL 모델을 효과적으로 구축하고 활용하는 데 실질적인 가이드를 제공합니다.
