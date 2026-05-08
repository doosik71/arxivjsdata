# PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models

Jenny Schmalfuss, Nadine Chang, Vibashan VS, Maying Shen, Andrés Bruhn, Jose M. Alvarez (2025)

## 🧩 Problem to Solve

본 논문은 Vision Language Models(VLMs)가 사용자 프롬프트의 미세한 변화에 얼마나 민감하게 반응하는지, 즉 Prompt Sensitivity 문제를 정량적으로 분석하고자 한다. VLMs는 자율 주행이나 질병 진단과 같은 안전 중심(safety-critical) 애플리케이션에 적용되고 있으나, 입력 프롬프트의 구성 방식에 따라 결과가 일관되지 않게 출력되는 불안정성 문제가 존재한다.

기존의 Large Language Models(LLMs)에서는 프롬프트 민감성이 활발히 연구되었으나, VLMs 분야에서는 상대적으로 연구가 부족했다. 특히 기존의 VLM 관련 연구들은 사람이 실제로 사용할 법하지 않은 노이즈 섞인 프롬프트나 손상된 이미지에 집중하는 경향이 있었다. 따라서 본 연구의 목표는 실제 사용 시나리오에서 발생 가능한 현실적인 프롬프트 변형을 정의하고, 이를 통해 VLM의 신뢰성을 측정할 수 있는 정량적 프레임워크를 구축하여 어떤 모델이 프롬프트 변화에 가장 둔감(agnostic)한지 밝히는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 VLM의 프롬프트 민감도를 분석하기 위한 정량적 프레임워크인 **PARC (Prompt Analysis via Reliability and Calibration)**를 제안한 것이다. PARC의 중심 아이디어는 다음의 세 가지 기둥으로 구성된다.

1. **현실적인 프롬프트 변형(Plausible Prompt Variations) 정의**: 언어(Language)와 시각(Vision) 도메인 모두에서 의미를 유지하는 '재구성(Reformulation)'과 정답을 변경하는 '의미적 변화(Semantic Change)'를 포함하는 11가지 변형 세트를 구축하였다.
2. **새로운 신뢰도 점수(Reliability Score) 설계**: 모델의 정확도(Accuracy)와 확신도(Certainty)를 결합하여, 확신을 가지고 틀린 답을 내놓는 모델을 효과적으로 식별할 수 있는 단일 지표를 제안하였다.
3. **점수 보정(Score Calibration) 메커니즘**: 데이터셋마다 다른 무작위 성능(random performance)의 영향을 제거하여, 서로 다른 데이터셋과 프롬프트 변형 간에도 성능을 직접적으로 비교할 수 있도록 하는 보정 단계를 도입하였다.

## 📎 Related Works

기존의 VLM 프롬프트 민감도 연구들은 주로 이미지에 노이즈를 추가하거나 텍스트 프롬프트의 글자를 무작위로 섞는 등, 실제 인간 사용자가 생성하지 않을 법한 비현실적인 변형에 집중하였다. 반면, LLM 연구에서는 문장 재구성이나 부정문 사용과 같은 의미적 변화를 다루었으나, 이를 VLM의 시각적 요소와 결합하여 분석한 시도는 부족했다.

또한, 기존의 신뢰도 측정 방식들은 정확도, 확신도, 일관성 등을 개별적으로 측정하여 모델 선택 시 통합적인 해석이 어려웠으며, 특히 정답의 개수가 다른 데이터셋 간의 성능 비교 시 무작위 추측 확률(expected random performance)의 차이를 고려하지 않는 한계가 있었다. PARC는 이러한 한계를 극복하기 위해 통합된 신뢰도 점수와 무작위 성능 기반의 보정법을 제시함으로써 차별성을 가진다.

## 🛠️ Methodology

### 1. 프롬프트 변형 생성 (Prompt Variations)

PARC는 프롬프트를 언어 성분과 시각 성분으로 나누고, 각각에 대해 두 가지 유형의 변형을 적용한다.

* **Prompt Reformulations (정답 유지)**:
  * **Language (LR)**: 질문을 지시문 형태로 변경(LR-I), 간결하게 작성(LR-C), 또는 상세하게 작성(LR-V)한다.
  * **Vision (VR)**: 이미지에 블러 처리(VR-B), 조명 변경(VR-L), 또는 90도 회전(VR-R)을 적용한다.
* **Semantic Prompt Variations (정답 변경)**:
  * **Language (LS)**: 'not'을 추가한 부정문(LS-N), 반의어 사용(LS-A), 또는 'More'를 'Less'로 변경(LS-M)하여 정답이 바뀌도록 유도한다.
  * **Vision (VS)**: 두 이미지의 위치를 서로 바꿈(VS-S) 또는 정답이 되는 이미지를 다른 이미지로 교체(VS-E)하여 정답을 변경한다.

### 2. 신뢰도 점수 (Reliability Score)

단순 정확도뿐만 아니라 모델이 자신의 답변에 얼마나 확신하는지를 결합하여 신뢰도를 계산한다.

* **Accuracy ($acc$)**: 모델의 답변이 정답 집합 $A(p)$에 포함되는지 측정하는 지표 함수이다.
* **Certainty ($cert$)**: Conformal Prediction을 사용하여 모델이 예측한 가능성 높은 답변 집합 $C(p)$의 크기를 기반으로 계산한다.
    $$cert(p) = 1 - \frac{|C(p)| - 1}{|P(p)| - 1}$$
    여기서 $|P(p)|$는 가능한 모든 선택지의 수이다.
* **Reliability ($rel$)**: 정확도와 확신도를 결합하여 단일 수치로 요약한다.
    $$rel = (2 \cdot acc - 1) \cdot cert$$
    이 점수는 확신을 가지고 정답을 맞힌 경우 $1$, 확신을 가지고 틀린 경우 $-1$, 불확실하거나 정확도가 $0.5$인 경우 $0$이 된다.

### 3. 점수 보정 (Score Calibration)

데이터셋의 난이도(무작위 정답 확률)를 맞추기 위해 보정 단계를 거친다.

* **일반 보정 (Accuracy, Consistency)**: 무작위 성능 $s_{rand}$를 기준으로 개선 정도를 측정한다.
    $$s_{calib} = \begin{cases} \frac{s - s_{rand}}{1 - s_{rand}} & \text{for } s \geq s_{rand} \\ \frac{s - s_{rand}}{s_{rand}} & \text{for } s < s_{rand} \end{cases}$$
* **신뢰도 보정**: 신뢰도 점수의 기준점 $0$을 측정된 무작위 정확도 $acc_{rand}$로 이동시킨다. 이를 위해 $acc$ 대신 $acc^m$ (여기서 $m = \frac{\log(2)}{\log(1/acc_{rand})}$)을 사용하여 계산한다.

## 📊 Results

### 실험 설정

* **대상 모델**: LLaVA-1.5, LLaVA-1.6, Qwen-VL, CogVLM, InternVL2, Cambrian 등 7개 가족의 22개 VLM 모델.
* **데이터셋**: MMBench 및 비교 기반 데이터셋(MIT-States, MIT-Attributes, VAW-States, VAW-Attributes, NYU-Depth V2, Fashionpedia) 등 7개 데이터셋.
* **측정 지표**: 보정된 정확도, 확신도, 일관성, 신뢰도.

### 주요 결과

1. **프롬프트 민감도 확인**: 모든 VLM은 언어 및 시각 변형에 민감하게 반응했다. 특히 의미적 변화(Semantic Changes), 그 중에서도 부정문(LS-N)이나 반의어(LS-A) 사용 시 성능이 급격히 저하되었으며, 이는 LLM의 민감도 양상과 매우 유사하다.
2. **모달리티 간 대칭성**: VLM은 언어적 재구성보다 시각적 재구성에 더 강한 모습을 보였으나, 전체적인 경향성은 두 모달리티 모두에서 동일하게 나타났다. 즉, 언어 프롬프트의 민감성이 시각 도메인에서도 그대로 mirror(반영)됨을 확인하였다.
3. **가장 강건한 모델**: **InternVL2** 가족이 프롬프트 변형에 대해 가장 둔감하고 신뢰도가 높았으며, 특히 **InternVL2-40B** 모델이 가장 뛰어난 성능을 보였다.
4. **민감도의 원인 분석**:
    * **모델 크기**: 동일 가족 내에서는 모델 크기가 클수록 민감도가 낮아졌으나, 모델 크기보다 **모델 가족(Family)** 자체가 신뢰도에 더 큰 영향을 미쳤다.
    * **데이터 품질**: 단순히 학습 데이터의 양보다 **고품질로 큐레이션된 데이터(Curated Data)**를 사용한 모델(예: InternVL2, Cambrian)이 훨씬 더 강건한 모습을 보였다.

## 🧠 Insights & Discussion

본 연구는 VLM이 단순히 시각 정보를 처리하는 것을 넘어, LLM으로부터 상속받은 프롬프트 민감성 문제를 그대로 가지고 있음을 정량적으로 증명하였다. 특히 흥미로운 점은 시각적 변형과 언어적 변형이 서로 다른 물리적 특성을 가짐에도 불구하고, '정답 유지(Reformulation)'와 '정답 변경(Semantic Change)'이라는 논리적 층위에서는 동일한 민감도 패턴을 보인다는 것이다. 이는 VLM의 추론 능력이 입력 모달리티에 상관없이 일관된 취약점을 가지고 있음을 시사한다.

또한, 모델 크기의 확장(Scaling)만으로는 프롬프트 민감도 문제를 완전히 해결할 수 없으며, 정교하게 설계된 학습 데이터 셋의 구성이 모델의 강건성(Robustness)을 높이는 핵심 경로임을 밝혀냈다.

**한계점**:

* 본 프레임워크는 Certainty 계산을 위해 Logit 값에 접근할 수 있는 **White-box 모델**만을 대상으로 한다.
* 정답을 명확히 정의할 수 있는 **Multiple-Choice VQA(MC-VQA)** 작업에 집중되어 있어, 자유 형식의 생성 작업(Generative tasks)으로의 확장은 추가적인 LLM 평가 모델이 필요하여 노이즈가 발생할 수 있다.

## 📌 TL;DR

본 논문은 VLM의 프롬프트 민감도를 정량적으로 분석하기 위한 **PARC** 프레임워크를 제안한다. PARC는 현실적인 11가지 프롬프트 변형과 새로운 신뢰도 점수, 그리고 무작위 성능을 고려한 보정법을 통해 모델의 강건성을 측정한다. 실험 결과, VLMs는 LLM과 유사한 프롬프트 민감성을 보이며, 특히 정답을 바꾸는 의미적 변형에 취약하다는 것이 드러났다. 또한, **InternVL2** 모델이 가장 강건하며, 이는 단순한 모델 크기 증가보다 **고품질 데이터 큐레이션**이 프롬프트 민감도를 줄이는 데 더 효과적임을 시사한다.
