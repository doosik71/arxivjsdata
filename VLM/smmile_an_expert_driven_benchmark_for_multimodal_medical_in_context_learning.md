# SMMILE: AN EXPERT-DRIVEN BENCHMARK FOR MULTIMODAL MEDICAL IN-CONTEXT LEARNING

Melanie Rieff, Maya Varma, Ossian Rabow, et al. (2025)

## 🧩 Problem to Solve

본 논문은 의료 분야에서 Multimodal In-Context Learning (ICL) 능력을 체계적으로 평가하기 위한 벤치마크의 부재 문제를 해결하고자 한다. 실제 임상 환경에서 의사들은 소수의 유사 사례나 제한된 차별 진단 세트를 참고하여 전문적인 과업을 수행하는 경우가 많다. 이러한 임상 워크플로우는 ICL의 메커니즘과 매우 유사하다.

기존의 Multimodal Large Language Models (MLLMs)는 의료 시각적 질의응답(Medical VQA)에서 어느 정도 성과를 보였으나, 문맥(Context)을 통해 새로운 작업을 학습하는 ICL 능력에 대해서는 알려진 바가 거의 없다. 특히 기존의 few-shot 평가 방식은 예시를 무작위로 선택하는 경향이 있어, 전문가가 의도한 과업 데모스트레이션(Task Demonstration)으로서의 역할을 제대로 수행하지 못하며, 이로 인해 zero-shot 대비 성능 향상이 미미하게 나타나는 한계가 있었다. 따라서 본 연구의 목표는 전문가가 직접 설계한 고품질의 의료 multimodal ICL 벤치마크를 구축하고, 이를 통해 현재 MLLM들의 실제 능력을 정밀하게 측정하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 전문가 주도의 의료 multimodal ICL 벤치마크인 **SMMILE**과 그 확장 버전인 **SMMILE++**를 제안한 것이다. 중심적인 설계 아이디어는 단순한 데이터 수집이 아니라, 11명의 의료 전문가가 직접 쿼리와 그에 적합한 '과업 데모스트레이션'으로서의 In-context examples를 큐레이션하게 함으로써, 모델이 실제로 문맥을 통해 학습할 수 있는지를 엄격하게 테스트하는 것이다. 또한, 예시의 순서(Ordering)와 품질(Quality)이 모델 성능에 미치는 영향을 분석하여 현재 MLLM들이 가진 Recency Bias와 노이즈 취약성을 드러냈다.

## 📎 Related Works

기존의 ICL 연구는 주로 텍스트 기반의 LLM(예: GPT-3)에서 시작되었으며, 이후 Flamingo와 같은 모델을 통해 이미지-텍스트가 교차되는 multimodal 설정으로 확장되었다. 의료 분야에서도 Med-PaLM M, Med-Flamingo, LLaVA-Med 등 다양한 의료 전용 MLLM들이 제안되었다.

그러나 기존의 의료 VQA 벤치마크(VQA-RAD, PathVQA, SLAKE, MIMIC-CXR 등)에서 수행된 few-shot 평가들은 예시를 무작위로 선택하는 방식을 취했다. 이는 전문가가 특정 진단을 내리기 위해 참고하는 '전형적인 사례'를 제시하는 실제 임상 과정과 차이가 있다. SMMILE은 무작위 선택이 아닌, 전문가가 의도적으로 설계한 예시를 제공함으로써 기존 접근 방식과 차별화되며, 의료 도메인 특유의 전문성과 복잡성을 반영한 첫 번째 multimodal ICL 벤치마크라는 점에서 의의가 있다.

## 🛠️ Methodology

### 1. 데이터셋 구축 파이프라인 (SMMILE)

SMMILE은 11명의 의료 전문가(전문의 및 의대생)가 참여하여 구축되었다. 각 문제는 다음과 같은 구조를 가진다.

- **Query**: MLLM에게 제시될 질문, 관련 이미지, 그리고 정답(Ground-truth).
- **In-context Examples**: 쿼리 과업을 수행하는 데 도움이 되는 2개 이상의 예시(질문-이미지-정답 쌍).

전문가들은 웹 인터페이스를 통해 주제 범위, 데이터 출처, 답변 형식을 준수하며 문제를 생성하였다. 최종적으로 6개 의료 전문 분야와 13개 영상 모달리티를 포함하는 111개의 문제(총 517개의 triplet)가 구축되었다.

### 2. 확장 데이터셋 (SMMILE++)

모델의 강건성을 테스트하기 위해, SMMILE의 하위 집합에서 In-context examples의 순서를 치환(Permutation)하여 생성한 **SMMILE++**를 도입하였다. 추론 과정(Reasoning)이 필요한 문제를 제외하고, 문제당 최대 $4! = 24$가지의 순열을 생성하여 총 1,038개의 문제를 구축하였다.

### 3. 평가 태스크 및 지표

본 연구는 두 가지 형태의 생성 태스크를 정의한다.

- **Open-ended Generation**: 자유 텍스트 응답을 생성하는 과업.
- **Closed-ended Generation**: In-context examples에 나타난 답변 후보군 중에서 정답을 선택하는 다지선다형(MCQA) 과업.

평가 지표는 다음과 같다.

- **Exact Match (EM)**: 정규화 후 정답과 완전히 일치하는지 측정.
- **LLM-as-a-Judge**: Llama 3.3 70B 모델을 판별자로 사용하여, 생성된 응답이 정답과 의미적으로 일치하는지 이진(0 또는 1)으로 판정.
- **Accuracy**: MCQA 태스크에서의 정답 선택 확률.

## 📊 Results

### 1. 전반적인 성능 분석

15개의 MLLM을 평가한 결과, 대부분의 모델이 의료 multimodal ICL 능력에서 매우 취약한 모습을 보였다.

- **ICL의 제한적 효과**: Open-ended 평가(LLM-as-a-Judge)에서 ICL을 적용했을 때 zero-shot 대비 평균 성능 향상은 8%($\text{SMMILE}$) 및 9.4%($\text{SMMILE++}$)에 불과했다.
- **최고 성능 모델**: GPT-4o가 SMMILE에서 가장 높은 성능(Open-ended 49.88%, MCQA 58.85%)을 보였으며, SMMILE++에서는 Qwen2.5-VL-72B가 우세하였다. 하지만 최고 성능 모델조차 정답률이 약 50% 수준에 머물렀다.
- **의료 전용 모델의 한계**: MedGemma 4B, LLaVA-Med-7B와 같은 의료 특화 모델들이 유사한 크기의 일반 모델보다 유의미하게 뛰어난 ICL 능력을 보여주지 못했다. 특히 LLaVA-Med-7B는 ICL 적용 시 오히려 성능이 급격히 저하되는 현상이 관찰되었다.

### 2. 세부 분석 (Fine-Grained Analysis)

- **답변 형식별 성능**: Binary (Yes/No) 답변에서는 상대적으로 높은 성능을 보였으나, 수치적(Numerical) 답변이 필요한 문제에서는 모든 모델이 거의 실패하였다.
- **예시 개수의 영향**: 예시가 2개일 때는 zero-shot보다 성능이 향상되었으나, 예시 개수가 계속 증가할수록 오히려 성능이 떨어지거나 zero-shot 수준으로 회귀하는 경향이 나타났다. 이는 긴 입력값(interleaved image-text pairs)을 처리하는 능력이 부족함을 시사한다.
- **모달리티별 성능**: MRI와 Illustration 모달리티에서는 모든 모델이 정답을 맞히지 못했으며, 이는 MLLM이 다양한 의료 영상의 특성을 문맥을 통해 학습하는 데 어려움이 있음을 보여준다.

### 3. 예시 구성 요소 분석

- **품질의 중요성**: 단 하나의 무관한 예시(Irrelevant example)가 추가되는 것만으로도 성능이 평균 9.5%까지 하락하였다.
- **순서의 영향 (Recency Bias)**: 정답과 일치하는 가장 관련성 높은 예시가 리스트의 마지막에 위치할 때 성능이 최대 71%까지 향상되는 강한 Recency Bias가 모든 모델에서 공통적으로 관찰되었다.

## 🧠 Insights & Discussion

본 논문은 현재의 MLLM들이 고도로 설계된 전문가의 예시가 제공되더라도, 이를 통해 새로운 의료 과업을 학습하는 능력이 매우 부족하다는 것을 입증하였다. 특히 주목할 점은 도메인 특화 파인튜닝(Domain-specific fine-tuning)이 반드시 ICL 능력의 향상으로 이어지지 않는다는 것이다.

또한, 모델들이 수치적 추론과 특정 의료 영상 모달리티(MRI 등)에서 완전히 실패한다는 점은 실제 임상 적용을 위해서는 단순한 스케일 확장 이상의 구조적 개선이나 데이터 전략이 필요함을 시사한다. 특히 Recency Bias에 대한 극심한 의존도는 모델이 문맥 전체를 통합적으로 이해하는 것이 아니라, 마지막에 입력된 정보에 편향되어 응답하고 있음을 보여준다.

## 📌 TL;DR

본 연구는 의료 전문가들이 직접 큐레이션한 multimodal ICL 벤치마크 **SMMILE**과 그 확장판 **SMMILE++**를 제안하였다. 15종의 최신 MLLM을 평가한 결과, 의료 도메인의 multimodal ICL 능력은 매우 낮으며, 특히 노이즈에 취약하고 강한 Recency Bias를 보였다. 이는 현재의 MLLM이 임상 현장의 실제 워크플로우를 지원하기에는 일반화 능력이 부족함을 의미하며, 향후 의료 AI 연구가 단순 VQA를 넘어 정밀한 문맥 학습 능력 향상에 집중해야 함을 시사한다.
