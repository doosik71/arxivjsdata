# A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise

Chaoyou Fu et al. (2023)

## 🧩 Problem to Solve

본 논문은 최근 공개된 Google의 Multi-modal Large Language Model(MLLM)인 Gemini가 기존의 최강 모델로 평가받던 OpenAI의 GPT-4V(ision)에 도전할 수 있는 수준의 시각적 전문성을 갖추었는지를 분석하는 것을 목표로 한다. 

현재 MLLM 분야에서는 대규모 언어 모델(LLM)에 시각적 이해 능력을 결합하여 다양한 멀티모달 태스크를 수행하려는 시도가 활발하다. 특히 GPT-4V는 업계의 표준이 될 만큼 강력한 성능을 보여주었으나, Gemini의 등장으로 새로운 경쟁 구도가 형성되었다. 따라서 본 연구는 Gemini Pro의 시각적 이해 능력을 다각도에서 평가하여 그 한계와 가능성을 확인하고, 폐쇄형 시스템(Closed-source)과 오픈 소스 모델 간의 성능 격차를 분석하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Gemini Pro의 시각적 전문성을 정성적 및 정량적 분석을 통해 종합적으로 평가한 프레임워크를 제시했다는 점이다. 연구진은 단순한 벤치마크 점수 비교를 넘어, 다음과 같은 네 가지 핵심 도메인을 설정하여 모델의 능력을 세밀하게 탐색하였다.

1. **Fundamental Perception**: 객체 중심, 장면 수준, 지식 기반의 기초적인 시각 인지 능력 평가.
2. **Advanced Cognition**: 텍스트가 풍부한 추론, 추상적 시각 추론, 과학 문제 해결 등 고차원적 인지 능력 평가.
3. **Challenging Vision Tasks**: 객체 검출(Object Detection), 참조 표현 이해(Referring Expression Comprehension) 등 전통적인 컴퓨터 비전의 난제 수행 능력 평가.
4. **Expert Capacity**: 자율주행, 의료 진단, 경제 분석 등 전문 영역에서의 지식 적용 능력 평가.

이를 통해 Gemini가 GPT-4V와 대등한 수준의 추론 능력을 갖추었음을 입증함과 동시에, 현세대 MLLM들이 공통적으로 겪고 있는 한계를 명확히 규명하였다.

## 📎 Related Works

논문에서는 GPT-4V를 비롯하여 LLaMA 기반의 모델들인 LLaVA, MiniGPT-4, 그리고 최신 오픈 소스 MLLM인 Sphinx를 언급한다. 

기존의 오픈 소스 모델들은 특정 데이터셋에 최적화된 경향이 있으며, 폐쇄형 모델인 GPT-4V에 비해 일반화 능력(Generalizability)과 복잡한 추론 능력에서 상당한 격차를 보인다. 특히 Sphinx와 같은 모델은 기초적인 인지 능력에서는 준수한 성능을 보일 수 있으나, 도메인 전반을 아우르는 범용적인 전문성에서는 폐쇄형 시스템인 Gemini나 GPT-4V에 비해 뒤처지는 한계가 있다.

## 🛠️ Methodology

본 연구는 정성적 분석과 정량적 분석의 두 가지 경로로 평가를 진행한다.

### 1. 정성적 분석 및 프롬프트 전략
연구진은 다양한 난이도의 샘플을 수집하여 모델의 반응을 분석하였으며, 이를 위해 다음과 같은 Prompt Technique를 사용하였다.
- **Simple Instruction Following**: "이 이미지를 묘사하라"와 같은 직접적인 명령 수행.
- **Visual Referring Prompt**: 시각적 마커나 실제 객체(손가락, 펜 등)를 사용하여 특정 영역을 지정하는 방식.
- **Chain-of-Thought (CoT)**: "단계별로 생각하라"는 지시를 통해 논리적 추론 과정을 유도.
- **In-context Few-shot Learning**: 추론 시 몇 가지 예시를 제공하여 모델이 작업 의도를 파악하게 하는 방식.

### 2. 분석 도메인 구성
- **Fundamental Perception**: 객체 수 세기, 차이점 찾기, 장면 묘사, 상식 및 전문 지식 기반의 인지 능력을 평가한다.
- **Advanced Cognition**: 표/차트 추론, LaTeX 코드 생성, Raven's Progressive Matrices와 같은 추상적 추론, 수학 및 물리 문제 해결 능력을 평가한다.
- **Challenging Vision Tasks**: Bounding Box 좌표 생성, 객체 추적(Object Tracking), 비디오 액션 인식 등을 수행한다.
- **Expert Capacity**: 의료 X-ray 판독, 주식 차트 분석, 자율주행 상황 판단, 로봇 조립 순서 계획 등 전문 분야의 적용 능력을 평가한다.

### 3. 정량적 평가
MLLM 전용 벤치마크인 **MME**를 사용하여 인지와 지각의 14개 서브 태스크를 평가하였다. MME는 각 이미지에 대해 'Yes'와 'No'가 정답인 두 가지 질문을 던져, 모델이 단순 추측이 아닌 실제 이해를 바탕으로 답했는지를 측정하는 Accuracy와 Accuracy+ 지표를 사용한다.

## 📊 Results

### 1. 정량적 결과 (MME Benchmark)
종합 점수 결과, Gemini가 가장 높은 점수를 기록하며 GPT-4V를 근소한 차이로 앞섰다.
- **Gemini**: $1933.4$
- **GPT-4V**: $1926.6$
- **Sphinx**: $1870.2$

세부적으로 보면, Sphinx는 기초적인 Perception(지각) 작업에서 강세를 보였으나, Cognition(인지) 작업, 특히 코드 추론(Code Reasoning)에서는 GPT-4V가 압도적인 성능을 보였다. Gemini는 전반적으로 균형 잡힌 성능을 보여 종합 점수 1위를 차지하였다.

### 2. 정성적 결과 분석
- **Gemini vs GPT-4V**: 두 모델 모두 강력한 추론 능력을 보였으나 답변 스타일에서 차이가 나타났다. GPT-4V는 매우 상세한 설명과 중간 추론 단계를 제공하는 경향이 있는 반면, Gemini는 직접적이고 간결한 답변을 선호한다.
- **시각적 전문성**: Gemini는 특정 전문가 영역(예: 원격 탐사 이미지 분석, 일부 전문 지식 인식)에서 GPT-4V보다 더 넓은 지식 범위를 보여주었다.
- **오픈 소스 모델의 한계**: Sphinx는 학습 데이터의 다양성 부족으로 인해 과학적 지식, HTML 코드 생성, 추상적 추론 등에서 폐쇄형 모델들에 비해 크게 뒤처지는 모습을 보였다.

## 🧠 Insights & Discussion

본 논문은 Gemini가 GPT-4V의 강력한 경쟁자임을 입증하였으나, 동시에 현세대 MLLM들이 공통적으로 가진 네 가지 치명적인 한계를 지적한다.

1. **공간 지각 능력의 부족 (Spatial Blindness)**: 두 모델 모두 객체의 상대적 위치(왼쪽/오른쪽)를 정확히 판별하는 데 어려움을 겪는다. 이는 정량적 지표(MME의 Position 메트릭)와 정성적 샘플 모두에서 확인되었다.
2. **불충분한 OCR 및 추상적 이해**: 이미지 내의 숫자나 문자를 오인식하는 경우가 많으며, 기하학적 도형의 관계를 파악하는 추상적 유도 능력(Abstract Inductive Ability)이 떨어진다.
3. **논리적 자기 일관성 결여 (Logical Self-consistency)**: 추론 과정(CoT)에서는 올바른 방향으로 가다가 최종 답변에서 갑자기 틀린 답을 내놓는 등, 중간 단계와 최종 결과가 일치하지 않는 현상이 발견된다.
4. **프롬프트 취약성 (Prompt Robustness)**: 동일한 질문이라도 프롬프트의 구성 방식에 따라 정반대의 답변을 내놓는 등, 입력값에 대한 안정성이 낮다.

또한, 감정 조건부 출력(Emotion-conditioned output) 태스크에서 존재하지 않는 사물을 묘사하는 환각(Hallucination) 현상이 관찰되었는데, 이는 모델의 상관관계 분석 능력이 지나치게 강력하여 발생하는 부작용으로 해석된다.

## 📌 TL;DR

본 연구는 Gemini Pro가 GPT-4V와 대등하거나 일부 영역에서 우월한 시각적 전문성을 갖추었음을 정성적/정량적으로 분석하였다. Gemini는 간결한 답변 스타일과 광범위한 지식 적용 능력을 보이며 GPT-4V의 강력한 도전자로 자리매김했다. 그러나 공간 지각력 부족, OCR 정확도 저하, 논리적 일관성 결여라는 MLLM의 공통적 한계는 여전히 존재하며, 이는 향후 AGI(인공일반지능)로 나아가기 위해 해결해야 할 핵심 과제이다. 본 연구는 향후 MLLM 연구가 세밀한 시각적 인코딩과 논리적 일관성 확보에 집중해야 함을 시사한다.