# What Does Explainable AI Really Mean? A New Conceptualization of Perspectives

Derek Doran, Sarah Schulz, and Tarek R. Besold (2017)

## 🧩 Problem to Solve

본 논문은 인공지능(AI) 분야에서 광범위하게 사용되는 '설명 가능성(Explainability)'과 '해석 가능성(Interpretability)'이라는 용어의 정의가 명확하지 않고 학술적으로 합의되지 않았다는 문제를 제기한다. 많은 연구가 자신의 시스템이 '설명 가능하다'고 주장하지만, 실제로는 결정의 근거인 'Why'가 아닌 단순한 작동 방식인 'How'만을 제시하는 경우가 많다.

특히 금융, 안전, 보안 및 개인의 권리와 직결된 고위험 상황에서 AI의 결정에 맹목적으로 의존하는 것은 위험하며, 윤리적·법적 기준을 평가하기 위해서는 시스템이 도출한 결론의 근거에 대한 심층적인 이해가 필수적이다. 따라서 본 논문의 목표는 다양한 AI 연구 분야에서 혼용되고 있는 설명 가능성의 개념을 체계적으로 분석하고, 이를 명확히 구분할 수 있는 새로운 개념적 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AI 시스템의 투명성과 설명 가능성을 네 가지 단계의 개념으로 정립한 것이다. 저자들은 시스템이 단순히 정보를 제공하는 수준을 넘어, 인간이 이해할 수 있는 추론 과정을 생성하는 단계까지 나아가야 한다고 주장한다.

제안된 핵심 개념은 다음과 같다:
1. **Opaque systems**: 내부 메커니즘이 완전히 가려진 블랙박스 시스템이다.
2. **Interpretable systems**: 입력이 출력으로 어떻게 수학적으로 매핑되는지 사용자가 분석하고 이해할 수 있는 시스템이다.
3. **Comprehensible systems**: 내부 메커니즘은 불투명할 수 있으나, 사용자가 결론에 도달한 이유를 추론할 수 있도록 돕는 기호(symbols, 시각화 등)를 출력하는 시스템이다.
4. **Truly explainable systems**: 인간의 사후 처리 없이도 자동화된 추론(automated reasoning)을 통해 스스로 설명 가능한 논거를 생성하는 시스템이다.

## 📎 Related Works

저자들은 Lipton(2016)의 연구를 인용하며, 머신러닝 컨퍼런스에서 '해석 가능성(interpretability)'이라는 용어가 수학적 정의 없이 준-수학적(quasi-mathematical) 방식으로 남용되고 있다는 점을 지적한다. 

기존의 접근 방식들은 주로 다음과 같은 한계를 보인다:
- **결정 트리(Decision Trees)**와 같은 모델이 생성하는 '규칙'은 결정이 내려진 방식(how)을 보여줄 뿐, 그 이유(why)를 설명하지 않는다.
- 딥러닝 기반의 **시각화(Visualizations)**나 **어노테이션(Annotations)**은 시스템이 직접 설명을 제공하는 것이 아니라, 인간 분석가가 자신의 추론 체계에 따라 사후에 해석해야 하는 보조 도구에 불과하다.

## 🛠️ Methodology

본 논문은 제안하는 개념적 프레임워크를 뒷받침하기 위해 두 가지 단계의 분석을 수행한다.

### 1. 코퍼스 기반의 언어 분석 (Corpus Analysis)
다양한 AI 연구 커뮤니티가 '설명 가능성'을 어떻게 다루는지 정량적으로 분석하기 위해 다음 네 가지 주요 컨퍼런스의 2007년부터 2016년까지의 논문 제목과 텍스트를 분석하였다:
- **ACL** (Natural Language Processing)
- **NIPS** (Connectionist/Neural Networks)
- **COGSCI** (Cognitive Science)
- **ICCV/ECCV** (Computer Vision)

분석 방법으로는 "explain", "interpret", "compreh"와 같은 특정 부분 문자열(substring) 매칭을 통한 빈도 분석과, 해당 용어 주변의 단어들을 추출하는 워드 클라우드(Word Cloud) 분석을 사용하였다.

### 2. 개념적 구분 및 프레임워크 설계
분석 결과를 바탕으로 AI 시스템을 다음과 같이 정의하고 구분한다.

- **Interpretable AI**: 모델의 투명성(Transparency)을 전제로 한다. 예를 들어, 선형 회귀 모델에서 가중치(weights)를 비교하여 피처의 중요도를 파악하는 것이 이에 해당한다. 이는 수학적 매핑 관계를 이해하는 능력에 기반한다.
- **Comprehensible AI**: 모델 자체가 불투명하더라도, 출력과 함께 제공되는 기호(단어, 이미지 등)를 통해 사용자가 자신의 직관과 지식을 동원해 의미를 구성하는 것이다. 예를 들어, CNN의 수용 영역(receptive field) 시각화가 이에 해당한다.
- **Truly Explainable AI**: 본 논문에서 제안하는 최종 단계로, 신경망의 연결주의적 접근과 기호적 추론(symbolic reasoning)을 결합한 **Neural-Symbolic Integration**을 통해, 시스템이 스스로 논리적 연역을 수행하여 설명을 생성하는 구조이다.

## 📊 Results

### 정량적 분석 결과
- **분야별 빈도 차이**: COGSCI 커뮤니티에서 설명 가능성 관련 용어의 사용 빈도가 다른 분야보다 압도적으로 높게 나타났다. 이는 인지과학의 본질적인 목적이 마음과 그 프로세스를 설명하는 것이기 때문으로 분석된다.
- **분야별 관점 차이 (Word Cloud 분석)**:
    - **ACL**: 'features', 'examples', 'words'와 함께 등장하며, 주로 특정 예시를 통해 결정 과정을 보여주는 데 집중한다.
    - **NIPS**: 'methods', 'algorithms', 'results'와 결합되어, 신경망 시스템이 입력을 출력으로 변환하는 방식(how)에 대한 설명에 집중한다.
    - **ICCV/ECCV**: 'data(images)', 'features(objects)'와 연결되며, 알고리즘이 이미지를 어떻게 사용하는지에 초점을 맞춘다.
    - **COGSCI**: 'participant', 'task', 'effect' 등의 단어와 결합되어, 기계 학습 모델보다는 인간의 인지 프로세스 설명에 집중한다.

### 개념적 분석 결과
저자들은 '해석 가능성'과 '이해 가능성'이 서로 독립적인 개념임을 밝혔다. 예를 들어, 의사는 환자에게는 증상과 지표라는 기호를 통해 설명하는 **Comprehensible model**처럼 행동하지만, 동료 의사와 논의할 때는 전문적인 진단 근거를 제시하는 **Interpretable model**처럼 행동한다.

## 🧠 Insights & Discussion

### 강점 및 통찰
본 논문은 단순히 기술적인 방법론을 제시하는 것이 아니라, AI 설명 가능성에 대한 철학적, 언어적 토대를 제공한다는 점에서 가치가 있다. 특히, 현재의 많은 'Explainable AI' 연구들이 실제로는 'Explanations을 가능하게 하는(enable)' 도구를 만드는 수준에 그치고 있으며, 시스템이 스스로 'Explanations을 생성하는(yield)' 단계에는 이르지 못했다는 날카로운 비판을 제기한다.

### 한계 및 논의사항
- **추론 엔진의 구현 문제**: Truly Explainable AI를 위해 제안한 Neural-Symbolic Integration은 개념적으로는 훌륭하나, 이를 실제 대규모 딥러닝 모델에 어떻게 효율적으로 통합할 것인가에 대한 구체적인 아키텍처나 수식은 제시되지 않았다.
- **사용자 주관성**: Comprehensibility는 사용자의 배경 지식과 직관에 의존하는 '등급적 개념(graded notion)'이므로, 객관적인 측정 지표를 설정하기 어렵다는 한계가 있다.

## 📌 TL;DR

본 논문은 AI의 설명 가능성을 **Opaque $\rightarrow$ Interpretable/Comprehensible $\rightarrow$ Truly Explainable**의 네 단계로 체계화하여 정의한다. 기존 연구들이 단순히 내부 구조를 보여주거나(Interpretable) 보조 기호를 제공하는(Comprehensible) 수준에 머물러 있음을 지적하며, 향후 연구는 신경망과 기호적 추론을 결합하여 시스템이 스스로 논리적인 설명 근거를 생성하는 **Truly Explainable AI** 방향으로 나아가야 함을 강조한다.