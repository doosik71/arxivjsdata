# Metrics for Explainable AI: Challenges and Prospects

Robert R. Hoffman, Shane T. Mueller, Gary Klein, Jordan Litman (2018)

## 🧩 Problem to Solve

본 논문은 설명 가능한 인공지능(Explainable AI, XAI) 시스템이 실제로 사용자에게 유용한지, 그리고 사용자가 AI의 작동 방식을 실용적인 수준에서 이해했는지를 어떻게 측정할 것인가라는 근본적인 문제를 다룬다. 특히 현대의 Machine Learning 및 Deep Net 시스템은 모델의 복잡성이 증가함에 따라 해석 가능성이 낮아졌으며, 이로 인해 의사 결정자가 시스템의 결과가 합리적이고 공정한지를 정당화하기 어려워졌다.

유럽연합(EU)의 알고리즘 결정에 대한 '설명 요구권(right to an explanation)'과 같은 법적 규제 움직임은 XAI의 중요성을 더욱 증폭시키고 있다. 따라서 본 연구의 목표는 XAI 시스템의 성능과 인간-기계 협업 성능을 평가하기 위한 핵심 측정 개념을 정의하고, 구체적인 측정 방법론과 심리측정적(psychometric) 평가 도구를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 XAI의 평가를 단순한 알고리즘의 해석 가능성 지표가 아닌, 인간의 인지 과정과 심리학적 관점에서 접근하여 포괄적인 측정 프레임워크를 제시했다는 점이다. 저자들은 설명의 질(Goodness)과 만족도(Satisfaction), 사용자의 멘탈 모델(Mental Model), 호기심(Curiosity), 신뢰 및 의존성(Trust and Reliance), 그리고 최종적인 작업 성능(Performance)이라는 6가지 핵심 측정 영역을 정의하였다.

특히, 단순한 설문 조사를 넘어 Content Validity Ratio(CVR)와 Discriminant Validity 분석을 통해 검증된 설명 만족도 척도를 제안하고, 멘탈 모델의 정량적 분석을 위한 Propositional Analysis 방법론을 제시함으로써 XAI 평가의 학술적/실무적 가이드라인을 제공하였다.

## 📎 Related Works

논문은 XAI의 뿌리가 과거의 Expert Systems 연구와 Intelligent Tutoring Systems(ITS)에 있음을 언급한다. 기존 연구들은 설명 생성 방법론에 집중했으나, 정작 생성된 설명이 사용자에게 어떻게 작용하는지를 측정하는 방법론은 부족했다.

또한, 신뢰(Trust) 측정과 관련하여 Cahour-Forzy, Jian et al., Madsen-Gregor 등 기존의 자동화 시스템 신뢰 척도들을 검토하였다. 그러나 기존의 척도들은 대개 인간 간의 신뢰나 특정 도메인(예: 의료 진단, 로봇 협업)에 특화되어 있어, XAI라는 일반적인 상황에 그대로 적용하기에는 한계가 있음을 지적하며 본 연구에서 이를 통합 및 수정하여 새로운 척도를 제안하였다.

## 🛠️ Methodology

본 논문은 XAI 평가를 위한 다각적인 측정 파이프라인을 제안하며, 각 구성 요소의 역할은 다음과 같다.

### 1. 설명의 질(Goodness)과 만족도(Satisfaction)

저자들은 '설명'을 단순한 문장이 아닌 상호작용의 결과로 정의하며, 이를 두 가지 관점에서 구분한다.

- **Explanation Goodness**: 설명 자체의 명확성과 정밀성을 평가하는 것으로, 독립적인 전문가가 사전에(a priori) 판단하는 체크리스트 방식이다.
- **Explanation Satisfaction**: 사용자가 실제로 설명을 듣고 느낀 만족도로, 사후에(a posteriori) 측정하는 심리적 척도이다.

만족도 척도 검증을 위해 $35$명의 전문가를 대상으로 CVR 방법을 사용하였으며, 내부 일관성을 측정하기 위해 Cronbach's alpha 계수를 산출하였다.

### 2. 멘탈 모델(Mental Model) 측정

사용자가 AI 시스템을 어떻게 이해하고 있는지 나타내는 멘탈 모델을 추출하기 위해 다음과 같은 방법론을 제시한다.

- **추출 방법**: Think-Aloud(사고 구술법), Prediction Task(예측 과제), Diagramming Task(다이어그램 작성) 등을 통해 사용자의 이해도를 끌어낸다.
- **분석 방법(Propositional Analysis)**: 추출된 사용자의 설명을 '개념(Concept) - 관계(Relation) - 명제(Proposition)' 단위로 분해하여 전문가의 모델과 비교 분석한다. 예를 들어, 사용자 모델에 포함된 명제가 전문가 모델의 명제와 얼마나 일치하는지를 통해 모델의 완전성(Completeness)을 측정한다.

### 3. 호기심(Curiosity) 및 신뢰(Trust) 측정

- **Curiosity**: 사용자가 왜 설명을 요청했는지 분석하는 'Curiosity Checklist'를 통해 지식 격차(Knowledge Gap)를 식별한다.
- **Trust**: 신뢰를 정적인 상태가 아닌 동적인 프로세스로 보며, 반복 측정(Repeat Measure)을 권장한다. 신뢰 척도는 신뢰성(Reliability), 예측 가능성(Predictability), 효율성(Efficiency) 등의 요인을 포함하는 Likert 척도로 구성된다.

### 4. 시스템 성능(Performance) 평가

최종적으로 인간-AI 협업 시스템의 성공 여부를 세 가지 수준에서 측정한다.

- **Primary Task Goals**: 작업 성공률, 효율성, 시간 대비 과업 완료 수.
- **User Performance**: AI의 출력을 얼마나 정확하게 예측하는가에 대한 정답률.
- **Work System Level**: 학습 곡선(Learning Curves) 분석 및 'Trials-to-Criterion'(기준 도달까지 필요한 시도 횟수)을 통해 시스템의 채택 가능성과 학습 속도를 측정한다.

## 📊 Results

논문은 제안한 설명 만족도 척도(Explanation Satisfaction Scale)의 타당성을 검증하기 위해 수행한 실험 결과를 제시한다.

- **내용 타당도(Content Validity)**: 전문가 집단으로부터 수집한 CVR 값은 $0.43$에서 $0.60$ 사이로 나타났으며, 이는 도메인 전문가들이 해당 척도가 만족도를 측정하는 유효한 지표라는 점에 상당 부분 동의함을 의미한다.
- **신뢰도 분석**: Cronbach's alpha 값이 $0.86$으로 측정되어 매우 높은 내부 일관성(Internal Consistency)을 보였다.
- **판별 타당도(Discriminant Validity)**: 의도적으로 작성된 '좋은 설명'과 '나쁜 설명'을 사용자가 구분하는지 테스트한 결과, 효과 크기(Effect Size)가 $Cohen's d = 1.5$ (이는 $r^2 = 0.36$에 해당)로 매우 크게 나타났다. 즉, 제안된 척도가 설명의 질적 차이를 효과적으로 구분해낼 수 있음을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 XAI 평가에서 가장 간과되기 쉬운 **'측정(Measure)'과 '지표(Metric)'의 차이**를 명확히 논의한다. 측정은 데이터를 수집하는 방법(어떻게 측정하는가)인 반면, 지표는 그 데이터를 해석하는 기준(어느 정도여야 성공적인가)이다. 저자들은 지표의 설정은 이론이 아닌 정책(Policy)과 적용 맥락에 따라 결정되어야 함을 강조한다.

또한, 인간과 XAI의 상호작용은 매우 복잡한 인지 시스템이므로 단일 방법론이 아닌 다중 방법론(Multi-method approach)을 사용해야 한다고 주장한다. 예를 들어, 사용자가 그린 다이어그램의 정확도가 높다고 해서 반드시 예측 과제(Prediction Task)의 성능이 높은 것은 아니라는 점을 들어, 상호 보완적인 측정 도구의 필요성을 역설한다.

한계점으로는 본 논문이 주로 측정 도구의 제안과 검증에 집중하고 있으며, 실제 다양한 AI 모델에 적용하여 일반화된 성능 지표를 도출하는 단계까지는 나아가지 않았다는 점이 있다.

## 📌 TL;DR

본 연구는 XAI 시스템의 '좋음'을 정의하기 위해 심리학과 인지 과학의 방법론을 결합한 종합적인 측정 프레임워크를 제안한다. 설명의 질, 사용자 만족도, 멘탈 모델, 호기심, 신뢰, 작업 성능이라는 6가지 차원을 정의하고, 특히 검증된 만족도 척도와 멘탈 모델 분석법을 통해 XAI 평가의 정량적 기반을 마련하였다. 이 연구는 향후 XAI 시스템이 단순히 '설명을 제공하는 것'을 넘어, '사용자가 실제로 이해하고 신뢰하게 만드는지'를 과학적으로 검증하는 데 중요한 역할을 할 것으로 기대된다.
