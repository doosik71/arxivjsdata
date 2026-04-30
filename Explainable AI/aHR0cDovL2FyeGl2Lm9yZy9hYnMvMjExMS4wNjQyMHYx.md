# EXPLAINABLE AI (XAI): A SYSTEMATIC META-SURVEY OF CURRENT CHALLENGES AND FUTURE OPPORTUNITIES

Waddah Saeed, Christian Omlin (2021)

## 🧩 Problem to Solve

최근 10년 동안 인공지능(AI) 기술은 비약적으로 발전하여 다양한 복잡한 문제들을 해결하고 있다. 그러나 이러한 성능 향상은 모델의 복잡성 증가와 함께 투명성이 결여된 Black-box AI 모델의 채택이라는 부작용을 초래하였다. 특히 의료나 보안과 같이 높은 정확도 외에도 신뢰성과 투명성이 필수적인 임계 도메인(Critical domains)에서는 이러한 불투명성이 AI 도입의 주요 장애물이 되고 있다.

현재까지 Explainable AI (XAI)에 관한 다수의 리뷰 논문들이 발표되었으나, 각 논문에서 제시하는 도전 과제(Challenges)와 향후 연구 방향들이 파편화되어 있어 연구자들이 통합적인 관점에서 참고하기 어렵다는 문제가 있다. 따라서 본 논문의 목표는 기존의 XAI 관련 서베이 연구들을 체계적으로 분석하여, XAI의 도전 과제와 미래 기회들을 일반적인 관점과 머신러닝 생애주기(ML Life Cycle) 관점에서 체계적으로 구조화하여 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 파편화된 XAI 연구 방향들을 체계적으로 정리한 '메타-서베이(Meta-survey)'를 수행했다는 점에 있다. 구체적인 설계 아이디어와 기여 사항은 다음과 같다.

1.  **XAI 도전 과제의 체계적 분류**: XAI의 문제점과 연구 방향을 (1) 일반적인 도전 과제와 (2) 머신러닝 생애주기(설계, 개발, 배포) 단계별 도전 과제로 나누어 구조화하였다.
2.  **Explainability와 Interpretability의 정의 제안**: 학계에서 혼용되어 사용되는 두 용어에 대해 다음과 같이 차별화된 정의를 제안하여 개념적 혼선을 줄이고자 하였다.
    *   **Explainability**: 특정 니즈를 충족시키기 위해 대상 청중(Targeted audience)에게 통찰(Insights)을 제공하는 행위.
    *   **Interpretability**: 제공된 통찰이 대상 청중의 도메인 지식(Domain knowledge)에 비추어 얼마나 이해 가능한가에 대한 정도.
3.  **미래 연구 가이드라인 제공**: 58편의 주요 서베이 논문을 분석하여 총 39개의 세부 도전 과제를 도출함으로써, 향후 XAI 연구자들이 탐색해야 할 구체적인 이정표를 제시하였다.

## 📎 Related Works

논문은 XAI가 필요한 이유를 다섯 가지 관점에서 설명하며 기존 연구들의 배경을 다룬다.

-   **규제적 관점(Regulatory perspective)**: EU의 GDPR과 같은 법적 규제에서 명시하는 '설명 요구권(Right to explanation)'을 충족시키기 위해 XAI가 필수적이다.
-   **과학적 관점(Scientific perspective)**: Black-box 모델이 데이터로부터 추출한 지식을 가시화함으로써 새로운 과학적 발견을 가능하게 한다.
-   **산업적 관점(Industrial perspective)**: 모델의 성능과 해석 가능성 사이의 트레이드-오프(Trade-off)를 해결하여 산업 현장의 신뢰도를 높이고 규제 리스크를 줄인다.
-   **모델 개발 관점(Model's developmental perspective)**: 모델의 디버깅, 강건성(Robustness) 향상, 편향(Bias) 제거 및 최적화를 위해 내부 작동 원리를 이해해야 한다.
-   **최종 사용자 및 사회적 관점(End-user and social perspectives)**: 적대적 공격(Adversarial attacks)에 의한 오작동이나 사회적 편견이 반영된 불공정한 결정에 대응하여 사용자 신뢰를 구축해야 한다.

## 🛠️ Methodology

본 연구는 Kitchenham과 Charters가 제안한 체계적 문헌 고찰(Systematic Literature Review, SLR) 프로토콜을 기반으로 수행되었다.

### 1. 연구 질문 설정
본 메타-서베이의 핵심 연구 질문은 "기존의 서베이 연구들에서 보고된 XAI의 도전 과제와 연구 방향은 무엇인가?"이다.

### 2. 문헌 검색 및 선정 절차
-   **검색 키워드**: `explainable`, `XAI`, `interpretable`과 같은 XAI 핵심어와 `survey`, `review`, `challenge`, `research direction` 등의 리뷰 관련 키워드를 조합하여 사용하였다.
-   **데이터베이스**: Scopus, Web of Science, Science Direct, IEEE Xplore, Springer Link, ACM, Google Scholar, arXiv 등 8개 주요 DB를 활용하였다.
-   **선정 기준(Inclusion Criteria)**: $\text{XAI에 대한 서베이 논문일 것}$ $\land$ $\text{XAI의 도전 과제 또는 연구 방향을 제시할 것}$.
-   **제외 기준(Exclusion Criteria)**: 영어로 작성되지 않은 논문, 도전 과제나 연구 방향에 대한 논의가 없는 서베이 논문.

### 3. 분석 및 구조화 프레임워크
최종 선정된 58편의 논문을 분석하여 다음과 같은 프레임워크로 도전 과제를 분류하였다.
-   **General Challenges**: 도메인에 관계없이 공통적으로 발생하는 문제.
-   **ML Life Cycle Phases**: 
    -   **Design Phase**: 데이터 수집 및 준비 단계.
    -   **Development Phase**: 모델 아키텍처 설계 및 학습 단계.
    -   **Deployment Phase**: 실제 시스템 적용 및 운영 단계.

## 📊 Results

본 논문은 XAI의 도전 과제를 크게 세 가지 범주로 나누어 상세히 분석하였다.

### 1. 일반적 도전 과제 (General Challenges)
-   **형식화(Formalism)**: 설명에 대한 표준 정의와 정량적 평가 지표의 부재가 심각하며, 객관적 지표와 인간 중심 지표 사이의 조화가 필요하다.
-   **다학제적 협력(Multidisciplinary collaborations)**: 심리학, 인지과학, HCI(Human-Computer Interaction) 전문가와의 협업을 통해 인간 중심의 XAI를 구현해야 한다.
-   **사용자 경험(User Experience)**: 전문가와 비전문가 등 사용자의 배경 지식에 따라 서로 다른 수준의 설명이 제공되어야 한다.
-   **신뢰성 및 공정성**: 단순한 설명 제공을 넘어, 모델의 편향성을 제거하고 책임성(Accountability)을 확보하는 방안이 연구되어야 한다.
-   **인과적/대조적 설명(Causal/Contrastive Explanations)**: "어떻게"가 아닌 "왜"에 집중하는 인과적 설명과, "왜 A가 아니라 B인가"를 다루는 대조적 설명의 발전이 필요하다.
-   **데이터 다양성**: 이미지와 텍스트 외에 그래프(Graph), 시공간 데이터(Spatio-temporal data) 및 이종 데이터(Heterogeneous data)에 대한 XAI 기법이 부족하다.

### 2. 설계 단계의 도전 과제 (Design Phase)
-   **데이터 품질 소통**: 데이터의 편향이나 불완전성이 설명 결과에 미치는 영향을 사용자에게 어떻게 전달할 것인가에 대한 연구가 필요하다.
-   **데이터 공유 및 프라이버시**: 설명 제공을 위해 원본 데이터를 보존하거나 공유할 때 발생하는 프라이버시 침해 문제를 해결하기 위해 Federated Learning 등의 기법이 제안된다.

### 3. 개발 단계의 도전 과제 (Development Phase)
-   **지식 주입(Knowledge Infusion)**: 전문가의 도메인 지식을 학습 과정에 직접 주입하여 더 해석 가능한 모델을 만드는 방안이 필요하다.
-   **학습 과정 설명**: 모델의 최종 결과뿐만 아니라 학습 과정(Training process)을 모니터링하고 디버깅할 수 있는 시각적 분석 도구가 필요하다.
-   **모델 혁신**: 불투명한 모델의 표현력과 투명한 모델의 의미론적 특성을 결합한 하이브리드 모델 연구가 필요하다.

### 4. 배포 단계의 도전 과제 (Deployment Phase)
-   **온톨로지(Ontology) 활용**: 도메인 지식을 구조화한 온톨로지를 사용하여 설명의 질을 높이는 방안이 제시된다.
-   **보안 및 안전**: XAI가 제공하는 정보가 오히려 적대적 공격(Adversarial attacks)의 힌트가 될 수 있는 보안 취약점 문제가 존재한다.
-   **인간-기계 팀워크(Human-machine teaming)**: 정적인 설명에서 벗어나 대화형/인터랙티브한 설명을 통해 인간과 AI가 협력하는 체계가 필요하다.
-   **특수 분야 적용**: 강화학습(RL), AI 플래닝(XAIP), 추천 시스템 등 특정 태스크에 최적화된 XAI 기법의 개발이 요구된다.

## 🧠 Insights & Discussion

본 논문은 단순한 나열을 넘어 XAI 연구의 본질적인 방향성에 대해 다음과 같은 통찰을 제공한다.

-   **정의의 중요성**: 현재 XAI 분야의 가장 큰 문제는 '설명(Explanation)'과 '해석 가능성(Interpretability)'에 대한 합의된 정의가 없다는 점이다. 이는 평가 지표의 부재로 이어지며, 결과적으로 연구 간의 정량적 비교를 불가능하게 만든다.
-   **사용자 중심 설계의 필요성**: 많은 XAI 기법들이 데이터 과학자의 관점에서 개발되었으나, 실제 XAI의 가치는 도메인 전문가나 최종 사용자가 이를 어떻게 받아들이느냐에 달려 있다. 따라서 HCI 관점의 접근이 필수적이다.
-   **성능-해석 가능성 트레이드-오프**: 복잡한 모델이 더 정확하다는 믿음이 지배적이지만, 특정 도메인에서는 약간의 성능 저하를 감수하더라도 완전히 투명한 모델을 사용하는 것이 더 안전하고 효율적일 수 있다.
-   **비판적 해석**: 본 논문은 매우 방대한 도전 과제를 제시하고 있으나, 각 과제 간의 우선순위나 상호 의존성에 대한 분석은 부족하다. 또한, 제안된 39개의 포인트가 다소 나열식으로 구성되어 있어, 실제 구현을 위한 구체적인 알고리즘적 가이드라인보다는 개념적 로드맵에 가깝다는 한계가 있다.

## 📌 TL;DR

이 논문은 파편화되어 있던 XAI의 도전 과제와 미래 연구 방향을 체계적으로 정리한 **메타-서베이 보고서**이다. 특히 **'Explainability'와 'Interpretability'의 개념적 구분**을 시도하였으며, XAI의 과제들을 **일반적 관점**과 **머신러닝 생애주기(설계 $\rightarrow$ 개발 $\rightarrow$ 배포) 관점**으로 구조화하여 제시하였다. 이 연구는 향후 XAI 연구자들이 자신의 연구 위치를 파악하고 새로운 연구 주제를 발굴하는 데 중요한 참조점이 될 것으로 기대된다.