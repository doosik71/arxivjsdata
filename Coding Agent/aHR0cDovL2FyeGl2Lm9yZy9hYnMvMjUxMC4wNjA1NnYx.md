# Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research

Gang Liu, Yihan Zhu, Jie Chen, Meng Jiang (2025)

## 🧩 Problem to Solve

본 논문은 과학적 알고리즘 발견(Scientific Algorithm Discovery)을 자동화하려는 LLM 기반 에이전트들의 한계를 해결하고자 한다. 기존의 접근 방식은 크게 두 가지 방향으로 나뉘지만 각각 치명적인 결함을 가지고 있다. 첫째, AlphaEvolve와 같은 순수 알고리즘 진화(Pure Algorithm Evolution) 방식은 LLM의 내부 지식에만 의존하기 때문에 복잡한 도메인에서 성능 향상이 빠르게 정체(plateau)되는 경향이 있다. 둘째, 순수 딥 리서치(Pure Deep Research) 방식은 외부 지식을 통해 아이디어를 제안하지만, 이를 실제 코드로 구현하고 검증하는 피드백 루프가 부족하여 비현실적이거나 구현 불가능한 솔루션을 제안하는 경우가 많다.

따라서 본 연구의 목표는 외부 지식 검색(External Knowledge Retrieval)과 실제 코드 구현 및 평가를 통한 피드백 루프를 통합하여, 단순한 개선을 넘어 실질적인 성능 도약을 이뤄낼 수 있는 에이전트인 DeepEvolve를 구축하는 것이다.

## ✨ Key Contributions

DeepEvolve의 핵심 아이디어는 **딥 리서치(Deep Research)와 알고리즘 진화(Algorithm Evolution)의 유기적 결합**이다. 단순히 아이디어를 생성하는 것에 그치지 않고, 다음과 같은 설계를 통해 연구-구현-검증의 사이클을 완성한다.

1.  **지식 기반의 가설 생성**: 웹 검색과 학술 자료(arXiv, PubMed 등)를 통해 도메인 특화된 지식을 습득하고, 이를 바탕으로 구체적인 pseudo-code를 포함한 연구 제안서를 작성한다.
2.  **실행 가능한 구현 체계**: 단일 파일 수정을 넘어 다중 파일에 걸친 코드 편집(Cross-file code editing) 기능을 도입하고, 실행 오류를 기반으로 스스로 수정하는 디버깅 에이전트를 추가하여 구현 성공률을 높였다.
3.  **진화적 데이터베이스 관리**: Island-based populations와 MAP-Elites 알고리즘을 사용하여, 성능뿐만 아니라 다양성과 복잡성을 고려한 알고리즘 샘플링 전략을 구축함으로써 탐색(Exploration)과 활용(Exploitation)의 균형을 맞추었다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 DeepEvolve의 차별점을 제시한다.

-   **FunSearch 및 AlphaEvolve**: LLM을 이용해 프로그램을 생성하고 평가 피드백을 통해 최적화하는 시스템이다. 그러나 AlphaEvolve는 외부 지식의 활용 없이 LLM 내부 지식에만 의존하며, 단일 파일 내에서만 코드를 진화시킨다는 한계가 있다.
-   **Deep Research 에이전트**: OpenAI의 Deep Research나 Gemini와 같은 시스템은 방대한 온라인 정보를 합성하여 가설을 세우는 데 능숙하지만, 제안된 가설이 실제로 작동하는지 검증하는 단계가 결여되어 있다.
-   **AI Scientists 및 Paper2Code**: 논문을 코드로 변환하거나 가설 생성을 자동화하려는 시도가 있었으나, EXP-Bench와 같은 벤치마크 결과 전체 파이프라인의 성공률이 1% 미만으로 매우 낮을 만큼 구현 능력이 부족함을 보여주었다.

DeepEvolve는 이러한 '근거 없는 진화'와 '검증 없는 연구' 사이의 간극을 메우기 위해 외부 검색-다중 파일 편집-자동 디버깅-진화적 선택을 하나의 루프로 통합하였다.

## 🛠️ Methodology

### 전체 파이프라인
DeepEvolve는 입력으로 문제 $P$, 초기 알고리즘 $f$, 사용자 지침 $u$를 받으며, 다음과 같은 6개 모듈의 순차적 실행을 통해 알고리즘을 업데이트한다.

$$\text{Plan} \rightarrow \text{Search} \rightarrow \text{Write} \rightarrow \text{Code} \rightarrow \text{Evaluation} \rightarrow \text{Evolutionary Selection}$$

### 주요 구성 요소 및 역할
1.  **Algorithmic Deep Research (Plan, Search, Write)**:
    -   **Planning**: 현재 알고리즘의 성숙도에 따라 탐색적 질문 또는 고영향력(high-impact) 질문을 생성한다.
    -   **Searching**: PubMed, arXiv 등에서 정보를 검색하고 요약한다.
    -   **Writing**: 수집된 증거와 기존 알고리즘, 영감을 주는 알고리즘들을 통합하여 pseudo-code가 포함된 제안서를 작성한다.

2.  **Algorithmic Implementation (Code, Debugging)**:
    -   **Coding Agent**: 다중 파일 코드베이스를 분석하여 최소한의 수정 영역을 찾아 타겟 업데이트를 수행한다.
    -   **Debugging Agent**: 코드 실행 중 발생하는 에러 메시지를 피드백으로 받아 최대 5회까지 자동으로 수정 시도를 한다. 성공하지 못할 경우 해당 알고리즘의 점수는 0점으로 처리된다.

3.  **Evaluation and Evolutionary Database**:
    -   **평가**: 문제 $P$의 평가 함수 $g$를 통해 점수 $s$를 산출한다.
    -   **Island-based populations**: 다음 세대의 후보를 샘플링하기 위한 풀로, 성능이 높은 후보를 선호하면서도 일정 확률로 탐색을 수행한다.
    -   **MAP-Elites**: 성능(Performance), 다양성(Diversity), 복잡성(Complexity)의 3차원 그리드에 알고리즘을 배치하여, 현재 후보와 인접한(유사한) 알고리즘을 '영감(Inspiration)'으로 제공한다.

### 수학적 정의 및 공식
과학적 문제는 $P = (D, g)$로 정의되며, 여기서 $D = \{(q_i, a_i)\}_{i=1}^N$는 평가 데이터, $g$는 예측값 $\hat{a}_i$와 정답 $a_i$를 비교하는 함수이다. 최종 점수는 다음과 같이 계산된다.
$$s = g(\{a_i\}_{i=1}^N, \{\hat{a}_i\}_{i=1}^N)$$
알고리즘 발견의 목표는 $s$를 최대화하는 함수 $f: Q \rightarrow A$를 찾는 것이다.

예를 들어, 분자 특성 예측(Molecular Prediction)에서 도입된 **InfoNCE loss**는 다음과 같이 정의되어 대조 학습(Contrastive Learning)을 수행한다.
$$\text{loss} = -\log \frac{\exp(\text{sim}(\text{pos})/\tau)}{\exp(\text{sim}(\text{pos})/\tau) + \sum \exp(\text{sim}(\text{neg})/\tau)}$$
여기서 $\tau$는 temperature 파라미터이며, DeepEvolve는 불확실성 가이드 기반의 부정 샘플링(Uncertainty-guided negative sampling)을 통해 이를 최적화한다.

## 📊 Results

### 실험 설정
-   **데이터셋 및 도메인**: 화학, 수학, 생물학, 재료, 특허 등 5개 도메인의 9개 벤치마크를 사용하였다. (예: Molecular Prediction, Circle Packing, Burgers' Equation 등)
-   **측정 지표**: 각 문제의 지표(AUC-ROC, RMSE, Pearson correlation 등)를 'New Score'라는 공통 형식으로 표준화하여 값이 높을수록 성능이 좋게 설계하였다.
-   **기준선(Baseline)**: SOTA 논문의 알고리즘이나 Kaggle 경진대회 우승 솔루션을 초기 알고리즘으로 설정하였다.

### 주요 결과
-   **정량적 성과**: 9개 작업 중 6개에서 효과성(Effectiveness)과 효율성(Efficiency) 모두 개선되었다. 특히 Circle Packing의 경우, 기존 알고리즘의 일반화 실패 문제를 해결하며 매우 큰 성능 향상을 보였다 (Table 2 참조).
-   **구현 성공률**: 디버깅 에이전트의 도입으로 Open Vaccine 작업의 실행 성공률이 $0.13$에서 $0.99$로 비약적으로 상승하였다 (Table 3 참조).
-   **아이디어의 질**: LLM-as-a-judge 평가 결과, DeepEvolve가 생성한 알고리즘이 초기 알고리즘보다 독창성(Originality)과 미래 잠재력(Future Potential) 면에서 높은 점수를 받았다.

## 🧠 Insights & Discussion

### 강점 및 분석
DeepEvolve는 단순한 하이퍼파라미터 튜닝이 아니라, **도메인 특화된 귀납적 편향(Inductive Bias)**을 알고리즘에 주입하는 능력을 보여주었다. 예를 들어, 화학 도메인에서는 분자 모티프(Molecular motifs)나 화학 문법을 활용한 제약 조건을 도입하였고, 수학적 문제에서는 전역 최적화(Global Optimization)나 Krylov subspace solver와 같은 원리 중심의 방법론으로 전환하는 양상을 보였다. 이는 피드백 루프가 휴리스틱한 수정에서 이론적 근거를 가진 방법론으로의 진화를 유도했음을 시사한다.

### 한계 및 비판적 해석
논문에서는 매우 긍정적인 결과가 제시되었으나, 일부 사례(Burgers' Equation, Polymer Prediction 등)에서 구현된 코드가 실제로 워크플로우 내에서 실행되지 않고 'place-holder' 형태로 작성되었다는 점이 명시되어 있다. 이는 딥 리서치 에이전트가 제안하는 아이디어의 복잡도가 여전히 코딩 에이전트의 구현 능력을 상회하는 경우가 있음을 의미하며, 제안된 모든 고도화된 아이디어가 실제 성능 향상에 직접적으로 기여했는지는 추가 검증이 필요하다.

## 📌 TL;DR

DeepEvolve는 외부 지식 검색을 수행하는 **Deep Research**와 실제 코드를 구현하고 검증하는 **Algorithm Evolution**을 결합한 AI 과학자 에이전트이다. 다중 파일 편집 및 자동 디버깅 시스템을 통해 구현 가능성을 높였으며, MAP-Elites 기반의 진화적 데이터베이스로 효율적인 탐색을 수행한다. 결과적으로 화학, 생물학 등 다양한 과학 도메인에서 기존 SOTA 알고리즘을 능가하는 새로운 알고리즘을 성공적으로 발견하였으며, 이는 향후 AI 기반의 자율적 과학 발견 연구에 중요한 프레임워크를 제공한다.