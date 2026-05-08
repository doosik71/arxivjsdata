# CAUSALEVOLVE: TOWARDS OPEN-ENDED DISCOVERY WITH CAUSAL SCRATCHPAD

Yongqiang Chen, Chenxi Liu, Zhenhao Chen, Tongliang Liu, Bo Han, Kun Zhang (2026)

## 🧩 Problem to Solve

본 논문은 AlphaEvolve와 같은 LLM 기반의 진화형 에이전트(Evolve-based agents)가 직면한 한계를 해결하고자 한다. 기존의 진화형 에이전트들은 LLM의 사전 지식과 추론 능력을 활용해 프로그램을 반복적으로 개선하며 오픈 엔드(open-ended) 과학 문제에 접근하지만, 두 가지 핵심적인 결함이 존재한다. 첫째, 진화 과정에서 구체적이고 타겟팅된 가이드(targeted guidance)가 부족하다. 둘째, 과거의 진화 경험을 통해 획득한 지식을 체계적으로 조직하고 활용하는 메커니즘이 부재하다.

이로 인해 기존 에이전트들은 진화 효율성이 저하되는 문제를 겪으며, 특히 성능의 임계치에 도달했을 때 최적해를 찾지 못하고 성능이 진동하는 oscillatory behavior를 보인다. 이는 인간 과학자가 관찰 데이터로부터 과학적 통찰을 요약하고 목적이 분명한 실험을 설계하는 '가이드된 발견(guided discovery)' 과정과 대조적이다. 따라서 본 연구의 목표는 인간과 유사하게 인과 관계를 추론하고 이를 통해 진화 과정을 안내하는 새로운 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 진화 기반의 과학적 발견 과정을 인과 관계 추론의 관점에서 재정의하고, 이를 구현한 **CausalEvolve** 프레임워크를 제안한 것이다. 중심적인 설계 아이디어는 **Causal Scratchpad**를 도입하여 진화의 가이드가 되는 인과적 요인(causal factors)을 식별하고 추론하는 것이다.

구체적으로는 프로그램 실행 결과에서 추출할 수 있는 결과 수준의 요인(outcome-level factors)과 프로그램 구조 자체에서 도출되는 절차 수준의 요인(procedure-level factors)을 모두 활용한다. 특히, 예상과 다른 결과가 나타나는 '놀라운 패턴(surprise patterns)'을 감지하고 이에 대해 가추법적 추론(abductive reasoning)을 수행함으로써 새로운 가설과 요인을 지속적으로 발굴하는 메커니즘을 구축하였다.

## 📎 Related Works

본 논문은 크게 두 가지 관련 연구 분야를 다룬다.

1. **AI Scientist Agents**: 최근 LLM을 활용해 문헌 조사, 가설 생성, 실험 설계 등을 자동화하려는 시도가 증가하고 있다. 특히 AlphaEvolve나 ShinkaEvolve와 같은 진화형 코딩 에이전트들은 특정 과학 문제에 대해 최적의 솔루션을 반복적으로 탐색하는 능력을 보여주었다. 그러나 이러한 방식은 주로 진화 알고리즘이나 상관관계 분석에 의존하며, 인과적 통찰에 기반한 체계적인 가이드가 부족하다는 한계가 있다.
2. **Causality for Scientific Discovery**: 구조화된 데이터에서 인과 그래프를 학습하거나, LLM의 지식을 활용해 인과 구조 탐색을 보조하는 연구들이 진행되어 왔다. 최근에는 LLM 기반 에이전트에 인과 추론 도구를 통합하여 표 형식 데이터(tabular data)를 분석하는 연구 등이 제안되었으나, 이를 프로그램 진화 기반의 과학적 발견 프로세스에 직접적으로 통합한 시도는 부족했다.

CausalEvolve는 기존의 단순 반복적 진화에서 벗어나, 인과 관계를 명시적으로 모델링하여 샘플 효율성을 높이고 일반화 성능을 개선한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 과학적 발견의 이론적 정식화 (POMDP)

본 논문은 과학적 발견 과정을 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP)으로 정의한다.

- **상태($S$):** 숨겨진 과학적 지식 $\theta^{sci}$이며, 이는 구조적 인과 모델(Structural Causal Model, SCM) $\theta^{sci} = (G, F, P^U)$로 파라미터화된다.
- **행동($A$):** 평가할 후보 프로그램 $p_t$의 선택이다.
- **관측($O$):** 프로그램 실행 결과인 $y_t$이다.
- **목표:** 유한한 예산 $T$ 내에서 $\hat{p} = \arg \max_p F(p, \theta^{sci})$를 찾는 것이다.

이때 프로그램 $p$의 평가는 SCM 상의 개입(intervention)으로 간주하며, 결과 값은 다음과 같이 정의된다.
$$F(p; \theta^{sci}) := \mathbb{E}[Y | \text{do}(X = x_p), \theta^{sci}]$$

### 2. Causal Scratchpad의 구성 요소

CausalEvolve는 두 가지 수준의 요인을 통해 진화를 가이드한다.

**A. 결과 수준 요인 (Outcome-level Factors):**
프로그램 실행 결과(예: 행렬의 행 직교성, 원들의 밀도 등)에서 추출 가능한 실수 값 지표들이다.

- **Causal Planner:** LLM이 정의한 요인 집합 $m$을 바탕으로 $\mathcal{A} := \cup_{m \in m} \{(m, +1), (m, -1)\}$의 행동 공간을 정의한다.
- **작동 방식:** 선택된 행동 $(m, d)$에 따라 프로그램을 정렬하고 상위 프로그램을 영감(inspiration)으로 삼아 다음 세대를 생성한다.
- **보상 함수:** 새로운 자식 프로그램의 결과 $y^c$와 현재까지의 최적값 $v_t$를 이용하여 보상을 계산한다.
$$\text{Reward } R^a := (y^c - \tau \cdot v_t)^+$$
여기서 $\tau \in (0, 1)$는 최적값 갱신이 드문 사건임을 고려한 할인 계수이다. Multi-Armed Bandit(MAB)을 통해 탐색(exploration)과 활용(exploitation)을 교대로 수행하며 최적의 요인을 찾는다.

**B. 절차 수준 요인 (Procedure-level Factors):**
프로그램 코드 자체의 설계 특성(예: 사용된 최적화 기법)을 의미한다.

- **COAT 프레임워크:** LLM을 활용해 비정형 데이터에서 성능 차이를 설명하는 절차적 요인을 식별한다.
- **가추법적 추론 (Abductive Reasoning):** 추정된 치료 효과(treatment effect)가 예상과 반대로 나타나거나 급격한 변화가 생기는 'surprise patterns'를 감지한다. LLM은 이 패턴을 설명하기 위해 숨겨진 교란 변수(confounder)나 새로운 가설을 제안하며, 이는 향후 실험의 새로운 방향이 된다.

## 📊 Results

### 실험 설정

- **비교 대상:** 최신 진화형 에이전트인 ShinkaEvolve, 그리고 CausalPlanner(Meta) 및 COAT 단독 모델과 비교하였다.
- **사용 모델:** Grok-4.1-fast-reasoning.
- **평가 작업 (4가지):**
    1. **Hadamard Matrix ($n=29$):** 절대 행렬식 $|det(H)|$ 최대화.
    2. **Second Autocorrelation Inequality:** 특정 비율 $R(f)$ 최소화.
    3. **Circle Packing ($N=26$):** 단위 정사각형 내 원들의 반지름 합 $\sum r_i$ 최대화.
    4. **AIME Mathematical Problem Solving:** 2024년 AIME 수학 문제 풀이 정확도.

### 정량적 결과

실험 결과, CausalEvolve는 모든 작업에서 ShinkaEvolve보다 우수한 평균 성능과 최적 성능을 보였다.

- **효율성:** CausalPlanner의 결과-수준 요인 덕분에 초기 단계(early steps)에서 더 빠르게 성능이 향상되었다.
- **최적성:** COAT와 가추법적 추론을 포함한 절차-수준 요인 덕분에 최종 도달 성능이 더 높았다.
- **특이 사항:** AIME 작업에서 CausalEvolve는 38.89%의 정확도를 기록하였는데, 이는 ShinkaEvolve의 원 논문에서 더 복잡한 앙상블 모델을 사용했을 때의 결과(34.4%)보다 높은 수치이다.

## 🧠 Insights & Discussion

본 논문은 이론적 분석을 통해 인과적 지식의 필요성을 증명하였다.

- **샘플 효율성:** Theorem 3.2를 통해, 인과 구조를 알고 있는 경우 $O(d \log K)$의 턴만으로 최적해에 근접할 수 있지만, 블랙박스 방식은 $O(K)$의 턴이 필요함을 보였다 ($K \gg d$인 상황).
- **일반화 가능성:** Theorem 3.3을 통해, 소스 환경(source environment)에서 인과적 정체성을 구분하지 못하면 타겟 환경(target environment)에서 항상 최적해를 찾지 못하는 하한선이 존재함을 증명하였다.

**비판적 해석:**
CausalEvolve는 단순히 LLM의 생성 능력에 의존하는 것이 아니라, 관측 $\rightarrow$ 요인 추출 $\rightarrow$ 인과 추론 $\rightarrow$ 가이드된 생성이라는 루프를 구축함으로써 진화의 무작위성을 획기적으로 줄였다. 다만, 결과 수준 요인의 정의가 LLM의 초기 프롬프트에 의존하므로, 매우 생소한 도메인에서는 LLM이 유의미한 요인을 처음에 정의하지 못할 가능성이 있다. 또한, 가추법적 추론이 실제 물리적/수학적 진실보다는 LLM의 환각(hallucination)에 기반한 가설을 생성할 위험이 존재한다.

## 📌 TL;DR

CausalEvolve는 LLM 기반 진화형 에이전트의 무작위 탐색 한계를 극복하기 위해 **Causal Scratchpad**를 도입한 프레임워크이다. 결과-수준 및 절차-수준의 인과적 요인을 식별하고, 특히 예상 밖의 결과(surprise patterns)에 대해 가추법적 추론을 수행함으로써 진화 과정을 정교하게 가이드한다. 실험 결과, 4가지 고난도 과학 문제에서 기존 SOTA 모델인 ShinkaEvolve보다 뛰어난 탐색 효율성과 최종 성능을 달성하였으며, 이는 과학적 발견 과정에 인과 관계 추론을 통합하는 것이 필수적임을 시사한다.
