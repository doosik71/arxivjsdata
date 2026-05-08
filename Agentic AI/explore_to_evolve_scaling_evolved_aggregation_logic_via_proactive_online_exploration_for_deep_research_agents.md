# Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents

Rui Wang, et al. (2025)

## 🧩 Problem to Solve

본 논문은 심층 연구 웹 에이전트(Deep Research Web Agents)가 단순히 정보를 찾는 것(Information Seeking)을 넘어, 수집된 다양한 정보를 엄격하게 분석하고 합성하는 정보 집계(Information Aggregation) 능력이 부족하다는 문제를 제기한다.

기존의 오픈소스 웹 에이전트 시스템들은 주로 특정 정보를 정확히 찾아내는 검색 능력 향상에 집중해 왔으며, 이는 에이전트가 일관된 통찰력을 생성하거나 심도 있는 연구 보고서를 작성하는 능력을 제한하는 핵심 요인이 된다. 특히, 기존의 다단계 추론(Multi-hop) 데이터셋들은 실제 웹의 동적인 특성을 반영하지 못하거나, 단순히 엔티티를 추적하는 수준에 그쳐 복잡한 집계 논리를 학습시키기에 부족하다는 한계가 있다.

따라서 본 연구의 목표는 웹상의 다양한 소스에서 정보를 수집하고, 이를 복잡한 논리로 집계하여 검증 가능한 정답을 도출해야 하는 학습 데이터를 확장 가능하게(Scalable) 구축하고, 이를 통해 정보 집계 능력이 강화된 에이전트 파운데이션 모델인 `WebAggregator`를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Explore to Evolve**라는 패러다임이다. 이는 사람이 일일이 데이터를 구축하는 대신, 에이전트가 스스로 실제 웹을 탐색하고 그 결과물을 바탕으로 복잡한 집계 과제를 생성하도록 만드는 자동화 파이프라인이다.

1. **Explore (Proactive Online Web Exploring):** 에이전트가 앵커 URL(Anchor URL)로부터 시작하여 실제 라이브 웹을 능동적으로 탐색하며, 텍스트, 파일, 이미지 등 다양한 형태의 근거 데이터를 수집한다.
2. **Evolve (Automatic Aggregation Logic Synthesis):** 수집된 데이터를 바탕으로 12가지 고수준 논리 유형(High-level logical types)을 조합하고 구체화하여, 단순 검색으로는 풀 수 없는 복잡한 QA 쌍을 스스로 생성한다.
3. **WebAggregatorQA 데이터셋:** 위 과정을 통해 11개 도메인, 5만 개 이상의 웹사이트를 아우르는 약 10K 규모의 고난도 데이터셋을 구축하였다.
4. **WebAggregator 모델:** Qwen3 시리즈를 기반으로 SFT(Supervised Fine-Tuning)를 진행하여, GPT-4.1의 성능에 필적하거나 이를 능가하는 웹 에이전트 모델 제품군을 제안한다.

## 📎 Related Works

기존의 웹 에이전트 학습을 위한 데이터셋 및 벤치마크는 다음과 같은 한계를 가진다.

- **정적 데이터셋의 한계:** HotpotQA나 Musique와 같은 기존 다단계 QA 데이터셋은 실제 웹 상호작용이 결여되어 있으며, 모델의 내부 파라미터 지식만으로 해결 가능한 경우가 많다.
- **정보 탐색 중심의 설계:** TaskCraft, WebShaper, WebDancer 등 최신 연구들은 오프라인의 정적 페이지를 그래프로 연결해 문제를 구성하며, 주로 엔티티 추적이나 단순 필터링과 같은 '정보 탐색'에 치중되어 있다. 본 논문의 분석에 따르면, 일부 데이터셋(WebWalkerQA)의 30.79%는 단순 텍스트 파싱만으로 해결 가능할 정도로 집계 복잡도가 낮다.
- **평가 지표의 부재:** GAIA와 같은 일반 벤치마크가 존재하지만, 최근 에이전트들이 높은 성능을 보이면서 정보 집계 능력을 정밀하게 측정할 수 있는 더 어려운 벤치마크의 필요성이 대두되었다.

## 🛠️ Methodology

### 전체 파이프라인 구조

데이터 구축 과정은 `Anchor URL 수집 $\rightarrow$ Proactive Web Exploring $\rightarrow$ Automatic Aggregation Logic Synthesis $\rightarrow$ Quality Control $\rightarrow$ Trajectory Sampling` 순으로 진행된다.

### 1. Proactive Online Web Exploring (Explore)

에이전트가 앵커 URL에서 시작하여 `Search`, `Visit`, `Click`, `FileRead` 등의 도구를 사용하여 웹을 탐색한다. 이때 과제의 난이도와 지식 범위의 포괄성을 보장하기 위해 최소 방문 페이지 수($N=7$)를 강제하며, 이를 통해 정적 텍스트뿐 아니라 PDF, CSV, JavaScript 렌더링 콘텐츠 등 이질적인 소스로부터 정보를 수집한다.

### 2. Automatic Aggregation Logic Synthesis (Evolve)

수집된 정보를 바탕으로 QA 쌍을 생성하기 위해 4가지 상위 카테고리, 12가지 하위 논리 연산을 정의하여 에이전트에게 가이드라인으로 제공한다.

- **Element Operations:** 개별 요소 간의 수학적 계산(`Math`), 간접적인 단서를 통한 대상 식별(`Inverse`)
- **Set Operations:** 조건에 맞는 요소 추출(`Filter`), 집합 간의 교집합/합집합(`Compose`), 멤버십 확인(`Existence`)
- **Temporal Reasoning:** 시간에 따른 변화 분석(`Change`), 시간 간격 계산(`TempCalc`)
- **Scientific Analysis:** 표준편차/평균 등 통계 분석(`Statistic`), 피어슨 상관계수 계산(`Correlate`), 추세 예측(`Predict`), 연산 집약적 작업(`CompIntensive`)

에이전트는 이 고수준 가이드를 구체적인 추론 체인으로 발전시켜, 예를 들어 "통계 분석 $\rightarrow$ 표준편차 계산"과 같은 구체적인 문제를 생성한다.

### 3. Quality Control 및 학습 절차

- **품질 관리:** 2단계 정제 과정을 거친다. 1단계에서는 체크리스트 기반의 자가 수정(Self-refinement)을 수행하고, 2단계에서는 별도의 데이터 체크 에이전트가 참조 URL과 정답의 일치성을 검증한다.
- **데이터 누출 방지:** 키워드 블랙리스트를 통해 기존 벤치마크 데이터셋이 포함된 페이지를 제외한다.
- **훈련 방법:** GPT-4.1 기반의 에이전트가 `SmolAgents` 프레임워크를 사용하여 과제를 해결한 궤적(Trajectory)을 수집한다. 정답이 일치하는 궤적만을 선택하는 Rejection Sampling을 적용하여 6,184개의 고품질 궤적을 확보하였다.
- **학습 목표:** 궤적 데이터 $(\text{question}, a_1, o_1, \dots, a_n, o_n, \text{answer})$를 사용하여 Qwen 모델을 SFT한다. 학습 시 질문과 관찰값($o_i$)은 마스킹 처리하여 에이전트가 적절한 행동($a_i$)과 최종 정답을 예측하도록 유도한다.

## 📊 Results

### 실험 설정

- **모델:** Qwen2.5 및 Qwen3 (8B, 32B) 기반의 WebAggregator.
- **벤치마크:** GAIA-text(텍스트 기반 하위 집합) 및 자체 구축한 WebAggregatorQA 테스트 셋.
- **지표:** Pass@1 (정답률)을 사용하며, 정답 여부는 GPT-4.1가 판정한다.

### 주요 결과

- **성능 향상:** WebAggregator-8B는 GAIA-text에서 GPT-4.1와 대등한 성능을 보였으며, WebAggregator-32B는 GPT-4.1보다 10% 이상 높은 성능을 기록하며 Claude-3.7-sonnet에 근접하였다.
- **집계 능력의 병목 확인:** GPT-4.1과 Claude-3.7-sonnet 모두 GAIA-text에서는 높은 성능을 보였으나, 정보 집계 능력을 집중 평가하는 WebAggregatorQA 테스트 셋에서는 각각 25.8%, 28.3%로 성능이 급격히 하락하였다.
- **전이 가능성:** WebWalkerQA 및 XBench와 같은 타 벤치마크에서도 WebAggregator 모델들이 기존 SFT 모델들보다 우수한 성능을 보여, 제안된 데이터 구축 방식의 범용성을 입증하였다.

## 🧠 Insights & Discussion

### 정보 집계의 어려움

저자들은 에이전트가 모든 참조 URL을 성공적으로 방문했음에도 불구하고 문제를 틀리는 사례를 분석하였다. 표 5에 따르면 모든 URL을 방문했을 때의 정답률이 전체 평균보다 높긴 하지만, 여전히 완벽한 정답을 내지 못하는 경우가 많다. 이는 단순히 정보를 '찾는' 것과 찾은 정보를 '분석 및 합성'하는 것 사이에 큰 간극이 있음을 시사한다.

### 도구 사용 패턴의 변화

WebAggregatorQA를 해결하는 에이전트는 WebWalkerQA에 비해 총 단계(Steps) 수는 많지만, 도구 호출 밀도(Tool call density)는 낮게 나타났다. 이는 에이전트가 외부 도구를 통해 새로운 정보를 계속 찾는 것보다, 이미 수집된 정보를 바탕으로 내부적인 추론 및 집계 단계(Reasoning steps)에 더 많은 시간을 할애해야 함을 의미한다.

### 데이터 효율성

Qwen3-8B 모델을 대상으로 한 소규모 실험에서, 단 1,200개의 샘플만으로도 GAIA-text에서 38.83%의 정답률을 기록하였다. 이는 `Explore to Evolve`로 생성된 데이터의 개별 샘플이 가진 정보 밀도와 품질이 매우 높음을 보여준다.

## 📌 TL;DR

본 논문은 웹 에이전트의 고질적인 약점인 **정보 집계(Information Aggregation)** 능력을 강화하기 위해, 에이전트가 스스로 웹을 탐색하고 복잡한 논리 과제를 생성하는 **Explore to Evolve** 패러다임을 제안한다. 이를 통해 구축된 **WebAggregatorQA** 데이터셋으로 학습된 **WebAggregator** 모델은 GPT-4.1를 능가하는 성능을 보였으며, 특히 기존 모델들이 단순 검색은 잘하지만 복잡한 데이터 합성에는 취약하다는 점을 정량적으로 입증하였다. 이 연구는 향후 심층 연구 에이전트가 단순한 '검색기'를 넘어 '분석가'로서 기능하게 하는 데 중요한 기여를 할 것으로 보인다.
