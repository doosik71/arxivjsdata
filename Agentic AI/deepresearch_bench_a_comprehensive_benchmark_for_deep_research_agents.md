# DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, Zhendong Mao

## 🧩 Problem to Solve

거대 언어 모델(LLM) 기반 에이전트의 한 유형인 심층 연구 에이전트(Deep Research Agents, DRAs)는 다단계 웹 탐색, 표적 검색, 고차원 합성 기능을 자율적으로 수행하여 방대한 온라인 정보를 분석가 수준의 인용이 풍부한 보고서로 변환함으로써 수작업 연구 시간을 크게 단축합니다. 그러나 이러한 DRA의 역량을 체계적으로 평가할 수 있는 **포괄적인 벤치마크가 부재**하다는 문제가 있습니다. 기존의 벤치마크들은 주로 웹 브라우징이나 정보 검색과 같은 고립된 기능이나 실시간 정보 획득과 무관한 생성 능력을 평가하는 데 초점을 맞추어, DRA의 다면적인 최종 보고서 품질을 평가하기에는 한계가 있습니다. 또한, 복잡한 연구 질문에 대한 명확한 '정답'을 설정하기 어렵다는 점도 평가의 어려움을 가중시킵니다.

## ✨ Key Contributions

* **DeepResearch Bench 제안**: 실제 사용자 요구를 반영한 체계적인 과정을 통해 구축된, 심층 연구 에이전트 평가를 위한 최초의 전문 벤치마크를 제시합니다. 이 벤치마크는 22개 분야에 걸쳐 100개의 박사 수준 연구 과제로 구성됩니다.
* **RACE 및 FACT 평가 프레임워크 개발**: DRA의 보고서 생성 품질을 평가하는 `RACE` (Reference-based and Adaptive Criteria-driven Evaluation framework with Dynamic Weighting)와 정보 검색 및 인용 능력(fact-grounding)을 평가하는 `FACT` (Factual Abundance and Citation Trustworthiness) 두 가지 새로운 평가 프레임워크를 제안합니다.
* **인간 일치도 검증 및 공개**: 제안된 프레임워크의 신뢰성을 검증하기 위한 포괄적인 인간 연구를 수행했으며, 향후 연구를 장려하기 위해 벤치마크와 평가 프로토콜을 공개합니다.

## 📎 Related Works

* **LLM 기반 에이전트 평가**: LLM 역량의 발전과 함께 과학, 창의적 글쓰기, 코드 생성, 인간 보조 역할 등 다양한 분야에서 에이전트 평가 벤치마크가 개발되었습니다. 이러한 배경은 실제 시나리오에 기반한 심층 연구 에이전트 전용 벤치마크와 인간 평가와 일치도가 높은 평가 방법 개발이 시급함을 강조합니다.
* **심층 연구 에이전트 관련 연구**: OpenAI와 Google Gemini의 DRA 출시 이후 관련 프레임워크가 빠르게 등장했으나, 이들의 역량을 비교 분석할 수 있는 표준화된 평가 방법론이 부족했습니다. 기존 연구들은 QA 데이터셋이나 '판사 LLM(LLM-as-a-Judge)' 방법론을 사용했지만, 포괄적인 프레임워크 설계나 인간 일치도 검증이 미흡했습니다. 본 연구는 이러한 격차를 해소합니다.

## 🛠️ Methodology

이 논문은 `DeepResearch Bench` 데이터셋과 두 가지 보완적인 평가 프레임워크인 `RACE` 및 `FACT`를 소개합니다.

* **DeepResearch Bench 구축**:
  * **주제 분포 분석**: 웹 검색이 가능한 LLM 챗봇의 96,147개 사용자 쿼리 로그를 익명화하여 수집. DeepSeek-V3-0324를 사용하여 심층 연구 요구 사항에 부합하는 44,019개 쿼리를 필터링. WebOrganizer의 22개 주제 분류법을 채택하여 쿼리를 분류하고 실제 사용자 수요 분포를 파악.
  * **벤치마크 과제 수집**: 실제 사용자 수요 분포에 따라 각 주제 도메인의 목표 과제 수를 결정. DRA 실행 및 평가에 필요한 자원을 고려하여, 이 분포를 비례적으로 압축하여 총 100개의 과제(중국어 50개, 영어 50개)로 최종 데이터셋 구성. 각 과제는 박사 학위 소지자 또는 5년 이상 경력의 전문가가 제안하고, 품질, 명확성, 복잡성, 심층 연구 정의와의 일치 여부를 수동으로 검증하여 고품질의 도전적인 과제를 선별.

* **RACE (Report Quality Evaluation) 프레임워크**: '판사 LLM(LLM-as-a-Judge)' 방식을 활용하여 보고서 생성 품질을 평가합니다.
  * **동적 가중치 및 적응형 기준 생성**:
    * `Comprehensiveness` (포괄성), `Insight/Depth` (통찰력/심층성), `Instruction-Following` (지시사항 준수), `Readability` (가독성)의 네 가지 상위 차원 정의.
    * 판사 LLM이 각 과제 $t$에 대해 여러 시도 $T$에서 얻은 각 차원 $d$에 대한 가중치 $w^{(j)}_d$를 평균하여 최종 차원 가중치 $W_d$를 동적으로 생성합니다.
    * $$W_d = \frac{1}{T} \sum_{j=1}^{T} w^{(j)}_d$$
    * 각 차원 $d$에 대해 판사 LLM이 $K_d$개의 맞춤형 기준 $\{c_{d,k}\}$와 해당 가중치 $\{w_{d,k}\}$를 생성(단, $\sum_{k=1}^{K_d} w_{d,k} = 1$).
  * **참조 기반 점수 부여**: 모델이 고르게 높은 점수를 부여하는 경향을 완화하기 위해, 각 과제 $t$에 대해 고품질 참조 보고서 $R_{ref}$를 선정. 판사 LLM이 대상 보고서 $R_{tgt}$와 참조 보고서 $R_{ref}$를 종합 기준 목록 $C_t$에 대해 비교하여 점수 $\{s_{tgt,c}\}_{c \in C_t}$와 $\{s_{ref,c}\}_{c \in C_t}$를 산출.
  * **전체 점수 계산**:
    * 차원별 점수 $S_d(R)$는 기준별 점수 $s_{R,c_{d,k}}$에 기준 가중치 $w_{d,k}$를 곱하여 계산.
    * 이 차원별 점수 $S_d(R)$에 차원 가중치 $W_d$를 곱하여 중간 전체 점수 $S_{int}(R)$를 산출.
    * 최종적으로 대상 보고서의 점수 $S_{final}(R_{tgt})$는 참조 보고서 $S_{int}(R_{ref})$에 대한 상대적 점수로 결정.
    * $$S_{final}(R_{tgt}) = \frac{S_{int}(R_{tgt})}{S_{int}(R_{tgt}) + S_{int}(R_{ref})}$$

* **FACT (Web Retrieval Evaluation) 프레임워크**: 보고서 내용의 사실적 근거와 웹 정보 검색 및 활용의 효과성을 평가합니다.
  * **진술-URL 쌍 추출 및 중복 제거**: 판사 LLM을 사용하여 보고서에서 개별 진술과 해당 인용 URL을 추출. 동일한 URL과 관련된 여러 진술 중 동일한 사실을 설명하는 경우 중복을 제거.
  * **지원 여부 판단**: 각 고유한 진술-URL 쌍에 대해 Jina Reader API로 웹페이지 내용을 검색. 판사 LLM이 웹페이지 내용이 진술을 충분히 뒷받침하는지 여부를 이진(`support` 또는 `not support`)으로 판단.
  * **인용 지표 계산**:
    * **인용 정확도 (C. Acc.)**: 에이전트 인용의 정확성을 측정하며, 적절한 소스로 진술의 근거를 올바르게 제시하는 능력을 반영합니다.
      * 단일 과제 $t$에 대한 정확도 $Acc_t$:
      * $$Acc_t = \begin{cases} \frac{N_{s,t}}{N_{u,t}} & \text{if } N_{u,t} > 0 \\ 0 & \text{if } N_{u,t} = 0 \end{cases}$$
      * 전체 인용 정확도 $C. Acc.$:
      * $$C. Acc. = \frac{1}{|T|} \sum_{t \in T} Acc_t$$
    * **과제당 평균 유효 인용 수 (E. Cit.)**: 에이전트가 과제당 검색하고 제시하는 가치 있고 검증 가능한 정보의 양을 정량화합니다.
      * $$E. Cit. = \frac{\sum_{t \in T} N_{s,t}}{|T|}$$

## 📊 Results

* **RACE 프레임워크 평가**:
  * DRA 범주에서 **Gemini-2.5-Pro Deep Research**가 전반적으로 가장 우수한 성능을 보였고, OpenAI Deep Research도 지시사항 준수(Instruction-Following) 차원에서 Gemini-2.5-Pro를 능가하는 등 강력한 성능을 나타냈습니다. 이들은 다른 두 DRA(Perplexity Deep Research, Grok Deeper Search)보다 훨씬 뛰어났습니다.
  * 검색 도구를 갖춘 LLM 중에서는 **Claude-3.7-Sonnet w/Search**가 인상적인 성능을 보여 Grok Deeper Search를 능가했으며, Perplexity-Sonar-Reasoning-Pro도 유사한 성능을 보였습니다.
  * 각 모델은 다양한 주제 및 언어(중국어, 영어)에 걸쳐 비교적 안정적인 성능을 유지하여 RACE 평가 프레임워크의 견고성을 입증했습니다.

* **FACT 프레임워크 평가**:
  * Grok을 제외한 DRA는 검색 도구를 갖춘 LLM보다 더 많은 유효 인용(Effective Citations)을 포함하는 경향이 있었고, 특히 **Gemini-2.5-Pro Deep Research**는 평균 111.21개의 유효 인용으로 다른 모델들을 크게 앞질렀습니다. 이는 RACE의 포괄성(Comprehensiveness) 차원에서의 최고 점수와 일치합니다.
  * 하지만 Gemini-2.5-Pro Deep Research와 OpenAI Deep Research의 인용 정확도(Citation Accuracy)는 Perplexity Deep Research에 비해 현저히 낮았는데, 이는 Perplexity Deep Research가 검색된 텍스트에서 관련 내용을 더 정확하게 회상하는 능력이 강하다는 것을 시사합니다.
  * 검색 도구를 갖춘 LLM 중 Claude-3.7-Sonnet w/Search는 두 번째로 높은 유효 인용 수와 강력한 인용 정확도를 달성했으며, 이는 RACE 프레임워크에서 가장 좋은 전반적인 점수를 받은 것과도 일치합니다.

* **인간 일치도 검증**:
  * **RACE(Full) 프레임워크**는 Pairwise Agreement Rate (71.33%), Overall Pearson Correlation (99.54%), Filtered Average Pearson (60.24%), Filtered Average Spearman (59.12%)에서 가장 강력한 전반적인 성능을 보여주었으며, 이는 Vanilla Prompt를 포함한 다른 변형 모델들을 크게 능가합니다.
  * 특히 RACE(Full)의 Pairwise Agreement Rate는 인간 전문가 간의 일치도(Human Inter-Agreement, 68.44%)까지 넘어섰습니다. 이는 RACE가 심층 연구 보고서를 효율적으로 신뢰성 높고 정확하게 평가하며 높은 인간 일치도를 달성할 수 있음을 입증합니다.
  * 판사 LLM 비교에서는 **Gemini 2.5 Pro Preview**가 최고의 전반적인 성능을 달성하면서도 경쟁력 있는 평균 비용($0.13/쿼리)을 유지하여 최종 프레임워크의 Judge LLM으로 선정되었습니다.

## 🧠 Insights & Discussion

* **DRA 평가의 중요성**: DeepResearch Bench는 DRA의 역량 개발을 위한 견고한 프레임워크를 제공함으로써 연구와 혁신을 가속화하고 다양한 분야의 사용자에게 고급 연구 방법론에 대한 접근성을 높일 수 있습니다.
* **RACE 및 FACT의 신뢰성**: 제안된 RACE 및 FACT 평가 프레임워크는 인간 평가와 높은 일치도를 보여주어, 심층 연구 보고서의 품질과 사실적 근거를 신뢰성 있게 측정할 수 있음을 입증했습니다.
* **한계점**:
  * **벤치마크 규모**: 100개의 과제는 고품질 유지를 위해 전문가들이 신중하게 큐레이션했지만, 통계적 견고성과 주제 포괄성을 높이기 위한 확장 필요성이 있습니다.
  * **도메인 커버리지 편향**: 큐레이션 과정에서 의도치 않은 편향이 있을 수 있으며, 향후 외부 전문가를 통한 추가 검토가 필요합니다.
  * **인간 평가 처리량**: 심층 연구 보고서 평가는 시간이 많이 소요되는 작업이므로, 더 광범위한 인간 평가 캠페인이 통계적 신뢰도를 높이고 자동화된 지표를 정교화하는 데 도움이 될 것입니다.
* **사회적 영향**: 강력한 DRA의 발전은 정교한 오정보 생성 가능성, 비판적 사고 및 수동 연구 기술 저하, 기존 모델 및 웹 데이터에 내재된 편향 증폭과 같은 사회적 문제를 야기할 수 있습니다. DeepResearch Bench는 FACT 및 RACE와 같은 다면적 평가를 통해 이러한 위험을 완화하고 보다 신뢰할 수 있고 투명하며 책임감 있는 AI 에이전트 개발을 촉진하는 데 기여할 것으로 기대됩니다.

## 📌 TL;DR

이 논문은 **심층 연구 에이전트(DRA)의 포괄적인 평가를 위한 최초의 벤치마크인 DeepResearch Bench**를 소개합니다. 실제 사용자 요구를 반영하여 22개 분야의 100개 박사 수준 연구 과제로 구성된 이 벤치마크는 DRA의 최종 보고서 품질과 웹 정보 검색 능력을 효과적으로 평가합니다. 이를 위해 **RACE (보고서 품질 평가)**와 **FACT (웹 검색 및 인용 신뢰성 평가)**라는 두 가지 새로운 LLM 기반 평가 프레임워크를 제안합니다. 실험 결과, 이 프레임워크들은 **높은 인간 일치도**를 보였으며, Gemini-2.5-Pro Deep Research와 OpenAI Deep Research가 보고서 품질에서 선도적인 성능을, Gemini-2.5-Pro Deep Research가 가장 많은 유효 인용 수를 기록했습니다. DeepResearch Bench는 미래의 DRA 개발을 안내하고 더욱 강력하고 인간 중심적인 AI 에이전트 시스템 구축에 기여할 것으로 기대됩니다.
