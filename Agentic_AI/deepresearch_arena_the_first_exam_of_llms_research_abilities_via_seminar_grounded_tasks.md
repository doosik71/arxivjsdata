# DeepResearch Arena: The First Exam of LLMs’ Research Abilities via Seminar-Grounded Tasks

Haiyuan Wan, Chen Yang, Junchi Yu, Meiqi Tu, Jiaxuan Lu, Di Yu, Jianbao Cao, Ben Gao, Jiaqing Xie, Aoran Wang, Wenlong Zhang, Philip Torr, Dongzhan Zhou (2025)

## 🧩 Problem to Solve

최근 LLM 기반의 Deep Research Agent들이 문헌 합성, 방법론 설계, 실험 검증과 같은 다단계 연구 워크플로우를 자동화하며 주목받고 있다. 그러나 이러한 에이전트들의 실제 연구 능력을 충실하게 평가하는 것은 매우 어려운 과제이다. 그 이유는 연구자의 지적 호기심과 관심을 실제로 자극하는 '최첨단(frontier)' 연구 질문을 수집하는 것이 어렵기 때문이다.

기존의 벤치마크들은 크게 두 가지 방식으로 연구 질문을 획득한다. 첫째는 학술 문헌이나 웹 콘텐츠와 같은 정적 코퍼스(static corpora)를 활용하는 방식인데, 이는 모델의 학습 데이터에 이미 포함되어 있을 가능성이 커 데이터 누수(data leakage) 위험이 높다. 둘째는 도메인 전문가가 직접 연구 과제를 설계하는 방식이나, 이는 확장성(scalability)이 부족하고 실제 연구 현장에서 발생하는 역동성과 다양성을 담아내지 못한다는 한계가 있다.

결과적으로, 실제 연구는 담론과 모호함, 학제 간 탐색을 통해 역동적으로 진화하는 '비구조적 문제 해결(Ill-Structured Problem Solving)'의 특성을 갖는다. 본 논문의 목표는 이러한 실제 연구 환경을 반영하여, 데이터 누수 위험을 최소화하면서도 인지적으로 요구 수준이 높은 정교한 연구 능력을 평가할 수 있는 벤치마크인 **DeepResearch Arena**를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 실제 연구자들이 오픈 엔디드(open-ended) 질문을 던지고 불확실한 아이디어를 탐색하는 **학술 세미나(Academic Seminars)**를 데이터 소스로 활용하는 것이다. 세미나 영상은 일반적인 웹 코퍼스보다 LLM의 사전 학습 데이터에 포함될 확률이 낮아 데이터 누수 위험을 줄일 수 있으며, 전문가 간의 상호작용을 통해 연구 문제가 자연스럽게 도출되는 과정을 포착할 수 있다.

주요 기여 사항은 다음과 같다.

1. **세미나 기반 데이터 수집**: 12개 학문 분야에 걸쳐 200개 이상의 학술 세미나 코퍼스를 구축하였다.
2. **계층적 과제 생성 프레임워크 (MAHTG)**: 세미나 스크립트에서 '연구 가치가 있는 영감(inspiration)'을 추출하고, 이를 다시 구체적인 연구 과제로 변환하는 Multi-Agent Hierarchical Task Generation 시스템을 제안하였다.
3. **하이브리드 평가 체계**: 사실적 근거를 측정하는 KAE(Keypoint-Aligned Evaluation)와 고차원적 추론 능력을 측정하는 ACE(Adaptively-generated Checklist Evaluation)를 결합하여 정량적·정성적 평가를 동시에 수행한다.

## 📎 Related Works

기존의 Deep Research Agent 벤치마크는 크게 두 갈래로 나뉜다.

- **정적 코퍼스 기반**: AcademicBrowse, BrowseComp, ResearchBench 등이 있으며, 주로 다단계 추론(multi-hop reasoning)이나 단순 과학 질의를 다룬다. 하지만 이는 정해진 논리 경로를 테스트하는 수준에 그쳐, 연구 질문이 어떻게 생성되고 반복되는지에 대한 실제 연구 프로세스를 반영하지 못한다.
- **전문가 큐레이션 기반**: Humanity’s Last Exam, DeepResearchBench, ExpertLongBench 등이 있으며, PhD 수준의 고난도 과제를 제공한다. 그러나 수동 구축 방식의 특성상 데이터셋의 규모가 작고 확장성이 떨어지며, 담론을 통해 문제가 구체화되는 실제 연구의 역동성을 포착하지 못한다.

DeepResearch Arena는 세미나 담론(Seminar Discourse)을 활용함으로써 확장성, 자동화, 데이터 누수 방지, 그리고 연구 리얼리즘이라는 네 가지 측면에서 기존 벤치마크들의 한계를 동시에 극복하고자 한다.

## 🛠️ Methodology

### 1. Multi-Agent Hierarchical Task Generation (MAHTG)

MAHTG 시스템은 원본 세미나 영상에서 텍스트 스크립트를 추출한 후, 다음의 단계를 거쳐 연구 과제를 생성한다.

**가. 영감 추출 (Inspiration Extraction)**
`Inspira Agent`가 스크립트에서 학술적 가치가 있는 '영감'을 추출한다. 이때 다음 네 가지 기준 중 최소 두 가지를 만족해야 한다:

- **Novelty (신규성)**: 새로운 아이디어나 관점을 제시하는가?
- **Explorability (탐색 가능성)**: 추가적인 모델링이나 실험의 시작점이 되는가?
- **Challenge (도전성)**: 기존의 한계나 병목 현상을 드러내는가?
- **Verifiability (검증 가능성)**: 데이터나 시뮬레이션으로 확인 가능한가?

추출된 영감은 $\text{Limitation}$, $\text{Methodology}$, $\text{Transdisciplinarity}$, $\text{Hypothesis}$의 네 가지 유형으로 분류된다.

**나. 과제 생성 및 필터링 (Task Generation & Filtering)**
`TaskWeaver Agent`가 추출된 영감들을 종합하여 연구 워크플로우의 세 단계($\text{Synthesize} \rightarrow \text{Design} \rightarrow \text{Evaluate}$)에 맞춘 구체적인 연구 과제를 설계한다. 이후 `RankEval Agent`가 Elo 레이팅 시스템을 사용하여 과제의 품질을 평가하고 상위 과제만을 선별한다.

Elo 레이팅의 기대 승률 $e_a$는 다음과 같이 계산된다:
$$e_a = \frac{1}{1 + 10^{(r_b-r_a)/400}}$$
여기서 $r_a, r_b$는 각각 과제 $a, b$의 현재 점수이다. 실제 결과 $s_a$와 기대 승률 $e_a$의 차이를 바탕으로 점수를 업데이트한다:
$$r'_a = r_a + K \cdot (s_a - e_a)$$
($K=32$로 설정)

### 2. 하이브리드 평가 프레임워크

**가. Keypoint-Aligned Evaluation (KAE)**
모델이 생성한 보고서 $R$ 내의 인용 URL들로부터 사실적 핵심 포인트(keypoints)를 추출하여 정량적으로 평가한다.

- **KSR (Keypoint Supported Rate)**: 전체 핵심 포인트 중 보고서가 명시적으로 지지하는 비율.
- **KCR (Keypoint Conflict Rate)**: 핵심 포인트와 보고서의 내용이 충돌하는 비율.
- **KOR (Keypoint Omission Rate)**: 보고서가 누락한 핵심 포인트의 비율.

**나. Adaptively-generated Checklist Evaluation (ACE)**
정답이 없는 오픈 엔디드 과제를 평가하기 위해 LLM 기반의 적응형 체크리스트를 사용한다.

1. **체크리스트 생성**: 고성능 LLM(예: GPT-4o)이 과제 프롬프트를 분석하여 맞춤형 평가 기준(rubric)과 가중치를 생성한다.
2. **점수 산정**: 별도의 LLM 평가자가 생성된 체크리스트를 바탕으로 응답을 개별 평가하고, 가중 평균을 통해 최종 점수를 도출한다.

## 📊 Results

### 실험 설정

- **평가 모델**: `gpt-4o`, `gpt-4.1`, `o4-mini-deepresearch`, `gemini-2.5-pro`, `gemini-2.5-flash`, `grok-4` 등 최신 Deep Research Agent 및 검색 증강 모델들을 대상으로 하였다.
- **데이터셋**: 12개 학문 분야의 세미나에서 추출된 10,000개 이상의 과제 중 상위 100개 샘플을 집중 평가하였다.

### 주요 결과

1. **모델별 성능 차이**:
   - `o4-mini-deepresearch` 모델이 ACE 점수 4.03으로 가장 높은 정성적 품질을 보였으며, KAE 지표에서도 강력한 성능을 나타냈다.
   - `gpt-4.1`은 KCR(충돌률)이 가장 낮아 사실적 정확도는 매우 높았으나, ACE 점수는 낮아 깊이와 일관성 면에서는 부족함을 보였다.
   - `gemini-2.5-flash`는 사실적 커버리지가 높았으나, 타 모델 대비 토큰 사용량이 압도적으로 많아 효율성 측면의 트레이드오프가 관찰되었다.
   - `grok-4`는 영어 과제에서는 매우 높은 KSR(83.3%)을 기록했으나, 중국어 과제에서는 성능이 급격히 하락하여 다국어 일반화 능력이 제한적임을 보였다.

2. **과제 유형별 분석**:
   - `o4-mini-deep-research`와 `gemini-2.5-flash`는 가설 생성($\text{Hypothesis Generation}$), 평가 지표 설계($\text{Evaluation Metric Design}$)와 같은 고차원적 설계 과제에서 특히 강점을 보였다. 반면 `gpt-4o-mini` 계열은 다단계 논리와 구조적 출력이 필요한 과제에서 고전하는 경향을 보였다.

3. **데이터 누수 검증**:
   - 텍스트 복원 실험(Task Prefix를 주었을 때 Suffix를 맞추는 실험)을 수행한 결과, 모든 모델의 복합 유사도 점수가 임계치($\tau=0.7$)보다 훨씬 낮은 수준으로 나타나, 본 벤치마크가 사전 학습 데이터로부터 오염되지 않았음을 확인하였다.

4. **인간 평가와의 일치도**:
   - KAE 및 ACE의 자동 평가 점수와 인간 전문가의 점수 간의 Spearman 상관계수가 각각 0.84, 0.81로 매우 높게 나타나, 제안된 평가 체계의 신뢰성을 입증하였다.

## 🧠 Insights & Discussion

**강점 및 의의**
본 연구는 정적인 텍스트 데이터에서 벗어나 '세미나 담론'이라는 역동적인 소스를 활용함으로써, 실제 연구 과정에서 문제가 정의되는 방식을 벤치마크에 성공적으로 이식하였다. 특히 MAHTG 시스템을 통해 대규모의 고품질 연구 과제를 자동 생성하면서도, Elo 레이팅을 통해 품질을 관리한 점이 돋보인다. 또한, 단순히 정답 일치 여부를 따지는 것이 아니라, 사실적 근거(KAE)와 방법론적 타당성(ACE)을 분리하여 평가함으로써 Deep Research Agent의 능력을 다각도로 분석할 수 있게 하였다.

**한계 및 논의사항**

- **평가 모델 의존성**: ACE 평가의 경우 LLM-as-a-judge 방식에 의존하고 있다. 비록 체크리스트를 통해 편향을 줄이려 노력했으나, 평가 모델 자체의 내재적 편향이 결과에 영향을 줄 가능성이 있다.
- **토큰 효율성**: Gemini 모델의 사례에서 보듯, 높은 성능이 단순히 방대한 양의 텍스트 생성(verbose outputs)에서 기인하는 것인지, 아니면 실제 추론의 깊이가 깊은 것인지에 대한 더 정밀한 분석이 필요하다.
- **실제 실행 가능성**: 생성된 과제들이 '설계' 수준에서는 훌륭하지만, 실제 실험 환경에서 실행했을 때의 결과까지 검증하는 폐루프(closed-loop) 평가 체계로의 확장이 향후 과제가 될 것이다.

## 📌 TL;DR

본 논문은 실제 학술 세미나의 담론을 기반으로 LLM의 연구 능력을 평가하는 벤치마크 **DeepResearch Arena**를 제안한다. MAHTG 시스템을 통해 12개 분야의 10,000개 이상의 고난도 연구 과제를 자동으로 생성하였으며, 사실성(KAE)과 추론능력(ACE)을 동시에 측정하는 하이브리드 평가 체계를 구축하였다. 실험 결과, 최신 에이전트들 사이에서도 성능 격차가 뚜렷하게 나타났으며, 특히 `o4-mini-deep-research`가 종합적인 연구 수행 능력에서 우위를 보였다. 이 연구는 향후 실제 연구 프로세스를 보조하는 차세대 AI 연구 조수 개발을 위한 엄격한 평가 기준을 제공한다.
