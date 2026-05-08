# A Survey of AIOps in the Era of Large Language Models

Lingzhe Zhang, Tong Jia, Mengxi Jia, Yifan Wu, Aiwei Liu, Yong Yang, Zhonghai Wu, Xuming Hu, Philip S. Yu, Ying Li (2025)

## 🧩 Problem to Solve

현대 소프트웨어 시스템은 규모가 거대해지고 구조가 복잡해짐에 따라, 아주 작은 결함만으로도 막대한 경제적 손실과 평판 저하를 초래할 수 있다. 따라서 대규모 분산 시스템의 가용성과 신뢰성을 유지하기 위해 장애를 신속하게 탐지하고, 진단하며, 근본 원인을 파악하여 해결하는 AIOps(Artificial Intelligence for IT Operations)의 중요성이 증대되고 있다.

기존의 머신러닝(ML) 및 딥러닝(DL) 기반 AIOps 접근 방식은 다음과 같은 한계점을 가진다:

- **복잡한 특징 추출(Feature Extraction) 공학 필요**: 비정형 데이터(로그, 트레이스 등)를 처리하기 위해 광범위한 전처리와 수동 피처 엔지니어링이 필요하다.
- **교차 플랫폼 범용성 부족**: 특정 시스템에 최적화되어 학습되므로, 시스템 환경이 조금만 변경되어도 성능이 급격히 저하된다.
- **교차 작업 유연성 결여**: 모델이 단일 작업만 수행할 수 있어, 실제 환경에서는 여러 모델을 동시에 실행해야 하는 비효율성이 존재한다.
- **제한적인 모델 적응력**: 시스템 변경 시 빈번한 재학습이 필요하며, 이는 갑작스러운 이벤트 대응에 지연을 초래한다.
- **낮은 자동화 수준**: 단순한 티켓 분류나 솔루션 추천 수준에 머물러 있으며, 실제 조치(Remediation)까지의 자동화 수준이 낮다.

본 논문의 목표는 Large Language Models(LLMs)가 이러한 한계를 어떻게 극복하고 AIOps의 프로세스를 최적화하며 성과를 향상시킬 수 있는지에 대해 체계적인 분석 보고서를 제공하는 것이다.

## ✨ Key Contributions

본 논문은 LLM 기반 AIOps의 영향과 잠재력, 한계를 이해하기 위해 2020년 1월부터 2024년 12월까지 발표된 183편의 연구 논문을 분석하였다. 핵심적인 기여는 다음과 같다:

- **LLM4AIOps의 포괄적 프레임워크 제시**: 데이터 전처리, 장애 인지(Failure Perception), 근본 원인 분석(Root Cause Analysis), 자동 복구(Auto Remediation)로 이어지는 AIOps의 전 과정을 LLM 관점에서 체계화하였다.
- **데이터 소스의 확장 분석**: 기존의 시스템 생성 데이터(Metrics, Logs, Traces)뿐만 아니라 LLM을 통해 활용 가능해진 인간 생성 데이터(Software Information, QA, Incident Reports)의 역할을 정의하였다.
- **AIOps 작업의 진화 정의**: LLM의 등장으로 인해 새롭게 생성된 작업(예: Root Cause Report Generation, Script Generation 등)과 기존 작업의 변화 양상을 분석하였다.
- **LLM 기반 방법론의 분류**: Foundation Model, Fine-tuning, Embedding-based, Prompt-based, Knowledge-based 접근 방식으로 방법론을 분류하고 각각의 특성을 분석하였다.
- **평가 방법론의 현대화**: 생성 작업(Generation Task)과 실행 작업(Execution Task)의 증가에 따른 새로운 평가 지표와 데이터셋을 정리하였다.

## 📎 Related Works

논문은 기존의 AIOps 관련 서베이 연구들을 검토하며 본 연구와의 차별점을 제시한다.

- **기존 서베이의 한계**: Paolop et al. [97], Qian et al. [17], Youcef et al. [108], Wei et al. [134] 등의 연구는 주로 전통적인 ML/DL 알고리즘에 기반하고 있으며, LLM 기반 접근 방식을 다루지 않는다.
- **범위의 제한성**: Jing et al. [121]의 연구는 LLM을 다루고는 있으나, AIOps 전체 프로세스에 대한 체계적인 요약을 제공하지 않고 특정 작업(시계열 예측, 이상 탐지)에 국한되어 있다.

본 논문은 LLM을 중심으로 데이터 전처리부터 자동 복구까지의 **전체 파이프라인을 포괄적으로 다루는 최초의 서베이**라는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 체계적 문헌 고찰(Systematic Review) 프로세스를 따르며, 4가지 핵심 연구 질문(RQ)을 중심으로 분석을 수행한다.

### 1. 데이터 변환 (RQ1)

- **전통적 데이터 처리**: LLM을 사용하여 Metrics의 결측치 보간(Imputation), Traces의 합성(Synthesis), 특히 Logs의 파싱(Parsing) 성능을 향상시키는 기법들을 분석한다.
- **새로운 데이터 소스**: 소프트웨어 아키텍처, 설정 파일, 소스 코드, Q&A 데이터, 사고 보고서(Incident Reports)와 같은 인간 생성 데이터를 LLM이 어떻게 의미적으로 이해하고 활용하는지 설명한다.

### 2. 작업의 진화 (RQ2)

AIOps 작업을 세 단계의 파이프라인으로 구분한다:

- **Failure Perception**: 장애 예방, 예측, 이상 탐지를 포함하며, 최근에는 모델의 범용성 향상과 zero-shot 예측에 집중하고 있다.
- **Root Cause Analysis (RCA)**: 장애 위치 파악(Localization)과 카테고리 분류에서 나아가, LLM의 생성 능력을 이용한 **근본 원인 보고서 생성(Root Cause Report Generation)**이라는 새로운 작업이 등장하였다.
- **Assisted Remediation**: 자동화 수준에 따라 다음과 같이 분류한다:
  $$\text{Assisted Questioning} \rightarrow \text{Solution Generation} \rightarrow \text{Command Recommendation} \rightarrow \text{Script Generation} \rightarrow \text{Automatic Execution}$$

### 3. LLM 기반 방법론 (RQ3)

- **Foundation Model**: Transformer 기반의 Encoder-only, Decoder-only, Encoder-Decoder 구조의 활용 사례를 분석한다.
- **Fine-tuning**: 전체 파라미터 튜닝(Full Fine-Tuning)과 PEFT(Parameter-Efficient Fine-Tuning, 예: LoRA, Adapters) 및 Instruction Tuning의 적용 방식을 다룬다.
- **Embedding-based**: 사전 학습된 임베딩을 직접 사용하거나, Prompt Embedding을 통해 시계열 데이터를 LLM이 이해 가능한 형태로 변환하는 기법을 설명한다.
- **Prompt-based**: In-Context Learning(ICL), Chain-of-Thought(CoT), Task Instruction Prompting을 통해 모델의 추론 능력을 극대화하는 방법을 분석한다.
- **Knowledge-based**: 외부 지식을 검색하여 결합하는 RAG(Retrieval-Augmented Generation)와 외부 API/도구를 호출하는 TAG(Tool-Augmented Generation) 기법을 설명한다.

### 4. 평가 방법론 (RQ4)

- **지표(Metrics)**:
  - 분류 작업: Precision, Recall, F1-score, MAE, RMSE 등.
  - 생성 작업: BLEU, ROUGE(Lexical), BERTScore(Semantic) 등.
  - 실행 작업: Functional Correctness(FC), Execution Success Rate 등.
  - 수동 평가: 전문가에 의한 정성적 평가 및 Human Preferences.
- **데이터셋**: LogEval, OpsEval, OWL-bench, KubePlaybook, AIOpsLab 등 LLM 전용 벤치마크를 분석한다.

## 📊 Results

본 논문은 정량적 수치보다는 체계적인 분류와 분석 결과를 제시한다.

- **출판 트렌드**: ChatGPT 출시 이후 LLM 기반 AIOps 연구가 급격히 증가하는 추세를 보였다.
- **작업의 확장**: 전통적인 AIOps가 '탐지'와 '분류'에 집중했다면, LLM 시대의 AIOps는 '보고서 작성'과 '복구 스크립트 생성 및 실행'이라는 고차원적 자동화 단계로 진입하였다.
- **방법론적 경향**: 단순한 Prompting에서 시작하여, 현재는 RAG와 TAG를 결합하여 도메인 지식을 보완하고 외부 도구를 직접 제어하는 AI Agent 형태로 발전하고 있다.
- **데이터 활용**: 시스템 생성 데이터(로그, 메트릭)와 인간 생성 데이터(코드, 문서)를 함께 사용하는 멀티모달적 접근이 RCA 성능을 유의미하게 향상시킨다는 점이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 가능성

LLM은 비정형 데이터에 대한 강력한 이해력과 추론 능력을 바탕으로, 기존 AIOps의 최대 난제였던 '범용성'과 '자동화 수준'을 획기적으로 높일 수 있는 잠재력을 가지고 있다. 특히 RAG와 TAG의 결합은 모델의 환각(Hallucination) 문제를 완화하고 실제 운영 환경에서의 실행 가능성을 높인다.

### 한계 및 해결 과제

- **시간 효율성 및 비용**: LLM의 추론 비용과 지연 시간(Latency)은 실시간으로 작동해야 하는 '장애 인지(Failure Perception)' 단계에서 치명적인 약점이 된다.
- **데이터 활용의 불균형**: 로그와 메트릭 데이터 활용은 활발하나, 복잡도가 높은 트레이스(Trace) 데이터의 LLM 통합 연구는 여전히 부족하다.
- **소프트웨어 진화에 따른 적응력**: 모델이 학습하지 않은 새로운 시스템 변경 사항에 대해 얼마나 유연하게 대응하는지에 대한 체계적인 검증이 부족하다.
- **기존 툴체인과의 통합**: LLM을 완전히 새로운 시스템으로 구축하기보다, 기존의 가벼운 모델(Small Models)이나 룰 기반 시스템과 어떻게 상호 보완적으로 통합할 것인가에 대한 논의가 필요하다.

## 📌 TL;DR

본 논문은 LLM이 AIOps의 전 과정(데이터 $\rightarrow$ 인지 $\rightarrow$ 분석 $\rightarrow$ 복구)에 가져온 변화를 분석한 포괄적인 서베이이다. LLM은 전통적인 AIOps의 한계인 피처 엔지니어링 의존성과 낮은 범용성을 극복하게 하며, 특히 **근본 원인 보고서 생성** 및 **자동 복구 스크립트 실행**과 같은 고차원적 자동화를 가능하게 한다. 향후 연구는 LLM의 높은 비용 문제를 해결하기 위한 **경량 모델과의 하이브리드 구조** 및 **트레이스 데이터의 효율적 통합** 방향으로 나아가야 한다.
