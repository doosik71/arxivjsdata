# AutoML-Agent: Full-Pipeline AutoML을 위한 멀티 에이전트 LLM 프레임워크

Patara Trirat, Wonyong Jeong, Sung Ju Hwang

## 🧩 Problem to Solve

자동화된 머신러닝(AutoML)은 AI 개발 파이프라인의 여러 작업을 자동화하지만, 기존 시스템은 복잡한 도구 설정에 기술적 전문성을 요구하여 비전문가에게 장벽이 됩니다. 최근 LLM(Large Language Model) 기반 AutoML 프레임워크는 자연어 인터페이스를 제공하지만, 대부분 AI 개발 파이프라인의 특정 프로세스(예: 특징 엔지니어링, 하이퍼파라미터 최적화)나 특정 태스크(예: NLP, 컴퓨터 비전)에만 초점을 맞추어 LLM의 내재된 탐색 능력을 비효율적으로 사용합니다.

이 논문은 다음과 같은 주요 문제들을 해결하고자 합니다:

* **복잡한 전체 파이프라인 계획의 어려움**: 데이터 검색부터 모델 배포까지 전체 AutoML 파이프라인은 단계 간의 상호 의존성 때문에 계획이 매우 복잡하며, 다양한 다운스트림 태스크에 걸쳐 광범위한 탐색 공간을 가집니다.
* **정확한 구현의 어려움**: LLM을 사용하여 완전한 ML 파이프라인을 자율적으로 생성할 때 코드 불완전성, 잘못된 의존성, 미발견 버그와 같은 환각(hallucination) 문제가 발생할 수 있습니다. 모호한 태스크 설명은 코드 생성 정확도를 더욱 저해합니다.

## ✨ Key Contributions

* 데이터 검색부터 모델 배포까지 전체 AI 개발 파이프라인을 자동화하도록 설계된 새로운 LLM 기반 멀티 에이전트 프레임워크인 **AutoML-Agent**를 제안합니다. 이는 태스크에 구애받지 않는(task-agnostic) AutoML에서 LLM을 활용한 최초의 시도입니다.
* 전체 파이프라인 AutoML의 복잡한 계획 문제를 해결하기 위해 **검색 증강 계획(Retrieval-Augmented Planning, RAP)**과 역할별 계획 분해(role-specific plan decomposition), 프롬프트 기반 계획 실행(prompting-based plan execution)을 도입하여 탐색 과정의 유연성과 효율성을 향상합니다.
* 전체 파이프라인 구현의 정확도를 높이기 위해 **구조 기반 프롬프트 파싱(structure-based prompt parsing)**과 **다단계 검증(multi-stage verification)**을 통합하여 실제 코드 구현 전에 솔루션과 지시 사항의 품질을 보장하고 전반적인 성능을 개선합니다.
* 7가지 다운스트림 태스크와 14개 데이터셋에 대한 광범위한 실험을 통해 제안된 `AutoML-Agent` 프레임워크의 우수성을 입증합니다.
* 소스 코드를 [https://github.com/deepauto-ai/automl-agent](https://github.com/deepauto-ai/automl-agent)에 공개합니다.

## 📎 Related Works

* **전통적인 AutoML 시스템**: 특징 엔지니어링, 모델 선택, 하이퍼파라미터 최적화 등 특정 ML 파이프라인 요소에 집중하며, 복잡한 구성으로 인해 비전문가 접근이 어렵습니다 (예: Auto-Sklearn, AutoGluon).
* **LLM 기반 ML/데이터 과학 프레임워크**: 자연어 인터페이스를 통해 ML 및 데이터 과학 작업을 지원하지만, 대부분 특정 파이프라인 단계(예: 특징 엔지니어링, HPO, 모델 검색) 또는 특정 다운스트림 태스크(예: NLP, CV)에 한정됩니다. 또한, LLM의 내재된 탐색 능력을 간과하여 비효율적인 검색 프로세스를 가집니다 (예: AutoML-GPT, Prompt2Model, HuggingGPT, DS-Agent, SELA, Agent K).

`AutoML-Agent`는 기존 LLM 기반 프레임워크가 놓쳤던 **전체 파이프라인 지원**, **태스크 불가지론(task-agnosticism)**, **훈련 없는 검색(training-free search)**, 그리고 포괄적인 **계획 및 검증(planning and verification)** 기능을 외부 지식 **검색(retrieval)**과 함께 제공하여 차별화됩니다.

## 🛠️ Methodology

`AutoML-Agent`는 사용자 태스크 설명(`I`)을 받아 여러 전문 LLM 에이전트 간의 협업을 통해 최적의 ML 파이프라인을 식별하고, 배포 준비가 된 모델을 제공하는 멀티 에이전트 프레임워크입니다.

* **에이전트 구성**:
  * **Agent Manager ($\text{A}_{\text{mgr}}$)**: 사용자 및 다른 에이전트 간의 핵심 인터페이스 역할을 하며, 검색 프로세스를 조율하고, 전역 계획을 수립하며, 태스크를 분배하고, 결과를 검증합니다.
  * **Prompt Agent ($\text{A}_{\text{p}}$)**: 사용자 지시를 표준화된 JSON 객체로 파싱하도록 fine-tuning된 LLM입니다.
  * **Data Agent ($\text{A}_{\text{d}}$)**: 데이터 조작 및 분석(검색, 전처리, 증강, 특성 분석)을 담당하는 LLM입니다.
  * **Model Agent ($\text{A}_{\text{m}}$)**: 모델 검색, HPO, 모델 프로파일링, 후보 모델 랭킹과 같은 태스크를 수행하는 LLM입니다.
  * **Operation Agent ($\text{A}_{\text{o}}$)**: $\text{A}_{\text{d}}$와 $\text{A}_{\text{m}}$이 찾은 솔루션을 실제 코드로 구현하고 실행 결과를 기록하는 LLM입니다.

* **프레임워크 개요**:
    1. **초기화(Initialization)**: $\text{A}_{\text{mgr}}$이 사용자 지시(`I`)를 받고 **요청 검증(Request Verification, ReqVer)**을 통해 유효성을 확인합니다.
    2. **계획(Planning)**: $\text{A}_{\text{p}}$이 `I`를 파싱하여 표준 JSON 객체(`R`)로 만들고, $\text{A}_{\text{mgr}}$이 **검색 증강 계획(Retrieval-Augmented Planning, RAP)**을 사용하여 여러 파이프라인 계획(`P`)을 생성합니다.
    3. **실행(Execution)**: $\text{A}_{\text{d}}$와 $\text{A}_{\text{m}}$이 **계획 분해(Plan Decomposition, PD)**와 **프롬프트 기반 계획 실행(Prompting-Based Plan Execution)**을 통해 각 계획(`p_i`)을 실행합니다. 이 과정에는 **가상 데이터 분석(Pseudo Data Analysis)** 및 **훈련 없는 모델 검색 및 HPO(Training-Free Model Search and HPO)**가 포함됩니다. 결과(`O`)는 **실행 검증(Execution Verification, ExecVer)**을 통해 검증됩니다.
    4. **구현 및 배포**: $\text{A}_{\text{mgr}}$이 최적의 결과($\text{O}^{\star}$)를 선택하여 $\text{A}_{\text{o}}$에게 코드를 작성하도록 지시합니다. 코드 생성 후 **구현 검증(Implementation Verification, ImpVer)**을 통해 배포 준비 완료 여부를 확인합니다. 검증 실패 시 수정(revision) 단계로 전환됩니다.

* **주요 전략**:
  * **검색 증강 계획(RAP)**: `A_{mgr}`은 `R`과 외부 API (arXiv, Kaggle, 웹 검색 등)를 통해 검색된 최신 지식을 활용하여 $P$개의 상이한 엔드투엔드 계획을 생성합니다. 이는 탐색을 강화하고 병렬화를 가능하게 합니다.
  * **프롬프트 기반 계획 실행**:
    * **계획 분해(PD)**: 복잡한 계획 `p_i`를 에이전트의 역할과 전문 지식에 맞는 하위 태스크(`s^i_d`, `s^i_m`)로 분해하여 LLM의 효율성을 높입니다.
    * **가상 데이터 분석**: $\text{A}_{\text{d}}$는 실제 코드를 실행하지 않고, 데이터셋 특성과 사용자 요구 사항에 따라 데이터 검색, 전처리, 증강, 분석을 "실제로 수행하는 것처럼" 요약된 결과($\text{O}^{\text{d}}_{\text{i}}$)를 생성합니다.
    * **훈련 없는 모델 검색 및 HPO**: $\text{A}_{\text{m}}$은 $\text{A}_{\text{d}}$의 결과를 포함하여 $\text{A}_{\text{mgr}}$이 수집한 인사이트로 프롬프트가 강화됩니다. 이를 통해 실제 코드 실행 없이 모델 검색, HPO, 모델 프로파일링을 "실제로 수행하는 것처럼" 요약된 결과($\text{O}^{\text{m}}_{\text{i}}$)를 생성합니다.
  * **다단계 검증**: 정확성과 효율성을 보장하기 위해 3단계 검증을 사용합니다.
    * **요청 검증(ReqVer)**: 사용자 지시가 ML 태스크 수행에 충분하고 명확한지 초기 검증합니다.
    * **실행 검증(ExecVer)**: $\text{A}_{\text{d}}$와 $\text{A}_{\text{m}}$의 가상 실행 결과(`O`)가 사용자 요구 사항을 충족하는지 검증합니다.
    * **구현 검증(ImpVer)**: $\text{A}_{\text{o}}$가 생성하고 실행한 코드의 실제 결과가 사용자 요구 사항을 만족하는지 검증합니다.

## 📊 Results

* **실험 설정**: 이미지, 텍스트, 표, 그래프, 시계열 등 5가지 데이터 양식의 7개 다운스트림 태스크와 14개 데이터셋을 사용했습니다. 제약 조건 유무에 따른 두 가지 설정에서 실험했으며, 성공률(SR), 정규화된 성능 점수(NPS), 종합 점수(CS = $0.5 \times \text{SR} + 0.5 \times \text{NPS}$)를 평가 지표로 사용했습니다. Human Models, AutoGluon, GPT-3.5, GPT-4 (zero-shot), DS-Agent, SELA를 비교 대상으로 삼았습니다. 백본 LLM으로는 GPT-4o를 주로 사용했으며, $\text{A}_{\text{p}}$는 Mixtral-8x7B를 instruction-tuning하여 사용했습니다.

* **주요 결과**:
  * **성공률(SR)**: `AutoML-Agent`는 제약 조건이 있는 설정에서 평균 SR 87.1%를 달성하여 모든 기준선 대비 일관되게 뛰어난 성능을 보였습니다. 이는 계획 과정에서 검색된 지식이 제약 조건을 충족하는 데 도움이 되었음을 시사합니다.
  * **다운스트림 성능(NPS)**: `AutoML-Agent`는 제약 조건이 있는 설정에서 모든 태스크에 걸쳐 최상의 성능을 달성했으며, 심지어 Human Models보다도 우수했습니다. 이는 RAP 전략이 다양한 시나리오에 효과적으로 적응하며, 주어진 제약 조건에 맞는 파이프라인을 발견하는 데 뛰어남을 나타냅니다.
  * **종합 점수(CS)**: `AutoML-Agent`는 모든 기준선을 능가했으며, 특히 복잡한 태스크에서 강세를 보였습니다. 일반 목적 LLM은 고전적인 표 데이터 태스크에서 상대적으로 잘 작동하지만, `AutoML-Agent`와 같은 정교한 방법은 복잡한 태스크에서 훨씬 더 뛰어난 성능을 발휘했습니다.

* **추가 분석**:
  * **Ablation Study**: `AutoML-Agent`의 각 구성 요소(RAP, Plan Decomposition, Multi-Stage Verification)는 개별적으로 성능 하락을 보였으며, 모든 구성 요소를 통합했을 때 가장 높은 CS를 달성하며 LLM 에이전트가 외부 지식을 효과적으로 활용하여 완전한 AutoML 시스템을 구축하는 능력을 강화함을 입증했습니다.
  * **Hyperparameter Study (계획 수 $P$)**: 계획 수가 SR에 미치는 영향은 미미했지만, NPS와 CS에는 상당한 영향을 미쳤습니다. $P=3$일 때 최적의 성능을 보였으며, 계획 수를 무한정 늘리는 것이 반드시 더 나은 결과를 가져오지는 않았습니다.
  * **Prompt Sensitivity & Noise Robustness**: 에이전트의 역할이 명확하게 정의되는 한, 프롬프트 문구의 변화에 대해 견고한 성능을 보였습니다. 또한, 내장된 오류 수정 및 다단계 검증 메커니즘 덕분에 외부 지식 소스에서 발생할 수 있는 노이즈에 대해서도 견고했습니다.
  * **Training-Based Search (SELA)와 비교**: `AutoML-Agent`는 SELA보다 약 8배 빠른 검색 시간을 달성하면서도 유사하거나 더 우수한 성능을 유지했습니다 (평균 CS 0.612 대 0.599). 이는 훈련 기반 검색의 높은 계산 비용을 피하면서도 실용적인 적용 가능성에 초점을 맞춘 우리의 접근 방식의 효율성을 입증합니다.
  * **자원 비용**: 단일 모델 검색 및 배포에 평균 525초와 약 0.30 USD (GPT-4o 사용)가 소요되었습니다. 계획 단계에서 상당한 시간이 소요되는 것은 전체 파이프라인 AutoML 계획의 어려움을 시사합니다.

## 🧠 Insights & Discussion

* **함의**: `AutoML-Agent`는 AI 기반 혁신을 촉진하고 AI 전문 지식이 제한된 개인도 AI 역량을 효과적으로 활용할 수 있게 함으로써 상당한 이점을 제공할 것으로 기대됩니다.
* **제한 사항**: 완전히 새로운 태스크에 대한 스켈레톤 코드 부재는 코드 환각의 위험을 증가시킬 수 있습니다. 또한, 다른 백본 모델을 사용할 때 코드 생성 품질에 상당한 차이가 존재합니다. 미래 연구는 특정 LLM 백본에 덜 의존하는 견고한 프레임워크 개발에 초점을 맞춰야 합니다.
* **미래 연구**: 계획 과정에서 문헌 검색 구성 요소를 개선하기 위해 PaperQA와 같은 도구를 통합하는 것이 유망합니다. 또한, `AutoML-Agent`를 강화 학습 및 추천 시스템과 같이 개발 파이프라인이 크게 다른 ML 태스크로 확장하려면 추가 에이전트 개발이 필요할 것입니다.
* **영향 진술**: `AutoML-Agent`는 AI 접근성을 민주화하는 긍정적인 사회적 영향을 미치지만, 악의적인 사용자(예: 불쾌한 콘텐츠, 악성 소프트웨어 생성)에 의한 오용 가능성과 외부 API 통합으로 인한 잠재적인 프라이버시 문제를 인지하고 있습니다. 호스트 파일 시스템으로부터의 격리를 위해 Docker 컨테이너 내에서 실행하고, API 프롬프트에 포함된 데이터를 신중하게 검토하여 의도치 않은 데이터 노출을 방지할 것을 권장합니다.

## 📌 TL;DR

**문제**: 기존 AutoML은 전문 지식을 요구하고, LLM 기반 AutoML은 전체 파이프라인 대신 특정 단계에만 초점을 맞추어 복잡한 계획 및 정확한 구현에 어려움이 있습니다.

**제안 방법**: `AutoML-Agent`는 데이터 검색부터 모델 배포까지 전체 파이프라인을 자동화하는 LLM 기반 멀티 에이전트 프레임워크입니다. 이 프레임워크는 `Retrieval-Augmented Planning (RAP)`과 역할별 `Plan Decomposition`, `Prompting-Based Plan Execution`을 통해 효율적인 탐색을 가능하게 합니다. 또한, `Structure-Based Prompt Parsing`과 `Multi-Stage Verification` (요청, 실행, 구현)을 통해 구현의 정확성과 결과의 품질을 보장합니다.

**주요 결과**: 7가지 ML 작업과 14개 데이터셋에 대한 광범위한 실험에서 `AutoML-Agent`는 성공률(SR), 정규화된 성능 점수(NPS) 및 종합 점수(CS) 측면에서 기존 방법을 능가합니다. 특히 복잡한 제약 조건이 있는 시나리오에서 탁월한 성능을 보이며, 훈련 기반 탐색 방식보다 약 8배 빠른 검색 시간을 달성하면서도 유사하거나 더 나은 성능을 유지합니다.
