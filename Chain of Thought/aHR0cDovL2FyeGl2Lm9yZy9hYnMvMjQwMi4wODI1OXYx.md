# A Survey of Table Reasoning with Large Language Models

Xuanliang Zhang, Dingzirui Wang, Longxu Dou, Qingfu Zhu, Wanxiang Che (2024)

## 🧩 Problem to Solve

본 논문은 제공된 테이블(표)과 선택적으로 제공되는 텍스트 설명을 바탕으로 사용자의 요구사항에 맞는 정답을 생성하는 **Table Reasoning** 과제를 다룬다. Table Reasoning은 방대한 양의 테이블 데이터로부터 정보를 얻는 효율성을 극대화하는 핵심적인 기술이다.

최근 Large Language Models(LLMs)의 등장으로 인해 데이터 어노테이션 비용이 획기적으로 감소하고 기존 모델들을 뛰어넘는 성능이 달성되면서, Table Reasoning의 주류 방법론이 LLM 기반으로 빠르게 전환되었다. 그러나 LLM 시대의 Table Reasoning 연구를 체계적으로 정리한 요약 분석이 부족하여, 어떤 기술이 성능을 향상시키는지, 왜 LLM이 이 작업에 뛰어난지, 그리고 향후 어떻게 능력을 강화할 수 있는지에 대한 탐구가 미비한 상태이다. 따라서 본 논문의 목표는 기존 LLM 기반 Table Reasoning 연구를 분석하고 분류하여 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM 시대의 Table Reasoning 기술을 체계적으로 분류하고, LLM이 기존 모델(pre-LLMs)보다 우수한 이유를 분석하며, 미래의 발전 방향을 제시한 점에 있다. 구체적인 설계 아이디어와 분석 관점은 다음과 같다.

1. **기술적 분류**: LLM 기반의 Table Reasoning 기법을 'pre-LLM 시대의 계승 기술'과 'LLM 고유의 창발적 기술'로 나누어 분석한다.
2. **성능 우위 분석**: LLM이 Table Reasoning의 두 가지 핵심 난제인 Structure Understanding(구조 이해)과 Schema Linking(스키마 연결)을 어떻게 효율적으로 해결하는지 논리적으로 설명한다.
3. **미래 로드맵 제시**: 단순 성능 향상을 넘어 Multi-modal, Agent, Dialogue, RAG와 같은 실질적인 응용 확장 방향을 구체적으로 제안한다.

## 📎 Related Works

Table Reasoning 연구는 크게 세 단계의 발전 과정을 거쳤으며, 본 논문은 이를 **pre-LLM era**로 정의한다.
- **발전 단계**: Rule-based $\rightarrow$ Neural network-based $\rightarrow$ Pre-trained language model-based 순으로 발전하였다.
- **기존 방식의 한계**: pre-LLM 시대의 접근 방식은 주로 모델의 구조 변경이나 전용 사전 학습(pre-training) 태스크 설계에 집중하였다. 이는 높은 어노테이션 비용을 초래하며, 학습 데이터에 포함되지 않은 새로운 태스크(unseen tasks)에 대한 일반화 능력이 떨어진다는 치명적인 한계가 있었다.
- **차별점**: LLM 기반 방식은 모델 구조를 수정하는 대신 Prompt 설계, Pipeline 구축, 그리고 LLM의 창발적 능력(Emergent abilities)을 활용하여 적은 비용으로 더 높은 범용 성능을 달성한다.

## 🛠️ Methodology

본 논문은 LLM 기반 Table Reasoning의 성능을 향상시키는 기술을 다섯 가지 카테고리로 분류하여 설명한다.

### 1. pre-LLM 계승 기술 (Following pre-LLMs)
- **Supervised Fine-Tuning (SFT)**: 어노테이션된 데이터를 사용하여 LLM을 미세 조정한다. 기존의 수동 레이블링 데이터뿐만 아니라, LLM을 교사 모델로 사용하여 생성한 Distilled data를 활용해 오픈소스 소형 모델의 성능을 높이는 방식이 사용된다.
- **Result Ensemble**: LLM이 생성한 여러 결과 중 최적의 답안을 선택한다. Prompt나 모델을 다양하게 하여 결과의 다양성을 확보하고, 별도의 Verifier(검증기)를 학습시켜 가장 점수가 높은 답안을 채택하는 방식이 핵심이다.

### 2. LLM 고유 기술 (Unique to LLMs)
- **In-context Learning (ICL)**: 추가 학습 없이 적절한 지시어(Instruction)와 몇 가지 예시(Demonstrations)만을 Prompt에 포함시켜 정답을 유도한다. 특히 Text-to-SQL 작업에서 문법의 단순성 덕분에 매우 높은 효율을 보인다.
- **Instruction Design**: LLM의 지시어 수행 능력을 활용해 복잡한 작업을 하위 작업으로 분해한다.
    - **Modular Decomposition**: 전체 과제를 모듈별로 분해하여 추론 난이도를 낮춘다.
    - **Tool Using**: 수치 계산이나 검색 등 LLM이 취약한 부분은 외부 API나 도구를 호출하여 해결한다.
- **Step-by-Step Reasoning**: 복잡한 질문을 더 단순한 여러 단계의 하위 질문으로 분해하여 순차적으로 해결한다. 이는 추론의 복잡도를 낮추어 정확도를 높이는 효과가 있다.

## 📊 Results

### 실험 설정 및 벤치마크
논문은 Table Reasoning의 4가지 핵심 작업과 벤치마크를 정의한다.
- **Table QA**: WikiTableQuestions (WikiTQ) / 지표: Accuracy
- **Table Fact Verification**: TabFact / 지표: Accuracy
- **Table-to-Text**: FeTaQA / 지표: ROUGE-1
- **Text-to-SQL**: Spider / 지표: Execution Accuracy

### 주요 결과
- **LLM vs Pre-LLM**: 모든 벤치마크에서 LLM 기반 방법론이 pre-LLM 모델들보다 일관되게 높은 성능을 보였다. (예: TabFact의 경우 pre-LLM 0.85 $\rightarrow$ LLM 0.93)
- **기술별 효과**: Instruction Design과 Step-by-Step Reasoning 기술이 다양한 태스크 전반에 걸쳐 성능 향상에 가장 기여하는 것으로 나타났다.
- **태스크 특성**: Text-to-SQL 작업은 In-context Learning을 통해 가장 비약적인 성능 향상을 보였는데, 이는 SQL의 구조적 특성이 자연어보다 예시를 통한 학습에 유리하기 때문이다.

## 🧠 Insights & Discussion

### LLM이 우수한 이유에 대한 분석
본 논문은 Table Reasoning의 두 가지 난제를 LLM이 어떻게 해결하는지 분석한다.
1. **Structure Understanding**: 테이블의 스키마와 관계를 파악하는 능력이다. LLM의 Instruction Following 능력, 특히 코드 파싱 능력은 평면적인 텍스트 입력에서 계층적 구조를 인식하는 능력과 밀접하게 연관되어 있어 테이블 구조 이해에 도움을 준다.
2. **Schema Linking**: 질문 속 엔티티를 테이블의 엔티티와 연결하는 작업이다. LLM의 Step-by-Step Reasoning 능력은 복잡한 질문을 문장 단위에서 스팬(span) 단위로 단순화하고 불필요한 맥락을 필터링함으로써 정교한 연결을 가능하게 한다.

### 비판적 해석 및 한계
- **Error Cascade**: Step-by-Step Reasoning의 경우, 중간 단계에서 한 번의 오류가 발생하면 이후의 모든 추론 단계가 무너지는 '오류 전이(Error Cascade)' 문제가 발생한다. 이를 해결하기 위해 Tree-of-Thought(ToT)와 같은 다중 경로 탐색 기법의 도입이 필요하다.
- **데이터 품질**: Distilled data를 활용한 SFT의 경우, 데이터의 양은 늘릴 수 있으나 품질 저하 문제가 발생하므로 최소한의 인간 개입으로 품질을 높이는 방법이 연구되어야 한다.

## 📌 TL;DR

본 논문은 LLM 시대의 Table Reasoning 연구를 총망라한 서베이 논문으로, 기존의 모델 구조 중심 연구에서 **Prompt 및 Pipeline 설계 중심**으로 패러다임이 전환되었음을 보여준다. 특히 Instruction Design과 Step-by-Step Reasoning이 성능 향상의 핵심이며, 이는 LLM의 구조 이해 및 스키마 연결 능력 덕분임을 밝히고 있다. 향후 연구는 **Multi-modal 테이블 인식, 자율적 Agent 협업, Multi-turn 대화형 추론, 그리고 RAG를 통한 외부 지식 주입** 방향으로 확장될 가능성이 매우 높다.