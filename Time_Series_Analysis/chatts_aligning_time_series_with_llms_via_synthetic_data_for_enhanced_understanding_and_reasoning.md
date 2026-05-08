# ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning

Zhe Xie, Zeyan Li, Xiao He, Longlong Xu, Xidao Wen, Tieying Zhang, Jianjun Chen, Rui Shi, and Dan Pei (2025)

## 🧩 Problem to Solve

본 연구는 시계열 데이터(Time Series)에 대한 이해와 추론을 수행할 수 있는 다중모달 거대언어모델(Multimodal LLM, MLLM)을 구축하는 것을 목표로 한다. 시계열 데이터의 분석은 전력, 의료, 교통, 금융 등 다양한 실제 도메인에서 매우 중요하지만, 이를 텍스트 정보와 정렬(Alignment)하여 학습시킬 수 있는 고품질의 데이터셋이 매우 부족하다는 문제가 있다.

기존의 LLM 기반 시계열 분석 방식은 크게 세 가지로 나뉘지만, 각각 명확한 한계를 가진다. 첫째, Text-based 방식은 수치 데이터를 텍스트로 직접 입력하므로 프롬프트 길이 제한이 심하며 전역적 특징(Global features) 파악 능력이 떨어진다. 둘째, Vision-based 방식은 시계열 그래프 이미지를 입력으로 사용하지만, 이미지 해상도의 한계로 인해 세부적인 수치 정보나 국소적 특징(Local features)을 정확히 해석하는 데 어려움이 있다. 셋째, Agent-based 방식은 외부 분석 도구를 사용하지만, 도구의 기능적 제한과 도구 호출 과정에서의 환각(Hallucination) 및 높은 토큰 소모량 문제가 발생한다. 따라서 본 논문은 시계열을 이미지처럼 하나의 독립적인 모달리티(Modality)로 처리하여 전역적·국소적 특징을 모두 포착하고 다변량 시계열(Multivariate Time Series, MTS) 간의 관계를 추론할 수 있는 TS-MLLM의 필요성을 제기한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 실제 정렬 데이터의 부족 문제를 해결하기 위해 **속성 기반의 합성 데이터 생성(Attribute-based synthetic data generation)**과 **진화적 지시어 학습(Time Series Evol-Instruct)**을 통해 모델을 학습시키는 것이다.

핵심 기여 사항은 다음과 같다.

1. **속성 기반 합성 데이터 생성 방법론**: 시계열의 추세(Trend), 주기성(Periodicity), 노이즈(Noise), 국소 변동(Local Fluctuation)이라는 네 가지 속성을 정의하고, 이를 기반으로 정밀한 텍스트 설명이 포함된 합성 시계열 데이터를 생성하여 모달리티 정렬 문제를 해결하였다.
2. **Time Series Evol-Instruct (TSEvol)**: 속성 풀(Attribute Pool)을 활용하여 단순한 질의응답을 넘어 복잡한 추론이 필요한 고난도 Q&A 데이터셋을 점진적으로 생성하여 모델의 추론 능력을 강화하였다.
3. **ChatTS 아키텍처 설계**: 다변량 시계열의 가변 길이와 수량을 처리할 수 있는 Context-aware 인코더를 설계하였으며, 수치 정보 손실을 방지하는 Value-preserved 정규화 기법을 도입하였다.
4. **성능 검증**: 실제 데이터가 포함된 벤치마크에서 GPT-4o와 같은 최신 모델 대비 정렬 작업에서 $46.0\%$, 추론 작업에서 $25.8\%$의 성능 향상을 입증하였다.

## 📎 Related Works

기존 연구들은 LLM을 시계열 예측(Forecasting)과 같은 특정 작업에 활용하거나, 시계열 데이터를 텍스트 또는 이미지로 변환하여 입력하는 방식에 집중하였다. 특히 TimeLLM과 같은 연구는 특정 태스크에 특화되어 있어, 일반적인 시계열 속성에 대한 이해나 다이나믹한 대화형 추론에는 한계가 있었다.

최근의 Vision-based MLLM들은 시계열 그래프를 통해 패턴을 파악하려 했으나, 이는 수치적 정밀도가 떨어지는 문제가 있었다. 또한, Agent-based 방식은 ReAct 프레임워크 등을 통해 분석 도구를 사용하지만, 도구 선택의 오류나 도구 자체의 부정확성이 전체 시스템의 성능 저하로 이어진다. 본 연구는 이러한 우회적인 방법 대신, 시계열 데이터를 텍스트와 직접 정렬시키는 네이티브 다중모달 접근 방식을 취함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

ChatTS는 시계열 데이터를 텍스트 임베딩 공간으로 매핑하는 인코더와 이를 처리하는 LLM(QWen2.5-14B-Instruct)으로 구성된다. 전체 파이프라인은 [속성 정의 $\rightarrow$ 합성 데이터 생성 $\rightarrow$ TSEvol을 통한 Q&A 생성 $\rightarrow$ 모델 학습] 순으로 진행된다.

### 주요 구성 요소 및 방법론

#### 1. 속성 기반 시계열 생성기 (Attribute-Based Generator)

시계열의 특징을 Trend, Periodicity, Noise, Local Fluctuation의 네 가지 범주로 분류한다.

- **Attribute Selector**: 실제 도메인의 메트릭 이름(예: CPU Usage)을 샘플링하고, GPT를 이용해 해당 메트릭에 적합한 속성 부분집합을 선택한다.
- **Attribute Sampler**: 선택된 속성에 구체적인 수치(위치, 진폭 등)를 할당하여 속성 풀(Attribute Pool)을 생성한다.
- **Time Series Generator**: 규칙 기반(Rule-based) 방식으로 속성 풀의 내용과 정확히 일치하는 시계열 배열을 생성한다.

#### 2. Time Series Evol-Instruct (TSEvol)

단순한 Q&A를 복잡한 추론 문제로 진화시키는 알고리즘이다. 속성 풀과 상관관계 풀(Correlation Pool)에서 정보를 추출하여, 다음과 같은 진화 유형을 적용한다.

- **Breadth/Depth**: 질문의 범위나 깊이를 확장한다.
- **Reasoning/Situation**: 단순 속성 확인을 넘어, 특정 상황에서의 원인 추론이나 물리적 의미 해석을 요구하는 질문으로 발전시킨다.

#### 3. 모델 아키텍처 (ChatTS)

- **Context-Aware TS Encoder**: 입력 시계열을 고정된 크기의 패치(Patch)로 나누고, 5층 MLP(Multi-Layer Perceptron)를 통해 텍스트 임베딩 공간으로 투영한다.
- **Insert & Concatenate**: 인코딩된 시계열 패치들을 원래 텍스트 쿼리의 적절한 위치(해당 메트릭이 언급되는 지점)에 삽입하여, LLM이 텍스트 문맥과 시계열 데이터를 동시에 참조할 수 있게 한다.
- **Value-Preserved Normalization**: 시계열 데이터는 일반적으로 $0 \sim 1$ 사이로 Min-Max 정규화를 수행하지만, 이 과정에서 실제 수치 정보가 손실된다. 이를 해결하기 위해 정규화에 사용된 스케일링 인자(Scaling factor)와 오프셋(Offset) 값을 텍스트 프롬프트에 함께 제공하여 LLM이 원래 수치를 복원하여 인식할 수 있도록 한다.

### 학습 절차

학습은 두 단계로 진행된다.

1. **Large-Scale Alignment Training**: UTS(단변량), MTS-Shape(다변량 전역 상관관계), MTS-Local(다변량 국소 특징) 데이터셋을 사용하여 텍스트와 시계열 모달리티 간의 기본적인 정렬을 학습시킨다.
2. **Supervised Fine-Tuning (SFT)**: TSEvol로 생성된 데이터와 Instruction Following 데이터셋을 사용하여 복잡한 질의응답 및 추론 능력을 강화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 실제 도메인 데이터가 포함된 Dataset A, 합성 데이터 기반의 Dataset B, 오픈소스 데이터인 MCQ2를 사용하였다.
- **평가 작업**: 정렬 작업(추세, 주기성, 노이즈, 국소 변동, 상관관계, 클러스터링)과 추론 작업(귀납적, 연역적, 인과적, 비교 추론)으로 구분하였다.
- **지표**: 범주형 작업은 F1-Score, 수치형 작업은 Relative Accuracy($1.0 - \text{relative error}$)를 사용하였다.

### 주요 결과

- **정렬 성능**: ChatTS는 모든 작업에서 baseline 모델들을 압도하였다. 특히 GPT-4o 대비 범주형 지표에서 $46.0\% \sim 75.9\%$, 수치형 지표에서 $80.7\% \sim 112.7\%$ 향상된 성능을 보였다.
- **추론 성능**: 귀납적 추론(Inductive Reasoning)에서 baseline 대비 $34.5\%$의 성능 향상을 보였으며, 전체 평균 추론 정확도는 $0.667$로 가장 높았다.
- **효율성**: 네이티브 모달리티 인코딩을 통해 텍스트 기반 방식보다 훨씬 적은 토큰을 사용하며, 비용 면에서도 매우 효율적임을 입증하였다.

### 분석 결과 (Ablation Study)

- **합성 데이터의 효과**: GPT가 생성한 단순 Python 코드 기반 데이터보다 본 논문의 속성 기반 데이터가 국소 변동 및 수치 분석에서 훨씬 뛰어난 성능을 보였다.
- **모달리티의 중요성**: 시계열 인코더를 제거한 Text-only 모델은 다변량 시계열(MTS) 작업에서 거의 동작하지 않았으며, 32B 크기의 LLM을 SFT 하더라도 native 모달리티를 가진 ChatTS(14B)의 성능을 넘지 못했다.
- **Agent 방식의 한계**: "Perfect Tools"(정확도가 100%인 도구)를 제공하더라도, LLM이 도구를 잘못 선택하는 'Error Tool Using' 문제가 발생하여 ChatTS보다 성능이 낮게 나타났다.

## 🧠 Insights & Discussion

본 연구는 시계열 데이터를 처리함에 있어 텍스트 변환이나 이미지 변환이라는 우회로 대신, **네이티브 다중모달 정렬**이라는 정공법이 가장 효과적임을 보여주었다. 특히 실제 데이터가 부족한 상황에서 정밀하게 설계된 속성 기반 합성 데이터가 실제 데이터에 대한 일반화 성능(Generalization)까지 확보할 수 있다는 점은 매우 고무적이다. 이는 시계열의 패턴이 어느 정도 규칙적이라는 특성을 잘 활용한 결과로 해석된다.

하지만 몇 가지 한계점과 논의 사항이 존재한다. 첫째, MLP 기반의 단순 인코더가 효과적이었으나, 더 복잡한 시계열 구조를 포착하기 위한 고도화된 인코더 설계가 필요하다. 둘째, 합성 데이터만으로 학습되었기에, 향후 실제 정렬 데이터셋이 구축된다면 성능이 더욱 비약적으로 향상될 가능성이 크다. 셋째, 현재는 이해와 추론에 집중하고 있으나, 텍스트를 입력받아 시계열을 생성하는 생성(Generation) 작업으로의 확장이 필요하다.

## 📌 TL;DR

ChatTS는 시계열 데이터를 네이티브 모달리티로 처리하는 최초의 다변량 시계열 MLLM이다. 데이터 부족 문제를 해결하기 위해 정밀한 **속성 기반 합성 데이터 생성** 및 **TSEvol** 알고리즘을 제안하였으며, 이를 통해 수치 정보가 보존되는 정규화 및 컨텍스트 인식 인코딩 구조를 구현하였다. 실험 결과, GPT-4o를 포함한 기존 텍스트/비전/에이전트 기반 방식보다 정렬 및 추론 성능이 월등히 뛰어나며 비용 효율적이다. 이 연구는 향후 AIOps, 금융 분석 등 복잡한 시계열 추론이 필요한 실무 분야에 LLM을 통합하는 핵심적인 방법론을 제시한다.
