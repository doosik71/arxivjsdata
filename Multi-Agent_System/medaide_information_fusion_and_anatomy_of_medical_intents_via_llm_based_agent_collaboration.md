# MedAide: Information Fusion and Anatomy of Medical Intents via LLM-based Agent Collaboration

Dingkang Yang et al. (2025)

## 🧩 Problem to Solve

현대 의료 지능 시스템에서 다양한 임상 소스로부터 발생하는 이질적이고 다중적인 의도(multi-intent) 정보를 융합하는 능력은 신뢰할 수 있는 의사결정 시스템 구축의 핵심이다. 최근 LLM 기반의 정보 상호작용 시스템이 의료 분야에서 가능성을 보여주고 있으나, 복잡한 의료 의도를 처리할 때 정보의 중복성(redundancy)과 결합(coupling) 문제가 발생하며, 이는 심각한 환각(hallucination) 현상과 성능 병목으로 이어진다.

본 논문의 목표는 의도 인식 기반의 정보 융합과 전문 의료 도메인 간의 조율된 추론을 가능하게 하는 LLM 기반 의료 멀티 에이전트 협업 프레임워크인 **MedAide**를 제안하는 것이다.

## ✨ Key Contributions

MedAide의 핵심 아이디어는 복잡한 사용자 쿼리를 정형화된 표현으로 분해하고, 이를 동적으로 매칭되는 전문 에이전트들이 순차적으로 처리하도록 하여 정보의 누락이나 중복 없이 정밀한 의료 추론을 수행하는 것이다. 이를 위해 다음의 세 가지 핵심 모듈을 설계하였다.

1. **Regularization-guided Information Extraction (RIE)**: 구문 제약 조건과 RAG(Retrieval-Augmented Generation)를 결합하여 복잡한 쿼리를 구조화된 시맨틱 표현으로 변환함으로써 추출의 정밀도를 높인다.
2. **Dynamic Intent Prototype Matching (IPM)**: BioBERT 기반의 인코더를 통해 의료 의도 프로토타입 임베딩을 생성하고, 시맨틱 유사도 매칭을 통해 다회차 대화 속에서 에이전트의 의도를 적응적으로 인식하고 업데이트한다.
3. **Rotation Agent Collaboration (RAC)**: 전문 에이전트 간의 동적 역할 교체(Role Rotation)와 결정 레벨의 정보 융합을 통해 일관성 있고 최적화된 의사결정을 도출한다.

## 📎 Related Works

### LLM 기반 정보 융합 및 의료 응용

최근 LLM은 이질적인 데이터 소스를 조화롭게 통합하는 융합 엔진으로서의 가능성을 보여주었으며, HuatuoGPT-II나 ZhongJing과 같은 의료 특화 LLM들이 등장하였다. 그러나 이러한 모델들은 여전히 도메인 특화 숙련도가 부족하거나, 실제 임상 시나리오에서 요구되는 복잡한 추론 및 다각도 정보 통합 능력에 한계가 있다.

### LLM 기반 멀티 에이전트 협업

MEDCO나 MedAgents와 같이 여러 에이전트가 협력하여 문제를 해결하는 방식이 제안되었다. 하지만 기존 연구들은 주로 의료 교육 훈련이나 선택적인 질의응답에 집중되어 있으며, 실제 진단 및 치료 시나리오 뒤에 숨겨진 정교한 의도를 포괄적으로 이해하는 능력이 부족하다. MedAide는 단순한 역할 수행을 넘어 '의도 인식(Intent-awareness)'과 '종합적 분석' 능력을 갖추었다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

MedAide의 전체 파이프라인은 **RIE $\rightarrow$ IPM $\rightarrow$ RAC**의 세 단계로 구성된다.

### 1. Regularization-guided Information Extraction (RIE)

사용자의 자연어 쿼리는 모호성이 크므로, 이를 정규화된 쿼리 $Q_{std}$로 변환하는 과정을 거친다.

- **Syntactic Regularization**: Context-Free Grammar(CFG) 규칙을 사용하여 쿼리를 토큰화하고 구문 트리(Syntactic Tree)를 생성하여 구조적으로 분석한다.
- **Key Element Extraction**: 증상, 상태 설명, 병력 등 핵심 요소를 추출하여 요소 집합 $V_i$를 형성한다.
- **RAG 및 Recall Analysis**: 1,095개의 전문가 검토 가이드라인 데이터베이스에서 $V_i$와 관련된 문서를 검색하여 문서 집합 $D_{ref}$를 구축하고, BERT-Score 기반으로 관련 문서를 선별한다.
- **Refined Query Constructor**: 생성된 서브 쿼리들을 필터링, 중복 제거, 통합, 문법 정규화, 의도 우선순위 지정, 포맷 표준화라는 6가지 규칙을 적용하여 최종적으로 정제된 쿼리를 생성한다.

### 2. Dynamic Intent Prototype Matching (IPM)

정제된 쿼리 $Q_{opt}$를 17가지 의료 의도 카테고리 중 하나로 분류하여 적절한 에이전트를 활성화한다.

- **Contextual Encoder**: BioBERT를 기반으로 하며, 쿼리를 768차원 임베딩 공간으로 매핑한다.
- **Similarity Matching**: 쿼리와 각 의도 프로토타입 임베딩 $V_i$ 사이의 코사인 유사도 $S_{ij}$를 계산한다.
$$S_{ij} = \frac{Q_{opt} \cdot V_i}{\|Q_{opt}\| \|V_i\|}$$
- **Probability Distribution**: Softmax 함수를 통해 각 의도의 확률 분포 $P_{ij}$를 구한다.
$$P_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^{17} \exp(S_{ik})}$$
- **Activation**: 확률 $P_{ij}$가 설정된 임계값(Threshold)을 초과하면 해당 의도가 활성화되어 대응하는 에이전트가 호출된다.

### 3. Rotation Agent Collaboration (RAC)

활성화된 에이전트들은 **Pre-diagnosis $\rightarrow$ Diagnosis $\rightarrow$ Medicament $\rightarrow$ Post-diagnosis** 순서로 역할을 교대하며 정보를 융합한다.

- **Diagnosis Agent의 하이브리드 융합**: 키워드 기반 매칭(lexical matching)과 시맨틱 기반 매칭(embedding-based fusion)을 병렬로 수행하여 최종 문서 집합 $D_{final}$을 구성한다.
  - 키워드 채널: $D_{slice} = \{d \in D \mid \text{KeywordMatch}(Q, d) = \text{True}\}$
  - 시맨틱 채널: $D_{match} = \{d \in D \mid S(Q, d) > \tau\}$
  - 최종 융합: $D_{final} = D_{slice} \cup D_{match}$

- **Polling-Based Coordination Protocol**: 각 단계 $s$에서 메인 컨택 에이전트 $A_{mc}^s$가 주도하여 다른 보조 에이전트들로부터 지식을 수집하고 통합한다.
  - 메인 에이전트의 초기 출력: $O_s^{(0)} = A_{mc}^s(P_s, D_{input}^{(s)})$
  - 보조 에이전트의 기여: $C_i^{(s)} = A_i(P_s, D_{input}^{(s)}, O_s^{(0)})$
  - 최종 단계 출력: $O_{final}^s = \text{Integrate}_s(O_s^{(0)}, \{C_i^{(s)} \mid i \neq s\})$
  - 전체 결과 합성: $\text{Output}_{final} = \text{Synthesize}(O_{final}^1, O_{final}^2, O_{final}^3, O_{final}^4)$

## 📊 Results

### 실험 설정

- **데이터셋**: 17가지 의료 의도를 포함하는 4가지 벤치마크(Pre-Diagnosis, Diagnosis, Medicament, Post-Diagnosis)를 사용하였으며, 각 벤치마크는 500개의 복합 의도 인스턴스로 구성된다.
- **비교 모델**: ZhongJing, Meditron-7B, HuatuoGPT-II, Baichuan4 (의료 특화) 및 Llama-3.1-8B, GPT-4o, Claude 3.7 Sonnet, DeepSeek-R1 (범용).
- **평가 지표**: BLEU-1/2, ROUGE-L, GLEU, Meteor, BERT-Score 및 전문가 의사 평가.

### 주요 결과

1. **정량적 성능 향상**: 모든 벤치마크에서 MedAide를 결합했을 때 베이스라인 모델들의 성능이 일관되게 향상되었다. 특히 DeepSeek-R1과 결합했을 때 Meteor 및 BERT-Score에서 최고 성능을 기록하였다.
2. **에이전트 협업의 효과**: Medicament 벤치마크에서 MedAide는 MedAgents 대비 BLEU-1에서 16.44%p, ROUGE-L에서 16.70%p의 절대적 이득을 보였다. 이는 특화된 약물 검색 및 융합 메커니즘의 효과로 해석된다.
3. **전문가 평가**: 6명의 전문의가 참여한 평가 결과, MedAide 기반 모델들의 Win-rate가 압도적으로 높았으며, 특히 사실적 정확성과 권고안의 실용성 면에서 우수하다는 평가를 받았다.
4. **Ablation Study**: RIE 모듈을 제거했을 때 성능이 가장 크게 하락하여, 입력 쿼리의 정규화 및 구조화가 전체 파이프라인의 성능을 결정짓는 핵심 요소임을 확인하였다. 또한, 단순한 GPT-4o 프롬프트 기반 의도 인식보다 학습된 BioBERT 인코더 기반의 IPM이 더 정밀한 매칭을 수행함을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

MedAide는 단순한 RAG나 멀티 에이전트 구조를 넘어, **'구문 정규화 $\rightarrow$ 의도 프로토타입 매칭 $\rightarrow$ 순차적 역할 교대'**라는 체계적인 파이프라인을 구축하였다. 특히 reasoning 모델(DeepSeek-R1 등)과 결합했을 때, 프레임워크가 제공하는 구조화된 추론 경로가 모델의 내재적 추론 능력을 극대화하는 시너지 효과를 낸다.

### 한계 및 미해결 과제

1. **데이터 커버리지**: 현재 26,684개의 약물 샘플과 506개의 실제 임상 케이스를 사용하고 있으나, 희귀 질환이나 매우 전문적인 약물에 대한 데이터는 여전히 부족하다.
2. **단일 모달리티**: 현재는 텍스트 기반의 언어 처리 위주로 설계되어 있어, 실제 의료 진단에 필수적인 의료 영상(Medical Imaging) 데이터와의 통합(Multimodal)이 향후 과제로 남아 있다.
3. **API 의존성**: GPT-4o 등 폐쇄형 API에 의존하는 부분이 있어, 운영 비용 및 데이터 프라이버시 문제를 해결하기 위해 효율적인 오픈소스 모델로의 대체 연구가 필요하다.

### 비판적 해석

본 연구는 에이전트의 수를 2개에서 6개까지 변화시키며 실험하였으며, 4개(Pre-Diagnosis, Diagnosis, Medicament, Post-Diagnosis)일 때 최적의 성능을 보였다. 이는 너무 세분화된 에이전트 구성은 오히려 조율 오버헤드(coordination overhead)를 발생시켜 성능을 저하시킬 수 있음을 시사한다. 따라서 무조건적인 에이전트의 세분화보다는 도메인 지식에 기반한 적절한 계층 구조 설계가 더 중요하다.

## 📌 TL;DR

MedAide는 복잡한 의료 쿼리를 구조적으로 분해(RIE)하고, BioBERT 기반으로 정밀하게 의도를 매칭(IPM)하며, 전문 에이전트들이 순차적으로 협력(RAC)하여 답변을 생성하는 프레임워크이다. 실험 결과, 범용 및 의료 특화 LLM 모두의 의료 추론 능력을 유의미하게 향상시켰으며, 특히 DeepSeek-R1과 같은 최신 추론 모델과 결합 시 최상의 성능을 보인다. 이 연구는 향후 멀티모달 의료 진단 시스템 및 전문 의료 AI 에이전트 설계에 있어 중요한 방법론적 기틀을 제공한다.
