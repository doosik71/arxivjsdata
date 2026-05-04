# KNOWLEDGE-IN-CONTEXT: TOWARDS KNOWLEDGE-ABLE SEMI-PARAMETRIC LANGUAGE MODELS

Xiaoman Pan, Wenlin Yao, Hongming Zhang, Dian Yu, Dong Yu & Jianshu Chen (2023)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(Large Language Models, LLMs)이 직면한 두 가지 핵심적인 한계를 해결하고자 한다. 첫째, Fully-parametric 언어 모델은 제로샷(zero-shot) 또는 퓨샷(few-shot) 설정에서 다양한 자연어 처리(NLP) 작업을 수행하기 위해 필요한 방대한 지식을 모델 파라미터 내부에 직접 저장해야 하므로, 모델의 크기가 비정상적으로 커져야 하는 문제가 있다. 둘째, 세상의 지식은 계속해서 변화하지만, 이를 업데이트하기 위해 모델을 매번 다시 학습시키는 것은 비용 면에서 매우 비효율적이다.

기존의 Semi-parametric 접근 방식은 외부 메모리를 활용하여 이 문제를 해결하려 했으나, 유용한 지식이 거대한 말뭉치 속에 희소하게 분포되어 있어 정확한 텍스트 청크(chunk)를 검색하기 어렵고, 적절한 지식의 단위(granularity)를 결정하는 것이 까다롭다는 한계가 있었다. 따라서 본 논문의 목표는 구조화된 지식 자원을 활용하여 효율적으로 지식을 검색하고, 이를 통해 모델 크기를 획기적으로 줄이면서도 높은 성능을 유지하는 semi-parametric 언어 모델 아키텍처를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Knowledge-in-Context (KiC)**라는 새로운 semi-parametric 언어 모델 아키텍처를 제안한 것이다. KiC의 중심 아이디어는 단순한 텍스트 뭉치가 아닌, 6가지의 다양한 범주(Entity, Dictionary, Commonsense, Event, Script, Causality)로 구성된 **지식 풍부한 외부 메모리(Knowledge-rich external memory)**를 구축하고, 입력 인스턴스에 따라 가장 적절한 지식 유형을 동적으로 선택하는 **인스턴스 적응형 지식 선택기(Instance-adaptive knowledge selector)**를 도입한 것이다.

특히, 저자들은 KiC를 일종의 **Mixture-of-Experts (MoE)** 모델로 재정의하였다. 여기서 지식 선택기는 라우터(router) 역할을 하며, 각 지식 범주와 텍스트-투-텍스트(text-to-text) 모델의 조합이 개별 전문가(expert)가 된다. 이러한 관점의 전환을 통해 이산적인(discrete) 지식 선택 과정을 미분 가능한 형태로 학습시킬 수 있는 새로운 알고리즘을 개발하였다.

## 📎 Related Works

기존의 연구들은 크게 두 가지 방향으로 진행되었다. 첫째는 사전 학습된 언어 모델(PLMs)에 지식을 주입하는 방식(Knowledge Injection)으로, 어휘, 엔티티, 구문 지식 등을 사전 학습 단계에서 추가하여 지식 집약적 작업의 성능을 높이려 했다. 하지만 이는 여전히 지식을 파라미터 내에 암기시키는 방식이라는 한계가 있다.

둘째는 Semi-parametric 언어 모델로, 모델 외부에 대규모 텍스트 청크를 저장하고 검색하여 활용하는 방식이다. 하지만 기존 방식들은 주로 일반 텍스트(plain text) 기반의 메모리를 사용했다. 본 논문은 일반 텍스트보다 밀도 높고 정확한 정보가 담긴 구조화된 지식 자원(예: 지식 그래프)을 활용함으로써, 검색의 정확도를 높이고 모델이 더 작은 규모에서도 효율적으로 지식을 활용할 수 있도록 차별화하였다.

## 🛠️ Methodology

### 전체 파이프라인

KiC의 전체 구조는 **지식 선택기(Knowledge Selector)**, **외부 지식 메모리 및 리트리버(External Knowledge Memory & Retriever)**, 그리고 **텍스트-투-텍스트 백본 모델(Text-to-Text Backbone, 예: T5)**의 세 가지 모듈로 구성된다.

1. **지식 선택**: 입력 인스턴스가 들어오면 지식 선택기가 6가지 지식 범주 중 가장 적절한 하나를 선택하거나, 지식이 필요 없는 경우 'Generalist' 모드를 선택한다.
2. **지식 검색**: 선택된 범주 내에서 Dense Retrieval 기술을 통해 가장 관련성이 높은 지식 조각들을 검색한다.
3. **답변 생성**: 검색된 지식을 입력 텍스트와 결합(concatenation)하여 프롬프트를 구성하고, 이를 T5 모델에 입력하여 최종 답변을 생성한다.

### 외부 지식 메모리 구성

메모리는 $\langle \text{subject, relation, object} \rangle$ 형태의 트리플렛(triplet)으로 구성된 6가지 리소스를 포함한다.

- **Dictionary**: Wiktionary 기반의 단어 정의 및 예문.
- **Commonsense**: ConceptNet 기반의 일상 상식.
- **Entity**: Wikipedia 및 Wikidata 기반의 엔티티 속성 및 관련 문장.
- **Event**: ATOMIC, GLUCOSE, ASER 기반의 일상 사건 지식.
- **Script**: 영화 스크립트 기반의 상황-대화-비언어적 정보.
- **Causality**: CausalBank 기반의 원인-결과 관계.

검색을 위해 MPNet 인코더를 사용하여 키-값 쌍을 벡터화하며, MIPS(Maximum Inner Product Search) 알고리즘인 SCaNN을 통해 효율적으로 검색한다.

### 학습 절차 및 MoE 정식화

KiC는 미분 불가능한 선택 과정을 해결하기 위해 MoE 구조로 정식화된다. 지식 선택기 $S(x)$는 $(K+1)$ 클래스의 선형 분류기로 모델링되며, 각 범주를 선택할 확률 $S^k(x)$를 출력한다.

최종 출력 $\hat{y}$는 다음과 같이 계산된다.
$$\bar{k} = \arg \max_{k} S^k(x)$$
$$\hat{y} = T(x \oplus c_{\bar{k}}) \cdot S^{\bar{k}}(x)$$
여기서 $x \oplus c_{\bar{k}}$는 입력 $x$와 검색된 지식 $c_{\bar{k}}$의 결합을 의미하며, $\hat{y}$는 T5 모델 $T(\cdot)$의 출력 로짓(logits)에 선택 확률 $S^{\bar{k}}(x)$를 곱하여 계산함으로써 전체 과정을 미분 가능하게 만든다.

### 손실 함수

학습 시에는 표준 교차 엔트로피(Cross-Entropy) 손실과 더불어, 특정 전문가(지식 범주)로의 쏠림 현상을 방지하기 위해 SwitchTransformer의 **로드 밸런싱 손실(Load Balancing Loss)**을 추가한다.
$$L(x, y) = \sum_{t=1}^{T} \text{CrossEntropy}(\hat{y}_t, y_t) + \alpha \cdot \text{Balancing}(S(x))$$
여기서 $\alpha$는 두 손실 간의 가중치를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: P3(held-out unseen tasks) 및 MMLU 벤치마크.
- **기준선(Baselines)**: BERT, RoBERTa, GPT-Neo, GPT-J, OPT, GPT-NeoX, T0 등.
- **평가 지표**: 제로샷 정확도(Accuracy %).

### 주요 결과

1. **파라미터 효율성**: $0.77\text{B}$ 파라미터를 가진 $\text{KiC}_{\text{Large}}$ 모델이 $20\text{B}\sim 30\text{B}$ 규모의 GPT-NeoX나 OPT보다 우수한 제로샷 성능을 보였다. 이는 모델 크기를 약 25~38배 줄였음에도 더 높은 성능을 낸 것이다.
2. **MMLU 성능**: $\text{KiC}_{\text{Large}}$는 MMLU에서 $39.4\%$의 정확도를 기록하였으며, 이는 $175\text{B}$ 파라미터를 가진 GPT-3의 5-shot 성능($43.9\%$)에 근접한 수치이다.
3. **창발적 능력(Emergent Abilities)**: Fully-parametric 모델(T5, T0)은 모델 크기가 매우 커져야 성능이 급격히 상승하는 창발적 특성을 보이지만, KiC는 훨씬 작은 규모($0.22\text{B} \to 0.77\text{B}$)에서 이러한 성능 점프가 나타남을 확인하였다.
4. **인-도메인 평가**: 멀티태스크 학습 시, $\text{KiC}_{\text{Large}}$는 T0 Large보다 특히 지식 집약적인 작업(CosmosQA, DREAM 등)에서 월등한 성능 향상을 보였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 **구조화된 외부 지식이 일반 텍스트보다 훨씬 효율적임**을 입증한 것이다. Ablation study를 통해 일반 텍스트(Wikipedia)만 사용했을 때보다 6종의 구조화된 지식을 사용했을 때 성능이 크게 향상됨을 확인하였다. 또한, 단순히 모든 지식을 섞어서 검색하는 것보다 인스턴스에 맞게 범주를 먼저 선택하는 'Selector'의 역할이 결정적임을 보였다.

특히 흥미로운 점은 **'Generalist' 전문가**의 필요성이다. 분석 결과, 모델은 모든 경우에 지식을 사용하는 것이 아니라, 특정 인스턴스에 대해서는 외부 지식 없이 내부 파라미터만으로 판단하는 것이 유리하다는 것을 스스로 학습하였다.

한계점으로는 현재 Top-1 선택 전략만을 사용했다는 점이 있으며, 향후 Top-n 선택으로 확장하여 더 다양한 지식을 조합하는 연구가 필요할 것으로 보인다. 또한, 구조화된 지식 외에 비구조화된 일반 텍스트를 어떻게 효율적으로 통합할 것인가에 대한 과제가 남아 있다.

## 📌 TL;DR

본 논문은 6가지 범주의 구조화된 외부 지식 메모리와 이를 동적으로 선택하는 MoE 기반의 라우터를 결합한 **Knowledge-in-Context (KiC)** 모델을 제안한다. 이를 통해 $0.77\text{B}$라는 작은 크기의 모델로 $20\text{B}$ 이상의 거대 모델들을 능가하는 제로샷 성능을 달성하였으며, 모델의 크기를 획기적으로 줄이면서도 지식 능력을 극대화할 수 있는 Semi-parametric 아키텍처의 가능성을 제시하였다.
