# Sim-GPT: Text Similarity via GPT Annotated Data

Shuhe Wang, Beiming Cao, Shengyu Zhang, Xiaoya Li, Jiwei Li, Fei Wu, Guoyin Wang, Eduard Hovy (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Semantic Textual Similarity (STS) 태스크를 위한 고품질의 지도 학습(supervised learning) 데이터셋의 부족이다. STS는 두 문장 사이의 의미적 유사도를 측정하는 작업으로, 이를 위해 정밀하게 레이블링된 대규모 문장 쌍 데이터가 필요하다.

기존의 접근 방식들은 다음과 같은 한계를 가지고 있다:

- **비지도 학습(Unsupervised techniques):** 구체적인 STS 정보(유사도의 세밀한 수준)에 대한 가이드 없이 일반적인 언어 패턴만을 학습한다.
- **NLI 기반 학습:** SNLI나 MNLI와 같이 함의(entailment)나 모순(contradiction) 관계를 이용한 데이터를 사용하지만, 이는 텍스트 유사도와 부분적으로만 상관관계가 있을 뿐 직접적인 유사도 신호가 되지 않는다.
- **LLM 직접 활용:** GPT-4와 같은 대규모 언어 모델(LLM)을 직접 사용하여 유사도를 측정할 수 있으나, 매번 추론을 요청할 때 발생하는 비용이 매우 높고, BERT나 RoBERTa와 같은 소형 모델에 비해 추론 속도가 현저히 느리다.

따라서 본 연구의 목표는 LLM의 강력한 이해 능력을 활용하여 고품질의 STS 학습 데이터를 생성하고, 이를 통해 효율적이고 빠른 소형 모델을 학습시키는 Sim-GPT 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 LLM을 직접 추론 엔진으로 사용하는 대신, **'데이터 생성기(Data Annotator)'**로 활용하는 것이다.

1. **Sim-GPT 프레임워크 제안:** GPT-4를 통해 대규모의 STS 학습 데이터를 생성하고, 이 데이터를 바탕으로 BERT 또는 RoBERTa 기반의 소형 모델을 학습시키는 파이프라인을 구축하였다. 이를 통해 LLM의 성능을 유지하면서도 추론 비용을 절감하고 속도를 높였다.
2. **대규모 합성 데이터셋 구축:** GPT-4를 이용하여 371K 개의 고품질 STS 주석(annotation) 예제를 수집하고 이를 공개하여 후속 연구의 발판을 마련하였다.
3. **SOTA 성능 달성:** 제안된 방법론을 통해 7개의 주요 STS 벤치마크에서 기존의 SOTA 모델인 PromCSE보다 평균 0.42점 높은 성능을 기록하며 최첨단 성능을 입증하였다.

## 📎 Related Works

### Semantic Textual Similarity (STS)

기존 STS 연구는 크게 세 가지 방향으로 진행되었다:

- **벡터 거리 기반:** 문장을 고차원 공간에 인코딩하고 코사인 유사도 등을 통해 거리를 측정하는 비지도 학습 방식이다.
- **NLI 기반 지도 학습:** SNLI, MNLI 데이터셋을 활용해 함의 관계는 가깝게, 모순 관계는 멀게 학습시키는 Siamese networks 기반의 접근법이다.
- **대조 학습(Contrastive Learning):** SimCSE, PromCSE와 같이 유사한 문장 쌍은 가깝게, 유사하지 않은 쌍은 멀게 배치하여 임베딩 공간을 최적화하는 방식이다.

### Large Language Models (LLMs)

최근의 LLM 연구는 파라미터 규모를 키우는 방향(GPT-3, PaLM)과 효율적인 학습 방법(LLaMA)으로 발전하였다. 특히, 프롬프트를 통해 모델의 내부 지식을 끌어내는 In-context learning이나, LLM의 지식을 소형 모델로 전이하는 지식 증류(Knowledge Distillation) 연구가 활발히 진행되고 있다. Sim-GPT는 이러한 LLM의 지식을 합성 데이터 형태로 추출하여 소형 모델에 주입하는 방식의 연장선에 있다.

## 🛠️ Methodology

Sim-GPT의 전체 파이프라인은 크게 두 단계로 구성된다.

### 1. GPT-4 Data Annotation (데이터 생성 단계)

GPT-4가 주어진 입력 문장에 대해 **(유사한 문장, 유사하지 않은 문장)**의 쌍을 생성하도록 유도한다.

- **데이터 소스:** 데이터의 포괄성을 위해 세 가지 소스를 활용한다.
  - **Captions:** Flickr30k (이미지 묘사 텍스트)
  - **Questions:** Quora Question Pairs (사용자 질문)
  - **Multi-genre Long Sentences:** RedPajamas의 일부 (다양한 장르의 긴 문장)
- **프롬프트 설계:** GPT-4가 단순한 구문 변경(paraphrase)이나 부정(negation)을 하지 않도록 정교한 지침을 제공한다.
  - **Task Description:** 상황이나 사건을 읽고 반드시 유사한 문장 하나와 유사하지 않은 문장 하나를 작성하도록 명시한다.
  - **Few-shot Examples:** 모델이 출력 형식을 모방할 수 있도록 8개의 예시를 제공한다.
  - **Input Sentence:** 실제 주석을 달 대상 문장을 입력한다.

### 2. Contrastive-based STS Learning (모델 학습 단계)

생성된 데이터를 사용하여 SimCSE 또는 PromCSE 프레임워크 기반의 소형 모델(RoBERTa 등)을 학습시킨다.

- **학습 데이터 구조:** 각 데이터는 $\mathcal{D} = \{(x_i, x^+_i, x^-_i)\}_{i=1}^m$ 형태의 트리플렛(triplet)으로 구성된다. 여기서 $x_i$는 원문, $x^+_i$는 유사 문장, $x^-_i$는 비유사 문장이다.
- **SimCSE 손실 함수:**
    학습된 RoBERTa 모델이 생성한 벡터 표현 $(h, h^+, h^-)$를 이용하여 다음의 대조 손실 함수(Contrastive Loss)를 최적화한다.
    $$L_{CL} = -\log \frac{e^{\text{sim}(h_i, h^+_i)/\tau}}{\sum_{j=1}^N (e^{\text{sim}(h_j, h^+_j)/\tau} + \alpha * e^{\text{sim}(h_i, h^-_j)/\tau})}$$
    여기서 $\text{sim}(\cdot)$은 코사인 유사도, $\tau$는 정규화 파라미터, $\alpha$는 비유사 문장에 대한 가중치이다.

- **PromCSE 확장:** SimCSE가 어려운 부정 샘플(hard negatives)을 구분하는 데 한계가 있음을 해결하기 위해 마진 손실($L_M$)을 추가한다.
    $$L = L_{CL} + \lambda \cdot L_M$$
    $$L_M = \max(0, m + \text{sim}(h_i, h^*_i) - \text{sim}(h_i, h^+_i))$$
    여기서 $h^*_i$는 가장 유사도가 낮은 부정 샘플을 의미하며, $m$과 $\lambda$는 하이퍼파라미터이다.

- **추론 절차:** 테스트 단계에서는 두 문장을 인코딩하여 코사인 유사도를 계산하며, 최종적으로 0~5 범위의 점수로 정규화하여 출력한다.

## 📊 Results

### 실험 설정

- **데이터셋:** GPT-4 생성 데이터 371K 및 NLI 데이터 275K (총 646K) 사용.
- **평가 지표:** Spearman’s correlation (스피어만 상관계수).
- **평가 작업:** STS12-16, STS Benchmark, SICK-Relatedness 등 총 7개의 벤치마크.
- **비교 대상:** 비지도 학습 모델(GloVe, BERT), 지도 학습 모델(SBERT, SimCSE, PromCSE), 그리고 GPT-4를 직접 이용한 In-context Learning 방식.

### 주요 결과

- **SOTA 달성:** Sim-GPT를 통해 학습된 `PromCSE-RoBERTa-large` 모델은 평균 **85.52** 점을 기록하며, 기존 SOTA인 `PromCSE-RoBERTa-large` (85.10) 대비 **+0.42**의 성능 향상을 보였다.
- **In-context Learning 대비 우위:** GPT-4에 32-shot 프롬프트를 주어 직접 유사도를 측정하게 한 경우(평균 74.45)보다 Sim-GPT 학습 모델(평균 84.75)이 훨씬 높은 성능을 보였다. 이는 제한된 수의 예시보다 대량의 합성 데이터로 학습하는 것이 훨씬 효과적임을 시사한다.
- **안정성:** In-context Learning은 샷(shot)의 선택에 따라 성능 변동이 심했으나, Sim-GPT는 데이터 생성 단계에서의 프롬프트 변화에 매우 안정적인 성능을 유지했다.

## 🧠 Insights & Discussion

### 강점

- **효율적인 지식 전이:** LLM의 고차원적인 의미 이해 능력을 소형 모델로 성공적으로 전이시켰다. 이를 통해 실시간 서비스에 적용 가능한 수준의 추론 속도와 비용 효율성을 확보했다.
- **데이터 증강의 효과:** 단순히 기존 데이터를 사용하는 것이 아니라, LLM을 이용해 '정교하게 설계된' 유사/비유사 쌍을 생성함으로써 모델이 문장 간의 미세한 차이를 학습할 수 있게 하였다.

### 한계 및 비판적 해석

- **GPT-4 의존성:** 데이터 생성 단계에서 GPT-4의 성능에 전적으로 의존한다. 만약 LLM이 생성한 데이터에 편향이 있거나 잘못된 유사도 판단이 포함되어 있다면, 소형 모델이 이를 그대로 학습할 위험이 있다.
- **데이터 생성 비용:** 비록 추론 시에는 비용이 들지 않지만, 371K 규모의 데이터를 생성하기 위한 초기 API 호출 비용은 상당했을 것으로 추측된다. 다만 논문에서는 이를 '일회성 비용'으로 정의하여 정당화하고 있다.

## 📌 TL;DR

본 논문은 STS를 위한 고품질 데이터 부족 문제를 해결하기 위해 **GPT-4를 이용해 대규모의 유사/비유사 문장 쌍을 생성하고, 이를 통해 소형 모델(BERT/RoBERTa)을 학습시키는 Sim-GPT 프레임워크**를 제안한다. 실험 결과, 7개의 STS 벤치마크에서 SOTA 성능을 달성하였으며, LLM을 직접 사용하는 것보다 훨씬 빠르고 비용 효율적이며 안정적인 성능을 보였다. 이 연구는 LLM을 단순한 추론 도구가 아닌 **'고성능 데이터 레이블러'**로 활용하여 특정 태스크의 성능을 극대화할 수 있음을 보여주었다.
