# Optimal path for Biomedical Text Summarization Using Pointer-GPT

Hyunkyung Han, Jaesik Choi (n.d.)

## 🧩 Problem to Solve

본 논문은 임상 의사들이 환자의 상태를 효율적으로 파악할 수 있도록 돕는 생의학 텍스트 요약(Biomedical Text Summarization)의 정확성을 높이는 문제를 다룬다.

전통적으로 텍스트 요약은 긴 문서를 짧게 압축하는 Transformer 기반 모델, 특히 GPT 모델을 통해 수행되어 왔다. 그러나 이러한 생성적 모델들은 다음과 같은 치명적인 한계를 가진다.

- **사실적 오류(Factual Errors) 생성**: 모델이 학습된 통계적 확률에 의존하여 실제와 다른 내용을 생성하는 환각 현상이 발생한다.
- **문맥 부족(Lack of Context)**: 의학적 데이터의 특수한 문맥을 충분히 유지하지 못하는 경향이 있다.
- **단어의 과도한 단순화(Oversimplification)**: 전문적인 의학 용어를 지나치게 단순화하여 정보의 손실이 발생한다.

따라서 본 연구의 목표는 GPT 모델의 구조를 수정하여 원문의 핵심 가치와 전문 용어를 그대로 보존하면서도 정확한 요약을 제공하는 Pointer-GPT 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GPT 모델의 기존 Attention 메커니즘을 **Pointer Network**로 교체하는 것이다.

일반적인 Attention은 입력 벡터의 가중 합을 통해 새로운 벡터를 생성하지만, Pointer Network는 입력 시퀀스의 특정 위치를 직접 '가리킴(pointing)'으로써 원문의 단어를 그대로 추출할 수 있게 한다. 이를 통해 생성 모델이 임의로 단어를 만들어내는 대신, 의료 기록 내의 핵심 단어와 구절을 명시적으로 선택하게 하여 사실적 오류를 줄이고 원문의 의도를 보존하고자 하였다.

## 📎 Related Works

논문에서는 텍스트 요약과 관련된 세 가지 주요 접근 방식을 설명한다.

1. **Neural Abstractive Summarization**: Local Attention 기반 모델을 통해 입력 문장의 중요한 정보를 함축적으로 요약하는 방식이다. 하지만 문법적으로 틀린 요약을 생성하는 경우가 있으며, 주로 문장 단위 요약에 머물러 단락 단위 요약으로 확장하는 데 한계가 있다.
2. **Seq2seq (Sequence-to-Sequence)**: Encoder-Decoder 구조를 통해 입력 텍스트를 잠재 표현(latent representation)으로 변환하고 요약문을 생성한다. 하지만 대규모 말뭉치의 통계적 특성에 의존하기 때문에, 통계적으로는 그럴듯하지만 사실 관계가 틀린 요약을 생성할 위험이 있다.
3. **Generative Pre-training Transformer (GPT)**: 방대한 데이터셋을 통해 사전 학습되어 유창하고 읽기 쉬운 요약문을 생성하는 능력이 뛰어나다. 하지만 앞서 언급한 바와 같이 의료 분야와 같이 정확성이 생명인 영역에서는 사실적 오류라는 한계가 명확하다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인

제안된 **Pointer-GPT**는 오픈소스 모델인 GPT-2를 기반으로 하며, 모델 내부의 Attention 메커니즘을 Pointer Network로 대체한 구조를 가진다.

### 주요 구성 요소 및 작동 원리

- **Pointer Network의 역할**: 요약 작업에 가장 관련성이 높은 단어를 입력 텍스트에서 직접 선택하는 역할을 수행한다.
- **추론 절차**:
    1. Encoder와 Decoder의 Hidden State를 입력으로 받는다.
    2. 입력 텍스트에 존재하는 단어들에 대해 확률 분포(Probability Distribution)를 계산한다.
    3. 이 확률 분포를 바탕으로 요약문에 포함될 가장 적절한 단어를 원문에서 직접 선택하여 생성한다.

### 상세 구현 사항

- **기반 모델**: GPT-2 (Open-source)
- **수정 사항**: 기존 GPT-2의 Attention 레이어를 Pointer Network 구조로 변경하여, 생성 시 Vocabulary에서 단어를 샘플링하는 대신 입력 시퀀스의 인덱스를 참조하도록 설계하였다.
- **수식**: 논문 내에 구체적인 방정식(Equation)은 명시되지 않았으나, Pointer Network의 일반적인 메커니즘에 따라 입력 단어 $w_i$에 대한 선택 확률 $P(w_i)$를 계산하여 이를 기반으로 요약문을 구성하는 방식을 취한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 의료 사례 연구인 "Case Study: 33-Year-Old Female Presents with Chronic SOB and Cough"를 포함한 의료 케이스 데이터셋을 사용하였다.
- **비교 대상**: 원본 GPT-2 모델.
- **평가 지표**: $\text{ROUGE-1}$ 및 $\text{ROUGE-2}$ (Recall-Oriented Understudy for Gisting Evaluation) 점수를 사용하여 요약의 품질을 정량적으로 평가하였다.

### 정량적 결과

실험 결과, Pointer-GPT가 모든 지표에서 GPT-2를 크게 상회하는 성능을 보였다.

| Algorithm | Metric | Precision | Recall | F-measure |
| :--- | :--- | :---: | :---: | :---: |
| **GPT-2** | $\text{Rouge-1}$ | 0.2857 | 0.3529 | 0.3157 |
| | $\text{Rouge-2}$ | 0.1 | 0.125 | 0.1111 |
| **Pointer-GPT** | $\text{Rouge-1}$ | 1.0 | 0.4705 | 0.6399 |
| | $\text{Rouge-2}$ | 0.8571 | 0.375 | 0.5217 |

$\text{ROUGE-1}$의 F-measure 기준, GPT-2(0.3157) 대비 Pointer-GPT(0.6399)가 약 2배 이상의 성능 향상을 보였으며, 특히 $\text{ROUGE-2}$에서는 0.1111에서 0.5217로 매우 비약적인 상승을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 생성 모델의 고질적인 문제인 환각(Hallucination) 현상을 Pointer Network라는 구조적 변경을 통해 효과적으로 억제하였다. 특히 의료 데이터와 같이 단어 하나하나의 정확성이 중요한 도메인에서 원문의 텍스트를 그대로 유지하는 Pointer 메커니즘이 유효함을 입증하였다.

### 한계 및 비판적 해석

1. **데이터셋의 규모**: 본문에서 언급된 데이터셋이 특정 케이스 스터디(Case Study)를 중심으로 이루어져 있어, 일반적인 의료 기록 전체에 대해 일반화(Generalization)가 가능한지 확인하기에는 데이터의 양이 부족해 보인다.
2. **상세 방법론 부족**: Attention을 구체적으로 어떻게 Pointer Network로 대체했는지에 대한 아키텍처 다이어그램이나 수식, 손실 함수(Loss Function)의 변경 사항이 명시되지 않아 재현성이 떨어진다.
3. **비교 대상의 제한**: 최신 의료 특화 언어 모델(예: BioBERT, ClinicalBERT 기반 요약 모델)과의 비교 없이 GPT-2와만 비교한 점은 결과의 객관성을 다소 약화시킨다.

## 📌 TL;DR

본 논문은 생의학 텍스트 요약 시 발생하는 사실적 오류와 문맥 손실을 해결하기 위해, GPT-2의 Attention 메커니즘을 **Pointer Network**로 대체한 **Pointer-GPT**를 제안하였다. 실험 결과, $\text{ROUGE}$ 점수에서 기존 GPT-2보다 월등한 성능 향상을 보였으며, 이는 의료 기록 요약 시스템(EMR)에서 임상 의사들에게 더 정확하고 신뢰할 수 있는 정보를 제공할 가능성을 제시한다.
