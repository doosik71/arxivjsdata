# Large Language Models and Multimodal Retrieval for Visual Word Sense Disambiguation

Anastasia Kritharoula, Maria Lymperaiou and Giorgos Stamou (2023)

## 🧩 Problem to Solve

본 논문은 **Visual Word Sense Disambiguation (VWSD)**라는 새로운 과제를 해결하고자 한다. VWSD의 목표는 주어진 문맥(context) 내에서 중의성을 가진 단어(ambiguous word)의 정확한 의미를 파악하고, 여러 후보 이미지 중에서 해당 의미를 가장 잘 나타내는 이미지를 검색하는 것이다.

이 문제의 중요성과 어려움은 다음과 같다.

- **제한적인 문맥**: 중의성을 해결하기 위해 제공되는 문맥이 매우 짧으며, 대부분 단 하나의 단어로 구성되어 있어 검색 모듈이 의존할 수 있는 정보가 극히 적다.
- **세밀한 구분이 필요 (Fine-grained discrimination)**: 후보 이미지들이 중의적 단어의 여러 가지 의미(예: 'Andromeda'가 별자리, 물고기, 나무, 파충류 등일 수 있음)를 모두 포함하고 있어, 매우 정밀한 검색 능력이 요구된다.
- **시각적 편향 (Visual Bias)**: 모델이 중의적 단어 자체보다는 문맥 단어(context word)에 과도하게 의존하여, 단순히 문맥 단어가 포함된 이미지를 선택하는 편향이 발생할 수 있다.
- **희귀 개념의 존재**: 모델이 학습 과정에서 접하지 못한 희귀한 단어가 타겟으로 등장할 경우, 검색 결과의 무작위성이 높아지는 문제가 발생한다.

따라서 본 논문의 목표는 LLM의 지식 기반 능력과 멀티모달 검색 기법을 결합하여 VWSD의 성능을 높이고, 검색 과정에 대한 설명 가능성(Explainability)을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **LLM을 외부 지식 베이스(Knowledge Base)로 활용하여 부족한 문맥 정보를 보강하고, 이를 다양한 검색 관점(멀티모달, 단일 모달, QA)에서 접근하여 최종적으로 랭킹 모델로 통합**하는 것이다.

주요 기여 사항은 다음과 같다.

- **LLM 기반 구문 보강 (Phrase Enhancement)**: 짧은 구문을 LLM에 입력하여 더 상세한 설명으로 확장함으로써, 검색 모델이 중의성을 더 쉽게 해결할 수 있도록 돕는다.
- **다각적 접근 (Unimodal & Multimodal)**: 문제를 텍스트-이미지 검색뿐만 아니라, 텍스트-텍스트(캡셔닝 활용), 이미지-이미지(위키피디아 활용), 그리고 질의응답(QA) 형태로 변환하여 각 모델의 잠재력을 최대한 활용한다.
- **Learn to Rank (LTR) 모델 도입**: 앞서 수행한 다양한 접근 방식에서 추출된 특징들을 결합하여 최종 순위를 결정하는 경량 랭킹 모델을 학습시킨다.
- **Chain-of-Thought (CoT) 프롬프팅**: LLM이 정답을 선택하는 추론 과정을 단계별로 생성하게 함으로써, VWSD 결과에 대한 설명 가능한 근거를 제시한다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구를 바탕으로 한다.

- **Text-Image Retrieval**: Transformer 구조의 도입 이후 CLIP, ALIGN과 같은 대비 학습(Contrastive Learning) 기반의 모델들이 텍스트-이미지 표현 학습의 성능을 혁신적으로 끌어올렸으며, 본 연구에서도 이를 베이스라인으로 사용한다.
- **VWSD**: 최근 SemEval 2023 챌린지를 통해 소개된 과제로, Dadas(2023)의 연구가 주요 비교 대상이 된다.
- **LLM-as-KB**: 기존에는 지식 그래프(Knowledge Graphs)를 통해 지식을 보강했으나, 최근에는 LLM 자체가 방대한 지식을 내재하고 있다는 점을 이용해 프롬프팅만으로 지식을 추출하는 패러다임이 주목받고 있다.

## 🛠️ Methodology

본 연구는 VWSD를 해결하기 위해 다음과 같은 6가지 접근 방식을 제안한다.

### 1. Image-Text Similarity Baseline

기존의 사전 학습된 Vision-Language (VL) 트랜스포머(CLIP, ALIGN, BLIP 등)를 사용하여 텍스트 구문 $t$와 후보 이미지 $i$ 사이의 코사인 유사도를 계산한다. 이때, 특정 이미지가 너무 많은 구문과 높은 유사도를 보여 발생하는 편향을 줄이기 위해 penalty factor $p(i)$를 도입한 식은 다음과 같다.
$$\text{score}(t, i) = \text{sim}(t, i) - p(i)$$

### 2. LLMs for Phrase Enhancement

짧은 구문 $t$를 LLM(GPT-3, GPT-3.5, BLOOMZ 등)에 입력하여 상세한 설명인 $t^e$로 확장한다. 사용된 프롬프트 템플릿으로는 "What is $\langle\text{phrase}\rangle$?", "Describe $\langle\text{phrase}\rangle$." 등이 있으며, 보강된 구문을 사용하여 검색 점수를 다시 계산한다.
$$\text{score}(t^e, i) = \text{sim}(t^e, i) - p(i)$$

### 3. Image Captioning for Text Retrieval

이미지를 텍스트로 변환하여 단일 모달(unimodal) 문제로 접근한다. BLIP-Captions나 GiT 모델을 사용하여 각 후보 이미지 $i$에 대한 캡션 $c_i$를 생성한 뒤, 구문 $t$(또는 보강된 $t^e$)와 캡션 $c_i$ 사이의 유사도를 계산한다.

### 4. Wikipedia & Wikidata Image Retrieval

외부 웹 지식을 이용한 이미지-이미지 검색 방식이다. Wikipedia API를 통해 구문 $t$와 관련된 대표 이미지 $i^w$를 검색하고, 이 $i^w$와 후보 이미지 $i$ 간의 시각적 유사도 $\text{score}(i^w, i)$를 측정한다.

### 5. Learn to Rank (LTR)

위의 4가지 방식에서 얻은 점수들을 특징(feature)으로 사용하여 최종 이미지를 선택하는 경량 랭킹 모델을 학습시킨다. LightGBM의 `LGBMRanker`를 사용하며, 입력 특징으로는 $\text{score}(t, i)$, $\max(\text{score})$, $\text{mean}(\text{score})$ 및 이들의 차이 등이 사용된다.

### 6. Question-Answering for VWSD and CoT

VWSD를 다지선다형 QA 문제로 변환한다. 질문에 구문 $t$를 넣고, 선택지(A~J)에는 각 후보 이미지의 캡션 $c_i$를 배치하여 GPT-3.5-turbo가 정답을 고르게 한다. 특히 **Chain-of-Thought (CoT)** 프롬프팅("Let's think step by step")을 적용하여 모델이 왜 특정 이미지를 선택했는지에 대한 논리적 추론 과정을 출력하게 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: VWSD 데이터셋 (Train: 12,869개, Test: 463개). 각 샘플은 10개의 후보 이미지를 포함한다.
- **평가 지표**: Accuracy 및 Mean Reciprocal Rank (MRR).

### 주요 결과

- **LLM 보강 효과**: GPT-3 및 GPT-3.5-turbo와 같은 대규모 모델을 사용했을 때 베이스라인 대비 성능이 뚜렷하게 향상되었다. 반면, 6.7B 이하의 소규모 모델은 희귀 단어에 대한 지식이 부족하여 성능 향상이 미비하거나 오히려 하락하는 경향을 보였다.
- **단일 모달 검색**: 이미지 캡셔닝 기반의 텍스트 검색은 유의미한 성능을 보였으나, 시각 $\rightarrow$ 텍스트 변환 과정에서의 정보 손실로 인해 멀티모달 검색보다는 성능이 낮았다. 이미지-이미지 검색(Wikipedia)은 가장 낮은 성능을 기록했다.
- **LTR 모델의 우수성**: 단일 방식으로는 성적이 낮았던 텍스트/이미지 검색 특징들이 LTR 모델에 통합되었을 때 최종 성능을 높이는 데 기여했다. 모든 특징을 결합한 LTR 모델이 가장 높은 성능을 달성하였다.
- **CoT의 효용성**: CoT 프롬프팅은 특히 세밀한 구분이 필요한 사례에서 정답률을 높였으며, 모델의 내부 추론 과정을 통해 검색 결과의 타당성을 검토할 수 있게 했다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 계산 비용이 많이 드는 VL 모델의 파인튜닝 대신, **프롬프팅 기반의 지식 보강과 경량 랭킹 모델(LTR)**을 결합함으로써 효율적으로 SOTA 수준의 성능을 달성했다. 특히 LTR 모델은 CPU만으로도 20분 내에 학습이 가능하여 실용성이 매우 높다.

### 한계 및 비판적 해석

- **LLM의 환각 (Hallucination)**: LLM이 구문을 보강할 때 잘못된 정보를 생성할 위험이 있으며, 본 논문에서는 이를 명시적으로 탐지하거나 해결하는 메커니즘이 부족하다.
- **캡셔닝 의존성**: QA 방식의 경우 이미지 캡셔너가 이미지의 핵심 특징(예: 'cimarron' 양의 뿔)을 제대로 포착하지 못하면 LLM이 아무리 뛰어나도 정답을 맞출 수 없는 한계가 있다.
- **언어 제한**: 영어 데이터셋에 대해서만 실험이 진행되어, 다른 언어에서의 범용성은 검증되지 않았다.

## 📌 TL;DR

본 논문은 중의적 단어가 포함된 구문에서 적절한 이미지를 찾는 **Visual Word Sense Disambiguation (VWSD)** 과제를 해결하기 위해, **LLM을 통한 문맥 보강 $\rightarrow$ 다각적 검색 특징 추출 $\rightarrow$ 경량 랭킹 모델(LTR) 통합** 파이프라인을 제안한다. 특히 대규모 LLM의 지식을 활용한 구문 확장과 CoT 프롬프팅을 통한 설명 가능한 추론 과정이 성능 향상에 핵심적인 역할을 함을 입증하였다. 이 연구는 향후 LLM의 외부 지식을 멀티모달 검색의 정밀도를 높이는 데 활용하는 방향성을 제시한다.
