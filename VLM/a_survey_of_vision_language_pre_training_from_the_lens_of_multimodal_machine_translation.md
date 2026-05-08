# A Survey of Vision-Language Pre-training from the Lens of Multimodal Machine Translation

Jeremy Gwinnup, Kevin Duh (2023)

## 🧩 Problem to Solve

본 논문은 최근 급격히 발전하고 있는 Vision-Language (V-L) 사전 학습 모델들을 Multimodal Machine Translation (MMT) 관점에서 분석하고 정리하는 것을 목표로 한다.

최근 BERT나 GPT와 같은 대규모 언어 모델(LLM)의 성공으로 인해, 대규모 데이터셋으로 사전 학습을 수행한 후 특정 태스크에 맞춰 미세 조정(fine-tuning)하는 패러다임이 정착되었다. CLIP과 같은 V-L 모델들이 이미지 캡셔닝이나 시각적 질의응답(VQA) 등에서 뛰어난 성능을 보이고 있음에도 불구하고, 이미지나 비디오의 시각적 정보를 활용하여 텍스트 간 번역 성능을 높이는 MMT 분야에 이러한 모델들을 적용한 연구는 상대적으로 부족한 실정이다.

따라서 본 연구는 MMT의 관점에서 V-L 사전 학습의 전반적인 지형을 조사하고, 공통적인 아키텍처, 사전 학습 목표(pre-training objectives), 데이터셋을 요약함으로써 MMT의 발전을 위해 무엇이 필요한지 고찰하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **MMT 관점의 V-L 사전 학습 체계화**: MMT 성능 향상을 위해 활용 가능한 V-L 사전 학습 모델들의 아키텍처와 학습 목적을 분류하고 분석하였다.
2. **Grounding 개념의 정의 및 분석**: 텍스트와 시각적 정보 사이의 연관성 정도를 나타내는 Grounding(근거 제시) 개념을 'Strong'과 'Weak'으로 구분하여 제시하고, 이것이 MMT 태스크에 미치는 영향을 논의하였다.
3. **데이터셋 Grounding 측정 방법 제안**: CLIP의 Zero-shot 분류 능력을 활용하여, 특정 MMT 데이터셋이 얼마나 강하게 Grounding 되어 있는지를 정량적으로 평가하는 실험적 방법론을 제시하였다.

## 📎 Related Works

기존의 MMT 연구들은 주로 Multi30k와 같이 규모가 작은 병렬 말뭉치와 이미지 쌍을 사용하여 모델을 처음부터 학습(train from scratch)시키는 방식에 집중해 왔다. 하지만 이러한 방식은 데이터 수집 비용이 높고 모델의 일반화 능력이 떨어진다는 한계가 있다.

반면, 최근의 V-L 사전 학습 연구들은 대규모의 이미지-텍스트 쌍을 활용하여 범용적인 표현력을 학습하는 방향으로 나아가고 있다. 본 논문은 기존의 MMT 연구와 최신 V-L 사전 학습 연구 사이의 간극을 메우기 위해, 두 분야의 접점을 찾고 최신 V-L 모델들을 MMT의 프론트엔드로 활용할 가능성을 탐색한다.

## 🛠️ Methodology

### 1. Multimodal Machine Translation (MMT) 정의

MMT는 시각적 신호(이미지 또는 비디오) $v$와 소스 언어 문장 $x$가 주어졌을 때, 타겟 언어 번역문 $y$를 생성하는 작업이다. 수학적으로는 다음과 같은 조건부 확률 분포를 학습하는 것으로 정의한다.

$$P(y|x, v)$$

이 모델의 성공 여부는 시각적 정보 없이 텍스트만으로 번역하는 베이스라인 $P(y|x)$보다 성능이 우수한지로 판단한다.

### 2. 사전 학습 목표 (Pre-training Objectives)

본 논문은 V-L 모델의 사전 학습 목표를 세 가지 범주로 분류한다.

* **Masking**: 입력 시퀀스에서 일부 토큰을 제거하고, 모델이 누락된 토큰을 예측하게 하는 방식이다. ViT(Vision Transformer)는 이미지를 패치 단위의 토큰으로 변환하여 텍스트 토큰과 결합함으로써 이 방식을 적용한다.
* **Matching**: 텍스트와 시각적 쌍 사이의 임베딩 공간 거리를 최소화하는 방식이다. CLIP과 같은 모델은 Contrastive Loss를 사용하여 올바른 쌍은 가깝게, 잘못된 쌍은 멀게 배치하도록 학습한다.
* **Dataset-specific Supervised Objective**: 특정 데이터셋의 목적에 맞춘 지도 학습 방식이다. 예를 들어, 이미지 입력만으로 캡션을 생성하는 Visual Captioning(VC)이나, 이미지와 질문 텍스트를 통해 정답을 도출하는 Visual Question Answering(VQA) 등이 있다.

### 3. 모델 아키텍처 및 퓨전 전략

V-L 모델들은 주로 ViT나 CNN을 이미지 인코더로, BERT 스타일의 아키텍처를 텍스트 인코더로 사용한다. 핵심 차이점은 두 모달리티를 결합하는 **Fusion 전략**에 있다.

* **Contrastive Learning**: CLIP, ALIGN 등은 이미지와 텍스트 인코더를 각각 두고 두 임베딩의 유사도를 극대화한다.
* **Early Fusion**: ViT 계열은 이미지 패치 토큰과 텍스트 토큰을 단순히 연결(concatenate)하여 하나의 트랜스포머에 입력한다.
* **Interleaved/Gated Layers**: Flamingo는 텍스트 모델의 레이어를 동결(frozen)시킨 상태에서, 그 사이에 시각적 정보를 통합하는 Gated Vision Layer를 삽입한다.
* **Query Transformer**: BLIP-2는 동결된 이미지 인코더와 텍스트 인코더 사이에 가벼운 Query Transformer를 두어 시각적 특징을 텍스트 모델이 이해할 수 있는 형태로 변환하여 전달한다.

## 📊 Results

### 1. Grounding 측정 실험 설계

저자들은 V-L 모델이 MMT 데이터셋의 특성에 따라 다르게 반응하는지 확인하기 위해, CLIP을 이용한 Zero-shot 재순위화(re-ranking) 실험을 수행하였다.

하나의 이미지/비디오에 대해 다음 네 가지 후보 중 CLIP 스코어가 가장 높은 것을 선택하게 한다.

1. **Reference**: 실제 정답 캡션/문장
2. **Neighbor**: 동일한 이미지/비디오에서 추출된 다른 캡션/문장
3. **In-Domain**: 동일 데이터셋 내의 다른 무작위 문장
4. **Out-Domain**: 전혀 다른 코퍼스(예: Europarl)에서 추출한 문장

### 2. 실험 결과 및 분석

실험 결과는 다음과 같다 (표 4 참조).

* **CoCo 및 VaTeX**: Reference 또는 Neighbor의 선택 비율이 매우 높게 나타났다. 이는 해당 데이터셋들이 이미지/비디오를 직접 설명하는 캡셔닝 기반으로 구축되어 **Strong Grounding** 특성을 가졌음을 의미한다.
* **How2**: Reference가 선택되는 비율이 약 $51\%$로, Neighbor($33\%$)와 상대적으로 차이가 적었다. 이는 How2의 텍스트(나레이션)가 특정 시점의 영상보다는 비디오 전체의 맥락과 더 밀접하며, 특정 구간의 시각 정보와 텍스트 간의 연관성이 낮음(**Weak Grounding**)을 시사한다.
* **Diffusion Mirror-corpus**: Stable Diffusion으로 생성한 이미지의 경우, 생성 시 사용된 프롬프트(Reference)가 압도적으로 높게 선택되었다.

## 🧠 Insights & Discussion

본 논문의 분석을 통해 도출된 주요 통찰은 다음과 같다.

첫째, **Grounding의 수준에 따라 V-L 모델의 효용성이 달라진다.** 대부분의 V-L 사전 학습 모델은 이미지 캡셔닝과 같이 Strong Grounding 데이터셋으로 학습된다. 따라서 이러한 모델들을 How2와 같은 Weak Grounding MMT 태스크에 바로 적용했을 때 기대만큼의 성능 향상이 없을 수 있다는 가능성을 제기한다.

둘째, **데이터 가용성의 문제**이다. MMT를 위한 고품질의 다국어-시각 데이터셋은 수작업 주석 비용으로 인해 규모가 작다. 이를 해결하기 위해 Conceptual Captions와 같은 대규모 단일 언어 데이터셋을 기계 번역으로 합성하여 학습에 활용하려는 시도가 있으나, 여전히 데이터의 품질과 정렬 문제가 남아 있다.

셋째, **단순한 결합 이상의 전략이 필요하다.** 단순히 사전 학습된 인코더를 앞단에 배치하는 것을 넘어, MMT 특유의 Weak Grounding 상황을 해결할 수 있는 정교한 퓨전 메커니즘이나 데이터 증강 전략이 연구되어야 한다.

## 📌 TL;DR

본 논문은 V-L 사전 학습 모델들을 Multimodal Machine Translation (MMT)의 관점에서 분석한 서베이 논문이다. V-L 모델의 사전 학습 목표와 아키텍처를 체계적으로 정리하였으며, 특히 텍스트와 시각 정보의 연관성을 뜻하는 **Grounding** 개념을 도입하여 데이터셋별 특성을 분석하였다. CLIP을 이용한 실험을 통해 MMT 데이터셋마다 Grounding 강도가 다름을 증명하였으며, 이는 향후 V-L 모델을 MMT에 적용할 때 데이터셋의 특성에 맞는 전략적 접근이 필요함을 시사한다.
