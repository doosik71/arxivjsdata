# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi

## 🧩 Problem to Solve

기존 Vision-Language Pre-training (VLP) 모델은 다음 두 가지 주요 한계점을 가지고 있습니다.

1. **모델 아키텍처 관점:** 대부분의 모델은 이해 기반(understanding-based) 작업(예: 이미지-텍스트 검색)이나 생성 기반(generation-based) 작업(예: 이미지 캡셔닝) 중 하나에만 특화되어 있으며, 양쪽 모두에서 뛰어난 성능을 보이지 못했습니다. 인코더 기반 모델은 텍스트 생성 작업에, 인코더-디코더 모델은 이미지-텍스트 검색 작업에 적용하기 어렵습니다.
2. **데이터 관점:** 대부분의 최첨단 VLP 모델은 웹에서 수집된 대규모의 노이즈가 많은 이미지-텍스트 쌍(noisy image-text pairs)으로 사전 학습됩니다. 데이터셋 규모를 확장하여 성능 향상을 얻었음에도 불구하고, 웹 텍스트의 노이즈는 시각-언어 학습에 최적이지 않으며 모델의 학습을 저해할 수 있습니다.

## ✨ Key Contributions

BLIP은 이러한 한계점을 해결하기 위해 다음과 같은 두 가지 주요 기여를 합니다.

- **Multimodal Mixture of Encoder-Decoder (MED) 모델 아키텍처 제안:**
  - 이 모델은 단일 유니파이드(unified) 아키텍처로, unimodal encoder, image-grounded text encoder, image-grounded text decoder의 세 가지 기능으로 유연하게 작동할 수 있습니다.
  - 이해 기반 작업과 생성 기반 작업 모두에 효과적으로 전이 학습할 수 있도록 설계되었습니다.
  - Image-Text Contrastive (ITC), Image-Text Matching (ITM), Image-conditioned Language Modeling (LM)의 세 가지 시각-언어 목표로 공동 사전 학습됩니다.
- **Captioning and Filtering (CapFilt) 데이터 부트스트래핑(bootstrapping) 방법 제안:**
  - 노이즈가 많은 웹 이미지-텍스트 데이터셋을 효과적으로 활용하기 위한 새로운 방법입니다.
  - `Captioner` 모듈은 웹 이미지에 대한 합성 캡션($T_{s}$)을 생성합니다.
  - `Filter` 모듈은 원본 웹 텍스트($T_{w}$)와 합성 텍스트($T_{s}$) 모두에서 노이즈가 많은 캡션을 제거합니다.
  - 이를 통해 더 깨끗하고 고품질의 학습 데이터셋을 구축합니다.
- **최첨단 성능 달성:** 이미지-텍스트 검색, 이미지 캡셔닝, VQA(Visual Question Answering) 등 광범위한 시각-언어 작업에서 새로운 최첨단(SOTA) 결과를 달성했습니다. 또한, zero-shot 방식으로 비디오-언어 작업에 직접 전이했을 때 강력한 일반화 능력을 보였습니다.

## 📎 Related Works

- **Vision-Language Pre-training (VLP):**
  - CLIP (Radford et al., 2021), ALBEF (Li et al., 2021a), SimVLM (Wang et al., 2021) 등 대규모 웹 이미지-텍스트 쌍으로 사전 학습하는 기존 VLP 방법들이 참조됩니다. 이 연구들은 웹 데이터의 노이즈 문제를 간과하고 스케일업을 통해 성능을 개선했습니다.
  - 단일 프레임워크에서 이해 및 생성 작업을 통합하려는 UNITER, VL-T5, Oscar 등의 시도가 있었으나, 인코더 기반 또는 인코더-디코더 기반 모델 모두 양쪽 유형의 작업에서 뛰어나지 못했습니다.
- **Knowledge Distillation (KD):**
  - CapFilt는 `Captioner`가 의미론적으로 풍부한 합성 캡션을 통해 지식을 증류(distill)하고, `Filter`가 노이즈 캡션을 제거하여 지식을 증류하는 VLP 맥락에서의 효과적인 KD 방식으로 해석될 수 있습니다.
- **Data Augmentation (DA):**
  - 자연어 처리(NLP) 분야에서 생성 언어 모델을 사용하여 합성 데이터를 만드는 연구들이 있었으나, BLIP은 이를 대규모 시각-언어 사전 학습에 적용하여 합성 캡션의 이점을 보여줍니다.

## 🛠️ Methodology

BLIP은 새로운 모델 아키텍처인 MED와 데이터 부트스트래핑 방법인 CapFilt를 결합하여 노이즈가 많은 이미지-텍스트 쌍으로부터 학습합니다.

1. **모델 아키텍처 (Multimodal Mixture of Encoder-Decoder, MED):**

   - **이미지 인코더:** Visual Transformer (ViT)를 사용하며, 이미지를 패치로 나누어 임베딩 시퀀스로 인코딩합니다.
   - **텍스트 트랜스포머:** BERT (Devlin et al., 2019)를 기반으로 하며, 세 가지 기능으로 작동합니다.
     - **Unimodal encoder:** 이미지와 텍스트를 개별적으로 인코딩하여 시각 및 언어 표현의 정렬을 학습합니다.
     - **Image-grounded text encoder:** 텍스트 인코더 블록의 self-attention (SA) 레이어와 feed forward network (FFN) 사이에 cross-attention (CA) 레이어를 추가하여 시각적 정보를 주입합니다. 이미지-텍스트 쌍의 멀티모달 표현을 학습합니다.
     - **Image-grounded text decoder:** Image-grounded text encoder에서 양방향(bi-directional) self-attention 레이어를 인과적(causal) self-attention 레이어로 대체하여 주어진 이미지에 대한 캡션 생성을 수행합니다.
   - 텍스트 인코더와 디코더는 SA 레이어를 제외한 모든 파라미터를 공유하여 학습 효율성을 높입니다.

2. **사전 학습 목표:**

   - **Image-Text Contrastive Loss (ITC):** Unimodal encoder를 활성화하여 긍정적인 이미지-텍스트 쌍이 유사한 표현을 가지도록 하여 시각 및 언어 특징 공간을 정렬합니다. 모멘텀 인코더와 소프트 레이블을 사용합니다.
   - **Image-Text Matching Loss (ITM):** Image-grounded text encoder를 활성화하여 이미지-텍스트 간의 미세한 정렬을 포착하는 멀티모달 표현을 학습합니다. 매칭 여부를 예측하는 이진 분류 작업이며, 정보성이 높은 negative pair를 위해 hard negative mining을 사용합니다.
   - **Language Modeling Loss (LM):** Image-grounded text decoder를 활성화하여 주어진 이미지에 대한 텍스트 설명을 자동회귀 방식(autoregressive manner)으로 생성하도록 학습합니다. MLM(Masked Language Model) 대신 LM을 사용하여 시각 정보를 일관된 캡션으로 변환하는 일반화 능력을 부여합니다.

3. **CapFilt (Captioning and Filtering):**
   - 사전 학습된 MED 모델로 `Captioner`와 `Filter`를 초기화한 후, COCO 데이터셋과 같은 소규모의 고품질 인간 주석 데이터셋에서 개별적으로 파인튜닝합니다.
   - **Captioner:** Image-grounded text decoder로서, 웹 이미지($I_{w}$)가 주어지면 nucleus sampling을 사용하여 합성 캡션($T_{s}$)을 생성합니다. 다양성을 확보하는 데 중점을 둡니다.
   - **Filter:** Image-grounded text encoder로서, 이미지와 텍스트의 매칭 여부를 학습합니다. ITM 헤드가 이미지에 매칭되지 않는다고 예측하면 원본 웹 텍스트($T_{w}$)와 합성 텍스트($T_{s}$) 모두에서 해당 텍스트를 노이즈로 간주하여 제거합니다.
   - 최종적으로 필터링된 이미지-텍스트 쌍과 인간 주석 쌍을 결합하여 새로운 데이터셋을 구성하고, 이 데이터셋으로 새로운 BLIP 모델을 사전 학습합니다.

## 📊 Results

BLIP은 광범위한 시각-언어 다운스트림 작업에서 최첨단 성능을 달성했습니다.

- **CapFilt의 효과:**
  - `Captioner`와 `Filter`를 함께 적용했을 때, 원본의 노이즈가 많은 웹 텍스트를 사용한 경우보다 이미지-텍스트 검색 및 이미지 캡셔닝에서 상당한 성능 향상을 보였습니다.
  - 더 큰 데이터셋(129M 이미지)과 더 큰 비전 백본(ViT-L)을 사용할 때 CapFilt의 성능 향상 효과는 더욱 커져, 방법론의 확장성을 입증했습니다.
- **합성 캡션 생성의 다양성:**
  - 확률적 디코딩 방식인 nucleus sampling으로 생성된 합성 캡션이 결정론적 방식인 beam search보다 더 높은 노이즈 비율에도 불구하고 더 나은 성능을 달성했습니다. 이는 nucleus sampling이 모델에 더 새롭고 다양한 정보를 제공하기 때문으로 분석됩니다.
- **파라미터 공유 및 분리:**
  - 사전 학습 시 텍스트 인코더와 디코더 간에 self-attention (SA) 레이어를 제외한 모든 파라미터를 공유하는 것이 가장 효율적이고 성능이 좋았습니다.
  - CapFilt 과정에서 `Captioner`와 `Filter`의 파라미터를 분리하는 것이 확인 편향(confirmation bias)을 줄여 성능을 향상시켰습니다.
- **최첨단 성능 달성:**
  - **이미지-텍스트 검색:** COCO 및 Flickr30K에서 기존 ALBEF 모델보다 평균 Recall@1에서 +2.7%p 향상된 SOTA를 달성했습니다. Zero-shot 검색에서도 큰 폭으로 기존 모델들을 능가했습니다.
  - **이미지 캡셔닝:** NoCaps 및 COCO Caption에서 14M 이미지로 사전 학습 시 기존 모델들을 능가하며, 129M 이미지 사용 시 LEMON (200M 이미지, 객체 탐지기 사용)과 같은 대규모/복잡한 모델과 경쟁력 있는 성능을 보였습니다.
  - **VQA (Visual Question Answering):** VQA2.0 데이터셋에서 ALBEF보다 +1.64%p 높은 SOTA를 기록했습니다.
  - **NLVR$_2$ (Natural Language Visual Reasoning):** 대부분의 기존 방법론을 능가했으며, ALBEF가 추가적인 맞춤형 사전 학습을 수행했을 때와 비슷한 성능을 보였습니다.
  - **VisDial (Visual Dialog):** VisDial v1.0 검증 세트에서 SOTA 성능을 달성했습니다.
  - **Zero-shot 비디오-언어 작업 전이:** 비디오의 시간적 정보는 무시하고 정적 프레임만 사용했음에도 불구하고, 텍스트-투-비디오 검색 및 비디오 QA에서 강력한 zero-shot 일반화 능력을 보여주며, 일부 경우 파인튜닝된 모델보다 우수한 성능을 보였습니다.

## 🧠 Insights & Discussion

- **데이터 품질의 중요성:** CapFilt의 성공은 단순히 데이터셋의 규모를 늘리는 것을 넘어, 웹 데이터의 노이즈를 효과적으로 처리하여 데이터 품질을 향상시키는 것이 VLP 성능에 결정적인 역할을 한다는 것을 보여줍니다.
- **다양한 합성 캡션의 가치:** `Captioner`가 다양하고 새로운 정보를 담은 합성 캡션(nucleus sampling 사용)을 생성하는 것이 성능 향상에 매우 중요함을 확인했습니다. 이는 모델이 더 풍부하고 이전에 보지 못했던 지식을 학습하는 데 도움이 됩니다.
- **최적의 파라미터 공유 전략:** MED 모델의 사전 학습에서 인코딩 및 디코딩 작업의 본질적 차이를 반영하여 self-attention 레이어를 분리하고 다른 레이어를 공유하는 것이 효율성과 성능 모두에 이점을 제공합니다. CapFilt 과정에서 `Captioner`와 `Filter`를 파라미터 공유 없이 개별적으로 파인튜닝하는 것은 확인 편향을 방지하고 성능을 극대화하는 데 필수적입니다.
- **통합 모델의 유연성:** MED 아키텍처는 이해 기반 및 생성 기반 V-L 작업 모두에 유연하게 전이될 수 있는 강력한 기반을 제공하며, 이는 기존 모델들의 한계를 극복하는 데 기여합니다.
- **향후 연구 방향:** 여러 차례의 데이터 부트스트래핑, 이미지당 여러 개의 합성 캡션 생성, `Captioner`와 `Filter` 앙상블 등을 통해 BLIP의 성능을 더욱 향상시킬 수 있는 잠재력이 있습니다. 또한 비디오-언어 작업에서 시간적 모델링을 통합하면 더 큰 개선이 가능할 것입니다.
- **학습 전략:** 부트스트랩된 데이터셋으로 **새로운 모델**을 사전 학습해야 하며, 기존 사전 학습 모델을 계속 이어서 학습하는 것은 성능 향상에 도움이 되지 않습니다. 이는 지식 증류의 일반적인 관행과 일치합니다.

## 📌 TL;DR

BLIP은 이미지-언어 사전 학습(VLP)에서 **이해 및 생성 작업 모두에 유연하게 작동하지 못하고 노이즈가 많은 웹 데이터에 의존하는 기존 모델의 한계**를 해결합니다. 이를 위해 **MED (Multimodal Mixture of Encoder-Decoder)**라는 새로운 통합 모델 아키텍처를 제안하여 다양한 V-L 작업에 유연하게 전이될 수 있도록 하고, **CapFilt (Captioning and Filtering)**라는 데이터 부트스트래핑 방법을 도입하여 `Captioner`로 합성 캡션을 생성하고 `Filter`로 노이즈 캡션을 제거하여 데이터 품질을 향상시킵니다. 결과적으로 BLIP은 이미지-텍스트 검색, 캡셔닝, VQA 등 **광범위한 시각-언어 작업에서 최첨단 성능을 달성**했으며, 비디오-언어 작업에 대한 **강력한 zero-shot 일반화 능력**을 보여주며 모델 아키텍처의 유연성과 정제된 데이터의 중요성을 입증했습니다.
