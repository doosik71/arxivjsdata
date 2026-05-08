# MedBLIP: Fine-tuning BLIP for Medical Image Captioning

Manshi Limbu, Diwita Banerjee (2025)

## 🧩 Problem to Solve

본 연구는 의료 영상 캡셔닝(Medical Image Captioning) 작업에서 범용 시각-언어 모델(Vision-Language Models, VLMs)이 겪는 한계를 해결하고자 한다. BLIP, BLIP-2, Gemini, ViT-GPT2와 같은 최신 모델들은 일반 이미지 데이터셋(예: MS-COCO)에서는 매우 강력한 성능을 보이지만, 전문적인 의료 도메인에 적용했을 때는 지나치게 일반적이거나 부정확한 캡션을 생성하는 경향이 있다.

의료 영상은 일반 영상과 비교하여 시각적 구조와 내용이 크게 다르며, 생성되는 캡션 또한 전문 용어, 해부학적 참조, 임상적 소견 등 매우 특수한 언어 체계를 포함한다. 따라서 고도의 정밀도와 사실적 근거(factual grounding)가 필수적인 임상 환경에서 이러한 모델의 부정확성은 치명적인 문제가 될 수 있다. 본 논문의 목표는 BLIP 모델을 의료 영상 데이터셋인 ROCO를 사용하여 미세 조정(fine-tuning)함으로써, 방사선 영상 캡셔닝의 정확성과 관련성을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 범용 VLM인 BLIP을 의료 도메인에 특화시켜 적응(adaptation)시켰을 때의 효과를 정량적, 정성적으로 분석한 것이다. 주요 기여 사항은 다음과 같다.

- **도메인 특화 미세 조정**: ROCO 데이터셋을 사용하여 BLIP 모델을 미세 조정하고, 이를 통해 의료 영상 캡셔닝 성능을 유의미하게 향상시켰다.
- **광범위한 비교 분석**: 미세 조정된 BLIP을 제로샷(zero-shot) BLIP, BLIP-2, BLIP-2 Instruct, Gemini 1.5 Flash, 그리고 ViT-GPT2와 같은 다양한 Transformer 기반 아키텍처와 비교하여 성능을 검증하였다.
- **해석 가능성 분석**: 디코더의 교차 주의 집중(cross-attention) 맵을 시각화하여, 모델이 생성하는 토큰이 이미지의 어느 영역과 정렬되는지를 분석함으로써 모델의 추론 과정을 해석하였다.
- **구성 요소별 기여도 평가**: Ablation Study를 통해 인코더 전용, 디코더 전용, 그리고 전체 모델 미세 조정의 효과와 효율성을 비교 분석하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계

- **CNN-LSTM 구조**: 초기 캡셔닝 모델들은 CNN을 통해 특징을 추출하고 LSTM으로 텍스트를 생성하였다. 그러나 이러한 방식은 장거리 의존성(long-range dependencies)을 포착하는 능력이 부족하고 국소적인 주의 집중(localized attention) 이상의 맥락적 근거를 제시하지 못하는 한계가 있었다.
- **Transformer 기반 모델**: ViT-GPT2, BLIP-2 등은 ViT 인코더와 GPT 스타일의 자기회귀(autoregressive) 디코더를 사용하여 성능을 높였다. 하지만 이들은 일반 데이터셋으로 학습되었기에 의료 도메인의 어휘, 스타일, 이미지 분포와의 불일치로 인해 성능이 저하된다.
- **BLIP 프레임워크**: BLIP은 대조 학습(contrastive learning), 이미지-텍스트 매칭, 생성적 목적 함수를 단일 아키텍처로 통합하여 제로샷 및 퓨샷(few-shot) 설정에서 강점을 보인다.

### 차별점

본 연구는 단순히 모델을 적용하는 것에 그치지 않고, 의료 분야에서 특히 중요한 **임상적 정확성(clinical correctness)**을 평가하기 위해 모달리티(modality), 측성(laterality), 해부학적 구체성, 진단 정확성을 포함하는 별도의 평가 표를 도입하여 기존의 단순 텍스트 유사도 지표의 한계를 보완하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 연구는 BLIP 모델을 기반으로 의료 영상 캡셔닝 시스템을 구축하였다. BLIP은 기본적으로 ViT-B/16 인코더와 Transformer 기반의 BERT 디코더로 구성된다.

### 훈련 절차 및 세부 설정

- **데이터셋**: ROCO 데이터셋을 사용하였으며, 흉부 X-ray, MRI, CT, 초음파 등 주요 모달리티에 집중하였다. 모든 이미지는 모델 입력 요구 사항에 맞춰 $384 \times 384$ 크기로 조정되었다.
- **학습 설정**:
  - **최적화 알고리즘**: AdamW 옵티마이저를 사용하였다.
  - **학습률(Learning Rate)**: $5 \times 10^{-5}$로 설정하였다.
  - **학습 기법**: Gradient Accumulation(4 steps), 혼합 정밀도(Mixed Precision) 훈련(PyTorch `GradScaler` 이용), 그리고 선형 학습률 스케줄(linear learning rate schedule)을 적용하였다.
  - **종료 조건**: 검증 손실(validation loss)에 기반한 조기 종료(early stopping) 전략을 사용하여 1~3 epoch 동안 학습하였다.
- **손실 함수**: Teacher Forcing 기법과 함께 교차 엔트로피 손실(Cross-Entropy Loss)을 사용하여 학습하였다.

### 주의 집중 시각화 (Attention Interpretability)

모델의 해석 가능성을 높이기 위해, 추론 과정에서 BLIP의 BERT 디코더 최종 레이어로부터 **Decoder Cross-Attention Weights**를 추출하였다. Forward hook을 사용하여 캡처한 attention 텐서를 헤드(head) 전체에 대해 평균낸 후, 이를 히트맵(heatmap) 형태로 원본 이미지 위에 오버레이하여 특정 토큰 생성 시 모델이 이미지의 어느 부분에 집중했는지 분석하였다.

### 평가 지표

- **어휘적/의미적 지표**: CIDEr (n-gram 중첩), SPICE (장면 그래프 기반 의미 유사도), BERTScore (문맥 임베딩 기반 유사도), Cosine Similarity (BioClinicalBERT 임베딩 기반)를 사용하였다.
- **임상적 지표**: 텍스트 지표의 한계를 극복하기 위해 모달리티 인식, 좌우 구분(laterality), 해부학적 위치, 진단 정확성을 평가하는 Clinical Evaluation Table을 작성하였다.

### Ablation Study 설정

미세 조정 범위에 따른 성능 차이를 확인하기 위해 다음 세 가지 설정을 비교하였다.

1. **Full Fine-Tuning**: ViT 인코더와 BERT 디코더를 모두 업데이트한다.
2. **Decoder-Only Fine-Tuning**: ViT 인코더는 동결(frozen)하고 BERT 디코더만 업데이트한다.
3. **Encoder-Only Fine-Tuning**: BERT 디코더는 동결하고 ViT 인코더만 업데이트한다.

## 📊 Results

### 정량적 결과

Table 1의 결과에 따르면, 미세 조정된 BLIP 모델은 모든 지표(CIDEr, SPICE, BERTScore 등)에서 베이스 BLIP 및 다른 베이스라인 모델들보다 우수한 성능을 보였다. 특히 BioClinicalBERT를 이용한 BERTScore와 Cosine Similarity에서 높은 수치를 기록하여 도메인 적응이 의미 있음을 입증하였다.

### 정성적 결과 및 임상적 분석

- **캡션 품질**: 베이스 BLIP은 "a medical scan of the brain"과 같이 매우 일반적인 설명을 생성하는 반면, 미세 조정된 BLIP은 "MRI showing a hyperintense lesion"과 같이 전문 용어와 모달리티 특성을 반영한 캡션을 생성하였다.
- **임상적 한계**: 정량적 지표의 향상에도 불구하고, Table 2의 임상 평가 결과 미세 조정된 모델조차 **환각(hallucination)** 현상을 보였다. 예를 들어, 실제로는 없는 흉수를 언급하거나, 좌우 방향을 잘못 지정하거나, 핵심 병변을 누락하는 경우가 발생하였다.

### 주의 집중 시각화 결과

시각화 결과, 베이스 BLIP은 주의 집중 영역이 매우 분산되어 관련 없는 영역을 참조하는 경향이 있었다. 반면 미세 조정된 모델은 "pleural effusion"과 같은 용어를 생성할 때 폐 영역과 같은 핵심 해부학적 부위에 더 국소적으로 집중하는 양상을 보였다. 그러나 공간적 정렬이 개선되었다고 해서 반드시 임상적 정확성으로 이어지는 것은 아님을 확인하였다.

### Ablation Study 결과

- **Full Fine-Tuning**이 모든 지표에서 가장 높은 성능을 기록하였다.
- **Decoder-Only Fine-Tuning**은 Full Fine-Tuning에 근접하는 경쟁력 있는 성능을 보이면서도, 학습 시간을 약 5% 단축시켜 자원 효율성이 높음을 보여주었다.
- **Encoder-Only Fine-Tuning**은 상대적으로 낮은 성능을 보였는데, 이는 캡셔닝 논리 자체에 대한 도메인 적응(언어적 적응)이 부족했기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 연구는 범용 VLM을 특정 전문 도메인에 맞게 미세 조정하는 것이 텍스트 생성의 품질과 시각적 정렬(visual alignment)을 개선하는 데 효과적임을 보여주었다. 특히 디코더만 미세 조정하는 방식이 효율적인 대안이 될 수 있음을 시사한다.

하지만 가장 중요한 통찰은 **"정량적 지표의 상승이 반드시 임상적 안전성을 보장하지 않는다"**는 점이다. CIDEr나 BERTScore 같은 지표는 기준 문장과의 유사도를 측정할 뿐, 의료 현장에서 치명적일 수 있는 '잘못된 진단(hallucination)'을 잡아내지 못한다. 이는 의료 영상 캡셔닝 모델이 실무에 적용되기 위해서는 단순한 데이터 기반 학습을 넘어, 구조화된 의료 지식(structured medical knowledge)의 통합이나 사후 검증(post-hoc validation) 체계가 필수적임을 의미한다.

## 📌 TL;DR

본 논문은 범용 시각-언어 모델인 BLIP을 ROCO 데이터셋으로 미세 조정하여 의료 영상 캡셔닝 성능을 높이는 방법을 연구하였다. 실험 결과, 도메인 특화 미세 조정은 텍스트의 전문성을 높이고 시각적 주의 집중을 정교하게 만들지만, 여전히 임상적 환각 현상이 발생함을 확인하였다. 결론적으로 전면 미세 조정이 가장 우수한 성능을 보이며, 디코더 전용 미세 조정은 효율적인 대안이 된다. 향후 의료 AI의 신뢰성을 위해서는 단순 지표 최적화보다 임상적 근거를 강제하는 구조적 제약 조건의 도입이 필요하다.
