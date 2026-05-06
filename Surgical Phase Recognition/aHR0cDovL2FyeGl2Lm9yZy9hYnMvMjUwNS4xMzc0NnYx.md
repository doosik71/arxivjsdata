# ReSW-VL: Representation Learning for Surgical Workflow Analysis Using Vision-Language Model

Satoshi Kondo (2025)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 문제는 수술 비디오에서 수술 단계(surgical phase)를 자동으로 분류하는 '수술 워크플로우 분석(surgical workflow analysis)'의 성능 향상이다. 수술 단계 인식 기술은 실시간 수술 지원, 의료 자원 최적화, 수술 교육 및 숙련도 평가, 그리고 환자의 안전성 향상이라는 측면에서 매우 중요한 가치를 지닌다.

기존의 딥러닝 기반 수술 단계 인식 방법들은 일반적으로 개별 프레임에서 공간적 특징(spatial features)을 추출하는 CNN/ViT 모델과, 이를 시계열로 분석하는 Temporal modeling(LSTM, TCN, Transformer 등) 모델을 결합한 2단계 구조를 취한다. 그러나 대부분의 연구가 시계열 모델링에 집중되어 있으며, 정작 특징 추출기(feature extractor)인 CNN을 어떻게 학습시켜 최적의 표현(representation)을 얻을 것인가에 대한 연구, 즉 Representation Learning에 대한 논의는 부족한 실정이다. 따라서 본 논문의 목표는 Vision-Language Model(VLM)과 Prompt Learning을 도입하여 수술 단계 인식에 최적화된 공간적 특징 추출기를 학습시키는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CLIP(Contrastive Language-Image Pre-training)과 같은 Vision-Language Model을 활용하여, 단순한 이미지 분류가 아닌 언어적 맥락이 포함된 Prompt Learning을 통해 이미지 인코더를 미세 조정(fine-tuning)하는 것이다.

주요 기여 사항은 다음과 같다.

1. 수술 단계 인식을 위한 Representation Learning 연구 중 드물게 Vision-Language Model과 Prompt Learning을 결합한 최초의 방법론을 제안하였다.
2. 제안된 ReSW-VL 방법론이 기존의 ImageNet 사전 학습 기반의 일반적인 미세 조정 방식보다 세 가지 주요 수술 데이터셋에서 더 우수한 성능을 보임을 입증하였다.
3. 수술 단계의 순차적 특성을 고려한 $\text{ReSW-VLo}$와 독립적인 특성을 고려한 $\text{ReSW-VLi}$ 두 가지 접근 방식을 통해 데이터셋의 특성에 따른 최적의 학습 전략을 제시하였다.

## 📎 Related Works

기존의 수술 단계 인식 연구들은 주로 다음과 같은 구조를 따른다.

- **공간적 특징 추출기:** ResNet-50이나 ViT-B/16과 같은 모델을 사용하며, 대개 ImageNet 데이터셋으로 사전 학습된 가중치를 사용한 후 수술 데이터로 미세 조정을 수행한다.
- **시계열 모델링:** 추출된 특징 시퀀스를 LSTM, TCN, Transformer 등을 통해 분석하여 최종 단계를 예측한다.

본 논문은 이러한 기존 방식들이 특징 추출기의 학습 방법론을 깊이 있게 다루지 않았다는 점을 지적한다. 기존 방식은 단순히 ImageNet $\rightarrow$ 수술 데이터로 이어지는 전이 학습에 의존하지만, ReSW-VL은 텍스트 인코더와 이미지 인코더의 상호작용을 이용하는 VLM 구조를 도입함으로써 수술 단계의 의미론적 표현력을 높이고자 한다.

## 🛠️ Methodology

제안된 ReSW-VL(Representation learning in Surgical Workflow analysis using a Vision-Language model)은 총 2단계의 학습 파이프라인으로 구성된다.

### 1단계: 공간적 특징 표현 학습 (Representation Learning)

이 단계의 목적은 수술 이미지에서 단계별 특징을 가장 잘 추출하는 이미지 인코더를 학습시키는 것이다.

- **구조:** CLIP의 이미지 인코더(Image Encoder)와 텍스트 인코더(Text Encoder)를 사용한다. 텍스트 인코더는 고정(frozen)시키고, 이미지 인코더와 프롬프트(prompt)만을 학습시킨다.
- **Prompt Learning:** 각 수술 단계 $p$에 대해 프롬프트를 구성한다. 프롬프트는 단계 번호를 나타내는 첫 번째 토큰 $[E]_p$와 학습 가능한 $m$개의 토큰으로 이루어진다. 즉, 모델이 "비디오 내의 $p$번째 수술 단계 이미지"와 같은 의미를 학습하도록 유도한다.
- **학습 절차:**
    1. 이미지 인코더는 입력 이미지를 $d$-차원 벡터로 변환한다.
    2. 텍스트 인코더는 $P$개의 프롬프트를 $P \times d$-차원 벡터로 변환한다.
    3. 이미지 벡터와 $P$개의 텍스트 벡터 간의 내적(inner product)을 계산하여 $P$-차원의 로짓(logit)을 생성한다.
    4. 이 로짓과 실제 정답 레이블 간의 Cross-Entropy 손실 함수를 사용하여 이미지 인코더와 프롬프트를 업데이트한다.

- **두 가지 학습 전략:**
  - **$\text{ReSW-VLi}$ (Independent):** $P$개의 프롬프트 첫 토큰 $[E]_p$를 서로 독립적으로 학습한다.
  - **$\text{ReSW-VLo}$ (Order):** 수술 단계의 순차적 특성을 반영한다. $n$개의 기준 토큰을 학습하고, 나머지 토큰들은 이들의 보간(interpolation)을 통해 생성함으로써 임베딩 공간에서 단계의 순서가 유지되도록 한다.

### 2단계: 시계열 모델링 학습 (Temporal Modeling)

1단계에서 학습된 이미지 인코더를 고정(frozen)시키고, 비디오 시퀀스 전체의 맥락을 파악하는 모델을 학습시킨다.

- **구조:** $\text{Image Encoder} \rightarrow \text{Temporal Model (LSTM, TCN, or Transformer)} \rightarrow \text{Prediction}$.
- **학습 절차:**
  - 이미지 인코더를 통해 각 프레임을 $d$-차원 특징 벡터로 변환한다.
  - 이 벡터 시퀀스를 Temporal Model(본 실험에서는 Causal TCN 사용)에 입력한다.
  - 정답 레이블과의 Weighted Cross-Entropy 손실 함수를 사용하여 Temporal Model을 학습시킨다. 이때 가중치는 Median Frequency Balancing 기법을 사용하여 클래스 불균형을 해소한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Cholec80, Autolaparo, m2cai16 (모두 복강경 수술 비디오).
- **모델 설정:** 이미지 인코더로 CLIP의 ResNet-50을 사용하였으며, 시계열 모델로는 Causal TCN을 사용하였다.
- **비교 대상:** 'Conventional Method' (ImageNet 사전 학습 $\rightarrow$ 단순 미세 조정 $\rightarrow$ TCN).
- **평가 지표:** Accuracy, Precision, Recall, Jaccard index, F1 score.

### 정량적 결과

실험 결과, ReSW-VL 방법론이 모든 데이터셋에서 기존 방식보다 우수한 성능을 보였다.

- **정확도(Accuracy):** 기존 방식 대비 약 $1.0 \sim 4.3\%$ 포인트 향상.
- **Jaccard Index:** 약 $3.7 \sim 6.1\%$ 포인트 향상.
- **F1 Score:** 약 $2.4 \sim 3.9\%$ 포인트 향상.

특히 데이터셋의 특성에 따라 최적의 모델이 다르게 나타났다.

- **Cholec80:** $\text{ReSW-VLo}$가 가장 높은 성능을 보였다. 이는 해당 데이터셋의 단계가 시간에 따라 순차적으로 증가하는 경향이 강해, 순서 정보가 반영된 $\text{ReSW-VLo}$가 유리했기 때문이다.
- **Autolaparo:** $\text{ReSW-VLi}$가 더 우수한 성능을 보였다. 이 데이터셋은 단계가 증가했다가 다시 감소하는 등 비순차적인 특성이 있어 독립적 프롬프트 학습이 더 효과적이었다.

### 정성적 결과

예측 결과 그래프(Figure 3, 4)에서 기존 방법은 단계 예측이 매우 불안정하며 빈번하게 단계가 바뀌는 '채터링' 현상이 관찰되었다. 반면, ReSW-VL은 훨씬 더 안정적이고 매끄러운 단계 전환 예측 결과를 보여주었다.

## 🧠 Insights & Discussion

본 논문은 수술 단계 인식에서 단순히 모델의 깊이나 시계열 구조를 변경하는 것보다, **입력 이미지에서 의미론적으로 풍부한 특징을 어떻게 추출하느냐(Representation Learning)**가 성능 향상에 결정적인 영향을 미친다는 것을 보여주었다. 특히 Vision-Language Model의 텍스트-이미지 정렬 능력을 Prompt Learning으로 이용함으로써, 수술 도구와 환경이라는 특수한 도메인에서도 효과적인 특징 추출이 가능함을 입증하였다.

**강점 및 한계:**

- **강점:** VLM을 수술 워크플로우 분석에 성공적으로 도입하였으며, 데이터셋의 순차적 특성에 따른 두 가지 학습 전략($\text{VLi}, \text{VLo}$)을 제시하여 유연성을 확보하였다.
- **한계:** 본 실험에서는 이미지 인코더로 ResNet-50을 사용하였으나, 최신 ViT(Vision Transformer) 계열의 인코더를 사용했을 때의 성능 향상 폭은 아직 확인되지 않았다. 또한, TCN 외에 Transformer Decoder 등을 결합했을 때의 시너지 효과에 대한 추가 검증이 필요하다.

**비판적 해석:**
제안 방법론이 기존 방식보다 안정적인 예측을 하는 이유는 Prompt Learning을 통해 각 단계의 '전형적인 이미지 특성'이 텍스트 임베딩 공간에 더 명확하게 정의되었기 때문으로 분석된다. 이는 단순히 레이블 번호로 분류하는 것보다 모델이 각 단계의 의미적 경계를 더 잘 학습하게 만들었음을 시사한다.

## 📌 TL;DR

본 논문은 수술 단계 인식을 위해 CLIP 기반의 Vision-Language Model과 Prompt Learning을 활용한 새로운 공간적 특징 학습 방법인 **ReSW-VL**을 제안한다. 이미지 인코더를 텍스트 프롬프트와 정렬시키는 방식으로 학습시킨 결과, 기존의 단순 전이 학습 방식보다 훨씬 안정적이고 높은 정확도의 수술 단계 인식 성능을 달성하였다. 특히 수술 단계의 순차성 여부에 따라 $\text{ReSW-VLo}$와 $\text{ReSW-VLi}$를 선택적으로 적용할 수 있음을 보였으며, 이는 향후 다양한 수술 도메인에 맞춤형 특징 추출기를 설계하는 데 중요한 기초 연구가 될 것으로 보인다.
