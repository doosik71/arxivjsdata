# MLIM: Vision-and-Language Model Pre-training with Masked Language and Image Modeling

Tarik Arici, Mehmet Saygin Seyfioglu, Tal Neiman, Yi Xu, Son Tran, Trishul Chilimbi, Belinda Zeng, Ismail Tutar (2021)

## 🧩 Problem to Solve

본 논문은 Vision-and-Language Pre-training (VLP) 모델들이 직면한 복잡성과 의존성 문제를 해결하고자 한다. 기존의 VLP 접근 방식들은 크게 세 가지 지점에서 한계를 보인다. 첫째, 이미지 임베더(Image Embedder)로 ResNet과 같은 무거운 딥러닝 모델이나 객체 탐지기(Object Detector)를 사용함으로써 연산 복잡도가 증가하고, 탐지기의 사전 정의된 카테고리에 종속되는 문제가 있다. 둘째, 손실 함수 측면에서 Masked Language Modeling (MLM) 외에 이미지-텍스트 간의 정렬(Alignment)을 위한 목적 함수를 사용하는데, 이는 정답(Ground Truth)이 없는 경우가 많으며 휴리스틱(Heuristic)한 방식이나 부정 쌍(Negative pairs) 생성에 의존하여 일반화 성능을 저하시킬 수 있다. 셋째, 마스킹 정책이 다중 모달리티의 특성을 충분히 활용하지 못하거나 타 모델이 생성한 정렬 결과에 강하게 결합되어 있다.

따라서 본 연구의 목표는 텍스트 전용 트랜스포머의 사전 학습처럼 단순하면서도 강력한 VLP 방법론을 제안하는 것이다. 구체적으로는 객체 탐지기와 휴리스틱한 정렬 손실 함수를 제거하고, 정답 기반의 단순한 재구성(Reconstruction) 태스크와 새로운 마스킹 전략을 통해 성능을 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지-텍스트 간의 정렬을 위한 복잡한 손실 함수 대신, 이미지 자체를 복원하는 Image Reconstruction (RECON) 손실 함수와 이를 극대화하는 Modality Aware Masking (MAM) 전략을 사용하는 것이다.

주요 기여 사항은 다음과 같다.

- **RECON 손실 함수의 도입**: 정답이 명확한 이미지 재구성 태스크를 통해 정렬 기반 손실 함수나 객체 영역 모델링(MIRM)의 필요성을 제거하고 시스템을 단순화하였다.
- **Shallow CNN 이미지 임베더**: 무거운 객체 탐지기 대신 경량의 Shallow CNN을 사용하여 픽셀 데이터를 고차원 표현으로 변환함으로써, 트랜스포머 내부에서 텍스트 임베딩과 더 효율적인 상호작용이 가능하도록 설계하였다.
- **Modality Aware Masking (MAM) 제안**: 이미지와 텍스트 중 어느 한쪽을 강하게 마스킹함으로써, 모델이 부족한 정보를 다른 모달리티에서 찾도록 강제하여 교차 모달리티 정보 흐름(Cross-modality information flow)을 촉진하였다.

## 📎 Related Works

기존의 VLP 연구들은 크게 두 가지 아키텍처로 나뉜다. 두 개의 단일 모달 트랜스포머를 사용하는 Two-stream 구조(예: ViLBERT, LXMERT)와 이미지와 텍스트 임베딩을 하나의 트랜스포머에 입력하는 Single-stream 구조(예: VL-BERT)가 그것이다. 본 논문은 상호작용 능력이 더 뛰어난 Single-stream 구조를 채택한다.

이미지 특징 추출을 위해 많은 모델이 객체 탐지기의 RoI(Region of Interest) 특징을 사용하지만, 이는 모델이 무겁고 특정 카테고리에 국한된다는 단점이 있다. ViLT와 같은 최신 연구는 선형 투영(Linear Projection)을 통해 픽셀을 직접 입력하여 복잡도를 낮췄으나, 이는 이미지 특징 학습을 트랜스포머 후반부 레이어에 의존하게 만든다는 한계가 있다.

또한, 기존의 이미지 영역 모델링(MIRM)은 객체 탐지기가 예측한 가짜 정답(Pseudo-targets)에 의존하며, ITM(Image Text Matching)과 같은 정렬 기반 손실 함수는 부정 쌍을 생성하는 과정에서 데이터셋 구축의 어려움과 일반화 성능 저하 문제를 야기한다. MLIM은 이러한 휴리스틱한 접근 대신, 이미지 픽셀 자체를 정답으로 사용하는 RECON 손실 함수를 통해 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

MLIM의 전체 파이프라인은 **Shallow CNN Image Embedder $\rightarrow$ Transformer $\rightarrow$ (MLM Head / CNN Image Decoder)** 순으로 구성된다.

1. **Image Embedder**: $2 \times 2$ 커널 크기와 stride 2를 가진 CNN 레이어를 사용하여 이미지를 임베딩한다. 커널 크기와 stride를 동일하게 설정하여 각 입력 픽셀이 단 하나의 필터 출력에만 기여하도록 함으로써, 특정 임베딩을 마스킹했을 때 원본 이미지의 특정 영역 정보가 완전히 삭제되도록 설계하였다.
2. **Transformer**: BERT-large-uncased 모델을 기반으로 하며, 이미지 임베딩과 텍스트 토큰 임베딩이 함께 입력된다. 각 모달리티는 별도의 positional embedding을 가지며, 명시적인 modality embedding(segment embedding)은 사용하지 않는다.
3. **Image Decoder**: 트랜스포머의 출력 중 이미지 임베딩에 해당하는 벡터들을 다시 2D 그리드로 재구성한 뒤, 전치 합성곱(Deconvolutional) 레이어를 통과시켜 원본 이미지 해상도와 3채널 색상을 복원한다. 마지막에 sigmoid 함수를 적용하여 픽셀 강도를 $[0, 1]$ 범위로 맞춘다.

### 학습 목표 및 손실 함수

모델은 다음 두 가지 손실 함수의 합을 최소화하는 방향으로 학습된다.

- **Masked Language Modeling (MLM) Loss**: 마스킹된 텍스트 토큰을 예측하는 작업으로, BERT와 동일하게 MLM head를 통한 Negative Log-Likelihood 손실을 사용한다.
- **Image Reconstruction (RECON) Loss**: 마스킹된 영역을 포함하여 이미지 전체를 복원하는 작업이다. 손실 함수로는 픽셀 단위의 평균 제곱 오차(Sum of Squared Errors, SSE)를 사용한다.
  $$ \text{RECON Loss} = \text{average of pixel-wise SSE} $$
  이미지 임베딩이 손실 압축된 표현이므로, 마스킹된 영역뿐만 아니라 이미지 전체를 재구성하는 것이 더 적절하다고 판단하여 전체 이미지에 대해 손실을 계산한다.

### Modality Aware Masking (MAM)

교차 모달리티 상호작용을 강제하기 위해 MAM은 세 가지 모드로 작동하며, 각 모드는 동일한 확률로 선택된다.

1. **Heavy Image-masking**: 이미지의 $80\%$를 마스킹하고 텍스트는 유지한다. (텍스트 $\rightarrow$ 이미지 정보 흐름 촉진)
2. **Heavy Text-masking**: 텍스트의 $80\%$를 마스킹하고 이미지는 유지한다. (이미지 $\rightarrow$ 텍스트 정보 흐름 촉진)
3. **Light Masking**: 이미지와 텍스트 모두 가볍게 마스킹한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 사전 학습에는 600만 개의 Amazon 카탈로그 아이템(이미지 및 텍스트 속성)을 사용하였으며, 평가에는 내부 데이터셋인 Closely-matching (CM) 아이템 쌍 찾기 작업(학습 30K, 테스트 10K)을 사용하였다.
- **지표**: PR AUC (Precision-Recall Area Under the Curve)를 사용하여 성능을 측정하였다.
- **구현 세부사항**: BERT-large-uncased를 기반으로 하며, 이미지 해상도는 $384 \times 384$로 설정하였다. 이미지 임베더는 7개 레이어(200K 파라미터), 디코더는 10개 레이어(2.8M 파라미터)로 구성된 경량 모델이다.

### 주요 결과

- **Downstream Task 성능**: CM 작업에서 Modality Dropout (MDO)을 적용하여 파인튜닝했을 때 PR AUC $0.884$를 달성하였으며, MDO 미적용 시($0.864$)보다 성능이 향상되었다.
- **손실 함수 비교**: RECON 손실을 사용한 모델이 ITM(정렬 기반) 손실만 사용한 모델(PR AUC $0.855$)보다 높은 성능($0.884$)을 보였다. RECON과 ITM을 동시에 사용하더라도 추가적인 성능 향상은 없었다.
- **교차 모달리티 상호작용 검증**:
  - MLM 손실 측정 시, 무작위 이미지나 빈 이미지보다 실제 쌍이 맞는 이미지를 입력했을 때 손실이 감소하였다. 이는 모델이 텍스트 복원을 위해 이미지 정보를 활용함을 의미한다.
  - RECON 손실 측정 시, 실제 쌍이 맞는 텍스트를 입력했을 때 손실이 가장 낮았다. 이는 모델이 이미지 복원을 위해 텍스트 정보를 활용함을 의미한다.
  - 특히 텍스트 $\rightarrow$ 이미지 정보 흐름이 이미지 $\rightarrow$ 텍스트 흐름보다 더 유의미하게 작용하는 경향이 관찰되었다.

## 🧠 Insights & Discussion

본 논문은 VLP 모델에서 흔히 사용되는 무거운 객체 탐지기와 휴리스틱한 정렬 손실 함수 없이도, 단순한 이미지 재구성(RECON)과 전략적인 마스킹(MAM)만으로 충분한 성능을 낼 수 있음을 입증하였다. 특히, 정답이 명확한 픽셀 데이터를 타겟으로 설정함으로써 학습의 안정성을 높이고 파이프라인을 단순화한 점이 강점이다.

그러나 본 연구의 실험은 주로 이커머스 데이터셋(단일 품목 위주의 단순한 배경 이미지)에서 수행되었다. 저자들 또한 결론에서 언급하였듯이, 여러 객체가 등장하거나 배경이 복잡한 일반적인 이미지 데이터셋에서도 동일한 효과가 나타날지는 미지수이다. Shallow CNN 임베더가 복잡한 장면의 고차원 특징을 충분히 포착할 수 있을지에 대한 추가적인 연구가 필요할 것으로 보인다.

또한, Modality Dropout (MDO)이 파인튜닝 단계에서 성능을 향상시킨 점은, 사전 학습 단계의 MAM 전략이 모델이 각 모달리티에 과도하게 의존하지 않고 균형 잡힌 표현을 학습하도록 도왔음을 시사한다.

## 📌 TL;DR

본 논문은 복잡한 객체 탐지기와 휴리스틱한 정렬 손실 함수를 제거하고, **Shallow CNN 임베더**, **이미지 재구성(RECON) 손실**, 그리고 **Modality Aware Masking (MAM)**을 결합한 단순화된 VLP 사전 학습 방법론(MLIM)을 제안한다. 이 방법론은 이커머스 데이터셋에서 기존 정렬 기반 방식보다 우수한 성능을 보였으며, 텍스트와 이미지 간의 상호 보완적인 정보 흐름을 효과적으로 학습함을 확인하였다. 향후 복잡한 이미지 데이터셋으로의 확장 가능성이 크며, 경량 임베더 기반의 VLP 구조 설계에 중요한 통찰을 제공한다.
