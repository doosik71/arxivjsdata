# ADTR: Anomaly Detection Transformer with Feature Reconstruction

Zhiyuan You, Kai Yang, Wenhan Luo, Lei Cui, Yu Zheng, and Xinyi Le

## 🧩 Problem to Solve

기존의 CNN 기반 픽셀 재구성 방식의 이상 탐지 모델들은 두 가지 주요 문제점을 가지고 있습니다.

1. **의미 정보 부족:** 재구성 대상이 원본 픽셀 값이기 때문에 의미론적 정보가 구별하기 어렵습니다. 이는 정상 및 이상 영역이 유사한 픽셀 값을 가지지만 다른 의미 정보를 가질 때 탐지에 실패하게 만듭니다.
2. **동일 매핑 경향:** CNN은 정상 샘플과 이상 샘플 모두를 잘 재구성하려는 "동일 매핑(identical mapping)" 경향이 있어, 재구성 후에도 이상을 구별하기 어렵습니다.

또한, 실제 생산 라인에서는 이상 샘플이 극히 부족하여 오직 정상 샘플의 사전 지식만을 활용하는 이상 탐지 방식의 중요성이 커지고 있으며, 소량의 이상 샘플이 점차 수집될 경우 이를 활용할 수 있는 유연한 접근 방식이 필요합니다.

## ✨ Key Contributions

- **특징 재구성 Transformer 제안 (ADTR):** 원본 픽셀 값 대신 사전 학습된 특징(pre-trained features)을 재구성하는 Anomaly Detection TRansformer (ADTR)를 제안합니다. 사전 학습된 특징은 구별 가능한 의미 정보를 포함합니다.
- **Transformer의 이점 활용:** Transformer의 어텐션 레이어(attention layer)에 있는 보조 쿼리 임베딩(auxiliary query embedding)이 이상 샘플을 잘 재구성하지 못하게 하여, 재구성에 실패했을 때 이상을 쉽게 감지할 수 있도록 합니다. 이를 통해 정상 샘플과 이상 샘플 사이에 명확한 일반화 간극(generalization gap)을 만듭니다.
- **새로운 손실 함수 제안:** 정상 샘플만 있는 경우와 이미지-레벨(image-level) 또는 픽셀-레벨(pixel-level) 레이블이 있는 이상 샘플이 활용 가능한 경우 모두에 호환되는 새로운 손실 함수들을 제안합니다.
- **성능 향상:** 간단한 합성 이상 샘플(synthetic anomalies) 또는 외부 관련 없는 이상 샘플을 추가하여 성능을 더욱 향상시킬 수 있음을 입증합니다.
- **최첨단 성능 달성:** MVTec-AD 및 CIFAR-10 데이터셋에서 모든 기준 모델 대비 뛰어난 성능을 달성합니다.

## 📎 Related Works

- **재구성 기반 접근 방식 (Reconstruction-based approaches):** AutoEncoder (AE), Variational AutoEncoder (VAE), Generative Adversarial Network (GAN) 등을 사용하여 정상 샘플 분포를 모델링하고, 이상 샘플에 대한 재구성 실패를 통해 탐지합니다 (예: AnoGAN, Mem-AE).
- **투영 기반 접근 방식 (Projection-based approaches):** 샘플을 임베딩 공간(embedding space)으로 투영하여 정상 샘플과 이상 샘플을 더 잘 구별합니다 (예: SVDD, FCDD, Teacher-Student networks인 TS, KDAD, PaDiM).
- **이상 탐지의 Transformer (Transformer in anomaly detection):** 일부 초기 시도(예: InTra, VT-ADL, AnoViT)는 Transformer를 사용하여 원본 픽셀을 재구성하지만, ADTR은 사전 학습된 특징을 재구성하며 Transformer의 개선 원인(쿼리 임베딩의 역할)을 명확히 제시합니다.

## 🛠️ Methodology

ADTR은 특징 임베딩(embedding), 특징 재구성(reconstruction), 그리고 비교(comparison)의 세 단계로 구성됩니다.

1. **임베딩 (Embedding):**
   - 사전 학습된 CNN 백본(예: EfficientNet-B4)을 사용하여 이미지에서 다중 스케일 특징(multi-scale features)을 추출합니다.
   - 다중 스케일 특징 맵 $f \in \mathbb{R}^{C \times H \times W}$는 서로 다른 레이어의 특징 맵들을 동일한 크기로 조절하고 연결하여 생성됩니다.
2. **재구성 (Reconstruction):**
   - 특징 맵 $f$는 $H \times W$개의 특징 토큰(feature tokens)으로 분할됩니다. 계산 효율성을 위해 $1 \times 1$ 컨볼루션을 통해 차원을 축소합니다.
   - Transformer 인코더는 입력 특징 토큰을 잠재 특징 공간에 임베딩합니다.
   - Transformer 디코더는 학습 가능한 **보조 쿼리 임베딩(auxiliary query embedding)**을 사용하여 특징 토큰을 재구성합니다. 디코더는 멀티 헤드 셀프 어텐션(multi-head self-attention) 및 인코더-디코더 어텐션(encoder-decoder attention) 메커니즘을 활용합니다.
   - Transformer는 순열 불변(permutation-invariant)이므로 학습 가능한 위치 임베딩(position embedding)이 포함됩니다.
3. **비교 (Comparison) 및 손실 함수:**
   - **특징 차이 맵 (Feature difference map)** $d(i, u) = f(i, u) - \hat{f}(i, u)$를 계산하며, 여기서 $f$는 추출된 특징, $\hat{f}$는 재구성된 특징입니다.
   - **이상 점수 맵 (Anomaly score map)** $s(u) = ||d(:, u)||_2$는 특징 차이 벡터의 $L_2$ 노름(norm)으로 계산됩니다.
   - **이미지 이상 점수 (Image anomaly score)**는 평균 풀링(averagely pooled)된 $s(u)$의 최대값으로 결정됩니다.
   - **정상 샘플만 있는 경우:** $L_{norm}$은 추출된 특징 $f$와 재구성된 특징 $\hat{f}$ 간의 MSE 손실을 사용합니다:
     $$L_{norm} = \frac{1}{H \times W} ||f - \hat{f}||_2^2$$
   - **이상 샘플이 있는 경우 (픽셀-레벨 레이블):** 특징 차이 맵 $d(i, u)$에서 유사 휴버 손실(pseudo-Huber loss) $\phi(u)$를 계산합니다. 손실 함수 $L_{px}$는 "밀고 당기기(push-pull)" 방식으로 설계됩니다:
     $$L_{px} = \frac{1}{HW} \sum_u (1-y(u))\phi(u) - \alpha \log(1 - \exp(-\frac{1}{HW} \sum_u y(u)\phi(u)))$$
     여기서 $y(u)$는 픽셀-레벨 레이블입니다 (0: 정상, 1: 이상). 첫 번째 항은 정상 특징을 당기고, 두 번째 항은 이상 특징을 밀어냅니다.
   - **이상 샘플이 있는 경우 (이미지-레벨 레이블):** $k$개의 최대 $\phi(u)$ 값들의 평균 $q = \frac{1}{k} \sum_{topk}(\phi)$을 이미지의 이상 점수로 사용합니다. 손실 함수 $L_{img}$는 다음과 같습니다:
     $$L_{img} = (1-y)q - \alpha y \log(1 - \exp(-q))$$
     여기서 $y$는 이미지-레벨 레이블입니다 (0: 정상, 1: 이상).
4. **Transformer가 "동일 매핑"을 방지하는 원리:**
   - CNN의 컨볼루션 레이어는 가중치가 항등 행렬(identity matrix)이 되고 편향이 0이 되는 "지름길(shortcut)"을 학습하여 이상도 잘 재구성할 수 있습니다.
   - Transformer의 어텐션 레이어는 학습 가능한 쿼리 임베딩 $q$를 포함합니다. 재구성 $\hat{x}$가 $x^+$에 가까워지려면 어텐션 맵 $softmax(q(x^+)^T / \sqrt{C})$이 항등 행렬에 근접해야 합니다. 이는 $q$가 정상 샘플 $x^+$에 매우 관련되도록 학습된다는 것을 의미하며, 결과적으로 $q$는 이상 샘플 $x^-$를 잘 재구성하지 못하게 됩니다.

## 📊 Results

- **MVTec-AD 데이터셋:**
  - **이상 위치 탐지 (Pixel-level AUROC):** ADTR (97.2%)은 SPADE (96.0%)를 1.2%p 능가합니다. 합성 이상 샘플을 추가한 ADTR+는 97.5%로 성능을 더 개선합니다.
  - **이상 탐지 (Image-level AUROC):** ADTR (96.4%)은 모든 기준 모델(예: TS 92.5%)을 크게 능가합니다. ADTR+는 96.9%로 성능을 더 개선합니다.
  - 정성적 결과는 미묘한 이상(예: 뒤집힌 'Metal Nut')에 대해서도 높은 위치 탐지 정확도를 보여줍니다.
- **CIFAR-10 데이터셋:**
  - **이상 탐지 (Image-level AUROC):** ADTR (94.7%)은 KDAD (87.2%)보다 7.5%p 높은 성능을 보입니다. 외부 관련 없는 이상 샘플을 활용한 ADTR+는 96.1%로 성능을 더욱 향상시킵니다.
- **Ablation Study (MVTec-AD, Pixel-level AUROC):**
  - **어텐션 및 보조 쿼리 임베딩:** 어텐션 레이어(w/o Attn) 또는 보조 쿼리 임베딩(w/o Query)을 제거하면 Transformer의 성능이 CNN 수준으로 각각 2.4%, 3% 하락하여, 쿼리 임베딩이 "동일 매핑" 방지에 핵심적인 역할을 함을 입증합니다.
  - **픽셀 vs. 특징 재구성:** 특징을 재구성하는 방식 (97.2%)이 픽셀을 재구성하는 방식 (91.3%)보다 월등히 뛰어납니다.
  - **백본:** EfficientNet-B4 외에 ResNet-18/34, EfficientNet-B0 등 다양한 백본에서도 우수한 성능을 보여줍니다.
  - **다중 스케일 특징:** 다중 스케일 특징 (97.2%)이 단일 마지막 레이어 특징 (96.0%)보다 뛰어납니다.
- **특징 차이 벡터 시각화:** t-SNE를 통해 특징 차이 벡터를 시각화한 결과, 정상 샘플과 이상 샘플 사이에 넓은 일반화 간극이 명확하게 나타났습니다.

## 🧠 Insights & Discussion

- **의미론적 특징의 중요성:** 원본 픽셀 값 대신 풍부한 의미론적 정보를 담고 있는 사전 학습된 특징을 재구성하는 것이 이상 탐지 성능에 결정적인 역할을 합니다.
- **Transformer의 차별점:** Transformer의 핵심 메커니즘인 보조 쿼리 임베딩은 CNN이 겪는 "동일 매핑" 문제를 효과적으로 방지합니다. 이로 인해 모델이 정상 샘플에는 특화되어 이상 샘플은 재구성하지 못하게 함으로써, 정상과 이상을 명확하게 구별하는 일반화 간극을 생성합니다.
- **실용적인 확장성:** 제안된 새로운 손실 함수들은 정상 샘플만 있는 이상 탐지 시나리오뿐만 아니라, 소량의 이상 샘플(픽셀 또는 이미지 레벨 레이블)이 사용 가능한 현실적인 시나리오까지 효과적으로 처리할 수 있도록 모델의 적용 범위를 확장합니다.
- **강력한 일반화 및 견고성:** ADTR은 다양한 유형의 이상(질감/색상 왜곡, 오배치 등)과 다양한 카테고리에 걸쳐 안정적이고 높은 성능을 보이며, 이는 모델의 강력한 일반화 능력과 견고성을 입증합니다.

## 📌 TL;DR

ADTR은 기존 CNN 기반 픽셀 재구성 방식의 한계(의미 정보 부족, 정상/이상 모두 잘 재구성)를 극복하기 위해 제안된 이상 탐지 모델이다. 사전 학습된 CNN으로 특징을 추출하고, 보조 쿼리 임베딩을 가진 Transformer를 사용하여 이 특징을 재구성한다. Transformer는 이상을 잘 재구성하지 못하게 함으로써 정상과 이상 간의 명확한 일반화 간극을 생성한다. 또한, 정상 샘플만 있는 경우와 이상 샘플(픽셀/이미지 레벨 레이블)이 있는 경우 모두에 적용 가능한 새로운 손실 함수를 제안하여 성능을 더욱 향상시킨다. MVTec-AD 및 CIFAR-10 데이터셋에서 최첨단 성능을 달성했다.
