# GMT: Guided Mask Transformer for Leaf Instance Segmentation

Feng Chen, Sotirios A. Tsaftaris, Mario Valerio Giuffrida (2024)

## 🧩 Problem to Solve

본 논문은 식물 이미지에서 각 잎을 개별적으로 분리하고 경계를 획정하는 **Leaf Instance Segmentation** 문제를 해결하고자 한다. 식물 잎 세그멘테이션은 다음과 같은 이유로 매우 어려운 과제이다:

- **높은 유사성 및 변동성:** 잎들은 모양과 색상이 매우 비슷하며, 종 내외적으로 형태적 변동성이 크다.
- **심한 폐색(Occlusion):** 잎들이 서로 겹쳐 있는 경우가 많아 개별 인스턴스를 구분하기 어렵다.
- **데이터셋 규모의 한계:** 일반적인 컴퓨터 비전 데이터셋에 비해 주석이 달린 식물 잎 데이터셋의 크기가 매우 작아, 대규모 모델을 효과적으로 학습시키기 어렵다.

이러한 문제는 기존의 SOTA Transformer 모델(예: Mask2Former)조차 작은 잎이나 겹쳐진 잎을 놓치거나, 잎과 유사한 다른 녹색 객체를 잎으로 오인하는 결과로 이어진다. 따라서 본 논문의 목표는 식물 잎의 **공간적 분포 특성(Spatial Distribution Priors)**을 모델에 통합하여, 제한된 데이터 환경에서도 정밀한 인스턴스 세그멘테이션을 달성하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 식물의 성장 패턴(예: 로제트 식물의 경우 중심에서 바깥쪽으로 성장하며 위치에 따라 크기와 밀도가 달라짐)에 따른 **공간적 분포 우선순위(Spatial Distribution Priors)**를 활용하는 것이다.

이를 위해 저자들은 **Guide Functions**라는 개념을 도입하여, 픽셀 좌표를 분리 가능한 임베딩 공간으로 매핑하고, 이를 Transformer 구조에 통합한 **Guided Mask Transformer (GMT)**를 제안한다. 구체적으로는 다음 세 가지 모듈을 통해 공간적 사전 지식을 모델에 주입한다:

1. **Guided Positional Encoding (GPE):** 픽셀 수준의 공간 표현력을 강화한다.
2. **Guided Embedding Fusion Module (GEFM):** 인스턴스 간의 특징 분별력을 높인다.
3. **Guided Dynamic Positional Queries (GDPQ):** 인스턴스 위치 정보를 바탕으로 쿼리를 동적으로 생성한다.

## 📎 Related Works

### 관련 연구 및 한계

- **CNN 기반 모델:** 과거에는 다양한 CNN 모델이 잎 질병 검출이나 과일 세그멘테이션에 사용되었으나, 복잡한 폐색 상황과 정밀한 경계 획정에서 한계가 있었다.
- **Transformer 기반 모델:** MaskFormer와 Mask2Former 같은 모델들이 일반적인 인스턴스 세그멘테이션에서 뛰어난 성능을 보였으나, 식물 데이터의 특수성(작은 데이터셋, 복잡한 구조)을 고려한 도메인 지식의 통합이 부족했다.

### 차별점

기존 모델들이 일반적인 이미지 특징에 의존하는 것과 달리, GMT는 식물 잎의 공간적 배치 특성을 수학적으로 정의한 **Guide Functions**를 통해 사전 지식으로 활용함으로써, 데이터 부족 문제를 완화하고 인스턴스 분별력을 획기적으로 높였다.

## 🛠️ Methodology

### 1. Guide Functions

먼저, 픽셀 좌표 $(x, y)$를 임베딩 공간으로 매핑하기 위해 조화 함수(Harmonic Functions)를 사용한다.

$$f_i(x,y;\psi_i) = \sin\left(\frac{\psi_i[1]}{W}x + \frac{\psi_i[2]}{H}y + \psi_i[3]\right)$$

여기서 $\psi_i[1], \psi_i[2]$는 학습 가능한 주파수 파라미터이며, $\psi_i[3]$은 위상 파라미터이다. 특정 인스턴스 $S$에 대한 임베딩 $e(S; \Psi)$는 $d_g$개의 가이드 함수 결과값들의 벡터로 표현된다. 이 함수들은 서로 다른 인스턴스들이 임베딩 공간에서 최소 거리 $\epsilon$만큼 떨어지도록 학습된다:

$$\ell(\Psi) = \sum_{I \in \mathcal{I}} \frac{1}{|P_I|} \sum_{(S, S') \in P_I} \max(0, \epsilon - \|e(S; \Psi) - e(S'; \Psi)\|_1)$$

### 2. GMT 아키텍처 및 핵심 모듈

GMT는 Mask2Former를 기본 뼈대로 하며, 여기에 세 가지 가이드 모듈을 추가한다.

#### (1) Guided Positional Encoding (GPE)

기존의 Sinusoidal Positional Encoding (SPE)에 학습된 가이드 함수 정보를 추가한다. 가이드 함수의 차원 $d_g$가 픽셀 특징 차원 $d_p$보다 훨씬 작기 때문에, 위상 $\psi[3]$을 시프트하여 차원을 확장한 후 SPE와 더해줌으로써 공간 표현력을 높인다.

#### (2) Guided Embedding Fusion Module (GEFM)

픽셀 디코더의 출력을 $1 \times 1$ 합성곱 층을 통해 **Guided Features**로 투영한다. 이 특징들은 정답(GT) 마스크를 가이드 함수로 인코딩하여 얻은 **GT Guided Embeddings**와 $L_1$ 손실 함수를 통해 직접적으로 감독 학습된다. 이를 통해 모델이 잎 인스턴스들을 임베딩 공간에서 더 잘 구분하도록 강제한다.

#### (3) Guided Dynamic Positional Queries (GDPQ)

기존의 정적인 Positional Query 대신, 이전 Transformer 블록에서 예측된 마스크 결과($S_{t-1}$)를 가이드 함수로 인코딩하여 얻은 임베딩 $E(S_{t-1}; \Psi)$를 기반으로 쿼리를 동적으로 생성한다.

$$Q_t^p = h(E(S_{t-1}; \Psi) + B)$$

여기서 $h(\cdot)$는 MLP이며, $B$는 학습 가능한 바이어스이다. 이를 통해 모델은 예측이 진행됨에 따라 인스턴스의 위치 정보를 지속적으로 정교화할 수 있다.

### 3. 학습 목표 (Loss Function)

전체 손실 함수는 다음과 같이 정의된다:
$$L = L_{guide} + L_{M2F}$$
$L_{guide}$는 GEFM에서 사용되는 가이드 특징에 대한 가중 $L_1$ 손실이며, $L_{M2F}$는 Mask2Former의 표준 손실(Binary Cross-Entropy, Dice Loss, Classification Loss)의 합이다.

## 📊 Results

### 실험 설정

- **데이터셋:** CVPPP LSC, MSU-PID, KOMATSUNA 세 가지 공공 데이터셋 사용.
- **비교 대상:** Mask2Former (Baseline) 및 기타 최신 잎 세그멘테이션 모델.
- **평가 지표:** Best Dice (BD), Symmetric Best Dice (SBD), Difference in Count (|DiC|).

### 주요 결과

- **정량적 성능:** 모든 데이터셋에서 Mask2Former보다 우수한 성능을 보였다. 특히 CVPPP LSC에서 SBD가 89.5에서 90.1로, |DiC|가 0.67에서 0.48로 개선되었다.
- **잎 크기별 분석:** 특히 **작은(Small) 및 중간(Medium) 크기의 잎**에서 성능 향상이 두드러졌다. 이는 기존 모델들이 작은 잎을 놓치거나 겹친 잎을 하나로 합치는 경향이 있었으나, GMT는 이를 효과적으로 분리해냈음을 의미한다.
- **백본 영향:** ResNet-50이 Swin-T-B보다 SBD와 |DiC| 측면에서 더 좋은 성능을 보였는데, 이는 식물 데이터셋의 규모가 작아 거대 모델의 경우 오버피팅(Overfitting)이 발생하기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 딥러닝 모델에 단순한 데이터 증강이 아닌, 식물학적 특성(공간적 분포)을 수학적 제약 조건(Guide Functions)으로 통합함으로써 데이터 부족 문제를 해결했다. 특히 GDPQ를 통해 예측 결과와 위치 정보를 피드백 루프로 연결한 점이 인스턴스 분별력을 높이는 데 결정적인 역할을 한 것으로 보인다.

### 한계 및 논의

- **계산 복잡도:** 가이드 함수를 통한 임베딩 생성 및 추가 모듈 도입으로 인해 연산량이 증가했을 가능성이 있다.
- **가정의 일반성:** 본 모델은 잎이 특정 공간적 패턴을 가지고 성장한다는 가정에 기반한다. 만약 매우 불규칙하게 성장하는 식물이나 다른 종류의 객체에 적용할 경우, 가이드 함수의 설계가 변경되어야 할 것이다.
- **백본의 선택:** 최신 고성능 백본(Swin Transformer 등)이 오히려 성능을 저하시킨 점은, 도메인 특화 데이터셋에서는 모델의 크기보다 적절한 Prior의 주입이 더 중요하다는 통찰을 제공한다.

## 📌 TL;DR

본 논문은 식물 잎의 공간적 분포 특성을 활용한 **Guided Mask Transformer (GMT)**를 제안한다. 학습 가능한 **Guide Functions**를 통해 잎의 위치 정보를 임베딩 공간에 매핑하고, 이를 **GPE, GEFM, GDPQ** 세 가지 모듈을 통해 Transformer 디코더에 통합하였다. 실험 결과, 특히 작은 잎과 겹쳐진 잎의 세그멘테이션 성능을 크게 향상시켜 SOTA 성능을 달성하였으며, 이는 데이터가 부족한 특수 도메인에서 공간적 사전 지식(Spatial Prior)의 통합이 매우 유효함을 입증한다.
