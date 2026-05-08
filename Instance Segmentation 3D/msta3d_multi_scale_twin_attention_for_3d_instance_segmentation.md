# MSTA3D: Multi-scale Twin-attention for 3D Instance Segmentation

Duc Dang Trung Tran, Byeongkeun Kang, and Yeejin Lee (2024)

## 🧩 Problem to Solve

본 논문은 3D Point Cloud의 Instance Segmentation에서 발생하는 **Over-segmentation(과분할)** 문제를 해결하고자 한다. 특히 문(door), 커튼(curtain), 책장(bookshelf)과 같이 크기가 큰 객체나 배경 영역에서 하나의 인스턴스가 여러 개의 작은 조각으로 잘못 분리되는 현상이 두드러지게 나타난다.

이러한 문제의 근본적인 원인은 최근 Transformer 기반 기법들이 연산 효율성을 위해 Superpoint를 활용하는데, Superpoint의 마스크 예측이 불완전하거나 Superpoint를 다시 Point로 변환하는 과정에서 범주적 그룹화의 신뢰도가 떨어지기 때문이다. 또한, 3D 장면의 포인트 분포가 희소하고 불규칙하여 포인트 단위의 분류 및 집계 학습이 어렵다는 점이 문제로 지적된다. 따라서 본 연구의 목표는 Multi-scale 특성 표현과 전역/지역 공간 제약 조건을 동시에 도입하여 과분할 문제를 완화하고 인스턴스 마스크 예측의 신뢰도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체의 크기에 상관없이 효과적으로 특징을 추출할 수 있는 **Multi-scale Superpoint** 표현과 이를 융합하는 **Twin-attention** 메커니즘을 도입하는 것이다.

또한, 기존의 Semantic Query 외에 공간적 제약을 제공하는 **Box Query**와 이를 정교화하는 **Box Regularizer**를 제안한다. 이는 추가적인 어노테이션 없이도 인스턴스의 공간적 위치 정보를 학습 과정에 통합함으로써, 객체 국소화(Localization) 능력을 향상시키고 배경 노이즈를 줄이는 역할을 한다.

## 📎 Related Works

3D Instance Segmentation 방법론은 크게 네 가지로 분류된다.

1. **Proposal-based**: 3D Bounding Box를 먼저 생성한 후 마스크를 세그멘테이션한다.
2. **Grouping-based**: 포인트별 특징을 이용해 클러스터링을 수행한다.
3. **Kernel-based**: 각 인스턴스를 동적 컨볼루션(Dynamic Convolution)의 커널로 취급한다.
4. **Transformer-based**: 각 인스턴스를 Query로 취급하고 Decoder를 통해 정교화한다.

최근에는 연산량 감소를 위해 Superpoint를 활용하는 Transformer 기반 방식이 주목받고 있으나, 앞서 언급한 과분할 문제와 마스크 예측의 불확실성이라는 한계가 존재한다. MSTA3D는 이러한 한계를 극복하기 위해 단일 스케일이 아닌 멀티 스케일 특징을 활용하고, 명시적인 공간 제약(Box Query)을 추가하여 기존 Transformer 기반 방식들과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

MSTA3D는 크게 (1) Backbone Network, (2) Twin-attention Decoder, (3) Box Regularizer의 세 가지 구성 요소로 이루어져 있다.

### 1. Backbone Network 및 Multi-scale Superpoint

Backbone으로는 3D U-Net을 사용하여 복셀화된 포인트 클라우드에서 특징을 추출한다. 메모리 효율을 위해 포인트 단위가 아닌 Superpoint 단위의 레이블을 사용한다. 이때, Superpoint 생성 시 이웃 포인트의 수를 조절하여 **Low-scale($S_\ell$)**과 **High-scale($S_h$)** 두 가지 스케일의 Superpoint 특징을 미리 계산하고 풀링 레이어를 통해 집계한다. 이는 다양한 크기의 객체를 효과적으로 표현하기 위함이다.

### 2. Twin-attention Decoder

Multi-scale 특징을 통합하기 위해 6개의 Twin-attention 블록으로 구성된 디코더를 사용한다.

**Region Constraint Instance Query**:
단순한 Semantic Query($X_{sem}$)에 6개의 좌표 정보로 구성된 Box Query($X_{box}$)를 결합하여 전체 쿼리 $X_0$를 구성한다.
$$X_0 = [X_{sem}; X_{box}]$$

**Twin-Attention-Based Feature Extraction**:
각 블록은 Cross-attention, Self-attention, Feature Fusion 순으로 작동한다. 특히 Cross-attention 단계에서 저스케일($S_\ell$)과 고스케일($S_h$) 특징에 대해 각각 독립적인 어텐션을 수행하는 Twin-attention(TATT)을 적용한다.

$$ \text{TATT}(Q, K_h, V_h) = \text{softmax}\left(\frac{QK_h^T}{\sqrt{d}} + A_{h}\right)V_h $$
$$ \text{TATT}(Q, K_\ell, V_\ell) = \text{softmax}\left(\frac{QK_\ell^T}{\sqrt{d}}\right)V_\ell $$

여기서 $A_h$는 Superpoint 마스크 어텐션이다. 이후 Self-attention을 거친 두 스케일의 결과물 $Z_\ell$과 $Z_h$를 요소별 곱셈(element-wise multiplication)으로 융합하고 FFN을 통해 최종 특징 $X$를 도출한다.
$$ X = \text{FFN}(Z_\ell \otimes Z_h) $$

### 3. Spatial Constraint Regularizer (Box Regularizer)

Box Regularizer는 인스턴스별 Box 예측값($\hat{B}$)과 장면 전체의 시맨틱 스코어($F_m$), 장면 전체의 Box 정보($F_{box}$)를 입력으로 받는다.

먼저, 개별 인스턴스 Box와 전체 장면 Box 간의 상대적 위치 차이 $R_i$를 계산한다.
$$ R_i = \{\hat{b}_i\} - F_{box} $$
이 상대적 위치 정보 $R_i$를 시맨틱 특징 $F_m$과 결합하여 최종 바이너리 마스크 $\hat{M}$을 예측함으로써, 전역적 맥락과 지역적 인스턴스 특성을 모두 반영한 정교한 마스크를 생성한다.

### 4. 학습 및 추론 절차

**손실 함수**:
전체 손실 함수 $L$은 다음과 같이 구성된다.
$$ L = \beta_m(L_{bce} + L_{dice}) + \beta_{cls}L_{cls} + \beta_s L_s + \beta_b L_b + \beta_{bs} L_{bs} $$
여기서 $L_{bce}$와 $L_{dice}$는 마스크 예측, $L_{cls}$는 클래스 분류, $L_s$는 마스크 스코어, $L_b$는 Box 좌표(L1 loss), $L_{bs}$는 Box 스코어(L2 loss)를 최적화한다.

**매칭 방법**:
Hungarian 알고리즘을 사용하여 예측된 Proposal과 Ground Truth 인스턴스를 일대일 매칭하며, 이때 시맨틱 확률과 Superpoint 마스크 매칭 비용을 함께 고려한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ScanNetV2, ScanNet200, S3DIS (Area 5)
- **평가 지표**: mAP, $\text{mAP}_{50}$, $\text{mAP}_{25}$
- **구현 세부사항**: PyTorch 기반, AdamW 옵티마이저, A100 GPU 사용, 복셀 크기 2cm(ScanNet) / 5cm(S3DIS).

### 주요 결과

1. **ScanNetV2**: $\text{mAP}$ 기준 기존 SOTA 모델들을 상회하는 성능을 보였다. 특히 책장(bookshelf)과 같은 대형 객체에서 $\text{mAP}_{25}$가 10% 이상 향상되어 과분할 문제가 효과적으로 해결되었음을 입증하였다.
2. **ScanNet200**: SPFormer 및 TD3D 대비 $\text{mAP}$와 $\text{mAP}_{50}$에서 유의미한 성능 향상을 보였다.
3. **S3DIS**: Area 5에서 $\text{mAP}_{50}$ 및 $\text{mPrec}$ 지표에서 경쟁력 있는 결과를 나타냈다.
4. **효율성**: QueryFormer나 MAFT보다 훨씬 적은 파라미터 수(약 24.5M개 적음)를 사용하면서도 더 높은 성능을 달성하여 모델의 경량화 및 효율성을 입증하였다.

## 🧠 Insights & Discussion

**강점**:
본 논문은 Multi-scale Superpoint와 Twin-attention 구조를 통해 객체의 크기 변화에 유연하게 대응함으로써 3D 인스턴스 세그멘테이션의 고질적인 문제인 과분할을 성공적으로 억제하였다. 특히 Box Query와 Regularizer를 통해 추가 데이터 없이 공간적 제약 조건을 학습시킨 점이 매우 효율적인 설계라고 판단된다.

**한계 및 비판적 해석**:
ScanNet200 실험 결과에서 언급되었듯이, 클래스 수가 매우 많은 환경에서는 Superpoint 기반 방식이 포인트 단위(point-wise) 방식보다 불리할 수 있다는 점이 드러났다. 이는 Superpoint 생성 단계에서의 정보 손실이 미세한 클래스 구분에는 걸림돌이 될 수 있음을 시사한다. 또한, 추론 시 상위 100개의 인스턴스만 선택하는 방식은 객체가 매우 많은 복잡한 장면에서 일부 객체를 누락시킬 가능성이 있다.

**결론**:
결과적으로 MSTA3D는 연산 효율성과 세그멘테이션 정확도 사이의 균형을 잘 잡은 모델이며, 특히 대형 객체 인식 능력을 획기적으로 높였다는 점에서 학술적 가치가 크다.

## 📌 TL;DR

MSTA3D는 3D 인스턴스 세그멘테이션에서 발생하는 **과분할(Over-segmentation)** 문제를 해결하기 위해 **Multi-scale Superpoint** 특징과 **Twin-attention Decoder**, 그리고 **Box Regularizer**를 제안한 프레임워크이다. 이를 통해 대형 객체 및 배경에 대한 인식 성능을 크게 향상시켰으며, 기존 SOTA 모델 대비 더 적은 파라미터로 더 높은 정확도를 달성하였다. 이 연구는 향후 3D 장면 이해 및 로보틱스 분야에서 정교한 객체 분리가 필요한 작업에 핵심적인 역할을 할 가능성이 높다.
