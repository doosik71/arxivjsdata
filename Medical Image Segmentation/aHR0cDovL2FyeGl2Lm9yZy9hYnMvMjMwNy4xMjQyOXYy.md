# SwIPE: Efficient and Robust Medical Image Segmentation with Implicit Patch Embeddings

Yejia Zhang, Pengfei Gu, Nishchal Sapkota, and Danny Z. Chen (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 기존의 이산적 표현(Discrete Representation) 방식이 가진 한계점을 해결하고자 한다. 

전통적인 방식인 래스터화된 마스크(Rasterized Masks) 형태의 이산적 표현은 다음과 같은 세 가지 주요 문제를 야기한다. 첫째, 공간적 유연성이 부족하여 고해상도 이미지로 확장할 때 메모리 사용량이 급격히 증가(Quadratic or Cubic increase)하거나, 결과물을 보간(Interpolation)할 경우 이산화 아티팩트(Discretization artifacts)가 발생한다. 둘째, 픽셀 또는 복셀 단위의 학습은 객체의 전반적인 형태나 경계(Shape/Boundaries)를 직접적으로 모델링하지 못하므로, 특히 데이터셋이 제한적이거나 분포 외(Out-of-distribution) 데이터가 입력될 때 비현실적인 형태의 예측 결과를 생성하는 경향이 있다.

최근 이를 해결하기 위해 Implicit Neural Representations (INRs)를 도입한 연구들이 있었으나, 이들은 주로 3D 형상 복원(Shape Reconstruction) 설계를 그대로 차용하여, 전역적 문맥(Global context)에 치중하여 세부 경계 묘사가 부족하거나, 반대로 포인트 단위(Point-based) 정보에만 의존하여 전역적인 형상 일관성이 떨어지는 문제가 있었다. 따라서 본 논문의 목표는 국소적인 세부 경계 묘사와 전역적인 형상 일관성을 동시에 달성할 수 있는 효율적이고 강건한 의료 영상 분할 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체를 포인트나 이미지 전체 단위가 아닌, **패치 단위(Patch-level)의 암시적 신경 표현(INR)**으로 학습하는 것이다. 이를 위해 다음과 같은 설계를 제안한다.

1.  **Patch-based INR 도입**: 이미지를 패치 단위로 분해하여 표현함으로써 국소적인 경계 정밀도와 전역적인 형태 유지라는 두 마리 토끼를 잡고자 한다.
2.  **Multi-stage Embedding Attention (MEA)**: 다양한 스케일의 특징 맵(Feature maps)에서 각 패치의 특성에 맞게 전역적/추상적 정보와 국소적/세부 정보를 동적으로 선택하여 융합하는 어텐션 메커니즘을 제안한다.
3.  **Stochastic Patch Overreach (SPO)**: 패치 경계에서 발생할 수 있는 불연속성 문제를 해결하기 위해, 특정 패치의 임베딩이 인접 패치의 좌표에 대해서도 예측을 수행하도록 강제하는 확률적 정규화 기법을 제안한다.

## 📎 Related Works

기존의 의료 영상 분할 연구는 주로 CNN 기반의 U-Net이나 Transformer 기반의 UNETR와 같은 이산적 데이터 표현 방식을 사용하였다. 이러한 방식은 매우 효과적이지만, 앞서 언급한 대로 해상도 변화에 취약하고 형상에 대한 직접적인 이해가 부족하다는 한계가 있다.

이를 대체하기 위해 등장한 INR 기반 접근 방식 중 OSSNet은 전역 임베딩(Global embedding)을 사용하여 형상의 일관성을 높였으나 세부 경계 표현이 부족했다. 반면 IFA, IOSNet, NUDF 등은 포인트 단위의 특징을 추출하여 세밀함을 높였으나, 전역적인 문맥 이해가 부족하여 이산적 분할 방식과 마찬가지로 제약 없는(Unconstrained) 예측 문제(비현실적인 형태 생성)를 겪었다. SwIPE는 이러한 포인트 기반 방식과 전역 기반 방식의 중간 지점인 '패치 기반 표현'을 도입하여 두 방식의 단점을 상쇄하고 차별화를 꾀하였다.

## 🛠️ Methodology

### 전체 시스템 구조
SwIPE의 전체 파이프라인은 **인코더(Encoder) $\rightarrow$ 넥(Neck) $\rightarrow$ 디코더(Decoder)** 구조로 이루어져 있다. 입력 이미지를 패치 임베딩($z_P$)과 이미지 임베딩($z_I$)으로 인코딩한 후, 이를 좌표 정보($p$)와 함께 MLP 디코더에 입력하여 각 좌표의 점유 확률(Occupancy score)을 예측한다.

### 주요 구성 요소 및 역할

**1. Image Encoding and Patch Embeddings**
- **Backbone ($E_b$)**: Res2Net-50과 같은 완전 합성곱 인코더를 사용하여 4개의 다중 스케일 특징 맵 $\{F_n\}_{n=2}^5$를 생성한다.
- **Neck ($E_n$)**: RFB-Lite(Receptive Field Block-Lite)를 통해 문맥 정보를 강화하고, 이를 다운샘플링하여 동일한 크기의 중간 임베딩 $\{F'_n\}_{n=2}^5$를 생성한다.
- **MEA (Multi-stage Embedding Attention)**: 각 위치의 4개 스케일 임베딩 벡터 $\{e_n\}_{n=2}^5$를 입력받아 가중치 $W$를 계산하고, 이를 통해 최종 패치 임베딩 $z_P$를 생성한다.
    - 가중치 계산: $W = \text{Softmax}(\text{MLP}_1(\text{cat}(\text{MLP}_0(e_2), \dots, \text{MLP}_0(e_5))))$
    - 최종 임베딩: $z_P = \text{MLP}_2(\sum_{n=2}^5 e_n + \sum_{n=2}^5 w_{n-2} \cdot e_n)$

**2. Implicit Patch Decoding**
- **Patch Decoder ($D_P$)**: 국소 패치 임베딩 $z_P$와 상대 좌표 $p_P$, 전역 정보 $z_I, p_I$, 그리고 소스 이미지 좌표 $p_S$를 입력받아 점유 확률 $\hat{o}_P$를 예측한다.
- **Image Decoder ($D_I$)**: 전역 임베딩 $z_I$와 이미지 좌표 $p_I$만을 사용하여 전체적인 형상 $\hat{o}_I$를 예측한다.
- **SPO (Stochastic Patch Overreach)**: 학습 시 무작위로 인접 패치의 임베딩을 선택하여 현재 좌표의 값을 예측하게 함으로써, 패치 간 경계의 연속성을 보장한다.

### 훈련 목표 및 손실 함수
학습은 Latin Hypercube sampling을 통해 샘플링된 포인트 세트 $\{p^S_i, o_i\}$에 대해 수행된다.

- **점유 손실 ($L_{occ}$)**: Cross Entropy($L_{ce}$)와 Dice Loss($L_{dc}$)를 동일 가중치로 합산하여 사용한다.
    - $L_{occ}(o_i, \hat{o}_i) = 0.5 \cdot L_{ce}(o_i, \hat{o}_i) + 0.5 \cdot L_{dc}(o_i, \hat{o}_i)$
- **전체 손실 함수 ($L$)**:
    $$L = \alpha L_{occ}(o_i, \hat{o}_P) + (1-\alpha) L_{occ}(o_i, \hat{o}_I) + \beta L_{SPO}(o_i, \hat{o}'_i) + \lambda (\|z_P\|^2_2 + \|z_I\|^2_2)$$
    여기서 $\alpha$는 국소-전역 균형 계수, $\beta$는 SPO 가중치, $\lambda$는 임베딩 정규화 계수이다.

## 📊 Results

### 실험 설정
- **데이터셋**: 2D 폴립 분할(Kvasir-Sessile, CVC-ClinicDB) 및 3D 복부 장기 분할(BCV, AMOS).
- **기준선(Baselines)**: 이산적 방식(U-Net, PraNet, UNETR, Res2UNet) 및 암시적 방식(OSSNet, IOSNet).
- **지표**: Dice Score.

### 주요 결과
- **정량적 성능**: 2D 폴립 분할에서 기존 암시적 방법 대비 +6.7%, 최신 이산적 방법(PraNet) 대비 +2.5%의 Dice score 향상을 보였다. 3D 장기 분할에서도 UNETR를 근소하게 앞서거나 대등한 성능을 보였다.
- **모델 효율성**: 특히 파라미터 수 측면에서 매우 효율적이다. PraNet 대비 약 1/10 수준의 파라미터만으로 더 높은 성능을 달성하였다.
- **강건성(Robustness)**:
    - **해상도 변화**: 출력 해상도를 변경하여 테스트했을 때, 이산적 방식은 보간법으로 인해 성능이 급감하지만, SwIPE는 INR의 특성상 일관된 성능을 유지하였다.
    - **데이터셋 전이**: Sessile $\rightarrow$ CVC, BCV $\rightarrow$ AMOS 전이 학습 환경에서도 다른 방식들보다 높은 Dice score를 기록하여 일반화 능력이 뛰어남을 입증하였다.
- **데이터 효율성**: 학습 데이터의 양을 10%, 25%, 50%로 줄였을 때, 다른 모델들에 비해 성능 하락 폭이 완만하여 적은 양의 데이터로도 효율적인 학습이 가능함을 보였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 INR의 적용 범위를 '패치 단위'로 확장함으로써 기존의 전역-국소 트레이드오프 문제를 효과적으로 해결하였다. 특히 MEA 모듈을 통해 각 패치가 필요로 하는 정보의 추상화 수준(Global vs Local)을 동적으로 결정하게 한 점이 성능 향상의 핵심으로 분석된다.

또한, SPO 기법은 INR 기반 모델들이 흔히 겪는 경계 불연속성 문제를 정규화 관점에서 해결하여, 결과적으로 더 매끄럽고 현실적인 객체 형상을 생성하게 하였다.

비판적으로 해석하자면, 본 연구는 파라미터 효율성과 강건성 면에서 압도적인 이점을 보였으나, 3D 데이터셋(BCV)에서의 성능 향상 폭이 2D에 비해 상대적으로 작다는 점은 향후 복잡한 3D 구조에 대한 추가적인 모델링 개선이 필요함을 시사한다.

## 📌 TL;DR

SwIPE는 의료 영상 분할을 위해 **패치 기반의 암시적 신경 표현(Implicit Neural Representations)**을 도입한 모델이다. MEA(동적 특징 융합)와 SPO(경계 연속성 정규화)를 통해 **최신 이산적 분할 모델보다 10배 적은 파라미터로 더 높은 정확도**를 달성하였으며, 특히 해상도 변화와 데이터셋 전이에 대해 매우 강건한 성능을 보인다. 이는 향후 고해상도 의료 영상 분석 및 데이터 부족 환경에서의 분할 연구에 중요한 방향성을 제시한다.