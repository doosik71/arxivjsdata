# Gated Feedback Refinement Network for Coarse-to-Fine Dense Semantic Image Labeling

Md Amirul Islam, Mrigank Rochan, Shujon Naha, Neil D. B. Bruce, and Yang Wang

## 🧩 Problem to Solve

의미론적 분할(Semantic Segmentation) 및 밀집 이미지 레이블링(Dense Image Labeling)은 이미지의 픽셀 단위로 정확한 레이블을 할당하는 작업입니다. 기존의 심층 합성곱 신경망(ConvNets)은 특징 추출 과정에서 풀링(pooling) 등으로 인해 공간 해상도(spatial resolution)를 잃어버려 픽셀 수준의 정밀도를 확보하기 어렵습니다. 반면 초기 계층은 공간적 정밀도는 높지만, 분류 모호성(categorical ambiguity)이 커서 예측 품질을 저해할 수 있습니다. 따라서, 미세한 공간적 세부 정보와 풍부한 특징 표현을 효과적으로 통합하고, 초기 계층의 모호한 정보를 해결하면서 세밀한 픽셀 레이블링을 수행하는 것이 주요 과제입니다.

## ✨ Key Contributions

- **Coarse-to-Fine 예측:** 최종 분할 결과를 한 번에 예측하는 대신, 먼저 거친(coarse) 레이블링을 예측한 다음, 이를 점진적으로 미세하게(finer) 다듬어 나가는 새로운 관점을 제시합니다.
- **새로운 게이팅(Gating) 메커니즘 도입:** 인코더에서 디코더로 정보가 전달될 때 정보 흐름을 조절하는 게이팅 메커니즘을 제안합니다. 이 메커니즘은 네트워크가 객체 범주에 대한 모호성을 걸러낼 수 있도록 하며, 의미론적 분할에서 지역적(local) 및 전역적(global) 문맥 정보를 결합하기 위해 인코더-디코더 프레임워크에 게이팅 메커니즘을 사용한 최초의 접근 방식입니다.
- **다중 해상도 심층 감독(Deep Supervision):** 대부분의 기존 방법과 달리, 네트워크의 여러 단계에서 손실 함수를 정의하여 다중 해상도에서 감독을 제공합니다.
- **일반화된 아키텍처:** 제안하는 네트워크 아키텍처는 의미론적 이미지 분할에 중점을 두지만, 다른 픽셀 단위 레이블링 작업에도 적용할 수 있을 만큼 일반적입니다.
- **G-FRNet 확장:** 이전 버전인 G-FRNet을 확장하여 LRN(Label Refinement Network)을 기본 모델로 제시하고, PASCAL-Person-Part 및 SUN-RGBD 데이터셋에 대한 추가 실험을 수행하며, 심층 감독 기법의 역할을 분석하고, 게이팅 메커니즘의 대체 설계(예: 곱셈 대신 덧셈 상호작용)를 탐색합니다.

## 📎 Related Works

- **CNN 기반 밀집 레이블링:** FCN(Fully Convolutional Networks) [4], DeepLab-CRF [17], DeepLabv2 [22], CRF-RNN [23], Dilated Convolution [24] 등 CNN을 픽셀 단위 예측에 적용한 연구들.
- **다중 스케일 처리 및 정제:** Laplacian pyramids [25], 다중 스케일 CNN [26], coarse-to-fine 이미지 생성 [27], Hypercolumn [19], PixelNet [29], SegNet [6], DeconvNet [7] 등 여러 스케일의 특징을 활용하거나 점진적으로 해상도를 복구하는 방법.
- **심층 감독:** Inception [16], Deeply-supervised Network [30], Holistically-nested Edge Detection [31] 등 네트워크의 여러 중간 계층에 보조 손실을 추가하여 학습을 강화하는 기법.
- **하향식 변조 및 피드백 정제:** coarse-grained 예측 맵을 정제하는 [32, 33, 34, 35, 36] 연구들. 특히 인간 시각 경로의 신경 정보 처리(top-down modulation)에서 영감을 받아 모호한 표현을 고수준 특징으로 조절하는 방식 [10, 37, 38, 39]과 유사성을 가집니다.

## 🛠️ Methodology

본 논문은 `Label Refinement Network (LRN)`와 이를 기반으로 확장된 `Gated Feedback Refinement Network (G-FRNet)` 두 가지 아키텍처를 제안합니다.

**1. Label Refinement Network (LRN):**

- **아키텍처:** VGG16 기반의 인코더-디코더 프레임워크를 사용합니다.
- **Coarse-to-Fine 예측:** 네트워크의 끝에서 한 번만 예측하는 대신, 디코더 네트워크는 여러 단계에서 coarse-to-fine 방식으로 예측을 수행합니다.
  - **초기 Coarse 예측:** 인코더의 마지막 특징 맵 $f_7(I)$를 $3 \times 3$ 합성곱을 통해 클래스 개수 $C$에 해당하는 채널을 가진 coarse 레이블 맵 $P_{G}^{m}$으로 변환합니다.
  - **점진적 정제:** 이전 단계의 업샘플링된 레이블 맵과 인코더의 해당 특징 맵(skip connection)을 연결(concatenate)한 후 $3 \times 3$ 합성곱을 적용하여 더 큰 공간 해상도의 정제된 레이블 맵 $P_{RU_k}^{m}$을 생성합니다.
    $$P_{RU_k}^{m} = \text{conv}_{3 \times 3}(\text{concat}(U(P_{RU_{k-1}}^{m}), f_{7-k}(I)))$$
- **심층 감독:** 각 디코딩 단계에서 생성된 레이블 맵 $P_{RU_k}^{m}$과 해당 크기로 조정된 Ground-truth $R_k(Y)$ 간의 차이를 측정하는 손실 함수 $l_k$ (cross-entropy loss)를 정의하여 네트워크 학습에 추가적인 감독을 제공합니다. 총 손실 $\mathcal{L} = \sum_{k=1}^{6} l_k$를 최적화합니다.

**2. Gated Feedback Refinement Network (G-FRNet):**

- LRN을 기반으로 하지만, skip connection을 통해 정보가 전달될 때 `게이트 유닛(Gate Unit)`을 사용하여 정보를 변조(modulate)하는 것이 주요 특징입니다.
- **게이트 유닛 (Gate Unit):**
  - 두 개의 연속적인 인코더 특징 맵 $f_g^i$ (높은 해상도, 작은 수용장)와 $f_g^{i+1}$ (낮은 해상도, 큰 수용장)를 입력으로 받습니다.
  - $f_g^{i+1}$은 더 깊은 계층의 특징으로, $f_g^i$에 존재하는 모호성을 해결하는 데 도움을 줄 수 있다는 직관을 활용합니다.
  - 두 특징 맵에 각각 $3 \times 3$ 합성곱, 배치 정규화, ReLU를 적용한 후, $f_g^{i+1}$을 $f_g^i$와 공간 차원이 일치하도록 업샘플링합니다.
  - 최종 게이트된 특징 맵 $M_f$는 $f_g^i$와 변환된 $f_g^{i+1}$의 요소별 곱($\odot$)으로 계산됩니다.
    $$M_f = v_i \odot u_i \quad \text{where} \quad v_i = T_f(f_g^{i+1}), u_i = T_f(f_g^i)$$
  - 곱셈 방식의 게이팅은 고수준 특징에 의해 '잘못된' 활성화가 강력하게 억제될 수 있도록 합니다.
- **게이트 정제 유닛 (Gated Refinement Unit, RU):**
  - 이전 단계의 coarse 레이블 맵 $R_f$와 게이트 유닛에서 나온 게이트된 특징 맵 $M_f$를 입력으로 받습니다.
  - $M_f$에 합성곱 및 배치 정규화를 적용하여 $m_f$를 얻습니다: $m_f = B(C_{3 \times 3}(M_f))$.
  - $m_f$와 $R_f$를 연결하여($\gamma = m_f \oplus R_f$) 새로운 특징 맵을 생성합니다. 특징 맵 채널 차이로 인한 정보 손실을 막기 위해 $m_f$의 채널 수를 $R_f$와 동일하게 맞춥니다.
  - 마지막으로, $3 \times 3$ 합성곱을 통해 정제된 레이블 맵 $R'_f$를 생성합니다. $R'_f$는 다음 단계로 전달되기 위해 2배 업샘플링됩니다.

## 📊 Results

- **데이터셋:** CamVid, PASCAL VOC 2012, Horse-Cow Parsing, PASCAL-Person-Part, SUN-RGBD 등 5가지 벤치마크 데이터셋에서 실험을 수행했습니다.
- **CamVid:** G-FRNet은 68.0%의 평균 IoU(Intersection over Union)를 달성하여 SegNet, DilatedNet, FSO, DeepLab 등 기존 최신 기술들을 능가하는 SOTA(State-of-the-Art) 성능을 보였습니다. 특히 얇고 작은 객체(예: 기둥, 자전거 타는 사람)의 정밀도가 향상되었습니다.
- **PASCAL VOC 2012:**
  - VGG-16 기반 G-FRNet + CRF는 71.0%의 평균 IoU를 달성하여 기존 인코더-디코더 기반 모델 중 가장 우수한 성능을 보였습니다.
  - ResNet-101 기반 G-FRNet-Res101 + CRF는 검증 세트에서 77.8%, 테스트 세트에서 79.3%의 평균 IoU를 달성하며 최신 SOTA 방법들과 경쟁력 있는 결과를 보였습니다.
  - **심층 감독의 중요성:** 심층 감독이 없는 G-FRNet-Res101(w/o DS)은 71.8%의 평균 IoU를 기록하여, 심층 감독을 사용했을 때(76.5%)보다 4.7% 낮은 성능을 보여 심층 감독 전략의 중요성을 입증했습니다.
- **Horse-Cow Parsing:** G-FRNet은 Horse Parsing에서 70.83%, Cow Parsing에서 65.35%의 평균 IoU를 달성하며 모든 기준선을 능가했습니다. 유사한 객체 부분 구별에 효과적임을 입증했습니다.
- **PASCAL-Person-Part 및 SUN-RGBD:** G-FRNet-Res101은 각각 64.61% 및 36.86%의 평균 IoU를 달성하며 경쟁력 있는 성능을 보였습니다.
- **단계별 분석 (Ablation Study):** coarse 예측 $P_{G}^{m}$부터 최종 $P_{RU_5}^{m}$까지 모든 데이터셋에서 평균 IoU가 점진적으로 향상되는 것을 확인했습니다. LRN과 비교했을 때, G-FRNet은 모든 정제 단계에서 게이트 유닛의 포함으로 인한 성능 향상을 일관되게 보여주었습니다.
- **모델 파라미터 효율성:** G-FRNet은 FCN 및 DeconvNet에 비해 12~25% 적은 모델 파라미터 수로도 매우 경쟁력 있는 성능을 달성하여 제안된 게이팅 메커니즘의 효율성을 입증했습니다.
- **게이팅 메커니즘 설계:** 요소별 곱셈($\odot$)을 사용하는 게이트 유닛이 덧셈 상호작용보다 더 나은 성능(PASCAL VOC 검증 세트에서 68.7% vs. 66.76%)을 보였습니다.

## 🧠 Insights & Discussion

- **게이팅 메커니즘의 효과:** 게이트 유닛은 초기 인코더 계층의 모호한 정보를 더 깊고 식별력이 높은 계층의 정보로 효과적으로 변조하고 필터링합니다. 이는 특히 미세한 세부 사항이나 시각적으로 유사한 범주에 대해 더욱 정밀한 분할을 가능하게 합니다. 곱셈 방식의 게이팅은 부정확한 표현을 강력하게 억제하는 능력을 제공하여 모호성 해결에 더욱 효과적입니다.
- **Coarse-to-Fine 정제:** 점진적인 정제 과정은 깊은 계층에서 손실된 공간적 정밀도를 효과적으로 복구하여 최종 예측의 품질을 향상시킵니다.
- **심층 감독의 중요성:** 중간 단계에서 손실 함수를 제공하는 심층 감독은 정제 기반 아키텍처의 학습과 성능 향상에 결정적인 역할을 합니다.
- **모델 효율성 및 일반성:** G-FRNet은 적은 수의 파라미터로도 뛰어난 성능을 발휘하며, 이는 게이팅 메커니즘이 간단하지만 강력한 아키텍처 수정임을 시사합니다. 이 모델은 의미론적 분할 외의 다른 픽셀 단위 레이블링 문제에도 쉽게 적용될 수 있습니다.
- **인간 시각과의 유사성:** 고수준 특징이 저수준의 모호한 표현을 조절하는 인간 시각 시스템의 작동 방식과 유사성을 가집니다.
- **향후 연구 방향:** 중간 단계 예측의 정확성보다는 오류를 수정하고 변조하는 메커니즘에 초점을 맞춰, 오류 교정적인 반복 게이트 정제(error correcting iterative gated refinement)를 탐구할 수 있습니다. 이는 G-FRNet에서 제안된 모호성 해결 메커니즘이 제공하는 추가적인 여유를 통해 더 강력한 표현 능력을 가진 네트워크를 가능하게 할 수 있습니다.

## 📌 TL;DR

- **문제:** 의미론적 분할에서 심층 CNN의 공간 해상도 손실과 초기 계층의 특징 모호성을 해결하여 픽셀 수준의 정밀도를 달성하는 것이 중요합니다.
- **해결책:** coarse-to-fine 방식으로 예측을 점진적으로 정제하는 `Gated Feedback Refinement Network (G-FRNet)`를 제안합니다. 이 네트워크는 고유한 **게이트 유닛**을 사용하여, 더 깊은 계층의 식별력 있는 특징을 활용해 스킵 연결을 통해 전달되는 모호한 정보를 필터링하고 변조합니다. 또한, 네트워크의 여러 단계에서 심층 감독을 적용합니다.
- **결과:** G-FRNet은 CamVid, PASCAL VOC 2012, Horse-Cow Parsing 등 여러 까다로운 밀집 레이블링 데이터셋에서 SOTA 또는 경쟁력 있는 성능을 달성했습니다. 특히 게이트 유닛을 통한 정제와 심층 감독의 효과를 입증하며, 파라미터 효율성 또한 뛰어남을 보여주었습니다.
