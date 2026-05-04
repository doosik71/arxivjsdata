# Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation

Minghan Li, Shuai Li, Lida Li, Lei Zhang (2021)

## 🧩 Problem to Solve

본 논문은 현대적인 One-stage Video Instance Segmentation(VIS) 네트워크가 가진 두 가지 주요 한계점을 해결하고자 한다.

첫째, Convolutional features가 Anchor boxes 또는 Ground-truth bounding boxes와 정렬(aligned)되지 않아, 마스크 생성 시 공간적 위치에 대한 민감도(spatial location sensitivity)가 떨어진다는 점이다. 기존의 One-stage 방식은 다양한 모양의 앵커들이 동일한 컨볼루션 피처를 공유하기 때문에, 객체의 실제 형태와 피처의 수용 영역(receptive field) 사이의 불일치가 발생한다.

둘째, 비디오를 개별 프레임으로 나누어 프레임 단위의 인스턴스 세그멘테이션을 수행함으로써, 인접 프레임 간의 시간적 상관관계(temporal correlation)를 무시한다는 점이다. 이로 인해 모션 블러(motion blur), 부분 가려짐(partial occlusion), 또는 특이한 객체-카메라 포즈와 같은 까다로운 상황에서 성능이 저하되는 문제가 발생한다.

따라서 본 연구의 목표는 공간적 피처 교정(Spatial Feature Calibration)과 시간적 융합(Temporal Fusion)을 통해 정확도와 속도 사이의 균형을 맞춘 효율적인 One-stage VIS 프레임워크인 **STMask**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 One-stage 네트워크의 속도 이점을 유지하면서, Two-stage 네트워크가 가진 공간적 정밀도와 시간적 연속성을 도입하는 것이다.

1. **Spatial Feature Calibration**: 앵커 박스와 실제 바운딩 박스에 맞게 피처를 정렬하는 두 가지 전략(FCA, FCB)을 제안하여 위치 민감도를 높였다.
2. **Temporal Fusion Module**: 인접 프레임 간의 피처를 융합하여 이전 프레임의 인스턴스 마스크를 현재 프레임으로 추론함으로써, 탐지 누락을 줄이고 추적 성능을 향상시켰다.
3. **Effective Trade-off**: ResNet-50/101 백본을 사용하여 높은 Mask AP를 달성함과 동시에 실시간에 가까운 FPS(28.6 / 23.4)를 확보하였다.

## 📎 Related Works

### 관련 연구 및 한계

- **Image Instance Segmentation**: Bottom-up 방식과 Top-down 방식으로 나뉜다. Mask R-CNN과 같은 Top-down 방식은 RoIAlign 등을 통해 피처를 정렬하여 높은 정확도를 보이지만 속도가 느리다. 반면 Yolact와 같은 One-stage 방식은 Prototype과 Mask Coefficient의 선형 결합을 통해 속도를 높였으나 공간적 정렬이 부족하다.
- **Video Object Detection**: 모션 블러나 가려짐 문제를 해결하기 위해 Optical flow, Correlation operation, Deformable convolution 등을 사용하여 프레임 간 피처를 전파하거나 정렬한다.
- **Video Instance Segmentation**: 주로 Mask R-CNN에 트래킹 브랜치를 추가한 형태로 발전했다. MaskProp과 같은 최신 기법은 클립 단위의 마스크 전파를 통해 최상위 성능을 보이지만, 처리 속도가 매우 느려 실시간 적용이 어렵다.

### 차별점

기존의 One-stage VIS 방식(예: SipMask)은 단순히 트래킹 브랜치를 추가하거나 일부 바운딩 박스 정렬에만 집중했다. STMask는 앵커 단계(FCA)와 바운딩 박스 단계(FCB) 모두에서 공간적 교정을 수행하며, 동시에 Correlation operation 기반의 시간적 융합 모듈을 통해 시간적 중복성을 적극적으로 활용한다는 점에서 차별화된다.

## 🛠️ Methodology

STMask의 전체 구조는 **Frame-level Instance Segmentation**과 **Cross-frame Instance Segmentation**의 두 단계로 구성된다.

### 1. Spatial Calibration

#### 1.1 Feature Calibration for Anchors (FCA)

기존 One-stage 검출기는 모든 앵커에 대해 동일한 $3 \times 3$ 컨볼루션을 사용한다. 하지만 큰 앵커는 더 큰 수용 영역이 필요하다. 이를 위해 STMask는 서로 다른 종횡비(aspect ratio)를 가진 여러 개의 컨볼루션 커널($3 \times 3, 3 \times 5, 5 \times 3$)을 도입하고, 이에 맞춰 앵커의 종횡비도 $[1, 3/5, 5/3]$로 조정하여 피처와 앵커 간의 정렬을 맞춘다.

#### 1.2 Feature Calibration for Bounding Boxes (FCB)

앵커 수준의 교정 이후에도 예측된 바운딩 박스와 피처 사이에는 여전히 불일치가 존재한다. 이를 해결하기 위해 먼저 바운딩 박스를 회귀(regression)하여 예측하고, 그 영역에서 피처를 다시 추출한다.

바운딩 박스 회귀를 통해 얻은 변환 값 $d = [d_x, d_y, d_w, d_h]$를 사용하여 예측된 바운딩 박스 $B$를 계산한다:
$$B_x = P_w d_x + P_x, \quad B_y = P_h d_y + P_y$$
$$B_w = P_w \exp(d_w), \quad B_h = P_h \exp(d_h)$$

이후 2D Deformable Convolution을 사용하여 피처를 추출하며, 오프셋 $O$를 결정하는 두 가지 전략을 사용한다:

- **Adaptive Features**: $1 \times 1$ 컨볼루션 층을 통해 변환 값 $d$로부터 학습된 오프셋 $O = \mathcal{N}_O(d)$를 예측한다.
- **Aligned Features**: 수학적 기하학 유도를 통해 오프셋을 직접 계산한다.
$$O = (\Delta y, \Delta x) \mathbf{I} + (\Delta h, \Delta w) \mathbf{R}$$
여기서 $\Delta x = k_w d_x, \Delta w = \exp(d_w) - 1$ 등과 같이 정의되며, 이는 Two-stage 네트워크의 RoIAlign 연산의 특수한 사례와 동일하다.

### 2. Temporal Fusion Module

시간적 상관관계를 활용하기 위해 $t-1$ 프레임의 인스턴스를 $t$ 프레임으로 전파한다.

#### 2.1 프레임 단위 세그멘테이션

Yolact와 유사하게 마스크를 인스턴스 독립적인 Prototype $P$와 인스턴스 특화된 Mask Coefficient $C$의 결합으로 생성한다:
$$M_i = \text{Crop}(\sigma(PC_i), B_i)$$

#### 2.2 크로스 프레임 세그멘테이션 (TemporalNet)

인접한 두 프레임 $I_{t-1}, I_t$의 FPN 피처와 이들의 Correlation operation 결과($x_{t-1,t}^{\text{corr}}$)를 결합하여 TemporalNet에 입력한다. TemporalNet은 다음 두 가지를 예측한다:

1. **바운딩 박스 변위 $d_{t-1,t}$**: 이전 프레임의 박스 $B_{t-1}$을 현재 프레임의 박스 $B_{t-1,t}$로 업데이트한다.
2. **마스크 계수 변화량 $\Delta C_{t-1,t}$**: $C_{t-1,t} = C_{t-1} + \Delta C_{t-1,t}$를 통해 현재 프레임의 계수를 추론한다.

최종적으로 크로스 프레임 마스크 $M_{t-1,t}$는 다음과 같이 생성된다:
$$M_{t-1,t} = \text{Crop}(\sigma(P_t C_{t-1,t}), B_{t-1,t})$$

#### 2.3 마스크 병합 및 ID 할당

프레임 단위로 예측된 마스크 $M_t$와 크로스 프레임으로 추론된 마스크 $M_{t-1,t}$를 병합한다. 매칭 점수 $s_{ij}^t$는 임베딩 벡터의 코사인 유사도와 Mask IoU의 가중 합으로 계산한다:
$$s_{ij}^t = \alpha \frac{E_i^t \cdot E_j^{t-1}}{\|E_i^t\| \|E_j^{t-1}\|} + \beta \text{MIoU}(M_i^t, M_{t-1,t}^j)$$
이 점수가 임계값 $\epsilon$보다 높으면 기존 ID를 부여하고, 낮으면 새로운 ID를 부여한다. 만약 프레임 단위 탐지에서 누락되었으나 크로스 프레임 추론으로 살아남은 인스턴스가 있다면, 이를 최종 결과에 보충(supplement)한다.

## 📊 Results

### 실험 설정

- **데이터셋**: YouTube-VIS valid set.
- **백본**: ResNet-50, ResNet-101 (DCN 적용).
- **지표**: Mask AP, FPS.
- **학습**: MS COCO 사전 학습 모델 사용, 160k iteration 학습.

### 주요 결과

- **정량적 성과**:
  - **ResNet-50**: Mask AP 33.5%, 속도 28.6 FPS.
  - **ResNet-101**: Mask AP 36.8%, 속도 23.4 FPS.
- **비교 분석**:
  - One-stage 방식인 SipMask++(R101)의 35.0% AP보다 높은 36.8% AP를 달성하였다.
  - Two-stage 방식인 MaskProp(R101)의 42.5% AP보다는 낮지만, MaskProp은 처리 속도가 매우 느려 실시간성이 없다. STMask는 정확도와 속도 사이에서 최적의 Trade-off를 달성하였다.
- **절제 연구 (Ablation Study)**:
  - FCA(+1.9%), FCB(+2.3~2.4%), TF(+5.2%) 순으로 성능 향상에 기여했으며, 특히 시간적 융합(TF) 모듈의 영향력이 가장 컸다.
  - FCB에서는 수학적 정렬(Aligned)보다 학습 가능한 적응형 피처(Adaptive)가 약간 더 좋은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 One-stage VIS의 고질적인 문제인 '공간적 불일치'와 '시간적 단절'을 매우 효율적인 구조로 해결하였다. 특히 복잡한 연산 대신 컨볼루션 커널의 종횡비 조정(FCA)과 단순한 변위 예측 기반의 TemporalNet을 사용함으로써, 추론 속도를 크게 유지하면서도 Two-stage 모델의 정밀함에 근접하는 성과를 냈다.

### 한계 및 논의

1. **정확도의 한계**: MaskProp과 같은 클립 기반 전파 방식보다 AP가 낮은 이유는, 본 모델이 인접한 두 프레임 간의 정보만 활용하기 때문이다. 비디오 전체의 컨텍스트를 활용한다면 성능을 더 높일 수 있겠으나, 이는 속도 저하를 초래할 것이다.
2. **가정**: Temporal fusion 과정에서 이전 프레임의 마스크 계수 $C_{t-1}$과 현재 프레임의 $C_t$가 서로 '가깝다'고 가정한다. 이는 급격한 장면 전환이나 매우 빠른 객체 이동 시에는 한계가 있을 수 있다.
3. **적응형 vs 정렬 피처**: Adaptive features가 Aligned features보다 성능이 좋게 나온 점은, 수학적 정의보다 데이터로부터 직접 학습된 오프셋이 실제 객체의 변형을 더 잘 포착함을 시사한다.

## 📌 TL;DR

STMask는 One-stage Video Instance Segmentation에서 발생하는 **공간적 피처 불일치**와 **시간적 정보 무시** 문제를 해결한 프레임워크이다. 앵커 및 바운딩 박스 수준의 **Spatial Calibration(FCA, FCB)**과 인접 프레임 간 정보를 융합하는 **Temporal Fusion Module**을 도입하여, 실시간에 가까운 속도(최대 28.6 FPS)를 유지하면서도 높은 마스크 정확도(최대 36.8% AP)를 달성하였다. 이 연구는 고속 VIS 시스템에서 공간/시간적 정렬의 중요성을 입증하였으며, 향후 실시간 비디오 분석 시스템의 기본 구조로 활용될 가능성이 높다.
