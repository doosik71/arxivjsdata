# Learning Gaussian Instance Segmentation in Point Clouds

Shih-Hung Liu, Shang-Yi Yu, Shao-Chi Wu, Hwann-Tzong Chen, and Tyng-Luh Liu (2020)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드(Point Cloud)에서의 **인스턴스 세그멘테이션(Instance Segmentation)** 문제를 해결하고자 한다. 3D 인스턴스 세그멘테이션은 3D 장면 내의 개별 객체를 분리하고 각각에 올바른 클래스 레이블을 할당하는 작업이다.

이 문제는 3D 장면의 복잡성과 데이터의 희소성으로 인해 2D 인스턴스 세그멘테이션보다 훨씬 도전적인 과제이다. 특히 기존의 많은 방법론이 미리 정의된 앵커(Anchor) 박스에 의존하거나, 복잡한 박스 제안(Box Proposal) 단계 및 이후의 비최대 억제(Non-Maximum Suppression, NMS) 과정에 의존함으로써 계산 비용이 증가하고 훈련이 어렵다는 문제점이 있었다. 따라서 본 연구의 목표는 앵커 프리(Anchor-free) 방식의 단일 단계(Single-stage) 아키텍처를 통해 효율적이고 정확한 3D 인스턴스 세그멘테이션을 수행하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Gaussian Instance Center Network (GICN)**라는 센터 중심의 메커니즘을 도입하는 것이다.

GICN은 장면 전체에 흩어져 있는 인스턴스 센터들의 분포를 **가우시안 센터 히트맵(Gaussian Center Heatmaps)**으로 근사하여 예측한다. 이 히트맵을 기반으로 소수의 센터 후보군을 효율적으로 선택하고, 각 센터에 대해 다음의 과정을 수행한다:

1. **인스턴스 크기 예측**: 각 센터의 크기를 예측하여 특징 추출을 위한 적절한 범위를 결정한다.
2. **바운딩 박스 생성**: 크기 인식(Size-aware) 특징을 바탕으로 객체의 바운딩 박스를 예측한다.
3. **인스턴스 마스크 생성**: 최종적으로 객체의 정밀한 마스크를 생성한다.

이러한 설계는 미리 정의된 앵커 박스가 필요 없으며, NMS 과정 없이도 중복 예측을 방지할 수 있는 직관적이고 효율적인 파이프라인을 제공한다.

## 📎 Related Works

기존의 3D 인스턴스 세그멘테이션 연구는 크게 두 가지 표현 방식에 따라 나뉜다:

- **Voxel-based methods**: 3D 공간을 복셀 그리드로 나누어 처리하는 방식(예: MTML, MASC, PanopticFusion)이다.
- **Point-cloud based methods**: 포인트 클라우드에서 직접 특징을 추출하는 방식(예: SGPN, 3D-BoNet, GSPN)이며, 주로 PointNet++와 같은 백본을 사용한다.

또한, 픽셀 또는 포인트 간의 거리(Embedding)를 학습하여 군집화하는 메트릭 학습(Metric Learning) 기반의 접근 방식이 많이 사용되었으나, 이는 Mean-shift clustering과 같은 무거운 후처리 단계가 필요하다는 한계가 있다.

본 논문이 비교 대상으로 삼는 **3D-BoNet**은 앵커 프리 방식을 제안했지만, 고정된 수의 바운딩 박스를 예측한다는 점이 GICN과 다르다. 또한 **VoteNet**은 Hough voting 메커니즘을 사용하지만, GICN은 전체 포인트 클라우드에서 직접 히트맵을 예측하며 객체별로 적응적인(Adaptive) 클러스터 크기를 결정한다는 점에서 차별화된다.

## 🛠️ Methodology

GICN은 크게 세 가지 네트워크 단계로 구성된다.

### 1. Center Prediction Network ($\Phi_C$)

입력 포인트 클라우드 $P=\{p_i = (x_i, y_i, z_i)\}_{i=1}^N$에 대해 각 포인트 $p_i$가 인스턴스의 센터일 확률 $Q_i \in [0, 1]$를 예측하여 히트맵 $Q$를 생성한다. 백본으로는 PointNet++를 사용하며, 최종 출력층에 Sigmoid 함수를 적용한다.

**학습 방법**: 각 인스턴스의 센터에 가장 가까운 포인트를 가우시안 센터로 설정하고, 다른 포인트들과의 거리에 가우시안 함수를 적용하여 Ground-truth 히트맵을 생성한다.

**Center Selection Mechanism**: 예측된 $Q$에서 후보 센터 $C=\{C_t\}_{t=1}^T$를 선택할 때, 단순히 높은 값만 뽑으면 동일 객체에서 여러 포인트가 선택되는 중복 문제가 발생한다. 이를 해결하기 위해 별도의 시맨틱 네트워크($\Phi_S$)를 통해 각 클래스별 대표 반경 $r_\ell$을 정의하고, 한 포인트가 센터로 선택되면 해당 반경 내의 다른 포인트들의 히트맵 값을 0으로 만들어 중복 선택을 방지한다.

### 2. Bounding-Box Prediction Network ($\Phi_B$)

선택된 센터 $C_t$에 대해 적절한 바운딩 박스를 생성하기 위해, 먼저 인스턴스의 크기를 $K$개의 그룹($K=6$) 중 하나로 분류하여 예측한다.

- 예측된 크기 $s_k^*$에 해당하는 이웃 영역에서 PointNet++를 통해 지역 특징(Local features)을 추출한다.
- 추출된 지역 특징과 전역 특징(Global features)을 결합하여 바운딩 박스의 정점 $B_t = \{(x_{min}^t, y_{min}^t, z_{min}^t), (x_{max}^t, y_{max}^t, z_{max}^t)\}$를 예측한다.

### 3. Mask Prediction Network ($\Phi_M$)

앞서 예측된 바운딩 박스 정보를 활용하여 인스턴스 마스크를 생성한다. 포인트별 특징과 전역 특징을 결합하고, 바운딩 박스 좌표 정보를 함께 입력하여 각 포인트가 해당 인스턴스에 속하는지 여부를 결정하는 마스크를 예측한다.

### 4. Loss Functions

전체 네트워크는 다음과 같은 통합 손실 함수 $L_{total}$을 통해 end-to-end로 학습된다:
$$L_{total} = L_{center} + L_{bound} + L_{IoU} + L_{mask} + L_{size}$$

- **$L_{center}$ & $L_{mask}$**: 클래스 불균형 문제를 해결하기 위해 Focal Loss를 사용한다.
  - $L_{center}$의 경우, 가우시안 분포 값 $\hat{G}_{t(i)}$가 임계값 $\sigma_G = 0.4$보다 큰 경우에만 정답으로 간주한다.
- **$L_{size}$**: 크기 그룹 예측을 위한 Cross-Entropy Loss를 사용한다.
- **$L_{bound}$**: 예측된 박스 정점과 정답 정점 사이의 Smooth $l_1$ Loss를 사용한다.
- **$L_{IoU}$**: 박스의 정확도를 높이기 위해 GIoU(Generalized IoU) Loss를 사용한다:
  $$L_{IoU} = \frac{1}{T} \sum_{t=1}^T (1 - \text{GIoU}(B_t, \hat{B}_t))$$

## 📊 Results

### 실험 설정

- **데이터셋**: S3DIS (6-fold cross validation), ScanNet v2 (online benchmark)
- **지표**: S3DIS에서는 mean Precision(mPrec) 및 mean Recall(mRec), ScanNet에서는 AP@50%를 사용한다.
- **구현**: PyTorch 사용, Adam optimizer, 1m$^3$ 큐브 단위로 장면을 분할하여 처리 후 블록 병합(Block-merging) 알고리즘 적용.

### 주요 결과

- **S3DIS 결과**: mPrec 68.5%, mRec 50.8%를 달성하여 기존 SOTA 모델인 3D-BoNet 대비 mAP가 약 2.9% 향상되었다.
- **ScanNet 결과**: mean AP@50% 63.8%를 기록하며 기존 문헌에 발표된 방법들 중 가장 높은 성능을 보였다. 특히 변기(toilet)나 욕조(bathtub)와 같이 구조가 명확한 객체에서 높은 성능을 보였다.
- **효율성**: Table 4에 따르면, Mean Shift나 NMS 같은 무거운 후처리 과정이 없기 때문에 3D-BoNet보다 더 빠른 추론 속도를 보였다 (ScanNet 검증 세트 기준 GICN 2,688초 $\ll$ 3D-BoNet 2,871초).

### Ablation Study (S3DIS Area-5)

- **Instance Size Prediction**: 이를 제거했을 때 mAP가 3.7% 감소하여, 적응적 크기 예측이 성능 향상에 중요함을 입증했다.
- **Focal Loss**: 일반 Cross-Entropy 사용 시 mAP가 14%나 급감하여, 희소한 센터 예측 문제에서 Focal Loss의 필수성이 확인되었다.
- **Center Prediction/Selection**: 무작위 선택이나 단순 상위 값 선택 시 성능이 크게 떨어져, 제안한 히트맵 기반 선택 메커니즘의 유효성이 증명되었다.

## 🧠 Insights & Discussion

**강점 및 기여**:
본 연구는 3D 인스턴스 세그멘테이션을 '센터 예측 $\rightarrow$ 크기 인식 특징 추출 $\rightarrow$ 박스 및 마스크 생성'이라는 직관적인 파이프라인으로 재구성하였다. 특히 가우시안 히트맵을 통해 중간 결과를 시각적으로 확인하고 분석할 수 있게 한 점과, 클래스별 반경 정보를 이용해 NMS 없이 중복을 제거한 점이 매우 효율적이다.

**한계 및 분석**:
실험 결과, 커튼(curtain)이나 사진(picture)과 같이 수직 평면 형태를 띠는 객체들에 대해서는 성능이 상대적으로 낮게 나타났다. 이는 이러한 객체들이 포인트 클라우드 상에서 '센터'라는 개념을 정의하고 찾기에 구조적으로 더 어렵기 때문인 것으로 분석된다.

**비판적 해석**:
GICN은 앵커 프리 방식의 이점을 극대화하여 속도와 정확도를 모두 잡았으나, 센터 예측에 지나치게 의존하는 구조이다. 만약 포인트 클라우드의 노이즈가 심하거나 객체의 중심부가 비어 있는(hollow) 경우가 많다면 센터 예측 단계에서의 오류가 전체 파이프라인으로 전이될 위험이 있다. 비록 저자들이 크기 예측 메커니즘으로 이를 보완하려 했으나, 더 복잡한 형태의 객체에 대한 강건함은 향후 과제로 보인다.

## 📌 TL;DR

GICN은 3D 포인트 클라우드에서 인스턴스 센터의 분포를 가우시안 히트맵으로 학습하여 객체를 분리하는 단일 단계(Single-stage) 앵커 프리 네트워크이다. 센터 기반의 적응적 크기 예측 및 특징 추출 방식을 통해 S3DIS와 ScanNet 데이터셋에서 SOTA 성능을 달성하였으며, 무거운 후처리 과정을 제거하여 추론 효율성을 크게 높였다. 이 연구는 3D 장면 분석에서 앵커 없이도 효율적인 객체 국지화가 가능함을 보여주었으며, 향후 3D 실시간 인스턴스 세그멘테이션 연구에 중요한 기반이 될 것으로 보인다.
