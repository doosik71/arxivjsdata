# Classifying, Segmenting, and Tracking Object Instances in Video with Mask Propagation

Gedas Bertasius, Lorenzo Torresani (2021)

## 🧩 Problem to Solve

본 논문은 비디오 인스턴스 분할(Video Instance Segmentation, VIS) 문제를 해결하고자 한다. 이 작업은 비디오의 모든 프레임에서 미리 정의된 클래스에 속하는 모든 객체 인스턴스를 분할(Segmentation)하고, 각 인스턴스를 분류(Classification)하며, 전체 시퀀스에 걸쳐 동일한 인스턴스를 연결하여 추적(Tracking)하는 것을 목표로 한다.

비디오 인스턴스 분할이 어려운 이유는 정밀한 객체 지역화를 위해 매우 높은 공간 해상도가 필요하지만, 시간적 일관성을 유지하기 위해 여러 프레임을 동시에 처리하려면 GPU 메모리 제약이 크기 때문이다. 기존 방식들은 해상도를 낮추어 성능 저하를 겪거나, 프레임별 분할 후 후처리 단계에서 추적을 수행하여 최적의 결과를 얻지 못하는 한계가 있었다. 따라서 본 연구의 목표는 높은 탐지 정확도를 유지하면서 비디오 내 객체를 효과적으로 추적할 수 있는 통합 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **MaskProp**이라 불리는 단순하고 효율적인 마스크 전파 프레임워크를 도입하는 것이다. MaskProp은 기존의 Mask R-CNN을 비디오 영역으로 확장하여, 특정 프레임(중심 프레임)에서 생성된 인스턴스 마스크를 주변 프레임(클립 내 다른 프레임)으로 전파하는 **Mask Propagation Branch**를 추가한다.

이 설계의 직관은 중심 프레임의 정밀한 인스턴스 정보를 시간적 이웃 프레임으로 전파함으로써, 모션 블러(Motion Blur)나 객체 가려짐(Occlusion)과 같은 까다로운 상황에서도 강건한 클립 수준의 인스턴스 트랙을 생성할 수 있다는 점이다. 이렇게 생성된 밀집한 클립 단위 트랙들을 최종적으로 결합하여 비디오 전체 수준의 인스턴스 분할 및 분류 결과를 도출한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개한다.

- **이미지 인스턴스 분할**: Mask R-CNN과 같은 모델이 개별 프레임의 분할에는 뛰어나지만, 프레임 간 인스턴스 대응 관계를 결정하는 기능이 부족하다.
- **비디오 객체 탐지**: 시공간 특성 정렬(Spatiotemporal feature alignment)을 통해 정확도를 높이지만, 개별 인스턴스를 추적하는 기능은 설계되지 않았다.
- **비디오 객체 분할(VOS)**: 클래스 구분 없이 전경 객체를 분할하며, 주로 첫 프레임의 Ground Truth 마스크를 활용한다. 반면 VIS는 클래스 분류와 자동 탐지가 동시에 이루어져야 한다.
- **기존 VIS 접근 방식**:
  - **MaskTrack R-CNN**: 통합 모델이지만 성능이 상대적으로 낮다.
  - **EnsembleVIS (ICCV 2019 우승작)**: 탐지, 분류, 분할, 추적을 각각 독립적인 모델(또는 앙상블)로 해결한 뒤 결합한다. 성능은 좋으나 구조가 매우 복잡하고 튜닝 비용이 높으며, 방대한 양의 학습 데이터가 필요하다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조

MaskProp은 입력 비디오 $V$를 길이 $(2T+1)$인 여러 개의 클립으로 나누어 처리한다. 각 클립의 중심 프레임 $t$를 기준으로 인스턴스 트랙을 생성하며, 이후 이 클립 단위 결과들을 통합하여 비디오 전체의 마스크 시퀀스를 생성한다.

### 상세 동작 절차 (Mask Propagation Branch)

마스크 전파 과정은 다음의 세 단계로 이루어진다.

1. **인스턴스 특성 계산 (Instance-Specific Feature Computation)**:
   중심 프레임 $t$에서 Mask R-CNN을 통해 예측된 프레임 수준 마스크 $M_t^i$와 백본 네트워크의 특성 텐서 $f_t$를 요소별 곱(Element-wise product) 연산한다. 이를 통해 해당 객체 인스턴스 영역 외의 특성 값은 0으로 만들어 인스턴스 전용 특성 $f_t^i$를 추출한다.

2. **시간적 특성 전파 (Temporal Propagation of Instance Features)**:
   프레임 $t$의 특성 $f_t^i$를 프레임 $t+\delta$로 전파하여 $g_{t,t+\delta}^i$를 생성한다. 이때 **Deformable Convolution (DCN)**을 사용하여 특성을 워핑(Warping)한다.
   - 두 프레임 특성($f_t, f_{t+\delta}$)의 차이를 잔차 블록(Residual Block)에 입력하여 모션 오프셋 $o_{t,t+\delta}$를 예측한다.
   - 이 오프셋을 DCN의 샘플링 위치로 사용하여 $f_t^i$를 전파함으로써, 명시적인 Ground Truth 정렬 없이도 특징을 정렬한다.

3. **전파된 인스턴스 분할 (Segmenting Propagated Instances)**:
   전파된 특성 $g_{t,t+\delta}^i$와 해당 프레임의 특성 $f_{t+\delta}$를 더하여 $\phi_{t,t+\delta}^i = g_{t,t+\delta}^i + f_{t+\delta}$를 구성한다. 이후 $1 \times 1$ 컨볼루션과 Softmax를 거쳐 마스크를 예측하며, 인스턴스 불가지론적 어텐션 맵(Instance-agnostic attention map) $A_{t+\delta}$를 곱해 객체가 없는 배경 영역을 제거한다.

### 손실 함수 및 학습 절차

모델은 다중 작업 손실 함수(Multi-task loss)를 사용하여 학습한다.
$$L_t = L_{cls}^t + L_{box}^t + L_{mask}^t + L_{prop}^t$$
여기서 $L_{prop}^t$는 전파된 마스크와 Ground Truth 마스크 간의 **soft IoU loss**를 사용한다.
$$sIoU(A,B) = \frac{\sum_p A(p)B(p)}{\sum_p A(p) + B(p) - A(p)B(p)}$$

### 비디오 수준의 인스턴스 ID 할당

클립 단위 트랙들을 연결하기 위해 soft IoU 기반의 매칭 스코어 $m_{i,j}^{t,t'}$를 계산한다. 새로운 클립의 트랙이 기존에 할당된 비디오 수준 ID $y$와 높은 유사도를 보이면 동일 ID를 부여하고, 그렇지 않으면 새로운 ID를 생성하여 할당하는 방식으로 전체 비디오 시퀀스를 구성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: YouTube-VIS (검증 세트에서 평가).
- **평가 지표**: mAP (mean Average Precision), AP@75, AR@1, AR@10.
- **비교 대상**: MaskTrack R-CNN, EnsembleVIS 및 FlowNet2 기반 전파 방식.

### 정량적 결과

MaskProp은 모든 지표에서 기존 모델들을 압도하는 성능을 보였다.

| Method | mAP | AP@75 | AR@1 | AR@10 | Pre-training Data |
| :--- | :---: | :---: | :---: | :---: | :--- |
| MaskTrack R-CNN | 30.3 | 32.6 | 31.0 | 35.5 | ImageNet, COCO |
| EnsembleVIS | 44.8 | 48.9 | 42.7 | 51.7 | $\dots$ + Instagram, OpenImages |
| **MaskProp** | **46.6** | **51.2** | **44.0** | **52.6** | ImageNet, COCO |
| **MaskProp (+OpenImages)** | **50.0** | **55.9** | **44.6** | **54.5** | $\dots$ + OpenImages |

특히 EnsembleVIS보다 훨씬 단순한 구조이며 사용된 학습 데이터의 양이 수백 배 적음에도 불구하고 더 높은 성능(46.6% vs 44.8% mAP)을 달성하였다.

### 어블레이션 연구 (Ablation Study)

- **전파 브랜치의 효율성**: FlowNet2 기반의 광학 흐름(Optical Flow) 전파 방식(31.4 mAP)이나 MaskTrack R-CNN의 추적 브랜치(36.9 mAP)보다 MaskProp의 DCN 기반 전파 방식이 월등히 우수함을 확인하였다.
- **클립 길이**: $T=6$ (전체 길이 $2T+1=13$ 프레임)일 때 최적의 성능을 보였다.
- **고해상도 마스크 정밀화**: 해당 단계를 제거했을 때 mAP가 1.9% 하락하였다.
- **백본의 영향**: 더 강력한 백본(예: STSN-ResNeXt-101)을 통해 얻은 프레임 수준 마스크가 좋을수록 비디오 전체 성능이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점

MaskProp의 가장 큰 강점은 **단순함과 효율성**이다. 복잡한 앙상블 구조 없이 단일 통합 모델만으로 SOTA 성능을 달성했으며, 특히 DCN을 이용한 특징 전파가 모션 블러나 객체 변형, 가려짐 상황에서도 매우 강건하게 작동함을 시각화 결과(Figure 5, 7)를 통해 입증하였다.

### 한계 및 논의

본 모델은 프레임 수준의 인스턴스 마스크 품질에 크게 의존한다. 어블레이션 실험에서 보여주었듯, 초기 Mask R-CNN의 성능이 높을수록 최종 VIS 결과가 좋아지므로, 향후 이미지 인스턴스 분할 기술의 발전이 MaskProp의 성능을 더욱 끌어올릴 수 있을 것으로 보인다. 또한, 현재는 마스크 주석이 있는 데이터로 학습하지만, 향후 바운딩 박스만으로 학습 가능한 시나리오로 확장할 필요가 있다.

## 📌 TL;DR

본 논문은 Mask R-CNN에 **Deformable Convolution 기반의 마스크 전파 브랜치**를 추가하여, 단순하면서도 강력한 비디오 인스턴스 분할 모델인 **MaskProp**을 제안한다. 이 모델은 복잡한 앙상블 방식보다 적은 데이터와 단순한 구조로 YouTube-VIS 데이터셋에서 SOTA 성능을 달성하였다. 이는 비디오 내 객체 추적 문제를 특징 공간에서의 정밀한 전파 문제로 치환하여 해결함으로써, 실제 적용 시 높은 효율성과 강건함을 제공할 수 있음을 시사한다.
