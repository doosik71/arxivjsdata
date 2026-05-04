# Weakly Supervised Instance Segmentation for Videos with Temporal Mask Consistency

Qing Liu, Vignesh Ramanathan, Dhruv Mahajan, Alan Yuille, Zhenheng Yang (2021)

## 🧩 Problem to Solve

본 논문은 이미지 수준의 클래스 레이블(image-level class labels)만을 사용하여 인스턴스 분할(instance segmentation)을 수행하는 Weakly Supervised Instance Segmentation (WSIS)의 한계를 해결하고자 한다. 기존의 WSIS 방식들은 정밀한 픽셀 단위의 정답(ground truth) 없이 학습하기 때문에 주로 두 가지 핵심적인 문제에 직면한다. 첫째는 **부분적 인스턴스 분할(partial segmentation of objects)**로, 모델이 클래스 레이블을 예측하는 데 가장 결정적인(discriminative) 영역만을 식별하여 객체 전체를 덮지 못하는 현상이다. 둘째는 **객체 예측 누락(missing object predictions)**으로, 동일 클래스의 여러 객체가 존재하거나 폐색(occlusion) 및 포즈 변화가 있을 때 일부 인스턴스를 완전히 놓치는 문제이다.

이 연구의 목표는 정적인 이미지 대신 비디오 데이터를 활용하여, 객체의 움직임(motion)과 프레임 간의 시간적 일관성(temporal consistency)이라는 추가적인 신호를 도입함으로써 앞서 언급한 두 가지 문제를 완화하고 WSIS의 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비디오 내의 광학 흐름(optical flow)과 시간적 연속성을 활용하여 가짜 레이블(pseudo-labels)의 품질을 높이고, 학습 과정에서 누락된 인스턴스를 복구하는 것이다. 이를 위해 다음과 같은 두 가지 주요 기여를 제시한다.

1. **flowIRN**: 기존의 Inter-pixel Relation Network (IRN)을 확장하여 광학 흐름 정보를 학습 과정에 통합하였다. 이를 통해 객체의 전체적인 영역을 더 잘 포착하고(Flow-Amplified CAMs), 동일 클래스 내에서도 서로 다른 인스턴스를 효과적으로 구분(Flow-boundary loss)할 수 있게 한다.
2. **MaskConsist**: 시간적 일관성을 강제하는 새로운 모듈로, 특정 프레임에서 누락된 객체 인스턴스를 인접 프레임의 안정적인 예측 결과로부터 전이(transfer)받아 보완함으로써 예측 누락 문제를 해결한다.

## 📎 Related Works

기존의 약지도 학습 기반 세그멘테이션 연구들은 주로 Class Activation Maps (CAMs)에 의존하여 가짜 레이블을 생성해 왔다. 하지만 CAMs는 객체의 가장 특징적인 부분만을 활성화하는 경향이 있어 부분 분할 문제가 빈번하게 발생한다. 이를 해결하기 위해 객체 제안(object proposals)을 결합하거나(WISE 등), 픽셀 간 관계를 모델링하는 방식(IRN 등)이 제안되었다.

비디오 데이터의 경우, 일부 연구가 시간적 일관성이나 광학 흐름을 이용해 시맨틱 세그멘테이션(semantic segmentation) 성능을 높이려 시도하였으나(F2F 등), 본 논문은 이를 **인스턴스 분할(instance segmentation)** 영역으로 확장하여 적용한 첫 번째 연구라는 점-에서 차별점을 갖는다. 특히, 단순히 가짜 레이블을 생성하는 단계뿐만 아니라, Mask R-CNN과 같은 지도 학습 모델을 훈련시키는 단계에서도 시간적 일관성을 활용하는 MaskConsist 구조를 제안하였다.

## 🛠️ Methodology

본 프레임워크는 2단계 학습 절차를 따른다. 먼저 `flowIRN`을 통해 초기 가짜 레이블을 생성하고, 이후 이 레이블들을 사용하여 `MaskConsist` 모듈이 포함된 모델을 학습시킨다.

### 1. flowIRN Module

`flowIRN`은 기존 IRN의 구조를 유지하면서 광학 흐름(optical flow) 정보를 두 가지 방식으로 통합한다.

**Flow-Amplified CAMs**: CAMs가 객체의 일부만을 포착하는 문제를 해결하기 위해, 광학 흐름의 크기가 큰 영역(움직임이 활발한 전경 객체 영역)의 CAMs 값을 증폭시킨다.
$$f\text{-CAM}^c(x) = \text{CAM}^c(x) \times A^{I(||F(x)||^2 > T)}$$
여기서 $A$는 증폭 계수, $T$는 흐름 크기 임계값이며, 이를 통해 객체 전체 영역에 대한 활성도를 높인다.

**Flow-boundary loss**: 동일 클래스의 중첩된 인스턴스를 구분하기 위해 광학 흐름의 공간적 기울기(spatial gradient)를 활용한다. 동일한 강체 객체는 유사한 흐름 기울기를 갖는다는 점에 착안하여 다음과 같은 손실 함수를 정의한다.
$$L_F^B = \sum_{j \in N_i} ||F'(i) - F'(j)|| \alpha_{i,j} + \lambda |1 - \alpha_{i,j}|$$
여기서 $F'(i)$는 픽셀 $i$에서의 광학 흐름 기울기이며, $\alpha_{i,j}$는 픽셀 간의 친밀도(affinity)이다. 이 손실 함수는 흐름 기울기가 유사한 픽셀들은 높은 친밀도를 갖게 하고, 다른 픽셀들은 낮은 친밀도를 갖도록 강제하여 인스턴스 경계를 더 정확히 예측하게 한다.

### 2. MaskConsist Module

`MaskConsist`는 `flowIRN`이 생성한 가짜 레이블이 여전히 일부 인스턴스를 누락할 수 있다는 점을 보완한다. 인접 프레임 간에 안정적으로 예측된 마스크를 전이하여 누락된 부분을 채우는 것이 핵심이며, 총 3단계로 구성된다.

- **Intra-frame matching**: 현재 Mask R-CNN의 예측 결과 중 신뢰도가 높거나 `flowIRN`의 가짜 레이블과 겹치는 마스크들의 합집합을 구해 확장된 예측 집합 $P_t^{exp}$를 생성한다.
- **Inter-frame matching**: 두 인접 프레임 $t$와 $t+\delta$ 사이의 예측 집합들 간에 이분 그래프(bipartite graph)를 생성한다. 광학 흐름을 이용해 마스크를 워핑(warping)한 후 IoU를 계산하여 헝가리안 알고리즘(Hungarian algorithm)으로 일대일 매칭을 수행함으로써 시간적으로 안정적인 예측 쌍 $M_{t, t+\delta}$를 찾는다.
- **Temporally consistent labels**: 매칭된 쌍 중에서 `flowIRN`의 가짜 레이블과 일정 수준 이상($IoU > 0.5$) 겹치는 고품질 예측만을 선택한다. 이를 광학 흐름으로 워핑하여 대상 프레임의 예측 결과와 병합(Merge)함으로써 새로운 가짜 레이블 $P_{maskCon}$을 생성한다.

최종적으로 $P_{maskCon} \cup P_{fIRN}$을 합친 후, IoM(Intersection over Minimum) 기반의 NMS를 적용하여 중복된 레이블을 제거하고 이를 통해 Mask R-CNN을 학습시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: Youtube-VIS (YTVIS) 및 Cityscapes.
- **측정 지표**: 프레임 수준에서는 $AP_{50}$, 비디오 수준(VIS)에서는 $mAP$, $AP_{50}$, $AP_{75}$, $AR$ 등을 사용하였다.
- **비교 대상**: WISE, IRN, F2F+MCG 및 광학 흐름을 단순 적용한 IRN+F2F 등이 비교군으로 설정되었다.

### 주요 결과

1. **프레임 수준 성능**: YTVIS에서 $AP_{50}$ 기준, 기존 IRN 대비 약 5%, IRN+F2F 대비 4% 이상의 성능 향상을 보였다. Cityscapes에서도 IRN 및 WISE 대비 3.7% 이상 높은 성능을 기록하였다.
2. **비디오 수준 성능 (VIS)**: YTVIS 검증 서버 평가 결과, $AP_{50}$ 지표에서 IRN이나 WISE 대비 8% 이상의 큰 폭의 향상을 보였다. 이는 시간적 일관성 모델링이 비디오 인스턴스 추적 성능에 결정적인 영향을 미침을 시사한다.
3. **Ablation Study**:
    - `flowIRN`의 구성 요소 중 f-CAMs와 f-Bound loss 모두 성능 향상에 기여하였으며, 둘을 모두 사용했을 때 최적의 가짜 레이블 품질을 보였다.
    - `MaskConsist`에서는 Inter-frame matching이 가장 큰 성능 향상을 이끌어냈으며, IoM-NMS가 거짓 긍정(false positive)을 줄여 성능을 추가로 개선하였다.

## 🧠 Insights & Discussion

본 논문은 이미지 기반의 약지도 학습이 가진 고질적인 문제인 '부분 분할'과 '인스턴스 누락'을 비디오의 시간적 정보로 해결할 수 있음을 입증하였다. 특히 $AP_{50}$보다 더 엄격한 지표인 $AP_{75}$에서 더 큰 상대적 향상을 보였다는 점은, 제안 방법이 단순히 객체를 찾는 것을 넘어 더 정밀한 마스크 경계를 생성하고 있음을 의미한다. 또한, 프레임당 평균 예측 인스턴스 수가 증가한 것은 누락되었던 객체들을 성공적으로 복구했음을 뒷받침한다.

다만, 한계점으로는 객체들이 항상 함께 움직이는 경우(예: 스케이트보드와 그 위의 사람)에는 광학 흐름만으로 인스턴스를 구분하기 어렵다는 점이 언급되었다. 이는 제안 방법이 "서로 다른 인스턴스는 서로 다른 움직임을 가진다"는 가설에 기반하고 있기 때문이다.

## 📌 TL;DR

본 연구는 이미지 수준의 클래스 레이블만으로 학습하는 약지도 인스턴스 분할의 한계를 극복하기 위해, 비디오의 **광학 흐름(Optical Flow)**과 **시간적 일관성(Temporal Consistency)**을 도입하였다. `flowIRN`을 통해 객체 영역 확장 및 경계 구분을 정교화하고, `MaskConsist` 모듈을 통해 인접 프레임 간 예측을 전이함으로써 누락된 인스턴스를 보완하였다. 그 결과 Youtube-VIS와 Cityscapes 데이터셋에서 기존 SOTA 약지도 학습 모델들을 상회하는 성능을 달성하였으며, 향후 비디오 기반의 약지도 학습 연구에 중요한 방법론적 틀을 제공할 것으로 기대된다.
