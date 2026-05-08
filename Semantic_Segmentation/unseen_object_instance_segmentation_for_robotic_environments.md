# Unseen Object Instance Segmentation for Robotic Environments

Christopher Xie, Yu Xiang, Arsalan Mousavian, Dieter Fox

## 🧩 Problem to Solve

로봇이 비정형 환경에서 기능하려면 이전에 본 적 없는(unseen) 객체를 인식하는 능력이 필수적입니다. 특히, 로봇이 잡거나 조작하는 등의 작업이 이루어지는 탁상 환경에서 **미확인 객체 인스턴스 분할(Unseen Object Instance Segmentation, UOIS)**은 중요한 문제입니다. 그러나 이 작업을 위한 대규모 실세계 데이터셋은 대부분의 로봇 환경에서 존재하지 않으며, 이로 인해 합성 데이터 사용의 필요성이 대두됩니다. 하지만 합성 데이터와 실세계 데이터 사이에는 도메인 갭이 존재하여 학습된 모델의 실세계 적용에 어려움이 있습니다. 기존 방법들은 주로 RGB 이미지에만 초점을 맞추거나, 깊이(depth) 정보만 사용했을 때 센서 노이즈에 취약한 한계가 있었습니다.

## ✨ Key Contributions

* **UOIS-Net 제안:** 미확인 객체 인스턴스 분할을 위해 합성 RGB와 깊이(depth) 정보를 개별적으로 활용하는 두 단계 네트워크인 UOIS-Net을 제안했습니다.
* **비현실적인 합성 RGB 활용:** 놀랍게도, UOIS-Net의 두 번째 단계인 RRN(Region Refinement Network)은 비현실적인(non-photorealistic) 합성 RGB 이미지로 훈련되었음에도 실세계 데이터에 잘 일반화되어 선명하고 정확한 마스크를 생성할 수 있음을 보였습니다.
* **TOD (Tabletop Object Dataset) 도입:** 대규모 합성 탁상 객체 데이터셋인 TOD를 구축하여 모델 훈련에 활용했습니다.
* **3D DSN 아키텍처 및 새로운 손실 함수:** 2D 추론 방식의 한계를 극복하기 위해 3D 공간에서 추론하는 DSN(Depth Seeding Network) 아키텍처를 도입하고, 혼잡한 환경에서 성능을 크게 향상시키는 새로운 **분리 손실($\mathcal{L}_{\text{sep}}$)**을 제안했습니다.
* **최첨단 성능 달성:** 기존 최첨단(SOTA) 미확인 객체 인스턴스 분할 방법들을 능가하는 성능을 보였습니다.
* **로봇 조작 적용 가능성:** 제안된 방법이 로봇의 미확인 객체 파지(grasping) 작업에 성공적으로 적용될 수 있음을 입증했습니다.

## 📎 Related Works

* **범주 레벨 객체 분할:** 2D(FCN, DeepLab, SegNet 등) 및 3D(PointNet, Sparse Convolutional Networks) 방법론들이 발전했으며, RGB-D 센서의 활용 연구(HHA 인코딩, Depth-aware CNN 등)도 이루어졌습니다.
* **인스턴스 레벨 객체 분할:** 2D에서는 Mask R-CNN과 같은 top-down 방식과 ClusterNet 같은 bottom-up 방식이 있으며, class-agnostic 훈련을 통해 미확인 객체에 적용하려는 시도가 있었습니다. 3D에서는 Center Voting 기반 기법(VoteNet, PointGroup 등)이 활발히 연구되었습니다.
* **Sim-to-Real 지각:** 합성 데이터의 실세계 적용을 위한 도메인 무작위화(domain randomization)와 도메인 적응(domain adaptation) 기술이 개발되었으며, 깊이 정보는 Sim-to-Real 일반화에 비교적 강점을 보이는 것으로 알려져 있습니다.

## 🛠️ Methodology

UOIS-Net은 깊이와 RGB를 분리하여 처리하는 두 단계 프레임워크입니다.

1. **Depth Seeding Network (DSN):**
    * **입력:** 카메라 내장(intrinsics)을 사용하여 깊이 맵에서 역투영(backprojection)된 3채널 XYZ 좌표의 조직화된 포인트 클라우드 $D \in \mathbb{R}^{H \times W \times 3}$를 받습니다.
    * **2D 추론 ($UOIS\text{-}Net\text{-}2D$):** U-Net 아키텍처를 사용하여 의미론적 분할 마스크 $F$ (배경, 탁상, 객체)와 객체 중심의 2D 방향 $V \in \mathbb{R}^{H \times W \times 2}$를 예측합니다. Hough voting 레이어를 통해 초기 인스턴스 마스크를 생성합니다.
        * **손실 함수:** 의미론적 분할을 위한 가중치 교차 엔트로피($\mathcal{L}_{\text{fg}}$)와 방향 예측을 위한 가중치 코사인 유사도 손실($\mathcal{L}_{\text{dir}}$)을 사용합니다.
    * **3D 추론 ($UOIS\text{-}Net\text{-}3D$):** ESP(Efficient Spatial Pyramid) 모듈을 포함한 U-Net 아키텍처로 수용 필드(receptive field)를 높여 3D 객체 오프셋 $V' \in \mathbb{R}^{H \times W \times 3}$를 예측합니다. 예측된 중심 투표(center votes) $D+V'$에 평균 이동(Mean Shift) 클러스터링을 적용하여 초기 인스턴스 마스크를 생성합니다.
        * **손실 함수:** $\mathcal{L}_{\text{fg}}$ 외에, 중심 오프셋에 대한 Huber 손실($\mathcal{L}_{\text{co}}$), 클러스터링 손실($\mathcal{L}_{\text{cl}}$), 그리고 중심 투표 간의 분리를 장려하는 새로운 **분리 손실($\mathcal{L}_{\text{sep}}$)**을 사용합니다. $\mathcal{L}_{\text{sep}}$는 다음 텐서를 최대화하도록 설계되었습니다:
            $$M_{ij} = \frac{\exp(-\tau d(c_j, D_i+V'_i))}{\sum_{j'=1}^{N} \exp(-\tau d(c_{j'}, D_i+V'_i))}$$
            여기서 $c_j$는 $j$번째 정답 객체 중심이고, $d(\cdot, \cdot)$는 유클리드 거리, $\tau$는 하이퍼파라미터입니다.
2. **Initial Mask Processor (IMP):** DSN에서 생성된 초기 마스크의 노이즈를 제거하기 위해 열기(opening), 닫기(closing)와 같은 형태학적 변환(morphological transform)과 가장 큰 연결 구성 요소(connected component)를 선택하는 과정을 거칩니다.
3. **Region Refinement Network (RRN):**
    * **입력:** 초기 인스턴스 마스크 주변으로 잘라내고 패딩된 4채널 이미지(RGB + 마스크). $224 \times 224$로 크기 조정됩니다.
    * **아키텍처:** U-Net을 사용하여 정제된 마스크 확률을 출력합니다.
    * **훈련:** DSN과 별도로 훈련되며, 정답 마스크에 다양한 변형(이동, 회전, 추가/제거, 형태학적 연산, 무작위 타원 추가)을 가하여 생성된 증강된 마스크를 사용합니다.

## 📊 Results

* **정량적 평가:** OCID 및 OSD 데이터셋에서 Overlap 및 Boundary P/R/F(Precision/Recall/F-measure) 지표를 사용하여 평가했습니다.
  * UOIS-Net-2D와 UOIS-Net-3D 모두 기존 베이스라인(GCUT, SCUT, LCCP, V4R) 및 SOTA(Mask R-CNN, PointGroup) 대비 우수한 성능을 보였습니다.
  * 특히, UOIS-Net-3D는 2D 버전에 비해 Overlap F-measure 4.5%, Boundary F-measure 5.5%의 상대적 개선을 달성하여 3D 추론의 이점을 입증했습니다.
  * Mask R-CNN 및 PointGroup과 비교하여 UOIS-Net-3D는 OCID에서 Overlap F-measure 7.9% 및 Boundary F-measure 6.3%, OSD에서 각각 5.7% 및 8.9%의 상대적 향상을 보였습니다.
* **Sim-to-Real 일반화:** TOD(합성)로 훈련된 RRN은 OID(실제)로 훈련된 RRN과 비슷한 성능을 보여, 마스크 정제 작업이 Sim-to-Real 도메인 갭에 비교적 덜 민감함을 시사했습니다.
* **어블레이션 연구:**
  * IMP 모듈은 노이즈가 많은 초기 마스크를 처리하여 RRN의 성능을 크게 향상시키는 데 필수적임을 확인했습니다.
  * 3D DSN의 새로운 $\mathcal{L}_{\text{sep}}$ 손실은 특히 혼잡한 장면에서 성능을 10.1% (Overlap F-measure) 및 14.4% (Boundary F-measure) 향상시키는 데 결정적인 역할을 했습니다.
  * DSN에 ESP 모듈을 사용하여 수용 필드를 높이는 것이 성능 향상에 기여했습니다.
* **로봇 파지 응용:** Franka 로봇을 사용하여 테이블 위의 미확인 객체를 수집하는 작업에서 51개 객체 중 41개(80.3% 성공률)를 성공적으로 파지하여 실세계 로봇 조작에서의 유용성을 입증했습니다.

## 🧠 Insights & Discussion

* **RGB와 깊이 정보의 분리 활용:** 깊이 정보는 Sim-to-Real 일반화에 강점을 가지고 초기 마스크를 생성하는 데 효과적이지만, 경계면이 노이즈에 취약합니다. 반면, RGB 정보는 마스크 경계를 정교하게 다듬는 데 탁월합니다. UOIS-Net은 이 두 가지 모달리티의 장점을 분리하여 활용함으로써 시뮬레이션에서 학습된 모델이 실제 환경에서 뛰어난 성능을 발휘할 수 있음을 보여주었습니다.
* **비현실적인 합성 RGB의 효과적인 활용:** RRN이 비현실적인 합성 RGB 데이터로 훈련되었음에도 실제 이미지에서 잘 작동하는 것은 마스크 정제 작업이 이미지의 전반적인 사실성보다는 경계면의 지역적 특징에 더 크게 의존하기 때문인 것으로 분석됩니다. 이는 포토리얼리스틱한 합성 데이터 생성의 높은 비용 문제를 완화할 수 있는 중요한 발견입니다.
* **3D 추론의 중요성:** 2D 기반 방법이 객체 중심이 가려지거나 객체가 매우 얇을 때 실패할 수 있는 반면, 3D 기반 DSN은 이러한 한계를 극복하여 혼잡한 환경에서 더욱 강력한 성능을 제공합니다.
* **새로운 손실 함수의 역할:** $\mathcal{L}_{\text{sep}}$ 손실은 객체 중심 투표들이 서로 충분히 떨어져 있도록 장려함으로써 후처리 클러스터링의 난이도를 낮추고, 결과적으로 혼잡한 장면에서 탁월한 분할 정확도를 달성하는 데 핵심적인 기여를 합니다.
* **한계점:** 객체가 너무 밀집되어 있거나 평평한 표면을 가진 경우(예: 시리얼 박스들), 깊이만으로는 객체를 분리하기 어려워 과소 분할(under-segmentation)이 발생할 수 있습니다. 또한, 매우 볼록하지 않은(non-convex) 객체(예: 전동 드릴)는 과대 분할(over-segmentation)될 수 있으며, RRN은 복잡한 텍스처나 불충분한 패딩에 취약한 경우가 있습니다.

## 📌 TL;DR

본 논문은 로봇이 본 적 없는 객체를 분할하기 위한 UOIS-Net을 제안합니다. 이 네트워크는 깊이(depth)만을 사용하여 거친 초기 마스크를 생성하는 DSN (2D 또는 3D 추론, 3D의 경우 새로운 분리 손실 도입)과, 이 마스크를 비현실적인 합성 RGB 이미지로 훈련된 RRN이 정제하는 두 단계로 구성됩니다. UOIS-Net은 탁월한 Sim-to-Real 일반화 능력을 보여 실세계 데이터셋에서 최첨단 성능을 능가하며, 로봇의 미확인 객체 파지 작업에 성공적으로 적용될 수 있음을 입증했습니다.
