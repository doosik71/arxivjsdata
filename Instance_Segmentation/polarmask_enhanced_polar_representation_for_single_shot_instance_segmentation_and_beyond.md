# PolarMask++: Enhanced Polar Representation for Single-Shot Instance Segmentation and Beyond

Enze Xie, Wenhai Wang, Mingyu Ding, Ruimao Zhang, Ping Luo

## 🧩 Problem to Solve

기존 인스턴스 분할 파이프라인의 복잡성과 계산 오버헤드가 주요 문제입니다. 특히 Mask R-CNN과 같은 2단계 방식은 먼저 바운딩 박스를 감지한 다음 그 안에서 분할을 수행하여 비효율적입니다. 단일 단계 방식조차 바운딩 박스 예측에 의존하는 경우가 많아, 혼잡하거나 회전된 객체(예: 텍스트 감지)와 같은 까다로운 시나리오에서 효율성과 성능이 제한됩니다. 따라서 인스턴스 분할과 객체 감지를 통합하고 다양한 작업에서 우수한 성능을 보이는 더 간단하고 앵커 박스 없는 단일 샷 프레임워크가 필요합니다.

## ✨ Key Contributions

* **새로운 단일 샷 인스턴스 분할 프레임워크 제안 (PolarMask):** 객체의 컨투어(윤곽)를 극좌표계에서 직접 예측하는 방식으로 인스턴스 분할 및 회전된 객체 감지를 효율적으로 수행합니다.
* **극좌표계 최적화를 위한 모듈 개발:**
  * **Soft Polar Centerness:** 고품질의 중심 예시를 샘플링하고 객체 중심의 위치 정확도를 향상시킵니다. 기존 Centerness의 한계를 극복합니다.
  * **Polar IoU Loss:** 밀집된 컨투어 회귀 문제의 최적화를 용이하게 하며, Smooth-$L_1$ 손실보다 더 나은 정확도를 달성합니다. 마스크 IoU를 극좌표 공간에서 직접 최적화합니다.
* **Refined Feature Pyramid (RFP) 도입 (PolarMask++):** 다양한 스케일에서 특징 표현 능력을 강화하여 작은 객체 감지 능력을 향상시키고 전반적인 성능을 높입니다.
* **다양한 벤치마크에서 SOTA 성능 달성:** COCO 인스턴스 분할, ICDAR2015 회전 텍스트 감지, DSB2018 세포 분할 등 여러 도전적인 벤치마크에서 낮은 계산 오버헤드로 경쟁력 있거나 최첨단 성능을 달성하여 극좌표 표현의 강력한 잠재력을 입증합니다.

## 📎 Related Works

* **일반 객체 감지 (General Object Detection):**
  * **2단계 검출기 (Two-stage detectors):** R-CNN [17] 계열 (SPP-Net [21], Fast R-CNN [16], Faster R-CNN [51], R-FCN [8], HyperNet [31], FPN [36]). 제안 생성 후 정제하는 방식.
  * **1단계 검출기 (One-stage detectors):** YOLO [49, 50], SSD [40], DSSD [15], RetinaNet [37]. 제안 생성 단계 없이 직접 예측하여 효율성 증대.
  * **앵커 프리 1단계 검출기 (Anchor-free one-stage detectors):** Cornernet [32], FCOS [56], Reppoints [66], CenterNet [13]. 앵커 박스 없이 객체 감지.
* **회전 객체 감지 (Rotated Object Detection):**
  * 씬 텍스트 감지 등 특화된 영역에서 발전 (TextBoxes [34], EAST [74], RRD [35], RRPN [46], SPCNet [62], PSENet [58], PAN [59]). PolarMask++는 마스크의 특별한 경우로 텍스트를 처리하여 통합된 접근법을 제시합니다.
* **비전 애플리케이션에서의 극좌표 표현 (Polar Representation in Vision Applications):**
  * 딥러닝 이전 시대에는 컨투어 추출 및 추적에 사용 (Denzler et al. [12]의 Active Rays).
  * 딥러닝 시대에는 세포 감지 (StarDist [53]의 star-convex polygon), 건물 분할 (DARNet [6]), 실시간 인스턴스 분할 (ESE-Seg [63]) 등에서 적용.
  * 최근에는 씬 텍스트 감지 및 원격 탐사 객체 감지에도 활용 (Bi and Hu [1], Wang et al. [57], Zhao et al. [71], Zhou et al. [72]).

## 🛠️ Methodology

PolarMask++는 앵커 박스 없는 단일 샷 인스턴스 분할 프레임워크로, 인스턴스 마스크 분할 문제를 극좌표계에서 객체 중심 분류 및 레이(ray) 길이 회귀 문제로 재정의합니다.

1. **극좌표 표현 (Polar Representation):**
    * 객체 마스크의 **질량 중심 (mass center)** $({x}_{c}, {y}_{c})$을 객체의 중심으로 샘플링합니다. 바운딩 박스 중심보다 객체 내부에 있을 확률이 높기 때문에 질량 중심을 사용합니다.
    * 중심에서 객체 컨투어까지 $n$개의 레이(ray)를 균일한 각도 간격 $\Delta\theta$으로 방출하며, 각 레이의 길이 $d_i$를 예측합니다 (예: $n=36$, $\Delta\theta=10^\circ$).
    * 테스트 시에는 예측된 중심 $({x}_{c}, {y}_{c})$과 레이 길이 $\{d_1, \dots, d_n\}$을 사용하여 컨투어 포인트 $({x}_{i}, {y}_{i})$를 계산하고 (수식 1, 2 참조), 이 포인트들을 연결하여 마스크를 생성합니다.
        $$x_i = \cos\theta_i \times d_i + x_c$$
        $$y_i = \sin\theta_i \times d_i + y_c$$
2. **Soft Polar Centerness 예측:**
    * 객체 중심의 품질을 측정하여 낮은 품질의 마스크를 억제하는 데 사용됩니다.
    * 기존 Polar Centerness는 $\sqrt{\frac{\min(\{d_1, \dots, d_n\})}{\max(\{d_1, \dots, d_n\})}}$로 정의됩니다.
    * **Soft Polar Centerness**는 $n$개의 레이를 네 개의 하위 집합 $D_1, D_2, D_3, D_4$ (각각 $0^\circ \sim 90^\circ$, $90^\circ \sim 180^\circ$, $180^\circ \sim 270^\circ$, $270^\circ \sim 360^\circ$)로 나누어, 각 하위 집합에서 대표 값 $F(D_i)$를 사용하여 다음과 같이 정의됩니다:
        $$\text{Soft Polar Centerness} = \sqrt{\frac{F(D_1)}{F(D_3)} \times \frac{F(D_2)}{F(D_4)}}$$
    * $F$ 함수로는 평균, 최댓값, 첫 번째 값 등을 사용할 수 있습니다. 이를 통해 복잡한 모양의 객체에 대한 centerness 값을 개선합니다.
3. **Polar IoU 손실 (Polar IoU Loss)을 이용한 레이 회귀:**
    * $i$-번째 레이의 실제 길이 $d^*_i$와 예측 길이 $d_i$를 사용하여 Polar IoU를 정의합니다:
        $$\text{Polar IoU} = \frac{\sum_{i=1}^{n} \min(d_i, d^*_i)}{\sum_{i=1}^{n} \max(d_i, d^*_i)}$$
    * Polar IoU 손실은 이 Polar IoU의 이진 교차 엔트로피(BCE) 손실로 정의됩니다:
        $$\text{Polar IoU Loss} = \log \frac{\sum_{i=1}^{n} \max(d_i, d^*_i)}{\sum_{i=1}^{n} \min(d_i, d^*_i)}$$
    * 이 손실 함수는 미분 가능하며 병렬 계산이 용이하고, Smooth-$L_1$ 손실과 달리 모든 레이를 전체적으로 최적화하여 더 정확한 지역화를 가능하게 합니다.
4. **네트워크 아키텍처 (PolarMask++):**
    * **백본 (Backbone):** ResNet-101 또는 ResNeXt-101과 같은 표준 백본 사용.
    * **Refined Feature Pyramid (RFP):** FPN [36]을 개선한 모듈로, P3-P7 스케일의 특징 맵을 1/8 해상도로 리스케일링하여 융합한 다음, Non-local [60] 연산을 적용하여 장거리-단거리 픽셀 관계를 모델링하여 특징 표현을 정제합니다. 이후 원래 스케일로 재분배하고 숏컷 연결을 통해 최종 특징 표현을 얻습니다. 이는 작은 객체에 대한 성능을 크게 향상시킵니다.
    * **헤드 (Heads):** 분류 (classification), Polar Centerness, 마스크 회귀 (mask regression)의 세 가지 병렬 브랜치로 구성됩니다.
5. **레이블 생성 (Label Generation):**
    * OpenCV의 `cv2.findContours`를 사용하여 인스턴스의 컨투어를 얻습니다.
    * 각 컨투어 포인트에서 객체 중심까지의 거리와 각도를 계산하여 36개 레이의 길이를 레이블로 만듭니다.
    * 해당 각도의 레이가 없을 경우 가장 가까운 각도의 레이 길이를 사용하거나, 객체 외부에 중심이 있는 경우 작은 상수 값을 할당합니다.
6. **모델 학습 (Model Training):**
    * 다중 작업 손실 함수 $L=L_{cls}+\alpha_1 L_{reg}+\alpha_2 L_{ct}$를 사용하여 공동 최적화합니다.
    * $L_{cls}$는 Focal Loss, $L_{reg}$는 Polar IoU Loss, $L_{ct}$는 Soft Polar Centerness에 대한 이진 교차 엔트로피 손실입니다. $\alpha_1=\alpha_2=1$로 설정합니다.
7. **마스크 조립 및 NMS (Mask Assembling and NMS):**
    * 추론 시, 네트워크가 예측한 중심 점수와 레이 길이로부터 컨투어 포인트를 계산하고 연결하여 마스크를 생성합니다.
    * NMS (Non-Maximum Suppression)는 마스크의 최소 바운딩 박스를 기반으로 적용하여 중복된 마스크를 제거합니다.

## 📊 Results

* **COCO 인스턴스 분할 (General Instance Segmentation):**
  * PolarMask++는 ResNeXt-101-DCN 백본 사용 시 38.7% 마스크 mAP를 달성하여 Mask R-CNN보다 1.6% 포인트 높은 성능을 보입니다.
  * 특히 $AP_{50}$에서 64.1%를 달성하여 Mask R-CNN보다 4.1% 높았으나, $AP_{75}$에서는 0.6% 개선에 그쳐 복잡한 모양의 객체에 대한 정교한 분할에 한계가 있음을 시사합니다.
  * TensorMask(3 FPS)보다 4배 이상 빠른 14 FPS (ResNet-101 백본, 짧은 길이 800)의 추론 속도를 보여 효율성도 뛰어납니다.
* **ICDAR2015 회전 텍스트 감지 (Rotated Text Detection):**
  * ResNet-50 백본을 사용하여 외부 데이터 없이 83.4% F-measure를 달성, 이전 SOTA보다 1% 높습니다.
  * 외부 데이터셋으로 사전 학습 시 85.4% F-measure를 달성하여 새로운 SOTA를 수립합니다.
  * 10 FPS의 빠른 추론 속도를 보이며, PSENet, TextSnake 등 복잡한 후처리 과정이 필요한 다른 방법들보다 빠릅니다.
* **DSB2018 세포 분할 (Cell Segmentation):**
  * ResNet-50 백본으로 74.2% mAP를 달성하여 이전 SOTA인 Nuclei R-CNN보다 4.6% 높은 성능을 보입니다.
  * 세포와 같이 규칙적인 모양의 객체 분할에 특히 강점을 보입니다.
* **어블레이션 스터디 (Ablation Study) 주요 결과:**
  * **레이 수 (Number of Rays):** 36개의 레이가 성능과 계산량의 균형을 잘 이룹니다.
  * **Polar IoU Loss vs. Smooth-$L_1$ Loss:** Polar IoU Loss는 Smooth-$L_1$ Loss (최고 설정 25.1% AP)보다 2.6% 높은 27.7% AP를 달성하여 더 효과적임을 입증합니다.
  * **Centerness 전략 (Centerness Strategy):** Soft Polar Centerness는 기존 Polar Centerness보다 0.6% AP 이상 향상됩니다.
  * **특징 피라미드 전략 (Strategy of Feature Pyramid):** 제안된 Refined Feature Pyramid (RFP)는 FPN 및 PANet보다 0.4% AP를 향상시키며, 특히 작은 객체에 강점을 보입니다.
  * **바운딩 박스 브랜치 (Box Branch):** 마스크 예측 성능에 거의 영향을 미치지 않으므로, PolarMask++는 바운딩 박스 감지 없이 마스크를 직접 생성합니다.
* **마스크 후처리 FCN (Mask Refinement with Post-processing FCN):** 불규칙한 모양의 객체(사람, 동물 등)에 대해 픽셀 단위 FCN을 후처리로 추가하면 30.2 AP에서 34.1 AP로 크게 향상됩니다. 각 객체당 1.5ms의 적은 시간이 소요됩니다.

## 🧠 Insights & Discussion

* **극좌표 표현의 유연성 및 일반성:** PolarMask++는 인스턴스, 회전된 객체, 세포 등 다양한 형태의 객체를 모델링할 수 있는 유연하고 일반적인 표현 방식을 제공합니다. 이는 특히 규칙적인 모양의 객체(예: 세포, 텍스트)에서 큰 장점을 가집니다.
* **단일 샷 방식의 효율성:** 바운딩 박스 예측에 의존하지 않는 단일 샷 파이프라인 덕분에 기존 2단계 방식보다 계산 비용이 낮고 학습 및 추론이 효율적입니다. 특히 혼잡한 장면에서 기존 NMS가 겹치는 바운딩 박스를 억제하여 객체 손실을 일으키는 문제에 대한 해결책을 제시합니다.
* **Polar Centerness 및 Polar IoU Loss의 중요성:** Soft Polar Centerness는 고품질 중심 샘플을 효과적으로 가중치 부여하여 특히 높은 IoU 임계값에서의 성능($AP_{75}$)을 향상시킵니다. Polar IoU Loss는 전체 레이 회귀를 동시에 최적화함으로써 Smooth-$L_1$ Loss보다 더 정확하고 매끄러운 컨투어 예측을 가능하게 합니다.
* **Refined Feature Pyramid (RFP)의 효과:** RFP는 다양한 스케일의 특징 융합 능력을 강화하고 Non-local 연산을 통해 픽셀 간의 문맥적 관계를 모델링하여, 특히 작은 객체의 감지 성능을 효과적으로 개선합니다.
* **한계점 및 향후 연구:** COCO 데이터셋에서 사람, 의자 등 복잡하고 불규칙한 모양의 객체에 대한 $AP_{75}$ 성능이 상대적으로 낮게 나타났습니다. 이는 극좌표 표현이 이러한 객체의 정교한 컨투어를 모델링하는 데 어려움이 있음을 시사합니다. 향후에는 이러한 복잡한 모양의 객체에 대한 극좌표 표현의 능력을 개선하는 데 더 많은 노력을 기울일 필요가 있습니다.

## 📌 TL;DR

PolarMask++는 기존 인스턴스 분할의 복잡성과 바운딩 박스 의존성을 해결하기 위해, 객체 컨투어를 극좌표계에서 직접 예측하는 앵커 박스 없는 단일 샷 프레임워크입니다. Soft Polar Centerness와 Polar IoU Loss를 도입하여 중심 예측과 레이 길이 회귀를 최적화하고, Refined Feature Pyramid로 특징 표현 능력을 강화합니다. 이 방법은 COCO, ICDAR2015, DSB2018 등 다양한 벤치마크에서 높은 정확도와 효율성을 달성하며, 특히 회전 객체 감지와 규칙적인 모양의 객체 분할에서 SOTA 성능을 기록했습니다. 복잡하고 불규칙한 모양의 객체에 대한 성능 개선은 향후 연구 과제입니다.
