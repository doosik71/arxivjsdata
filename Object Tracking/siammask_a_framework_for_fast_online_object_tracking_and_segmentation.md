# SiamMask: A Framework for Fast Online Object Tracking and Segmentation

Weiming Hu, Qiang Wang, Li Zhang, Luca Bertinetto, Philip H.S. Torr

## 🧩 Problem to Solve

기존 객체 추적(Object Tracking) 연구는 대개 바운딩 박스(Bounding Box) 형태로 객체의 위치를 추정하며, 비디오 객체 분할(Video Object Segmentation, VOS) 연구는 픽셀 단위의 마스크를 생성하지만 실시간 처리가 어렵고 초기화에 마스크 정보가 필요하다는 한계가 있습니다. 이 논문은 객체 추적과 비디오 객체 분할이라는 두 가지 문제를 동시에 해결하면서, 실시간으로 동작하고 간단한 바운딩 박스 초기화만으로도 정확한 픽셀 단위 마스크를 생성할 수 있는 통합된 프레임워크를 제안하는 것을 목표로 합니다.

## ✨ Key Contributions

- **통합 프레임워크:** 시각적 객체 추적(Visual Object Tracking)과 비디오 객체 분할(VOS)을 단일하고 간단한 방법으로 실시간 수행하는 SiamMask를 제안합니다.
- **멀티태스크 학습:** 인기 있는 완전 컨볼루션(Fully-Convolutional) Siamese 네트워크 접근 방식을 확장하여, 기존의 유사성 측정 및 바운딩 박스 회귀 손실에 이진 분할(Binary Segmentation) 손실을 추가한 멀티태스크 학습 절차를 개선합니다.
- **실시간 성능:** 오프라인 학습 완료 후, 단일 바운딩 박스 초기화만으로 초당 약 55 프레임(FPS)의 높은 속도로 객체 추적 및 분할을 동시에 수행합니다.
- **경쟁력 있는 결과:** 시각적 객체 추적 벤치마크(VOT-2016, VOT-2018, GOT-10k, TrackingNet)에서 실시간 최신 기술(SOTA) 결과를 달성하며, 동시에 비디오 객체 분할 벤치마크(DAVIS-2016, DAVIS-2017, YouTube-VOS)에서도 경쟁력 있는 성능을 고속으로 입증합니다.
- **다중 객체 추적 및 분할 확장:** 제안하는 프레임워크가 다중 객체 추적 및 분할(Multiple Object Tracking and Segmentation, MOTS) 문제에도 효과적으로 확장될 수 있음을 보여주며, YouTube-VIS 챌린지에서 2위를 차지했습니다.

## 📎 Related Works

- **시각적 객체 추적 (Visual Object Tracking):**
  - **온라인 학습 방식:** 초기에는 첫 프레임의 정보를 기반으로 온라인에서 판별 분류기(discriminative classifier)를 학습하고 업데이트하는 방식이 주류였습니다. 대표적으로 푸리에 도메인(Fourier domain)을 활용하여 고속 처리가 가능한 상관 필터(Correlation Filter) 기반 방법들(e.g., [36], [37], [38], [39], [40], [41], [42], [22], [43])이 있습니다.
  - **오프라인 학습 방식 (Siamese Network):** 2016년부터 Siamese 네트워크 [23], [44], [45] 기반의 새로운 패러다임이 인기를 얻었습니다. 이 방식은 수백만 쌍의 비디오 프레임에 대해 오프라인으로 유사성 함수를 학습하며, 테스트 시에는 단순히 함수를 평가하기만 합니다. SiamFC [23]와 SiamRPN [24]이 대표적이며, 영역 제안(Region Proposal) [24], [29], 하드 네거티브 마이닝(Hard Negative Mining) [25], 앙상블(Ensembling) [26], [46] 등의 개선이 이루어졌습니다.
- **비디오 객체 분할 (Video Object Segmentation, VOS):**
  - **정확성 중심:** 전통적으로 VOS 알고리즘은 객체 표현의 정확성에 더 중점을 두었으며, 프레임 간 일관성을 활용하기 위해 첫 프레임의 마스크를 시간적으로 인접한 프레임으로 전파하는 방식(e.g., 그래프 레이블링 [11], [12], [14], [16], [68], [69])이 사용되었습니다. OSVOS-S [31]와 Mask-Track [13]은 각각 온라인 미세 조정(fine-tuning)과 최신 마스크 예측 및 광학 흐름(optical flow)을 활용하여 정확도를 높였습니다.
  - **속도 향상 노력:** 최근에는 더 빠른 VOS 접근 방식(e.g., OSMN [15], RGMP [17])에 대한 관심이 증가했지만, 여전히 실시간 동작에는 미치지 못했습니다.
- **추적 및 분할 통합:** 과거에는 추적기가 대략적인 이진 마스크를 생성하는 경우도 있었으나, 현대에는 대부분 직사각형 바운딩 박스를 사용하고 정확한 마스크를 위해 속도나 온라인 동작을 포기하는 경향이 있었습니다. 최근 예외로는 초고해상도 기반 추적기 [75]나 간단한 직사각형 초기화로 마스크를 출력하는 방법 [13], [76]이 있습니다.

## 🛠️ Methodology

SiamMask는 완전 컨볼루션 Siamese 네트워크 프레임워크를 기반으로 하며, SiamFC [23] 및 SiamRPN [24] 아키텍처를 확장하여 마스크 분할 브랜치를 추가합니다.

1. **네트워크 아키텍처:**

   - **공유 백본 ($\text{f}_{\theta}$):** ResNet-50 [88] 아키텍처를 사용하며, 깊은 레이어에서 높은 공간 해상도를 얻기 위해 출력 스트라이드(output stride)를 8로 줄이고 팽창 컨볼루션(dilated convolutions) [89]을 사용하여 수용 필드(receptive field)를 늘립니다.
   - **깊이별 상호 상관관계 (Depth-wise Cross-Correlation):** 기존의 단순한 교차 상관관계($\ast$)를 깊이별 교차 상관관계($\star_d$)로 대체하여 다중 채널 응답 맵($g_{n, \theta}$)을 생성합니다. 이는 각 RoW(Response of a candidate Window)가 대상 객체에 대한 더 풍부한 정보를 인코딩할 수 있도록 합니다.
   - **두 가지 변형:**
     - **2-브랜치 SiamMask:** SiamFC 기반으로, RoW를 대상 객체/배경으로 분류하는 $\text{p}_{\omega}$ 브랜치와 각 RoW에 대한 마스크를 출력하는 $\text{h}_{\phi}$ 분할 브랜치로 구성됩니다.
     - **3-브랜치 SiamMask:** SiamRPN 기반으로, 2-브랜치에 바운딩 박스 회귀 브랜치 $\text{b}_{\sigma}$를 추가합니다.

2. **마스크 표현 및 정제 (Mask Representation and Refinement):**

   - DeepMask [30] 및 SharpMask [84]의 정신을 따라 평면화된 객체 표현(17x17 RoW)에서 마스크를 생성합니다.
   - 분할 브랜치 $\text{h}_{\phi}$는 두 개의 1x1 컨볼루션 레이어로 구성되어 각 픽셀 분류기가 전체 RoW의 정보를 활용할 수 있도록 합니다.
   - Mask R-CNN [83]과 유사하게, 여러 개의 정제 모듈(refinement modules)을 사용하여 저해상도와 고해상도 특징을 결합하여 더 정확한 객체 마스크를 생성합니다. 각 모듈은 업샘플링 레이어와 스킵 연결(skip connections)을 통해 해상도를 점진적으로 높입니다.

3. **손실 함수 (Loss Function):**

   - **분할 브랜치 손실 ($\text{L}_{\text{mask}}$):** 각 RoW에 대한 픽셀 단위 이진 로지스틱 회귀 손실을 사용합니다. 오직 양성 RoW(긍정 샘플)에 대해서만 계산됩니다.
     $$L_{\text{mask}}(\theta, \phi) = \sum_n \left( \frac{1+y_n}{2wh} \sum_{ij} \log(1+e^{-c_{ij}^n m_{ij}^n}) \right)$$
   - **다중 태스크 손실:**
     - $\text{L}_{\text{2B}}$ (2-브랜치): $L_{\text{2B}} = \lambda_1 \cdot L_{\text{mask}} + \lambda_2 \cdot L_{\text{sim}}$
     - $\text{L}_{\text{3B}}$ (3-브랜치): $L_{\text{3B}} = \lambda_1 \cdot L_{\text{mask}} + \lambda_2 \cdot L_{\text{score}} + \lambda_3 \cdot L_{\text{reg}}$
     - $\lambda_1=32$, $\lambda_2=\lambda_3=1$로 설정합니다.

4. **바운딩 박스 생성:**

   - VOS 벤치마크는 이진 마스크를 요구하지만, 추적 벤치마크는 바운딩 박스를 요구합니다.
   - **세 가지 전략:**
     1. **Min-max (축 정렬 직사각형):** 객체를 포함하는 최소 축 정렬 직사각형.
     2. **MBR (Minimum Bounding Rectangle):** 최소 회전 경계 직사각형.
     3. **Opt (VOT 최적화):** VOT 벤치마크에서 제안된 자동 바운딩 박스 생성 최적화 전략.

5. **다중 객체 추적 및 분할 (Multiple Object Tracking and Segmentation, MOTS):**
   - **두 단계 캐스케이드 전략:** 사전 학습된 SiamMask 모델을 사용하여 각 객체에 대해 두 단계로 적용됩니다.
     1. **Stage 1 (Coarse Location):** SiamMask의 회귀 브랜치가 각 객체의 대략적인 위치를 예측합니다.
     2. **Stage 2 (Refine Mask):** 첫 번째 단계에서 최고 점수를 받은 바운딩 박스에 해당하는 검색 이미지 영역을 추출하여 분할 브랜치에서 정제된 마스크를 예측합니다.
   - **데이터 연관 (Data Association):** 외부 객체 탐지기 [87]로 감지된 마스크와 SiamMask가 예측한 마스크 간의 IoU를 기반으로 헝가리안 알고리즘(Hungarian algorithm)을 사용하여 객체 ID를 프레임 간에 연결합니다.

## 📊 Results

- **속도:** NVIDIA RTX 2080 GPU에서 2-브랜치 SiamMask는 60 FPS, 3-브랜치 SiamMask는 55 FPS로 실시간 동작합니다.
- **객체 추적 성능 (VOT-2016, VOT-2018):**
  - SiamMask-Opt가 가장 높은 EAO(Expected Average Overlap)와 정확도를 보이지만, 계산 비용이 높습니다.
  - SiamMask-MBR은 SiamMask-box(회귀 브랜치에서 직접 바운딩 박스 출력)보다 EAO 및 정확도 면에서 뛰어난 성능을 보이며 실시간 속도를 유지합니다.
  - VOT-2018 벤치마크에서 3-브랜치 SiamMask(MBR 전략)는 DaSiamRPN [25]을 크게 능가하며, 실시간 추적기 중 최신 기술을 달성했습니다.
  - SiameseFC 대비 EAO 19.8%p 증가를 보였습니다.
- **객체 추적 성능 (GOT-10k, TrackingNet):**
  - GOT-10k에서 SiamMask는 CFNet [43] 대비 평균 오버랩 37%, 성공률 최대 150%p 향상을 달성하여 뛰어난 일반화 능력을 입증했습니다.
  - TrackingNet에서 SiamMask는 ATOM [100]을 포함한 모든 경쟁자보다 우수한 성능을 보였으며, 온라인 적응을 수행하는 ATOM보다도 약간 더 나은 결과를 얻었습니다.
- **비디오 객체 분할 성능 (DAVIS-2016, DAVIS-2017, YouTube-VOS):**
  - DAVIS-2016에서 SiamMask는 다른 빠른 VOS 방법들보다 훨씬 빠르면서도 유사한 정확도를 보였습니다.
  - OnAVOS [32] 및 MSK [13]와 같은 고정밀 방법들보다 수백 배 빠릅니다.
  - 영역 오버랩($J_D$) 및 윤곽 정확도($F_D$)의 감소(decay)에서 가장 낮은 값(최고 성능)을 보여, 시간 경과에 따른 뛰어난 견고성을 입증했습니다.
  - YouTube-VOS에서 SiamMask는 `seen` 클래스에 대해 가장 높은 정확도를 달성했습니다.
- **다중 객체 추적 및 분할 (YouTube-VIS):**
  - 두 단계 SiamMask는 단일 단계 방식 대비 mAP 1.5%p, 평균 리콜 1.6%p 향상을 보였습니다.
  - 2019 YouTube-VIS 챌린지에서 공식 베이스라인 대비 mAP를 46% 상대적으로 향상시키며 2위를 차지했습니다.

## 🧠 Insights & Discussion

- **강점:**
  - SiamMask는 추적과 분할을 위한 통합된 실시간 프레임워크를 제공하여, 기존 바운딩 박스 추적의 한계를 넘어선 픽셀 단위의 풍부한 객체 표현을 제공합니다.
  - 간단한 바운딩 박스 초기화만으로도 높은 정확도의 마스크를 생성하며, 테스트 시 온라인 업데이트 없이 동작하여 높은 실용성을 가집니다.
  - 시간 경과에 따른 성능 저하(decay)가 낮아 긴 시퀀스에서도 효과적입니다.
  - 추가적인 태스크 학습(분할)이 추적 성능 향상에 정규화(regularization) 효과를 줄 수 있다는 가능성을 보여줍니다.
- **한계:**
  - **극심한 모션 블러:** 갑작스러운 카메라 움직임이나 객체의 빠른 움직임으로 인한 극심한 모션 블러 상황에서는 성능 저하가 발생할 수 있습니다.
  - **"비객체" 초기화:** 사용자가 객체가 아닌 텍스처나 객체의 일부를 추적 대상으로 초기화할 경우, 모델이 "객체"에 편향되어 있어 실패할 수 있습니다.
  - **혼란 (Confusion) 문제 (MOTS):** 다중 객체가 서로 매우 가까이 있을 때 픽셀을 올바른 객체 ID에 매핑하는 데 어려움이 있습니다. 한 객체가 다른 객체의 마스크를 "하이재킹"할 수 있습니다.
- **향후 연구 방향:**
  - 혼란 문제를 해결하기 위해 동일 마스크 내 픽셀 간의 관계를 모델링하는 방법(e.g., 조건부 랜덤 필드, 그래프 신경망)을 고려할 수 있지만, 이는 필연적으로 추적 속도를 저하시킬 것입니다.

## 📌 TL;DR

SiamMask는 시각적 객체 추적과 비디오 객체 분할을 단일 Siamese 네트워크 프레임워크에서 실시간으로 수행하는 멀티태스크 학습 기반 방법입니다. 기존 Siamese 추적기에 이진 분할 브랜치를 추가하고 깊이별 상호 상관관계 및 마스크 정제 모듈을 사용하여 픽셀 단위 마스크를 생성합니다. 간단한 바운딩 박스 초기화만으로 초당 55프레임 이상의 속도로 추적 및 분할을 동시에 수행하며, 최신 실시간 추적 벤치마크에서 우수한 성능을, VOS 벤치마크에서는 가장 빠른 속도를 달성했습니다. 또한, 2단계 캐스케이드 방식을 통해 다중 객체 추적 및 분할로도 확장 가능하여 뛰어난 성능을 보였습니다.
