# Robust Semantic Segmentation of Brain Tumor Regions from 3D MRIs

Andriy Myronenko, Ali Hatamizadeh

## 🧩 Problem to Solve

자동화된 3D MRI 뇌종양 분할은 질병의 진단 및 치료 계획에 필수적인 기본적인 비전 작업입니다. BraTS(Multimodal Brain Tumor Segmentation Challenge)는 이러한 3D MRI 뇌종양 분할을 위한 자동화된 방법을 개선하기 위해 연구자들을 모으는 것을 목표로 합니다. 이 논문은 BraTS 2019 챌린지에 참여하여 뇌종양 분할 정확도를 더욱 향상시키기 위한 3D 의미론적 분할의 모범 사례를 탐구합니다.

## ✨ Key Contributions

* 기존 인코더-디코더 아키텍처와 결합된 손실 함수를 포함하여 3D 의미론적 분할의 모범 사례를 탐구했습니다.
* 다중 모드 3D MRI에서 뇌종양을 분할하기 위한 견고한 심층 학습 기반 접근 방식을 제안했습니다.
* BraTS 2019 챌린지 검증 및 테스트 데이터셋에서 경쟁력 있는 성능을 달성하여 종양 하위 영역(WT, TC, ET)에 대한 높은 Dice 계수와 낮은 Hausdorff 거리를 입증했습니다.
* 다양한 정규화 함수(Group Normalization, Instance Normalization, Batch Normalization)를 비교하고 Instance Normalization이 효율적인 선택임을 확인했습니다.
* 네트워크 너비를 늘리는 것이 성능 향상에 일관되게 기여함을 확인했습니다.

## 📎 Related Works

* **BraTS 2018 최고 제출작들:**
  * **Myronenko [16]:** 보조 작업 디코더를 통해 네트워크에 추가 구조를 부여하는 방법을 탐구. (본 연구의 이전 작업)
  * **Isensee et al. [11]:** 약간의 수정이 있는 일반적인 U-net 아키텍처로도 경쟁력 있는 성능을 달성할 수 있음을 입증.
  * **McKinly et al. [13]:** DenseNet [9] 구조와 확장 컨볼루션(dilated convolutions)이 U-net과 유사한 네트워크에 내장된 분할 CNN을 제안.
  * **Zhou et al. [19]:** 멀티 스케일 컨텍스트 정보, 공유 백본 가중치를 갖는 3개 종양 하위 영역의 계단식 분할, 어텐션 블록 추가 등을 고려한 다양한 네트워크 앙상블 사용을 제안.

## 🛠️ Methodology

* **아키텍처:** 이전 연구 [16]의 인코더-디코더 기반 CNN 아키텍처를 따릅니다.
  * **인코더:**
    * ResNet [8] 블록을 사용하며, 각 블록은 정규화(Instance Normalization 기본 사용), ReLU 및 스킵 연결(skip connection)을 포함한 두 개의 $3 \times 3 \times 3$ 컨볼루션으로 구성됩니다.
    * 스트라이드 컨볼루션을 통해 점진적으로 이미지 차원을 2배 줄이고 특징 크기를 2배 늘립니다.
    * 인코더 엔드포인트는 입력 이미지보다 공간적으로 8배 작습니다.
  * **디코더:**
    * 각 공간 레벨은 특징 수를 2배 줄이고($1 \times 1 \times 1$ 컨볼루션), 3D 선형 업샘플링을 통해 공간 차원을 2배 늘립니다.
    * 해당 인코더 레벨의 출력을 추가합니다.
    * 최종 출력은 원본 이미지와 동일한 공간 크기를 가지며, $1 \times 1 \times 1$ 컨볼루션과 시그모이드 함수를 통해 3개의 종양 하위 영역(강화 종양, 전체 종양, 종양 코어)을 예측합니다.
* **손실 함수:** 다음 항들로 구성된 하이브리드 손실 함수 $L = L_{dice} + L_{focal} + L_{acl}$를 사용합니다.
  * **Dice 손실 ($L_{dice}$):** 소프트 Dice 손실 [15]로, 각 종양 하위 영역에 대해 계산 후 합산됩니다.
        $$ L_{dice} = 1 - \frac{2 \cdot \sum p_{true} \cdot p_{pred}}{\sum p_{true}^2 + \sum p_{pred}^2 + \epsilon} $$
  * **Active Contour 손실 ($L_{acl}$):** 3D 확장 버전의 지도 Active Contour 손실 [6]로, 볼륨 항($L_{vol}$)과 길이 항($L_{length}$)으로 구성됩니다.
        $$ L_{vol} = |\sum p_{pred}(c_1 - p_{true})^2| + |\sum (1 - p_{pred})(c_2 - p_{true})^2| $$
        $$ L_{length} = \sum \sqrt{|(\nabla p_{pred,x})^2 + (\nabla p_{pred,y})^2 + (\nabla p_{pred,z})^2| + \epsilon} $$
  * **Focal 손실 ($L_{focal}$):** 클래스 불균형 문제를 해결하기 위한 손실 함수 [12]입니다 ($\gamma=2$ 설정).
        $$ L_{focal} = -\frac{1}{N} \sum (1 - p_{pred})^\gamma p_{true} \log (p_{pred} + \epsilon) $$
* **최적화:**
  * Adam optimizer를 사용하며, 초기 학습률 $\alpha_0 = 1e-4$를 다음 공식에 따라 점진적으로 감소시킵니다: $\alpha = \alpha_0 \cdot \left(1 - \frac{e}{N_e}\right)^{0.9}$.
* **정규화:**
  * $1e-5$의 가중치로 컨볼루션 커널 파라미터에 L2 노름 정규화를 적용합니다.
  * 초기 인코더 컨볼루션 후 0.2의 비율로 공간 드롭아웃(spatial dropout)을 사용합니다.
* **데이터 전처리 및 증강:**
  * 입력 이미지를 평균 0, 표준편차 1로 정규화합니다 (0이 아닌 복셀 기준).
  * 무작위 강도 시프트 및 스케일, 0.5 확률로 무작위 축 미러 플립을 적용합니다.
  * 훈련 시 $160 \times 192 \times 128$ 크기의 무작위 크롭을 사용합니다.

## 📊 Results

* **BraTS 2019 검증 데이터셋 (125 케이스):**
  * 단일 모델 (배치 크기 8) 기준 Dice 계수 (ET, WT, TC): 0.800, 0.894, 0.834
  * 단일 모델 (배치 크기 8) 기준 Hausdorff 거리 (mm) (ET, WT, TC): 3.921, 5.89, 6.562
* **BraTS 2019 테스트 데이터셋:**
  * 앙상블 모델 기준 Dice 계수 (ET, WT, TC): 0.826, 0.882, 0.837
  * 앙상블 모델 기준 Hausdorff 거리 (mm) (ET, WT, TC): 2.203, 4.713, 3.968
* **훈련 시간:** NVIDIA Tesla V100 32GB GPU 1개에서 300 에포크 훈련에 약 2일 소요. NVIDIA DGX-1 서버(V100 GPU 8개)에서는 약 8시간 소요.
* **추론 시간:** V100 GPU 1개에서 단일 모델 당 0.4초 소요.

## 🧠 Insights & Discussion

* **정규화 함수:** Group Normalization과 Instance Normalization은 유사한 성능을 보였지만, Batch Normalization은 열등했습니다. Instance Normalization이 구현 및 이해가 더 간단하여 기본으로 채택되었습니다.
* **다중 GPU 시스템:** 다중 GPU를 사용한 데이터 병렬화는 단일 GPU와 비슷한 성능을 유지하면서 훈련 속도를 약 8배 향상시켰습니다.
* **데이터 증강 효과:** 무작위 히스토그램 매칭, 아핀 변환, 회전 등 더 정교한 데이터 증강 기법을 시도했지만 추가적인 성능 향상을 가져오지 못했습니다.
* **네트워크 구조:** 네트워크 깊이를 늘리는 것은 성능 개선에 기여하지 못했지만, 네트워크 너비(필터 수)를 늘리는 것은 일관되게 결과를 개선했습니다.
* **결론:** 제안된 의미론적 분할 네트워크는 BraTS 2019 챌린지에서 효과적인 3D 뇌종양 분할 성능을 입증했습니다.

## 📌 TL;DR

이 논문은 BraTS 2019 챌린지를 위한 다중 모드 3D MRI 기반 뇌종양 분할을 위한 견고한 심층 학습 접근 방식을 제시합니다. ResNet 블록을 활용한 인코더-디코더 CNN 아키텍처와 Dice, Focal, Active Contour 손실을 결합한 하이브리드 손실 함수를 사용하여 종양 하위 영역을 분할합니다. Instance Normalization 사용, 네트워크 너비 확장, 효과적인 데이터 증강 기법이 제안되었으며, BraTS 2019 테스트 데이터셋에서 평균 Dice 계수 ET 0.826, WT 0.882, TC 0.837이라는 경쟁력 있는 결과를 달성하여 3D 뇌종양 분할의 정확도를 향상시켰습니다.
