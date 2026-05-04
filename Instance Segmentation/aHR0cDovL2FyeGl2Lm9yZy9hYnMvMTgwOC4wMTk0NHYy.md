# V-FCNN: Volumetric Fully Convolution Neural Network For Automatic Atrial Segmentation

Nicolò Savioli, Giovanni Montana, Pablo Lamata (2018)

## 🧩 Problem to Solve

본 논문은 심방세동(Atrial Fibrillation, AF) 환자의 심방(atria) 구조 변화를 정밀하게 분석하기 위한 자동 분할(automatic segmentation) 문제를 다룬다. 심방의 해부학적 변화를 특성화하는 것은 임상적 바이오마커(clinical biomarkers)를 정의하는 데 매우 중요하며, 이는 특히 후기 가돌리늄 강화 자기공명영상(Late Gadolinium Enhanced Magnetic Resonance Imaging, LGE-MRI) 연구에서 필수적이다.

현재의 심방 분할은 주로 수동 절차에 의존하고 있어 시간이 많이 소요될 뿐만 아니라, 작업자 간의 편차가 발생하는 오류에 취약하다. 특히 심방 조직과 주변 배경 사이의 대비(contrast)가 낮아 자동 분할이 매우 어려운 과제라는 점이 문제의 핵심이다. 따라서 본 연구의 목표는 LGE-MRI 영상에서 심방 전체 부피를 완전 자동으로 분할할 수 있는 딥러닝 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D-convolution 커널을 사용하는 Volumetric Fully Convolutional Neural Network (V-FCNN)를 통해 심방의 전체 볼륨을 한 번에 처리(one-shot)함으로써, 고해상도 이미지에 내재된 공간적 중복성(spatial redundancy)을 통합하는 것이다.

또한, 전역적인 형태(bulk shape)를 캡처하는 능력과 과분할(over-segmentation)로 인한 국소적 오류를 줄이는 능력을 결합하기 위해, Mean Square Error (MSE)와 Dice Loss (DL)를 혼합한 하이브리드 손실 함수를 도입하였다. 이를 통해 네트워크가 국소 최솟값(local minimum)에 빠지는 것을 방지하고 학습의 수렴 속도를 높이고자 하였다.

## 📎 Related Works

기존의 심방 분할 접근 방식으로는 3D level-set 내의 multi-atlas registration 방식이 제안되었으며, 심방 본체와 폐정맥(pulmonary vein) 영역에서 준수한 성능을 보였으나 계산 비용이 매우 크다는 한계가 있었다. 이러한 계산 부담은 Convolutional Neural Networks (CNNs)를 통해 완화될 수 있으며, 일부 연구에서는 2D MRI 슬라이스를 분석하는 방식이 사용되었다.

본 연구의 기초가 된 V-Net은 3D 커널을 사용하여 볼륨 데이터의 공간적 중복성을 고려하는 구조이다. V-FCNN은 V-Net의 구조를 참고하되, 메모리 부담을 줄이고 학습 속도를 높이기 위해 일부 구조를 수정하였다. 특히, V-Net에서 사용되는 skip-connection(skip-paths)이 그래디언트 전파 과정에서 GPU 메모리 사용량을 증가시키고 학습 속도를 늦춘다는 점에 주목하여, 이를 제거한 효율적인 구조를 탐색하였다.

## 🛠️ Methodology

### 전체 시스템 구조

V-FCNN은 크게 볼륨 다운샘플링 경로(volumetric down-sampling path)와 볼륨 업샘플링 경로(volumetric up-sampling path)의 두 가지 경로로 구성된다.

1. **Down-sampling Path**: 입력된 $(X, Y, Z)$ 전체 볼륨을 받아 4개의 3D-convolution 블록을 통해 점진적으로 슬라이스 크기와 스택의 수(Z축)를 줄이며 특징을 압축한다.
2. **Up-sampling Path**: 압축된 특징 맵을 다시 원래의 입력 크기로 복원하여 최종적으로 3D 마스크(mask)를 생성한다.

각 경로의 블록은 3D-convolution 이후 PreLU 활성화 함수와 3D-Batch Normalisation (BN) 층으로 구성되며, 커널의 개수는 $16, 32, 64, 128$개로 순차적으로 증가한다. 커널 크기는 $3 \times 3 \times 3$으로 고정되었다.

### 주요 설계 결정

- **Max-pooling 제거**: 공간적 해상도 손실을 방지하기 위해 max-pooling 연산을 사용하지 않았다.
- **Skip-connection 제거**: 학습 속도를 높이고 메모리 효율성을 극대화하기 위해 V-Net의 skip-layers를 제거하였다.
- **해상도 조정**: GPU 메모리 한계($\sim 32\text{ GB}$ 미만 환경)를 고려하여, 입력 이미지 크기를 bi-cubic interpolation을 통해 $127 \times 127 \times 88$로 축소하여 학습하였으며, 최종 결과물은 다시 $640 \times 640$ 크기로 복원하였다.

### 손실 함수 (Loss Function)

본 모델은 전역적 특징과 국소적 세부 사항을 동시에 학습하기 위해 다음과 같은 bimodal loss를 사용한다.

$$Loss = MSE_{loss} + \lambda \cdot DICE_{loss}$$

상세 식은 다음과 같다.

$$Loss = \frac{1}{Z} \sum_{s=0}^{Z} (y[s] - \hat{y}[s])^2 + \lambda \cdot \frac{1}{Z} \sum_{s=0}^{Z} \frac{2 \sum_{i=1}^{N} y[s]_i \hat{y}[s]_i}{\sum_{i=1}^{N} y[s]_i + \sum_{i=1}^{N} \hat{y}[s]_i}$$

여기서 $y[s]$는 정답(Ground Truth) 이진 슬라이스, $\hat{y}[s]$는 예측 마스크, $s$는 Z축의 인덱스, $i$는 각 마스크 내의 픽셀 인덱스를 의미한다. $\lambda$는 Dice Loss의 비중을 조절하는 하이퍼파라미터로 $1 \times 10^{-3}$으로 설정되었다. $MSE_{loss}$는 전역적인 특징에 집중하는 정규화 역할을 하며, $DICE_{loss}$는 국소적인 세부 사항을 복원하는 역할을 한다.

### 학습 및 전처리 절차

- **전처리**: CLAHE(Contrast Limited Adaptive Histogram Equalization)를 통한 그레이스케일 강도 균일화, High-Pass Filter 및 Gaussian blurring filter를 이용한 노이즈 제거를 수행하였다.
- **학습 설정**: SGD 옵티마이저를 사용하였으며, 학습률(learning rate)은 $1 \times 10^{-4}$, 모멘텀은 $0.9$, weight decay는 $1 \times 10^{-5}$로 설정하여 1000 epoch까지 학습하였다.
- **데이터 증강**: 일반화 성능 향상을 위해 랜덤 수직/수평 뒤집기(flip)와 평면 이동(translation)을 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: [2018 Atrial Segmentation Challenge]에서 제공한 100개의 3D GE-MRI 데이터와 마스크를 사용하였다.
- **실험 단계**: 1단계 준비 단계(훈련 90명, 검증 5명)와 2단계 경쟁 단계(테스트 54명)로 나누어 진행하였다.
- **평가 지표**: Dice Metric (DM)과 surface Hausdorff Distance (HD)를 사용하였다. 특히 DM은 심방의 상단(Top, 폐정맥 포함), 중단(Middle), 하단(Bottom, 판막 평면 포함) 세 영역으로 나누어 측정하였다.

### 정량적 결과

- **준비 단계(5명 환자)**: 평균 DM은 $76.58 \pm 7.87\%$를 기록하였다. 영역별로는 중단(Middle) 영역에서 $82.1 \pm 1.4\%$로 가장 높은 성능을 보였으며, 상단(Top) 영역은 $69.6 \pm 16.1\%$로 상대적으로 낮았다. 평균 HD는 $0.59\text{ mm}$였다.
- **경쟁 단계(54명 환자)**: 최종 Dice Metric $92.5\%$를 달성하였다.

### 분석 및 관찰

시각적 분석 결과, 심방의 중단 섹션에서는 매우 정확한 분할이 이루어졌으나, 해부학적 변동성이 큰 폐정맥이 포함된 상단 섹션과 심방-심실 경계인 하단 판막 평면 영역에서는 정확도가 떨어지는 경향이 확인되었다. 일부 케이스에서는 형상이 인위적으로 평평해지는(artificial flattening) 현상이 관찰되었다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 3D-convolution 커널을 통해 볼륨 데이터의 공간적 일관성(spatial coherence)을 효과적으로 활용하였으며, MSE와 Dice Loss의 결합을 통해 학습의 안정성을 확보하고 전역/국소 특징을 동시에 포착하는 성과를 거두었다.

### 한계점 및 비판적 해석

가장 큰 한계는 **하드웨어 제약으로 인한 입력 해상도의 과도한 축소**이다. $640 \times 640$의 원본 이미지를 $127 \times 127$로 줄임으로써 GPU 메모리 문제는 해결했으나, 이 과정에서 불가피하게 세부 정보가 손실되었다. 저자들은 이것이 Dice score가 더 높아지지 못한 주된 이유라고 분석하고 있으며, NVIDIA V100과 같은 고용량 메모리 GPU의 필요성을 언급한다.

또한, 학습 속도를 위해 제거한 skip-connection은 사실 정밀한 위치 정보(localization)를 복원하는 데 매우 중요한 역할을 한다. 이를 제거함으로써 수렴 속도는 빨라졌을지 모르나, 폐정맥과 같은 미세 구조의 분할 성능 저하에 영향을 주었을 가능성이 크다.

### 향후 연구 방향

- **해상도 개선**: 더 높은 해상도의 입력을 처리할 수 있는 하드웨어 도입 또는 ROI(Region of Interest)를 먼저 추출한 뒤 고해상도로 재처리하는 Dual-CNN 전략이 필요하다.
- **구조적 보완**: 3D 커널의 메모리 효율성을 높이기 위해 Recurrent units(순환 신경망)를 결합하여 인접 슬라이스 간의 중복성을 캡처하는 방안이 제시되었다.
- **손실 함수 최적화**: L1/L2 loss나 통계적 거리 측정법을 추가하여 최적의 가중치를 찾는 연구가 필요하다.

## 📌 TL;DR

본 논문은 LGE-MRI 영상에서 심방을 자동으로 분할하기 위해 3D-CNN 기반의 **V-FCNN** 아키텍처를 제안하였다. **MSE와 Dice Loss를 결합한 하이브리드 손실 함수**를 통해 전역적 형태와 국소적 세부 사항을 동시에 학습하였으며, 경쟁 단계에서 **92.5%의 Dice Metric**을 달성하였다. 다만, GPU 메모리 한계로 인해 입력 해상도를 크게 낮춘 점이 세부 구조(폐정맥 등)의 분할 정확도를 제한하는 요소가 되었으며, 향후 고해상도 처리 기법이나 순환 신경망(RNN)의 결합이 필요함을 시사한다.
