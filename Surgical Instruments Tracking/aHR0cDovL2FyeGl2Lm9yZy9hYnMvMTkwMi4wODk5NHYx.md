# U-NetPlus: A Modified Encoder-Decoder U-Net Architecture for Semantic and Instance Segmentation of Surgical Instrument

S. M. Kamrul Hasan and Cristian A. Linte, Senior Member, IEEE

## 🧩 Problem to Solve

최소 침습 수술(Minimally Invasive Surgery)에서 로봇 보조 수술 기구의 정확한 위치 추적 및 식별은 수술의 정밀도를 높이는 데 매우 중요합니다. 하지만 변화하는 조명, 시각적 폐색(occlusion), 비관련 객체의 존재 등의 문제로 인해 수술 영상에서 수술 기구를 정확하게 분할하고 식별하는 것은 여전히 어려운 과제입니다. 기존 U-Net 아키텍처는 전치 컨볼루션(transposed convolution)으로 인한 체커보드 아티팩트(checkerboard artifacts)와 느린 수렴 속도 등의 한계를 가지고 있어, 다중 클래스(multi-class) 수술 기구 분할(instrument part 및 type)에서 충분한 정확도를 달성하지 못했습니다.

## ✨ Key Contributions

- **U-NetPlus 아키텍처 제안**: U-Net 모델을 개선하여 수술 기구의 의미론적(semantic) 및 인스턴스(instance) 분할을 위한 U-NetPlus를 제안했습니다.
- **사전 훈련된 인코더 도입**: Batch Normalization이 적용된 사전 훈련된 VGG-11 및 VGG-16 네트워크를 인코더로 사용하여 네트워크의 수렴 속도를 높이고 성능을 향상시켰습니다. 이는 제한된 의료 영상 데이터셋에서 특히 효과적입니다.
- **디코더 재설계**: 기존의 전치 컨볼루션 연산을 가장 가까운 이웃(nearest-neighbor, NN) 보간법 기반의 업샘플링(upsampling)으로 대체하여 체커보드 아티팩트를 제거하고 학습 가능한 파라미터 수를 줄였습니다.
- **효과적인 데이터 증강**: Albumentations 라이브러리를 활용한 빠르고 유연한 데이터 증강(affine 및 elastic 변환) 기법을 사용하여 과적합(overfitting)을 방지하고 모델의 견고성을 높였습니다.
- **성능 우수성 입증**: MICCAI 2017 EndoVis Challenge 데이터셋에서 기존 U-Net 및 최신 TernausNet을 포함한 다양한 최신 기법보다 우수한 분할 성능(DICE 및 IoU)을 달성했습니다.

## 📎 Related Works

- **FCN (Fully Convolutional Network)**: Long et al. [4]이 제안한 최초의 의미론적 영상 분할을 위한 심층 컨볼루션 신경망입니다.
- **U-Net**: Ronneberger et al. [7]이 제안한 의료 영상 분할을 위한 대표적인 인코더-디코더 아키텍처입니다.
- **전치 컨볼루션 문제**: Odena et al. [11], Radford et al. [15], Salimans et al. [16] 등은 전치 컨볼루션으로 인한 체커보드 아티팩트 문제를 언급했습니다.
- **가장 가까운 이웃 보간**: Jiang & Wang [9] 및 Jia et al. [10]은 이미지 재구성 및 초해상도(super-resolution)에 이 기법을 사용했습니다.
- **수술 기구 분할 연구**: ToolNetH, ToolNetMS, FCN-8s [12]와 같은 초기 시도들이 있었으며, Shvets et al. [13]과 Pakhomov et al. [14]는 다중 클래스 기구 분할을 제안했지만 전치 컨볼루션을 사용했습니다.
- **TernausNet**: Iglovikov & Shvets [17]이 VGG11 인코더와 U-Net을 결합하여 ImageNet에서 사전 훈련된 모델로 뛰어난 성능을 보여주었으나, 아티팩트 문제가 여전히 존재했습니다.
- **데이터 증강 라이브러리**: Buslaev et al. [23]은 Albumentations 라이브러리를 제안하여 빠르고 유연한 데이터 증강을 가능하게 했습니다.

## 🛠️ Methodology

본 연구에서 제안하는 U-NetPlus 모델은 다음과 같은 구성으로 이루어져 있습니다.

- **아키텍처 개요 (U-NetPlus)**:

  - **인코더 (Encoder)**: 배치 정규화(Batch Normalization)가 적용된 사전 훈련된 VGG-11 또는 VGG-16 네트워크를 사용합니다. ImageNet으로 사전 훈련된 가중치를 재사용하여 제한된 데이터셋에서도 빠른 수렴과 높은 정확도를 얻습니다. $3 \times 3$ 커널 크기의 컨볼루션 레이어와 ReLU 활성화 함수로 구성되며, 최대 풀링(max pooling)으로 특징 맵 크기를 줄입니다.
  - **디코더 (Decoder)**: 기존 U-Net의 전치 컨볼루션(deconvolution) 레이어 대신 가장 가까운 이웃(Nearest-Neighbor, NN) 업샘플링 레이어를 사용합니다. 각 블록 시작 시 스케일 팩터 2로 업샘플링한 후 두 개의 컨볼루션 레이어와 ReLU 함수를 거쳐 공간 차원을 두 배로 증가시킵니다. 이 방식은 아티팩트를 줄이고 학습 파라미터 수를 줄입니다.
  - **스킵 연결 (Skip Connections)**: 인코더와 디코더의 동일한 크기 블록 사이에 스킵 연결을 추가하여 상세한 특징 정보를 전달하고 마스크 정렬의 정확도를 높이며, 그래디언트 소실(vanishing gradient) 문제를 완화합니다.

- **데이터셋**: MICCAI 2017 EndoVis Challenge의 Robotic Instruments 데이터셋을 사용했습니다.

  - 훈련 데이터: 8개의 225프레임 시퀀스 (총 1800프레임).
  - 테스트 데이터: 8개의 75프레임 시퀀스와 2개의 300프레임 시퀀스.
  - 클래스: 7가지 기구 유형(prograsp forceps, needle driver 등)과 3가지 기구 부분(shaft, wrist, claspers)으로 수동으로 레이블링되어 있습니다.

- **데이터 증강**: `albumentations` 라이브러리를 활용하여 Affine 및 Elastic 변환을 적용했습니다.

  - **Affine 변환**: 스케일링, 이동, 수평/수직 뒤집기, 무작위 밝기 조절, 노이즈 추가 등.
  - **Elastic 변환**: 무작위 변위 필드($\delta x, \delta y$)를 생성하고 가우시안 필터와 스케일링 팩터 $\alpha$를 적용하여 이미지의 전역적인 형태를 유지하면서 비선형적인 변형을 가합니다.

- **구현 상세**:

  - PyTorch 프레임워크로 구현했습니다.
  - 전처리: 영상 프레임에서 불필요한 검은색 경계를 자르고, 평균을 빼고 표준편차로 나누는 Z-점수 정규화(Z-score normalization)를 수행했습니다.
  - Batch Normalization은 각 가중치 레이어 전에 적용되어 학습 속도를 가속화했습니다.
  - 최적화 도구: Adam optimizer, 학습률(learning rate) $0.00001$.
  - 드롭아웃(dropout)은 성능 저하로 인해 사용하지 않았습니다.
  - 총 100 epoch 동안 훈련했으며, 각 epoch 전에 훈련 세트를 섞었고, 배치 크기(batch size)는 4였습니다.
  - 하드웨어: NVIDIA GTX 1080 Ti GPU (11GB 메모리).

- **성능 지표**:
  - **Jaccard Index (IoU)**: $J(T_1, P_1) = \frac{|T_1 \cap P_1|}{|T_1 \cup P_1|}$ 로 정의되며, 예측과 실제 값 사이의 겹침을 측정합니다.
  - **DICE Coefficient**: $D(T_1, P_1) = \frac{2|T_1 \cap P_1|}{|T_1| + |P_1|}$ 로 정의되며, 두 집합의 유사도를 측정합니다.
  - **손실 함수**: 픽셀 분류 문제이므로 Cross-entropy $H$와 IoU $J$를 결합한 $L = H - \text{log}J$를 사용해 손실을 최소화합니다.

## 📊 Results

- **정량적 결과**:

  - **이진 분할(Binary Segmentation)**: U-NetPlus-VGG-16은 90.20% DICE와 83.75% IoU를 달성하여 가장 우수한 성능을 보였습니다. 이는 기존 U-Net 대비 6.91% DICE 및 11.01% IoU 향상이며, TernausNet 대비로도 0.21% DICE 및 0.18% IoU 향상입니다.
  - **기구 부분 분할(Instrument Part Segmentation)**: U-NetPlus-VGG-16은 76.26% DICE와 65.75% IoU를 기록하여 U-Net, U-Net+NN, TernausNet을 능가했습니다.
  - **기구 유형 분할(Instrument Type Segmentation)**: U-NetPlus-VGG-11이 46.07% DICE와 34.84% IoU로 U-NetPlus-VGG-16(45.32% DICE, 34.19% IoU)보다 약간 더 나은 성능을 보였습니다.
  - 가장 가까운 이웃 보간법을 U-Net 디코더에 적용(U-Net+NN)하는 것만으로도 기존 U-Net 대비 성능 향상이 관찰되었습니다.
  - 사전 훈련된 인코더(VGG) 사용은 모델의 훈련 정확도 수렴을 가속화하고 전반적인 정확도를 높였습니다.

- **정성적 결과**:

  - 시각화된 결과(Fig. 4)는 U-NetPlus가 이진 분할, 기구 부분 분할, 기구 유형 분할 모두에서 기존 U-Net, U-Net+NN, TernausNet보다 뛰어난 성능을 보임을 명확히 보여줍니다. U-NetPlus는 불필요한 영역을 더 잘 제거하고 클래스 간 구분을 더 정확하게 수행했습니다.
  - 특히 기구 유형 분할에서 U-NetPlus는 다른 모델들이 구분하지 못하는 클래스들을 더 정확하게 분류하는 능력을 보였습니다.

- **어텐션 연구(Attention Study)**:
  - 새로운 이미지 살리언시(saliency) 기법 [25]을 사용하여 모델이 이미지의 어느 부분에 "집중"하는지 시각화했습니다(Fig. 5).
  - U-NetPlus는 양극성 겸자(bipolar forceps)의 손목(wrist)과 집게(claspers) 같은 목표 영역에 훨씬 더 정확하게 집중하는 모습을 보였습니다. 이는 U-Net, U-Net+NN, TernausNet보다 우수하며, 통합된 기술(사전 훈련 인코더 + NN 보간)의 효과를 입증합니다.

## 🧠 Insights & Discussion

- **혁신적인 통합**: 본 연구는 U-Net 아키텍처에 사전 훈련된 VGG 인코더와 가장 가까운 이웃 보간을 디코더의 업샘플링 기법으로 도입함으로써 의료 영상 분할 분야의 주요 과제들을 효과적으로 해결했습니다.
- **성능 향상 요인**:
  - **사전 훈련된 인코더**: 제한된 의료 영상 데이터셋의 특성을 고려할 때, ImageNet으로 사전 훈련된 VGG 네트워크는 모델의 초기 가중치를 효과적으로 초기화하여 빠른 수렴과 높은 정확도 달성에 결정적인 역할을 했습니다. Batch Normalization은 이러한 최적화 과정을 더욱 안정화하고 가속화했습니다.
  - **가장 가까운 이웃 보간**: 전치 컨볼루션의 고질적인 문제인 체커보드 아티팩트와 불필요한 학습 파라미터를 제거함으로써, 더욱 깨끗하고 정확한 분할 결과를 얻을 수 있었습니다. 이는 메모리 효율성 측면에서도 이점을 제공합니다.
  - **데이터 증강**: 견고한 데이터 증강 전략은 과적합을 방지하고 모델이 다양한 시각적 조건에서도 잘 작동하도록 기여했습니다.
- **시사점**: 이 연구는 완전히 새로운 딥러닝 프레임워크를 제안하기보다는, 기존의 강력한 도구들(U-Net, VGG, NN 보간)을 "숙련되게 통합하고 적응"시키는 것이 특정 작업(수술 기구 분할)의 성능을 크게 향상시킬 수 있음을 입증합니다.
- **제한 및 향후 연구**: U-NetPlus-VGG-16이 기구 부분 분할에서 뛰어난 성능을 보인 반면, 7가지 기구 유형 분할에서는 U-NetPlus-VGG-11에 비해 약간 낮은 성능을 보였습니다. 이는 다중 유형 분류에 대한 추가적인 최적화 또는 아키텍처 개선의 여지가 있음을 시사합니다.

## 📌 TL;DR

- **문제**: 로봇 보조 최소 침습 수술에서 수술 기구의 정확한 식별과 분할은 필수적이지만, 기존 딥러닝 모델(특히 U-Net의 전치 컨볼루션)은 아티팩트와 느린 수렴으로 한계가 있었습니다.
- **해결책**: U-NetPlus라는 개선된 U-Net 아키텍처를 제안합니다. 이 모델은 배치 정규화가 적용된 사전 훈련된 VGG 인코더를 사용하여 학습 속도와 정확도를 높이고, 디코더의 전치 컨볼루션을 가장 가까운 이웃 보간법으로 대체하여 아티팩트를 제거하고 파라미터 수를 줄입니다. 효과적인 데이터 증강 기법도 함께 사용됩니다.
- **결과**: U-NetPlus는 MICCAI 2017 EndoVis 데이터셋에서 이진 분할, 기구 부분 분할, 기구 유형 분할 모두에서 기존 U-Net 변형 및 TernausNet보다 우수한 성능을 달성했으며, 정성적 결과와 어텐션 연구를 통해 더욱 정확하고 집중된 분할 능력을 입증했습니다.
