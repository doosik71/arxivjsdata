# DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation

Debesh Jha, Michael A. Riegler, Dag Johansen, Pål Halvorsen, Håvard D. Johansen (2020)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation)의 정확도와 일반화 성능을 높이는 것이다. 의료 영상 분할은 임상 진단 및 치료 계획 수립에 필수적이지만, 다음과 같은 현실적인 어려움이 존재한다.

- **데이터의 부족 및 불균형**: 고품질의 레이블링된 데이터셋을 확보하기 어렵고, 데이터가 불균형하게 분포되어 있다.
- **영상의 낮은 품질 및 다양성**: 환자마다 영상의 특성이 매우 다양하며, 이미지 품질이 낮아 일관된 프로토콜을 적용하기 어렵다.
- **까다로운 대상의 검출**: 특히 대장 내시경 영상에서 평평한 용종(flat polyps)과 같이 경계가 불분명한 대상은 기존 모델들이 놓치기 쉬우며, 이는 암 진단 지연으로 이어질 수 있다.

따라서 본 논문의 목표는 다양한 의료 영상 모달리티(colonoscopy, dermoscopy, microscopy 등)에서 강건하게(robust) 작동하며, 특히 까다로운 이미지에서도 높은 정확도를 보이는 일반화 가능한 세그멘테이션 모델인 DoubleU-Net을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 두 개의 U-Net 구조를 직렬로 쌓아(stacked) 특징 추출 능력을 극대화하는 것이다. 주요 기여 사항은 다음과 같다.

- **DoubleU-Net 아키텍처 제안**: 두 개의 U-Net을 순차적으로 배치하여 첫 번째 네트워크의 출력을 바탕으로 두 번째 네트워크가 정밀하게 보정하는 구조를 설계하였다.
- **전이 학습(Transfer Learning) 활용**: 첫 번째 U-Net의 Encoder로 ImageNet에서 사전 학습된 VGG-19를 사용하여, 데이터가 부족한 의료 영상 도메인에서도 효과적으로 특징을 추출할 수 있도록 하였다.
- **문맥 정보 캡처**: Atrous Spatial Pyramid Pooling (ASPP)과 Squeeze-and-Excitation (SE) 블록을 통합하여 다양한 스케일의 문맥 정보와 중요한 채널 특성을 효율적으로 학습하게 하였다.
- **다양한 데이터셋을 통한 검증**: 네 가지 서로 다른 의료 영상 데이터셋을 통해 모델의 범용성과 일반화 성능을 입증하였다.

## 📎 Related Works

논문에서는 의료 영상 분할을 위해 사용된 기존의 딥러닝 접근 방식들을 소개하며 그 한계를 지적한다.

- **FCN 및 U-Net**: 픽셀 단위 예측을 수행하는 기본 구조로 널리 사용되나, 매우 까다로운 이미지(예: 평평한 용종)에서는 성능이 제한적이다.
- **DeepLabV3**: Atrous Convolution을 통해 다중 스케일 문맥 정보를 통합하며 성능을 개선하였다.
- **UNet++ 및 MultiResUNet**: Skip connection을 재설계하거나 잔차 연결(residual connections)을 도입하여 U-Net의 성능을 높이려 시도하였다.
- **차별점**: 기존 연구들이 단일 네트워크의 깊이나 연결 구조를 최적화하는 데 집중했다면, DoubleU-Net은 두 개의 U-Net을 쌓고 사전 학습된 모델과 ASPP를 결합하여 '초기 예측 $\rightarrow$ 정밀 보정'의 파이프라인을 구축함으로써 더 높은 강건성을 확보하였다.

## 🛠️ Methodology

### 전체 시스템 구조

DoubleU-Net은 **NETWORK 1**과 **NETWORK 2**라는 두 개의 수정된 U-Net 구조로 구성된다. 전체 흐름은 다음과 같다.

1. 입력 이미지가 NETWORK 1에 입력되어 중간 마스크인 $\text{Output}_1$을 생성한다.
2. 입력 이미지와 $\text{Output}_1$을 원소별 곱셈(element-wise multiplication)하여 NETWORK 2의 입력으로 사용한다.
3. NETWORK 2는 이 입력을 바탕으로 최종 마스크인 $\text{Output}_2$를 생성한다.

### 주요 구성 요소 및 역할

- **Encoder 1 (VGG-19)**: ImageNet으로 사전 학습된 VGG-19를 사용하여 기본 특징을 추출한다. 이는 학습 데이터 부족 문제를 해결하고 수렴 속도를 높인다.
- **Encoder 2**: 처음부터 학습되는(from scratch) 구조이며, $3 \times 3$ Convolution, Batch Normalization, ReLU, SE 블록, 그리고 Max-pooling으로 구성된다.
- **ASPP (Atrous Spatial Pyramid Pooling)**: 네트워크의 바닥(bottom) 부분에 위치하여 다양한 샘플링 비율(sampling rates)을 가진 Atrous convolution을 통해 고해상도 특징 맵과 넓은 문맥 정보를 동시에 캡처한다.
- **Squeeze-and-Excitation (SE) Block**: 불필요한 정보를 줄이고 중요한 특징 채널에 가중치를 부여하여 특징 맵의 품질을 향상시킨다.
- **Decoder**: $2 \times 2$ Bi-linear up-sampling을 수행하며, Encoder로부터의 Skip connection을 통해 공간 해상도를 유지한다. 특히 NETWORK 2의 Decoder는 두 Encoder 모두로부터 Skip connection을 받아 더욱 정밀한 복원이 가능하다.

### 학습 절차 및 손실 함수

- **손실 함수**: 모든 네트워크에 대해 Binary Cross Entropy (BCE) 손실 함수를 기본으로 사용하였다. 다만, 피부 병변 및 핵 분할 데이터셋의 경우 Dice loss가 더 높은 성능을 보여 이를 적용하였다.
- **최적화**: Nadam 또는 Adam 옵티마이저를 사용하였으며, 학습률은 $1 \times 10^{-5}$로 설정하였다.
- **학습 설정**: 총 300 에포크(epochs) 동안 학습하였으며, Early stopping과 ReduceLROnPlateau 기법을 적용하여 과적합을 방지하고 최적의 학습률을 유지하였다.
- **데이터 증강**: 데이터 부족 문제를 해결하기 위해 Center crop, Random rotation, Transpose, Elastic transform 등을 적용하여 이미지 한 장당 총 26장의 이미지로 확장하였다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - 2015 MICCAI Polyp detection (ETIS-Larib, CVC-ClinicDB)
  - ISIC-2018 Lesion Boundary Segmentation
  - 2018 Data Science Bowl (Nuclei segmentation)
- **평가 지표**: Sørensen-Dice Coefficient (DSC), mean Intersection over Union (mIoU), Precision, Recall.

### 주요 결과

- **Polyp Segmentation**: 2015 MICCAI 데이터셋에서 DoubleU-Net은 $\text{DSC} = 0.7649$, $\text{mIoU} = 0.6255$를 기록하며 U-Net 및 Mask R-CNN 대비 월등한 성능을 보였다. 특히 CVC-ClinicDB에서는 $\text{DSC} = 0.9239$로 기존 최신 모델들을 상회하였다.
- **Skin Lesion & Nuclei**: ISIC-2018에서 $\text{mIoU} = 0.8212$를 달성하여 Multi-ResUNet보다 약 $1.83\%$ 향상된 결과를 보였으며, 핵 분할 작업에서도 $\text{DSC} = 0.9133$으로 UNet++보다 높은 성능을 기록하였다.
- **정성적 결과**: 시각적 분석 결과, $\text{Output}_1$보다 $\text{Output}_2$에서 경계선이 더 명확하고 정밀하게 예측되었으며, 특히 평평한 용종(flat polyps)과 같이 까다로운 샘플에서 그 효과가 두드러졌다.

## 🧠 Insights & Discussion

### 강점 및 유효성

- **일반화 능력**: CVC-ClinicDB에서 학습하고 ETIS-Larib에서 테스트한 교차 데이터셋(cross-dataset) 평가에서 DoubleU-Net이 U-Net보다 훨씬 높은 성능을 보였다. 이는 사전 학습된 VGG-19와 이중 구조가 모델의 일반화 성능을 크게 향상시켰음을 시사한다.
- **전이 학습의 중요성**: ImageNet 사전 학습 가중치를 사용한 모델이 처음부터 학습한 모델보다 의료 영상 도메인에서도 훨씬 뛰어난 성능을 보였다. 이는 의료 데이터의 희소성 문제를 해결하는 강력한 도구임을 확인시켜 준다.

### 한계 및 비판적 해석

- **연산 복잡도**: U-Net에 비해 파라미터 수가 훨씬 많아 학습 시간이 증가한다는 명확한 한계가 있다. 실시간 진단 시스템에 적용하기 위해서는 모델 경량화 연구가 추가적으로 필요할 것으로 보인다.
- **구조적 복잡성**: 두 개의 네트워크를 쌓는 방식이 직관적으로는 성능을 높이지만, 구체적으로 어떤 특징이 두 번째 네트워크에서 '보정'되는지에 대한 수학적/이론적 분석보다는 실험적 결과에 의존하고 있다.

## 📌 TL;DR

본 논문은 사전 학습된 VGG-19, ASPP, SE 블록을 결합한 두 개의 U-Net을 직렬로 연결한 **DoubleU-Net** 아키텍처를 제안한다. 이 모델은 의료 영상의 고질적인 문제인 데이터 부족과 까다로운 대상(평평한 용종 등)의 분할 문제를 효과적으로 해결하며, 네 가지 서로 다른 의료 데이터셋에서 기존 U-Net 및 변형 모델들보다 우수한 성능과 뛰어난 일반화 능력을 입증하였다. 향후 의료 영상 분할을 위한 강력한 베이스라인 모델로 활용될 가능성이 높다.
