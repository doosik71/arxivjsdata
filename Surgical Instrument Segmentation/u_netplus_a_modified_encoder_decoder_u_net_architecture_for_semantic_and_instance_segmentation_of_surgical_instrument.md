# U-NetPlus: A Modified Encoder-Decoder U-Net Architecture for Semantic and Instance Segmentation of Surgical Instrument

S. M. Kamrul Hasan and Cristian A. Linte (2019)

## 🧩 Problem to Solve

본 연구는 로봇 보조 수술(robot-assisted surgery) 환경에서 수술 도구의 정확한 위치 추적 및 식별을 위한 세그멘테이션(segmentation) 문제를 해결하고자 한다. 수술 장면은 조명 변화, 시각적 가려짐(occlusion), 그리고 클래스에 해당하지 않는 다양한 객체들의 존재로 인해 도구의 정밀한 검출과 식별이 매우 어렵다.

기존의 딥러닝 기반 세그멘테이션 모델들은 몇 가지 한계를 가지고 있다. 첫째, 의료 영상 데이터의 제한적인 양으로 인해 모델 학습 시 과적합(overfitting) 문제가 발생하기 쉽고 수렴 속도가 느리다. 둘째, 기존 U-Net 아키텍처의 디코더(decoder) 부분에서 사용되는 Transposed Convolution 연산은 학습 파라미터 수를 증가시켜 학습을 어렵게 만들 뿐만 아니라, '체커보드 아티팩트(checkerboard artifacts)'라고 불리는 불균일한 겹침 현상을 야기하여 결과물에 시각적 노이즈를 생성한다. 따라서 본 논문의 목표는 이러한 아티팩트를 제거하고 학습 효율과 세그멘테이션 정확도를 높인 U-NetPlus 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존 U-Net의 인코더와 디코더 구조를 최적화하여 계산 효율성을 높이고 시각적 왜곡을 최소화하는 것이다.

1. **Pre-trained Encoder 도입**: ImageNet으로 사전 학습된 VGG-11 및 VGG-16 네트워크를 인코더로 사용하여 제한된 의료 데이터셋 환경에서도 빠른 수렴 속도와 높은 정확도를 확보하였다. 특히 각 합성곱 층 이후에 Batch Normalization을 추가하여 내부 공변량 변화(internal covariate shift)를 줄였다.
2. **Nearest-Neighbor (NN) Interpolation 적용**: 디코더의 upsampling 단계에서 기존의 Transposed Convolution을 Nearest-Neighbor 보간법으로 대체하였다. 이는 학습 가능한 파라미터를 줄여 메모리 효율성을 높이는 동시에, Transposed Convolution 특유의 체커보드 아티팩트를 제거하는 효과를 준다.
3. **강력한 데이터 증강(Data Augmentation) 전략**: `albumentations` 라이브러리를 사용하여 Affine 및 Elastic transformation을 적용함으로써 데이터 부족 문제를 해결하고 모델의 일반화 성능을 향상시켰다.

## 📎 Related Works

논문에서는 Fully Convolutional Network (FCN)와 U-Net을 비롯한 기존의 세그멘테이션 연구들을 언급한다. FCN은 CNN의 능력을 활용해 픽셀 단위 예측을 가능하게 했으나, 의료 분야에서는 데이터 부족 문제로 인해 패치 기반 학습이나 전이 학습(transfer learning)이 필수적이었다.

U-Net은 의료 영상 세그멘테이션에서 표준적으로 사용되는 구조이지만, 디코더의 Transposed Convolution 연산이 가중치와 파라미터를 증가시켜 학습 속도를 늦추고, 결과물에 격자 무늬 형태의 아티팩트를 생성한다는 점이 지적된다. TernausNet은 VGG-11 인코더를 사전 학습하여 사용함으로써 성능을 개선한 사례로 제시되었으나, 본 논문은 여기서 더 나아가 디코더 구조의 변경(NN Interpolation)을 통해 아티팩트 문제를 직접적으로 해결하고자 하며, 이를 통해 TernausNet보다 더 정밀한 결과를 얻을 수 있음을 주장한다.

## 🛠️ Methodology

### 전체 시스템 구조

U-NetPlus는 전형적인 오토인코더(auto-encoder) 형태의 인코더-디코더 구조를 따른다. 인코더에서 추출된 고수준 특징 맵을 디코더에서 복원하며, 이때 인코더의 각 단계에서 추출된 특징을 디코더의 대응되는 단계로 전달하는 Skip Connection을 사용하여 마스크의 정렬 정밀도를 높이고 기울기 소실(vanishing gradient) 문제를 완화한다.

### 주요 구성 요소 및 절차

1. **Encoder**: 사전 학습된 VGG-11 또는 VGG-16을 사용한다. $3 \times 3$ 커널의 합성곱 층과 ReLU 활성화 함수, 그리고 Max Pooling 층으로 구성되며, 각 합성곱 층 뒤에 Batch Normalization을 배치하여 최적화 성능을 높였다.
2. **Decoder**: 기존의 Transposed Convolution 대신 Nearest-Neighbor (NN) upsampling을 수행한다. 구체적으로, NN upsampling으로 공간 해상도를 2배 확장한 후, 두 개의 합성곱 층과 ReLU 함수를 통과시켜 특징을 정제한다.
3. **NN Interpolation 공식**: 입력 그리드 $I_i$에 대해 출력 그리드를 생성하는 선형 변환 $\tau_{\theta}(I_i)$는 다음과 같이 정의된다.
    $$\begin{pmatrix} p_{o_i} \\ q_{o_i} \end{pmatrix} = \tau_{\theta}(I_i) = \begin{bmatrix} \theta & 0 \\ 0 & \theta \end{bmatrix} \begin{pmatrix} p_{t_i} \\ q_{t_i} \end{pmatrix}, \theta \ge 1$$
    여기서 $(p_{o_i}, q_{o_i})$는 원래의 입력 좌표이며, $(p_{t_i}, q_{t_i})$는 타겟 좌표, $\theta$는 upsampling 계수이다. 이 방식은 고정된 보간 가중치를 사용하므로 학습이 필요 없으며 메모리 효율적이다.

### 학습 목표 및 손실 함수

본 연구에서는 픽셀 분류 문제로 접근하여 교차 엔트로피(Cross Entropy) 손실 $H$와 Jaccard Index(IoU) $J$를 결합한 일반화된 손실 함수 $L$을 최소화하는 것을 목표로 한다.
교차 엔트로피 손실 $H$는 다음과 같다.
$$H = -\frac{1}{k} \sum_{j=1}^{k} (z_j \log \hat{z}_j + (1-z_j) \log(1-\hat{z}_j))$$
Jaccard Index $J$는 다음과 같다.
$$J = \frac{1}{k} \sum_{j=1}^{k} \frac{z_j \hat{z}_j}{z_j + \hat{z}_j - z_j \hat{z}_j}$$
최종 손실 함수는 다음과 같이 정의된다.
$$L = H - \log J$$

## 📊 Results

### 실험 설정

- **데이터셋**: MICCAI 2017 EndoVis Challenge 데이터셋을 사용하였다. 학습셋은 $8 \times 225$ 프레임, 테스트셋은 $8 \times 75$ 프레임 및 $2 \times 300$ 프레임의 고해상도 스테레오 카메라 영상이다.
- **평가 지표**: IoU(Intersection over Union)와 DICE coefficient를 사용하였다.
- **비교 대상**: 기본 U-Net, U-Net+NN, TernausNet, ToolNetH, ToolNetMS, FCN-8s, CSL 등이 비교 대상으로 사용되었다.

### 정량적 결과

VGG-16 기반의 U-NetPlus 모델이 전반적으로 가장 우수한 성능을 보였다.

- **Binary Segmentation (이진 세그멘테이션)**: DICE score 90.20%를 달성하여 기존 U-Net 대비 6% 이상의 향상을 보였다.
- **Instrument Part (도구 부위 세그멘테이션)**: shaft, wrist, claspers의 3개 클래스에 대해 DICE score 76.26%를 기록하였다.
- **Instrument Type (도구 종류 세그멘테이션)**: 7개 종류의 도구를 구분하는 작업에서는 VGG-11 기반의 U-NetPlus가 46.07% DICE로 VGG-16보다 약간 더 나은 성능을 보였다.

### 정성적 결과 및 분석

- **시각적 품질**: qualitative 결과 분석에서 U-NetPlus는 배경과 도구를 명확히 구분하였으며, 특히 TernausNet이나 기본 U-Net에서 나타나는 불필요한 영역의 검출(false positive)을 현저히 줄였다.
- **Attention Study**: Saliency 기법을 이용한 분석 결과, U-NetPlus는 타 모델에 비해 수술 도구의 wrist 및 claspers 영역에 더 집중적으로 "시선"이 고정되어 있어, 보다 정밀한 국소화(localization)가 가능함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 단순한 아키텍처 변경만으로도 의료 영상 세그멘테이션의 고질적인 문제인 아티팩트를 제거하고 성능을 높일 수 있음을 입증하였다. 특히 Transposed Convolution을 NN Interpolation으로 대체한 것은 파라미터 수를 줄여 모델을 경량화하면서도, 격자 무늬의 노이즈를 제거하여 결과물의 품질을 높이는 실질적인 이득을 가져왔다.

또한, 사전 학습된 VGG 네트워크와 Batch Normalization의 결합이 소규모 데이터셋 환경에서 학습 수렴 속도를 비약적으로 높였다는 점은 전이 학습의 중요성을 다시 한번 강조한다. 다만, 도구 종류(Instrument Type) 세그멘테이션에서 VGG-11이 VGG-16보다 성능이 높게 나타난 점은 흥미로운 부분이며, 이는 더 깊은 네트워크가 반드시 더 높은 정밀도를 보장하지 않는다는 점과 해당 작업의 복잡도에 따른 적절한 모델 용량(capacity) 설정의 필요성을 시사한다.

논문의 한계점으로는 도구 종류 세그멘테이션의 정확도가 부위 세그멘테이션에 비해 현저히 낮다는 점이 있으며, 저자들 또한 이 부분에 대한 추가적인 개선이 필요함을 명시하고 있다.

## 📌 TL;DR

본 논문은 수술 도구 세그멘테이션을 위해 **사전 학습된 VGG 인코더**와 **Nearest-Neighbor 보간법 기반의 디코더**를 결합한 **U-NetPlus** 아키텍처를 제안한다. 이를 통해 기존 Transposed Convolution의 체커보드 아티팩트를 제거하고 학습 속도를 향상시켰으며, MICCAI 2017 데이터셋에서 이진 세그멘테이션(DICE 90.20%) 및 부위 세그멘테이션(DICE 76.26%)에서 SOTA 성능을 달성하였다. 이 연구는 의료 영상 분야에서 모델의 효율성과 시각적 무결성을 동시에 확보하는 실용적인 구조적 개선 방향을 제시한다.
