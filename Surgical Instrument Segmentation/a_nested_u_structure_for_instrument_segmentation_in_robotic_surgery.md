# A Nested U-Structure for Instrument Segmentation in Robotic Surgery

Yanjie Xia, Shaochen Wang, and Zhen Kan (2023)

## 🧩 Problem to Solve

본 논문은 로봇 보조 수술(Robot-assisted surgery) 환경에서 수술 도구의 픽셀 단위 세그멘테이션(Pixel-wise instrument segmentation) 문제를 해결하고자 한다. 수술 장면의 정확한 이해는 외과 의사의 시각적 인지 능력을 향상시키고, 수술 도구의 위치 및 포즈(Pose) 추정을 가능하게 하여 수술의 안전성과 정밀도를 높이는 데 핵심적인 역할을 한다.

그러나 실제 수술 환경은 매우 복잡하며, 도구의 정확한 위치를 찾고 그 형태를 정밀하게 분리해내는 것은 여전히 도전적인 과제이다. 특히, 단순한 이진 세그멘테이션(Binary segmentation)을 넘어 도구의 각 부분(Parts)을 구분하거나 도구의 종류(Type)를 식별하는 정밀한 세그멘테이션이 요구된다. 따라서 본 연구의 목표는 복잡한 수술 환경에서도 로컬(Local) 정보와 글로벌(Global) 정보를 효과적으로 융합하여 높은 정밀도의 세그멘테이션을 수행하는 새로운 네트워크 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다중 스케일 특징 추출과 다단계 딥 피처 통합을 위해 **Two-level Nested U-structure**라는 새로운 아키텍처를 제안한 점이다.

중심적인 설계 아이디어는 단순한 컨볼루션 층의 중첩 대신, 네트워크의 각 층 자체를 다시 U-구조(U-structure)로 설계하여 인코더-디코더 구조 내에 또 다른 인코더-디코더 구조가 중첩되도록 만든 것이다. 이를 통해 네트워크는 다양한 스케일에서 컨텍스트(Context) 정보를 더 풍부하게 포착할 수 있으며, 국소적인 세부 특징과 전체적인 전역 특징을 더 효율적으로 융합할 수 있다. 또한, Dilated Convolution을 도입하여 특성 맵(Feature map)의 해상도를 유지하면서도 수용 영역(Reception field)을 확장함으로써 세그멘테이션의 정확도를 높였다.

## 📎 Related Works

기존의 수술 도구 세그멘테이션 연구는 초기에는 색상 및 텍스처 특징, Haar wavelets, HoG(Histogram of Oriented Gradients)와 같은 전통적인 컴퓨터 비전 기법을 사용하였으며, 이후 Random Forest나 Gaussian Mixture Model과 같은 머신러닝 알고리즘이 적용되었다. 그러나 이러한 방식들은 주로 단순한 이진 세그멘테이션에 국한되었다는 한계가 있다.

최근에는 CNN(Convolutional Neural Networks) 및 U-Net 계열의 딥러닝 모델이 도입되어 의료 영상 분야에서 괄목할 만한 성과를 거두었다. 특히 U-Net은 적은 양의 데이터로도 효과적인 세그멘테이션이 가능함을 보여주었으며, 이후 ResUNet, UNet++, TernausNet, LinkNet 등이 제안되며 성능이 개선되었다. 하지만 이러한 모델들은 여전히 복잡한 수술 환경에서 발생하는 정밀도 저하 문제를 완전히 해결하지 못했으며, 다양한 세그멘테이션 작업(이진, 부분, 종류별 세그멘테이션)에 범용적으로 적용하면서도 효율성을 높이는 방법론에 대한 요구가 계속되고 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

제안된 네트워크는 RGB 이미지를 입력받아 픽셀 단위의 세그멘테이션 마스크를 생성하는 인코더-디코더 구조를 가진다. 전체 구조는 크게 두 단계의 중첩된 U-구조로 이루어져 있으며, 이는 국소 특징과 전역 특징을 동시에 보존하고 융합하도록 설계되었다.

### 주요 구성 요소 및 역할

1. **Encoder Backbone**:
   - 네트워크의 시작 부분은 ResNet18과 UNet++가 결합된 **ResUNetpp** 구조를 사용하여 초기 특징을 추출한다.
   - 이후 4단계의 인코더 층이 이어지며, 각 단계는 **ResUNet34** 구조를 채택하여 공간 해상도를 줄이면서 채널 수를 늘려나간다.
2. **Bottom Module (RSU-4F)**:
   - 인코더의 가장 하단부에는 **RSU-4F** 모듈이 위치한다. 이는 4층 구조의 U-Net 형태로, 입력과 출력의 해상도를 동일하게 유지하면서 깊은 수준의 특징을 추출하여 정보 손실을 방지한다.
3. **Decoder**:
   - 디코더는 4개의 **ResSdual U-blocks (RSU)**로 구성된다. RSU는 5층 U-Net의 변형으로, 각 잔차 블록(Residual block) 내에서 다중 스케일 특징을 추출할 수 있게 한다.
4. **기타 최적화 요소**:
   - **Dilated Convolutions**: 해상도를 유지하며 수용 영역을 넓히기 위해 사용되었다.
   - **Activation & Normalization**: 기존의 ReLU 대신 **LeakyReLU**를 사용하고, BatchNorm2d 대신 **InstanceNorm2d**를 적용하여 성능을 최적화하였다.
   - **Skip-connections**: 인코더의 특징 맵을 디코더로 직접 전달하여 저수준의 세부 정보를 복원한다.

### 손실 함수 (Loss Function)

본 논문은 Jaccard Index(IoU)를 기반으로 한 일반화된 세그멘테이션 손실 함수를 사용한다. 손실 함수는 다음과 같이 정의된다.

$$ \text{Loss} = H - \log \left( \frac{1}{n} \sum_{i=1}^{n} \frac{m_i n_i}{m_i + n_i - m_i n_i} \right) $$

여기서 $m_i$는 정답(Ground Truth) 픽셀 값, $n_i$는 예측된 출력 픽셀 값을 의미한다. $H$는 작업의 성격에 따라 다른 함수로 정의된다.

- **이진 세그멘테이션**: $H$는 $\text{BCEWithLogitsLoss}$를 사용한다.
- **다중 클래스 세그멘테이션**: $H$는 $\text{Cross-Entropy Loss}$를 사용한다.

이 함수는 정답과 예측값 사이의 겹침 정도(Overlap rate)를 최대화함으로써 손실을 최소화하도록 유도한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MICCAI EndoVis Challenge 2017 데이터셋을 사용하였다. 8개의 수술 비디오에서 추출된 이미지와 수동으로 라벨링된 정답 데이터를 활용하였다.
- **평가 지표**: $\text{Intersection Over Union (IoU)}$와 $\text{Dice coefficient (Dice)}$를 사용하여 정량적으로 평가하였다.
- **비교 대상**: U-Net, TernausNet, LinkNet-34, PlainNet 등 최신 SOTA 모델들과 비교하였다.
- **학습 환경**: AdamW 옵티마이저, 학습률 $1\times 10^{-4}$, 4-fold 교차 검증을 수행하였으며, NVIDIA GTX 3090 GPU 2대를 사용하였다.

### 주요 결과

실험은 세 가지 태스크(이진 세그멘테이션, 부분 세그멘테이션, 종류 세그멘테이션)로 나누어 진행되었다.

1. **Binary Segmentation**: 제안 방법이 $\text{IoU } 82.94\%$, $\text{Dice } 89.42\%$로 가장 우수한 성능을 보였다. 특히 U-Net 대비 IoU 기준 7.5 포인트 향상되었다.
2. **Parts Segmentation**: TernausNet이나 LinkNet-34보다는 약간 낮았으나, U-Net보다는 훨씬 높은 성능을 보였다.
3. **Type Segmentation**: $\text{IoU } 41.72\%$, $\text{Dice } 48.22\%$를 기록하며 다른 모든 모델을 압도하는 성능을 보여주었다.

또한, 10개의 서로 다른 테스트 비디오 시퀀스에 대해 mIoU를 측정한 결과, 제안 모델이 10개 데이터셋 중 6개에서 가장 높은 점수를 획득하여 높은 일반화 성능을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 모델은 특히 이진 세그멘테이션과 도구 종류 세그멘테이션에서 매우 강력한 성능을 보여주었다. 이는 Nested U-structure가 다중 스케일의 특징을 효과적으로 캡처하여, 도구의 전반적인 외형과 세부적인 특징을 동시에 잘 파악했기 때문으로 분석된다.

다양한 클래스의 도구 세그멘테이션 성능이 상대적으로 낮게 나타난 이유는 일부 도구 클래스의 학습 데이터 수가 매우 적었기 때문(Class Imbalance)으로 보인다. 저자들은 데이터셋의 크기를 확대한다면 성능을 더 끌어올릴 수 있을 것이라고 언급하였다.

또한, 정성적 분석 결과 일부 이미지(Case 3, 5)에서 도구의 부분이 정확히 분리되지 않는 현상이 발견되었는데, 이는 수술 도구 표면에서 발생하는 빛 반사(Light reflection)가 모델의 판단을 방해했기 때문으로 추측된다. 이는 향후 조명 조건의 변화나 반사광에 강건한 모델 설계가 필요함을 시사한다.

## 📌 TL;DR

본 연구는 로봇 수술 도구의 정밀한 세그멘테이션을 위해 인코더-디코더 내부에 다시 U-구조를 중첩시킨 **Two-level Nested U-structure** 모델을 제안하였다. 이 구조는 다중 스케일 특징 추출과 Dilated Convolution을 통해 로컬 및 글로벌 정보를 효과적으로 융합하며, EndoVis 2017 데이터셋의 이진 및 종류별 세그멘테이션 작업에서 SOTA 성능을 달성하였다. 이 연구는 향후 수술 내비게이션 및 자동화된 수술 시스템의 시각적 인지 능력을 향상시키는 데 중요한 기여를 할 가능성이 크다.
