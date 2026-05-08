# Deep Semantic Segmentation of Natural and Medical Images: A Review

Saeid Asgari Taghanaki, Kumar Abhishek, Joseph Paul Cohen, Julien Cohen-Adad, Ghassan Hamarneh (2020)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 핵심 과제 중 하나인 Semantic Image Segmentation(의미론적 이미지 분할) 분야의 최신 딥러닝 연구 동향을 분석하고 정리하는 것을 목표로 한다. Semantic Segmentation은 이미지의 모든 픽셀을 특정 클래스에 할당하여 이미지의 전역적인 문맥을 이해하는 작업이다.

이 기술은 일반적인 자연 이미지(Natural Images)의 장면 이해(Scene Understanding)뿐만 아니라, 의료 영상 분석(Medical Image Analysis) 분야에서 영상 유도 중재술(Image-guided interventions), 방사선 치료, 그리고 정밀한 방사선 진단 등의 핵심적인 단계로 활용된다. 특히 의료 영상의 경우, 데이터의 부족, 클래스 불균형(Class Imbalance), 그리고 다양한 영상 모달리티(MRI, CT, X-ray 등)로 인해 발생하는 기술적 난제들을 해결하는 것이 매우 중요하다.

## ✨ Key Contributions

본 논문의 가장 중심적인 기여는 딥러닝 기반의 이미지 분할 솔루션을 체계적으로 분류하여 광범위한 리뷰를 제공했다는 점이다. 저자들은 기존 연구들을 다음과 같은 6가지 주요 그룹으로 범주화하였다.

1. **Deep Architectural Improvements**: 네트워크 구조의 최적화 및 새로운 레이어 설계.
2. **Optimization Function-based Improvements**: 손실 함수(Loss Function) 설계를 통한 성능 향상.
3. **Data Synthesis-based Improvements**: GAN 등을 이용한 데이터 증강 및 합성.
4. **Weakly Supervised Methods**: 적은 양의 레이블이나 낮은 품질의 레이블을 활용한 학습.
5. **Sequenced Models**: RNN/LSTM 등을 이용한 시퀀스 데이터 처리.
6. **Multi-task Models**: 분할과 분류, 검출 등 여러 작업을 동시에 수행하는 모델.

또한, 다양한 손실 함수들이 픽셀 수준의 오차(False Positive/Negative)에 따라 어떻게 반응하는지를 분석하여, 특히 작은 객체를 분할할 때의 안정성 문제를 논의하였다.

## 📎 Related Works

기존의 리뷰 논문들은 주로 특정 영역에 국한된 경향이 있었다. 예를 들어, Guo et al. (2018)은 region-based, FCN-based, weakly supervised 방법론으로 나누어 분석하였고, Hu et al. (2018b)은 RGB-D 데이터셋에 집중하였다. 의료 영상 분야의 경우, Goceri and Goceri (2017)나 Hesamian et al. (2019) 등이 네트워크 구조와 학습 기법을 다루었으나, 자연 이미지와 의료 영상을 동시에 포괄하며 통합적인 프레임워크를 제시한 연구는 부족했다.

본 논문은 이러한 한계를 극복하기 위해 2D 및 3D(Volumetric) 영상을 모두 포함하며, 특히 모델 아키텍처뿐만 아니라 데이터 합성, 최적화 함수, 약지도 학습 등 학습 파이프라인 전반에 걸친 종합적인 분석을 제공함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하는 대신, 기존의 방대한 연구들을 분석하는 방법론을 취한다. 주요 분석 내용은 다음과 같다.

### 1. 네트워크 아키텍처의 진화 (Model Architecture)

- **FCN (Fully Convolutional Networks)**: 모든 완전 연결층을 컨볼루션층으로 대체하여 픽셀 단위의 예측을 가능하게 하였다. 얕은 층의 특징 맵과 깊은 층의 특징 맵을 융합하여 공간 정보를 보존한다.
- **Encoder-Decoder 구조**: SegNet, U-Net 등이 대표적이다. Encoder가 특징을 압축하고 Decoder가 이를 다시 원래 해상도로 복원한다. 특히 U-Net은 Skip Connection을 통해 Encoder의 세밀한 정보를 Decoder에 전달함으로써 정밀한 localization을 가능하게 한다.
- **고급 기법**: Atrous (Dilated) Convolution을 사용하여 수용 영역(Receptive Field)을 넓히면서도 해상도 손실을 줄이는 DeepLab 시리즈와, 다양한 스케일의 문맥 정보를 캡처하는 Pyramid Scene Parsing Network(PSPNet) 등이 분석되었다.

### 2. 최적화 함수 및 손실 함수 (Optimization Functions)

모델의 학습 목표가 되는 손실 함수를 분석하며, 특히 클래스 불균형 문제 해결에 집중한다.

- **Cross Entropy (CE)**: 픽셀별로 예측값과 정답을 비교한다.
  $$CE = -\sum_{classes} p \log \hat{p}$$
- **Dice Loss (DL)**: 예측 결과와 정답 간의 겹침(Overlap) 정도를 측정하며, F1 Score와 유사하다.
  $$DL(p, \hat{p}) = \frac{2 \langle p, \hat{p} \rangle}{\|p\|_1 + \|\hat{p}\|_1}$$
- **Focal Loss (FL)**: 쉬운 예제의 가중치를 낮추고 어려운 예제에 집중하게 한다.
  $$FL(p, \hat{p}) = -(\alpha(1-\hat{p})^\gamma p \log \hat{p} + (1-\alpha)\hat{p}^\gamma (1-p) \log (1-\hat{p}))$$
- **Combo Loss**: 저자들은 CE의 안정성과 Dice의 불균형 해결 능력을 결합한 Combo Loss를 제안하며, 가중치 $\alpha$와 $\beta$를 통해 False Positive(FP)와 False Negative(FN)에 대한 페널티를 조절한다.

### 3. 데이터 합성 및 약지도 학습 (Synthesis & Weakly Supervised)

- **GAN 기반 합성**: 데이터가 부족한 의료 영상 분야에서 GAN을 이용해 가상의 영상을 생성하여 학습 데이터셋을 확장하는 기법을 다룬다.
- **Weakly Supervised**: 픽셀 단위의 정교한 레이블 대신 이미지 레벨 레이블이나 Bounding Box만을 사용하여 모델을 학습시키는 전략을 분석한다.

### 4. 멀티태스크 모델 (Multi-Task Models)

- **Mask R-CNN**: 객체 검출(Detection)과 분할(Segmentation)을 동시에 수행하는 구조로, 다양한 의료 영상 분석(세포 핵 분할 등)에 적용된 사례를 설명한다.

## 📊 Results

### 1. 정량적 결과 분석 (Natural Images)

PASCAL VOC 2012 데이터셋을 기준으로 분석한 결과, 초기 모델인 FCN (mean IoU 62.2%)에서 최신 모델인 DeepLabV3+ (mean IoU 89.0%)에 이르기까지 약 27%의 성능 향상이 있었음을 확인하였다. 이는 정교한 Decoder, Dilated Convolution, Feature Pyramid Pooling의 도입 결과이다.

### 2. 의료 영상 분석의 특이점

의료 영상의 경우, 모달리티가 매우 다양(13가지 이상)하고 데이터셋의 크기가 극단적으로 작기 때문에 자연 이미지처럼 단일 벤치마크 지표로 성능을 일반화하기 어렵다는 점을 발견하였다. 특히 MRI, PET와 같이 획득 비용이 높은 데이터셋일수록 샘플 수가 적은 경향이 뚜렷하다.

### 3. 손실 함수 행동 분석

실험을 통해 Overlap 기반 손실 함수(Dice 등)는 큰 객체 분할에는 효과적이지만, 작은 객체를 분할할 때는 예측값의 작은 변화에도 손실 값이 크게 요동쳐 최적화가 불안정해지는 특성이 있음을 보였다. 반면, CE 기반의 함수는 상대적으로 매끄러운(Smooth) 최적화 곡선을 보여주었다.

## 🧠 Insights & Discussion

본 논문은 Semantic Segmentation의 현재 기술 수준을 집대성함과 동시에 다음과 같은 비판적 통찰을 제시한다.

- **손실 함수의 트레이드-오프**: 단순히 Dice Loss를 사용하는 것보다, CE를 기본으로 하고 Dice를 정규화 항(Regularizer)으로 사용하는 것이 학습의 안정성 측면에서 유리하다.
- **의료 영상의 데이터 갈증**: 딥러닝의 성능은 데이터 양에 의존하지만, 의료 분야는 법적/윤리적 문제로 데이터 수집이 어렵다. 따라서 Physics-based imaging simulators를 통한 합성 데이터 생성이나, 전이 학습(Transfer Learning)의 위험성 연구가 필수적이다.
- **아키텍처의 한계**: 현재의 Encoder-Decoder 구조는 유용하지만 메모리 사용량이 많다. Skip Connection을 통해 전달되는 정보 중 불필요한 정보를 필터링하는 최적화 연구가 필요하다.

## 📌 TL;DR

본 논문은 자연 및 의료 영상의 Semantic Segmentation 기술을 **아키텍처, 최적화 함수, 데이터 합성, 약지도 학습, 시퀀스 모델, 멀티태스크**의 6가지 관점에서 체계적으로 분석한 리뷰 논문이다. 특히 FCN에서 DeepLabV3+로 이어지는 구조적 진화와, 클래스 불균형 해결을 위한 다양한 Loss 함수들의 특성을 심도 있게 다루었다. 이 연구는 향후 의료 영상 분할에서 데이터 부족 문제를 해결하기 위한 합성 데이터 생성 및 최적화된 손실 함수 설계의 방향성을 제시한다는 점에서 중요한 가치를 지닌다.
