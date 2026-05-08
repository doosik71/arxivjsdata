# Deep Learning in Medical Ultrasound Image Segmentation: a Review

Ziyang Wang (2021)

## 🧩 Problem to Solve

본 논문은 의료 초음파 이미지 분할(Medical Ultrasound Image Segmentation) 분야에 적용된 딥러닝 기술의 현황을 분석하고 체계적으로 정리하는 것을 목표로 한다.

초음파 영상은 비침습적이고 비용이 저렴하며 실시간 진단이 가능하다는 장점이 있어 임상 진단, 조직의 3D 재구성, 이미지 가이드 중재술 등에 필수적이다. 그러나 초음파 영상은 숙련된 작업자의 필요성, 조직과 가스 간의 구분 어려움, 제한된 시야(Field of View) 등의 고유한 한계가 존재하며, 이는 이미지 처리 알고리즘 설계에 있어 큰 도전 과제가 된다. 따라서 본 논문은 이러한 문제를 해결하기 위해 제안된 다양한 딥러닝 기반 분할 방법론을 분류하고, 평가 지표와 데이터셋을 정리하여 향후 연구 방향을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 주요 기여 사항은 다음과 같다.

- **포괄적인 방법론 분류**: 의료 초음파 이미지 분할을 위한 딥러닝 접근 방식을 아키텍처와 학습 방법에 따라 Fully Convolutional Neural Networks (FCN), Encoder-Decoder Neural Networks, Recurrent Neural Networks (RNN), Generative Adversarial Networks (GAN), Weakly Supervised Learning (WSL), Deep Reinforcement Learning (DRL)의 6가지 그룹으로 체계화하였다.
- **대표 알고리즘 분석**: 각 그룹별로 최신 대표 알고리즘들을 선정하여 그 구조와 특성을 상세히 분석하고 요약하였다.
- **평가 프레임워크 정리**: 초음파 이미지 분할 성능을 측정하기 위해 사용되는 공통 평가 지표(Evaluation Methods)와 공개 데이터셋(Datasets)을 체계적으로 정리하여 비교 가능하게 하였다.
- **미래 연구 방향 제시**: 현재 기술의 한계점(데이터 라벨링 비용, 2D 모델 의존성, 하드웨어 최적화 문제 등)을 지적하고 이를 해결하기 위한 잠재적 연구 방향을 논의하였다.

## 📎 Related Works

논문에서는 딥러닝 이전의 전통적인 비-머신러닝 접근 방식(Non-machine learning approaches)을 언급한다. 대표적으로 Deformable models, Watershed, Region grow, Graph-based methods 등이 있으며, 이러한 방법들은 일반적인 초음파 분할 작업의 기초가 되었다.

딥러닝의 도입 이후에는 단순한 분류 네트워크에서 시작하여 픽셀 단위의 정밀한 예측이 가능한 세그멘테이션 네트워크로 발전하였다. 기존 연구들이 주로 단일 모델의 성능 향상에 집중했다면, 본 논문은 다양한 아키텍처의 특성을 초음파 영상의 특성(예: 3D 볼륨 데이터, 시퀀스 데이터, 노이즈 등)과 연결하여 종합적으로 검토한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

본 논문은 리뷰 논문으로서, 초음파 이미지 분할에 사용되는 딥러닝 방법론을 다음과 같이 6가지 핵심 아키텍처로 분류하여 설명한다.

### 1. Fully Convolutional Neural Networks (FCN)

FCN은 완전히 연결된 층(Fully Connected layers)을 컨볼루션 층으로 대체하여 입력 이미지의 크기에 상관없이 픽셀 대 픽셀(pixels-to-pixels)의 지도 학습을 수행하는 구조이다.

- **핵심 기법**: Skip connection과 bilinear interpolation을 통해 분류 네트워크를 밀집 예측(dense prediction)으로 확장하여 정밀한 픽셀 수준의 분할을 구현한다.
- **특이 사례**: DF-FCN은 3D 초음파 볼륨을 2D 이미지로 나누어 처리한 후, 세 가지 방향의 특성 맵을 융합(direction-fused)하여 슬라이스 간 문맥 정보를 강화한다.

### 2. Encoder-Decoder Neural Networks

Pooling 연산 과정에서 손실되는 픽셀의 위치 정보를 복원하기 위해 대칭적인 구조를 사용하는 방식이다.

- **구조**: Encoder가 특징을 추출하고, Decoder가 Deconvolution(또는 Transpose Convolution)과 Unpooling을 통해 공간 차원을 복원한다.
- **대표 모델**: U-Net은 Encoder의 특징 맵을 Decoder에 직접 연결하는 skip connection을 통해 국소화(localization) 성능을 높였으며, V-Net은 이를 3D 볼륨 분할로 확장하였다.

### 3. Recurrent Neural Networks (RNN)

연속적인 2D 스캔 슬라이스와 같은 시퀀스 데이터에서 역사적 문맥 정보와 전역 특징을 추출하기 위해 사용된다.

- **핵심 기법**: LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Unit)를 통해 장기 기억을 모델링하며, 인접 슬라이스의 정보를 활용해 현재 슬라이스의 분할 성능을 향상시킨다.

### 4. Generative Adversarial Networks (GAN)

생성자(Generator)와 판별자(Discriminator)가 서로 대립하며 학습하는 적대적 학습 방식을 통해 정밀한 분할 맵을 생성한다.

- **목표**: 생성자는 실제 정답(Ground Truth)과 구별할 수 없는 분할 결과를 만들어내려 하며, 이를 통해 고차원의 불일치성을 수정하고 장거리 공간적 연속성을 확보한다.

### 5. Weakly Supervised Learning (WSL)

픽셀 단위의 정밀한 라벨링 비용을 줄이기 위해 coarse한 주석(annotation)만을 사용하는 학습 방법이다.

- **접근 방식**: Deconvolutional layers를 통해 가짜 양성(false positives)을 줄이거나, 확률적 추론(stochastic inference)을 통해 정밀한 경계를 예측하는 FickleNet과 같은 방식이 제안되었다.

### 6. Deep Reinforcement Learning (DRL)

신경망을 가치 함수 추정기로 사용하여 에이전트가 최적의 분할 전략을 학습하게 하는 방식이다.

- **절차**: 상태(State) 정의 $\rightarrow$ 행동(Action) 수행 $\rightarrow$ 보상(Reward) 계산의 루프를 통해 최적의 분할 파라미터나 임계값을 찾는다.

## 📊 Results

### 평가 지표 (Evaluation Metrics)

논문은 분할 성능을 정량적으로 측정하기 위한 다양한 수학적 지표를 정의한다.

- **Pixel Accuracy (PA)**: 전체 픽셀 중 정확하게 분류된 픽셀의 비율이다.
$$PA = \frac{\sum_{i=0}^{k} P_{ii}}{\sum_{i=0}^{k} \sum_{j=0}^{k} P_{ij}}$$
- **Dice Coefficient (DSC)**: 두 경계 간의 유사도를 측정하며, 의료 영상 분할에서 가장 널리 쓰인다.
$$DSC = \frac{2 * \sum_{i=0}^{k} P_{ii}}{2 * \sum_{i=0}^{k} P_{ii} + \sum_{i=0}^{k} \sum_{j=0}^{k} (P_{ij} + P_{ji})}$$
- **Intersection Over Union (IOU)**: 예측 영역과 정답 영역의 합집합 대비 교집합의 비율이다.
$$IOU = \frac{\sum_{i=0}^{k} P_{ii}}{\sum_{i=0}^{k} \sum_{j=0}^{k} P_{ij} - \sum_{j=0}^{k} P_{jj}}$$

### 실험 결과 요약

- **데이터셋**: Ultrasound Nerve Segmentation Challenge, Breast Ultrasound Teaching File 등 다양한 공개 데이터셋이 활용되고 있다.
- **성능 분석**: Table 2에 따르면, CasFCN은 태아 머리 및 복부 이미지 분할에서 $DSC: 0.9843$이라는 높은 성능을 보였으며, NAS-Unet은 신경 초음파 분할에서 $MIOU: 0.992$를 기록하였다. 전반적으로 Encoder-Decoder 구조(U-Net 계열)가 높은 빈도로 사용되며 우수한 성능을 나타낸다.

## 🧠 Insights & Discussion

### 강점 및 성과

지난 5년간 딥러닝은 초음파 이미지 분할에서 SOTA(State-of-the-art) 성능을 입증하였다. 특히 2D FCN과 Encoder-Decoder 구조는 매우 안정적인 성능을 보여주며 표준적인 접근 방식으로 자리 잡았다.

### 한계 및 비판적 해석

- **데이터 의존성 및 라벨링 비용**: 대부분의 모델이 정밀한 픽셀 단위 라벨링을 요구하지만, 실제 임상에서는 bounding box나 단순 선으로 라벨링하는 경우가 많아 데이터 부족 문제가 심각하다.
- **차원 확장 부족**: 모델의 80%가 2D 기반이며, 초음파 영상의 연속적인 특성이나 스캐너의 위치 정보를 충분히 활용하는 3D 모델이나 RNN 기반 연구가 여전히 부족하다.
- **효율성 간과**: 현재의 연구들은 주로 정확도(Accuracy)에만 집중하고 있으며, 휴대용 초음파 기기 등 실제 환경에 배포하기 위한 모델의 경량화(Lightweight)나 추론 속도, 메모리 비용에 대한 고려가 부족하다.
- **영상의 이질성(Heterogeneity)**: 장기의 위치, 깊이, 주변 조직의 경도 등에 따라 영상의 패턴이 크게 달라지는 특성이 있어 일반화 성능을 확보하는 것이 매우 어렵다.

### 향후 연구 방향

- **약지도 학습(WSL) 및 GAN의 확대**: 라벨링 비용 절감을 위한 WSL과 데이터 시뮬레이션을 위한 GAN의 적용 확대가 필요하다.
- **교차 모달 전이 학습(Cross-modal Transfer Learning)**: MRI나 CT와 같은 텍스처가 풍부한 영상의 지식을 초음파 영상으로 전이하는 연구가 유망할 것으로 보인다.
- **임상적 가치 확대**: 단순 분할을 넘어, 의사가 발견하지 못한 미세 특징을 학습하여 질병을 조기에 예측하는 모델로의 발전이 필요하다.

## 📌 TL;DR

본 논문은 의료 초음파 이미지 분할을 위한 딥러닝 방법론을 **FCN, Encoder-Decoder, RNN, GAN, WSL, DRL의 6가지 범주**로 체계적으로 분류하고 분석한 리뷰 논문이다. 딥러닝이 기존의 정밀도를 비약적으로 높였으나, 여전히 **고비용의 라벨링 데이터 의존성**과 **2D 모델 중심의 설계**, **실시간 배포를 위한 경량화 부족**이라는 과제가 남아 있음을 시사한다. 향후 3D 컨텍스트 활용, 약지도 학습, 그리고 타 모달리티와의 전이 학습이 초음파 분할 연구의 핵심 방향이 될 것으로 전망된다.
