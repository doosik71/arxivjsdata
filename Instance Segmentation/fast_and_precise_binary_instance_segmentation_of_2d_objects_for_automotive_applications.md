# Fast and Precise Binary Instance Segmentation of 2D Objects for Automotive Applications

Ganganna Ravindra, Darshan Dinges, Laslo Ayoub, Al-Hamadi, Vasili Baranau (2022)

## 🧩 Problem to Solve

본 논문은 자율 주행 및 도로 장면 이해를 위한 데이터셋 구축 과정에서 발생하는 수동 라벨링의 비효율성 문제를 해결하고자 한다. 특히, 객체의 정밀한 경계를 나타내는 폴리곤(Polygon)을 수동으로 그리는 작업은 막대한 시간과 인력이 소모되며, 작업자의 실수 가능성이 높다는 문제점이 있다.

연구의 목표는 사용자가 단순히 객체 주변에 Bounding Box를 그리거나 몇 개의 점을 지정하는 것만으로도 정밀한 폴리곤을 자동으로 생성할 수 있는 빠른 Binary Instance Segmentation 시스템을 개발하는 것이다. 특히, 실제 현장의 라벨러들이 GPU 서버 없이 일반적인 CPU 환경에서 작업한다는 점을 고려하여, CPU 상에서 추론 시간(Inference Time)이 200ms 이하로 작동하는 실시간성을 확보하는 것을 핵심 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 기존의 Encoder-Decoder 네트워크 구조에 객체의 외곽 경계 정보인 **Extreme Points**를 추가적인 입력 채널로 제공하는 것이다.

단순히 RGB 이미지 정보만을 사용하는 것이 아니라, 객체의 가장 왼쪽, 오른쪽, 위, 아래 끝점에 해당하는 픽셀 좌표(Extreme Points)를 이진 마스크(Binary Mask) 형태로 생성하여 총 4채널($RGB + Extreme Points$) 입력을 구성한다. 이를 통해 네트워크가 객체의 대략적인 위치와 범위에 대한 명시적인 가이드를 얻게 함으로써, 복잡한 아키텍처 수정 없이도 세그멘테이션의 정밀도(IoU)를 유의미하게 향상시켰다.

## 📎 Related Works

본 논문은 객체 탐지(Object Detection)와 시맨틱 세그멘테이션(Semantic Segmentation)을 동시에 수행하는 Instance Segmentation 기술을 기반으로 한다. 특히, 특정 객체 하나에 집중하여 배경과 전경을 분리하는 Binary Segmentation 방식에 주목한다.

기존 연구에서는 U-Net과 같은 Encoder-Decoder 구조가 널리 사용되었으며, 본 논문은 이러한 구조의 성능을 높이기 위해 Residual Blocks, Dense Blocks, Contextual Convolutions, U-Net++ 등 최신 아키텍처 변형들을 검토하였다. 또한, Extreme Points를 활용하여 객체 세그멘테이션 성능을 높인 [Man18]의 연구를 참고하여 이를 본 시스템에 접목하였다. 기존 방식들이 주로 모델의 깊이나 복잡도를 늘려 성능을 높이려 했다면, 본 논문은 입력 데이터의 질(Extra Information)을 개선하는 방향으로 차별점을 두었다.

## 🛠️ Methodology

### 1. 시스템 구조 및 아키텍처

전체적인 파이프라인은 **U-Net** 아키텍처를 기반으로 하며, 입력부에서 RGB 3채널에 Extreme Points 채널 1개를 추가하여 총 4채널을 입력으로 받는다. Extreme Points 채널은 각 끝점에 원을 그려 표시한 이진 마스크 형태이다.

네트워크의 세부 설정은 다음과 같다:

- **Encoder**: Max Pooling을 통해 해상도를 $128 \times 128 \rightarrow 64 \times 64 \rightarrow 32 \times 32 \rightarrow 16 \times 16 \rightarrow 8 \times 8$ 순으로 감소시킨다. 필터 깊이는 $f=16$을 기준으로 $f, 2f, 4f, 8f, 16f$로 점진적으로 증가한다.
- **Decoder**: Upsampling을 통해 해상도를 다시 $128 \times 128$까지 복원하며, 필터 깊이는 역순으로 감소한다.
- **Skip Connections**: Encoder의 고해상도 특징 맵을 Decoder에 결합하여 정밀한 복원을 가능하게 한다.

### 2. 학습 절차 및 손실 함수

Cityscapes 데이터셋에서 가장 빈번하게 등장하는 9개 클래스(car, traffic sign, bicycle, person, rider, motorcycle, traffic light, truck, bus)를 대상으로 학습을 진행하였다. 모든 인스턴스는 $128 \times 128$ 크기로 리사이징되었다.

본 논문에서는 Differentiable IoU loss나 Binary Cross Entropy 대신, 경계선 오차를 더 잘 반영하는 **Custom Loss Function**을 제안하였다. 이 손실 함수는 예측된 경계와 실제 경계 사이의 평균 거리 오차($\Delta d$)를 근사화한다.

평균 거리 오차 $\Delta d$는 다음과 같이 정의된다:
$$\Delta d = \frac{A_u - A_i}{\sqrt{A_g}}$$
여기서 각 변수의 의미는 다음과 같다:

- $A_u$: 예측 마스크와 정답 마스크의 합집합 면적(Area of Union)
- $A_i$: 예측 마스크와 정답 마스크의 교집합 면적(Area of Intersection)
- $A_g$: 정답 경계의 면적(Area of the groundtruth boundary)

이 식은 실제의 선적분 형태인 $\int_{border} EDT(s)ds/L$를 효율적으로 근사한 것으로, 예측 경계가 실제 경계에서 벗어난 정도를 정량화한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Cityscapes (학습 83,403개, 검증 15,926개 인스턴스)
- **지표**: aIoU(Average IoU), mIoU(Mean IoU), iIoU(Instance IoU), Border Error(per-pixel distance)
- **하드웨어**: 학습은 Nvidia GTX 1080 Ti에서 진행, 추론 성능 측정은 Intel Core i5-6300U CPU에서 수행

### 2. 정량적 결과

Extreme Points를 추가한 모델이 기본 U-Net 및 기타 아키텍처 변형 모델보다 모든 지표에서 우수한 성능을 보였다.

- **IoU 성능**: Extreme Points 모델은 $\text{aIoU}=89.16\%$, $\text{mIoU}=90.65\%$, $\text{iIoU}=87.25\%$를 기록하여 기본 U-Net 대비 각각 3.7, 4.5, 4.5 포인트 향상되었다.
- **추론 시간**: Extreme Points 모델의 CPU 추론 시간은 약 $140\text{ms}$로, 목표치인 $200\text{ms}$를 만족하며 기본 U-Net과 동일한 속도를 유지하였다.
- **클래스별 성능**: 특히 Person(84.92%)과 Bicycle(81.16%) 클래스에서 다른 모델들보다 뚜렷한 성능 향상을 보였다.

### 3. 정성적 결과

Border Error의 분포를 분석한 결과, Extreme Points 모델은 0px 오차 구간에서 가장 높은 확률 밀도를 보였다. 이는 예측 경계가 실제 정답 경계와 거의 일치하는 경우가 훨씬 많음을 의미한다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 시사점은 **CPU 기반의 실시간 제약 조건 하에서는 모델 아키텍처의 복잡도를 높이는 것보다, 네트워크가 활용할 수 있는 입력 정보(Context)를 최적화하는 것이 훨씬 효율적**이라는 점이다. Residual blocks나 Dense blocks와 같은 구조적 변경은 추론 시간만 증가시킬 뿐 성능 향상은 미미했으나, Extreme Points라는 추가 정보를 제공하는 것은 연산 비용 증가 없이 비약적인 정확도 향상을 가져왔다.

다만, 본 논문은 $128 \times 128$이라는 비교적 작은 입력 해상도에서 실험이 진행되었다는 한계가 있다. 고해상도 이미지에서도 동일한 성능 향상 폭이 유지될지는 명시되지 않았다. 또한, Extreme Points를 사용자가 직접 찍어줘야 한다는 전제가 있으나, 저자들은 이 작업이 Bounding Box를 그리는 것만큼 빠르다고 주장하며 실용성을 강조하였다.

향후 발전 방향으로 생성적 적대 신경망(GAN)의 도입을 제안하였다. Generator가 세그멘테이션 마스크를 생성하고 Discriminator가 실제 마스크와의 유사성을 판별하게 함으로써, 손실 함수를 직접 설계하는 대신 네트워크가 스스로 평가 지표를 학습하게 하는 방식이다.

## 📌 TL;DR

본 논문은 자동차 산업의 데이터 라벨링 효율을 높이기 위해, CPU 환경에서 빠르게 동작하는 Binary Instance Segmentation 모델을 제안한다. 핵심은 U-Net 입력에 **Extreme Points(객체의 최외곽 4점)** 정보를 추가 채널로 제공하는 것이며, 이를 통해 아키텍처를 복잡하게 만들지 않고도 $\text{aIoU}$를 $89.16\%$까지 끌어올리고 추론 시간 $140\text{ms}$를 달성하였다. 이 연구는 제한된 컴퓨팅 자원 환경에서 입력 데이터의 가이드를 통해 딥러닝 모델의 성능을 극대화하는 실용적인 접근법을 제시한다.
