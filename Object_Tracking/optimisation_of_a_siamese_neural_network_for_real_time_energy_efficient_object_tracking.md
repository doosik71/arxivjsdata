# Optimisation of a Siamese Neural Network for Real-Time Energy Efficient Object Tracking

Dominika Przewlocka, Mateusz Wasala, Hubert Szolc, Krzysztof Blachut, and Tomasz Kryjak (2020)

## 🧩 Problem to Solve

본 연구는 임베디드 비전 시스템에서 실시간 객체 추적(Object Tracking)을 수행하기 위해 Siamese Neural Network의 계산 복잡도와 에너지 소비를 최적화하는 문제를 해결하고자 한다. 일반적으로 최신 객체 추적 알고리즘들은 높은 정확도를 제공하지만, 이를 실시간으로 구동하기 위해서는 전력 소모가 큰 고성능 GPU가 필수적이다. 그러나 자율 주행 차량(AV), 첨단 운전자 지원 시스템(ADAS), 지능형 영상 감시 시스템(AVSS)과 같은 엣지 디바이스 환경에서는 제한된 전력 예산 내에서 고해상도(FHD 및 UHD) 비디오 스트림을 실시간으로 처리해야 하는 제약이 존재한다. 따라서 본 논문의 목표는 모델의 메모리 점유율과 계산 복잡도를 획기적으로 낮추어 FPGA(Field Programmable Gate Array)와 같은 저전력 임베디드 장치에서 실시간 추적이 가능하도록 최적화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Siamese Neural Network를 기반으로 한 객체 추적기의 가중치와 활성화 함수에 다양한 수준의 양자화(Quantization)를 적용하여 임베디드 환경에 최적화한 연구 결과를 제시한 점이다. 특히 단순한 정수 양자화(Integer Quantization)를 넘어 4비트, Ternary(3진법), Binary(이진법) 필터까지 적용 범위를 넓혀 그 영향을 분석하였다. 저자들은 양자화가 메모리 및 계산 복잡도를 줄일 뿐만 아니라, 네트워크의 과적합(Overfitting)을 방지하여 오히려 추적 성능을 향상시킬 수 있다는 직관적인 결과를 도출하였다.

## 📎 Related Works

기존의 Siamese 기반 추적기인 GOTURN, SiamFC, DSiam, CFNet 등은 높은 정확도를 보이지만 대부분 Nvidia Titan X와 같은 고성능 GPU에서 가속화되었으며, CPU 환경에서는 실시간 처리가 불가능한 수준의 계산 복잡도를 가진다. 최근 일부 연구에서 경량화된 Siamese 네트워크를 제안하거나 FPGA 구현을 시도한 사례(MiniTracker 등)가 있었으나, 본 논문은 다양한 수준의 양자화 시나리오(16비트부터 Binary까지)를 체계적으로 비교 분석했다는 점에서 기존 연구와 차별점을 가진다. 또한, CNN의 일반적인 압축 기법인 Pruning(가지치기)과 양자화 연구들을 언급하며 이를 Siamese 구조에 적용하려는 시도를 정당화한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 Siamese 네트워크

본 연구는 기본적으로 Fully-Convolutional Siamese Network 구조를 채택한다. Siamese 네트워크는 두 개의 입력 브랜치가 동일한 가중치를 공유하며 유사도를 측정하는 Y자형 구조이다. 전체적인 동작은 다음과 같은 방정식으로 정의된다.

$$y = \gamma(\phi(z), \phi(x))$$

여기서 $\phi$는 딥러닝 특징 추출기(Feature Extractor)를 의미하며, $z$는 추적할 대상의 템플릿 이미지(Exemplar), $x$는 검색 영역(Region of Interest, ROI)을 나타낸다. $\gamma$는 두 특징 맵 간의 상호 상관(Cross-correlation) 연산을 수행하여 대상의 위치를 나타내는 히트맵(Heat map) $y$를 생성한다.

입력 데이터의 크기는 템플릿 $z$의 경우 $127 \times 127$ 픽셀, ROI $x$의 경우 $255 \times 255$ 픽셀로 설정된다. 특징 추출 후 $z$는 $17 \times 17 \times 32$, $x$는 $49 \times 49 \times 32$ 크기의 특징 맵으로 변환되며, 최종적으로 $33 \times 33$ 크기의 히트맵을 통해 객체의 중심 위치를 파악한다.

### 최적화 도구 및 기법

- **Brevitas & FINN**: PyTorch 기반의 양자화 학습 라이브러리인 Brevitas를 사용하여 양자화된 네트워크를 훈련시키고, 이를 FPGA 하드웨어 설계로 컴파일하기 위해 FINN 도구를 사용한다.
- **Quantisation (양자화)**: 가중치와 활성화 함수의 비트 정밀도를 낮추는 기법이다. 본 연구에서는 Uniform Integer Quantization(16, 4 bits), Ternary $\{-1, 0, 1\}$, Binary $\{-1, 1\}$ 방식을 적용하였다. 특히 입력층과 출력층은 학습의 안정성을 위해 비교적 높은 정밀도를 유지하고, 은닉층(Hidden layers)에 공격적인 양자화를 적용하는 전략을 사용한다.
- **Pruning (가지치기)**: 중요도가 낮은 뉴런이나 연결을 제거하여 파라미터 수를 줄이는 기법으로, L1/L2 노름(Norm) 등을 기준으로 불필요한 가중치를 제거하고 재학습(Fine-tuning)하는 반복 과정을 거친다.

## 📊 Results

### 실험 설정

- **데이터셋**: 훈련에는 ILSVRC15 데이터셋의 약 8%(10,000쌍)를 사용하였으며, 추적 성능 평가는 TempleColor, VOT14, VOT16 데이터셋을 사용하여 수행하였다.
- **평가 지표**: 훈련 단계에서는 손실 함수(Loss function)와 중심 오차(Centre error)를 측정하였고, 추적 단계에서는 정밀도(Precision, 중심 오차가 20픽셀 미만인 프레임의 비율)와 IOU(Intersection Over Union)를 측정하였다.

### 정량적 결과

양자화 수준에 따른 메모리 사용량과 추적 성능의 결과는 다음과 같다.

1. **메모리 효율성**: 부동 소수점(FP32) 기준 $85.5\text{ MB}$였던 컨볼루션 필터의 크기가 Binary 네트워크 적용 시 $7.2\text{ MB}$까지 감소하여, 약 10배 이상의 메모리 절감 효과를 거두었다.
2. **추적 정밀도**:
   - $\text{FP32}$: Precision $38.58\%$, IOU $28.42\%$
   - $\text{Ternary}$: Precision $44.35\%$, IOU $33.40\%$
   - $\text{Binary}$: Precision $44.22\%$, IOU $32.59\%$

놀랍게도 Binary 및 Ternary 네트워크가 Baseline인 FP32 네트워크보다 더 높은 추적 정밀도와 IOU를 기록하였다.

## 🧠 Insights & Discussion

본 연구의 가장 중요한 발견은 **양자화가 모델의 정규화(Regularization) 효과를 가져와 과적합을 방지한다**는 점이다. 특히 작은 데이터셋(ImageNet의 8%)으로 학습할 때, 정밀도를 낮추는 것이 오히려 일반화 성능을 높여 실제 추적 성능이 향상되는 결과가 나타났다.

또한, 마지막 레이어의 양자화 수준이 전체 네트워크 성능에 큰 영향을 미친다는 점을 확인하였다. 모든 레이어를 동일하게 낮게 양자화하는 것보다, 출력층의 정밀도를 일정 수준 유지하는 것이 더 정교한 추적을 가능하게 한다.

**한계 및 논의 사항**:

- 본 실험은 ImageNet의 일부 데이터셋만을 사용했으므로, 전체 데이터셋을 사용했을 때도 양자화된 모델이 FP32 모델보다 우수한 성능을 보일지는 추가 검증이 필요하다.
- FPGA 실제 구현 단계에서의 추론 시간(Inference time)에 대한 정량적 수치는 본 텍스트에 명시되지 않았으며, 향후 과제로 남아 있다.

## 📌 TL;DR

본 논문은 실시간 임베디드 객체 추적을 위해 Siamese Neural Network에 다양한 양자화(INT, Ternary, Binary)를 적용하여 최적화하였다. 실험 결과, Binary/Ternary 네트워크는 모델 크기를 10배 이상 줄이면서도 과적합 방지 효과를 통해 기본 FP32 모델보다 오히려 높은 추적 정밀도를 달성하였다. 이는 전력 효율이 중요한 FPGA 기반의 고해상도 실시간 비전 시스템 구현에 있어 매우 중요한 가능성을 제시한다.
