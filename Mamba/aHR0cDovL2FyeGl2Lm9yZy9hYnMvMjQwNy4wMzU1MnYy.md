# Vision Mamba for Classification of Breast Ultrasound Images

Ali Nasiri-Sarvi, Mahdi S. Hosseini, and Hassan Rivaz (2024)

## 🧩 Problem to Solve

본 논문은 유방 초음파 영상을 이용한 유방암 분류 문제에서 최적의 인코더 아키텍처를 탐색하는 것을 목표로 한다. 유방 초음파 데이터셋은 일반적으로 크기가 작기 때문에, 사전 학습된 가중치를 사용하는 전이 학습(Transfer Learning) 패러다임이 필수적이다.

기존의 주류 모델인 Convolutional Neural Networks(CNNs)는 지역적 패턴 포착에는 뛰어나나 수용 영역(Receptive Field)의 한계로 인해 장거리 의존성(Long-range dependencies)을 학습하는 데 어려움이 있다. 반면, Vision Transformers(ViTs)는 Attention 메커니즘을 통해 장거리 의존성을 효과적으로 포착할 수 있지만, CNN이 가진 귀납적 편향(Inductive Bias)이 부족하여 유사한 성능을 내기 위해 훨씬 더 많은 양의 데이터와 학습 시간이 필요하다는 단점이 있다. 따라서 본 연구의 목표는 최신 State Space Models(SSMs) 기반의 Mamba 아키텍처가 이러한 CNN과 ViT의 한계를 극복하고, 제한된 의료 데이터셋 환경에서 유방 초음파 영상 분류 성능을 향상시킬 수 있는지 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba 기반의 비전 인코더인 Vim과 VMamba를 유방 초음파 분류 작업에 적용하고, 이를 기존의 CNN 및 ViT 모델들과 정량적, 통계적으로 비교 분석한 것이다.

중심적인 설계 아이디어는 Mamba 모델이 CNN의 귀납적 편향(인접한 패치들이 유사한 콘텐츠를 가진다는 가정)과 ViT의 장거리 정보 처리 능력을 동시에 결합할 수 있다는 점에 있다. 특히, VMamba의 2D Selective Scan 메커니즘이 이미지의 복잡한 관계를 더 포괄적으로 학습할 수 있게 하여, 데이터가 제한적인 상황에서도 효율적인 표현 학습이 가능하다는 것을 입증하고자 하였다.

## 📎 Related Works

기존 연구들에서 CNN(ResNet, VGG 등)은 유방 초음파 분류에서 널리 사용되었으며 지역적 특성 추출에 강점을 보였다. ViT는 이를 대체하기 위해 제안되었으며, 귀납적 편향을 제거함으로써 데이터로부터 더 자유롭게 학습할 수 있는 가능성을 제시하였다. 그러나 두 방식 모두 의료 영상과 같이 데이터셋이 작은 환경에서는 전이 학습에 크게 의존해야 하며, 각각 수용 영역의 한계와 과도한 데이터 요구량이라는 명확한 한계를 가진다.

최근 등장한 Mamba 기반 모델들은 SSM을 활용하여 선형 시간 복잡도로 시퀀스를 모델링하며, 이미 비디오 이해, 원격 탐사, 병리 데이터셋 등 다양한 분야에서 가능성을 보였다. 의료 영상 분야에서는 주로 세그멘테이션(Segmentation) 작업에 적용된 사례(UMamba 등)가 있었으나, 유방 초음파 분류 작업에 대한 성능 검증은 아직 이루어지지 않은 상태였다. 본 논문은 이러한 공백을 메우기 위해 분류 작업에 Mamba 아키텍처를 도입하였다.

## 🛠️ Methodology

### 1. State Space Models (SSMs) 및 Mamba

State Space Models는 연속 선형 시스템을 기술하는 수학적 프레임워크로, 입력 $x(t)$에 따른 상태 $h(t)$의 변화와 출력 $y(t)$의 관계를 다음과 같이 정의한다.

- 상태 방정식: $$h_t = Ah_{t-1} + Bx_t$$
- 출력 방정식: $$y_t = Ch_t + Dx_t$$

여기서 $A, B, C$ 파라미터가 시간에 따라 변하지 않는(Time-invariant) 경우, 전역 합성곱(Global Convolution)을 통해 시퀀스 데이터를 효율적으로 처리할 수 있다.

**Mamba**는 여기서 더 나아가 $B, C, \Delta$(이산화 파라미터)를 각 입력 시퀀스에 따라 동적으로 계산하는 **Selective State Spaces**를 도입하였다. 이는 모델이 입력 데이터의 시간적 변화와 복잡한 의존성을 더 잘 포착하게 한다. 또한, GPU의 SRAM과 HBM 메모리 계층 구조를 고려한 하드웨어 인식 알고리즘(Hardware-aware algorithm)을 통해, 순차적 처리 방식임에도 불구하고 트랜스포머와 유사한 병렬 처리 효율성을 확보하였다.

### 2. 비전 Mamba 아키텍처

본 논문에서는 두 가지 Mamba 기반 비전 모델을 사용한다.

- **Vim (Vision Mamba):** 이미지를 작은 패치로 나누고 각 패치를 임베딩한 후, 양방향 Mamba(Bidirectional Mamba)를 통해 이전과 다음 토큰을 모두 고려하여 처리한다. 공간 정보를 유지하기 위해 Positional Encoding이 추가되며, 구조적으로는 ViT와 유사하나 Attention 블록 대신 Mamba 블록을 사용한다.
- **VMamba:** 이미지를 토큰화하는 대신 특징 맵(Feature Maps)으로 처리하며, CNN과 유사하게 레이어를 거칠 때마다 다운샘플링을 수행한다. 특히 **2D Selective Scan**을 도입하여 4가지 방향으로 스캔을 수행함으로써, 이미지 패치 간의 더 포괄적이고 복잡한 관계를 캡처한다.

### 3. 모델 간 비교 관점

- **CNN:** 수용 영역이 제한되어 멀리 떨어진 패치 간의 관계 포착이 어렵다.
- **ViT:** Attention을 통해 장거리 의존성을 잡지만, 귀납적 편향이 없어 인접 패치 간의 관계를 배우기 위해 많은 데이터가 필요하다.
- **Mamba:** 순차적 처리를 통해 CNN의 귀납적 편향을 다시 도입하면서도, SSM의 특성을 통해 ViT와 같은 장거리 정보 처리가 가능하다.

## 📊 Results

### 실험 설정

- **데이터셋:** BUSI, B dataset, 그리고 이를 합친 BUSI+B 데이터셋을 사용하였다.
- **비교 모델:**
  - CNN: ResNet50, VGG16
  - ViT: ViT-ti16, ViT-s16, ViT-s32, ViT-b16, ViT-b32
  - Mamba: Vim-s16, VMamba-ti, VMamba-s, VMamba-b
- **평가 지표:** 데이터 불균형을 고려하여 AUC(Area Under the Curve)를 주지표로 사용하였으며, Accuracy(ACC)를 함께 측정하였다.
- **절차:** ImageNet 사전 학습 가중치를 사용한 전이 학습을 수행하였으며, 5회 반복 실험 후 평균과 표준편차를 산출하였다. 모델 간 성능 차이의 유의성을 검증하기 위해 Paired t-test(p-value < 0.05)를 실시하였다.

### 주요 결과

1. **BUSI+B 데이터셋:**
    - VMamba-ti가 VGG16, ViT-ti16, ViT-s32, ViT-b32보다 통계적으로 유의미하게 높은 성능을 보였다.
    - VMamba-b 또한 VGG16과 ViT-ti16 대비 유의미한 성능 향상을 나타냈다.
    - ResNet50이 평균 AUC는 높았으나, 통계적 유의성 분석 결과 다른 모델들과의 차이가 유의미하지 않은 것으로 나타났다.

2. **BUSI 데이터셋:**
    - VMamba-b가 ResNet50, VGG16, ViT-ti16, ViT-s16, ViT-s32, ViT-b32 등 대부분의 모델을 통계적으로 유의미하게 압도하였다.
    - VMamba-ti 역시 ResNet50, VGG16, ViT-s32보다 뛰어난 성능을 보였다.

3. **B 데이터셋:**
    - **VMamba-ti**가 모든 인코더 중 가장 우수한 성능을 보였으며, 특히 AUC와 ACC 모두에서 큰 폭의 향상을 기록하였다.
    - VMamba-ti는 ResNet50, VGG16 및 모든 ViT 변형 모델들과 비교했을 때 통계적으로 유의미하게 우수하였다.

## 🧠 Insights & Discussion

본 연구 결과는 유방 초음파 영상 분류와 같이 데이터가 제한적인 의료 환경에서 Mamba 기반 모델이 CNN과 ViT의 훌륭한 대안이 될 수 있음을 시사한다. 특히 Mamba 모델이 CNN의 지역적 귀납적 편향과 ViT의 글로벌 컨텍스트 포착 능력을 동시에 보유하고 있다는 점이 성능 향상의 주요 원인으로 분석된다. 또한, Vim보다 VMamba가 전반적으로 더 좋은 성능을 보였는데, 이는 4방향으로 스캔하는 2D Selective Scan이 이미지의 공간적 특징을 더 효과적으로 표현했기 때문으로 풀이된다.

다만, 본 연구의 한계점으로는 다중 클래스 분류 분석 시 데이터 불균형 문제를 통계적 유의성 분석에 완전히 반영하지 못했다는 점이 언급되었다. 특히 여러 클래스의 AUC를 비교하는 과정이 복잡하여 단순 평균 기반의 t-test를 수행하였으므로, 향후 다중 클래스 AUC를 포함한 더 정교한 통계 분석이 필요하다.

## 📌 TL;DR

본 논문은 최신 SSM 기반의 Vision Mamba 모델(Vim, VMamba)을 유방 초음파 영상 분류에 적용하여 CNN 및 ViT 모델들과 비교 분석하였다. 실험 결과, **VMamba-ti**를 비롯한 Mamba 기반 모델들이 통계적으로 유의미하게 우수한 성능을 보였으며, 특히 데이터가 적은 환경에서 CNN의 지역적 특성과 ViT의 전역적 특성을 모두 활용할 수 있다는 강점을 입증하였다. 이는 향후 의료 영상 분석 분야에서 효율적인 인코더 아키텍처로서 Mamba의 활용 가능성을 높이는 중요한 연구이다.
