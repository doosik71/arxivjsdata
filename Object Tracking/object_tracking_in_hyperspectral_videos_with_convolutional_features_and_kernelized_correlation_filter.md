# Object Tracking in Hyperspectral Videos with Convolutional Features and Kernelized Correlation Filter

Kun Qian, Jun Zhou, Fengchao Xiong, Huixin Zhou, and Juan Du (2018)

## 🧩 Problem to Solve

본 논문은 초분광 비디오(Hyperspectral Video)에서의 객체 추적(Object Tracking) 문제를 해결하고자 한다. 초분광 영상은 수백 개의 연속적이고 좁은 스펙트럼 밴드를 제공하여 매우 풍부한 분광 정보를 포함하고 있지만, 기존의 객체 추적 연구는 대부분 그레이스케일(Grayscale)이나 RGB 비디오에 집중되어 있었다.

초분광 비디오 처리 연구가 부족했던 주된 이유는 고속 촬영이 가능한 초분광 카메라의 보급이 늦었기 때문이다. 최근 저비용의 고속 초분광 카메라가 등장함에 따라, 본 연구는 초분광 데이터가 가진 풍부한 spectral-spatial 정보를 활용하여 기존 RGB/그레이스케일 기반 추적 방식보다 더 강건한 추적 성능을 달성하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 초분광 영상의 특성을 반영할 수 있는 **3D Convolutional Features**를 추출하고, 이를 **Kernelized Correlation Filter (KCF)** 프레임워크에 통합하는 것이다.

가장 중심적인 설계 아이디어는 오프라인 학습 과정 없이, 타겟 영역에서 정규화된 3D 큐브(cubes)를 직접 추출하여 이를 고정된 convolution filter로 사용하는 것이다. 이를 통해 타겟 주변의 분광 정보를 효과적으로 인코딩하고, 별도의 대규모 데이터셋 학습 없이도 강건한 표현(representation)을 학습할 수 있는 단순한 2계층 합성곱 네트워크 구조를 제안하였다.

## 📎 Related Works

논문에서는 Discriminative Correlation Filter (DCF) 기반의 추적 방법론들을 언급한다.

- **MOSSE**: 그레이스케일 특징을 사용하여 매우 빠른 속도를 구현한 초기 DCF 기반 추적기이다.
- **KCF (Kernelized Correlation Filter)**: 커널 트릭과 HOG(Histogram of Oriented Gradients) 특징을 결합하여 성능을 높였으며, 순환 행렬(circulant matrix) 구조를 통해 FFT(Fast Fourier Transform)를 이용한 효율적인 학습을 가능하게 했다.
- **STC (Spatio-Temporal Context)**: 확률론적 관점에서 상관 필터를 탐구하고 조밀한 샘플링을 통해 추적 성능을 개선했다.
- **Deep Learning 기반 추적기 (DLT, CNT)**: 수작업으로 설계된 특징 대신 CNN을 통해 자동으로 학습된 특징을 사용하여 강건성을 크게 향상시켰다. 특히 CNT는 풀링 층이 없는 단순한 2계층 CNN만으로도 경쟁력 있는 결과를 낼 수 있음을 보였다.

기존 접근 방식과의 차별점은, 이들이 주로 RGB나 그레이스케일 영상의 공간적 특징에 의존하는 반면, 제안 방법은 초분광 영상의 **분광(spectral) 정보**를 3D 합성곱 연산을 통해 직접적으로 추출하여 활용한다는 점이다.

## 🛠️ Methodology

### 전체 파이프라인

본 논문이 제안하는 방법론은 크게 두 단계로 구성된다: (1) 초분광 타겟으로부터 3D 합성곱 특징을 추출하는 단계, (2) 추출된 다채널 특징을 KCF 프레임워크에 입력하여 타겟을 추적하는 단계이다.

### 3D Convolutional Features 추출

입력되는 초분광 이미지 패치는 $I \in \mathbb{R}^{n \times n \times d}$로 정의되며, 여기서 $n$은 패치 크기, $d$는 스펙트럼 밴드 수이다.

1. 타겟 영역 내부에서 슬라이딩 윈도우를 통해 조밀하게 샘플링된 $l = n \times n$개의 로컬 이미지 패치 $Y_i \in \mathbb{R}^{w \times w \times d}$ 세트를 생성한다.
2. 이 중 $d$개의 필터 $f_j$를 무작위로 선택하여 입력 이미지 $I$와 합성곱 연산을 수행한다.
3. 각 필터 $f_j$에 의한 응답 맵(feature map) $S_j \in \mathbb{R}^{n \times n}$는 다음과 같이 계산된다:
   $$S_j = I \otimes f_j$$
4. 생성된 여러 개의 $S_j$를 쌓아(stack) 타겟 객체의 3차원 표현(three-dimensional representation)을 형성한다. 이는 KCF의 입력으로 들어가는 다채널 특징(multichannel feature)이 된다.

### KCF Tracking Framework

KCF는 릿지 회귀(Ridge Regression) 모델을 통해 분류기를 학습시키며, 목적 함수는 다음과 같다:
$$\min_{w} ((w^T x - y)^2 + \lambda \|w\|^2)$$
여기서 $y$는 회귀 값, $\lambda$는 정규화 매개변수, $w$는 회귀 계수이다. 학습 효율을 위해 순환 행렬(circulant matrix) $C(x)$를 도입하고, DFT(Discrete Fourier Transform)를 통해 주파수 영역에서 연산을 수행한다.

커널 트릭을 적용하여 특징을 고차원 공간으로 매핑하면, 계수 $\alpha$는 다음과 같이 도출된다:
$$\alpha = (K + \lambda I)^{-1} y$$
주파수 영역에서의 식은 다음과 같다:
$$F\alpha = ((Fk_{xx}) + \lambda)^{-1} (Fy)$$
여기서 $k_{xx}$는 커널 행렬의 첫 번째 행이며, $F$는 DFT 행렬이다.

### 초분광 데이터로의 확장

초분광 데이터의 다채널 표현 $x = [x_1, x_2, \dots, x_d]$를 처리하기 위해, 가우시안 커널 함수를 다음과 같이 수정하여 각 채널의 내적 합으로 계산한다:
$$k_{xz} = \exp\left(-\frac{1}{\sigma^2}(\|x\|^2 + \|z\|^2) - 2F^{-1}\left(\sum_{d} (Fx_d) \otimes (Fz_d)\right)\right)$$
이 수식을 통해 3D stacked convolutional features를 KCF의 다채널 특징으로 활용할 수 있게 된다.

## 📊 Results

### 실험 설정

- **데이터셋**: 3가지 장면(apple, deer, people)에 대해 각각 그레이스케일, RGB, 초분광(HSI) 버전의 총 9개 시퀀스를 사용하였다.
- **비교 대상**: DLT, CNT, STC, KCF 등 최신 추적 알고리즘들과 비교하였다.
- **평가 지표**: Precision (타겟 중심점 간의 거리가 20px 이내인 프레임의 비율)을 사용하였다.
- **환경**: Intel i7-7700HQ CPU, 32GB RAM 환경에서 구현되었으며, 제안 방법(CNHT)의 속도는 1 FPS이다.

### 주요 결과

1. **정량적 결과**: 제안된 CNHT 방법은 HSI 시퀀스에서 **0.982 (98.2%)**라는 가장 높은 정밀도를 기록하였다. 이는 그레이스케일에서 가장 좋은 성능을 보인 CNT(0.927)보다 높으며, 단일 밴드만 사용한 KCF보다 83% 향상된 결과이다.
2. **정성적 결과 (Background Clutter)**:
   - `apple-RGB` 시퀀스에서 사과(빨간색)와 배경(빨간색)의 색상이 유사하여 기존 방법(DLT, STC, KCF)들은 타겟을 놓치거나 표류(drift)하는 현상이 발생했다.
   - 반면, CNHT는 초분광 정보를 활용하여 색상 유사성 문제를 해결하고 시퀀스 전체에서 안정적으로 추적하였다.
3. **부분 폐쇄 (Partial Occlusion)**: 손가락에 의한 사과 가림이나 수풀에 의한 사람 가림 상황에서도 HSI 기반의 CNHT가 RGB/그레이스케일 방식보다 훨씬 정확하고 안정적인 추적 성능을 보였다.

## 🧠 Insights & Discussion

### 강점

본 연구의 가장 큰 강점은 **초분광 데이터의 고유한 분광 특성을 3D Convolution을 통해 효과적으로 추출**했다는 점이다. 특히 RGB 영상에서는 구분이 불가능한 '색상이 유사한 배경과 객체' 상황에서 초분광 정보가 매우 강력한 판별력을 제공함을 실험적으로 입증하였다. 또한, 대규모 데이터셋을 통한 사전 학습 없이 타겟 자체에서 필터를 추출하는 방식을 통해 실용성을 높였다.

### 한계 및 비판적 해석

1. **연산 속도**: 현재 제안 방법의 속도는 1 FPS로, 실시간 추적(Real-time tracking)과는 거리가 멀다. 논문에서는 GPU 가속을 통해 개선 가능하다고 언급하였으나, 구체적인 GPU 구현 결과는 제시되지 않았다.
2. **필터 선택의 임의성**: 3D 필터를 샘플링할 때 무작위(randomly)로 선택하는 방식을 사용했다. 필터의 선택 기준이 성능에 미치는 영향에 대한 심층적인 분석이 부족하다.
3. **데이터셋의 규모**: 실험에 사용된 데이터셋이 9개의 시퀀스로 매우 제한적이다. 더 다양하고 대규모인 초분광 비디오 데이터셋에서의 검증이 필요하다.

## 📌 TL;DR

본 논문은 초분광 비디오의 풍부한 spectral-spatial 정보를 활용하기 위해 **3D 합성곱 특징 추출과 KCF(Kernelized Correlation Filter)를 결합한 추적기(CNHT)**를 제안한다. 타겟 영역에서 직접 추출한 3D 필터를 통해 학습 없이도 강건한 특징을 얻으며, 이를 통해 RGB 영상에서는 해결하기 어려운 **배경 유사성(Background Clutter) 문제를 효과적으로 해결**하여 98.2%의 높은 정밀도를 달성하였다. 이 연구는 향후 초분광 영상 기반의 객체 추적 및 원격 탐사 분야의 발전 가능성을 제시한다.
