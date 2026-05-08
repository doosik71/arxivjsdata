# Searching for Efficient Architecture for Instrument Segmentation in Robotic Surgery

Daniil Pakhomov and Nassir Navab (2020)

## 🧩 Problem to Solve

본 논문은 로봇 보조 수술(Robot-assisted Minimally Invasive Surgery, RMIS) 환경에서 수술 도구를 실시간으로 정밀하게 분할(Segmentation)하는 문제를 해결하고자 한다. 수술 도구 분할은 수술 도구의 포즈 추정(Pose estimation)을 위한 필수 단계이며, 수술 중 증강 현실(Augmented Reality, AR) 오버레이가 도구에 의해 가려지지 않도록 마스킹 처리하는 데 직접적으로 사용된다.

기존의 연구들은 주로 분할 마스크의 정확도를 높이는 데 집중하였으나, 고해상도 수술 영상을 처리할 때 발생하는 막대한 연산 비용으로 인해 실제 실시간 응용 분야에 적용하기 어렵다는 한계가 있다. 따라서 본 연구의 목표는 고해상도 이미지에 대해 실시간 추론이 가능하면서도 높은 정확도를 유지하는 경량화된 딥 러닝 아키텍처를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 연산 비용의 주범인 네트워크 후반부의 필터 수를 줄여 속도를 높이고, 이로 인해 발생하는 정확도 저하를 추가적인 파라미터 증가 없이 **미분 가능한 탐색(Differentiable Search)**을 통한 최적의 **Dilation Rate(확장률)** 설정으로 보완하는 것이다.

1. **Light Residual Network 설계**: 기존 ResNet-18의 후반부 스테이지에서 필터 수를 대폭 줄여 추론 지연 시간(Latency)과 GPU 메모리 사용량을 낮추었다.
2. **최적 Dilation Rate 탐색**: Gumbel-Softmax 분포를 이용한 미분 가능한 탐색 기법을 도입하여, 각 residual unit에 최적화된 dilation rate를 자동으로 찾아내어 정확도를 회복하였다.

## 📎 Related Works

본 연구는 Dilated Residual Network 기반의 기존 state-of-the-art 방식(TernausNet 등)을 개선하였다. 기존 방식은 ResNet-18을 기반으로 하며, 해상도 손실을 막기 위해 마지막 두 블록의 stride를 1로 설정하고 dilated convolution을 적용하여 $8\times$ 다운샘플링만 수행한다.

그러나 이러한 방식은 ImageNet으로 사전 학습된 모델의 후반부 필터 수가 매우 많기 때문에, 다운샘플링이 제거되어 입력 특징 맵의 공간적 해상도가 커질 경우 연산량이 기하급수적으로 증가하는 문제가 있다. 본 논문은 필터 수를 줄인 경량 백본과 최적의 dilation rate 탐색을 통해 기존 방식의 연산 효율성 문제를 해결하고 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 시스템은 다음의 과정을 거쳐 구축된다: `ImageNet 사전 학습` $\rightarrow$ `Light ResNet 구조 설계` $\rightarrow$ `Differentiable Dilation Search` $\rightarrow$ `최종 모델 학습 및 추론`.

### 2. Light Residual Networks

추론 속도와 메모리 효율을 높이기 위해 ResNet-18의 후반부 채널 수를 줄인 두 가지 버전의 모델을 제안한다.

- **Light ResNet-18-v1**: 마지막 스테이지의 채널 수를 64로 설정.
- **Light ResNet-18-v2**: 마지막 스테이지의 채널 수를 32로 설정.

채널 수를 줄이면 ImageNet 분류 작업에서는 성능이 크게 떨어지지만, 클래스 수가 적은(2~4개) 수술 도구 분할 작업에서는 성능 저하가 상대적으로 적다는 점을 이용하였다.

### 3. Searching for Optimal Dilation Rates

정확도 손실을 보전하기 위해, 추가 파라미터 없이 각 residual unit의 dilation rate를 최적화한다. 이를 위해 **Gated Residual Unit**을 도입한다.

#### 주요 방정식 및 메커니즘

기존의 residual unit $x_{l+1} = x_l + F(x_l)$을 다음과 같이 확장한다:
$$x_{l+1} = x_l + \sum_{i=0}^{N} Z_i \cdot F_i(x_l)$$
여기서 $Z_i$는 어떤 dilation rate를 가진 $F_i$를 선택할지 결정하는 discrete gate 변수이며, $\sum Z_i = 1, Z_i \in \{0, 1\}$의 조건을 만족해야 한다.

Discrete 변수인 $Z$는 직접적으로 미분할 수 없으므로, **Gumbel-Softmax** 분포를 사용하여 continuous하게 근사화한다:
$$\bar{Z}_i = \text{softmax}((\log \alpha_i + G_i) / \tau)$$

- $G_i$: Gumbel random variable.
- $\alpha_i$: 학습 가능한 파라미터 (어떤 유닛을 선택할지 결정).
- $\tau$: 온도(Temperature) 파라미터로, 학습이 진행됨에 따라 0에 가깝게 어닐링(annealing)되어 hard decision에 수렴하게 한다.

이 과정을 통해 네트워크는 $\{1, 2, 4, 8, 16\}$의 predefined dilation rates 중 최적의 조합을 스스로 학습한다.

### 4. 학습 절차

- **데이터셋**: EndoVis 2017 Robotic Instruments dataset.
- **손실 함수**: Normalized pixel-wise cross-entropy loss를 사용한다.
- **최적화**: Adam optimizer와 Poly learning rate policy를 적용하였으며, 초기 학습률은 $0.001$, 배치 사이즈는 32, 크롭 사이즈는 799로 설정하였다.
- **과정**: ImageNet 사전 학습 $\rightarrow$ Gated unit을 통한 dilation rate 탐색 $\rightarrow$ 결정된 dilation rate로 최종 모델 학습.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis 2017 (7종의 수술 도구 포함).
- **태스크**: Binary segmentation (도구 vs 배경) 및 Parts segmentation (Shaft, Wrist, Jaws, 배경).
- **평가 지표**: mean Intersection over Union (mIoU) 및 추론 시간(Latency).
- **하드웨어**: NVIDIA GTX 1080Ti 및 Tesla P100 (입력 크기 $1024 \times 1280$).

### 정량적 결과

- **속도**: Light ResNet-18-v2 (Learnt Dilations 적용) 모델은 Tesla P100에서 **7.95ms**의 지연 시간을 기록하며, 최대 **125 FPS**의 속도를 달성하였다.
- **정확도**:
  - Binary segmentation에서 v1 (Learnt Dilations)은 **0.869 IoU**를 기록하여, 기존 Dilated ResNet-18(0.896)보다는 약간 낮지만 TernausNet(0.888)에 근접하는 성능을 보였다.
  - Parts segmentation에서도 v1 (Learnt Dilations)은 **0.742 IoU**를 기록하여 효율성 대비 높은 정확도를 유지하였다.

| Model                                  | Binary IoU | Binary Time | Parts IoU | Parts Time |
| :------------------------------------- | :--------: | :---------: | :-------: | :--------: |
| TernausNet-16                          |   0.888    |   184 ms    |   0.737   |   202 ms   |
| Dilated ResNet-18                      |   0.896    |   126 ms    |   0.764   |   126 ms   |
| Light ResNet-18-v1 w/ Learnt Dilations |   0.869    |   17.4 ms   |   0.742   |  17.4 ms   |
| Light ResNet-18-v2 w/ Learnt Dilations |   0.852    |   11.8 ms   |   0.729   |  11.8 ms   |

*(Table 1 기반, GTX 1080Ti 측정 기준)*

## 🧠 Insights & Discussion

본 논문은 모델의 복잡도를 줄이면서도 성능을 유지할 수 있는 영리한 접근 방식을 보여준다. 특히, 일반적인 분류 문제(ImageNet)와 달리 특정 도메인의 분할 문제에서는 네트워크 후반부의 채널 수가 절대적으로 많이 필요하지 않다는 통찰을 통해 과감한 경량화를 수행하였다.

또한, 단순한 경량화는 정보 손실을 야기하지만, 이를 파라미터 증가 없이 dilation rate 최적화라는 구조적 튜닝으로 해결한 점이 돋보인다. Gumbel-Softmax를 이용한 differentiable search는 수동으로 하이퍼파라미터를 찾는 수고를 덜어주며 데이터 기반의 최적 구조를 도출하게 한다.

다만, 본 연구는 EndoVis 2017이라는 단일 데이터셋에서만 검증되었으므로, 다른 수술 환경이나 도구 종류에 대해서도 일반화 성능이 유지되는지에 대한 추가 검증이 필요할 것으로 보인다.

## 📌 TL;DR

이 논문은 로봇 수술 도구 분할을 위해 **추론 속도를 극대화한 경량 Dilated Residual Network**를 제안한다. 네트워크 후반부의 채널 수를 줄여 연산량을 낮추고, **Gumbel-Softmax 기반의 미분 가능한 탐색**으로 최적의 dilation rate를 찾아 정확도 저하를 방지하였다. 그 결과, 고해상도 이미지에서 최대 **125 FPS**라는 압도적인 속도와 경쟁력 있는 정확도를 동시에 달성하였으며, 이는 실시간 수술 보조 시스템 구축에 중요한 기여를 할 수 있다.
