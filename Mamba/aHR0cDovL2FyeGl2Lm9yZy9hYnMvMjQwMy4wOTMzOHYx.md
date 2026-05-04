# LocalMamba: Visual State Space Model with Windowed Selective Scan

Tao Huang, Xiaohuan Pei, Shan You, Fei Wang, Chen Qian, and Chang Xu (2024)

## 🧩 Problem to Solve

본 논문은 최근 시퀀스 모델링에서 뛰어난 성능을 보이는 State Space Model(SSM), 특히 Mamba를 비전 작업에 적용할 때 발생하는 효율성 및 성능 정체 문제를 해결하고자 한다. 

기존의 Vision Mamba(ViM) 계열 모델들은 2D 이미지 데이터를 1D 시퀀스로 변환하여 처리하는데, 이 과정에서 이미지를 단순히 평탄화(flatten)함으로써 인접한 픽셀 간의 지역적 2D 의존성(local 2D dependencies)이 파괴되는 문제가 발생한다. 이는 결과적으로 의미적으로 가까운 토큰들 사이의 거리를 멀어지게 만들어, 모델이 이미지의 세밀한 지역적 특성을 포착하는 능력을 저하시킨다. 

따라서 본 연구의 목표는 이미지의 지역적 의존성을 효과적으로 유지하면서도 글로벌 컨텍스트를 함께 학습할 수 있는 새로운 스캔 전략을 도입하여, 기존 CNN이나 Vision Transformer(ViT) 대비 SSM 기반 비전 모델의 경쟁력을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Windowed Selective Scan**과 **Adaptive Scan Direction Search**이다.

1.  **지역적 스캔 전략(Local Scanning Strategy):** 이미지를 여러 개의 독립적인 윈도우(window)로 분할하여 각 윈도우 내에서 먼저 스캔을 수행함으로써, 지역적 의미 영역 내의 토큰들이 서로 가깝게 배치되도록 하여 지역적 의존성 포착 능력을 극대화한다.
2.  **동적 스캔 방향 탐색(Dynamic Scan Direction Search):** 네트워크의 각 층(layer)마다 최적의 스캔 패턴이 다를 수 있다는 점에 착안하여, 가능한 여러 스캔 방향 중 해당 층에 가장 적합한 조합을 독립적으로 찾아내는 미분 가능한 탐색 방법을 제안한다.
3.  **SCAttn 모듈 도입:** 서로 다른 스캔 방향에서 추출된 다양한 특징들을 효율적으로 통합하고 불필요한 정보를 필터링하기 위해 공간 및 채널 어텐션 모듈인 SCAttn(Spatial and Channel Attention)을 설계하였다.

## 📎 Related Works

### 관련 연구 및 한계
- **Generic Vision Backbones:** CNN(ResNet 등)은 지역적 특성 추출에 강점이 있고, ViT는 글로벌 컨텍스트 파악에 유리하지만 연산 비용이 높다는 한계가 있다.
- **State Space Models (SSMs):** S4와 Mamba는 시퀀스 길이에 대해 선형 시간 복잡도를 가지며 긴 의존성을 모델링하는 데 탁월하다.
- **Vision Mamba (Vim, VMamba):** Mamba를 비전에 적용하려는 시도가 있었으며, Vim은 양방향 스캔을, VMamba는 가로-세로 교차 스캔(Cross-Scan)을 통해 2D 공간 특성을 반영하려 했다.

### 기존 접근 방식과의 차별점
기존의 VMamba 등이 2D 스캔을 도입했음에도 불구하고, 여전히 스캔된 시퀀스 내에서 원래 인접했던 토큰들의 근접성을 완전히 유지하지 못한다는 한계가 있다. LocalMamba는 이미지를 윈도우 단위로 나누어 처리함으로써, 윈도우 내부의 지역적 응집력을 강제적으로 높여 이 문제를 해결한다.

## 🛠️ Methodology

### 1. State Space Model (SSM) 기초
SSM은 1차원 시퀀스 $x(t)$를 잠재 상태 $h(t)$를 통해 출력 $y(t)$로 매핑하는 모델이다.
$$\text{h}'(t) = Ah(t) + Bx(t)$$
$$\text{y}(t) = Ch(t)$$
여기서 $A, B, C$는 시스템 행렬이다. 실제 구현을 위해 zero-order hold 가정을 통해 이산화(discretization) 과정을 거치며, 샘플링 타임스케일 $\Delta$를 사용하여 다음과 같이 변환된다.
$$\bar{A} = e^{\Delta A}$$
$$\bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B$$
이후 이산화된 모델은 $\text{h}_t = \bar{A}\text{h}_{t-1} + \bar{B}\text{x}_t, \text{y}_t = \text{C}\text{h}_t$ 형태로 동작하며, 이는 글로벌 컨볼루션 연산 $\text{y} = \text{x} \circledast \text{K}$로 가속화할 수 있다. Mamba(S6)는 여기서 $\bar{A}, \bar{B}, \text{C}$ 등을 입력값에 따라 동적으로 결정하는 선택적 메커니즘을 도입하였다.

### 2. Local Scan Mechanism
이미지를 특정 크기의 윈도우로 나누고, 윈도우 내부에서 먼저 스캔을 수행한 뒤 윈도우 간 스캔을 진행한다. 이를 통해 동일한 지역적 의미 영역에 속한 토큰들이 시퀀스 상에서 가깝게 배치되도록 한다. 
본 모델은 글로벌 컨텍스트 유지를 위해 다음 4가지 방향의 브랜치를 병렬로 구성한다.
- 기존의 수평/수직 스캔 방향 및 그 반대 방향(flipped)

### 3. SCAttn (Spatial and Channel Attention)
4개의 스캔 브랜치에서 나온 특징들을 단순 합산하는 대신, SCAttn 모듈을 통해 가중치를 조절한다.
- **Channel Attention:** 공간 차원에 대해 평균 풀링을 수행한 후 선형 변환을 통해 채널별 중요도를 계산한다.
- **Spatial Attention:** 글로벌 표현을 토큰 특징에 더해 각 토큰의 중요도를 평가하고 가중치를 부여한다.

### 4. Adaptive Scan Direction Search
각 층마다 최적의 스캔 방향을 찾기 위해 DARTS의 개념을 도입하여 미분 가능한 탐색을 수행한다.
- **탐색 공간(Search Space):** 수평, 수직, $2 \times 2$ 지역 스캔, $7 \times 7$ 지역 스캔 및 각각의 반대 방향을 포함한 총 8개의 후보군 $S$를 설정한다.
- **연산 과정:** 각 층 $l$에 대해 학습 가능한 파라미터 $\alpha_s^{(l)}$를 도입하고, Softmax 확률을 통해 각 스캔 방향의 기여도를 결정한다.
$$\text{y}^{(l)} = \sum_{s \in S} \frac{\exp(\alpha_s^{(l)})}{\sum_{s' \in S} \exp(\alpha_{s'}^{(l)})} \text{SSM}_s(\text{x}^{(l)})$$
학습 후 확률값이 가장 높은 상위 4개의 방향을 최종 선택한다.

### 5. 아키텍처 변형
- **LocalVim:** Plain 구조를 가지며, 기존 Vim의 블록을 LocalMamba 블록으로 대체한다. 연산량 유지를 위해 블록 수를 24개에서 20개로 조정하였다.
- **LocalVMamba:** 계층적(Hierarchical) 구조를 가지며, VMamba의 블록을 LocalMamba 블록으로 대체한다.

## 📊 Results

### 실험 설정
- **데이터셋:** ImageNet-1K (분류), MSCOCO 2017 (객체 탐지), ADE20K (시맨틱 세그멘테이션).
- **평가 지표:** Top-1 Accuracy (%), Box/Mask AP, mIoU.

### 주요 결과
1.  **ImageNet 분류:**
    - **LocalVim-T**는 1.5G FLOPs에서 76.2%의 정확도를 기록하여, 동일 수준의 DeiT-Ti(72.2%)와 Vim-Ti(73.1%)를 크게 상회한다.
    - **LocalVMamba-T**는 82.7%를 기록하여 Swin-T(81.3%)보다 1.4% 높다.
    - 스캔 방향 탐색을 적용하지 않은 모델($\ast$)보다 탐색을 적용한 모델이 더 높은 성능을 보였으며, 이는 층별 최적 방향 설정의 중요성을 입증한다.

2.  **객체 탐지 (COCO):**
    - Mask R-CNN을 기반으로 측정했을 때, LocalVMamba-T는 Box AP 46.7, Mask AP 42.2를 기록하여 Swin-T 대비 각각 4.0, 2.9 포인트 높은 성능을 보였다.

3.  **시맨틱 세그멘테이션 (ADE20K):**
    - UperNet을 사용한 결과, LocalVim-S는 Vim-S보다 mIoU(SS) 기준 1.5 포인트 높은 성능을 달성하였다.
    - LocalVMamba-S는 mIoU(MS) 51.0을 기록하며 VMamba-S(50.5)를 능가했다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **지역성-전역성 균형:** 윈도우 기반 스캔을 통해 지역적 의존성을 포착하면서도, 전체 이미지를 SSM으로 처리함으로써 글로벌 컨텍스트를 유지하는 데 성공하였다.
- **층별 특성 차이:** 탐색된 스캔 방향을 시각화한 결과, Plain 구조(LocalVim)에서는 초기/말기 층에서 지역 스캔을 선호하고 중간 층에서는 글로벌 스캔을 선호하는 경향이 나타났다. 특히 네트워크 후반부로 갈수록 더 작은 윈도우($2 \times 2$) 스캔을 사용하는 특징이 발견되었다.

### 한계 및 비판적 해석
- **계산 프레임워크의 제약:** SSM은 이론적으로 선형 복잡도를 가지지만, 현재의 딥러닝 프레임워크(PyTorch 등)에서는 컨볼루션이나 셀프 어텐션만큼 최적화된 가속 연산이 구현되어 있지 않아 실제 병렬 연산 효율이 낮을 수 있다.
- **자원 소모:** 미분 가능한 아키텍처 탐색(NAS) 과정에서 많은 계산 자원이 소모된다는 점이 환경적, 비용적 부담이 될 수 있다.

## 📌 TL;DR

LocalMamba는 비전 Mamba 모델들이 이미지의 2D 지역적 특성을 무시하고 1D로 평탄화하여 처리하던 문제를 해결하기 위해 **윈도우 기반 지역 스캔(Windowed Selective Scan)**과 **층별 최적 스캔 방향 탐색**을 제안한 연구이다. 이 방법론을 통해 모델은 세밀한 지역 정보와 광범위한 전역 정보를 동시에 효과적으로 학습할 수 있게 되었으며, 이미지 분류, 객체 탐지, 세그멘테이션 등 다양한 비전 작업에서 기존의 CNN, ViT 및 기존 Mamba 기반 모델들보다 우수한 성능을 입증하였다. 향후 SSM의 효율적인 하드웨어 가속 구현이 뒷받침된다면 매우 강력한 비전 백본으로 활용될 가능성이 높다.