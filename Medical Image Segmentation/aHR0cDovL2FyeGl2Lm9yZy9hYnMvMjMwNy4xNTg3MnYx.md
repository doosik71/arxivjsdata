# Cross-dimensional transfer learning in medical image segmentation with deep learning

Hicham Messaoudi, Ahror Belaid, Douraied Ben Salem, Pierre-Henri Conze (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 발생하는 **주석 데이터의 부족(lack of annotated data)** 문제를 해결하고자 한다. 딥러닝 네트워크, 특히 합성곱 신경망(CNN)이 효과적으로 일반화되기 위해서는 방대한 양의 데이터가 필요하지만, 의료 영상의 경우 국가별 규제와 전문의의 수동 주석 작업에 드는 막대한 비용 및 시간으로 인해 고품질의 데이터셋을 확보하는 것이 매우 어렵다.

특히 3D 볼륨 데이터의 경우, 2D 데이터보다 주석 작업의 난이도가 훨씬 높으며 계산 비용 또한 증가한다. 기존의 전이 학습(Transfer Learning)은 주로 2D 분류 네트워크에서 2D 분할 네트워크로 가중치를 전이하는 방식에 머물러 있었으며, 2D에서 학습된 지식을 3D 네트워크로 확장하는 **차원 간 전이 학습(Cross-dimensional transfer learning)**에 대한 연구는 매우 부족한 실정이다. 따라서 본 연구의 목표는 자연 이미지로 학습된 2D 분류 네트워크의 효율성을 2D 및 3D, 단일 및 다중 모달리티 의료 영상 분할 작업으로 효과적으로 전이하는 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 2D 사전 학습 모델의 가중치를 고차원 네트워크에 적용하는 두 가지 전이 학습 원칙을 설계한 것이다.

1.  **Weight Transfer Learning (WTL, 가중치 전이 학습):** 사전 학습된 2D 인코더를 그대로 유지하면서 이를 더 높은 차원의 네트워크(예: 3D U-Net 구조) 내부에 임베딩하는 방식이다.
2.  **Dimensional Transfer Learning (DTL, 차원 전이 학습):** 2D 인코더의 가중치를 3D 가중치로 확장(Extrapolation)하여 3D 네트워크의 인코더를 초기화하는 방식이다. 구체적으로는 2D 가중치를 3D 깊이 방향으로 반복적으로 연결(Concatenating)하여 3D 파라미터를 생성한다.

이러한 원칙을 바탕으로 각각 **Omnia-Net** (2D 및 3D 복부 장기 분할용), **DS-Net** (3D 뇌종양 분할용 - WTL 기반), **DX-Net** (3D 뇌종양 분할 분할용 - DTL 기반)이라는 세 가지 아키텍처를 제안하였다.

## 📎 Related Works

의료 영상 분할에서는 **U-Net**과 그 변형 구조(Attention U-Net, U-Net++, U-Net3+)가 지배적으로 사용되고 있다. 최근에는 EfficientNet과 같이 신경망 구조 탐색(NAS)을 통해 최적화된 백본을 사용하는 연구들이 등장하였다.

기존 연구들은 주로 다음과 같은 한계를 보였다:
- **Random Initialization:** 많은 3D 네트워크가 무작위 초기화 상태에서 학습되어 계산 시간이 오래 걸리고, 데이터가 부족한 상황에서 일반화 성능이 떨어진다.
- **Limited Transfer Learning:** TernausNet이나 v16U-Net처럼 ImageNet으로 사전 학습된 2D 가중치를 사용하는 시도가 있었으나, 이는 주로 2D 영역에 국한되었다.
- **Dimensionality Gap:** 2D 가중치를 3D로 확장하여 초기화하는 방식이 분류(Classification) 작업에서는 일부 시도되었으나, 분할(Segmentation) 작업에 적용하여 성능을 검증한 사례는 거의 없다.

본 논문은 이러한 간극을 메우기 위해 2D $\rightarrow$ 3D로의 차원 간 가중치 전이를 제안하며, 특히 Noisy Student 학습법으로 강화된 가중치를 활용한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 네트워크 설계

본 연구는 **EfficientNet**을 기본 인코더로 사용하며, 세 가지 네트워크를 제안한다.

*   **Omnia-Net:** 2D U-Net 형태의 구조이다. 사전 학습된 2D EfficientNet 인코더를 사용하며, 입력 이미지의 전체 스케일 특성을 활용하기 위해 인코더 앞단에 추가적인 합성곱 블록을 배치하였다.
*   **DS-Net (Dimensionally-Stacked Network):** 3D 데이터를 2D 데이터로 인코딩하여 압축한 후, 사전 학습된 2D EfficientNet 인코더를 통과시키고, 다시 3D 디코더를 통해 복원하는 구조이다. 즉, 3D $\rightarrow$ 2D $\rightarrow$ 3D 흐름을 가진다.
*   **DX-Net (Dimensionally-eXpanded Network):** 완전한 3D U-Net 구조이다. 하지만 인코더의 가중치를 무작위로 설정하지 않고, 사전 학습된 2D EfficientNet의 가중치를 깊이 방향으로 확장하여 초기화한 3D EfficientNet 인코더를 사용한다.

### 2. 가중치 전이 및 확장 절차 (DTL)

DX-Net에서 사용하는 **Dimensional Transfer**의 핵심은 다음과 같다. 2D 필터 가중치가 $\mathbf{W}_{2D} \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$일 때, 이를 3D 필터 $\mathbf{W}_{3D} \in \mathbb{R}^{d \times k \times k \times C_{in} \times C_{out}}$로 변환하기 위해 2D 가중치를 깊이(depth) 차원에 대해 반복적으로 복사하여 배치한다. 이를 통해 2D에서 학습된 풍부한 특징 추출 능력을 3D 공간으로 전이시킨다.

### 3. 학습 절차 및 손실 함수

학습에는 **Nadam Optimizer**를 사용하였으며, 학습률(Learning Rate)은 초기값에서 시작하여 매 에폭마다 5%씩 감소시키는 전략을 사용하였다.

손실 함수로는 클래스 불균형 문제를 해결하기 위해 **Dice Loss**와 **Binary Cross-Entropy (BCE) Loss**를 결합한 복합 손실 함수(Compound Loss)를 사용하였다.

$$L_{loss} = 1 - \frac{2}{N} \sum_{n=0}^{N} \frac{\sum_{i=0}^{I} y_{i,n} g_{i,n}}{\sum_{i=0}^{I} y_{i,n} + \sum_{i=0}^{I} g_{i,n} + \epsilon} - \frac{1}{N} \sum_{n=0}^{N} \sum_{i=0}^{I} [y_{i,n} \log(g_{i,n}) + (1 - y_{i,n}) \log(1 - g_{i,n})]$$

여기서 $N$은 클래스 수, $I$는 공간 좌표의 총합, $y_{i,n}$은 네트워크의 출력값, $g_{i,n}$은 이진 Ground Truth를 의미하며, $\epsilon$은 0으로 나누는 것을 방지하기 위한 작은 상수이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** 
    - **CAMUS:** 2D 심초음파 영상 (좌심실 분할)
    - **CHAOS:** 3D 복부 CT 및 MR 영상 (간, 신장, 비장 분할)
    - **BraTS 2022:** 3D 다중 모달리티 뇌종양 MR 영상 (WT, TC, ET 분할)
- **평가 지표:** Dice Score, Mean Absolute Distance (MAD), Hausdorff Distance (HD), Relative Absolute Volume Difference (RAVD), Average Symmetric Surface Distance (ASSD), Maximum Symmetric Surface Distance (MSSD).

### 2. 정량적 결과
- **CAMUS Challenge:** Omnia-Net이 **1위**를 기록하며 기존 state-of-the-art를 능가하였다. 특히 End-diastole 단계의 내막(Endocardium) Dice 점수에서 기존 대비 +0.6% 향상된 결과를 보였다.
- **CHAOS Challenge:** 2D 기반 네트워크들 중 가장 우수한 성적을 거두었으며, 온라인 평가 플랫폼에서 **전체 3위**를 기록하였다. 특히 다중 모달리티 작업인 Task 1에서 타 2D 네트워크 대비 Dice 점수 기준 11% 이상의 큰 격차로 우위를 점하였다.
- **BraTS 2022:** 
    - **DS-Net (WTL):** Whole Tumor (WT) 영역에서 91.69%의 Dice Score를 달성하여 부종(Edema) 영역 묘사에 강점을 보였다.
    - **DX-Net (DTL):** Tumor Core (TC) 84.77%, Enhancing Tumor (ET) 83.88%의 점수를 기록하여, DS-Net보다 핵심 종양 영역의 정밀한 분할 성능이 더 우수함을 확인하였다.

### 3. 분석 결과 요약
| Network | Domain | Dataset | Ranking/Result |
| :--- | :--- | :--- | :--- |
| **Omnia-Net** | 2D Echo / 3D Abd | CAMUS / CHAOS | 1st (CAMUS) / 3rd (CHAOS) |
| **DS-Net** | 3D Brain Tumor | BraTS | Promising (Strong in WT) |
| **DX-Net** | 3D Brain Tumor | BraTS | Promising (Strong in TC, ET) |

## 🧠 Insights & Discussion

본 연구는 사전 학습된 2D 가중치를 고차원 의료 영상 분할에 어떻게 활용할 수 있는지에 대한 구체적인 방법론을 제시하였다.

**강점 및 분석:**
- **차원 전이의 유효성:** DX-Net의 결과는 2D 가중치를 3D로 확장하여 초기화하는 것이 무작위 초기화보다 훨씬 효율적이며, 특히 복잡한 3D 볼륨 특징을 포착하는 데 유리함을 보여준다.
- **구조적 trade-off:** DS-Net은 3D $\rightarrow$ 2D $\rightarrow$ 3D 과정을 거치므로 부종 같은 넓은 영역의 특징 추출에 유리한 반면, DX-Net은 완전한 3D 컨볼루션을 수행하므로 Tumor Core와 같은 세밀한 볼륨 특징을 포착하는 데 더 적합하다.
- **계산 효율성:** DX-Net은 DS-Net보다 이미지당 학습 시간이 짧아(1.5s vs 2.0s) 계산 효율성 면에서도 이점이 있다.

**한계 및 논의:**
- **학습 에폭의 제한:** 대부분의 실험이 100 에폭 미만으로 진행되었으며, 더 공격적인 데이터 증강(Data Augmentation)이나 장기적인 학습을 통해 성능을 더 끌어올릴 여지가 있다.
- **후처리 부재:** 본 연구는 정교한 후처리 과정 없이 모델의 순수 성능을 측정하였으므로, 최신 챌린지 우승팀들이 사용하는 후처리 기법을 적용한다면 결과가 더 개선될 가능성이 크다.

## 📌 TL;DR

본 논문은 데이터가 부족한 의료 영상 분할 문제를 해결하기 위해, 자연 이미지로 학습된 **2D EfficientNet의 가중치를 2D/3D 분할 네트워크로 전이하는 WTL(Weight Transfer) 및 DTL(Dimensional Transfer) 방법론**을 제안한다. 제안된 Omnia-Net은 CAMUS 챌린지 1위, CHAOS 챌린지 3위를 기록하였으며, DX-Net은 2D 가중치를 3D로 확장하여 뇌종양의 핵심 영역을 정밀하게 분할하는 성과를 거두었다. 이 연구는 특히 3D 의료 영상 학습 시 발생하는 막대한 데이터 및 계산 비용 문제를 사전 학습된 2D 지식의 전이로 해결할 수 있음을 입증하여, 향후 고차원 의료 영상 분석 연구에 중요한 방향성을 제시한다.