# Multi-scale Cell Instance Segmentation with Keypoint Graph based Bounding Boxes

Jingru Yi, Pengxiang Wu, Qiaoying Huang, Hui Qu, Bo Liu, Daniel J. Hoeppner, and Dimitris N. Metaxas (2019)

## 🧩 Problem to Solve

본 연구는 생물 의학 이미지 분석에서 매우 중요한 과제인 세포 인스턴스 분할(Cell Instance Segmentation) 문제를 해결하고자 한다. 세포 이미지에서는 경계의 낮은 대비, 배경 노이즈, 세포 간의 접촉 및 군집화 현상이 빈번하게 발생하며, 특히 서로 맞닿아 있는 세포들을 개별 객체로 분리하는 것이 매우 어렵다.

기존의 접근 방식은 크게 두 가지로 나뉜다. 첫째, Bounding Box 없이 직접 분할하는 Box-free 방식은 객체에 대한 전역적 이해(Global understanding)가 부족하여 맞닿은 세포를 과분할(Over-segmentation)하거나 제대로 분리하지 못하는 경향이 있다. 둘째, 객체 검출과 분할을 결합한 Box-based 방식은 전역적 특징을 활용해 분리 성능이 좋지만, 주로 Anchor box 기반 검출기를 사용하기 때문에 양성-음성 샘플 간의 심각한 클래스 불균형(Class imbalance) 문제로 인해 성능 저하가 발생한다.

따라서 본 논문의 목표는 Anchor box의 한계를 극복하기 위해 Keypoint 기반의 검출 방식을 도입하고, 이를 통해 정밀한 Bounding Box를 생성하여 맞닿은 세포들을 효과적으로 분리하는 새로운 인스턴스 분할 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 세포의 기하학적 구조를 대표하는 5개의 핵심 점(Keypoints)을 검출하고, 이를 그래프 구조로 연결하여 Bounding Box를 복원하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **Keypoint Graph 기반 Bounding Box 생성**: 세포 사각형의 네 모서리(Top-left, Top-right, Bottom-left, Bottom-right)와 중심점(Center)의 총 5개 점을 검출한다. 이 중 어느 세 점 혹은 대각선 방향의 두 점만으로도 Bounding Box를 복원할 수 있도록 설계하여, 일부 점이 검출되지 않더라도 박스를 성공적으로 생성할 수 있는 강건함(Robustness)을 확보하였다.
2.  **Multi-scale Feature 활용**: 다양한 크기의 세포를 효과적으로 검출하기 위해 여러 스케일의 특징 맵(Feature maps)을 사용하여 수용 영역(Receptive field)의 한계를 극복하였다.
3.  **개별 세포 분할 브랜치(Individual Cell Segmentation Branch) 설계**: 단순히 특징 맵을 재사용하는 것이 아니라, 생성된 Bounding Box로 크롭된 영역에 대해 별도의 분할 브랜치를 운영함으로써 인접 세포의 간섭을 제거하고 세포의 '객체성(Objectness)' 개념을 학습하게 하였다.

## 📎 Related Works

논문에서는 기존의 세포 인스턴스 분할 방법론들을 다음과 같이 분석하고 한계를 지적한다.

-   **Box-free Methods**:
    -   **DCAN**: 세포 윤곽선을 세그멘테이션 결과에 투영하여 인스턴스를 추출하지만, 맞닿은 세포 사이의 모호한 경계로 인해 과분할 문제가 발생한다.
    -   **StarDist**: 볼록 다각형(Convex polygons)을 사용하여 분리하지만, 세포 모양이 볼록해야 한다는 가정이 필요하다.
    -   **CosineEmbedding**: 픽셀 임베딩을 클러스터링하여 인스턴스를 구분하지만, 하나의 세포가 여러 개의 클러스터로 분리되어 위양성(False positive)이 많이 발생한다.
-   **Box-based Methods**:
    -   **Mask R-CNN 및 FCIS**: Anchor box 기반 검출기를 사용하여 전역적 특징을 활용하므로 분리 성능은 좋으나, 앞서 언급한 클래스 불균형 문제와 ROI Align 메커니즘으로 인해 길고 가느다란 세포 구조를 예측하는 데 한계가 있다.
    -   **CornerNet**: 객체의 좌상단과 우하단 두 점만으로 박스를 생성하여 Anchor box의 문제를 해결하려 했으나, 두 점 중 하나만 누락되어도 박스 제안(Proposal) 자체가 사라지는 취약점이 있다.

## 🛠️ Methodology

전체 시스템은 ResNet-50 (Conv1-4)을 백본 네트워크로 사용하며, **Keypoints Detection Branch**와 **Individual Cell Segmentation Branch**의 두 가지 경로로 구성된다.

### 1. Bounding Box Generation (Keypoints Detection Branch)

Bounding Box를 생성하는 과정은 다음의 3단계로 진행된다.

**Step 1: Keypoints Voting**
각 스케일 $i \in \{1, 2, 3, 4\}$에서 세 가지 맵을 출력한다.
-   **Heatmap $h(x)$**: 핵심 점 $y$ 주변의 반지름 $r=5$인 디스크 영역 $d^r(y)$를 1로, 나머지를 0으로 설정한 맵이다. Binary Cross Entropy 손실 함수를 통해 학습한다.
-   **Single Offset map $s(x)$**: 디스크 내부의 점 $x$와 실제 핵심 점 $y$ 사이의 거리(변위)를 저장한다.
    $$s(x) = y - x, \quad x \in d^r(y)$$
    $L_1$ 손실 함수를 사용하며, 디스크 내부에서만 역전파를 수행한다.
-   **Hough Voting**: 위 두 맵을 결합하여 최종 핵심 점 점수 맵 $h'(x)$를 생성한다.
    $$h'(x) = \frac{1}{\pi r^2} \sum_{i=1}^{N} h(x_i) B(x_i + s(x_i) - x)$$
    여기서 $B$는 Bilinear interpolation 커널이다.

**Step 2: Keypoints Grouping**
$h'(x)$에서 임계값(0.004) 이상의 피크 지점을 추출하여 후보 점들을 찾고, 이를 **Keypoint Graph**를 통해 그룹화한다.
-   **Group Offset map $g(x)$**: 두 핵심 점 $(k, l)$ 사이의 방향성 변위를 저장한다.
    $$g_{k,l}(x) = y_l - x, \quad x \in d^r(y_k)$$
    이 역시 $L_1$ 손실 함수로 학습한다.
-   **Grouping 과정**: 검출된 점들을 점수에 따라 내림차순으로 정렬한 후, $g(x)$를 이용해 탐욕적(Greedily)으로 연결하여 하나의 세포에 속하는 점들의 집합을 구성한다.

**Step 3: Bounding Box Retrieval**
그룹화된 점들 중에서 어느 3개의 점이 모였거나, 대각선 방향의 두 점이 쌍을 이룬 경우 이를 통해 Bounding Box를 복원한다. 최종적으로 NMS(Non-Maximum Suppression)를 적용하여 중복 검출을 제거한다.

### 2. Cell Segmentation (Individual Cell Segmentation Branch)

Bounding Box가 결정되면, 해당 영역의 멀티 스케일 특징 맵을 크롭(Crop)하여 개별 세포 분할을 수행한다.
-   **구조**: U-Net과 유사하게 얕은 층의 세부 정보와 깊은 층의 의미론적 정보를 결합한다.
-   **특징**: 단순히 백본의 특징 맵을 사용하는 것이 아니라 별도의 분할 브랜치를 둠으로써, 인접 세포의 간섭을 배제하고 해당 ROI 내의 객체에만 집중하도록 유도한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: 
    -   Neural Cell Dataset: 불규칙한 모양과 크기를 가진 쥐의 CNS 줄기세포 이미지 (640x512).
    -   DSB2018 Dataset: 다양한 조건에서 촬영된 규칙적인 모양의 세포 핵 이미지 (512x512).
-   **비교 대상**: DCAN, CosineEmbedding, Mask R-CNN, CornerNet.
-   **평가 지표**: Bounding Box 수준의 $AP@0.5, AP@0.7$ 및 Mask 수준의 $AP$와 $mIOU$.

### 주요 결과
1.  **정량적 성능**: 제안 방법(Ours, seg branch)이 모든 데이터셋에서 기존 방법론보다 우수한 성능을 보였다. 특히 Neural Cell 데이터셋에서 Mask R-CNN 대비 $AP@0.5$ 기준 $88.03\%$ 대 $66.02\%$로 압도적인 성능 향상을 보였다.
2.  **분할 브랜치의 효과**: 단순히 특징 맵 $s_1$에서 분할을 수행한 결과($Segs_1$)보다 별도의 분할 브랜치를 사용한 결과($Seg \text{ branch}$)가 더 좋았다. 이는 모델이 개별 세포에 대한 '객체성'을 더 잘 학습하여 인접 세포의 간섭을 효과적으로 제거했기 때문이다.
3.  **Multi-scale 검출의 중요성**: Neural Cell 데이터셋에서 단일 스케일 검출보다 멀티 스케일 검출의 성능이 훨씬 높았다. 이는 큰 세포의 경우 얕은 층(Small receptive field)에서는 인식이 어렵지만, 깊은 층(Large receptive field)에서는 효과적으로 검출되기 때문이다. (DSB2018은 세포 크기가 일정하여 스케일 영향이 적었다.)

## 🧠 Insights & Discussion

본 논문은 Keypoint 기반의 검출 방식을 인스턴스 분할에 도입하여 기존 Anchor-based 방식의 클래스 불균형 문제와 Box-free 방식의 전역 정보 부족 문제를 동시에 해결하였다.

특히 **Keypoint Graph**를 통해 5개의 점 중 일부만 검출되어도 박스를 복원할 수 있게 한 점은 CornerNet과 같은 기존 Keypoint 검출기의 취약점을 잘 보완한 설계라고 판단된다. 또한, 정성적 결과에서 Mask R-CNN이 잡지 못하는 길고 가느다란 세포 구조를 정밀하게 잡아내는 것을 통해, 제안하는 방식이 생물 의학 이미지의 특수성에 매우 적합함을 입증하였다.

다만, 본 논문에서는 5개의 특정 점(네 모서리와 중심)만을 정의하여 사용하였는데, 세포의 모양이 극단적으로 왜곡된 경우에 이 5개의 점이 여전히 최적의 대표성을 갖는지에 대한 추가적인 분석이 있었다면 더 완벽했을 것이다. 또한, 학습 과정에서 Ground-truth Bounding Box를 사용하여 분할 브랜치를 학습시켰으므로, 실제 추론 시 검출 단계에서 발생한 박스 오차가 분할 단계에 미치는 영향(Error propagation)에 대한 논의가 부족한 점은 아쉬움으로 남는다.

## 📌 TL;DR

이 논문은 세포의 다섯 가지 핵심 점(Keypoints)을 검출하고 이를 그래프로 연결하여 Bounding Box를 생성하는 **Keypoint Graph 기반의 인스턴스 분할 방법**을 제안한다. 이 방법은 Anchor box의 클래스 불균형 문제를 해결하고, 멀티 스케일 특징 맵과 전용 분할 브랜치를 통해 맞닿아 있거나 모양이 불규칙한 세포들을 매우 정밀하게 분리해낸다. 향후 다양한 생물 의학 영상 분석 및 정밀한 객체 분리가 필요한 의료 AI 분야에 핵심적인 프레임워크로 활용될 가능성이 높다.