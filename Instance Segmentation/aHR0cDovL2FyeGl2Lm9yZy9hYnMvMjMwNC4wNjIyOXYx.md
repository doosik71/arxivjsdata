# Improving Segmentation of Objects with Varying Sizes in Biomedical Images using Instance-wise and Center-of-Instance Segmentation Loss Function

Muhammad Febrian Rachmadi, Charissa Poon, Henrik Skibbe (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Biomedical Image Segmentation) 작업에서 객체들의 크기가 매우 다양할 때 발생하는 **Instance Imbalance(인스턴스 불균형)** 문제를 해결하고자 한다. 

일반적으로 의료 영상에서는 특정 클래스의 픽셀 수가 다른 클래스보다 훨씬 많은 Class Imbalance 문제가 빈번하게 논의되지만, Instance Imbalance는 동일한 클래스 내에서도 크기가 큰 인스턴스가 작은 인스턴스보다 손실 함수(Loss function)에 더 큰 영향을 주어, 결과적으로 모델이 작은 객체들을 무시하거나 제대로 탐지하지 못하게 만드는 현상을 의미한다. 특히 뇌 MRI의 뇌졸중 병변(Stroke lesion)과 같이 크기가 매우 다양한 객체를 분할해야 하는 작업에서 이 문제는 매우 중요하다.

따라서 본 논문의 목표는 픽셀 단위(Pixel-wise) 손실 함수의 한계를 극복하고, 인스턴스의 크기에 관계없이 균형 잡힌 분할 성능을 제공하는 새로운 손실 함수인 **Instance-wise and Center-of-Instance (ICI) loss**를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 인스턴스별로 손실을 계산하여 크기 차이로 인한 가중치 쏠림을 방지하고, 모든 인스턴스를 동일한 크기로 정규화하여 위치 및 존재 여부를 학습시키는 것이다. 이를 위해 다음과 같은 두 가지 핵심 구성 요소를 설계하였다.

1.  **Instance-wise loss**: 각 개별 인스턴스에 대해 독립적으로 손실을 계산함으로써 작은 객체의 탐지율을 높인다. 특히 GPU 상에서 실시간으로 Connected Component Analysis (CCA)를 수행하여, 정답(Ground Truth) 인스턴스와 실제로 겹치는 예측 인스턴스만을 고려함으로써 불필요한 오탐지(False Positive)의 영향을 줄였다.
2.  **Center-of-Instance loss**: 모든 인스턴스를 중심점(Center of mass) 기준으로 고정된 크기의 정사각형(2D) 또는 정육면체(3D) 형태로 변환하여 손실을 계산한다. 이를 통해 객체의 원래 크기와 상관없이 모든 인스턴스가 학습에 동일한 비중으로 기여하게 하여 전반적인 탐지 정확도를 향상시킨다.

## 📎 Related Works

논문에서는 Instance Imbalance 문제를 해결하려 했던 기존의 접근 방식들을 다음과 같이 설명하며 차별점을 제시한다.

-   **Inverse Weighting (IW)**: 각 인스턴스의 크기에 반비례하는 가중치를 부여하는 방식이다. 그러나 이는 결국 개별 픽셀에 가중치를 할당하는 픽셀 단위 방식이므로, 진정한 의미의 인스턴스별 계산이라고 보기 어렵다.
-   **Blob loss**: 각 인스턴스별로 손실을 계산하고 평균을 내는 방식이다. 하지만 CCA가 라벨 이미지에 대해서만 사전 계산되므로, 예측 결과에서 정답 인스턴스와 겹치지 않는 다른 잘못된 예측 블록들까지 손실 계산에 포함시키는 과도한 민감성 문제가 있다.
-   **Lesion-wise loss (LesLoss)**: 모든 인스턴스를 고정된 크기의 구(Sphere) 형태로 변환하여 계산한다. 하지만 구 형태의 변환 계산이 복잡하고, 이를 수행하기 위해 별도의 추가적인 분할 네트워크가 필요하다는 한계가 있어 범용성이 떨어진다.

## 🛠️ Methodology

### 전체 시스템 구조
제안된 ICI loss는 기존의 픽셀 단위 손실 함수와 함께 정규화(Regularization) 용도로 사용될 수 있는 복합 손실 함수(Compound loss) 형태를 띤다. 전체 손실 함수 $L$은 다음과 같이 정의된다.

$$L = a \times L_{global} + b \times L_{instance} + c \times L_{center}$$

여기서 $L_{global}$은 전역 Dice loss이며, $a, b, c$는 각각의 가중치이다. 모든 구성 요소의 기본 계산에는 다음과 같은 Dice loss 식이 사용된다.

$$\text{diceLoss} = 1 - \frac{2|y \cap y_{pred}|}{|y| + |y_{pred}|} = 1 - \frac{2TP}{2TP + FP + FN}$$

### 주요 구성 요소 설명

#### 1. Instance-wise loss ($L_{instance}$)
이 구성 요소는 주로 작은 인스턴스의 누락(False Negative)을 줄이는 데 목적이 있다.
-   **절차**:
    1.  정답 라벨($label$)과 모델의 예측 결과($output$) 모두에 대해 실시간으로 CCA를 수행하여 개별 인스턴스들을 식별한다.
    2.  정답 이미지의 각 인스턴스(Instance-of-interest)에 대해, 이와 1픽셀이라도 겹치는 예측 결과의 인스턴스들만을 추출한다.
    3.  추출된 예측 부분과 정답 인스턴스 간의 Dice loss를 계산하고, 이를 모든 정답 인스턴스 수로 나누어 평균을 낸다.
-   **특징**: Kornia 라이브러리를 수정하여 CCA 과정의 모든 그래디언트가 역전파(Back-propagation)될 수 있도록 구현하였다.

#### 2. Center-of-Instance loss ($L_{center}$)
이 구성 요소는 작은 인스턴스의 탐지력을 높이는 동시에, 무분별하게 생성되는 작은 가짜 인스턴스(False Positive)를 억제하는 역할을 한다.
-   **절차**:
    1.  CCA를 통해 계산된 각 인스턴스의 무게 중심(Center of mass)을 찾는다.
    2.  해당 중심점을 기준으로 모든 인스턴스를 고정된 크기의 정육면체(3D 기준 $7 \times 7 \times 7$ voxels)로 변환한다.
    3.  이렇게 정규화된 정답 큐브 집합과 예측 큐브 집합 간의 Dice loss를 계산한다.
-   **특징**: 구(Sphere) 대신 정육면체를 사용함으로써 GPU 상에서의 계산 복잡도를 낮추고 효율성을 높였다.

## 📊 Results

### 실험 설정
-   **데이터셋**: MICCAI 2022의 ATLAS v2.0 챌린지 데이터셋(T1w 뇌 MRI 및 뇌졸중 병변 마스크)을 사용하였다.
-   **모델**: MONAI 라이브러리의 3D Residual U-Net을 사용하였다.
-   **비교 대상**: Global Dice loss(Baseline), Blob loss.
-   **지표**: Dice Similarity Coefficient (DSC), Volume Difference, Lesion-wise F1 Score, Simple Lesion Count 등을 사용하였다.

### 주요 결과
실험은 전체 이미지 분할(Whole image)과 패치 기반 분할(Patch-based) 두 가지 환경에서 수행되었다.

1.  **전체 이미지 분할**:
    -   ICI loss를 적용했을 때, 특히 최적 가중치($a=1/4, b=1/2, c=1/4$)를 사용한 경우 DSC가 가장 높게 나타났다.
    -   테스트 세트 결과, ICI loss는 DSC, Volume Difference, F1 Score에서 모두 최상위 성능(Mean Rank 1.25)을 기록하였다.
    -   반면 Blob loss는 Baseline인 Global Dice loss보다도 낮은 성능을 보이는 경우가 많았다.

2.  **패치 기반 분할**:
    -   이미지 크기가 달라져도 ICI loss는 Baseline 및 Blob loss보다 일관되게 우수한 성능을 보였다.
    -   다만, 최적 가중치는 입력 이미지의 크기와 작업에 따라 달라질 수 있음을 확인하였다.

3.  **정성적 분석**:
    -   훈련 곡선 분석 결과, ICI loss는 Global Dice loss만 사용했을 때보다 더 빠르게 수렴하며, 특히 누락된 인스턴스(Missed Instances)와 가짜 인스턴스(False Instances)의 수를 낮고 안정적으로 유지하는 경향을 보였다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문은 픽셀 단위 손실 함수가 가진 고질적인 문제인 '큰 객체 편향'을 인스턴스 단위의 접근 방식으로 효과적으로 해결하였다. 특히 정답 인스턴스와 겹치는 예측값만을 선택적으로 계산하는 Instance-wise loss와 크기를 완전히 정규화하는 Center-of-Instance loss의 조합이 상호보완적으로 작용하여, 작은 병변의 탐지율을 획기적으로 높였다는 점이 고무적이다.

### 한계 및 논의사항
-   **계산 비용**: 실시간으로 CCA를 수행해야 하므로 추가적인 계산 자원이 필요하다. 실험 결과, Dice loss(0.26초) 대비 ICI loss(2.19초)는 배치당 계산 시간이 상당히 증가하였다.
-   **하이퍼파라미터 민감도**: $a, b, c$ 가중치 및 Center-of-Instance의 큐브 크기 설정에 따라 성능 차이가 발생하며, 이는 데이터셋이나 이미지 크기에 따라 튜닝이 필요함을 시사한다.
-   **범용성**: 본 연구는 이진 분할(Binary segmentation)에 집중하였으나, 다양한 크기의 객체가 존재하는 다중 클래스 분할(Multi-class segmentation) 문제에서도 동일한 효과가 있을지는 추가 연구가 필요하다.

## 📌 TL;DR

이 논문은 의료 영상 분할 시 발생하는 **Instance Imbalance** 문제를 해결하기 위해, 인스턴스별로 손실을 계산하는 **Instance-wise loss**와 모든 인스턴스를 고정 크기로 정규화하여 계산하는 **Center-of-Instance loss**를 결합한 **ICI loss**를 제안한다. 뇌졸중 병변 분할 실험 결과, 기존 Dice loss 및 Blob loss 대비 DSC 성능이 유의미하게 향상되었으며, 특히 크기가 작은 객체들의 탐지 성능을 크게 개선하였다. 이는 다양한 크기의 객체가 혼재된 의료 영상 분석 분야에서 매우 유용한 정규화 도구가 될 가능성이 높다.