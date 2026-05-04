# UD-Mamba: A pixel-level uncertainty-driven Mamba model for medical image segmentation

Weiren Zhao, Feng Wang, Yanran Wang, Yutong Xie, Qi Wu, and Yuyin Zhou (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 Mamba 프레임워크가 가진 한계점을 해결하고자 한다. Mamba는 선형 계산 복잡도로 긴 범위의 의존성(long-range dependencies)을 캡처할 수 있는 효율적인 State Space Model(SSM)이지만, 전통적인 위치 기반 스캐닝(location-based scanning) 방식은 이미지의 서로 다른 의미론적 영역을 간헐적으로 스캔하는 경향이 있다.

특히 의료 영상은 배경이 복잡하고 경계선이 모호한 경우가 많기 때문에, 이러한 기존의 스캔 방식으로는 세밀한 국소 특징(local features)을 정확하게 모델링하는 데 어려움이 있다. 따라서 본 연구의 목표는 픽셀 수준의 불확실성(uncertainty)을 스캐닝 메커니즘에 도입하여, 중요한 영역을 우선적으로 처리함으로써 분할 정밀도를 높이는 UD-Mamba 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **채널 불확실성(Channel Uncertainty)을 기반으로 픽셀 스캔 순서를 재정의**하는 것이다. 단순히 위치 순서대로 스캔하는 것이 아니라, 정보의 불확실성이 높은 영역(주로 객체의 경계나 전경)을 식별하여 이를 효율적으로 처리하도록 설계하였다.

주요 기여 사항은 다음과 같다:

1. **Uncertainty-Driven Selective Scanning Model (UD-SSM)**: 채널 불확실성을 가이드로 사용하여 픽셀 스캔 순서를 결정하는 새로운 메커니즘을 제안하였다.
2. **이중 스캐닝 전략**: 고불확실성 영역을 집중적으로 모델링하는 Sequential Scanning과 배경-전경 간의 상호작용을 돕는 Skip Scanning을 도입하였다.
3. **학습 가능한 가중치 및 일관성 손실**: 서로 다른 스캔 방향의 중요도를 조절하는 4개의 학습 가능 파라미터를 도입하고, 전방 및 후방 스캔 결과의 정렬을 위해 Cosine Consistency Loss를 적용하였다.

## 📎 Related Works

### 1. 의료 영상 분할 (Medical Image Segmentation)

기존에는 CNN 기반의 UNet과 그 변형 모델(UNet++, Att-UNet)이 주를 이루었으나, 수용 영역(receptive field)의 제한으로 인해 전역적 문맥(global context) 캡처에 한계가 있었다. 이를 해결하기 위해 TransUNet, Swin-UNet 등 Transformer 기반 모델이 등장하였으나, 입력 크기에 따른 이차 복잡도(quadratic complexity)로 인해 고해상도 의료 영상 처리 시 계산 비용이 매우 높다는 단점이 있다.

### 2. 분할을 위한 State Space Models (SSMs)

최근 Mamba 아키텍처가 선형 복잡도로 전역 문맥을 모델링할 수 있어 U-Mamba, Swin-UMamba 등 Mamba 기반의 분할 모델들이 제안되었다. 그러나 이들은 여전히 전통적인 위치 기반 스캔 방식을 사용하여, 복잡한 배경과 모호한 경계를 가진 의료 영상에서 국소 특징을 일관되게 캡처하지 못하는 한계가 있다.

### 3. 분할에서의 불확실성 추정 (Uncertainty Estimation)

불확실성 추정은 모델의 신뢰성을 높이고 노이즈 섞인 라벨의 영향을 줄이는 데 사용되어 왔다. 본 논문은 이러한 불확실성 개념을 단순한 결과 분석이나 라벨 정제가 아닌, Mamba의 **입력 데이터 스캔 순서를 결정하는 가이드**로 사용했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

UD-Mamba는 기본적으로 UNet의 인코더-디코더 구조를 따른다. 입력 이미지는 Patch Embedding 레이어를 통해 시퀀스로 변환되며, 인코더와 디코더는 **UD-Block**으로 구성된다. 각 UD-Block 내에서는 Layer Normalization, Linear layer, Depthwise Convolution(DW-Conv), 그리고 핵심 모듈인 **UD-SSM**이 순차적으로 적용된다.

### 2. UD-SSM (Uncertainty-Driven Selective Scanning Model)

#### (1) 채널 불확실성 계산 (Channel Uncertainty Computation)

입력 특징 텐서 $X \in \mathbb{R}^{C \times H \times W}$에 대해, 각 픽셀 위치 $(h, w)$에서의 채널 간 표준편차(Standard Deviation, STD)를 사용하여 불확실성 맵 $U$를 계산한다.

$$U_{h,w} = \sqrt{\frac{1}{C} \sum_{c=1}^{C} (X_{c,h,w} - \mu_{h,w})^2}$$

여기서 $\mu_{h,w}$는 해당 위치의 채널 평균이다. 표준편차가 높을수록 해당 픽셀은 전경이나 경계선일 가능성이 높고, 낮을수록 배경일 가능성이 높다고 판단한다. 이후 불확실성 값을 기준으로 픽셀들을 내림차순 정렬하여 특징 맵을 재배치한다.

#### (2) 스캔 확장 작업 (Scan Extension Operation)

재배치된 특징 맵 $X'$에 대해 두 가지 스캔 전략을 적용하며, 각각 전방(고 $\to$ 저 불확실성)과 후방(저 $\to$ 고 불확실성) 방향으로 수행하여 총 4개의 경로를 생성한다.

* **Sequential Scanning ($\text{Scan}_{se}$)**: 불확실성 순서대로 엄격하게 스캔하여 경계선과 전경의 세부 정보를 밀도 있게 캡처한다.
* **Skip Scanning ($\text{Scan}_{sk}$)**: 일정 간격으로 픽셀을 샘플링하여 스캔함으로써 배경과 전경 간의 상호작용을 촉진한다.

네 가지 스캔 결과 $y_1, y_2, y_3, y_4$에 대해 학습 가능한 가중치 $\alpha_i$를 곱해 중요도를 조절한다:
$$y'_i = y_i \cdot \alpha_i \quad (i=1, 2, 3, 4)$$

#### (3) S6 블록 및 복구 (S6 block & Recovery)

가중치가 적용된 특징들은 Mamba의 S6 블록을 통과한 후, 원래의 공간적 위치로 복구(Recover)된다. 최종 출력 $y_{\text{UD-SSM}}$은 네 가지 경로의 복구된 특징들의 합으로 계산된다:
$$y_{\text{UD-SSM}} = \sum_{i=1}^{4} y_{ri}$$

### 3. 목적 함수 (Objective Function)

전방 스캔과 후방 스캔 간의 특징 표현 일관성을 높이기 위해 **Cosine Consistency Loss**를 도입하였다.

$$L_{\cos} = 1 - \frac{\text{cos\_sim}(y_{r1}, y_{r3}) + \text{cos\_sim}(y_{r2}, y_{r4})}{2}$$

최종 손실 함수는 지도 학습 손실($L_{sup}$, Cross-Entropy + Dice Loss)과 일관성 손실의 가중 합으로 정의된다:
$$L = L_{sup} + \lambda L_{\cos}$$

## 📊 Results

### 1. 실험 설정

* **데이터셋**: DigestPath (결장경 종양), ISIC 2018 (피부 병변), ACDC (심장 MRI)
* **지표**: Dice Similarity Coefficient (DSC), mean Intersection over Union (mIoU), Accuracy (ACC), Sensitivity (Sen), Specificity (Spe), 및 ACDC의 경우 $HD_{95}$ (Hausdorff Distance)
* **비교 대상**: CNN 기반 (UNet, UNet++, Att-UNet), Transformer 기반 (TransUNet, SwinUNet, H2Former), Mamba 기반 (Mamba-UNet, Swin-UMamba)

### 2. 정량적 결과

* **분할 성능**: UD-Mamba는 세 가지 데이터셋 모두에서 기존 모델들보다 우수한 성능을 보였다. 특히 ISIC 2018과 DigestPath에서 DSC 기준 CNN 대비 각각 1.68%, 2.52% 향상되었으며, Mamba-UNet보다도 mIoU가 1.23%~1.58% 높게 나타났다.
* **효율성**: ACDC 데이터셋 실험 결과, UD-Mamba는 파라미터 수(19.12M)와 FLOPs(5.91G)가 비교 대상 모델들 중 가장 낮았다. 이는 Transformer 기반 모델들의 높은 연산 비용 문제를 해결하면서도 더 높은 성능을 낼 수 있음을 시사한다.

### 3. 절제 연구 (Ablation Study)

* **스캔 전략**: 단일 스캔보다 네 가지 스캔 경로($y_1+y_2+y_3+y_4$)를 모두 결합하고 재가중치(Reweight) 및 $L_{\cos}$를 적용했을 때 DSC가 80.89%로 가장 높았다.
* **불확실성 측정 지표**: MAD, Variance, Entropy 등을 비교한 결과, 표준편차(STD)를 사용했을 때 가장 안정적인 성능을 보였다.
* **계산 영역**: 픽셀 수준의 불확실성 계산이 특정 블록 단위(region-based) 계산보다 더 정밀한 분할 결과를 생성함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 고정된 스캔 순서가 의료 영상의 국소 특징 캡처를 방해한다는 점을 정확히 짚어냈으며, 이를 '불확실성'이라는 메트릭으로 해결하려 한 점이 매우 독창적이다.

**강점**:

* **연산 효율성**: 파라미터 수와 FLOPs를 획기적으로 줄이면서도 SOTA 성능을 달성하여, 자원이 제한된 의료 환경에서의 적용 가능성을 높였다.
* **전략적 스캐닝**: 단순한 전역 모델링을 넘어, 정보 밀도가 높은 영역을 우선 처리하는 방식을 통해 Mamba의 한계를 보완하였다.

**한계 및 논의**:

* **불확실성 추정의 의존성**: 논문에서도 언급되었듯, 모델의 성능이 채널 불확실성을 얼마나 정확하게 추정하느냐에 크게 의존한다. 현재는 STD를 사용하고 있으나, 더 정교한 불확실성 추정 방법이 도입된다면 성능이 추가로 향상될 여지가 있다.
* **학습 파라미터의 경향**: $\alpha_3, \alpha_4$ (저 $\to$ 고 불확실성 스캔)의 가중치가 $\alpha_1, \alpha_2$보다 더 유지되는 경향이 관찰되었는데, 이는 배경에서 전경으로 나아가는 정보 흐름이 분할 작업에 더 유리함을 시사한다.

## 📌 TL;DR

UD-Mamba는 의료 영상의 모호한 경계 문제를 해결하기 위해 **픽셀 수준의 채널 불확실성을 기반으로 스캔 순서를 결정하는 새로운 Mamba 모델**이다. 표준편차(STD)를 이용해 중요 영역을 식별하고, Sequential 및 Skip 스캐닝을 통해 국소 및 전역 특징을 효율적으로 캡처하며, Cosine Consistency Loss로 특징의 일관성을 확보하였다. 결과적으로 기존 Transformer 및 Mamba 기반 모델보다 **훨씬 적은 연산량(FLOPs)과 파라미터로 더 높은 분할 정확도를 달성**하였으며, 이는 향후 고해상도 의료 영상 분석 연구에 중요한 효율적 아키텍처를 제시한 것으로 평가된다.
