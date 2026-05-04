# Generalisable 3D Fabric Architecture for Streamlined Universal Multi-Dataset Medical Image Segmentation

Siyu Liu, Wei Dai, Craig Engstrom, Jurgen Fripp, Stuart Crozier, Jason A. Dowling, Shekhar S. Chandra (2022)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델의 성능을 제한하는 가장 큰 요인은 전문가의 레이블링이 된 데이터의 부족, 즉 데이터 희소성(Data Scarcity) 문제이다. 이를 해결하기 위해 여러 데이터셋을 동시에 학습시키거나 전이 학습(Transfer Learning)을 활용하는 방법들이 제안되었으나, 다음과 같은 한계가 존재한다.

첫째, 의료 영상 데이터셋은 데이터셋마다 이미지의 크기와 특징(Feature)이 매우 다양하다. 기존의 모델들은 대부분 단일 데이터셋에 최적화되어 설계되었으며, nnU-Net과 같이 스스로 설정(Self-configuring)되는 모델조차 한 번에 하나의 데이터셋만을 처리하도록 최적화되어 있다. 둘째, 여러 데이터셋을 동시에 처리하려는 기존의 시도(예: 3D MDUNet, $3\text{D } U^2\text{Net}$)들은 매우 제한적인 데이터셋 하위 집합에 대해서만 적용 가능성을 보였다.

따라서 본 논문의 목표는 다양한 크기의 이미지와 특징을 가진 임의의 수의 데이터셋에 대해, 하이퍼파라미터 튜닝 없이 동시에 분할 작업을 수행하거나 전이 학습을 적용할 수 있는 범용적인 3D 아키텍처인 **FIRENet (Fabric Image Representation Encoding Network)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 'Fabric(직물)' 구조의 아키텍처를 통해 다양한 스케일의 서브 아키텍처들을 중첩(Superposition)시켜 표현하는 것이다.

1.  **Fabric Representation Module (FRM)**: 다양한 이미지 및 특징 크기에 적응하기 위해, 여러 개의 멀티 스케일 분기(Branch)와 노드(Node)로 구성된 3D Fabric 모듈을 도입하여 아키텍처 수준의 일반화 성능을 확보하였다.
2.  **ASPP3D의 통합**: 각 Fabric 노드 내에 Atrous Spatial Pyramid Pooling 3D (ASPP3D)를 적용하여, 다양한 크기의 특징을 세밀하게 포착할 수 있는 넓은 수용역(Receptive Field)을 확보하였다.
3.  **Trainable Feature Sum (TFS)**: 노드 간의 연결에 학습 가능한 가중치를 부여하여, 입력 데이터셋의 특성에 맞게 최적의 아키텍처 경로가 암시적으로 학습되도록 설계하였다.
4.  **단순화된 학습 파이프라인**: 데이터셋마다 개별적인 하이퍼파라미터 수정 없이도 전이 학습 및 다중 데이터셋 학습이 가능한 통합 네트워크를 구현하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계를 지적하며 차별점을 제시한다.

-   **CNN 기반 분할 모델**: U-Net과 같은 Encoder-Decoder 구조는 전역적 특징과 지역적 특징을 모두 학습하려 하지만, 과도한 다운샘플링은 해상도 손실을 초래한다. 이를 해결하기 위해 Shortcut 연결 등이 도입되었으나, 여전히 특정 데이터셋에 고정된 구조를 가진다.
-   **멀티 스케일 특징 추출**: ASPP나 HRNet과 같이 병렬 분기를 통해 다양한 스케일의 특징을 추출하는 방식이 제안되었다. 이는 본 연구의 FRM 설계에 영감을 주었으나, 기존 방식들은 데이터셋 간의 전이 가능성보다는 단일 데이터셋의 성능 향상에 초점이 맞춰져 있다.
-   **Self-configuring 및 NAS**: nnU-Net은 데이터셋의 기하학적 특성에 따라 모델을 자동 설정하지만, 이는 한 번에 하나의 데이터셋만을 위한 최적화이다. Neural Architecture Search (NAS) 역시 특정 데이터셋에 맞춘 컴팩트한 구조를 찾는 데 집중하므로, 다양한 데이터셋을 동시에 수용하는 범용성 측면에서는 한계가 있다.
-   **다중 데이터셋 학습**: 기존의 3D MDUNet 등은 다중 데이터셋 학습을 시도했으나, MSD(Medical Segmentation Decathlon) 챌린지의 일부 데이터셋에 대해서만 성능을 입증했을 뿐, 매우 다양한 크기의 전체 데이터셋을 아우르는 범용성은 부족하였다.

## 🛠️ Methodology

### 전체 시스템 구조
FIRENet은 기본적으로 3D CNN Encoder-Decoder 구조를 따른다. Encoder는 입력 영상을 점진적으로 다운샘플링하며 채널 수를 확장하고, Decoder는 이를 다시 복원한다. 이 구조의 병목(Bottleneck) 지점에 본 논문의 핵심인 **Fabric Representation Module (FRM)**이 위치한다.

### Fabric Representation Module (FRM)
FRM은 하이퍼파라미터 $B$(분기의 수)와 $N$(분기당 노드의 수)으로 정의된다.

1.  **멀티 스케일 분기 (Branches)**: 입력 데이터는 $B$개의 서로 다른 스케일로 리사이징되어 처리된다. 첫 번째 분기($b=1$)는 원본 크기를 유지하며, 분기 번호가 증가할수록 공간 해상도는 $2^{b-1}$ 배로 감소하고, 이를 보완하기 위해 채널 수는 $2^{b-1}$ 배로 증가한다.
2.  **Fabric 노드 ($\psi_{n,b}$)**: 각 노드는 다음과 같은 입력물을 받아 처리한다.
    -   동일 분기의 이전 노드 출력 ($\psi_{n-1,b}$)
    -   인접 분기(상위/하위)의 이전 노드 출력 ($\psi_{n-1,b-1}, \psi_{n-1,b+1}$)

### 노드 세부 구성 요소
각 노드는 다음의 세 단계 프로세스를 거친다.

1.  **Feature Alignment (FA)**: 인접 분기에서 들어온 서로 다른 해상도의 입력들을 현재 노드의 크기에 맞게 리사이징한다. 3D Convolution으로 채널 수를 맞춘 후, Trilinear up/down-sampling을 통해 공간 크기를 조정한다.
2.  **Trainable Feature Sum (TFS)**: 리사이징된 세 개의 입력값에 각각 학습 가능한 가중치 $w$를 곱하여 합산한다. 가중치는 $\text{sigmoid}$ 함수를 통해 $[0, 1]$ 범위로 정규화된다.
    $$\text{Fused Feature} = \sum_{i=1}^{3} \sigma(w_i) \cdot \text{Input}_i$$
3.  **ASPP3D**: 융합된 특징에 대해 dilation rate가 1, 2, 4인 세 개의 병렬 $3 \times 3$ dilated convolution을 적용한다. 이를 통해 매우 다양한 수용역을 확보하며, FRM의 멀티 분기 구조와 결합되어 총 $3 \times B$ 개의 고유한 수용역 크기를 가질 수 있게 된다.

### 기타 구현 세부 사항
-   **Multiple Decoders**: 데이터셋 간 클래스 중복이 적은 경우, 메모리 효율을 위해 각 데이터셋 전용 Decoder를 별도로 둔다. 학습 시에는 해당 데이터셋에 맞는 Decoder만 활성화하며, 핵심 특징 추출부인 FRM은 모든 데이터셋이 공유한다.
-   **손실 함수 및 학습**: Categorical Cross-Entropy Loss와 Dice Similarity Coefficient (DSC) Loss를 결합하여 사용한다. 또한, FRM 출력과 중간 Decoder 블록의 출력을 이용해 보조 분할 맵을 생성하는 **Deep Supervision** 기법을 적용하여 전체 손실 함수를 구성한다.
-   **학습 환경**: Adam Optimizer를 사용하였으며, NVIDIA Tesla V100 (32GB)에서 배치 사이즈 1로 약 36시간 동안 50 epoch 학습을 진행하였다.

## 📊 Results

### 실험 I: 다중 데이터셋 전이 학습 (Multi-dataset Transfer Learning)
전립선(Prostate) MR 데이터셋으로 사전 학습한 후, 4가지 골격(Bone) 데이터셋(Knee, Shoulder, Hip, OAI Knee)으로 구성된 복합 데이터셋에 적용하였다.

-   **사전 학습 성능**: 전립선 데이터셋에서 3D UNet, VNet 등 대비 DSC, HD, MSD 지표에서 우수한 성능을 보였으며, 특히 이상치(Outlier) 케이스에 대한 회복력이 높았다.
-   **전이 학습 효과**: 무작위 초기화 모델(FIRENet-R)보다 사전 학습 모델(FIRENet-T)이 모든 골격 데이터셋에서 더 높은 DSC를 기록했으며, 수렴 속도가 훨씬 빨랐다.
-   **Baseline 비교**: nnU-Net은 단일 데이터셋 최적화에는 강하지만, 다중 데이터셋을 동시에 처리할 때 매우 불안정하며 많은 경우 분할에 실패(Segmentation Failure)하였다. 반면 FIRENet은 일관되게 높은 성능을 유지하였다.

### 실험 II: MSD 챌린지 다중 데이터셋 분할
MSD의 10개 데이터셋 전체를 동시에 분할하는 실험을 수행하였다.

-   **정량적 결과**: 3D MDUNet 및 $3\text{D } U^2\text{Net}$과 비교했을 때, Pancreas(췌장), Spleen(비장), Hippocampus(해마) 등에서 눈에 띄는 DSC 향상을 보였다. 특히 췌장 분할에서는 $3\text{D } U^2\text{Net}$ 대비 DSC가 12.1% 향상되었다.
-   **범용성 확인**: 이미지 크기가 최대 4배까지 차이 나는 매우 다양한 데이터셋들에 대해 하이퍼파라미터 수정 없이 하나의 통합 네트워크로 준수한 결과를 얻었다.

## 🧠 Insights & Discussion

**강점 및 기여**
FIRENet의 가장 큰 강점은 **아키텍처 수준의 일반화(Architecture-level Generalisability)**이다. FRM은 수많은 가능한 서브 아키텍처들의 중첩으로 작동하므로, 데이터셋의 특성에 맞춰 모델이 스스로 최적의 경로를 선택하는 효과를 낸다. 이는 의료 영상 데이터의 고질적인 문제인 데이터 희소성과 데이터셋 간의 이질성 문제를 구조적으로 해결한 접근 방식이다.

**한계점**
1.  **특수 최적화의 부재**: 모든 데이터셋에 적용 가능한 범용 구조를 지향하므로, 특정 단일 데이터셋에 극도로 최적화된 전문 모델(Specialized model)보다는 수치적 성능이 낮을 수 있다.
2.  **모델 크기 증가**: 데이터셋의 수가 늘어날수록 출력 채널을 맞추기 위한 개별 Decoder의 수가 증가하여 전체 모델 파라미터가 늘어난다.
3.  **CNN의 내재적 한계**: 딥러닝 모델 특성상 임상적 설명 가능성(Explainability)이 부족하며, 학습 데이터 분포를 벗어난 데이터에 대한 외삽(Extrapolation) 능력이 제한적이다.

**비판적 해석**
본 연구는 "하나의 모델이 여러 데이터셋을 동시에 학습할 수 있는가"에 대한 긍정적인 답변을 제시하였다. 특히 nnU-Net과 같은 강력한 Baseline이 다중 데이터셋 환경에서 무너지는 모습을 통해, 단순한 데이터 통합이 아니라 '다양한 스케일을 수용할 수 있는 유연한 구조'가 다중 데이터셋 학습의 핵심임을 입증하였다.

## 📌 TL;DR

본 논문은 의료 영상의 데이터 희소성 문제를 해결하기 위해, 다양한 크기의 이미지와 특징을 동시에 처리할 수 있는 범용 3D 아키텍처인 **FIRENet**을 제안한다. 핵심 요소인 **Fabric Representation Module (FRM)**과 **ASPP3D**를 통해 하이퍼파라미터 수정 없이도 다중 데이터셋 학습 및 전이 학습이 가능함을 보였으며, 특히 MSD 챌린지의 10개 데이터셋 동시 분할 실험을 통해 그 범용성과 성능을 입증하였다. 이 연구는 향후 3D 의료 영상 분석을 위한 범용 백본(Backbone) 네트워크로서 높은 활용 가능성을 가진다.