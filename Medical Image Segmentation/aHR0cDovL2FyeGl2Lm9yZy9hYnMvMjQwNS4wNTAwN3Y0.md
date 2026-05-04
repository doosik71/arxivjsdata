# HC-Mamba: VisionMamba with Hybrid Convolutional Techniques for Medical Image Segmentation

Jiashu Xu(2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 발생하는 정보 손실과 계산 효율성 문제를 해결하고자 한다. 의료 영상은 일반적으로 매우 복잡한 텍스처와 구조를 가지고 있으며, 기존의 딥러닝 모델들은 다운샘플링(downsampling) 과정에서 이미지 해상도가 낮아지고 중요한 세부 정보가 손실되는 문제에 직면해 있다.

특히, 전역적인 문맥 정보를 파악하기 위해 수용 영역(receptive field)을 넓히려는 시도가 많으나, 이는 대개 계산 비용의 증가나 해상도 저하를 동반한다. 따라서 본 연구의 목표는 의료 영상의 복잡한 구조적 특징을 유지하면서도, 계산 비용을 낮추고 넓은 범위의 문맥 정보를 효과적으로 캡처할 수 있는 새로운 분할 모델인 HC-Mamba를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 현대적인 상태 공간 모델(State Space Model, SSM)인 Mamba의 선형 복잡도 특성과 하이브리드 합성곱(Hybrid Convolution) 기술을 결합하는 것이다. 구체적인 기여 사항은 다음과 같다.

- **Dilated Convolution의 도입**: 수용 영역을 확장하여 계산 비용의 증가 없이 더 광범위한 문맥 정보를 캡처한다. 특히, Dilated Convolution으로 인해 발생하는 데이터의 불연속성(voids) 문제를 Mamba의 상태 전이 능력을 통해 보완함으로써 공간적 상관관계를 강화하였다.
- **Depthwise Separable Convolution의 적용**: 일반적인 합성곱 연산을 Depthwise 및 Pointwise 합성곱으로 분해하여 모델의 파라미터 수와 연산량을 획기적으로 줄였다.
- **HC-SSM 모듈 제안**: SSM 브랜치와 HC-Conv 브랜치로 구성된 이중 경로 구조를 통해 국소적 특징과 전역적 특징을 동시에 추출하며, 이를 Channel Shuffle로 융합하여 효율적인 특징 표현을 가능하게 하였다.

## 📎 Related Works

의료 영상 분할을 위해 기존에는 주로 CNN 기반의 UNet과 그 변형 모델들이 사용되었으며, 이후 전역 정보 캡처 능력이 뛰어난 Transformer 기반의 TransUnet, Swin-UNet 등이 제안되었다. 하지만 이러한 모델들은 여전히 다운샘플링으로 인한 해상도 저하와 정보 손실 문제에서 자유롭지 못했다.

최근에는 선형 시간 복잡도로 긴 의존성을 모델링할 수 있는 상태 공간 모델(SSM) 기반의 Mamba가 주목받고 있으며, U-Mamba와 같은 연구가 SSM과 CNN을 결합하려는 시도를 보였다. 본 논문은 여기서 더 나아가 Dilated Convolution과 Depthwise Separable Convolution이라는 최적화된 합성곱 기법들을 Mamba 구조에 하이브리드로 통합함으로써, 기존 Mamba 기반 모델들보다 적은 파라미터로 경쟁력 있는 성능을 달성하려 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
HC-Mamba의 구조는 크게 **Patch Embedding layer**, **HC-SSM Block**, 그리고 **Patch Merging layer**로 구성된다. 입력 이미지는 $4 \times 4$ 크기의 패치로 분할되어 임베딩되며, 이후 4개의 스테이지를 거친다. 각 스테이지는 [2, 4, 2, 2] 개의 HC-SSM 블록으로 구성되며, 채널 수는 $[C, 2C, 4C, 8C]$ 순으로 증가한다.

### 2. HC-SSM Block 및 SS2D 모듈
HC-SSM 블록은 입력 채널을 두 개의 분기(branch)로 나누어 처리하는 구조를 가진다.

- **SSM Branch**: 입력 데이터가 Layer Normalization을 거친 후 **SS2D 모듈**로 들어간다. SS2D 모듈은 입력을 상, 하, 좌, 우 네 방향의 시퀀스로 확장(Scan expansion)하고, S6 블록을 통해 선택적 상태 공간 모델링을 수행한 후 다시 원래 차원으로 병합(Scan merging)한다. 이후 Depthwise Separable Convolution과 SiLU 활성화 함수를 통해 특징을 정제한다.
- **HC-Conv Branch**: **Dilated Convolution**을 사용하여 수용 영역을 확장한다. 특히 본 논문은 격자 효과(gridding effect)를 방지하고 공간적 연속성을 유지하기 위해 확장률(expansion rate)을 $[1, 2, 3, 1]$ 순으로 배치하는 톱니바퀴 모양(sawtooth-like) 전략을 사용한다.

두 브랜치의 출력은 채널 차원을 따라 병합(merge)되며, **Channel Shuffle** 연산을 통해 서로 다른 경로에서 추출된 특징들 간의 상호작용을 촉진한다.

### 3. 수학적 배경 및 손실 함수
모델의 기반이 되는 SSM은 다음과 같은 연속 시스템 방정식으로 정의된다.

$$\begin{aligned} h'(t) &= Ah(t) + Bx(t) \\ y(t) &= Ch(t) \end{aligned}$$

여기서 $A$는 상태 행렬, $B$와 $C$는 투영 파라미터이다. 이를 딥러닝에 적용하기 위해 시간 척도 파라미터 $\Delta$를 도입하여 이산화(discretization)하며, $\hat{A} = \exp(\Delta A)$와 같은 규칙을 통해 이산 파라미터로 변환한다.

학습을 위해 본 논문은 mIoU 손실, Dice 손실, Boundary 손실을 결합한 가중치 손실 함수를 사용한다.

$$L = w_{mIoU} \cdot L_{mo} + w_{Dice} \cdot L_{Dc} + w_{Boundary} \cdot L_{Budr}$$

각 손실 함수의 세부 식은 다음과 같다.
- **mIoU Loss ($L_{mo}$)**: 예측 영역과 실제 영역의 교집합과 합집합의 비율을 최적화하여 겹침 정도를 평가한다.
- **Dice Loss ($L_{Dc}$)**: $1 - \frac{2|P \cap G|}{|P| + |G|}$ 식으로 정의되며, 예측 영역과 실제 영역의 유사도를 측정한다.
- **Boundary Loss ($L_{Budr}$)**: 예측 경계 $B_P$와 실제 경계 $B_G$ 사이의 유클리드 거리 $d(p, q)$의 합을 최소화하여 경계선의 정확도를 높인다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Synapse(복부 다기관 분할), ISIC2017, ISIC2018(피부 병변 분할)
- **비교 대상**: UNet, TransFuse, VM-UNet, MedMamba 등 최신 SOTA 모델 및 Mamba 기반 모델
- **평가 지표**: mIoU, Dice Similarity Coefficient (DSC), Accuracy (Acc), Hausdorff Distance (HD95)

### 2. 정량적 결과
- **피부 병변 분할 (ISIC17/18)**: HC-Mamba는 ISIC17에서 mIoU 77.88%, DSC 87.38%를 기록하여 MedMamba보다 각각 0.29%, 0.2% 높은 성능을 보였으며, UNet 대비 유의미한 향상을 달성하였다.
- **다기관 분할 (Synapse)**: DSC 기준 79.58%를 달성하여 MedMamba(79.27%)와 VM-UNet(79.08%)을 앞섰으며, 특히 HD95 지표에서 26.34로 가장 낮은 수치를 기록하여 경계 분할 능력이 뛰어남을 입증하였다.

### 3. 절제 실험 (Ablation Study)
Dilated Convolution과 Depthwise Separable Convolution의 효과를 분석한 결과, 두 기법을 모두 적용했을 때 파라미터 수가 약 13.88M으로 감소하였다. 이는 아무것도 적용하지 않았을 때(27.43M)보다 약 60% 감소한 수치임에도 불구하고 mIoU(78.42%)와 DSC(87.89%) 면에서 가장 높은 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 효율적인 시퀀스 모델링 능력과 합성곱의 국소 특징 추출 능력을 전략적으로 결합하였다. 특히 Dilated Convolution이 가진 고유의 한계인 '데이터 불연속성' 문제를 SSM의 상태 전이 능력을 통해 보완했다는 점이 기술적인 강점이다.

또한, 파라미터 효율성이 매우 뛰어나다. VM-UNet(약 30M)이나 MedMamba(약 25M)와 비교했을 때, HC-Mamba는 약 13M의 파라미터만으로 대등하거나 더 높은 성능을 낸다. 이는 모델을 경량화하면서도 성능을 유지함으로써, 저사양 의료 기기나 실시간 진단 시스템으로의 배포 가능성을 높였다는 점에서 실용적 가치가 크다.

다만, 논문에서 제시된 $\Delta$ (step size)의 구체적인 설정 값이나 학습 하이퍼파라미터에 대한 상세한 설명이 부족하여 재현성 측면에서 보완이 필요해 보인다. 또한, 다양한 의료 영상 도메인에 대한 일반화 성능을 검증하기 위해 더 많은 종류의 데이터셋으로 확장 실험을 진행할 필요가 있다.

## 📌 TL;DR

본 연구는 의료 영상 분할을 위해 Mamba(SSM) 구조에 Dilated Convolution과 Depthwise Separable Convolution을 결합한 **HC-Mamba**를 제안하였다. 이 모델은 $[1, 2, 3, 1]$ 확장률 전략을 통해 넓은 수용 영역을 확보하고 계산 비용을 획기적으로 줄였으며, 결과적으로 기존 Mamba 기반 모델 대비 파라미터 수를 약 50% 감소시키면서도 Synapse 및 ISIC 데이터셋에서 SOTA 수준의 성능을 달성하였다. 이 연구는 고성능-저비용 의료 영상 분석 모델 설계의 새로운 방향성을 제시한다.