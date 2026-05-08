# VM-UNet: Vision Mamba UNet for Medical Image Segmentation

Jiacheng Ruan, Jincheng Li, and Suncheng Xiang (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서는 그동안 CNN 기반 모델과 Transformer 기반 모델이 널리 사용되어 왔다. 그러나 이 두 방식은 각각 명확한 한계점을 가지고 있다. CNN 기반 모델은 수용 영역(receptive field)이 국소적이기 때문에 장거리 의존성(long-range dependencies)을 포착하는 능력이 부족하며, 이는 결과적으로 불충분한 특징 추출로 이어져 분할 성능을 저하시킨다. 반면, Transformer 기반 모델은 글로벌 정보 획득 능력은 뛰어나지만, self-attention 메커니즘의 계산 복잡도가 입력 이미지 크기에 대해 제곱 비례($O(N^2)$)하기 때문에 의료 영상과 같이 조밀한 예측(dense prediction)이 필요한 작업에서 막대한 계산 비용이 발생한다.

본 논문의 목표는 이러한 CNN의 국소적 한계와 Transformer의 높은 계산 복잡도를 동시에 해결하는 것이다. 이를 위해 입력 크기에 대해 선형 복잡도(linear complexity)를 유지하면서도 강력한 장거리 모델링 능력을 갖춘 상태 공간 모델(State Space Models, SSMs)을 의료 영상 분할에 적용하여 효율적이고 효과적인 새로운 아키텍처를 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 영상 분할 분야에서 최초로 순수 SSM 기반의 U-자형 아키텍처인 **VM-UNet**을 제안했다는 점이다. 핵심 설계 아이디어는 다음과 같다.

1. **순수 SSM 기반 아키텍처**: 기존의 SSM-CNN 하이브리드 모델과 달리, 인코더와 디코더 모두에 SSM 기반의 VSS(Visual State Space) 블록을 배치하여 순수 SSM 모델의 잠재력을 탐색하였다.
2. **비대칭 인코더-디코더 구조**: 계산 비용을 절감하기 위해 대칭 구조 대신 강한 인코더와 상대적으로 가벼운 디코더를 사용하는 비대칭 설계를 채택하였다.
3. **기초 베이스라인 수립**: 순수 SSM 기반 분할 모델의 가장 기본적인 형태를 구현함으로써, 향후 더 효율적인 SSM 기반 의료 영상 분석 시스템 개발을 위한 기준점(baseline)을 제공하였다.

## 📎 Related Works

본 논문은 의료 영상 분할 방법론을 세 가지 범주로 나누어 설명한다.

* **CNN 기반 모델**: UNet, UNet++, Attention-UNet 등이 대표적이다. 단순한 구조와 확장성이 뛰어나지만, 근본적으로 장거리 의존성을 모델링하는 데 어려움이 있다.
* **Transformer 기반 모델**: ViT를 도입한 TransUNet이나 Swin Transformer를 결합한 Swin-UNet 등이 있다. 글로벌 정보 포착 능력은 우수하나, 입력 크기에 따른 제곱 복잡도로 인해 계산 부담이 매우 크다.
* **SSM 기반 모델**: 최근 Mamba와 같은 모델이 선형 복잡도로 긴 시퀀스를 모델링할 수 있음을 보여주었다. 의료 영상 분야에서는 U-Mamba나 SegMamba 같은 하이브리드 모델이 등장하였으나, 본 논문 이전까지 순수 SSM 기반의 분할 모델에 대한 탐구는 이루어지지 않았다.

VM-UNet은 기존 하이브리드 방식에서 벗어나 순수하게 SSM의 능력을 활용하여 글로벌 컨텍스트를 캡처하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. State Space Model (SSM) 기초

VM-UNet의 근간이 되는 SSM은 1차원 입력 함수 $x(t)$를 중간 상태 $h(t)$를 거쳐 출력 $y(t)$로 매핑하는 연속 시스템이다. 이는 다음과 같은 선형 상미분 방정식(ODE)으로 표현된다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A$는 상태 행렬, $B$와 $C$는 투영 파라미터이다. 딥러닝 적용을 위해 zero-order hold (ZOH) 규칙을 사용하여 이 연속 시스템을 이산화하며, 이산화된 파라미터 $\bar{A}$와 $\bar{B}$는 다음과 같이 정의된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

이산화 후의 연산은 선형 재귀(linear recurrence) 또는 글로벌 컨볼루션(global convolution) 형태로 계산될 수 있다.

### 2. VM-UNet 아키텍처

VM-UNet은 Patch Embedding, Encoder, Decoder, Final Projection 및 Skip Connection으로 구성된다.

* **Patch Embedding**: 입력 이미지($H \times W \times 3$)를 $4 \times 4$ 크기의 겹치지 않는 패치로 나누고 채널 수를 $C$(기본값 96)로 매핑한다.
* **Encoder**: 4단계의 스테이지로 구성되며, 각 스테이지 끝에 Patch Merging 연산을 통해 해상도를 낮추고 채널 수를 $[C, 2C, 4C, 8C]$로 증가시킨다. 각 스테이지는 2개의 VSS 블록을 포함한다.
* **Decoder**: 마찬가지로 4단계로 구성되며, Patch Expanding 연산을 통해 해상도를 복구하고 채널 수를 $[8C, 4C, 2C, C]$로 감소시킨다. 각 스테이지는 $[2, 2, 2, 1]$개의 VSS 블록을 사용한다.
* **Skip Connection**: 추가 파라미터를 도입하지 않기 위해 단순 덧셈(additive operation) 방식을 채택하였다.
* **Final Projection**: 최종적으로 패치 확장과 투영 레이어를 통해 타겟 분할 맵의 크기로 복원한다.

### 3. VSS Block 및 SS2D

VSS(Visual State Space) 블록은 VM-UNet의 핵심 모듈이다.

1. 입력은 Layer Normalization 후 두 갈래로 나뉜다.
2. **첫 번째 경로**: Linear layer $\to$ Activation 함수를 거친다.
3. **두 번째 경로**: Linear layer $\to$ Depthwise Separable Convolution $\to$ Activation 함수 $\to$ **SS2D(2D-Selective Scan)** 모듈을 거쳐 특징을 추출한다.
4. 두 경로의 출력은 Element-wise product로 결합된 후, 다시 Linear layer와 잔차 연결(residual connection)을 통해 최종 출력된다.

**SS2D 모듈**은 이미지를 네 가지 방향(좌상$\to$우하, 우하$\to$좌상, 우상$\to$좌하, 좌하$\to$우상)으로 펼치는 **Scan Expanding** 작업을 수행하고, 이를 **S6 블록**(입력에 따라 파라미터를 조정하는 선택적 메커니즘)으로 처리한 뒤, 다시 원래 크기로 합치는 **Scan Merging** 과정을 거친다.

### 4. 손실 함수

작업의 성격에 따라 다음과 같은 손실 함수를 사용한다.

* **이진 분할 (Binary Segmentation)**: $\mathcal{L}_{BceDice} = \lambda_1 \mathcal{L}_{Bce} + \lambda_2 \mathcal{L}_{Dice}$
* **다중 클래스 분할 (Multi-class Segmentation)**: $\mathcal{L}_{CeDice} = \lambda_3 \mathcal{L}_{Ce} + \lambda_4 \mathcal{L}_{Dice}$

여기서 $\mathcal{L}_{Bce}$와 $\mathcal{L}_{Ce}$는 각각 Binary/Cross Entropy 손실이며, $\mathcal{L}_{Dice}$는 다음과 같이 정의된다.
$$\mathcal{L}_{Dice} = 1 - \frac{2|X \cap Y|}{|X| + |Y|}$$

## 📊 Results

### 1. 실험 설정

* **데이터셋**: ISIC17, ISIC18 (피부 병변 분할), Synapse (복부 장기 분할).
* **평가 지표**: mIoU, DSC(Dice Similarity Coefficient), Accuracy(Acc), Sensitivity(Sen), Specificity(Spe), HD95(95% Hausdorff Distance).
* **구현 세부사항**: 이미지 크기는 $256 \times 256$(ISIC) 및 $224 \times 224$(Synapse)로 조정하였으며, AdamW 옵티마이저와 CosineAnnealingLR 스케줄러를 사용하였다. 가중치는 ImageNet-1k에서 사전 학습된 VMamba-S로 초기화하였다.

### 2. 정량적 결과

* **ISIC 데이터셋**: VM-UNet은 ISIC17과 ISIC18 모두에서 mIoU, DSC, Acc 지표에서 최상위 성능을 보였다. 특히 강력한 하이브리드 모델인 TransFuse와 비교했을 때, ISIC18에서 mIoU는 0.72%, DSC는 0.44% 향상되었다.
* **Synapse 데이터셋**: 순수 Transformer 기반인 Swin-UNet 대비 DSC가 1.95% 향상되었고, HD95는 2.34mm 감소하였다. 특히 '위(Stomach)' 장기 분할에서 DSC 81.40%를 기록하며 유의미한 성능 향상을 보였다.

### 3. 절제 연구 (Ablation Studies)

* **초기 가중치**: 사전 학습된 가중치(VMamba-S)를 사용했을 때 mIoU와 DSC가 평균적으로 각각 2.67%, 1.65% 향상되어 사전 학습의 중요성이 확인되었다.
* **드롭아웃**: 데이터셋마다 최적값이 달랐다. ISIC17은 0.0일 때, ISIC18은 0.2일 때 성능이 가장 좋았다.
* **아키텍처**: 대칭 구조보다 제안된 비대칭 구조$\{2,2,2,2 - 2,2,2,1\}$가 파라미터 수와 계산량(GFLOPs) 면에서 효율적이며 성능 또한 더 우수하였다.
* **입력 해상도**: 입력 크기를 $256 \times 256 \to 384 \times 384 \to 512 \times 512$로 늘렸을 때 오히려 성능이 하락하는 현상이 관찰되었으며, 저자들은 이에 대해 추가 연구가 필요하다고 명시하였다.

## 🧠 Insights & Discussion

### 강점 및 성과

VM-UNet은 순수 SSM 기반 모델이 의료 영상 분할에서 기존의 CNN 및 Transformer 기반 모델과 경쟁 가능한 수준의 성능을 낼 수 있음을 입증하였다. 특히 선형 복잡도를 유지하면서 글로벌 컨텍스트를 효과적으로 포착하여, 작은 타겟이나 복잡한 경계 영역에서도 강건한 분할 성능을 보였다.

### 한계 및 비판적 해석

1. **길이 일반화 문제**: SSM은 본래 연속 시스템의 이산화이므로 연속 시간 데이터에 강한 귀납적 편향(inductive bias)을 가진다. 이로 인해 학습 시퀀스 길이를 벗어난 데이터에 대한 일반화 능력이 저하될 가능성이 있다.
2. **해상도 역설**: 일반적으로 입력 해상도가 높아지면 성능이 향상되어야 하나, VM-UNet은 $512 \times 512$에서 성능이 급격히 떨어진다. 이는 SSM이 시각적 데이터의 매우 긴 시퀀스를 처리할 때 발생하는 내부적인 문제일 수 있으며, 분석이 필요한 지점이다.
3. **특정 도메인 취약성**: 피부 병변 분할에서 밝은 색 영역의 분할 실패, 털(hair)과 같은 노이즈에 민감하게 반응하는 문제가 발견되었다. 이는 SSM의 컨텍스트 판별 능력이 시각적 표현에서 특정 패턴에 과하게 반응할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 의료 영상 분할을 위해 최초의 순수 SSM 기반 모델인 **VM-UNet**을 제안하였다. 이 모델은 Mamba의 VSS 블록을 U-자형 비대칭 구조에 적용하여, Transformer의 글로벌 모델링 능력과 CNN의 효율성을 동시에 확보(선형 복잡도)하고자 하였다. 실험 결과, 피부 병변 및 복부 장기 분할에서 최신 모델들과 경쟁하거나 능가하는 성능을 보였다. 이 연구는 순수 SSM 기반 분할 모델의 베이스라인을 구축함으로써, 향후 더 효율적인 SSM 기반 의료 영상 분석 연구의 가능성을 열었다는 점에서 중요한 의미를 가진다.
