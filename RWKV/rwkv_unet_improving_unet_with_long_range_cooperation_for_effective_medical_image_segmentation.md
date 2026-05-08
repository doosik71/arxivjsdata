# RWKV-UNet: Improving UNet with Long-Range Cooperation for Effective Medical Image Segmentation

Juntao Jiang, Jiangning Zhang, Weixuan Liu, Muxuan Gao, Xiaobin Hu, Xiaoxiao Yan, Feiyue Huang, Yong Liu (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 기존 딥러닝 모델들이 가진 구조적 한계를 해결하고자 한다. 의료 영상 분할에서는 복잡한 해부학적 구조를 정확히 파악하기 위해 전역적 문맥(Global Context) 정보와 세부적인 지역적 특징(Local Detail) 정보를 모두 포착하는 것이 필수적이다.

기존의 주요 접근 방식인 Convolutional Neural Networks (CNNs)는 고정된 크기의 커널을 사용하므로 지역적 특징 추출에는 능숙하지만, 장거리 의존성(Long-range dependencies)을 캡처하는 데 한계가 있다. 반면, Transformer 기반 모델들은 Self-attention 메커니즘을 통해 전역적 문맥을 효과적으로 파악할 수 있으나, 연산 복잡도가 입력 크기의 제곱에 비례하는 $O(N^2)$의 특성을 가져 고해상도 의료 영상 처리 시 막대한 계산 비용이 발생한다.

따라서 본 연구의 목표는 전역적 의존성을 효과적으로 캡처하면서도 계산 복잡도를 선형적으로 유지하여, 효율성과 정확성을 동시에 확보한 의료 영상 분할 모델인 RWKV-UNet을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 RNN의 선형 복잡도와 Transformer의 병렬 처리 능력을 결합한 RWKV (Receptance Weighted Key Value) 구조를 U-Net 아키텍처에 통합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **IR-RWKV (Inverted Residual RWKV) 블록 설계**: CNN의 지역적 특징 추출 능력과 RWKV의 전역적 문맥 파악 능력을 결합한 인코더 블록을 제안하였다. 이는 Inverted Residual 구조를 통해 효율적으로 차원을 확장 및 축소하며 특징 표현력을 높인다.
2. **CCM (Cross-Channel Mix) 모듈 제안**: U-Net의 Skip connection을 개선하기 위해 multi-scale 특징 융합을 수행하는 CCM 모듈을 도입하였다. 이를 통해 서로 다른 스케일의 인코더 특징들 사이에서 전역적인 채널 정보를 통합하여 모델의 표현력을 강화하였다.
3. **효율적인 디코더 및 경량화 모델**: 대형 커널($9\times9$)의 Depth-wise Convolution을 활용한 효율적인 디코더를 설계하였으며, 다양한 자원 환경에 대응할 수 있도록 RWKV-UNet-T, RWKV-UNet-S와 같은 경량 변체 모델을 함께 제시하였다.

## 📎 Related Works

### UNet 변형 모델들

U-Net은 인코더-디코더 구조와 Skip connection을 통해 의료 영상 분할의 표준이 되었다. 이후 ResNet을 인코더로 사용하거나, Attention U-Net, U-Net++, U-Net3+와 같이 Skip connection을 정교화하여 세부 정보를 보존하려는 시도가 이어졌다.

### Attention 기반 개선 및 한계

Transformer 기반 모델(TransUNet, Swin-UNet 등)은 전역적 문맥 파악 능력을 통해 성능을 높였으나, 앞서 언급한 $O(N^2)$의 연산 복잡도가 가장 큰 병목 현상으로 지적되었다. 최근에는 Mamba와 같은 State Space Models (SSM)나 RWKV와 같은 선형 어텐션(Linear Attention) 메커니즘이 등장하여, 선형 복잡도로 장거리 의존성을 해결하려는 연구가 진행되고 있다.

## 🛠️ Methodology

### 전체 시스템 구조

RWKV-UNet은 전형적인 U-shaped 아키텍처를 따르며, 전역-지역 특징을 모두 추출하는 **Effective Encoder**, 다중 스케일 정보를 융합하는 **CCM Module**, 그리고 효율적인 업샘플링을 수행하는 **Decoder**로 구성된다.

### 주요 구성 요소 및 상세 설명

#### 1. IR-RWKV Block (Inverted Residual RWKV)

인코더의 핵심 블록으로, 다음과 같은 단계로 연산이 진행된다.

1. **차원 확장**: $1\times1$ Convolution을 통해 입력 특징 맵 $X$를 더 높은 차원 $C_{mid}$로 투영한다.
   $$I_1 = \text{LayerNorm}(\text{Conv}_{1\times1}(X))$$
2. **전역 특징 추출 (Spatial Mix)**: 특징 맵을 1D 시퀀스로 펼친 후(Unfolding), Vision-RWKV의 Spatial Mix 모듈을 적용하여 전역적 의존성을 캡처한다.
   $$I_3 = \text{SpatialMix}(\text{LayerNorm}(I_2)) + I_2$$
3. **지역 특징 추출 (DW-Conv)**: 다시 2D로 복원(Folding)한 후, $5\times5$ Depth-wise Convolution을 적용하여 지역적인 세부 정보를 보강한다.
   $$I_5 = \text{DW-Conv}(I_4) + I_4$$
4. **차원 축소 및 잔차 연결**: $1\times1$ Convolution으로 원래 차원으로 되돌리고, 입력값 $X$를 더하는 Global skip connection을 적용한다.
   $$F = \text{Conv}_{1\times1}(I_5) + X$$

#### 2. Cross-Channel Mix (CCM) Module

인코더의 서로 다른 단계(Stage I, II, III)에서 나온 특징 맵들을 융합하여 Skip connection의 질을 높이는 모듈이다.

- **정렬**: 각 단계의 특징 맵 $F_1, F_2, F_3$를 가장 큰 해상도와 동일한 차원으로 업샘플링 및 투영하여 크기를 맞춘다.
- **융합**: 정렬된 특징 맵들을 채널 방향으로 연결(Concatenate)한 후, RWKV의 Channel Mix 메커니즘을 적용하여 채널 간의 전역적 정보를 융합한다.
- **복원**: 융합된 결과물을 다시 원래의 스케일과 차원으로 분리 및 복원하여 디코더의 각 단계에 전달한다.

#### 3. 효율적인 디코더 설계

디코더 블록은 $\text{Conv}_{1\times1} \rightarrow 9\times9 \text{ DW-Conv} \rightarrow \text{Conv}_{1\times1}$ 구조를 사용한다. 일반적인 Convolution 대신 Point-wise와 Depth-wise Convolution을 분리함으로써, 큰 커널($9\times9$)을 사용하면서도 연산 복잡도를 획기적으로 낮추었다.

#### 4. 학습 절차 및 손실 함수

- **사전 학습 (Pre-training)**: 인코더는 ImageNet-1K 데이터셋에서 300 epoch 동안 AdamW 옵티마이저를 사용하여 사전 학습되었다.
- **손실 함수**: Cross Entropy (CE) 손실과 Dice 손실을 결합한 Mixed Loss를 사용한다.
  $$\mathcal{L} = \alpha \text{CE}(\hat{y}, y) + \beta \text{Dice}(\hat{y}, y)$$
  ($\alpha, \beta$ 값은 데이터셋에 따라 $0.5:0.5$, $0.3:0.7$, $0.5:1$ 등으로 조정됨)

## 📊 Results

### 실험 설정

- **데이터셋**: Synapse(복부 CT), ACDC(심장 MRI), BUSI(유방 초음파), CVC-ClinicDB/ColonDB/Kvasir-SEG(폴립), ISIC 2017(피부 병변), GLAS(선 조직) 등 다양한 모달리티의 데이터셋을 사용하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC, $\uparrow$)와 95% Hausdorff Distance (HD95, $\downarrow$)를 사용하였다.
- **비교 대상**: U-Net, TransUNet, Swin-UNet, VM-UNet 등 최신 CNN 및 Transformer 기반 모델들과 비교하였다.

### 주요 결과

1. **SOTA 달성**: Synapse 데이터셋에서 평균 DSC 84.02%, HD95 15.70을 기록하며 기존의 모든 CNN 및 Transformer 기반 모델을 능가하는 성능을 보였다. 특히 가장 까다로운 췌장(Pancreas) 분할에서 최고의 성능을 보였다.
2. **심장 및 이진 분할**: ACDC 데이터셋에서 RV, LV 모든 카테고리에 대해 최고 성능을 달성하였으며, 유방 초음파 및 피부 병변 등 이진 분할 작업에서도 매우 높은 효율성과 정확도를 입증하였다.
3. **연산 효율성**: TransUNet 등 기존 Transformer 모델 대비 파라미터 수와 FLOPs가 훨씬 적으면서도 더 높은 정확도를 보였다. 특히 고해상도($512\times512$) 입력 시 연산량 증가 폭이 훨씬 완만하여 고해상도 이미지 처리에 매우 유리함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **전역-지역 협력**: IR-RWKV 블록에서 Spatial Mix(전역)와 DW-Conv(지역)를 순차적으로 배치함으로써, 의료 영상에서 중요한 구조적 맥락과 세부 경계선을 동시에 잘 잡아낼 수 있었다.
- **CCM의 효과**: Ablation study를 통해 Skip connection에 CCM 모듈을 추가하는 것이 단순 연결보다 DSC 성능을 유의미하게 향상시킴을 확인하였다. 이는 다중 스케일의 전역 정보 융합이 분할 정밀도에 직접적인 영향을 준다는 것을 시사한다.
- **사전 학습의 중요성**: ImageNet-1K 사전 학습 여부에 따른 실험 결과, 사전 학습을 진행했을 때 특징 추출 능력이 비약적으로 상승하여 수렴 속도와 최종 성능이 크게 향상되었다.

### 한계 및 향후 과제

- **2D 기반 모델**: 본 모델은 현재 2D 영상 분할에 최적화되어 있으며, 3D 볼륨 데이터(CT/MRI의 3D 스캔)에 직접 적용하는 기능은 구현되지 않았다.
- **계산 비용의 트레이드-오프**: CCM 모듈이 성능을 높이지만 전역 정보 집계 과정에서 어느 정도의 계산 비용을 증가시킨다는 점이 명시되었다.

## 📌 TL;DR

본 논문은 **RWKV의 선형 복잡도 기반 전역 모델링 능력**을 **U-Net의 계층적 구조**에 통합한 **RWKV-UNet**을 제안한다. 전역 특징을 잡는 Spatial Mix와 지역 특징을 잡는 DW-Conv를 결합한 **IR-RWKV 블록**, 그리고 다중 스케일 정보를 융합하는 **CCM 모듈**을 통해, 기존 Transformer 모델의 고비용 문제를 해결하면서도 SOTA 수준의 의료 영상 분할 성능을 달성하였다. 이 연구는 특히 고해상도 의료 영상 처리 시 계산 효율성을 획기적으로 높일 수 있어 실제 임상 환경 적용 가능성이 매우 높다.
