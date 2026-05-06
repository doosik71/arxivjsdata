# Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model

Haobo Yuan, Xiangtai Li, Lu Qi, Tao Zhang, Ming-Hsuan Yang, Shuicheng Yan, Chen Change Loy (2024)

## 🧩 Problem to Solve

본 논문은 Segment Anything Model (SAM)이 가진 두 가지 핵심적인 한계점을 해결하고자 한다. 첫째는 연산 비용의 문제이다. SAM은 Transformer 기반의 구조를 사용하므로, 입력 이미지의 해상도가 높아질수록 연산 복잡도가 제곱으로 증가하여 실시간 추론에 어려움이 있다. 둘째는 세그멘테이션 품질의 문제이다. SAM은 종종 지나치게 매끄러운(overly smooth) 경계선을 생성하여, 정밀한 디테일이 필요한 고품질 세그멘테이션 작업에서 한계를 보인다.

기존 연구들은 효율성을 높이기 위해 모델 크기를 줄이거나(EfficientSAM), 품질을 높이기 위해 추가적인 연산 비용을 감수하는(HQ-SAM) 등 두 문제 중 하나에만 집중하는 경향이 있었다. 따라서 본 연구의 목표는 고해상도 이미지에서도 효율적인 추론 속도를 유지하면서 동시에 정밀한 마스크를 생성할 수 있는 고품질-고효율의 Segment Anything 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 quadratic complexity 문제를 해결하기 위해 Linear Attention 아키텍처인 RWKV를 도입하는 것이다. 특히, 단순히 모델 전체를 RWKV로 대체하는 것이 아니라, Convolution과 RWKV 연산을 혼합한 Mixed Backbone을 설계하여 국소적 특징(local features)과 전역적 문맥(global context)을 동시에 포착하도록 하였다.

또한, 백본에서 추출된 다중 스케일(multi-scale) 토큰을 효율적으로 활용하는 디코더를 설계하고, 정밀하게 레이블링된 고품질 데이터셋을 활용한 훈련 파이프라인을 구축함으로써 효율성과 품질이라는 두 마리 토끼를 모두 잡고자 하였다.

## 📎 Related Works

기존의 효율적인 세그멘테이션 연구들은 주로 특정 도메인이나 폐쇄 집합(closed-set) 문제에 집중해 왔으며, SAM의 효율성을 높이려는 시도(Mobile SAM, Fast SAM, EfficientSAM 등)는 주로 모델 크기를 줄이는 방식에 치중하여 고해상도 입력 시의 연산 복잡도 문제를 완전히 해결하지 못했다.

최근 컴퓨터 비전 분야에서는 Mamba나 RWKV와 같은 Linear Attention 모델들이 Transformer를 대체하여 긴 시퀀스를 효율적으로 처리하려는 시도가 늘고 있다. 그러나 이러한 모델들을 SAM과 같은 프롬프트 기반의 세그멘테이션 작업에 적용하여 성능과 효율성을 정밀하게 분석한 연구는 이전까지 없었다. 본 논문은 특히 Mamba보다 고해상도 입력에서 더 빠른 추론 속도를 보이는 RWKV의 특성에 주목하여 차별점을 두었다.

## 🛠️ Methodology

### 1. Efficient Segmentation Backbone

RWKV-SAM의 백본은 총 3단계(Stage)로 구성되며, 연산 효율성과 특징 추출 능력을 최적화하기 위해 MBConv와 VRWKV 블록을 혼합하여 사용한다.

- **Macro-Level Design**: Stage 1과 Stage 2에서는 MBConv(Mobile Convolution) 블록을 사용하여 고해상도 특징 맵을 생성하고, 각 단계마다 해상도를 $2\times$ 다운샘플링한다. Stage 3에서는 VRWKV(Vision-RWKV) 블록을 쌓아 전역적인 인지 능력을 확보한다. 이러한 다중 스케일 구조는 모델이 다양한 크기의 객체와 세부 디테일에 적응적으로 집중할 수 있게 한다.
- **Micro-Level Design (VRWKV Block)**: VRWKV 블록은 Spatial-mix 모듈과 Channel-mix 모듈로 구성된다.
  - **Spatial-mix**: $\text{Q-Shift}$ 모듈을 통해 각 토큰이 4방향 인접 픽셀과 보간(interpolate)되도록 하여 이미지의 지역성(locality)을 유지한다. 이후 $\text{Bi-WKV}$ 메커니즘을 통해 모든 토큰 간의 전역적 상호작용을 선형 복잡도로 수행한다.
  - **Channel-mix**: 각 토큰에 대해 독립적으로 계산되며, MLP와 유사하지만 $\text{Q-Shift}$를 추가하여 지역성을 한 번 더 강화한다.

### 2. Mask Decoder 및 특징 융합

정밀한 마스크 생성을 위해 백본의 서로 다른 해상도 특징을 융합한다. Stage 1, 2에서 출력된 저수준 국소 특징($X^{hr}, X^{mr}$)과 Stage 3의 전역 특징($X$)을 사용한다.
기존 SAM의 디코더($\Phi_{dec}$)가 생성한 초기 마스크 특징 $F_M$을 다음과 같이 정교화(refine)한다.
$$F'_M = \Phi'_{dec}(F_M, X, X^{mr}, X^{hr})$$
여기서 $\Phi'_{dec}$는 단순하고 효율적인 두 개의 Convolution 레이어를 사용하여 서로 다른 스케일의 특징들을 융합하며, 최종 마스크 $M$은 인스턴스 쿼리 $Q$와 정교화된 특징 $F'_M$의 내적($\otimes$)으로 생성된다.

### 3. Training Pipeline

훈련은 총 2단계로 진행된다.

- **Step 1 (Distillation)**: 원본 SAM(ViT-H)의 지식을 전수받기 위해, 효율적인 백본의 출력이 ViT-H의 출력과 일치하도록 MSE Loss를 사용하여 증류(distillation) 학습을 수행한다.
  $$L_{S1} = \text{MSE}(X_{SAM}, X)$$
- **Step 2 (Joint Training)**: 고품질 데이터셋(COCONut-B, EntitySeg, DIS5K)을 사용하여 전체 모델을 학습시킨다. 이때 마스크의 정확도를 높이기 위해 Cross Entropy (CE) Loss와 Dice Loss를 결합하여 사용한다.
  $$L_{S2} = \lambda_{ce}L_{ce} + \lambda_{dice}L_{dice}$$

## 📊 Results

### 실험 설정

- **데이터셋**: COCO, DIS, COIFT, HR-SOD 등 다양한 벤치마크를 사용하였다.
- **지표**: COCO에서는 mAP와 mBAP를, 단일 객체 세그멘테이션 데이터셋에서는 mIoU와 mBIoU(경계선 IoU)를 측정하였다.
- **비교 대상**: SAM, EfficientSAM, HQ-SAM, 그리고 Mamba 기반의 VMamba-S, VRWKV-S, Vim-S 등이 비교 대상이 되었다.

### 주요 결과

- **추론 효율성**: $1024 \times 1024$ 해상도 입력 시, RWKV-SAM은 SAM 대비 약 $1/16$의 추론 시간만을 소요하며, EfficientSAM보다도 2배 이상 빠른 FPS(40.3)를 기록하였다. 특히 입력 해상도가 증가함에 따라 ViT 기반 모델은 지연 시간이 제곱으로 증가하는 반면, RWKV-SAM은 선형적으로 증가함을 입증하였다.
- **세그멘테이션 품질**: 고품질 데이터셋(DIS, COIFT, HRSOD)에서 RWKV-SAM은 SAM과 EfficientSAM을 압도하였으며, HQ-SAM과 대등하거나 더 나은 디테일을 보여주었다. 이는 백본의 저수준 지역 특징을 디코더에서 효과적으로 활용했기 때문이다.
- **타 모델 비교**: 동일 스케일의 Mamba 기반 모델(Vim-S 등)보다 더 적은 파라미터 수로도 더 높은 ImageNet 분류 정확도와 ADE20K 시맨틱 세그멘테이션 성능을 보였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 Linear Attention 아키텍처 중에서도 특히 RWKV가 고해상도 이미지 처리에서 Mamba보다 더 효율적이며 성능이 우수함을 실험적으로 증명했다는 점이다. 또한, 단순한 모델 경량화가 아니라 $\text{Convolution} \rightarrow \text{RWKV}$로 이어지는 계층적 구조를 통해 계산 효율성과 세부 디테일 보존이라는 상충하는 목표를 동시에 달성하였다.

다만, 한계점으로는 프롬프트 기반 방식의 특성상 Mask2Former와 같은 인스턴스 세그멘테이션 모델처럼 스스로 객체를 제안(propose)하거나 인식하는 능력은 부족하다는 점이 언급되었다. 또한, 아주 얇은 선 형태의 객체나 일부 부분 세그멘테이션(part-level)에서는 여전히 실패 사례가 존재한다. 이는 향후 더 방대한 데이터셋(예: 전체 SA-1B)을 통한 추가 학습으로 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

RWKV-SAM은 Transformer의 제곱 복잡도 문제를 해결하기 위해 **Linear Attention 모델인 RWKV와 Convolution을 결합한 효율적인 백본**을 제안한 모델이다. 이 모델은 고해상도 이미지에서도 **선형적인 연산 복잡도**를 유지하여 추론 속도를 획기적으로 높였으며, 다중 스케일 특징 융합과 고품질 데이터 학습을 통해 **SAM보다 정밀한 경계선 추출 능력**을 갖추었다. 결과적으로 효율성과 품질의 균형을 맞춘 새로운 SAM baseline을 제시하였으며, 이는 향후 실시간 고정밀 인터랙티브 세그멘테이션 도구 개발에 중요한 역할을 할 것으로 기대된다.
