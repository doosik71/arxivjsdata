# CV 3315 Is All You Need – Semantic Segmentation Competition

Akide Liu, Zihan Wang (2022)

## 🧩 Problem to Solve

본 논문은 차량 카메라 뷰를 기반으로 한 도시 환경의 Semantic Segmentation(의미론적 분할) 문제를 해결하고자 한다. 특히, 제공된 Urban-Sense 이미지 데이터셋의 클래스 분포가 매우 불균형(highly unbalanced)하다는 점이 기존 솔루션들에 큰 도전 과제가 된다.

연구의 주요 목표는 모델의 예측 정확도를 나타내는 $\text{mIoU}$(mean Intersection over Union)와 연산 효율성을 나타내는 $\text{FLOPs}$(Floating Point Operations per Second) 사이의 최적의 트레이드-오프(trade-off)를 달성하는 것이다. 이를 위해 다양한 딥러닝 아키텍처를 실험하고, 성능을 극대화할 수 있는 학습 전략을 탐색한다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 기존의 CNN 기반 모델들이 가진 국소적 수용역(local receptive field)의 한계를 극복하기 위해 Transformer 기반의 모델인 $\text{SegFormer}$를 도입하고, 이를 정교한 학습 트릭들을 통해 최적화하는 것이다.

주요 기여 사항은 다음과 같다:

1. **효율적인 모델 선정**: $\text{SegFormer}$ 시리즈($\text{B0, B2, B5}$)를 통해 성능과 효율성 간의 상관관계를 분석하고, 타겟 태스크에 가장 적합한 $\text{SegFormer-B2}$를 도출하였다.
2. **단계적 성능 향상 파이프라인**: $\text{Transfer Learning}$부터 $\text{Class Balanced Loss}$, $\text{OHEM}$, $\text{Multi-scale Testing}$, $\text{Auxiliary Loss}$에 이르기까지 성능을 단계적으로 끌어올리는 체계적인 최적화 과정을 제시하였다.
3. **실용적 벤치마크 제공**: $\text{Baseline}$ 모델부터 $\text{U-Net}$, $\text{DeepLabV3+}$, $\text{PSPNet}$ 등 다양한 모델의 정량적 비교 결과를 제공하여 Transformer 기반 모델의 우수성을 입증하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 검토하며 기존 방식의 한계를 지적한다.

- **CNN 기반 모델 ($\text{ResNet, U-Net, DeepLabV3+}$)**: $\text{CNN}$은 이미지 필터를 통해 특징 맵을 생성하는 데 매우 효과적이다. $\text{U-Net}$은 $\text{Encoder-Decoder}$ 구조를 통해 세밀한 복원을 시도하며, $\text{DeepLabV3+}$는 $\text{Atrous Convolution}$과 $\text{Spatial Pyramid Pooling}$을 사용하여 수용역을 넓히고 다중 스케일 컨텍스트를 캡처한다. 그러나 이러한 $\text{CNN}$ 기반 모델들은 전역적 관계(global relations)를 설정하는 능력이 부족하다는 한계가 있다.
- **Transformer 기반 모델 ($\text{ViT, SETR, TransUNet}$)**: $\text{Attention}$ 메커니즘을 통해 전역적인 관계를 잘 파악하지만, 일반적으로 연산 복잡도가 매우 높아 실시간성이나 효율성이 중요한 세그멘테이션 작업에 적용하기 어렵다.
- **$\text{SegFormer}$**: 위 두 방식의 절충안으로, 계층적 Transformer 인코더와 경량 $\text{MLP}$ 디코더를 결합하여 효율성과 성능을 동시에 잡은 모델이다. 위치 인코딩(positional encoding)을 제거하여 다양한 해상도에서도 강건한 성능을 보인다.

## 🛠️ Methodology

### 1. $\text{SegFormer}$ 아키텍처

$\text{SegFormer}$는 계층적 구조의 $\text{Transformer Encoder}$와 가벼운 $\text{MLP Decoder}$로 구성된다.

- **Hierarchical Transformer Encoder**: $\text{ViT}$와 달리 다양한 스케일의 특징 맵($1/4, 1/8, 1/16, 1/32$ 해상도)을 생성한다.
- **Overlapped Patch Merging**: 패치 경계의 불연속성을 방지하기 위해 겹치는 패치 임베딩 방식을 사용하여 연속적인 특징 추출을 가능하게 한다.
- **Efficient Self-Attention**: 연산 복잡도를 $O(N^2)$에서 $O(N^2/R)$로 줄이기 위해 $\text{decay ratio } R$을 도입한다. 구체적으로 $\text{K}$의 차원을 $\text{Reshape}$ 및 $\text{Linear}$ 레이어를 통해 축소한다.
  $$ \hat{K} = \text{Reshape}(N/R, C \cdot R)(K) $$
  $$ K = \text{Linear}(C \cdot R, C)(\hat{K}) $$
- **Mix-FFN**: 위치 인코딩 대신 $3 \times 3 \text{ depthwise separable convolution}$을 $\text{MLP}$ 내부에 배치하여 인접 픽셀 간의 위치 정보를 학습한다.
  $$ x_{out} = \text{MLP}(\text{GELU}(\text{Conv}_{3\times3}(\text{MLP}(x_{in})))) \cdot x_{in} $$
- **Lightweight All-MLP Decoder**: 서로 다른 해상도의 특징 맵들을 $1/4$ 크기로 업샘플링하고 결합($\text{Concat}$)하여 최종 세그멘테이션 결과를 도출한다.

### 2. 학습 최적화 전략 (Training Tricks)

성능 향상을 위해 다음과 같은 절차를 적용하였다.

- **Transfer Learning**: $\text{Cityscapes}$ 데이터셋으로 사전 학습된 가중치를 사용하여 전이 학습을 수행하였다.
- **Learning Rate Scheduler**: $\text{AdamW}$ 옵티마이저와 함께 다음과 같은 다항식(Polynomial) 학습률 감소 정책을 사용하였다.
  $$ lr = lr_0 * (1 - \frac{i}{T_i})^{power} $$
- **Class Balanced Loss**: 데이터셋의 클래스 불균형 문제를 해결하기 위해, 유효 샘플 수에 반비례하는 가중치를 $\text{CrossEntropyLoss}$에 추가하여 클래스 균형 손실 함수를 구성하였다.
- **Online Hard Example Mining (OHEM)**: 학습 과정에서 예측 확신도가 낮거나 손실 값이 높은 '어려운 샘플'만을 선택적으로 학습에 사용하여 모델의 변별력을 높였다. ($\text{thresh}=0.5, \text{min\_kept}=10\text{k}$)
- **Multiple Scale Testing**: 입력 이미지의 스케일을 다양하게 변경하여 예측하고, 그 결과값들을 평균 내어 최종 예측값으로 사용하는 방식을 통해 작은 객체에 대한 검출 성능을 높였다.
- **Auxiliary Loss**: 기울기 소실(vanishing gradient) 문제를 방지하기 위해 두 번째 중간 레이어에 $\text{FCN}$ 형태의 보조 헤드를 추가하였다. 보조 손실과 주 손실의 가중치 비율은 $0.4 : 1.0$으로 설정하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: $\text{Cityscapes}$(사전 학습용), $\text{KITTI}$(타겟 데이터셋의 모집합) 및 제공된 타겟 데이터셋(학습 150장, 테스트 50장).
- **평가 지표**: $\text{mIoU}$ 및 $\text{FLOPs}$.
- **구현 도구**: $\text{PyTorch}$ 및 $\text{MMSegmentation}$ 툴박스.

### 정량적 결과

다양한 모델의 성능 비교 결과는 다음과 같다.

| 모델 아키텍처 | 사전 학습(Encoder/Decoder) | mIoU | FLOPs |
| :--- | :--- | :--- | :--- |
| Baseline (180 Epochs) | Baseline | 0.2760 | 67.003G |
| U-Net | - | 0.4555 | 356G |
| DeepLabV3+ & ResNet101 | ImageNet / - | 0.6063 | 455G |
| SegFormer-B0* | ImageNet / - | 0.6141 | 15.6G |
| SegFormer-B0 | ImageNet / Cityscapes | 0.7460 | 15.6G |
| **SegFormer-B2** | **ImageNet / Cityscapes** | **0.7845** | **50.6G** |
| SegFormer-B5 | ImageNet / Cityscapes | 0.8018 | 150G |

$^*$ 표시된 모델은 $\text{Cityscapes}$ 사전 학습을 거치지 않은 모델이다.

### 분석 결과

$\text{Baseline}$ 모델의 $\text{mIoU}$는 $0.2760$으로 매우 저조했으나, $\text{SegFormer}$ 도입과 최적화 트릭 적용을 통해 비약적인 상승을 보였다. 특히 $\text{SegFormer-B5}$가 $80.18\%$로 가장 높은 성능을 기록했지만, 연산량 대비 효율성을 고려하여 최종 모델로 $\text{SegFormer-B2}$($\text{mIoU } 78.45\%$, $\text{FLOPs } 50.6\text{G}$)를 선정하였다.

## 🧠 Insights & Discussion

본 연구를 통해 확인된 주요 강점과 한계는 다음과 같다.

- **강점**: 단순한 아키텍처 변경보다 $\text{Transfer Learning}$과 $\text{Learning Rate}$ 스케줄링, $\text{Class Balanced Loss}$와 같은 학습 전략의 정교한 튜닝이 실질적인 성능 향상에 더 큰 기여를 함을 확인하였다. 특히 $\text{Cityscapes}$ 사전 학습 가중치 적용 시 $\text{mIoU}$가 약 $7\%$p 가량 급증하는 결과를 보였다.
- **한계 및 미해결 과제**: 시간적 제약으로 인해 $\text{Knowledge Distillation}$($\text{Channel-wise Distillation}$)을 최종 모델에 완전히 적용하지 못하였다. $\text{PSPNet}$에서는 성공적으로 실험되었으나, $\text{SegFormer}$에 적용했을 때의 효과는 향후 연구 과제로 남겨두었다.
- **비판적 해석**: 본 논문에서 사용한 타겟 데이터셋은 $\text{KITTI}$의 부분집합으로 규모가 매우 작다(학습 150장). 이러한 소규모 데이터셋에서는 모델의 구조적 복잡성보다 강력한 사전 학습 모델의 선택과 과적합 방지를 위한 데이터 증강 및 정규화 전략이 더 결정적인 역할을 한다고 판단된다.

## 📌 TL;DR

본 연구는 차량 카메라 뷰의 도시 장면 분할 태스크에서 클래스 불균형 문제를 해결하기 위해 $\text{SegFormer}$ 모델과 체계적인 최적화 파이프라인을 적용하였다. $\text{Transfer Learning}, \text{Class Balanced Loss}, \text{OHEM}$ 등의 트릭을 통해 성능을 단계적으로 향상시켰으며, 최종적으로 $\text{mIoU } 78.45\%$와 $\text{50.6 GFLOPS}$를 가진 $\text{SegFormer-B2}$ 모델을 최적의 솔루션으로 제안한다. 이 연구는 제한된 데이터 환경에서 Transformer 기반 모델의 효율적인 활용 방안을 제시했다는 점에서 향후 자율주행 관련 실시간 세그멘테이션 연구에 기여할 가능성이 크다.
