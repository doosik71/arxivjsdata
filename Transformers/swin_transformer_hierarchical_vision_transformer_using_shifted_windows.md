# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo

## 🧩 Problem to Solve

기존 Transformer 모델(특히 ViT)은 자연어 처리(NLP) 분야에서 뛰어난 성능을 보였으나, 컴퓨터 비전에 적용하기에는 여러 가지 본질적인 문제점이 존재합니다.

- **시각적 엔티티 스케일 다양성:** 이미지 내 객체의 크기가 매우 다양하며, 기존 Transformer는 고정된 크기의 토큰을 사용하기 때문에 객체 탐지(Object Detection)와 같은 다양한 스케일을 처리해야 하는 비전 태스크에 적합하지 않습니다.
- **고해상도 이미지 처리의 비효율성:** 이미지의 픽셀 해상도가 텍스트의 단어보다 훨씬 높아, 픽셀 단위의 밀집 예측(Dense Prediction, 예: 시맨틱 분할)을 요구하는 고해상도 이미지에 기존 Transformer의 자기-어텐션(Self-Attention) 연산($O(N^2)$, $N$은 토큰 수)을 직접 적용할 경우 계산 복잡도가 이미지 크기에 대해 제곱으로 증가하여 비효율적이고 비실용적입니다.
- **범용 백본으로서의 한계:** ViT는 이미지 분류에서 성공적이었지만, 밀집 예측 태스크를 위한 계층적 피처 맵을 생성하지 않아 컴퓨터 비전의 다양한 응용 분야에서 범용 백본 네트워크로 활용되기 어렵습니다.

## ✨ Key Contributions

- **계층적 비전 Transformer 아키텍처 제안:** CNN과 유사하게 계층적 피처 맵을 생성하는 새로운 Vision Transformer인 Swin Transformer를 제안하여, 객체 탐지 및 시맨틱 분할과 같은 밀집 예측 태스크에 대한 호환성을 높였습니다.
- **Shifted Windowing Scheme (이동 윈도우 메커니즘) 도입:** 비-중첩 로컬 윈도우(non-overlapping local windows) 내에서 자기-어텐션 계산을 제한하여 이미지 크기에 대한 선형 계산 복잡도($O(N)$)를 달성했습니다. 또한, 연속적인 Transformer 블록에서 윈도우 파티셔닝을 이동(shifted)시켜 윈도우 간 연결을 도입하여 모델의 표현 능력을 크게 향상시켰습니다.
- **다양한 비전 태스크에서의 SOTA 성능 달성:**
  - **ImageNet-1K 이미지 분류:** 87.3% top-1 정확도로 최첨단 성능 달성.
  - **COCO 객체 탐지 및 인스턴스 분할:** 58.7 box AP 및 51.1 mask AP를 달성하여 이전 SOTA 대비 각각 +2.7 box AP 및 +2.6 mask AP의 큰 개선을 이루었습니다.
  - **ADE20K 시맨틱 분할:** 53.5 mIoU를 달성하여 이전 SOTA 대비 +3.2 mIoU 개선을 보였습니다.
- **효율성 및 실용성 입증:** 제안된 이동 윈도우 방식은 효율적인 하드웨어 구현이 가능하며, 실제 추론 속도(Latency)가 이전 슬라이딩 윈도우(Sliding Window) 방식보다 훨씬 빠릅니다.
- **설계의 일반화 가능성:** 계층적 설계와 이동 윈도우 접근 방식이 All-MLP 아키텍처(예: MLP-Mixer)에도 유익함을 입증하여, 그 범용성을 시사했습니다.

## 📎 Related Works

- **CNN 및 변형(CNN and variants):** AlexNet, VGG, GoogleNet, ResNet, DenseNet, HRNet, EfficientNet 등 컴퓨터 비전에서 오랫동안 지배적인 백본 네트워크로 사용되어 온 컨볼루션 신경망입니다.
- **자기-어텐션 기반 백본 아키텍처(Self-attention based backbone architectures):** ResNet과 같은 CNN 백본의 일부 또는 모든 공간 컨볼루션 레이어를 자기-어텐션 레이어로 대체하려는 시도(예: [33, 50, 80]). 이러한 접근 방식은 로컬 윈도우 내에서 어텐션을 계산하여 최적화를 가속화하지만, 메모리 접근 비용으로 인해 실제 지연 시간이 길다는 단점이 있습니다.
- **CNN 보완을 위한 자기-어텐션/Transformer (Self-attention/Transformers to complement CNNs):** 표준 CNN 아키텍처에 자기-어텐션 레이어나 Transformer를 추가하여 장거리 의존성 또는 이질적인 상호작용을 인코딩하는 능력을 보완하는 연구(예: [67, 7, 3, 71, 23, 74, 55]).
- **Transformer 기반 비전 백본(Transformer based vision backbones):** Vision Transformer (ViT) [20] 및 그 후속 연구(DeiT [63], [72, 15, 28, 66]). ViT는 이미지 분류에서 인상적인 결과를 보였지만, 단일 해상도 피처 맵과 이미지 크기에 대한 이차 복잡도로 인해 고해상도 밀집 비전 태스크에는 부적합하다는 한계를 가집니다.

## 🛠️ Methodology

Swin Transformer는 다음과 같은 핵심 구성 요소를 통해 계층적 표현과 효율적인 자기-어텐션을 구현합니다.

- **전반적인 아키텍처 (Overall Architecture)**

  - **패치 분할 (Patch Splitting):** 입력 RGB 이미지를 겹치지 않는 패치(patch)로 분할합니다. 각 $4 \times 4$ 패치는 하나의 '토큰'으로 간주되며, 원본 픽셀 RGB 값의 연결로 피처가 설정됩니다.
  - **선형 임베딩 (Linear Embedding):** 초기 패치 피처는 선형 임베딩 레이어를 통해 임의의 차원 $C$로 투영됩니다.
  - **계층적 표현 생성 단계 (Hierarchical Representation Stages):**
    - **Stage 1:** 선형 임베딩 후 패치 토큰에 여러 Swin Transformer 블록이 적용되며, 토큰 수는 $H/4 \times W/4$로 유지됩니다.
    - **패치 병합 레이어 (Patch Merging Layers):** 네트워크가 깊어질수록 토큰 수를 줄여 계층적 표현을 생성합니다. 첫 번째 패치 병합 레이어는 $2 \times 2$ 이웃 패치 그룹의 피처를 연결한 후 선형 레이어를 적용하여 토큰 수를 4배($2 \times 2$) 줄이고 해상도를 절반으로 감소시킵니다.
    - 이 과정은 Stage 2, 3, 4에서 반복되며, 출력 해상도는 각각 $H/8 \times W/8$, $H/16 \times W/16$, $H/32 \times W/32$가 되어 CNN과 유사한 계층적 피처 맵을 생성합니다.

- **Swin Transformer 블록 (Swin Transformer Block)**

  - 표준 Transformer 블록의 Multi-head Self-Attention (MSA) 모듈을 Shifted Window 기반 MSA (SW-MSA) 모듈로 대체합니다.
  - 각 SW-MSA 모듈 뒤에는 GELU 비선형성을 갖는 2계층 MLP(Multi-Layer Perceptron)가 연결됩니다.
  - 각 MSA 모듈과 MLP 이전에 LayerNorm (LN) 레이어가 적용되며, 각 모듈 뒤에는 잔차 연결(Residual Connection)이 사용됩니다.

- **Shifted Window 기반 자기-어텐션 (Shifted Window based Self-Attention)**

  - **비-중첩 윈도우 내 자기-어텐션:** 글로벌 자기-어텐션의 $O(N^2)$ 계산 복잡도를 피하기 위해, 이미지를 비-중첩 로컬 윈도우로 분할하고 각 윈도우 내에서 자기-어텐션을 계산합니다. 각 윈도우는 $M \times M$ 크기의 패치(기본값 $M=7$)를 포함하며, 이로 인해 계산 복잡도가 이미지 크기에 대해 선형($O(N)$)으로 감소합니다.
  - **연속 블록에서의 이동 윈도우 분할:** 윈도우 내 어텐션은 윈도우 간 정보 흐름이 없다는 단점이 있습니다. 이를 해결하기 위해, 연속적인 Swin Transformer 블록에서 두 가지 윈도우 파티셔닝 설정을 번갈아 사용합니다.
    - `$l$` 레이어에서는 일반적인 윈도우 분할을 사용합니다.
    - 다음 `$l+1$` 레이어에서는 윈도우를 $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ 픽셀만큼 이동시킨 분할을 채택합니다.
    - 이동된 윈도우는 이전 레이어 윈도우의 경계를 넘나들어 윈도우 간 연결을 효과적으로 도입하고 표현 능력을 향상시킵니다.
  - **이동 구성에 대한 효율적인 배치 연산 (Efficient batch computation for shifted configuration):** 이동 윈도우 분할 시 윈도우 수가 증가하거나 일부 윈도우가 작아지는 문제가 발생할 수 있습니다. 이를 해결하기 위해 순환적 이동(cyclic-shifting) 방식을 사용하여 윈도우 수를 유지하고, 마스킹 메커니즘을 통해 자기-어텐션 계산을 각 서브-윈도우 내로 제한하여 효율적인 배치 연산을 가능하게 합니다.

- **상대적 위치 편향 (Relative Position Bias)**
  - 자기-어텐션 계산 시 각 헤드에 상대적 위치 편향 $B \in \mathbb{R}^{M^2 \times M^2}$를 추가합니다:
    $$\text{Attention}(Q,K,V) = \text{SoftMax}(QK^T/\sqrt{d}+B)V$$
  - 이 상대적 위치 편향은 모델의 성능을 크게 향상시키며, 기존 ViT에서 사용된 절대 위치 임베딩보다 우수하거나 최소한 동등한 성능을 보입니다. 사전 학습된 상대적 위치 편향은 bi-cubic interpolation을 통해 다른 윈도우 크기에서도 전이 학습에 활용될 수 있습니다.

## 📊 Results

Swin Transformer는 다양한 컴퓨터 비전 태스크에서 SOTA 성능을 달성했으며, 주요 결과는 다음과 같습니다.

- **ImageNet-1K 이미지 분류:**

  - 표준 ImageNet-1K 훈련에서 Swin-T는 DeiT-S (224$^2$ 입력) 대비 +1.5%p 높은 81.3% top-1 정확도를 달성했습니다. Swin-B는 DeiT-B 대비 최대 +1.5%p 높은 성능을 보였습니다.
  - ImageNet-22K 사전 학습 후 ImageNet-1K 미세 조정 시, Swin-B는 86.4%, Swin-L은 87.3% top-1 정확도를 달성하여, ViT 대비 더 나은 속도-정확도 트레이드오프를 보여주었습니다.

- **COCO 객체 탐지 및 인스턴스 분할:**

  - Cascade Mask R-CNN 프레임워크에서 Swin-T는 ResNet-50 대비 +3.4~4.2 box AP의 일관된 성능 향상을 보였습니다.
  - Swin-T는 DeiT-S 대비 +2.5 box AP, +2.3 mask AP 높은 성능을 달성하면서도 훨씬 빠른 추론 속도를 보여주었습니다.
  - 최고 모델인 Swin-L은 COCO test-dev에서 58.7 box AP 및 51.1 mask AP를 달성하여, 이전 SOTA 모델들(Copy-paste, DetectoRS) 대비 각각 +2.7 box AP 및 +2.6 mask AP의 큰 폭의 성능 개선을 이루었습니다.

- **ADE20K 시맨틱 분할:**

  - UperNet 프레임워크에서 Swin-S는 DeiT-S 대비 +5.3 mIoU (49.3 vs. 44.0) 향상을 보였습니다.
  - Swin-L (ImageNet-22K 사전 학습)은 53.5 mIoU를 달성하여, 이전 SOTA 모델인 SETR 대비 +3.2 mIoU의 개선을 이루었습니다.

- **핵심 요소 분석 (Ablation Study):**
  - **이동 윈도우 (Shifted Windows):** 이동 윈도우 방식은 ImageNet에서 +1.1%p top-1 정확도, COCO에서 +2.8 box AP / +2.2 mask AP, ADE20K에서 +2.8 mIoU를 향상시켰습니다. 이는 윈도우 간 연결의 효과를 명확히 보여줍니다.
  - **상대적 위치 편향 (Relative Position Bias):** 상대적 위치 편향은 위치 인코딩이 없는 경우나 절대 위치 임베딩을 사용하는 경우보다 ImageNet, COCO, ADE20K에서 일관되게 우수한 성능을 보여주며, 특히 밀집 예측 태스크에서 그 효과가 두드러졌습니다.

## 🧠 Insights & Discussion

- **Transformer의 범용 비전 백본으로서의 가능성:** Swin Transformer는 계층적 피처 맵과 선형 계산 복잡도를 제공함으로써, 기존 Transformer가 고해상도 및 밀집 예측 비전 태스크에서 가졌던 한계를 성공적으로 극복했습니다. 이는 Transformer 기반 모델이 이미지 분류뿐만 아니라 객체 탐지, 시맨틱 분할 등 다양한 비전 문제에서 범용 백본으로서 CNN을 대체하거나 보완할 수 있음을 입증합니다.
- **효율성과 표현 능력의 균형:** 이동 윈도우 방식은 로컬 자기-어텐션을 통해 계산 효율성을 극대화하는 동시에, 윈도우 이동을 통해 인접 윈도우 간의 정보 교환을 가능하게 하여 모델의 표현 능력을 유지하는 독창적이고 효과적인 해결책을 제시합니다. 이러한 접근 방식은 실제 하드웨어에서도 낮은 지연 시간으로 높은 성능을 제공합니다.
- **귀납적 편향의 중요성 재확인:** 상대적 위치 편향의 효과는 특히 밀집 예측 태스크에서 번역 불변성(Translation Invariance)과 같은 특정 귀납적 편향이 시각 모델링에 여전히 중요함을 시사합니다.
- **비전 및 언어 신호의 통합 모델링 촉진:** Swin Transformer가 다양한 비전 문제에서 보여준 강력한 성능은 컴퓨터 비전과 자연어 처리 분야 전반에 걸쳐 통합된 아키텍처에 대한 믿음을 강화하며, 시각 및 텍스트 신호의 공동 모델링 연구를 더욱 활성화할 것으로 기대됩니다.

## 📌 TL;DR

- **문제:** 기존 Vision Transformer(ViT)는 이미지 스케일 다양성 및 고해상도 이미지 처리 시 발생하는 제곱에 비례하는 계산 복잡도($O(N^2)$) 문제로 인해 컴퓨터 비전의 범용 백본으로 활용하기 어려웠습니다.
- **제안 방법:** Swin Transformer는 CNN과 유사한 계층적 피처 맵을 생성하고, 비-중첩 로컬 윈도우 내에서 자기-어텐션을 계산하여 선형 계산 복잡도($O(N)$)를 달성합니다. 핵심은 연속적인 레이어에서 윈도우를 이동(Shifted Windows)시켜 윈도우 간의 상호작용을 가능하게 하여 표현 능력을 유지한 점입니다.
- **주요 결과:** Swin Transformer는 ImageNet 분류뿐만 아니라 COCO 객체 탐지 및 ADE20K 시맨틱 분할에서 기존 SOTA를 크게 능가하는 성능을 달성하며, 뛰어난 효율성과 모델링 능력을 모두 입증하여 Transformer 기반 모델의 범용 비전 백본으로서의 강력한 잠재력을 제시했습니다.
