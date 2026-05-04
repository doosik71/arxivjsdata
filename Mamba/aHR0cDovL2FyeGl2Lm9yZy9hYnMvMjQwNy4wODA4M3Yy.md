# MambaVision: A Hybrid Mamba-Transformer Vision Backbone

Ali Hatamizadeh, Jan Kautz (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 고성능 모델을 구축할 때 발생하는 효율성과 표현력 사이의 트레이드오프(trade-off) 문제를 해결하고자 한다. 기존의 Vision Transformer(ViT)는 Self-attention 메커니즘을 통해 강력한 전역 수용장(global receptive field)을 가지지만, 시퀀스 길이에 따라 계산 복잡도가 제곱으로 증가하는 quadratic complexity 문제로 인해 학습 및 배포 비용이 매우 높다.

최근 등장한 State Space Model(SSM) 기반의 Mamba는 선형 시간 복잡도(linear time complexity)를 달성하며 효율적인 대안으로 주목받았으나, 비전 작업에 적용할 때 다음과 같은 한계가 존재한다. 첫째, Mamba의 자기회귀적(autoregressive) 구조는 순차적 데이터 처리에는 적합하지만, 픽셀 간의 공간적 관계가 병렬적으로 고려되어야 하는 이미지 데이터 처리에는 비효율적이다. 둘째, 단방향 처리 방식은 한 번의 forward pass만으로 전역 문맥(global context)을 충분히 캡처하는 데 한계가 있다. 이를 해결하기 위해 Bidirectional SSM(예: Vim)이 제안되었으나, 이는 계산 오버헤드를 증가시키고 추론 지연 시간을 늘리는 문제를 야기한다.

따라서 본 연구의 목표는 Mamba의 효율성과 Transformer의 강력한 전역 문맥 캡처 능력을 결합하여, 높은 처리량(throughput)과 정확도를 동시에 달성하는 하이브리드 비전 백본인 MambaVision을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba를 비전 작업에 최적화하여 재설계하고, 이를 Transformer와 전략적으로 결합한 하이브리드 아키텍처를 제안한 것이다.

가장 중심적인 아이디어는 Mamba 블록의 구조적 개선과 계층적(hierarchical) 배치 전략이다. 구체적으로, Mamba의 인과적 컨볼루션(causal convolution)을 일반 컨볼루션으로 대체하고, SSM 경로와 병렬로 작동하는 대칭적 경로(symmetric branch)를 추가하여 정보 손실을 방지했다. 또한, 모든 레이어에 Transformer를 배치하는 대신, 모델의 마지막 단계(final stages)에 Self-attention 블록을 집중 배치함으로써, Mamba가 효율적으로 추출한 특징들을 바탕으로 최종적인 전역 공간 의존성을 정교하게 모델링하도록 설계하였다.

## 📎 Related Works

기존의 비전 백본 연구는 크게 네 가지 방향으로 전개되었다. CNN 기반 모델(ConvNeXt, EfficientNetV2 등)은 효율적이지만 전역 수용장이 부족하며, Transformer 기반 모델(Swin, DeiT 등)은 성능은 뛰어나나 계산 비용이 높다. 이를 보완하기 위해 CNN과 Transformer를 섞은 하이브리드 모델(FasterViT, NextViT 등)이 등장하여 효율성을 개선했다.

최근에는 Mamba를 비전에 적용하려는 시도가 있었다. Vim은 양방향 SSM을 도입하여 공간 이해력을 높이려 했으나 지연 시간이 증가하는 단점이 있으며, VMamba는 Cross-Scan Module(CSM)을 통해 4방향 스캔을 수행하지만 수용장이 여전히 경로에 의해 제한된다. EfficientVMamba는 고해상도에서 SSM을, 저해상도에서 CNN을 사용했으나, 본 논문의 MambaVision은 반대로 고해상도에서 CNN을 사용하여 빠르게 특징을 추출하고 저해상도에서 SSM/Attention을 사용하는 전략을 취함으로써 더 나은 정확도와 처리량을 달성했다.

## 🛠️ Methodology

### 전체 시스템 구조 (Macro Architecture)

MambaVision은 4단계의 계층적 구조로 이루어져 있다.

1. **Stem 및 Stage 1, 2**: 입력 이미지를 처리하는 초기 단계에서는 CNN 기반의 residual block을 사용한다. 이는 고해상도 특징 맵에서 빠르게 기초적인 특징을 추출하기 위함이다.
2. **Stage 3, 4**: 이 단계에서는 제안된 MambaVision Mixer와 Transformer 블록이 혼합되어 배치된다.
3. **Downsampling**: 각 스테이지 사이에는 $3 \times 3$ 컨볼루션(stride 2)을 사용하여 해상도를 절반으로 줄인다.

### MambaVision Mixer (Micro Architecture)

본 논문은 기존 Mamba 블록을 비전 작업에 맞게 다음과 같이 재설계하였다.

1. **Causal Conv 대체**: 단방향으로만 영향을 주는 causal convolution을 일반 1D convolution으로 교체하여 공간적 제약을 없앴다.
2. **대칭적 경로(Symmetric Branch) 추가**: SSM의 순차적 제약으로 인해 손실될 수 있는 정보를 보완하기 위해, SSM이 없는 병렬 경로(Linear $\rightarrow$ Conv $\rightarrow$ SiLU)를 추가하였다.
3. **결합 방식**: 두 경로의 출력을 단순히 게이팅(gating)하는 대신 연결(concatenation)한 후 최종 linear layer로 투영한다.

이 과정은 다음 방정식으로 표현된다:
$$X_1 = \text{Scan}(\sigma(\text{Conv}(\text{Linear}(C, \frac{C}{2})(X_{in})))),$$
$$X_2 = \sigma(\text{Conv}(\text{Linear}(C, \frac{C}{2})(X_{in}))),$$
$$X_{out} = \text{Linear}(\frac{C}{2}, C)(\text{Concat}(X_1, X_2)),$$
여기서 $\text{Scan}$은 Mamba의 selective scan 연산이며, $\sigma$는 SiLU 활성화 함수이다. 각 브랜치의 차원을 $C/2$로 설정하여 전체 파라미터 수를 기존 Mamba 블록과 유사하게 유지했다.

### 하이브리드 통합 패턴 (Hybrid Pattern)

Stage 3와 4에서 MambaVision Mixer와 Self-attention 블록을 어떻게 배치할 것인지에 대한 실험을 진행하였다. 분석 결과, 전체 레이어 수 $N$ 중 앞부분의 $N/2$는 MambaVision Mixer를 배치하고, 나머지 뒷부분의 $N/2$에 Self-attention 블록을 배치하는 구조가 가장 높은 성능을 보였다. 이는 모델의 후반부에서 전역 문맥을 복구하고 장거리 공간 의존성을 캡처하는 것이 성능 향상에 핵심적임을 시사한다.

## 📊 Results

### 이미지 분류 (ImageNet-1K)

MambaVision은 정확도(Top-1 Accuracy)와 처리량(Throughput) 사이의 새로운 Pareto front를 형성하였다.

- **MambaVision-B**는 $84.2\%$의 정확도를 기록하며, ConvNeXt-B($83.8\%$) 및 Swin-B($83.5\%$)보다 높은 성능을 보이면서도 이미지 처리 속도는 훨씬 빠르다.
- 특히 기존 Mamba 기반 모델인 VMamba-B($83.9\%$)보다 정확도가 높으면서 처리량 면에서 압도적인 우위를 점했다.
- 계산 효율성 측면에서도 MambaVision-B는 MaxViT-B 대비 GFLOPs가 $56\%$ 더 낮다.

### 다운스트림 작업 (Object Detection & Segmentation)

MS COCO 및 ADE20K 데이터셋을 통해 범용성을 검증하였다.

- **Object Detection & Instance Segmentation (MS COCO)**: Cascade Mask R-CNN 헤드를 사용한 결과, MambaVision-T/S/B 모델이 ConvNeXt 및 Swin의 대응 모델보다 box AP 및 mask AP에서 일관되게 우수한 성능을 보였다.
- **Semantic Segmentation (ADE20K)**: UperNet 헤드를 사용했을 때, MambaVision-T/S/B가 Swin-T/S/B보다 mIoU 기준 각각 $+1.5, +0.6, +1.0$ 포인트 높은 성능을 기록하였다.

### 확장성 실험 (Scalability)

Mamba 기반 모델 최초로 ImageNet-21K 데이터셋으로 사전 학습을 진행하였다.

- MambaVision-B의 정확도가 $84.2\% \rightarrow 84.9\%$로 향상되었으며, 더 큰 모델인 MambaVision-L3는 해상도 $512$에서 $88.1\%$의 Top-1 정확도를 달성하여 대규모 데이터셋과 모델 크기에 대한 확장성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문의 가장 큰 성과는 Mamba의 선형 복잡도라는 효율성을 유지하면서도, Transformer의 전역 모델링 능력을 전략적으로 결합하여 비전 작업의 고질적인 문제(공간적 이해 부족)를 해결한 점이다. 특히, Self-attention을 모델의 후반부에 배치하는 설계가 실질적인 성능 향상을 이끌어냈음을 ablation study를 통해 정밀하게 증명하였다.

### 한계 및 논의사항

논문에서는 window-based attention을 사용하여 효율성을 높였으나, 윈도우 크기에 따른 성능 변화가 존재함을 언급하였다. 또한, 하이브리드 구조의 최적 비율($N/2$ 분할)이 실험적으로 도출되었으나, 이는 특정 모델 크기에 기반한 결과일 수 있으며 다른 규모의 모델에서도 동일하게 적용될지는 추가 연구가 필요하다. 다만, ImageNet-21K 실험을 통해 어느 정도의 일반적인 확장성을 보여주었다는 점에서 긍정적이다.

## 📌 TL;DR

MambaVision은 효율적인 State Space Model(Mamba)과 강력한 Transformer를 결합한 새로운 하이브리드 비전 백본이다. Mamba 블록을 비전 친화적으로 재설계하고, 모델의 후반부에 Self-attention 레이어를 배치함으로써 전역 문맥 캡처 능력을 극대화하였다. 그 결과, ImageNet-1K 분류 및 COCO, ADE20K 등 다양한 비전 작업에서 기존 CNN, ViT, Mamba 기반 모델들보다 더 나은 정확도-처리량 효율(Pareto front)을 달성하였으며, 대규모 데이터셋에 대한 확장성까지 입증하였다.
