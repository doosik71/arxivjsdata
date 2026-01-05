# Multiscale Vision Transformers

Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer

## 🧩 Problem to Solve

기존 Vision Transformer(ViT) 모델은 일반적으로 네트워크 전체에서 일정한 채널 용량과 해상도를 유지합니다. 이러한 단일 스케일 접근 방식은 특히 고밀도 시각 신호를 처리하는 비디오 및 이미지 인식 작업에서 효율성 및 성능 측면에서 제한적일 수 있습니다. 또한, 많은 최신 비전 트랜스포머는 대규모 외부 사전 학습 데이터셋(예: ImageNet-21K)에 크게 의존하며, 상당한 연산 및 파라미터 비용을 요구합니다. 본 연구는 이러한 문제를 해결하고, 시각 신호 모델링에 필수적인 멀티스케일 특징 계층 구조의 개념을 트랜스포머 모델에 통합하는 것을 목표로 합니다.

## ✨ Key Contributions

- **멀티스케일 Vision Transformer (MViT) 아키텍처 제안**: 멀티스케일 특징 계층 구조를 트랜스포머 모델에 성공적으로 통합하여, 채널 용량을 계층적으로 확장하고 공간 해상도를 감소시킵니다.
- **Multi Head Pooling Attention (MHPA) 도입**: 쿼리, 키, 값 텐서를 풀링하여 시퀀스 길이를 줄이는 유연한 어텐션 메커니즘을 제안하여, 다양한 스케일에서 효율적인 연산을 가능하게 합니다.
- **우수한 성능 및 효율성**: 외부 사전 학습 없이 다양한 비디오 인식 벤치마크 (Kinetics-400/600, SSv2, Charades, AVA)에서 동시 Vision Transformer 모델들을 뛰어넘는 성능을 달성했으며, 동시에 5-10배 낮은 연산 비용과 파라미터 수를 보였습니다.
- **강력한 시간 정보 모델링 능력**: MViT는 프레임 셔플링 테스트에서 성능 저하를 보여 기존 ViT가 시간 정보를 제대로 활용하지 못하는 한계와 대조적으로 강력한 시간적 바이어스를 가짐을 입증했습니다.
- **이미지 분류 성능 향상**: 시간 차원을 제거하고 ImageNet-1K 이미지 분류에 적용했을 때, 단일 스케일 Vision Transformer 모델보다 우수한 성능을 달성했습니다.

## 📎 Related Works

- **Convolutional Networks (ConvNets)**: 이미지 및 비디오 인식의 de-facto 표준으로, 다운샘플링, 이동 불변성, 가중치 공유 등 계층적 시각 처리의 주요 개념을 확립했습니다. (예: ResNet, EfficientNet)
- **Self-attention in ConvNets**: 컨볼루션 네트워크 내에서 자기-어텐션 메커니즘을 통합하여 성능을 향상시키려는 시도들입니다.
- **Vision Transformers (ViT)**: 트랜스포머 아키텍처를 이미지 분류에 직접 적용한 선구적인 연구 (Dosovitskiy et al., 2020). DeiT (Touvron et al., 2020)는 데이터 효율적인 ViT 훈련 방법을 제안했습니다.
- **Efficient Transformers**: 어텐션의 2차 복잡도를 줄여 NLP 애플리케이션의 효율성을 높이는 연구들입니다.
- **Concurrent Video Transformers**: 비디오를 위한 ViT 기반 아키텍처 (VTN, TimeSformer, ViViT)로, 대부분 대규모 외부 데이터(ImageNet-21K) 사전 학습에 의존합니다.

## 🛠️ Methodology

MViT의 핵심 아이디어는 네트워크 전체에 걸쳐 채널 용량을 점진적으로 확장하면서 공간 해상도를 풀링하여 줄이는 것입니다.

- **Multi Head Pooling Attention (MHPA)**:

  - 기존 MHA와 달리, 입력 $X \in \mathbb{R}^{L \times D}$에서 생성된 쿼리 $\hat{Q}$, 키 $\hat{K}$, 값 $\hat{V}$ 텐서를 풀링 연산자 $P(\cdot; \Theta)$를 사용하여 시퀀스 길이를 줄입니다.
  - 풀링 연산자 $P$는 커널 $k=(k_T, k_H, k_W)$, 스트라이드 $s=(s_T, s_H, s_W)$, 패딩 $p=(p_T, p_H, p_W)$를 사용하여 시퀀스 길이 $L$을 $\tilde{L}$로 줄입니다.
  - 어텐션은 짧아진 벡터 $Q=P(\hat{Q}; \Theta_Q)$, $K=P(\hat{K}; \Theta_K)$, $V=P(\hat{V}; \Theta_V)$에 대해 계산됩니다: $Attention(Q,K,V) = Softmax(QK^T/\sqrt{D})V$.
  - 이러한 풀링은 어텐션 연산의 연산 및 메모리 복잡도를 크게 줄입니다.

- **Multiscale Transformer Networks**:
  - **Scale Stages**: 여러 개의 트랜스포머 블록으로 구성되며, 각 스테이지는 동일한 채널 및 공간-시간 해상도를 유지합니다.
  - **Initial Projection (cube_1)**: 입력 비디오를 작은 채널 차원이지만 긴 시퀀스 길이의 3D 패치(큐브)로 분할하여 잠재 차원 $D$로 투영합니다.
  - **Channel Expansion**: 스테이지 전환 시 이전 스테이지 MLP 레이어의 출력을 2배 증가시켜 채널 차원을 확장합니다 (예: 해상도가 4배 감소하면 채널은 2배 증가).
  - **Query Pooling**: 각 스테이지의 첫 번째 어텐션 연산에서만 $s_Q > 1$인 쿼리 풀링을 사용하여 공간-시간 해상도를 줄입니다.
  - **Key-Value Pooling**: $K$와 $V$ 텐서에 풀링을 적용하여 연산 비용을 줄이며, 출력 시퀀스 길이는 변경하지 않습니다.
  - **Skip Connections**: 채널 차원 및 시퀀스 길이 불일치를 해결하기 위해 풀링 연산 및 추가 선형 레이어를 사용하여 스킵 연결을 조정합니다.
  - **Pooling Function**: 풀링 시 Max Pooling 대신 학습 가능한 채널별 컨볼루션(conv-pooling)을 사용하여 성능을 추가적으로 향상시킵니다.

## 📊 Results

- **비디오 인식 (Kinetics-400/600, SSv2, Charades, AVA)**:
  - Kinetics-400에서 MViT-B (32x3)는 80.2%의 Top-1 정확도를 달성했으며, 이는 ImageNet-21K 사전 학습에 의존하는 ViT-L-ViViT (81.3%)와 유사한 수준입니다. 하지만 MViT는 외부 사전 학습 없이 달성되었고, FLOPs는 6.8배, 파라미터는 8.5배 적습니다.
  - SSv2, Charades, AVA와 같은 다른 비디오 데이터셋에서도 ConvNet 기반 SOTA 모델 및 기존 비전 트랜스포머를 능가하는 성능을 보였습니다.
  - 프레임 셔플링 테스트에서 MViT는 -7.1%의 정확도 하락을 보인 반면, ViT-B는 -0.1%로 거의 변화가 없어 MViT가 시간 정보를 효과적으로 모델링함을 입증했습니다.
- **이미지 분류 (ImageNet-1K)**:
  - ImageNet-1K에서 MViT-B-16은 83.0%의 Top-1 정확도를 달성하여 DeiT-B (81.8%)보다 0.7% 높았으며, 연산 비용은 2.3배 적었습니다.
  - 더 큰 모델인 MViT-B-24-wide (320^2 해상도)는 84.8%의 정확도를 기록, DeiT-B↑384^2 (83.1%)보다 1.7% 높고 FLOPs는 1.7배 적습니다.
- **속도-정확도 트레이드오프**: MViT 모델은 기존 ViT-B baseline 및 ConvNet 모델(SlowFast, X3D)에 비해 훨씬 빠르고 정확합니다. (예: MViT-S는 ViT-B보다 3.4배 높은 처리량, +5.8% 정확도, 3.3배 적은 파라미터).

## 🧠 Insights & Discussion

- **멀티스케일 디자인의 중요성**: MViT는 시각 신호의 고밀도 특성을 모델링하는 데 멀티스케일 특징 계층 구조가 트랜스포머에도 근본적인 이점을 제공한다는 것을 보여줍니다. 초기 레이어에서는 높은 공간 해상도에서 간단한 저수준 정보를 모델링하고, 깊은 레이어에서는 공간적으로 거칠지만 복잡한 고수준 특징에 집중할 수 있도록 합니다.
- **강력한 귀납적 편향 (Inductive Bias)**: MViT의 멀티스케일 디자인은 모델 초기화 단계에서부터 다양한 어텐션 거리를 보이며 강력한 귀납적 편향을 부여합니다. 이는 기존 ViT가 훈련 후에도 균일한 어텐션 패턴을 보이는 것과 대조적입니다.
- **효율성 향상**: MHPA의 풀링 메커니즘은 어텐션의 2차 복잡도를 효과적으로 관리하여, 기존 ViT 모델보다 훨씬 적은 연산량과 파라미터로도 경쟁력 있는 또는 우수한 성능을 달성합니다. 이는 대규모 사전 학습 없이도 강력한 성능을 내는 데 기여합니다.
- **시간 정보 활용**: 기존 ViT의 단순한 비디오 적용이 시간 정보를 제대로 활용하지 못하는 문제점을 MViT는 멀티스케일 아키텍처를 통해 해결하며, 이는 특히 비디오 인식에서 중요한 강점입니다.
- **확장성 및 범용성**: MViT는 비디오 인식뿐만 아니라 이미지 분류에서도 우수한 성능을 보여, 다양한 시각 인식 작업에 대한 아키텍처의 범용성을 시사합니다.

## 📌 TL;DR

Multiscale Vision Transformer (MViT)는 멀티스케일 특징 계층 구조를 트랜스포머 모델에 통합하여 비디오 및 이미지 인식의 효율성과 성능을 향상시키는 새로운 아키텍처를 제안합니다. Multi Head Pooling Attention (MHPA)을 통해 채널 용량을 계층적으로 확장하고 공간 해상도를 감소시키며, 대규모 외부 사전 학습 없이도 기존 Vision Transformer 모델들보다 뛰어난 정확도와 현저히 낮은 연산 비용을 달성했습니다. 특히, 비디오에서 강력한 시간 정보 모델링 능력을 보여주며, 시각 신호 처리를 위한 트랜스포머의 새로운 설계 방향을 제시합니다.
