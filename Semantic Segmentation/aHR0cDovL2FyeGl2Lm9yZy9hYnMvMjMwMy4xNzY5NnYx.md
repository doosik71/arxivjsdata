# Dual Cross-Attention for Medical Image Segmentation

Gorkem Can Ates, Prasoon Mohan, Emrah Celik

## 🧩 Problem to Solve

U-Net 기반의 컨볼루션 신경망(CNN)은 의료 영상 분할에서 뛰어난 성능을 보였지만, 두 가지 주요 한계에 직면해 있습니다.

1. **긴 범위 의존성(Long-range dependencies) 포착의 어려움:** 컨볼루션 연산의 본질적인 국소성(locality)으로 인해, 멀리 떨어진 특징들 간의 상호작용 및 전역적인 문맥(global context) 정보를 효과적으로 캡처하기 어렵습니다.
2. **의미론적 간극(Semantic Gap):** U-Net의 스킵 연결(skip-connections)은 인코더의 낮은 수준 특징(low-level features)과 디코더의 높은 수준 특징(high-level features)을 단순히 연결하여 공간 정보를 복구하는 데 도움을 주지만, 이질적인 특징 맵을 직접 결합할 때 의미론적 불일치(semantic mismatch)가 발생합니다. 이는 모델의 특징 표현력을 저해할 수 있습니다. 기존 U-Net 변형 모델들도 이러한 간극을 줄이기 위해 노력했지만, 여전히 한계가 존재합니다.

## ✨ Key Contributions

* **Dual Cross-Attention (DCA) 모듈 제안:** U-Net 기반 아키텍처의 스킵 연결을 강화하여 의료 영상 분할 성능을 향상시키는 간단하고 효과적인 어텐션 모듈인 DCA를 제안합니다.
* **순차적인 채널 및 공간 의존성 포착:** DCA는 다중 스케일 인코더 특징 전반에 걸쳐 채널 및 공간 방향의 긴 범위 상호 의존성을 순차적으로 캡처하여 의미론적 간극 문제를 해결합니다.
  * **Channel Cross-Attention (CCA):** 다중 스케일 인코더 특징의 채널 토큰 간 교차 어텐션을 활용하여 전역적인 채널별 의존성을 추출합니다.
  * **Spatial Cross-Attention (SCA):** 공간 토큰 간 교차 어텐션을 수행하여 공간적 의존성을 포착합니다.
* **경량화된 설계:** 계산 오버헤드를 최소화하기 위해 패치 임베딩에 2D 평균 풀링(Average Pooling)을 사용하고, 선형 프로젝션 대신 $1 \times 1$ 깊이별 컨볼루션(depth-wise convolutions)을 활용하며, MLP(Multi-Layer Perceptron) 레이어를 제거합니다.
* **우수한 성능 및 범용성:** U-Net, V-Net, R2Unet, ResUnet++, DoubleUnet, MultiResUnet 등 6가지 U-Net 기반 아키텍처에 DCA를 통합하여 5가지 벤치마크 의료 영상 데이터셋에서 Dice Score를 최대 2.74%까지 향상시키는 것을 입증했습니다. DCA는 다양한 인코더-디코더 아키텍처에 쉽게 통합될 수 있습니다.

## 📎 Related Works

* **U-Net 및 변형:** U-Net [44]은 스킵 연결을 통해 저수준 특징을 디코더에 전달하여 분할 성능을 향상시켰습니다. U-Net++ [75]는 중첩된 스킵 경로를, MultiResUnet [25]은 잔여 연결을 활용하여 스킵 연결의 품질을 개선하려 했습니다.
* **트랜스포머(Transformers) 및 어텐션(Attention) 메커니즘:** 자연어 처리(NLP)에서 시작된 트랜스포머 [54]는 Vision Transformer (ViT) [14]를 통해 컴퓨터 비전 분야에서도 지배적인 아키텍처가 되었습니다.
  * **Self-Attention:** 트랜스포머의 핵심으로, 긴 범위의 의존성을 직접 포착하는 능력 [43]이 있습니다.
  * **Dual Attention Schemes:** DANet [16]과 같이 채널 및 공간 어텐션 모듈을 병렬로 통합하거나, DaViT [11]와 같이 공간 및 채널 어텐션을 순차적으로 사용하는 방식이 제안되었습니다.
  * **Channel Cross-Attention:** Wang et al. [57]은 채널 교차 어텐션을 도입하여 U-Net의 의미론적 간극 문제를 해결하고, 다중 스케일 인코더 특징의 채널 축에서 전역적인 문맥을 효과적으로 포착할 수 있음을 보여주었습니다. DCA는 이러한 순차적 듀얼 어텐션 [11]과 채널 교차 어텐션 [57]의 성공에 영감을 받았습니다.

## 🛠️ Methodology

제안하는 DCA 블록은 스킵 연결이 있는 일반적인 인코더-디코더 아키텍처에 통합될 수 있으며, 크게 두 단계로 구성됩니다.

1. **다중 스케일 인코더 단계로부터의 패치 임베딩(Patch Embedding from Multi-Scale Encoder Stages):**
    * $n$개의 인코더 단계에서 생성된 다중 스케일 특징 $E_i \in \mathbb{R}^{C_i \times H_{2^{i-1}} \times W_{2^{i-1}}}$ (여기서 $i=1, \dots, n$)를 입력으로 받습니다.
    * 각 $E_i$에 대해 패치 크기 $P_{s_i}$를 갖는 2D 평균 풀링(AvgPool2D)을 적용하여 패치를 추출합니다.
    * 추출된 패치는 평탄화(Reshape)된 후 $1 \times 1$ 깊이별 컨볼루션(DConv1D)을 통해 프로젝션되어 토큰 $T_i \in \mathbb{R}^{P \times C_i}$를 생성합니다. 이때, 모든 $T_i$는 동일한 수의 패치 $P$를 갖습니다.
    * 수식: $T_i = \text{DConv1D}_{E_i}(\text{Reshape}(\text{AvgPool2D}_{E_i}(E_i)))$

2. **Dual Cross-Attention (DCA) 메커니즘:**
    DCA는 Channel Cross-Attention (CCA)과 Spatial Cross-Attention (SCA) 모듈을 순차적으로 적용합니다.
    * **Channel Cross-Attention (CCA):**
        * 각 토큰 $T_i$에 Layer Normalization (LN)을 적용합니다.
        * 모든 $T_i$를 채널 차원을 따라 연결하여 키(Key)와 값(Value) $T_c$를 생성하고, 쿼리(Query)로는 각 $T_i$를 사용합니다.
        * 쿼리, 키, 값은 $1 \times 1$ 깊이별 컨볼루션으로 프로젝션됩니다:
            $$Q_i = \text{DConv1D}_Q(T_i)$$
            $$K = \text{DConv1D}_K(T_c)$$
            $$V = \text{DConv1D}_V(T_c)$$
        * 채널 차원을 따라 교차 어텐션을 수행합니다. 이때, 쿼리, 키, 값의 전치(transpose)를 사용합니다:
            $$CCA(Q_i,K,V) = \text{Softmax}\left(\frac{Q_i^T K}{\sqrt{C_c}}\right) V^T$$
        * CCA의 출력에 깊이별 컨볼루션 프로젝션을 적용한 후 SCA 모듈로 전달합니다.
    * **Spatial Cross-Attention (SCA):**
        * CCA 모듈의 재구성된 출력 $\bar{T}_i \in \mathbb{R}^{P \times C_i}$에 Layer Normalization을 적용하고, 채널 차원을 따라 연결하여 쿼리(Q)와 키(K) $\bar{T}_c$를 생성합니다.
        * 각 토큰 $\bar{T}_i$는 값(V)으로 사용됩니다.
        * 쿼리, 키, 값은 $1 \times 1$ 깊이별 컨볼루션으로 프로젝션됩니다:
            $$Q = \text{DConv1D}_Q(\bar{T}_c)$$
            $$K = \text{DConv1D}_K(\bar{T}_c)$$
            $$V_i = \text{DConv1D}_{V_i}(\bar{T}_i)$$
        * SCA는 다음과 같이 표현됩니다:
            $$SCA(Q,K,V_i) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V_i$$
        * SCA 모듈의 출력은 깊이별 컨볼루션을 통해 최종 DCA 출력을 형성합니다.
    * 최종 DCA 출력에는 Layer Normalization과 GeLU 활성화 함수가 적용되며, 그 후 업샘플링 레이어와 $1 \times 1$ 컨볼루션, Batch Normalization, ReLU를 거쳐 해당 디코더 부분에 연결됩니다.

## 📊 Results

* **Dice Score 및 IoU 개선:** DCA 모듈은 U-Net, V-Net, R2Unet, ResUnet++, DoubleUnet, MultiResUnet 등 6가지 U-Net 기반 아키텍처에 통합되었을 때, 5가지 벤치마크 의료 영상 분할 데이터셋(GlaS, MoNuSeg, CVC-ClinicDB, Kvasir-Seg, Synapse) 전반에 걸쳐 상당한 성능 향상을 보였습니다.
  * GlaS에서 최대 2.05%, MoNuSeg에서 최대 2.74%, CVC-ClinicDB에서 최대 1.37%, Kvasir-Seg에서 최대 1.12%, Synapse에서 최대 1.44%의 Dice Score 개선을 달성했습니다.
* **경미한 파라미터 증가:** DCA 통합으로 인한 모델 파라미터 증가는 매우 적습니다. 대부분의 모델에서 0.3%에서 1.5% 사이의 증가를 보였으며, 가장 많은 스킵 연결 체계를 가진 DoubleUnet의 경우에도 3.4% 증가에 그쳤습니다. 이는 DCA가 높은 계산 효율성을 유지하면서도 성능을 크게 향상시킨다는 것을 의미합니다.
* **시각적 개선:** DCA가 적용된 모델은 평범한 모델에 비해 더 일관된 경계를 생성하고, 정확한 형태 정보를 유지하며, 오탐(false positive) 예측을 제거하여 이산적인 부분을 더 명확하게 구별하는 등 시각적으로도 우수한 분할 결과를 보여주었습니다.

## 🧠 Insights & Discussion

* **의미론적 간극 해소의 효과성:** DCA는 다중 스케일 인코더 특징 간의 채널 및 공간 의존성을 효과적으로 포착함으로써 인코더-디코더 간의 의미론적 간극을 성공적으로 줄였습니다. 이는 모델이 보다 미세하고 문맥적으로 일관된 특징 표현을 학습하도록 돕습니다.
* **순차적 어텐션의 우위:** CCA 다음에 SCA를 적용하는 순차적 듀얼 어텐션 방식(CCA-SCA)이 개별 어텐션 모듈이나 병렬 융합(합산, 연결) 방식보다 일관되게 더 나은 성능을 보였습니다. 이는 채널별 및 공간별 교차 어텐션 메커니즘이 서로를 보완하며, 더 풍부하고 심층적인 전역 문맥 정보를 효과적으로 추출할 수 있음을 시사합니다.
* **경량화된 설계의 효율성:** 매개변수 없는 2D 평균 풀링을 사용한 패치 임베딩과 깊이별 컨볼루션을 사용한 프로젝션 레이어는 계산 오버헤드를 최소화하면서도 성능 저하 없이 효과적인 특징 추출을 가능하게 합니다. 특히, 평균 풀링이 컨볼루션 기반 패치 임베딩보다 더 나은 결과를 제공하며, 추가 매개변수 없이 더 효율적임을 입증했습니다.
* **높은 범용성:** DCA 모듈은 특정 U-Net 변형에 구애받지 않고 다양한 U-Net 기반 아키텍처에 쉽게 통합되어 성능을 향상시키는 것으로 나타났습니다. 이는 DCA가 광범위한 의료 영상 분할 작업에서 유연하게 적용될 수 있는 잠재력을 가짐을 의미합니다.

## 📌 TL;DR

이 논문은 U-Net 기반 의료 영상 분할 모델의 **의미론적 간극** 및 **긴 범위 의존성 포착 문제**를 해결하기 위해 **Dual Cross-Attention (DCA) 모듈**을 제안합니다. DCA는 **Channel Cross-Attention (CCA)**과 **Spatial Cross-Attention (SCA)**을 순차적으로 적용하여 다중 스케일 인코더 특징의 채널 및 공간 차원에서 긴 범위 의존성을 효과적으로 캡처합니다. 경량화를 위해 2D 평균 풀링과 깊이별 컨볼루션을 활용하며, 6가지 U-Net 기반 모델에 통합되어 5가지 의료 영상 데이터셋에서 **최대 2.74%의 Dice Score 향상**을 달성, 뛰어난 성능과 파라미터 효율성을 입증했습니다.
