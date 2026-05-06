# Med-URWKV: Pure RWKV With ImageNet Pre-training For Medical Image Segmentation

Zhenhuan Zhou (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 컴퓨터 보조 진단 및 치료의 핵심 기술이다. 기존의 접근 방식은 크게 세 가지로 분류된다. 첫째, Convolutional Neural Network (CNN) 기반 모델은 국소적 특징 추출 능력이 뛰어나지만, Receptive Field가 제한적이라 전역적 문맥(Global Context) 모델링에 취약하다. 둘째, Transformer 기반 모델은 전역적 의존성을 잘 학습하지만, 연산 복잡도가 입력 크기의 제곱에 비례하는 Quadratic Complexity를 가져 고해상도 의료 영상 처리 시 계산 비용과 메모리 오버헤드가 매우 크다. 셋째, 이 둘을 결합한 하이브리드 구조가 존재한다.

최근 Linear Computational Complexity를 가지면서도 강력한 장거리 모델링 능력을 갖춘 Receptance Weighted Key Value (RWKV) 모델이 대안으로 부상하였다. 하지만 기존의 의료 영상 분할을 위한 RWKV 연구들은 주로 Vision-RWKV (VRWKV) 메커니즘의 일부를 수정하여 처음부터 학습(Train from scratch)시키는 방식에 집중했다. 또한, 인코더에만 RWKV를 적용하고 디코더는 여전히 CNN을 사용하는 하이브리드 구조를 채택함으로써, 순수 RWKV 아키텍처의 잠재력을 충분히 활용하지 못하고 대규모 사전 학습된(Pre-trained) VRWKV 모델의 이점을 활용하지 않았다는 한계가 있다.

본 논문의 목표는 ImageNet으로 사전 학습된 VRWKV 인코더를 직접 활용할 수 있는 순수 RWKV 기반의 분할 모델인 Med-URWKV를 제안하여, 의료 영상 분할 작업에서 RWKV의 성능을 극대화하고 배포 효율성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대규모 데이터셋으로 사전 학습된 VRWKV 모델을 의료 영상 분할 프레임워크에 통합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **사전 학습된 VRWKV 인코더의 재사용**: ImageNet과 ADE20K로 학습된 VRWKV-Tiny 모델의 인코더를 직접 활용하여, 학습 수렴 속도를 높이고 분할 성능을 향상시켰다.
2. **순수 RWKV 아키텍처 설계**: 인코더, 보틀넥(Bottleneck), 디코더 모두에 RWKV 메커니즘을 적용한 순수 RWKV 기반의 U-Net 구조를 제안함으로써, 하이브리드 모델의 한계를 극복하고 RWKV의 장점을 완전히 활용하였다.
3. **효율성 및 성능 입증**: 적은 파라미터 수로도 기존의 CNN, ViT 및 하이브리드 RWKV 모델보다 우수하거나 대등한 성능을 보임을 7개의 공개 데이터셋을 통해 검증하였다.

## 📎 Related Works

의료 영상 분할 분야에서는 U-Net과 그 변형 모델들이 국소 특징 추출 능력을 바탕으로 널리 사용되어 왔다. 이후 전역 문맥 학습을 위해 Swin-Unet, TransUNet과 같은 Transformer 기반 모델들이 등장하였으나, 앞서 언급한 계산 복잡도 문제가 제기되었다. 이에 대한 해결책으로 Mamba 기반 모델들이 제안되었으나, 계산 비용은 낮췄지만 정확도 면에서 손실이 발생하는 경우가 많았다.

RWKV 모델은 Transformer의 성능과 RNN의 효율성을 동시에 갖춘 구조로 주목받고 있으며, 이를 비전 분야로 확장한 Vision-RWKV (VRWKV)는 ImageNet 분류, COCO 객체 탐지 등에서 ViT보다 뛰어난 성능을 보였다. 의료 영상 분할 분야에서도 RWKV-Unet, Zig-RiR, BSBP-RWKV, HFE-RWKV 등의 연구가 진행되었으나, 이들은 대부분 하이브리드 구조(CNN-RWKV)를 사용하거나 사전 학습 없이 처음부터 학습시킨다는 점에서 본 연구와 차별화된다.

## 🛠️ Methodology

### 1. VRWKV Block 구조

VRWKV 블록은 입력 이미지 $X \in \mathbb{R}^{H \times W \times C}$를 패치 임베딩(Patch Embedding)을 통해 토큰 시퀀스 $X' \in \mathbb{R}^{T \times C}$로 변환한 뒤, 이를 처리하는 두 가지 핵심 컴포넌트로 구성된다.

**Spatial Mix Block (Global Attention 역할):**
입력 $X'$는 먼저 Q-shift 메커니즘을 통해 이전 타임스텝의 지식을 통합하며, 이후 세 개의 선형 변환을 통해 $R_s, K_s, V_s$ 행렬을 생성한다.
$$R_s = \text{Q-shift}(X')W_R$$
$$K_s = \text{Q-shift}(X')W_K$$
$$V_s = \text{Q-shift}(X')W_V$$

이후 bi-WKV 메커니즘을 통해 attention 연산자 $wkv$를 계산하며, $R_s$는 게이팅 메커니즘(Gating Mechanism)으로 작용하여 최종 출력 $O_s$를 결정한다.
$$\text{Bi-WKV}(K, V)_t = \frac{\sum_{i=0, i \neq t}^{T-1} e^{-(|t-i|-1)/T \cdot w + k_i} v_i + e^{u + k_t} v_t}{\sum_{i=0, i \neq t}^{T-1} e^{-(|t-i|-1)/T \cdot w + k_i} + e^{u + k_t}}$$
$$O_s = (\sigma(R_s) \odot wkv)W_O$$
여기서 $w$와 $u$는 학습 가능한 벡터이며, $\sigma$는 시그모이드 함수이다.

**Channel Mix Block (FFN 역할):**
Spatial Mix의 출력을 받아 Spatial Mix와 유사한 선형 변환 및 게이팅 과정을 거쳐 최종 특징을 생성한다.

### 2. Med-URWKV 전체 아키텍처

Med-URWKV는 U-Net의 구조를 따르며, 다음과 같은 세 가지 주요 구성 요소로 이루어져 있다.

* **Pre-trained VRWKV Encoder**: ImageNet으로 사전 학습된 VRWKV-Tiny 인코더를 사용한다. 입력 영상으로부터 계층적 특징 $X_i \in \mathbb{R}^{\frac{H}{2^{i+1}} \times \frac{W}{2^{i+1}} \times \text{Dims}}$를 추출한다.
* **RWKV Bottleneck Block**: 인코더의 최종 출력 $X_4$를 입력으로 받아 추가적인 특징 추출 및 차원 축소를 수행하여 디코더로 전달한다.
* **VRWKV Decoder**: 보틀넥의 출력을 입력으로 하여 점진적으로 패치를 확장(Patch Expanding)하고 특징을 디코딩한다. 이때 인코더의 계층적 특징들이 Skip Connection을 통해 통합된다.

최종적으로 $1 \times 1$ Convolutional Layer로 구성된 Segmentation Head를 통해 클래스 수 $n$에 맞는 예측 맵 $Y \in \mathbb{N}^{H \times W \times n}$를 출력한다.

### 3. 학습 절차 및 손실 함수

* **사전 학습 활용**: VRWKV-Tiny 구조 중 ImageNet 학습 후 ADE20K로 미세 조정된 모델의 인코더 부분만 추출하여 사용하였다.
* **학습 전략**: 초기 5 에포크(epoch) 동안 인코더의 파라미터를 동결(frozen)하여 나머지 네트워크(보틀넥, 디코더, 헤드)를 먼저 정렬시킨 후, 이후 모든 파라미터를 해제하여 전체 모델을 학습시켰다.
* **최적화**: AdamW 옵티마이저를 사용하였으며, 'poly' 학습률 조정 전략을 적용하였다.
* **손실 함수**: Cross-Entropy Loss와 Dice Loss를 결합하여 사용하였다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**: ISIC2017, ISIC2018 (피부암), GLAS (병리), TDD (치과 X-ray), BUSI (유방초음파), KvasirSEG (폴립), NKUT (사랑니) 등 총 7개의 데이터셋에서 검증하였다.
* **평가 지표**: Dice Similarity Coefficient (DSC)와 Intersection over Union (IoU)를 사용하였다.

### 2. 정량적 결과

실험 결과, Med-URWKV는 기존의 CNN, ViT 기반 모델뿐만 아니라 최신 RWKV 하이브리드 모델(Zig-RiR 등)보다 우수한 성능을 보였다. 특히 파라미터 수 측면에서 Med-URWKV는 $14.33\text{M}$으로, 비교 대상 모델들(예: TransUNet $92.23\text{M}$, Swin-Unet $27.15\text{M}$)보다 현저히 적으면서도 높은 정확도를 달성하였다.

### 3. 사전 학습의 효과 (Ablation Study)

BUSI 데이터셋을 이용해 사전 학습 유무에 따른 성능을 비교한 결과, 사전 학습을 적용한 모델이 적용하지 않은 모델보다 훨씬 빠르게 수렴하였으며, 최종 DSC 성능 또한 크게 향상됨을 확인하였다 (사전 학습 적용 시 Best DSC $77.98\%$, 미적용 시 $51.17\%$).

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할 작업에서 순수 RWKV 아키텍처와 대규모 사전 학습된 가중치의 결합이 매우 효과적임을 입증하였다. 특히, Transformer의 성능을 유지하면서도 Linear Complexity를 통해 연산 효율성을 확보했다는 점이 강점이다. 또한, 하이브리드 구조(CNN-RWKV) 대신 순수 RWKV 구조를 채택함으로써 모델의 일관성을 높이고 파라미터 효율성을 극대화하였다.

다만, 본 연구에는 몇 가지 한계점이 존재한다. 첫째, 다양한 규모(scale)의 사전 학습된 인코더가 성능에 미치는 영향에 대한 분석이 부족하다. 둘째, 의료 영상의 특성에 특화된 전용 Attention 메커니즘에 대한 설계가 아직 이루어지지 않았다. 향후 연구에서는 다양한 크기의 VRWKV 모델을 실험하고, 의료 도메인에 최적화된 RWKV 변형 구조를 탐색함으로써 성능을 더욱 끌어올릴 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 ImageNet으로 사전 학습된 VRWKV 인코더를 활용한 순수 RWKV 기반의 의료 영상 분할 모델인 **Med-URWKV**를 제안한다. 이 모델은 기존의 CNN 및 Transformer 기반 모델보다 적은 파라미터($14.33\text{M}$)로도 우수한 분할 성능을 보이며, 특히 사전 학습된 가중치를 통해 학습 안정성과 정확도를 크게 향상시켰다. 이는 향후 고해상도 의료 영상 처리에서 계산 효율성과 정확도를 동시에 잡을 수 있는 실용적인 방향성을 제시한다.
