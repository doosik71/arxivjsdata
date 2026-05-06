# VISION-RWKV: Efficient and Scalable Visual Perception with RWKV-like Architectures

Yuchen Duan, Weiyun Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Hongsheng Li, Jifeng Dai, Wenhai Wang (2025)

## 🧩 Problem to Solve

본 논문은 Vision Transformer(ViT)가 가진 핵심적인 한계점인 **이차 복잡도(Quadratic Computational Complexity)** 문제를 해결하고자 한다. ViT는 전역적인 정보 처리 능력과 유연성 덕분에 다양한 컴퓨터 비전 작업에서 뛰어난 성능을 보이지만, 입력 이미지의 해상도가 높아질수록 토큰 수가 증가하며 이에 따라 연산량과 메모리 사용량이 기하급수적으로 증가하는 문제가 발생한다.

이러한 특성은 고해상도 이미지 처리나 긴 컨텍스트 분석을 어렵게 만들며, 이를 해결하기 위해 기존에는 Window-based attention과 같은 국소적 연산 방식이 도입되었다. 하지만 이러한 방식은 전역적인 수용역(Global Receptive Field)을 희생한다는 단점이 있다. 따라서 본 연구의 목표는 ViT의 전역 처리 능력과 확장성(Scalability)을 유지하면서도, 연산 복잡도를 선형 수준으로 낮춘 효율적인 비전 인코더를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 NLP 분야의 선형 복잡도 모델인 RWKV 아키텍처를 비전 작업에 맞게 최적화하여 **Vision-RWKV (VRWKV)**를 제안한 것이다. 중심적인 설계 아이디어는 다음과 같다.

1. **Bi-WKV (Bidirectional Global Attention):** 기존 RWKV의 인과적(Causal) 어텐션을 양방향 전역 어텐션으로 수정하여, 모든 토큰이 서로를 참조할 수 있게 하면서도 선형 복잡도를 유지한다.
2. **Q-Shift (Quad-directional Token Shift):** 이미지의 공간적 특성을 반영하기 위해 네 방향으로 토큰을 시프트하여 인접 토큰 간의 정보를 교환하는 메커니즘을 도입함으로써 국소적 문맥을 효과적으로 캡처한다.
3. **Scalability & Stability:** 모델의 크기를 키울 때 발생하는 수치적 불안정성(Gradient Vanishing/Exploding)을 해결하기 위해 Bounded Exponential, Extra Layer Normalization, Layer Scale 등의 안정화 전략을 적용하여 대규모 파라미터 및 데이터셋에서도 안정적인 학습이 가능하도록 하였다.

## 📎 Related Works

**1. Vision Encoder의 발전**
초기에는 CNN이 지배적이었으며, 이후 전역 수용역을 가진 ViT가 등장하였다. ViT의 연산 비용을 줄이기 위해 PVT는 다운샘플링된 특징 맵을 사용했고, Swin Transformer는 Window-based attention을 도입하였다. 최근에는 Mamba나 RWKV와 같은 선형 복잡도 모델을 비전에 적용하려는 시도(Vim, VMamba 등)가 있었으나, 대부분 100M 파라미터 이하의 소규모 모델에서만 검증되었다는 한계가 있다.

**2. Feature Aggregation Mechanism**
특징 집계 방식은 국소적 인식을 하는 Convolution에서 전역적 인식을 하는 Self-attention으로 이동하였다. 최근에는 RNN의 순차적 처리 능력과 Transformer의 병렬 처리 능력을 결합한 선형 복잡도 연산자들이 연구되고 있다. VRWKV는 이러한 선형 복잡도 모델들을 비전 도메인으로 확장하되, 특히 대규모 모델로의 확장 가능성과 학습 안정성에 집중하여 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

VRWKV는 ViT와 유사하게 Patch Embedding 층과 $L$개의 동일한 **VRWKV Encoder Layer**가 쌓인 구조를 가진다. 각 레이어는 **Spatial-mix** 모듈과 **Channel-mix** 모듈로 구성된다.

### 주요 구성 요소 및 역할

**1. Spatial-mix Module (전역 어텐션 역할)**

- **Q-Shift:** 입력 토큰을 상, 하, 좌, 우 네 방향으로 시프트하여 인접 토큰과 선형 보간한다. 이를 통해 어텐션 메커니즘이 별도의 추가 비용 없이 주변 토큰의 정보를 미리 반영할 수 있게 하여 유효 수용역(ERF)을 넓힌다.
- **Bi-WKV:** 선형 복잡도를 가진 양방향 전역 어텐션을 수행한다.
- **Gating:** $\sigma(R^s)$ (Sigmoid 함수)를 사용하여 출력 값의 확률을 제어한다.

**2. Channel-mix Module (FFN 역할)**

- Spatial-mix와 유사하게 Q-Shift를 거친 후, SquaredReLU 활성화 함수와 게이트 메커니즘을 통해 채널 차원의 특징 융합을 수행한다.

### 핵심 방정식 및 알고리즘

**Bi-WKV의 수식 표현**
Bi-WKV는 이론적인 이해를 돕는 Summation Form과 실제 구현을 위한 RNN Form 두 가지로 표현된다.

- **Summation Form:** $t$번째 토큰의 결과 $wkv_t$는 다음과 같이 계산된다.
$$wkv_t = \frac{\sum_{i=0, i \neq t}^{T-1} e^{-(|t-i|-1)/T \cdot w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i=0, i \neq t}^{T-1} e^{-(|t-i|-1)/T \cdot w + k_i} + e^{u+k_t}}$$
여기서 $w$는 채널별 공간적 감쇠(decay) 벡터, $u$는 현재 토큰에 대한 보너스 벡터, $T$는 전체 토큰 수이다. $|t-i|/T$를 통해 상대적 편향(Relative Bias)을 부여한다.

- **RNN Form:** 실제 구현에서는 위 식을 재귀적 형태로 변환하여 $O(TC)$의 복잡도로 계산한다. 네 가지 은닉 상태(hidden states) $a, b, c, d$를 유지하며, 앞방향(forward)과 뒷방향(backward)으로 스캔하여 전역 정보를 취합한다.
$$wkv_t = \frac{a_{t-1} + b_{t-1} + e^{k_t+u} v_t}{c_{t-1} + d_{t-1} + e^{k_t+u}}$$

**Q-Shift 연산**
$$Q\text{-Shift}(X) = X + (1 - \mu) X^\dagger$$
여기서 $X^\dagger$는 상, 하, 좌, 우 인접 토큰들을 Concatenation 하여 만든 텐서이며, $\mu$는 학습 가능한 보간 벡터이다.

### 학습 안정화 전략 (Scale-up Stability)

- **Bounded Exponential:** 지수 항을 토큰 수 $T$로 나누어 $\exp(-(|t-i|-1)/T \cdot w)$ 형태로 만들어, 해상도가 높아져도 값이 폭주하거나 소멸하지 않도록 제한한다.
- **Extra Layer Normalization:** 모델이 깊어짐에 따라 출력값이 오버플로우 되는 것을 방지하기 위해 어텐션 메커니즘과 SquaredReLU 이후에 추가적인 LayerNorm을 배치한다.

## 📊 Results

### 실험 설정

- **데이터셋:** ImageNet-1K, ImageNet-22K, COCO, ADE20K.
- **모델 규모:** VRWKV-Tiny(6M)부터 VRWKV-Large(335M)까지 제공.
- **지표:** Top-1 Accuracy, box mAP, mIoU, FLOPs, GPU Memory, Inference Speed (FPS).

### 주요 결과

**1. 이미지 분류 (ImageNet-1K)**

- **VRWKV-T**는 DeiT-T보다 2.9%p 높은 **75.1%**의 정확도를 달성하며 더 적은 FLOPs를 기록했다.
- **VRWKV-L**은 ViT-L(85.15%)보다 높은 **86.0%**의 정확도를 보였으며, Bamboo-47K 데이터셋으로 사전 학습 시 **86.5%**까지 상승하여 대규모 데이터셋에 대한 확장성을 입증했다.

**2. 객체 검출 및 세그멘테이션 (COCO, ADE20K)**

- **COCO (Detection):** VRWKV-L은 **50.6% box mAP**를 기록하여 ViT-L(48.7%)보다 1.9%p 향상된 성능을 보였다. 특히 Window-based ViT보다 적은 FLOPs로 더 높은 성능을 냈다.
- **ADE20K (Segmentation):** VRWKV-S는 ViT-S보다 1%p 높은 mIoU를 기록했으며, 연산량은 14% 감소했다.

**3. 효율성 및 강건성 분석**

- **추론 속도 및 메모리:** 해상도가 $2048 \times 2048$일 때, VRWKV-T는 ViT-T보다 **10배 빠른 속도**를 보였으며, **GPU 메모리 사용량을 80% 절감**했다.
- **해상도 강건성:** 224 해상도에서 학습하고 1024 해상도에서 평가했을 때, ViT-B의 정확도는 57.5%로 급락한 반면, VRWKV-B는 **67.2%**를 유지하여 고해상도 시나리오에서 훨씬 강력한 강건성을 보였다.

## 🧠 Insights & Discussion

**강점**
VRWKV는 전역 수용역을 유지하면서도 연산 복잡도를 $O(T^2)$에서 $O(T)$로 낮추는 데 성공하였다. 특히 단순한 효율성 개선을 넘어, 수치적 안정성 장치를 통해 300M 파라미터 이상의 대규모 모델까지 성공적으로 확장했다는 점이 고무적이다. 또한, 테스트 시 입력 해상도가 변하더라도 성능 하락이 적은 특성은 실제 고해상도 이미지 분석 작업에서 매우 큰 이점이 된다.

**한계 및 비판적 해석**
논문에서 언급되었듯이, 현재 PyTorch로 구현된 Q-Shift 연산은 비효율적이어서 실제 모델 전체 속도를 저하시키는 요인이 된다. CUDA 커널 최적화가 이루어진다면 더 큰 성능 향상이 가능할 것이나, 현재 보고된 속도 결과는 Bi-WKV 자체의 효율성에 치중되어 있다. 또한, Vision Mamba(Vim)와 비교했을 때 소형 모델에서는 Vim이 약간 더 우세한 경향이 있으나, VRWKV는 대형 모델로의 확장성에서 우위를 점하고 있다.

## 📌 TL;DR

본 논문은 ViT의 고질적인 문제인 이차 복잡도를 해결하기 위해, RWKV 아키텍처를 비전에 최적화한 **Vision-RWKV (VRWKV)**를 제안한다. **Bi-WKV**와 **Q-Shift**를 통해 선형 복잡도로 전역 및 국소 정보를 모두 캡처하며, 특수 설계된 안정화 기법으로 대규모 모델 확장성을 확보하였다. 결과적으로 ViT 수준의 성능을 유지하면서 고해상도 이미지 처리 시 메모리 사용량은 80% 줄이고 속도는 10배 높였으며, 해상도 변화에 대한 강건성 또한 크게 향상시켰다. 이 연구는 고해상도 비전 태스크에서 ViT를 대체할 매우 효율적인 대안이 될 가능성이 높다.
