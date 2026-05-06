# THE EVOLUTION OF RWKV: ADVANCEMENTS IN EFFICIENT LANGUAGE MODELING

Akul Datta (2024)

## 🧩 Problem to Solve

본 논문은 현대 자연어 처리(NLP)의 주류인 Transformer 아키텍처와 전통적인 순환 신경망(RNN)이 가진 상충되는 한계점을 해결하고자 한다.

Transformer는 Self-attention 메커니즘을 통해 병렬 학습이 가능하고 장거리 의존성(Long-range dependencies)을 잘 포착하지만, 시퀀스 길이 $n$에 대해 시간 및 메모리 복잡도가 $O(n^2)$로 증가하는 Quadratic Complexity 문제를 가지고 있다. 이는 매우 긴 시퀀스를 처리하거나 자원이 제한된 환경에서 추론을 수행할 때 심각한 병목 현상을 야기한다.

반면, RNN은 시퀀스를 순차적으로 처리하여 $O(n)$의 선형 복잡도를 가지므로 추론 효율성이 높지만, 기울기 소실/폭발(Vanishing/Exploding Gradients) 문제로 인해 장거리 의존성을 학습하기 어렵고 학습 과정의 병렬화가 불가능하여 학습 속도가 매우 느리다는 단점이 있다.

따라서 본 논문의 목표는 Transformer의 병렬 학습 효율성과 RNN의 상수 시간($O(1)$) 추론 효율성을 동시에 달성하는 RWKV(Receptance Weighted Key Value) 아키텍처의 발전 과정과 그 확장 가능성을 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Transformer와 RNN의 장점을 결합한 RWKV 아키텍처의 설계 원리와 이를 다양한 도메인으로 확장한 사례들을 체계적으로 분석한 것이다.

중심적인 설계 아이디어는 **Linear Attention** 메커니즘을 통해 Attention 연산을 재구성하여, 학습 시에는 Transformer처럼 병렬적으로 계산하고, 추론 시에는 RNN처럼 이전 상태(Hidden state)를 업데이트하는 순차적 방식으로 동작하게 만드는 것이다. 이를 통해 연산 복잡도를 $O(n)$으로 낮추면서도 대규모 데이터셋에서의 성능을 유지할 수 있게 하였다.

## 📎 Related Works

논문에서는 RWKV의 배경이 되는 다음과 같은 관련 연구들을 소개한다.

1. **Transformer 모델**: Self-attention을 통해 혁신을 일으켰으나, 앞서 언급한 $O(n^2)$ 복잡도와 메모리 요구량, 추론 속도 저하라는 한계가 있다.
2. **RNN 및 변형 모델 (LSTM, GRU)**: 순차적 처리로 효율적이지만, 병렬 학습의 불가능함과 정보 병목(Information Bottleneck) 현상이 존재한다.
3. **Efficient Attention 메커니즘**: Sparse Attention (Longformer), Linear Attention (Performers, Linear Transformers), Local Attention (Transformer-XL) 등이 제안되었으나, 여전히 성능과 효율성 사이의 트레이드오프가 존재한다.
4. **State Space Models (SSMs)**: S4, Mamba와 같이 제어 이론을 이용해 선형 복잡도로 시퀀스를 모델링하는 방식이 제시되었으나, 가변 길이 입력 처리나 비선형 표현력 측면에서 도전 과제가 남아 있다.

RWKV는 이러한 기존 접근 방식들과 달리, RNN의 재귀적 구조를 유지하면서도 Transformer의 병렬 학습 능력을 완전히 수용하는 Linear Attention 구조를 채택함으로써 차별점을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

RWKV의 기본 단위는 **Time-Mixing Block**과 **Channel-Mixing Block**으로 구성되며, 이들은 **Token Shifting** 메커니즘과 함께 잔차 연결(Residual Connection)로 쌓여 있는 구조이다.

### 2. 주요 구성 요소 및 수학적 공식

#### (1) Time-Mixing Block

이 블록은 RWKV의 핵심인 Linear Attention을 수행하며, 네 가지 핵심 벡터 $R$(Receptance), $W$(Weight), $K$(Key), $V$(Value)를 사용한다.

- **Parallel Formulation (학습 시)**: 모든 타임스텝의 가중치 합을 한 번에 계산하여 GPU 병렬 처리를 극대화한다.
$$WKV(K, V, W) = \frac{\sum_{i=1}^{T} \exp(k_i - (T-i)w)v_i}{\sum_{i=1}^{T} \exp(k_i - (T-i)w)}$$
여기서 $w$는 시간적 감쇠(Time-decay) 인자로, 과거 토큰의 영향력을 조절한다.

- **Sequential Formulation (추론 시)**: 이전 상태를 저장하는 누적 변수 $a_t, b_t$를 사용하여 상수 시간 내에 업데이트한다.
$$a_t = \exp(w)a_{t-1} + \exp(k_t)$$
$$b_t = \exp(w)b_{t-1} + \exp(k_t)v_t$$
$$WKV_t = \frac{b_t}{a_t}$$

#### (2) Channel-Mixing Block

Transformer의 Feed-forward 층과 유사하게 특징 차원 간의 비선형 변환을 수행한다. 게이팅 메커니즘을 도입하여 정보 흐름을 제어한다.
$$\text{ChannelMix}(x_t) = \sigma(W_r x_t) \odot (W_v \phi(W_k x_t))$$
여기서 $\sigma$는 Sigmoid 함수, $\phi$는 Squared ReLU 함수이며, $\odot$은 요소별 곱셈(Element-wise multiplication)이다.

#### (3) Token Shifting

현재 토큰과 이전 토큰의 임베딩을 선형 보간하여 시간적 문맥(Temporal Context)을 제공한다.
$$\text{Shift}(x_t, x_{t-1}) = \mu x_t + (1-\mu)x_{t-1}$$
학습 가능한 파라미터 $\mu$를 통해 두 토큰의 혼합 비율을 결정한다.

### 3. 학습 및 추론 절차

- **학습**: Parallel Formulation을 사용하여 Transformer와 동일한 방식으로 모든 토큰을 동시에 처리하여 학습 속도를 높인다.
- **추론**: Sequential Formulation을 사용하여 이전의 상태값만 유지하면 되므로, 시퀀스 길이에 상관없이 토큰 생성당 $O(1)$의 비용만 발생한다.

## 📊 Results

### 1. 자연어 처리 (NLP)

- **성능**: RWKV-v5-7B 모델은 MMLU 등 여러 벤치마크에서 이전 버전인 v4보다 크게 향상된 성능을 보였다.
- **효율성**: WikiText-103 데이터셋에서 Transformer 모델보다 적은 파라미터로 더 낮은 Perplexity(더 높은 예측 성능)를 달성하였다.

### 2. 컴퓨터 비전 (Vision-RWKV)

- **ImageNet 분류**: Vision-RWKV-L 모델은 Top-1 Accuracy 86.0%를 기록하며, ViT-B/32(83.4%) 및 ResNet-50(76.6%) 대비 경쟁력 있는 성능을 보였다.
- **특징**: 2D Token Shifting과 양방향 Attention을 도입하여 이미지의 공간적 관계를 효율적으로 포착하였다.

### 3. 멀티모달 학습 (RWKV-CLIP)

- **이미지-텍스트 검색**: Flickr30k 및 MSCOCO 데이터셋에서 R@1 지표 기준, CLIP-ViT-B/32를 크게 상회하는 성능(MSCOCO 기준 50.3% vs 20.8%)을 보여주었다.

### 4. 의료 영상 복원 (Restore-RWKV)

- **MRI 초해상도(Super-resolution)**: PSNR 32.091, SSIM 0.9408을 기록하여 SRCNN, VDSR, SwinIR 등 기존의 CNN 및 Transformer 기반 모델보다 우수한 복원 성능을 보였다.

### 5. 3D 포인트 클라우드 (PointRWKV)

- **ModelNet40 분류**: 96.89%의 정확도를 달성하며 PointNet++(90.7%) 및 DGCNN(92.9%)보다 높은 성능을 보였으며, 연산 효율성 또한 유지하였다.

## 🧠 Insights & Discussion

### 강점

RWKV는 **"학습은 Transformer처럼, 추론은 RNN처럼"**이라는 목표를 성공적으로 구현하였다. 특히 $O(n)$의 선형 복잡도를 유지하면서도 텍스트를 넘어 이미지, 3D 데이터, 의료 영상 등 다양한 모달리티에 유연하게 적용될 수 있음을 입증하였다. 이는 하드웨어 자원이 제한된 엣지 디바이스나 초거대 문맥(Context) 처리가 필요한 서비스에 매우 강력한 대안이 될 수 있다.

### 한계 및 미해결 질문

1. **이론적 근거**: RWKV의 장거리 의존성 모델링 능력이 수학적으로 어떻게 보장되는지에 대한 엄밀한 증명이 부족하며, Transformer와의 표현력(Expressiveness) 차이에 대한 이론적 분석이 더 필요하다.
2. **확장성(Scaling)**: 모델 파라미터를 100B 이상으로 극단적으로 늘렸을 때, Transformer에서 관찰되는 Scaling Law가 동일하게 적용되는지에 대한 추가 검증이 필요하다.
3. **해석 가능성**: 내부 상태(Hidden state)가 어떻게 정보를 압축하고 유지하는지에 대한 해석 가능성(Interpretability) 연구가 부족하다.

### 비판적 해석

본 논문은 RWKV의 다양한 변형 모델들을 리뷰하는 형식으로 작성되어, 개별 모델의 구체적인 하이퍼파라미터나 학습 디테일보다는 결과적인 성능 수치에 치중하는 경향이 있다. 또한, 대부분의 비교 대상이 구버전 모델이거나 특정 조건 하의 모델들이므로, 최신 SOTA(State-of-the-art) 모델들과의 더 엄격한 비교 분석이 수반되어야 한다.

## 📌 TL;DR

본 논문은 Transformer의 병렬 학습 능력과 RNN의 추론 효율성을 결합한 **RWKV 아키텍처의 진화 과정과 다방면의 적용 사례**를 분석한다. RWKV는 Linear Attention과 Token Shifting을 통해 연산 복잡도를 $O(n)$으로 낮추었으며, 이를 통해 NLP뿐만 아니라 컴퓨터 비전, 의료 영상, 3D 포인트 클라우드 처리 등 다양한 분야에서 기존 Transformer 기반 모델 대비 높은 효율성과 경쟁력 있는 성능을 보였다. 향후 이론적 체계 확립과 초거대 모델로의 스케일업이 이루어진다면, 차세대 효율적 딥러닝 아키텍처로서 핵심적인 역할을 할 가능성이 높다.
