# VIDEORWKV: VIDEO ACTION RECOGNITION BASED RWKV

Zhuowen Yin, Chengru Li, Xingbo Dong (2024)

## 🧩 Problem to Solve

비디오 이해(Video Understanding) 작업의 핵심은 시공간적 특징(Spatio-temporal features)을 효과적으로 캡처하는 것이다. 그러나 기존의 3D-CNN이나 Transformer 기반 아키텍처는 다음과 같은 한계점을 가진다.

첫째, Transformer의 Attention 메커니즘은 연산 복잡도가 입력 시퀀스 길이의 제곱에 비례하는 quadratic complexity를 가지므로, 고해상도 또는 장시간 비디오를 처리할 때 막대한 계산 자원이 소모된다. 둘째, CNN 기반 방식은 국소적인(local) 정보에 지나치게 집중하여 비디오 데이터 전반의 광범위한 시공간적 의존성을 놓치는 경향이 있다. 셋째, 비디오 데이터에는 중복된 정보가 매우 많아 이를 효율적으로 처리하면서도 중요한 피사체에 집중하는 능력이 요구된다.

본 논문의 목표는 연산 효율성을 극대화하는 linear complexity를 유지하면서, 장기 의존성 문제를 해결하고 중복 정보를 제거하여 비디오 행동 인식(Video Action Recognition) 성능을 높이는 LSTM-CrossRWKV (LCR) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 RWKV의 선형 복잡도 Attention 메커니즘과 LSTM의 장기 기억 능력을 결합하고, 여기에 'Edge Prompt'라는 외부 정보를 주입하여 모델이 중요한 피사체에 더 잘 집중하게 만드는 것이다.

주요 기여 사항은 다음과 같다.

1. **LSTM-CrossRWKV (LCR) 프레임워크 제안**: LSTM 구조와 Cross RWKV 블록을 융합하여 시공간 표현 학습을 효율적으로 수행하는 새로운 재귀 유닛(recurrent cell)을 설계하였다.
2. **CrossRWKV 게이트 도입**: 현재 프레임의 Edge 정보를 Key($K$)와 Value($V$)로 사용하고, 과거의 시간적 정보와 현재 프레임 정보를 Receiver($R$) 벡터로 통합하여 피사체 중심의 특징 추출을 가능하게 하였다.
3. **Edge Prompt 기반 메모리 관리**: Canny 연산자로 추출한 Edge 정보를 LSTM의 forgetting gate에 연결함으로써, 배경 소음을 줄이고 중요한 정보만을 장기 기억에 유지하도록 유도하였다.
4. **효율성 증명**: Kinetics-400, Something-Something V2, Jester 데이터셋에서 기존 SOTA 모델 대비 적은 파라미터와 연산량으로 경쟁력 있는 성능을 달성하였다.

## 📎 Related Works

비디오 이해를 위한 기존 접근 방식은 크게 두 가지 흐름으로 나뉜다.

1. **CNN 기반 방식**: 2D CNN과 RNN의 결합, 혹은 3D Convolution을 통해 시공간 특징을 추출한다. ConvLSTM, PredRNN, E3D-LSTM 등이 대표적이다. 이러한 모델들은 국소적 특징 추출에 능숙하지만, 전역적인 시공간 의존성을 캡처하는 데 한계가 있다.
2. **Transformer 기반 방식**: ViT, Swin Transformer, TimeSformer 등이 등장하며 전역적 컨텍스트 캡처 능력을 비약적으로 향상시켰다. 그러나 $\mathcal{O}(n^2)$의 시간 복잡도로 인해 긴 비디오 시퀀스 처리 시 메모리 및 계산 비용이 급격히 증가한다.

최근에는 Mamba와 같은 State Space Models (SSMs)나 RWKV와 같이 RNN의 선형 복잡도와 Transformer의 성능을 동시에 잡으려는 시도가 늘고 있다. 본 논문은 이러한 RWKV의 이점을 비디오 도메인으로 확장하며, 특히 단순한 적용을 넘어 LSTM과 Edge Prompt를 결합하여 비디오 특유의 중복성 문제를 해결하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인

입력 비디오 $X_v \in \mathbb{R}^{3 \times T \times H \times W}$는 먼저 $1 \times 16 \times 16$ 크기의 3D convolution을 통해 $L$개의 겹치지 않는 시공간 패치 $x \in \mathbb{R}^{L \times C}$로 분할된다. 이후 학습 가능한 classification token $X_{cls}$와 Rotary Position Encoding (RoPE)이 추가되어 LCR 블록으로 입력된다.

### 2. Cross RWKV Gate

Cross RWKV 게이트는 현재 프레임 $x_t$와 그에 대응하는 Edge 이미지 $x^e_t$, 그리고 이전 상태인 $h_{t-1}, c_{t-1}$을 입력으로 받는다.

- **Spatial Mixing**: 전역 Attention 역할을 수행한다. 현재 이미지와 Edge 정보는 세 개의 병렬 선형 층을 통과하여 다음과 같은 행렬을 생성한다.
  - $R, G$ (Receiver, Gate): 현재 프레임과 이전 hidden state의 결합에서 유도된다.
  - $K, V$ (Key, Value): 현재 프레임의 **Edge 정보**에서 유도된다.
- **Attention 계산**: Linear complexity의 bidirectional attention을 사용하여 다음과 같이 계산한다.
  $$a_t = \text{concat}(\text{SiLU}(G_t) \odot \text{LN}(R_t \cdot \text{Bi-wKV}_t)) W_a$$
  여기서 $\text{Bi-wKV}_t$는 다음과 같이 정의되며, 이는 $\mathcal{O}(n)$의 복잡도를 가진다.
  $$\text{wKV}_t = \text{diag}(u) \cdot K^T_t \cdot V_t + \sum_{i=1}^{t-1} \text{diag}(w)^{t-1-i} \cdot K^T_i \cdot V_i$$
- **Channel Mixing**: 이후 channel-wise fusion을 거쳐 최종 출력 $A'_t$를 생성하며, 이때 $\sigma(R'_t)$ 게이트가 출력을 제어한다.

### 3. Edge Prompt Learning

현재 프레임 $x_t$에 Canny 연산자를 적용하고 Otsu's method로 임계값을 적응적으로 결정하여 Edge 이미지 $x^e_t$를 추출한다. 추출된 정보는 zero-initialized convolution 레이어를 통해 임베딩된다. 이 Edge 정보는 LSTM의 forgetting gate에 입력되어 배경 노이즈를 억제하고 중요한 객체 정보만을 유지하도록 돕는다.

### 4. LCR Unit 및 학습 절차

LCR 유닛은 CrossRWKV 게이트의 출력 $A_t$와 Edge 정보 $x^e_t$를 사용하여 LSTM의 상태를 업데이트한다.

- **Cell State 업데이트**:
  $$C_t = x^e_t \times (\text{Tanh}(A_t) + C_{t-1}) + C_{t-1}$$
- **Hidden/Output State 업데이트**:
  $$O_t = H_t = \text{Tanh}(C_t) \times \sigma(A_t)$$
- **최종 분류**: 최종 레이어의 $[CLS]$ 토큰 표현인 $X^O_{cls}$와 $X^C_{cls}$를 결합(concat)하고 LayerNorm 및 선형 층 $W_{class}$를 통과시켜 클래스를 예측한다.
  $$\text{result} = W_{class} \times \text{LN}(\text{Concat}(X^O_{cls}, X^C_{cls}))$$

또한, 비디오 내 중복 정보를 줄이기 위해 **Tube Masking** 전략을 사용하여 과적합을 방지하고 효율성을 높였다.

## 📊 Results

### 실험 설정

- **데이터셋**: Kinetics-400 (K400), Something-Something V2 (SSv2), Jester.
- **평가 지표**: Top-1, Top-5 Accuracy, 파라미터 수, FLOPs, 메모리 사용량.
- **구현 세부사항**: K400과 SSv2는 ImageNet-1K로 사전 학습된 Vision RWKV 가중치를 사용하였으며, Jester는 scratch부터 학습하였다.

### 주요 결과 (Jester 데이터셋 중심)

LCR 모델은 매우 적은 연산 자원으로도 기존의 무거운 모델들을 능가하는 성능을 보였다.

- **정량적 성능**: LCR은 Top-1 정확도 **90.83%**를 달성하였다.
- **비교 분석**:
  - **TimeSformer-L**: 정확도 89.94%, 파라미터 46.6M, FLOPs 1.568T.
  - **ResNet3D-50**: 정확도 90.75%, 파라미터 46.6M, FLOPs 50.2T.
  - **LCR (본 제안 모델)**: 정확도 90.83%, 파라미터 **5.14M**, FLOPs **0.022T**.

LCR은 ResNet3D-50 대비 파라미터 수는 약 1/9 수준으로 줄였으며, 연산량(FLOPs)은 극단적으로 낮추었음에도 불구하고 더 높은 정확도를 기록하였다.

## 🧠 Insights & Discussion

본 논문은 RWKV의 선형 복잡도와 LSTM의 기억 메커니즘을 적절히 융합하여 비디오 이해 작업에서 효율성과 정확도라는 두 마리 토끼를 잡았다. 특히 Edge 정보를 Prompt로 사용하여 모델의 Attention을 피사체에 강제로 집중시킨 점과, 이를 LSTM의 forgetting gate와 연결하여 메모리 관리에 활용한 점이 매우 영리한 설계이다.

다만, 다음과 같은 한계점이 존재한다.

1. **병렬 처리의 한계**: 클래식한 LSTM 구조를 그대로 사용했기 때문에, Transformer나 CNN과 달리 학습 및 추론 시의 병렬 연산 능력이 떨어진다. 이는 시퀀스 길이가 매우 길어질 때 학습 속도 저하의 원인이 될 수 있다.
2. **기울기 문제**: 저자 스스로 언급했듯이, 고전적 LSTM 구조로 인해 Gradient Vanishing 또는 Explosion 문제에서 완전히 자유롭지 못하다.
3. **추측 가능한 확장성**: 현재는 행동 인식 작업에 집중되어 있으나, 제안된 LCR 구조를 더 큰 네트워크로 스케일업(scale-up)했을 때 성능 향상 폭이 어느 정도일지는 명확히 제시되지 않았다.

결론적으로, 본 연구는 고비용의 Transformer 구조를 대체할 수 있는 효율적인 대안으로서 linear complexity 기반의 recurrent 모델이 비디오 도메인에서도 강력한 성능을 낼 수 있음을 입증하였다.

## 📌 TL;DR

본 논문은 LSTM과 RWKV를 결합한 **LSTM-CrossRWKV (LCR)** 프레임워크를 제안하여, 비디오 행동 인식에서 **$\mathcal{O}(n)$의 선형 복잡도**로 고효율-고성능 학습을 구현하였다. 특히 **Edge Prompt**를 통해 피사체 집중도를 높이고 배경 노이즈를 제거함으로써, 기존 3D-CNN 및 Transformer 모델 대비 훨씬 적은 파라미터(약 5.14M)와 연산량으로도 더 높은 정확도를 달성하였다. 이 연구는 실시간 비디오 분석 및 자원이 제한된 환경에서의 비디오 이해 시스템 구축에 매우 중요한 기여를 할 가능성이 높다.
