# An Attention Free Transformer

Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, Josh Susskind

## 🧩 Problem to Solve

Transformer 모델은 시퀀스 내의 모든 요소 간의 직접적인 상호작용을 가능하게 하여 장기 의존성(long-term dependencies)을 포착하는 데 탁월하지만, 표준 셀프 어텐션 연산의 컨텍스트 크기(sequence length)에 대한 이차(quadratic) 시간 및 공간 복잡도로 인해 높은 계산 비용을 초래합니다. 이는 특히 입력 시퀀스 길이가 길거나 모델 크기가 커질 때 Transformer의 확장성을 저해하는 주된 원인입니다. 기존의 많은 연구들이 희소성(sparsity), LSH(locality sensitive hashing), 저랭크 분해(low-rank decomposition), 커널 근사(kernel approximation) 등을 통해 완전한 어텐션 연산을 근사하려고 시도했지만, 본 논문은 이러한 근사 방식 없이 어텐션 연산을 완전히 제거하는 새로운 접근 방식을 제안합니다.

## ✨ Key Contributions

- **어텐션 없는 Transformer (AFT) 도입:** 기존의 점곱(dot product) 셀프 어텐션을 대체하는 효율적인 연산 모듈인 AFT를 제안합니다.
- **선형 메모리 복잡도 달성:** AFT 레이어는 키(key)와 값(value)을 학습된 위치 편향(position biases)과 결합한 후 쿼리(query)와 요소별 곱셈(element-wise multiplication)을 수행하여 컨텍스트 크기($T$) 및 피처 차원($d$)에 대해 선형적인 메모리 복잡도 $O(T d)$를 가집니다. 이는 대규모 입력 및 모델 크기에 적합합니다.
- **AFT 변형 모델 제안:** 지역성(locality)과 공간 가중치 공유(spatial weight sharing) 아이디어를 활용하면서도 전역 연결성(global connectivity)을 유지하는 AFT-local 및 AFT-conv 두 가지 변형 모델을 소개합니다.
- **경쟁력 있는 성능 및 효율성:** 이미지 자기회귀 모델링 (CIFAR10), 문자 레벨 언어 모델링 (Enwik8), 이미지 분류 (ImageNet-1K) 세 가지 벤치마크에서 기존 Transformer 모델 및 다른 효율적인 변형 모델들과 비교하여 경쟁력 있는 성능과 뛰어난 효율성(속도 및 메모리)을 입증합니다.
- **다양한 디자인 선택 및 특성 분석:** AFT의 여러 디자인 선택에 대한 광범위한 Ablation 연구를 수행하고, Transformer와의 호환성, 희소성, 가변 크기 입력 처리 등 AFT의 고유한 특성들을 논의합니다.

## 📎 Related Works

- **점곱 근사(Approximating the dot product):** Reformer, Linear Transformer, Performer 등은 점곱 연산을 근사하여 효율성을 높였으나, AFT는 점곱 자체를 제거합니다. Linear Transformer와 Performer는 $O(T d^2)$ 복잡도를 가지지만 AFT는 $O(T d)$ 복잡도를 가집니다.
- **희소 및 지역 어텐션(Sparse, local attention):** Sparse Transformer, Image Transformer 등은 고정된 희소 또는 지역 컨텍스트 패턴을 사용합니다. AFT-local 및 AFT-conv는 지역성 아이디어를 차용하지만, 엄격한 제약보다는 유도 편향(inductive bias)으로 활용하여 전역 연결성을 유지합니다.
- **컨텍스트 압축(Context compression):** Adaptive-Span Transformer, Routing Transformer, Linformer, Compressive Transformer 등은 컨텍스트 길이를 줄이거나 압축된 표현에 어텐션합니다. AFT는 연산 수준에서 시퀀스 복잡도를 개선하는 데 중점을 둡니다.
- **점곱 어텐션 제거(Eliminating dot product attention):** Synthesizer는 입력에서 어텐션 가중치를 예측하고, LightConv는 동적 경량 컨볼루션을 사용하며, Sinkhorn Transformer는 미분 가능한 정렬 연산을 사용합니다. AFT는 이 분야에서 강력한 성능과 효율성을 제공하는 새로운 접근 방식을 제시합니다.
- **비전용 MLP (MLPs for vision):** MLP-Mixer와 같은 동시 연구들은 비전 작업에서 어텐션 대신 MLP를 탐구합니다. AFT도 유사하게 볼 수 있지만, 키와 위치 편향으로 구성되고 정규화되는 정교한 게이팅 메커니즘을 갖추고 있으며, AFT-conv는 CNN의 이점을 계승합니다.

## 🛠️ Methodology

AFT(Attention Free Transformer)는 기존 Transformer의 Multi-Head Attention (MHA) 모듈을 대체하는 플러그인 모듈입니다.

1. **AFT (Attention Free Transformer) 기본 연산:**

   - 입력 $X$를 선형 변환하여 쿼리 $Q=XW_Q$, 키 $K=XW_K$, 값 $V=XW_V$를 얻습니다.
   - 다음 연산을 수행합니다:
     $$ Y*t = \sigma_q(Q_t) \odot \frac{\sum*{t'=1}^T \exp(K*{t'} + w*{t,t'}) \odot V*{t'}}{\sum*{t'=1}^T \exp(K*{t'} + w*{t,t'})} $$
        여기서 $\odot$는 요소별 곱셈(element-wise product)을 의미하고, $\sigma_q$는 쿼리에 적용되는 비선형 함수(기본값은 시그모이드)입니다. $w \in \mathbb{R}^{T \times T}$는 학습된 쌍별 위치 편향(pair-wise position biases)입니다.
   - 이 연산은 명시적인 어텐션 행렬 계산 없이도 쿼리와 값 사이의 전역 상호작용을 유지하며, 메모리 복잡도는 $O(T d)$입니다.
   - 각 피처 차원마다 독립적인 어텐션 벡터를 가지는 "암묵적 어텐션"으로 해석될 수 있습니다.

2. **AFT-full:** 위에서 정의된 AFT의 기본 버전입니다.

3. **AFT-local:**

   - 학습된 위치 편향 $w_{t,t'}$를 지역 영역($|t-t'| < s$)으로 제한합니다 ($s$는 지역 윈도우 크기).
   - $w_{t,t'} = w_{t,t'}$ if $|t-t'| < s$, else $0$.
   - 효율성을 높이면서도 전역 연결성(global connectivity)은 유지됩니다.

4. **AFT-simple:**

   - AFT-local에서 $s=0$인 극단적인 형태로, 위치 편향을 학습하지 않습니다.
   - 연산이 $Y_t = \sigma_q(Q_t) \odot \frac{\sum_{t'=1}^T \exp(K_{t'}) \odot V_{t'}}{\sum_{t'=1}^T \exp(K_{t'})}$ 으로 단순화됩니다.
   - $O(T d)$ 복잡도로 점곱 연산을 완전히 제거합니다.

5. **AFT-conv:**

   - 지역성 아이디어를 확장하여 공간 가중치 공유(spatial weight sharing), 즉 컨볼루션 개념을 통합합니다.
   - $w_{t,t'}$가 상대적 위치에만 의존하게 하여 여러 '헤드'를 학습할 수 있습니다.
   - AFT-conv는 전역 연결성, 비음수 컨볼루션 가중치, 정교한 나누기/곱하기 게이팅 메커니즘을 가진 특수 컨볼루션 레이어로 해석됩니다.
   - 가변 크기 입력 처리 능력을 가집니다.

6. **위치 편향 파라미터화 (Parameterization):**
   - **AFT-full 및 AFT-local:** 위치 편향 $w$를 $w_{t,t'} = u_t^T v_{t'}$와 같이 팩터화된 형태로 사용합니다. ($u, v \in \mathbb{R}^{T \times d'}$), 이는 파라미터 수를 크게 줄이고 성능을 향상시킵니다.
   - **AFT-conv:** 학습 가능한 스케일 및 바이어스 파라미터 $\gamma_i, \beta_i$를 사용하여 $w_i = \gamma_i \frac{w_i - \text{mean}(w_i)}{\text{std}(w_i)} + \beta_i$와 같이 재파라미터화합니다.

## 📊 Results

- **이미지 자기회귀 모델링 (CIFAR10):**
  - AFT-local(L=24, d=256)은 2.74 bits/dim으로 SOTA 성능을 달성하여 Sparse Transformer, Image Transformer, 표준 Transformer를 능가했습니다.
  - 표준 Transformer 대비 24% 더 빠르고 (1.67 Iters/Sec vs 1.35), 메모리 사용량은 절반 (12.8 GB/GPU vs 30.4 GB/GPU)이었습니다.
  - 팩터화된 파라미터화는 9.6M에서 0.6M으로 파라미터 수를 크게 줄이고 성능도 향상시켰습니다.
- **문자 레벨 언어 모델링 (Enwik8):**
  - AFT-local(L=24, d=256, window=32)은 테스트 bpc 1.154로 경쟁력 있는 성능을 보였으며, Reformer, Synthesizer, Linear Transformer, Performer를 능가했습니다.
  - 표준 Transformer 대비 44% 더 빠르고 메모리 사용량은 1/3에 불과했습니다.
  - 지역 윈도우 크기 32에서 최적의 성능을 보였고, 시퀀스 길이가 길어질수록 (T=4096) 성능이 더욱 개선되었습니다.
- **이미지 분류 (ImageNet-1K):**
  - AFT-full은 DeiT Transformer와 유사한 Top-1 정확도(예: 'small' 설정에서 79.8% vs 79.9%)를 달성하며 더 나은 메모리 효율성을 보였습니다.
  - AFT-conv는 DeiT 대비 Top-1 정확도를 크게 향상시켰습니다 (예: 'tiny'에서 2.6%, 'small'에서 1.1% 절대 개선). 파라미터 수는 유사하거나 더 적고, 메모리 효율성이 좋으며, 속도는 비슷했습니다.
  - AFT-conv는 훈련 시와 다른 가변 크기 입력(224x224 훈련 후 384x384에서 81.0% $\rightarrow$ 81.6%)에서도 성능 향상을 보여 완전히 컨볼루션적(fully convolutional)임을 입증했습니다.
  - 사전 훈련된 DeiT 모델 가중치로 AFT-conv를 파인튜닝했을 때, 무작위 초기화보다 훨씬 높은 정확도(83.4% vs 81.6%)를 달성하여 Transformer와의 높은 호환성을 보여주었습니다.
  - 쿼리 항(query term)의 기여가 중요하며 (제거 시 성능 크게 하락), 전역 연결성 또한 중요함이 확인되었습니다.
  - 학습된 위치 편향에서 지역적이고 대칭적인 희소 패턴이 나타났으며, 희소성 정규화를 통해 정확도 향상과 더 높은 희소성을 동시에 달성했습니다.
  - 극단적인 희소성(각 헤드당 하나의 위치 편향만 학습)에서도 79.9%의 높은 정확도를 유지했습니다.

## 🧠 Insights & Discussion

- AFT는 점곱 어텐션을 효율적인 새로운 연산으로 성공적으로 대체하여 Transformer 모델의 가장 큰 병목 중 하나인 이차 복잡도 문제를 해결했습니다.
- 선형적인 메모리 복잡도는 대규모 시퀀스 및 모델에서 Transformer를 적용하는 새로운 가능성을 열었습니다.
- AFT-local 및 AFT-conv를 통해 지역성(locality)이 강력한 유도 편향임을 입증했으며, 이는 효율성과 성능을 동시에 향상시킬 수 있음을 보여줍니다. 특히, AFT-conv는 전역 연결성을 유지하면서도 컨볼루션의 이점을 활용하는 새로운 형태의 모델 디자인을 제시합니다.
- 위치 편향의 팩터화된 파라미터화 및 재파라미터화는 모델의 효율성과 성능에 매우 중요하며, 이는 모델의 학습 능력을 최적화하는 데 핵심적인 요소였습니다.
- 학습된 "키"는 레이어가 깊어질수록 "객체 감지기"처럼 작동하는 흥미로운 특성을 보였으며, 이는 AFT 내부의 특징 추출 메커니즘에 대한 통찰을 제공합니다.
- 비록 계산량이 적지만 "쿼리" 항의 존재는 AFT의 성능에 결정적인 기여를 합니다.
- 학습된 위치 편향은 자연적으로 희소성을 띠며, 이는 정규화나 Gumbel-softmax를 통해 더욱 극대화될 수 있어 모델 압축 및 경량화 가능성을 시사합니다.
- 사전 훈련된 Transformer 가중치로 AFT-conv를 파인튜닝했을 때 성능 향상이 확인되어, 기존 Transformer 생태계와의 호환성 및 재활용 가능성을 보여줍니다.
- 이 연구는 Transformer와 유사한 모델을 위한 새로운 디자인 공간을 개척하며, 셀프 어텐션이 필요한 다양한 분야에 영향을 미칠 것으로 기대됩니다.

## 📌 TL;DR

Transformer의 이차 복잡도 셀프 어텐션 문제를 해결하기 위해, 본 논문은 점곱 어텐션을 제거한 **Attention Free Transformer (AFT)**를 제안합니다. AFT는 키, 값, 학습된 위치 편향을 결합한 후 쿼리와 요소별 곱셈을 수행하여, 시퀀스 길이 및 피처 차원에 대해 **선형적인 메모리 복잡도 $O(T d)$**를 달성합니다. 지역성(locality)과 공간 가중치 공유를 활용한 **AFT-local 및 AFT-conv** 변형 모델은 전역 연결성을 유지하면서 효율성을 극대화합니다. 광범위한 실험 결과, AFT는 이미지 및 언어 모델링, 이미지 분류 등 다양한 벤치마크에서 기존 Transformer 및 효율적인 변형 모델들을 능가하거나 동등한 성능을 보이면서 **훨씬 뛰어난 속도와 메모리 효율성**을 입증했습니다. 특히 AFT-conv는 완전히 컨볼루션적이며 가변 크기 입력 처리가 가능하고, 사전 훈련된 Transformer 모델과의 호환성도 보여줍니다.
