# Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence

Bo Peng, Daniel Goldstein, Quentin Anthony, et al. (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM) 분야에서 지배적인 Transformer 아키텍처가 가진 치명적인 한계인 입력 시퀀스 길이에 대한 이차 시간 복잡도($O(N^2)$) 문제를 해결하고자 한다. 기존의 RNN은 추론 시 $O(1)$의 시간 복잡도를 가져 효율적이지만, 훈련 시 병렬화가 어렵고 장기 의존성(Long-range dependency)을 포착하는 능력이 Transformer에 비해 떨어진다는 단점이 있다.

RWKV-4는 이러한 Transformer의 병렬 훈련 가능성과 RNN의 효율적인 추론 능력을 결합하여 성능을 끌어올렸으나, 여전히 표현력(Expressivity) 측면에서 개선의 여지가 있었다. 따라서 본 연구의 목표는 RWKV-4를 기반으로 표현력을 극대화하면서도 RNN의 추론 효율성을 유지하는 새로운 아키텍처인 Eagle(RWKV-5)과 Finch(RWKV-6)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 모델의 상태(State)를 벡터에서 행렬 형태로 확장하고, 고정된 감쇠(Decay) 메커니즘을 데이터에 따라 동적으로 변하는 구조로 변경하는 것이다.

1. **Eagle (RWKV-5)**: 벡터 값 상태(Vector-valued states)를 멀티헤드 행렬 값 상태(Multi-headed matrix-valued states)로 확장하여 모델의 기억 용량과 표현력을 높였다. 또한 Receptance 구조를 재설계하고 새로운 게이팅 메커니즘을 도입하였다.
2. **Finch (RWKV-6)**: Eagle의 구조를 더욱 발전시켜, Time-mixing과 Token-shift 모듈에 데이터 의존적(Data-dependent) 기능을 추가하였다. 특히 Low Rank Adaptation(LoRA) 함수를 활용해 학습된 감쇠 벡터를 문맥에 따라 동적으로 보정하는 Dynamic Recurrence 메커니즘을 도입하였다.
3. **데이터 및 토크나이저**: 다국어 성능 향상을 위해 1.12조 개의 토큰으로 구성된 RWKV World v2 데이터셋과 Trie 기반의 탐욕적 매칭(Greedy matching)을 사용하는 고속 다국어 토크나이저를 개발하였다.

## 📎 Related Works

기존의 Linear Attention은 Transformer의 Softmax Attention을 $\phi(Q)\phi(K)^T V$ 형태로 대체하여 복잡도를 낮추려 했으나, 성능 저하 문제가 있었다. Attention Free Transformer(AFT)는 이를 개선하기 위해 학습 가능한 위치 편향(Positional bias)을 도입하였으며, RWKV-4는 이를 채널별 가중치 감쇠(Weight decay) 형태로 재정의하여 RNN의 효율성과 Transformer의 성능을 동시에 확보하였다.

최근에는 Mamba와 같은 State Space Models(SSMs)가 데이터 의존적인 선택 메커니즘을 통해 장기 의존성 문제를 해결하며 주목받고 있다. Finch는 이러한 SSM들의 데이터 의존적 특성을 RWKV 아키텍처에 통합함으로써, 기존의 고정된 감쇠 방식이 가진 한계를 극복하고 Transformer 수준의 표현력을 확보하고자 한다.

## 🛠️ Methodology

### 1. Eagle (RWKV-5)

Eagle은 RWKV-4의 구조를 유지하면서 상태의 차원을 확장하고 게이팅을 강화하였다.

**Token Shift**: 현재 토큰 $x_t$와 이전 토큰 $x_{t-1}$ 사이의 선형 보간(lerp)을 통해 새로운 정보와 이전 정보의 배분 비율을 학습한다.
$$\text{lerp}_{\square}(a, b) = a + (b - a) \odot \mu_{\square}$$

**Time Mixing**: 핵심은 행렬 값 상태(Matrix-valued state)의 도입이다. 각 헤드별로 $\mathbb{R}^{(D/h) \times (D/h)}$ 크기의 상태를 유지하며, 다음과 같이 계산된다.
$$\text{wkv}_t = \text{diag}(u) \cdot k_t^T \cdot v_t + \sum_{i=1}^{t-1} \text{diag}(w)^{t-1-i} \cdot k_i^T \cdot v_i$$
여기서 $w = \exp(-\exp(\omega))$로 정의되어 $0$과 $1$ 사이의 값을 가지며, 상태가 기하급수적으로 감쇠하도록 보장한다. 최종 출력 $o_t$는 다음과 같이 계산된다.
$$o_t = \text{concat}(\text{SiLU}(g_t) \odot \text{LayerNorm}(r_t \cdot \text{wkv}_t)) W_o$$

**Channel Mixing**: RWKV-4와 유사하나, Eagle의 새로운 게이팅 가중치를 수용하기 위해 hidden dimension을 $4D$에서 $3.5D$로 소폭 줄였다.

### 2. Finch (RWKV-6)

Finch는 Eagle의 정적인 구조에 데이터 의존성(Data-dependence)을 부여하여 유연성을 높였다.

**Dynamic Token Shift (DDLerp)**: LoRA 구조를 활용해 입력 값에 따라 보간 계수를 동적으로 결정한다.
$$\text{lora}_{\square}(x) = \lambda_{\square} + \tanh(x A_{\square}) B_{\square}$$
$$\text{ddlerp}_{\square}(a, b) = a + (b - a) \odot \text{lora}_{\square}(a + (b - a) \odot \mu_x)$$

**Dynamic Recurrence**: Eagle에서는 감쇠율 $w$가 학습된 고정 벡터였으나, Finch에서는 이를 데이터 의존적 함수로 변경하여 매 시점 $w_t$가 변하게 한다.
$$w_t = \exp(-\exp(\text{lora}_d(\text{ddlerp}_d(x_t, x_{t-1}))))$$
이로 인해 모델은 입력 문맥에 따라 특정 정보의 기억 유지 기간을 동적으로 조절할 수 있게 된다.

## 📊 Results

### 1. 언어 모델링 성능

다국어 및 영어 벤치마크(LM Evaluation Harness) 결과, Eagle과 Finch는 유사한 파라미터 규모의 타 모델들보다 뛰어난 성능을 보였다. 특히 다국어 벤치마크에서 Pareto frontier를 크게 개선하며, 동일한 훈련 FLOPs 대비 훨씬 높은 정확도를 달성하였다.

### 2. 연상 회상(Associative Recall) 및 장기 문맥

- **MQAR**: Finch는 행렬 값 상태와 동적 재귀 덕분에 기존의 비-Transformer 아키텍처들보다 월등히 높은 정확도를 보였으며, Mamba와 비교해도 우수한 성능을 나타냈다.
- **PG19**: 훈련 시 문맥 길이를 4096으로 제한했음에도 불구하고, Finch는 그 이상의 긴 시퀀스에서도 손실(Loss)이 계속해서 떨어지는 강력한 외삽(Extrapolation) 능력을 보였다.
- **Bamboo**: 4k 문맥 윈도우 테스트에서 Finch와 Eagle은 vanilla Mamba보다 평균 7% 이상 높은 점수를 기록하였다.

### 3. 속도 및 메모리 효율성

A100 80GB 환경에서 측정 결과, Finch는 시퀀스 길이가 4k를 넘어설 때 Flash Attention v2보다 훨씬 빠르며(16k에서 약 4.2배), Mamba보다도 메모리 사용량이 약 17% 적은 것으로 나타났다. 이는 RNN 특유의 $O(1)$ 추론 복잡도와 최적화된 CUDA 커널의 결과이다.

### 4. 멀티모달 확장성

- **VisualRWKV**: Eagle 1.5B/3B 모델을 LLM 백본으로 사용하고 CLIP-L을 인코더로 사용하여, 훨씬 더 큰 모델(7B, 13B)들과 경쟁 가능한 시각-언어 이해 성능을 입증하였다.
- **AudioRWKV**: 2차원 오디오 스펙트로그램 처리를 위해 Quad-directional shift(Q-Shift)를 도입하여, 훨씬 작은 모델 사이즈로도 기존 SOTA 모델(AST-AT)과 대등한 성능을 냈다.

## 🧠 Insights & Discussion

본 논문은 RNN 구조에서도 상태의 차원을 행렬로 확장하고(Matrix-valued states), 그 업데이트 방식을 데이터에 의존적으로(Dynamic Recurrence) 설계하면 Transformer의 표현력에 근접할 수 있음을 보여주었다. 특히 Finch의 LoRA 기반 동적 감쇠 메커니즘은 모델이 중요한 정보는 오래 유지하고 불필요한 정보는 빠르게 망각하게 함으로써, 효율적인 기억 관리를 가능케 한다.

**한계점 및 논의사항**:

- **임베딩 성능**: MTEB 벤치마크에서 임베딩 모델로서의 성능은 기대에 미치지 못했다. 이는 상태 값 자체는 고품질의 문맥 임베딩을 포함하고 있으나, 이를 적절히 집계(Aggregate)하는 방법이 아직 부족하기 때문으로 분석된다.
- **데이터 오염**: 훈련 데이터에 GPT-3.5/ChatGPT의 합성 데이터가 포함되어 있어, 모델이 때때로 자신이 OpenAI에 의해 훈련되었다고 주장하는 등의 동작을 보인다. 이는 아키텍처의 문제가 아닌 데이터셋의 특성에서 기인한 것이다.

## 📌 TL;DR

본 연구는 RWKV-4를 계승하여 행렬 값 상태를 도입한 **Eagle(RWKV-5)**과 데이터 의존적 동적 재귀를 구현한 **Finch(RWKV-6)** 아키텍처를 제안한다. 이를 통해 RNN의 효율성($O(1)$ 추론, 선형 메모리 사용량)을 유지하면서도 Transformer에 필적하는 표현력과 장기 문맥 처리 능력을 확보하였다. 특히 다국어 데이터셋과 최적화된 토크나이저를 통해 범용성을 높였으며, 시각 및 오디오 도메인으로의 확장 가능성을 입증하였다. 향후 MoE(Mixture of Experts) 도입 및 더 큰 규모의 모델 훈련을 통해 효율적인 LLM의 새로운 대안이 될 가능성이 높다.
