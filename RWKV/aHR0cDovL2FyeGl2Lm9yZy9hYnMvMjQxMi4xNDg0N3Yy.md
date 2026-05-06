# A Survey of RWKV

Zhiyuan Li, Tingyu Xia, Yi Chang and Yuan Wu (2024)

## 🧩 Problem to Solve

현대 딥러닝의 주류인 Transformer 아키텍처는 Self-Attention 메커니즘을 통해 입력 시퀀스의 장거리 의존성(long-range dependencies)을 매우 효과적으로 포착한다. 그러나 Self-Attention의 계산 복잡도가 시퀀스 길이 $T$에 대해 제곱 비례($O(T^2)$)하는 특성 때문에, 시퀀스가 길어질수록 메모리 사용량이 급증하고 추론 속도가 느려지는 심각한 효율성 문제가 발생한다.

반면, 전통적인 Recurrent Neural Networks(RNN)는 시퀀스를 순차적으로 처리하여 계산 복잡도를 선형 수준($O(T)$)으로 유지하지만, 병렬 연산이 불가능하여 학습 속도가 느리고, 기울기 소실/폭주(vanishing/exploding gradients) 문제로 인해 매우 긴 시퀀스의 정보를 유지하는 데 한계가 있다.

본 논문은 이러한 Transformer의 고비용 구조와 RNN의 학습 효율성 및 장기 기억 상실 문제를 동시에 해결하고자 하는 RWKV(Receptance Weighted Key Value) 모델의 구조, 진화 과정 및 다양한 응용 분야를 체계적으로 분석하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 RWKV 모델에 대한 최초의 종합적인 리뷰를 제공한다는 점에 있다. RWKV의 중심 아이디어는 **RNN의 선형 복잡도와 Transformer의 병렬 학습 및 성능을 결합**하는 것이다.

구체적으로는 다음과 같은 설계 직관을 가진다:

1. **Linear Attention의 도입**: 기존의 $O(T^2)$ Attention을 선형 복잡도 $O(T)$로 대체하여 연산 효율을 극대화한다.
2. **재귀적 구조와 병렬 학습의 조화**: 추론 시에는 RNN처럼 상태(state)를 유지하며 효율적으로 동작하고, 학습 시에는 Transformer처럼 데이터를 병렬로 처리할 수 있도록 설계한다.
3. **모델의 진화 추적**: 초기 RWKV-4부터 Matrix-valued states를 도입한 RWKV-5(Eagle), 데이터 기반 동적 재귀 메커니즘을 적용한 RWKV-6(Finch)까지의 발전 과정을 상세히 분석한다.

## 📎 Related Works

논문은 RWKV의 위치를 정의하기 위해 다음과 같은 관련 연구들을 설명한다.

1. **RNN 및 LSTM/GRU**: 시퀀스 데이터를 처리하는 기초적인 도구이나, 기울기 소실 문제로 인해 장거리 의존성 포착 능력이 떨어진다는 한계가 있다.
2. **Transformers**: Self-Attention을 통해 병렬화와 고성능을 달성했으나, $O(T^2)$의 복잡도로 인해 긴 시퀀스 처리에 비효율적이다.
3. **Attention Free Transformer (AFT)**: 행렬 곱셈을 제거하여 메모리 복잡도를 $O(Td)$로 낮추었으나, 시간 복잡도 측면에서는 여전히 개선의 여지가 남아 있었다.
4. **기타 선형 모델**:
    - **Linear Transformers**: 커널 방법을 통해 복잡도를 $O(Td^2)$로 낮추었으나, RWKV의 $O(Td)$보다 연산량이 많고 표현력이 일부 저하될 수 있다.
    - **Mamba**: State-Space Models(SSM)를 활용하여 선형 확장성을 달성했다. 시퀀스 분할 작업에서는 RWKV보다 우수하지만, 추론 속도와 메모리 효율 면에서는 RWKV가 더 유리한 측면이 있다.
    - **RetNet**: Retention 메커니즘을 통해 병렬 학습과 저비용 추론을 동시에 달성하며, 고차원 상태를 보존하여 RWKV보다 높은 표현력을 보여준다.

## 🛠️ Methodology

RWKV는 기본적으로 Residual Block들이 쌓인 구조이며, 각 블록은 **Time-Mixing**과 **Channel-Mixing**이라는 두 가지 서브 블록으로 구성된다.

### 1. RWKV-4의 핵심 메커니즘

RWKV-4는 선형 Attention을 통해 다음과 같이 동작한다.

- **Time-Mixing**: Transformer의 Self-Attention과 유사한 역할을 수행하며, 전역적인 상호작용을 담당한다.
  - $r_t$ (Receptance): 과거 정보의 수용량을 결정하는 벡터이다.
  - $w$ (Weight): 위치 가중치 감쇠(position weight decay) 벡터로, 학습 가능한 파라미터이다.
  - $k_t, v_t$ (Key, Value): 기존 Attention의 Key, Value와 유사한 역할을 한다.
  - **핵심 방정식**:
      $$wkv_t = \frac{\sum_{i=1}^{t-1} \exp(-(t-1-i)w + k_i) \odot v_i + \exp(u + k_t) \odot v_t}{\sum_{i=1}^{t-1} \exp(-(t-1-i)w + k_i) + \exp(u + k_t)}$$
      이 식은 과거 토큰들과 현재 토큰 간의 관계를 지수적 감쇠 합(exponential decay sum)으로 계산하여, linear attention의 특성을 갖게 하며 계산 복잡도를 $O(Td)$로 유지한다.

- **Channel-Mixing**: 개별 토큰 내의 특징 차원(feature dimension) 간의 상호작용을 모델링한다. 입력 $x_t$와 $x_{t-1}$의 가중합을 통해 새로운 $r'_t, k'_t$를 계산하고, 이를 통해 특징 벡터의 값을 갱신한다.

### 2. 모델의 진화 (v5 $\rightarrow$ v6)

- **RWKV-5 (Eagle)**: 기존의 벡터 기반 상태(vector-valued states)를 **행렬 기반 상태(matrix-valued states)**로 확장하여 표현력을 높였다. 또한, Multi-head 구조와 새로운 학습 감쇠 전략을 도입하였다.
- **RWKV-6 (Finch)**: 데이터 기반의 선형 보간법(ddlerp)과 **LoRA(Low-Rank Adaptation)** 함수를 도입하였다. 특히, 감쇠 벡터 $w$가 정적인 값이 아니라 입력 데이터에 따라 동적으로 변화하도록 설계하여 문맥 적응력을 강화하였다.

## 📊 Results

본 논문은 특정 실험 결과보다는 RWKV가 적용된 광범위한 생태계와 벤치마크 분석 결과에 집중한다.

### 1. 응용 분야 (Applications)

RWKV는 효율적인 추론 속도 덕분에 매우 다양한 분야에 적용되고 있다.

- **자연어 생성 (NLG)**: 웹소설 작성, 챗봇(QQ, Telegram 등), RAG 기반 질의응답 시스템.
- **자연어 이해 (NLU)**: 몽골어-중국어 기계 번역, 텍스트 분류, 가상 비서.
- **컴퓨터 비전 (CV)**: 의료 영상 분할(BSBP-RWKV), 3D 포인트 클라우드 인식(PointRWKV), 실시간 행동 감지(TLS-RWKV).
- **오디오 및 음악**: MIDI 음악 생성, 실시간 자동 음성 인식(ASR).

### 2. 모델 평가 (Evaluation)

다양한 벤치마크(RULER, S3EVAL, CMATH 등)를 통해 분석한 결과는 다음과 같다.

- **강점**: 추론 속도와 확장성이 뛰어나며, 특정 재귀적 처리 작업에서는 Transformer보다 우수한 성능을 보인다.
- **한계**:
  - **장거리 의존성**: 매우 긴 컨텍스트에서는 정보 감쇠(information decay) 현상이 발생하여 성능이 하락하는 경향이 있다.
  - **추론 능력**: 수학적 문제 해결(CMATH)이나 복잡한 공간 추론(MANGO) 작업에서는 GPT-4와 같은 거대 Transformer 모델에 비해 정확도가 낮다.

## 🧠 Insights & Discussion

### 강점 및 기회

RWKV는 Transformer의 성능과 RNN의 효율성이라는 두 마리 토끼를 잡으려 한 시도로, 특히 **에지 디바이스(Edge Device)나 실시간 스트리밍 서비스**와 같이 메모리와 지연 시간(latency)이 중요한 환경에서 매우 강력한 대안이 될 수 있다. 또한, 오픈소스 커뮤니티를 통해 C, C++, Rust, Go 등 다양한 언어로 구현체가 확산되고 있다는 점이 고무적이다.

### 한계 및 비판적 해석

본 논문에서 언급된 바와 같이, RWKV는 구조적으로 '상태'를 압축하여 전달하는 RNN의 특성을 갖는다. 이는 필연적으로 **정보의 손실(decay)**을 야기하며, 이는 Transformer가 모든 토큰을 직접 참조하는 것과 대조되는 지점이다. 따라서 매우 정밀한 정보 추출이 필요한 Long-context 작업에서는 구조적인 한계가 존재할 수밖에 없다.

### 향후 연구 방향

1. **정보 감쇠 해결**: 장거리 의존성을 더 명시적으로 추적하거나 메모리 메커니즘을 개선하여 정보 손실을 줄여야 한다.
2. **멀티모달 확장**: 텍스트 외에 이미지, 오디오 등 다양한 모달리티 간의 효율적인 결합 모델 연구가 필요하다.
3. **하드웨어 최적화**: GPU/TPU뿐만 아니라 NPU, ASIC 등 전용 가속기에서의 최적화된 커널 구현이 필수적이다.

## 📌 TL;DR

본 논문은 Transformer의 $O(T^2)$ 복잡도 문제를 해결하기 위해 RNN의 선형 복잡도 $O(T)$를 결합한 **RWKV 모델의 종합 서베이 보고서**이다. RWKV는 Linear Attention과 재귀적 구조를 통해 효율적인 추론과 병렬 학습을 동시에 달성하였으며, v4에서 v6에 이르기까지 행렬 상태 도입 및 동적 감쇠 메커니즘을 통해 성능을 고도화하였다. 텍스트 생성부터 의료 영상 분석까지 광범위하게 적용되고 있으나, 초장거리 시퀀스에서의 정보 손실 문제는 향후 해결해야 할 핵심 과제이다. 이 연구는 향후 저전력·고효율 LLM 설계 및 에지 AI 구현에 중요한 지침을 제공한다.
