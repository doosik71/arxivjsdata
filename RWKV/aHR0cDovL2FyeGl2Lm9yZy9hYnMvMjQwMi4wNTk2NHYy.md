# A Survey on Transformer Compression

Yehui Tang, Yunhe Wang, Jianyuan Guo, Zhijun Tu, Kai Han, Hailin Hu, and Dacheng Tao (2024)

## 🧩 Problem to Solve

최근 자연어 처리(NLP)와 컴퓨터 비전(CV) 분야에서는 Transformer 아키텍처를 기반으로 한 거대 언어 모델(LLM)과 거대 비전 모델(LVM)이 주류를 이루고 있다. 이러한 모델들은 뛰어난 확장성과 성능을 보여주지만, 수십억 개에서 수천억 개에 이르는 방대한 파라미터 수로 인해 실질적인 배포 단계에서 심각한 문제에 직면한다.

가장 핵심적인 문제는 막대한 메모리 저장 공간과 계산 비용이다. 예를 들어 GPT-3 모델은 1,750억 개의 파라미터를 가지며, float16 정밀도 기준으로 약 350GB의 메모리가 필요하다. 이는 일반적인 하드웨어, 특히 모바일 기기와 같은 에지 디바이스(edge device)에서의 구현을 불가능하게 만들며, 데이터 센터 수준에서도 막대한 전력 소모와 탄소 배출을 야기한다.

따라서 본 논문의 목표는 Transformer 모델의 중복성을 줄여 메모리와 계산 비용을 낮추는 다양한 모델 압축(Model Compression) 기법들을 체계적으로 분류하고, 최신 연구 동향을 분석하여 실용적인 배포 가능성을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Transformer 아키텍처의 특수성(Attention과 FFN 모듈의 교차 구조)을 고려하여 압축 방법론을 네 가지 핵심 카테고리로 체계화한 것이다.

1. **양자화(Quantization):** 가중치와 활성화 함수 값을 낮은 비트로 표현하여 메모리 사용량을 줄이는 기법.
2. **지식 증류(Knowledge Distillation):** 거대 모델(Teacher)의 지식을 작은 모델(Student)에게 전달하는 학습 전략.
3. **가지치기(Pruning):** 불필요한 파라미터나 구조(헤드, 레이어 등)를 제거하여 모델 크기를 줄이는 기법.
4. **효율적 아키텍처 설계(Efficient Architecture Design):** 계산 복잡도를 근본적으로 낮춘 새로운 구조(Mamba, RetNet 등)를 탐색하는 방법.

특히, NLP와 CV라는 서로 다른 도메인에서 사용되는 압축 기법들이 실제로는 유사한 원리를 공유하고 있음을 분석하고, 모델을 전체적으로 재학습시키는 것이 불가능한 LLM의 특성을 고려한 효율적인 압축 경로를 제시한다.

## 📎 Related Works

Transformer 이전에는 MLP, CNN, RNN, LSTM 등이 딥러닝의 주축을 이루었으나, Transformer는 강력한 확장성(Scalability)을 바탕으로 현재의 파운데이션 모델(Foundation Models)의 기반이 되었다.

기존의 일반적인 모델 압축 연구들은 주로 CNN에 집중되어 있었으나, Transformer는 다음과 같은 차별점을 가진다. 첫째, 전역 정보를 캡처하는 Attention 모듈과 토큰별 정보를 추출하는 FFN 모듈이 교차로 배치된 독특한 구조를 가진다. 둘째, 모델의 규모가 극단적으로 커짐에 따라 전체 데이터셋으로 모델을 재학습(Retraining)하는 비용이 천문학적으로 증가하여, 학습 없이 또는 매우 적은 데이터만으로 압축하는 Post-training 기법의 중요성이 비약적으로 높아졌다.

## 🛠️ Methodology

본 논문은 Transformer 압축의 핵심 방법론을 다음과 같이 상세히 설명한다.

### 1. Transformer의 기본 구조

압축 대상을 이해하기 위해 먼저 표준 Transformer의 핵심 방정식을 정의한다.
Multi-Head Attention(MHA)은 다음과 같이 계산된다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
이후 여러 개의 헤드를 결합하여 $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$ 형태로 출력한다. FFN 모듈은 각 토큰을 독립적으로 변환하며 다음과 같은 구조를 가진다.
$$\text{FFN}(x) = \phi(xW_1 + b_1)W_2 + b_2$$
여기서 $\phi$는 GELU와 같은 활성화 함수이다.

### 2. 양자화 (Quantization)

양자화는 부동 소수점 텐서 $x$를 정수 $x_{int}$로 변환하는 과정이다.
$$x_{int} = \text{Clamp}(\lfloor x/s \rceil + z, 0, 2^b - 1)$$
$$x_{quant} = s(x_{int} - z)$$
여기서 $s$는 스케일 팩터, $z$는 제로 포인트, $b$는 비트 너비이다.

- **PTQ (Post-Training Quantization):** 적은 양의 보정 데이터(Calibration data)만을 사용하여 가중치와 활성화 함수의 양자화 파라미터를 최적화한다. LLM에서는 Outlier(이상치) 제거를 위해 SmoothQuant와 같은 기법이 사용된다.
- **QAT (Quantization-Aware Training):** 학습 과정에 양자화 노드를 삽입하여 가중치와 양자화 파라미터를 함께 최적화한다. 성능 저하가 적지만 계산 비용이 높다.

### 3. 지식 증류 (Knowledge Distillation)

Teacher 모델의 출력을 Student 모델이 모방하도록 학습시킨다.

- **Logits-based KD:** Teacher($p_t$)와 Student($p_s$)의 확률 분포 간 KL-Divergence를 최소화한다.
$$L_{logits} = \text{KL}(p_t || p_s) = \sum_{j=1}^C p_{t,j} \log\left(\frac{p_{t,j}}{p_{s,j}}\right)$$
- **Hint-based KD:** 중간 특징 맵(Intermediate features) $F_t$와 $F_s$ 사이의 거리를 최소화한다.
$$L_{hint} = \|F_t - \phi(F_s)\|^2$$
- **API-based KD:** 모델 내부 파라미터에 접근할 수 없는 경우(예: GPT-4), 생성된 텍스트(Chain-of-Thought 등)만을 이용하여 Student를 학습시킨다.

### 4. 가지치기 (Pruning)

- **Unstructured Pruning:** 개별 가중치 단위로 제거하며, 이론적 압축률은 높으나 하드웨어 가속이 어렵다.
- **Structured Pruning:** 어텐션 헤드, FFN 레이어, 전체 블록 등 구조 단위로 제거하여 실제 추론 속도를 향상시킨다.
- **Context Pruning:** 입력 토큰 중 불필요한 부분을 제거하여 $O(N^2)$의 계산 복잡도를 줄인다.

### 5. 효율적 아키텍처 설계 (Efficient Architecture)

- **Attention 최적화:** Linear Attention, Sparse Attention, FlashAttention 등을 통해 계산 복잡도를 $O(N^2)$에서 $O(N)$ 또는 $O(N \log N)$으로 낮춘다.
- **Non-Transformer 구조:** RNN의 선형 복잡성과 Transformer의 병렬 학습 능력을 결합한 RetNet, RWKV, 그리고 Selective State Space Model(SSM)을 사용하는 Mamba 등이 제시된다.

## 📊 Results

논문은 다양한 벤치마크를 통해 압축 기법의 성능을 정량적으로 분석한다.

1. **양자화 성능:** Table 2와 3을 통해, 8-bit 양자화(INT8)는 FP16 대비 성능 저하가 거의 없음을 보여준다. 그러나 4-bit 이하의 극저비트 양자화에서는 PTQ의 성능 저하가 심하며, 이를 보완하기 위해 QAT나 정밀한 Outlier 제어 기법이 필수적임을 확인하였다.
2. **지식 증류 효과:** Table 4에서 BERT 압축 사례를 분석한 결과, Logits 기반 KD보다 Hint-based KD(중간 특징 전달)가 Student 모델의 성능을 더 효과적으로 끌어올림을 보여준다.
3. **가지치기 효율성:** Table 5(ViT)와 Table 6(LLM)에서 구조적 가지치기가 실제 추론 속도(Speed-up)를 유의미하게 향상시킴을 보여준다. 특히 LLM의 경우 SparseGPT와 같은 기법이 50-60%의 희소성을 확보하면서도 Perplexity(PPL) 증가를 억제함을 입증하였다.
4. **추론 지연 시간:** Figure 3에서 INT8 양자화가 FP16 대비 ViT 및 OPT 모델의 추론 지연 시간(Latency)을 상당히 단축시킴을 정량적으로 제시한다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 보고서는 Transformer 압축이 단순히 파라미터를 줄이는 것이 아니라, **"하드웨어 효율성"**과 **"모델 표현력"** 사이의 트레이드-오프를 최적화하는 과정임을 강조한다. 특히 LLM 시대에 들어서면서 '재학습 없는 압축'이 가장 중요한 연구 방향이 되었음을 명확히 짚어냈다.

### 한계 및 비판적 해석

1. **데이터 의존성:** 많은 PTQ 및 Pruning 기법들이 소량의 '보정 데이터'를 필요로 하는데, 이 데이터의 도메인이 실제 추론 데이터와 다를 경우 성능이 급격히 하락하는 문제가 여전히 존재한다.
2. **극저비트의 한계:** 4-bit 이하의 양자화에서는 여전히 심각한 성능 저하가 발생하며, 이를 해결하기 위한 이론적 근거보다는 휴리스틱한 최적화 방법(예: Weight Clipping, Scaling)에 의존하는 경향이 크다.
3. **아키텍처 전환의 리스크:** Mamba와 같은 SSM 기반 모델들이 효율적이지만, 기존 Transformer의 생태계(소프트웨어 라이브러리, 최적화 커널)를 대체하기까지는 상당한 시간이 걸릴 것으로 보인다.

## 📌 TL;DR

본 논문은 Transformer 기반 거대 모델의 배포 가능성을 높이기 위해 **양자화, 지식 증류, 가지치기, 효율적 아키텍처 설계**라는 네 가지 관점에서 최신 압축 기술을 총망라한 서베이 논문이다.

**핵심 요약:**

- **양자화:** INT8은 안정적이나 4-bit 이하는 QAT나 Outlier 제어가 필수적이다.
- **지식 증류:** 단순 Logits 모방보다 중간 특징(Hint) 및 API 기반의 추론 과정(CoT) 전수가 더 효과적이다.
- **가지치기:** 단순 가중치 제거보다 구조적 제거(Structured Pruning)가 실제 하드웨어 가속에 유리하다.
- **아키텍처:** $O(N^2)$의 복잡도를 해결하기 위해 Linear Attention 및 SSM(Mamba 등)으로의 패러다임 전환이 일어나고 있다.

이 연구는 향후 초거대 모델을 온디바이스(On-device) 환경에 구현하기 위한 기술적 로드맵을 제공하며, 특히 서로 다른 압축 기법을 결합한 **Joint Search** 전략이 차세대 핵심 연구 분야가 될 것임을 시사한다.
