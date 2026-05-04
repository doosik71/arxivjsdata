# PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization

Mengzhao Chen, Yi Liu, Jiahao Wang, Yi Bin, Wenqi Shao, Ping Luo (2024)

## 🧩 Problem to Solve

대규모 언어 모델(Large Language Models, LLMs)의 효율적인 배포를 위해 Weight와 Activation을 낮은 비트로 양자화하는 기술이 필수적이다. 기존의 양자화 방법들은 주로 특정 채널에서 발생하는 Channel-wise outliers 문제를 해결하는 데 집중해 왔다. 그러나 본 논문은 일부 특정 토큰에서만 매우 거대한 활성화 값이 나타나는 Token-wise outliers가 양자화 오차의 주된 원인이 된다는 점을 지적한다.

분석 결과, 2048개의 입력 컨텍스트 중 단 2개의 outlier token이 전체 양자화 오차의 94.7%를 차지하는 것으로 나타났다. 이러한 Token-wise outliers는 기존의 Hadamard rotation과 같은 채널 기반 분산 기법으로도 완전히 해결되지 않으며, 결과적으로 양자화 모델의 정확도를 심각하게 저하시킨다. 따라서 본 연구의 목표는 이러한 Token-wise outliers를 효과적으로 격리하여, 낮은 비트 정밀도(W4A4, W4A8)에서도 높은 정확도와 추론 속도를 유지하는 양자화 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **고빈도 outlier token들을 입력 시퀀스의 앞부분에 Prefix로 추가하여, outlier가 발생하는 위치를 특정 영역으로 한정시키는 것**이다.

구체적으로, 모델마다 반복적으로 outlier를 유발하는 특정 토큰들을 식별하고, 이를 KV cache의 앞부분에 미리 채워 넣는다(Prefilling). 이렇게 하면 실제 추론 과정에서 입력되는 일반 토큰들은 outlier의 영향에서 벗어나 보다 균일한 분포를 갖게 되며, 양자화에 최적화된 분포를 형성하게 된다. 또한, 격리된 outlier token들은 full-precision으로 KV cache에 저장되어 정보 손실을 방지하며, 이후 Block-wise fine-tuning을 통해 양자화 오차를 추가로 보상한다.

## 📎 Related Works

### 기존 연구 및 한계
1.  **Channel-Wise Outlier 대응**: SmoothQuant, OmniQuant, QuaRot 등은 특정 채널의 큰 값을 스케일링하거나 Hadamard rotation을 통해 전체 채널로 분산시켜 해결하려 했다. 하지만 이는 토큰 단위로 발생하는 거대 활성화 값(Token-wise outliers)을 완전히 제거하지 못한다.
2.  **Token-Wise Outlier 대응**: StreamingLLM이나 Cushion-Cache와 같은 연구들이 초기 토큰의 중요성(Attention sinks)을 다루었으나, Cushion-Cache의 경우 outlier 토큰을 찾는 과정에서 막대한 그리드 서치 비용(Llama-3-8B 기준 약 12시간)이 발생한다는 한계가 있다.

### PrefixQuant의 차별점
PrefixQuant는 훈련이 필요 없는(Training-free) 매우 효율적인 토큰 식별 과정을 통해 outlier token을 찾아내며, Llama-3-8B 기준으로 단 12초 만에 이 과정을 완료한다. 또한, 단순히 outlier를 제거하는 것이 아니라 KV cache에 prefix 형태로 저장함으로써 추론 효율성과 정확도를 동시에 잡았다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. Outlier Token 탐색 및 정의
먼저, 토큰별 최대값 $M_i$와 그 중앙값(median)의 비율을 통해 outlier 정도를 측정한다.
$$R_i = \frac{M_i}{\text{median}(M)}$$
여기서 $R_i > \eta_1$이면 Upper outlier token, $R_i^{-1} > \eta_2$이면 Lower outlier token으로 정의한다 ($\eta_1=64, \eta_2=8$). 분석 결과, Llama-2-7B의 경우 `\n`이나 `.`과 같은 저의미(low-semantic) 토큰들이 시퀀스 앞부분에서 빈번하게 outlier를 유발함을 확인하였다.

### 2. Prefixed Outliers 기법
식별된 고빈도 outlier token들을 입력 시퀀스 앞에 추가한다. 이 토큰들은 full-precision 모델을 통해 한 번만 prefilling 되어 KV cache에 저장되며, 이후 모든 추론 단계에서 재사용된다.

Self-attention 메커니즘에 prefixed tokens $(k', v')$가 적용된 수식은 다음과 같다.
$$\text{Attention}(Q, K, V; k', v') = \text{Softmax} \left( \frac{Q[K^T, k'^T]}{\sqrt{d}} \right) [V, v']^T$$
여기서 $k', v' \in \mathbb{R}^{o \times C}$는 KV cache에 저장된 prefixed tokens이다. 이를 통해 일반 토큰들의 활성화 값 분포가 매우 균일해지며, 양자화 오차가 획기적으로 줄어든다.

### 3. Block-wise Fine-tuning
양자화로 인한 잔여 오차를 줄이기 위해 각 Transformer block을 순차적으로 미세 조정한다. 손실 함수로는 Mean Squared Error (MSE)를 사용한다.
- **Dynamic Quantization (PrefixQuant-O1)**: 텐서 단위의 Clipping factor를 학습 가능한 파라미터로 설정하여 Rounding error와 Clipping error 사이의 균형을 맞춘다.
- **Static Quantization (PrefixQuant-O2)**: 스케일링 팩터(Scaling factor)와 제로 포인트(Zero-point)를 직접 학습시킨다.
- **Weight Quantization**: EfficientQAT 방식을 따라 가중치와 양자화 파라미터를 함께 학습시킨다.

## 📊 Results

### 실험 설정
- **정밀도**: W4A4KV4 (Weight 4-bit, Activation 4-bit, KV cache 4-bit) 및 W4A8KV4.
- **모델**: Llama-2, Llama-3, Mistral-7B-v0.3, Qwen-2-7B.
- **지표**: WikiText2 Perplexity (PPL), 5가지 zero-shot reasoning task의 평균 정확도(Acc.), MMLU.

### 주요 결과
1.  **정확도 향상**: Llama-3-8B (W4A4KV4) 모델에서 기존 SOTA인 SpinQuant 대비 zero-shot reasoning task에서 평균 $+3.08$ (dynamic) 및 $+2.85$ (static) 포인트의 성능 향상을 보였다.
2.  **정적 양자화의 효율성**: 특히 PrefixQuant-O2(정적 양자화)가 기존의 동적 양자화 방법들보다 더 나은 성능을 보이거나 대등한 성능을 내면서도 추론 속도는 더 빠르다는 점을 입증하였다.
3.  **추론 속도**: W4A4 정밀도 적용 시, FP16 모델 대비 Prefilling 속도는 약 $2.74\times$, Decoding 속도는 약 $2.16\times$ 향상되었다.
4.  **MMLU 결과**: Llama-3-8B (W4A4KV4) 기준, PrefixQuant-O1은 56.00%의 정확도를 기록하여 SpinQuant(51.93%)를 크게 상회하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 LLM 양자화의 고질적인 문제인 outlier를 '채널' 관점이 아닌 '토큰' 관점에서 접근하여 해결책을 제시했다. 특히, outlier를 완전히 제거하는 것이 아니라 KV cache라는 구조적 특징을 이용해 격리(Isolate)하고 이를 full-precision으로 유지함으로써, 계산 효율성과 모델 표현력을 동시에 확보했다는 점이 매우 영리한 설계이다.

### 한계 및 논의
- **KV Cache 메모리**: prefixed tokens를 추가함에 따라 아주 소량의 KV cache 메모리가 추가로 소모된다. 하지만 $o$ (outlier token 수)가 매우 작기 때문에(보통 1~4개) 실질적인 오버헤드는 무시할 만한 수준이다.
- **토큰 의존성**: 모델마다 outlier를 유발하는 토큰이 다르므로, 모델별로 최적의 prefix 토큰 세트를 찾는 과정이 선행되어야 한다.
- **일반화**: 8192 길이의 long-context 시나리오에서도 성능이 유지됨을 확인하여, 제안 방법이 특정 시퀀스 길이에 국한되지 않고 일반화 능력이 있음을 보였다.

## 📌 TL;DR

PrefixQuant는 LLM 양자화 성능을 저하시키는 **Token-wise outliers를 KV cache의 prefix 영역으로 격리**하여 해결하는 방법론이다. 고빈도 outlier 토큰을 미리 식별해 KV cache에 저장함으로써, 일반 토큰들의 분포를 균일하게 만들어 양자화 오차를 극적으로 줄인다. 결과적으로 W4A4와 같은 초저정밀도에서도 SOTA 수준의 정확도를 달성했으며, 추론 속도를 2배 이상 향상시켜 실제 서비스 배포 가능성을 높인 연구이다.