# HiMix: Reducing Computational Complexity in Large Vision-Language Models

Xuange Zhang, Dengjie Li, Bo Liu, Zenghao Bao, Yao Zhou, Baisong Yang, Zhongying Liu, Yujie Zhong, Zheng Zhao, Tongtong Yuan (2025)

## 🧩 Problem to Solve

최근 대규모 언어 모델(LLM)과 모달리티 정렬 기술의 발전으로 Large Vision-Language Models(LVLMs)가 다양한 시나리오에서 뛰어난 성능을 보이고 있다. 그러나 이러한 모델들의 과도한 계산 복잡도는 실제 응용 분야로의 광범위한 적용을 제한하는 주요 병목 현상이 된다.

특히, 대부분의 LVLM은 시각적 특징과 언어 특징을 단순히 연결(concatenation)하여 LLM의 입력으로 사용하는 방식을 취한다. 시각 시퀀스는 일반적으로 언어 시퀀스보다 훨씬 길기 때문에, 모든 디코더 층에서 시각 토큰이 함께 계산되는 과정에서 계산량이 기하급수적으로 증가한다. 본 논문은 언어 디코더 내에서 시각 정보가 모든 전방 전파 과정에 참여할 필요가 없으며, 많은 부분에서 중복된 계산이 발생한다는 문제 제기에서 시작한다. 따라서 본 연구의 목표는 모델의 성능 저하를 최소화하면서 계산 복잡도를 획기적으로 줄이는 효율적인 시각-언어 상호작용 메커니즘을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Hierarchical Vision injection for Mixture Attention (HiMix)**이다.

가장 중심적인 직관은 "언어 모델은 기본적으로 텍스트 이해와 처리에 집중해야 하며, 시각 정보는 보조적인 지원이 필요할 때만 참조하면 된다"는 것이다. 이를 위해 시각 시퀀스를 모든 층의 전방 전파에 참여시키는 대신, 각 디코더 층의 특정 단계에서만 선택적으로 주입하는 계층적 구조를 제안한다.

핵심 설계 요소는 다음과 같다:

1. **Hierarchical Vision Injection**: 각 디코더 층마다 전용 시각 투영 층(Vision Projection Layers)을 두어, 층별로 최적화된 다양한 시각적 단서를 주입한다.
2. **Mixture Attention (MA)**: 언어 시퀀스만 전체 전방 전파를 수행하고, 시각 시퀀스는 Attention 단계에서 $KV$ 값으로만 참여하게 하여 계산 복잡도를 낮춘다.

## 📎 Related Works

### 기존 LVLM 구조

기존의 Flamingo, BLIP-2, LLaVA, MiniGPT-4 등 대부분의 모델은 시각 인코더를 통해 추출된 특징을 커넥터(Connector)를 통해 정렬한 뒤, 언어 토큰과 연결하여 LLM에 입력한다. 이 방식은 단순하고 효과적이지만, 시각 토큰의 수가 많아질수록 계산 비용이 급격히 증가하는 한계가 있다.

### Vision Token Reduction 기술

계산 비용을 줄이기 위해 FastV, VTW, LLaVA-PruMerge 등 시각 토큰을 가지치기(pruning)하거나 특정 층 이후에 제거하는 연구들이 진행되었다. 그러나 이러한 방식은 단순히 토큰의 수를 줄이거나 완전히 배제하는 것에 집중한다. 반면, HiMix는 토큰을 단순히 제거하는 것이 아니라, 언어 시퀀스의 흐름은 유지하면서 시각 정보를 계층적으로 주입함으로써 성능과 효율성 사이의 균형을 맞춘다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### Vanilla-LVLM (기존 방식)

기존 방식에서는 시각 특징 $X_v'$와 언어 특징 $X_l$을 연결하여 $X_{vl} = [X_v'; X_l]$ 형태로 입력한다. 각 Transformer 층의 Self-Attention 계산 복잡도는 $O((N+M)^2 d)$이며, Feed-Forward Network(FFN)의 복잡도는 $O(8(N+M)d^2)$이다 (여기서 $N$은 시각 토큰 수, $M$은 언어 토큰 수).

### HiMix 구조

#### 1. Hierarchical Vision Injection

HiMix는 모든 층에 동일한 시각 특징을 넣는 Uniform Injection 대신, 각 층마다 독립적인 투영 층을 사용하는 계층적 주입 방식을 사용한다. 이는 모델이 층의 깊이에 따라 더 다양한 시각적 단서를 포착할 수 있게 하며, 시각-언어 간의 간섭을 줄이기 위해 전용 시각 투영 층을 통해 정보를 주입한다.

#### 2. Mixture Attention (MA)

Mixture Attention은 시각 시퀀스와 언어 시퀀스를 분리하여 처리한다.

- **입력 처리**: 언어 특징은 $W_Q^l, W_K^l, W_V^l$를 통해 $Q_l, K_l, V_l$을 생성하고, 시각 특징은 $W_K^v, W_V^v$를 통해 $K_v, V_v$만을 생성한다.
- **KV 연결**: 시각과 언어의 $K, V$ 값을 연결하여 전체 $KV$ 시퀀스를 구성한다.
  $$K_{vl} = [K_v; K_l], \quad V_{vl} = [V_v; V_l]$$
- **Attention 계산**: 언어 쿼리 $Q_l$만을 사용하여 Attention을 수행한다.
  $$\text{MA} = \text{Softmax}\left(\frac{Q_l K_{vl}^T}{\sqrt{d}}\right) V_{vl}$$
- **마스킹**: 모든 언어 토큰이 이전의 모든 시각 토큰을 참조할 수 있게 하되, 언어 토큰 간에는 인과적 마스크(Causal Mask)를 적용하여 미래 정보 유출을 방지한다.

#### 3. 계산 복잡도 분석

HiMix의 전체 복잡도는 다음과 같다:
$$O((N+M)M d) + O(8M d^2)$$
시각 시퀀스는 FFN 계산에서 완전히 제외되며, Attention에서도 쿼리($Q$) 역할을 하지 않으므로 $N^2$ 항이 사라진다. $N \gg M$인 일반적인 상황에서 이는 계산량을 획기적으로 줄이는 결과로 이어진다.

## 📊 Results

### 실험 설정

- **모델**: Qwen2-0.5B, TinyLlama-1.1B, Llama3.2-1B, Llama3.2-3B 및 Vicuna-7B.
- **시각 인코더**: SigLIP (SoViT-400m/14).
- **지표**: VQAv2, GQA, TextVQA, MM-Vet, POPE, MME, MMMU 등 7개 벤치마크.
- **학습 전략**: LLaVA-1.5 데이터만 사용하는 Regular Paradigm과 ShareGPT4V 데이터를 추가한 Enhanced Paradigm으로 구분하여 실험하였다.

### 주요 결과

1. **계산 효율성**: 다양한 베이스 모델에서 언어 디코더의 계산 비용(GFLOPs)을 **약 10배(90% 감소)** 줄였다. VRAM 사용량 또한 유의미하게 감소하였다.
2. **성능 유지**: Regular Paradigm 하에서 대부분의 벤치마크에서 Baseline과 대등한 성능을 보였다. 특히 Enhanced Paradigm을 적용했을 때, 일부 지표에서는 Baseline을 상회하는 성능 향상을 보였다.
3. **FastV와의 비교**: 기존의 토큰 pruning 방식인 FastV와 비교했을 때, FastV는 계산량을 줄일수록 성능이 급격히 하락하는 반면, HiMix는 계산량을 9% 수준으로 낮추면서도 Baseline에 근접한 성능을 유지하였다.
4. **한계점**: TextVQA 및 MME 작업에서 성능 저하가 관찰되었는데, 이는 OCR(광학 문자 인식) 능력이 일부 감소했기 때문이다. 하지만 이는 Enhanced Paradigm의 데이터 증강을 통해 완화될 수 있음이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 가치

HiMix는 LVLM의 계산 병목이 시각 토큰의 과도한 참여에서 온다는 가설을 실험적으로 증명하였다. 특히 시각 시퀀스를 $KV$ 값으로만 활용하고 언어 시퀀스 중심의 전방 전파를 수행함으로써, 모델의 표현력을 유지하면서도 추론 속도와 메모리 효율을 극적으로 개선하였다.

### 한계 및 비판적 해석

본 논문은 주로 소규모 LLM(0.5B ~ 3B)을 중심으로 실험을 진행하였다. 비록 Vicuna-7B에 대한 보조 실험을 통해 확장성을 보여주었으나, 훨씬 더 거대한 모델(예: 70B 이상)에서도 동일한 효율성-성능 트레이드오프가 유지될지는 추가 검증이 필요하다. 또한, OCR 능력의 감소는 시각 정보의 '선택적 주입'이 세밀한 텍스트 인식과 같은 저수준(low-level) 특징 추출에는 불리할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 LVLM의 언어 디코더에서 시각 토큰이 모든 층에서 계산될 필요가 없다는 점에 착안하여, 시각 정보를 계층적으로 주입하고 언어 쿼리 중심의 **Mixture Attention**을 사용하는 **HiMix**를 제안한다. 이를 통해 성능 저하를 거의 없이 언어 디코더의 계산 비용을 **10배 감소**시켰으며, 이는 특히 자원이 제한된 엣지 디바이스나 실시간 서비스 환경에서 LVLM을 배포하는 데 매우 중요한 기여를 할 것으로 기대된다.
