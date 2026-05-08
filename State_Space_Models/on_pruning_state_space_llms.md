# On Pruning State-Space LLMs

Tamer Ghattas, Michael Hassid, Roy Schwartz (2025)

## 🧩 Problem to Solve

최근 State-Space Models (SSMs)는 Transformer 기반의 대규모 언어 모델(LLM)을 대체할 수 있는 효율적인 대안으로 주목받고 있다. SSM은 선택적 기억 능력과 RNN의 특성을 결합하여 Transformer의 Attention 메커니즘보다 계산 복잡도를 낮추면서도 경쟁력 있는 성능을 보여준다. 그러나 SSM 기반의 LLM 역시 여전히 방대한 파라미터를 보유하고 있어, 연산 비용을 더욱 줄이기 위한 모델 압축의 필요성이 제기된다.

본 논문은 기존에 주로 Transformer 구조를 대상으로 개발된 Pruning(가지치기) 방법론들을 SSM 구조에 적용할 수 있는지, 그리고 적용했을 때 모델의 성능이 얼마나 유지되는지를 탐구하는 것을 목표로 한다. 특히 구조적 특성이 다른 SSM 기반 LLM들에 다양한 Pruning 기법을 적용하여, SSM 모델의 압축 가능성과 각 구성 요소의 민감도를 분석하고자 한다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 Transformer를 위해 설계된 다양한 Pruning 방법론을 SSM 구조에 맞게 변형하여 적용하고, 그 효과를 체계적으로 분석했다는 점이다. 연구진은 비구조적(Unstructured) 방법인 WANDA와 더불어, SSM의 특성을 반영한 네 가지 구조적(Structured) Pruning 방식(State pruning, Head dimension pruning, Head merging, SSM-FLAP)을 제안하고 실험하였다. 이를 통해 SSM 기반 LLM이 특정 Pruning 방식에는 매우 강건하지만, 다른 방식에는 매우 취약하다는 설계적 통찰을 제공한다.

## 📎 Related Works

### SSM Architectures

최근의 SSM 연구는 sub-quadratic 복잡도를 달성하면서도 Transformer와 대등한 성능을 내는 방향으로 발전했다. Mamba는 입력의 특정 부분에 적응적으로 집중하는 Selective SSM 변형을 사용하며, Mamba-2는 Structured State Space Duality(SSD) 프레임워크를 통해 재귀적 처리와 Attention 유사 연산을 통합하여 하드웨어 최적화를 달성하고 속도를 2~8배 향상시켰다.

### Pruning Methods

LLM 압축을 위해 널리 사용되는 Pruning 방법은 크게 두 가지로 나뉜다.

- **Unstructured Pruning**: 개별 가중치의 중요도(예: 절대값 크기)에 따라 희소성(Sparsity)을 부여한다. 특히 WANDA는 가중치뿐만 아니라 Activation 값을 함께 고려하여 Fine-tuning 없이도 효과적인 압축을 수행한다.
- **Structured Pruning**: 가중치 행렬의 전체 열이나 행, 또는 특정 헤드(Head)와 같은 구조적 단위를 제거한다. FLAP은 입력 채널의 변동성(Fluctuation)을 기반으로 중요도를 판단하여 구조적 Pruning을 수행하며, GQA(Group-Query Attention) 역시 KV 헤드를 평균 풀링(Mean-pooling)으로 병합한다는 점에서 구조적 Pruning의 일종으로 볼 수 있다.

## 🛠️ Methodology

### 시스템 구조 및 대상 모델

본 연구는 Mamba-2 아키텍처를 중심으로 분석을 진행한다. Mamba-2의 SSM 블록은 입력값이 $\text{InProj}$를 통해 다섯 개의 행렬($W_X, W_Z, W_B, W_C, W_{\Delta}$)로 투영되고, 이후 1D Convolution 레이어와 SSD 알고리즘을 거쳐 $\text{OutProj}$를 통해 최종 출력되는 구조를 가진다. 실험 대상 모델은 MAMBA-2-2.7B, PHI-MAMBA-1.5B, Hybrid-Llama3-Mamba2-3B(HLM-3B), SMOL-MAMBA-1.9B의 네 가지 모델이다.

### Pruning 방법론의 SSM 적용

1. **Unstructured Pruning (WANDA)**:
   WANDA는 선형 레이어의 크기에 관계없이 적용 가능하므로, SSM 레이어에 최소한의 수정만으로 직접 적용하였다.

2. **State Pruning (Structured)**:
   $W_B$와 $W_C$ 행렬에서 각 헤드에 해당하는 $D \times N$ 텐서 중 중요도가 낮은 부분을 제거한다. 중요도는 2차 테일러 근사(Second-order Taylor approximation) 기반의 추정치를 사용하며, $\text{OutProj}$와 $\text{conv1d}$ 레이어의 가중치도 이에 맞춰 함께 제거하여 차원을 유지한다.

3. **Head Dimension Pruning (Structured)**:
   $W_X$와 $W_Z$ 텐서의 헤드 차원을 Pruning 하며, 이와 연동된 $\text{conv1d}$ 필터와 $\text{OutProj}$의 행을 함께 제거한다.

4. **Merging Heads (Structured)**:
   Transformer의 GQA 방식에서 영감을 얻어, 인접한 헤드들을 평균 풀링(Mean-pooling)하여 병합한다. Mamba-2와 같이 단일 BC 헤드를 가진 경우 X 헤드를 병합한다.

5. **SSM-FLAP (Structured)**:
   FLAP의 프레임워크를 SSM에 이식하여 $\text{InProj}$ 하위 행렬들에 Pruning 마스크를 적용한다. Bias 보상을 위해 bias 항을 추가하며, 모델의 헤드 패턴(GVA 등)에 따라 $\text{X}$ 헤드의 수를 $\text{BC}$ 헤드 수의 배수로 유지하며 Pruning 한다.

### 학습 및 추론 절차

- **Fine-tuning**: Pruning 후 성능 회복을 위해 $\text{wikitext2}$ 데이터셋으로 Fine-tuning을 수행하였다.
- **LoRA**: SSM 및 MLP 레이어를 대상으로 LoRA(Low-Rank Adaptation)를 적용하였다.
- **Loss Function**: 일반적인 Cross-Entropy(CE) 손실 함수보다 지식 증류(Knowledge Distillation, KD) 손실 함수를 사용했을 때 성능 회복 효과가 더 뛰어남을 확인하여 KD loss를 사용하였다.

## 📊 Results

### 실험 설정

- **지표 및 데이터셋**: Lambada, HellaSwag, PIQA, ARC-easy, ARC-challenge, Winogrande 등 6개 벤치마크의 평균 정확도를 측정하였다.
- **Pruning 비율**: WANDA, State, Head, SSM-FLAP은 25%와 50% 비율로 적용하였으며, Head Merging은 50%와 75%의 헤드를 병합하였다.

### 주요 결과

- **WANDA의 강건함**: WANDA는 대부분의 모델에서 25% Pruning 시 성능을 잘 유지했으며, 50%에서도 완전히 붕괴(Collapse)되지 않는 강건함을 보였다. 단, FFN 레이어가 없는 Mamba-2-2.7B 모델은 상대적으로 빠르게 성능이 하락하였다.
- **State Pruning의 효율성**: 구조적 방법 중 State pruning이 가장 효과적이었다. 4개 모델 중 3개 모델에서 50% Pruning 시에도 성능 하락이 매우 적거나 무시할 수 있는 수준이었다.
- **Head Pruning의 취약성**: 반면, Head pruning(Head Dimension Pruning 및 Head Merging)을 적용한 모든 모델은 매우 급격한 성능 저하를 보였다. 특히 일부 모델에서는 25% Pruning만으로도 성능이 심각하게 하락하였다.
- **구성 요소별 민감도**: Mamba-2-2.7B 모델을 대상으로 $\text{InProj}$와 $\text{OutProj}$를 각각 Pruning 하여 분석한 결과, $\text{OutProj}$를 Pruning 했을 때 Perplexity가 급격히 상승하였다. 이는 $\text{InProj}$보다 $\text{OutProj}$가 Pruning에 훨씬 더 민감함을 의미한다.

## 🧠 Insights & Discussion

본 연구는 SSM 기반 LLM이 비구조적 Pruning(WANDA)과 특정 구조적 Pruning(State pruning)에 상당히 강건하다는 점을 밝혀냈다. 이는 SSM 모델의 파라미터 효율성을 더욱 높일 수 있는 가능성을 시사한다.

특히 주목할 점은 SSM 내에서도 구성 요소마다 Pruning에 대한 민감도가 크게 다르다는 것이다. $\text{OutProj}$ 레이어의 높은 민감도는 모델의 최종 출력 단계에서 정보 손실이 발생할 때 치명적임을 보여준다. 또한, 모델의 아키텍처(예: FFN 레이어의 존재 여부, 헤드 패턴)에 따라 Pruning에 대한 반응이 상이하게 나타났다.

**한계점 및 비판적 해석**:

- 실험에 사용된 네 가지 모델이 크기, 학습 데이터, 구조(Hybrid vs Pure SSM)가 모두 달라, 관찰된 결과가 특정 모델의 특성인지 SSM 전체의 공통 특성인지 단정하기 어렵다.
- 본 연구는 Mamba-2 구성 요소에만 집중하고 FFN(Feed-Forward Network) 레이어는 제외하였다. 실제 전체 모델의 효율성을 극대화하려면 SSM과 FFN 간의 상호작용을 고려한 통합 Pruning 연구가 필요하다.

## 📌 TL;DR

본 논문은 SSM 기반 LLM의 압축 가능성을 탐구하기 위해 WANDA 및 다양한 구조적 Pruning 기법을 적용하고 분석하였다. 실험 결과, SSM 모델은 **WANDA와 State pruning에는 강건**하지만, **Head pruning과 $\text{OutProj}$ 레이어의 제거에는 매우 취약**하다는 것을 발견하였다. 이 연구는 SSM 기반 LLM을 더욱 효율적으로 만들기 위해 어떤 부분을 보존하고 어떤 부분을 제거해야 하는지에 대한 실무적인 가이드를 제공하며, 향후 SSM 모델 최적화 및 경량화 연구의 기초 자료로 활용될 가능성이 크다.
