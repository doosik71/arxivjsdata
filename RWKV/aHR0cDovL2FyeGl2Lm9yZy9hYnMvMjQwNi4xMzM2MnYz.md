# VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models

Haowen Hou, Peigen Zeng, Fei Ma, and Fei Richard Yu (2024)

## 🧩 Problem to Solve

본 논문은 현재 주류를 이루고 있는 Transformer 기반의 Visual Language Models(VLMs)가 가진 계산 및 메모리 복잡도 문제를 해결하고자 한다. Transformer의 핵심인 self-attention 메커니즘은 시퀀스 길이($L$)에 따라 계산량과 메모리 요구량이 제곱($O(L^2)$)으로 증가하는 특성이 있다. 이러한 특성은 추론 비용을 급격히 상승시키며, 특히 자원이 제한된 엣지 디바이스(edge devices)로의 배포와 적용을 어렵게 만드는 결정적인 병목 현상이 된다.

따라서 본 연구의 목표는 효율적인 linear Recurrent Neural Networks(RNNs) 아키텍처, 특히 RWKV(Receptance Weighted Key Value) 모델을 VLM에 최초로 적용하여, 기존 Transformer 기반 모델과 경쟁 가능한 성능을 유지하면서도 선형적인 확장성($O(L)$)과 효율적인 추론 속도를 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 고성능 언어 모델인 RWKV를 기반으로 시각적 정보를 처리할 수 있는 능력을 부여하는 것이며, 이를 위해 다음과 같은 세 가지 핵심 설계 요소를 제안한다.

1. **Data-dependent Recurrence**: 기존의 데이터 독립적인 재귀 구조를 넘어, 입력 데이터에 따라 동적으로 변화하는 Token Shift와 Time Mixing 메커니즘을 도입하여 모델의 용량과 표현 능력을 확장하였다.
2. **Sandwich Prompt**: RNN의 순차적 특성으로 인해 발생하는 정보 망각 문제를 해결하기 위해, 지시어(instruction) 사이에 이미지 토큰을 배치하는 샌드위치 구조의 프롬프트를 제안하여 모델이 이미지 정보를 보다 효과적으로 추출하고 활용하게 하였다.
3. **2D Image Scanning**: 1차원 시퀀스 처리에 최적화된 RNN의 한계를 극복하기 위해, 이미지를 전방, 후방, 상방, 하방 등 다양한 방향으로 스캔하는 2D 스캐닝 메커니즘을 도입하여 시각적 정보의 2차원적 특성을 모델링하였다.

## 📎 Related Works

### 기존 Visual Language Models (VLMs)

Flamingo, BLIP-2, LLaVA, MiniGPT-4와 같은 기존 VLM들은 강력한 LLM 아키텍처를 기반으로 시각적 입력을 통합하여 뛰어난 시각적 이해 및 추론 능력을 보여주었다. 그러나 이러한 모델들은 대부분 Transformer 아키텍처를 사용하므로, 앞서 언급한 제곱 복잡도 문제로 인해 훈련 및 배포 비용이 매우 높다는 한계가 있다.

### Linear RNN Large Language Models

최근 Transformer의 대안으로 Linear RNN들이 등장하였으며, 특히 RWKV는 시퀀스 길이에 대해 선형적인 공간 복잡도($O(L)$)와 상수 시간의 추론 복잡도($O(1)$)를 가지면서도 대규모 데이터에서 Transformer에 필적하는 성능을 보여주었다. RWKV는 temporal decay를 통해 과거 정보의 영향력을 조절하고 token-shift 메커니즘으로 인접 토큰 간의 정보를 통합한다. 본 논문은 이러한 효율적인 Linear RNN 구조를 멀티모달 태스크로 확장한 첫 번째 시도라는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

VisualRWKV는 사전 훈련된 RWKV 언어 모델을 백본으로 사용하며, 시각 인코더(Vision Encoder)와 이를 LLM의 입력 공간으로 연결하는 프로젝터(Projector)로 구성된다. 훈련은 1단계 시각-언어 정렬 사전 훈련(Alignment Pretraining)과 2단계 시각 지시어 튜닝(Visual Instruction Tuning)으로 진행된다.

### 핵심 구성 요소 및 상세 설명

#### 1. Data-dependent Recurrence

기존 RWKV의 데이터 독립적 구조를 개선하여 입력 값에 따라 가중치가 변하는 구조를 도입하였다.

- **Data-dependent Token Shift**: 현재 토큰 $x_t$와 이전 토큰 $x_{t-1}$의 선형 결합 시, 고정된 $\mu$ 대신 LoRA와 $\text{ddlerp}$(data-dependent linear interpolation)를 사용하여 동적으로 결정한다.
  $$\text{lora}_\alpha(x) = \lambda_\alpha + \tanh(x A_\alpha) B_\alpha$$
  $$\text{ddlerp}_\alpha(a, b) = a + (b - a) \odot \text{lora}_\alpha(a + (b - a) \odot \mu_x)$$
  최종적으로 $\alpha_t = \text{ddlerp}_\alpha(x_t, x_{t-1}) W_\alpha$ 형태로 계산되어 모델의 용량을 확장한다.

- **Data-dependent Time Mixing**: 고정된 시간 감쇠 벡터 $w$를 입력 데이터 $x_t$에 반응하는 동적 벡터 $w_t$로 변경하였다.
  $$d_t = \text{lora}_d(\text{ddlerp}_d(x_t, x_{t-1}))$$
  $$w_t = \exp(-\exp(d_t))$$
  이를 통해 모델은 고정된 구조에 얽매이지 않고 다양한 입력 데이터에 더 유연하게 적응할 수 있다.

#### 2. Sandwich Prompt

RNN은 정보를 한 번 처리하면 다시 되돌아가서 볼 수 없는 순차적 특성이 있다. 이를 해결하기 위해 다음과 같은 프롬프트 전략을 비교 분석하였다.

- **Image First**: $\langle \text{Image} \rangle \to \langle \text{Instruction} \rangle$
- **Image Last**: $\langle \text{Instruction} \rangle \to \langle \text{Image} \rangle$
- **Sandwich Prompt**: $\langle \text{Instruction\_part1} \rangle \to \langle \text{Image} \rangle \to \langle \text{Instruction\_part2} \rangle$

샌드위치 프롬프트는 모델이 먼저 지시어를 통해 무엇을 찾아야 할지 인지한 후 이미지를 처리하고, 다시 한번 지시어를 확인하여 답변을 생성하게 함으로써 정보 손실을 최소화하고 최적의 조건을 제공한다.

#### 3. Image Scanning

시각 데이터의 2차원적 특성을 반영하기 위해 RWKV 블록 내에 다양한 스캐닝 방향을 교차 배치하였다.

- **Unidirectional**: 전방 스캔만 수행 (Base)
- **Bidirectional**: 전방 스캔과 후방 스캔 블록을 교대로 배치
- **Multidirectional**: 전방, 후방, 상방, 하방 스캔 블록을 교대로 배치

실험 결과, 전방/후방을 교차하는 Bidirectional 방식이 2D 시각 정보 처리에서 가장 효율적임을 확인하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: LLaVA-1.5와 동일한 데이터를 사용 (LAION-CC-SBU 558K, GPT 생성 지시어 데이터 150K, VQA 데이터 515K).
- **평가 지표**: VQA-v2, GQA, ScienceQA, TextVQA, MME, MMBench, MMBench-CN, POPE 등 8개 벤치마크 사용.
- **비교 대상**: LLaVA-1.5 (7B) 및 기타 SOTA VLM들.

### 주요 결과

- **정량적 성능**: VisualRWKV-7B는 LLaVA-1.5 7B와 동일한 훈련 데이터를 사용했음에도 불구하고 SQA, GQA, MMB, MMB-cn 등 4개 벤치마크에서 더 높은 성능을 기록하였다. 특히 중국어 벤치마크인 MMB-cn(63.7 vs 30.5)에서 압도적인 우위를 보여, RWKV 백본의 강력한 다국어 능력을 입증하였다.
- **효율성 분석**: LLaVA-1.5와 비교했을 때 추론 속도가 **3.98배** 빠르며, 추론 길이가 24K 토큰에 도달했을 때 GPU 메모리 사용량을 **54%** 절감하였다. 이는 RWKV가 고정된 상태(state) 크기를 유지하기 때문에 메모리 사용량이 시퀀스 길이에 관계없이 일정하기 때문이다.
- **텍스트 능력**: 시각 지시어 튜닝 후에도 텍스트 처리 능력이 저하되지 않았으며, 오히려 일부 영어 데이터셋에서는 성능이 향상되었다.

### 절제 연구(Ablation Study) 결과

- **Data-dependence**: 데이터 의존적 재귀 구조를 적용했을 때 성능이 비약적으로 상승하여, Linear RNN 기반 VLM에서 이 구조가 필수적임을 확인하였다.
- **Prompting**: 샌드위치 프롬프트가 가장 우수한 성능을 보였으며, 이미지 토큰 수가 적을 때도 정보 손실을 효과적으로 억제하였다.
- **Scanning**: 단방향 스캔보다 양방향(Bidirectional) 스캔이 2D 시각 정보 이해에 훨씬 유리하였다.
- **Learning Rate**: LLaVA($2\times 10^{-5}$)보다 더 높은 초기 학습률($4\times 10^{-5}$)이 필요함을 발견하였다.

## 🧠 Insights & Discussion

### 강점 및 성과

VisualRWKV는 Linear RNN을 VLM에 성공적으로 적용하여, Transformer 기반 모델의 고질적인 문제인 계산 복잡도 문제를 해결하면서도 경쟁력 있는 성능을 확보하였다. 특히 다국어 처리 능력과 추론 효율성(속도 및 메모리) 면에서 압도적인 강점을 가진다.

### 한계 및 미해결 질문

1. **Recall 능력의 한계**: TextVQA와 같이 이미지에서 특정 텍스트 정보를 정확히 추출하여 회상해야 하는 태스크에서는 LLaVA-1.5보다 낮은 성능을 보였다. 이는 RNN 계열 아키텍처가 가진 일반적인 MQAR(Multi-Query Associative Recall) 문제로 해석된다.
2. **다중 이미지 처리 불가**: 현재는 단일 이미지 처리만 가능하며, 다중 이미지 입력 및 처리를 위한 데이터와 컨텍스트 길이 확장이 필요하다.
3. **인코더 최적화 미비**: LLaVA-1.5와의 공정한 비교를 위해 CLIP-L 인코더를 고정하여 사용하였으나, 더 발전된 시각 인코더를 사용한다면 성능을 더욱 끌어올릴 가능성이 있다.

### 비판적 해석

본 논문은 RNN의 효율성을 입증했으나, '회상(Recall)' 능력의 부족이라는 RNN의 근본적 한계를 완전히 해결하지는 못했다. 다만, 저자들이 언급한 고해상도 이미지 처리(VisualRWKV-HD)나 하이브리드 구조(Attention 일부 도입)를 통해 이 문제를 완화할 수 있음을 시사하였다.

## 📌 TL;DR

본 연구는 효율적인 Linear RNN인 RWKV를 최초로 VLM에 적용한 **VisualRWKV**를 제안한다. **Data-dependent Recurrence**, **Sandwich Prompt**, **Bidirectional Scanning**이라는 세 가지 핵심 설계를 통해 Transformer 기반의 LLaVA-1.5와 대등하거나 일부 능가하는 성능을 달성하였다. 특히 **추론 속도는 약 4배 빠르고 GPU 메모리는 절반 이상 절약**할 수 있어, 향후 대규모 시각-언어 모델의 경량화 및 엣지 디바이스 배포 연구에 매우 중요한 이정표를 제시한 연구라고 평가할 수 있다.
