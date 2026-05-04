# MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts

Maciej Piŕoro, Kamil Ciebiera, Krystian Król, Jan Ludziejewski, Michał Krutul, Jakub Krajewski, Szymon Antoniak, Piotr Miłoś, Marek Cygan, Sebastian Jaszczur (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)의 성능 향상을 위한 모델 확장(Scaling) 문제와 계산 효율성 사이의 트레이드오프를 해결하고자 한다. 현재 LLM 시장은 Transformer 아키텍처가 주도하고 있으나, Transformer의 Attention 메커니즘은 시퀀스 길이에 따라 계산 복잡도가 이차적으로 증가하는 한계가 있다. 최근 State Space Models(SSMs), 특히 Mamba는 선형 시간 추론과 병렬 학습을 통해 Transformer의 강력한 대안으로 부상하였다.

동시에, Mixture of Experts(MoE)는 모델의 파라미터 수를 대폭 늘리면서도 실제 연산에 사용되는 활성 파라미터(Active Parameters) 수를 낮게 유지함으로써, 계산 비용의 증가를 최소화하며 모델 용량을 확장할 수 있는 검증된 방법이다. 따라서 본 연구의 목표는 Mamba의 효율적인 시퀀스 모델링 능력과 MoE의 확장 가능성을 결합하여, 기존 Mamba나 Transformer-MoE보다 더 적은 학습 비용으로 더 높은 성능을 내는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Mamba 레이어와 MoE 레이어를 교차 배치(Interleave)** 하는 것이다. Mamba 레이어는 모든 토큰에 대해 무조건적인(Unconditional) 처리를 수행하여 시퀀스 전체의 문맥을 내부 표현으로 효율적으로 통합하고, 뒤따르는 MoE 레이어는 각 토큰에 가장 적합한 전문가(Expert)를 선택적으로 활성화하는 조건부(Conditional) 처리를 수행한다.

주요 기여 사항은 다음과 같다.

- Mamba와 Switch-style MoE를 결합한 MoE-Mamba 아키텍처를 제안하였다.
- MoE-Mamba가 vanilla Mamba와 동일한 성능에 도달하는 데 필요한 학습 단계를 2.35배 단축시켰음을 입증하였다.
- 모델 크기, 전문가의 수, 아키텍처 설계 변경 등에 대한 광범위한 실험을 통해 제안 방법론의 강건함(Robustness)을 확인하였다.
- MoE를 Mamba 블록 내부에 통합하는 다양한 대안적 설계들을 탐색하고 비교 분석하였다.

## 📎 Related Works

### State Space Models (SSMs)

SSM은 신호 처리에서 유래하여 RNN의 순차적 특성과 CNN의 병렬 학습 특성을 결합한 형태이다. 최근 Mamba는 선택적 메커니즘(Selective Mechanism)과 하드웨어 최적화 설계를 통해 Transformer에 필적하는 성능을 보이면서도 추론 시 메모리 사용량이 문맥 길이에 의존하지 않는 이점을 제공한다.

### Mixture of Experts (MoE)

MoE는 희소 활성화(Sparse Activation)를 통해 연산량(FLOPs)의 급격한 증가 없이 파라미터 수를 늘리는 기법이다. 주로 Transformer의 Feed-Forward Network(FFN) 층을 MoE로 대체하여 Mixtral과 같은 거대 모델들이 효율적으로 구현되었다.

### 차별점

기존 연구들이 주로 Transformer 기반의 MoE에 집중한 반면, 본 논문은 최신 SSM인 Mamba에 MoE를 결합하여 SSM의 확장성 한계를 돌파하려 했다는 점에서 차별성을 가진다. 또한, 단순한 결합을 넘어 Mamba 블록 내부의 프로젝션 층을 MoE로 대체하는 등 구조적 최적화를 깊이 있게 다루었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

MoE-Mamba는 vanilla Mamba의 블록 구조에 MoE 레이어를 추가한 형태이다. 기본적으로 Mamba 레이어와 MoE 레이어가 순차적으로 교차 배치되는 구조를 가진다.

### 2. 주요 구성 요소 및 역할

- **Mamba Layer**: 시퀀스 전체의 문맥을 통합하여 내부 표현을 생성하는 역할을 수행한다.
- **MoE Layer (Switch Transformer 설계)**: 입력 토큰에 대해 가장 적합한 전문가 하나만을 선택하여 연산하는 희소 레이어이다.
  - **전문가 집합**: $\text{experts}\{E_i\}_{i=1}^{N_{\text{experts}}}$ 로 정의되며, 각 전문가는 동일한 크기의 학습 가능한 Feed-Forward Network(FFN)이다.
  - **라우팅(Routing)**: 입력 임베딩 $x$에 대해 선형 투영 $W$를 통해 점수를 계산하고 Softmax를 적용한다.
    $$p_i(x) = \frac{\exp(h(x)_i)}{\sum_{j=1}^{N_{\text{experts}}} \exp(h(x)_j)}, \quad \text{where } h(x) = Wx$$
  - **출력 결정**: 가장 높은 확률을 가진 전문가 $I = \text{argmax}_i p_i(x)$ 하나만을 선택하여 출력을 계산한다.
    $$y = p_I E_I(x)$$

### 3. 학습 절차 및 손실 함수

학습은 다음 토큰 예측(Next Token Prediction) 작업으로 수행되며, 기본적으로 Cross Entropy 손실 함수를 사용한다. 또한, 특정 전문가에게 토큰이 쏠리는 현상을 방지하기 위해 **Load Balancing Loss**를 추가하였으며, 가중치 $\alpha = 0.01$을 적용하였다.

### 4. 대안적 설계 (Alternative Designs)

- **Parallel MoE-Mamba**: Mamba 레이어와 MoE 레이어를 순차적이 아닌 병렬로 배치하여 각 결과를 합산하는 구조이다.
- **Inner MoE**: Mamba 블록 내부의 세 가지 선형 투영(Conv Projection, Gate Projection, Output Projection) 중 일부 또는 전체를 MoE 레이어로 교체하는 방식이다.

## 📊 Results

### 실험 설정

- **데이터셋**: C4 데이터셋을 사용하였다.
- **비교 대상**: Vanilla Mamba, Vanilla Transformer, Transformer-MoE.
- **지표**: EMA-smoothed training log perplexity를 주 지표로 사용하였다.
- **모델 크기**: $\Box 25\text{M}$ 모델(약 10B 토큰 학습)과 $\Box 100\text{M}$ 모델(약 30B 토큰 학습) 두 가지 규모에서 검증하였다.

### 주요 결과

- **학습 효율성**: MoE-Mamba $100\text{M}$ 모델은 vanilla Mamba $100\text{M}$과 동일한 성능(Perplexity)에 도달하는 데 필요한 학습 단계가 **2.35배 적었다**.
- **정량적 성능**: Table 1에 따르면, MoE-Mamba $100\text{M}$의 최종 Log Perplexity는 2.81로, Mamba $100\text{M}$(2.99) 및 Transformer-MoE $100\text{M}$(2.88)보다 낮아 가장 우수한 성능을 보였다.
- **전문가 수의 영향**: 전문가 수가 4개 이상일 때부터 vanilla Mamba보다 성능이 우수해지며, 32개 전문가를 사용했을 때 가장 좋은 결과를 얻었다. 이는 MoE-Mamba가 전문가 수 증가에 따라 성능이 향상되는 좋은 확장성(Scaling)을 가졌음을 의미한다.
- **설계 비교**: 순차적 배치(Sequential) 구조가 병렬 배치(Parallel) 구조보다 성능 면에서 우위에 있음이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 SSM의 효율성과 MoE의 용량 확장성을 성공적으로 결합하였다. 특히, Mamba 레이어가 문맥을 통합하고 MoE 레이어가 조건부 지식을 적용하는 '무조건적-조건부 처리의 교차' 구조가 성능 향상의 핵심임을 보여주었다.

### 한계 및 비판적 해석

- **정확도와 Perplexity의 괴리**: 저자들은 MoE-Mamba가 Transformer-MoE보다 낮은 Perplexity를 기록함에도 불구하고, 일부 작업에서 정확도(Accuracy)는 더 낮게 나타나는 현상을 발견하였다. 이는 SSM이 고정된 크기의 은닉 상태(Hidden State)에 과거 정보를 압축하여 저장하기 때문에, Transformer의 Attention 메커니즘처럼 토큰을 그대로 복사(Verbatim Copying)하는 능력이 부족하기 때문으로 분석된다.
- **추론 최적화**: MoE의 도입으로 파라미터 수는 크게 늘어났으나, 활성 파라미터 수는 유지되어 추론 효율성은 보존되었다. 하지만 실제 하드웨어에서의 통신 비용이나 라우팅 오버헤드에 대한 상세한 분석은 부족한 편이다.

## 📌 TL;DR

본 논문은 효율적인 시퀀스 모델인 **Mamba**에 **Mixture of Experts(MoE)** 기법을 결합한 **MoE-Mamba**를 제안한다. 실험 결과, MoE-Mamba는 vanilla Mamba 대비 학습 속도를 **2.35배** 높이면서도 더 우수한 언어 모델링 성능을 달성하였다. 이는 SSM 기반 모델이 MoE를 통해 거대 모델로 확장될 수 있는 가능성을 제시하며, 향후 초거대 SSM 모델 설계의 중요한 기초 연구가 될 것으로 기대된다.
