# Routing Mamba: Scaling State Space Models with Mixture-of-Experts Projection

Zheng Zhan, Liliang Ren, Shuohang Wang, Liyuan Liu, Yang Liu, Yeyun Gong, Yanzhi Wang, Yelong Shen (2025)

## 🧩 Problem to Solve

본 논문은 Linear State Space Models (SSMs), 특히 Mamba 아키텍처의 모델 용량(Capacity)을 효율적으로 확장하는 문제를 다룬다. Mamba는 추론 시 계산 및 메모리 복잡도가 일정하며, 긴 시퀀스 모델링에서 Transformer의 강력한 대안으로 부상하였다. 일반적으로 대규모 언어 모델(LLM)의 파라미터를 효율적으로 확장하기 위해 Mixture-of-Experts (MoE) 기법이 사용되지만, 이를 SSM에 단순하게 통합하려는 시도는 성능 저하나 지연 시간(Latency) 증가와 같은 문제로 인해 성공적이지 못했다.

특히 기존의 MoE 접근 방식은 주로 Feed-forward Network (FFN) 층에 집중되어 있었으며, Mamba와 같이 FFN 층이 없는 구조에서는 적용이 어렵거나, Mamba의 투영 층(Projection layers)에 개별적으로 MoE를 적용했을 때 성능이 오히려 하락하는 경향을 보였다. 따라서 본 연구의 목표는 Mamba의 구조적 특성을 고려하여, 모델의 표현력을 효과적으로 확장하면서도 계산 효율성을 유지할 수 있는 새로운 MoE 통합 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Routing Mamba (RoM)**라는 새로운 프레임워크의 제안이다. RoM의 중심 아이디어는 Mamba의 선형 투영 층들을 Sparse Mixture of Experts로 변환하되, 이들 간의 **라우팅 결정(Routing Decisions)을 공유**하는 것이다.

단순히 각 층에 독립적인 라우터를 두는 대신, 공유 라우팅 전략을 통해 토큰별로 최적의 "전문가 경로(Expert Pathway)"를 설정함으로써 라우터의 학습 난이도를 낮추고 전문가 간의 시너지 효과를 극대화한다. 또한, 모든 파라미터를 확장하는 것이 아니라 계산 및 표현력에 영향이 큰 주요 투영 층만을 선택적으로 확장함으로써 효율적인 스케일링을 달성하였다.

## 📎 Related Works

**State Space Models (SSMs):** Mamba는 선택적 상태 공간(Selective State Spaces)을 도입하여 입력 데이터에 따라 정보를 선택적으로 기억함으로써 시퀀스 모델링의 효율성을 극대화하였다. 최근에는 Mamba-2, Jamba, Samba와 같은 변형 모델들이 등장하며 확장성과 효율성을 증명하고 있다.

**Mixture of Experts (MoE):** MoE는 입력 토큰에 따라 일부 전문가 네트워크만을 활성화하여 계산 비용을 유지하면서 파라미터 수를 획기적으로 늘리는 기법이다. Transformer의 FFN 층에 적용된 사례는 많으나, SSM에 이를 적용한 사례는 드물다.

**기존 접근 방식과의 차별점:** 기존의 MoE-Mamba 시도들은 Mamba의 투영 층들에 독립적으로 MoE를 적용하였다. 하지만 본 논문은 이러한 방식이 라우팅 로직을 파편화하여 일관된 토큰 특화 표현 학습을 방해한다는 점을 지적한다. RoM은 공유 라우팅을 통해 이를 해결하며, 단순한 FFN-MoE 결합을 넘어 SSM 내부의 투영 메커니즘 자체를 MoE화 했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

RoM은 Mamba 레이어 내의 주요 선형 투영 층인 **Convolution Projection ($\text{Conv Proj}$), Gate Projection ($\text{Gate Proj}$), 그리고 Output Projection ($\text{Out Proj}$)**을 전문가 네트워크의 집합으로 구성한다.

### 공유 라우팅 메커니즘 (Shared Routing Mechanism)

RoM은 하나의 라우터를 통해 선택된 Top-K 전문가를 결정하고, 이 결정을 세 가지 투영 층에서 공유한다. 라우팅 가중치는 다음과 같이 계산된다:

$$P(X_t) = \text{Softmax}(X_t \cdot W_r)$$
$$R_i(X_t) = P(X_t) \cdot \mathbb{1}_{i \in \text{TopK}(P)}$$

여기서 $W_r \in \mathbb{R}^{D_m \times N}$은 학습 가능한 라우터 파라미터이며, $\mathbb{1}$은 지시 함수(Indicator function)이다.

### 상세 연산 절차 및 방정식

1. **Gate Projection:** 먼저 $\text{Gate Proj}$에서 공유 라우팅을 통해 가중합을 계산한다.
   $$G = \text{SiLU}\left( \sum_{i=1}^{N} \mathbb{1}_{i \in \text{TopK}(\text{Softmax}(X_t \cdot W_r))} \cdot X_t W_{g,i} \right)$$

2. **Conv Projection:** $\text{Gate Proj}$에서 결정된 동일한 전문가 인덱스를 사용하여 입력을 확장한다.
   $$H_t = \sum_{i=1}^{N} \mathbb{1}_{i \in \text{TopK}(P_G)} \cdot X_t W_{in,i}$$

3. **Output Projection:** 최종 출력 단계에서도 동일한 라우팅 결정을 공유하며, 전문가별 출력값에 가중치를 곱해 합산한다.
   $$O_t = \sum_{i=1}^{N} R_i(X_t) \cdot E_i(Y_t, X_t)$$
   $$E_i(Y_t, X_t) = Y_t \odot G W_{out,i}$$

### 설계 선택 (Design Choices)

- **선택적 확장:** 파라미터 수가 상대적으로 적은 $x \text{ Proj}, dt \text{ Proj}, \text{Conv 1D}$ 층은 모든 전문가가 공유하는 단일 파라미터 셋을 사용한다. 이는 Multi-Query Attention에서 KV 헤드를 공유하는 방식과 유사하며, 불필요한 파라미터 증가를 막고 효율성을 높인다.
- **부하 균형 (Load Balance):** 일반적으로 MoE 모델은 Auxiliary loss를 통해 전문가 간 부하를 균등하게 배분하지만, RoM은 별도의 부하 균형 손실 함수 없이도 자연스럽게 부하가 분산됨을 확인하였다.

## 📊 Results

### 실험 설정

- **데이터셋:** SlimPajama (20B 토큰 사전 학습)
- **평가 지표:** Perplexity (PPL) 및 다운스트림 태스크(LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande)의 평균 정확도
- **모델 규모:** 활성 파라미터(Active Params) 115M부터 1.3B까지, 총 파라미터 최대 10B 규모로 실험 수행

### 주요 결과

1. **Dense Mamba 대비 효율성:** RoM은 동일한 PPL 성능을 내기 위해 Dense Mamba 모델보다 최대 $2.3\times$ 적은 활성 파라미터만으로도 충분함을 보였다. (예: RoM 1.3B 모델은 총 10B 파라미터를 가지나, 활성 파라미터는 훨씬 적음)
2. **하이브리드 모델(Samba) 적용:** Samba 모델에 RoM을 적용했을 때, 유사한 성능을 내는 Dense 확장 모델 대비 **FLOPS를 23% 절감**하였다.
3. **길이 외삽(Length Extrapolation) 능력:** 4K, 8K, 16K의 다양한 훈련 시퀀스 길이에서 일관된 PPL 성능을 보였으며, 훈련 시보다 더 긴 시퀀스에 대해서도 우수한 일반화 능력을 유지하였다.
4. **다운스트림 성능:** Hybrid RoM + FFN-MoE 구조는 단순 FFN-MoE 구조보다 더 적은 총 파라미터로 대등하거나 더 높은 평균 정확도(약 49.2%~50.1%)를 달성하였다.

## 🧠 Insights & Discussion

### RoM이 효과적인 이유

본 논문은 RoM의 성공 요인을 **"통합 전문가 경로(Unified Expert Pathway)"** 개념으로 설명한다. Mamba의 투영 층들은 기능적으로 상호 의존적이다. 만약 각 층마다 독립적인 라우터를 둔다면, 하나의 토큰이 층마다 서로 다른 전문가를 거치게 되어 일관성 없는 표현이 생성될 위험이 크다. 반면, 공유 라우팅은 토큰 하나에 대해 최적화된 일련의 전문가 셋(Pathway)을 지정함으로써, 전문가들이 더 일관되고 전문화된 기능을 학습할 수 있게 한다.

### 아키텍처별 전략의 차이

실험을 통해 SSM 종류에 따라 최적의 MoE 적용 방식이 다름을 발견하였다.

- **Mamba2, Gated DeltaNet:** 구조가 더 통합적이며 단순하므로, 모든 주요 투영 층을 전문가화하는 **포괄적 전문가화(Comprehensive Expertization)**가 효과적이다.
- **Original Mamba:** 매우 작은 특수 파라미터 층들이 존재하므로, 큰 투영 층만 전문가화하고 작은 층은 공유하는 **선택적 전문가화(Selective Expertization)**가 더 유리하다.

### 한계 및 비판적 해석

본 논문은 RoM의 효율성을 강력하게 입증하였으나, 최적의 전문가 수($N$)나 Top-K 값에 대한 하이퍼파라미터 탐색 범위가 제한적이다. 또한, 빠르게 진화하는 SSM 변형 모델들 전체에 대해 일반화될 수 있는지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

본 연구는 Mamba의 투영 층들을 Sparse MoE 구조로 확장하되, 라우팅 결정을 공유하여 효율성을 극대화한 **Routing Mamba (RoM)**를 제안하였다. RoM은 Dense Mamba 대비 약 $2.3\times$ 적은 계산 비용(활성 파라미터 기준)으로 동일한 성능을 내며, 하이브리드 모델에서 FLOPS를 23% 절감하는 성과를 거두었다. 이는 SSM의 스케일링 문제를 해결하는 새로운 방향성을 제시하며, 향후 초거대 SSM 기반 언어 모델 구축에 중요한 역할을 할 가능성이 크다.
