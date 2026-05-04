# Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces

Toshihiro Ota (2024)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL) 문제를 시퀀스 모델링 문제로 전환하여 해결하려는 시도에서 출발한다. 기존의 Decision Transformer (DT)는 Causal Self-Attention 메커니즘을 사용하여 상태(state), 행동(action), 보상(reward)의 시퀀스를 모델링하며 우수한 성능을 보였으나, Self-Attention의 계산 복잡도와 시퀀스 모델링 효율성 측면에서 개선의 여지가 남아 있다.

따라서 본 연구의 목표는 최신 시퀀스 모델링 프레임워크인 Mamba를 Decision Transformer 아키텍처에 통합하여, 순차적 의사결정(sequential decision-making) 작업에서 성능 향상 및 효율적인 시퀀스 모델링 가능성을 탐색하는 것이다. 특히, Mamba의 Selective State Space Model(SSM)이 복잡한 의존성을 가진 RL 데이터에서 Transformer의 Attention 메커니즘을 효과적으로 대체할 수 있는지 검증하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Decision Transformer의 핵심 모듈인 **Causal Self-Attention을 Mamba 블록으로 대체**하는 것이다. 

Mamba는 데이터 의존적인 선택 메커니즘(Selective Mechanism)을 갖춘 구조적 상태 공간 모델(Structured State Space Model, SSM)을 사용하여, 입력 데이터에 따라 필수적인 정보는 선택적으로 추출하고 불필요한 노이즈는 필터링한다. 이를 통해 RL의 궤적(trajectory) 데이터에 존재하는 시간적 의존성과 복잡한 패턴을 더욱 정교하게 인코딩하여, 더 정확하고 강건한 의사결정 출력을 생성할 수 있다는 가설을 제시한다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 기반으로 한다.

1.  **Decision Transformer (DT):** 강화학습의 가치 함수(Value Function)나 벨만 연산자(Bellman Operator) 대신, 상태-행동-보상 시퀀스를 직접 모델링하여 최적의 행동을 예측하는 Sequence-to-Sequence 접근 방식을 제안하였다.
2.  **State Space Models (SSMs):** 선형 상미분 방정식(Linear ODE)을 기반으로 입력 신호를 잠재 상태를 통해 출력 신호로 매핑하는 모델이다. S4와 같은 구조적 SSM은 RNN과 CNN의 장점을 결합하여 긴 시퀀스 모델링에서 효율성을 보였다.
3.  **Mamba:** 기존 SSM의 데이터 불변성(time-invariant) 문제를 해결하기 위해 데이터 의존적 선택 메커니즘을 도입하고, 하드웨어 최적화 설계를 통해 Transformer 수준의 성능과 선형 시간 복잡도를 달성한 모델이다.

기존의 Decision S4 (DS4) 등이 SSM을 RL에 적용한 사례가 있으나, 본 논문은 Mamba의 '선택적(Selective)' 특성이 RL의 복잡한 시퀀스 데이터에서 어떤 성능 차이를 만드는지에 집중한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조
Decision Mamba (DMamba)는 기본적으로 GPT 스타일의 트랜스포머 구조를 따르며, 토큰 믹싱(token-mixing) 모듈로 사용되던 Self-Attention을 Mamba 블록으로 대체하였다.

1.  **입력 데이터:** 최근 $K$ 타임스텝의 궤적 $\tilde{\tau}_i = (R_{i-K+1}, s_{i-K+1}, a_{i-K+1}, \dots, R_i, s_i, a_i)$를 입력으로 사용한다. 여기서 $R$은 Return-to-go (RTG)를 의미한다.
2.  **임베딩 층:** 입력 궤적을 선형 층(Linear layer) 또는 2D 컨볼루션 층을 통해 토큰 임베딩 $I_i \in \mathbb{R}^{3K \times D}$로 변환한다.
3.  **Mamba 레이어 스택:** 변환된 임베딩은 다음과 같은 구조의 Mamba 레이어를 통과한다.
    -   **Token-mixing layer:** LayerNorm $\rightarrow$ Mamba Block $\rightarrow$ Residual Connection.
    -   **Channel-mixing layer:** LayerNorm $\rightarrow$ Channel MLP $\rightarrow$ Residual Connection.

### Mamba 블록의 상세 동작
Mamba 블록 내부에서는 다음과 같은 연산이 수행된다.

-   입력 $x$에 대해 선형 투영을 통해 은닉 상태 $x$와 $z$를 생성한다.
-   Causal 1D Convolution과 $\text{SiLU}$ 활성화 함수를 적용한다.
-   **Selective SSM:** 입력 $x$에 따라 파라미터 $B, C, \Delta$를 동적으로 결정한다.
    $$B = \text{Linear}(x), \quad C = \text{Linear}(x)$$
    $$\Delta = \text{Softplus}(\text{Parameter} + s_\Delta(x))$$
-   Zero-Order Hold (ZOH) 이산화 규칙을 사용하여 연속 SSM을 이산 SSM으로 변환한 후, 선택적 SSM 연산을 수행한다.
-   최종적으로 gating 메커니즘($y \odot \text{SiLU}(z)$)을 거쳐 출력을 생성한다.

### 학습 및 추론 절차
-   **학습(Training):** 오프라인 RL 데이터셋에서 길이 $3K$의 서브 궤적을 샘플링한다. 모델은 현재 상태 $s_i$와 RTG $R_i$가 주어졌을 때 다음 행동 $\hat{a}_i$를 예측하며, 손실 함수는 행동 공간의 특성에 따라 다음과 같이 정의된다.
    $$\text{Loss} = \mathbb{E}_{\tilde{\tau} \sim \mathcal{M}, \pi} \left[ \frac{1}{K} \sum_{i=0}^{K-1} \mathcal{L}_{\text{MSE/CE}}(\hat{a}_i; a_i) \right]$$
    (연속적 행동은 MSE, 이산적 행동은 Cross-Entropy 손실 사용)
-   **추론(Inference):** 목표 성능을 나타내는 초기 RTG $R_0$를 설정한다. 행동을 생성한 후, 환경으로부터 받은 보상 $r_i$를 이전 RTG에서 차감하여 다음 타임스텝의 $R_i$를 갱신하는 방식으로 진행한다.

## 📊 Results

### 실험 설정
-   **데이터셋:** OpenAI Gym (D4RL benchmark) 및 Atari 게임.
-   **비교 대상:** Decision Transformer (DT), Decision S4 (DS4), Decision ConvFormer (DC).
-   **평가 지표:** 전문가 대비 정규화된 리턴 (Expert-normalized returns).

### 주요 결과
1.  **OpenAI Gym (D4RL):** Table 1 결과에 따르면, DMamba는 DT 및 DS4와 비교하여 경쟁력 있는 성능을 보였으며, 특히 일부 환경(HalfCheetah-m 등)에서는 DT보다 높은 성능을 기록하였다.
2.  **Atari:** Table 2에서 Breakout, Qbert 등의 게임을 평가한 결과, 모델별로 성능 차이가 존재하며 일부 작업에서는 DC나 DT 대비 낮은 성능을 보이기도 하였다.
3.  **Ablation Study:**
    -   **Channel-mixing layer 제거:** 이 레이어를 제거하거나 레이어 수를 늘려도 성능이 비슷하게 유지되었다. 이는 Mamba 블록 자체가 토큰 및 채널 믹싱 기능을 충분히 수행하고 있음을 시사한다.
    -   **Context length $K$ 영향:** Atari의 Breakout에서는 $K$가 길어질수록 성능이 향상되었으나, Qbert에서는 $K$가 길어질수록 성능이 급격히 저하되었다. 이는 Mamba의 선택 메커니즘이 특정 환경에서는 불필요하게 중요한 토큰을 제거하여 일반화 성능을 해칠 수 있음을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 연구는 Mamba의 Selective SSM이 강화학습의 시퀀스 모델링 작업에서도 Transformer의 Attention을 대체할 수 있는 유효한 도구임을 입증하였다. 특히, 하이퍼파라미터 최적화 없이 기본 설정만으로도 기존 DT 계열 모델들과 경쟁 가능한 성능을 냈다는 점이 긍정적이다.

### 한계 및 비판적 해석
1.  **실제 효율성 문제:** Mamba의 핵심 강점은 하드웨어 최적화를 통한 선형 시간 복잡도와 효율성이다. 하지만 본 실험에서는 RL 환경의 특성상 CPU와 GPU 간의 잦은 상호작용(interaction)이 병목 현상을 일으켜, Mamba의 아키텍처적 효율성이 실제 학습/추론 속도 향상으로 이어지지 않았다.
2.  **RL 데이터 구조의 특수성:** NLP와 달리 RL 궤적은 상태-행동-보상이 순차적으로 반복되는 고유한 구조를 가진다. 본 논문은 단순히 Mamba 블록을 대체 적용하였으나, RL 데이터 구조에 최적화된 전처리나 아키텍처 수정이 이루어지지 않았다.
3.  **하이퍼파라미터 탐색 부재:** 저자 스스로 언급했듯이, 하이퍼파라미터 튜닝이 수행되지 않아 Mamba의 잠재력을 완전히 끌어냈다고 보기 어렵다.

## 📌 TL;DR

본 논문은 Decision Transformer의 Self-Attention 메커니즘을 **Mamba의 Selective State Space Model로 대체한 'Decision Mamba'**를 제안한다. 실험 결과, DMamba는 오프라인 RL 작업에서 기존 DT 계열 모델들과 대등하거나 일부 우월한 성능을 보였다. 비록 실제 구현상의 병목으로 인해 계산 효율성 이득을 완전히 누리지는 못했으나, Selective SSM이 RL의 순차적 의사결정 모델링에 효과적일 수 있음을 확인하였다. 향후 RL 데이터 구조에 특화된 Mamba 아키텍처 설계나 Non-Markov 결정 과정으로의 확장이 중요한 연구 방향이 될 것으로 보인다.