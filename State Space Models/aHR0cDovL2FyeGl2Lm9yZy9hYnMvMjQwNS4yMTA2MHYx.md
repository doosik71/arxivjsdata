# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
Tri Dao, Albert Gu

## 🧩 Problem to Solve
Transformer 모델은 언어 모델링에서 큰 성공을 거두었지만, 학습 시퀀스 길이에 대한 이차적인 복잡성과 추론 시 선형적인 캐시 크기 요구 사항으로 인해 효율성 문제가 발생합니다. 이와 병행하여 선형적 시퀀스 확장이 가능한 상태 공간 모델(SSM)이 등장하여 작은 규모의 언어 모델링에서 Transformer와 동등하거나 더 나은 성능을 보여주었지만, Transformer 커뮤니티의 이론적 이해 및 하드웨어 최적화 노력과는 별개로 발전해 왔습니다. 본 연구는 Transformer의 효율성 문제를 해결하고 SSM을 더 효율적으로 이해하고 훈련하기 위해, 이 두 모델 패밀리 간의 이론적, 시스템적 연결을 구축하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **SSM과 구조화된 행렬의 동등성:** 상태 공간 모델(SSM)과 잘 연구된 구조화된 행렬 클래스인 semiseparable 행렬 간의 동등성을 이론적으로 입증했습니다. 이는 SSM의 새로운 속성 및 알고리즘을 밝히는 핵심 연결고리입니다.
*   **선형 어텐션 이론 개선 및 일반화:** 선형 어텐션(Linear Attention)의 순환 형태에 대한 명확한 텐서 축약 기반 증명을 제공하고, 이를 구조화된 마스크 어텐션(Structured Masked Attention, SMA)이라는 새로운 모델 패밀리로 일반화했습니다.
*   **상태 공간 이중성(State Space Duality, SSD) 프레임워크:** SSM과 SMA가 서로 이중성을 가지는 큰 교집합을 형성하며, 선형 및 이차 형태를 모두 가짐을 보여주어 두 모델 클래스 간의 깊은 관계를 정립했습니다. 또한, 빠른 순환 형태를 가진 모든 커널 어텐션 메서드는 SSM이어야 함을 증명했습니다.
*   **효율적인 SSD 알고리즘 (Mamba-2의 핵심 레이어):** SSD 프레임워크를 기반으로 Mamba의 선택적 SSM을 개선한 Mamba-2 아키텍처의 핵심 레이어를 위한 새로운 SSD 알고리즘을 개발했습니다. 이 알고리즘은 기존 Mamba의 최적화된 선택적 스캔 구현보다 2-8배 빠르며, 훨씬 더 큰 재귀 상태 크기를 처리할 수 있습니다.
*   **Mamba-2 아키텍처 설계:** 어텐션에서 영감을 받은 아이디어를 SSM에 적용하여, 텐서 병렬화에 적합한 병렬 매개변수 투영과 안정성을 위한 추가 정규화 레이어를 포함하는 Mamba-2 아키텍처를 제안했습니다. 이는 Mamba-2가 perplexity 및 벽시계 시간 모두에서 Mamba 및 Transformer++를 Pareto 우위함을 보여주었습니다.
*   **시스템 최적화 적용:** SSD 프레임워크를 통해 Transformer를 위해 개발된 텐서 병렬화, 시퀀스 병렬화, 가변 길이 시퀀스 처리와 같은 시스템 최적화 기법들을 Mamba-2에 효율적으로 적용할 수 있게 했습니다.

## 📎 Related Works
*   **Transformers:** GPT, Llama (효율성 문제 해결 및 성능 비교 대상)
*   **State Space Models (SSMs):** S4 (Gu, Goel, and Ré 2022), Mamba (Gu and Dao 2023) (선형 확장성을 가진 대안 시퀀스 모델)
*   **Linear Attention:** Katharopoulos et al. (2020) (선형 RNN과의 연결을 처음 제시한 선행 연구)
*   **기타 효율적인 어텐션/SSM 변형:** RetNet (Y. Sun et al. 2023), GateLoop (Katsch 2023), TransNormerLLM (Qin, Dong Li, et al. 2023), Gated Linear Attention (GLA) (Yang et al. 2024), HGRN (Qin, Yang, and Zhong 2023), Griffin (De et al. 2024), RecurrentGemma (Botev et al. 2024), Jamba (Lieber et al. 2024), xLSTM (Beck et al. 2024), RWKV (B. Peng, Alcaide, et al. 2023).
*   **구조화된 행렬:** Toeplitz, Cauchy, Vandermonde, butterfly 행렬 (효율적인 모델 구축에 사용). 특히 semiseparable 행렬 (Pernet and Storjohann 2018).
*   **하드웨어 최적화:** FlashAttention-2 (Dao 2024) (최적화된 어텐션 구현), Megatron (Shoeybi et al. 2019) (텐서 병렬화).

## 🛠️ Methodology
본 연구의 핵심 방법론은 **상태 공간 이중성(Structured State Space Duality, SSD) 프레임워크**와 이를 활용한 **효율적인 SSD 알고리즘**입니다.

1.  **상태 공간 모델(SSM)의 행렬 변환 형태:**
    *   SSM의 재귀 방정식
        $$h_t = A_t h_{t-1} + B_t x_t$$
        $$y_t = C_t^{\top} h_t$$
        를 사용하여 입력 시퀀스 $x \in \mathbb{R}^{T}$를 출력 시퀀스 $y \in \mathbb{R}^{T}$로 매핑하는 행렬 변환 $Y = MX$로 표현합니다. 여기서 행렬 $M$은 SSS(Sequentially Semiseparable) 표현을 가지는 $N$-semiseparable 행렬입니다 ($M_{ji} = C_j^{\top} A_{j:\text{i}} B_i$).
    *   $N$-semiseparable 행렬은 $O(NT)$ 매개변수로 압축 가능하며, $O(NT)$ 시간 복잡도로 행렬-벡터 곱셈을 수행할 수 있습니다.
2.  **구조화된 마스크 어텐션(SMA) 프레임워크:**
    *   마스크 어텐션 $Y = (L \circ (QK^{\top}))V$를 4방향 텐서 축약으로 재해석합니다:
        $$Y = \text{contract(TN, SN, SP, TS -> TP)}(Q, K, V, L)$$
    *   표준 이차 어텐션 계산 순서와 선형 어텐션의 순환 계산 순서가 이 축약의 서로 다른 연산 순서에 해당함을 보입니다.
    *   $L$을 sub-quadratic 행렬 곱셈이 가능한 구조화된 행렬로 설정함으로써 SMA를 정의하고, 이는 선형 어텐션을 일반화합니다.
3.  **SSM과 SMA의 이중성 (SSD):**
    *   $A$ 행렬이 스칼라-항등(scalar-identity) 구조를 가진 SSM의 경우, naive한 이차 계산이 마스크 커널 어텐션의 한 형태로 정확히 일치함을 보입니다.
    *   $L$ 마스크가 1-semiseparable 구조를 가진 SMA는 대각선 SSM의 특수한 경우로 나타납니다.
    *   효율적인 자기회귀(autoregressive) 어텐션 프로세스는 semiseparable 행렬을 마스크로 사용해야 함을 증명하여, SSM과 SMA가 본질적으로 연결되어 있음을 보여줍니다.
4.  **하드웨어 효율적인 SSD 알고리즘 (Mamba-2의 핵심):**
    *   SSM 행렬 $M$을 블록 분해하여 계산합니다.
    *   **대각선 블록:** 각 시퀀스 청크(chunk) 내의 연산으로, 듀얼 이차 SMA 형태를 사용하여 효율적인 행렬 곱셈으로 병렬 처리됩니다. 작은 청크 길이에 최적화되어 있습니다.
    *   **비대각선 블록:** 시퀀스 청크 간의 연산으로, semiseparable 행렬의 랭크-1 분해 특성을 활용하여 더 작은 순환 관계로 축소됩니다.
    *   이 블록 분해 기반 알고리즘은 총 $O(TN^2)$ 학습 FLOPs, $O(TN)$ 추론 FLOPs, $O(N^2)$ 추론 메모리를 필요로 하며, 대부분 행렬 곱셈 연산으로 구성되어 GPU와 같은 최신 가속기의 이점을 활용합니다.
5.  **Mamba-2 아키텍처 설계:**
    *   **병렬 매개변수 투영:** SSM의 $A, B, C, X$ 매개변수를 블록 초기에 단일 투영으로 병렬 생성하여 텐서 병렬화 효율성을 높입니다.
    *   **추가 정규화:** 최종 출력 투영 직전에 정규화 레이어를 추가하여 학습 안정성을 향상시킵니다.
    *   **Multi-input SSM (MIS) 헤드 패턴:** Multi-value attention (MVA)와 유사하게 $B, C$ 매트릭스를 입력 $X$의 모든 채널에 걸쳐 공유하는 구조를 채택했습니다.

## 📊 Results
*   **Multi-Query Associative Recall (MQAR):** Mamba-2는 Mamba-1 및 표준 어텐션보다 훨씬 우수한 성능을 보였습니다. 특히 상태 크기 ($N$)를 16에서 64, 256으로 늘렸을 때 MQAR 성능이 일관되게 향상되어, SSM의 정보 기억 능력에서 상태 크기의 중요성을 입증했습니다.
*   **언어 모델링 스케일링 법칙:** Pile 데이터셋에서 훈련된 모델들의 스케일링 법칙을 분석한 결과, Mamba-2는 perplexity와 실제 벽시계 시간 모두에서 Mamba 및 Transformer++ 레시피를 Pareto 우위하는 것으로 나타났습니다.
*   **제로샷 평가:** Pile 데이터셋 300B 토큰으로 훈련된 Mamba-2 (2.7B 매개변수)는 Mamba-2.8B, Pythia-2.8B, 심지어 Pythia-6.9B보다 표준 다운스트림 제로샷 평가에서 더 좋은 성능을 보였습니다.
*   **효율성 벤치마크:**
    *   SSD 알고리즘은 Mamba의 최적화된 스캔 구현보다 2-8배 빠릅니다.
    *   시퀀스 길이 $2K$ 이상에서는 FlashAttention-2보다 빠르게 작동합니다.
    *   Mamba의 스캔 구현은 상태 확장이 증가함에 따라 선형적으로 느려지지만, SSD는 훨씬 큰 상태 확장 요소를 최소한의 성능 저하로 처리할 수 있습니다.
*   **하이브리드 모델:** 전체 레이어 중 약 10%를 어텐션 레이어로 조합한 하이브리드 아키텍처가 순수 Mamba-2 또는 Transformer++ 아키텍처보다 더 좋은 perplexity를 달성하여, SSD와 어텐션 레이어의 상호 보완성을 입증했습니다.

## 🧠 Insights & Discussion
*   **개념적 연결의 중요성:** 상태 공간 이중성(SSD) 프레임워크는 SSM과 어텐션 모델 간의 개념적 간극을 메우며, Transformer에서 개발된 알고리즘 및 시스템 최적화 기술을 SSM에 효과적으로 적용할 수 있는 길을 열었습니다.
*   **Mamba-2의 실용적 개선:** Mamba-2는 SSD 프레임워크를 바탕으로 설계되어, Mamba-1의 효율성을 크게 개선하고 언어 모델링에서 Transformers와 동등하거나 우수한 성능을 달성함으로써 이론적 연결이 실제 모델 개선으로 이어질 수 있음을 입증했습니다.
*   **SSM의 상태 크기 중요성:** MQAR과 같은 연관 기억 태스크에서의 Mamba-2 성능 향상은 SSM의 상태 크기 ($N$)가 정보 기억 용량과 직결되며, 복잡한 태스크에서 중요한 역할을 한다는 것을 보여줍니다.
*   **SSM과 어텐션의 상호 보완성:** 하이브리드 모델 실험을 통해 SSM 레이어는 일반적인 시퀀스-투-시퀀스 매핑에, 어텐션 레이어는 특정 토큰 검색 메커니즘으로 효과적으로 작용하며 서로 보완적임을 시사합니다. 이는 최적의 LLM 아키텍처를 위한 새로운 방향을 제시합니다.
*   **새로운 위치 정보 인코딩:** SSD의 1-semiseparable 마스크는 데이터 의존적인 방식으로 위치 정보를 제공하며, 이는 Transformer의 휴리스틱한 위치 임베딩을 대체하는 보다 원리적인 상대적 위치 임베딩 형태로 해석될 수 있습니다.
*   **향후 연구 방향:** semiseparable 행렬의 풍부한 이론을 활용하여 더욱 일반적인 SSM에 대한 하드웨어 효율적인 알고리즘 개발, 비인과적 Mamba 변형 설계, 소프트맥스 어텐션과 sub-quadratic 모델 간의 간극 특성화, SSM에 대한 해석 가능성 기법 전이 등 다양한 연구 기회를 제시합니다.

## 📌 TL;DR
이 논문은 구조화된 행렬(semiseparable matrices)을 통해 상태 공간 모델(SSM)과 어텐션을 연결하는 **상태 공간 이중성(Structured State Space Duality, SSD)** 프레임워크를 제안합니다. 이 프레임워크는 기존 Mamba보다 2-8배 빠르고, Transformer에 필적하는 언어 모델 성능을 보이며, 특히 긴 시퀀스 및 연관 기억 태스크에서 뛰어난 효율성과 성능을 제공하는 새로운 아키텍처 **Mamba-2**의 설계와 효율적인 알고리즘 개발을 가능하게 했습니다.