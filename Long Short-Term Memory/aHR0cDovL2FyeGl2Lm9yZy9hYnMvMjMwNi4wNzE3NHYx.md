# Augmenting Language Models with Long-Term Memory
Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, Furu Wei

## 🧩 Problem to Solve
기존의 대규모 언어 모델(LLM)은 고정된 입력 길이 제한으로 인해 과거 입력으로부터 풍부한 장문 컨텍스트 정보를 활용하지 못합니다. 컨텍스트 길이를 단순히 늘리는 방식(예: GPT-3)은 계산 비용이 많이 들고(Transformer의 자기-어텐션은 이차 복잡도 $O(n^2)$) 종종 처음부터 모델을 재학습해야 합니다. 메모리 기반의 Transformer 모델(MemTRM)은 65k 토큰까지 확장되었지만, 메모리 인코딩과 융합을 위한 단일 모델을 사용하기 때문에 모델 매개변수 업데이트 시 캐시된 이전 표현이 최신 모델의 표현과 분포적으로 달라지는 '메모리 노후화(memory staleness)' 문제에 직면합니다.

## ✨ Key Contributions
*   LLM이 긴 이력을 기억할 수 있도록 하는 **LONGMEM** 프레임워크를 제안합니다.
*   원래의 LLM 백본을 메모리 인코더로 고정하고, 적응형 잔차 사이드-네트워크(SideNet)를 메모리 검색 및 리더로 사용하는 **새로운 디커플링된 네트워크 아키텍처**를 설계했습니다.
*   이러한 디커플링된 메모리 설계는 메모리 노후화 문제 없이 장기 과거 컨텍스트를 쉽게 캐시하고 업데이트하여 메모리 검색에 활용할 수 있습니다.
*   메모리 증강 적응 훈련을 통해 LONGMEM은 긴 과거 컨텍스트를 기억하고 장기 메모리를 언어 모델링에 사용할 수 있습니다.
*   제안된 메모리 검색 모듈은 무제한 길이의 컨텍스트를 메모리 뱅크에서 처리하여 다양한 하위 작업에 이점을 줍니다.
*   LONGMEM은 장문 메모리를 65k 토큰으로 확장하여 수많은 추가 시연 예제를 인컨텍스트 학습을 위한 장문 메모리로 캐시할 수 있습니다.
*   도전적인 장문 컨텍스트 모델링 벤치마크인 ChapterBreak에서 강력한 장문 컨텍스트 모델들을 능가하며, LLM 대비 메모리 증강 인컨텍스트 학습에서 현저한 개선을 달성했습니다.

## 📎 Related Works
*   **대규모 언어 모델(LLMs)**: GPT-2, GPT-3, OPT, BLOOM 등은 제로샷 프롬프팅, 인컨텍스트 학습, CoT(Chain-of-Thought) 추론과 같은 '긴급 능력(emergent abilities)'을 보여주며 NLP 분야에 혁명을 가져왔습니다.
*   **x-formers**: Transformer-XL(과거 세그먼트 캐싱), LinFormer, LongFormer, Routing Transformer, BigBird와 같은 다양한 희소 어텐션 메커니즘을 통해 $O(n^2)$ 복잡도를 $O(n \log n)$ 또는 $O(n)$으로 줄였지만, 여전히 최대 시퀀스 길이가 16k 토큰으로 제한됩니다.
*   **사이드-튜닝(Side-Tuning)**: 경량의 사이드-네트워크를 고정된 사전 학습된 네트워크와 합산하여 작업별 튜닝을 수행하는 방식입니다. LONGMEM은 사이드-네트워크 아이디어를 계승하지만, 학습 목표와 크로스-네트워크 융합 방식에서 차별점을 가집니다.
*   **Memorizing Transformer (MemTRM)**: 컨텍스트 내 토큰과 메모리에서 검색된 토큰에 대한 밀집 어텐션을 통해 희소 어텐션을 근사화하여 최대 65k 토큰까지 처리할 수 있지만, '메모리 노후화' 문제에 직면합니다.

## 🛠️ Methodology
LONGMEM은 고정된 백본 LLM에 디커플링된 메모리 모듈을 추가하여 장기 메모리를 활용합니다. 이 메모리 컨텍스트 정보를 융합하기 위해 효율적으로 연속 학습할 수 있는 경량의 잔차 SideNet을 설계했습니다.

1.  **아키텍처 구성 요소**:
    *   **고정된 백본 LLM**: 기존 LLM(예: GPT-2*)을 그라디언트 계산 없이 고정된 상태로 사용합니다. 과거 입력의 경우 $m$-번째 Transformer 디코더 레이어의 키-값 쌍을 캐시 메모리 뱅크에 저장합니다. 현재 입력의 경우 각 LLM 디코더 레이어의 히든 스테이트를 SideNet으로 전달합니다.
    *   **SideNet (잔차 사이드-네트워크)**: 백본 LLM의 출력 히든 스테이트를 입력으로 받으며, $(L-1)$개의 일반 Transformer 디코더 레이어와 하나의 특별한 **메모리 증강 디코더 레이어**로 구성됩니다. SideNet은 백본 LLM의 $L'$ 레이어 수에 비해 레이어 수 $L$이 작게 설계되며, 해당 깊이의 백본 LLM 레이어로부터 가중치를 초기화합니다.
    *   **캐시 메모리 뱅크**: 가장 최근의 $M$개 이전 입력에 대한 헤드별 키-값 쌍($Z_k, Z_v \in \mathbb{R}^{H \times M \times d}$)을 큐 형태로 유지합니다. 새로운 입력이 들어오면 가장 오래된 키-값 쌍을 제거하고 현재 시퀀스의 키-값 쌍을 추가하여 업데이트합니다.

2.  **메모리 검색 (Memory Retrieval)**:
    *   **토큰-청크 검색(Token-to-Chunk Retrieval)**: 토큰-토큰 검색 대신 가속화 및 무결성을 위해 청크($csz$개 연속 토큰) 단위로 검색을 수행합니다. 메모리 뱅크는 청크 수준에서 키-값 쌍을 저장하며, 검색 시 현재 입력 토큰의 어텐션 쿼리와 후보 청크의 평균 풀링된 어텐션 키 간의 내적을 사용하여 상위 $K/csz$개의 청크를 검색합니다. 최종적으로 이 청크들을 토큰 수준의 $K$개 키-값 쌍으로 평탄화합니다. 효율적인 검색을 위해 Faiss 툴킷을 사용합니다.

3.  **메모리 융합 (Memory Fusion)**:
    *   SideNet 내부의 특별한 **메모리 증강 레이어**에서 수행됩니다.
    *   기존의 멀티-헤드 자기-어텐션을 **공동-어텐션(joint-attention)** 메커니즘으로 확장하여 각 토큰이 로컬 컨텍스트와 검색된 메모리 컨텍스트 모두에 어텐션할 수 있도록 합니다.
    *   출력 히든 스테이트 $H^l$은 이전 레이어의 히든 스테이트 $H^{l-1}$로부터 계산된 쿼리($Q$), 키($K$), 값($V$)으로 생성된 로컬 어텐션($A$)과, 검색된 메모리 키($\tilde{K}$), 값($\tilde{V}$)으로 생성된 메모리 어텐션($M$)을 활용하여 다음과 같이 결합됩니다:
        $$A=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
        $$M=\text{Concat}\left\{\text{softmax}\left(\frac{Q_i \tilde{K}^T_i}{\sqrt{d}}\right)\tilde{V}_i\right\}_{i=1}^{|x|}$$
        $$H^l=\text{sigmoid}(g)\cdot A + (1-\text{sigmoid}(g))\cdot M$$
        여기서 $g$는 학습 가능한 헤드별 게이팅 벡터입니다.

4.  **크로스-네트워크 잔차 연결(Cross-Network Residual Connections)**:
    *   사전 학습된 백본 LLM으로부터 지식을 활용하기 위해, 백본 LLM의 2l-번째 및 (2l-2)-번째 레이어의 출력 히든 스테이트 차이를 SideNet의 l-번째 레이어 출력에 잔차 연결로 추가합니다:
        $$H^l_{Side} = f_{\Theta_l^{Side}}(H^{l-1}_{Side}) + (H^{2l}_{LLM} - H^{2l-2}_{LLM})$$
        이는 지식 전이 및 학습 안정성에 기여합니다.

5.  **훈련**:
    *   **메모리 증강 적응 훈련(Memory-augmented adaptation training)**: 표준 좌-우 언어 모델링 목표를 사용하여 SideNet만 학습합니다.
    *   **훈련 데이터 및 하이퍼파라미터**: Pile 데이터셋의 일부(BookCorpus2, Books3 등)를 훈련 코퍼스로 사용합니다. 재현된 GPT-2*(407M 매개변수, $L'=24$)를 백본 LLM으로 사용하며, SideNet은 151M 매개변수, $L=12$ 레이어를 가집니다. 메모리 크기는 65k 토큰, 검색되는 키-값 쌍은 $K=64$개, 청크 크기는 $csz=4$ 토큰입니다.

## 📊 Results
LONGMEM은 다양한 벤치마크에서 기존 강력한 모델들을 크게 능가하는 성능을 보였습니다.

*   **장문 컨텍스트 언어 모델링**:
    *   PG-22 및 ArXiv 데이터셋에서 GPT-2* 및 MemTRM 대비 현저히 낮은 perplexity(더 좋음)를 달성했습니다. PG-22의 다른 길이 구간에서 -1.38에서 -1.62 perplexity, ArXiv 데이터셋에서 -1.0 perplexity 개선을 보였습니다.
*   **ChapterBreak 벤치마크 (장문 컨텍스트 이해)**:
    *   AO3 서브셋에서 40.5%의 최고 식별 정확도를 달성하며, 강력한 장문 컨텍스트 트랜스포머 및 313배 더 큰 매개변수를 가진 GPT-3마저도 크게 능가했습니다. 이는 LONGMEM이 캐시된 메모리 내 과거 장문 컨텍스트를 잘 이해하여 미래 입력을 모델링하는 데 효과적임을 보여줍니다.
*   **메모리 증강 인컨텍스트 학습 (NLU 및 QA)**:
    *   5개의 NLU 태스크(SST-2, MPQA, MR, Subj, SST-5)에서 2000개의 추가 시연 예제를 메모리에 로드하여 학습했을 때, 20-샷 설정에서 GPT-2* 및 MemTRM 대비 평균 +8.0의 정확도 향상을 보였습니다. 4-샷 설정에서도 성능 개선이 관찰되었습니다.
    *   SQuAD 질의응답 태스크에서 200개의 추가 시연 예제를 메모리에 로드했을 때 +4.5 EM 점수 향상을 달성했습니다.
    *   이는 캐시된 메모리에 로드된 시연 예제가 보조적인 컨텍스트 정보로 활용되어 인컨텍스트 학습에 큰 도움이 됨을 입증합니다.
*   **어블레이션 스터디 (Ablation Studies)**:
    *   **청크 크기($csz$)**: NLU 태스크에서는 `csz=2`가 가장 좋은 성능을 보였는데, 이는 분류 레이블 토큰과 같은 미세한 정보를 검색하고 융합하는 데 더 적합하기 때문입니다.
    *   **메모리 크기($msz$)**: PG-22 언어 모델링 데이터셋에서는 타겟 문서의 평균 길이(예: 8k-50k 길이의 책에 대해 16k)와 일치하는 메모리 크기가 최적의 perplexity를 제공했습니다.
*   **효율성**:
    *   일반적인 자기-어텐션 기반 모델(GPT-2* 4k/8k 컨텍스트 길이)에 비해 추론 속도와 GPU 메모리 사용량을 크게 개선했습니다. (예: 3k 인-메모리 컨텍스트 사용 시, GPT-2* 4k는 1466 tokens/s, 20671MBs인 반면 LONGMEM 1k+3k는 2263 tokens/s, 13335MBs)

## 🧠 Insights & Discussion
*   **디커플링된 메모리 설계의 효과**: LONGMEM의 디커플링된 메모리 아키텍처는 "메모리 노후화" 문제를 성공적으로 해결하고 효율적인 적응 훈련을 가능하게 합니다. 백본 LLM을 고정하고 SideNet만 훈련함으로써, 사전 학습된 지식을 활용하면서도 치명적 망각(catastrophic forgetting)을 방지합니다.
*   **무제한 길이 컨텍스트 처리**: 고정된 입력 길이 제한을 넘어 무제한 길이의 컨텍스트를 캐싱하여 처리할 수 있습니다. 이는 특히 긴 문서 언어 모델링(예: 책) 및 수천 개의 시연 예제가 필요한 인컨텍스트 학습에서 매우 중요합니다.
*   **인컨텍스트 학습의 확장**: 기존 인컨텍스트 학습이 소수샷(few-shot) 시연에 국한되었던 한계를 넘어, 메모리에 수천 개의 보조 시연 예제를 캐싱하여 인컨텍스트 학습의 효율성과 성능을 크게 향상시킬 수 있음을 입증했습니다. 이는 LLM이 로컬 컨텍스트와 메모리 내 정보를 모두 활용하여 더 나은 이해와 생성을 수행하게 합니다.
*   **다양한 적용 가능성**: 제안된 방법은 장문 언어 모델링뿐만 아니라 NLU 및 QA와 같은 다양한 하위 작업에서도 뛰어난 성능을 보이며, 광범위한 장문 텍스트 시나리오에 적용 가능함을 시사합니다.

## 📌 TL;DR
LLM이 고정된 입력 길이 제한과 기존 메모리 증강 모델의 메모리 노후화 문제로 인해 장문 컨텍스트를 활용하기 어려웠습니다. LONGMEM은 고정된 백본 LLM이 메모리(키-값 쌍)를 캐시하고, 훈련 가능한 SideNet이 이 메모리를 현재 입력과 결합 어텐션 및 크로스-네트워크 잔차 연결을 통해 검색하고 융합하는 디커플링된 아키텍처를 제안합니다. 이 모델은 최대 65k 토큰의 장문 콘텐츠를 효과적으로 기억하고 활용하여, 장문 컨텍스트 언어 모델링(ChapterBreak 벤치마크)에서 최고 성능을 달성하고 수천 개의 시연 예제를 통한 메모리 증강 인컨텍스트 학습에서 상당한 개선을 이루었습니다.