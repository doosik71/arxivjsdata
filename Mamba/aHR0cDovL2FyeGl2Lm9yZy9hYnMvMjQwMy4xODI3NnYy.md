# RankMamba: Benchmarking Mamba’s Document Ranking Performance in the Era of Transformers

Zhichao Xu(2024)

## 🧩 Problem to Solve

본 논문은 정보 검색(Information Retrieval, IR) 분야, 특히 문서 랭킹(Document Ranking) 작업에서 Transformer 아키텍처의 계산 효율성 문제를 해결하고, 이에 대한 대안으로 최근 주목받는 Mamba 모델의 효용성을 검증하고자 한다.

Transformer 구조는 자연어 처리(NLP)와 정보 검색 분야에서 표준으로 자리 잡았으나, 핵심 메커니즘인 Attention이 학습 시 $O(n^2)$의 시간 복잡도를 가지며, 추론 시에는 KV-cache 저장을 위해 $O(n \cdot d)$의 공간 복잡도가 요구된다는 치명적인 단점이 있다. 또한, 사전 학습된 길이보다 긴 컨텍스트로 확장하는 데 어려움이 있다.

문서 랭킹 작업은 쿼리와 문서 사이의 세밀한 상호작용을 캡처하고 긴 컨텍스트를 이해해야 하므로 Transformer의 Attention 메커니즘이 매우 유리한 구조이다. 따라서 본 연구의 목표는 State Space Model(SSM) 기반의 Mamba 구조가 이러한 Transformer 기반 언어 모델들과 비교하여 문서 랭킹 작업에서 대등한 성능을 낼 수 있는지, 그리고 실제 학습 효율성은 어떠한지를 벤치마킹하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Mamba 모델을 클래식한 IR 작업인 문서 랭킹에 적용하여 그 성능과 효율성을 체계적으로 분석했다는 점이다. 구체적인 설계 아이디어와 기여 사항은 다음과 같다.

1. **Mamba의 랭킹 성능 검증**: 다양한 크기와 구조(Encoder-only, Decoder-only, Encoder-Decoder)의 Transformer 모델들과 Mamba를 동일한 학습 레시피 하에서 비교하여, Mamba가 유사한 파라미터 규모의 Transformer 모델들과 경쟁 가능한 수준의 성능을 보임을 입증하였다.
2. **학습 처리량(Throughput) 분석**: 이론적인 선형 복잡도에도 불구하고, 현재의 구현체 수준에서 Mamba가 Flash Attention을 적용한 효율적인 Transformer 구현체보다 오히려 낮은 학습 처리량을 보인다는 사실을 밝혀내어 향후 최적화 방향을 제시하였다.
3. **재현성 제공**: 다양한 모델 설정과 학습된 체크포인트를 공개하여 Mamba 기반 IR 연구의 출발점을 제공하였다.

## 📎 Related Works

### 관련 연구 및 한계
- **Transformer 기반 모델**: BERT와 RoBERTa 같은 모델들이 문서 랭킹에서 뛰어난 성능을 보였으며, 긴 문서를 처리하기 위해 FirstP, MaxP, SumP 등의 전략이나 Longformer, Llama 같은 롱 컨텍스트 모델들이 제안되었다. 하지만 여전히 추론 시의 메모리 및 시간 복잡도 문제가 존재한다.
- **State Space Models (SSM)**: S4와 같은 초기 SSM은 선형 시간 불변성(Linear Time Invariance, LTI) 특성을 가져 효율적이지만, 입력 데이터에 따라 유연하게 정보를 압축하는 능력이 부족하여 모델의 효과가 제한적이었다.

### 차별점
기존 연구들이 주로 Transformer의 Attention 메커니즘을 개선(예: Flash Attention)하는 데 집중했다면, 본 논문은 Attention을 완전히 대체하는 구조인 Mamba(Selective SSM)를 문서 랭킹이라는 구체적인 IR 태스크에 적용하여 그 실효성을 직접적으로 비교했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. Mamba 및 Selective SSM의 구조
Mamba는 기존 SSM의 한계를 극복하기 위해 입력 값에 따라 파라미터가 변하는 **Selective Scan** 구조를 채택하였다.

- **기초 SSM (S4)**: 1차원 시퀀스 $x(t)$를 잠재 상태 $h(t)$를 통해 $y(t)$로 매핑하며, 다음과 같은 연속 방정식으로 정의된다.
  $$h'(t) = Ah(t) + Bx(t)$$
  $$y(t) = Ch(t)$$
  이를 이산화(Discretization)하면 다음과 같은 재귀 형태로 나타낼 수 있다.
  $$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
  $$y_t = Ch_t$$

- **Mamba (S6)**: 기존 S4의 파라미터 $(\Delta, B, C)$를 입력 데이터에 의존하도록 수정하여 '선택적(Selective)' 능력을 부여하였다. 이를 효율적으로 계산하기 위해 다음과 같은 **Hardware-aware optimization**을 적용한다.
    - **SRAM 활용**: GPU HBM 대신 빠른 SRAM에서 SSM 파라미터를 로드하여 계산 속도를 높인다.
    - **Parallel Scan**: 순차적인 재귀 계산을 병렬화하여 GPU 스레드에 분산시킨다.
    - **Recomputation**: 역전파 시 중간 상태를 저장하는 대신 다시 계산하여 메모리 사용량을 줄인다.

### 2. 문서 랭킹 작업 설정
- **태스크 정의**: 쿼리 $q$와 문서 $d$를 입력받아 두 텍스트 간의 관련성 점수 $s$를 예측하는 함수 $f(q, d) \to s \in \mathbb{R}$를 학습한다.
- **입력 포맷**:
    - **Encoder-only / Encoder-Decoder**: `[CLS]{q}[SEP]{d}[EOS]` 형식을 사용하며, `[CLS]` 토큰의 표현력을 사용하여 점수를 예측한다.
    - **Decoder-only / Mamba**: `document:{d}\n\nquery:{q}[EOS]` 형식을 사용한다. 쿼리를 뒤에 배치함으로써 쿼리 토큰이 이전의 문서 토큰들을 참조할 수 있게 하여 관련성을 더 잘 포착하도록 설계하였다. `[EOS]` 토큰의 표현력을 사용하여 점수를 예측한다.

### 3. 학습 목표 및 절차
- **손실 함수**: 모델은 **InfoNCE loss**를 사용하여 학습된다. 이는 정답 문서($d^+$)의 점수는 높이고, 샘플링된 부정 문서($d^-$)들의 점수는 낮추는 방향으로 최적화한다.
  $$\mathcal{L} = -\log \frac{\exp(f(q_i, d^+_i))}{\exp(f(q_i, d^+_i)) + \sum_{j \in D^-_i} \exp(f(q_i, d_j))}$$
- **학습 설정**: MS MARCO 데이터셋을 사용하며, BM25 리트리버를 통해 상위 100개 중 7개의 하드 네거티브(Hard Negatives)를 샘플링하여 학습 데이터를 구성하였다. Optimizer는 AdamW를 사용하였으며, 모델 크기가 700M를 초과하는 경우 메모리 절약을 위해 **LoRA (Low-Rank Adaptation)**를 적용하였다.

## 📊 Results

### 1. 정량적 성능 평가
실험 결과, Mamba 모델은 유사한 크기의 Transformer 모델들과 비교했을 때 매우 경쟁력 있는 성능을 보였다.

- **MS MARCO Dev 세트 (MRR)**: $\approx 110\text{M}$ 규모에서 `mamba-130m` (0.4089)은 `bert-base-uncased` (0.4126)에 근접한 성능을 보였으며, $\approx 330\text{M}$ 규모의 `mamba-370m` (0.4250)은 `roberta-large` (0.4334)에 근접하였다.
- **TREC DL19/DL20 (NDCG@10)**: Mamba 모델들이 특히 강세를 보였다. `mamba-130m`은 110M 규모 모델 중 DL19와 DL20에서 가장 높은 NDCG를 기록하였다. 또한 `mamba-370m` 역시 DL19에서 최고 성능을 달성하였다.

### 2. 학습 처리량 (Training Throughput)
성능과는 대조적으로, 학습 효율성 측면에서는 부정적인 결과가 나타났다.
- Flash Attention을 적용한 Transformer 모델(OPT, Pythia)이 가장 높은 처리량과 가장 낮은 공간 복잡도를 보였다.
- **Mamba는 이론적으로 $O(n)$의 복잡도를 가지지만, 실제 구현체에서의 학습 처리량은 Flash Attention 기반 Transformer보다 낮았으며 GPU 메모리 소비량은 더 높았다.**

## 🧠 Insights & Discussion

### 강점 및 성과
본 연구는 Mamba 구조가 단순히 언어 생성 작업뿐만 아니라, 쿼리와 문서 간의 복잡한 상호작용을 분석해야 하는 **문서 랭킹 작업에서도 충분히 유효함**을 입증하였다. 특히 unidirectional(단방향) 정보 흐름을 가짐에도 불구하고, 적절한 입력 포맷 설정을 통해 bidirectional(양방향) 모델들과 대등한 성능을 낼 수 있음을 확인하였다.

### 한계 및 비판적 해석
가장 큰 문제는 **'이론과 실제의 괴리'**이다. Mamba의 핵심 셀링 포인트는 Transformer의 quadratic complexity를 극복한 선형 시간 복잡도이지만, 실제 벤치마킹 결과에서는 Flash Attention이라는 고도로 최적화된 Transformer 구현체에 밀려 더 낮은 처리량을 기록하였다. 이는 Mamba의 하드웨어 최적화가 아직 Transformer의 성숙도에 미치지 못했음을 시사한다.

또한, 본 연구는 1B 파라미터 미만의 소형 모델들만을 대상으로 하였으므로, 모델 규모가 훨씬 커졌을 때 Mamba의 효율성이 Transformer를 압도하는 '임계점'이 어디인지에 대해서는 답하지 못하였다.

## 📌 TL;DR

본 논문은 Mamba 모델이 문서 랭킹 작업에서 Transformer 기반 모델들과 대등하거나 때로는 더 우수한 성능을 낼 수 있음을 실험적으로 증명하였다. 하지만 실제 학습 처리량은 Flash Attention을 적용한 Transformer보다 낮게 나타나, 이론적 효율성이 실제 하드웨어 성능으로 완전히 전이되지 않았음을 보여준다. 이 연구는 향후 SSM 기반의 효율적인 IR 모델 설계 및 최적화를 위한 중요한 벤치마크 기반을 제공한다.