# BlackMamba: Mixture of Experts for State-Space Models

Quentin Anthony, Yury Tokpanov, Paolo Glorioso, Beren Millidge (2024)

## 🧩 Problem to Solve

본 논문은 현대 대규모 언어 모델(LLM)의 주류인 Decoder-only Transformer 아키텍처가 가진 계산 효율성 문제를 해결하고자 한다. Transformer의 핵심인 Attention 메커니즘은 시퀀스 길이($L$)에 대해 연산량(FLOPs)과 메모리 요구량이 제곱 비례하는 $O(L^2)$의 시간 및 공간 복잡도를 가진다. 이는 문맥 길이(Context length)를 제한하고, 자동 회귀 생성(Autoregressive generation) 시 비용을 급격히 증가시키며, 결과적으로 무제한적인 시퀀스 처리나 연속적인 데이터 스트림 학습을 저해하는 병목 현상이 된다.

이를 해결하기 위해 선형 복잡도를 가진 State-Space Models(SSMs)와 파라미터 효율성을 극대화한 Mixture of Experts(MoE)라는 두 가지 대안이 제시되어 왔으나, 이 두 가지 이점을 동시에 결합하여 대규모로 학습시키고 검증한 연구는 부족했다. 따라서 본 논문의 목표는 Mamba SSM의 선형 복잡도와 MoE의 저비용 추론 능력을 결합한 **BlackMamba** 아키텍처를 제안하고, 이를 통해 기존 Transformer 및 Mamba 베이스라인 대비 뛰어난 성능과 효율성을 달성하는 것이다.

## ✨ Key Contributions

BlackMamba의 핵심 아이디어는 **'Attention-free Mamba 블록'과 'Routed MLP(MoE) 블록'을 교차 배치**하여, SSM의 효율적인 시퀀스 처리 능력과 MoE의 파라미터 희소성(Sparsity)을 동시에 확보하는 것이다.

구체적인 기여 사항은 다음과 같다:

1. **하이브리드 아키텍처 설계**: Mamba SSM과 MoE를 결합하여 추론 지연 시간(Latency)과 학습 FLOPs를 획기적으로 줄인 BlackMamba를 구현하였다.
2. **대규모 모델 학습 및 공개**: 300B 토큰의 커스텀 데이터셋을 사용하여 340M/1.5B 및 630M/2.8B (Forward-pass 파라미터/전체 파라미터) 규모의 모델을 학습시키고 가중치와 코드를 공개하였다.
3. **Sinkhorn 라우팅 최적화**: MoE의 전문가(Expert) 간 부하 균형을 맞추기 위한 Sinkhorn 알고리즘에 새로운 초기화 방식을 도입하여, 수렴 속도를 대폭 향상시켜 라우팅 속도를 개선하였다.
4. **효율성 입증**: 동일한 추론 비용(Forward-pass 파라미터)과 학습 FLOPs 조건에서 Dense Transformer 및 Dense Mamba 모델보다 우수한 성능을 보임을 정량적으로 증명하였다.

## 📎 Related Works

### State-Space Models (SSMs)

SSM은 선형 동역학 시스템에서 영감을 받아 시퀀스 길이에 대해 $O(L)$의 복잡도를 가진다. 초기 SSM은 Transformer보다 표현력이 부족했으나, 최근의 **Mamba**는 $A, B, C$ 행렬을 입력값에 의존하게 만드는 'Selective scan' 메커니즘을 도입하여 Transformer에 근접한 성능을 구현하였다.

### Mixture of Experts (MoE)

MoE는 모든 파라미터를 활성화하는 대신, 라우터(Router)를 통해 입력 토큰을 특정 전문가(Expert) MLP로 보내는 방식이다. 이를 통해 전체 파라미터 수는 늘리면서도 실제 연산에 참여하는 파라미터 수는 낮게 유지하여, 추론 비용을 낮추면서도 모델의 용량을 키울 수 있다. 하지만 학습 불안정성과 전문가 간 부하 불균형(Load-balancing) 문제가 주요 한계로 지적된다.

### 차별점

기존 연구에서도 Mamba-MoE의 조합이 시도된 적이 있으나(예: Moe-Mamba), 이는 수억 개 수준의 매우 작은 모델과 적은 데이터량(<10B 토큰)으로 진행되었다. 반면, BlackMamba는 수십억 개 파라미터 규모의 모델을 300B 토큰이라는 의미 있는 규모의 데이터로 학습시켜 실제 확장 가능성(Scaling potential)을 입증했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 아키텍처

BlackMamba는 기존 Transformer의 Attention 블록을 Mamba SSM 레이어로, MLP 블록을 MoE 레이어로 대체한 구조이다. 단일 블록의 연산 과정은 다음과 같은 방정식으로 표현된다:
$$x^{l+1} = x^l + \text{MoE}(\text{LN}(x^l + \text{mamba}(\text{LN}(x^l))))$$
여기서 $\text{LN}$은 LayerNorm을 의미하며, Mamba 블록과 MoE 블록이 순차적으로 배치되는 Sequential 구조를 채택하였다.

### Mamba SSM 블록

Mamba 블록은 연속 시간 시스템의 상태 방정식 $\frac{dh}{dt} = Ah + Bx, y = Ch$를 이산화하여 사용한다. 핵심은 $A, B, C$ 행렬이 입력 $x$에 따라 동적으로 변하는 함수라는 점이다.

- **입력 처리**: 입력 $x$는 먼저 선형 투영 및 1D 컨볼루션을 거친다.
- **Selective Scan**: 입력 의존적인 $B, C, \Delta t$를 계산하고, 이를 통해 상태 $h$를 업데이트한다.
- **출력**: 최종 출력 $y$는 학습 가능한 바이어스 $D$와 게이팅 벡터 $z$를 통해 조절된 후 다시 잔차 연결(Residual stream)에 더해진다.

### MoE 블록 및 라우팅

MoE 블록은 8개의 전문가 MLP 중 상위 1개(Top-1)를 선택하여 연산한다.

- **전문가 구조**: 각 전문가는 SwiGLU 활성화 함수를 사용하는 표준 MLP 구조이다.
- **개선된 Sinkhorn 라우팅**: 전문가 간 부하 균형을 맞추기 위해 Sinkhorn 알고리즘을 사용한다. 본 논문은 초기 조건 $d^{(0)}_i = 1$ 및 $d^{(1)}_\alpha = \frac{S}{N} \sum_i e^{L_{i\alpha}}$를 설정하는 새로운 초기화 방법을 제안하였다. 이를 통해 기존에 10~20회 필요했던 반복 횟수를 단 1회로 줄여 라우팅 속도를 획기적으로 개선하였다.

### 학습 설정

- **데이터셋**: The Pile, SlimPajama, Starcoder, PeS2o, ProofPile 등을 혼합한 1.8T 토큰 규모의 커스텀 데이터셋을 구축하였으며, 총 300B 토큰을 학습에 사용하였다.
- **정밀도 및 프레임워크**: bf16 정밀도를 사용하였으며, Megatron-LM 분산 학습 프레임워크를 이용하였다.

## 📊 Results

### 성능 평가 (Zero-shot)

HellaSwag, PIQA, WinoGrande 등 8가지 벤치마크에서 평가한 결과, BlackMamba는 동일한 **Forward-pass 파라미터** 규모의 Dense 모델(Pythia, OPT 등) 및 Dense Mamba 모델보다 우수한 평균 점수를 기록하였다.

- **BlackMamba 340M/1.5B**: 추론 시 활성화되는 파라미터는 342M이지만, 성능은 더 큰 규모의 Dense Mamba(343M)나 Pythia(410M)와 경쟁하거나 상회한다.
- **BlackMamba 630M/2.8B**: 역시 동일한 추론 비용의 모델들보다 뛰어난 성능을 보였으며, 특히 추론 FLOPs 대비 효율성이 극대화됨을 확인하였다.

### 추론 지연 시간 및 복잡도

- **시간 복잡도**: Transformer는 시퀀스 길이에 따라 지연 시간이 선형적으로 증가하지만, 이는 KV 캐시로 인한 메모리 증가를 동반한다. 반면 BlackMamba는 **상수 메모리 풋프린트**를 유지하면서 선형적인 시간 복잡도로 텍스트를 생성한다.
- **상대적 속도**: BlackMamba는 Dense Transformer, Transformer-MoE, 심지어 Dense Mamba보다도 빠른 생성 속도를 보였으며, 시퀀스 길이가 길어질수록 그 이점이 더욱 커진다.

### 전문가 활용 분석

학습 과정에서 전문가들의 토큰 할당 분포를 분석한 결과, 대부분의 레이어에서는 Sinkhorn 알고리즘 덕분에 균형 잡힌 분포를 보였다. 그러나 모델의 **최종 레이어(340M 모델의 경우 20층 이후)**에서는 전문가 간 불균형이 발생하는 현상이 관찰되었는데, 이는 후반 레이어의 전문화(Specialization) 또는 수치적 불안정성 때문으로 추측된다.

## 🧠 Insights & Discussion

### 강점 및 가능성

BlackMamba는 SSM의 선형 복잡도와 MoE의 파라미터 효율성을 성공적으로 결합하였다. 특히, 추론 비용을 고정한 채 모델의 전체 용량을 늘릴 수 있는 MoE의 특성과, KV 캐시 없이 긴 문맥을 처리할 수 있는 SSM의 특성이 시너지를 일으켜 **'매우 빠르고 효율적이면서도 강력한'** 언어 모델의 가능성을 보여주었다.

### 한계 및 미해결 과제

1. **평가 범위의 제한**: Zero-shot 평가에 집중되어 있으며, Many-shot In-context Learning 성능이나 사실 관계 정확성(Factual accuracy), 독성(Toxicity) 등에 대한 검증은 이루어지지 않았다.
2. **데이터 오염 가능성**: 사용된 데이터셋(RedPajama 등)에 평가 데이터가 포함되었을 가능성이 있으며, 이에 대한 명시적인 중복 제거 작업이 완벽하지 않았다.
3. **하이퍼파라미터 최적화**: 학습률(Learning rate) 외의 다른 하이퍼파라미터 튜닝이 충분히 이루어지지 않아 최적의 성능에 도달하지 못했을 가능성이 있다.
4. **정량화 및 미세 조정**: SSM-MoE 구조에 대한 효율적인 양자화(Quantization) 방법이나 RLHF를 통한 정렬(Alignment) 파이프라인은 아직 탐구되지 않은 영역이다.

## 📌 TL;DR

BlackMamba는 **Mamba SSM의 선형 시간/메모리 복잡도**와 **MoE의 효율적인 추론 비용**을 결합한 새로운 하이브리드 아키텍처이다. 본 연구는 300B 토큰으로 학습된 수십억 개 파라미터 모델을 통해, 동일한 추론 비용 및 학습 FLOPs 조건에서 기존 Dense Transformer 및 Mamba 모델보다 뛰어난 성능과 압도적으로 빠른 생성 속도를 가짐을 입증하였다. 이는 특히 매우 긴 시퀀스를 효율적으로 처리해야 하는 LLM의 실무 적용 및 향후 연구에 있어 매우 중요한 아키텍처적 방향성을 제시한다.
