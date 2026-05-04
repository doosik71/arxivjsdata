# TransMamba: Flexibly Switching between Transformer and Mamba

Yixing Li, Ruobing Xie, Zhen Yang, Xingwu Sun, Shuaipeng Li, Weidong Han, Zhanhui Kang, Yu Cheng, Chengzhong Xu, Di Wang, Jie Jiang (2025)

## 🧩 Problem to Solve

본 논문은 현대 대규모 언어 모델(LLM)의 핵심인 Transformer와 최근 효율적인 대안으로 떠오른 Mamba(State Space Model, SSM) 사이의 트레이드오프 문제를 해결하고자 한다. 

Transformer는 뛰어난 문맥 학습 능력과 다중 작업 일반화 성능을 보이지만, 시퀀스 길이에 대해 이차 복잡도($O(T^2)$)의 계산 비용이 발생하여 긴 시퀀스 처리 시 효율성이 급격히 떨어진다. 반면, Mamba는 선형 복잡도($O(T)$)를 가져 효율적이지만, 문맥 학습의 안정성과 일반화 성능 면에서 Transformer보다 불안정한 모습을 보인다. 

기존의 하이브리드 모델들은 단순히 두 구조를 층별로 교차 배치하는 정적인 구조를 가졌으며, 이는 층의 순서나 비율 등에 제약이 있어 최적의 성능을 내기 어렵고 구조적 유연성이 부족하다는 한계가 있다. 따라서 본 논문의 목표는 Transformer의 효과성과 Mamba의 효율성을 동시에 확보하면서, 시퀀스 길이와 레이어에 따라 두 메커니즘을 유연하게 전환할 수 있는 통합 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Transformer의 Attention 메커니즘과 Mamba의 SSM 메커니즘 사이의 수학적 유사성(Consistency)에 기반하여, **파라미터 행렬을 공유하는 통합 구조**를 설계하는 것이다.

1.  **파라미터 공유(Shared Parameters):** Attention의 $QKV$ 행렬과 SSM의 $CBx$ 행렬을 동일한 파라미터 세트로 공유함으로써, 추가적인 파라미터 증가 없이 두 모드 간의 전환을 가능하게 하였다.
2.  **Memory Converter 설계:** Transformer 모드에서 생성된 정보를 Mamba 모드가 이해할 수 있는 상태(State)로 손실 없이 변환해주는 Memory Converter를 제안하여, 전환 지점(TransPoint)에서 정보의 흐름이 끊기지 않도록 보장하였다.
3.  **유연한 TransPoint 스케줄링:** 각 레이어와 토큰 길이에 따라 Attention에서 SSM으로 전환되는 지점인 TransPoint를 최적으로 배치하는 전략을 탐구하여 학습 효율과 성능의 균형을 맞추었다.

## 📎 Related Works

기존 연구들은 Transformer의 KV 캐시 압박과 긴 시퀀스 처리의 한계를 극복하기 위해 다양한 선형 어텐션이나 SSM 기반 모델을 제시하였다. 최근 Mamba2는 Transformer의 Attention과 SSM의 듀얼 폼(Dual Form)이 수학적으로 일관성이 있음을 밝혔으며, 일부 연구에서는 Transformer의 $QKV$ 가중치를 Mamba의 $CBx$로 증류(Distillation)하여 전이 학습이 가능함을 입증하였다.

하지만 기존의 하이브리드 모델(예: Jamba 등)은 단순히 두 모듈을 직렬로 연결하는 방식에 그쳤다. 본 논문은 단순히 층을 섞는 것을 넘어, **시퀀스 레벨(Sequence Level)**에서 동일한 파라미터를 사용해 동적으로 메커니즘을 전환한다는 점에서 기존 접근 방식과 명확히 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인
TransMamba는 Decoder-only 자기회귀 모델 구조를 가지며, 각 레이어는 Transformer와 Mamba의 기능을 모두 포함한다. 특정 토큰 위치인 $\text{TransPoint}$를 기준으로, 그 이전의 토큰들은 Attention 메커니즘으로 처리하고, 그 이후의 토큰들은 SSM 메커니즘으로 처리한다.

### 주요 구성 요소 및 작동 원리
1.  **파라미터 공유:**
    본 모델은 $W_{QKV}$와 $W_{CBx}$를 공유한다. 구체적으로는 $Q \leftrightarrow C$, $K \leftrightarrow B$, $V \leftrightarrow x$의 대응 관계를 가진다.

2.  **계산 절차:**
    -   **$\text{TransPoint}$ 이전 ($\text{Attention Mode}$):** 
        입력 $\mathbf{h}^s$에 대해 $Q, K, V$를 계산하고 표준 Attention을 통해 출력 $\mathbf{y}^s$를 생성한다.
        $$\mathbf{y}^s = \text{softmax}(QK^T)V$$
    -   **$\text{TransPoint}$ 이후 ($\text{SSM Mode}$):**
        $\text{Memory Converter}$를 통해 초기 상태 $\mathbf{h}_0$를 얻은 후, SSM 메커니즘을 통해 출력 $\mathbf{y}^l$을 생성한다.
        $$\mathbf{y}^l = (A^\times \circ CB^T)(\Delta \circ \mathbf{x})$$

3.  **Lossless Memory Converter:**
    Attention의 중간 결과물인 $K$와 $V$를 SSM의 은닉 상태 $\mathbf{h}$로 변환하는 핵심 모듈이다. SSM의 수학적 구조를 행렬 형태로 확장하면 다음과 같이 표현된다.
    $$\mathbf{h} = (A^\times \circ B^T)(\Delta \circ \mathbf{x})$$
    이를 기반으로 Attention의 $K, V$를 이용해 추정된 은닉 상태 $\mathbf{h}^s$를 다음과 같이 계산한다.
    $$\mathbf{h}^s = (A^\times \circ K^T)V$$
    최종적으로 $\text{TransPoint}$에서의 초기 상태 $\mathbf{h}_0$는 $\mathbf{h}^s$의 마지막 값($\mathbf{h}^s[-1]$)으로 설정된다. 이 과정은 추가 파라미터 없이 이론적인 수식만으로 수행되므로 정보 손실이 없다.

4.  **TransPoint Scheduling:**
    -   **레이어별 차등 배치:** 모든 레이어가 동시에 전환되면 성능이 저하되므로, 레이어마다 서로 다른 $\text{TransPoint}$를 설정한다.
    -   **분포 전략:** $\text{TransPoint}$를 시퀀스 전체에 걸쳐 로그 형태(예: 0, 128, ..., 8192)로 넓고 세밀하게 분포시켜 구조적 급변을 방지하고 부드러운 전환을 유도한다. 8개 레이어를 주기로 이 패턴을 반복한다.

## 📊 Results

### 실험 설정
-   **모델 크기:** 400M, 1.5B 파라미터.
-   **비교 대상:** Transformer, Mamba2, Hybrid 모델.
-   **데이터셋:** 중국어 및 영어 혼합 데이터셋으로 83B 토큰 학습.
-   **평가 지표:** ARC-E/C, CoQA, OBQA, PIQA, PhoneBook, BoolQ (정확도 및 F1-score), LongBench-v2 (긴 문맥 이해).

### 주요 결과
1.  **종합 성능:** TransMamba는 대부분의 일반 작업에서 Baseline보다 우수한 성능을 보였다. 특히 1.5B 모델의 경우 여러 지표에서 Hybrid 모델보다 높은 성능을 기록하였다.
2.  **정밀 검색 및 긴 문맥 성능:** 
    -   **PhoneBook task:** Mamba 기반 모델들이 취약한 정밀 검색 작업에서 Transformer와 거의 동일한 높은 정확도를 보였다. 이는 시퀀스 초반부에 Attention을 사용하여 정보를 정확히 캡처했기 때문이다.
    -   **LongBench-v2:** 1.5B 모델 기준, TransMamba(38.76)가 Hybrid(35.79) 및 Transformer(31.61)보다 높은 점수를 기록하여 $\text{Memory Converter}$의 정보 보존 능력을 입증하였다.
3.  **학습 효율성:** 
    -   이론적 FLOPs 분석 결과, Transformer 대비 최대 25%의 학습 시간 단축 효과가 확인되었다. 
    -   실제 측정에서도 Transformer보다 빠른 학습 속도를 보였으며, 이는 $\text{TransPoint}$ 설정에 따른 이차 함수적 경향성을 띤다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문은 Transformer와 Mamba가 단순히 상호 보완적인 관계를 넘어, 파라미터 수준에서 통합될 수 있음을 실험적으로 증명하였다. 특히 $\text{Memory Converter}$를 통해 두 메커니즘 사이의 정보 전이를 수학적으로 해결함으로써, 구조적 유연성과 성능을 동시에 잡았다는 점이 매우 높게 평가된다.

### 한계 및 논의 사항
1.  **엔지니어링 최적화:** 이론적인 FLOPs 감소폭에 비해 실제 런타임 개선 폭이 적은데, 이는 Attention과 SSM 각각에 최적화된 CUDA 커널의 가속 성능 차이 때문이다. 향후 통합 커널 최적화가 필요하다.
2.  **추론 전략의 유연성:** 학습 시에는 효율적인 $\text{TransPoint}$ 스케줄을 사용하고, 추론 시에는 작업의 특성에 따라 다른 스케줄(예: 순수 Transformer 모드)을 적용해도 정상 작동하며 때로는 더 좋은 성능을 낸다는 점이 발견되었다. 이는 구조적 디커플링의 가능성을 시사한다.
3.  **Scaling Law:** 본 연구는 1.5B 규모까지 검증되었으나, 더 거대한 모델에서도 동일한 효율성과 성능 이득이 유지되는지에 대한 추가 연구가 필요하다.

## 📌 TL;DR

TransMamba는 Transformer의 $QKV$와 Mamba의 $CBx$ 파라미터를 공유하여, 시퀀스 길이에 따라 Attention과 SSM 모드를 유연하게 전환하는 통합 프레임워크이다. $\text{Memory Converter}$를 통해 두 모드 간의 정보 손실 없는 전환을 구현하였으며, 최적화된 $\text{TransPoint}$ 스케줄링을 통해 학습 효율성을 높이면서도 긴 문맥 처리 성능과 정밀 검색 능력을 모두 확보하였다. 이 연구는 차세대 시퀀스 모델링을 위한 확장 가능한 구조적 대안을 제시한다.