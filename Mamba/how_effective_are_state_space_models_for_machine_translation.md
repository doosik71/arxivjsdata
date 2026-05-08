# How Effective are State Space Models for Machine Translation?

Hugo Pitorro, Pavlo Vasylenko, Marcos Treviso, André F. T. Martins (2024)

## 🧩 Problem to Solve

본 논문은 기계 번역(Machine Translation, MT) 분야에서 Transformer 아키텍처가 가진 고유한 한계를 해결하고, 이를 대체할 수 있는 효율적인 대안으로서 State Space Models(SSMs) 및 Linear Recurrent 모델들의 효용성을 검증하고자 한다.

Transformer의 핵심인 Attention 메커니즘은 입력 시퀀스 길이 $n$에 대해 $O(n^2)$의 계산 복잡도를 가지며, 이는 긴 문맥(long context)을 처리해야 하는 상황에서 메모리 및 계산 비용의 급격한 증가를 초래한다. 또한, 학습 시 경험하지 못한 긴 시퀀스에 대한 일반화(length generalization) 능력이 떨어진다는 문제점이 있다.

최근 SSMs와 같은 선형 재귀 모델들이 언어 모델링(Language Modeling) 작업에서 효율성과 성능을 입증하였으나, 기계 번역 작업에서 Transformer와 경쟁할 만큼 성능이 우수한지는 여전히 불분명하다. 따라서 본 연구의 목표는 Transformer, RetNet, Mamba 및 이들의 하이브리드 모델들을 문장 및 단락 수준의 번역 작업에서 정밀하게 비교 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **포괄적인 벤치마크 수행**: Transformer, RetNet, Mamba뿐만 아니라, Mamba에 Attention 메커니즘을 결합한 하이브리드 모델들의 성능을 문장 및 단락 수준 데이터셋에서 엄밀하게 비교하였다.
2. **하이브리드 아키텍처 제안**: SSM의 효율성과 Attention의 정밀한 문맥 파악 능력을 결합한 Mamba-MHA, Mamba-Local, Mamba Enc-Dec 구조를 실험하여, 소수의 Attention 레이어 추가만으로도 번역 품질과 고유 명사 회상(Recall) 능력이 크게 향상됨을 보였다.
3. **시퀀스 길이 일반화 분석**: 단락 수준 번역에서 모델들이 학습 데이터의 길이 분포에 매우 민감함을 발견하였으며, 문장 결합(Concatenation) 전략을 통해 학습 분포를 긴 시퀀스 쪽으로 이동시킴으로써 성능 저하를 완화할 수 있음을 입증하였다.
4. **추론 효율성 검증**: Mamba 모델이 Transformer(Pythia) 대비 메모리 사용량은 3~5배 낮고, 추론 속도는 최대 2배가량 빠름을 정량적으로 확인하였다.

## 📎 Related Works

기존 연구인 Vardasbi et al. (2023)은 S4(Structured State Spaces) 기반 모델이 MT 작업에서 Transformer보다 성능이 떨어진다는 결과를 제시하였다. 이는 SSM이 입력에 독립적인(input-independent) 파라미터를 사용하여 상태를 업데이트하므로, 번역과 같이 정밀한 정보 유지와 선택적 집중이 필요한 작업에 취약하기 때문으로 해석된다.

본 논문은 이러한 한계를 극복하기 위해, 파라미터가 입력 데이터에 따라 동적으로 변하는 선택 메커니즘(Selection Mechanism)이 도입된 최신 모델인 Mamba와 RetNet을 분석 대상으로 삼았다. 또한, 최근 언어 모델링 연구에서 SSM과 Attention을 섞어 사용하는 하이브리드 접근 방식이 효과적이라는 점에 착안하여, 이를 MT 도메인에 적용하고 그 효용성을 검증하였다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 분석 대상 모델 구조

#### (1) Transformer

표준적인 Attention 메커니즘을 사용하며, 쿼리 $Q$, 키 $K$, 값 $V$에 대해 다음과 같이 정의된다.
$$Y = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$
본 논문에서는 기본 Encoder-Decoder 구조와 최신 LLaMA 아키텍처를 반영한 Decoder-only 구조(Transformer++)를 베이스라인으로 사용한다.

#### (2) Linear Attention (RetNet)

Softmax를 제거하고 유사도 함수를 커널 형태로 재구성하여 $O(n)$의 복잡도를 달성한다. 특히 RetNet은 지수적 감쇠 마스크 $\gamma$를 도입하여 다음과 같이 재귀적으로 상태를 업데이트한다.
$$S_i = \gamma S_{i-1} + k_i v_i^\top, \quad y_i = S_i^\top q_i$$

#### (3) State Space Models (Mamba)

Mamba는 SSM의 파라미터를 입력 $x$에 의존하게 만드는 선택 메커니즘을 도입하였다. 상태 $H_i$의 업데이트 식은 다음과 같다.
$$H_i = A_i \odot H_{i-1} + B_i \odot X_i, \quad y_i = H_i^\top c_i$$
여기서 $A_i, B_i, c_i$는 입력에 따라 결정되는 데이터 의존적 파라미터이며, $\odot$은 하다마르 곱(Hadamard product)을 의미한다.

#### (4) Hybrid Models

- **Mamba-MHA**: Mamba 레이어 중 일부(네트워크 중간과 출력단)를 Multi-Head Attention으로 대체한다.
- **Mamba-Local**: 전체 Attention 대신 슬라이딩 윈도우(Sliding Window) 기반의 Local Attention을 사용하여 효율성을 유지한다.
- **Mamba Enc-Dec**: Transformer의 self-attention 부분을 Mamba 블록으로 교체하고, cross-attention은 그대로 유지한다.

### 2. 학습 및 실험 절차

- **데이터셋**: 문장 수준의 WMT14, WMT16과 단락 수준의 WMT23 데이터셋을 사용한다.
- **학습 전략**:
  - **From Scratch**: 모든 모델을 처음부터 학습시켜 아키텍처 자체의 성능을 비교한다.
  - **Finetuning**: The Pile 데이터셋으로 사전 학습된 Pythia(Transformer)와 Mamba 모델을 미세 조정하여 비교한다.
- **단락 수준 데이터 증강 (WMT23-CAT-N)**: 긴 문맥 처리를 위해 무작위로 $N$개의 문장을 결합하여 학습 데이터의 시퀀스 길이를 인위적으로 늘린 데이터셋을 구축한다.
- **평가 지표**: BLEU와 더불어 인간의 판단과 상관관계가 높은 COMET 지표를 주 지표로 사용한다.

## 📊 Results

### 1. 문장 수준 번역 (Sentence-level)

- **Mamba의 경쟁력**: 처음부터 학습시킨 경우, Mamba는 Decoder-only Transformer++뿐만 아니라 Encoder-Decoder Transformer와 비교해서도 대등하거나 더 우수한 성능을 보였다.
- **하이브리드의 우위**: Mamba-MHA(Attention 2개 레이어 추가)가 거의 모든 언어 쌍에서 Transformer와 Mamba 모두를 능가하는 최고 성능을 기록하였다.
- **사전 학습의 효과**: 사전 학습된 모델(Pythia, Mamba)을 미세 조정했을 때, 처음부터 학습시킨 모델보다 COMET 점수가 약 4~8점 높게 나타났다.

### 2. 고유 명사 회상 능력 (Recall of Named Entities)

- Transformer와 Mamba는 고유 명사 회상 능력에서 비슷하게 우수한 성능을 보인 반면, RetNet은 현저히 낮은 성능을 보였다.
- Mamba-MHA와 Mamba Enc-Dec 같은 하이브리드 모델이 가장 높은 회상 능력을 보였으며, 이는 Attention 메커니즘이 특정 토큰에 정밀하게 집중하는 능력을 보완해주었기 때문으로 분석된다.

### 3. 단락 수준 번역 (Paragraph-level)

- **길이 민감도**: 모든 모델은 학습 시 경험하지 못한 긴 시퀀스에서 성능이 하락하는 경향을 보였다.
- **결합 전략의 효과**: 문장을 결합한 CAT-5, CAT-10 데이터셋으로 학습했을 때, 긴 입력에 대한 번역 품질이 크게 향상되었으며, 특히 Mamba Enc-Dec과 Mamba-MHA의 성능 향상이 두드러졌다.
- **외삽(Extrapolation) 능력**: 학습 데이터보다 훨씬 긴 시퀀스(최대 2048 토큰)를 처리하는 실험에서, 사전 학습된 Mamba-M이 Pythia-M보다 훨씬 강건(robust)한 성능을 보였으며, 시퀀스가 길어질수록 그 격차가 최대 20점(COMET)까지 벌어졌다.

### 4. 추론 비용 (Inference Cost)

- **메모리 및 속도**: Mamba-M은 Pythia-M 대비 메모리 사용량이 3~5배 적었으며, 생성 토큰 길이가 1024일 때 추론 속도가 약 2배 더 빨랐다(Mamba-M: ~806 tokens/s vs Pythia-M: ~405 tokens/s).

## 🧠 Insights & Discussion

본 연구는 SSM, 특히 Mamba가 기계 번역 작업에서 Transformer의 실질적인 대안이 될 수 있음을 입증하였다. Mamba의 성공 요인은 데이터 의존적인 상태 업데이트 메커니즘에 있으며, 이는 이전의 S4 모델이 가졌던 한계를 극복하고 필요한 정보를 더 정밀하게 유지할 수 있게 한다.

특히 주목할 점은 **'Attention과 SSM의 상호보완성'**이다. Mamba에 아주 적은 수의 Attention 레이어를 추가하는 것만으로도 번역 품질과 고유 명사 회상 능력이 향상되었다는 점은, SSM이 전반적인 문맥을 효율적으로 압축하고 Attention이 국소적인 정밀도를 보완하는 구조가 최적임을 시사한다.

다만, 여전히 시퀀스 길이의 분포가 성능에 큰 영향을 미친다는 점은 한계로 남는다. 비록 Mamba가 Transformer보다 길이 외삽 능력이 우수하다고는 하나, 학습 데이터의 길이 분포를 조정하는 전략(Concatenation)이 필수적이라는 사실은 SSM만으로 모든 길이 문제를 해결할 수 없음을 보여준다.

## 📌 TL;DR

본 논문은 기계 번역에서 Mamba(SSM)가 Transformer와 경쟁 가능한 수준의 성능을 가지며, 특히 추론 속도와 메모리 효율성 면에서 압도적임을 보였다. 특히 Mamba에 소수의 Attention 레이어를 결합한 하이브리드 모델이 가장 뛰어난 성능을 기록하였다. 또한, 긴 문맥 번역 시 학습 데이터의 길이 분포를 조정하는 것이 필수적이며, Mamba가 Transformer보다 긴 시퀀스에 대한 외삽 능력이 더 뛰어남을 확인하였다. 이 연구는 향후 효율적인 초장문 번역 모델 설계에 있어 SSM과 Attention의 하이브리드 구조가 핵심적인 방향이 될 것임을 시사한다.
