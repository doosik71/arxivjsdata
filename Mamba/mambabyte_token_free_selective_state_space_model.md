# MambaByte: Token-free Selective State Space Model

Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, Alexander M. Rush (2024)

## 🧩 Problem to Solve

본 연구는 언어 모델링에서 널리 사용되는 **subword tokenization(서브워드 토큰화)**이 갖는 한계를 해결하고자 한다. 서브워드 토큰화는 학습 효율성과 미등록 단어(out-of-vocabulary) 처리 사이의 절충안이지만, 오타, 철자 및 대소문자 변형, 형태론적 변화에 취약하여 견고성(robustness)이 떨어진다는 문제가 있다.

이를 해결하기 위해 원본 바이트(raw bytes)를 직접 학습하는 **token-free language model**이 대안으로 제시되어 왔다. 그러나 바이트 단위로 모델링을 수행하면 시퀀스 길이가 서브워드 대비 대폭 증가하게 된다. 기존의 autoregressive Transformer는 어텐션 연산의 복잡도가 시퀀스 길이에 대해 이차적으로 증가($O(L^2)$)하므로, 매우 긴 바이트 시퀀스를 효율적으로 처리하는 데 한계가 있다.

따라서 본 논문의 목표는 **Mamba State Space Model(SSM)**을 활용하여, 토큰화 없이도 효율적이고 성능이 뛰어나며, 특히 긴 바이트 시퀀스를 처리할 수 있는 token-free 언어 모델인 **MambaByte**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Mamba SSM의 고정된 크기의 메모리 상태(fixed-sized memory state)**를 활용하는 것이다.

1. **토큰 없는 효율적 모델링**: Transformer와 달리 Mamba는 컨텍스트 길이에 독립적인 고정 크기 메모리 상태를 유지한다. 이를 통해 별도의 표현 압축(representational compression)이나 패칭(patching) 없이도 매우 긴 바이트 시퀀스를 선형 복잡도($O(L)$)로 처리할 수 있다.
2. **Speculative Decoding 도입**: 바이트 단위 생성은 한 번에 한 바이트씩 생성해야 하므로 추론 속도가 느리다. 이를 해결하기 위해 작은 서브워드 모델이 초안을 작성(drafting)하고, 큰 MambaByte 모델이 이를 검증(verification) 및 수정하는 **speculative decoding** 기법을 제안하여 추론 효율성을 극대화하였다.
3. **견고성 및 외삽 성능 입증**: MambaByte가 기존의 subword 모델보다 노이즈(오타, 대소문자 변형 등)에 훨씬 더 강하며, 학습 시보다 훨씬 긴 시퀀스에 대해 성능 저하 없이 대응하는 length extrapolation 능력이 뛰어남을 보였다.

## 📎 Related Works

- **Token-free Language Models**: Byte-Pair Encoding(BPE)이나 WordPiece 같은 토큰화 방식의 한계를 극복하기 위해 캐릭터 단위나 바이트 단위 모델링(예: MegaByte, ByT5)이 연구되었다. 특히 MegaByte는 바이트를 패치 단위로 묶어 Transformer의 연산량을 줄였으나, 본 논문은 이러한 압축 없이 SSM을 통해 직접 해결하는 방식을 취한다.
- **Attention-free Models**: S4, Gated SSM 등은 효율적인 시퀀스 모델링을 가능하게 했으며, 최신 Mamba 아키텍처는 입력에 따라 상태를 선택하는 **selective mechanism**을 도입하여 Transformer에 근접한 성능과 선형 복잡도를 동시에 달성하였다.
- **Speculative Decoding**: 작은 모델이 생성한 후보를 큰 모델이 병렬로 검증하여 속도를 높이는 기법이다. 기존에는 동일한 토큰화 체계를 사용하는 모델 간에 적용되었으나, 본 연구는 **서브워드 모델(drafter)과 바이트 모델(verifier)**이라는 서로 다른 토큰화 체계를 결합한 새로운 방식을 제안한다.

## 🛠️ Methodology

### 1. Selective SSM 및 Mamba 아키텍처

MambaByte는 기본적으로 Mamba SSM 아키텍처를 따른다. SSM은 연속 시간 시스템의 미분 방정식을 이산화하여 시퀀스를 모델링한다.

**연속 시간 시스템:**
$$\frac{dh(t)}{dt} = Ah(t) + B(t)x(t), \quad y(t) = C(t)h(t)$$
여기서 $h(t) \in \mathbb{R}^n$은 hidden state, $x(t)$는 입력, $y(t)$는 출력이다.

**이산화된 시스템 (Discrete-time):**
위 식을 이산화하면 다음과 같은 재귀 형태로 나타낼 수 있다.
$$h[k] = A[k]h[k-1] + B[k]x[k], \quad y[k] = C[k]h[k]$$

Mamba의 핵심은 $B, C, \Delta$가 입력 $x[k]$에 따라 변하는 **선택적(selective)** 함수라는 점이다. 예를 들어, 타임스텝 $\Delta[k]$는 다음과 같이 정의된다.
$$\Delta[k] = \text{softplus}(W_\Delta(W_R x[k]))$$
이러한 선택 메커니즘 덕분에 모델은 입력 내용에 따라 어떤 정보를 유지하고 버릴지를 결정할 수 있으며, 이는 언어 모델링과 같은 복잡한 작업에 필수적이다.

### 2. MambaByte의 바이트 수준 모델링

MambaByte는 별도의 토큰화 과정 없이 raw bytes를 직접 입력으로 받는다. Mamba의 hidden state 메모리 크기는 입력 시퀀스 길이 $L_{ctx}$와 무관하게 고정되어 있으므로, 바이트 단위로 인해 시퀀스가 길어져도 메모리 요구량이 급증하지 않는다. 또한, 학습 시에는 **Parallel Scan** 알고리즘을 사용하여 $O(nL)$의 시간 복잡도로 효율적인 병렬 학습을 수행한다.

### 3. Subword Drafting을 통한 Speculative Decoding

바이트 단위의 순차적 생성 속도 문제를 해결하기 위해 다음과 같은 파이프라인을 사용한다.

1. **Drafting (초안 작성)**: 상대적으로 크기가 작은 서브워드 기반 Mamba 모델($M_{subword}$)이 $m$개의 서브워드를 autoregressively 생성한다.
2. **Verification (검증)**: 생성된 서브워드를 바이트로 변환하여 MambaByte 모델($M_{byte}$)에 입력한다. $M_{byte}$는 **Parallel Scan**을 통해 이 바이트들이 자신의 상위 $\beta$개 후보군에 속하는지 한 번에 검증한다.
3. **Correction (수정)**: 검증 결과 오류가 발생한 지점(bifurcation position $c$)을 찾는다. $c$ 이후의 바이트들은 폐기하고, $M_{byte}$가 직접 autoregressively 수정 바이트를 생성하며, 경계 바이트(예: 공백)가 나올 때까지 이 과정을 반복한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: PG19, Stories, Books, ArXiv, Code 등 다양한 롱폼 텍스트 데이터셋을 사용하였다.
- **비교 대상**: Transformer, MegaByte(patch size 4, 8), PerceiverAR, S4D 등.
- **측정 지표**: Bits per byte (BPB) 및 Word-level Perplexity (PPL)를 사용하여 성능을 측정하였다.

### 2. 주요 결과

- **언어 모델링 성능**: MambaByte-353M은 동일한 연산량(FLOPs)을 가진 MegaByte-758M+262M보다 모든 데이터셋에서 일관되게 더 낮은 BPB를 기록하며 우수한 성능을 보였다.
- **대규모 모델 성능**: MambaByte-972M은 기존의 바이트 레벨 모델들을 압도했으며, 최신 서브워드 기반 모델들과 경쟁 가능한 수준의 Word-level PPL을 달성하였다.
- **길이 외삽 (Length Extrapolation)**: MambaByte는 학습 시 사용한 시퀀스 길이($8,192$ bytes)보다 훨씬 긴 시퀀스(최대 $64\times$)에 대해서도 성능 저하 없이 대응하는 능력을 보였으며, 이는 Transformer 계열 모델들이 위치 임베딩의 한계로 인해 보여주지 못한 결과이다.
- **노이즈 견고성**: Antspeak, Character swap 등 인위적인 노이즈 주입 실험에서, 서브워드 모델(Mamba-1.03B)은 PPL이 급격히 상승하며 무너진 반면, MambaByte는 매우 안정적인 성능을 유지하였다.
- **추론 효율성**: MambaByte는 recurrent generation 특성상 MegaByte보다 $2.6\times$ 더 빠른 생성 속도를 보였으며, 제안한 speculative decoding을 적용했을 때 서브워드 기반 Mamba 모델과 유사한 수준의 추론 속도에 도달하였다.

## 🧠 Insights & Discussion

본 논문은 SSM이 token-free 언어 모델링을 실현하는 데 있어 매우 강력한 도구가 될 수 있음을 입증하였다. 특히 Transformer의 이차 복잡도 문제를 해결하면서도, 기존 바이트 모델들이 사용했던 '패칭(patching)'과 같은 표현 압축 없이도 고성능을 낼 수 있다는 점이 중요하다.

**강점**:

- **유연성**: 토큰화의 제약이 없으므로 오타나 특이한 표기법에 매우 강하며, 이는 실제 환경의 노이즈가 많은 데이터 처리에서 큰 이점이 된다.
- **효율성**: 학습 시의 선형 복잡도와 추론 시의 고정 메모리 비용은 매우 긴 문맥을 처리해야 하는 작업에서 Transformer보다 압도적인 이점을 제공한다.

**한계 및 논의**:

- **추론 속도**: SSM의 효율성에도 불구하고 바이트 단위 생성은 본질적으로 생성 단계가 매우 많다. 본 논문은 speculative decoding으로 이를 완화했지만, 이는 별도의 드래프트 모델이 필요하다는 추가적인 오버헤드를 수반한다.
- **데이터 효율성**: 서브워드 모델에 비해 동일한 정보량을 학습시키기 위해 더 많은 바이트 시퀀스를 처리해야 하므로, 학습 데이터의 물리적 길이가 매우 길어진다는 점이 고려되어야 한다.

## 📌 TL;DR

MambaByte는 서브워드 토큰화 없이 raw bytes를 직접 처리하는 Mamba 기반의 언어 모델이다. 고정 크기의 메모리 상태를 활용해 매우 긴 시퀀스를 선형 복잡도로 효율적으로 처리하며, 서브워드 기반 모델보다 노이즈에 강하고 길이 외삽 능력이 뛰어나다. 특히, 서브워드 모델을 활용한 speculative decoding 기법을 통해 바이트 모델의 고질적인 추론 속도 문제를 해결함으로써, token-free 모델이 실제 대규모 언어 모델의 실용적인 대안이 될 수 있음을 증명하였다.
