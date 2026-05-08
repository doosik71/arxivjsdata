# Differential Mamba

Nadav Schneider, Itamar Zimerman, Eliya Nachmani (2025)

## 🧩 Problem to Solve

본 논문은 Transformer와 RNN 같은 시퀀스 모델들이 무관한 컨텍스트에 과도하게 주의를 할당하는 **Over-allocation** 문제에 집중한다. 이러한 현상은 중간 표현(intermediate representations)에 노이즈를 생성하며, 결과적으로 대규모 언어 모델(LLM)의 환각(hallucination)을 유발하고, 장거리 의존성 파악 및 정보 검색(retrieval) 능력을 저하시키며, 전반적인 강건성(robustness)을 약화시킨다.

특히, 저자들은 Mamba와 같은 Selective State-Space Model(S6) 기반 아키텍처가 Transformer보다 Over-allocation 문제에 더 취약할 수 있다고 가설을 세운다. 그 이유는 다음과 같다.

1. **Softmax-free 아키텍처**: Mamba는 무관한 어텐션 가중치를 억제하는 효과가 있는 Softmax의 지수적 스케일링 효과가 없다.
2. **상태 기반 모델의 국소성**: 상태 기반 모델로서 Mamba는 모든 중간 토큰을 고려해야만 먼 거리의 토큰을 처리할 수 있으며, 이 과정에서 중요한 토큰이 무관한 토큰들 사이에 분산되어 희석될 가능성이 크다.

따라서 본 연구의 목표는 Transformer에서 Over-allocation을 완화하기 위해 제안된 **Differential design**을 Mamba 아키텍처에 성공적으로 적용하여 모델의 강건성과 검색 능력을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba 아키텍처의 노이즈를 줄이기 위한 **Diff-Mamba** 메커니즘의 제안이다. 중심 아이디어는 두 개의 병렬적인 Mamba 경로를 생성하고, 그 출력값의 차이(difference)를 이용해 무관한 컨텍스트로 인한 노이즈를 제거하는 '차동 제거(differential denoising)' 방식을 도입하는 것이다.

단순히 S6 레이어 수준에서 차동 설계를 적용하는 것은 부족하며, 전체 Mamba 블록 수준에서 이 메커니즘을 적용하고 적절한 정규화(normalization)를 결합해야 실질적인 성능 향상이 가능함을 입증하였다. 이를 통해 모델의 일반적인 언어 모델링 능력뿐만 아니라 특히 장거리 컨텍스트에서의 정보 검색 능력을 획기적으로 개선하였다.

## 📎 Related Works

### Differential Transformer

Ye et al. [43]은 Transformer의 어텐션 헤드를 두 개로 나누고, 한 어텐션 맵에서 다른 맵을 빼는 방식을 통해 무관한 토큰에 대한 과도한 할당을 줄이는 Diff-Transformer를 제안하였다. 이는 노이즈 캔슬링 헤드폰의 원리와 유사하게 불필요한 신호를 상쇄시키는 방식이다.

### State-Space Layers 및 Mamba

S6(Selective State-Space) 레이어는 입력에 따라 시스템 행렬이 변하는 선택적 메커니즘을 통해 효율적인 시퀀스 모델링을 수행한다. 특히 최근 연구들은 Mamba가 암시적 어텐션(Implicit Attention)의 한 형태로 볼 수 있으며, 데이터 제어 선형 연산자(Data-controlled linear operator)로 해석될 수 있음을 보였다.

### 기존 접근 방식과의 차별점

기존의 Mamba 모델들이 주로 추론 효율성과 계산 복잡도($O(L)$) 개선에 집중했다면, Diff-Mamba는 아키텍처 구조적 변경을 통해 **표현의 품질(Representation Quality)**과 **강건성**을 개선하려는 시도라는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

Diff-Mamba는 기존 Mamba 블록을 확장하여 두 개의 경로를 구성하고 그 차이를 계산하는 구조를 가진다. 저자들은 먼저 S6 레이어에만 차동 설계를 적용한 `Diff-S6`를 시도했으나 성능이 저조함을 확인하였고, 최종적으로 전체 Mamba 블록에 적용한 `Diff-Mamba`를 제안하였다.

### 상세 방법론 및 방정식

**1. Diff-Mamba의 기본 원리**
Diff-Mamba는 두 개의 독립적인 Mamba 경로 $\text{Mamba}_1$과 $\text{Mamba}_2$를 통해 얻은 출력값의 차이를 계산한다.
$$\text{Diff-Mamba}(X) = \text{Mamba}_1(X) - \lambda \text{Mamba}_2(X)$$
여기서 $\lambda$는 학습 가능한 스칼라 값으로, 두 경로 간의 가중치를 조절하여 노이즈를 효과적으로 상쇄시킨다.

**2. 정규화된 Diff-Mamba (N-Diff-Mamba)**
S6는 Softmax와 달리 출력이 정규화되지 않고 범위가 제한되지 않으므로, 단순 뺄셈은 불안정할 수 있다. 이를 해결하기 위해 뺄셈 전후에 정규화(Normalization, $N$) 단계를 추가한다.
$$\text{N-Diff-Mamba}(x) = N(\text{Mamba}_1(x) - \lambda \text{Mamba}_2(x))$$
최종 출력은 $\lambda_{\text{init}}$을 이용하여 $(1 - \lambda_{\text{init}})$만큼 스케일링되어 최종적으로 출력된다.

**3. $\lambda$의 파라미터화**
$\lambda$는 양수 값을 유지하며 안정적으로 학습되도록 다음과 같이 설계된다.
$$\lambda = \text{Sigmoid}(\bar{\lambda}) + \lambda_{\text{init}}$$
여기서 $\bar{\lambda}$는 학습 가능한 파라미터이다.

### 효율적인 구현 (Efficient Implementation)

두 개의 Mamba 경로를 독립적으로 계산하면 추론 지연 시간이 두 배로 증가한다. 이를 방지하기 위해 다음과 같은 최적화를 적용하였다.

- **병렬 처리**: 단일 forward pass 내에서 입력을 복제하여 두 경로를 동시에 계산한다.
- **파라미터 수 유지**: 일반적인 Mamba 블록의 채널 확장(channel expansion) 대신, 입력 표현을 채널 차원에서 복제함으로써 모델의 전체 파라미터 수와 메모리 사용량을 기존 Mamba와 유사한 수준으로 유지하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: WikiText-103, Text8, Enwik8 (언어 모델링), BABILong (검색 능력), The Pile (중규모 사전 학습).
- **지표**: Perplexity (PPL), Bits per byte (bpb), Retrieval Accuracy (Ratio).
- **비교 대상**: Vanilla Mamba.

### 주요 결과

**1. 언어 모델링 성능**
모든 벤치마크에서 Diff-Mamba가 Mamba보다 낮은 PPL을 기록하며 우수한 성능을 보였다. 특히 레이어 수가 많아질수록(6층 $\rightarrow$ 12층) Diff-Mamba의 개선 폭이 더 커졌는데, 이는 상위 레이어일수록 복잡하고 긴 의존성을 처리해야 하므로 Over-allocation 문제의 영향이 더 크기 때문으로 분석된다.

**2. 정보 검색 능력 (Retrieval)**
BABILong 벤치마크에서 Diff-Mamba는 특히 컨텍스트 길이가 길어질수록 Mamba 대비 월등한 성능 향상을 보였다. Zero-shot 및 Fine-tuned 설정 모두에서 Diff-Mamba가 Mamba보다 더 느리게 성능이 저하되는 경향을 보였으며, 이는 장거리 컨텍스트에서 유의미한 정보를 더 잘 추출함을 의미한다.

**3. 중간 표현 노이즈 분석**
Tuned-lens 도구를 사용하여 각 레이어의 활성화 값을 예측 확률로 변환해 분석한 결과, Diff-Mamba의 중간 표현에서 '바늘 토큰(needle token)'을 예측할 확률(Signal-to-Noise Ratio, SNR)이 Mamba보다 훨씬 높게 나타났다. 특히 초기 레이어에서 이 차이가 극명하게 나타나, 차동 설계가 실제로 노이즈를 효과적으로 제거하고 있음을 정량적으로 입증하였다.

**4. 중규모 모델 실험**
370M 파라미터 모델을 The Pile 데이터셋으로 사전 학습시킨 결과, **Mamba 레이어와 Diff-Mamba 레이어를 교차로 배치(Alternating)**했을 때 가장 높은 성능을 보였다. 이는 두 구조의 장점을 결합하여 강건성과 효율성을 동시에 잡을 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 Over-allocation 문제가 Transformer뿐만 아니라 Mamba와 같은 SSM 기반 모델에서도 공통적으로 발생하는 일반적인 문제임을 밝혔으며, 이를 차동 설계(Differential Design)를 통해 해결할 수 있음을 보여주었다.

**강점 및 해석:**

- **SNR의 향상**: Tuned-lens 분석을 통해 차동 설계가 단순한 성능 향상을 넘어, 실제로 중간 표현의 신호 대 잡음비(SNR)를 높인다는 점을 증명한 것이 매우 인상적이다.
- **실용적 구현**: 추론 효율성 저하 문제를 병렬 구현과 입력 복제 전략으로 해결하여, 실제 적용 가능성을 높였다.

**한계 및 미해결 질문:**

- **이론적 근거 부족**: 저자들도 언급했듯이, 왜 차동 설계가 성능을 향상시키는지에 대한 엄격한 수학적/이론적 프레임워크는 아직 제시되지 않았다.
- **규모의 제한**: 학술적 예산 한계로 인해 소형 및 중형 모델 수준에서만 실험이 이루어졌다. 초대형 LLM 스케일에서도 동일한 효과가 나타날지는 추가 검증이 필요하다.
- **도메인 확장성**: NLP 외에 컴퓨터 비전이나 시계열 분석 등 다른 도메인에서도 동일한 메커니즘이 작동할지는 미지수이다.

## 📌 TL;DR

본 논문은 Mamba 모델의 고질적인 문제인 **Over-allocation(무관한 정보에 과하게 집중하는 현상)**을 해결하기 위해, 두 경로의 출력 차이를 이용하는 **Diff-Mamba** 아키텍처를 제안한다. 실험 결과, Diff-Mamba는 일반적인 언어 모델링 성능을 높일 뿐만 아니라, 특히 **장거리 컨텍스트에서의 정보 검색(Retrieval) 능력을 획기적으로 개선**하며 중간 표현의 노이즈를 유의미하게 감소시킨다. 이는 향후 더 강건하고 효율적인 Recurrent LLM을 설계하는 데 중요한 설계 지침을 제공한다.
