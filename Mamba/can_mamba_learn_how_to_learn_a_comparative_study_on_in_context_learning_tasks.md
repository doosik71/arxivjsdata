# Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks

Jongho Park, Jaeseung Park, Zheyang Xiong, Nayoung Lee, Jaewoong Cho, Samet Oymak, Kangwook Lee, Dimitris Papailiopoulos (2024)

## 🧩 Problem to Solve

현대의 거대 언어 모델(LLM)은 파라미터의 추가적인 최적화 없이도 몇 가지 예시만으로 새로운 작업을 수행하는 In-Context Learning(ICL) 능력을 보여준다. 이러한 능력은 주로 Transformer 구조의 핵심인 Attention 메커니즘과 밀접하게 연관되어 연구되어 왔으며, 모델의 규모가 커짐에 따라 창발적 특성으로 나타난다.

최근 Transformer의 이차 복잡도(quadratic cost) 문제를 해결하기 위해 Mamba와 같은 State-Space Models(SSMs)가 대안으로 제시되었고, 언어 모델링 성능 면에서 경쟁력을 입증하였다. 그러나 ICL 능력은 일반적으로 수십억 개의 파라미터 규모에서 나타나는데, SSMs에 대한 연구는 상대적으로 소규모 모델에 집중되어 있어 SSMs가 Transformer만큼의 ICL 능력을 갖추었는지, 그리고 어떤 작업에서 강점과 약점을 보이는지에 대해서는 충분히 탐구되지 않았다. 따라서 본 논문의 목표는 Mamba를 포함한 SSMs의 ICL 성능을 Transformer와 다각도로 비교 분석하고, 두 구조의 장점을 결합한 하이브리드 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 다음과 같다.

1. **SSMs의 ICL 능력 검증**: 다양한 ICL 벤치마크 작업을 통해 Mamba가 Transformer와 대등하거나, 특정 작업(예: Sparse Parity)에서는 오히려 능가하는 ICL 능력을 갖추고 있음을 입증하였다.
2. **아키텍처별 강점 및 한계 규명**: Mamba는 순차적 계산 능력 덕분에 Sparse Parity나 특정 Outlier 제거 작업에 강점이 있지만, Decision Tree 학습이나 Vector MQAR와 같은 정보 검색(Retrieval) 작업에서는 Transformer보다 성능이 떨어진다는 점을 밝혀냈다.
3. **MambaFormer 제안**: Mamba의 효율적인 순차 처리 능력과 Transformer의 강력한 검색 능력을 결합한 하이브리드 아키텍처인 MambaFormer를 제안하였다. 특히, 초기 레이어에 Mamba 블록을 배치함으로써 Positional Encoding 없이도 뛰어난 ICL 성능을 달성하였다.

## 📎 Related Works

**Transformer 기반의 ICL**
기존 연구들은 Attention 메커니즘이 ICL을 가능케 하는 핵심이라고 주장하며, 이것이 내부적으로 경사 하강법(Gradient Descent)과 같은 최적화 알고리즘을 모방하거나 'Induction Heads'를 통해 정보를 검색하는 방식으로 동작한다고 설명한다.

**Sub-quadratic 아키텍처**
Transformer의 연산 비용을 줄이기 위해 S4, H3, 그리고 최신 Mamba와 같은 SSMs가 제안되었다. S4는 선형 시불변(LTI) 시스템을 기반으로 하며, Mamba는 여기에 입력 의존적인 선택 메커니즘(Selection Mechanism)을 추가하여 입력값에 따라 정보를 선택적으로 유지하거나 버릴 수 있게 하였다. 하지만 이전 연구(Arora et al., 2023)에서는 이러한 sub-quadratic 모델들이 다중 쿼리 리콜(Multi-query recall) 작업에서 Attention보다 뒤처진다는 점이 지적된 바 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 학습 절차

본 연구는 모델을 특정 함수 클래스 $\mathcal{F}$를 학습하도록 처음부터 훈련시킨다. 학습 과정은 다음과 같다.

1. 함수 분포 $\mathcal{D}_F$에서 함수 $f$를 선택하고, 입력 분포 $\mathcal{D}_X$에서 무작위 입력 $\{x_1, \dots, x_N\}$을 샘플링한다.
2. 프롬프트 $P = (x_1, f(x_1), \dots, x_N, f(x_N))$를 생성한다.
3. 모델 $f_\theta$는 다음과 같은 기대 손실 함수를 최소화하도록 학습된다.
$$\min_{\theta} \mathbb{E}_P \left[ \frac{1}{N} \sum_{i=1}^{N-1} \ell(f_\theta(P_i), f(x_i)) \right]$$
여기서 $P_i = (x_1, f(x_1), \dots, x_i, f(x_i), x_{i+1})$이며, 모델은 주어진 예시들을 바탕으로 $x_{i+1}$에 대한 결과값 $f(x_{i+1})$을 예측해야 한다.

### 평가 작업 (ICL Tasks)

- **Regression**: Linear, Sparse Linear, 2NN regression 등을 통해 수치 예측 능력을 평가한다.
- **Learning with Outliers**: 데이터에 무작위 노이즈나 더미 벡터를 섞어, 유용한 정보만 필터링하여 학습하는 능력을 측정한다.
- **Discrete Functions**: Sparse Parity 작업을 통해 이진 값의 곱셈 조합을 학습하는 능력을 평가한다.
- **Chain-of-Thought (CoT) I/O**: 중간 단계의 은닉 특징(hidden features)을 함께 제공했을 때의 성능을 측정한다.
- **Retrieval (Vector MQAR)**: 입력된 키-값(key-value) 쌍 중에서 쿼리에 맞는 값을 정확히 찾아내는 능력을 평가한다.

### MambaFormer 아키텍처

MambaFormer는 Transformer와 Mamba의 장점을 결합한 하이브리드 모델이다.

- **구조적 특징**: Transformer의 MLP 블록을 Mamba 블록으로 대체하고, MHA(Multi-Head Attention) 블록과 교차 배치한다.
- **Positional Encoding 제거**: 아키텍처의 최상단(첫 번째 레이어)에 Mamba 블록을 배치한다. Mamba의 순차적 특성이 자연스럽게 위치 정보를 인코딩하므로, 별도의 Positional Encoding이 필요 없다.

## 📊 Results

### 정량적 성능 분석

실험 결과, 각 모델의 작업별 성능은 다음과 같이 요약된다 (표 1 참조).

- **Mamba**: 표준 회귀(Regression) 작업에서는 Transformer와 대등하며, 특히 **Sparse Parity** 작업에서는 Transformer가 무작위 추측 수준에 머무는 반면 Mamba는 매우 높은 정확도를 보였다. 그러나 **Vector MQAR(검색)** 작업에서는 매우 낮은 성능을 보였다.
- **Transformer**: 검색 작업과 Decision Tree 학습에서는 Mamba보다 우수하지만, Sparse Parity 작업은 수행하지 못했다.
- **MambaFormer**: 모든 평가 작업에서 Transformer와 Mamba의 성능을 모두 달성하거나 능가하였다. 특히 Mamba가 실패한 검색 작업과 Transformer가 실패한 Parity 작업을 동시에 해결하는 'Best-of-both-worlds' 성능을 보였다.

### 하이브리드 모델의 효율성

MambaFormer는 특히 Outlier-robust regression 작업에서 적은 학습 횟수($10^{17}$ FLOPs 미만)만으로도 훨씬 더 많은 연산량을 사용한 모델들과 대등한 성능을 보였다. 또한, 합성 언어 벤치마크인 RegBench에서 Transformer와 Mamba보다 훨씬 빠르게 수렴하며 높은 정확도를 기록하였다.

## 🧠 Insights & Discussion

### 모델별 특성 분석

1. **순차적 계산의 이점**: Mamba가 Sparse Parity 작업에서 압도적인 성능을 보인 것은 recurrent한 구조가 순차적 논리 연산을 처리하는 데 유리하기 때문으로 분석된다.
2. **Context Compression의 한계**: Mamba가 검색(Retrieval) 작업에서 고전하는 이유는 SSM이 모든 컨텍스트를 고정된 크기의 은닉 상태(hidden state)로 압축하기 때문이다. 반면 Transformer는 Attention을 통해 모든 토큰에 직접 접근하므로 정보 손실 없이 검색이 가능하다.
3. **하이브리드의 시너지**: MambaFormer의 성공은 Mamba가 초기 단계에서 위치 정보와 기초적인 순차 특징을 추출하고, 이후 Attention 레이어가 정밀한 정보 검색을 수행하는 상호 보완적 구조 덕분이다. 특히 첫 레이어를 Mamba로 설정하는 것이 Parity 학습 효율성을 높이는 핵심 요소임을 확인하였다.

### 한계점 및 향후 과제

본 연구는 주로 비언어적(non-language) 합성 작업과 소규모 모델에 집중하였다. 실제 거대 언어 모델 규모에서 이러한 아키텍처적 특성이 동일하게 유지될지는 추가적인 검증이 필요하다. 또한, 어떤 구체적인 아키텍처 요소가 ICL의 각 세부 능력(검색 vs 추론)을 결정짓는지에 대한 이론적 규명이 더 필요하다.

## 📌 TL;DR

본 논문은 Mamba와 같은 SSMs가 Transformer 못지않은 In-Context Learning(ICL) 능력을 갖추고 있음을 실험적으로 입증하였다. Mamba는 특히 **Sparse Parity**와 같은 순차적 추론 작업에서 Transformer를 능가하지만, **정보 검색(Retrieval)** 작업에서는 취약함을 보였다. 이를 해결하기 위해 제안된 **MambaFormer**는 Mamba와 Attention 레이어를 전략적으로 배치하여 두 구조의 장점을 모두 취했으며, 결과적으로 모든 ICL 벤치마크에서 가장 우수한 성능을 기록하였다. 이 결과는 향후 효율적인 LLM 설계를 위해 하이브리드 아키텍처가 매우 유망한 방향임을 시사한다.
