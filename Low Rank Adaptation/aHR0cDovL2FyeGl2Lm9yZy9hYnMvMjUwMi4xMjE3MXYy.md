# GoRA: Gradient-driven Adaptive Low Rank Adaptation

Haonan He, Peng Ye, Yuchen Ren, Yuan Yuan, Luyang Zhou, Shucun Ju, Lei Chen (2025)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)의 효율적인 미세 조정 방법인 Low-Rank Adaptation(LoRA)의 성능을 결정짓는 두 가지 핵심 요소인 **rank 선택(rank selection)**과 **가중치 초기화(weight initialization)** 문제를 해결하고자 한다.

기존의 LoRA 변형 방법들은 다음와 같은 한계점을 가지고 있다.

- **Adaptive Rank 할당의 한계**: AdaLoRA와 같은 방법은 중요도에 따라 rank를 조정하지만, 마스킹 메커니즘을 위해 더 큰 파라미터 공간을 미리 할당해야 하므로 학습 파라미터 수가 증가하고 rank의 상한선이 제한되는 문제가 있다.
- **초기화 전략의 한계**: PiSSA나 LoRA-GA와 같은 비제로(nonzero) 초기화 방법들은 사전 학습된 가중치 $W_0$를 직접 수정(manipulation)해야 한다. 이는 학습과 추론 사이에 간극(training-inference gap)을 발생시키며, 체크포인트 저장 용량을 증가시키거나 다중 어댑터 서비스 시 유연성을 떨어뜨리는 결과를 초래한다.

따라서 본 논문의 목표는 사용성과 계산 효율성을 저해하지 않으면서, 사전 학습된 가중치의 수정 없이 rank 할당과 초기화 전략을 동시에 최적화하는 통합 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 학습된 가중치의 **그레이디언트(gradient) 정보**를 활용하여 가중치의 중요도를 평가하고, 이를 바탕으로 rank를 적응적으로 할당하며, 최적의 초기값으로 어댑터를 설정하는 것이다.

구체적인 기여 사항은 다음과 같다.

1. **Gradient-driven Adaptive Rank Allocation**: 학습 전 소량의 데이터로 계산된 그레이디언트를 통해 가중치의 중요도를 측정하고, 정해진 파라미터 예산 내에서 각 레이어에 최적의 rank를 동적으로 할당한다.
2. **Pseudo-inverse 기반의 최적 초기화**: LoRA 어댑터를 '그레이디언트 압축기(gradient compressor)'로 정의하고, Moore-Penrose pseudo-inverse를 사용하여 초기 단계에서 어댑터의 출력이 실제 그레이디언트와 가장 유사하도록 $B$ 행렬을 초기화한다.
3. **통합 프레임워크**: rank 할당과 초기화를 하나의 흐름으로 통합하여, 사전 학습 가중치를 수정하지 않고도 Full Fine-tuning에 근접하거나 이를 능가하는 성능을 달성하였다.

## 📎 Related Works

### LoRA의 Rank 관련 연구

기존 연구들은 rank를 높이면 성능이 향상된다는 점에 주목하였다. 일부 연구는 학습 중 저차원 서브스페이스를 병합하거나(ReLoRA), 여러 미니 어댑터를 대각선으로 배치(MeLoRA)하여 rank를 높이려 했으나, 이는 학습 과정의 복잡도를 높이거나 구조적 변경을 필요로 하여 사용성이 떨어진다. AdaLoRA는 중요도에 따라 rank를 동적으로 조정하지만, 앞서 언급한 것처럼 파라미터 오버헤드가 발생한다.

### LoRA의 초기화 관련 연구

Vanilla LoRA는 $A$를 정규분포로, $B$를 0으로 초기화하여 초기 상태의 모델 출력을 유지한다. 이를 개선하기 위해 SVD 기반의 PiSSA, MiLoRA나 그레이디언트의 특이값을 사용하는 LoRA-GA 등이 제안되었다. 그러나 이러한 비제로 초기화 방식들은 학습 시작 시 $W_0$에서 초기화 값을 빼주는 조작이 필요하며, 이는 추론 시 재계산 비용을 발생시키거나 저장 공간 효율성을 저해하는 원인이 된다.

## 🛠️ Methodology

### 1. LoRA 어댑터의 재해석: Gradient Compressor

본 논문은 LoRA 어댑터가 학습 과정에서 그레이디언트를 압축하고 축적하는 역할을 한다고 가설을 세운다. 특히 $A$ 행렬이 고정된 경우, 가중치 업데이트 $\Delta W$는 다음과 같이 표현된다.
$$\Delta W = \frac{\alpha}{r} A_0 \Delta B = -\eta \frac{\alpha}{r} \sum_{t=0}^{T} A_0 A_0^T \frac{\partial L_t}{\partial W_t}$$
여기서 $A_0 A_0^T$는 그레이디언트를 저차원으로 투영하는 압축기로 작동한다.

### 2. 적응적 Rank 할당 전략 (Adaptive Rank Allocation)

학습 시작 전, $n$-단계 누적 그레이디언트 $G = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial L_i}{\partial W}$를 계산하여 가중치의 중요도를 측정한다.

- **중요도 측정**: 가중치 $W$와 그레이디언트 $G$의 요소별 곱(element-wise product)의 평균을 사용한다.
$$I(W) = \text{avg}(|W \odot G|)$$
- **정규화 및 이점 계산**: 전체 레이어의 중요도 합 대비 개별 레이어의 비중을 계산하여 advantage $A_i$를 구한다.
$$A_i = \frac{I(W_i)}{\sum_{j=1}^{N} I(W_j)}$$
- **Rank 할당**: 기준 rank $r_{\text{ref}}$에 기반한 총 파라미터 예산 $B$를 설정하고, 각 레이어의 rank $r_i$를 다음과 같이 결정한다.
$$r_i = \left[ \frac{B \times A_i}{\sqrt{m+n}} \right], \quad \text{s.t. } r_{\min} \le r_i \le r_{\max}$$
이 과정을 통해 전체 파라미터 수는 vanilla LoRA와 유사하게 유지하면서 중요도가 높은 레이어에 더 많은 rank를 배분한다.

### 3. 적응적 초기화 전략 (Adaptive Initialization)

rank가 할당된 후, 초기 어댑터의 계산 결과 $AB$가 누적 그레이디언트 $G$를 최적으로 근사하도록 초기화한다.

- **최적의 $B$ 계산**: $A$가 랜덤하게 초기화되었을 때, Frobenius norm $\|G - AB\|_F$를 최소화하는 $B$는 다음과 같다.
$$B = -(A^T A)^{-1} A^T G$$
- **스케일링 적용**: 초기 어댑터의 출력이 확률적 경사 하강법(SGD)의 한 단계와 유사하도록 스케일링 팩터 $\xi$를 도입한다.
$$\xi = \gamma \cdot \frac{\sqrt{m}}{\alpha}$$
최종적으로 $B$는 $\xi \cdot B$로 설정되며, 이를 통해 학습 초기 단계에서 안정적인 최적화 기반을 마련한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 모델**:
  - NLU: T5-Base (GLUE benchmark)
  - 생성: Llama-3.1-8B, Llama-2-7B (GSM8K, HumanEval, MTBench)
  - 이미지 분류: CLIP-ViT-B/16 (Stanford-Cars 등 7개 데이터셋)
- **비교 대상**: Full Fine-tuning, LoRA, RSLoRA, DoRA, LoRA+, PiSSA, LoRA-GA, AdaLoRA 등.
- **지표**: Accuracy, PASS@1, GPT-4 기반 평가 점수 등.

### 주요 결과

1. **NLU 작업**: GLUE 벤치마크에서 평균 87.96점을 기록하며, 모든 저차원 어댑터 방법론은 물론 Full Fine-tuning(87.91)보다 높은 성능을 보였다.
2. **생성 작업**:
    - Llama-3.1-8B 모델의 GSM8K에서 72.91점을 기록하여 LoRA-GA(71.39)를 1.52점 상회하였다.
    - HumanEval에서는 48.98점으로 RSLoRA(45.78) 대비 3.20점 높은 성능을 보였다.
    - 특히 $r_{\text{ref}}=128$ 설정 시, GSM8K(75.74)와 HumanEval(52.03)에서 Full Fine-tuning의 성능을 뛰어넘었다.
3. **이미지 분류**: 7개 데이터셋 평균 89.47%의 정확도를 달성하여 Full Fine-tuning(88.06) 및 LoRA-Pro(89.20)보다 우수한 성능을 입증하였다.
4. **효율성**: Peak GPU 메모리와 학습 시간 측면에서 vanilla LoRA와 거의 동일한 수준을 유지하였다. 반면 AdaLoRA는 학습 시간이 유의미하게 더 길었다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **Rank 분포의 유효성**: 분석 결과, 대부분의 rank가 $wv$ (value) 레이어에 할당되고 $wq$ (query) 레이어에는 적게 할당되는 경향이 나타났다. 이는 이전 연구들의 발견과 일치하며, 본 논문의 적응적 할당 전략이 유효함을 시사한다.
- **초기화의 영향**: 스케일링 팩터 $\gamma$가 성능에 큰 영향을 미쳤으며, 적절한 $\gamma$ 설정이 초기 최적화 경로를 안정화하여 빠른 수렴과 더 낮은 최종 손실(loss)을 가능하게 했다.
- **중요도 지표**: Nuclear norm 기반의 지표보다 파라미터 민감도 기반 지표 $\text{avg}(|W \odot G|)$가 rank 할당 시 더 뛰어난 성능을 보였다.

### 한계 및 비판적 해석

- **모델 규모의 제한**: 실험이 주로 8B 이하의 모델에서 진행되었다. 70B 이상의 초거대 모델에서도 동일한 효율성과 성능 향상이 나타날지는 추가 검증이 필요하다.
- **초기화의 단순성**: $A$ 행렬을 여전히 랜덤하게 초기화하고 있다. $A$ 또한 사전 학습된 가중치의 특징을 반영하여 초기화한다면 더 높은 성능을 낼 가능성이 있다.
- **그레이디언트 계산 비용**: 학습 전 $n$-step의 그레이디언트를 미리 계산해야 하는 오버헤드가 존재한다. 비록 본 논문에서 이 시간이 매우 짧다고 주장하지만, 데이터셋 규모가 극도로 커질 경우 이에 대한 영향 분석이 필요하다.

## 📌 TL;DR

GoRA는 그레이디언트 정보를 활용하여 **LoRA의 rank 할당과 초기화를 동시에 최적화**하는 통합 프레임워크이다. 사전 학습된 가중치를 수정하지 않고도 중요도 기반의 rank 배분과 pseudo-inverse 기반의 초기화를 수행함으로써, **학습-추론 간극을 제거하고 계산 효율성을 유지**하면서도 Full Fine-tuning에 필적하거나 이를 능가하는 성능을 달성하였다. 이 연구는 향후 다양한 모달리티와 더 거대한 규모의 모델 효율적 튜닝에 중요한 기준점을 제시할 것으로 기대된다.
