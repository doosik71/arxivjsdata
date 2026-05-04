# LSR-Adapt: Ultra-Efficient Parameter Tuning with Matrix Low Separation Rank Kernel Adaptation

Xin Li, Anand D. Sarwate (2025)

## 🧩 Problem to Solve

최근 거대 언어 모델(Large Language Models, LLMs)을 다양한 다운스트림 작업에 적응시키기 위해 파라미터 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 시스템을 설계하는 것이 핵심적인 패러다임으로 자리 잡았다. 그중에서도 가중치 행렬에 저차원 구조(low-rank structure)를 가정하는 Low-Rank Adaptation (LoRA) 방식이 대표적이다.

그러나 현대의 거대 모델들의 규모가 기하급수적으로 커짐에 따라, 단순히 저차원 구조를 가정하는 것만으로는 학습 가능한 파라미터 수를 충분히 줄이면서 동시에 만족스러운 정확도를 유지하는 것이 점점 더 어려워지고 있다. 즉, 기존의 PEFT 방식들은 모델의 규모가 커질수록 파라미터 효율성과 성능 사이의 트레이드오프를 해결하는 데 한계가 있으며, 많은 구조적 설계가 이론적 근거보다는 경험적인 선택에 의존하고 있어 세밀한 성능 제어가 어렵다는 문제가 존재한다. 본 논문의 목표는 수치 해석(numerical analysis)의 행렬 분리 표현(separated representation) 개념을 도입하여, 기존 PEFT 방법론보다 훨씬 적은 파라미터로 더 높은 정확도를 달성하는 ultra-efficient한 커널을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 고차원 수치 해석에서 사용되는 Matrix Low-Separation-Rank (LSR) 표현식을 PEFT의 어댑터 행렬에 적용하는 것이다.

기존의 LoRA가 가중치 업데이트 행렬 $\Delta W$를 두 개의 저차원 행렬 $A$와 $B$의 곱으로 표현했다면, LSR-Adapt는 여기서 한 단계 더 나아가 행렬 $A$와 $B$ 자체를 Kronecker product($\otimes$)를 이용한 분리 표현식의 합으로 분해한다. 이러한 구조적 가정을 통해 학습해야 할 파라미터 수를 획기적으로 줄일 수 있으며, 수치 해석적 관점에서 근거가 명확한 이론적 토대를 제공함으로써 미세 조정 과정에 대한 더 정밀한 제어가 가능하도록 설계되었다.

## 📎 Related Works

PEFT 분야에서는 LoRA가 가중치 행렬의 저차원 구조를 활용하여 파라미터 효율성을 높인 선구적인 연구로 평가받는다. 이후 LoRA의 아이디어를 확장하여 낮은 정밀도의 양자화를 결합한 QLoRA나, 가중치 행렬의 의미론적 이해를 바탕으로 분해하는 DoRA와 같은 연구들이 진행되었다.

또한, 파라미터 수를 더욱 줄이기 위해 Kronecker product 기반의 팩토라이제이션(factorization)을 활용한 연구(예: KronA, KAdaptation)들이 제안되었다. 하지만 본 논문은 단순히 Kronecker product를 사용하는 것을 넘어, 수치 해석의 '분리 표현(Separated Representation)' 개념을 도입하여 다수의 Kronecker product 항들의 합으로 행렬을 근사함으로써, 기존의 Kronecker 기반 방식들보다 더 유연하고 효율적인 파라미터 표현이 가능함을 강조하며 차별점을 둔다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인

LSR-Adapt는 기본적으로 LoRA의 구조를 따르되, LoRA의 팩터 행렬인 $A$와 $B$를 LSR 커널로 대체한다. 대상 네트워크 층의 가중치 $W$에 대한 업데이트 식은 다음과 같다.
$$W' = W + \alpha \Delta W$$
여기서 $\Delta W$는 기존 LoRA에서는 $\Delta W \approx AB$로 표현되었으나, LSR-Adapt에서는 $A$와 $B$ 각각을 다음과 같이 LSR 표현식으로 근사한다.

### 주요 구성 요소 및 방정식

행렬 $A \in \mathbb{R}^{w_1 \times r}$와 $B \in \mathbb{R}^{r \times w_2}$에 대하여, 분리 랭크(separation rank) $s$를 도입하여 다음과 같이 정의한다.

$$A \approx \sum_{k=1}^{s} \lambda_{A,k} A_k^{(1)} \otimes A_k^{(2)}$$
$$B \approx \sum_{k=1}^{s} \lambda_{B,k} B_k^{(1)} \otimes B_k^{(2)}$$

여기서 $\otimes$는 Kronecker product를 의미하며, $A_k^{(1)}, A_k^{(2)}, B_k^{(1)}, B_k^{(2)}$는 크기가 매우 작은 커널 행렬들이다. 실제 구현에서는 스칼라 $\lambda$ 값을 최종 학습 계수 $\alpha$에 통합하여 단순화한다. 따라서 최종적인 가중치 업데이트 행렬 $\Delta W$는 다음과 같은 형태를 갖는다.

$$\Delta W \approx \left( \sum_{k=1}^{s} A_k^{(1)} \otimes A_k^{(2)} \right) \times \left( \sum_{k=1}^{s} B_k^{(1)} \otimes B_k^{(2)} \right)$$

이때 각 커널 행렬의 차원은 다음과 같은 제약 조건을 만족해야 한다.

- $A_k^{(1)} \in \mathbb{R}^{a_{k,1}^{(1)} \times a_{k,2}^{(1)}}$, $A_k^{(2)} \in \mathbb{R}^{a_{k,1}^{(2)} \times a_{k,2}^{(2)}}$
- $a_{k,1}^{(1)} \times a_{k,1}^{(2)} = w_1$ 및 $a_{k,2}^{(1)} \times a_{k,2}^{(2)} = r$
- $B$ 행렬에 대해서도 유사하게 $r \times w_2$ 차원을 만족하도록 구성된다.

### 학습 절차 및 이론적 근거

본 논문은 행렬 $M$을 $\sum \lambda_k M_k^{(1)} \otimes \dots \otimes M_k^{(r)}$ 형태로 근사할 수 있다는 수치 해석적 정의(Definition 3.2)를 바탕으로 한다. 특히, 근사 오차 $\epsilon$을 제어하기 위해 condition number $\gamma$를 정의하고, 기계 정밀도(machine round-off) $\mu$와의 관계를 통해 정밀도를 제어할 수 있는 이론적 기반을 제시한다.

## 📊 Results

### 실험 설정

- **모델 및 데이터셋**: RoBERTa 모델을 사용하여 GLUE 및 SuperGLUE 벤치마크에서 성능을 측정하였다.
- **비교 대상**: LoRA, KronA, KAdaptation.
- **하이퍼파라미터**:
  - LoRA: rank $r=8$, 파라미터 수 $2 \times 768 \times 8 = 12,288$개 ($\alpha=32$).
  - LSR-Adapt: $r=4$, $s=16$, 커널 행렬 크기를 $32 \times 2$ 및 $24 \times 2$로 설정. 이때 파라미터 수는 $2 \times (32 \times 2 + 24 \times 2) \times 16 = 3,584$개로, LoRA 대비 약 25% 수준이다.

### 주요 결과

실험 결과, LSR-Adapt는 파라미터 수를 획기적으로 줄였음에도 불구하고 다른 baseline 모델들보다 우수한 성능을 보였다.

- **정량적 성과**: GLUE와 SuperGLUE의 평균 점수에서 LSR-Adapt는 **76.14**를 기록하여, LoRA(74.19), KronA(74.42), KAdaptation(74.69)보다 높은 성능을 달성하였다.
- **효율성**: 특히 LoRA와 비교했을 때, 훨씬 더 적은 파라미터(3,584 vs 12,288)를 사용하면서도 더 높은 정확도를 얻었다는 점이 핵심적인 결과이다.

## 🧠 Insights & Discussion

### 강점 및 이론적 가치

LSR-Adapt는 단순한 경험적 구조 변경이 아니라, 수치 해석의 분리 표현이라는 탄탄한 이론적 배경을 가지고 있다. 부록(Appendix)에서는 임의의 rank-$r$ 행렬이 Kronecker product들의 합으로 표현될 수 있음을 수학적으로 증명함으로써, 제안하는 구조가 이론적으로 타당함을 보였다. 또한, Kronecker product의 특성상 현대 GPU에서 매우 효율적으로 병렬 연산이 가능하다는 잠재적 이점을 가지고 있다.

### 한계 및 미해결 과제

논문에서 명시적으로 언급된 한계점은 현재의 구현이 Kronecker product의 계산적 이점을 완전히 활용하지 않았다는 점이다. 즉, 커스텀 CUDA 커널을 통해 GPU의 Tensor Core를 최적화하여 활용한다면 훈련 속도와 메모리 효율성을 더욱 높일 수 있으나, 본 연구에서는 이를 미래 작업(future work)으로 남겨두었다.

### 비판적 해석

파라미터 수를 1/4 수준으로 줄이면서 성능을 높였다는 결과는 매우 인상적이다. 이는 단순한 low-rank 가정이 현대의 거대 모델 가중치 변화량을 캡처하기에는 부족하며, 더 복잡하지만 효율적인 구조적 가정이 필요함을 시사한다. 다만, 실험이 RoBERTa 모델에 집중되어 있어, 최신 초거대 언어 모델(Llama 3 등)에서도 동일한 파라미터 감소 비율과 성능 향상이 나타날지는 추가적인 검증이 필요해 보인다.

## 📌 TL;DR

본 논문은 LoRA의 팩터 행렬을 수치 해석의 **Low Separation Rank (LSR)** 구조로 분해하여 파라미터 효율성을 극대화한 **LSR-Adapt**를 제안한다. 이를 통해 LoRA 대비 약 **25%의 파라미터만으로도 GLUE/SuperGLUE 벤치마크에서 더 높은 정확도**를 달성하였다. 이 연구는 PEFT의 구조적 설계에 이론적 근거를 제공하였으며, 향후 GPU 최적화 커널 개발을 통해 훈련 속도를 획기적으로 개선할 수 있는 가능성을 제시한다.
