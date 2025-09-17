# Hierarchical Reasoning Model

Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori

## Problem to Solve

인공지능 분야에서 복잡한 목표 지향적 행동 시퀀스를 고안하고 실행하는 추론은 여전히 중요한 도전 과제입니다. 현재 대규모 언어 모델(LLM)은 주로 CoT (Chain-of-Thought) 기법을 사용하지만, 이는 다음과 같은 한계에 직면합니다:

* **취약한 작업 분해:** CoT는 인간이 정의한 단계에 의존하며, 단일 오류가 전체 추론 과정을 망가뜨릴 수 있습니다.
* **과도한 데이터 요구량:** 복잡한 추론 작업에 상당한 양의 훈련 데이터와 많은 토큰 생성이 필요합니다.
* **높은 지연 시간:** 많은 토큰 생성으로 인해 응답 시간이 느려집니다.
* **계산적 깊이의 제약:** 표준 트랜스포머는 본질적으로 얕은 아키텍처를 가지며, 다항 시간 계산이 필요한 복잡한 알고리즘 추론을 수행할 수 없습니다.
* **기존 딥러닝 및 순환 네트워크의 문제점:** 단순히 층을 쌓는 것은 경사 소실 문제를 겪으며, 순환 네트워크는 조기 수렴(early convergence)과 계산 비용이 높은 BPTT (Backpropagation Through Time)에 의존합니다.

## Key Contributions

본 논문은 인간 두뇌의 계층적이고 다중 시간 규모 처리에 영감을 받은 새로운 순환 아키텍처인 **계층적 추론 모델 (Hierarchical Reasoning Model, HRM)**을 제안합니다. 주요 기여는 다음과 같습니다:

* **계산적 깊이 심화:** 단일 순방향 전달만으로도 상당한 계산적 깊이를 달성하며 훈련 안정성과 효율성을 유지합니다.
* **계층적 순환 모듈:** 추상적 계획을 담당하는 고수준 (slow) 모듈과 빠르고 상세한 계산을 처리하는 저수준 (rapid) 모듈의 두 가지 상호 의존적인 순환 모듈을 도입합니다.
* **계층적 수렴 (Hierarchical Convergence):** 저수준 모듈이 로컬 평형에 도달한 후 고수준 모듈이 업데이트되어 저수준 모듈을 "재시작"하는 메커니즘으로, 기존 순환 모델의 조기 수렴 문제를 방지합니다.
* **1단계 경사 근사 (One-step Gradient Approximation):** BPTT가 필요 없는 효율적인 훈련 방법을 제안하여 O(1) 메모리 공간을 유지합니다. 이는 `Deep Equilibrium Models (DEQ)`의 수학적 원리 (Implicit Function Theorem)에 기반합니다.
* **심층 감독 (Deep Supervision):** 주기적인 감독을 통해 고수준 모듈에 더 빈번한 피드백을 제공하고 정규화 효과를 제공합니다.
* **적응형 계산 시간 (Adaptive Computational Time, ACT):** Q-학습 알고리즘을 통해 작업 복잡도에 따라 계산 시간을 동적으로 조절하여 계산 효율성을 높입니다.
* **우수한 성능:** 사전 훈련이나 CoT 데이터 없이 2,700만 개의 매개변수와 단 1,000개의 훈련 샘플만을 사용하여 ARC-AGI, Sudoku-Extreme, Maze-Hard와 같은 복잡한 추론 작업에서 기존 대규모 LLM 및 CoT 모델을 능가하는 탁월한 성능을 달성합니다.
* **뇌-HRM 대응 관계:** 훈련된 HRM의 내부 표현이 인간 두뇌와 유사한 계층적 차원 조직(고수준 모듈의 유효 차원성이 더 높음)을 보이며, 이는 훈련을 통해 나타나는 **자발적 속성 (emergent property)**임을 입증합니다.

## Methodology

HRM은 뇌의 계층적 처리, 시간적 분리, 순환 연결성이라는 세 가지 기본 원리에 영감을 받아 설계되었습니다. 모델은 다음 네 가지 학습 가능한 구성 요소로 이루어져 있습니다: `input network (f_I)`, `low-level recurrent module (f_L)`, `high-level recurrent module (f_H)`, `output network (f_O)`.

1. **모델 아키텍처:**
    * 입력 `x`는 `f_I`를 통해 `~x`로 투영됩니다.
    * 모델의 동역학은 `N`개의 고수준 주기(`high-level cycles`)와 각 주기 내 `T`개의 저수준 시간 단계(`low-level timesteps`)에 걸쳐 전개됩니다.
    * **저수준 모듈 (`f_L`):** 각 시간 단계 `i`마다 이전 상태 `z_{i-1}^L`, 고수준 모듈의 현재 상태 `z_{i-1}^H`, 그리고 입력 표현 `~x`에 따라 상태 `z_i^L`를 업데이트합니다.
        $$z_i^L = f_L(z_{i-1}^L, z_{i-1}^H, \tilde{x}; \theta_L)$$
    * **고수준 모듈 (`f_H`):** `T` 시간 단계마다 한 번 (주기 끝에) 저수준 모듈의 최종 상태를 사용하여 상태 `z_i^H`를 업데이트하며, 그 외에는 이전 상태를 유지합니다.
        $$z_i^H = \begin{cases} f_H(z_{i-1}^H, z_{i-1}^L; \theta_H) & \text{if } i \equiv 0 \pmod T \\ z_{i-1}^H & \text{otherwise} \end{cases}$$
    * `N`개 주기가 완료된 후, 고수준 모듈의 최종 은닉 상태 `z_{NT}^H`로부터 `f_O`를 통해 최종 예측 `$\hat{y}$`가 추출됩니다.
    * `f_L`과 `f_H`는 `Rotary Positional Encoding`, `Gated Linear Units (GLU)`, `RMSNorm` 등 최신 LLM에서 사용되는 개선 사항이 적용된 인코더 전용 트랜스포머 블록으로 구현됩니다.

2. **훈련 절차:**
    * **1단계 경사 근사:** 기존 BPTT의 O(T) 메모리 문제를 해결하기 위해, 각 모듈의 마지막 상태 경사만을 사용하여 다른 상태를 상수로 간주하는 1단계 경사 근사를 적용합니다. 이는 `Deep Equilibrium Models (DEQ)`에서 사용되는 `Implicit Function Theorem (IFT)`에 기반하며 O(1) 메모리 공간을 가능하게 합니다.
    * **심층 감독:** 학습 과정에서 `(x, y)` 데이터 샘플에 대해 HRM 모델의 여러 순방향 전달(`segment`)을 실행합니다. 각 세그먼트 `m`에서, `$(z^m, \hat{y}^m) \leftarrow \text{HRM}(z^{m-1}, x; \theta)$`를 계산하고 손실 `$\mathcal{L}_m \leftarrow \text{LOSS}(\hat{y}^m, y)$`를 계산한 후 매개변수를 업데이트합니다. 다음 세그먼트의 입력 상태 `z^m`은 `detach()`되어 이전 세그먼트의 경사가 전파되지 않도록 합니다.

3. **적응형 계산 시간 (ACT):**
    * 모델이 작업 복잡성에 따라 계산 자원을 동적으로 조절할 수 있도록 합니다.
    * 고수준 모듈의 최종 상태를 사용하는 `Q-head`를 통해 "중단 (`halt`)" 및 "계속 (`continue`)" 행동에 대한 Q-값 `$\hat{Q}^m = (\hat{Q}^m_{\text{halt}}, \hat{Q}^m_{\text{continue}})$`을 예측합니다.
    * 중단 조건은 최대 세그먼트 수 `M_{max}` 초과 또는 예측된 `Q_{halt}` 값이 `Q_{continue}` 값을 초과하고 최소 세그먼트 수 `M_{min}`에 도달했을 때입니다.
    * Q-헤드는 예측 정확도와 중단 결정에 대한 손실 `$\mathcal{L}_m^{\text{ACT}} = \text{LOSS}(\hat{y}^m, y) + \text{BINARYCROSSENTROPY}(\hat{Q}^m, \hat{G}^m)$`을 최소화하도록 Q-학습 알고리즘을 통해 업데이트됩니다.

## Results

HRM은 ARC-AGI, Sudoku-Extreme, Maze-Hard와 같은 복잡한 추론 벤치마크에서 뛰어난 성능을 보였습니다.

* **ARC-AGI 챌린지:**
  * ARC-AGI-1에서 40.3%의 정확도를 달성하여, o3-mini-high (34.5%) 및 Claude 3.7 8K (21.2%)와 같은 훨씬 큰 CoT 기반 모델을 크게 능가했습니다.
* **Sudoku-Extreme (9x9) 및 Maze-Hard (30x30):**
  * 1,000개의 훈련 샘플만으로 Sudoku-Extreme에서 55.0%, Maze-Hard에서 74.5%의 정확도를 달성했습니다. 이는 기존 CoT 기반 최첨단 모델들이 거의 0%의 정확도를 보이며 완전히 실패한 것과 대조적입니다.
* **계층적 수렴 효과:**
  * HRM의 고수준 모듈은 꾸준히 수렴하는 반면, 저수준 모듈은 주기 내에서 반복적으로 수렴하고 재설정되어, 표준 순환 신경망에서 발생하는 조기 수렴 문제를 극복하고 긴 계산 단계에서도 높은 계산 활성도를 유지합니다 (Figure 3).
* **추론 시간 확장성:**
  * HRM은 훈련 없이 추론 시 계산 한계 매개변수 `M_max`를 증가시킴으로써 성능을 향상시킬 수 있음을 입증했습니다. 특히 Sudoku와 같이 깊은 추론이 필요한 작업에서 강한 확장성을 보였습니다 (Figure 5).
* **뇌-HRM 차원 조직 대응:**
  * 훈련된 HRM은 고수준 모듈(`z_H`)이 저수준 모듈(`z_L`)보다 훨씬 높은 유효 차원성을 가지는 계층적 차원 조직(Participation Ratio, PR)을 보였습니다 (Figure 8-(d)에서 `z_L` PR=30.22, `z_H` PR=89.95). 이는 마우스 피질에서 관찰되는 것과 유사하며, 훈련을 통해 나타나는 자발적 속성임이 무작위 가중치를 가진 훈련되지 않은 네트워크와의 비교를 통해 확인되었습니다.
