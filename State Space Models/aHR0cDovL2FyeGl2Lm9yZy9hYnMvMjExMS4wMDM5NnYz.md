# Efficiently Modeling Long Sequences with Structured State Spaces
Albert Gu, Karan Goel, and Christopher R ́e

## 🧩 Problem to Solve
이 논문은 시퀀스 데이터, 특히 10,000단계 이상의 매우 긴 시퀀스에서 장거리 의존성(Long-Range Dependencies, LRDs)을 효율적으로 모델링하는 데 따르는 근본적인 문제를 다룹니다. 기존 RNN, CNN, Transformer 모델들은 LRD를 포착하기 위한 변형 모델을 가지고 있음에도 불구하고, 이러한 긴 시퀀스에서는 확장이 어렵고 성능이 저하되는 문제가 있었습니다. 선행 연구에서 상태 공간 모델(State Space Model, SSM)을 활용하여 LRD를 수학적, 경험적으로 다룰 수 있음을 보였으나, 이 방법은 계산 및 메모리 요구 사항이 지나치게 커서 실제 적용이 불가능했습니다.

## ✨ Key Contributions
*   **새로운 SSM 파라미터화 (S4):** SSM의 상태 행렬 $A$를 `Normal Plus Low-Rank (NPLR)` 형태로 재파라미터화하여 기존 방법보다 훨씬 효율적인 계산을 가능하게 했습니다.
*   **효율적인 계산 방법론:** `Woodbury Identity`를 통해 저랭크 항을 보정하고, `Cauchy kernel` 계산으로 안정적으로 대각화하여 SSM 계산을 $\tilde{O}(N+L)$ 시간 복잡도와 $O(N+L)$ 메모리 사용으로 줄였습니다. 이는 이론적으로 거의 최적에 가깝습니다.
*   **다양한 벤치마크에서 SOTA 달성:** `Long Range Arena (LRA)` 벤치마크의 모든 태스크에서 SOTA를 달성했으며, 특히 다른 모든 모델이 실패했던 길이 16K의 `Path-X` 태스크를 해결했습니다.
*   **원시 음성 분류 성능 향상:** 16000 길이의 원시 음성 시퀀스 분류에서 전문화된 Speech CNN의 테스트 오류율을 절반으로 줄였습니다.
*   **다목적 시퀀스 모델로서의 잠재력:** 이미지, 언어 모델링에서 Transformer와의 성능 격차를 크게 줄였으며, 60배 빠른 생성 속도를 보여주었습니다. 또한 2D 귀납적 바이어스 없이도 Sequential CIFAR-10에서 2D ResNet과 동등한 성능을 보였고, 샘플링 해상도 변경에도 재훈련 없이 적응하는 능력을 입증했습니다.

## 📎 Related Works
*   **LSSL (Linear State Space Layer) [Gu et al., 2021]:** SSM을 기반으로 LRD를 다루는 선행 연구로, S4가 해결하고자 하는 비실용적인 계산 및 메모리 문제를 가지고 있었습니다.
*   **HiPPO (High-order Polynomial Projection Operator) 이론 [Gu et al., 2020; Voelker et al., 2019]:** SSM이 LRD를 포착할 수 있도록 특별한 상태 행렬 $A$를 유도한 연속 시간 기억 이론으로, S4는 이 HiPPO 행렬의 구조를 활용합니다.
*   **RNN, CNN, Transformer:** 기존 시퀀스 모델 아키텍처들로, LRD를 다루기 위한 다양한 변형이 존재하지만, 매우 긴 시퀀스에서는 확장성 문제와 성능 한계를 보였습니다.
*   **효율적인 Transformer 변형 [Choromanski et al., 2020; Katharopoulos et al., 2020]:** 시퀀스 길이의 이차 의존성을 줄이기 위해 제안되었으나, 여전히 LRA와 같은 도전적인 벤치마크에서 한계를 보였습니다.

## 🛠️ Methodology
S4는 SSM을 연속(continuous), 재귀(recurrent), 컨볼루션(convolutional) 세 가지 관점에서 효율적으로 계산하는 방법을 제시합니다.

1.  **HiPPO 행렬의 NPLR 재파라미터화:** SSM의 핵심인 상태 행렬 $A$를 `Normal Plus Low-Rank (NPLR)` 형태로 재정의합니다. 즉, $A = V \Lambda V^* - P Q^*$ (유니타리 $V$, 대각 $Λ$, 저랭크 $P, Q$) 형태로 분해합니다. HiPPO 행렬이 비정규 행렬이지만, 이 형태로 분해될 수 있음을 증명합니다.
2.  **생성 함수를 통한 컨볼루션 커널 $K$ 계산:** $K$를 직접 계산하는 대신, 이의 `truncated generating function` $\hat{K}(z; A,B,C) = C^*(I - A^L z^L)(I - Az)^{-1}B$를 `unity의 근(roots of unity)`에서 평가하여 `Inverse FFT`를 통해 $K$를 복구합니다. 이는 시퀀스 모델의 시간 복잡도를 $\tilde{O}(L)$로 만듭니다.
3.  **Woodbury Identity 적용:** 생성 함수 계산에 필요한 행렬 역행렬 $(I - Az)^{-1}$에 `Woodbury Identity`를 적용하여 저랭크 수정 항을 효율적으로 처리합니다. 이를 통해 대각 행렬 경우로 문제를 환원합니다.
4.  **Cauchy Kernel 계산으로 환원:** 대각 행렬 경우의 계산은 `Cauchy kernel` $1/(\omega_j - \zeta_k)$의 계산과 동등함을 보입니다. 이는 수치 해석 분야에서 `Fast Multipole Method (FMM)` 기반의 안정적이고 준선형 시간 복잡도 $\tilde{O}(N+L)$를 갖는 알고리즘이 잘 연구되어 있습니다.
5.  **재귀적 표현의 효율성:** 이산화된 SSM의 행렬 $A$도 `DPLR (Diagonal Plus Low-Rank)` 형태를 유지하므로, 한 단계의 재귀 계산을 $O(N)$ 연산으로 수행할 수 있습니다.

## 📊 Results
*   **효율성 벤치마크:** LSSL 대비 최대 30배 빠르고 메모리 사용량은 400배 적으며, Linear Transformer, Performer 등 최신 효율적인 Transformer 변형들과 유사한 속도와 메모리 효율성을 보였습니다.
*   **LRD 벤치마크 (LRA):** `Path-X` (길이 16384) 태스크에서 88%의 정확도를 달성하여 기존 모델들의 50% 무작위 추측 성능을 크게 뛰어넘었으며, LRA의 모든 태스크에서 평균 20점 이상 향상된 SOTA 성능을 기록했습니다. 학습된 컨볼루션 필터 시각화를 통해 S4가 2D 데이터의 공간적 구조와 LRD를 효과적으로 학습함을 확인했습니다.
*   **원시 음성 분류 (SC10):** 길이 16000의 원시 음성 시퀀스에 대해 98.3% 정확도를 달성하여, MFCC 피처를 사용하는 모델이나 전문화된 Speech CNN (WaveGAN-D)보다 우수했습니다.
*   **대규모 생성 모델링:** CIFAR-10 밀도 추정에서 2.85 bits per dim으로 SOTA autoregressive 모델과 경쟁력 있는 성능을 보였고, WikiText-103 언어 모델링에서 Transformer와의 perplexity 격차를 0.8까지 줄이며 `attention-free` 모델 중 SOTA를 기록했습니다.
*   **빠른 생성:** 재귀적 모드를 사용하여 CIFAR-10 및 WikiText-103에서 표준 autoregressive 모델보다 60배 빠른 토큰/픽셀 생성을 시연했습니다.
*   **샘플링 해상도 변경:** 재훈련 없이 내부 `step size ∆`만 변경하여 0.5배 주파수의 음성 데이터에서 96.3% 정확도를 유지했습니다.
*   **약한 귀납적 바이어스 학습:** 시간 시계 예측에서 Informer를 능가하고, Sequential CIFAR에서 2D ResNet에 필적하는 91% 이상의 정확도를 달성하여 다양한 도메인에 적용 가능함을 입증했습니다.
*   **HiPPO의 중요성:** HiPPO 초기화가 SSM의 성능에 결정적인 역할을 함을 `sequential CIFAR-10` 태스크를 통한 `ablation study`로 확인했습니다.

## 🧠 Insights & Discussion
S4는 SSM의 이론적 강점(연속 시간, 컨볼루션, 재귀 표현 간 전환 능력)을 실제 문제에 적용 가능하게 하여, LRD를 효과적으로 다루는 강력한 일반 목적 시퀀스 모델로서의 잠재력을 제시합니다. 특히, `Path-X`와 같은 매우 도전적인 LRD 태스크를 해결한 것은 주목할 만합니다.

한계점으로는 언어 모델링에서 여전히 Transformer와의 성능 격차가 존재한다는 점이 언급되었습니다. 향후 연구 방향으로는 S4와 다른 시퀀스 모델의 조합을 통해 강점을 보완하고, 오디오 데이터에서의 사전 훈련 및 생성, 그리고 이미지 및 비디오와 같은 고차원 데이터로의 확장 가능성이 제시되었습니다. 초기 S4 버전에서 `A` 행렬의 고유값이 우반평면에 있을 때 발생할 수 있는 수치적 불안정성 문제도 `Λ - P P^*` 파라미터화를 통해 개선될 수 있음을 언급했습니다.

## 📌 TL;DR
S4는 `HiPPO` 이론에 기반한 시퀀스 모델로, `Normal Plus Low-Rank (NPLR)` 파라미터화를 통해 SSM의 `장거리 의존성(LRD)` 모델링에 내재된 심각한 계산 및 메모리 병목 현상을 해결했습니다. 이 방법은 `Woodbury Identity`와 `Cauchy kernel` 계산을 활용하여 $\tilde{O}(N+L)$의 효율적인 연산 복잡도를 달성하며, `Long Range Arena (LRA)` 벤치마크, 원시 음성 분류, 이미지/언어 생성 모델링 등 다양한 태스크에서 기존 모델을 능가하는 `SOTA` 성능을 입증하여 강력한 일반 목적 시퀀스 모델로서의 가능성을 보여주었습니다.