# Compensate Quantization Errors: Make Weights Hierarchical to Compensate Each Other

Yifei Gao, Jie Ou, Lei Wang, Yuting Xiao, Zhiyuan Xiang, Ruiting Dai, Jun Cheng (2024)

## 🧩 Problem to Solve

거대 언어 모델(LLM)은 탁월한 성능과 추론 능력을 갖추고 있으나, 이를 유지하고 실행하기 위한 계산 자원과 저장 공간의 비용이 매우 높다. 이를 해결하기 위해 모델의 가중치를 낮은 비트로 표현하는 양자화(Quantization)가 필수적인 해결책으로 부상하였다.

하지만 양자화 과정에서 발생하는 양자화 오차(Quantization Error)는 모델의 정확도를 저하시키는 주요 원인이 된다. 특히 훈련 후 양자화(Post-Training Quantization, PTQ) 방식은 훈련 중 양자화(Quantization-Aware Training, QAT)보다 계산 비용이 훨씬 적어 선호되지만, 성능 저하 문제를 효과적으로 해결해야 하는 과제를 안고 있다. 본 논문의 목표는 PTQ 설정에서 가중치 간의 상호 보완을 통해 양자화 오차를 최소화하고, 특히 매우 낮은 비트(extremely low bit) 설정에서도 모델의 성능을 유지하거나 향상시키는 새로운 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 가중치의 구조를 계층적(Hierarchical)으로 만들어 양자화 설정에 스스로 적응하게 만드는 **Learnable Singular value Increment (LSI)** 기법을 도입하는 것이다.

기존의 양자화 방법들이 단순히 오차를 줄이는 데 집중했다면, LSI는 특이값 분해(Singular Value Decomposition, SVD)를 통해 추출된 특이값에 학습 가능한 증분(Increment)을 추가하여 가중치 분포에 의도적인 '미세한 교란'을 일으킨다. 이러한 교란은 가중치들이 전역적인 최적점(Global Optimum)을 찾도록 유도하며, 결과적으로 가중치들이 서로의 오차를 보완하는 계층적 구조를 갖게 하여 균일 양자화(Uniform Quantization) 설정에 최적화되도록 한다.

## 📎 Related Works

양자화 방법론은 크게 두 가지 흐름으로 나뉜다.

1. **Weight-Only Quantization**: 가중치 행렬만을 저비트로 변환하는 방식이다. GPTQ 시리즈는 래그랑주 방정식을 통해 헤시안(Hessian) 행렬을 업데이트하여 가중치 오차를 상쇄하며, AWQ는 활성화 값(Activation)을 고려한 스케일링 전략을 사용한다. 하지만 이러한 방식들은 하드웨어 효율성이 떨어지는 혼합 정밀도(Mixed-precision)를 사용하거나 양자화 시간이 오래 걸리는 한계가 있다.
2. **Weight-Activation Quantization**: 가중치와 활성화 값을 모두 양자화하는 방식으로, 메모리와 시간 효율을 극대적으로 높일 수 있다. SmoothQuant와 OmniQuant는 가중치와 활성화 값 사이의 크기(Magnitude)를 전이시켜 양자화 난이도를 낮추는 Smoothing 기법을 사용한다.

본 연구는 이러한 기존의 Smoothing 기법과 가중치 보완 아이디어를 통합하여, 데이터 의존성을 최소화하면서도 매우 적은 수의 파라미터(전체 가중치의 0.1% 미만)만을 학습시켜 성능을 극대화한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. LSI (Learnable Singular value Increment) 구조

LSI는 단일 선형 층의 가중치 행렬 $W \in \mathbb{R}^{a \times b}$를 SVD를 통해 세 개의 행렬 $U, S, V^h$로 분해한다. 여기서 $S$는 대각 성분만을 가진 특이값 행렬이다. 본 방법론은 $S$에 학습 가능한 증분 행렬 $I'$를 추가하여 가중치 분포를 미세하게 조정한다.

양자화된 가중치 $\tilde{W}$를 구하는 과정은 다음과 같다.

$$ \text{Q}(W, k, s_h, z, I') = \text{Clamp}\left(\left\lfloor \frac{U \odot \text{diag}(S + I') \odot V^h}{s_h} \right\rfloor + z, 0, 2^k - 1\right) $$
$$ \tilde{W} = (\text{Q}(W, k, s_h, z, I') - z)s_h $$

여기서 $s_h$는 shift, $z$는 zero-point, $k$는 비트 수이다. LSI의 목표는 원래의 함수 출력 $F(W, X)$와 양자화된 가중치를 사용한 출력 $F(\tilde{W}, X)$ 사이의 차이를 최소화하는 최적의 $I'$를 찾는 것이다.

$$ \arg \min_{I'} \| F(W, X) - F(\tilde{W}, X) \| $$

### 2. Group-wise Scaling 보완

가중치를 여러 그룹으로 나누어 서로 다른 스케일을 적용하는 group-wise 설정에서는 $I'$만으로는 최적의 파라미터를 찾기 어렵다. 이를 해결하기 위해 대각 행렬 $\text{diag}(S + I')$의 앞부분(상위 $n \times n$ 영역)에 작은 정방 행렬을 추가로 도입하여, 영향력이 큰 상위 특이값들이 그룹 간의 균형을 더 잘 잡을 수 있도록 설계하였다.

### 3. 통합 파이프라인 (Smooth & Clipping)

LSI는 독립적으로 작동하는 것이 아니라, OmniQuant에서 제안된 **LET (Learnable Equivalent Transformation)** 및 **LWC (Learnable Weight Clipping)** 기술과 통합되어 사용된다.

- **LET**: 가중치와 활성화 값 사이의 스케일 인자를 학습 가능하게 만들어 양자화 난이도를 조절한다.
- **LWC**: 양자화 함수의 상한과 하한 경계를 학습 가능하게 만들어 이상치(Outlier) 문제를 해결한다.

## 📊 Results

### 1. 실험 설정

- **모델 및 데이터셋**: OPT (125M ~ 66B) 및 LLaMA (7B ~ 30B) 모델을 사용하였으며, WikiText2, PTB, C4 데이터셋에서 Perplexity(PPL)를 측정하였다.
- **양자화 설정**: Weight-only (INT4, INT3, INT2) 및 Weight-Activation (W6A6, W4A4) 설정을 모두 테스트하였다.

### 2. 정량적 결과

- **Weight-only Quantization**: OPT 모델의 W2A16, W3A16 설정에서 GPTQ, AWQ, OmniQuant보다 낮은 PPL을 기록하며 SOTA 성능을 달성하였다. 특히 매우 낮은 비트(W2) 설정에서 성능 향상이 두드러졌다.
- **Weight-Activation Quantization**: W4A4 설정에서 OmniQuant를 유의미하게 앞질렀으며, LLaMA 모델의 zero-shot 태스크(PIQA, ARC, HellaSwag 등)에서도 더 높은 정확도를 보였다.
- **양자화 효율**: LSI는 학습해야 할 파라미터가 매우 적어 훈련 속도가 매우 빠르다. 예를 들어 OPT-30B 모델의 W4A16 설정에서는 32개의 샘플과 2 epoch만으로 충분한 성능을 냈으며, 훈련 시간은 1.5시간 미만으로 소요되었다.

### 3. Quantized Model Fine-tuning

LSI의 특성을 이용해 양자화된 모델의 마지막 몇 개 층(layer)만을 빠르게 파인튜닝했을 때, 특정 데이터셋에 대한 성능이 크게 향상됨을 확인하였다. 이는 LSI가 이전 층에서 발생한 양자화 오차를 보완하는 능력이 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 성과

LSI는 가중치를 계층적으로 재구성함으로써 양자화 오차를 '제거'하는 것이 아니라 '상쇄'하는 전략을 취한다. 이는 하드웨어 효율성을 해치지 않는 균일 양자화(Uniform Quantization)를 유지하면서도, 극소수의 파라미터 학습만으로 FP16 모델에 근접한 성능을 낼 수 있다는 점에서 매우 실용적이다.

### 한계 및 비판적 해석

1. **Overfitting 문제**: LSI는 활성화 값(Activation)을 고려하여 가중치를 재분배하므로, 학습 데이터셋에 과적합될 위험이 있다. 실제로 한 데이터셋에서 PPL이 낮아지면 다른 데이터셋에서 성능이 떨어지는 트레이드-오프가 관찰되었다.
2. **훈련 불안정성**: 모델 크기에 따라 최적의 epoch 수가 다르며, 이에 대한 고정된 패러다임이 없어 세밀한 튜닝이 필요하다.
3. **Group-wise Scaling과의 충돌**: SVD는 가중치 행렬 전체의 전역적 분포를 다루는 반면, group-wise scaling은 국소적인 영역을 다룬다. 이로 인해 모델 크기가 커지고 group-wise scaling이 강해질수록 LSI의 효과가 감소하는 경향이 발견되었다.

## 📌 TL;DR

본 논문은 LLM의 양자화 오차를 해결하기 위해 SVD 기반의 학습 가능한 특이값 증분 기법인 **LSI (Learnable Singular value Increment)**를 제안한다. LSI는 가중치를 계층적 구조로 만들어 양자화 설정에 최적화시키며, OmniQuant 등의 기존 Smoothing 기술과 결합하여 Weight-only 및 Weight-Activation 양자화 모두에서 SOTA 성능을 달성하였다. 특히 매우 적은 파라미터 학습만으로 고성능을 낼 수 있어 효율적이며, 양자화된 모델의 빠른 파인튜닝 가능성을 열었다는 점에서 향후 저비용 고성능 LLM 배포에 중요한 역할을 할 것으로 기대된다.
