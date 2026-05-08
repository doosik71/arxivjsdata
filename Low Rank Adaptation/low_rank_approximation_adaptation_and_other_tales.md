# Low-Rank Approximation, Adaptation, and Other Tales

Jun Lu (2024)

## 🧩 Problem to Solve

현대 데이터 분석에서 데이터의 양이 급격히 증가함에 따라 발생하는 계산 복잡도 증가와 노이즈 제거 문제가 핵심 연구 대상이다. 특히 고차원 데이터는 수많은 상호 연관된 변수들로 구성되어 정보가 모호하거나 중복되는 경우가 많으며, 이를 효율적으로 분석하기 위해서는 데이터의 핵심 구조를 유지하면서 차원을 축소하는 방법이 필수적이다.

또한, 최근의 대규모 언어 모델(LLM)과 같은 거대 모델의 경우, 전체 파라미터를 미세 조정(Full Fine-tuning)하는 것은 엄청난 계산 자원과 저장 공간을 요구한다. 따라서 모델의 성능을 최대한 유지하면서 학습 가능한 파라미터 수를 획기적으로 줄여 특정 태스크에 적응시키는 Parameter-Efficient Tuning (PET) 기술의 필요성이 대두되었다. 본 논문은 Low-rank approximation(저차원 근사)의 수학적 메커니즘을 명확히 하고, 이를 기반으로 한 다양한 행렬 분해 기법과 모델 적응(Adaptation) 방안을 제시하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 저차원 근사 및 적응에 대한 포괄적인 수학적 가이드를 제공하고, 기존의 Low-Rank Adaptation(LoRA)을 확장한 새로운 구조들을 제안한 점이다.

1. **ALS 알고리즘의 체계화**: Alternating Least Squares (ALS)를 이용한 저차원 행렬 분해 과정을 정형화하고, 정규화(Regularization) 및 누락된 데이터(Missing entries) 처리 방안을 수학적으로 증명하였다.
2. **특수 행렬 곱 기반의 분해 기법 제시**: Kronecker product, Khatri-Rao product, Hadamard product를 이용한 저차원 분해 방법론과 각 업데이트 식을 도출하였다.
3. **LoKH(Low-Rank Adaptation with Khatri-Rao product) 제안**: 기존 LoRA의 낮은 랭크 제약을 극복하기 위해 Khatri-Rao product를 도입한 LoKH 구조를 제안하였다. 이는 LoRA(낮은 랭크)와 LoHA(매우 높은 랭크) 사이의 균형을 맞추며 더 복잡한 모델 표현력을 가질 수 있음을 이론적으로 제시하였다.
4. **트랜스포머 압축으로의 확장**: 제안된 저차원 근사 기법들을 사전 학습된 트랜스포머 아키텍처의 가중치 압축에 적용하여 추론 및 학습 효율성을 높이는 방향성을 제시하였다.

## 📎 Related Works

논문은 저차원 행렬 분해의 전통적인 방법론으로 Singular Value Decomposition (SVD), Principal Component Analysis (PCA), Non-negative Matrix Factorization (NMF) 등을 언급한다. 특히 추천 시스템의 Matrix Completion 문제에서 데이터의 Low-rank 성질을 가정하여 누락된 값을 예측하는 방식이 널리 사용되었음을 설명한다.

LLM 적응 분야에서는 다음과 같은 기존 연구들을 다룬다:

- **Adapters**: 사전 학습된 모델 층 사이에 작은 신경망을 삽입하는 방식이다.
- **LoRA (Low-Rank Adaptation)**: 가중치 업데이트 행렬 $\Delta W$를 두 개의 저차원 행렬 $B A$의 곱으로 분해하여 학습 파라미터를 줄이는 방식이다.
- **LoHA (Low-Rank Adaptation with Hadamard product)**: 두 저차원 행렬 곱의 Hadamard product를 사용하여 LoRA보다 더 높은 랭크의 표현력을 확보하려는 시도이다.
- **LoKr (Low-Rank Adaptation with Kronecker product)**: Kronecker product를 사용하여 파라미터 효율성을 극대화한 방식이다.

## 🛠️ Methodology

### 1. Alternating Least Squares (ALS) 기반 저차원 분해

행렬 $A \in \mathbb{R}^{M \times N}$를 두 개의 저차원 행렬 $W \in \mathbb{R}^{M \times K}$와 $Z \in \mathbb{R}^{K \times N}$ ($K < \min(M, N)$)의 곱으로 근사하는 문제이다.

**손실 함수 (Loss Function):**
Frobenius norm을 사용하여 원본 행렬과 근사 행렬 간의 차이를 최소화한다.
$$L(W, Z) = \|WZ - A\|_F^2$$

**학습 절차:**
$W$와 $Z$에 대해 각각 convex 하다는 점을 이용해, 하나를 고정하고 다른 하나를 최적화하는 좌표 하강법(Coordinate Descent)을 사용한다.

- **$Z$ 업데이트**: $W$가 고정되었을 때, $Z$의 최적해는 다음과 같다.
  $$Z = (W^\top W)^{-1} W^\top A$$
- **$W$ 업데이트**: $Z$가 고정되었을 때, $W$의 최적해는 다음과 같다.
  $$W^\top = (ZZ^\top)^{-1} ZA^\top$$

**정규화 (Regularization):**
과적합 방지와 Hessian 행렬의 Positive Definite 보장을 위해 $\ell_2$ 정규화 항을 추가한다.
$$L(W, Z) = \|WZ - A\|_F^2 + \lambda_w \|W\|_F^2 + \lambda_z \|Z\|_F^2$$
이 경우 업데이트 식은 $Z = (W^\top W + \lambda_z I)^{-1} W^\top A$ 형태로 변한다.

### 2. 특수 행렬 곱 기반 분해

- **Hadamard Decomposition**: $A \approx (C_1 D_1) \circ (C_2 D_2)$ 형태로 분해한다. 폐쇄형 해(Closed-form solution)가 없으므로 Gradient Descent를 이용한 Alternating Descent 방식을 사용한다.
- **Kronecker Decomposition**: $A \approx B \otimes C$ 형태로 분해하며, 블록 단위의 최소제곱법을 통해 $B$와 $C$를 교대로 업데이트한다.
- **Khatri-Rao Decomposition**: $A \approx B \odot C$ 형태로 분해하며, 각 열별로 독립적인 최소제곱 문제를 풀어 최적의 $B$와 $C$를 찾는다.

### 3. Low-Rank Adaptation (LoRA 및 변형)

사전 학습된 가중치 $W$를 고정하고, 업데이트 행렬 $\Delta W$를 저차원 구조로 설계하여 $\text{output} = Wx + b + \alpha \Delta W x$ 로 계산한다.

- **LoRA**: $\Delta W = BA$
- **LoHA**: $\Delta W = (B_1 A_1) \circ (B_2 A_2)$
- **LoKr**: $\Delta W = A \otimes B$
- **LoKH (제안)**: $\Delta W = A \odot B \odot C \odot D$ (Khatri-Rao product의 연쇄 적용)
  - LoKH는 $\text{rank}(A \odot B) \geq \max\{\text{rank}(A), \text{rank}(B)\}$ 성질을 이용하여, 적은 파라미터로도 LoRA보다 높은 랭크를 구현할 수 있다.

## 📊 Results

본 논문은 특정 데이터셋에 대한 정량적인 벤치마크 결과나 실험 수치를 제시하는 실험 논문이라기보다, 저차원 근사의 수학적 체계와 새로운 구조적 가능성을 제시하는 **이론 및 방법론 중심의 가이드라인 논문**이다.

따라서 구체적인 성능 지표(Accuracy, Perplexity 등)는 명시되어 있지 않으며, 대신 다음과 같은 이론적 결과들을 제시한다:

- **랭크 분석**: Hadamard product의 랭크는 최대 $r_1 \cdot r_2$이며, Kronecker product의 랭크는 $\text{rank}(A)\text{rank}(B)$로 곱해진다는 점을 증명하였다.
- **LoKH의 표현력**: LoKH가 LoRA보다 더 높은 랭크의 모델 공간을 탐색할 수 있음을 Theorem 4($\text{rank}_k$ 관련)를 통해 이론적으로 뒷받침하였다.
- **계산 효율성**: LoRA 계열의 방법론들이 Full Fine-tuning 대비 파라미터 수를 수만 배 줄일 수 있으며, 추론 시 지연 시간(Latency)이 없음을 설명하였다.

## 🧠 Insights & Discussion

**강점 및 통찰:**

- 단순한 알고리즘 소개를 넘어, 기초 수학부터 최신 LLM 적응 기법까지 하나의 일관된 흐름(Low-rank approximation)으로 엮어내어 학술적 가치가 높다.
- 특히 Khatri-Rao product의 성질을 이용해 LoKH라는 새로운 적응 방식을 제안함으로써, 파라미터 효율성과 모델 표현력 사이의 Trade-off를 해결할 수 있는 새로운 방향을 제시하였다.
- 트랜스포머의 Attention 메커니즘에서 Khatri-Rao product가 토큰/패치별 특성 매칭을 용이하게 하여 세밀한 Attention을 가능케 할 수 있다는 해석이 매우 흥미롭다.

**한계 및 논의사항:**

- **실험적 검증 부재**: LoKH 및 다양한 분해 기법들이 실제 LLM이나 Vision Transformer에서 어느 정도의 성능 향상을 가져오는지에 대한 실증적인 실험 결과가 부족하다. 제안된 이론이 실제 하드웨어 가속기(GPU)에서 얼마나 효율적으로 연산되는지에 대한 분석도 필요하다.
- **Hadamard 분해의 제약**: 모든 행렬이 저차원 Hadamard product로 분해될 수 없음을 명시하였는데, 이는 실제 모델 적용 시 $\Delta W$의 표현 가능 범위에 제약이 생길 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 저차원 근사(Low-rank approximation)의 수학적 기초인 ALS부터 특수 행렬 곱(Kronecker, Khatri-Rao, Hadamard) 기반의 분해 기법을 체계적으로 정리하고, 이를 LLM의 효율적 미세 조정에 적용하는 방법론을 다룬다. 특히 기존 LoRA의 표현력 한계를 극복하기 위해 Khatri-Rao product를 이용한 **LoKH** 구조를 제안하였으며, 이는 향후 거대 모델의 압축 및 효율적 적응 연구에 중요한 이론적 토대가 될 가능성이 높다.
