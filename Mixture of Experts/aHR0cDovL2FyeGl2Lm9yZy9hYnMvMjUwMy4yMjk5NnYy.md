# Unified Sparse Mixture of Experts

Giang Do, Hung Le, Truyen Tran (2025)

## 🧩 Problem to Solve

본 논문은 Sparse Mixture of Experts (SMoE) 모델의 라우팅(routing) 방식에서 발생하는 근본적인 한계점을 해결하고자 한다. 기존의 SMoE는 크게 두 가지 접근 방식으로 나뉜다. 첫째는 각 토큰이 최적의 전문가를 선택하는 **Token Choice (TC)** 방식이며, 둘째는 각 전문가가 처리할 최적의 토큰을 선택하는 **Expert Choice (EC)** 방식이다.

이러한 기존 방식들은 다음과 같은 세 가지 주요 문제점을 가지고 있다:
1. **라우팅 효율성 저하**: TC는 노이즈가 많은 토큰을 효과적으로 처리하지 못하고 전문가 간 부하 불균형(unbalanced loading) 문제가 발생한다. 반면, EC는 중요한 토큰이 선택되지 못하고 버려지는 **Token Dropping** 문제가 발생한다.
2. **정보 누출 (Information Leakage)**: 특히 autoregressive 모델에서 EC 방식은 Softmax 연산으로 인해 미래의 토큰 정보가 누출되어 성능이 저하되는 문제가 있다.
3. **표현 붕괴 (Representation Collapse)**: 소수의 전문가만이 지배적으로 선택되거나, 모든 전문가가 유사한 표현을 학습하게 되어 모델의 용량이 효율적으로 활용되지 못하는 현상이 발생한다.

따라서 본 연구의 목표는 고정된 계산 예산 내에서 전문가와 토큰을 선택하는 최적의 기준을 정의하고, 위 문제들을 동시에 해결할 수 있는 통합 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 SMoE의 전문가 선택 과정을 **선형 계획법(Linear Programming, LP)** 관점에서 재해석하고, 이를 바탕으로 **Unified Sparse Mixture of Experts (USMoE)** 프레임워크를 제안한 것이다.

핵심 설계 아이디어는 다음과 같다:
- **통합 점수(Unified Score)**: Token Choice의 토큰 중심 관점과 Expert Choice의 전문가 중심 관점을 선형 결합하여, 두 방식의 상보적인 장점을 모두 취한다.
- **통합 메커니즘(Unified Mechanism)**: 토큰 차원과 전문가 차원을 동시에 고려하여 전체 유사도 행렬에서 글로벌하게 최적의 쌍을 선택함으로써, 토큰 버림(token dropping)과 노이즈 토큰 문제를 동시에 해결한다.
- **이론적 정당성 확보**: 제안하는 방식이 전역적 예산 제약 하에서 총 유사도를 최대화하는 최적해임을 수학적으로 증명하고, 표현 붕괴(representation collapse)를 완화함을 Jacobian 행렬 분석을 통해 입증하였다.

## 📎 Related Works

기존 SMoE 연구들은 주로 라우터 정책을 개선하여 부하 불균형이나 표현 붕괴를 막으려 했다. 예를 들어, XMoE는 저차원 라우팅 점수를 사용하고, SMoE-dropout은 점진적으로 전문가를 활성화하는 방식을 사용한다. HyperRouter나 StableMoE는 라우터의 안정성과 강건성을 높이는 데 집중했다.

하지만 기존 연구들은 다음과 같은 한계가 있다:
- **보조 손실 함수(Auxiliary Loss) 의존성**: 대부분의 TC 방식은 부하 균형을 위해 보조 손실을 사용하는데, 이는 태스크 손실 함수와의 정밀한 균형 조절이 필요하며 트레이드오프를 발생시킨다.
- **단일 관점의 선택**: TC는 토큰의 중요도를 무시하고 모든 토큰에 동일한 전문가 수를 할당하며, EC는 전문가의 용량 제한으로 인해 중요한 정보를 손실할 위험이 크다.

USMoE는 이러한 단일 관점의 선택에서 벗어나, LP 관점에서 토큰과 전문가의 관계를 전역적으로 최적화함으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
USMoE는 표준 Transformer의 MLP 층을 MoE 층으로 대체하며, 입력 토큰 $h \in \mathbb{R}^{T \times d}$와 전문가 함수 $\{FFN_j\}_{j=1}^N$ 사이의 라우팅을 제안된 통합 점수와 메커니즘으로 수행한다.

### 주요 구성 요소 및 절차

**1. 전문가 선택의 선형 계획법(LP) 공식화**
저자들은 전문가 선택 문제를 다음과 같이 유사도 합계를 최대화하는 최적화 문제로 정의한다.
$$\text{maximize} \sum_{i=1}^T \sum_{j=1}^N S_{ij} x_{ij}$$
$$\text{subject to} \sum_{i,j} x_{ij} \le c, \quad x_{ij} \in \{0, 1\}$$
여기서 $S$는 토큰과 전문가 간의 유사도 행렬이며, $c$는 전체 계산 예산(활성화될 총 엣지 수)이다. 본 논문은 이 문제의 최적해가 유사도 행렬 $S$에서 전역적으로 상위 $c$개의 항목을 선택하는 것($X^{USMoE} = \text{TopK}(S, c)$)임을 증명하였다.

**2. 통합 점수 함수 (Unified Score Function)**
토큰 중심의 점수($f^t$)와 전문가 중심의 점수($f^e$)를 선형 결합하여 최종 점수를 산출한다.
$$f^{USMoE}(S) = \alpha \cdot f^e(S) + \beta \cdot f^t(S)$$
단, $\alpha + \beta = 1$이며 $\alpha, \beta \ge 0$이다.
- $f^t(S)$: 행(row) 기준의 점수 (예: Softmax), Token Choice의 관점을 반영한다.
- $f^e(S)$: 열(column) 기준의 점수 (예: Sigmoid), Expert Choice의 관점을 반영한다.
특히, EC에서 발생하는 정보 누출 문제를 해결하기 위해 Softmax 대신 **Sigmoid** 함수를 사용하는 것이 더 효과적임을 발견하고 이를 적용하였다.

**3. 통합 메커니즘 (Unified Mechanism)**
기존의 행/열 기반 선택 대신, 계산된 통합 점수 행렬을 1차원 벡터로 평탄화(flatten)한 후, 전역적으로 가장 높은 점수를 가진 상위 $n$개의 토큰-전문가 쌍을 선택한다. 이를 통해 특정 토큰이 무시되거나 불필요한 전문가가 선택되는 것을 방지한다.

### 학습 및 추론 절차 (Algorithm 1)
1. 입력 $X$와 라우터 가중치 $R$의 내적을 통해 유사도 $\text{logits}$를 계산한다.
2. $\text{logits}$에 Softmax(TC용)와 Sigmoid(EC용)를 각각 적용하여 $tc\_score$와 $ex\_score$를 얻는다.
3. 가중치 $\alpha$를 사용하여 두 점수를 결합하고, 전체 행렬을 평탄화한다.
4. $\text{TopK}$ 연산을 통해 전역적으로 최적의 쌍 인덱스를 추출한다.
5. 선택된 전문가들을 통해 연산을 수행하고 결과를 반환한다.

## 📊 Results

### 실험 설정
- **LLM 평가**: OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, DeepSeekMoE-16B 모델을 사용하였으며, MTEB 벤치마크를 통해 성능을 측정하였다.
- **비전 평가**: ViT-MoE 아키텍처(10M, 110M 파라미터)를 사용하여 CIFAR-10/100, STL-10, SVHN, ImageNet-1K 데이터셋에서 정확도를 측정하였다.
- **시나리오**: Training-free (플러그인 방식), SFT (Supervised Fine-Tuning), Training from scratch (처음부터 학습) 세 가지 설정을 모두 검증하였다.

### 주요 결과
**1. 언어 모델 (LLM)**
- **Training-free**: MTEB 벤치마크에서 USMoE가 TC와 EC보다 일관되게 높은 성능을 보였다. 특히 요약(Summarization) 태스크에서 Qwen1.5-MoE의 경우 TC 대비 성능이 비약적으로 향상되었다.
- **강건성(Robustness)**: 데이터에 노이즈(random "AAA" tokens)를 주입한 corrupted 설정에서도 USMoE는 타 방식보다 훨씬 낮은 Perplexity를 유지하며 강력한 강건성을 보였다.
- **효율성**: 정수 형태의 $k$값이 아닌 소수점 단위의 전문가 선택(예: 토큰당 평균 1.5개 전문가)이 가능하여, 성능 저하를 최소화하면서 FLOPs를 약 14% 절감하였다.

**2. 비전 태스크 (Vision)**
- **정확도**: 8개 태스크와 4개 데이터셋 모두에서 TC, EC, SoftMoE보다 높은 평균 정확도를 기록하였으며, 표준 편차가 낮아 안정적인 성능을 보였다.
- **적대적 공격 강건성**: PGD, FGSM, SPSA 공격 하에서 TC는 성능이 급격히 하락했으나, USMoE는 가장 높은 평균 강건성(PGD 46.2%, FGSM 43.3%, SPSA 63.9%)을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 분석
본 논문은 단순한 실험적 제안을 넘어 수학적 증명을 통해 USMoE의 우월성을 입증하였다.
- **최적성**: 전역적 Top-K 선택이 주어진 예산 하에서 유사도 합계를 최대화하는 최적해임을 증명하여, 왜 USMoE가 TC나 EC보다 이론적으로 더 나은 선택을 하는지 설명한다.
- **표현 붕괴 완화**: Jacobian 행렬 분석을 통해, 통합 점수 함수를 사용하면 가중치 업데이트 경로가 더 다양해져($2n \gg n$) 전문가들이 서로 다른 특성을 학습하게 됨으로써 표현 붕괴 문제를 효과적으로 억제할 수 있음을 보였다.

### 한계 및 비판적 해석
- **계산 리소스의 제약**: 저자들은 pre-training 실험을 Transformer-XL 베이스 모델과 중간 규모 데이터셋에서 수행하였다. 따라서 100B 이상의 초거대 모델에서도 동일한 확장성(scalability)이 유지될지는 추가적인 검증이 필요하다.
- **하이퍼파라미터 $\alpha$ 의존성**: 통합 점수에서 $\alpha$ 값에 따라 성능이 달라지며, 실험적으로 $\alpha \in (0.3, 0.7)$ 범위에서 최적의 성능이 나타남을 확인하였다. 이는 실제 적용 시 최적의 $\alpha$를 찾기 위한 탐색 과정이 필요함을 의미한다.

## 📌 TL;DR

본 논문은 SMoE의 라우팅 문제를 선형 계획법(LP) 관점에서 재정의하고, 토큰 중심(TC)과 전문가 중심(EC)의 장점을 결합한 **Unified Sparse Mixture of Experts (USMoE)**를 제안한다. 전역적 Top-K 선택 메커니즘과 통합 점수 함수를 통해 **표현 붕괴(Representation Collapse), 토큰 버림(Token Dropping), 정보 누출(Information Leakage)** 문제를 동시에 해결하였다. LLM과 Vision 분야 모두에서 기존 방식 대비 높은 성능과 강건성을 입증했으며, 특히 계산 비용을 14% 줄이면서도 경쟁력 있는 정확도를 유지할 수 있어 자원 제한적인 환경에서의 모델 배포에 중요한 기여를 할 것으로 기대된다.