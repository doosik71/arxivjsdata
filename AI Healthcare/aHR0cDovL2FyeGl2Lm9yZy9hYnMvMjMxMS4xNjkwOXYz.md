# Multinomial belief networks for healthcare data

H. C. Donker, D. Neijzen, J. de Jong, G. A. Lunter (2024)

## 🧩 Problem to Solve

본 논문은 의료 데이터(healthcare data)가 갖는 고유한 특성인 희소성(sparsity), 높은 결측률(high missingness), 그리고 상대적으로 작은 샘플 크기로 인해 발생하는 분석적 어려움을 해결하고자 한다. 

일반적인 최대 가능도 추정(Maximum Likelihood Estimation, MLE) 기반의 머신러닝 방법론은 이러한 데이터 특성으로 인해 편향된 결과를 낳거나, 분포 외(Out-of-Distribution) 데이터에 대해 과도하게 확신하는(overconfident) 경향이 있으며, 불확실성(uncertainty)을 적절히 처리하지 못하는 한계가 있다. 특히 의료 분야에서는 진단 및 치료 결정에 있어 불확실성의 정량화가 매우 중요하므로, 이를 해결하기 위한 강건한 확률론적 모델이 필요하다.

논문의 목표는 다항 분포(Multinomial distribution)를 출력 변수로 사용하는 딥 생성 베이지안 모델(Deep Generative Bayesian Model)인 Multinomial Belief Network(MBN)를 제안하여, 데이터 효율성을 높이고 과적합을 방지하며 불확실성을 체계적으로 다루는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 기존의 Latent Dirichlet Allocation(LDA)이 가진 단층 구조의 한계와 Poisson Gamma Belief Network(PGBN)의 출력 분포 제약을 극복하는 것이다.

1. **심층 다항 구조 설계**: LDA의 다항 분포 출력과 PGBN의 심층 계층 구조를 결합하여, 토픽 간의 상호작용을 여러 계층에 걸쳐 캡처할 수 있는 MBN을 설계하였다.
2. **효율적인 추론 알고리즘 개발**: Dirichlet-Multinomial 분포, Chinese Restaurant Table(CRT) 분포, 그리고 Polya urn scheme 사이의 새로운 수학적 관계(Theorem 1)를 정립하여, Collapsed Gibbs Sampling을 통한 효율적인 사후 분포 샘플링을 가능하게 하였다.
3. **의료 데이터에 최적화된 유연성**: 다항 분포를 채택함으로써 범주형 변수, 텍스트, DNA 변이 등 다양한 의료 데이터를 통합적으로 모델링할 수 있으며, 결측치를 관측 횟수 0으로 설정함으로써 자연스럽게 처리할 수 있도록 하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 한계를 지적한다.

- **LDA (Latent Dirichlet Allocation)**: 다항 분포를 사용하여 성공적으로 적용되었으나, 하이퍼파라미터 추론이 느리고 샘플 간 토픽 가중치의 상관 구조(correlation structure)를 무시하는 단층 구조라는 한계가 있다.
- **PGBN (Poisson Gamma Belief Network)**: Gamma 변수와 Poisson 관측치를 사용하여 심층 구조를 구현했으나, 출력 변수가 Poisson 분포로 제한되어 있어 범주형 데이터나 다항 분포가 필요한 데이터에 적용하기 어렵다.
- **Variational Approaches**: 학습 속도는 빠르나 사후 분포의 형태를 미리 고정하거나 Mean-field 가정을 사용하는 등 근사치에 의존하므로, 베이지안 모델의 이상적인 이점(정확한 불확실성 측정 등)을 완전히 누리지 못한다.

MBN은 PGBN의 심층 구조와 LDA의 다항 분포 출력을 결합하여, 심층 신경망과 유사한 표현력을 가지면서도 베이지안 추론의 정밀함을 유지한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 생성 모델
MBN은 여러 층의 Dirichlet-분포 hidden unit들이 쌓여 있는 구조이다. 각 층의 활성화 값은 하위 층의 가중치 합으로 결정되며, 최종 층에서 다항 분포를 통해 관측값이 생성된다.

**생성 과정의 주요 방정식:**
1. **최상위 활성화**: $\{r_v\} \sim \text{Dir}(\{\gamma_0/K_T\})$
2. **은닉 유닛 (Hidden Units)**: $\text{t}=T, \dots, 1$에 대해,
   $$\{\theta_{vj}^{(t)}\}_v \sim \text{Dir}(\{c^{(t+1)} a_{vj}^{(t+1)}\}_v)$$
3. **층간 활성화 (Activation)**: 
   $$a_{vj}^{(t)} = \sum_{k=1}^{K_t} \phi_{vk}^{(t)} \theta_{kj}^{(t)}$$
   여기서 $\phi_{vk}^{(t)}$는 층 간 연결 가중치이며, $\phi_{vk}^{(t)} \sim \text{Dir}(\{\eta_v^{(t)}\}_v)$를 따른다.
4. **최종 관측값**: $\{x_{vj}\}_v \sim \text{Mult}(n_j, \{a_{vj}^{(1)}\}_v)$

### 심층 다항 표현 및 추론 (Inference)
모델의 복잡성으로 인해 직접적인 추론이 어렵기 때문에, $\theta^{(t)}$를 적분하여 제거한(integrating out) 대안적 표현을 사용한다. 이때 본 논문의 핵심 기여인 **Theorem 1**이 사용된다.

**Theorem 1 (핵심 수학적 관계):**
Dirichlet-Multinomial 분포에서 추출된 샘플은 CRT 분포와 Polya urn scheme의 조합과 동일한 결합 분포를 가진다. 이를 통해 다음과 같은 변환이 가능해진다.
- $\text{DirMult} \to \text{CRT} \to \text{Mult} \to \text{Polya}$

이 관계를 이용하여 **Collapsed Gibbs Sampling**을 수행한다. 추론 절차는 다음과 같은 상향식(Upward pass)과 하향식(Downward pass) 과정으로 구성된다.
- **Upward Pass**: 관측값 $x^{(1)}$부터 시작하여 $y^{(t)} \to m^{(t)} \to x^{(t+1)}$ 순으로 잠재 변수들을 샘플링하여 상위 층으로 정보를 전달한다.
- **Downward Pass**: 상위 층에서 결정된 파라미터를 바탕으로 $\phi^{(t)}, \theta^{(t)}, c^{(t)}$ 및 최상위 $r$을 업데이트한다.

## 📊 Results

### 실험 설정
- **데이터셋**: UCI 손글씨 숫자 데이터셋(8x8 이미지), 암 환자의 DNA point mutation 데이터셋(8,500만 개 변이, 4,645명 환자).
- **비교 대상**: NMF (Non-negative Matrix Factorization), SigProfilerExtractor, PGBN.
- **평가 지표**: Held-out Perplexity (낮을수록 우수).

### 주요 결과
1. **손글씨 숫자 데이터**:
   - MBN은 층이 깊어질수록 Perplexity가 소폭 감소하며 NMF($34.2$)보다 우수한 성능($\approx 30.7$)을 보였다.
   - **정성적 결과**: 하위 층에서는 작은 패치(patch)를 학습하고, 상위 층으로 갈수록 이를 조합해 일반적인 숫자 형태를 학습하는 계층적 추상화 능력을 확인하였다.

2. **암 변이 데이터 (Mutational Signature Attribution)**:
   - MBN과 PGBN은 SigProfilerExtractor($64.5$)보다 낮은 Perplexity($\approx 61.9$)를 기록하여 더 정확한 변이 할당 능력을 보였다.
   - **Meta-signature 발견**: 2층 MBN을 통해 4개의 핵심 **Meta-signatures ($M_1 \sim M_4$)**를 식별하였다.
     - $M_1$: POLE 손상 및 Mismatch-repair deficiency(MMR)의 결합.
     - $M_2$: 산화 스트레스(oxidative stress) 및 BRCA1/2 기능 장애 관련.
     - $M_3$: 전사 가닥 편향(transcriptional strand bias) 및 노화, 아리스톨로카산 노출 관련.
     - $M_4$: 자외선(UV), 화학요법(thiopurine) 등 다양한 외인성 요인의 결합.

## 🧠 Insights & Discussion

### 강점
- **강건한 추론**: fully Bayesian 접근 방식을 통해 데이터가 희소한 의료 환경에서도 과적합에 강하며, 모든 추론 결과에 불확실성 추정치를 함께 제공한다.
- **계층적 해석력**: 단순한 차원 축소를 넘어, 데이터의 저수준 특징에서 고수준 개념으로 이어지는 계층적 구조를 학습함으로써 생물학적으로 의미 있는 '메타-시그니처'를 발견할 수 있었다.

### 한계 및 논의
- **계산 복잡도**: Gibbs sampling 기반의 추론은 GPU를 사용하더라도 대규모 데이터셋에서 연산 시간이 매우 오래 걸린다(실험에서 GPU 수십 일이 소요됨).
- **확장 가능성**: 논문에서는 이를 해결하기 위해 Approximate MCMC나 하이브리드 추론 방식의 도입이 필요함을 언급하며, 이를 향후 연구 과제로 남겨두었다.

### 비판적 해석
본 논문은 수학적으로 매우 우아한 추론 프레임워크를 제시하였으며, 특히 Theorem 1을 통해 복잡한 베이지안 네트워크의 샘플링을 가능하게 한 점이 돋보인다. 다만, 실용적인 관점에서는 학습 시간이 매우 길어 실제 임상 현장에서 실시간으로 적용하기에는 한계가 있을 것으로 보인다. 그럼에도 불구하고, 데이터 기반으로 생물학적 기전을 역으로 추적해낸 점은 큰 학술적 가치가 있다.

## 📌 TL;DR

본 논문은 의료 데이터의 희소성과 불확실성 문제를 해결하기 위해 심층 생성 베이지안 모델인 **Multinomial Belief Network (MBN)**를 제안한다. Dirichlet-Multinomial과 CRT 분포의 새로운 관계를 이용한 효율적인 Gibbs Sampling을 구현하였으며, 이를 통해 암 변이 데이터에서 생물학적으로 유의미한 **4가지 메타-시그니처**를 성공적으로 추출하였다. 이 연구는 향후 정밀 의료에서 변이 분석의 신뢰성을 높이고 불확실성을 정량화하는 데 중요한 기초가 될 것으로 기대된다.