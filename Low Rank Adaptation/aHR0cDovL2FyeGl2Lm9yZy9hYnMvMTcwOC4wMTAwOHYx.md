# Beyond Low Rank: A Data-Adaptive Tensor Completion Method

Lei Zhang, Wei Wei, Qinfeng Shi, Chunhua Shen, Anton van den Hengel, and Yanning Zhang (2017)

## 🧩 Problem to Solve

본 논문은 다차원 데이터인 텐서에서 일부 누락된 항목을 복원하는 Tensor Completion 문제에 집중한다. 기존의 Tensor Completion 방법론들은 대부분 잠재 텐서(latent tensor)가 Low-rank 구조를 가진다는 가정을 기반으로 하지만, 실제 응용 분야에서는 다음과 같은 두 가지 핵심적인 난관에 봉착한다.

첫째는 Tensor Rank 결정 문제이다. 텐서의 Rank를 미리 알 수 없는 상황에서 이를 수동으로 설정하거나 행렬 기반의 Rank norm을 최소화하는 방식은 텐서 고유의 다차원 구조를 충분히 반영하지 못하며, 잘못된 Rank 추정은 결과적으로 Over-fitting으로 이어진다.

둘째는 실제 데이터의 Low-rank 가정 준수 여부이다. 대부분의 실제 데이터는 엄격한 Low-rank 조건을 만족하지 않으며, Low-rank 구조와 더불어 복잡한 Non-low-rank 구조가 공존한다. 기존 연구들은 Non-low-rank 성분을 단순히 무시($E=0$)하거나 매우 희소(sparse)하다고 가정했지만, 실제 데이터의 Non-low-rank 성분은 Heavy-tailed 또는 Multimodal과 같은 복잡한 분포를 띠는 경우가 많아 정교한 모델링이 필요하다.

따라서 본 논문의 목표는 Tensor Rank를 자동으로 결정하고, 데이터의 특성에 맞게 Low-rank와 Non-low-rank 구조를 동시에 적응적으로 모델링할 수 있는 Data-adaptive Tensor Completion 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 잠재 텐서 $L$을 Low-rank 구조인 $X$와 Non-low-rank 구조인 $E$의 합으로 명시적으로 분리하여 표현하는 것이다 ($L = X + E$).

1. **Sparsity-induced Low-rank Model**: CP Factorization을 기반으로 새로운 Tensor Rank를 정의하고, 가중치 벡터 $\lambda$에 Sparsity-induced prior를 적용하여 Tensor Rank를 자동으로 결정하도록 설계하였다. 특히 Reweighted Laplace prior를 도입하여 중요한 구조는 보존하고 불필요한 성분은 효과적으로 제거한다.
2. **Mixture of Gaussians (MOG) for Non-low-rank Structure**: $E$의 분포가 매우 다양할 수 있다는 점에 착안하여, 이를 Mixture of Gaussians로 모델링하였다. MOG는 범용적인 근사 능력을 갖추고 있어 Zero, Sparse, 혹은 그 이상의 복잡한 분포를 모두 수용할 수 있는 데이터 적응적(data-adaptive) 특성을 제공한다.
3. **Bayesian MMSE Framework**: MAP(Maximum A Posteriori) 추정의 한계를 극복하기 위해 Bayesian Minimum Mean Squared Error (MMSE) 프레임워크를 채택하였다. Gibbs Sampler를 통해 각 변수의 사후 분포(posterior distribution)로부터 샘플을 추출함으로써, 단순한 점 추정이 아닌 사후 평균과 불확실성을 함께 추론한다.

## 📎 Related Works

기존의 Tensor Completion 연구는 크게 두 가지 방향으로 나뉜다.

1. **Completion Models**: 텐서를 여러 개의 행렬로 Unfolding한 후, Matrix rank norm(예: Nuclear norm)을 최소화하는 방식이다. 하지만 이 방식은 텐서의 다차원 구조를 완전히 캡처하지 못한다는 한계가 있다.
2. **Factorization Models**: 텐서를 고정된 Rank를 가진 여러 팩터로 분해하는 방식이다. 그러나 적절한 Tensor Rank를 수동으로 설정하기 어렵고, 잘못된 Rank 설정은 성능 저하를 야기한다.

또한, Non-low-rank 성분 $E$에 대한 처리 방식에서도 기존 연구들은 $E=0$으로 가정하거나 매우 단순한 Sparse 모델만을 사용하였다. 일부 연구에서 MOG를 사용한 사례가 있으나, 이는 주로 관측 노이즈를 모델링하는 데 그쳤으며 본 논문처럼 잠재 텐서 내부의 Non-low-rank 구조 자체를 모델링한 시도는 처음이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
본 모델은 관측 모델 $Y_\Omega = L_\Omega + M_\Omega$에서 시작하며, 여기서 $M_\Omega$는 Gaussian white noise이다. 잠재 텐서 $L$은 다음과 같이 분해된다.
$$L = X + E$$
여기서 $X$는 Low-rank 구조를, $E$는 Non-low-rank 구조를 나타낸다.

### 2. Low-rank 구조 모델링 ($X$)
$X$는 CP Factorization을 통해 $R$개의 rank-one 텐서의 합으로 표현된다.
$$X = \sum_{r=1}^{R} \lambda_r u_r^{(1)} \circ \cdots \circ u_r^{(K)}$$
- **Sparsity-induced Rank**: 본 논문은 $\lambda$ 벡터의 $\ell_0$ norm을 Rank로 정의한다. $\lambda$에 Two-level reweighted Laplace prior를 부여하여 $\lambda$의 sparsity를 유도함으로써 Tensor Rank를 자동으로 결정한다.
- **Reweighted Laplace Prior**: 가중치 $\lambda$가 Gaussian 분포를 따르고, 그 분산 $\gamma$가 Gamma 분포를 따르도록 설계하여, 큰 가중치는 적게 수축시키고 작은 가중치는 강하게 수축시켜 중요한 구조를 보존한다.
- **Factor Matrix Regularization**: 각 팩터 행렬 $U^{(k)}$의 원소들이 Gaussian 분포를 따르도록 하여 $\ell_2$ norm 제약을 부여하고, 하이퍼파라미터 $\mu^{(k)}, \tau^{(k)}$에 대해서도 공액 사전 분포(conjugate prior)를 설정하여 베이지안 추론이 가능하게 하였다.

### 3. Non-low-rank 구조 모델링 ($E$)
$E$의 각 원소 $e_i$는 $D$개의 Gaussian 성분으로 구성된 혼합 모델(MOG)을 따른다.
$$e_i \sim \sum_{d=1}^{D} \pi_d \mathcal{N}(e_i | \mu_d, \tau_d^{-1})$$
- $\pi_d$는 혼합 비율이며, Dirichlet 분포를 사전 분포로 가진다.
- $\mu_d, \tau_d$는 각 Gaussian 성분의 평균과 정밀도(precision)이며, Gaussian-gamma 분포를 따른다.
- 이를 통해 $E$가 Sparse 하거나 Multimodal 한 경우 등 다양한 데이터 분포에 유연하게 적응할 수 있다.

### 4. 추론 절차 (Inference)
본 모델은 MAP 추정 대신 MMSE 추정을 사용하며, 최종 복원 텐서 $\hat{L}$은 사후 분포의 기대값 $\mathbb{E}[L | Y_\Omega]$으로 정의된다.
$$\hat{L} = \text{arg min}_{\tilde{L}} \int \|\tilde{L} - L\|_F^2 p(L | Y_\Omega) dL$$
실제 계산을 위해 Gibbs Sampling을 사용하여 $\lambda, \gamma, \kappa, U, \mu, \tau, E, z, \pi$ 등의 모든 변수를 순차적으로 샘플링하고, 수집된 샘플들의 평균을 통해 최적의 $\hat{L}$을 얻는다.

### 5. 공간적 유사성 제약 (Spatial Similarity Constraint)
시각적 데이터(이미지, 비디오)의 경우 인접 픽셀 간의 유사성이 높다는 점을 이용해, 팩터 행렬 $U^{(k)}$의 인접한 행들이 서로 유사하도록 하는 추가적인 Prior를 도입하여 복원 정확도를 더욱 향상시켰다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 합성 텐서(Synthetic tensor), 벤치마크 이미지 10종, 비디오('suzie', 'foreman'), CMU-PIE 얼굴 데이터셋.
- **비교 대상**: FaLRTC, HaLRTC, RPTC, TMac, STDC, t-SVD, FBCP, BRTF 등 8가지 최신 기법.
- **평가 지표**: Relative Reconstruction Error (RRE), PSNR, SSIM.

### 2. 주요 결과
- **Rank 결정 및 PDF 피팅**: 합성 데이터 실험 결과, 제안 모델은 누락률(missing ratio)이 90%에 달하는 상황에서도 실제 Tensor Rank를 정확하게 추정하였다. 또한 MOG를 통해 $E$의 복잡한 확률 밀도 함수(PDF)를 매우 정교하게 복원함을 확인하였다.
- **합성 데이터 복원**: 다양한 $E$ (Zero, Sparse, Mixture) 설정에서 제안 모델이 가장 낮은 RRE를 기록하며 압도적인 성능을 보였다. 특히 $E$가 복잡할수록 기존 방법론과의 성능 격차가 커졌다.
- **실제 응용**:
    - **Image Inpainting**: 90%의 높은 누락률에서도 세부 디테일을 가장 잘 복원하였으며, PSNR과 SSIM 지표에서 우위를 점했다.
    - **Video Completion**: 비디오 프레임의 손상된 부분을 복원함에 있어 타 방법론보다 더 선명하고 정교한 결과를 생성하였다.
    - **Facial Synthesis**: CMU-PIE 데이터셋에서 누락된 얼굴 이미지를 합성할 때 아티팩트가 적고 디테일이 살아있는 결과를 얻었다.

## 🧠 Insights & Discussion

본 논문은 텐서의 Low-rank 성분과 Non-low-rank 성분을 명시적으로 분리하고, 각각에 대해 적절한 확률 모델을 적용함으로써 Tensor Completion의 강건성을 높였다.

**강점**:
- **유연한 모델링**: MOG를 도입함으로써 실제 데이터가 가질 수 있는 임의의 분포를 수용할 수 있게 되었다. 이는 "실제 데이터는 완벽한 Low-rank가 아니다"라는 실용적인 관점을 수학적으로 잘 풀어낸 결과이다.
- **자동화된 Rank 추정**: Sparsity-induced prior를 통해 사용자가 Rank를 수동으로 설정해야 하는 번거로움과 그로 인한 Over-fitting 위험을 제거하였다.
- **통계적 추론**: MMSE 프레임워크와 Gibbs Sampling을 통해 점 추정의 불안정성을 극복하고 확률적인 평균값을 도출함으로써 복원 성능을 극대화하였다.

**한계 및 논의**:
- **계산 복잡도**: Gibbs Sampling은 반복적인 샘플링 과정이 필요하므로, 매우 거대한 텐서 데이터에 적용할 때 수렴 속도와 계산 비용 문제가 발생할 수 있다. (논문에서는 $O(RKN)$으로 선형 복잡도임을 주장하지만, 반복 횟수 $N_b, N_s$에 따른 총 시간은 상당할 수 있다.)
- **하이퍼파라미터**: 비정보적(non-informative) 사전 분포를 사용했다고 하지만, MOG의 성분 개수 $D$나 초기 Rank $R$ 설정이 결과에 어느 정도 영향을 줄 가능성이 있다.

## 📌 TL;DR

본 논문은 잠재 텐서를 Low-rank($X$)와 Non-low-rank($E$) 구조로 분해하여 모델링하는 새로운 Tensor Completion 방법을 제안한다. CP Factorization의 가중치에 Sparsity-induced prior를 적용해 Tensor Rank를 자동으로 결정하며, $E$의 복잡한 분포를 Mixture of Gaussians(MOG)로 적응적으로 피팅한다. 베이지안 MMSE 프레임워크와 Gibbs Sampling을 통해 복원 성능을 높였으며, 이미지 및 비디오 복원 등 실제 응용 분야에서 기존 SOTA 방법론들을 능가하는 성능을 입증하였다. 이 연구는 실제 세계의 불완전한 다차원 데이터를 복원하는 데 있어 매우 중요한 이론적, 실천적 토대를 제공한다.