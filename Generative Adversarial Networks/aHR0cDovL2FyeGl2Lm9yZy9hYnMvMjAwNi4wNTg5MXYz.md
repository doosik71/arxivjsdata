# Understanding Noise Injection in GANs

Ruili Feng, Deli Zhao, Zhengjun Zha (2021)

## 🧩 Problem to Solve

본 논문은 Generative Adversarial Networks (GANs)에서 고해상도 이미지 생성 시 사용되는 Noise Injection 기술의 이론적 배경과 메커니즘을 규명하고자 한다. 특히 StyleGAN과 같은 최신 모델에서 Noise Injection이 이미지의 세부 디테일을 향상시키는 데 매우 효과적임에도 불구하고, 왜 이러한 현상이 발생하는지에 대한 수학적 설명이 부족했다는 점에 주목한다.

저자들은 GAN의 생성기(Generator)가 가진 내재적인 결함인 **Adversarial Dimension Trap**을 해결하는 것이 핵심 문제라고 정의한다. 생성기의 표현력은 Jacobian 행렬의 Rank에 의해 제한되며, 네트워크가 깊어질수록 이 Rank가 단조 감소하여 실제 데이터 매니폴드(Data Manifold)의 내재적 차원($d_x$)을 충분히 표현하지 못하게 된다. 이러한 차원의 불일치는 생성기의 불연속성(non-smoothness)을 초래하거나, 데이터 분포를 제대로 캡처하지 못해 이미지 반전(Inversion)에 실패하는 등의 문제를 일으킨다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Noise Injection을 Riemannian Geometry(리만 기하학) 관점에서 해석하여 이론적 프레임워크를 제시하고, 이를 기반으로 일반화된 Noise Injection 방법론을 제안한 것이다.

1. **이론적 분석**: 생성기의 Jacobian Rank가 데이터 매니폴드의 차원보다 작을 때 발생하는 **Adversarial Dimension Trap**을 수학적으로 증명하였다.
2. **기하학적 프레임워크 제안**: Riemannian Geometry의 Exponential Map을 이용하여, 생성기가 저차원의 '뼈대(Skeleton)'만을 학습하고, 주입된 노이즈가 나머지 '살(Flesh)'을 채우는 방식으로 고차원 매니폴드를 근사할 수 있음을 보였다.
3. **Riemannian Noise Injection (RNI) 구현**: 단순한 유클리드 공간의 노이즈 주입(ENI)을 넘어, 특징 맵의 국소적 기하 구조(Local Geometry)를 반영하는 $\sigma(x)$를 학습하여 주입하는 RNI 방법론을 제안하였다.

## 📎 Related Works

기존의 Noise Injection 연구들은 주로 학습 안정화(Regularization)나 과적합 방지를 위한 수단으로 활용되었다. 예를 들어, Arjovsky & Bottou (2017)는 이미지 공간에 노이즈를 추가하여 분포를 부드럽게 함으로써 학습을 안정화하는 방법을 제안하였다. BigGAN은 잠재 벡터를 레이어별로 나누어 Batch Normalization의 Gain과 Bias에 투영하는 방식을 사용하였다.

StyleGAN과 StyleGAN2는 각 레이어에 독립적인 노이즈를 주입하여 다중 스케일의 확률적 변이(Stochastic Variations)를 생성하고 세부 디테일을 향상시켰다. 그러나 이러한 기존 방식들은 노이즈 주입이 특징 공간의 기하학적 구조와 어떤 관계가 있는지에 대한 이론적 설명이 부족했으며, 단순히 경험적인 설계에 의존하였다는 한계가 있다. 본 논문은 이를 리만 기하학으로 정식화하여 이론적 근거를 마련했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Adversarial Dimension Trap의 정의
생성기 $G: Z \to X$에서 생성된 매니폴드 $G^d_g$의 내재적 차원 $d_g$는 Jacobian 행렬 $J_z^G$의 Rank와 같다. 본 논문은 다음과 같은 정리를 제시한다.

**Theorem 1**: 만약 $\text{rank}(J_z^G) < d_x$라면, 다음 두 경우 중 적어도 하나가 발생한다.
1. $\sup_{z \in Z} \|J_z^G\| = \infty$ (그래디언트 폭주로 인한 불안정성)
2. 생성기가 데이터 분포를 캡처하지 못하며, 임의의 점 $x \in X$에 대해 $G^{-1}(x) = \emptyset$일 확률이 1이다. 또한 Jensen-Shannon Divergence $D_{JS}(P_g, P_r) \ge \frac{\log 2}{2}$가 성립한다.

### 2. Riemannian Noise Injection (RNI) 프레임워크
저자들은 생성기가 직접 고차원 매니폴드를 학습하는 대신, 저차원의 뼈대 $\mu(x)$를 학습하고 노이즈 $\epsilon$을 통해 주변 영역을 채우는 2단계 절차를 제안한다.

$$g_k(x) = \mu_k(x) + \sigma_k(x)\epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

여기서 $\mu_k(x)$는 특징 공간의 중심점(Representative point)이고, $\sigma_k(x)$는 해당 지점의 국소적 기하 구조를 결정하는 가중치 행렬이다. 이는 리만 매니폴드에서 Exponential Map을 통해 접공간(Tangent Space)의 유클리드 볼을 매니폴드 위의 지오데식 볼(Geodesic Ball)로 사영하는 과정을 근사한 것이다.

### 3. $\sigma(x)$의 기하학적 구현 (Geometric Realization)
$\sigma(x)$를 단순히 상수로 두지 않고, 특징 맵 $\mu$의 공간적/의미적 정보에서 유도한다.

1. **의미적 식별자 추출**: 특징 맵 $\mu$의 채널 방향 합을 구하여 $\tilde{\mu}$를 생성한다.
   $$\tilde{\mu}_{jk} = \sum_{i=1}^{c} \mu_{ijk}$$
2. **정규화**: $\tilde{\mu}$를 평균 제거 및 최대값으로 나누어 $\text{mean}(\tilde{\mu})=0, \max(|s|)=1$인 $s$를 얻는다.
3. **성분 분해 및 안정화**: 학습 가능한 파라미터 $A, b$를 이용한 아핀 변환을 통해 주 콘텐츠 성분 $s_m$과 노이즈 유도 변이 성분 $s_v$로 분리한다.
   $$s' = \alpha s_d + (1-\alpha)\mathbf{1}, \quad \sigma = \frac{s'}{\|s'\|_2}$$
4. **최종 출력**:
   $$o = r\sigma * \mu + r\sigma * \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

여기서 $r$과 $\alpha$는 학습 가능한 파라미터이다. StyleGAN2에서 사용하는 방식은 $\sigma$를 Identity 행렬로 간주하는 **Euclidean Noise Injection (ENI)**의 특수 사례로 볼 수 있다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: FFHQ (얼굴), LSUN-Church (건물), CIFAR-10 (일반 객체)
- **비교 대상**: Plain StyleGAN2, StyleGAN2 + ENI, DCGAN $\pm$ ENI/RNI
- **지표**: PPL (Path Perceptual Length), FID (Fréchet Inception Distance), Condition Number (MC, TTMC)

### 2. 정량적 결과
- **이미지 품질**: Table 1에 따르면, RNI는 FFHQ와 LSUN-Church 데이터셋 모두에서 ENI보다 우수한 PPL 및 FID 점수를 기록하였다. 특히 LSUN-Church와 같은 복잡한 씬 이미지에서 RNI의 성능 향상이 두드러졌는데, 이는 RNI가 생성기에 더 많은 자유도를 제공하여 실제 분포를 더 잘 맞추기 때문으로 분석된다.
- **수치적 안정성**: Condition Number를 통해 네트워크의 민감도를 측정한 결과, RNI가 가장 낮은 MC(Mean Condition)와 TTMC(Top Thousand Mean Condition) 값을 보였다. 이는 RNI가 생성기의 수치적 안정성을 높여 학습 시 그래디언트 폭주 위험을 줄였음을 의미한다.
- **GAN Inversion**: 이미지-잠재 공간 투영 능력을 평가한 결과, 정면 얼굴과 같은 쉬운 케이스에서는 차이가 적었으나, 큰 포즈 변화나 가려짐(Occlusion)이 있는 **Hard Cases**에서 RNI가 가장 낮은 MSE와 Perceptual Loss를 기록하며 압도적인 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 Noise Injection이 단순히 '무작위성'을 더하는 것이 아니라, 생성기가 도달하기 어려운 고차원 매니폴드의 빈 공간을 기하학적으로 메우는 역할을 한다는 점을 이론적으로 밝혀냈다.

**강점 및 해석**:
- **차원 함정 탈출**: $\text{rank}(J_z^G) < d_x$인 상황에서도 노이즈 주입을 통해 유효한 매니폴드를 근사함으로써 Adversarial Dimension Trap을 회피할 수 있음을 증명하였다.
- **적응적 노이즈**: 모든 영역에 동일한 노이즈를 주는 대신, 이미지의 의미적 특성($\sigma(x)$)에 따라 노이즈의 강도와 방향을 조절함으로써 세부 디테일(머리카락, 배경 등)을 훨씬 정교하게 생성할 수 있게 되었다.

**한계 및 논의**:
- $\sigma(x)$를 유도하는 과정에서 사용된 채널 합 기반의 휴리스틱이 모든 도메인의 GAN 아키텍처에 보편적으로 적용 가능한지에 대해서는 추가 연구가 필요하다.
- 본 논문은 리만 기하학적 근사를 통해 문제를 해결했으나, 실제 구현에서는 유클리드 공간에서의 연산으로 근사하여 처리하므로 이론과 실제 구현 사이의 오차(approximation error $o(r)$)가 존재한다.

## 📌 TL;DR

GAN 생성기는 깊어질수록 Jacobian Rank가 감소하여 실제 데이터의 고차원 구조를 표현하지 못하는 **Adversarial Dimension Trap**에 빠지기 쉽다. 본 논문은 이를 해결하기 위해 리만 기하학(Riemannian Geometry)을 도입, 생성기가 저차원 뼈대를 학습하고 노이즈가 국소적 기하 구조에 맞게 공간을 채우는 **Riemannian Noise Injection (RNI)**을 제안하였다. 실험 결과 RNI는 기존 StyleGAN의 유클리드 노이즈 방식보다 이미지 품질(FID, PPL), 수치적 안정성, 그리고 이미지 반전(Inversion) 성능 면에서 모두 우수함을 입증하였다. 이 연구는 향후 GAN의 표현력을 높이기 위한 기하학적 설계 방향을 제시한다.