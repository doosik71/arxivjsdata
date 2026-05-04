# ACCELERATING DIFFUSION MODELS VIA PRE-SEGMENTATION DIFFUSION SAMPLING FOR MEDICAL IMAGE SEGMENTATION

Xutao Guo, Yanwu Yang, Chenfei Ye, Shang Lu, Yang Xiang, Ting Ma (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 Denoising Diffusion Probabilistic Model (DDPM)을 적용할 때 발생하는 심각한 추론 효율성 저하 문제를 해결하고자 한다. 

DDPM 기반의 분할 방식은 의료 영상 분할을 조건부 이미지 생성 작업으로 정의함으로써, 픽셀 단위의 uncertainty map을 계산할 수 있고, 암시적인 앙상블(implicit ensemble)을 통해 분할 성능을 높일 수 있다는 강력한 장점이 있다. 특히 의료 진단에서는 단일 결과보다 여러 가능성(hypotheses)을 제공하는 것이 오진을 줄이는 데 중요하므로 DDPM의 특성은 매우 유용하다.

그러나 Vanilla DDPM은 가우시안 노이즈로부터 깨끗한 영상을 생성하기 위해 수백에서 수천 번의 반복적인 denoising 단계를 거쳐야 하며, 매 단계마다 신경망의 forward prediction이 필요하다. 이러한 특성은 실시간 진단이 중요한 의료 환경에서 추론 속도를 극도로 느리게 만들어 실제 적용에 큰 제약이 된다. 따라서 본 연구의 목표는 DDPM의 생성 품질을 유지하거나 오히려 향상시키면서도, 추론에 필요한 reverse step의 수를 획기적으로 줄이는 가속화 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Pre-segmentation Diffusion Sampling (PD-DDPM)**이다. 기존 DDPM이 아무런 정보가 없는 표준 가우시안 노이즈 $\mathcal{N}(0, I)$에서 시작하는 것과 달리, 별도로 학습된 분할 네트워크를 통해 얻은 **사전 분할 결과(pre-segmentation result)**를 시작점으로 사용하는 것이다.

구체적으로, 사전 분할 결과를 forward diffusion 규칙에 따라 일정 단계 $T'$까지 노이즈를 섞은 후, 이 지점부터 reverse process를 시작함으로써 전체 denoising 단계 수를 크게 단축한다. 이는 단순히 단계를 줄이는 것이 아니라, 분할 작업에 특화된 유효한 초기값을 제공함으로써 추론 속도 향상과 분할 정확도 개선이라는 두 마리 토끼를 동시에 잡으려는 설계이다.

## 📎 Related Works

논문에서는 DDPM의 추론 속도를 높이기 위한 기존의 여러 접근 방식을 소개한다.

1. **샘플링 단계 축소 및 스케줄링**: Non-Markovian reverse process를 사용하는 방법(Song et al.)이나, 추론 시 입력값에 따라 노이즈 파라미터를 추정하는 adaptive noise scheduling(San-Roman et al.) 등이 제안되었다.
2. **증류(Distillation)**: 전체 샘플링 프로세스를 더 빠른 샘플러로 증류하여 단계 수를 절반으로 줄이는 방식(Salimans & Ho)이 있다.
3. **잠재 공간(Latent Space) 활용**: 사전 학습된 autoencoder를 이용해 diffusion process를 latent space로 옮기는 방식(Vahdat et al., Rombach et al.)이 존재한다.
4. **프로세스 절단(Truncation)**: Forward 및 reverse process의 일부를 절단하여 효율을 높이는 방법(Zheng et al., Lyu et al.)이 있으나, 이는 GAN이나 VAE 모델과의 결합이 필요하여 학습이 어렵고 Vanilla DDPM의 이미지 차원 유지 특성을 깨뜨리는 한계가 있다.

본 논문의 PD-DDPM은 이러한 기존 방법들과 달리, 분할 작업의 특성을 직접적으로 활용하며 가우시안 가정을 깨뜨리지 않으면서도 효율적인 가속화를 달성한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Background: Vanilla DDPM
DDPM은 두 개의 마르코프 체인으로 구성된다.

- **Forward Diffusion Process**: 깨끗한 데이터 $x_0$에 점진적으로 가우시안 노이즈를 추가하여 표준 가우시안 분포에 수렴하게 만든다.
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$
- **Reverse Diffusion Process**: 학습된 신경망 $p_\theta$를 통해 가우시안 노이즈 $x_T$로부터 데이터를 복원한다.
$$p_\theta(x_{0:(T-1)}|x_T) = \prod_{t=1}^T p_\theta(x_{t-1}|x_t), \quad p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

### 2. PD-DDPM 파이프라인
PD-DDPM은 위 과정을 수정하여 추론 효율을 높인다.

1. **Pre-segmentation**: 먼저 별도로 학습된 분할 네트워크 $f_\psi$에 의료 영상 $I$를 입력하여 사전 분할 결과 $x_{pre}$를 얻는다.
$$x_{pre} = f(I; \psi)$$
2. **Diffusion to $T'$**: 얻어진 $x_{pre}$에 forward diffusion 규칙을 적용하여 $T'$ 단계의 노이즈 섞인 샘플 $\hat{x}_{T'}$를 생성한다. (여기서 $\hat{x}_{T'}$는 표준 가우시안 분포가 아닌 non-Gaussian 분포를 가진다.)
3. **Accelerated Reverse Sampling**: 이제 시작점을 $x_T$가 아닌 $\hat{x}_{T'}$로 설정하고, $T'$ 단계부터 $0$ 단계까지 denoising을 수행한다.
$$p_\theta(x_{0:(T'-1)}|\hat{x}_{T'}) = \prod_{t=1}^{T'-1} p_\theta(x_{t-1}|x_t) p_\theta(x_{T'-1}|\hat{x}_{T'})$$

이 방식은 Vanilla DDPM의 denoising 프로세스에 내재된 가우시안 가정을 깨뜨리지 않으면서도, 유의미한 정보가 포함된 지점에서 샘플링을 시작하므로 훨씬 적은 단계만으로도 고품질의 분할 결과를 얻을 수 있다.

## 📊 Results

### 실험 설정
- **데이터셋**: MICCAI 2017 WMH(White Matter Hyperintensities) 데이터셋 (Brain MRI T1 및 FLAIR 영상).
- **지표**: Dice score, HD95(95th percentile Hausdorff Distance), Jaccard index, F1 score.
- **구현**: AttUnet을 pre-segmentation 모델로 사용하였으며, $T=1000$ 단계의 cosine noise schedule을 적용하였다. 추론 시에는 5개의 마스크를 샘플링하여 평균을 내는 앙상블 방식을 사용하였다.

### 주요 결과
1. **정량적 성능 (Table 1)**:
   - PD-DDPM($T'=300$)은 Dice 0.812, HD95 3.494로 비교 대상인 U-Net, AttUnet, Vanilla DDPM 및 기타 가속화 모델(TDPM, ES-DDPM)보다 모든 지표에서 우수한 성능을 보였다.
2. **최적의 $T'$ 탐색 (Figure 3)**:
   - $T'$ 값에 따른 분석 결과, $T'=300$일 때 가장 높은 Dice score를 기록하였다. Uncertainty는 $T' < 500$ 구간에서 $T'$가 증가함에 따라 함께 증가하다가 이후 포화(saturate)되는 경향을 보였다.
3. **앙상블 크기의 효과 (Figure 4)**:
   - 단일 마스크보다 다중 마스크 앙상블의 성능이 우수했으며, 앙상블 크기가 커질수록 성능 향상이 이루어지다가 포화되었다. 본 연구에서는 최적의 효율을 위해 크기를 5로 설정하였다.
4. **사전 분할 정확도의 영향 (Table 2)**:
   - Pre-segmentation 모델의 Dice score가 높을수록 최종 PD-DDPM의 성능이 향상되는 양의 상관관계를 확인하였다. 이는 PD-DDPM이 기존의 고성능 분할 모델들과 결합하여 시너지를 낼 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 DDPM의 고질적인 문제인 추론 속도를 '분할 작업의 특성'을 활용해 해결하였다. 특히 주목할 점은 단순히 속도만 높인 것이 아니라, 오히려 Vanilla DDPM보다 더 높은 정확도를 달성했다는 점이다. 이는 무작위 노이즈에서 시작하는 것보다, 어느 정도 정답에 근접한 $x_{pre}$에서 시작하는 것이 모델이 최적의 솔루션을 찾는 데 긍정적인 가이드 역할을 했음을 의미한다.

또한, PD-DDPM은 기존의 어떤 분할 네트워크와도 결합 가능한 **직교적(orthogonal)** 성격을 가진다. 즉, 더 강력한 pre-segmentation 모델을 사용할수록 최종 결과가 좋아지므로, 최신 SOTA 분할 모델을 pre-segmentation 단계에 배치함으로써 성능을 지속적으로 끌어올릴 수 있는 확장성을 갖추고 있다.

다만, 본 논문에서 제시한 최적의 $T'=300$이라는 수치가 다른 데이터셋이나 다른 의료 영상 도메인에서도 동일하게 적용될지는 명시되지 않았으므로, 새로운 도메인 적용 시 $T'$에 대한 하이퍼파라미터 튜닝이 필수적일 것으로 판단된다.

## 📌 TL;DR

본 연구는 의료 영상 분할에서 DDPM의 느린 추론 속도를 해결하기 위해, 가우시안 노이즈 대신 **사전 학습된 분할 네트워크의 결과물($x_{pre}$)**을 시작점으로 사용하는 **PD-DDPM**을 제안하였다. 이를 통해 reverse sampling 단계 수를 획기적으로 줄였음에도 불구하고, Vanilla DDPM 및 기존 가속화 방법론들보다 더 뛰어난 분할 성능과 uncertainty 추정 능력을 보여주었다. 이 방법은 기존의 다양한 분할 모델과 결합 가능하여, 향후 의료 영상 분석의 효율성과 정확성을 동시에 높이는 데 기여할 가능성이 크다.