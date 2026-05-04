# Probabilistic Modeling of Inter- and Intra-observer Variability in Medical Image Segmentation

Arne Schmidt, Pablo Morales-Alvarez, Rafael Molina (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 작업에서 가장 큰 난제 중 하나는 숙련된 전문가들 사이에서도 라벨링 결과가 서로 다른 **Inter-observer variability**(관찰자 간 변동성)와 동일한 전문가가 동일한 데이터를 다른 시점에 평가할 때 발생하는 **Intra-observer variability**(관찰자 내 변동성)가 존재한다는 점이다.

이러한 변동성은 데이터에 단일한 정답(Ground Truth)이 존재하지 않음을 의미하며, 기존의 AI 모델들이 이러한 불확실성을 충분히 반영하지 못한 채 학습될 경우 진단 과정에서 잘못된 예측에 의존하게 되는 위험을 초래할 수 있다. 따라서 본 논문의 목표는 의료 영상 분할에서 관찰자 간 및 관찰자 내 변동성을 명시적으로 모델링하여 예측 성능을 높이고, 신뢰할 수 있는 불확실성 추정치를 제공하는 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문은 **Pionono**(Probabilistic Inter-Observer and iNtra-Observer variation NetwOrk)라는 새로운 확률론적 딥러닝 모델을 제안한다. 이 모델의 핵심 아이디어는 **각 라벨러(Rater)의 라벨링 행동을 잠재 공간(Latent Space)에서의 다차원 확률 분포로 표현**하는 것이다.

- **Inter-observer variability**는 서로 다른 라벨러들의 확률 분포 간의 차이(Distance/Overlap)를 통해 모델링한다.
- **Intra-observer variability**는 개별 라벨러 확률 분포의 분산(Variance)을 통해 모델링한다.
- 이를 통해 모델은 단순히 하나의 예측값을 내놓는 것이 아니라, "특정 전문가 X라면 이 영상을 어떻게 분할했을 것인가"를 시뮬레이션할 수 있는 능력을 갖춘다.

## 📎 Related Works

### 기존 연구 및 한계

1. **Probabilistic Deep Learning**: Bayesian Neural Networks나 Sparse Gaussian Processes 등이 불확실성을 다루어 왔으며, 특히 **Probabilistic U-Net**은 잠재 변수를 통해 라벨링의 일반적인 변동성을 인코딩하였다. 하지만 이는 라벨러 개개인의 특성(Rater identity)을 명시적으로 모델링하지 못하므로, 특정 전문가의 의견을 시뮬레이션하는 것이 불가능하다.
2. **Crowdsourcing & Label Fusion**: 여러 명의 라벨링 결과를 하나로 합치는 **STAPLE**과 같은 기법이 사용되었다. 또한 **Confusion Matrices (CM)** 기반의 접근 방식(Global/Pixel-wise CM)이 제안되었으나, 픽셀 간의 통계적 독립성을 가정함으로써 출력 결과의 일관성(Coherence)이 떨어지며, 라벨러 수가 늘어날 때 모델 크기가 비대해지는 확장성(Scalability) 문제가 있다.

### Pionono의 차별점

Pionono는 확률론적 딥러닝과 크라우드소싱 방법론의 장점을 결합하였다. 잠재 공간의 분포를 사용함으로써 출력의 일관성을 유지하면서도, 각 라벨러의 고유한 특성을 효율적으로 학습하여 확장성을 확보하고 전문가 의견 시뮬레이션 기능을 제공한다.

## 🛠️ Methodology

### 전체 시스템 구조

Pionono의 파이프라인은 크게 특성 추출기(Backbone), 라벨러별 잠재 분포(Latent Distributions), 그리고 분할 헤드(Segmentation Head)로 구성된다.

1. **Feature Extraction**: 이미지 $x_i$가 입력되면 U-Net 아키텍처(ResNet34 backbone)를 통해 특성 맵 $v_i \in \mathbb{R}^{H \times W \times L}$를 추출한다.
   $$v_i = f_\omega(x_i)$$
2. **Rater Latent Space**: 각 라벨러 $r$의 라벨링 행동을 나타내는 확률 분포 $q(z|r)$를 정의한다. 잠재 변수 $z \in \mathbb{R}^D$ (여기서 $D=8$)를 사용하며, 각 라벨러는 다음과 같은 다변량 가우시안 분포를 가진다.
   $$q(z|r) = \mathcal{N}(z|\mu_r, \Sigma_r)$$
   여기서 $\mu_r$과 $\Sigma_r$은 학습 가능한 파라미터이다.
3. **Segmentation Head**: 분할 헤드 $f_\theta$는 이미지 특성 맵 $v_i$와 잠재 변수 $z$에서 샘플링된 $\tilde{z}_j$를 결합(Concatenate)하여 입력받아 최종 분할 맵을 생성한다.
   $$\tilde{s}_{i,j} = f_\theta(v_i, \tilde{z}_j), \quad \tilde{z}_j \sim q(z|r)$$

### 훈련 목표 및 손실 함수

모델은 **Variational Inference**를 통해 최적화되며, 목적 함수로 **ELBO**(Evidence Lower BOund)를 최대화한다. 손실 함수 $\mathcal{L}_{ELBO}$는 다음과 같이 정의된다.
$$\mathcal{L}_{ELBO} = \mathbb{E}_q \log p(S^r | X, r, \omega, \theta) - \lambda KL(q(Z|r) || p(Z))$$

- **Log-Likelihood (LL) term**: 모델의 예측이 각 라벨러 $r$의 실제 어노테이션 $S^r$과 일치하도록 유도한다. 본 논문에서는 일반적인 Dice Loss를 사용하여 최적화하였다.
- **KL Divergence term**: 학습된 사후 분포 $q(Z|r)$가 사전 분포 $p(Z) = \mathcal{N}(z|0, \sigma_{prior}^2 I)$에서 너무 벗어나지 않도록 규제(Regularization)하는 역할을 한다.

### 학습 절차

- **Reparameterization Trick**: 확률적 샘플링 과정을 미분 가능하게 만들어 $\mu_r, \Sigma_r$ 및 네트워크 가중치 $\omega, \theta$를 End-to-End로 학습시킨다.
- **수치적 안정성**: 공분산 행렬 $\Sigma_r$을 직접 학습하는 대신 Cholesky 분해 $\Sigma_r = L_r L_r^\top$를 통해 하삼각행렬 $L$을 학습시킨다.
- **최적화**: Adam optimizer를 사용하며, 라벨러 분포 파라미터($\mu_r, \Sigma_r$)에는 CNN 가중치보다 더 높은 학습률($\nu=0.02$)을 적용하여 충분한 최적화가 이루어지게 한다.

### 추론 및 예측

- **Gold Prediction**: 전문가 합의(Expert agreement)를 예측하기 위해 별도의 'Gold' 분포 $q(z|r=M+1)$를 학습시키며, 여기서 샘플링된 예측값들의 평균을 통해 최종 결과를 낸다.
- **Uncertainty Estimation**: 샘플링된 여러 예측 결과의 분산을 통해 예측의 불확실성을 측정한다.
- **Expert Simulation**: 특정 라벨러 $r'$의 분포 $q(z|r')$에서 샘플링하여 해당 전문가가 내렸을 법한 분할 결과를 시뮬레이션한다.

## 📊 Results

### 실험 설정

- **데이터셋**: prostate cancer (Gleason 2019, Arvaniti TMA) 및 breast cancer (bc segmentation)의 병리 영상 데이터셋을 사용하였다.
- **지표**: Cohen's kappa ($\kappa$, unweighted 및 quadratic weighted)와 Accuracy를 사용하였다.
- **비교 대상**: STAPLE, Probabilistic U-Net, CM global, CM pixel.

### 주요 결과

1. **정량적 성능**:
   - Gleason 2019 데이터셋에서 Pionono는 $\kappa=0.758, \text{Acc}=0.84$를 기록하며 기존 챌린지 우승 모델 및 타 베이스라인 모델들을 크게 상회하였다.
   - 외부 데이터셋(Arvaniti TMA)을 이용한 일반화 성능 테스트에서도 대부분의 지표에서 타 모델보다 우수한 성능을 보였다.
   - 유방암 분할 데이터셋에서도 $\kappa=0.711$로 가장 높은 성능을 달성하여, 제안 방법론의 범용성을 입증하였다.

2. **정성적 분석 및 시뮬레이션**:
   - **불확실성 시각화**: 예측이 틀린 영역(예: G3 $\rightarrow$ G4 오분류)에서 높은 불확실성이 나타남을 확인하여, 진단 보조 도구로서의 가치를 보여주었다.
   - **라벨러 모델링**: 학습 후 잠재 공간의 분포를 분석한 결과, 라벨링 경향이 비슷한 전문가들의 분포는 서로 겹쳐 있고, 독특한 경향(예: 과소 분할)을 가진 전문가의 분포는 멀리 떨어져 있음이 확인되었다.
   - **전문가 시뮬레이션**: 특정 라벨러의 특성(예: 특정 클래스 G5를 더 많이 할당하는 경향)을 모델이 성공적으로 모사하여 개별 전문가의 의견을 일관되게 생성해낼 수 있음을 보였다.

## 🧠 Insights & Discussion

### 강점

- **개별 전문성의 보존**: 기존 모델들이 여러 전문가의 의견을 평균 내어 정보를 희석시키는 것과 달리, Pionono는 각 전문가의 고유한 지식을 개별 분포로 보존하여 필요에 따라 호출할 수 있게 한다.
- **임상적 유용성**: 불확실성 지도를 제공함으로써 의료진이 어떤 부분을 재검토해야 하는지 알려줄 수 있으며, 서로 다른 전문가의 의견을 시뮬레이션함으로써 의사결정 지원 시스템으로서의 확장성이 높다.
- **효율성 및 확장성**: 새로운 라벨러가 추가되어도 매우 작은 크기의 벡터($\mu_r$)와 행렬($\Sigma_r$)만 추가하면 되므로 연산 비용이 매우 낮다.

### 한계 및 비판적 해석

- **세부 형상 캡처의 한계**: 일부 클래스(NC, G4)에서 라벨러 간의 미세한 형상 차이를 완벽하게 캡처하지 못하고 유사한 형태로 예측하는 경향이 있다. 저자는 이를 해결하기 위해 더 넓은 커널을 가진 segmentation head를 사용하는 방안을 제시하였다.
- **데이터 의존성**: 본 연구는 주어진 데이터셋 내에서 라벨러의 특성을 학습하므로, 학습 데이터에 포함되지 않은 새로운 성향의 라벨러에 대해서는 추가 학습이 필요하다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 발생하는 **관찰자 간/내 변동성(Inter/Intra-observer variability)**을 해결하기 위해, 각 라벨러의 행동을 **잠재 공간의 가우시안 분포로 모델링하는 Pionono**를 제안한다. 이 모델은 기존의 합의 기반 방식보다 높은 분할 정확도를 보일 뿐만 아니라, **특정 전문가의 진단 성향을 시뮬레이션**하고 **예측 불확실성을 정량적으로 제공**할 수 있다. 이는 단순한 자동 분할을 넘어, 의료 전문가의 다양한 의견을 존중하고 보조하는 정밀한 진단 지원 도구로 활용될 가능성이 매우 높다.
