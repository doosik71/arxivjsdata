# A Comprehensive Survey on Knowledge Distillation of Diffusion Models

Weijian Luo (2023)

## 🧩 Problem to Solve

딥러닝 기반 생성 모델(Deep Generative Models)은 사실적인 데이터를 생성하는 것을 목표로 한다. 이 중 확산 모델(Diffusion Models, DMs)은 스코어 함수(score functions)를 직접 모델링하여 유연하고 높은 표현력을 가진다. DMs는 기저 분포의 미세한 지식(marginal score functions)을 학습할 수 있으며, 이미지 합성 및 편집, 오디오 및 분자 합성, 3D 객체 생성 등 다양한 분야에서 탁월한 성능을 보이며 선도적인 생성 모델로 부상하였다. 그러나 DMs의 샘플링 과정은 많은 신경 함수 평가(Neural Function Evaluations, NFEs)를 요구하여 계산 효율성이 떨어진다는 문제가 있다. 예를 들어, FID(Frechet Inception Distance) 벤치마크에서 DMs는 우수한 성능을 달성했지만, 이러한 성능을 위해서는 수백에서 수천 단계의 샘플링이 필요하며 이는 실시간 애플리케이션에 적용하기 어렵게 만든다.

이 논문의 목표는 이러한 확산 모델의 "지식"을 어떻게 증류(distill)하고 활용할 것인가에 대한 포괄적인 개요를 제공하는 것이다. 구체적으로는 DMs의 샘샘플링 효율성을 크게 가속화하고, DMs와 다른 생성 모델(예: 암시적 생성 모델, 정규화 흐름) 간의 연결성을 구축하여 향후 연구에 기여하는 것을 목표로 한다.

## ✨ Key Contributions

이 논문의 핵심 아이디어는 확산 모델(DMs)의 지식 증류(Knowledge Distillation) 기법들을 체계적으로 분류하고 설명함으로써, DMs의 샘플링 효율성 문제를 해결하고 DMs의 잠재력을 최대한 활용하는 현대적 접근 방식들을 종합적으로 조망하는 것이다. 구체적인 기여는 다음과 같다.

1. **확산 모델 지식 증류의 포괄적 분류**: 논문은 DMs의 지식 증류 방법을 세 가지 주요 범주로 분류한다:
   * **Diffusion-to-Field (D2F) 증류**: DMs의 생성 ODE(Ordinary Differential Equation)를 더 적은 NFE로 샘플을 생성할 수 있는 생성 벡터 필드(generative vector field)로 증류한다.
   * **Diffusion-to-Generator (D2G) 증류**: DMs가 학습한 분포 지식을 효율적인 암시적 생성기(implicit generator)로 전달한다.
   * **학습 없는(Training-free) 가속 샘플링 알고리즘**: 새로운 모델 학습 없이 기존 DMs의 추론 효율성을 개선하는 방법을 증류의 한 형태로 제시한다.
2. **DMs의 배경 및 과제 설명**: 확산 모델의 기본 개념과 함께 DMs를 신경 벡터 필드(neural vector fields)로 증류할 때 발생하는 과제에 대한 논의를 제공한다.
3. **다양한 증류 접근법 개요**: 확률적(stochastic) 및 결정론적(deterministic) 암시적 생성기(implicit generators) 모두에 DMs를 증류하는 기존 연구들을 개괄한다.
4. **튜토리얼로서의 역할**: 생성 모델에 대한 기본적인 이해를 가진 연구자들이 DMs의 증류를 적용하거나 이 분야의 연구 프로젝트를 시작하는 데 도움이 되는 튜토리얼을 제공한다.

중심적인 직관은 DMs의 뛰어난 생성 능력은 유지하면서도, 샘플링 속도를 극적으로 개선하기 위해 DMs의 "지식"을 더 작고 효율적인 "학생 모델" 또는 "효율적인 샘플링 메커니즘"으로 이전하는 것이다.

## 📎 Related Works

논문은 DMs의 지식 증류에 대한 논의를 시작하기 전에, DMs의 발전 배경을 이해하기 위해 선행 생성 모델인 자기회귀 모델(Auto-Regressive Models, ARMs), 에너지 기반 모델(Energy-Based Models, EBMs), 스코어 기반 모델(Score-Based Models, SBMs)을 소개한다.

### 자기회귀 모델 (ARMs)

* **설명**: ARMs는 신경망을 사용하여 데이터 분포의 로그 가능도 함수(log-likelihood function), 즉 확률 밀도 함수의 로그를 직접 매개변수화하고, 기저 데이터의 로그 가능도를 일치시키도록 학습된다. ARMs는 KL 발산 최소화를 통해 훈련된다.
  * 모델의 가능도 함수 $p_\theta(x)$는 조건부 요인화(conditional factorization)를 통해 다음과 같이 명시적으로 표현된다.
    $$ p_\theta(x) = f_\theta(x^{(1)}) + \sum_{i=2}^{D} f_\theta(x^{(i)}|x^{(1)},...,x^{(i-1)}) \quad (1) $$
    연속 데이터의 경우 $f_\theta(x^{(i)}|x^{(1)},...,x^{(i-1)})$는 가우시안 분포의 조건부 분포로 구현되며, 이의 평균과 분산은 신경망의 출력으로 얻어진다.
    $$ f_\theta(x^{(i)}|x^{(1)},...,x^{(i-1)}) = \log N(x^{(i)};\mu_\theta(x^{(1)},..,x^{(i-1)}),\Sigma_\theta(x^{(1)},..,x^{(i-1)})) \quad (2) $$
  * KL 발산 최소화는 $E_{p_d}[\log p_\theta(x)]$를 최대화하는 것과 동등하다.
* **한계**:
    1. 로그 가능도 함수를 명시적으로 모델링하므로 구현의 유연성이 제한된다 (예: 연속 값 ARMs는 조건부 분포를 다변량 가우시안 분포로 제한).
    2. 샘플링 알고리즘에 엄격한 순차적 순서가 필요하여 계산적으로 비효율적이다.

### 에너지 기반 모델 (EBMs)

* **설명**: ARMs의 정규화(normalization) 제약을 극복하기 위해 제약 없는 신경망을 사용하여 데이터의 잠재 함수(potential functions), 즉 비정규화된 밀도 함수의 로그를 표현하고 일치시킨다.
  * EBMs는 모델 잠재 함수를 $E_\theta(x)$로 표현하며, 분포 $p_\theta(x) = e^{E_\theta(x)} / Z_\theta$ (여기서 $Z_\theta$는 정규화 상수) 형태를 갖는다.
  * 최대 우도 학습(Maximum Likelihood Estimation)을 사용하여 훈련하며, 이 과정에서 $p_\theta$로부터 샘플을 얻기 위해 MCMC(Markov Chain Monte Carlo) 알고리즘(예: Langevin dynamics)이 사용된다.
  * 기대 로그-가능도 기울기(expected log-likelihood gradient)는 다음과 같다.
    $$ \frac{\partial}{\partial\theta} L(\theta) = E_{p_d}[\frac{\partial}{\partial\theta}E_\theta(x)] - E_{p_\theta}[\frac{\partial}{\partial\theta}E_\theta(x)] \quad (13) $$
* **기존 접근 방식과의 차별점**: ARMs가 명시적 조건부 밀도의 구성 요소로 신경망을 사용하는 반면, EBMs는 제약 없는 신경망을 사용하여 잠재 함수를 모델링함으로써 신경망의 완전한 표현력을 활용한다.

### 스코어 기반 모델 (SBMs)

* **설명**: EBMs의 MCMC 방법 중 랑제방 동역학(Langevin dynamics, LD)에서 영감을 받아, 분포 $p(x)$의 잠재 함수 기울기, 즉 스코어 함수 $\nabla_x \log p(x)$를 신경망으로 학습한다.
  * SBMs는 신경 스코어 함수 $S_\theta(x)$가 기저 데이터 분포를 일치시키도록 훈련한다. $S_d(x) := \nabla_x \log p_d(x)$를 잘 일치시키면 LD 시뮬레이션에서 $\nabla_x \log p(x)$를 $S_\theta(x)$로 대체하여 샘플을 얻을 수 있다.
  * SBMs는 모델의 잠재 함수 $p_\theta$가 비가산적(intractable)이므로, ARMs 및 EBMs에 사용되는 KL 발산 대신 모델의 스코어 함수만 요구하는 피셔 발산(Fisher divergence)을 최소화하여 훈련한다.
    $$ D_F(p_d,p_\theta) = E_{p_d}[\frac{1}{2} \Vert S_d(x) - S_\theta(x) \Vert_2^2] \quad (16) $$
  * 피셔 발산 최소화는 스코어 매칭(Score Matching, SM) 최적화 문제와 동등하다.
    $$ L_{SM}(\theta) := E_{p_d}[\frac{1}{2} \Vert S_\theta(x) \Vert_2^2 + E_{p_d} \sum_{d=1}^D \frac{\partial s_\theta^{(d)}(x)}{\partial x^{(d)}}] \quad (18) $$
  * SM의 데이터 기울기(data gradient) 계산의 비효율성을 해결하기 위해, 노이즈 예측(denoising score matching, DSM) 목표 함수가 제안되었다.
    $$ L_{DSM}(\theta) := E_{x \sim p_d, \tilde{x} \sim p(\tilde{x}|x)} \Vert S_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x}|x) \Vert_2^2 \quad (19) $$
* **한계**: 단일 섭동 커널(perturbation kernel)을 사용하는 SBMs는 고차원 데이터에서 샘플링 시 데이터가 고차원 공간에 임베딩된 저차원 매니폴드 주변에 집중되는 문제에 취약하다.

### 확산 모델 (DMs)

* **설명**: SBMs의 한계를 극복하기 위해 다중 레벨 또는 연속 인덱스 스코어 네트워크 $S_\theta(x,t)$를 사용한다. 또한, 단일 섭동 커널 대신 확률 미분 방정식(Stochastic Differential Equations, SDEs)에 의해 유도된 조건부 전이 커널(conditional transition kernels) 패밀리를 사용하여 데이터를 섭동한다.
  * **순방향 확산 SDE (Forward Diffusion SDE)**: 데이터 $X_0 \sim p_d$에서 시작하여 노이즈를 점진적으로 추가하여 표준 가우시안 분포로 변환한다.
    $$ dX_t = F(X_t,t)dt + G(t)dW_t, \quad X_0 \sim p_d^{(0)}=p_d \quad (20) $$
  * **VP Diffusion (Variance Preserving)**:
    $$ dX_t = -\frac{1}{2}\beta(t)X_t dt + \sqrt{\beta(t)}dW_t, \quad t \in [0,T] \quad (21) $$
    조건부 전이 커널은 다음과 같은 명시적 표현을 갖는다.
    $$ p(x_t|x_0) = N(x_t; \sqrt{\alpha_t}x_0; (1-\alpha_t)I) \quad (22) $$
    여기서 $\alpha_t=e^{-\int_0^t \beta(s)ds}$이다.
  * **VE Diffusion (Variance Exploding)**:
    $$ dX_t = \sqrt{\frac{d\sigma^2(t)}{dt}}dW_t, \quad t \in [0,T] \quad (24) $$
    전이 커널은 다음과 같다.
    $$ p(x_t|x_0) = N(x_t; x_0, \sigma(t)I) \quad (25) $$
  * **훈련 방법**: 각 시간 $t$에서의 섭동 커널 $p_t(.|.)$을 사용하는 DSM의 가중 조합을 최소화한다.
    $$ L_{WDSM}(\theta) = \int_0^T w(t)L_{DSM}^{(t)}(\theta)dt \quad (27) $$
    $L_{DSM}^{(t)}(\theta) = E_{x_0 \sim p_d^{(0)}, x_t|x_0 \sim p_t(x_t|x_0)} \Vert S_\theta(x_t,t) - \nabla_{x_t} \log p_t(x_t|x_0) \Vert_2^2 \quad (28)$
    이 목표는 노이즈 예측 목표(noise-prediction objective)로 재구성될 수 있다.
  * **샘플링 전략**: 학습된 스코어 네트워크 $S_\theta(x,t)$는 순방향 SDE와 동일한 한계 분포를 공유하는 역방향 SDE 또는 ODE(Ordinary Differential Equation)의 수치 해법을 통해 샘플링에 사용된다.
    * **역방향 SDE**:
        $$ dX_t = [F(X_t,t) - G^2(t)\nabla_{x_t} \log p^{(t)}(X_t)]dt + G(t)d\bar{W}_t, \quad t \in [T,0] \quad (34) $$
    * **ODE**:
        $$ dX_t = [F(X_t,t) - \frac{1}{2}G^2(t)\nabla_{x_t} \log p^{(t)}(X_t)]dt \quad (35) $$
* **성공**: DMs는 이미지 합성 및 편집, 오디오 및 분자 합성, 이미지 분할, 비디오 및 3D 객체 생성 등 다양한 분야에서 우수한 성능을 보여왔으며, FID 점수 등 정량적 지표에서 지속적인 개선을 이루었다.

### 지식 증류 (Knowledge Distillation)

* **설명**: 지식 증류는 크고 복잡한 "교사(teacher)" 모델에서 작고 효율적인 "학생(student)" 모델로 지식을 전달하여 정확도를 유지하면서 모델 크기 및 추론 효율성을 개선하는 기술이다. 분류기(classifiers) 분야에서 큰 성공을 거두었다.
* **확산 증류의 동기**: DMs의 성공에도 불구하고 샘플링 속도(많은 NFE 필요)가 여전히 병목 현상으로 작용한다. 확산 증류는 DMs의 학습된 지식을 효율적인 샘플링 메커니즘으로 추출하여, 학생 모델이 훨씬 적은 NFE로 교사 모델과 유사한 생성 성능을 제공하게 한다.
* **확산 증류의 추가 역할**: DMs와 암시적 생성 모델, 정규화 흐름 등 다른 생성 모델 간의 연결을 구축하는 수단이 된다.

## 🛠️ Methodology

이 논문은 확산 모델(DMs)의 지식 증류 전략을 세 가지 주요 범주로 분류하여 설명한다: Diffusion-to-Field (D2F) 증류, Diffusion-to-Generator (D2G) 증류, 그리고 학습 없는(Training-Free) 가속 확산 샘플링 알고리즘이다.

### 1. Diffusion-to-Field (D2F) Distillation

D2F 증류는 DMs의 결정론적 샘플링 방법(생성 ODE)의 비효율성을 해결하기 위해, 더 적은 NFEs로 비교 가능한 샘플을 생성하는 생성 벡터 필드로 증류하는 것을 목표로 한다. 이는 출력 증류(Output Distillation)와 경로 증류(Path Distillation) 두 가지로 분류할 수 있다.

#### 1.1 Output Distillation

출력 증류는 학생 벡터 필드가 DM의 결정론적 샘플링 방법의 출력을 복제하도록 학습시키는 것을 목표로 한다.

* **기본 아이디어**: 교사(teacher) DM의 생성 ODE(식 35)를 수치적으로 풀 때 발생하는 비효율성(작은 스텝 사이즈 필요)을 극복하기 위해, 학생 신경망 $S_\phi^{(stu)}(x,t)$가 더 큰 스텝 사이즈를 사용하여 ODE의 출력을 학습하도록 훈련한다. 즉, 시간 $t$와 $t-\Delta t$ 사이의 ODE 출력 변화량 $\Delta X$를 근사하도록 학습한다.
  $$ \Delta X := X_{t-\Delta t} - X_t = \int_{s=t}^{t-\Delta t} [F(X_s,s) - \frac{1}{2}G^2(s)\nabla_{X_s} \log p^{(s)}(X_s)]ds \quad (36) $$
  여기서 $\theta$는 교사 모델(DM)의 파라미터를, $\phi$는 학생 모델의 파라미터를 나타낸다.

* **Knowledge Distillation (KD) (Luhman and Luhman, 2021)**:
  * **목표**: DDIM(Denoising Diffusion Implicit Models) 샘플러(식 37)를 샘플링 시 단 하나의 NFE만 요구하는 가우시안 모델로 증류한다.
    $$ X_{t_{i-1}} = \frac{\sqrt{\alpha_{t_{i-1}}}}{\sqrt{\alpha_{t_i}}} (X_{t_i} - \sqrt{1-\alpha_{t_i}} \epsilon_\theta(X_{t_i})) + \sqrt{1-\alpha_{t_{i-1}}} \epsilon_\theta(X_{t_i}) \quad (37) $$
    여기서 $\epsilon_\theta(x,t)$는 DM의 노이즈 예측 네트워크이다.
  * **학생 모델**: 조건부 가우시안 모델 $p_{stu}(x_0|x_T) = N(x_0; f_\phi(x_T),I)$를 사용하며, $f_\phi(.)$는 교사 DM의 스코어 네트워크와 동일한 아키텍처를 갖는다.
  * **훈련 목표**: 학생 모델과 DDIM 샘플러 간의 조건부 KL 발산을 최소화한다.
    $$ L(\phi) = E_{x_T \sim N(0,I)} D_{KL}[p_{teacher}(x_0|x_T), p_{stu}(x_0|x_T)] \quad (39) $$
    $$ = E_{x_T \sim N(0,I)} [\frac{1}{2} \Vert f_\phi(x_T) - DDIM(x_T) \Vert_2^2] \quad (40) $$
  * **샘플링**: 가우시안 랜덤 변수 $x_T$를 뽑아 신경망 $f_\phi$에 통과시켜 평균 벡터 $x_0$를 얻는다. 이는 1-NFE 샘플링 모델을 가능하게 한다.
  * **한계**: 단일 훈련 배치에 대해 수백 NFEs가 필요한 DDIM 또는 다른 ODE 샘플러의 최종 출력을 생성해야 하므로 계산적으로 비효율적이다.

* **Progressive Distillation (PD) (Salimans and Ho, 2022)**:
  * **목표**: 교사 모델의 NFE 수를 절반으로 줄이는 학생 신경망을 학습한다. 교사 DM의 결정론적 샘플링 전략의 두 단계 예측을 학습한다.
  * **훈련 목표**: 학생 네트워크 $f_\phi(\tilde{x},t)$가 교사 DM의 DDIM 방법을 사용하여 시간 $t_j$에서 $t_i$로의 두 단계 업데이트 $DDIM(\tilde{x}, t_j, t_i)$를 예측하도록 $L_2$ 오류를 최소화한다.
    $$ L(\phi) = E_{x_0 \sim p_d, i \sim Unif(T'), \epsilon \sim N(0,I)} \Vert f_\phi(\tilde{x},t_i) - DDIM(\tilde{x},t_i,t_{i-2}) \Vert \quad (41) $$
    여기서 $\tilde{x} = \sqrt{\alpha_{t_i}}x_0 + \sqrt{1-\alpha_{t_i}}\epsilon$는 VP 확산의 시간 $t_i$에서의 순방향 확산 데이터이다.
  * **과정**: 학생 모델이 교사의 두 단계 샘플링 전략을 정확하게 예측하도록 훈련된 후, 학생 모델이 교사 모델을 대체하고, 새로운 학생 모델이 샘플링 단계 수를 다시 절반으로 줄이도록 훈련하는 과정을 반복한다.
  * **차이점 (KD vs PD)**: PD는 점진적으로 필요한 함수 평가 횟수를 줄이는 반면, KD는 최종 예측을 위한 1단계 학생 모델을 직접 훈련한다. KD는 PD의 극단적인 경우로 볼 수 있다.

* **Two-stage Distillation (Meng et al., 2022)**:
  * **목표**: Classifier-Free Guidance (CFG)가 적용된 조건부 확산 모델(예: GLIDE, DALL·E-2, Stable Diffusion)의 지식 증류 문제를 해결한다.
  * **1단계**: CFG 입력이 있는 학생 조건부 확산 모델을 훈련하여 교사 확산 모델로부터 학습한다. 학생 모델 $f_{\phi_1}(x_t, t, w)$는 교사 모델의 출력 $\hat{x}_w^\theta(x_t)$와 일치하도록 최소화한다.
    $$ L(\phi_1) = E_{w \sim p_w, t \sim U[0,1], x \sim p_d} [\lambda(t)\Vert f_{\phi_1}(x_t,t,w) - \hat{x}_w^\theta(x_t) \Vert_2^2] \quad (42) $$
    여기서 $\hat{x}_w^\theta(x_t) = (1+w)\hat{x}_{c,\theta}(x_t) - w\hat{x}_\theta(x_t)$ 이고, $w$는 CFG 스케일이다. 1단계는 NFE 감소를 목표로 하지 않는다.
  * **2단계**: 1단계에서 훈련된 학생 모델에 PD 전략을 적용하여 확산 단계 수를 크게 줄인다.

* **Classifier-based Feature Distillation (CFD) (Sun et al., 2022)**:
  * **목표**: 교사 모델의 출력과 학생 모델의 몇 단계 출력 간의 픽셀을 직접 정렬하는 것이 어렵다는 문제를 해결한다. 대신, 사전 훈련된 분류기(classifier)에 의해 추출된 특징 공간(feature space)에서 학생 네트워크가 교사 모델의 다단계 출력과 정렬되도록 훈련한다.
  * **방법**: 학생 모델의 1단계 출력과 교사 모델의 다단계 출력의 예측 확률 분포(Softmax 후) 간의 KL 발산을 최소화한다. 엔트로피 및 다양성 정규화 항을 추가하여 성능을 향상시킨다.

* **Consistency Models (CM) (Song et al., 2023)**:
  * **목표**: 생성 ODE의 자체 일관성(self-consistency) 함수 차이를 최소화하여 출력 증류를 구현한다.
  * **방법**: 실제 데이터 샘플을 무작위로 확산시키고, 생성 ODE의 몇 단계를 시뮬레이션하여 동일한 ODE 경로에 있는 다른 노이즈 데이터 샘플을 얻는다. 이 두 노이즈 샘플을 학생 모델에 입력하고 출력의 차이를 최소화하여 생성 ODE의 자체 일관성을 보장한다.

#### 1.2 Path Distillation

경로 증류는 DMs의 샘플링 전략을 개선하여 잠재적으로 더 나은 경로 특성을 갖도록 하는 것을 목표로 한다. 즉, 데이터 분포와 사전 분포를 연결하는 곡선을 더 효율적인(예: 직선에 가까운) 샘플링 경로로 다듬는 데 중점을 둔다.

* **Reflow (Liu et al., 2022b)**:
  * **목표**: 사전 훈련된 교사 신경 ODE의 경로를 학생 모델을 통해 "곧게 펴서(straighten)" 생성 속도를 향상시킨다.
  * **훈련 목표**: 학생 모델 $f_\phi$가 데이터 샘플의 보간(interpolations)과 사전 훈련된 모델의 해당 출력 사이의 $L_2$ 손실의 시간 평균을 최소화한다.
    $$ L(\phi) = E_{x_T \sim p_T, x_0 \sim p_{teacher}(x_0|x_T), t \sim U[0,T]} [\Vert x_T - x_0 - f_\phi(\frac{t}{T}x_T + \frac{T-t}{T}x_0,t) \Vert_2^2] \quad (43) $$
  * **특징**: Reflow 증류는 여러 라운드에 걸쳐 반복될 수 있으며, 교사 모델의 경로를 더욱 곧게 만들 수 있다. 또한, PD와 함께 사용하여 교사 모델의 경로를 곧게 한 다음 PD를 적용하여 학생 모델을 더 효율적으로 만들 수 있다.

* **Reflow를 3D 포인트 클라우드 생성에 적용 (Wu et al., 2022)**:
  * **1단계**: 교사 생성 ODE $f_\theta$를 훈련한다.
    $$ L(\theta) = E_{x_0 \sim p_d, x_T \sim N(0,I), t \sim U[0,T]} [\Vert f_\theta(x_t,t) - (x_0 - x_T) \Vert_2^2] \quad (44) $$
    여기서 $x_t = (t/T)x_0 + (T-t/T)x_T$는 시간 $t$에서의 보간점이다.
  * **2단계**: Reflow 전략을 적용하여 교사 모델을 더욱 곧게 만든다.
  * **3단계**: 학생 모델 $f_\phi$를 사용하여 다단계 교사 모델을 단일 단계 학생 모델로 증류한다.
    $$ L(\phi) = E_{x_T \sim p_d} [Dist(x_T + f_\phi(x_T,T), x_0)] \quad (45) $$
    여기서 $x_0 \sim p_{teacher}(x_0|x_T)$는 교사 모델로부터 얻어지며, $Dist(.,.)$는 두 점 사이의 거리 함수이다(예: Chamfer distance).

### 2. Diffusion-to-Generator (D2G) Distillation

D2G 증류는 확산 모델이 학습한 분포 지식을 효율적인 생성기(generator)로 전달하는 것을 목표로 한다. D2F 증류와 달리, D2G 증류는 데이터 공간과 잠재 공간의 차원이 다를 수 있는 암시적 생성기(implicit generators)를 학생 모델로 사용한다. 학생 생성기는 특정 애플리케이션에 따라 결정론적일 수도 있고 확률적일 수도 있다.

#### 2.1 Distill Deterministic Generator

결정론적 생성기(예: 신경 방사 필드, Neural Radiance Field, NeRF)를 학생 모델로 증류하는 연구가 활발하다. 특히, 사전 훈련된 텍스트-투-이미지 확산 모델은 주어진 텍스트 프롬프트와 관련된 콘텐츠를 가진 NeRF를 학습하는 데 매우 유용하다는 것이 밝혀졌다.

* **Score Distillation Sampling (SDS) (Poole et al., 2022a)**:
  * **목표**: 2D 텍스트-투-이미지 확산 모델을 3D NeRF로 증류한다. 전통적인 NeRF 구성과 달리, 텍스트 기반 NeRF 구성은 3D 객체와 여러 시점의 이미지가 부족하다.
  * **방법**: SDS는 NeRF가 렌더링한 이미지(고정된 시점에서 렌더링된 $x=g(\phi)$)를 사용하여 확산 모델의 손실 함수를 최소화함으로써 NeRF를 최적화한다. 확산 모델의 손실 함수를 직접 최적화하는 계산 비용을 피하기 위해 Unet 야코비안 항을 생략하여 근사한다.
  * **파라미터 업데이트**: 매개변수 $\phi$ (NeRF의 MLP 파라미터)를 업데이트하기 위한 기울기는 다음과 같다.
    $$ Grad(\phi) = \frac{\partial}{\partial\phi}L(\phi) = E_{t, \epsilon, x=g(\phi)}[w(t)(\epsilon_\theta(x_t,t,c) - \epsilon) \frac{\partial x}{\partial\phi}] \quad (46) $$
    여기서 $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon$이고, $\epsilon_\theta(x_t,t,c)$는 사전 훈련된 텍스트-조건부 확산 모델이다. $c$는 텍스트 프롬프트이다.

#### 2.2 Distill Stochastic Generator

확률적 생성기(implicit generative models)로의 증류는 극단적인 추론 효율성(extreme inference efficiency)을 위한 필요성에서 동기 부여되었다. 이러한 모델은 잠재 벡터를 데이터 샘플로 매핑하여 무작위성을 가진 데이터를 생성한다.

* **Luhman and Luhman (2021)의 작업 재해석**: 섹션 2에서 논의된 Luhman and Luhman (2021)의 Knowledge Distillation (KD) 방법은 확률적 생성기를 위한 스코어 증류의 한 유형으로 해석될 수 있다. 그들은 사전 훈련된 확산 모델과 동일한 신경 아키텍처를 가진 Unet을 사용하여 잠재 벡터를 가우시안 출력의 평균으로 매핑한다. 이 경우 KL 발산 최소화는 모델의 평균과 샘플러의 출력 간의 MSE(Mean Squared Error) 최소화와 동일하다.

### 3. Accelerated Sampling Algorithms as Diffusion Distillation

확산 모델은 훈련과 샘플링 프로세스가 분리되어 있다는 독특한 특징을 가진다. 훈련 중에는 모든 이산 또는 연속 노이즈 레벨을 사용하지만, 샘플링 시에는 반드시 모든 확산 시간 레벨을 쿼리할 필요는 없다. 소수의 확산 시간 레벨만 사용하여 샘플링 프로세스를 크게 가속화할 수 있다. 이러한 가속 샘플링 알고리즘은 "큰 모델을 훈련하고 작은 모델을 사용하는" 확산 모델의 일반화된 증류로 간주될 수 있다. 이들은 훈련 기반(Training-based)과 학습 없는(Training-free) 가속 알고리즘으로 나뉜다.

#### 3.1 Training-based Accelerating Algorithms

* **Watson et al. (2022)**: 샘플링 프로세스에서 사용할 시간 단계의 부분집합을 선택하기 위해 FID(Fréchet Inception Distance)를 목표로 동적 계획 문제(dynamic programming problem)를 해결한다. 이를 통해 효율적인 스케줄러 모델을 학습한다.
* **Bao et al. (2022a)**: 추가 공분산 네트워크(covariance networks)를 훈련하여 잠재 공간 분포의 전체 공분산 행렬을 캡처한다 (원본 DM처럼 대각 공분산을 가정하는 대신). 이는 데이터의 복잡한 상관 구조를 더 잘 포착하여 더 사실적인 샘플을 생성한다.
* **Kim et al. (2022)**: 가능도 비율 추정 모델(likeliratio-estimation model), 즉 판별자(discriminator)를 사용하여 실제 스코어 함수와 모델의 스코어 함수 간의 차이를 학습한다. 이 차이는 학습된 확산을 통해 미분한 후 DM의 스코어 함수와 결합하여 편향 없는 스코어 함수를 얻는 데 사용된다.

#### 3.2 Training-free Accelerating Algorithms

학습 없는 알고리즘은 새로운 매개변수 모델을 훈련하지 않고 사전 훈련된 DMs를 쿼리하여 훨씬 적은 수의 확산 시간 레벨로 비교 가능한 생성 성능을 달성하는 것을 목표로 한다. 이러한 알고리즘의 대부분은 생성 SDE 또는 ODE의 다양한 수치 해법에 중점을 둔다.

* **Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020a)**:
  * **설명**: VP 확산의 이산화 버전을 구현하며 이산화된 샘플링 방식을 사용한다.
    $$ x_{t_{i-1}} = \frac{1}{\sqrt{\alpha_{t_i}}} (x_{t_i} - \frac{1-\alpha_{t_i}}{\sqrt{1-\bar{\alpha}_{t_i}}} \epsilon_\theta(x_{t_i},t_i)) + \sigma_{t_i} z_i \quad (48) $$
    여기서 $\alpha_{t_i} = 1-\beta_{t_i}$, $\bar{\alpha}_{t_i} = \prod_{j=1}^i \alpha_{t_j}$이다. DDPM은 샘플링에 1000개의 모든 확산 레벨을 사용한다.

* **Denoising Diffusion Implicit Models (DDIM) (Song et al., 2020a)**:
  * **설명**: 비-마르코프 순방향 확산 모델(non-Markov forward diffusion model) 하에서 DDPM의 파생을 재구성하여 새로운 샘플러 패밀리를 도출한다.
    $$ x_{t_{i-1}} = \sqrt{\alpha_{t_{i-1}}} [\frac{x_{t_i} - \sqrt{1-\alpha_{t_i}}\epsilon_\theta(x_{t_i},t_i)}{\sqrt{\alpha_{t_i}}}] + \sqrt{1-\alpha_{t_{i-1}}-\sigma_{t_i}^2} \cdot \epsilon_\theta(x_{t_i},t_i) + \sigma_{t_i}z_{t_i} \quad (49) $$
    여기서 $\sigma_{t_i}$는 DDIM 샘플러의 무작위성 강도를 제어하는 자유 하이퍼-파라미터이다. $\sigma_{t_i}=0$일 때 샘플러는 결정론적이다. DDIM 샘플러는 더 적은 확산 시간 레벨을 쿼리하여도 DDPM과 유사한 생성 성능을 유지할 수 있음을 보여준다.

* **Optimal Reverse Variance (Bao et al., 2022b)**:
  * **설명**: DDIM 샘플러 패밀리 내에서 순방향 및 역방향 마르코프 체인 간의 KL 발산을 최소화하는 최적의 역방향 분산 $\sigma_{t_i}$가 존재함을 보여준다. 이 최적 분산의 명시적 표현을 도출한다.
    $$ \sigma_n^{*2} = \lambda_n^2 + [\sqrt{\bar{\beta}_n\alpha_n} - \sqrt{\bar{\beta}_{n-1}-\lambda_n^2}]^2 \cdot [1 - \bar{\beta}_n E_{q_n(x_n)}[\Vert\nabla_{x_n}\log q_n(x_n)\Vert_d^2]] \quad (50) $$
  * 실제로는 사전 훈련된 DM의 도움을 받아 몬테카를로 방법으로 최적 분산을 추정한다.
    $$ \hat{\sigma}_n^2 = \lambda_n^2 + [\sqrt{\bar{\beta}_n\alpha_n} - \sqrt{\bar{\beta}_{n-1}-\lambda_n^2}]^2 (1 - \bar{\beta}_n \Gamma_n) \quad (52) $$
    여기서 $\Gamma_n = \frac{1}{M}\sum_{m=1}^M \Vert s_n(x_{n,m})\Vert_d^2$이다.

* **Exponential Integrator 및 고차 솔버 (Liu et al., 2022a; Zhang and Chen, 2022; Lu et al., 2022a)**:
  * **설명**: VP 확산의 역방향 ODE가 준선형 구조(semi-linear structure)를 가짐을 발견한다.
    $$ \frac{dX_t}{dt} = F(t)X_t - \frac{1}{2}G^2(t)\nabla_{X_t}\log p^{(t)}(X_t) \quad (53) $$
    여기서 $F(t) = -\frac{1}{2}\beta(t)$ 이고 $G(t) = \sqrt{\beta(t)}$ 이다.
  * ODE에 지수 적분기(exponential integrator) 기술을 적용하여 VP 역방향 ODE를 단순화한다.
    $$ x_t = \frac{\alpha_t}{\alpha_s}X_s - \alpha_t \int_s^t \frac{d\lambda_\tau}{d\tau}\frac{\sigma_\tau}{\alpha_\tau}\epsilon_\theta(X_\tau,\tau)d\tau \quad (54) $$
    여기서 $\lambda_t := \log(\alpha_t/\sigma_t)$는 로그-SNR 함수이다. 또한 변수 변환(change of variable) 트릭과 테일러 전개(Taylor expansion)를 통해 VP 역방향 ODE에 대한 고차 솔버를 얻는다.

* **Higher-order SDE Solvers (Jolicoeur-Martineau et al., 2021; Karras et al., 2022)**:
  * **설명**: EM 이산화의 대안으로 고차 SDE 솔버가 제안되었다. Karras et al. (2022)는 DM의 개선된 신경 사전 컨디셔닝(neural preconditioning)과 함께 역방향 ODE 및 SDE 시뮬레이션을 위한 2차 Heun 이산화를 사용하여 DM의 생성 성능에 대한 새로운 기록을 세웠다.

* **UniPC Framework (Zhao et al., 2023)**:
  * **설명**: 확산 모델의 빠른 샘플링을 위한 예측-수정(prediction-correction) 유형의 수치 해법을 도입하여 고차 ODE 솔버를 분석하고 설계하는 통합 프레임워크를 제안한다. 이 프레임워크는 기존의 많은 잘 연구된 솔버들을 특수한 경우로 포함한다.

## 📊 Results

이 보고서에서 다루는 논문은 주로 확산 모델 지식 증류에 대한 "종합적인 조사"이므로, 특정 새로운 실험 결과를 직접 제시하기보다는 기존 연구들의 결과를 요약하고 인용한다. 논문에 언급된 주요 정량적/정성적 결과는 다음과 같다.

* **DMs의 성능 향상**: 확산 모델은 CIFAR10 데이터셋에서 무조건부 Frechet Inception Score (FID)를 25.32 (Song and Ermon, 2019)에서 1.97 (Karras et al.)로 꾸준히 감소시켜왔다. 이는 DMs가 우수한 생성 모델로 자리매김했음을 보여준다.

* **Knowledge Distillation (KD) (Luhman and Luhman, 2021)**:
  * **작업**: DDIM 샘플러를 1-NFE 가우시안 모델로 증류한다.
  * **결과**: 학생 모델은 9.39의 FID를 달성했으며, 이는 교사 생성 ODE의 FID 4.16에 비해 성능 저하가 있음을 나타낸다. 하지만 단일 NFE로 샘플링이 가능하다는 점이 중요하다.
  * **측정 방법**: FID (Fréchet Inception Distance).

* **Progressive Distillation (PD) (Salimans and Ho, 2022)**:
  * **작업**: 교사 DDIM 샘플링 전략을 UNet 아키텍처의 학생 모델로 점진적으로 증류한다.
  * **결과**: 연속적인 PD 라운드를 통해 필요한 벡터 필드를 단 4 NFEs로 줄여, 교사 확산 모델의 ODE 샘플러보다 250배 더 효율적이다. 생성 성능은 FID로 측정 시 약 5% 감소한다.
  * **측정 방법**: FID (Fréchet Inception Distance), NFE 감소율.

* **Two-stage Distillation (Meng et al., 2022)**:
  * **작업**: Classifier-Free Guided 조건부 확산 모델(픽셀 공간 및 잠재 공간)을 증류한다.
  * **결과**: ImageNet-64x64 데이터셋의 픽셀 공간 클래스-조건부 생성 실험에서 증류된 학생 모델이 심지어 교사 모델보다 더 나은 FID 점수를 달성했다. 이는 지식 증류가 단순히 효율성 개선을 넘어 성능 향상 가능성도 있음을 시사한다.
  * **측정 방법**: FID (Fréchet Inception Distance).

* **Classifier-based Feature Distillation (CFD) (Sun et al., 2022)**:
  * **작업**: CIFAR10 데이터셋에서 이미지 데이터에 대한 특징 공간 증류를 수행한다.
  * **결과**: 4 NFEs만으로 FID 3.80을 달성하여, Salimans and Ho (2022)의 PD 구현보다 낮은(더 좋은) FID를 기록했다.
  * **측정 방법**: FID (Fréchet Inception Distance).

* **Score Distillation Sampling (SDS) (Poole et al., 2022a)**:
  * **작업**: 2D 텍스트-투-이미지 확산 모델을 3D NeRF로 증류하여 텍스트 프롬프트로부터 3D NeRF를 생성한다.
  * **결과**: 텍스트 프롬프트로부터 3D NeRF를 성공적으로 생성하는 데 주목할 만한 결과를 달성하여, 3D 생성 모델링 분야에서 DMs 증류 연구를 촉진했다.

* **Training-free Accelerated Sampling Algorithms (Karras et al., 2022)**:
  * **작업**: DM의 신경 사전 컨디셔닝 및 2차 Heun 이산화를 사용한 역방향 ODE/SDE 시뮬레이션.
  * **결과**: FID 1.79로 DMs의 생성 성능에 대한 새로운 기록을 세웠다.
  * **측정 방법**: FID (Fréchet Inception Distance).
  * **일반적인 한계**: 최고의 학습 없는 방법으로도 비교 가능한 생성 성능을 달성하려면 일반적으로 10개 이상의 NFEs가 필요하다.

이러한 결과는 확산 모델의 지식 증류가 샘플링 효율성을 크게 개선하면서도, 대부분의 경우 납득할 만한 수준의 생성 품질을 유지하거나 심지어 능가할 수 있음을 보여준다. 특히 NFE 감소는 확산 모델의 실제 적용 가능성을 넓히는 중요한 진전이다.

## 🧠 Insights & Discussion

이 논문은 컴퓨터 비전 및 딥러닝 분야에서 빠르게 발전하는 확산 모델(DMs)의 지식 증류(Knowledge Distillation)에 대한 포괄적이고 시의적절한 개요를 제공한다.

### 논문에서 뒷받침되는 강점

1. **체계적인 분류 및 구조**: DMs의 지식 증류 전략을 Diffusion-to-Field (D2F), Diffusion-to-Generator (D2G), 그리고 학습 없는(Training-Free) 가속 샘플링 알고리즘이라는 세 가지 주요 범주로 명확하게 분류한다. 이러한 체계적인 분류는 복잡하고 빠르게 확장되는 이 분야를 이해하고 탐색하는 데 큰 도움을 준다. 각 범주 내에서도 출력 증류, 경로 증류, 결정론적/확률적 생성기 증류 등으로 세분화하여 다양한 접근 방식을 명료하게 제시한다.

2. **기초 개념에 대한 충실한 배경 설명**: DMs의 등장 배경이 되는 ARMs, EBMs, SBMs에 대한 상세한 설명을 제공하여, 독자들이 DMs가 어떻게 발전해왔으며 왜 스코어 함수 모델링이 중요한지에 대한 깊은 이해를 돕는다. 이는 DMs에 대한 기본적인 지식이 있는 독자뿐만 아니라 이 분야에 처음 접하는 독자들에게도 유용하다.

3. **핵심 방법론에 대한 상세한 설명**: 각 증류 기법에 대해 그 목표, 학생 모델의 구조, 훈련 목표, 그리고 관련 주요 방정식들을 구체적으로 설명한다. 특히 Luhman and Luhman (2021)의 KD, Salimans and Ho (2022)의 PD, Poole et al. (2022a)의 SDS 등 주요 landmark 연구들을 충분한 맥락과 함께 제시하여 독자들이 각 방법의 핵심 아이디어를 쉽게 파악할 수 있도록 한다.

4. **샘플링 효율성 개선의 중요성 강조**: DMs의 가장 큰 병목 중 하나인 느린 샘플링 속도 문제를 해결하기 위한 지식 증류의 중요성을 일관되게 강조한다. 이는 학술 연구뿐만 아니라 실제 애플리케이션 관점에서도 핵심적인 동기이며, 논문은 이러한 필요성을 효과적으로 전달한다.

5. **다양한 적용 사례 제시**: 이미지 생성뿐만 아니라 3D NeRF 생성, TTS(Text-to-Speech) 등 다양한 모달리티와 작업에 DMs 증류가 어떻게 적용될 수 있는지 보여줌으로써, 이 분야의 광범위한 잠재력을 드러낸다.

### 한계, 가정 또는 미해결 질문

1. **결과에 대한 심층 분석 부족**: 논문은 주로 다양한 연구의 방법론을 설명하는 데 집중하며, 각 증류 기법의 정량적/정성적 결과는 간략하게 인용하는 수준에 그친다. 서로 다른 증류 방법론 간의 성능 트레이드오프(예: NFE 감소와 생성 품질 저하), 특정 데이터셋이나 작업에 대한 방법론의 강점/약점, 그리고 계산 자원(훈련 시간, 메모리) 요구 사항에 대한 심층적인 비교 분석은 제한적이다.

2. **방법론의 가정과 제약에 대한 명시적 언급 부족**: 예를 들어, 특정 증류 방법이 특정 유형의 DMs(예: VP diffusion vs. VE diffusion)에 더 적합한지, 또는 특정 아키텍처(예: UNet)에 대한 의존성이 있는지 등, 각 방법론이 내포하는 암묵적인 가정이나 제약 조건에 대한 논의가 더 명확했으면 한다.

3. **새로운 방법론 제안이 아닌 조사 논문의 한계**: 본 논문은 조사(survey) 논문이므로 새로운 방법론을 제안하거나 새로운 실험을 수행하지 않는다. 따라서 DMs 지식 증류 분야의 최신 연구 동향을 종합적으로 이해하는 데는 탁월하지만, 특정 문제에 대한 혁신적인 해결책을 제시하지는 않는다.

4. **윤리적, 사회적 함의 논의 부재**: DMs와 그 증류 모델의 강력한 생성 능력은 딥페이크, 오정보 생성 등 윤리적 문제를 야기할 수 있다. 이러한 기술의 사회적 함의나 잠재적 위험성에 대한 논의는 포함되어 있지 않다. 이는 조사 논문의 범위 밖일 수 있지만, 점차 중요해지는 고려 사항이다.

### 논문에 근거한 간략한 비판적 해석 및 논의사항

이 조사는 DMs 지식 증류 분야의 현황을 매우 효과적으로 요약하고 있다. DMs의 샘플링 속도 문제를 해결하기 위한 지식 증류의 중요성은 아무리 강조해도 지나치지 않으며, 본 논문은 이를 위한 다양한 접근 방식을 체계적으로 분류하여 연구자들에게 귀중한 로드맵을 제공한다. 특히, "학습 없는" 가속 알고리즘을 지식 증류의 한 형태로 포함시킨 점은 DMs의 독특한 훈련-샘플링 분리 특성을 잘 포착한 통찰력 있는 분류이다.

그러나, DMs 증류의 궁극적인 목표는 "최소한의 성능 손실로 최대의 효율성"을 달성하는 것이다. 논문에서 언급된 결과들을 보면, 1-NFE 또는 소수의 NFE로 샘플링이 가능해졌음에도 불구하고 여전히 교사 모델 대비 FID 성능 저하가 관찰되는 경우가 많다. 이는 효율성과 품질 사이의 근본적인 트레이드오프가 여전히 존재하며, 이를 극복하기 위한 새로운 연구 방향이 필요함을 시사한다. 예를 들어, FID와 같은 특정 지표 외에 지각적 품질(perceptual quality)이나 특정 응용 분야의 요구사항을 반영하는 새로운 평가 지표의 도입도 논의될 수 있다.

또한, 다양한 증류 기법들이 소개되었으나, 이들이 서로 독립적으로 발전하는 경향이 있다. 향후 연구에서는 D2F, D2G, 그리고 가속 샘플링 알고리즘 간의 시너지를 탐색하여, 예를 들어 경로를 곧게 펴는 동시에 특정 생성기로 지식을 전달하고, 그 과정에서 학습 없는 가속화 기법을 결합하는 등의 하이브리드 접근 방식이 유망할 것으로 보인다. 궁극적으로는 DMs의 학습된 지식을 활용하여 매우 빠르고 고품질의 생성 결과를 얻는 "원샷(one-shot)" 또는 "극소수 단계(few-step)" 생성 모델을 개발하는 것이 이 분야의 중요한 장기 목표가 될 것이다.

## 📌 TL;DR

이 논문은 컴퓨터 비전 및 딥러닝 분야의 선도적인 생성 모델인 확산 모델(DMs)의 지식 증류(Knowledge Distillation)에 대한 포괄적인 조사 보고서이다. DMs는 뛰어난 생성 성능을 보이지만, 느린 샘플링 속도라는 고질적인 문제에 직면해 있다. 이 문제를 해결하기 위해, 논문은 DMs의 "지식"을 더 작고 효율적인 모델 또는 샘플링 메커니즘으로 전달하는 다양한 접근 방식을 세 가지 주요 범주로 분류하여 제시한다:

1. **Diffusion-to-Field (D2F) 증류**: DMs의 다단계 생성 ODE를 더 적은 신경 함수 평가(NFE)로 작동하는 생성 벡터 필드(generative vector field)로 압축한다. 이 범주에는 교사 모델의 출력을 모방하는 **출력 증류(Output Distillation)**(예: KD, PD, CFG 모델 증류, CM)와 생성 경로 자체를 더 효율적으로 개선하는 **경로 증류(Path Distillation)**(예: Reflow)가 포함된다.

2. **Diffusion-to-Generator (D2G) 증류**: DMs가 학습한 풍부한 분포 지식을 효율적인 암시적 생성기(implicit generator)로 전달한다. 이는 주로 3D 객체 생성(예: 텍스트-투-3D NeRF를 위한 SDS)과 같이 특정 응용을 위한 결정론적 생성기 또는 극단적인 추론 효율성을 위한 확률적 생성기(stochastic generator)로의 증류를 다룬다.

3. **학습 없는(Training-free) 가속 샘플링 알고리즘**: 새로운 모델을 훈련할 필요 없이, 사전 훈련된 DMs의 샘플링 프로세스 자체를 가속화하는 다양한 수치 해법 기반의 기술들을 다룬다(예: DDIM, 최적 역방향 분산 추정, 지수 적분기 및 고차 솔버).

이 연구는 DMs의 샘플링 효율성을 크게 향상시키고, DMs와 다른 생성 모델 간의 연결을 구축함으로써, 실제 응용(예: 실시간 이미지 생성, 3D 콘텐츠 생성)이나 향후 연구(예: 새로운 증류 전략 개발)에 중요한 역할을 할 잠재력을 가진다.
