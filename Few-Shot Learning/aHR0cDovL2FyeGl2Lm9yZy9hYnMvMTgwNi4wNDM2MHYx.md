# MSplitLBI: Realizing Feature Selection and Dense Estimation Simultaneously in Few-shot and Zero-shot Learning

Bo Zhao, Xinwei Sun, Yanwei Fu, Yuan Yao, Yizhou Wang (2018)

## 🧩 Problem to Solve

본 논문은 두 공간(spaces) 또는 부분 공간(subspaces) 사이의 표현 계수(representation coefficients)를 효율적으로 학습하는 임베딩 모델 학습 문제를 다룬다. 특히 Few-shot Learning (FSL)과 Zero-shot Learning (ZSL) 상황에서 이미지 특징을 레이블 공간이나 시맨틱 공간으로 매핑하는 선형 임베딩 모델을 학습시키는 것이 핵심이다.

기존의 접근 방식은 주로 $L_1$ regularization(Lasso) 또는 $L_2$ regularization(Ridge)을 사용하여 가중치를 규제한다. 하지만 $L_1$ regularization은 강한 신호(strong signals)를 포착하는 Feature Selection 능력은 뛰어나지만, 약한 신호(weak signals)를 무시하여 훈련 데이터에 대한 과소적합(underfitting)을 유발할 수 있다. 반면, $L_2$ regularization은 모든 차원을 비례적으로 축소시키기 때문에 추정치에 편향(bias)이 발생한다는 한계가 있다. 따라서 본 논문의 목표는 Feature Selection(강한 신호 포착)과 Dense Estimation(약한 신호 포착)을 동시에 달성하여, 데이터 적합도를 높이면서도 해석 가능한 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 임베딩 특징이 세 가지 직교하는 구성 요소, 즉 **희소한 강한 신호(Sparse Strong Signals)**, **밀집된 약한 신호(Dense Weak Signals)**, 그리고 **무작위 노이즈(Random Noise)**로 이루어져 있다는 가설에서 출발한다.

이를 실현하기 위해 제안된 **MSplit LBI (Multiple Split Linear Bregman Iteration)** 알고리즘은 임베딩 가중치를 이 세 가지 요소로 분해하여 학습한다. 이를 통해 모델은 해석 가능성을 위한 희소 추정치(sparse estimator)와 예측 성능 향상을 위한 밀집 추정치(dense estimator)를 동시에 얻을 수 있다.

## 📎 Related Works

### Feature Selection 및 Variable Split
기존의 Feature Selection 방법은 필터, 래퍼, 임베디드 방법으로 나뉘며, $L_0$나 $L_1$ 규제를 사용하는 임베디드 방법이 효율성 덕분에 널리 쓰였다. 특히 Lasso($L_1$)는 희소성을 부여하여 중요한 특징을 선택하는 데 유용하다. 또한, Variable Splitting 기법(예: ADMM)은 제약 조건이 있는 최적화 문제를 풀기 위해 변수를 분리하여 하나는 밀집된 상태로, 다른 하나는 희소한 상태로 유지함으로써 예측력을 높이는 접근 방식을 취해왔다.

### Few-shot 및 Zero-shot Learning
FSL은 극소수의 샘플로 새로운 개념을 학습하는 것이 목적이며, ZSL은 훈련 샘플 없이 보조 지식(시맨틱 공간)을 통해 새로운 클래스를 인식하는 것이 목적이다. 두 분야 모두 이미지 특징 공간에서 레이블/시맨틱 공간으로의 선형 매핑을 학습하는 모델이 주로 사용되며, 과적합을 방지하기 위해 $L_1$ 또는 $L_2$ 규제가 빈번하게 적용되어 왔다.

## 🛠️ Methodology

### 전체 시스템 구조 및 목표
기본적으로 시각적 특징 $X \in \mathbb{R}^{N \times d}$와 레이블 임베딩 $E \in \mathbb{R}^{N \times p}$ 사이의 선형 관계 $E = XB$를 학습하는 것을 목표로 한다. 여기서 $B \in \mathbb{R}^{d \times p}$는 학습해야 할 임베딩 행렬이다. 최적화 문제는 다음과 같이 정의된다.

$$B = \arg \min_{B} \ell(B, X) + \lambda \Omega(B)$$

여기서 $\ell(B, X) = \|XB - E\|_2^2$이며, $\Omega(B)$는 규제 항이다.

### MSplit LBI 알고리즘
본 논문은 $L_1$ 규제의 한계를 극복하기 위해 보조 변수 $\Gamma$를 도입하고, 다음과 같은 손실 함수를 정의한다.

$$\ell(B, \Gamma) = \ell(B, X) + \frac{1}{2\nu} \|B - \Gamma\|_F^2 \quad (\nu > 0)$$

여기서 $\Gamma$는 희소성을 강제하여 강한 신호를 선택하는 역할을 하며, $B$는 $\Gamma$와 가깝게 유지되면서도 밀집된 추정을 통해 약한 신호를 포착한다. 학습 과정은 Linearized Bregman Iteration (LBI)을 기반으로 하며, 다음과 같은 반복 업데이트 절차를 따른다.

1. **$B$ 업데이트 (Gradient Descent):**
   $$B_{k+1} = B_k - \kappa \alpha \nabla_B \ell(B_k, \Gamma_k)$$
2. **보조 변수 $Z$ 업데이트:**
   $$Z_{k+1} = Z_k - \alpha \nabla_\Gamma \ell(B_k, \Gamma_k)$$
3. **$\Gamma$ 업데이트 (Soft-thresholding):**
   $$\Gamma_{k+1} = \kappa \cdot S(Z_{k+1}, 1)$$
   여기서 $S(Z, \lambda) = \text{sign}(Z) \cdot \max(|Z| - \lambda, 0)$이다.
4. **희소 추정치 $\tilde{B}$ 생성 (Projection):**
   $$\tilde{B}_{k+1} = B_{k+1} \circ [1_{\{i \in \tilde{S}^{(j), k+1}\}}]$$
   $\tilde{S}$는 $\Gamma_{k+1}$의 서포트 셋(non-zero index set)을 의미하며, $B$의 값 중 $\Gamma$가 선택한 인덱스의 값만 남긴 것이다.

### 특징 분해 (Decomposition Property)
MSplit LBI를 통해 얻은 밀집 추정치 $B_k$는 다음과 같이 직교 분해된다.
$$B_k = \text{Signal}_{\text{strong}} \oplus \text{Signal}_{\text{weak}} \oplus \text{Random Noise}$$

- $\tilde{B}_k$: 강한 신호만을 포착하여 Feature Selection에 사용된다.
- $B_k$: 강한 신호와 약한 신호를 모두 포함하여 실제 예측(Prediction)에 사용된다.
- $\nu$ 파라미터는 $B$와 $\tilde{B}$ 사이의 거리를 조절하며, $\nu$가 클수록 $B$가 약한 신호를 포착할 수 있는 자유도가 높아진다.

### FSL 및 ZSL 적용 방법
- **Few-shot Learning:** 이미지 특징 $X$에서 원-핫(one-hot) 레이블 임베딩 $E$로의 매핑 행렬 $B$를 MSplit LBI로 학습한다. 테스트 시 $\hat{e} = x B$를 계산하여 가장 큰 값을 가진 클래스로 분류한다.
- **Zero-shot Learning:** 소스 도메인 레이블 $E_s$에서 타겟 도메인 레이블 $E_t$로의 구조적 지식 $B$를 학습한다 ($E_t = E_s B$). 이후 소스 클래스의 프로토타입 $F_s$에 $B$를 곱해 타겟 클래스의 가상 프로토타입을 생성한다 ($\hat{F}_t = F_s B$).

## 📊 Results

### 시뮬레이션 실험
$N=100, p=80$ 설정에서 MLE, Lasso, Ridge, Elastic Net과 비교하였다. 상대 오차 $\frac{\|\hat{\beta} - \beta^*\|_2}{\|\beta^*\|_2}$를 측정한 결과, MSplit LBI의 밀집 추정치 $\beta$가 모든 케이스에서 가장 낮은 오차를 보였다. 특히 변수 간 상관관계가 높아지는 $\sigma=0.8$ 상황에서 Lasso보다 훨씬 뛰어난 성능을 보여, Irrepresentable Condition에 더 강건함을 입증하였다.

### Zero-shot Learning 실험
AwA, CUB, ImageNet 데이터셋에서 성능을 평가하였다.
- **AwA 및 CUB:** MSplit LBI ($\beta$)가 각각 85.34%, 57.52%의 정확도로 SOTA 성능을 달성하였다.
- **ImageNet:** Top-1 8.35%, Top-5 18.76%의 결과를 보였다.
- **분석:** $\tilde{\beta}$(희소)보다 $\beta$(밀집)의 성능이 약 1% 높게 나타났는데, 이는 약한 신호가 실제 임베딩 학습에 기여함을 정량적으로 증명한다.

### Few-shot Learning 실험
Omniglot 및 SUN 데이터셋에서 평가하였다.
- **Omniglot:** 대부분의 설정에서 Lasso보다 우수하였으며, 딥러닝 모델인 M-Net에 근접한 성능을 보였다. 특히 5-shot 설정으로 갈수록 선형 모델의 안정성이 높아져 성능 갭이 줄어들었다.
- **SUN:** Lasso(59.09%) 대비 MSplit LBI ($\beta$)(61.47%)가 2.38%p 높은 성능을 보였으며, 이는 다시 한번 약한 신호 포착의 중요성을 시사한다.

## 🧠 Insights & Discussion

본 논문은 단순히 희소성을 추구하는 것($L_1$)과 단순히 가중치를 억제하는 것($L_2$) 사이의 절충안을 제시한 것이 아니라, 신호의 강도에 따라 가중치를 세 가지 성분으로 분해하는 새로운 관점을 제시하였다.

**강점 및 이론적 근거:**
- 이론적으로 MSplit LBI는 강한 신호에 대해 편향이 없는(unbiased) 추정치임을 보였다.
- $\nu$ 파라미터를 통해 모델 선택 일관성(model selection consistency)과 예측 성능 사이의 트레이드오프를 조절할 수 있다.
- 시각화 결과, "pig" 클래스를 예측할 때 "cow", "rhinoceros" 같은 강한 상관관계뿐만 아니라, 서식지가 유사한 "hamster", "skunk" 같은 약한 상관관계(Weak Signals)까지 포착하여 성능을 높였음을 확인하였다.

**한계 및 논의사항:**
- 하이퍼파라미터 $\nu$와 반복 횟수 $k$에 대한 의존성이 있으며, 이를 결정하기 위해 교차 검증이 필요하다.
- FSL 실험에서 PCA를 통한 차원 축소 후 Ridge가 더 좋게 나온 경우가 있는데, 이는 특징 공간이 이미 충분히 정제된 경우 추가적인 Feature Selection이 오히려 성능을 저하시킬 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 임베딩 가중치를 **강한 신호, 약한 신호, 노이즈**의 세 가지 직교 성분으로 분해하여 학습하는 **MSplit LBI** 알고리즘을 제안하였다. 이를 통해 $L_1$의 과소적합 문제와 $L_2$의 편향 문제를 동시에 해결하였으며, 특히 데이터가 극도로 부족한 **Few-shot 및 Zero-shot Learning** 환경에서 SOTA 수준의 성능을 달성하였다. 이 연구는 희소 추정치를 통한 해석 가능성과 밀집 추정치를 통한 예측 성능을 동시에 확보할 수 있는 일반적인 정규화 프레임워크를 제공한다는 점에서 가치가 크다.