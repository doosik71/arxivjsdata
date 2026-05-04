# Generative Modeling with Diffusion

Justin Le (2025)

## 🧩 Problem to Solve

본 논문은 데이터의 분포를 모방하여 새로운 샘플을 생성하는 Generative Model, 특히 Diffusion Model의 수학적 원리를 체계적으로 설명하고 이를 실제 불균형 데이터셋(Imbalanced Dataset) 문제에 적용하는 것을 목표로 한다.

일반적인 생성 모델의 목표는 주어진 샘플 데이터의 기저 분포(underlying distribution)를 학습하여 유사한 데이터를 임의로 생성하는 것이다. 특히, 분류 문제에서 특정 클래스의 데이터가 극도로 적은 불균형 데이터 상황에서는 모델이 소수 클래스를 제대로 학습하지 못해 Recall(재현율)이 낮아지는 문제가 발생한다. 본 논문은 Diffusion Model을 통해 소수 클래스의 합성 데이터를 생성하고, 이를 통해 분류기의 성능을 향상시킬 수 있는지 탐구한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **수학적 정식화**: Ornstein-Uhlenbeck(OU) 방정식을 기반으로 Diffusion Model의 Forward Process(노이즈 추가 과정)와 Reverse Process(노이즈 제거 과정)를 확률미분방정식(SDE) 관점에서 정밀하게 정의하였다.
2. **이산화 및 학습 알고리즘 제시**: 연속적인 SDE를 이산적인 타임스텝으로 변환하여 실제 컴퓨터로 구현 가능한 학습 및 생성 알고리즘을 도출하였으며, 특히 노이즈 $\epsilon_0$를 예측하는 방식의 효율성을 제시하였다.
3. **불균형 데이터 해결책 제시**: 신용카드 부정 결제 탐지(Credit Card Fraud Detection)와 같은 극심한 불균형 데이터셋에서 Diffusion Model을 이용한 데이터 증강(Data Augmentation)이 분류기의 Recall 성능을 높일 수 있음을 실험적으로 입증하였다.

## 📎 Related Works

논문은 생성 모델의 주요 아키텍처로 Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), 그리고 Transformers를 언급한다. 이러한 기존 모델들과 달리, Diffusion Model은 데이터에 반복적으로 노이즈를 추가하여 완전한 가우시안 노이즈 상태로 만든 뒤, 이 과정을 역전시켜 데이터를 복원하는 독특한 접근 방식을 취한다.

특히, 본 연구는 이미지 생성이라는 일반적인 Diffusion Model의 용도에서 벗어나, 표 형식의 데이터(Tabular Data)에 대한 데이터 증강 도구로서의 가능성을 탐구했다는 점에서 기존 연구들과 차별점을 가진다.

## 🛠️ Methodology

### 1. Forward Process: Ornstein-Uhlenbeck (OU) Equation

Forward process는 원래의 데이터 $X_0$를 표준 정규 분포 $N(0, I)$로 변환하는 과정이다. 이를 위해 다음과 같은 OU 방정식(SDE)을 사용한다.

$$dX_t = -X_t dt + \sqrt{2} dB_t, \quad X(0) = X_0$$

여기서 $-X_t dt$는 Drift 항으로 데이터의 지수적 감쇠를 유도하며, $B_t$는 Wiener process(브라운 운동)로 무작위성을 부여한다. 이 방정식의 해는 다음과 같이 표현된다.

$$X_t = e^{-t} X_0 + \sqrt{1-e^{-2t}} Z, \quad Z \sim N(0, I)$$

편의를 위해 $\gamma_t = e^{-t}$와 $\beta_t = \sqrt{1-e^{-2t}}$로 정의하면, $X_t = \gamma_t X_0 + \beta_t Z$가 된다. $t \to \infty$일 때 $X_t$는 표준 정규 분포로 수렴한다.

### 2. Discretization (이산화)

실제 구현을 위해 시간을 $N$개의 단계로 나눈다. $n$번째 단계에서 $n+1$번째 단계로 넘어가는 재귀 식은 다음과 같다.

$$X_{n+1} = \gamma(\Delta t_{n+1}) X_n + \beta(\Delta t_{n+1}) Z_{n+1}$$

### 3. Reverse Diffusion (역과정)

Reverse process의 목표는 표준 정규 분포에서 샘플링한 $X_N$으로부터 $X_0$를 복원하는 것이다. 조건부 확률 밀도 $\rho(x_n | x_{n+1}, x_0)$는 가우시안 분포를 따르며, 다음과 같이 정의된다.

$$\rho(x_n | x_{n+1}, x_0) \sim N(\mu, \sigma_n^2)$$

이때 평균 $\mu$는 $x_{n+1}$과 $x_0$ 모두에 의존하며, 분산 $\sigma_n^2$은 타임스텝 $n$에만 의존한다.

### 4. Model Training and Generation

생성 시에는 $x_0$를 알 수 없으므로, 신경망 $\theta$를 통해 $\mu$를 추정해야 한다. 저자는 $\mu$를 직접 예측하는 대신, $x_0$를 $x_{n+1}$로 만드는 '노이즈' $\epsilon_0$를 예측하는 방식(Method 3)을 채택하였다.

* **학습 목표**: 실제 노이즈 $\epsilon_0$와 모델이 예측한 노이즈 $\epsilon_\theta$ 사이의 Mean Square Error (MSE)를 최소화한다.
    $$L[\theta] = \frac{1}{N} \sum_{n=1}^N ||\epsilon_0 - \epsilon_\theta||^2$$
* **생성 절차**: $x_N \sim N(0, I)$에서 시작하여, 학습된 모델 $\epsilon_\theta$를 이용해 $\hat{\mu}_\theta$를 계산하고, 이를 바탕으로 $x_n$을 순차적으로 샘플링하여 최종적으로 $x_0$를 얻는다.

## 📊 Results

### 실험 설정

* **데이터셋**: Kaggle의 신용카드 부정 결제 데이터셋 (전체 284,807건 중 부정 결제 492건, 비율 약 $1.73 \cdot 10^{-3}$)
* **작업**: 부정 결제 클래스를 Diffusion Model로 증강한 후, 분류기 성능 비교
* **분류기**: XGBoost, Random Forest
* **지표**: Precision(정밀도), Recall(재현율), F1-Score
* **검증**: UMAP을 이용한 차원 축소 시각화 결과, 생성된 합성 데이터가 실제 부정 결제 데이터의 구조적 특성을 잘 보존하고 있음을 확인하였다.

### 정량적 결과

**Table 1: XGBoost 결과**

| Method | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| No Augmentation | 0.8901 | 0.8265 | 0.8571 |
| **Diffusion** | 0.8737 | **0.8469** | **0.8601** |

**Table 2: Random Forest 결과**

| Method | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| No Augmentation | 0.9524 | 0.8163 | 0.8791 |
| **Diffusion** | 0.9053 | **0.8776** | **0.8912** |

두 모델 모두 Diffusion 기반 데이터 증강을 사용했을 때 **Recall이 유의미하게 상승**하였으며, 결과적으로 F1-Score 또한 개선되었다. 하지만 Precision은 다소 하락하는 Trade-off가 관찰되었다.

## 🧠 Insights & Discussion

본 논문은 Diffusion Model이 단순한 이미지 생성을 넘어, 데이터 불균형이 심한 정형 데이터의 증강 도구로 활용될 수 있음을 보여주었다. 특히 Recall의 상승은 금융 사기 탐지와 같이 '부정 사례를 놓치는 비용'이 '오탐지 비용'보다 훨씬 큰 도메인에서 매우 강력한 이점이 된다.

**강점**:

* SDE라는 엄밀한 수학적 토대 위에서 Diffusion 과정을 설명하여 모델의 동작 원리를 명확히 규명하였다.
* 실제 산업 데이터(Credit Card Fraud)를 활용하여 실용적인 가치를 입증하였다.

**한계 및 논의**:

* **Precision 하락**: 데이터를 증강함으로써 Recall은 높아졌으나 Precision이 낮아졌다는 점은, 생성된 데이터 중 일부가 실제 데이터 분포의 경계를 모호하게 만들어 False Positive를 증가시켰을 가능성을 시사한다.
* **가정**: 본 논문은 데이터가 가우시안 노이즈로 변환되고 복원될 수 있다는 가정을 전제로 하지만, 실제 정형 데이터의 복잡한 범주형 변수 처리에 대한 구체적인 방법론은 명시되지 않았다.

## 📌 TL;DR

이 논문은 OU 방정식을 이용해 Diffusion Model의 수학적 원리를 정립하고, 이를 활용해 신용카드 부정 결제 데이터의 소수 클래스를 증강하는 방법을 제안하였다. 실험 결과, Diffusion 기반 증강은 분류기의 Recall을 높여 부정 결제 탐지 능력을 향상시켰으며, 이는 데이터 불균형 문제가 심한 다양한 분류 작업에 응용될 가능성이 높다.
