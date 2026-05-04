# Non-Normal Mixtures of Experts

Faicel Chamroukhi (2015)

## 🧩 Problem to Solve

본 논문은 회귀 분석, 분류, 군집화에서 데이터의 이질성(heterogeneity)을 모델링하기 위해 널리 사용되는 Mixture of Experts (MoE) 프레임워크의 한계점을 해결하고자 한다. 

기존의 MoE, 특히 연속형 데이터를 다루는 Normal Mixture of Experts (NMoE)는 전문가(expert) 컴포넌트가 가우시안 분포(Gaussian distribution)를 따른다고 가정한다. 그러나 실제 데이터셋에는 비대칭적인 거동(asymmetric behavior), 두터운 꼬리(heavy tails), 혹은 전형적이지 않은 이상치(outliers)가 포함된 경우가 많다. 정규 분포는 이러한 특성에 매우 민감하므로, NMoE를 그대로 적용할 경우 모델의 적합도(fit)가 크게 저하되는 문제가 발생한다.

따라서 본 연구의 목표는 비대칭성, 두터운 꼬리, 그리고 이상치 문제를 동시에 해결할 수 있는 **Non-Normal Mixture of Experts (NNMoE)** 모델들을 제안하고, 이를 위한 효율적인 파라미터 추정 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 정규 분포를 일반화한 세 가지 비정규 분포(Skew-Normal, $t$, Skew-$t$)를 MoE의 전문가 컴포넌트로 도입하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **세 가지 새로운 NNMoE 모델 제안**:
    *   **SNMoE (Skew-Normal MoE)**: 데이터의 비대칭성(skewness)을 처리하기 위해 Skew-Normal 분포를 사용한다.
    *   **TMoE ($t$-MoE)**: 이상치에 강건하며 두터운 꼬리를 가진 데이터를 모델링하기 위해 $t$-분포를 사용한다.
    *   **STMoE (Skew-$t$ MoE)**: 비대칭성과 두터운 꼬리 문제를 동시에 해결하고 이상치에 강건하도록 Skew-$t$ 분포를 사용한다.
2.  **최적화 알고리즘 개발**: 각 모델의 파라미터를 단조적으로 최대화하여 추정할 수 있도록 전용 Expectation-Maximization (EM) 및 Expectation Conditional Maximization (ECM) 알고리즘을 설계하였다.
3.  **유연한 응용 가능성 제시**: 제안된 모델들을 비선형 회귀 예측뿐만 아니라, 회귀 데이터를 기반으로 한 모델 기반 군집화(model-based clustering)에 적용하는 방법론을 제시하였다.

## 📎 Related Works

기존의 MoE 연구는 주로 정규 분포 기반의 NMoE에 집중되어 왔다. 최근 일부 연구에서 이상치 문제를 해결하기 위해 Laplace Mixture of Linear Experts (LMoLE)와 같은 강건한 모델이 제안되었으나, 본 논문은 다음과 같은 차별점을 가진다.

*   **분포의 확장성**: 기존의 강건한 회귀 혼합 모델(Robust regression mixture models)들이 단순히 $t$-분포나 Laplace 분포만을 사용한 것과 달리, 본 논문은 비대칭성까지 고려한 Skew-Normal 및 Skew-$t$ 분포를 MoE 프레임워크에 통합하였다.
*   **조건부 혼합 비율(Conditional Mixing Proportions)**: 이전의 많은 강건한 회귀 모델들은 혼합 비율 $\pi_k$를 상수로 가정하는 단순 혼합 모델(mixture models)이었다. 반면, 본 논문의 NNMoE는 gating function을 통해 혼합 비율이 입력 변수(covariates)에 따라 변하는 **완전 조건부 혼합(fully conditional mixture)** 구조를 채택하여 모델링 유연성을 극대화하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
NNMoE의 일반적인 확률 밀도 함수는 다음과 같이 정의된다.
$$f(y|r,x; \Psi) = \sum_{k=1}^K \pi_k(r; \alpha) f_k(y|x; \Psi_k)$$
여기서 $\pi_k(r; \alpha)$는 gating function(혼합 비율)이며, $f_k(y|x; \Psi_k)$는 $k$번째 전문가(expert)의 조건부 밀도 함수이다.

*   **Gating Function**: 다항 로지스틱 모델(Multinomial Logistic Model)을 사용하여 다음과 같이 계산한다.
    $$\pi_k(r; \alpha) = \frac{\exp(\alpha_k^T r)}{\sum_{\ell=1}^K \exp(\alpha_\ell^T r)}$$
*   **Expert Components**: $f_k$는 모델에 따라 $\text{SN}(\cdot)$, $t(\cdot)$, 또는 $\text{ST}(\cdot)$ 분포를 따른다. 각 전문가의 평균 $\mu(x; \beta_k)$는 입력 $x$에 대한 회귀 함수로 정의된다.

### 2. 모델별 세부 분포 및 특성
*   **SNMoE**: $\text{SN}(\mu, \sigma^2, \lambda)$ 분포를 사용하며, $\lambda$는 비대칭도를 조절하는 skewness 파라미터이다. $\lambda=0$이면 정규 분포와 동일해진다.
*   **TMoE**: $t(\mu, \sigma^2, \nu)$ 분포를 사용하며, $\nu$는 자유도(degrees of freedom) 파라미터이다. $\nu \to \infty$이면 정규 분포로 수렴하며, $\nu$가 작을수록 꼬리가 두꺼워져 이상치에 강건해진다.
*   **STMoE**: $\text{ST}(\mu, \sigma^2, \lambda, \nu)$ 분포를 사용하여 비대칭성($\lambda$)과 강건성($\nu$)을 동시에 확보한다.

### 3. 학습 절차 및 파라미터 추정 (EM/ECM)
관측 데이터의 로그 가능도(log-likelihood)를 최대화하기 위해 EM 및 ECM 알고리즘을 사용한다.

*   **E-Step**: 현재 파라미터를 이용하여 각 데이터 포인트가 $k$번째 전문가에 속할 사후 확률 $\tau_{ik}$와 잠재 변수(latent variables, 예: $U_i, W_i$)의 조건부 기댓값을 계산한다.
*   **M-Step (또는 CM-Step)**:
    *   **$\alpha$ 업데이트**: $\pi_k$의 파라미터 $\alpha$는 closed-form 해가 없으므로 **Iteratively Reweighted Least Squares (IRLS)** 알고리즘을 통해 수치적으로 최적화한다.
    *   **$\beta, \sigma^2$ 업데이트**: 전문가의 평균 함수 $\mu(x; \beta_k)$가 선형일 경우, 가중 최소제곱법(weighted least squares)과 유사한 방식으로 분석적 해를 통해 업데이트한다.
    *   **$\lambda, \nu$ 업데이트**: 비선형 방정식 형태로 나타나며, **Brent's method**와 같은 root-finding 알고리즘을 사용하여 해결한다.

특히 STMoE와 같은 복잡한 모델은 파라미터 공간을 부분 집합으로 나누어 순차적으로 최적화하는 **ECM (Expectation Conditional Maximization)** 알고리즘을 적용하여 수치적 안정성을 높였다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: 인위적으로 생성한 시뮬레이션 데이터 및 실제 데이터(Tone perception, Temperature anomalies)를 사용하였다.
*   **비교 대상**: NMoE, SNMoE, TMoE, STMoE 간의 성능을 비교하였다.
*   **지표**: 회귀 분석에서는 True mean function과의 **MSE (Mean Squared Error)**를 측정하였고, 모델 선택에서는 **BIC, AIC, ICL** 지표를 사용하였다.

### 2. 주요 결과
*   **수렴성 확인**: 시뮬레이션 결과, 표본 크기 $n$이 증가함에 따라 파라미터 추정 오차(MSE)가 감소하여 MLE의 일치성(consistency)을 확인하였다.
*   **이상치 강건성 (Robustness)**: 
    *   데이터에 이상치를 0%에서 5%까지 추가하며 실험한 결과, NMoE와 SNMoE는 이상치 비율이 증가함에 따라 MSE가 급격히 증가하였다.
    *   반면 **TMoE와 STMoE는 이상치가 존재함에도 불구하고 MSE가 매우 낮게 유지**되었으며, 이는 $t$-분포 기반의 전문가들이 이상치의 영향을 효과적으로 억제함을 보여준다.
*   **실제 데이터 적용**:
    *   **Tone perception 데이터**: BIC 지표를 통해 최적의 전문가 수 $K=2$를 찾아냈으며, 이상치가 추가된 상황에서도 TMoE와 STMoE가 정교한 적합도를 유지하였다.
    *   **Temperature anomalies 데이터**: 전 지구적 온도 편차 데이터를 분석한 결과, 제안된 모델들이 기존의 LMoLE와 유사하거나 더 우수한 성능을 보였으며, 특히 전 지구 온난화의 시기적 구분(segmentation)을 성공적으로 수행하였다.

## 🧠 Insights & Discussion

본 논문은 정규 분포의 경직성을 극복하기 위해 비정규 분포를 MoE에 도입함으로써 모델의 유연성과 강건성을 동시에 확보하였다.

**강점 및 기여**:
*   단순한 강건 회귀를 넘어, gating function을 통한 조건부 혼합 구조를 유지하면서 비정규 분포를 통합한 점이 돋보인다.
*   특히 STMoE는 비대칭성과 두터운 꼬리를 모두 처리할 수 있어, 현실 세계의 복잡한 데이터 분포를 모델링하는 데 매우 강력한 도구가 될 수 있음을 입증하였다.
*   EM 알고리즘의 변형인 ECM과 IRLS를 적절히 조합하여 복잡한 분포의 파라미터 추정 문제를 효율적으로 해결하였다.

**한계 및 논의사항**:
*   본 연구는 단변량(univariate) 응답 변수 $Y$에 대해서만 논의되었으며, 다변량(multivariate) 데이터로의 확장은 향후 과제로 남겨두었다.
*   또한 표준 MoE 구조만을 다루었으며, 계층적 MoE(Hierarchical MoE) 구조로의 확장이 필요하다.
*   실험 결과에서 AIC가 모델의 복잡도를 과소평가하여 전문가 수 $K$를 과다하게 추정하는 경향이 발견되었으므로, 본 모델군에서는 BIC를 우선적으로 사용할 것을 권장한다.

## 📌 TL;DR

이 논문은 기존 Normal MoE가 이상치와 비대칭 데이터에 취약하다는 점을 해결하기 위해, 전문가 컴포넌트로 **Skew-Normal, $t$, Skew-$t$ 분포를 도입한 NNMoE 모델**들을 제안한다. 전용 EM/ECM 알고리즘을 통해 파라미터를 추정하며, 실험을 통해 특히 **TMoE와 STMoE가 이상치가 많은 환경에서도 매우 강건한 회귀 성능과 군집화 성능**을 보임을 입증하였다. 이 연구는 실제 기후 변화 데이터 및 음향 인지 데이터 분석에 적용되어 실용성을 증명하였으며, 향후 다변량 및 계층적 MoE 구조로 확장될 가능성이 높다.