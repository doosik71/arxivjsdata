# Deep Gaussian Mixture Models
Cinzia Viroli, Geoffrey J. McLachlan

## 🧩 Problem to Solve
이 연구는 복잡한 데이터 관계를 효율적으로 설명하기 위해 심층 학습(Deep Learning)의 계층적 추론 능력을 비지도 학습(unsupervised learning) 영역, 특히 클러스터링에 적용하는 문제를 다룹니다. 기존 가우시안 혼합 모델(Gaussian Mixture Models, GMM)은 강력하지만, 매우 복잡하거나 고차원적인 데이터를 모델링하는 데 한계가 있으며, 심층 신경망(Deep Neural Networks)의 성공이 주로 지도 학습(supervised learning)에 집중되어 있다는 점을 지적하며 비지도 심층 모델의 필요성을 강조합니다.

## ✨ Key Contributions
*   **심층 가우시안 혼합 모델(Deep Gaussian Mixture Model, DGMM) 정의:** 잠재 변수들의 여러 계층으로 구성되며, 각 계층의 변수들이 가우시안 혼합 분포를 따르는 새로운 모델을 제안합니다. 이는 전역적으로 데이터를 유연하게 설명하는 비선형 모델을 제공합니다.
*   **모델 식별성(Identifiability) 논의 및 해결:** DGMM의 식별성 문제를 명확히 하고, 이를 해결하기 위해 각 계층의 잠재 변수 차원 $r_1, r_2, ..., r_h$를 $p > r_1 > r_2 > ... > r_h \ge 1$와 같이 점진적으로 감소시키는 차원 축소 기법을 적용합니다.
*   **혁신적인 확률적 EM(Stochastic EM, SEM) 알고리즘 제안:** DGMM의 파라미터 추정을 위해 고안된 EM 알고리즘의 확장으로, 복잡한 E-단계 계산을 효율적으로 처리하기 위해 잠재 변수를 조건부 분포에서 샘플링하는 방법을 사용합니다. 이를 통해 대규모 데이터에 대한 동시 파라미터 추정을 가능하게 합니다.
*   **다양한 기존 모델 통합:** DGMM이 고전적인 가우시안 혼합 모델, 혼합 인자 분석기(Mixtures of Factor Analyzers), 혼합-혼합 모델(Mixtures of Mixtures Model) 등을 특별한 경우로 포함하는 일반적인 프레임워크임을 보여줍니다.
*   **우수한 클러스터링 성능 입증:** 시뮬레이션 및 실제 데이터셋(Smiley, Wine, Olive, Ecoli, Vehicle, Satellite)에 대한 실험을 통해 기존의 다양한 클러스터링 방법보다 일관되게 우수한 클러스터링 성능을 달성함을 입증합니다.

## 📎 Related Works
*   **가우시안 혼합 모델(GMM):** 모델 기반 클러스터링의 기본 도구로, 데이터의 분포를 여러 가우시안 분포의 선형 조합으로 모델링합니다.
*   **혼합 인자 분석기(Mixtures of Factor Analyzers, MFA):** 고차원 데이터 모델링에 사용되는 GMM의 확장으로, 각 가우시안 컴포넌트 내에 인자 모델을 통합하여 차원 축소와 클러스터링을 동시에 수행합니다.
*   **심층 신경망(Deep Neural Networks):** 이미지 인식, 음성 인식 등 지도 학습 태스크에서 큰 성공을 거두었으며, Facebook의 DeepFace와 같은 사례가 언급됩니다. 이 연구는 이러한 심층 학습의 개념을 비지도 클러스터링에 확장합니다.
*   **SEM/MCEM 알고리즘:** EM 알고리즘의 계산적 어려움을 극복하기 위해 사용되는 확률적 버전 또는 몬테카를로 대안입니다.

## 🛠️ Methodology
DGMM은 $h$개의 계층으로 구성되며, 각 계층 $l$에서 관측 데이터 $y_i$ (또는 이전 계층의 잠재 변수 $z^{(l-1)}_i$)는 다음과 같이 선형 모델과 확률 $\pi^{(l)}_{s_l}$의 혼합으로 표현됩니다:

$$
z^{(l-1)}_i = \eta^{(l)}_{s_l} + \Lambda^{(l)}_{s_l} z^{(l)}_i + u^{(l)}_i \quad \text{with prob. } \pi^{(l)}_{s_l}
$$

여기서 $z^{(h)}_i \sim \mathcal{N}(0, I_p)$이고 $u^{(l)}_i \sim \mathcal{N}(0, \Psi^{(l)}_{s_l})$입니다.

**주요 방법론:**
1.  **계층적 구조:** 가장 낮은 계층($l=1$)에서 관측 데이터 $y_i$를 모델링하고, 각 상위 계층으로 갈수록 $z^{(l)}_i$와 같은 잠재 변수들이 이전 계층의 잠재 변수 $z^{(l-1)}_i$를 설명합니다. 이는 중첩된 선형 모델의 집합으로 비선형 모델을 구성합니다.
2.  **차원 축소:** 과적합과 식별성 문제를 피하기 위해, 각 계층의 잠재 변수 $z^{(l)}_i$의 차원 $r_l$이 $p > r_1 > r_2 > ... > r_h \ge 1$와 같이 점진적으로 감소하도록 제약합니다. 이는 $\Lambda^{(l)}_{s_l}$ 행렬의 차원에 영향을 줍니다.
3.  **식별성 제약:** 잠재 변수의 존재로 인한 식별성 문제를 해결하기 위해, 각 계층에서 인자 모델에 적용되는 제약(예: $\Lambda^{\top}\Psi^{-1}\Lambda$가 대각 행렬이며 요소가 감소하는 순서로 배치)을 적용하여 파라미터가 유일하게 식별되도록 합니다.
4.  **확률적 EM (SEM) 알고리즘:**
    *   **E-단계:** EM 알고리즘의 E-단계에서 필요한 잠재 변수들의 사후 분포 계산의 복잡성을 해결하기 위해 몬테카를로 샘플링을 사용합니다. 특히, 조건부 밀도 $f(z^{(l)}_i | z^{(l-1)}_i, s; \Theta')$로부터 $M$개의 복제본 $z^{(l)}_{i,m}$을 생성합니다.
    *   **M-단계:** 샘플링된 잠재 변수를 사용하여 각 계층의 파라미터 $\Lambda^{(l)}_{s_l}$, $\Psi^{(l)}_{s_l}$, $\eta^{(l)}_{s_l}$, $\pi^{(l)}_s$를 추정합니다. 이 과정은 각 계층에서 독립적으로 최대화될 수 있는 형태로 변환됩니다.

## 📊 Results
*   **Smiley 데이터:** 인위적으로 생성된 3차원 데이터셋(클러스터링 관련 변수 2개, 노이즈 변수 1개)에 대해 DGMM은 Adjusted Rand Index (ARI) 0.788, 오분류율(misclassification rate) 0.087로 다른 모든 비교 방법(k-means, PAM, Hclust, GMM, SNmm, STmm)보다 가장 우수한 클러스터링 성능을 보였습니다.
*   **실제 데이터셋:**
    *   **Wine 데이터:** 대부분의 방법이 잘 수행되었으나, DGMM과 MFA가 ARI 0.983, 오분류율 0.006으로 가장 우수했습니다.
    *   **Olive 데이터:** DGMM이 ARI 0.997, 오분류율 0.002로 단 1개의 오분류만 발생시키며 압도적인 성능을 보였습니다.
    *   **Ecoli 데이터:** 불균형한 클래스를 가진 고차원 데이터로 SNmm과 STmm은 수렴에 실패했으나, DGMM은 ARI 0.749, 오분류율 0.187로 다른 방법보다 우수했습니다.
    *   **Vehicle 데이터:** 어려운 분류 과제로, DGMM이 ARI 0.191, 오분류율 0.481로 다른 방법(대부분 0.0X ARI)보다 월등히 나은 성능을 보였습니다.
    *   **Satellite 데이터:** 6개의 불균형한 클래스와 높은 차원($p=36$)을 가진 어려운 클러스터링 문제에서, DGMM은 ARI 0.604, 오분류율 0.249로 MFA와 비견되거나 약간 우수한 결과를 나타냈습니다.

전반적으로 DGMM은 다양한 복잡성과 특성을 가진 데이터셋에서 기존의 GMM 기반 및 비모델 기반 클러스터링 방법보다 우수한 성능을 일관되게 보여주었습니다.

## 🧠 Insights & Discussion
*   **모델의 유연성:** DGMM은 심층 학습의 계층적 구조를 통해 복잡한 비선형 데이터 관계를 유연하게 포착할 수 있음을 입증합니다. 이는 단일 계층 모델로는 어려운 클러스터링 문제를 해결하는 데 효과적입니다.
*   **기존 방법론의 일반화:** DGMM은 GMM, MFA, 혼합-혼합 모델 등을 특수한 경우로 포괄하는 일반적인 프레임워크를 제공합니다. 이는 기존 방법론의 장점을 계승하면서 확장된 기능을 제공한다는 의미가 있습니다.
*   **확률적 EM의 효율성:** 복잡한 계층 구조와 많은 파라미터로 인해 발생할 수 있는 EM 알고리즘의 계산적 병목 현상을 확률적 EM 알고리즘으로 성공적으로 해결하여 대규모 데이터에도 적용 가능함을 보여줍니다.
*   **차원 축소의 중요성:** 고차원 데이터 및 모델 과적합 문제를 해결하는 데 있어 잠재 변수의 차원 점진적 감소가 중요한 역할을 한다는 것을 강조합니다. 이는 모델의 강건성과 해석 가능성을 높입니다.
*   **BIC를 통한 모델 선택:** BIC(Bayesian Information Criterion)와 같은 정보 기준을 사용하여 계층 수, 서브컴포넌트 수, 잠재 변수 차원 등 DGMM의 최적 구조를 효과적으로 선택할 수 있음을 보여주었습니다.
*   **한계:** 특정 데이터셋(Ecoli)에서 SNmm 및 STmm이 수렴에 실패한 사례가 있지만, DGMM은 이러한 상황에서도 강건하게 작동했습니다. 모델의 복잡성으로 인한 파라미터 튜닝의 어려움은 여전히 존재할 수 있으나, 본 연구에서는 BIC를 통해 해결책을 제시했습니다.

## 📌 TL;DR
이 논문은 복잡한 비선형 데이터의 비지도 클러스터링을 위해 여러 계층의 잠재 변수와 가우시안 혼합 분포를 사용하는 **심층 가우시안 혼합 모델(DGMM)**을 제안합니다. 모델의 식별성 문제를 해결하기 위해 각 계층의 잠재 변수 차원을 점진적으로 줄이는 방법을 적용하고, 대규모 데이터에서도 효율적인 파라미터 추정을 위해 **확률적 EM 알고리즘**을 개발했습니다. 시뮬레이션 및 다양한 실제 데이터셋에 대한 실험을 통해 DGMM이 기존의 클러스터링 방법보다 **우수한 성능**을 보이며, GMM과 MFA를 포함한 여러 기존 모델을 일반화하는 유연한 프레임워크임을 입증했습니다.