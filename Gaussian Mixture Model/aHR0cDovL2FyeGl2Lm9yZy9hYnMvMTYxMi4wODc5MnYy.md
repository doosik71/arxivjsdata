# Superpixel Segmentation Using Gaussian Mixture Model

Zhihua Ban, Jianguo Liu, and Li Cao (2017)

## 🧩 Problem to Solve

본 논문은 이미지의 픽셀들을 지각적으로 일관된 원자적 영역으로 분할하는 Superpixel segmentation 문제에 집중한다. Superpixel은 컴퓨터 비전의 다양한 후속 작업(분할, 추적, 매칭 등)에서 입력 데이터의 수를 획기적으로 줄여주는 전처리 단계로 널리 사용된다.

Superpixel 알고리즘이 충족해야 할 네 가지 핵심 속성은 다음과 같다:
1. **Accuracy (정확도)**: 객체의 경계선에 잘 밀착되어야 한다.
2. **Regularity (규칙성)**: Superpixel의 모양이 규칙적이어야 후속 알고리즘의 그래프 구축이 용이하다.
3. **Similar size (유사한 크기)**: Superpixel들이 서로 비슷한 크기를 가져야 편향 없는 처리가 가능하다.
4. **Efficiency (효율성)**: 실시간 적용을 위해 낮은 계산 복잡도를 가져야 한다.

문제의 핵심은 **정확도와 규칙성 사이의 상충 관계(trade-off)**이다. 제한된 크기 내에서 경계선에 완벽히 밀착하려면 모양이 불규칙해질 수밖에 없으며, 반대로 규칙성을 강조하면 경계선 밀착도가 떨어진다. 본 논문의 목표는 Gaussian Mixture Model(GMM)을 이용하여 이 두 가지 특성 사이의 균형을 맞추면서도 계산 효율성이 높은 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 각 Superpixel을 하나의 Gaussian 분포로 간주하고, 각 픽셀이 자신과 관련된 여러 Gaussian 분포들의 혼합(Mixture)으로부터 생성되었다고 가정하는 것이다.

주요 기여 사항은 다음과 같다:
- **Pixel-related GMM**: 전통적인 GMM과 달리, 각 픽셀이 서로 다른 GMM을 가질 수 있도록 설계하여 데이터 포인트들이 독립 동일 분포(i.i.d.)를 따르지 않는 모델을 제안하였다.
- **정규성 제어**: 공분산 행렬(Covariance matrix)의 고유값(Eigenvalue)을 조정함으로써 Superpixel의 모양 규칙성을 제어할 수 있는 메커니즘을 제공한다.
- **선형 복잡도 및 병렬성**: 픽셀 수 $N$에 대해 $O(N)$의 시간 복잡도를 가지며, 알고리즘 구조상 OpenMP 등을 통한 병렬 구현이 매우 용이하다.
- **성능 향상**: 최신 Superpixel 알고리즘들과 비교하여 정확도 면에서 우수한 성능을 보이며, 계산 효율성 측면에서도 경쟁력을 갖추었다.

## 📎 Related Works

기존의 Superpixel 알고리즘은 크게 두 가지 접근 방식으로 나뉜다.

1. **경계 최적화 (Optimize boundaries)**:
   - 경계선을 마킹하거나 진화하는 곡선을 사용하여 분할하는 방식이다.
   - SEEDS, TurboPixels, VCells 등이 이에 속하며, 특히 SEEDS는 경계 밀착도가 높지만 모양이 매우 불규칙한 경향이 있다.
2. **픽셀 그룹화 (Grouping pixels)**:
   - 모든 픽셀에 레이블을 직접 할당하는 방식이다.
   - SLIC가 대표적이며, $k$-means 클러스터링을 기반으로 효율성과 규칙성이 좋지만 경계 밀착도는 상대적으로 떨어진다. LSC는 SLIC를 개선하여 성능을 높였다.

기존 연구들은 대부분 $k$-means의 목적 함수를 변형하여 사용해 왔으나, 본 논문은 GMM이라는 확률 모델을 도입하여 Superpixel 문제에 접근함으로써 차별성을 둔다.

## 🛠️ Methodology

### 1. 모델 정의
이미지의 각 픽셀 $i$는 위치와 색상 정보를 포함하는 벡터 $z_i = (x_i, y_i, c_i)^T$로 표현된다. 

각 Superpixel $k \in \mathcal{K}$는 평균 $\mu_k$와 공분산 $\Sigma_k$를 가지는 Gaussian 분포 $p(z; \theta_k)$에 대응된다. 이때, 픽셀 $i$가 속할 수 있는 Superpixel들의 집합 $\mathcal{K}_i$를 정의하여, 픽셀 $i$의 확률 밀도 함수 $p_i(z)$를 다음과 같은 혼합 모델로 정의한다:
$$p_i(z) = P_i \sum_{k \in \mathcal{K}_i} p(z; \theta_k)$$
여기서 $P_i = 1/|\mathcal{K}_i|$이며, 이는 각 픽셀이 주변의 제한된 Superpixel 후보군 중에서만 선택되도록 하여 Superpixel의 크기를 유사하게 유지하는 역할을 한다.

최종적으로 픽셀 $i$의 레이블 $L_i$는 사후 확률(Posterior probability)이 최대가 되는 $k$로 결정된다:
$$L_i = \arg \max_{k \in \mathcal{K}_i} p(z_i; \theta_k)$$

### 2. 파라미터 추정 (EM 알고리즘)
최대 가능도 추정(Maximum Likelihood Estimation)을 위해 Expectation-Maximization (EM) 알고리즘을 사용한다.

- **E-step**: 현재 파라미터 $\theta$를 기반으로 픽셀 $i$가 Superpixel $k$에 속할 책임도(Responsibility) $R_{i,k}$를 계산한다.
  $$R_{i,k} = \frac{p(z_i; \theta_k)}{\sum_{k \in \mathcal{K}_i} p(z_i; \theta_k)}$$
- **M-step**: 계산된 $R_{i,k}$를 사용하여 평균 $\mu_k$와 공분산 $\Sigma_k$를 업데이트한다.
  $$\mu_k = \frac{\sum_{i \in I_k} R_{i,k} z_i}{\sum_{i \in I_k} R_{i,k}}, \quad \Sigma_k = \frac{\sum_{i \in I_k} R_{i,k} (z_i - \mu_k)(z_i - \mu_k)^T}{\sum_{i \in I_k} R_{i,k}}$$

### 3. 실제 구현 및 규칙성 제어
계산 효율을 위해 full covariance matrix 대신 블록 대각 행렬(Block diagonal matrix)을 사용한다:
$$\Sigma_k = \begin{bmatrix} \Sigma_{k,s} & 0 \\ 0 & \Sigma_{k,c} \end{bmatrix}$$
여기서 $\Sigma_{k,s}$는 공간(spatial) 공분산, $\Sigma_{k,c}$는 색상(color) 공분산을 의미한다.

특히, 공분산 행렬의 고유값 분해(Eigendecomposition) 후, 고유값 $\lambda$가 임계값 $\epsilon$보다 작을 경우 $\epsilon$으로 대체하는 방식을 사용한다.
- $\epsilon_s$ (공간 임계값): 기본적으로 2로 설정한다.
- $\epsilon_c$ (색상 임계값): **이 값을 높이면 Superpixel의 모양이 더 규칙적으로 변하고, 낮추면 경계선 밀착도(정확도)가 높아진다.**

### 4. 추론 및 후처리 절차
1. 파라미터 $\theta$를 초기화한다.
2. 정해진 횟수 $T$(일반적으로 10회)만큼 E-step과 M-step을 반복한다.
3. 사후 확률에 따라 픽셀 레이블을 결정한다.
4. **연결성 강화(Connectivity enforcement)**: GMM 특성상 연결성이 보장되지 않으므로, 매우 작은 고립된 Superpixel들을 인접한 가장 유사한 Superpixel에 병합하는 후처리를 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋**: BSDS500
- **평가 지표**:
  - **Boundary Recall (BR)**: 실제 경계선이 얼마나 잘 회복되었는가.
  - **Under-segmentation Error (UE)**: Superpixel이 객체 경계를 넘어 다른 객체를 포함하는 정도 (낮을수록 좋음).
  - **Achievable Segmentation Accuracy (ASA)**: Superpixel을 통해 도달할 수 있는 최대 분할 정확도.
- **비교 대상**: LSC, SLIC, SEEDS, ERS, TurboPixels, LRW, VCells, Waterpixels 등 8종의 최신 알고리즘.

### 결과 분석
- **정량적 결과**: 제안 방법은 특히 UE와 ASA 지표에서 다른 알고리즘들을 압도하는 성능을 보였다. BR의 경우 LSC와 경쟁 수준이나, $\epsilon_c$를 낮게 설정하면 LSC보다 더 높은 BR을 달성할 수 있었다.
- **계산 효율성**: 픽셀 수 증가에 따라 실행 시간이 선형적으로 증가함을 확인하여 이론적 복잡도 $O(N)$을 실험적으로 증명하였다. 또한, 멀티코어 CPU에서 OpenMP를 사용했을 때 코어 수에 비례하여 속도가 향상되는 병렬 확장성을 보였다.
- **정성적 결과**: 시각적 비교 결과, 제안 방법은 규칙성을 유지하면서도 객체의 세밀한 경계를 매우 정확하게 포착하는 모습을 보였다. SEEDS나 ERS는 정확도는 높으나 모양이 너무 불규칙했고, SLIC 등은 모양은 규칙적이나 경계 밀착도가 낮았다.

## 🧠 Insights & Discussion

본 논문은 GMM이라는 확률적 프레임워크를 Superpixel 문제에 성공적으로 적용하였다. 가장 큰 통찰은 **픽셀마다 서로 다른 혼합 분포를 가지게 함으로써 지역적 특성을 반영**하고, **공분산 행렬의 고유값을 제어하여 규칙성과 정확도 사이의 균형점을 사용자가 직접 조정**할 수 있게 한 점이다.

**강점**:
- 기존 $k$-means 기반 방식들이 해결하지 못한 '규칙성 vs 정확도'의 딜레마를 파라미터 $\epsilon_c$를 통해 제어 가능한 형태로 풀어냈다.
- $O(N)$의 복잡도와 병렬 가능 구조를 통해 실용적인 속도를 확보하였다.

**한계 및 논의**:
- GMM 기반 방식의 고질적인 문제인 '연결성(connectivity)'을 수학적으로 완벽히 해결하지 못하고 후처리(merging) 단계에 의존하였다는 점이 아쉽다.
- $\epsilon_c$와 $\lambda$ 같은 하이퍼파라미터의 설정이 결과에 영향을 미치지만, 이에 대한 자동 최적화 방법론은 제시되지 않았다.

## 📌 TL;DR

본 연구는 각 Superpixel을 Gaussian 분포로 모델링하고 픽셀별로 최적화된 GMM을 구축하는 새로운 Superpixel 분할 방법을 제안한다. 공분산 행렬의 제어를 통해 모양의 규칙성과 경계 정확도 사이의 균형을 맞추었으며, $O(N)$의 선형 복잡도와 병렬 처리 능력을 갖추어 효율성까지 확보하였다. 이 연구는 고정밀 전처리가 필요한 컴퓨터 비전 파이프라인에서 매우 유용한 도구가 될 가능성이 높다.