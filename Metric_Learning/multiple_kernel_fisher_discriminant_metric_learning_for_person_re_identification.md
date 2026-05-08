# Multiple Kernel Fisher Discriminant Metric Learning for Person Re-identification

T M Feroz Ali, Kalpesh K Patel, Rajbabu Velmurugan, Subhasis Chaudhuri (2019)

## 🧩 Problem to Solve

본 논문은 서로 겹치지 않는(disjoint) 카메라 뷰 사이에서 보행자 이미지를 매칭하는 Person Re-identification (re-ID) 문제를 해결하고자 한다. Person re-ID는 비디오 감시, 보안, 생체 인식 및 법의학 분야에서 매우 중요하지만, 다음과 같은 이유로 매우 도전적인 과제이다.

- **환경적 변동성:** 서로 다른 카메라에서 촬영된 동일 인물의 이미지는 조명, 포즈, 시점(viewpoint), 카메라 특성 및 배경 잡음으로 인해 매우 다르게 보일 수 있다.
- **데이터 품질 및 유사성:** 낮은 해상도로 인해 신체적 특성만으로 개인을 구별하기 어려우며, 서로 다른 사람이 매우 유사한 의상을 입고 있을 경우 구분이 더욱 힘들어진다.

따라서 본 논문의 목표는 이러한 변동성을 극복하고 클래스 내 분산은 최소화하며 클래스 간 분산은 최대화하는 효율적인 Metric Learning 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 중심 아이디어는 Kernel Fisher Discriminant Analysis (KFDA)를 통해 판별력 있는 Metric Space를 학습하고, 이를 Mahalanobis distance metric으로 유도하는 것이다. 주요 기여 사항은 다음과 같다.

- **KFDA의 적용 가능성 확인:** 별도의 보조 방법 없이도 KFDA가 Person re-ID의 Metric Learning에 매우 강력하고 효율적인 후보임을 입증하였다.
- **Multiple Kernel Learning (MKL) 도입:** 단일 커널 사용 시 발생할 수 있는 편향(bias)을 제거하고 고차원 특징 공간에서 더 풍부한 표현을 얻기 위해 두 가지 MKL 프레임워크(NP-MFML, SM-MFML)를 제안하였다.
- **효율적인 거리 계산:** KFDA 기반의 Mahalanobis metric을 닫힌 형태(closed-form solution)로 유도하여, 계산 복잡도를 줄이면서도 높은 판별력을 유지하는 방법을 제시하였다.

## 📎 Related Works

Person re-ID를 위한 기존 Metric Learning 연구들은 유사한 클래스 샘플 간의 거리는 가깝게, 서로 다른 클래스 간의 거리는 멀게 만드는 것을 목표로 한다.

- **기존 접근 방식:** KISSME, PRDC, LADF, KNFST 등이 제안되었으며, 특히 LFDA(Local Fisher Discriminant Analysis)와 kLFDA(Kernel LFDA)는 데이터의 국소적 구조(local neighborhood information)를 활용하여 임베딩 공간을 학습한다.
- **차별점:** 본 논문은 LFDA나 kLFDA와 달리 국소적 구조 정보를 사용하지 않는다. 저자들은 Person re-ID 데이터셋의 경우 클래스당 샘플 수가 매우 적어 국소 구조를 정확하게 추정하는 것이 오히려 성능을 저하시킬 수 있다고 주장하며, 단순한 KFDA 기반 접근이 더 효율적임을 강조한다. 또한, MKML이나 XQDA와 같은 기존 방법들과는 최적화 방식 및 커널 조합 방법론에서 차이가 있다.

## 🛠️ Methodology

### 1. Kernel Fisher Discriminant Analysis (KFDA)

KFDA는 입력 데이터를 고차원 특징 공간 $\mathcal{F}$로 매핑하는 비선형 함수 $\Phi(x)$를 사용하여, 클래스 간 분산(between-class variance)을 최대화하고 클래스 내 분산(within-class variance)을 최소화하는 판별 서브스페이스를 학습한다.

최적화 문제는 다음과 같은 Fisher criterion을 최대화하는 것이다.
$$\text{maximize } \text{tr}\{(W^T S_{\Phi}^w W)^{-1} (W^T S_{\Phi}^b W)\}$$
여기서 $S_{\Phi}^b$는 특징 공간에서의 클래스 간 산포 행렬(between-class scatter matrix)이고, $S_{\Phi}^w$는 클래스 내 산포 행렬(within-class scatter matrix)이다.

커널 트릭을 적용하여 $\Phi(x)$를 직접 계산하지 않고 커널 함수 $k(x_i, x_j)$를 이용해 다음과 같은 최적화 문제로 변환한다.
$$\text{maximize } \text{tr}\{(A^T Q A)^{-1} (A^T P A)}$$
여기서 $A$는 확장 벡터(expansion vector)들의 행렬이며, $P$와 $Q$는 각각 커널 기반의 클래스 간 및 클래스 내 산포를 나타내는 행렬이다. $Q$가 특이 행렬(singular)인 경우, $Q^{-1}P$의 상위 고유벡터를 통해 최적의 판별 벡터를 얻는다.

### 2. KFDA 기반 Mahalanobis Metric 유도

본 논문은 KFDA의 투영 행렬 $W$를 Mahalanobis 행렬 $M = VV^T$에서 $V=W$로 선택함으로써 특징 공간 $\mathcal{F}$에서의 Mahalanobis 거리와 KFDA 판별 서브스페이스에서의 유클리드 거리가 동일함을 보였다. 최종적으로 두 샘플 $y$와 $z$ 사이의 매칭 점수는 다음과 같이 단순화된 수식으로 계산된다.
$$d^2(\Phi(y), \Phi(z)) = \| A^T (k_y - k_z) \|^2$$
여기서 $k_y = [k(x_1, y), \dots, k(x_n, y)]^T$이다. 이 방식은 파라미터 수가 적고 닫힌 형태의 솔루션을 가져 구현이 매우 효율적이다.

### 3. Multiple Kernel Fisher Metric Learning (MFML)

단일 커널의 한계를 극복하기 위해 두 가지 MKL 방법을 제안한다.

- **NP-MFML (N-Proportionally Weighted MFML):**
  사전에 정의된 $q$개의 커널 중 교차 검증을 통해 성능이 가장 좋은 상위 $N$개의 커널만을 선택하여 가중 결합한다. 각 커널 $k_t$의 가중치 $\beta_t$는 다음과 같이 결정된다.
  $$\beta_t = \begin{cases} \frac{\pi_t - \delta'}{\sum_{r \in S} (\pi_r - \delta')}, & \text{if } t \in S \\ 0, & \text{otherwise} \end{cases}$$
  여기서 $\pi_t$는 해당 커널의 정확도, $S$는 상위 $N$개 커널의 집합, $\delta'$는 $(N+1)$번째 커널의 정확도이다.

- **SM-MFML (Square Matrix MFML):**
  두 개의 커널 행렬 $K_1, K_2$를 사용하여 다음과 같이 최종 커널 $\bar{K}$를 구성한다.
  $$\bar{K} = \frac{1}{2}(K_1 + K_2) + \tau (K_1 - K_2)^2$$
  여기서 $\tau$는 양의 상수이며, 제곱항 $(K_1 - K_2)^2$을 통해 최종 커널의 양의 준정부호성(positive semi-definiteness)을 보장하면서 두 커널 간의 정보 차이를 반영한다.

## 📊 Results

### 실험 설정

- **데이터셋:** PRID450S, GRID, VIPeR 세 가지 벤치마크 데이터셋을 사용하였다.
- **특징 추출기:** GOG(기본) 및 LOMO를 사용하였다.
- **지표:** Rank-1, Rank-10, Rank-20 정확도를 측정하였으며, 10회 반복 실험의 평균값을 보고하였다.

### 주요 결과

- **KFDA의 성능:** 단일 커널 KFDA만으로도 KISSME, LFDA, XQDA, kLFDA 등 기존의 많은 Metric Learning 방법보다 우수하거나 경쟁력 있는 성능을 보였다. 특히 국소 구조를 학습하는 LFDA 및 kLFDA보다 성능이 높게 나타났다.
- **MKL의 효과:** NP-MFML과 SM-MFML은 단일 KFDA보다 성능을 더욱 향상시켰다. 예를 들어 GRID 데이터셋에서 GOG 특징을 사용할 때 NP-MFML은 Rank-1 정확도 25.76%를 달성하여 baseline들을 상회하였다.
- **SOTA 비교:**
  - **GRID:** NP-MFML이 제안된 모든 Metric Learning 방법 중 가장 높은 성능을 보였으며, 일부 Re-ranking 기반 방법과도 경쟁 가능한 수준이다.
  - **VIPeR:** NP-MFML(50.76%)과 SM-MFML(50.47%)이 많은 Deep Learning 기반 방법 및 기존 Metric Learning 방법들보다 우수한 Rank-1 성능을 보였다.
- **분석 결과:**
  - **구성 요소 영향:** 특징 추출기 단독 $\rightarrow$ KFDA $\rightarrow$ NP/SM-MFML 순으로 정확도가 뚜렷하게 상승함을 CMC 곡선을 통해 확인하였다.
  - **서브스페이스 차원:** 판별 벡터의 수(차원)가 증가함에 따라 정확도가 상승하다가 일정 수준(약 50차원 이상)에서 수렴하는 경향을 보였다.
  - **런타임:** 테스트 시간은 다른 Metric Learning 방법들과 비교하여 유사한 수준(약 0.5~0.6초)으로 매우 효율적이다.

## 🧠 Insights & Discussion

본 논문은 Person re-ID에서 복잡한 보조 정보(예: 국소 구조)를 사용하는 것보다, KFDA를 통한 기본적이지만 강력한 판별 공간 학습이 더 효과적일 수 있음을 시사한다. 특히 **"소규모 데이터셋에서는 클래스당 샘플 수가 적어 국소 구조 추정이 부정확하며, 이것이 LFDA/kLFDA의 성능 저하 원인이 된다"**는 분석은 매우 중요한 통찰이다.

또한, 단일 커널이 가진 편향을 Multiple Kernel Learning을 통해 해결함으로써, 딥러닝 기반 방법들이 대량의 학습 데이터가 필요한 것과 대조적으로, 적은 데이터에서도 매우 경쟁력 있는 성능을 낼 수 있음을 보여주었다. 이는 데이터 획득 비용이 높은 실제 카메라 네트워크 환경에서 본 제안 방법이 실용적인 대안이 될 수 있음을 의미한다.

다만, Re-ranking 기반의 최신 방법들(예: SSM)보다는 여전히 약간 낮은 성능을 보이며, 이는 Metric Learning 단독으로는 해결할 수 없는 갤러리 전체의 구조적 정보 활용 능력이 필요함을 시사한다.

## 📌 TL;DR

본 논문은 Person re-ID를 위해 **KFDA 기반의 Mahalanobis 거리 학습 프레임워크**를 제안하였다. 국소 구조 정보 없이도 강력한 판별력을 가진 KFDA를 기본으로 하며, 여기에 **NP-MFML과 SM-MFML이라는 두 가지 다중 커널 학습 기법을 결합**하여 커널 편향을 제거하고 성능을 극대화하였다. 실험 결과, 소규모 데이터셋 환경에서 기존의 Metric Learning 및 일부 딥러닝 방법보다 우수한 성능과 효율적인 계산 속도를 증명하였다. 이 연구는 데이터가 부족한 환경에서의 효율적인 Person re-ID 시스템 구축에 중요한 기여를 한다.
