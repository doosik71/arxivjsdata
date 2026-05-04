# Bounded-Distortion Metric Learning

Renjie Liao, Jianping Shi, Ziyang Ma, Jun Zhu, Jiaya Jia (2015)

## 🧩 Problem to Solve

본 논문은 Metric Learning 과정에서 발생하는 과적합(Overfitting)과 수치적 불안정성(Numerical Inaccuracy) 문제를 해결하고자 한다. 일반적으로 Metric Learning은 데이터를 잘 표현하기 위해 메트릭 공간을 변형시키는데, 이때 발생하는 '왜곡(Distortion)'이 너무 크면 훈련 데이터에 과하게 맞추어지는 과적합이 발생할 가능성이 높다.

또한, Mahalanobis Metric Learning의 경우, 메트릭을 결정하는 파라미터 행렬 $M$의 왜곡도가 매우 높으면 행렬의 Condition Number가 커지게 되어 ill-conditioned 상태가 되며, 이는 수치 계산의 부정확함으로 이어진다. 따라서 본 연구의 목표는 왜곡도에 상한선을 두는 Bounded-Distortion Metric Learning (BDML) 프레임워크를 제안하여, 데이터 적합성과 모델의 안정성 및 일반화 성능 사이의 균형을 맞추는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mahalanobis 메트릭 공간의 왜곡도를 제어하기 위해 **Condition Number $\kappa(M)$에 제약 조건을 부과**하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **BDML 프레임워크 제안**: 메트릭 학습 목적 함수에 왜곡도 상한 제약($\kappa(M) \le K$)을 추가하여 일반화 능력을 향상시켰다.
2.  **효율적인 최적화 알고리즘**: Multiplicative Weights Update (MWU) 방법을 기반으로 한 Bisection 알고리즘을 제안하여 convex feasibility 문제를 효율적으로 해결한다.
3.  **Pseudo-metric 확장**: $M$이 Positive Semidefinite (PSD)인 경우로 확장하여, Semidefinite Relaxation (SDP)과 Randomized Algorithm을 통해 차원 축소와 메트릭 학습을 동시에 수행하는 방법을 제시하였다.
4.  **이론적 분석**: 왜곡도가 알고리즘의 안정성(Stability)과 일반화 오차(Generalization Bound)에 직접적인 영향을 미침을 수학적으로 증명하였다.

## 📎 Related Works

기존의 Metric Learning 연구들은 주로 다음과 같은 기준에 따라 분류된다.
- **제약 조건 기준**: Pairwise 방식(dissimilar한 쌍의 거리를 일정 임계값 이상으로 유지)과 Triplet-wise 방식(Large-Margin Nearest Neighbor, LMNN 등, target neighbor보다 imposter neighbor가 더 멀리 있도록 유지)으로 나뉜다.
- **정규화 방식**: $\log \det(M)$이나 Frobenius norm 등을 사용하여 모델의 복잡도를 제어하려는 시도가 있었다.

본 논문은 기존의 정규화 방식들이 메트릭 공간의 '상대적인' 변형 정도를 직접적으로 제어하지 못한다는 점을 지적한다. 예를 들어, $\log \det$나 F-norm 값이 작더라도 타원체가 매우 길쭉한 형태(ill-conditioned)가 될 수 있으며, 이는 왜곡도가 높음을 의미한다. BDML은 Condition Number를 직접 제약함으로써 이러한 한계를 극복하고 메트릭 공간의 기하학적 구조를 더 안정적으로 제어한다.

## 🛠️ Methodology

### 1. 왜곡도의 정의 및 기하학적 의미
Mahalanobis 거리 함수는 $d_M(x_i, x_j) = \sqrt{(x_i-x_j)^T M (x_i-x_j)}$로 정의되며, 여기서 $M$은 Positive Definite (PD) 행렬이다. 본 논문은 Euclidean 공간에서 Mahalanobis 공간으로의 임베딩 시 발생하는 왜곡도가 행렬 $M$의 **Condition Number $\kappa(M)$**와 동일함을 Proposition 1을 통해 보였다.
$$\kappa(M) = \frac{\lambda_{\max}(M)}{\lambda_{\min}(M)}$$
이는 타원체의 가장 긴 축과 가장 짧은 축의 비율을 의미하며, 이 값이 작을수록 메트릭 공간이 Euclidean 공간의 구조를 더 잘 유지함을 뜻한다.

### 2. BDML의 공식화
본 논문은 두 가지 형태의 BDML을 제안한다. $S$는 target neighbors(같은 클래스) 집합, $I$는 imposter neighbors(다른 클래스) 집합이다.

- **p-BDML (Pair-constrained)**: target neighbor 간의 평균 거리를 최소화하면서, imposter neighbor 간의 거리가 $\mu$ 이상이 되도록 하며 왜곡도를 제한한다.
$$\min_{M \in \mathcal{P}_d^R} \frac{1}{n} \sum_{(i,j) \in S} M \bullet X_{ij} \quad \text{s.t.} \quad M \bullet X_{ij} \ge \mu, \forall (i,j) \in I, \quad \kappa(M) \le K$$
- **t-BDML (Triplet-constrained)**: imposter 거리와 target 거리의 차이가 $\mu$ 이상이 되도록 제한한다.
$$\min_{M \in \mathcal{P}_d^R} \frac{1}{n} \sum_{(i,j) \in S} M \bullet X_{ij} \quad \text{s.t.} \quad M \bullet X_{ik} - M \bullet X_{ij} \ge \mu, \forall (i,j,k) \in T, \quad \kappa(M) \le K$$
(여기서 $M \bullet X_{ij}$는 Frobenius 내적으로, $(x_i-x_j)^T M (x_i-x_j)$와 같다.)

### 3. 최적화 알고리즘: Bisection & MWU
Condition Number 제약 조건은 non-convex하지만 quasi-convex 성질을 가지므로, 이를 convex feasibility 문제로 변환하여 해결할 수 있다.

1.  **Bisection Method**: 최적의 목적 함수 값 $g^*$를 찾기 위해 구간 $[L, U]$를 설정하고, 반복적으로 중간값 $\bar{g}$를 이용해 feasibility 문제를 푼다.
2.  **MWU (Multiplicative Weights Update)**: 
    - 각 제약 조건에 가중치 $w_i$를 부여한다.
    - 가중치에 따라 구성된 가중 합 제약 조건을 만족하는 $Y$를 찾는 ORACLE을 호출한다.
    - 제약 조건 만족 여부에 따라 가중치를 업데이트한다: $w_i^{(t+1)} = w_i^{(t)}(1 - \epsilon \eta_i^{(t)})$.
    - 최종적으로 $Y$들의 평균값을 통해 근사해를 구한다.

### 4. Pseudo-metric 및 차원 축소
$M$이 PSD인 경우, $M = Q^T \Lambda Q$ (여기서 $Q$는 orthogonal matrix, $\Lambda$는 diagonal matrix)로 분해할 수 있다.
- $\Lambda$를 고정하면 $Q$를 찾는 문제는 Quadratic Constrained Quadratic Programming (QCQP)가 된다.
- 이를 SDP(Semidefinite Programming)로 완화(Relaxation)하여 해결한 후, **Gaussian Randomization Procedure**를 통해 $Q$의 근사해를 얻는다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: UCI 데이터셋 (Wine, Iris, Diabetes, Waveform, Segment), Domain Adaptation 데이터셋 (Caltech, Amazon, Webcam, Dslr), LFW (Face Verification).
- **비교 대상**: Euclidean, Xing, LMNN, ITML, BoostMetric 등.
- **지표**: Test Error (Classification), Accuracy (Domain Adaptation, Face Verification).

### 2. 주요 결과
- **Classification (UCI)**: t-BDML이 p-BDML보다 일관되게 우수한 성능을 보였으며, 특히 t-BDML은 여러 데이터셋에서 최상위권의 성능을 기록하였다.
- **Domain Adaptation**: Pseudo-metric learning scheme을 적용한 t-BDML이 비지도 및 준지도 학습 설정 모두에서 경쟁 모델보다 우수한 정확도를 보였다.
- **Face Verification (LFW)**: p-BDML은 단순한 SIFT 특징량을 사용했음에도 불구하고, 복잡한 특징량 결합 모델들과 대등한 성능을 냈으며, 특히 표준 편차가 매우 낮아 메트릭의 안정성이 높음을 입증하였다.
- **왜곡도와 오차의 관계**: 실험 결과, $\log \kappa(M)$ 값이 너무 작으면 Underfitting이 발생하고, 너무 크면 Overfitting이 발생하여 오차가 증가하는 **U-자형 곡선**이 나타났다. 이는 적절한 $K$ 값의 설정이 필수적임을 보여준다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 뒷받침
본 논문은 단순히 실험적 결과만 제시한 것이 아니라, Uniform-Replace-One stability 분석을 통해 왜곡도 상한 $K$가 일반화 오차 상한(Generalization Bound)과 직결됨을 수학적으로 증명하였다. 특히 $\beta = 4(K+1)R\Gamma^2/d$라는 안정성 지표를 통해, $K$를 통해 모델의 안정성을 직접 제어할 수 있음을 보였다.

### 2. 한계 및 비판적 해석
- **파라미터 의존성**: 상한 값 $K$의 선택이 성능에 큰 영향을 미치며, 이를 위해 교차 검증(Cross-validation)이 필요하다는 점은 실무적 부담이 될 수 있다.
- **가정의 제약**: 이론적 분석에서 $M$이 full rank라는 가정을 사용하였다. 논문에서도 언급되었듯이, $M$이 rank-deficient한 경우(예: 매우 희소한 고차원 공간)에는 본 논문의 일반화 분석이 그대로 적용되지 않는다.
- **계산 복잡도**: MWU 기반의 Bisection 방법은 수렴할 때까지 반복적인 ORACLE 호출이 필요하므로, 매우 대규모 데이터셋에서의 효율성에 대한 추가 검토가 필요해 보인다.

## 📌 TL;DR

본 연구는 Mahalanobis 메트릭 학습 시 행렬의 **Condition Number를 제한하는 Bounded-Distortion Metric Learning (BDML)** 프레임워크를 제안하였다. 왜곡도를 적절히 제한함으로써 과적합을 방지하고 수치적 안정성을 확보할 수 있음을 이론적으로 증명하고 실험적으로 입증하였다. 이 연구는 특히 고차원 데이터의 메트릭 학습에서 일반화 성능을 높이기 위한 정규화 전략으로 활용될 가능성이 높다.