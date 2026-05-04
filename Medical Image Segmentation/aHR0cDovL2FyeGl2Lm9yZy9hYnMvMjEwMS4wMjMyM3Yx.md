# Diminishing Uncertainty within the Training Pool: Active Learning for Biomedical Image Segmentation

Vishwesh Nath, Dong Yang, Bennett A. Landman, Daguang Xu, Holger R. Roth (2019)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 딥러닝 모델 학습에 필요한 방대한 양의 레이블링된 데이터 확보가 어렵다는 점을 해결하고자 한다. 의료 영상의 특성상 숙련된 방사선 전문의가 데이터를 레이블링하는 데 매우 많은 시간과 비용이 소요되므로, 적은 양의 데이터만으로도 전체 데이터를 사용했을 때와 유사한 성능을 달성할 수 있는 효율적인 데이터 선택 전략이 필요하다.

특히, 기존의 Active Learning(AL) 방식들이 주로 분류(Classification) 작업에 집중되어 있었으며, 분할 작업의 경우 데이터셋에 따라 성능 편차가 크고 Random acquisition(무작위 선택)이라는 강력한 베이스라인을 극복하는 것이 어렵다는 점을 문제로 제기한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단순히 새로운 데이터를 추가하는 것을 넘어, **학습 풀(Training Pool) 내의 불확실성을 감소시키기 위해 '어려운 샘플(Hard-samples)'의 빈도를 높이는 전략**을 제안하는 것이다.

주요 기여 사항은 다음과 같다:
1. **데이터 중복 활용(Data Duplication):** 선택된 데이터 포인트를 무조건 unlabeled pool에서 제거하는 대신, 불확실성이 높은 샘플을 학습 풀에 중복해서 포함시켜 모델이 해당 케이스에 더 집중하게 함으로써 주석 비용을 절감하고 성능을 높인다.
2. **Mutual Information(MI)을 통한 다양성 확보:** 데이터 중복으로 인해 발생할 수 있는 편향을 방지하기 위해, 학습 풀과 unlabeled pool 사이의 Mutual Information을 정규화 항으로 사용하여 학습 데이터셋의 다양성을 보장한다.
3. **SVGD 기반의 Query-by-Committee(QBC) 구현:** 앙상블 모델들의 학습을 효율적으로 수행하기 위해 Stein Variational Gradient Descent(SVGD)를 도입하고, 이를 분할 작업에 맞게 Dice log-likelihood로 최적화하였다.

## 📎 Related Works

### 기존 연구 및 한계
- **Active Learning 프레임워크:** Uncertainty sampling, Query-by-Committee(QBC), Variance reduction 등이 제안되었으나, 대부분 분류 작업에 치중되어 있다.
- **의료 영상 분할의 AL:** 단순한 엔트로피 기반의 획득 함수는 의료 영상의 복잡성으로 인해 성능이 제한적이며, 많은 경우 Random selection보다 크게 우월하지 않은 결과를 보였다.
- **불확실성 추정 방식:** MC-Dropout, Bayesian Neural Networks, Ensemble-based methods 등이 사용된다. 하지만 MC-Dropout은 하이퍼파라미터 튜닝이 어렵고, Bayesian 방식보다 Ensemble 방식이 실무적으로 더 우수한 성능을 보인다는 보고가 있다.

### 본 연구의 차별점
본 연구는 기존 AL이 unlabeled pool에서 데이터를 선택하고 제거하는 것에만 집중한 것과 달리, **이미 레이블링된 데이터를 어떻게 더 효율적으로 재사용(중복 배치)할 것인가**에 초점을 맞춘다. 또한, 단순 앙상블이 아닌 SVGD를 통해 입자(particles) 간의 반발력을 이용함으로써 모델들이 서로 다른 지역 최적점(local optima)을 찾도록 유도하여 더 정교한 불확실성 추정을 가능하게 한다.

## 🛠️ Methodology

### 전체 시스템 구조
본 프레임워크는 U-Net 기반의 아키텍처를 사용하며, SVGD를 통해 최적화된 모델 앙상블(Committee)을 구성하여 unlabeled pool에서 가장 유용한 데이터를 선택하고 이를 학습 풀에 추가하는 순환 구조를 가진다.

### 주요 구성 요소 및 방법론

#### 1. Stein Variational Gradient Descent (SVGD)
SVGD는 여러 개의 모델 파라미터 집합 $\Theta = \{\theta_q\}_{q=1}^M$ (particles)을 동시에 업데이트하는 joint optimizer이다. 각 입자는 다음과 같은 업데이트 규칙을 따른다:

$$\theta_{k+1} \leftarrow \theta_k + \epsilon_k \phi(\theta_k)$$

여기서 $\phi(\theta_k)$는 다음과 같이 정의된다:
$$\phi(\theta_k) = \frac{1}{M} \sum_{j=1}^M [r(\theta_j^k, \theta_k) \nabla_{\theta_j^k} \log p(\theta_j^k) + \nabla_{\theta_j^k} r(\theta_j^k, \theta_k)]$$

- $\epsilon_k$는 step-size이며, $r(\cdot, \cdot)$은 RBF 커널이다.
- 첫 번째 항은 로그 가능도(log-likelihood)를 최대화하는 방향으로 이동하게 하며, 두 번째 항은 입자들 사이의 **반발력(repulsive force)**으로 작용하여 모델들이 서로 중복되지 않고 다양한 분포를 갖게 한다.

#### 2. Dice log-likelihood
의료 영상의 클래스 불균형 문제를 해결하기 위해, 기존의 cross-entropy 대신 Dice loss를 기반으로 한 log-likelihood를 SVGD에 적용하였다. Dice loss $L_{Dice}$는 다음과 같다:

$$L_{Dice}(y, \hat{y}) = 1 - \frac{2 \sum_{i=1}^n y_i \hat{y}_i}{\sum_{i=1}^n y_i^2 + \sum_{i=1}^n \hat{y}_i^2}$$

학습 시 $\log(L_{Dice})$를 목적 함수로 사용하여 SVGD를 수행한다.

#### 3. 데이터 획득 함수 (Acquisition Function)
데이터를 선택하기 위해 Epistemic uncertainty(모델 불확실성)와 Mutual Information(MI)을 결합한 스코어를 사용한다.

- **Entropy-based Uncertainty ($H$):** 각 모델 입자가 예측한 확률 분포의 엔트로피를 계산하여 3D 맵의 합으로 스코어를 산출한다.
- **Mutual Information ($MI$):** 학습 풀 $T$와 unlabeled pool $U$ 사이의 이미지 유사도를 측정하여 데이터의 다양성을 확보한다.
  $$MI = \sum_{x_i^T} \sum_{x_j^U} P(x_i^T, x_j^U) \log \frac{P(x_i^T, x_j^U)}{P(x_i^T)P(x_j^U)}$$
- **최종 선택 스코어:**
  $$\text{Score} = \alpha(H(x_i^U)) - \frac{1}{\alpha}(MI(x_i^U, T))$$
  불확실성이 높고($H \uparrow$), 기존 학습 데이터와의 유사도가 낮은($MI \downarrow$) 샘플이 높은 우선순위를 갖게 된다.

#### 4. 학습 절차 (Data Duplication)
- **Delete Flag:** $\text{Delete} = \text{True}$인 경우 선택된 데이터를 $U$에서 제거한다. $\text{Delete} = \text{False}$인 경우, 이미 레이블링된 데이터라도 불확실성이 높다면 다시 선택되어 학습 풀 $T$에 중복 추가된다. 이를 통해 모델이 어려운 샘플에 더 많이 노출되도록 유도한다.
- 모든 AL 반복(iteration)마다 모델을 처음부터 다시 학습(train from scratch)시킨다.

## 📊 Results

### 실험 설정
- **데이터셋:** Medical Segmentation Decathlon (MSD) 2018의 Hippocampus MRI 및 Pancreas/Tumor CT.
- **지표:** Dice-Sorensen Coefficient (DSC).
- **비교 대상:** Random selection, 단순 Ensemble, Delete vs NoDelete 전략, MI 포함 여부 등.

### 주요 결과
1. **Hippocampus 데이터셋:**
   - 제안 방법인 `Epistemic + MI + NoDelete` 조합이 가장 효율적이었다.
   - 전체 데이터의 **22.69%만 사용하고도** 전체 데이터를 사용했을 때와 대등하거나 더 높은 성능을 달성하였다.
   - 이는 데이터 중복 활용을 통해 어려운 경계 영역의 불확실성을 효과적으로 감소시켰기 때문이다.

2. **Pancreas 데이터셋:**
   - `NoDelete` 방식이 `Delete` 방식보다 사용 데이터 양은 적었으나(예: 48.85% vs 97.01%), 최종 성능은 `Delete` 방식이 더 높게 나타났다.
   - 이는 Pancreas가 장기 크기가 작고 배경이 복잡하여, 단순 중복보다는 더 많은 고유 샘플을 확보하는 것이 유리했음을 시사한다.

3. **SVGD vs Ensemble:**
   - SVGD 기반 모델이 단순 앙상블보다 수렴 속도가 빠르고 분산이 적으며, 더 안정적인 수렴 곡선을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **데이터 효율성:** 특히 Hippocampus와 같은 데이터셋에서 극소수의 데이터만으로 고성능을 낼 수 있음을 입증하여, 레이블링 비용 절감 가능성을 보여주었다.
- **불확실성 제어:** TSNE 시각화 결과, AL을 통해 선택된 데이터들이 클러스터의 경계면에 위치하며 클래스 간 균형을 맞추는 방향으로 선택됨을 확인하였다.

### 한계 및 비판적 논의
- **데이터셋 의존성:** 제안한 `NoDelete`(데이터 중복) 전략이 모든 데이터셋에서 동일하게 작동하지 않았다. 저자들은 이를 "AL은 작업과 데이터에 매우 의존적이다"라고 결론지으며, Pancreas와 같이 작은 장기의 경우 전체 볼륨 단위가 아닌 더 세밀한 **패치 단위의 AL 전략**이 필요할 수 있음을 언급한다.
- **계산 비용:** 매 iteration마다 모델을 처음부터 다시 학습시켜야 하므로 GPU 연산 시간이 매우 길다 (Pancreas의 경우 총 160 GPU hours).
- **전처리 영향:** 최적의 하이퍼파라미터나 데이터 증강(Augmentation)을 적용하지 않은 상태에서의 실험이므로, 이를 적용했을 때의 절대적 성능 향상 폭은 더 클 것으로 예상된다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 레이블링 비용을 줄이기 위해 **불확실성이 높은 데이터를 학습 풀에 중복 배치하는 Active Learning 전략**을 제안한다. 이를 위해 **SVGD 최적화 기반의 모델 앙상블**과 **Mutual Information을 이용한 다양성 규제**를 도입하였다. 실험 결과, Hippocampus 데이터셋에서는 전체 데이터의 약 23%만으로 전체 성능을 구현하는 획기적인 효율성을 보였으나, Pancreas 데이터셋에서는 성능 향상이 제한적이었다. 이는 AL 전략이 데이터의 특성(장기의 크기, 차원 등)에 따라 매우 민감하게 반응함을 시사하며, 향후 더 세밀한 패치 단위 선택 전략의 필요성을 제시한다.