# Three New Validators and a Large-Scale Benchmark Ranking for Unsupervised Domain Adaptation

Kevin Musgrave, Serge Belongie, Ser-Nam Lim (2023)

## 🧩 Problem to Solve

본 논문은 Unsupervised Domain Adaptation (UDA) 환경에서 모델의 성능을 평가하고 최적의 체크포인트를 선택하기 위한 **Validator(검증기)**의 신뢰성 문제를 해결하고자 한다.

일반적인 지도 학습(Supervised Learning)에서는 레이블이 있는 검증 세트(Validation Set)를 통해 정확도를 직접 계산하여 하이퍼파라미터를 튜닝하고 최적의 모델 체크포인트를 선택한다. 그러나 UDA의 핵심 가정은 타겟 도메인(Target Domain)에 레이블이 전혀 없다는 점이다. 따라서 타겟 도메인의 정확도를 직접 측정할 수 없으며, 대신 Validator를 통해 이를 추정(Estimate)해야 한다.

만약 Validator가 실제 정확도와 낮은 상관관계를 가진다면, 잘못된 체크포인트나 하이퍼파라미터가 선택되어 최종 모델의 성능이 저하되는 결과로 이어진다. 기존 연구들은 대부분 알고리즘 개선에만 집중하고 Validator 연구는 부족했으며, 특히 타겟 레이블을 사용하는 'Oracle Validator'에 의존하여 실제 환경에서의 적용 가능성을 간과했다는 문제가 있다. 본 논문의 목표는 신뢰할 수 있는 UDA Validator를 제안하고, 대규모 벤치마크를 통해 기존 방법들과의 성능을 체계적으로 비교 및 순위를 매기는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **세 가지 새로운 Validator 제안**: 기존 방법론을 개선하거나 새로운 직관을 적용한 **BNM**, **ClassAMI**, **DEVN**을 제안하였다.
2. **대규모 벤치마크 데이터셋 구축**: 10가지 UDA 알고리즘과 100가지 하이퍼파라미터 설정, 31가지 전이 작업(Transfer Tasks)을 조합하여 총 1,000,000개의 모델 체크포인트로 구성된 전례 없는 규모의 데이터셋을 구축하였다.
3. **Weighted Spearman Correlation (WSC) 도입**: 단순한 상관관계를 넘어, 높은 검증 점수를 가졌으나 실제 성능은 낮은 '치명적인 오판' 사례에 더 큰 페널티를 주는 가중치 기반의 평가 지표를 제안하였다.
4. **실증적 분석**: Validator의 선택이 UDA 알고리즘의 선택보다 최종 성능에 더 큰 영향을 미칠 수 있음을 입증하였다.

## 📎 Related Works

논문에서는 기존의 Validator 접근 방식들과 그 한계를 다음과 같이 설명한다.

- **Source Accuracy**: 소스 도메인에서의 정확도를 그대로 사용한다. 소스와 타겟 도메인이 매우 유사하다는 가정이 필요하며, 도메인 간 차이가 클 경우 무용지물이다.
- **Reverse Validation**: 타겟 데이터를 의사 레이블링(Pseudo-labeling)하여 다시 소스 도메인을 예측하는 방식이다. 모델을 두 번 학습시켜야 하므로 시간 비용이 두 배로 든다.
- **Entropy**: 예측 벡터의 엔트로피를 측정하여 모델의 확신도(Confidence)를 평가한다. 모델이 틀린 답에 대해 강한 확신을 갖는 경우(Incorrectly confident) 성능을 잘못 예측한다.
- **Deep Embedded Validation (DEV)**: 도메인 분류기를 통해 타겟 도메인에 속할 확률이 높은 샘플에 가중치를 두어 손실을 계산한다. 점수의 범위가 제한되지 않아(Unbounded) 매우 큰 값이 발생할 수 있는 불안정성이 있다.
- **Proxy Risk**: 별도의 'Check 모델'을 학습시켜 두 모델 간의 의견 불일치 정도를 측정한다. 체크포인트마다 모델을 새로 학습시켜야 하므로 시간 복잡도가 $O(\text{epochs}^2)$로 증가하여 비실용적이다.
- **Ensemble-based Model Selection (EMS)**: 여러 신호를 이용해 회귀 모델을 학습시킨다. 하지만 이를 위해 레이블이 있는 타겟 데이터셋이 별도로 필요하며, 실제 작업에 과적합(Overfitting)될 위험이 있다.
- **Soft Neighborhood Density (SND)**: 타겟 피처들의 유사도 행렬을 통해 클러스터링 품질을 측정한다. 모든 피처가 하나의 거대한 클러스터로 뭉쳐버린 경우, 정확도는 낮음에도 불구하고 높은 점수를 주는 문제가 있다.

## 🛠️ Methodology

### 1. 제안하는 새로운 Validator

본 논문은 기존의 한계를 극복하기 위해 다음 세 가지 Validator를 제안한다.

** (1) Batch Nuclear-Norm Maximization (BNM)**
예측 행렬 $P$ (크기: 샘플 수 $N \times$ 클래스 수 $C$)의 핵 규범(Nuclear Norm)을 계산하여 사용한다. 핵 규범은 $P$의 특이값(Singular Values)들의 합으로 정의된다.
$$\text{BNM} = \|P\|_*$$
이는 예측 결과가 다양하면서도(Diverse) 확신이 있어야(Confident) 높은 값을 가지므로, 이를 타겟 정확도의 대리 지표로 사용한다.

** (2) ClassAMI**
타겟 피처 $F$를 k-means로 클러스터링한 결과와 모델의 예측 레이블 $X$ 사이의 조정 상호 정보량(Adjusted Mutual Information, AMI)을 계산한다.
$$\text{ClassAMI} = \text{AMI}(X, \text{kmeans}(F).\text{labels})$$
단순히 클러스터링의 품질(Silhouette score 등)만 보는 것이 아니라, 모델의 실제 예측값이 클러스터 구조와 얼마나 일치하는지를 평가함으로써 예측 성능을 직접적으로 반영한다.

** (3) DEV with Normalization (DEVN)**
기존 DEV의 불안정성(Unbounded scores)을 해결하기 위해 가중치 $W$를 최대값으로 정규화(Max-normalization)한다.
$$V = \frac{W}{\max W}$$
이를 통해 $\eta$ 값이 비정상적으로 커지는 것을 방지하여 더 안정적인 점수를 산출한다.

### 2. 벤치마크 실험 설계

- **데이터셋**: MNIST $\rightarrow$ MNISTM, Office31, OfficeHome, DomainNet126 총 31개 작업.
- **알고리즘**: ATDOC, BNM, BSP, CDAN, DANN, GVB, IM, MCC, MCD, MMD 총 10가지 UDA 알고리즘 사용.
- **체크포인트 생성**: 각 알고리즘당 100개의 랜덤 하이퍼파라미터 설정 $\times$ 학습 중 20개 시점의 체크포인트 저장.
- **평가 지표 (WSC)**: 단순 Spearman 상관계수가 아닌, 높은 점수/정확도 영역에 가중치를 부여한 **Weighted Spearman Correlation (WSC)**를 사용한다. 가중치 $w_i$는 검증 점수의 순위와 실제 정확도의 순위 중 더 높은 쪽의 제곱으로 설정하여, 상위 모델을 잘못 선택했을 때의 페널티를 극대화한다.
$$\text{WSC} = \frac{\sum_{i=1}^{N} w_i(x_i - \hat{x})(y_i - \hat{y})}{\sqrt{\sum_{i=1}^{N} w_i(x_i - \hat{x})^2 \sum_{i=1}^{N} w_i(y_i - \hat{y})^2}}$$

## 📊 Results

### 1. 정량적 결과 및 순위

- **ClassAMI의 우수성**: 대부분의 UDA 알고리즘에서 ClassAMI가 가장 높은 WSC를 기록하였으며, 특히 MCC, BNM, IM 알고리즘과 결합했을 때 최상의 성능(AATN)을 보였다.
- **데이터셋별 특성**:
  - 소스 검증 정확도(Source Val Accuracy)가 잘 작동하는 Office31, OfficeHome에서는 **DEVN**이 기존 DEV보다 우수하였다.
  - 소스 정확도가 지표로서 기능하지 못하는 MNIST, DomainNet126에서는 **BNM**이 가장 우수한 Validator로 나타났다.
- **Baseline의 재발견**: 놀랍게도 많은 경우에 단순한 **Source validation accuracy**가 다른 복잡한 Validator들보다 더 높은 평균 성능을 보였다.
- **SND의 한계**: SND는 거의 모든 설정에서 다른 Validator들에 비해 일관되게 낮은 성능을 보였다.

### 2. Oracle vs Non-Oracle 비교

- Oracle Validator(타겟 레이블 사용)를 사용할 때는 대부분의 UDA 알고리즘이 Source-only 모델보다 성능이 좋았다.
- 그러나 실제 환경과 동일한 Non-Oracle Validator를 사용할 경우, 체크포인트 선택의 오류로 인해 성능이 수 %p 하락하며, 심지어는 **학습을 하지 않은 Source-only 모델보다 성능이 떨어지는 경우**가 빈번하게 발생하였다.

## 🧠 Insights & Discussion

본 논문은 UDA 분야에서 간과되었던 '모델 선택'의 중요성을 날카롭게 지적한다.

**강점 및 통찰**:

- **Validator의 영향력**: UDA 알고리즘 자체를 변경하는 것보다, 어떤 Validator를 사용하여 체크포인트를 선택하느냐가 최종 정확도에 훨씬 더 결정적인 영향을 미친다는 것을 증명하였다.
- **실무적 가이드라인 제공**: 사용자가 어떤 알고리즘(예: MCC)을 사용할 때 어떤 Validator(예: ClassAMI)를 조합해야 최적의 결과를 얻을 수 있는지에 대한 구체적인 맵을 제시하였다.

**한계 및 논의**:

- **여전한 성능 격차**: 최적의 Validator를 사용하더라도 Oracle Validator와의 성능 격차가 여전히 존재한다. 이는 현재의 UDA Validator들이 타겟 정확도를 완벽하게 추정하지 못하고 있음을 의미하며, 향후 연구의 필요성을 시사한다.
- **알고리즘 의존성**: 특정 Validator가 특정 알고리즘에서만 잘 작동하는 경향이 있어, 모든 UDA 알고리즘에 범용적으로 적용 가능한 'Universal Validator'의 개발이 필요하다.

## 📌 TL;DR

본 연구는 UDA 모델의 최적 체크포인트를 선택하기 위한 Validator들을 대규모(100만 개 체크포인트)로 벤치마킹하고, **ClassAMI, BNM, DEVN**이라는 세 가지 새로운 검증기를 제안하였다. 실험 결과, Validator의 선택이 알고리즘 선택보다 최종 성능에 더 큰 영향을 미치며, 특히 **ClassAMI**가 많은 경우에 최상의 성능을 보였다. 이 연구는 타겟 레이블이 없는 실제 환경에서 UDA 모델을 배포할 때 발생할 수 있는 성능 하락의 원인이 '잘못된 모델 선택'에 있음을 입증하였으며, 향후 Validator 정밀도 향상이 UDA의 잠재력을 끌어올리는 핵심이 될 것임을 시사한다.
