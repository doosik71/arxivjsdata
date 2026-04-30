# Ensemble deep learning: A review

M.A. Ganaie, Minghui Hu, A.K. Malik, M. Tanveer, P.N. Suganthan (2022)

## 🧩 Problem to Solve

본 논문은 개별 딥러닝 모델이 가진 한계를 극복하고 일반화 성능(generalization performance)을 향상시키기 위한 방법론인 딥 앙상블 학습(Deep Ensemble Learning)에 대해 다룬다. 딥러닝 아키텍처는 다층 처리 구조를 통해 뛰어난 특징 표현 능력을 보여주지만, 여전히 vanishing/exploding gradients나 degradation problem과 같은 최적화 문제와 높은 계산 비용이라는 병목 현상이 존재한다. 

연구의 핵심 문제는 여러 모델의 예측을 결합하여 단일 모델보다 더 나은 성능을 내는 앙상블 학습의 원리를 딥러닝에 효율적으로 적용하는 것이다. 특히, 딥러닝 모델은 파라미터 수가 방대하여 여러 모델을 학습시키는 데 막대한 비용이 소모되므로, 다양성(diversity)을 어떻게 확보하고, 계산 복잡도를 어떻게 낮추며, 최종 예측값을 어떻게 효과적으로 융합(fusion)할 것인가가 주요 해결 과제이다. 본 논문의 목표는 최신 딥 앙상블 모델들의 상태를 종합적으로 검토하고, 이를 체계적으로 분류하여 연구자들에게 광범위한 요약 정보를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 파편화되어 있던 딥 앙상블 학습 방법론을 체계적인 분류 체계(Taxonomy)로 정리했다는 점에 있다. 저자들은 딥 앙상블 모델을 다음과 같은 기준에 따라 범주화하였다.

1. **전통적 전략**: Bagging, Boosting, Stacking 기반의 딥 앙상블 모델.
2. **학습 메커니즘**: Negative Correlation 기반, Explicit/Implicit 앙상블.
3. **모델 구성**: Homogeneous(동종) 및 Heterogeneous(이종) 앙상블.
4. **의사결정 융합**: 다양한 Decision Fusion 전략.

또한, 단순한 나열을 넘어 앙상블 학습이 성공하는 이론적 배경(Bias-Variance Decomposition, Statistical/Computational/Representational aspects)을 상세히 설명함으로써, 딥 앙상블의 설계 직관을 제공한다.

## 📎 Related Works

논문은 기존의 앙상블 학습 관련 리뷰 논문들이 주로 분류(classification), 회귀(regression), 혹은 클러스터링(clustering)과 같은 특정 작업에 국한되었거나, 생물정보학(bioinformatics)과 같은 특정 도메인에 한정되어 있었다고 지적한다. 일부 종합 리뷰 논문이 존재했으나, 딥러닝 기반의 앙상블 모델만을 전문적으로, 그리고 포괄적으로 다룬 리뷰는 부족했다는 점을 강조하며 본 연구의 차별성을 제시한다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하는 것이 아니라 기존 연구들을 분석하는 리뷰 논문이므로, 딥 앙상블의 이론적 토대와 방법론적 분류를 상세히 설명하는 구조를 취한다.

### 1. 이론적 배경
앙상블 학습의 성공 이유는 다음과 같은 수식과 이론으로 설명된다.

**Bias-Variance Decomposition**: 앙상블 모델의 오차는 편향(bias), 분산(variance), 그리고 공분산(covariance)의 합으로 분해될 수 있다.
$$\text{Error} = \text{bias}^2 + \frac{1}{M}\text{var} + \left(1 - \frac{1}{M}\right)\text{covar}$$
여기서 $M$은 앙상블 크기이며, $\text{bias}$는 베이스 학습기와 타겟 간의 평균 차이, $\text{var}$는 학습기들의 평균 분산, $\text{covar}$는 학습기들 사이의 쌍별 차이를 의미한다. 즉, 다양성이 높아져 공분산이 낮아질수록 전체 오차가 줄어든다.

**다양성(Diversity)**: 모델들이 서로 다른 오류를 범하게 함으로써 이를 상쇄시키는 것이 핵심이다. 데이터 샘플링(Bagging), 가중치 조정(Boosting), 혹은 서로 다른 알고리즘 사용(Heterogeneous)을 통해 다양성을 확보한다.

### 2. 주요 앙상블 전략
- **Bagging**: 중복 허용 샘플링(Bootstrap)을 통해 여러 독립적인 모델을 학습시킨 후, 다수결 투표(Majority Voting)나 평균(Averaging)으로 결과를 통합하여 분산을 줄인다.
- **Boosting**: 약한 학습기(Weak Learner)를 순차적으로 학습시키며, 이전 단계에서 틀린 샘플에 가중치를 두어 편향과 분산을 동시에 줄인다. $\text{ResNet}$과 같은 구조가 Boosting의 원리와 유사하다고 분석한다.
- **Stacking**: 베이스 모델들의 출력을 다시 입력으로 사용하는 메타 학습기(Meta-learner)를 두어 최종 예측을 수행하는 계층적 구조이다.
- **Implicit/Explicit Ensembles**:
    - **Implicit**: Dropout, DropConnect와 같이 하나의 네트워크 내에서 가중치를 공유하며 학습 시 무작위성을 부여해 테스트 시 앙상블 효과를 내는 방식이다.
    - **Explicit**: Snapshot Ensembling처럼 서로 다른 로컬 미니마(local minima)에 도달한 모델들을 저장하여 결합하는 방식으로 가중치를 공유하지 않는다.
- **Homogeneous vs Heterogeneous**: 동일한 알고리즘을 사용하느냐(HOE), 서로 다른 알고리즘(예: CNN + SVM)을 섞어 사용하느냐(HEE)의 차이이다.

### 3. 의사결정 융합(Decision Fusion) 절차
- **Unweighted Averaging**: 단순 평균을 내며, Softmax 확률값 $P_i^j$를 다음과 같이 계산하여 합산한다.
$$P_i^j = \text{softmax}_j(O_i) = \frac{\exp(O_j^i)}{\sum_{k=1}^K \exp(O_k^i)}$$
- **Majority Voting**: 가장 많은 득표를 얻은 클래스를 최종 결과로 선택한다.
- **Bayes Optimal Classifier**: 각 모델의 사후 확률(posterior probability)을 고려하여 최적의 클래스를 선택한다.
- **Stacked Generalization**: 메타 학습기가 최적의 가중치 벡터 $w$를 학습하여 선형 결합한다.
$$f_{\text{stacking}}(x) = \sum_{j=1}^m w_j f_j(x)$$

## 📊 Results

본 논문은 수많은 선행 연구를 분석하여 딥 앙상블 모델의 적용 현황을 정량적으로 제시한다.

- **적용 도메인 분석**: 헬스케어(27%), 이미지 분류(22.5%), 기타(36%), 예측(9%), 음성 인식(5.6%) 순으로 적용되었다. 특히 헬스케어 분야에서는 이종 앙상블(Heterogeneous Ensemble)의 성능이 높게 나타나는 경향이 있다.
- **전략별 채택 비율**: Decision Fusion(29.5%)이 가장 많이 사용되었으며, 그 뒤를 Boosting(18.2%), Stacking(12.5%), Heterogeneous(11.4%), Implicit(10.2%) 순으로 따랐다. Bagging(4.5%)의 비중은 상대적으로 낮았는데, 이는 딥러닝 모델의 학습 비용이 너무 커서 여러 번의 독립적 학습이 어렵기 때문으로 해석된다.

## 🧠 Insights & Discussion

**강점 및 해석**
본 논문은 딥 앙상블의 복잡한 지형을 이론(Bias-Variance) $\rightarrow$ 전략(Bagging/Boosting 등) $\rightarrow$ 융합(Fusion) $\rightarrow$ 응용(Application)으로 이어지는 논리적 흐름으로 정리하여, 새로운 연구자가 어떤 방향으로 모델을 설계해야 할지에 대한 가이드라인을 제공한다. 특히 implicit 앙상블과 snapshot ensembling을 통해 계산 비용 문제를 해결하려는 시도들을 잘 짚어내었다.

**한계 및 비판적 논의**
1. **모델 선택 기준의 부재**: 논문에서도 언급되었듯이, 앙상블에 몇 개의 모델을 넣어야 하는지, 어떤 알고리즘 조합이 최적인지에 대한 일반적인 기준(Criterion)은 여전히 문제 종속적(problem-dependent)이며 명확한 해답이 제시되지 않았다.
2. **계산 비용의 실질적 문제**: 이론적으로는 앙상블이 우수하지만, 실제 대규모 데이터셋에서 수십 개의 딥러닝 모델을 유지하고 추론하는 데 발생하는 메모리 및 지연 시간(latency) 문제에 대한 심층적인 분석은 부족하다.
3. **융합 전략의 단순성**: 많은 연구가 여전히 단순 평균(Naive Averaging)에 의존하고 있으며, 데이터 적응형(data-adaptive) 융합 전략에 대한 연구가 더 필요함을 시사한다.

## 📌 TL;DR

본 논문은 딥러닝의 일반화 성능을 높이기 위한 **딥 앙상블 학습 방법론을 체계적으로 분류하고 분석한 포괄적인 리뷰 논문**이다. 앙상블의 이론적 근거인 Bias-Variance 분해부터 시작하여 Bagging, Boosting, Stacking, Implicit/Explicit 앙상블 등의 전략과 다양한 의사결정 융합 방법을 정리하였다. 헬스케어와 이미지 분류 분야에서 활발히 사용되고 있음을 확인하였으며, 향후 연구 방향으로 **계산 비용 절감을 위한 무작위 모델(Randomized models) 활용, 단일 모델 내 다양성 확보 방안, 그리고 최적의 모델 선택 기준 확립**을 제시한다. 이 연구는 딥러닝 모델의 성능 한계를 돌파하려는 연구자들에게 필수적인 지식 맵을 제공한다.