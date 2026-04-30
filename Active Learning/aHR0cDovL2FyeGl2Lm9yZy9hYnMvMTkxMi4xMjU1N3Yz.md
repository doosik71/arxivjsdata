# Active Learning in Video Tracking

Sima Behpour (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 다변량 예측(multivariate prediction) 작업, 특히 비디오 객체 추적(video object tracking)과 같은 구조적 예측(structured prediction) 환경에서 레이블링 비용을 줄이기 위한 효율적인 Active Learning 방법을 구축하는 것이다. 

비디오 추적과 같이 한 프레임 내의 여러 객체를 다음 프레임의 객체와 매칭시켜야 하는 작업은 각 학습 예제마다 많은 양의 레이블이 필요하므로, 모든 데이터를 전수 조사하여 어노테이션(annotation)하는 것은 매우 비용이 많이 드는 작업이다. 이를 해결하기 위해 가장 정보 가치가 높은 데이터만을 선택적으로 레이블링하는 Active Learning의 도입이 필요하다.

특히, 이분 매칭(bipartite matching) 구조를 갖는 예측 문제에서 기존의 확률적 방법론인 Conditional Random Fields (CRF)는 정규화 항(normalization term) 계산이 $\#P$-hard 문제인 matrix permanent 계산을 포함하여 계산 복잡도가 너무 높다는 한계가 있다. 반면, Structured Support Vector Machines (SSVM)는 계산 효율성은 좋으나 Fisher consistency가 결여되어 있어, 이상적인 학습 조건에서도 최적의 매칭을 학습하지 못할 수 있다는 치명적인 단점이 존재한다. 따라서 본 논문의 목표는 계산 효율성과 통계적 일관성을 동시에 확보하면서, 구조적 예측 도메인에 적용 가능한 Active Learning 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 Adversarial Bipartite Matching (ABM)을 통해 확률적 예측의 이점과 계산 효율성을 동시에 달성하고, 이를 Active Learning의 샘플 선택 전략에 결합하는 것이다.

핵심 직관은 학습 과정을 예측자(predictor)와 적대자(adversary) 사이의 제로섬 게임(zero-sum game)으로 모델링하는 것이다. 적대자는 학습 데이터의 통계적 특성을 유지하면서 예측자의 손실을 최대화하는 최악의 분포를 생성하고, 예측자는 이에 대응하여 기대 손실을 최소화하는 방향으로 학습한다. 이러한 적대적 접근 방식은 CRF와 같은 전통적인 확률 모델이 직면하는 계산 복잡성 문제(정규화 항 계산)를 피하면서도, 변수 간의 상관관계를 포함하는 유의미한 확률 분포를 제공한다. 이 분포를 통해 각 변수의 정보 가치를 측정함으로써, 최소한의 레이블링으로 최대의 성능 향상을 끌어낼 수 있는 샘플을 선택할 수 있게 된다.

## 📎 Related Works

논문에서는 구조적 예측을 위한 기존의 두 가지 주요 접근 방식과 그 한계를 다음과 같이 설명한다.

1. **Exponential Family Probabilistic Models (예: CRF):**
   - **특징:** 통계적 일관성(statistical consistency)을 보장한다.
   - **한계:** 이분 매칭 구조에서 분포의 정규화 항을 계산하는 것이 계산적으로 불가능(intractable)할 정도로 복잡하다.

2. **Structured Support Vector Machines (SSVM):**
   - **특징:** 계산 효율성이 뛰어나며 대규모 데이터셋에 적용 가능하다.
   - **한계:** Fisher consistency가 부족하여, 데이터의 진정한 분포가 주어지더라도 최적의 매칭 함수를 학습하지 못하는 경우가 발생한다. 또한, 확률적 출력이 기본적으로 제공되지 않아 Active Learning을 위한 불확실성(uncertainty) 측정에 어려움이 있다.

본 논문이 제안하는 ABM 방식은 적대적 학습 프레임워크를 사용하여 SSVM의 효율성과 CRF의 통계적 특성을 동시에 확보함으로써 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
본 논문은 Adversarial Bipartite Matching (ABM)을 기반으로 한 Active Learning 파이프라인을 제안한다. 전체 흐름은 다음과 같다:
초기 소량의 레이블링 데이터($D_l$)로 ABM 모델을 학습시킨 후, 미레이블 풀($D_u$)에서 가장 정보 가치가 높은 샘플을 선택하여 오라클(Oracle)에게 레이블을 요청하고, 이를 다시 학습 세트에 추가하여 모델을 갱신하는 반복적인 구조를 가진다.

### Adversarial Bipartite Matching (ABM)
ABM은 예측자와 적대자 간의 Minimax 게임으로 정의된다. 예측자 $\hat{P}$는 기대 손실을 최소화하려 하고, 적대자 $\check{P}$는 제약 조건 내에서 이를 최대화하려 한다.

$$
\min_{\hat{P}} \max_{\check{P}} \mathbb{E}_{x \sim \tilde{P}; \check{y}|x \sim \check{P}; \hat{y}|x \sim \hat{P}} [\text{loss}(\hat{Y}, \check{Y})]
$$

여기서 $\text{loss}(\hat{Y}, \check{Y})$는 Hamming loss로 정의되며, 이는 매칭된 노드 쌍 중 틀린 개수의 합을 의미한다. 적대자는 학습 데이터의 특성 함수 $\phi$에 기반한 통계적 제약 $\mathbb{E}_{x \sim \tilde{P}; \check{y}|x \sim \check{P}} [\phi(X, \check{Y})] = \tilde{c}$를 만족해야 한다.

계산 효율성을 위해 **Double Oracle** 방법을 사용하여 게임의 평형 상태를 지원하는 활성 제약 세트를 생성하며, 각 단계에서 최적의 반응(best response)을 찾기 위해 **Hungarian algorithm** (Kuhn-Munkres algorithm)을 사용하여 $O(|V|^3)$ 시간 복잡도로 최대 가중치 매칭을 수행한다.

### Active Learning 샘플 선택 전략
단순한 유니베리에이트(uni-variate) 선택이 아니라, 변수 간의 상관관계를 고려한 다변량 선택 전략을 사용한다. 특정 변수 $Y_j$를 관측함으로써 얻을 수 있는 전체 변수들에 대한 불확실성 감소량(정보 이득) $V_j$를 상호 정보량(Mutual Information)으로 계산한다.

$$
V_j = \sum_{i=1}^{n} H(Y_i | D_l) - \sum_{y_j \in Y} P(y_j | D_l) \sum_{i=1}^{n} H(Y_i | D_l, y_j) = \sum_{i=1}^{n} I(Y_i ; Y_j | D_l)
$$

이 상호 정보량 값은 ABM의 적대적 평형 분포에서 도출된 쌍별 주변 확률(pairwise marginal probabilities) $\check{P}(y_i, y_j)$를 통해 효율적으로 계산될 수 있다. 최종적으로 모든 노드에 대한 정보 이득의 합이 최대인 샘플 $X^*$를 선택하여 쿼리한다.

## 📊 Results

### 실험 설정
- **작업 및 데이터셋:** 비디오 프레임 간 객체 추적(Object Tracking) 작업을 수행하였으며, MOT challenge 데이터셋의 TUD 및 ETH 데이터셋을 사용하였다. 
- **특수 처리:** 객체의 진입과 퇴장을 처리하기 위해 각 프레임의 노드 수를 최대 객체 수 $N^*$의 두 배인 $n=2N^*$로 설정하여 'invisible' 노드를 추가하였다.
- **비교 대상 (Baseline):** Platt scaling을 적용하여 확률적 출력을 생성하도록 구현된 Active SSVM을 기준으로 삼았다.
- **지표:** Hamming Loss를 통해 정확도를 측정하였다.

### 주요 결과
- **정량적 성능:** 실험 결과, Active ABM은 초기 반복 단계에서는 다소 혼재된 성능을 보였으나, 전반적으로 Active SSVM보다 더 낮은 Hamming Loss를 기록하며 우수한 성능을 보였다. 이는 ABM이 Platt scaling 방식보다 더 정교한 불확실성 모델을 제공하기 때문으로 분석된다.
- **추론 시간:** ABM은 평형 상태를 구축하기 위해 여러 개의 순열(permutations)을 사용하므로, 단일 순열을 사용하는 SSVM보다 추론 속도가 약 6~20배 정도 느린 것으로 나타났다. 다만, 학습 과정에서는 이전의 평형 상태 전략을 캐싱하여 재사용함으로써 학습 시간은 다른 방법들과 유사한 수준으로 유지하였다.

## 🧠 Insights & Discussion

본 논문은 구조적 예측 문제, 특히 이분 매칭 작업에서 확률적 모델의 통계적 이점과 결정론적 모델의 계산 효율성 사이의 간극을 적대적 학습(adversarial learning)으로 성공적으로 메웠다. 

**강점:**
- 기존의 CRF가 가진 계산 불가능성 문제를 적대적 게임 프레임워크로 해결하여, 실질적으로 사용 가능한 확률적 분포를 얻어냈다.
- 이 분포를 통해 계산된 상호 정보량을 Active Learning에 도입함으로써, 단순히 불확실성이 높은 샘플을 뽑는 것보다 구조적 관계를 고려한 효율적인 샘플링이 가능함을 입증하였다.

**한계 및 논의:**
- 추론 단계에서의 시간 복잡도가 SSVM 대비 높다는 점은 실시간 시스템 적용 시 제약 사항이 될 수 있다.
- 본 연구는 이분 매칭에 집중되어 있으나, 제안된 적대적 프레임워크가 다른 구조적 예측 문제(예: 그래프 컷, 객체 인식)에서도 동일한 효율성을 보일지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 비디오 객체 추적과 같은 이분 매칭 구조의 예측 문제에서 레이블링 비용을 줄이기 위한 **Adversarial Bipartite Matching (ABM) 기반의 Active Learning** 방법을 제안한다. 적대적 학습을 통해 계산 효율성과 통계적 일관성을 동시에 확보한 확률 분포를 생성하고, 이를 통해 도출된 상호 정보량을 기반으로 최적의 학습 샘플을 선택한다. 실험을 통해 기존 SSVM 기반 방식보다 높은 정확도를 달성하였으며, 이는 복잡한 구조적 데이터의 레이블링 효율을 높이는 데 기여할 수 있다.