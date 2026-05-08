# Active Learning with Importance Sampling

Muni Sreenivas Pydi, Vishnu Suresh Lokhande (2019)

## 🧩 Problem to Solve

본 논문은 대규모의 레이블이 없는 데이터 풀(unlabeled pool)과 소규모의 레이블이 있는 데이터 풀(labeled pool)이 존재하는 Active Learning 환경을 다룬다. Active Learning의 핵심 문제는 오라클(oracle)에게 레이블을 요청하는 데 드는 계산적, 금전적 비용이 높기 때문에, 모델의 일반화 성능(generalization)을 가장 효과적으로 향상시킬 수 있는 데이터 포인트들을 '현명하게' 선택하는 것이다.

본 연구의 목표는 중요도 샘플링(Importance Sampling) 이론을 기반으로 하여, 매 반복(iteration)마다 모델의 실제 손실(true loss)을 최소화할 수 있는 최적의 확률적 쿼리 분포(probabilistic querying distribution)를 설계하고 이를 통해 효율적인 데이터 선택 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 데이터 선택 과정을 확률적 샘플링 문제로 정의하고, 중요도 샘플링을 통해 전체 데이터셋의 손실에 대한 편향되지 않은 추정치(unbiased estimate)를 얻는 것이다. 주요 기여 사항은 다음과 같다.

1. **ALIS(Active Learning with Importance Sampling) 알고리즘 제안**: 중요도 샘플링을 기반으로 쿼리할 데이터 포인트를 선택하는 확률적 절차를 도입하였다.
2. **실제 손실의 상한선(Upper Bound) 도출**: 임의의 확률적 샘플링 절차에 대해 알고리즘이 겪게 될 실제 손실의 확률적 상한선을 수학적으로 유도하였다.
3. **최적 샘플링 분포 도출**: 유도된 상한선을 직접적으로 최소화하는 최적의 샘플링 확률 분포 $(p^*)_t$를 정의하였다.
4. **이론적 우위 증명**: 최적 샘플링 분포를 사용하는 것이 일반적인 균등 샘플링(uniform sampling)보다 실제 손실의 상한선을 더 낮게 유지함을 수학적으로 증명하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 본 제안 방법과의 차이점을 설명한다.

* **Beygelzimer et al. [2009]**: 중요도 가중치 샘플링을 사용한 Active Learning을 다루었으나, 데이터가 온라인 방식으로 들어오는 상황을 가정하여 데이터 선택의 자유도가 낮다. 반면 ALIS는 전체 언레이블 풀을 미리 가지고 있어, pseudo-loss와 같은 추가 정보를 활용해 샘플링 분포를 세밀하게 조정할 수 있다. 또한, 기존 연구는 유한한 가설 공간(finite hypothesis class)에 제한이 있으나 ALIS는 그러한 제한이 없다.
* **Sener and Silvio [2018]**: 본 논문과 유사한 실제 오차 분해(true error decomposition) 방식을 사용하지만, 데이터 선택 시 결정론적(deterministic) 방법인 코어셋(core-sets) 접근 방식을 취한다. 이에 비해 ALIS의 확률적 접근 방식은 계산 비용이 훨씬 적다는 장점이 있다.

## 🛠️ Methodology

### 전체 파이프라인

ALIS 알고리즘은 다음과 같은 반복적인 절차를 통해 진행된다.

1. 현재 모델 $A_t$를 사용하여 언레이블 데이터 $U_t$에 대한 **pseudo-label** $\hat{y} = -\text{sign}(f^{A_t}(x))$을 생성한다.
2. pseudo-label 기반의 손실을 계산하여 최적의 샘플링 확률 분포 $p_t$를 계산한다.
3. 분포 $p_t$에 따라 언레이블 풀에서 데이터 포인트 $V_t$를 샘플링한다.
4. 오라클로부터 $V_t$에 대한 실제 레이블을 획득하고 학습 세트 $S$를 업데이트한다.
5. 가중치가 적용된 손실 함수 $\sum_{x_k \in V_t} \frac{1}{p_t^k} l(z_k, A_{t+1})$을 최소화하도록 모델을 재학습하여 $A_{t+1}$을 얻는다.

### 상세 방법 및 방정식 설명

#### 1. 실제 손실의 분해 (True Loss Decomposition)

알고리즘 $A_t$의 실제 손실 $\mathbb{E}_Z[l(z, A_t)]$은 다음과 같이 세 가지 항으로 분해된다.
$$ \mathbb{E}_Z[l(z, A_t)] \leq \underbrace{|\mathbb{E}_Z[l(z, A_t)] - \frac{1}{n_t} \sum_{j \in [n_t]} l(z_{u_t(j)}, A_t)|}_{(A)} + \underbrace{\frac{1}{n_t} \sum_{j \in [n_t]} \frac{Q_{u_t(j)}^t}{p_{u_t(j)}^t} l(z_{u_t(j)}, A_t)}_{(B)} + \underbrace{|\frac{1}{n_t} \sum_{j \in [n_t]} l(z_{u_t(j)}, A_t) - \frac{1}{n_t} \sum_{j \in [n_t]} \frac{Q_{u_t(j)}^t}{p_{u_t(j)}^t} l(z_{u_t(j)}, A_t)|}_{(C)} $$

* **(A) 일반화 오차(Generalization Error)**: 모든 언레이블 데이터에 대한 평균 손실과 실제 분포 간의 차이이다.
* **(B) 가중 평균 손실(Weighted Average Loss)**: 샘플링된 데이터에 대해 $1/p_t$ 가중치를 곱한 손실의 합이다. 이는 중요도 샘플링 원리에 의해 전체 언레이블 데이터 손실의 **편향되지 않은 추정치(unbiased estimate)**가 된다.
* **(C) 샘플링 편차**: 전체 평균 손실과 가중 평균 손실 간의 절대적 차이이며, 이는 샘플링 분포 $p_t$에 따른 분산(variance)에 의존한다.

#### 2. Pseudo-loss의 도입

실제 손실 $l(z, A_t)$는 레이블 $y$를 모르기 때문에 직접 계산할 수 없다. 따라서 본 논문은 모델이 예측한 값의 반대 부호를 레이블로 가정하는 pseudo-label $\hat{y}$를 도입하여, 항상 실제 손실보다 크거나 같은 **pseudo-loss** $l(\hat{z}, A_t)$를 계산한다.
$$ l(z_{u_t(j)}, A_t) \leq l(\hat{z}_{u_t(j)}, A_t) $$

#### 3. 최적 샘플링 분포 $(p^*)_t$

분석 결과, 실제 손실의 상한선을 최소화하기 위해서는 항 (C)의 분산을 최소화해야 하며, 이는 다음과 같은 확률 분포를 가질 때 달성된다.
$$ (p^*)_t^{u_t(j)} = \frac{l(\hat{z}_{u_t(j)}, A_t)^{1/2}}{\sum_{j \in [n_t]} l(\hat{z}_{u_t(j)}, A_t)^{1/2}} $$
즉, 특정 데이터 포인트의 **pseudo-loss의 제곱근에 비례하여** 샘플링 확률을 부여하는 것이 최적이다.

## 📊 Results

본 논문은 수치적인 벤치마크 실험 결과보다는 이론적인 분석과 증명에 집중한다.

* **정량적 분석 (Theorem 6.1)**: 알고리즘 1을 사용할 때, 확률 $1-\delta$ 이상으로 실제 손실이 다음과 같은 상한선을 가짐을 증명하였다.
    $$ \mathbb{E}_Z[l(x, y; A_t)] \leq \text{Generalization Error} + c_\delta \frac{M_t^p}{n_t} $$
    여기서 $M_t^p$는 가중 평균 pseudo-loss $\sum_{j \in [n_t]} \frac{l(\hat{z}_{u_t(j)}, A_t)}{p_{u_t(j)}^t}$이다.

* **최적성 증명 (Remark 6.5)**: Cauchy-Schwarz 부등식을 이용하여, 최적 분포 $p^*$를 사용했을 때의 가중 평균 pseudo-loss $M_t^{p^*}$가 균등 분포 $q$를 사용했을 때의 $M_t^q$보다 항상 작거나 같음을 보였다 ($M_t^{p^*} \leq M_t^q$). 이는 ALIS가 단순 무작위 샘플링보다 이론적으로 더 타이트한 손실 상한선을 제공함을 의미한다.

## 🧠 Insights & Discussion

본 논문은 Active Learning의 데이터 선택 문제를 통계적인 중요도 샘플링 문제로 치환하여 수학적인 최적해를 제시했다는 점에서 강점이 있다. 특히, 결정론적인 코어셋 방식에 비해 계산 복잡도가 매우 낮아 대규모 데이터셋에 적용하기 유리한 구조를 가지고 있다.

다만, 몇 가지 한계점과 가정이 존재한다.
첫째, 이론적 전개 과정에서 가설 공간(hypothesis space)이 충분히 표현력이 있어 가중 경험적 위험 최소화(Weighted ERM)를 통해 항 (B)를 0으로 만들 수 있다는 가정을 전제로 한다.
둘째, pseudo-label $\hat{y} = -\text{sign}(f(x))$를 사용하여 상한선을 구하는데, 이는 최악의 경우를 가정한 것이므로 실제 손실과의 괴리가 클 수 있다. 초기 모델의 성능이 매우 낮을 경우, 잘못된 pseudo-loss 계산으로 인해 초기 샘플링 효율이 떨어질 가능성이 있다.

## 📌 TL;DR

이 논문은 중요도 샘플링(Importance Sampling)을 도입하여 실제 손실의 수학적 상한선을 최소화하는 확률적 Active Learning 알고리즘 **ALIS**를 제안한다. 핵심은 데이터의 **pseudo-loss의 제곱근에 비례하여 샘플링 확률을 부여**하는 것이며, 이를 통해 균등 샘플링보다 더 낮은 이론적 손실 상한선을 달성한다. 이 연구는 계산 비용이 높은 기존의 결정론적 샘플링 방법론을 대체하여, 이론적 보장과 효율성을 동시에 갖춘 데이터 선택 프레임워크를 제공한다는 점에서 향후 연구 및 실제 적용 가치가 높다.
