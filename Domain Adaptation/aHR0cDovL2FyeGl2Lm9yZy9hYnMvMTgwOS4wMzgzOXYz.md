# Unsupervised Domain Adaptation Based on Source-guided Discrepancy

Seiichi Kuroki, Nontawat Charoenphakdee, Han Bao, Junya Honda, Issei Sato, Masashi Sugiyama (2018)

## 🧩 Problem to Solve

본 논문은 Unsupervised Domain Adaptation (UDA) 상황에서 소스 도메인(Source Domain)과 타겟 도메인(Target Domain) 사이의 차이를 어떻게 정확하고 효율적으로 측정할 것인가에 대한 문제를 다룬다. UDA는 소스 도메인에는 레이블이 풍부하지만, 타겟 도메인에는 레이블이 전혀 없는 상황에서 타겟 도메인에 적합한 분류기를 찾는 것이 목표이다.

도메인 간의 차이, 즉 Discrepancy를 측정하는 것은 매우 중요하다. 만약 두 도메인이 너무 다르면 소스 데이터를 활용한 학습이 오히려 타겟 도메인의 성능을 저하시키는 Negative Transfer가 발생할 수 있기 때문이다. 기존의 Discrepancy 측정 방식들은 다음과 같은 한계를 가지고 있다:
- **High Computation Cost**: $\text{X-disc}$와 같이 이론적 보장은 있으나 계산 복잡도가 매우 높아 실용성이 떨어진다.
- **Lack of Theoretical Guarantee**: $d_H$와 같이 계산은 효율적이지만 일반화 오차(Generalization Error)에 대한 이론적 보장이 없다.
- **Requirement of Target Labels**: $\text{Y-disc}$는 이론적으로 더 타이트한 bound를 제공하지만, UDA 설정에서 불가능한 '타겟 도메인의 레이블'을 필요로 한다.

따라서 본 논문의 목표는 계산 효율성과 이론적 보장을 모두 갖추며, 타겟 레이블 없이도 계산 가능한 새로운 Discrepancy 측정 지표인 Source-guided Discrepancy (S-disc)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **소스 도메인의 레이블 정보를 활용하여 Discrepancy를 측정**함으로써, 기존의 $\text{X-disc}$가 가졌던 계산적 부담을 줄이고 더 정교한 일반화 오차 bound를 제공하는 것이다. 구체적인 기여 사항은 다음과 같다:

1. **S-disc 제안**: 소스 도메인의 True Risk Minimizer인 $h^*_S$를 기준으로 도메인 간의 차이를 측정하는 새로운 지표 $\varsigma^\ell_H$를 정의하였다.
2. **효율적인 추정 알고리즘**: 0-1 loss에 대해 S-disc 추정 문제를 Cost-sensitive classification 문제로 환원하여 효율적으로 계산하는 알고리즘을 제시하였다.
3. **이론적 분석**: 제안한 S-disc 추정량의 일관성(Consistency)과 수렴 속도(Convergence rate)를 입증하였으며, $\text{X-disc}$보다 더 타이트한 타겟 도메인 일반화 오차 bound를 유도하였다.
4. **실험적 검증**: Toy experiment 및 MNIST-M 데이터셋을 통해 S-disc가 기존 지표($d_H, \text{X-disc}$)보다 우수한 도메인 선택 성능과 계산 효율성을 가짐을 보였다.

## 📎 Related Works

논문에서는 기존의 주요 Discrepancy 측정 방식들을 다음과 같이 분석한다.

- **$\text{X-disc}$**: 가설 클래스 $H$ 내의 임의의 두 가설 쌍 $(h, h')$에 대해 소스와 타겟 간의 기대 손실 차이의 최댓값을 측정한다.
  $$\text{disc}^\ell_H(P_T, P_S) = \sup_{h, h' \in H} | R^\ell_T(h, h') - R^\ell_S(h, h') |$$
  이 방식은 이론적 보장이 있지만, 최악의 가설 쌍을 찾아야 하므로 계산 비용이 매우 높고 bound가 느슨(loose)해지는 경향이 있다.
- **$d_H$ (Ben-David et al., 2007)**: $\text{X-disc}$의 효율적인 근사치로 제안되었으며, 소스와 타겟을 구분하는 분류기의 성능을 측정한다. 계산은 빠르나 일반화 오차에 대한 학습 보장이 없다.
- **$\text{Y-disc}$**: 타겟 도메인의 레이블링 함수 $f_T$를 직접 사용한다.
  $$\text{Y-disc}^\ell_H(P_T, P_S) = \sup_{h \in H} | R^\ell_T(h, f_T) - R^\ell_S(h, f_S) |$$
  가장 타이트한 bound를 제공하지만, 타겟 레이블이 없는 UDA에서는 사용할 수 없다.

## 🛠️ Methodology

### Source-guided Discrepancy (S-disc) 정의
S-disc는 $\text{X-disc}$의 정의에서 $h'$를 소스 도메인의 True Risk Minimizer인 $h^*_S$로 고정하여 정의한다.

**Definition 1 (S-disc):**
$$\varsigma^\ell_H(P_{D_1}, P_{D_2}) = \sup_{h \in H} | R^\ell_{D_1}(h, h^*_S) - R^\ell_{D_2}(h, h^*_S) |$$
여기서 $h^*_S = \arg \min_{h \in H} R^\ell_S(h, f_S)$이다.

이 정의를 통해 S-disc는 $\text{X-disc}$보다 계산 비용이 낮으면서도, $\text{X-disc}$보다 항상 작거나 같으므로(즉, $\varsigma^\ell_H \le \text{disc}^\ell_H$) 더 타이트한 일반화 오차 bound를 제공할 수 있다.

### 0-1 Loss에 대한 S-disc 추정 알고리즘
이론적인 S-disc는 $h^*_S$를 알아야 하지만, 실제로는 소스 데이터로부터 학습된 $\hat{h}_S$를 사용한다. 특히 0-1 loss의 경우, S-disc 추정 문제는 다음과 같은 Cost-sensitive classification 문제로 변환된다.

**Theorem 2:** 대칭적 가설 클래스 $H$에 대해 다음이 성립한다.
$$\varsigma^{\ell_{01}}_H(\hat{P}_T, \hat{P}_S) = 1 - \min_{h \in H} J^{\ell_{01}}(h)$$
여기서 $J^\ell(h)$는 다음과 같이 정의된다:
$$J^\ell(h) = \frac{1}{n_S} \sum_{j=1}^{n_S} \ell(h(x^S_j), h^*_S(x^S_j)) + \frac{1}{n_T} \sum_{i=1}^{n_T} \ell(h(x^T_i), -h^*_S(x^T_i))$$

**알고리즘 흐름 (Algorithm 1):**
1. **Source Learning**: 레이블이 있는 소스 데이터 $S$를 사용하여 분류기 $\hat{h}_S$를 학습한다.
2. **Pseudo Labeling**: 
   - 소스 데이터에 대해 $\tilde{S} = \{(x, \text{sign}(\hat{h}_S(x))) | x \in S_X\}$ 생성.
   - 타겟 데이터에 대해 $\tilde{T} = \{(x, -\text{sign}(\hat{h}_S(x))) | x \in T\}$ 생성 (부호를 반전시켜 가짜 레이블 부여).
3. **Cost Sensitive Learning**: $\tilde{S}$와 $\tilde{T}$를 사용하여 surrogate loss(예: Hinge loss)를 최소화하는 분류기 $h''$를 학습한다.
4. **결과 도출**: $\varsigma^{\ell_{01}}_H(\hat{P}_T, \hat{P}_S) = 1 - J^{\ell_{01}}(h'')$를 반환한다.

### 일반화 오차 Bound (Generalization Error Bound)
본 논문은 S-disc를 이용한 타겟 도메인의 일반화 오차 bound를 다음과 같이 유도하였다 (Theorem 7).

$$R^\ell_T(h, f_T) - R^\ell_T(h^*_T, f_T) \le R^\ell_S(h, h^*_S) + R^\ell_T(h^*_S, h^*_T) + \varsigma^\ell_H(P_T, P_S)$$

이 식은 타겟 도메인에서의 후회(Regret)가 (i) 소스 도메인에서의 오차, (ii) 소스와 타겟의 최적 가설 간의 차이, (iii) S-disc에 의해 결정됨을 보여준다. $\text{X-disc}$ 기반의 bound와 비교했을 때, $\varsigma^\ell_H \le \text{disc}^\ell_H$이므로 더 정확한(타이트한) 예측이 가능하다.

## 📊 Results

### 실험 설정
- **Toy Experiment**: 2D 가우시안 분포를 통해 두 개의 소스($S_1, S_2$)와 하나의 타겟($T$)을 생성하여 어떤 소스가 더 적합한지 측정하였다.
- **Computation Time**: $n_S=100, n_T=100$인 2차원 데이터에서 S-disc, $d_H, \text{X-disc}$의 계산 시간을 측정하였다.
- **Empirical Convergence**: MNIST 데이터셋을 사용하여 샘플 수 증가에 따른 Discrepancy 추정치의 수렴 속도를 측정하였다.
- **Source Selection**: MNIST-M(깨끗한 소스 5개, 노이즈 섞인 소스 5개)과 MNIST(타겟)를 사용하여 깨끗한 소스를 상위 5개로 올바르게 랭킹하는지 측정하였다.

### 주요 결과
1. **정성적 분석 (Toy Exp)**: $d_H$는 $S_2$가 더 좋은 소스라고 판단했으나, 실제 타겟 성능은 $S_1$에서 학습한 모델이 훨씬 좋았다. 반면 S-disc는 $S_1$을 더 적합한 소스로 올바르게 선택하였다.
2. **계산 복잡도**: S-disc와 $d_H$는 $O((n_T + n_S)^3)$의 복잡도를 가져 실용적이었으나, $\text{X-disc}$는 SDP relaxation을 사용하더라도 $O((n_T + n_S + d)^8)$이라는 막대한 비용이 발생하여 사실상 계산이 불가능함을 보였다.
3. **수렴 속도**: MNIST 실험에서 S-disc는 샘플 수가 증가함에 따라 매우 빠르게 0으로 수렴하였으나, $d_H$는 수렴 속도가 훨씬 느리거나 0이 아닌 값으로 수렴하는 양상을 보였다.
4. **소스 선택 성능**: 노이즈 수준($\epsilon=30, 40, 50$)이 높아질수록 S-disc는 깨끗한 소스를 매우 정확하게 식별해낸 반면, $d_H$는 노이즈 유무를 전혀 구분하지 못하고 낮은 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
S-disc의 핵심적인 강점은 **'불필요한 정보의 제거'**에 있다. 기존의 $d_H$나 $\text{X-disc}$는 단순히 입력 분포 $P(X)$의 차이를 측정하려 한다. 하지만 입력 분포가 다르더라도, 두 도메인에서 최적의 분류 경계($h^*$)가 비슷하다면 실제 태스크 성능에는 영향이 없다. S-disc는 소스 도메인의 레이블을 통해 $h^*_S$라는 가이드를 제공함으로써, 태스크 관점에서 유의미한 차이만을 측정한다.

### 한계 및 논의사항
- **가설 클래스 의존성**: S-disc는 특정 가설 클래스 $H$에 의존한다. 만약 선택한 $H$가 너무 단순하거나 부적절하다면, 측정된 Discrepancy가 실제 도메인 간의 차이를 충분히 반영하지 못할 가능성이 있다.
- **Symmetric Hypothesis Class 가정**: 0-1 loss 추정을 위해 사용한 Theorem 2는 $H$가 negation에 대해 닫혀 있다는 대칭성 가정을 필요로 한다. 실제 딥러닝 모델과 같은 복잡한 구조에서 이 가정이 어떻게 유지될 수 있는지에 대한 추가 논의가 필요하다.

## 📌 TL;DR

본 논문은 Unsupervised Domain Adaptation에서 소스 도메인의 레이블을 활용해 도메인 간 차이를 측정하는 **Source-guided Discrepancy (S-disc)**를 제안하였다. S-disc는 $\text{X-disc}$의 높은 계산 비용 문제를 해결하면서도, $d_H$보다 정교한 이론적 보장(tighter generalization bound)을 제공한다. 실험을 통해 S-disc가 특히 **소스 도메인 선택(Source Selection)** 작업에서 기존 지표들보다 압도적인 성능과 효율성을 보임을 입증하였다. 이 연구는 다중 소스 도메인 적응(Multi-source DA)이나 데이터 리웨이팅(Reweightng) 분야에서 최적의 소스를 선택하는 기준으로 활용될 가능성이 높다.