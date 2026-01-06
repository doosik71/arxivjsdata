# Multi-Adversarial Domain Adaptation

Zhongyi Pei, Zhangjie Cao, Mingsheng Long, and Jianmin Wang

## 🧩 Problem to Solve

기존의 딥 도메인 적응(Deep Domain Adaptation) 방법들은 소스 도메인과 타겟 도메인 간의 분포 불일치(distribution discrepancy)를 줄이기 위해 적대적 학습(adversarial learning)을 사용했습니다. 그러나 단일 도메인 판별자(discriminator)를 기반으로 하는 이 방법들은 데이터 분포의 복잡한 다중 모드(multimode) 구조를 활용하지 않고 전체 분포만을 정렬합니다. 이로 인해 소스 및 타겟 데이터의 모든 모드가 혼합되거나, 해당 식별 구조가 잘못 정렬되어 (예: 소스 `cat` 클래스가 타겟 `dog` 클래스와 잘못 정렬됨) **부정적 전이(negative transfer)**가 발생할 위험이 있었습니다. 주요 과제는 다음과 같습니다:

1. **긍정적 전이(positive transfer) 강화**: 도메인 간 데이터 분포에 내재된 다중 모드 구조를 최대한 일치시킵니다.
2. **부정적 전이 완화**: 도메인 간 다른 분포 모드의 잘못된 정렬을 방지합니다.

## ✨ Key Contributions

- **다중 적대적 도메인 적응(Multi-Adversarial Domain Adaptation, MADA) 접근 방식 제안**: 데이터의 다중 모드 구조를 포착하고 여러 도메인 판별자를 기반으로 미세 조정된 분포 정렬을 가능하게 합니다.
- **클래스별 도메인 판별자 활용**: 단일 판별자 대신 $K$개의 클래스별 도메인 판별자를 사용하여 각 클래스에 해당하는 소스 및 타겟 데이터를 개별적으로 정렬합니다.
- **예측된 레이블 확률을 통한 가중치 부여**: 타겟 데이터의 레이블이 없으므로, 레이블 예측자($G_y$)의 출력 확률 $\hat{y}_i^k$를 사용하여 각 데이터 포인트 $x_i$가 각 $k$-번째 도메인 판별자 $G_d^k$에 얼마나 기여해야 하는지를 가중치로 부여합니다. 이는 관련 없는 클래스에 대한 정렬을 방지하여 부정적 전이를 효과적으로 완화하고 긍정적 전이를 촉진합니다.
- **선형 시간 백프로파게이션 기반 SGD 최적화**: 제안된 모델은 백프로파게이션을 통해 계산된 기울기를 사용하여 확률적 경사 하강법(Stochastic Gradient Descent, SGD)으로 효율적으로 최적화됩니다.
- **최첨단 성능 달성**: 표준 도메인 적응 데이터셋에서 최첨단(state-of-the-art) 방법들보다 우수한 성능을 보여줍니다.

## 📎 Related Works

- **전이 학습 (Transfer Learning)**: 도메인 불일치 완화를 위한 방법론으로, 딥러닝과 결합하여 추상적인 표현 학습 (e.g., Donahue et al. 2014; Yosinski et al. 2014).
- **딥 도메인 적응 (Deep Domain Adaptation)**: 딥 네트워크에 적응 계층(adaptation layers)을 추가하여 분포 일치 (e.g., DDC by Tzeng et al. 2014; DAN by Long et al. 2015; RTN by Long et al. 2016) 또는 도메인 적대적 훈련(domain-adversarial training)을 통한 도메인 불변 표현 학습 (e.g., RevGrad by Ganin and Lempitsky 2015; Tzeng et al. 2015).
- **적대적 학습 (Adversarial Learning)**: GAN(Generative Adversarial Networks) (Goodfellow et al. 2014)과 같이 생성 모델링에 활용되며, GMAN(Generative Multi-Adversarial Network) (Durugkar et al. 2017)은 여러 판별자를 사용하여 훈련을 용이하게 하고 분포 매칭을 강화합니다.

## 🛠️ Methodology

MADA는 특징 추출기($G_f$), 레이블 예측기($G_y$), 그리고 $K$개의 클래스별 도메인 판별자($G_d^k$)로 구성됩니다.

1. **기존 도메인 적대적 네트워크(DANN) 목표 함수**:
   $$ C*0(\theta_f, \theta_y, \theta_d) = \frac{1}{n_s} \sum*{x*i \in D_s} L_y(G_y(G_f(x_i)), y_i) - \frac{\lambda}{n} \sum*{x_i \in (D_s \cup D_t)} L_d(G_d(G_f(x_i)), d_i) $$
    여기서 $\theta_f, \theta_y$는 $\min$을 목표로 하고, $\theta_d$는 $\max$를 목표로 합니다. $L_y$는 레이블 예측 손실, $L_d$는 도메인 판별 손실입니다.

2. **MADA의 다중 판별자 도입**:
   단일 도메인 판별자 $G_d$를 $K$개의 클래스별 도메인 판별자 $G_d^k$ ($k=1, \dots, K$)로 대체합니다. 각 $G_d^k$는 클래스 $k$와 관련된 소스 및 타겟 도메인 데이터를 매칭하는 역할을 합니다.

3. **예측된 레이블 확률을 이용한 가중치 부여**:
   타겟 데이터에는 레이블이 없으므로, 레이블 예측기 $G_y(x_i)$의 출력인 클래스별 확률 분포 $\hat{y}_i = G_y(G_f(x_i))$를 활용합니다. 각 데이터 포인트 $x_i$에 대해 $k$-번째 클래스에 할당될 확률 $\hat{y}_i^k$를 해당 $k$-번째 도메인 판별자의 입력 특징 $G_f(x_i)$에 대한 가중치로 사용합니다.

4. **MADA의 목표 함수**:
   $$ C(\theta*f, \theta_y, \{\theta_d^k\}*{k=1}^K) = \frac{1}{n*s} \sum*{x*i \in D_s} L_y(G_y(G_f(x_i)), y_i) - \frac{\lambda}{n} \sum*{k=1}^K \sum\_{x_i \in D} L_d^k(G_d^k(\hat{y}\_i^k G_f(x_i)), d_i) $$

   - $\theta_f$, $\theta_y$는 이 목표 함수를 **최소화**하고, $\theta_d^k$ ($k=1, \dots, K$)는 이 목표 함수를 **최대화**하는 방향으로 학습됩니다.
   - 이러한 최적화는 특징 추출기 및 레이블 예측기가 도메인 판별자들을 혼동시키고 (도메인 불변 특징 학습), 도메인 판별자들이 소스와 타겟을 구분하도록 (세분화된 도메인 경계 학습) 하는 적대적 과정입니다.

5. **훈련 과정**:
   - **경사 역전 계층(Gradient Reversal Layer, GRL)**을 사용하여 $\lambda$ 항의 그래디언트 부호를 뒤집어 적대적 학습을 구현합니다.
   - 학습률과 $\lambda$ 값은 훈련 진행도($p$)에 따라 점진적으로 조정됩니다.

## 📊 Results

- **Office-31 및 ImageCLEF-DA 데이터셋**: MADA는 AlexNet 및 ResNet 기반에서 대부분의 전이 태스크에서 모든 비교 방법(TCA, GFK, DDC, DAN, RTN, RevGrad)보다 우수한 성능을 달성했습니다.
- **어려운 전이 태스크에서 특히 강점**: 소스 도메인과 타겟 도메인이 크게 다른 `A→W`, `A→D`, `D→A`, `W→A`와 같은 어려운 태스크에서 분류 정확도를 크게 향상시켰습니다.
- **부정적 전이 방지 입증**: 소스 도메인 클래스 수가 타겟 도메인보다 많은 시나리오(예: Office-31에서 31개 클래스에서 25개 클래스로 전이)에서 MADA는 표준 AlexNet과 RevGrad보다 훨씬 뛰어난 성능을 보이며 부정적 전이 문제를 성공적으로 회피했습니다.
- **특징 시각화 (t-SNE)**: RevGrad가 도메인 불변성을 달성했지만 클래스 분리 능력이 떨어진 반면, MADA는 소스 및 타겟 도메인을 더욱 구별하기 어렵게 만들면서도 다른 카테고리들을 더 명확하게 분리하는 특징을 학습했음을 보여주었습니다.

## 🧠 Insights & Discussion

- **다중 모드 구조 활용의 중요성**: MADA의 성능 향상은 기존 단일 판별자 방식이 간과했던 복잡한 다중 모드 구조를 활용하여 미세 조정된 도메인 정렬을 수행한 결과입니다.
- **부정적 전이 완화**: 예측된 레이블 확률을 이용한 클래스별 판별자 가중치 부여는 관련 없는 클래스 간의 잘못된 정렬을 방지하여 부정적 전이의 주요 기술적 병목 현상을 해결하는 데 핵심적인 역할을 합니다.
- **판별자 파라미터 공유 전략**: 다중 판별자 네트워크의 파라미터를 많이 공유할수록 전이 성능이 감소하는 실험 결과는 각 클래스에 대해 독립적인 판별자가 필요하다는 MADA의 설계 동기를 뒷받침합니다.
- **A-거리(A-distance) 감소**: MADA 특징은 ResNet 및 RevGrad 특징보다 현저히 작은 A-거리를 보여, 도메인 간 격차를 더욱 효과적으로 줄임을 시사합니다.
- **수렴 안정성**: MADA는 RevGrad와 유사하게 안정적인 수렴 성능을 보이며, 계산 복잡성도 크게 증가시키지 않으면서 전반적인 성능을 능가했습니다.

## 📌 TL;DR

MADA(Multi-Adversarial Domain Adaptation)는 단일 판별자의 한계를 넘어, 데이터 분포의 복잡한 다중 모드 구조를 활용하기 위해 **다수의 클래스별 도메인 판별자**를 제안합니다. 각 판별자는 레이블 예측기의 출력 확률에 따라 가중치를 부여받아 특정 클래스에 해당하는 소스 및 타겟 데이터를 **미세 조정하여 정렬**합니다. 이 접근 방식은 **긍정적 전이를 강화하고 부정적 전이를 효과적으로 완화**하며, 표준 벤치마크에서 기존 최첨단 방법들을 뛰어넘는 뛰어난 도메인 적응 성능을 달성합니다.
