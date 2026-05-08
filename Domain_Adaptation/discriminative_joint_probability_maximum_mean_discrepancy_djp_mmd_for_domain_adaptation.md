# Discriminative Joint Probability Maximum Mean Discrepancy (DJP-MMD) for Domain Adaptation

Wen Zhang, Dongrui Wu

## 🧩 Problem to Solve

기존 통계적 기계 학습에서는 훈련 데이터와 테스트 데이터가 동일한 분포에서 나온다고 가정하지만, 실제 환경에서는 이 가정이 깨지는 경우가 많습니다. 도메인 적응(Domain Adaptation, DA)은 레이블이 있는 소스 도메인의 지식을 레이블이 없거나 일부만 있는 타겟 도메인으로 전이하여 이러한 문제를 해결합니다.

MMD(Maximum Mean Discrepancy)는 도메인 간 분포 차이를 측정하는 데 널리 사용되지만, 기존 MMD 기반 DA 방법론(예: JDA, BDA)은 다음과 같은 한계를 가집니다:

1. **근사적인 결합 분포 측정**: 대부분의 방법은 주변 분포(marginal distribution)와 조건부 분포(conditional distribution)의 차이를 (가중치 적용하여) 합산하는 방식으로 결합 분포 차이를 측정합니다. 이는 결합 확률 분포를 직접 측정하는 것보다 덜 정확하고, 주변 분포와 조건부 분포 사이의 의존성을 무시합니다.
2. **식별성(Discriminability) 무시**: 대부분의 메트릭은 도메인 간 전이성(transferability)을 높이는 데만 초점을 맞추고, 다른 클래스 간의 식별성을 충분히 고려하지 않아 분류 성능이 저하될 수 있습니다.

이 논문은 이러한 문제를 해결하기 위해 보다 정확한 결합 확률 분포 차이를 직접 계산하고, 동시에 전이성 및 식별성을 향상시키는 새로운 MMD를 제안합니다.

## ✨ Key Contributions

- **새로운 이론적 기반 제시**: 결합 확률 분포 차이를 직접 고려함으로써 도메인 간 차이를 계산하기 위한 새롭고 더 정확하며 계산하기 쉬운 이론적 기반을 제공합니다.
- **DJP-MMD (Discriminative Joint Probability MMD) 제안**: 동일 클래스의 도메인 간 결합 확률 분포 차이를 최소화하여 전이성을 높이고, 다른 클래스의 도메인 간 결합 확률 분포 차이를 최대화하여 식별성을 동시에 향상시키는 새로운 MMD를 제안합니다.
- **성능 검증**: 광범위한 실험을 통해 제안된 DJP-MMD가 기존 MMD 기반 방법론보다 우수한 성능을 보임을 입증합니다.

## 📎 Related Works

- **JDA (Joint Distribution Adaptation) [10]**: 주변 MMD와 조건부 MMD를 결합하여 도메인 간 분포 차이를 측정하는 인기 있는 DA 방법론입니다. 하지만 조건부 분포 간의 관계나 주변-조건부 분포 간의 의존성을 무시합니다.
- **BDA (Balanced Distribution Adaptation) [16, 20]**: JDA의 한계를 개선하기 위해 주변 MMD와 조건부 MMD에 서로 다른 가중치를 부여하는 방법입니다. `A`-거리 [23]를 사용하여 가중치 $\mu$를 추정하며, 이는 JDA보다 성능 개선을 항상 보장하지는 않으며, $\mu$ 계산을 위해 $C+1$개의 분류기를 훈련해야 하므로 계산 비용이 높을 수 있습니다.

## 🛠️ Methodology

이 논문은 도메인 간 결합 확률 분포 $P(X, Y)$가 다르다는 가정하에, 새로운 특징 매핑 함수 $h$를 학습하여 $h(X_s)$와 $h(X_t)$를 가깝게 만들고, 이를 통해 소스 도메인에서 훈련된 분류기가 타겟 도메인에서도 잘 작동하도록 합니다.

1. **전통적인 MMD 메트릭 재고찰**:

   - 기존 MMD는 $P(Y|X)P(X)$ 또는 $P(X|Y)P(Y)$를 사용하여 결합 확률을 추정하며, 보통 $P(Y|X) + P(X)$와 같은 방식으로 주변 분포와 조건부 분포의 합으로 근사합니다 (예: (5)번 식). 이는 $P(Y|X)$와 $P(X)$ 사이의 의존성을 무시하고, $P(Y|X)$를 계산하기 어려운 문제 때문에 종종 $P(X|Y)$로 대체됩니다.
   - 선형 매핑 $h(x) = A^{\top}x$를 고려할 때, 기존 MMD는 다음과 같이 표현될 수 있습니다 (6)번 식:
     $$d(D_s,D_t) \approx \mu_1 \left\| \frac{1}{n_s} \sum_{i=1}^{n_s} A^{\top}x_{s,i} - \frac{1}{n_t} \sum_{j=1}^{n_t} A^{\top}x_{t,j} \right\|_2^2 + \mu_2 \sum_{c=1}^C \left\| \frac{1}{n_s^c} \sum_{i=1}^{n_s^c} A^{\top}x_{s,i}^c - \frac{1}{n_t^c} \sum_{j=1}^{n_t^c} A^{\top}x_{t,j}^c \right\|_2^2$$
     여기서 $n_s^c, n_t^c$는 각 클래스의 샘플 수입니다.

2. **DJP-MMD (Discriminative Joint Probability MMD) 제안**:

   - DJP-MMD는 베이즈 정리($P(X,Y) = P(X|Y)P(Y)$)에 따라 결합 확률 분포 차이를 직접 정의합니다 (7)번 식:
     $$d(D_s,D_t) = d(P(X_s|Y_s)P(Y_s), P(X_t|Y_t)P(Y_t)) = M_T + M_D$$
     여기서 $M_T$는 동일 클래스 간의 결합 확률 차이(전이성), $M_D$는 다른 클래스 간의 결합 확률 차이(식별성)를 측정합니다.
   - **전이성 (Transferability) MMD ($M_T$)**: 동일 클래스 간의 도메인 분포 차이를 최소화합니다.
     $$M_T = \sum_{c=1}^C \left\| \frac{1}{n_s} \sum_{i=1}^{n_s^c} A^{\top}x_{s,i}^c - \frac{1}{n_t} \sum_{j=1}^{n_t^c} A^{\top}x_{t,j}^c \right\|_2^2$$
     주목할 점은 기존 조건부 MMD가 각 클래스의 샘플 수($n_s^c, n_t^c$)로 나누는 것과 달리, DJP-MMD의 $M_T$는 전체 도메인 샘플 수($n_s, n_t$)로 나누어, 클래스 사전 확률 $P(Y)$를 효과적으로 통합합니다.
   - **식별성 (Discriminability) MMD ($M_D$)**: 다른 클래스 간의 도메인 분포 차이를 최대화하여 클래스 분리도를 높입니다.
     $$M_D = \sum_{c \neq \hat{c}}^C \sum_{\hat{c}=1}^C \left\| \frac{1}{n_s} \sum_{i=1}^{n_s^c} A^{\top}x_{s,i}^c - \frac{1}{n_t} \sum_{j=1}^{n_t^{\hat{c}}} A^{\top}x_{t,j}^{\hat{c}} \right\|_2^2$$
   - **DJP-MMD 목적 함수**: 전이성을 최소화하고 식별성을 최대화하는 형태로 정의됩니다 (8)번 식.
     $$d(D_s,D_t) = M_T - \mu M_D$$
     $\mu > 0$는 트레이드오프 파라미터입니다.

3. **JPDA (Joint Probability Domain Adaptation) 프레임워크에 DJP-MMD 적용**:

   - DJP-MMD를 주성분 보존 제약 및 정규화 항이 포함된 목적 함수에 삽입합니다.
     $$\min_A \left\| A^{\top}X_s N_s - A^{\top}X_t N_t \right\|_F^2 - \mu \left\| A^{\top}X_s M_s - A^{\top}X_t M_t \right\|_F^2 + \lambda \|A\|_F^2 \quad \text{s.t.} \quad A^{\top}XHX^{\top}A=I$$
     여기서 $N_s = Y_s/n_s$, $N_t = \hat{Y}_t/n_t$이며, $M_s = F_s/n_s$, $M_t = \hat{F}_t/n_t$입니다. $\hat{Y}_t$와 $\hat{F}_t$는 반복적으로 업데이트되는 타겟 도메인의 의사 레이블(pseudo-label)을 기반으로 합니다.
   - 이 문제는 일반화된 고유값 분해(generalized eigen-decomposition) 문제로 변환하여 효율적으로 해결할 수 있습니다.
   - 비선형 DA를 위해 커널 함수를 사용하여 확장할 수 있습니다.

4. **계산 복잡도**: 반복 횟수 $T$, 부분 공간 차원 $p$, 데이터 차원 $d$, 샘플 수 $n$에 대해 총 계산 복잡도는 $O(Tpd^2 + Tn^2 + Tdn)$입니다.

## 📊 Results

- **성능 우수성**: Office+Caltech, COIL, Multi-PIE, MNIST, USPS 등 6개 벤치마크 데이터셋에 대한 실험에서 JPDA (DJP-MMD를 사용한 프레임워크)는 TCA, JDA, BDA와 같은 전통적인 DA 방법론보다 대부분의 태스크에서 우수한 분류 정확도를 보였으며, 평균 성능이 가장 높았습니다. 이는 JPDA가 더 전이 가능하고 식별력 있는 특징 매핑을 학습함을 시사합니다.
- **시각화 (t-SNE)**: t-SNE를 이용한 데이터 분포 시각화 결과, JPDA는 소스 및 타겟 도메인의 동일 클래스 샘플들을 가깝게 모으는 동시에, 다른 클래스 샘플들을 명확히 분리하여 더 나은 식별성을 보여주었습니다. JDA와 BDA는 JPDA만큼 좋은 식별성을 보여주지 못했습니다.
- **수렴 및 시간 복잡도**: JPDA는 기존 MMD 기반 방법론보다 빠르게 수렴하며, 더 작은 MMD 거리와 높은 정확도를 달성했습니다. 계산 비용 측면에서는 BDA보다 훨씬 빠르며, 특히 대규모 데이터셋(Multi-PIE)에서는 50% 이상의 시간을 절약했습니다. TCA는 비반복적이라 가장 빨랐습니다.
- **파라미터 민감도**: 트레이드오프 파라미터 $\mu$ (0.001에서 0.2 사이) 및 정규화 파라미터 $\lambda$ (0.01에서 10 사이)에 대해 JPDA는 광범위한 값 범위에서 안정적이고 만족스러운 성능을 보였습니다.
- **Ablation Study**: 식별성 MMD ($M_D$)의 기여를 확인하기 위한 ablation study 결과, $M_D$를 고려하지 않은 JP-MMD는 Joint MMD보다 우수했지만, $M_D$를 추가로 고려한 DJP-MMD가 가장 좋은 분류 성능을 달성하여 $M_D$의 유효성을 입증했습니다.

## 🧠 Insights & Discussion

- **이론적 우수성**: DJP-MMD는 기존 MMD가 주변 분포와 조건부 분포를 합산하여 결합 분포를 근사하는 방식과 달리, 베이즈 정리를 기반으로 결합 확률 분포 차이 $P(X|Y)P(Y)$를 직접 측정함으로써 이론적으로 더 정확하고 간결한 분포 차이 측정 방식을 제공합니다. 이는 실제 데이터로부터 직접 계산될 수 있어 근사 없이 더 정확합니다.
- **동시 최적화**: 이 방법은 동일 클래스의 도메인 간 전이성을 최대화하는 동시에, 다른 클래스 간의 식별성을 최대화하여 분류 성능을 효과적으로 향상시킵니다. 기존 방법들이 전이성에만 집중했던 한계를 극복합니다.
- **실용적 이점**: 제안된 JPDA 프레임워크는 기존 MMD 기반 접근 방식보다 더 간단하고 효율적입니다. 특히, BDA가 $\mu$를 계산하기 위해 $C+1$개의 분류기를 훈련해야 하는 것에 비해, JPDA는 계산 비용 측면에서도 이점을 가집니다.
- **향후 연구**: DJP-MMD의 개념을 딥러닝 및 적대적 학습(adversarial learning) 분야로 확장하는 것이 향후 연구 방향으로 제시되었습니다.

## 📌 TL;DR

이 논문은 도메인 적응(DA)에서 도메인 간 분포 차이 측정의 정확도를 높이고 클래스 식별성을 향상시키기 위해 **DJP-MMD (Discriminative Joint Probability Maximum Mean Discrepancy)**를 제안합니다. 기존 MMD 방법론이 주변 및 조건부 분포의 합으로 결합 분포를 근사하고 식별성을 무시하는 한계를 넘어, DJP-MMD는 결합 확률 분포 차이 $P(X|Y)P(Y)$를 직접 계산하며, 동일 클래스 간의 전이성을 최소화하고 다른 클래스 간의 식별성을 최대화합니다. JPDA(Joint Probability Domain Adaptation) 프레임워크에 DJP-MMD를 적용한 결과, 6개 이미지 분류 데이터셋에서 기존 MMD 기반 방법론 대비 우수한 분류 성능을 보였으며, 시각화 및 ablation study를 통해 전이성과 식별성 동시 개선 효과를 입증했습니다.
