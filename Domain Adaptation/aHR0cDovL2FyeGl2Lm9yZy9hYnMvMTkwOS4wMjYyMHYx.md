# Multi-layer Domain Adaptation for Deep Convolutional Networks

Ozan Ciga, Jianan Chen and Anne Martel (2019)

## 🧩 Problem to Solve

본 논문은 Deep Convolutional Networks(DCN)가 일반화 성능을 확보하기 위해 방대한 양의 레이블링된 데이터(labeled data)를 필요로 한다는 점과, 학습 시 노출되지 않은 새로운 도메인의 데이터(unseen domain)에 대해 성능이 보장되지 않는다는 문제를 해결하고자 한다. 이러한 특성은 특히 의료 영상 분석 분야에서 심각한 제약이 된다. 의료 데이터는 전문 인력에 의한 어노테이션 과정이 매우 까다롭고 비용이 많이 들어 데이터 확보가 어렵고, 동일한 소스에서 생성된 데이터라 하더라도 기관이나 장비에 따라 상당한 도메인 간 분산(intra- and inter-domain variance)이 존재하기 때문이다. 따라서 본 연구의 목표는 레이블링된 데이터에 대한 의존도를 낮추면서, 딥러닝 네트워크가 다양한 도메인에 걸쳐 강건하게 작동할 수 있도록 하는 도메인 적응(Domain Adaptation, DA) 기법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 도메인 적응 방식이 주로 네트워크의 마지막 특징 추출 계층(final feature layer)에서만 수행되었던 것과 달리, 네트워크의 여러 중간 계층(intermediate layers)에서 동시에 도메인 적응을 수행하는 **Multi-layer Domain Adaptation**을 제안한 것이다.

이를 위해 저자들은 Squeeze-and-Excite (SE) 모듈의 구조를 재설계하여, 각 합성곱 블록의 끝에서 'Squeeze' 연산을 통해 요약 통계량을 추출하고, 이를 바탕으로 도메인 분류(Domain Classification)를 수행함으로써 각 계층에서 도메인 독립적인 특징(domain-independent features)을 학습하도록 유도한다.

## 📎 Related Works

기존의 도메인 적응 연구들은 소스 도메인(source domain)과 타겟 도메인(target domain) 간의 공변량 변화(covariate shift)를 수정하기 위해 샘플에 가중치를 부여하거나, 특징 매핑(feature mappings) 간의 거리를 최소화하는 방식을 취해왔다. 대표적으로 Ganin 등이 제안한 DANN(Domain-Adversarial Neural Network)은 Gradient Reversal Layer(GRL)를 사용하여 특징 추출기가 도메인 분류기를 속이도록 학습함으로써 도메인 불변 특징을 추출한다.

그러나 저자들은 기존 방식이 딥러닝 네트워크의 깊이가 깊어질 때 다음과 같은 한계가 있다고 지적한다. 첫째, 네트워크가 깊어질수록 도메인 분류기에서 오는 오차 신호가 소멸(vanishing gradient)되어 초기 계층의 도메인 특성 제거가 어렵다. 둘째, 초기 계층에서 이미 도메인 의존적인 특징이 추출되었다면 이후 계층에서 이를 독립적인 특징으로 변환하는 것이 더 어려워진다. 셋째, 도메인 특성이 제거되지 않은 채 전파되면 네트워크의 용량(capacity)이 낭비된다.

## 🛠️ Methodology

### 전체 파이프라인

제안된 방법론은 기존의 FCN(Fully Convolutional Network) 구조에 각 계층별로 도메인 분류기를 부착하여, 특징 추출 과정 전반에 걸쳐 도메인 적응을 수행한다. 이 방식은 크게 두 가지 접근법인 **L-DANN**과 **L-WASS**로 구현된다.

### Squeeze Operation

각 계층의 특징 맵 $X \in \mathbb{R}^{H' \times W' \times C'}$이 주어졌을 때, 이를 벡터 $z \in \mathbb{R}^{C'}$로 변환하기 위해 Global Average Pooling(GAP)을 수행한다. 이 연산은 다음과 같이 정의된다.
$$z_k = \frac{1}{H' \times W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} u_k(i,j)$$
여기서 $u_k(i,j)$는 $X$의 $k$번째 커널에 대한 $(i,j)$ 위치의 응답 값이다. 이 'Squeeze' 연산을 통해 얻은 $z$는 해당 계층의 요약 통계량을 담고 있으며, 도메인 분류기의 입력으로 사용된다.

### L-DANN (Layer-wise Domain-Adversarial Neural Network)

L-DANN은 DANN의 구조를 각 계층으로 확장한 것이다.

- **구조**: 각 특징 맵 $X_i$의 끝에 소형 도메인 분류기 $D_i$를 부착한다. 분류기의 복잡도는 네트워크의 깊이에 비례하여 점진적으로 증가시킨다.
- **학습 목표**: 도메인 분류기는 $N$개 도메인에 대한 Cross-Entropy Loss($L_d$)를 최소화하려 하지만, GRL(Gradient Reversal Layer)을 통해 특징 추출기 $\theta_f$로 전달되는 그래디언트는 부호를 반전시켜 $L_d$를 최대화하도록 한다.
- **효과**: 이를 통해 각 계층의 특징 추출기가 도메인 식별 정보를 제거하고 도메인 불변 특징을 학습하게 된다.

### L-WASS (Layer-wise Wasserstein Domain Adaptation)

L-WASS는 레이블 기반의 분류 대신 두 도메인 간의 분포 거리를 직접 최소화하는 방식이다.

- **목표 함수**: 소스 도메인 $z_s$와 타겟 도메인 $z_t$ 사이의 Wasserstein-1 거리(Earth Mover's Distance)를 최소화한다.
- **학습 절차**: 'Critic'이라 불리는 네트워크를 사용하여 두 분포의 차이를 측정하며, 학습의 안정성을 위해 Gradient Penalty를 적용하여 Lipschitz 연속성 제약 조건을 충족시킨다.
- **수식적 접근**: Critic $D_j$에 대해 다음과 같은 손실 함수를 최적화한다.
$$L^{(i)} = D_j(z_s^j) - D_j(z_t^j) - \lambda(||\nabla_{\hat{z}_j} D_j(\hat{z}_j)||^2 - 1)^2$$
여기서 $\hat{z}_j$는 $z_s^j$와 $z_t^j$ 사이의 무작위 보간(interpolation) 샘플이다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, MNIST-M, SVHN (단순 네트워크 테스트), 흉부 X-ray (심층 네트워크 테스트), BACH (유방암 조직 병리 영상)
- **네트워크**: SE-ResNet-101, SENet 154, SE-ResNet-50
- **지표**: Accuracy, Precision, Recall, F1-score 및 BACH 챌린지 전용 스코어

### 주요 결과

1. **소규모 네트워크 (MNIST 등)**:
   - L-DANN은 기존 DANN과 유사한 성능을 보였다.
   - 반면 L-WASS는 수렴에 실패하거나 낮은 성능을 보여, 단순한 분포에서는 Layer-wise Wasserstein 매칭이 부적합할 수 있음을 시사한다.

2. **심층 네트워크 (Chest X-ray)**:
   - L-DANN과 L-WASS 모두 'No adaptation' 베이스라인보다 일관되게 높은 성능을 보였다.
   - 특히 SENet 154와 같은 매우 깊은 네트워크에서 기존 DANN은 수렴에 실패하거나 불안정한 모습을 보였으나, 제안된 L-DANN/L-WASS는 안정적으로 성능을 향상시켰다.
   - 소량의 데이터($S$)에서 대량의 데이터($L$)로 전이하는 $S \rightarrow L$ 설정에서 L-DANN은 정확도를 최대 약 7% 향상시켰다.

3. **특징 정규화 효과 (BACH)**:
   - 유방암 조직 병리 영상의 세그멘테이션 작업에서 L-DANN은 0.68의 스코어를 기록하여, No adaptation(0.63) 및 DANN(0.65)보다 우수한 성능을 보였다. 이는 공개 리더보드의 2위 성적(0.63)을 상회하는 결과이다.

## 🧠 Insights & Discussion

본 논문은 도메인 적응을 네트워크의 최종 단계가 아닌 전 과정(Layer-wise)에서 수행함으로써 딥러닝 모델의 일반화 능력을 크게 향상시킬 수 있음을 입증하였다. 특히 모델의 깊이가 깊어질수록 기존 DANN 방식이 겪는 그래디언트 소멸 문제를 효과적으로 해결하였다는 점이 강점이다.

분석 결과, 도메인 분류기의 복잡도를 계층 깊이에 따라 다르게 설정한 점이 유효하게 작용하였는데, 이는 초기 계층이 텍스처나 엣지와 같은 저수준(low-level) 정보를 추출하고 후기 계층이 고수준(high-level) 세만틱 정보를 추출하는 CNN의 특성을 반영했기 때문으로 해석된다.

다만, L-WASS가 단순한 데이터셋에서 수렴하지 못한 점은 Wasserstein 거리 기반의 적응 방식이 요구하는 데이터의 복잡도나 하이퍼파라미터 민감도가 존재함을 시사한다. 또한, 본 연구는 주로 의료 영상에 집중되어 있어, 다른 일반적인 컴퓨터 비전 도메인에서도 동일한 수준의 성능 향상이 나타날지에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

이 논문은 깊은 합성곱 신경망(DCN)의 도메인 적응 문제를 해결하기 위해, 각 중간 계층에서 도메인 불변 특징을 추출하는 **L-DANN**과 **L-WASS** 기법을 제안한다. Squeeze 연산을 통해 각 계층의 특징을 요약하고 이를 도메인 분류기나 Critic에 입력함으로써, 깊은 네트워크에서도 그래디언트 소멸 없이 효과적으로 도메인 적응을 수행할 수 있다. 특히 레이블이 부족한 의료 영상 데이터셋에서 기존 DANN보다 훨씬 안정적이고 우수한 성능을 보였으며, 이는 향후 데이터 희소성 문제가 심한 의료 AI 연구에 중요한 방법론적 기여를 할 것으로 보인다.
