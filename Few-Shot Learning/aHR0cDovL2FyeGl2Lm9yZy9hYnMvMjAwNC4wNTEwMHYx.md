# $MA^3$: Model Agnostic Adversarial Augmentation for Few Shot learning

Rohit Jena, Shirsendu Sukanta Halder, Katia Sycara (2020)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델이 학습 데이터에 없는 새로운 예시(unseen examples)에 대해 일반화 성능을 높이는 문제, 특히 데이터가 극소수인 Few-Shot Learning (FSL) 환경에서의 일반화 성능 향상을 해결하고자 한다. 

일반적인 지도 학습과 달리 FSL은 매우 적은 수의 레이블된 데이터만으로 새로운 클래스를 인식해야 하므로, 모델이 주어진 소수의 데이터에 과적합(overfitting)되기 쉽다는 치명적인 문제가 있다. 기존의 데이터 증강(Data Augmentation) 방식이나 생성적 적대 신경망(GAN) 기반의 증강 기법들은 이미지 자체의 분포를 학습해야 하므로 학습 속도가 느리고 복잡하며, 때로는 불안정한 결과를 초래한다. 따라서 본 연구의 목표는 네트워크 구조의 변경 없이도 적용 가능하며, 학습이 빠르고 효율적인 새로운 적대적 데이터 증강 기법을 제안하여 FSL의 일반화 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 전체의 픽셀 분포를 생성하는 대신, 이미지를 변형시키는 **변환 파라미터(transformation parameters)의 확률 분포를 학습**하는 것이다. 

연구진은 인간이 새로운 물체를 인식할 때 단일한 스냅샷이 아니라 다양한 관점에서 관찰하며 일반화하는 방식에서 영감을 얻었다. 이를 구현하기 위해 3D 카메라 핀홀 모델의 투영 변환(projective transformation) 이론을 바탕으로, 원거리 물체의 관점 변화를 2D Affine Transform으로 근사화하였다. 이렇게 도출된 변환 파라미터를 Spatial Transformer Networks (STN)를 통해 예측하고, 이를 적대적(adversarial) 방식으로 학습시켜 분류기가 가장 취약한 변형 예시를 생성하게 함으로써 모델의 강건성을 강제로 높이는 전략을 취한다.

## 📎 Related Works

기존의 Few-Shot Learning 연구들은 주로 Prototypical Networks, Matching Networks와 같은 Metric-based approach나 MAML과 같은 Meta-learning 프레임워크를 통해 적은 데이터로의 빠른 적응을 꾀하였다. 또한, 데이터 증강을 위해 GAN을 사용하여 가짜 데이터를 생성하거나(MetaGAN), 강화학습을 통해 최적의 증강 정책을 검색하는 방식(AutoAugment)이 제안된 바 있다.

그러나 GAN 기반 방식은 데이터 분포 전체를 학습해야 하므로 비용이 많이 들고, 강화학습 기반 방식은 보상 함수가 진화함에 따라 학습이 불안정해질 수 있다는 한계가 있다. 본 논문이 제안하는 $MA^3$는 이미지 자체가 아닌 Affine 변환 행렬의 요소들을 직접 학습하며, 전체 파이프라인이 미분 가능(fully differentiable)하도록 설계되어 기존의 어떤 FSL 베이스라인 모델에도 쉽게 결합될 수 있다는 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 작동 원리
본 모델은 적대적 관계에 있는 두 네트워크, 즉 Few-Shot Learner ($g_\theta$)와 Adversarial Transformer ($f_\phi$)로 구성된다. Transformer는 입력 이미지에 적용할 Affine 변환 파라미터를 생성하여 이미지를 변형시키고, Learner는 변형된 이미지를 통해 분류를 수행한다.

### 수학적 근거 및 Affine Transform 근사
연구진은 3D 공간의 점 $(x, y, z, 1)^T$가 이미지 평면 $(u, v, 1)^T$로 투영될 때, 약간의 회전(roll, yaw, pitch)과 이동(translation)이 발생한다고 가정한다. 물체가 카메라에서 충분히 멀리 떨어져 있다는 가정($z \approx z_0 \gg 1$) 하에 테일러 전개(Taylor expansion)와 이항 전개(binomial expansion)를 적용하면, 투영 변환은 다음과 같은 Affine 변환 형태로 근사될 수 있다.

$$
\begin{bmatrix} u_2 \\ v_2 \end{bmatrix} = \begin{bmatrix} 1 + \delta_1 & \delta_2 & \delta_3 \\ \delta_4 & 1 + \delta_5 & \delta_6 \end{bmatrix} \begin{bmatrix} u_1 \\ v_1 \\ 1 \end{bmatrix}
$$

여기서 $\delta_i$는 작은 변위 값들을 의미하며, 이는 이미지가 정체성 변환(identity transform)에서 크게 벗어나지 않도록 유도한다.

### 학습 목표 및 손실 함수
본 모델은 Min-Max 게임 형태의 최적화 문제를 해결한다. Transformer는 분류기의 손실을 최대화하는 방향으로 파라미터를 학습하고, 분류기는 이를 최소화하는 방향으로 학습한다.

$$
\max_{\phi} \min_{\theta} \sum_{i=1}^{m} L(g_\theta(q_i | f_\phi(s_1), f_\phi(s_2), \dots, f_\phi(s_n)))
$$

여기서 $S$는 support set, $Q$는 query set을 의미한다. 하지만 Transformer가 이미지를 완전히 망가뜨려(예: 과도한 줌) 분류기가 학습 불가능한 상태로 만드는 것을 방지하기 위해, 정체성 행렬(Identity Matrix)과의 차이를 벌주는 정규화 항 $L_{reg}$를 도입한다.

$$
L_{reg}(f_\phi(s)) = \left\| \begin{bmatrix} a_1(s) & a_2(s) & a_3(s) \\ a_4(s) & a_5(s) & a_6(s) \end{bmatrix} - \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} \right\|^2
$$

최종 최적화 식은 다음과 같다.

$$
\max_{\phi} \min_{\theta} \sum_{i=1}^{m} L(g_\theta(q_i | f_\phi(s_1), \dots, f_\phi(s_n))) - \lambda \sum_{j=1}^{n} L_{reg}(f_\phi(s_j))
$$

여기서 $\lambda$는 정규화의 강도를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정
- **데이터셋**: Omniglot, miniImageNet
- **베이스 모델**: Prototypical Networks, Matching Networks, MAML (Model-Agnostic 특성 검증)
- **평가 방법**: 학습 시에는 support set에만 증강을 적용하고 query set은 그대로 유지하며, 테스트 시에는 STN을 비활성화한다.

### 주요 결과
1. **Omniglot**: 데이터셋 자체가 비교적 단순하여 베이스라인의 정확도가 이미 매우 높았기에, $MA^3$ 적용 시 성능 향상은 미미한 수준이었다.
2. **miniImageNet**: 데이터의 변동성이 커서 효과가 뚜렷하게 나타났다. 베이스 모델의 구조 변경 없이 증강 모듈만 추가했을 때, 최대 **3.8%의 성능 향상**을 보였다.
3. **정규화의 중요성**: 정규화 계수 $\lambda=0$일 때 성능이 급격히 하락하는 것이 관찰되었다. 이는 STN이 분류기를 방해하기 위해 이미지를 알아볼 수 없게 변형시키는 '취약점 공격'에 치중하게 되어, 결과적으로 분류기가 유용한 특징을 학습하지 못하게 만들기 때문이다.
4. **표준 증강과의 비교**: 단순한 무작위 회전, 이동, 스케일링보다 적대적으로 학습된 STN 기반 증강이 더 일관되고 높은 성능 향상을 보였다.

## 🧠 Insights & Discussion

본 논문은 데이터 증강을 단순한 데이터 늘리기가 아니라, **분류기가 학습하지 못한 취약한 관점을 능동적으로 찾아내어 보완하는 과정**으로 재정의했다는 점에서 강점이 있다. 특히 STN을 사용하여 파라미터 공간에서 최적화를 수행함으로써 GAN보다 훨씬 가볍고 빠르게 학습할 수 있음을 증명하였다. 또한, 다양한 FSL 프레임워크(Metric-based, Meta-learning)에 공통적으로 적용 가능함을 보여줌으로써 모델 불가지론적(Model-Agnostic) 특성을 입증하였다.

다만, 본 연구는 Affine Transform 범위 내에서의 변형만을 다루고 있다. 실제 환경에서 발생할 수 있는 비선형적인 왜곡(Non-linear deformation)이나 복잡한 3D 회전까지는 커버하지 못한다는 한계가 있다. 또한, 정규화 파라미터 $\lambda$에 대한 민감도가 높아, 최적의 $\lambda$를 찾기 위한 그리드 서치 과정이 필수적이라는 점이 실용적 측면에서의 부담이 될 수 있다.

## 📌 TL;DR

본 논문은 Few-Shot Learning의 일반화 성능을 높이기 위해, 3D 투영 이론을 바탕으로 한 **적대적 Affine 변환 파라미터 학습 기법($MA^3$)**을 제안한다. STN을 이용해 분류기가 가장 어려워하는 변형 이미지를 생성하고 이를 학습함으로써, 모델 구조 변경 없이도 miniImageNet 등에서 약 3.8%의 성능 향상을 달성하였다. 이 연구는 데이터가 극소수인 환경에서 효율적인 적대적 증강이 강건한 특징 추출에 중요한 역할을 할 수 있음을 시사한다.