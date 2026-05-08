# Domain-Adversarial Training of Neural Networks

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Fran ̧cois Laviolette, Mario Marchand, Victor Lempitsky

## 🧩 Problem to Solve

학습(소스) 및 테스트(타겟) 데이터 분포가 유사하지만 다른 경우(도메인 이동, domain shift) 머신러닝 모델의 성능 저하 문제가 발생합니다. 특히 레이블이 지정된 타겟 도메인 데이터가 부족하거나 없는 경우(비지도 도메인 적응) 이러한 문제가 두드러집니다. 이 연구의 목표는 소스 도메인에서는 주요 학습 작업을 위해 판별적(discriminative)이면서도, 소스와 타겟 도메인 간의 이동에 대해서는 비판별적(indiscriminate), 즉 도메인 불변(domain-invariant)적인 특징 표현을 학습하는 것입니다.

## ✨ Key Contributions

- **도메인 적대적 신경망(Domain-Adversarial Neural Networks, DANN) 도입**: 비지도 도메인 적응을 위한 새로운 표현 학습(representation learning) 접근 방식을 제안합니다.
- **Gradient Reversal Layer (GRL)**: 표준 신경망 아키텍처에 GRL이라는 새로운 레이어를 추가하여, 역전파 과정에서 특정 기울기의 부호를 역전시킴으로써 도메인 불변 특징 학습을 가능하게 합니다. 이는 표준 역전파와 확률적 경사 하강법(SGD)을 사용하여 훈련될 수 있습니다.
- **통합된 학습 프레임워크**: 특징 추출기(feature extractor), 레이블 예측기(label predictor), 도메인 분류기(domain classifier)를 단일 딥 러닝 아키텍처 내에서 동시에 훈련하여, 도메인 불변적이고 판별적인 특징을 공동으로 최적화합니다.
- **최첨단 성능 달성**: 문서 감성 분석, 이미지 분류, 사람 재식별(person re-identification)과 같은 다양한 도메인 적응 벤치마크에서 최첨단(state-of-the-art) 성능을 달성했습니다.
- **이론적 기반**: Ben-David et al.의 도메인 적응 이론, 특히 $H$-divergence의 개념에서 직접 영감을 받아 도메인 간 특징 분포를 구분할 수 없도록 최적화합니다.

## 📎 Related Works

- **기존 도메인 적응 연구**: 선형 가설에 초점을 맞추거나(Blitzer et al., 2006) 특징 분포를 직접 정합하는 방식(MMD 기반: Borgwardt et al., 2006; Huang et al., 2006; 서브스페이스 정렬: Fernando et al., 2013)들이 있었습니다.
- **딥 러닝 기반 도메인 적응**: Marginalized Stacked Denoising Autoencoders (mSDA) (Chen et al., 2012)와 같이 오토인코더를 사용하여 견고한 특징을 학습하거나, 점진적 도메인 전환(Chopra et al., 2013)을 시도하는 방법들이 있었습니다.
- **생성적 적대 신경망(Generative Adversarial Networks, GANs) (Goodfellow et al., 2014)**: DANN은 GAN과 유사하게 적대적 학습 개념을 사용하여 두 분포 간의 불일치를 최소화하지만, GAN은 데이터를 생성하는 반면 DANN은 도메인 불변 특징을 학습하는 것이 목적입니다.
- **동시 연구**: DDC (Tzeng et al., 2014) 및 DAN (Long and Wang, 2015)과 같은 다른 딥 도메인 적응 방법들은 도메인 간 데이터 분포 평균의 거리를 측정하고 최소화하는 데 초점을 맞춘 반면, DANN은 판별적 분류기로 두 분포를 구별할 수 없게 만듭니다.
- **이론적 배경**: 이 연구는 $H$-divergence를 기반으로 한 Ben-David et al. (2006, 2010)의 도메인 적응 이론에서 직접 파생되었습니다.

## 🛠️ Methodology

1. **네트워크 아키텍처**: DANN은 크게 세 가지 부분으로 구성됩니다.
   - **특징 추출기($G_f(\mathbf{x}; \theta_f)$)**: 입력 $\mathbf{x}$를 D차원 특징 공간으로 매핑하는 신경망(예: CNN).
   - **레이블 예측기($G_y(G_f(\mathbf{x}); \theta_y)$)**: 추출된 특징을 사용하여 소스 도메인 데이터의 클래스 레이블을 예측하는 부분.
   - **도메인 분류기($G_d(G_f(\mathbf{x}); \theta_d)$)**: 추출된 특징이 소스 도메인에서 왔는지 타겟 도메인에서 왔는지 분류하는 부분.
2. **최적화 목표**: 네트워크는 다음과 같은 saddle point 최적화 문제를 해결하도록 훈련됩니다.
   $$(\hat{\theta}_f, \hat{\theta}_y) = \text{argmin}_{\theta_f, \theta_y} E(\theta_f, \theta_y, \hat{\theta}_d)$$
   $$\hat{\theta}_d = \text{argmax}_{\theta_d} E(\hat{\theta}_f, \hat{\theta}_y, \theta_d)$$
   여기서 목적 함수 $E$는 다음과 같습니다:
   $$E(\theta_f, \theta_y, \theta_d) = \frac{1}{n} \sum_{i=1}^n L_y(G_y(G_f(x_i;\theta_f);\theta_y),y_i) - \lambda \left( \frac{1}{n} \sum_{i=1}^n L_d(G_d(G_f(x_i;\theta_f);\theta_d),0) + \frac{1}{n'} \sum_{j=1}^{n'} L_d(G_d(G_f(x_j;\theta_f);\theta_d),1) \right)$$
   $L_y$는 레이블 예측 손실(예: 교차 엔트로피), $L_d$는 도메인 분류 손실(예: 이항 교차 엔트로피)이며, $\lambda$는 도메인 적응 강도를 조절하는 하이퍼파라미터입니다.
3. **Gradient Reversal Layer (GRL)**: 이 레이어는 특징 추출기($G_f$)와 도메인 분류기($G_d$) 사이에 삽입됩니다.
   - **순전파**: 입력값을 그대로 통과시킵니다 ($R(x) = x$).
   - **역전파**: 다음 레이어에서 전달된 기울기에 $-1$을 곱하여 이전 레이어로 전달합니다 ($\frac{dR}{dx} = -I$).
   - GRL 덕분에, 도메인 분류기 손실 $L_d$에 대한 특징 추출기 파라미터 $\theta_f$의 기울기는 자동으로 역전되어 $\theta_f$는 $L_y$를 최소화하면서 동시에 $L_d$를 *최대화*하려는 방향으로 업데이트됩니다. 이는 특징 추출기가 도메인 분류기를 혼란스럽게 만드는, 즉 도메인 불변 특징을 생성하도록 유도합니다.
4. **훈련 과정**: 표준 역전파와 SGD를 사용하여 모델을 훈련하며, 학습률 스케줄과 적응 파라미터 $\lambda$ 스케줄을 사용합니다. 하이퍼파라미터 선택을 위해 역검증(reverse validation) 방법을 사용합니다.

## 📊 Results

- **장난감 문제 (Inter-twinning moons)**: DANN은 표준 신경망(NN)이 잘 적응하지 못하는 타겟 분포에 대해 완벽하게 분류하고, 특징 공간에서 소스와 타겟 데이터가 더 잘 섞이는(indistinguishable) 것을 시각적으로 보여주었습니다.
- **감성 분석 (Amazon Reviews)**: DANN은 표준 NN과 SVM보다 훨씬 뛰어난 성능을 보였으며, mSDA와 결합했을 때도 성능을 더욱 향상시켰습니다.
- **이미지 분류 (MNIST, SVHN, Office)**:
  - **MNIST $\to$ MNIST-M, Synthetic Numbers $\to$ SVHN, Synthetic Signs $\to$ GTSRB**: DANN은 source-only 모델 대비 성능을 크게 향상시켰으며, Subspace Alignment (SA)와 같은 다른 도메인 적응 방법들을 능가했습니다. 특히 SVHN 데이터셋에서는 source-only와 target-only 모델 간의 성능 격차를 거의 80% 채웠습니다.
  - **Office 데이터셋**: DANN은 Amazon, DSLR, Webcam 도메인 간의 전환 작업에서 기존의 최첨단 방법들을 뛰어넘는 새로운 최고 정확도를 달성했습니다.
- **Proxy A-distance (PAD) 분석**: DANN으로 학습된 특징 표현은 PAD 값을 현저히 낮추어, 소스와 타겟 도메인 간의 구별 불가능성을 정량적으로 입증했습니다.
- **사람 재식별 (VIPeR, PRID, CUHK)**: DANN은 분류 문제를 넘어 특징 기술자 학습(descriptor learning)에도 성공적으로 적용되었으며, 여러 데이터셋 쌍에서 재식별 정확도를 일관되게 향상시켰습니다.

## 🧠 Insights & Discussion

- **이론과 실제의 연결**: DANN은 도메인 적응 이론에서 제안하는 $H$-divergence 최소화 아이디어를 딥 신경망 아키텍처에 직접 구현하여, 이론적 통찰이 실제 성능 향상으로 이어진다는 것을 보여줍니다.
- **일반성과 유연성**: Gradient Reversal Layer는 개념적으로 간단하면서도 기존의 거의 모든 역전파 기반 피드포워드 신경망에 쉽게 통합될 수 있어 높은 유연성을 가집니다. 이를 통해 분류 외에 기술자 학습과 같은 다양한 작업에도 성공적으로 적용될 수 있음을 입증했습니다.
- **도메인 불변 특징의 중요성**: 실험 결과는 소스와 타겟 도메인 간의 특징 분포를 최대한 구별할 수 없게 만드는 것이 타겟 도메인에서 모델의 일반화 성능을 크게 향상시킨다는 것을 명확히 보여줍니다.
- **정규화 효과**: 도메인 분류기 브랜치는 훈련 초기 단계의 노이즈 신호에 대한 민감도를 줄이고 과적합을 방지하는 정규화 역할을 수행할 수 있습니다.
- **한계**: MNIST $\to$ SVHN과 같이 도메인 간 차이가 극심한 경우에는 DANN도 성능 향상에 실패하는 사례가 있었습니다. 이는 도메인 적응 방법의 근본적인 한계일 수 있습니다.

## 📌 TL;DR

이 논문은 비지도 도메인 적응을 위한 **도메인 적대적 신경망(DANN)**을 제안합니다. DANN은 **Gradient Reversal Layer (GRL)**를 활용하여 신경망이 소스 도메인 레이블에 대해 판별적이면서도 소스와 타겟 도메인 간에 구분 불가능한(도메인 불변) 특징 표현을 공동으로 학습하도록 합니다. 이러한 적대적 훈련 방식은 표준 역전파를 통해 쉽게 구현될 수 있으며, 감성 분석, 이미지 분류, 사람 재식별 등 다양한 도메인 적응 벤치마크에서 최첨단 성능을 달성하며 폭넓은 적용 가능성과 효과를 입증했습니다.
