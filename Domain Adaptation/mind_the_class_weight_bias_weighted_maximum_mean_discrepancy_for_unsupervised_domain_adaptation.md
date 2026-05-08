# Mind the Class Weight Bias: Weighted Maximum Mean Discrepancy for Unsupervised Domain Adaptation

Hongliang Yan, Yukang Ding, Peihua Li, Qilong Wang, Yong Xu, Wangmeng Zuo

## 🧩 Problem to Solve

기존의 MMD(Maximum Mean Discrepancy) 기반 도메인 적응(Domain Adaptation) 방법들은 소스 도메인과 타겟 도메인 간의 클래스 사전 분포(prior distribution), 즉 '클래스 가중치 편향(class weight bias)'의 변화를 일반적으로 무시합니다. 이러한 편향은 샘플 선택 기준이나 애플리케이션 시나리오의 변화로 인해 흔히 발생하며, MMD가 클래스 가중치 편향을 제대로 처리하지 못하여 도메인 적응 성능 저하를 초래하는 문제가 있습니다. 특히 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서는 타겟 도메인의 레이블을 알 수 없기 때문에 이 문제를 해결하기가 더욱 어렵습니다.

## ✨ Key Contributions

- 도메인 적응에서 클래스 가중치 편향의 영향을 완화하기 위한 새로운 가중 MMD(Weighted MMD, WMMD) 모델을 제안했습니다. WMMD는 클래스 사전 분포를 고려하여 도메인 불일치에 대한 더 나은 측정 기준을 제공합니다.
- 다중 커널 MMD의 편향 없는 추정치를 사용하여, 제안된 WMMD는 선형 시간 복잡도로 계산될 수 있으며, 비지도 도메인 적응을 위해 CNN에 통합될 수 있습니다. 또한, WMMD 모델 훈련을 위한 분류 EM(Classification EM, CEM) 알고리즘을 개발했습니다.
- 광범위한 실험을 통해 WMMD가 도메인 적응에서 기존 MMD를 능가함을 입증했습니다. WMMD의 우수성은 다양한 CNN 아키텍처와 다양한 데이터셋에서 검증되었습니다.

## 📎 Related Works

- **MMD 기반 도메인 적응:** 많은 기존 방법들(예: [26,29,3,41,36])이 MMD를 도메인 분포 불일치 측정에 사용했지만, 클래스 가중치 편향을 고려하지 않았습니다. 특히 [25]의 DAN(Deep Adaptation Networks)은 CNN에 MMD 기반 적응 레이어를 도입했습니다.
- **도메인 불일치 측정 지표:** MMD 외에도 Hellinger 거리 [3], 도메인 분류 [11], 도메인 혼동 [35]과 같은 방법들이 도메인 불변 표현 학습에 사용되었으나, 이들 역시 클래스 가중치 편향은 고려하지 않았습니다.
- **샘플 재가중 또는 선택 방법:** [13,19]와 같은 일부 샘플 재가중 또는 선택 방법은 소스 및 타겟 분포를 일치시키는 데 중점을 두지만, 제안된 WMMD와는 달리 샘플별 가중치를 학습하거나 특정 소스 샘플을 선택하는 방식으로 클래스 가중치 편향을 직접적으로 다루지 않습니다.

## 🛠️ Methodology

본 논문은 클래스 가중치 편향 문제를 해결하기 위해 가중 MMD(WMMD)를 제안하고, 이를 심층 신경망에 통합한 WDAN(Weighted Domain Adaptation Network) 모델을 제시합니다.

1. **가중 MMD(Weighted MMD) 정의:**

   - 소스 도메인 $p_s(x_s)$와 타겟 도메인 $p_t(x_t)$의 분포가 클래스 조건부 분포의 혼합으로 표현될 수 있음을 가정합니다:
     $$p_u(x_u) = \sum_{c=1}^{C} w_u_c p_u(x_u|y_u=c), \quad u \in \{s, t\}$$
     여기서 $w_u_c$는 클래스 $c$의 사전 확률입니다.
   - 클래스 가중치 편향을 제거하기 위해, 타겟 도메인과 동일한 클래스 가중치를 가지면서 소스 도메인의 클래스 조건부 분포를 유지하는 참조 소스 분포 $p_{s,\alpha}(x_s)$를 구성합니다. 각 클래스 $c$에 대해 가중치 $\alpha_c = w_t_c / w_s_c$를 도입하여, $p_{s,\alpha}(x_s) = \sum_{c=1}^{C} \alpha_c w_s_c p_s(x_s|y_s=c)$로 정의합니다.
   - 가중 MMD는 이 참조 소스 분포와 타겟 분포 간의 불일치를 측정합니다:
     $$\text{MMD}^2_w(D_s, D_t) = \left\| \frac{1}{\sum_{i=1}^{M} \alpha_{y^s_i}} \sum_{i=1}^{M} \alpha_{y^s_i} \phi(x^s_i) - \frac{1}{N} \sum_{j=1}^{N} \phi(x^t_j) \right\|^2_H$$
   - 선형 시간 복잡도를 갖는 WMMD의 근사치 $\text{MMD}^2_{l,w}$도 제시됩니다.

2. **가중 도메인 적응 네트워크 (Weighted Domain Adaptation Network, WDAN):**

   - WDAN은 소스 및 타겟 샘플에 대한 경험적 손실과 WMMD 기반 정규화 항을 결합하여 학습됩니다. WMMD 정규화 항은 CNN의 상위 레이어에 추가됩니다.
   - 목표 함수는 다음과 같습니다:
     $$ \min_{W, \{\hat{y}_j\}_{j=1}^{N}, \alpha} \frac{1}{M}\sum_{i=1}^{M} \ell(x^s_i, y^s_i; W) + \gamma \frac{1}{N}\sum_{j=1}^{N} \ell(x^t*j, \hat{y}^t_j; W) + \lambda \sum_{l=l*1}^{l_2} \text{MMD}_{l,w}(D^l_s, D^l_t) $$
        여기서 $W$는 모델 파라미터, $\hat{y}_j$는 타겟 샘플의 의사 레이블, $\alpha$는 보조 가중치, $\ell$은 소프트맥스 손실, $\lambda$와 $\gamma$는 트레이드오프 파라미터입니다.

3. **CEM (Classification EM) 알고리즘을 통한 최적화:**
   WDAN 모델은 타겟 도메인의 레이블 정보 없이 분류 EM 프레임워크를 통해 최적화됩니다. 이는 다음 세 단계로 구성됩니다:
   - **E-단계 (Expectation-step):** 현재 모델 파라미터 $W$를 고정하고, 각 타겟 샘플 $x^t_j$에 대한 클래스 사후 확률 $p(y^t_j=c|x^t_j)$를 CNN의 소프트맥스 출력으로 추정합니다.
   - **C-단계 (Classification-step):** 추정된 사후 확률을 바탕으로 타겟 샘플 $x^t_j$에 의사 레이블 $\hat{y}_j = \arg \max_c p(y^t_j=c|x^t_j)$를 할당합니다. 이 의사 레이블을 사용하여 타겟 클래스 가중치 $\hat{w}^t_c$를 추정하고, 이를 통해 보조 가중치 $\alpha_c = \hat{w}^t_c / w^s_c$를 업데이트합니다.
   - **M-단계 (Maximization-step):** $\alpha$와 의사 레이블 $\hat{y}_j$를 고정하고, 미니 배치 SGD(Stochastic Gradient Descent)를 통해 모델 파라미터 $W$를 업데이트합니다. 역전파를 통해 손실 함수의 기울기가 계산됩니다.

## 📊 Results

- **다양한 데이터셋 및 CNN 아키텍처에서의 우수성:** Office-10+Caltech-10, ImageCLEF, Digit Recognition (MNIST, SVHN, USPS), Office-31 등 널리 사용되는 벤치마크에서 LeNet, AlexNet, GoogLeNet, VGGnet-16과 같은 다양한 CNN 아키텍처를 기반으로 WDAN을 평가했습니다.
- **기존 MMD 방법 대비 성능 향상:**
  - Office-10+Caltech-10에서 WDAN은 AlexNet, GoogLeNet, VGGnet-16 기반으로 DAN(Deep Adaptation Network)보다 각각 1.9%, 0.9%, 0.7% 더 나은 평균 성능을 달성했습니다.
  - ImageCLEF에서 WDAN은 GoogLeNet, DDC, DAN보다 평균적으로 각각 1.4%, 1.2%, 0.9%의 이득을 보였습니다.
  - Digit Recognition에서는 LeNet, SA, DAN보다 각각 평균 11.7%, 10.4%, 3.7% 더 높은 성능을 보였습니다.
  - Office-31에서도 WDAN은 기존 MMD 방법보다 더 좋은 결과를 달성하여, 더 많은 클래스를 가진 데이터셋에서도 효과적임을 입증했습니다.
- **하이퍼파라미터 $\lambda$의 영향:** WMMD 정규화 항의 중요도를 조절하는 $\lambda$ 값에 대한 분석에서 WDAN은 DAN보다 일관되게 우수한 성능을 보였으며, 적절한 $\lambda$ 값이 중요함을 확인했습니다.
- **클래스 가중치 편향에 대한 강건성:** 인위적으로 클래스 가중치 편향을 조절한 실험에서, 기존 MMD 기반 방법(DAN)은 편향이 증가함에 따라 성능이 크게 저하되었으나, WDAN은 클래스 가중치 편향에 대해 훨씬 더 강건한 성능을 보였습니다.
- **특징 시각화:** t-SNE를 이용한 특징 시각화 결과, WDAN은 DAN에 비해 더 나은 클래스 구분 거리를 유지하며 특징을 학습하는 것을 보여주었습니다. 이는 WDAN이 클래스 가중치 편향을 최소화하지 않고 효과적으로 다루기 때문입니다.

## 🧠 Insights & Discussion

- 이 연구의 핵심 통찰은 기존 MMD 기반 도메인 적응 방법들이 '클래스 가중치 편향'이라는 중요한 문제, 즉 소스와 타겟 도메인 간의 클래스 사전 분포 차이를 간과하고 있다는 점입니다. 이러한 편향은 도메인 불일치 측정의 정확도를 떨어뜨려 적응 성능을 저해합니다.
- 제안된 WMMD는 타겟 도메인의 클래스 분포를 반영하는 참조 소스 분포를 구성함으로써, 이러한 클래스 가중치 편향의 영향을 효과적으로 완화합니다. 이는 MMD가 단순히 전체 도메인 분포를 일치시키는 것을 넘어, 클래스별 분포의 상대적 중요성을 고려하도록 확장된 것입니다.
- 비지도 설정에서 타겟 도메인 레이블이 없는 상황을 해결하기 위해 Classification EM 알고리즘을 도입한 것은 실용적인 중요성을 가집니다. 의사 레이블 할당과 보조 가중치 추정을 반복함으로써, 모델은 타겟 도메인의 숨겨진 클래스 구조를 활용하여 도메인 적응을 수행할 수 있게 됩니다.
- 실험 결과는 WDAN이 기존 MMD 기반 방법들보다 지속적으로 우수한 성능을 보이며, 특히 클래스 가중치 편향이 심한 시나리오에서 훨씬 더 강건하다는 것을 증명했습니다. 이는 WMMD가 도메인 불일치를 측정하는 데 있어 더 정확하고 현실적인 메트릭임을 시사합니다.
- **한계점 및 향후 연구:** 본 연구는 주로 CNN 기반 모델에 초점을 맞추었습니다. 향후 연구에서는 WMMD를 비-CNN 기반 UDA 모델에 적용하고, 이미지 생성과 같이 분포 간의 불일치를 측정해야 하는 다른 태스크에 적용하는 방안이 논의될 수 있습니다.

## 📌 TL;DR

본 논문은 비지도 도메인 적응에서 MMD 기반 방법들이 간과하는 '클래스 가중치 편향' 문제를 해결하기 위해, 타겟 도메인의 클래스 사전 분포를 반영하는 보조 가중치를 도입한 가중 MMD(WMMD)를 제안합니다. 이를 CNN에 통합한 WDAN(Weighted Domain Adaptation Network) 모델은 Classification EM 알고리즘을 통해 의사 레이블과 보조 가중치를 추정하며 학습됩니다. 실험 결과, WDAN은 기존 MMD 기반 방법보다 우수한 성능과 클래스 가중치 편향에 대한 강건성을 입증했습니다.
