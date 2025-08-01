# Learning to Adapt Structured Output Space for Semantic Segmentation
Yi-Hsuan Tsai, Wei-Chih Hung, Samuel Schulter, Kihyuk Sohn, Ming-Hsuan Yang, Manmohan Chandraker

## 🧩 Problem to Solve
의미론적 분할을 위한 합성곱 신경망(CNN) 기반 접근 방식은 픽셀 수준의 정확한 정답(ground truth) 주석에 크게 의존하지만, 새로운 이미지 도메인에는 잘 일반화되지 않습니다. 이미지 주석 작업은 지루하고 노동 집약적이므로, 소스 도메인의 정답 레이블을 타겟 도메인에 적응시킬 수 있는 알고리즘을 개발하는 것이 매우 중요합니다.

## ✨ Key Contributions
*   **출력 공간에서의 적대적 학습 제안:** 픽셀 수준 의미론적 분할을 위한 도메인 적응 방법으로, 분할 결과를 구조화된 출력으로 간주하고 출력 공간에서 적대적 학습(adversarial learning)을 수행합니다.
*   **효과적인 출력 공간 정렬 입증:** 출력 공간에서의 적응이 소스 및 타겟 이미지 간의 장면 레이아웃(spatial layout) 및 로컬 컨텍스트(local context)를 효과적으로 정렬할 수 있음을 입증합니다.
*   **다단계 적대적 학습 체계 개발:** 분할 모델의 다양한 특징 레벨에서 특징을 효과적으로 적응시키기 위한 다단계(multi-level) 적대적 학습 방식을 개발하여 성능을 향상시킵니다.

## 📎 Related Works
*   **의미론적 분할:** FCN(Fully Convolutional Network) [24]을 시작으로 CNN 기반 방법론이 크게 발전했으며, 맥락 정보(context information) 활용 [15, 42] 및 수용 필드(receptive field) 확대 [2, 40]를 통해 성능을 개선해왔습니다. 대규모 주석 데이터의 필요성 때문에 약지도 학습(weakly supervised) [5, 14, 17, 29, 30] 및 합성 데이터셋(GTA5 [32], SYNTHIA [33]) 활용 연구가 진행되었으나, 합성 데이터와 실제 데이터 간의 도메인 차이가 문제였습니다.
*   **도메인 적응:** 이미지 분류 분야에서는 DANN(Domain-Adversarial Neural Network) [7, 8]과 같이 특징 분포를 정렬하는 방식이나 PixelDA [1]와 같이 소스 이미지를 타겟 도메인으로 변환하는 방식 등 다양한 CNN 기반 방법론이 개발되었습니다.
*   **픽셀 수준 도메인 적응:** 픽셀 수준 예측 작업에서는 Hoffman et al. [13]이 특징 표현에 적대적 학습을 적용했으며, CyCADA [12]는 CycleGAN [44]과 특징 공간 적대적 학습을 결합했습니다. 본 논문은 기존 특징 공간 적응 방식의 한계(고차원 특징의 복잡성)를 극복하기 위해 출력 공간 적응의 필요성을 강조합니다.

## 🛠️ Methodology
*   **모델 구성:** 제안하는 도메인 적응 알고리즘은 분할 네트워크 $G$와 판별자 $D_i$ ($i$는 다단계 학습의 레벨)로 구성됩니다.
*   **출력 공간 적응 동기:** 이미지 외관은 도메인마다 매우 다를 수 있지만, 의미론적 분할 출력은 공간적 레이아웃과 로컬 컨텍스트에서 강한 유사성을 공유한다는 점에 착안했습니다. 이를 통해 저차원 분할 softmax 출력을 적대적 학습을 통해 적응시킵니다.
*   **목표 함수:**
    $$ L(I_s, I_t) = L_{seg}(I_s) + \lambda_{adv} L_{adv}(I_t) $$
    여기서 $L_{seg}(I_s)$는 소스 도메인 $I_s$의 정답 주석에 대한 교차 엔트로피 분할 손실이며, $L_{adv}(I_t)$는 타겟 이미지 $I_t$의 예측 분할을 소스 예측 분포에 가깝게 만들도록 유도하는 적대적 손실입니다. $\lambda_{adv}$는 두 손실의 균형을 맞추는 가중치입니다.
*   **단일 레벨 적대적 학습:**
    *   **판별자 훈련:** 분할 네트워크의 softmax 출력 $P = G(I) \in \mathbb{R}^{H \times W \times C}$를 입력으로 받아, $P$가 소스 도메인($z=1$)에서 왔는지 타겟 도메인($z=0$)에서 왔는지를 구별하도록 이진 교차 엔트로피 손실 $L_d(P)$로 훈련됩니다.
    *   **분할 네트워크 훈련:** 소스 이미지에 대해서는 $L_{seg}(I_s)$를 최소화하고, 타겟 이미지 $I_t$에 대해서는 예측 $P_t = G(I_t)$가 판별자를 속여 소스 예측으로 인식되도록 적대적 손실 $L_{adv}(I_t)$를 최소화합니다.
*   **다단계 적대적 학습:** 출력 공간 적응이 저수준 특징을 잘 적응시키지 못할 수 있다는 점을 고려하여, 분할 모델의 다양한 특징 레벨(예: conv4, conv5)에 보조 분류기(auxiliary classifier)와 추가 판별 모듈을 통합합니다. 확장된 목표 함수는 다음과 같습니다:
    $$ L(I_s, I_t) = \sum_{i} \lambda_{i}^{seg} L_{i}^{seg}(I_s) + \sum_{i} \lambda_{i}^{adv} L_{i}^{adv}(I_t) $$
    여기서 $i$는 분할 출력을 예측하는 데 사용된 레벨을 나타냅니다. 최적화는 $\max_D \min_G L(I_s, I_t)$의 min-max 기준으로 수행됩니다.
*   **네트워크 아키텍처:**
    *   **판별자:** 4x4 커널과 stride 2를 가진 5개의 합성곱 레이어로 구성된 완전 합성곱 네트워크(FCN)를 사용하며, Leaky ReLU 활성화 함수를 적용합니다.
    *   **분할 네트워크:** ImageNet [6]으로 사전 학습된 ResNet-101 [11]을 백본으로 하는 DeepLab-v2 [2] 프레임워크를 기반으로 합니다. Dilated Convolution [40]과 ASPP(Atrous Spatial Pyramid Pooling) [2]를 사용하여 수용 필드를 확장합니다.
*   **훈련:** 분할 네트워크와 판별자는 한 단계에서 공동으로 훈련됩니다. SGD(Stochastic Gradient Descent) 및 Adam 옵티마이저를 사용하며, 학습률은 다항식 감쇠(polynomial decay) 방식으로 조정합니다.

## 📊 Results
*   **합성-실제 적응 (GTA5/SYNTHIA → Cityscapes):**
    *   제안된 출력 공간 적응(single-level)은 VGG-16 기반 모델에서 기존 SOTA 특징 적응 방법들(FCNs in the Wild [13], CDA [41], CyCADA [12])보다 우수한 mIoU(mean Intersection-over-Union) 성능을 보였습니다.
    *   ResNet-101 기반의 강력한 베이스라인 사용 시, 다단계 적대적 학습이 단일 레벨 적응 및 특징 공간 적응보다 mIoU를 추가로 향상시켰습니다 (GTA5 $\rightarrow$ Cityscapes에서 41.4% $\rightarrow$ 42.4%).
    *   적응 모델과 완전 지도 학습 모델(oracle) 간의 mIoU 격차를 측정한 결과, 제안된 다단계 모델이 가장 작은 격차를 달성했습니다 (SYNTHIA $\rightarrow$ Cityscapes에서 25.0%).
*   **파라미터 분석:** 출력 공간 적응은 특징 공간 적응보다 $\lambda_{adv}$ 가중치에 덜 민감하여, 더 넓은 범위의 가중치에서 안정적인 성능을 보였습니다.
*   **도시 간 적응 (Cityscapes → Cross-City):** Cityscapes를 소스로 하여 Rome, Rio, Tokyo, Taipei의 타겟 도시에 적응시킨 실험에서도 제안된 다단계 모델이 일관된 성능 향상을 보여, 더 작은 도메인 격차에서도 효과적임을 입증했습니다.
*   **LS-GAN (부록):** Least Squares GAN(LS-GAN) [28] 목적 함수를 사용했을 때, 바닐라 GAN(vanilla GAN)보다 더 높은 mIoU를 달성하여 더 안정적인 학습과 고품질 결과 생성을 가능하게 했습니다.
*   **Synscapes (부록):** Synscapes [39]에서 Cityscapes로의 적응에서도 출력 공간 적응이 성능을 개선함을 보여주었습니다.

## 🧠 Insights & Discussion
*   **구조화된 출력 공간의 이점:** 이 연구의 핵심 통찰은 의미론적 분할 결과가 이미지 외관의 도메인 변화에도 불구하고 공간적 및 로컬 컨텍스트에서 유사성을 유지하는 '구조화된 출력'이라는 점을 활용한 것입니다. 이는 고차원적이고 복잡한 특징 공간에서 직접 적응하는 것보다, 의미론적으로 더 일관된 저차원의 출력 공간에서 적응하는 것이 픽셀 수준 예측 작업에 더 효과적임을 시사합니다.
*   **다단계 적응의 중요성:** 고수준 출력 레이블에서 멀리 떨어진 저수준 특징까지 적응시키는 다단계 접근 방식은 모델의 일반화 능력과 성능을 향상시키는 데 필수적입니다. 이는 마치 심층 감독(deep supervision)과 유사하게, 다양한 추상화 수준에서 도메인 불변적인 표현을 학습하도록 유도합니다.
*   **실용적 의의:** 본 방법은 값비싼 픽셀 수준 주석 없이도 모델을 새로운 도메인에 효과적으로 적응시킬 수 있는 실용적인 해결책을 제시하며, 자율 주행과 같은 실제 애플리케이션에서 합성 데이터의 활용도를 높이는 데 기여할 수 있습니다.
*   **한계:** 전봇대나 교통 표지판과 같이 작은 객체들은 배경 클래스와 쉽게 병합될 수 있어 여전히 적응이 어렵다는 한계가 있습니다. 이는 향후 연구에서 특정 객체에 대한 적응 능력 향상이나 더 정교한 컨텍스트 모델링이 필요함을 시사합니다.

## 📌 TL;DR
픽셀 수준 의미론적 분할은 도메인 간 주석 데이터 부족과 도메인 차이로 인한 일반화 문제에 직면합니다. 본 논문은 분할 결과가 '구조화된 출력'으로서 도메인 간 공간적 유사성을 공유한다는 점에 착안하여, 출력 공간에서 적대적 학습을 통한 도메인 적응 방법(AdaptSegNet)을 제안합니다. 특히, 분할 모델의 여러 특징 레벨에서 적대적 학습을 수행하는 다단계 적응 방식을 통해 저수준 특징까지 효과적으로 정렬합니다. 광범위한 실험을 통해 제안 방법이 합성-실제 및 도시 간 시나리오에서 기존 SOTA 방법을 능가하며, 성능 격차를 크게 줄임을 입증했습니다. 이는 출력 공간 적응이 픽셀 수준 예측 작업에 효과적이며, 다양한 도메인에 걸쳐 일반화 가능한 강력한 분할 모델 학습에 기여함을 보여줍니다.