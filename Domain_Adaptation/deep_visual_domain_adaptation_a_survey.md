# Deep Visual Domain Adaptation: A Survey

Mei Wang, Weihong Deng

## 🧩 Problem to Solve

최근 딥러닝은 다양한 분야에서 놀라운 성과를 보였지만, 대량의 레이블링된 데이터 확보는 여전히 매우 비싸고 시간이 많이 소요되는 문제입니다. 이러한 데이터 부족 문제를 해결하기 위해 전이 학습(Transfer Learning, TL)의 한 분야인 도메인 적응(Domain Adaptation, DA)이 중요해지고 있습니다. 특히, 딥러닝 모델이 소스 도메인에서 학습한 지식이 타겟 도메인으로 전이될 때 발생하는 **도메인 시프트(domain shift) 또는 분포 변화(distribution change)**로 인해 성능이 저하되는 문제를 해결하는 것이 핵심 연구 과제입니다. 기존의 얕은(shallow) DA 방법론은 한계가 있으므로, 딥러닝의 표현 학습 능력을 활용하여 더욱 전이 가능한(transferable) 표현을 학습하는 딥 도메인 적응(Deep Domain Adaptation, Deep DA) 방법론이 필요합니다.

## ✨ Key Contributions

이 논문은 컴퓨터 비전 분야의 딥 도메인 적응 방법론에 대한 포괄적인 조사를 제공하며, 다음과 같은 네 가지 주요 기여를 합니다.

- **다양한 Deep DA 시나리오 분류:** 두 도메인 간의 데이터 특성 차이를 기반으로 다양한 Deep DA 시나리오에 대한 분류 체계를 제시합니다.
- **Deep DA 접근 방식 요약 및 비교:** 학습 손실(training loss)을 기준으로 Deep DA 접근 방식을 여러 범주로 요약하고, 각 범주별 최신 방법론들을 분석 및 비교합니다.
- **컴퓨터 비전 애플리케이션 개요:** 이미지 분류를 넘어 얼굴 인식, 시맨틱 분할, 객체 탐지 등 Deep DA의 다양한 컴퓨터 비전 응용 분야를 조망합니다.
- **현재 방법론의 잠재적 결점 및 미래 방향 제시:** 현재 Deep DA 방법론의 한계점과 향후 연구 방향을 강조합니다.

## 📎 Related Works

기존에도 전이 학습(TL) 및 도메인 적응(DA)에 대한 다양한 연구(예: Pan et al. [83], Shao et al. [101], Patel [84], Zhang et al. [137])가 있었지만, 대부분 얕은(shallow) TL 또는 DA 방법론에 중점을 두었습니다. Csurka et al. [19]의 연구는 얕은 DA 방법론을 주로 다루면서 딥 DA를 간략하게 언급하고 학습 손실에 기반하여 세 가지 하위 범주로 분류했지만, 주로 이미지 분류 애플리케이션에만 국한되었습니다. 본 논문은 이러한 기존 연구의 한계를 넘어서, 딥러닝 기반의 DA 방법론에 초점을 맞추고, 더 상세한 분류 체계와 광범위한 컴퓨터 비전 응용 분야를 다루는 시의적절한 리뷰를 제공합니다.

## 🛠️ Methodology

본 논문은 Deep DA를 **"도메인 적응을 위한 딥러닝 아키텍처를 기반으로 백프로파게이션을 통해 최적화되는 방법"**이라는 좁은 의미로 정의하고, "좋은" 특징 표현(feature representations)을 학습하기 위한 다양한 접근 방식을 제시합니다.

### A. 도메인 적응(DA) 설정 분류

DA는 도메인 특성에 따라 크게 **동종(Homogeneous) DA**와 **이종(Heterogeneous) DA**로 나뉩니다.

- **동종 DA**: 소스 도메인과 타겟 도메인의 특징 공간($\mathcal{X}_s = \mathcal{X}_t$)이 동일하며 차원도 같지만, 데이터 분포($P(\mathbf{X})_s \neq P(\mathbf{X})_t$)가 다른 경우입니다.
- **이종 DA**: 특징 공간($\mathcal{X}_s \neq \mathcal{X}_t$)과 차원($d_s \neq d_t$)이 모두 다른 경우입니다.

또한, 타겟 도메인의 레이블된 데이터 존재 여부에 따라 **지도(Supervised) DA**, **준지도(Semi-supervised) DA**, **비지도(Unsupervised) DA**로 분류됩니다.

도메인 간의 거리를 고려하여 **단일 단계(One-step) DA**와 **다단계(Multi-step) DA**로 나눌 수 있습니다.

### B. 단일 단계 도메인 적응(One-step DA) 접근 방식

단일 단계 DA의 딥러닝 기반 접근 방식은 주로 세 가지 범주로 요약됩니다:

1. **불일치 기반(Discrepancy-based) 접근 방식**:
   딥 네트워크 모델을 미세 조정(fine-tuning)하여 두 도메인 간의 시프트를 줄입니다.

   - **클래스 기준(Class Criterion)**: 레이블 정보(소스 도메인 또는 타겟 도메인의 제한된 레이블)를 활용합니다.
     - 소프트 레이블(soft label) [45], 의사 레이블(pseudo labels) [130], 속성 표현(attribute representation) [29], 메트릭 학습(metric learning) [53] 등을 통해 지식을 전이합니다.
   - **통계 기준(Statistic Criterion)**: 도메인 간의 통계적 분포 차이를 정렬합니다.
     - 최대 평균 불일치(Maximum Mean Discrepancy, MMD) [74], 상관 관계 정렬(Correlation Alignment, CORAL) [109], 쿨백-라이블러(Kullback-Leibler, KL) 발산 [144], H-발산(H-divergence) 등을 사용합니다.
   - **아키텍처 기준(Architecture Criterion)**: 딥 네트워크의 아키텍처를 조정하여 전이 가능한 특징 학습 능력을 향상시킵니다.
     - 적응형 배치 정규화(Adaptive Batch Normalization, BN) [69], 약하게 연관된 가중치(weak-related weight) [95], 도메인 안내 드롭아웃(domain-guided dropout) [128] 등이 있습니다.
   - **기하학적 기준(Geometric Criterion)**: 소스 도메인과 타겟 도메인 사이의 지오데식 경로(geodesic path)에 중간 부분 공간(intermediate subspaces)을 통합하여 도메인 시프트를 완화합니다.

2. **적대적 기반(Adversarial-based) 접근 방식**:
   도메인 판별자(domain discriminator)를 사용하여 도메인 혼란(domain confusion)을 유도하고, 이를 통해 도메인 불변 특징(domain-invariant features)을 학습합니다.

   - **생성 모델(Generative Models)**: GAN(Generative Adversarial Network) [39] 기반의 생성 컴포넌트를 사용하여 타겟 도메인과 유사한 합성 데이터를 생성하고 레이블 정보를 유지합니다.
     - CoGAN [70], S+U 학습 [104], Bousmalis et al. [4]의 조건부 GAN 등이 있습니다.
   - **비생성 모델(Non-Generative Models)**: 생성 모델 없이 특징 추출기가 소스 도메인의 레이블을 사용하여 판별적 표현을 학습하고, 도메인 혼란 손실(domain-confusion loss)을 통해 타겟 데이터를 동일한 공간에 매핑합니다.
     - DANN(Domain-Adversarial Neural Network) [25], ADDA(Adversarial Discriminative Domain Adaptation) [119], SAN(Selective Adversarial Network) [8] 등이 있습니다.

3. **재구성 기반(Reconstruction-based) 접근 방식**:
   데이터 재구성(data reconstruction)을 보조 작업으로 사용하여 도메인 불변성과 개별 도메인 특성을 동시에 보장합니다.
   - **인코더-디코더 재구성(Encoder-Decoder Reconstruction)**: 오토인코더(autoencoder) [2] 또는 스택형 잡음 제거 오토인코더(stacked denoising autoencoders, SDA) [122]와 같은 구조를 사용하여 공유 인코더로 도메인 불변 표현을 학습하고, 재구성 손실로 도메인 특이적 표현을 유지합니다.
     - DRCN(Deep Reconstruction Classification Network) [33], DSNs(Domain Separation Networks) [5], TLDA(Transfer Learning with Deep Autoencoders) [144] 등이 있습니다.
   - **적대적 재구성(Adversarial Reconstruction)**: 듀얼 GAN(dual GAN) [131], 사이클 GAN(cycle GAN) [143], 디스코 GAN(disco GAN) [59] 등과 같이 생성자와 판별자를 사용하여 원본과 재구성된 이미지 간의 불일치를 측정하고 도메인 간 이미지 변환을 수행합니다.

### C. 다단계 도메인 적응(Multi-step DA) 접근 방식

소스와 타겟 도메인 간의 거리가 멀 때, 일련의 중간 도메인(intermediate domains)을 연결하여 지식 전이를 수행합니다.

- **수작업(Hand-Crafted)**: 사용자가 경험을 기반으로 중간 도메인을 결정합니다 [129].
- **인스턴스 기반(Instance-Based)**: 보조 데이터셋에서 특정 데이터 부분을 선택하여 중간 도메인을 구성합니다 [114], [16].
- **표현 기반(Representation-Based)**: 이전에 학습된 네트워크의 가중치를 고정하고, 그 중간 표현을 새로운 네트워크의 입력으로 사용합니다. 프로그레시브 네트워크(Progressive Networks) [96]가 대표적입니다.

## 📊 Results

Deep DA 방법론은 비전 작업에서 비적응(non-adaptation) 방법론에 비해 상당한 성능 향상을 가져옵니다. 예를 들어, Office-31 데이터셋(Amazon, DSLR, Webcam 도메인)을 사용한 이미지 분류 태스크에서 AlexNet과 같은 표준 딥 네트워크 대비 DANN, DAN, JAN, RTN, CMD 등의 Deep DA 방법들이 일관되게 높은 정확도를 보였습니다. 특히 MMD 기반의 DAN, JAN, RTN이나 적대적 기반의 DANN과 같은 모델들은 평균 70% 초반대의 기본 모델 정확도를 75~80% 이상으로 크게 끌어올렸습니다.

MNIST, USPS, SVHN과 같은 숫자 데이터셋을 사용한 교차 도메인 필기 숫자 인식 태스크에서도 VGG-16을 베이스라인으로 CoGAN, ADDA, DANN과 같은 적대적 기반 방법들이 기본 모델보다 우수한 전이 성능을 보여주었습니다. 이는 Deep DA가 도메인 시프트 문제를 효과적으로 완화하여 타겟 도메인에서의 모델 일반화 능력을 향상시킨다는 것을 의미합니다.

Deep DA는 이미지 분류 외에도 얼굴 인식(예: SSPP-DAN [51]), 객체 탐지(예: LSDA [47], Faster R-CNN 기반 적응 [13]), 시맨틱 분할(예: FCNs with adversarial training [50]), 이미지-대-이미지 변환(예: pix2pix [57], CycleGAN [143]), 사람 재식별(person re-ID, 예: SPGAN [21]), 이미지 캡셔닝(예: Captioner v.s. Critics [11]) 등 다양한 컴퓨터 비전 응용 분야에서 성공적으로 활용되고 있습니다. 이러한 응용 분야에서 Deep DA는 제한된 레이블 데이터 상황에서도 강력한 성능 향상을 제공하는 핵심 기술로 자리 잡고 있습니다.

## 🧠 Insights & Discussion

본 논문은 딥 도메인 적응 기술의 현재 상태를 포괄적으로 분석하며 중요한 통찰을 제공합니다. 딥러닝과 도메인 적응의 결합은 레이블 데이터 부족 문제를 해결하는 강력한 도구임을 입증했습니다. 특히, 도메인 불일치를 줄이기 위한 손실 함수 기반 접근 방식, 도메인 불변 표현 학습을 위한 적대적 학습, 그리고 데이터 특성을 보존하며 전이하는 재구성 기반 접근 방식들이 핵심적인 발전을 이끌었습니다.

그러나 여전히 다음과 같은 한계점과 미래 연구 방향이 존재합니다.

- **이종 Deep DA의 부족**: 대부분의 기존 Deep DA 알고리즘은 소스 및 타겟 도메인의 특징 공간이 동일하다는 동종 DA 가정을 기반으로 합니다. 하지만 실제 응용에서는 이미지와 텍스트, RGB와 깊이(depth) 등 이질적인 데이터 간의 전이가 필요합니다. 이종 Deep DA는 앞으로 더 많은 관심이 필요한 분야입니다.
- **분류/인식 외의 응용 확대**: 이미지 분류 및 인식을 넘어선 객체 탐지, 시맨틱 분할, 얼굴 인식, 사람 재식별 등 복잡한 작업에서 레이블 데이터 없이 또는 극히 제한된 데이터로 Deep DA를 어떻게 효과적으로 적용할지는 주요 과제입니다.
- **부분 도메인 적응 및 오픈셋 DA**: 기존 Deep DA는 대개 소스 및 타겟 도메인이 공유된 레이블 공간을 가진다고 가정합니다. 그러나 실제 시나리오에서는 두 도메인의 클래스 집합이 다르거나 일부 클래스만 공유될 수 있습니다. 부분 전이 학습(partial transfer learning)이나 오픈셋 도메인 적응(open set domain adaptation)과 같은 시나리오에 대한 연구가 더욱 중요해질 것입니다.

## 📌 TL;DR

이 논문은 레이블 데이터 부족과 도메인 시프트 문제를 해결하기 위한 **딥 도메인 적응(Deep DA)**에 대한 포괄적인 조사를 제공합니다. 주요 문제점은 소스-타겟 도메인 간의 분포 차이로 인한 딥러닝 모델의 성능 저하입니다. 논문은 Deep DA 시나리오를 동종/이종, 지도/비지도, 단일/다단계로 분류하고, 학습 손실에 따라 **불일치 기반, 적대적 기반, 재구성 기반**의 세 가지 핵심 접근 방식을 상세히 분석합니다. 또한, 다양한 컴퓨터 비전 응용 분야에서의 성공 사례를 제시하며, 이종 DA, 복잡한 작업으로의 확장, 부분/오픈셋 DA 연구의 필요성 등 미래 방향을 제시합니다.
