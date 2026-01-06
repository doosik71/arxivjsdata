# Domain Adaptation for Visual Applications: A Comprehensive Survey

Gabriela Csurka

## 🧩 Problem to Solve

- 시각 애플리케이션에서 훈련(소스) 데이터와 테스트(타겟) 데이터 간의 분포 차이(도메인 시프트)로 인해 모델 성능이 저하되는 문제.
- 대규모 레이블링된 데이터셋을 획득하는 높은 비용 문제. 특히 새로운 환경이나 조건(예: 배경, 위치, 포즈, 이미지 유형 변경)에서 모델을 재배포할 때 발생.
- 기존 기계 학습 모델은 훈련 및 테스트 데이터가 동일한 분포에서 추출되었다고 가정하는데, 이 가정이 깨질 때 발생하는 성능 저하를 극복해야 함.

## ✨ Key Contributions

- 시각 애플리케이션을 위한 도메인 적응(DA) 및 전이 학습(TL)에 대한 포괄적인 개요를 제공.
- DA를 전이 학습 문제의 큰 틀 안에서 정의하고, 다양한 시나리오(동질적/이질적 DA)에 대한 최첨단 방법들을 분석.
- 얕은(shallow) 방법부터 딥 컨볼루션 아키텍처를 통합한 최신 딥(deep) DA 방법에 이르기까지 DA 방법론을 체계적으로 분류 및 설명.
- 이미지 분류를 넘어 객체 탐지, 이미지 분할, 비디오 분석 등 다른 컴퓨터 비전 문제에 대한 DA 솔루션들을 검토.
- 도메인 적응과 다른 기계 학습 솔루션(예: 준지도 학습, 능동 학습, 멀티태스크 학습) 간의 관계를 명확히 함.

## 📎 Related Works

- **전이 학습(Transfer Learning) 및 도메인 적응(Domain Adaptation) 정의:** Pan과 Yang의 전이 학습 서베이 [37, 36]를 기반으로 개념을 정립.
- **얕은 DA 방법:**
  - **인스턴스 재가중치:** Kullback-Leibler Importance Estimation Procedure [43, 44], Maximum Mean Discrepancy (MMD) [47], TrAdaBoost [49].
  - **파라미터 적응:** Transductive SVM [53], Adaptive SVM (A-SVM) [54], Domain Transfer SVM [55], Adaptive Multiple Kernel Learning (A-MKL) [23].
  - **특징 증강:** Geodesic Flow Sampling (GFS) [61, 62], Geodesic Flow Kernel (GFK) [18, 63].
  - **특징 공간 정렬:** Subspace Alignment (SA) [19], Correlation Alignment (CORAL) [21].
  - **비지도/지도 특징 변환:** Transfer Component Analysis (TCA) [14], Domain Invariant Projection (DIP) [65], Statistically Invariant Embedding (SIE) [66], Transfer Sparse Coding (TSC) [68], Transfer Joint Matching (TJM) [40], Marginalized Denoising Autoencoder (MDA) [12], Joint Distribution Adaptation (JDA) [20].
  - **거리 학습:** Information-Theoretic Metric Learning (ITML) [75], Domain Specific Class Means (DSCM) [77], Naive Bayes Nearest Neighbor based Domain Adaptation (NBNN-DA) [78].
- **딥 DA 방법:**
  - **심층 특징 활용:** DeCAF [123] (AlexNet [124], VGGNET [125] 등).
  - **심층 DA 아키텍처:** Stacked Denoising Autoencoders [137], Deep Domain Confusion (DDC) [142], Deep Adaptation Network (DAN) [143], Joint Adaptation Networks (JAN) [145], Deep CORAL [144], Domain-Adversarial Neural Network (DANN) [147], Coupled Generative Adversarial Networks [150], Domain Separation Networks (DSN) [153].
- **합성 데이터 기반 적응:** SYNTHIA [176], Virtual KITTI [177], GTA-V [178].
- **객체 탐지 DA:** PMT-SVM [195], SA-SSVM [212].
- **도메인 일반화(Domain Generalization), 멀티태스크 학습(Multi-task Learning), 퓨샷 학습(Few-shot Learning):** [221, 222], [225, 226], [228, 229] 등.

## 🛠️ Methodology

본 논문은 DA를 위한 새로운 방법론을 제안하기보다는, 기존의 다양한 DA 방법론들을 포괄적으로 분류하고 설명한다.

1. **전이 학습 및 도메인 적응 정의 (Section 2):**

   - **도메인($D$):** $d$차원 특징 공간 $X \subset \mathbb{R}^{d}$와 주변 확률 분포 $P(X)$로 구성.
   - **태스크($T$):** 레이블 공간 $Y$와 조건부 확률 분포 $P(Y|X)$로 정의.
   - **전이 학습(TL):** 소스 도메인 $D_s$ 및 태스크 $T_s$의 정보를 활용하여 타겟 도메인 $D_t$의 $P(Y_t|X_t)$를 학습하는 과정.
   - **도메인 적응(DA):** 전이 학습의 특정 사례로, 태스크는 동일($T_t=T_s$)하지만 도메인 분포는 다른($D_t \neq D_s$) 상황에서 소스 도메인의 레이블된 데이터를 타겟 도메인의 레이블 없는 데이터에 활용하여 분류기를 학습.

2. **얕은 도메인 적응 방법 (Shallow DA Methods) (Section 3):**

   - **동질적 DA (Homogeneous DA):** 특징 공간은 동일($X_t=X_s$)하지만 분포는 다름($P(X_t) \neq P(X_s)$).
     - **인스턴스 재가중치 (Instance Re-weighting):** 소스 샘플에 가중치를 부여하여 타겟 분포에 맞춤. (예: TrAdaBoost, MMD 기반 가중치).
     - **파라미터 적응 (Parameter Adaptation):** 소스에서 훈련된 분류기(예: SVM)의 파라미터를 타겟 도메인에 맞게 조정. (예: A-SVM).
     - **특징 증강 (Feature Augmentation):** 원본 특징에 도메인별 정보를 추가하여 새로운 특징 공간 생성. (예: GFS, GFK).
     - **특징 공간 정렬 (Feature Space Alignment):** 소스와 타겟 특징 공간을 공통 부분 공간으로 정렬. (예: SA, CORAL).
     - **특징 변환 (Feature Transformation):** 데이터가 잠재 공간으로 투영되어 도메인 간의 불일치를 줄이도록 학습. (예: TCA, MDA, JDA). 비지도 및 지도 방식이 있음.
     - **거리 학습 (Metric Learning):** 소스와 타겟 도메인 간의 관련성을 연결하기 위해 거리 측정 학습. (예: DSCM, NBNN-DA).
     - **지역 특징 변환 (Local Feature Transformation):** 전역 변환 대신 샘플 기반의 지역 변환을 적용. (예: ATTM, Optimal Transport for DA).
     - **랜드마크 선택 (Landmark Selection):** 적응 모델 훈련에 가장 관련성 높은 소스 인스턴스를 선택.
   - **다중 소스 DA (Multi-source DA):** 여러 소스 도메인을 효율적으로 활용하는 방법. (예: FA, A-SVM 확장, 소스 도메인 가중치 부여).
   - **이질적 DA (Heterogeneous DA):** 소스와 타겟 도메인의 특징 표현 공간이 다름($X_t \neq X_s$).
     - **보조 도메인 활용:** 중간 다중 뷰 데이터셋을 통해 도메인 간의 격차 해소.
     - **대칭적/비대칭적 특징 변환:** 공통 잠재 공간으로의 투영(대칭적) 또는 소스에서 타겟 공간으로의 직접 변환(비대칭적). (예: HFA, SDDL).

3. **심층 도메인 적응 방법 (Deep DA Methods) (Section 4):**

   - **심층 특징을 이용한 얕은 방법:** 미리 훈련된 CNN에서 추출한 특징(DeCAF)을 기존 얕은 DA 방법에 적용.
   - **심층 CNN 미세 조정 (Fine-tuning):** 소스 도메인에서 사전 훈련된 딥 네트워크를 타겟 도메인의 소량의 레이블 데이터를 사용하여 미세 조정.
   - **심층 DA 아키텍처 (DeepDA Architectures):** DA를 위해 특별히 설계된 딥러닝 모델.
     - **잡음 제거 오토인코더 (Denoising Autoencoders):** Stacked Denoising Autoencoders, Marginalized Denoising Autoencoders.
     - **불일치 기반 (Discrepancy-based) 방법:** MMD 또는 CORAL 손실을 사용하여 두 도메인 간의 특징 분포 차이를 최소화. (예: DDC, DAN, JAN, Deep CORAL).
     - **적대적 판별 모델 (Adversarial Discriminative Models):** 도메인 판별기를 사용하여 도메인 혼동을 유도하고, 도메인 불변 특징을 학습. (예: DANN, ADDA).
     - **적대적 생성 모델 (Adversarial Generative Models):** GAN을 활용하여 소스 이미지를 타겟 도메인처럼 보이게 생성하거나, 도메인 간의 공동 분포를 학습. (예: Coupled GANs).
     - **데이터 재구성 기반 (Data Reconstruction-based) 방법:** 인코더-디코더 구조를 통해 공유 및 도메인 고유 특징을 학습하고 입력 재구성. (예: DRCN, DSN).
     - **이질적 DeepDA:** 멀티모달 데이터(예: Transfer Neural Trees, Weakly-shared DTN)를 위한 딥러닝 적응.

4. **이미지 분류를 넘어선 응용 (Beyond Image Classification) (Section 5):**

   - **일반적인 접근:** 복잡한 비전 문제를 벡터 특징 분류 문제로 재작성하여 얕은 DA 방법 적용.
   - **합성 데이터 기반 적응:** 3D CAD 모델이나 게임 엔진으로 생성된 합성 데이터(소스)를 활용하여 실제 데이터(타겟)에 모델 적응.
   - **객체 탐지 (Object Detection):** HOG 기반 탐지기, 딥러닝 기반 탐지기(예: RCNN)에 DA 적용. 온라인 적응 및 다중 객체 추적 포함.
   - **다른 작업:** 이미지 분할, 비디오 이벤트/액션 인식, 3D 포즈 추정 등.

5. **다른 ML 방법과의 관계 (Beyond Domain Adaptation) (Section 6):**
   - **전이 학습과의 관계:** DA는 전이 학습의 특정 형태(Transductive TL)이며, 도메인 일반화, 멀티태스크 학습, 퓨샷 학습, 제로샷 학습 등과 밀접한 관련이 있음.
   - **전통적인 ML 방법과의 관계:** 준지도 학습, 능동 학습, 온라인 학습, 거리 학습, 분류기 앙상블 등이 DA 솔루션에 활용되거나 확장됨.
   - **이질적 DA와 멀티뷰/멀티모달 학습:** HDA는 멀티뷰 학습과 유사하지만, 훈련 시 모든 뷰가 제공되지 않는다는 차이가 있으며, 웹 기반 보조 데이터를 활용하는 경우가 많음.

## 📊 Results

본 논문은 서베이 논문으로, 새로운 실험 결과를 제시하지는 않는다. 대신 기존 연구 결과를 종합하여 다음을 강조한다:

- **심층 특징의 우수성:** 딥 컨볼루션 네트워크에서 추출된 특징(예: DeCAF)은 도메인 적응 기법을 적용하지 않더라도 기존의 얕은 특징(예: SURFBOV)을 사용한 도메인 적응보다 훨씬 뛰어난 성능을 보인다. 이는 딥러닝 모델이 더 추상적이고 견고한 표현을 학습하여 도메인 편향을 어느 정도 완화하기 때문으로 분석된다.
- **DeepDA의 추가 개선:** 딥 특징을 활용한 얕은 DA 방법이나 딥 DA 아키텍처는 대부분 기본 분류기보다 분류 정확도를 더욱 향상시킨다.
- **평가 및 비교의 어려움:**
  1. 많은 DA 방법들이 벤치마크 데이터셋(Office31, OC10)에서 테스트되었지만, 각 논문마다 실험 프로토콜(소스 샘플링, 데이터 활용, 파라미터 튜닝 전략 등)이 상이하여 공정한 비교가 어렵다.
  2. 동일한 방법(예: GFK, TCA, SA)이라도 재구현(re-implementation)에 따라 결과가 크게 달라지는 경우가 많다.
  3. 현재의 벤치마크 데이터셋은 규모가 작고, 최신 딥 모델 및 DeepDA 아키텍처는 이러한 데이터셋에서 적응 없이도 매우 높은 성능을 보여 일반적인 결론을 도출하기 어렵다.
- **새로운 도전 과제:** 이미지 분류를 넘어 객체 탐지, 비디오 이해, 의미론적 분할, 3D 장면 이해와 같은 복잡한 비전 문제에 대한 DA 솔루션은 여전히 부족하며, 이를 위한 대규모 도전적인 데이터셋의 필요성이 크다.

## 🧠 Insights & Discussion

- **딥러닝의 역할:** 딥 컨볼루션 아키텍처는 도메인 불변 특징을 학습하는 데 강력한 능력을 보여주며, 이는 도메인 적응 문제 해결에 중요한 진전을 가져왔다. 딥 특징은 도메인 시프트를 완화하는 데 탁월하며, 딥 DA 아키텍처는 이를 더욱 최적화한다.
- **합성 데이터의 잠재력:** 컴퓨터 그래픽스 기술의 발전으로 생성된 현실적인 합성 데이터(예: SYNTHIA, Virtual KITTI)는 레이블링 비용 문제를 해결하고, 특히 대규모 레이블 데이터가 필요한 딥러닝 모델 훈련에 큰 잠재력을 제공한다. 이를 실제 데이터에 적응시키는 DA 기법의 중요성이 커지고 있다.
- **연구 방향의 확장:** 기존 DA 연구의 대부분은 이미지 분류에 집중되어 왔지만, 미래에는 객체 탐지, 시맨틱 분할, 3D 이해, 액션 인식 등과 같은 더욱 복잡한 시각 작업으로의 확장이 필요하다. 이러한 작업은 벡터 표현으로 변환하기 어렵거나, 픽셀 단위의 정확도 등 추가적인 요구사항을 가지므로 새로운 DA 접근 방식이 필요하다.
- **평가 프로토콜의 표준화 필요성:** DA 방법론의 공정한 비교 및 발전을 위해서는 일관된 실험 프로토콜과 더 크고 도전적인 벤치마크 데이터셋의 개발 및 활용이 시급하다. 기존 데이터셋은 이미 딥러닝 모델에 의해 포화 상태에 이르렀다.
- **DA의 광범위한 연관성:** DA는 전이 학습, 준지도 학습, 능동 학습, 멀티태스크 학습 등 다양한 기계 학습 분야와 개념적, 방법론적으로 깊이 연관되어 있으며, 이러한 연결성을 이해하는 것이 혁신적인 DA 솔루션 개발에 중요하다.

## 📌 TL;DR

- **문제:** 시각 애플리케이션에서 도메인 간 데이터 분포 차이(도메인 시프트)와 데이터 레이블링의 고비용 문제로 인해 모델 성능이 저하됩니다.
- **방법:** 본 논문은 이 문제를 해결하기 위한 다양한 도메인 적응(DA) 방법을 얕은 방식(인스턴스 재가중치, 특징 변환 등)과 딥러닝 기반 방식(불일치 기반, 적대적 학습, 재구성 기반 네트워크)으로 나누어 포괄적으로 서술하고, 동질적/이질적 시나리오 및 이미지 분류 외의 응용 분야를 다룹니다.
- **핵심 발견:** 딥러닝 특징은 도메인 시프트에 강력하며, 딥 DA 방법들은 성능을 더욱 향상시키지만, 방법론 간의 공정한 비교를 위한 표준화된 프로토콜과 더 도전적인 대규모 데이터셋 개발이 향후 연구의 주요 과제입니다.
