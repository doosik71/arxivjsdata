# A BRIEF REVIEW OF DOMAIN ADAPTATION
Abolfazl Farahani, Sahar Voghoei, Khaled Rasheed, Hamid R. Arabnia

## 🧩 Problem to Solve
고전적인 기계 학습(Machine Learning)은 훈련 데이터와 테스트 데이터가 동일한 분포에서 추출되었다고 가정합니다. 그러나 실제 응용 분야에서는 데이터를 수집하는 소스가 다르거나 시간이 지남에 따라 데이터의 특성이 변하는 등 다양한 이유로 훈련 데이터와 테스트 데이터의 분포가 달라지는 '도메인 시프트(Domain Shift)'가 자주 발생합니다. 이러한 분포 불일치(Discrepancy)는 학습된 모델의 성능 저하를 초래합니다. 도메인 적응(Domain Adaptation, DA)은 이러한 문제에 대처하기 위해 도메인 간의 불일치를 정렬하여, 특정 도메인에서 학습된 모델이 관심 도메인으로 일반화될 수 있도록 하는 머신러닝 분야입니다. 본 논문은 특히 타겟 도메인에 레이블이 없는 '비지도 도메인 적응(Unsupervised Domain Adaptation)'에 초점을 맞춰 도메인 적응의 다양한 분류와 성공적인 접근 방식들을 개괄적으로 설명합니다.

## ✨ Key Contributions
*   도메인 적응을 다양한 관점에서 체계적으로 분류하고 정의합니다.
*   도메인 시프트의 주요 유형(공변량 시프트, 사전 시프트, 개념 시프트)을 명확히 설명합니다.
*   범주 간의 불일치(Category Gap)에 따른 도메인 적응 시나리오(닫힌 집합, 열린 집합, 부분, 보편적 도메인 적응)를 제시합니다.
*   얕은(Shallow) 및 심층(Deep) 도메인 적응 접근 방식들을 상세히 소개하고, 각 방식의 핵심 아이디어와 대표적인 기술들을 설명합니다.
*   다양한 도메인 적응 기법들을 비교하고, 실세계 문제 해결을 위한 이들의 중요성을 강조합니다.

## 📎 Related Works
*   **전이 학습(Transfer Learning) [1]:** 도메인 적응은 전이 학습의 특별한 경우로, 소스 및 타겟 도메인의 작업(Task)은 동일하지만 도메인(분포)이 다른 시나리오에 해당합니다. 전이 학습은 귀납적(Inductive), 변환적(Transductive), 비지도(Unsupervised) 전이 학습으로 분류됩니다.
*   **준지도 학습(Semi-supervised Learning) [2, 3]:** 레이블이 부족한 문제를 다루지만, 레이블된 데이터와 레이블 없는 데이터가 동일한 분포에서 추출되었다고 가정합니다. 이는 분포가 다를 수 있음을 허용하는 도메인 적응과 대조됩니다.
*   **다중 작업 학습(Multi-task Learning) [4]:** 여러 관련 작업을 동시에 훈련하여 일반화 성능을 향상시키는 것을 목표로 합니다. 전이 학습과 유사하게 지식 공유를 활용하지만, 단일 타겟 학습자의 성능 향상보다는 여러 관련 작업의 동시 개선에 중점을 둡니다.
*   **다중 뷰 학습(Multi-view Learning):** 오디오-비디오, 이미지-텍스트 등 여러 가지 특징 집합을 가진 데이터로부터 학습하는 기술입니다. 상호 보완적인 정보를 활용하여 포괄적인 표현을 학습합니다. 대표적인 기법으로는 CCA(Canonical Correlation Analysis) [14]와 Co-training [15]이 있습니다.
*   **도메인 일반화(Domain Generalization) [17]:** 여러 개의 레이블된 소스 도메인에서 모델을 훈련하여 훈련 시 접근할 수 없는 보이지 않는(Unseen) 타겟 도메인으로 일반화하는 것을 목표로 합니다. 도메인 적응이 훈련 시 타겟 데이터가 필요하다는 점에서 차이가 있습니다.

## 🛠️ Methodology
본 논문은 도메인 적응 기법들을 '도메인 시프트 유형'과 '접근 방식 아키텍처'라는 두 가지 주요 관점에서 분류합니다.

### 도메인 시프트 유형별 분류 (닫힌 집합 도메인 적응 중심)
*   **사전 시프트(Prior Shift):** $p_S(y|x) = p_T(y|x)$이지만 $p_S(y) \neq p_T(y)$인 경우 (클래스 불균형). 양 도메인에 레이블 데이터가 필요합니다.
*   **공변량 시프트(Covariate Shift):** $p_S(x) \neq p_T(x)$이지만 $p_S(y|x) = p_T(y|x)$인 경우 (대부분의 도메인 적응 기법이 목표).
*   **개념 시프트(Concept Shift):** $p_S(x) = p_T(x)$이지만 $p_S(y|x) \neq p_T(y|x)$인 경우 (데이터 드리프트). 양 도메인에 레이블 데이터가 필요합니다.

### 접근 방식 아키텍처별 분류
**1. 얕은(Shallow) 도메인 적응:**
*   **인스턴스 기반 적응 (Instance-Based Adaptation):**
    *   주로 공변량 시프트 문제를 해결하기 위해 소스 도메인 샘플에 타겟/소스 밀도 비율 $w(x) = p_T(x) / p_S(x)$을 이용한 가중치를 부여하여 소스 분포를 타겟 분포에 가깝게 재가중합니다.
    *   **핵심 아이디어:** 재가중된 소스 데이터 분포와 실제 타겟 데이터 분포 간의 불일치를 최소화하는 최적의 가중치를 찾는 밀도 비율 추정(Density Ratio Estimation, DRE).
    *   **대표 기법:**
        *   Kernel Mean Matching (KMM) [25, 40]: 재현 커널 힐베르트 공간(RKHS)에서 최대 평균 불일치(MMD)를 최소화하여 가중치를 직접 추정합니다.
        *   Kullback-Leibler Importance Estimation Procedure (KLIEP) [41]: 타겟 분포와 가중된 소스 분포 간의 KL-발산(KL-divergence)을 최소화하여 가중치를 추정합니다.
*   **특징 기반 적응 (Feature-Based Adaptation):**
    *   도메인 간 불변적인 특징 표현을 학습하여 원본 특징을 새로운 특징 공간으로 변환합니다.
    *   **부분 공간 기반 (Subspace-based):**
        *   소스와 타겟 도메인 간에 공유되는 공통 중간 표현을 발견합니다.
        *   **대표 기법:** Sampling Geodesic Flow (SGF) [26], Geodesic Flow Kernel (GFK) [45], Subspace Alignment (SA) [46] (투영 행렬 $M = \arg\min_{M} \|X_S M - X_T\|_{F}^{2}$ 학습), Subspace Distribution Alignment (SDA) [47].
    *   **변환 기반 (Transformation-based):**
        *   원본 데이터의 내재된 구조를 보존하면서 주변 분포 및 조건부 분포 간의 불일치를 최소화하도록 특징을 변환합니다.
        *   **대표 기법:** Transfer Component Analysis (TCA) [27] (RKHS에서 주변 분포 MMD 최소화), Joint Domain Adaptation (JDA) [48] (주변 분포와 조건부 분포를 동시에 정렬, 의사 레이블 사용).
    *   **재구성 기반 (Reconstruction-based):**
        *   중간 특징 표현에서 샘플 재구성을 통해 도메인 분포 간의 불일치를 줄입니다.
        *   **대표 기법:** Robust Visual Domain Adaptation with Low-rank Reconstruction (RDALR) [28] (소스 샘플이 타겟 샘플에 의해 선형적으로 표현되도록 투영 행렬 $W$ 학습), Low-Rank Transfer Subspace Learning (LTSL) [49].

**2. 심층(Deep) 도메인 적응:**
*   **불일치 기반 (Discrepancy-based):**
    *   심층 신경망 내에서 도메인 간의 특징 분포 불일치를 측정하고 최소화합니다.
    *   **대표 기법:** Deep Adaptation Network (DAN) [33] (여러 적응 계층에서 다중 커널 MMD(MK-MMD)를 사용하여 주변 분포 정렬), Deep Transfer Network (DTN) [58] (주변 분포와 조건부 분포를 동시에 정렬).
*   **재구성 기반 (Reconstruction-based):**
    *   오토인코더(Autoencoder)를 사용하여 재구성 오류를 최소화하고 도메인 불변적이고 전이 가능한 표현을 학습합니다.
    *   **대표 기법:** Stacked Auto Encoders (SDA) [34], Marginalized SDA (mSDA) [77], Deep Reconstruction-Classification Network (DRCN) [73] (인코더-디코더 네트워크를 사용하여 비지도 도메인 적응 수행).
*   **적대적 기반 (Adversarial-based):**
    *   생성적 적대 신경망(GAN) [61]에서 영감을 받아 도메인 분류자와 특징 추출기 간의 적대적 학습을 통해 도메인 불변 특징을 추출합니다.
    *   **핵심 아이디어:** 특징 추출기는 도메인 분류자를 혼란시키려 하고, 도메인 분류자는 도메인을 구별하려 하여, 궁극적으로 도메인을 구별할 수 없는 특징을 학습합니다.
    *   **대표 기법:**
        *   Gradient Reversal Layer (GRL) [35, 36]: 역전파 시 그래디언트를 반전시키는 계층을 삽입하여 특징 추출기가 도메인 불변 특징을 학습하도록 유도합니다.
        *   Multi-Adversarial Domain Adaptation (MADA) [62]: 단일 도메인 분류자가 아닌 다중 클래스별 도메인 분류자를 사용하여 클래스 조건부 분포 불일치도 줄입니다.
        *   Minimax Optimization [63, 36]: 분류 손실, 소프트 레이블 손실, 도메인 분류자 손실, 도메인 혼란 손실 등을 최소화하여 주변 및 조건부 분포를 정렬합니다.
        *   Visual Adversarial Domain Adaptation (PixelDA [69], SimGAN [70] for pixel-level, DAN [72] for feature-level, CyCADA [66] for both): GAN을 활용하여 픽셀 수준 또는 특징 수준에서 도메인 간 스타일을 변환하거나 불변 특징을 학습합니다.

## 📊 Results
*   전통적인 기계 학습 모델은 훈련 및 테스트 데이터 분포 간에 도메인 시프트가 발생할 경우 성능이 크게 저하됩니다.
*   도메인 적응 기법들은 이러한 성능 저하를 완화하기 위해 도메인 간의 분포 불일치를 효과적으로 줄입니다.
*   **얕은 도메인 적응 기법**들은 데이터 재가중, 특징 변환, 부분 공간 학습, 재구성 등을 통해 도메인 간의 차이를 줄이는 데 효과적임을 보여주었지만, 때로는 데이터 규모나 특정 가정(예: 공변량 시프트)에 제약될 수 있습니다.
*   **심층 도메인 적응 기법**들은 신경망의 강력한 특징 추출 능력을 활용하여 복잡한 도메인 시프트 문제, 특히 이미지 분류와 같은 고차원 데이터에서 뛰어난 성능을 보였습니다.
*   특히 **적대적 학습 기반 심층 도메인 적응**은 도메인 불변적인 특징을 학습하는 데 매우 성공적이며, 실제 응용 분야에서 복잡한 분포 변화에 강건함을 입증했습니다.

## 🧠 Insights & Discussion
*   도메인 적응은 훈련-테스트 데이터의 독립 동일 분포(i.i.d.) 가정이 현실 세계에서 종종 충족되지 않는 상황에서 기계 학습 모델의 실제 적용 가능성을 높이는 데 필수적입니다.
*   타겟 도메인의 레이블이 없는 비지도 도메인 적응은 실제 시나리오에서 레이블링 비용과 노력을 줄일 수 있어 특히 중요합니다.
*   도메인 시프트의 본질(예: 공변량 시프트, 사전 시프트, 개념 시프트)과 범주 간의 관계(닫힌 집합, 열린 집합, 부분, 보편적)를 이해하는 것이 적절한 도메인 적응 전략을 선택하는 데 중요합니다.
*   심층 학습의 발전은 도메인 적응 분야에 혁신을 가져왔으며, 특히 적대적 학습은 도메인 불변 특징 추출의 강력한 프레임워크를 제공합니다.
*   향후 연구는 보편적 도메인 적응(Universal Domain Adaptation)과 같은 보다 일반적인 시나리오에서 모델의 강건성을 높이고, 부정적 전이(Negative Transfer)를 효과적으로 방지하며, 이론적 일반화 경계에 대한 더 깊은 이해를 모색하는 방향으로 진행될 것입니다.

## 📌 TL;DR
*   **문제:** 기존 기계 학습 모델은 훈련 데이터와 테스트 데이터의 분포가 다를 때(도메인 시프트) 성능이 저하됩니다.
*   **방법:** 도메인 적응(DA)은 소스 및 타겟 도메인 간의 분포 불일치를 줄여 학습된 모델이 새로운 도메인에 잘 일반화되도록 합니다. 본 논문은 타겟 레이블이 없는 비지도 도메인 적응에 초점을 맞춰, 소스 샘플 재가중, 특징 변환 및 정렬을 사용하는 얕은 적응 기법(인스턴스 기반, 특징 기반)과 심층 신경망을 활용하는 심층 적응 기법(불일치 기반, 재구성 기반, 적대적 기반)들을 포괄적으로 검토합니다.
*   **핵심 발견:** 도메인 적응 기법들은 도메인 불일치를 효과적으로 완화하며, 특히 적대적 학습에 기반한 심층 학습 모델들은 전이 가능한 도메인 불변 특징을 학습하는 데 강력한 성능을 보여줍니다.