# Domain Adaptation for the Segmentation of Confidential Medical Images

Serban Stan, Mohammad Rostami (2022)

## 🧩 Problem to Solve

본 논문은 서로 다른 모달리티(Modality) 간의 도메인 차이(Domain Shift)로 인해 발생하는 의료 영상 세그멘테이션 모델의 성능 저하 문제를 해결하고자 한다. 예를 들어, MRI 데이터로 학습된 모델을 CT 데이터에 적용할 경우, 두 영상의 물리적 생성 원리가 다르기 때문에 성능이 크게 떨어진다.

일반적으로 이러한 문제를 해결하기 위해 Unsupervised Domain Adaptation (UDA) 기법이 사용되지만, 기존의 대부분의 UDA 알고리즘은 타겟 도메인에 적응하는 과정에서 소스 도메인의 데이터에 직접 접근해야 한다. 하지만 의료 분야에서는 환자의 개인정보 보호 규정과 데이터 기밀 유지(Confidentiality) 문제로 인해 소스 데이터를 타겟 도메인으로 공유하거나 함께 사용하는 것이 사실상 불가능한 경우가 많다.

따라서 본 연구의 목표는 타겟 도메인 적응 단계에서 소스 데이터에 직접 접근하지 않고도 도메인 간의 간극을 줄일 수 있는 Source-Free Unsupervised Domain Adaptation (SF-UDA) 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 소스 도메인의 정보를 잠재 공간(Latent Space) 상의 **내부 분포(Internal Distribution)** 형태로 인코딩하여 저장하고, 이를 소스 데이터를 대체하는 대리자(Surrogate)로 사용하는 것이다.

구체적으로는 소스 데이터의 임베딩 특성을 Gaussian Mixture Model (GMM)으로 모델링하여 저장하며, 이후 타겟 도메인의 데이터를 적응시킬 때 실제 소스 샘플 대신 이 GMM에서 생성된 샘플과 타겟 임베딩 간의 분포를 정렬함으로써 데이터 기밀성을 유지하면서도 성능을 향상시킨다.

## 📎 Related Works

기존의 UDA 접근 방식은 크게 두 가지 전략으로 나뉜다. 첫째는 GAN(Generative Adversarial Networks)을 이용하여 소스와 타겟 데이터가 도메인 불변(Domain-invariant) 특징 공간으로 매핑되도록 하는 적대적 학습 방식이다. 둘째는 공유 임베딩 공간에서 두 도메인의 분포 거리(Probability distance metric)를 직접 최소화하는 방식이다.

이러한 기존 방식들의 공통적인 한계는 손실 함수를 계산하기 위해 소스와 타겟 데이터 모두에 동시에 접근해야 한다는 점이다. 최근 일부 Source-free UDA 연구들이 제안되었으나, 이들은 주로 이미지 분류(Classification) 작업에 집중되어 있어 픽셀 단위의 정밀한 예측이 필요한 의료 영상 세그멘테이션에 적용하기에는 한계가 있다. 또한, 생성 모델을 통해 가짜 데이터를 만드는 방식은 의료 영상의 세부 디테일을 유지하기 어렵고, 생성된 이미지가 실제 환자 데이터와 너무 유사할 경우 여전히 기밀 유지 문제가 발생할 수 있다는 위험이 있다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문에서 제안하는 모델 $f$는 다음과 같은 세 가지 서브 네트워크의 합성 함수로 정의된다: $f = \phi \circ \chi \circ \psi$.

1. **Encoder ($\psi$):** 입력 영상을 잠재 공간으로 매핑한다.
2. **Decoder ($\chi$):** 잠재 표현을 다시 원래의 영상 크기로 업샘플링한다.
3. **Classifier ($\phi$):** 최종적으로 각 픽셀에 대해 클래스 확률 값을 할당한다.

본 연구에서는 $\chi \circ \psi$의 출력 공간을 공유 임베딩 공간으로 설정하고, 이 공간에서의 분포 정렬을 통해 도메인 적응을 수행한다.

### 내부 분포 추정 (GMM)

소스 데이터를 직접 사용할 수 없으므로, 소스 도메인의 임베딩 분포 $P_Z$를 GMM으로 근사한다. $K$개의 세그멘테이션 클래스가 있을 때, 클래스당 $\omega$개의 가우시안 성분을 사용하여 다음과 같이 정의한다.

$$P_Z(z) = \sum_{c=1}^{\omega K} \alpha_c \mathcal{N}(z|\mu_c, \Sigma_c)$$

여기서 $\alpha_c$는 혼합 확률, $\mu_c$는 평균, $\Sigma_c$는 공분산 행렬이다. EM(Expectation-Maximization) 알고리즘을 통해 소스 데이터의 잠재 특징으로부터 이 파라미터들을 추정한다. 이때, 모델의 예측 확신도(Confidence)가 임계값 $\rho$보다 높은 샘플만을 사용하여 이상치(Outlier)를 제거함으로써 더 정밀한 분포를 학습한다.

### 적응 절차 및 손실 함수

학습된 GMM으로부터 의사 데이터셋(Pseudo-dataset) $D^P = (Z^P, Y^P)$를 생성한다. 이후 타겟 데이터 $X^T$를 사용하여 다음의 적응 손실 함수 $L_{adapt}$를 최소화한다.

$$L_{adapt} = L_{ce}(\phi(Z^P), Y^P) + \lambda D(\chi(\psi(X^T)), Z^P)$$

- **첫 번째 항:** $\phi(Z^P)$와 $Y^P$ 사이의 Cross-Entropy 손실이다. 이는 분류기가 GMM 샘플에 대해서도 일반화 성능을 유지하도록 미세 조정(Fine-tuning)하는 역할을 한다.
- **두 번째 항:** 타겟 도메인의 임베딩 분포와 GMM 기반의 의사 임베딩 분포 $Z^P$ 사이의 거리를 최소화한다. 이때 거리 측정 지표 $D(\cdot, \cdot)$로는 계산 효율성이 높은 Sliced Wasserstein Distance (SWD)를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋:**
    1. MMWHS (Multi-Modality Whole Heart Segmentation): MRI(Source) $\to$ CT(Target) 적응.
    2. CHAOS $\to$ Multi-Atlas: Abdominal MR(Source) $\to$ CT(Target) 적응.
- **평가 지표:** Dice coefficient (세그멘테이션 정확도), Average Symmetric Surface Distance (ASSD, 경계면 품질).
- **비교 대상:** PnP-AdaNet, CycleGAN, CyCADA 등 소스 데이터에 접근 가능한 SOTA UDA 방법론(상한선으로 설정)과 AdaEnt, AdaMI 등 기존의 Source-free 방법론.

### 정량적 결과

- **MMWHS 데이터셋:** 제안 방법(SFS)은 AA(Ascending Aorta) 클래스에서 SOTA 성능을 달성했으며, 전체적인 Dice 점수에서 매우 높은 경쟁력을 보였다. 특히 Source-free 방식인 AdaEnt, AdaMI보다 우수한 성능을 기록했다.
- **Abdominal 데이터셋:** Liver 클래스에서 SOTA 성능을 보였으며, 다른 장기들에 대해서도 경쟁력 있는 결과를 나타냈다.
- **분석:** 소스 데이터에 직접 접근하는 방식들이 여전히 약간 더 높은 성능을 보이는 경우가 있으나, 이는 기밀 유지 제약이 없는 상태에서의 상한선임을 감안할 때, SFS는 데이터 보안을 유지하면서도 이에 근접한 성능을 낸다고 볼 수 있다.

### 정성적 결과

- UMAP 시각화 결과, 적응 전에는 타겟 도메인의 임베딩이 서로 겹치거나 분산되어 있었으나, 적응 후에는 GMM이 정의한 소스 도메인의 클래스별 모드(Mode) 방향으로 타겟 임베딩이 이동하며 명확하게 구분되는 것을 확인하였다.
- 시각적 세그멘테이션 맵에서도 Source-only 모델에 비해 Ground Truth에 훨씬 가까운 결과물을 생성함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 소스 데이터 없이 GMM이라는 중간 매개체만을 이용해 도메인 정렬을 수행할 수 있음을 입증하였다.

**주요 분석 포인트:**

- **$\rho$ 파라미터의 영향:** $\rho$ 값을 높여 확신도가 높은 샘플만으로 GMM을 구성했을 때, 잠재 공간에서의 클래스 간 분리도가 향상됨을 확인하였다. 이는 내부 분포의 품질이 적응 성능의 핵심임을 시사한다.
- **$\omega$ (클래스당 가우시안 성분 수)의 영향:** $\omega=1$일 때보다 $\omega \ge 3$일 때 성능이 크게 향상되었다. 이는 단일 가우시안보다 여러 개의 성분을 사용하는 것이 복잡한 소스 분포를 더 잘 표현할 수 있게 하여, 타겟 데이터가 더 정확한 위치로 정렬되도록 돕기 때문이다.
- **분류기 미세 조정:** GMM 샘플을 통해 분류기를 추가 학습시키는 것이 단순 소스 학습 상태를 유지하는 것보다 성능 향상에 기여하였다. 이는 실제 소스 분포와 근사된 GMM 분포 사이의 미세한 차이를 보정해주기 때문이다.

**한계점 및 비판적 해석:**

- 본 연구는 GMM을 통해 소스 분포를 모사하지만, 소스 데이터의 분포가 매우 복잡하거나 비정형적일 경우 단순한 가우시안 혼합 모델만으로는 충분히 표현하지 못할 가능성이 있다.
- 또한, 타겟 도메인의 데이터 양이 극단적으로 적을 경우, GMM으로의 정렬 과정에서 오버피팅이 발생하거나 잘못된 모드로 수렴할 위험이 존재한다.

## 📌 TL;DR

본 논문은 의료 영상 데이터의 기밀성을 보장하기 위해, 적응 단계에서 소스 데이터 없이 학습 가능한 **Source-Free UDA** 알고리즘인 **SFS (Source Free semantic Segmentation)**를 제안한다. 소스 도메인의 잠재 분포를 **GMM(Gaussian Mixture Model)**으로 모델링하고, 타겟 데이터를 이 GMM 분포에 정렬시키는 방식을 통해 데이터 공유 없이도 MRI $\to$ CT와 같은 교차 모달리티 세그멘테이션 성능을 효과적으로 높였다. 이 연구는 의료 데이터 보안과 모델 성능이라는 두 마리 토끼를 잡아야 하는 실제 의료 AI 현장에서 매우 중요한 실용적 가치를 지닌다.
