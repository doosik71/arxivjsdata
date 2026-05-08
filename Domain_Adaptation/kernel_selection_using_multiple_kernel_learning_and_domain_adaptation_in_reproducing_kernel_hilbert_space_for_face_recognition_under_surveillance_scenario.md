# Kernel Selection using Multiple Kernel Learning and Domain Adaptation in Reproducing Kernel Hilbert Space, for Face Recognition under Surveillance Scenario

Samik Banerjee and Sukhendu Das (2016)

## 🧩 Problem to Solve

본 논문은 감시 카메라(surveillance cameras) 환경에서 획득한 얼굴 이미지의 낮은 해상도(low-resolution)와 낮은 대비(low-contrast)로 인해 발생하는 얼굴 인식(Face Recognition, FR)의 성능 저하 문제를 해결하고자 한다. 일반적으로 얼굴 인식 시스템은 통제된 환경에서 촬영된 고해상도 이미지(gallery)로 학습되지만, 실제 테스트 단계에서 사용되는 감시 영상의 이미지(probe)는 블러(blur), 노이즈, 조명 변화 등 심각한 품질 저하를 겪는다.

이러한 소스 도메인(gallery)과 타겟 도메인(probe) 사이의 분포 차이(domain gap)는 분류기의 성능을 크게 떨어뜨린다. 따라서 본 연구의 목표는 **Multi-Feature Kernel Learning (MFKL)**을 통한 최적의 특징-커널 쌍(feature-kernel pairing) 선택과 **Reproducing Kernel Hilbert Space (RKHS)** 상에서의 **Domain Adaptation (DA)**을 결합하여, 비제약 환경에서도 강건한 얼굴 인식 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 서로 다른 특징 추출 방법과 커널 함수 간의 최적 조합을 학습하고, 이를 RKHS로 확장하여 도메인 간의 분포 차이를 보정하는 것이다.

1. **MFKL (Multi-Feature Kernel Learning) 제안**: 다양한 얼굴 특징(feature)과 다양한 커널(kernel)의 조합 중 최적의 쌍을 선택하는 기법을 제안하였다. 이는 단순한 교차 검증이 아니라 MKL 프레임워크를 통해 가중치를 학습함으로써 최적의 조합을 찾아낸다.
2. **RKHS 기반의 Unsupervised Domain Adaptation**: 소스 도메인의 고유 벡터(eigen-vectors)를 타겟 도메인과 일치하도록 변환하는 Eigen-domain transformation을 RKHS로 확장하여, 비선형적인 도메인 적응을 가능하게 하였다.
3. **통합 전처리 파이프라인**: Face Hallucination(초해상도 복원), Gallery Degradation(의도적 품질 저하), Power Law Transformation(대비 향상)을 통해 소스와 타겟 간의 간극을 물리적으로 먼저 좁히는 전략을 취하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 본 연구의 차별점을 제시한다.

- **Face Detection**: Viola-Jones 알고리즘과 최근의 Chehra face tracker 등을 언급하며, 본 연구에서는 정교한 얼굴 정렬을 위해 Chehra를 사용한다.
- **Multiple Kernel Learning (MKL)**: Bach et al.의 SMO 기법과 sparse representation 기반의 MKL을 언급한다. 본 논문은 Structured MKL의 변형을 사용하여 특징과 커널의 최적 조합을 찾는 MFKL을 제안함으로써 기존의 단일 커널 사용 방식과 차별화한다.
- **Domain Adaptation (DA)**: Transfer Component Analysis (TCA)나 Subspace Alignment 등을 언급한다. 기존의 선형 변환 방식에서 벗어나 RKHS를 이용한 비선형 변환을 적용함으로써 더 복잡한 도메인 변화에 대응한다.
- **Low-resolution FR**: Super-resolution 및 Multidimensional Scaling (MDS) 기반의 접근법들이 있었으나, 본 연구는 이를 전처리 단계에 통합하고 MKL과 DA를 결합한 전체 파이프라인을 통해 성능을 극대화한다.

## 🛠️ Methodology

### 1. 전처리 단계 (Pre-processing)

이미지 품질 차이를 줄이기 위해 다음 과정을 거친다.

- **Robust Face Detection**: Chehra face tracker를 사용하여 얼굴 영역을 정교하게 크롭한다.
- **Face Hallucination**: Solo dictionary learning 기반 기법을 통해 probe 이미지의 해상도를 높인다.
- **Gallery Degradation**: gallery 이미지를 다운샘플링하고, KL-Divergence를 통해 추정된 최적의 $\sigma$ 값으로 Gaussian blur를 적용하여 probe 이미지와 유사한 수준으로 품질을 낮춘다.
- **Power Law Transformation**: $\gamma=1.25$인 파워 로 변환을 통해 probe 이미지의 대비를 향상시킨다.
    $$P(i,j) = k \cdot C(i,j)^\gamma$$

### 2. 특징 추출 (Feature Extraction)

총 8가지의 다양한 특징 추출 방법을 사용한다: LBP, Eigen Faces, Fisher Faces, Gabor faces, Weber Faces, Bag of Words (BOW), FV-SIFT, VLAD-SIFT.

### 3. MFKL을 통한 커널 선택

각 특징 $F_i$에 대해 최적의 커널 $K_i$를 선택한다. 사용되는 커널은 Linear, Polynomial, Gaussian, RBF, Chi-square, RBF+Chi-square이다.

- **목적 함수**: 각 특징 공간 $X_f$에서 다음의 볼록 최적화(convex optimization) 문제를 푼다.
    $$\min_{\alpha} \max_{j} \left( \frac{1}{2d_j^2} \alpha^T D(y) K_{fj} D(y) \alpha - \alpha^T e \right)$$
    여기서 $K_{fj}$는 $f$번째 특징에 대한 $j$번째 커널의 Gram matrix이다.
- **학습 절차**: Subgradient method를 사용하여 각 특징별 최적 커널을 결정하며, KKT 조건을 통해 가중치가 0에 가까운 불필요한 커널을 제거하고 최적의 특징-커널 쌍 $\langle F_i, K_i \rangle$을 선택한다.

### 4. RKHS 기반 Domain Adaptation

비선형 도메인 변환을 위해 RKHS로 확장된 Eigen-domain transformation을 수행한다.

- **변환 원리**: 소스 도메인 $\Phi(S)$의 주성분(principal components)을 타겟 도메인 $\Phi(T)$와 일치하도록 변환한다.
- **변환된 소스 도메인**:
    $$\Phi(\tilde{S}) = K_{SS} V_{\Phi S} V_{\Phi T}^T \Phi(T)$$
- **Mean Shifting**: 변환 후, 소스 도메인의 평균이 타겟 도메인의 평균과 일치하도록 Gram matrix를 수정한다.
    $$\hat{K}_{\tilde{S}\tilde{S}}(i,j) = K_{\tilde{S}\tilde{S}}(i,j) - K_{\tilde{S}\tilde{S}}(i, \cdot)o_S + K_{\tilde{S}T}(i, \cdot)o_T \dots (\text{상세 식은 원문 식 12 참조})$$

### 5. 분류 (Classification)

최종적으로 수정된 Gram matrix $\hat{K}$를 통해 RKHS 상의 유클리드 거리를 계산한다.
$$\text{dist}(i,j) = \hat{K}(i,i) + \hat{K}(j,j) - 2 \times \hat{K}(i,j)$$
이 거리 행렬을 기반으로 **K-Nearest Neighbor (KNN)** 분류기를 적용하며, 여러 특징에 대해 **Majority Voting**을 통해 최종 클래스를 결정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: SCface, FRSURV, ChokePoint 등 3종의 실제 감시 데이터셋 사용.
- **측정 지표**: Rank-1 Recognition Rate, ROC 곡선, CMC 곡선.
- **비교 대상**: EDA1, COMPDEG, MDS, KDA1, Gopalan, Kliep 및 Naive 조합.

### 정량적 결과 (Rank-1 Accuracy)

| Algorithm | SCface | FRSURV | ChokePoint |
| :--- | :---: | :---: | :---: |
| EDA1 | 47.6 | 57.8 | 54.21 |
| COMPDEG | 4.32 | 43.14 | 62.59 |
| MDS | 42.26 | 12.06 | 52.13 |
| KDA1 | 35.04 | 38.24 | 56.25 |
| Naive | 75.27 | 45.78 | 65.76 |
| **Proposed** | **78.31** | **55.23** | **84.62** |

### 결과 분석

- 제안 방법이 모든 데이터셋에서 기존 최신 기법(state-of-the-art)보다 상당한 차이로 높은 성능을 보였다.
- 특히 ChokePoint 데이터셋에서 가장 높은 정확도를 보였으며, FRSURV 데이터셋에서는 상대적으로 낮은 성능을 보였다. 이는 FRSURV가 야외 환경에서 촬영되어 도메인 복잡도가 매우 높기 때문으로 분석된다.
- Naive 조합(DA 없이 MFKL만 적용)보다 제안 방법(MFKL + DA)의 성능이 월등히 높음을 통해, 커널 선택과 도메인 적응의 결합이 유효함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 감시 환경의 얼굴 인식 문제를 해결하기 위해 전처리 $\rightarrow$ 특징-커널 최적화 $\rightarrow$ 비선형 도메인 적응으로 이어지는 체계적인 파이프라인을 구축하였다.

**강점 및 통찰**:

- **상호 보완적 설계**: MFKL은 입력 데이터의 표현력을 극대화하고, DA는 그 표현력이 타겟 도메인에서도 유효하도록 정렬한다. 두 기법의 결합이 단일 기법 적용보다 훨씬 강력한 성능 향상을 가져온다는 점을 확인하였다.
- **비선형성의 활용**: 단순 선형 DA(EDA1)보다 RKHS 기반의 비선형 DA(KDA1 및 제안 방법)가 더 효과적임을 보여줌으로써, 얼굴 데이터의 복잡한 분포 차이는 비선형 공간에서 더 잘 해결됨을 시사한다.

**한계 및 논의**:

- **데이터셋별 성능 편차**: FRSURV와 같은 야외 데이터셋에서의 정확도가 여전히 낮다는 점은, 현재의 DA 기법만으로는 극심한 환경 변화(조명, 각도 등)를 완전히 극복하기 어렵음을 보여준다.
- **계산 복잡도**: MKL과 RKHS 상의 Gram matrix 연산은 데이터의 크기가 커질 경우 계산 비용이 급격히 증가할 가능성이 있으나, 이에 대한 구체적인 시간 복잡도 분석은 제시되지 않았다.

## 📌 TL;DR

본 연구는 감시 카메라의 저화질 이미지 환경에서 얼굴 인식 성능을 높이기 위해, **최적의 특징-커널 조합을 학습하는 MFKL**과 **RKHS 상에서의 비선형 도메인 적응(DA)**을 결합한 프레임워크를 제안하였다. 실험 결과, 제안 방법은 기존의 도메인 적응 및 초해상도 기반 방법들보다 높은 인식 정확도를 달성하였으며, 이는 특히 물리적 전처리와 수학적 도메인 정렬이 조화를 이룬 결과이다. 향후 야외 환경의 극심한 변동성을 해결하기 위한 더 강력한 변환 함수의 연구가 필요할 것으로 보인다.
