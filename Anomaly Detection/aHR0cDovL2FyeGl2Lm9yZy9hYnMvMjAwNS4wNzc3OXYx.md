# Transformation Based Deep Anomaly Detection in Astronomical Images

Esteban Reyes, Pablo A. Estévez (2020)

## 🧩 Problem to Solve

본 논문은 천문학 이미지 데이터셋에서 발생하는 'Bogus' alert(가짜 경보)를 자동으로 탐지하는 문제를 해결하고자 한다. Zwicky Transient Facility(ZTF)나 Large Synoptic Survey Telescope(LSST)와 같은 최신 망원경 시스템은 매일 수백만 건의 천문학적 이벤트 경보를 생성한다. 그러나 이러한 경보 중 상당수는 실제 천문 현상이 아니라 이미지 정렬 불량, 배경 변동, CCD 픽셀 결함 등으로 인한 'Artifact(아티팩트)'이다.

기존의 방식은 전문가가 직접 데이터를 레이블링하는 지도 학습(Supervised Learning) 방식에 의존했으나, 이는 막대한 시간과 노력이 소요될 뿐만 아니라 새로운 형태의 아티팩트가 지속적으로 등장한다는 한계가 있다. 따라서 본 연구의 목표는 **Bogus 클래스의 데이터를 학습에 사용하지 않고, 정상 데이터(Inliers, 즉 실제 천문 객체)만을 이용하여 아티팩트를 찾아내는 One-class Anomaly Detection(일류 이상치 탐지)** 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 기존의 기하학적 변환 기반 이상치 탐지 모델인 GeoTransform을 천문학 이미지 특성에 맞게 개선한 것에 있다.

1.  **천문학 특화 필터 변환 도입**: 천문학 아티팩트가 주로 차분 이미지(Difference Image)에서 배경과 대비되는 날카로운 경계(Sharp edges)를 가진다는 점에 착안하여, Laplacian 필터(경계 강조)와 Gaussian 필터(경계 흐림) 변환을 도입함으로써 아티팩트의 특징을 더욱 명확히 구분할 수 있게 하였다.
2.  **변환 선택 전략(Transformation Selection Strategy) 제안**: 모든 변환을 사용하는 대신, 신경망 분류기를 통해 서로 구분이 불가능한 변환 쌍을 찾아내어 중복된 변환을 제거하는 전략을 제안하였다. 이를 통해 연산 복잡도를 낮추고 분류 공간의 차원을 효율적으로 축소하였다.

## 📎 Related Works

### 관련 연구 및 한계
- **전통적 이상치 탐지**: Isolation Forest(IF)나 One-Class SVM(OC-SVM) 등이 있으나, 이들은 주로 저차원 특징 공간에서 잘 작동하며 이미지와 같은 고차원 매니폴드 데이터에서는 성능이 떨어진다.
- **딥러닝 기반 방식**: AutoEncoder(AE)의 재구성 오차를 이용하거나, GAN을 통한 데이터 분포 학습 방식이 제안되었으나 천문학 데이터의 특수성을 완전히 반영하기는 어렵다.
- **GeoTransform**: 정상 이미지에 다양한 기하학적 변환을 가해 self-labeled 데이터셋을 만들고, 어떤 변환이 적용되었는지 맞추는 분류기를 학습시키는 방식이다. 정상 데이터는 변환을 잘 맞추지만, 이상치는 변환 분류에 실패한다는 점을 이용한다.

### 차별점
본 연구는 GeoTransform의 기본 프레임워크를 유지하되, 단순히 일반적인 기하학적 변환(회전, 이동 등)에 그치지 않고 천문학적 도메인 지식을 활용한 필터 기반 변환을 추가하고, 데이터셋의 불변성(Invariance)을 분석하여 최적의 변환 집합만을 선택한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. GeoTransform 기본 구조
GeoTransform은 정상 샘플 $x$에 $k$개의 변환 $T = \{T_0, T_1, \dots, T_{k-1}\}$을 적용하여 self-labeled 데이터셋 $S_T$를 생성한다. 이후 $k$-class 분류기 $f_\theta$를 학습시켜 적용된 변환의 인덱스를 예측하게 한다.

테스트 단계에서는 샘플 $x$에 모든 변환 $T_i$를 적용하고, 분류기의 출력 소프트맥스 벡터 $y(T_i(x))$의 로그 가능도(log-likelihood)를 합산하여 Normality Score $n_S(x)$를 계산한다.

$$n_S(x) = \frac{1}{k} \sum_{i=0}^{k-1} \log p(y(T_i(x)) | T = T_i)$$

여기서 각 조건부 분포는 Dirichlet 분포 $\text{Dir}(\alpha_i)$를 따른다고 가정하며, 최종 스코어 식은 다음과 같다.

$$n_S(x) = \frac{1}{k} \sum_{i=0}^{k-1} (\tilde{\alpha}_i - 1) \cdot \log y(T_i(x))$$

스코어 값이 더 부정적(negative)일수록 해당 샘플은 더 이상치(anomalous)일 가능성이 높다.

### 2. 제안하는 개선 사항
#### A. 새로운 필터 기반 변환
천문학 아티팩트의 날카로운 경계를 포착하기 위해 다음 필터를 도입하였다.
- **Laplacian Filter**: 에지를 검출하여 아티팩트의 특성을 강조한다.
- **Gaussian Filter**: 에지를 흐리게(blurring) 하여 정상 객체와의 차이를 부각시킨다.
- 이 필터들을 기존의 Shift 연산과 결합하여 $\text{GeoTransform}_{99}$ 등의 변형 모델을 구성하였다.

#### B. 변환 선택 전략 (Transformation Selection)
데이터셋이 특정 변환(예: 회전)에 대해 불변성(Invariance)을 가진다면, 해당 변환은 분류기 입장에서 중복 정보가 된다. 이를 제거하기 위해 다음 절차를 수행한다.
1. 변환 쌍 $(T_i, T_j)$로 구성된 이진 분류 데이터셋을 생성한다.
2. 신경망 분류기를 통해 두 변환을 구분하는 정확도를 측정한다.
3. 정확도가 $50\%$ 근처($49\% \sim 51\%$)라면 두 변환은 구분이 불가능한 것으로 간주하고, 연산량이 더 적은 변환 하나만 남기고 나머지는 제거한다.

### 3. 시스템 아키텍처 및 학습 절차
- **모델**: Wide Residual Network(WRN)를 사용하였으며, depth 10, widen factor 4의 설정을 가진다.
- **손실 함수**: Cross-entropy loss를 사용한다.
- **학습 최적화**: ADAM optimizer를 사용하며, 과적합을 방지하기 위해 검증 데이터셋을 활용한 Early-stopping(patience 0)을 적용하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - **HiTS**: 21x21 픽셀 이미지 4장(template, science, difference, SNR difference). 훈련셋 7,000개(정상만), 테스트셋 4,000개(정상 2,000, 이상치 2,000).
    - **ZTF**: 21x21 픽셀로 크롭된 이미지 3장. 훈련셋 7,000개(정상만), 테스트셋 6,000개(정상 3,000, 이상치 3,000).
- **평가 지표**: AUROC (Area Under the ROC Curve) 및 Accuracy.

### 주요 결과
- **필터 변환의 효과**: Gaussian 및 Laplacian 필터를 추가한 $\text{GeoTransform}_{99}$ 모델이 기본 $\text{GeoTransform}_{72}$보다 성능이 향상되었다.
- **변환 선택의 효과**: 변환 선택 전략을 적용하여 차원을 축소한 모델($\text{GeoTransform}_{35}$ for HiTS, $\text{GeoTransform}_{29}$ for ZTF)이 AUROC 관점에서 가장 높은 성능을 기록하였다.
- **정량적 성능**:
    - **HiTS**: 최적 모델의 AUROC **99.20%** 달성.
    - **ZTF**: 최적 모델의 AUROC **91.39%** 달성.

### 베이스라인 비교
제안 방법은 RAW-OC-SVM, CAE-OC-SVM, Isolation Forest, DSEBM, ADGAN, MO-GAAL 등 기존의 One-class anomaly detection 방법론들보다 통계적으로 유의미하게 높은 성능을 보였다. 특히 원본 GeoTransform 대비 성능 향상이 뚜렷하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 일반적인 딥러닝 모델에 도메인 지식(천문학 아티팩트의 에지 특성)을 결합했을 때 성능이 크게 향상됨을 입증하였다. 특히 변환 선택 전략은 단순히 연산량을 줄이는 것을 넘어, 데이터셋이 가진 기하학적 불변성을 분석하여 불필요한 노이즈(중복 변환)를 제거함으로써 모델의 일반화 성능을 높이는 효과를 가져왔다.

### 한계 및 향후 과제
- **이론적 근거 부족**: 변환 기반 방법론이 왜 이미지 데이터에서 잘 작동하는지에 대한 명확한 수학적/이론적 분석이 부족하다.
- **변환의 고정성**: 현재는 사람이 설계한 필터와 변환을 사용하고 있다. 향후 연구에서는 데이터로부터 유용한 변환을 스스로 학습하는(Learning useful transformations) 방식에 대한 탐구가 필요하다.

## 📌 TL;DR

본 논문은 천문학 이미지의 아티팩트(Bogus)를 탐지하기 위해 정상 데이터만으로 학습하는 **One-class Anomaly Detection** 모델을 제안하였다. 기존 GeoTransform 모델에 **Laplacian/Gaussian 필터 기반 변환**을 추가하여 도메인 특성을 반영하고, **변환 선택 전략**을 통해 중복 변환을 제거함으로써 효율성과 정확도를 동시에 높였다. 실험 결과 HiTS 데이터셋에서 AUROC 99.20%, ZTF 데이터셋에서 91.39%라는 SOTA 성능을 달성하였으며, 이는 전문가의 수동 레이블링 없이도 효과적인 아티팩트 필터링이 가능함을 시사한다.