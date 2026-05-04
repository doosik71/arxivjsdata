# Instance-Aware Robust Consistency Regularization for Semi-Supervised Nuclei Instance Segmentation

Zenan Lin, Wei Li, Jintao Chen, Zihao Wu, Wenxiong Kang, Changxin Gao, Liansheng Wang, Jin-Gang Yu (2025)

## 🧩 Problem to Solve

본 논문은 병리 이미지 내의 핵 인스턴스 분할(Nuclei Instance Segmentation)에서 발생하는 데이터 어노테이션의 높은 비용과 희소성 문제를 해결하고자 한다. 핵 인스턴스 분할은 종양 미세환경 분석 및 면역 스코어링과 같은 하위 분석 작업에 필수적이지만, 단일 핵을 어노테이션 하는 데 평균 8.43초가 소요되며 전체 슬라이드 이미지(WSI) 하나에는 수십만 개의 핵이 포함되어 있어 대규모 학습 데이터를 확보하는 것이 매우 어렵다.

기존의 준지도 학습(Semi-Supervised Learning, SSL) 방법들은 주로 Teacher-Student 구조의 일관성 규제(Consistency Regularization)를 사용하지만, 이는 전역 맵(Global Map)을 전체적으로 비교하는 'Holistic Consistency' 방식에 의존한다. 이러한 방식은 인스턴스 수준의 세밀한 규제가 부족하여 인스턴스의 병합(Merge)이나 분리(Split) 오류가 발생하기 쉽고, 특히 학습 초기 단계에서 학생 모델의 잘못된 예측이 교사 모델로 전이되어 성능을 저하시키는 노이즈 pseudo-label 문제가 발생한다는 한계가 있다. 따라서 본 논문의 목표는 인스턴스 수준에서 견고한 일관성 규제를 수행하여 적은 양의 라벨링 데이터만으로도 정확한 핵 분할을 달성하는 IRCR-Net을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 일관성 규제를 전역적인 맵 단위가 아닌 개별 인스턴스 단위로 수행하는 **Instance-Aware Robust Consistency Regularization (IRCR)** 개념을 도입하는 것이다. 이를 위해 다음의 두 가지 핵심 메커니즘을 설계하였다.

1. **Matching-Driven Instance-Aware Consistency (MIAC):** 교사 모델과 학생 모델이 예측한 인스턴스들 사이의 이분 매칭(Bipartite Matching)을 통해 서로 일치하는 인스턴스 쌍에 대해서만 일관성 손실을 적용한다. 이를 통해 매칭되지 않는 잘못된 예측(노이즈)이 학습에 반영되는 것을 방지한다.
2. **Prior-Driven Instance-Aware Consistency (PIAC):** 병리 이미지 속 핵이 가지는 형태학적 특성(Morphological Prior)을 사전 지식으로 활용한다. 커널 밀도 추정(Kernel Density Estimation, KDE)을 통해 예측된 인스턴스가 실제 핵일 확률을 계산하고, 확률이 낮은 저품질 pseudo-label은 제거하고 고품질 예측은 강화하여 학습의 견고함을 높인다.

## 📎 Related Works

기존의 준지도 학습 방식 중 Mean-Teacher 프레임워크는 가중치 평균(EMA)을 통해 모델의 견고함을 높이며 의료 영상 분할에서 널리 사용되어 왔다. 또한, Hover-Net과 같은 딥러닝 기반의 핵 분할 모델은 전경 세그멘테이션과 공간 구조 정보(Distance Maps)를 결합하여 개별 핵을 구분하는 성능을 보여주었다.

그러나 기존의 일관성 기반 SSL 방법들은 병리 이미지의 특성인 핵의 밀집도와 겹침(Overlapping) 문제를 충분히 고려하지 않았다. 특히 전역 특징 맵의 일관성만을 강조하는 방식은 개별 인스턴스의 경계를 정밀하게 정렬하거나 제약하지 못하며, 이로 인해 인스턴스 수준의 분할 작업에서는 효율성이 떨어진다는 차별점이 존재한다.

## 🛠️ Methodology

### 전체 시스템 구조

IRCR-Net은 기본적으로 **Mean-Teacher** 구조를 따르며, 베이스 네트워크로는 **modified Hover-Net**을 사용한다. 학생 모델 $\theta_s$는 역전파를 통해 학습되며, 교사 모델 $\theta_t$는 다음과 같은 지수 이동 평균(EMA) 방식으로 업데이트된다.

$$\theta_t^{(k+1)} \leftarrow \alpha \theta_t^{(k)} + (1-\alpha)\theta_s^{(k+1)}$$

여기서 $\alpha = 0.95$이다. 학습 과정에서 라벨링된 데이터는 지도 학습 손실 $\mathcal{L}_{sup}$을 생성하고, 라벨링되지 않은 데이터는 강한 증강(Strong Augmentation)과 약한 증강(Weak Augmentation)을 거쳐 각각 학생과 교사 모델에 입력된 후 일관성 손실 $\mathcal{L}_{cons}$를 생성한다.

### Matching-Driven Instance-Aware Consistency (MIAC)

MIAC는 두 모델이 예측한 인스턴스 세트 $T^{(k)}$(교사)와 $S^{(k)}$(학생)를 정렬한다. 각 인스턴스의 공간적 중심점(Centroid) 간의 유클리드 거리를 기반으로 거리 행렬 $W$를 구성하고, **Munkres 알고리즘**을 사용하여 최적의 일대일 매칭 함수 $\sigma$를 찾는다. 거리 $w_{ij}$는 다음과 같이 계산된다.

$$w_{ij} = \| c(T_i^{(k)}) - c(S_j^{(k)}) \|$$

매칭된 인스턴스들에 대해서만 다음과 같이 특징 맵 $F$와 경계 강조 맵 $B$를 이용한 손실 함수를 정의한다.

$$\mathcal{L}_{MIAC}^{(k+1)} = \frac{1}{N} \sum_{i=1}^{N} \left( \| F_s^{(k+1)} \odot S_{\sigma(i)}^{(k)} - F_t^{(k+1)} \odot T_i^{(k)} \|^2 + \beta \| B_s^{(k+1)} \odot \tilde{S}_{\sigma(i)}^{(k)} - B_t^{(k+1)} \odot \tilde{T}_i^{(k)} \|^2 \right)$$

여기서 $\beta=0.5$이며, $\tilde{S}$와 $\tilde{T}$는 Sobel 연산자와 팽창(Dilation)을 통해 추출된 경계 영역이다.

### Prior-Driven Instance-Aware Consistency (PIAC)

PIAC는 공개 데이터셋에서 추출한 핵의 형태학적 특징(면적, Solidity, Circularity, 강도, Extent)의 통계적 분포를 사전 지식으로 활용한다. 비모수적 방법인 **커널 밀도 추정(KDE)**을 사용하여 특징 $x_1$에 대한 확률 밀도 함수 $p(x_1)$를 다음과 같이 정의한다.

$$p(x_1) = \frac{1}{\sqrt{2\pi}Nh} \sum_{n=1}^{N} \exp \left[ -\frac{(x_1 - x_1^{(n)})^2}{2h^2} \right]$$

개별 인스턴스 $z$가 실제 핵일 확률 $p(z)$는 $K$개 특징 확률의 평균으로 계산되며, 임계값 $\tau=0.35$를 기준으로 마스크 $U^{(k)}$를 생성하여 고품질 pseudo-label에 가중치 $w=2$를 부여한다. PIAC 손실 함수는 다음과 같다.

$$\mathcal{L}_{PIAC}^{(k+1)} = \frac{1}{N} \sum_{i=1}^{N} \| (F_s^{(k+1)} - F_t^{(k+1)}) \odot U_i^{(k)} \|^2$$

### 최종 손실 함수

전체 손실 함수 $\mathcal{L}$은 지도 학습 손실과 일관성 손실의 합으로 구성된다.

$$\mathcal{L} = \mathcal{L}_{sup} + \gamma_1 \mathcal{L}_{PIAC} + \gamma_2 \mathcal{L}_{MIAC}$$

$\mathcal{L}_{sup}$은 NP 브랜치(Dice loss, CE loss)와 HV 브랜치(MSE loss, MSGE loss)의 합으로 구성되며, $\gamma_1=0.1, \gamma_2=100$으로 설정되었다.

## 📊 Results

### 실험 설정

- **데이터셋:** MoNuSeg, MoNuSAC, PanNuke, CoNSeP 4종의 공개 데이터셋 사용.
- **라벨 데이터 비율:** 1/32, 1/16, 1/8, 1/4 비율로 설정하여 준지도 학습 성능 측정.
- **평가 지표:** AJI (Aggregated Jaccard Index), Dice coefficient, $F1_{obj}$ (Object-level F1-score).

### 주요 결과

- **정량적 결과:** IRCR-Net은 모든 데이터셋과 라벨 비율 설정에서 기존의 ST, MT, PG-FANet보다 우수한 성능을 보였다. 특히 라벨이 극도로 부족한 1/32 설정에서 성능 향상 폭이 컸다.
- **지도 학습 대비 성능:** 1/4 라벨 비율 설정에서는 MoNuSeg와 MoNuSAC 데이터셋에서 완전 지도 학습(Full Supervision) 모델의 성능을 일부 상회하는 결과를 보였다.
- **정성적 결과:** 시각화 분석 결과, IRCR-Net은 겹쳐진 핵의 분리와 복잡한 경계 묘사에서 FullSup 모델에 근접한 정밀함을 보였으며, 특징 맵의 Attention이 핵 영역에 더 집중되는 양상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문의 가장 큰 강점은 일관성 규제를 적용하기 전 **'인스턴스 매칭'**과 **'사전 지식 기반 필터링'** 단계를 두어 pseudo-label의 신뢰도를 획기적으로 높였다는 점이다. 기존 방식들이 전역 맵을 비교함으로써 발생시켰던 '잘못된 예측의 전이' 문제를 인스턴스 수준에서 원천적으로 차단함으로써, 매우 적은 라벨 데이터만으로도 견고한 학습이 가능함을 입증하였다. 특히 경계 강조 손실($B$ term)을 추가한 것이 겹쳐진 핵을 분리하는 데 결정적인 역할을 하였다.

### 한계 및 향후 과제

그럼에도 불구하고, 매우 심하게 겹쳐 있거나 경계가 모호하여 형태학적 변동성이 극심한 영역에서는 여전히 오분류가 발생하는 failure case가 관찰되었다. 이는 단순한 형태학적 통계치만으로는 해결하기 어려운 도메인 특성일 수 있으며, 향후 멀티태스크 학습이나 멀티모달 학습 전략을 도입하여 일반화 성능을 높일 필요가 있다.

## 📌 TL;DR

본 논문은 준지도 학습 기반의 핵 인스턴스 분할을 위해, 전역적 일관성 대신 **인스턴스 수준의 일관성 규제(IRCR)**를 제안하였다. 이분 매칭(MIAC)과 형태학적 사전 지식(PIAC)을 통해 저품질 pseudo-label을 제거함으로써 학습의 견고함을 확보하였으며, 그 결과 매우 적은 라벨 데이터만으로도 기존 SOTA 및 완전 지도 학습 모델에 근접하거나 이를 능가하는 성능을 달성하였다. 이 연구는 의료 영상 분야의 어노테이션 비용 문제를 해결하고 정밀한 병리 분석을 가능하게 하는 데 중요한 기여를 할 것으로 보인다.
