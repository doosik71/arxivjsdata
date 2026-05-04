# Vicinal Feature Statistics Augmentation for Federated 3D Medical Volume Segmentation

Yongsong Huang et al. (2023)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 기반의 3D 의료 영상 분할(Medical Volume Segmentation)에서 발생하는 두 가지 주요 문제를 해결하고자 한다.

첫째는 데이터 부족 문제이다. 의료 영상 분할 작업은 전문적인 라벨링 비용이 매우 높기 때문에, 소규모 의료 기관(small institutes)의 경우 딥러닝 모델을 효과적으로 학습시키기에 충분한 양의 라벨링된 데이터를 확보하기 어렵다.

둘째는 데이터의 비균질성(Heterogeneity), 즉 Non-IID(not independently and identically distributed) 문제이다. 서로 다른 기관에서 수집된 데이터는 하드웨어 벤더, 촬영 프로토콜, 환자 집단 등의 차이로 인해 서로 다른 분포를 가지게 되며, 이는 모델의 수렴을 방해하고 전역 모델(Global Model)의 일반화 성능을 저하시키는 요인이 된다.

기존의 데이터 증강(Data Augmentation) 기법은 중앙 집중식 학습에서는 효과적이었으나, FL 환경에서는 엄격한 개인정보 보호 제한으로 인해 타 기관의 데이터에 접근할 수 없으므로 단순한 로컬 증강만으로는 기관 간의 데이터 편향(Bias)을 해결하기 어렵다. 따라서 본 논문의 목표는 개인정보를 보호하면서도 로컬 및 전역적 데이터 분포의 차이를 반영할 수 있는 효율적인 특성 수준의 데이터 증강 기법을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 특성 공간(Feature Space)에서 Vicinal Risk Minimization(VRM) 관점을 적용하여, 데이터를 점(point)이 아닌 확률적 분포로 확장하는 **Vicinal Feature-level Data Augmentation (VFDA)** 프레임워크를 제안하는 것이다.

VFDA의 중심 설계 직관은 다음과 같다. 각 기관의 배치별 특성 통계량(Batch-wise Feature Statistics, 즉 평균과 표준편차)이 데이터의 도메인 특성을 추상적으로 나타낸다는 점에 착안하여, 이 통계량을 가우시안 프로토타입(Gaussian Prototype)으로 모델링한다. 이때 가우시안 분포의 중심은 원래의 통계량으로 설정하고, 분산(Variance)을 통해 증강의 범위를 결정한다. 특히, 이 분산을 결정할 때 개별 기관의 로컬 편향뿐만 아니라 모든 참여 기관의 전역적 통계 특성을 함께 고려함으로써, 원본 데이터를 직접 교환하지 않고도 전역적인 데이터 분포의 다양성을 학습에 반영할 수 있도록 설계하였다.

## 📎 Related Works

기존의 연합 학습 기반 데이터 증강 연구들은 주로 다음과 같은 한계를 가진다.

1. **중앙 집중식 증강의 한계**: 로컬 데이터만을 이용한 MixUp이나 CutMix 같은 기법은 로컬 분포 내에서의 일반화는 돕지만, 전역적인 분포의 차이(Global shift)를 반영하지 못해 기관의 편향을 그대로 학습하게 된다.
2. **데이터 전송 기반 접근 방식**: 일부 연구(FedMix, XORMix 등)는 이미지 수준에서 MixUp을 수행하기 위해 기관 간에 이미지 평균값이나 변형된 이미지를 전송한다. 하지만 이는 잠재적인 개인정보 유출 위험이 있으며, 이미지 수준의 단순 결합으로는 고차원적인 시맨틱 변환(Semantic transform)을 구축하는 데 한계가 있다.
3. **작업 범위의 제한**: 기존의 많은 FL 증강 기법들이 분류(Classification) 작업에 집중되어 있어, 3D 의료 영상 분할과 같이 데이터 증강의 수요가 매우 높고 복잡한 작업에는 적용되지 않았다.

VFDA는 원본 데이터나 그 변형물을 전송하지 않고, 추상화된 특성 통계량만을 공유함으로써 개인정보 보호 문제를 해결하는 동시에 특성 수준에서 유연한 증강을 수행한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인
VFDA는 UNet과 같은 인코더-디코더 구조의 모델에서 각 인코더 레이어 뒤에 플러그 앤 플레이(Plug-and-play) 방식으로 삽입된다. 학습 과정에서 각 기관은 로컬 데이터를 통해 특성 통계량을 계산하고, 서버와 통계량의 분산을 교환하며, 이를 통해 샘플링된 새로운 특성을 생성하여 모델을 학습시킨다.

### 주요 구성 요소 및 절차

#### 1. 특성 통계량의 확률적 모델링
인코더의 $l$번째 레이어에서 생성된 배치 특성 맵을 $Z^n_l \in \mathbb{R}^{B \times C \times H \times W \times S}$라고 할 때, 채널별 평균 $\mu^n_l$과 표준편차 $\sigma^n_l$을 다음과 같이 계산한다.
$$\mu^n_l = \frac{1}{H \times W \times S} \sum_{h,w,s} Z^n_l ; \quad \sigma^n_l = \sqrt{\frac{1}{H \times W \times S} \sum_{h,w,s} (Z^n_l - \mu^n_l)^2}$$
여기서 $\mu^n_l, \sigma^n_l \in \mathbb{R}^{B \times C}$이다. VFDA는 이러한 통계량이 가우시안 분포를 따른다고 가정한다.
$$\mu^n_l \sim \mathcal{N}(\mu^n_l, \hat{\Sigma}^2_{\mu^n_l}), \quad \sigma^n_l \sim \mathcal{N}(\sigma^n_l, \hat{\Sigma}^2_{\sigma^n_l})$$

#### 2. 로컬 및 전역 분산의 정량화
증강 범위(분산)를 결정하기 위해 로컬 분산과 전역 분산을 모두 계산한다.

- **로컬 통계 분산(Local Statistic Variances)**: 각 기관 내 미니 배치들 사이의 통계량 변동성을 측정한다.
$$\Sigma^2_{\mu^n_l} = \frac{1}{B} \sum_{b=1}^B (\mu^n_l - E[\mu^n_l])^2, \quad \Sigma^2_{\sigma^n_l} = \frac{1}{B} \sum_{b=1}^B (\sigma^n_l - E[\sigma^n_l])^2$$
- **전역 통계 분산(Global Statistic Variances)**: 모든 기관이 공유하는 통계량의 편차를 측정한다. 먼저 지수 이동 평균(Exponential Momentum Decay, EMD) 전략을 사용하여 모멘텀 통계량 $\bar{\mu}^n_l, \bar{\sigma}^n_l$을 업데이트하고 서버로 전송한다. 서버는 이를 취합하여 전역 분산을 계산한다.
$$\Sigma^2_{\mu_l} = \frac{1}{N} \sum_{n=1}^N (\bar{\mu}^n_l - E[\bar{\mu}_l])^2, \quad \Sigma^2_{\sigma_l} = \frac{1}{N} \sum_{n=1}^N (\bar{\sigma}^n_l - E[\bar{\sigma}_l])^2$$
- **최종 증강 범위 결정**: 로컬 분산과 전역 분산을 곱하여 최종 분산을 산출함으로써, 기관 개별의 특성과 전체 시스템의 다양성을 모두 반영한다.
$$\hat{\Sigma}^2_{\mu^n_l} = \Sigma^2_{\mu_l} \cdot \Sigma^2_{\mu^n_l}, \quad \hat{\Sigma}^2_{\sigma^n_l} = \Sigma^2_{\sigma_l} \cdot \Sigma^2_{\sigma^n_l}$$

#### 3. 특성 증강의 구현 (Inference 및 Training)
최종적으로, 가우시안 분포에서 샘플링된 새로운 통계량 $\hat{\mu}^n_l, \hat{\sigma}^n_l$을 사용하여 기존 특성 $Z^n_l$을 변환한다.
$$\hat{Z}^n_l = \hat{\sigma}^n_l \frac{Z^n_l - \mu^n_l}{\sigma^n_l} + \hat{\mu}^n_l$$
이때 샘플링 과정의 미분 가능성을 확보하기 위해 재매개변수화 기법(Re-parameterization trick)을 사용한다.
$$\hat{\mu}^n_l = \mu^n_l + \epsilon_\mu \hat{\Sigma}_{\mu^n_l}, \quad \hat{\sigma}^n_l = \sigma^n_l + \epsilon_\sigma \hat{\Sigma}_{\sigma^n_l} \quad (\text{where } \epsilon \sim \mathcal{N}(0,1))$$
이 과정은 라벨을 유지한 채 특성 공간만 확장하므로 Label-consistent augmentation이 된다.

## 📊 Results

### 실험 설정
- **데이터셋**: FeTS 2021 (뇌종양 분할), M&M 및 Emidec (심장 해부학적 구조 분할)
- **평가 지표**: Dice Similarity Coefficient (DSC)
- **비교 대상**: FedAvg, FedProx, FedBN, FedNorm, PRRF, FedCRLD 등 6가지 최신 FL 방법론
- **백본 모델**: ResNet 기반 3D UNet

### 주요 결과
1. **정량적 성능 향상**: 뇌종양 분할(FeTS 2021) 작업에서 FedAvg와 FedNorm에 VFDA를 적용했을 때, 단순 MixUp 적용 시보다 더 높은 Dice Score 향상을 보였다. 예를 들어 FedAvg의 경우, 기본 모델(71.14%) $\rightarrow$ MixUp(71.83%) $\rightarrow$ VFDA(72.85%) 순으로 성능이 개선되었다.
2. **범용성 입증**: 심장 분할 작업에서도 FedProx, FedBN, PRRF, FedCRLD 등 다양한 non-IID 대응 FL 방법론들에 VFDA를 추가했을 때 일관되게 성능이 향상됨을 확인하였다. 특히 FedCRLD에 적용했을 때 평균 Dice Score가 85.96%에서 87.48%로 상승하였다.
3. **절제 연구(Ablation Study)**: EMD(지수 이동 평균) 모듈이나 전역 통계 분산(Global Statistic Variances)을 제거했을 때 성능이 하락하는 것을 통해, 전역적인 데이터 분포 정보를 반영하는 것이 매우 중요함을 입증하였다.
4. **정성적 분석 및 학습 안정성**: VFDA를 적용했을 때 분할 경계가 더 정확해졌으며, 특히 데이터 분포가 매우 다른 클라이언트(예: Emidec 데이터셋을 가진 클라이언트 F)에서 학습 손실(Training Loss) 곡선이 훨씬 안정적으로 수렴하는 양상을 보였다.

## 🧠 Insights & Discussion

본 논문은 특성 수준에서의 데이터 증강이 이미지 수준의 증강보다 FL 환경에서 훨씬 유연하고 효과적일 수 있음을 보여주었다. 특히, 단순한 랜덤 증강이 아니라 전역적인 통계량의 분산을 활용하여 증강 범위를 동적으로 조절함으로써, 로컬 모델이 겪는 데이터 드리프트(Data drift) 문제를 효과적으로 완화하였다.

강점으로는 원본 데이터를 전혀 전송하지 않으면서도 전역적인 분포 특성을 학습에 반영할 수 있는 프라이버시 보존형 구조라는 점과, 기존의 어떤 FL 프레임워크에도 쉽게 결합할 수 있는 플러그 앤 플레이 방식이라는 점을 꼽을 수 있다.

다만, 실험 결과에서 VFDA를 디코더(Decoder) 레이어에 적용했을 때는 성능 향상이 나타나지 않았는데, 이는 특성 통계량 기반의 증강이 고수준의 시맨틱 정보를 추출하는 인코더 단계에서 더 효과적임을 시사한다. 또한, 가우시안 분포라는 단순한 확률 모델을 가정했으나, 실제 의료 영상의 특성 분포가 더 복잡한 형태일 경우 이에 대한 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 FL 기반 3D 의료 영상 분할에서 발생하는 데이터 부족 및 Non-IID 문제를 해결하기 위해, 배치 특성 통계량을 가우시안 분포로 모델링하여 증강하는 **VFDA(Vicinal Feature-level Data Augmentation)**를 제안한다. 로컬 및 전역 분산을 결합하여 최적의 증강 범위를 설정하며, 이를 통해 개인정보를 보호하면서도 전역적 데이터 다양성을 확보한다. 뇌종양 및 심장 분할 작업의 다양한 FL 알고리즘에 적용하여 일관된 성능 향상을 입증하였으며, 향후 다양한 FL 시나리오에서 효율적인 데이터 증강 솔루션으로 활용될 가능성이 크다.