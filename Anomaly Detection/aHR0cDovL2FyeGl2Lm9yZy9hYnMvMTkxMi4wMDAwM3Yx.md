# A Case for the Score: Identifying Image Anomalies using Variational Autoencoder Gradients

David Zimmerer, Jens Petersen, Simon A. A. Kohland, Klaus H. Maier-Hein (2018)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 의료 영상에서 정답 라벨(annotation)이 없는 상태로 비정상 영역(anomaly), 즉 종양과 같은 병변을 정확하게 식별하는 것이다. 의료 영상 분석에서 딥러닝 모델을 학습시키기 위해서는 방대한 양의 라벨링된 데이터가 필요하지만, 전문의의 수작업에 의존하는 라벨링 과정은 시간이 매우 많이 소요되며 모달리티, 하드웨어 장비, 질환의 종류에 따른 조합이 너무 다양하여 모든 경우에 대해 지도 학습(supervised learning) 모델을 구축하는 것은 현실적으로 불가능하다.

따라서 본 논문의 목표는 라벨이 없는 데이터를 활용하여 비정상 영역을 강조함으로써 전문의의 진단을 보조할 수 있는 무감독 학습(unsupervised learning) 기반의 anomaly detection 방법을 제안하는 것이다. 특히, 기존의 재구성 오차(reconstruction error) 기반 방식보다 이론적 근거가 더 명확하고 충실한 비정상 수치 추정 방법을 찾는 데 집중한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Variational Autoencoder(VAE)의 재구성 오차를 사용하는 대신, 데이터의 로그 밀도(log-density)에 대한 입력값의 미분값인 $\text{score}$를 사용하여 비정상 영역을 탐지하는 것이다.

저자들은 $\text{score}$가 정상 데이터 샘플들이 존재하는 방향을 가리키며, $\text{score}$의 크기가 해당 픽셀이 얼마나 비정상적인지를 나타낸다는 가설을 세운다. 이를 위해 VAE의 목적 함수인 Evidence Lower Bound(ELBO)를 입력 이미지에 대해 미분하여 $\text{score}$를 근사함으로써, 픽셀 단위의 정밀한 비정상 수치(anomaly rating)를 계산하는 방법론을 제시한다.

## 📎 Related Works

기존의 의료 영상 anomaly detection 연구들은 주로 재구성 오차(reconstruction error)에 의존해 왔다. 대표적으로 다음과 같은 접근 방식들이 있었다.
- **통계적 모델 기반:** 실제 이미지와 모델 예측치 사이의 차이를 정량화하여 비정상을 식별하는 방식(Leemput et al.).
- **저차원 분해 기반:** 이미지를 정상 부분을 나타내는 저차원 성분과 병리학적 변이를 나타내는 고주파 성분으로 분해하는 방식(Liu et al.).
- **딥러닝 AE/VAE 기반:** Autoencoder(AE)나 VAE를 사용하여 이미지를 재구성하고, 모델이 학습 과정에서 보지 못한 비정상 영역은 제대로 재구성하지 못한다는 가정을 통해 재구성 오차가 큰 영역을 비정상으로 간주하는 방식(Chen et al., Baur et al., Pawlowski et al.).

그러나 저자들은 이러한 재구성 오차 기반의 접근 방식들이 실질적인 성과를 보임에도 불구하고, 비정상 영역을 정확히 찾아낼 것이라는 이론적인 보장이 부족하다는 점을 한계로 지적한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
본 논문은 VAE를 사용하여 정상 데이터의 분포를 학습한 후, 테스트 이미지의 픽셀별 $\text{score}$를 계산하여 비정상 영역을 탐지한다. 전체 과정은 VAE 학습 $\rightarrow$ ELBO 기반 $\text{score}$ 근사 $\rightarrow$ 픽셀별 비정상 수치 산출 순으로 진행된다.

### VAE 및 ELBO 정의
VAE는 학습 데이터의 생성 모델을 학습하기 위해 Evidence Lower Bound(ELBO)를 최대화한다. ELBO는 다음과 같이 정의된다.

$$\log p(x) \geq -D_{KL}(q(z|x)||p(z)) + \mathbb{E}_{q(z|x)}[\log p(x|z)]$$

여기서 $q(z|x)$는 추론 모델(inference model), $p(z)$는 잠재 변수의 사전 분포(prior), $D_{KL}$은 Kullback-Leibler 발산, $\log p(x|z)$는 생성 모델의 로그 가능도이다.

### Score Approximation (비정상 수치 계산)
$\text{score}$는 로그 밀도의 입력에 대한 미분으로 정의된다. 본 논문에서는 $\log p(x)$를 직접 계산하는 대신, 그 하한선인 ELBO를 사용하여 $\text{score}$를 근사한다.

$$\frac{\partial \log p(x)}{\partial x} \approx \frac{\partial (-D_{KL}(q(z|x)||p(z)) + \mathbb{E}_{q(z|x)}[\log p(x|z)])}{\partial x}$$

이 식은 완전히 미분 가능하므로, 역전파(backpropagation) 알고리즘을 통해 이미지 $x$에 대한 기울기(gradient)를 구할 수 있다. 결과적으로 $\text{score}$는 다음 두 가지 성분의 합으로 구성된다.
1. **KL-loss gradient:** 잠재 공간의 분포가 사전 분포에서 얼마나 벗어났는지를 나타내는 기울기.
2. **Reconstruction-loss gradient:** 입력 이미지와 재구성된 이미지 사이의 차이를 나타내는 기울기.

### 학습 및 추론 절차
- **네트워크 구조:** 5층의 Fully Convolutional Neural Network를 사용하며, 활성화 함수로 LeakyReLUs를, 잠재 공간(latent size)의 크기를 1024로 설정하였다.
- **학습 설정:** Adam 옵티마이저(learning rate=0.0002), 배치 크기 64, 총 60 epoch 동안 학습하였다.
- **후처리:** 컨볼루션 연산으로 인한 checkerboard artifact를 제거하기 위해 Gaussian smoothing을 적용하였으며, 기울기의 노이즈를 줄이기 위해 Smoothgrad 알고리즘을 사용하였다.

## 📊 Results

### 실험 설정
- **학습 데이터셋:** Human Connectome Project(HCP)의 T2 MRI 이미지 1,092장 (정상 데이터).
- **평가 데이터셋:** BraTS-2017 데이터셋 (종양 탐지 작업).
- **입력 해상도:** 64x64 픽셀로 리샘플링 및 정규화.
- **평가 지표:** 픽셀 단위의 ROC-AUC.

### 정량적 결과
다양한 방식의 픽셀별 비정상 수치 계산법을 비교한 결과, ELBO gradient를 이용한 $\text{score}$ 근사 방식이 가장 우수한 성능을 보였다.

| 방법론 | ROC-AUC |
| :--- | :--- |
| Denoising AE (DAE) | $0.808 \pm 0.009$ |
| VAE Reconstruction Error | $0.817 \pm 0.003$ |
| Smoothed Reconstruction Error | $0.843 \pm 0.008$ |
| Sampling Variance | $0.855 \pm 0.013$ |
| Reconstruction-Loss Gradient | $0.894 \pm 0.020$ |
| KL-Loss Gradient | $0.939 \pm 0.007$ |
| **ELBO Gradient (Proposed)** | $\mathbf{0.939 \pm 0.008}$ |

### 주요 분석 결과
- **Score vs Reconstruction Error:** ELBO gradient 기반의 $\text{score}$ 방식이 단순 재구성 오차 방식보다 월등히 높은 성능(0.939 vs 0.817)을 기록하였다.
- **KL-loss의 중요성:** $\text{score}$를 구성하는 두 성분 중 KL-loss gradient가 지배적인 역할을 수행한다. KL-loss gradient 단독 성능(0.939)과 ELBO gradient 전체 성능(0.939)이 거의 동일하며, 재구성 오차 기울기(0.894)는 그보다 낮은 성능을 보였다.
- 이는 KL-loss가 데이터 분포와의 거리(distance to the data distribution)를 더 잘 포착하는 반면, 재구성 오차는 단순한 복원 작업에 더 집중하기 때문인 것으로 해석된다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 연구는 VAE의 기울기를 통해 $\text{score}$를 근사하여 비정상 영역을 탐지하는 새로운 관점을 제시하였다. 특히 기존에 무시되었던 KL-loss의 기울기가 픽셀 단위의 anomaly detection 성능을 끌어올리는 데 결정적인 역할을 한다는 것을 실험적으로 증명하였다.

### 한계 및 가정
- **분포 가정:** $\text{score}$가 비정상 수치를 잘 나타낸다는 가정은 정상 데이터 분포에서 매우 멀리 떨어진 샘플의 경우 성립하지 않을 수 있다.
- **잠재 변수 영향:** 잠재 변수의 개수나 KL-loss의 가중치 설정에 따라 재구성 오차의 상대적인 영향력이 달라질 수 있으며, 이는 전체적인 성능에 영향을 줄 수 있다.

### 비판적 해석
본 논문은 $\text{score}$ 기반 방식의 우수성을 입증하였으나, KL-loss gradient와 ELBO gradient의 성능 차이가 거의 없다는 점은 역설적으로 재구성 오차 성분이 $\text{score}$ 근사에 큰 기여를 하지 못하고 있음을 시사한다. 또한, Smoothgrad나 Gaussian smoothing 같은 후처리 과정이 결과에 어느 정도 영향을 미쳤는지에 대한 정밀한 ablation study가 부족하다는 점이 아쉽다.

## 📌 TL;DR

본 논문은 VAE의 재구성 오차 대신 **ELBO의 입력에 대한 기울기($\text{score}$ 근사치)**를 사용하여 의료 영상의 비정상 영역을 탐지하는 방법을 제안한다. BraTS-2017 데이터셋 실험 결과, 제안 방법은 **ROC-AUC 0.94**라는 높은 성능을 기록하며 기존 재구성 오차 기반 방식들을 압도하였다. 특히 **KL-loss의 기울기**가 비정상 영역 식별에 핵심적인 역할을 한다는 점을 밝혀냈으며, 이는 향후 다른 밀도 추정 모델(Glow, Pixel-CNN++ 등)의 anomaly detection 연구에 중요한 기초가 될 가능성이 높다.