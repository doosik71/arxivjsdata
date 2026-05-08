# Multi-scale Anomaly Detection for Big Time Series of Industrial Sensors

Quan Ding, Shenghua Liu, Bin Zhou, Huawei Shen, Xueqi Cheng (2021)

## 🧩 Problem to Solve

본 논문은 산업용 센서에서 발생하는 다변량 대용량 시계열 데이터(Multivariate Big Time Series)에서 이상치(Anomaly)를 조기에 탐지하는 문제를 다룬다. 대용량 시계열 데이터는 길이가 매우 길기 때문에, 기존의 많은 딥러닝 모델들은 최적화 알고리즘의 한계로 인해 데이터를 경험적인 기준에 따라 작은 조각으로 자르는 sliding window 방식을 사용한다.

그러나 저자들은 이러한 단순한 절단 방식이 시계열 데이터가 가진 고유한 의미론적 세그먼트(Semantic segments)를 훼손하여, 마치 문장에서 구두점을 잘못 찍는 것과 같은 부작용을 초래할 수 있다고 지적한다. 따라서 본 연구의 목표는 데이터의 내재적 특성을 반영하여 적절한 절단 지점을 찾고, 이를 통해 다중 스케일(Multi-scale)로 데이터를 재구성함으로써 정교한 이상 탐지를 수행하는 MissGAN 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **다중 스케일 재구성(Multi-scale Reconstruction)**과 **반복적 세그멘테이션(Iterative Segmentation)**의 결합이다.

1. **점진적 정밀화**: 초기에는 거친(Coarse) 단계의 긴 세그먼트로 학습을 시작하고, 학습된 은닉 표현(Hidden representation)을 바탕으로 더 정밀한(Fine-grained) 절단 지점을 찾아내어 재학습하는 반복적 구조를 가진다.
2. **조건부 재구성**: 추가적인 조건 상태(Conditional states, $y$)를 입력으로 사용하여 다양한 모드(Multi-mode)의 시계열 데이터를 효과적으로 재구성한다.
3. **HMM 기반 세그멘테이션**: 저차원 표현 공간에서 Hidden Markov Model(HMM)과 최소 기술 길이(Minimum Description Length, MDL) 원칙을 사용하여 데이터의 특성에 맞는 최적의 세그먼트 경계를 결정한다.

## 📎 Related Works

논문에서는 이상 탐지 방법을 크게 세 가지 카테고리로 분류하여 설명한다.

1. **선형 모델 기반 방법**: PCA(Principal Component Analysis) 등이 있으며, 직교 변환을 통해 차원을 축소하고 정보를 추출한다.
2. **거리 및 확률 기반 방법**: KNN(K-Nearest Neighbor)이나 HMM 등이 있다. HMM은 세그멘테이션에 유용하지만, 시계열 데이터의 분포가 매우 가변적일 경우 적용이 어렵다는 한계가 있다.
3. **딥러닝 기반 방법**: Autoencoder(AE)와 LSTM-AE 등이 대표적이며, 최근에는 GAN(Generative Adversarial Networks)을 이용한 재구성 기반 방법(AnoGAN, Ganomaly, MAD-GAN 등)이 주목받고 있다.

**기존 방식과의 차별점**: 기존의 딥러닝 기반 모델들은 대부분 고정된 길이의 세그먼트를 사용하며, 조건부 정보(Conditional information)를 활용하지 못한다. 반면 MissGAN은 데이터를 동적으로 세그멘테이션하고 조건부 정보를 통합하여 재구성 성능을 높였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

MissGAN은 크게 **재구성 모델(Reconstruction Model)**과 **세그멘테이션 모델(Segmentation Model)** 두 부분으로 구성된다. 전체적인 흐름은 거친 세그먼트로 학습 $\rightarrow$ 은닉 표현 추출 $\rightarrow$ HMM 기반 세그멘테이션 $\rightarrow$ 정밀한 세그먼트로 재학습의 반복 루프로 이루어진다.

### 2. 재구성 모델 (Reconstruction Model)

재구성 네트워크는 GRU(Gated Recurrent Unit) 기반의 Encoder-Decoder 구조와 GAN의 Discriminator로 구성된다.

* **구조**: Encoder $G_E$가 입력 $x$와 조건 $y$를 받아 은닉 표현 $h$를 생성하고, Decoder $G_D$는 이를 바탕으로 원래의 시계열을 역순으로 재구성한다.
* **손실 함수**:
  * **Generator ($L_G$)**: 단순한 재구성 오차와 Discriminator의 중간층 특징을 맞추는 Pairwise Feature Matching Loss를 결합하여 사용한다.
        $$L_G = \|x - G_D(G_E(x))\|^2 + \lambda \|f_D(x|y) - f_D(G_D(G_E(x))|y)\|^2$$
        여기서 $f_D(\cdot)$는 Discriminator의 은닉층 활성화 벡터이며, $\lambda$는 정규화 파라미터이다.
  * **Discriminator ($L_D$)**: 실제 데이터와 재구성된 데이터를 구별하는 전형적인 GAN 손실 함수를 사용한다.
        $$L_D = \log D(x|y) + \log(1 - D(G_D(G_E(x))|y))$$

### 3. 세그멘테이션 모델 (Segmentation Model)

데이터의 최적 절단 지점 $p$를 찾기 위해 HMM 기반의 방법을 사용하며, MDL 원칙에 따라 전체 비용(Cost)을 최소화하는 방향으로 학습한다.

* **비용 함수**:
    $$\text{Cost}(x; \Theta_H) = \alpha \times \text{Cost}_{\text{model}}(\Theta_H) + \text{Cost}_{\text{assign}} + \text{Cost}_{\text{like}}(x|\Theta_H)$$
    여기서 $\alpha$는 패턴의 세분화 정도를 조절하는 하이퍼파라미터이다.

### 4. 추론 및 이상치 판별 (Anomaly Detection)

학습된 Generator를 통해 테스트 데이터를 재구성하고, 실제 값과 재구성 값의 차이를 통해 이상치 점수(Anomalousness score)를 계산한다.
$$A(x_{jt}) = \|x_{jt} - x'_{jt}\|^2$$
특정 시점 $t$의 오차가 크다면 해당 지점을 이상치로 판별한다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**:
  * **SWaT**: 산업용 수처리 시스템 데이터 (입력 25차원, 조건 26차원).
  * **CMU Motion Capture**: 인간의 움직임 데이터 (4개 센서, 걷기/달리기 등 레이블을 조건으로 사용).
* **평가 지표**: AUC (Area Under ROC Curve), Ideal F1 Score.
* **비교 대상**: BeatGAN, LSTM-AE, MAD-GAN 및 ablation study를 위한 CRGAN(세그멘테이션 제외), AEGAN(PCA 제외).

### 2. 주요 결과

* **정량적 성능 (SWaT)**: MissGAN은 AUC와 Ideal F1 Score 모두에서 baseline 모델들을 능가하였다. 특히 CRGAN과의 비교를 통해 다중 스케일 세그멘테이션이 성능 향상에 기여함을 입증하였다.
* **정성적 분석 (Motion Dataset)**: 걷기와 달리기 데이터로 학습시킨 후, 점프나 호핑(hopping)과 같은 예상치 못한 동작이 입력되었을 때 재구성 오차가 크게 발생하는 것을 확인하였다. 또한, 조건(label)을 잘못 부여했을 때 해당 구간의 오차가 급증하는 Heatmap을 통해 모델의 설명 가능성(Explainability)을 보여주었다.
* **확장성 (Scalability)**: 테스트 샘플의 길이에 따른 실행 시간이 선형적으로 증가함을 확인하여, 대용량 시계열 데이터에 대한 처리 효율성을 입증하였다.

## 🧠 Insights & Discussion

**강점**:

* **동적 세그멘테이션**: 고정된 윈도우 크기를 사용하는 대신, 데이터의 내재적 특성을 반영해 세그먼트 길이를 조절함으로써 의미론적 손실을 줄였다.
* **조건부 학습**: 단순한 재구성이 아니라 특정 상태($y$)에서의 정상 패턴을 학습함으로써, 다중 모드 데이터를 가진 산업 현장 데이터에 더 유연하게 대응할 수 있다.
* **설명 가능성**: 시점별 재구성 오차를 통해 정확히 어느 시점에 이상이 발생했는지 핀포인트(Pinpoint)할 수 있다.

**한계 및 논의**:

* **데이터 특성 가정**: 본 모델은 GRU를 사용하므로 시계열 데이터가 기본적으로 부드러운(Smooth) 형태라고 가정한다. 따라서 급격한 스파이크(Spike)가 정상인 데이터셋에서는 성능이 제한될 수 있다.
* **반복 학습 비용**: 세그멘테이션과 재구성을 반복적으로 수행하므로 초기 학습 시간이 증가할 가능성이 있다.

## 📌 TL;DR

MissGAN은 대용량 산업용 시계열 데이터의 이상 탐지를 위해 **HMM 기반의 동적 세그멘테이션**과 **조건부 GAN 재구성 모델**을 결합한 프레임워크이다. 거친 단계에서 정밀한 단계로 세그먼트를 최적화하며 학습함으로써 데이터의 의미론적 구조를 보존하고, 이를 통해 기존 고정 윈도우 방식보다 높은 탐지 정확도와 설명 가능성을 제공한다. 이 연구는 특히 다중 모드 상태를 가진 복잡한 산업 시스템의 모니터링 및 이상 진단에 중요한 역할을 할 가능성이 크다.
