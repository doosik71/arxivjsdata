# VELC: A New Variational AutoEncoder Based Model for Time Series Anomaly Detection

Chunkai Zhang, Shaocong Li, Hongye Zhang, and Yingyang Chen (2020)

## 🧩 Problem to Solve

본 논문은 시계열 데이터(Time Series Data)에서 이상치 탐지(Anomaly Detection)를 수행하는 것을 목표로 한다. 시계열 데이터의 이상치 탐지는 사이버 공격 탐지, 금융 사기 적발, 산업 현장의 센서 오류 진단 등 다양한 분야에서 매우 중요한 문제이다.

기존의 딥러닝 기반 생성 모델(Generative Model)을 이용한 이상치 탐지 방식은 정상 데이터만을 학습하여, 테스트 단계에서 정상 샘플은 잘 재구성(reconstruct)하고 이상 샘플은 재구성 오차가 크게 나타나게 하는 원리를 이용한다. 그러나 생성 모델의 강력한 일반화(generalization) 능력으로 인해, 학습 과정에서 보지 못한 이상 샘플조차 너무 잘 재구성해버리는 문제가 발생한다. 이 경우 이상 샘플의 재구성 오차가 작아져 정상 샘플과 구분하기 어려워지며, 결과적으로 탐지 성능이 저하된다.

따라서 본 논문의 목표는 생성 모델이 이상 샘플을 과도하게 잘 재구성하는 것을 방지하고, 원본 공간(original space)뿐만 아니라 잠재 공간(latent space)에서도 이상치 점수를 계산할 수 있는 새로운 VAE 기반 모델인 VELC를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 VAE의 구조에 **Re-Encoder**와 **Latent Constraint Network**를 추가하여 모델의 재구성 능력을 제어하고 특징 추출 능력을 극대화하는 것이다.

1. **Latent Constraint Network**: 잠재 공간에 제약 네트워크를 추가하여, 모델이 학습 데이터(정상 샘플)와 유사한 잠재 변수만을 생성하도록 강제함으로써 이상 샘플이 너무 잘 재구성되는 현상을 억제한다.
2. **Re-Encoder**: 재구성된 데이터를 다시 잠재 공간으로 매핑하는 Re-Encoder를 도입한다. 이를 통해 모델의 복잡도를 높여 더 많은 특징을 추출할 수 있게 하며, 원본 공간과 잠재 공간이라는 두 가지 서로 다른 특징 공간에서 이상치 점수를 계산하여 탐지 정확도를 높인다.
3. **LSTM 기반 구조**: 시계열 데이터의 복잡한 시간적 상관관계를 효과적으로 모델링하기 위해 VAE의 Encoder와 Decoder, 그리고 Re-Encoder 부분에 Bidirectional LSTM을 사용한다.

## 📎 Related Works

논문에서는 기존의 비지도 학습 기반 이상치 탐지 방법을 두 가지 범주로 설명한다.

1. **예측 기반 방법(Prediction-based)**: LSTM 등을 이용하여 다음 시점의 값을 예측하고, 실제 값과의 예측 오차(prediction error)를 통해 이상치를 탐지하는 방식이다.
2. **재구성 기반 방법(Reconstruction-based)**: VAE, GAN 등의 생성 모델을 사용하여 데이터를 재구성하고, 입력값과 재구성값 사이의 재구성 오차(reconstruction error)를 이용하는 방식이다. 대표적으로 AnoGAN, ALAD, MLP-VAE 등이 있다.

**기존 방식의 한계 및 차별점**: 기존 생성 모델들은 일반화 능력이 너무 뛰어나 이상치까지 재구성해버리는 경향이 있다. VELC는 이를 해결하기 위해 Latent Constraint Network를 통해 잠재 공간의 표현력을 제한함으로써, 정상 데이터의 분포 내에서만 재구성이 이루어지도록 강제한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

VELC는 크게 네 가지 구성 요소로 이루어져 있다: **Original VAE Encoder**, **Original VAE Decoder**, **Re-Encoder**, 그리고 **Constraint Network**이다. 모든 네트워크의 기본 구조는 시계열 특성을 반영하기 위해 Bidirectional LSTM을 사용한다.

전체 흐름은 다음과 같다:
$\text{Input } X \rightarrow \text{Encoder} \rightarrow \text{Latent Space } (z) \rightarrow \text{Constraint Network} \rightarrow \text{Decoder} \rightarrow \text{Reconstructed } X' \rightarrow \text{Re-Encoder} \rightarrow \text{New Latent Space } (rez')$

### 주요 구성 요소 및 상세 설명

**1. Constraint Network**
모델이 이상 샘플을 과도하게 잘 재구성하는 것을 막기 위해 도입되었다.

- 학습 가능한 행렬 $C \in \mathbb{R}^{z_{dim} \times N}$를 정의하며, 각 행은 정상 샘플의 대표 벡터(representative vector) 역할을 한다.
- 샘플링된 잠재 변수 $z$와 $C$의 각 행 사이의 코사인 유사도(cosine similarity)를 계산하여 가중치 벡터 $w$를 구한다.
- $$w_i = \frac{z \cdot c_i}{\|z\| \cdot \|c_i\|}$$
- 정규화된 $w$에 대해 특정 임계값($ths$)보다 작은 값은 0으로 만드는 희소 제약(sparse constraint)을 적용하여 $\hat{w}$를 얻는다.
- 최종적으로 $\hat{w}$와 $C$의 선형 결합을 통해 제약된 잠재 벡터 $\hat{z}$를 생성한다.
- $$\hat{z} = \sum_{i=1}^{N} w'_i c_i$$

**2. Re-Encoder**
재구성된 샘플 $X'$를 다시 잠재 공간으로 매핑하여 새로운 잠재 변수 $rez'$를 생성한다. 이는 모델이 원본 공간과 잠재 공간 모두에서 최적화를 수행하게 하여 더 정밀한 모델링을 가능하게 한다.

### 훈련 목표 및 손실 함수

모델은 정상 데이터만을 사용하여 학습하며, 총 4가지 손실 함수의 합으로 정의된다.
$$L_{VELC} = L_{recx} + L_{KL1} + L_{lat} + L_{KL2}$$

- $L_{recx}$: 원본 데이터 $X$와 재구성 데이터 $X'$ 사이의 $L_2$ 거리 (재구성 손실)
- $L_{KL1}$: VAE의 잠재 공간 분포와 표준 정규 분포 사이의 KL Divergence
- $L_{KL2}$: Re-Encoder가 생성한 잠재 공간 분포의 KL Divergence
- $L_{lat}$: Encoder가 생성한 $z$와 Re-Encoder가 생성한 $z'$ 사이의 $L_2$ 거리

### 추론 및 이상치 점수 계산

테스트 단계에서 샘플 $x_i$에 대한 이상치 점수 $A(x_i)$는 원본 공간의 재구성 오차와 두 잠재 공간 사이의 오차를 가중 합산하여 계산한다.
$$A(x_i) = \alpha \|x - x'\|_1 + \beta \|z' - rez'\|_1$$
여기서 $\alpha + \beta = 1$이며, $\alpha$와 $\beta$는 하이퍼파라미터이다. 계산된 점수는 $[0, 1]$ 범위로 정규화되어 임계값 $\phi$와 비교함으로써 이상 여부를 판별한다.

## 📊 Results

### 실험 설정

- **데이터셋**: UCR 및 UCI 공공 데이터셋에서 추출한 10개의 시계열 데이터셋(KDD99, Arrhythmia, ItalyPowerDemand 등)을 사용하였다.
- **비교 대상**: AnoGAN, ALAD, MLP-VAE, Isolation Forest.
- **측정 지표**: AUC (Area Under the ROC Curve).

### 정량적 결과

실험 결과, VELC는 10개 모든 데이터셋에서 기존 베이스라인 모델들보다 높은 AUC를 기록하였다.

- KDD99 데이터셋에서는 약 0.8% 향상되었으며, 다른 데이터셋들에서는 약 1%에서 5%까지 성능 향상을 보였다.
- 특히 $\alpha=0.6, \beta=0.4$일 때 평균적으로 가장 좋은 성능을 나타냈다.

### 정성적 결과 및 시각화

KDD99와 ECGFiveDays 데이터셋을 통해 원본과 재구성된 시퀀스를 시각화하여 비교하였다.

- **정상 샘플**: 재구성된 시퀀스가 원본과 매우 유사하며 전반적으로 매끄러운 형태를 띤다.
- **이상 샘플**: 재구성된 시퀀스가 원본의 급격한 변동을 따라가지 못하고 정상 데이터의 분포에 가깝게 생성되어, 원본과 재구성 결과 사이에 큰 차이가 발생한다. 이는 Constraint Network가 이상치의 재구성을 성공적으로 억제했음을 보여준다.

## 🧠 Insights & Discussion

본 연구는 VAE의 강력한 일반화 능력이 이상치 탐지에서는 오히려 독이 될 수 있다는 점을 정확히 짚어내고, 이를 해결하기 위해 '제약(Constraint)'이라는 개념을 도입하였다.

**강점**:

1. **이중 공간 분석**: 단순히 픽셀/값 단위의 재구성 오차만 보는 것이 아니라, 잠재 공간에서의 거리(distance)를 함께 고려함으로써 탐지 성능을 높였다.
2. **메모리 기반 제약**: Constraint Network의 행렬 $C$가 정상 데이터의 전형적인 특징들을 기억하는 메모리 역할을 수행하며, 이상치에 대해서는 이 메모리의 조합으로 표현하는 것을 제한함으로써 효과적으로 이상치를 걸러낸다.

**한계 및 논의**:

- 논문에서는 $\alpha$와 $\beta$라는 하이퍼파라미터에 의존하여 점수를 계산하는데, 데이터셋마다 최적의 가중치가 다를 수 있으므로 이를 자동으로 결정하는 방법에 대한 연구가 필요해 보인다.
- Constraint Network의 크기 $N$이나 희소성 임계값 $ths$가 성능에 미치는 영향에 대한 심층적인 분석은 부족한 편이다.

## 📌 TL;DR

본 논문은 시계열 이상치 탐지를 위해 VAE에 **Re-Encoder**와 **Latent Constraint Network**를 결합한 **VELC** 모델을 제안한다. 이 모델은 생성 모델이 이상치를 너무 잘 재구성하는 문제를 해결하기 위해 잠재 공간에 제약을 가하며, 원본 공간과 잠재 공간 모두에서 이상치 점수를 계산한다. 10개의 벤치마크 데이터셋에서 SOTA 모델들보다 우수한 AUC 성능을 입증하였으며, 이는 다양한 시계열 도메인의 이상치 탐지 시스템에 적용될 가능성이 높음을 시사한다.
