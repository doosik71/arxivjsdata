# Neural Contextual Anomaly Detection for Time Series

Chris U. Carmona, François-Xavier Aubet, Valentin Flunkert, Jan Gasthaus

## 🧩 Problem to Solve

시계열 데이터에서 이상(anomaly)을 탐지하는 것은 장비 고장 모니터링, IoT 센서 데이터의 이상 행동 감지, IT 인프라 가용성 향상 등 다양한 실제 응용 분야에서 매우 중요합니다. 기존 이상 탐지(AD)는 주로 비지도 학습 문제로 다뤄져 왔지만, 실제로는 제한된 수의 레이블이 지정된 이상 데이터(semi-supervised)나 일반적인 이상 패턴에 대한 도메인 지식(domain knowledge)이 종종 존재합니다.

이러한 가용한 정보를 효과적으로 활용하면서도 비지도 학습부터 지도 학습까지 원활하게 확장 가능하고, 단변량(univariate) 및 다변량(multivariate) 시계열 모두에 적용 가능한 범용적인 시계열 이상 탐지 프레임워크가 필요합니다.

## ✨ Key Contributions

- **상태 최신(SOTA) 시계열 이상 탐지 프레임워크 NCAD(Neural Contextual Anomaly Detection) 제안:** 단변량 및 다변량 시계열, 그리고 비지도, 준지도, 완전 지도 설정 전반에서 SOTA 성능을 달성하는 단순하지만 효과적인 프레임워크를 제시했습니다. (구현 코드가 공개되어 있습니다.)
- **Contextual Hypersphere Detection 도입:** Hypersphere Classifier (HSC)를 확장하여, 시계열의 컨텍스트(context) 표현에 따라 하이퍼스피어의 중심을 동적으로 조정하는 컨텍스트 기반 하이퍼스피어 탐지 기법을 제안했습니다.
- **시계열 이상 탐지를 위한 Outlier Exposure(OE) 및 Mixup 기법 적용:** 컴퓨터 비전 분야에서 개발된 Outlier Exposure와 Mixup 기법을 시계열 이상 탐지에 맞게 조정하여, 합성 이상(synthetic anomalies) 주입을 통한 학습 효율 및 일반화 성능 향상을 입증했습니다.

## 📎 Related Works

- **전통적 통계 기반 시계열 AD:** Shewhart의 통계적 공정 관리(SPC) 이후 ARIMA, 지수 평활법(exponential smoothing) 등의 예측 모델과 SPOT/DSPOT과 같은 극단값 이론(extreme value theory)을 활용한 방법들이 연구되었습니다.
- **딥러닝 기반 시계열 AD:**
  - **예측 기반:** RNN과 같은 딥 신경망을 사용하여 미래 값의 분포를 예측하고, 예측에서 벗어나는 관측치를 이상으로 간주합니다 (예: Shipmon et al., 2017).
  - **재구성(Reconstruction) 기반:** VAE(DONUT, LSTM-VAE, OMNIANOMALY) 또는 GAN(ANOGAN)을 사용하여 정상 데이터를 모델링하고 재구성 오류가 큰 경우 이상으로 판단합니다. MSCRED는 CNN 오토인코더를 사용합니다.
  - **압축(Compression) 기반 / One-Class Classification:** Support Vector Data Description(SVDD)에서 파생된 접근 방식으로, DeepSVDD(정상 데이터를 잠재 공간의 한 점 주위에 집중)와 THOC(DeepSVDD를 시계열에 적용, 다중 구 사용)가 있습니다. HSC(Hypersphere Classifier)는 DeepSVDD를 준지도/지도 설정으로 확장합니다.
- **준지도 및 데이터 증강:** Outlier Exposure(OE)는 보조 데이터셋의 OOD(Out-of-Distribution) 샘플을 활용하여 탐지 성능을 개선하며, Mixup은 훈련 예시의 선형 조합을 통해 모델의 일반화를 향상시키는 데이터 증강 기법입니다.

## 🛠️ Methodology

저자들은 시계열 이상 탐지 문제를 시계열 $x^{(i)}_{1:T_i}$와 부분적인 이상 레이블 $y^{(i)}_{1:T_i} \in \{0,1,?\}$이 주어졌을 때, 각 시점에 대한 이상 점수를 예측하는 것으로 정의합니다.

- **윈도우 기반 Contextual Hypersphere Detection:**

  - 시계열을 고정 크기의 겹치는 윈도우 $w$로 분할합니다. 각 윈도우는 **컨텍스트 윈도우 $w^{(c)}$**와 **의심 윈도우 $w^{(s)}$**로 구성됩니다. 목표는 $w^{(s)}$ 내의 이상을 $w^{(c)}$에 대한 상대적인 기준으로 탐지하는 것입니다.
  - 신경망 인코더 $\phi(\cdot; \theta)$는 $w$와 $w^{(c)}$를 각각 임베딩 벡터 $z=\phi(w;\theta)$와 $z^{(c)}=\phi(w^{(c)};\theta)$로 매핑합니다.
  - 핵심 아이디어는 $w^{(s)}$에 이상이 없는 경우 $z$와 $z^{(c)}$가 유사해야 하며, 이상이 있는 경우 서로 멀어져야 한다는 것입니다.
  - **Contextual Hypersphere Loss**는 기존 HSC 손실을 확장한 것으로, 하이퍼스피어의 중심을 컨텍스트 표현 $z^{(c)}$로 동적으로 설정합니다. 손실 함수는 다음과 같습니다:
    $$ (1-y_i) \left\| \phi(w_i;\theta) - \phi(w_i^{(c)};\theta) \right\|^2_2 - y_i \log \left( 1-\exp \left( -\left\| \phi(w_i;\theta) - \phi(w_i^{(c)};\theta) \right\|^2_2 \right) \right) $$
        여기서 $y_i$는 해당 윈도우의 이상 레이블(0: 정상, 1: 이상)이며, $\| \cdot \|_2^2$는 Euclidean distance의 제곱입니다.

- **NCAD 아키텍처:**

  1. **신경망 인코더 $\phi(\cdot; \theta)$:** 입력 시퀀스를 $R^E$ 공간의 표현 벡터로 매핑합니다. TCN(Temporal Convolutional Network)에 지수 확장(exponentially dilated) 인과 컨볼루션(causal convolutions) 및 적응형 맥스 풀링(adaptive max-pooling)을 사용합니다.
  2. **거리 함수 $dist(\cdot, \cdot)$:** 임베딩 벡터 $z$와 $z^{(c)}$ 사이의 유사도를 계산합니다. 실험에서는 Euclidean distance($L_2$ norm)를 사용했습니다.
  3. **확률적 스코어링 함수 `$\mathcal{L}(z) = \exp(-|z|)$`:** $dist(z, z^{(c)})$를 이상 확률로 변환하여 컨텍스트 윈도우의 임베딩을 중심으로 하는 구형 결정 경계(spherical decision boundary)를 생성합니다.

- **데이터 증강 기법:**
  - **Contextual Outlier Exposure (COE):** 훈련 시 의심 윈도우 $w^{(s)}$의 일부 값을 다른 시계열에서 가져온 값으로 대체하여 컨텍스트와 일치하지 않는 OOD(Out-of-Distribution) 예시를 생성합니다. 다변량 시계열의 경우, 무작위로 차원(dimension)을 선택하여 교체합니다.
  - **Anomaly Injection (Point Outliers, `po`):** 시계열의 무작위 시점에 주변 값의 IQR(Inter-Quartile Range)에 비례하는 스파이크(spike)를 추가하여 단일 지점 이상을 주입합니다.
  - **Window Mixup:** 훈련 배치에서 두 개의 윈도우와 그에 상응하는 레이블을 무작위 계수 $\lambda$로 볼록 조합(convex combination)하여 새로운 훈련 예시와 부드러운 레이블(soft labels)을 생성합니다. 이는 의사결정 경계를 부드럽게 하고 일반화 능력을 향상시킵니다.

## 📊 Results

- **단변량 시계열 벤치마크 (YAHOO, KPI):**
  - YAHOO 데이터셋에서 NCAD는 기존 SOTA 방법인 SR-CNN(65.2% F1)을 크게 능가하는 81.16% F1 점수를 달성했습니다.
  - KPI 데이터셋에서는 SOTA 비지도 방법과 유사한 성능을 보였고, SOTA 지도 방법보다는 약간 낮은 성능을 보였으나, NCAD는 두 설정 모두에서 유연하게 사용 가능합니다.
  - YAHOO 지도 설정에서는 U-NET-DEWA(69.3% F1)를 훨씬 능가하는 79% F1 점수를 달성했습니다.
- **다변량 시계열 벤치마크 (SMAP, MSL, SWaT, SMD):**
  - SMAP, MSL, SWaT 데이터셋에서 NCAD는 THOC를 상당한 마진으로 능가하며 SOTA 성능을 달성했습니다 (예: MSL 95.60% vs 93.67%).
  - SMD 데이터셋에서는 OmniAnomaly(88.57% F1) 다음으로 두 번째 높은 성능(80.16% F1)을 보였습니다. OmniAnomaly가 각 시계열별로 모델을 훈련하는 반면, NCAD는 단일 모델로 전체 데이터셋을 처리하여 효율성 측면에서 강점을 가집니다.
- **어블레이션 스터디:**
  - Contextual hypersphere loss는 모델 성능에 상당한 향상을 제공하며, 데이터 증강 기법이 없는 경우에도 경쟁력 있는 성능을 보였습니다.
  - COE, `po`, Mixup 각각이 성능을 추가적으로 개선하는 것으로 나타났습니다. 특히 COE와 po를 모두 사용하지 않으면 성능이 급격히 저하되었습니다.
- **준지도 학습으로의 스케일링:**
  - 훈련 레이블 수가 증가함에 따라 NCAD의 성능이 꾸준히 향상되는 것을 확인했습니다.
  - 합성 이상(po 또는 COE) 주입은 레이블이 적거나 없는 경우 성능을 크게 향상시켰습니다.
  - Mixup은 주입된 이상과 실제 이상 간의 불일치(mismatch)가 큰 경우에도 모델의 일반화 능력을 개선하는 데 도움이 되었습니다.
- **특정 이상 주입의 효과:**
  - 도메인 지식을 활용하여 실제 이상(예: SMAP 데이터셋의 느린 경사)과 유사한 특정 이상을 주입하면 탐지 성능이 더욱 향상될 수 있음을 보여주었습니다.

## 🧠 Insights & Discussion

- NCAD는 시계열 이상 탐지에서 SOTA 성능을 달성하며, 예측 모델이나 재구성 오류 기반 방법과 같은 전통적인 접근 방식보다 뛰어난 성능을 보입니다. 이는 시계열을 위한 표현 학습과 데이터 증강 기법의 효과적인 결합 덕분입니다.
- Contextual hypersphere loss는 컨텍스트 정보를 효과적으로 활용하여 이상 탐지의 정확도를 높이는 핵심 요소이며, 컨텍스트가 주어졌을 때 의심 윈도우의 변화를 정량화하는 유도 편향(inductive bias)을 제공합니다.
- 데이터 증강을 통한 합성 이상 주입은 실제 레이블이 부족하거나 없는 비지도/준지도 설정에서 모델이 이상과 정상 간의 경계를 효과적으로 학습하도록 돕는 강력한 도구입니다. 특히, Mixup은 주입된 이상과 실제 이상 간의 차이가 있을 때 모델의 일반화를 강화합니다.
- 도메인 지식을 활용하여 특정 유형의 이상을 모방한 합성 이상을 주입함으로써 모델을 특정 이상 감지에 더욱 특화시킬 수 있는 가능성을 보여줍니다.
- 이 연구는 사이버 보안, 인프라 모니터링 등 다양한 분야에서 긍정적인 사회적 영향을 가질 잠재력이 크지만, 환자 건강에 직접적인 영향을 미치는 의료 애플리케이션에서는 알고리즘의 탐지 결과를 맹목적으로 따르지 않고 신중한 접근이 필요함을 명시했습니다.

## 📌 TL;DR

- **문제:** 시계열 이상 탐지는 비지도에서 지도까지 유연하게 확장되며, 단변량/다변량 데이터에서 컨텍스트를 고려한 효과적인 방법이 필요.
- **제안 방법:** **NCAD**는 윈도우 기반 접근 방식을 사용하여 시계열을 컨텍스트 및 의심 윈도우로 분할하고, 동일한 TCN 인코더로 임베딩을 생성. Contextual Hypersphere Loss를 통해 컨텍스트 표현을 중심으로 이상 여부를 판단하며, Contextual Outlier Exposure (COE), Point Outliers (po) 주입, Mixup과 같은 데이터 증강 기법을 활용하여 학습 효율과 일반화 성능을 극대화.
- **주요 결과:** 다양한 벤치마크 데이터셋에서 비지도, 준지도, 지도 설정 모두에서 SOTA 또는 경쟁력 있는 성능을 달성. 컨텍스트 기반 손실 함수와 데이터 증강이 핵심적인 성능 향상 요인임을 입증.
