# Memory-augmented Adversarial Autoencoders for Multivariate Time-series Anomaly Detection with Deep Reconstruction and Prediction

Qinfeng Xiao, Shikuan Shao, Jing Wang (2021)

## 🧩 Problem to Solve

본 논문은 수동 감독(manual supervision)이 없는 상태에서 다변량 시계열(multivariate time-series) 데이터의 이상치(anomaly)를 탐지하는 문제를 다룬다. 최근 IT 모니터링 시스템의 규모와 복잡성이 증가함에 따라 다변량 시계열 데이터의 차원이 늘어나고 데이터 분포가 복잡해져 효율적인 이상 탐지가 어려워지고 있다.

기존의 비지도 학습 기반 이상 탐지는 주로 Autoencoder(AE)를 활용하여 정상 데이터로 학습시킨 후, 입력 데이터의 재구성 오차(reconstruction error)가 큰 데이터를 이상치로 판별하는 방식을 사용한다. 그러나 이러한 접근 방식은 다음과 같은 세 가지 핵심적인 한계를 가진다.

1. **다변량 시계열의 복잡성**: 데이터의 내부 복잡성이 지수적으로 증가하면서, 단순한 AE 구조만으로는 데이터 분포를 모델링하는 능력이 부족하다.
2. **이상치의 이질적인 유형**: AE는 점 단위(point-wise) 재구성 오차를 사용하므로 Point anomaly는 잘 잡아내지만, 값의 범위는 정상이나 주변 문맥이나 패턴이 이상한 Contextual anomaly 및 Collective anomaly를 탐지하는 데 한계가 있다.
3. **비정상 입력에 대한 과도한 일반화(Unexpected Generalization)**: 신경망의 강력한 일반화 성능으로 인해, AE가 이상치마저 너무 완벽하게 재구성해버리는 현상이 발생한다. 특히 학습 데이터에 이상치가 섞여 있는 Anomaly Contamination 상황에서는 정상과 이상의 경계가 모호해져 탐지 성능이 크게 저하된다.

따라서 본 논문의 목표는 이러한 과도한 일반화를 방지하고, 다양한 유형의 이상치를 효과적으로 탐지할 수 있는 새로운 비지도 학습 프레임워크인 **MemAAE**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **재구성(Reconstruction)**과 **예측(Prediction)**이라는 두 가지 보완적인 프록시 태스크(proxy tasks)를 결합하고, 여기에 **메모리 모듈(Memory module)**과 **적대적 학습(Adversarial training)**을 도입하여 정상 패턴의 매니폴드(manifold)를 명시적으로 모델링하는 것이다.

- **Adversarial Autoencoder (AAE)**: 단순한 MSE 손실 함수가 유발하는 과도한 평활화(over-smoothing) 문제를 해결하기 위해 판별자(Discriminator)를 도입하여 재구성된 샘플의 세부 디테일을 살리고 분포의 유사성을 확보한다.
- **Memory Module**: 잠재 변수(latent variable)가 전체 공간에 퍼지는 것을 막고, 정상 패턴들의 대표 벡터를 저장하는 딕셔너리 형태의 메모리를 구축한다. 이를 통해 디코더가 오직 정상 패턴의 선형 조합으로만 재구성을 수행하도록 강제하여, 이상치에 대한 일반화 성능을 억제한다.
- **Prediction Module**: 재구성 태스크가 놓치기 쉬운 Contextual 및 Collective anomaly를 잡기 위해, 잠재 벡터를 기반으로 미래 값을 예측하는 브랜치를 추가한다. 시계열의 시간적 상관관계를 이용함으로써 문맥적 이상치에 더욱 민감하게 반응하도록 설계하였다.

## 📎 Related Works

논문에서는 기존의 이상 탐지 기법들을 다음과 같이 분류하고 한계를 지적한다.

1. **전통적 방식**: 거리 기반, 밀도 기반, 고립 기반(Isolation-based), 통계 기반 방식들이 있으나, 복잡한 다변량 시계열의 정상 패턴을 모델링하는 데 한계가 있다.
2. **예측 기반 방식 (Prediction-based)**: LSTM 등을 사용하여 예측 오차를 측정한다. 하지만 복잡한 시계열은 정확한 예측 자체가 매우 어려워 예측 모델의 성능에 전적으로 의존한다는 단점이 있다.
3. **재구성 기반 방식 (Reconstruction-based)**: VAE, GAN 기반 AE 등이 사용된다. 그러나 모델이 너무 강력하면 이상치까지 잘 재구성하는 문제가 있으며, 국소적인 문맥 이상치(local contextual anomalies)에 둔감하다.
4. **멀티태스크 학습 (Multi-task Learning)**: 공유 네트워크 구조로 여러 태스크를 학습하는 방식이 비디오 이상 탐지 등에서는 효과적이었으나, 시계열 분야에서의 적용은 아직 미비한 상태이다.
5. **메모리 모델 (Memory Models)**: 일부 연구에서 일반화를 방지하기 위해 메모리를 사용했으나, 시계열 데이터의 특성(시간적 상관관계 등)을 충분히 반영하지 못했다.

## 🛠️ Methodology

### 전체 시스템 구조

MemAAE는 공유된 인코더(Encoder)를 바탕으로 **재구성 브랜치(Reconstruction branch)**와 **예측 브랜치(Prediction branch)**가 병렬로 연결된 구조를 가진다. 그 사이에 **메모리 모듈(Memory module)**이 위치하여 인코더의 출력을 정상 패턴의 조합으로 필터링한다.

### 주요 구성 요소 및 작동 원리

#### 1. Memory-augmented Adversarial Autoencoder

인코더 $f_e(\cdot)$가 입력 $x$를 잠재 벡터 $z$로 매핑하면, 이 $z$는 메모리 모듈의 쿼리로 사용된다.

- **메모리 모듈**: 메모리는 $M$개의 대표 벡터 $m_i$를 가진 행렬 $M \in \mathbb{R}^{M \times K}$이다. 쿼리 $z$는 프로젝션 헤드 $\psi(\cdot)$와 Softmax를 거쳐 가중치 $w$가 된다.
  $$w = \text{Softmax}(\psi(z))$$
  최종적으로 디코더에 입력되는 잠재 벡터 $\hat{z}$는 메모리 슬롯들의 선형 조합으로 계산된다.
  $$\hat{z} = \sum_{i=1}^{M} w_i \cdot m_i$$
  이를 통해 디코더는 오직 메모리에 저장된 '정상 패턴'만을 사용하여 재구성 $\hat{x} = f_d(\hat{z})$를 수행하게 된다.

- **적대적 학습 (Adversarial Training)**: 판별자 $\phi(\cdot)$를 도입하여 실제 입력 $x$와 재구성된 $\hat{x}$를 구분하도록 학습한다.
  $$L_{adv} = \mathbb{E}_{x \sim p_{data}(x)}[\log(\phi(x))] + \mathbb{E}_{\hat{x} \sim p_{gen}(\hat{x})}[\log(1 - \phi(\hat{x}))]$$
  전체 재구성 손실은 다음과 같다.
  $$L_{full-adv} = L_{adv} + \lambda \cdot \|x - \hat{x}\|^2$$

#### 2. Prediction Module

문맥적 이상치를 잡기 위해 LSTM 기반의 예측 모델 $g(\cdot)$를 사용한다.

- **순방향 및 역방향 예측**: 과거 $k$개의 잠재 벡터를 통해 미래 값 $\tilde{x}_t$를 예측하는 순방향 예측과, 미래 값을 통해 과거를 예측하는 역방향 예측을 모두 수행한다.
- **가중치 감쇠 손실 (Weighted Decay Loss)**: 가까운 시점의 예측이 더 중요하다고 가정하여 시간적 거리에 따른 가중치를 부여한다.
  $$L_{pred} = \frac{1}{T} \sum_{i=1}^{T} (T-i) \cdot \|x_{t+i} - \tilde{x}_{t+i}\|^2$$
  역방향 예측 손실 $L_{pred-back}$ 또한 동일한 방식으로 계산된다.

### 최종 학습 목표 및 추론

전체 손실 함수는 다음과 같이 정의된다.
$$L_{full} = L_{adv} + \lambda \cdot L_{rec} + \gamma_1 \cdot L_{pred} + \gamma_2 \cdot L_{pred-back}$$

이상치 탐지 단계에서는 학습된 모델을 통해 각 샘플의 **이상치 점수(Anomaly Score)**를 다음과 같이 계산한다.
$$S(x) = \lambda \cdot L_{rec} + \gamma_1 \cdot L_{pred} + \gamma_2 \cdot L_{pred-back}$$
이 점수가 미리 정의된 임계값(threshold)보다 높으면 이상치로 판별한다.

## 📊 Results

### 실험 설정

- **데이터셋**: SMAP, MSL (NASA 위성/로버), SMD (서버 머신), SWaT (수처리 시설) 등 4개의 공개 다변량 시계열 데이터셋을 사용하였다.
- **지표**: Precision, Recall 및 Adjusted F1 score를 사용하였다. 특히, 이상 구간 내의 한 점이라도 탐지하면 해당 구간 전체를 탐지한 것으로 간주하는 Modified F1 score 방식을 채택하였다.
- **비교 대상**: IF, DAGMM, AE, LSTM-VAE, OmniAnomaly, USAD 등 6개의 SOTA 모델과 비교하였다.

### 정량적 결과

실험 결과, MemAAE는 모든 데이터셋에서 가장 높은 F1 score를 기록하였다.

- **종합 성능**: 4개 데이터셋 평균 F1 score 0.90을 달성하였으며, 최상위 베이스라인 대비 약 0.02의 성능 향상을 보였다.
- **모델별 비교 분석**:
  - **IF, DAGMM**: 복잡한 시계열 패턴 및 시간적 상관관계를 모델링하지 못해 성능이 낮았다.
  - **OmniAnomaly**: VAE 기반의 재구성이 너무 평활화되어 미세한 이상치를 놓치며, 재구성 오차에만 의존하여 문맥 이상치 탐지에 한계가 있었다.
  - **USAD**: 적대적 학습을 사용해 성능이 좋았으나, 두 개의 판별자를 동시에 학습시켜야 하는 GAN의 불안정성으로 인해 MemAAE보다 낮은 성능을 보였다.

### 추가 분석

- **하이퍼파라미터 민감도**: 재구성 가중치 $\lambda$와 예측 가중치 $\gamma$에 따라 성능 변동이 있으며, 특히 SMAP과 SWaT 데이터셋에서 예측 가중치에 매우 민감하게 반응함을 확인하였다.
- **Ablation Study**:
  - 예측 모듈 제거 시 $\rightarrow$ Contextual/Collective anomaly 탐지 능력이 저하되어 성능 하락.
  - 메모리 모듈 제거 시 $\rightarrow$ 이상치에 대한 과도한 일반화(False Negative 증가) 및 Anomaly Contamination 문제로 성능 하락.

## 🧠 Insights & Discussion

본 논문은 단순한 모델의 깊이를 더하는 것이 아니라, **"정상 패턴의 명시적 보존"**과 **"보완적 태스크의 결합"**이라는 전략을 통해 문제를 해결하였다.

1. **메모리 모듈의 필요성**: AE가 이상치까지 잘 재구성하는 문제는 단순히 네트워크를 작게 만든다고 해결되지 않는다. 메모리 모듈을 통해 잠재 공간을 정상 패턴의 기저(basis)들로 제한함으로써, 모델이 '정상'이 무엇인지 명확히 정의하게 만들고 이상치 입력 시 강제로 정상 패턴으로 투영시켜 재구성 오차를 극대화한 점이 매우 효과적이었다.
2. **재구성과 예측의 시너지**: 재구성은 전체적인 값의 변동(Global anomaly)을 잡는 데 유리하고, 예측은 시계열의 흐름과 문맥(Local/Contextual anomaly)을 잡는 데 유리하다. 이 두 가지를 공유된 인코더 위에서 동시에 학습시킴으로써 다변량 시계열의 다양한 이상 유형을 포괄적으로 탐지할 수 있게 되었다.
3. **한계점**: 하이퍼파라미터(가중치 $\lambda, \gamma$, 메모리 크기 $M$)에 대한 민감도가 데이터셋마다 다르게 나타나는데, 이를 자동으로 최적화하는 방법론에 대해서는 명시적으로 제시되지 않았다. 또한, 임계값(threshold) 설정을 브루트 포스(brute force) 방식으로 최적의 값을 찾았으므로, 실제 실시간 환경에서의 임계값 설정 전략에 대한 논의가 추가될 필요가 있다.

## 📌 TL;DR

MemAAE는 다변량 시계열 이상 탐지에서 발생하는 **과도한 일반화 문제**와 **문맥적 이상치 탐지 불가 문제**를 해결하기 위해 **메모리 모듈**과 **예측 브랜치**를 결합한 적대적 오토인코더 모델이다. 메모리 모듈은 정상 매니폴드를 저장하여 이상치의 재구성을 방해하고, 예측 브랜치는 시간적 상관관계를 학습하여 문맥적 이상치를 잡아낸다. 실험을 통해 4개의 실제 데이터셋에서 SOTA 성능(F1 0.90)을 입증하였으며, 이는 향후 복잡한 IT 인프라 모니터링 및 산업 장비 고장 진단 시스템에 유용하게 적용될 수 있을 것으로 기대된다.
