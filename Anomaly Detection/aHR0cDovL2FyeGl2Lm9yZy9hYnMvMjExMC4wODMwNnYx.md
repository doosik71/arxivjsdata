# Memory-augmented Adversarial Autoencoders for Multivariate Time-series Anomaly Detection with Deep Reconstruction and Prediction

Qinfeng Xiao, Shikuan Shao, Jing Wang (2021)

## 🧩 Problem to Solve

본 논문은 수동적인 감독 없이 다변량 시계열 데이터(Multivariate Time-series)에서 이상치를 탐지하는 문제를 해결하고자 한다. 현대의 IT 모니터링 시스템은 데이터의 차원이 증가하고 복잡성이 높아짐에 따라, 기존의 비지도 학습 기반 이상 탐지 방법론들이 다음과 같은 세 가지 주요 한계에 직면해 있다.

첫째, 다변량 시계열의 복잡성으로 인해 일반적인 Autoencoder(AE)가 데이터 분포를 모델링하는 능력이 한계에 도달하여 성능이 저하된다. 둘째, 이상치의 유형이 다양하다는 점이다. 단순한 Point Anomaly는 재구성 오차(Reconstruction Error)로 탐지 가능하지만, 정상 범위 내에 있으나 패턴이 다른 Contextual Anomaly나 Collective Anomaly는 점 단위의 재구성 목적 함수만으로는 식별하기 어렵다. 셋째, 신경망의 강력한 일반화(Generalization) 능력으로 인해 AE가 이상치까지 너무 잘 재구성해버리는 문제가 발생하며, 특히 학습 데이터에 이상치가 포함된 Anomaly Contamination 상황에서 이 문제는 더욱 심화된다.

따라서 본 연구의 목표는 이러한 한계들을 극복하여 Point, Contextual, Collective Anomaly를 모두 효과적으로 탐지할 수 있는 새로운 비지도 학습 프레임워크인 MemAAE를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 재구성(Reconstruction)과 예측(Prediction)이라는 두 가지 상보적인 프록시 태스크(Proxy Task)를 결합하고, 이를 메모리 모듈과 적대적 학습으로 강화하는 것이다.

1.  **Adversarial Training의 도입**: 단순한 $\ell_2$ 거리 기반의 손실 함수는 결과물을 과도하게 매끄럽게(Over-smoothed) 만드는 경향이 있다. 이를 해결하기 위해 Discriminator를 도입한 적대적 학습을 통해 재구성된 샘플의 디테일을 살리고 분포의 유사성을 높였다.
2.  **Compressive Memory Module의 설계**: 잠재 공간(Latent Space)에 정상 패턴의 기저(Basis)를 저장하는 메모리 모듈을 도입하였다. 인코더가 생성한 쿼리를 통해 메모리 슬롯들의 선형 결합으로 잠재 벡터를 재구성함으로써, 디코더가 오직 '정상 패턴'만을 사용하여 복원하도록 강제하여 이상치에 대한 과도한 일반화를 방지한다.
3.  **Bi-directional Prediction Branch의 추가**: 재구성이 놓치기 쉬운 Contextual 및 Collective Anomaly를 잡기 위해 미래와 과거 값을 동시에 예측하는 예측 브랜치를 설계하였다. 예측 태스크는 시간적 정보를 활용하므로 국소적인 패턴 변화에 더 민감하게 반응한다.

## 📎 Related Works

기존의 시계열 이상 탐지 연구는 크게 거리 기반, 밀도 기반, 격리 기반, 통계 기반 방법론으로 나뉜다. 최근에는 딥러닝 기반의 두 가지 주류 패러다임이 존재한다.

1.  **Prediction-based 방법론**: LSTM 등을 이용해 미래 값을 예측하고 예측 오차가 큰 지점을 이상치로 판단한다. 하지만 복잡한 시계열의 경우 정확한 예측 자체가 매우 어려워 성능에 한계가 있다.
2.  **Reconstruction-based 방법론**: VAE나 GAN 기반 AE를 사용하여 정상 데이터로 학습시킨 후 재구성 오차를 이용한다. 그러나 모델의 성능이 너무 좋으면 이상치까지 잘 재구성해버리는 문제가 있으며, 국소적인 Contextual Anomaly 탐지 능력이 떨어진다는 단점이 있다.

MemAAE는 이러한 기존 방식들과 달리, 적대적 학습을 통해 재구성 품질을 높이고 메모리 모듈로 정상 매니폴드를 명시적으로 모델링하며, 예측 태스크를 병렬로 수행함으로써 두 패러다임의 장점을 결합하고 단점을 보완하였다.

## 🛠️ Methodology

### 전체 시스템 구조
MemAAE는 공유된 인코더를 기반으로 재구성 브랜치와 예측 브랜치가 병렬로 연결된 구조를 가진다. 데이터는 $\text{Encoder} \rightarrow \text{Memory Module} \rightarrow \text{Decoder/Predictor}$ 순으로 흐른다.

### 주요 구성 요소 및 작동 원리

**1. Memory-augmented Adversarial Autoencoder**
인코더 $f_e(\cdot)$는 입력 $x$를 잠재 벡터 $z$로 매핑한다. 하지만 $z$를 그대로 디코더에 넣지 않고, 메모리 모듈 $M = [m_1, \dots, m_M] \in \mathbb{R}^{M \times K}$를 거쳐 $\hat{z}$를 생성한다.

*   **메모리 가중치 산출**: 프로젝션 헤드 $\psi(\cdot)$와 Softmax를 통해 가중치 $w$를 계산한다.
    $$w = \text{Softmax}(\psi(z))$$
*   **잠재 벡터 재구성**: 메모리 슬롯들의 선형 결합으로 $\hat{z}$를 구한다.
    $$\hat{z} = \sum_{i=1}^M w_i \cdot m_i$$
*   **재구성**: 디코더 $f_d(\cdot)$를 통해 $\hat{x} = f_d(\hat{z})$를 얻는다.

**2. Adversarial Training**
재구성된 $\hat{x}$가 실제 $x$와 분포상으로 유사하도록 Discriminator $\phi(\cdot)$를 도입한다.
$$L_{\text{adv}} = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log(\phi(x))] + \mathbb{E}_{\hat{x} \sim p_{\text{gen}}(\hat{x})}[\log(1 - \phi(\hat{x}))]$$
최종 재구성 손실은 $\ell_2$ 거리 기반의 $L_{\text{rec}}$와 $L_{\text{adv}}$의 가중 합으로 정의된다.
$$L_{\text{full-adv}} = L_{\text{adv}} + \lambda \cdot L_{\text{rec}}$$

**3. Bi-directional Prediction Module**
LSTM 기반의 예측기 $g(\cdot)$를 사용하여 미래($L_{\text{pred}}$)와 과거($L_{\text{pred-back}}$) 값을 예측한다. 예측 오차 계산 시, 시간적 거리(Temporal Distance)가 가까울수록 가중치를 더 많이 부여하는 Weighted Decay Loss를 사용한다.
$$L_{\text{pred}} = \frac{1}{T} \sum_{i=1}^{2T} (T-i) \cdot \|x_{t+i} - \tilde{x}_{t+i}\|^2$$

### 전체 학습 목표 및 이상치 점수
전체 손실 함수는 다음과 같다.
$$L_{\text{full}} = L_{\text{adv}} + \lambda \cdot L_{\text{rec}} + \gamma_1 \cdot L_{\text{pred}} + \gamma_2 \cdot L_{\text{pred-back}}$$
테스트 단계에서 각 샘플 $x$의 이상치 점수 $S(x)$는 학습 시 사용한 각 손실 항의 가중 합으로 계산하며, 이 점수가 임계값보다 높으면 이상치로 판단한다.
$$S(x) = \lambda \cdot L_{\text{rec}} + \gamma_1 \cdot L_{\text{pred}} + \gamma_2 \cdot L_{\text{pred-back}}$$

## 📊 Results

### 실험 설정
*   **데이터셋**: SMAP, MSL, SMD, SWaT 등 4개의 공공 다변량 시계열 데이터셋을 사용하였다.
*   **지표**: F1 Score를 주 지표로 사용하였으며, 임계값 설정의 영향을 최소화하기 위해 Brute-force search를 통해 최적의 F1 score를 도출하였다. 또한, 이상치 세그먼트 내의 한 점만 탐지해도 해당 세그먼트 전체를 탐지한 것으로 간주하는 Modified F1 score를 적용하였다.
*   **비교 대상**: IF, DAGMM, AE, LSTM-VAE, OmniAnomaly, USAD 등 6개의 SOTA 모델과 비교하였다.

### 주요 결과
MemAAE는 모든 데이터셋에서 가장 높은 성능을 기록하며, 평균 F1 Score 0.90을 달성하였다. 이는 최우수 베이스라인 대비 약 0.02의 성능 향상을 보인 것이다.

*   **분석**: IF와 DAGMM은 복잡한 시간적 상관관계를 모델링하지 못해 성능이 낮았다. OmniAnomaly는 VAE 특성상 과도하게 매끄러운 재구성이 발생하여 미세한 이상치를 놓치는 경향이 있었으나, MemAAE는 적대적 학습과 메모리 모듈을 통해 이를 극복하였다. USAD 역시 적대적 학습을 사용하지만, 두 개의 판별자를 학습시켜야 하는 불안정성 때문에 MemAAE보다 낮은 성능을 보였다.

### Ablation Study 결과
*   **w/o prediction**: 예측 모듈을 제거했을 때, 특히 Contextual Anomaly 탐지 능력이 떨어져 전체 F1 score가 하락하였다.
*   **w/o memory**: 메모리 모듈을 제거했을 때, 이상치에 대한 과도한 일반화(Generalization) 현상이 발생하여 False Negative가 증가하고 성능이 저하되었다.

## 🧠 Insights & Discussion

본 논문은 다변량 시계열 이상 탐지에서 발생하는 '과도한 일반화'와 '이상치 유형의 다양성' 문제를 구조적 설계로 해결하였다.

**강점 및 해석**:
1.  **메모리 모듈의 역할**: 단순한 정규화(Regularization)가 아니라, 정상 데이터의 전형적인 패턴을 메모리 슬롯에 저장하고 이를 조합하여 복원하게 함으로써, 모델이 학습하지 않은 이상 패턴을 강제로 '정상 패턴의 조합'으로 변환하게 만든다. 이 과정에서 발생하는 큰 재구성 오차가 이상치 탐지의 핵심 동력이 된다.
2.  **상보적 태스크의 결합**: 재구성은 전역적인 형태(Global shape)의 이상을 잡는 데 유리하고, 예측은 국소적인 흐름(Local flow)의 이상을 잡는 데 유리하다. 이 두 가지를 결합함으로써 Point, Contextual, Collective Anomaly를 포괄적으로 탐지할 수 있는 강건함을 확보하였다.

**한계 및 논의사항**:
*   **하이퍼파라미터 민감도**: 실험 결과, $\lambda, \gamma$ 및 메모리 크기 $M$에 따라 성능 변동이 크게 나타났다. 특히 데이터셋마다 최적의 예측 스텝 수($T$)가 다르다는 점은 모델의 범용적인 하이퍼파라미터 설정 가이드를 제시하는 데 어려움이 있음을 시사한다.
*   **임계값 설정**: 논문에서는 Brute-force search로 최적의 임계값을 찾았으나, 실제 실시간 시스템에서는 정답(Label) 없이 임계값을 동적으로 설정하는 방법론에 대한 추가 연구가 필요하다.

## 📌 TL;DR

MemAAE는 **적대적 학습(Adversarial Training)**, **정상 패턴 저장 메모리(Memory Module)**, **양방향 예측(Bi-directional Prediction)**을 결합한 비지도 학습 기반 다변량 시계열 이상 탐지 모델이다. 메모리 모듈을 통해 이상치에 대한 모델의 과도한 일반화를 막고, 예측 태스크를 통해 문맥적 이상치를 효과적으로 포착함으로써 기존 SOTA 모델들보다 우수한 F1 score(0.90)를 달성하였다. 이 연구는 복잡한 산업 시스템의 모니터링 및 이상 징후 포착을 위한 실용적인 프레임워크를 제공한다.