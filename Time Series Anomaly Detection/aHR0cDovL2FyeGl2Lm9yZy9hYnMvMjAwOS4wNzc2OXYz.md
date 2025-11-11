# TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks

Alexander Geiger, Dongyu Liu, Sarah Alnegheimish, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

## 🧩 Problem to Solve

시간 시계열 데이터에서 이상 탐지는 금융, 항공우주, IT, 보안, 의료 등 다양한 분야에서 중요하지만, 다음과 같은 본질적인 문제로 인해 어렵습니다:

- **모호한 이상 정의**: '이상'의 정의 자체가 모호하여 규칙 기반의 접근 방식이 어렵습니다.
- **레이블 부족**: 대부분의 실제 시계열 데이터는 이상에 대한 레이블이 없어 지도 학습 모델 학습이 불가능합니다.
- **복잡한 시간적 상관관계**: 시계열 데이터는 복잡한 시간적 패턴과 비선형성을 가지므로, 컨텍스트를 고려한 이상 탐지가 필요합니다.
- **기존 방법의 한계**: 기존의 비지도 학습 기반 이상 탐지 방법들은 확장성(scalability) 및 이식성(portability) 문제가 있으며, 높은 오탐율(false positive rates)을 보이는 경향이 있습니다.

## ✨ Key Contributions

- **새로운 GAN 기반 비지도 이상 탐지 방법 제안**: 시계열 데이터 재구성을 위한 순환 일관성(cycle-consistent) GAN 아키텍처인 TadGAN을 제안합니다. 이 아키텍처는 시계열-대-시계열(time-series-to-time-series) 매핑을 가능하게 합니다.
- **시계열 유사성 측정 및 이상 점수 계산**: 원본 시퀀스와 GAN이 재구성한 시퀀스 간의 컨텍스트 유사성을 평가하기 위한 두 가지 새로운 시계열 유사성 측정 방법을 식별하고, GAN의 Generator 및 Critic 출력을 활용하여 강력하고 견고한 이상 점수를 계산합니다.
- **광범위한 성능 평가**: NASA, Yahoo, Numenta 등 3개 출처의 11개 시계열 데이터셋에 대해 8가지 다른 기준선(baseline) 방법과 비교하여 TadGAN의 우수성을 입증하고, GAN을 사용한 시계열 이상 탐지에 대한 여러 통찰을 제공합니다. 특히, 11개 데이터셋 중 6개에서 기준선 방법보다 뛰어난 성능을 보였습니다.
- **오픈 소스 벤치마킹 시스템 개발**: 시계열 이상 탐지를 위한 벤치마킹 시스템을 오픈 소스로 공개하여, 추가적인 접근 방식 및 데이터셋으로 확장 가능하게 합니다.

## 📎 Related Works

- **이상 탐지 방법론**:
  - **근접성(Proximity-based) 기반**: K-Nearest Neighbor (KNN), Local Outlier Factor (LOF) 등이 있으며, 객체 간의 거리를 측정하여 이상을 식별합니다. 시계열 데이터에서는 시간적 상관관계를 포착하기 어렵다는 단점이 있습니다.
  - **예측(Prediction-based) 기반**: ARIMA, Holt-Winters와 같은 통계 모델 및 HTM(Hierarchical Temporal Memory), LSTM RNNs 등이 있으며, 미래 값을 예측하고 예측 오류를 통해 이상을 탐지합니다.
  - **재구성(Reconstruction-based) 기반**: PCA, Auto-Encoder (AE), Variational Auto-Encoder (VAE), LSTM Encoder-Decoder 등이 있으며, 데이터의 잠재 구조를 학습하여 재구성 오류가 큰 경우를 이상으로 간주합니다.
  - **GANs 기반 이상 탐지**: 이미지 분야에서 성공적으로 활용되었으며, 최근 시계열 데이터에 대한 MAD-GAN, BeatGAN, Time-Series GAN 등의 연구가 진행되었습니다.
- **본 연구의 차별점**: 기존 GAN 기반 시계열 이상 탐지 연구들은 복잡한 시간적 상관관계로 인해 드물었으며, 본 연구는 시계열 데이터에 순환 일관성 GAN 아키텍처를 도입하여 Generator를 재구성에 직접 활용하고 Critic 및 Generator 출력을 이상 점수 계산에 체계적으로 활용하는 첫 시도입니다.

## 🛠️ Methodology

TadGAN은 비지도 학습 기반의 시계열 이상 탐지 프레임워크로, 재구성 기반의 접근 방식을 따르며 Generative Adversarial Networks (GANs)를 핵심으로 활용합니다.

1. **모델 아키텍처**:

   - **Encoder (E)**: 입력 시계열 시퀀스 $x \in X$를 잠재 공간 $Z$로 매핑합니다 ($E: X \rightarrow Z$).
   - **Decoder (G)**: 잠재 공간 $Z$의 벡터를 재구성된 시계열 시퀀스 $\hat{x} \in X$로 변환합니다 ($G: Z \rightarrow X$).
   - **Critic ($C_x$)**: 실제 시계열 시퀀스 $x$와 Generator $G(z)$가 생성한 가짜 시퀀스를 구별합니다.
   - **Critic ($C_z$)**: 무작위로 샘플링된 잠재 벡터 $z$와 Encoder $E(x)$가 생성한 인코딩된 벡터를 구별합니다.
   - **기반 모델**: Generator E와 G, Critic $C_x$와 $C_z$ 모두 LSTM(Long Short-Term Memory) Recurrent Neural Network를 기본 모델로 사용하여 시계열의 복잡한 시간적 상관관계를 포착합니다.

2. **학습 목표 함수**:

   - **Wasserstein Loss**: 모드 붕괴(mode collapse) 문제를 해결하고 학습 안정성을 높이기 위해 Wasserstein-1 거리를 사용합니다. Critic 네트워크 훈련 시 그라디언트 페널티(gradient penalty)를 적용합니다.
     - $V_X(C_x, G) = E_{x \sim P_X}[C_x(x)] - E_{z \sim P_Z}[C_x(G(z))]$
     - $V_Z(C_z, E) = E_{z \sim P_Z}[C_z(z)] - E_{x \sim P_X}[C_z(E(x))]$
   - **Cycle Consistency Loss**: 재구성의 정확성을 보장하기 위해 도입됩니다. 입력 시퀀스 $x$가 인코딩-디코딩 과정을 거쳐 재구성된 $\hat{x}$($G(E(x))$)와 $x$의 차이를 최소화합니다. L2 norm을 사용하여 이상 값의 영향을 강조합니다.
     - $V_{L2}(E, G) = E_{x \sim P_X}[\Vert x - G(E(x)) \Vert^2]$
   - **최종 목적 함수**: 세 가지 손실 함수를 결합한 MinMax 문제입니다.

3. **이상 점수 계산**:

   - **재구성 오류 (Reconstruction Errors)**: 원본 시계열 $x$와 재구성된 시계열 $\hat{x}$의 차이를 측정합니다.
     - **Point-wise difference**: 각 시점 $t$에서의 절대 차이 $s_t = \vert x_t - \hat{x}_t \vert$.
     - **Area difference**: 특정 길이 $l$의 윈도우 내에서 두 곡선 아래 영역의 평균 차이. 작은 차이가 장기간 지속되는 영역 식별에 유용합니다.
     - **Dynamic Time Warping (DTW)**: 두 시계열 간의 최적의 정합 경로를 찾아 유사성을 측정합니다. 시간 지연(time shift) 문제에도 강건합니다.
   - **Critic 출력 (Critic Outputs)**: $C_x$는 입력 시퀀스가 얼마나 '실제' 같은지를 나타내는 점수를 출력합니다. 이 점수를 스무딩하여 이상 점수로 활용합니다.
   - **점수 결합**: 재구성 오류의 Z-점수 $Z_{\text{RE}}(x)$와 Critic 출력의 Z-점수 $Z_{C_x}(x)$를 정규화한 후 결합합니다.
     - **가중 합 (Convex Combination)**: $a(x) = \alpha Z_{\text{RE}}(x) + (1-\alpha) Z_{C_x}(x)$
     - **곱셈 (Multiplication)**: $a(x) = \alpha Z_{\text{RE}}(x) \cdot Z_{C_x}(x)$ (높은 이상 점수를 증폭하는 효과를 가집니다.)

4. **이상 시퀀스 식별**:
   - **적응적 임계값**: 각 시점에서 계산된 이상 점수에 슬라이딩 윈도우 기반의 임계값 (예: 윈도우 평균으로부터 4 표준편차)을 적용하여 이상 시퀀스를 식별합니다.
   - **오탐 감소 (False Positive Mitigation)**: 식별된 이상 시퀀스들의 최대 이상 점수를 기준으로 정렬하고, 점수 감소율이 특정 임계값 $\theta$를 초과하지 않으면 후속 시퀀스들을 정상으로 재분류하여 오탐을 줄입니다.

## 📊 Results

- **전반적인 성능**: TadGAN은 11개 데이터셋 전체에서 모든 기준선 방법(ARIMA, HTM, LSTM, LSTM AutoEncoder, Dense AutoEncoder, MAD-GAN, DeepAR, MS Azure Anomaly Detector)을 능가하는 가장 높은 평균 F1-점수(0.7)를 달성했습니다.
  - LSTM(0.623) 및 ARIMA(0.599)보다 각각 12.46%, 16.86% 더 높은 성능을 보였습니다.
- **ARIMA 대비 개선**: TadGAN은 ARIMA 대비 15% 이상 F1-점수 개선을 보이며 가장 우수한 성능 향상을 기록했습니다.
- **AutoEncoder 대비 우수성**: AutoEncoder 기반 방법들은 점 이상(point anomalies) 탐지에서 낮은 성능을 보인 반면, TadGAN은 더 높은 점수를 달성하여 GAN이 이상 데이터 과적합 문제에 더 강건함을 시사합니다.
- **MadGAN 대비 우수성**: TadGAN은 Mad-GAN(평균 F1-점수 0.35)보다 훨씬 우수한 성능(0.7)을 보였습니다. 이는 순환 일관성 손실(cycle-consistency loss)의 도입이 Generator E와 G 사이의 불일치를 방지하고 최적의 매핑 경로를 찾는 데 중요한 역할을 했기 때문입니다.

- **Ablation Study (변형 연구) 결과**:
  - **Critic 단독 사용의 불안정성**: Critic 점수만 사용했을 때 평균 F1-점수(0.29)가 가장 낮고 표준 편차(0.237)가 가장 높아 불안정했습니다.
  - **재구성 오류 유형**: DTW(Dynamic Time Warping) 기반 재구성 오류가 다른 두 유형(Point-wise, Area)보다 약간 더 나은 성능을 보였습니다.
  - **Critic 출력과 재구성 오류 결합의 효과**: 대부분의 경우, Critic 출력과 재구성 오류를 결합했을 때 성능이 향상되었습니다. 특히, DTW와 Critic을 곱하여 결합한 'Critic $\times$ DTW'가 0.629로 가장 좋은 F1-점수와 안정성을 보였습니다.
  - **결합 방식**: '곱셈(Multiplication)' 방식이 '가중 합(Convex Combination)' 방식보다 일관되게 더 높은 평균 F1-점수와 더 작은 표준 편차를 보였습니다. 이는 곱셈이 높은 이상 점수를 더 잘 증폭시키기 때문입니다.

## 🧠 Insights & Discussion

- **GAN의 효과적인 활용**: 본 연구는 GAN이 시계열 데이터의 이상 탐지에 효과적으로 사용될 수 있음을 보여주었습니다. 특히, 순환 일관성 GAN 아키텍처와 Generator 및 Critic 출력을 활용한 이상 점수 계산 전략은 기존 방법 대비 강력한 성능을 제공합니다.
- **복합 이상 점수의 중요성**: 재구성 오류와 Critic 출력을 결합하는 것이 단일 방법론보다 더 강력하고 견고한 이상 점수를 제공하여 오탐을 줄이고 정탐을 늘리는 데 기여했습니다.
- **한계점**:
  - **다른 GAN 아키텍처와의 비교 부족**: Time-Series GAN과 같은 다른 GAN 기반 시계열 재구성 아키텍처와의 직접적인 비교가 이루어지지 않았습니다.
  - **재구성 정확도와 이상 탐지 성능의 관계**: 더 나은 신호 재구성이 이상 탐지 성능에 미치는 영향에 대한 추가적인 연구가 필요합니다. 재구성 모델이 이상 데이터에도 과적합되어 오히려 이상을 잘 재구성할 위험이 있을 수 있습니다.

## 📌 TL;DR

본 논문은 시계열 데이터의 이상 탐지 문제를 해결하기 위해 **TadGAN**이라는 새로운 비지도 학습 프레임워크를 제안합니다. TadGAN은 LSTM 기반의 **순환 일관성 GAN** 아키텍처를 사용하여 정상 시계열의 패턴을 학습하고 재구성하며, 재구성 오류(점별, 면적, **DTW**)와 Critic의 출력을 결합하여 강력한 이상 점수를 계산합니다. 광범위한 실험 결과, TadGAN은 11개 데이터셋에서 8가지 기준선 방법을 능가하며 **가장 높은 평균 F1-점수(0.7)**를 달성했습니다. 특히, **DTW 기반 재구성 오류와 Critic 출력을 곱셈 방식으로 결합**했을 때 가장 안정적이고 우수한 성능을 보였습니다. 본 연구는 시계열 이상 탐지에서 GAN의 효과를 입증하고, 관련 연구를 위한 오픈 소스 벤치마킹 시스템을 제공합니다.
