# A Structured Learning Approach with Neural Conditional Random Fields for Sleep Staging

Karan Aggarwal, Swaraj Khadanga, Shafiq Joty, Louis Kazaglis, Jaideep Srivastava (2018)

## 🧩 Problem to Solve

본 연구는 수면 무호흡증(Sleep Apnea) 환자가 사용하는 지속적 양압기(CPAP, Continuous Positive Airway Pressure)의 유량 신호(flow signal)만을 이용하여 수면 단계(Sleep Staging)를 자동으로 판별하는 문제를 해결하고자 한다.

수면 단계의 정확한 판별은 수면의 질을 평가하고 CPAP 치료의 효능을 모니터링하는 데 필수적이다. 기존의 표준 방법인 수면다원검사(PSG, Polysomnography)는 다양한 생체 신호를 측정하므로 정확하지만, 비용이 매우 높고 환자에게 불편함을 주어 장기적인 모니터링에 부적합하다. 따라서 CPAP 장치에서 기본적으로 수집되는 유량 신호만을 활용하여 수면 단계를 추론할 수 있다면, 의료진이 환자의 치료 경과를 지속적으로 관찰할 수 있는 효율적인 메커니즘을 제공할 수 있다.

본 논문의 최종 목표는 입력 신호의 특징 추출과 출력 시퀀스의 전이 구조(transition structure)를 동시에 고려하는 딥러닝 모델을 구축하여, 기존 방법론보다 정확한 수면 단계 판별 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **입력 신호의 고수준 특징 추출(Deep Feature Extraction)**과 **출력 레이블 간의 시계열적 전이 확률(Temporal Transition Structure)**을 결합하여 모델링하는 것이다.

기존의 딥러닝 기반 수면 단계 판별 모델들은 주로 입력 신호에서 유용한 특징을 뽑아내는 데 집중했으며, 각 시점(epoch)의 레이블을 독립적으로 예측하는 Softmax 층을 사용했다. 하지만 수면 단계는 특정 단계에서 다른 단계로 넘어가는 강한 전이 규칙(예: REM 수면에서 Deep 수면으로 바로 넘어가지 않고 중간 단계를 거침)이 존재한다.

이를 해결하기 위해 저자들은 **CNN-RNN-CRF**로 이어지는 end-to-end 프레임워크를 제안하였다. CNN과 RNN을 통해 입력 데이터의 지역적/전역적 특징을 추출하고, 최종 출력단에 **조건부 랜덤 필드(CRF, Conditional Random Field)**를 배치하여 수면 단계 간의 전이 역학을 명시적으로 모델링함으로써 예측 성능을 극대화하였다.

## 📎 Related Works

### 기존 연구 및 한계

1. **신호원 기반 접근**: EEG 기반 센서는 가장 정확하지만 불편함이 크다. 심혈관-호흡 센서나 액티그래피(actigraphy), RF 기반 비접촉 센서 등이 제안되었으나, 각각 정확도가 낮거나 추가 장비가 필요하다는 한계가 있다.
2. **머신러닝 모델**: 초기에는 수작업으로 설계한 특징(hand-crafted features)을 사용했으나, 최근에는 CNN, RNN(LSTM, GRU) 등 딥러닝 모델이 도입되어 성능이 향상되었다.
3. **구조적 정보의 부재**: 최신 R-CNN(Recurrent-CNN) 모델들조차 각 epoch을 독립적으로 분류하는 경향이 있으며, 수면 단계의 전이 구조라는 강력한 도메인 지식을 충분히 활용하지 못하고 있다. 또한, 지역적 정규화(local normalization)로 인한 레이블 편향(label bias) 문제가 발생한다.

### 차별점

본 연구는 최초로 **CPAP 유량 신호**만을 사용하여 수면 단계를 판별하며, 단순한 분류기를 넘어 CRF를 통해 전체 수면 시퀀스의 최적 레이블 조합을 찾는 구조적 학습(Structured Learning) 방식을 도입하였다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 모델은 **CNN $\rightarrow$ RNN $\rightarrow$ CRF** 순으로 구성된 파이프라인을 가지며, 전체 네트워크는 end-to-end 방식으로 학습된다.

#### 1. Convolutional Neural Network (CNN)

입력된 유량 신호 $\mathbf{x}$에서 고수준의 추상적 특징을 추출한다. ResNet 구조에서 영감을 얻어 1D convolution, ReLU, Dropout, Max-pooling으로 구성된 층을 쌓았다.
특히, 기울기 소실 문제를 방지하고 정체성 매핑(identity mapping)을 학습하기 위해 다음과 같은 잔차 연결(Residual Connection)을 사용한다.
$$X'' = F(X) + U^T X$$
여기서 $F(X)$는 합성곱-풀링 층의 출력이며, $U$는 차원을 맞추기 위한 변환 행렬이다. 최종적으로 CNN은 $m=900$개의 특징 벡터 $\mathbf{Z}$를 생성한다.

#### 2. Recurrent Neural Network (RNN)

CNN이 추출한 특징들의 시계열적 맥락을 인코딩하기 위해 GRU(Gated Recurrent Unit)를 사용한다. GRU는 업데이트 게이트 $u_t$와 리셋 게이트 $r_t$를 통해 과거의 정보를 적절히 유지하거나 삭제하며 현재 상태 $h_t$를 갱신한다.
$$u_t = \sigma(W_z z_t + U_z h_{t-1} + b_z)$$
$$r_t = \sigma(W_r z_t + U_r h_{t-1} + b_r)$$
$$\tilde{h}_t = \tanh(W_h z_t + r_t \odot U_h h_{t-1} + b_h)$$
$$h_t = u_t \odot h_{t-1} + (1 - u_t) \odot \tilde{h}_t$$

#### 3. Conditional Random Field (CRF)

RNN의 은닉 상태 $H$를 입력으로 받아 수면 단계 시퀀스 $\mathbf{y}$의 조건부 확률을 모델링한다.

- **Unary Potential ($\Psi_n$)**: 특정 시점 $t$에서 레이블 $y_t$가 가질 점수 (노드 잠재력).
$$\Psi_n(y_t | H, w_n, b_n) = \exp(w_n^T \phi(y_t, H) + b_n)$$
- **Edge Potential ($\Psi_e$)**: 시점 $t-1$에서 $t$로 전이될 때의 점수 (에지 잠재력).
$$\Psi_e(y_{t-1}, y_t | H, w_e, b_e) = \exp(w_e^T \phi(y_{t-1}, y_t, H) + b_e)$$

전체 시퀀스 $\mathbf{y}$에 대한 결합 확률은 다음과 같이 정의되며, 전역 정규화 상수 $Z(H, \theta)$를 통해 확률 분포를 형성한다.
$$p(\mathbf{y} | H, \theta) = \frac{1}{Z(H, \theta)} \prod_{t=1}^m \Psi_n(y_t | \dots) \prod_{t=2}^m \Psi_e(y_{t-1}, y_t | \dots)$$

최종 학습 목표는 음의 로그 가능도(Negative Log-Likelihood)를 최소화하는 것이며, CRF 파라미터에 대해 $l_1$ 정규화를 추가하여 희소성(sparsity)을 부여한다.
$$\min_{\theta} -\log p(\mathbf{y} | H, \theta) + \lambda \|\theta'\|_1$$

#### 4. Class Distribution Cost-Sensitive Prior

수면 데이터의 불균형(REM 및 Deep sleep의 비중이 매우 낮음)을 해결하기 위해 클래스 가중치 $\alpha_k$를 도입한다.
$$\alpha_k = \frac{n}{\mu n_k}$$
여기서 $n$은 전체 레이블 수, $n_k$는 해당 클래스의 레이블 수이다. 이를 손실 함수에 곱해 데이터가 적은 클래스에 더 많은 가중치를 부여하여 학습을 균형 있게 수행한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MESA(Multi-Ethnic Study of Atherosclerosis)의 수면 무호흡증 환자 400명의 유량 신호 데이터.
- **작업**: 30초 단위의 에포크(epoch)를 Wake, REM, Light, Deep 4단계로 분류.
- **평가 지표**: 정확도(Accuracy), 코헨의 카파 계수(Cohen's Kappa, $\kappa$), 수면 효율(Sleep Efficiency, SE)의 평균 절대 오차(MAE).

### 주요 결과

| Approach | Accuracy | Kappa ($\kappa$) | SE MAE% |
| :--- | :---: | :---: | :---: |
| CRF (Handcrafted) | 52.4% | 0.28 | 29.4% |
| R-CNN (Softmax) | 71.5% | 0.49 | 12.5% |
| Conditional Adversarial | 71.1% | 0.49 | 12.6% |
| **Regularized Cost-Sensitive Neural CRF** | **74.1%** | **0.57** | **9.9%** |

- **딥러닝의 효과**: 수작업 특징 기반 CRF보다 R-CNN이 훨씬 높은 성능을 보여, 딥러닝 기반 특징 추출의 중요성이 입증되었다.
- **구조적 학습의 효과**: 단순 R-CNN에 CRF 층을 추가한 Neural CRF는 $\kappa$ score를 0.49에서 0.54로 약 10.2% 향상시켰다.
- **도메인 지식의 효과**: 비용 민감도(Cost-sensitive)와 정규화를 적용했을 때 $\kappa$ score가 0.57까지 상승하여, 희소 클래스(REM, Deep)의 판별 능력이 크게 개선되었다.
- **수면 효율 측정**: SE MAE가 9.9%까지 낮아져, 실제 임상에서 CPAP 치료 반응을 모니터링하는 지표로 활용 가능함을 보였다.

## 🧠 Insights & Discussion

### 강점

- **결합 모델의 시너지**: 입력단의 딥러닝(CNN-RNN)을 통한 강력한 특징 추출과 출력단의 CRF를 통한 전이 구조 모델링을 결합하여 최적의 성능을 달성하였다.
- **현실적 적용 가능성**: 추가 센서 없이 기존 CPAP 장비의 데이터만으로 수면 단계 추론이 가능하게 함으로써 임상적 활용 가치가 매우 높다.
- **해석 가능성 제공**: Saliency map 분석을 통해 모델이 호흡 주기 중 유량이 0에 가까운 정체 구간(plateaus)과 유량 변화가 최대인 지점에 주목하고 있음을 밝혀내어, 모델의 판단 근거가 호흡 생리학적 특징과 연관될 가능성을 제시하였다.

### 한계 및 비판적 해석

- **특정 단계 판별의 어려움**: t-SNE 시각화 결과, Wake와 Light sleep은 잘 구분하지만 REM과 Deep sleep의 구분은 여전히 어렵다. 이는 유량 신호만으로는 뇌파(EEG)만큼의 세밀한 수면 단계 정보를 얻는 데 한계가 있음을 시사한다.
- **데이터 특성**: 본 연구는 수면 무호흡증 환자 데이터를 사용했으므로, 건강한 성인에게 동일한 모델을 적용했을 때의 일반화 성능에 대해서는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 CPAP 유량 신호만을 이용해 수면 단계를 자동으로 판별하는 **Neural CRF (CNN-RNN-CRF)** 모델을 제안하였다. 딥러닝을 통한 특징 추출과 CRF를 통한 수면 단계 전이 구조 모델링을 결합하고, 클래스 불균형을 해결하기 위한 비용 민감도 가중치를 적용하여 기존 딥러닝 모델 대비 코헨의 카파 계수를 약 14% 향상시켰다. 이 연구는 수면 무호흡증 환자의 CPAP 치료 효능을 장기적으로 모니터링할 수 있는 자동화된 도구를 제공한다는 점에서 임상적 가치가 매우 크다.
