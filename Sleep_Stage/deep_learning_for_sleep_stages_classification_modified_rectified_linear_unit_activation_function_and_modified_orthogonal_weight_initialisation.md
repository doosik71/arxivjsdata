# Deep Learning for Sleep Stages Classification: Modified Rectified Linear Unit Activation Function and Modified Orthogonal Weight Initialisation

Akriti Bhusal, Abeer Alsadoon, P. W. C. Prasad, Nada Alsalami & Tarik A. Rashid (2022)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 수면 단계 분류(Sleep Stage Classification)의 정확도를 높이고 학습 및 처리 시간을 단축하는 것이다. 전통적인 수면 단계 분류 방식인 수면다원검사(Polysomnography, PSG)는 복잡한 장비 부착으로 인해 환자의 수면을 방해하며, 전문가가 수동으로 점수를 매기는 과정은 노동 집약적이고 시간이 많이 소요될 뿐만 아니라 판독자 간의 일관성 부족으로 인해 변동성이 크다는 한계가 있다.

최근 딥러닝, 특히 Convolutional Neural Network (CNN) 기반의 자동 분류 시스템이 도입되었으나, 기존 모델들은 높은 복잡성과 낮은 정확도 문제로 인해 실제 시스템에 성공적으로 구현되는 데 어려움이 있었다. 특히 기존의 State-of-the-art (SOTA) 모델인 Orthogonal CNN (OCNN)의 경우, 활성화 함수로 Sigmoid를 사용하여 가중치 업데이트가 느려지는 Gradient Saturation(기울기 포화) 현상이 발생하며, 이는 결과적으로 분류 정확도 저하와 학습 시간 증가로 이어진다. 따라서 본 논문의 목표는 활성화 함수와 최적화 알고리즘을 개선하여 Gradient Saturation 문제를 해결하고, 학습 효율과 정확도를 동시에 향상시킨 ESSC(Enhanced Sleep Stage Classification) 시스템을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존 OCNN 구조의 강점은 유지하되, 신경망의 학습 역학을 저해하는 활성화 함수와 최적화 방식을 수정하는 것이다.

1. **Leaky ReLU 활성화 함수 도입**: Sigmoid 함수에서 발생하는 Gradient Saturation 문제를 해결하기 위해, 음수 영역에서도 작은 기울기를 유지하는 Leaky ReLU를 도입하였다. 이를 통해 기울기 소실 문제를 방지하고 가중치 업데이트 속도를 높여 수렴 속도와 정확도를 개선하였다.
2. **Adam Optimizer 적용**: 가중치 초기화 단계 및 학습 과정에 Adam 최적화 알고리즘을 결합하여 학습 신호의 노이즈를 줄이고, 최적의 결과에 더 빠르게 도달하도록 하여 전체적인 처리 시간을 단축하였다.
3. **효율적인 파이프라인 구성**: Hilbert-Huang Transform (HHT)을 통한 시공간 이미지 표현과 Orthogonal weight initialisation을 유지하여, 연산 효율성을 높임으로써 웨어러블 기기에서도 동작 가능한 가벼운 모델을 지향하였다.

## 📎 Related Works

수면 단계 분류를 위한 기존 연구들은 크게 세 가지 방향으로 진행되었다.

- **특징 추출 기반 방식**: STFT(Short Time Fourier Transform)나 시계열 정보를 활용하여 특징을 추출하는 방식이다. 그러나 일부 연구에서는 인접 에포크(epoch) 간의 연결성이 약해질 때 결과의 신뢰도가 떨어지는 한계가 지적되었다.
- **신경망 모델 기반 방식**: WDBN(Window Deep Belief Network)이나 LSTM(Long Short-Term Memory) 등이 제안되었다. 하지만 수동으로 설계된 특징(hand-crafted features)에 지나치게 의존하거나, 층이 깊어질수록 Overfitting(과적합) 문제가 발생하는 단점이 있었다.
- **OCNN 기반 방식 (SOTA)**: Zhang et al. [3]은 Orthogonal weight initialisation과 Squeeze-and-Excitation (SENet) 블록을 결합한 OCNN을 제안하여 학습 속도를 높이고 풍부한 특징을 추출하였다. 하지만 본 논문에서 지적하듯, Sigmoid 활성화 함수를 사용하여 가중치 업데이트가 매우 느린 Gradient Saturation 문제가 존재한다.

## 🛠️ Methodology

제안된 ESSC 시스템은 **전처리(Pre-processing) $\rightarrow$ 시공간 이미지 표현(Time Frequency Image Representation) $\rightarrow$ Orthogonal CNN**의 세 단계 파이프라인으로 구성된다.

### 1. 전처리 및 이미지 변환

- **전처리**: 데이터셋 간의 차이를 제거하기 위해 64Hz로 리샘플링하고, 50/60Hz Notch 필터 및 0.5~30Hz 대역의 8차 Butterworth 필터를 적용한다. 또한 클래스 불균형을 해소하기 위해 오버샘플링을 수행한다.
- **시공간 이미지 표현**: 1차원 EEG 신호를 2차원으로 변환하기 위해 Empirical Mode Decomposition(EMD) 후 Hilbert Transform을 적용하여 시공간 신호를 생성하며, Autoencoder를 통해 차원을 축소한다.

### 2. Orthogonal CNN 아키텍처

모델은 7개의 Convolution 층, 1개의 Fully Connected 층, 그리고 최종 SoftMax 층으로 구성된다. 특히 가중치를 직교 행렬(Orthogonal Matrix)로 초기화하여 학습 속도를 높이고, SENet 블록을 통해 유용한 특징의 가중치는 높이고 불필요한 특징의 가중치는 낮춘다.

### 3. 핵심 수정 사항 및 방정식

#### (1) Modified Leaky ReLU 활성화 함수

기존의 Sigmoid 함수 $$\sigma(y) = \frac{1}{1 + e^{-y}}$$ 대신, 음수 값에 대해 $0.1$의 기울기를 갖는 Modified Leaky ReLU를 사용한다.

$$Mf(x) = \begin{cases} x & \text{if } x > 0 \\ 0.1x & \text{if } x \le 0 \end{cases}$$

이를 통해 가중치 업데이트가 정체되는 Gradient Saturation을 방지하고 더 빠른 수렴을 가능하게 한다.

#### (2) Adam Optimization

가중치 $W_i$를 초기화하기 전, 다음의 Adam 알고리즘을 통해 최적화한다.

$$W_i = d_i - \alpha \cdot \widehat{dw}$$

여기서 $\widehat{dw}$는 지수 이동 평균을 이용한 모멘텀과 RMSprop의 결합으로 계산된다. 구체적으로 학습률 $lr = 3 \times 10^{-5}$, $\beta_1 = 0.9, \beta_2 = 0.999$를 사용하였다.

#### (3) 최종 출력 특징 맵 (Modified Output Feature Map)

Adam으로 최적화된 가중치 $MW_i$와 Modified ReLU 함수 $Mf$를 결합하여 최종 특징 맵을 산출한다.

$$My_{1,1}^1 = Mf\left(\sum_{i=1}^{N} X_i \cdot MW_{i1} + b_1\right)$$

## 📊 Results

### 실험 설정

- **데이터셋**: UCD, MIT-BIH, Sleep EDF, Sleep EDF Extended, MASS, SHHS 등 6개의 공개 데이터베이스를 사용하였다.
- **평가 지표**: 분류 정확도(Accuracy)와 처리 시간(Processing Time)을 측정하였다.
- **환경**: Intel Core i7, 8GB RAM, Nvidia RTX 2070 GPU, TensorFlow/Keras 프레임워크.

### 정량적 결과

SOTA 모델과 비교했을 때 모든 데이터셋에서 정확도가 향상되었고 처리 시간이 단축되었다.

| 데이터셋 | SOTA 정확도 (%) | 제안 모델 정확도 (%) | 정확도 향상분 | SOTA 처리시간 | 제안 모델 처리시간 | 단축 시간 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCD** | 88.64 | 91.0 | +2.36% | 1250s | 1192s | -58s |
| **MIT-BIH** | 88.1 | 89.6 | +1.5% | 1240s | 1191s | -49s |
| **Sleep EDF** | 83.9 | 87.2 | +3.3% | 1290s | 1203s | -87s |
| **EDF Exp.** | 85.86 | 88.6 | +2.74% | 1260s | 1188s | -72s |

### 주요 분석

- **단계별 정확도**: Wake, S1, SWS, REM 단계에서 유의미한 성능 향상이 관찰되었다. 특히 UCD 데이터셋의 SWS 단계에서는 100%의 정확도를 기록하였다.
- **예외 사항**: UCD 데이터셋의 S2 단계에서는 정확도가 다소 하락하는 경향이 있었는데, 이는 훈련/테스트 데이터의 무작위 샘플링 과정에서 발생한 특성일 수 있다고 분석한다.
- **효율성**: Adam Optimizer와 Leaky ReLU의 결합이 가중치 업데이트 속도를 높여, 모든 데이터셋에서 약 49초에서 87초 사이의 처리 시간 단축 효과를 가져왔다.

## 🧠 Insights & Discussion

본 연구의 강점은 복잡한 모델 구조를 완전히 바꾸는 대신, 딥러닝 모델의 학습 효율에 결정적인 영향을 미치는 **활성화 함수와 최적화 알고리즘이라는 핵심 하이퍼파라미터/구성 요소를 적절히 수정**하여 실질적인 성능 향상을 이끌어냈다는 점이다. 특히 Gradient Saturation 문제를 Leaky ReLU로 해결한 점은 이론적으로 타당하며, 실험 결과에서도 수렴 속도 향상으로 증명되었다.

다만, 몇 가지 한계점과 논의 사항이 존재한다.

1. **데이터 샘플링 방식**: 저자들은 무작위 샘플링 대신 선택적 절차(selective procedure)를 사용했다면 성능이 더 개선되었을 가능성을 언급하였다. 이는 현재 결과가 데이터 분포에 민감할 수 있음을 시사한다.
2. **특정 단계(S2)의 성능 하락**: 대부분의 단계에서 성능이 올랐으나 S2 단계에서 일부 하락이 발생한 원인에 대한 심층적인 분석이 부족하다.
3. **현실적 적용 가능성**: OCNN 구조와 최적화된 파라미터를 통해 연산량을 줄였으므로, 실제 웨어러블 기기에 탑재하여 실시간 수면 모니터링 시스템으로 확장될 가능성이 매우 높다.

## 📌 TL;DR

이 논문은 기존의 Orthogonal CNN 기반 수면 단계 분류 모델이 가진 **Gradient Saturation(기울기 포화) 문제**를 해결하기 위해, 활성화 함수를 **Sigmoid에서 Modified Leaky ReLU($\alpha=0.1$)**로 교체하고 **Adam Optimizer**를 도입한 ESSC 시스템을 제안한다. 실험 결과, 4개의 주요 데이터셋에서 **정확도가 평균 1.5%~3.3% 향상**되었으며, **처리 시간은 데이터셋당 약 49~87초 단축**되었다. 이 연구는 저전력/저사양 웨어러블 수면 진단 기기의 구현 가능성을 높였다는 점에서 중요한 의의를 가진다.
