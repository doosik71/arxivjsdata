# TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks

Tobias Czempiel, Magdalini Paschali, Matthias Keicher, Walter Simson, Hubertus Feussner, Seong Tae Kim, Nassir Navab (2020)

## 🧩 Problem to Solve

본 논문은 복강경 수술 영상에서 수술 단계(Surgical Phase)를 자동으로 인식하는 문제를 해결하고자 한다. 수술 단계 인식은 환자의 안전을 높이고, 수술 중 의사결정 지원 시스템의 핵심 요소가 될 수 있으며, 수술 프로토콜의 자동 추출을 통해 교육 및 사후 모니터링에 기여할 수 있는 중요한 과제이다.

하지만 다음과 같은 이유로 이 작업은 매우 어렵다:

1. **변동성 및 모호성**: 환자의 해부학적 구조와 외과 의사의 수술 스타일이 다양하며, 각 단계 간의 유사성이 높고 전이 구간(Transition)이 모호하여 성능 저하와 일반화의 어려움이 발생한다.
2. **데이터의 한계**: 사용 가능한 영상 자료의 양이 제한적이며 품질이 일정하지 않다.
3. **기존 모델의 한계**: 기존의 RNN이나 LSTM 기반 접근 방식은 슬라이딩 윈도우(Sliding Window) 검출기를 사용하므로, 수술 전체 시간(수분에서 수시간)에 걸친 장기적인 시간적 패턴(Long-term temporal patterns)을 캡처하는 데 어려움이 있으며, 순차적 처리 특성상 추론의 병렬화가 불가능하여 실시간 온라인 환경 적용에 제약이 있다.

따라서 본 논문의 목표는 넓은 수용 영역(Receptive Field)을 가지며 빠른 추론과 정밀한 단계 예측이 가능한 새로운 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 수술 워크플로우 분석 분야에 처음으로 **Causal Dilated Multi-Stage Temporal Convolutional Networks (MS-TCN)**를 도입한 것이다. 제안된 모델의 명칭은 **TeCNO** (Temporal Convolutional Networks for the Operating room)이다.

중심적인 설계 아이디어는 다음과 같다:

- **Causal Convolution**: 미래의 프레임에 의존하지 않고 현재와 과거의 프레임만을 사용하여 예측함으로써, 수술 중 실시간(Online) 추론이 가능하게 한다.
- **Dilated Convolution**: 층이 깊어질수록 확장 계수(Dilation factor)를 지수적으로 증가시켜, 파라미터 수를 크게 늘리지 않고도 매우 넓은 시간적 수용 영역(Temporal Receptive Field)을 확보한다.
- **Multi-Stage Refinement**: 여러 단계의 TCN을 쌓아 이전 단계의 예측 결과를 점진적으로 정제(Refinement)함으로써 예측의 일관성을 높이고 모호한 전이 구간의 정확도를 개선한다.

## 📎 Related Works

기존의 수술 단계 인식 연구는 다음과 같이 발전해 왔다:

- **초기 접근 방식**: 이진 수술 신호(Binary signals)를 활용하거나, Hidden Markov Models (HMMs)와 Dynamic Time Warping (DTW)를 통해 시간 정보를 캡처하였다. 그러나 이러한 방식은 전체 영상 시퀀스가 필요하므로 온라인 시나리오에 적용할 수 없었다.
- **딥러닝 기반 접근 (RNN/LSTM)**: EndoNet은 CNN과 계층적 HMM을 결합하였고, 이후 EndoLSTM, Endo2N2, SV-RCNet 등이 등장하며 CNN을 통한 특징 추출과 LSTM을 통한 시간적 정제를 수행하였다. MTRCNet-CL은 도구 인식과 단계 인식을 동시에 수행하는 멀티태스크 학습 방식을 제안하였다.

**기존 방식과의 차별점**:
가장 큰 차이점은 LSTM 계열 모델들이 가진 메모리 한계와 순차적 처리 방식의 비효율성을 TCN으로 해결했다는 점이다. TCN은 병렬 처리가 가능하여 학습 및 추론 속도가 빠르며, Dilated Convolution을 통해 LSTM보다 훨씬 더 긴 시간적 맥락을 효율적으로 파악할 수 있다.

## 🛠️ Methodology

TeCNO는 크게 시각적 특징 추출(Visual Feature Extraction)과 시간적 정제(Temporal Refinement)의 두 단계 파이프라인으로 구성된다.

### 1. Feature Extraction Backbone

- **구조**: ResNet50을 사용하여 프레임별로 시각적 특징을 추출한다.
- **학습 방식**: 단계 인식 단일 작업 또는 단계 인식과 도구 식별을 동시에 수행하는 멀티태스크 네트워크로 학습될 수 있다.
- **손실 함수**:
  - 단계 인식: 클래스 불균형을 해결하기 위해 Median Frequency Balancing으로 계산된 가중치를 적용한 **Weighted Cross Entropy Loss**를 사용하며, Softmax 활성화 함수를 적용한다.
  - 도구 식별: 다중 레이블 문제이므로 Sigmoid 활성화 함수와 **Binary Cross Entropy Loss**를 사용한다.

### 2. Temporal Convolutional Networks (TeCNO)

추출된 특징 벡터 $x_{1:t}$를 입력받아 각 시간 단계 $t$에서의 클래스 레이블 $y_t$를 예측한다.

#### Dilated Residual Layer (D)

모델은 풀링 레이어나 완전 연결 레이어 없이 오직 시간적 컨볼루션 레이어로만 구성된다. 각 $D$ 레이어는 다음과 같이 동작한다:

- **Dilated Convolutional Layer (Z)**:
  $$Z^l = \text{ReLU}(W^{1,l} * D^{l-1} + b^{1,l})$$
- **Residual Connection (D)**:
  $$D^l = D^{l-1} + W^{2,l} * Z^l + b^{2,l}$$
여기서 $W^{1,l}$은 확장 컨볼루션 커널, $W^{2,l}$은 $1 \times 1$ 컨볼루션 커널, $*$는 컨볼루션 연산자를 의미한다.

#### Causal Convolution 및 Receptive Field

- **Causal 특성**: 예측값 $\hat{y}_t$가 오직 $t$ 시점과 그 이전의 프레임들($x_{t-n}, \dots, x_t$)에만 의존하도록 하여 온라인 배포를 가능케 한다.
- **수용 영역(RF)**: 확장 계수를 레이어마다 2배씩 증가시킴으로써 수용 영역을 지수적으로 확장한다. 레이어 수 $l$에 따른 수용 영역 $\text{RF}(l)$은 다음과 같다:
  $$\text{RF}(l) = 2^{l+1} - 1$$

#### Multi-Stage Refinement

- **구조**: 첫 번째 단계($S_1$)의 출력을 두 번째 단계($S_2$)의 입력으로 사용하는 계층적 구조이다.
- **학습**: 각 단계 끝에서 독립적인 Weighted Cross Entropy Loss를 계산하며, 전체 단계의 손실을 합산하여 공동 학습한다.
- **전체 손실 함수**:
  $$L^C = \frac{1}{M} \sum_{m=1}^{M} L_{Cm} = -\frac{1}{M} \frac{1}{T} \sum_{m=1}^{M} \sum_{t=1}^{T} w_c y_{mt} \cdot \log(\hat{y}_{mt})$$
  여기서 $M$은 단계 수, $T$는 총 프레임 수, $w_c$는 클래스 가중치이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(80개 영상, 7개 단계, 도구 정보 포함)과 Cholec51(51개 영상, 7개 단계, 도구 정보 없음)을 사용하였다.
- **평가 지표**: Accuracy (Acc), Precision (Prec), Recall (Rec) 세 가지 지표를 사용하였다.
- **비교 대상**: PhaseLSTM, EndoLSTM, MTRCNet, ResNetLSTM 등 LSTM 기반 모델들과 비교하였다.

### 주요 결과

1. **특징 추출기 영향 (Ablation)**: ResNet50이 AlexNet보다 모든 지표에서 우수하였으며, 특히 정확도에서 2~8% 향상을 보였다.
2. **TCN 단계 수의 영향**: TCN을 추가하는 것만으로도 정확도가 크게 향상되었다(ResNet50 기준 6% 상승). 특히 2단계(Stage II)까지는 성능이 지속적으로 향상되었으나, 3단계(Stage III)에서는 오히려 성능이 소폭 하락하였는데, 이는 데이터 부족으로 인한 과적합(Overfitting)으로 분석된다.
3. **베이스라인 비교**:
    - **Cholec80**: TeCNO가 $88.56\%$의 정확도를 기록하며 ResNetLSTM($86.58\%$) 및 MTRCNet($82.76\%$)보다 높은 성능을 보였다. 특히 Precision과 Recall에서 6~10%의 상당한 향상을 보였다.
    - **Cholec51**: TeCNO가 $87.34\%$의 정확도를 기록하여 ResNetLSTM($86.15\%$)보다 우수함을 입증하였다.
4. **정성적 결과**: 시각화 결과, TeCNO는 단순한 단계 예측뿐만 아니라 모호한 단계 전이 구간에서도 매우 매끄럽고 일관된(Smooth and Consistent) 예측 결과를 보여주었다. 특히 지속 시간이 짧은 단계(P5, P7)에서도 강건한 인식 능력을 보였다.

## 🧠 Insights & Discussion

**강점**:

- **장기 의존성 해결**: Dilated Convolution을 통해 LSTM이 해결하지 못한 긴 시간적 맥락을 효율적으로 파악함으로써, 희소하게 등장하는 단계에 대해서도 높은 Precision과 Recall을 달성하였다.
- **실시간성 확보**: Causal Convolution 설계를 통해 미래 프레임 없이도 정확한 예측이 가능하므로, 실제 수술실 내 온라인 시스템으로의 통합 가능성을 열었다.
- **범용성**: 특징 추출기와 시간 정제 모델을 분리한 2단계 접근 방식을 취함으로써, 어떤 특징 추출기(CNN)를 사용하더라도 TCN을 통해 성능을 향상시킬 수 있음을 보였다.

**한계 및 논의**:

- **데이터 규모의 제약**: TCN 단계를 3단계 이상으로 늘렸을 때 성능이 하락한 점은 학습 데이터의 양이 모델의 복잡도를 충분히 지원하지 못함을 시사한다. 더 대규모의 데이터셋이 확보된다면 더 깊은 정제 단계가 효과적일 수 있다.
- **가정**: 본 연구는 특징 추출기가 프레임 단위로 학습되었다고 가정하며, 특징 추출 단계에서의 시간적 정보 활용은 배제하고 오직 TCN 단계에서만 시간 정보를 처리하였다.

## 📌 TL;DR

본 논문은 수술 단계 인식의 고질적인 문제인 **장기적 시간 패턴 캡처**와 **실시간 추론 불가** 문제를 해결하기 위해 **Causal Dilated MS-TCN(TeCNO)**을 제안하였다. 이 모델은 지수적으로 확장되는 수용 영역을 통해 효율적으로 문맥을 파악하고, 다단계 정제 과정을 통해 예측의 일관성을 높였다. 실험 결과, 기존 LSTM 기반 모델들을 성능(특히 정밀도와 재현율)과 추론 속도 면에서 모두 압도하였으며, 이는 향후 실시간 수술 보조 시스템 및 워크플로우 분석 연구에 매우 중요한 기반이 될 것으로 평가된다.
