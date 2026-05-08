# Forecasting with time series imaging

Xixi Li, Yanfei Kang, Feng Li

## 🧩 Problem to Solve

기존의 특징 기반 시계열 예측 방법론들은 예측 모델 선택 및 앙상블을 위해 시계열 특징을 수동으로 선택해야 하는 제약을 가지고 있었습니다. 이러한 수동 특징 선택은 유연하지 않으며, 주로 시계열의 전역적(global) 특징에 초점을 맞춰 지역적(local) 특징을 간과하는 경향이 있습니다. 심장 박동이나 불규칙한 기후 변화와 같은 중요한 지역적 동역학 정보가 포함된 경우 이러한 접근 방식은 한계를 보입니다. 따라서, 예측 모델 선택 및 앙상블을 위해 시계열 특징을 자동으로 추출하는 유연한 접근 방식이 필요합니다.

## ✨ Key Contributions

- **시계열 이미지 변환 기반 자동 특징 추출**: 시계열을 회귀도(Recurrence Plots)로 변환하여 이미지로 인코딩하고, 컴퓨터 비전 알고리즘을 사용하여 전역 및 지역 시계열 특징을 자동으로 추출하는 혁신적인 접근 방식을 제안합니다.
- **두 가지 이미지 특징 추출 기법 적용**: SIFT(Scale-Invariant Feature Transform) 기반의 공간 Bag-of-Features (SBoF) 모델과 전이 학습(transfer learning)을 활용한 합성곱 신경망(CNN) 기반 특징 추출을 통해 시계열 이미지를 분석합니다.
- **예측 모델 앙상블에 적용**: 추출된 이미지 특징을 사용하여 여러 후보 예측 모델의 가중치를 학습하고, 이를 통해 모델 앙상블(forecast model averaging)을 수행합니다.
- **경쟁 데이터셋에서의 우수한 성능 입증**: M4 예측 경쟁(M4 competition) 데이터셋에서 최고 수준의 방법론들과 견줄 만한 성능을 보였으며, 관광 예측 경쟁(Tourism competition) 데이터셋에서는 기존 최고 방법론들을 능가하는 결과를 달성했습니다.
- **인간 개입 감소 및 포괄적인 특징 활용**: 수동 개입을 최소화하면서 시계열 데이터의 전역 및 지역적 특성을 포괄적으로 활용할 수 있는 유연한 예측 프레임워크를 제공합니다.

## 📎 Related Works

- **특징 기반 시계열 표현**: 시계열 클러스터링(Wang et al., 2006; Bandara et al., 2020), 분류(Fulcher and Jones, 2014; Nanopoulos et al., 2001), 이상 탐지(Hyndman et al., 2015; Talagala et al., 2019) 등 다양한 데이터 마이닝 작업에서 중요하게 활용됩니다.
- **자동 인코더를 통한 특징 추출**: Corizzo et al. (2020)은 중력파 탐지에, Laptev et al. (2017)과 Abdollahi et al. (2020)은 시계열 예측에 사용했습니다.
- **특징 기반 예측 모델 선택 및 앙상블**: Collopy and Armstrong (1992), Arinze (1994), Shah (1997), Meade (2000), Petropoulos et al. (2014), Kang et al. (2017) 등이 다양한 통계적 특징을 활용했습니다.
- **메타 학습 기반 예측 모델 선택**: Talagala et al. (2018)은 랜덤 포레스트를 사용했으며, Montero-Manso et al. (2020)은 M4 경쟁에서 2위를 차지한 특징 기반 가중 예측 조합 모델 FFORMA를 제안했습니다.
- **시계열 이미지 변환**: Hatami et al. (2017)과 Wang and Oates (2015)는 시계열 분류 작업에서 회귀도를 사용한 시계열 이미지 변환의 가능성을 보여주었습니다.

## 🛠️ Methodology

본 논문은 시계열 이미징 및 특징 추출을 기반으로 한 예측 모델 앙상블 프레임워크를 제안합니다.

1. **시계열 이미지 변환 (Time Series Imaging)**

   - 시계열 $x$를 회귀도(Recurrence Plots, RPs)로 인코딩하여 이미지로 변환합니다. 회귀도는 시계열이 이전 상태를 재방문하는 시점을 시각화하여 시계열의 주기성을 보여줍니다.
   - 수정된 회귀도 공식:
     $$
     R(i,j) = \begin{cases}
     \qquad \text{, if } \lVert x_i - x_j \rVert > \epsilon \\
     \lVert x_i - x_j \rVert \quad \text{otherwise.}
     \end{cases}
     $$
     여기서 $R(i,j)$는 회귀 행렬의 요소, $i$와 $j$는 시간 인덱스, $\epsilon$는 임계값, $\lVert \cdot \rVert$는 유클리드 노름입니다. 이 방식은 이진 출력이 아닌 색상 정보를 포함하여 더 많은 값을 제공합니다.

2. **이미지 특징 추출 (Feature Extraction)**

   - **공간 Bag-of-Features (SBoF) 모델**:
     - **핵심점(Key points) 탐지**: SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 이미지의 지역적 특징을 식별합니다 (4단계: 스케일 공간에서 극값 탐지, 핵심점 찾기, 특징 방향 할당, 핵심점 묘사).
     - **기본 기술자(Basic descriptors) 생성**: K-평균 군집화를 통해 코드북을 형성합니다.
     - **표현 생성**: Locality Constrained Linear Coding (LLC) 방법을 사용하여 각 기술자를 지역 좌표계에 투영하고, 이를 최대 풀링(max pooling)으로 통합합니다.
       $$
       \min_c \sum_i \lVert x_i - B c_i \rVert^2 + \lambda \lVert d_i \odot c_i \rVert^2, \quad \text{s.t. } 1^T c_i = 1, \forall i
       $$
       여기서 $x_i$는 기술자 벡터, $B$는 기본 기술자, $d_i = \exp(\text{dist}(x_i, B)/\sigma)$는 지역성 어댑터입니다.
     - **공간 정보 추출**: 공간 피라미드 매칭(Spatial Pyramid Matching, SPM)과 최대 풀링을 적용하여 이미지의 공간 분포 정보를 포착합니다. 이미지를 1x1, 2x2, 4x4 그리드로 나누어 각 하위 영역에서 특징을 추출하고 결합합니다.
   - **사전 학습된 심층 신경망 (Fine-tuned Deep Neural Networks, CNNs)**:
     - ImageNet 데이터셋으로 사전 학습된 ResNet-v1-101, ResNet-v1-50, Inception-v1, VGG-19와 같은 CNN 모델을 전이 학습(transfer learning) 기법 중 하나인 미세 조정(fine-tuning)을 통해 활용합니다.
     - 사전 학습된 네트워크의 이전 계층(convolutional layers) 가중치는 고정하고, 마지막 완전 연결 계층만 본 연구의 시계열 이미지 특징 추출 작업에 맞게 미세 조정하여 고차원 특징을 얻습니다.

3. **이미지 특징을 활용한 시계열 예측 (Time Series Forecasting with Image Features)**
   - **후보 예측 모델 풀**: 자동 ARIMA, ETS, NNET-AR, TBATS, STLM-AR, RW-DRIFT, THETA, NAIVE, SNAIVE를 포함한 9가지 대중적인 시계열 예측 모델을 사용합니다.
   - **가중치 학습**: 추출된 이미지 특징과 각 후보 모델의 OWA(Overall Weighted Average) 값을 기반으로 XGBoost 모델을 훈련하여 예측 모델 앙상블을 위한 9가지 가중치를 생성합니다. OWA는 MASE(Mean Absolute Scaled Error)와 sMAPE(symmetric Mean Absolute Percentage Error)를 결합한 지표입니다.
     - sMAPE: $\text{sMAPE} = \frac{1}{h} \sum_{t=1}^{h} \frac{2|Y_t - \hat{Y}_t|}{|Y_t| + |\hat{Y}_t|}$
     - MASE: $\text{MASE} = \frac{1}{h} \sum_{t=1}^{h} \frac{|Y_t - \hat{Y}_t|}{\frac{1}{n-m} \sum_{t=m+1}^{n} |Y_t - Y_{t-m}|}$
     - OWA: $\text{OWA} = \frac{1}{2}(\text{sMAPE}/\text{sMAPE}_{\text{Naive2}} + \text{MASE}/\text{MASE}_{\text{Naive2}})$
   - **예측 조합**: 학습된 가중치와 각 모델의 예측을 결합하여 최종 예측을 생성합니다.

## 📊 Results

- **M4 예측 경쟁 데이터셋**:
  - 전체적으로 M4 경쟁 상위 10개 방법론과 매우 유사한 성능을 보였으며, 총 OWA 점수에서 6위를 기록했습니다.
  - 특히, 주간(Weekly) 데이터에서 MASE 2.266 (SIFT), sMAPE 7.899 (SIFT)로 높은 경쟁력을 보였습니다.
  - t-SNE 시각화를 통해 추출된 이미지 특징이 연간, 분기별, 월간, 일별, 시간별 데이터와 같은 다양한 시계열 유형을 잘 구분할 수 있음을 확인했습니다.
- **관광 예측 경쟁 데이터셋**:
  - MASE와 MAPE 모두에서 경쟁 상위 방법론들(ARIMA, ETS, THETA, SNAIVE, DAMPED)을 뛰어넘는 우수한 성능을 달성했습니다.
  - 특히 월별(Monthly) 및 분기별(Quarterly) 데이터에서 탁월한 성능을 보였습니다.
  - 연간(Yearly) 데이터에서는 기존 방법론보다 약간 낮은 성능을 보였는데, 이는 훈련 데이터 부족 때문일 수 있다고 분석했습니다.
- **SIFT vs. CNN 특징 추출**: SIFT 기반 모델이 일부 시나리오에서 딥 CNN 모델보다 더 나은 성능을 보였으나, 딥 CNN은 더욱 자동화된 프로세스와 지속적인 컴퓨터 비전 기술 발전을 활용할 수 있다는 장점이 있습니다.

## 🧠 Insights & Discussion

- **자동화된 특징 추출의 중요성**: 본 연구는 수동 특징 설계의 복잡성을 해결하고 중요한 지역적 패턴을 포착하는 자동화된 시계열 이미지 특징 추출의 잠재력을 입증했습니다.
- **컴퓨터 비전 기술의 활용**: 시계열을 이미지로 변환하여 컴퓨터 비전 알고리즘(SBoF, CNN)을 적용함으로써, 예측 성능을 향상시키고 특징 추출 과정을 자동화할 수 있음을 보여주었습니다. 이는 컴퓨터 비전 분야의 지속적인 발전이 시계열 예측 성능 향상으로 이어질 수 있음을 시사합니다.
- **전이 학습의 효율성**: 사전 학습된 CNN 모델을 미세 조정함으로써, 복잡한 네트워크 구조 설정과 하이퍼파라미터 튜닝의 필요성을 줄여 계산 효율성을 크게 높였습니다.
- **한계 및 향후 연구 방향**:
  - SIFT는 특허 보호 문제와 부분적인 수동 설정이 필요하다는 한계가 있습니다.
  - 후보 예측 모델 선택은 여전히 전문가 지식을 요구하는 부분입니다.
  - 다중 채널 이미지(cross-correlation recurrence plots 등)를 활용하여 더욱 포괄적인 정보를 담는 특징 추출을 시도할 수 있습니다.
  - 시변(time-varying) 이미지 특징, 계층적 시계열, 다변량 시계열 등 복잡한 시계열 데이터에 대한 확장이 필요하며, CNN과 RNN의 결합을 탐색할 수 있습니다.

## 📌 TL;DR

본 논문은 시계열 예측을 위한 자동화된 특징 추출 방법으로 "시계열 이미징"을 제안한다. 시계열 데이터를 회귀도(Recurrence Plots)로 변환하여 이미지로 인코딩한 후, SIFT 기반 SBoF 모델 또는 전이 학습된 CNN을 사용하여 전역 및 지역 특징을 자동으로 추출한다. 추출된 특징은 XGBoost 모델을 통해 여러 후보 예측 모델의 가중치를 학습하는 데 사용되며, 이를 통해 예측 모델 앙상블을 수행한다. M4 및 관광 예측 경쟁 데이터셋 실험 결과, 제안된 방법은 기존의 최고 방법론들과 견줄 만하거나 능가하는 예측 정확도를 달성하며, 수동 개입을 최소화한 유연한 특징 기반 예측 프레임워크의 가능성을 입증했다.
