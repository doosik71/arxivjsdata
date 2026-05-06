# Respecting Time Series Properties Makes Deep Time Series Forecasting Perfect

Li Shen, Yuning Wei and Yangzhu Wang (2022)

## 🧩 Problem to Solve

최근 딥러닝 기반의 시계열 예측 모델들이 CNN, RNN, Transformer 등 다양한 신경망 구조를 도입하며 성능을 높이고 있으나, 정작 시계열 데이터가 가진 고유한 특성(Time Series Properties)은 무시하거나 잘못 이해하고 있다는 문제가 있다. 많은 최신 모델들이 컴퓨터 비전이나 자연어 처리 분야의 기법들을 무분별하게 차용함으로써, 실제 시계열 데이터의 비정상성(Non-stationarity), 무경계성(Unboundedness), 복잡한 이상치(Anomalies) 등을 충분히 반영하지 못하고 있다.

이로 인해 모델의 효율성이 떨어지고, 학습 및 추론 과정이 불안정해지며, 잠재적인 성능을 완전히 끌어내지 못하는 결과가 초래된다. 본 논문의 목표는 시계열의 고유 특성을 엄격하게 분석하고, 이를 반영한 새로운 예측 네트워크인 RTNet(RespectingTime Network)을 제안하여 예측 정확도와 효율성을 동시에 개선하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 직관은 "시계열 특성을 존중하는 것(Respecting Time Series Properties)"이 모델 설계의 중심이 되어야 한다는 점이다. 이를 위해 저자들은 다음의 세 가지 핵심 설계를 제안한다.

1. **적절한 정규화(Normalization)의 선택**: 데이터의 스케일 변화에 민감한 시계열 특성을 고려하여, 은닉 유닛의 분포를 직접 변경하지 않는 Weight Normalization(WN)의 필요성을 이론적으로 제시한다.
2. **변수 간 상관관계의 명시적 처리**: 다변량 예측 시 무관한 변수가 노이즈로 작용하는 것을 막기 위해, 코사인 유사도 기반의 Cos-Relation Matrix를 도입하여 변수 간의 기여도를 조절한다.
3. **입력 시퀀스 길이의 최적화 및 구조적 효율화**: 무조건 긴 입력 시퀀스가 장기 의존성(Long-term dependency)을 해결하는 것이 아니며, 오히려 과적합을 유발할 수 있음을 지적한다. 이를 해결하기 위해 입력 길이를 점진적으로 줄이며 특징을 추출하는 구조와 다중 스케일을 캡처하는 Causal Pyramid Network(CPN)를 제안한다.

## 📎 Related Works

기존의 딥러닝 시계열 예측 연구들은 주로 강력한 특징 추출기(Feature Extractor)를 설계하거나, 계절성(Seasonal) 및 추세(Trend) 성분을 분해하여 학습하는 방식에 집중해 왔다. 하지만 이러한 접근 방식은 시계열 데이터가 정체되어 있거나(Stationary) 이상적인 조건에 있다는 가정 하에 이루어지는 경우가 많다.

본 논문은 기존 연구들이 신경망의 구조적 성능 향상에만 치중했을 뿐, 시계열 데이터 자체의 통계적 특성을 고려한 정규화나 변수 간의 실제 관계, 그리고 입력 길이의 통계적 타당성에 대한 분석이 부족했다는 점을 차별점으로 내세운다.

## 🛠️ Methodology

### 1. 시계열 특성 분석 및 해결책

#### 정규화 (Normalization)

저자들은 Batch Normalization(BN)과 Layer Normalization(LN)이 시계열의 스케일 불변성(Scale invariance) 결여 특성(P1)과 인과성(Causality, P2)을 위반하여 성능을 저하시킨다고 분석한다. 특히 LN은 Transformer 디코더와 같은 인과적 구조에서 미래 정보 누수(Information leakage) 문제를 일으킬 수 있다. 따라서 데이터의 재매개변수화에 불변하지 않으며 인과성을 존중하는 Weight Normalization(WN)을 사용한다.
$$w = g\hat{v} = \frac{g}{\|v\|}v$$

#### 다변량 예측 (Multivariate Forecasting)

타겟 변수와 무관한 독립 변수가 입력될 경우 예측 성능이 오히려 저하될 수 있다(Corollary 2). 이를 해결하기 위해 변수 간의 절대 코사인 유사도를 이용한 **Cos-Relation Matrix** $W(x)$를 정의한다.
$$W(x)_{n \times n} = Mat(w_{ij}) = Mat(\cos(x_i, x_j))$$
이 행렬을 입력 텐서에 곱함으로써 각 변수는 다른 변수들과의 상관관계에 따라 가중 합산된 형태로 변환되어 독립적인 네트워크로 전달된다. 또한, 특정 임계값 $\theta$ 미만의 관계값은 0으로 처리하여 무관한 변수의 간섭을 완전히 차단한다.

#### 입력 시퀀스 길이 (Input Sequence Length)

입력 길이가 너무 길면 과적합(Overfitting)이 발생하며, 특히 Transformer의 어텐션 스코어가 희소(Sparse)하게 분포하는 현상이 발생한다. 이는 실제 예측에 유효한 특정 길이 $P$가 존재함을 시사한다.

### 2. RTNet 아키텍처

RTNet은 위에서 분석한 특성들을 통합한 구조로, 크게 세 가지 구성 요소로 이루어져 있다.

* **RTBlock 및 Backbone**: ResNet-18을 1차원 버전으로 변형한 구조이다. 모든 컨볼루션 층은 Group Convolution을 사용하여 변수 간의 독립성을 유지하며, 층이 깊어질수록 시퀀스 길이를 줄이고 채널 수를 늘려 계층적 특징을 추출한다.
* **Causal Pyramid Network (CPN)**: 시계열의 인과성을 고려하여, 예측 시점과 가까운 데이터가 더 중요하다는 점에 착안한 구조이다. 서로 다른 길이의 입력 부분을 처리하는 여러 개의 독립적인 네트워크를 병렬로 배치하고, 최종적으로 이를 결합하여 다중 스케일의 특징 맵을 생성한다.
* **Decoupled Time Embedding**: 시간 정보(Time Embedding)를 입력 윈도우가 아닌 예측 윈도우에 직접 적용하는 독립적인 TimeNet을 사용한다. 이는 시간 정보가 일반 변수와는 다른 결정론적 특성을 가지므로, 특징 추출 과정을 분리하여 과적합을 방지하기 위함이다.

### 3. 학습 절차 및 손실 함수

RTNet은 두 가지 학습 포맷을 지원한다.

1. **End-to-End 포맷**: 입력 시퀀스 특징과 시간 특징을 합산하여 최종 예측값을 도출하며, MSE(Mean Squared Error) 손실 함수를 사용한다.
2. **Contrastive Learning 기반 포맷**:
    * **1단계 (자기지도 학습)**: 입력 시퀀스를 증강(Scaling, Jittering 등)하여 표현체(Representation)를 학습한다. 이때 self-similarity를 고려한 수정된 InfoNCE 손실 함수를 사용한다.
    $$Loss_m = -\log \frac{e + \sum_{i=1}^I \text{sim}(h_m, h_{mi})}{\sum_{j=1}^B (\text{sim}(h_m, h_j) + \sum_{i=1}^I \text{sim}(h_m, h_{ji}))}$$
    * **2단계 (예측 학습)**: 1단계에서 고정된 특징 추출기를 사용하여 최종 예측을 수행한다.

## 📊 Results

### 실험 설정

* **데이터셋**: ETT (전력 변압기 온도), WTH (기상 데이터), ECL (전력 소비 부하)
* **지표**: MSE, MAE
* **비교 대상**: ARIMA, Prophet, N-BEATS, Informer, TS2Vec, CoST 등 수십 개의 SOTA 모델

### 주요 결과

* **정량적 성능**: RTNet은 거의 모든 조건에서 기존 모델들을 압도하는 성능을 보였다. 특히 ETTh1 데이터셋의 예측 길이 24에서 MSE 0.03 미만이라는 기록적인 수치를 달성하였다.
* **정규화 검증**: WN을 사용했을 때 BN/LN을 사용했을 때보다 MSE가 낮게 나타나, Corollary 1이 실증적으로 증명되었다.
* **Cos-Relation Matrix 효과**: 다변량 예측 시 Cos-Relation Matrix를 적용한 모델이 적용하지 않은 모델보다 월등히 높은 성능을 보였다.
* **입력 길이 분석**: WTH 데이터셋 실험 결과, 입력 길이를 32~48 정도로 설정했을 때 가장 성능이 좋았으며, 이를 초과하여 길이를 늘릴 경우 오히려 MSE가 증가하는 과적합 현상이 관찰되었다.
* **효율성**: 입력 시퀀스 길이를 점진적으로 줄이는 구조 덕분에 SCINet이나 TS2Vec보다 학습 및 추론 시간이 현저히 짧았다.

## 🧠 Insights & Discussion

본 논문은 딥러닝 모델의 복잡성을 높이는 것보다, 데이터의 도메인 특성(시계열의 고유 성질)을 정확히 이해하고 이를 아키텍처에 반영하는 것이 성능 향상의 핵심임을 보여준다. 특히, 업계에서 통용되는 "장기 의존성(Long-term dependency)"을 위해 무작정 입력 길이를 늘리는 관습이 실제로는 독이 될 수 있음을 실험적으로 입증한 점이 매우 인상적이다.

**강점**:

* 단순한 모델 제안에 그치지 않고, 정규화, 다변량 관계, 시퀀스 길이라는 세 가지 관점에서 이론적 분석과 실증적 검증을 병행하였다.
* 다양한 데이터셋 특성(정상성 여부, 변수 간 관계 등)에 따라 유연하게 대응할 수 있는 구조(Decoupled Time Embedding, Cos-Relation Matrix)를 갖추고 있다.

**한계 및 논의**:

* Cos-Relation Matrix의 임계값 $\theta$와 같은 하이퍼파라미터가 성능에 영향을 미치는데, 이를 자동으로 최적화하는 방법론은 제시되지 않았다.
* 대부분의 실험이 MSE/MAE 기반의 포인트 예측에 집중되어 있어, 불확실성을 고려한 확률적 예측(Probabilistic Forecasting)으로의 확장 가능성에 대한 논의가 부족하다.

## 📌 TL;DR

이 논문은 딥러닝 기반 시계열 예측에서 무시되었던 **시계열 고유 특성(정규화, 변수 간 관계, 입력 길이의 적절성)**을 분석하고 이를 반영한 **RTNet**을 제안한다. RTNet은 Weight Normalization, Cos-Relation Matrix, Causal Pyramid Network를 통해 예측 정확도를 획기적으로 높였으며, 특히 불필요한 계산량을 줄여 추론 효율성을 극대화하였다. 이 연구는 향후 시계열 모델 설계 시 단순한 구조 복제가 아닌 도메인 특성 기반의 설계가 필수적임을 시사한다.
