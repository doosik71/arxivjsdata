# Leveraging Latent Causal Relationships Among Web Services for Traffic Prediction

Chang Tian, Mingzhe Xing, Zenglin Shi, Matthew Blaschko, Yinliang Yue, and Marie-Francine Moens (2025)

## 🧩 Problem to Solve

본 논문은 웹 서비스 트래픽 예측의 정확도를 높이는 문제를 해결하고자 한다. 웹 서비스 트래픽은 사용자 행동의 다양성으로 인해 시간에 따라 매우 빈번하고 급격한 변동성을 보이며, 이는 동적 자원 확장(dynamic resource scaling), 이상 탐지(anomaly detection), 사기 탐지(fraud detection)와 같은 시스템 운영 작업에 결정적인 영향을 미친다.

기존의 통계적 방법론이나 딥러닝 기반의 시계열 예측 모델들은 주로 개별 서비스의 과거 트래픽 데이터에서 특징을 추출하는 데 집중하였다. 그러나 이러한 접근 방식은 서로 다른 웹 서비스들 사이에 존재하는 잠재적인 인과 관계(latent causal relationships)를 간과한다는 한계가 있다. 예를 들어, 여가용 서비스(Netflix)의 사용량 증가가 업무용 서비스(Outlook)의 사용량 감소로 이어지는 것과 같은 서비스 간의 상관관계를 활용한다면 예측 성능을 더욱 향상시킬 수 있다. 따라서 본 논문의 목표는 서비스 간의 인과 관계를 추출하여 이를 시계열 예측 모델에 통합하는 CCMPlus 모듈을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 생태계의 인과 관계 분석 방법론인 Convergent Cross Mapping (CCM) 이론을 딥러닝 아키텍처에 접목하는 것이다.

주요 기여 사항은 다음과 같다.

- **CCMPlus 모듈 제안**: 웹 서비스 간의 잠재적 인과 관계를 포착하여 특징 표현(feature representation)으로 생성하는 신경망 모듈을 설계하였다.
- **범용적 통합 구조**: CCMPlus 모듈은 특정 모델에 종속되지 않고, 기존의 다양한 시계열 예측 모델(Backbone Time Series Model)과 원활하게 통합되어 성능을 일관되게 향상시킬 수 있다.
- **멀티 매니폴드 임베딩(Multi-manifold Embedding)**: 전문가의 주관적인 하이퍼파라미터 설정에 의존하던 기존 CCM의 한계를 극복하기 위해, 학습 가능한 파라미터를 통한 다중 매니폴드 공간을 구축하여 인과 관계를 더 정교하게 포착하였다.
- **실증적 검증**: Microsoft Azure, Alibaba Group, Ant Group의 실제 데이터셋을 통해 제안 방법론이 기존 SOTA 모델들보다 MSE(Mean Squared Error)와 MAE(Mean Absolute Error) 측면에서 우수함을 증명하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계

1. **통계적 방법 및 머신러닝**: Moving Average, ARIMA 등의 통계적 방법은 단순하고 해석 가능하지만, 비선형 데이터나 다차원 데이터 처리에 취약하다. SVR, Logistic Regression 등의 머신러닝 방법은 이를 개선했으나 표현력의 한계로 정확도가 낮다.
2. **딥러닝 기반 모델**: LSTM, GRU를 활용한 모델이나 최근의 Transformer 기반 모델(iTransformer, TimesNet 등)은 시계열 데이터의 복잡한 패턴을 잘 포착한다. 특히 TimesNet은 다중 주기성(multi-periodicity)을 분석하고, iTransformer는 변수 간의 상관관계를 효과적으로 모델링한다.

### 차별점

기존의 SOTA 모델들은 시계열 데이터 내부의 시간적 패턴(temporal patterns) 분석에만 집중하며, 서로 다른 시계열 변수 간의 외부적 인과 관계를 명시적으로 모델링하지 않는다. 반면, 본 논문은 CCM 이론을 통해 서비스 간의 인과 관계를 수치화하고 이를 특징 벡터로 변환하여 모델에 입력함으로써, 내부 패턴과 외부 인과 관계를 동시에 고려한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

전체 파이프라인은 크게 **CCMPlus 모듈**과 **Backbone Time Series Model (BTSM)**의 결합으로 구성된다.

1. 입력된 원시 트래픽 데이터는 CCMPlus 모듈을 통해 인과 관계가 반영된 특징 표현($e_{h_{ccm}}$)으로 변환된다.
2. 동시에 BTSM(예: iTransformer, TimesNet)을 통해 시간적 특징 표현($e_{h_{ts}}$)이 추출된다.
3. 두 특징 표현을 결합(Concatenation)하여 MLP(Multi-Layer Perceptron)에 통과시킨 후 최종 트래픽 값을 예측한다.

### CCMPlus 모듈 상세 설명

#### 1. Multi-Manifold Embedding (다중 매니폴드 임베딩)

기존 CCM은 시차 $\tau$와 임베딩 차원 $E$를 전문가가 설정해야 하지만, CCMPlus는 이를 학습 가능한 형태로 확장한다.

- 입력 시계열 $\hat{X}$에 날짜 특징을 결합하여 임베딩 $X \in \mathbb{R}^{N \times L \times C}$를 생성한다.
- $\text{Conv1D}$ 레이어를 사용하여 다양한 커널 크기($E_i$)와 팽창 계수(dilation, $\tau_i$)를 적용함으로써 여러 개의 쉐도우 매니폴드(shadow manifold) 공간을 구축한다.
- 이를 통해 각 서비스의 상태를 나타내는 $D$차원의 좌표 벡터 $x^{(k)}$를 생성한다.

#### 2. 인과 관계 추정 (Estimation within Multi-Manifold Space)

서비스 $X$의 매니폴드 $M_x$를 이용하여 서비스 $Y$의 값 $y(k)$를 예측함으로써 인과 관계를 확인한다.

- $M_x$ 공간에서 현재 시점 $x^{(k)}$와 가장 가까운 $D+1$개의 이웃을 찾는다.
- 각 이웃과의 유클리드 거리 기반 가중치 $w_{is}$를 계산하며, 가중치 식은 다음과 같다.
  $$u_{is} = \exp\left(-\frac{d[x^{(k)}, x^{(t_{is})}]}{d[x^{(k)}, x^{(t_{i1})}]}\right), \quad w_{is} = \frac{u_{is}}{\sum_{j=1}^{D+1} u_{ij}}$$
- 가중 평균을 통해 예측값 $\hat{y}(k)|M_x$를 산출한다.
  $$\hat{y}(k)|M_x = \sum_{s=1}^{D+1} w_{is} y(t_{is})$$

#### 3. Momentum-Updated Correlation Matrix (모멘텀 업데이트 상관 행렬)

예측값과 실제값의 상관계수 $r$을 계산하여 인과 관계 행렬 $M \in \mathbb{R}^{N \times N}$을 생성한다. 학습의 안정성을 위해 모멘텀 업데이트 메커니즘을 적용한다.
$$M_z = (1 - \lambda) \cdot M + \lambda \cdot M_{z-1}$$
이후 Softmax 함수를 통해 정규화된 인과 관계 행렬 $cM_z$를 얻는다.

#### 4. 인과 관계 강화 특징 표현 (Causality Enhanced Representation)

최종적으로 인과 관계 점수를 가중치로 사용하여 쉐도우 매니폴드 임베딩 $\tilde{X}_{ccm}$을 가중 합산한다.
$$h_{ccm}^{(i)} = cM_z^{(i)} \cdot \tilde{X}_{ccm}$$
모든 매니폴드 공간에서 얻은 결과의 평균을 내어 최종 CCMPlus 표현 $e_{h_{ccm}}$을 생성한다.

### 학습 및 최적화

- **손실 함수**: Mean Squared Error (MSE)를 사용하여 모델을 최적화한다.
- **추론 절차**: $e_{h_{ccm}}$과 $e_{h_{ts}}$를 결합하여 MLP를 통해 최종 예측값 $\hat{x}$를 출력한다.
  $$\hat{x} = \text{MLP}(e_{h_{ccm}} \parallel e_{h_{ts}})$$

## 📊 Results

### 실험 설정

- **데이터셋**: Ant Group (113개 서비스), Microsoft Azure (1,000개 서비스), Alibaba Group (1,000개 서비스).
- **비교 대상(Baselines)**: Llama3, TimeLLM (LLM 기반), MagicScaler, OptScaler (전용 모델), TimeMixer, iTransformer, TimesNet (일반 시계열 모델).
- **평가 지표**: MSE, MAE.
- **예측 입도($\alpha$)**: 1분, 5분, 15분, 30분 단위로 테스트.

### 주요 결과

- **정량적 성능**: CCMPlus를 iTransformer 및 TimesNet에 결합한 모델(CCM+iTransformer, CCM+TimesNet)이 모든 데이터셋에서 가장 낮은 MSE와 MAE를 기록하였다. 특히 5분 및 30분 입도에서 성능 향상이 뚜렷하게 나타났다.
- **입도별 분석**: 입도가 세밀할수록(1분) 성능 향상 폭이 줄어드는데, 이는 매우 짧은 시간 단위의 데이터는 인과 관계를 포착하기 위한 충분한 정보가 축적되지 않았기 때문으로 분석된다. 그럼에도 불구하고 여전히 baseline보다 우수한 성능을 보였다.
- **Ablation Study**: BTSM(백본 모델), MME(멀티 매니폴드 임베딩), CER(인과 강화 표현) 각각을 제거했을 때 성능이 하락함을 확인하였으며, 특히 CER이 인과 관계를 포착하여 예측력을 높이는 데 핵심적인 역할을 함을 증명하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 단순한 시계열 패턴 분석을 넘어, 서비스 간의 '인과 관계'라는 외부 요인을 딥러닝 모델에 성공적으로 통합하였다. 특히 기존 CCM 이론의 하이퍼파라미터 의존성 문제를 $\text{Conv1D}$ 기반의 멀티 매니폴드 구조로 해결하여 학습 가능하게 만든 점이 기술적 성취라고 판단된다. 또한, 특정 백본 모델에 국한되지 않는 모듈형 구조를 가져 범용성이 매우 높다.

### 한계 및 논의 사항

1. **데이터 입도 영향**: 1분 단위의 매우 짧은 입도에서는 인과 관계 추출의 효율이 떨어진다. 이는 인과 관계가 발현되기까지 일정 시간의 시차가 필요하다는 이론적 가정과 일치하지만, 초단기 예측을 위한 개선 방안이 필요하다.
2. **계산 복잡도**: $N \times N$ 크기의 상관 행렬을 계산하고 $k$-nearest neighbors를 찾는 과정이 포함되어 있어, 서비스의 개수 $N$이 매우 커질 경우 계산 비용이 급격히 증가할 가능성이 있다. 이에 대한 확장성(Scalability) 논의가 부족하다.
3. **인과 관계의 해석**: Case Study를 통해 일부 서비스 간의 인과 관계를 시각화하였으나, 실제로 어떤 도메인적 특성 때문에 이러한 인과 관계가 발생하는지에 대한 심도 있는 분석은 제시되지 않았다.

## 📌 TL;DR

본 논문은 웹 서비스 트래픽 예측을 위해 생태계 인과 분석 이론인 CCM을 확장한 **CCMPlus 모듈**을 제안한다. 이 모듈은 서비스 간의 잠재적 인과 관계를 다중 매니폴드 공간에서 추출하여 특징 벡터로 변환하며, 이를 기존 SOTA 시계열 모델(iTransformer, TimesNet 등)과 결합하여 예측 정확도를 유의미하게 향상시킨다. 이 연구는 단순한 시간적 패턴 분석을 넘어 서비스 간의 상호작용을 모델링하는 것이 실제 클라우드 트래픽 예측에 매우 중요하다는 것을 입증하였다.
