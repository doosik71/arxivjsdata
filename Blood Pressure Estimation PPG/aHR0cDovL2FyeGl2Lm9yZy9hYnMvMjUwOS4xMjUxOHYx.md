# Generalizable Blood Pressure Estimation from Multi-Wavelength PPG Using Curriculum-Adversarial Learning

Zequan Liang, Ruoyu Zhang, Wei Shao, Mahdi Pirayesh Shirazi Nejad, Ehsan Kourkchi, Setareh Rafatirad, Houman Homayoun (2025)

## 🧩 Problem to Solve

본 연구는 광혈류측정장치(Photoplethysmography, PPG) 신호를 이용한 비침습적 혈압(Blood Pressure, BP) 추정의 정확도와 일반화 성능을 높이는 것을 목표로 한다. 기존의 PPG 기반 혈압 추정 연구들은 다음과 같은 두 가지 주요한 한계점을 가지고 있다.

첫째, 대부분의 연구가 단일 파장(Single-wavelength) PPG 신호에만 의존하여, 다양한 파장-대역의 신호가 제공하는 상호 보완적인 생리학적 정보를 충분히 활용하지 못하고 있다. 둘째, 데이터 분할 방식의 부적절함으로 인해 데이터 누수(Data Leakage) 문제가 발생하고 있다. 특히 학습 데이터와 테스트 데이터에 동일한 피험자의 데이터가 섞여 들어가는 세그먼트 단위의 무작위 분할(Random segment-level splitting)을 사용할 경우, 모델이 피험자의 개인적 특성을 암기하여 성능이 과하게 높게 측정되는 경향이 있다.

따라서 본 논문은 다파장 PPG 신호를 활용하고 엄격한 피험자 단위 분할(Subject-level splitting)을 적용함으로써, 실제 환경에서도 일반화 가능한 혈압 추정 프레임워크를 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 다파장 PPG의 보완적 정보를 추출하는 구조에 **Curriculum-Adversarial Learning** 전략을 결합하는 것이다.

1. **Subject-Invariant Feature Learning**: Domain-Adversarial Neural Network(DANN) 구조를 도입하여, 모델이 피험자의 개별적 정체성(Subject Identity)을 구분하지 못하게 함으로써 특정 개인에게 종속되지 않는 일반적인 특징을 학습하도록 유도한다.
2. **Coarse-to-Fine Curriculum Learning**: 모델이 처음부터 정밀한 수치 회귀(Regression)를 학습하는 대신, 고혈압 여부를 분류하는 거친 단계의 분류(Classification) 작업부터 시작하여 점진적으로 정밀한 혈압 추정 작업으로 전환하는 커리큘럼 학습 방식을 적용한다.
3. **Multi-Wavelength Fusion**: 4가지 다른 파장의 PPG 신호를 독립적인 CNN 채널로 처리하고, Attention 메커니즘을 통해 각 채널의 중요도를 동적으로 반영하여 융합한다.

## 📎 Related Works

논문에서는 기존의 PPG 기반 혈압 추정 방식들을 언급하며 본 연구와의 차별점을 제시한다.

- **기존 접근 방식**: A-BiLSTM, MLP, CNN1D 등의 모델이 제안되었으나, 많은 경우 단일 파장을 사용하거나 데이터 분할 과정에서의 누수 문제로 인해 실제 일반화 성능이 과대평가되는 경향이 있었다.
- **Multi-CNN 기반 구조**: 본 연구는 Multi-CNN 구조를 백본(Backbone)으로 사용하지만, 여기에 커리큘럼 학습과 적대적 학습(Adversarial Learning)을 추가하여 일반화 성능을 극대화했다는 점에서 차별화된다. 특히 기존의 ACNN-BiLSTM 모델 등이 데이터 분할 전략을 명시하지 않아 발생한 신뢰성 문제를 피험자 단위 분할을 통해 정면으로 해결하고자 하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 데이터 전처리
본 시스템은 4가지 파장(660nm, 730nm, 850nm, 940nm)의 PPG 신호를 입력으로 받는다. 각 신호는 다음과 같은 전처리 과정을 거친다.
- 0.5~8Hz 대역의 Band-pass filter 적용 및 Z-score 정규화.
- 시간적 동역학을 포착하기 위해 1차 및 2차 미분값(Derivatives) 계산.
- 각 신호와 미분값에서 피크(Peak) 정보를 추출하기 위해 상단 포락선(Upper Envelope)을 계산.

### 2. 아키텍처 및 특징 추출
각 PPG 채널은 두 개의 병렬 CNN을 통과한다. 하나는 정규화된 PPG와 그 미분값들을 처리하고, 다른 하나는 포락선 신호를 처리한다. 결과적으로 채널당 6개의 입력 스트림이 구성되며, 최종적으로 FC 레이어를 통해 각 채널의 특징 벡터 $Z_i$가 생성된다.

이후 **Multi-Channel Attention Fusion** 레이어를 통해 각 채널에 소프트 웨이트 $w_i$를 할당하고 가중 합(Weighted sum)을 구하여 융합된 특징을 생성한다. 이 융합 특징은 다음 세 가지 헤드로 전달된다.
- **Regressor**: 수축기 혈압(SBP)과 이완기 혈압(DBP) 수치를 추정.
- **Classifier**: 고혈압 여부(Binary)를 예측.
- **Adversarial Discriminator**: 해당 신호가 어떤 피험자의 것인지(Subject ID)를 식별.

### 3. 학습 전략 및 손실 함수

#### Domain-Adversarial Training
피험자 간의 변동성을 줄이기 위해 Gradient Reversal Layer(GRL)를 사용한다. 판별기(Discriminator)는 피험자를 구분하려 하지만, GRL은 역전파 과정에서 그래디언트를 반전시켜 특징 추출기가 피험자를 구분할 수 없는 '피험자 불변 특징(Subject-invariant features)'을 학습하게 만든다. 이때 손실 함수는 Categorical Cross-Entropy(CE)를 사용한다.
$$L_{adv} = CE(\hat{y}_{sbj}, y_{sbj})$$

#### Curriculum Learning
학습은 '고혈압 분류 $\rightarrow$ 혈압 회귀' 순서로 진행된다.
- **1단계 (분류)**: Binary Cross-Entropy(BCE) 손실 함수를 사용하여 고혈압 여부를 먼저 학습한다.
$$L_{cls} = -[c \cdot \log(\hat{c}) + (1-c) \cdot \log(1-\hat{c})]$$
- **2단계 (회귀)**: 분류 성능이 안정화되면 Mean Squared Error(MSE)를 통해 정밀한 SBP, DBP 값을 추정한다.
$$L_{reg} = \frac{1}{2} [(\hat{y}_{sbp} - y_{sbp})^2 + (\hat{y}_{dbp} - y_{dbp})^2]$$

#### 최종 통합 손실 함수
최종 손실 함수는 다음과 같이 정의되며, $\lambda_1$은 학습 에폭에 따라 0에서 1로 점진적으로 증가하여 학습의 중심을 분류에서 회귀로 이동시킨다.
$$L = \lambda_1 L_{reg} + (1 - \lambda_1) L_{cls} - \lambda_2 L_{adv}$$

## 📊 Results

### 실험 설정
- **데이터셋**: 180명의 피험자가 포함된 공개 다파장 PPG 데이터셋을 사용하였다.
- **데이터 분할**: 피험자 단위로 학습/테스트 세트를 4:1 비율로 엄격히 분리하여 데이터 누수를 방지하였다.
- **평가 지표**: Mean Absolute Error(MAE)와 British Hypertension Society(BHS) 프로토콜에 따른 BHS-5, 10, 15(오차 범위 내 샘플 비율)를 사용하였다.

### 주요 결과
- **다파장 융합의 효과**: 단일 파장 모델(예: 660nm 또는 940nm)보다 모든 채널을 융합한 모델이 가장 낮은 MAE와 높은 BHS 점수를 기록하였다. 이는 다양한 파장의 신호가 상호 보완적인 정보를 제공함을 입증한다.
- **기존 방법론과의 비교**: 제안된 방법은 A-BiLSTM, MLP, CNN1D, Multi-CNN 등의 베이스라인 모델보다 우수한 성능을 보였다.
  - **SBP MAE**: $14.2\text{ mmHg}$
  - **DBP MAE**: $6.4\text{ mmHg}$
- **Ablation Study**: 분류기(cls)만 추가하거나 판별기(adv)만 추가했을 때보다, 두 구성 요소를 모두 포함했을 때 가장 낮은 MAE를 달성하여 커리큘럼 학습과 적대적 학습 모두가 일반화 성능 향상에 기여함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 엄격한 피험자 단위 분할을 적용함으로써 기존 연구들보다 수치상의 에러는 높게 나타날 수 있으나, 이는 오히려 실제 환경에서의 성능을 더 정확하게 반영한 현실적인 평가라고 주장한다. 

특히 고혈압 분류라는 '쉬운 문제'에서 혈압 회귀라는 '어려운 문제'로 단계적으로 접근하는 커리큘럼 학습과, 개인의 고유 특성을 제거하는 적대적 학습의 결합이 PPG 기반 혈압 추정의 고질적인 문제인 '개인차(Inter-subject variability)' 문제를 효과적으로 완화했음을 알 수 있다. 다만, 매우 극단적인 혈압 범위나 신호 품질이 낮은 상황에서의 정확도 개선은 여전히 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

본 연구는 다파장 PPG 신호를 이용하여 일반화 가능한 혈압 추정 프레임워크를 제안하였다. **피험자 단위 데이터 분할**을 통해 데이터 누수를 방지하였으며, **Domain-Adversarial Training**으로 피험자 불변 특징을 학습하고, **Curriculum Learning**을 통해 분류에서 회귀로 단계적 학습을 수행함으로써 성능을 높였다. 실험 결과 SBP 14.2mmHg, DBP 6.4mmHg의 MAE를 달성하며 기존 모델 대비 우수한 일반화 성능을 입증하였다. 이 연구는 향후 웨어러블 기기를 통한 실시간 고혈압 스크리닝 및 장기적 혈압 모니터링 시스템 구축에 중요한 기초가 될 것으로 보인다.