# Generalizable deep learning for photoplethysmography-based blood pressure estimation– A Benchmarking Study

Mohammad Moulaeifard, Peter H. Charlton and Nils Strodthoff (2025)

## 🧩 Problem to Solve

본 연구는 광혈류측정(Photoplethysmography, PPG) 신호를 이용한 혈압(Blood Pressure, BP) 추정 모델의 일반화 성능 문제를 해결하고자 한다. 최근 딥러닝 모델을 활용해 raw PPG 파형으로부터 혈압을 추론하려는 시도가 증가하고 있으나, 대부분의 기존 연구들은 학습 데이터와 테스트 데이터의 분포가 동일한 In-Distribution(ID) 테스트 세트에서만 평가되는 경향이 있다.

현실 세계의 응용 시나리오에서는 센서 하드웨어의 차이, 신호 품질, 피험자의 생리학적 특성 및 혈압 분포의 차이로 인해 테스트 데이터가 학습 데이터와 다른 분포를 갖는 Out-of-Distribution(OOD) 상황이 빈번하게 발생한다. 따라서 본 논문의 목표는 대규모 데이터셋인 PulseDB를 활용해 다양한 딥러닝 모델의 ID 및 OOD 일반화 성능을 벤치마킹하고, 도메인 적응(Domain Adaptation)을 통해 OOD 성능을 향상시킬 수 있는 방법을 탐색하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **종합적인 벤치마킹 수행**: 대규모 고품질 데이터셋인 PulseDB를 사용하여 최신 딥러닝 기반 시계열 분류 및 회귀 알고리즘들을 구현하고, 이들의 혈압 추정 성능을 체계적으로 비교 분석하였다.
2. **ID 및 OOD 일반화 분석**: PulseDB의 다양한 서브셋뿐만 아니라 4개의 외부 데이터셋을 통해 모델의 일반화 능력을 평가하였으며, 특히 레이블 분포(혈압 수치 분포)의 차이가 OOD 성능에 미치는 영향을 분석하였다.
3. **도메인 적응 기법 제안**: 소스 도메인과 타겟 도메인의 레이블 분포 차이를 이용하여 학습 샘플에 가중치를 부여하는 Importance-weighted Empirical Risk Minimization(ERM) 접근 방식을 제안하여 OOD 성능 개선 가능성을 입증하였다.

## 📎 Related Works

기존의 PPG 기반 혈압 추정은 주로 맥파 분석(Pulse Wave Analysis)과 같은 특징 기반 접근 방식에 의존하였으나, 이는 개인별 생리학적 변동성에 취약하고 잦은 보정(Calibration)이 필요하다는 한계가 있었다. 이를 극복하기 위해 특징을 자동으로 추출하는 머신러닝 및 딥러닝 기법이 도입되었으며, 데이터가 풍부한 환경에서는 높은 정확도를 보였다.

그러나 기존 벤치마크 연구들은 주로 동일 분포 내의 테스트(ID testing)에 집중하여, 실제 환경에서 발생할 수 있는 데이터 분포의 변화(Distribution Shift)를 간과하는 경향이 있었다. 최근 ECG나 EEG 신호 분석 분야에서는 도메인 일반화(Domain Generalization) 및 자기지도 학습(Self-supervised Learning)을 통해 OOD 문제를 해결하려는 시도가 있었으며, 본 연구는 이러한 직관을 PPG 기반 혈압 추정 문제에 적용하여 raw 시계열 데이터를 다루는 딥러닝 모델의 강건성을 평가하고자 한다.

## 🛠️ Methodology

### 1. 데이터셋 구성

학습을 위해 MIMIC-III와 VitalDB에서 추출된 대규모 데이터셋인 **PulseDB**를 사용하며, 다음과 같은 세 가지 시나리오의 서브셋을 생성하여 사용한다.

- **Calib**: 각 피험자의 데이터가 학습과 테스트 세트에 모두 포함되어, 환자별 특성에 적응하는 보정 기반 접근 방식을 평가한다.
- **CalibFree**: 학습과 테스트 세트가 피험자를 공유하지 않아, 완전히 새로운 환자에 대한 일반화 성능을 평가한다.
- **AAMI**: AAMI(Association for the Advancement of Medical Instrumentation) 표준을 준수하여 혈압 분포의 꼬리 부분(극단값)에 더 큰 비중을 둔 엄격한 일반화 시나리오이다.

외부 평가를 위해서는 Sensors, UCI, BCG, PPGBP 등 성격이 다른 4개의 외부 데이터셋을 OOD 테스트 세트로 활용한다.

### 2. 모델 아키텍처

본 연구에서는 다음과 같은 다양한 CNN 및 시퀀스 모델을 평가한다.

- **LeNet1D**: 단순한 피드포워드 CNN 구조이다.
- **XResNet1d (50, 101)**: Skip connection을 통해 그래디언트 흐름을 개선한 ResNet의 1차원 변형 모델이다.
- **Inception1D**: 다양한 크기의 커널 필터를 병렬로 사용하여 광범위한 특징 패턴을 캡처하는 구조이다.
- **S4 (Structured State Space Sequence)**: 긴 범위의 의존성(Long-range dependencies)을 효과적으로 포착할 수 있는 상태 공간 모델이다.

### 3. 학습 및 평가 절차

- **학습 설정**: AdamW 옵티마이저와 Mean Squared Error(MSE) 손실 함수를 사용하며, SBP(수축기 혈압)와 DBP(이완기 혈압)를 동시에 예측하는 두 개의 출력 노드를 구성한다.
- **성능 지표**:
  - **MAE (Mean Absolute Error)**: 예측값과 실제값의 평균 절대 오차를 측정한다.
    $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\text{Predicted}_i - \text{Reference}_i|$$
  - **MASE (Mean Absolute Scaled Error)**: 학습 세트의 혈압 중앙값(Median)을 기준으로 모델의 상대적 성능을 평가한다.
    $$\text{MASE} = \frac{\text{MAE}}{\text{MAE}_{\text{Baseline}}}$$

### 4. 도메인 적응: Importance Weighting

소스 도메인(학습)과 타겟 도메인(테스트)의 혈압 레이블 분포 차이를 줄이기 위해 샘플 가중치 방식을 도입한다.

- 레이블 분포를 정규화된 히스토그램($h$)으로 표현하고, 특정 빈(bin) $i$에 속하는 샘플의 가중치 $w_i$를 다음과 같이 정의한다.
$$w_i = \max\left(\tau, \frac{h_{\text{test},i}}{h_{\text{train},i}}\right) \quad \text{if } h_{\text{train},i} > 0, \text{ else } \tau$$
여기서 $\tau$는 가중치가 너무 작아져 샘플이 완전히 배제되는 것을 방지하는 하이퍼파라미터이다. 이 가중치를 손실 함수에 곱하여 타겟 도메인의 분포에 더 가깝게 학습하도록 유도한다.

## 📊 Results

### 1. 모델 성능 비교

- **최적 모델**: 전반적으로 **XResNet1d101**이 가장 일관되게 높은 성능을 보였으며, Inception-based 모델 또한 우수한 성능을 기록하였다. 반면, S4 모델은 ECG/EEG 분야에서의 성과와 달리 PPG 혈압 추정에서는 두드러진 성능 향상을 보이지 않았다.
- **시나리오별 차이**: Calib 시나리오에서는 모델이 특정 환자의 패턴을 기억할 수 있어 가장 낮은 MAE를 기록하였으며, AAMI 시나리오에서는 레이블 분포의 불일치로 인해 가장 높은 오차가 발생하였다.

### 2. ID vs OOD 일반화

- **ID 성능의 한계**: 동일 데이터셋 내에서 평가한 ID 성능은 매우 낙관적이지만, 외부 데이터셋으로 평가했을 때 성능이 급격히 저하된다. 이는 ID 결과만으로 모델의 실제 일반화 능력을 판단할 수 없음을 시사한다.
- **데이터셋 영향**: VitalDB 기반 모델이 MIMIC 기반 모델보다 외부 데이터셋에 대해 더 좋은 일반화 성능을 보였다. 특히 SBP 분포의 유사성(Earth Mover's Distance, EMD로 측정)과 OOD MAE 사이에 강한 상관관계가 있음이 확인되었다.

### 3. 도메인 적응의 효과

- **성능 개선**: Importance weighting을 적용했을 때, 전체 케이스의 58%에서 OOD 성능이 향상되었다.
- **특이 사항**: 특히 MIMIC 기반 모델과 AAMI 시나리오에서 성능 향상 폭이 컸으며, 일부 경우(AAMI Vital 모델)에서는 외부 데이터셋에서 ID 수준의 성능에 근접하는 결과를 얻었다.

## 🧠 Insights & Discussion

### 1. 주요 강점 및 발견

본 연구는 PPG 기반 혈압 추정에서 **레이블 분포의 불일치(Label Distribution Shift)**가 도메인 간 성능 저하의 핵심 요인임을 밝혀냈다. 단순한 가중치 조절만으로도 OOD 성능을 유의미하게 개선할 수 있음을 보여주었으며, 이는 타겟 도메인의 대략적인 통계 정보(레이블 분포)만으로도 모델을 최적화할 수 있다는 실무적 가능성을 제시한다.

### 2. 한계 및 비판적 해석

- **임상적 유효성 부족**: IEEE 표준에 따르면 임상적으로 사용 가능하려면 MAE가 $7\text{mmHg}$ 미만이어야 한다(Grade A-B). 하지만 본 연구의 모든 모델은 평균적으로 이 기준을 충족하지 못하는 Grade D 수준에 머물러 있어, 실제 의료 기기 적용까지는 여전히 큰 간극이 존재한다.
- **단순한 적응 방식**: 제안된 중요도 가중치 방식은 레이블 분포만을 고려하며, 센서의 특성, 신호 품질, 피험자의 인구통계학적 특성 등 다른 중요한 도메인 차이점은 해결하지 못한다.

### 3. 향후 연구 방향

저자들은 향후 연구로 사전 학습된 파운데이션 모델(Foundation Models)의 활용, 임상 메타데이터의 통합, 그리고 타겟 분포를 모르는 상태에서의 강건성을 높이기 위한 연구가 필요함을 언급하였다.

## 📌 TL;DR

본 논문은 PPG 기반 혈압 추정 모델의 OOD 일반화 성능을 체계적으로 벤치마킹하고, 레이블 분포의 차이를 이용한 **Importance Weighting** 기법을 통해 이를 개선하는 방법을 제시하였다. **XResNet1d101**이 가장 우수한 성능을 보였으며, ID 평가 결과가 OOD 성능을 보장하지 않는다는 점을 경고하며 외부 데이터셋 검증의 중요성을 강조하였다. 이 연구는 향후 혈압 추정 모델이 단순한 정확도를 넘어 실제 임상 환경의 다양성에 적응하기 위한 도메인 적응 연구의 기초를 제공한다.
