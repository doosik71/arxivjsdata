# Estimating blood pressure trends and the nocturnal dip from photoplethysmography

Mustafa Radha et al. (2018)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 기존의 혈압(Blood Pressure, BP) 측정 방식이 가진 침습성과 불편함이다. 현재 혈압 측정의 골드 표준인 24시간 활동 혈압 측정기(Ambulatory BP Monitor, ABPM)는 가압 커프(Inflatable cuff)를 사용하는데, 이는 특히 수면 중에 반복적으로 팽창하며 소음을 발생시키고 혈류를 차단하여 수면을 방해한다.

특히 수축기 혈압(Systolic Blood Pressure, SBP)의 야간 강하(Nocturnal dip, 밤 시간대에 혈압이 낮아지는 현상)는 심혈관 질환의 위험을 예측하는 매우 중요한 임상적 지표이다. 야간 SBP가 $5\text{ mmHg}$ 감소할 때 심혈관 위험이 $17\%$ 감소한다는 보고가 있을 만큼 그 중요성이 크지만, 측정 과정의 불편함으로 인해 실제 임상 현장에서 널리 적용되기 어렵다. 따라서 본 논문은 손목 착용형 광혈류 측정(Photoplethysmography, PPG) 센서와 딥러닝 모델을 이용하여, 일상생활(Free-living) 환경에서 비침습적으로 혈압 추세와 야간 SBP 강하를 추정하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 절대적인 혈압 수치를 맞추는 것이 아니라, 개인의 평균 혈압 대비 상대적인 변화량인 '혈압 추세(Trends)'를 예측하는 것이다. 야간 SBP 강하 역시 상대적인 측정값이므로, 상대적 혈압 변화만 정확히 추적할 수 있다면 임상적으로 유의미한 정보를 추출할 수 있다는 직관에 기반한다.

주요 기여 사항은 다음과 같다.

1. 실제 일상생활 환경에서 수집된 대규모 데이터셋(103명, 226일)을 사용하여 비침습적 혈압 추정의 유효성을 검증하였다.
2. PPG의 파형 분석(Morphology)과 심박 변이도(Heart Rate Variability)를 결합한 특징 추출 파이프라인을 제안하였다.
3. 시계열 데이터의 특성을 반영할 수 있는 LSTM(Long Short-Term Memory) 네트워크가 단순한 머신러닝 모델보다 야간 SBP 강하 추정에 훨씬 효과적임을 입증하였다.

## 📎 Related Works

기존의 비침습적 혈압 측정 연구는 크게 두 가지 방향으로 진행되었다.

1. **생리학적 모델 (Physiological Models):** 주로 Moens-Korteweg 방정식을 기반으로 맥파 전달 시간(Pulse Arrival Time, PAT)을 측정하여 혈압을 추정한다. 하지만 PAT 기반 방식은 체위 변화(Posture change)에 민감하여 일상생활 환경에서의 정확도가 낮으며, ECG와 PPG 두 개의 센서가 필요하여 착용감이 떨어진다는 한계가 있다.
2. **머신러닝 모델 (Machine Learning Models):** PPG 파형의 형태학적 분석(Morphology analysis)을 통해 혈압을 예측하려는 시도가 있었다. Random Forest나 MLP 등의 모델이 사용되었으나, 대부분의 연구가 통제된 실험실 환경(Controlled lab setting)에서 짧은 시간 동안 측정된 데이터만을 사용하였다. 실험실 데이터는 외부 타당성(External validity)이 낮아 실제 일상생활에 그대로 적용하기 어렵다.

본 논문은 이러한 한계를 극복하기 위해 일상생활 데이터셋을 구축하고, 단일 시점의 특징이 아닌 시퀀스 데이터를 처리할 수 있는 LSTM 모델을 도입하여 차별성을 두었다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

시스템은 **[데이터 수집 $\rightarrow$ 특징 추출 $\rightarrow$ 최적화 및 전처리 $\rightarrow$ 모델 학습 $\rightarrow$ SBP 강하 계산]** 순으로 구성된다.

### 2. 주요 구성 요소 및 특징 추출

- **활동 특징 (Activity features):** 3축 가속도계를 사용하여 사용자가 휴식 상태인지 여부를 판단하며, 이는 야간 SBP 강하 계산을 위한 수면/각성 구간 분할에 사용된다.
- **심박 변이도 (HRV):** PPG 신호에서 개별 심박을 추출하고, 다중 스케일 샘플 엔트로피(Multi-scale sample entropy) 알고리즘을 적용하여 스케일 6~10 범위의 특징을 추출한다.
- **PPG 형태학적 특징 (Morphology features):**
  - 파형의 피크(Peak) 및 미분 값 분석.
  - **가우시안 혼합 모델(Gaussian Mixture Model, GMM):** 단일 PPG 펄스를 4개의 가우시안 함수로 분해하여 각 성분의 진폭, 너비, 타이밍을 특징으로 사용한다.
  - 신호 처리 기법: Shannon entropy, Kaiser-teager energy 등을 통해 신호의 불규칙성과 주파수 특성을 정량화한다.

### 3. 전처리 및 최적화

특징들은 개인별 특성(Person-specific nature)이 강하므로, 이를 제거하기 위해 Z-score, Min-Max, Boxcox, Quantile, Winsor 정규화 중 SBP와의 상관관계가 가장 높은 방법을 개별적으로 선택하여 적용하였다. 이후 Butterworth 필터나 Rolling mean 필터 등을 통해 노이즈를 제거하였다.

### 4. 학습 모델 및 손실 함수

모델은 상대적 혈압(Relative BP)을 예측하도록 설계되었으며, 입력 특징에서 개인별 일일 평균을 뺀 값을 사용한다.

- **비교 모델:** Linear Regression, Random Forest, Dense Network (MLP), LSTM Network.
- **LSTM 아키텍처:** $\text{Dense Layer (32)} \rightarrow \text{LSTM Layer (8/16/32 cells)} \rightarrow \text{Dense Layer (Output)}$ 순으로 구성된다.
- **손실 함수:** 기본적으로 평균 제곱 오차(Mean Squared Error, MSE)를 사용하되, 정규 분포 특성상 발생하는 이상치(Outlier) 무시 현상을 방지하기 위해 다음과 같이 손실을 증폭시켰다.
  $$\text{Loss} = \text{MSE} \times |\text{SBP}_{\text{true}} - \text{Mean SBP}_{\text{train}}|$$
  즉, 평균 혈압에서 멀리 떨어진 값(변동성이 큰 값)일수록 손실 가중치를 높여 모델이 극단적인 값의 변화에도 민감하게 반응하도록 유도하였다.

### 5. 추론 및 SBP 강하 계산

학습된 모델을 통해 일일 혈압 추세를 예측한 후, 가속도계 기반의 수면-각성 분류기를 통해 구간을 나눈다. SBP 강하(Dip)는 다음과 같이 계산된다.
$$\text{SBP dip} = \text{Mean SBP}_{\text{wake}} - \text{Mean SBP}_{\text{sleep}}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** 103명의 건강한 성인, 총 226일 분량의 데이터.
- **분할:** 82명 데이터로 학습, 21명 데이터로 테스트 수행.
- **지표:** RMSE(Root Mean Squared Error), Pearson 상관계수.

### 2. 주요 결과

- **혈압 추적 (BP Tracking):** 상대적 SBP와 DBP를 추적하는 성능은 모든 모델(Random Forest, Dense, LSTM)이 유사한 수준의 RMSE와 상관관계를 보였다.
- **SBP 강하 추정 (SBP Dip Estimation):**
  - **LSTM (32 cells)** 모델이 가장 우수한 성능을 보였다.
  - **RMSE:** $3.12 \pm 2.20\ \Delta\text{mmHg}$
  - **상관계수:** $0.69\ (p = 3 \times 10^{-5})$
  - 반면, 인구통계학적 정보(나이, 몸무게 등)나 심박수(Heart rate)만을 이용한 베이스라인 모델은 실제 SBP 강하와 유의미한 상관관계를 보이지 않았다.

### 3. 분석 결과 (Bland-Altman Analysis)

Random Forest 모델은 SBP 추적 자체의 RMSE는 낮았으나, SBP 강하 추정에서는 과대평가 경향과 진폭 의존적 오류(Amplitude-dependent error)가 나타났다. 반면 LSTM 모델은 편향이 적고 진폭 변화에 더 일관된 예측력을 보여, 평균에서 벗어난 변동 값을 포착하는 능력이 더 뛰어남이 확인되었다.

## 🧠 Insights & Discussion

### 1. 시퀀스 모델링의 중요성

본 연구는 혈압의 '추세'를 예측하는 데 있어 단순한 비선형 모델(Random Forest)보다 시계열적 구조를 학습할 수 있는 LSTM이 훨씬 유리함을 보여주었다. SBP 강하를 정확히 계산하려면 단순히 방향성만 맞추는 것이 아니라 변화의 크기(Magnitude)를 정확히 예측해야 하는데, LSTM의 recurrent 구조가 이러한 시간적 의존성을 효과적으로 포착한 것으로 해석된다.

### 2. 실험실 환경 vs 실제 환경

본 논문에서 보고된 RMSE가 기존 실험실 연구들보다 높게 나타난 점은 주목할 만하다. 이는 통제된 환경에서 얻은 결과가 실제 일상생활(Free-living)로 확장될 때 외부 타당성이 떨어진다는 것을 시사한다. 실제 환경에서는 움직임으로 인한 아티팩트(Motion artifacts)와 체위 변화 등이 강력한 노이즈로 작용하기 때문이다.

### 3. 한계점 및 향후 과제

- **움직임 아티팩트:** 손목의 움직임이 많을 경우 PPG 데이터 품질이 저하되어 많은 데이터가 버려지는 문제가 발생한다. 이를 해결하기 위해 가슴이나 머리 등 움직임이 적은 부위의 센서 적용을 고려할 수 있다.
- **그라운드 트루스의 한계:** ABPM 역시 수면 중 불편함을 유발하며 연속 측정이 불가능하다는 한계가 있다. 침습적 카테터 데이터나 다른 연속 측정 장비의 데이터를 전이 학습(Transfer learning)에 활용하는 방안이 제시되었다.

## 📌 TL;DR

본 논문은 손목 착용형 PPG 센서와 LSTM 네트워크를 이용하여 일상생활 중 비침습적으로 혈압 추세와 야간 SBP 강하를 추정하는 방법을 제안하였다. 실험 결과, LSTM 모델이 SBP 강하 추정에서 가장 높은 상관관계($0.69$)와 낮은 오차($3.12\text{ mmHg}$)를 기록하며 단순 머신러닝 모델보다 우월함을 입증하였다. 이 연구는 고위험군 환자의 야간 혈압 모니터링을 위한 저비용, 고효율의 스크리닝 도구 개발 가능성을 열어주었다는 점에서 큰 의미가 있다.
