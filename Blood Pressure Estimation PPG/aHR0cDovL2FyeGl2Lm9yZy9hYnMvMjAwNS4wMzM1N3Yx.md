# Estimating Blood Pressure from Photoplethysmogram Signal and Demographic Features using Machine Learning Techniques

Moajjem Hossain Chowdhury, Md Nazmul Islam Shuzan, Muhammad E.H. Chowdhury, Zaid B Mahbub, M. Monir Uddin, Amith Khandakar, Mamun Bin Ibne Reaz (2020)

## 🧩 Problem to Solve

본 연구에서 해결하고자 하는 핵심 문제는 비침습적이고 연속적인 혈압(Blood Pressure, BP) 측정 시스템을 구축하는 것이다. 고혈압은 심장마비, 뇌졸중, 신장 질환 및 치매와 같은 심각한 건강 합병증을 유발할 수 있으므로 정기적인 모니터링이 필수적이다.

현재 표준으로 사용되는 커프(Cuff) 기반 측정 방식은 간헐적인 측정만 가능하며, 측정 시 사용자가 느끼는 불편함이 크고 수면 중 측정이 어렵다는 단점이 있다. 반면, 침습적인 동맥관 관리 방식은 연속 측정이 가능하지만 감염 위험이 존재한다. 따라서 본 논문은 이러한 한계를 극복하기 위해 광혈류측정(Photoplethysmogram, PPG) 신호와 인구통계학적 특징(Demographic Features)을 결합하여, 커프 없이도 연속적으로 혈압을 추정할 수 있는 머신러닝 기반의 시스템을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 PPG 신호의 단순한 분석을 넘어, 시간-주파수 영역의 특징과 사용자의 신체적 특성을 통합적으로 활용하여 추정 정확도를 극대화하는 것이다.

주요 기여 사항은 다음과 같다.
1. **다차원 특징 추출**: PPG 신호에서 시간 도메인(t-domain), 주파수 도메인(f-domain), 그리고 통계적 특징을 모두 추출하여 총 107개의 광범위한 특징 집합을 구성하였다.
2. **인구통계학적 데이터 결합**: 신호 데이터뿐만 아니라 나이, 성별, 키, 몸무게, BMI 등의 인구통계학적 정보를 입력 변수로 포함하여 개인별 생체 차이를 반영하였다.
3. **최적의 ML 파이프라인 구축**: ReliefF 특징 선택 알고리즘과 Gaussian Process Regression(GPR) 모델을 결합하고, Bayesian Optimization을 통해 하이퍼파라미터를 최적화함으로써 수축기 혈압(SBP)과 이완기 혈압(DBP) 추정 성능을 크게 향상시켰다.

## 📎 Related Works

기존의 비침습적 혈압 추정 연구들은 주로 다음과 같은 접근 방식을 취해왔다.
- **PTT 기반 방식**: Pulse Transit Time(PTT)을 이용하여 혈압을 예측하는 방법이 제안되었으나, 잦은 보정(Calibration)이 필요하다는 한계가 있다.
- **딥러닝 기반 방식**: 최근에는 LSTM과 같은 순환 신경망(RNN)이나 Spectro-temporal Deep Neural Network 등이 사용되어 성능 향상을 꾀하였다. 특히 일부 연구에서는 원시(raw) PPG 데이터를 직접 입력으로 사용하기도 하였다.
- **특징 기반 ML 방식**: SVM이나 ANN을 이용해 시간 도메인 특징으로 혈압을 추정하는 연구들이 진행되었다.

본 연구는 기존 연구들이 주로 특정 도메인의 특징만을 사용하거나 소규모 데이터셋에 의존했다는 점에 주목한다. 특히, 시간, 주파수, 통계적 특징과 인구통계학적 데이터를 모두 통합하여 머신러닝으로 접근한 시도는 기존 연구와 차별화되는 지점이다.

## 🛠️ Methodology

### 전체 파이프라인
전체 시스템은 `신호 품질 평가 $\rightarrow$ 전처리 $\rightarrow$ 특징 추출 $\rightarrow$ 특징 선택 $\rightarrow$ ML 모델 학습 및 최적화 $\rightarrow$ 혈압 추정`의 순서로 구성된다.

### 1. 전처리 (Preprocessing)
수집된 raw PPG 신호는 다음과 같은 단계로 정제된다.
- **Normalization**: 진폭의 영향을 줄이기 위해 Z-score 정규화를 수행한다.
  $$\text{Z-score Normalized Signal} = \frac{x - \mu}{\sigma}$$
- **Filtration**: 고주파 노이즈를 제거하기 위해 차단 주파수(Cut-off frequency)가 $25\text{Hz}$인 6차 Butterworth IIR 저역 통과 필터를 사용한다.
- **Baseline Correction**: 호흡으로 인한 기저선 변동(Baseline Wandering)을 제거하기 위해 4차 다항식 피팅(Polynomial fit)을 통해 추세를 찾고 이를 원래 신호에서 뺀다.

### 2. 특징 추출 (Feature Extraction)
총 107개의 특징을 추출하며, 세부 구성은 다음과 같다.
- **Time-domain (75개)**: 수축기/이완기 피크 진폭, 노치(Notch)의 높이 및 시간, 펄스 폭, 증폭 지수(Augmentation Index) 및 PPG 1차·2차 미분 신호에서 추출한 지점들($a1, b1, a2, b2$)을 포함한다.
- **Frequency-domain (16개)**: FFT(Fast Fourier Transform)를 통해 상위 3개 피크의 진폭과 주파수, 특정 대역($0\text{--}2\text{Hz}, 2\text{--}5\text{Hz}$)의 면적 등을 추출한다.
- **Statistical (10개)**: 평균, 중앙값, 표준편차, 왜도(Skewness), 첨도(Kurtosis), 샤논 엔트로피(Shannon's Entropy) 등을 계산한다.
- **Demographic (6개)**: 키, 몸무게, 성별, 나이, BMI, 심박수를 포함한다.

### 3. 특징 선택 및 모델링
- **특징 선택**: 과적합을 방지하기 위해 Correlation-based Feature Selection(CFS), ReliefF, FSCMRMR 세 가지 알고리즘을 비교 분석하였다.
- **ML 알고리즘**: Linear Regression, Regression Trees, SVR, GPR, Ensemble Trees 등 총 19가지 모델을 10-fold 교차 검증으로 테스트하였다.
- **최적화**: 최종 선정된 GPR 모델의 하이퍼파라미터를 Bayesian Optimization을 통해 최적화하여 MSE를 최소화하였다.

### 4. 평가 지표
모델의 성능은 다음의 수식들을 통해 평가한다.
- **MAE**: $\text{MAE} = \frac{1}{n} \sum |X_p - X|$
- **MSE**: $\text{MSE} = \frac{1}{n} \sum (X_p - X)^2$
- **RMSE**: $\text{RMSE} = \sqrt{\text{MSE}}$
- **Correlation Coefficient (R)**: 예측값과 실제값 사이의 선형적 관계를 측정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Liang et al.의 공개 데이터셋을 사용하였으며, 신호 품질 지수(SQI)를 통해 필터링 된 126명의 피험자로부터 얻은 222개의 신호를 사용하였다.
- **데이터 분할**: 학습 및 검증에 85%, 최종 테스트에 15%를 할당하였다.

### 정량적 결과
최적화된 **ReliefF + GPR** 조합이 가장 우수한 성능을 보였다.
- **수축기 혈압 (SBP)**: $\text{RMSE} = 6.74$, $\text{MAE} = 3.02$, $\text{R} = 0.95$
- **이완기 혈압 (DBP)**: $\text{RMSE} = 3.59$, $\text{MAE} = 1.74$, $\text{R} = 0.96$

### 임상 표준 비교
- **AAMI 표준**: DBP의 경우 표준을 완전히 충족한다. SBP는 평균 오차는 허용 범위 내에 있으나, 표준편차(SD)가 허용 범위를 약간 초과한다.
- **BHS 표준**: SBP와 DBP 모두 Grade B 등급을 획득하여 임상적으로 유효한 수준임을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 단순히 딥러닝 모델의 복잡성을 높이는 대신, 도메인 지식에 기반한 정교한 특징 추출과 고전적인 머신러닝 모델의 최적화를 통해 매우 높은 정확도를 달성하였다. 특히 인구통계학적 특징과 PPG의 다각적 특징(t, f, statistical)을 통합한 것이 성능 향상의 핵심 요인으로 분석된다. 결과적으로 최적화된 GPR 모델은 소규모 데이터셋에서도 안정적인 예측 성능을 보였으며, 이는 실제 웨어러블 하드웨어 시스템으로 구현 가능성이 높음을 시사한다.

### 한계 및 비판적 논의
1. **데이터셋 규모**: 126명의 피험자는 일반 인구 집단을 대표하기에 다소 부족하며, 이는 모델의 일반화 성능에 영향을 줄 수 있다.
2. **SBP 표준편차**: AAMI 표준의 표준편차 요건을 완전히 충족하지 못한 점은 SBP 추정 시 개인별 변동성에 대한 모델의 대응력이 아직 완벽하지 않음을 의미한다.
3. **현실적 제약**: 실제 환경에서는 키, 몸무게와 같은 인구통계학적 데이터를 실시간으로 입력받기 어려울 수 있으므로, 이에 대한 대안이나 자동화 방안이 필요하다.

## 📌 TL;DR

본 논문은 PPG 신호의 시간, 주파수, 통계적 특징과 사용자의 인구통계학적 정보를 결합하여 비침습적으로 혈압을 추정하는 ML 프레임워크를 제안하였다. **ReliefF 특징 선택과 최적화된 Gaussian Process Regression(GPR)**을 사용하여 $\text{SBP RMSE} = 6.74$, $\text{DBP RMSE} = 3.59$라는 높은 정확도를 달성하였으며, 이는 BHS Grade B 수준의 성능이다. 이 연구는 향후 커프 없는 연속 혈압 모니터링 웨어러블 기기 개발에 중요한 기초 자료가 될 것으로 기대된다.