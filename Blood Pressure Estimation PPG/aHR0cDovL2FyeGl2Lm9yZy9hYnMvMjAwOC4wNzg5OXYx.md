# Accelerometric Method for Cuffless Continuous Blood Pressure Measurement

Mousumi Das, Tilendra Choudhary, L.N. Sharma, and M.K. Bhuyan (2020)

## 🧩 Problem to Solve

고혈압은 뇌졸중, 관상동맥 질환과 같은 심각한 심혈관 질환(CVD)의 주요 위험 요인이므로, 이를 조기에 발견하기 위한 지속적이고 효과적인 혈압(BP) 모니터링이 필수적이다. 현재 중환자실(ICU)에서 사용하는 침습적 방식(카뉴라 바늘 삽입)은 정확하지만 출혈이나 감염의 위험이 있으며, 비침습적인 커프(Cuff) 기반 방식은 사용자에게 통증과 불편함을 주고 간헐적인 측정만 가능하다는 한계가 있다.

최근에는 커프가 필요 없는(Cuffless) 혈압 측정 연구가 활발하며, 주로 맥파 전달 시간(Pulse Transit Time, PTT)을 이용한다. 하지만 기존의 PTT 방식은 심전도(ECG)와 광전용적맥파(PPG), 또는 심음도(SCG)와 PPG처럼 두 개 이상의 서로 다른 생체 신호를 동시에 측정해야 하므로, 다수의 센서가 필요하고 시스템 구성이 복잡하며 비용이 증가하는 문제가 있다. 따라서 본 논문은 단일 센서만을 이용하여 연속적으로 혈압을 측정할 수 있는 저비용, 경량화된 시스템을 개발하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 다중 모달리티 시스템 대신 단일 MEMS 가속도계 센서를 통해 획득한 **심음도(Seismocardiogram, SCG) 신호의 단일 성분(z-축)만을 활용하여 혈압을 추정**하는 것이다. 구체적으로는 SCG 신호에서 **좌심실 박출 시간(Left Ventricular Ejection Time, LVET)**을 추출하고, 이를 심박수(Heart Rate, HR)와 결합하여 개인별 맞춤형 보정(Calibration)을 통해 수축기 혈압(SBP)과 이완기 혈압(DBP)을 계산하는 방법론을 제안한다.

## 📎 Related Works

기존의 비침습적 혈압 측정 방식은 주로 PTT를 활용하였다. PTT는 심장의 근위 지점에서 말단 지점까지 맥파가 전달되는 시간으로 정의되며, 일반적으로 ECG의 R-peak와 PPG의 systolic peak 또는 onset 사이의 간격으로 측정한다. 또한 일부 연구에서는 SCG의 대동맥 판막 개방(Aortic Valve Opening, AO) 지점과 PPG의 지점을 사용하여 PTT를 측정하기도 하였다.

이러한 기존 방식들은 공통적으로 두 개의 상관관계가 있는 심장 신호가 필요하며, 특징점(Feature point)을 찾는 계산 과정이 복잡하고 센서 배치의 유연성이 떨어진다는 한계가 있다. 단일 SCG 센서의 서로 다른 직교 축(x, z축)을 이용한 연구도 있었으나, 온셋(Onset) 지점을 정확히 결정하는 것이 매우 어렵고 정확도가 낮아 실제 소비자용 응용 프로그램으로 배포하기에는 성숙도가 부족한 상태이다.

## 🛠️ Methodology

### 전체 시스템 구조 및 원리

본 제안 방법은 가속도계 센서로 SCG 신호를 획득하고, 여기서 $\text{LVET}$와 $\text{HR}$을 추출한 뒤, 선형 회귀 모델을 통해 혈압을 추정하는 파이프라인을 가진다. 혈압은 기본적으로 심박출량($Q$)과 말초 저항($\Omega$)의 곱으로 표현되며, 심박출량은 심박수($\text{HR}$)와 1회 박출량($\text{SV}$)에 의존한다. 여기서 $\text{LVET}$는 $\text{SV}$와 직접적인 연관이 있으므로, $\text{LVET}$를 통해 혈압을 추정할 수 있다는 생리학적 근거를 바탕으로 한다.

### 주요 방정식 및 모델링

실제 $\text{LVET}$를 측정하기 위한 $\text{AC}$ 지점(대동맥 판막 폐쇄)의 식별이 어렵기 때문에, 본 논문에서는 $\text{AC}$ 직후에 나타나는 $\text{pAC}$ 피크를 사용하여 $\text{LVET}$를 근사한 $\text{LVET}'$를 정의한다.

$$ \text{LVET}' = \text{pAC} - \text{AO} $$

추정된 $\text{LVET}'$와 심박수($\text{HR}$)를 이용하여 혈압($\text{BP}$)을 모델링하는 식은 다음과 같다.

$$ \text{BP} = a \cdot \ln(\text{LVET}') + b \cdot \text{HR} + c $$

여기서 $a, b, c$는 사용자마다 다른 개인별 파라미터이며, 최소제곱법(Least Square)을 통해 추정한다. 이를 행렬 형태로 나타내면 다음과 같다.

$$ \begin{bmatrix} \text{BP}_1 \\ \text{BP}_2 \\ \vdots \\ \text{BP}_n \end{bmatrix} = \begin{bmatrix} \ln(\text{LVET}'_1) & \text{HR}_1 & 1 \\ \ln(\text{LVET}'_2) & \text{HR}_2 & 1 \\ \vdots & \vdots & \vdots \\ \ln(\text{LVET}'_n) & \text{HR}_n & 1 \end{bmatrix} \begin{bmatrix} a \\ b \\ c \end{bmatrix} $$

알 수 없는 계수 벡터 $X = [a, b, c]^T$는 의사 역행렬(Pseudo-inverse)을 이용한 선형 회귀로 계산한다.

$$ X = (A^T A)^{-1} A^T \text{BP} $$

### 신호 처리 및 피크 검출 절차

1. **AO Peak 검출**: MVMD(Multivariate Variational Mode Decomposition)를 사용하여 저주파 잡음을 제거하고, 가우시안 미분 필터(Gaussian derivative filter)를 적용하여 수축기 프로필을 강화한다. 이후 힐베르트 변환(Hilbert transform)과 심장 주기 엔벨로프(CCE)를 사용하여 $\text{AO}$ 지점을 정밀하게 찾는다.
2. **후처리(Post-processing)**: $\text{AO}$ 피크 검출 시 발생하는 오검출(Missing peak, False positive)을 줄이기 위해, 인접한 피크 간의 거리와 중앙값(Median) 차이를 비교하여 잘못된 피크를 제거하거나 누락된 피크를 보간한다.
3. **pAC Peak 검출**: $\text{AO}$-$\text{AO}$ 간격을 4등분 하여 특정 구간($M_2$에서 $M_1$ 사이)을 설정하고, 버터워스 고역 통과 필터(Butterworth high pass filter, 10Hz)를 적용하여 해당 구간의 최댓값을 $\text{pAC}$ 피크로 결정한다.

## 📊 Results

### 실험 설정

- **대상**: 건강한 성인 10명 (앙와위, Supine position)
- **측정 데이터**: 6분간 SCG, ECG, PPG 및 기준 혈압(ABP)을 동시 측정
- **기준 장비**: CNAP Monitor 500 (Finger-cuff 방식)
- **평가 지표**: 평균 오차($\text{ME}$), 평균 절대 오차($\text{MAE}$), 표준편차($\text{STD}$)
- **데이터 분할**: 각 대상자 데이터의 70%를 보정(Calibration)에 사용, 30%를 테스트에 사용

### 정량적 결과

제안된 $\text{LVET}'$ 기반 방법의 결과는 다음과 같다.

- **수축기 혈압(SBP)**: $\text{ME} = -0.19 \pm 3.3\text{ mmHg}$, $\text{MAE} = 3.2\text{ mmHg}$
- **이완기 혈압(DBP)**: $\text{ME} = -1.29 \pm 2.6\text{ mmHg}$, $\text{MAE} = 2.6\text{ mmHg}$
- 위 결과는 IEEE 표준 요구 사항인 $5 \pm 8\text{ mmHg}$ 오차 범위를 만족한다.

### 비교 분석

기존의 PTT 방식(ECG-PPG 조합) 및 $\text{PTT}_1$ 방식(SCG-PPG 조합)과 비교했을 때, 제안 방법은 유사한 수준의 정확도를 보였다. Bland-Altman plot과 회귀 분석(Regression plot) 결과, 제안 방법과 기존 방법 간의 상관관계가 매우 높게 나타났으며, 이는 단일 센서만으로도 다중 센서 기반의 기존 시스템을 대체할 수 있는 가능성을 시사한다.

## 🧠 Insights & Discussion

본 논문은 혈압 측정에 필수적이라고 여겨졌던 PPG 신호를 제거하고, 단일 SCG z-축 신호만으로 혈압을 추정할 수 있음을 입증하였다. 이는 하드웨어 구성을 단순화하여 사용자의 활동성을 높이고, 비용을 절감하며, 웨어러블 기기로의 적용 가능성을 크게 높인 성과이다. 특히 $\text{LVET}$의 근사치인 $\text{LVET}'$를 도입하여 실제 구현 시 발생할 수 있는 특징점 검출의 어려움을 효과적으로 해결하였다.

다만, 본 연구에는 몇 가지 한계점이 존재한다. 첫째, 모든 실험이 건강한 피험자를 대상으로 앙와위(Supine position)에서 진행되었으므로, 다양한 자세나 고혈압 환자 등 병리적 상태에서의 강건성은 검증되지 않았다. 둘째, 개인별 보정(Calibration) 과정이 필수적이므로, 초기 측정 시 기준 혈압 장비가 필요하다는 점은 완전한 자립형 시스템으로 가기 위해 해결해야 할 과제이다.

## 📌 TL;DR

본 연구는 MEMS 가속도계를 이용해 획득한 **단일 SCG z-축 신호에서 $\text{LVET}'$와 $\text{HR}$을 추출하여 혈압을 연속적으로 측정하는 방법**을 제안하였다. 실험 결과, IEEE 표준 오차 범위를 만족하며 기존의 다중 센서 기반 PTT 방식과 대등한 성능을 보였다. 이 연구는 향후 저비용, 고효율의 상시 혈압 모니터링 웨어러블 기기 개발에 중요한 기초가 될 것으로 기대된다.
