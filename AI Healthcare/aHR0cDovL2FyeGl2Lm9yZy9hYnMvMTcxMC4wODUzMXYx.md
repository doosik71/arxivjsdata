# Benchmark of Deep Learning Models on Large Healthcare MIMIC Datasets

Sanjay Purushotham, Chuizheng Meng, Zhengping Che, Yan Liu (2017)

## 🧩 Problem to Solve

본 논문은 중환자실(ICU) 입원 환자의 상태를 정량화하고 미래의 임상 결과를 예측하는 문제에 집중한다. 구체적으로는 병원 내 사망률(Mortality), 입원 기간(Length of Stay, LOS), 그리고 질병 분류 코드인 ICD-9 코드 그룹을 예측하는 것을 목표로 한다.

이러한 예측은 환자의 질병 중증도를 평가하고, 새로운 치료법이나 의료 정책의 가치를 결정하는 데 매우 중요하다. 기존에는 SAPS-II, SOFA와 같은 임상 점수 체계(Scoring systems)나 일부 머신러닝 모델이 사용되었으나, 대규모 공개 헬스케어 데이터셋을 활용하여 최신 딥러닝 모델의 성능을 기존의 SOTA 머신러닝 모델 및 점수 체계와 일관되고 포괄적으로 비교 분석한 벤치마크 연구가 부족하다는 점이 본 연구의 핵심 문제이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 MIMIC-III라는 대규모 데이터셋을 사용하여 다양한 임상 예측 작업에 대해 딥러닝 모델의 성능을 정밀하게 벤치마킹한 것이다. 

가장 중심적인 설계 아이디어는 **'가공된(Processed)' 특징**과 **'원시(Raw)' 시계열 특징**의 성능 차이를 분석하는 것이다. 연구진은 딥러닝 모델이 사람이 직접 설계한 규칙 기반의 전처리가 수행된 특징보다, 원시 시계열 데이터를 직접 입력받았을 때 더 뛰어난 성능을 보인다는 점을 입증하였다. 이는 딥러닝 모델이 복잡한 임상 데이터로부터 유용한 특징 표현(Feature representation)을 자동으로 학습할 수 있음을 시사한다.

## 📎 Related Works

기존의 연구들은 주로 다음과 같은 접근 방식을 취하였다.
- **임상 점수 체계:** SAPS-II, SOFA, APACHE 등은 소수의 선택된 변수와 로지스틱 회귀와 같은 단순한 모델을 사용하여 사망률을 예측한다. 그러나 이러한 방식은 변수 간의 선형성 및 가산성 가정을 전제로 하므로 실제 임상 환경의 복잡성을 반영하기 어렵다는 한계가 있다.
- **머신러닝 및 딥러닝 접근법:** Super Learner와 같은 앙상블 모델이나 RNN 기반의 모델들이 제안되었으나, 기존 벤치마크 연구들은 비교 대상 모델이 제한적이거나(예: 로지스틱 회귀와만 비교), 다양한 예측 작업에 대해 일관된 기준을 제시하지 못했다.

본 논문은 이러한 한계를 극복하기 위해 다수의 예측 작업(사망률, LOS, ICD-9 코드)과 다양한 특징 세트(Processed vs Raw), 그리고 여러 최신 모델을 동시에 비교하는 포괄적인 실험 설계를 도입하였다.

## 🛠️ Methodology

### 1. 데이터셋 및 전처리
본 연구는 2001년부터 2012년까지 Beth Israel Deaconess Medical Center ICU에 입원한 성인 환자의 데이터를 포함하는 **MIMIC-III (v1.4)** 데이터셋을 사용한다. 정보 누출을 방지하기 위해 환자당 첫 번째 ICU 입원 기록만을 사용하였다.

### 2. 특징 세트 (Feature Sets)
비교 분석을 위해 세 가지 서로 다른 특징 세트를 구성하였다.
- **Feature Set A (Processed):** SAPS-II 점수 계산에 사용되는 17개 특징을 포함하며, 의료 지식에 기반하여 이상치를 제거하고 단위를 통합하는 등 전처리를 수행한 세트이다.
- **Feature Set B (Raw - Small):** Set A와 동일한 변수를 사용하지만, 전처리를 최소화하고 원시 값을 그대로 유지한 20개 특징 세트이다.
- **Feature Set C (Raw - Large):** 결측률이 낮은 135개의 원시 특징을 포함하며, 모델이 대량의 원시 데이터로부터 표현을 스스로 학습할 수 있는지 확인하기 위해 구축되었다.

### 3. 예측 모델
- **Scoring Methods:** SAPS-II, SOFA 및 수정된 New SAPS-II를 사용한다.
- **Super Learner:** 여러 머신러닝 알고리즘(Random Forest, GBM, Neural Network 등)의 최적 가중치 조합을 찾는 앙상블 모델이다.
- **Deep Learning Models:** 
    - **Feedforward Neural Networks (FFN):** 비시계열 데이터를 처리하는 표준 신경망이다.
    - **Recurrent Neural Networks (RNN):** 시계열 데이터를 위해 GRU(Gated Recurrent Unit)를 사용한다.
    - **Multimodal Deep Learning Model (MMDL):** 본 논문에서 제안한 구조로, 비시계열 데이터는 FFN으로, 시계열 데이터는 GRU로 처리한 후, 이들을 **공통 표현 층(Shared Representation Layer)**에서 결합하여 최종 예측을 수행하는 하이브리드 구조이다.

### 4. 학습 및 추론 절차
- **손실 함수 및 최적화:** 분류 작업에는 Cross-entropy loss를, 회귀 작업(LOS 예측)에는 Mean Squared Error (MSE)를 사용한다. 최적화 알고리즘으로는 RMSProp을 사용하였다.
- **평가 지표:** 분류 성능은 AUROC(Area under the ROC curve)와 AUPRC(Area under Precision-Recall Curve)로 측정하며, LOS 예측은 MSE로 측정한다.
- **검증 방법:** 5-fold stratified cross-validation을 통해 결과의 신뢰성을 확보하였다.

## 📊 Results

### 1. 사망률 예측 (Mortality Prediction)
- **결과:** 원시 특징을 사용한 MMDL 모델이 모든 모델 중 가장 높은 성능을 보였다. 특히 Feature Set C(135개 원시 특징)를 사용했을 때, Super Learner 대비 AUROC는 약 7-8%, AUPRC는 최대 50%까지 향상되었다.
- **특이사항:** 입원 후 24시간 데이터만으로도 48시간 데이터와 유사한 성능이 나타나, 사망률 예측에 있어 데이터의 길이가 늘어나는 것이 큰 이점이 되지 않음을 확인하였다.

### 2. ICD-9 코드 그룹 예측
- **결과:** Feature Set C를 기반으로 학습된 MMDL 모델이 Super Learner 모델들보다 평균적으로 4-5% 더 높은 AUPRC 성능을 기록하며 우위를 점하였다.

### 3. 입원 기간 예측 (Length of Stay)
- **결과:** 회귀 문제인 LOS 예측에서도 MMDL 모델이 가장 낮은 MSE를 기록하였다. 특히 Super Learner II와 비교했을 때 오차가 거의 50% 가까이 감소하는 괄목할 만한 성능 향상을 보였다.

### 4. 계산 효율성
- **결과:** Python 기반의 Super Learner 구현체가 R 버전보다 훨씬 빨랐으며, MMDL 모델은 Super Learner-Python보다도 평가 시간이 짧아 효율적인 학습 및 추론이 가능함을 보였다.

## 🧠 Insights & Discussion

본 연구의 결과는 헬스케어 데이터 분석에서 **특징 공학(Feature Engineering)의 필요성에 대한 새로운 관점**을 제시한다. 기존의 전통적인 머신러닝이나 점수 체계에서는 전문가가 직접 변수를 선택하고 가공하는 과정이 필수적이었으나, 딥러닝 모델(특히 MMDL)은 대량의 원시 데이터에서 직접 유의미한 패턴을 추출하는 능력이 뛰어남을 입증하였다.

특히, 가공된 특징(Set A)보다 원시 특징(Set B, C)에서 성능이 더 좋게 나타난 점은, 인간이 정의한 전처리 규칙이 오히려 데이터에 내재된 중요한 정보를 손실시킬 수 있음을 시사한다. 

다만, 본 논문은 모델의 성능 향상에 집중하고 있으며, 딥러닝 모델의 고질적인 문제인 '블랙박스' 특성, 즉 **예측 결과에 대한 임상적 해석 가능성(Interpretability)**에 대해서는 구체적으로 다루지 않았다. 실제 의료 현장에 적용하기 위해서는 왜 그러한 예측이 나왔는지에 대한 설명력이 추가로 요구될 것이다.

## 📌 TL;DR

본 논문은 MIMIC-III 데이터셋을 통해 사망률, 입원 기간, 질병 코드 예측 작업에서 딥러닝 모델의 성능을 벤치마킹하였다. 실험 결과, 제안된 **MMDL(Multimodal Deep Learning Model)**이 기존의 임상 점수 체계 및 앙상블 머신러닝 모델(Super Learner)을 압도하는 성능을 보였으며, 특히 사람이 가공한 특징보다 **대량의 원시 시계열 데이터를 직접 사용했을 때 성능이 극대화**됨을 확인하였다. 이 연구는 향후 의료 데이터 분석에서 복잡한 전처리 과정 없이 딥러닝을 통해 직접 특징을 학습시키는 방향성의 중요성을 제시한다.