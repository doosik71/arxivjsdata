# Systolic blood pressure estimation using ECG and PPG in patients undergoing surgery

Shaoxiong Sun et al. (2023)

## 🧩 Problem to Solve

수술 중 환자의 혈압(Blood Pressure, BP) 모니터링은 매우 중요하며, 일반적으로는 비침습적인 간헐적 커프 측정 방식이나 침습적인 연속적 카테터 측정 방식이 사용된다. 하지만 간헐적 측정 방식은 측정 주기 사이의 급격한 혈압 변화를 감지하지 못할 위험이 있으며, 침습적 방식은 원위부 허혈, 출혈, 혈전 및 감염과 같은 부작용과 높은 비용을 초래할 수 있다.

따라서 본 연구의 목표는 수술 중인 환자를 대상으로 심전도(Electrocardiography, ECG)와 광전용적맥파(Photoplethysmography, PPG) 신호를 이용하여 비침습적으로 수축기 혈압(Systolic Blood Pressure, SBP)을 높은 정확도로 연속 추정하는 방법을 제안하는 것이다. 특히, 임상 현장에서 제공되는 간헐적 혈압 측정값을 활용하여 추정 모델을 동적으로 보정함으로써 모니터링의 공백을 메우고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **동적 특성 선택(Dynamic Feature Selection, DFS)**과 **온라인 모델 업데이트**를 결합하여, 환자의 생체 상태 변화를 실시간으로 반영하는 회귀 모델을 구축하는 것이다. 

단순히 고정된 특징량을 사용하는 것이 아니라, 특성별 강건성(Robustness)과 상관관계 기반 특성 선택(Correlation-based Feature Selection, CFS) 원리를 적용하여 매 간헐적 측정 시점마다 최적의 특징 조합을 선택한다. 이를 통해 개별 환자의 혈관 특성 변화와 심혈관 상태를 반영한 맞춤형 SBP 추정 모델을 구현하였다.

## 📎 Related Works

기존의 비침습적 혈압 추정 연구는 주로 PPG 유도 특성을 이용한 회귀 모델에 집중해 왔으며, 특히 ECG의 R-peak와 PPG의 특정 지점 사이의 시간 지연인 맥파 도달 시간(Pulse Arrival Time, PAT)이 가장 널리 사용되었다. PAT는 혈압이 상승하면 혈관의 탄성도가 감소하여 맥파 전달 속도가 빨라지고, 결과적으로 PAT가 감소한다는 생리학적 근거를 가진다.

그러나 기존 방식들은 다음과 같은 한계가 있다. 첫째, PAT는 혈압과 직접 연관된 맥파 전달 시간(Pulse Transit Time, PTT) 외에 심장 박출 전 시간(Pre-ejection period)을 포함하고 있어 신뢰도가 떨어진다. 둘째, 혈관 평활근 긴장도(Vascular smooth muscle tone)의 변화가 혈압-PTT 관계에 영향을 미친다. 셋째, 고정된 보정 모델은 시간이 지남에 따라 심혈관 특성이 변함에 따라 성능이 저하된다. 이러한 이유로 기존 연구들은 AAMI(Association for the Advancement of Medical Instrumentation) 표준(평균 오차 $\le 5\text{ mmHg}$, 표준편차 $\le 8\text{ mmHg}$)을 충족하는 데 어려움을 겪었다.

## 🛠️ Methodology

### 1. 전체 파이프라인
전체 시스템은 신호 획득 $\rightarrow$ 특징 추출 $\rightarrow$ 동적 특성 선택 $\rightarrow$ 다중 선형 회귀 모델 구축 $\rightarrow$ SBP 추정의 순서로 진행된다. 30초 단위로 SBP를 추정하며, 10분마다 제공되는 간헐적 SBP 측정값을 사용하여 모델의 특징량과 계수를 업데이트한다.

### 2. 특징 추출 (Feature Extraction)
SBP와 유의미한 관계가 있다고 알려진 9가지 특징을 4개의 그룹으로 분류하여 추출한다.
- **Group 1:** $\text{PAT}$ (ECG R-peak부터 PPG foot까지의 시간 지연)
- **Group 2:** $\text{PPG amplitude}$, $\text{sp}_{\text{mean}}$ (수축기 단계 1차 미분 평균), $\text{sp}_{\text{var}}$ (수축기 단계 1차 미분 분산)
- **Group 3:** $\text{Pulse Delay (PD)}$ (PPG의 첫 번째 피크와 두 번째 피크 사이의 시간 지연)
- **Group 4:** 2차 미분 기반 특성 ($\text{c/a}$, $\text{e/a}$, $\text{norm a}$, $\text{norm b}$)

### 3. 동적 특성 선택 (Dynamic Feature Selection, DFS)
특징 선택은 다음의 단계로 이루어진다.
1. **첫 번째 특징 선택:** 최근 9개의 간헐적 측정값과 PAT 사이의 절대 상관계수($\text{ACC}$)를 계산한다. $\text{ACC} > 0.7$이면 PAT를 선택하고, 그렇지 않으면 Group 2 $\rightarrow$ Group 3 $\rightarrow$ Group 4 순으로 탐색하여 가장 높은 $\text{ACC}$를 가진 특성을 선택한다. (부트스트래핑을 통해 이상치 영향을 최소화한 정밀 선택 수행)
2. **추가 특징 선택 (CFS):** 첫 번째 선택된 특성과 상호 보완적인 정보를 제공하는 특성을 추가한다. 이때 다음과 같은 $\text{merit}$ 함수를 사용하여 중복성을 최소화하고 타겟 변수와의 상관관계를 최대화하는 부분 집합을 찾는다.
$$\text{merit} = \frac{\sqrt{\sum_{i=1}^{k} r_{sf_i}}}{\sqrt{k + 2 \sum_{i=1}^{k-1} \sum_{j=i+1}^{k} r_{f_i f_j}}}$$
여기서 $r_{sf_i}$는 $i$번째 특성과 $\text{SBP}$ 사이의 상관계수이고, $r_{f_i f_j}$는 특성 간의 상관계수이며, $k$는 선택된 특성의 수이다. 과적합 방지를 위해 최대 특성 수는 3개로 제한한다.

### 4. 회귀 모델 및 추론
선택된 특성들을 바탕으로 최근 9개의 간헐적 측정 데이터를 이용하여 다중 선형 회귀(Multiple Linear Regression) 모델을 구축한다. 모델의 최종 출력값에는 가장 최근의 간헐적 측정값과 일치하도록 오프셋(Offset)을 추가하여 정밀도를 높인다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** 수술을 받는 29명의 환자 중 신호 품질이 양호한 23명의 데이터(총 91.2시간)를 사용하였다.
- **비교 대상:** Zero-order hold(최근 측정값 유지), PAT-only 모델, PPG-only 모델, PCA 기반 모델 등과 비교하였다.
- **평가 지표:** $\text{RMSE}$, $\text{Bland-Altman analysis (Bias } \pm \text{ SD)}$, $\text{Pearson correlation coefficient}$를 사용하였다.

### 2. 정량적 결과
제안 방법은 다음과 같은 성능을 기록하며 AAMI 표준을 충족하였다.
- **Mean Difference (Bias):** $0.08\text{ mmHg}$
- **Standard Deviation of Difference (SDD):** $7.97\text{ mmHg}$
- **Correlation Coefficient:** $0.89$ ($p < 0.001$)
- **Pooled RMSE:** $7.97\text{ mmHg}$

### 3. 주요 분석 결과
- **모델 비교:** 제안 방법은 Zero-order hold, PAT-only, PPG-only 및 PCA 기반 방법보다 통계적으로 유의미하게 낮은 $\text{RMSE}$와 높은 상관계수를 보였다.
- **특성 수의 영향:** 최대 특성 수를 3개로 설정했을 때, 1개 또는 2개로 설정했을 때보다 성능이 우수하였다.
- **업데이트의 필요성:** 모델 업데이트 직후의 $\text{RMSE}$($5.42\text{ mmHg}$)가 다음 업데이트 직전의 $\text{RMSE}$($9.38\text{ mmHg}$)보다 훨씬 낮게 나타나, 실시간 모델 업데이트의 필요성이 입증되었다.

## 🧠 Insights & Discussion

본 연구는 ECG와 PPG 신호를 결합하고, 간헐적 측정값을 통한 동적 보정 메커니즘을 도입함으로써 수술 중 SBP 추정의 정확도를 AAMI 표준 수준까지 끌어올렸다. 특히 특징량의 생리학적 근거와 추출 강건성을 고려한 계층적 선택 방식과 CFS 기반의 중복 제거 전략이 주효했다. 또한, 단순 선형 회귀를 사용함으로써 복잡한 딥러닝 모델($O(N)$ 수준의 낮은 복잡도)에 비해 연산 효율성이 매우 뛰어나 실시간 임상 적용 가능성이 높다.

다만, 본 연구에는 몇 가지 한계가 존재한다. 첫째, 샘플 크기가 작아 LASSO와 같은 고도화된 정규화 회귀 모델을 적용하지 못했다. 둘째, 간헐적 측정값을 실제 커프 측정값이 아닌 연속 혈압 신호의 평균값으로 시뮬레이션하여 사용하였다. 실제 커프 측정 시 발생하는 노이즈나 오차가 포함될 경우 성능이 저하될 가능성이 있다. 셋째, 심각한 부정맥이 있는 환자의 데이터는 제외되었으므로, 향후 비정상 심장 활동 상황에서의 특성 추출 강건성을 확보하는 연구가 필요하다.

## 📌 TL;DR

본 논문은 수술 중 환자의 수축기 혈압(SBP)을 30초 간격으로 정밀하게 추정하기 위해, ECG/PPG 기반의 **동적 특성 선택(DFS)**과 **다중 선형 회귀**를 결합한 모델을 제안한다. 10분마다 제공되는 간헐적 혈압 측정값으로 모델을 실시간 업데이트함으로써 AAMI 표준을 충족하는 높은 정확도($\text{SDD} = 7.97\text{ mmHg}$)를 달성하였다. 이 연구는 기존의 침습적 모니터링의 위험성과 간헐적 모니터링의 정보 손실 문제를 동시에 해결하여 수술실 내 환자 안전을 향상시킬 수 있는 잠재력을 가진다.