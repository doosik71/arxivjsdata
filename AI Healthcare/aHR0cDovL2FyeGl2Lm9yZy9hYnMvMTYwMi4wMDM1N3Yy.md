# DeepCare: A Deep Dynamic Memory Model for Predictive Medicine

Trang Pham, Truyen Tran, Dinh Phung and Svetha Venkatesh (2017)

## 🧩 Problem to Solve

본 논문은 개인 맞춤형 예측 의료(Personalized predictive medicine)를 위해 환자의 질병 및 진료 과정을 모델링하는 문제를 다룬다. 의료 데이터의 특성상 미래의 질병 상태나 진료 결과는 과거의 이력에 크게 의존하는 장기 의존성(Long-term dependencies)을 가지지만, 전자 건강 기록(Electronic Medical Records, EMR)은 다음과 같은 네 가지 주요한 도전 과제를 가지고 있다.

첫째, 의료 데이터는 매우 긴 시간 범위에 걸친 장기 의존성을 가진다. 예를 들어, 중년에 발병한 당뇨병은 평생의 위험 요소가 되며, 암은 수년 후 재발할 수 있다. 둘째, 한 번의 입원(Admission) 시 발생하는 진단 및 처치 데이터는 그 크기가 가변적인 이산 집합(Discrete set) 형태이다. 셋째, 진료 기록은 환자가 병원을 방문할 때만 기록되므로 에피소드 중심적이며, 방문 간격이 불규칙한 시간적 불규칙성(Irregular timing)을 띤다. 넷째, 질병의 진행 과정과 의료적 처치(Intervention) 사이의 복잡한 상호작용이 존재하며, 이는 미래의 위험도를 변화시킨다.

따라서 본 논문의 목표는 수동적인 피처 엔지니어링 없이 EMR을 직접 읽어 현재의 질병 상태를 추론하고 미래의 의료 결과를 예측할 수 있는 end-to-end 딥 다이내믹 메모리 네트워크인 DeepCare를 제안하는 것이다.

## ✨ Key Contributions

DeepCare의 핵심 아이디어는 Long Short-Term Memory (LSTM) 아키텍처를 의료 데이터의 특성에 맞게 확장하여, 환자의 건강 상태 궤적(Health state trajectories)을 명시적인 메모리로 관리하는 것이다. 주요 기여 사항은 다음과 같다.

1. **가변 크기 입원의 벡터 표현**: 이산적인 진단 및 처치 코드들을 연속적인 벡터 공간으로 임베딩하고, 이를 풀링(Pooling)하여 고정 크기의 벡터로 변환함으로써 가변적인 입원 데이터를 처리한다.
2. **시간적 불규칙성 모델링**: LSTM의 Forget gate에 시간 매개변수화(Time parameterization)를 도입하여, 방문 간격에 따라 메모리의 망각과 통합을 조절하는 메커니즘(Monotonic decay 및 Full time-parameterization)을 구현하였다.
3. **의료적 처치의 영향 반영**: 처치(Intervention)가 질병의 진행 방향을 바꾼다는 점에 착안하여, 현재의 처치가 Output gate에, 과거의 처치가 Forget gate에 영향을 주도록 설계하였다.
4. **다중 스케일 시간 풀링(Multiscale Temporal Pooling)**: 최근의 사건에 더 큰 가중치를 두는 Recency attention 메커니즘과 다양한 시간 범위(12개월, 24개월, 전체 이력)의 풀링을 통해 미래 예측의 정확도를 높였다.

## 📎 Related Works

기존의 의료 예측 모델들은 주로 다음과 같은 한계를 보였다. 마르코프 모델(Markov models)이나 동적 베이지안 네트워크(Dynamic Bayesian Networks)는 마르코프 가정(현재 상태가 미래를 결정한다는 가정)에 기반하므로, 의료 데이터의 핵심인 장기 의존성을 캡처하지 못하며 메모리가 없어 무관한 에피소드가 입력될 경우 이전의 주요 질병 이력을 완전히 잊어버리는 문제가 있다. 

또한, 일부 연구에서 시간적 불규칙성을 처리하기 위해 인터벌 기반 추출(Interval-based extraction) 방식을 사용하였으나, 이는 다소 거칠게 데이터를 처리하며 질병의 동역학(Dynamics)을 명시적으로 모델링하지 못한다. 딥러닝의 경우 NLP 분야에서 큰 성공을 거두었으나, 의료 분야에서는 불규칙한 시간 간격과 집합 형태의 입력 데이터 처리 문제로 인해 그 잠재력이 충분히 실현되지 않은 상태였다. DeepCare는 이러한 한계를 극복하기 위해 LSTM을 기반으로 의료 데이터 전용 수정 사항을 적용하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인
DeepCare는 크게 세 개의 계층으로 구성된다. 최하단 레이어는 수정된 LSTM으로 입원 시퀀스를 읽어 질병 상태 $\text{h}_t$를 생성하고, 중간 레이어는 이를 다중 스케일 가중 풀링(Weighted pooling)하여 집계하며, 최상단 레이어는 신경망(Neural Network)을 통해 최종 예측값 $\text{y}$를 산출한다. 전체 연산 과정은 다음과 같이 요약된다.
$$P(y|u_{1:n}) = P(\text{nnet}_y(\text{pool}\{\text{LSTM}(u_{1:n})\}))$$
여기서 $u_{1:n}$은 입원 관측치 시퀀스이며, $\text{nnet}_y$는 예측 결과에 대한 신경망 추정치이다.

### 2. 입원 데이터의 벡터 표현 (Admission Embedding)
입원 시 발생하는 진단 코드 집합 $\text{D}$와 처치 코드 집합 $\text{I}$를 각각 임베딩 행렬 $\text{A}, \text{B}$를 통해 벡터로 변환한다. 가변 크기의 집합을 고정 크기 벡터 $\text{x}_t, \text{p}_t$로 만들기 위해 세 가지 풀링 방식을 제안한다.
- **Max pooling**: 각 차원에서 가장 큰 값을 선택하여 가장 영향력이 큰 진단/처치에 집중한다.
- **Normalized sum pooling**: 진단들을 합산한 후 정규화하여 공존 질환(Comorbidity)이 많을수록 위험도가 높아지는 특성을 반영한다.
- **Mean pooling**: 평균값을 사용하여 일반적인 상태를 표현한다.

### 3. 수정된 LSTM 유닛
DeepCare는 의료적 특성을 반영하기 위해 LSTM의 게이트 구조를 다음과 같이 수정하였다.

- **Input Gate ($\text{i}_t$)**: 입원 방식(계획된 입원 vs 응급 입원) $\text{m}_t$에 따라 새로운 정보의 유입량을 조절한다.
$$\text{i}_t = \frac{1}{m_t} \sigma(\text{W}_i \text{x}_t + \text{U}_i \text{h}_{t-1} + \text{b}_i)$$
- **Output Gate ($\text{o}_t$)**: 현재의 처치 $\text{p}_t$가 현재 질병 상태의 출력을 조절하도록 설계한다.
$$\text{o}_t = \sigma(\text{W}_o \text{x}_t + \text{U}_o \text{h}_{t-1} + \text{P}_o \text{p}_t + \text{b}_o)$$
- **Forget Gate ($\text{f}_t$)**: 과거의 처치 $\text{p}_{t-1}$와 시간 간격 $\Delta t$가 메모리 망각에 영향을 준다.
$$\text{f}_t = \sigma(\text{W}_f \text{x}_t + \text{U}_f \text{h}_{t-1} + \text{P}_f \text{p}_{t-1} + \text{b}_f)$$

### 4. 시간적 불규칙성 처리
시간 간격 $\Delta t$를 처리하기 위해 두 가지 방식을 제안한다.
- **Time decay**: 시간이 흐름에 따라 자연스럽게 잊혀지는 특성을 반영하여 $\text{f}_t$에 감쇠 함수 $\text{d}(\Delta t) = [\log(e + \Delta t)]^{-1}$를 곱한다.
- **Parametric time**: 더 복잡한 역학을 모델링하기 위해 시간 차이 $\Delta t$를 벡터 $\text{q}_{\Delta t}$로 변환하여 Forget gate의 입력으로 직접 사용한다.
$$\text{f}_t = \sigma(\text{W}_f \text{x}_t + \text{U}_f \text{h}_{t-1} + \text{Q}_f \text{q}_{\Delta t} + \text{P}_f \text{p}_{t-1} + \text{b}_f)$$

### 5. 다중 스케일 풀링 및 예측
추론된 질병 상태 $\text{h}_{0:n}$을 집계할 때, 최근 사건에 더 높은 가중치를 주는 Recency attention을 적용한다. 가중치 $\text{r}_t$는 다음과 같다.
$$\text{r}_t = [m_t + \log(1 + \Delta_{t:n})]^{-1}$$
이를 통해 12개월, 24개월, 전체 이력이라는 세 가지 윈도우에 대해 각각 풀링을 수행하고, 이 벡터들을 결합(Concatenate)하여 최종 신경망에 입력한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 호주 지역 병원의 12년간 기록(2002-2013)에서 추출한 당뇨병(Diabetes) 코호트(7,191명)와 정신 건강(Mental Health) 코호트(6,109명)를 사용하였다.
- **평가 작업**: 질병 진행 예측(다음 진단 예측), 처치 추천(현재 처치 예측), 미래 위험 예측(12개월/3개월 내 응급 재입원 및 고위험군 예측)의 세 가지 작업을 수행하였다.
- **지표**: 진단/처치 예측에는 $\text{Precision@K}$를, 위험 예측에는 $\text{F-score}$를 사용하였다.

### 2. 주요 결과
- **질병 진행 및 처치 예측**: $\text{Precision@1}$ 기준, 당뇨병 데이터에서 DeepCare는 Plain RNN 대비 약 2%의 성능 향상을 보였으며, 마르코프 모델보다는 월등히 높은 성능을 기록하였다. 특히 정신 건강 데이터에서 마르코프 모델은 거의 예측에 실패(9.5%)한 반면, DeepCare는 RNN 대비 추가적인 향상을 이끌어냈다.
- **미래 위험 예측 (응급 재입원)**: 당뇨병 환자의 12개월 내 재입원 예측에서 Random Forest(71.4%)와 같은 전통적 머신러닝 모델보다 높은 79.0%의 F-score를 기록하였다. 이는 Plain RNN(75.1%)이나 단순 LSTM(75.9%)보다 높은 수치이며, 특히 시간 매개변수화(Parametric time)가 적용되었을 때 가장 좋은 성능이 나타났다.
- **고위험군 예측**: 고위험군(특정 기간 내 3회 이상 재입원) 예측에서도 DeepCare는 당뇨병 데이터에서 약 60%, 정신 건강 데이터에서 50%의 F-score를 달성하며 베이스라인들을 상회하였다.

## 🧠 Insights & Discussion

DeepCare는 인간의 메모리 체계(Semantic, Episodic, Working memory)에서 영감을 얻어 설계되었다. 진단/처치 임베딩은 의미론적 메모리를, LSTM의 메모리 셀은 에피소드 중심의 경험을, 그리고 최종 풀링 레이어는 작업 메모리와 유사한 역할을 수행한다. 특히 의료 데이터에서 '최신성(Recency)'이 미래 위험에 큰 영향을 미친다는 점을 Forget gate의 시간 감쇠와 Multiscale pooling이라는 두 가지 경로로 적절히 구현하였다.

본 모델의 강점은 수동적인 피처 엔지니어링 없이 end-to-end로 학습이 가능하다는 점과, 의료 데이터의 불규칙한 시간 간격 및 처치의 개입 효과를 아키텍처 수준에서 모델링했다는 점이다. 다만, 특정 병원의 데이터로만 학습되었으므로 다양한 의료 기관과 데이터셋에 대한 일반화 성능 검증이 필요하며, 미래에는 시퀀스-투-시퀀스(Seq2Seq) 구조를 도입하여 특정 시점의 연속적인 결과물을 예측하는 방향으로 확장할 수 있을 것으로 보인다.

## 📌 TL;DR

DeepCare는 EMR의 특성인 불규칙한 시간 간격, 가변적 입력, 의료적 처치의 영향을 반영하여 설계된 LSTM 기반의 예측 모델이다. 임베딩 풀링과 수정된 LSTM 게이트, 다중 스케일 시간 풀링을 통해 환자의 질병 궤적을 효과적으로 학습하며, 당뇨병 및 정신 건강 데이터셋에서 기존의 마르코프 모델 및 일반 RNN/LSTM보다 뛰어난 재입원 예측 및 질병 진행 예측 성능을 입증하였다. 이 연구는 정밀 의료(Predictive Medicine)를 위한 딥러닝 모델의 체계적인 설계 방향을 제시하였다.