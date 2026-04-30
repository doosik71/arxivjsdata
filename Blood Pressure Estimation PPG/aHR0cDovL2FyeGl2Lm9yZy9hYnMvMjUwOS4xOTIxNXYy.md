# PPG-DISTILL: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation

Juntong Ni, Saurabh Kataria, Shengpu Tang, Carl Yang, Xiao Hu, Wei Jin (2025)

## 🧩 Problem to Solve

본 논문은 웨어러블 기기에서의 효율적인 광전용적맥파(Photoplethysmography, PPG) 신호 분석을 위해, 거대 PPG 파운데이션 모델(Foundation Model)의 지식을 경량 모델로 전이하는 문제를 다룬다. 

PPG 신호는 혈류량의 변화를 측정하는 시계열 데이터로, 심혈관 이벤트와 관련된 국소적 파형 형태(Local waveform morphology)와 자율신경 조절을 반영하는 장기적 구조적 리듬(Long-range structural rhythm)이라는 두 가지 핵심 특성을 가진다. 최근 이러한 특성을 학습한 대규모 파운데이션 모델들이 등장하여 높은 성능을 보이고 있으나, 이들은 연산량과 메모리 사용량이 매우 커서 자원이 제한된 웨어러블 디바이스에 직접 배포하기 어렵다는 한계가 있다. 

기존의 지식 증류(Knowledge Distillation, KD) 기법들은 주로 모델의 최종 출력값(Prediction)이나 중간 특징값(Feature)을 일치시키는 Global KD 방식에 의존한다. 그러나 이러한 방식은 PPG 신호의 핵심인 국소적 형태와 패치 간의 리듬 정보를 충분히 보존하지 못해, 경량 모델의 성능 저하를 초래한다. 따라서 본 연구의 목표는 PPG 신호의 특수성을 반영하여 전역적 지식과 국소적 지식을 모두 효율적으로 전이할 수 있는 distillation 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 PPG 신호의 특성인 '형태(Morphology)'와 '리듬(Rhythm)'을 보존하기 위해, 기존의 Global KD에 더해 패치 수준(Patch-level)의 증류 전략을 도입한 **PPG-DISTILL** 프레임워크를 제안하는 것이다.

구체적으로, 국소적 세그먼트 간의 변별력을 강화하는 **Morphology Distillation**과 패치 간의 시간적 구조 및 의존성을 캡처하는 **Rhythm Distillation**을 통해, 학생 모델(Student model)이 교사 모델(Teacher model)의 정교한 PPG 표현 능력을 학습하도록 설계하였다. 이를 통해 모델 크기를 획기적으로 줄이면서도 파운데이션 모델 수준의 성능을 유지하거나, 특정 작업에서는 오히려 능가하는 효율적인 경량 모델을 생성할 수 있다.

## 📎 Related Works

논문에서는 PPG 신호 분석과 파운데이션 모델, 그리고 지식 증류에 관한 연구를 소개한다.

1.  **PPG 신호 분석**: PPG는 심박수, 심박 변이도, 혈압 추정 및 심방세동(Atrial Fibrillation) 탐지와 같은 진단 응용분야뿐만 아니라 스트레스 및 인지 상태 측정 등 정신 건강 모니터링에도 널리 사용되고 있다.
2.  **PPG 파운데이션 모델**: 최근 임상 데이터 기반의 PaPaGei, GPT-PPG 및 웨어러블 필드 데이터 기반의 Pulse-PPG와 같은 모델들이 제안되었다. 이들은 대규모 데이터를 통해 일반화된 표현을 학습하지만, 모델 파라미터 수가 많아 엣지 디바이스 배포에 제약이 있다.
3.  **지식 증류(Knowledge Distillation)**: 교사 모델의 출력을 학생 모델이 모방하게 하여 모델을 압축하는 기법이다. 시계열 데이터에 특화된 TimeDistill 등이 존재하지만, PPG 신호의 형태적, 리듬적 특성을 직접적으로 다룬 연구는 본 논문이 처음이다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
PPG-DISTILL은 입력 PPG 신호를 패치 단위로 분할한 후, 전역적(Global) 수준과 국소적(Patch-level) 수준에서 동시에 지식을 전이한다. 전체 학습 목표 함수는 다음과 같이 정의된다.

$$\mathcal{L} = \mathcal{L}_{sup} + \alpha\mathcal{L}_{Y}^{KD} + \beta\mathcal{L}_{H}^{KD} + \gamma(\mathcal{L}_{mor} + \mathcal{L}_{rhy})$$

여기서 $\mathcal{L}_{sup}$은 정답 라벨을 이용한 지도 학습 손실이며, $\alpha, \beta, \gamma$는 각 손실 항의 가중치를 조절하는 하이퍼파라미터이다.

### 주요 구성 요소 및 상세 설명

#### 1. Global KD (전역적 증류)
- **Prediction-matching ($\mathcal{L}_{Y}^{KD}$)**: 교사와 학생 모델의 최종 예측값 $\hat{Y}_t, \hat{Y}_s$를 일치시킨다.
- **Feature-matching ($\mathcal{L}_{H}^{KD}$)**: 신호 수준의 내부 특징값 $H_t, H_s$를 정렬하여 전반적인 신호 표현을 학습한다.

#### 2. Patch-level Distillation (국소적 증류)
PPG 신호 $X \in \mathbb{R}^L$를 패치 크기 $P$로 나누어 $N = L/P$개의 패치 시퀀스 $X^p \in \mathbb{R}^{P \times N}$를 생성한다.

**A. PPG Morphology Distillation ($\mathcal{L}_{mor}$)**
국소적 파형의 형태적 특징을 보존하기 위해 InfoNCE 스타일의 대조 학습(Contrastive Learning)을 사용한다.
- 학생 모델의 특징 $H_s^p \in \mathbb{R}^{N \times d_s}$와 교사 모델의 특징 $H_t^p \in \mathbb{R}^{N \times d_t}$의 차원을 맞추기 위해 학습 가능한 선형 어댑터 $A \in \mathbb{R}^{d_t \times d_s}$를 도입하여 $\tilde{H}_t^p = H_t^p A$를 계산한다.
- $\ell_2$-정규화를 거친 후, 유사도 행렬 $Z$를 생성하며, 동일한 위치의 패치는 긍정(Positive), 다른 위치의 패치는 부정(Negative)으로 간주하여 다음 손실 함수를 최소화한다.
$$\mathcal{L}_{mor} = \frac{1}{N} \sum_{i=1}^{N} \left( -\log \frac{\exp(Z_{ii})}{\sum_{j=1}^{N} \exp(Z_{ij})} \right)$$

**B. PPG Rhythm Distillation ($\mathcal{L}_{rhy}$)**
비트 간 주기성과 타이밍 규칙성(Rhythm)을 보존하기 위해 패치 간의 상대적 거리 구조를 전이한다.
- 교사와 학생 모델 각각에 대해 패치 간 유클리드 거리 행렬 $[D_t]_{ij}$와 $[D_s]_{ij}$를 계산한다.
- 두 거리 행렬의 구조적 차이를 smooth L1 penalty를 통해 최소화한다.
$$\mathcal{L}_{rhy} = \frac{1}{N(N-1)} \sum_{i \neq j} \text{smoothL1}([ \tilde{D}_s ]_{ij}, [ \tilde{D}_t ]_{ij})$$

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**: 
    - 회귀(Regression): DaLiA 데이터셋을 이용한 심박수(Heart Rate) 추정.
    - 분류(Classification): StanfordAF 데이터셋을 이용한 심방세동(Atrial Fibrillation) 탐지.
- **모델 구성**:
    - 교사 모델: GPT-PPG-19m, PaPaGei.
    - 학생 모델: MLP, GPT-PPG-1m.
- **측정 지표**: MSE, MAE (회귀), Accuracy, F1-score (분류).

### 정량적 결과
- **성능 향상**: PPG-DISTILL을 적용한 GPT-PPG-1m은 Global KD 대비 뚜렷한 성능 향상을 보였다. 특히 StanfordAF 데이터셋에서 F1-score가 최대 21.8% 상대적으로 향상되었으며, DaLiA 데이터셋의 MSE는 최대 20.68% 개선되었다.
- **교사 모델 능가**: 놀랍게도 DaLiA 데이터셋에서 GPT-PPG-19m을 교사로 사용했을 때, PPG-DISTILL로 학습된 학생 모델(GPT-PPG-1m)이 교사 모델보다 더 낮은 MSE를 기록하였다. 이는 구조적 KD가 모델 용량의 격차를 극복하고 오히려 역전시킬 수 있음을 시사한다.
- **아키텍처 영향**: MLP 모델은 Global KD를 적용하더라도 GPT-PPG-1m의 성능에 미치지 못했다. 이는 단순한 얕은 구조로는 PPG의 복잡한 동역학을 모델링하는 데 한계가 있음을 보여준다.

### 효율성 분석
- **추론 속도 및 메모리**: GPT-PPG-1m은 GPT-PPG-19m 및 PaPaGei에 비해 추론 속도(Throughput)가 최대 7배 빠르며, 메모리 사용량은 최대 19배 적다. 이는 파운데이션 모델의 성능을 유지하면서도 웨어러블 기기 배포가 가능한 수준의 효율성을 달성했음을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 PPG 신호의 도메인 지식(형태와 리듬)을 지식 증류 과정에 직접적으로 통합함으로써, 단순한 특징 일치를 넘어선 효과적인 압축 방법을 제시하였다. 특히 $\gamma$(패치 수준 손실 가중치)에 대한 민감도 분석 결과, $\gamma=1$에서 최적의 MAE가 나타난 점은 PPG 분석에서 국소적 특성 보존이 필수적임을 입증한다.

### 한계 및 논의사항
- **데이터셋 의존성**: StanfordAF 데이터셋의 경우 교사 모델 간의 성능 차이가 DaLiA보다 작게 나타났는데, 이는 특정 태스크에서는 교사의 성능보다 데이터셋 자체의 특성이 더 큰 영향을 미칠 수 있음을 시사한다.
- **하이퍼파라미터 민감도**: $\alpha$ 값에 따라 성능 변화가 크게 나타나는 경향이 있어, 최적의 가중치를 찾기 위한 탐색 과정이 필요하다.
- **범용성**: 본 연구는 파운데이션 모델을 교사로 사용하였으나, 다양한 크기와 구조의 다른 교사 모델들에 대해서도 동일한 효과가 있는지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

**PPG-DISTILL**은 거대 PPG 파운데이션 모델의 지식을 경량 모델로 전이하기 위해 전역적 지식과 더불어 **국소적 파형 형태(Morphology)** 및 **패치 간 리듬(Rhythm)**을 보존하는 패치 수준 증류 기법을 제안한다. 이를 통해 모델 파라미터를 19배 줄이고 추론 속도를 7배 높이면서도, 심박수 추정 및 심방세동 탐지 성능을 획기적으로 향상시켜 웨어러블 기기에서의 실시간 PPG 분석 가능성을 열었다.