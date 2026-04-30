# Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting

Divij Gupta, Anubhav Bhatti, Suraj Parmar, Chen Dan, Yuwei Liu, Bingjie Shen, San Lee (2024)

## 🧩 Problem to Solve

본 논문은 시계열 데이터에 특화된 기초 모델(Time Series Foundational Models, TSFMs)을 특정 도메인, 특히 의료 데이터와 같이 데이터가 희소하고 민감한 'Out-of-Domain' 모달리티에 효율적으로 적응시키는 문제를 해결하고자 한다. 

일반적으로 거대 모델을 특정 태스크에 맞게 조정하기 위해서는 Full Fine-tuning을 수행하지만, 이는 막대한 계산 자원을 소모할 뿐만 아니라 모델이 가진 일반화 능력을 손상시킬 위험(Catastrophic Forgetting)이 있다. 특히 의료 분야의 중환자실(ICU) 내 패혈증(Sepsis) 환자의 생체 신호 예측과 같은 특수 도메인에서는 가용한 데이터가 제한적이므로, 적은 양의 데이터로도 효율적으로 성능을 높일 수 있는 적응 방법론이 필수적이다. 따라서 본 연구의 목표는 Low-Rank Adaptation(LoRA)을 TSFM에 적용하여 파라미터 효율성을 높이면서도 도메인 특화 예측 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 거대 시계열 기초 모델의 가중치를 모두 수정하는 대신, Low-Rank 분해 행렬을 추가하여 매우 적은 수의 파라미터만을 학습시키는 LoRA 기법을 TSFM에 도입하는 것이다. 이를 통해 모델의 기존 지식을 보존하면서도 새로운 도메인의 특성을 빠르게 학습할 수 있도록 설계하였다.

주요 기여 사항은 다음과 같다.
- Lag-Llama, MOIRAI, Chronos라는 세 가지 대표적인 TSFM 아키텍처에 LoRA를 적용하여 생체 신호 예측 성능을 검증하였다.
- Zero-shot, Full Fine-tuning, 그리고 LoRA-based Fine-tuning의 성능을 비교 분석하여, LoRA가 계산 비용을 획기적으로 줄이면서도 Full Fine-tuning과 대등하거나 오히려 더 나은 성능을 보임을 입증하였다.
- 특정 모델(Chronos variants)의 경우 LoRA를 통해 기존의 State-of-the-Art(SOTA) 모델의 성능을 능가하거나 근접함을 확인하였다.
- LoRA의 Rank($r$) 변화에 따른 성능 변화와 학습 가능 파라미터 수 사이의 트레이드-오프 관계를 분석하였다.

## 📎 Related Works

### Parameter-Efficient Fine-Tuning (PEFT)
PEFT는 모델의 전체 파라미터를 업데이트하지 않고 일부만 수정하는 기법으로, 크게 두 가지로 나뉜다.
- **Selective Approach**: Attention 레이어나 Bias, LayerNorm과 같은 특정 파라미터만 선택적으로 학습하는 방식이다.
- **Additive Approach**: 기존 모델에 새로운 가중치(Adapter)를 추가하고 이를 학습시키는 방식이다. LoRA는 대표적인 Additive 접근법으로, 가중치 행렬을 두 개의 낮은 랭크 행렬의 곱으로 표현하여 파라미터 수를 줄인다.

### Time Series Foundational Models (TSFM)
최근 다양한 TSFM들이 제안되었으며, 본 논문은 다음의 모델들을 활용한다.
- **Lag-Llama**: LLaMA 아키텍처에서 영감을 받은 Decoder-only Transformer 모델로, 과거 관측치와 Lagged feature, 타임스탬프를 사용한다.
- **MOIRAI**: Encoder-only Transformer 모델로, 개별 데이터 포인트가 아닌 비중첩 패치(Non-overlapping patches) 단위로 데이터를 처리하여 컨텍스트 정보를 강화한다.
- **Chronos**: T5 모델의 Encoder-Decoder 구조를 활용하며, 실수 값을 Binning 및 Scaling 함수를 통해 이산화(Discretization)하여 처리하는 특징이 있다.

기존의 연구들이 LLM을 시계열 예측에 적응시키려 했다면, 본 연구는 처음부터 시계열 데이터로 학습된 기초 모델(TSFMs)에 집중하여 LoRA의 효용성을 탐구한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### Problem Statement
시계열 데이터셋 $\mathcal{D}$에서 하나의 시계열 $t$를 $t = t_1, t_2, \dots, t_N$으로 정의한다. 컨텍스트 길이 $C$가 주어졌을 때, 과거 데이터 $t_{1:C} = t_1, t_2, \dots, t_C$를 이용하여 미래의 예측 구간 $t_{C:C+h} = t_{C+1}, t_{C+2}, \dots, t_{C+h}$를 예측하는 것을 목표로 한다. 여기서 $h$는 예측 호라이즌(Forecast horizon)이다.

### Backbone Architecture
본 연구에서는 Lag-Llama, MOIRAI, Chronos 세 가지 모델을 백본으로 사용한다. 공정한 비교를 위해 모든 모델을 단변량(Univariate) 설정으로 구현하여, 타임스탬프를 제외한 추가 공변량 없이 하나의 입력 변수로 하나의 출력 변수를 예측하도록 구성하였다.

### Low-Rank Adaptation (LoRA)
LoRA는 기존의 가중치 행렬 $W \in \mathbb{R}^{d \times k}$를 고정(Freeze)하고, 두 개의 작은 행렬 $A \in \mathbb{R}^{r \times k}$와 $B \in \mathbb{R}^{d \times r}$의 곱을 더해 가중치를 업데이트한다. 여기서 $r$은 랭크(Rank)이며 $r \ll \min(d, k)$ 조건을 만족한다.

업데이트된 가중치 행렬 $W'$는 다음과 같은 방정식으로 표현된다.
$$W' = W + \frac{\alpha}{r} BA$$
여기서 $\alpha$는 업데이트가 기존 가중치에 미치는 영향력을 조절하는 스케일링 파라미터이다. 

학습 절차는 다음과 같다.
1. 기존 모델의 모든 가중치 $W$는 고정한다.
2. $B$는 0으로 초기화하고, $A$는 가우시안 분포에서 샘플링된 작은 랜덤 값으로 초기화한다.
3. Transformer 아키텍처의 Multi-Head Self-Attention 모듈 내의 네 가지 가중치 행렬인 Query($W_q$), Key($W_k$), Value($W_v$), Output($W_o$)에 LoRA를 적용하여 Attention 메커니즘의 적응력을 극대화한다.
4. 도메인 특화 데이터(생체 신호)를 사용하여 오직 $A$와 $B$ 행렬만을 학습시킨다.

## 📊 Results

### 실험 설정
- **데이터셋**: eICU Collaborative Research Database를 사용하여 패혈증 환자의 평균 혈압(MeanBP)과 심박수(HR)를 예측하였다.
- **전처리**: Forward fill을 통한 결측치 보간, 저역 통과 필터(Low-pass filter)를 통한 노이즈 제거, Global min-max scaling을 적용하였다.
- **윈도우 설정**: 컨텍스트 윈도우 6시간(72 points), 예측 윈도우 3 hours(36 points)로 설정하였다(5분 간격 샘플링).
- **평가 지표**: Mean Squared Error(MSE), Dynamic Time Warping(DTW), Mean Average Percentage Error(MAPE)를 사용하였다.
- **측정 방법**: 확률적 예측 특성을 고려하여 20개 샘플의 중앙값(Median)을 예측값으로 사용하였으며, 10회 실행 결과의 평균을 측정하였다.

### 정량적 결과
실험 결과, 다음과 같은 사실이 확인되었다.
- **Chronos의 우수성**: Zero-shot 설정에서 Chronos가 Lag-Llama와 MOIRAI보다 일관되게 우수한 성능을 보였다.
- **LoRA vs Full FT**: 
    - Lag-Llama의 경우 MeanBP 예측에서는 Full FT가 약간 우세했으나, HR 예측에서는 LoRA FT가 더 나은 성능을 보였다.
    - MOIRAI는 Full FT가 가장 좋았으나 LoRA FT가 매우 근소한 차이로 뒤따랐다. 특히 모델 사이즈가 커질수록(Base, Large) Full FT 시 오히려 성능이 저하되는 경향이 나타났는데, 이는 파라미터가 너무 많은 패치 임베딩 및 디코딩 레이어가 과적합되었기 때문으로 추측된다.
    - Chronos는 MeanBP와 HR 모두에서 LoRA FT가 Full FT보다 성능이 좋거나 대등하게 나타났다.
- **SOTA 비교**: LoRA를 적용한 Chronos (Small) 모델은 HR 예측에서 기존 SOTA(Bhatti et al.)의 성능을 능가하였으며, Chronos (Tiny)는 MeanBP 예측에서 SOTA 수준에 근접하였다.

### Ablation Study
- **파라미터 수 대비 성능**: LoRA는 Full FT 대비 학습 파라미터 수를 획기적으로 줄이면서도 동등 이상의 성능을 유지하였다. 이는 기초 모델의 일반화 능력을 보존하면서 효율적으로 적응했음을 시사한다.
- **Rank ($r$)의 영향**: $r$ 값이 증가함에 따라 성능이 향상되다가 특정 지점에서 평탄해지는(Plateau) 현상이 관찰되었다. 특히 작은 모델일수록 $r$ 변화에 따른 성능 변화가 뚜렷했으며, 큰 모델일수록 영향이 적었다.

## 🧠 Insights & Discussion

본 연구는 TSFM에 LoRA를 적용함으로써 거대 모델의 효율적인 도메인 적응 가능성을 입증하였다. 

**강점 및 통찰:**
- **일반화 능력 유지**: Full FT는 도메인 특화 데이터가 적을 때 과적합을 유발하여 성능을 떨어뜨릴 수 있지만, LoRA는 원래의 가중치를 고정함으로써 기초 모델이 가진 일반적인 시계열 패턴 인식 능력을 보존하는 동시에 도메인 지식을 추가하는 효과를 거두었다.
- **효율성**: 극소수의 파라미터만 학습함으로써 계산 자원을 절약할 수 있으며, 다양한 태스크에 대해 각각의 LoRA 가중치 세트만 저장하면 되므로 메모리 효율성이 매우 높다.

**한계 및 논의 사항:**
- **단변량 예측의 한계**: 본 연구는 비교의 공정성을 위해 단변량 예측(Univariate forecasting)만 수행하였다. 하지만 실제 의료 데이터는 여러 생체 신호가 상호작용하는 다변량(Multivariate) 특성이 강하므로, 향후 다변량 설정으로의 확장이 필요하다.
- **모델별 반응 차이**: MOIRAI와 같이 패치 기반 구조를 가진 모델은 LoRA 적용 범위(Attention 레이어 한정) 외의 레이어(Patch embedding 등)가 성능에 큰 영향을 미칠 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 시계열 기초 모델(TSFMs)을 의료 도메인의 생체 신호 예측에 적응시키기 위해 **LoRA(Low-Rank Adaptation)**를 적용한 연구이다. 실험 결과, LoRA는 전체 파라미터를 학습시키는 Full Fine-tuning보다 훨씬 적은 비용으로 대등하거나 더 우수한 성능을 냈으며, 특히 **Chronos 모델과 결합했을 때 기존 SOTA 성능을 상회**하는 결과를 보였다. 이는 거대 시계열 모델을 데이터가 부족한 특수 도메인에 적용할 때 LoRA가 매우 강력하고 효율적인 도구가 될 수 있음을 의미한다.