# ProductGraphSleepNet: Sleep Staging using Product Spatio-Temporal Graph Learning with Attentive Temporal Aggregation

Aref Einizade, Samaneh Nasiri, Sepideh Hajipour Sardouie, and Gari Clifford (2022)

## 🧩 Problem to Solve

수면 단계 분류(Sleep Stage Scoring)는 수면 생리학의 이해와 수면 관련 질환 진단에 있어 매우 중요한 과정이다. 전통적으로 수면 단계 분류는 전문가가 수면다원검사(Polysomnography, PSG) 데이터를 시각적으로 분석하여 판독하는 방식에 의존해 왔으나, 이는 많은 시간이 소요될 뿐만 아니라 판독자 간의 주관적인 차이로 인한 정답 레이블의 노이즈 문제가 발생할 수 있다.

최근 딥러닝 기반의 자동 수면 단계 분류 모델들이 제안되었지만, 기존의 Convolutional Neural Networks(CNN)나 Recurrent Neural Networks(RNN) 기반 하이브리드 모델들은 다음과 같은 한계를 가진다. 첫째, 이들 네트워크는 데이터를 일반적인 그리드(Euclidean space) 형태로 처리하기 때문에 뇌 영역 간의 기하학적 연결성(Non-Euclidean geometry)을 무시한다. 둘째, 시간적으로 인접한 수면 에포크(epoch)들 사이의 순차적 연결성과 그 동역학(dynamics)을 충분히 반영하지 못한다. 결과적으로 이러한 한계는 임상 의사들이 모델의 출력 결과를 해석하는 것을 어렵게 만든다. 본 논문의 목표는 뇌의 공간적 연결성과 수면 단계 전이의 시간적 특성을 동시에 학습하여, 성능과 의료적 해석 가능성을 모두 갖춘 자동 수면 단계 분류 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Product Graph Learning(PGL)**을 활용하여 공간(Spatial) 그래프와 시간(Temporal) 그래프를 결합한 시공간 그래프(Spatio-temporal graph)를 적응적으로 학습하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **시공간 그래프의 공동 학습 및 해석**: 각 수면 단계에 대해 공간적 연결성과 시간적 연결성을 함께 학습함으로써, 모델의 결정 근거를 의료적으로 해석할 수 있게 하였다. 특히 기존 연구(GraphSleepNet)가 단순히 주변 에포크의 특징을 연결(concatenation)했던 것과 달리, 시간적 연결 가중치를 직접 학습한다.
2. **Bi-directional Gated Recurrent Units(BiGRU) 도입**: 수면 단계 간의 전이 규칙을 학습하고, 이를 통해 시간 그래프의 노드에 해당하는 특징 벡터를 생성한다.
3. **Graph-wise Attention Network(GwAT) 제안**: 기존의 Graph Attention Network(GAT)를 수정하여 그래프 분류 시나리오에 적용 가능한 GwAT를 제안하였다. 이를 통해 순차적 에포크들 간의 시간적 가중치 중요도를 적응적으로 학습한다.

## 📎 Related Works

기존의 자동 수면 단계 분류 접근 방식은 크게 다음과 같이 분류된다.

- **CNN-RNN 하이브리드 모델**: DeepSleepNet, SeqSleepNet 등은 CNN으로 특징을 추출하고 RNN으로 시간적 의존성을 캡처하여 우수한 성능을 보였으나, 뇌의 비유클리드(non-Euclidean) 기하학적 정보를 반영하지 못한다는 한계가 있다.
- **Graph Neural Networks(GNN) 및 GCN**: 그래프 구조를 통해 노드 간 관계를 학습하려는 시도가 있었으며, 최근 GraphSleepNet과 같은 적응형 그래프 학습 모델이 등장하였다. 그러나 GraphSleepNet은 순차적 에포크 간의 상호작용 가중치를 무시하고 단순히 특징을 결합하여 사용했다는 한계가 있다.

본 논문은 고정된 그래프가 아니라 데이터로부터 최적의 그래프 구조를 찾아내는 **Adaptive Graph Learning**을 채택하며, 특히 공간과 시간을 분리하여 학습한 뒤 결합하는 Product Graph 개념을 도입하여 계산 효율성과 해석력을 높였다.

## 🛠️ Methodology

### 1. 전체 파이프라인

입력 데이터인 PSG 신호로부터 Differential Entropy(DE) 특징을 추출한 후, [공간 주의 집중 레이어 $\rightarrow$ PGL 레이어 $\rightarrow$ Attentive GCN $\rightarrow$ BiGRU $\rightarrow$ GwAT] 순으로 처리가 진행되어 최종 수면 단계(Wake, REM, N1, N2, N3)를 분류한다.

### 2. Product Graph Learning (PGL)

본 모델은 전체 그래프 $G_N$을 더 작은 두 개의 팩터 그래프인 공간 그래프 $G_Q$와 시간 그래프 $G_P$의 데카르트 곱(Cartesian product)으로 정의한다. 이때 전체 라플라시안 행렬 $L_N$은 다음과 같이 크로네커 합(Kronecker sum)으로 표현된다.
$$L_N = L_P \oplus L_Q = L_P \otimes I_Q + I_P \otimes L_Q$$
여기서 $\otimes$는 크로네커 곱이며, $I$는 단위 행렬이다. 학습 목표는 그래프 신호의 매끄러움(smoothness)을 측정하는 Total Variation(TV)을 최소화하는 것이며, 손실 함수는 다음과 같이 정의된다.
$$\{W_P, W_Q\} = \arg \min_{W_P, W_Q} \frac{1}{2} \sum_{m=1}^{M} \left[ \sum_{r,s} ||Y_m(r,:) - Y_m(s,:)||_2^2 W_P(r,s) + \sum_{r',s'} ||Y_m(:,r') - Y_m(:,s')||_2^2 W_Q(r',s') \right]$$
이 PGL 손실 함수는 분류 손실 함수(Cross entropy)와 함께 최적화된다.

### 3. 주요 구성 요소

- **Spatial Attention Layer**: 수면 단계 전이에 따라 변화하는 동적 공간 주의 집중 가중치 $P$를 학습한다.
- **Attentive Graph Convolutional Layer**: 학습된 공간 그래프 $L_Q$와 Chebyshev 다항식 전개를 사용하여 공간적 정보를 집계한다.
$$X^{(1)}(i,:,:) = \sum_{k=0}^{K} [T_k(\tilde{L}_Q) \odot P] X^{(0)}(i,:,:) \Theta^{(k)}$$
여기서 $P$는 공간 주의 집중 가중치이며, $\Theta^{(k)}$는 학습 가능한 계수 행렬이다.
- **BiGRU**: GCN의 출력을 평탄화(flatten)하여 입력으로 받고, 수면 단계의 전이 규칙을 학습하여 시간 그래프의 노드 특징 벡터 $X^{(3)}$를 생성한다.
- **Graph-wise Attention Network (GwAT)**: 학습된 시간 그래프 $W_P$와 BiGRU의 출력 $X^{(3)}$를 이용하여 에포크 간의 중요도를 학습한다.
$$\hat{\alpha}(r,s) = W_P(r,s) \times \exp(\text{LeakyReLU}(\gamma^T [X^{(3)}(r,:)W || X^{(3)}(s,:)W]))$$
최종적으로 $K_{GwAT}$개의 어텐션 헤드를 사용하여 특징을 통합하고 소프트맥스 함수를 통해 최종 단계를 분류한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MASS-SS3 (62명 건강한 피험자) 및 SleepEDF (20명 건강한 피험자) 데이터셋을 사용하였다.
- **전처리**: PSG 채널을 9개의 주파수 대역으로 분해하고 DE 특징을 추출하였다. 타겟 에포크 전후로 각각 4개씩, 총 9개의 에포크($P=9$)를 입력으로 사용하였다.
- **평가 지표**: Accuracy, F1-score, Kappa 지수를 사용하였다.

### 2. 정량적 결과

- **MASS-SS3**: Accuracy 0.867, F1-score 0.818, Kappa 0.802를 달성하여 SOTA 모델들과 대등하거나 더 우수한 성능을 보였다.
- **SleepEDF**: Accuracy 0.838, F1-score 0.774, Kappa 0.775를 기록하였다.
- **Ablation Study**: BiGRU를 사용하지 않은 Baseline 1과 GwAT를 단순 연결로 대체한 Baseline 2보다 제안 방법의 성능이 더 높게 나타났으며, 이는 시간적 전이 규칙 학습과 적응적 어텐션의 중요성을 입증한다.

### 3. 정성적 분석 및 해석 가능성

학습된 그래프를 이진화하여 분석한 결과, 다음과 같은 의료적 통찰을 얻었다.

- **공간 그래프**: Wake 상태에서 기능적 연결성이 가장 높았으며, 이는 Non-REM 단계에서 수면 안정을 위해 시상하부 연결성이 감소한다는 기존 신경과학 연구와 일치한다. 또한 REM 단계에서는 후두엽(occipital region)의 활동과 연결성이 눈에 띄게 증가함을 확인하였다.
- **채널 간 연결**: REM 단계에서 ECG와 EEG 채널 간의 연결성이 크게 증가하는데, 이는 뇌가 자율 신경계에 미치는 영향이 커진다는 기존 보고와 일치한다.
- **시간 그래프**: 시간적 연결이 단순히 인접한 에포크(one-hop) 간에만 이루어지는 것이 아니라, REM 단계 등에서 멀리 떨어진 에포크 간의 연결이 관찰되어, 단순 연결(concatenation)보다 그래프 기반 학습이 더 효과적임을 보여주었다.

## 🧠 Insights & Discussion

본 논문은 딥러닝 모델의 고질적인 문제인 '블랙박스' 특성을 해결하기 위해 GSP(Graph Signal Processing)와 PGL을 도입하여 해석 가능성을 확보하였다. 특히, 학습된 그래프 구조가 실제 신경과학적 발견(후두엽의 REM 활성, 시상하부의 NREM 연결성 감소 등)과 일치한다는 점은 모델이 단순히 통계적 패턴을 찾는 것이 아니라 실제 생리학적 특징을 학습하고 있음을 시사한다.

다만, 본 연구는 건강한 피험자의 데이터셋을 주로 사용하였으므로, 수면 장애가 있는 환자 데이터에서도 동일한 연결성 패턴이 나타날지는 미지수이다. 또한, PGL을 통해 그래프를 학습하는 과정에서의 계산 복잡도와 실시간 적용 가능성에 대한 심층적인 논의가 부족한 점이 아쉽다. 그럼에도 불구하고, 공간적 뇌 연결성과 시간적 전이 특성을 통합적으로 모델링한 점은 향후 수면 진단 보조 시스템 개발에 중요한 기여를 할 것으로 판단된다.

## 📌 TL;DR

본 논문은 수면 단계 분류를 위해 공간적 뇌 연결성과 시간적 전이 특성을 공동으로 학습하는 **ProductGraphSleepNet**을 제안한다. Product Graph Learning(PGL), BiGRU, 그리고 수정된 Graph Attention Network(GwAT)를 결합하여 성능을 높였으며, 학습된 그래프를 통해 수면 단계별 뇌의 연결성 변화를 의료적으로 해석할 수 있게 하였다. 이 연구는 자동 수면 판독의 정확도를 높임과 동시에 임상적 근거를 제공할 수 있다는 점에서 향후 정밀 의료 및 수면 진단 시스템에 적용될 가능성이 매우 높다.
