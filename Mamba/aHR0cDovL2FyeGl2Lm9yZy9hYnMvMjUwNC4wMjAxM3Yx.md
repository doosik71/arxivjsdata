# Attention Mamba: Time Series Modeling with Adaptive Pooling Acceleration and Receptive Field Enhancements

Sijie Xiong, Shuqing Liu, Cheng Tang, Fumiya Okubo, Haoling Xiong, and Atsushi Shimada (2025)

## 🧩 Problem to Solve

본 논문은 시계열 예측(Time Series Forecasting, TSF)에서 시계열 데이터의 시간적 의존성(Temporal Dependencies, TD)과 변수 간 상관관계(Inter-variate Correlations, VC)를 효과적으로 포착하는 문제를 해결하고자 한다.

기존의 모델들은 다음과 같은 한계점을 가지고 있다. 첫째, Transformer 기반 모델은 전역적/지역적 패턴 파싱 능력이 뛰어나지만, 데이터와 변수가 증가함에 따라 연산 비용이 기하급수적으로 증가하는 quadratic complexity 문제를 겪는다. 둘째, Linear 기반 모델은 연산 속도는 빠르지만 비선형 의존성 모델링 능력이 부족하여 복잡한 시나리오에서 성능이 크게 저하된다. 셋째, 최근 주목받는 Mamba 기반의 State Space Models(SSMs)는 선형적인 연산 복잡도와 높은 정확도를 동시에 달성했으나, 합성곱(Convolution)으로 인한 제한된 수용 영역(Receptive Field)과 attention 메커니즘 내에서의 비선형 의존성 모델링 부족이라는 문제점을 안고 있다.

따라서 본 연구의 목표는 Mamba의 효율성을 유지하면서도, 제한된 수용 영역을 확장하고 비선형 의존성 추출 능력을 강화하여 복잡한 시계열 데이터에서도 높은 예측 성능을 내는 Attention Mamba 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Adaptive Pooling 블록과 Bidirectional Mamba 블록을 결합하여 연산 효율성과 모델의 표현력을 동시에 높이는 것이다.

1. **Adaptive Pooling 블록 설계**: Query와 Key의 차원을 adaptive average pooling과 max pooling을 통해 $1/4$ 수준으로 축소한다. 이를 통해 attention 연산 속도를 가속화하는 동시에, 전역적인 정보를 통합하여 Mamba 모델의 고질적인 문제인 제한된 수용 영역 문제를 해결한다. 또한, pooling 과정의 비선형성을 통해 Linear 기반 모델의 단점을 보완한다.
2. **Attention Mamba 프레임워크 제안**: 앞서 제안한 Adaptive Pooling 블록과 Bidirectional Mamba 블록을 통합한 새로운 구조를 제안한다. Bidirectional Mamba는 입력 데이터의 장단기 특징을 효율적으로 추출하여 attention 메커니즘의 Value 표현형으로 변환하는 역할을 수행한다.
3. **성능 검증**: 다양한 벤치마크 데이터셋을 통해 Attention Mamba가 기존의 Transformer, Linear, Mamba 기반 모델들보다 우수한 성능을 보임을 입증하고, 특히 수용 영역 확장과 비선형 의존성 추출 측면에서 이점이 있음을 증명하였다.

## 📎 Related Works

논문은 시계열 예측을 위한 세 가지 주요 접근 방식을 소개하며 각 방식의 한계를 지적한다.

* **Transformer-based models**: iTransformer, PatchTST 등이 있으며 전역적 패턴 파악에 능하지만, 데이터 규모가 커질수록 연산 비용이 급증한다. 이를 해결하기 위한 경량화 모델들은 특징 추출 능력이 떨어져 정확도가 낮아지는 trade-off가 발생한다.
* **Linear-based models**: RLinear, DLinear 등은 표준 선형 층을 사용하여 연산 효율을 극대화하지만, 비선형 의존성 모델링 능력이 부족하여 단순한 시나리오 외에는 성능이 저하된다.
* **Mamba-based models (SSMs)**: Mamba는 하드웨어 인식 선택 메커니즘(hardware-aware selective mechanisms)을 통해 선형 시간 복잡도로 시퀀스를 모델링한다. S-Mamba, SiMBA 등이 제안되었으나, 여전히 Transformer 대비 수용 영역이 제한적이라는 한계가 있다.

Attention Mamba는 이러한 기존 연구들의 간극을 메우기 위해, Mamba의 효율적인 구조 위에 Adaptive Pooling 기반의 attention 메커니즘을 얹어 수용 영역과 비선형성이라는 두 마리 토끼를 잡고자 한다.

## 🛠️ Methodology

Attention Mamba의 전체 구조는 입력 데이터를 Linear projection을 통해 Query($Q$)와 Key($K$)로 분리하고, Adaptive Pooling 블록을 통해 가중치(Weights)를 계산하며, Bidirectional Mamba 블록을 통해 정제된 Value($V$)를 생성하여 최종 예측값을 도출하는 파이프라인을 가진다.

### 1. Adaptive Pooling Acceleration

수용 영역을 확장하고 연산 속도를 높이기 위해 $Q$와 $K$에 대해 다음과 같은 융합 adaptive pooling을 적용한다.

$$F_Q = \text{AvgPooling}(Q) + \text{MaxPooling}(Q)$$
$$F_K = \text{AvgPooling}(K) + \text{MaxPooling}(K)$$

여기서 $Q, K \in \mathbb{R}^{B \times N \times E}$이며, $B$는 batch size, $N$은 변수 개수, $E$는 embedding dimension이다. AvgPooling과 MaxPooling은 마지막 두 차원을 $1/4$ 수준으로 축소하여 전역 특징을 포함하는 $F_Q, F_K \in \mathbb{R}^{B \times E/4 \times E/4}$를 생성한다.

이후 비선형성을 추가하기 위해 GeLU 활성화 함수를 적용하고 행렬 곱을 통해 점수(Scores)를 계산한다.

$$\text{PoolQ} = \text{GeLU}(F_Q)$$
$$\text{PoolK} = \text{GeLU}(F_K)$$
$$\text{Scores} = \text{PoolQ} @ \text{PoolK}$$

최종적으로 Softmax와 Linear Projection을 거쳐 가중치 $\text{Weights} \in \mathbb{R}^{B \times N \times E}$를 획득한다.

$$\text{Weights} = \text{LinearProjection}(\text{Softmax}(\text{Scores}))$$

### 2. Attention Mechanism with Bidirectional Mamba

입력 데이터를 정제된 Value로 변환하기 위해 Bidirectional Mamba 블록을 사용한다. 이는 정방향과 역방향 프로세스를 결합한 형태이다.

$$\text{NormalProcess: } \text{Mamba}(x)$$
$$\text{ReverseProcess: } \text{Mamba}(\text{RVS}(x))$$
$$\text{Value} = \text{RVS}(\text{NormalProcess} + \text{ReverseProcess})$$

여기서 $\text{RVS}$는 시퀀스를 반전시키는 연산이다. 최종 attention 결과물 $\text{Att}$는 다음과 같이 계산된다.

$$\text{Att} = \text{Weights} \times \text{Value}$$

## 📊 Results

### 1. 실험 설정

* **데이터셋**: Electricity, Weather, Solar-Energy, PEMS03, PEMS04, PEMS07, PEMS08 등 7개의 실세계 데이터셋을 사용하였다.
* **비교 모델**: iTransformer, PatchTST (Transformer-based), RLinear, DLinear (Linear-based), TimesNet (Conv-based), S-Mamba (Mamba-based) 등 10개의 SOTA 모델과 비교하였다.
* **평가 지표**: MSE(Mean Squared Error) 및 MAE(Mean Absolute Error)를 사용하였다.

### 2. 정량적 결과

* **비선형 의존성 추출**: Electricity, Weather, Solar-Energy 데이터셋에서 S-Mamba 및 iTransformer와 경쟁하며 최상위권 성능을 보였다. 특히 Solar-Energy 데이터셋에서는 S-Mamba 대비 MSE 기준 약 3.75%의 성능 향상을 보였다.
* **수용 영역 확장 효과**: 변수 개수가 많은 PEMS 데이터셋(PEMS03, 04, 07 등)에서 S-Mamba를 압도하는 성능 향상을 기록하였다. PEMS04의 경우 MSE 기준 13.59%라는 큰 폭의 향상을 보였는데, 이는 Adaptive Pooling이 전역 정보를 제공함으로써 수용 영역이 확장되었기 때문으로 분석된다.
* **종합 순위**: Friedman non-parametric test 결과, Attention Mamba는 MSE와 MAE 모두에서 전체 모델 중 1위를 기록하였다.

### 3. 복잡도 및 메모리 분석 (PEMS07 기준)

* **학습 시간**: S-Mamba 대비 학습 시간이 42.6% 감소하여 효율성이 증대되었다.
* **메모리 점유**: attention 메커니즘의 도입으로 S-Mamba 대비 메모리 사용량은 34% 증가하였다.
* **결론**: 메모리 사용량은 약간 증가했으나, 학습 시간 단축과 정확도 향상(S-Mamba 대비 10.5% 향상)의 이점이 더 크다고 판단하였다.

## 🧠 Insights & Discussion

본 논문은 Adaptive Pooling이 단순한 연산 가속을 넘어, 모델의 수용 영역을 넓히고 비선형성을 부여함으로써 시계열 예측 성능을 획기적으로 높일 수 있음을 보여주었다.

특히 흥미로운 점은 embedding dimension($E$)과 데이터셋의 변수 개수($N$) 사이의 관계이다. PEMS 데이터셋처럼 변수 개수가 embedding dimension보다 많은 경우 Adaptive Pooling의 전역 정보 통합 능력이 극대화되어 성능 향상 폭이 컸다. 반면, Weather 데이터셋처럼 변수 개수가 적은 경우에는 상대적으로 향상 폭이 적었다. 이는 현재의 Adaptive Pooling이 고정된 비율($1/4$)로 축소하는 방식이기 때문이며, 저자들은 향후 입력 데이터에 따라 동적으로 조절되는 adaptive pooling scheme이 필요함을 시사하였다.

또한, MSE 중심의 학습 프로세스가 MAE 측정값에 일부 부정적인 영향을 줄 수 있다는 점이 PEMS08 결과에서 관찰되었다. 이는 손실 함수 설정과 평가 지표 간의 불일치라는 일반적인 딥러닝의 한계를 보여준다.

## 📌 TL;DR

Attention Mamba는 Mamba의 효율적인 시퀀스 모델링 능력에 **Adaptive Pooling 기반의 attention 메커니즘**을 결합하여, 기존 Mamba 모델의 한계였던 **제한된 수용 영역(Receptive Field)과 비선형 의존성 추출 능력 부족을 해결**한 모델이다. 실험 결과, 특히 다변수 시계열 데이터셋에서 SOTA 성능을 달성하였으며, 학습 속도 또한 S-Mamba 대비 약 42% 개선하였다. 이 연구는 향후 효율적인 상태 공간 모델(SSM)과 attention 메커니즘의 하이브리드 설계 방향에 중요한 가이드라인을 제시한다.
