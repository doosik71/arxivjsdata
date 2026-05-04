# TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting

Md Atik Ahamed and Qiang Cheng (2024)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 데이터의 장기 예측(Long-term Time-series Forecasting, LTSF)에서 발생하는 세 가지 핵심 난제를 해결하고자 한다. 첫째는 시계열 데이터 내의 장기 의존성(long-term dependencies)을 효과적으로 포착하는 것이며, 둘째는 데이터 양에 따른 모델 파라미터의 선형적 확장성(linear scalability)을 확보하는 것이고, 셋째는 메모리 사용량을 최소화하여 계산 효율성을 높이는 것이다.

기존의 Transformer 기반 모델들은 self-attention 메커니즘을 통해 장기 의존성을 잘 포착하지만, 시퀀스 길이에 대해 이차 복잡도(quadratic complexity)를 가지므로 메모리 사용량이 급격히 증가하고 확장성이 떨어진다. 반면 DLinear와 같은 선형 모델들은 효율적이지만, 단순한 MLP 구조로 인해 복잡한 장기 상관관계를 충분히 학습하지 못하는 한계가 있다. 따라서 본 연구의 목표는 Mamba라는 State-Space Model(SSM)을 활용하여, Transformer 수준의 예측 정확도를 유지하면서도 선형적인 복잡도와 높은 메모리 효율성을 동시에 달성하는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba 모델의 선택적 스캔(selective scan) 능력을 시계열 데이터의 특성에 맞게 최적화한 **Integrated Quadruple-Mamba** 아키텍처를 설계한 것이다. 주요 기여 사항은 다음과 같다.

1. **Mamba 기반의 LTSF 모델 제안**: 순수 SSM 모듈을 사용하여 다변량 시계열의 장기 의존성을 포착하는 TimeMachine을 제안하였다. 이를 통해 선형적 확장성과 매우 작은 메모리 footprint를 달성하였다.
2. **Channel-Mixing 및 Channel-Independence의 통합**: 데이터의 특성에 따라 채널 간 상관관계를 활용하는 channel-mixing 방식과 각 채널을 독립적으로 처리하는 channel-independence 방식을 모두 수용할 수 있는 통합 구조를 설계하였다.
3. **다중 스케일 컨텍스트 추출**: 두 단계의 해상도 감소(downsampling) 과정을 통해 고해상도와 저해상도에서 각각 전역적(global) 및 지역적(local) 컨텍스트 큐를 추출하여 예측의 정확도를 높였다.

## 📎 Related Works

논문에서는 LTSF 접근 방식을 세 가지 범주로 분류하여 설명한다.

1. **Non-Transformer 기반 지도 학습**: ARIMA, RNN과 같은 고전적 방법과 DLinear, TiDE 등 MLP 기반 모델, TimesNet과 같은 CNN 기반 모델이 포함된다. 이들은 효율적이지만 복잡한 장기 상관관계 포착에 한계가 있다.
2. **Transformer 기반 지도 학습**: iTransformer, PatchTST, Autoformer 등이 있으며, self-attention을 통해 높은 정확도를 보이지만 이차 복잡도로 인한 계산 비용이 매우 크다.
3. **자기지도 표현 학습(Self-supervised Representation Learning)**: MTS의 유용한 표현을 학습하여 다운스트림 태스크에 적용하는 방식이나, 현재까지는 지도 학습 방식에 비해 성능이 낮다고 평가된다.

본 연구는 기존 SSM 기반 모델들이 시계열 데이터의 컨텍스트를 효과적으로 표현하지 못했다는 점과, 최근에야 내용/컨텍스트 선택적 SSM(Mamba)이 개발되었다는 점에 주목하여, 이를 LTSF에 최적화된 구조로 적용함으로써 기존 모델들과 차별화한다.

## 🛠️ Methodology

### 전체 파이프라인

TimeMachine의 전체 구조는 **정규화 $\to$ 임베딩 $\to$ 4개의 Mamba 블록 $\to$ 출력 투영** 단계로 구성된다.

### 1. 정규화 (Normalization)

입력 데이터 $\mathbf{x}$는 $\text{RevIN}$(Reversible Instance Normalization) 또는 Z-score 정규화를 통해 $\mathbf{x}^{(0)}$으로 변환된다. 실험적으로 $\text{RevIN}$이 더 효과적인 것으로 나타났다.

### 2. 채널 처리 방식 (Channel Handling)

* **Channel-Independence**: 각 채널을 독립적인 1차원 벡터로 처리하여 오버피팅을 줄인다. 입력 형태를 $(B, M, L) \to (B \times M, 1, L)$로 변형한다.
* **Channel-Mixing**: 채널 간 상관관계가 강한 경우, $M$개의 채널을 함께 처리하여 전역적인 컨텍스트를 학습한다.

### 3. 임베딩 표현 (Embedded Representations)

입력 시퀀스는 두 단계의 MLP($E_1, E_2$)를 통해 고정된 길이의 토큰으로 압축된다.
$$\mathbf{x}^{(1)} = E_1(\mathbf{x}^{(0)}), \quad \mathbf{x}^{(2)} = E_2(\text{DO}(\mathbf{x}^{(1)}))$$
여기서 $\mathbf{x}^{(1)}$의 길이는 $n_1$, $\mathbf{x}^{(2)}$의 길이는 $n_2$ ($n_1 > n_2$)로 설정되어 다중 스케일의 표현을 생성한다.

### 4. Integrated Quadruple Mambas

모델의 핵심인 4개의 Mamba 블록은 $n_1$ 스케일(outer)과 $n_2$ 스케일(inner)에서 각각 쌍을 이루어 배치된다.

**Mamba의 기본 메커니즘:**
연속 시간 SSM은 다음과 같은 상태 방정식으로 정의된다.
$$\frac{dh(t)}{dt} = \mathbf{A}h(t) + \mathbf{B}u(t), \quad v(t) = \mathbf{C}h(t)$$
이를 이산화(discretization)하면 다음과 같다.
$$h_k = \bar{\mathbf{A}}h_{k-1} + \bar{\mathbf{B}}u_k, \quad v_k = \mathbf{C}h_k$$
여기서 $\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$, $\bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I})\Delta \mathbf{B}$이다. Mamba의 핵심은 $\mathbf{B}, \mathbf{C}, \Delta$가 입력 $u$에 따라 변하는 함수라는 점이며, 이를 통해 입력 내용에 따라 정보를 선택적으로 유지하거나 잊는 '선택성(selectivity)'을 가진다.

**구조적 특징:**

* 각 스케일(outer, inner)마다 두 개의 Mamba가 존재한다.
* **Channel-Independence 상황**에서는 한 Mamba는 전역 컨텍스트를, 다른 Mamba는 전치된(transposed) 데이터를 입력받아 지역 컨텍스트를 학습함으로써 상호 보완한다.
* **Channel-Mixing 상황**에서는 모든 Mamba가 채널 간 상관관계를 학습하며 전역적 컨텍스트를 추출한다.

### 5. 출력 투영 (Output Projection)

Mamba 블록의 결과물은 두 단계의 MLP($P_1, P_2$)를 통해 최종 예측값으로 변환된다. $P_1$은 저해상도 표현($n_2$)을 고해상도($n_1$)로 확장하고, $P_2$는 최종적으로 예측 길이 $T$만큼의 시퀀스를 생성한다. 이때 학습 안정성을 위해 잔차 연결(residual connections)이 적용된다.
$$\mathbf{y} = P_2(\mathbf{x}^{(5)} \parallel (\mathbf{x}^{(4)} \oplus \mathbf{x}^{(1)}))$$
(여기서 $\parallel$은 concatenation, $\oplus$는 element-wise addition을 의미한다.)

## 📊 Results

### 실험 설정

* **데이터셋**: Weather, Traffic, Electricity, ETTh1, ETTh2, ETTm1, ETTm2 (총 7종).
* **지표**: MSE (Mean Square Error), MAE (Mean Absolute Error).
* **비교 모델**: iTransformer, PatchTST, DLinear, Autoformer 등 11개의 SOTA 모델.
* **설정**: Look-back window $L=96$, 예측 길이 $T \in \{96, 192, 336, 720\}$.

### 주요 결과

1. **예측 정확도**: Table 2 결과에 따르면, TimeMachine은 거의 모든 데이터셋과 예측 길이에서 기존 SOTA 모델들을 압도하거나 대등한 성능을 보였다. 특히 채널 수가 많은 Traffic과 Electricity 데이터셋에서 iTransformer와 경쟁 가능한 수준의 높은 성능을 기록하였다.
2. **확장성 및 메모리 효율성**: Figure 4에서 Transformer 기반 모델들이 $L$이 증가함에 따라 메모리 사용량이 급격히 증가하는 반면, TimeMachine은 DLinear와 유사한 수준의 매우 낮은 메모리 footprint를 유지하였다.
3. **Look-back Window 영향**: $L$을 $\{192, 336, 720\}$으로 확장했을 때 대부분의 데이터셋에서 성능이 더욱 향상되었으며, 이는 TimeMachine이 매우 긴 입력 시퀀스도 효율적으로 처리할 수 있음을 입증한다.

## 🧠 Insights & Discussion

### 강점

TimeMachine은 Transformer의 장점인 '장기 의존성 포착 능력'과 선형 모델의 장점인 '계산 효율성'을 성공적으로 결합하였다. 특히 $\text{Mamba}$의 선택적 스캔 메커니즘을 시계열의 다중 스케일 구조에 적용하여, 전역적 흐름과 지역적 변동을 동시에 학습할 수 있게 한 점이 주효했다.

### 한계 및 논의

* **특정 데이터셋의 성능**: Weather 데이터셋의 짧은 예측 구간($T$)에서는 다른 모델보다 순위가 밀리는 모습이 관찰되었으며, 이는 향후 개선 과제로 남아있다.
* **정성적 분석**: 시각화 결과(Figure 3)에서 전반적으로 Ground Truth를 잘 따라가지만, 일부 구간에서 실제 값과의 정렬(alignment)을 더 정밀하게 개선할 여지가 있다.
* **하이퍼파라미터 민감도**: 상태 확장 계수(State Expansion Factor, $N$)가 높을수록 성능이 향상되는 경향이 있으나, 이는 계산 비용 증가와 트레이드오프 관계에 있다. 본 논문에서는 $N=256$을 최적으로 제안하였다.

## 📌 TL;DR

본 논문은 Mamba(Selective SSM)를 기반으로 한 **TimeMachine**을 제안하여, 다변량 시계열의 장기 예측(LTSF) 문제를 해결하였다. 4개의 Mamba 블록으로 구성된 통합 아키텍처를 통해 채널 독립성과 믹싱 방식을 모두 지원하며, 다중 스케일 컨텍스트를 추출함으로써 **Transformer급의 정확도와 선형 모델급의 효율성(메모리, 속도)**을 동시에 달성하였다. 이 연구는 향후 매우 긴 시퀀스를 처리해야 하는 엣지 컴퓨팅 환경이나 초장기 시계열 예측 연구에 중요한 기초가 될 것으로 보인다.
