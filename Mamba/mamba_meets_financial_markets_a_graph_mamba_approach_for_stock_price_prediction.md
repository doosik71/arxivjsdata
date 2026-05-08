# Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction

Ali Mehrabian, Ehsan Hoseinzade, Mahdi Mazloum, and Xiaohong Chen (2025)

## 🧩 Problem to Solve

본 논문은 주식 시장의 수익률을 정확하게 예측하고자 하는 문제를 다룬다. 주식 가격 예측은 글로벌 경제에서 매우 중요하지만, 시장의 내재적인 복잡성으로 인해 예측이 매우 어려운 과제이다.

기존의 Transformer 기반 모델들은 Long Short-Term Memory (LSTM)나 Convolutional Neural Networks (CNN)보다 우수한 성능을 보였으나, 추론 시 발생하는 이차 복잡도(quadratic complexity)로 인해 계산 비용과 메모리 요구량이 매우 높다는 치명적인 단점이 있다. 이는 실시간 거래(real-time trading)나 매우 긴 시퀀스 데이터를 처리해야 하는 환경에서 실용성을 크게 떨어뜨린다. 따라서 본 연구의 목표는 Transformer 수준의 예측 성능을 유지하면서도 계산 복잡도를 획기적으로 낮춘 효율적인 주가 예측 프레임워크인 SAMBA를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 효율적인 시퀀스 모델링인 Mamba 아키텍처와 데이터 간의 관계를 모델링하는 Graph Neural Networks (GNN)를 결합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Bidirectional Mamba (BI-Mamba) 도입**: 기존 Mamba의 단방향 선택 메커니즘(selection mechanism)이 가진 한계를 극복하기 위해, 정방향과 역방향 시퀀스를 모두 처리하는 BI-Mamba 블록을 설계하여 과거 가격 데이터의 장기 의존성(long-term dependencies)을 효과적으로 포착한다.
2. **Adaptive Graph Convolutional (AGC) 블록 설계**: 일일 주식 특성(daily stock features) 간의 상호작용을 모델링하기 위해 적응형 그래프 구조를 도입하였다. 이는 고정된 그래프가 아닌, 학습 가능한 노드 임베딩을 통해 데이터로부터 직접 최적의 관계를 학습하게 함으로써 예측력을 높인다.
3. **효율성과 성능의 동시 달성**: Selective Scan 알고리즘을 통해 거의 선형(near-linear)에 가까운 계산 복잡도를 유지하면서도, 최신 baseline 모델들보다 뛰어난 예측 정확도를 달성하였다.

## 📎 Related Works

주가 예측을 위해 다양한 딥러닝 모델들이 연구되어 왔다.

- **MLP 및 CNN**: 비선형 관계를 포착하거나 시간적/종목 간 패턴을 추출하는 데 사용되었다.
- **LSTM**: 시계열 데이터의 장기 의존성을 학습하는 데 강점이 있다.
- **Transformer**: Self-attention 메커니즘을 통해 글로벌 의존성을 포착하며 높은 성능을 보였으나, 앞서 언급한 계산 복잡도 문제가 존재한다.
- **Mamba (State Space Models)**: 최근 제안된 Mamba는 Selective State Space Models (SSMs)를 기반으로 하며, 선형 시간 복잡도로 긴 시퀀스를 처리할 수 있어 Transformer의 대안으로 주목받고 있다. 특히 MambaStock과 같은 최신 연구가 있었으나, 이는 단방향 모델이라는 한계가 있다.
- **GNN**: 주식 간의 상관관계를 그래프 구조로 모델링하여 시장의 공동 움직임(co-movements)을 포착하는 데 효과적이다.

SAMBA는 이러한 Mamba의 효율성과 GNN의 관계 모델링 능력을 통합하여, 기존 모델들이 개별적으로 가졌던 계산 비용 문제나 단방향 정보 처리의 한계를 동시에 해결하고자 한다.

## 🛠️ Methodology

### 1. 문제 정의 (Problem Formulation)

본 연구는 절대적인 주가가 아닌, 주가 변동분(return ratio)을 예측하는 회귀 문제로 정의한다.

- 입력: 과거 $L$일 동안의 $N$개 일일 특성 행렬 $X \in \mathbb{R}^{L \times N}$.
- 목표 변수: 1일 후의 수익률 $so_{L+1} = \frac{c_{L+1} - c_L}{c_L}$ (여기서 $c$는 종가).

### 2. State Space Models (SSM) 기초

S4 모델과 Mamba의 근간이 되는 SSM은 연속 시간 시스템을 다음과 같은 미분 방정식으로 표현한다.
$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$
이를 이산화(discretization)하면 다음과 같은 재귀 형태로 변환되어 효율적인 계산이 가능해진다.
$$h_k = \hat{A}h_{k-1} + \hat{B}x_k, \quad y_k = Ch_k$$
여기서 $\hat{A} = \exp(\Delta A)$이며, $\Delta$는 step size를 의미한다.

### 3. Bidirectional Mamba (BI-Mamba) 블록

Mamba는 입력 데이터에 따라 파라미터 $\Delta, B, C$가 동적으로 변하는 Selective Scan 알고리즘을 사용한다. SAMBA는 이를 확장하여 양방향으로 처리하는 BI-Mamba를 제안한다.

- **절차**: 입력 $X$를 정방향 Mamba와 역방향 Mamba(anti-diagonal permutation matrix $P$ 적용)에 각각 통과시킨 후, 결과를 결합한다.
$$Y_1 = \text{Mamba}(X), \quad Y_2 = \text{Mamba}(PX)$$
- **결합 및 정규화**: $Y_3 = \text{Norm}(X + Y_1 + PY_2)$와 같이 잔차 연결(residual connection)과 Layer Normalization을 적용한다. 이후 Feed-Forward Network (FFN)를 통해 최종적으로 시간적 의존성을 포착한다.

### 4. Adaptive Graph Convolutional (AGC) 블록

특성 간의 복잡한 관계를 학습하기 위해 적응형 그래프 구조를 사용한다.

- **적응형 그래프 생성**: 학습 가능한 노드 임베딩 $\Psi \in \mathbb{R}^{N \times d_e}$를 정의하고, Gaussian 커널을 사용하여 노드 간의 거리 행렬 $D$를 구한 뒤 Softmax를 통해 인접 행렬 $\tilde{A}_G$를 생성한다.
$$\tilde{A}_G = \text{Softmax}(\exp(-\psi D))$$
- **그래프 컨볼루션**: Chebyshev 다항식 $T_n$의 $K$차 근사치를 사용하여 spectral filtering을 수행한다.
- **파라미터 효율화 (Matrix Factorization)**: 필터 가중치 $W_{\text{Filter}}$와 $b_{\text{Filter}}$의 파라미터 수가 너무 많아 발생하는 과적합(overfitting)을 방지하기 위해, 노드 임베딩 $\Psi$를 활용한 행렬 분해 기법을 적용하여 학습 파라미터 수를 획기적으로 줄였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 미국 주식 시장의 NASDAQ, NYSE, DJIA (2010년 1월 ~ 2023년 11월) 데이터 사용.
- **특성**: 82개의 일일 주식 특성($N=82$) 사용.
- **평가 지표**:
  - $\text{RMSE} \downarrow$: 예측 오차 측정.
  - $\text{IC} \uparrow$: 예측값과 실제값의 Pearson 상관계수 (예측 정확도).
  - $\text{RIC} \uparrow$: 예측 순위와 실제 순위의 Spearman 상관계수 (순위 예측력).
- **비교 대상**: LSTM, Transformer, FreTS, StockMixer, AGCRN, FourierGNN, MambaStock.

### 주요 결과

- **예측 성능**: SAMBA는 모든 데이터셋에서 RMSE, IC, RIC 모든 지표에서 baseline 모델들을 압도하였다.
  - **RMSE 개선**: 최우수 차순위 모델 대비 NASDAQ 11.72%, NYSE 10.07%, DJIA 6.90% 개선.
  - **IC/RIC 개선**: NASDAQ의 경우 최대 85.38% 및 80.30%라는 매우 높은 성능 향상을 보였다.
- **계산 효율성**:
  - Transformer나 AGCRN보다 훨씬 낮은 계산 복잡도(MACs)를 기록하였다.
  - AGCRN 대비 에폭당 학습 시간을 약 6.70% 단축시켰으며, 모델 파라미터 수 또한 효율적으로 관리되었다.

## 🧠 Insights & Discussion

### 강점 및 해석

SAMBA의 뛰어난 성능은 **시간적 장기 의존성 포착(BI-Mamba)**과 **특성 간 관계 모델링(AGC)**이라는 두 가지 핵심 요소가 조화롭게 작용했기 때문으로 분석된다. 특히 기존 MambaStock이 단방향성으로 인해 글로벌 의존성 포착에 한계가 있었던 점을 BI-Mamba가 성공적으로 해결하였으며, 고정된 그래프가 아닌 데이터 기반의 적응형 그래프를 통해 시장의 동적인 특성을 더 잘 반영할 수 있었다.

### 한계 및 논의사항

- **과적합 위험**: 본 논문에서는 행렬 분해 기법을 통해 파라미터를 줄여 과적합을 방지했으나, 금융 데이터의 특성상 여전히 noise에 취약할 가능성이 존재한다.
- **가정**: 본 연구는 1일 후의 수익률이라는 단기 예측에 집중하고 있다. 실제 투자 전략에서는 다단계(multi-step) 예측이 필요할 수 있다.
- **비판적 시각**: 계산 복잡도는 Transformer보다 낮지만, 단순 MLP 기반 모델(StockMixer 등)보다는 높다. 하지만 성능 향상 폭이 매우 크기 때문에 이는 충분히 납득 가능한 trade-off라고 볼 수 있다.

## 📌 TL;DR

본 논문은 주가 예측의 고질적인 문제인 **'예측 성능'**과 **'계산 복잡도'** 사이의 상충 관계를 해결하기 위해, 양방향 Mamba(BI-Mamba)와 적응형 그래프 컨볼루션(AGC)을 결합한 **SAMBA** 모델을 제안한다. 실험 결과, SAMBA는 기존 Transformer나 GNN 기반 모델보다 훨씬 효율적이면서도 예측 정확도는 크게 향상시켰다. 이 연구는 실시간 금융 분석 및 고빈도 매매 시스템에서 Mamba 아키텍처가 매우 강력한 도구가 될 수 있음을 시사한다.
