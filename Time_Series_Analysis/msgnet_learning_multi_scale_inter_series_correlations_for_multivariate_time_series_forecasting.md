# MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting

Wanlin Cai, Yuxuan Liang, Xianggen Liu, Jianshuai Feng, Yuankai Wu

## 🧩 Problem to Solve

다변량 시계열 예측은 다양한 분야에서 지속적인 도전 과제를 안고 있습니다. 기존 딥러닝 모델들은 시계열 내(intra-series) 및 시계열 간(inter-series) 상관관계를 포착하는 데 초점을 맞추었지만, 여러 시계열 간에 **다양한 시간 스케일(time scales)에 걸쳐 변화하는 시계열 간 상관관계**를 이해하고 모델링하는 부분에서는 상당한 연구 격차가 존재합니다. 이러한 변화하는 다중 스케일 상관관계가 예측 정확도와 모델의 일반화 능력에 미치는 영향이 제대로 다뤄지지 않았습니다.

## ✨ Key Contributions

- 시계열 간 상관관계가 다양한 시간 스케일과 복잡하게 연관되어 있다는 핵심적인 관찰을 기반으로, 이러한 다중 스케일 시계열 간 상관관계를 효율적으로 발견하고 포착하는 새로운 구조인 **MSGNet**을 제안했습니다.
- 시계열 내 상관관계와 시계열 간 상관관계를 동시에 포착하기 위해 **멀티 헤드 어텐션(multi-head attention)**과 **적응형 그래프 컨볼루션(adaptive graph convolution)** 모듈을 조합하여 통합했습니다.
- 실제 데이터셋에 대한 광범위한 실험을 통해 MSGNet이 시계열 예측 작업에서 기존 딥러닝 모델들을 꾸준히 능가하며, **더 나은 일반화 능력**을 보여줌을 경험적으로 입증했습니다.

## 📎 Related Works

- **고전적 시계열 예측**: VAR, Prophet 등은 시계열 내 변화가 미리 정의된 패턴을 따른다고 가정하지만, 실제 데이터의 복잡한 변화를 다루기 어렵습니다.
- **딥러닝 기반 시계열 예측**: MLP(N-BEATS), TCN, RNN(DeepAR), Transformer 기반 모델(Informer, Autoformer, TimesNet) 등이 시계열 내 동적 특징을 포착하는 데 활용되었습니다. TimesNet은 주기성 분해를 통해 다중 스케일 정보를 사용하지만, 서로 다른 주기성 스케일에서 발생하는 다양한 시계열 간 상관관계를 고려하지 않습니다.
- **GNN을 이용한 시계열 간 상관관계 학습**: GNN(Graph Convolutional Networks, Mixhop)은 트래픽 예측 등에서 시계열 간 복잡한 상호 의존성을 모델링하는 데 성공적입니다. 하지만 대부분의 GNN은 미리 정의된 고정된 그래프 구조를 가정하거나, 학습 가능한 그래프 구조를 사용하더라도 이를 다른 시간 스케일과 연결하지 않아 복잡하고 진화하는 시계열 간 상관관계를 충분히 포착하지 못합니다.

## 🛠️ Methodology

MSGNet은 여러 ScaleGraph 블록으로 구성되며, 각 블록은 다음 4단계로 동작합니다.

1. **입력 임베딩 및 잔차 연결**: 입력 $X_{t-L:t} \in \mathbb{R}^{N \times L}$를 $d_{\text{model}}$ 차원으로 임베딩하여 $X_{\text{emb}} \in \mathbb{R}^{d_{\text{model}} \times L}$를 생성하고 잔차 연결을 사용합니다.
   $$X_{\text{emb}} = \alpha \text{Conv1D}(\hat{X}_{t-L:t}) + \text{PE} + \sum_{p=1}^P \text{SE}_p$$
2. **스케일 식별(Scale Identification)**: 고속 푸리에 변환(FFT)을 사용하여 데이터에서 가장 지배적인 $k$개의 주기성(시간 스케일 $s_i = L/f_i$)을 감지합니다. 이 스케일에 따라 입력 시계열을 3D 텐서 $X_i \in \mathbb{R}^{d_{\text{model}} \times s_i \times f_i}$로 재구성합니다.
   $$F = \text{Avg}(\text{Amp}(\text{FFT}(X_{\text{emb}})))$$
   $$f_1, \cdots, f_k = \text{argTopk}_{f^* \in \{1, \cdots, \frac{L}{2}\}}(F), s_i = \frac{L}{f_i}$$
   $$X_i = \text{Reshape}_{s_i, f_i}(\text{Padding}(X_{\text{in}})), \quad i \in \{1, \cdots, k\}$$
3. **다중 스케일 적응형 그래프 컨볼루션(Multi-scale Adaptive Graph Convolution)**: 각 시간 스케일 $i$에 대해 고유한 적응형 인접 행렬 $A_i \in \mathbb{R}^{N \times N}$를 학습합니다.
   $$A_i = \text{SoftMax}(\text{ReLu}(E_i^1 (E_i^2)^T))$$
   이후 Mixhop 그래프 컨볼루션(Mixhop GCN)을 적용하여 시계열 간 상관관계를 포착합니다.
   $$H_i^{\text{out}} = \sigma \left( \Vert_{j \in P} (A_i)^j H_i \right)$$
4. **멀티 헤드 어텐션 및 스케일 통합(Multi-head Attention and Scale Aggregation)**: 각 스케일 텐서 $\hat{X}_i$에 멀티 헤드 어텐션(MHA)을 적용하여 시계열 내 상관관계를 학습합니다.
   $$\hat{X}_i^{\text{out}} = \text{MHA}_s(\hat{X}_i)$$
   마지막으로, FFT로 계산된 각 스케일의 진폭($\hat{a}_i$)을 SoftMax를 통해 가중치로 사용하여 서로 다른 $k$개의 스케일 텐서를 통합합니다. 이는 Mixture of Experts (MoE) 전략을 따릅니다.
   $$\hat{X}_{\text{out}} = \sum_{i=1}^k \hat{a}_i \hat{X}_i^{\text{out}}$$
5. **출력 레이어**: 선형 변환을 통해 통합된 특징 $\hat{X}_{\text{out}}$를 최종 예측값 $\hat{X}_{t:t+T} \in \mathbb{R}^{N \times T}$으로 변환합니다.
   $$\hat{X}_{t:t+T} = W_s \hat{X}_{\text{out}} W_t + b$$

## 📊 Results

- **예측 성능**: 8가지 실제 데이터셋(Flight, Weather, ETT, Exchange-Rate, Electricity)에 대한 실험에서 MSGNet은 평균 MSE에서 5개 데이터셋에서 최고 성능을, 2개 데이터셋에서 두 번째 최고 성능을 달성했습니다. 특히 Flight 데이터셋에서는 TimesNet(기존 SOTA) 대비 MSE를 21.5%, MAE를 13.7% 감소시켰습니다.
- **일반화 능력**: COVID-19 팬데믹으로 인한 OOD(Out-of-Distribution) 샘플이 포함된 Flight 데이터셋에서 MSGNet은 가장 적은 성능 감소를 보이며 강력한 일반화 능력을 입증했습니다. 이는 모델이 다중 시계열 간 상관관계를 포착하여 외부 교란에 대한 저항력을 갖추었기 때문입니다.
- **학습된 시계열 간 상관관계 시각화**: MSGNet은 24시간, 6시간, 4시간과 같은 다른 시간 스케일에 대해 상이한 인접 행렬을 학습하며, 이는 공항 간의 물리적 근접성과 같은 실제 시나리오와 일치하는 설명 가능한 다중 스케일 상관관계를 보여줍니다.
- **효율성**: TimesNet보다 높은 운영 효율성을 달성하며, 훈련 시간을 크게 단축하면서 유사한 시간 비용으로 다양한 예측 길이에 걸쳐 성능을 유지했습니다.

## 🧠 Insights & Discussion

- **다중 스케일 시계열 간 상관관계의 중요성**: 기존 모델들이 간과했던 "다양한 시간 스케일에서 변화하는 시계열 간 상관관계"를 MSGNet이 효과적으로 포착함으로써 예측 성능이 크게 향상됨을 보여주었습니다. 이는 시계열 분석에서 주기성을 기반으로 한 스케일 식별과 스케일별 그래프 구조 학습의 중요성을 강조합니다.
- **모델의 설명 가능성**: 학습된 적응형 인접 행렬을 통해 다양한 시간 스케일에서 시계열 간의 상호작용이 어떻게 달라지는지 시각적으로 설명할 수 있습니다. 이는 모델의 예측 결과를 해석하는 데 도움을 줍니다.
- **강력한 일반화 능력**: MSGNet은 OOD(Out-of-Distribution) 샘플에 대해서도 견고한 성능을 보여주는데, 이는 여러 시계열 간 상관관계 중 일부는 외부 교란에도 불구하고 유효하게 유지될 수 있다는 가설을 뒷받침합니다.
- **Long-term 예측에서의 효율성**: 스케일 변환을 통해 긴 시퀀스를 짧은 주기로 변환하여 Multi-head Attention이 효과적으로 작동하도록 함으로써, Transformer 기반 모델의 장기 시계열 상관관계 포착 한계를 극복했습니다.

## 📌 TL;DR

MSGNet은 다변량 시계열 예측에서 **다양한 시간 스케일별로 변화하는 시계열 간 상관관계**를 포착하는 딥러닝 모델입니다. **FFT 기반 스케일 식별**, **적응형 Mixhop 그래프 컨볼루션**, **멀티 헤드 어텐션**을 통해 시계열 내/간 상관관계를 통합적으로 학습합니다. 실제 데이터셋에서 SOTA 모델을 능가하며, **OOD 샘플에 대한 뛰어난 일반화 능력**과 **설명 가능한 다중 스케일 상관관계** 학습 능력을 입증했습니다.
