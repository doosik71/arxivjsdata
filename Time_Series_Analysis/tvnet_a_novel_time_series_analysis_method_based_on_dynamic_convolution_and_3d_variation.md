# TVNET: A NOVEL TIME SERIES ANALYSIS METHOD BASED ON DYNAMIC CONVOLUTION AND 3D-VARIATION

Chenghan Li, Mingchen Li & Ruisheng Diao (2025)

## 🧩 Problem to Solve

본 논문은 시계열 분석(Time Series Analysis) 분야에서 기존 딥러닝 모델들이 가진 한계를 해결하고자 한다. 현재 시계열 분석은 주로 Transformer 기반 모델과 MLP 기반 모델을 중심으로 발전해 왔으나, 다음과 같은 문제점이 존재한다.

첫째, Transformer 기반 모델은 강력한 성능을 보이지만, Attention 메커니즘의 이차 복잡도(Quadratic Complexity)로 인해 계산 자원 효율성이 낮고 대규모 시계열 데이터 처리 시 확장성 문제가 발생한다. 둘째, MLP 기반 모델은 계산 복잡도는 낮으나 변수 간의 복잡한 의존성(Complex dependencies among variables)을 효과적으로 캡처하는 데 어려움이 있다. 셋째, RNN 기반 모델은 전역적 시간 상관관계(Global temporal correlations)를 모델링하는 데 한계가 있다.

Convolutional Neural Networks(CNNs)는 효율성과 효과성의 균형을 맞출 수 있는 잠재력이 있음에도 불구하고, 그동안 시계열 분석보다는 이미지나 비디오 처리 위주로 연구되어 시계열 분야에서의 활용도가 낮았다. 따라서 본 논문의 목표는 CNN의 표현 능력을 강화하여 시계열 분석의 다양한 작업에서 Transformer나 MLP 모델보다 우수한 효율성과 성능의 균형을 갖춘 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시계열 데이터를 비디오 데이터와 유사한 관점에서 접근하는 것이다. 비디오가 프레임별로 연속적인 시각적 변화를 캡처하듯, 시계열 데이터 또한 연속적인 시간적 변화를 반영한다는 점에 착안하였다. 이를 위해 다음과 같은 핵심 설계를 제안한다.

1. **3D-Embedding 기술**: 1차원 시계열 텐서를 **Intra-patch**(패치 내부), **Inter-patch**(패치 간), **Cross-variable**(변수 간)의 세 가지 차원을 고려한 3차원 텐서로 변환하여 데이터의 표현력을 극대화한다.
2. **Dynamic Convolution 도입**: 정적인 가중치를 사용하는 대신, 입력 데이터의 내용에 따라 가중치가 적응적으로 변하는 Dynamic Convolution을 적용하여 시계열의 복잡한 시간적 패턴과 모드 드리프트(Mode drift) 현상을 효과적으로 캡처한다.
3. **범용적 분석 프레임워크**: 제안한 TVNet을 장단기 예측, 결측치 보간(Imputation), 분류, 이상치 탐지 등 5가지 주요 시계열 분석 작업에 적용하여 높은 일반화 성능과 효율성을 입증하였다.

## 📎 Related Works

### 기존 연구 및 한계

* **전통적 방법**: ARIMA, Holt-Winters 모델 등은 이론적 기반이 탄탄하지만, 복잡한 시간적 패턴을 가진 현대의 대규모 데이터셋을 처리하기에는 부족하다.
* **MLP 및 Transformer 기반 모델**: DLinear, PatchTST, iTransformer 등이 제안되었으나, 앞서 언급한 대로 계산 효율성(Transformer)이나 변수 간 관계 캡처 능력(MLP)에서 각각 한계가 있다.
* **기존 CNN 기반 모델**: TimesNet은 1D 시계열을 2D로 변환하여 처리하고, ModernTCN은 큰 커널을 사용하여 전역 특징을 캡처하려 했다. 하지만 이러한 모델들은 주로 단일 시간 윈도우 내의 특징 분석에 집중하며, 전역-지역-변수 간 상호작용을 통합적으로 고려하지 못했다.

### TVNet의 차별점

TVNet은 단순한 2D 변환을 넘어 3D 관점의 임베딩을 통해 더 풍부한 정보를 추출하며, 특히 비디오 처리 분야에서 검증된 Dynamic Convolution을 시계열에 도입함으로써 정적 필터가 잡지 못하는 동적인 시간적 특성을 모델링한다는 점에서 기존 CNN 기반 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 3D-Embedding

입력 시계열 데이터 $X_{in} \in \mathbb{R}^{L \times C}$ ($L$: 길이, $C$: 차원)를 3차원 텐서로 변환하는 과정은 다음과 같다.

1. **특징 임베딩**: 입력 데이터를 임베딩 차원 $C_m$으로 투영하여 $X_{in} \in \mathbb{R}^{L \times C_m}$을 얻는다.
2. **패치 분할(Patch-Split)**: 커널 크기가 $P$인 Conv1D 레이어를 사용하여 시퀀스를 $N$개의 패치로 나눈다. 결과물은 $X_{emb} \in \mathbb{R}^{N \times P \times C_m}$이 된다.
3. **홀짝 분리(Odd-Even Split)**: 각 패치의 길이 차원에서 홀수 인덱스와 짝수 인덱스를 분리하여 $X_{odd}, X_{even} \in \mathbb{R}^{N \times (P/2) \times C_m}$을 생성하고, 이를 다시 쌓아(stacking) 최종적으로 $X_{emb} \in \mathbb{R}^{N \times 2 \times (P/2) \times C_m}$ 형태의 3D 텐서를 생성한다.

### 2. 3D-Block 및 Dynamic Convolution

TVNet은 적응형 가중치를 통해 시간적 역동성을 캡처한다. $i$번째 패치에 대한 가중치 $W_i$는 모든 패치에 공통으로 적용되는 기본 가중치 $W_b$와 각 패치 고유의 가중치 $\alpha_i$의 곱으로 정의된다.

$$\tilde{x}_i = W_i \cdot x_i = (\alpha_i \cdot W_b) \cdot x_i$$

여기서 시간 가변 가중치 $\alpha_i$는 다음과 같은 생성 함수 $G(X_{emb})$를 통해 결정된다.

$$\alpha_i = G(X_{emb}) = 1 + F(v_{inter}) + F(v_{intra})$$

* **$F(v_{intra})$ (Intra-patch 특징)**: 3D Adaptive Average Pooling을 통해 각 패치의 핵심 특징인 $v_{intra} \in \mathbb{R}^{C_m \times N}$를 추출한 후, Conv1D $\rightarrow$ Batch Normalization $\rightarrow$ ReLU를 거쳐 생성한다.
* **$F(v_{inter})$ (Inter-patch 특징)**: $v_{intra}$에 다시 1D Adaptive Average Pooling을 적용하여 전역 정보인 $v_{inter} \in \mathbb{R}^{C_m \times 1}$를 얻고, 이를 Conv1D $\rightarrow$ ReLU를 통해 처리한다.

### 3. 전체 구조 및 학습 절차

TVNet은 위에서 설명한 3D-Block을 잔차 연결(Residual Connection) 방식으로 쌓아 올린 구조이다.

$$X_{3D}^{i+1} = \text{3D-block}(X_{3D}^i) + X_{3D}^i$$

최종적으로 추출된 3D 표현물 $X_{3D}$는 다시 1차원으로 리셰이프(Reshape)되어, 각 작업(예측, 분류 등)에 맞는 **Task linear-head**를 통해 최종 출력값을 생성한다. 학습 시에는 작업에 따라 MSE(예측, 보간, 이상치 탐지), SMAPE(단기 예측), Cross Entropy(분류) 손실 함수를 사용하며 ADAM 옵티마이저로 최적화한다.

## 📊 Results

### 실험 설정

* **작업**: 장기 예측(Long-term Forecasting), 단기 예측(Short-term Forecasting), 데이터 보간(Imputation), 분류(Classification), 이상치 탐지(Anomaly Detection).
* **데이터셋**: Weather, Traffic, Electricity, Exchange, ILI, ETT(h1, h2, m1, m2), M4, UEA Archive, SMD, SWaT 등 다수의 벤치마크 데이터셋 사용.
* **비교 대상**: Transformer 기반(PatchTST, iTransformer 등), MLP 기반(DLinear, MTS-Mixer 등), CNN 기반(TimesNet, ModernTCN 등).

### 주요 결과

1. **성능**: TVNet은 5가지 모든 작업에서 기존 SOTA 모델들과 경쟁하거나 이를 능가하는 성능을 보였다. 특히 장기 예측에서 대다수의 MLP 및 Transformer 모델보다 낮은 MSE/MAE를 기록하였다.
2. **효율성**: Transformer 기반 모델 대비 학습 속도가 훨씬 빠르고 메모리 사용량이 적다. 예를 들어 ETTm2 데이터셋에서 PatchTST보다 훨씬 적은 메모리 점유율과 빠른 에포크당 학습 시간을 기록하였다.
3. **복잡도**: 시간 복잡도는 $O(LC_m^2)$, 공간 복잡도는 $O(C_m^2)$로, Transformer의 $O(L^2)$ 복잡도보다 효율적이며, 다른 CNN 모델들과 달리 공간 복잡도가 시퀀스 길이 $L$에 의존하지 않는 이점이 있다.
4. **강건성 및 전이 학습**: 노이즈 주입 실험에서 10%까지의 노이즈에 대해 안정적인 성능을 보였으며, 전이 학습(Transfer Learning) 실험에서도 baseline 모델들보다 우수한 일반화 능력을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 근거

본 논문은 Dynamic Convolution이 정적 가중치 모델보다 낮은 오차를 가짐을 이론적으로 증명하였다($E_d < E_f$). 이는 가중치 $\alpha_i$가 각 패치별로 최적화될 수 있는 유연성을 제공하여 타겟 값 $y^*$에 더 가깝게 근사할 수 있기 때문이다. 또한, 3D-Embedding이 1D나 2D 임베딩보다 시계열의 특성을 더 잘 표현한다는 점을 ablation study를 통해 확인하였다.

### 한계 및 논의사항

논문에서는 다양한 작업에서 우수한 성능을 보였으나, 하이퍼파라미터(임베딩 차원 $C_m$, 패치 길이 $P$, 커널 크기 $k$) 설정에 따라 성능 변동이 존재함을 보여주었다. 특히 커널 크기가 너무 작거나 크면 성능이 저하되는 경향이 있어 $3 \times 3$ 커널이 가장 적절함을 확인하였다. 또한, 대규모 사전 학습(Large-scale pre-training)에 대한 논의는 부족하며, 이는 향후 연구 과제로 남겨두었다.

## 📌 TL;DR

TVNet은 1차원 시계열 데이터를 3차원 텐서(Intra-patch, Inter-patch, Cross-variable)로 변환하는 **3D-Embedding**과 내용 적응형 **Dynamic Convolution**을 결합한 새로운 시계열 분석 모델이다. 이 모델은 Transformer의 높은 계산 비용과 MLP의 제한적인 관계 캡처 능력을 동시에 해결하여, **효율성(속도/메모리)과 성능의 최적의 균형**을 달성하였다. 특히 예측, 보간, 분류, 이상치 탐지 등 다양한 작업에서 SOTA급 성능을 보임으로써, CNN이 고급 시계열 분석에서도 매우 강력한 도구가 될 수 있음을 증명하였다.
