# Warping Resilient Scalable Anomaly Detection in Time Series

Abilasha S, Sahely Bhadra, Deepak P and Anish Mathew (2021)

## 🧩 Problem to Solve

본 논문은 시계열 데이터(Time Series Data)에서 발생하는 **Warping** 현상으로 인해 발생하는 이상치 탐지(Anomaly Detection)의 어려움을 해결하고자 한다. Warping이란 시계열 데이터의 전반적인 패턴은 유지되면서 시간 축을 따라 국소적으로 확장(expansion) 또는 압축(compression)이 발생하는 현상을 의미한다.

현존하는 많은 이상치 탐지 모델들은 이러한 Warping 변형을 의미론적 변형이 아닌 비정상적인 패턴으로 인식하여 잘못된 이상치로 판정(False Positive)하는 경향이 있다. 특히 Euclidean distance 기반의 방법론이나 일반적인 CNN, RNN 기반 모델들은 신호와 그 Warping 변형본을 서로 다른 신호로 처리하기 때문에 이 문제가 심각하다. 한편, Dynamic Time Warping (DTW)과 같은 기법은 이러한 Warping에 강건하지만, 계산 복잡도가 시퀀스 길이의 제곱 수준($O(n^2)$)으로 매우 높아 대규모 데이터셋에 적용하기에는 확장성(Scalability)이 떨어진다는 한계가 있다.

따라서 본 연구의 목표는 Warping 변형에 강건하면서도 계산 효율성이 높아 대규모 데이터셋에 적용 가능한 비지도 학습 기반의 시계열 이상치 탐지 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **자기지도 학습(Self-supervised Learning)**과 **데이터 증강(Data Augmentation)**을 통해 Warping에 강건한 임베딩(Embedding) 공간을 학습하는 것이다.

중심적인 설계 아이디어는 시계열 데이터에 인위적인 Warping 변형을 가하는 특수 연산자(Warping Operators)를 정의하고, 원본 데이터와 변형된 데이터가 임베딩 공간에서 서로 가깝게 위치하도록 유도하는 **Twin Auto-encoder** 구조를 사용하는 것이다. 이를 통해 모델이 Warping으로 인한 차이는 무시하고, 실제 의미론적인 이상치만을 식별할 수 있는 'Blind spot'을 학습하게 한다.

## 📎 Related Works

논문에서는 이상치 탐지를 크게 두 가지 범주로 나누어 설명한다.

1. **Point Anomalies (점 이상치):**
    * **Density-based:** LOF 등이 있으며, 국소적 밀도를 측정하지만 시간적 의존성을 모델링하지 못한다.
    * **Prediction-error based:** NumentaHTM 등이 있으며, 예측값과 실제값의 차이를 이용한다.
    * **Deep-learning:** RNNAE 등이 있으며, 주로 재구성 오차(Reconstruction error)를 기반으로 한다. 그러나 일반화 성능을 위해 극단적인 값을 모델링하지 못하는 경향이 있어 실제 이상치를 놓치거나 Warping을 이상치로 오인하는 문제가 있다.
    * **Discord Discovery:** MERLIN 등이 있으며, 일반적인 패턴과 크게 다른 하위 시퀀스를 찾지만 파라미터 설정에 매우 민감하다.

2. **Sequence Anomalies (시퀀스 이상치):**
    * **Deep-learning:** BeatGAN 등이 있으며, GAN과 Auto-encoder를 결합해 비정상 리듬을 탐지한다.
    * **Similarity-based:** DTW가 가장 대표적이며 Warping에 매우 강건하지만, 앞서 언급한 대로 높은 계산 복잡도로 인해 대규모 데이터셋에서는 사용이 불가능하다.

본 연구는 기존 딥러닝 모델들이 Warping을 처리하지 못하는 점과 DTW의 낮은 확장성을 동시에 해결함으로써 차별점을 갖는다.

## 🛠️ Methodology

본 논문에서 제안하는 **WaRTEm-AD**는 크게 두 단계(Representation Learning $\rightarrow$ Anomaly Scoring)로 구성된다.

### 1. WaRTEm (Warp Resilient Time-series Embedding)

시계열 $T$를 저차원 벡터 $V=f(T)$로 변환하는 임베딩 학습 단계이다.

#### (1) Warping Operators

데이터 증강을 위해 두 가지 종류의 Warping 연산자를 정의한다. 특정 윈도우 영역을 대상으로 수행된다.

* **Copy Warping:** 특정 지점의 값을 복사하여 구간을 확장한다.
  * **LCW (Left Copy Warp):** 윈도우의 왼쪽을 줄이고 오른쪽 끝점 값을 반복하여 확장한다.
  * **RCW (Right Copy Warp):** 윈도우의 오른쪽을 줄이고 왼쪽 끝점 값을 반복하여 확장한다.
* **Interpolation Warping:** 복사 대신 선형 보간(Linear Interpolation)을 통해 값을 채운다.
  * **LIW (Left Interpolation Warp) / RIW (Right Interpolation Warp):** 시작점과 끝점 사이를 선형적으로 연결하여 구간을 변형한다.

#### (2) Twin Auto-Encoder Architecture

두 개의 Auto-encoder(AE)가 병렬로 배치된 구조이다. 가중치는 공유되지 않지만 손실 함수를 통해 학습이 연결된다.

* **입력:** 한쪽 AE에는 $[T, RW(T)]$ (원본과 우측 변형본)를, 다른 쪽에는 $[LW(T), T]$ (좌측 변형본과 원본)를 입력한다.
* **목표:** 두 AE의 디코더 모두 원본 시퀀스 $T$를 재구성하도록 학습한다.

#### (3) 손실 함수 (Loss Function)

최적화 목표는 다음과 같은 세 가지 손실 함수의 합으로 정의된다.
$$\min_{f_1, f_2} \underbrace{\|f_1^{-1}(f_1(T)) - T\|_2^2 + \|f_2^{-1}(f_2(\hat{T})) - T\|_2^2}_{L_1 + L_2} + \lambda \underbrace{\|f_1(T) - f_2(\hat{T})\|_2^2}_{L_3}$$

* $L_1$: 표준 Auto-encoder 손실로, 원본 $T$의 재구성 성능을 높인다.
* $L_2$: 변형된 $\hat{T}$로부터 원본 $T$를 복원하는 Denoising AE 형태의 손실이다.
* $L_3$: **Representation Coupling Loss**로, 원본 $T$와 변형본 $\hat{T}$의 임베딩 간 유클리드 거리를 최소화하여 두 표현이 가깝게 위치하도록 강제한다.

### 2. Anomaly Scoring (이상치 점수 산출)

학습된 임베딩 $V$를 사용하여 이상치 점수 $AS(\cdot)$를 계산한다. 기본적으로 임베딩 공간에서의 **K-NN 거리**(K번째 최근접 이웃과의 거리)를 이상치 점수로 사용한다.

* **Sequence Anomaly:** 각 시퀀스의 임베딩 $V_i$에 대해 K-NN 거리를 직접 계산한다.
* **Point Anomaly:**
    1. 슬라이딩 윈도우를 사용하여 긴 시계열을 여러 개의 짧은 시퀀스 $D(T, m)$로 나눈다.
    2. 각 윈도우에 대한 임베딩 $V_i$를 생성하고 K-NN 거리로 점수를 매긴다.
    3. 특정 데이터 포인트 $s$가 포함된 윈도우들의 점수를 평균 내어 최종 점수를 산출한다.
    $$AS(s \in T) = \frac{1}{\|SW_{T,m,p}(s)\|} \sum_{T_i \in SW_{T,m,p}(s)} AS(V_i)$$
    여기서 $SW_{T,m,p}(s)$는 $s$가 중앙 $p$개 포인트 내에 포함되는 윈도우들의 집합이다.

## 📊 Results

### 1. 실험 설정

* **데이터셋:** Point Anomaly는 NAB(Numenta Anomaly Benchmark)의 5개 데이터셋을, Sequence/Sub-sequence Anomaly는 UCR 저장소 및 Marotta Valve 등의 데이터셋을 사용하였다.
* **평가 지표:** PR-AUC, ROC-AUC, Reciprocal Rank (RR)를 사용하여 임계값 설정에 관계없이 성능을 측정하였다.
* **비교 대상:** LOF, NumentaHTM, MERLIN, RNNAE, DTW, BeatGAN 등.

### 2. 주요 결과

* **Point Anomaly:** WaRTEm-AD가 대부분의 데이터셋에서 베이스라인 모델들을 압도하는 성능을 보였다. 특히 RNNAE는 Z-score가 높은 극단적 값 탐지에는 강했으나, Warping이 포함된 일반적인 이상치 탐지에서는 WaRTEm-AD가 더 우수했다.
* **Sequence/Sub-sequence Anomaly:** PR-AUC 기준으로 14개 데이터셋 중 6개에서 1위, 4개에서 2위를 기록하였다. 특히 Warping에 특화된 DTW와 유사하거나 더 높은 성능을 보이면서도 계산 속도는 훨씬 빨랐다.
* **확장성 (Scalability):** 훈련 및 추론 시간 분석 결과, WaRTEm-AD는 데이터 길이와 수에 따라 선형적으로 증가하는 효율적인 시간 복잡도를 보였다. 이는 quadratic 복잡도를 가진 DTW에 비해 압도적인 이점이다.

## 🧠 Insights & Discussion

본 연구는 시계열 데이터의 특성인 Warping을 '제거해야 할 노이즈' 혹은 '학습해야 할 불변성(Invariance)'으로 정의하고, 이를 자기지도 학습 구조에 녹여내어 성공적으로 해결하였다.

**강점 및 분석:**

* **강건한 임베딩:** Twin AE 구조와 Coupling Loss를 통해 Warping 변형에 관계없이 동일한 의미를 가진 시퀀스는 임베딩 공간에서 가깝게 배치되도록 설계되었다.
* **효율성:** DTW 수준의 Warping 강건성을 확보하면서도 딥러닝의 추론 속도를 가져감으로써 실시간 대규모 데이터 적용 가능성을 증명하였다.
* **연산자 분석:** 실험을 통해 단순 복사(Copy) 방식보다 선형 보간(Interpolation) 방식의 Warping 연산자가 이상치 탐지 성능을 약간 더 향상시킨다는 점을 발견하였다.

**한계 및 향후 과제:**

* 본 논문은 단변량(Univariate) 시계열에 집중하였으며, 다변량(Multivariate) 시계열로의 확장 가능성을 향후 과제로 제시하였다.
* Warping 외의 다른 정당한 왜곡(legitimate distortions)에 대해서도 강건한 모델을 만드는 것이 필요하다.

## 📌 TL;DR

WaRTEm-AD는 시계열 데이터의 국소적 시간 변형인 **Warping에 강건한 임베딩을 학습**하여 이상치를 탐지하는 비지도 학습 프레임워크이다. **특수 Warping 연산자**를 이용한 데이터 증강과 **Twin Auto-encoder**의 Coupling Loss를 통해, Warping은 무시하고 실제 이상 패턴만 식별하는 능력을 갖추었다. 결과적으로 DTW의 강건함과 딥러닝의 확장성을 동시에 확보하여, 점/시퀀스/하위 시퀀스 이상치 탐지 작업에서 기존 모델들보다 우수한 성능과 효율성을 입증하였다.
