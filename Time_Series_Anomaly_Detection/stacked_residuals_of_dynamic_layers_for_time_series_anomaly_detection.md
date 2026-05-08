# STACKED RESIDUALS OF DYNAMIC LAYERS FOR TIME SERIES ANOMALY DETECTION

Luca Zancato, Alessandro Achille, Giovanni Paolini, Alessandro Chiuso, Stefano Soatto (2022)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 데이터(multivariate time series)에서 비지도 방식의 이상치 탐지(unsupervised anomaly detection)를 수행하는 문제를 다룬다. 산업, 의료, 상업 분야에서 생성되는 대규모 시계열 데이터는 시간의 흐름에 따라 추세(trend)와 계절성(seasonality)과 같은 선형적인 성분뿐만 아니라, 변수 간의 복잡한 비선형적 상관관계를 동시에 가지고 있다.

기존의 딥러닝 기반 모델, 특히 Transformer와 같은 범용적인 신경망 구조는 강력한 성능을 보이지만, 시계열 데이터 특유의 인덕티브 바이어스(inductive bias)가 부족하여 데이터셋별 튜닝이 어렵고 과적합(overfitting)되기 쉬우며, 실패 모드에 대한 해석력이 떨어진다는 문제가 있다. 또한 단순한 선형 모델은 해석력은 높으나 복잡한 비선형 패턴을 포착하지 못하는 한계가 있다. 따라서 본 논문의 목표는 선형 모델의 견고함과 해석력, 그리고 비선형 모델의 유연성을 동시에 갖춘 엔드투엔드 미분 가능한 신경망 아키텍처를 설계하는 것이다.

## ✨ Key Contributions

본 논문이 제안하는 핵심 아이디어는 **STRIC(Stacked Residuals of Dynamic Layers)**라는 계층적 잔차 구조의 아키텍처를 통해 시계열의 성분을 단계적으로 분리하여 학습하는 것이다.

1. **해석 가능한 계층적 아키텍처**: 추세 $\rightarrow$ 계절성 $\rightarrow$ 일반 선형 성분 $\rightarrow$ 비선형 성분 순으로 데이터를 처리하는 Cascade 구조를 설계하여, 각 단계에서 예측 잔차(prediction residual)를 다음 층으로 전달함으로써 모델의 해석력을 높였다.
2. **Fading Memory 정규화**: TCN(Temporal Convolutional Network)이 너무 먼 과거의 불필요한 데이터에 의존하여 과적합되는 것을 방지하기 위해, 과거 데이터의 영향력이 지수적으로 감소하도록 강제하는 새로운 정규화 기법을 도입하였다.
3. **비모수적(Non-parametric) CUMSUM 탐지기**: 예측 잔차의 분포를 사전에 가정하거나 추정할 필요 없이, Pearson divergence의 변분 근사(variational approximation)를 이용하여 우도비(likelihood ratio)를 직접 추정하는 CUMSON 기반의 이상치 탐지 알고리즘을 제안하였다.

## 📎 Related Works

기존의 비지도 이상치 탐지 방식은 크게 밀도 추정(density-estimation), 클러스터링 기반(clustering-based), 재구성 기반(reconstruction-based) 방법으로 나뉜다.

* **기존 접근 방식의 한계**: 최근의 딥러닝 모델들은 Euclidean distance나 재구성 오차(reconstruction error)를 점수화하여 사용하는 방식이 주를 이룬다. 하지만 이러한 방식들은 시계열의 시간적 구조를 충분히 활용하지 못하며, 특히 TCN과 같은 모델을 단독으로 사용할 경우 단순한 시계열에서도 과적합이 발생할 가능성이 크다.
* **STRIC의 차별점**: STRIC은 재구성 기반 방식에 속하지만, 단순한 오차 계산에 그치지 않고 **CUMSUM(Cumulative Sum)** 알고리즘을 결합하여 시간적 구조를 활용한 순차적 확률비 검정(SPRT)을 수행한다. 또한, 선형 동적 층(LDL)을 통해 추세와 계절성을 명시적으로 분리함으로써 일반적인 DNN이 갖지 못하는 시계열 특화 바이어스를 제공한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

STRIC의 예측기는 다음과 같은 4개의 동적 층(Dynamic Layers)이 직렬로 연결된 구조이다. 각 층은 입력에서 자신이 예측할 수 있는 성분을 추출하고, 그 잔차를 다음 층으로 넘긴다.

1. **Linear Dynamic Layers (LDL)**:
    * **Trend Layer**: Hodrick-Prescott 필터를 모사하여 저주파 성분(추세)을 제거한다.
    * **Seasonal Layer**: 단위 원(unit circle) 상의 폴(pole)을 가진 필터를 사용하여 주기적 성분을 제거한다.
    * **General Linear Layer**: 앞선 층들이 잡지 못한 나머지 선형 성분을 모델링한다.
2. **Non-linear Dynamic Layer (TCN)**: 앞선 선형 층들의 잔차를 입력으로 받아, 여러 시계열 간의 글로벌 통계 정보를 통합하여 비선형 성분을 예측한다.

### 2. Fading Memory 정규화 및 학습 목표

TCN의 복잡도를 자동으로 조절하기 위해 베이지안 프레임워크 기반의 정규화를 도입하였다. 예측 계수 $b$가 과거로 갈수록 영향력이 지수적으로 감소한다는 가정을 세우고, 다음과 같은 변분 상한(variational upper bound) 손실 함수를 최소화한다.

$$U_{b,W,\Lambda} = \frac{1}{\eta^2} \|Y_f - \hat{Y}_{b,W}\|^2 + b^T \Lambda^{-1} b + \log \det(F_W \Lambda F_W^T + \eta^2 I)$$

여기서 $\Lambda$는 대각 행렬로, $\Lambda_{j,j} = \kappa \lambda^j$ 형태를 가진다. $\lambda \in (0, 1)$는 모델이 과거를 잊어버리는 속도를 조절하는 파라미터이며, 이를 통해 모델은 데이터에 맞는 최적의 시간 척도(time scale)를 자동으로 학습한다.

### 3. 비모수적 CUMSUM 이상치 탐지

학습된 모델의 예측 잔차 $e_t$를 이용하여 이상치를 탐지한다. CUMSUM 알고리즘은 기본적으로 두 확률 분포의 우도비(likelihood ratio)를 누적하여 사용하지만, 실제 데이터의 분포를 알기 어렵다는 점이 문제다.

이를 해결하기 위해 본 논문은 **Pearson Divergence**를 이용한 변분 근사를 통해 우도비 $\hat{\phi}$를 직접 추정한다. RKHS(Reproducing Kernel Hilbert Space) 내에서 다음과 같은 닫힌 형태(closed form)의 해를 통해 우도비를 계산한다.

$$\hat{\phi}(e) = \frac{n_n}{n_a} K(e, S_{tr}) (K_n^T K_n + \gamma n_n I_{n_n} + n_a)^{-1} K_a^T \mathbf{1}$$

최종적으로 누적 합 $S_t = \sum \log \hat{\phi}_i$를 계산하고, 이 값이 임계값 $\epsilon$을 넘을 때 이상치로 판정한다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**: Credit Card, CICIDS, GECCO, SWAN-SF, SMD, PSM 등 다변량 시계열 데이터셋과 Yahoo, NAB, NYT 데이터셋을 사용하였다.
* **비교 대상**: VAR, OCSVM, IForest, DAGMM, LSTM, OmniAnomaly, Anomaly Transformer 등 SOTA 모델들과 비교하였다.
* **평가 지표**: F1-Score 및 AUC(Area Under the Curve)를 사용하였다.

### 2. 주요 결과

* **정량적 성능**: Table 1 및 Table 5에서 확인되듯, STRIC은 대부분의 벤치마크에서 기존 통계적 방법론과 딥러닝 모델보다 우수한 성능을 보였다. 특히 일반적인 DNN이 과적합으로 인해 고전하는 데이터셋에서도 강건한 성능을 유지하였다.
* **일반화 성능**: Ablation study(Table 2, Table 6) 결과, Fading memory 정규화가 적용된 STRIC이 일반 TCN보다 일반화 갭(Generalization Gap)이 훨씬 작음을 확인하였다. 이는 모델이 불필요한 과거 데이터에 과적합되지 않고 핵심적인 패턴만을 학습했음을 의미한다.
* **정성적 분석 (NYT 데이터셋)**: BERT 임베딩으로 변환된 뉴욕타임즈 기사 시계열에서 9/11 테러, 2004년 인도양 쓰나미와 같은 주요 역사적 사건의 시점을 정확하게 탐지해내는 능력을 보였다.

## 🧠 Insights & Discussion

본 연구의 강점은 **"인덕티브 바이어스의 명시적 주입"**과 **"복잡도의 자동 제어"**에 있다. 단순한 블랙박스 모델인 TCN 앞에 해석 가능한 LDL을 배치함으로써 모델이 학습해야 할 타겟을 '잔차'로 한정시켰고, 이는 학습 효율과 일반화 능력을 동시에 끌어올렸다.

또한, CUMSUM이라는 고전적인 통계 기법을 비모수적 방식으로 확장하여 딥러닝 모델과 결합한 점이 인상적이다. 이는 점 단위의 임계값 판단(pointwise thresholding)보다 시간적 맥락을 훨씬 더 잘 반영하는 탐지 방식이다.

**한계 및 논의사항**:

* 우도비 추정을 위한 윈도우 크기($n_n, n_a$) 설정에 따라 성능 민감도가 높게 나타났다. 이에 대한 최적화된 자동 설정 방법론이 향 uma 연구 과제로 남아있다.
* 비모수적 접근법을 사용했음에도 불구하고, 탐지 임계값 $\epsilon$은 여전히 사용자가 설정해야 하는 하이퍼파라미터라는 점이 한계로 지적될 수 있다.

## 📌 TL;DR

STRIC은 **선형 동적 층(LDL) $\rightarrow$ TCN $\rightarrow$ 비모수적 CUMSUM**으로 이어지는 파이프라인을 통해 다변량 시계열의 이상치를 탐지하는 모델이다. 추세와 계절성을 명시적으로 분리하는 구조와 과적합을 방지하는 Fading Memory 정규화를 통해 기존 DNN 모델보다 훨씬 높은 일반화 성능과 해석력을 확보하였으며, 다양한 벤치마크에서 SOTA 성능을 입증하였다. 이 연구는 딥러닝의 유연성과 통계적 모델의 견고함을 결합하는 효과적인 방법론을 제시하였다.
