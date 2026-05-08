# StackVAE-G: An efficient and interpretable model for time series anomaly detection

Wenkai Li, Wenbo Hu, Ting Chen, Ning Chen, Cheng Feng (2022)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series) 데이터에서 이상치 탐지(Anomaly Detection)를 수행할 때 발생하는 효율성과 해석 가능성의 문제를 해결하고자 한다.

일반적으로 Autoencoder(AE) 기반 모델들은 비지도 학습 방식으로 복잡한 데이터를 잘 피팅하여 우수한 성능을 보이지만, 기존 접근 방식들은 다음과 같은 한계를 가진다. 첫째, **Step-wise 모델**(예: LSTM-VAE)은 모든 타임스텝에서 재구성을 수행하므로 파라미터 수가 많아 훈련 데이터의 노이즈에 과적합(Overfitting)되기 쉽다. 둘째, **Block-wise 모델**(예: USAD)은 윈도우 단위로 병렬 재구성을 수행하여 노이즈에 강건하지만, 입력 윈도우 크기가 커질수록 모델 파라미터가 급격히 증가하여 장기 의존성(Long-term dependency)을 캡처하기 어렵다. 셋째, 다변량 데이터의 핵심인 채널 간 상호관계(Interrelation structure)를 명시적으로 모델링하지 못해, 탐지 결과에 대한 해석 가능성이 부족하다.

따라서 본 연구의 목표는 적은 계산 비용과 메모리 사용량으로도 높은 탐지 성능을 유지하며, 채널 간의 관계를 명시적으로 학습하여 이상치의 원인을 진단할 수 있는 해석 가능한 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Weight-sharing 기반의 Stacking Block-wise 구조**와 **Graph Learning 모듈**을 결합하는 것이다.

1. **Stacking Block-wise Framework**: 모든 채널에 대해 동일한 가중치를 공유하는 단일 채널 재구성 유닛을 쌓아 올리는 방식을 제안한다. 이를 통해 모델 크기를 획기적으로 줄여 과적합을 방지하고, 효율적인 Block-wise 재구성을 통해 더 긴 윈도우 크기를 사용할 수 있게 한다.
2. **Graph Learning Module**: 채널 간의 상호관계를 나타내는 희소 인접 행렬(Sparse Adjacency Matrix)을 학습한다. 이 행렬을 통해 각 채널의 잠재 표현(Latent representation)을 인접 채널의 정보와 융합함으로써, 단순한 개별 채널 분석을 넘어 구조적 정보를 활용한 재구성을 가능하게 한다.
3. **Interpretability**: 학습된 인접 행렬을 통해 어떤 채널들이 서로 밀접하게 연관되어 있는지 확인할 수 있으며, 이를 통해 발생한 이상치가 단일 센서의 오류(Clerical error)인지 시스템 전체의 구조적 오류(Systematic error)인지 진단할 수 있는 근거를 제공한다.

## 📎 Related Works

기존의 시계열 이상치 탐지 연구는 크게 예측 기반(Prediction-based) 방식과 재구성 기반(Reconstruction-based) 방식으로 나뉜다.

- **Step-wise AE 모델**: LSTM-VAE, OmniAnomaly 등은 RNN 셀을 사용하여 매 타임스텝마다 데이터를 재구성한다. 표현력은 높으나 과적합 위험이 크다.
- **Block-wise AE 모델**: USAD 등은 슬라이딩 윈도우 단위로 데이터를 처리한다. 로컬 노이즈에 강건하지만, 윈도우 크기가 증가함에 따라 파라미터 수가 $\text{channel}^2 \times \text{window}^2$에 비례하여 폭발적으로 증가하는 문제가 있다.
- **구조 학습(Structure Learning)**: 기존의 GNN 기반 시계열 모델들은 채널 간 관계를 암시적으로 학습하거나, 외부에서 주어진 그래프 구조에 의존하는 경우가 많아 해석력이 부족하거나 일반화 성능이 떨어진다는 한계가 있었다.

본 논문은 가중치 공유(Weight-sharing)를 통해 Block-wise 모델의 파라미터 폭발 문제를 해결하고, 자기지도 학습(Self-supervised learning) 기반의 그래프 학습 손실 함수를 도입하여 명시적이고 해석 가능한 구조를 학습한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

StackVAE-G는 **Stacking Block-wise VAE(StackVAE)**와 **Graph Neural Network(GNN) 모듈**의 두 가지 핵심 구성 요소로 이루어져 있다.

### 1. Stacking Block-wise VAE

입력 슬라이스 $\mathbf{X} \in \mathbb{R}^{n \times w}$ ($n$은 채널 수, $w$는 윈도우 크기)가 들어오면, 모델은 모든 채널 $\mathbf{x}_n$에 대해 동일한 가중치를 가진 선형 레이어 $f$를 공유하여 처리한다.

**인코딩 과정**:

1. 각 채널 $\mathbf{x}_n$을 공유 레이어 $f$를 통해 중간 잠재 특징 $\mathbf{v}_{n,1}$로 변환한다.
2. 학습된 인접 행렬 $\tilde{A}$를 사용하여 채널 간 정보를 융합한 $\mathbf{v}_{n,2}$를 계산한다:
   $$\mathbf{V}_2 = (1 - \gamma)\mathbf{V}_1 + \gamma \tilde{A} \mathbf{V}_1$$
   여기서 $\gamma$는 융합 비율을 조절하는 하이퍼파라미터이다.
3. 융합된 특징을 통해 각 채널의 잠재 분포 $q(\mathbf{z}_n | \mathbf{x}_n) = \mathcal{N}(\mu_n, \sigma_n^2 \mathbf{I})$를 추론하고 잠재 코드 $\mathbf{z}_n$을 샘플링한다.

**디코딩 과정**:
샘플링된 $\mathbf{z}_n$을 다시 공유 레이어를 통해 재구성 출력 $\hat{\mathbf{x}}_n$으로 복원한다. 최종 출력은 $\hat{\mathbf{X}} = [\hat{\mathbf{x}}_1, \dots, \hat{\mathbf{x}}_n]^\top$이 된다.

### 2. Graph Learning Module

채널 간의 안정적인 상호관계를 학습하기 위해 다음과 같은 절차를 거친다.

1. **인접 행렬 생성**: 노드 임베딩 $\mathbf{U}$를 학습하고, $\text{tanh}$ 활성화 함수와 $\text{ReLU}$를 거쳐 초기 인접 행렬 $\mathbf{A}$를 생성한다.
2. **희소화(Sparsification)**: 각 행에서 상위 $k$개의 값만 남기고 나머지는 0으로 설정하여 $\tilde{A}$를 구축한다.
3. **Graph Learning Loss**: 채널 간의 선형 의존성을 캡처하기 위해 자기지도 회귀 손실 함수를 사용한다:
   $$\mathcal{L}_{Graph} = \|\mathbf{X} - \tilde{A}\mathbf{X}\|_2^2 = \sum_{n=1}^{n} \|\mathbf{x}_n - \sum_{j=1}^{n} \tilde{A}_{nj} \mathbf{x}_j\|_2^2$$
   이는 각 채널이 인접 채널들의 선형 조합으로 자신을 재구성할 수 있도록 강제하여, $\tilde{A}$가 물리적으로 의미 있는 관계를 학습하게 한다.

### 3. 전체 손실 함수 및 학습 절차

모델은 VAE의 ELBO(Evidence Lower Bound) 손실과 그래프 학습 손실을 결합하여 최적화한다:
$$\mathcal{L}_{total} = \mathcal{L}_{VAE} + \lambda \mathcal{L}_{Graph}$$
여기서 $\mathcal{L}_{VAE} = -\mathbb{E}_{q(\mathbf{Z}|\mathbf{X})}[\log p(\mathbf{X}|\mathbf{Z})] + \text{KL}(q(\mathbf{Z}|\mathbf{X}) \| p(\mathbf{Z}))$이다.

### 4. 이상치 탐지 절차

학습된 모델을 통해 재구성된 $\hat{\mathbf{X}}$와 원본 $\mathbf{X}$ 사이의 제곱 오차 합을 타임스텝 $t$의 이상치 점수 $S_t$로 정의한다:
$$S_t = \sum_{n=1}^{n} (\hat{x}_{n,t} - x_{n,t})^2$$
$S_t$가 설정된 임계값 $c$보다 크면 해당 시점을 이상치로 판정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: SMD (서버 머신), SMAP (위성), MSL (화성 탐사선)
- **지표**: Precision, Recall, F1-score
- **비교 대상**: OC-SVM, Isolation Forest, DeepSVDD, LSTM-VAE, OmniAnomaly, USAD, Anomaly Transformer 등 11종의 최신 모델

### 정량적 결과

- **탐지 성능**: StackVAE-G는 대부분의 데이터셋에서 최신 모델인 Anomaly Transformer와 대등하거나 더 우수한 F1-score를 기록하였다. 특히 SMD와 MSL 데이터셋에서 매우 높은 성능을 보였다.
- **효율성**:
  - **파라미터 수**: OmniAnomaly나 Anomaly Transformer에 비해 모델 크기가 압도적으로 작다. 예를 들어, 윈도우 크기가 100일 때 USAD는 파라미터가 폭증하지만, StackVAE-G는 가중치 공유 덕분에 매우 작은 크기를 유지한다.
  - **학습 속도**: 에포크당 학습 시간이 타 모델 대비 현저히 낮아, 자원이 제한된 엣지 디바이스(Edge devices)에 배포하기 유리함을 입증하였다.

### 정성적 결과 및 해석 가능성

- **그래프 구조 분석**: Lasso 회귀 결과와 비교했을 때, StackVAE-G가 학습한 인접 행렬 $\tilde{A}$가 실제 채널 간의 유사성(Clustering)을 더 정확하게 캡처함을 확인하였다.
- **장애 진단**:
  - **Sensor Clerical Error**: 특정 채널만 이상치가 발생하고 인접 채널들은 정상인 경우.
  - **Systematic Error**: 해당 채널과 그 인접 채널들이 동시에 이상 징후를 보이는 경우.
    이처럼 $\tilde{A}$를 통해 이상치의 성격을 구분할 수 있음을 보여주었다.

## 🧠 Insights & Discussion

### 강점

본 연구는 다변량 시계열 탐지에서 성능, 효율성, 해석 가능성이라는 세 마리 토끼를 동시에 잡았다. 특히 Weight-sharing을 통해 Block-wise 모델의 고질적인 문제인 파라미터 폭발 문제를 해결하면서도, GNN을 도입하여 블랙박스 형태의 AE 모델에 해석 가능성을 부여한 점이 매우 뛰어나다.

### 한계 및 가정

논문에서 명시했듯이, 제안된 그래프 학습 손실 함수 $\mathcal{L}_{Graph}$는 **선형적인 상호관계(Linear interrelation)**만을 학습한다. 따라서 $y = x^2$와 같은 비선형적 관계는 캡처하지 못한다는 한계가 있다. 하지만 실험 결과, 선형 관계만으로도 충분히 우수한 성능과 해석력을 제공함을 확인하였다.

### 비판적 해석

모델의 성능이 매우 뛰어나지만, 임계값 $c$를 상수로 설정하여 사용하는 방식은 데이터의 동적인 특성에 따라 성능 편차가 발생할 수 있다. 향후 연구에서 동적 임계값(Dynamic thresholding) 기법을 도입한다면 더욱 견고한 모델이 될 것으로 보인다.

## 📌 TL;DR

**StackVAE-G**는 가중치 공유(Weight-sharing) 기반의 Stacking Block-wise VAE 구조를 통해 **모델의 경량화와 계산 효율성**을 극대화하고, 자기지도 학습 기반의 GNN 모듈을 통해 **채널 간 상호관계를 명시적으로 학습**하는 모델이다. 이를 통해 최신 SOTA 모델들과 대등한 탐지 성능을 내면서도 훨씬 적은 자원을 사용하며, 학습된 인접 행렬을 통해 **이상치의 원인을 진단할 수 있는 해석 가능성**을 제공한다. 이 연구는 특히 자원이 제한된 IoT 환경의 엣지 디바이스 내 실시간 이상 탐지 시스템에 적용될 가능성이 매우 높다.
