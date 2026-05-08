# Graph Anomaly Detection in Time Series: A Survey

Thi Kieu Khanh Ho, Ali Karami, Narges Armanfard (2024)

## 🧩 Problem to Solve

시계열 이상치 탐지(Time-Series Anomaly Detection, TSAD)는 전자상거래, 사이버 보안, 헬스케어 모니터링 등 다양한 분야에서 매우 중요한 과제이다. 하지만 시계열 데이터에는 변수 내부의 시간적 흐름에 따른 의존성인 Intra-variable dependency와 여러 변수 간의 상호작용에 의한 의존성인 Inter-variable dependency가 동시에 존재하며, 이를 모두 고려하는 것은 매우 도전적인 작업이다.

기존의 많은 TSAD 알고리즘들은 데이터를 단변량(Univariate)으로 취급하거나, 다변량 데이터라 하더라도 각 변수를 개별적인 모달리티로 분석하여 Intra-variable dependency만을 고려하는 경향이 있었다. 그러나 실제 물리적 시스템(예: 뇌의 센서 배치, 서버 네트워크, 비디오 프레임)에서는 변수 간의 공간적 상호작용이 이상치 판단에 결정적인 역할을 한다. 따라서 본 논문은 이러한 Inter-variable dependency를 효과적으로 모델링하기 위해 그래프 표현형을 도입한 G-TSAD(Graph-based TSAD)의 최신 동향을 체계적으로 분석하고 정리하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 시계열 데이터의 이상치 탐지를 위해 그래프 구조를 활용하는 G-TSAD 분야의 첫 번째 종합 서베이를 제공한다는 점이다. 구체적인 기여 사항은 다음과 같다.

첫째, Intra-variable dependency와 Inter-variable dependency라는 개념을 명확히 정의하고, 이를 캡처하기 위한 그래프 구성 방식(정적 그래프 및 동적 그래프)을 제시하여 G-TSAD 연구를 위한 새로운 관점을 제공한다.

둘째, G-TSAD 방법론을 손실 함수의 특성에 따라 Autoencoder(AE) 기반, GAN 기반, 예측(Predictive) 기반, 그리고 자기지도학습(Self-supervised) 기반의 네 가지 범주로 분류한 통합 분류 체계(Unified Taxonomy)를 제안한다.

셋째, 각 방법론의 기술적 세부 사항, 강점, 한계점을 분석하고, 실제 적용된 데이터셋과 평가 지표를 정리하여 연구자들이 효율적으로 최신 기술을 파악할 수 있도록 한다.

넷째, 이론적 기반 부족, 그래프 증강 전략의 부재, 다중 이상치 유형 탐지의 어려움 등 현재 G- uma TSAD 분야가 직면한 기술적 과제와 향후 연구 방향을 제시한다.

## 📎 Related Works

논문에서는 기존의 서베이 연구들을 세 가지 방향으로 구분하여 G-TSAD 서베이의 필요성을 역설한다.

먼저 그래프 기반 접근 방식(GA) 서베이들은 분류, 클러스터링, 예측 등 일반적인 그래프 구조 학습 기술을 다루지만, 이상치 탐지(AD)에 특화되어 있지 않다. 둘째, 그래프 이상치 탐지(GAD) 서베이들은 정적인 그래프 데이터(예: 정적 사회 관계망, 이미지)에서의 이상 객체 탐지에 집중하며, 시계열 데이터 특유의 시간적 의존성(Intra-variable dependency)을 다루지 않는다. 셋째, 기존의 TSAD 서베이들은 주로 시계열 신호 자체의 분석에 집중하며, 비디오나 동적 사회 네트워크와 같은 그래프 구조 데이터의 특성을 간과하고 Inter-variable dependency의 중요성을 충분히 다루지 않는다.

결과적으로 본 논문은 기존의 GA, GAD, TSAD 서베이들이 놓치고 있었던 '시계열 데이터의 시간적-공간적 의존성을 동시에 캡처하는 그래프 기반 방법론'이라는 공백을 메우고자 한다.

## 🛠️ Methodology

### 그래프 표현 및 정의

본 논문에서는 시계열 데이터를 그래프 집합 $\mathcal{G}$로 정의한다.

$$\mathcal{G} = \{G_j, \text{Sim}\{G_j, G_{j'}\}\}_{j, j' \in \{1, \dots, N\}, j \neq j'}$$

여기서 $G_j = \{M_j, A_j\}$는 $j$번째 관측치에 대한 그래프이다. $M_j$는 $K \times m$ 크기의 노드 특징 행렬(Node-feature matrix)이며, $A_j$는 $K \times K \times m'$ 크기의 에지 특징 행렬(Edge-feature matrix)로 변수 간의 관계를 나타낸다. $\text{Sim}\{\cdot, \cdot\}$ 함수는 서로 다른 관측치 간의 관계를 정의하여 Intra-variable dependency를 캡처한다.

그래프 구성 방식은 학습 가능한 파라미터의 존재 여부에 따라 두 가지로 나뉜다.

- **Dynamic Graphs**: $G_j$나 $\text{Sim}$ 함수에 학습 가능한 파라미터가 포함되어 데이터로부터 관계를 동적으로 학습하는 방식이다.
- **Static Graphs**: 사전 지식(예: 센서 간의 유클리드 거리)을 바탕으로 고정된 노드 특징과 에지를 사용하는 방식이다.

### G-TSAD 방법론의 분류 체계

#### 1. AE-based Methods

입력 그래프를 저차원으로 압축했다가 다시 복원하는 과정에서 발생하는 복원 오차(Reconstruction Error)를 이용한다.
$$\theta^*, \phi^* = \arg\min_{\theta, \phi} \mathcal{L}_{rec}(D_\phi(E_\theta(G)), G)$$
정상 데이터의 패턴을 학습하여 정상 샘플은 낮게, 이상 샘플은 높게 복원 오차가 나오도록 설계한다. VAE(Variational AE)를 통해 잠재 공간을 정규화하거나, Normalizing Flow(NF)를 통해 복잡한 확률 분포를 모델링하여 성능을 높이는 방향으로 발전하고 있다.

#### 2. GAN-based Methods

생성자(Generator)와 판별자(Discriminator)의 적대적 학습을 이용한다.
$$\theta^*, \phi^* = \arg\min_{\theta, \phi} \mathcal{L}_{gen}(D_\phi(E_\theta(G, z)), G)$$
$$\psi^* = \arg\min_{\psi} \mathcal{L}_{disc}(D_\psi(G', G))$$
생성자는 실제 데이터와 유사한 가짜 그래프를 생성하고, 판별자는 이를 구분한다. 복원 오차뿐만 아니라 판별자의 판단 결과까지 이상치 점수로 활용함으로써 더 정교한 탐지가 가능하다.

#### 3. Predictive-based Methods

과거와 현재의 데이터를 바탕으로 미래의 그래프 상태 $\bar{G}_{j+1}$을 예측하고, 실제 값 $G_{j+1}$과의 차이를 계산한다.
$$\theta^*, \phi^* = \arg\min_{\theta, \phi} \mathcal{L}_{pred}(D_\phi(E_\theta(G)), G_{j+1})$$
미래 예측 실패를 이상치로 간주하며, 특히 GNN과 Attention 메커니즘을 결합하여 변수 간의 동적인 관계를 예측하는 방식(예: GDN)이 많이 사용된다.

#### 4. Self-supervised Methods

레이블이 없는 데이터에서 Pretext task를 통해 유용한 표현을 학습한다. 특히 Contrastive Learning(CL)이 주를 이루며, 유사한 샘플(Positive pair)은 가깝게, 서로 다른 샘플(Negative pair)은 멀게 배치하도록 학습한다.
$$\theta^*, \phi^* = \arg\min_{\theta, \phi} \mathcal{L}_{con}(D_\phi(E_\theta(G^{(1)}), E_\theta(G^{(2)})))$$
이후 학습된 표현 공간에서의 거리나 유사도를 기반으로 이상치를 탐지한다.

## 📊 Results

### 데이터셋 및 적용 도메인

논문은 G-TSAD가 적용되는 세 가지 주요 도메인을 제시한다.

- **Time-series Signals**: Yahoo S5, NASA, SWaT, WADI 등의 벤치마크 데이터셋과 PhysioNet 같은 생체 신호 데이터셋이 사용된다.
- **Social Networks**: UCI Messages, Email-DNC 등 사용자 간 상호작용이 시간에 따라 변하는 동적 그래프 데이터셋이 사용된다.
- **Videos**: UCF-Crime, Xd-Violence 등 비디오 프레임을 시계열 그래프로 변환하여 이상 행동을 탐지하는 데이터셋이 활용된다.

### 평가 지표 및 비판적 분석

일반적으로 F1-score, Precision, Recall, AUC, APR 등이 사용된다. 특히 저자들은 시계열 탐지 분야에서 흔히 사용되는 Point Adjustment(PA) 기법의 위험성을 지적한다. PA는 이상 구간 중 단 한 점만 맞추어도 전체 구간을 맞춘 것으로 간주하는 방식인데, 이는 모델의 성능을 과하게 부풀리는 경향이 있다. 이를 해결하기 위해 일정 비율 이상의 정밀도를 요구하는 $\text{PA}\%K$ 프로토콜이나, 임계값에 독립적인 AUC-ROC, APR 지표를 사용할 것을 권장한다.

## 🧠 Insights & Discussion

### 강점 및 분석

G-TSAD는 기존 TSAD가 해결하지 못한 Inter-variable dependency를 그래프 구조로 명시화함으로써, 단일 변수만으로는 보이지 않는 '관계적 이상치(Relational Anomaly)'를 찾아낼 수 있다는 강력한 강점을 가진다. 또한, 단순한 신호 분석을 넘어 사회 네트워크의 이상 사용자 탐지나 비디오 내 객체 간 상호작용 분석으로 확장성이 매우 높다.

### 한계 및 미해결 과제

1. **이론적 근거 부족**: 대부분의 방법론이 실험적 결과에 의존하며, 학습된 표현이 왜 이상치를 구분하는지에 대한 이론적 증명이나 설명 가능성(Explainability)이 부족하다.
2. **장기 의존성 문제**: 많은 모델이 단기적인 윈도우 내에서는 효과적이지만, 시계열의 특성인 계절성(Seasonality)이나 장기 트렌드(Long-term trend)를 캡처하는 능력은 여전히 부족하다.
3. **그래프 증강의 어려움**: 자기지도학습에서 중요한 데이터 증강(Augmentation)이 이미지와 달리 그래프 구조에서는 매우 까다롭다. 단순한 랜덤 샘플링 외에 유의미한 그래프 증강 전략이 필요하다.
4. **현실 데이터의 오염**: 훈련 데이터가 완전히 깨끗하다는 가정(Unsupervised assumption) 하에 설계된 모델이 많아, 실제 현장에서 발생하는 데이터 오염(Contaminated data)이나 Open-set 상황에서의 강건성이 떨어진다.

## 📌 TL;DR

본 논문은 시계열 데이터의 시간적-공간적 의존성을 동시에 해결하기 위해 그래프 구조를 도입한 **G-TSAD(Graph-based Time-Series Anomaly Detection)** 분야를 집대성한 첫 번째 서베이 보고서이다. 방법론을 **AE, GAN, 예측, 자기지도학습**의 네 가지 체계로 분류하여 상세히 분석하였으며, 특히 단순한 성능 수치보다 평가 프로토콜의 엄밀함(PA 문제)과 실제 도메인 확장성(비디오, 사회망)에 주목한다. 이 연구는 향후 G-TSAD가 단순한 딥러닝 구조 적용을 넘어, 이론적 정립과 강건한 그래프 증강 전략, 그리고 다중 이상치 유형을 동시에 탐지하는 방향으로 나아가야 함을 시사한다.
