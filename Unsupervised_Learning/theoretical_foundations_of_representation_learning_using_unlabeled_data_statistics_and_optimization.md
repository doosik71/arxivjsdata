# Theoretical Foundations of Representation Learning using Unlabeled Data: Statistics and Optimization

Pascal Mattia Esser, Maximilian Fleissner, Debarghya Ghoshdastidar (2025)

## 🧩 Problem to Solve

본 논문은 라벨이 없는 데이터(unlabeled data)를 이용한 표현 학습(Representation Learning)의 이론적 기반을 구축하는 것을 목표로 한다. 최근 시각적 파운데이션 모델(visual foundation models)과 대규모 언어 모델(LLMs)은 자기지도 학습(self-supervision) 및 마스크드 오토인코더(masked autoencoders) 등을 통해 엄청난 성공을 거두었으나, 이러한 모델들이 구체적으로 어떤 표현을 학습하며 왜 다양한 예측 작업에서 뛰어난 성능을 보이는지에 대한 이론적 설명은 여전히 부족한 상태이다.

기존의 다변량 통계학 기반의 차원 축소 기법(예: PCA)은 선형 투영에 국한되어 현대의 딥러닝 모델이 보여주는 비선형적 표현 능력을 설명하기 어렵다. 반면, 최신 딥러닝 모델은 최적화 과정의 복잡성과 손실 함수(loss landscape)의 특성으로 인해 전통적인 통계적 일반화 이론으로는 그 성능을 완전히 설명할 수 없다. 따라서 본 논문은 통계학과 최적화 이론의 도구들을 결합하여, 비지도 표현 학습의 최적화 역학(optimization dynamics)과 통계적 특성을 규명하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 비지도 표현 학습의 두 가지 주요 패러다임인 **재구성 기반 방식(Reconstruction-based)**과 **공동 임베딩 방식(Joint Embedding)**에 대한 이론적 분석을 제공한 것이다.

1. **재구성 기반 방식의 분석**: 선형 Denoising Autoencoder(DAE)에서 병목 층(bottleneck layer)의 차원 $k$가 일반화 오차에 미치는 영향을 분석하고, Skip Connection이 분산(variance)의 급격한 증가를 어떻게 완화하는지 수학적으로 증명하였다. 또한, 클러스터별 표현을 학습하는 Tensorized Autoencoder(TAE)의 최적해를 규명하였다.
2. **공동 임베딩 방식의 NTK 분석**: Barlow Twins와 같은 비대조 학습(non-contrastive learning) 모델이 매우 넓은 신경망(infinite width) 환경에서 Neural Tangent Kernel(NTK) regime에 머무르며, 최종적으로는 커널 모델의 해로 수렴함을 보였다.
3. **최적 데이터 증강(Data Augmentation)의 이론적 도출**: 특정 타겟 표현 $f^*$를 학습하기 위해 필요한 최적의 데이터 증강 함수 $T$가 존재함을 RKHS(Reproducing Kernel Hilbert Space) 상에서 증명하고 이를 수치적으로 구현하는 방법을 제시하였다.
4. **다운스트림 예측 성능의 보장**: CURL 프레임워크를 통해 SimCLR와 같은 자기지도 학습 손실 함수와 실제 지도 학습의 Cross-Entropy 손실 간의 관계를 정립하여, 사전 학습된 표현의 일반화 성능에 대한 PAC-Bayesian 경계(bound)를 제시하였다.

## 📎 Related Works

논문은 기존의 비지도 학습을 크게 두 시대로 구분하여 설명한다.

- **고전적 접근 방식**: PCA, ICA, Factor Analysis 등 다변량 통계학 기반의 방법론들이다. 이들은 주로 데이터 압축과 차원 축소에 집중하였으나, 학습된 표현이 데이터의 선형 투영이라는 한계가 있다. 이를 극복하기 위해 t-SNE, Kernel methods, RBM 등이 제안되었으나 현대의 딥러닝 스케일에서는 한계가 있다.
- **현대적 딥러닝 접근 방식**: 딥 뉴럴 네트워크를 이용한 비선형 표현 학습이다. 특히 최근의 파운데이션 모델들은 단순한 압축을 넘어, 다운스트림 태스크에 유용한 '유용한 표현(useful representation)'을 학습하는 것에 집중한다.

본 연구는 기존 연구들이 단순히 경험적인 스케일링 법칙(scaling laws)에 의존하거나, 아주 제한적인 설정(예: 선형 모델, 단순한 손실 함수)에서만 분석을 진행했던 한계를 지적하며, 최적화 역학과 통계적 일반화를 통합적으로 다룸으로써 차별성을 갖는다.

## 🛠️ Methodology

본 논문은 표현 학습의 두 가지 핵심 원리를 중심으로 방법론을 전개한다.

### 1. 재구성 기반 방식 (Reconstruction-based)

데이터를 잠재 공간으로 인코딩한 후 다시 원래의 공간으로 복원하는 과정을 통해 특징을 추출한다.

- **Linear Denoising Autoencoder (DAE)**:
  입력 데이터 $X$에 노이즈 $A$가 추가되었을 때, 다음의 손실 함수를 최소화한다.
  $$\min_{W_1, W_2} \|X - W_2 W_1(X + A)\|^2 + \lambda \|W_2 W_1\|^2$$
  여기서 $W_1 \in \mathbb{R}^{k \times d}$는 인코더, $W_2 \in \mathbb{R}^{d \times k}$는 디코더이며 $k$는 병목 차원이다. 본 논문은 $\lambda \to 0$인 경우의 전역 최적해 $W^*$가 다음과 같이 수렴함을 증명한다.
  $$W^* = P_{[k]}(X)(X + A)^\dagger$$
  (여기서 $P_{[k]}(X)$는 $X$의 rank-$k$ 근사치이며 $\dagger$는 무어-펜로즈 유사역행렬이다.)

- **Tensorized Autoencoder (TAE)**:
  $m$개의 AE를 병렬로 배치하여 각 클러스터별로 서로 다른 표현을 학습하게 한다. 각 데이터 $x_i$가 $j$번째 AE에 할당될 확률 $S_{j,i}$를 함께 학습하며, 이는 사실상 $k$-means 클러스터링 비용이 정규화된 재구성 오차 최소화 문제로 정의된다.

### 2. 공동 임베딩 방식 (Joint Embedding)

데이터 증강(augmentation)을 통해 생성된 두 뷰(view) $x$와 $x^+$가 잠재 공간에서 서로 가까워지도록 학습한다.

- **Barlow Twins Loss**:
  두 표현 $f(x)$와 $f(x^+)$ 사이의 교차 상관 행렬(cross-correlation matrix) $C$를 단위 행렬 $I$에 가깝게 만든다.
  $$L^{BT}(f) = \sum_{j=1}^k (1 - C_{jj})^2 + \lambda \sum_{j \neq l} C_{jl}^2$$
  이는 각 차원이 서로 다른 특징을 학습하게 하여 차원 붕괴(collapse)를 방지한다.

- **NTK (Neural Tangent Kernel) 분석**:
  신경망의 너비 $M$이 무한대로 갈 때, 학습 과정 중 NTK $K_t$가 일정하게 유지됨을 보였다. 이를 통해 딥러닝 모델의 학습 역학을 커널 회귀(kernel regression) 문제로 치환하여 분석할 수 있다.

- **최적 증강 도출**:
  타겟 표현 $f^*$가 주어졌을 때, 이를 학습시키기 위한 최적의 증강 맵 $T_H$를 RKHS 상에서 정의한다. Spectral Contrastive 모델의 경우, $T_H$는 RKHS 내에서의 rank-$k$ 투영으로 정의된다.

## 📊 Results

### 1. DAE의 일반화 오차와 병목 차원

실험 결과, DAE의 테스트 오차는 병목 차원 $k$와 오버파라미터화 비율 $c = d/N$에 의해 결정된다.

- $k$가 너무 작으면 편향(bias)이 증가하고, $k$가 너무 크면 분산(variance)이 증가하는 전형적인 **Bias-Variance Trade-off**가 나타난다.
- 특히 $c \approx 1$ 부근에서 분산이 급격히 증가하는 현상이 발견되었으나, **Skip Connection**을 추가했을 때 이러한 변동성이 크게 완화됨을 정량적으로 확인하였다.

### 2. NTK 수렴성 검증

MNIST 데이터셋을 이용한 수치 실험에서 네트워크 너비 $M$이 증가할수록:

- 초기 NTK와 수렴 시 NTK의 차이가 감소한다.
- 신경망이 학습한 표현이 이론적으로 도출된 커널 모델의 최적해와 매우 유사해짐(Procrustes distance 감소)을 확인하였다.

### 3. 다운스트림 예측 성능 보장

CIFAR-10 데이터셋으로 사전 학습된 모델에 대해 SimCLR 손실과 다운스트림 Cross-Entropy 손실 간의 관계를 분석하였다.

- 제안된 PAC-Bayesian 경계가 기존의 uniform convergence 경계보다 훨씬 타이트(tight)하며, 실제 테스트 오차의 경향성을 더 정확하게 반영함을 보였다.
- 특히 온도 하이퍼파라미터 $\tau$가 낮을 때, 제안된 방법론의 오차 경계가 유효함(non-vacuous)을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 비지도 표현 학습이 단순한 데이터 압축이 아니라, **데이터 증강이라는 도메인 지식을 통해 유용한 특징 공간을 투영하는 과정**임을 이론적으로 규명하였다.

**강점 및 통찰**:

- **증강의 역할**: self-supervised learning의 성공 요인이 단순히 모델 구조에 있는 것이 아니라, 어떤 증강을 사용하느냐가 곧 어떤 표현(spectral projection)을 학습하느냐를 결정한다는 점을 명확히 하였다.
- **구조적 최적화**: Skip Connection이 단순한 성능 향상 도구가 아니라, 고차원 데이터에서의 분산 폭발을 막는 통계적 정규화 역할을 수행함을 보였다.

**한계 및 논의사항**:

- **커널 모델의 한계**: NTK 분석은 무한 너비 가정을 전제로 하므로, 실제 유한한 너비의 모델이나 Transformer와 같은 복잡한 어텐션 구조에서 발생하는 '특징 학습(feature learning)' 효과를 완전히 포착하지 못한다.
- **증강 함수의 구현**: RKHS 상에서 최적 증강 $T_H$를 찾았더라도, 이를 실제 입력 공간(pixel space)의 함수 $T$로 역투영하는 과정(kernel pre-image problem)에서 근사 오차가 발생할 수 있다.

## 📌 TL;DR

본 논문은 현대의 비지도 표현 학습(SSL)을 통계학과 최적화 관점에서 분석한 종합 보고서이다. **DAE의 병목 구조와 Skip Connection의 효과**, **Barlow Twins의 NTK 수렴성**, 그리고 **타겟 표현 학습을 위한 최적 데이터 증강의 존재성**을 수학적으로 증명하였다. 이 연구는 파운데이션 모델의 '창발적 특성(emergent property)'을 이해하기 위한 이론적 토대를 제공하며, 향후 비지도 학습의 하한 오차(lower bound) 분석 및 어텐션 메커니즘의 통계적 이점 연구로 확장될 가능성이 높다.
