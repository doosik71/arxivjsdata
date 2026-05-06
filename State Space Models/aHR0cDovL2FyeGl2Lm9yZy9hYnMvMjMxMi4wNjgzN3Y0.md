# Spectral State Space Models

Naman Agarwal, Daniel Suo, Xinyi Chen, Elad Hazan (2024)

## 🧩 Problem to Solve

본 논문은 긴 범위의 의존성(long-range dependencies)을 가진 시퀀스 예측 작업에서 발생하는 효율적인 모델링 문제를 해결하고자 한다. 전통적인 Recurrent Neural Networks(RNN)는 기울기 소실 및 폭발 문제와 순차적 연산으로 인한 확장성 한계가 있으며, Transformer 모델은 컨텍스트 길이에 따라 메모리와 계산량이 이차적으로 증가하는 비용 문제가 존재한다.

최근 주목받는 State Space Models(SSMs)는 선형 동적 시스템(Linear Dynamical Systems, LDS)을 사용하여 이 문제를 해결하려 하지만, 시스템 행렬 $A$의 고유값 크기가 1에 가까워지는 'marginally stable'한 시스템의 경우, 학습의 불안정성이 급격히 증가하는 문제가 발생한다. 기존의 SSM이나 Linear Recurrent Units(LRU)는 이를 해결하기 위해 안정적인 지수 파라미터화(stable exponential parameterization), 링 초기화(ring initialization), $\gamma$-정규화와 같은 휴리스틱한 기법들에 의존해 왔으나, 이러한 방법들은 이론적 보장이 부족하며 설정에 매우 민감하다는 한계가 있다.

따라서 본 연구의 목표는 이론적으로 보장된 견고함을 바탕으로, 특별한 초기화나 정규화 없이도 매우 긴 메모리를 효율적으로 처리할 수 있는 새로운 SSM 구조인 Spectral State Space Model을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LDS의 시스템 행렬을 직접 학습하는 대신, **Spectral Filtering** 알고리즘을 통해 입력을 특수한 스펙트럼 기저(spectral basis)로 투영하여 표현하는 것이다.

주요 기여 사항은 다음과 같다.

1. **이론적 기반의 고정 커널 사용**: 데이터로부터 커널을 학습하는 기존 방식과 달리, 이론적으로 도출된 고정된 합성곱 필터(fixed convolutional filters)를 사용하여 파라미터 수를 줄이면서도 성능을 높였다.
2. **강건성 증명**: 제안된 모델의 성능이 시스템의 spectral gap($\delta$)이나 문제의 차원에 의존하지 않음을 이론적으로 증명하여, 매우 긴 컨텍스트에서도 학습 안정성을 보장한다.
3. **STU 및 AR-STU 아키텍처 제안**: Spectral Filtering을 신경망 레이어 형태로 구현한 Spectral Transform Unit(STU)과, 여기에 자기회귀(auto-regressive) 성분을 추가하여 표현력을 높인 AR-STU를 제안하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들과의 차별점을 강조한다.

- **State Space Models (SSMs)**: S4, S5, H3 등의 모델은 HiPPO 이론을 바탕으로 긴 시퀀스를 처리한다. 특히 LRU는 단순한 선형 재귀 구조를 사용하지만, 긴 컨텍스트 학습을 위해서는 매우 정교한 파라미터화와 정규화가 필수적이다. 반면, 본 논문의 모델은 이러한 인위적인 장치 없이도 안정적인 학습이 가능하다.
- **Spectral Filtering**: [HSZ17]에서 제안된 기법으로, LDS의 입력을 저차원 부분 공간으로 투영하여 효율적으로 학습하는 방법이다. 본 논문은 이 이론을 확장하여 대칭 행렬(symmetric matrices)과 음수 고유값을 처리할 수 있는 신경망 레이어 구조로 발전시켰다.
- **Convolutional Models**: 일부 모델들은 합성곱 커널을 직접 학습하거나 특정 구조(예: wavelet transform)를 부여한다. 하지만 본 논문의 커널은 파라미터가 없는 고정형이며, 이론적으로 LDS보다 더 높은 표현력을 가짐이 증명되었다.

## 🛠️ Methodology

### 1. Spectral Transform Unit (STU)

STU는 입력 시퀀스를 고정된 필터 $\phi_k$를 통해 특성 공간으로 투영한 뒤, 이를 선형 결합하여 출력을 생성하는 구조이다.

**필터의 생성**:
먼저 다음과 같은 Hankel 행렬 $Z \in \mathbb{R}^{L \times L}$를 정의한다.
$$Z[i,j] \triangleq \frac{2}{(i+j)^3 - (i+j)}$$
이 행렬 $Z$의 상위 $K$개 고유벡터 $\{\phi_k\}_{k=1}^K$가 고정 필터로 사용된다.

**특성 추출 (Featurization)**:
입력 시퀀스 $u$에 대해 양(+)과 음(-)의 성분을 모두 캡처하기 위해 두 가지 투영 벡터 $U_{t,k}^+$와 $U_{t,k}^-$를 계산한다.
$$U_{t,k}^+ = \sum_{i=0}^{t-1} u_{t-i} \cdot \phi_k(i)$$
$$U_{t,k}^- = \sum_{i=0}^{t-1} u_{t-i} \cdot (-1)^i \cdot \phi_k(i)$$
이 연산은 FFT(Fast Fourier Transform)를 통해 $O(L \log L)$의 시간 복잡도로 효율적으로 계산될 수 있다.

**출력 생성**:
최종 출력 $\hat{y}_t$는 자기회귀 성분(Auto-regressive Component)과 스펙트럼 성분(Spectral Component)의 합으로 구성된다.
$$\hat{y}_t = \hat{y}_{t-2} + \sum_{i=1}^3 M_{u_i} u_{t+1-i} + \sum_{k=1}^K M_{\phi_k}^+ \sigma_k^{1/4} U_{t-2,k}^+ + \sum_{k=1}^K M_{\phi_k}^- \sigma_k^{1/4} U_{t-2,k}^-$$
여기서 $M_{u_i}, M_{\phi_k}^+, M_{\phi_k}^-$는 학습 가능한 행렬이며, $\sigma_k$는 행렬 $Z$의 고유값이다.

### 2. AR-STU (Auto-Regressive STU)

단순 STU에서 더 나아가, 이전 출력값 $\hat{y}$들에 대한 의존성을 추가하여 표현력을 극대화한 모델이다.
$$\hat{y}_t = \sum_{i=1}^{k_y} M_{y_i} \hat{y}_{t-i} + \text{Auto-regressive Component} + \text{Spectral Component}$$
이 구조는 모든 선형 동적 시스템(LDS)을 완벽하게 예측할 수 있다는 이론적 근거(Theorem 5.1)를 바탕으로 설계되었다.

### 3. 이론적 보장

Theorem 3.1에 따르면, $K = O(\log L)$개의 필터만으로도 임의의 marginally stable한 대칭 LDS를 매우 작은 오차 $\epsilon$ 내로 근사할 수 있음이 증명되었다. 이는 메모리 요구량이 시퀀스 길이에 대해 로그 스케일로 증가함을 의미하며, 매우 효율적인 표현 방식임을 시사한다.

## 📊 Results

### 1. 합성 LDS 학습 실험

매우 높은 안정성 상수($\approx 10^4$)를 가진 합성 시스템을 통해 STU와 LRU의 학습 효율을 비교하였다.

- **결과**: STU는 볼록한(convex) 파라미터화 덕분에 매우 빠르게 수렴하였으며, 광범위한 학습률(Learning Rate) 범위에서 안정적인 궤적을 보였다.
- **LRU 비교**: LRU는 정교한 초기화와 정규화 기법을 모두 적용했음에도 불구하고, STU보다 약 8배 더 많은 샘플이 필요했으며 초기 학습 단계에서 정체(plateau) 현상이 관찰되었다.

### 2. Long Range Arena (LRA) 벤치마크

텍스트, 이미지 등 다양한 모달리티와 $1\text{K}$에서 $16\text{K}$에 이르는 컨텍스트 길이를 가진 LRA 벤치마크에서 성능을 평가하였다.

- **PathX (16K context)**: 가장 어려운 과제인 PathX에서 STU 모델은 특별한 정규화 없이도 높은 정확도를 달성하였다. 이는 LRU가 동일한 성능을 내기 위해 필수적으로 요구했던 모든 'tricks' 없이 가능했다는 점에서 매우 유의미하다.
- **정량적 결과**: AR-STU 모델은 특히 이미지 관련 작업(CIFAR, Pathfinder, PathX)에서 $k_y=32$ 설정을 통해 S4 및 LRU 대비 우수한 성능을 보였다. 전체 6개 작업 중 4개에서 베이스라인을 상회하거나 대등한 성능을 기록하였다.

## 🧠 Insights & Discussion

본 논문은 SSM의 학습 불안정성 문제를 '시스템 행렬의 직접 학습'이라는 비볼록 최적화 문제에서 '고정된 스펙트럼 기저로의 투영'이라는 문제로 전환함으로써 해결하였다.

**강점**:

- **이론적 견고함**: spectral gap에 관계없이 일정한 성능을 보장하므로, 모델 설계자가 하이퍼파라미터 튜닝이나 복잡한 정규화에 쏟는 시간을 획기적으로 줄일 수 있다.
- **효율성**: 고정 커널을 사용함으로써 파라미터 수를 줄였음에도 불구하고, 이론적으로 LDS 이상의 표현력을 가짐을 보였다.

**한계 및 논의**:

- **대칭 행렬 가정**: 본 모델의 이론적 보장은 시스템 행렬 $A$가 대칭(symmetric)인 경우에 국한된다. 하지만 저자들은 최신 SSM(예: Mamba)들이 실수 대각 행렬을 사용하는 경향이 있으며, 실제 작업에서 대칭성과 일반 행렬 간의 성능 차이가 크지 않을 것이라고 주장한다.
- **비대칭 확장**: 비대칭 행렬로의 확장은 가능하지만 계산 효율성이 떨어지는 문제가 있으며, 이는 향후 연구 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 Spectral Filtering 이론을 딥러닝 아키텍처에 접목하여, **특별한 초기화나 정규화 없이도 매우 긴 시퀀스를 안정적으로 처리하는 Spectral SSM(STU/AR-STU)**을 제안한다. 고정된 스펙트럼 필터를 사용하여 계산 효율성을 높이고 학습의 볼록성을 확보함으로써, 기존 SSM들이 겪었던 학습 불안정성 문제를 해결하고 LRA 벤치마크에서 경쟁력 있는 성능을 입증하였다. 이 연구는 향후 매우 긴 컨텍스트를 처리해야 하는 언어 모델이나 시계열 예측 모델의 안정적인 설계 방향을 제시한다.
