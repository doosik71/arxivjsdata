# Mamba Adaptive Anomaly Transformer with association discrepancy for time series

Abdellah Zakaria Sellam, Ilyes Benaissa, Abdelmalik Taleb-Ahmed, Luigi Patrono, Cosimo Distante (2025)

## 🧩 Problem to Solve

본 논문은 시계열 데이터에서 비지도 학습(Unsupervised Learning) 기반의 이상치 탐지(Anomaly Detection) 문제를 해결하고자 한다. 시계열 이상치 탐지는 산업 모니터링, 환경 센싱, 인프라 신뢰성 확보를 위해 매우 중요하지만, 복잡한 시간적 패턴 속에서 정상 데이터와 이상치를 정확히 구분하는 것은 여전히 어려운 과제이다.

기존의 대표적인 방법론인 Anomaly Transformer는 Prior-association과 Series-association 간의 Association Discrepancy(연관성 불일치)를 활용하여 성능을 높였고, DCdetector는 Dual-attention Contrastive Learning을 도입하였다. 그러나 이러한 기존 방식들은 다음과 같은 한계점을 가지고 있다. 첫째, 짧은 컨텍스트 윈도우(Context Window)에 민감하여 장기 의존성 포착에 한계가 있다. 둘째, Transformer의 Self-attention 메커니즘으로 인한 계산 효율성 저하 문제가 존재한다. 셋째, 실제 환경의 노이즈가 심하거나 비정상적(Non-stationary)인 조건에서 성능이 저하되는 경향이 있다.

따라서 본 논문의 목표는 Association Discrepancy 모델링을 정교화하고 재구성(Reconstruction) 품질을 향상시켜, 노이즈에 강건하면서도 계산 효율적인 새로운 구조의 MAAT(Mamba Adaptive Anomaly Transformer)를 제안하는 것이다.

## ✨ Key Contributions

MAAT의 핵심 아이디어는 효율적인 국소 패턴 포착과 강력한 전역 의존성 모델링을 적응적으로 결합하는 것이다. 이를 위해 다음과 같은 설계를 도입하였다.

1. **Sparse Attention의 도입**: 표준 Self-attention의 연산 복잡도를 줄이기 위해 블록 단위의 Sparse Attention을 적용하였다. 이를 통해 연산 중복을 줄이면서도 이상치 판별에 중요한 장거리 의존성을 효율적으로 캡처한다.
2. **Mamba-Selective State Space Model (Mamba-SSM) 통합**: 재구성 모듈에 Mamba-SSM을 통합하여 시계열 데이터의 복잡한 장기 의존성을 선형 시간 복잡도로 학습할 수 있게 하였다.
3. **Gated Attention 기반의 적응적 융합**: Sparse Attention의 출력과 Mamba-SSM의 출력을 단순히 합치는 것이 아니라, Gated Attention 메커니즘을 통해 두 경로의 특징을 동적으로 융합한다. 이를 통해 국소적 세부 사항과 전역적 문맥 간의 균형을 맞추어 이상치 위치 정밀도와 탐지 성능을 동시에 향상시켰다.

## 📎 Related Works

시계열 이상치 탐지는 ARIMA나 Gaussian Processes와 같은 통계적 방법론에서 시작되었다. 이러한 방법들은 선형적인 가정을 바탕으로 예측값과의 편차를 통해 이상치를 찾지만, 비선형적이고 고차원적인 실제 데이터에서는 한계가 명확하였다. 이후 SVM, Random Forest, Isolation Forest 등의 머신러닝 기법이 도입되었으나, 여전히 수동적인 특성 공학(Feature Engineering)에 의존하며 복잡한 시간적 의존성을 학습하는 데 어려움이 있었다.

딥러닝의 발전으로 RNN, LSTM, Autoencoder(VAE 등)가 등장하며 비지도 학습 기반의 재구성 오차(Reconstruction Error) 방식이 주류가 되었다. 하지만 재구성 기반 방식은 이상치가 정상 데이터와 유사한 패턴을 가질 경우 False Positive(오탐)가 발생하는 문제가 있다. 이를 해결하기 위해 Anomaly Transformer는 Association Discrepancy라는 개념을 도입하여 정상과 이상치의 연관성 패턴 차이를 이용해 구분력을 높였다. 또한 DCdetector는 Contrastive Learning을 통해 글로벌 및 로컬 의존성을 분리하여 학습하였다.

MAAT는 이러한 기존 연구들의 흐름을 계승하면서도, Transformer의 2차 복잡도 문제를 Sparse Attention으로 해결하고, RNN/Transformer의 장기 의존성 한계를 Mamba-SSM으로 보완함으로써 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

MAAT는 크게 세 가지 주요 모듈로 구성된다: **Anomaly Sparse Attention**, **Reconstruction Block**, 그리고 **MAAT Block**이다. 전체 파이프라인은 입력 시계열 데이터로부터 연관성 불일치를 계산하고, 이를 적응적으로 재구성된 신호와 결합하여 최종 이상치 점수를 산출하는 구조이다.

### 주요 구성 요소 및 동작 원리

#### 1. Anomaly Sparse Attention

기존 Anomaly Transformer의 dense self-attention을 대체한다.

- **Prior-Association Branch**: 학습 가능한 Gaussian Kernel을 사용하여 예상되는 의존성을 인코딩한다.
- **Series-Association Branch**: 입력 데이터에서 관찰된 의존성을 캡처하며, 이때 연산 효율을 위해 $\text{block size}$ 내의 국소 윈도우 $\Omega_i$에 대해서만 Attention을 계산하는 Sparse Softmax를 적용한다.

$$\Omega_i = \{j \mid |j-i| \le \text{block size}/2\}$$

이 구조를 통해 연산 중복을 줄이면서 핵심적인 의존성만을 효율적으로 추출한다.

#### 2. MAAT Block 및 Mamba-SSM

재구성 모듈의 핵심인 MAAT Block은 Mamba-SSM과 Skip Connection을 결합한 형태이다.

- **Mamba-SSM**: 선택적 상태 공간 모델(Selective State Space Model)을 통해 매우 긴 시퀀스의 의존성을 효율적으로 모델링하여 $x_{\text{mamba}}$를 생성한다.
- **Skip Connection**: 원본 입력 $x_{\text{orig}}$와 Mamba의 출력을 더해 세부 정보를 보존하며, LayerNorm을 통해 정규화한다.

$$x_{\text{skip}} = \text{LayerNorm}(x_{\text{mamba}} + x_{\text{orig}})$$

#### 3. Gated Attention Fusion

Sparse Attention을 통해 처리된 메인 경로의 출력 $x$와 Mamba 경로의 출력 $x_{\text{skip}}$을 적응적으로 융합한다. 학습 가능한 게이팅 인자 $g$는 다음과 같이 계산된다.

$$g = \sigma(\text{Linear}([x; x_{\text{skip}}])) = \sigma(W[x; x_{\text{skip}}] + b)$$

최종 적응적 재구성 결과 $X_{\text{adapt}}$는 다음과 같이 결정된다.

$$X_{\text{adapt}} = g \odot x_{\text{skip}} + (1-g) \odot x$$

여기서 $g$가 1에 가까우면 전역 문맥(Mamba 경로)을 중시하고, 0에 가까우면 국소 특징(메인 경로)을 중시하게 된다.

### 이상치 판별 기준 (Anomaly Criterion)

최종 이상치 점수는 연관성 불일치(Association Discrepancy)와 재구성 오차의 곱으로 정의된다.

$$\text{AnomalyScore}(X) = \text{Softmax}(-\text{AssDis}(P, S; X)) \odot \|X_{i,:} - X_{\text{adapt} i,:}\|_2^2$$

이 공식은 시간적 패턴의 이상함(Association Discrepancy)과 값 자체의 복원 실패(Reconstruction Error)를 동시에 고려하여 탐지 성능을 극대화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MSL, SMAP, PSM, SMD, SWaT (NASA 및 산업 데이터) 및 NIPS-TS-SWAN, NIPS-TS-GECCO 등 총 8개의 벤치마크를 사용하였다.
- **비교 대상**: Anomaly Transformer, DCdetector, LSTM-VAE, OmniAnomaly 등 기존 SOTA 모델들과 비교하였다.
- **평가 지표**: Precision, Recall, F1-Score를 비롯하여 시간적 연속성을 고려한 Affiliation metrics (Aff-P, Aff-R), Range-based metrics (RAR, RAP), Volume-based metrics (VROC, VPR)를 종합적으로 사용하였다.

### 주요 결과

- **정량적 성능**: 대부분의 데이터셋에서 MAAT가 기존 모델들을 압도하였다. 특히 SMD 데이터셋에서 Anomaly Transformer 대비 F1-Score가 $+2.18\%$, DCdetector 대비 $+8.64\%$ 향상되었다.
- **특수 데이터셋 성능**: NIPS-TS-GECCO와 NIPS-TS-SWAN 데이터셋에서도 매우 강력한 성능을 보였다. GECCO 데이터셋의 경우 DCdetector 대비 F1-score가 $6.2\%$ 향상되었으며, SWAN 데이터셋에서는 Anomaly Transformer 대비 Recall이 $12.5\%$ 증가하는 성과를 거두었다.
- **재구성 성능**: 재구성 손실(Reconstruction Loss) 분석 결과, MAAT가 Anomaly Transformer보다 지속적으로 낮은 손실 값을 기록하며 더 정확한 신호 복원이 가능함을 입증하였다.

## 🧠 Insights & Discussion

### 구성 요소별 기여도 (Ablation Study)

- **Sparse Attention (SA)**: 노이즈로 인한 False Positive를 줄여 Precision을 높이는 효과가 있으나, 단독 사용 시 일부 국소 패턴을 놓쳐 Recall이 약간 감소하는 경향이 있었다.
- **Mamba-SSM**: 장거리 의존성 모델링을 통해 재구성을 안정화시키고 Recall을 보완한다.
- **Gated Attention**: SA의 정밀함과 Mamba의 전역적 문맥 파악 능력을 적응적으로 조절함으로써, 두 모듈의 단점을 상쇄하고 시너지를 낸다. 갑작스러운 변화(Sudden events) 시에는 SA에, 안정적인 구간에서는 Mamba에 가중치를 둠으로써 강건성을 확보하였다.

### 한계 및 비판적 해석

MAAT는 높은 성능을 보이지만 몇 가지 한계점이 존재한다.

1. **하이퍼파라미터 민감도**: 재구성 모듈과 Mamba 경로 사이의 균형을 맞추는 하이퍼파라미터 튜닝이 성능에 큰 영향을 미친다.
2. **극심한 노이즈 환경**: 노이즈가 매우 심하거나 이상치 패턴이 정상 패턴과 거의 흡사한 경우 성능이 저하될 수 있다.
3. **추론 속도**: 훈련 효율성은 개선되었으나, 실제 실시간 적용을 위해서는 추론(Inference) 속도의 추가적인 최적화가 필요하다.

## 📌 TL;DR

본 논문은 시계열 이상치 탐지를 위해 **Sparse Attention**과 **Mamba-SSM**을 **Gated Attention**으로 융합한 **MAAT** 구조를 제안한다. 이 모델은 Transformer의 계산 복잡도 문제를 해결함과 동시에, 국소적 세부 패턴과 전역적 장기 의존성을 모두 효과적으로 캡처하여 기존 SOTA 모델(Anomaly Transformer, DCdetector)보다 뛰어난 F1-Score와 Recall을 달성하였다. 특히 노이즈가 많은 실제 산업 데이터 및 고차원 시계열 데이터에서 강력한 일반화 성능을 보여, 향후 실시간 산업 모니터링 시스템의 기반 기술로 활용될 가능성이 높다.
