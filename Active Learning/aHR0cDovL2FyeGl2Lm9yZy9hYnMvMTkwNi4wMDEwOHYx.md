# ActiveHARNet: Towards On-Device Deep Bayesian Active Learning for Human Activity Recognition

Gautham Krishna Gudur, Prahalathan Sundaramoorthy, Venkatesh Umaashankar (2019)

## 🧩 Problem to Solve

본 연구는 웨어러블 기기를 이용한 인간 활동 인식(Human Activity Recognition, HAR) 시스템을 온디바이스(On-device) 환경에서 구현할 때 발생하는 두 가지 핵심 문제를 해결하고자 한다.

첫째, 기존의 딥러닝 모델은 높은 연산량으로 인해 주로 서버나 GPU 환경에서 동작하며, 새로운 사용자의 행동 패턴을 학습하기 위한 온디바이스 증분 학습(Incremental Learning) 지원이 부족하다. 실시간으로 유입되는 데이터로부터 사용자 특성을 학습하면서도 자원 효율성을 유지하는 모델이 필요하다.

둘째, 실시간 데이터에 대한 정답 레이블(Ground Truth)을 확보하는 과정에서 발생하는 비용 문제이다. 모든 데이터에 대해 오라클(Oracle, 전문가 또는 사용자)에게 레이블을 요청하는 것은 불가능하므로, 가장 정보 가치가 높은 데이터만을 선택적으로 쿼리하는 능동 학습(Active Learning) 기법의 도입이 필수적이다.

결과적으로 본 논문의 목표는 자원 제한적인 온디바이스 환경에서 모델의 불확실성을 측정하여 효율적으로 데이터를 선택하고, 이를 통해 모델을 지속적으로 업데이트할 수 있는 Bayesian Active Learning 기반의 HAR 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Bayesian Neural Networks(BNNs)의 근사치인 MC-dropout을 사용하여 모델의 불확실성(Uncertainty)을 추정하고, 이를 능동 학습의 획득 함수(Acquisition Function)와 결합하는 것이다.

주요 기여 사항은 다음과 같다.
1. **자원 효율적인 Bayesian 딥러닝 모델 제안**: 매우 적은 수의 파라미터($\sim 31,000$개)를 가진 경량화된 아키텍처를 통해 온디바이스 환경에서도 동작 가능한 BNN 모델을 설계하였다.
2. **불확실성 기반의 능동 학습 적용**: 다양한 획득 함수를 분석하여 오라클의 레이블링 부하를 최소화하면서도 모델 성능을 빠르게 향상시킬 수 있는 최적의 데이터 선택 전략을 제시하였다.
3. **온디바이스 증분 학습 구현**: 사용자가 변경되더라도 모델을 처음부터 다시 학습시킬 필요 없이, 유입되는 데이터의 일부만을 사용하여 모델을 업데이트하는 사용자 적응형(User Adaptability) 시스템을 구현하였다.

## 📎 Related Works

기존의 HAR 연구들은 주로 합성곱 신경망(CNN)이나 순환 신경망(RNN)을 사용하여 높은 성능을 달성하였으나, 대부분의 데이터가 이미 레이블링 되어 있다고 가정하거나 서버 기반의 연산을 전제로 한다. 

능동 학습(Active Learning) 분야에서는 주로 저차원 데이터에 대한 불확실성 추정이 이루어졌으며, 고차원 데이터인 딥러닝 모델에 적용하는 연구는 상대적으로 부족했다. 특히 센서 기반의 시계열 데이터(Time-series data)를 대상으로 온디바이스 환경에서 증분 학습과 능동 학습을 결합한 연구는 거의 수행되지 않았다. 

기존의 일부 연구에서는 k-means 클러스터링이나 Hidden Markov Models(HMMs)를 활용하여 레이블링 비용을 줄이려 했으나, 이는 수동으로 설계된 특징(Heuristic hand-picked features)에 의존하거나 딥러닝에 비해 성능이 낮다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조
ActiveHARNet은 데이터 수집, 불확실성 측정, 데이터 쿼리, 그리고 모델 업데이트로 이어지는 파이프라인을 가진다. 특히 Bayesian approximation을 통해 모델의 예측값뿐만 아니라 그 예측의 불확실성을 함께 산출하여 능동 학습에 활용한다.

### 모델 아키텍처 (HARNet)
특징 추출을 위해 1D CNN과 2D CNN을 결합한 구조를 사용한다.
1. **Intra-Axial Dependency**: 2개의 stacked 1D CNN 레이어(필터 수 8, 16 / 커널 크기 2)를 통해 가속도계의 각 축 내부의 특징을 추출한다. 이후 Batch Normalization과 Max-pooling을 적용한다.
2. **Inter-axial Dependency**: 앞서 추출된 축별 특징들을 결합하여 2개의 stacked 2D CNN 레이어(필터 수 8, 16 / 리셉티브 필드 $3 \times 3$)를 통해 축 간의 상호작용 특징을 추출한다.
3. **Classification**: 두 개의 Fully-Connected(FC) 레이어(뉴런 수 16, 8)와 ReLU 활성화 함수, L2 정규화를 적용하며, 최종적으로 Softmax 레이어를 통해 클래스 확률을 출력한다.

### 불확실성 모델링 및 MC-dropout
본 모델은 Bayesian Neural Network의 사후 분포(Posterior distribution)를 직접 계산하는 것이 매우 어렵다는 점을 해결하기 위해, Dropout을 Bayesian 근사치로 사용하는 MC-dropout 방식을 채택한다.

추론 시에 Dropout을 끄지 않고 유지한 채 $T$번의 stochastic forward pass를 수행하며, 이를 통해 예측값의 평균과 분산을 구함으로써 모델의 불확실성을 측정한다. 본 논문에서는 $T=10$을 최적의 반복 횟수로 설정하였다.

### 능동 학습 획득 함수 (Acquisition Functions)
모델 $M$과 데이터 풀 $D_{pool}$이 주어졌을 때, 다음 쿼리 포인트 $x^*$를 선택하는 함수 $a(x, M)$를 정의한다.
$$x^* = \arg\max_{x \in D_{pool}} a(x, M)$$

본 연구에서는 다음 네 가지 함수를 비교 분석하였다.
1. **Max Entropy**: 예측 확률 분포의 엔트로피를 최대화하는 점을 선택한다.
   $$H[y|x, D_{train}] := -\sum_{c} p(y=c|x, D_{train}) \log p(y=c|x, D_{train})$$
2. **BALD (Bayesian Active Learning by Disagreement)**: 예측값과 모델 파라미터 간의 상호 정보량(Mutual Information)을 최대화하여, 모델의 가중치들이 서로 가장 많이 의견이 갈리는 점을 선택한다.
   $$I[y, \omega | x, D_{train}] = H[y|x, D_{train}] - \mathbb{E}_{p(\omega|D_{train})} [H[y|x, \omega]]$$
3. **Variation Ratios (VR)**: 가장 확신이 낮은(Least Confident) 샘플을 선택하는 방식으로, 최대 확률값을 1에서 뺀 값을 사용한다.
   $$\text{variation-ratio}[x] := 1 - \max_{y} p(y|x, D_{train})$$
4. **Random Sampling**: 무작위로 샘플을 선택한다.

## 📊 Results

### 실험 설정
- **데이터셋**: HHAR(6개 활동, 9명 사용자) 및 Notch(낙상 감지, 7명 사용자) 데이터셋 사용.
- **전처리**: 2초 비중첩 윈도우 분할, Decimation(샘플링 주파수 통일), Discrete Wavelet Transform(DWT)을 통한 데이터 압축($\sim 50\%$ 감소) 수행.
- **평가 방법**: Leave-One-User-Out (LOOCV) 전략을 사용하여 사용자 적응성을 평가하였다.
- **하드웨어**: Raspberry Pi 2에서 구현하여 실제 온디바이스 성능을 측정하였다.

### 정량적 결과
1. **HHAR 데이터셋**: 
   - 베이스라인 평균 정확도는 $\sim 61\%$였으나, VR 획득 함수를 이용한 증분 학습 후 최대 $\sim 86\%$까지 향상되었다.
   - 특히 성능이 가장 낮았던 사용자 'i'의 경우, 데이터 풀의 $50\%$($\eta=0.5$)만 쿼리하여도 정확도가 $25\%$에서 $70\%$로 급격히 상승하였다.
   - 전체 데이터 풀을 다 사용했을 때($\eta=1.0$, $85.87\%$)와 비교해 매우 적은 양의 데이터($\eta=0.4$, $83.05\%$)만으로도 경쟁력 있는 성능을 얻었다.

2. **Notch 데이터셋 (낙상 감지)**:
   - F1-score 기준으로 베이스라인 $0.928$에서 $\eta=0.6$일 때 $0.948$까지 향상되었다.
   - VR 함수가 가장 효율적이었으며, 적은 수의 윈도우만으로도 빠르게 수렴함을 확인하였다.

3. **온디바이스 효율성**:
   - **모델 크기**: HHAR $\sim 315\text{kB}$, Notch $\sim 180\text{kB}$로 매우 작다.
   - **연산 시간**: 윈도우당 추론 시간은 $11\sim 14\text{ms}$이며, $T=10$번의 MC-dropout 반복을 통한 데이터 쿼리에는 평균 $\sim 14\text{초}$가 소요된다.

## 🧠 Insights & Discussion

본 연구의 결과는 온디바이스 환경에서 딥러닝 모델을 배포할 때, 단순한 추론을 넘어 **'불확실성 측정 $\rightarrow$ 효율적 데이터 선택 $\rightarrow$ 증분 학습'**의 루프를 구축하는 것이 실용적임을 시사한다.

특히 **Variation Ratios(VR)** 함수가 BALD나 Max Entropy보다 HAR 작업에서 더 빠르게 수렴하고 높은 성능을 보였다는 점은 흥미롭다. 이는 센서 데이터의 특성상 복잡한 상호 정보량 계산보다 단순한 확신도 기반의 선택이 더 효과적일 수 있음을 의미한다.

또한, 사용자별 활동 수행 방식의 차이로 인해 베이스라인 성능 편차가 매우 컸으나(HHAR의 경우 $25\% \sim 84\%$), 능동 학습을 통한 증분 학습이 이러한 사용자 간 편차를 효과적으로 줄여준다는 점이 입증되었다.

한계점으로는, 오라클이 레이블을 제공하는 시점과 데이터 수집 시점 사이의 시간 간격이 너무 길면 사용자가 자신의 활동을 기억하지 못할 수 있다는 점이 언급되었다. 이를 해결하기 위해 본 논문은 데이터 쿼리를 특정 시간 단위(예: 15분)로 제한하는 벤치마크 방식을 제안하였다.

## 📌 TL;DR

본 논문은 자원 제한적인 온디바이스 환경에서 동작하는 Bayesian 기반의 HAR 및 낙상 감지 프레임워크인 **ActiveHARNet**을 제안한다. MC-dropout을 통해 모델의 불확실성을 추정하고, Variation Ratios 획득 함수를 사용하여 가장 정보 가치가 높은 데이터만을 선택적으로 학습함으로써, 매우 적은 양의 레이블링 데이터만으로도 새로운 사용자에 빠르게 적응하는 증분 학습을 구현하였다. 이 연구는 향후 웨어러블 헬스케어 기기에서 개인 맞춤형 실시간 행동 모니터링 시스템을 구축하는 데 중요한 기반이 될 수 있다.