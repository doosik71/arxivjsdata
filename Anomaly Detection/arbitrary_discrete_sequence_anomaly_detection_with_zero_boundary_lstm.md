# Arbitrary Discrete Sequence Anomaly Detection with Zero Boundary LSTM

Chase Roberts, Manish Nair (2018)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 이산 시퀀스 데이터셋(Discrete Sequence Datasets), 특히 텍스트나 바이트 데이터에서 정상적인 패턴을 따르지 않는 이상치(Anomaly)를 탐지하는 것이다. 이산 시퀀스 이상 탐지는 사이버 보안이나 항공 안전과 같은 중요한 도메인에서 매우 가치 있는 문제이지만, 실제로 구현하는 데에는 큰 어려움이 따른다.

가장 핵심적인 문제는 '무엇이 시퀀스를 이상하게 만드는가'에 대한 공식적인 정의를 내리는 것이 매우 까다롭다는 점이다. 이러한 정의의 부재는 알고리즘의 성공 여부를 측정하는 지표를 불분명하게 만들며, 결국 인간의 주관적인 검증에 의존하게 만든다. 따라서 본 연구의 목표는 이산 시퀀스에 대한 이론적인 이상치 정의를 제시하고, 이 정의로부터 직접 유도된 머신러닝 아키텍처를 개발하여 안정적인 이상 탐지 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 이산 시퀀스의 이상치를 "다음 요소가 나타날 확률이 0인 경우"로 정의하고, 이를 판별하기 위해 LSTM의 Context Vector와 각 알파벳별로 구성된 One-Class SVM(OCSVM) 배열을 결합하는 것이다.

단순히 LSTM의 예측 확률값만을 사용하는 기존 방식은 학습 데이터의 빈도수에 따라 예측 분산이 커져 임계값(Threshold) 설정을 어렵게 만든다. 이를 해결하기 위해 본 연구에서는 LSTM을 통해 시퀀스의 문맥 정보를 압축한 Context Vector를 추출하고, 각 문자(Element)별로 전용 OCSVM을 두어 해당 문자가 나타나기에 적절한 문맥 공간(Context Space)인지 여부를 결정하는 경계선을 학습시킨다.

## 📎 Related Works

기존의 이산 시퀀스 이상 탐지 방식은 크게 세 가지로 나뉜다.

1. **전통적 방식 (Sliding Window, Markov Chain, Subsequence Frequency Analysis):** 이러한 방식은 시퀀스 내의 국소적인 이상치는 잘 잡아내지만, 긴 거리에 걸친 의존성(Long-term dependencies)을 파악하지 못한다. 예를 들어 문장의 문법적 오류를 찾으려면 수십 개의 문자 전으로 돌아가 확인해야 하는데, n-gram 모델의 윈도우 크기를 무작정 키우면 더 많은 학습 데이터가 필요하거나 오탐(False Positive)이 급격히 증가하는 문제가 발생한다.
2. **딥러닝 방식 (LSTM 기반 예측):** LSTM을 통해 다음 요소의 확률 분포를 예측하고, 이 확률이 특정 임계값보다 낮을 때 이상치로 판별하는 방식이다. 하지만 Standard Minibatch 및 Cross Entropy Loss를 사용하여 학습할 경우, 드물게 나타나는 값들의 예측 분산이 매우 커진다. 이로 인해 정상과 이상을 가르는 경계가 모호해지며, 알람 시스템으로 사용할 때 너무 많은 오탐이 발생하거나 정탐을 놓치는 불안정성이 나타난다.
3. **Autoencoder 기반 방식:** 입력 데이터를 압축했다가 복원하는 reconstruction loss를 지표로 사용한다. 그러나 reconstruction loss가 기저 확률 분포의 에너지 함수의 그래디언트와 같기 때문에, 그래디언트가 0에 가까운 일부 이상치들이 정상으로 오분류되는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구에서 제안하는 **Zero Boundary LSTM**은 크게 LSTM 인코더, MLP 디코더, 그리고 알파벳 $\Sigma$의 각 요소에 대응하는 OCSVM 배열로 구성된다.

1. **LSTM Encoder ($E$):** 입력 시퀀스를 받아 고정된 크기의 벡터인 Context Vector(문맥 벡터)를 생성한다.
2. **MLP Decoder ($D$):** Context Vector를 바탕으로 다음 요소의 확률 분포를 예측한다.
3. **OCSVM Array ($O$):** 각 알파벳 $\sigma \in \Sigma$에 대해 개별적인 $O_\sigma$가 존재하며, 특정 문맥 벡터가 해당 문자가 나타나기에 정상적인 범위 내에 있는지 판별한다.

### 학습 절차 및 손실 함수

학습은 두 단계로 진행된다.

**단계 1: LSTM Autoencoder 학습**
일반적인 Autoencoder가 입력을 그대로 복원하는 것과 달리, 본 모델은 현재까지의 시퀀스를 통해 다음 요소($i+1$번째)를 예측하도록 학습한다. 손실 함수로는 Cross Entropy Loss를 사용한다.

$$\mathcal{L} = \sum_{i=1}^{n-1} -\log P(x_{i+1} | D(E(x; \theta_E)_i; \theta_D))$$

이 과정을 통해 LSTM의 bottleneck layer에서 생성되는 Context Vector는 시퀀스의 구조와 데이터 분포에 대한 풍부한 정보를 담게 된다.

**단계 2: OCSVM 배열 학습**
학습 데이터셋 $X$에서 각 문자 $\sigma$가 나타날 때의 Context Vector들을 수집하여 집합 $Y_\sigma$를 구성한다.

$$Y_\sigma = \{E(x; \theta'_E)_i : x_i = \sigma \forall x \in X\}$$

각 $O_\sigma$는 이 $Y_\sigma$를 사용하여 해당 문자의 '정상 문맥 영역'을 정의하는 경계선을 학습한다. 이때 OCSVM은 다음의 Quadratic Programming 문제를 해결하여 경계를 최적화한다.

$$\min_{A} \sum_{ij} \alpha_i \alpha_j K(x_i, x_j) \quad \text{subject to} \quad 0 \le \alpha_i \le \frac{1}{\nu l}, \sum_i \alpha_i = 1$$

여기서 $K$는 Gaussian kernel이며, $\nu$는 이상치 비율을 제어하는 하이퍼파라미터이다.

### 추론 및 이상 판별

새로운 시퀀스 $x$가 들어왔을 때, 각 요소 $x_i$에 대해 이전 단계까지의 문맥 벡터 $E(x; \theta'_E)_{i-1}$를 해당 문자의 OCSVM $O_{x_i}$에 입력한다.

$$g(x) = \begin{cases} 1, & \text{if } \exists x_i \in x : O_{x_i}(E(x; \theta'_E)_{i-1}) < t_\sigma \\ 0, & \text{otherwise} \end{cases}$$

여기서 $t_\sigma$는 각 OCSVM별로 설정된 임계값으로, 주로 0보다 약간 작은 값으로 설정하여 제1종 오류(Type I error)를 제어한다.

## 📊 Results

### 실험 설정

- **데이터셋:** IPv4 주소 생성 데이터, JSON 객체 생성 데이터 (두 가지 Toy Dataset).
- **비교 대상:** Standard LSTM (Next character prediction), Naive Sliding Window (n-gram).
- **평가 지표:** 정량적 탐지 성능 및 학습 안정성.
- **구현 세부사항:** TensorFlow 및 Scikit-learn 사용. LSTM은 5개 층(각 128 hidden units), bottleneck MLP는 7개 층으로 구성.

### 주요 결과

1. **IPv4 데이터셋:**
    - trivial, length, digit, dot placement의 네 가지 이상 유형을 테스트했다.
    - Zero Boundary LSTM은 모든 카테고리에서 Standard LSTM보다 우수한 성능을 보였다.
    - n-gram 모델은 숫자 자체를 외울 수 있어 digit anomaly 탐지에는 강했으나, 윈도우 크기의 한계로 인해 digit grouping 개수가 너무 많은 경우(length anomaly)는 탐지하지 못했다. 반면 제안 모델은 이를 성공적으로 탐지했다.

2. **JSON 데이터셋:**
    - colon, comma, quote, nesting의 네 가지 이상 유형을 테스트했다.
    - 최적의 하이퍼파라미터를 적용했을 때 Zero Boundary LSTM은 Standard LSTM과 유사한 높은 성능을 기록했다.

3. **학습 안정성 (Training Stability):**
    - 가장 주목할 만한 결과는 Epoch 수에 따른 성능 변화이다. Standard LSTM은 Epoch에 따라 성능 변동이 심해 안정적인 결정 경계를 찾기 어려운 반면, Zero Boundary LSTM은 매우 일관되고 안정적인 성능 추이를 보였다.

## 🧠 Insights & Discussion

### 강점

본 모델의 가장 큰 강점은 **결정 경계의 안정성**이다. 표준 LSTM 기반 방식은 SGD 학습의 분산으로 인해 임계값 설정이 매우 까다롭지만, 본 연구는 문맥 벡터를 추출한 후 별도의 밀도 추정기(OCSVM)를 사용함으로써 이 문제를 해결했다. 또한, LSTM을 통해 장기 의존성을 캡처함으로써 n-gram 모델의 한계를 극복했다.

### 한계 및 비판적 해석

1. **학습 데이터의 순수성:** 제안된 시스템은 학습 데이터셋 내에 이상치가 포함되어 있을 때 매우 취약하다. $\nu$ 값을 조정하여 어느 정도 보완할 수 있으나, 이는 다시 오탐율(False Positive)을 높이는 트레이드-오프 관계에 있다.
2. **계산 복잡도:** OCSVM의 학습 시간 복잡도는 $O(n^3)$이다. 이는 데이터셋의 규모가 커질 경우 심각한 확장성 문제를 야기한다. 실제 대규모 네트워크 트래픽 데이터 등에 적용하기 위해서는 OCSVM을 대체할 수 있는 더 효율적인 밀도 추정 방법이 필요하다.
3. **실험 데이터의 단순성:** 실험이 생성된 Toy Dataset(IPv4, JSON)에서만 이루어졌으므로, 실제 복잡한 현실 세계의 데이터에서도 동일한 안정성과 성능이 유지될지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 이산 시퀀스 이상 탐지를 위해 **"다음 요소의 발생 확률이 0인 상태"**를 이론적 기반으로 삼고, **LSTM 인코더와 문자별 One-Class SVM 배열**을 결합한 **Zero Boundary LSTM** 아키텍처를 제안한다. 이 방법은 기존 LSTM 방식의 고질적인 문제였던 임계값 설정의 불안정성을 해결하여 매우 안정적인 탐지 성능을 보이며, n-gram 방식이 놓치는 장기 의존성 기반의 이상치까지 잡아낼 수 있다. 다만, OCSVM의 높은 계산 복잡도로 인해 대규모 데이터셋으로의 확장을 위해서는 최적화가 필수적이다.
