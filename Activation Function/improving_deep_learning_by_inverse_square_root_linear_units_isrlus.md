# IMPROVINGDEEPLEARNING BYINVERSESQUARE ROOTLINEARUNITS(ISRLUS)

Brad Carlile, Guy Delamarter, Paul Kinney, Akiko Marti, Brian Whitney

## 🧩 Problem to Solve

본 논문은 딥러닝 신경망의 학습 속도를 저해하는 요소를 해결하고자 합니다. 기존의 활성화 함수인 ReLU는 양수 범위에서만 활성화되어 '죽은 뉴런(dead neuron)' 문제를 일으킬 수 있으며, ELU와 같은 고급 활성화 함수는 계산 복잡성이 높아 전체 학습 시간을 증가시킬 수 있습니다. 특히 컨볼루션 연산의 효율성이 증가하면서 활성화 함수의 계산 비용이 전체 딥러닝 학습 성능에 미치는 영향이 커지고 있습니다. 또한, 활성화 함수의 출력 평균값이 0에서 멀어질 경우 바이어스 시프트(bias shift) 문제가 발생하여 학습을 늦출 수 있습니다.

## ✨ Key Contributions

- **ISRLU(Inverse Square Root Linear Unit) 도입:** 딥 컨볼루션 신경망(CNN)의 학습 속도와 정확성을 향상시키기 위한 새로운 활성화 함수인 ISRLU를 제안합니다.
- **계산 효율성:** ISRLU는 역제곱근(inverse square root) 연산을 기반으로 하여 ELU보다 계산 복잡성이 훨씬 낮아, CPU 및 맞춤형 하드웨어 구현에서 상당한 성능 이점을 제공합니다.
- **ELU의 장점 유지:** ISRLU는 ELU와 유사한 활성화 곡선 특성을 가지며, 음수 값을 허용하여 평균 활성화 값을 0에 가깝게 유지하고 바이어스 시프트를 줄여 학습 속도를 높입니다.
- **ISRU(Inverse Square Root Unit) 제안:** 순환 신경망(RNN)에서 주로 사용되는 tanh 및 sigmoid 활성화 함수를 대체할 수 있는 계산 효율적인 ISRU를 제안하며, 이는 미래 연구 방향을 제시합니다.
- **실험적 검증:** TensorFlow를 이용한 CNN 실험에서 ISRLU가 ReLU보다 더 빠른 학습과 우수한 일반화 성능을 보이며, ELU와 유사하거나 더 나은 정확도를 달성함을 입증했습니다.

## 📎 Related Works

- **Rectified Linear Unit (ReLU)** (Glorot et al., 2011): 널리 사용되는 활성화 함수로, 양수 입력에 대해 항등 함수, 음수 입력에 대해 0을 반환합니다.
- **Exponential Linear Unit (ELU)** (Clevert et al., 2015): ReLU의 대안으로, 음수 입력에 대해 -1로 점근적으로 접근하는 지수 함수를 사용하며, 바이어스 시프트 감소에 기여합니다.
- **Natural Gradient** (Amari, 1998; Clevert et al., 2015): Fisher 최적 학습을 위한 개념으로, 활성화 함수가 0 주변에 집중되거나 음수 값을 가질 때 바이어스 시프트를 줄일 수 있음을 설명합니다.
- **Parametric ReLUs (PReLUs)** (He et al., 2015): ReLU의 음수 기울기($\alpha$)를 학습 가능한 파라미터로 도입한 함수입니다.
- **Self-normalizing neural networks** (Klambauer et al., 2017): 자기 정규화 신경망의 맥락에서 ISRLU의 더 깊은 포화(saturation)에 대한 잠재적 적용 가능성이 언급되었습니다.
- **CNN 효율성 개선 기법:** Inception-v3, -v4 아키텍처 (Szegedy et al., 2016)의 필터 분해 및 Winograd의 최소 필터링 알고리즘 (Lavin & Gray, 2016; Winograd, 1980) 등 컨볼루션 계산 복잡도를 줄이는 방법들이 참조되었습니다.
- **고속 역제곱근 계산:** John Carmack과 Terje Mathisen (Lomont, 2003)의 고속 역제곱근 구현 트릭이 언급되었으며, 본 논문의 저자 중 한 명인 Kinney도 1986년에 유사한 방법을 발명했습니다.
- **RNN 활성화 함수:** LSTM (Hochreiter & Schmidhuber, 1997) 및 GRU (Chung et al., 2014)에서 사용되는 sigmoid 및 tanh 함수가 ISRU의 비교 대상으로 언급되었습니다.

## 🛠️ Methodology

1. **ISRLU(Inverse Square Root Linear Unit) 정의:**
   ISRLU는 다음과 같이 정의됩니다:
   $$f(x) = \begin{cases} x & \text{if } x \ge 0 \\ x \left( \frac{1}{\sqrt{1 + \alpha x^2}} \right) & \text{if } x < 0 \end{cases}$$
   그 미분은 다음과 같습니다:
   $$f'(x) = \begin{cases} 1 & \text{if } x \ge 0 \\ \left( \frac{1}{\sqrt{1 + \alpha x^2}} \right)^3 & \text{if } x < 0 \end{cases}$$
   여기서 $\alpha$는 하이퍼파라미터로, 음수 입력에 대한 포화 값을 제어하며, 훈련 중에 학습될 수 있습니다. ISRLU는 ELU와 유사한 곡선 특성을 가지면서도 첫 번째 및 두 번째 미분이 모두 부드럽고 연속적입니다.

2. **ISRU(Inverse Square Root Unit) 정의:**
   RNN을 위해 제안된 ISRU는 다음과 같이 정의됩니다:
   $$f(x) = x \left( \frac{1}{\sqrt{1 + \alpha x^2}} \right)$$
   그 미분은 다음과 같습니다:
   $$f'(x) = \left( \frac{1}{\sqrt{1 + \alpha x^2}} \right)^3$$

3. **계산 복잡성 분석:**

   - Intel Xeon CPU (Haswell, Broadwell, Skylake)에서 `InvSqrt`, `Exp`, `Tanh` 등 벡터 내장 함수들의 CPE(Cycles per Element)를 비교하여 역제곱근 연산이 지수 및 tanh 연산보다 훨씬 빠름을 정량적으로 보여줍니다.
   - AVX2 구현을 사용하는 Intel Core i7-7700 프로세서에서 ISRLU, ISRU, ELU, ReLU의 실제 성능(nsec/element)을 측정하여 ISRLU가 ELU보다 2.6배 빠르고, ISRLU의 빠른 근사치가 ReLU와 거의 동일한 속도를 가짐을 입증합니다.

4. **CNN 실험:**
   - **데이터셋:** MNIST 손글씨 숫자 분류 데이터셋을 사용했습니다 (6만 개 훈련, 1만 개 테스트).
   - **아키텍처:** 두 가지 다른 CNN 아키텍처를 사용했습니다.
     - **아키텍처 1:** 3개의 컨볼루션 레이어, 1개의 완전 연결 레이어, 소프트맥스 출력 레이어로 구성됩니다.
     - **아키텍처 2:** 4개의 컨볼루션 레이어, 2개의 Maxpooling, 2개의 Dropout, 1개의 완전 연결 레이어, 소프트맥스 출력 레이어로 구성됩니다.
   - **활성화 함수:** ISRLU($\alpha=1.0, \alpha=3.0$), ELU($\alpha=1.0$), ReLU를 비교했습니다.
   - **훈련:** ADAM 옵티마이저, 학습률 0.003에서 0.0001로 지수적으로 감소, 미니배치 크기 100, 가중치는 표준편차 0.1의 절단된 정규분포로 초기화되었습니다.

## 📊 Results

- **계산 속도:**

  - CPU 벤치마크 결과, 역제곱근(InvSqrt) 연산은 지수(Exp) 연산보다 1.2배 ~ 2.2배 빠르고, Tanh 연산보다 3.3배 ~ 6.9배 빠릅니다.
  - AVX2 구현에서 ISRLU는 ELU보다 2.6배 빠르게 계산됩니다.
  - ISRLU의 빠른 근사(approximation)는 ReLU의 계산 속도와 1% 이내의 차이를 보이며 유사한 성능을 나타냅니다.

- **MNIST CNN 분류 성능:**
  - **아키텍처 1 (Table 4):**
    - ISRLU($\alpha=3.0$, DropOut $p_{\text{keep}}=0.25$)가 99.30%의 최고 테스트 정확도와 2.308의 가장 낮은 교차 엔트로피 손실을 달성하여, ELU 및 ReLU보다 우수한 성능을 보였습니다.
    - 일반적으로 ISRLU와 ELU가 ReLU보다 더 나은 정확도와 낮은 손실을 보여주었습니다.
  - **아키텍처 2 (Table 5):**
    - ISRLU($\alpha=1.0$)는 99.32%, ISRLU($\alpha=3.0$)는 99.30%, ELU는 99.29%의 최고 테스트 정확도를 기록하며 세 함수 모두 유사하게 높은 정확도를 보였습니다. 이는 얕은 네트워크에서는 ELU와 ISRLU의 곡선이 비슷하여 큰 차이가 없을 수 있음을 시사합니다.
  - **학습 속도:** ISRLU 네트워크는 다른 활성화 함수를 사용한 네트워크보다 훈련 오류가 훨씬 더 빠르게 감소하는 것을 확인했습니다.

## 🧠 Insights & Discussion

- 컨볼루션 연산의 효율성 증가는 활성화 함수의 성능을 딥러닝 전체 학습 속도에서 더욱 중요한 요소로 만듭니다.
- ISRLU는 ELU와 유사한 활성화 특성(예: 음수 값, 평균 활성화 값을 0에 가깝게 유지)을 유지하여 바이어스 시프트를 줄이고 학습 속도를 높이는 이점을 가집니다.
- 가장 큰 장점은 ISRLU의 계산 복잡성이 ELU보다 현저히 낮다는 점입니다. 역제곱근 연산은 지수 연산보다 훨씬 빠르며, 이는 특히 CPU 및 전용 하드웨어 구현에서 큰 성능 향상을 가져올 수 있습니다.
- ISRLU의 $\alpha$ 하이퍼파라미터는 PReLU의 기울기처럼 학습될 수 있어 성능 최적화의 여지가 있습니다.
- 얕은 네트워크에서는 ISRLU와 ELU 간의 정확도 차이가 크지 않았지만, 논문 저자들은 더 깊은 네트워크에서 ISRLU의 이점이 ELU와 유사하게 더 크게 나타날 것으로 예상합니다.
- ISRU의 도입은 RNN에서 tanh 및 sigmoid를 대체할 수 있는 효율적인 대안을 제시하며, 향후 연구를 통해 RNN 성능 향상에 기여할 수 있을 것으로 보입니다.
- 고속 역제곱근 계산 트릭은 ISRLU의 하드웨어 구현에 영감을 줄 수 있으며, 효율적인 하드웨어 근사치는 FMA(fused multiply and add)와 유사한 실행 시간을 가질 것으로 예상됩니다.

## 📌 TL;DR

- **문제:** 딥러닝 학습에서 활성화 함수의 높은 계산 비용과 바이어스 시프트 문제로 인한 속도 저하.
- **제안:** **ISRLU (Inverse Square Root Linear Unit)** 및 **ISRU (Inverse Square Root Unit)**.
- **방법:** ISRLU는 $x \left( \frac{1}{\sqrt{1 + \alpha x^2}} \right)$ ($x < 0$일 때) 형태의 활성화 함수로, ELU와 유사하게 음수 값을 허용하여 평균 활성화 값을 0에 가깝게 유지함으로써 바이어스 시프트를 줄이고 학습을 가속화합니다. 주요 혁신은 지수 함수 대신 계산 효율적인 역제곱근 함수를 사용하는 것입니다. ISRU는 RNN의 tanh/sigmoid를 대체할 목적으로 제안됩니다.
- **결과:** ISRLU는 ELU보다 2.6배 빠르게 계산되며, MNIST CNN 분류에서 ReLU보다 더 빠른 학습과 우수한 일반화 성능을 보이고 ELU와 비슷한 최고 정확도를 달성합니다. ISRLU는 하드웨어 구현에 매우 유리하여 딥러닝 학습 효율성을 크게 높일 잠재력을 가집니다.
