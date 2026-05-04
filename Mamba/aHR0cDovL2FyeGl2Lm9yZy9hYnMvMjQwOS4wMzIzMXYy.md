# State-space models are accurate and efficient neural operators for dynamical systems

Zheyuan Hu, Nazanin Ahmadi Daryakenari, Qianli Shen, Kenji Kawaguchi, George Em Karniadakis (2025)

## 🧩 Problem to Solve

본 논문은 물리 기반 기계 학습(Physics-Informed Machine Learning, PIML)을 이용하여 동적 시스템(Dynamical Systems)의 연산자 학습(Operator Learning) 문제를 해결하고자 한다. 동적 시스템은 시간의 흐름에 따라 상태가 변화하는 시스템으로, 물리, 생물, 경제 등 다양한 분야에서 중요하게 다뤄지지만, 많은 경우 해석적 해(Analytical solution)를 구할 수 없어 수치적 해법에 의존해야 한다.

기존의 PIML 모델들은 다음과 같은 한계점을 가지고 있다.
- **Recurrent Neural Networks (RNNs):** 계산 비용은 선형적이지만, 유효 컨텍스트 윈도우의 제한으로 인해 장기 의존성(Long-range dependencies)을 캡처하지 못하며, 순차적 처리 방식으로 인해 GPU 병렬 처리 효율이 낮다.
- **Transformers:** Self-attention 메커니즘을 통해 장기 의존성을 잘 파악하지만, 시퀀스 길이에 대해 제곱 시간 복잡도 $O(n^2)$를 가지므로 장시간 적분(Long-time integration) 문제에서 효율성이 극도로 떨어진다.
- **Neural Operators (DeepONet, FNO, LNO):** 비선형 ODE/PDE 연산자 근사는 가능하지만, 동적 시스템의 시간적 정보(Temporal information)를 충분히 고려하지 못하며, 가변 길이 입력 처리나 더 긴 시퀀스로의 일반화 능력이 부족하다.

따라서 본 연구의 목표는 장기 의존성을 효율적으로 캡처하면서도 계산 복잡도를 낮추고, 특히 훈련 데이터 분포를 벗어난 외삽(Extrapolation) 상황에서도 강건한 성능을 보이는 새로운 신경 연산자 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Mamba**로 구현된 **상태 공간 모델(State-Space Models, SSMs)**을 동적 시스템 연산자 학습에 도입하는 것이다. 

중심적인 설계 직관은 다음과 같다.
1. **선택적 메커니즘(Selection Mechanism):** 입력 데이터에 따라 동적으로 상태를 조정하여 장기 의존성을 효율적으로 포착한다.
2. **선형 복잡도와 병렬성:** 재매개변수화(Reparameterization) 기술을 통해 RNN의 선형 복잡도와 Transformer의 병렬 학습 능력을 동시에 확보한다.
3. **강건한 외삽 능력:** 단순한 보간(Interpolation)을 넘어, 훈련되지 않은 긴 시간 영역이나 다른 분포의 입력 함수에 대해서도 높은 일반화 성능을 유지하도록 설계하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 검토하고 차별점을 제시한다.

- **Neural Operators:** DeepONet은 보편 근사 정리를 기반으로 하며, FNO는 푸리에 공간에서 커널 적분을 수행하고, LNO는 라플라스 공간에서 전이 응답을 캡처한다. 하지만 이들은 전반적으로 시간적 의존성을 내재적으로 인코딩하지 못해 오버피팅에 취약하거나 일반화 능력이 떨어진다.
- **Transformers:** Self-attention을 통해 복잡한 상호작용을 모델링하지만 계산 비용이 문제다. 이를 해결하기 위해 Galerkin attention과 같은 선형 복잡도 변형 모델들이 제안되었으나, 효율성과 모델 용량 사이의 트레이드오프가 존재한다.
- **State-Space Models (SSMs):** S4와 같은 구조적 SSM은 효율적인 대안으로 제시되었으며, Mamba는 여기서 더 나아가 선택적 메커니즘을 통해 컨텍스트를 동적으로 캡처함으로써 Transformer에 필적하는 성능과 선형 효율성을 동시에 달성하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 문제 정의
본 연구는 동적 시스템 연산자 학습을 시퀀스-투-시퀀스(seq2seq) 문제로 정의한다. 입력 함수 $x(t)$와 출력 함수 $y(t) = (Ox)(t)$를 이산 시간 그리드 $\{t_1, t_2, \dots, t_{N_{grid}}\}$ 상의 값으로 표현하여, 입력 시퀀스에서 출력 시퀀스로 매핑하는 연산자 $O$를 학습한다.

### SSM 및 Mamba 아키텍처
기본적인 상태 공간 모델(SSM)은 다음과 같은 연속 시간 시스템으로 정의된다.
$$\dot{h}(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$
여기서 $h(t)$는 은닉 상태(Hidden state)이다. 실제 구현을 위해 **Zero-Order Hold (ZOH)** 규칙을 사용하여 이산화하며, 다음과 같은 재귀식으로 표현된다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$
이때 $\bar{A} := \exp(\Delta \cdot A)$, $\bar{B} := (\Delta \cdot A)^{-1}(\exp(\Delta \cdot A) - I) \cdot \Delta B$ 이며, $\Delta$는 이산화 단계 크기이다. 

학습의 효율성을 위해 이 재귀식은 합성곱(Convolution) 형태로 변환되어 FFT(Fast Fourier Transform)를 통해 병렬 처리된다.
$$y = u * \bar{K}$$
여기서 $\bar{K}$는 합성곱 커널이다.

**Mamba 블록**은 위 SSM 구조를 기반으로 하며, 두 개의 브랜치로 구성된다.
1. **SSM 브랜치:** 선형 투영(Linear Projection) $\rightarrow$ 1D 합성곱(Conv) $\rightarrow$ 비선형 활성화 함수 $\sigma$ $\rightarrow$ SSM 변환 순으로 진행된다.
2. **Skip Connection 브랜치:** 선형 투영 후 비선형 활성화를 거친다.
최종 출력은 두 브랜치의 결과물을 곱하고 다시 선형 변환하여 얻는다.

### 적용 방식
Mamba는 다음과 같은 다양한 시나리오에 적용 가능하다.
- **시퀀스-투-시퀀스 매핑:** 외부 힘(External force)이 존재하는 시스템.
- **초기 조건 기반 예측:** 입력 시퀀스를 초기값 $u_0$의 상수 시퀀스로 구성하여 해결.
- **파라미터 결합:** ODE 계수 $\alpha$를 입력 시퀀스에 결합($\tilde{u}(t) = [u(t), \alpha]$)하여 다양한 ODE를 동시에 해결.

## 📊 Results

### 실험 설정
- **데이터셋:** 1D 동적 시스템, Izhikevich 뉴런 모델, Tempered Fractional LIF 모델, Lorenz 시스템, Duffing 진동자, 중력 진자, PK-PD 모델 등.
- **비교 모델:** GRU, LSTM, DeepONet, FNO, LNO, Transformer, Oformer (Vanilla/Galerkin/Fourier), GNOT.
- **지표:** 평균 제곱 오차(MSE) 및 상대 $L^2$ 오차(Relative $L^2$ error).

### 주요 결과
1. **기본 성능 (1D Systems):** Mamba는 모든 작업(반도함수 연산자, 비선형 ODE, 중력 진자)에서 가장 낮은 MSE를 기록하며 타 모델을 압도하였다. 특히 RNN과 유사한 낮은 메모리 비용과 빠른 학습 속도를 유지하면서 정확도는 Transformer보다 높았다.
2. **불연속 해 처리 (Finite Regularity):** Izhikevich 및 LIF 모델에서 Mamba는 불연속적인 스파이킹 현상을 성공적으로 캡처하였으며, 특히 Izhikevich 모델의 학습 과정에서 RNN보다 더 안정적인 손실 함수 수렴 곡선을 보였다.
3. **OOD 일반화 및 외삽:** 훈련 분포와 다른 테스트 분포를 사용하는 OOD 실험에서 Mamba는 매우 강건한 성능을 보였다. 특히 댐핑(Damping)이 없는 시스템이나 카오스 시스템에서 LNO가 일부 우위를 보였으나, 전반적인 안정성은 Mamba가 가장 높았다.
4. **장시간 적분 (Long-Time Integration):** 시퀀스 길이를 최대 32,768까지 확장했을 때, Mamba는 메모리와 시간 비용이 선형적으로 증가함을 확인하였다. 반면 일반 Transformer는 $O(n^2)$ 비용으로 인해 메모리 부족(OOM)이 발생하였다.
5. **시간 외삽 (Time Extrapolation):** $[0, 1]$ 구간에서 훈련하고 $[0, 4]$ 구간을 예측하는 테스트에서 Mamba는 다른 모든 모델보다 현저히 낮은 오차를 기록하며 뛰어난 외삽 능력을 입증하였다.
6. **실제 응용 (PK-PD 모델):** 암세포 성장 모델링에 적용한 결과, 상대 $L^2$ 오차가 0.1685%에 불과할 정도로 정밀한 예측이 가능함을 확인하였다. 또한 데이터가 매우 부족한 상황(5개 샘플)에서도 물리 정보(Physics-informed loss)를 결합했을 때 성능이 크게 향상됨을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
Mamba는 RNN의 효율성과 Transformer의 표현력을 동시에 갖춘 모델이다. 특히 동적 시스템 학습에서 다음과 같은 통찰을 제공한다.
- **시간적 의존성 인코딩:** Mamba는 시퀀스 모델로서 "미래의 상태는 과거의 상태에만 의존한다"는 인과적 특성을 자연스럽게 학습에 활용한다. 이는 시간 정보를 무시하는 일반 신경 연산자(FNO, DeepONet)보다 우월한 성능을 보이는 이유이다.
- **댐핑 효과와 오차:** 실험 결과, 시스템에 댐핑 효과가 있을 때는 시간이 흐를수록 오차가 감소하는 경향이 있고, 카오스 시스템에서는 시간이 지남에 따라 오차가 선형적으로 증가하는 특성이 발견되었다. 이는 리아푸노프 지수(Lyapunov exponent)에 따른 궤적의 발산 특성이 모델의 예측 오차에 직접적인 영향을 미침을 시사한다.

### 한계 및 비판적 논의
- **카오스 시스템의 한계:** 매우 격렬하게 진동하는 무댐핑 시스템이나 강한 카오스 시스템에서는 여전히 LNO와 같은 모델이 전이 응답을 더 잘 캡처하는 경항이 있다. 이는 SSM이 고주파 성분을 완벽하게 처리하는 데 아직 한계가 있을 수 있음을 의미한다.
- **데이터 의존성:** 데이터가 극도로 적은 상황에서는 순수 데이터 기반 학습만으로는 한계가 있으며, 물리 법칙을 손실 함수에 통합하는 하이브리드 방식이 필수적이다.

## 📌 TL;DR

본 논문은 동적 시스템의 연산자 학습을 위해 선택적 상태 공간 모델인 **Mamba**를 도입하여, 기존 RNN, Transformer, Neural Operator들의 한계인 **계산 비용, 장기 의존성 캡처, 외삽 능력** 문제를 동시에 해결하였다. Mamba는 선형적인 계산 복잡도를 유지하면서도 매우 긴 시퀀스 처리와 훈련 분포 밖의 데이터에 대한 강력한 일반화 성능을 보여주었으며, 실제 약물 효능 분석(PK-PD) 모델에서도 높은 정밀도를 입증하였다. 이 연구는 과학적 기계 학습(SciML) 분야에서 SSM이 복잡한 물리 시스템을 모델링하는 효율적이고 강력한 도구가 될 수 있음을 시사한다.