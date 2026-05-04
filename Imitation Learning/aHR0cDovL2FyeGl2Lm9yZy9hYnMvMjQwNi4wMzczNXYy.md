# Phase-Amplitude Reduction-Based Imitation Learning

Satoshi Yamamori and Jun Morimoto (2025)

## 🧩 Problem to Solve

본 논문은 로봇이 인간의 주기적인 움직임 궤적을 모방할 때 발생하는 **과도 상태(transient movement) 재현의 어려움**을 해결하고자 한다. 

일반적인 모방 학습, 특히 Dynamical Movement Primitives (DMP)와 같은 동역학 시스템 기반 접근 방식은 안정적인 폐곡선 궤적인 Limit Cycle 주변의 움직임은 잘 근사하지만, 초기 상태나 외부 교란이 발생한 지점에서 Limit Cycle로 수렴하는 과정인 과도 상태를 적절하게 표현하지 못하는 한계가 있다. 이로 인해 외부 교란이 발생한 직후나 특정 초기 상태에서 로봇이 예측 불가능하고 위험한 동작을 생성할 가능성이 크다.

따라서 본 연구의 목표는 Limit Cycle뿐만 아니라, 초기 상태 혹은 교란 상태에서 Limit Cycle로 복귀하는 과도 동역학까지 정확하게 모방할 수 있는 새로운 모방 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Phase-Amplitude Reduction(위상-진폭 축소)** 이론을 사용하여 고차원의 시스템 상태를 저차원의 위상(Phase)과 진폭(Amplitude) 공간으로 임베딩하는 것이다.

1. **과도 동역학의 명시적 모델링**: 위상-진폭 축소법을 통해 시스템 상태를 'Limit Cycle 상의 위치(위상)'와 'Limit Cycle로부터의 거리(진폭)'로 분리하여 표현함으로써, 수렴 과정인 과도 상태를 효율적으로 학습할 수 있게 하였다.
2. **Variational Inference 기반의 인코더-디코더 학습**: 변분 추론(Variational Inference) 프레임워크를 도입하여 관측 데이터로부터 위상-진폭 잠재 공간으로의 매핑(Encoder)과 다시 물리 공간으로의 복원(Decoder)을 안정적으로 학습하였다.
3. **Interactive Feedback 메커니즘**: 로봇의 실제 상태를 잠재 공간에 피드백하는 구조를 설계하여, 외부 교란이 발생했을 때 잠재 변수를 실시간으로 조절함으로써 안전하고 부드럽게 목표 궤적으로 복귀할 수 있도록 하였다.

## 📎 Related Works

### 기존 연구 및 한계
- **Trajectory-based Imitation Learning (DMP 등)**: DMP는 단순한 메커니즘으로 안정적인 궤적을 생성할 수 있지만, 앞서 언급했듯이 Limit Cycle 외부의 과도 상태를 표현하는 능력이 부족하다.
- **Latent Representation in Dynamics**: Koopman operator, Gaussian Process, Neural Networks 등을 이용해 비선형 동역학을 잠재 공간에서 학습하려는 시도가 많았다. 하지만 대부분의 방법은 잠재 공간에 특정한 동역학적 의미(예: 주기성, 수렴성)를 부여하지 않고 단순히 데이터의 분포나 매핑을 학습하는 데 집중한다.

### 본 연구의 차별점
본 연구는 잠재 공간의 구조를 **위상-진폭 방정식(Phase-Amplitude Equation)**이라는 특수한 동역학적 성질을 갖도록 강제한다. 이를 통해 잠재 공간 내에서 위상의 가속/감속을 통한 속도 조절과, 진폭 조절을 통한 궤적의 확장/축소가 가능해지며, 특히 물리적으로 의미 있는 '수렴 특성'을 직접적으로 다룰 수 있다는 점에서 기존의 블랙박스형 잠재 표현 학습과 차별화된다.

## 🛠️ Methodology

### 1. Phase-Amplitude Reduction Latent Dynamics
본 방법론은 물리적 상태 변수 $\mathbf{x} \in \mathbb{R}^n$를 저차원의 잠재 변수 $\mathbf{z} = [\phi, \mathbf{r}] \in \mathbb{R}^m$ ($m < n$)로 매핑한다. 여기서 $\phi$는 위상(Phase)을, $\mathbf{r}$은 진폭(Amplitude) 벡터를 의미한다.

잠재 공간에서의 동역학은 다음과 같은 위상-진폭 방정식으로 정의된다:
$$\dot{\mathbf{z}} = f(\mathbf{z}) = \begin{bmatrix} \omega \\ -\lambda \odot \mathbf{r} \end{bmatrix}$$
여기서 $\omega > 0$는 특성 주파수(characteristic frequency)이며, $\lambda > 0$는 수렴 속도를 결정하는 특성 지수(characteristic exponent)이다. $\odot$는 요소별 곱셈(element-wise multiplication)을 의미한다.

이 방정식의 해석적 해는 $\phi(t) = \phi(0) + \omega t$, $\mathbf{r}(t) = \exp(-\lambda t) \odot \mathbf{r}(0)$가 된다. 즉, 시간이 흐름에 따라 진폭 $\mathbf{r}$은 0으로 수렴하며, 이는 로봇이 Limit Cycle로 복귀함을 의미한다.

### 2. Encoder-Decoder 및 학습 절차
인코더 $h$는 물리 상태 $\mathbf{x}$를 잠재 변수 $\mathbf{z}$로 투영하고, 디코더 $g$는 $\mathbf{z}$를 다시 $\mathbf{x}$로 복원한다.

- **학습 프레임워크**: $\beta$-VAE와 유사하게 KL divergence 최소화를 기반으로 한 변분 추론을 사용한다. 
- **손실 함수**: 단순한 복원 오차뿐만 아니라, 잠재 공간에서의 동역학적 일관성을 유지하기 위해 다음과 같은 다각적인 손실 함수 $\mathcal{L}$을 정의한다:
  $$\mathcal{L} = \mathcal{L}_{Rec} + \mathcal{L}_{Enc} + \mathcal{L}_{Dec} + \mathcal{L}_{Lat} + \sqrt{\Delta t} \cdot \mathcal{L}_{Rec,Diff} + \sqrt{\Delta t} \cdot \mathcal{L}_{Dec,Diff}$$
  - $\mathcal{L}_{Rec}, \mathcal{L}_{Dec}$: 예측된 궤적과 실제 데이터 간의 복원 오차.
  - $\mathcal{L}_{Enc}, \mathcal{L}_{Lat}$: 인코더가 출력한 값과 동역학 모델이 예측한 값 사이의 일관성.
  - $\mathcal{L}_{Rec,Diff}, \mathcal{L}_{Dec,Diff}$: 속도 성분의 정확도를 높이기 위한 차분 기반 손실 함수.
- **네트워크 구조**: MLP를 사용하며, 위상 $\phi$를 처리하기 위해 인코더 끝단에 $\text{atan2}$ 함수를, 디코더 시작단에 $\sin, \cos$ 함수를 배치하여 위상의 주기성을 처리한다. 또한 $\pm \pi$ 경계에서의 불연속성을 해결하기 위해 'Unwrap' 과정을 거친다.

### 3. Interactive Feedback
외부 교란에 대응하기 위해 로봇 시스템과 잠재 동역학 시스템을 상호 결합한다:
$$\dot{\mathbf{x}} = \mathbf{K}(\mathbf{x}, \mathbf{u}), \quad \mathbf{u} = \mathbf{K} \Delta \mathbf{x}$$
$$\dot{\mathbf{z}} = f(\mathbf{z}) + \mathbf{\Gamma} \odot \Delta \mathbf{z}$$
여기서 $\Delta \mathbf{x} = g(\mathbf{z}) - \mathbf{x}$는 물리 공간의 오차이며, $\Delta \mathbf{z} = h(\mathbf{x}) - \mathbf{z}$는 잠재 공간의 오차이다. 로봇의 현재 상태가 인코더를 통해 잠재 공간으로 피드백됨으로써, 물리적 상태가 변하면 잠재 변수 $\mathbf{z}$가 즉각적으로 수정되어 궤적을 다시 추종하게 된다.

## 📊 Results

### 1. 단순 Limit Cycle 재구성
- **내용**: 2차원 상태 공간에서 단순한 Limit Cycle 어트랙터를 생성하고 이를 재구성하는 실험을 수행하였다.
- **결과**: 데이터 크기가 50k steps 이상일 때 RMSE가 충분히 낮아지며, 특히 Limit Cycle뿐만 아니라 초기 상태에서 궤적으로 수렴하는 과도 동역학을 성공적으로 복원함을 확인하였다.

### 2. Lemniscate (레무니스케이트) 궤적 추종 제어
- **비교 대상**: 기존의 DMP 방식과 비교하였다.
- **실험 시나리오 및 결과**:
    - **이상 상황(Anomaly)**: 제어 신호가 6초간 중단된 후 복구될 때, DMP는 급격하게 원래 위치로 돌아가려 하여 불안정한 움직임을 보였으나, 제안 방법은 Interactive Feedback을 통해 잠재 위상을 동기화하여 부드럽게 복귀하였다.
    - **외부 노이즈**: 가우시안 노이즈 주입 시, 제안 방법이 DMP보다 훨씬 빠르게 목표 궤적으로 수렴하였으며 RMSE 점수 또한 우수하였다.
    - **속도 및 형태 변경 (Slow motion, Reshaping)**: 위상 $\phi$와 진폭 $\mathbf{r}$이 분리되어 있어, 궤적의 모양을 유지한 채 속도만 줄이거나 크기만 줄이는 작업에서 DMP보다 압도적인 성능을 보였다. (DMP는 위상과 진폭이 섞여 있어 형태가 뭉개지는 현상이 발생함)

### 3. 실제 로봇 팔을 이용한 인간 동작 모방
- **내용**: 인간이 지휘봉(baton)을 흔드는 주기적인 동작을 모션 캡처하여 UR5e 로봇 팔이 모방하도록 하였다.
- **결과**: 인간의 복잡한 주기적 궤적을 성공적으로 학습하고 재현하였다. 특히, DMP는 2차원 토러스(Torus) 형태의 동역학을 제대로 표현하지 못해 예측 성능이 떨어졌으나(RMSE 0.094), 제안 방법은 이를 정확히 포착하여 더 낮은 RMSE(0.080)를 기록하였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 단순한 데이터 피팅이 아니라 **동역학적 구조(Phase-Amplitude)**를 잠재 공간에 강제함으로써 얻는 이점을 명확히 보여주었다. 특히, 과도 상태의 모델링을 통해 외부 교란 상황에서도 예측 가능하고 안전한 복귀 경로를 생성할 수 있다는 점은 실제 로봇 운용 관점에서 매우 중요한 기여이다. 또한, 변분 추론과 Laplace 분포를 도입하여 이상치(outlier)에 강건한 학습 체계를 구축한 점이 돋보인다.

### 한계 및 미해결 과제
논문에서 명시한 바와 같이, 현재의 시스템은 **단일 어트랙터(Single Attractor)**를 가정하고 있다. 따라서 인간이 여러 개의 서로 다른 동작(어트랙터) 사이를 전환하며 수행하는 복잡한 작업(예: 조립 작업 중 양팔의 서로 다른 움직임)은 현재 모델로 수행할 수 없다. 이를 해결하기 위해 여러 개의 네트워크를 통합하여 어트랙터 간의 전환을 학습하는 연구가 향후 필요하다.

### 비판적 해석
제안 방법이 DMP보다 우수한 성능을 보이는 이유는 잠재 공간에서 위상과 진폭을 명시적으로 분리했기 때문이다. 이는 수학적으로 매우 효율적이지만, 실제 환경에서 특성 주파수 $\omega$와 특성 지수 $\lambda$를 미리 추정(FFT 등 사용)해야 한다는 제약이 있다. 만약 주파수가 시간에 따라 급격히 변하는 비정상(non-stationary) 신호의 경우, 고정된 $\omega$를 사용하는 현재의 방식이 얼마나 유효할지에 대한 추가 논의가 필요해 보인다.

## 📌 TL;DR

본 논문은 **Phase-Amplitude Reduction** 이론을 모방 학습에 도입하여, 로봇이 인간의 주기적 동작뿐만 아니라 **초기 상태에서 목표 궤적으로 수렴하는 과도 동역학(Transient Dynamics)**까지 학습할 수 있는 프레임워크를 제안하였다. Variational Inference 기반의 인코더-디코더 학습과 Interactive Feedback 메커니즘을 통해 외부 교란에 강건하고 안전한 궤적 추종이 가능함을 입증하였으며, 이는 특히 주기적인 인간 동작을 정밀하게 모방해야 하는 로봇 제어 분야에 중요한 기여를 할 것으로 보인다.