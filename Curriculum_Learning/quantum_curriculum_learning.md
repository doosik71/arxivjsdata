# Quantum Curriculum Learning

Quoc Hoan Tran, Yasuhiro Endo, and Hirotaka Oshima (2025)

## 🧩 Problem to Solve

양자 머신러닝(Quantum Machine Learning, QML)은 실세계의 복잡한 문제를 해결하기 위해 상당한 양자 자원을 요구한다. 특히 데이터가 계층적 구조를 가질 때, 학습 복잡성과 일반화 성능의 한계가 두드러지게 나타난다. 현재의 양자 모델 최적화 과정에서는 손실 함수(loss function)의 지형에서 국소 최적해(local minima)에 빠지거나, 기울기가 소실되는 Barren Plateau 현상이 빈번하게 발생하여 학습 효율이 저하되는 문제가 있다.

본 논문의 목표는 양자 자원의 사용을 최적화하고 학습 효율을 높이기 위해, 쉬운 과제나 데이터부터 단계적으로 학습시키는 Curriculum Learning의 개념을 양자 데이터에 적용한 Quantum Curriculum Learning (Q-CurL) 프레임워크를 제안하는 것이다. 이를 통해 특히 노이즈가 존재하는 현재의 Noisy Intermediate-Scale Quantum (NISQ) 장치에서 모델의 강건성(robustness)과 일반화 능력을 향상시키고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 학습 방식과 유사하게, 단순한 작업이나 데이터를 먼저 학습시킨 후 점진적으로 복잡한 단계로 나아가는 전략을 QML에 도입하는 것이다. 이를 위해 두 가지 구체적인 접근 방식을 제안한다.

1. **Task-based Q-CurL**: 양자 데이터의 밀도 비율(density ratio)을 기반으로 메인 과제와 보조 과제 간의 유사도를 측정하여, 최적의 학습 순서를 결정하고 파라미터를 전이(transfer)하는 방식이다.
2. **Data-based Q-CurL**: 학습 과정에서 각 샘플의 난이도를 동적으로 예측하여 가중치를 조절하는 동적 학습 스케줄(dynamic learning schedule)을 도입한다. 이를 통해 노이즈가 섞인 라벨에 대한 과적합을 방지하고 일반화 성능을 높인다.

## 📎 Related Works

기존의 Curriculum Learning은 주로 고전 머신러닝 분야에서 데이터 샘플을 단순한 것부터 복잡한 순서로 배치하여 수렴 속도를 높이는 방향으로 연구되었다. 최근에는 GAN의 점진적 성장(Progressive growing)이나 교사-학생(Teacher-student) 구조 등 다양한 형태로 발전하였다.

QML 분야에서도 하이브리드 고전-양자 네트워크에서의 전이 학습(transfer learning)이나, 해밀토니안의 바닥 상태(ground state)를 찾기 위한 양자 회로 구조 탐색에서 Warm-start 전략 등이 시도된 바 있다. 그러나 본 논문은 단순히 모델의 구조를 전이하는 것을 넘어, 양자 데이터 자체의 특성을 활용해 과제의 순서와 샘플의 가중치를 체계적으로 스케줄링한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. Task-based Q-CurL

메인 과제 $T_M$을 해결하기 전, 유사한 보조 과제 $T_m$들을 먼저 해결하여 그 최적 파라미터 $\theta_{opt}^{(m)}$를 $T_M$의 초기값으로 사용하는 방식이다.

**핵심 메커니즘: Curriculum Weight ($c_{M,m}$)**
보조 과제 $T_m$이 메인 과제 $T_M$에 얼마나 기여하는지를 정량화하기 위해 밀도 비율 $r(x,y) = \frac{p^{(M)}(x,y)}{p^{(m)}(x,y)}$를 도입한다. 실제 밀도 함수를 직접 추정하는 대신, 선형 모델 $\hat{r}_{\alpha}(x,y) = \alpha^\top \phi(x,y)$를 사용하여 근사한다. 여기서 기저 함수 $\phi(x,y)$는 양자 상태의 유사도를 측정하는 Fidelity 커널의 곱으로 정의된다:
$$\phi_l(x,y) = \text{Tr}[xx^{(M)}_l] \text{Tr}[yy^{(M)}_l]$$

최종적으로 Curriculum Weight $c_{M,m}$은 다음과 같이 계산된다:
$$c_{M,m} = \frac{1}{N_m} \sum_{i=1}^{N_m} \hat{r}_{\alpha}(x^{(m)}_i, y^{(m)}_i)$$
이 값이 클수록 $T_m$을 먼저 학습하는 것이 $T_M$의 기대 리스크를 줄이는 데 효과적임을 의미한다. 이를 바탕으로 그리디(greedy) 알고리즘을 통해 $T_{i_1} \rightarrow T_{i_2} \rightarrow \dots \rightarrow T_M$ 순으로 학습 경로를 설계한다.

### 2. Data-based Q-CurL

추가적인 양자 자원 없이 손실 함수를 수정하여 샘플의 중요도를 동적으로 조절하는 방식이다.

**동적 손실 함수 (Dynamical Loss Function)**
기존의 단순 평균 손실 대신 다음과 같은 형태의 가중 손실 함수를 사용한다:
$$\hat{R}(h,w) = \frac{1}{N} \sum_{i=1}^{N} \left[ (\ell_i - \eta)e^{w_i} + \gamma w_i^2 \right]$$
여기서 $\eta$는 이전 에포크의 평균 손실값으로, 샘플의 '쉬움'과 '어려움'을 가르는 기준점이 된다. 가중치 $w_i$는 양자 자원을 사용하지 않고 고전적으로 최적화되며, 이 과정에서 Lambert W 함수가 사용된다.

**Easy vs Hard Q-CurL**

- **Easy Q-CurL ($\gamma > 0$)**: 손실이 작은(쉬운) 샘플에 더 높은 가중치를 부여한다. 노이즈 섞인 라벨이 최적화를 방해하는 상황에서 효과적이다.
- **Hard Q-CurL ($\gamma < 0$)**: 손실이 큰(어려운) 샘플을 강조한다. 복잡한 양자 특성을 추출해야 하는 상황에서 유리할 수 있다.

## 📊 Results

### 1. Unitary Learning (Task-based Q-CurL 검증)

- **작업**: spin-1/2 XY 모델의 유니터리 동역학을 학습하여 타겟 유니터리 $V$를 근사하는 작업이다.
- **측정 지표**: Hilbert-Schmidt (HS) 거리 및 테스트 손실.
- **결과**: Q-CurL을 통해 설계된 학습 순서가 무작위 순서보다 학습 수렴 속도가 빠르고 일반화 성능(테스트 손실 감소)이 뛰어남을 확인하였다. 특히 학습 데이터가 제한적인 상황($N=10$)에서도 과적합을 억제하며 더 나은 성능을 보였다.

### 2. Quantum Phase Recognition (Data-based Q-CurL 검증)

- **작업**: Cluster Ising 모델의 바닥 상태를 입력받아 물질의 상(phase)을 분류하는 QCNN(Quantum Convolutional Neural Network) 모델을 사용하였다.
- **설정**: 학습 라벨에 $p=0.2, 0.3$의 확률로 노이즈(오답 라벨)를 섞은 상황을 가정하였다.
- **결과**:
  - **강건성 향상**: 노이즈 레벨이 높아질수록 기존 학습 방식은 일반화에 실패하지만, Easy Q-CurL을 적용한 경우 테스트 손실이 낮아지고 정확도가 유의미하게 향상되었다.
  - **큐比特 수의 영향**: 큐비트 수가 증가할수록 양자 파동함수에 포함된 정보가 많아져 Q-CurL의 강건성이 더욱 뚜렷하게 나타났다.
  - **Easy vs Hard**: 라벨 노이즈가 있는 경우 Easy Q-CurL이 최적의 성능을 보였으나, 노이즈가 없는 깨끗한 데이터의 경우 충분한 학습 시간이 주어지면 Hard Q-CurL이 가장 낮은 테스트 손실을 기록하였다.

## 🧠 Insights & Discussion

본 논문은 양자 데이터를 고전적인 표현으로 변환하여 처리하는 대신, 양자 상태 그대로(directly) Curriculum Learning을 적용함으로써 양자 데이터 고유의 특성을 보존할 수 있음을 보여주었다. 이는 고전적 표현으로 변환할 때 발생하는 정보 손실과 인터페이스 설계의 어려움을 극복하는 실질적인 이점을 제공한다.

**한계 및 비판적 해석**:

- 본 연구는 기존 QML 알고리즘의 성능을 향상시키는 프레임워크를 제안한 것이며, 최신 고전 머신러닝 모델과의 직접적인 성능 비교는 수행되지 않았다.
- 하드웨어 효율적 안사츠(HE ansatz) 등을 사용하였으나, 실제 물리적 장치에서의 하드웨어 노이즈(gate noise, decoherence)가 Curriculum Learning의 효과를 어느 정도 상쇄할지에 대한 분석은 부족하다.

그럼에도 불구하고, Barren Plateau 문제를 해결하기 위해 손실 함수 자체를 점진적으로 설계하는 방향이나, 고전적으로 시뮬레이션 가능한 단순한 손실 함수부터 시작해 타겟 함수로 나아가는 전략 등 향후 확장 가능성이 매우 높다.

## 📌 TL;DR

이 논문은 양자 머신러닝의 학습 효율과 일반화 성능을 높이기 위해 **Quantum Curriculum Learning (Q-CurL)** 프레임워크를 제안한다. **Task-based approach**는 데이터 밀도 비율을 통해 최적의 과제 학습 순서를 결정하여 파라미터를 전이시키며, **Data-based approach**는 동적 손실 함수를 통해 샘플 가중치를 조절하여 라벨 노이즈에 대한 강건성을 확보한다. 이 연구는 특히 NISQ 시대의 양자 자원 제약을 극복하고, 양자 화학 및 물리 시스템의 상태 인식 성능을 높이는 데 중요한 역할을 할 것으로 기대된다.
