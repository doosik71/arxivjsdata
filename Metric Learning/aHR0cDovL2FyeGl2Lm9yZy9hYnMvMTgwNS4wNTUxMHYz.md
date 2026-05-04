# A Multilayer Framework for Online Metric Learning

Wenbin Li, Yanfang Liu, Jing Huo, Yinghuan Shi, Yang Gao, Lei Wang, and Jiebo Luo (2018/2023)

## 🧩 Problem to Solve

본 논문은 스트리밍 데이터 환경에서 효율적으로 작동하는 **Online Metric Learning (OML)**의 성능 한계를 해결하고자 한다. 기존의 OML 알고리즘들은 주로 제약 조건의 빠른 생성이나 업데이트 복잡도를 낮추는 데 집중해 왔으나, 다음과 같은 근본적인 문제점을 가지고 있다.

1. **비선형성 부족**: 대부분의 기존 OML 알고리즘은 단일 선형 메트릭 공간(single linear metric space)만을 학습한다. 이로 인해 데이터 분포가 복잡한 실제 세계의 비선형 데이터 분포를 효과적으로 캡처하지 못하는 한계가 있다.
2. **데이터 효율성 문제**: 모든 라벨링된 스트리밍 데이터를 관찰할 수 없는 제한적인 상황에서 학습 능력이 저하되는 문제가 발생한다.

따라서 본 연구의 목표는 비선형 유사성을 포착할 수 있는 **다층 구조의 온라인 메트릭 학습 프레임워크**를 설계하여, 복잡한 데이터 분포에서도 높은 분류 및 검색 성능을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 OML 알고리즘 자체를 하나의 **Metric Layer**로 정의하고, 이를 비선형 활성화 함수(Nonlinear Layer)와 함께 계층적으로 쌓아 올리는 것이다.

- **MLOML (Multi-Layer Online Metric Learning) 프레임워크 제안**: 여러 개의 메트릭 레이어와 비선형 레이어(ReLU, Sigmoid, tanh 등)를 적층하여, 이전 레이어에서 학습된 특징 공간 위에 새로운 메트릭을 순차적으로 학습함으로써 특징 표현을 점진적으로 정교화(progressively refined)한다.
- **MOML (Mahalanobis-based Online Metric Learning) 알고리즘 개발**: MLOML의 기본 구성 요소가 될 효율적인 메트릭 레이어를 위해, Passive-Aggressive 전략과 One-pass triplet construction 전략을 결합한 새로운 Mahalanobis 기반 OML 알고리즘을 제안하였다.
- **새로운 학습 전략 도입**: 전방 전파(Forward Propagation, FP)를 통한 즉각적인 레이어 업데이트와 후방 전파(Backward Propagation, BP)를 통한 전역적 미세 조정을 결합하여 학습 효율과 성능을 동시에 높였다.

## 📎 Related Works

메트릭 학습은 크게 두 가지 방향으로 발전해 왔다.

1. **Mahalanobis distance 기반 방법**: 대칭 양정치(Symmetric Positive Semi-Definite, PSD) 제약 조건을 가진 거리 행렬을 학습하며, POLA, LEGO, RDML 등이 이에 해당한다.
2. **Bilinear similarity 기반 방법**: PSD 제약 없이 쌍선형 유사도 행렬을 학습하며, OASIS, SOML, OMKS 등이 대표적이다.

**기존 연구의 한계 및 차별점**:
기존의 OML 연구들은 주로 계산 복잡도를 줄이거나 빠른 수렴에 집중했으나, 단일 선형 공간만을 학습한다는 점이 가장 큰 한계였다. 본 논문은 이를 해결하기 위해 딥러닝의 구조적 아이디어를 OML에 접목하여, 메트릭 학습 알고리즘을 레이어화함으로써 비선형 특징 공간을 탐색할 수 있게 하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

MLOML은 $\text{OML Layer} \rightarrow \text{Nonlinear Layer}$의 쌍이 $n$번 반복되는 구조이다. 각 메트릭 레이어는 독립적인 OML 알고리즘으로 동작하며, 그 결과로 나온 특징 벡터는 비선형 함수(ReLU, Sigmoid, tanh 중 하나)를 거쳐 다음 레이어로 전달된다.

### 2. MOML (Mahalanobis-based Online Metric Learning)

MLOML의 개별 레이어로 사용되는 MOML은 triplet 제약 조건 $\langle x, x^+, x^- \rangle$ (여기서 $x$는 $x^+$와 유사하고 $x^-$와는 서로 다름)을 기반으로 학습한다.

**목표 함수**:
MOML은 각 타임 스텝 $t$에서 다음과 같은 convex objective function을 최소화하는 Mahalanobis 행렬 $M$을 찾는다.
$$ \Gamma = \arg \min_{M \succeq 0} \frac{1}{2} \|M - M_{t-1}\|_F^2 + \gamma [1 + \text{Tr}(MA_t)]_+ $$
여기서 $\| \cdot \|_F$는 Frobenius norm이며, $[z]_+ = \max(0, z)$는 hinge loss를 의미한다. $A_t$는 다음과 같이 정의된다.
$$ A_t = (x_t - x_p)(x_t - x_p)^\top - (x_t - x_q)(x_t - x_q)^\top $$

**업데이트 규칙**:
위 식의 gradient를 통해 도출된 closed-form solution에 따라, $M$은 다음과 같이 업데이트된다.
$$ M_t = \begin{cases} M_{t-1} - \gamma A_t & \text{if } [z]_+ > 0 \\ M_{t-1} & \text{if } [z]_+ = 0 \end{cases} $$
이 업데이트 과정의 시간 복잡도는 $O(d^2)$로 매우 효율적이다.

### 3. MLOML의 학습 절차 및 손실 함수

전체 프레임워크의 최적화를 위해 다음과 같은 통합 손실 함수 $\Gamma$를 정의하여 사용한다.
$$ \Gamma = \frac{1}{2} \Gamma_{\text{triplet}} + \sum_{i=1}^{n} w_i \Gamma_i^{\text{local}} + \frac{\lambda}{2} \sum_{i=1}^{n} \|L_i\|_F^2 $$

- $\Gamma_{\text{triplet}}$: 최종 출력 레이어의 triplet loss.
- $\Gamma_i^{\text{local}}$: $i$번째 메트릭 레이어의 국소 손실.
- $\|L_i\|_F^2$: 파라미터 행렬 $L_i$ (여기서 $M_i = L_i^\top L_i$)의 정규화 항.

### 4. 전파 전략 (Propagation Strategies)

- **MLOML-FP**: 전방 전파 과정에서 각 레이어가 자신의 국소 손실을 바탕으로 즉각 업데이트된다. 새로운 특징 공간을 순차적으로 탐색하는 효과가 있다.
- **MLOML-BP**: 최종 손실을 바탕으로 후방 전파를 통해 모든 레이어의 파라미터를 업데이트한다.
- **MLOML-FBP**: FP와 BP를 동시에 사용하여, FP로 공간을 탐색하고 BP로 이를 보정(amend)한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: UCI Machine Learning Repository에서 수집한 12개의 벡터화된 데이터셋을 사용하였다.
- **비교 대상**: RDML, LEGO, OASIS, OPML, SLMOML 및 단일 레이어 MOML, 그리고 오프라인 방식인 LMNN, KISSME 등을 비교군으로 설정하였다.
- **평가 지표**: Error Rate (오류율)를 사용하였으며, $k\text{-NN} (k=5)$ 분류기를 통해 최종 성능을 측정하였다.

### 2. 주요 결과

- **정량적 성능**: MLOML(특히 ReLU를 사용한 MLOML-r)은 대부분의 데이터셋에서 기존의 단일 레이어 OML 알고리즘보다 낮은 오류율을 기록하였다. t-test 결과, 많은 경우에서 통계적으로 유의미하게 성능이 향상되었음을 확인하였다.
- **전파 전략 비교**: MLOML-FP가 MLOML-BP보다 우수한 성능을 보였는데, 이는 BP의 경우 gradient vanishing 문제가 발생할 수 있는 반면, FP는 각 레이어가 독립적으로 최적의 메트릭을 학습하며 전진하기 때문이다. MLOML-FBP는 일부 데이터셋에서 최고의 성능을 보였으나 계산 비용이 증가한다.
- **점진적 학습 능력**: 레이어 수가 증가할수록(1L $\rightarrow$ 3L $\rightarrow$ 5L) 오류율이 점진적으로 감소하는 경향이 확인되었으며, 이는 시각화 결과(PCA 2D projection)를 통해 클래스 간 분리도가 점점 높아지는 것으로 증명되었다.
- **데이터 재사용 능력**: 동일한 데이터 스캔 횟수(epochs) 대비 MLOML이 MOML보다 훨씬 빠르게 수렴하고 낮은 오류율에 도달하여, 다층 구조가 데이터로부터 더 많은 정보를 추출할 수 있음을 보여주었다.

## 🧠 Insights & Discussion

### 강점

- **비선형성 확보**: OML에 다층 구조와 비선형 활성화 함수를 도입함으로써, 기존 선형 OML이 해결하지 못한 복잡한 데이터 분포 문제를 효과적으로 해결하였다.
- **효율적인 학습**: 전방 전파(FP)만으로도 충분히 강력한 성능을 낼 수 있음을 보였으며, 이는 딥러닝의 일반적인 학습 방식과는 다른 OML만의 효율적인 학습 가능성을 제시한다.
- **이론적 뒷받침**: MOML의 convex objective function과 regret bound를 통해 수렴성을 이론적으로 증명하였다.

### 한계 및 비판적 해석

- **고차원 데이터 처리 문제**: MOML이 full matrix $M$을 학습하기 때문에, 특징 차원 $d$가 매우 커질 경우 계산 및 메모리 비용이 급격히 증가한다. 논문에서도 이를 언급하며 향후 diagonal matrix 학습이나 online feature selection이 필요함을 시사한다.
- **하이퍼파라미터 의존성**: 레이어의 최적 개수나 $\gamma, \lambda$ 등의 파라미터가 작업마다 다를 수 있으나, 이를 자동으로 결정하는 메커니즘은 제시되지 않았고 cross-validation에 의존하고 있다.

## 📌 TL;DR

본 논문은 기존 온라인 메트릭 학습(OML)의 선형적 한계를 극복하기 위해, OML 알고리즘을 레이어 형태로 쌓아 올린 **MLOML (Multi-Layer Online Metric Learning)** 프레임워크를 제안한다. 특히 효율적인 기본 레이어인 **MOML**을 설계하고, 전방 전파(FP) 기반의 즉각적인 업데이트 전략을 통해 비선형 특징 공간을 점진적으로 학습한다. 실험 결과, MLOML은 기존 OML 및 일부 배치 학습 방식보다 우수한 분류 성능을 보였으며, 층이 깊어질수록 특징 표현 능력이 향상됨을 입증하였다. 이 연구는 향후 실시간 스트리밍 데이터의 복잡한 패턴 인식 및 고도화된 메트릭 학습 연구에 중요한 기초가 될 것으로 보인다.
