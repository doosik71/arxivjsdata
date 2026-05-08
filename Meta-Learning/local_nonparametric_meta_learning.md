# Local Nonparametric Meta-Learning

Wonjoon Goo, Scott Niekum (2020)

## 🧩 Problem to Solve

메타러닝의 핵심 목표는 주어진 태스크 집합에 적절한 inductive bias를 학습하여 새로운 태스크에 빠르게 적응(fast adaptation)하는 학습 규칙을 찾는 것이다. 그러나 기존의 대부분의 메타러닝 알고리즘은 고정된 크기의 표현(fixed-size representation)을 사용하는 글로벌 학습 규칙(global learning rule)을 찾으려 하며, 이는 다음과 같은 문제점을 야기한다.

첫째, 태스크 집합에 필요한 적절한 표현 능력을 사전에 선택하기 어렵기 때문에 meta-underfitting 또는 meta-overfitting이 발생하기 쉽다. 둘째, 학습 데이터와 테스트 데이터의 분포가 다른 Out-of-Distribution(OOD) 태스크에 직면했을 때 일반화 성능이 급격히 떨어진다. 본 논문은 특히 두 가지 형태의 OOD 상황에 주목한다.

- **Meta-range shift**: 태스크의 잠재 변수(예: 사인 함수의 진폭이나 주기)가 메타-학습 단계에서 관찰된 범위를 벗어나는 경우이다.
- **Meta-scale shift**: 테스트 태스크가 메타-학습 태스크보다 더 많은 비트의 표현력을 요구하는 경우로, 예를 들어 더 큰 규모의 미로(maze)를 탐색해야 하는 상황을 의미한다.

결과적으로 본 논문의 목표는 이러한 meta-range 및 meta-scale shift 상황에서도 강건하게 작동하는 메타 일반화(meta-generalization) 능력을 갖춘 새로운 비매개변수적(nonparametric) 메타러닝 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문은 **MeRLOT(Meta-Regression using Local updates for Out-of-distribution Tasks)**이라는 알고리즘을 제안한다. MeRLOT의 핵심 아이디어는 전체 함수를 한 번에 수정하는 글로벌 업데이트 대신, 각 데이터 포인트 주변에 적합화되는 **지역적 적응 규칙(local adaptation rule)**을 학습하는 것이다.

주요 설계 직관은 다음과 같다.

1. **비매개변수적 특성(Nonparametric nature)**: 데이터 포인트마다 개별적인 지역 함수를 생성하므로, 표현력이 미리 정의되지 않고 데이터의 양이나 태스크의 규모에 따라 유연하게 확장된다. 이를 통해 meta-scale shift 문제를 자연스럽게 해결할 수 있다.
2. **지역성(Locality)**: 학습 규칙이 지역적으로 적용되므로, 글로벌 사전 확률(global prior)에 의존하기보다 지역 데이터 포인트에 더 밀착된(grounded) 함수를 생성할 수 있어 meta-range shift에 강건하다.
3. **불연속성 처리**: 단일 신경망으로 불연속 함수를 표현하는 것은 어려우나, 지역 함수들의 집합으로 표현하면 각 조각(piecewise)을 별도로 학습하여 불연속성을 더 정확하게 표현할 수 있다.

## 📎 Related Works

메타러닝은 크게 최적화 기반(Optimization-based), 메트릭 기반(Metric-based), 모델 기반(Model-based)으로 나뉜다.

- **최적화 기반 (예: MAML)**: 초기 파라미터를 학습하여 적은 단계의 경사 하강법으로 적응한다. 이론적으로는 범용적이지만, 고정된 파라미터 크기로 인해 meta-scale shift에 취약하다.
- **모델 기반 (예: CNP, ANP)**: 인코더가 컨텍스트 세트를 고정 크기 벡터나 무한 차원 표현으로 인코딩하고 디코더가 이를 통해 함수를 생성한다. ANP는 Attention 메커니즘을 통해 쿼리 의존적 인코딩을 수행하여 성능을 높였으나, 여전히 글로벌한 업데이트 특성을 가진다.
- **메트릭 기반**: 유사도 메트릭이나 커널을 학습하여 Nearest-neighbor 방식으로 추론한다. 비매개변수적 특성 덕분에 meta-scale shift를 잘 처리하지만, 분류 외의 문제에 적용하기 어렵다는 한계가 있다.

MeRLOT은 ANP의 컨텍스트 기반 Attention과 MetaFun의 반복적 functional gradient updates의 장점을 결합하고, 여기에 '지역성'이라는 개념을 추가하여 기존 방식들의 한계를 극복하고자 한다.

## 🛠️ Methodology

MeRLOT은 각 컨텍스트 데이터 포인트 주변에 지역 함수들을 적합시키고, 이를 Attention 메커니즘으로 결합하여 최종 예측을 수행한다.

### 1. 전체 파이프라인 및 구성 요소

시스템은 크게 네 가지 학습 가능 구성 요소로 이루어진다: **시드 함수 생성기($\psi$), 업데이트 규칙($u$), 디코더($\Phi$), 컨텍스트 의존적 유사도 메트릭($k$)**.

1. **시드 함수 생성 (Seed Function Generation)**:
   각 컨텍스트 데이터 포인트 $c_i = (x_i, y_i)$에 대해 초기 지역 함수 $f_i(0)$를 생성한다. 함수는 functional representation $r_i^x$의 집합으로 표현되며, $\psi$는 다음과 같이 초기값을 생성한다.
   $$r_i^x(0) = \psi(c_i)(x)$$

2. **반복적 지역 업데이트 (Iterative Local Updates)**:
   Functional gradient descent를 통해 $r_i^x$를 $T$번 반복 업데이트한다.
   - 각 컨텍스트 포인트에서의 그래디언트 계산: $u_i(t) = u(x_i, y_i, r_i^x(t))$
   - 커널 $k(\cdot, \cdot)$를 이용한 그래디언트 평활화: $\Delta r_i^x(t) = \sum_{(x_i, \cdot) \in C} k(x_i, x) u_i(t)$
   - 업데이트 적용: $r_i^x(t+1) = r_i^x(t) - \alpha \Delta r_i^x(t)$

3. **최종 예측 및 결합 (Final Prediction)**:
   업데이트된 $r_i^x(T)$를 디코더 $\Phi$에 통과시켜 지역 함수 집합 $\{f_i(x) := \Phi(r_i^x(T))\}_{i=1}^m$을 얻는다. 최종 예측값은 유사도 메트릭 $k(x, x_i)$를 가중치로 하여 결합한다.
   - 점 추정 시: $\hat{y} = \sum_i k(x, x_i) f_i(x)$
   - 불확실성 추정 시 (Gaussian Mixture Model):
     $$p(y|x, \{c_i\}; \psi, u, k, \Phi) = \sum_i k(x, x_i) \mathcal{N}(\mu_i, \sigma_i)$$

### 2. 유사도 메트릭 $k$의 구현

MeRLOT은 단순한 거리 기반 커널 대신 Transformer 스타일의 Encoder-Decoder 구조를 사용하여 컨텍스트 의존적인 유사도를 계산한다. 쿼리 $x$와 컨텍스트 포인트들 사이의 dot-product attention을 통해 $k(x, x_i)$를 산출함으로써, 데이터 간의 관계를 더 유연하게 파악한다.

### 3. 학습 절차

- $\psi, u, \Phi, k$는 모두 MLP 또는 Attention 네트워크로 구현된다.
- 손실 함수는 예측값이 점 추정인 경우 $l_2$ loss를, 가우시안 분포인 경우 Negative Log-Likelihood(NLL)를 사용한다.
- 하이퍼파라미터: 반복 횟수 $T=3$, 학습률 $\alpha=0.01$.

## 📊 Results

### 1. 1D 함수 회귀 (1D Function Regression)

불연속성을 포함한 1D 함수를 통해 meta-range 및 meta-scale shift 성능을 측정하였다.

- **결과**: MeRLOT은 NLL 지표에서 모든 시나리오(Interpolation, Extrapolation, Scale shift)에서 MAML, ANP, MetaFun을 큰 차이로 압도하였다.
- **특이사항**: 특히 데이터 포인트가 증가할수록 RMSE와 NLL이 다른 모델보다 더 가파르게 감소하며, 이는 MeRLOT이 추가 데이터를 더 효율적으로 활용함을 보여준다. 또한, MAML이나 MetaFun이 뭉개뜨리는 불연속 지점을 매우 날카롭고 정확하게 예측하였다.

### 2. 미로 환경 (Maze Dynamics)

2D 미로에서 공의 forward dynamics를 예측하는 작업으로 meta-scale shift(미로 크기 증가)를 테스트하였다.

- **결과**: 컨텍스트 궤적의 수가 증가함에 따라 성능이 다소 저하되지만, ANP보다 훨씬 완만하게 저하되었다.
- **분석**: MAML은 네트워크 크기가 태스크 규모에 딱 맞을 때만 좋은 성능을 보였으나, MeRLOT은 네트워크 크기 튜닝 없이도 우수한 일반화 성능을 보였다.

### 3. Omnipush (로보틱스 벤치마크)

다양한 물체를 미는 실제 로봇 데이터셋을 통해 meta-range shift 및 OOD 성능을 검증하였다.

- **결과**: 새로운 표면(New surface), 새로운 물체(New object) 등 OOD 데이터셋에서 NLL과 RMSE 기준 SOTA(State-of-the-art)를 달성하였다. 특히 불확실성 예측(NLL)에서 매우 강한 면모를 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

MeRLOT의 성능 향상은 다음과 같은 이유에서 기인한다.

1. **계산 경로의 단축**: ANP와 달리 $\psi$를 통해 생성된 지역 함수가 직접 출력에 기여하므로, 데이터로부터 예측값까지의 경로가 짧아져 더 정확한 적응이 가능하다.
2. **비매개변수적 확장성**: 고정된 크기의 파라미터에 의존하는 MAML과 달리, 데이터 포인트 수에 따라 지역 함수의 개수가 결정되므로 meta-scale shift 상황에서 표현력이 자동으로 확장된다.
3. **지역적 바이어스의 활용**: 글로벌 함수 prior 대신 지역적 업데이트 규칙을 학습함으로써, 학습 데이터 범위를 벗어난 값(meta-range shift)에 대해서도 지역 데이터를 기반으로 더 유연하게 대응할 수 있다.

### 한계 및 논의

- **계산 비용**: 데이터 포인트 $m$에 대해 $m$개의 지역 함수를 생성하고 업데이트해야 하므로, 데이터 양이 매우 많아질 경우 계산 복잡도가 증가할 가능성이 있다.
- **시드 생성기의 중요성**: Ablation study 결과, $\psi$를 제거하면 업데이트 규칙 $u$가 지역적이 아닌 글로벌하게 작동하여 성능이 크게 떨어진다. 이는 적절한 초기화가 지역적 적응의 핵심임을 시사한다.

## 📌 TL;DR

본 논문은 메타러닝에서 발생하는 OOD 문제, 특히 태스크의 규모가 커지거나(meta-scale shift) 값의 범위가 변하는(meta-range shift) 상황을 해결하기 위해 **비매개변수적 지역 적응 알고리즘인 MeRLOT**을 제안하였다. MeRLOT은 각 데이터 포인트별로 지역 함수를 생성하고 이를 반복적으로 업데이트한 뒤 Attention으로 결합하는 방식을 취한다. 실험을 통해 1D 회귀, 미로 예측, Omnipush 로봇 데이터셋에서 기존 모델(MAML, ANP, MetaFun)보다 월등한 일반화 성능을 입증하였으며, 특히 불확실성 추정과 대규모 태스크 적응에서 강력한 이점을 가진다. 이 연구는 향후 데이터 효율적인 로봇 제어 및 능동 학습(active learning) 분야에 중요한 기여를 할 것으로 보인다.
