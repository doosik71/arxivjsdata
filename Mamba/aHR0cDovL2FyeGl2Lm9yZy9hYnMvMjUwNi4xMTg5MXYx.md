# Understanding Input Selectivity in Mamba: Impact on Approximation Power, Memorization, and Associative Recall Capacity

Ningyuan Huang, Miguel Sarabia, Abhinav M uma Moudgil, Pau Rodríguez, Luca Zappella, Federico Danieli (2025)

## 🧩 Problem to Solve

본 논문은 최근 Transformer의 대안으로 주목받고 있는 State-Space Models(SSMs), 특히 Mamba 아키텍처의 핵심 기제인 **입력 선택성(input selectivity)**의 역할을 이론적으로 규명하고자 한다. Mamba는 기존의 SSM 선행 모델들과 달리 S6 레이어를 통해 입력에 따라 매개변수가 변하는 선택적 메커니즘을 도입하였으며, 여기에 합성곱(convolution)과 게이팅(gating)을 결합하여 성능을 향상시켰다.

그러나 Mamba가 왜 기존 SSM보다 뛰어난 성능을 보이는지, 특히 입력 선택성이 함수 근사 능력, 장기 기억(long-term memorization), 그리고 연상 회상(associative recall) 능력에 구체적으로 어떤 영향을 미치는지에 대한 구조적 설명은 여전히 부족한 상태이다. 따라서 본 연구의 목표는 S6 레이어의 표현력(expressivity)이 실제 작업 성능으로 어떻게 이어지는지 분석하고, Mamba 블록 내의 다양한 구성 요소들이 상호작용하여 구체적인 태스크를 해결하는 메커니즘을 밝히는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba의 입력 선택성이 제공하는 기능적 이점을 수학적으로 증명하고, 이를 통해 모델의 작동 원리를 메커니즘적으로 이해했다는 점에 있다. 주요 기여 사항은 다음과 같다.

1.  **Haar Wavelet 표현 능력 증명**: S6 레이어가 Haar wavelet으로의 투영(projection)을 표현할 수 있음을 증명하였다. 이는 불연속적인 함수를 근사할 때 S4D와 같은 기존 SSM보다 훨씬 효율적임을 의미하며, 실제 데이터에서 빈번하게 발생하는 불연속 신호를 처리하는 데 유리함을 보였다.
2.  **메모리 감쇠의 동적 제어**: 민감도 분석(sensitivity analysis)을 통해 S6 레이어 역시 기본적으로는 지수적인 메모리 감쇠를 겪지만, 입력 선택성을 통해 시간의 흐름을 동적으로 조절함으로써 이러한 감쇠를 상쇄할 수 있는 메커니즘을 제시하였다.
3.  **연상 회상(Associative Recall)의 분석적 솔루션**: MQAR(Multiple-Query Associative Recall) 태스크를 해결하기 위한 1계층 Mamba, Mamba-2, S4D 모델의 분석적 구조를 제안하였다. 특히 Mamba-2가 더 적은 파라미터로 효율적인 솔루션을 찾을 수 있음을 보였으며, S6의 선택성이 없더라도 합성곱과 게이팅만으로 MQAR를 해결할 수 있다는 점을 밝혀 아키텍처 내 구성 요소들의 역할을 재조명하였다.
4.  **Mamba-$\Delta^\top$ 변형 제안**: Induction Heads 태스크의 성능을 높이기 위해, 입력 의존성을 임베딩 차원이 아닌 상태(state) 차원에 적용하는 $\text{Mamba-}\Delta^\top$ 구조를 제안하고 그 효과를 입증하였다.

## 📎 Related Works

본 논문은 크게 세 가지 관련 연구 분야를 다룬다.

**State-Space Models (SSMs)**: 기존의 HiPPO 기반 SSM과 S4, S4D 모델들은 선형 시불변(Linear Time-Invariant) 시스템으로, 계산 효율성은 높으나 입력에 따라 동적으로 정보를 처리하는 능력이 부족했다. Mamba는 이를 입력 의존적 매개변수로 해결하여 Transformer에 필적하는 성능을 보였으며, Mamba-2는 상태 행렬을 단순화하여 효율성을 더욱 높였다.

**SSM의 표현력(Expressivity)**: 기존 연구들은 SSM이 어떤 형식 언어를 인식할 수 있는지(Formal Language Theory) 또는 어떤 함수 클래스를 근사할 수 있는지(Approximation Theory)를 연구했다. 일부 연구는 SSM이 지수적 메모리 감쇠 문제를 겪으며, MLP를 추가해야만 보편적 근사기(universal approximator)가 될 수 있음을 지적했다. 본 논문은 여기서 더 나아가 Mamba의 S6 레이어가 실제 구현 가능한 Haar wavelet 투영을 통해 불연속 함수를 효율적으로 근사함을 보임으로써 차별점을 둔다.

**연상 회상 능력(Associative Recall)**: Induction Heads나 MQAR와 같은 태스크는 모델이 메모리에서 특정 정보를 정확히 추출하는 능력을 측정한다. 기존의 Transformer 기반 솔루션들은 모델 크기가 시퀀스 길이에 비례해 커져야 하는 한계가 있었다. 본 논문은 1계층 Mamba 모델이 시퀀스 길이와 독립적인 모델 크기로도 이러한 태스크를 정확히 해결할 수 있음을 이론적으로 증명하였다.

## 🛠️ Methodology

### 1. S6 레이어의 함수 근사 능력 분석

본 논문은 Linear RNN을 연속 시간 시스템으로 해석하여, Mamba의 기저 함수(basis function)를 다음과 같이 정의한다.

$$h^M_n(t) = \int_{0}^{t} e^{-\lambda_n \int_{s}^{t} \Delta(x_r) dr} B_n x(s) ds$$

여기서 $\Delta(x)$는 입력 의존적인 이산화 단계(discretization step)이다. S4D는 $\Delta(x) \equiv 1$인 특수 사례로, 오직 지수 함수 형태의 기저 함수만을 가진다. 반면, Mamba는 $\Delta(x)$를 조절함으로써 **Heaviside 함수**를 근사할 수 있으며, 이를 선형 결합하여 **Haar wavelet** $\psi_{j,k}$를 구현할 수 있다. 

이러한 특성 덕분에 Mamba는 불연속적인 피스와이즈 상수 함수(piecewise-constant function)를 근사할 때, $O(N^{-1})$의 오차를 보이는 S4D에 비해 $O(2^{-N/3m})$라는 압도적으로 빠른 수렴 속도를 보인다.

### 2. 메모리 민감도 분석 (Sensitivity Analysis)

모델이 과거의 입력 $x_j$를 얼마나 기억하는지는 현재 상태 $h_t$의 $x_j$에 대한 미분 값인 민감도로 측정할 수 있다.

$$\left| \frac{\partial h^M_t}{\partial x_j} \right| \approx \tilde{c} e^{-\lambda_n \sum_{r=j+1}^{t} \Delta(x_r)}$$

분석 결과, Mamba 역시 지수적 감쇠를 겪지만, 입력 선택성을 통해 $\Delta(x_r) \to 0$으로 만들어 지수 항을 1에 가깝게 유지함으로써 메모리 감쇠를 동적으로 억제(freezing time)할 수 있음을 보였다.

### 3. MQAR 태스크를 위한 아키텍처 구성

MQAR 태스크(키-값 쌍을 저장하고 쿼리에 따라 값을 회상하는 작업)를 해결하기 위해 1계층 Mamba 모델의 파이프라인을 다음과 같이 설계한다.

1.  **Embedding**: 키($k$)와 값($v$)을 서로 직교하는 공간에 임베딩한다.
2.  **Convolution**: 크기 2의 비선형 커널을 사용하여 $(k_i, v_j)$ 쌍을 추출하고, 무의미한 쌍(예: $v, v$)은 제거한다.
3.  **SSM (S6)**: 상태 행렬 $h \in \mathbb{R}^{d \times N}$의 각 열을 특정 키에 할당하고, 해당 열에 연관된 값의 임베딩을 저장한다. 이때 $B_t$와 $C_t$가 입력에 따라 선택적으로 작동하여 올바른 열에 쓰고 읽는다.
4.  **Gating & Output**: 게이팅 레이어를 통해 필요한 정보를 필터링하고, 최종 선형 층에서 값을 분류한다.

### 4. Mamba-$\Delta^\top$ 및 Induction Heads 해결

Induction Heads 태스크(최근 발생한 동일 토큰 이후의 토큰을 예측)를 해결하기 위해, 저자들은 $\Delta$의 적용 방식을 변경한 **Mamba-$\Delta^\top$**를 제안한다. 기존 Mamba는 $\Delta$가 임베딩 차원 $d$를 따라 적용되지만, $\text{Mamba-}\Delta^\top$는 이를 상태 차원 $N$을 따라 적용한다.

$$h_t = e^{\Lambda \odot (1_d \otimes \Delta(\hat{x}_t))} \odot h_{t-1} + \hat{x}_t \otimes B_t$$

이 구조는 특정 키에 해당하는 메모리 열만 선택적으로 삭제($\Delta \to \infty$)하고 나머지는 유지($\Delta \to 0$)할 수 있게 하여, 최신 정보만을 효율적으로 갱신하는 메커니즘을 구현한다.

## 📊 Results

### 1. KEEPn-TH 태스크 (함수 근사 및 메모리 검증)
시퀀스의 $n$번째 토큰을 기억하는 태스크에서, Positional Encoding(PE)을 추가한 Mamba는 100% 정확도를 보였다. 반면 S4D는 시퀀스 길이가 길어질수록 정확도가 급격히 하락하였으며, Mamba에서 PE를 제거했을 때도 성능이 S4D 수준으로 떨어졌다. 이는 Mamba의 입력 선택성이 PE와 결합했을 때 불연속적인 위치 정보를 정확히 포착할 수 있음을 시사한다.

### 2. MQAR 태스크 (연상 회상 능력 및 효율성)
다양한 모델 크기($d, N$)에 따른 MQAR 정확도를 측정한 결과, 이론적으로 도출한 모델 크기 하한선(theoretical bound)이 실제 임계값과 매우 유사함을 확인하였다. 
- **효율성**: $\text{Mamba-2} > \text{Mamba} > \text{S4D}$ 순으로 파라미터 효율성이 높았다. 특히 Mamba-2는 독립적인 세 개의 합성곱 층을 사용하여 가장 작은 모델 크기로도 100% 정확도를 달성하였다.

### 3. Induction Heads 태스크 ($\text{Mamba-}\Delta^\top$ 검증)
표준 Mamba와 $\text{Mamba-}\Delta^\top$를 비교한 결과, 모든 모델 크기에서 $\text{Mamba-}\Delta^\top$가 동등하거나 더 우수한 성능을 보였다. 특히 상태 차원에서의 선택적 삭제 메커니즘이 장기 기억 유지와 최신 정보 갱신이라는 상충하는 목표를 동시에 달성하게 함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 우수성이 단순히 '입력 의존성'이라는 추상적인 개념이 아니라, **기저 함수의 확장(Wavelets), 메모리 감쇠의 동적 제어, 그리고 상태 행렬의 구조적 활용**이라는 구체적인 메커니즘에서 기인함을 밝혔다.

특히 흥미로운 지점은 MQAR 분석에서 나타났다. 1계층 Mamba가 S6의 선택성 없이 S4D 믹서만으로도 MQAR를 해결할 수 있다는 결과는, Mamba 아키텍처에서 **합성곱(convolution)과 게이팅(gating)이 단순한 보조 도구가 아니라 연상 회상 능력을 구현하는 핵심 요소**임을 보여준다. 즉, S6의 선택성은 이 과정을 더 효율적으로(더 작은 모델 크기로) 만들어주는 역할을 한다.

또한, $\text{Mamba-}\Delta^\top$의 제안은 SSM의 상태 업데이트 방식을 어떻게 변경하느냐에 따라 모델의 인지 능력을 획기적으로 개선할 수 있음을 시사한다. 이는 향후 SSM 설계 시 입력 의존성을 어디에 배치할 것인가에 대한 중요한 가이드라인을 제공한다.

## 📌 TL;DR

이 논문은 Mamba의 S6 레이어가 **Haar wavelet을 표현**하여 불연속 신호를 잘 처리하고, **입력 선택성을 통해 메모리 감쇠를 동적으로 제어**함을 이론적으로 증명하였다. 또한, Mamba가 MQAR 및 Induction Heads와 같은 연상 회상 태스크를 해결하는 메커니즘을 분석하여, 합성곱과 게이팅의 중요성을 재확인하고 성능을 개선한 $\text{Mamba-}\Delta^\top$ 변형을 제안하였다. 이 연구는 SSM의 구조적 이해를 높여 향후 더 효율적인 시퀀스 모델링 아키텍처 설계에 기여할 가능성이 크다.