# Reservoir memory machines

Benjamin Paaßen and Alexander Schulz (2020)

## 🧩 Problem to Solve

본 논문은 신경망의 유연성과 튜링 머신(Turing machine)의 계산 능력을 결합한 Neural Turing Machines(NTM)가 가진 치명적인 단점인 **학습의 어려움(hard to train)**을 해결하고자 한다. 

일반적인 신경망 모델들은 간섭 없이 매우 장기적인 기억을 유지해야 하는 작업에서 어려움을 겪으며, 이를 해결하기 위해 메모리와 계산을 분리하는 구조가 필요하다. NTM은 외부 메모리를 통해 이를 구현했으나, 학습 과정이 매우 까다로워 실제 적용에 한계가 있었다. 따라서 본 연구의 목표는 NTM이 해결할 수 있는 벤치마크 과제들을 수행할 수 있으면서도, 학습 속도가 훨씬 빠르고 효율적인 새로운 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 NTM의 복잡한 학습 기반 컨트롤러를 **Echo State Network(ESN)**로 대체하는 것이다. 

구체적으로, 모델 전체를 역전파(Backpropagation)로 학습시키는 대신, 외부 메모리에 접근하는 Read/Write Head의 컨트롤러와 최종 출력 매핑만을 학습시킨다. 이때 학습 방법으로 정교한 최적화 알고리즘 대신 **Dynamic Time Warping(DTW)** 기반의 정렬(alignment) 알고리즘과 **선형 회귀(Linear Regression)**만을 사용한다. 이를 통해 학습 시간을 획기적으로 단축하면서도, ESN의 고유한 한계인 '뉴런 수에 비례하는 단기 기억 용량' 문제를 외부 메모리 추가를 통해 극복하여 임의의 길이만큼 데이터를 저장할 수 있는 Reservoir Memory Machine(RMM)을 제안한다.

## 📎 Related Works

본 논문에서 다루는 관련 연구와 그 한계는 다음과 같다.

1.  **Echo State Networks (ESN):** 입력 데이터를 비선형적으로 전처리하는 고정된 가중치의 Reservoir를 사용하며, 출력층의 가중치만을 학습시키는 효율적인 모델이다. 그러나 메모리 호라이즌(memory horizon)이 Reservoir 내의 뉴런 수에 의해 제한되므로, 매우 긴 시간 동안 정보를 유지해야 하는 작업에는 부적합하다.
2.  **Neural Turing Machines (NTM):** 순환 신경망(RNN)에 명시적인 외부 메모리를 추가하여 읽기/쓰기 접근을 가능하게 한 모델이다. 이론적으로 강력한 계산 능력을 갖추고 있으나, 학습이 매우 어렵다는 점이 주요 한계로 지적된다.

RMM은 ESN의 빠른 학습 속도와 NTM의 외부 메모리 구조를 결합함으로써, 기존 ESN의 메모리 용량 한계를 극복하고 NTM의 학습 난이도 문제를 해결한다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
RMM의 상태는 $(\vec{h}_t, M_t, k_t, l_t)$라는 4가지 요소로 정의된다. 여기서 $\vec{h}_t$는 Reservoir의 활성화 값, $M_t$는 크기가 $K$인 외부 메모리 상태, $k_t$와 $l_t$는 각각 Write head와 Read head의 현재 위치를 나타낸다.

### 2. 주요 구성 요소 및 동작 원리

**A. Reservoir (Echo State Network)**
입력 $\vec{x}_t$에 대해 다음과 같이 상태 $\vec{h}_t$를 계산한다.
$$\vec{h}_t = \tanh(U \cdot \vec{x}_t + W \cdot \vec{h}_{t-1})$$
여기서 $U$와 $W$는 학습되지 않는 고정된 가중치이며, 본 논문에서는 결정론적인 'cycle reservoir with jumps' 스킴을 사용하여 초기화한다.

**B. Write Head (쓰기 헤드)**
입력과 Reservoir 상태를 이용해 쓰기 제어 값 $c^w_t$를 계산한다.
$$c^w_t = \vec{u}_w \cdot \vec{x}_t + \vec{v}_w \cdot \vec{h}_t$$
만약 $c^w_t > 0$이면, 현재 입력 $\vec{x}_t$를 메모리의 $k$번째 위치($\vec{m}_{t,k}$)에 쓰고, 위치 인덱스를 $k_t \leftarrow k_{t-1} + 1$로 증가시킨다. (범위를 초과하면 1로 리셋된다.)

**C. Read Head (읽기 헤드)**
읽기 제어 벡터 $\vec{c}^r_t$를 다음과 같이 계산한다.
$$\vec{c}^r_t = U_r \cdot \vec{x}_t + V_r \cdot \vec{h}_t$$
$\vec{c}^r_t$의 세 성분 중 최댓값에 따라 Read head의 동작이 결정된다.
- 첫 번째 성분이 최대: 현재 위치 유지 ($l_t \leftarrow l_{t-1}$)
- 두 번째 성분이 최대: 위치 증가 ($l_t \leftarrow l_{t-1} + 1$)
- 세 번째 성분이 최대: 위치 리셋 ($l_t \leftarrow 1$)
이후 $l_t$번째 행의 메모리 값을 읽어 $\vec{r}_t$로 설정한다.

**D. 최종 출력**
최종 출력 $\vec{y}_t$는 Reservoir 상태와 메모리 읽기 값의 선형 조합으로 생성된다.
$$\vec{y}_t = V \cdot \vec{h}_t + R \cdot \vec{r}_t$$

### 3. 학습 절차 (Training)
RMM은 교대 최적화(alternating optimization) 알고리즘을 통해 학습하며, 모든 과정은 선형 회귀를 기반으로 한다.

1.  **Write Head 학습:** 출력 $\vec{y}_t$와 가장 유사한 입력 $\vec{x}_\tau$를 찾아 이상적인 쓰기 시퀀스를 생성하고, 이를 타겟으로 $\vec{u}_w, \vec{v}_w$를 선형 회귀로 학습시킨다.
2.  **Read Head 학습:** DTW 변형 알고리즘을 사용하여 메모리 상태 $M_t$와 출력 $\vec{y}_t$ 사이의 최적 경로(읽기 위치 이동)를 찾는다. 이 경로를 타겟으로 $U_r, V_r$를 선형 회귀로 학습시킨다.
3.  **출력 가중치 학습:** 결정된 $\vec{r}_t$를 사용하여 $V$와 $R$을 선형 회귀로 학습시킨다.
4.  **반복:** $R$의 변화가 이전 단계의 최적 정렬에 영향을 주므로, 손실이 증가하거나 수렴할 때까지 위 과정을 반복한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** Latch task (스파이크에 따른 0/1 반전), Copy task (입력 시퀀스 복제), Repeat copy task (시퀀스 반복 복제)
- **비교 대상:** 표준 ESN, ESGRU (GRU 구조에 고정 가중치 적용), NTM
- **지표:** Root Mean Square Error (RMSE)

### 2. 정량적 결과
- **정확도:** RMM은 모든 과제에서 매우 낮은 RMSE를 기록하며 과제를 성공적으로 해결했다. 반면, ESN과 ESGRU는 유의미하게 높은 오차를 보여 과제를 해결하지 못했다. NTM은 모든 과제에서 오차가 0에 수렴하는 완벽한 성능을 보였다.
- **학습 속도:** RMM의 학습 시간은 ESN보다 약 15배 느리지만, 여전히 1초 미만으로 매우 빠르다. 반면 NTM은 동일한 Copy task 학습에 30분 이상이 소요되었다.

### 3. 분석 및 외삽(Extrapolation) 성능
Latch task에서 학습 데이터(길이 200)보다 훨씬 긴 시퀀스(길이 1700)에 대해 테스트한 결과, RMM은 완벽하게 외삽하는 능력을 보였다. 분석 결과, 모델은 첫 번째 스파이크 때만 메모리에 1을 쓰고, 이후 스파이크가 발생할 때마다 읽기 헤드의 위치를 변경하여 출력을 생성하는 전략을 사용함이 확인되었다.

## 🧠 Insights & Discussion

**강점 및 성과:**
RMM은 ESN의 극도로 빠른 학습 방식과 NTM의 외부 메모리 능력을 성공적으로 결합하였다. 특히 역전파 없이 선형 회귀와 DTW만으로 메모리 접근 제어기를 학습시켰다는 점이 매우 효율적이다. 또한, 단순한 위치 기반 접근만으로도 Latch task에서 뛰어난 일반화 및 외삽 능력을 보여주었다.

**한계 및 향후 과제:**
본 모델은 메모리의 특정 위치를 참조하는 '위치 기반 주소 지정(location-based addressing)' 방식만을 사용한다. 따라서 NTM의 핵심 기능 중 하나인 '내용 기반 주소 지정(content-based addressing)'이 필요한 더 복잡한 과제는 해결할 수 없다. 저자들은 이를 향후 연구 과제로 제시하였다. 또한, RMM이 이론적으로 ESN보다 엄격하게 더 강력한 모델인지에 대한 형식적 증명이 아직 부족하다는 점이 한계로 언급된다.

## 📌 TL;DR

본 논문은 NTM의 강력한 메모리 기능과 ESN의 빠른 학습 속도를 결합한 **Reservoir Memory Machine(RMM)**을 제안한다. RMM은 외부 메모리와 Read/Write Head를 갖춘 ESN 구조로, 복잡한 역전파 대신 **DTW 정렬과 선형 회귀**를 통해 학습된다. 실험 결과, RMM은 NTM이 해결하는 벤치마크 과제들을 성공적으로 수행하면서도 학습 시간을 수십 분에서 1초 미만으로 획기적으로 단축시켰으며, 특히 장기 기억이 필요한 작업에서 뛰어난 외삽 성능을 보였다. 이는 실시간 학습이 중요하거나 계산 자원이 제한된 환경에서 장기 메모리 시스템을 구축하는 데 중요한 방향성을 제시한다.