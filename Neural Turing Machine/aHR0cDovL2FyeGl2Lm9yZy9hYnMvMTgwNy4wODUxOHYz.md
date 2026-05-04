# Implementing Neural Turing Machines

Mark Collier and Joeran Beel (2018)

## 🧩 Problem to Solve

본 논문은 Neural Turing Machines (NTMs)의 실제 구현 과정에서 발생하는 불안정성과 성능 재현의 어려움을 해결하고자 한다. NTM은 외부 메모리 유닛을 도입하여 계산과 메모리를 분리한 Memory Augmented Neural Networks (MANNs)의 일종으로, 복잡한 메모리 접근 패턴이 필요한 시퀀스 학습 작업에서 LSTM보다 우수한 성능을 보인다. 

그러나 원 논문에서 소스 코드를 제공하지 않았기 때문에, 이후 등장한 여러 오픈소스 구현체들은 학습 도중 그래디언트가 $\text{NaN}$이 되어 학습이 실패하거나, 보고된 성능에 도달하지 못하는 수렴 속도 저하 문제를 겪었다. 따라서 본 연구의 목표는 NTM을 성공적으로 구현하기 위한 핵심 설정들을 정의하고, 특히 메모리 초기화 방식이 성능에 미치는 영향을 분석하여 안정적이고 성능이 검증된 오픈소스 구현체를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 NTM의 성공적인 학습을 결정짓는 결정적인 요소가 **메모리 내용의 초기화 방식(Memory Contents Initialization Scheme)**임을 밝혀낸 것이다. 저자들은 다양한 초기화 전략을 실험적으로 비교하여, 메모리를 작은 상수 값으로 초기화하는 방식이 다른 방식들에 비해 수렴 속도를 획기적으로 높이며 학습의 안정성을 보장한다는 점을 증명하였다. 또한, 학습 가능한 바이어스 벡터를 이용한 읽기/쓰기 헤드의 초기화 및 적절한 비선형 활성화 함수 설정을 통해 $\text{NaN}$ 문제 없이 안정적으로 작동하는 TensorFlow 기반의 NTM 구현체를 제시하였다.

## 📎 Related Works

NTM은 외부 메모리를 사용하는 MANNs의 대표적인 모델이다. 기존의 gated recurrent neural networks인 LSTM은 내부 벡터 형태의 메모리를 유지하지만, NTM은 이를 외부 행렬 형태로 분리하여 더 큰 메모리 용량과 복잡한 주소 지정(Addressing)이 가능하다.

본 논문은 기존의 오픈소스 NTM 구현체들이 가진 한계를 지적한다. 많은 구현체들이 메모리를 무작위로 초기화(Random Initialization)하여 학습 속도가 느리거나 불안정하며, 일부는 최적화 과정에서 수치적 불안정성으로 인해 학습이 중단되는 문제를 보인다. 또한, NTM의 후속 모델인 Differentiable Neural Computer (DNC)가 존재하지만, 표준적인 NTM의 안정적인 구현 기준이 부재하여 연구자들이 이를 새로운 문제에 적용하는 데 어려움이 있었다는 점을 언급한다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인
NTM은 컨트롤러(Controller)와 외부 메모리 유닛(External Memory Unit)으로 구성된다. 컨트롤러는 피드포워드 네트워크 또는 RNN(본 논문에서는 LSTM 사용)일 수 있으며, 메모리는 $N \times W$ 크기의 행렬 $M$으로 정의된다. 여기서 $N$은 메모리 위치의 수, $W$는 각 셀의 차원이다.

### 메모리 주소 지정 및 읽기/쓰기 절차
컨트롤러는 각 타임스텝 $t$에서 읽기 및 쓰기 헤드를 통해 메모리에 접근하며, 이는 Soft Attentional Mechanism을 통해 이루어진다.

1. **Content-based Addressing**:
   컨트롤러가 생성한 키 벡터 $k_t$와 메모리 $M_t$ 사이의 유사도를 계산하여 가중치 $w_t^c$를 생성한다.
   $$w_t^c(i) \leftarrow \frac{\exp(\beta_t K[k_t, M_t(i)])}{\sum_{j=0}^{N-1} \exp(\beta_t K[k_t, M_t(j)])}$$
   여기서 $\beta_t$는 정밀도를 조절하는 파라미터이며, $K$는 코사인 유사도(Cosine Similarity)를 사용한다.
   $$K[u,v] = \frac{u \cdot v}{\|u\| \cdot \|v\|}$$

2. **Location-based Addressing (Convolutional Shift)**:
   이전 가중치 $w_{t-1}$을 기반으로 위치를 이동시켜 접근하는 방식이다.
   - 게이팅: $w_t^g \leftarrow g_t w_t^c + (1-g_t) w_{t-1}$
   - 컨볼루션 시프트: $\tilde{w}_t(i) \leftarrow \sum_{j=0}^{N-1} w_t^g(j) s_t(i-j)$
   - 정규화: $w_t(i) \leftarrow \frac{\tilde{w}_t(i)}{\gamma_t \sum_{j=0}^{N-1} \tilde{w}_t(j)}$

3. **Read Operation**:
   최종 가중치 $w_t$를 사용하여 메모리에서 읽기 벡터 $r_t$를 추출한다.
   $$r_t \leftarrow \sum_{i=0}^{N-1} w_t(i) M_t(i)$$

4. **Write Operation**:
   지우기 벡터 $e_t$와 더하기 벡터 $a_t$를 사용하여 메모리를 갱신한다.
   $$\tilde{M}_t(i) \leftarrow M_{t-1}(i)[1 - w_t(i) e_t]$$
   $$M_t(i) \leftarrow \tilde{M}_t(i) + w_t(i) a_t$$

### 세부 구현 설정 (Implementation Choices)
- **메모리 초기화**: 메모리 내용을 $10^{-6}$과 같은 작은 상수로 초기화하는 Constant Initialization을 채택하였다.
- **파라미터 초기화**: 초기 읽기 벡터 $r_0$와 주소 가중치 $w_0$를 고정된 값이 아닌, 역전파를 통해 학습 가능한 바이어스 벡터로 설정하였다.
- **비선형 함수**: 
    - $k_t, a_t$: $\tanh$를 적용하여 $[-1, 1]$ 범위로 제한한다.
    - $e_t$: $\text{sigmoid}$를 적용한다.
    - $\beta_t$: $\text{softplus}(x) = \log(\exp(x)+1)$를 적용하여 $\beta_t \ge 0$를 보장한다.
    - $g_t$: $\text{logistic sigmoid}$를 적용하여 $[0, 1]$ 범위를 유지한다.
    - $s_t$: $\text{softmax}$를 적용하여 확률 분포로 만든다.
    - $\gamma_t$: $\text{softplus}$ 적용 후 1을 더해 $\gamma_t \ge 1$을 보장한다.

## 📊 Results

### 실험 설정
- **태스크**: Copy, Repeat Copy, Associative Recall 세 가지 인공 시퀀스 학습 작업 수행.
- **비교 대상**: LSTM, DNC (Differentiable Neural Computer).
- **하이퍼파라미터**: 메모리 크기 $128 \times 20$, LSTM 컨트롤러 100 유닛, Adam 옵티마이저 (학습률 0.001), 그래디언트 클리핑 (최대 노름 50).

### 메모리 초기화 방식 비교
세 가지 초기화 방식(Constant, Learned, Random)을 비교한 결과:
- **Copy 태스크**: Constant 방식이 Learned 방식보다 약 3.5배 빠르게 수렴하였으며, Random 방식은 주어진 시간 내에 문제를 해결하지 못하였다.
- **Repeat Copy 태스크**: Constant 방식이 Learned(1.43배) 및 Random(1.35배) 방식보다 빨랐다.
- **Associative Recall 태스크**: Constant 방식이 Learned(1.15배) 및 Random(5.3배) 방식보다 빠른 수렴 속도를 보였다.
결과적으로 **Constant Initialization**이 모든 태스크에서 가장 효율적임을 확인하였다.

### 아키텍처 간 성능 비교
Constant 초기화를 적용한 NTM을 LSTM 및 DNC와 비교하였다.
- **Copy**: NTM은 DNC보다 약간 느리지만(1.2배), LSTM보다는 4~5배 빠르게 학습하여 원 논문의 성능을 재현하였다.
- **Repeat Copy**: LSTM이 MANNs에 비해 상대적으로 좋은 성능을 보였으나, 결과적으로 NTM과 DNC가 LSTM보다 빠르게 0에 가까운 오차에 도달하였다.
- **Associative Recall**: NTM과 DNC는 거의 동일한 속도로 수렴한 반면, LSTM은 주어진 시간 내에 작업을 해결하지 못하였다.

## 🧠 Insights & Discussion

본 연구는 NTM과 같은 복잡한 메모리 구조를 가진 네트워크에서 하이퍼파라미터뿐만 아니라 **초기 상태의 설정**이 학습의 성패를 가르는 핵심 요소임을 보여주었다. 특히 많은 오픈소스 구현체들이 채택했던 Random Initialization이 실제로는 수렴을 방해하는 요인이었음을 정량적으로 입증한 점이 의미가 크다.

또한, $r_0$와 $w_0$를 학습 가능한 파라미터로 설정함으로써, 엔지니어가 각 태스크에 맞춰 초기 주소를 하드코딩해야 하는 부담을 줄이고 모델의 일반화 능력을 높였다. 다만, 본 연구는 인공적인 시퀀스 태스크에 집중했으므로, 실제 복잡한 데이터셋에서도 이러한 초기화 전략이 동일하게 유효한지에 대해서는 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 불안정했던 Neural Turing Machines (NTM)의 구현 문제를 해결하고, **메모리를 작은 상수 값으로 초기화하는 방식(Constant Initialization)**이 학습 속도와 안정성을 결정짓는 핵심임을 밝혀냈다. 이를 통해 원 논문의 성능을 재현한 안정적인 오픈소스 구현체를 제공하였으며, 이는 향후 Memory Augmented Neural Networks 연구를 위한 신뢰할 수 있는 베이스라인으로 활용될 가능성이 높다.