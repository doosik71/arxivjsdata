# Piecewise Linear Units Improve Deep Neural Networks

Jordan Inturrisi, Sui Yang Khoo, Abbas Kouzani, Riccardo Pagliarella (연도 미기재)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델의 핵심 구성 요소인 활성화 함수(Activation Function)의 선택이 학습의 성공 여부에 큰 영향을 미친다는 점에 주목한다. 현재 많은 연구자와 실무자들이 단순함과 신뢰성 덕분에 Rectified Linear Unit (ReLU)를 선호하지만, ReLU는 다음과 같은 몇 가지 치명적인 단점을 가지고 있다.

첫째, ReLU는 비음수(non-negative) 특성으로 인해 활성화 값의 평균이 0이 아니며, 이는 다음 레이어로의 bias shift를 유발하여 성능에 부정적인 영향을 준다. 둘째, 데이터의 대칭적 또는 반대칭적 특성을 학습하기 위해 대칭 함수보다 두 배 더 많은 뉴런이 필요하다. 셋째, $x \le 0$ 구간에서 기울기가 0이 되어 가중치 업데이트가 일어나지 않는 "Dead Neurons" 문제가 발생하며, 이는 학습 정체 및 수렴 성능 저하로 이어진다.

기존에 ReLU를 대체하기 위해 제안된 많은 함수들은 주로 수동으로 설계(hand-designed)되었으며, 모델이나 데이터셋에 따라 성능 개선 효과가 일관되지 않은 경향이 있다. 따라서 본 논문의 목표는 학습 과정에서 활성화 함수 자체를 최적화할 수 있는 적응형 piecewise linear 활성화 함수를 제안하고, 이를 통해 ReLU의 한계를 극복하고 일반화된 성능 향상을 달성하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 네트워크의 각 차원(dimension)별로 독립적으로 학습 가능한 적응형 piecewise linear 활성화 함수인 Piecewise Linear Unit (PiLU)와 DoubleReLU를 제안한 것이다. 

특히 PiLU의 중심 아이디어는 기존의 PReLU를 더욱 일반화하여, 적응형 매듭(adaptive knot) $\gamma$와 이 매듭의 양옆에서 작동하는 두 개의 적응형 기울기(adaptive gradients) $\alpha, \beta$를 도입하는 것이다. 이를 통해 모델은 학습 데이터에 최적화된 비선형 형태를 스스로 찾아낼 수 있으며, 이는 수동 설계된 함수보다 유연한 대응을 가능하게 한다.

## 📎 Related Works

논문에서는 ReLU를 개선하기 위해 제안된 Leaky ReLU, Parametric ReLU (PReLU), Exponential Linear Units (ELU), Swish, Adaptive Piecewise Linear (APL) units, PLU 등을 언급한다. 

기존 연구들의 한계점으로 저자들은 성능 비교 방식의 불완전함을 지적한다. 대부분의 선행 연구가 단 한 번의 실행 결과나 5회 실행의 중앙값만을 보고하여 결과에 노이즈가 섞여 있을 가능성이 크다고 주장한다. 본 논문은 이러한 한계를 극복하기 위해 30회의 서로 다른 랜덤 시드(random seed)를 이용한 실험 결과의 분포를 분석함으로써 통계적 유의성을 확보하고자 한다.

## 🛠️ Methodology

### 1. DoubleReLU
DoubleReLU는 $\pm \alpha$ 구간 사이에 '진정한 제로 영역(zone of true zeroes)'을 생성하는 함수이다. 출력값은 임계값 처리(thresholded)되지 않으며, 제로 영역 밖에서 0부터 다시 시작된다. 수학적 정의는 다음과 같다.

$$f(x) = \begin{cases} x - \alpha & \text{if } x > \alpha \\ 0 & \text{if } -\alpha \le x \le \alpha \\ x + \alpha & \text{if } x < -\alpha \end{cases}$$

이에 따른 도함수는 다음과 같다.

$$f'(x) = \begin{cases} 1 & \text{if } |x| > \alpha \\ 0 & \text{if } |x| \le \alpha \end{cases}$$

DoubleReLU는 ReLU와 유사하지만 출력이 음수 값을 가질 수 있어 활성화 평균을 0 근처로 맞출 수 있다는 장점이 있다.

### 2. Piecewise Linear Unit (PiLU)
PiLU는 PReLU를 일반화한 형태로, 매듭 $\gamma$를 기준으로 서로 다른 두 개의 기울기를 가진다.

$$f(x) = \begin{cases} \alpha x + \gamma(1 - \alpha) & \text{if } x > \gamma \\ \beta x + \gamma(1 - \beta) & \text{if } x \le \gamma \end{cases}$$

도함수는 다음과 같이 두 개의 분리된 상수로 나타난다.

$$f'(x) = \begin{cases} \alpha & \text{if } x > \gamma \\ \beta & \text{if } x \le \gamma \end{cases}$$

PiLU는 $\alpha, \beta, \gamma$라는 3개의 적응형 파라미터를 추가로 가지며, 이들은 경사 하강법(gradient descent)을 통해 학습된다. 특히 PiLU는 ReLU, LReLU, PReLU를 특수 사례로 포함하는 일반화된 Rectifier Unit이다. 예를 들어 $\gamma=0, \alpha=1, \beta=0$으로 설정하면 ReLU가 된다.

### 3. 복잡도 및 구현 상세
- **시간/공간 복잡도**: ReLU, PReLU, DoubleReLU, PiLU 모두 시간 및 공간 복잡도는 $O(n)$으로 선형적이다. 다만 계산 단계가 늘어남에 따라 $\text{ReLU} \rightarrow \text{PReLU} \rightarrow \text{DoubleReLU} \rightarrow \text{PiLU}$ 순으로 계산 시간이 소폭 증가하지만, 입력 크기가 커질수록 ReLU와의 상대적 시간 차이는 감소한다.
- **학습 절차**: $\text{Adam}$ 옵티마이저($\eta=0.001$)와 $\text{Glorot normal}$ 초기화 기법을 사용하였으며, 손실 함수로는 $\text{categorical cross entropy}$를 사용하였다.
- **파라미터 공유**: 각 레이어의 채널별로 적응형 가중치를 공유하는 Channel-wise adaptive weight sharing scheme을 적용하여 파라미터 증가량을 최소화하였다.

## 📊 Results

### 실험 환경
- **데이터셋**: CIFAR-10, CIFAR-100
- **모델 구조**: 5개의 Convolutional Layer와 Average Pooling, Dropout, Fully Connected Layer로 구성된 CNN 아키텍처를 사용하였다.
- **평가 방법**: 30개의 서로 다른 랜덤 시드를 사용하여 성능 분포를 측정하였다.

### 정량적 결과
실험 결과, 모든 조건에서 PiLU가 가장 우수한 성능을 보였다.

- **CIFAR-10**:
    - PiLU는 ReLU 대비 분류 오류(classification error)를 $18.53\%$ 감소시켰으며, 정확도는 $6.2$ percentage points 향상되었다.
    - 정확도 순위: $\text{PiLU} (72.74\%) > \text{PReLU} (70.81\%) > \text{DoubleReLU} (68.49\%) > \text{ReLU} (66.54\%)$.
- **CIFAR-100**:
    - PiLU는 ReLU 대비 분류 오류를 $13.13\%$ 감소시켰으며, 정확도는 $9.55$ percentage points 향상되었다.
    - 정확도 순위: $\text{PiLU} (36.81\%) > \text{PReLU} (33.76\%) \approx \text{DoubleReLU} (33.39\%) > \text{ReLU} (27.26\%)$.

### 파라미터 증가량
PiLU를 도입했을 때 발생하는 파라미터 증가량은 매우 미미하다. CIFAR-10 기준 ReLU 대비 약 $1.34\%$ ($480$개 파라미터 추가), CIFAR-100 기준 약 $1.15\%$ 증가에 그쳤다.

## 🧠 Insights & Discussion

본 논문은 활성화 함수를 수동으로 설계하는 대신 학습 과정에서 최적화하는 것이 성능 향상에 훨씬 효율적임을 입증하였다. PiLU는 매우 적은 수의 파라미터 추가만으로도 ReLU나 PReLU보다 더 빠르게, 그리고 더 나은 Local Minimum에 도달하는 것으로 분석된다.

**강점**:
- 통계적 분포 분석(30회 실험)을 통해 결과의 신뢰성을 높였다.
- PiLU가 기존의 다양한 Rectifier 함수들을 포괄하는 일반화된 형태임을 수학적으로 증명하였다.

**한계 및 논의사항**:
- 본 연구는 이미지 분류 작업(CIFAR)에 한정되어 있으며, ImageNet과 같은 대규모 데이터셋이나 머신 트랜슬레이션, 음성 인식 등 다른 도메인에서의 효과는 검증되지 않았다.
- scalar-to-scalar 함수에만 집중했으므로, Max-pooling이나 BatchNorm과 같은 Many-to-one/many-to-many 형태의 함수로의 확장은 미래 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 학습 가능한 적응형 매듭과 기울기를 가진 **Piecewise Linear Unit (PiLU)**를 제안하여 ReLU의 Dead Neuron 및 Bias Shift 문제를 해결하고자 하였다. CIFAR-10/100 실험 결과, PiLU는 파라미터를 아주 소폭 증가시키면서도 ReLU 대비 오류율을 각각 $18.53\%$, $13.13\%$ 감소시키는 괄목할만한 성능 향상을 보였다. 이는 활성화 함수를 고정하지 않고 데이터에 맞게 학습시키는 방향이 딥러닝 모델 최적화에 매우 중요함을 시사한다.