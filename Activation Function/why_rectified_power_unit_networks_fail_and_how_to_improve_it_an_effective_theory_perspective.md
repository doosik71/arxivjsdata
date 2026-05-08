# Why Rectified Power Unit Networks Fail and How to Improve It: An Effective Theory Perspective

Taeyoung Kim, Myungjoo Kang (2024)

## 🧩 Problem to Solve

본 논문은 Rectified Power Unit (RePU) 활성화 함수를 사용하는 신경망이 층이 깊어짐에 따라 발생하는 심각한 불안정성 문제를 해결하고자 한다. RePU는 ReLU의 일반화된 형태로, 고차 미분이 가능하다는 장점 덕분에 Sobolev 공간에서의 매끄러운 함수 근사나 미분 가능한 신경망 구축에 유리하다고 알려져 있었다.

그러나 실제 실험적으로는 하이퍼파라미터 초기화 값과 관계없이 층이 깊어질수록 출력값이 기하급수적으로 폭발(exploding)하거나 소멸(vanishing)하여 학습이 완전히 실패하는 현상이 관찰되었다. 본 연구의 목표는 Effective Theory(유효 이론) 관점에서 이러한 실패의 근본 원인을 이론적으로 규명하고, RePU의 장점인 미분 가능성은 유지하면서 깊은 층에서도 안정적으로 동작하는 새로운 활성화 함수를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 RePU 네트워크의 실패 원인을 양자장론(Quantum Field Theory)의 언어를 빌린 Effective Theory로 분석하여 '임계 조건(Criticality Condition)'의 불충족을 증명한 것이다.

중심적인 아이디어는 네트워크의 **Susceptibility(감수성)**를 계산하는 것이다. 저자들은 병렬 감수성($\chi_\parallel$)과 수직 감수성($\chi_\perp$)이라는 두 가지 지표를 정의하고, RePU의 경우 $p > 1$일 때 이 두 값의 비율이 고정되어 있어 동시에 1로 만드는 것이 불가능함을 보였다. 이를 통해 RePU가 구조적으로 불안정할 수밖에 없음을 이론적으로 입증하였으며, 이를 해결하기 위해 두 감수성의 비율이 1로 수렴하도록 설계된 **Modified Rectified Power Unit (MRePU)**를 제안하였다.

## 📎 Related Works

논문은 기존의 다양한 활성화 함수들(Perceptron, Sigmoid, Tanh, ReLU, GELU 등)과 그 한계를 언급한다. 특히 ReLU는 미분 불가능성이라는 단점이 있지만, 상수 미분값을 가져 기울기 소멸 문제를 완화하며 깊은 망 구성이 가능하다는 점을 강조한다.

RePU는 ReLU의 일반화 버전으로, $p$-차 미분이 가능하여 매끄러운 함수의 최적 근사가 가능하다는 이론적 연구들이 존재했다. 하지만 이러한 이론적 우수성에도 불구하고 실제 깊은 망에서의 학습 실패 문제는 다루어지지 않았다. 또한, 본 논문은 신경망의 너비가 무한할 때 가우시안 프로세스를 따르며, 유한한 너비에서는 renormalization group (RG) flow로 정보 전달을 해석하는 Effective Theory 프레임워크를 기반으로 하고 있다.

## 🛠️ Methodology

### 1. 신경망 구조 및 RePU 정의

본 연구는 Fully Connected Network (FCN)를 대상으로 한다. RePU 활성화 함수는 다음과 같이 정의된다.
$$\sigma(z) = \begin{cases} z^p, & \text{if } z \ge 0 \\ 0, & \text{if } z < 0 \end{cases}$$
여기서 $p$는 양의 정수이다. $p=1$이면 ReLU가 되며, $p=2, 3$인 경우는 각각 ReQU, ReCU라고 한다.

### 2. Effective Theory를 통한 분석 도구

저자들은 신경망의 사전 활성화 값(preactivation)의 분포를 분석하기 위해 $M$-point connected correlators(누적량)를 사용한다. 네트워크의 커널(Kernel) $K^{(l)}$의 재귀적 관계를 통해 정보의 흐름을 분석하며, 특히 두 입력 사이의 거리가 층을 통과하며 어떻게 변하는지를 결정하는 **Susceptibility**를 다음과 같이 정의한다.

- **Parallel Susceptibility ($\chi_\parallel$):** 입력의 변화 방향과 동일한 방향으로의 커널 변화율.
- **Perpendicular Susceptibility ($\chi_\perp$):** 입력의 변화 방향과 수직인 방향으로의 커널 변화율.

안정적인 네트워크가 되려면 두 감수성 모두 $\chi = 1$인 임계 상태(criticality)에 있어야 한다. 만약 $1$보다 크면 값이 폭발하고, $1$보다 작으면 소멸하기 때문이다.

### 3. RePU의 실패 원인 분석

RePU에 대해 계산한 감수성은 다음과 같다.
$$\chi_\parallel(K) = C_W p (2p-1)!! K^{\frac{p-1}{2}}, \quad \chi_\perp(K) = C_W p^2 (2p-3)!! K^{\frac{p-1}{2}}$$
분석 결과, $\chi_\parallel : \chi_\perp = (2p-1) : p$ 의 비율을 가진다. $p=1$(ReLU)이 아닌 이상, 두 값을 동시에 1로 설정하는 것이 불가능하다. 즉, 하나를 1로 맞추면 다른 하나는 반드시 1보다 크거나 작게 되어, 깊은 층을 통과할 때 커널이 이중 지수적으로(double exponentially) 폭발하거나 소멸하게 된다.

### 4. Modified RePU (MRePU) 제안

위 문제를 해결하기 위해 저자들은 다음과 같은 MRePU를 제안한다.
$$\sigma_{m;p}(z) = \begin{cases} z(z+1)^p, & \text{if } z \ge -1 \\ 0, & \text{if } z < -1 \end{cases}$$
MRePU는 수치적 계산 결과, 커널 값 $K$가 0에 가까워질 때 $\chi_\parallel / \chi_\perp$의 비율이 1로 수렴함을 확인하였다. 이는 MRePU가 $\chi = 1$인 임계 조건을 만족할 수 있으며, 적절한 $C_W$ 설정을 통해 깊은 층에서도 안정적인 학습이 가능함을 의미한다.

## 📊 Results

### 1. 초기화 상태의 커널 거동 실험

RePU($p=2$) 네트워크에서 $C_W$ 값에 따라 커널이 이중 지수적으로 폭발하거나 소멸하는 것을 실험적으로 확인하였다. 반면, MRePU는 층이 깊어져도 커널의 스케일이 일정하게 유지되는 안정적인 모습을 보였다.

### 2. 함수 근사 성능 평가 (Regression)

다양한 함수 클래스에 대해 MRePU, ReLU, GELU의 성능을 비교하였다.

- **다항함수 근사:** MRePU가 압도적인 성능을 보였다. 특히 $xyz$와 같은 고차 다항함수에서 다른 모델들은 학습에 실패했으나, MRePU는 성공적으로 근사하였다. 이는 MRePU가 다항함수 근사에 적합한 **Inductive Bias**를 제공하기 때문이다.
- **미분 가능 함수 근사:** $L^2$ 손실 값 자체는 ReLU가 낮았으나, 미분값의 연속성과 등고선의 매끄러움(smoothness) 측면에서는 MRePU가 훨씬 더 안정적인 근사 결과를 보여주었다.
- **PINNs (Burgers Equation):** 물리 정보 신경망(PINN) 실험에서 MRePU는 7개의 은닉층을 가진 깊은 구조에서도 유의미한 학습 성과를 거두었다. 성능 순위는 $\text{GELU} > \text{MRePU} > \text{ReLU}$ 순이었으며, ReLU는 학습에 실패하였다.

### 3. 이론적 근사 특성

저자들은 MRePU 네트워크가 임의의 $p$차 다항식을 정확하게 재구성할 수 있음을 수학적으로 증명(Theorem 8)하였으며, 미분 가능한 함수 클래스에 대한 보편적 근사 정리(Universal Approximation Theorem)를 유도(Corollary 10)하여 RePU의 이론적 장점을 그대로 계승했음을 보였다.

## 🧠 Insights & Discussion

본 논문은 단순히 새로운 활성화 함수를 제안한 것이 아니라, **Effective Theory라는 물리적 분석 도구를 통해 딥러닝 모델의 안정성 문제를 정밀하게 진단**했다는 점에서 큰 의의가 있다.

특히, 활성화 함수의 수학적 형태가 네트워크에 특정한 Inductive Bias를 부여하며, 이것이 해결하려는 문제의 성격(다항식, 매끄러운 함수, 물리 방정식 등)에 따라 성능 차이를 만든다는 점을 시사한다. RePU의 실패는 단순히 하이퍼파라미터 튜닝의 문제가 아니라, $\chi_\parallel$와 $\chi_\perp$의 불일치라는 구조적 결함에서 기인했다는 점을 명확히 하였다.

한계점으로는 MRePU가 매우 뛰어난 성능을 보였음에도 불구하고, PINN 실험에서는 여전히 GELU보다 성능이 낮았다는 점이 있다. 이는 아주 매끄러운(smooth) 함수를 근사할 때는 MRePU보다 더 높은 차원의 매끄러움을 가진 GELU 등이 더 유리할 수 있음을 암시한다.

## 📌 TL;DR

RePU 활성화 함수는 이론적 잠재력에도 불구하고 깊은 신경망에서 커널이 폭발/소멸하는 치명적 결함이 있다. 본 논문은 이를 Effective Theory의 Susceptibility 분석으로 규명하고, 임계 조건을 만족하도록 설계된 **MRePU**를 제안하였다. MRePU는 깊은 층에서도 안정적으로 학습되며, 특히 다항함수 및 미분 가능 함수의 근사에서 ReLU/GELU보다 강력한 Inductive Bias를 제공하여 뛰어난 성능을 입증하였다.
