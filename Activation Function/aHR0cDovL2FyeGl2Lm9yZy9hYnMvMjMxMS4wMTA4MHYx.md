# Optical ReLU-like Activation Function Based on a Semiconductor Laser with Optical Injection

Guan-Ting Liu, Yi-Wei Shen, Rui-Qian Li, Jingyi Yu, Xuming He, and Cheng Wang (Year not explicitly provided)

## 🧩 Problem to Solve

인공신경망(ANN)은 기본적으로 선형 연산인 곱셈-누산(Multiply-Accumulate, MAC)과 비선형 활성화 함수(Activation Function)의 반복으로 구성된다. 최근 광신경망(Optical Neural Networks, ONNs)이 높은 대역폭, 에너지 효율성, 낮은 지연 시간 등의 장점으로 인해 주목받고 있으나, 대부분의 ONN은 선형 연산만을 광학 도메인에서 수행하고 비선형 활성화 함수는 여전히 디지털 도메인에서 처리하는 한계가 있다.

이러한 구조는 각 은닉층(hidden layer) 사이에서 광-전 변환(Optical-to-Electrical conversion)과 아날로그-디지털 변환(Analog-to-Digital conversion)을 필수적으로 요구하며, 이는 막대한 전력 소비와 높은 지연 시간을 초래한다. 따라서 본 논문의 목표는 이러한 변환 과정 없이 광학 도메인에서 직접 작동하는 전광(all-optical) 비선형 활성화 함수, 특히 ReLU(Rectified Linear Unit)와 유사한 기능을 구현하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **광 주입(Optical Injection)이 적용된 반도체 레이저의 비선형 동역학**을 활용하는 것이다. 특히, 주입 잠금 도표(Injection-locking diagram)에서 **Hopf 분기(Hopf bifurcation) 이상의 영역**에서 작동할 때 ReLU-like 활성화 함수가 나타난다는 점을 발견하였다.

또한, 마스터 레이저(Master laser)와 슬레이브 레이저(Slave laser) 사이의 **디튜닝 주파수(Detuning frequency, $\Delta f$)를 조절함으로써 활성화 함수의 기울기를 재구성(Reconfigurable)**할 수 있음을 실험적으로 증명하였다.

## 📎 Related Works

기존의 전광 활성화 함수 구현 방식은 다음과 같은 한계와 특징이 있다.
- **SOA 기반 방식:** SOA(Semiconductor Optical Amplifier) 기반 파장 변환기나 MZI를 사용하여 3차 다항식이나 Sigmoid 함수를 구현하였으나, 다수의 레이저가 필요하다는 복잡성이 있다.
- **소재 기반 방식:** 박막 리튬 나이오베이트(Thin-film lithium niobate)의 제2고조파 생성(SHG) 효과나 미세 링 공진기(MRR)의 열-광학 효과를 이용하여 ReLU, ELU 등을 구현하였다.
- **레이저 기반 방식:** 광 주입이 적용된 멤브레인 레이저가 임계값 아래에서 ReLU 함수를 나타내거나, saddle-node 분기 근처에서 $\tanh$-like 함수가 나타남이 보고된 바 있다.

본 논문은 기존 연구와 달리 **Hopf 분기 이상의 넓은 영역**에서 ReLU-like 함수를 구현하며, 입력과 출력의 파장을 동일하게 유지하여 심층 광신경망(Deep ONN)의 계층적 구조(Cascading)를 형성하기에 매우 유리한 구조를 가진다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조
실험 장치는 마스터 레이저, 광 서큘레이터(Optical circulator), 슬레이브 레이저, 그리고 대역 통과 필터(Bandpass filter)로 구성된다.
1. 마스터 레이저의 빛이 슬레이브 레이저로 단방향 주입된다.
2. 슬레이브 레이저는 비선형 뉴런 역할을 수행한다.
3. 출력단에 설치된 필터는 마스터 레이저의 파장($\lambda_m$)만을 통과시켜, 입력과 출력이 동일한 파장을 갖도록 보장한다. 이는 광 뉴런을 직렬로 연결하는 cascading 구조를 가능하게 한다.

### ReLU-like 함수의 작동 원리
슬레이브 레이저의 비선형 응답은 주입 비율(Injection ratio)과 디튜닝 주파수($\Delta f$)에 의해 결정된다. 연구팀은 $\Delta f > 0$인 Hopf 분기 영역에서 다음과 같은 메커니즘으로 ReLU-like 함수를 구현하였다.

- **임계값 전(Below Threshold):** 마스터 레이저 모드와 슬레이브 레이저 모드 사이에 강한 전력 경쟁(Power competition)이 발생하며, 두 모드의 위상이 거의 상관관계가 없다. 이때 입력 전력이 증가함에 따라 마스터 레이저 모드의 전력이 빠르게 증가한다.
- **임계값 후(Above Threshold):** 슬레이브 레이저 모드가 충분히 억제되어 마스터 레이저 모드가 지배적이 된다. 이때는 강한 주파수 밀어내기(Frequency pushing) 효과가 발생하며, 전력 증가 속도가 완만해진다.

이로 인해 출력 전력 $P_{out}$은 입력 전력 $P_{in}$에 대해 굴곡(kink)이 있는 비선형 곡선을 그리게 되며, 이는 결과적으로 ReLU와 유사한(단, 형태적으로는 180도 회전된 형태의) 특성을 갖는다.

### 수학적 모델 및 재구성
활성화 함수의 기울기 $D$는 디튜닝 주파수 $\Delta f$에 의해 결정된다.
- **임계값 이하 기울기:** $\Delta f$가 $0 \text{ GHz}$에서 $30 \text{ GHz}$로 증가함에 따라 기울기 $D$는 $2.55$에서 $0.32$로 비선형적으로 감소한다.
- **임계값 이상 기울기:** 약 $0.065$로 거의 일정하게 유지된다.

## 📊 Results

### 실험 조건 및 지표
- **데이터셋:** MNIST 손글씨 숫자 데이터셋 (훈련 60,000장, 테스트 10,000장)
- **네트워크 구조:** 입력층(784) $\to$ 은닉층1(784) $\to$ 은닉층2(256) $\to$ 출력층(10)의 완전 연결 신경망(Fully-connected NN)
- **비교 지표:** 분류 정확도(Accuracy)

### 정량적 결과
연구팀은 두 가지 형태의 광학 ReLU-like 함수를 정의하여 성능을 테스트하였다.
- **ReLU-I:** $y = \min(Dx, 0.065x)$
- **ReLU-II:** $y = \min(Dx, 0.065x + y_0)$ (단, $x \ge 0$)

결과는 다음과 같다.
- **ReLU-I 정확도:** 기울기 $D=0.32$일 때 $97.73\%$, $D=2.55$일 때 $98.02\%$를 기록하였다.
- **ReLU-II 정확도:** $97.22\% \sim 97.64\%$ 범위를 기록하였다.
- **기준선(Baseline):** 표준 Leaky ReLU($y = \max(0.01x, x)$)의 정확도는 $98.08\%$였다.

결과적으로 광학 ReLU-like 함수를 사용한 신경망의 성능은 표준 Leaky ReLU와 매우 유사한 수준(Comparable)임을 확인하였다.

## 🧠 Insights & Discussion

### 강점
본 연구는 전광 활성화 함수를 구현함으로써 ONN의 최대 난제인 O/E 및 A/D 변환 문제를 해결할 가능성을 제시하였다. 특히 입력과 출력의 파장이 동일하다는 점은 실제 하드웨어로 심층 신경망을 구축할 때 매우 강력한 이점이 된다. 또한, $\Delta f$ 조절을 통해 기울기를 하드웨어적으로 튜닝할 수 있다는 점은 유연한 네트워크 설계를 가능하게 한다.

### 한계 및 비판적 해석
실험적으로 구현된 함수는 엄밀히 말해 표준 ReLU($\max(0, x)$)와는 다른 형태(Concave-like)이다. 논문에서는 이를 'ReLU-like'라고 명명하였으나, 실제로는 기울기가 급격히 증가하다가 완만해지는 형태이다. 다행히 MNIST 테스트를 통해 이러한 형태의 비선형성만으로도 충분한 분류 성능이 나옴을 증명하였으나, 다른 복잡한 데이터셋에서도 동일한 효과가 있을지는 추가 검증이 필요하다. 또한, 본 실험은 정적인(static) 입력 전력을 사용하였으므로, 실제 통신 속도 수준의 고속 변조 신호에서도 동일한 비선형 응답이 유지되는지는 아직 미지수이다.

## 📌 TL;DR

본 논문은 반도체 레이저의 광 주입 동역학, 특히 Hopf 분기 영역을 이용하여 **전광(all-optical) ReLU-like 활성화 함수**를 구현하였다. 이 함수는 디튜닝 주파수를 통해 기울기를 조절할 수 있으며, 입력과 출력의 파장이 동일하여 광신경망의 계층적 구성에 유리하다. MNIST 분류 실험 결과, 표준 Leaky ReLU와 대등한 수준의 정확도를 달성하여 전광 뉴런으로서의 실용성을 입증하였다. 이는 향후 전광 시냅스와 전광 뉴런이 결합된 완전한 형태의 All-Optical Neural Network 구현에 중요한 기여를 할 것으로 보인다.