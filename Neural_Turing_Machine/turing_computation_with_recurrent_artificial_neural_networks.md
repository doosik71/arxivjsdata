# Turing Computation with Recurrent Artificial Neural Networks

Giovanni S. Carmantini, Peter beim Graben, Mathieu Desroches and Serafim Rodrigues (2015)

## 🧩 Problem to Solve

본 논문은 튜링 머신(Turing Machine, TM)의 계산 능력을 순환 인공신경망(Recurrent Artificial Neural Networks, R-ANNs)으로 구현하는 효율적인 매핑 방법을 제시하고자 한다. 튜링 머신은 계산 가능성 이론의 기초가 되는 모델이며, 이를 신경망으로 시뮬레이션하는 것은 인공신경망의 계산적 한계와 잠재력을 이해하는 데 매우 중요하다.

기존의 Siegelmann과 Sontag의 연구에서도 튜링 머신을 신경망으로 매핑하는 시도가 있었으나, 본 연구는 더 적은 수의 뉴런을 사용하면서도 구조적으로 단순하고 투명한 구성적 매핑(constructive mapping)을 제공하는 것을 목표로 한다. 이를 통해 학습 과정 없이도 튜링 머신의 명세로부터 직접 R-ANN을 프로그래밍할 수 있는 프레임워크를 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 튜링 머신과 R-ANN 사이의 중간 단계로 **비선형 역학 오토마타(Nonlinear Dynamical Automata, NDA)**를 도입하는 것이다.

전체적인 설계 직관은 다음과 같다. 먼저 튜링 머신의 이산적인 상태와 테이프 내용을 **괴델 수화(Gödelization)**를 통해 단위 정사각형 $[0, 1]^2$ 내의 연속적인 좌표값으로 변환한다. 이후 튜링 머신의 전이 함수를 단위 정사각형 위에서 정의된 **구간별 아핀-선형 사상(piecewise affine-linear map)**으로 변환하여 NDA를 구성한다. 최종적으로 이 NDA의 동역학을 R-ANN의 아키텍처와 가중치로 그대로 투영함으로써, 튜링 머신의 동작을 실시간으로 시뮬레이션하는 신경망을 구현한다.

## 📎 Related Works

본 논문은 Siegelmann과 Sontag의 연구[1, 2]를 주요 비교 대상으로 삼는다. 기존 연구는 튜링 머신의 계산 능력을 신경망이 가질 수 있음을 증명하였으나, 본 논문의 저자들은 제안하는 방법론이 훨씬 더 **절약적(parsimonious)**이고 구조적으로 단순하다고 주장한다.

또한, Google DeepMind의 Neural Turing Machines(NTM)[5]를 언급하며, 본 연구의 투명한 구조가 향후 신경망이 학습한 알고리즘을 기호적으로 읽어내거나(symbolic read-out), NTM과 같은 데이터 접근 및 조작 방식과 통합될 수 있는 가능성을 제시한다.

## 🛠️ Methodology

연구의 방법론은 크게 두 단계의 매핑 과정으로 구성된다.

### 1. 튜링 머신에서 NDA로의 매핑

튜링 머신의 구성(Configuration)은 현재 상태, 읽기-쓰기 헤드의 위치, 테이프의 내용으로 정의된다. 이를 위해 다음과 같은 절차를 거친다.

- **Generalized Shift:** 튜링 머신의 전이 함수 $\delta$를 점 찍힌 수열(dotted sequences) 상에서 작동하는 Generalized Shift $\Omega$로 변환한다.
- **Gödelization:** 무한 수열을 실수 구간 $[0, 1]$로 매핑하는 함수 $\psi$를 정의한다.
  $$\psi(s) := \sum_{k=1}^{\infty} \gamma(r_k) g^{-k}$$
  여기서 $\gamma$는 기호를 자연수로 매핑하는 함수이고, $g$는 알파벳의 크기이다. 이를 통해 튜링 머신의 상태는 단위 정사각형 $[0, 1]^2$ 위의 한 점 $(\psi_x(\alpha'), \psi_y(\beta))$로 표현된다.
- **NDA 구성:** 튜링 머신의 기호적 변환(치환 및 시프트)은 괴델 수화된 공간에서 아핀-선형 변환으로 나타난다. 따라서 NDA는 단위 정사각형을 직사각형 영역 $D_{i,j}$로 분할하고, 각 영역에서 다음과 같은 아핀-선형 사상 $\Phi_{i,j}$를 적용하는 시스템이 된다.
  $$\Phi_{i,j}(x,y) = \begin{pmatrix} a_{i,j}^x \\ a_{i,j}^y \end{pmatrix} + \begin{pmatrix} \lambda_{i,j}^x & 0 \\ 0 & \lambda_{i,j}^y \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

### 2. NDA에서 R-ANN으로의 매핑

NDA의 궤적을 R-ANN의 뉴런 활성화 패턴으로 구현하기 위해 세 가지 계층으로 구성된 아키텍처를 제안한다.

- **Machine Configuration Layer (MCL):** 현재의 괴델 수화된 상태 $(c_x, c_y)$를 저장하는 두 개의 뉴런으로 구성된다. Ramp 함수 $R(x) = \max(0, x)$를 활성화 함수로 사용한다.
- **Branch Selection Layer (BSL):** 현재 상태가 어떤 영역 $D_{i,j}$에 속하는지 판단하여 적절한 LTL 유닛을 활성화하는 스위칭 규칙 $\Theta(x,y)$를 구현한다. Heaviside 함수 $H(x)$를 사용하여 구현되며, 특정 임계값을 넘으면 활성화된다.
- **Linear Transformation Layer (LTL):** 선택된 브랜치에 해당하는 아핀-선형 변환을 수행한다. 각 브랜치 $(i,j)$에 대해 두 개의 뉴런 $(t_{i,j}^x, t_{i,j}^y)$이 존재하며, 다음과 같이 계산된다.
  $$t_{i,j}^x = R(\lambda_{i,j}^x c_x + a_{i,j}^x - h + B_{i,x} + B_{j,y})$$
  여기서 $h$는 강한 억제성 바이어스(bias)이며, $B_{i,x}, B_{j,y}$는 BSL로부터 오는 흥분성 입력이다. 오직 BSL의 입력이 $h$를 상쇄할 만큼 충분할 때만 해당 LTL 뉴런이 활성화되어 다음 상태를 계산한다.

### 학습 및 종료 절차

본 모델은 학습을 통한 가중치 최적화가 아니라, 튜링 머신의 명세로부터 가중치를 직접 계산하여 설정하는 방식이다. 계산의 종료(Halting)는 시스템이 고정점(fixed point)에 도달했을 때, 즉 $\zeta_1(x', y') = (x', y')$가 될 때 완료된 것으로 간주한다.

## 📊 Results

본 논문은 제안된 방법론의 효율성을 입증하기 위해 Minsky의 범용 튜링 머신(Universal Turing Machine, UTM, 7개 상태 및 4개 기호)을 시뮬레이션하는 R-ANN을 설계하였다.

- **정량적 결과:** 제안된 매핑 방식을 통해 구현된 UTM 시뮬레이터는 총 **259개**의 뉴런만을 사용한다. 이는 Siegelmann과 Sontag의 방식에서 요구되었던 **886개**의 뉴런에 비해 약 1/3 수준으로 줄어든 수치이다.
- **의미:** 이는 본 논문이 제안하는 NDA 기반의 매핑 방식이 기존 방식보다 훨씬 더 절약적(parsimonious)이며, 아키텍처가 단순함을 정량적으로 보여준다.

## 🧠 Insights & Discussion

본 연구는 튜링 머신이라는 이산적인 계산 모델을 연속적인 동역학 시스템인 NDA를 거쳐 신경망으로 구현함으로써, 상징적 계산(symbolic computation)과 신경망의 결합 가능성을 이론적으로 제시하였다.

**강점:**

- 학습 없이도 튜링 머신 명세로부터 신경망을 직접 생성할 수 있는 구성적 방법을 제공한다.
- 기존 연구 대비 뉴런 수를 획기적으로 줄였으며, 시스템의 투명성이 높아 내부 동작을 분석하기 쉽다.

**한계 및 미해결 과제:**

- **미분 불가능성:** Heaviside 및 Ramp 함수를 사용하므로 end-to-end 미분이 불가능하여, 일반적인 경사 하강법(Gradient Descent)을 통한 학습에 적용하기 어렵다.
- **인코딩 문제:** 현재의 괴델 수화 방식은 상태(state)와 데이터(data)가 하나의 좌표값에 결합되어 있어, 이를 분리하여 처리하는 메커니즘이 부족하다.

저자들은 향후 연구로 NTM과 같은 데이터 접근 방식의 통합과, 고정점 동역학을 넘어선 연속 시간 동역학 시스템(continuous-time dynamical systems)으로의 확장을 제시하고 있다.

## 📌 TL;DR

본 논문은 튜링 머신 $\rightarrow$ 비선형 역학 오토마타(NDA) $\rightarrow$ 순환 인공신경망(R-ANN)으로 이어지는 새로운 매핑 경로를 제안하여, 튜링 머신의 계산 능력을 갖춘 매우 단순하고 효율적인 신경망 구조를 설계하였다. 특히 Minsky의 UTM을 구현할 때 기존 방식보다 뉴런 수를 약 70% 감소시켰으며, 이는 향후 신경망 기반의 상징적 알고리즘 학습 및 구현 연구에 중요한 이론적 토대가 될 가능성이 높다.
