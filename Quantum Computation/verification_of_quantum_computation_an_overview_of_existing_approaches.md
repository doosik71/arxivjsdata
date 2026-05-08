# Verification of quantum computation: An overview of existing approaches

Alexandru Gheorghiu, Theodoros Kapourniotis, and Elham Kashefi (2018)

## 🧩 Problem to Solve

본 논문은 양자 컴퓨터가 클래식 컴퓨터로는 해결 불가능한(intractable) 문제를 효율적으로 해결할 수 있다는 약속에서 비롯된 '검증의 역설'을 해결하고자 한다. 핵심 문제는 다음과 같다. 만약 양자 컴퓨터가 클래식 컴퓨터가 효율적으로 계산하거나 검증할 수 없는 문제를 풀었다면, 그 결과가 정말로 올바른지 어떻게 확인할 수 있는가?

이 문제는 단순히 기술적인 문제를 넘어 계산 복잡도 이론(Complexity Theory)의 관점에서 중요한 의미를 갖는다. 특히 $\text{BQP}$(양자 컴퓨터로 효율적 해결 가능한 문제 클래스)에 속하는 모든 문제가 클래식 검증자($\text{BPP}$)가 효율적으로 검증할 수 있는 인터랙티브 증명(Interactive Proof, $\text{IP}$) 시스템을 가지는지, 그리고 이때 증명자(Prover)의 능력이 $\text{BQP}$로 제한되어도 가능한지가 핵심 연구 대상이다.

## ✨ Key Contributions

본 논문의 주요 기여는 양자 계산 검증(Quantum Verification)을 위한 기존의 다양한 접근 방식들을 체계적으로 분류하고, 각 프로토콜의 구조, 복잡도, 요구 자원을 비교 분석한 것이다.

핵심적인 설계 아이디어는 **Blindness(맹목성)**와 **Self-testing(자기 테스트)**이다.

- **Blindness**: 검증자가 계산 내용을 암호화하여 증명자가 자신이 무엇을 계산하는지 알지 못하게 함으로써, 증명자가 특정 계산 결과만을 조작하는 것을 방지하고 '함정(Trap)'을 심어 올바른 수행 여부를 확인한다.
- **Self-testing**: 비국소성 게임(Non-local game, 예: CHSH 게임)의 결과가 양자 역학적 최댓값에 근접할 경우, 증명자들이 공유하는 상태와 측정 연산자가 특정 형태(예: Bell state)임이 강제된다는 점을 이용하여 하드웨어의 정당성을 검증한다.

## 📎 Related Works

논문은 양자 계산 검증을 이해하기 위해 다음과 같은 계산 복잡도 클래스를 소개하며 기존 연구의 한계를 설명한다.

- $\text{BPP} \subseteq \text{BQP}$ 및 $\text{BPP} \subseteq \text{MA}$ 관계가 성립하지만, $\text{BQP} \subseteq \text{MA}$인지는 불분명하며 일반적으로 그렇지 않다고 믿어진다. 즉, 일반적인 양자 계산 결과에 대한 단순한 증거(Witness)는 존재하지 않을 가능성이 크다.
- 이를 해결하기 위해 $\text{BQP} \subseteq \text{IP}$임이 알려져 있으나, $\text{IP}$에서의 증명자는 계산 능력이 무한하다는 가정이 있다. 실제 시스템에서는 증명자가 $\text{BQP}$ 능력을 가진 양자 컴퓨터여야 하므로, 증명자의 능력을 $\text{BQP}$로 제한한 $\text{QPIP}$ 클래스에 대한 연구가 필요하다.
- 기존 접근 방식들은 크게 검증자가 양자 상태를 준비하여 보내는 방식, 증명자가 보낸 상태를 측정하는 방식, 그리고 여러 증명자 간의 얽힘을 이용하는 방식으로 나뉜다.

## 🛠️ Methodology

논문은 양자 검증 프로토콜을 세 가지 주요 범주로 분류하여 설명한다.

### 1. Prepare-and-send protocols ($\text{QPIP}$)

검증자가 상수 크기의 양자 장치를 가져 상태를 준비해 증명자에게 보내는 방식이다.

- **Quantum Authentication-based**: $\text{QAS}$(양자 인증 스킴)를 사용하여 상태가 변조되었는지 확인한다. Clifford-QAS는 Clifford 연산을 통해 인코딩하며, Poly-QAS는 다항식 CSS 코드를 사용하여 통신 횟수를 줄인다.
- **Trap-based (VUBQC)**: $\text{MBQC}$(측정 기반 양자 계산)를 활용한다. 계산 큐비트 사이에 무작위로 함정(Trap) 큐비트와 더미(Dummy) 큐비트를 섞어 배치하며, Blindness 덕분에 증명자는 함정의 위치를 알 수 없다. 함정 측정 결과가 예상과 다르면 프로토콜을 중단한다.
- **Repeated runs (Test-or-Compute)**: 계산 실행과 테스트 실행(Identity computation)을 무작위로 교차 수행한다. 증명자는 현재가 테스트 중인지 실제 계산 중인지 알 수 없으므로 정직하게 수행할 수밖에 없다.

### 2. Receive-and-measure protocols ($\text{QPIP}$)

증명자가 상태를 준비해 보내면 검증자가 이를 측정하여 검증하는 방식이다.

- **Measurement-only**: 증명자가 여러 복사본의 그래프 상태(Graph state)를 보내면, 검증자가 일부는 Stabilizer 측정을 통해 상태의 정당성을 확인하고, 나머지는 $\text{MBQC}$ 계산에 사용한다.
- **Post hoc verification (1S-Post-hoc)**: 계산 후 검증하는 방식으로, $\text{BQP}$ 문제를 $\text{QMA}$-완전 문제인 $k$-local Hamiltonian 문제로 변환한다. Feynman-Kitaev clock state의 에너지 추정치를 측정하여 정답 여부를 판단한다.

### 3. Entanglement-based protocols ($\text{MIP}^*$)

완전한 클래식 검증자가 통신이 불가능한 두 명 이상의 증명자와 상호작용하는 방식이다.

- **CHSH Rigidity 기반 (RUV)**: CHSH 게임의 승률이 최댓값에 가까우면 증명자들이 Bell pair를 공유하고 특정 측정치를 사용함이 보장된다(Rigidity). 이를 통해 상태/프로세스 토모그래피를 수행하고 계산을 검증한다.
- **GKW 및 HPDF**: CHSH Rigidity를 통해 리소스 상태의 준비를 검증한 후, $\text{VUBQC}$나 $\text{MBQC}$를 통해 계산을 수행한다.
- **NV Protocol**: constant robustness를 가진 self-testing 결과를 사용하여, $\text{BQP}$ 문제에 대응하는 Hamiltonian의 기저 상태 에너지를 상수로 제한된 라운드 내에 검증한다.

## 📊 Results

본 논문은 각 프로토콜의 효율성과 자원 요구량을 비교 분석한 결과를 제시한다.

### 정량적 비교 요약

- **검증자 자원**:
  - $\text{VUBQC}$, $\text{Test-or-Compute}$, $\text{ la la measurement-only}$ 등은 단일 큐비트 준비/측정 장치($O(1)$)만으로 가능하다.
  - $\text{Poly-QAS}$, $\text{Clifford-QAS}$ 등은 $\log(1/\epsilon)$ 크기의 양자 컴퓨터가 필요하다.
- **통신 복잡도**:
  - $\text{VUBQC}$ 및 $\text{ la la la receive-and-measure}$ 프로토콜들은 계산 크기 $|C|$에 대해 선형($O(|C|)$) 또는 다항식 수준의 통신 복잡도를 가진다.
  - 반면, 초기 $\text{RUV}$ 및 $\text{GKW}$ 프로토콜은 지수적으로 큰 통신 오버헤드($O(|C|^c)$, $c$가 매우 큼)가 발생하여 실용성이 낮다. $\text{NV}$ 프로토콜은 이를 상수 라운드로 줄였다.
- **Blindness**: $\text{VUBQC}$와 $\text{ la la la la l la measurement-only}$는 Blindness를 제공하지만, $\text{Post hoc}$ 방식(1S-Post-hoc, FH, NV)은 일반적으로 Blind 하지 않다.

## 🧠 Insights & Discussion

### 강점 및 한계

- **강점**: 본 논문은 파편화되어 있던 양자 검증 프로토콜들을 복잡도 이론과 물리적 구현 가능성 관점에서 통합적으로 정리하였다. 특히 $\text{QPIP}$와 $\text{MIP}^*$의 관계를 명확히 하여 클래식 검증자의 가능성을 탐색했다.
- **한계**: 이론적으로는 완벽하나, 실제 구현 시 발생하는 노이즈(Noise) 문제가 해결되지 않았다. 대부분의 프로토콜이 이상적인(Ideal) 환경을 가정한다.

### 비판적 해석 및 논의

- **결함 허용(Fault Tolerance)**: 논문에서 언급하듯, 검증자가 사용하는 장치 자체의 노이즈가 증명자의 조작과 구분이 되지 않을 수 있다. 따라서 실용적인 검증을 위해서는 $\text{VUBQC}$ 등에 Topological code 같은 결함 허용 기법을 결합하는 것이 필수적이다.
- **현실적 제약**: $\text{MIP}^*$ 기반 프로토콜은 증명자 간의 통신 차단이라는 강력한 가정이 필요하다. 이를 물리적으로 보장하기 위해 공간적 분리(Space-like separation)를 구현하는 것이 실제 시스템의 핵심 과제가 될 것이다.

## 📌 TL;DR

본 논문은 양자 계산의 정당성을 검증하기 위한 다양한 프로토콜을 $\text{QPIP}$와 $\text{MIP}^*$ 관점에서 분석한 종합 리뷰 보고서이다. **Prepare-and-send(함정 기반)**, **Receive-and-measure(에너지 추정 기반)**, **Entanglement-based(Self-testing 기반)**의 세 가지 경로를 제시하며, 검증자의 양자 자원 최소화와 통신 복잡도 최적화라는 두 가지 상충하는 목표 사이의 트레이드-오프를 상세히 다루고 있다. 향후 연구는 결함 허용(Fault tolerance)의 구현과 클래식 검증자-단일 증명자 구조의 실현에 집중될 것으로 보인다.
