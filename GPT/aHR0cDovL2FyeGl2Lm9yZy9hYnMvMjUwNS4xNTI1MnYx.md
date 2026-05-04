# An Efficient Private GPT Never Autoregressively Decodes

Zhengyi Li, Yue Guan, Kang Yang, Yu Feng, Ning Liu, Yu Yu, Jingwen Leng, Minyi Guo (2025)

## 🧩 Problem to Solve

본 논문은 생성형 사전 학습 트랜스포머(GPT) 모델을 배포할 때 발생하는 클라이언트와 서버 간의 프라이버시 문제를 해결하고자 한다. 특히, 모델 소유자는 모델 가중치를 보호하고 클라이언트는 입력 데이터를 보호해야 하는 상황에서 보안 2자 계산(Secure Two-Party Computation, 2PC) 기술이 사용된다.

동형 암호(Homomorphic Encryption, HE)와 다자간 계산(Multi-Party Computation, MPC)과 같은 암호학적 기본 요소(Cryptographic Primitives)를 통해 프라이버시를 보존하는 추론이 가능하지만, 이는 막대한 계산 오버헤드와 통신 비용을 초래한다. 특히 선형 층(Linear Layer)의 행렬 곱셈과 Softmax, GELU와 같은 복잡한 비선형 층은 많은 수의 통신 라운드를 요구하며, 이는 전체 추론 속도를 심각하게 저하시킨다. 따라서 본 연구의 목표는 프라이버시 수준과 생성 품질을 유지하면서도 보안 GPT 추론의 효율성을 획기적으로 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **보안 디코딩 시 입력 길이의 변화가 지연 시간(Latency)에 거의 영향을 주지 않는다**는 관찰에서 출발한다. 이를 바탕으로 저자들은 **POST(Public decOding and Secure verificaTion)**라는 접근 방식을 제안한다.

POST의 중심 설계 직관은 다음과 같다. 공개적으로 사용 가능한 GPT 모델(Public Model)이 비공개 모델(Private Model)과 어느 정도의 언어적, 논리적 지식을 공유한다는 점을 이용하는 것이다. 클라이언트가 공개 모델을 사용하여 여러 개의 초안 토큰(Draft Tokens)을 먼저 생성하고, 서버의 비공개 모델은 이 토큰들을 한 번의 보안 포워드(Secure Forward) 단계에서 검증함으로써 전체 디코딩 단계 수를 줄이는 방식이다. 이는 표준적인 자동 회귀(Autoregressive) 방식의 보안 디코딩이 매 단계 단 하나의 토큰만 생성하는 비효율성을 극복하려는 시도이다.

## 📎 Related Works

기존의 보안 GPT 추론 연구들은 주로 두 가지 방향으로 진행되었다. 첫째는 암호학적 프로토콜 자체를 최적화하여 통신 및 계산 비용을 줄이는 방식이고, 둘째는 암호학적 연산에 유리하도록 GPT 아키텍처를 수정(예: 비선형 함수 근사)하는 방식이다. 그러나 이러한 방법들은 여전히 상당한 오버헤드가 존재하며, 아키텍처 수정의 경우 모델 정확도 손실이 발생할 수 있다.

또한, 일반적인 LLM 추론 가속화 기술인 **Speculative Decoding**이 존재한다. 이는 작은 모델이 초안을 만들고 큰 모델이 이를 검증하는 방식이다. 하지만 기존의 Speculative Decoding은 암호화된 환경에서의 연산 특성(특히 입력 길이에 따른 지연 시간의 둔감함)을 고려하지 않았으며, 보안 환경에서 Speculative Sampling을 효율적으로 구현하는 프로토콜에 대한 연구가 부족했다.

## 🛠️ Methodology

### 1. POST 전체 파이프라인
POST는 크게 오프라인 단계와 온라인 단계로 구성된다.

- **오프라인 단계**: 공개 모델과 비공개 모델 간의 출력 분포를 맞추기 위해 **지식 증류(Knowledge Distillation)**를 통한 모델 정렬(Model Alignment)을 수행한다.
- **온라인 단계**:
    1. 클라이언트는 공개 모델 $\mathcal{M}_{pub}$을 사용하여 $\gamma$개의 초안 토큰 $[x_1, \dots, x_\gamma]$를 자동 회귀적으로 샘플링한다.
    2. 클라이언트와 서버는 비공개 모델 $\mathcal{M}_{pri}$를 통해 이 초안 토큰들을 한 번에 처리하여 보안 분포 $\langle p(x|x_{<t}) \rangle, \dots, \langle p(x|x_{<t+\gamma}) \rangle$를 생성한다.
    3. 보안 검증(Secure Verification) 과정을 통해 수용 가능한 토큰을 결정하고, 첫 번째 거절된 토큰에서 보너스 토큰을 샘플링한다.

### 2. 보안 검증 및 Speculative Sampling
단순한 일치(Hard Matching)는 수용률을 낮추므로, 본 논문은 **Speculative Sampling**을 도입하여 '소프트 매칭'을 구현한다. 토큰 $x_i$가 거절될 확률은 다음과 같이 정의된다:
$$\max(0, 1 - \frac{p(x_i|x_{<t+i})}{q(x_i|x_{<t+i})})$$
여기서 $p$는 비공개 모델, $q$는 공개 모델의 확률 분포이다.

암호화 환경에서 나눗셈과 비교 연산은 매우 느리므로, 저자들은 이를 최적화한 프로토콜을 제안한다:
- **나눗셈의 곱셈화**: 나눗셈을 피하기 위해 클라이언트가 랜덤 행렬 $R^{mul}$을 생성하여 하다마르 곱(Hadamard multiplication)을 수행하고, 이를 통해 점수 행렬 $\langle S \rangle = \langle Q \cdot R^{mul} \rangle - P \pmod{2^\ell}$를 구하여 0과의 비교로 문제를 치환한다.
- **선택 후 비교(Selection then Comparison)**: 모든 요소의 부호 비트를 계산하는 대신, Oblivious Transfer(OT)를 통해 필요한 초안 토큰의 요소만 선택적으로 추출하여 비교함으로써 통신 복잡도를 $O(V \cdot \ell)$ 수준으로 낮춘다.

### 3. 모델 정렬 (Model Alignment)
공개 모델과 비공개 모델의 괴리를 줄여 수용률을 높이기 위해 지식 증류를 사용한다. 공개 데이터셋을 이용하여 다음과 같은 교차 엔트로피(Cross Entropy) 손실 함수를 최소화함으로써 공개 모델을 정렬한다:
$$\ell(x^{(i)}) = \sum_{t=1}^{n_i} D(p(y_t^{(i)}|x^{(i)}, y_{<t}^{(i)}) \parallel q(y_t^{(i)}|x^{(i)}, y_{<t}^{(i)}))$$
여기서 $D(p \parallel q) = -\sum p \log q$이다.

## 📊 Results

### 실험 설정
- **모델 쌍**: (Vicuna-7B, LLaMA-68M/160M), (FLAN-T5-XL, T5-efficient-small/base), (FLAN-T5-XL, FLAN-T5-small/base).
- **작업**: Text-to-SQL (Spider), 수학 (Gsm8k), 파이썬 코드 생성, 금융 질의응답.
- **네트워크 조건**: LAN (1 Gbps, 10ms) 및 WAN (400 Mbps, 40ms).
- **비교 대상**: 표준 보안 디코딩(Standard Secure Decoding).

### 주요 결과
- **가속 성능**: 네트워크 조건과 모델 쌍에 따라 표준 보안 디코딩 대비 **$2.1\times \sim 6.0\times$의 속도 향상**을 달성하였다.
- **수용률 향상**: 모델 정렬(AL)을 통해 수용률이 크게 증가하였다. 예를 들어, Vicuna-7B와 LLaMA-68M 쌍의 경우 수용률이 $0.24$에서 $0.61$로 상승하였다.
- **정렬 효율성**: 약 1백만 개의 토큰만으로도 충분한 정렬 효과를 얻었으며, 이는 실제 API 비용으로 환산 시 약 10달러 수준으로 실용적임을 보였다.
- **검증 오버헤드**: 최적화된 보안 샘플링 프로토콜은 기존의 Naive 방식보다 지연 시간을 약 10배 감소시켜, 전체 시스템 지연 시간에 미치는 영향이 무시할 수 있는 수준임을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 학술적 가치는 **보안 추론 환경에서의 지연 시간이 입력 길이에 둔감하다(Insensitivity)**는 점을 분석하고 이를 시스템 설계에 활용했다는 점이다.

저자들은 이 현상의 원인을 세 가지로 분석한다:
1. **One-way Delay**: 통신 라운드 수가 입력 길이에 독립적이므로, 고정적인 네트워크 지연 시간이 전체 시간의 상당 부분을 차지한다.
2. **Computation Time**: HE의 SIMD(Single Instruction Multiple Data) 특성상, 입력 길이가 짧을 때는 폴리노미얼 링의 슬롯이 충분히 활용되지 않는다. 따라서 입력 길이를 어느 정도 늘려도 계산 효율이 증가하여 지연 시간 증가분이 상쇄된다.
3. **Transmission Time**: MPC에서는 선형적으로 증가하지만, HE에서는 SIMD 패킹 덕분에 하위 선형(sub-linear)으로 증가하여 영향이 제한적이다.

한계점으로는 공개 모델의 크기가 커질수록 수용률이 높아지지만, 그에 따라 클라이언트 측의 연산 부담이 늘어날 수 있다는 점이 있다. 또한, 본 연구는 반-정직(Semi-honest) 위협 모델을 가정하였으므로, 악의적인(Malicious) 공격자에 대한 보안성은 추가적인 논의가 필요하다.

## 📌 TL;DR

이 논문은 보안 GPT 추론 시 입력 길이에 따른 지연 시간 증가가 매우 적다는 점을 발견하고, 이를 이용해 **공개 모델이 초안을 생성하고 비공개 모델이 이를 일괄 검증하는 POST 프레임워크**를 제안한다. 지식 증류를 통한 모델 정렬과 최적화된 보안 샘플링 프로토콜을 통해 프라이버시 손실 없이 **$2.1\times \sim 6.0\times$의 속도 향상**을 이루었으며, 이는 향후 보안 LLM 서비스의 실용성을 높이는 데 중요한 기여를 할 것으로 보인다.