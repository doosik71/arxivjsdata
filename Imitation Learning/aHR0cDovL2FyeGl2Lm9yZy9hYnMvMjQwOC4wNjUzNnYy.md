# A Comparison of Imitation Learning Algorithms for Bimanual Manipulation

Michael Drolet, Simon Stepputtis, Siva Kailas, Ajinkya Jain, Jan Peters, Stefan Schaal, and Heni Ben Amor (2024)

## 🧩 Problem to Solve

본 논문은 로봇의 양팔 조작(Bimanual Manipulation)을 위한 다양한 모방 학습(Imitation Learning, IL) 알고리즘들의 성능과 특성을 정밀하게 비교 분석하는 것을 목표로 한다.

양팔 조작은 인간과 유사한 정교한 도구 사용 및 무거운 물체 핸들링을 가능하게 하는 핵심 기술이다. 그러나 접촉이 빈번한(contact-rich) 환경에서의 정밀한 조작은 지각, 계획 및 제어 측면에서 매우 까다롭다. 특히, 최근 다양한 IL 알고리즘이 제안되었음에도 불구하고, 산업 현장과 유사한 고정밀 환경에서 각 알고리즘이 하이퍼파라미터 민감도, 학습 용이성, 데이터 효율성 및 성능 면에서 어떤 차이를 보이는지에 대한 체계적인 연구가 부족한 상태이다.

따라서 본 연구는 매우 좁은 공차(약 1mm)를 가진 양팔 핀 삽입(Peg Insertion) 작업이라는 도전적인 환경을 구축하고, 이를 통해 주요 IL 알고리즘들의 실질적인 한계와 이점을 분석하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다양한 철학적 배경을 가진 6가지 주요 IL 알고리즘을 동일한 고정밀 양팔 조작 환경에서 벤치마킹하여 그 특성을 정량적으로 분석한 것이다.

중심적인 설계 아이디어는 단순히 성공률만 측정하는 것이 아니라, **노이즈 내성(Noise Tolerance), 하이퍼파라미터 민감도(Hyperparameter Sensitivity), 계산 효율성(Compute Efficiency), 그리고 학습 안정성(Training Stability)**이라는 네 가지 핵심 지표를 통해 알고리즘을 다각도로 평가하는 것이다. 이를 통해 사용자가 특정 작업의 제약 조건(예: 계산 자원 제한, 데이터 부족, 환경 노이즈 등)에 따라 어떤 알고리즘을 선택해야 하는지에 대한 가이드를 제공한다.

## 📎 Related Works

기존의 양팔 조작 연구는 크게 두 가지 방향으로 진행되었다. 첫째는 고전적인 제어 기반 접근 방식과 모델 기반 계획법이며, 둘째는 강화 학습(RL) 기반의 접근 방식이다. RL은 새로운 전략을 발견할 수 있는 잠재력이 있으나, 보상 함수(Reward Function) 설계가 매우 어렵고 실제 환경에서 안전하지 않은 동작을 유발할 수 있다는 한계가 있다.

반면, 모방 학습(IL)은 명시적인 보상 함수 없이 전문가의 시연(Demonstration)을 통해 학습하므로 하드웨어 손상 위험이 적고 학습 효율이 높다. 최근 ALOHA와 같은 연구에서 ACT(Action Chunking Transformer)를 통해 복잡한 양팔 조작의 가능성을 보여주었으나, 본 논문은 여기서 더 나아가 기존의 기초적인 알고리즘(BC, DAgger)부터 최신 생성 모델 기반 알고리즘(Diffusion Policy, ACT) 및 에너지 기반 모델(IBC)까지 광범위하게 비교함으로써, 기존 연구들이 간과했던 하이퍼파라미터 영향성과 노이즈 강건성을 심층 분석한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 환경
본 연구는 두 개의 UR5 로봇 팔이 회전하는 토르소(Torso)에 장착된 시스템을 사용한다. 작업 목표는 4개의 구멍이 뚫린 어댑터(Dynamic adapter)를 4개의 핀이 있는 고정 어댑터(Stationary adapter)에 정밀하게 삽입하는 것이다. 구멍의 지름은 11mm, 핀의 지름은 10mm로, 단 $1\text{mm}$의 매우 좁은 공차를 가진 고정밀 작업이다.

### 2. 제어 프레임워크: Operational Space Controller (OSC)
학습 효율을 높이기 위해 관절 공간이 아닌 작업 공간(Task Space)에서 제어를 수행하는 OSC를 도입하였다. 어드미턴스 제어(Admittance Control)를 통해 외력 $\text{f}_{\text{ext}}$를 반영하며, 최종 제어 입력 $u$는 다음과 같은 방정식으로 계산된다.

$$u = J(q)^\top M_x(q) \tilde{x} + M(q) \tilde{\dot{q}} + g(q)$$

여기서 $J(q)$는 Jacobian 행렬, $M(q)$는 관성 행렬, $g(q)$는 중력 보상 항이며, $\tilde{x}$는 목표 위치와 현재 위치의 오차를 의미한다. 또한, 중복 자유도를 활용하기 위해 Nullspace filter를 적용하여 시스템의 안정성을 높였다.

### 3. 비교 대상 알고리즘
본 논문은 다음과 같은 서로 다른 특성을 가진 알고리즘들을 구현하였다.

*   **Behavioral Cloning (BC):** 전문가 데이터를 사용하여 상태 $s$에서 행동 $a$를 예측하는 가우시안 정책을 학습하는 지도 학습 방식이다. 목적 함수는 $\max_{\theta} \mathbb{E}_{(s,a)\sim\tau_E} [\log(\pi_\theta(a|s))]$로 정의된다.
*   **Action Chunking Transformer (ACT):** CVAE(Conditional Variational Autoencoder) 구조의 트랜스포머를 사용하여 단일 행동이 아닌 행동 시퀀스(Chunk)를 예측한다.
*   **Implicit Behavioral Cloning (IBC):** 에너지 기반 모델(EBM)을 사용하여 $\hat{a} = \arg\min_a E_\theta(s,a)$ 형태로 최적 행동을 찾는다. 이는 BC가 해결하지 못하는 다봉 분포(Multimodal distribution) 문제를 해결할 수 있다.
*   **Diffusion Policy:** 노이즈로부터 행동을 점진적으로 복원하는 디퓨전 프로세스를 사용하며, U-Net 아키텍처를 통해 행동 시퀀스를 생성한다.
*   **Generative Adversarial Imitation Learning (GAIL):** 생성자(정책)와 판별자가 서로 경쟁하는 GAN 구조를 통해 전문가의 상태-행동 분포를 모방하며, TRPO 알고리즘을 사용하여 정책을 업데이트한다.
*   **DAgger:** 정책이 실행 중에 마주치는 상태에 대해 전문가(Oracle)에게 정답 레이블을 요청하여 데이터셋을 확장함으로써 공변량 변화(Covariate Shift) 문제를 해결한다.

### 4. 학습 및 실험 절차
*   **상태 공간(Observation Space):** 36차원 (전문가 포즈와의 차이, 핀과의 거리, 그리퍼의 힘/토크 센서 값 등).
*   **행동 공간(Action Space):** 18차원 (양팔의 $\Delta$ 위치 및 $\Delta$ 회전).
*   **평가 환경:** 노이즈 수준에 따라 Zero Noise, Low Noise, High Noise의 세 가지 환경으로 구분하여 실험을 진행하였다.

## 📊 Results

### 1. 정량적 성능 분석
실험 결과, **Diffusion Policy, ACT, GAIL**이 모든 환경에서 높은 보상과 성공률을 기록하며 가장 우수한 성능을 보였다. 반면 BC와 IBC는 노이즈가 증가함에 따라 성능이 급격히 저하되는 모습을 보였다.

### 2. 노이즈 강건성 (Noise Tolerance)
노이즈가 존재하는 환경에서 **ACT와 Diffusion Policy**, 그리고 환경과 상호작용하는 **GAIL과 DAgger**가 매우 강한 내성을 보였다. 특히 ACT와 Diffusion Policy의 'Chunking(시퀀스 예측)' 전략이 비마르코프적(non-Markovian) 환경 특성을 극복하는 데 효과적임이 확인되었다.

### 3. 하이퍼파라미터 민감도 및 효율성
*   **민감도:** GAIL은 성능은 좋으나 하이퍼파라미터 변화에 매우 민감하여 최적의 설정을 찾는 것이 어려웠다.
*   **계산 효율성:** GAIL은 학습에 약 2일이 소요될 정도로 계산 비용이 매우 높았다. 반면 ACT와 Diffusion Policy는 상대적으로 짧은 시간 내에 수렴하며 높은 성능을 달성하였다.

### 4. 종합 지표 (Figure 4 기준)
*   **Diffusion Policy:** 성능, 노이즈 내성, 학습 안정성 모든 면에서 가장 균형 잡힌 최적의 선택지로 나타났다.
*   **ACT:** Diffusion과 유사하게 매우 우수한 성능과 효율성을 보였다.
*   **GAIL/DAgger:** 환경 상호작용의 이점이 있으나, 계산 비용과 설정의 어려움이 크다.

## 🧠 Insights & Discussion

본 논문은 고정밀 양팔 조작 작업에서 **행동 청킹(Action Chunking)**과 **생성 모델(Generative Models)**의 도입이 결정적인 성능 향상을 가져온다는 점을 시사한다. 단순한 회귀 기반의 BC는 데이터 분포 밖의 상태에서 쉽게 무너지지만, Diffusion Policy와 ACT는 시퀀스 단위의 예측을 통해 더 매끄럽고 강건한 제어를 가능하게 한다.

또한, GAIL과 DAgger 같은 상호작용 기반 알고리즘이 노이즈에 강한 이유는 학습 과정에서 다양한 상태를 탐색하며 전문가의 가이드를 직접 받기 때문이다. 하지만 이는 막대한 계산 시간과 데이터 수집 비용을 초래한다. 따라서 실용적인 관점에서는 오프라인 데이터만으로도 유사한 강건성을 확보할 수 있는 Diffusion Policy나 ACT가 훨씬 유리하다는 결론을 내릴 수 있다.

한 가지 논의할 점은, 본 실험이 MuJoCo 시뮬레이션 환경에서 진행되었다는 것이다. 실제 물리 환경에서는 센서 노이즈의 특성이 다르며, 하드웨어의 마찰이나 백래시(backlash) 같은 비선형 요소가 더 강하게 작용하므로, 시뮬레이션에서의 결과가 실제 환경으로 그대로 전이될지는 추가적인 연구가 필요하다.

## 📌 TL;DR

본 연구는 고정밀 양팔 핀 삽입 작업을 통해 6가지 주요 모방 학습 알고리즘을 체계적으로 비교 분석하였다. 실험 결과, **Diffusion Policy와 ACT**가 성능, 노이즈 강건성, 학습 효율성 측면에서 가장 우수하며, 특히 행동 시퀀스를 예측하는 Chunking 기법이 복잡한 조작 작업의 안정성을 크게 높인다는 것을 입증하였다. 이 결과는 향후 정밀 로봇 조작 시스템 설계 시, 계산 자원과 데이터 가용성에 따라 어떤 알고리즘을 선택해야 하는지에 대한 중요한 공학적 근거를 제공한다.