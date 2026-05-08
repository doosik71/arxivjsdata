# A Survey on Vision-Language-Action Models for Embodied AI

Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, Irwin King (2025)

## 🧩 Problem to Solve

본 논문은 물리적 세계에서 에이전트를 제어하여 과업을 수행하는 Embodied AI의 핵심 요소인 Vision-Language-Action (VLA) 모델에 관한 포괄적인 분석을 목표로 한다. 기존의 대화형 AI(예: ChatGPT)와 달리, Embodied AI는 환경과의 상호작용을 통해 물리적 실체를 제어해야 하며, 이는 특히 로보틱스 분야에서 매우 중요하다.

전통적인 심층 강화학습(Deep Reinforcement Learning) 접근 방식은 제어된 환경 내의 제한적인 과업(예: 물체 잡기)에 집중해 왔으나, 복잡하고 다양한 환경에서 일반화 가능한 다중 과업 정책(Multi-task Policy)에 대한 수요가 증가하고 있다. 특히 사용자의 자연어 지시를 이해하고, 시각적 환경을 인식하며, 이를 바탕으로 적절한 로봇 동작을 생성하는 언어 조건부 로봇 정책(Language-conditioned Robot Policy)의 개발이 필수적이다. 본 논문은 급격히 발전하는 VLA 모델의 지형을 체계적으로 정리하여 연구자들에게 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Embodied AI를 위한 VLA 모델의 첫 번째 종합 서베이로서, 다음과 같은 체계적인 분류 체계(Taxonomy)와 분석을 제시하는 것이다.

1. **VLA의 체계적 분류**: VLA 연구를 세 가지 주요 라인(개별 구성 요소, 저수준 제어 정책, 고수준 과업 플래너)으로 구조화하여 제시한다.
2. **모델 계층 구조 정의**: 고수준의 Task Planner가 복잡한 과업을 하위 과업(Subtasks)으로 분해하면, 저수준의 Control Policy가 구체적인 동작을 수행하는 계층적 프레임워크를 분석한다.
3. **리소스 요약**: VLA 학습 및 평가에 필요한 데이터셋, 시뮬레이터, 벤치마크를 광범위하게 정리하여 제공한다.
4. **미래 방향성 제시**: 안전성, 파운데이션 모델의 일반화, 실시간 응답성 등 현재 VLA가 직면한 도전 과제와 향후 연구 방향을 논의한다.

## 📎 Related Works

기존의 Embodied AI 관련 서베이들은 주로 로보틱스에서의 파운데이션 모델 전반이나 LLM의 적용 사례, 혹은 실세계 응용 분야에 초점을 맞추어 왔다. 반면, 본 논문은 시각, 언어, 동작의 세 가지 모달리티를 통합하여 직접적으로 동작을 생성하는 **VLA 모델 자체**에 집중함으로써 기존 문헌을 보완하고 확장한다.

특히 본 논문은 VLA를 "시각과 언어 입력을 처리하여 로봇 동작을 생성하는 모든 모델"로 광범위하게 정의하며, LLM/VLM 기반의 **Large VLA (LVLA)**와 일반적인 **Generalized VLA**를 구분하여 분석한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 VLA 시스템을 구성하는 요소를 저수준 제어와 고수준 계획의 계층적 구조로 나누어 설명한다.

### 1. VLA의 핵심 구성 요소 (Components)

VLA의 성능은 다음과 같은 개별 모듈의 역량에 의존한다.

- **Pretrained Visual Representations (PVRs)**: 환경의 상태($s_t$)를 인코딩한다. CLIP과 같은 대조 학습(Contrastive Learning) 기반 모델, MAE 기반의 자기지도 학습 모델(MVP), 그리고 DINOv2와 같은 자기 증류(Self-distillation) 모델 등이 사용된다.
- **Dynamics Learning**: 환경의 변화 원리를 학습한다.
  - Forward Dynamics: 현재 상태와 동작을 통해 다음 상태를 예측한다. $\hat{s}_{t+1} \leftarrow f_{fwd}(s_t, a_t)$
  - Inverse Dynamics: 두 상태 사이의 전이를 일으킨 동작을 예측한다. $\hat{a}_t \leftarrow f_{inv}(s_t, s_{t+1})$
- **World Models**: 상식적 지식을 인코딩하고 미래 상태를 예측하여 가상 공간에서 최적의 동작 시퀀스를 탐색하게 한다. $\hat{s}_{t+1} \sim P(\hat{s}_{t+1} | s_t, a_t)$
- **Reasoning**: Chain-of-Thought (CoT) 등의 기법을 통해 복잡한 의사결정 과정을 단계적으로 추론하여 성공률을 높인다.

### 2. 저수준 제어 정책 (Low-level Control Policies)

언어 지시 $p$와 시각적 상태 $s$를 입력받아 구체적인 동작 $\hat{a}_t$를 생성하는 정책 $\pi_\theta$이다.
$$\hat{a}_t \sim \pi_\theta(\hat{a}_t | p, s_{\le t}, a_{<t})$$

**주요 아키텍처 및 학습 방법**:

- **Non-Transformer**: CLIPort(CLIP+Transporter), BC-Z(FiLM layer) 등이 초기 모델로 활용되었다.
- **Transformer-based**: RT-1과 같이 시각 토큰과 언어 토큰을 입력받아 동작 토큰을 예측하는 구조가 주류를 이룬다.
- **Diffusion-based**: 동작 생성을 확률적 확산 과정으로 모델링하여 다봉 분포(Multimodal distribution)를 더 잘 처리한다.
- **Large VLA (LVLA)**: RT-2, OpenVLA와 같이 거대 VLM을 로봇 데이터로 파인튜닝하여 인터넷 규모의 지식을 로봇 제어에 전이시킨다.

**손실 함수 (Loss Functions)**:

- **Continuous BC**: 평균 제곱 오차(MSE)를 사용한다. $L_{Cont} = \sum_t \text{MSE}(a_t, \hat{a}_t)$
- **Discrete BC**: 동작 범위를 빈(bin)으로 나누어 교차 엔트로피(CE)를 사용한다. $L_{Disc} = \sum_t \text{CE}(a_t, \hat{a}_t)$
- **Diffusion Policy**: 노이즈 $\epsilon$을 예측하는 MSE 손실을 사용한다. $L_{DDPM} = \text{MSE}(\epsilon_k, \epsilon_\theta(a_{t+\epsilon_k}, k))$

### 3. 고수준 과업 플래너 (High-level Task Planners)

복잡한 과업 $\ell$을 하위 과업 시퀀스 $[p_1, p_2, \dots, p_N]$로 분해하는 역할을 수행한다.
$$[p_1, p_2, \dots, p_N] \sim \pi_\phi(\ell, s_t)$$

- **Monolithic Planners**: 단일 MLLM이 end-to-end로 계획을 생성한다. (예: PaLM-E)
- **Modular Planners**: 기존 LLM/VLM을 도구(Tool)처럼 조합하여 사용한다.
  - **Language-based**: 자연어 묘사를 통해 정보를 교환한다. (예: Inner Monologue)
  - **Code-based**: LLM이 API를 호출하는 코드를 생성하여 계획을 세운다. (예: Code as Policies)

## 📊 Results

본 논문은 개별 모델의 성능 수치보다는 VLA 생태계를 구성하는 리소스의 현황을 중심으로 결과를 제시한다.

### 1. 데이터셋 및 벤치마크

- **실세계 데이터셋**: Open X-Embodiment (OXE)와 같이 다양한 로봇 플랫폼의 데이터를 통합한 거대 데이터셋이 등장하여 일반화 성능을 높이고 있다.
- **시뮬레이터**: SAPIEN, AI2-THOR, Habitat 등이 활용되며, 이는 실세계 데이터의 희소성 문제를 해결하고 자동화된 평가를 가능하게 한다.
- **EQA (Embodied QA)**: 단순 동작 수행 외에 공간 추론 및 물리 이해 능력을 측정하는 벤치마크(예: OpenEQA)가 활용되고 있다.

### 2. 정성적 분석 및 경향성

- **Scaling Law**: LLM과 마찬가지로 모델 크기, 데이터의 질과 다양성이 증가함에 따라 로봇 제어 성능이 향상되는 경향이 확인되었다.
- **3D Vision의 효과**: 2D 이미지보다 Point Cloud나 Voxel 기반의 3D 정보가 정밀한 조작(Manipulation) 과업에서 더 우수한 성능을 보임을 분석하였다.

## 🧠 Insights & Discussion

### 강점 및 기회

- **지식 전이**: LVLA는 인터넷 규모의 시각-언어 데이터를 학습한 VLM의 능력을 로봇 제어로 전이시켜, 학습하지 않은 새로운 객체나 환경에서도 어느 정도의 제로샷(Zero-shot) 일반화 능력을 보여준다.
- **계층적 구조의 효율성**: 고수준 플래너는 추론 능력(Capacity)에, 저수준 정책은 실행 속도와 정밀도에 집중함으로써 복잡한 Long-horizon 과업을 효율적으로 해결할 수 있다.

### 한계 및 미해결 질문

- **Sim-to-Real Gap**: 시뮬레이션에서 학습된 모델이 실제 환경의 물리적 특성(비정형 물체, 유체 등)이나 렌더링 차이로 인해 성능이 급격히 저하되는 문제가 여전하다.
- **실시간성 (Real-time Responsiveness)**: 거대 모델(LVLA)의 추론 속도가 느려 동적인 환경 변화에 즉각적으로 대응하지 못하는 문제가 발생한다.
- **데이터 불일치**: 서로 다른 로봇의 제어 방식, 센서 사양 등의 불일치로 인해 데이터를 통합하여 학습시키는 데 어려움이 있다.

### 비판적 해석

본 논문은 광범위한 서베이를 제공하지만, 각 모델 간의 정량적 성능 비교 수치(Benchmark Table)가 부족하여 어떤 아키텍처가 특정 상황에서 절대적으로 우월한지 판단하기 어렵다. 또한, VLA의 핵심인 'Action'의 정의가 모델마다 상이하여(예: End-effector pose vs. Joint torque), 통일된 평가 지표의 부재가 향후 연구의 큰 병목이 될 것으로 보인다.

## 📌 TL;DR

본 논문은 시각, 언어, 동작을 통합하여 물리적 에이전트를 제어하는 **Vision-Language-Action (VLA) 모델**에 대한 최초의 종합 서베이이다. VLA를 **[구성 요소 $\rightarrow$ 저수준 제어 $\rightarrow$ 고수준 계획]**의 계층적 구조로 분류하여 분석하였으며, 특히 거대 VLM을 활용한 **Large VLA (LVLA)**의 가능성과 한계를 심도 있게 다루었다. 이 연구는 파편화되어 있던 로봇 학습 모델들을 체계적으로 정리함으로써, 향후 범용 로봇 파운데이션 모델(Robotic Foundation Models) 개발을 위한 이론적 기반과 리소스 지도를 제공한다는 점에서 매우 중요한 역할을 할 것으로 기대된다.
