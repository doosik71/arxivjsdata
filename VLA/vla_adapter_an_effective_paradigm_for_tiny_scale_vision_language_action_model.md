# VLA-ADAPTER: AN EFFECTIVE PARADIGM FOR TINY-SCALE VISION-LANGUAGE-ACTION MODEL

Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu, Donglin Wang

## 🧩 Problem to Solve

기존 Vision-Language-Action (VLA) 모델은 로봇 데이터를 기반으로 대규모 Vision-Language Model (VLM)을 사전 학습하여 인지 및 행동 공간을 연결합니다. 이 방식은 성능 향상에 기여하지만, 막대한 훈련 비용, 대규모 VLM 의존성, 느린 미세 조정 속도, 높은 GPU 메모리(VRAM) 소비, 낮은 추론 효율성 등의 병목 현상을 야기합니다. 본 논문은 이러한 문제점을 해결하고, 시각-언어(VL) 표현과 행동(A) 간의 간극을 효과적이고 효율적으로 연결하는 새로운 패러다임을 제안합니다.

## ✨ Key Contributions

- **VLA 브리징 패러다임에 대한 체계적인 분석:** 행동 생성에 대한 다양한 VL 조건의 영향을 체계적으로 분석하고 VLA 모델 설계에 중요한 핵심 발견들을 제시합니다.
- **VLA-Adapter 제안:** 대규모 VLM과 광범위한 사전 훈련에 대한 의존도를 줄이도록 설계된 새로운 경량 브리징 패러다임을 도입합니다.
- **경량 정책 모듈 및 Bridge Attention:** 최적의 VL 조건을 행동 공간에 자율적으로 주입하는 경량 Policy 모듈과 Bridge Attention을 제안합니다.
- **탁월한 성능 및 효율성:** 단 0.5B 파라미터 백본과 로봇 데이터 사전 훈련 없이 SOTA(State-Of-The-Art) 수준의 성능을 달성하며, 기존 방법론 대비 3배 빠른 추론 속도를 제공합니다.
- **낮은 배포 장벽:** 단일 소비자용 GPU로 8시간 만에 강력한 VLA 모델 훈련을 가능하게 하여 VLA 모델 배포의 장벽을 크게 낮춥니다.

## 📎 Related Works

- **VLA 모델:** 사전 훈련된 VLM을 활용하여 로봇 제어 및 다양한 일상 작업을 수행하는 모델들이 주류 연구 분야로 부상했습니다. 이들은 대규모 로봇 데이터셋(예: Open X-Embodiment)으로 VLM을 사전 훈련하고 Policy 네트워크와 통합하여 end-to-end 방식으로 행동 시퀀스를 생성합니다.
- **이중 시스템 VLA 아키텍처:** VLM과 Policy를 연결하는 중간 잠재 토큰을 도입하고 비동기 메커니즘을 사용하여 두 시스템 간의 조정을 강화하여 행동 생성 시 지연 문제를 완화합니다.
- **인지에서 행동 공간으로의 브리징:**
  - **이전 연구 (불연속 행동 공간):** 초기 연구에서는 행동을 토큰으로 이산화하여 인지-행동 공간을 직접 정렬하려 했으나, 이산화로 인한 본질적인 손실이 발생했습니다.
  - **최근 연구 (연속 행동 공간):** 연속 행동 공간으로 초점을 전환하며, 사용되는 지각 피처 유형에 따라 다음과 같이 분류됩니다.
    - **VLM의 Raw Features:** VLM의 최종 계층 또는 중간 계층에서 시각 및 언어 표현을 직접 추출합니다 (예: RoboVLMs, GR00T N1, $\pi_0$). 중간 계층 피처가 더 풍부한 다중 모달 정보를 보유할 수 있다고 주장합니다.
    - **Additional Query as Interface:** Raw 피처 대신 추가 학습 가능한 쿼리를 VLM과 Policy 사이의 브릿지로 사용하여 다중 모달 정보를 통합하고 더 나은 성능을 보여줍니다 (예: OpenVLA-OFT).

## 🛠️ Methodology

### 3.1 Preliminary

VLA-Adapter는 Prismatic-VLMs 아키텍처를 따르며 $M$개의 레이어로 구성됩니다. 시점 $t$에서 VLM의 입력은 $\{X_v_t, X_g_t, L_t, AQ_t\}$ (3인칭 이미지 $X_v_t$, 그리퍼 이미지 $X_g_t$, 명령 $L_t$, 추가 ActionQuery $AQ_t$)입니다. DINOv2와 SigLIP을 통해 시각 임베딩을 추출하고 $L_t$를 토큰화합니다. VLM의 출력은 특정 계층의 Raw 잠재 표현 $C_R_t$와 ActionQuery 잠재 표현 $C_AQ_t$이며, 이들이 Policy의 조건으로 사용됩니다. 기본 백본은 Qwen2.5-0.5B입니다.

### 3.2 Which Condition Is Essential for Bridging from VL to A?

행동 생성에 어떤 지각 정보가 필수적인지 체계적으로 탐색합니다.

- **Question 1.1:** VLM 내 어떤 계층의 피처가 Policy 네트워크에 더 효과적인가?
- **Question 1.2:** ActionQuery 피처가 Raw 피처보다 더 나은 선택인가?

**주요 발견 (Key Findings):**

1. **Raw 잠재 표현 ($C_R_t$)에 관하여:** 중간 계층의 $C_R_t$가 깊은 계층의 $C_R_t$보다 성능이 우수합니다. 깊은 계층의 $C_R_t$는 의미 정보에 편향되어 행동 생성에 덜 효과적인 반면, 중간 계층의 $C_R_t$는 이미지와 텍스트 정보를 효과적으로 통합하고 풍부한 다중 모달 세부 정보를 유지하여 행동 생성을 촉진합니다.
2. **ActionQuery 잠재 표현 ($C_AQ_t$)에 관하여:** 깊은 계층의 $C_AQ_t$가 다른 계층보다 성능이 우수합니다. ActionQuery는 처음부터 훈련되며, 깊은 계층의 $C_AQ_t$는 더 풍부한 다중 모달 세부 정보를 통합하여 얕은 계층보다 행동 생성을 효과적으로 촉진합니다.
3. **다중 계층 피처의 우수성:** 단일 계층보다 모든 계층 피처를 사용하는 것이 전반적으로 성능이 우수하며, 설계 시 최적 계층 선택 시간도 절약됩니다.

**조건 결정:** 모든 계층의 $C_AQ_t$가 $C_R_t$보다 우수하지만, 일부 어려운 작업에서는 중간 계층의 $C_R_t$가 뛰어난 성능을 보입니다. 따라서 $C_R_t$의 특정 지식을 활용하여 성능을 더욱 향상시키는 것을 목표로 합니다.

### 3.3 Policy with Bridge Attention

- **전반적인 구조:** 모델의 단순화를 위해 L1 기반 Policy 네트워크를 설계했습니다. $t$ 시점의 Policy 입력은 $\{C_R_t, C_AQ_t, A^0_t, P_t\}$ (Raw 및 ActionQuery 잠재 표현, H-단계 초기 행동 $A^0_t$, 고유 감각 상태 $P_t$)입니다. Policy의 각 계층은 Bridge Attention 모듈과 FFN(Feed-Forward Network)으로 구성됩니다.
- **Bridge Attention:** $C_R_t$ 및 $C_AQ_t$ 조건을 통해 행동 생성을 최대한 유도하는 것을 목표로 합니다. 각 Bridge Attention은 두 개의 Cross-Attention과 하나의 Self-Attention으로 구성됩니다.
  - 첫 번째 Cross-Attention에서는 $C_R_t$를 MLP $\sigma_1$을 통해 $K_1, V_1$로 변환하고, 행동 잠재 표현 $e_A^{\tau}_t$를 $Q_1$으로 사용하여 $CA_1(e_A^{\tau}_t, \sigma_1(C_R_t))$를 얻습니다.
  - 두 번째 Cross-Attention에서는 $C_AQ_t$를 $\sigma_0(P_t)$와 연결하고 MLP $\sigma_2$를 통해 $K_2, V_2$로 변환하며, $e_A^{\tau}_t$를 $Q_2$로 사용하여 $CA_2(e_A^{\tau}_t, \sigma_2[C_AQ_t, \sigma_0(P_t)])$를 얻습니다.
  - Self-Attention은 $e_A^{\tau}_t$를 $Q, K, V$로 사용하여 $SA(e_A^{\tau}_t, e_A^{\tau}_t)$를 수행합니다.
  - $C_R_t$의 선택적 주입을 위해 학습 가능한 파라미터 Ratio $g$를 도입하여 $CA_1$의 영향을 조절합니다. $g$는 0으로 초기화되며 $\tanh(g) \in [-1, 1]$로 안정성을 확보합니다. 세 가지 Attention은 연결되어 $b_A^{\tau}_t$를 형성합니다:
    $$b_A^{\tau}_t = [CA_1(e_A^{\tau}_t,\sigma_1(C_R_t)) \cdot \tanh(g), CA_2(e_A^{\tau}_t,\sigma_2[C_AQ_t,\sigma_0(P_t)]), SA(e_A^{\tau}_t,e_A^{\tau}_t)]$$
  - $b_A^{\tau}_t$는 잔차 FFN을 거쳐 $e_A^{\tau+1}_t$를 얻고, 최종적으로 LN과 MLP를 통해 행동 청크 $A_M-1_t$를 생성합니다.
- **Policy 아키텍처 선택:** DiT 기반 Policy도 설계했으나, L1 기반 Policy가 전반적으로 성능과 추론 속도 면에서 더 우수함을 확인하여 L1 아키텍처를 채택했습니다.

### 3.4 Training

VLA-Adapter는 Policy를 처음부터 훈련하며, end-to-end 방식으로 훈련됩니다. 훈련 목표는 Ground Truth 행동 궤적 $A_t$와 행동 잠재 표현 $A^{\tau}_t$를 사용하여 다음 목적 함수를 최소화하는 것입니다:
$$\min_{\theta} J (\theta) = \mathbb{E}_{A_t, C_R_t, C_AQ_t, \sigma_0(P_t), \tau} \left\| \pi_{\theta}(A^{\tau}_t, C_R_t, C_AQ_t, \sigma_0(P_t), \tau) - A_t \right\|_1$$
AdamW 옵티마이저와 LoRA 스키마를 사용하며, 학습률은 1e-4, 코사인 어닐링 스케줄러를 적용합니다.

## 📊 Results

- **VLA-Adapter의 필요성 (LIBERO-Long):**

  - 로봇 사전 훈련 없는 VLM 사용 시, VLA-Adapter는 OpenVLA-OFT 대비 성능 향상이 뚜렷합니다 (B1 85.8% $\rightarrow$ 95.0%, B2 87.5% $\rightarrow$ 95.2%).
  - 백본이 고정된 경우에도 VLA-Adapter는 강한 성능을 보입니다 (OpenVLA-OFT 0.0%, SmolVLA 77.0% $\rightarrow$ VLA-Adapter 86.4%). 이는 VLA-Adapter가 로봇 사전 훈련 없이 효율적인 미세 조정을 가능하게 함을 보여줍니다.
  - **효율성:** VLA-Adapter는 OpenVLA-OFT 대비 3배 빠른 추론 속도 (Throughput 219.2Hz vs 71.4Hz)와 낮은 지연 시간 (Latency 0.0365s vs 0.1120s)을 달성합니다.

- **다양한 작업에서의 전체 성능 (LIBERO):**

  - LIBERO 벤치마크 (Spatial, Object, Goal, Long)에서 0.5B 백본으로 OpenVLA-OFT (7B)와 비슷한 수준의 SOTA 성능 (평균 97.3%)을 달성합니다.
  - 동일 스케일 백본을 사용하는 VLA-OS 대비 LIBERO-Long에서 29.0%p 높은 성능 (95.0% vs 66.0%)을 보입니다.
  - $\pi_0$, SmolVLA, GR00T N1과 같은 대표적인 경량 모델을 능가합니다.

- **일반화 작업 성능 (CALVIN ABC→D):**

  - 제로샷 일반화 작업 (ABC→D)에서 강력한 일반화 능력을 보입니다 (평균 길이 4.42).
  - VPP (4.33) 및 Seer Large (4.28)와 같은 SOTA 모델보다 우수하거나 동등한 성능을 달성합니다.

- **실세계 작업 성능:**

  - 단순 Pick-and-Place, CALVIN 기반 어려운 조작 작업 (블록 옮기기, 블록 쌓기), LIBERO 기반 복합 장기 작업 등 4가지 범주의 실제 로봇 작업에서 ACT 및 OFT-style 변형보다 뛰어난 일반화 능력을 보여줍니다 (평균 성공률 90%).

- **Ablation 실험 (LIBERO-Long):**
  - **ActionQuery 수:** 64개의 ActionQuery 토큰이 성능과 효율성 사이의 최적의 균형을 제공합니다 (95.0% 성공률). 너무 적으면 다중 모달 통합이 약화되고, 너무 많으면 중복이 발생합니다.
  - **조건 유형:** 모든 계층의 Raw 및 ActionQuery를 모두 사용하는 것이 단일 조건 유형보다 월등히 뛰어난 성능을 달성합니다 (95.0% 성공률 vs 85.8%, 90.2%, 88.4%, 90.6%, 92.6%). 이는 본 논문의 브리징 패러다임의 우월성을 간접적으로 입증합니다.
  - **Policy에 대한 주입 정도:** $C_R_t$에 대해 학습 가능한 $\tanh(g)$를, $C_AQ_t$에 대해 1을 주입하는 방식이 가장 좋은 성능을 보입니다 (95.0%). $C_R_t$의 성능이 $C_AQ_t$보다 열등하므로, $C_R_t$는 학습을 통해 효과적인 정보를 Policy에 주입해야 하며, $C_AQ_t$는 다중 모달 정보를 집계하므로 완전히 주입되어야 합니다.

## 🧠 Insights & Discussion

VLA-Adapter는 대규모 VLM 및 막대한 훈련 비용에 대한 VLA 모델의 의존성을 성공적으로 완화합니다. 특히, 로봇 사전 훈련이 없는 VLM을 사용하거나 백본이 고정된 상황에서도 뛰어난 성능과 효율성을 보여줌으로써 VLA 모델의 배포 장벽을 크게 낮춥니다. 체계적인 조건 분석을 통해 중간 계층 Raw 피처와 깊은 계층 ActionQuery 피처의 중요성을 밝히고, 이들을 Bridge Attention에서 효과적으로 통합하는 방법을 제안한 것이 핵심 성공 요인입니다.

**한계점:**

1. **실세계 일반화:** 대규모 로봇 데이터 사전 훈련이 없고 모델 스케일이 작기 때문에, 실세계 시스템에서의 일반화 성능은 개선될 여지가 있습니다.
2. **조건의 품질 및 활용:** Policy 네트워크가 생성하는 행동의 품질은 VLM이 제공하는 조건과 이러한 조건이 어떻게 사용되는지에 달려있습니다. 향후 연구에서는 이러한 조건의 표현을 개선하고 효율적인 활용 방안을 탐색할 수 있습니다.
3. **훈련 과정의 복잡성:** VLA-Adapter의 기본적인 훈련 과정은 상대적으로 단순하므로, 강화 학습과 같은 복합적인 과정을 탐색하여 성능을 더욱 향상시킬 수 있습니다.

## 📌 TL;DR

VLA-Adapter는 기존 VLA 모델의 대규모 VLM 의존성과 높은 훈련 비용 문제를 해결하기 위해 제시된 효율적인 브리징 패러다임입니다. 이 방법은 Raw 및 ActionQuery 잠재 표현을 활용하여 다중 모달 지식을 Policy에 효과적으로 전달합니다. 결과적으로, 0.5B 백본만으로 SOTA 수준의 성능과 로봇 데이터 사전 훈련 없는 효율적인 미세 조정을 가능하게 하며, 3배 빠른 추론 속도와 낮은 VRAM 사용량으로 VLA 모델의 배포 장벽을 획기적으로 낮춥니다.
