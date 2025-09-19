# Refined Policy Distillation: From VLA Generalists to RL Experts

Tobias Jülg, Wolfram Burgard and Florian Walter

## 🧩 Problem to Solve

- Vision-Language-Action (VLA) 모델은 실제 환경에서 뛰어난 일반화 능력을 보였지만, 전문가 정책(expert policies)에 비해 성공률이 낮고, 환경 설정 변경 시 미세 조정(fine-tuning)이 필요하다는 한계가 있습니다.
- 강화 학습(RL)은 확장성이 뛰어나고 더 작은 정책을 생성할 수 있지만, 샘플 비효율성(sample-inefficiency)과 하이퍼파라미터 민감성으로 인해 로봇에 직접 적용하기 어렵습니다.
- 기존 정책 증류(Policy Distillation) 방법은 주로 RL 정책 간의 증류에 초점을 맞추어, VLA와 같은 일반화 정책을 효율적으로 고성능의 전문가 정책으로 전환하는 데 적합하지 않습니다.

## ✨ Key Contributions

- 대규모 VLA에서 온-폴리시(on-policy) RL을 통해 소형의 작업별 전문가 정책을 증류하고 개선(refine)하는 새로운 방법인 RPD(Refined Policy Distillation)를 제안합니다.
- 밀집 보상(dense reward) 및 희소 보상(sparse reward) 설정 모두에서 6가지 VLA 분포 내(in-distribution) 작업에 대해 RPD의 샘플 효율성 증가와 안정성을 입증했습니다.
- 두 가지 분포 외(out-of-distribution) 작업 변형과 변경된 카메라 시점에 대한 일반화 성능을 시연했습니다.
- ManiSkill3에 맞게 미세 조정된 Octo 및 OpenVLA 버전을 사용하여 현재 VLA가 시뮬레이션에서 어떻게 작동하는지에 대한 통찰력을 제공합니다.

## 📎 Related Works

- **일반화 로봇 정책 (Generalist Robot Policies)**: 대규모 데이터셋(예: Open X-Embodiment)으로 훈련된 VLM(Vision-Language Model) 백본을 미세 조정하거나, 비전 및 언어 입력 또는 목표 이미지 지침으로 처음부터 학습된 대규모 트랜스포머 기반 모델(예: Octo [4], OpenVLA [7], RT-1 [2], RT-2 [3])을 포함합니다.
- **정책 증류 (Policy Distillation, PD)**: RL에서 훈련 속도를 높이고 정책 복잡성을 줄이기 위해 교사 정책(teacher policy)을 학생 정책(student policy)으로 증류하는 광범위한 방법입니다 [19]. KL 발산(divergence) 또는 MSE 손실을 사용하여 Q-함수 또는 정책 분포를 맞추는 방식이 연구되었습니다 [20], [21], [22].
- **프록시멀 정책 증류 (Proximal Policy Distillation, PPD)**: PPO 교사 정책을 PPO 학생 정책으로 증류하는 최근 연구 [23]로, 교사와 학생 정책 분포 간의 KL 발산을 PPO 손실에 추가합니다. 본 논문에서는 VLA의 행동 분포 샘플링의 계산 비용과 다중 모드 분포의 KL 발산 계산의 어려움을 지적합니다.
- **행동 복제 (Behavioral Cloning, BC)**: 오프라인 모방 학습(imitation learning)의 총칭으로, 전문가 시연 데이터셋을 기반으로 정책을 학습하는 것을 목표로 합니다 [24]. RL과 결합하여 고정된 사전 기록된 데이터셋을 통해 에이전트의 탐색 속도를 높이는 데 사용되기도 합니다 [25], [26].
- **RL 데이터 생성 (RL-generated data)**: Xu et al. [10]은 오프라인 SAC(Soft Actor-Critic) 변형을 사용하여 VLA 미세 조정을 위한 데이터셋 생성을 연구했으나, 이는 이미 훈련된 RL 전문가 정책을 필요로 합니다. RPD는 스크래치부터 RL 정책을 훈련하지 않고 대규모 파운데이션 모델을 소규모 온-폴리시 RL 에이전트로 증류합니다.

## 🛠️ Methodology

RPD는 VLA 정책의 일반적인 작업 지식을 소규모의 빠르고 작업별 전문가 정책으로 증류하는 방법으로, PPO(Proximal Policy Optimization) RL 에이전트와 BC(Behavioral Cloning) 목적 함수를 결합합니다.

1. **Objective Function Formulation**:
   - RPD의 핵심은 PPO의 표준 클립된 대리 목적 함수($L_{\text{RL}}(\theta)$)에 행동 복제(BC) 항을 추가하여 $L_{\text{RPD}}(\theta)$를 정의하는 것입니다.
   - $L_{\text{RPD}}(\theta) = L_{\text{RL}}(\theta) - L_{\text{MSE}}(\theta)$
   - 여기서 $L_{\text{MSE}}(\theta)$는 PPO 정책이 예측한 행동 평균 $\mu(\pi_{\theta}(a_t|s_t))$과 VLA 정책 $\pi_{\text{VLA}}(a_t|s_t)$이 반환하는 행동 $a_{\text{VLA},t}$의 기댓값 $E[a_{\text{VLA},t}]$ 사이의 MSE(Mean Squared Error) 기댓값입니다.
   - $L_{\text{MSE}}(\theta) = E_t \left[ \left\| \mu(\pi_{\theta}(a_t|s_t)) - E[a_{\text{VLA},t}] \right\|^2 \right]$
   - $L_{\text{RL}}$은 최대화되고 $L_{\text{MSE}}$는 최소화되어야 하므로, 목적 함수에 음수 부호로 포함됩니다.
   - 실제로 VLA 행동의 기댓값을 추정하기 위해 단계당 하나의 VLA 행동을 샘플링합니다.
2. **Experimental Setup (ManiSkill3)**:
   - ManiSkill3 [31]의 8가지 조작 작업(manipulation tasks)을 사용하여 RPD를 평가합니다. 모든 행동 공간은 상대적인 작업 공간 행동 $a = (\Delta x, \Delta \omega, \text{gripper}) \in \mathbb{R}^7$을 포함합니다.
   - **VLA 미세 조정**: Octo [4] 및 OpenVLA [7]는 실제 세계 데이터셋으로만 훈련되었기에, 시뮬레이션 환경에서 성공률이 0이었습니다. RPD 평가를 위해 ManiSkill3에서 제공하는 RL 생성 전문가 시연을 사용하여 Octo와 OpenVLA를 시뮬레이션 환경에 맞게 미세 조정했습니다.
   - **카메라 관점**: ManiSkill3의 '인간 카메라(human camera)' 시점을 사용하여 VLA 및 RL 에이전트 모두에 시각적 관찰을 제공하며, VLA 미세 조정 데이터도 이 시점으로 기록되었습니다. 이 카메라는 작업 목표에 대한 시각적 힌트를 포함합니다.
   - **VLA 통합**: 다양한 VLA를 RL 훈련 루프에 통합하기 위해 범용 정책 인터페이스와 HTTP API 서버를 사용했습니다. VLA 추론과 RL 훈련이 동일한 머신에서 실행될 경우 직렬화 오버헤드를 피하기 위해 공유 메모리를 활용합니다. OpenVLA의 배치 처리 미지원으로 인해 훈련 속도가 느려, OpenVLA를 사용한 실험은 일부 작업에 한정되었습니다.
3. **RPD Variants and Baselines**:
   - **RPD-MSE**: 위에 설명된 주된 RPD 변형 (MSE 손실 사용).
   - **RPD-L1**: MSE 대신 L1 손실을 사용하는 변형.
   - **RPD-BC**: 순수한 최대 우도(maximum likelihood) 손실을 사용하는 변형 (분산 항 포함).
   - **PPO Baseline**: CleanRL [32]의 PPO 구현.
   - **PPD Baseline**: 자체 구현한 PPD [23] (KL 발산 계산을 위해 VLA에서 10개의 행동을 샘플링하고 다변량 가우시안 분포에 피팅).

## 📊 Results

- **RPD 변형 및 기준선 비교 (PickCube)**:
  - RPD-MSE는 PPO, PPD, RPD-L1, RPD-BC를 포함한 모든 다른 변형 및 기준선보다 성능이 우수했으며 (그림 3), RPD-L1이 그 뒤를 이었습니다.
  - RPD-MSE는 Octo (67% 성공률)와 OpenVLA (27% 성공률) VLA를 빠르게 능가하며, 약 80%의 성공률을 달성하여 증류된 정책을 개선합니다.
  - PPO 기준선도 비슷한 성공률에 수렴하지만, 학습 속도가 현저히 느리고 훈련 과정에서 더 큰 변동성을 보입니다.
  - PPD는 VLA의 이봉형(bimodal) 행동 분포로 인해 KL 발산 계산이 어려워 Octo 기준선 이상으로 정책 성능을 올리는 데 실패했습니다.
- **다양한 작업에서의 RPD 성능 (밀집/희소 보상)**:
  - 6가지 VLA 미세 조정 데이터셋 내 작업(그림 4)에서 RPD는 PPO 기준선보다 일관되게 빠르게 학습하고 종종 더 높은 성공률에 수렴합니다.
  - RollBall 및 StackCube와 같은 일부 작업에서는 PPO 기준선이 ManiSkill3의 기본 하이퍼파라미터로 실패했지만, RPD는 성공적인 정책을 찾았습니다.
  - **희소 보상(Sparse Rewards)**: 희소 보상 설정에서 RPD는 PPO보다 훨씬 뛰어난 성능을 보였으며, PPO가 작업을 완전히 학습하지 못하는 경우에도 RPD는 성공했습니다. RPD는 교사 정책의 지침을 통해 에이전트가 에피소드를 성공적으로 완료하고 보상을 획득하여 학습을 촉진하는 데 더 효과적이었습니다.
- **새로운 환경에 대한 일반화 (분포 외 작업 & 카메라 변경)**:
  - **보류 작업(Hold-Out Tasks)**: PullCubeTool 및 PokeCube와 같이 VLA 미세 조정 데이터셋에서 제외된 작업에서 RPD는 밀집 보상 설정에서는 실패했지만 (VLA가 도구 사용을 요구하는 의미 있는 행동을 제공하지 못함), VLA 정책이 일반화되는 재해석된 작업(방해물만 있는 PullCube/PushCube)의 희소 보상 설정에서는 성공적으로 학습하고 PPO보다 우수했습니다 (그림 5).
  - **카메라 관점 변경**: 카메라 각도가 약간 변경된 PushCube 작업(그림 6)에서 OpenVLA의 성공률은 27%에서 4.5%로, Octo는 67%에서 0%로 급락했습니다. 그러나 RPD는 이러한 VLA 성능 저하에도 불구하고 PPO보다 우수한 성능을 보였습니다 (그림 7). 이는 VLA의 행동이 여전히 탐색을 좋은 방향으로 안내했기 때문입니다.

## 🧠 Insights & Discussion

- **VLA의 전문가 정책 전환**: RPD는 대규모 일반화 VLA 모델이 특정 작업에 대한 소형의 효율적인 전문가 정책으로 개선될 수 있음을 보여주어, VLA의 실제 적용 가능성을 높입니다.
- **샘플 효율성 및 안정성 향상**: RPD는 VLA 교사 정책의 지침을 통해 RL 에이전트의 탐색을 효율적으로 유도하여, PPO보다 더 빠르고 안정적으로 수렴하며, 밀집 및 희소 보상 설정 모두에서 더 높은 성능을 달성합니다.
- **일반화 능력**: VLA가 특정 환경 변경(예: 카메라 시점 변경, 교란 물체 존재)에 취약할 때에도, RPD는 VLA의 부분적인 지식(예: 좋은 방향으로의 탐색 안내)을 활용하여 RL 에이전트가 해당 작업을 학습하고 VLA의 성능을 능가할 수 있도록 돕습니다.
- **제한 사항**: RPD의 성능은 기본 VLA의 능력에 크게 의존합니다. VLA가 도메인 시프트(domain shift) 등으로 인해 유의미한 행동을 제공하지 못하면, RPD는 PPO 훈련을 방해하고 성능을 저하시킬 수 있습니다. 또한, 이미 쉽거나 잘 튜닝된 작업의 경우 RPD의 개선 효과는 제한적일 수 있습니다.
- **향후 연구 방향**: 시뮬레이션에서 훈련된 RPD를 sim-to-real(시뮬레이션-실제 전이) 방법 [33]과 결합하여 실제 로봇에 배포하거나, 증류된 전문가 정책을 사용하여 새로운 훈련 데이터를 수집하고 이를 다시 VLA 미세 조정에 활용하여 추가적인 인간 시연 데이터 없이 일반화 정책을 개선하는 연구가 가능합니다.

## 📌 TL;DR

RPD(Refined Policy Distillation)는 VLA(Vision-Language-Action) 모델의 일반화 능력을 RL 전문가 정책의 고성능으로 연결하는 새로운 온-폴리시 RL 기반 방법입니다. 이 방법은 PPO(Proximal Policy Optimization)에 VLA 예측 행동에 대한 MSE 손실을 추가하여 RL 에이전트의 탐색을 효과적으로 안내하고, PPO 대비 더 빠른 수렴과 높은 성공률을 달성합니다. 특히 희소 보상 설정과 VLA 교사의 성능이 저하되는 카메라 시점 변경이나 분포 외 작업에서도 VLA의 지식을 활용하여 RL 에이전트가 전문가 수준의 정책을 학습하도록 돕습니다. 이는 VLA를 시뮬레이션 환경에서 활용할 수 있는 가능성을 열고, 미래 로봇 학습의 효율성을 크게 향상시킬 잠재력을 보여줍니다.
