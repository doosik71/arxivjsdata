# FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies

Moritz Reuss, Hongyi Zhou, Marcel Rühlem, Ömer Erdinç Yağmurlu, Fabian Otto, Rudolf Lioutikov

## 🧩 Problem to Solve

로봇 공학에서 다양한 작업을 수행하고 여러 로봇 신체(embodiment)에 걸쳐 적용 가능한 범용 조작 정책(generalist manipulation policies)을 개발하는 것이 주요 목표입니다. 하지만 현재 Vision-Language-Action (VLA) 정책들은 수십억 개의 파라미터를 가진 모델과 방대한 데이터셋을 필요로 하여 계산 비용과 자원 요구 사항이 엄청나다는 한계가 있습니다. 특히 확산 기반(diffusion-based) VLA 정책들은 복잡하고 다중 모달(multimodal)의 액션 분포를 모델링하는 데 뛰어나지만, 높은 계산 비용과 긴 학습 및 추론 시간으로 인해 더 다양한 연구와 실제 로봇 배포에 장벽이 되고 있습니다. 이 논문은 이러한 효율성 문제를 해결하는 것을 목표로 합니다.

## ✨ Key Contributions

이 논문의 주요 기여는 다음과 같습니다:

- **중간 모달리티 퓨전 (Intermediate-modality fusion):** LLM 레이어의 최대 50%를 가지치기(pruning)하여 확산 헤드(diffusion head)에 용량을 재할당합니다. 이는 VLM의 의미론적 깊이를 유지하면서 파라미터 수를 줄이는 효과를 가져옵니다.
- **액션 공간별 Global-AdaLN 컨디셔닝 (Action-specific Global-AdaLN conditioning):** 모듈형 적응(modular adaptation)을 통해 파라미터를 20% 줄이는 동시에 각 액션 유형에 대한 고유한 조절 신호를 생성하는 새로운 정규화 메커니즘을 제안합니다.
- **광범위한 설계 선택 평가 (Extensive evaluation of design choices):** 다양한 벤치마크에 걸쳐 VLM 아키텍처 및 사전 학습 목표가 효율적인 VLA 채택에 미치는 영향을 평가하는 심층적인 어블레이션(ablation) 연구를 수행합니다.
- **FLOWER (Florence With Embodied Flow) 개발:** 위 기여들을 통합하여 9억 5천만 개의 파라미터를 가진 VLA 정책인 FLOWER를 제안합니다. FLOWER는 단 200 H100 GPU 시간만에 사전 학습되어, 10개의 시뮬레이션 및 실제 벤치마크에 걸친 190개 작업에서 기존 VLA 모델들과 동등하거나 우수한 성능을 달성하며 새로운 SoTA를 기록합니다.

## 📎 Related Works

- **초기 모방 학습 (Early Imitation Learning):** Pinto와 Gupta [11]는 데이터셋 크기가 정책 성능에 미치는 영향을 보여주었으며, OXE 벤치마크 [6]는 대규모 데이터셋을 제공하여 범용 정책 연구를 가능하게 했습니다.
- **Diffusion 기반 정책:** Octo [3]는 Transformer 기반 확산 정책을 적용했지만 사전 학습된 VLM 인코더가 부족하여 일반화에 한계가 있었습니다. RDT-1B [8]는 1.2B 파라미터의 확산 Transformer와 11.4B VLM을 사용했으나, 48 A100 GPU로 한 달간 사전 학습하는 등 엄청난 계산 비용이 문제였습니다.
- **VLM을 통합한 범용 정책:** OpenVLA [1]는 7.7B VLM을 미세 조정하여 이산형(discrete) 액션 예측에 사용했으나, 크기 때문에 실제 로봇 배포가 어려웠습니다. π$_{0}$ [7] 및 GR00T-N1 [9]도 2B 이상의 파라미터를 가진 범용 플로우 기반(flow-based) VLA를 제안했습니다. 이들 모두 매우 큰 VLM을 유지하여 높은 메모리 요구 사항과 느린 수렴 속도를 가졌습니다.
- **모델 크기 및 퓨전 전략 감소 노력:** TinyVLA [19]는 경량 VLM에 소형 확산 헤드를 늦은 퓨전(late-fusion) 방식으로 연결했습니다. 퓨전 전략은 초기 퓨전(early fusion), 늦은 퓨전, 그리고 FLOWER가 제안하는 중간 퓨전(intermediate fusion) 등으로 나뉩니다. 중간 퓨전은 중간 VLM 토큰을 Flow Transformer에 주입하여 VLM을 선택적으로 가지치기하면서도 의미론적 풍부함을 유지합니다.

## 🛠️ Methodology

FLOWER는 상태 $s_t$, 텍스트 목표 $g_t$, 메타-신체(meta-embodiment) 정보 $e_i$에 따라 액션을 생성하는 효율적인 범용 정책 $\pi_{\theta}$를 학습합니다.

1. **중간 모달리티 퓨전 VLA (Intermediate Modality Fusion Vision-Language-Action-Models):**
   - **VLM 구조 가지치기:** 사전 학습된 VLM의 중간 레이어에서 은닉 상태(hidden states)를 추출하여 의미론적 깊이와 계산 효율성의 균형을 맞춥니다.
     - 인코더-디코더 VLM (예: Florence-2 [24])의 경우, 전체 디코더를 제거하고 인코더 LLM 레이어만 유지하여 레이어 수를 50% 줄입니다.
     - 디코더 전용 VLM (예: SmolFlow2-Video [25])의 경우, 마지막 30%의 Transformer 레이어를 제거하여 파라미터 수를 20-35% 줄입니다.
   - **Flow Transformer에 주입:** VLM 잠재 토큰(latent tokens)을 선형 레이어와 RMSNorm [26]을 통해 Flow Transformer에 크로스-어텐션(cross-attention) 방식으로 주입합니다. 이는 시맨틱하게 풍부한 VLM 특징으로 각 Flow 레이어를 컨디셔닝하여 빠른 정책 수렴을 가능하게 합니다.
2. **크로스-액션 공간 Flow Transformer (Cross-Action Space Flow Transformer):**
   - **Action-Space Global-AdaLN-Zero:** 이 새로운 정규화 계층은 시간 신호(예: flow time step)와 액션 유형별 임베딩(per-action-type embeddings) 모두에 대해 각 Transformer 블록을 컨디셔닝합니다.
     - 기존 AdaLN-Zero가 레이어당 별도의 스케일-앤-시프트(scale-and-shift) 파라미터를 사용하여 30%의 파라미터를 추가하는 것과 달리, Global-AdaLN-Zero는 모든 레이어에서 단일 변조 가중치(modulation weights) 세트를 공유하며 각 액션 카테고리(예: delta-EEF vs. joint angle)에 고유한 신호를 생성합니다. 이는 파라미터 수를 20% 이상 줄입니다.
     - 각 레이어에 경량 LoRA 어댑터(adapter)를 추가하여 레이어별 미세 조정이 가능하게 합니다.
   - **액션 공간별 인코더/디코더:** 각 액션 유형은 Transformer 잠재 공간으로/에서 액션을 매핑하기 위한 소형 인코더/디코더를 사용하여, 다양한 액션 차원을 일관되게 처리할 수 있도록 합니다.
3. **액션 생성을 위한 Rectified Flow (Rectified Flow for Action Generation):**
   - 노이즈 분포와 데이터 분포 사이의 직선 속도장(straight-line velocity fields)을 사용하는 Rectified Flow [28, 29]를 활용합니다.
   - 이를 통해 추론 계산을 줄이고 표현력을 유지하며, 이는 지연 시간이 중요한 로봇 정책에 필수적입니다.
   - 모델은 다음과 같은 손실 함수를 최적화합니다:
     $$ L(\theta) = E*{t,z_1} \left[ \left\|z_1 - \bar{a}*{n,k} - v*{\theta}(z_t,t, \bar{s}\_n, g, e)\right\|^2 \right] $$
     여기서 $z_t = (1-t)\bar{a}*{n,k} + tz*1$ 이고 $z_1 \sim N(0,I)$ 입니다. $\bar{a}*{n,k}$는 실제 액션 시퀀스, $v_{\theta}$는 상태 $\bar{s}$, 언어 목표 $g$, 신체 $e$에 컨디셔닝된 플로우 모델입니다.
4. **FLOWER 아키텍처:**
   - Florence-2-L VLM의 절반을 주 백본으로 사용하며, 1024 잠재 차원의 18개 레이어 Flow Transformer를 포함합니다.
   - 총 9억 4천 7백만 개의 파라미터와 1.85 GB의 VRAM만을 요구합니다.
   - **비용 효율적인 사전 학습:** 약 250k 궤적을 포함하는 8개의 공개 로봇 데이터셋 "OXE-soup"을 사용하여 48시간(약 200 GPU-시간) 내에 360,000 스텝을 완료합니다.

## 📊 Results

FLOWER는 다양한 실험을 통해 그 효율성과 성능을 입증했습니다.

- **주요 설계 결정 평가 (Critical Design Decisions Evaluation):**
  - **중간 퓨전의 효율성:** 중간 퓨전은 Florence-VLM 및 SmolFlow-VLM 모두에서 다른 퓨전 전략(초기, 늦은 퓨전)보다 뛰어난 성능을 보였으며, LIBERO-Long 벤치마크에서 93.4%의 성공률을 달성했습니다.
  - **VLM 백본 유형:** Florence-2의 사전 학습 목표가 로봇 조작 작업에 더 효과적임을 확인했습니다.
  - **Global-AdaLN 효율성:** Global-AdaLN은 성능 저하 없이 파라미터 수를 20% 줄였습니다.
  - **Diffusion Transformer 용량:** 고용량 Diffusion Transformer가 성능에 중요함을 확인했습니다.
- **시뮬레이션 실험 (Simulation Experiments):**
  - FLOWER는 CALVIN, LIBERO, ALOHA, SIMPLER 등 10개 벤치마크에 걸친 190개 작업에서 OpenVLA 및 $\pi_{0}$와 같은 최신 SoTA 모델과 동등하거나 우수한 성능을 꾸준히 보였습니다.
  - 특히 CALVIN과 LIBERO 벤치마크에서 OpenVLA를 큰 폭으로 능가했으며, LIBERO-Long에서는 90% 이상의 성공률을 달성한 유일한 정책이었습니다.
  - CALVIN ABC 벤치마크에서 4.53이라는 새로운 SoTA를 달성했습니다.
- **실세계 평가 및 일반화 (Real-World Evaluation and Generalization):**
  - 실제 주방 환경에서 20개 작업에 대해 평가했을 때, FLOWER는 평균 성공률 61%를 달성하여 차순위 OpenVLA(31%)보다 두 배 높은 성능을 보였습니다.
  - 새로운 객체, 손전등 조명, 배경 방해물, 새로운 작업 구성 등 도전적인 일반화 시나리오에서 OpenVLA를 꾸준히 능가하며 평균 51.0%의 성공률을 달성했습니다 (OpenVLA는 23.4%).
- **추론 효율성 (Inference Efficiency):**
  - RTX 4090 GPU에서 FLOWER는 311Hz의 처리량(throughput)을 달성하여 $\pi_{0}$보다 8% 빠르고 OpenVLA보다 5007% 빨랐습니다.
  - VLA 중 가장 낮은 메모리 사용량(1.85GB VRAM)을 보여, $\pi_{0}$의 27.6%, OpenVLA의 12.7% 수준이었습니다.

## 🧠 Insights & Discussion

- **효율성과 성능의 균형:** FLOWER는 중간 모달리티 퓨전과 Global-AdaLN 컨디셔닝을 통해 계산 효율성을 크게 높이면서도 SoTA 성능을 유지했습니다. 이는 로봇 공학에서 대규모 모델의 높은 계산 비용 문제를 해결하는 중요한 진전입니다.
- **VLM 가지치기의 효과:** VLM 레이어를 가지치기하고 중간 임베딩을 활용하는 전략이 의미론적 정보를 효과적으로 유지하면서 모델 크기를 줄이는 데 성공했음을 보여주었습니다.
- **액션 공간 적응성:** Global-AdaLN 메커니즘은 다양한 액션 공간을 효율적으로 처리할 수 있게 하여, FLOWER가 이질적인 로봇 신체와 작업 설정에 잘 적응함을 입증했습니다.
- **실세계 일반화 능력:** 시뮬레이션뿐만 아니라 실제 환경에서도 우수한 성능과 일반화 능력을 보였으며, 특히 새로운 객체나 환경 변화에 대한 강건함이 인상적입니다.
- **한계점:**
  - 반복적인 샘플링 절차(iterative sampling procedure)를 사용하여 본질적으로 결정론적 정책(deterministic policies)의 단일 순방향 패스(single forward pass)보다 느립니다.
  - 주로 세 가지 조작 액션 공간에서 검증되었으며, 모바일 내비게이션이나 휴머노이드 보행과 같은 다른 신체로의 일반화는 아직 탐구되지 않았습니다.
  - SIMPLER Google Robot 벤치마크에서 제로샷 배포(zero-shot deployment) 성능이 더 큰 모델에 비해 개선이 필요함을 시사합니다.
  - 9억 5천만 개의 파라미터는 대부분의 SoTA VLA 모델보다 작지만, 저자원(low-resource) 또는 실시간(real-time) 환경에서는 여전히 배포에 어려움이 있을 수 있습니다.
  - 대부분의 벤치마크(10개 중 8개)가 시뮬레이션에서 수행되어 실세계 일반화에 대한 증거가 제한적일 수 있습니다.

## 📌 TL;DR

FLOWER는 기존 VLA 정책의 높은 계산 비용과 자원 요구 사항을 해결하기 위해, VLM 레이어를 가지치기하는 **중간 모달리티 퓨전**과 액션 공간별 파라미터를 20% 줄이는 **Global-AdaLN 컨디셔닝**을 도입한 효율적인 범용 로봇 정책입니다. 9억 5천만 개의 파라미터를 가진 FLOWER는 단 200 GPU 시간만에 사전 학습되어, 10개 벤치마크의 190개 작업에서 기존 SoTA VLA 모델과 동등하거나 우수한 성능을 달성하고, CALVIN ABC에서 새로운 SoTA를 기록했습니다. FLOWER는 낮은 계산 비용으로 강력한 성능과 실세계 일반화 능력을 제공하며, 효율적인 로봇 정책 개발의 가능성을 제시합니다.
