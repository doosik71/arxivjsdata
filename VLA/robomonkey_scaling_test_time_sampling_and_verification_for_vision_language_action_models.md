# RoboMonkey: Vision-Language-Action 모델을 위한 테스트 시간 샘플링 및 검증 확장

Jacky Kwok, Christopher Agia, Rohan Sinha, Matt Foutter, Shulu Li, Ion Stoica, Azalia Mirhoseini, Marco Pavone

## 🧩 Problem to Solve

Vision-Language-Action (VLA) 모델은 시각-운동 제어에서 뛰어난 능력을 보여주었지만, 비정형적인 실제 환경에서 이들의 **강건성(robustness)을 보장하는 것은 여전히 어려운 과제**입니다. 기존의 VLA 개선 노력은 주로 사전 학습(pre-training)이나 사후 학습(post-training, 미세 조정 등) 단계에 집중되었으며, **배포 시점(test-time)에 추가적인 연산 자원을 활용하여 VLA의 성능을 향상시키는 방법은 충분히 탐구되지 않았습니다.** 본 논문은 반복적인 샘플링 및 검증을 통해 VLA의 정밀도와 강건성을 높일 수 있는지 질문합니다.

## ✨ Key Contributions

- **추론 시간 스케일링 법칙(Inference-Time Scaling Law) 규명:** 액션 오류(action error)와 생성 샘플 수 간의 관계가 다양한 VLA에서 근사적인 거듭제곱 법칙(exponentiated power law)을 따른다는 것을 보여주어, 로봇 제어에서도 LLM과 유사한 추론 시간 스케일링 법칙이 존재함을 입증했습니다.
- **확장 가능한 합성 데이터 생성 파이프라인 제안:** 인간 주석 없이 대규모 모방 학습 데이터셋에 대한 합성 액션 비교 데이터를 자동으로 생성하고, 이를 활용하여 VLM 기반 액션 검증기(action verifier)를 학습하는 방법을 제시했습니다.
- **RoboMonkey 프레임워크를 통한 VLA 성능 대폭 향상:** 기존 VLA에 RoboMonkey를 적용하여 실제 환경 OOD(Out-of-Distribution) 작업에서 25%의 절대 성능 향상을, In-Distribution SIMPLER 환경에서 9%의 성능 향상을 달성했습니다.
- **새로운 로봇 환경 적응력 개선:** VLA와 액션 검증기를 모두 미세 조정할 경우, VLA만 미세 조정하는 것보다 LIBERO-Long 벤치마크에서 7%의 성능 향상을 보였습니다.

## 📎 Related Works

- **Vision Language Action (VLA) 모델:** $\pi_0$[1], OpenVLA[2], PaLM-E[3], RT-X[4,5] 등은 대규모 로봇 데이터셋으로 학습되어 강력한 일반화 능력을 보이지만, 분포 변화(distribution shift)나 누적 예측 오류로 인해 실패할 수 있습니다. RoboMonkey는 이러한 일반주의 정책의 강건성을 배포 시점에 향상시킵니다.
- **Out-of-Distribution (OOD) 강건성:** 로봇 분야에서 학습 기반 시스템의 OOD 데이터에 대한 불안정한 성능 문제는 잘 알려져 있습니다[8,9]. 기존 연구들은 강건한 학습 방법이나 환경 분포 변화에 대한 모델 적응에 중점을 두었습니다[47,48,49]. 본 연구는 파운데이션 모델(FM)을 고수준 계획보다는 **저수준 액션 검증기**로 활용하는 새로운 관점을 제시합니다[53,54].
- **반복 샘플링 (Repeated Sampling):** LLM(Large Language Model) 분야에서는 수학 문제 해결, 코딩, 요약 등 다양한 작업에서 테스트 시점에 추가 연산을 적용하는 반복 샘플링 기법의 효과가 입증되었습니다[22,28,55]. 로봇 분야의 V-GPS[42]는 오프라인 RL로 가치 함수를 학습하여 후보 액션을 재순위화하지만, RoboMonkey는 더 확장 가능한 데이터 큐레이션 파이프라인과 모델 아키텍처를 사용하며 효율적인 액션 생성을 위해 가우시안 섭동(Gaussian perturbation)을 통합합니다.

## 🛠️ Methodology

RoboMonkey는 **생성-검증(generate-then-verify)** 패러다임을 따르는 테스트 시간 스케일링 프레임워크입니다.

### 1단계: 액션 검증기 학습 (Training Action Verifier)

1. **합성 데이터 생성 파이프라인:**
   - 기존 모방 학습 데이터셋($\mathcal{D}_{buf}$)에서 각 상태($s_t$) 및 언어 지침($\mathcal{I}$)에 대해 VLA 정책($\pi_\theta$)으로부터 $N$개의 후보 액션을 샘플링합니다.
   - 다양성을 위해 클러스터링 알고리즘을 적용하여 $K$개의 대표 액션으로 줄입니다.
   - $K$개의 액션으로부터 $\binom{K}{2}$개의 모든 쌍별 비교를 구성합니다.
   - 각 샘플링된 액션과 정답 액션($a_t^*$) 간의 RMSE(Normalized Root Mean Squared Error)를 계산하여 "우승(winning)" 액션($a_t^W$)과 "패배(losing)" 액션($a_t^L$)을 결정하고, 이를 통해 합성 선호도 데이터셋($\mathcal{D}_{comp}$)을 생성합니다.
2. **보상 모델링 (Reward Modeling):**
   - LLaVA-7B[38,39]를 백본으로 사용하고 최종 unembedding 레이어를 보상 헤드(reward head)로 교체하여 VLM 기반 액션 검증기 $R_\phi(a, s, \mathcal{I})$를 학습합니다.
   - 손실 함수는 선호도 수준을 고려한 수정된 Bradley-Terry 모델[37]을 따릅니다:
     $$L(\phi; \mathcal{D}_{comp}) = -\mathbb{E}_{(a_t^W, a_t^L, a_t^*, s_t, \mathcal{I}) \sim \mathcal{D}_{comp}} \left[ \log \sigma \left( R_\phi(a_t^W, s_t, \mathcal{I}) - R_\phi(a_t^L, s_t, \mathcal{I}) - \alpha \| \Delta_t^* - \hat{\Delta}_t \|_2^2 \right) \right]$$
     여기서 $\Delta_t^*$는 정답 RMSE 차이이고 $\hat{\Delta}_t$는 예측된 보상 차이입니다. 마진 항 $\alpha$를 포함하여 명확하게 다른 액션들을 더 잘 구별합니다.

### 2단계: 테스트 시간 연산 스케일링 (Scaling Test-Time Compute)

배포 시 각 시간 단계 $t$에서:

1. **액션 샘플링:**
   - VLA 모델 $\pi_\theta$로부터 $\hat{N}$개의 초기 액션 후보($\hat{A}$)를 샘플링합니다.
   - 이 후보들의 이산 그리퍼(gripper) 컴포넌트에 대한 다수결 투표(majority voting)를 통해 그리퍼 상태($g_t$)를 결정합니다.
   - 나머지 6개의 연속적인 이동 및 회전 컴포넌트($\Delta x, \Delta y, \Delta z, \Delta u, \Delta v, \Delta w$)에 가우시안 분포 $\mathcal{N}(\mu_t, \Sigma_t)$를 적합시킵니다.
   - 이 가우시안 분포로부터 $\hat{K}$개의 새로운 액션($\tilde{A}$)을 샘플링하여 액션 제안 분포를 구성합니다 (가우시안 섭동).
2. **액션 검증 및 실행:**
   - 학습된 보상 모델 $R_\phi(\tilde{a}_t^i, s_t, \mathcal{I})$을 사용하여 $\hat{K}$개의 각 액션 $\tilde{a}_t^i$에 점수를 매깁니다.
   - 가장 높은 보상을 가진 액션을 최종 실행 액션 $a_t = \text{argmax}_{\tilde{a}_t^i \in \tilde{A}_t} R_\phi(\tilde{a}_t^i, s_t, \mathcal{I})$으로 선택합니다.

**실용적인 배포를 위한 최적화:** VLA 서빙 엔진(SGLang 기반)과 가우시안 섭동을 활용하여 계산 오버헤드를 줄였습니다. 이를 통해 16개의 후보 액션을 약 650ms(1.5 Hz) 내에 샘플링 및 검증할 수 있으며, 이는 순진한 정책 샘플링보다 41.3% 낮은 지연 시간(latency)을 제공합니다.

## 📊 Results

- **추론 시간 스케일링 법칙:** 액션 오류는 샘플 수가 증가함에 따라 일관되게 감소하며, 가우시안 섭동을 통한 샘플링이 CogACT, Octo, OpenVLA, SpatialVLA 등 다양한 VLA 모델에서 근사적인 거듭제곱 법칙을 따름을 확인했습니다.
- **In-Distribution 성능:** SIMPLER 환경의 in-distribution 작업에서 RoboMonkey는 평균 47.5%의 성공률을 달성하여 OpenVLA보다 평균 9%p 성능을 향상시켰습니다 (예: 가지 바구니에 넣기 작업에서 19%p, 블록 쌓기 작업에서 10%p).
- **Out-of-Distribution 강건성:** 실제 WidowX 로봇 OOD 작업에서 RoboMonkey는 평균 60%의 성공률을 달성하여 OpenVLA(35%) 및 V-GPS(30%)를 각각 25%p, 30%p 상회했습니다. 특히 시각적, 의미적 일반화가 필요한 "바나나를 바구니에 넣기" 같은 작업에서 OpenVLA는 0% 성공률을 보인 반면 RoboMonkey는 크게 개선되었습니다.
- **합성 데이터셋 스케일링 효과:** 합성 학습 데이터셋의 크기를 늘리면(최대 2천만 개 비교) SIMPLER 환경에서 RoboMonkey 액션 검증기의 폐쇄 루프 성공률이 37.5%에서 46.3%로 꾸준히 향상됨을 확인했습니다.
- **새로운 로봇 환경 적응:** LIBERO-Long 벤치마크에서 VLA와 액션 검증기를 함께 미세 조정했을 때, VLA만 미세 조정하는 것보다 평균 성공률이 6.7%p 증가했습니다.
- **효율성:** SGLang 기반의 최적화된 서빙 엔진과 가우시안 섭동 덕분에 16개의 후보 액션을 약 650ms(1.5 Hz) 내에 처리하여 실용적인 배포가 가능합니다. 가우시안 섭동이 순진한 정책 샘플링보다 컴퓨팅 예산당 더 낮은 액션 오류를 달성합니다.

## 🧠 Insights & Discussion

RoboMonkey는 생성-검증 패러다임을 통해 테스트 시간 컴퓨팅을 확장하는 것이 범용 로봇 파운데이션 모델을 구축하는 실용적이고 효과적인 방법임을 보여줍니다. 기존 VLA 모델의 정밀도와 강건성을 크게 향상시킬 수 있으며, 특히 OOD 시나리오에서의 성능 개선이 두드러집니다.

**한계점:**

- **계산 오버헤드:** 여러 후보 액션 샘플링 및 별도의 VLM 기반 액션 검증기 사용으로 인한 계산 오버헤드가 발생하며, 고주파수 제어가 필요한 작업에는 적합하지 않을 수 있습니다. 효율적인 모델 아키텍처나 시스템 수준 최적화가 필요합니다.
- **합성 데이터셋 스케일링:** 컴퓨팅 제약으로 인해 Bridge V2 데이터셋에서 2천만 개의 합성 액션 비교로 실험을 제한했습니다. 향후 더 큰 규모의 합성 데이터 생성을 탐구할 잠재력이 있습니다.
- **평가 범위:** WidowX 250S 및 Franka 로봇 팔에 중점을 두었으며, 향후 더 광범위한 로봇 형태(embodiments)에 대한 평가가 필요합니다.

## 📌 TL;DR

VLA 모델의 실제 환경 강건성 부족 문제를 해결하기 위해, 본 논문은 **RoboMonkey**라는 **테스트 시간 스케일링 프레임워크**를 제안합니다. RoboMonkey는 VLA에서 초기 액션을 샘플링한 후, **가우시안 섭동**과 **다수결 투표**를 사용하여 액션 제안 분포를 구축하고, **합성 선호도 데이터로 학습된 VLM 기반 검증기**를 통해 최적의 액션을 선택합니다. 이 방법은 VLA에서 추론 시간 스케일링 법칙이 존재함을 밝히고, 기존 VLA의 OOD 작업에서 25%p, In-Distribution 작업에서 9%p, 새로운 로봇 환경 미세 조정 시 7%p의 **상당한 성능 향상**을 달성하며, 로봇 분야에서 **생성-검증 패러다임**의 효과를 입증합니다.
