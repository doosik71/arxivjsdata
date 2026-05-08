# DART: Noise Injection for Robust Imitation Learning

Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, Ken Goldberg

## 🧩 Problem to Solve

모방 학습(Imitation Learning)에서 행동 복제(Behavior Cloning)와 같은 오프-정책(off-policy) 접근 방식은 로봇이 감독자의 시연 궤적에서 벗어날 때 오류가 누적되는 공변량 이동(covariate shift) 문제에 직면합니다. 이는 로봇이 학습 데이터와 다른 상태 분포를 방문하게 하여 성능 저하와 잠재적 위험을 초래합니다. DAgger와 같은 온-정책(on-policy) 방법은 이 문제를 완화하지만, 사람 감독자에게 피드백 제공이 번거롭고(tedious), 정책을 반복적으로 업데이트해야 하므로 계산 비용이 많이 들며(computationally expensive), 훈련 중 로봇이 위험한 상태(dangerous states)를 방문할 수 있다는 한계가 있습니다. 이 논문은 이러한 온-정책 방법의 단점을 피하면서도 공변량 이동을 효과적으로 줄이는 로버스트한 오프-정책 접근 방식을 제안합니다.

## ✨ Key Contributions

- **DART(Disturbances for Augmenting Robot Trajectories) 알고리즘 제안**: 감독자의 제어 스트림에 주입할 노이즈의 적절한 수준을 설정하는 새로운 알고리즘을 제시합니다.
- **노이즈 주입에 대한 이론적 분석**: 로봇 정책의 오류가 0이 아닐 때 DART가 행동 복제보다 성능을 향상시킬 수 있는 조건을 설명합니다.
- **실험적 검증**: 알고리즘 감독자(MuJoCo) 및 사람 감독자(Toyota HSR 로봇의 복잡한 환경에서 물체 파지)를 통한 광범위한 실험을 통해 노이즈 주입이 공변량 이동을 효과적으로 줄이고 로버스트한 정책 학습을 가능하게 함을 입증합니다.

## 📎 Related Works

- **행동 복제 (Behavior Cloning, BC)**: 감독자의 시연을 수동적으로 관찰하여 상태-제어 매핑을 학습하는 오프-정책 모방 학습 기법입니다. 공변량 이동으로 인해 실행 시 오류가 누적되는 문제가 있습니다.
- **DAgger (Dataset Aggregation)**: 로봇의 현재 정책에서 상태를 샘플링하고 감독자에게 수정(corrective actions)을 요청하여 공변량 이동을 보정하는 온-정책 모방 학습 방법입니다. 오류 누적을 완화하지만, 감독자 개입의 번거로움, 훈련 중 위험 상태 방문, 높은 계산 비용 등의 단점이 있습니다.
- **강건성(Robustness) 향상을 위한 노이즈 주입**: 강건 제어 이론(robust control theory)에서 모델 식별 시 공변량 이동과 유사한 현상을 해결하기 위해 "지속적 여기(persistence excitation)" 조건을 만족시키기 위해 제어 신호에 등방성 가우시안 노이즈(isotropic Gaussian noise)를 주입하는 기법이 활용됩니다.

## 🛠️ Methodology

DART는 감독자의 시연 중 제어 신호에 노이즈를 주입하여 로봇이 테스트 시 발생할 수 있는 오류를 시뮬레이션하고, 감독자가 이러한 오류로부터 복구하는 방법을 시연하도록 강제합니다. 이는 오프-정책 학습의 장점을 유지하면서도 온-정책 방법의 강건성을 얻습니다.

1. **노이즈 주입 개념**:
   - 감독자의 정책 $\pi_{\theta^*}$에 노이즈 $\psi$를 주입하여 $\pi_{\theta^*}(u|x,\psi)$ 분포를 따르는 궤적을 수집합니다.
   - 이 노이즈는 로봇의 학습된 정책 $\pi_{\hat{\theta}}$의 오류를 근사하도록 최적화됩니다.
2. **노이즈 수준 최적화**:
   - DART는 로봇 정책의 최종 분포 $p(\xi|\pi_{\hat{\theta}})$가 노이즈가 주입된 감독자의 분포 $p(\xi|\pi_{\theta^*}, \psi)$에 가까워지도록 $\psi$를 최적화하는 것을 목표로 합니다.
   - 이는 KL-발산(KL-divergence)의 상한을 최소화하는 문제로 귀결되며, 다음과 같은 목적 함수를 최소화하는 방식으로 수행됩니다.
     $$\min_{\psi} E_{p(\xi|\pi_{\hat{\theta}})} \left[ - \sum_{t=0}^{T-1} \log[\pi_{\theta^*}(\pi_{\hat{\theta}}(x_t)|x_t, \psi)] \right]$$
   - 그러나 로봇의 최종 정책은 데이터 수집 후에야 알 수 있으므로, "닭이 먼저냐 달걀이 먼저냐"의 문제가 발생합니다.
3. **반복적 최적화 (DART 알고리즘)**:
   - DART는 다음 단계를 반복적으로 수행하여 노이즈 파라미터 $\psi$를 업데이트하고 데이터를 수집합니다:
     1. 초기 노이즈 파라미터 $\psi_k$로 감독자로부터 $N$개의 시연을 수집합니다.
     2. 수집된 데이터를 사용하여 경험적 위험 최소화(empirical risk minimization)를 통해 로봇 정책 $\pi_{\hat{\theta}}$를 학습합니다.
     3. 현재 로봇 정책 $\pi_{\hat{\theta}}$를 기반으로 샘플 추정치를 사용하여 Eq. 3을 최적화하여 노이즈 파라미터 $\hat{\psi}_{k+1}$를 업데이트합니다:
        $$\hat{\psi}_{k+1} = \operatorname{argmin}_{\psi} E_{p(\xi|\pi_{\theta^*}, \psi_k)} \left[ - \sum_{t=0}^{T-1} \log[\pi_{\theta^*}(\pi_{\hat{\theta}}(x_t)|x_t, \psi)] \right]$$
        (가우시안 노이즈의 경우, $\hat{\Sigma}_{k+1} = \frac{1}{T} E_{p(\xi|\pi_{\theta^*}, \Sigma_{\alpha}^{k})} \left[ \sum_{t=0}^{T-1} (\pi_{\hat{\theta}}(x_t) - \pi_{\theta^*}(x_t))(\pi_{\hat{\theta}}(x_t) - \pi_{\theta^*}(x_t))^{\text{T}} \right]$)
     4. 로봇의 최종 오류에 대한 사전 추정치 $\alpha$에 따라 $\hat{\psi}_{k+1}$를 스케일링하여 $\psi_{\alpha}^{k+1}$를 얻습니다 (수축 추정(shrinkage estimation)에서 영감):
        $$\Sigma_{\alpha}^{k+1} = \frac{\alpha}{T \operatorname{tr}(\hat{\Sigma}_{k+1})} \hat{\Sigma}_{k+1}$$
     5. 업데이트된 노이즈 파라미터 $\psi_{\alpha}^{k+1}$를 사용하여 $N$개의 새로운 시연을 수집합니다.
     6. 누적된 데이터셋으로 로봇을 재훈련합니다.
     7. 총 $K$번 반복합니다.

## 📊 Results

- **MuJoCo 이동 환경 (알고리즘 감독자)**:
  - DART는 Walker, Hopper, Half-Cheetah, Humanoid와 같은 MuJoCo 환경에서 DAgger와 동등한 수준의 성능을 달성합니다.
  - 특히 Humanoid와 같은 고차원 작업에서 DART는 계산 시간이 3배 더 빠릅니다.
  - 훈련 중 감독자의 누적 보상 감소는 DART가 5%에 불과한 반면, DAgger는 80% 이상의 누적 보상 감소를 보였습니다. 이는 DART가 감독자의 개입 부담과 학습 중 위험을 현저히 줄임을 의미합니다.
  - 최적화되지 않은 등방성 가우시안(Isotropic-Gaussian) 노이즈는 성능이 좋지 않아, 노이즈 수준 최적화의 중요성을 보여줍니다.
- **로봇의 복잡한 환경에서의 물체 파지 (사람 감독자)**:
  - Toyota HSR 로봇을 이용한 실제 실험에서, DART( $\alpha=3$)는 기존 행동 복제(성공률 49%) 대비 62% 향상된 79%의 성공률을 달성했습니다.
  - 높은 노이즈 수준($\alpha=6$)은 행동 복제보다 우수했지만(성공률 72%), $\alpha=3$보다는 낮은 성능을 보여, 사람 감독자에게는 특정 노이즈 수준이 최적임을 시사합니다.

## 🧠 Insights & Discussion

- **공변량 이동 감소**: DART는 감독자의 정책 "경계(boundary)"에서 로봇이 발생할 수 있는 오류를 시뮬레이션하는 교정 예시(corrective examples)를 제공하여 공변량 이동을 효과적으로 줄입니다. 이는 로봇이 오류로부터 복구하는 방법을 학습할 수 있게 합니다.
- **효율성 및 안전성**: DART는 온-정책 방법의 강건성(robustness)과 오프-정책 학습의 효율성 및 안전성을 결합합니다. 감독자 정책 주변에 집중된 시연을 수집하여 로봇이 크게 비최적적인 상태를 방문하는 것을 피하고, 사람 감독자의 인지 부하를 DAgger보다 줄여줍니다.
- **이론적 기반**: Proposition 4.1은 로봇 정책이 감독자에 대해 0이 아닌 오류를 가질 때, DART가 행동 복제보다 공변량 이동(KL-발산의 상한)을 더 효과적으로 줄임을 보여줍니다.
- **실제 적용의 중요성**: 완벽하게 감독자를 표현할 수 있는 충분한 데이터를 얻기 어려운 실제 애플리케이션에서는 DART의 필요성이 더욱 부각됩니다.
- **제한 사항**: 사람 감독자의 경우, 노이즈 주입이 감독 행동에 영향을 미칠 수 있다는 가정이 항상 참이 아닐 수 있습니다. 따라서 노이즈 수준의 신중한 최적화가 중요합니다.

## 📌 TL;DR

**문제**: 모방 학습(Imitation Learning)에서 행동 복제(Behavior Cloning)는 로봇의 오류가 누적되어 학습된 궤적에서 벗어나는 공변량 이동(covariate shift) 문제를 겪습니다. DAgger와 같은 온-정책(on-policy) 방법은 이를 해결하지만, 감독자에게 번거롭고, 계산 비용이 많이 들며, 훈련 중 위험한 상태를 방문할 수 있습니다.

**해결책**: 이 논문은 감독자의 시연 과정에 노이즈를 주입하여 로봇이 오류로부터 복구하는 방법을 배우도록 하는 오프-정책(off-policy) 접근 방식인 DART(Disturbances for Augmenting Robot Trajectories)를 제안합니다. DART는 로봇 정책의 오류를 근사하도록 노이즈 수준을 반복적으로 최적화합니다.

**결과**: DART는 MuJoCo 시뮬레이션 환경에서 DAgger와 유사한 성능을 달성하면서도 계산 시간을 3배 단축하고 감독자의 누적 보상 감소를 5%로 최소화합니다. 실제 로봇을 이용한 복잡한 환경에서의 물체 파지(grasping in clutter) 작업에서는 행동 복제보다 평균 62%의 성능 향상을 보였습니다. 이는 DART가 공변량 이동을 효과적으로 줄이고 효율적이며 안전한 모방 학습을 가능하게 함을 보여줍니다.
