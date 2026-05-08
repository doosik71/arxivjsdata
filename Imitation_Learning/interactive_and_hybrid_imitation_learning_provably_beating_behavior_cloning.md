# Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning

Yichen Li, Chicheng Zhang (2025)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)에서 발생하는 고질적인 문제인 **compounding error**(복합 오류)를 해결하고자 한다. Behavior Cloning(BC)과 같은 오프라인 모방 학습은 전문가의 궤적(trajectory)만을 사용하여 학습하므로, 학습된 정책이 실행 중에 전문가가 방문하지 않은 상태(unseen states)에 진입할 경우 오류가 누적되어 성능이 급격히 저하되는 현상이 발생한다.

기존의 연구들은 주석 비용(annotation cost)을 궤적 단위로 측정했을 때, 대화형(interactive) 방법이 BC보다 일반적으로 더 나은 효율성을 보이지 않는다는 결론을 내린 바 있다. 이에 따라 대화형 모방 학습의 실질적인 이점이 제한적인 조건에서만 나타난다는 회의적인 시각이 존재했다. 본 논문의 목표는 주석 비용의 측정 단위를 **상태 단위(state-wise)**로 재정의함으로써, 대화형 및 하이브리드 모방 학습이 BC보다 이론적·실무적으로 더 높은 샘플 효율성을 가질 수 있음을 증명하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 주석 비용을 궤적 전체가 아닌 개별 상태에 대해 지불하는 것으로 설정하여, 학습자가 현재 정책에서 취약한 상태만을 선택적으로 전문가에게 질문할 수 있게 하는 것이다.

1. **STAGGER 알고리즘 제안**: 상태 단위 주석 오라클을 사용하는 대화형 모방 학습 알고리즘인 STAGGER를 제안하며, 전문가 정책이 실수로부터 빠르게 회복할 수 있는 **$\mu$-recoverable** 설정에서 BC보다 적은 수의 상태 주석만으로도 더 높은 성능을 낼 수 있음을 이론적으로 증명하였다.
2. **WARM-STAGGER 알고리즘 제안**: 이미 확보된 오프라인 데이터와 대화형 주석을 동시에 사용하는 **Hybrid Imitation Learning(HyIL)** 프레임워크를 설계하였다. 오프라인 데이터로 초기 정책 공간을 제한(warm-start)한 후 대화형 학습을 수행함으로써, 두 데이터 소스 중 어느 하나만 사용했을 때보다 더 나은 성능 보장을 제공한다.
3. **이론적 및 실험적 검증**: 특정 MDP 예시를 통해 하이브리드 학습이 BC의 커버리지 문제와 STAGGER의 **cold-start** 문제를 동시에 해결함을 보였으며, MuJoCo 연속 제어 태스크 실험을 통해 비용 효율성을 입증하였다.

## 📎 Related Works

기존의 모방 학습은 크게 두 가지 흐름으로 나뉜다.

- **Offline IL (Behavior Cloning)**: 전문가의 궤적 데이터를 지도 학습(supervised learning) 방식으로 학습한다. 단순하지만 앞서 언급한 compounding error에 취약하다.
- **Interactive IL (DAgger 등)**: 학습자가 환경과 상호작용하며 전문가에게 실시간으로 정답을 묻는 방식이다. DAgger는 이 분야의 대표적인 알고리즘으로, 전문가의 corrective feedback을 통해 데이터 분포의 불일치를 해소한다.

**기존 연구와의 차별점**: 기존 분석(예: Foster et al., 2024)은 주석 비용을 궤적 단위로 측정하여 BC가 minimax optimal하다고 주장하였다. 그러나 본 논문은 실제 환경에서 궤적 전체를 다시 라벨링하는 것보다 특정 상태에 대해 조언을 받는 것이 훨씬 저렴하다는 점에 착안하여, **state-wise annotation** 관점에서 분석을 수행함으로써 대화형 학습의 우위를 다시 입증하였다.

## 🛠️ Methodology

### 1. STAGGER (State-wise DAgger)

STAGGER는 DAgger의 변형으로, 매 반복(iteration)마다 단 하나의 상태에 대해서만 전문가의 주석을 요청한다.

- **절차**:
    1. 현재 정책 $\pi_n$을 사용하여 환경에서 궤적을 생성한다.
    2. 생성된 궤적 중 하나의 상태 $s_n$을 샘플링한다.
    3. 전문가 오라클 $O_{\text{State}}$에게 해당 상태에서의 정답 액션 $a^*_{n} = \pi_E(s_n)$을 요청한다.
    4. $\ell_n(\pi) = \log \frac{1}{\pi(a^*_{n}|s_n)}$ 손실 함수를 사용하여 정책을 업데이트한다.
    5. 최종 정책 $\hat{\pi}$는 학습 과정에서 생성된 정책들의 균등 혼합(uniform mixture)으로 산출된다.

- **이론적 보장**: 전문가 정책이 $\mu$-recoverable하고 결정론적 실현 가능성(deterministic realizability)이 만족될 때, 서브옵티말리티(suboptimality)는 다음과 같이 바운드된다.
$$J(\pi_E) - J(\hat{\pi}) \le \frac{\mu H (\log B + 2 \log(1/\delta))}{N_{\text{int}}}$$
여기서 $\mu$는 회복 비용, $H$는 에피소드 길이, $N_{\text{int}}$는 대화형 주석 횟수이다.

### 2. WARM-STAGGER (Hybrid IL)

오프라인 데이터 $D_{\text{off}}$와 대화형 오라클 $O_{\text{State}}$를 모두 활용하는 방법이다.

- **절차**:
    1. **Warm-start**: 오프라인 데이터 $D_{\text{off}}$와 일치하는 정책들로 구성된 제한된 정책 클래스 $\mathcal{B}_{bc}$를 구축하여 초기화한다.
    2. 이후 과정은 STAGGER와 동일하게 $\mathcal{B}_{bc}$ 내에서 온라인 최적화를 수행한다.

- **성능 보장**: WARM-STAGGER의 성능은 오프라인 BC의 보장치와 대화형 STAGGER의 보장치 중 더 좋은 쪽을 따른다.
$$J(\pi_E) - J(\hat{\pi}) \le O \left( \min \left[ \frac{R \log(\tilde{B}/\delta)}{N_{\text{off}}}, \frac{\mu H \log(B_{bc}/\delta)}{N_{\text{int}}} \right] \right)$$

## 📊 Results

### 1. MuJoCo 연속 제어 실험

- **환경**: Ant, Hopper, HalfCheetah, Walker2D
- **설정**: $H=1000$, MLP 전문가 정책 사용. 대화형 주석 비용 $C$를 오프라인 주석 비용의 배수로 설정 ($C=1$ 또는 $C=2$).
- **결과**:
  - $C=1$일 때, STAGGER는 BC보다 훨씬 적은 수의 주석(약 50% 수준)만으로도 대등하거나 더 높은 성능을 보였다.
  - $C=2$인 비용 민감 설정에서 WARM-STAGGER는 적절한 오프라인 데이터 양을 확보했을 때 BC와 STAGGER 모두를 능가하는 효율성을 보였다.

### 2. 합성 MDP 실험 (Theoretical Example)

특정 MDP 구조(이상적 상태 $E$, 회복 가능 상태 $E'$, 실패 상태 $B$, 리셋 상태 $B'$)를 통해 다음을 증명하였다.

- **BC**: $E'$에 대한 커버리지가 낮아 compounding error로 인해 $B'$에 갇히며 성능이 저하된다.
- **STAGGER**: 초기 정책이 매우 불안정하여 $E$를 효율적으로 탐색하지 못하는 **cold-start** 문제로 인해 학습 속도가 매우 느리다.
- **WARM-STAGGER**: 오프라인 데이터로 $E$를 먼저 학습하여 cold-start를 방지하고, 소수의 대화형 쿼리로 $B'$에서의 회복 방법을 배워 최적 성능에 빠르게 도달한다.

## 🧠 Insights & Discussion

본 논문은 모방 학습의 효율성 분석에서 '비용'을 어떻게 정의하느냐에 따라 결론이 완전히 달라질 수 있음을 보여주었다. 궤적 단위의 비용 측정은 대화형 학습의 가치를 과소평가하게 만들지만, 상태 단위의 비용 측정은 대화형 및 하이브리드 접근법의 강력한 이점을 드러낸다.

**강점**:

- 이론적 증명과 실제 시뮬레이션, 그리고 연속 제어 벤치마크 실험을 통해 일관된 결론을 제시하였다.
- 특히 하이브리드 학습이 단순히 두 방법의 합이 아니라, 서로의 단점(cold-start vs compounding error)을 상쇄하는 시너지 효과가 있음을 논리적으로 설명하였다.

**한계 및 논의**:

- 본 논문의 이론적 보장은 **결정론적 실현 가능성(deterministic realizability)**과 **$\mu$-recoverability**라는 강한 가정을 전제로 한다. 실제 환경에서 전문가가 확률적(stochastic)이거나 회복 비용이 매우 높을 때의 보장은 추가적인 연구가 필요하다.
- 이론적 분석은 이산 액션 공간을 기준으로 했으나, 실험은 연속 제어 환경에서 수행되었다. 이 간극을 메우기 위한 이론적 확장이 필요하다.

## 📌 TL;DR

본 연구는 주석 비용을 '상태 단위'로 측정할 때 대화형 모방 학습이 BC보다 이론적으로 우월함을 증명하고, 오프라인 데이터와 대화형 주석을 결합한 **WARM-STAGGER**를 통해 샘플 효율성을 극대화하였다. 하이브리드 방식은 오프라인 학습의 커버리지 부족 문제와 대화형 학습의 초기 학습 불안정성(cold-start)을 동시에 해결하며, 이는 실제 자율 주행이나 로봇 제어와 같이 전문가의 실시간 개입 비용이 궤적 전체 라벨링보다 저렴한 환경에서 매우 중요한 역할을 할 것으로 기대된다.
