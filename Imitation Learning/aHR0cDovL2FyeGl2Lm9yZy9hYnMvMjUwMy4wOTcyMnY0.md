# The Pitfalls of Imitation Learning when Actions are Continuous

Max Simchowitz, Daniel Pfrommer, Ali Jadbabaie (2025)

## 🧩 Problem to Solve

본 논문은 이산 시간(discrete-time), 연속 상태 및 액션 제어 시스템에서 전문가의 시연을 모방하는 Imitation Learning (IL)의 근본적인 한계를 분석한다. 특히, 학습자가 환경이나 전문가와 추가로 상호작용할 수 없는 '비상호작용 설정(non-interactive setting)'에서의 성능 저하 문제를 다룬다.

가장 핵심적인 문제는 **Compounding Error**(복합 오차)이다. 학습 과정에서 발생하는 작은 예측 오차가 시간이 흐름에 따라 누적되어, 결국 학습자가 전문가의 궤적에서 기하급수적으로 벗어나는 현상을 의미한다. 기존의 이산 토큰(discrete token) 기반 모방 학습에서는 이 오차가 문제의 horizon $H$에 대해 다항식(polynomial) 수준으로 증가하며, 특정 손실 함수를 사용할 경우 완전히 제거될 수도 있다는 결과가 있었다.

그러나 본 연구는 액션 공간이 연속적일 때, 시스템이 제어 이론적 관점에서 **Exponential Stability**(지수적 안정성)를 만족하고 전문가가 매끄러운(smooth) 결정론적 정책을 따르더라도, '단순한' 형태의 모방 정책은 실행 오차가 horizon $H$에 대해 지수적으로 증가하는 $\exp(\Omega(H))$의 하한을 가짐을 이론적으로 증명하고자 한다.

## ✨ Key Contributions

본 논문의 중심적인 직관은 **학습자가 시스템의 동역학(dynamics)을 알지 못하는 상태에서, 전문가의 데이터만으로는 서로 다른 두 개의 안정적인 시스템을 구분할 수 없다**는 점에 있다. 만약 학습자가 선택한 정책이 두 시스템 중 하나는 안정화시키지만 다른 하나는 불안정하게 만든다면, 실제 환경이 후자일 경우 오차는 지수적으로 증폭된다.

주요 기여 사항은 다음과 같다:
1. **단순 정책의 한계 증명**: Smooth하고 Deterministic한 전문가와 시스템을 모방할 때, 학습자가 'Simple' 정책(Smooth Deterministic 또는 단순히 확률적인 정책)을 사용할 경우 execution error가 training error보다 지수적으로 커짐을 보였다. 이는 Behavior Cloning(BC)뿐만 아니라 Offline RL, Inverse RL 등 비상호작용 알고리즘 전체에 적용된다.
2. **복잡한 정책의 필요성 제시**: Diffusion Policy나 Action-chunking과 같은 최신 로봇 학습 기법들이 단순히 '다중 모드(multi-modal)' 데이터를 처리하기 위해서가 아니라, 결정론적이고 단순한 전문가를 모방할 때조차 이러한 복잡한 파라미터화가 compounding error를 완화하는 데 필수적일 수 있음을 이론적/실험적으로 제시하였다.
3. **동역학 불안정성의 영향**: 시스템 동역학 자체가 불안정할 경우, 어떤 알고리즘이나 정책 클래스를 사용하더라도 비상호작용 IL에서는 지수적 오차가 불가피함을 증명하였다.
4. **데이터 커버리지의 중요성**: 전문가 데이터가 충분히 '넓게 퍼져(well-spread)' 있다면, 단순한 Behavior Cloning만으로도 compounding error를 피할 수 있음을 보였다.

## 📎 Related Works

기존의 Imitation Learning 연구들은 주로 다음과 같은 방향으로 compounding error를 해결하려 했다:
- **상호작용 기반 방법**: DAGGER와 같이 전문가에게 실시간으로 피드백을 받는 방식이 대표적이다.
- **데이터 증강**: 전문가 데이터의 분포를 넓혀 실패 모드에 대한 커버리지를 높이는 방식이다.
- **이론적 접근**: Ross and Bagnell (2010) 등은 이산 영역에서 오차가 다항식 수준으로 증가함을 보였고, Foster et al. (2024)은 log-loss를 통해 horizon의 영향을 제거할 수 있음을 보였다.

본 논문은 이러한 기존 이론들이 **$\ell_2$ 거리나 Hellinger 거리**와 같은 강력한 메트릭에서 전문가 정책을 추정할 수 있다는 가정에 의존하고 있음을 지적한다. 하지만 연속 액션 공간의 결정론적 정책에서는 이러한 메트릭에서의 추정이 불가능(impossible)하며, 이로 인해 기존의 다항식 오차 바운드가 연속 제어 시스템에서는 적용되지 않는다는 차별점을 가진다.

## 🛠️ Methodology

### 1. 시스템 모델 및 정의
상태 $x \in \mathbb{R}^d$, 액션 $u \in \mathbb{R}^m$, 동역학 $x_{t+1} = f(x_t, u_t)$인 시스템을 가정한다. 전문가 정책은 $\pi^\star: X \to U$이며, 학습자는 $n$개의 길이 $H$인 궤적 샘플 $S_{n,H}$를 통해 정책 $\hat{\pi}$를 학습한다.

### 2. Exponential Incremental Input-to-State Stability (E-IISS)
본 논문은 시스템의 '양호함'을 판단하기 위해 E-IISS라는 강력한 안정성 조건을 사용한다. 이는 두 궤적의 차이가 시간이 지남에 따라 지수적으로 감소함을 의미하며, 다음과 같이 정의된다:
$$\|x_{t+1} - x'_{t+1}\| \le C\rho^t \|x_1 - x'_1\| + \sum_{1 \le k \le t} C\rho^{t-k} \|u_k - u'_k\|$$
여기서 $\rho \in [0, 1)$는 감쇠율을 나타낸다.

### 3. Simple Policies 정의
학습자가 사용하는 정책 $\hat{\pi}$가 다음과 같을 때 **Simply-stochastic** (또는 Simple) 하다고 정의한다:
- 정책의 평균 $\text{mean}[\hat{\pi}](x)$가 $L$-Lipschitz이고 $M$-smooth하다.
- 평균으로부터의 편차(noise) 분포가 상태 $x$에 의존하지 않는다. (예: 고정된 공분산을 가진 가우시안 정책)

### 4. 이론적 하한 구축 (Proof Intuition)
연구진은 다음과 같은 'challenging pair' $(\pi_i, f_i), i \in \{1, 2\}$를 설계한다:
- 두 시스템 모두 개별적으로는 안정적이다.
- 하지만 $\pi_1$은 $f_2$를 불안정하게 만들고, $\pi_2$는 $f_1$을 불안정하게 만든다.
- 전문가 데이터 분포상에서 두 시스템은 서로 구분이 불가능하다.

결과적으로 학습자는 $f_1$과 $f_2$ 중 무엇이 실제 시스템인지 알 수 없으며, 어떤 $\hat{\pi}$를 선택하더라도 최소 한 쪽의 시스템에서는 폐루프(closed-loop) 불안정성이 발생하여 오차가 지수적으로 증폭된다.

## 📊 Results

### 1. 이론적 결과 (Theorems)
- **Theorem 1 & 2**: Simple 정책을 사용할 경우, 실행 오차(execution error) $R^{\text{cost}}$는 학습 오차 $R^{\text{expert}, L_2}$보다 $\exp(\Omega(H))$배 더 크다.
- **Theorem 3**: 일반적인 확률적 정책(non-simple)을 사용하더라도 여전히 오차가 발생하지만, 그 정도는 Simple 정책보다는 낮을 수 있다 (임계값 $\varepsilon^{1-\Omega(1)}$까지).
- **Theorem 4**: 동역학이 unstable할 경우, 어떠한 정책 클래스나 알고리즘을 사용하더라도 비상호작용 IL에서는 지수적 오차가 불가피하다.
- **Theorem 5**: 데이터 분포가 충분히 'well-spread' 하다면, 단순 BC로도 compounding error를 피할 수 있다.

### 2. 수치 실험
'Hard' stable dynamical system을 구축하여 다양한 방법론의 성능을 비교하였다.
- **측정 지표**: Rollout Cost ($\max_{s \le t} |\langle x_s, e_1 \rangle|$)
- **비교 대상**: 
    - $L_2$ Behavior Cloning (MLP policy)
    - Diffusion Policy (DP)
    - DP + Action-chunking (4-chunk, 8-chunk)
    - Random Noise / Open Loop

**주요 결과**:
- **Behavior Cloning**: 전형적인 지수적 오차 증가 양상을 보이며 성능이 가장 낮았다.
- **Diffusion Policy**: BC보다 훨씬 나은 성능을 보였으며, 이는 Diffusion 모델이 'non-simply-stochastic'한 성질을 가지기 때문으로 해석된다.
- **Action-chunking**: DP에 chunking을 결합했을 때 성능이 더욱 향상되었다. 이는 chunk 단위의 open-loop 실행이 오차 누적을 일시적으로 차단하기 때문으로 분석된다.
- **Random Noise**: 놀랍게도 단순 랜덤 노이즈가 학습된 BC 정책보다 나은 경우가 있었는데, 이는 시스템 자체의 open-loop stability 덕분에 오차가 누적되지 않았기 때문이다.

## 🧠 Insights & Discussion

본 논문은 로봇 학습 분야에서 관습적으로 사용되던 기법들의 이론적 근거를 제시한다. Diffusion Policy나 Action-chunking이 단순히 복잡한 다중 모드 데이터를 표현하기 위한 도구가 아니라, **연속 액션 공간에서 발생하는 근본적인 compounding error를 억제하기 위한 기제**일 수 있음을 시사한다.

특히, '단순한' 정책(Deterministic/Gaussian)이 가진 한계는 학습 알고리즘의 문제가 아니라 **정책의 파라미터화(parameterization) 자체의 문제**라는 점이 중요하다. 이는 Offline RL이나 Inverse RL과 같이 정교한 알고리즘을 사용하더라도, 최종적으로 출력되는 정책이 Simple하다면 동일한 지수적 오차의 굴레에서 벗어날 수 없음을 의미한다.

한계점으로는, 본 연구가 주로 최악의 경우(worst-case) 하한을 분석했다는 점이다. 실제 환경에서는 시스템의 구조적 특성이나 데이터의 특성에 따라 이러한 병목 현상이 덜 나타날 수 있다. 또한, 제시된 'well-spread' 조건이 실제 데이터셋에서 어느 정도 수준으로 충족되는지에 대한 실증적 분석이 추가로 필요하다.

## 📌 TL;DR

본 논문은 연속 액션 공간의 비상호작용 모방 학습에서, 시스템이 안정적이더라도 **단순한 정책(Smooth, Deterministic/Gaussian)은 실행 오차가 horizon $H$에 대해 지수적으로 증가하는 $\exp(\Omega(H))$의 치명적인 결함**이 있음을 이론적으로 증명하였다. 이를 통해 Diffusion Policy나 Action-chunking과 같은 복잡한 정책 표현 방식이 단순한 전문가를 모방할 때조차 필수적일 수 있음을 보였으며, 데이터의 커버리지(well-spreadness)가 이 문제를 해결하는 핵심 열쇠임을 제시하였다.