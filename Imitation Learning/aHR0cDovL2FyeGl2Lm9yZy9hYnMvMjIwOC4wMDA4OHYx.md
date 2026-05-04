# Improved Policy Optimization for Online Imitation Learning

J. Wilder Lavington, Sharan Vaswani, Mark Schmidt (2022)

## 🧩 Problem to Solve

본 논문은 **Online Imitation Learning (OIL)** 환경에서 정책 최적화 알고리즘의 이론적 분석과 실무적 적용 사이의 간극을 메우는 것을 목표로 한다. OIL의 핵심 과제는 에이전트가 환경과 능동적으로 상호작용하며 전문가(Expert)의 행동을 모방하는 정책을 찾는 것이다.

가장 널리 사용되는 OIL 알고리즘인 **DAGGER**는 이론적으로 손실 함수가 강볼록성(Strong-convexity)을 가질 때 낮은 Regret을 보장하지만, 실제로는 신경망과 같은 비볼록(Non-convex) 함수를 사용하는 환경에서도 뛰어난 성능을 보인다. 저자들은 이러한 이론과 실제의 불일치가 정책 클래스의 **표현력(Expressivity)**에서 기인한다고 분석하며, 이를 이론적으로 규명하고 더 안정적인 최적화 방법론을 제안하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **FTL(Follow-the-Leader)의 이론적 재분석**: 정책 클래스가 전문가 정책을 포함할 만큼 충분히 표현력이 있다면, FTL(및 DAGGER)이 OIL 환경에서 **Constant Regret**(상수 후회도)을 달성함을 증명하였다. 이는 기존의 강볼록성 가정 없이도 가능함을 보여준 것이다.
2.  **FTRL 및 AdaFTRL 제안**: FTL에서 발생할 수 있는 진동(Oscillation) 현상을 방지하기 위해 정규화 항을 추가한 **Follow-the-Regularized-Leader (FTRL)** 및 그 적응형 변형인 **AdaFTRL**을 OIL 설정에 도입하였다.
3.  **메모리 효율적 구현**: FTRL의 업데이트 식을 재구성하여, 모델 크기와 반복 횟수에 비례해 증가하던 메모리 요구량을 FTL 수준인 $O(m+T)$로 줄인 효율적인 구현 방법을 제시하였다.
4.  **Regret Bound 증명**: 손실 함수가 매끄럽고(Smooth) 볼록(Convex)하다고 가정할 때, FTRL과 AdaFTRL이 전문가 정책을 포함하는 경우 Constant Regret을, 그렇지 않은 최악의 경우에도 Sublinear Regret($O(\sqrt{T})$)을 달성함을 증명하였다.

## 📎 Related Works

논문은 OIL을 **Online Convex Optimization (OCO)** 프레임워크로 해석한 기존 연구들을 언급한다. 특히 DAGGER는 OCO의 FTL 알고리즘의 일종으로 볼 수 있으며, 기존 이론은 손실 함수의 강볼록성을 요구했다.

또한, **ILOA (Imitation Learning from Observation Alone)** 및 **Apprenticeship Learning**과의 차이점을 명시한다. ILOA는 전문가의 궤적만을 사용하여 보상 표면을 재구성하는 복잡한 하위 문제를 풀어야 하지만, 본 논문에서 다루는 OIL은 전문가 오라클(Oracle)로부터 직접 최적 행동을 제공받아 최적화 문제로 단순화하여 해결한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 문제 정의 및 목적 함수
에이전트는 MDP $\langle S, A, P, r, \rho, \gamma \rangle$에서 전문가 정책 $\pi^e$를 모방하려 한다. 두 정책 분포 사이의 차이를 측정하는 발산 함수(Divergence) $D$가 주어졌을 때, 목적 함수는 다음과 같다.
$$\pi^* = \min_{\pi \in \Pi} \mathbb{E}_{s \sim d^\pi} [D(\pi(\cdot|s), \pi^e(\cdot|s))]$$
여기서 $d^\pi$는 정책 $\pi$에 의한 상태 방문 분포이다.

### 2. Follow-the-Leader (FTL)
FTL은 매 단계 $t$에서 지금까지 관측된 모든 손실 함수의 합을 최소화하는 파라미터 $w_{t+1}$을 선택한다.
$$w_{t+1} = \arg \min_{w \in W} F_t(w) := \sum_{i=1}^t l_i(w)$$
이 방식은 과거의 모든 데이터를 사용하는 **'Offline' 업데이트** 방식이므로 샘플 효율성이 높다.

### 3. Follow-the-Regularized-Leader (FTRL)
FTL의 진동 문제를 해결하기 위해 근접 정규화(Proximal Regularization) 항을 추가한다.
$$w_{t+1} = \arg \min_{w \in W} \left[ \sum_{i=1}^t l_i(w) + \sum_{i=1}^t \frac{\sigma_i^2}{2} \|w - w_i\|^2 \right]$$
위 식은 모든 과거 파라미터를 저장해야 하므로 메모리 소모가 크다. 저자들은 이를 다음과 같이 재구성하여 메모리 효율성을 높였다.
$$w_{t+1} = \arg \min_{w \in W} \left[ \sum_{i=1}^t l_i(w) - \left\langle w, \sum_{i=1}^{t-1} \nabla l_i(w_t) \right\rangle + \frac{1}{2\eta_t} \|w - w_t\|^2 \right]$$
여기서 $\eta_t$는 학습률이며, $\eta_t := 1 / (\sum_{i=1}^t \sigma_i)$로 정의된다.

### 4. Adaptive FTRL (AdaFTRL)
하이퍼파라미터 $\eta$ 설정의 어려움을 해결하기 위해, 기울기의 누적 합을 이용해 $\eta_t$를 동적으로 조절하는 AdaFTRL을 제안한다.
$$\eta_t = \frac{\alpha}{\sqrt{\sum_{i=1}^t \|\nabla l_i(w_i)\|^2}}$$

## 📊 Results

### 실험 설정
- **데이터셋 및 환경**: Mujoco (Hopper, Walker-2D) 및 Atari (Pong, Breakout).
- **비교 대상**: BC (Behavioral Cloning), OGD (Online Gradient Descent), AdaGrad, FTL, FTRL, AdaFTRL.
- **평가 지표**: 누적 보상(Cumulative Reward) 및 평균 누적 손실(Average Cumulative Loss).
- **모델 설정**: 선형 모델(Linear)과 신경망 모델(Neural Network) 두 가지를 모두 테스트하여 볼록/비볼록 환경을 모두 검증하였다.

### 주요 결과
1.  **Offline 업데이트의 우위**: FTL과 FTRL 변형들이 OGD, AdaGrad보다 일관되게 우수한 성능을 보였다. 이는 과거 데이터를 모두 활용하는 업데이트 방식이 OIL에서 매우 중요함을 시사한다.
2.  **정규화의 효과**: FTRL 계열이 FTL보다 평균 누적 손실 면에서 더 나은 성능을 보였다. 이는 정규화가 학습의 안정성을 높여준다는 것을 의미한다.
3.  **AdaFTRL의 성능**: AdaFTRL은 하이퍼파라미터 튜닝 없이도 FTRL과 유사하거나 더 나은 성능을 보였으며, 특히 손실 값의 감소 속도가 빨랐다.
4.  **표현력의 영향**: 신경망 모델을 사용하여 전문가 정책을 포함할 수 있는 충분한 표현력을 가졌을 때, FTL과 FTRL 모두 매우 빠르게 수렴하며 Constant Regret의 특성을 보였다.

## 🧠 Insights & Discussion

본 논문은 OIL에서 이론과 실제의 괴리가 발생하는 이유가 **'정책 클래스의 표현력'**과 **'데이터 활용 방식(Offline vs Online)'**에 있음을 밝혀냈다.

- **강점**: 단순히 알고리즘을 제안한 것에 그치지 않고, Interpolation(보간) 가정을 통해 FTL이 왜 실제 환경에서 강볼록성 없이도 잘 작동하는지를 수학적으로 증명하였다. 또한 FTRL의 메모리 문제를 해결한 재구성 식은 실용적인 가치가 높다.
- **한계 및 논의**: 실험 결과에서 **평균 손실의 감소가 반드시 보상의 증가로 이어지지는 않는다**는 점이 관찰되었다. 이는 손실 함수 $D$와 실제 MDP의 가치 함수 $V^\pi$ 사이의 관계가 복잡하기 때문이며, 이는 향후 연구 과제로 남겨져 있다. 또한, 최적화 과정에서의 부정확한 계산(Inexact optimization)이 이론적 보장에 어떤 영향을 미치는지에 대해서는 명확히 다루지 않았다.

## 📌 TL;DR

이 논문은 Online Imitation Learning에서 DAGGER(FTL)의 이론적 분석을 통해, 정책 클래스의 표현력이 충분하다면 강볼록성 가정 없이도 Constant Regret을 달성할 수 있음을 증명하였다. 더불어 학습 안정성을 높이기 위해 정규화된 FTRL 및 적응형 AdaFTRL을 제안하고, 이를 메모리 효율적으로 구현하는 방법을 제시하였다. Mujoco와 Atari 실험을 통해 "과거 데이터를 모두 활용하는 Offline 업데이트"와 "적절한 정규화"가 OIL 성능 향상의 핵심임을 입증하였다.