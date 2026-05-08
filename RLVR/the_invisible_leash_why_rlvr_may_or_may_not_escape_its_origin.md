# THE INVISIBLE LEASH? WHY RLVR MAY OR MAY NOT ESCAPE ITS ORIGIN

Fang Wu, Weihao Xuan, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, Yejin Choi (2025)

## 🧩 Problem to Solve

본 논문은 최근 DeepSeek-R1이나 OpenAI-o1과 같은 대형 추론 모델(Large Reasoning Models)의 성능 향상에 핵심적인 역할을 한 **Reinforcement Learning with Verifiable Rewards (RLVR)**의 근본적인 메커니즘을 분석한다. RLVR은 정답 여부가 명확히 확인 가능한 보상 함수를 사용하여 모델을 최적화하는 방법이다.

연구의 핵심 문제는 **RLVR이 모델의 추론 경계(Reasoning Boundary)를 실제로 확장하여 새로운 해결 능력을 부여하는 것인지, 아니면 단순히 베이스 모델이 이미 알고 있던 정답 경로의 확률을 높여 정밀도(Precision)를 향상시키는 것인지**를 규명하는 것이다. 이는 RLVR이 진정한 의미의 '추론 능력 확장'을 이끌어내는지, 아니면 단순한 '기존 지식의 증폭'에 불과한지를 판단하는 매우 중요한 학술적 질문이다.

## ✨ Key Contributions

본 논문의 중심적인 직관은 RLVR이 **'보이지 않는 목줄(Invisible Leash)'**에 묶여 있다는 것이다. 즉, RLVR은 베이스 모델이 생성할 수 있는 확률 분포의 범위 내에서만 최적화를 수행하며, 초기 분포를 완전히 벗어난 새로운 추론 경로를 발견하는 능력은 극히 제한적이라는 점을 제시한다.

주요 기여 사항은 다음과 같다:

- **Empirical Support(경험적 지지)**라는 개념을 도입하여, 모델이 유한한 샘플링 하에서 현실적으로 발견할 수 있는 정답 집합을 정의하고 이를 측정하는 프레임워크를 제안하였다.
- RLVR이 정답의 다양성을 희생하고 정밀도를 높이는 **Precision-Diversity Trade-off**가 도메인에 관계없이 보편적으로 나타남을 실험적으로 입증하였다.
- **토큰 수준의 엔트로피(Token-level Entropy)**와 **정답 수준의 엔트로피(Answer-level Entropy)**가 서로 다르게 움직이는 현상을 발견하여, 지역적인 불확실성 증가가 반드시 전역적인 탐색 확장으로 이어지지 않음을 보였다.

## 📎 Related Works

최근 RLVR의 성공으로 인해 많은 연구가 진행되었으나, 일부 연구자들은 RLVR이 베이스 모델의 기존 능력을 보수적으로 최적화하는 수준이라고 주장하는 반면, 다른 이들은 특정 도메인에서 추론 능력을 실질적으로 확장한다고 주장하며 대립해 왔다. 특히 `pass@k` 지표에서 베이스 모델이 큰 $k$ 값에 대해 RLVR 모델보다 더 좋은 성능을 보이는 경우가 보고되었는데, 이는 RLVR 모델이 정답의 범위를 좁혔을 가능성을 시사한다. 본 논문은 이러한 논쟁에 대해 이론적 증명과 광범위한 실험을 통해 RLVR의 한계를 명확히 규정함으로써 기존 접근 방식과의 차별점을 둔다.

## 🛠️ Methodology

### 1. 정답 접근성 및 경험적 지지 (Empirical Support)

논문은 정답 집합 $C = \{y \in Y | R(x,y) = 1\}$를 정의하고, 모델 $p$가 정답을 발견할 수 있는 능력을 $\text{supp}(p) := \{y \in C | p(y|x) > 0\}$로 정의한다. 하지만 실제 소프트맥스 층에서는 모든 토큰의 확률이 0보다 크므로, 실질적인 관찰 가능성을 판단하기 위해 임계값 $\epsilon$을 도입한 **Empirical Support**를 다음과 같이 정의한다:
$$\text{supp}_\epsilon(q) := \{y \in C | q(y|x) > \epsilon\}$$

### 2. 지지 동역학 지표 (Support Dynamics Metrics)

RLVR 전후의 정답 집합 변화를 측정하기 위해 네 가지 범주(보존 $P$, 확장 $E$, 축소 $S$, 지지 없음 $O$)를 정의하고 다음과 같은 지표를 제안한다:

- **Support Retention Rate (SRR)**: $\frac{P}{P + S}$ (기존 정답을 얼마나 잘 유지하는가)
- **Net Discovery Rate (NDR)**: $\frac{E}{P + E}$ (새로운 정답을 얼마나 발견했는가)
- **Support Dynamic Score (SDS)**: SRR과 NDR의 조화 평균
- **Net Support Change Rate (NSCR)**: $\frac{E - S}{P + E + S}$ (전체적인 정답 범위의 순증감)

### 3. 이론적 분석 및 방정식

논문은 RLVR이 베이스 모델 $q$의 지지를 벗어날 수 없음을 이론적으로 증명한다.

- **Support Preservation (Theorem C.1)**: RLVR은 온-폴리시(On-policy) 샘플링에 의존하므로, $\text{supp}(\pi_\theta) \subseteq \text{supp}(q)$가 성립한다. 즉, 베이스 모델이 확률 0으로 생성하는 정답은 RLVR을 통해서도 절대 발견할 수 없다.
- **Variational Perspective (Prop C.4)**: RLVR의 목적 함수는 다음과 같은 지수적 기울기(Exponential Tilting) 형태로 최적화된다:
$$\pi^*(y|x) \propto q(y|x) \cdot \exp(\beta R(x,y))$$
여기서 $\beta$는 보상과 KL 발산 사이의 트레이드오프를 조절하는 하이퍼파라미터이다. 이는 RLVR이 베이스 분포 $q$를 기반으로 보상이 높은 쪽으로 확률 질량을 재배치할 뿐, 새로운 모드를 생성하지 않음을 의미한다.

## 📊 Results

### 1. 실험 설정

- **모델**: ProRL-1.5B, Nemotron-7B/14B, Skywork-OR1-7B, Phi4-Reason-Plus-14B 등 다양한 규모의 모델 사용.
- **데이터셋**: 수학(MATH500, AIME 2024/2025, OlympiadBench 등) 및 비수학 추론(SimpleQA, LiveBench, SciBench, Reasoning Gym) 작업.
- **평가 지표**: `pass@1` (정밀도), `pass@k` (커버리지), SRR, NDR, NSCR.

### 2. 주요 결과

- **지배적인 보존과 제한적 확장**: 모든 모델에서 SRR은 매우 높게($0.93 \sim 0.99$) 나타난 반면, NDR은 극히 낮았다($\le 0.04$). 이는 RLVR이 새로운 해결책을 찾기보다 기존 것을 유지하는 데 치중함을 보여준다.
- **확장보다 큰 축소**: 대부분의 경우 $S$(축소)가 $E$(확장)보다 훨씬 컸으며, NSCR 값은 일관되게 음수($-0.01$ to $-0.06$)로 나타났다. 이로 인해 `pass@1`은 향상되지만, 매우 큰 $k$ 값에서의 `pass@k` 성능은 오히려 베이스 모델보다 떨어지는 역설적인 결과가 발생한다.
- **엔트로피의 디커플링 (Decoupling)**:
  - **Answer-level Entropy**: RLVR 이후 일관되게 감소한다. 이는 모델이 소수의 정답 경로로 수렴(Mode Collapse)함을 의미한다.
  - **Token-level Entropy**: 모델에 따라 증가하는 경우가 많다. 이는 생성 과정에서 지역적인 불확실성은 높아지지만, 결국 도달하는 최종 정답은 더 적어지는 '전역적 탐색 없는 지역적 무작위성' 현상을 보여준다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 RLVR이 단순히 성능을 높이는 도구가 아니라, 베이스 모델의 확률 분포라는 강력한 제약 조건 하에서 작동하는 **'정밀도 향상 도구(Precision Enhancer)'**임을 명확히 하였다. 특히 이론적 증명과 실험적 지표(SRR, NDR)를 결합하여 RLVR의 한계를 정량적으로 제시한 점이 매우 뛰어나다.

### 한계 및 비판적 논의

- **초기화 의존성**: RLVR의 성능이 전적으로 베이스 모델의 초기 지지 집합(Support set)에 의존한다는 점은, 추론 능력을 근본적으로 확장하기 위해서는 단순한 RLVR 이상의 메커니즘이 필요함을 시사한다.
- **탐색의 부재**: 현재의 RLVR 레시피는 보상이 높은 경로를 강화하는 '착취(Exploitation)'에는 능숙하지만, 새로운 경로를 찾는 '탐색(Exploration)' 기제는 부족하다.
- **해결 방안**: 저자들은 이를 해결하기 위해 명시적인 탐색 메커니즘(Explicit Exploration)이나 확률 질량을 저밀도 영역에 강제로 주입하는 하이브리드 전략의 필요성을 제안한다.

## 📌 TL;DR

본 논문은 RLVR이 모델의 추론 능력을 실제로 확장하는지 분석하여, 실제로는 베이스 모델이 이미 가지고 있던 정답들의 확률을 높여 **정밀도를 향상시킬 뿐, 새로운 추론 경로를 발견하는 능력은 거의 없음**을 밝혀냈다. RLVR은 정답의 다양성을 희생시켜 정밀도를 얻는 '보이지 않는 목줄'에 묶여 있으며, 진정한 추론 확장력을 갖추기 위해서는 단순한 보상 최적화를 넘어선 새로운 탐색 전략이 필요하다.
