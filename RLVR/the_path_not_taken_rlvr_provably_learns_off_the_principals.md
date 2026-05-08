# The Path Not Taken: RLVR Provably Learns Off the Principals

Hanqing Zhu et al. (2025)

## 🧩 Problem to Solve

본 논문은 검증 가능한 보상을 이용한 강화학습(Reinforcement Learning with Verifiable Rewards, RLVR)이 거대 언어 모델(LLM)의 추론 능력을 비약적으로 향상시키면서도, 실제로는 극히 일부의 파라미터만을 수정하는 것처럼 보이는 '희소성(sparsity)의 역설'을 해결하고자 한다. 일반적으로 높은 계산 비용이 소요되는 RL 과정이 상당한 성능 향상을 가져오기 위해서는 광범위한 가중치 변경이 필요할 것이라고 예상하지만, 최근 연구들은 RL이 SFT(Supervised Fine-Tuning)에 비해 훨씬 희소한 업데이트를 수행한다는 상반된 결과를 보여주었다.

따라서 본 연구의 목표는 RLVR에서 발생하는 업데이트 희소성의 근본적인 메커니즘을 규명하고, RL이 파라미터 공간의 어느 영역에서 어떻게 학습하는지를 이론적, 실험적으로 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 RLVR의 업데이트 희소성이 단순한 현상이 아니라, 사전 학습된 모델의 기하학적 구조에 기반한 **'모델 조건분 최적화 편향(model-conditioned optimization bias)'**의 결과임을 밝혀낸 것이다.

연구진은 이를 설명하기 위해 **Three-Gate Theory**를 제안한다. 이 이론에 따르면 RLVR의 업데이트는 KL 제약(Gate I)에 의해 제한되고, 모델의 기하학적 구조(Gate II)에 의해 Principal directions(주요 방향)가 아닌 저곡률 영역으로 유도되며, 마지막으로 $bf16$의 정밀도 한계(Gate III)로 인해 미세한 업데이트들이 가려지면서 겉보기에 희소한 패턴으로 나타나게 된다. 결론적으로 RLVR는 Principal weights를 피해서 학습하는 **'Off-principal'** 학습 체계를 따른다는 점을 증명하였다.

## 📎 Related Works

기존 연구들은 주로 RL과 SFT의 하위 작업 성능 비교에 집중해 왔다. 특히 Mukherjee et al. (2025)은 RL이 SFT보다 훨씬 적은 수의 파라미터를 수정한다는 점을 관찰하여 업데이트 희소성을 보고하였다. 또한, SFT 분야에서는 모델의 가장 영향력 있는 경로인 Principal weights를 식별하고 이를 타겟팅하여 효율적으로 튜닝하는 PEFT(Parameter-Efficient Fine-Tuning) 방법론(예: PiSSA)들이 제안되었다.

본 논문은 이러한 기존 접근 방식들이 SFT의 최적화 기하학에 지나치게 맞춰져 있음을 지적한다. RLVR는 SFT와는 완전히 다른 최적화 영역(optimization regime)에서 작동하므로, SFT 시대의 PEFT 기법들을 RL에 그대로 적용하는 것이 부적절할 수 있다는 차별점을 제시한다.

## 🛠️ Methodology

### 1. Three-Gate Theory (메커니즘 설명)

RLVR의 최적화 역학을 설명하는 세 가지 단계의 필터링 과정은 다음과 같다.

* **Gate I: KL Anchor (제약 단계)**
    On-policy RL의 목적 함수는 $\pi_{\text{ref}}$와의 KL 발산을 제약한다. 이는 파라미터 업데이트 $\Delta \theta$가 현재 정책에서 너무 멀어지지 않도록 묶어두는 'leash' 역할을 하며, 한 단계 업데이트 시 가중치 변화의 Frobenius norm을 제한한다.
    $$\|\Delta W\|_F \le \sqrt{2K}/\mu$$
    여기서 $K$는 KL 예산, $\mu$는 곡률의 하한선이다.

* **Gate II: Model Geometry (유도 단계)**
    KL 제약 하에서 업데이트는 모델의 에너지 구조상 곡률이 낮은(low-curvature) 방향으로 흐르는 경향이 있다. 사전 학습된 모델은 매우 구조화된 기하학적 특성을 가지고 있으며, RLVR는 이 구조를 파괴하지 않으면서 스펙트럼을 보존하는 Off-principal subspace로 업데이트 방향이 유도된다.

* **Gate III: Precision (필터링 단계)**
    $bf16$ 정밀도는 맨티사(mantissa) 비트가 제한적이다. Gate II에 의해 유도된 Off-principal 영역의 업데이트들은 매우 미세한 $\text{micro-updates}$ 형태로 발생하는데, 이 값들이 $bf16$의 최소 표현 단위인 ULP(Unit in the Last Place)보다 작을 경우 실제 저장 값에 반영되지 않는다. 이로 인해 분석 시에는 가중치가 변하지 않은 것으로 보여 '희소성'이라는 표면적 결과가 나타난다.

### 2. 분석 방법론

* **Spectral Analysis:** Singular Value Decomposition(SVD)을 통해 Principal subspace의 회전 각도(Principal angle)와 스펙트럼 드리프트(Spectrum drift)를 측정하여, RLVR가 사전 학습된 구조를 얼마나 보존하는지 분석한다.
* **Principal Weights Proxy:** 고곡률 방향을 직접 계산하는 대신, 저차원 근사 후 크기가 가장 큰 가중치들을 Principal weights로 정의하고, RLVR의 업데이트 마스크와 이들의 겹침 정도(Overlap ratio)를 측정한다.
* **Causal Intervention:** 모델의 가중치 행렬에 직교 회전(Orthogonal rotation)이나 헤드 치환(Head permutation)을 적용하여 기하학적 구조를 인위적으로 뒤섞은 후, 최적화 편향이 사라지는지 확인한다.

## 📊 Results

### 1. RLVR vs SFT의 최적화 역학 비교

* **스펙트럼 보존:** RLVR는 SFT와 달리 사전 학습된 모델의 Singular value 프로필을 거의 그대로 유지한다. Principal subspace의 회전이 매우 적으며, 이는 RLVR가 모델의 근본적인 구조를 파괴하지 않고 학습함을 의미한다.
* **Off-principal 업데이트:** RLVR의 업데이트 마스크는 Principal weights와는 거의 겹치지 않는(sub-random overlap) 반면, 저크기 가중치(low-magnitude weights) 영역과는 강하게 겹치는(super-random overlap) 특성을 보였다. 반면 SFT는 Principal weights를 집중적으로 타겟팅한다.

### 2. 기하학적 구조의 인과 관계

* 모델의 특정 레이어에 회전 및 치환 개입을 가했을 때, 해당 레이어에서의 업데이트 중복도가 무작위 수준으로 급락하였다. 이는 RLVR의 최적화 편향이 데이터나 알고리즘이 아닌, **사전 학습된 모델의 기하학적 구조**에서 기인한다는 강력한 증거가 된다.

### 3. PEFT 방법론에 대한 영향 분석

* **Sparse Fine-tuning:** Principal weights만을 업데이트하는 마스크($M_{\text{princ}}$)를 사용했을 때 학습 궤적이 가장 좋지 않았으며 성능 또한 크게 하락했다. 반면, Non-principal 및 Low-magnitude 영역을 업데이트하는 'Safe Mask'는 Full-parameter RLVR의 성능과 KL 궤적을 거의 완벽하게 재현하였다.
* **LoRA 변체:** Principal directions를 타겟팅하는 PiSSA는 표준 LoRA보다 나은 성능을 보이지 않았으며, 특히 학습률을 높일 경우 Principal 방향으로의 강제 업데이트가 발생하여 학습 붕괴(collapse)가 더 빈번하게 일어났다.

## 🧠 Insights & Discussion

본 논문은 RLVR가 SFT와는 완전히 분리된 최적화 체계(disjoint optimization regime)에서 작동한다는 점을 시사한다. SFT는 모델의 주요 경로(Principal directions)를 수정하여 빠르게 타겟 분포에 도달하려 하지만, RLVR는 기존의 지식 구조를 최대한 보존하면서 그 사이의 미세한 틈새(Off-principal subspace)를 통해 추론 능력을 정교화한다.

이러한 발견은 매우 중요한 실무적 함의를 갖는다. 현재 많은 PEFT 기법들이 SFT의 최적화 역학에 기반하여 설계되었기 때문에, 이를 RLVR에 그대로 적용하는 것은 이론적으로 맞지 않으며 실제 성능 저하나 불안정성을 초래할 수 있다. 따라서 RLVR를 위한 효율적인 튜닝 방법론을 설계하려면, Principal weights를 찾는 것이 아니라 오히려 이를 피하거나 보존하는 **'Geometry-aware, RL-native'** 접근 방식이 필요하다.

## 📌 TL;DR

RLVR의 파라미터 업데이트 희소성은 $bf16$ 정밀도와 모델의 사전 학습된 기하학적 구조가 결합되어 나타나는 표면적 현상이다. 이론적 분석과 실험을 통해 RLVR는 모델의 주요 방향(Principal directions)을 피해 저곡률 영역에서 학습함을 증명하였다. 이는 SFT 기반의 PEFT 기법(예: PiSSA, Principal-targeted sparse tuning)이 RLVR에서 효과가 없거나 불안정한 이유를 설명하며, 향후 RL 전용의 기하학적 인식 PEFT 알고리즘 설계의 필요성을 제시한다.
