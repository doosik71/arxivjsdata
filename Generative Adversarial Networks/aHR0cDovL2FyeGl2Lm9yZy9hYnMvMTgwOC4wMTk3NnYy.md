# Adversarial Vision Challenge

Wieland Brendel, Jonas Rauber, Alexey Kurakin, Nicolas Papernot, Behar Veliqi, Marcel Salathé, Sharada P. Mohanty, Matthias Bethge (2018)

## 🧩 Problem to Solve

본 논문은 현대 머신 비전 알고리즘이 입력 데이터의 미세하고 인지하기 어려운 섭동(perturbation)인 adversarial examples에 매우 취약하다는 점을 해결하고자 한다. 이러한 취약성은 인간의 시각 인지와 기계의 정보 처리 방식 사이에 상당한 간극이 있음을 시사하며, 특히 자율주행 자동차와 같이 안전이 필수적인(safety-critical) 시스템에서 심각한 보안 문제를 야기한다.

연구의 핵심 문제는 기존의 강건성(robustness) 평가 방식이 불충분하다는 점이다. 많은 모델이 강건해 보이지만, 이는 실제로 강건한 것이 아니라 gradient masking과 같은 단순한 기법을 통해 공격자가 경사도를 찾지 못하게 방해함으로써 얻어진 가짜 보안(false sense of security)인 경우가 많다. 따라서 본 논문의 목표는 강건한 머신 비전 모델과 더 강력하고 범용적인 adversarial attacks를 개발하도록 촉진하는 측정 가능한 경쟁 환경을 구축하는 것이다.

## ✨ Key Contributions

본 연구의 중심적인 설계 아이디어는 모델과 공격 간의 '공진화(co-evolution)'를 유도하는 2인 게임(two-player game) 형태의 경쟁 프레임워크를 제안하는 것이다.

단순히 고정된 공격 세트로 모델을 평가하는 것이 아니라, 모델과 공격자가 서로를 지속적으로 상대하게 함으로써 공격자는 모델의 방어 전략에 적응하고, 모델은 더 강력해진 공격에 대응하여 진화하도록 설계하였다. 특히, 내부 정보(gradient, confidence score 등)를 사용하지 않고 오직 최종 결정(final decision)만을 이용하는 decision-based attacks에 집중함으로써, 실제 보안 시나리오에 더 가까운 극한의 강건성 평가 환경을 제공한다.

## 📎 Related Works

논문에서는 이전의 두 가지 관련 경쟁을 언급하며 본 챌린지의 차별점을 제시한다.

1. **NIPS 2017 Competition**: 모델과 공격을 대결시켰으나, 공격자가 모델에 쿼리를 보낼 수 없는 간접적인 방식이었다. 이로 인해 공격자는 여러 모델에 공통으로 적용되는 generic adversarial examples를 만들어야 했으며, 이는 모델 특정적(model-specific) 공격보다 방어가 훨씬 쉽다는 한계가 있었다.
2. **Robust Vision Benchmark (RVB)**: 지속적인 벤치마크 환경을 제공하며, 공격자가 모델의 confidence score와 gradients를 모두 쿼리할 수 있는 화이트박스(white-box) 설정이었다. 이는 모델의 강건성 평가에는 유용하지만, 실제 보안 시나리오를 대변하기에는 부적절하다.

본 챌린지는 이러한 한계를 극복하기 위해 공격자가 모델에 직접 쿼리를 보낼 수 있게 하여 model-specific 공격을 가능하게 하되, 내부 정보가 아닌 결정값만을 이용하게 함으로써 실제 환경에서의 보안 위협을 더 정확하게 반영한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 작업 정의

본 챌린지는 TINYIMAGENET 데이터셋(64x64 픽셀, 200 클래스, 100,000 이미지)을 기반으로 하며, 세 가지 주요 태스크로 구성된다.

1. **Minimum Untargeted Adversarial Examples 생성**: 주어진 이미지 $s$와 모델 $m$에 대해, 모델이 잘못 분류하게 만들면서 $s$와 가장 유사한(최소 $L_2$ 거리) adversarial image $\hat{s}$를 생성하는 작업이다.
2. **Minimum Targeted Adversarial Examples 생성**: 타겟 클래스가 주어졌을 때, 모델이 해당 타겟 클래스로 분류하도록 유도하는 최소 섭동의 adversarial image를 생성하는 작업이다.
3. **Minimum Adversarial Examples 크기 확대**: 모델 설계자가 공격자가 찾을 수 있는 최소 섭동의 크기($L_2$ 거리)를 최대한 크게 만들어 모델의 강건성을 높이는 작업이다.

### 평가 지표 및 방정식

두 이미지 $s_1$과 $s_2$ 사이의 거리 측정 방식으로는 $L_2$ 거리를 사용한다.
$$d(s_1, s_2) = \|s_1 - s_2\|_2$$

**모델 점수(Model Score)**:
상위 5개 공격 집합 $A_5$에 대해, 각 샘플 $s$에서 발생하는 가장 작은 섭동의 크기를 먼저 구한다.
$$d_{min}^m(s, A_5) = \min_{a \in A_5} d(s, \hat{s}_a(s, m))$$
모델의 최종 점수는 모든 샘플 $s \in S$에 대한 위 거리 값들의 중앙값(median)으로 계산한다.
$$\text{ModelScore}_m = \text{median}(\{d_{min}^m(s, A_5) \mid s \in S\})$$

**공격 점수(Attack Score)**:
상위 5개 모델 집합 $M_5$에 대해, 공격 $a$가 생성한 섭동의 거리 $d_a(s, m) = d(s, \hat{s}_a(s, m))$를 계산하고, 이에 대한 중앙값을 구한다.
$$\text{AttackScore}_a = \text{median}(\{d_a(s, m) \mid s \in S, m \in M_5\})$$
모델은 점수가 높을수록, 공격은 점수가 낮을수록 우수한 것으로 평가한다.

### 학습 및 추론 절차 (제한 사항)

실제 구현 및 운영을 위해 다음과 같은 엄격한 제약 조건을 둔다.

- **쿼리 제한**: 공격자는 샘플당 최대 1,000번까지만 모델에 쿼리를 보낼 수 있다.
- **시간 제한**: 모델은 K80 GPU 기준 이미지당 40ms 이내에 처리해야 하며, 공격자는 이미지 10개 배치당 900s 이내에 처리해야 한다.
- **상태 비저장(Stateless)**: 모델은 과거 입력에 의존하지 않고 독립적으로 결정해야 하며, 결정론적(deterministic)이어야 한다.

## 📊 Results

본 문서는 챌린지의 제안서 및 운영 가이드라인이므로, 실험을 통한 정량적 결과 값보다는 평가를 위한 baseline 설정 내용을 다룬다.

- **모델 Baseline**: ResNet-50을 기반으로 (1) Vanilla 모델, (2) Adversarially trained 모델, (3) Intrinsic frozen noise가 추가된 모델의 세 가지 버전을 제공한다.
- **공격 Baseline**:
  - **Untargeted**: Gaussian noise, Salt and pepper noise, Boundary Attack, Single-step transfer attack, Iterative transfer attack.
  - **Targeted**: Interpolation-based attack, Pointwise attack, Boundary attack, Iterative transfer attack.

최종 평가는 공개되지 않은 500장의 secret test images를 사용하여 수행하며, 모든 참가자는 코드의 오픈소스 공개를 전제로 최종 순위에 오를 수 있다.

## 🧠 Insights & Discussion

본 논문은 머신러닝 모델의 강건성 평가가 암호학(cryptography)의 평가 방식과 유사해야 한다고 주장한다. 즉, 단순히 알려진 공격에 방어하는 것이 아니라, 해당 모델을 무너뜨리기 위해 설계된 전용 공격(specifically designed attacks)에 얼마나 잘 견디는지가 진정한 강건성의 척도라는 것이다.

특히 decision-based attacks의 중요성을 강조하는데, 이는 모델의 내부 파라미터에 접근할 수 없는 실제 배포 환경에서 가장 위협적인 공격 형태이기 때문이다. 이러한 설계를 통해 단순한 gradient masking과 같은 '눈속임' 방어 기법을 무력화하고, DNN이 상관관계(correlational features)가 아닌 인과적 특징(causal features)을 학습하도록 유도함으로써 인간의 시각 시스템에 더 가까운 모델을 만들 수 있을 것으로 기대한다.

## 📌 TL;DR

본 연구는 모델과 공격자가 서로를 상대하며 함께 진화하는 '공진화' 기반의 adversarial vision 챌린지를 제안한다. decision-based attack과 $L_2$ 거리 기반의 중앙값 점수를 사용하여 가짜 보안을 배제하고 실질적인 모델 강건성을 측정하며, 이를 통해 안전 필수 애플리케이션에 적용 가능한 수준의 강건한 비전 모델 개발과 정교한 공격 기법의 발전을 목표로 한다.
