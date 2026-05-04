# A Few-Shot Label Unlearning in Vertical Federated Learning

Hanlin Gu, Hong Xi Tae, Chee Seng Chan, and Lixin Fan (2024)

## 🧩 Problem to Solve

본 논문은 수직 연합 학습(Vertical Federated Learning, VFL) 환경에서 특정 클래스의 레이블 정보를 제거하는 **레이블 언러닝(Label Unlearning)** 문제를 다룬다. VFL은 서로 다른 특징(feature)을 가진 여러 참여자가 동일한 샘플 ID를 공유하며 협업 학습하는 구조로, 레이블을 보유한 Active Party와 특징만을 보유한 Passive Party로 나뉜다.

최근 GDPR이나 CCPA와 같은 데이터 보호 규정으로 인해 사용자의 '잊힐 권리'가 중요해짐에 따라, 학습된 모델에서 특정 데이터를 제거하는 언러닝 기술이 필수적으로 요구되고 있다. 하지만 기존의 연합 언러닝 연구는 주로 수평 연합 학습(Horizontal Federated Learning, HFL)에 집중되어 있었으며, VFL에서의 언러닝 연구는 주로 참여자의 탈퇴로 인한 특징 제거에 치중되어 있었다.

특히, VFL에서 레이블 언러닝을 수행할 때 가장 큰 문제는 **레이블 유출(Label Leakage)** 위험이다. 전통적인 언러닝 방식(예: 재학습, Boundary Unlearning)을 적용하려면 Active Party가 Passive Party에게 언러닝 대상 샘플 정보를 알리거나 관련 그래디언트(gradient)를 전송해야 하는데, 이 과정에서 Passive Party가 전송된 그래디언트를 클러스터링함으로써 민감한 레이블 정보를 역으로 추론할 수 있는 보안 취약점이 존재한다. 따라서 본 논문은 레이블 프라이버시를 보호하면서도 효율적으로 레이블 정보를 삭제하는 방법을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **소량의 프라이빗 데이터만을 사용하는 Few-shot 언러닝**과 **Manifold Mixup**을 결합하여 레이블 유출 위험을 최소화하고 언러닝 효율을 극대화하는 것이다.

1. **VFL 최초의 레이블 언러닝 제안**: VFL 환경에서 특정 클래스의 레이블을 제거하는 접근 방식을 최초로 제안하였다.
2. **레이블 유출 메커니즘 규명**: 기존 언러닝 방법론을 VFL에 그대로 적용했을 때, 전송되는 그래디언트를 통해 Passive Party가 레이블을 추론할 수 있음을 체계적으로 분석하였다.
3. **Few-shot 기반의 효율적 삭제**: 극소량의 레이블 데이터($D_p$)만을 사용하여 레이블 유출 가능성을 낮추고, Manifold Mixup을 통해 데이터 부족 문제를 해결하며, Gradient Ascent를 통해 Active 및 Passive 모델 모두에서 정보를 빠르게 삭제한다.

## 📎 Related Works

### 기존 언러닝 연구

머신 언러닝은 크게 정확한 언러닝(Exact Unlearning, 예: SISA)과 근사 언러닝(Approximate Unlearning, 예: Fine-tuning, Gradient Ascent, Knowledge Distillation)으로 나뉜다. 연합 학습 환경에서도 이러한 기법들이 적용되었으나, 대다수의 연구는 데이터 분포가 동일하고 샘플만 다른 수평 연합 학습(HFL) 설정에서 이루어졌다.

### VFL 언러닝의 한계 및 차별점

VFL 언러닝에 관한 기존 연구들은 주로 Passive Party가 네트워크에서 완전히 이탈할 때 해당 참여자의 모든 특징을 제거하는 것에 집중하였다. 그러나 본 논문이 다루는 시나리오는 **모든 참여자가 네트워크에 유지된 상태에서 특정 클래스의 레이블 정보만을 선택적으로 삭제**하는 것이다. 이는 기존 VFU(Vertical Federated Unlearning) 연구들이 다루지 않은 영역이며, 중앙 집중식이나 HFL 방식의 클래스 언러닝 기법은 VFL의 분산된 특징 구조와 서로 다른 계산 능력으로 인해 직접 적용하기 어렵다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

본 논문이 제안하는 방법론은 소량의 레이블 데이터 $D_p$를 활용하여 두 단계의 프로세스로 진행된다.

### 1. Vertical Manifold Mixup

레이블 유출을 막기 위해 아주 적은 양의 데이터만을 사용하면 언러닝 성능이 떨어지는 문제가 발생한다. 이를 해결하기 위해 특징 공간의 임베딩을 보간(interpolation)하는 Manifold Mixup을 VFL 구조에 도입한다.

각 Passive Party $k$는 자신의 모델 $G_{\theta_k}$를 통해 포워드 임베딩 $H_k$를 생성한다. 이후 다음과 같은 Mixup 함수를 통해 증강된 임베딩 $H'_k$를 생성하여 Active Party에게 전송한다.

$$\text{Mix}_\lambda(a, b) = \lambda \cdot a + (1 - \lambda) \cdot b$$

여기서 $\lambda$는 $0$과 $1$ 사이의 혼합 계수이다. 이 과정을 통해 데이터의 상태 분포를 평탄하게 만들어, 적은 데이터로도 효과적인 언러닝이 가능하도록 한다.

### 2. Gradient Ascent를 이용한 레이블 삭제

증강된 임베딩 $\{H'_1, \dots, H'_K\}$가 준비되면, Active Party는 이를 결합하여 $H' = [H'_1, \dots, H'_K]$를 구성하고, 손실 함수 $\ell(F_\omega(H'), y')$를 최대화하는 Gradient Ascent(GA)를 수행한다.

**Active Model ($F_\omega$)의 업데이트:**
Active Party는 자신의 모델 파라미터 $\omega$를 다음과 같이 업데이트하여 타겟 클래스의 정보를 삭제한다.
$$\omega = \omega + \eta \nabla_\omega \ell(F_\omega(H'), y')$$

**Passive Model ($G_{\theta_k}$)의 업데이트:**
이후 Active Party는 임베딩에 대한 그래디언트 $g'_k = \frac{\partial \ell}{\partial H'_k}$를 계산하여 각 Passive Party에게 전송한다. Passive Party는 이를 전달받아 자신의 모델 파라미터 $\theta_k$를 업데이트한다.
$$\theta_k = \theta_k + \eta \nabla_{H'_k} \ell(F_\omega(H'), y') \cdot \nabla_{\theta_k} H'_k$$

여기서 $\eta$는 매우 작은 학습률을 사용하여 모델의 전체적인 유틸리티가 급격히 파괴되거나 그래디언트 소실이 발생하는 것을 방지한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, CIFAR-10, CIFAR-100, ModelNet.
- **모델**: ResNet18, VGG16.
- **시나리오**: 단일 클래스 삭제, 2개 클래스 삭제, 다중 클래스(4개) 삭제.
- **평가 지표**:
  - **Utility**: 남은 데이터($D_r$)에 대한 정확도 유지율.
  - **Effectiveness**: 삭제 대상 데이터($D_u$)의 정확도 감소 및 MIA(Membership Inference Attack)의 공격 성공률(ASR) 측정.
  - **Efficiency**: 언러닝 수행 시간(Runtime).

### 주요 결과

1. **유틸리티 보존**: 제안 방법은 Fine-tuning과 유사하게 $D_r$에 대한 정확도를 매우 잘 유지하였다. 반면 Fisher Forgetting이나 Amnesiac Unlearning은 $D_r$의 정확도를 크게 떨어뜨리는 경향을 보였다.
2. **언러닝 효과**: $D_u$에 대한 정확도를 거의 $0\%$까지 낮추었으며, MIA 공격에 대한 ASR 수치 또한 타 방법론 대비 낮고 안정적으로 유지되어 정보 삭제가 효과적으로 이루어졌음을 입증하였다.
3. **시간 효율성**: Retrain(재학습)은 당연히 가장 오래 걸렸으며, 제안 방법은 전체 데이터셋을 사용하는 다른 근사 언러닝 방법들보다 훨씬 빠른 '수 초 내'에 프로세스를 완료하였다.
4. **Few-shot 성능**: Ablation Study를 통해 단순 Gradient Ascent(GA)는 40개의 샘플만으로는 효과가 낮았으나(정확도 $40.48\%$), Manifold Mixup을 결합한 제안 방법은 동일한 40개 샘플만으로 정확도를 $0\%$까지 낮추는 데 성공하였다.
5. **강건성**: Differential Privacy(DP)나 Gradient Compression이 적용된 VFL 환경에서도 여전히 효과적으로 언러닝이 수행됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 VFL의 특수한 구조에서 발생할 수 있는 **레이블 유출**이라는 보안 문제를 정확히 짚어냈으며, 이를 해결하기 위해 데이터의 양을 극도로 제한하는 전략을 취했다. 보통 데이터가 적으면 성능이 떨어지지만, 이를 **Manifold Mixup**이라는 임베딩 증강 기법으로 보완하여 '프라이버시 보호'와 '삭제 성능'이라는 두 마리 토끼를 잡은 점이 인상적이다.

특히, 단순히 Active Party만 업데이트하는 것이 아니라, 계산된 그래디언트를 다시 Passive Party에게 전달하여 **전체 파이프라인의 모든 모델에서 정보를 삭제**하도록 설계한 점이 논리적으로 타당하다.

다만, Gradient Ascent의 특성상 학습률 $\eta$와 에포크 수에 매우 민감하게 반응하며, 잘못 설정할 경우 모델 전체의 성능(Utility)이 붕괴될 위험이 있다. 논문에서는 작은 $\eta$를 사용해 이를 해결했다고 언급하지만, 다양한 모델과 데이터셋에 대해 최적의 하이퍼파라미터를 찾는 자동화된 방법론에 대한 논의는 부족한 편이다.

## 📌 TL;DR

이 논문은 VFL 환경에서 레이블 유출 위험 없이 특정 클래스를 삭제하는 **Few-shot Label Unlearning** 방법을 제안한다. 소량의 데이터와 **Manifold Mixup**을 통해 데이터 부족 문제를 해결하고, **Gradient Ascent**를 통해 Active/Passive 모델 모두에서 정보를 빠르게 삭제한다. 실험 결과, 기존 방법 대비 프라이버시 보호 능력이 뛰어나며, 연산 효율성이 극도로 높아 실제 VFL 시스템의 '잊힐 권리' 구현에 중요한 기여를 할 것으로 보인다.
