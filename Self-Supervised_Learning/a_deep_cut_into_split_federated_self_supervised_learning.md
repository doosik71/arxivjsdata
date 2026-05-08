# A deep cut into Split Federated Self-supervised Learning

Marcin Przewięźlikowski, Marcin Osial, Bartosz Zieliński, and Marek Śmieja (2024/2025)

## 🧩 Problem to Solve

본 논문은 분산 환경에서 데이터 프라이버시를 보호하면서 효율적으로 모델을 학습시키기 위한 **Split Federated Self-supervised Learning (SFL-SSL)**의 최적화 문제를 다룬다. 특히, 기존의 최신 방법론인 MocoSFL이 네트워크의 매우 초기 레이어(shallow layers)에서 모델을 분할하도록 최적화되어 있다는 점에 주목한다.

모델을 초기 레이어에서 분할할 경우 다음과 같은 두 가지 심각한 문제가 발생한다:

1. **프라이버시 위험**: 낮은 레이어의 활성화 값(activations)은 입력 데이터의 형태를 많이 유지하고 있어, Model Inversion Attack (MIA)과 같은 공격에 취약하며 데이터 유출 가능성이 높다.
2. **통신 오버헤드**: 네트워크 구조상 낮은 레이어의 표현(representation)은 공간적 차원이 크기 때문에, 서버로 전송해야 하는 데이터 양이 많아져 통신 비용이 증가한다.

따라서 본 연구의 목표는 모델 분할 지점을 더 깊은 레이어(deeper layers)로 옮겨 프라이버시를 강화하고 통신 효율성을 높이면서도, 학습 성능(정확도)을 유지할 수 있는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **MonAcoSFL (Momentum-Aligned contrastive Split Federated Learning)**이라는 새로운 학습 프레임워크를 제안한 것이다.

핵심 아이디어는 **온라인 모델(online model)뿐만 아니라 모멘텀 모델(momentum model)까지 함께 동기화**하는 것이다. 기존 MocoSFL은 온라인 모델만 동기화했기 때문에, 분할 지점이 깊어질수록 온라인 모델과 모멘텀 모델 간의 정렬(alignment)이 깨져 성능이 급격히 저하되는 현상이 발생한다. MonAcoSFL은 이를 해결하여 깊은 분할 지점에서도 높은 정확도를 유지하며 실용적인 통신 효율성과 프라이버시를 동시에 달성한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급한다:

- **Federated Learning (FL)**: 데이터를 중앙으로 모으지 않고 로컬에서 학습 후 파라미터를 동기화하는 방식이다. $\text{FedAVG}$가 대표적이지만, 모든 모델을 클라이언트에 저장해야 하므로 자원 제한이 있는 기기에서는 부담이 크다.
- **Split Federated Learning (SFL)**: 모델을 클라이언트 부분($f_c$)과 서버 부분($f_s$)으로 나누어 계산 부담을 분산하는 방식이다.
- **Self-supervised Learning (SSL)**: 레이블 없는 데이터를 활용하여 표현을 학습하는 방식으로, 특히 MoCo와 같은 Joint-embedding 아키텍처는 모멘텀 인코더를 사용하여 대조 학습(contrastive learning)을 수행한다.
- **MocoSFL**: MoCo와 SFL을 결합하여 100명 이상의 대규모 클라이언트 환경에서도 확장 가능한 SSL 프레임워크를 제안했다. 그러나 본 논문은 MocoSFL이 깊은 분할 지점에서 성능이 붕괴된다는 한계를 지적하며, 이를 개선하는 방향으로 접근한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구는 모델 $f$를 클라이언트 부분 $f_c$와 서버 부분 $f_s$로 분리한 $f = f_s \circ f_c$ 구조를 가진다. 각 클라이언트는 자신의 데이터로 로컬 모델을 최적화하고, 주기적으로 서버를 통해 다른 클라이언트들과 파라미터를 동기화한다.

### 학습 목표 및 손실 함수

학습에는 **InfoNCE (Information Noise-Contrastive Estimation)** 손실 함수를 사용하여, 서로 다른 뷰(view)로 증강된 동일 이미지의 임베딩은 가깝게, 서로 다른 이미지의 임베딩은 멀게 학습한다.

$$L_{\text{InfoNCE}}(z', z'', M) = -\log \frac{\exp(z' \cdot z'' / \tau)}{\exp(z' \cdot z'' / \tau) + \sum_{z_M \in M} \exp(z' \cdot z_M / \tau)}$$

여기서 $z' = f_{\phi}(x')$는 온라인 모델의 출력이고, $z'' = f_{\text{EMA}(\phi)}(x'')$는 모멘텀 모델의 출력이다. $M$은 서버가 유지하는 부정적 예시(negative examples)의 큐(queue)이다.

### MonAcoSFL의 핵심 절차: 모멘텀 정렬 동기화

기존 MocoSFL은 동기화 단계에서 온라인 파라미터 $\phi_c$만 평균 내어 업데이트한다. 하지만 MonAcoSFL은 모멘텀 파라미터 $\text{EMA}(\phi_c)$ 또한 동일한 방식으로 동기화한다.

$$\hat{\text{EMA}}(\phi_c) = \frac{\sum_{i=0}^N \text{EMA}(\phi_c^i)}{N}$$

이렇게 함으로써 $\text{EMA}(\hat{\phi}_c) = \hat{\text{EMA}}(\phi_c)$ 관계가 유지되어, 온라인 모델과 모멘텀 모델 사이의 괴리를 방지하고 학습의 안정성을 확보한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-10, CIFAR-100 (non-IID 설정)
- **백본 모델**: ResNet-18, MobileNetV2
- **클라이언트 수**: 5, 20, 200명 (cross-silo 및 cross-client 설정)
- **평가 지표**: 선형 평가(Linear Evaluation) 정확도, MIA 공격을 통한 MSE 기반 프라이버시 측정

### 주요 결과

1. **정확도 성능**:
   - 분할 지점이 깊어질수록(더 많은 레이어가 클라이언트에 위치할수록) MocoSFL의 성능은 급격히 하락한다.
   - 반면, MonAcoSFL은 깊은 분할 지점에서도 성능 저하가 매우 적다. 특히 가장 통신 효율적인 지점(ResNet-18 기준 11~13번 레이어)에서 MocoSFL 대비 30%p 이상의 정확도 향상을 보였다.
2. **통신 오버헤드**:
   - 모델을 더 깊은 곳에서 자를수록 서버로 전송하는 활성화 값의 크기가 줄어들어 통신 비용이 감소한다. ResNet-18의 경우 11번 레이어, MobileNetV2의 경우 7번 레이어에서 최적의 통신 효율이 나타난다.
3. **프라이버시 보호**:
   - MIA 공격 결과, 분할 지점이 깊어질수록 재구성된 이미지의 MSE가 높아져(즉, 원본 이미지 복원이 어려워져) 프라이버시 보호 능력이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 분산 SSL 환경에서 **'모델 분할 지점'**이 단순한 계산 분배의 문제가 아니라, **프라이버시, 통신 효율, 그리고 모델의 수렴 안정성**이라는 세 가지 요소가 얽힌 핵심 변수임을 입증하였다.

MocoSFL의 실패 원인이 온라인 모델과 모멘텀 모델의 '정렬 불일치(misalignment)'에 있다는 점을 실험적으로 증명한 것이 매우 인상적이다. 파라미터 동기화 시 온라인 모델만 업데이트하면 모멘텀 모델은 과거의 파라미터 상태에 머물게 되어, 대조 학습의 기본 전제인 '두 모델의 유사한 표현력'이 깨지게 된다.

다만, 본 논문은 MoCo 기반의 프레임워크에 집중하고 있다. BYOL이나 SimCLR와 같이 모멘텀 모델을 사용하는 다른 SSL 기법들에도 이 '모멘텀 동기화' 아이디어가 동일하게 적용될 수 있을지는 추가적인 연구가 필요해 보인다.

## 📌 TL;DR

이 논문은 Split Federated SSL에서 모델 분할 지점을 깊게 설정하면 프라이버시와 통신 효율은 좋아지지만 성능이 급격히 떨어진다는 문제를 발견하고, 이를 해결하기 위해 온라인 모델과 모멘텀 모델을 동시에 동기화하는 **MonAcoSFL**을 제안하였다. 이를 통해 통신 오버헤드를 최소화하고 데이터 프라이버시를 강화하면서도 최신 수준의 정확도를 달성하였으며, 이는 실제 분산 환경에서의 SSL 적용 가능성을 크게 높인 연구이다.
