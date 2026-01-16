# Three scenarios for continual learning

Gido M. van de Ven, Andreas S. Tolias

## 🧩 Problem to Solve

- 표준 인공 신경망은 순차적으로 여러 작업을 학습할 때 발생하는 "치명적인 망각(catastrophic forgetting)" 문제로 어려움을 겪습니다.
- 최근 몇 년간 수많은 지속 학습(continual learning) 방법들이 제안되었지만, 평가 프로토콜의 차이로 인해 이들의 성능을 직접적으로 비교하기 어렵습니다.
- 이러한 비교의 어려움 때문에 지속 학습 분야에 대한 구조화된 평가 프레임워크가 필요합니다.

## ✨ Key Contributions

- 테스트 시점에 작업 정체성(task identity) 제공 여부 및 추론 필요 여부에 따라 지속 학습을 위한 세 가지 고유 시나리오(Task-Incremental Learning (Task-IL), Domain-Incremental Learning (Domain-IL), Class-Incremental Learning (Class-IL))를 정의했습니다.
- Split MNIST 및 Permuted MNIST 프로토콜을 사용하여 각 시나리오에서 최근 제안된 지속 학습 방법들의 광범위한 비교를 수행했습니다.
- 세 가지 시나리오 간의 난이도와 각 방법의 효율성에서 상당한 차이가 있음을 입증했습니다.
- 특히 작업 정체성을 추론해야 하는 Class-IL 시나리오에서는 EWC(Elastic Weight Consolidation)와 같은 정규화 기반(regularization-based) 접근 방식이 실패하며, 이전 경험의 재현(replaying representations of previous experiences)이 필수적임을 발견했습니다.
- 비교된 모든 방법에 대해 잘 문서화되고 쉽게 적용 가능한 코드를 공개하여 재현성을 높였습니다.

## 📎 Related Works

이 연구는 지속 학습 분야의 핵심 과제인 "치명적인 망각"을 다루며, 기존 연구들이 해결하지 못한 방법론 비교의 어려움을 해결하고자 합니다. 주요 참고 문헌 및 관련 개념은 다음과 같습니다:

- **치명적인 망각(Catastrophic Forgetting)**: 신경망이 새로운 작업을 학습할 때 이전 작업에 대한 정보를 잊는 현상입니다 (Goodfellow et al., 2013).
- **기존 지속 학습 방법론**:
  - **정규화 기반(Regularization-based)**: Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017), Synaptic Intelligence (SI) (Zenke et al., 2017), Online EWC (Schwarz et al., 2018).
  - **아키텍처 기반(Architecture-based)**: Context-dependent Gating (XdG) (Masse et al., 2018).
  - **데이터 재현 기반(Replay-based)**: Learning without Forgetting (LwF) (Li & Hoiem, 2017), Deep Generative Replay (DGR) (Shin et al., 2017), iCaRL (Rebuffi et al., 2017).
- **평가 프로토콜**: Multi-headed vs. Single-headed (Farquhar & Gal, 2018; Chaudhry et al., 2018), Split MNIST (Zenke et al., 2017) 및 Permuted MNIST (Goodfellow et al., 2013).
- **기반 기술**: Variational Autoencoder (VAE) (Kingma & Welling, 2013), Adam optimizer (Kingma & Ba, 2014).

## 🛠️ Methodology

본 연구는 세 가지 지속 학습 시나리오와 두 가지 태스크 프로토콜에 걸쳐 다양한 최신 지속 학습 방법론을 평가했습니다.

### 1. 지속 학습 시나리오

모델 평가 조건에 따라 난이도가 증가하는 세 가지 시나리오를 정의합니다.

- **Task-Incremental Learning (Task-IL)**:
  - **테스트 시점**: 모델에 현재 처리해야 할 **작업 정체성(task identity)이 제공**됩니다.
  - **특징**: 가장 쉬운 시나리오이며, 각 작업에 고유한 출력 유닛을 사용하는 "다중 헤드(multi-headed)" 출력 레이어 아키텍처를 사용할 수 있습니다.
- **Domain-Incremental Learning (Domain-IL)**:
  - **테스트 시점**: **작업 정체성이 제공되지 않지만**, 모델이 어떤 작업인지 **추론할 필요는 없습니다.**
  - **특징**: 입력 분포는 변하지만 작업의 구조는 동일한 경우에 해당합니다. 일반적으로 모든 작업에 동일한 출력 유닛을 사용하는 "단일 헤드(single-headed)" 출력 레이어를 사용합니다.
- **Class-Incremental Learning (Class-IL)**:
  - **테스트 시점**: **작업 정체성이 제공되지 않으며**, 모델이 어떤 작업인지 **추론해야 합니다.**
  - **특징**: 가장 어려운 시나리오로, 현실 세계의 새로운 클래스(객체)를 점진적으로 학습하는 문제에 해당합니다. 모든 이전에 학습된 클래스에 대한 출력 유닛이 활성화됩니다.

### 2. 태스크 프로토콜

세 가지 시나리오를 시연하고 비교하기 위해 두 가지 MNIST 기반 태스크 프로토콜을 사용했습니다.

- **Split MNIST**: 원본 MNIST 데이터셋을 5개의 작업으로 분할하여 각 작업은 2개의 클래스(예: 0과 1 분류, 2와 3 분류 등)를 분류하는 이진 분류 작업입니다.
- **Permuted MNIST**: 10개의 작업으로 구성되며, 각 작업은 모든 10개의 MNIST 숫자를 분류하지만, 각 작업마다 픽셀에 무작위 순열(permutation)이 적용됩니다.

### 3. 비교 방법론

다음 지속 학습 방법들을 비교했습니다.

- **XdG (Context-dependent Gating)**: 각 히든 레이어 유닛의 $X\%$를 무작위로 게이팅하여 작업별 구성 요소를 정의합니다. Task-IL 시나리오에서만 사용 가능합니다.
- **정규화 기반 방법**: **EWC (Elastic Weight Consolidation)**, **Online EWC**, **SI (Synaptic Intelligence)**: 이전 작업에 중요한 파라미터의 변경을 제한하는 정규화 항을 손실 함수에 추가합니다.
- **재현 기반 방법**:
  - **LwF (Learning without Forgetting)**: 현재 작업의 입력 데이터를 이전 작업 모델의 소프트 타겟(soft targets)으로 재현합니다.
  - **DGR (Deep Generative Replay)**: 별도의 생성 모델(VAE)을 사용하여 이전 작업의 입력 이미지를 생성하고 주 모델에서 생성된 하드 타겟(hard targets)과 함께 재현합니다.
  - **DGR+distill**: DGR과 유사하지만, 생성된 이미지에 소프트 타겟(LwF처럼)을 사용합니다.
- **재현 + 예시 기반 방법**: **iCaRL (Incremental Classifier and Representation Learning)**: 이전 작업의 데이터를 "예시"로 저장하고, 이를 특징 추출기 훈련 및 최근접 평균 분류(nearest-class-mean classification)에 사용합니다. Class-IL 시나리오에서만 사용 가능합니다.
- **기준선(Baselines)**: **None (Fine-tuning)** (하한선), **Offline (Joint training)** (상한선).

### 4. 실험 상세

- **네트워크 아키텍처**: 2개의 은닉층(split MNIST: 400개, permuted MNIST: 1000개 노드)을 가진 MLP(Multi-Layer Perceptron)를 사용했으며, ReLU 비선형성을 적용했습니다.
- **훈련**: ADAM 최적화기(Optimizer)를 사용하여 각 작업을 훈련했습니다 (split MNIST: 2000 iteration, 학습률 0.001; permuted MNIST: 5000 iteration, 학습률 0.0001). 재현 기반 방법의 경우 현재 및 재현 데이터에 대한 손실 가중치를 조정했습니다.
- **생성 모델**: DGR 및 DGR+distill에서는 2개의 은닉층(400개 또는 1000개 유닛)과 100개의 잠재 변수 층을 가진 대칭 변분 오토인코더(VAE)를 사용했습니다.

## 📊 Results

세 가지 지속 학습 시나리오와 두 가지 태스크 프로토콜(Split MNIST, Permuted MNIST)에 대한 광범위한 실험 결과는 다음과 같습니다.

### 1. Split MNIST 태스크 프로토콜 (표 4)

- **Task-IL 시나리오**: 모든 테스트 방법들이 상한선(Offline)에 근접하는 우수한 성능을 보였습니다.
- **Domain-IL 시나리오**: LwF와 정규화 기반 방법(EWC, Online EWC, SI)이 상당한 어려움을 겪었으며, 오직 재현 기반 방법(DGR, DGR+distill) 및 iCaRL만이 90% 이상의 좋은 성능을 달성했습니다.
- **Class-IL 시나리오**: 가장 어려운 시나리오에서 정규화 기반 방법들이 완전히 실패하여 하한선(None)과 유사한 매우 낮은 정확도(약 20%)를 보였습니다. DGR, DGR+distill, iCaRL과 같은 재현 기반 방법만이 90% 이상의 만족스러운 성능을 유지했습니다.
- EWC는 Task-IL 시나리오에서 광범위한 하이퍼파라미터 탐색을 통해 경쟁력 있는 성능을 달성했습니다.

### 2. Permuted MNIST 태스크 프로토콜 (표 5)

- **Task-IL 및 Domain-IL 시나리오**: LwF를 제외한 대부분의 방법들이 이 두 시나리오에서 우수한 성능을 보였으며, 시나리오 간 성능 차이는 크지 않았습니다. 이는 이 프로토콜에서 작업 정체성 정보가 네트워크의 하위 레이어에서 더 유용할 수 있음을 시사합니다. 실제로 XdG와 결합 시 Task-IL 성능이 크게 향상되었습니다 (표 B.1).
- **Class-IL 시나리오**: Split MNIST와 마찬가지로 정규화 기반 방법들은 다시 실패했습니다. DGR, DGR+distill, iCaRL과 같은 재현 기반 방법만이 90% 이상의 높은 성능을 보였습니다.
- LwF는 Permuted MNIST 프로토콜에서는 Split MNIST와 달리 실패했는데, 이는 무작위 순열로 인해 다른 작업의 입력이 서로 상관관계가 없어졌기 때문으로 보입니다.

### 3. 저장된 데이터 재현 (그림 C.1)

- Class-IL 시나리오에서 클래스당 단 하나의 예시를 저장하는 경우에도 모든 정확한 재현(exact replay) 방법이 정규화 기반 방법들을 능가했습니다. 그러나 생성적 재현(generative replay)과 유사한 성능을 얻으려면 훨씬 더 많은 데이터 저장 공간이 필요했습니다.

## 🧠 Insights & Discussion

- **시나리오의 중요성**: 제안된 세 가지 지속 학습 시나리오(Task-IL, Domain-IL, Class-IL)는 지속 학습 평가에서 중요한 차이점을 명확히 하며, 기존 방법론들의 직접적인 비교를 위한 구조화된 프레임워크를 제공합니다.
- **재현(Replay)의 필수성**: 작업 정체성을 추론해야 하는 가장 어려운 Class-IL 시나리오(그리고 부분적으로 Domain-IL 시나리오)의 경우, 현재로서는 **재현 기반 방법만이 허용 가능한 결과를 달성할 수 있는 유일한 접근 방식**임을 입증했습니다. EWC나 SI와 같은 정규화 기반 방법은 이 시나리오에서 근본적으로 실패했습니다. 이는 작업 정체성이 주어지지 않는 보다 도전적인, 실제 환경과 관련된 시나리오에서 **재현이 피할 수 없는 도구일 수 있음**을 시사합니다.
- **연구의 한계**: 본 연구는 MNIST 데이터셋을 사용하여 이미지가 상대적으로 생성하기 쉽다는 한계가 있습니다. 더 복잡한 입력 분포를 가진 태스크 프로토콜에서 생성적 재현(generative replay)이 여전히 효과적일지는 추가 연구가 필요합니다.
- **향후 방향**: 생성 모델의 빠른 발전은 생성적 재현의 잠재력을 더욱 높이고 있습니다. 또한, iCaRL에서처럼 이전 작업의 예시(exemplars)를 저장하고 재현하는 방식은 생성적 재현의 대안 또는 보완책이 될 수 있습니다.
- **하이퍼파라미터 설정의 주의**: 지속 학습 환경에서 전통적인 그리드 탐색(grid search)을 통해 하이퍼파라미터를 설정하는 방식이, 모든 작업의 검증 데이터를 지속적으로 사용할 경우 공정하지 못한 이점을 제공할 수 있음을 지적했습니다.

## 📌 TL;DR

**문제**: 지속 학습 환경에서 신경망의 치명적인 망각 문제와 다양한 평가 프로토콜로 인한 방법론 비교의 어려움.
**해결책**: 테스트 시점의 작업 정체성 제공 여부와 추론 필요성에 따라 Task-IL, Domain-IL, Class-IL의 세 가지 지속 학습 시나리오를 정의하고, 이를 바탕으로 주요 방법론들을 체계적으로 비교.
**주요 발견**: 가장 어려운 Class-IL 시나리오에서 EWC와 같은 **정규화 기반 방법들은 완전히 실패**했으며, DGR, iCaRL과 같은 **재현(replay) 기반 방법들만이 모든 시나리오, 특히 Class-IL에서 좋은 성능을 달성**하여, 견고한 지속 학습을 위해 재현이 필수적임을 강조.
