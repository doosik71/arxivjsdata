# TAME: Task Agnostic Continual Learning using Multiple Experts

Haoran Zhu, Maryam Majzoubi, Arihant Jain, Anna Choromanska (2022)

## 🧩 Problem to Solve

본 논문은 **Continual Learning (지속 학습)** 환경에서 발생하는 **Catastrophic Forgetting (파괴적 망각)** 문제를 해결하고자 한다. 특히, 기존의 연구들이 학습 과정에서 Task의 정체성(Task Identity)이 제공된다는 가정하에 진행된 것과 달리, 본 논문은 학습과 추론 단계 모두에서 Task ID가 제공되지 않는 **Task-Agnostic (태스크 불가지론적)** 설정에 집중한다.

실제 현실 세계의 데이터 스트림은 비정상성(Non-stationary) 분포를 가지며, 에이전트는 어떤 Task가 들어오는지 알지 못한 채 데이터로부터 이를 스스로 추론하고 적응해야 한다. 따라서 본 연구의 목표는 Task ID 없이도 데이터 분포의 변화를 자동으로 감지하고, 새로운 지식을 습득하면서도 이전 지식을 보존할 수 있는 효율적인 학습 시스템을 구축하는 것이다.

## ✨ Key Contributions

TAME의 핵심 아이디어는 **Multiple Experts (다중 전문가)** 구조를 활용하여 각 Task를 개별 전문가 네트워크가 담당하게 하는 것이다. 중심적인 설계 직관은 다음과 같다.

1.  **손실 함수 기반의 Task 전환 감지**: 새로운 Task가 시작될 때 현재 활성화된 전문가 네트워크의 Loss 값이 통계적으로 유의미하게 급증한다는 점에 착안하여, 이를 통해 Task 전환 시점을 자동으로 감지한다.
2.  **Selector Network 도입**: 추론 단계에서 입력 샘플을 적절한 전문가 네트워크로 전달하기 위해, 학습 과정에서 무작위로 수집된 소량의 데이터를 사용하여 Task ID를 예측하는 별도의 선택기 네트워크를 학습시킨다.
3.  **모델 효율성 최적화**: 전문가 네트워크와 Selector 네트워크의 개수가 늘어남에 따라 발생하는 파라미터 증가 문제를 해결하기 위해 **Pruning (가지치기)** 기술을 적용하여 모델 크기를 제어한다.

## 📎 Related Works

본 논문은 지속 학습 방법론을 크게 세 가지(보완 학습 시스템 및 메모리 리플레이, 정규화 기반 방법, 동적 아키텍처 방법)로 분류하며, 특히 Task-Agnostic 설정의 기존 연구들과 차별점을 둔다.

*   **BGD (Bayesian Gradient Descent)**: 온라인 변분 베이즈를 사용하지만, Task ID를 클래스 라벨에서 추론하는 'Label Trick'에 의존하여 완전한 Task-Agnostic 설정이라 보기 어렵다.
*   **iTAML**: 메타 러닝을 통해 일반화된 파라미터를 유지하지만, 학습 단계의 내부 루프(Inner loop)에서 Task 라벨이 필요하다.
*   **CN-DPM**: Dirichlet Process Mixture Model을 사용하여 전문가를 할당하며, 짧은 기간의 메모리(STM)를 사용하여 새로운 전문가 생성 여부를 결정한다.
*   **HCL**: Normalizing Flow 모델을 사용하여 Task 분포를 모델링하고 이상 탐지(Anomaly Detection) 기법으로 Task를 식별한다.

TAME은 복잡한 확률 모델이나 메타 러닝 대신, **Loss 함수의 통계적 변동성**이라는 단순하고 직관적인 신호를 사용하여 Task 전환을 감지한다는 점에서 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인
TAME은 하나의 전문가 네트워크로 시작하여, 데이터 분포의 변화가 감지될 때마다 기존 전문가를 선택하거나 새로운 전문가를 추가하는 방식으로 동작한다. 학습 시에는 오직 하나의 'Active Expert'만이 학습되며, 추론 시에는 Selector Network가 입력을 적절한 전문가에게 라우팅한다.

### 2. Task 전환 감지 및 전문가 선택
데이터의 단기적인 변동(Noise)을 제거하고 장기적인 패턴을 파악하기 위해 **Exponentially Weighted Moving Average (EWMA)**를 사용하여 손실 함수 값을 평활화(Smoothing)한다.

$$L_s = \alpha L_c + (1 - \alpha) L_{s, prev}$$

여기서 $L_s$는 평활화된 손실 값, $L_c$는 현재 배치에서 계산된 손실 값, $\alpha$는 평활화 계수이다.

Task 전환 여부를 결정하는 임계값(Threshold)은 현재 전문가가 관찰한 손실 값들의 이동 평균($\mu$)과 표준 편차($\sigma$)를 이용하여 다음과 같이 정의한다.

$$\text{Threshold} = \mu + 3\sigma$$

**작동 절차:**
1.  현재 Active Expert의 $L_s$가 임계값을 초과하면 Task 전환이 발생한 것으로 간주한다.
2.  기존에 생성된 모든 전문가 네트워크를 순회하며, 해당 데이터에 대해 임계값 이하의 손실을 보이는 전문가가 있는지 확인한다.
3.  적절한 기존 전문가가 있다면 해당 전문가를 Active Expert로 전환하고, 없다면 새로운 전문가 네트워크를 생성한다.

### 3. Selector Network 및 Pruning
*   **Selector Network**: 학습 과정에서 모든 Task로부터 균등하게 무작위 추출된 소량의 샘플들을 Priority Queue 형태의 버퍼($C_s$)에 저장한다. 이 데이터들을 사용하여 입력 샘플 $\to$ 전문가 ID(Task ID)를 예측하는 분류기를 학습시킨다.
*   **Pruning 및 Retraining**: 모델 크기를 줄이기 위해 **L1 Unstructured Pruning**을 적용한다. 가지치기 후 성능 하락을 막기 위해, 각 전문가별로 저장된 소량의 데이터 버퍼($C_p$)를 사용하여 재학습(Retraining)을 수행한다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: Permuted MNIST (20 tasks), Split MNIST (5 tasks), Split CIFAR-100 (10, 20 tasks), Split CIFAR-10 (5 tasks), SVHN-MNIST, MNIST-SVHN 등.
*   **비교 대상**: Task-Agnostic 방법(BGD, iTAML, HCL, CN-DPM) 및 Task ID를 아는 방법(DEN, EWC, SI, A-GEM, RWALK).
*   **지표**: 모든 Task에 대한 평균 정확도(Average Accuracy, ACC).

### 2. 주요 결과
*   **성능 우위**: TAME은 거의 모든 데이터셋에서 기존의 Task-Agnostic 방법론들보다 높은 정확도를 기록하였다. 놀라운 점은 Task ID를 학습/추론 시에 제공받는 방법론들(EWC, SI 등)보다도 더 높은 성능을 보였다는 것이다.
*   **모델 효율성**: Pruning을 통해 매우 효율적인 파라미터 수를 유지하면서도 높은 성능을 냈다. 예를 들어, Split CIFAR-100(20)에서 TAME은 타 모델 대비 현저히 적은 파라미터로 최고 성능을 달성하였다.
*   **Task 재방문 처리**: 특정 Task가 순차적으로 등장하다가 나중에 다시 등장하는 경우($T=\{t_1, t_2, t_3, t_2, t_4\}$), TAME은 새로운 전문가를 만드는 대신 기존에 학습된 전문가($t_2$)를 정확히 다시 선택하여 활용함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰
*   **단순함의 미학**: 복잡한 확률적 모델링 없이 Loss 값의 통계적 변동성이라는 단순한 지표만으로 Task 전환을 효과적으로 감지할 수 있음을 증명하였다.
*   **평활화(Smoothing)의 중요성**: EWMA를 통한 Loss 평활화가 필수적이다. 평활화를 적용하지 않을 경우 단기적인 노이즈로 인해 Task 전환으로 오인하는 False Positive가 빈번하게 발생하여 불필요한 전문가 네트워크가 과도하게 생성되는 문제가 발생한다.
*   **실용적 확장성**: Pruning과 소량의 버퍼를 이용한 재학습 구조를 통해 전문가 네트워크 증가에 따른 메모리 문제를 실질적으로 해결하였다.

### 한계 및 논의사항
*   **임계값 설정**: $\mu + 3\sigma$라는 정적인 통계 기준을 사용하는데, 데이터셋의 특성에 따라 최적의 $\sigma$ 배수가 다를 수 있다. 이에 대한 적응형(Adaptive) 임계값 설정 방안이 논의될 수 있다.
*   **Selector Network의 의존성**: 추론 시 성능이 Selector Network의 Task 식별 정확도에 전적으로 의존한다. Task 간의 유사도가 매우 높을 경우 Selector Network의 성능이 떨어지며, 이는 전체 시스템의 성능 저하로 직결될 가능성이 크다.

## 📌 TL;DR

TAME은 Task ID가 전혀 제공되지 않는 **Task-Agnostic Continual Learning** 환경에서, **Loss 함수의 통계적 급증**을 통해 Task 전환을 감지하고 **다중 전문가 네트워크(Multiple Experts)**를 동적으로 할당하는 알고리즘이다. 추론 시에는 **Selector Network**를 통해 적절한 전문가에게 데이터를 라우팅하며, **L1 Pruning**으로 모델 크기를 최적화한다. 실험 결과, Task ID를 알고 학습하는 기존 방법론들보다도 뛰어난 성능과 효율성을 보였으며, 특히 과거에 학습한 Task가 다시 나타났을 때 이를 재사용하는 능력이 탁월하다. 이는 향후 실제 환경의 비정상성 데이터 스트림 처리 연구에 중요한 기초가 될 것으로 보인다.