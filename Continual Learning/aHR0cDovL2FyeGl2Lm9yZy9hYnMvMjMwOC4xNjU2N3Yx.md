# ScrollNet: Dynamic Weight Importance for Continual Learning

Fei Yang, Kai Wang, Joost van de Weijer (2023)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델이 순차적인 작업(Sequential Tasks)을 학습할 때 발생하는 **치명적 망각(Catastrophic Forgetting)** 문제를 해결하고자 한다. 신경망은 새로운 지식을 습득하는 **가소성(Plasticity)**과 기존 지식을 유지하는 **안정성(Stability)** 사이의 상충 관계, 즉 **Stability-Plasticity Dilemma**를 겪게 된다.

기존의 지속 학습(Continual Learning, CL) 방법론들은 특정 가중치가 이전 작업에 얼마나 중요한지를 판단하여, 중요한 가중치의 변화는 억제(안정성)하고 덜 중요한 가중치의 변화는 허용(가소성)하는 방식을 취한다. 그러나 기존 방식들은 데이터에 노출된 이후에야 가중치의 중요도를 명시적(마스크 학습 등) 또는 암시적(정규화 항 도입 등)으로 결정한다는 한계가 있다. 본 논문의 목표는 데이터 노출 전(prior to data exposure)에 가중치의 중요도 순위를 미리 할당함으로써, 더 효율적인 안정성-가소성 트레이드오프를 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **동적 네트워크(Dynamic Network)**의 특성을 활용하여 가중치 중요도를 사전에 정의하고, 이를 작업 순서에 따라 변경하는 **스크롤링(Scrolling)** 메커니즘을 도입하는 것이다.

1. **가중치 중요도 사전 할당**: 네트워크를 여러 개의 서브 네트워크(Sub-networks)로 구성하여, 가장 작은 서브 네트워크에 포함된 파라미터가 가장 높은 중요도를 갖도록 설계한다.
2. **스크롤링 전략**: 새로운 작업이 시작될 때마다 가중치의 중요도 순위를 재배치(Scrolling)하여, 이전 작업에서 가장 중요했던 파라미터가 다음 작업에서는 가장 덜 중요한 파라미터가 되도록 함으로써 안정성과 가소성의 균형을 맞춘다.
3. **방법론의 직교성(Orthogonality)**: ScrollNet은 특정 알고리즘에 종속되지 않으며, 정규화 기반(Regularization-based) 또는 리플레이 기반(Replay-based)의 다양한 기존 CL 방법론과 결합하여 성능을 향상시킬 수 있다.

## 📎 Related Works

### 1. 지속 학습 (Continual Learning)
- **정규화 기반 방법(Regularization-based)**: EWC, MAS, LwF 등이 있으며, 이전 작업에 중요한 파라미터의 변화에 페널티를 부여한다.
- **리플레이 기반 방법(Replay-based)**: iCaRL, BiC, LUCIR 등이 있으며, 이전 작업의 데이터 일부(Exemplars)를 저장하거나 생성 모델을 통해 재현하여 망각을 방지한다.
- **파라미터 격리 기반 방법(Parameter Isolation-based)**: HAT, PackNet 등이 있으며, 작업별로 서로 다른 파라미터 서브셋(마스크)을 할당한다.

### 2. 동적 네트워크 (Dynamic Networks)
- 추론 효율성을 위해 구조나 파라미터를 동적으로 변경하는 네트워크이다. 본 논문은 특히 채널 수를 조절할 수 있는 **Slimmable Neural Networks** 구조를 기반으로 하여 가중치 중요도 랭킹을 구현한다.

### 3. 기존 방식과의 차별점
기존의 파라미터 격리 방식은 학습 과정에서 마스크를 학습하거나 데이터에 기반해 중요도를 결정하지만, ScrollNet은 데이터를 보기 전(Pre-assignment)에 가중치 중요도를 수동으로 명시하고 이를 스크롤링한다는 점에서 근본적인 차이가 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
ScrollNet은 전체 모델 가중치 $\theta$를 $N$개의 겹치지 않는 집합 $\{\theta_{w1}, \theta_{w2}, \dots, \theta_{wN}\}$으로 분할한다. 이 집합들은 중요도의 내림차순으로 정의된다.

### 가중치 중요도 할당 (Weight Importance Assignment)
Slimmable Neural Network 구조를 활용하여 $N$개의 서브 네트워크 $f_n$을 다음과 같이 정의한다.
- $f_1(\cdot; \theta_1 = \theta_{w1})$
- $f_2(\cdot; \theta_2 = \theta_{w1} \cup \theta_{w2})$
- $\dots$
- $f_N(\cdot; \theta_N = \theta)$

학습 시에는 다음과 같은 통합 손실 함수를 사용한다.
$$L_{dynamic} = \sum_{n=1}^{N} L(f_n(X_t; \theta_n), Y_t)$$
여기서 $\theta_{w1}$은 모든 서브 네트워크($f_1$부터 $f_N$까지)의 학습에 참여하므로 가장 높은 중요도를 갖게 되며, $\theta_{wN}$은 오직 전체 네트워크($f_N$) 학습에만 참여하므로 가장 낮은 중요도를 갖게 된다.

### 스크롤링 메커니즘 (Scrolling Operation)
작업 $t$가 종료되고 다음 작업으로 넘어갈 때, 가중치의 중요도 순위를 재배치한다. 스크롤링 단계 크기를 $S$라고 할 때, 작업 $t$에서의 서브 네트워크 구성은 다음과 같다.
$$\theta_1 = \theta_{w(t\%N)*S}, \quad \theta_2 = \theta_{w(t\%N)*S} \cup \theta_{w(t\%N)*S+1}, \quad \dots, \quad \theta_N = \theta$$
이 과정을 통해 이전 작업에서 가장 중요했던(가장 많이 업데이트된) 파라미터는 다음 작업에서 가장 덜 중요한 위치로 이동하여 가소성을 확보하고, 반대로 덜 중요했던 파라미터들이 새로운 작업의 중심이 되어 안정성을 유지한다.

### 타 CL 방법론과의 결합
ScrollNet은 기존 손실 함수에 서브 네트워크들의 손실 합을 추가하는 방식으로 결합된다.
- **ScrollNet + LwF**: 새로운 작업의 손실과 이전 모델과의 지식 증류(Knowledge Distillation) 손실, 그리고 $N-1$개 서브 네트워크의 손실을 모두 더한다.
- **ScrollNet + EWC**: 새로운 작업의 손실, 피셔 정보 행렬(Fisher Information Matrix) 기반의 파라미터 변화 페널티, 그리고 서브 네트워크들의 손실을 결합한다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR100, TinyImageNet
- **작업 분할**: 5, 10, 20 splits
- **아키텍처**: ResNet18을 Slimmable 버전으로 변형하여 사용 (ScrollNet-2, ScrollNet-4 변형 존재)
- **평가 지표**: Average Accuracy (Task-agnostic 및 Task-aware 설정)
- **비교 대상**: FT(Fine-tuning), LwF, EWC, MAS, iCaRL, BiC, LUCIR

### 주요 결과
1. **전반적인 성능 향상**: Task-agnostic 설정에서 ScrollNet을 결합한 모델들이 대부분의 베이스라인보다 높은 성능을 보였다.
2. **정규화 기반 방법과의 시너지**: 특히 EWC, MAS와 결합했을 때 비약적인 향상이 관찰되었다. CIFAR100(5 splits)에서 EWC와 결합 시 최대 $9.40\%$의 정확도 향상을 보였다.
3. **분할 수의 영향**: ScrollNet-4(4개 분할)가 ScrollNet-2(2개 분할)보다 일관되게 좋은 성능을 보였는데, 이는 더 세밀한 가중치 랭킹이 정교한 안정성-가소성 트레이드오프를 가능하게 하기 때문이다.
4. **데이터셋 규모의 영향**: CIFAR100에 비해 TinyImageNet에서의 성능 향상 폭이 적었는데, 이는 ResNet18의 모델 용량이 더 큰 데이터셋을 처리하기에 제한적이었기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점
본 연구는 지속 학습에서 가중치 중요도를 결정하는 시점을 '데이터 노출 후'에서 '데이터 노출 전'으로 앞당겼다는 점에서 매우 참신하다. 특히 복잡한 마스크 학습 없이 단순한 스크롤링 연산과 서브 네트워크 손실 함수만으로 중요도를 제어했다는 점이 효율적이다. 또한, 기존의 다양한 CL 알고리즘과 결합 가능하다는 직교성은 실용적인 가치가 높다.

### 한계 및 미해결 질문
- **학습 시간 증가**: 여러 개의 서브 네트워크에 대해 순방향 전파(Forward pass)를 수행해야 하므로 학습 시간이 증가한다. 다만, 추론 시에는 전체 네트워크만 사용하므로 추론 비용은 동일하다.
- **단순한 스크롤링 전략**: 현재는 고정된 단계($S=1$)로 스크롤링하지만, 작업 간의 상관관계에 따라 스크롤링 크기를 동적으로 조절하는 방법이 고려될 수 있다.
- **모델 용량 문제**: TinyImageNet에서 성능 향상이 제한적이었던 점을 통해, 더 큰 모델(ResNet101, ViT 등)에서의 검증이 필요함을 시사한다.

## 📌 TL;DR

ScrollNet은 동적 네트워크(Slimmable Network)를 활용하여 **데이터 학습 전 가중치 중요도를 미리 할당**하고, 작업이 바뀔 때마다 이 순위를 **스크롤링(Scrolling)**하여 변경하는 새로운 지속 학습 방법론이다. 이 방식은 기존의 정규화 및 리플레이 기반 CL 방법들과 결합하여 **치명적 망각을 효과적으로 억제**하며, 특히 파라미터 정규화 기반 방법론들과 결합했을 때 탁월한 성능 향상을 보인다. 향후 모델 용량 확장 및 정교한 스크롤링 전략 연구를 통해 더욱 발전할 가능성이 크다.