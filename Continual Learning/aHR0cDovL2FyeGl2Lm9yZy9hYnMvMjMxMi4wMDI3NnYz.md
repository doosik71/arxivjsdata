# Metalearning Continual Learning Algorithms

Kazuki Iriekirie, Róbert Csordás, Jürgen Schmidhuber (2025)

## 🧩 Problem to Solve

본 논문은 신경망이 새로운 작업을 학습할 때 이전에 습득한 지식을 잃어버리는 **Catastrophic Forgetting (CF, 치명적 망각)** 문제를 해결하고자 한다. 일반적인 학습 알고리즘은 모든 데이터를 한 번에 사용할 수 있는 정적 환경을 가정하여 설계되었기 때문에, 데이터가 순차적으로 제공되는 Continual Learning (CL) 환경에서는 부적합하다.

기존의 CF 해결 방식은 Elastic Weight Consolidation (EWC)나 Synaptic Intelligence (SI)와 같이 인간이 직접 설계한 정규화 항(regularization terms)을 추가하여 가중치 변화를 제한하는 방식이었다. 하지만 이러한 방식은 수동 설계에 의존하며 모든 상황에 최적화되기 어렵다는 한계가 있다. 따라서 본 연구의 목표는 인간의 개입 없이 **신경망이 스스로 자신의 In-context Continual Learning 알고리즘을 Meta-learning 하도록 하는 Automated Continual Learning (ACL)** 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Self-referential Neural Networks (자가 참조 신경망)**를 활용하여, 모델이 입력 시퀀스를 처리하는 과정에서 자신의 가중치를 동적으로 수정하는 '학습 알고리즘' 자체를 학습하게 하는 것이다.

단순한 In-context Learning 역시 새로운 정보가 들어오면 이전 정보를 잊어버리는 'In-context Catastrophic Forgetting'을 겪게 된다. 이를 방지하기 위해 저자들은 CL의 필수 조건인 **Forward Transfer (새 작업을 배울 때 이전 지식 활용)**와 **Backward Transfer (새 작업을 배운 후에도 이전 작업 성능 유지)**를 Meta-learning의 목적 함수(Objective Function)에 직접 인코딩하였다. 이를 통해 모델은 경사 하강법(Gradient Descent)을 통해 CF를 자동으로 회피하는 최적의 가중치 수정 메커니즘을 스스로 발견하게 된다.

## 📎 Related Works

기존의 CL 연구는 크게 두 가지 방향으로 나뉜다. 첫째는 앞서 언급한 EWC, SI와 같은 정규화 기반 방법이며, 둘째는 과거 데이터를 저장해두었다가 다시 학습하는 Replay-based 방법이다. 본 논문은 별도의 Replay 메모리를 사용하지 않는 **Replay-free** 설정에 집중한다.

최근에는 MAML(Model-Agnostic Meta-Learning) 기반의 Meta-CL 연구(예: OML)나 생성 모델을 이용한 GeMCL 등이 제안되었다. 그러나 이러한 방법들은 여전히 Meta-test 단계에서 학습률(learning rate)이나 반복 횟수와 같은 하이퍼파라미터를 수동으로 튜닝해야 하는 번거로움이 있다. 반면, ACL은 가중치의 재귀적 자기 수정(recursive self-modification)을 통해 추론 과정에서 자동으로 CL이 수행되므로, 별도의 하이퍼파라미터 튜닝 없이도 작동하는 알고리즘을 지향한다.

## 🛠️ Methodology

### 전체 시스템 구조

ACL은 이미지 처리-특징 추출을 위한 **Vision Backend**와 순차적 데이터를 처리하며 가중치를 업데이트하는 **Recursive Self-Transformer**로 구성된다.

- **Vision Backend**: 기본적으로 Conv-4 아키텍처를 사용하며, CL 설정에서 일반화 성능을 높이기 위해 Batch Normalization 대신 **Instance Normalization (IN)**을 사용한다.
- **Recursive Self-Transformer**: 표준 Transformer의 Self-attention 층을 **Self-Referential Weight Matrix (SRWM)** 층으로 대체한 구조이다.

### Self-Referential Weight Matrix (SRWM)

SRWM은 입력 데이터에 반응하여 자신의 가중치 행렬 $W$를 실시간으로 수정하는 선형 Transformer의 일종이다. 시점 $t$에서 입력 $u_t$가 들어왔을 때의 동작은 다음과 같다.

1. **출력 및 벡터 생성**: 가중치 $W_{t-1}$을 통해 출력 $o_t$와 쿼리 $q_t$, 키 $k_t$, 학습률 $\beta_t$를 생성한다.
   $$ [o_t, k_t, q_t, \beta_t] = W_{t-1} u_t $$
2. **가치 벡터 계산**: $q_t$와 $k_t$를 이용해 가치 벡터 $v_t$와 $\bar{v}_t$를 계산한다.
   $$ v_t = W_{t-1} \phi(q_t), \quad \bar{v}_t = W_{t-1} \phi(k_t) $$
3. **가중치 업데이트 (Delta Rule)**: 델타 학습 규칙에 따라 가중치 행렬을 랭크-1 업데이트한다.
   $$ W_t = W_{t-1} + \sigma(\beta_t)(v_t - \bar{v}_t) \otimes \phi(k_t) $$
   여기서 $\sigma$는 시그모이드 함수, $\phi$는 소프트맥스 함수이며, $\otimes$는 외적(outer product)을 의미한다.

### ACL Meta-Training 목표 함수

ACL은 두 개의 작업 $A, B$를 순차적으로 학습하는 시나리오를 가정한다. 모델은 작업 $A$의 예시들을 처리한 후의 상태 $W^A$와, 작업 $B$까지 모두 처리한 후의 상태 $W^{A,B}$를 모두 사용한다. 목적 함수는 다음과 같다.

$$ \text{minimize}_{\theta} - \left( \log p(y^A_{\text{target}} | x^A_{\text{query}}; W^A(\theta)) + \log p(y^B_{\text{target}} | x^B_{\text{query}}; W^{A,B}(\theta)) + \log p(y^A_{\text{target}} | x^A_{\text{query}}; W^{A,B}(\theta)) \right) $$

이 수식의 각 항은 다음과 같은 CL의 목적을 가진다.

- **첫 번째 항**: 작업 $A$를 빠르게 습득하도록 유도한다.
- **두 번째 항**: 작업 $B$를 습득하며, 이때 $A$에서 배운 지식을 활용하는 **Forward Transfer**를 최적화한다.
- **세 번째 항**: 작업 $B$를 배운 후에도 $A$를 기억하게 함으로써 **Backward Transfer**를 최적화하고 CF를 방지한다.

## 📊 Results

### 실험 설정 및 데이터셋

- **데이터셋**: Meta-training에는 Mini-ImageNet, Omniglot, FC100을 사용하였으며, Meta-testing에는 MNIST, FashionMNIST, CIFAR-10 등을 사용하였다.
- **벤치마크**: Split-MNIST (5-task)를 통해 Domain-Incremental (DIL) 및 Class-Incremental (CIL) 설정을 평가하였다.
- **비교 대상**: SGD, Adam, EWC, SI, MAS, LwF, OML, GeMCL 등.

### 주요 결과

1. **In-context CF 확인**: ACL 목적 함수(특히 Backward Transfer 항) 없이 학습한 모델은 작업 $B$를 배우는 순간 작업 $A$의 성능이 무작위 수준(20%)으로 급락하는 'In-context CF' 현상이 관찰되었다. 반면, ACL 모델은 이를 효과적으로 억제하였다.
2. **Split-MNIST 성능**:
   - **Out-of-the-box**: 별도의 튜닝 없이도 기존의 수동 설계 알고리즘(EWC, SI 등)보다 우수한 성능을 보였다.
   - **Meta-finetuned**: 5-task ACL 손실 함수로 추가 학습시킨 모델은 모든 설정(DIL, CIL)에서 기존의 모든 Replay-free 방법론을 압도하는 최고 성능을 기록하였다.
3. **ViT 기반 확장 실험**: Frozen ViT-B/16 위에 SRWM을 얹어 실험한 결과, L2P와 같은 Prompt-based 방법론보다는 성능이 낮았다. 이는 현재의 ACL이 더 방대한 양의 다양하고 정교한 Meta-training 데이터셋을 필요로 함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 연구는 CL 문제를 '시퀀스 처리 문제'로 재정의하고, 학습 알고리즘 자체를 신경망의 가중치 업데이트 규칙으로 구현함으로써 **ALgorithm의 자동화**를 달성하였다. 특히 Replay 메모리 없이도 Meta-learning을 통해 CF를 극복할 수 있음을 보인 점이 고무적이다.

### 한계 및 비판적 해석

- **일반화 문제**: 데이터 기반의 학습 알고리즘이기 때문에 Meta-training 데이터셋에 포함되지 않은 완전히 새로운 도메인으로의 일반화(Domain Generalization)에 취약한 모습을 보인다.
- **길이 일반화 (Length Generalization)**: 학습 시 사용한 작업의 개수나 예시의 수보다 더 많은 데이터가 들어올 때 성능이 저하되는 현상이 관찰되었다. 이는 시퀀스 모델링 기반 접근법의 공통적인 한계이다.
- **해석 가능성**: 학습된 가중치 수정 규칙이 매우 복잡하여, 이를 통해 인간이 이해할 수 있는 새로운 CL 알고리즘 설계 원칙을 추출하는 것은 현재로서는 매우 어렵다.

## 📌 TL;DR

본 논문은 신경망이 스스로 CF를 방지하는 학습 규칙을 배우게 하는 **Automated Continual Learning (ACL)**을 제안한다. SRWM(자가 참조 가중치 행렬)과 Forward/Backward Transfer가 반영된 Meta-loss를 통해, 모델은 추론 단계에서 가중치를 동적으로 수정하며 지속적으로 학습한다. Split-MNIST 벤치마크에서 기존의 수동 설계 CL 알고리즘과 Meta-CL 방법론을 뛰어넘는 성능을 보였으며, 이는 향후 인간의 개입 없는 개방형 지속 학습 시스템 구축의 가능성을 제시한다.
