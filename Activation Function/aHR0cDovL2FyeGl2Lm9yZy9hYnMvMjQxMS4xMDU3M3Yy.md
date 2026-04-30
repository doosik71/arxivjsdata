# Hysteresis Activation Function for Efficient Inference

Moshe Kimhi, Idan Kashani, Avi Mendelson, Chaim Baskin (2024/2025)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델에서 널리 사용되는 ReLU(Rectified Linear Unit) 활성화 함수의 치명적인 단점인 **"Dying ReLU"** 문제를 해결하고자 한다. Dying ReLU 현상이란 학습 과정에서 뉴런의 입력값이 음수가 되어 출력이 0이 되고, 이로 인해 gradient가 흐르지 않아 해당 뉴런이 영구적으로 비활성화되는 현상을 의미한다. 이는 네트워크의 학습 능력을 저하시키고 일반화 성능을 떨어뜨리는 결과를 초래한다.

이를 해결하기 위해 GELU나 Swish와 같은 대안적인 활성화 함수들이 제안되었으나, 이러한 함수들은 추가적인 연산(지수 함수, 곱셈 등)을 필요로 하여 추론(Inference) 단계에서의 하드웨어 효율성이 떨어진다는 문제가 있다. 따라서 본 연구의 목표는 ReLU의 하드웨어적 효율성(추론 시 단순한 sign check만으로 구현 가능)을 그대로 유지하면서, Dying ReLU 문제를 해결하여 성능을 높일 수 있는 효율적인 활성화 함수를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 물리 및 전자 공학의 **Hysteresis(이력 현상)** 개념을 활성화 함수에 도입하는 것이다. Hysteresis는 입력 조건이 변하더라도 이전의 상태가 일정 기간 유지되는 특성을 말한다. 

연구진은 이를 신경망에 적용하여, **순전파(Forward pass)와 역전파(Backward pass) 시에 서로 다른 임계값(Threshold)을 적용**하는 **HeLU(Hysteresis Rectified Linear Unit)**를 제안한다. 즉, 추론 시에는 ReLU와 동일하게 동작하여 연산 비용을 제로(0)로 유지하되, 학습 시의 역전파 과정에서는 임계값을 낮게 설정하여 음수 영역에 있는 뉴런들에게도 gradient가 전달되도록 함으로써 뉴런이 완전히 "죽는" 것을 방지한다.

## 📎 Related Works

기존의 연구들은 모델 효율성을 높이기 위해 Pruning(가지치기)이나 Quantization(양자화)과 같은 기법을 사용해 왔다. 하지만 본 논문은 이러한 고수준의 최적화 이전에, 가장 기본 단위인 활성화 함수 자체를 최적화하는 것이 추론 처리량(Throughput) 향상에 더 직접적인 영향을 줄 수 있다고 주장한다.

활성화 함수와 관련하여 다음과 같은 기존 방식들의 한계점을 지적한다:
- **ReLU**: 하드웨어 구현이 매우 간단하지만 Dying ReLU 문제에 취약하다.
- **Sigmoid**: Vanishing Gradient 문제로 인해 깊은 신경망 학습이 어렵다.
- **GELU / Swish**: 성능은 우수하나, Tanh나 지수 연산 등이 포함되어 추론 시 레이턴시(Latency)를 증가시킨다. 특히 GPU 환경에서 GELU 연산이 전체 GPT-3 레이턴시의 약 6%를 차지한다는 분석을 인용하며 효율성 문제를 제기한다.
- **Learnable Activation (e.g., KAN)**: 학습 가능한 활성화 함수는 유연하지만 학습 시간이 지나치게 오래 걸리고 추론 효율성이 보장되지 않는다.

## 🛠️ Methodology

### 전체 구조 및 동작 원리
HeLU는 순전파 단계에서는 표준 ReLU와 완전히 동일하게 동작하며, 오직 역전파 단계에서만 미분 정의를 수정하여 Hysteresis 효과를 구현한다.

### 주요 방정식

**1. 순전파 (Forward Pass)**
HeLU의 순전파 정의는 ReLU와 동일하다.
$$ \text{HeLU}_\alpha(x) = \text{ReLU}(x) = \max\{0, x\} $$

**2. 역전파 (Backward Pass / Modified Derivative)**
역전파 시에는 하이퍼파라미터 $\alpha \in \mathbb{R}$를 도입하여 gradient가 흐르는 임계값을 이동시킨다.
$$ \frac{d}{dx} \text{HeLU}_\alpha(x) = \begin{cases} 1 & \text{if } x > -\alpha \\ 0 & \text{if } x \le -\alpha \end{cases} $$

### 학습 및 추론 절차
- **학습 단계**: 뉴런의 입력값 $x$가 $0$보다 작더라도 $-\alpha$보다 크다면 gradient가 1로 전달된다. 이를 통해 $\text{ReLU}$에서는 출력이 0이 되어 학습이 멈췄을 영역에서도 가중치 업데이트가 가능해지며, 뉴런의 활성 상태를 복구할 기회를 제공한다.
- **추론 단계**: 학습 시 사용된 $\alpha$는 역전파 전용 파라미터이므로 추론 시에는 완전히 제거된다. 결과적으로 추론 시의 연산 구조는 $\max\{0, x\}$인 ReLU와 완전히 동일하여 추가 비용이 발생하지 않는다.

## 📊 Results

### 1. 이미지 분류 (Image Classification)
- **데이터셋 및 모델**: CIFAR10, CIFAR100, Imagenette 데이터셋을 사용하였으며, Wide ResNet 40-4 아키텍처를 채택하였다.
- **결과**: HeLU는 ReLU보다 일관되게 높은 정확도를 보였으며, 특히 $\alpha=0.001$ 설정에서 GELU에 근접하는 성능을 기록하였다.
    - CIFAR10: ReLU(92.84%) $\rightarrow$ HeLU $\alpha=0.001$(95.80%) / GELU(96.11%)
    - CIFAR100: ReLU(75.31%) $\rightarrow$ HeLU $\alpha=0.001$(77.51%) / GELU(79.26%)

### 2. 언어 이해 평가 (GLUE Benchmark)
- **데이터셋 및 모델**: BERT-base-uncased 모델을 사용하여 8개의 GLUE 태스크를 수행하였다.
- **결과**: $\alpha=0.05$ 설정의 HeLU가 ReLU보다 우수한 성능을 보였으며, GELU와 ReLU 사이의 성능 간극을 메우는 효과를 확인하였다.
- **양자화 영향**: QBERT-INT8 양자화 환경에서 HeLU($\alpha=0.05$)는 ReLU나 GELU보다 성능 저하가 적거나 오히려 개선되는 경향을 보였다.

### 3. 계산 효율성 분석 (Computational Analysis)
- **추론 시간**: BERT 모델 기준, HeLU는 GELU보다 추론 시간이 단축되었다. 특히 다양한 트랜스포머 아키텍처(RoBERTa, XLNet, GPT-2, Electra)에서 모두 원본 활성화 함수 대비 추론 시간 감소가 관찰되었다.
- **처리량(Throughput)**: INT8 양자화 모델에서 HeLU는 GELU보다 더 높은 토큰 처리량을 기록하였다. 이는 메모리 제약이 심한 환경에서 HeLU의 단순한 연산 구조가 더 큰 이점을 제공함을 시사한다.

## 🧠 Insights & Discussion

### 강점
HeLU의 가장 큰 강점은 **"추론 비용 제로(Zero-cost inference)"**이면서 **"학습 안정성 확보"**라는 두 마리 토끼를 잡았다는 점이다. 기존의 Dying ReLU 해결책들이 추론 시 연산 복잡도를 높였던 것과 달리, HeLU는 오직 학습 단계의 Gradient flow만 수정함으로써 동일한 추론 효율성을 유지하며 성능을 끌어올렸다.

### 한계 및 논의사항
- **하이퍼파라미터 의존성**: 결과에서 볼 수 있듯이 $\alpha$ 값에 따라 성능 편차가 존재한다. $\alpha$가 너무 크면(예: $\alpha=2$) 오히려 성능이 급격히 하락하는 현상이 관찰되었다. 따라서 최적의 $\alpha$를 찾기 위한 추가적인 탐색 과정이 필요하다.
- **적용 범위**: 본 연구는 분류(Discriminative) 작업에 집중하였다. 저자들은 향후 연구에서 세그멘테이션(Segmentation)이나 노이즈 섞인 라벨 데이터 환경에서도 HeLU가 효과적일지 탐구할 필요가 있다고 언급한다.

### 비판적 해석
HeLU는 새로운 수학적 함수를 제안했다기보다, 역전파 시의 gradient masking 영역을 조정하는 일종의 **정규화(Regularization) 기법**으로 해석될 수 있다. 이는 ReLU의 단순함을 유지하면서 GELU의 부드러운 특성을 모사하려는 실용적인 접근법이며, 특히 edge device와 같이 연산 자원이 극도로 제한된 환경에서 매우 유용한 전략이 될 것이다.

## 📌 TL;DR

본 논문은 ReLU의 하드웨어 효율성은 유지하면서 "Dying ReLU" 문제를 해결하기 위해, 순전파와 역전파의 임계값을 다르게 설정하는 **HeLU(Hysteresis ReLU)**를 제안한다. HeLU는 추론 시에는 ReLU와 완전히 동일하게 동작하여 추가 연산 비용이 없지만, 학습 시에는 $\alpha$라는 하이퍼파라미터를 통해 음수 영역에서도 gradient가 흐르게 하여 뉴런의 소멸을 방지한다. 실험 결과, HeLU는 이미지 분류 및 NLP 작업에서 ReLU를 상회하고 GELU에 근접하는 성능을 보였으며, 추론 속도와 양자화 효율성 면에서 상당한 이점을 가짐을 입증하였다. 이는 향후 고효율 저전력 AI 모델 설계에 중요한 기여를 할 것으로 보인다.