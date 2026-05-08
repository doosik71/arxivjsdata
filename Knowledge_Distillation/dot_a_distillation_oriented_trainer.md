# DOT: A Distillation-Oriented Trainer

Borui Zhao, Quan Cui, Renjie Song, Jiajun Liang (2023)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD) 과정에서 발생하는 **Task Loss(작업 손실)와 Distillation Loss(증류 손실) 사이의 트레이드-오프(Trade-off)** 문제를 해결하고자 한다.

일반적인 KD 프레임워크에서는 학생 모델이 교사 모델의 지식을 학습하도록 두 가지 손실 함수를 결합하여 사용한다. 저자들은 실험적 관찰을 통해 증류 손실을 도입하는 것이 학생 모델이 더 평탄한 최솟값(Flat Minima)에 도달하게 하여 일반화 성능을 높이는 긍정적인 효과가 있지만, 동시에 Task Loss의 수렴을 방해하여 작업 손실 값이 오히려 증가하는 현상을 발견하였다.

연구의 목표는 이러한 트레이드-오프 관계를 깨뜨려, Task Loss의 충분한 수렴과 증류 손실의 최적화를 동시에 달성함으로써 학생 모델의 최종 성능과 일반화 능력을 극대화하는 최적화 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **증류 손실의 최적화를 가속화하여 최적화 방향을 증류 중심으로 유도(Distillation-Oriented)**하는 것이다.

저자들은 교사 모델이 학생 모델보다 항상 더 낮은 Task Loss를 가진다는 점에 주목하였다. 따라서 증류 손실이 충분히 최적화되어 학생이 교사와 더 유사해진다면, 결과적으로 Task Loss 또한 자연스럽게 감소할 것이라고 가설을 세웠다. 이를 위해 Task Loss와 Distillation Loss에 서로 다른 모멘텀(Momentum)을 적용하여, 증류 손실이 최적화 과정을 주도하도록 설계한 **DOT(Distillation-Oriented Trainer)**를 제안하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Knowledge Distillation**: 기존 연구들은 주로 어떤 지식(Logits 또는 Intermediate Features)을 전달할 것인가에 집중하거나, KD가 왜 작동하는지에 대한 이론적 분석(데이터 기하학, 특권 정보 관점 등)에 치중하였다.
2. **Flatness of Minima**: 손실 함수 지형(Loss Landscape)의 평탄도가 모델의 일반화 성능과 밀접한 관련이 있다는 연구들이 존재한다. 평탄한 최솟값은 훈련 데이터와 테스트 데이터 사이의 간극을 줄여 더 나은 일반화 성능을 보장한다.

### 기존 방식과의 차별점

기존의 KD 방법론들은 단순히 두 손실 함수를 가중합(Weighted Sum) 형태로 결합하여 최적화하였다. 이는 최적화 관점에서 보면 네트워크가 두 작업 사이의 균형을 찾는 멀티태스크 학습(Multi-task Learning) 형태로 퇴화하며, 이 과정에서 필연적으로 트레이드-오프가 발생한다. 반면, DOT는 손실 함수의 구성이 아닌 **최적화 단계(Optimizer)에서의 모멘텀 조절**을 통해 이 문제를 해결하려 한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

DOT는 기존의 SGD(Stochastic Gradient Descent) 모멘텀 메커니즘을 수정하여, Task Loss와 Distillation Loss 각각에 대해 **독립적인 모멘텀 버퍼**를 유지한다.

### 상세 작동 원리 및 학습 절차

1. **독립적 모멘텀 적용**: Task Loss($L_{CE}$)와 Distillation Loss($L_{KD}$)에서 발생하는 기울기(Gradient)를 각각 계산한다.
2. **모멘텀 계수 차별화**: 하이퍼파라미터 $\Delta$를 도입하여, 두 손실 함수에 서로 다른 모멘텀 계수를 적용한다.
    - Task Loss 모멘텀: $\mu - \Delta$
    - Distillation Loss 모멘텀: $\mu + \Delta$
3. **최적화 주도권 확보**: 더 큰 모멘텀이 적용된 증류 손실의 기울기가 더 많이 축적되고 유지되므로, 전체 최적화 방향은 증류 손실에 의해 주도된다.

### 주요 방정식

각 미니배치 데이터에 대해 다음과 같이 모멘텀 버퍼 $v_{ce}$와 $v_{kd}$를 업데이트한다.

$$v_{ce} \leftarrow g_{ce} + (\mu - \Delta)v_{ce}$$
$$v_{kd} \leftarrow g_{kd} + (\mu + \Delta)v_{kd}$$

최종적으로 네트워크 파라미터 $\theta$는 두 모멘텀 버퍼의 합을 이용하여 업데이트된다.

$$\theta \leftarrow \theta - \gamma(v_{ce} + v_{kd})$$

여기서 $g_{ce}$와 $g_{kd}$는 각각의 손실 함수에 대한 기울기, $\mu$는 기본 모멘텀 계수, $\gamma$는 학습률(Learning Rate)을 의미한다.

### 이론적 분석

저자들은 DOT와 바닐라(Vanilla) SGD의 차이를 분석하여, 두 손실 함수의 기울기가 충돌할 때 DOT가 다음과 같은 차분 벡터를 생성함을 보였다.

$$v_{diff} = v_{dot} - v_{sgd} = \Delta(v_{incon}^{kd} - v_{incon}^{ce})$$

이는 최적화 과정에서 증류 손실의 기울기 축적을 가속화하여, 기울기 간의 충돌을 완화하고 최적화 방향을 증류 중심으로 유도함을 수학적으로 뒷받침한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-100, Tiny-ImageNet, ImageNet-1k.
- **비교 대상**: Vanilla KD, CRD(Contrastive Representation Distillation), DKD(Decoupled Knowledge Distillation).
- **측정 지표**: Top-1 및 Top-5 정확도(Accuracy).

### 주요 결과

1. **성능 향상**: DOT는 기존의 Logit 기반 및 Feature 기반 KD 방법론 모두와 결합 가능하며, 일관된 성능 향상을 보였다.
    - **ImageNet-1k**: ResNet50(교사)-MobileNetV1(학생) 쌍에서 Vanilla KD 대비 **+2.59%**의 정확도 향상을 기록하며 새로운 SOTA를 달성하였다.
    - **Tiny-ImageNet**: ResNet18-MobileNetV2 쌍에서 Top-1 정확도를 58.35%에서 64.01%로 크게 끌어올렸다.
2. **트레이드-오프 해결**: 손실 곡선 시각화를 통해 DOT가 Vanilla Trainer나 단순히 가중치 $\alpha$를 조절한 방식과 달리, **Task Loss와 Distillation Loss를 동시에 낮게 유지**하며 수렴함을 확인하였다.
3. **최솟값의 평탄도**: Loss Landscape 시각화 결과, DOT를 적용했을 때 학생 모델이 Vanilla KD보다 더 평탄한 최솟값(Flatter Minima)에 도달함을 확인하였으며, 이는 더 높은 일반화 성능으로 이어졌다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 KD의 성능 향상을 단순히 '무엇을 증류할 것인가'의 관점이 아니라 '어떻게 최적화할 것인가'라는 최적화 관점에서 접근하여 유의미한 성과를 거두었다. 특히, 단순히 손실 함수의 가중치 $\alpha$를 조절하는 것만으로는 Task-Distillation 간의 트레이드-오프를 해결할 수 없음을 실험적으로 증명하고, 모멘텀 조절이라는 단순하면서도 강력한 해결책을 제시하였다.

### 한계 및 논의사항

- **하이퍼파라미터 $\Delta$**: $\Delta$ 값은 데이터셋마다 약간의 조정이 필요하다(CIFAR-100: 0.075, ImageNet-1k: 0.09). 다만, 저자들은 $\Delta$가 매우 민감한 파라미터는 아니라고 주장한다.
- **Feature 기반 KD 적용**: Feature 기반 방법론의 경우, 모든 파라미터가 두 손실 함수에 공통으로 관여하지 않으므로, 두 손실 모두에 참여하는 파라미터에만 DOT를 적용해야 하는 구현상의 세밀함이 요구된다.

## 📌 TL;DR

본 논문은 지식 증류 과정에서 Task Loss와 Distillation Loss 사이에 발생하는 수렴 트레이드-오프 문제를 발견하고, 이를 해결하기 위해 **증류 손실에 더 큰 모멘텀을 부여하는 DOT(Distillation-Oriented Trainer)**를 제안한다. DOT는 최적화 방향을 증류 중심으로 유도하여 더 평탄한 최솟값에 도달하게 하며, ImageNet-1k 등 주요 벤치마크에서 기존 SOTA 방법론들을 뛰어넘는 성능 향상을 입증하였다. 이 연구는 향후 KD 연구가 최적화 관점에서도 다뤄져야 함을 시사한다.
