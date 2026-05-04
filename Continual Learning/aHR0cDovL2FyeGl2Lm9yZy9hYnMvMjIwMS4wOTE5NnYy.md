# Learning to Predict Gradients for Semi-Supervised Continual Learning

Yan Luo, Yongkang Wong, Mohan Kankanhalli, and Qi Zhao (202X)

## 🧩 Problem to Solve

본 논문은 기계 지능의 핵심 과제인 새로운 시각적 개념을 학습하면서도 이전의 지식을 잊어버리지 않는 Continual Learning (CL)의 문제, 특히 Semi-Supervised Continual Learning (SSCL) 환경에서의 한계를 해결하고자 한다.

기존의 CL 및 SSCL 방법론들은 학습 샘플이 알려진 라벨(known labels)과 연관되어 있다는 가정을 전제로 한다. 그러나 실제 인간의 학습 과정은 일상생활에서 라벨이 알려진 데이터뿐만 아니라 라벨이 알려지지 않은(unknown labels) 데이터로부터도 지속적으로 학습한다는 점에서 기존 방식과 큰 차이가 있다.

따라서 본 연구는 다음 두 가지 핵심 질문에 집중한다.

1. SSCL 작업에서 관련 없는(unrelated) unlabeled data를 어떻게 활용할 것인가?
2. unlabeled data가 CL 작업의 학습 성능과 Catastrophic Forgetting(치명적 망각)에 어떤 영향을 미치는가?

결과적으로 본 논문의 목표는 라벨이 없는 데이터의 실제 클래스가 기존에 알려진 클래스인지 혹은 완전히 새로운 클래스인지에 관계없이, 이를 활용하여 모델의 일반화 능력을 높이고 망각 현상을 완화하는 새로운 SSCL 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 라벨이 없는 데이터에 대해 직접적인 Pseudo-label을 생성하는 대신, 라벨이 있는 데이터로부터 학습된 **Gradient Learner**를 통해 해당 데이터의 **Pseudo-gradients**를 예측하여 모델을 업데이트하는 것이다.

기존의 Pseudo-labeling 방식은 특정 카테고리로 데이터를 강제 할당하기 때문에, 데이터의 실제 라벨이 알려지지 않은 상황에서는 잘못된 방향으로 학습이 진행될 위험이 크다. 반면, 제안된 Gradient Learner는 특징(feature)과 그래디언트 사이의 매핑을 학습함으로써, 특정 클래스에 국한되지 않은 일반적인 지식을 바탕으로 그래디언트를 예측한다. 이를 통해 unlabeled data를 감독 학습(Supervised Learning) 프레임워크에 자연스럽게 통합시킬 수 있다.

## 📎 Related Works

### 1. Continual Learning (CL)

CL은 순차적인 태스크를 학습하며 지식을 유지하는 것이 목표이다. 주요 전략으로는 메모리 버퍼를 사용하는 Rehearsal-based, 파라미터 변화를 제한하는 Regularization-based, 이전 모델의 지식을 전수하는 Knowledge Distillation-based 방법들이 있다. GEM, DCL, ACL 등이 대표적이며, 이들은 모두 ground-truth 라벨이 필요하다는 제약이 있다.

### 2. Semi-supervised Learning (SSL)

SSL은 적은 양의 labeled data와 많은 양의 unlabeled data를 함께 사용한다. 대부분의 현대적 딥러닝 기반 SSL은 Pseudo-labeling이나 Self-training에 의존한다. 하지만 CL 설정에서는 태스크마다 클래스가 변하고 학습 데이터가 부족하여 강력한 Teacher 모델을 구축하기 어렵기 때문에, 기존 SSL의 Pseudo-labeling 방식을 그대로 적용하기에는 한계가 있다.

### 3. 차별점

본 논문은 unlabeled data의 라벨이 알려진 클래스 집합에 속한다는 제약을 없앴다. 또한, 예측된 라벨을 통해 손실 함수를 계산하는 간접적인 방식이 아니라, 그래디언트 자체를 예측하여 역전파(back-propagation)에 직접 사용하는 방식을 제안함으로써 Pseudo-labeling의 오류 전파 문제를 회피한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 방법론은 크게 **Gradient Learning** 단계와 **Gradient Prediction** 단계로 구성된다.

### 2. Gradient Learning (그래디언트 학습)

모델 $f_\theta$가 labeled data $(x_i, y_i)$를 학습할 때, Gradient Learner $h(\cdot; \omega)$ 역시 함께 학습된다. $h$는 모델의 출력인 logits $z_i$를 입력으로 받아 그래디언트 $g_i$를 예측하는 매핑 함수이다.

$$g_i = h(z_i; \omega)$$

이때 $h$를 학습시키기 위해 단순한 모방 학습이 아닌, 예측된 그래디언트가 실제 손실 함수를 얼마나 효과적으로 줄이는지를 측정하는 **Fitness Loss** $\ell_{fit}$를 정의한다.

$$\ell_{fit}(z_i, \bar{g}_i, y_i) = \lambda \ell(z_i - \eta \bar{g}_i, y_i)$$

여기서 $\bar{g}_i$는 학습의 안정성을 위해 vanilla gradient의 크기 $\tau_i = \|\frac{\partial \ell}{\partial z_i}\|$를 참조하여 정규화된 그래디언트이다.

$$\bar{g}_i = \alpha \tau_i \frac{g_i}{\|g_i\|}$$

$\alpha$는 예측 그래디언트의 크기를 조절하는 하이퍼파라미터이며, $\lambda$는 fitness loss의 스케일을 조정하는 계수이다. Gradient Learner $\omega$는 이 $\ell_{fit}$를 최소화하는 방향으로 업데이트된다.

### 3. Gradient Prediction (그래디언트 예측)

학습된 $h$를 사용하여 unlabeled data $\tilde{x}_i$에 대한 pseudo-gradient $\bar{g}|_{\tilde{x}_i}$를 생성한다. 이때 $\tilde{x}_i$는 라벨이 없으므로 $\tau_i$를 계산할 수 없으며, 대신 직전 labeled sample의 그래디언트 크기 $\tau_{i-1}$를 사용한다. 예측된 그래디언트는 다음과 같이 모델 파라미터 $\theta$를 업데이트하는 데 사용된다.

$$\theta \leftarrow \theta - \eta \bar{g}|_{\tilde{x}_i} \frac{\partial \bar{g}|_{\tilde{x}_i}}{\partial \theta}$$

### 4. Sampling Policy 및 Trade-off

unlabeled data가 너무 많거나 분포가 너무 다르면 예측 오류가 누적되어 학습을 방해할 수 있다(Overwhelming). 이를 방지하기 위해 확률적 임계값 $p$를 도입하여, $q < p$인 경우에만 unlabeled data를 샘플링하여 학습에 사용함으로써 일반화(Generalizing)와 과부하(Overwhelming) 사이의 균형을 맞춘다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MNIST-R, MNIST-P, iCIFAR-100 (GEM/DCL 설정), CIFAR-100, miniImageNet (ACL 설정). SSL 검증을 위해 SVHN, CIFAR-10/100 사용.
- **unlabeled data**: Tiny ImageNet 및 MS COCO 데이터를 외부 데이터셋으로 활용.
- **비교 대상**: EWC, GEM, DCL, ACL 등 기존 CL 모델 및 Pseudo-labeling (1-PL, P-PL), Meta-gradient (MG) 방식.
- **측정 지표**: Average Accuracy (ACC), Backward Transfer (BWT), Forward Transfer (FWT).

### 2. 주요 결과

- **성능 향상**: 제안된 방법(Proposed)을 적용했을 때, 모든 백본(MLP, ResNet, EffNet)과 CL 알고리즘(GEM, DCL, ACL)에서 ACC와 BWT가 일관되게 향상되었다.
- **망각 완화**: BWT 수치가 개선된 것은 unlabeled data의 활용이 Catastrophic Forgetting을 유의미하게 완화했음을 보여준다.
- **Pseudo-labeling 대비 우위**: 1-PL, P-PL 방식은 잘못된 pseudo-label로 인해 ACC가 오히려 하락하는 경향을 보였으나, 제안 방법은 이를 극복하고 성능을 높였다.
- **SSL 일반화**: SSL 작업(SVHN 등)에서도 MG 방식과 유사하거나 더 나은 에러율을 기록하여, 제안 방법이 범용적으로 작동함을 입증하였다.

## 🧠 Insights & Discussion

### 1. Pseudo-labeling vs. Gradient Prediction

논문은 왜 Pseudo-labeling이 CL에서 작동하지 않는지 분석한다. CL에서는 각 클래스당 학습 샘플 수가 매우 적어 강력한 Teacher 모델을 만들 수 없으며, 태스크가 진행됨에 따라 시각적 개념이 계속 변하기 때문에 고정된 라벨 예측이 어렵다. 반면, Gradient Prediction은 특정 클래스에 매몰되지 않고 학습된 전반적인 시각적 지식을 그래디언트에 투영하므로 더 높은 일반화 성능을 보인다.

### 2. 시각적 다양성의 영향

실험 결과, unlabeled data가 학습 데이터와 시각적으로 유사할수록(예: Tiny ImageNet) 성능 향상이 컸으며, 매우 이질적인 데이터(예: FGVC-aircraft)에서는 향상 폭이 적었다. 이는 예측 그래디언트가 결국 기존에 학습된 특징 공간 내에서 작동하기 때문이다.

### 3. 한계 및 해석

- **초기 학습 단계**: Gradient Learner가 충분히 학습되기 전인 초기 태스크에서는 unlabeled data를 사용하는 것이 오히려 성능을 떨어뜨릴 수 있다. 이를 위해 일정 스텝 이후부터 예측 그래디언트를 적용하는 전략을 사용하였다.
- **하이퍼파라미터 민감도**: $p, \alpha, \lambda$ 값에 따라 성능 변화가 크며, 특히 $p$ 값이 너무 높으면 예측 오류가 누적되어 성능이 급격히 하락하는 특성을 보인다.

## 📌 TL;DR

본 논문은 라벨이 없는 데이터의 클래스가 알려지지 않은 상황에서도 이를 CL 학습에 활용할 수 있는 **Gradient Learner** 기반의 SSCL 방법론을 제안한다. Pseudo-label을 생성하는 대신, 모델의 출력(logits)으로부터 최적의 그래디언트를 직접 예측하여 모델을 업데이트함으로써, **Catastrophic Forgetting을 완화하고 모델의 일반화 성능을 크게 향상**시켰다. 이 연구는 대규모의 unlabeled data를 활용해 지속적 학습의 효율성을 높이는 새로운 방향성을 제시하며, 향후 데이터 라벨링 비용이 높은 실제 환경의 CL 시스템 구축에 중요한 역할을 할 가능성이 크다.
