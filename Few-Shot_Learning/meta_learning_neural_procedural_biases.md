# Neural Procedural Bias Meta-Learning (NPBML)

Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang (2024)

## 🧩 Problem to Solve

본 논문은 Few-shot Learning(FSL) 환경에서 새로운 태스크에 빠르게 적응하기 위한 Gradient-based Meta-learning의 한계를 해결하고자 한다. 기존의 Model-Agnostic Meta-Learning(MAML)과 그 변형 모델들은 주로 공유된 파라미터 초기화(shared parameter initialization)를 학습하는 데 집중하였다. 그러나 이러한 방식들은 내부 최적화(inner optimization) 과정에서 단순한 Stochastic Gradient Descent(SGD)와 고정된 손실 함수(Cross-entropy 또는 Squared loss)를 사용한다는 한계가 있다.

모든 태스크에 동일한 학습 규칙을 적용하는 것은 데이터가 매우 제한적인 Few-shot 상황에서 최적의 성능을 끌어내는 데 제약이 된다. 따라서 본 연구의 목표는 학습 알고리즘의 수렴 속도, 샘플 효율성, 일반화 성능을 결정짓는 '절차적 편향(Procedural Biases)'을 메타 학습함으로써, 각 태스크에 최적화된 적응형 학습 규칙을 생성하는 Neural Procedural Bias Meta-Learning(NPBML) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 학습 알고리즘의 세 가지 핵심 구성 요소인 **파라미터 초기화(Initialization), 옵티마이저(Optimizer), 그리고 손실 함수(Loss Function)**를 동시에 메타 학습하고, 이를 각 태스크에 맞게 동적으로 조정하는 것이다.

특히, Feature-wise Linear Modulation(FiLM)을 도입하여 메타 학습된 구성 요소들이 개별 태스크의 특성에 맞게 변형(modulation)되도록 설계하였다. 이를 통해 각 태스크는 매우 적은 수의 gradient step만으로도 높은 성능을 달성할 수 있는 고유한 절차적 편향을 갖게 된다. 또한, 이러한 명시적 학습을 통해 학습률(learning rate)이나 가중치 감쇠(weight decay)와 같은 하이퍼파라미터들이 암묵적으로(implicitly) 함께 학습된다는 점을 밝혀냈다.

## 📎 Related Works

기존의 Few-shot Learning 연구는 크게 세 가지로 나뉜다:

1. **Metric-based methods**: 클래스 간의 유사도 측정 방식을 학습한다.
2. **Memory-based methods**: 메모리 구조를 사용하여 빠른 적응을 돕는다.
3. **Optimization-based methods**: 빠른 적응을 위한 최적화 알고리즘 자체를 학습하며, MAML이 대표적이다.

MAML 이후의 연구들은 주로 내부 최적화 규칙을 개선하려 노력했다. 일부는 학습률(learning rate)을 메타 학습하거나, Preconditioned Gradient Descent(PGD)를 통해 옵티마이저를 개선하였고, 또 다른 연구들은 고정된 손실 함수를 대체하는 메타 학습된 손실 함수를 제안하였다.

하지만 기존 연구들은 옵티마이저나 손실 함수 중 어느 하나에만 집중하는 경향이 있었으며, 학습된 구성 요소들이 개별 태스크에 따라 유연하게 변하는 '태스크 적응성(task-adaptivity)'이 부족했다. NPBML은 이 모든 요소를 통합하고 FiLM을 통해 태스크별 최적화를 구현함으로써 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

NPBML은 외곽 최적화(Outer optimization)를 통해 메타 파라미터 $\Phi = \{\theta, \omega, \phi, \psi\}$를 학습하고, 내부 최적화(Inner optimization)에서 이를 이용해 태스크별 파라미터 $\theta_i$를 빠르게 업데이트한다. 여기서 $\theta$는 초기화 값, $\omega$는 옵티마이저, $\phi$는 손실 함수, $\psi$는 FiLm 모듈의 파라미터이다.

### 주요 구성 요소 및 역할

**1. Meta-learned Optimizer ($\omega$):**
단순 SGD 대신 Preconditioned Gradient Descent(PGD) 방식을 사용한다. 메타 파라미터 $\omega$로 구성된 선형 투영 층(linear projection layers)을 모델의 각 층 사이에 삽입하여, 그래디언트의 기하학적 구조를 변경하는 preconditioner $P_\omega$를 학습한다. 이는 블록 대각 행렬(block-diagonal matrix) 형태로 정의되며, $\omega \omega^T$ 표현식을 통해 그래디언트를 변형한다.

**2. Meta-learned Loss Function ($\phi$):**
고정된 $L_{base}$ 대신 다음과 같이 세 가지 요소의 합으로 구성된 메타 학습 손실 함수 $M_{(\phi, \psi)}$를 사용한다:
$$M_{(\phi, \psi)} = L_S(\phi, \psi) + L_Q(\phi, \psi) + R(\phi, \psi)$$

- $L_S$ (Inductive loss): 서포트 세트의 정답 레이블과 모델 예측값, 그리고 $L_{base}$ 값을 입력으로 받는 유도 손실이다.
- $L_Q$ (Transductive loss): 쿼리 세트의 예측값과 사전 학습된 Relation Network의 임베딩 값을 활용하는 전이 손실이다.
- $R$ (Weight regularizer): 모델 파라미터 $\theta$의 평균, 표준편차, $L_1, L_2$ 노름을 입력으로 받아 가중치를 규제한다.

**3. Task-Adaptive Modulation ($\psi$):**
위의 구성 요소들이 각 태스크에 적응할 수 있도록 FiLM(Feature-wise Linear Modulation) 층을 삽입한다. FiLM은 입력 $x$에 대해 다음과 같은 아핀 변환(affine transformation)을 수행한다:
$$\text{FiLM}_\psi(x) = (\gamma_\psi(x) + 1) \odot x + \beta_\psi(x)$$
여기서 $\gamma$는 스케일링 벡터, $\beta$는 시프트 벡터이며, $\psi$에 의해 결정된다. 이 모듈은 인코더와 손실 함수 네트워크 내부에서 작동하여 태스크별 특성을 반영한다.

### 학습 및 추론 절차

- **내부 업데이트 규칙:** 태스크 $i$의 $j$번째 스텝에서 파라미터는 다음과 같이 업데이트된다:
$$\theta_{i,j+1} = \theta_{i,j} - \alpha P_{(\omega, \psi)} \nabla_{\theta_{i,j}} M_{(\phi, \psi)}(D^S_i, \theta_{i,j})$$
- **외곽 최적화:** 쿼리 세트 $D^Q_i$에서의 최종 손실 $L_{meta}$를 최소화하도록 $\Phi$를 업데이트한다:
$$\Phi_{new} = \Phi - \eta \nabla_\Phi \sum_{T_i \sim p(T)} L_{meta}(D^Q_i, \theta_{i,j}(\Phi))$$

## 📊 Results

### 실험 설정

- **데이터셋**: mini-ImageNet, tiered-ImageNet, CIFAR-FS, FC-100.
- **설정**: 5-way 1-shot 및 5-way 5-shot 분류 작업.
- **모델**: 4-CONV (저용량) 및 ResNet-12 (고용량) 아키텍처.
- **지표**: Meta-testing accuracy.

### 주요 결과

1. **정량적 성능**: NPBML은 모든 벤치마크에서 MAML, MetaSGD, WarpGrad, ALFA 등 기존의 state-of-the-art(SOTA) 모델들을 일관되게 상회하였다. 특히 데이터셋 규모가 큰 tiered-ImageNet에서 성능 향상 폭이 더 두드러지게 나타났다.
2. **아키텍처 영향**: 4-CONV와 ResNet-12 모두에서 성능 향상이 확인되었으며, 이는 NPBML이 모델 용량에 관계없이 범용적으로 작동함을 시사한다.
3. **Ablation Study (구성 요소 분석)**:
    - 옵티마이저($P_\omega$) 추가 시 정확도가 약 2.09% 상승하였고, 메타 손실 함수($M_\phi$) 추가 시 6.37% 상승하였다.
    - 옵티마이저와 손실 함수를 동시에 적용했을 때 시너지 효과가 발생하여 7.41% 상승하였으며, 여기에 FiLM 기반 태스크 적응성을 추가했을 때 최종적으로 9.63%의 가장 높은 성능 향상을 보였다.
    - 메타 손실 함수의 세부 구성 요소($L_S, L_Q, R$) 각각이 단독으로도 약 5%의 성능 향상을 가져오지만, 함께 사용했을 때 더 나은 성능을 보였다.

## 🧠 Insights & Discussion

**강점 및 시사점:**

- **상보적 관계 확인**: 옵티마이저 학습과 손실 함수 학습이 서로 독립적이면서도 상보적인(complementary and orthogonal) 관계에 있음을 실험적으로 증명하였다.
- **암묵적 학습(Implicit Learning)**: 본 논문은 명시적으로 하이퍼파라미터를 학습시키지 않았음에도, 메타 학습된 손실 함수와 옵티마이저가 수학적으로 학습률 스케줄링, 층별 학습률(layer-wise LR), 가중치 감쇠(weight decay) 및 레이블 스무딩(label smoothing) 효과를 암묵적으로 구현하고 있음을 이론적으로 논의하였다.

**한계 및 비판적 해석:**

- **메타 오버피팅(Meta-overfitting)**: 규모가 작은 데이터셋에서 메타 오버피팅이 발생할 수 있음을 언급하였다. 저자들은 이를 해결하기 위해 정규화 기법을 사용하거나 옵티마이저 $\omega$의 표현력을 낮추는 방법을 제안하였다.
- **계산 복잡도**: 초기화, 옵티마이저, 손실 함수를 모두 학습하고 FiLM 층을 다수 추가함에 따라 학습해야 할 메타 파라미터의 수가 상당히 증가하였다. 다만, 인코더의 앞부분을 동결(freeze)함으로써 메모리 사용량을 줄인 점은 실용적인 절충안으로 보인다.

## 📌 TL;DR

본 논문은 Few-shot Learning을 위해 **초기화, 옵티마이저, 손실 함수라는 세 가지 절차적 편향(Procedural Biases)을 동시에 메타 학습하고, 이를 FiLM을 통해 태스크별로 적응시키는 NPBML 프레임워크**를 제안한다. 실험 결과, 이 방식은 기존의 고정된 학습 규칙을 사용하는 MAML 계열 모델들보다 월등한 성능을 보였으며, 학습률 및 규제화와 같은 최적화 하이퍼파라미터들을 암묵적으로 학습하는 효과를 가진다. 이 연구는 향후 일반 목적의 메타 학습 알고리즘 설계에 있어 학습 규칙 자체를 유연하게 설계해야 한다는 중요한 방향성을 제시한다.
