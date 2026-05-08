# Training few-shot classification via the perspective of minibatch and pretraining

Meiyu Huang, Xueshuang Xiang, and Yao Xu (2020)

## 🧩 Problem to Solve

본 논문은 Few-shot classification(FSC) 모델의 학습 효율성과 수렴 속도를 개선하고자 한다. 일반적인 FSC 연구에서는 Meta-learning의 일환으로 Episodic training 프레임워크를 사용하는데, 이는 테스트 환경과 유사하게 작은 Support set과 Query set으로 구성된 '에피소드'를 구성하여 학습하는 방식이다.

그러나 기존의 많은 방법론들은 매 반복(iteration)마다 단 하나의 에피소드만을 사용하여 가중치를 업데이트한다. 이는 일반적인 지도 학습(Supervised Learning) 관점에서 볼 때 배치 크기(batch size)를 1로 설정하고 학습하는 것과 같아, 학습 과정이 불안정하거나 수렴 속도가 매우 느릴 수 있다는 문제점이 있다. 따라서 본 연구의 목표는 일반적인 지도 학습에서 검증된 **Minibatch training**과 **Pretraining**의 개념을 FSC에 맞게 재정의하여 적용함으로써, 정확도 손실 없이 학습 속도를 가속화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 **Few-shot classification 문제를 일반적인 지도 학습 문제의 특수한 형태로 정식화(Formalization)** 하는 것이다.

1. **정식화된 평행 구조 제시**: 일반적인 분류 문제의 '클래스-샘플' 관계를 FSC의 '태스크 클래스(Task Class)-태스크 샘플(Task Example)' 관계로 대응시켜, FSC 역시 지도 학습의 관점에서 최적화할 수 있음을 이론적으로 제시하였다.
2. **Multi-episode training**: 일반 분류의 Minibatch training에 대응하는 개념으로, 한 번의 업데이트에 여러 개의 에피소드를 묶어 처리하는 방식을 제안하여 수렴 속도를 높였다.
3. **Cross-way training**: 일반 분류의 Pretraining에 대응하는 개념으로, 타겟 태스크보다 더 높은 Way($\hat{K} > K$)의 문제로 먼저 학습시킨 후 이를 초기값으로 사용하는 전략을 제안하였다.

## 📎 Related Works

논문은 FSC 접근 방식을 크게 세 가지로 분류한다.

- **Optimization-based methods**: MAML, REPTILE 등이 있으며, 새로운 태스크에 빠르게 적응할 수 있도록 모델의 초기 파라미터를 최적화하는 방식이다.
- **Memory-based methods**: MANNs와 같이 외부 메모리 구조를 사용하여 지식을 저장하고 검색하는 방식이다.
- **Metric-based methods**: Siamese Networks, Matching Networks, Prototypical Networks 등이 있으며, 임베딩 공간에서 클래스 간 거리를 통해 분류하는 방식이다.

본 연구는 이 중 **Prototypical Networks**를 기반으로 한다. 특히 Matching Networks가 도입한 Episodic training 프레임워크가 본 논문의 정식화 작업에 영감을 주었으나, 기존 방법들은 단일 에피소드 기반의 업데이트에 의존했다는 점에서 본 연구의 Multi-episode 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. Prototypical Networks 배경

본 연구의 베이스라인인 Prototypical Networks는 임베딩 함수 $f_\phi$를 통해 입력을 특징 공간으로 매핑한다. 각 클래스 $k$의 prototype $c^V_k$는 서포트 세트 $S^V_k$에 속한 샘플들의 임베딩 평균으로 계산된다.
$$c^V_k = \frac{1}{|S^V_k|} \sum_{(x'_i, y'_i) \in S^V_k} f_\phi(x'_i)$$
쿼리 샘플 $x_i$의 클래스 확률은 각 prototype과의 거리 $d$를 기반으로 Softmax 함수를 통해 계산된다.
$$p_\phi(y=V_k | S^V, x_i) = \frac{\exp(-d(f_\phi(x_i), c^V_k))}{\sum_{k'} \exp(-d(f_\phi(x_i), c^{V}_{k'}))}$$

### 2. 일반 분류와 FSC의 정식화된 평행 구조

저자들은 일반 분류 문제와 FSC 문제를 다음과 같이 대응시킨다.

- **일반 분류**: 데이터셋 $D$에서 샘플 $x_i$를 뽑아 클래스 $y_i$를 예측한다.
- **FSC**: '태스크 클래스 $V$'(K개의 클래스 조합)를 정의하고, 이에 해당하는 '태스크 샘플 $\tau_i$'(서포트 세트 $S^V$와 쿼리 샘플 $x_i$의 쌍)를 통해 학습한다.
결과적으로 FSC는 "태스크를 처리하는 능력(Tasker)"을 배우는 지도 학습 문제로 치환된다.

### 3. Multi-episode Training

일반적인 SGD 업데이트 식은 다음과 같다.
$$\phi_{t+1} = \phi_t - \alpha \sum_{(s_i, y_i) \in B_t} \frac{\partial l(f_\phi; s_i, y_i)}{\partial \phi} \bigg|_{\phi=\phi_t}$$
여기서 $B_t$는 미니배치이다. 저자들은 FSC에서도 이와 동일하게 $E$개의 에피소드를 모아 하나의 미니배치 $B_t := \cup_{e=1}^E B^e_t$를 구성하여 업데이트하도록 제안한다. 이는 단일 에피소드 사용 시 발생하는 태스크 샘플링의 불균형 문제를 완화하고 최적화 과정을 더 견고하게 만든다.

### 4. Cross-way Training

일반적인 Pretraining이 더 큰 데이터셋에서 학습하는 것처럼, FSC에서는 더 복잡한 문제(Higher-way)로 먼저 학습하는 전략을 취한다.

- **절차**: 타겟 태스크가 $K$-way일 때, $\hat{K} > K$인 $\hat{K}$-way 문제로 모델을 먼저 학습시켜 초기 파라미터 $\phi_0$를 얻는다.
- **직관**: 더 높은 Way의 학습은 더 보편적인(universal) 특징 표현을 생성하며, 이는 이후 낮은 Way의 타겟 문제로 전이되었을 때 수렴 속도를 획기적으로 높인다.

## 📊 Results

### 실험 설정

- **데이터셋**: Omniglot, miniImageNet
- **베이스라인**: Prototypical Networks (Euclidean 및 Cosine 거리 사용)
- **지표**: 테스트 정확도(Accuracy) 및 수렴까지 걸리는 반복 횟수(Iterations)
- **설정**: Multi-episode는 $E \in \{1, 3, 5\}$를 비교하였으며, Cross-way는 $\hat{K} \in \{20, 30, 60\}$ 등을 사용하였다.

### 주요 결과

1. **Multi-episode 효과**:
    - 에피소드 수 $E$가 증가할수록 대부분의 태스크에서 수렴 속도가 빨라졌다. 특히 Omniglot 5-way 1-shot의 경우 $E=5$일 때 $E=1$보다 수렴 횟수가 절반 이하로 감소하였다.
    - 정확도 측면에서도 $E=5$가 가장 우수한 성능을 보이는 경향이 있었으며, 이는 Hessian spectrum 분석을 통해 모델이 더 평평한 최솟값(flatter minima)에 도달하여 강건함이 향상되었기 때문임이 확인되었다.
2. **Cross-way 효과**:
    - Pretraining을 적용한 경우, 타겟 문제 학습 시 수렴 속도가 극적으로 향상되었다. 모든 모델이 20,000번의 iteration 이내에 수렴하였으며, 이는 baseline보다 훨씬 빠른 수치이다.
    - 기존 Prototypical Networks 논문에서 "Higher-way가 항상 더 높은 정확도를 가진다"고 주장한 바 있으나, 본 논문에서는 학습률 감소 정책(learning rate decaying policy)을 느리게 설정했을 때 Lower-way 모델이 Higher-way 모델의 성능을 따라잡거나 추월할 수 있음을 밝혀, 기존 주장이 편향되었음을 지적하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 FSC를 meta-learning이라는 특수한 틀에서만 보지 않고, 일반적인 지도 학습의 관점에서 재해석함으로써 딥러닝의 표준적인 최적화 기법(Minibatch, Pretraining)을 FSC에 성공적으로 이식하였다. 특히 Hessian spectrum 분석을 통해 Multi-episode training이 단순한 속도 향상을 넘어 최적화 지점의 기하학적 특성을 개선하여 일반화 성능을 높일 수 있음을 이론적으로 뒷받침하였다.

### 한계 및 논의사항

- **과적합 문제**: Omniglot 데이터셋에서 Cross-way training 후 Fine-tuning을 진행했을 때 오히려 성능이 하락하는 경우가 관찰되었는데, 이는 소규모 데이터셋에 대한 과적합(overfitting) 가능성을 시사한다.
- **범용성**: 본 연구는 Prototypical Networks에 국한되어 검증되었으므로, MAML과 같은 Optimization-based 방식이나 다른 Metric-based 모델에서도 동일한 효과가 나타나는지에 대한 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 Few-shot classification을 일반적인 지도 학습 문제로 정식화하여, **Multi-episode training**(미니배치 대응)과 **Cross-way training**(사전학습 대응)이라는 두 가지 학습 전략을 제안하였다. 실험 결과, 이 방법들은 정확도 손실 없이 학습 수렴 속도를 획기적으로 가속화하며 일부 태스크에서는 정확도와 강건성을 향상시켰다. 이는 향후 FSC 모델의 학습 효율성을 높이는 표준적인 훈련 전략으로 활용될 가능성이 크다.
