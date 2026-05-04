# Tree-structured Auxiliary Online Knowledge Distillation

Wenye Lin, Yangning Li, Yifeng Ding, Hai-Tao Zheng (2022)

## 🧩 Problem to Solve

전통적인 Knowledge Distillation (KD)은 거대한 Teacher 모델을 먼저 사전 학습시킨 후, 그 지식을 작은 Student 모델로 전달하는 2단계 학습 과정을 거친다. 이러한 방식은 파이프라인의 복잡성을 증가시키고 학습 비용을 높이는 한계가 있다. 이를 해결하기 위해 Teacher 모델 없이 여러 Student 모델이 서로 지식을 주고받으며 동시에 학습하는 Online Knowledge Distillation (OKD)이 제안되었다.

최근의 OKD 연구들은 주로 Attention이나 Gate 메커니즘과 같은 Distillation 목적 함수(Objective)의 설계에 집중해 왔다. 그러나 본 논문은 목적 함수뿐만 아니라 전반적인 네트워크 아키텍처 설계가 Student 모델의 성능에 핵심적인 영향을 미친다는 점에 주목한다. 따라서 본 연구의 목표는 계층적 트리 구조의 보조 네트워크를 도입하여 OKD의 성능을 극대화하는 Tree-Structured Auxiliary online knowledge distillation (TSA) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 네트워크의 출력층에 가까워질수록 병렬적인 피어(Peer) 네트워크를 계층적으로 추가하여 트리 구조를 형성하는 것이다.

- **계층적 구조의 직관**: 네트워크의 앞부분(Early layers)은 일반적인 특징(General features)을 추출하고, 뒷부분(Later layers)으로 갈수록 작업 특화적인 특징(Task-specific features)을 추출한다. TSA는 출력층에 가까운 부분에 더 많은 분기(Branch)를 생성함으로써, 지식이 '일반적인 것'에서 '특정한 것'으로 흐르도록 유도한다.
- **다양한 뷰(View) 제공**: 각 분기는 서로 다른 초기값을 가지므로 입력 데이터에 대해 서로 다른 관점을 형성하며, 이것이 Peer 간 지식 전달의 원천이 된다.
- **범용적 프레임워크**: 특정 모델에 국한되지 않고 다양한 아키텍처에 적용 가능한 통합 프레임워크를 제시하며, 특히 기계 번역(Machine Translation) 분야에 OKD를 처음으로 적용하여 그 효과를 입증하였다.

## 📎 Related Works

- **Knowledge Distillation**: 모델 압축과 가속화를 위해 Teacher의 출력 분포를 Student가 모방하게 하는 기술이다. 중간 표현이나 Attention map 등을 전달하는 방식이 연구되었으나, 여전히 사전 학습된 Teacher가 필요하다는 2단계 학습의 한계가 있다.
- **Deeply Supervised Nets (DSN)**: 중간 레이어에 분류기(Classifier)를 추가하여 직접적인 감독을 가함으로써 변별력 있는 특징 추출을 돕는 방식이다. 하지만 이는 깊은 신경망의 일반적인 특징 $\rightarrow$ 특수 특징 전이 과정을 방해하여 성능을 저하시킬 수 있다.
- **Multi-branch Online KD**: Deep Mutual Learning (DML), ONE, OKDDip 등이 있으며, 여러 네트워크를 동시에 학습시키며 서로 가르치게 한다. 기존 연구들은 주로 앙상블 Teacher를 설계하거나 Attention 메커니즘을 통해 지식의 질을 높이는 데 집중했으나, TSA는 아키텍처 자체의 구조적 설계를 통해 성능을 높인다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
TSA는 대상 네트워크를 깊이에 따라 몇 개의 블록으로 나눈 뒤, 출력층에 가까운 블록들에 복제본(Counterparts)을 추가하여 트리 구조를 형성한다. 학습 시에는 루트(Root)에서 모든 리프(Leaf) 노드로 데이터가 동시에 흐르며, 각 리프에 위치한 분류기들이 서로 지식을 교환한다. 추론(Inference) 단계에서는 모든 보조 모듈을 제거하고 원래의 단일 네트워크 구조만 남기므로, 배포 시 추가적인 연산 비용이 발생하지 않는다.

### 학습 목표 및 손실 함수
각 분기의 분류기 $\Theta_k$ ($k \in \{1, 2, \dots, K\}$)는 다음 두 가지 손실 함수의 합으로 학습된다.

**1. Softmax 함수**
분류기의 Logit $z$를 확률 분포 $p$로 변환하기 위해 Temperature $T$를 적용한 Softmax를 사용한다.
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**2. Joint Loss Function**
- **Cross-Entropy Loss ($\mathcal{L}^C_k$)**: 정답 라벨(Hard label)과 예측값 사이의 오차를 줄여 정확한 분류를 학습한다.
$$\mathcal{L}^C_k = -\sum_{i,t} \delta_{i,t} \log p_{i,t}$$
- **Distillation Loss ($\mathcal{L}^D_k$)**: 다른 Peer들의 예측 분포와 자신의 분포를 일치시키기 위해 Kullback-Leibler (KL) Divergence를 사용한다.
$$\mathcal{L}^D_k = \frac{1}{K-1} \sum_{j \neq k} \text{KL}(p_k || p_j)$$
여기서 $\text{KL}(p_k || p_j)$는 다음과 같이 계산된다.
$$\text{KL}(p_k || p_j) = \sum_{j \neq k} \sum_{i,t} p^t_k(x_i) \log \frac{p^t_k(x_i)}{p^t_j(x_i)}$$

**3. 전체 손실 함수**
최종 손실 함수 $\mathcal{L}$은 모든 분류기 $k$에 대한 손실의 합으로 정의되며, 하이퍼파라미터 $\alpha$를 통해 두 손실의 비중을 조절한다.
$$\mathcal{L} = \sum_k [(1-\alpha) * \mathcal{L}^C_k + \alpha * T^2 * \mathcal{L}^D_k]$$

### 아키텍처 변형
- **Balanced Tree**: 대칭적인 구조를 가지며, 더 강한 정규화(Regularization) 효과를 보여 기본 설정으로 사용된다.
- **Unbalanced Tree**: 비대칭 구조이며, 데이터셋이 매우 클 때(예: ImageNet) Balanced Tree와 유사한 성능을 보인다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-10, CIFAR-100, ImageNet-1K, IWSLT'14, IWSLT'17, WMT'17.
- **대상 모델**: VGG-16, ResNet-18/32/34/110, MobileNetV3, WRN-28-10, Transformer.
- **측정 지표**: Top-1 Accuracy (이미지 분류), BLEU score (기계 번역).

### 주요 결과
- **이미지 분류 (CIFAR-100)**: TSA는 Vanilla 학습 및 기존 OKD 방법(DML, ONE, OKDDip)보다 일관되게 높은 성능을 보였다. 특히 MobileNetV3에서 78.93%의 정확도를 기록하며 기존 보고된 최고 성능을 1.23% 경신하였다.
- **이미지 분류 (ImageNet)**: ResNet-34에서 Vanilla 대비 1.8% 향상된 74.97%의 정확도를 달성하였다. 학습 곡선 분석 결과, TSA는 훈련 정확도는 낮추면서 테스트 정확도를 높이는 정규화 효과를 보였다.
- **기계 번역 (NMT)**: Transformer 모델에 적용한 결과, IWSLT 데이터셋에서 평균 0.9 BLEU 상승을 기록하였다. WMT'17 대규모 데이터셋에서는 학습 과정의 변동성(Fluctuation)을 줄여 더욱 안정적인 학습 곡선을 보여주었다.
- **효율성**: 학습 시에는 파라미터와 시간이 약간 증가하지만, 추론 시에는 보조 분기를 제거하므로 baseline과 완전히 동일한 비용으로 더 높은 성능을 낼 수 있다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 OKD의 성능 향상을 위해 목적 함수라는 소프트웨어적 접근 대신 아키텍처라는 하드웨어적 접근이 유효함을 증명하였다. 특히 트리 구조를 통해 하위 레이어는 공유하고 상위 레이어만 확장함으로써, 연산 효율성을 챙기면서도 최적화 지형(Optimization landscape)에서 더 넓은 최소값(Wider minimum)을 찾게 하여 일반화 성능을 높였다.

### 한계 및 논의사항
- **이론적 근거**: 저자들은 지식 증류가 왜 작동하는지에 대한 이론적 논의가 여전히 초기 단계임을 인정하며, 트리 구조의 효과에 대해 '일반적 특징 $\rightarrow$ 특수 특징'의 전이라는 가설(Conjecture)을 제시하였다. 이는 엄밀한 수학적 증명보다는 실험적 관찰에 기반한 해석이다.
- **하이퍼파라미터**: Weight decay 등의 하이퍼파라미터에 대해 어느 정도 강건함(Robustness)을 보였으나, 최적의 설정을 찾기 위한 추가적인 튜닝이 성능을 더 높일 수 있음을 명시하였다.

## 📌 TL;DR

본 논문은 Online Knowledge Distillation 과정에서 네트워크를 계층적 트리 구조로 설계하여 Peer 간의 지식 전달 효율을 극대화하는 **TSA (Tree-Structured Auxiliary online knowledge distillation)** 프레임워크를 제안한다. 단순한 손실 함수 사용만으로도 아키텍처 설계의 최적화를 통해 컴퓨터 비전(CIFAR, ImageNet)과 자연어 처리(NMT) 분야에서 SOTA 수준의 성능 향상을 이루었으며, 특히 추론 시에는 추가 비용 없이 성능 이점만 챙길 수 있다는 점에서 실용성이 매우 높다. 향후 연구에서 고도화된 학습 목적 함수와 결합한다면 더 큰 성능 향상이 기대된다.