# Domain-Agnostic Clustering with Self-Distillation

Mohammed Adnan, Yani A. Ioannou, Chuan-Yung Tsai, Graham W. Taylor (2021)

## 🧩 Problem to Solve

최근 Self-Supervised Learning (SSL)의 발전으로 지도 학습과 비지도 학습 간의 표현 학습(Representation Learning) 격차가 줄어들었다. 하지만 대부분의 SSL 및 Deep Clustering 기법들은 데이터 증강(Data Augmentation)에 크게 의존하고 있다. 데이터 증강은 효과적인 정규화 도구이지만, 이를 설계하기 위해서는 해당 데이터 도메인에 대한 깊은 지식이 필요하다.

예를 들어, 일반적인 이미지 분류에서 사용되는 Color Jittering은 흑백 X-ray 이미지에 적용할 수 없으며, Random Cropping은 관심 영역(Region of Interest)이 매우 작은 조직 병리 이미지(Histopathology images)에 적절하지 않다. 따라서 도메인 지식이 부족하거나 데이터 증강을 적용하기 어려운 환경에서도 작동할 수 있는 Domain-Agnostic(도메인 불가지론적) 클러스터링 및 SSL 알고리즘의 개발이 필수적이다. 본 논문의 목표는 데이터 증강 없이도 일반화 가능한 특징을 학습할 수 있는 self-distillation 기반의 클러스터링 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Knowledge Distillation(KD)을 통해 모델 내부의 'Dark Knowledge'를 추출하여 데이터 증강의 부재로 인한 성능 저하를 보완하는 것이다.

구체적으로, 모델의 깊은 층(Teacher)이 가진 풍부한 세부 정보를 얕은 층(Student)으로 전달하는 Self-Distillation 구조를 Deep Clustering 프레임워크에 통합하였다. 이를 통해 데이터 증강이 제공하던 정규화(Regularization) 효과를 일부 대체하고, 단순한 의사 라벨(Pseudo-labels)만으로는 학습하기 어려운 클래스 간의 세부적인 시맨틱 관계를 학습함으로써 모델의 일반화 성능과 학습 안정성을 높였다.

## 📎 Related Works

### 기존 연구 및 한계

- **Self-Supervised Learning (SSL):** SimCLR, MoCo와 같은 Contrastive Learning 방식은 서로 다른 뷰(View)를 생성하기 위해 강력한 데이터 증강에 의존한다. DeepCluster, SwAV, BYOL 등 Non-contrastive 방식 역시 특징 클러스터링이나 타겟 임베딩 생성을 위해 증강 기법을 정규화 도구로 사용한다.
- **Knowledge Distillation (KD):** 큰 Teacher 모델의 Soft label을 작은 Student 모델이 학습하게 하여 모델을 압축하거나 성능을 높이는 기법이다. Self-distillation은 별도의 모델 없이 단일 네트워크 내에서 깊은 층의 지식을 얕은 층으로 전달한다.
- **Deep Clustering:** DeepCluster-v2는 k-means 클러스터링을 통해 얻은 의사 라벨을 사용하여 네트워크를 반복적으로 학습시키며, Spherical k-means를 통해 초기화 안정성을 높였다.

### 차별점

기존의 Domain-Agnostic 접근법들이 Mixup 노이즈를 추가하거나 생성 모델을 통해 가상 뷰를 만드는 방식을 사용한 반면, 본 논문은 추가적인 데이터 생성 과정 없이 모델 내부의 구조적 변경(Bottleneck branches)과 distillation loss만으로 데이터 증강 없는 환경에서의 성능 향상을 꾀했다는 점이 차별적이다.

## 🛠️ Methodology

### 전체 시스템 구조

본 방법론은 DeepCluster-v2 프레임워크를 기반으로 하며, 백본 네트워크로 ResNet을 사용한다. 기존 ResNet 구조에 세 개의 Bottleneck branch를 추가하고, 각 브랜치 상단에 보조 분류기(Auxiliary Classifier)를 배치하였다.

- **Teacher Model:** 네트워크의 가장 깊은 층에 위치한 분류기 $q_c$가 선생 역할을 수행한다.
- **Student Models:** 중간 단계의 Bottleneck branch에 위치한 분류기 $q_i (i=1, 2, 3)$가 학생 역할을 수행한다.

### 훈련 목표 및 손실 함수

모델은 총 네 가지의 손실 함수를 결합하여 학습하며, 이를 통해 의사 라벨 학습과 지식 전수를 동시에 수행한다.

1. **Pseudo-label Loss ($L_i$):** k-means 클러스터링을 통해 얻은 의사 라벨 $y_k(x)$와 모든 분류기(학생 및 선생)의 출력값 사이의 Cross-entropy loss를 계산한다.
   $$L_i = \sum_{\forall x} y_k(x) \log(q_i(x)) \quad ; i \in \{1, 2, 3, c\}$$

2. **KL Divergence Loss ($L_{KL}$):** 선생 모델($q_c$)의 Softmax 출력값과 학생 모델($q_i$)의 출력값 사이의 차이를 줄여 'Dark Knowledge'를 전달한다.
   $$L_{KL} = \sum_{\forall x} q_c(x) \log \left( \frac{q_c(x)}{q_i(x)} \right)$$

3. **Hint Loss ($L_{hints}$):** Bottleneck feature map $F_i$와 가장 깊은 층의 feature map $F_c$ 사이의 $L_2$ 거리를 최소화하여 명시적인 힌트를 제공한다.
   $$L_{hints} = \|F_i - F_c\|^2$$

### 최종 학습 절차

전체 손실 함수 $L_{total}$은 다음과 같이 정의되며, 가중치 $\alpha$와 $\lambda$를 통해 각 항의 영향력을 조절한다.
$$L_{total} = L_c + (1-\alpha) \sum_{i=1,2,3} L_i + \alpha L_{KL} + \lambda L_{hints}$$
여기서 $L_c$는 선생 모델의 pseudo-label loss이다. 학습 과정에서 k-means를 통한 클러스터링과 네트워크 최적화가 반복적으로 이루어지며, 각 에폭마다 분류기는 재초기화된다.

## 📊 Results

### 실험 설정

- **데이터셋:** CIFAR-10 (훈련 데이터 50,000장 사용, 데이터 증강 미적용)
- **평가 방법:** Linear Evaluation. 학습된 CNN의 가중치를 고정(freeze)하고, 그 위에 선형 분류기를 추가하여 CIFAR-10 테스트 세트(10,000장)에 대한 정확도를 측정한다.
- **비교 대상:** ID, IDFD, ConCURL 등 기존 Domain-Agnostic 방법론 및 데이터 증강을 제거한 DeepCluster-v2.

### 정량적 결과

실험 결과, 제안 방법론(DeepCluster + KD)이 기존의 데이터 증강 없는 방법론들보다 우수한 성능을 보였다.

| Method | Accuracy |
| :--- | :--- |
| ID | 18.7% |
| IDFD | 23.6% |
| ConCURL | 29.88% |
| DeepCluster-v2 | 33.27 $\pm$ 0.06% |
| **DeepCluster + KD (Ours)** | **38.00 $\pm$ 0.34%** |

제안 방법은 DeepCluster-v2 대비 약 4.73%p의 성능 향상을 달성하였다.

### 학습 안정성 및 수렴 속도

훈련 손실(Cross entropy)의 변화를 분석한 결과, 제안 방법이 DeepCluster-v2보다 더 빠르게 수렴하며, 다양한 랜덤 초기화 설정에서도 일관된 학습 곡선을 보여 학습 안정성이 향상되었음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 데이터 증강이 SSL에서 단순한 데이터 확장을 넘어, 모델이 더 평탄한 최솟값(Flat minima)을 찾게 하여 일반화 성능을 높이는 정규화 역할을 한다는 점에 주목하였다. Self-distillation은 이러한 정규화 효과를 소프트 라벨을 통해 모사함으로써, 데이터 증강 없이도 모델의 일반화 능력을 끌어올릴 수 있음을 입증하였다. 특히, k-means의 하드 라벨(Hard label)이 놓칠 수 있는 클래스 간의 유사성(예: 개와 고양이의 유사성)을 소프트 라벨의 'Dark Knowledge'가 보완해준다는 점이 주요 성능 향상의 원인으로 분석된다.

### 한계 및 논의사항

- **데이터셋의 규모:** 본 실험은 CIFAR-10이라는 상대적으로 작은 데이터셋에서 진행되었다. ImageNet과 같은 대규모 데이터셋이나 실제 의료 영상(X-ray 등) 데이터셋에서도 동일한 효과가 나타날지는 추가 검증이 필요하다.
- **하이퍼파라미터 민감도:** $\alpha$와 $\lambda$와 같은 distillation 관련 하이퍼파라미터가 성능에 미치는 영향에 대한 심층적인 분석이 부족하다.
- **계산 복잡도:** 보조 분류기와 추가적인 loss 계산으로 인해 학습 시간이 다소 증가할 수 있으나, 추론 시에는 보조 브랜치를 제거하므로 추론 효율성에는 영향이 없다.

## 📌 TL;DR

본 논문은 데이터 증강(Data Augmentation)을 사용할 수 없는 도메인을 위해, Self-Distillation을 결합한 도메인 불가지론적 클러스터링 알고리즘을 제안한다. ResNet 내부에 Bottleneck branch를 구축하여 깊은 층의 지식을 얕은 층으로 전달함으로써, 데이터 증강의 정규화 효과를 대체하고 학습 안정성을 높였다. CIFAR-10 실험에서 DeepCluster-v2 대비 성능을 33.27%에서 38.00%로 향상시켰으며, 이는 향후 도메인 지식이 부족한 특수 영상 데이터의 비지도 학습 연구에 중요한 기초가 될 수 있다.
