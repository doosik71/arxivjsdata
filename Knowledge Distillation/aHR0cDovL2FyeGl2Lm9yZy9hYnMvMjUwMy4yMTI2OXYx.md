# Delving Deep into Semantic Relation Distillation

Zhaoyi Yan, Kangjun Liu, Qixiang Ye (2025)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델 압축의 핵심 기술인 Knowledge Distillation (KD)에서 기존 방식들이 가진 한계를 해결하고자 한다. 전통적인 KD 방식들은 주로 인스턴스 수준(instance-level)에서 지식을 전달하는 데 집중한다. 구체적으로는 Teacher 모델과 Student 모델 간의 출력 확률(logits)을 맞추는 logit-based 방식이나, 중간 특징 맵(intermediate features)을 일치시키는 feature-based 방식이 주를 이룬다.

이러한 방식들의 문제점은 데이터 내에 존재하는 정교하고 미묘한 **시맨틱 관계(semantic relationships)**를 포착하지 못한다는 점이다. 특히 Vision Transformer (ViT)와 같이 토큰 기반의 표현 방식을 사용하는 최신 아키텍처에서는 단순히 개별 토큰의 값을 맞추는 것보다, 토큰들 사이의 구조적이고 의미론적인 관계를 전달하는 것이 모델의 성능과 일반화 능력을 높이는 데 훨씬 중요하다. 따라서 본 논문의 목표는 superpixel을 이용해 시맨틱 구성 요소를 추출하고, 이를 기반으로 관계 지식을 증류하는 **Semantics-based Relation Knowledge Distillation (SeRKD)** 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 지식 증류의 관점을 인스턴스 수준에서 **시맨틱-관계(semantics-relation)** 관점으로 전환하는 것이다. 단순히 Teacher의 특징값을 모방하는 것이 아니라, 이미지 내의 의미 있는 영역인 superpixel을 정의하고, 이 superpixel 토큰들 사이의 관계 구조를 Student 모델이 학습하게 함으로써 더 풍부한 문맥 정보를 전달한다.

주요 기여 사항은 다음과 같다:

1. **시맨틱 중심의 KD 접근법 제안**: 단순한 값의 일치를 넘어 데이터의 시맨틱 구조를 보존하는 새로운 증류 패러다임을 제시하였다.
2. **SeRKD 프레임워크 설계**: superpixel 추출 기술과 Relation-based KD를 결합하여, ViT뿐만 아니라 CNN 기반 모델에도 적용 가능한 범용적인 모델 압축 프레임워크를 구축하였다.
3. **실증적 성능 검증**: ImageNet-1k 및 다양한 다운스트림 데이터셋에서 기존의 DeiT, DearKD, CSKD 등 최신 방법론보다 우수한 성능을 입증하였다.

## 📎 Related Works

### 1. Knowledge Distillation (KD)

Hinton 등의 초기 연구 이후, KD는 Logit-based, Feature-based, 그리고 Relation-based 접근법으로 발전하였다. 특히 Relation Knowledge Distillation (RKD)는 배치 내 샘플들 간의 상관관계를 학습하여 구조적 정보를 전달한다. 하지만 기존 RKD는 주로 CNN에 최적화되어 있으며, ViT와 같은 트랜스포머 기반 모델에 직접 적용했을 때의 효율성에 대한 연구가 부족했다.

### 2. Vision Transformer (ViT) KD

DeiT는 CNN에서 ViT로의 지식 전달을 제안했고, DearKD나 CSKD는 중간 특징이나 공간적 지식을 전달하는 방식을 사용했다. 그러나 이들 역시 '시맨틱' 관점에서의 관계 증류보다는 특징 맵의 정렬(alignment)에 집중하는 경향이 있다.

### 3. Superpixel Methods

이미지를 의미 있는 작은 영역으로 묶는 superpixel 알고리즘은 그래프 기반 및 클러스터링 기반으로 나뉜다. 최근에는 SSN과 같은 미분 가능한 superpixel 생성 네트워크가 등장하여 딥러닝 파이프라인에 통합될 수 있게 되었다. 본 논문은 이러한 superpixel 기술을 활용해 ViT의 패치 토큰들을 시맨틱 토큰으로 그룹화한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

SeRKD의 전체 과정은 크게 **Superpixel Token 구축 $\rightarrow$ 관계 지식 추출 $\rightarrow$ 손실 함수 최적화** 단계로 이루어진다.

### 2. Superpixel Token 구축

- **ViT의 경우**: 입력 이미지가 ViT를 통해 패치 토큰 $\{T_i\}$로 변환되면, Attention과 유사한 방식으로 연관성 맵(association map) $Q^t$를 계산한다.
  $$Q^t = \text{Softmax} \left( \frac{T S_{t-1}^T}{\sqrt{d}} \right)$$
  여기서 $S^t$는 업데이트된 superpixel 토큰이며, 열 정규화된 $\hat{Q}^t$를 사용하여 다음과 같이 계산된다.
  $$S^t = (\hat{Q}^t)^T T$$
- **CNN의 경우**: CNN 특징 맵 $F$는 구조화된 토큰이 아니므로, Tokenize 연산을 통해 이산적인 토큰 $P$로 변환한 후 위와 동일한 superpixel 과정을 거친다. 이때 Max-pooling, Average-pooling 또는 Strided-convolution이 사용된다.

### 3. Semantics-based RKD Loss

구축된 superpixel 토큰들을 대상으로 Teacher($s'$)와 Student($s$) 간의 관계를 맞춘다.

- **Distance-wise Loss ($L_{SP}^{RD}$)**: 두 superpixel 토큰 쌍 사이의 유클리드 거리를 정규화하여 일치시킨다.
  $$L_{SP}^{RD} = \frac{1}{\nu'} \sum_{i,j} l_\delta (\psi_D(s_i, s_j), \psi_D(s'_i, s'_j))$$
  여기서 $\psi_D$는 정규화된 유클리드 거리이며, $l_\delta$는 Huber loss이다.
- **Angle-wise Loss ($L_{SP}^{RA}$)**: 세 개의 superpixel 토큰이 이루는 각도(코사인 유사도)를 통해 고차원적인 구조 정보를 일치시킨다.
  $$L_{SP}^{RA} = \sum_{i,j,k} l_\delta (\psi_A(s_i, s_j, s_k), \psi_A(s'_i, s'_j, s'_k))$$

### 4. 최종 학습 목표

최종 손실 함수 $L_{dis}$는 분류 손실, 전통적인 KD 손실, 특징 일치 손실, 그리고 제안하는 superpixel 기반 관계 손실들의 가중 합으로 구성된다.
$$L_{dis} = L_{cls} + \lambda_K L_{KD} + \lambda_F L_F + \lambda_D L_{SP}^{RD} + \lambda_A L_{SP}^{RA}$$

## 📊 Results

### 1. 실험 설정

- **Teacher 모델**: MAE-base (ImageNet-1k Top-1 Acc: 83.6%)
- **Student 모델**: DeiT-Tiny (SeRKD-Ti), DeiT-Small (SeRKD-S)
- **데이터셋**: ImageNet-1k, CIFAR10, CIFAR100, Stanford Cars

### 2. 주요 결과

- **ImageNet-1k 성능**: SeRKD-Ti는 76.8%, SeRKD-S는 82.5%의 Top-1 정확도를 달성하여, CSKD(76.3% / 82.3%) 및 DeiT(74.5% / 81.2%)보다 우수한 성능을 보였다.
- **CNN 모델 적용**: ResNet-101(Teacher) $\rightarrow$ ResNet-18(Student) 증류 실험에서, Strided Convolution이나 Average Pooling을 Tokenizer로 사용한 SeRKD가 전통적인 KD나 RKD보다 훨씬 높은 성능(최대 72.25%)을 기록하였다.
- **전이 학습(Transfer Learning)**: CIFAR-100 및 Cars 데이터셋에서 CSKD-S 대비 각각 1.0%, 0.4% 더 높은 성능 향상을 보이며 강한 일반화 능력을 입증하였다.

### 3. Ablation Study

- **구성 요소 영향**: Superpixel clustering과 $L_{SP}^{RD}$, $L_{SP}^{RA}$를 모두 포함했을 때 가장 높은 성능(76.8%)이 나타났다.
- **토큰 병합 방법**: 단순 Pooling 기반 병합보다 Superpixel 기반 병합이 baseline 대비 2.3% 높은 정확도를 보였다. 이는 단순한 공간적 결합보다 시맨틱 레이아웃을 반영한 결합이 더 효과적임을 의미한다.
- **그리드 크기**: $2 \times 2$ 그리드 설정이 국소 문맥 포착과 계산 효율성 사이의 최적의 균형을 이루어 가장 좋은 성능을 냈다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 ViT의 수많은 패치 토큰들에 직접 RKD를 적용했을 때 발생하는 성능 저하와 메모리 문제(OOM)를 **superpixel을 통한 토큰 응축**으로 해결하였다. 단순히 토큰 수를 줄인 것이 아니라, 이미지의 의미론적 경계를 보존하며 응축했기 때문에 Teacher 모델이 가진 고차원적인 시맨틱 구조를 Student 모델에게 효율적으로 전달할 수 있었다. 시각화 결과에서도 Teacher와 Student의 superpixel 맵이 유사한 영역을 공유하고 있음이 확인되어, 제안 방법이 시맨틱 레이아웃 정렬에 효과적임을 알 수 있다.

### 한계 및 논의사항

- **계산 복잡도**: Angle-wise loss($L_{SP}^{RA}$)의 공간 복잡도가 $O(BL^3)$으로 매우 높다. 비록 superpixel을 통해 $L$을 49로 줄여 해결했지만, 더 세밀한 superpixel을 사용하고자 할 때는 여전히 메모리 제약이 클 것으로 보인다.
- **하이퍼파라미터 의존성**: $\lambda_D, \lambda_A$ 및 그리드 크기 $H_t \times W_t$에 따라 성능 차이가 발생하므로, 다양한 모델 구조에 적용할 때 최적의 값을 찾는 과정이 필수적이다.
- **반복 횟수 $T$**: 반복 횟수를 늘리는 것이 성능 향상으로 이어지지 않고 오히려 feature over-smoothing을 유발할 수 있다는 점은 흥미로운 지점이며, 이는 시맨틱 추출 단계에서의 적절한 정지 시점이 중요함을 시사한다.

## 📌 TL;DR

본 논문은 ViT 및 CNN 모델의 압축을 위해 **superpixel 기반의 시맨틱 관계 증류 방법론인 SeRKD**를 제안한다. 기존의 인스턴스 수준 KD와 달리, 이미지를 의미 있는 단위인 superpixel로 그룹화하고 이들 간의 거리 및 각도 관계를 증류함으로써 모델의 시맨틱 이해도를 높였다. 실험적으로 ImageNet-1k 및 다양한 downstream task에서 기존 SOTA KD 방법론들을 능가하는 성능을 보였으며, 특히 ViT의 토큰 구조에 최적화된 지식 전달 방식을 제시하였다. 이는 향후 대형 비전 모델의 효율적인 경량화 및 지식 전이에 중요한 기여를 할 것으로 평가된다.
