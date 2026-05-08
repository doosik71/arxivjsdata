# FEW-SHOT-INSPIRED GENERATIVE ZERO-SHOT LEARNING

Md Shakil Ahamed Shohag, Q. M. Jonathan Wu, Farhad Pourpanah (2025)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL)에서 기존 Generative 방법론들이 가진 과도한 계산 비용과 데이터 의존성 문제를 해결하고자 한다. 일반적인 Generative ZSL 방식은 사전 정의된 semantic attributes를 이용하여 unseen class에 대한 가상 visual features를 대량으로 합성한 뒤, 이를 통해 완전 감독 학습(fully supervised learning) 기반의 분류기를 훈련시킨다. 그러나 이러한 접근 방식은 ZSL의 본래 가정(unseen 데이터에 대한 접근 제한)을 과도하게 완화하며, 실제 분류에 얼마나 유용한지에 대한 평가 없이 너무 많은 양의 데이터를 생성한다는 한계가 있다.

또한, 기존 방법론들은 클래스 수준의 속성(class-level attributes)이 모든 인스턴스에 동일하게 적용된다고 가정한다. 하지만 실제 이미지에서는 특정 속성이 일부만 보이거나 아예 나타나지 않는 instance-level variability(인스턴스 수준의 가변성)가 존재하며, 이를 간과하는 것은 성능 저하의 원인이 된다. 따라서 본 연구의 목표는 Few-Shot Learning (FSL)에서 영감을 얻어, 대규모 합성 데이터 없이도 소수의 핵심적인 prototype만을 생성하여 효율적으로 ZSL을 수행하는 프레임워크인 FSIGenZ를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 unseen class의 visual features를 대량으로 생성하는 대신, 인스턴스 수준의 가변성을 반영한 소수의 '그룹 수준 prototype'만을 생성하여 학습에 활용하는 것이다. 이를 위해 다음과 같은 설계 아이디어를 제시한다.

1. **Model-Specific Attribute Scoring (MSAS)**: 클래스 수준의 고정된 속성 값을 모델 최적화 관점에서 동적으로 재점수화하여 인스턴스별 가변성을 근사한다.
2. **Group-level Prototype Synthesis**: MSAS로 조정된 속성 값과 Sparse Coding의 정규화 파라미터 $\lambda$의 변화를 통해, 클래스 내의 다양한 하위 그룹을 대표하는 소수의 prototype을 생성한다.
3. **Semantic-Aware Contrastive Classifier (SCC) 및 DPSR**: 합성된 데이터의 양이 매우 적어 발생하는 클래스 불균형 문제를 해결하기 위해, 시각-의미 대조 학습(visual-semantic contrastive learning)과 Dual-Purpose Semantic Regularization (DPSR) 전략을 도입한다.

## 📎 Related Works

ZSL 연구는 크게 Embedding 기반 방법과 Generative 기반 방법으로 나뉜다.

- **Embedding 방법**: 시각적 특성과 의미적 특성을 공유 공간으로 매핑하여 유사도를 측정한다. 하지만 GZSL(Generalized ZSL) 설정에서 seen class로 예측이 쏠리는 bias 문제가 심각하다.
- **Generative 방법**: GAN이나 VAE를 사용하여 unseen class의 특징을 합성함으로써 ZSL을 감독 학습 문제로 전환한다. 이는 seen-class bias를 줄이는 데 효과적이지만, 계산 비용이 높고 Mode Collapse와 같은 GAN 특유의 문제와 대량의 합성 데이터 생성으로 인한 ZSL 가정의 완화 문제가 존재한다.

FSIGenZ는 비적대적(non-adversarial) 합성 전략을 취하며, 기존의 대규모 데이터 생성 방식과 달리 FSL의 관점에서 소수의 대표 prototype만을 추출한다는 점에서 기존 Generative 방법론과 차별화된다.

## 🛠️ Methodology

FSIGenZ는 크게 특징 생성(Feature Generation) 단계와 대조 학습(Contrastive Learning) 단계의 두 페이즈로 구성된다.

### 1. Model-Specific Attribute Scoring (MSAS)

인스턴스 수준의 속성 가변성을 모델링하기 위해, 고정된 속성 행렬 $A^o$를 다음과 같이 동적으로 재점수화한다.

$$A = (A^o + A^{mdf})W_A$$

여기서 $A^{mdf}$는 임계값 $T_h$를 이용한 이진 마스크를 통해 계산된다.
$$A^{mdf} = A^o \odot (A^o > T_h)$$
이 과정을 통해 모델은 하이퍼파라미터 $W_A$와 $T_h$를 조정하여 해당 모델에 최적화된 속성 값을 가질 수 있게 된다.

### 2. Unseen Data Synthesis

본 논문은 Sparse Coding을 통해 semantic manifold를 visual feature space로 전이한다. 먼저 다음과 같은 최적화 문제를 통해 희소 계수 벡터 $\alpha$를 구한다.

$$\min_{\alpha} \|a^u_c - A^s \alpha\|_2^2 + \lambda \|\alpha\|_2^2$$

여기서 $a^u_c$는 unseen 클래스의 속성, $A^s$는 seen 클래스의 속성 행렬이다. 이후 unseen 클래스의 클러스터 중심 $\mu^u_k$를 다음과 같이 추정한다.
$$\mu^u_k = M^s \alpha$$
이때, 정규화 파라미터 $\lambda$의 값을 다르게 설정함으로써 서로 다른 $\alpha$ 값이 생성되며, 결과적으로 한 클래스 내에서도 서로 다른 특성을 가진 여러 개의 prototype(시각적 프로토타입)을 얻을 수 있다.

### 3. Semantic-Aware Contrastive Classifier (SCC)

생성된 prototype과 실제 seen feature를 결합하여 학습 세트를 구성한다. 인스턴스 특징 $F(x_i)$와 클래스 의미 임베딩 $E(a_j)$를 요소별 곱(element-wise multiplication)으로 융합하여 $Z_{ij}$를 생성한다.

$$Z_{ij} = F(x_i) \otimes E(a_j)$$

최종 대조 점수 $c_{ij} = f(Z_{ij})$를 계산하며, 전체 손실 함수는 다음과 같이 정의된다.
$$L = L_S + \beta L_U$$
여기서 $L_S$는 seen class에 대한 Binary Cross-Entropy (BCE) 손실이며, $L_U$는 unseen class에 대한 정규화된 BCE 손실이다.

### 4. Dual-Purpose Semantic Regularization (DPSR)

데이터 불균형 문제를 해결하기 위해 클래스 간 의미적 유사도 $s_{pq}$를 활용하는 DPSR을 도입한다.
$$s_p = \arg \min_{s_p} \left\| a_p - \sum_{q=1}^{K+L} a_q s_{pq} \right\|_2^2 + \phi \|s_p\|_2$$
DPSR은 두 가지 역할을 수행한다. 첫째, seen instance에 대해 의미적으로 유사한 unseen class에 적절한 활성화를 유도하여 전이 성능을 높인다. 둘째, 합성된 unseen instance에 대해 결정 경계를 너무 날카롭게 잡지 않도록 손실 신호를 완화(soften)하여 일반화 성능을 향상시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: SUN, AwA2, CUB 세 가지 벤치마크를 사용하였다.
- **지표**: Conventional ZSL(CZSL)에서는 Top-1 Accuracy를, Generalized ZSL(GZSL)에서는 seen/unseen accuracy의 조화 평균인 Harmonic Mean ($H$)을 측정하였다.
- **백본**: ImageNet-1k로 사전 학습된 ViT-Base를 사용하여 786차원 특징을 추출하였다.

### 주요 결과

- **성능**: Table 1에 따르면 FSIGenZ는 SUN과 AwA2의 CZSL에서 최고 성능을 기록하였으며, GZSL에서도 AwA2(74.2%)와 CUB(69.1%)에서 매우 경쟁력 있는 성능을 보였다.
- **효율성**: 가장 주목할 점은 합성 데이터의 양이다. Table 2에서 확인되듯, 기존 Generative 모델들이 클래스당 수천 개의 특징을 생성하는 반면, FSIGenZ는 클래스당 단 10~90개의 prototype만을 사용하고도 동등하거나 더 높은 성능을 달성하였다.
- **Ablation Study**: MSAS와 DPSR을 모두 제거했을 때 성능이 급격히 하락하였으며, 특히 DPSR 제거 시 GZSL 성능이 크게 낮아져 일반화 성능 유지에 핵심적임을 확인하였다.
- **시각화**: t-SNE 분석 결과, FSIGenZ가 생성한 prototype(cross)이 실제 이미지 특징의 클러스터 중심(circle)과 잘 일치함을 확인하여, 소수의 prototype이 클래스의 내부 구조를 잘 대표하고 있음을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 ZSL을 "완전 감독 학습"의 관점이 아닌 "Few-Shot Learning"의 관점에서 재해석함으로써, 불필요하게 많은 합성 데이터를 생성하던 기존의 비효율성을 제거하였다.

**강점**:

1. 계산 비용을 획기적으로 줄이면서도 성능을 유지하거나 향상시켰다.
2. 클래스 수준의 속성을 그대로 쓰지 않고 MSAS를 통해 인스턴스 수준의 가변성을 모델링하려 시도한 점이 논리적으로 타당하다.
3. DPSR을 통해 학습 단계에서 직접 semantic similarity를 통합함으로써, 추론 단계에서 post-hoc으로 조정하는 기존 방식보다 더 안정적인 결정 경계를 학습하였다.

**한계 및 논의**:

- MSAS의 하이퍼파라미터($W_A, T_h$)가 데이터셋마다 수동으로 조정되어야 한다는 점이 한계로 보이며, 이를 자동화하는 방안에 대한 논의가 필요하다.
- 저자들은 향후 연구로 fine-grained 도메인에서의 성능 향상을 위한 adaptive subgroup modeling을 언급하였다. 이는 현재의 $\lambda$ 기반 prototype 생성보다 더 정교한 분포 모델링이 필요함을 시사한다.

## 📌 TL;DR

FSIGenZ는 대규모 데이터 합성에 의존하던 기존 Generative ZSL의 패러다임을 깨고, 소수의 핵심 prototype만을 생성하여 학습하는 Few-Shot-inspired 프레임워크이다. MSAS를 통한 속성 재점수화, $\lambda$ 조절을 통한 하위 그룹 prototype 생성, 그리고 DPSR 기반의 대조 학습을 통해 매우 적은 양의 합성 데이터만으로도 최신 SOTA 모델들과 경쟁 가능한 성능을 달성하였다. 이는 ZSL 연구가 단순히 데이터를 많이 만드는 방향이 아니라, 데이터의 질과 의미적 구조를 효율적으로 포착하는 방향으로 나아가야 함을 보여준다.
