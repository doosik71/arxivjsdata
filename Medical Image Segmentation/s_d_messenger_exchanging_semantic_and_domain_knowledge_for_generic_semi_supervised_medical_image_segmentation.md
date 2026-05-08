# S&D Messenger: Exchanging Semantic and Domain Knowledge for Generic Semi-Supervised Medical Image Segmentation

Qixiang Zhang, Haonan Wang, and Xiaomeng Li (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 레이블링 비용 문제를 해결하기 위한 Semi-supervised Medical Image Segmentation (SSMIS)을 다룬다. 특히, 실제 의료 데이터셋에서는 도메인 간의 변동성(Domain Variations)이 빈번하게 발생하며, 이에 따라 다음과 같은 세 가지 파생 시나리오가 존재한다.

1. **SSMIS**: 단일 도메인 내에서 소량의 레이블 데이터와 다량의 레이블 없는 데이터를 사용하는 경우.
2. **Unsupervised Medical Domain Adaptation (UMDA)**: 소스 도메인에는 레이블이 있지만, 타겟 도메인에는 레이블이 전혀 없는 상태에서 타겟 도메인에 적응시키는 경우.
3. **Semi-supervised Medical Domain Generalization (Semi-MDG)**: 여러 도메인에서 일부 레이블 데이터를 얻고, 학습 시 보지 못한 새로운 도메인(Unseen Domain)에 대해 일반화 성능을 확보하는 경우.

기존의 Generic SSL 프레임워크나 일반적인 SSMIS 방법론들은 이 세 가지 작업을 동시에 해결하려 했으나, 도메인 시프트(Domain Shift)가 발생하는 UMDA와 Semi-MDG 작업에서는 성능 향상이 미미하거나 오히려 성능이 하락하는 한계를 보였다.

논문은 그 원인을 **Semantic Knowledge(의미론적 지식)**와 **Domain Knowledge(도메인 지식)**의 불균형에서 찾는다. 정밀한 분할 성능을 위한 Semantic Knowledge는 레이블 데이터셋에만 존재하고, 일반화 성능을 위한 풍부한 Domain Knowledge는 레이블 없는 데이터셋에만 존재한다. 기존 방법들은 두 데이터셋의 학습 흐름(Learning Flow)을 분리하여 처리함으로써, 모델이 한 종류의 지식만을 우선적으로 학습하게 하여 결국 도메인 일반화 능력이 저하되는 문제를 야기한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 레이블 데이터셋과 레이블 없는 데이터셋 사이의 장벽을 허물고 두 지식을 직접적으로 교환하게 하는 **S&D Messenger (Semantic & Domain Knowledge Messenger)**를 설계하는 것이다.

중심적인 직관은 레이블 데이터의 정밀한 감독 신호를 레이블 없는 데이터 흐름에 전달하여 수렴을 돕고, 반대로 레이블 없는 데이터가 가진 다양한 도메인 패턴을 레이블 데이터 흐름에 전달하여 오버피팅을 방지하고 일반화 성능을 높이는 것이다. 이를 통해 단일 모델이 두 가지 상보적인 지식을 동시에 습득할 수 있도록 한다.

## 📎 Related Works

### 1. Semi-supervised Medical Image Segmentation (SSMIS)

최근의 SSMIS 연구들은 Self-training 기반의 방법론과 Consistency Regularization 전략을 주로 사용한다. 특히 Pseudo-labeling을 통해 레이블 없는 데이터의 활용도를 높이려 하지만, 이러한 방법들은 도메인 시프트가 발생하는 실질적인 시나리오(UMDA, Semi-MDG)에서는 한계를 보인다.

### 2. Transformer in Semi-supervised Segmentation

Vision Transformer (ViT) 기반 모델들은 강력한 일반화 능력을 갖추고 있으나, CNN에 비해 Inductive Bias가 약하고 방대한 학습 데이터를 요구한다. 의료 영상과 같이 데이터가 제한적인 환경에서는 오버피팅 문제가 심화되며, 이를 해결하기 위해 CNN-Transformer 하이브리드 구조가 제안되었으나 네트워크 구조의 복잡성과 튜닝의 어려움이라는 단점이 있다.

### 3. UMDA 및 Semi-MDG

UMDA는 주로 Cycle-GAN과 같은 이미지 수준의 정렬이나 특징 수준의 정렬(Feature-level alignment)을 통해 도메인 불변성을 찾으려 한다. Semi-MDG는 메타 학습(Meta-learning)이나 푸리에 변환(Fourier transformation) 등을 통해 일반화 성능을 높이려 한다. 본 논문은 이러한 특정 작업 맞춤형 설계 없이도, 범용적인 semi-supervised 프레임워크 내에서 지식 교환만으로 유사하거나 더 뛰어난 성능을 낼 수 있음을 입증한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 Baseline

본 연구는 **SegFormer**를 백본으로 하며, 기본적으로 **Pseudo-labeling** 프레임워크를 기반으로 한다. 전체 손실 함수는 다음과 같이 지도 학습 손실 $\mathcal{L}_s$와 비지도 학습 손실 $\mathcal{L}_u$의 합으로 정의된다.

$$\mathcal{L} = \mathcal{L}_s + \mathcal{L}_u = \frac{1}{N_l} \sum_{i=0}^{N_l} \mathcal{L}_{CE}(f(x^l_i), y_i) + \frac{1}{N_u} \sum_{j=0}^{N_u} \mathcal{L}_{CE}(f(x^u_j), \hat{y}_j)$$

여기서 $f$는 세그멘테이션 모델, $\mathcal{L}_{CE}$는 픽셀 수준의 Cross-Entropy 손실이며, $\hat{y}_j$는 모델이 생성한 Pseudo-label이다.

### 2. Labeled-to-Unlabeled (L2U) Knowledge Delivery

레이블 없는 데이터셋은 정밀한 감독 신호가 없어 학습 수렴이 느리고 Semantic Knowledge가 부족하다. 이를 해결하기 위해 레이블 데이터셋의 전경(Foreground) 영역 패치를 무작위로 선택하여 레이블 없는 이미지와 그에 대응하는 Pseudo-label에 붙여넣는 Copy-paste 방식을 사용한다.

$$x^u_i[:, p_h:p_h+s, p_w:p_w+s] = x^l_j[:, p_h:p_h+s, p_w:p_w+s]$$
$$\hat{y}_i[:, p_h:p_h+s, p_w:p_w+s] = y_j[:, p_h:p_h+s, p_w:p_w+s]$$

이렇게 수정된 이미지 $x^{u'}_j$와 레이블 $\hat{y}'_j$를 사용하여 $\mathcal{L}_u$를 계산함으로써, 레이블 없는 데이터 학습 흐름에 명시적인 Semantic Knowledge를 주입한다.

### 3. Unlabeled-to-Labeled (U2L) Knowledge Delivery

소량의 레이블 데이터만으로는 도메인 패턴을 파악하기 어려워 오버피팅이 발생하기 쉽다. 이를 방지하기 위해 레이블 없는 데이터의 특징 채널에서 공통적인 Domain Knowledge를 추출하여 레이블 데이터 흐름에 전달하는 **Channel-wise Cross-Attention** 메커니즘을 도입한다.

1. **Query, Key, Value 생성**: 레이블 데이터의 특징 $f^l$에서 Query $Q$를, 레이블 없는 데이터의 특징 $f^u$에서 Key $K$와 Value $V$를 생성한다.
   $$Q, K, V = f^l W_Q, f^u W_K, f^u W_V$$
2. **정규화 특징 계산**: Cross-attention을 통해 레이블 데이터의 개별적 편향을 정규화하는 특징 $\hat{f}^l$을 계산한다.
   $$\hat{f}^l = \text{Attention}(Q^\top, K^\top, V^\top) = \text{softmax}[\psi(Q^\top K)] V^\top$$
   여기서 $\psi(\cdot)$는 Instance Normalization을 의미한다.
3. **최종 출력**: 원래의 레이블 특징 $f^l$과 정규화된 특징 $\hat{f}^l$을 가중 합산하여 최종 출력 $\tilde{f}^l$을 얻는다.
   $$\tilde{f}^l = [\alpha \times \hat{f}^l + (1-\alpha)f^l] W_O$$

이 메커니즘은 SegFormer 인코더 내의 모든 Transformer 블록에 삽입된다. 추론 단계에서는 데이터가 두 흐름을 모두 통과하도록 하여 단순화된 Self-attention 형태로 동작한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: SSMIS (Synapse, LASeg, AMOS), UMDA (MMWHS), Semi-MDG (M&Ms, SCGM) 총 6개 데이터셋 사용.
- **지표**: Dice score, Average Surface Distance (ASD), Jaccard, HD95.
- **구현**: PyTorch 2.1.0, NVIDIA H800 GPU, SGD 옵티마이저, SegFormer-B5 백본.

### 2. 정량적 결과

- **SSMIS**: Synapse 데이터셋(20% 레이블)에서 Dice score를 기존 SOTA 대비 약 7.5% 향상시켰다. LASeg와 AMOS 데이터셋에서도 모두 SOTA 성능을 달성하였다.
- **UMDA**: MMWHS 데이터셋에서 MR $\to$ CT 작업 시 Fully-supervised 상한선(Upper-bound)과 단 0.5% 차이의 매우 높은 성능을 보였으며, 일부 어려운 클래스(AA)에서는 상한선을 상회하는 결과를 얻었다.
- **Semi-MDG**: M&Ms 및 SCGM 데이터셋에서 기존의 도메인 특화 방법론(EPL, StyleMatch 등)보다 높은 성능을 기록하였다. 특히 M&Ms 데이터셋(2%, 5% 레이블 설정)에서 각각 1.18%p, 1.87%p의 Dice 향상을 보였다.

### 3. 정성적 결과 및 분석

- **시각적 비교**: 결과 영상이 기존 GenericSSL이나 다른 SOTA 방법들보다 더 정확하고 매끄러운(Smooth) 분할 결과를 생성함을 확인하였다.
- **T-SNE 시각화**: M&Ms 데이터셋 실험에서, S&D Messenger가 기존 EPL 방법론보다 소스 도메인과 타겟 도메인 간의 특징 거리(Domain distance)를 효과적으로 줄였음을 확인하였다.

## 🧠 Insights & Discussion

### 1. Transformer의 한계 극복

본 논문은 Transformer 기반 모델이 적은 데이터셋에서 오버피팅되기 쉽다는 점을 지적한다. 이는 Transformer의 Inductive Bias가 CNN보다 약하기 때문인데, S&D Messenger의 U2L 전달 메커니즘이 레이블 없는 데이터로부터 도메인 지식을 주입함으로써 Transformer가 더 일반적인 표현(General representation)을 학습하도록 돕는다.

### 2. 범용 프레임워크의 가능성

특정 작업(UMDA 혹은 Semi-MDG)을 위해 정교하게 설계된 기존 알고리즘들보다, 단순한 Pseudo-labeling 구조에 '지식 교환'이라는 메커니즘을 추가한 본 방법론이 더 우수한 성능을 냈다는 점은 매우 고무적이다. 이는 도메인 시프트 문제의 핵심이 결국 'Semantic'과 'Domain' 지식의 효율적인 통합에 있음을 시사한다.

### 3. 한계 및 확장성

본 논문은 주로 아키텍처 수준의 설계에 집중하였으며, 하이퍼파라미터 $\alpha$와 $s$에 대해 비교적 둔감한 성능 변화를 보였으나, 여전히 최적의 값을 찾는 과정이 필요하다. 하지만 저자들은 이 방식이 SAM(Segment Anything Model)과 같은 다른 Transformer 기반 프레임워크의 레이블 효율적 학습에도 적용될 수 있을 가능성을 제시한다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 레이블 데이터(Semantic Knowledge 보유)와 레이블 없는 데이터(Domain Knowledge 보유) 사이의 지식 불균형 문제를 해결하기 위해 **S&D Messenger**를 제안한다. **L2U(Labeled-to-Unlabeled)**는 Copy-paste를 통해 정밀한 감독 신호를 전달하고, **U2L(Unlabeled-to-Labeled)**은 Cross-attention을 통해 도메인 일반화 지식을 전달한다. 이를 통해 단일 프레임워크로 **SSMIS, UMDA, Semi-MDG**라는 세 가지 서로 다른 과제를 모두 SOTA 수준으로 해결하였으며, 특히 데이터가 부족한 환경에서 Transformer 기반 모델의 오버피팅 문제를 효과적으로 완화하였다.
