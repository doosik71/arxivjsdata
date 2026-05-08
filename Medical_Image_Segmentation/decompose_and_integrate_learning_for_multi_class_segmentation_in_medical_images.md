# Decompose-and-Integrate Learning for Multi-class Segmentation in Medical Images

Yizhe Zhang, Michael T. C. Ying, Danny Z. Chen (2019)

## 🧩 Problem to Solve

본 논문은 의료 영상의 다중 클래스 분할(Multi-class Segmentation) 문제에서 발생하는 학습 효율성 저하 문제를 해결하고자 한다. 의료 영상의 경우, 서로 다른 객체 클래스 간에 강한 상호 위치 관계 및 공간적 상관관계(Spatial Correlations)가 존재하는 경우가 많다.

전통적인 방식의 다중 클래스 분할은 전체 주석 맵(Annotation Map)을 하나의 대상으로 취급하고 Spatial Cross-Entropy Loss를 사용하여 학습한다. 그러나 클래스 간의 강한 상관관계로 인해, 네트워크가 학습하기 쉬운 클래스의 특징을 이용해 어려운 클래스를 추론하려는 경향이 생긴다. 이로 인해 특히 크기가 작거나 외형이 불분명한 클래스에 대해 네트워크가 충분한 표현 학습(Representation Learning) 능력을 탐색하지 못하게 되는 문제가 발생한다.

따라서 본 연구의 목표는 주석 맵을 의도적으로 분해(Decompose)함으로써, 네트워크가 각 클래스 또는 하위 구조에 대해 더 풍부하고 분리된(Disentangled) 특징 변환(Feature Transform)을 학습하도록 강제하는 'Decompose-and-Integrate' 학습 체계를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 원본 주석 맵을 여러 개의 세부 문제로 분해하여 학습시킨 뒤, 이를 다시 통합하는 것이다. 이를 통해 모델이 특정 클래스에 의존하지 않고 모든 대상 클래스에 대해 명시적으로 표현을 학습하도록 유도한다.

주요 기여 사항은 다음과 같다.

1. **Annotation Map Decomposition (AD)**: 객체 클래스, 객체 모양, 이미지 수준의 정보라는 세 가지 기준에 따른 주석 맵 분해 방법을 제안하였다.
2. **K-to-1 Deep Network Framework**: 분해된 $K$개의 하위 문제를 각각 학습하는 모듈들과 이를 최종 결과로 통합하는 구조의 엔드-투-엔드(End-to-End) 학습 가능한 프레임워크를 설계하였다.
3. **유연한 적용 가능성**: 특정 아키텍처에 종속되지 않고 DenseVoxNet(3D)이나 CUMedNet(2D)과 같은 최신 FCN(Fully Convolutional Networks)에 결합하여 성능을 향상시킬 수 있음을 입증하였다.

## 📎 Related Works

논문에서는 주석 맵을 수정하여 학습 성능을 높이려 했던 기존 연구들을 언급한다.

- **Directional Map**: 픽셀과 객체 중심 간의 상대적 위치 정보를 이용해 추가적인 학습 손실을 생성하는 방식이다.
- **Deep Watershed Transform**: 주석 맵을 Watershed Energy Map으로 변환하여 모델 학습을 가이드하는 방식이다.

이러한 선행 연구들은 주석 맵에 상대적 위치나 인스턴스 수준의 정보를 추가하는 것이 도움이 된다는 점을 보여주었다. 본 논문은 여기서 더 나아가, 주석 맵을 단순히 수정하는 것이 아니라 '분해'하여 각 구성 요소에 최적화된 특징 변환을 학습하게 함으로써 기존 접근 방식과 차별화를 둔다.

## 🛠️ Methodology

### 1. Segmentation Annotation Map Decomposition (AD)

연구진은 시나리오에 따라 세 가지 방식의 분해 전략을 제시한다.

**가. 객체 클래스 기반 분해 (Object-class based AD)**
$K$개의 클래스가 있을 때, 이를 $K$개의 이진(Binary) 주석 맵 $y^k_i \in \{0, 1\}^{m \times n}$로 분해한다. 각 맵은 특정 클래스 $k$에 해당하는 영역만 1로 표시하고 나머지는 0으로 처리한다.

**나. 객체 모양 기반 분해 (Object-shape based AD)**
객체의 볼록성(Convexity)을 기준으로 분해한다. 객체 $p$의 크기와 그 객체의 Convex Hull $p^{convex}$의 크기 비율을 계산하여, 이 비율이 임계값 $T_{shape}$ (본 논문에서는 0.9)보다 크면 볼록한 모양(Convex-like), 작으면 오목한 모양(Concave-like)으로 분리하여 두 개의 맵을 생성한다.

**다. 이미지 수준 정보 기반 분해 (Image-level information based AD)**
이미지 내에 존재하는 객체의 개수를 기준으로 분해한다. 객체가 하나만 존재하는 경우와 여러 개 존재하는 경우를 나누어 각각 별도의 맵으로 관리함으로써, 모델이 전역적인(Global) 정보와 고차원적인 정보를 더 잘 인식하도록 유도한다.

### 2. K-to-1 Deep Network Framework

분해된 $K$개의 주석 맵을 학습하기 위해 $K$개의 독립적인 세그멘테이션 모듈($f_{1.k}$)을 배치하고, 그 상위에 이들의 솔루션을 통합하는 최종 모듈($f^{complete}$)을 두는 구조이다.

### 3. 학습 목표 및 손실 함수

전체 네트워크는 엔드-투-엔드 방식으로 학습되며, 전체 손실 함수는 다음과 같이 정의된다.

$$ \text{Total Loss} = \frac{1}{h} \sum_{i=1}^{h} \left( L(f^{complete}(x_i), y_i) + \lambda \sum_{k=1}^{K} L(f_{1.k}(x_i), y_{k,i}) \right) $$

여기서:

- $h$는 학습 데이터의 총 개수이다.
- $L$은 Spatial Cross Entropy Loss를 의미한다.
- $f^{complete}(x_i)$는 전체 네트워크의 최종 출력이며, $y_i$는 원본 주석 맵이다.
- $f_{1.k}(x_i)$는 $k$번째 세부 모듈의 출력이며, $y_{k,i}$는 분해된 $k$번째 주석 맵이다.
- $\lambda$는 정규화 항으로 $\frac{1}{K}$로 설정된다.

## 📊 Results

### 실험 설정

- **데이터셋 및 적용 AD 방법**:
  - 3D Cardiovascular MR (HVSMR): 객체 클래스 기반 AD 적용.
  - Gland Segmentation (H&E): 모양 기반 AD 적용.
  - Lymph Node Ultrasound: 이미지 수준 정보 기반 AD 적용.
- **기본 모델**: DenseVoxNet (3D), CUMedNet+ (2D).
- **학습 세부사항**: Adam Optimizer 사용, 초기 학습률 0.0005 (30,000회 반복 후 0.00005로 감소), 배치 크기 8, 최대 60,000회 반복 학습.

### 주요 결과

- **3D Cardiovascular MR**: Class-AD + K-to-1 DenseVoxNet 조합이 Dice, ADB, Hausdorff 지표 모두에서 기존 SOTA 모델들보다 우수한 성능을 보였다.
- **Gland Segmentation**: Shape-AD + K-to-1 CUMedNet+ 조합이 $F_1$ Score 및 Object Dice 등에서 다른 최신 모델들(MILD-Net, CUMedNet 등)보다 유의미하게 높은 성능을 기록하였다.
- **Lymph Node Ultrasound**: Image-level-AD + K-to-1 구조가 IoU 0.8102, $F_1$ Score 0.8952를 달성하며 U-Net 및 일반 CUMedNet+ 대비 큰 폭의 성능 향상을 보였다.

### Ablation Study

제안 방법의 유효성을 검증하기 위해 다음 세 가지 대조군과 비교하였다.

1. **Large Model**: K-to-1 네트워크와 유사한 파라미터 수를 가진 단일 대형 네트워크.
2. **2-stacked Model**: 두 개의 네트워크를 단순히 쌓은 구조.
3. **K-to-1 w/o AD**: K-to-1 구조는 유지하되 주석 맵을 분해하지 않고 원본을 그대로 사용한 경우.

모든 실험에서 **AD + K-to-1** 조합이 가장 우수한 성능을 보였으며, 이는 단순히 모델의 크기를 키우거나 층을 쌓는 것보다 '주석 맵의 분해를 통한 표현 학습의 강제'가 더 효과적임을 시사한다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 흔히 발생하는 클래스 간 공간적 상관관계가 오히려 학습의 방해 요소가 될 수 있다는 점을 날카롭게 지적하였다. 특히 '쉬운' 클래스가 '어려운' 클래스의 학습을 방해하는 현상을 방지하기 위해, 문제를 의도적으로 쪼개어 각 요소에 최적화된 특징을 추출하게 만드는 전략은 매우 실용적이다.

**강점**:

- 특정 네트워크 구조에 의존하지 않고, 데이터(주석 맵)의 구성 방식을 바꾸는 전략이므로 범용성이 매우 높다.
- 단순히 데이터 증강을 하는 것이 아니라, 도메인 지식(모양의 볼록성, 객체 개수 등)을 학습 과정에 직접 주입하는 효과를 낸다.

**한계 및 논의사항**:

- $K$개의 모듈을 사용하므로 학습 및 추론 시 계산 비용(Computational Cost)이 증가한다. 비록 의료 영상의 클래스 수가 적어 관리 가능한 수준이라고 명시되었으나, 클래스 수가 매우 많은 데이터셋으로 확장할 경우 효율성 문제가 발생할 수 있다.
- 분해 기준(예: $T_{shape} = 0.9$)이 경험적으로 설정되었으며, 데이터셋마다 최적의 임계값이 다를 수 있다는 점이 명시되지 않았다.

## 📌 TL;DR

이 논문은 의료 영상 분할 시 클래스 간의 강한 상관관계로 인해 일부 클래스의 학습이 저해되는 문제를 해결하기 위해, 주석 맵을 클래스/모양/개수 기준으로 분해하여 학습하고 다시 통합하는 **Decompose-and-Integrate** 학습 체계를 제안한다. 이를 위해 **K-to-1 딥 네트워크** 구조와 통합 손실 함수를 도입하였으며, 3D MR, 2D 조직학 영상, 초음파 영상 등 다양한 데이터셋에서 기존 SOTA 모델보다 뛰어난 성능을 입증하였다. 이 연구는 복잡한 다중 클래스 분할 문제에서 각 클래스의 독립적이고 풍부한 특징 표현을 학습시키는 새로운 방법론을 제시했다는 점에서 향후 의료 AI 연구에 중요한 역할을 할 것으로 보인다.
