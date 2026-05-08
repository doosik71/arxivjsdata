# Prior to Segment: Foreground Cues for Weakly Annotated Classes in Partially Supervised Instance Segmentation

David Biertimpel, Sindi Shkodrani, Anil S. Baslamisli, Nora Baka (2021)

## 🧩 Problem to Solve

본 논문은 **Partially Supervised Instance Segmentation (PSIS)** 환경에서 발생하는 성능 저하 문제를 해결하고자 한다. PSIS는 모든 클래스에 대해 Bounding Box 레이블이 존재하지만, 일부 클래스에 대해서만 정교한 Instance Mask 레이블이 제공되는 설정이다.

기존의 PSIS 방법론들은 주로 Class Agnostic Mask Head(클래스 구분 없는 마스크 헤드)를 사용하여, 모든 객체에 공통적으로 적용되는 '전경(Foreground)'의 개념을 학습함으로써 마스크 레이블이 없는 약하게 주석된(Weakly Annotated) 클래스로 일반화하려 한다. 그러나 저자들은 다음과 같은 두 가지 근본적인 문제가 있음을 지적한다.

1. **배경으로의 오학습(Learning classes as background):** 마스크 레이블이 없는 약한 클래스의 객체가 마스크 레이블이 있는 클래스의 RoI(Region of Interest) 내 배경에 포함될 경우, 모델은 해당 객체의 특징을 전경이 아닌 '배경'으로 학습하게 된다. 이는 특히 다른 클래스와 자주 상호작용하는 객체들의 일반화 성능을 심각하게 저하시킨다.
2. **모호한 RoI 해결의 어려움(Ambiguous RoIs):** 하나의 RoI 내에 여러 객체가 겹쳐 있거나 복잡하게 배치된 경우, Class Agnostic Mask Head는 어떤 픽셀이 현재 타겟 객체의 전경인지 판단하는 데 어려움을 겪으며, 결과적으로 잘못된 객체를 세그멘테이션하거나 마스크를 누락시키는 결과가 발생한다.

따라서 본 논문의 목표는 마스크 헤드가 전경의 개념을 더 잘 파악할 수 있도록 유도하는 효율적인 Prior를 도입하여, 약하게 주석된 클래스에 대한 세그멘테이션 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Object Mask Prior (OMP)**를 도입하여 마스크 헤드에 전경에 대한 명시적인 힌트를 제공하는 것이다.

- **Box Head의 활용:** Box Classification Head는 이미 모든 클래스에 대해 학습되었으므로, RoI 내에서 어떤 부분이 주 객체(Primary Object)인지를 판별하는 능력을 이미 갖추고 있다는 점에 착안하였다.
- **CAM 기반의 전경 추출:** Class Activation Maps (CAMs)를 사용하여 Box Head가 판단한 가장 변별력 있는 영역(Discriminative Region)을 추출하고, 이를 마스크 헤드의 입력 특징에 더해줌으로써 전경에 대한 가이드를 제공한다.
- **End-to-End 정교화:** OMP를 단순한 정적 Prior로 사용하는 것이 아니라, 네트워크에 내장하여 마스크 손실 함수로부터 오는 그래디언트(Mask Gradients)가 Box Head까지 전달되게 한다. 이를 통해 CAM의 공간적 범위가 확장되어 객체의 전체 형태를 더 잘 커버하는 OMP로 발전하게 된다.

## 📎 Related Works

본 논문은 기존의 Instance Segmentation 및 PSIS 접근 방식과 다음과 같이 차별화된다.

- **기존 PSIS 접근 방식:**
  - **Weight Transfer (Mask X R-CNN):** Box weight에서 Mask weight로의 매핑을 학습하여 클래스 인식 마스크 헤드를 구축한다.
  - **Shape Priors (ShapeMask, ShapeProp):** $k$-means 클러스터링이나 Multiple Instance Learning (MIL)을 통해 모양 사전(Shape Prior)을 구축한다. 하지만 이러한 방식들은 본 논문이 지적한 '모호한 RoI'나 '배경으로의 오학습' 문제를 직접적으로 해결하지 않는다.
  - **Commonality Parsing (CPMask):** 경계 예측 헤드와 어텐션 기반의 Affinity Parsing 모듈을 사용하여 공통점을 학습한다. 성능은 우수하지만 아키텍처가 복잡하고 추가적인 연산 오버헤드가 크다.

- **본 연구의 차별점:** 별도의 복잡한 모듈을 추가하지 않고, 이미 존재하는 Box Head의 특징을 재활용하여 전경 힌트를 생성한다. 또한, 마스크 그래디언트를 통해 Prior 자체를 최적화함으로써 아키텍처의 단순함과 성능을 동시에 잡고자 하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인

OPMask는 Mask R-CNN의 메타 아키텍처를 기반으로 하며, ResNet-50/101 백본과 FPN(Feature Pyramid Network)을 사용한다. 핵심은 Box Head에서 생성된 OMP가 Mask Head의 입력으로 들어가는 구조이다.

### 2. Object Mask Prior (OMP) 생성

Box Head의 마지막 컨볼루션 레이어 특징 맵 $F_{box}$에서 GAP(Global Average Pooling)를 적용하기 전 단계의 특징을 활용한다.

$$M_{cam} = f_{W_{cls}}(F_{box})$$

여기서 $f_{W_{cls}}$는 분류 가중치 $W_{cls}$를 파라미터로 하는 $1 \times 1$ 컨볼루션 연산이다. 이를 통해 각 클래스에 대한 CAM을 효율적으로 계산하며, 학습 시에는 Ground Truth 레이블을, 추론 시에는 예측된 클래스를 사용하여 해당 클래스의 CAM 슬라이스를 선택한다.

### 3. Prior의 통합 및 마스크 예측

추출된 $M_{cam}$은 bilinear interpolation을 통해 $F_{fpn}$과 동일한 공간 해상도로 조정된 후, 다음과 같이 합산되어 **Object-aware features ($F_{object}$)**가 된다.

$$F_{object} = F_{fpn} + M_{cam}$$

이렇게 생성된 $F_{object}$는 마스크 헤드 $f_{mask}$를 통과한다. 마스크 헤드는 7개의 $3 \times 3$ 컨볼루션 레이어, 1개의 Transposed Convolution(해상도 2배 확대), 1개의 $1 \times 1$ 컨볼루션 레이어로 구성된다.

최종 마스크 예측 $M_{mask}$는 다음과 같이 계산된다.

$$M_{mask} = f_{mask}(F_{object})$$

### 4. 학습 목표 및 손실 함수

마스크 레이블 $M_{gt}$가 존재하는 클래스에 대해서만 Binary Cross-Entropy (BCE) 손실 함수를 적용하여 학습한다.

$$L_{Mask} = BCE(M_{mask}, M_{gt})$$

중요한 점은 $M_{mask}$가 $M_{cam}$을 통해 계산되었기 때문에, $\frac{\partial L_{Mask}}{\partial M_{cam}}$의 그래디언트가 Box Head의 특징 맵 $F_{box}$까지 전달된다는 것이다. 이를 통해 CAM이 단순히 가장 변별력 있는 작은 지점만 짚는 것이 아니라, 객체 전체를 커버하도록 정교화된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** COCO 데이터셋.
- **설정:** 80개 클래스를 강하게 주석된(Strongly) 클래스와 약하게 주석된(Weakly) 클래스로 분할. (주로 Pascal VOC의 20개 클래스와 나머지 60개 클래스로 구분하여 $\text{voc} \to \text{non-voc}$ 및 $\text{non-voc} \to \text{voc}$ 실험 수행)
- **비교 대상:** Mask R-CNN (Baseline), Mask X R-CNN, ShapeMask, ShapeProp, CPMask.

### 2. 주요 결과

- **정량적 성능:** ResNet-50 백본 기준, Baseline 대비 $\text{non-voc} \to \text{voc}$에서는 $10.1\text{ AP}$, $\text{voc} \to \text{non-voc}$에서는 $13.0\text{ AP}$의 상당한 성능 향상을 보였다.
- **SOTA 비교:** ResNet-50 환경에서 CPMask보다 우수한 성능을 보였으며, ResNet-101 환경에서도 경쟁력 있는 성능을 달성하였다. 특히 CPMask보다 아키텍처가 훨씬 단순함에도 불구하고 유사한 성능을 낸다는 점이 강점이다.
- **모호한 RoI 성능:** RoI 내에 객체가 겹쳐 있는 'Ambiguous' 상황에서 Baseline보다 월등히 높은 성능 향상을 보였으며, 이는 OMP가 전경을 정확히 짚어주기 때문임을 입증하였다.

### 3. 분석 결과

- **배경 오학습 문제 해결:** 클래스 간 겹침 정도(mean IoU)와 Mask AP 사이의 강한 음의 상관관계가 Baseline에서는 나타났으나, OPMask에서는 이 상관관계가 약화되었다. 이는 OMP가 약한 클래스가 배경으로 학습되는 문제를 효과적으로 억제하고 있음을 시사한다.
- **CAM 정교화 효과:** 단순한 CAM보다 마스크 그래디언트를 통해 학습된 OMP가 객체의 더 넓은 영역을 커버하며, 이는 최종 세그멘테이션 정확도 향상으로 이어진다.

## 🧠 Insights & Discussion

본 논문은 PSIS에서 Class Agnostic Mask Head가 겪는 근본적인 한계를 '전경 인식의 모호함'으로 정의하고, 이를 Box Head의 지식을 전이함으로써 해결하였다.

**강점:**

- **효율성:** 추가적인 복잡한 모듈(Boundary parsing, Attention module 등) 없이 기존의 Box Head를 활용함으로써 연산 오버헤드를 최소화하였다.
- **직관적 해결책:** 전경/배경의 구분 문제를 Box Head의 분류 능력을 통해 해결했다는 점이 논리적으로 매우 명쾌하다.
- **범용성:** 완전 지도 학습(Full Supervision) 설정에서도 $1.6\text{ AP}$의 향상을 보여, OMP가 단순히 약한 클래스뿐만 아니라 일반적인 모호한 RoI 해결에도 도움이 됨을 보였다.

**한계 및 논의:**

- **CAM 의존성:** OMP의 성능이 Box Head의 분류 정확도에 크게 의존한다. 만약 Box Head가 객체를 잘못 분류한다면 잘못된 Prior가 제공될 위험이 있다.
- **가정:** 본 논문은 Box Head가 이미 전경의 개념을 내재적으로 학습하고 있다고 가정한다. 하지만 매우 희귀한 클래스나 분류 난이도가 높은 클래스의 경우, 이 Prior가 충분히 강력하지 않을 가능성이 있다.

## 📌 TL;DR

본 연구는 Partially Supervised Instance Segmentation에서 마스크 레이블이 부족한 클래스들이 배경으로 오학습되거나, 겹쳐진 객체들 사이에서 전경을 찾지 못하는 문제를 해결하기 위해 **Object Mask Prior (OMP)**를 제안하였다. Box Head의 Class Activation Maps (CAMs)를 활용하여 전경 힌트를 생성하고 이를 마스크 헤드에 통합했으며, 마스크 그래디언트를 통해 이 Prior를 end-to-end로 정교화하였다. 결과적으로 기존 Baseline 대비 최대 $13.0\text{ AP}$의 성능 향상을 이루었으며, 매우 단순한 구조로 SOTA 수준의 성능을 달성하였다. 이 연구는 복잡한 추가 모듈 없이 기존 네트워크의 구성 요소를 효율적으로 재활용하여 약하게 주석된 데이터의 성능을 끌어올릴 수 있음을 보여주었다.
