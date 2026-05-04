# Synthetic Instance Segmentation from Semantic Image Segmentation Masks

Yuchen Shen, Dong Zhang, Zhao Zhang, Liyong Fu, Qiaolin Ye (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Instance Segmentation 모델 학습에 필요한 고비용의 어노테이션 데이터 확보 문제이다. 일반적으로 Fully-supervised Instance Segmentation을 위해서는 픽셀 수준(pixel-level)의 시맨틱 클래스 정보뿐만 아니라, 동일 클래스 내에서 서로 다른 개체를 구분하는 인스턴스 수준(instance-level)의 정밀한 어노테이션이 필수적이다. 하지만 이러한 데이터 구축에는 막대한 인력과 시간이 소요된다.

이를 해결하기 위해 Image-level class label이나 Point label 등을 사용하는 Weakly-supervised Instance Segmentation 방식들이 제안되었으나, 이러한 방식들은 객체의 경계가 불분명하거나 정확도 및 재현율(Recall)이 낮아 실제 응용 시나리오의 요구사항을 충족시키지 못하는 한계가 있다. 따라서 본 논문의 목표는 인스턴스 수준의 어노테이션 없이, 기존의 Semantic Segmentation 마스크만을 활용하여 효율적이면서도 정밀한 Instance Segmentation 결과를 도출하는 새로운 패러다임인 SISeg(Synthetic Instance Segmentation)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 기존에 잘 학습된 Semantic Segmentation 모델을 '거인의 어깨'로 활용하여, 시맨틱 마스크로부터 인스턴스 정보를 합성(Synthetic)해내는 것이다. 주요 기여 사항은 다음과 같다.

- **인스턴스 인식 가이드 제공**: Displacement Field Detection Module(DFM)을 통해 각 픽셀에서 인스턴스 중심점(centroid)으로 향하는 2D 오프셋 벡터를 예측함으로써, 동일 클래스 내의 서로 다른 인스턴스를 구분한다.
- **경계 정밀화**: Learnable category-agnostic object boundary branch(CBR)를 도입하여 객체 간의 명확한 경계 정보를 학습하고, 이를 통해 세그멘테이션 결과를 정밀하게 수정한다.
- **높은 효율성 및 유연성**: 인스턴스 수준의 어노테이션이 필요 없으며, 기존의 어떤 Semantic Segmentation 프레임워크와도 결합할 수 있는 유연한 구조를 가진다. 또한 추가적인 모델 재학습 없이 효율적인 추론이 가능하다.

## 📎 Related Works

### 1. Instance Segmentation

기존 방법론은 크게 두 가지로 나뉜다.

- **Detection-based methods**: Mask R-CNN과 같이 Bounding Box를 먼저 예측하고 그 내부에서 마스크를 생성하는 Top-down 방식이다. 하지만 객체 간 겹침이 심한 경우 제안된 박스가 부정확할 수 있다는 한계가 있다.
- **Segmentation-based methods**: 픽셀 간의 관계나 임베딩을 학습하여 인스턴스를 그룹화하는 Bottom-up 방식이다. 고해상도 마스크를 얻을 수 있으나, 일반적으로 Top-down 방식보다 성능이 낮다. 본 논문의 SISeg는 이 Bottom-up 방식에 해당한다.

### 2. Semantic Segmentation

FCN, DeepLabv3, PSPNet 등의 CNN 기반 모델과 SETR, SegFormer 같은 Transformer 기반 모델들이 발전해 왔다. SISeg는 이러한 기존 모델들의 결과물을 입력으로 사용하므로, 기반이 되는 Semantic Segmentation 모델의 성능이 최종 결과에 직접적인 영향을 미친다.

### 3. Weakly-Supervised Image Segmentation

이미지 수준 라벨, 바운딩 박스, 포인트 라벨 등을 활용한 연구들이 진행되었으나, 정밀한 위치 정보의 부재로 인해 실제 시나리오에서 요구되는 높은 정확도를 달성하는 데 어려움이 있다. SISeg는 픽셀 수준의 시맨틱 어노테이션만을 사용한다는 점에서 일종의 Weakly-supervised 설정으로 볼 수 있으며, 기존 방식보다 더 나은 정확도-어노테이션 트레이드오프를 제공한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

SISeg는 크게 두 단계로 구성된다.

- **Step 1**: 기학습된 Semantic Segmentation 모델을 통해 입력 이미지 $X$로부터 시맨틱 마스크 $M^{sem}$을 획득한다.
- **Step 2**: 획득한 $M^{sem}$을 SISeg 네트워크에 입력하여 DFM과 CBR 두 개의 병렬 브랜치를 통해 인스턴스를 구분하고 경계를 정밀화한다.

### 2. Displacement Field Detection Module (DFM)

DFM은 동일 클래스의 서로 다른 인스턴스를 구분하기 위해 각 픽셀의 Displacement Field Vector를 예측한다.

- **구조**: ResNet-50/101 기반의 Backbone을 통해 Feature Pyramid $\{P_1, P_2, P_3, P_4, P_5\}$를 생성한다. Top-down 방식으로 특징 맵을 통합하여 최종적으로 2D 오프셋 벡터 $D \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times 2}$를 예측한다.
- **학습 원리**: 동일 인스턴스에 속한 픽셀 쌍 $(\alpha, \beta)$는 동일한 인스턴스 중심점으로 수렴하는 오프셋 벡터를 가진다는 가정하에 self-supervised 방식으로 학습한다.
- **손실 함수**:
  - 객체 픽셀 쌍에 대한 손실: $$L_{Ob+D} = \frac{1}{|S_{Ob}^+|} \sum_{(\alpha, \beta) \in S_{Ob}^+} |D_{\alpha, \beta} - \hat{D}_{\alpha, \beta}|$$ 여기서 $\hat{D}_{\alpha, \beta}$는 픽셀 좌표의 차이이며, $D_{\alpha, \beta}$는 예측된 벡터의 차이이다.
  - 배경 픽셀 쌍에 대한 손실: 중심점 추정을 억제하기 위해 $$L_{Ba+D} = \frac{1}{|S_{Ba}^+|} \sum_{(\alpha, \beta) \in S_{Ba}^+} |D_{\alpha, \beta}|$$를 사용한다.
- **인스턴스 추출**: 반복적인 정제 과정(Eq. 11)을 거쳐 오프셋 벡터가 0에 가까운 지점들을 후보 중심점으로 설정하고, Connected Component Labeling(CCL) 알고리즘을 통해 최종 인스턴스 마스크 $F$를 생성한다.

### 3. Class Boundary Refinement (CBR)

시맨틱 세그멘테이션의 불완전한 경계를 보완하기 위해 CBR 모듈을 사용한다.

- **구조**: Backbone의 특징 맵들을 통합하여 경계 맵 $B \in \mathbb{R}^{H \times W \times 1}$를 예측한다.
- **픽셀 쌍 유사도**: 경계 맵 $B$를 이용하여 두 픽셀 $\alpha, \beta$ 사이의 유사도 $r_{\alpha\beta}$를 계산한다.
    $$r_{\alpha\beta} = (1 - \max_{i \in \{\gamma, \dots, \eta\}} B(v_i))^\lambda$$
- **손실 함수**: 클래스 균형 교차 엔트로피 손실(Class-balanced Cross-Entropy Loss) $L_B$를 사용하여 유사도를 학습한다.

### 4. Semantic-aware Propagation

학습된 유사도 행렬 $R$을 전이 확률 행렬 $Q$로 변환하여 Random Walk 프로세스를 수행한다.
$$\bar{F}_{SISeg} = Q^U * \text{Vec}((1-B) \circ \bar{F})$$
여기서 $(1-B)$는 경계 영역의 점수를 낮추는 패널티 항으로 작용하며, 이 과정을 통해 인스턴스 마스크의 경계가 정밀하게 확산 및 수정된다.

### 5. 전체 손실 함수

네트워크는 다음과 같은 통합 손실 함수를 최소화하도록 학습된다.
$$L = L_{Ob+D} + L_{Ba+D} + L_B$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: PASCAL VOC 2012, ADE20K.
- **베이스라인**: OCRNet (CNN 기반), SegNeXt (Transformer 기반).
- **지표**: Average Precision (AP), $AP_{50}$, $AP_{75}$, Params, FLOPs, FPS.

### 2. 주요 결과

- **Ablation Study**: DFM만 적용했을 때보다 CBR을 함께 적용했을 때 PASCAL VOC 2012 기준 AP가 $27.09\% \rightarrow 33.39\%$로 크게 상승하여, 경계 정밀화의 중요성이 입증되었다.
- **속도 및 정확도 트레이드오프**:
  - 정확도 중심: DeepLabv3와 결합 시 $34.42\% AP$ 달성 (단, 속도는 $4.07 FPS$).
  - 속도 중심: PSPNet과 결합 시 $32.83\% AP$와 $24.34 FPS$라는 실시간 성능을 달성하였다.
- **SOTA 비교**:
  - Weakly-supervised 방법론(IRNet 등)보다 훨씬 높은 정확도와 속도를 보였다.
  - Fully-supervised 방법론들과 비교했을 때, 인스턴스 수준 라벨 없이도 그 성능의 약 $75\%$ 수준까지 도달하였다.
  - 특히 ADE20K 데이터셋에서 `SISeg (Semantic gd)`(정답 시맨틱 마스크 입력 시)는 Fully-supervised 모델인 Mask2Former보다 높은 $32.59\% AP$를 기록하였다.

## 🧠 Insights & Discussion

### 1. 강점

- **어노테이션 효율성**: 인스턴스 수준의 라벨 없이 픽셀 수준의 시맨틱 라벨만으로 경쟁력 있는 성능을 낸다는 점이 매우 강력하다.
- **범용성**: 특정 아키텍처에 종속되지 않고 기존의 다양한 Semantic Segmentation 모델에 즉시 적용 가능하다.
- **효율적 구조**: DFM과 CBR이 백본을 공유하며 단순한 Conv 레이어로 구성되어 추가적인 연산 오버헤드가 적다.

### 2. 한계 및 비판적 해석

- **시맨틱 모델 의존성**: 실험 결과에서 나타나듯, 기반이 되는 Semantic Segmentation의 품질이 낮으면 최종 인스턴스 세그멘테이션 성능이 급격히 저하된다. 이는 SISeg가 시맨틱 마스크를 입력으로 받는 구조적 한계에서 기인한다.
- **심한 가려짐(Severe Occlusion) 문제**: 시각화 결과에서 확인되듯, 객체가 심하게 겹쳐 인스턴스 중심점이 서로 매우 가까운 경우 구분에 실패하는 경향이 있다. 이는 Bottom-up 방식이 공통적으로 가지는 한계이며, 더 세밀한 라벨링이나 추가적인 제약 조건이 필요할 것으로 보인다.
- **입력 데이터의 품질**: 시맨틱 세그멘테이션 결과 자체가 잘못되어 객체가 여러 개로 쪼개진 경우, SISeg 역시 이를 수정하지 못하고 잘못된 인스턴스로 예측하는 문제가 발생한다.

## 📌 TL;DR

본 논문은 **인스턴스 수준의 고비용 어노테이션 없이**, 기존의 **Semantic Segmentation 마스크만을 활용**하여 인스턴스를 구분하는 **SISeg** 프레임워크를 제안한다. **Displacement Field(DFM)**를 통해 인스턴스 중심점을 찾고, **Boundary Refinement(CBR)**와 **Random Walk 기반 전파**를 통해 경계를 정밀화한다. 실험 결과, 실시간 추론 속도를 유지하면서도 기존 Weakly-supervised 방식들을 압도하고 일부 Fully-supervised 방식에 근접하는 성능을 보였다. 이 연구는 시맨틱 세그멘테이션의 성과를 인스턴스 세그멘테이션으로 확장하는 효율적인 가교 역할을 하며, 향후 Panoptic Segmentation 등으로의 확장 가능성이 높다.
