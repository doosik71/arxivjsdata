# Complete Instances Mining for Weakly Supervised Instance Segmentation

Zecheng Li, Zening Zeng, Yuqi Liang and Jin-Gang Yu (2024)

## 🧩 Problem to Solve

본 논문은 이미지 수준의 라벨(image-level labels)만을 사용하여 객체의 위치와 마스크를 동시에 추정하는 Weakly Supervised Instance Segmentation (WSIS) 문제를 다룬다. WSIS는 정밀한 인스턴스 수준의 어노테이션을 얻는 데 드는 비용과 시간을 줄일 수 있다는 점에서 매우 중요하지만, 거친(coarse) 수준의 라벨을 세밀한(fine) 세그멘테이션 작업과 정렬시키는 것이 매우 어렵다는 과제가 있다.

특히 기존의 Proposal-based paradigm에서는 하나의 인스턴스가 여러 개의 proposal로 표현되는 'redundant segmentation' 문제가 발생한다. 이로 인해 네트워크가 객체의 전체 모습이 아닌 가장 변별력이 높은(most discriminative) 일부 영역만을 예측하는 경향이 있으며, 결과적으로 세그멘테이션의 정확도가 떨어지는 문제가 발생한다. 따라서 본 논문의 목표는 이러한 redundant segmentation 문제를 명시적으로 모델링하고, 온라인 정제 과정을 통해 완전한 인스턴스(complete instances)를 마이닝함으로써 WSIS의 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 MaskIoU head를 통해 proposal의 완전성 점수(integrity score)를 예측하고, 이를 활용한 Complete Instances Mining (CIM) 전략으로 정제된 pseudo labels를 생성하는 것이다.

주요 기여 사항은 다음과 같다.

1. WSIS 분야에 처음으로 MaskIoU head를 도입하여 proposal의 품질을 평가하고, pre-computed pseudo labels 및 정제된 pseudo labels에서 발생하는 노이즈를 필터링하기 위한 Anti-noise 전략을 제안하였다.
2. Redundant segmentation 문제를 명시적으로 해결하기 위해, 네트워크가 완전한 인스턴스에 더 집중하도록 유도하는 CIM 전략을 제시하였다.
3. 제안한 방법론을 PASCAL VOC 2012 및 MS COCO 데이터셋에 적용하여 기존의 state-of-the-art 성능을 상당한 차이로 경신하였다.

## 📎 Related Works

WSIS는 크게 Proposal-based paradigm과 Proposal-free paradigm으로 나뉜다.

- **Proposal-based paradigm**: PRM, IAM, Label-PEnet, WS-RCNN, PDSL 등이 이에 해당하며, 인스턴스가 proposal로 표현될 수 있다고 가정하여 문제를 분류 작업으로 단순화한다. 그러나 본 논문에서 지적하듯 redundant segmentation 문제로 인해 정밀도가 떨어지는 한계가 있다.
- **Proposal-free paradigm**: IRNet, BESTIE 등이 있으며, proposal 없이 온라인으로 결과를 생성한다. 이들은 주로 pre-computed pseudo labels(예: CAM, WSSS map)에 크게 의존하며, 이로 인해 추가적인 성능 향상에 제약이 있다.

본 논문은 Proposal-based 방식이 높은 Recall을 가지지만 Precision이 낮다는 점에 주목하여, CIM 전략을 통해 Precision을 높임으로써 두 패러다임의 장점을 결합하고자 한다. 또한, 기존의 Random Walk(RW)나 Conditional Random Fields(CRF)를 이용한 인스턴스 마이닝 방식이 학습 속도를 크게 저하시키는 점을 극복하기 위해 효율적인 MaskIoU head와 CIM을 제안한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 하나의 Anti-noise branch와 $K$개의 Refinement branch로 구성된다. 먼저 AGPL 전략을 통해 pre-computed pseudo labels를 생성하고, MaskFuse를 통해 proposal feature를 추출한 뒤 각 브랜치로 전달한다. CIM 전략은 이전 브랜치의 출력을 바탕으로 정제된 pseudo labels를 생성하여 다음 브랜치를 지도적으로 학습시킨다.

### 주요 구성 요소 및 상세 설명

#### 1. MaskFuse

WSIS에서는 binary mask를 proposal로 사용하므로 기존의 RoIPool이나 RoIAlign을 직접 사용할 수 없다. 이를 해결하기 위해 MaskFuse라는 경량 연산을 제안한다. MaskFuse는 RoIAlign으로 얻은 bounding box feature $B \in \mathbb{R}^{h \times w \times D}$와 RoICrop으로 추출한 proposal mask $R_{crop} \in \mathbb{R}^{h \times w}$를 결합한다. 구체적으로 $R_{crop} \odot B$와 $B$를 concatenation한 후, convolutional layer와 fully connected layer를 통해 마스크 수준과 박스 수준의 특징을 융합한다.

#### 2. Refinement Branch 및 MaskIoU Head

각 Refinement branch는 분류 점수를 출력하는 classification head와 proposal의 완전성을 예측하는 MaskIoU head로 구성된다. $k$번째 branch의 출력은 $y_k, t_k \in \mathbb{R}^{N \times \{C+1\}}$이며, 각각 분류 및 완전성 점수를 나타낸다. 학습을 위한 손실 함수 $L_{ref}^k$는 다음과 같다.

$$L_{ref}^k = -\frac{1}{N_{fg} + N_{bg}} \sum_{n=1}^{N} \sum_{c=0}^{C} w_n^k \hat{y}_{n,c}^k \log y_{n,c}^k + \frac{1}{N_{fg}} \sum_{n=1}^{N} \sum_{c=1}^{C} w_n^k \hat{y}_{n,c}^k L_{Smooth-L1}(\hat{t}_{n,c}^k - t_{n,c}^k)$$

여기서 $w_n^k$는 노이즈로 인한 성능 저하를 막기 위한 loss weight이며, $\hat{y}^k, \hat{t}^k$는 CIM을 통해 생성된 정제된 pseudo labels이다.

#### 3. Complete Instances Mining (CIM)

CIM은 redundant segmentation 문제를 해결하기 위해 다음 두 단계로 진행된다.

- **Step 1 (Selecting Seeds)**: 분류 점수가 높은 상위 $p_{seed}$ 비율의 proposal을 선택하고, NMS를 적용하여 seed set을 구성한다.
- **Step 2 (Mining Pseudo Ground Truth)**: 각 seed에 대해, 해당 seed를 포함하면서 가장 높은 완전성 점수(integrity score)를 가진 proposal을 pseudo ground truth $P^k$로 선택한다.

이렇게 선택된 $P^k$를 기준으로, $\text{IoU}(R_i, P^k)$가 특정 임계값 $\tau_{cls}$보다 크면 해당 카테고리로 분류 라벨 $\hat{y}^k$를 부여하고, $\tau_{iou}$보다 크면 완전성 라벨 $\hat{t}^k$를 부여한다.

#### 4. Anti-noise Strategy

Pre-computed 및 refined pseudo labels의 노이즈 문제를 해결하기 위해 두 가지 전략을 사용한다.

- **Anti-noise Branch**: WSDDN 구조를 채택하여 분류 및 완전성 점수를 출력하며, 이미지 수준 라벨과 PCL loss를 사용하여 학습한다.
- **Anti-noise Sampling**: 정제된 pseudo labels 중 노이즈가 섞인 라벨은 일반적으로 loss weight가 낮다는 점에 착안하여, $w_n^k$를 샘플링 확률로 사용하여 pseudo ground truth를 샘플링함으로써 강건성을 높인다.

전체 손실 함수는 다음과 같다.
$$L_{total} = L_{anti} + \sum_{k=1}^{K} L_{ref}^k$$

## 📊 Results

### 실험 설정

- **데이터셋**: PASCAL VOC 2012 (20개 클래스), MS COCO (80개 클래스).
- **평가 지표**: VOC 2012의 경우 $\text{mAP}_{25}, \text{mAP}_{50}, \text{mAP}_{70}, \text{mAP}_{75}$를 측정하였고, COCO의 경우 $\text{mAP}_{50-95}$를 측정하였다.
- **백본**: ResNet-50을 기본으로 사용하였으며, 비교를 위해 VGG-16, HRNet-W48도 사용하였다.

### 정량적 결과

- **VOC 2012**: ResNet-50 백본 기준, $\text{mAP}_{25}$ 64.9%, $\text{mAP}_{50}$ 51.1%를 달성하였다. 특히 Mask R-CNN으로 정제(refinement)했을 때 $\text{mAP}_{25}$ 68.7%, $\text{mAP}_{50}$ 55.9%까지 상승하여 SOTA 성능을 보였다.
- **COCO**: $\text{mAP}_{50}$ 기준, BESTIE(14.3%) 대비 Ours(17.0%, refinement 적용 시)로 향상된 성능을 보였다.
- **효율성**: 제안 방법은 단순한 linear layer 기반의 head들로 구성되어 VOC 2012 기준 ResNet-50 사용 시 약 6.5시간 만에 학습이 완료될 정도로 효율적이다.

### Ablation Study

- **AGPL 및 MaskIoU/CIM 영향**: AGPL 도입 시 $\text{mAP}_{50}$가 38.1%에서 49.2%로 급증하였으며, CIM 전략을 추가했을 때 $\text{mAP}_{75}$가 20.6%에서 23.8%로 상승하여 완전한 인스턴스 마이닝의 효과를 입증하였다.
- **Cascaded Threshold**: 임계값을 단계적으로 높이는 설계를 통해 $\text{mAP}_{50}$이 50.1%에서 51.1%로 향상되었다.

## 🧠 Insights & Discussion

본 논문은 Proposal-based WSIS의 고질적인 문제인 redundant segmentation을 '완전성 점수'라는 개념과 '마이닝 전략'을 통해 효과적으로 해결하였다. 특히, 단순히 가장 높은 분류 점수를 가진 proposal을 선택하는 것이 아니라, 분류 점수가 높은 영역(seed)을 포함하면서도 가장 완전한 형태를 띤 proposal을 찾는 CIM 전략이 핵심적인 역할을 하였다.

BESTIE와 같은 Proposal-free 방식은 세밀한 세그멘테이션 맵을 생성하지만, 여러 인스턴스를 하나로 묶어버리는 'grouping instances' 문제가 발생하는 반면, 본 논문의 방식은 redundancy 문제는 있지만 인스턴스 간 분리 능력은 더 우수함을 시각화 결과(Figure 3)를 통해 보여주었다.

한계점으로는 Anti-noise sampling 전략이 분석을 복잡하게 만들 수 있다는 점이 언급되었으며, 향후 연구에서는 이러한 샘플링 전략 없이도 모델의 강건성을 높일 수 있는 방법을 탐구할 필요가 있다.

## 📌 TL;DR

본 논문은 이미지 수준 라벨만 사용하는 WSIS에서 발생하는 **redundant segmentation(하나의 객체가 여러 proposal로 쪼개져 예측되는 현상)** 문제를 해결하기 위해, **MaskIoU head**와 **Complete Instances Mining (CIM)** 전략을 제안하였다. 이를 통해 네트워크가 객체의 부분만 예측하는 것이 아니라 전체 형상을 예측하도록 유도하였으며, **Anti-noise 전략**으로 pseudo labels의 노이즈 문제를 완화하였다. 결과적으로 PASCAL VOC 2012와 MS COCO 데이터셋에서 SOTA 성능을 달성하였으며, 이는 효율적인 온라인 정제 과정이 WSIS의 성능 향상에 결정적임을 시사한다.
