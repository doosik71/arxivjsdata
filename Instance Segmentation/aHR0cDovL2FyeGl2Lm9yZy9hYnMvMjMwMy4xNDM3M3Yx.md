# DoNet: Deep De-overlapping Network for Cytology Instance Segmentation

Hao Jiang, Rushan Zhang, Yanning Zhou, Yumeng Wang, Hao Chen (2023)

## 🧩 Problem to Solve

세포학(Cytology) 이미지에서의 세포 인스턴스 분할(Instance Segmentation)은 생물학적 분석과 암 스크리닝에서 매우 중요한 역할을 한다. 하지만 이 작업은 다음과 같은 두 가지 주요한 기술적 난제로 인해 매우 어렵다.

첫째, 세포들이 서로 겹쳐져 군집(Cluster)을 형성하는 문제가 있다. 특히 세포의 세포질(Cytoplasm)은 반투명한 특성을 가지고 있어, 겹쳐진 영역에서 경계가 모호해지고 대비가 낮아져 정확한 경계 예측이 어렵다.

둘째, 배경에 존재하는 세포 유사체(Mimics)와 파편(Debris)들이 핵(Nuclei)으로 오인되는 문제가 있다. 예를 들어, 경부 세포 이미지에서 흔히 발견되는 백혈구와 점액 얼룩은 모델이 핵을 잘못 예측하게 만드는 주요 원인이 된다.

본 논문의 목표는 이러한 반투명한 세포의 중첩 문제를 해결하기 위해 세포 영역을 분해하고 다시 결합하는 전략을 통해 세포의 경계를 명확히 하고, 배경 노이즈의 영향을 최소화하는 DoNet이라는 네트워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'분해 후 재결합(Decompose-and-Recombined)'** 전략이다. 일반적인 인스턴스 분할 모델이 전체 마스크를 한 번에 예측하려 하는 것과 달리, DoNet은 중첩된 영역을 명시적으로 분리하여 인식한 뒤 이를 다시 통합함으로써 인식 능력을 높인다.

구체적으로, 세포 군집을 **교집합 영역(Intersection region)**과 **여집합 영역(Complement region)**으로 분해하여 각각의 특징을 학습하게 함으로써, 중첩된 반투명 영역에서 발생하는 모호함을 해결한다. 또한, 세포질 내부에 핵이 존재한다는 생물학적 제약 조건을 활용하여, 예측된 세포질 영역을 가이드로 삼아 핵을 예측하는 메커니즘을 도입함으로써 배경의 유사체로 인한 오탐지를 줄였다.

## 📎 Related Works

### 기존 연구 및 한계
1. **세포 분할 접근법**: 초기에는 픽셀 수준의 분할 후 Watershed 알고리즘이나 CRF(Conditional Random Field)와 같은 후처리 기법을 통해 인스턴스를 분리하는 '분할 후 정제(Segment-then-refine)' 방식이 사용되었다. 이후 Mask R-CNN과 같은 엔드 투 엔드(End-to-end) 딥러닝 모델들이 도입되었으나, 여전히 중첩된 영역의 상호작용을 명시적으로 모델링하지 못해 경계 예측이 불분명하다는 한계가 있다.
2. **Amodal Instance Segmentation**: 가려진 객체의 보이지 않는 부분을 추론하는 Amodal 분할 연구들이 진행되었다. 하지만 일반적인 Amodal 분할은 '완전 가려짐(Occlusion)'을 가정하는 반면, 세포학 이미지의 세포들은 '반투명(Semi-transparent)'하게 겹쳐져 있다. 따라서 기존의 Amodal 방식은 반투명 영역이 제공하는 풍부한 형태 정보를 충분히 활용하지 못한다.

### 차별점
DoNet은 단순히 가려진 부분을 추론하는 것을 넘어, 반투명한 중첩 영역(Intersection)과 중첩되지 않은 영역(Complement) 사이의 관계를 명시적으로 모델링하여, 세포의 전체 구조를 더 정확하게 복원한다.

## 🛠️ Methodology

### 전체 파이프라인 구조
DoNet은 Mask R-CNN을 베이스라인으로 하며, 크게 세 가지 모듈인 **DRM**, **CRM**, **MRP**가 순차적으로 연결된 구조를 가진다.

1. **Coarse Mask Segmentation**: 먼저 Mask R-CNN을 통해 거친(Coarse) 마스크 $\hat{e}^c_{k,i}$를 생성한다. 이는 이후 단계에서 배경 노이즈를 억제하고 하위 영역 분해를 위한 기초 정보로 활용된다. 이때의 손실 함수는 다음과 같다.
   $$L_{coarse} = L_{reg} + L_{cls} + L_{cmask}$$

2. **Dual-path Region Segmentation Module (DRM)**: 
   거친 마스크와 RoI 특징을 입력받아 세포 군집을 교집합 영역($o_{k,i}$)과 여집합 영역($m_{k,i}$)으로 분해한다. 두 개의 독립적인 마스크 헤드($H_o, H_m$)가 각각을 예측하며, 픽셀 단위의 교차 엔트로피(CE) 손실 함수 $L_{dec}$를 통해 학습한다.
   $$L_{dec} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{N_k} \sum_{i=1}^{N_k} (L_{ce}(\hat{o}_{k,i}, o_{k,i}) + L_{ce}(\hat{m}_{k,i}, m_{k,i}))$$

3. **Semantic Consistency-guided Recombination Module (CRM)**: 
   분해된 영역들의 특징을 다시 융합하여 정제된 전체 인스턴스 마스크 $\hat{e}^r_{k,i}$를 생성한다. 특히, 예측된 교집합 영역과 여집합 영역을 논리적 합집합으로 결합한 결과와 CRM이 예측한 최종 마스크 사이의 **의미론적 일관성(Semantic Consistency)**을 강제하는 $L_{cons}$ 손실을 도입한다. 이때 결합 연산은 XOR 기반의 $F_{merge}(\cdot)$ 함수를 사용한다.
   $$L_{cons} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{N_k} \sum_{i=1}^{N_k} L_{ce}(\hat{e}^r_{k,i}, F_{merge}(\hat{o}_{k,i}, \hat{m}_{k,i}))$$

4. **Mask-guided Region Proposal (MRP)**: 
   세포질 마스크를 Attention Map $\hat{M}_k$로 활용하여 FPN의 특징 맵 $f_k$를 재가중치화(Re-weighting)한다.
   $$f^w_k = \hat{M}_k \circ f_k$$
   이를 통해 세포질이 존재할 가능성이 낮은 배경 영역의 특징을 억제함으로써, 배경의 유사체(Mimics)가 핵으로 잘못 예측되는 False Positive 문제를 해결한다.

### 학습 절차 및 전체 손실 함수
모델은 지도 학습 방식으로 훈련되며, 전체 손실 함수는 다음과 같이 구성된다.
$$L = L_{coarse} + \lambda_{dec}L_{dec} + \lambda_{rmask}L_{rmask} + \lambda_{cons}L_{cons}$$
또한, 데이터 부족 문제를 해결하기 위해 투명도와 중첩 비율을 조절할 수 있는 인스턴스 수준의 데이터 증강(Synthetic clusters) 기법을 사용하여 일반화 성능을 높였다.

## 📊 Results

### 실험 설정
- **데이터셋**: ISBI2014(경부 세포 이미지) 및 CPS(액상 세포학 데이터셋)를 사용하였다.
- **평가 지표**: mAP, Dice coefficient, F1-score, AJI(Aggregated Jaccard Index) 등을 활용하였다.
- **비교 대상**: 일반 인스턴스 분할 모델(Mask R-CNN, Cascade R-CNN, HTC 등)과 Amodal 분할 모델(Occlusion R-CNN 등)을 비교 대상으로 설정하였다.

### 주요 결과
- **정량적 결과**: DoNet은 두 데이터셋 모두에서 SOTA 성능을 기록하였다. 특히 ISBI2014에서 mAP는 기존 최고 성능 대비 2.68%, AJI는 0.52% 향상되었다. CPS 데이터셋에서도 mAP 1.85% 향상을 보였다.
- **데이터 증강 효과**: 합성 데이터를 추가로 사용했을 때 CPS 데이터셋의 mAP와 AJI가 각각 0.45%, 0.68% 추가 상승하여, 중첩 영역 추론 능력이 강화됨을 확인하였다.
- **Ablation Study**: 
    - DRM을 통해 영역을 분해하는 것만으로도 mAP가 상승하였으며, 여기에 CRM의 재결합 전략과 일관성 제약($L_{cons}$)을 추가했을 때 성능 향상 폭이 가장 컸다.
    - MRP 모듈은 배경 노이즈를 효과적으로 억제하여 mAP를 추가적으로 향상시켰다.

## 🧠 Insights & Discussion

### 강점 및 분석
DoNet의 가장 큰 성과는 세포의 '반투명성'이라는 생물학적 특성을 딥러닝 아키텍처에 효과적으로 녹여낸 점이다. 단순히 가려진 부분을 추측하는 Amodal 방식이 아니라, 중첩 영역과 비중첩 영역을 동시에 예측하고 이들의 관계를 강제함으로써 모호한 경계를 명확히 구분해낼 수 있었다. 또한, 세포질과 핵의 포함 관계라는 도메인 지식을 MRP 모듈에 적용하여 의료 영상 특유의 노이즈 문제를 해결한 점이 인상적이다.

### 한계 및 논의사항
논문에서는 합성 데이터를 통해 성능을 높였다고 언급하고 있으나, 실제 의료 현장에서 발생하는 매우 복잡하고 다양한 변형의 중첩 사례를 모두 합성 데이터로 커버할 수 있는지는 추가적인 검증이 필요하다. 또한, 반투명도(Transparency)의 정도가 세포마다 다를 수 있는데, 이에 대한 적응적 처리 메커니즘이 명시적으로 제시되지 않은 점이 아쉽다.

## 📌 TL;DR

DoNet은 반투명한 세포들이 겹쳐진 세포학 이미지에서 경계 모호성과 배경 노이즈 문제를 해결하기 위해 **'분해-재결합'** 전략을 제안한 네트워크이다. 중첩 영역과 여집합 영역을 나누어 예측하는 **DRM**과 이를 다시 통합하여 일관성을 부여하는 **CRM**, 그리고 세포질 정보를 이용해 핵 예측의 정확도를 높이는 **MRP**를 통해 기존 SOTA 모델들을 뛰어넘는 성능을 달성하였다. 이 연구는 의료 영상뿐만 아니라 일반적인 비전 분야의 가려진 객체 분할(Occluded Instance Segmentation) 연구에도 중요한 통찰을 제공할 가능성이 크다.