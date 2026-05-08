# Incremental Few-Shot Instance Segmentation

Dan Andrei Ganea, Bas Boom, Ronald Poppe

## 🧩 Problem to Solve

기존 Few-Shot Instance Segmentation (FSIS) 방법들은 새로운 클래스를 유연하게 추가하기 어렵고, 학습 및 테스트 시 각 클래스의 예시들을 메모리에 유지해야 하므로 메모리 집약적이라는 한계를 가지고 있습니다. 특히 인스턴스 세그멘테이션의 경우 픽셀 단위 주석을 얻는 비용이 높기 때문에, 제한된 데이터로 새로운 클래스를 효과적으로 처리하는 것이 중요합니다.

## ✨ Key Contributions

* **최초의 증분형 Few-Shot Instance Segmentation 방법인 iMTFA를 제안했습니다.** iMTFA는 기존 FSIS 및 증분형 Few-Shot Object Detection (FSOD) 분야의 최신 기술(SOTA)을 능가하는 성능을 보였습니다.
* **비증분형 Few-Shot Instance Segmentation baseline인 MTFA를 개발했습니다.** MTFA는 기존 FSOD 방법인 TFA [36]를 인스턴스 세그멘테이션 태스크로 확장하여 SOTA 성능을 달성했습니다.
* **메모리 오버헤드 문제를 해결했습니다.** iMTFA는 이미지 대신 임베딩 벡터를 저장하여, 모든 COCO 클래스에 대해 FSIS 성능을 공동으로 평가할 수 있게 한 최초의 연구입니다.
* **새로운 클래스 추가에 유연성을 제공합니다.** iMTFA는 추가 훈련이나 기존 훈련 데이터에 대한 접근 없이 새로운 클래스를 증분적으로 추가할 수 있도록 합니다.

## 📎 Related Works

* **인스턴스 세그멘테이션:** Mask R-CNN [11]과 같은 제안 기반(proposal-based) 방식이 널리 사용되지만, 적은 양의 훈련 데이터에는 취약합니다.
* **Few-Shot Learning:** 에피소드 방식(episodic methodology), 최적화 기반(optimization-based) [1, 9, 26], 메트릭 학습(metric-learning) [5, 10, 14, 33, 34, 35] 접근법이 있습니다. 코사인 유사도 분류기 [10]와 가중치 임프린팅(weight-imprinting) [25, 31]은 중요한 관련 기술입니다.
* **Few-Shot Object Detection (FSOD):** TFA [36]는 Faster R-CNN [28]을 기반으로 한 2단계 미세 조정(fine-tuning) 접근 방식으로 SOTA를 달성했습니다.
* **Few-Shot Instance Segmentation (FSIS):** Meta R-CNN [39], Siamese Mask R-CNN [21], FGN [8] 등이 제안되었으나, 대부분 지원 세트(support set)의 예시를 훈련 및 테스트 시 제공해야 하여 메모리 제약이 있습니다.
* **증분형 Few-Shot Object Detection:** ONCE [23]는 클래스 독립적(class-agnostic) 특징 추출기(feature extractor)를 학습합니다.
* **증분형 Few-Shot Instance Segmentation:** 본 연구가 최초의 접근법입니다.

## 🛠️ Methodology

본 논문은 Mask R-CNN [11]을 기반으로 하는 증분형 Few-Shot Instance Segmentation 방법인 iMTFA와 그 baseline인 MTFA를 제안합니다.

### 1. MTFA (비증분형 baseline)

* **아키텍처:** TFA [36]를 확장하여 마스크 예측 브랜치($M$)를 추가한 Mask R-CNN 기반.
* **훈련 절차 (2단계 미세 조정):**
  * **1단계:** 전체 네트워크를 기본 클래스($C_{base}$) 데이터셋으로 훈련합니다.
  * **2단계:** 특징 추출기($F$)는 고정하고, RoI 분류기($C$), 바운딩 박스 회귀기($R$), 마스크 예측기($M$)를 기본 및 신규 클래스($C_{base} \cup C_{novel}$)의 균형 잡힌 K-샷(K-shot) 데이터셋으로 미세 조정합니다.
* **코사인 유사도 분류기:** 분류기 $C$는 특징 추출기의 출력과 클래스 대표 벡터($w_j$) 간의 코사인 유사도를 사용하여 분류 점수를 생성합니다. 이는 더 판별적인 클래스 대표를 학습하는 데 도움이 됩니다.
    $$S_{i,j} = \frac{\alpha F(X)^T_i \cdot w_j}{\|F(X)_i\| \|w_j\|}$$
    여기서 $\alpha$는 소프트맥스 레이어에 전달되기 전 점수를 스케일링하는 데 사용됩니다.

### 2. iMTFA (증분형)

* **목표:** MTFA의 한계(새로운 클래스 추가 시 미세 조정 단계 재실행 필요)를 극복하기 위해, 모델을 클래스 독립적으로 만들고 특징 추출기 수준에서 판별적인 임베딩을 학습합니다.
* **인스턴스 특징 추출기 (IFE, Instance Feature Extractor):**
  * Mask R-CNN의 RoI-레벨 특징 추출기 $G$를 훈련하여 각 인스턴스에 대한 판별적인 임베딩 $z_i$를 생성합니다.
  * **훈련 절차 (2단계 미세 조정):**
    * **1단계:** MTFA와 동일하게, 전체 Mask R-CNN을 $C_{base}$ 클래스로 훈련합니다.
    * **2단계:** 백본($B$)과 RPN은 고정하고, $G$, $C$, $R$을 $C_{base}$ 클래스만으로 미세 조정합니다. 이 단계의 목표는 $C_{novel}$ 클래스에 대한 일반화 능력을 향상시키는 것입니다.
* **클래스 대표 생성:**
  * 새로운 클래스 추가 시, IFE는 각 K-샷 이미지($X$)에 대해 특징 임베딩 $z_i$를 계산합니다.
  * 새로운 클래스 대표($w_{new}$)는 K-샷 임베딩의 정규화된 평균으로 계산됩니다.
        $$w_{new} = \frac{1}{K} \sum_{i=0}^K \frac{z_i}{\|z_i\|}$$
  * 이 $w_{new}$는 분류기 $C$의 가중치 행렬 $W$에 추가됩니다. 이 과정은 훈련 없이 이루어지며, 메모리 제약을 줄입니다.
* **클래스 독립적 바운딩 박스 및 마스크 예측기:** $R$과 $M$은 클래스 독립적(class-agnostic)으로 설계되어, 새로운 클래스에 대해 별도의 가중치 학습이나 인스턴스 마스크 제공이 필요 없습니다.
* **추론:** 추론 시에는 RoI 임베딩과 클래스 대표 간의 코사인 거리가 가장 작은 클래스를 예측합니다.

## 📊 Results

* **COCO 신규 클래스(COCO-Novel) 성능:**
  * iMTFA와 MTFA는 Meta R-CNN [39] 및 MRCN+ft-full보다 AP (Average Precision)에서 크게 우수한 성능을 보였습니다 (모든 K-샷 설정에서).
  * AP50에서는 MTFA가 Meta R-CNN을 능가했지만, iMTFA는 약간 뒤처졌는데, 이는 iMTFA가 객체의 대략적인 위치를 찾는 데 어려움이 있을 수 있음을 시사합니다.
* **COCO 전체 클래스(COCO-All) 성능:**
  * FSIS에서 기본 및 신규 클래스($C_{base} \cup C_{novel}$) 전체에 대한 성능을 보고한 첫 연구입니다.
  * iMTFA는 증분형 FSOD 방법인 ONCE [23]보다 객체 감지(object detection) 성능이 우수했습니다.
  * MTFA는 대부분의 경우 iMTFA보다 기본 클래스에서 일관되게 우수한 성능을 보였는데, 이는 iMTFA가 새로운 클래스 대표를 생성할 때 기존 클래스 대표에 적응하는 능력이 부족하기 때문일 수 있습니다.
* **타 FSIS 방법과의 비교:**
  * **Siamese Mask R-CNN [21] (COCO-Split-2, GTOE 평가):** iMTFA와 MTFA 모두 Siamese Mask R-CNN을 능가했습니다. 특히 iMTFA는 K=1샷에서 더 좋은 성능을 보였습니다.
  * **FGN [8] (COCO2VOC, GTOE 평가):** MTFA는 인스턴스 세그멘테이션에서 FGN보다 우수했으며, iMTFA는 FGN과 대등한 성능을 보였습니다. FGN의 높은 객체 감지 AP50은 RPN 및 분류기 단계의 가이던스(guidance)와 더 깊은 백본(ResNet-101 vs. ResNet-50)에 기인할 수 있습니다.
* **추론 예시:** 성공적인 세그멘테이션은 정확했지만, 오분류, 잘못된 위치 추적, 부정확한 세그멘테이션 등 실패 사례도 있었습니다. "dining table"이나 "person"처럼 외형이 다양한 클래스에서 오탐이 더 많았습니다.
* **어블레이션 스터디:**
  * 클래스 특정(class-specific) 구성 요소와 미세 조정이 MTFA의 세그멘테이션 성능에 도움이 됨을 확인했습니다.
  * IFE에 대한 2단계 미세 조정이 단일 단계 훈련보다 효과적임을 입증했습니다.
  * 코사인 유사도 스케일링 인자 $\alpha$ 값은 데이터셋과 클래스 수에 따라 최적값이 달라짐을 확인했습니다.

## 🧠 Insights & Discussion

* **증분 학습의 실용성:** iMTFA는 새로운 클래스를 기존 네트워크에 유연하게 추가할 수 있는 실용적인 FSIS 접근 방식을 제시합니다. 이는 레이블링 비용이 높은 분야에서 매우 유용합니다.
* **메모리 효율성:** 임베딩 벡터를 클래스 대표로 저장하는 방식은 기존 FSIS 방법들의 주요 병목인 메모리 문제를 효과적으로 해결했습니다.
* **성능 향상 잠재력:**
  * iMTFA의 한계 중 하나는 새로운 임베딩 생성 시 기존 임베딩에 적응하지 못한다는 점입니다. 이는 [10, 35]와 같은 어텐션(attention) 메커니즘을 통해 개선될 수 있습니다.
  * iMTFA의 클래스 독립적 지역화(localization) 및 세그멘테이션 구성 요소는 MTFA의 클래스 특정 구성 요소에 비해 성능이 최적화되지 않았습니다. 임베딩에서 클래스 특정 바운딩 박스 회귀기 및 마스크 예측기로의 전이 함수(transfer function) 학습이나, [8, 21]과 같은 가이던스 메커니즘과의 결합이 성능 향상에 기여할 수 있습니다.
  * iMTFA의 고정된 바운딩 박스 회귀기 및 마스크 예측기는 기본 클래스 편향을 유발할 수 있으며, 가이던스 메커니즘이 이 문제를 완화할 수 있습니다.
* **장기적인 비전:** iMTFA의 발전은 비증분형 FSIS와 증분형 FSIS 간의 격차를 좁히고, 이미 강력한 네트워크에 새로운 클래스를 유연하게 추가할 수 있는 가능성을 열어줍니다.

## 📌 TL;DR

본 연구는 **최초의 증분형 Few-Shot Instance Segmentation (FSIS) 방법인 iMTFA**를 제안합니다. iMTFA는 Mask R-CNN의 특징 추출기를 재활용하여 **판별적인 인스턴스별 임베딩을 생성**하고, 이를 **코사인 유사도 분류기에서 클래스 대표로 사용**합니다. 이를 통해 **새로운 클래스를 훈련 없이 유연하게 추가**할 수 있으며, 이미지 대신 임베딩만 저장하여 **메모리 오버헤드 문제를 해결**합니다. 실험 결과, iMTFA와 그 비증분형 baseline인 MTFA는 기존 FSIS 및 증분형 FSOD SOTA를 능가하는 성능을 보였습니다.
