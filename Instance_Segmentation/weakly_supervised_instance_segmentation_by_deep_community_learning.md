# Weakly Supervised Instance Segmentation by Deep Community Learning

Jaedong Hwang, Seohyunyun Kim, Jeany Son, Bohyung Han (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Weakly Supervised Instance Segmentation (WSIS)**이다. 일반적인 Instance Segmentation은 각 객체의 정교한 Bounding Box와 Pixel-level mask 주석(Annotation)이 필요하지만, 이러한 고품질의 데이터를 대규모로 구축하는 것은 막대한 비용과 인력이 소모된다. 따라서 본 연구는 이미지 수준의 클래스 레이블(Image-level class labels)만 사용하여 개별 객체를 식별하고 분할하는 것을 목표로 한다.

이 문제의 핵심적인 어려움은 Weakly Supervised 학습 시 모델이 객체의 전체 영역이 아닌, 클래스를 구분하기에 가장 유리한 **Discriminative parts(판별적 부분)**에만 과도하게 집중하는 경향이 있다는 점이다. 예를 들어, 사람의 전체 모습이 아닌 얼굴 부분만 활성화되어 마스크를 생성하는 문제가 발생한다. 또한, Object Detection과 Semantic Segmentation이라는 두 가지 서로 다른 작업을 동시에 수행할 때, 불완전한 Ground-truth로 인해 발생하는 노이즈가 서로에게 악영향을 미쳐 학습의 안정성을 저해하는 문제가 존재한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deep Community Learning (DCL)** 프레임워크를 통해 여러 태스크 간의 능동적인 상호작용을 유도하고 긍정적인 피드백 루프(Positive feedback loop)를 구축하는 것이다.

단순히 여러 목적 함수를 병렬로 학습하는 Multi-task learning과 달리, Community Learning은 Object Detection, Instance Mask Generation (IMG), Instance Segmentation (IS) 모듈이 서로의 결과물을 가이드로 삼아 유기적으로 연결된다. 구체적으로, Detector가 찾은 제안 영역(Proposal)이 IMG의 가이드가 되고, IMG가 생성한 Pseudo-GT mask가 IS의 학습 목표가 되며, 이 과정이 다시 Feature Extractor의 가중치 업데이트로 이어져 전체 시스템의 견고함을 높이는 구조이다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들의 한계를 지적한다.

- **Weakly Supervised Object Detection (WSOD):** 주로 Multiple Instance Learning (MIL) 방식을 사용하며, OICR과 같은 최신 기법들이 존재하지만 여전히 객체의 전체 영역보다는 판별적 부분에 집중하는 문제가 있다.
- **Weakly Supervised Semantic Segmentation (WSSS):** Class Activation Map (CAM)을 통해 픽셀 수준의 레이블을 추정하지만, 인스턴스 간의 구분(Instance-wise separation)이 어렵다.
- **Weakly Supervised Instance Segmentation (WSIS):** 기존 연구들은 복잡한 파이프라인을 가지거나, 단순한 후처리(Post-processing)에 의존하며, Detection과 Segmentation 모듈 간의 긴밀한 상호작용이 부족하여 성능 향상에 한계가 있었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 파이프라인

본 모델은 **Feature Extractor**, **Object Detector (with Regressor)**, **Instance Mask Generation (IMG)**, **Instance Segmentation (IS)**의 네 가지 주요 구성 요소로 이루어져 있으며, end-to-end 방식으로 학습된다.

1. **Feature Extractor:** ResNet50을 백본으로 사용하며, Spatial Pyramid Pooling (SPP) 레이어를 통해 각 Proposal의 특징을 추출한다.
2. **Object Detection Module:** OICR 알고리즘을 기반으로 객체의 위치를 찾고 클래스 레이블을 부여한다. 특히, 본 논문에서는 클래스 구분 없이 공통적으로 적용되는 **Class-agnostic bounding box regressor**를 도입하여 특정 클래스에 치우친 판별적 학습을 방지하고 정규화 효과를 얻었다.
3. **Instance Mask Generation (IMG) Module:** Detector가 제공한 Proposal-level 레이블을 바탕으로 Pseudo-GT mask를 생성한다. 표준 CAM의 한계를 극복하기 위해 다음 세 가지 기법을 적용한다.
    - **Background class CAM:** 배경 클래스를 추가하여 객체와 배경을 더 명확히 구분한다.
    - **Weighted GAP:** Isotropic Gaussian kernel을 사용하여 Proposal의 중심 픽셀에 더 높은 가중치를 부여한다.
    - **Feature Smoothing:** 입력 특징값 $f$를 $\log(1+f)$로 변환하여 과도하게 높은 피크(Peak) 값을 억제함으로써 공간적으로 정규화된 맵을 생성한다.
    - 최종 Pseudo-GT mask $\tilde{M}$은 다음과 같이 계산된다:
      $$\tilde{M} = \delta \left[ \frac{1}{3} \sum_{k=1}^{3} M_k > \xi \right]$$
      (여기서 $\delta[\cdot]$는 지시 함수, $\xi$는 임계값이다.)
4. **Instance Segmentation Module:** IMG 모듈이 생성한 $\tilde{M}$을 Ground-truth로 삼아 픽셀 단위의 이진 분류(Binary classification)를 수행함으로써 최종 마스크를 예측한다.

### 2. 손실 함수 (Loss Functions)

전체 손실 함수는 세 모듈의 손실 합으로 정의된다:
$$L = L_{det} + L_{img} + L_{seg}$$

- **Detection Loss ($L_{det}$):** 이미지 분류 손실($L_{cls}$), 정제 손실($L_{refine}$), 그리고 Bounding box 회귀 손실($L_{reg}$)의 합이다.
  - $L_{reg}$는 Proposal과 Pseudo-GT 간의 $\text{smooth } \ell^1\text{-norm}$을 사용한다.
- **IMG Loss ($L_{img}$):** Detector에서 얻은 Pseudo-label $\tilde{y}_{rc}$와 CAM 네트워크의 출력 $p_{rc}^k$ 사이의 Binary Cross Entropy (BCE) 손실을 사용한다.
- **Segmentation Loss ($L_{seg}$):** IMG 모듈이 생성한 Pseudo-GT mask $\tilde{M}$과 Segmentation 네트워크의 출력 $s_{ij}$ 사이의 픽셀 단위 BCE 손실을 사용한다.

### 3. 추론 및 후처리 (Inference & Post-processing)

추론 시에는 IMG 모듈의 결과 $M_c$와 IS 모듈의 결과 $S_c$를 앙상블하여 최종 마스크 $O_c$를 생성한다:
$$O_c = \delta \left[ \frac{M_c + S_c}{2} > \xi \right]$$
이후, 경계선 세부 묘사를 위해 **Multiscale Combinatorial Grouping (MCG)** proposal을 사용하여 최종 마스크를 보정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** PASCAL VOC 2012 segmentation dataset.
- **학습 조건:** 이미지 수준의 클래스 레이블만 사용. ResNet50 백본, 단일 Titan XP GPU 사용.
- **평가 지표:** mAP (IoU threshold 0.25, 0.5, 0.75) 및 ABO (Average Best Overlap).

### 2. 주요 결과

- **정량적 성능:** 본 제안 방법은 후처리를 제외하고도 기존의 Weakly Supervised 방식들(PRM, IAM, Label-PEnet 등)보다 전반적으로 우수한 성능을 보였다. 특히 MCG 후처리를 적용했을 때 mAP 0.5에서 높은 성능을 기록하였다.
- **Ablation Study:**
  - Detector $\rightarrow$ IMG $\rightarrow$ IS $\rightarrow$ REG 순으로 모듈을 추가할수록 mAP와 CorLoc 성능이 지속적으로 향상됨을 확인하였다.
  - IMG 모듈 내의 배경 클래스 추가(BG), Weighted GAP, Feature Smoothing(FS) 모두 성능 향상에 기여하였으며, 세 가지를 모두 적용했을 때 최적의 성능을 보였다.
  - **Class-agnostic regressor**가 Class-specific regressor보다 높은 성능을 보였는데, 이는 모든 클래스가 회귀 모델을 공유함으로써 개별 클래스의 편향(Bias)을 줄이고 정규화 효과를 얻었기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **Community Learning**이라는 개념을 통해 개별 모듈의 약점을 상호 보완적으로 해결했다는 점이다. 단순히 모듈을 연결한 것이 아니라, 서로가 생성한 Pseudo-GT가 다시 다른 모듈의 학습 가이드가 되고, 이것이 최종적으로 공통된 Feature Extractor를 정규화하는 선순환 구조를 구축하였다.

특히 **Class-agnostic regressor**의 도입은 매우 흥미로운 지점이다. 일반적으로는 클래스별 특성이 다르므로 Class-specific-ly 학습하는 것이 당연해 보이지만, Weakly Supervised 환경에서는 레이블의 불완전성으로 인해 오히려 특정 클래스의 판별적 부분에 과적합(Overfitting)될 위험이 크다. 이를 공통 회귀 모델로 해결함으로써 객체의 전체적인 형태를 더 잘 포착하게 되었다.

다만, 본 연구는 여전히 Pseudo-GT의 품질에 의존하며, 추론 단계에서 MCG와 같은 외부 제안(Proposal) 기반의 후처리를 사용한다는 점에서 완전히 독립적인 End-to-end segmentation이라기에는 한계가 있다. 또한, 매우 복잡한 객체나 서로 겹쳐 있는 동일 클래스 객체들에 대한 분리 성능은 여전히 개선의 여지가 남아있다.

## 📌 TL;DR

본 논문은 이미지 수준의 레이블만으로 인스턴스 분할을 수행하는 **Deep Community Learning** 프레임워크를 제안한다. Object Detection, Mask Generation, Instance Segmentation 모듈이 서로 피드백을 주고받는 루프를 형성하여, Weakly Supervised 학습의 고질적 문제인 '판별적 부분에만 집중하는 현상'을 억제하고 객체 전체 영역을 효과적으로 포착한다. 특히 Class-agnostic regressor와 개선된 CAM 기법을 통해 PASCAL VOC 2012 데이터셋에서 기존 SOTA 모델들을 상회하는 성능을 달성하였으며, 이는 향후 고비용의 마스크 주석 없이도 고성능 인스턴스 분할 모델을 구축하는 데 중요한 기여를 할 것으로 보인다.
