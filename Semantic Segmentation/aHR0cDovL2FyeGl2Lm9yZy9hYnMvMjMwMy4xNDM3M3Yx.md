# DoNet: Deep De-overlapping Network for Cytology Instance Segmentation

Hao Jiang, Rushan Zhang, Yanning Zhou, Yumeng Wang, Hao Chen

## 🧩 Problem to Solve

세포 영상에서 세포 인스턴스 분할은 생물학적 분석 및 암 검진에 매우 중요하지만, 다음과 같은 두 가지 주요 도전 과제가 존재합니다:

* **광범위하게 겹치는 반투명 세포 클러스터**: 세포질이 서로 가려지고 염색 대비가 낮아 세포 경계 예측이 모호해집니다. 이는 특히 자궁경부 세포 영상에서 두드러집니다.
* **핵과 유사한 모방체 및 잔해의 혼동**: 배경에 널리 퍼져 있는 백혈구, 점액 얼룩 등과 같은 비표적 객체가 핵으로 잘못 분류될 수 있어 인스턴스 분할 모델을 오도할 수 있습니다.
기존 방법들은 겹치는 영역 내의 교차 영역(intersection)과 보완 영역(complement) 간의 복잡한 상호작용을 명시적으로 모델링하는 데 실패하여, 영역 간 관계에 대한 이해가 제한적이었습니다.

## ✨ Key Contributions

* **새로운 분해-재조합(decompose-and-recombined) 전략을 갖춘 De-overlapping Network (DoNet) 제안**: DRM(Dual-path Region Segmentation Module)을 통해 세포 영역을 명시적으로 분해하고, CRM(Semantic Consistency-guided Recombination Module)을 통해 교차, 보완, 인스턴스(세포) 구성 요소 간의 의미론적 관계를 암묵적 및 명시적으로 모델링합니다. 이러한 설계는 겹치는 세포 하위 영역에서 네트워크의 지각 능력을 향상시킵니다.
* **Mask-guided Region Proposal (MRP) 모듈 설계**: 세포질 주의 맵을 활용하여 세포 내 핵 정제를 수행함으로써, 세포 인스턴스의 생물학적 사전 지식을 모듈에 도입하고 배경에 널리 퍼진 모방체의 영향을 효과적으로 완화합니다.
* **두 가지 겹치는 세포학 이미지 분할 데이터셋(ISBI2014 및 CPS)에 대한 광범위한 실험**: 제안된 DoNet이 다른 최신(SOTA) 세포 인스턴스 분할 방법들보다 훨씬 우수한 성능을 보여줍니다.

## 📎 Related Works

* **세포학 인스턴스 분할 (Cytology Instance Segmentation)**:
  * **초기 접근 방식**: 픽셀 수준 분할 모델과 시드 워터셰드($\text{seeded watershed}$), 랜덤 워크($\text{random walk}$), 조건부 무작위장($\text{conditional random field}$), 별-볼록 매개변수화($\text{star-convex parameterization}$)와 같은 후처리 기술의 조합. 일부는 형태학적 사전 지식($\text{morphological prior}$)을 도입하여 경계 정교화.
  * **딥러닝 기반 종단간 학습**: Mask R-CNN [11]을 기반으로 한 접근 방식이 주류. IRNet [40]은 겹치는 자궁경부 세포 분할에서 인스턴스 관계를 탐색하고, MMT-PSM [39]은 Mean-Teacher 기반 준지도 학습 방식을 제안하여 비레이블 데이터를 활용합니다.
* **가림 인스턴스 분할 (Occluded Instance Segmentation)**:
  * **목표**: 부분적으로 보이는 영역을 기반으로 전체 객체(amodal mask)를 추론하여 가림 문제 해결. 'amodal perception'이라 칭함.
  * **주요 접근 방식**: [22]는 최초의 amodal 인스턴스 분할 데이터셋인 COCOA를 구축. Occlusion R-CNN (ORCNN) [9]은 R-CNN에 가시 마스크($\text{visible mask}$) 및 amodal 마스크($\text{amodal mask}$) 헤드를 추가하여 직접 예측. [37]은 형태 사전 지식을 활용. BCNet [18]은 가림자($\text{occluder}$)와 가려진 객체($\text{occludee}$) 간의 관계를 직접 모델링. ASBU [26]는 경계 불확실성 추정을 통해 가상 Ground Truth를 생성하는 약지도 amodal segmenter를 도입.
  * 본 연구는 기존 amodal 분할이 가시 영역에서 비가시 영역 추론에 집중하는 반면, DoNet은 교차 영역과 보완 영역 간의 불일치를 해결하는 데 중점을 둡니다.

## 🛠️ Methodology

제안된 DoNet은 Mask R-CNN [11]을 기본 모델로 하여 '분해-재조합(decompose-and-recombined)' 전략을 따르며, 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. **Coarse Mask Segmentation (기본 Mask R-CNN)**:
    * $\text{FPN}$(Feature Pyramid Network)과 $\text{RPN}$(Region Proposal Network)을 통해 특징을 추출하고 후보 객체 바운딩 박스를 생성합니다.
    * $\text{RoIAlign}$ 레이어를 통해 $\text{RoI}$ 특징 $\text{f}_{\text{roi}}^{\text{k,i}}$를 얻고, 감지 헤드 및 인스턴스 마스크 헤드($\text{H}_{\text{i}}$)에서 객체 클래스 $\hat{\text{c}}_{\text{k,i}}$, 바운딩 박스 $\hat{\text{b}}_{\text{k,i}}$, 그리고 거친 의미론적 마스크 $\hat{\text{e}}_{\text{c}}^{\text{k,i}}$를 예측합니다.
    * 손실 함수는 $\text{L}_{\text{coarse}} = \text{L}_{\text{reg}} + \text{L}_{\text{cls}} + \text{L}_{\text{cmask}}$입니다.

2. **Dual-path Region Segmentation Module (DRM)**:
    * 인스턴스 특징 $\text{f}_{\text{roi}}^{\text{k,i}}$와 인스턴스 마스크 헤드 이전의 풍부한 의미론적 특징 $\text{f}_{\text{c}}^{\text{k,i}}$의 연결(concatenation)을 입력으로 받습니다.
    * 동일한 구조의 교차 마스크 헤드($\text{H}_{\text{o}}$)와 보완 마스크 헤드($\text{H}_{\text{m}}$)를 통해 세포 클러스터를 교차 영역 $\hat{\text{o}}_{\text{k,i}}$과 보완 영역 $\hat{\text{m}}_{\text{k,i}}$으로 명시적으로 분해합니다.
    * 손실 함수는 $\text{L}_{\text{dec}} = \frac{1}{\text{K}} \sum_{\text{k}=1}^{\text{K}} \frac{1}{\text{N}_{\text{k}}} \sum_{\text{i}=1}^{\text{N}_{\text{k}}} (\text{L}_{\text{ce}}(\hat{\text{o}}_{\text{k,i}}, \text{o}_{\text{k,i}}) + \text{L}_{\text{ce}}(\hat{\text{m}}_{\text{k,i}}, \text{m}_{\text{k,i}}))$ 입니다.

3. **Semantic Consistency-guided Recombination Module (CRM)**:
    * $\text{H}_{\text{o}}$와 $\text{H}_{\text{m}}$에서 나온 특징 $\text{f}_{\text{o}}^{\text{k,i}}$, $\text{f}_{\text{m}}^{\text{k,i}}$를 $\text{f}_{\text{roi}}^{\text{k,i}}$와 융합하여 인스턴스 마스크 헤드($\text{H}_{\text{i}}$)를 재사용하여 정제된 통합 인스턴스 $\hat{\text{e}}_{\text{r}}^{\text{k,i}}$를 예측합니다.
    * **정제 마스크 손실**: $\text{L}_{\text{rmask}} = \frac{1}{\text{K}} \sum_{\text{k}=1}^{\text{K}} \frac{1}{\text{N}_{\text{k}}} \sum_{\text{i}=1}^{\text{N}_{\text{k}}} \text{L}_{\text{ce}}(\hat{\text{e}}_{\text{r}}^{\text{k,i}}, \text{e}_{\text{k,i}})$ 입니다.
    * **의미 일관성 정규화**: $\text{L}_{\text{cons}} = \frac{1}{\text{K}} \sum_{\text{k}=1}^{\text{K}} \frac{1}{\text{N}_{\text{k}}} \sum_{\text{i}=1}^{\text{N}_{\text{k}}} \text{L}_{\text{ce}}(\hat{\text{e}}_{\text{r}}^{\text{k,i}}, \text{F}_{\text{merge}}(\hat{\text{o}}_{\text{k,i}}, \hat{\text{m}}_{\text{k,i}}))$를 통해 재조합된 예측과 분해된 하위 영역의 병합 간의 일관성을 강화합니다. $\text{F}_{\text{merge}}(\cdot)$는 두 마스크 로짓의 Mask Exclusive-OR 연산을 수행합니다.

4. **Mask-guided Region Proposal (MRP)**:
    * CRM에서 얻은 모든 재조합된 인스턴스 예측 $\hat{\text{e}}_{\text{r}}^{\text{k,i}}$를 통합하여 의미론적 마스크 $\hat{\text{M}}_{\text{k}}$를 생성합니다.
    * 이 $\hat{\text{M}}_{\text{k}}$를 원본 $\text{FPN}$의 특징 $\text{f}_{\text{k}}$에 요소별 곱셈($\text{f}_{\text{w}}^{\text{k}} = \hat{\text{M}}_{\text{k}} \circ \text{f}_{\text{k}}$)하여 특징에 가중치를 재조정합니다.
    * 이를 통해 세포 외 픽셀에 대한 확률을 억제하여 배경 모방체로 인한 오탐지를 줄이고, 핵 제안이 세포 내 영역에 집중하도록 유도합니다.

5. **Instance-level Augmentor**:
    * 주석이 달린 세포 인스턴스를 기반으로 제어 가능한 겹침 비율과 투명도를 가진 대규모 합성 세포 클러스터를 생성하여 데이터 다양성을 높이고 모델의 가림 추론 능력을 향상시킵니다. (주요 비교 실험에서는 사용되지 않음).

6. **전체 손실 함수**:
    * $\text{L} = \text{L}_{\text{coarse}} + \lambda_{\text{dec}} \text{L}_{\text{dec}} + \lambda_{\text{rmask}} \text{L}_{\text{rmask}} + \lambda_{\text{cons}} \text{L}_{\text{cons}}$로 구성됩니다.

## 📊 Results

* **정량적 결과**:
  * **ISBI2014 및 CPS 데이터셋**: DoNet은 Mask R-CNN [11], Cascade R-CNN [2], Mask Scoring R-CNN [13], HTC [8]와 같은 일반 인스턴스 분할 방법 및 Occlusion R-CNN [9], Xiao et al. [37]과 같은 amodal 인스턴스 분할 방법을 포함한 모든 최신 방법 중에서 가장 높은 성능을 달성했습니다.
    * **ISBI2014**: 기존 최고 성능 [9] 대비 $\text{mAP}$에서 $2.68\%$, $\text{AJI}$에서 $0.52\%$ 개선을 보였습니다.
    * **CPS**: 기존 최고 성능 [37] 대비 $\text{mAP}$에서 $1.85\%$, $\text{AJI}$에서 $1.02\%$ 개선을 달성했습니다.
  * **ISBI2014 챌린지 우승자들과 비교**: $\text{Dice}$, $\text{TP}_{\text{p}}$, $\text{FN}_{\text{o}}$ 지표에서도 경쟁 우위를 보였습니다 (예: $\text{Dice}$ $0.921$, $\text{TP}_{\text{p}}$ $0.948$).
  * **합성 클러스터 증강의 효과**: CPS 데이터셋에서 인스턴스 수준 증강으로 합성 클러스터를 도입했을 때 $\text{mAP}$ $0.45\%$, $\text{AJI}$ $0.68\%$의 추가 개선을 보였으며, 이는 모델의 가림 추론 능력을 더욱 향상시켰음을 의미합니다.
* **정성적 결과**:
  * DoNet은 표준, 멀티태스크, amodal 인스턴스 분할 모델 대비 겹치는 영역에서 강력한 지각 능력과 세부적인 경계 구분을 보여주었으며 (Figure 4, 5), 특히 낮은 대비의 세포 인스턴스에서도 뛰어난 성능을 보였습니다.
  * 교차 영역, 보완 영역, 통합 인스턴스에 대한 히트맵 시각화(Figure 6)를 통해 DoNet이 겹침 개념을 기반으로 하위 영역을 성공적으로 식별하며, 낮은 해상도와 높은 투명도 영역에서도 잘 작동함을 입증했습니다.

## 🧠 Insights & Discussion

* **네트워크 구성 요소의 효과 (Ablation Study)**:
  * **DRM (Dual-path Region Segmentation Module)**: 통합 인스턴스를 교차 및 보완 하위 영역으로 명시적으로 분해함으로써 ISBI2014에서 평균 $\text{mAP}$를 $3.25\%$ 증가시켰습니다. 이는 겹치는 인스턴스의 하위 영역 특징을 활용하여 경계 인식을 향상시켰음을 의미합니다.
  * **CRM (Semantic Consistency-guided Recombination Module)**: DRM 후 CRM을 추가하는 '분해-재조합' 전략은 두 데이터셋에서 각각 $7.34\%$ 및 $1.74\%$의 $\text{mAP}$ 개선을 가져왔습니다. 이는 겹치는 영역에 대한 모델의 인식을 강화하면서 형태학적 정보를 보존하는 데 결정적인 역할을 합니다.
  * **MRP (Mask-guided Region Proposal)**: 배경 모방체의 부작용을 완화하여 ISBI2014에서 $\text{mAP}$ $0.59\%$, CPS에서 $\text{mAP}$ $0.31\%$의 추가 개선을 달성했습니다. 이는 $\text{MRP}$가 $\text{RPN}$이 세포 내 영역에 집중하도록 유도하여 배경 잡음으로 인한 오탐지를 효과적으로 줄였음을 보여줍니다.
* **DRM 및 CRM의 설계 선택**:
  * $\text{H}_{\text{o}}$ (교차 마스크 헤드)와 $\text{H}_{\text{m}}$ (보완 마스크 헤드)를 추가함으로써 $\text{mAP}$가 크게 개선되었으며, 특히 세포질 분할에서 더 큰 향상을 보였습니다.
  * $\text{CRM}$의 Fusion Unit을 통한 풍부한 의미론적 특징 통합과 일관성 정규화($\text{L}_{\text{cons}}$)는 모델의 겹침 추론 능력을 더욱 강화하여 세포질 $\text{mAP}$를 추가로 향상시켰습니다.
* **의의 및 확장성**: 이 연구는 세포학 분야의 겹치는 객체 지각 문제를 해결할 뿐만 아니라, 일반적인 비전 응용 시나리오에서 가려진 인스턴스 분할에도 광범위한 잠재력을 제공합니다.

## 📌 TL;DR

세포 영상에서 겹치는 세포와 배경 노이즈로 인한 모호한 경계 문제를 해결하고자, 본 논문은 **De-overlapping Network (DoNet)**를 제안합니다. DoNet은 **분해-재조합(decompose-and-recombined)** 전략을 사용하여, 세포 클러스터를 교차 및 보완 영역으로 명시적으로 분해하고, 이를 일관성 기반으로 재조합하여 통합 인스턴스를 예측합니다. 또한, **Mask-guided Region Proposal (MRP)** 모듈을 통해 세포질 주의 맵을 활용, 핵 제안을 세포 내 영역으로 제한함으로써 배경 모방체의 영향을 효과적으로 줄입니다. ISBI2014와 CPS 데이터셋에 대한 광범위한 실험 결과, DoNet은 기존 최신 방법들보다 모든 주요 지표에서 우수한 성능을 보였으며, 특히 겹치는 반투명 세포 경계의 정확한 분할에 대한 모델의 지각 능력이 크게 향상되었음을 입증했습니다.
