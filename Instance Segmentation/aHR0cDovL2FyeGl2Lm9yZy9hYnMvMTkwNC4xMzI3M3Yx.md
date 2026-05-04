# Detecting Reflections by Combining Semantic and Instance Segmentation

David Owen, Ping-Lin Chang (2019)

## 🧩 Problem to Solve

자연 이미지, 특히 거울이 많은 환경(예: 체육관)에서 발생하는 반사(Reflection) 현상은 자동 객체 검출 시스템에서 심각한 False Positive(오탐지)를 유발한다. 이러한 오탐지는 객체 검출, 카운팅, 세그멘테이션 작업의 전반적인 정확도를 크게 떨어뜨리는 원인이 된다.

본 논문의 목표는 반사 영역에 대한 명시적인 레이블링 없이도, Instance Segmentation과 Semantic Segmentation을 융합하여 이러한 반사로 인한 False Positive를 자동으로 식별하고 제거하는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **두 단계(Two-stage) 검출기가 겪는 컨텍스트 손실(Loss of broader contextual features) 문제를 Semantic Segmentation의 전역적 컨텍스트 유지 능력을 통해 해결**하는 것이다.

Mask R-CNN과 같은 Two-stage 검출기는 Region Proposal Network(RPN) 단계 이후 제안된 영역(Proposal) 내부의 정보에만 집중하게 되어, 거울의 프레임과 같은 주변 맥락 정보를 잃어버린다. 반면, 단일 단계(Single-stage) 구조인 Semantic Segmentation은 이미지 전체의 컨텍스트를 유지하므로, 해당 영역이 거울 내부인지 외부인지를 더 잘 구분할 수 있다는 직관에 기반한다.

## 📎 Related Works

**Instance Segmentation**은 Mask R-CNN과 같이 RPN을 통해 후보 영역을 먼저 찾고 이후 분류 및 마스킹을 수행하는 Two-stage 방식이 주류를 이룬다. 이는 정밀도는 높으나 계산 복잡도가 크고, 본 논문에서 지적하듯 RPN 이후 컨텍스트가 제한되는 한계가 있다.

**Semantic Segmentation**은 픽셀 단위의 레이블을 생성하며, 전체 씬의 구성(Stuff)을 인식하는 데 집중한다. 단일 단계 구조 덕분에 더 넓은 유효 수용 영역(Effective Receptive Field)을 가지며, 전역적 컨텍스트 정보를 더 잘 보존한다.

**Panoptic Segmentation**은 위 두 가지 방식을 통합하여 모든 픽셀에 레이블을 부여하면서도 개별 인스턴스를 구분하는 접근법이다. 기존 연구들은 주로 두 네트워크를 별도로 학습시킨 후 최종 단계에서 알고리즘적으로 융합하는 방식을 사용했다.

**반사 검출(Detecting Reflections)**과 관련해서는 기존에 기하학적 모델이나 능동형 프로젝션, 혹은 SLAM 기반의 접근법이 있었으나, 일반적인 컬러 이미지 세그멘테이션 관점에서의 해결책은 부족한 실정이었다.

## 🛠️ Methodology

### 전체 파이프라인

본 논문은 Instance Segmentation 모델(Mask R-CNN)과 Semantic Segmentation 모델(UPerNet)을 병렬적으로 사용하고, 그 결과를 융합하는 파이프라인을 제안한다.

1. **Instance Branch**: Mask R-CNN을 사용하여 객체 후보와 마스크를 생성한다.
2. **Semantic Branch**: UPerNet(ResNet50 backbone)을 사용하여 픽셀 단위의 클래스 점수를 생성한다.
3. **Fusion Stage**: 두 결과를 결합하여 최종적으로 반사된 객체를 제거한다.

### 융합 방법 및 판별식

본 연구는 Panoptic Segmentation의 개념에서 영감을 얻었으나, 단순한 휴리스틱 융합 방식을 채택하였다. 각 제안된 인스턴스 마스크 영역 내에서 Semantic Segmentation의 평균 점수를 계산하여, 이 값이 특정 임계값 $c$보다 높을 때만 해당 인스턴스를 수용한다.

판별식은 다음과 같다:
$$\text{Accept instance if: } \frac{\sum_{I} \text{score}}{\text{area}(I)} > c$$

여기서 $\sum_{I} \text{score}$는 제안된 인스턴스 마스크 $I$ 내부의 세만틱 점수 합계이며, $c$는 튜닝 가능한 파라미터이다. 기존의 Panoptic Segmentation 융합 방식이 인스턴스 결과에 우선순위를 두는 것과 달리, 본 방법은 세만틱 결과가 강력할 경우(즉, 반사 영역이라고 판단될 경우) 이를 우선하여 인스턴스를 제거한다.

### 학습 및 튜닝 절차

- **학습**: Mask R-CNN은 COCO 데이터셋으로 사전 학습되었으며, UPerNet은 ADE20K 데이터셋으로 사전 학습되었다. 이후 체육관 감시 카메라 데이터셋을 사용하여 파인튜닝을 진행했다.
- **파라미터 튜닝**: 임계값 $c$는 검증 세트에서 Precision과 Recall의 트레이드-오프를 분석하여 결정되었으며, 실험 결과 $c=0.04$일 때 최적의 성능을 보였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 체육관에서 수집된 약 22,000장의 감시 이미지 ($1920 \times 1080$ 해상도). 이 중 527장이 반사가 심한 이미지로 분류되었다.
- **비교 대상**: (1) 사전 학습된 Mask R-CNN, (2) 반사 데이터로 파인튜닝된 Mask R-CNN, (3) 제안하는 Joint Segmentation 방식.
- **지표**: Precision, Recall, Average Precision(AP), Average Recall(AR), 그리고 이미지당 False Positive 수.

### 주요 결과

1. **반사 제거 성능**: Joint Segmentation은 False Positive의 수를 획기적으로 줄였다.
    - 사전 학습 모델 기준: False Positive가 502개에서 251개로 감소, Precision이 $71\% \rightarrow 83\%$로 향상되었으며 Recall은 거의 영향을 받지 않았다.
    - 파인튜닝 모델 기준: Precision이 $80\% \rightarrow 84\%$로 향상되었다.
2. **파인튜닝의 한계**: Mask R-CNN을 반사 데이터로 파인튜닝하는 것만으로는 False Positive를 완전히 제거할 수 없었다. 놀랍게도, 파인튜닝만 한 모델(Precision 80%)보다 사전 학습 모델에 융합 방식을 적용한 결과(Precision 83%)가 더 우수했다.
3. **컨텍스트의 중요성 (Ablation Study)**: 이미지 편집을 통한 실험 결과, 반사 여부를 결정짓는 핵심 요소는 기하학적 구조나 텍스처가 아니라 **거울의 프레임(Mirror Frame)**과 같은 주변 컨텍스트였다. 프레임을 지웠을 때 Semantic Segmentation 모델이 반사를 인식하지 못함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Two-stage 검출기의 구조적 결함, 즉 RPN 이후의 정보 손실이 반사 오탐지의 근본 원인임을 밝혀냈다. RPN은 넓은 영역을 보지만 너무 많은 후보를 생성해야 하므로 모든 거울 영역을 제외하도록 학습되기 어렵고, 이후 단계의 분류기는 이미 잘려나간(cropped) 영역만 보므로 거울 프레임 같은 외부 단서를 활용할 수 없다.

반면, Semantic Segmentation은 이미지 전체를 유지하며 추론하므로 이러한 전역적 맥락을 활용해 "이 영역은 거울 내부이다"라는 것을 쉽게 판단할 수 있다. 이는 Panoptic Segmentation의 접근 방식이 단순한 벤치마크 통합을 넘어, 실제 산업 현장의 오탐지 문제를 해결하는 강력한 도구가 될 수 있음을 시사한다.

다만, 본 연구의 융합 방식은 단순한 휴리스틱 임계값 기반이라는 한계가 있으며, 거울 프레임이 가려진 경우에는 여전히 오탐지가 발생한다는 점이 확인되었다.

## 📌 TL;DR

이 논문은 Mask R-CNN과 같은 Two-stage 검출기가 반사 이미지에서 False Positive를 많이 생성하는 이유가 RPN 이후 컨텍스트 손실 때문임을 분석하고, 이를 해결하기 위해 전역적 맥락을 유지하는 Semantic Segmentation(UPerNet) 결과를 융합하는 방법을 제안하였다. 실험 결과, 단순한 파인튜닝보다 세만틱-인스턴스 융합 방식이 반사 제거에 훨씬 효과적임을 보였으며, 특히 거울 프레임과 같은 주변 맥락 정보가 반사 식별의 핵심임을 입증하였다. 이는 향후 정교한 Panoptic Segmentation 모델이 실제 객체 검출의 신뢰성을 높이는 데 기여할 가능성을 보여준다.
