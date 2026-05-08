# Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation

Jordi Pont-Tuset, Pablo Arbel ́aez, Jonathan T. Barron, Member, IEEE, Ferran Marques, Senior Member, IEEE, Jitendra Malik, Fellow, IEEE

## 🧩 Problem to Solve

기존 객체 인식 시스템은 종종 외부 분할기(segmenter)가 생성하는 고정된 영역 또는 윤곽선(contour) 집합에 의존하며, 이는 객체 제안(object proposal)의 품질을 제한합니다. 본 연구는 다중 스케일 정보를 효과적으로 활용하면서, 높은 품질과 범주 독립적인 객체 제안을 생성하기 위한 효율적이고 통일된 바텀업(bottom-up) 계층적 이미지 분할 및 객체 제안 생성 접근 방식의 필요성을 다룹니다.

## ✨ Key Contributions

* **고속 정규화된 컷(Normalized Cuts) 알고리즘**: 윤곽선 전역화에 필요한 고유 벡터(eigenvector) 계산을 20배 가속화하는 효율적인 `DNCuts` (Downsampled Normalized Cuts) 알고리즘을 개발했습니다.
* **최첨단 다중 스케일 계층 분할기**: 다중 스케일 정보를 효과적으로 통합하여 고품질의 계층적 이미지 분할을 수행하는 분할기를 제안했습니다.
* **조합 그룹화 알고리즘**: 다중 스케일 영역의 조합 공간(combinatorial space)을 효율적으로 탐색하여 정확한 객체 제안을 생성하는 그룹화 전략을 제시했습니다.
* **`SCG` (Single-scale Combinatorial Grouping) 도입**: MCG의 더 빠른 버전으로, 경쟁력 있는 제안을 이미지당 5초 이내에 생성합니다.
* **광범위한 실험적 검증**: `BSDS500`, `SegVOC12`, `SBD`, `COCO` 데이터셋에서 `MCG`가 최첨단 윤곽선, 계층적 영역 및 객체 제안을 생성함을 입증했습니다.

## 📎 Related Works

* **고속 정규화된 컷**: Taylor [23]의 watershed oversegmentation, Maire와 Yu [24]의 multigrid solver와 같은 기존 기법들과 비교하여 효율성을 강조했습니다.
* **객체 제안 (창 기반)**: Alexe et al. [16]의 `Objectness`, Manen et al. [25]의 `Randomized Prim's`, Zitnick et al. [26]의 `Edge Boxes`, Cheng et al. [27]의 `BING`과 같은 방법론들과 차별점을 두어 픽셀 단위의 정확한 객체 추출에 중점을 둡니다.
* **객체 제안 (분할 기반)**: Carreira와 Sminchisescu [18]의 `CPMC`, Endres와 Hoiem [19]의 `Category-independent object proposals`, Kim과 Grauman [17]의 `Shape sharing`, Uijlings et al. [12]의 `Selective Search`와 같은 연구들과 비교하여 다중 스케일 정보 활용의 장점을 설명합니다.
* **조합 그룹화**: Malisiewicz와 Efros [4], Arbeláez et al. [8]의 초기 조합 그룹화 시도에서 발전하여 더 큰 조합 공간을 효율적으로 탐색합니다.
* **최근 고속 제안**: Krähenbühl와 Koltun [13]의 `GOP`, Rantalankila et al. [14]의 `GLS`, Humayun et al. [15]의 `RIGOR`와 같은 방법론들과 품질 및 속도 측면에서 경쟁력을 보입니다.

## 🛠️ Methodology

1. **고속 다운샘플링된 고유 벡터 계산 (`DNCuts`)**:
    * 대규모 어피니티 행렬(affinity matrix) $A$의 고유 벡터 계산 비용을 줄이기 위해, 이미지 피라미드(image pyramid) 기반의 스케일 유사성(scale-similar nature)을 활용합니다.
    * `A`를 직접 계산하는 대신, `A`의 제곱 `A^2`의 다운샘플링된 버전 `A^2[i, i]`의 고유 벡터를 계산한 후 업샘플링하여 원래 이미지 공간으로 되돌립니다.
    * 알고리즘은 `A`를 여러 번 반복적으로 제곱하고 다운샘플링하여 효율성을 높입니다.
2. **분할 계층 정렬**:
    * 독립적으로 계산된 다양한 이미지 해상도의 분할 계층을 결합하기 위해 정렬(alignment) 기법을 사용합니다.
    * 한 분할 $R = \{R_i\}_i$의 경계를 다른 대상 분할 $S = \{S_j\}_j$에 "스냅(snap)"하기 위해, $S_j \in S$ 영역의 새로운 레이블 $L(S_j)$을 $R$에서 해당 픽셀들의 다수 레이블로 정의합니다:
        $$L(S_{j}) = \text{arg max}_{i} \frac{|S_{j} \cap R_{i}|}{|S_{j}|}$$
    * 이 과정을 통해 모든 레벨이 동일한 대상 분할에 투영되어 계층 구조가 유지됩니다.
3. **다중 스케일 계층 분할**:
    * **단일 스케일 분할**: 밝기, 색상, 질감 차이, 희소 코딩(sparse coding) 패치, 구조화된 포레스트 윤곽선과 같은 지역 윤곽선 신호(local contour cues)를 `DNCuts`로 전역화(globalize)하고, 이를 선형적으로 결합하여 UCM(Ultrametric Contour Map)을 구성합니다. Segmentation Covering 메트릭을 최적화하여 계층을 학습합니다.
    * **계층 정렬**: 원본 이미지를 서브샘플링/슈퍼샘플링하여 $N$개의 스케일을 가진 다중 해상도 피라미드를 구성합니다. 각 스케일에서 독립적으로 생성된 UCM을 가장 높은 해상도의 슈퍼픽셀에 재귀적으로 투영하여 정렬합니다.
    * **다중 스케일 계층**: 정렬된 $N$개의 스케일에서 얻은 경계 강도(boundary strength)를 결합하여 단일 경계 확률 추정치를 생성합니다. 단순한 균일 가중치(uniform weights)와 플랫(Platt)의 방법을 사용한 확률 변환 방식을 채택합니다.
4. **객체 제안 생성**:
    * **디스크립터 고속 계산**: 영역 트리(region tree) 표현을 활용하여 바운딩 박스(bounding box), 둘레(perimeter), 이웃(neighbors) 등과 같은 영역 디스크립터를 효율적으로 계산합니다 (단일 이미지 스캔으로 $O(p+m)$).
    * **조합 그룹화**: 계층적 분할에서 $n$-튜플(예: 단일 영역, 두 영역, 세 영역, 네 영역)을 조합하여 완전한 객체를 나타낼 가능성이 있는 영역 집합을 탐색합니다. UCM 강도 임계값까지 영역 트리를 상위에서 하위로 탐색합니다.
    * **파레토 프론트 최적화를 통한 파라미터 학습**: 여러 순위 목록에서 제안을 결합하는 문제의 높은 계산 복잡도(지수적)를 줄이기 위해, 파레토 프론트(Pareto front) 최적화 [39], [40]를 사용하여 제안의 수와 달성 가능한 품질 간의 최적의 트레이드오프를 찾습니다. 이는 복잡도를 $(R-1)S^2$로 줄입니다.
    * **회귀 기반 제안 순위화**: 저수준 특징(크기, 위치, 모양, 윤곽선 속성)을 사용하여 객체 중첩(object overlap)을 회귀(regress)하도록 랜덤 포레스트(Random Forest)를 학습시키고, `Maximum Marginal Relevance` 측정값을 기반으로 순위 다양성을 높입니다.

## 📊 Results

* **BSDS500 데이터셋**:
  * 단일 스케일 분할기와 다중 스케일 분할기 모두 윤곽선 및 계층적 분할에서 최첨단 성능을 달성했습니다. `DNCuts`는 기존 방식 대비 20배 빠른 속도를 보였습니다.
  * 객체 및 부분(parts) 평가에서 `MCG`는 `SCG`보다 상당한 개선을 보였습니다.
* **PASCAL VOC12, SBD, COCO 데이터셋 (객체 제안)**:
  * `MCG`는 `SegVOC12`, `SBD`, `COCO`의 모든 레짐(regime)에서 기존의 분할된 객체 제안(segmented object proposals) 방법론들을 크게 능가하며 최첨단 성능을 달성했습니다.
  * 특히, `MCG`는 높은 Jaccard 임계값(예: $J=0.7, J=0.85$)에서 가장 높은 리콜(recall)을 달성하여 고정밀 객체 제안에 강점을 보였습니다.
  * `SCG`는 `MCG`보다 7배 빠르면서도 매우 경쟁력 있는 결과를 제공합니다.
* **바운딩 박스 제안(Bounding-Box Proposals)**:
  * 분할 제안에 맞춰진 방법임에도 불구하고, `MCG`는 바운딩 박스 제안에서도 기존 최첨단 방법들을 능가하며, 특히 높은 정밀도(precise localization)에서 뛰어난 성능을 보였습니다.
* **효율성**:
  * 이미지당 총 시간: `MCG`는 약 42.2초, `SCG`는 약 6.2초 (단일 코어 Linux 환경).
  * `MCG`는 뛰어난 품질을 유지하면서도 다른 최첨단 분할 제안 방법들과 비교하여 경쟁력 있는 처리 속도를 보였습니다.

## 🧠 Insights & Discussion

* **다중 스케일 통합의 중요성**: `MCG`는 이미지 분할 및 객체 제안 생성 과정 전반에 걸쳐 다중 스케일 정보를 성공적으로 통합하여 성능 향상을 이끌어냈습니다. 특히 다양한 크기의 객체를 포착하는 데 효과적입니다.
* **조합 그룹화의 위력**: 계층적 분할 내의 영역들을 조합하는 효율적인 전략이 단일 영역만으로는 포착하기 어려운 완전한 객체를 찾아내는 데 중요합니다. 이를 통해 저수준 특징만으로 구성된 계층의 한계를 극복했습니다.
* **파레토 프론트 최적화의 효율성**: 제안 목록 조합의 복잡성을 효과적으로 관리하며, 제안의 수와 품질 사이의 최적의 트레이드오프를 찾도록 하는 파레토 프론트 학습 전략은 실용적인 이점을 제공합니다.
* **우수한 일반화 능력**: `MCG`와 `SCG`는 `SegVOC12`에서 학습된 파라미터를 사용하여 `SBD`와 `COCO` 같은 미지의 데이터셋에서 강력한 일반화 성능을 보여주었습니다. 이는 재학습 없이도 다양한 애플리케이션에 적용 가능함을 시사합니다.
* **실용적 유연성**: `MCG`는 순위가 매겨진 제안 목록을 제공하므로, 사용자는 특정 애플리케이션의 요구사항(정밀도 또는 속도)에 따라 제안의 수를 유연하게 선택할 수 있으며, 재파라미터화(re-parameterization)가 필요 없습니다.
* **한계 및 향후 연구**: 4-튜플 이상의 복잡한 조합은 제안 품질에 미미한 추가 개선만을 가져왔으며, 이는 계산 비용 증가와 비교하여 효율성의 한계를 보였습니다.

## 📌 TL;DR

`MCG`는 고속 정규화된 컷, 다중 스케일 계층 분할, 효율적인 조합 그룹화를 통합하여 이미지 분할 및 객체 제안을 위한 통일된 접근 방식을 제안합니다. 이는 `BSDS500`, `SegVOC12`, `SBD`, `COCO` 데이터셋에서 최첨단 성능을 달성하며, 특히 높은 정밀도에서 뛰어난 객체 제안 품질과 우수한 일반화 능력을 보여줍니다. `SCG`는 더 빠른 속도로 경쟁력 있는 결과를 제공하는 버전입니다.
