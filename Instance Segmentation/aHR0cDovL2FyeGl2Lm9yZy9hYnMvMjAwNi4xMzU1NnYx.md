# NINEPINS: Nuclei Instance Segmentation with Point Annotations

Ting-An Yen, Hung-Chun Hsu, Pushpak Pati, Maria Gabrani, Antonio Foncubierta-Rodríguez, Pau-Choo Chung (2020)

## 🧩 Problem to Solve

디지털 병리학(Digital Pathology) 분야에서 세포핵 분할(Nuclei Segmentation)은 종양 등급 결정이나 조직 분류와 같은 고차원적인 분석을 수행하기 위한 필수적인 전처리 단계이다. 일반적으로 딥러닝 기반의 분할 모델은 높은 정확도를 달성하기 위해 대량의 정밀하게 어노테이션된 데이터셋을 필요로 한다.

그러나 세포핵의 경계는 식별하기 어렵고 세포의 외형이 다양하기 때문에, 픽셀 단위의 마스크(Mask) 어노테이션을 생성하려면 숙련된 병리 전문가의 많은 시간과 비용이 소모된다는 심각한 문제가 있다. 본 논문의 목표는 마스크 어노테이션 대신 상대적으로 생성 비용이 매우 낮은 포인트 어노테이션(Point Annotations)만을 사용하여 효과적인 인스턴스 분할(Instance Segmentation) 모델을 학습시키는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 포인트 어노테이션으로부터 자동 생성된 유사 라벨(Pseudo-label)을 활용하여 인스턴스 분할 모델인 HoVer-Net을 학습시키는 semi-supervised learning 프레임워크인 NINEPINS를 설계하는 것이다.

단순히 포인트 데이터를 사용하는 것에 그치지 않고, 기존의 유사 라벨 생성 방식의 한계를 극복하기 위해 유사 라벨 정제(Refinement) 과정을 도입하고, HoVer-Net의 핵심인 Distance Map 학습을 위해 포인트 및 유사 라벨로부터 거리 맵 라벨을 생성하는 방법론을 제안하였다. 또한, 이러한 약한 감독(Weakly-supervised) 방식의 학습이 실제 하위 작업인 조직 분류(Tissue Classification) 성능에 미치는 영향을 분석하여 실용성을 입증하였다.

## 📎 Related Works

기존 연구인 Hui et al. [11]은 포인트 어노테이션을 사용하여 마스크를 생성하는 약한 감독 학습 방식을 제안하였으나, 이는 이진 분할(Binary Segmentation) 결과만을 제공하므로 세포들이 서로 겹쳐 있는 경우 개별 인스턴스를 구분하지 못하는 한계가 있다. PseudoEdgeNet [14] 역시 포인트 어노테이션으로 학습하지만, 이 역시 인스턴스 분할보다는 이진 분할에 가깝다.

반면, HoVer-Net [4]은 분할 마스크와 Distance Map을 동시에 예측하여 정밀한 인스턴스 분할을 수행한다. 하지만 HoVer-Net을 학습시키기 위해서는 정밀한 인스턴스 마스크로부터 생성된 Distance Map 라벨이 필수적이다. NINEPINS는 이러한 HoVer-Net의 강력한 성능과 포인트 어노테이션의 낮은 비용이라는 두 가지 장점을 결합하기 위해, 포인트 데이터로부터 HoVer-Net 학습에 필요한 모든 유사 라벨을 생성하는 접근 방식을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조

NINEPINS는 기본적으로 HoVer-Net 아키텍처를 기반으로 한다. 모델은 ResNet-50 구조의 Encoder와 여러 개의 Decoder branch로 구성된다. 본 연구에서는 주로 분할(Segmentation)을 위한 디코더와 Distance Map 예측을 위한 디코더 두 가지를 사용하며, 필요에 따라 핵의 중심점(Nuclear Centroid)을 탐지하는 세 번째 디코더 브랜치를 추가하여 학습한다.

### 유사 라벨 정제 (Pseudo-label Refinement)

기존 연구[11]에서 Voronoi Diagram과 K-means clustering을 통해 생성된 유사 라벨은 세포의 영역을 정확히 묘사하지 못하거나 과대평가하는 경향이 있다. 이를 개선하기 위해 다음과 같은 정제 알고리즘을 수행한다.

1. **불필요한 컴포넌트 제거**: 유사 라벨 $\text{Ps}$ 내의 연결 성분(Connected Component) $\text{CC}$가 확장된 포인트 $\text{dilate}(\text{P}, 5)$를 포함하지 않는 경우, 해당 성분을 제거한다.
2. **누락된 영역 추가**: 포인트 라벨 $\text{P}$ 중 유사 라벨 $\text{Ps}$에 의해 커버되지 않는 점 $p$가 있다면, $\text{dilate}(p, 3)$을 통해 마스크 영역을 추가한다.

여기서 $\text{dilate}(\text{A}, b)$는 반지름 $b$인 원형 구조 요소로 객체를 확장하는 연산이며, $|\text{M}|$은 영역의 넓이를 의미한다.

### Distance Map Labels 생성

HoVer-Net의 Distance Map 브랜치를 학습시키기 위해서는 각 픽셀에서 가장 가까운 인스턴스 경계까지의 거리를 나타내는 라벨이 필요하다. NINEPINS에서는 포인트 라벨과 위에서 정제된 유사 라벨을 기반으로 Voronoi Diagram의 엣지(Edge)를 활용하여 유사 인스턴스 라벨을 분리하고, 이를 통해 Distance Map 라벨을 생성하여 학습에 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CoNSeP, MoNuSeg 데이터셋을 사용하여 분할 성능을 평가하였으며, Camelyon16 데이터셋을 사용하여 조직 분류 성능을 평가하였다.
- **평가 지표**:
  - $\text{DICE}$: 분할 품질을 측정하는 지표로 다음과 같이 정의된다.
    $$\text{DICE} = \frac{2 \times |X \cap Y|}{|X| + |Y|}$$
  - $\text{DQ}_{\text{point}}$: 탐지 품질을 측정하는 지표로, 예측된 세그먼트가 단일한 정답 중심점(Centroid)을 포함하는지 측정한다.
    $$\text{DQ} = \frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$

### 주요 결과

1. **어노테이션 비용과 성능의 트레이드오프**: MoNuSeg 데이터셋에서 정밀 마스크 라벨과 유사 라벨의 비율을 조절하며 실험한 결과, 전체 데이터의 70%를 유사 라벨로 대체하더라도 $\text{DICE}$ 및 $\text{DQ}_{\text{point}}$ 지표의 하락폭이 약 6% 내외로 매우 적었다. 이는 30%의 데이터만 정밀하게 어노테이션하고 나머지를 포인트 어노테이션으로 대체함으로써 효율적으로 학습이 가능함을 시사한다.
2. **포인트 라벨 부정확성에 대한 강건성**: 포인트 라벨을 가우시안 분포를 이용하여 무작위로 이동(Shift)시키는 실험을 진행하였다.
    $$P_X \sim N(0, (\epsilon D_X/3)^2), \quad P_Y \sim N(0, (\epsilon D_Y/3)^2)$$
    실험 결과, $\epsilon \le 1$ 범위 내에서 $\text{DICE}$ 점수는 매우 안정적으로 유지되었으며, $\text{DQ}_{\text{point}}$ 역시 소폭 변동(약 0.03)하는 수준으로, 포인트 어노테이션의 약간의 위치 오차에 대해 모델이 강건함을 보였다.
3. **고차원 작업(조직 분류)에 미치는 영향**: CGC-Net을 이용해 Camelyon16 데이터셋으로 종양 여부를 분류한 결과, 정밀 마스크로 학습한 HoVer-Net(평균 93.51%)과 NINEPINS(평균 91.60%) 사이의 정확도 차이가 크지 않았다. 이는 분할 성능의 수치적 하락이 반드시 최종 분석 작업의 성능 저하로 이어지지는 않음을 보여준다.

## 🧠 Insights & Discussion

본 논문은 정밀한 픽셀 단위의 어노테이션 없이도 포인트 데이터만으로 충분히 실용적인 인스턴스 분할 모델을 구축할 수 있음을 증명하였다. 특히, 분할 지표($\text{DICE}$, $\text{DQ}$)에서의 약간의 성능 저하가 실제 진단 작업인 조직 분류 정확도에는 결정적인 영향을 미치지 않는다는 점은 매우 중요한 통찰이다. 이는 의료 영상 분석에서 전문가의 노동력을 획기적으로 줄이면서도 임상적으로 유의미한 결과를 얻을 수 있는 가능성을 제시한다.

다만, Distance Map의 생성 과정이 Voronoi Diagram이라는 기하학적 구조에 의존하고 있어, 포인트의 분포가 매우 불균일하거나 극단적으로 부정확할 경우 $\text{DQ}_{\text{point}}$ 지표가 하락하는 경향이 관찰되었다. 이는 포인트 데이터의 품질이 일정 수준 이상으로 유지되어야 함을 의미한다.

## 📌 TL;DR

NINEPINS는 숙련된 전문가의 비용이 많이 드는 마스크 어노테이션 대신, 단순한 포인트 어노테이션으로부터 유사 라벨을 생성하여 HoVer-Net을 학습시키는 프레임워크이다. 실험을 통해 포인트 라벨의 오차에 대해 강건하며, 분할 성능이 소폭 하락하더라도 최종적인 조직 분류 성능에는 큰 영향이 없음을 확인하였다. 이 연구는 향후 의료 영상 분야에서 데이터 라벨링의 부담을 줄이고 자동화된 약한 감독 학습(Weakly-supervised learning)을 적용하는 데 중요한 기반이 될 수 있다.
