# BAISeg: Boundary Assisted Weakly Supervised Instance Segmentation

Tengbo Wang, Yu Bai (2024)

## 🧩 Problem to Solve

본 논문은 인스턴스 수준의 감독(instance-level supervision) 없이 인스턴스 마스크를 추출해야 하는 약지도 인스턴스 분할(Weakly Supervised Instance Segmentation, WSIS) 문제를 다룬다. 기존의 WSIS 방법들은 주로 픽셀 간의 관계를 학습하여 변위 필드(Displacement Field, DF)를 추정하고, 이를 기반으로 클러스터링을 수행하여 인스턴스의 중심점(centroid)을 찾아내는 방식을 사용한다.

그러나 이러한 방식은 다음과 같은 결정적인 한계가 있다. 첫째, 클러스터링을 통해 식별된 인스턴스 중심점은 본질적으로 불안정하며, 사용하는 클러스터링 알고리즘에 따라 결과가 크게 달라진다. 둘째, 서로 가까이 위치한 동일 클래스의 인스턴스들을 하나의 중심점으로 오인하거나, 가려진 인스턴스의 중심점을 누락하는 경우가 빈번하다.

따라서 본 논문의 목표는 중심점 추정 방식의 불안정성을 극복하기 위해, 픽셀 수준의 어노테이션(pixel-level annotations)만을 활용하여 인스턴스 경계(boundary)를 예측함으로써 인스턴스 분할을 수행하는 새로운 패러다임인 BAISeg를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스를 식별하기 위해 '중심점'이 아닌 '클래스 불가지론적 인스턴스 경계(class-agnostic instance boundaries)'를 예측하는 것이다. 주요 기여 사항은 다음과 같다.

1. **새로운 WSIS 패러다임 제안**: 인스턴스 수준의 어노테이션 없이 픽셀 수준의 어노테이션만으로 인스턴스 마스크를 추출하는 BAISeg 프레임워크를 제안한다. 이는 기존의 제안(proposal) 알고리즘이나 중심점 추정에 의존하던 한계를 탈피한다.
2. **Cascade Fusion Module (CFM) 및 Deep Mutual Attention (DMA) 설계**: IABD(Instance-Aware Boundary Detection) 브랜치 내에서 독립적이고 다중 스케일의 엣지 특징을 학습하기 위해 CFM을 설계하였으며, 특히 반응이 약한 인스턴스 경계 정보를 캡처하기 위해 DMA를 도입하였다.
3. **Pixel-to-Pixel Contrast (PPC) 도입**: 인스턴스 경계의 연속성과 폐쇄성(closedness)을 강화하고 시맨틱 드리프트(semantic drift) 문제를 해결하기 위해 가중치 기반의 픽셀 간 대조 학습을 적용하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **인스턴스 엣지 검출 (Instance Edge Detection)**: SharpContour, STEAL, InstanceCut 등이 엣지 정보를 활용해 마스크를 정교화하려 시도했다. 특히 InstanceCut은 시맨틱 분할과 인스턴스 경계를 통해 최적의 파티션을 유도하지만, 본 논문과 달리 더 강한 제약 조건(MultiCut 설정 등)에 의존한다.
2. **약지도 인스턴스 분할 (WSIS)**:
    - **Top-down 방식**: 이미지 수준 또는 박스 수준의 어노테이션을 사용하여 영역을 먼저 제안하고 마스크를 추출한다. 하지만 이는 제안 알고리즘의 성능에 크게 의존한다.
    - **Bottom-up 방식**: 픽셀 간 관계를 학습하여 DF를 생성하고 클러스터링을 수행한다. 앞서 언급했듯 중심점 추정의 불안정성이 가장 큰 문제이다.
3. **대조 학습 (Contrastive Learning)**: InfoNCE와 같은 손실 함수를 통해 유사한 샘플은 가깝게, 다른 샘플은 멀게 배치하여 표현력을 높인다. 최근에는 픽셀 단위의 예측 작업에 적용되어 클래스 내 응집도와 클래스 간 분별력을 높이는 데 사용된다.

### 차별점

BAISeg는 중심점 추정이나 외부 제안 알고리즘에 의존하지 않고, 오직 픽셀 수준의 시맨틱 마스크에서 유도된 경계 정보를 통해 인스턴스를 분리한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

BAISeg는 공유 백본(Backbone, 예: HRNet-W48)을 가진 두 개의 병렬 브랜치로 구성된다.

1. **Instance-Aware Boundary Detection (IABD) 브랜치**: 클래스에 관계없이 인스턴스 간의 경계 맵(Boundary map)을 예측한다.
2. **Semantic Segmentation 브랜치**: 이미지의 시맨틱 맵(Semantic map)을 예측한다.

최종적으로 이 두 결과물을 결합하여 인스턴스 분할 결과를 도출한다.

### 주요 구성 요소 및 작동 원리

#### 1. Instance-Aware Boundary Detection Module (IABM)

IABD 브랜치는 $R$개의 **Cascade Fusion Module (CFM)**과 $1 \times 1$ 합성곱 층으로 구성된다.

- **CFM**: 시맨틱 분할 브랜치와의 결합도를 낮추어 독립적인 엣지 특징을 학습하게 하며, 단계적으로 특징을 정제하여 다중 스케일의 경계 정보를 캡처한다.
- **Deep Mutual Attention (DMA)**: CFM의 하위 모듈로, 채널 주의 집중(Channel-wise Attention)과 공간 주의 집중(Spatial-wise Attention)을 결합한 **Mutual Attention Unit (MAU)**을 사용하여 약한 응답을 보이는 경계 정보를 강화한다.
- **MAU의 수식**:
  입력 특징 $f_n''$에 대해 채널 주의 집중 $F_c$와 공간 주의 집중 $F_s$를 계산하고, 이를 다음과 같이 결합한다.
  $$F_{att} = F_h + (f_n'' \times F_c + f_n'' \times F_s) \times \beta$$
  여기서 $F_h$는 identity transformation 결과이며, $\beta$는 학습 가능한 주의 집중 가중치이다.

#### 2. Pixel-to-Pixel Contrast (PPC)

경계가 끊어지거나 닫히지 않아 인스턴스가 병합되는 문제를 막기 위해 도입되었다. 시맨틱 경계 라벨을 이용해 샘플링하며, 배경 클래스의 오라벨 영향을 줄이기 위해 가중치 기반의 대조 손실 함수를 사용한다.
$$L_P = \frac{1}{|P_i|} \sum_{i^+ \in P_i} -\alpha \log \frac{\exp(i \cdot i^+ / \tau)}{\exp(i \cdot i^+ / \tau) + (1-\alpha) \sum_{i^- \in N_i} \exp(i \cdot i^- / \tau)}$$
여기서 $\alpha$는 경계 픽셀의 비율을 나타내는 가중치이며, $\tau$는 온도 하이퍼파라미터이다.

#### 3. 손실 함수 (Loss Functions)

전체 모델은 다음 세 가지 손실 함수의 가중 합으로 최적화된다.
$$L_{Overall} = \alpha \times L_P + \beta \times L_B + \gamma \times L_S$$

- $L_P$: Pixel-to-Pixel Contrast Loss
- $L_B$: 경계 예측을 위한 가중 교차 엔트로피 손실 (HED 방식)
- $L_S$: 시맨틱 분할을 위한 손실 (DeepLabv3 방식)

#### 4. Mask Extraction Pipeline (MEP)

예측된 경계 맵에서 실제 마스크를 추출하는 과정은 3단계로 진행된다.

- **Stage 1**: NMS(Non-Maximum Suppression)를 적용하여 경계선을 얇게 만든다.
- **Stage 2**: 팽창(Dilate) $\rightarrow$ 거리 변환(Distance Transform) $\rightarrow$ 임계값 처리 $\rightarrow$ CCL(Connected-Component Labeling)을 통해 라벨 맵을 생성한다.
- **Stage 3**: 원본 이미지와 라벨 맵을 **Watershed 알고리즘**에 입력하여 클래스 불가지론적 인스턴스 마스크를 생성한다.
- **Refinement**: 생성된 마스크의 구멍을 메우기 위해 Closing 연산을 수행하고, 시맨틱 맵을 필터로 사용하여 시맨틱 영역 외부로 확산된 픽셀을 제거한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PASCAL VOC 2012, MS COCO 2017.
- **지표**: mAP (VOC 2012: $mAP_{25}, mAP_{50}, mAP_{70}, mAP_{75}$ / COCO: $AP, AP_{50}, AP_{75}$).
- **백본**: HRNet-W48.

### 주요 결과

1. **PASCAL VOC 2012**:
   - GT 시맨틱 마스크를 사용한 경우(`BAISeg (Semanticgd)`) $mAP_{50}$ 기준 $48.4\%$를 달성하였다.
   - Mask R-CNN으로 정제한 후에는 $62.0\% mAP_{50}$까지 성능이 향상되어, 이미지 수준 또는 포인트 수준의 감독을 사용하는 기존 WSIS 방법들보다 우수한 성능을 보였다.
2. **MS COCO 2017**:
   - Test-Dev 세트에서 $AP_{50}$ 기준 $33.6\%$를 기록하였다. 이는 최고의 이미지 수준 WSIS 방법 대비 약 $3.9\%$ 향상된 수치이다.
   - 다만, 강력한 위치 감독(박스 수준 등)을 사용하는 Box2Mask나 Mask R-CNN보다는 낮은 성능을 보였다.

### 절제 연구 (Ablation Study)

- **CFM/DMA/PPC 영향**: CFM 도입 시 $mAP_{50}$이 $1.9\%$ 상승하며, DMA는 $1.7\%$, PPC는 $1.1\%$의 성능 향상을 가져온다. 모든 모듈을 결합했을 때 최적의 성능($48.4\% mAP_{50}$)을 나타냈다.
- **백본 영향**: HRNet-W48이 ResNet 계열보다 우수한 성능을 보였으며, 이는 고해상도 표현 유지 능력이 경계 검출에 중요함을 시사한다.

## 🧠 Insights & Discussion

### 강점

BAISeg는 중심점 추정이라는 불안정한 경로를 완전히 배제하고 경계 검출이라는 새로운 경로를 제시함으로써 WSIS의 고질적인 문제인 '중심점 불안정성'을 효과적으로 해결하였다. 특히 픽셀 수준의 어노테이션만으로도 경쟁력 있는 성능을 낸다는 점이 고무적이다.

### 한계 및 비판적 해석

1. **경계 품질 의존성**: 논문에서도 언급되었듯, 시맨틱 라벨에서 유도된 경계가 거칠기 때문에 최종 마스크의 정밀도가 제한된다.
2. **경계 폐쇄성 문제**: 경계가 완전히 닫히지 않을 경우 Watershed 알고리즘 특성상 인스턴스가 서로 병합되는 'leakage' 현상이 발생한다. 이는 PPC를 통해 완화하려 했으나 완전히 해결되지 않은 것으로 보인다.
3. **시맨틱 맵 의존성**: 시맨틱 분할 결과에 노이즈가 있을 경우 인스턴스 마스크 생성 단계에서 그대로 오류가 전파되는 구조적 취약성이 있다.

## 📌 TL;DR

본 논문은 중심점 추정 방식의 불안정성을 극복하기 위해 **인스턴스 경계 검출** 기반의 새로운 약지도 인스턴스 분할(WSIS) 프레임워크인 **BAISeg**를 제안한다. 픽셀 수준의 어노테이션만을 사용하며, CFM, DMA, PPC 모듈을 통해 정교한 경계 맵을 생성하고 이를 Watershed 알고리즘으로 마스크화한다. 이 연구는 인스턴스 수준의 고비용 라벨 없이도 효과적인 분할이 가능함을 입증하였으며, 향후 파놉틱 분할(Panoptic Segmentation) 등 더 복잡한 작업으로 확장될 가능성이 크다.
