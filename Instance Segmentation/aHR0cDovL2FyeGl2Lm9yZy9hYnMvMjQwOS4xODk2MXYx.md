# ProMerge: Prompt and Merge for Unsupervised Instance Segmentation

Dylan Li, Gyungin Shin (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 인간이 작성한 레이블 데이터 없이 이미지 내의 서로 다른 객체 인스턴스를 분할하는 **Unsupervised Instance Segmentation**이다. 인스턴스 분할은 자율 주행 시스템이나 의료 영상 분석과 같이 정밀한 픽셀 단위의 위치 파악이 필요한 분야에서 매우 중요하다. 그러나 모든 객체에 대해 정밀한 마스크(mask)를 수동으로 생성하는 것은 비용이 매우 높으며, 특히 전문 지식이 필요한 의료 분야에서는 더욱 어렵다.

최근 DINO와 같은 자기지도학습(self-supervised) 모델의 풍부한 시각적 특징 표현(visual feature representation) 덕분에 큰 진전이 있었다. 특히 이미지를 그래프로 표현하고 Generalized eigenvalue system(즉, normalized-cut)을 해결하여 전경 마스크를 생성하는 방식이 성능 면에서 우수함을 보였다. 하지만 이러한 방식은 다음과 같은 한계가 있다.
1. **계산 비용 및 속도**: eigenvalue 문제를 반복적으로 해결해야 하므로 추론 속도가 매우 느리다.
2. **고정된 마스크 개수**: 이미지당 생성할 마스크의 개수를 고정된 기준(예: MaskCut의 경우 3개)으로 결정하는 경우가 많아, 객체가 밀집된 복잡한 장면에서 모든 객체를 포착하지 못하는 한계가 있다.

따라서 본 논문의 목표는 계산 효율성을 높이면서도, 이미지 내 객체의 수에 유연하게 대응할 수 있는 새로운 비지도 인스턴스 분할 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Prompt and Merge**라는 단순하지만 효과적인 전략이다. 복잡한 eigenvalue 계산 대신, 자기지도학습 모델의 특징 공간에서 직접적인 유사도를 기반으로 초기 마스크를 생성하고, 이를 전략적으로 병합하는 방식을 취한다.

주요 기여 사항은 다음과 같다.
- **ProMerge 프레임워크 제안**: point-prompting을 이용한 초기 마스크 생성, 배경 기반의 정교한 마스크 프루닝(mask pruning), 그리고 픽셀 및 특징 공간의 유사도를 고려한 반복적 마스크 병합으로 구성된 파이프라인을 제안한다.
- **효율성 및 성능 입증**: 기존의 normalized-cut 기반 방식보다 추론 속도를 약 3.6배 향상시키면서도, 6개의 벤치마크 데이터셋에서 경쟁력 있는 성능을 보였다.
- **Pseudo-label 생성기의 가능성 확인**: ProMerge를 통해 생성한 마스크를 pseudo-label로 사용하여 객체 검출기(Object Detector)를 학습시킨 결과, 기존의 최신 비지도 모델(CutLER)보다 뛰어난 성능을 달성하였다.

## 📎 Related Works

본 연구는 다음 세 가지 분야와 밀접하게 연관되어 있다.

1. **Self-supervised Visual Representation Learning**: DINO와 같은 모델은 지도 학습 모델보다 더 세밀한 세그멘테이션 정보를 인코딩하는 경향이 있다. ProMerge는 이러한 DINO의 내재적인 그룹화 능력을 활용한다.
2. **Unsupervised Single Object Detection/Segmentation**: LOST와 같은 연구는 전경-배경 분리를 위해 seed expansion이나 normalized-cut 방식을 사용한다. 그러나 이들은 주로 이미지 내 단일 지배적 객체를 찾는 데 집중하여, 다중 객체 로컬라이제이션 작업에는 한계가 있다.
3. **Unsupervised Instance Segmentation**: 최근 MaskCut과 같은 방식은 normalized-cut 기반의 단일 객체 분할 기법을 반복 적용하여 다중 마스크를 생성한다. 하지만 앞서 언급했듯이 높은 계산 비용과 고정된 마스크 개수 제한이라는 치명적인 단점이 존재한다. ProMerge는 이러한 계산 복잡도를 제거하고 마스크 개수의 제한을 없앰으로써 차별화를 꾀한다.

## 🛠️ Methodology

ProMerge의 전체 파이프라인은 초기 마스크 생성, 배경 기반 프루닝, 마스크 병합의 세 단계로 구성된다.

### 1. Point-Prompting Visual Features (초기 마스크 생성)
이미지 인코더(DINO ViT)를 통해 추출된 패치 토큰 $F = \{f_{ij} \in \mathbb{R}^c\}$를 입력으로 사용한다. 2D 그리드 형태로 $K$개의 균등하게 배치된 패치 토큰을 선택하여 prompt tokens $P = \{p_l \in \mathbb{R}^c | l=1, \dots, K\}$를 구성한다. 각 prompt token $p_l$과 이미지 내 모든 패치 토큰 $f_{ij}$ 간의 코사인 유사도를 계산하여 어피니티 행렬(affinity matrix) $A^l$을 생성한다.

$$A^l = (A^l_{ij}) = \frac{p_l \cdot f_{ij}}{\|p_l\|_2 \|f_{ij}\|_2}$$

이후 이진 임계값 $\tau_b$를 적용하여 각 seed point에 대응하는 이진 마스크 $M^l$을 생성한다.

### 2. Background-based Mask Pruning (배경 기반 프루닝)
초기 생성된 마스크에는 노이즈가 섞인 배경 마스크가 많으므로, 병합 전 이를 제거하는 과정이 필수적이다.

- **Background Aggregation**: 마스크의 가장자리(edge) 중 두 곳 이상에서 픽셀의 절반 이상이 양수 값을 가지면 배경 마스크 후보로 분류한다. 이후 픽셀 단위 투표(pixel-wise voting)를 통해 단일 대표 배경 마스크 $\tilde{M}_{bg}$를 생성한다.
$$\tilde{M}_{bg_{ij}} = \left[ \frac{\sum_{l=1}^{|B|} M^{bg}_{l;ij}}{|B|} > 0.5 \right]$$
- **Cascade Filtering**: 마스크를 면적 기준 오름차순으로 정렬한 뒤 순차적으로 처리한다. 현재 마스크에서 이전 단계까지 고려되지 않은 '새로운' 픽셀들이 배경 마스크 $\tilde{M}_{bg}$와 IoA(Intersection-over-Area)가 높거나, 특징 유사도가 높으면 해당 마스크를 배경으로 간주하여 제외한다.

### 3. Merging Prompted Masks (마스크 병합)
프루닝된 마스크들을 면적 기준 내림차순으로 정렬하여 반복적으로 병합한다. 작은 마스크가 이미 처리된 더 큰 마스크와 다음 조건 중 하나라도 만족하면 병합한다.
- **픽셀 공간 조건**: 두 마스크의 IoA가 임계값 $\tau_{merge}^{IoA}$를 초과하는 경우.
- **특징 공간 조건**: 두 마스크의 평균 패치 임베딩 간 코사인 유사도가 $\tau_{merge}^{f}$를 초과하는 경우.

### 4. ProMerge+ (Detector 학습)
ProMerge를 통해 생성한 마스크를 pseudo-label로 사용하여 Cascade Mask R-CNN을 학습시킨다. 이는 노이즈가 섞인 pseudo-label로부터 더 견고한 모델을 학습시키고, 다른 데이터 분포로의 일반화 성능을 평가하기 위함이다.

## 📊 Results

### 실험 설정
- **데이터셋**: COCO2017, COCO-20K, LVIS, KITTI, Objects365, SA-1B 등 6개 벤치마크 사용.
- **지표**: Average Precision (AP) 및 Average Recall (AR)을 사용하여 정량적 성능 평가.
- **비교 대상**: TokenCut, MaskCut (Training-free 방식) 및 CutLER (Pseudo-label 학습 방식).

### 주요 결과
1. **정량적 성능**: Training-free 설정에서 ProMerge는 MaskCut보다 높은 AP와 AR을 기록하였다. 특히 마스크 개수의 제한이 없기 때문에 Recall 성능이 눈에 띄게 향상되었다.
2. **추론 속도**: ProMerge는 $0.54\text{ FPS}$를 기록하여, MaskCut($0.15\text{ FPS}$)보다 약 3.6배 빠르다. 이는 복잡한 eigenvalue 계산을 생략했기 때문이며, 특히 prompt token 수 $K$를 늘려 성능을 높이더라도 MaskCut보다 훨씬 빠른 속도를 유지한다.
3. **Pseudo-label 성능 (ProMerge+)**: ProMerge로 생성한 pseudo-label로 학습시킨 모델은 CutLER보다 평균 AP에서 0.9, AR에서 1.4 포인트 더 높은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 성공 요인
ProMerge의 성공 요인은 크게 두 가지로 분석된다. 첫째, 고정된 마스크 개수를 가정하지 않고 이미지 상황에 맞춰 유연하게 예측을 수행함으로써 다중 객체 장면에서 강점을 보인다. 둘째, 제안된 Cascade filtering을 통해 배경 노이즈를 효과적으로 제거함으로써 precision과 recall의 균형을 맞추었다.

### 한계 및 비판적 해석
본 모델은 Fully-supervised 방식(예: SAM)에 비해 여전히 성능 격차가 존재한다. 이는 사용된 DINO 특징이 객체의 로컬라이제이션이나 세그멘테이션을 위한 명시적인 pretext task로 학습되지 않았기 때문이다.
실제 결과에서도 다음과 같은 실패 사례가 관찰되었다.
- **Undersegmentation**: 동일한 색상이나 텍스처를 가진 인접한 객체들을 하나의 객체로 인식하는 경향이 있다.
- **Oversegmentation**: 객체의 일부가 가려진(occlusion) 경우, 이를 서로 다른 여러 개의 객체로 분할하는 문제가 발생한다. 이는 모델이 객체에 대한 고차원적인 세만틱 이해 없이 단순한 특징 유사도에 의존하기 때문이다.

## 📌 TL;DR

본 논문은 비지도 인스턴스 분할을 위해 **Prompt and Merge**라는 효율적인 프레임워크를 제안한다. DINO 특징의 point-prompting으로 초기 마스크를 생성하고, 정교한 배경 프루닝 및 유사도 기반 병합을 통해 최종 마스크를 도출한다. 결과적으로 기존 SOTA 모델인 MaskCut 대비 **추론 속도를 3.6배 향상**시키면서도 더 높은 정확도를 달성했으며, 이를 통한 pseudo-label 생성 능력이 매우 우수함을 입증하였다. 이 연구는 계산 비용이 큰 normalized-cut 방식의 대안을 제시했다는 점에서 향후 실시간 비지도 세그멘테이션 연구에 중요한 역할을 할 것으로 보인다.