# Weakly-Supervised 3D Medical Image Segmentation using Geometric Prior and Contrastive Similarity

Hao Du, Qihua Dong, Yan Xu, Jing Liao (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 컴퓨터 보조 진단에서 가장 중요한 전처리 과정 중 하나이지만, 분할 대상의 복잡한 형태와 의료 영상 특유의 아티팩트(낮은 대비의 조직, 불균일한 텍스처 등)로 인해 매우 도전적인 과제이다. 특히, 딥러닝 기반의 분할 모델은 픽셀 단위의 정밀한 주석(Pixel-wise annotation)이 포함된 대량의 학습 데이터가 필요하지만, 전문의가 이를 수행하는 비용이 매우 높다는 한계가 있다.

이를 해결하기 위해 Bounding-box 주석만을 사용하는 약지도 학습(Weakly-supervised learning) 방식이 제안되었으나, 여전히 다음과 같은 두 가지 주요 문제점이 존재한다.
1. **복잡한 형상(Complex Shapes):** 일부 장기는 내부 구조가 매우 정교하여 픽셀 단위의 지도 학습 없이는 정밀한 분할이 어렵다.
2. **영상 아티팩트(Imaging Artifacts):** 전통적으로 사용되는 Gray space에서는 대비가 낮은 조직이나 불균일한 텍스처를 주변 조직과 구분하기 어렵다.

본 논문의 목표는 Geometric Prior와 Contrastive Similarity를 손실 함수 기반으로 통합하여, Bounding-box 주석만으로도 복잡한 형상을 정밀하게 분할하고 낮은 대비의 조직을 효과적으로 구분하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 장기의 기하학적 정보와 특징 공간에서의 유사성을 활용하여 부족한 감독 신호를 보완하는 것이다.

1. **Point Cloud 기반의 Geometric Prior 도입:** 볼륨(Volume) 표현 대신 Point Cloud 표현을 사용하여 장기의 정교한 기하학적 구조를 학습한다. 표준 장기 템플릿(Template)과 예측된 결과 사이의 Chamfer Distance를 최소화함으로써 정밀한 형태 가이드를 제공한다.
2. **Contrastive Similarity를 통한 표현력 강화:** 단순한 Gray space의 한계를 극복하기 위해 고차원 임베딩 공간(Embedding space)으로 픽셀을 인코딩하고, 동일한 라벨의 픽셀들이 서로 가깝게 모이도록 유도하여 낮은 대비의 조직 간 변별력을 높인다.
3. **범용적인 프레임워크 설계:** 제안된 방법론은 특정 모델에 국한되지 않고 BoxInst, Ai+L 등 기존의 Bounding-box 기반 약지도 학습 모델에 쉽게 적용하여 성능을 향상시킬 수 있는 일반적인 구조를 가진다.

## 📎 Related Works

### 기존 약지도 학습 기반 분할
약지도 분할은 주석의 종류에 따라 Scribbles, Points, Image-level tags, Bounding-box 등으로 나뉜다. 특히 Bounding-box 방식은 주석 비용이 낮고 위치 정보를 제공한다는 점에서 선호되지만, 단순한 Bounding-box 내 픽셀을 포그라운드로 간주하는 방식은 노이즈가 많고 오차가 누적되는 경향이 있다.

### 기하학적 사전 정보(Geometric Prior)
의료 영상에서는 장기의 형태와 위치에 대한 해부학적 사전 정보(Anatomical prior)가 명확하다. 기존 연구는 크게 Graph-based 방법과 Loss-based 방법으로 나뉜다. Graph-based 방법은 정확도는 높으나 계산 비용이 매우 크다. Loss-based 방법은 계산 효율적이지만, 대개 영상을 2D/3D 패치로 나누어 처리함으로써 전역적인 기하학적 관계를 손실한다는 한계가 있으며, 균일한 복셀 그리드(Voxel grids) 기반의 볼륨 표현을 사용하여 정교한 구조 묘사가 어렵다.

### 대조 학습(Contrastive Learning)
대조 학습은 긍정/부정 쌍을 구분하여 특징 공간을 학습하는 방식으로, 주로 의료 영상의 사전 학습(Pre-training) 단계에서 사용되어 왔다. 본 논문은 이를 확장하여 임베딩 공간에서의 유사성을 통해 영상 아티팩트 문제를 해결하고 픽셀 간 변별력을 높이는 데 활용한다.

## 🛠️ Methodology

### 전체 시스템 구조
본 프레임워크는 기본적으로 Encoder-Decoder 구조(nnUNet 기반)를 따르며, 입력 이미지 $I$와 Bounding-box $B$를 통해 분할 마스크 $M$을 생성한다. 전체 학습 손실 함수 $L_{frame}$은 다음과 같이 정의된다.

$$L_{frame} = L_{ori} + L_{mask}$$

여기서 $L_{ori}$는 기존 약지도 학습 모델(예: BoxInst의 $L_{fcos}$)의 손실 함수이며, $L_{mask}$는 본 논문에서 제안하는 마스크 헤드 학습 손실로, 다음과 같이 구성된다.

$$L_{mask} = L_{geo} + L_{cons}$$

### Geometric Prior ($L_{geo}$)
복잡한 장기 구조를 학습하기 위해 3D Point Cloud 공간에서 기하학적 제약을 가한다.

1. **Conversion & Registration:** 'Gridding Reverse' 기법을 사용하여 볼륨 표현의 분할 결과를 Point Cloud 표현으로 변환한다. 이는 복셀 그리드의 제약을 벗어나 더 정교한 좌표 표현을 가능하게 한다. 이후, ICP(Iterative Closest Point) 등록 도구를 사용하여 표준 템플릿 $T$와 예측된 제안 영역 $S$ 사이의 변환 행렬을 계산하고 정렬한다.
2. **Chamfer Distance Loss:** 정렬된 두 점구름 사이의 거리를 최소화하기 위해 Chamfer Distance를 사용한다.

$$L_{geo} = \frac{1}{|S|} \sum_{x \in S} \min_{y \in T} ||x-y||^2 + \frac{1}{|T|} \sum_{y \in T} \min_{x \in S} ||y-x||^2$$

또한, 패치 단위 처리 시 장기가 잘릴 수 있으므로, **Completeness Head**를 통해 해당 제안 영역이 이미지 내에서 완전한 형태를 갖추었는지 판단하고, 완전한 경우에만 $L_{geo}$를 적용한다.

### Contrastive Similarity ($L_{cons}$)
Gray space의 낮은 변별력을 해결하기 위해 Contrastive Head를 통해 픽셀을 고차원 임베딩 공간으로 매핑한다.

1. **Pre-training:** Bounding-box만을 이용하여 Contrastive Head를 'Coarse-to-Fine' 방식으로 사전 학습한다.
   - **Coarse stage:** 박스 내부를 긍정, 외부를 부정 라벨로 간주하여 학습한다.
   - **Refine stage:** 무작위로 선택된 $K$개의 참조 픽셀과의 거리를 계산하여 긍정/부정 라벨을 정교화한 후 학습한다.
2. **Contrastive Similarity Loss:** 입력 이미지의 픽셀들을 정점으로, 인접 픽셀을 엣지로 하는 무방향 그래프를 구축한다. 두 인접 픽셀이 동일한 라벨을 가질 확률 $\text{Prob}(y_e=1)$을 계산하고, 임베딩 공간에서의 유사도가 임계값 $\tau$ 이상인 긍정 엣지들에 대해 다음과 같은 손실 함수를 적용한다.

$$L_{cons} = -\frac{1}{N} \sum_{e \in E} \mathbb{1}_{\{C_{e_{start}} \cdot C_{e_{end}} \geq \tau\}} \log \text{Prob}(y_e=1)$$

## 📊 Results

### 실험 설정
- **데이터셋:** LiTS 2017 (간), KiTS 2021 (신장), LPBA40 (해마)
- **비교 대상:** BoxInst, MIL, GMIL 및 Fully Supervised (Upper bound)
- **평가 지표:** Dice Score (DSC, $\uparrow$), Hausdorff Distance (HD95, $\downarrow$)
- **구현:** UNet 백본 기반, GeForce RTX 3090 GPU 사용

### 주요 결과
- **정량적 결과:** 모든 데이터셋에서 제안 방법이 BoxInst 및 MIL 기반 방법들보다 우수한 성능을 보였다. 특히 KiTS21 데이터셋에서 BoxInst 대비 DSC가 49.1% $\rightarrow$ 80.2%로 대폭 향상되었다.
- **정성적 결과:** MIL 기반 방법들이 장기의 외곽선 위주로 분할하는 반면, 제안 방법은 Geometric Prior 덕분에 장기 내부의 정교한 구조와 세부 디테일을 훨씬 더 잘 보존하는 것으로 나타났다.
- **Ablation Study:**
    - **Point Cloud vs Volume:** Point Cloud 표현이 볼륨 표현보다 DSC 기준 약 4.3% 높은 성능을 보여, 복잡한 구조 묘사에 더 효율적임이 입증되었다.
    - **Internal Details:** 템플릿에서 내부 구조 정보를 제거했을 때 성능이 3.9% 하락하여, 내부 기하 정보의 중요성이 확인되었다.
    - **Embedding Space:** 단순 Gray space 기반의 MSE, SSIM 유사도보다 Contrastive Embedding 공간에서의 유사도가 분할 성능과 효율성 면에서 압도적이었다.

## 🧠 Insights & Discussion

본 논문은 약지도 학습의 고질적인 문제인 '정밀도 부족'을 외부 지식(Geometric Prior)과 특성 공간의 재구성(Contrastive Similarity)으로 해결하였다는 점에서 강점이 있다. 특히 Point Cloud로의 변환을 통해 복셀 그리드의 이산화 문제를 해결하고, Chamfer Distance를 통해 형태적 유사성을 강제한 점이 효과적이었다.

다만, 다음과 같은 한계와 논의점이 존재한다.
1. **템플릿 의존성:** 본 방법론은 표준 장기 템플릿이 존재해야 한다. 만약 환자마다 장기 형태의 변이가 극심하거나 표준 템플릿을 정의하기 어려운 병변의 경우에는 적용이 어려울 수 있다.
2. **계산 복잡도:** Point Cloud 변환 및 ICP 등록 과정이 추가되어 학습 시간이 증가하며, 특히 3D 데이터의 특성상 메모리 사용량에 대한 최적화가 지속적으로 필요할 것으로 보인다.
3. **임계값 설정:** Contrastive Similarity와 Completeness Head에서 사용되는 임계값($\tau, 0.6$ 등)이 경험적으로 설정되었으므로, 데이터셋마다 최적의 값을 찾는 추가적인 튜닝이 필요할 수 있다.

## 📌 TL;DR

이 논문은 Bounding-box 주석만으로 3D 의료 영상을 정밀하게 분할하기 위해 **Point Cloud 기반의 Geometric Prior**와 **고차원 임베딩 공간의 Contrastive Similarity**를 도입한 프레임워크를 제안한다. 이를 통해 기존 약지도 학습의 한계였던 복잡한 내부 구조 묘사와 낮은 대비의 조직 구분 문제를 성공적으로 해결하였으며, 다양한 의료 영상 데이터셋에서 SOTA 수준의 성능 향상을 입증하였다. 이 연구는 정형화된 형태를 가진 장기의 약지도 분할 연구에 중요한 기준이 될 것으로 보인다.