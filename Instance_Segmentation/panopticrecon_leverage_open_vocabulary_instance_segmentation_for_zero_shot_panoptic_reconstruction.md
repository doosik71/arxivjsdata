# PanopticRecon: Leverage Open-vocabulary Instance Segmentation for Zero-shot Panoptic Reconstruction

Xuan Yu, Yili Liu, Chenrui Han, Sitong Mao, Shunbo Zhou, Rong Xiong, Yiyi Liao, Yue Wang (2024)

## 🧩 Problem to Solve

본 논문은 3D 장면 이해의 핵심 과제인 **Panoptic Reconstruction**(기하학적 구조와 함께 시맨틱 및 인스턴스 정보를 복원하는 작업)을 해결하고자 한다. 기존의 방식들은 주로 사전 학습된 시맨틱 세그멘테이션 모델이나 이미 알려진 3D 오브젝트 바운딩 박스(Bounding Box)에 크게 의존한다. 그러나 이러한 정보는 실제 야외나 일반적인 환경(In-the-wild scenes)에서는 얻을 수 없다는 한계가 있다.

특히, Open-vocabulary 모델인 Grounded-SAM을 사용하여 Zero-shot Panoptic Reconstruction을 시도할 때 다음과 같은 두 가지 주요 문제에 직면하게 된다:

1. **Partial Labeling**: 텍스트 프롬프트 기반의 VLM(Vision-Language Model)은 프롬프트에 해당하는 클래스만 예측하고 나머지는 'unknown'으로 처리하여 레이블이 불완전하게 생성된다.
2. **Instance Association**: 여러 프레임에 걸쳐 나타나는 2D 인스턴스 ID를 하나의 일관된 3D 인스턴스 ID로 연결하는 것이 매우 어렵다.

따라서 본 논문의 목표는 추가적인 인간의 주석(Annotation) 없이, RGB-D 이미지로부터 기하 구조, 시맨틱, 인스턴스 정보를 동시에 복원하는 **Zero-shot Panoptic Reconstruction** 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Open-vocabulary 인스턴스 세그멘테이션의 한계를 보완하기 위해 **DINOv2 기반의 레이블 전파(Label Propagation)**와 **3D 인스턴스 그래프 추론(3D Instance Graph Inference)**을 도입하는 것이다.

- **DINOv2를 이용한 레이블 전파**: Grounded-SAM이 생성한 불완전한 레이블을 보완하기 위해, 시맨틱 의미가 풍부한 DINOv2 특징(Feature)을 3D 공간으로 증류(Distill)하고, 이를 통해 학습된 포인트 레벨 분류기를 사용하여 빈 공간의 레이블을 채운다.
- **3D 인스턴스 그래프를 통한 ID 통합**: 2D 인스턴스 마스크와 3D 기하 구조(Normal 기반 Superface)를 결합하여 그래프를 구축하고, 이를 세그멘테이션 문제로 정의하여 전역적으로 유일한 3D Pseudo ID를 추론함으로써 인스턴스 연관성 문제를 해결한다.
- **신경 암시적 표현(Neural Implicit Representation) 기반 복원**: SDF(Signed Distance Function), 색상, 시맨틱, 인스턴스 정보를 별도의 특징 볼륨(Feature Volume)으로 관리하며, 이를 통합적으로 학습하여 고품질의 Panoptic Mesh를 생성한다.

## 📎 Related Works

### 1. Semantic Neural Fields

NeRF와 같은 신경 암시적 표현을 확장하여 시맨틱 정보를 함께 인코딩하려는 시도가 있었다. Semantic NeRF나 NeSF 등이 있으며, 최근에는 VLM을 결합하여 Open-vocabulary 시맨틱 복원을 수행하는 NIVLFF 등의 연구가 진행되었다.

### 2. Panoptic Neural Fields

PNF나 Panoptic NeRF는 2D 패놉틱 레이블과 3D 바운딩 박스를 사용하여 3D 공간의 인스턴스를 구분했다. 하지만 이들은 폐쇄형 어휘(Close-vocabulary) 환경에 국한되거나 수동 주석이 많이 필요하다는 단점이 있다.

### 3. 기존 Open-vocabulary 접근 방식과의 차이점

PVLFF와 같은 기존 연구는 서로 다른 두 개의 VLM(LSeg와 SAM)에서 특징을 추출하여 3D로 정렬하려 했으나, 두 모델 간의 레이블 불일치 문제가 있고 SAM의 특성상 오브젝트 레벨의 인스턴스를 정확히 분리하지 못하는 한계가 있었다. PanopticRecon은 Grounded-SAM을 기본으로 하되, 그래프 기반 추론과 DINOv2 전파를 통해 이 문제를 해결함으로써 더 높은 정확도와 일관성을 확보한다.

## 🛠️ Methodology

### 전체 파이프라인

PanopticRecon은 크게 **Reconstruction**과 **Segmentation**이라는 두 가지 구성 요소로 나뉘며, 총 2단계의 복원 과정을 거친다.

1. **1단계 복원**: 시맨틱/인스턴스 정보 없이 순수 기하학적 맵을 먼저 구축하여, 이후 단계의 인스턴스 연관성 추론을 위한 기초 데이터를 제공한다.
2. **세그멘테이션 단계**: Grounded-SAM으로 2D 레이블을 생성 $\rightarrow$ 3D 그래프 추론을 통해 인스턴스 ID를 통합하고 시맨틱 레이블을 교정 $\rightarrow$ DINOv2 기반으로 불완전한 레이블을 전파한다.
3. **2단계 복원**: 교정 및 전파된 레이블을 사용하여 SDF, 색상, 시맨틱, 인스턴스를 공동 학습하여 최종 Panoptic Mesh를 생성한다.

### 인스턴스 연관성 및 레이블 교정 (Instance Association & Label Correction)

- **그래프 구축**: 3D 메쉬의 법선(Normal) 방향 기반 유사도를 계산하여 맵을 **Superfaces**로 클러스터링한다. 각 Superface의 중심점을 그래프의 노드 $V$로 설정한다.
- **에지 투표(Edge Voting)**: 이미지 $I$에서 2D 인스턴스 마스크 $M_j$와 Superface의 투영 마스크 간의 겹침(Overlap) $U_{ij}$를 계산한다. 가장 많이 겹치는 노드를 중심 노드 $V^j_c$로 정하고, 다른 겹치는 노드들과의 에지에 투표한다. 반면, 다른 인스턴스에 속한 노드와의 에지 투표수는 감산한다.
- **ID 생성 및 교정**: 최종적으로 양수 투표수를 가진 에지만 남긴 그래프에서 클러스터링을 수행하여 3D 인스턴스를 정의한다. 이를 2D로 투영하여 일관된 3D ID를 부여하고, 다수결 원칙을 통해 잘못된 2D 시맨틱 레이블을 교정한다.

### 레이블 전파 (Label Propagation)

Grounded-SAM의 부분적인 레이블링 문제를 해결하기 위해 DINOv2 특징을 활용한다.

- RGB 이미지에서 DINOv2 특징을 추출하고 PCA를 통해 64차원으로 압축하여 $V^{64}$를 생성한다.
- 이를 3D 공간의 시맨틱 브랜치에 증류(Distill)시키고, DINOv2 특징과 3D 포인트 위치 인코딩을 입력으로 하는 MLP 분류기를 학습시켜, 레이블이 없는 영역까지 시맨틱 정보를 확장한다.

### 신경 암시적 표현 (Neural Implicit Representation)

본 모델은 4개의 특징 볼륨 $\Psi_G, \Psi_C, \Psi_S, \Psi_I$ (Geometry, Color, Semantics, Instance)를 사용한다.

**볼륨 렌더링(Volume Rendering):**
픽셀 $x$에 해당하는 레이 $(o, d)$를 따라 $N$개의 점 $p_i$를 샘플링하여 다음과 같이 렌더링한다.
$$u_f(x) = \sum_{i=1}^{N} T_i \alpha_i f(p_i)$$
여기서 $\alpha_i$는 다음과 같이 정의된다.
$$\alpha_i = \max\left(\frac{\Phi(s_i) - \Phi(s_{i+1})}{\Phi(s_i)}, 0\right)$$
($s$는 SDF 값, $\Phi$는 시그모이드 함수)

**손실 함수(Loss Functions):**
전체 손실 함수 $L$은 다음과 같이 구성된다.
$$L = L_{Depth} + L_{SDF} + L_{eik} + L_{Color} + L_{S/I} + L_{DN}$$

- $L_{SDF}$: 관측된 깊이 값 $b(p_i)$와 예측된 SDF 값 $s(p_i)$ 사이의 오차를 측정한다. (Eq. 9)
- $L_{eik}$: SDF의 기울기가 1이 되도록 강제하는 Eikonal loss이다. (Eq. 10)
- $L_{Depth}, L_{Color}$: 렌더링된 깊이 및 색상과 실제 값 사이의 L1/L2 거리이다.
- $L_{S/I}$: 시맨틱 및 인스턴스 레이블에 대한 Cross-Entropy 손실이다. (Eq. 13)
- $L_{DN}$: 렌더링된 DINOv2 특징과 실제 압축된 특징 $V^{64}$ 사이의 L2 거리이다. (Eq. 14)

## 📊 Results

### 실험 설정

- **데이터셋**: 실내 데이터셋인 **ScanNet V2**와 실외 데이터셋인 **KITTI-360**을 사용하였다.
- **지표**: 장면 레벨의 Panoptic Quality ($PQ_s$), mIoU, mAcc, mCov 등을 사용하여 평가하였다.
- **비교 대상**: Panoptic NeRF (Closed-vocabulary), Grounded-SAM + Panoptic Lifting, PVLFF 등을 비교군으로 설정하였다.

### 주요 결과

1. **시맨틱 세그멘테이션**: ScanNet에서 제안 방법이 다른 Open-vocabulary 방식보다 월등히 높은 성능을 보였으며, 이는 DINOv2 기반 레이블 전파의 효과임을 입증하였다.
2. **인스턴스 세그멘테이션**: ScanNet에서 제안 방법이 Panoptic NeRF(GT Bbox 사용)보다 높은 성능을 기록하였다. 이는 3D 그래프 추론을 통한 일관된 ID 부여가 복잡한 실내 환경에서 효과적임을 보여준다.
3. **패놉틱 세그멘테이션**: KITTI-360에서는 GT Bbox를 사용하는 Panoptic NeRF가 가장 우수했으나, ScanNet에서는 제안 방법이 가장 높은 $PQ_s$ (62.39)를 기록하며 Zero-shot 능력을 증명하였다.
4. **메쉬 복원 품질**: PVLFF와 비교했을 때, 제안 방법이 거리 오차(Acc)와 완료도(Comp) 면에서 훨씬 우수한 성능을 보였으며, 시각적으로도 더 세밀한 표면 복원이 가능함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

- **Zero-shot 확장성**: 특정 클래스에 국한되지 않고 텍스트 프롬프트만으로 3D 공간의 패놉틱 복원을 수행할 수 있다는 점이 매우 강력하다.
- **상호 보완적 구조**: 기하 구조 $\rightarrow$ 인스턴스 연관 $\rightarrow$ 시맨틱 교정 $\rightarrow$ 최종 복원으로 이어지는 파이프라인이 각 단계의 오류를 상쇄하며 품질을 높인다.
- **DINOv2의 활용**: 단순한 선형 분류기가 아닌, 특징 증류와 MLP 분류기를 통해 Open-vocabulary 모델의 고질적인 문제인 '부분적 레이블링' 문제를 효과적으로 해결하였다.

### 한계 및 논의사항

- **계산 비용**: 2단계에 걸친 복원 과정과 DINOv2 특징 추출, 그래프 구축 등의 과정이 포함되어 있어 학습 및 추론 시간이 상당할 것으로 추측된다(논문 내 구체적인 시간 측정치는 명시되지 않음).
- **초기 기하 구조 의존성**: 인스턴스 그래프를 구축하기 위해 1단계에서 순수 기하 구조를 먼저 복원해야 하므로, 초기 SDF 복원 품질이 낮을 경우 인스턴스 연관성 성능이 저하될 가능성이 있다.

## 📌 TL;DR

본 논문은 Grounded-SAM과 DINOv2를 결합하여, 사전 학습된 3D 정보 없이도 가능한 **Zero-shot Panoptic Reconstruction** 방법론인 **PanopticRecon**을 제안한다. 3D 인스턴스 그래프 추론을 통해 2D 인스턴스 ID를 전역적으로 통합하고, DINOv2 특징을 이용해 불완전한 시맨틱 레이블을 전파함으로써 기존 Open-vocabulary 방식의 한계를 극복하였다. 이 연구는 로봇의 자율 주행 및 환경 이해를 위한 3D 맵 생성 분야에서 인간의 개입 없는 자동화된 장면 이해를 가능케 하는 중요한 발판이 될 것으로 기대된다.
