# TraSeTR: Track-to-Segment Transformer with Contrastive Query for Instance-level Instrument Segmentation in Robotic Surgery

Zixu Zhao, Yueming Jin, and Pheng-Ann Heng (2022)

## 🧩 Problem to Solve

본 논문은 로봇 보조 수술(Robot-Assisted Surgery, RAS) 환경에서 수술 도구의 인스턴스 수준 세그멘테이션(Instance-level Segmentation) 문제를 해결하고자 한다. 수술 도구 세그멘테이션은 수술 워크플로우 최적화, 외과 의사의 인지 능력 보조, 그리고 수술 기술 평가 등을 위한 필수적인 기초 작업이다.

기존의 접근 방식은 크게 두 가지 패러다임으로 나뉜다. 첫째는 픽셀 분류(Pixel classification) 방식으로, 각 픽셀에 대해 클래스를 예측하는 방법이다. 하지만 이 방식은 하나의 도구가 여러 개의 서로 다른 타입으로 예측되는 '공간적 클래스 불일치(Spatial class inconsistency)' 문제를 겪는다. 둘째는 마스크 분류(Mask classification) 방식으로, 이진 마스크를 먼저 생성한 후 각 마스크에 클래스를 할당하는 방법이다. 그러나 수술 영상 특성상 내시경 카메라의 줌 인/아웃으로 인해 도구의 외형이 급격하게 변하는 '큰 시간적 변동(Large temporal variations)'이 발생하며, 이로 인해 프레임 간 클래스 일관성을 유지하는 것이 매우 어렵다는 한계가 있다.

따라서 본 논문의 목표는 추적(Tracking) 단서를 지능적으로 활용하여 수술 도구의 타입과 인스턴스를 정확하게 구분하고, 시간적 일관성을 확보하는 Track-to-Segment Transformer를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **추적 정보를 세그멘테이션 과정에 직접 통합하는 'Track-to-Segment' 구조**를 설계하는 것이다. 이를 위해 다음과 같은 핵심 메커니즘을 도입하였다.

1. **Prior Query의 도입**: 이전 프레임에서 학습된 도구의 정보를 담고 있는 Prior query를 현재 프레임의 입력으로 사용하여, 이전의 추적 신호를 현재 인스턴스로 전이시킨다.
2. **Identity Matching 전략**: 단순한 이분 매칭(Bipartite matching) 대신, 수술 도구의 특성(좌우 팔의 배치)을 고려한 Relative Horizontal Distance(RHD) 기반의 2단계 ID 매칭 방식을 통해 시간적 변동성이 큰 상황에서도 안정적으로 도구를 추적한다.
3. **Contrastive Query Learning**: 대조 학습(Contrastive Learning)을 통해 쿼리 특징 공간을 재구성함으로써, 도구의 외형 변화가 심하더라도 동일한 인스턴스에 대한 쿼리 임베딩은 가깝게, 서로 다른 클래스의 임베딩은 멀게 유지하도록 유도한다.
4. **Link-by-link 추론**: 자동 회귀(Auto-regressive) 방식으로 Prior query를 동적으로 업데이트하며 인스턴스를 추론하는 전략을 제안한다.

## 📎 Related Works

기존 연구들은 주로 U-Net과 같은 구조를 기반으로 픽셀 단위의 분류를 수행하였으며, 깊이 맵(Depth maps), 광학 흐름(Optical flows) 등의 공간적/시간적 단서를 활용해 도구 타입을 구분하려 했다. 하지만 앞서 언급한 것처럼 공간적 일관성 결여 문제가 지속되었다.

마스크 기반 접근법인 ISINet은 Mask R-CNN을 활용해 인스턴스별 클래스를 예측하고 이전 프레임의 결과를 반영하는 재라벨링(Relabelling) 전략을 사용했다. 그러나 이는 급격한 외형 변화가 일어날 때 잘못된 라벨을 할당하는 경향이 있다. 최근 DETR과 같은 Transformer 기반 모델들이 객체 탐지 및 추적에서 뛰어난 성능을 보였으나, 이를 수술 도구의 인스턴스 세그멘테이션과 시간적 일관성 유지 문제에 최적화하여 적용한 시도는 부족했다.

## 🛠️ Methodology

### 전체 파이프라인

TraSeTR은 마스크 분류 패러다임을 따르며, 크게 **Frame extractor $\rightarrow$ Transformer module $\rightarrow$ Instance fusion module** 순으로 구성된다.

1. **Frame Extractor**: ResNet-50 백본을 사용하여 프레임 특징 맵 $F \in \mathbb{R}^{H_S \times W_S \times C_F}$를 추출하고, 이를 픽셀 디코더를 통해 픽셀 수준 임베딩 $E^{pixel} \in \mathbb{R}^{HW \times C_E}$로 변환한다.
2. **Transformer Module**: 표준 Encoder-Decoder Transformer를 사용한다. 입력으로는 프레임 특징 $F$와 $N$개의 쿼리 임베딩(Prior queries + Current queries)을 사용하며, 출력으로 $N$개의 인스턴스 임베딩 $E^{inst} \in \mathbb{R}^{N \times C_Q}$를 생성한다.
3. **Instance Fusion Module**: $E^{inst}$를 MLP에 통과시켜 클래스 확률 $p_i$, 바운딩 박스 $b_i$를 예측한다. 마스크 $m_i$는 $E^{inst}$에서 생성된 마스크 임베딩 $E^{mask}$와 $E^{pixel}$의 내적(Dot product) 후 시그모이드 함수를 적용하여 생성한다.

### Identity Matching (ID 매칭)

시간적 일관성을 위해 2단계 매칭 전략을 사용한다.

- **1단계 (Prior Matching)**: Prior query로부터 생성된 예측값과 현재 Ground-Truth(GT)를 매칭한다. 이때 클래스가 동일하고, 수술 도구의 좌우 배치 특성을 반영한 상대적 수평 거리(Relative Horizontal Distance, RHD)가 최소가 되는 쌍을 찾는다. 매칭 비용 함수는 다음과 같다.
    $$L_{match}(z_{\sigma^1(j)}, \tilde{z}_j) = \mathbb{1}_{\tilde{c}^{prior}_{\sigma^1(j)} = \tilde{c}_j} \cdot |\tilde{x}^{prior}_{\sigma^1(j)} - \tilde{x}_j|$$
    여기서 $\tilde{x}$는 바운딩 박스의 중심 x좌표를 의미한다.
- **2단계 (Current Matching)**: 1단계에서 매칭되지 않은 GT와 Current query의 예측값을 일반적인 이분 매칭(Bipartite matching) 방식으로 연결하여 새로 등장한 도구를 탐지한다.

### 학습 목표 및 손실 함수

학습은 Hungarian Loss($L_{Hung}$)와 Contrastive Loss($L_{ctr}$)의 조합으로 이루어진다.

1. **Hungarian Loss**: DETR에서 사용되는 손실 함수와 동일하게 클래스 예측(NLL), 바운딩 박스($L_1 + \text{GIoU}$), 마스크(Dice + Focal loss) 손실의 합으로 구성된다.
    $$L_{Hung}(z, \tilde{z}) = \sum_{j=1}^{\tilde{N}} [-\log p_{\sigma(j)}(\tilde{c}_j) + L_{box}(b_{\sigma(j)}, \tilde{b}_j) + L_{mask}(m_{\sigma(j)}, \tilde{m}_j)]$$
2. **Contrastive Query Learning**: 동일 인스턴스의 현재 임베딩 $e_i$와 이전 프레임의 Prior 임베딩 $e^{prior}_i$는 가깝게, 서로 다른 클래스의 임베딩 $e_j$와는 멀어지도록 학습한다.
    $$L_{ctr} = -\frac{1}{|\sigma|} \log \sum_{i \in \sigma} \frac{\phi(e_i, e^{prior}_i)}{\phi(e_i, e^{prior}_i) + \sum_j \phi(e_i, e_j)}$$
    여기서 $\phi(\cdot, \cdot)$는 두 임베딩의 내적을 통한 유사도 함수이다.

최종 손실 함수는 $L = L_{Hung} + \lambda_{ctr} L_{ctr}$ 로 정의된다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis17, EndoVis18 (로봇 수술), CaDIS (백내장 수술)
- **지표**: mean Intersection-over-Union (mIoU), Dice coefficient
- **비교 대상**: 픽셀 분류 방식(TernausNet, MF-TAPNet, Dual-MF) 및 마스크 분류 방식(ISINet, DETR, TrackFormer)

### 주요 결과

- **EndoVis17 & 18**: TraSeTR은 EndoVis17에서 60.4% mIoU, 65.2% Dice를 기록하며 SOTA를 달성하였다. 특히 기존 ISINet 대비 mIoU 기준 4.8% 상승하는 결과를 보였다. EndoVis18에서도 76.2% mIoU를 기록하며 성능 우위를 증명하였다.
- **CaDIS**: 다양한 수술 환경에서도 강건함을 보이며 69.9% mIoU를 달성하여 DeepLabV3+, UPerNet, HRNetV2 등의 강력한 베이스라인보다 우수한 성능을 보였다.
- **추론 속도**: 추가적인 가속 없이 23 FPS의 빠른 속도로 동작하여 실시간 적용 가능성을 보여주었다.

### Ablation Study

- **Identity Matching vs Bipartite Matching**: 단순 이분 매칭을 사용했을 때의 추적 성공률은 27.3%에 불과했으나, 제안한 ID 매칭은 100%의 추적률(학습 시)을 보였으며, 이는 mIoU 3.5% 향상으로 이어졌다.
- **Query 구성**: Current query만 사용했을 때보다 Prior query를 추가했을 때, 그리고 여기에 Contrastive learning을 더했을 때 성능이 단계적으로 향상됨을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 수술 도구 세그멘테이션에서 가장 큰 난제인 **'시간적 일관성'** 문제를 Transformer의 쿼리 메커니즘과 추적 기술을 결합하여 효과적으로 해결하였다. 특히, 수술실의 특수한 환경(두 팔의 배치)을 반영한 RHD 기반 매칭과 외형 변화에 강건한 대조 학습(Contrastive Learning)의 결합이 성능 향상의 핵심 요인으로 분석된다.

다만, 본 논문에서는 추적 정확도를 평가하기 위해 GT ID가 없는 데이터셋에서 일부 추측 기반의 분석을 수행한 점이 있으며, 매우 빠른 움직임이나 심한 가려짐(Occlusion)이 발생하는 극단적인 상황에서의 견고성에 대한 상세 분석은 부족한 편이다. 향후 연구에서는 로봇 시스템의 키네마틱스(Kinematics) 데이터와 같은 멀티모달 정보를 통합함으로써 더욱 정밀한 인스턴스 인식을 달성할 수 있을 것으로 보인다.

## 📌 TL;DR

TraSeTR은 수술 도구의 인스턴스 세그멘테이션을 위해 **Prior query**와 **Identity matching**, 그리고 **Contrastive learning**을 도입한 Transformer 기반 모델이다. 이를 통해 기존 픽셀 분류 방식의 불일치 문제와 마스크 분류 방식의 시간적 불안정성 문제를 동시에 해결하였으며, EndoVis 및 CaDIS 데이터셋에서 SOTA 성능과 23 FPS의 실시간성을 확보하였다. 이 연구는 향후 수술 자동화(Suturing 등)를 위한 고정밀 인스턴스 인식 프레임워크의 기초가 될 가능성이 크다.
