# S2D: Sparse-To-Dense Keymask Distillation for Unsupervised Video Instance Segmentation

Leon Sick, Lukas Hoyer, Dominik Engel, Pedro Hermosilla, Timo Ropinski (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **비지도 비디오 인스턴스 분할(Unsupervised Video Instance Segmentation, UVIS)**이다. 비디오 인스턴스 분할은 각 프레임에서 객체를 분리할 뿐만 아니라, 시간적 흐름에 따라 동일한 객체의 정체성(Identity)을 유지하며 추적해야 하는 매우 도전적인 작업이다.

기존의 UVIS 연구들, 특히 최신 성능을 보이는 모델들은 주로 ImageNet과 같은 객체 중심 이미지 데이터셋에서 생성된 **합성 비디오 데이터(Synthetic Video Data)**에 크게 의존해 왔다. 하지만 단순히 이미지 마스크를 이동시키거나 크기를 조절하여 만든 합성 데이터는 원근 변화, 객체 부분의 움직임, 카메라 모션과 같은 **실제 비디오의 복잡한 동적 특성을 정확하게 모델링하지 못한다는 치명적인 한계**가 있다.

따라서 본 논문의 목표는 인간의 주석(Human Annotation) 없이 **오직 실제 비디오 데이터만을 사용하여 학습**하며, 복잡한 시간적 역동성을 학습할 수 있는 UVIS 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"희소한 고품질 마스크(Sparse Keymasks)를 먼저 발견하고, 이를 증류(Distillation) 과정을 통해 조밀한 마스크(Dense Masks)로 확장한다"**는 것이다. 이를 위해 다음과 같은 세 가지 핵심 설계를 제안한다.

1.  **Keymask Discovery**: 딥 모션 프라이어(Deep Motion Prior)인 포인트 트래킹을 활용하여, 시간적으로 일관되고 품질이 높은 프레임 마스크만을 선별해 희소한 가상 라벨셋(Sparse Pseudo-labelset)을 구축한다.
2.  **Sparse-To-Dense Distillation**: 희소하게 주어진 라벨을 조밀한 예측으로 확장하기 위해 Teacher-Student 구조의 증류 기법을 도입한다.
3.  **Temporal DropLoss**: 라벨이 없는 프레임에서 발생하는 잘못된 손실 계산(Penalty)을 방지하여, 모델이 자연스럽게 마스크를 전파(Propagation)하도록 유도한다.

## 📎 Related Works

### 기존 연구 및 한계
*   **비지도 이미지 인스턴스 분할**: MaskDistill, FreeSOLO, CutLER 등이 있으며, 주로 자기지도학습(Self-supervised learning) 모델의 특징이나 Normalized Cut 기반의 마스크 추출 방식을 사용한다. 본 논문은 단일 프레임 마스크 추출을 위해 $\text{CutS3D}$를 사용한다.
*   **비지도 비디오 분할**: 이전 연구들은 주로 비디오 객체 분할(VOS)에 집중하여 모든 움직이는 객체를 하나의 전경으로 처리하거나, 광학 흐름(Optical Flow) 네트워크에 의존했다.
*   **VideoCutLER**: 합성 비디오를 통해 학습하여 제로샷(Zero-shot) 성능을 높였으나, 앞서 언급했듯 실제 비디오의 복잡한 모션을 학습하지 못한다는 한계가 있다.

### 차별점
S2D는 합성 데이터가 아닌 **실제 비디오 데이터**만을 사용하며, 포인트 트래킹을 통해 시간적 일관성을 검증한 후, 학습 가능한 모델 내부에서 **암시적 마스크 전파(Implicit Mask Propagation)**를 수행함으로써 기존의 명시적 전파 알고리즘보다 유연하고 정확한 결과를 얻는다.

## 🛠️ Methodology

전체 파이프라인은 **Keymask Discovery $\rightarrow$ Sparse-To-Dense Distillation $\rightarrow$ Final Training**의 단계로 구성된다.

### 1. Keymask Discovery (희소 키마스크 발견)
단일 프레임 마스크($\text{CutS3D}$ 결과물)는 시간적 일관성이 없고 노이즈가 많다. 이를 해결하기 위해 다음 두 단계를 거친다.

*   **Visibility Grouping (가시성 그룹화)**: 
    각 마스크에 포인트 그리드를 설정하고 포인트 트래커를 통해 비디오 전체에서 추적한다. 특정 프레임 $t$에서 가시적인 포인트의 비율 $\gamma_t$가 임계값 $\gamma_{thr}$(0.3)을 넘으면 해당 인스턴스가 보인다고 판단한다. 이렇게 생성된 이진 가시성 벡터(Binary Visibility Vector)를 $\text{DBSCAN}$으로 클러스터링하여, 동시에 나타나고 사라지는 마스크들을 하나의 그룹으로 묶는다.
*   **Proxy Propagate-And-Match (프록시 전파 및 매칭)**:
    가시성 그룹 내에서도 서로 다른 객체가 있을 수 있다. 이를 분리하기 위해 포인트 트랙을 프록시로 사용하여 한 프레임의 마스크를 다른 프레임으로 전파하고, 대상 프레임의 마스크와 겹치는 정도를 **Point-Mask Jaccard Index**로 계산한다.
    $$J(i, k, t) = \frac{\sum_{j=1}^{N_i} \mathbb{1}(x^t_j \in m^t_k)}{N_i}, \quad x^t_j \in X^t_i$$
    여기서 $X^t_i$는 인스턴스 $i$의 전파된 포인트 집합, $m^t_k$는 프레임 $t$의 이미지 인스턴스 $k$의 마스크이다. 이 지수가 임계값 $\lambda_J$(0.5)보다 크면 동일 인스턴스로 간주한다.

### 2. Temporal DropLoss (시간적 드롭 손실)
희소한 라벨셋으로 학습할 때, 라벨이 없는 프레임에서 모델이 예측을 수행하면 표준 손실 함수는 이를 '오답'으로 처리하여 페널티를 준다. 이를 방지하기 위해 **Temporal DropLoss**를 제안한다.

$$\mathcal{L}^{TempDrop}(i) = \sum_{t=1}^T \mathbb{1}(\|m^t_i\| > 0) \mathcal{L}^{mask}(\hat{m}^t_i, m^t_i)$$
즉, 가상 라벨 $m^t_i$가 존재하는 프레임($\|m^t_i\| > 0$)에서만 마스크 손실 $\mathcal{L}^{mask}$를 계산하고, 라벨이 없는 프레임의 손실은 버린다. 이는 모델이 라벨이 있는 프레임의 정보를 라벨이 없는 프레임으로 전파하도록 유도한다.

### 3. Sparse-To-Dense Distillation (희소-조밀 증류)
$\text{VideoMask2Former}$를 기반으로 Teacher-Student 구조를 설계한다.
*   **Teacher 모델**: Student 모델의 가중치를 지수 이동 평균(EMA)으로 업데이트하여 점진적으로 더 정교한 조밀 예측을 생성한다.
*   **Student 모델**: 두 가지 신호를 동시에 학습한다.
    1.  **Sparse Anchor**: $\text{Temporal DropLoss}$를 통해 희소한 키마스크 라벨을 학습한다.
    2.  **Dense Supervision**: Teacher 모델이 예측한 조밀한 마스크를 학습하여 시간적 일관성을 정교화한다.
*   **최종 손실 함수**:
    $$\mathcal{L}^{Full} = \mathcal{L}^{TempDrop} + \mathcal{L}^{Distill}_{mask}$$

이 과정을 두 단계(Dual-Stage)로 수행하며, 2단계에서는 1단계에서 생성된 조밀한 예측치를 다시 라벨로 사용하여 모델을 한 번 더 정제한다.

## 📊 Results

### 실험 설정
*   **데이터셋**: In-domain 실험을 위해 $\text{YouTube-VIS 2019/2021}$ 및 $\text{DAVIS-all}$을 사용하였다. Zero-shot 실험을 위해 $\text{VIPSeg, MOSEv1, SA-V}$를 합친 13,000개의 실제 비디오 데이터셋을 구축하였다.
*   **기준선(Baseline)**: $\text{VideoCutLER}$ (130만 개의 합성 비디오로 학습된 SOTA 모델).
*   **평가 지표**: $\text{AP}_{50}$, $\text{AP}$, $\text{J\&F}$ (DAVIS).

### 주요 결과
*   **In-Domain 성능**: $\text{YouTube-VIS 2019}$에서 $\text{AP}_{50}$ 기준 $51.2$를 기록하며 $\text{VideoCutLER}$($48.2$) 대비 $+3.0$의 성능 향상을 보였다. $\text{DAVIS-all}$에서도 $\text{J\&F}$ 지표가 $50.6$으로 $\text{VideoCutLER}$($44.9$)를 크게 상회하였다.
*   **Zero-Shot 성능**: 1.3M 개의 합성 데이터로 학습한 $\text{VideoCutLER}$보다 **단 13K 개의 실제 데이터로 학습한 S2D**가 더 높은 성능을 보였다. 특히 $\text{YouTube-VIS 2022}$에서 $\text{AP}_{50}$ 기준 $41.8$ vs $37.0$으로 $+4.8$의 큰 차이를 보였다.
*   **반지도/지도 학습(Semi/Fully Supervised)**: 아주 적은 양의 데이터(1%, 5%, 10%)만으로 파인튜닝했을 때, 실제 데이터로 프리트레이닝된 S2D가 합성 데이터 기반 모델보다 훨씬 빠르게 수렴하고 높은 성능을 냈다.

## 🧠 Insights & Discussion

### 강점 및 의의
본 연구는 UVIS 분야에서 **합성 데이터의 의존도를 완전히 제거**하고 실제 데이터만으로 SOTA 성능을 달성했다는 점에서 매우 큰 의의가 있다. 특히 실제 비디오의 복잡한 모션을 학습함으로써 제로샷 전이 능력이 향상되었음을 입증하였다.

### 한계 및 비판적 해석
*   **소형 객체 분할**: 기존 SOTA 모델들과 마찬가지로 작은 객체를 분할하는 데 어려움을 겪는다. 저자들은 이를 위해 비디오용 Copy-paste 증강 기법이 필요할 수 있음을 언급하였다.
*   **도메인 특수성**: 주행 영상 데이터셋인 $\text{Cityscapes}$에서는 성능이 낮게 나타났다. 이는 학습 데이터셋에 주행 도메인의 데이터가 부족했기 때문으로 분석되며, 도메인 특화 데이터를 추가함으로써 해결 가능하다.
*   **모션 프라이어 의존성**: 본 모델은 포인트 트래킹이라는 외부 모션 큐에 의존하여 키마스크를 발견한다. 향후에는 외부 트래커 없이 모델 내부의 딥 템포럴 피처를 사용하여 이를 대체하는 방향으로 발전할 수 있을 것이다.

## 📌 TL;DR

S2D는 실제 비디오 데이터만을 사용하여 비지도 비디오 인스턴스 분할을 수행하는 프레임워크이다. 포인트 트래킹을 통해 고품질의 **희소 키마스크(Sparse Keymasks)**를 먼저 찾아내고, **Temporal DropLoss**와 **EMA 기반의 Sparse-To-Dense 증류** 과정을 통해 이를 조밀한 라벨로 확장하여 학습한다. 결과적으로 130만 개의 합성 데이터로 학습한 기존 SOTA 모델보다 훨씬 적은 양의 실제 데이터(1.3만 개)만으로도 더 뛰어난 In-domain 및 Zero-shot 성능을 달성하였다.