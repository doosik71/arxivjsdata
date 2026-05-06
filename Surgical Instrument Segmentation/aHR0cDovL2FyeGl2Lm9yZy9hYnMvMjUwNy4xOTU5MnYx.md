# SurgPIS: Surgical-instrument-level Instances and Part-level Semantics for Weakly-supervised Part-aware Instance Segmentation

Meng Wei, Charlie Budd, Oluwatosin Alabi, Miaojing Shi, and Tom Vercauteren (2025)

## 🧩 Problem to Solve

로봇 보조 수술(Robot-assisted surgery)의 자동화를 위해서는 수술 도구의 정밀한 세그멘테이션이 필수적이다. 기존의 연구들은 크게 두 가지 방향으로 진행되어 왔다. 하나는 개별 도구 인스턴스를 구분하는 Instrument-level Instance Segmentation (IIS)이고, 다른 하나는 도구의 각 부위(예: shaft, wrist, clasper)를 구분하는 Part-level Semantic Segmentation (PSS)이다.

그러나 기존 방식들은 이 두 작업을 독립적으로 처리하며, 인스턴스와 그 구성 부위 사이의 계층적 관계를 동시에 학습하는 통합된 모델이 부재했다. 또한, 실제 의료 데이터셋의 경우 IIS 라벨만 있거나 PSS 라벨만 있는 경우가 많아, 인스턴스와 부위 정보가 모두 포함된 Part-aware Instance Segmentation (PIS)을 위한 대규모 데이터셋을 확보하기 어렵다는 문제가 있다.

본 논문의 목표는 수술 도구의 인스턴스와 부위를 동시에 인식하는 통합 PIS 프레임워크인 **SurgPIS**를 제안하고, 라벨이 불완전한 서로 다른 데이터셋(disjoint datasets)을 활용하여 모델을 학습시킬 수 있는 약지도 학습(Weakly-supervised learning) 전략을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **통합 PIS 프레임워크 제안**: 수술 도구 세그멘테이션 분야에서 최초로 인스턴스 수준과 부위 수준의 세그멘테이션을 동시에 수행하는 SurgPIS 모델을 제안한다.
2. **부위 특화 쿼리 변환(Part-specific Query Transformation)**: Instrument-level object query로부터 부위 특화 쿼리를 유도함으로써, 부위와 상위 도구 인스턴스 간의 계층적 연결 고리를 명시적으로 구축한다.
3. **약지도 학습 전략(Weakly-supervised Learning Strategy)**: PIS 전체 라벨이 없는 상황에서도 IIS 또는 PSS 라벨만 존재하는 불완전한 데이터셋을 활용할 수 있도록, Teacher-Student 구조와 마스크 집계(Mask Aggregation) 방식을 도입한다.
4. **범용성 및 성능 검증**: 다양한 데이터셋(EndoVis2017, 2018, SAR-RARP50, GraSP)을 통해 PIS뿐만 아니라 IIS, PSS, ISS(Instrument-level Semantic Segmentation) 작업에서도 최신 기술(SOTA) 수준의 성능을 달성함을 입증한다.

## 📎 Related Works

### 기존 수술 도구 세그멘테이션 연구

기존 연구들은 주로 ISS(도구 종류 구분)나 PSS(부위 종류 구분)에 집중하였다. ISINet이나 TraSeTR과 같은 모델들은 IIS를 통해 개별 도구를 구분하려 했으나, 각 도구가 어떤 부위들로 구성되어 있는지에 대한 명시적인 연결 관계를 정의하지 않았다.

### Part-aware Panoptic Segmentation

자연어 이미지 분야의 TAPPS와 같은 모델들이 객체와 부위를 공동으로 예측하는 shared-query 방식을 제안하였다. 하지만 저자들은 TAPPS가 수술 도구 도메인으로 직접 전이되지 않는다고 지적한다. 그 이유는 자연어 이미지의 객체들은 서로 다른 부위 구성을 가지는 반면(예: 자동차는 팔다리가 없음), 수술 도구들은 서로 다른 종류일지라도 동일한 부위 타입(예: 모든 도구가 shaft를 가짐)을 공유하는 경우가 많기 때문이다.

### 약지도 인스턴스 세그멘테이션

기존의 약지도 학습은 주로 bounding box나 point-level 라벨을 사용하는 방식이었다. 하지만 수술 도구 도메인의 특이점은 라벨의 형태가 '약한' 것이 아니라, 데이터셋마다 제공되는 '정보의 수준(granularity)'이 서로 다르다는 점이다. 따라서 본 논문은 불완전하게 라벨링된 서로 다른 데이터셋을 통합하여 학습하는 전략을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조

SurgPIS는 Mask2Former를 기반으로 확장되었다. 전체 파이프라인은 Backbone(ResNet 또는 Swin Transformer)과 Pixel Decoder를 통해 고해상도 특징 맵 $F \in \mathbb{R}^{C_{\epsilon} \times H \times W}$를 추출하고, Transformer Decoder를 통해 $N_q$개의 학습 가능한 Instrument-level query $Q$를 생성한다.

### 주요 구성 요소 및 절차

**1. 부위 특화 쿼리 변환 (Part-specific Query Transformation)**
SurgPIS의 핵심 아이디어는 도구 수준의 쿼리 $Q$를 MLP(Multi-layer Perceptron)에 통과시켜 부위 특화 쿼리 $Q_{part} \in \mathbb{R}^{(C_{part} \times N_q) \times C_{\epsilon}}$를 생성하는 것이다.

- $\hat{y}_j$: $j$번째 쿼리가 예측하는 도구 인스턴스 마스크
- $\hat{m}_{j,k}$: $j$번째 도구 인스턴스 내의 $k$번째 부위 마스크
이렇게 생성된 $\hat{m}_{j,k}$는 $Q_{part}$와 특징 맵 $F$의 내적 및 시그모이드 활성화 함수를 통해 도출된다.

**2. 부위 인식 이분 매칭 (Part-aware Bipartite Matching)**
예측값과 Ground Truth(GT)를 매칭할 때, 단순한 클래스와 마스크 손실뿐만 아니라 부위 수준의 마스크 손실 $L_{pm}$을 비용 행렬(cost matrix)에 포함시킨다.
$$L_{pm}^{i \neq 0, j} = \frac{1}{C_{part}} \sum_{k=1}^{C_{part}} \ell_M(\hat{m}_{j,k}, m_{i,k})$$
여기서 $\ell_M$은 Focal loss와 Dice loss의 조합이다. 이를 통해 부위 정보가 매칭 과정에 직접적으로 반영된다.

**3. 약지도 학습 전략 (Weakly-supervised Learning)**
PIS 라벨이 부족한 문제를 해결하기 위해 Teacher-Student 프레임워크를 사용한다.

- **Student 모델**: 강한 증강(Strong Augmentation)이 적용된 이미지를 입력받아 학습한다.
- **Teacher 모델**: 약한 증강(Weak Augmentation)이 적용된 이미지를 입력받으며, Student 모델의 가중치를 EMA(Exponential Moving Average) 방식으로 업데이트 받는다.
- **마스크 집계 (Mask Aggregation)**: PSS 라벨만 있는 데이터셋의 경우, Student의 PIS 예측값을 다음과 같이 집계하여 PSS 맵 $\hat{s}$를 생성한다.
  $$\rho_k = \sum_{j|\hat{\gamma}_j > 0} \hat{c}_j[\hat{\gamma}_j] \cdot \hat{m}_{j,k} \quad (k > 0)$$
  여기서 $\hat{\gamma}_j$는 $j$번째 쿼리의 가장 확률 높은 도구 클래스이다. 이렇게 생성된 $\hat{s}$를 GT PSS 라벨과 비교하여 손실을 계산한다.

### 손실 함수 (Loss Functions)

- **완전 지도 학습 단계**: $L_{sup} = L_{ic} + L_{im} + L_{pm}$ (클래스 손실, 인스턴스 마스크 손실, 부위 마스크 손실의 합)
- **약지도 학습 단계**:
  $$L_{wks} = L_{sup}^{teach} + \begin{cases} L_{wks}^{pss}, & \text{if PSS label} \\ L_{wks}^{iis}, & \text{if IIS label} \end{cases}$$
  여기서 $L_{sup}^{teach}$는 Teacher가 생성한 Pseudo-label과 Student 예측 간의 일관성 손실이다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis2017(IIS 위주), EndoVis2018(PIS 구축), SAR-RARP50(PSS 위주), GraSP(외부 검증용 PIS 라벨 추가 생성)
- **지표**: PartPQ (PIS 성능), PartIoU (PSS 성능), PQ 및 ChIoU (IIS 성능)
- **기준선 (Baseline)**: Mask2Former 기반으로 PSS와 IIS 모델을 각각 학습시킨 뒤 마스크를 곱하여 결합한 $BPSS \oplus BIIS$ 방식을 강력한 베이스라인으로 설정하였다.

### 주요 결과

1. **PIS 성능**: EndoVis2018 테스트 셋에서 SurgPIS(RN-50)는 PartPQ 기준 72.96%를 기록하여, 베이스라인($BPSS \oplus BIIS$) 대비 11.07 percentage points(pp) 향상된 성능을 보였다.
2. **타 작업 성능**: PIS 모델임에도 불구하고, 결과물을 집계하여 평가했을 때 IIS 작업에서 ISINet을, PSS 작업에서 MATIS를 상회하는 성능을 기록하였다. 이는 세밀한 부위 정보를 함께 학습하는 것이 상위 인스턴스 수준의 이해도를 높인다는 것을 시사한다.
3. **일반화 능력**: 학습에 사용되지 않은 GraSP 데이터셋에서도 PartPQ 60.68%를 달성하며, 자연어 이미지 기반의 TAPPS(27.83%)나 베이스라인(55.17%)보다 뛰어난 강건함을 보였다.
4. **효율성**: ResNet-50 백본 사용 시 30 FPS의 추론 속도를 보여 실시간 수술 보조 시스템에 적용 가능함을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 수술 도구의 계층적 구조(인스턴스 $\rightarrow$ 부위)를 쿼리 변환이라는 단순하지만 강력한 구조로 풀어냈다. 특히, 서로 다른 수준의 라벨을 가진 데이터셋들을 마스크 집계 방식을 통해 하나의 모델로 통합 학습시킨 점이 매우 인상적이다. 실험 결과는 부위 수준의 세밀한 정보(Fine-grained information)를 학습하는 것이 결과적으로 인스턴스 수준의 세그멘테이션 성능까지 함께 끌어올리는 시너지 효과가 있음을 보여준다.

### 한계 및 논의사항

- **데이터 의존성**: 약지도 학습 전략을 사용하더라도, 초기 단계에서는 소규모의 완전한 PIS 데이터셋($D_{PIS}$)으로 사전 학습이 필요하다는 전제가 있다. 만약 완전히 라벨이 없는 상태에서 시작해야 한다면 성능 저하가 있을 수 있다.
- **백본의 영향**: DINOv2와 같은 거대 Foundation Model을 사용하면 성능은 소폭 상승하지만, 파라미터 수와 연산량이 급격히 증가하여 실시간성(Real-time)이 훼손된다. 의료 현장에서는 정확도와 속도의 트레이드-오프에 대한 정밀한 조율이 필요하다.

## 📌 TL;DR

SurgPIS는 수술 도구의 인스턴스와 그 구성 부위를 동시에 세그멘테이션하는 **최초의 통합 PIS 모델**이다. **부위 특화 쿼리 변환**을 통해 도구-부위 간 계층 구조를 학습하며, **Teacher-Student 기반 약지도 학습**을 통해 라벨이 불완전한 여러 데이터셋을 효율적으로 활용한다. 결과적으로 PIS뿐만 아니라 IIS, PSS 등 모든 수준의 세그멘테이션 작업에서 SOTA 성능을 달성하였으며, 실시간 추론이 가능하여 실제 로봇 수술 시스템으로의 확장 가능성이 매우 높다.
