# UVIS: Unsupervised Video Instance Segmentation

Shuaiyi Huang, Saksham Suri, Kamal Gupta, Sai Saketh Rambhatla, Ser-nam Lim, Abhinav Shrivastava (2024)

## 🧩 Problem to Solve

비디오 인스턴스 분할(Video Instance Segmentation, VIS)은 비디오 프레임 전반에 걸쳐 모든 객체를 분류(Classifying), 분할(Segmenting), 그리고 추적(Tracking)하는 작업이다. 이 작업은 객체의 외형 변화, 가려짐(Occlusion), 복잡한 배경 등으로 인해 매우 도전적인 과제이다.

기존의 VIS 접근 방식들은 대량의 정밀한 어노테이션(Dense annotations)을 필요로 한다. 이를 해결하기 위해 COCO와 같은 이미지 데이터셋에서 사전 학습(Pre-training)을 수행하거나, 일부 프레임만 라벨링하는 방식, 혹은 바운딩 박스나 카테고리 라벨만 사용하는 약지도 학습(Weakly-supervised) 방식이 제안되었다. 그러나 이러한 방법들은 여전히 어느 정도의 인간 개입(어노테이션)이 필요하거나, 사전 학습된 이미지 데이터셋의 카테고리와 겹치는 클래스만 처리할 수 있다는 한계가 있다.

본 논문의 목표는 비디오 어노테이션이나 정밀 라벨 기반의 사전 학습 없이, 주어진 카테고리 셋만으로 모든 클래스를 처리할 수 있는 완전 비지도 학습 기반의 VIS 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최신 비전 파운데이션 모델(Vision Foundation Models)의 능력을 결합하는 것이다. 구체적으로는 **DINO**의 조밀한 형태 사전 정보(Dense shape prior)와 **CLIP**의 오픈셋 인식 능력(Open-set recognition ability)을 활용하여 정답 라벨 없이도 객체를 분할하고 인식하고자 한다.

가장 주요한 기여는 다음과 같다:

- 비디오 어노테이션이나 정밀한 사전 학습 없이 작동하는 최초의 비지도 VIS 프레임워크를 제안하였다.
- 의사 라벨(Pseudo-label)의 품질을 높이기 위한 **Prototype Memory Filtering (PMF)**과 추적의 일관성을 유지하기 위한 **Tracking Memory Bank**로 구성된 **Dual-memory 디자인**을 도입하였다.
- YoutubeVIS-2019, YoutubeVIS-2021, Occluded VIS 등 세 가지 표준 벤치마크에서 그 가능성을 입증하였다.

## 📎 Related Works

- **Video Object Segmentation (VOS):** 배경에서 전경 객체를 분리하는 이진 분류 문제로, 주로 첫 프레임의 마스크가 주어진 상태에서 추적하는 Semi-supervised VOS가 주를 이룬다. UVIS는 여기서 더 나아가 분류와 추적까지 동시에 수행해야 하므로 훨씬 더 어려운 문제에 도전한다.
- **Supervised VIS:** 최근 Transformer 기반 모델들이 주류를 이루고 있으며, 특히 **MinVIS**는 이미지 기반 학습만으로도 비디오 추적이 가능함을 보였다. UVIS는 MinVIS의 구조를 기반으로 하되, 지도 학습 데이터 대신 비지도 방식으로 생성한 의사 라벨을 사용한다.
- **Weakly/Semi-supervised VIS:** 프레임별 카테고리 라벨이나 일부 샘플링된 프레임의 어노테이션을 사용한다. 반면 UVIS는 어떤 형태의 프레임별 라벨이나 COCO 사전 학습 없이 작동한다는 점에서 차별화된다.
- **VL Models 기반 분할:** CLIP과 DINO 같은 모델들이 이미지 수준의 제로샷 분할 능력을 보였으나, 이를 비디오 인스턴스 분할(VIS) 작업에 통합하려는 시도는 본 논문이 처음이다.

## 🛠️ Methodology

UVIS 프레임워크는 크게 세 단계의 파이프라인으로 구성된다.

### 1. 프레임 수준의 의사 라벨 생성 (Pseudo-label Generation)

먼저 각 프레임에 대해 클래스 구분 없는 마스크를 생성하고, 여기에 의미론적 라벨을 부여한다.

- **Class-agnostic Mask Generation:** 사전 학습된 self-supervised 모델인 **DINO**와 **CutLER**를 사용하여 각 프레임 $V_t$에서 바운딩 박스 $\{b_{it}\}$, 마스크 $\{M_{it}\}$, 그리고 객체성 점수(Objectness score) $\{o_{it}\}$를 추출한다.
- **CLIP 기반 텍스트-인스턴스 매칭:** 추출된 각 인스턴스 영역 $b_{it}^\oplus$를 CLIP의 이미지 인코더($f_{CLIP}^{vision}$)에 넣고, 정의된 카테고리 셋 $C$의 텍스트 프롬프트($f_{CLIP}^{text}$)와의 코사인 유사도를 계산하여 클래스를 할당한다.
  $$class(i) = \arg \max_{l \in C} \left( f_{CLIP}^{vision}(b_{it}^\oplus) \cdot f_{CLIP}^{text}(\text{a photo of } \langle l \rangle) \right)$$
- **Prototype Memory Filtering (PMF):** CLIP으로 생성된 초기 라벨은 노이즈가 많다. 이를 해결하기 위해 클래스별 prototype 메모리 뱅크를 구축한다. 각 클래스 $l$에 대해 CLIP 특징값들을 K-Means 클러스터링하여 중심점(Centroids)을 계산하고, 인스턴스의 특징값이 이 중심점들과의 최대 유사도가 임계값 $\tau$보다 낮으면 가짜 양성(False Positive)으로 간주하여 제거한다.

### 2. Transformer 기반 VIS 모델 학습

생성된 의사 라벨을 사용하여 인스턴스 분할 모델을 학습시킨다. 모델은 Convolutional 이미지 인코더 $E$와 Transformer 디코더 $D$로 구성된다.

- **학습 절차:** 인코더를 통해 추출된 특징 $F_t$와 학습 가능한 쿼리 $q \in Q$를 디코더에 입력하여 변환된 쿼리 $\hat{q}$를 얻는다.
- **출력:** $\hat{q}$는 분류 헤드 $f_{cls}$를 통해 클래스 점수 $s$를 생성하고, 인코더 특징 $F_t$와 컨볼루션 연산을 통해 세그멘테이션 마스크 $M$을 생성한다.
  $$M = \sigma(\hat{q} * F_t^{-1})$$
- **손실 함수:** 예측값과 의사 라벨 간의 Bipartite matching을 수행한 후, 분류 손실($L_{cls}$, Cross Entropy)과 분할 손실($L_{seg}$, Binary Cross Entropy + Dice Loss)의 합을 최소화한다.
  $$L_{vis} = L_{cls} + L_{seg}$$

### 3. 쿼리 기반 추적 및 메모리 활용

추론 단계에서는 프레임 간의 일관성을 유지하기 위해 쿼리 임베딩을 활용한다.

- **Query-based Tracking:** 연속된 프레임 $V_t$와 $V_{t+1}$의 쿼리 $Q_t, Q_{t+1}$ 사이에서 코사인 유사도 기반의 헝가리안 매칭(Hungarian matching)을 수행하여 인스턴스를 연결한다.
- **Tracking Memory Bank:** 단순한 프레임 간 매칭의 불안정성을 해결하기 위해 장기적인 시간 정보를 저장하는 메모리 모듈을 제안한다. 현재 프레임의 쿼리를 매칭할 때, 이전 프레임들의 가중 평균 쿼리 특징을 사용한다.
  $$Q_t^* = \lambda * P_{t-1}[Q_t] + (1-\lambda) * \mu_{t-1}$$
  여기서 $\mu_{t-1}$은 다음과 같이 정의되는 평균 메모리이다.
  $$\mu_{t-1} = \frac{1}{t-1} \sum_{i=1}^{t-2} (Q_1 + P_1[Q_2] + \dots + P_{t-2}[Q_{t-1}])$$

## 📊 Results

### 실험 설정

- **데이터셋:** YouTube-VIS 2019, YouTube-VIS 2021, Occluded VIS.
- **지표:** AP (Average Precision), AR (Average Recall).
- **백본:** ResNet-50.
- **비교 대상:** IDOL, MinVIS (Supervised), MaskFreeVIS, WeakVIS, DeepSort (Baseline).

### 정량적 결과

- **YouTube-VIS 2019:** UVIS는 어떠한 비디오 어노테이션이나 COCO 사전 학습 없이 **21.4 AP**를 달성하였다. 이는 프레임별 카테고리 라벨을 사용한 WeakVIS(10.5 AP)보다 월등히 높은 성능이며, DeepSort 기반 베이스라인(12.5 AP)보다도 우수하다.
- **YouTube-VIS 2021:** **17.5 AP**를 기록하며 DeepSort 베이스라인(10.3 AP)을 상회하였다.
- **Occluded VIS:** 가려짐이 심한 환경에서는 **3.5 AP**로 낮은 수치를 보였으나, 여전히 DeepSort(1.6 AP)보다는 개선된 결과를 보였다.

### Ablation Study

- **구성 요소 영향:** DeepSort(12.5) $\rightarrow$ 단순 학습(16.6) $\rightarrow$ 마스크/CLIP 점수 필터링(19.8) $\rightarrow$ PMF 적용(20.7) $\rightarrow$ Tracking Memory 적용(21.4) 순으로 성능이 향상됨을 확인하였다.
- **PMF 임계값:** $\tau=0.7$일 때 가장 높은 성능(20.7 AP)을 보였으며, 너무 높으면 정답 데이터까지 제거되어 성능이 하락하였다.
- **Tracking Memory:** 비지도 설정뿐 아니라 지도 학습 설정(MinVIS)에서도 성능 향상(+3.4 AP in YTVIS-2019)을 이끌어내어, 장기 메모리의 범용적 유효성을 입증하였다.

## 🧠 Insights & Discussion

### 강점

- **데이터 효율성:** 사람이 직접 라벨링한 데이터 없이 파운데이션 모델의 사전 지식만으로 VIS라는 복잡한 과제를 수행할 수 있음을 보였다.
- **범용성:** 특정 데이터셋에 종속되지 않고, 주어진 카테고리 이름만으로 대응 가능한 오픈셋 잠재력을 가지고 있다.
- **구조적 보완:** 단순한 의사 라벨 생성을 넘어, PMF와 Tracking Memory라는 두 가지 메모리 장치를 통해 비지도 학습의 고질적인 문제인 '노이즈'와 '시간적 불안정성'을 효과적으로 억제하였다.

### 한계 및 실패 사례

논문은 다음과 같은 실패 사례를 명시하고 있다:

1. **CLIP Labeling Failure:** CLIP 모델 자체가 영역을 잘못 분류하는 경우.
2. **Multi-Instance Failure:** 두 객체가 서로 가려질 때, 모델이 두 인스턴스를 하나의 마스크로 합쳐버리는 현상.
3. **Temporal Inconsistency:** 예측된 마스크가 시간에 따라 부드럽게 이어지지 않고 떨리는 현상.

### 비판적 해석

비지도 학습 모델임에도 불구하고 지도 학습 모델(MinVIS)과의 격차가 생각보다 크지 않다는 점(YTVIS-2019 기준 약 8.9 AP 차이)은 매우 고무적이다. 하지만 Occluded VIS에서의 낮은 성능은 비지도 방식이 복잡한 가려짐 상황에서 객체의 정체성을 유지하는 능력이 여전히 부족함을 시사한다. 향후 연구에서는 광학 흐름(Optical Flow)이나 더 강력한 비디오 전용 파운데이션 모델을 통합하는 방향이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 **DINO(형태 정보)**와 **CLIP(의미 정보)**을 결합하여, 비디오 어노테이션과 정밀 사전 학습 없이도 작동하는 최초의 비지도 비디오 인스턴스 분할(VIS) 프레임워크인 **UVIS**를 제안한다. **Prototype Memory**로 라벨 노이즈를 제거하고 **Tracking Memory**로 추적 일관성을 높였으며, 그 결과 YouTube-VIS 2019에서 21.4 AP라는 유의미한 성능을 달성하였다. 이 연구는 고비용의 비디오 라벨링 문제를 해결하고, 파운데이션 모델을 비디오 분석 작업으로 확장하는 중요한 가능성을 제시한다.
