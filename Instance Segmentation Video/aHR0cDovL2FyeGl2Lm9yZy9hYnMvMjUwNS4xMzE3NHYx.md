# FlowCut: Unsupervised Video Instance Segmentation via Temporal Mask Matching

Alp Eren Sari, Paolo Favaro (2025)

## 🧩 Problem to Solve

본 논문은 비지도 비디오 인스턴스 분할(Unsupervised Video Instance Segmentation, UVIS) 문제를 해결하고자 한다. 비디오 인스턴스 분할은 비디오 감시, 자율 주행, 비디오 편집 등 다양한 분야에서 필수적인 기술이지만, 각 프레임의 객체별 마스크를 생성하는 고품질 라벨링 작업은 비용과 시간이 매우 많이 소요된다는 치명적인 단점이 있다.

기존의 비지도 학습 기반 접근 방식들은 다음과 같은 한계를 지닌다. 첫째, Optical Flow(광학 흐름)에만 의존하는 방식은 모션 모델이 일반화되지 않는 시나리오나 객체의 움직임이 거의 없는 정적인 장면에서 성능이 급격히 저하된다. 둘째, Self-Supervised Learning(SSL) 기반의 표현 학습 방식은 주로 두드러진 객체(Salient objects)에만 집중하는 경향이 있어, 움직이는 다양한 인스턴스를 정확하게 분할하는 데 어려움이 있다.

따라서 본 논문의 목표는 Optical Flow의 모션 큐와 SSL의 강력한 표현 능력을 결합하여, 사람이 직접 라벨링하지 않고도 다중 인스턴스를 효과적으로 분할하고 추적할 수 있는 고품질의 의사 라벨(Pseudo-labels) 데이터셋을 구축하고 이를 통해 모델을 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비지도 방식으로 고품질의 비디오 의사 라벨 데이터셋을 구축하기 위한 **3단계 파이프라인(Three-stage framework)**을 제안하는 것이다.

1.  **모션-외형 결합 의사 마스크 생성**: RGB 이미지의 특징뿐만 아니라 Optical Flow의 시각화 특징을 함께 사용하여, 객체의 외형과 움직임 정보를 동시에 고려한 의사 마스크(Pseudo-mask)를 생성한다.
2.  **시간적 마스크 매칭(Temporal Mask Matching)**: 단순히 개별 프레임을 처리하는 것이 아니라, 인접한 프레임 간의 $\text{Intersection-over-Union (IoU)}$를 계산하여 일관성 있는 인스턴스를 매칭하고 필터링함으로써 고품질의 짧은 비디오 세그먼트를 구축한다.
3.  **비디오 분할 모델 학습**: 이렇게 정제된 의사 라벨 데이터셋을 사용하여 VideoMask2Former와 같은 강력한 모델을 학습시켜 최종적인 비지도 비디오 인스턴스 분할 성능을 확보한다.

## 📎 Related Works

### 관련 연구 및 한계
- **Supervised Methods**: 높은 성능을 보이지만 대규모 데이터셋에 적용하기에는 라벨링 비용이 너무 크다.
- **SSL representations (DINO, DINOv2)**: 이미지 내 객체 경계를 찾는 데 탁월하지만, 비디오에서의 시간적 일관성을 보장하거나 모든 움직이는 객체를 포착하는 데 한계가 있다.
- **Spectral Methods (Normalized Cuts)**: 친밀도 행렬(Affinity matrix)을 통해 세그멘테이션을 수행하지만, 고품질의 행렬을 구성하는 것이 어렵다.
- **Unsupervised Video Segmentation**: Optical Flow를 활용하는 방식(MotionGroup, OCLR 등)이 많으나, 모션 모델의 일반화 성능 문제나 정적 장면에서의 취약점이 존재한다.
- **VideoCutler**: 이미지넷(ImageNet)의 단일 이미지들로 인위적인 비디오를 만들어 학습시킨 방식으로, 실제 비디오 데이터의 분포와 맞지 않는 Out-of-distribution 문제가 발생한다.

### 차별점
FlowCut은 기존의 VideoCutler처럼 가공의 비디오를 만드는 대신, **실제 비디오 데이터셋(YouTubeVIS-2021)에서 직접 의사 라벨을 추출**하고, 이를 **시간적 매칭 알고리즘을 통해 정제**하여 실제 데이터 분포에 최적화된 학습 데이터를 구축한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

FlowCut의 전체 프로세스는 다음의 세 단계로 구성된다.

### 1. Pseudo-Mask Estimation (의사 마스크 추정)
각 비디오 프레임에서 인스턴스 마스크를 추출하기 위해 RGB 특징과 Optical Flow 특징을 결합한 친밀도 행렬(Affinity Matrix)을 사용한다.

- **특징 추출**: 입력 프레임 $I$와 RAFT로 계산된 Optical Flow의 시각화 이미지 $I_{of}$를 DINO 백본 $f$에 통과시켜 특징 $h_{rgb}$와 $h_{of}$를 추출한다.
- **친밀도 행렬 구성**: 두 특징 간의 코사인 유사도를 이용해 RGB 친밀도와 Flow 친밀도를 각각 계산한 후, 이를 볼록 조합(Convex combination)한다.
  $$w_{ij} = \begin{cases} 1, & \text{when } \alpha \langle h_{rgb}^i, h_{rgb}^j \rangle + (1-\alpha) \langle h_{of}^i, h_{of}^j \rangle > \tau \\ \epsilon, & \text{otherwise} \end{cases}$$
  여기서 $\alpha$는 가중치, $\tau$는 임계값, $\epsilon$은 그래프 연결성을 유지하기 위한 작은 값이다.
- **반복적 추출**: TokenCut의 Normalized Cuts 방식을 적용하여 마스크를 생성하며, Cutler의 방식을 따라 이미 추출된 마스크 영역의 친밀도를 제거하며 반복적으로 다중 인스턴스를 추출한다.

### 2. Automated Dataset Curation (자동 데이터셋 큐레이션)
추출된 의사 마스크들은 프레임마다 인덱스가 일치하지 않거나 일부 프레임에서 누락될 수 있다. 이를 해결하기 위해 다음의 필터링 알고리즘을 수행한다.

- **프레임 쌍 구성**: 최대 4프레임 간격의 두 프레임($F_1, F_2$)을 하나의 쌍으로 묶는다.
- **IoU 기반 매칭**: $F_1$의 모든 인스턴스와 $F_2$의 모든 인스턴스 간의 $\text{IoU}$ 행렬을 계산한다.
- **필터링 조건**: $F_1$의 특정 인스턴스에 대해 $F_2$에서 가장 높은 $\text{IoU}$를 가진 인스턴스를 찾고, 그 값이 $0.5$보다 높을 때만 유효한 쌍으로 인정하여 저장한다. 이 과정을 통해 일관성이 없는 마스크는 제거된다.

### 3. Video Segmentation Model Training (모델 학습)
정제된 2프레임 길이의 짧은 비디오 세그먼트들을 사용하여 비디오 분할 모델을 학습시킨다.

- **모델 아키텍처**: ResNet-50 백본을 가진 VideoMask2Former를 사용한다.
- **학습 절차**: YouTubeVIS-2021에서 생성한 의사 라벨 데이터셋(약 167,365개 세그먼트)을 사용하여 학습하며, DAVIS-2017 실험에서는 ImageNet의 의사 라벨 데이터 50,000개를 추가하여 경계 정밀도를 높였다. 후처리로 CRF(Conditional Random Fields)를 적용하여 경계를 더욱 정교하게 다듬었다.

## 📊 Results

### 실험 설정
- **데이터셋**: YouTubeVIS-2019, YouTubeVIS-2021, DAVIS-2017, DAVIS-2017 Motion.
- **비교 대상**: MotionGroup, OCLR, VideoCutler 등.
- **평가 지표**:
    - YouTubeVIS: $\text{Average Precision (AP)}$, $\text{Average Recall (AR)}$.
    - DAVIS: $\text{Region measure (J)}$, $\text{Boundary measure (F)}$, 및 그 평균인 $\text{J\&F}$.

### 주요 결과
FlowCut은 모든 벤치마크에서 기존 SOTA(State-of-the-art) 성능을 경신하였다.

- **DAVIS-2017**: $\text{J\&F}$ 지표에서 43.5를 기록하며 VideoCutler(42.4)를 앞섰다. 특히 DAVIS-2017 Motion에서는 $\text{J\&F}$ 58.3으로 매우 높은 성능을 보였다.
- **YouTubeVIS-2021**: $\text{AP}$ 18.0을 달성하여 VideoCutler(17.4) 대비 향상된 성능을 보였으며, 특히 작은 객체($\text{AP}_S$)와 중간 크기 객체($\text{AP}_M$)에서 유의미한 상승이 있었다.
- **정성적 결과**: Figure 1에서 확인되듯, 자전거와 사이클리스트를 동시에 정확히 추적하거나, 복잡한 댄스 장면에서 여러 사람을 명확히 구분해내는 능력이 기존 방식보다 뛰어남이 확인되었다.

### Ablation Study
- **Optical Flow의 영향**: $\alpha=1$로 설정하여 Optical Flow 정보를 제거했을 때, $\text{AP}$ 성능이 크게 하락하였다(예: YouTubeVIS-2019에서 $\text{AP}$ 25.1 $\rightarrow$ 22.0). 이는 모션 큐가 비디오 인스턴스 분할에 필수적임을 입증한다.
- **인-도메인(In-domain) 데이터**: 테스트 데이터와 동일한 데이터셋에서 큐레이션된 의사 라벨로 학습했을 때 성능이 가장 높게 나타났다.

## 🧠 Insights & Discussion

### 강점
FlowCut은 복잡한 수동 라벨링 없이도 실제 비디오 데이터의 분포를 활용해 고품질의 학습 셋을 구축함으로써, 계산 자원을 적게 사용하면서도 SOTA 성능을 달성하였다. 특히 RGB와 Flow의 특징을 결합한 친밀도 행렬과 시간적 매칭 기반의 큐레이션 과정이 시너지를 내어 다중 인스턴스 추적 성능을 높였다.

### 한계 및 미해결 과제
1.  **DINO 특징의 의존성**: 의사 마스크 생성의 기초가 되는 DINO 특징이 완벽하지 않아, 잘못된 마스크가 생성될 가능성이 있다.
2.  **작은 객체 탐지 어려움**: 추출된 특징 맵의 해상도가 원본 이미지보다 낮기 때문에, 매우 작은 객체를 정밀하게 분할하는 데 한계가 있다.
3.  **정적 돌출 영역(Static Salient Regions)**: 움직임은 없지만 시각적으로 매우 두드러진 영역(예: 파도, 나무 무늬 등)을 객체로 오인하여 분할하는 경우가 발생한다.
4.  **심한 가림(Severe Occlusion)**: 객체가 다른 객체에 의해 완전히 가려지는 경우 시간적 매칭이 끊기거나 마스크가 손실되는 문제가 있다.

### 비판적 해석
본 논문은 비지도 학습의 핵심인 '데이터 품질' 문제를 시간적 일관성(Temporal Consistency)이라는 제약 조건을 통해 영리하게 해결하였다. 다만, 현재의 방식은 프레임 간의 국소적인(Local) 매칭에 의존하고 있으므로, 비디오 전체 시퀀스의 글로벌한 맥락을 반영한다면 가림 문제나 일시적인 탐지 실패 문제를 더 효과적으로 해결할 수 있을 것으로 판단된다.

## 📌 TL;DR

FlowCut은 **[RGB+Optical Flow 친밀도 $\rightarrow$ 시간적 IoU 매칭 $\rightarrow$ VideoMask2Former 학습]**으로 이어지는 3단계 파이프라인을 통해 비지도 방식으로 고품질의 비디오 인스턴스 분할 데이터셋을 구축하는 방법론이다. 이 연구는 실제 비디오 데이터에서 의사 라벨을 추출하여 학습시킨 최초의 시도로, YouTubeVIS 및 DAVIS 벤치마크에서 SOTA 성능을 달성하며 비지도 비디오 인스턴스 분할의 실용 가능성을 입증하였다.