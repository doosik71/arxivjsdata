# SANGRIA: Surgical Video Scene Graph Optimization for Surgical Workflow Prediction

Çağhan Köksal, Ghazal Ghazaei, Felix Holm, Azade Farshad, and Nassir Navab (2024)

## 🧩 Problem to Solve

본 논문은 수술 비디오 분석에서 핵심적인 과제인 수술 워크플로우 예측(Surgical Workflow Prediction), 특히 수술 단계 분할(Surgical Phase Segmentation) 문제를 다룬다. 수술 비디오의 전반적인 장면 이해를 위해 Scene Graph 기반의 표현 방식이 효과적임이 입증되었으나, 이를 구현하기 위해서는 각 프레임에 대한 조밀한 시맨틱 어노테이션(dense semantic annotations)이 필요하다는 치명적인 한계가 있다. 수술 데이터의 특성상 전문가의 정밀한 라벨링 작업은 비용이 매우 높고 시간이 많이 소요되며, 전문적인 플랫폼이 필요하여 데이터 확보가 매우 어렵다. 따라서 본 연구의 목표는 조밀한 픽셀 단위 라벨 없이, 오직 약한 감독 신호인 수술 단계 라벨(weak surgical phase labels)만을 사용하여 비지도 방식으로 시맨틱 장면 그래프(Semantic Scene Graph)를 생성하고 이를 통해 수술 워크플로우를 예측하는 엔드투엔드(end-to-end) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기초 모델(Foundation Models)의 일반화 능력과 그래프 기반의 Spectral Clustering을 결합하여, 별도의 시맨틱 라벨 없이도 학습 가능한 속성을 가진 비지도 장면 그래프를 생성하는 것이다. 구체적으로는 다음의 세 가지 기여를 제시한다.

첫째, LightGlue를 도입하여 인접 프레임 간의 희소한 국소 특징 매칭(local feature matching)을 수행함으로써, 비디오 시퀀스에서 시간적 일관성(temporal consistency)을 갖춘 Dynamic Scene Graph(DSG)를 생성하는 모듈을 제안한다.

둘째, 생성된 DSG의 노드 특징과 관계를 하위 작업인 수술 단계 분할 작업과 함께 공동 최적화(joint optimization)함으로써, 어노테이션 부족 문제를 해결하고 작업 특성에 최적화된 장면 표현을 학습한다.

셋째, 소수의 샘플만을 사용하는 Prototype Matching 메커니즘을 통해 비지도 방식으로 생성된 클러스터에 시맨틱 클래스 정체성을 부여함으로써, 최소한의 어노테이션만으로도 효율적인 장면 그래프 생성이 가능함을 입증한다.

## 📎 Related Works

기존의 비지도 장면 및 비디오 분할 연구들은 주로 Optical Flow나 Normalized Cut의 변형, 그리고 Self-training 방식을 사용하여 장면 내의 주요 객체를 탐색해 왔다. 수술 장면 이해 분야에서도 Optical Flow나 형태 사전 지식(shape priors)이 사용되었으나, 형태 사전 지식은 새로운 도구나 환경에 대한 적응력이 떨어지며, Optical Flow는 수술 중 발생하는 해부학적 움직임이나 체액으로 인한 노이즈에 취약하다는 한계가 있다. 최근 DINO와 같은 기초 모델들이 등장하며 최소한의 어노테이션으로 장면 이해가 가능해졌으나, 일반적인 컴퓨터 비전 도메인과 의료 도메인 사이의 큰 간극으로 인해 여전히 미세 조정(fine-tuning)이 필요하다. 본 논문은 이러한 한계를 극복하기 위해, 기존의 정적 그래프 표현을 넘어 시간적 관계를 통합하고 하위 작업의 라벨을 통해 그래프 구조를 최적화하는 방식으로 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 구조

SANGRIA의 전체 파이프라인은 크게 세 단계로 구성된다: 1) Spectral Temporal Clustering을 통한 초기 Dynamic Scene Graph 생성, 2) 하위 작업을 통한 DSG Optimization, 3) GCN 기반의 Task Prediction이다.

### 1. Dynamic Scene Graph Generation

먼저 입력 이미지 $I$를 $n$개의 패치로 나누고 DINO 모델을 통해 특징 벡터 $f$를 추출한다. 패치 간의 유사도를 기반으로 정적 인접 행렬(Adjacency Matrix) $A$를 다음과 같이 생성한다.

$$A_{ij} = \begin{cases} f_i \cdot f_j & \text{if } f_i \cdot f_j > 0 \\ 0 & \text{otherwise} \end{cases}$$

단순히 프레임별로 클러스터링을 수행하면 시간적 일관성이 결여되므로, 본 연구에서는 LightGlue를 사용하여 인접 프레임 간의 특징점 매칭을 수행하고 희소한 시간적 연결(sparse temporal links)을 추가한다. 윈도우 크기 $w$ 내의 프레임들에 대해 공간적 엣지 $E^t$와 시간적 엣지 $E^{t \to t+1}$를 합쳐 동적 패치 그래프 $G_{t_i \to t_{i+w}} = (V, E)$를 구축한다. 이때 시간적/공간적 인코딩을 추가하여 객체의 상대적 위치와 순서 정보를 보강한다.

이후 Deep Modularity Networks(DMON)를 사용하여 이 그래프를 클러스터링함으로써, 최종적으로 풀링된 장면 그래프인 $G_{pool} = (V_{pool}, E_{pool})$을 생성한다. 여기서 $V_{pool}$의 크기는 클러스터의 개수인 $K$가 된다.

### 2. DSG Optimization 및 Task Prediction

생성된 DSG의 엣지 가중치 $W_{pool}$을 다음과 같이 학습 가능한 MLP를 통해 추정한다.

$$W_{pool} = \sigma(\text{MLP}(X_{pool}; \Theta_{\text{MLP}}))$$

이러한 유연한 설정을 통해 비지도 클러스터링 결과의 불확실성을 보완하고 하위 작업에 최적화된 관계를 학습할 수 있다. 최종적으로 이 DSG는 다층 GCN(Graph Convolutional Network)의 입력으로 들어가며, Global Sum-pooling과 Softmax 층을 거쳐 수술 단계(Phase)를 예측한다.

전체 시스템은 DMON의 손실 함수 $L_u$와 수술 단계 분할을 위한 교차 엔트로피 손실 함수 $L_{CE}$를 합친 공동 손실 함수를 통해 최적화된다.

$$L_{joint} = L_u + L_{CE}$$

### 3. Prototype Matching

비지도 방식으로 생성된 클러스터에 실제 의미(Semantic Class)를 부여하기 위해 Prototype Matching을 사용한다. 클래스당 단 5개의 어노테이션만 사용하여 해당 클래스 패치들의 DINO 특징 평균값을 프로토타입으로 설정하고, 각 클러스터 노드와의 코사인 유사도를 계산하여 가장 유사한 클래스를 할당한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CATARACTS (50개 비디오, 19개 단계), Cataract101 (101개 비디오, 11개 단계), CaDIS (픽셀 단위 어노테이션 이미지)를 사용하였다.
- **지표**: 수술 단계 예측에서는 Accuracy와 F1-score를, 시맨틱 분할에서는 mIoU와 Pixel-wise Accuracy(PAC)를 측정하였다.

### 주요 결과

1. **수술 워크플로우 예측**: CATARACTS 데이터셋에서 기존의 최신 그래프 기반 SOTA 모델(Holm et al.) 대비 Accuracy는 8%, F1-score는 10% 향상된 성능을 보였다. 비-그래프 기반 베이스라인(CNN 기반)보다도 Accuracy 6%, F1-score 4% 높은 성능을 기록하였다.
2. **절제 연구(Ablation Study)**: 윈도우 크기가 커지고 시간적 임베딩(Temporal Embeddings)이 추가될수록 예측 성능이 향상되었으나, 공간적 임베딩의 영향은 미미했다. 이는 수술 단계 인식에서 시간적 정보가 매우 중요함을 시사한다.
3. **시맨틱 분할**: 제안된 DMON 기반 클러스터링이 MaskCut 등 기존 비지도 객체 발견 방법보다 높은 mIoU를 기록하였다. 특히 하위 작업과의 공동 최적화를 통해 'Primary Knife'와 같은 핵심 도구의 분할 성능이 크게 향상됨을 확인하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 조밀한 라벨 없이도 수술 장면의 구성 요소와 그들 간의 관계를 효과적으로 학습했다는 점이다. 특히 하위 작업(단계 분할)의 라벨을 활용해 그래프를 최적화함으로써, 모델이 현재 수술 단계에서 가장 중요한 도구나 조직에 더 집중하게 만드는 '가이드' 효과를 얻었다. 이는 단순한 비지도 학습보다 훨씬 정교한 장면 표현이 가능함을 보여준다.

다만, Prototype Matching 단계에서 여전히 소량의 정답 마스크가 필요하다는 점과, 시간적 윈도우 크기를 늘릴수록 계산 비용이 증가한다는 점이 한계로 지적될 수 있다. 또한, 도구 분할(mIoU ins) 성능이 해부학적 구조 분할(mIoU ana)보다 낮게 나타나는데, 이는 수술 도구의 외형적 다양성과 복잡한 상호작용 때문으로 해석된다.

## 📌 TL;DR

SANGRIA는 조밀한 픽셀 라벨 없이 **수술 단계 라벨(weak labels)**만을 사용하여 **비지도 방식으로 Dynamic Scene Graph를 생성하고 최적화**하는 프레임워크이다. LightGlue를 통한 시간적 일관성 확보와 하위 작업과의 공동 최적화를 통해, CATARACTS 데이터셋에서 기존 SOTA 대비 **Accuracy 8%, F1-score 10% 향상**이라는 뛰어난 성능을 달성하였다. 이 연구는 어노테이션 비용이 높은 의료 영상 분야에서 효율적인 장면 이해와 워크플로우 예측을 가능케 하는 중요한 방법론을 제시한다.
