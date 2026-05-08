# Visual Object Tracking with Discriminative Filters and Siamese Networks: A Survey and Outlook

Sajid Javed, Martin Danelljan, Fahad Shahbaz Khan, Muhammad Haris Khan, Michael Felsberg, and Jiri Matas (2021)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 근본적인 난제 중 하나인 시각적 객체 추적(Visual Object Tracking, VOT) 문제를 다룬다. VOT의 핵심은 이미지 시퀀스에서 대상의 초기 위치가 주어졌을 때, 이후 프레임에서 대상의 궤적과 상태(주로 Bounding Box 형태)를 정확하게 추정하는 것이다.

이 문제는 대상 객체가 부분적 또는 완전히 가려지는 Occlusion, 크기가 변하는 Scale Variation, 형태가 변하는 Deformation과 같은 기하학적 변화뿐만 아니라, 조명 변화나 모션 블러(Motion Blur)와 같은 환경적 요인으로 인해 매우 까다롭다. 특히 배경에 대상과 유사한 외형을 가진 객체가 존재할 경우 모델이 혼동을 일으켜 추적에 실패할 가능성이 높다. 따라서 본 논문의 목표는 지난 10년간 VOT 분야를 주도해 온 두 가지 핵심 패러다임인 Discriminative Correlation Filters (DCF)와 Siamese Networks (SN)를 체계적으로 분석하고, 이들의 이론적 배경, 공통 및 개별 도전 과제, 그리고 최신 성능 동향을 종합적으로 검토하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 DCF와 Siamese 추적기라는 두 가지 지배적인 방법론을 통합적인 관점에서 분석했다는 점이다. 주요 기여 사항은 다음과 같다.

1. **체계적인 이론적 배경 제공**: DCF의 선형 회귀 기반 필터 학습 방식과 Siamese Network의 유사도 학습(Similarity Learning) 구조에 대한 상세한 이론적 설명을 제공한다.
2. **공통 및 특수 과제 분석**: 두 패러다임이 공유하는 문제(특징 표현, 상태 추정, 오프라인 학습)와 각 방법론이 가진 고유한 문제(DCF의 경계 효과 및 최적화 문제, Siamese의 온라인 적응성 부족)를 명확히 구분하여 분석한다.
3. **광범위한 벤치마크 평가**: 9개의 주요 추적 벤치마크 데이터셋을 바탕으로 90개 이상의 DCF 및 Siamese 추적기의 성능을 정량적으로 비교 분석하여 최신 기술 수준(SOTA)을 제시한다.
4. **향후 연구 방향 제시**: 분석 결과를 바탕으로 세그멘테이션 기반 추적, Transformer의 도입, 그리고 오픈월드 추적 문제로의 확장 가능성 등 향후 연구를 위한 권고 사항을 제안한다.

## 📎 Related Works

논문은 지난 20년간 발표된 다양한 VOT 서베이 연구들을 언급하며 본 연구와의 차별점을 명시한다. 기존의 서베이들은 주로 포인트/특징 대응 방식, 기하학적 모델, 혹은 딥러닝 기반 추적기의 네트워크 구조나 훈련 방식에 따른 분류(Taxonomy)에 집중해 왔다.

반면, 본 논문은 최근 가장 우수한 성능을 보이는 두 가지 패러다임인 DCF와 SN에만 집중하여 분석의 깊이를 더했다. 특히 단순한 분류를 넘어, 두 방법론이 직면한 구체적인 연구 과제(Open Research Challenges)를 중심으로 분석을 진행했다는 점과, 9개의 서로 다른 벤치마크를 통해 대규모의 실험적 비교를 수행했다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

논문은 DCF와 Siamese Networks의 핵심 작동 원리를 다음과 같이 설명한다.

### 1. Discriminative Correlation Filters (DCF)

DCF는 타겟을 배경으로부터 구분하기 위한 선형 회귀 모델을 온라인으로 학습하는 방식이다.

- **핵심 아이디어**: 학습 샘플을 원형으로 시프트(Circular Shifting)하여 조밀한 샘플링을 근사화하고, 이를 통해 고속 푸리에 변환(Fast Fourier Transform, FFT)을 이용하여 연산 속도를 획기적으로 높인다.
- **학습 과정**: 최소제곱법(Least-squares) 손실 함수를 최소화하여 필터 $w$를 학습한다.
  $$L(w) = ||Xw - y||^2 + \lambda ||w||^2$$
  여기서 $X$는 데이터 행렬, $y$는 가우시안 분포 형태의 타겟 라벨, $\lambda$는 정규화 파라미터이다.
- **최적해**: 푸리에 영역에서 다음과 같은 단순한 원소별 연산으로 최적의 필터 $\hat{w}_m$을 구할 수 있다.
  $$\hat{w}_m = \frac{\sum_{j=1}^{m} \bar{\hat{x}}_j \hat{y}_j}{\sum_{j=1}^{m} \bar{\hat{x}}_j \hat{x}_j + \lambda}$$
- **추론 절차**: 학습된 필터를 현재 프레임의 관심 영역(ROI)과 컨볼루션 연산하여 응답 맵(Response Map)을 생성하고, 값이 최대가 되는 지점을 타겟의 위치로 추정한다.

### 2. Siamese Networks (SN)

Siamese Network는 타겟 이미지와 검색 영역 이미지 간의 유사도를 측정하는 함수를 오프라인으로 학습하는 방식이다.

- **구조**: 두 개의 동일한 가중치를 공유하는 CNN 서브 네트워크(Template branch, Search branch)로 구성된다.
- **핵심 연산**: 두 네트워크에서 추출된 특징 맵을 상호 상관(Cross-correlation) 연산하여 유사도 맵을 생성한다.
  $$g_\rho(x, z) = f_\rho(x) \ast f_\rho(z) + b$$
  여기서 $f_\rho(\cdot)$는 가중치 $\rho$를 가진 CNN이며, $\ast$는 상호 상관 연산자이다.
- **훈련 목표**: 타겟-타겟 쌍은 높은 유사도를, 타겟-배경 쌍은 낮은 유사도를 갖도록 로지스틱 손실(Logistic Loss)을 최소화하며 학습한다.
  $$\ell(c, v) = \frac{1}{N} \sum_{i=1}^{N} \log(1 + \exp(-c_i v_i))$$

### 3. 주요 도전 과제 및 해결책

- **DCF의 경계 효과(Boundary Artifacts)**: 원형 컨볼루션 가정으로 인해 발생하는 인위적인 경계 문제를 해결하기 위해 공간 정규화(Spatial Regularization)나 공간 영역에서의 직접 최적화 방식(ATOM, DiMP)이 제안되었다.
- **SN의 온라인 적응성(Online Adaptability)**: SN은 기본적으로 오프라인 학습 모델이므로 온라인 업데이트가 어렵다. 이를 위해 이동 평균(Moving Average) 방식이나 메모리 네트워크(MemTrack), 혹은 업데이트 네트워크(UpdateNet)를 통한 동적 템플릿 갱신 방법이 연구되었다.

## 📊 Results

본 논문은 OTB100, TC128, UAV123, UAV20L, VOT 시리즈(2016, 2018, 2020), TrackingNet, LaSOT, GOT-10K 등 9개 벤치마크에서 92개 이상의 추적기를 비교하였다.

- **측정 지표**: 정밀도(Precision Rate), 성공률(Success Rate/AUC), 기대 평균 겹침(Expected Average Overlap, EAO) 등을 사용하였다.
- **주요 정량적 결과**:
  - **DCF 계열**: 최근의 end-to-end 학습 프레임워크인 DiMP와 PrDiMP가 대부분의 벤치마크에서 최상위권 성능을 기록했다. 특히 PrDiMP는 UAV123, LaSOT, GOT-10K에서 압도적인 성능을 보였다.
  - **Siamese 계열**: SiamAttn, SiamR-CNN 등이 우수한 성능을 보였다. 특히 SiamAttn은 OTB100과 UAV123에서, SiamR-CNN은 TC128과 TrackingNet에서 매우 높은 AUC를 기록했다.
- **데이터셋별 특성**: OTB100은 성능이 이미 포화 상태(Saturated)에 이르러 변별력이 낮아졌으나, LaSOT나 GOT-10K 같은 대규모 데이터셋에서는 여전히 개선의 여지가 많으며 SOTA 모델 간의 격차가 뚜렷하게 나타났다.
- **속도**: KCF와 STAPLE 같은 전통적인 DCF 방식이 가장 빠르며, 딥러닝 기반 추적기들은 상대적으로 느리지만 GPU 가속을 통해 실시간성을 확보하고 있다.

## 🧠 Insights & Discussion

**강점 및 성과**:
최근의 추적기들은 단순히 네트워크를 깊게 쌓는 것을 넘어, 오프라인 학습(End-to-End learning)과 온라인 적분 모듈을 결합하는 방향으로 발전했다. 특히 DCF의 온라인 적응성과 Siamese의 강력한 특징 추출 능력이 결합되면서 추적 성능이 비약적으로 향상되었다.

**한계 및 비판적 해석**:

1. **상태 추정의 한계**: 두 패러다임 모두 기본적으로 '변위(Translation)' 추정에 집중하고 있어, 정교한 Bounding Box 크기 추정(Scale Estimation)에는 여전히 어려움이 있다. 최근 RPN이나 Anchor-free 방식이 도입되었으나, 이는 추적 자체보다는 객체 검출(Detection) 기술의 전이에 가깝다.
2. **온라인 업데이트의 딜레마**: 모델을 너무 자주 업데이트하면 모델 표류(Model Drift)가 발생하고, 업데이트하지 않으면 외형 변화에 대응하지 못하는 트레이드-오프가 존재한다.
3. **계산 비용**: ResNet과 같은 무거운 백본 네트워크의 사용으로 인해 CPU 환경에서의 실시간 추적은 여전히 어려운 과제이다.

**결론 및 제언**:
저자는 향후 연구가 단순한 Bounding Box 추적을 넘어 픽셀 단위의 세그멘테이션(Segmentation)과 통합되어야 한다고 주장한다. 또한, 최근 비전 분야의 혁신인 Transformer 구조가 DCF의 전역적 정보 활용 능력과 유사한 점이 많으므로, 이를 추적 프레임워크에 효과적으로 이식하는 것이 핵심 연구 방향이 될 것으로 보인다.

## 📌 TL;DR

본 논문은 현대 시각적 객체 추적의 양대 산맥인 DCF와 Siamese Network를 총망라한 서베이 보고서이다. DCF의 효율적인 온라인 학습과 SN의 강력한 오프라인 특징 표현의 원리를 분석하고, 9개 벤치마크를 통해 PrDiMP(DCF)와 SiamR-CNN/SiamAttn(SN) 등이 현재 최고 수준의 성능을 내고 있음을 입증했다. 연구의 핵심은 "오프라인 학습을 통한 일반화 능력"과 "온라인 업데이트를 통한 적응성"의 조화에 있으며, 향후에는 Transformer의 도입과 세그멘테이션 기반의 정밀 추적으로 진화할 가능성이 높다.
