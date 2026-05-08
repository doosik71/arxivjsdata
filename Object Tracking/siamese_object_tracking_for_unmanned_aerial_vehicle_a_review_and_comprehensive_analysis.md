# Siamese Object Tracking for Unmanned Aerial Vehicle: A Review and Comprehensive Analysis

Changhong Fu, Kunhan Lu, Guangze Zheng, Junjie Ye, Ziang Cao, Bowen Li, and Geng Lu (2022)

## 🧩 Problem to Solve

본 논문은 무인 항공기(Unmanned Aerial Vehicle, UAV) 기반의 시각적 객체 추적(Visual Object Tracking)에서 Siamese 네트워크의 적용 가능성과 한계를 분석하는 것을 목표로 한다. UAV 기반 추적은 지능형 교통 시스템(Intelligent Transportation Systems) 등 다양한 분야에서 활용되지만, 실제 환경에서는 다음과 같은 심각한 문제들에 직면한다.

1. **항공 추적 특유의 도전 과제**: 저해상도(Low Resolution, LR), 가림 현상(Occlusion, OCC), 조명 변화(Illumination Variation, IV), 시점 변화(Viewpoint Change, VC), 그리고 빠른 움직임(Fast Motion, FM) 등이 발생한다.
2. **제한된 컴퓨팅 자원**: UAV에 탑재 가능한 프로세서는 서버급 GPU에 비해 연산 능력이 매우 낮아, 고성능 딥러닝 모델을 실시간(Real-time)으로 구동하는 데 어려움이 있다.
3. **열악한 환경 조건**: 특히 야간이나 저조도(Low-illumination) 환경에서는 전경과 배경의 구분이 어려워 추적 성능이 급격히 저하된다.

따라서 본 연구의 목표는 최신 Siamese 추적기들을 종합적으로 리뷰하고, 실제 UAV 온보드 프로세서(NVIDIA Jetson AGX Xavier)에서의 정량적 성능 평가를 통해 실용적인 배포 가능성을 검증하며, 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여 사항은 다음과 같다.

- **종합적인 리뷰**: 최신 SOTA(State-of-the-art) Siamese 추적기들을 체계적으로 분류하고, UAV 관점에서의 적용 가능성을 분석하였다.
- **통합 코드 라이브러리 제공**: 분산되어 있던 다양한 Siamese 추적기들을 하나의 라이브러리로 통합하여 연구 커뮤니티의 편의성을 높였다.
- **임베디드 환경에서의 정량적 평가**: 전형적인 UAV 온보드 프로세서인 NVIDIA Jetson AGX Xavier를 사용하여 6개의 권위 있는 UAV 벤치마크 데이터셋에서 성능과 속도를 측정하였다.
- **실제 온보드 테스트 수행**: 시뮬레이션이나 데이터셋 평가를 넘어, 실제 UAV에 모델을 탑재하여 실시간성 및 강건성을 검증하였다.
- **저조도 환경 분석 및 향후 방향 제시**: 저조도 데이터셋 평가를 통해 기존 모델의 한계를 명시하고, Transformer 도입 및 적대적 강건성 등 미래 연구 방향을 구체적으로 논의하였다.

## 📎 Related Works

기존의 시각적 객체 추적 연구는 크게 두 가지 방향으로 나뉜다.

1. **DCF(Discriminative Correlation Filter) 기반 방법**: 연산 효율이 매우 높아 CPU만으로도 실시간 추적이 가능하여 초기 UAV 추적에 많이 사용되었다. 그러나 수작업으로 설계된 특징(Handcrafted features)을 사용하여 동적인 복잡한 환경에서 강건성과 일반화 능력이 떨어진다는 한계가 있다.
2. **CNN 및 Siamese 네트워크 기반 방법**: 딥러닝을 통해 특징 추출 능력을 극대화하여 정확도와 강건성을 확보하였다. 특히 Siamese 네트워크는 템플릿과 검색 영역 간의 유사도를 계산하는 단순하고 효율적인 구조 덕분에 최근 각광받고 있다.

기존의 Siamese 네트워크 리뷰 논문들은 주로 일반적인 추적 성능에 집중했으며, UAV의 특수한 제약 사항(제한된 전력 및 연산 자원, 항공 뷰의 특성)과 실제 임베디드 하드웨어에서의 구동 속도를 심층적으로 분석한 연구는 부족했다는 점이 본 논문의 차별점이다.

## 🛠️ Methodology

본 논문은 특정 새로운 알고리즘을 제안하는 대신, 기존 Siamese 추적기들의 구조를 체계화하여 분석한다. Siamese 추적기의 일반적인 파이프라인은 **특징 모델링(Feature Modeling)**과 **타겟 로컬라이제이션(Target Localization)**의 두 단계로 구성된다.

### 1. 특징 모델링 (Feature Modeling)

- **특징 추출(Feature Extraction)**: 파라미터를 공유하는 두 개의 백본(Backbone) 네트워크(예: AlexNet, ResNet)를 통해 템플릿 이미지 $z$와 검색 영역 이미지 $x$로부터 특징 맵 $\phi(z), \phi(x)$를 생성한다.
- **특징 정제(Feature Refinement)**: 추출된 특징 맵에 Attention 메커니즘, 그래프 신경망(GCN), 또는 시간적 정보(Temporal information)를 결합하여 타겟의 변별력을 높이는 과정이다.

### 2. 타겟 로컬라이제이션 (Target Localization)

특징 맵들이 생성되면 **교차 상관(Cross-correlation)** 연산을 통해 유사도 맵(Similarity map)을 생성한다.
$$f(z, x) = \phi(z) \otimes \phi(x) + b$$
여기서 $\otimes$는 교차 상관 연산을 의미하며, 이 결과물을 바탕으로 타겟의 바운딩 박스를 예측한다. 로컬라이제이션 방식은 크게 두 가지로 나뉜다.

- **Anchor-based 방법**: 미리 정의된 다양한 크기와 비율의 앵커(Anchor) 박스를 설정하고, 각 앵커가 타겟인지 분류(Classification)한 후 위치를 미세 조정(Regression)한다. (예: SiamRPN, SiamMask)
- **Anchor-free 방법**: 앵커 없이 각 픽셀의 전경/배경 확률을 직접 계산하고, 타겟 중심으로부터의 거리나 오프셋을 직접 회귀한다. (예: SiamBAN, Ocean, SiamAPN)

### 3. 주요 모델 분석

- **SiamAPN / SiamAPN++**: UAV의 고속 추적 요구사항을 충족하기 위해 앵커 제안 네트워크(Anchor Proposal Network)를 도입하여, 고정된 앵커 대신 적응형 앵커를 생성함으로써 효율성과 정확도의 균형을 맞추었다.
- **SiamRPN++**: 더 깊은 백본(ResNet)을 사용하면서도 Translation Invariance를 유지하기 위해 공간 인식 샘플링 전략을 사용하였다.

## 📊 Results

### 실험 설정

- **평가 지표**: 성공률(Success rate, AUC), 정밀도(Precision, CLE=20px), 정규화된 정밀도(Normalized Precision).
- **벤치마크**: UAV123@10fps, UAV20L, DTB70, UAVDT, VisDrone-SOT2020-test, UAVTrack112.
- **하드웨어**: NVIDIA Jetson AGX Xavier.

### 주요 결과

1. **속도와 정확도의 트레이드-오프**: ResNet-50 기반의 모델들(SiamRPN++R, SiamBAN 등)은 정확도는 높으나 처리 속도가 10 FPS 미만으로 매우 느려 UAV 실시간 배포에 부적합하였다.
2. **실시간 최적 모델**: **SiamAPN**과 **SiamAPN++**는 30 FPS 이상의 속도를 유지하면서도 매우 높은 정확도와 성공률을 기록하여, UAV 온보드 환경에 가장 적합한 모델로 평가되었다.
3. **특성별 분석**:
    - **빠른 움직임(FM)** 및 **조명 변화(IV)** 상황에서는 SiamAPN 계열이 강점을 보였다.
    - **저해상도(LR)** 상황에서는 모든 모델의 성능이 전반적으로 하락하며, 여전히 해결해야 할 난제로 나타났다.
4. **온보드 테스트**: 실제 UAV에 SiamAPN과 SiamAPN++를 탑재하여 테스트한 결과, 가림 현상(OCC) 발생 시 일시적으로 오차가 증가하지만, 가림이 해제된 후 빠르게 타겟을 재포착하며 30 FPS 이상의 실시간성을 유지함을 확인하였다.
5. **저조도 평가**: 야간 데이터셋(UAVDark135 등)에서는 모든 모델의 성능이 주간 대비 급격히 하락하였다. 이는 전경-배경 구분이 어렵고 센서 노이즈가 심하기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 단순한 알고리즘 비교를 넘어, 실제 UAV 하드웨어 제약 조건을 반영한 정량적 분석을 수행했다는 점에서 가치가 높다. 특히, 이론적 성능과 실제 온보드 성능의 괴리를 명확히 짚어냈으며, 실용적인 관점에서 SiamAPN 계열의 효율성을 입증하였다.

### 한계 및 비판적 해석

- **저조도 문제의 심각성**: 현재의 Siamese 네트워크는 기본적으로 주간 이미지로 학습된 백본을 사용하므로, 도메인 차이(Domain Gap)가 큰 야간 환경에서는 구조적인 한계가 있다. 단순히 모델을 깊게 쌓는 것보다 도메인 적응(Domain Adaptation) 기법이 필수적이다.
- **연산 효율성**: 여전히 많은 모델이 GPU 의존적이며, CPU 기반의 초경량 모델에 대한 분석이 부족하다.

### 향후 연구 방향

1. **Transformer 도입**: 전역 정보 추출 능력이 뛰어난 Transformer를 Siamese 구조와 결합하여 VC나 OCC 상황에서의 강건성을 높여야 한다.
2. **적대적 강건성(Adversarial Robustness)**: UAV는 외부 공격(Perturbation)에 취약할 수 있으므로, 안전 필수 시스템으로서의 보안성 연구가 필요하다.
3. **예측 기반 추적(Predictive Tracking)**: 온보드 연산 지연(Latency)으로 인한 타겟 위치 불일치를 해결하기 위해, 칼만 필터나 딥러닝 기반의 상태 예측기를 통합하는 PVT(Predictive Visual Tracking) 접근법이 유망하다.

## 📌 TL;DR

본 논문은 UAV 기반 실시간 객체 추적을 위한 Siamese 네트워크들을 종합 분석하고, NVIDIA Jetson AGX Xavier 하드웨어에서 실증 평가를 수행하였다. 분석 결과, **SiamAPN**과 **SiamAPN++**가 속도(30+ FPS)와 정확도 측면에서 가장 우수한 균형을 보여 UAV 배포에 최적임을 확인하였다. 다만, 저조도 환경과 저해상도 이미지에 대한 취약점이 여전히 존재하며, 이를 해결하기 위해 Transformer 도입, 도메인 적응, 예측 기반 추적 기법 등의 연구가 필요함을 제시하였다.
