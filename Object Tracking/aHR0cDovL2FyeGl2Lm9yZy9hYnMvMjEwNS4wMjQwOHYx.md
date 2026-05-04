# Spatio-Temporal Matching for Siamese Visual Tracking

Jinpu Zhang, Yuehuan Wang (2021)

## 🧩 Problem to Solve

본 논문은 Siamese 기반 객체 추적기(Siamese Tracker)에서 핵심 연산인 유사도 매칭(Similarity Matching)의 한계를 해결하고자 한다. 기존의 대부분의 Siamese 추적기는 이미지 매칭 분야에서 유래한 교차 상관(Cross Correlation, XCorr) 방식을 사용하여 유사도를 학습한다. 그러나 객체 추적 작업에서는 단순한 2D 이미지 매칭을 넘어 높이(height), 너비(width), 채널(channel), 그리고 시간(time)이라는 4차원 정보의 통합적인 고려가 필요하다.

기존의 교차 상관 방식은 다음과 같은 두 가지 주요 문제를 가진다. 첫째, 채널 및 시간 차원의 정보를 무시함으로써 모호한 매칭 결과를 생성한다. 특히 심층 특징(deep features)은 희소하게 활성화(sparsely activated)되는 경향이 있어, 타겟을 설명하는 일부 채널 외에 많은 수의 불필요한 채널이 매칭 과정에 포함되어 방해 요소(distractor)와 타겟을 구분하기 어렵게 만든다. 둘째, 기존 방식은 각 프레임을 독립적인 매칭 문제로 처리하기 때문에 폐색(occlusion), 급격한 변형, 또는 배경 간섭이 발생했을 때 추적에 실패할 가능성이 높다.

따라서 본 논문의 목표는 공간(높이, 너비, 채널)과 시간 차원을 모두 활용하는 시공간 매칭(Spatio-Temporal Matching) 프로세스를 제안하여, 보다 강건하고 정확한 객체 추적 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 공간적 매칭에서는 채널 간의 관계를 동적으로 재조정하고, 시간적 매칭에서는 프레임 간의 응답 맵(response map)의 일관성을 유지하는 것이다.

1.  **Space-Variant Channel-guided Correlation (SVC-Corr)**: 기존의 DW-XCorr가 모든 채널에 동일한 가중치를 부여하는 것과 달리, 각 공간 위치에서 템플릿과 검색 영역 간의 채널 연관성을 독립적으로 계산하여 타겟에 특화된 응답을 강조하고 간섭 응답을 약화시킨다.
2.  **Aberrance Repressed Module (ARM)**: 인접 프레임 간의 응답 맵에 존재하는 시간적 문맥 관계를 조사하여, 응답 맵의 갑작스러운 변화(aberrance)를 억제한다. 이를 통해 추적 중 발생할 수 있는 드리프트(drifting) 현상을 방지한다.
3.  **SiamSTM 프레임워크**: SVC-Corr와 ARM을 통합한 새로운 Anchor-free 추적 프레임워크를 제시하여, 연산 효율성과 추적 정확도를 동시에 확보하였다.

## 📎 Related Works

Siamese 기반 추적기들은 템플릿과 검색 영역 간의 일반적인 유사도 메트릭을 학습하는 방식으로 발전해 왔다. 초기 SiamFC는 XCorr를 사용하였고, SiamRPN 및 SiamRPN++는 각각 UP-XCorr와 DW-XCorr를 도입하여 효율적인 정보 결합을 시도하였다. 최근에는 Anchor-free 모델(SiamFC++, SiamCAR, SiamBAN, Ocean 등)들이 픽셀 단위 예측 방식을 통해 하이퍼파라미터 의존도를 낮추는 방향으로 연구되고 있다.

또한, 픽셀 단위 상관관계(pixel-wise correlation)를 이용한 PG-Net 등의 연구가 진행되었으나, 이들 역시 채널 간의 연관성(channel association)을 충분히 고려하지 못한다는 한계가 있다. 시간적 정보를 활용하기 위해 기존에는 템플릿을 선형적으로 업데이트하거나 온라인 분류기(online classifier)를 학습시키는 방법이 사용되었으나, 전자는 변화하는 시나리오에 대응하기에 불충분하고 후자는 반복적인 최적화로 인해 연산 시간이 많이 소요된다는 단점이 있다. 본 논문은 이러한 한계를 극복하기 위해 매칭 개념을 시간 도메인으로 확장하여 효율성과 강건성을 동시에 달성하고자 한다.

## 🛠️ Methodology

### 1. Basic Siamese Tracker
SiamSTM은 CenterNet에서 영감을 받은 Anchor-free 구조를 기반으로 한다.
- **특징 추출**: ResNet50의 `conv1`부터 `conv4`까지를 사용하여 연산량을 줄였다.
- **타겟 로컬라이제이션**: 객체를 중심점(center), 크기(size), 로컬 오프셋(local offset)으로 표현한다.
- **손실 함수**: 중심점 히트맵 $\hat{Y}$는 Gaussian 라벨 $Y$를 사용하여 Penalty-reduced pixel-wise logistic regression(Focal loss)으로 학습하며, 오프셋과 크기는 $L_1$ loss를 사용한다. 전체 기본 손실 함수는 다음과 같다.
$$L_{base} = L_{cls} + \lambda_{off} L_{off} + \lambda_{size} L_{size}$$

### 2. Space-variant Channel-guided Correlation (SVC-Corr)
SVC-Corr는 2D 상관관계에 채널 간의 대응 관계를 추가하여 타겟 인식 능력을 높인다.
- **구조**: DW-XCorr 브랜치와 Channel Transform(Ch Trans) 브랜치의 두 경로로 구성된다.
- **Channel Transform**: 템플릿 $z$와 검색 영역 $x$에서 Max pooling과 Average pooling을 통해 채널 기술자(channel descriptors)를 추출하고, 공유된 FC 레이어를 통해 채널 간 상호의존성을 캡처한다.
$$\text{T}_{ch}(\omega) = \text{FC}(f_{max}^{k}(\phi_1(\omega))) \oplus \text{FC}(f_{avg}^{k}(\phi_1(\omega)))$$
- **가중치 생성 및 적용**: 생성된 기술자들을 결합하고 $1 \times 1$ 컨볼루션 $\phi_2$를 통해 공간 가변적 채널 가중치 $f_{ca}$를 생성한다. 최종 결과는 DW-XCorr의 공간 응답 $f_{sa}$에 이 가중치를 더하여 산출한다.
$$f_{ca}(z, x) = \phi_2(\text{T}_{ch}(z) \oplus \text{T}_{ch}(x))$$
$$\text{svc\_corr}(z, x) = f_{ca}(z, x) \oplus f_{sa}(z, x)$$

### 3. Aberrance Repressed Module (ARM)
ARM은 인접 프레임 간의 중심 히트맵을 입력으로 받아 시간적 제약 조건을 부여한다.
- **핵심 아이디어**: 이상적인 히트맵은 단일 피크(unimodal)를 가져야 하며, 정렬 후 인접 프레임 간의 분포가 최대한 유사해야 한다는 점을 이용한다.
- **손실 함수**: KL Divergence를 사용하여 정렬된 히트맵 간의 유사성을 최대화하고 가우시안 라벨과의 오차를 최소화한다.
$$L_{arm} = \text{KL}(Y_{i+k} \otimes Y_{i+k}, \hat{Y}_i[\Delta_{p,q}] \otimes \hat{Y}_{i+k}) + \text{KL}(Y_i \otimes Y_i, \hat{Y}_{i+k}[\Delta_{q,p}] \otimes \hat{Y}_i)$$
여기서 $\Delta_{p,q}$는 피크 $p$와 $q$를 일치시키기 위한 circular shifting 연산이다.
- **추론 절차**: 추론 단계에서는 이전 프레임의 히트맵 $\hat{Y}_{last}$와 현재 프레임의 후보 피크 위치들 중 $L_{arm}^k$를 최소화하는 최적의 위치 $\hat{q}$를 찾는다. 만약 현재의 최대 응답 위치와 $\hat{q}$가 다르다면, 이전 프레임의 정보를 가중치로 더해 현재 히트맵을 보정한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: OTB100, VOT2018, VOT2020, GOT-10k, LaSOT 등 5개 벤치마크 사용.
- **학습**: ImageNet VID, DET, Youtube-BB, COCO, GOT-10k 데이터셋으로 학습.
- **성능 지표**: EAO (Expected Average Overlap), AO (Average Overlap), Success Rate (SR), Precision 등을 사용.

### 2. 정량적 결과
- **VOT2018**: 모든 방법론 중 가장 높은 EAO와 Robustness(R)를 기록하였다. 특히 폐색(occlusion)과 카메라 모션(camera motion) 시나리오에서 압도적인 성능 향상을 보였다.
- **VOT2020**: Bounding box 출력 기반 추적기 중 최고의 성능(EAO, A, R)을 달성하였다.
- **GOT-10k**: AO 0.624를 기록하며 기존 SOTA 모델인 DiMP와 Ocean을 2.1포인트 상회하였다.
- **OTB100 & LaSOT**: OTB100에서 Precision 0.922, AUC 0.707로 최고 성능을 보였으며, LaSOT의 장기 추적(long-term tracking)에서도 AUC 기준 1위를 차지하여 드리프트 방지 능력을 입증하였다.
- **속도**: 66 FPS의 빠른 추적 속도를 유지하였다.

### 3. Ablation Study
- **Head 구조**: FCOS-based 구조보다 Center-based 구조를 사용했을 때 AO가 1.7포인트 향상되었다. 이는 Gaussian 라벨 할당 방식이 분류와 로컬라이제이션 품질을 더 잘 통합하기 때문으로 분석된다.
- **SVC-Corr vs. DW-XCorr**: SVC-Corr 도입 시 AO가 1.6포인트 증가하였다. 이는 채널 응답의 재교정(recalibration)을 통해 타겟 인식 능력이 향상되었음을 의미한다.
- **ARM의 효과**: ARM은 DW-XCorr 기반 모델에서 2.3포인트, SVC-Corr 기반 모델에서 1.8포인트의 AO 이득을 가져왔으며, 시간적 제약 조건을 통해 강건성을 높였다.

## 🧠 Insights & Discussion

본 논문은 Siamese 추적기가 직면한 고질적인 문제인 '채널 중복성'과 '시간적 문맥 결여'를 정교하게 해결하였다. 

**강점**으로는 우선 SVC-Corr가 단순한 전역 채널 어텐션과 달리 각 공간 위치마다 독립적인 채널 연관성을 학습함으로써 인스턴스 수준의 구별력을 확보했다는 점을 들 수 있다. 또한 ARM은 복잡한 온라인 최적화 과정 없이 KL Divergence 기반의 제약 조건만으로 시간적 일관성을 유지하여, 연산 효율성을 해치지 않으면서도 온라인 적응형 모델(online adaptation)에 근접하는 강건성을 확보하였다.

**한계 및 논의사항**으로는, 현재 프레임워크가 Bounding box 예측에 집중되어 있어 세그멘테이션 마스크(mask) 예측 능력이 부족하다는 점이 언급되었다. 저자들은 향후 연구에서 마스크 예측 기능을 end-to-end로 통합하여 VOT2020 및 VOS 작업으로 확장할 계획임을 밝혔다. 또한 ARM의 성능 향상이 단순히 triplet 입력 데이터 때문이 아니라 시간적 제약 조건 자체에서 기인한다는 점을 명시함으로써 제안 방법론의 타당성을 뒷받침하였다.

## 📌 TL;DR

본 논문은 기존 Siamese 추적기가 무시했던 채널과 시간 차원의 정보를 활용하는 **SiamSTM**을 제안한다. 공간적으로는 **SVC-Corr**를 통해 타겟 특화 채널 응답을 생성하고, 시간적으로는 **ARM**을 통해 프레임 간 응답 맵의 급격한 변화를 억제하여 드리프트를 방지한다. 이 연구는 5개 주요 벤치마크에서 SOTA 성능을 달성하였으며, 특히 66 FPS라는 높은 속도를 유지하면서도 강건한 추적이 가능함을 보였다. 이는 향후 고속-고정밀 실시간 객체 추적 시스템 구현에 중요한 역할을 할 것으로 기대된다.