# Track to Detect and Segment: An Online Multi-Object Tracker

Jialian Wu, Jiale Cao, Liangchen Song, Yu Wang, Ming Yang, Junsong Yuan (2021)

## 🧩 Problem to Solve

본 논문은 온라인 다중 객체 추적(Online Multi-Object Tracking, MOT) 분야에서 검출(Detection)과 추적(Tracking) 사이의 유기적 결합 부족 문제를 해결하고자 한다. 기존의 추적 패러다임은 크게 두 가지로 나뉜다. 첫째, Tracking-by-Detection(TBD) 방식은 검출기와 데이터 연관(Data Association) 모듈을 완전히 분리하여 처리하므로 연산 효율이 낮고 엔드-투-엔드(end-to-end) 최적화가 불가능하다. 둘째, 최근의 Joint Detection and Tracking(JDT) 방식은 단일 네트워크에서 두 작업을 동시에 수행하지만, 여전히 검출 단계가 추적 정보의 도움 없이 독립적으로 수행된다는 한계가 있다.

저자들은 부분 폐색(Partial Occlusion)이나 모션 블러(Motion Blur)와 같은 까다로운 상황에서 추적 단서(Tracking Clues)가 검출 성능을 향상시킬 수 있으며, 반대로 안정적인 검출이 일관된 트랙렛(Tracklet) 생성의 기초가 된다고 주장한다. 또한, 일반적인 Re-ID 손실 함수가 클래스 내 분산(Intra-class variance)에만 집중하여 검출 손실 함수와 상충됨으로써 오히려 검출 성능을 저하시킬 수 있다는 문제를 지적한다. 따라서 본 논문의 목표는 추적 정보를 활용해 검출 및 세그멘테이션을 보완하고, 검출 성능을 해치지 않는 새로운 Re-ID 학습 체계를 통합한 엔드-투-엔드 모델인 TraDeS를 제안하는 것이다.

## ✨ Key Contributions

TraDeS의 핵심 아이디어는 추적 과정에서 얻은 모션 단서를 검출 단계로 피드백하여, 현재 프레임에서 소실될 가능성이 있는 객체의 특징을 이전 프레임으로부터 전파(Propagate)하여 복구하는 것이다. 이를 위해 다음과 같은 핵심 설계를 도입하였다.

1. **Cost Volume based Association (CVA)**: 4차원 Cost Volume을 통해 두 프레임 간의 조밀한 매칭 유사도를 계산하고, 이를 바탕으로 Re-ID 임베딩 학습과 객체의 추적 오프셋(Tracking Offset) 추론을 동시에 수행한다.
2. **Motion-guided Feature Warper (MFW)**: CVA에서 추론된 추적 오프셋을 모션 단서로 사용하여, 이전 프레임의 객체 특징을 현재 프레임으로 워핑(Warping)하여 전파한다. 이를 통해 현재 프레임에서 흐릿하거나 가려진 객체의 특징을 보강하여 검출 성능을 높인다.
3. **검출-추적 호환적 Re-ID 학습**: Cost Volume에 직접 감독 신호를 주는 방식을 통해 클래스 내 분산뿐만 아니라 클래스 간 차이(Inter-class difference)를 동시에 학습함으로써, 검출 성능을 저하시키지 않으면서 효과적인 임베딩을 학습한다.

## 📎 Related Works

본 논문은 기존 연구들을 다음과 같이 분석하고 차별점을 제시한다.

- **Tracking-by-Detection (TBD)**: 오프더쉘(off-the-shelf) 검출기와 별도의 Re-ID 모델을 사용하며, 데이터 연관을 위해 칼만 필터나 그래프 최적화를 사용한다. 하지만 단계별 처리로 인해 계산 비용이 매우 높다.
- **Joint Detection and Tracking (JDT)**: 단일 포워드 패스로 검출과 추적을 동시에 수행하여 효율적이지만, 대부분의 JDT 모델은 검출 단계에서 추적 정보를 활용하지 않는다.
- **Tracking-guided Video Object Detection**: 추적 결과를 이용해 검출 점수를 재가중치(Reweight)하는 방식이 존재하나, 이는 후처리 단계에서만 이루어지며 수동 튜닝이 필요하다는 단점이 있다. 반면 TraDeS는 검출 자체가 추적 결과에 조건부로 학습되는 엔드-투-엔드 구조이다.
- **Cost Volume**: 주로 깊이 추정(Depth Estimation)이나 광학 흐름(Optical Flow) 추정에서 픽셀 간 연관성을 찾기 위해 사용되었으며, TraDeS는 이를 MOT의 임베딩 학습과 오프셋 추론으로 확장하여 적용하였다.

## 🛠️ Methodology

TraDeS는 포인트 기반 검출기인 CenterNet을 기반으로 구축되었으며, 전체 파이프라인은 CVA 모듈과 MFW 모듈의 유기적 결합으로 구성된다.

### 1. Cost Volume based Association (CVA)

CVA 모듈은 두 프레임 $I_t$와 $I_{t-\tau}$에서 추출된 임베딩 특징 $e_t, e_{t-\tau} \in \mathbb{R}^{H_C \times W_C \times 128}$를 사용하여 4차원 Cost Volume $C$를 생성한다.

**Cost Volume 생성**
Cost Volume의 각 원소 $C_{i,j,k,l}$은 두 프레임의 지점 간 유사도를 나타내며 다음과 같이 행렬 곱으로 계산된다.
$$C_{i,j,k,l} = e'_{t,i,j} \cdot e'_{t-\tau,k,l}$$

**추적 오프셋(Tracking Offset) 추론**
객체 $x$가 현재 프레임의 $(i,j)$ 지점에 있을 때, $C$에서 해당 지점의 2D 맵 $C_{i,j} \in \mathbb{R}^{H_C \times W_C}$를 추출한다. 이후 다음 과정을 거친다.

1. $C_{i,j}$를 가로/세로 방향으로 Max Pooling하고 Softmax를 적용하여, 객체가 이전 프레임의 특정 가로/세로 위치에 존재할 확률분포 $C^W_{i,j}$와 $C^H_{i,j}$를 구한다.
2. 미리 정의된 오프셋 템플릿 $M_{i,j}$ (가로)와 $V_{i,j}$ (세로)와의 내적을 통해 최종 추적 오프셋 $O_{i,j}$를 도출한다.
$$O_{i,j} = [C^H_{i,j} V_{i,j}, C^W_{i,j} M_{i,j}^\top]^\top$$

**학습 목표**
CVA는 임베딩 $e$를 학습시키기 위해 Cost Volume에 대해 Focal Loss 형태의 손실 함수 $L_{CVA}$를 적용한다. 이는 객체가 이전 프레임의 자신과는 가까워지게 하고, 다른 객체나 배경과는 멀어지게 강제함으로써 클래스 간 차이를 명확히 한다.

### 2. Motion-guided Feature Warper (MFW)

MFW는 CVA에서 예측된 추적 오프셋 $O_C$를 사용하여 이전 프레임의 특징 $f_{t-\tau}$를 현재 프레임으로 전파하여 $f_t$를 보강한다.

**특징 전파 (Temporal Propagation)**
Deformable Convolution (DCN)을 사용하여 특징을 워핑한다. 이때, 단순한 특징 맵이 아니라 CenterNet의 클래스 불가지론적 센터 히트맵 $P_{t-\tau}^{agn}$을 곱한 센터 집중 특징 $\bar{f}_{t-\tau}$를 전파한다.
$$\bar{f}_{t-\tau}^q = f_{t-\tau}^q \circ P_{t-\tau}^{agn}$$
이후 $\text{DCN}$을 통해 전파된 특징 $\hat{f}_{t-\tau}$를 얻는다.

**특징 보강 (Feature Enhancement)**
현재 프레임의 특징 $f_t$와 전파된 특징 $\hat{f}_{t-\tau}$를 적응적 가중치 $w$를 이용해 가중 합산하여 최종 강화된 특징 $\tilde{f}_t$를 생성한다.
$$\tilde{f}_t^q = w_t \circ f_t^q + \sum_{\tau=1}^T w_{t-\tau} \circ \hat{f}_{t-\tau}^q$$
이 $\tilde{f}_t$가 최종적으로 검출 및 세그멘테이션 헤드로 입력되어 객체를 탐지한다.

### 3. 데이터 연관 및 전체 손실 함수

데이터 연관은 2단계로 진행된다.

- **1라운드**: $O_C$를 이용해 이전 프레임의 근접 영역에서 가장 가까운 검출물과 매칭한다.
- **2라운드**: 1라운드에서 실패한 경우, 임베딩 $e_t$의 코사인 유사도를 기반으로 히스토리 트랙렛과 매칭하여 장기 추적(Long-term association)을 수행한다.

전체 손실 함수는 다음과 같다.
$$L = L_{CVA} + L_{det} + L_{mask}$$

## 📊 Results

### 실험 설정

- **데이터셋**: MOT16/17 (2D 추적), nuScenes (3D 추적), MOTS (인스턴스 세그멘테이션 추적), YouTube-VIS (인스턴스 세그멘테이션 추적).
- **지표**: MOTA, IDF1, AMOTA, AMOTP, AP 등 각 작업에 맞는 표준 지표 사용.
- **구현**: DLA-34 백본 사용, $\text{T}=2$ (MOT, MOTS) 또는 $\text{T}=1$ (nuScenes, YouTube-VIS) 설정.

### 주요 결과

1. **2D 추적 (MOT16/17)**: TraDeS는 MOT16에서 70.1 MOTA, MOT17에서 69.1 MOTA를 기록하며 기존 JDT 방식인 CenterTrack을 상회하는 성능을 보였다. 특히 FN(False Negative)을 크게 줄여 객체 복구 능력이 뛰어남을 입증했다.
2. **3D 추적 (nuScenes)**: 단안(Monocular) 3D 추적기 중 가장 높은 성능을 기록했다. 특히 nuScenes처럼 프레임 레이트가 낮고 움직임이 큰 데이터셋에서 Baseline 대비 압도적인 성능 향상을 보였는데, 이는 Cost Volume 기반의 오프셋 추정이 일반적인 회귀 방식보다 강건하기 때문이다.
3. **인스턴스 세그멘테이션 추적 (MOTS, YouTube-VIS)**: MOTS에서 sMOTSA 50.8을 기록하여 TrackR-CNN을 능가했으며, YouTube-VIS에서도 Baseline 대비 AP를 6.2나 향상시키며 범용적인 추적기임을 증명했다.

## 🧠 Insights & Discussion

**강점 및 분석**

- **상호 보완적 구조**: 추적 $\rightarrow$ 검출 $\rightarrow$ 추적으로 이어지는 선순환 구조를 구축하였다. 특히 MFW를 통한 특징 전파가 폐색이나 블러 상황에서 검출 누락을 방지하는 핵심 기제로 작용한다.
- **Cost Volume의 효용성**: 단순한 오프셋 예측 대신 Cost Volume을 사용함으로써, 학습 데이터에서 보지 못한 매우 큰 움직임(Large motion)에 대해서도 강건한 추적 오프셋을 생성할 수 있음을 확인하였다.
- **학습 호환성**: 제안한 $L_{CVA}$ 손실 함수가 Re-ID 성능을 유지하면서도 검출 성능을 저하시키지 않는다는 점은, JDT 모델 설계 시 임베딩 학습과 검출 학습 사이의 균형이 중요함을 시사한다.

**한계 및 논의**

- **계산 비용**: 특징 전파를 위해 이전 프레임의 특징을 저장하고 DCN을 수행하므로, 단순 검출기보다는 추론 시간이 증가한다 ($\text{T}=2$일 때 약 57ms).
- **가정**: 본 모델은 CenterNet과 같은 포인트 기반 검출기에 의존하고 있으며, 특징 전파를 위해 이전 프레임의 히트맵 정보가 정확하다는 가정하에 작동한다.

## 📌 TL;DR

TraDeS는 **Cost Volume 기반 연관(CVA)**과 **모션 가이드 특징 워퍼(MFW)**를 통해 추적 단서를 검출 단계에 통합한 엔드-투-엔드 온라인 MOT 모델이다. 추적 정보를 이용해 이전 프레임의 특징을 현재로 전파함으로써 가려지거나 흐릿한 객체를 효과적으로 복구하며, 이는 2D/3D 추적 및 인스턴스 세그멘테이션 추적의 4개 벤치마크에서 SOTA급 성능으로 입증되었다. 특히 저프레임/대동작 환경에서 매우 강건한 성능을 보여 향후 자율주행 및 실시간 비디오 분석 연구에 중요한 기여를 할 것으로 평가된다.
