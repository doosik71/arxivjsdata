# Video Instance Segmentation by Instance Flow Assembly

Xiang Li, Jinglu Wang, Xiao Li, and Yan Lu (2021)

## 🧩 Problem to Solve

본 논문은 비디오 인스턴스 세그멘테이션(Video Instance Segmentation, VIS)에서 발생하는 객체 추적의 불안정성 문제를 해결하고자 한다. VIS는 비디오 내 특정 클래스의 모든 객체 인스턴스를 분류하고 세그멘테이션하며, 프레임 간에 동일한 인스턴스를 추적하는 것을 목표로 한다.

기존의 상위-하향식(Top-down) 박스 기반 방식(Box-based methods)은 이미지 영역에서는 뛰어난 성능을 보이지만, 비디오 영역으로 확장했을 때 한계가 명확하다. 이러한 방법들은 주로 검출된 바운딩 박스(Bounding Box) 내부의 특징이나 이미지를 크롭(Crop)하여 처리하는데, 이 과정에서 픽셀 수준의 시간적 일관성(Temporal Consistency)을 캡처하기 위한 정렬(Alignment)이 부족하다. 결과적으로 유사한 외형을 가진 객체들이 서로 가깝게 위치할 때, 특징 임베딩의 모호함으로 인해 인스턴스 매칭 오류가 빈번하게 발생한다.

또한, 비디오 전체를 입력으로 사용하는 오프라인(Offline) 방식은 높은 성능을 내지만, 계산 비용이 매우 크고 지연 시간(Latency)이 길어 실시간 응용 분야에 적용하기 어렵다. 따라서 본 논문은 이미지 수준의 입력을 처리하면서도 강건한 추적이 가능한 효율적인 온라인(Online) 하위-상향식(Bottom-up) VIS 방법론을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 바운딩 박스 없이 특징을 처리하는 Bottom-up 방식이 프레임 간 공간적 상관관계(Spatial Correlation)를 더 잘 보존한다는 점에 착안하여, 이를 활용한 새로운 추적 전략을 구축하는 것이다.

주요 기여 사항은 다음과 같다.

1. **Bottom-up VIS 프레임워크**: 다중 스케일 시간적 문맥 융합(Temporal Context Fusion, TCF) 모듈을 탑재하여 프레임 간 상관관계를 효과적으로 인코딩하는 모델을 제안한다.
2. **Instance Flow**: 두 프레임 간 인스턴스 대응 관계를 2D 벡터 필드로 표현하는 효율적인 매칭 표현 방식인 '인스턴스 플로우(Instance Flow)'를 정의한다.
3. **병렬 가능 인스턴스 플로우 어셈블리**: 다수의 참조 프레임(Reference Frames)을 활용하여 가림(Occlusion)이나 모션 블러 상황에서도 강건하고 빠르게 매칭을 수행할 수 있는 어셈블리 방법을 제안한다.

## 📎 Related Works

### 이미지 인스턴스 세그멘테이션

기존 연구는 크게 두 가지 패러다임으로 나뉜다.

- **Top-down 방식**: Mask R-CNN과 같이 바운딩 박스를 먼저 검출한 후 내부 마스크를 예측한다. 정교하지만 박스 의존도가 높아 유사 객체 밀집 지역에서 취약하다.
- **Bottom-up 방식**: Panoptic-Deeplab과 같이 객체의 중심점(Center)이나 오프셋(Offset)을 예측하여 픽셀을 그룹화한다. 박스 없이 공간적 문맥을 보존하므로 비디오 작업에 더 적합할 수 있다.

### 비디오 인스턴스 세그멘테이션 (VIS)

- **Online 방식**: 각 프레임을 개별적으로 세그멘테이션한 뒤 규칙에 따라 연결한다. 실용적이지만 장기적인 시간적 상관관계 모델링이 부족하다.
- **Offline 방식**: 비디오 전체를 입력으로 받아 시공간 정보를 직접 모델링한다(예: MaskProp). 성능은 매우 높으나 계산 비용이 과도하다.

본 논문은 이러한 온라인 방식의 효율성과 보텀업 방식의 공간 정보 보존 능력을 결합하여, 기존 온라인 박스 기반 방식(SipMask, MaskTrack R-CNN 등)보다 우수한 성능을 달성하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

본 시스템은 타겟 프레임 $T$와 이전의 참조 프레임 집합 $R = \{R_i\}$를 입력으로 받는다. 전체 구조는 공유 백본(Shared Backbone), 인트라-프레임(Intra-frame) 예측 모듈, 인터-프레임(Inter-frame) 예측 모듈, 그리고 인스턴스 어셈블리(Instance Assembly) 단계로 구성된다.

### 주요 구성 요소 및 역할

#### 1. Intra-frame Prediction (단일 프레임 예측)

Deeplab V3+ 구조를 기반으로 하며, 두 개의 디코더를 통해 다음 세 가지 맵을 생성한다.

- **Semantic Segmentation map ($S$):** 각 픽셀의 클래스 분류.
- **Instance Center map ($C$):** 객체 중심점의 히트맵.
- **Intra-frame Offset map ($O_t$):** 각 픽셀 $(i, j)$에서 해당 객체 중심점 $c_{ij}^t$까지의 거리 벡터.
  $$O_t = \{(i, j) - c_{ij}^t\}$$

#### 2. Inter-frame Prediction (프레임 간 예측)

참조 프레임의 중심점으로 향하는 인터-프레임 오프셋 $O_r$을 예측한다.

- **Temporal Context Fusion (TCF) 모듈**: 타겟 프레임과 참조 프레임의 고수준 특징 맵을 융합하기 위해 도트 프로덕트 어텐션(Dot-product Attention) 스킴을 사용한다. 이 모듈은 인코더와 디코더 사이의 스킵 연결(Skip Connection) 중간에 피라미드 형태로 삽입되어 시간적 문맥을 보강한다.

#### 3. Loss Functions (손실 함수)

전체 손실 함수 $L_{total}$은 다음과 같이 다섯 가지 성분의 합으로 정의된다.
$$L_{total} = L_{sem} + \lambda_{cent} L_{cent} + \lambda_{inter} L_{inter} + \lambda_{intra} L_{intra} + \lambda_{shape} L_{shape}$$
여기서 $L_{sem}$은 교차 엔트로피 손실, $L_{cent}$는 평균 제곱 오차(MSE) 손실, $L_{intra}$와 $L_{inter}$는 $L_1$ 손실을 사용한다.

특히 **Shape Consistency Loss ($L_{shape}$)**는 인트라-프레임과 인터-프레임 오프셋의 형태적 일관성을 유지하기 위해 도입되었다.
$$L_{shape} = \| (\tilde{O}_r - \tilde{O}_t) - (O_r - O_t) \|_2^2$$
($\tilde{O}$는 정답 값, $O$는 예측 값)

### Instance Flow Assembly (인스턴스 플로우 어셈블리)

#### 인스턴스 플로우 (Instance Flow) 정의

인스턴스 플로우 $f_m$은 객체 중심점 간의 이동을 나타내는 벡터로, 인터-프레임 오프셋과 인트라-프레임 오프셋의 차이(Offset Residual)를 객체 영역 $\Omega_m$에 대해 평균 내어 계산한다.
$$f_m = \frac{\iint_{\Omega_m} (O_r^m - O_t^m) di dj}{\iint_{\Omega_m} di dj}$$

#### 추적 및 매칭 절차

1. **Intra-frame Grouping**: $C, O_t, S$를 이용해 타겟 프레임의 픽셀들을 인스턴스 ID로 그룹화한다.
2. **Instance Matching**: 타겟 프레임의 중심점 $c_t^m$에 인스턴스 플로우 $f_m$을 더해 워핑된 중심점 $\bar{c}_t^m = c_t^m + f_m$을 구한다. 이후 참조 프레임의 중심점 $c_r^n$과의 유클리드 거리를 측정하여 가장 가까운 인스턴스를 매칭한다.
   $$d_{m,n} = \| \bar{c}_t^m - c_r^n \|_2^2$$
3. **Identity Propagation**: 매칭된 참조 인스턴스로부터 ID를 계승하며, 임계값 $\epsilon$보다 거리가 먼 경우 새로운 인스턴스로 인식하여 새 ID를 부여한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Youtube-VIS validation set.
- **지표**: mAP (mean Average Precision).
- **비교 대상**: Box-based 방법(SipMask, MaskTrack R-CNN 등) 및 Box-free 방법(FEELVOS, STEm-Seg 등).
- **환경**: ResNet-50 백본 사용, NVIDIA V100 GPU.

### 정량적 결과

본 논문에서 제안한 방법은 Youtube-VIS 검증 세트에서 **34.1 mAP**를 달성하였다.

- **온라인 방법론 중 최고 성능**: 이미지 수준 입력을 사용하는 온라인 방법들 중에서 SOTA(State-of-the-art) 성능을 보였다.
- **Box-based 대비**: MaskTrack R-CNN 대비 3.8%p, SipMask 대비 0.4%p 높은 mAP를 기록했다.
- **Box-free 대비**: 기존 보텀업 방식들보다 큰 폭(약 3.5%p)으로 성능을 상회했다.
- **속도**: 병렬 최적화를 통해 4개의 참조 프레임을 사용할 때 **37 FPS**의 실시간 속도를 구현했다.

### 정성적 및 분석적 결과

- **객체 크기별 성능**: 중간 및 대형 객체에서는 매우 높은 정밀도(AP)와 재현율(AR)을 보였으나, 소형 객체에서는 성능이 다소 저하되었다. 이는 소형 객체의 중심점 예측이 어렵고, 플로우 계산 시 평균을 낼 픽셀 수가 적어 오차가 발생하기 때문으로 분석된다.
- **시각적 분석**: 유사한 외형의 객체가 밀집해 있는 상황에서 기존 박스 기반 방식(SipMask 등)은 ID 매칭 오류가 잦았으나, 제안 방법은 공간적 상관관계를 보존하는 Instance Flow 덕분에 안정적으로 추적함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 효과

- **TCF 모듈의 효용성**: Ablation study를 통해 TCF 모듈이 단순 컨볼루션이나 Cascade ASPP보다 mAP를 크게 향상(최대 3.6%p)시킴을 입증하였다. 이는 시간적 문맥 융합이 인터-프레임 대응 관계 학습에 필수적임을 시사한다.
- **다중 참조 프레임의 중요성**: 단일 프레임 매칭(26.2 mAP)보다 4개의 참조 프레임을 사용할 때(34.1 mAP) 성능이 비약적으로 상승했다. 특히 첫 번째 프레임을 참조에 포함함으로써 에러 전파(Error Propagation) 문제를 완화했다.

### 한계 및 비판적 해석

- **소형 객체 추적의 취약성**: 본 논문의 핵심인 '영역 내 오프셋 평균(Average)' 방식은 픽셀 수가 적은 소형 객체에서 노이즈에 취약하다는 근본적인 한계가 있다. 향후 연구에서 가중 평균이나 다른 어그리게이션 기법이 필요할 것으로 보인다.
- **배경 감독(Background Supervision)의 역설**: 실험 결과, 배경 영역에 손실 함수를 적용하여 0으로 강제하는 것보다, 오히려 감독하지 않았을 때 성능이 더 좋았다. 이는 네트워크가 배경을 억제하는 데 파라미터를 낭비하지 않고 전경의 엣지나 중심점 예측에 더 집중할 수 있게 하기 때문으로 해석된다.

## 📌 TL;DR

본 논문은 바운딩 박스 없이 객체 중심점을 추적하는 **Bottom-up 방식의 비디오 인스턴스 세그멘테이션(VIS)** 프레임워크를 제안한다. 핵심 기여는 타겟-참조 프레임 간의 오프셋 잔차(Offset Residual)를 이용해 정의한 **Instance Flow**와 이를 활용한 **병렬 매칭 전략**이다. 이를 통해 유사 객체가 밀집한 상황에서도 강건한 추적이 가능해졌으며, Youtube-VIS 데이터셋에서 34.1 mAP라는 온라인 방식 기준 최상위 성능과 37 FPS의 실시간성을 동시에 확보하였다. 이 연구는 연산 효율성이 중요한 실시간 비디오 분석 시스템 및 자율주행 분야의 객체 추적 기술에 중요한 기초가 될 수 있다.
