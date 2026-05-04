# 3D-Aware Instance Segmentation and Tracking in Egocentric Videos

Yash Bhalgat, Vadim Tschernezki, Iro Laina, João F. Henriques, Andrea Vedaldi, Andrew Zisserman (2024)

## 🧩 Problem to Solve

본 논문은 1인칭 시점의 egocentric 비디오에서 발생하는 인스턴스 세그멘테이션(Instance Segmentation) 및 트래킹(Tracking)의 어려움을 해결하고자 한다. egocentric 비디오는 카메라의 급격한 움직임, 빈번한 객체 가려짐(occlusion), 그리고 객체가 시야에서 완전히 사라졌다가 다시 나타나는 현상 등으로 인해 기존의 2D 기반 Video Object Segmentation (VOS) 모델만으로는 일관된 객체 추적이 매우 어렵다.

기존의 2D VOS 방식들은 주로 느리고 안정적인 카메라 움직임을 가정하며, 광학 흐름(optical flow)이나 포인트 트래킹(point tracking)과 같은 조밀한 대응 관계(correspondences)에 의존한다. 그러나 이러한 방식은 심한 시점 변화가 일어나는 egocentric 환경에서는 신뢰도가 급격히 떨어지며, 결과적으로 객체 트랙이 단절되고 불완전해지는 문제가 발생한다. 따라서 본 연구의 목표는 3D 공간 인지 능력을 통합하여 가려짐이나 시야 이탈 상황에서도 객체의 정체성을 유지할 수 있는 강건한 트래킹 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 '대상 영속성(object permanence)' 개념을 모방하여, 객체가 시야에서 사라지더라도 3D 공간상의 위치는 유지된다는 점을 이용하는 것이다. 이를 위해 2D VOS 모델이 생성한 초기 세그먼트를 3D 공간으로 투영(lifting)하고, 시각적 특징과 더불어 3D 기하학적 정보를 결합한 새로운 비용 함수(cost function)를 통해 트랙을 정제한다. 특히, 단순히 현재 프레임의 매칭에 그치지 않고, 매칭되지 않은 트랙을 데이터베이스에 유지함으로써 객체가 일시적으로 사라졌다가 다시 나타났을 때 재식별(re-identification)할 수 있도록 설계하였다.

## 📎 Related Works

본 논문은 크게 세 가지 관련 연구 분야를 다룬다. 첫째, VOS 분야에서는 memory networks(STM, XMem)나 Transformer 기반의 모델(VisTR, SeqFormer), 그리고 최근의 DEVA나 MASA와 같이 이미지 수준의 세그멘테이션과 시간적 전파를 분리한 접근 방식들이 제안되었다. 둘째, 포인트 트래킹 기반 방식(CoTracker, TAPIR)은 픽셀 단위의 대응 관계를 통해 VOS 성능을 높이려 했으나, 급격한 시점 변화 시 신뢰도가 낮다는 한계가 있다. 셋째, 3D-informed 세그멘테이션 연구(Panoptic Lifting, Gaussian Grouping)는 주로 정적인 장면에서 2D 라벨을 3D로 융합하는 데 집중해 왔다.

본 연구는 기존의 3D 트래킹 연구인 OSNOM의 아이디어를 계승하되, 단순히 3D 중심점(centroid)만을 이용하는 것이 아니라 베이스가 되는 2D VOS 모델의 인스턴스 ID와 카테고리 정보를 비용 함수에 통합함으로써, 노이즈가 많은 2D 입력 트랙을 효과적으로 정제할 수 있다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

본 제안 방법은 사전 학습된 2D VOS 모델로부터 얻은 초기 세그먼트와 트랙을 입력으로 받는다. 이후 각 프레임의 깊이 지도(depth map)와 카메라 파라미터를 이용하여 2D 세그먼트를 3D 공간으로 투영하고, 3D-aware 트래킹 비용 함수를 통해 프레임별로 트랙을 최적화하여 최종적으로 일관된 장기 트랙을 생성한다.

### 3D 투영 및 속성 추출

각 세그먼트 $m_i^t$에 대해 다음과 같은 속성 벡터 $b_i^t = (l_i^t, v_i^t, \bar{c}_i^t, \bar{s}_i^t)$를 정의한다.

1. **3D 위치 ($l_i^t$):** 2D 중심점 $(x_i^t, y_i^t)$를 깊이 값 $d_i^t$, 카메라 내적 파라미터 $K$, 외적 파라미터 $C^t$를 이용하여 3D 좌표로 변환한다.
   $$l_i^t = C^t \left[ d_i^t K^{-1} \begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix} \right]$$
2. **시각적 특징 ($v_i^t$):** DINOv2 인코더를 사용하여 세그먼트 영역의 특징 벡터를 추출한다.
3. **카테고리 및 인스턴스 라벨 ($\bar{c}_i^t, \bar{s}_i^t$):** 2D VOS 모델이 제공한 초기 클래스 및 ID 정보이다.

### 트래킹 비용 함수 및 최적화

현재 프레임 $t$의 세그먼트 $m_i^t$와 이전 프레임까지 유지된 트랙 $T_j^{t-1}$ 사이의 매칭을 위해 헝가리안 알고리즘(Hungarian algorithm)을 사용하여 다음의 비용 함수 $J$를 최소화한다.
$$J(s_i^t, \tilde{s}_j^{t-1}, b_i^t, \tilde{b}_j^{t-1}) = \mathbb{1}(s_i^t = \tilde{s}_j^{t-1}) \cdot \sum_{p=1}^{n} \delta_p(b_{i,p}^t, \tilde{b}_{j,p}^{t-1})$$

여기서 각 속성별 비용 함수 $\delta_p$는 다음과 같이 정의된다.

- **3D 위치 비용 ($\delta_l$):** 지수 분포를 사용하여 거리가 멀수록 비용이 증가한다.
  $$\delta_l(l_i^t, l_j^{t'}) = -\log \left( \frac{1}{\alpha_l} \exp(-\|l_i^t - l_j^{t'}\|^2) \right)$$
- **시각적 특징 비용 ($\delta_v$):** 코시 분포(Cauchy distribution)를 사용하여 특징 간의 유사도를 측정한다.
  $$\delta_v(v_i^t, v_j^{t'}) = -\log \left( \frac{1}{1 + \alpha_v \|v_i^t - v_j^{t'}\|^2_2} \right)$$
- **카테고리 및 인스턴스 비용 ($\delta_c, \delta_s$):** 동일하면 0, 다르면 가중치 $\alpha$를 부여하는 0-1 비용 함수를 사용한다.

### 트랙 유지 전략

매칭 비용이 임계값 $\gamma$보다 높은 경우 새로운 트랙을 생성한다. 중요한 점은 현재 프레임에서 매칭되지 않은 기존 트랙을 즉시 삭제하지 않고 데이터베이스에 유지하며 속성을 전파한다는 것이다. 이를 통해 객체가 일시적으로 시야에서 사라졌다가 다시 나타났을 때, 3D 위치와 시각적 특징을 바탕으로 기존 ID를 다시 부여할 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋:** EPIC Fields 데이터셋의 20개 장면을 주 실험 대상으로 하며, Ego4D 데이터셋에서도 추가 검증을 수행하였다.
- **비교 대상:** state-of-the-art 2D VOS 기반 트래커인 DEVA (with OWLv2) 및 MASA (with Detic)와 비교하였다.
- **평가 지표:** HOTA, DetA, AssA (Association Accuracy), IDF1 score를 사용하였다.

### 주요 결과

- **정량적 성능:** 제안 방법은 모든 지표에서 베이스라인을 능가하였다. 특히 DEVA 대비 HOTA는 25.14 $\rightarrow$ 27.72로, AssA는 36.72 $\rightarrow$ 43.90로 크게 향상되었다.
- **ID 스위치 감소:** 객체 클래스별 분석 결과, ID 스위치 횟수가 73%에서 80%까지 감소하였다. 예를 들어, 가려지기 쉬운 칼(knife)의 경우 평균 27.21회에서 5.29회로 급감하였다.
- **DetA의 일정함:** DetA(탐지 정확도)는 베이스라인과 거의 동일하게 유지되었는데, 이는 본 방법이 세그멘테이션 마스크 자체를 수정하는 것이 아니라 3D 정보를 이용해 ID 할당(association)만을 정제하기 때문이다.

### 절제 연구 (Ablation Study)

비용 함수에서 각 구성 요소를 제거하며 분석한 결과, **카테고리 정보($\delta_c$)**와 **인스턴스 정보($\delta_s$)**가 성능에 가장 큰 영향을 미쳤다. 3D 위치 정보($\delta_l$) 또한 중요하며, 이를 제거했을 때 HOTA 점수가 유의미하게 하락하였다.

## 🧠 Insights & Discussion

본 논문은 egocentric 비디오의 고유한 문제인 '시야 이탈'과 '잦은 가려짐'을 3D 공간 인지 능력을 통해 효과적으로 해결할 수 있음을 입증하였다. 특히, 2D VOS 모델의 결과를 완전히 대체하는 것이 아니라 이를 '사전 정보(prior)'로 활용하여 3D 정보와 결합함으로써, 순수 3D 매칭보다 훨씬 높은 강건성을 확보하였다.

또한, 이렇게 생성된 일관된 장기 트랙을 활용하여 두 가지 downstream application을 제시하였다.

1. **3D 객체 재구성:** 동일 객체의 일관된 트랙을 통해 다각도 뷰를 수집하고, 2D Gaussian Splatting을 통해 고품질의 3D 메쉬를 복원하였다.
2. **Amodal 세그멘테이션:** 복원된 3D 모델을 다시 렌더링함으로써, 원래 이미지에서 가려져 보이지 않는 부분까지 포함하는 전체 영역의 마스크(amodal mask)를 생성하였다.

**한계점:** 본 방법은 정확한 카메라 포즈(intrinsics, extrinsics)와 깊이 지도에 크게 의존한다. 따라서 카메라 파라미터 추정이 어렵거나, 모션 블러 및 저조도 환경으로 인해 깊이 추정의 정밀도가 떨어지는 경우 성능 저하가 발생할 수 있다.

## 📌 TL;DR

본 연구는 egocentric 비디오에서 2D VOS의 고질적인 문제인 트랙 단절을 해결하기 위해 **3D 공간 정보(위치, 깊이, 카메라 포즈)를 통합한 트래킹 정제 프레임워크**를 제안한다. 3D-aware 비용 함수와 트랙 유지 전략을 통해 객체가 시야에서 사라졌을 때도 정체성을 유지함으로써 **Association Accuracy(AssA)를 크게 향상**시켰으며, 이를 통해 **정밀한 3D 객체 복원 및 Amodal 세그멘테이션**이 가능함을 보여주었다.
