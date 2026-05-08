# SurgTrack: CAD-Free 3D Tracking of Real-world Surgical Instruments

Wenwu Guo, Jinlin Wu, Zhen Chen, Qingxiang Zhao, Miao Xu, Zhen Lei, Hongbin Liu (2024)

## 🧩 Problem to Solve

본 논문은 시각 기반의 수술 내비게이션 시스템에서 필수적인 수술 도구의 3D 트래킹(Tracking) 문제를 해결하고자 한다. 기존의 2D 트래킹 방식은 평면상의 위치와 회전만을 인식하여 자유도가 낮으며, 이는 정밀한 수술 내비게이션에 필요한 충분한 정보를 제공하지 못한다.

반면, 6-DoF(Degrees of Freedom)를 제공하는 3D 트래킹 방식은 임상적으로 더 가치가 높지만, 실제 적용에는 다음과 같은 세 가지 주요한 어려움이 존재한다. 첫째, 수술 도구의 3D 정합(Registration)에 필요한 CAD 모델이 특허 보호 등의 이유로 기업의 기밀 사항인 경우가 많아 입수하기 어렵다. 둘째, 수술 도구의 표면 텍스처가 매우 약해 특징점 추출이 어렵다. 셋째, 수술 과정에서 도구가 빈번하게 가려지는 Occlusion(폐색) 현상이 발생한다.

따라서 본 연구의 목표는 CAD 모델 없이도(CAD-free) 실제 수술 환경에서 강건하게 수술 도구의 3D 포즈를 트래킹할 수 있는 SurgTrack 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 두 단계의 파이프라인을 통해 CAD 모델의 부재와 환경적 제약을 극복하는 것이다.

1. **CAD-free 3D Registration**: CAD 모델 대신 RGB-D 비디오 프레임으로부터 수술 도구의 3D 표현을 학습하는 Instrument Signed Distance Field (SDF) 모델을 도입하여 3D 정합을 수행한다.
2. **Robust 3D Tracking**: 폐색과 약한 텍스처 문제를 해결하기 위해, 과거의 트래킹 결과를 저장하는 Posture Memory Pool과 이를 활용해 현재 포즈를 최적화하는 Posture Graph Optimization 모듈을 제안한다.
3. **Instrument3D 데이터셋**: 다양한 수술 도구의 3D 트래킹 성능을 종합적으로 평가하기 위해 새로운 RGB-D 데이터셋인 Instrument3D를 구축하여 공개하였다.

## 📎 Related Works

기존의 수술 도구 트래킹 연구는 크게 세 가지 방향으로 진행되었다. 초기 연구들은 도구에 마커(Marker)를 부착하여 트래킹하였으나, 이는 도구의 침습성을 높이고 확장성이 떨어진다는 단점이 있다. 이후 마커 없는(Marker-free) 2D 트래킹 방식들이 제안되었으며, 최근에는 YOLOv5와 ReID 기술을 결합해 정확도를 높였으나, 여전히 2D 평면상의 정보(3-DoF)만 제공한다는 한계가 있다.

3D 객체 트래킹 분야에서는 일반적으로 미리 정의된 CAD 모델을 사용하여 6-DoF 포즈를 추정한다. 하지만 앞서 언급한 바와 같이 수술 도구의 경우 CAD 모델 확보가 사실상 불가능하며, 수술 도구 특유의 낮은 텍스처와 빈번한 폐색은 일반적인 3D 트래킹 알고리즘의 성능을 저하시키는 요인이 된다. SurgTrack은 이러한 CAD 모델 의존성을 완전히 제거하고, 메모리 풀과 그래프 최적화를 통해 환경적 강건성을 확보함으로써 기존 방식들과 차별화된다.

## 🛠️ Methodology

SurgTrack은 크게 **Registration stage**와 **Tracking stage**의 두 단계로 구성된다.

### 1. CAD-free Instrument Registration

CAD 모델 없이 도구의 3D 형상을 정의하기 위해 Signed Distance Function (SDF)을 사용한다. RGB-D 카메라로 캡처한 포인트 클라우드 $\{v | v \in \mathbb{R}^3\}$가 주어졌을 때, 도구의 표면을 다음과 같이 정의한다.
$$S = \{v | \Psi(v) = 0\}$$
여기서 $\Psi(v) = 0$은 도구 표면 위의 점들을 의미한다.

폐색과 약한 텍스처 문제를 해결하기 위해 두 가지 제약 조건을 손실 함수에 추가한다.

- **Occlusion Constraint**: 부분 폐색으로 인해 배경과 도구 사이의 경계가 모호해지는 것을 방지하기 위해 양수 값 $\delta$를 도입한 손실 함수 $L_{occ}$를 정의한다.
$$L_{occ} = \frac{1}{|V_{occ}|} \sum_{v \in V_{occ}} (\Psi(v) - \delta)^2$$
- **Shape Constraint**: 텍스처가 약한 표면의 기하학적 구조를 더 잘 포착하기 위해 표면 근처의 점들을 고려하는 $L_{surf}$를 정의한다.
$$L_{surf} = \frac{1}{|V_{surf}|} \sum_{v \in V_{surf}} (\Psi(v) + d_v - d_\Delta)^2$$

최종 registration 손실 함수는 $L = \alpha L_{occ} + \beta L_{surf}$로 정의되며, 이를 통해 CAD 모델 없이도 도구의 3D 형상을 재구성한다.

### 2. Instrument Tracking

트래킹 단계는 거친 포즈 추정(Coarse estimation)과 정밀 최적화(Refinement) 순으로 진행된다.

**트래킹 초기화 (Initialization)**: RANSAC 알고리즘을 사용하여 현재 프레임과 인접 프레임 사이의 거리 오차를 최소화하는 초기 포즈 $\tilde{\xi}_t$를 추정한다.
$$\tilde{\xi}_t = \arg \min_{R,t} \sum_{i} \|Rp_i + t - q_i\|^2$$

**트래킹 최적화 (Optimization)**: 초기 포즈 $\tilde{\xi}_t$를 바탕으로 Posture Memory Pool $P$와 Posture Graph $G$를 이용하여 포즈를 정밀하게 수정한다.

- **Posture Memory Pool**: 과거 프레임들의 최적화된 포즈 $\xi_i$와 관련 3D 포인트 클라우드 $M_i$를 저장한다.
- **Posture Graph**: 현재 프레임 $F_t$와 메모리 풀에서 선택된 참조 프레임 $P_{pg}$를 노드로 하여 그래프 $G=(V, E)$를 구성한다.

최종 최적화된 포즈 $\xi_t$는 다음의 통합 손실 함수를 최소화함으로써 결정된다.
$$\xi_t \leftarrow \arg \min_{\xi_t} \left( w_s L_{SDF}(t) + \sum_{i,j \in V, i \neq j} [w_f L_{3D}(i,j) + w_p L_{2D}(i,j)] \right)$$

여기서 각 손실 함수의 역할은 다음과 같다.

- $L_{3D}(i,j)$: 대응되는 RGB-D 특징점 간의 유클리드 거리를 측정하며, 이상치에 강건하도록 Huber loss $\rho$를 적용한다.
- $L_{2D}(i,j)$: 투영 후의 픽셀 단위 point-to-plane 거리를 측정하여 정밀도를 높인다.
- $L_{SDF}(t)$: 현재 프레임과 앞서 학습한 Instrument SDF의 암시적 표면(Implicit surface) 사이의 거리를 측정하여, 텍스처가 없는 영역에서도 형상 정보를 바탕으로 트래킹이 가능하게 한다.
$$L_{SDF}(t) = \sum_{p \in I_t} \rho(|\Psi(\xi_t^{-1}(\pi_D^{-1}(p)))|)$$

## 📊 Results

### 실험 설정

- **데이터셋**: 자체 구축한 **Instrument3D**(5종의 수술 도구, 13개 비디오)와 일반 객체 3D 트래킹 데이터셋인 **HO3D**를 사용하였다.
- **평가 지표**: 트래킹 정확도는 ADD 및 ADD-S (AUC percentage)로 측정하였고, 재구성 오차는 Chamfer Distance (CD)를 사용하였다.

### 정량적 결과

- **Instrument3D**: SurgTrack은 ADD-S 88.82%, ADD 83.65%, CD 12.85cm를 기록하며 Pixtrack, OnePose 등의 기존 방법론보다 압도적인 성능을 보였다.
- **HO3D**: BundleTrack과 비교했을 때 ADD-S에서는 유사한 성능을 보였으나, ADD 지표와 재구성 오차(CD 0.65cm) 면에서 훨씬 뛰어난 결과를 얻었다. 특히 BundleTrack이 300회 이상의 학습이 필요한 반면, SurgTrack은 더 효율적으로 작동함을 확인하였다.

### 절제 연구 (Ablation Study)

각 모듈의 기여도를 분석한 결과:

- **SDF 최적화(Occlusion/Texture)** 제거 시 ADD-S가 12.43% 감소하였다.
- **Posture Memory Pool** 제거 시 ADD-S가 13.17% 감소하여, 과거 데이터 유지의 중요성이 입증되었다.
- **Posture Graph** 최적화 방식에서 무작위 선택 대신 매칭도가 높은 서브셋을 선택했을 때 CD 오차가 약 3cm 감소하고 ADD가 41.29% 증가하는 유의미한 개선이 있었다.

## 🧠 Insights & Discussion

SurgTrack의 가장 큰 강점은 수술 도구 트래킹의 고질적인 문제인 **CAD 모델 의존성**을 해결했다는 점이다. SDF를 이용해 런타임 또는 사전 단계에서 도구의 형상을 학습함으로써, 제조사의 기밀 모델 없이도 6-DoF 트래킹이 가능함을 보였다.

또한, 단순히 현재 프레임의 특징점에 의존하지 않고 **Posture Memory Pool**과 **SDF depth loss**를 결합한 점이 인상적이다. 이는 텍스처가 거의 없는 금속성 수술 도구 표면에서 특징점 매칭이 실패하더라도, 학습된 3D 형상 정보와 과거의 포즈 이력을 통해 트래킹의 연속성을 유지할 수 있게 한다.

다만, 본 연구는 RGB-D 카메라를 전제로 하고 있어, 실제 수술실에서 깊이 정보를 정확하게 제공할 수 있는 센서의 확보와 실시간성(Real-time performance)에 대한 구체적인 분석이 추가될 필요가 있다. 또한, 매우 동적인 환경에서 급격한 움직임이 발생했을 때 RANSAC 기반의 초기 추정이 실패할 가능성에 대한 논의가 보완되어야 할 것이다.

## 📌 TL;DR

본 논문은 CAD 모델 없이 수술 도구를 3D 트래킹하는 **SurgTrack** 프레임워크를 제안한다. SDF를 통해 도구의 3D 형상을 스스로 학습하여 정합을 수행하고, 포즈 메모리 풀과 그래프 최적화를 통해 폐색과 약한 텍스처 환경에서도 강건한 6-DoF 트래킹을 구현하였다. 새롭게 구축한 Instrument3D 데이터셋에서 SOTA 성능을 달성하였으며, 이는 향후 CAD 모델 확보가 어려운 실제 임상 수술 내비게이션 시스템에 직접적으로 적용될 가능성이 매우 높다.
