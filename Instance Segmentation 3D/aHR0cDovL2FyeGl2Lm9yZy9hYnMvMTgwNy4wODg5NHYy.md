# ClusterNet: 3D Instance Segmentation in RGB-D Images

Lin Shao, Ye Tian, and Jeannette Bohg (2018)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 RGB-D 이미지에서의 **인스턴스 세그멘테이션(Instance Segmentation)**이다. 특히 자율 주행 로봇이 복잡한 실제 환경에서 객체의 위치, 기하학적 구조, 개수를 정확히 파악하는 것은 안전하고 강건한 의사결정을 위해 필수적이다.

기존의 인스턴스 세그멘테이션 방식, 특히 Mask R-CNN과 같은 **Region Proposal 기반 방식**은 다음과 같은 한계가 있다.

- **객체 간 겹침(Occlusion) 문제:** 객체들이 심하게 겹쳐 있는 경우, 하나의 bounding box 안에 여러 인스턴스가 포함될 수 있어 정확한 분리가 어렵다.
- **가변적인 객체 수:** 이미지 내에 포함된 객체의 수가 임의적이며, 각 인스턴스에 부여되는 레이블은 순서에 상관없이 구별만 되면 되는 **순열 불변성(Permutation-invariance)** 특성을 가져야 한다.

따라서 본 논문의 목표는 Region Proposal 과정을 제거하고, Depth 데이터를 효율적으로 활용하여 겹침이 심한 환경에서도 강건하게 작동하는 3D 인스턴스 세그멘테이션 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 세그멘테이션 문제를 **3D 객체 특성 공간에서의 회귀(Regression) 및 클러스터링(Clustering) 문제로 변환**하는 것이다.

- **Explicit Embedding:** 객체를 표현하기 위해 객체 점유 함수(Object Occupancy Function)의 1차 및 2차 모멘트(Moments)를 사용한다. 이를 통해 각 픽셀이 자신이 속한 객체의 3D 중심점, 크기, 포즈를 예측하도록 한다.
- **Proposal-free Framework:** Bounding box를 생성하는 단계 없이, 픽셀 단위의 예측값들을 특성 공간에서 클러스터링함으로써 인스턴스를 분리한다.
- **RGB-D 통합 활용:** RGB 이미지뿐만 아니라 Depth 및 XYZ 포인트 클라우드 데이터를 함께 입력으로 사용하여 3D 기하학적 정보를 직접적으로 활용한다.

## 📎 Related Works

### 1. Region Proposal 기반 접근 방식

- **특징:** Bounding box를 먼저 생성하고 그 내부에서 객체를 분류 및 세그멘테이션한다 (예: Mask R-CNN, Fast R-CNN).
- **한계:** 객체가 심하게 가려져 있거나 밀집된 환경에서는 제안된 영역 내에 여러 객체가 섞여 있어 성능이 저하된다.

### 2. Region Proposal-free 접근 방식

- **특징:** 제안 단계를 생략하고 픽셀 특징을 직접 학습하거나 객체 표현 방식을 달리한다.
- **기존 연구와의 차별점:**
  - **Implicit Embedding 방식** (예: De Brabandere et al., Newell et al.)은 학습된 특징 공간에서 픽셀들을 가깝게 배치하지만, 본 논문은 객체의 중심과 포즈라는 **명시적(Explicit)인 기하학적 특성**을 사용한다.
  - **객체 수 예측 방식** (예: Liang et al.)은 전체 객체 수를 먼저 예측해야 하므로, 수 예측이 틀릴 경우 전체 결과가 왜곡되는 문제가 있다. 반면, ClusterNet은 특성 공간의 클러스터링을 통해 객체 수를 추론한다.

## 🛠️ Methodology

### 1. 객체 표현 (Object Representation)

객체 $O_k$에 대해, 카메라 좌표계에서의 bounding box 중심을 $C^x_k = (c_x, c_y, c_z)^T$라고 한다. 또한 객체 점들의 2차 모멘트(분산 및 공분산)를 $(xx, yy, zz, xy, yz, zx)^T$로 정의한다. 최종적으로 객체 특성 $\xi_k$는 다음과 같이 정의된다.

$$\xi_k = (c_x, c_y, c_z, c_x+xx, c_y+yy, c_z+zz, c_x+xy, c_y+yz, c_z+zx)$$

여기서 앞의 세 성분은 객체의 **위치(Location)**를, 뒤의 여섯 성분은 객체의 **크기 및 포즈(Size and Pose)**를 나타낸다.

### 2. 인스턴스 세그멘테이션 과정

네트워크는 각 픽셀 $(u, v)$에 대해 다음 세 가지를 예측한다.

- $\hat{\xi}_{uv}$: 해당 픽셀이 속한 객체의 특성 값 (중심 및 포즈).
- $\hat{\eta}_{uv}$: 해당 픽셀이 클러스터의 중심(Centroid)일 확률.
- $\hat{B}_{uv}$: 동일 객체 픽셀들을 포함하는 구(Sphere)의 반지름 추정치.

**클러스터링 절차:**

1. **초기화:** $\hat{\eta}_{uv}$가 가장 높은 픽셀을 시드로 선택한다.
2. **구 분할:** 선택된 시드의 $\hat{\xi}_{uv}$를 중심으로 하고 반지름이 $\hat{B}_{uv}$인 구 내에 포함되는 특성값 $\hat{\xi}_{mn}$을 가진 모든 픽셀을 하나의 객체로 할당한다.
3. **반복:** 할당되지 않은 픽셀들에 대해 위 과정을 반복한다.
4. **정교화(Refinement):** 초기화된 클러스터들을 기반으로 Gaussian Mixture Model(GMM)을 1회 수행하여 세그멘테이션 결과를 최적화한다.

### 3. 네트워크 아키텍처 (ClusterNet Architecture)

- **입력:** RGB, XYZ(Depth에서 변환), Depth 이미지 세 가지를 입력으로 받는다.
- **인코더:** RGB와 XYZ는 pre-trained ResNet50을 통해 특성을 추출하며, Depth는 VGG 아키텍처를 통해 처리된다.
- **특성 융합:** C1 레벨 특성과 C4 레벨 특성을 추출하고, Atrous Spatial Pyramid Pooling(ASPP) 모듈을 통해 수용 영역(Receptive field)을 확장한 뒤 이를 결합하여 최종 인코딩 단계 $E^f$를 생성한다.
- **디코더:** $E^f$로부터 객체 특성 $\hat{\xi}$, 중심 확률 $\hat{\eta}$, Foreground/Background 마스크를 예측하며, $\hat{B}$는 C4 특성에서 직접 디코딩한다.

### 4. 손실 함수 (Loss Function)

전체 손실 함수 $L$은 다음과 같이 정의된다.
$$L = \lambda_s L_s + \lambda_{cen} L_{cen} + L_p + \lambda_{var} L_{var} + \lambda_{vio} L_{vio}$$

- **Semantic Mask Loss ($L_s$):** 전경/배경을 구분하는 Cross-entropy 손실이다.
- **Cluster Center Loss ($L_{cen}$):** 픽셀이 중심점일 확률 $\eta_{uv}$에 대한 Cross-entropy 손실이다.
- **Pixel-wise Loss ($L_p$):** 예측된 특성 $\hat{\xi}_{uv}$와 반지름 $\hat{B}_{uv}$의 L2-norm 오차를 최소화한다.
- **Variance Loss ($L_{var}$):** 동일 객체에 속한 픽셀들이 서로 유사한 특성값을 갖도록 유도한다.
$$L_{var} = \sum_{k} \frac{1}{N_k} \sum_{(u,v) \in O_k} \| \hat{\xi}_{uv} - \bar{\hat{\xi}}_{uv} \|^2$$
- **Violation Loss ($L_{vio}$):** 예측값이 정답 $\xi_{uv}$로부터 일정 거리($\lambda_v \hat{B}_{uv}$) 이상 떨어져 있을 때만 강하게 페널티를 부여한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** ShapeNet 모델을 기반으로 생성된 합성 데이터셋(Synthetic Dataset)을 사용하였다. 객체 간 겹침이 심하여 기존 Proposal 기반 방식의 한계를 테스트하기에 적합하다.
- **평가 지표:** COCO 데이터셋의 기준인 AP(Average Precision)와 AR(Average Recall)을 사용하였으며, IoU 임계값 0.5 및 0.75를 기준으로 측정하였다. 또한 객체 크기별($S, M, L$) 및 가려짐 정도(Heavy, Medium, Little Occlusion)에 따른 분석을 수행하였다.
- **비교 대상:** Mask R-CNN (C4 및 FPN 백본).

### 2. 정량적 결과

- **전체 성능:** 제안 방법인 `Our(c)` (중심점만 사용)와 `Our(c+cov)` (중심점+공분산 사용) 모두 Mask R-CNN을 크게 상회하는 성능을 보였다. 특히 `Our(c)`의 AP는 66.2로 Mask R-CNN(FPN)의 51.4보다 훨씬 높다.
- **가려짐 상황:** Table II 결과, 가려짐이 심한 객체(AR HO)에 대해 `Our(c)`가 50.7의 Recall을 기록하여 Mask R-CNN(41.5~41.9)보다 우수한 강건성을 입증하였다.
- **입력 모달리티 영향:**
  - RGB만 사용했을 때보다 Depth를 추가했을 때 성능이 크게 향상되었다.
  - XYZ 포인트 클라우드를 추가하면 성능이 더 상승하며, 특히 큰 객체($AP_L, AR_L$)의 경우 Depth 이미지의 정보가 유용하게 작용함을 확인하였다.
- **특성 선택 영향:** 중심점만 예측하는 `Our(c)`가 2차 모멘트까지 예측하는 `Our(c+cov)`보다 성능이 더 좋게 나타났다. 이는 2차 모멘트 예측의 난이도가 높고 오차 분산이 크기 때문으로 분석된다.

### 3. 정성적 결과 (Real World Data)

실제 Intel RealSense SR300 카메라로 촬영한 데이터에 적용한 결과, Mask R-CNN은 합성 데이터에서 학습된 모델이 실제 데이터로 전이(Transfer)되지 않아 거의 작동하지 않은 반면, ClusterNet은 별도의 파인튜닝 없이도 객체 인스턴스들을 정확하게 분리해내는 뛰어난 일반화 성능을 보였다.

## 🧠 Insights & Discussion

**강점:**

- **Proposal-free의 이점:** Bounding box에 의존하지 않으므로 객체가 심하게 겹쳐 있는 상황에서도 3D 특성 공간에서의 분리를 통해 정확한 세그멘테이션이 가능하다.
- **명시적 특성 활용:** 단순한 임베딩이 아닌 3D 중심, 크기, 포즈라는 물리적 의미를 가진 특성을 학습함으로써 로봇 조작(Manipulation) 태스크에 직접적으로 활용 가능한 정보를 제공한다.
- **강력한 일반화:** 기하학적 특성에 기반한 학습 덕분에 합성 데이터에서 학습한 모델이 실제 환경에서도 잘 작동하는 낮은 Transfer Gap을 보여준다.

**한계 및 논의:**

- **2차 모멘트 예측의 어려움:** 이론적으로는 포즈와 크기를 알 수 있는 2차 모멘트가 유용해야 하지만, 실제로는 예측 오차가 커서 오히려 성능을 저하시켰다. 이는 고차원 특성을 더 정확하게 회귀하기 위한 손실 함수나 네트워크 구조의 개선이 필요함을 시사한다.
- **세그멘테이션 범위:** 현재는 Foreground와 Background라는 두 가지 클래스로만 구분되어 있으며, 실제 적용을 위해서는 구체적인 클래스 정보를 포함하는 Semantic Segmentation과의 통합이 필요하다.

## 📌 TL;DR

본 논문은 RGB-D 이미지를 입력으로 하여 3D 인스턴스 세그멘테이션을 수행하는 **ClusterNet**을 제안한다. 이 모델은 Region Proposal 과정 없이, 각 픽셀이 속한 객체의 **3D 중심점과 포즈(모멘트)를 직접 회귀 예측**하고 이를 특성 공간에서 **클러스터링**하여 객체를 분리한다. 실험 결과, 특히 객체 간 겹침이 심한 환경에서 기존 Mask R-CNN보다 월등한 성능을 보였으며, 실제 데이터에서도 높은 일반화 능력을 입증하였다. 이 연구는 특히 정밀한 3D 객체 파악이 필요한 **로봇 매니퓰레이션 분야**에 중요한 기여를 할 것으로 기대된다.
