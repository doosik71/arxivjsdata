# Applying Eigencontours to PolarMask-Based Instance Segmentation

Wonhui Park, Dongkwon Jin, Chang-Su Kim (2022)

## 🧩 Problem to Solve

본 논문은 인스턴스 분할(Instance Segmentation) 작업에서 발생하는 연산 효율성과 경계 복원 정확도 사이의 트레이드오프 문제를 해결하고자 한다. Mask R-CNN과 같은 기존의 픽셀 단위 분류(pixel-wise classification) 방식은 우수한 성능을 보이지만, 모든 픽셀을 분류해야 하므로 계산 비용이 매우 높다는 단점이 있다.

이를 해결하기 위해 경계 회귀(boundary regression) 기반의 기법들이 제안되었다. 예를 들어 PolarMask는 객체의 경계를 극좌표계(polar coordinates)로 표현하여 연산 효율성을 높였으나, 이러한 방식들은 복잡한 객체 경계를 정확하게 재구성하는 데 한계가 있다. 따라서 본 연구의 목표는 데이터 기반의 경계 기술자인 Eigencontours를 PolarMask 프레임워크에 통합하여, 연산 효율성을 유지하면서도 객체 경계 복원 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 특이값 분해(Singular Value Decomposition, SVD)를 통해 학습 데이터셋의 모든 객체 경계에서 공통적으로 나타나는 주성분을 추출하고, 이를 기반으로 한 저차원 표현법인 Eigencontours를 사용하는 것이다. 

기존 PolarMask가 경계를 표현하기 위해 다수의 방사형 좌표값($N$개의 변수)을 직접 예측해야 했다면, 제안 방법은 미리 정의된 $M$개의 Eigencontours의 선형 결합으로 경계를 표현한다. 결과적으로 네트워크는 $N$이 아닌 훨씬 작은 차원인 $M$개의 계수(coefficient)만을 예측하면 되므로, 모델의 효율성을 높이는 동시에 데이터에 내재된 기하학적 특성을 반영하여 더 정확한 경계를 생성할 수 있다.

## 📎 Related Works

인스턴스 분할 연구는 크게 두 가지 방향으로 나뉜다. 첫째는 Mask R-CNN 및 그 변형 모델들로, 바운딩 박스를 먼저 검출한 후 내부 픽셀을 분류하는 방식이다. 이는 정확도는 높지만 픽셀 단위 연산으로 인해 계산 비용이 크다.

둘째는 경계 기술자(contour descriptors)를 이용한 방식이다. ESE-Seg는 다항식 피팅 계수(polynomial fitting coefficients)를 사용하여 경계를 표현하며, PolarMask는 중심점으로부터의 방사형 거리(centroidal profiles)를 이용해 극좌표계로 경계를 기술한다. 하지만 이러한 고정된 방식의 기술자들은 실제 데이터의 복잡한 형태를 모두 담아내지 못하는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 데이터로부터 직접 학습된 Eigencontours를 도입하여 기존 방식과 차별화를 꾀한다.

## 🛠️ Methodology

### 1. Eigencontour Representation

객체 경계를 Eigencontour 공간으로 표현하는 과정은 다음과 같다.

**Step 1: Star-convex contour 생성**
먼저 객체의 중심점 $o = (x, y)$를 정의하고, 균일하게 샘플링된 각도 $\theta_i$ ($i=1, \dots, N$)에 대해 중심에서 경계까지의 최대 거리 $r_i$를 측정한다. 이를 통해 $N$차원 벡터 $r = [r_1, r_2, \dots, r_N]^\top$으로 표현되는 Star-convex contour를 생성한다.

**Step 2: Eigencontour 구축 (SVD)**
학습 세트의 모든 객체 경계 벡터를 열벡터로 갖는 행렬 $A = [r_1, r_2, \dots, r_L]$를 구성한다. 이 행렬 $A$에 대해 특이값 분해(SVD)를 수행한다.
$$A = U\Sigma V^\top$$
여기서 $U = [u_1, \dots, u_N]$와 $V = [v_1, \dots, v_L]$는 직교 행렬이며, $\Sigma$는 특이값 $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$을 대각 성분으로 갖는 행렬이다. 이때 상위 $M$개의 왼쪽 특이 벡터 $u_1, \dots, u_M$을 **Eigencontours**라고 정의하며, 이들이 생성하는 공간을 Eigencontour 공간이라 한다.

**Step 3: 경계 표현 및 복원**
임의의 경계 $r$은 $M$개의 Eigencontours의 선형 결합으로 근사할 수 있다.
$$\tilde{r} = U_M c = [u_1, \dots, u_M]c$$
여기서 $c$는 $M$차원의 계수 벡터이며, $c = U_M^\top r$로 계산된다. 즉, $N$차원의 경계 정보를 $M$차원의 계수 벡터 $c$로 압축하여 표현하는 것이다.

### 2. Network Architecture

제안하는 네트워크는 하나의 Encoder와 세 개의 Decoder로 구성된다.

- **Encoder**: ResNet50 백본과 Feature Pyramid Network(FPN)를 사용하여 이미지로부터 $H \times W \times C$ 크기의 특징 맵을 추출한다.
- **Decoders**:
    - **Classification Decoder**: 각 픽셀이 $K$개의 카테고리 중 어디에 속하는지 예측하는 맵 $P \in \mathbb{R}^{H \times W \times K}$를 생성한다.
    - **Centerness Decoder**: 픽셀이 객체의 중심점일 확률을 나타내는 맵 $O \in \mathbb{R}^{H \times W \times 1}$를 생성한다.
    - **Coefficient Decoder**: 해당 픽셀을 중심으로 하는 객체의 경계 계수를 예측하는 맵 $R \in \mathbb{R}^{H \times W \times M}$를 생성한다.

### 3. Learning & Inference

**손실 함수(Loss Function)**: 전체 손실 함수 $L$은 다음과 같이 세 가지 손실의 합으로 정의된다.
$$L = L_{cls} + L_{cen} + L_{coeff}$$
- $L_{cls}$: 예측 맵 $P$와 정답 간의 Focal Loss.
- $L_{cen}$: 예측 맵 $O$와 정답 간의 Binary Cross-Entropy Loss.
- $L_{coeff}$: 예측 맵 $R$과 정답 간의 Polar IoU Loss.

**추론 및 후처리**:
1. 각 픽셀 $i$에 대해 중심점 점수 $O_i$와 클래스 최대 확률 $P_i$를 곱하여 신뢰도 점수를 계산한다.
2. 신뢰도가 0.05 미만인 픽셀은 제거한다.
3. Non-Maximum Suppression(NMS)을 수행하여 중복되는 마스크를 제거한다.
4. 최종 선택된 픽셀의 계수 $c$를 사용하여 $\tilde{r} = U_M c$ 식으로 경계 벡터를 복원하고, 이를 다시 이미지 좌표계의 마스크로 변환한다.

## 📊 Results

### 실험 설정
- **데이터셋**: COCO2017 (80개 클래스), SBD (20개 클래스)
- **비교 대상**: PolarMask (동일한 수 $M$의 파라미터를 사용하도록 설정)
- **평가 지표**: $AP$, $AP_{50}$, $AP_{75}$

### 정량적 결과
실험 결과, 제안 방법이 모든 지표에서 PolarMask보다 우수한 성능을 보였다. 특히 SBD 데이터셋에서 성능 향상 폭이 뚜렷하게 나타났다.

- **COCO2017 ($M=36$ 기준)**:
    - PolarMask: $AP=29.0, AP_{50}=48.9, AP_{75}=29.8$
    - Proposed: $AP=29.6, AP_{50}=49.9, AP_{75}=30.4$
- **SBD**:
    - PolarMask: $AP=27.7, AP_{50}=50.6, AP_{75}=25.7$
    - Proposed: $AP=30.7, AP_{50}=54.9, AP_{75}=29.6$

### 정성적 결과
제안 방법은 특히 객체의 굴곡진 부분(예: 자동차 범퍼, 동물의 머리 부분)을 PolarMask보다 더 정확하게 복원하는 경향을 보였다. 이는 Eigencontours가 데이터로부터 추출된 곡선 성분들의 집합이기 때문이다.

## 🧠 Insights & Discussion

**강점**: 
Eigencontours는 데이터 기반으로 구축되었기 때문에, 일반적인 객체 경계의 기하학적 특성을 잘 포착한다. 이를 통해 단순한 방사형 좌표 예측보다 더 매끄럽고 정확한 곡선 복원이 가능하다.

**한계 및 비판적 해석**:
1. **노이즈 발생**: Eigencontours의 선형 결합 과정에서 일부 경계가 불필요하게 떨리는(wiggling) 현상이 발생할 수 있다.
2. **Star-convexity 가정**: 본 연구는 PolarMask의 프레임워크를 그대로 사용하므로, 객체가 중심점으로부터 모든 방향으로 단조 증가하는 'Star-convex' 형태라는 가정을 전제로 한다. 따라서 말의 다리와 같이 가늘고 오목한(hollow) 부분은 제대로 복원하지 못하는 근본적인 한계가 있다.
3. **데이터셋별 성능 차이**: COCO2017보다 SBD에서 성능 향상이 더 컸는데, 이는 COCO의 경우 객체의 다양성이 너무 높고 폐색(occlusion)이 심해 전형적인 경계 패턴(typical contour patterns)을 추출하기 어려웠기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 SVD를 통해 학습 데이터의 공통 경계 특성을 추출한 **Eigencontours**를 PolarMask 네트워크에 적용하여 인스턴스 분할 성능을 높였다. $N$개의 좌표를 직접 예측하는 대신 $M$개의 기저 벡터 계수를 예측함으로써 연산 효율성을 유지하며 경계 복원 정확도를 개선했으며, 특히 곡선 형태의 경계 표현에서 강점을 보인다. 다만, Star-convex 가정으로 인해 오목한 형태의 객체 분할에는 한계가 있으며, 이는 향후 연구에서 해결해야 할 과제로 보인다.