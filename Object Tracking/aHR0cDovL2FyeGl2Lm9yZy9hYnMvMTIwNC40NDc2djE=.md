# Dynamic Template Tracking and Recognition

Rizwan Chaudhry, Gregory Hager, René Vidal (2012)

## 🧩 Problem to Solve

본 논문은 시간에 따라 국부적인 외형(appearance)과 움직임(motion)이 변화하는 비정형 객체(non-rigid objects)의 추적 문제를 해결하고자 한다. 이러한 객체에는 증기, 불, 연기, 물과 같은 Dynamic Textures뿐만 아니라, 다양한 동작을 수행하는 인간과 같은 Articulated Objects가 포함된다. 

기존의 추적 방법들은 외형 분포의 일관성이나 형태 및 윤곽의 유지(consistency)를 가정하는 경우가 많아, 외형이 끊임없이 진화하는 Dynamic Templates를 추적하는 데 한계가 있다. 따라서 본 연구의 목표는 객체의 외형 및 움직임의 시계열적 진화 과정을 명시적으로 모델링하여, 새로운 비디오 시퀀스에서도 해당 객체를 정확하게 추적하고 동시에 인식(recognition)할 수 있는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Linear Dynamical Systems (LDS)를 사용하여 Dynamic Template의 외형 변화를 모델링하고, 이를 커널 기반의 추적 프레임워크에 통합하는 것이다.

1. **LDS 기반 외형 모델링**: 객체의 외형 변화를 상태 공간 모델인 LDS로 정의하고, 샘플 비디오로부터 이 모델의 파라미터를 학습하여 Dynamic Template으로 사용한다.
2. **위치 및 상태의 공동 최적화**: 객체의 위치($l_t$)와 LDS의 잠재 상태($x_t$)를 동시에 추정하는 Maximum A-Posteriori (MAP) 추정 방식을 제안한다.
3. **미분 가능한 히스토그램 근사**: 기존의 이산적인 히스토그램 함수는 미분이 불가능하여 경사 하강법(gradient descent)을 적용할 수 없으나, 이를 시그모이드 함수(sigmoid function) 기반의 연속 함수로 근사하여 효율적인 최적화를 가능하게 했다.
4. **동시 추적 및 인식**: 학습된 여러 클래스의 LDS 모델들을 사용하여, 추적 과정에서 발생하는 재구성 비용(reconstruction cost)이나 시스템 간의 거리를 측정함으로써 추적과 인식을 동시에 수행한다.

## 📎 Related Works

기존의 비정형 객체 추적 연구들은 주로 다음과 같은 접근 방식을 취했다.

- **윤곽 및 서브스페이스 모델**: Spline을 이용한 윤곽 모델링이나 Robust Appearance Subspace 모델을 사용하였으나, 이는 외형 변화의 시간적 역동성(temporal dynamics)을 무시하고 각 프레임을 독립적으로 처리하는 경향이 있다.
- **인간 추적 및 동작 모델링**: Dynamic Bayesian Network나 스켈레톤 구조를 이용한 방법들이 제안되었으나, 이는 관절 각도 등 국부적인 모델에 치중되어 객체 전체의 글로벌 모델을 제공하지 못하며, 학습 데이터 구축에 많은 비용이 소요된다.
- **판별적(Discriminative) 접근법**: Boosting이나 Multiple Instance Learning을 통해 전경-배경 분류기를 학습시키는 방식은 점진적인 외형 변화에는 강건하지만, Dynamic Texture 특유의 내재적 시간 모델을 활용하지 않는다.
- **Dynamic Texture 전용 추적**: Pétéri(2010)의 연구가 있었으나, 이는 광학 흐름(optical flow) 특징을 사용하는 정적 템플릿 추적기에 가깝고 외형 변화의 LDS 모델을 고려하지 않아 성능이 낮다는 한계가 있다.

본 논문은 이러한 한계를 극복하기 위해 외형의 시간적 역동성을 명시적으로 모델링하는 통합 프레임워크를 제안하며, 이는 기존의 정적 템플릿 추적(SSD, kernel-based tracking)을 일반화한 형태가 된다.

## 🛠️ Methodology

### 1. Linear Dynamical Systems (LDS) 모델링
Dynamic Template의 외형 진화는 다음과 같은 LDS 방정식으로 표현된다.

$$x_t = Ax_{t-1} + Bv_t$$
$$I_t = \mu + Cx_t$$

여기서 $x_t$는 시스템의 잠재 상태(latent state), $I_t$는 템플릿의 픽셀 강도 벡터, $\mu$는 평균 템플릿 이미지이다. $A$는 상태 전이 행렬(state transition matrix), $C$는 관측 행렬(observation matrix)이며, $Bv_t$는 가우시안 프로세스 노이즈를 나타낸다. 학습 단계에서는 SVD(Singular Value Decomposition)와 최소제곱법(least-squares)을 통해 $\mu, A, C, B$를 추정한다.

### 2. 추적 문제의 정식화 (MAP Estimation)
추적 문제는 현재 프레임의 관측값과 이전 상태가 주어졌을 때, 위치 $l_t$와 상태 $x_t$의 사후 확률을 최대화하는 문제로 정의된다. 이를 위해 다음과 같은 목적 함수 $O(l_t, x_t)$를 최소화한다.

$$O(l_t, x_t) = \frac{1}{2\sigma_H^2} \|\sqrt{\rho(y_t(l_t))} - \sqrt{\rho(\mu + Cx_t)}\|^2 + \frac{1}{2}(x_t - A\hat{x}_{t-1})^T Q^{-1} (x_t - A\hat{x}_{t-1})$$

- **재구성 항 (Reconstruction Term)**: 관측된 커널 가중 히스토그램 $\rho(y_t(l_t))$와 모델에 의해 예측된 히스토그램 $\rho(\mu + Cx_t)$ 사이의 Matusita 거리(제곱근 차이의 L2 노름)를 측정한다.
- **역동성 항 (Dynamics Term)**: 현재 상태 $x_t$가 이전 상태 $\hat{x}_{t-1}$로부터 예측된 상태 $A\hat{x}_{t-1}$와 얼마나 일치하는지를 측정한다.

### 3. 미분 가능한 히스토그램 근사 및 최적화
히스토그램의 빈(bin) 할당 과정에 사용되는 Kronecker delta 함수는 미분이 불가능하다. 이를 해결하기 위해 본 논문은 시그모이드 함수 $\phi$를 이용한 연속 함수 $\zeta$를 제안한다.

$$\zeta_u(y) = \frac{1}{\kappa} \sum_{z \in \Omega} K(z) (\phi_{u-1}(y(z)) - \phi_u(y(z)))$$

여기서 $\phi_u(s) = (1 + \exp\{-\sigma(s - r(u))\})^{-1}$이다. 이 근사를 통해 목적 함수를 $l_t$와 $x_t$에 대해 미분할 수 있게 되며, 다음과 같은 반복적인 경사 하강법(gradient descent)을 통해 최적의 위치와 상태를 동시에 업데이트한다.

$$\begin{bmatrix} l_{t}^{i+1} \\ x_{t}^{i+1} \end{bmatrix} = \begin{bmatrix} l_{t}^{i} \\ x_{t}^{i} \end{bmatrix} - 2\gamma \begin{bmatrix} L^T a - M^T a + d \end{bmatrix}$$
(상세한 $L, M, a, d$의 정의는 논문의 Appendix A에 기술되어 있으며, 이는 히스토그램의 기울기와 LDS의 상태 오차를 반영한다.)

### 4. 불변성(Invariance) 확보
학습 데이터와 테스트 데이터 간의 크기(scale), 방향(orientation), 이동 방향의 차이를 극복하기 위해, 학습된 $\mu$와 $C$를 테스트 패치의 특성에 맞게 변환(transformation)한다. 구체적으로, 평균 이미지 $\mu_{im}$과 기저 이미지 $C_{im}$에 대해 bilinear interpolation 등을 적용하여 크기를 조정하거나, 좌우 반전 등을 통해 이동 방향의 차이를 보정한다.

### 5. 동시 추적 및 인식 (Tracking and Recognition)
본 프레임워크는 생성 모델(generative model)이므로, 여러 클래스의 LDS 모델을 사용하여 동시에 추적과 인식을 수행할 수 있다.
- **DK-SSD-TR-R**: 각 클래스 모델로 추적을 수행한 후, 평균 목적 함수 값(재구성 비용)이 가장 낮은 클래스로 인식한다.
- **DK-SSD-TR-C**: 추적된 경로를 통해 테스트 시스템의 파라미터를 학습하고, 이를 학습 데이터의 모델들과 Martin distance(관측 가능 부분공간 사이의 각도 기반 거리)로 비교하여 최적의 클래스를 결정한다.

## 📊 Results

### 1. 상태 수렴성 평가 (Synthetic Data)
가상으로 생성된 Dynamic Texture를 통해 상태 추정 오차를 측정한 결과, 제안 방법(DK-SSD-T)은 EKF(Extended Kalman Filter)나 PF(Particle Filter)보다 훨씬 낮은 오차를 유지하며, 초기 상태 설정이 잘못된 경우에도 빠르게 실제 상태로 수렴함을 확인하였다.

### 2. Dynamic Texture 추적 (Real Data)
- **대상**: 촛불(Candle Flame), 깃발(Flags), 불(Fire) 등의 실제 비디오.
- **비교 대상**: Meanshift(MS), Online Boosting, DT-PF 등.
- **결과**: 깃발과 같이 외형 변화가 심한 객체에서 제안 방법이 가장 정확한 경로를 추적하였다. 특히, 배경 정보 없이 전경의 역동성만을 이용했음에도 불구하고 배경 정보를 활용하는 MS-VR 등의 방법과 대등하거나 더 우수한 성능을 보였다.

### 3. 인간 동작 추적 및 인식 (Human Actions)
- **특징**: 픽셀 강도 대신 Optical Flow를 LDS의 입력값으로 사용하여 동작 특유의 역동성을 모델링하였다.
- **추적 성능**: Walking, Running 동작에 대해 정밀한 추적 성능을 보였으며, 배경 제거(background subtraction) 없이도 기존 추적기들보다 월등한 성능을 기록하였다.
- **인식 성능**: 
    - Walking/Running 데이터셋에서 DK-SSD-TR-C 방식이 96%의 인식률을 기록하였다.
    - Weizmann Action 데이터셋(10개 동작)에서도 92.47%의 높은 인식률을 보였으며, 대부분의 클래스에서 중앙값 픽셀 오차가 6픽셀 미만으로 매우 정확하게 추적되었다.
    - 한 데이터셋에서 학습한 모델을 다른 데이터셋(Weizmann)에 적용했을 때도 100%의 인식률과 높은 추적 정확도를 보여 일반화 성능을 입증하였다.

## 🧠 Insights & Discussion

### 강점
- **역동성의 명시적 활용**: 단순한 외형 일치가 아니라, 외형이 시간에 따라 어떻게 변하는지에 대한 '규칙(LDS)'을 추적에 활용함으로써 비정형 객체 추적의 강건성을 크게 높였다.
- **범용적 프레임워크**: 강도(intensity)뿐만 아니라 Optical Flow와 같은 임의의 이미지 특징량에 적용 가능하여, 단순 텍스처 추적에서 복잡한 인간 동작 추적까지 확장 가능하다.
- **동시 최적화의 이점**: 위치와 상태를 동시에 추정함으로써, 추적 결과가 인식 결과에 영향을 주고, 인식된 클래스의 모델이 다시 추적의 정확도를 높이는 시너지 효과를 거두었다.

### 한계 및 비판적 해석
- **계산 복잡도**: EKF나 PF와 달리 매 프레임마다 목적 함수를 최소화하기 위해 수십 번의 반복적인 경사 하강법 연산을 수행해야 하므로 계산 비용이 높다.
- **지역 최적해(Local Minima) 문제**: 목적 함수가 비볼록(non-convex) 함수이므로, 초기화 값에 따라 지역 최적해에 빠질 위험이 있다. 논문에서는 이전 프레임의 결과와 pseudo-inverse를 통한 초기화로 이를 완화했으나, 이론적인 전역 최적성(global optimality)은 보장되지 않는다.
- **LDS의 선형성 가정**: 외형 변화를 선형 시스템(LDS)으로 모델링하였으나, 실제 세상의 복잡한 비정형 변화는 비선형적일 가능성이 크다. 향후 비선형 동역학 모델의 도입이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 비정형 객체의 외형 및 움직임 변화를 **Linear Dynamical Systems (LDS)**로 모델링하여, 객체의 **위치와 내부 상태를 동시에 추정**하는 새로운 추적 프레임워크를 제안한다. 미분 가능한 히스토그램 근사 기법을 통해 효율적인 최적화를 구현하였으며, 이를 통해 Dynamic Textures 및 인간 동작에 대해 기존의 정적 템플릿 기반 추적기보다 월등한 성능을 입증하였다. 특히 추적과 인식을 동시에 수행하는 기능을 통해 높은 인식 정확도를 달성하였으며, 이는 향후 비디오 분석 및 객체 인식 분야에서 실용적인 기반 기술이 될 가능성이 높다.