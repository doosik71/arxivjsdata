# Towards a Better Match in Siamese Network Based Visual Object Tracker

Anfeng He, Chong Luo, Xinmei Tian, and Wenjun Zeng (2018)

## 🧩 Problem to Solve

본 논문은 Siamese network 기반의 비주얼 객체 추적(Visual Object Tracking) 프레임워크가 가진 두 가지 주요 한계점을 해결하고자 한다.

첫째, 기존의 Siamese 네트워크는 이미지의 스케일 변화에는 어느 정도 대응하지만, 객체의 큰 회전(Rotation)이 발생했을 때 이를 적절히 처리하지 못한다. 이는 CNN 특징점(Feature)들이 회전과 같은 큰 이미지 변환에 대해 불변성(Invariance)을 가지지 않기 때문이다. 특히 추적 대상이 정사각형이 아닐 때, 방향이나 종횡비(Aspect Ratio)를 조정하는 메커니즘이 없어 성능이 크게 저하된다.

둘째, 배경에 추적 대상과 유사하거나 눈에 띄는(Salient) 객체가 존재할 경우 추적기가 쉽게 혼동되는 문제가 발생한다. 일반적으로 주변 컨텍스트 정보를 포함하는 것이 추적에 도움이 되지만, 배경에 방해 요소가 많을 경우 너무 많은 컨텍스트 정보는 오히려 독이 된다.

따라서 본 논문의 목표는 실시간 성능을 유지하면서도 객체의 회전에 강인하고 배경 노이즈를 효과적으로 억제하여, 서로 다른 프레임 간의 동일 객체에 대해 더 나은 매칭(Better Match)을 달성하는 것인 $\text{Siam-BM}$ 추적기를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 계산 오버헤드를 최소화하면서 특징 표현력을 높이는 두 가지 단순하지만 효과적인 메커니즘을 도입하는 것이다.

1. **각도 추정(Angle Estimation):** 객체의 회전을 처리하기 위해 여러 각도 옵션을 탐색하되, 계산량을 줄이기 위해 스케일과 각도를 동시에 변경하는 대신 한 번에 하나씩만 조정하는 전략을 사용한다.
2. **공간 마스킹(Spatial Masking):** 객체의 종횡비가 1:1에서 멀어질 때(가늘고 긴 형태일 때) 배경의 방해 요소가 포함될 가능성이 높다는 점에 착안하여, 특정 임계값 이상의 종횡비를 가진 객체에 대해 선택적으로 공간 마스크를 적용하여 특징 추출 영역을 제한한다.
3. **템플릿 업데이트(Template Updating):** 객체의 점진적인 외형 변화에 대응하기 위해 이동 평균(Moving Average) 기반의 템플릿 업데이트 메커니즘을 적용한다.

## 📎 Related Works

본 연구는 $\text{SiamFC}$를 기반으로 하며, 특히 $\text{SA-Siam}$의 구조를 계승한다. $\text{SiamFC}$는 동일한 CNN을 통해 타겟 패치와 검색 영역의 특징을 추출하고 상관관계(Correlation)를 통해 위치를 찾는 구조로, 온라인 학습이 거의 없어 매우 빠르다는 장점이 있다.

이후 $\text{SiamRPN}$은 영역 제안 네트워크(RPN)를 통해 종횡비를 추정했고, $\text{SA-Siam}$은 외형 특징 외에 세만틱 특징(Semantic Feature)을 추가하고 채널 주의 집중(Channel-wise Attention) 메커니즘을 도입했다.

기존의 회전 대응 방식인 $\text{RAJSSC}$ 등은 Log-Polar 변환을 사용했지만, Siamese 네트워크 기반의 추적기에서 각도 추정을 직접적으로 다룬 사례는 드물었다. 또한 배경 노이즈 억제를 위해 $\text{RASNet}$ 등이 주의 집중 메커니즘을 사용했으나, 본 논문은 학습 기반의 복잡한 주의 집중 모듈 대신 단순하고 안정적인 고정 공간 마스크(Fixed Spatial Mask)를 제안하며 차별점을 둔다.

## 🛠️ Methodology

$\text{Siam-BM}$은 $\text{SA-Siam}$을 기반으로 구축되었으며, 외형 브랜치(Appearance branch)와 세만틱 브랜치(Semantic branch)를 모두 사용하여 특징을 추출한다.

### 1. Angle Estimation

전통적인 $\text{SiamFC}$는 스케일 변화만 고려하여 $M$개의 후보 패치를 생성한다. 만약 $N$개의 각도 옵션을 추가한다면 후보 패치는 $M \times N$개가 되어 연산량이 급증한다. $\text{Siam-BM}$은 이를 해결하기 위해 스케일($s$)이나 각도($a$) 중 하나만 변경하는 전략을 취한다.

즉, $a \neq 0$이면 $s=1$로 고정하고, $s \neq 1$이면 $a=0$으로 고정한다. 이로 인해 후보 패치의 수는 $M \times N$에서 $M+N-1$로 줄어든다. 본 구현에서는 $M=N=3$으로 설정하여 총 5개의 후보 패치 $(s, a) \in \{(1.0375, 0), (0.964, 0), (1, 0), (1, \pi/8), (1, -\pi/8)\}$를 사용한다.

최종 추적 위치와 상태는 다음과 같이 결정된다:
$$(x_i, y_i, k_i) = \arg \max_{x,y,k} R_k, \quad (k= 1, 2, \dots, K)$$
여기서 $K=M+N-1$이며, $k_i$는 선택된 스케일과 각도 쌍을 의미한다.

### 2. Spatial Mask

객체의 종횡비 $r = \max(h/w, w/h)$가 임계값 $thr = 1.5$를 초과할 경우, 해당 객체를 'elongated object'로 판단하고 공간 마스크를 적용한다. 마스크는 $\text{conv4}$($8 \times 8$)와 $\text{conv5}$($6 \times 6$) 특징 맵에 적용되며, 타겟의 형태(가로형 또는 세로형)에 따라 미리 정의된 고정 마스크(White: 1, Black: 0)를 곱하여 배경 정보를 제거한다. 이 마스크는 특히 세만틱 브랜치에만 적용하는데, 이는 세만틱 응답이 외형 응답보다 더 희소하고 중심에 집중되어 있어 마스크로 인한 정보 손실 위험이 적기 때문이다.

### 3. Template Updating

객체의 외형 변화에 대응하기 위해 다음과 같은 템플릿 업데이트 식을 사용한다:
$$\phi(T_t) = \lambda^S \times \phi(T_1) + (1 - \lambda^S) \times \phi(T^u_t)$$
$$\phi(T^u_t) = (1 - \lambda^U) \times \phi(T^u_{t-1}) + \lambda^U \times \hat{\phi}(T_{t-1})$$
여기서 $\phi(T_1)$은 첫 프레임의 특징, $\phi(T^u_t)$는 업데이트된 특징의 이동 평균, $\hat{\phi}(T_{t-1})$은 이전 프레임에서 추적된 객체의 특징이다. 설정값은 $\lambda^S = 0.5, \lambda^U = 0.006$이다.

## 📊 Results

### 실험 설정

- **데이터셋:** OTB-2013, OTB-100, VOT2017.
- **지표:** $\text{EAO}$ (Expected Average Overlap), $\text{Accuracy}$, $\text{Robustness}$, $\text{AUC}$ (Success rate), $\text{Precision}$.
- **환경:** Tesla P100 GPU, TensorFlow 1.7.0.

### 주요 결과

- **VOT2017 결과:** $\text{Siam-BM}$은 $\text{EAO}$ 0.335를 기록하며 당시 실시간 추적기 중 최고의 성능을 보였다.
- **Ablation Study (각도 추정):** $\text{SA-Siam}$ 베이스라인(EAO 0.287) 대비 각도 추정만 추가했을 때 EAO가 0.301로 상승하여, 회전 대응의 중요성을 입증했다.
- **Ablation Study (공간 마스크):** OTB 데이터셋 실험 결과, 종횡비가 큰 객체(elongated objects)에서 공간 마스크 적용 시 성능이 유의미하게 향상되었다. 특히 훈련 단계에서 마스크를 적용했을 때 테스트 단계의 성능까지 함께 올라가는 경향이 확인되었다.
- **종합 성능 향상:** $\text{SA-Siam} \rightarrow \text{Angle Estimation} \rightarrow \text{Spatial Mask} \rightarrow \text{Template Updating}$ 순으로 기능을 추가함에 따라 EAO가 $0.287 \rightarrow 0.301 \rightarrow 0.322 \rightarrow 0.335$로 점진적으로 향상되었다.
- **속도:** 모든 기능을 포함한 최종 모델은 32 fps로 동작하여 실시간성을 유지하였다.

## 🧠 Insights & Discussion

본 논문은 복잡한 딥러닝 모듈을 추가하는 대신, 추적기의 동작 원리를 분석하여 효율적인 제약 조건(Angle-Scale 분리 탐색, 고정 마스크)을 도입함으로써 성능과 속도라는 두 마리 토끼를 잡았다.

특히 주목할 점은 객체의 종횡비와 공간 마스킹의 이득 사이의 양의 상관관계이다. 객체가 정사각형에서 멀어질수록 배경에 포함되는 노이즈가 많아지므로, 단순한 마스킹만으로도 큰 성능 향상을 얻을 수 있다는 점을 정량적으로 보여주었다.

한계점으로는 공간 마스크가 고정된 형태라는 점이다. 논문에서도 언급되었듯이, 추적 과정 중에 객체의 종횡비가 동적으로 변하는 상황에 대한 적응(Adaptation)은 향후 연구 과제로 남아 있다. 또한, $\text{Siam-BM}$이 VOT2017에서 매우 높은 성적을 거두었으나, 이는 주로 실시간 제약 조건 하에서의 성과이며, 연산량 제한이 없는 non-realtime 추적기들과의 격차를 어떻게 줄일 것인가에 대한 논의가 더 필요하다.

## 📌 TL;DR

$\text{Siam-BM}$은 Siamese 네트워크 기반 추적기의 고질적 문제인 **객체 회전 대응 불가**와 **배경 노이즈 취약성**을 해결하기 위해 **효율적인 각도 추정 전략**과 **종횡비 기반 공간 마스킹**을 제안하였다. 이를 통해 실시간성(32 fps)을 유지하면서도 $\text{VOT2017}$ 실시간 챌린지에서 우승하는 등 압도적인 성능 향상을 이루어냈으며, 이는 향후 실시간 객체 추적 연구에서 효율적인 특징 정제(Feature Refinement)의 중요성을 시사한다.
