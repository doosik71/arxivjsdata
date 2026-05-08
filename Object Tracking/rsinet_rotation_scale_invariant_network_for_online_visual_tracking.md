# RSINet: Rotation-Scale Invariant Network for Online Visual Tracking

Yang Fang, Geun-Sik Jo, Chang-Hee Lee (2020)

## 🧩 Problem to Solve

본 논문은 온라인 비주얼 트래킹(Online Visual Tracking)에서 기존 Siamese 네트워크 기반 트래커들이 가진 두 가지 주요 한계점을 해결하고자 한다.

첫째, 대부분의 Siamese 기반 트래커는 모델 업데이트 없이 추적을 수행하므로, 추적 대상(target)의 특성에 따른 변화에 적응적으로 대응하지 못한다.

둘째, 기존 트래커들은 새로운 객체 상태를 추론할 때 축 정렬 바운딩 박스(axis-aligned bounding boxes)를 생성한다. 이는 객체의 실제 회전이나 크기 변화를 정확히 추정하지 못하며, 결과적으로 바운딩 박스 내에 불필요한 배경 노이즈가 포함되어 트래킹 성능을 저하시키는 원인이 된다.

따라서 본 연구의 목표는 객체의 회전(Rotation)과 크기(Scale) 변화에 불변하는 특성을 학습하고, 이를 실시간으로 적응적으로 업데이트할 수 있는 Rotation-Scale Invariant Network(RSINet)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대상-방해물 판별(Target-Distractor Discrimination)과 회전-크기 추정(Rotation-Scale Estimation)을 동시에 수행하는 멀티태스크 학습 프레임워크를 구축하는 것이다.

구체적인 기여 사항은 다음과 같다.

1. **이중 브랜치 구조 제안**: Target-Distractor Discrimination (TDD) 브랜치와 Rotation-Scale Invariant (RSI) 브랜치로 구성된 통합 프레임워크를 제안하여, 기존 SOTA Siamese 트래커보다 정밀한 타겟 표현을 생성한다.
2. **Log-Polar 좌표계 기반 학습**: 타겟의 위치, 회전, 크기 변화를 자연스럽게 예측하기 위해 Cartesian 좌표계와 Log-Polar 좌표계에서 공유된 특징 표현을 학습하는 방식을 도입하였다.
3. **시공간 에너지 제어 기반 적응적 업데이트**: 모델의 안정성과 신뢰성을 확보하기 위해 Spatio-temporal energy control 하에서 모델을 적응적으로 최적화하고 업데이트하는 방식을 제안하였다.
4. **실시간 성능 입증**: OTB-100, VOT2018, LaSOT 벤치마크에서 최신 트래커들과 비교하여 우수한 성능을 보였으며, 약 45 FPS의 실시간 속도로 동작함을 입증하였다.

## 📎 Related Works

논문에서는 기존 연구를 세 가지 범주로 나누어 설명하며 각각의 한계를 지적한다.

1. **Offline Siamese Trackers**: SiamRPN, SiamRPN++, DaSiamRPN 등이 이에 해당한다. 이들은 빠른 속도를 자랑하지만, 대부분 모델 파라미터를 고정(freeze)한 채 추적하므로 타겟 특화 지식을 온라인으로 학습하지 못하며, 특히 회전 정보 학습이 부족하다. SiamMask의 경우 회전 바운딩 박스를 생성하지만, 마스크 브랜치에 의존하며 최적화 비용이 커 속도가 느리다는 단점이 있다.
2. **Online Deep Trackers**: ATOM, DiMP와 같은 트래커들은 온라인 학습을 통해 타겟 특화 모델을 구축한다. 하지만 이들 역시 회전 및 크기 정보를 직접적으로 학습하지 않는다는 한계가 있다.
3. **Log-Polar coordinate based Trackers**: SRCF와 같이 Log-Polar 좌표계를 사용하여 회전과 크기를 명시적으로 학습하려는 시도가 있었다. 그러나 이들은 주로 HOG와 같은 수작업 특징(hand-crafted features)을 사용하므로 타겟 표현력과 판별 모델 학습 능력에 한계가 있다.

RSINet은 이러한 한계를 극복하기 위해 Siamese 기반의 CNN을 통해 특징을 추출하되, Cartesian과 Log-Polar라는 서로 다른 샘플링 공간의 데이터를 동시에 사용하여 end-to-end 방식으로 학습하는 최초의 연구임을 강조한다.

## 🛠️ Methodology

### 전체 시스템 구조

RSINet은 공유된 ResNet-50 백본 네트워크를 기반으로 하며, 두 개의 주요 모듈로 구성된다.

- **TDD (Target-Distractor Discrimination) 모듈**: 타겟의 중심 위치를 추정하고 배경 방해물로부터 타겟을 판별한다.
- **RSI (Rotation-Scale Invariance) 모듈**: 타겟의 회전 각도와 크기 변화를 명시적으로 추정한다.

### Rotation-Scale Invariance (RSI) Module

RSI 모듈은 Cartesian 좌표계의 이미지를 Log-Polar (LP) 좌표계로 변환하여 회전과 크기 변화를 단순한 평행 이동(translation) 문제로 변환한다.

이미지 $I(x, y)$를 Log-Polar 좌표 $\hat{I}(\rho, \theta)$로 변환하면, 타겟의 회전 $\theta$와 크기 변화 $\rho$는 다음과 같은 관계식으로 표현된다.
$$I_{lp}^{t+1}(\rho, \theta) = I_{lp}^t(\rho - \Delta\rho, \theta - \Delta\theta)$$

이 모듈은 사전 학습된 ResNet-50으로 특징 맵을 추출하고, 3층의 Fully Convolutional Network (FCN) 회귀기를 통해 $(\rho^?, \theta^?)$를 예측한다. 예측 함수는 다음과 같다.
$$f(I_{lp}, h) = \psi_3(h_3 * \psi_2(h_2 * \psi_1(h_1 * I_{lp}))) = (\rho^?, \theta^?)$$

손실 함수 $L_{rs}$는 예측값과 정답(ground-truth) 간의 정규화된 잔차(residual)를 이용한 L2 손실을 사용한다.
$$L_{rs}(h) = \sum_{i=1}^N \|R(f(I_{lp}^i, h), g_i)\|^2 + \sum_j \lambda_j \|h_i\|^2$$
여기서 잔차 $R$은 다음과 같이 정의된다.
$$R(f(I_{lp}^i, h), g_i) = \left( \frac{\rho^? - \rho}{\rho}, \frac{\theta^? - \theta}{\theta} \right)$$

### Target-Distractor Discrimination (TDD) Module

TDD 모듈은 타겟과 주변 배경의 판별력을 높이는 것을 목표로 한다. 가우시안 함수를 통해 타겟 중심은 높은 값을, 배경은 0에 가까운 값을 갖는 라벨 $y$를 생성하고, 최소제곱 회귀 기반의 손실 함수 $L_{td}$를 사용한다.
$$L_{td}(w) = \frac{1}{N} \sum_{(x,y) \in S} \|s(x, w) - y\|^2 + \|\gamma * w\|^2$$

단, 단순한 $L_{td}$는 불균형한 데이터 분포로 인해 음성 샘플(background)을 0으로 만드는 데 치중하는 경향이 있다. 이를 해결하기 위해 SVM의 전략을 차용하여 다음과 같은 Hinge-like score map 공식을 적용한다.
$$s(x, w) = m \cdot (x * w) + (1 - m) \cdot \max(0, x * w)$$

### 최종 학습 및 업데이트 절차

전체 네트워크는 두 손실 함수의 가중 합으로 학습된다.
$$L = L_{rs} + \mu L_{td}$$
여기서 $\mu = 50$으로 설정하여 타겟 위치 추정의 중요도를 높였다.

온라인 추적 단계에서는 **Spatio-temporal energy ($\epsilon$)**를 통해 모델 업데이트 여부를 결정한다.
$$\epsilon = \frac{y_{max} - \mu_s}{\sigma_s} \times \frac{y_{max} - \mu_t}{\sigma_t}$$
($\mu_s, \sigma_s$: 현재 프레임 스코어 맵의 사이드로브 평균 및 표준편차 / $\mu_t, \sigma_t$: 이전 $H$개 프레임 최대 스코어의 평균 및 표준편차)

업데이트 조건 $\epsilon \ge \kappa \epsilon_0$ (여기서 $\kappa = 0.8$)를 만족할 때만 모델을 업데이트하며, 업데이트 속도 $\alpha$는 다음과 같이 결정된다.
$$\alpha = \min \left( \frac{1}{\epsilon}, \alpha_s \right)$$
여기서 $\alpha_s$는 steepest gradient descent 기반의 업데이트 속도이다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB-100, VOT2018, LaSOT
- **비교 대상**: ECO, SiamRPN, SiamRPN++, DaSiamRPN, TADT, ASRCF, DWSiamese, ATOM, DiMP
- **측정 지표**: Precision rate, Success rate, EAO(Expected Average Overlap), Accuracy(A), Robustness(R)
- **환경**: PyTorch, Intel Core i7-6700, GTX TITAN X 2장, 약 45 FPS 동작

### 주요 결과

1. **OTB-100**: Success rate에서 0.697을 기록하며 가장 높은 성능을 보였으며, Precision rate 또한 최상위권(0.829)에 위치하였다. 특히 Scale Variation(SV)과 In-plane Rotation(IPR) 속성에서 각각 0.843, 0.856의 정밀도를 기록하며 타 트래커를 압도하였다.
2. **VOT2018**: Accuracy(A)에서 0.604를 기록하여 SOTA를 달성하였다. EAO는 0.435로 DiMP(0.440)와 유사한 수준을 유지하였다.
3. **LaSOT**: Success rate에서 0.585를 기록하여 DiMP(0.575)보다 약간 우수한 성능을 보였다.
4. **Ablation Study**: TDD 단독 모델보다 RSI 모듈을 추가했을 때 Precision이 2.6% 향상되었으며, 여기에 제안한 적응적 경사 하강법(AGD)을 적용했을 때 성능이 더욱 최적화됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 기존 Siamese 트래커들이 간과했던 **회전(Rotation)과 크기(Scale)의 명시적 추정**을 Log-Polar 좌표계와 딥러닝 회귀기를 통해 성공적으로 구현하였다. 특히 단순히 구조를 제안한 것에 그치지 않고, 시공간 에너지를 이용한 적응적 업데이트 메커니즘을 통해 모델의 안정성과 실시간성(45 FPS)을 동시에 확보한 점이 높게 평가된다.

### 한계 및 비판적 해석

실험 결과에서 언급되었듯이, VOT2018의 Robustness(R) 지표가 DiMP보다 낮게 나타났다. 이는 RSINet이 단기적인 외관 변화(회전, 크기)에는 강하지만, 객체가 완전히 사라졌다가 다시 나타나는 **Long-term tracking의 재식별(Re-identification) 능력**은 부족함을 시사한다. 또한 $\mu=50$과 같은 하이퍼파라미터 설정이 경험적으로 이루어졌으므로, 다양한 환경에서의 일반화 성능에 대한 추가 검증이 필요해 보인다.

## 📌 TL;DR

RSINet은 Siamese 네트워크에 **Log-Polar 기반의 회전-크기 추정 브랜치**와 **시공간 에너지 제어 기반의 적응적 업데이트** 방식을 도입한 온라인 트래커이다. 이를 통해 객체의 회전 및 크기 변화에 매우 강건한 추적 성능을 보이며, 45 FPS의 실시간 속도로 동작한다. 향후 장기 추적을 위한 재식별 능력 보완이 이루어진다면 더욱 강력한 트래커가 될 가능성이 높다.
