# Rotation Equivariant Siamese Networks for Tracking

Deepak K. Gupta, Devanshu Arya, and Efstratios Gavves (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 시각적 객체 추적(Visual Object Tracking)에서 발생하는 **평면 내 회전(In-plane rotation)** 문제이다.

기존의 딥러닝 기반 추적 알고리즘들은 일반적인 CNN을 사용하며, 이는 기본적으로 평행 이동 등가성(Translation Equivariance)을 가지지만 회전 변환에 대해서는 등가성을 보장하지 않는다. 이로 인해 추적 대상이 회전할 경우, 네트워크가 학습 단계에서 보지 못한 방향의 특징을 추출하게 되어 성능이 급격히 저하된다. 

특히 드론 촬영 영상이나 Top-view 영상, 또는 에고센트릭(Egocentric) 비디오와 같이 카메라나 객체가 빈번하게 회전하는 환경에서 이 문제는 매우 중요하다. 기존의 데이터 증강(Data Augmentation) 방식은 모든 회전 변형에 대해 별도의 표현을 학습해야 하므로 계산 비용이 증가하고, 모델이 회전 불변성(Rotation Invariance)을 갖게 될 경우 주변에 유사한 객체가 많을 때(예: 물고기 떼 속의 물고기 추적) 변별력이 떨어지는 한계가 있다. 따라서 본 논문의 목표는 아키텍처 수준에서 **회전 등가성(Rotation Equivariance)**을 내장하여 추가적인 데이터 증강 없이도 회전 변화에 강건한 Siamese 네트워크(RE-SiamNet)를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Group-Equivariant CNN**과 **Steerable Filters**를 Siamese 네트워크에 결합하여, 입력 이미지의 회전이 특징 맵(Feature Map)의 회전으로 직접 연결되도록 설계하는 것이다. 주요 기여 사항은 다음과 같다.

1. **RE-SiamNets 설계**: Steerable filters를 사용하여 평면 내 회전에 대해 등가성을 갖는 Siamese 추적 구조를 제안하였다.
2. **비지도 상대 포즈 추정**: RE-SiamNet의 구조적 특성을 이용하여, 정답 라벨 없이도 템플릿 대비 타겟 객체의 상대적인 2D 회전 각도를 추정할 수 있음을 보였다.
3. **회전 운동 제약 조건(Rotational Motion Constraint)**: 연속된 프레임 사이의 회전 변화량에 제한을 두는 제약 조건을 도입하여 시간적 일관성(Temporal Correspondence)을 향상시켰다.
4. **ROB(Rotating Object Benchmark) 데이터셋**: 기존 데이터셋에 부족한 회전 사례를 집중적으로 포함한 새로운 벤치마크 데이터셋을 구축하여 제공하였다.

## 📎 Related Works

### 기존 Siamese 추적기
SiamFC, SiamRPN++와 같은 기존 Siamese 추적기들은 템플릿 이미지와 검색 영역 간의 유사도를 측정하여 객체를 위치시킨다. 이들은 강력한 변별력을 가지지만, 부분 폐색(Partial Occlusion), 스케일 변화, 그리고 특히 객체의 회전 상황에서 취약함을 보인다.

### 등가성 CNN (Equivariant CNNs)
최근 이미지 분류 및 세그멘테이션 분야에서 변환 등가성을 네트워크 구조에 직접 반영하려는 연구가 진행되었다. Cohen과 Welling의 Group-Convolutional layers가 대표적이며, 이후 계산 효율성을 높이기 위해 Steerable filters가 도입되었다. 본 논문은 이러한 이론을 객체 추적이라는 로컬라이제이션 작업에 처음으로 적용하였다.

### 기존 접근 방식과의 차별점
기존의 회전 대응 방식이 주로 데이터 증강을 통한 '학습'에 의존했다면, RE-SiamNet은 **가중치 공유(Weight Sharing)**를 통해 수학적으로 회전 등가성을 보장하는 '구조'를 채택함으로써 학습 효율성을 높이고 모델의 일반화 성능을 개선하였다.

## 🛠️ Methodology

### 1. 회전 등가성 및 Steerable Filters
함수 $f$가 변환 그룹 $G$에 대해 등가성을 갖는다는 것은 다음과 같이 정의된다.
$$f(\psi^Y_g(x)) = \psi^Y_g(f(x)), \quad g \in G, x \in X$$
여기서 $\psi$는 해당 공간에서의 그룹 작용(Group Action)을 의미한다. 본 논문은 **Circular Harmonics** $\psi_{jk}$를 기본 함수로 사용하는 Steerable filters를 도입한다.
$$\psi_{jk}(r, \phi) = \tau_j(r)e^{ik\phi}$$
임의의 각도 $\theta$로 회전된 필터 $\rho_\theta \Psi(x)$는 다음과 같이 위상 조작(Phase Manipulation)을 통해 효율적으로 계산될 수 있다.
$$\rho_\theta \Psi(x) = \sum_{j=1}^J \sum_{k=0}^K w_{jk} e^{-ik\theta} \psi_{jk}(x)$$

### 2. RE-SiamNet 아키텍처
전체 파이프라인은 기존 SiamFC 구조를 기반으로 하되, 모든 일반 컨볼루션 층을 회전 등가 모듈로 교체한다.

- **Rotation Equivariant Input**: 템플릿 헤드는 단일 이미지가 아니라, $\Lambda$개의 이산적인 회전 변형 세트 $Z = \{z_1, z_2, \dots, z_\Lambda\}$를 입력으로 받는다. 이는 추론 단계에서 미리 계산 가능하다.
- **Rotation Equivariant Convolutions**: 그룹 컨볼루션을 통해 공간적 회전과 특징 맵의 회전을 동시에 처리한다.
- **Rotation Equivariant Cross-Correlation**: 템플릿의 $\Lambda$개 회전 버전과 검색 영역 특징 맵을 각각 컨볼루션하여 $\Lambda$개의 히트맵 $\{ \hat{h}(z, x) \}$을 생성한다.
- **Group Max Pooling**: 생성된 $\Lambda$개의 히트맵 중 최댓값을 가진 맵을 선택하는 Global Max Pooling을 수행하여 최종 히트맵 $h(Z, x)$를 도출한다.

### 3. 비지도 상대 회전 추정 및 운동 제약
RE-SiamNet은 Group Max Pooling 단계에서 어떤 인덱스 $i$의 히트맵이 선택되었는지를 통해 상대적 포즈 변화를 추정한다.
$$h(Z, x) = \hat{h}(z_i, x) = \text{group-maxpool}(\{h(z, x)\})$$
이때 상대 회전 각도는 $\theta_{\text{diff}} = i \cdot 360 / \Lambda$로 계산된다.

또한, 시간적 일관성을 위해 프레임 $t$에서 선택된 최적 각도 $\theta_{t, \text{opt}}$를 기준으로, 프레임 $t+1$에서는 $\pm \gamma$ 범위 내의 각도만 후보군으로 두는 **회전 운동 제약 조건**을 적용하여 추적의 안정성을 높였다.

## 📊 Results

### 실험 설정
- **데이터셋**: ROB(신규), Rot-OTB100, Rot-MNIST, OTB100, GOT-10k.
- **비교 대상**: SiamFC, SiamFCv2, SiamRPN++, DiMP(SOTA).
- **지표**: Precision(정밀도), Success Rate(성공률).
- **구현**: $\Lambda \in \{4, 8, 16\}$의 회전 그룹을 사용하였으며, `e2cnn` 라이브러리를 통해 구현하였다.

### 주요 결과
1. **회전 추적 성능**: Rot-OTB100 실험에서 일반 SiamFC는 회전이 추가되었을 때 정밀도가 24.2%, 성공률이 26.3% 하락하였다. 반면, RE-SiamNet은 $\Lambda=4$만으로도 기존 모델을 크게 상회하였으며, $\Lambda=8$ 이상에서는 SOTA 모델인 DiMP와 경쟁 가능한 수준의 성능을 보였다.
2. **상대 포즈 추정**: ROB와 Rot-OTB100 데이터셋에서 실제 회전 각도와 예측 각도 사이의 오차가 $\pm \pi/4$ 범위 내에 있을 확률(SR)이 평균 60% 이상으로 나타나, 비지도 방식으로도 유의미한 포즈 추정이 가능함을 입증하였다.
3. **일반 성능 유지**: 회전이 없는 일반 OTB100 및 GOT-10k 데이터셋에서도 성능 하락이 2% 이내로 매우 적어, 회전 등가성 도입이 일반적인 추적 성능을 저해하지 않음을 확인하였다.
4. **운동 제약 효과**: 회전 운동 제약을 추가했을 때 정밀도와 성공률이 소폭 상승하였으며, 특히 강건성(Robustness) 측면에서 긍정적인 영향이 있었다.

## 🧠 Insights & Discussion

본 논문은 CNN의 구조적 제약으로 인해 발생하는 회전 문제를 데이터 증강이 아닌 **수학적 등가성(Equivariance)**으로 해결하려는 시도가 매우 효과적임을 보여주었다.

**강점**은 가중치 공유를 통해 파라미터 수의 증가 없이 회전 변화에 대응할 수 있다는 점과, 추적과 동시에 상대적 포즈라는 추가 정보를 비지도 방식으로 얻을 수 있다는 점이다.

**한계 및 논의사항**으로는 $\Lambda$(회전 그룹 수)를 높일수록 포즈 추정의 정밀도는 올라가지만, 동일한 파라미터 예산 내에서 채널 수를 줄여야 하므로 일반적인 변별력(Discriminative power)이 소폭 감소하는 트레이드-오프 관계가 존재한다는 점이 언급되었다. 또한 본 연구는 평면 내 회전(In-plane)에 집중하였으며, 3D 공간의 회전(Out-of-plane)을 처리하기 위해서는 3D 씬에 대한 추가 정보가 필요할 것이다.

## 📌 TL;DR

본 논문은 시각적 객체 추적에서 난제로 꼽히는 **평면 내 회전** 문제를 해결하기 위해, Steerable filters 기반의 **회전 등가 Siamese 네트워크(RE-SiamNet)**를 제안하였다. 이 모델은 추가 파라미터 비용 없이 회전 변화에 강건하며, 비지도 방식으로 타겟의 상대적 2D 포즈를 추정할 수 있다. 또한, 회전 특화 벤치마크인 **ROB 데이터셋**을 통해 그 성능을 입증하였으며, 이는 향후 드론 영상 분석이나 로봇 비전 등 회전 변화가 심한 환경의 추적 연구에 중요한 기초가 될 것으로 보인다.