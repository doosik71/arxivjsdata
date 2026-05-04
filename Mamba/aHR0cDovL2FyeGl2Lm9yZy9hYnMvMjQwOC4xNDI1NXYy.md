# MSFMamba: Multi-Scale Feature Fusion State Space Model for Multi-Source Remote Sensing Image Classification

Feng Gao, Xuepeng Jin, Xiaowei Zhou, Junyu Dong, Qian Du (2025)

## 🧩 Problem to Solve

본 논문은 초분광 이미지(Hyperspectral Image, HSI)와 LiDAR 또는 합성 개구 레이더(Synthetic Aperture Radar, SAR) 데이터의 결합 분류(Joint Classification) 문제를 해결하고자 한다.

다중 소스 원격 탐사 데이터 분류에서 해결해야 할 핵심 문제는 다음과 같다.
1. **수용 영역(Receptive Field)의 한계**: CNN 기반 방법은 국소적인 수용 영역으로 인해 장거리 의존성(Long-range dependency)을 포착하는 데 한계가 있다.
2. **계산 복잡도**: Transformer 기반 모델은 전역 수용 영역을 제공하지만, 어텐션(Attention) 메커니즘의 높은 계산 복잡도로 인해 효율성이 떨어진다.
3. **이질성 간극(Heterogeneity Gap)**: HSI, LiDAR, SAR 데이터는 서로 다른 물리적 특성과 분포를 가지므로, 이들 간의 이질적인 특성을 어떻게 효과적으로 융합하여 정렬할 것인가가 중요한 과제이다.
4. **특성 중복(Feature Redundancy)**: 최근 주목받는 State Space Model(SSM) 기반의 Mamba 모델은 다방향 스캔(Multi-scan) 전략을 사용하여 이미지의 비방향성을 해결하지만, 이는 중복된 특성을 반복적으로 추출하여 계산 비용을 증가시키는 문제를 야기한다.

본 논문의 목표는 Mamba의 선형 복잡도와 넓은 수용 영역이라는 장점을 활용하면서, 다중 소스 데이터의 이질성을 극복하고 계산 효율성을 최적화한 **MSFMamba** 네트워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 다중 스케일 공간 특성 추출, 분광 특성 추출, 그리고 교차 모달 융합을 위한 세 가지 특화된 Mamba 블록을 설계하여 통합하는 것이다.

1. **MSpa-Mamba (Multi-Scale Spatial Mamba)**: 서로 다른 stride를 가진 depth-wise convolution을 통해 다중 스케일 특성 맵을 생성하고, 일부 경로만 다운샘플링하여 SSM을 처리함으로써 다방향 스캔 시 발생하는 특성 중복과 계산 비용을 줄였다.
2. **Spe-Mamba (Spectral Mamba)**: HSI 데이터의 고유한 특성인 풍부한 분광 정보를 효과적으로 모델링하기 위해, 공간 차원이 아닌 채널(분광) 차원을 따라 스캔하는 메커니즘을 도입하였다.
3. **Fus-Mamba (Fusion Mamba)**: 기존 Mamba가 단일 입력만을 처리하는 한계를 극복하기 위해, 한 모달리티의 특성을 이용해 다른 모달리티의 SSM 파라미터를 생성하는 대칭적 구조를 설계함으로써 교차 모달 간의 상호작용을 극대화하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계
- **CNN 기반 방법**: End-to-end 방식으로 특징 인코더를 구성하며, 다양한 어텐션 메커니즘(Cross-guided attention, Cross-channel correlation 등)을 통해 특징을 융합한다. 그러나 구조적으로 수용 영역이 제한되어 장거리 의존성을 포착하기 어렵다.
- **Transformer 기반 방법**: 전역 수용 영역을 통해 장거리 특징 모델링 능력이 뛰어나지만, 시퀀스 길이에 따라 계산 복잡도가 제곱으로 증가하는 효율성 문제가 있다.
- **SSM/Mamba 기반 방법**: 최근 비전 및 원격 탐사 분야에 도입되어 선형 복잡도로 전역 문맥을 파악할 수 있음을 보여주었다. 하지만 다중 소스 데이터의 이질성 간극을 메우기 위한 교차-어텐션과 같은 설계가 부족하며, 이미지 적용 시 다방향 스캔으로 인한 중복성 문제가 존재한다.

### 차별점
MSFMamba는 Mamba의 효율성을 유지하면서도, **다중 스케일 전략**으로 중복성을 제거하고, **분광 전용 블록**과 **교차 모달 파라미터 생성 방식의 융합 블록**을 통해 기존 SSM 모델들이 다루지 못한 다중 소스 데이터 융합 문제를 체계적으로 해결하였다.

## 🛠️ Methodology

### 1. State Space Model (SSM) 및 Mamba 기초
SSM은 입력 시퀀스 $x(t)$를 출력 시퀀스 $y(t)$로 매핑하는 선형 시불변 시스템으로, 다음과 같은 미분 방정식으로 표현된다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

여기서 $h(t)$는 은닉 상태, $A$는 상태 전이 행렬, $B, C$는 투영 행렬, $D$는 잔차 연결을 의미한다. 딥러닝 네트워크 적용을 위해 제로차 홀드(Zeroth-order hold) 규칙을 사용하여 이산화하며, timescale 파라미터 $\Delta$를 도입하여 이산 파라미터 $\overline{A}, \overline{B}$로 변환한다. Mamba는 여기에 입력 의존적인 파라미터를 도입하여 선형 시변(Linear Time-Varying) 시스템으로 확장함으로써 선택적 스캔(Selective Scan)을 가능하게 한다.

### 2. 전체 파이프라인
MSFMamba는 $L$개의 **Spatial-Spectral Mamba Module**과 최종 분류를 위한 2층의 Fully-connected layer(Classifier)로 구성된다. HSI 데이터는 먼저 PCA를 통해 주요 $N_p$개 밴드로 압축된 후 입력된다.

각 Spatial-Spectral Mamba Module은 다음의 세 가지 블록을 순차적으로 거친다.
$$\text{Input} \rightarrow \text{MSpa-Mamba} \rightarrow \text{Spe-Mamba} \rightarrow \text{Fus-Mamba} \rightarrow \text{Output}$$

### 3. 상세 구성 요소

#### (1) MSpa-Mamba Block
공간적 장거리 의존성을 포착하면서 중복성을 줄이는 것이 목표이다.
- **구조**: 두 개의 병렬 브랜치로 구성된다. 브랜치 1은 $\text{Linear} \rightarrow \text{DWConv} \rightarrow \text{SiLU} \rightarrow \text{MSpa-SSM} \rightarrow \text{LayerNorm}$ 과정을 거치고, 브랜치 2는 $\text{Linear} \rightarrow \text{SiLU}$ 과정을 거친다. 두 결과는 원소별 곱셈($\odot$)으로 결합된다.
- **MSpa-SSM (Multi-scale Spatial SSM)**:
    - DWConv의 stride를 1과 2로 설정하여 원본 해상도($Z_1$)와 다운샘플링된 해상도($Z_2$)의 특성 맵을 생성한다.
    - 4방향 스캔 경로를 사용하는데, 2개 경로는 $Z_1$을 그대로 처리하고, 나머지 2개 경로는 $Z_2$를 처리한 후 다시 보간(Interpolate)하여 원래 크기로 복원한다.
    - 이를 통해 모든 경로에서 동일한 해상도를 처리하는 것보다 계산량을 줄이고 정보 중복을 방지한다.

#### (2) Spe-Mamba Block
HSI의 분광 관계를 모델링하기 위한 블록이다.
- MSpa-Mamba와 유사한 병렬 구조를 가지나, SSM 레이어의 동작 방식이 다르다.
- 공간 차원이 아닌 **채널(Spectral) 차원**을 따라 1D 시퀀스로 변환하여 스캔하며, 스캔 방향은 2방향으로 제한된다.

#### (3) Fus-Mamba Block
HSI($F_h$)와 LiDAR/SAR($F_x$) 간의 이질성 간극을 메우기 위한 융합 블록이다.
- **Fus-SSM (Fusion SSM)**: 대칭적 구조를 가진다.
    - **HSI 특성 생성**: $F_x$ (LiDAR/SAR)로부터 $\Delta, B, C$ 파라미터를 생성하고, 이를 사용하여 $F_h$ (HSI) 시퀀스를 처리하여 $F_{ho}$를 생성한다.
    - **LiDAR/SAR 특성 생성**: 반대로 $F_h$ (HSI)로부터 파라미터를 생성하여 $F_x$ 시퀀스를 처리함으로써 $F_{xo}$를 생성한다.
- 기존 Mamba가 자신의 입력에서 파라미터를 추출하는 것과 달리, **상대 모달리티의 정보를 통해 필터링 파라미터를 결정**함으로써 더 깊은 교차 모달 상호작용을 구현하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: Berlin, Augsburg (HSI + SAR), Houston 2018, Houston 2013 (HSI + LiDAR) 4종의 벤치마크 데이터셋 사용.
- **평가 지표**: 전체 정확도(OA), 평균 정확도(AA), Kappa 계수.
- **비교 대상**: TBCNN, FusAtNet, $S^2$ENet, DFINet, AsyFFNet, ExViT, HCT, MACN 등 최신 CNN 및 Transformer 기반 모델.

### 정량적 결과
MSFMamba는 모든 데이터셋에서 SOTA(State-of-the-art) 성능을 달성하였다.
- **OA 결과**: Berlin(76.92%), Augsburg(91.38%), Houston 2018(92.38%), Houston 2013(92.86%).
- 특히 Berlin 데이터셋에서 Industrial Area, Low Plants 등 미세한 텍스처 구분이 필요한 클래스에서 강점을 보였다.

### 효율성 분석 (Augsburg 데이터셋 기준)
| 모델 | Params (M) | FLOPS (G) | Inference Time (s) |
| :--- | :---: | :---: | :---: |
| ExViT | 52.41 | 0.2901 | 0.3271 |
| MACN | 84.78 | 0.1672 | 0.1917 |
| **MSFMamba** | **91.52** | **0.0377** | **0.1747** |

MSFMamba는 파라미터 수는 타 모델과 유사하거나 약간 많지만, **FLOPS와 추론 시간 면에서 압도적으로 낮은 수치**를 기록하여 실시간 적용 가능성이 높음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
1. **모듈 간 상보적 관계**: Ablation Study 결과, MSpa-Mamba(공간), Spe-Mamba(분광), Fus-Mamba(융합)가 모두 포함되었을 때 가장 높은 OA를 기록하였다. 이는 각 블록이 서로 다른 차원의 정보를 보완적으로 추출하고 있음을 의미한다.
2. **다중 스케일 전략의 유효성**: 다운샘플링 경로를 2개로 설정했을 때 최적의 성능이 나타났으며, 이는 세부 디테일 유지와 전역 패턴 포착 사이의 최적의 균형점임을 보여준다.
3. **교차 모달 융합의 효과**: t-SNE 시각화 결과, Fus-Mamba 적용 전에는 클래스 간 경계가 모호했으나 적용 후에는 클러스터가 매우 명확하고 콤팩트하게 형성되었다. 이는 상대 모달리티의 정보를 이용해 SSM 파라미터를 생성하는 방식이 특징 정렬(Feature Alignment)에 매우 효과적임을 시사한다.

### 한계 및 향후 과제
- **클래스 불균형**: 데이터셋 내 샘플 수가 적은 불균형 클래스에 대해서는 여전히 정확도가 낮아지는 경향이 있다.
- **해석 가능성**: 딥러닝 모델 특성상 도시 계획이나 재난 관리와 같은 임계 애플리케이션에서 요구되는 높은 수준의 해석 가능성(Explainability)을 충분히 제공하지 못한다.
- 저자는 향후 Mamba와 Transformer를 결합한 하이브리드 구조 연구와 동적 샘플링, 고급 손실 함수 도입을 통한 불균형 해소를 계획하고 있다.

## 📌 TL;DR

본 논문은 HSI와 LiDAR/SAR 데이터의 결합 분류를 위해 **Mamba(SSM) 기반의 MSFMamba** 네트워크를 제안하였다. 핵심은 **다중 스케일 공간-분광 특성 추출**과 **상대 모달리티 정보를 이용한 교차 융합 메커니즘**이다. 이를 통해 Transformer의 전역 수용 영역과 CNN의 효율성을 동시에 확보하였으며, 4개의 벤치마크 데이터셋에서 SOTA 성능을 달성함과 동시에 매우 낮은 계산 복잡도(FLOPS)와 추론 시간을 기록하였다. 이 연구는 자원이 제한된 에지 디바이스(드론, 모바일 플랫폼)에서의 실시간 원격 탐사 이미지 분류에 중요한 기여를 할 것으로 기대된다.