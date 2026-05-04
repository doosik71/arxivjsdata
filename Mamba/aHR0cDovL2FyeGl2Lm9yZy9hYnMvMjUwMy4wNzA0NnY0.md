# MambaFlow: A Mamba-Centric Architecture for End-to-End Optical Flow Estimation

Juntian Du, Zhihu Zhou, Runzhe Zhang, Yuan Sun, Pinyi Chen, Keji Mao (2025)

## 🧩 Problem to Solve

본 논문은 비디오 시퀀스의 연속된 프레임 사이에서 픽셀 단위의 움직임 벡터를 계산하는 Optical Flow Estimation(광학 흐름 추정) 문제를 다룬다. Optical Flow는 행동 인식, 비디오 보간, 자율 주행 등 다양한 컴퓨터 비전 분야에서 핵심적인 역할을 한다.

기존의 접근 방식들은 다음과 같은 한계점을 가지고 있다:
- **CNN 기반 방식**: Local perception의 한계로 인해 큰 변위(large displacement)나 복잡한 움직임을 처리하는 데 어려움이 있다.
- **Transformer 기반 방식**: Global modeling 능력이 뛰어나 정확도는 높지만, Attention 메커니즘의 이차 복잡도(quadratic computational complexity, $\mathcal{O}(N^2)$)로 인해 학습 및 추론 단계에서 계산 비용과 메모리 소모가 매우 크다.
- **Iterative Refinement 방식 (예: RAFT)**: 반복적인 정제 과정을 통해 정확도를 높이지만, 많은 반복 횟수로 인해 추론 시간이 증가하고 파라미터 수가 늘어나는 경향이 있다.

따라서 본 논문의 목표는 **정확도를 유지하면서도 계산 효율성을 획기적으로 높인 end-to-end Optical Flow 추정 프레임워크**를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 컴퓨터 비전 분야에서 CNN의 국소적 한계와 Transformer의 계산 복잡도 문제를 동시에 해결할 수 있는 **Mamba (Selective State Space Model)** 아키텍처를 Optical Flow 추정 파이프라인의 중심에 배치하는 것이다.

주요 기여 사항은 다음과 같다:
1. **MambaFlow 제안**: 특징 강화(Feature Enhancement)와 흐름 전파(Flow Propagation) 단계 모두에 Mamba를 적용한 최초의 Mamba-centric Optical Flow 아키텍처를 설계하였다.
2. **PolyMamba 모듈**: Self-Mamba와 Cross-Mamba를 통해 프레임 내(intra-frame) 및 프레임 간(inter-frame) 의존성을 모두 캡처하여 특징 표현력을 글로벌하게 최적화하였다.
3. **PulseMamba 모듈**: Attention Guidance Aggregator (AGA)를 통해 특징을 적응적으로 통합하고, Mamba의 자기회귀(autoregressive) 특성을 이용해 효율적인 반복적 흐름 정제(iterative refinement)를 수행한다.

## 📎 Related Works

### Optical Flow Estimation
초기에는 밝기 항상성(brightness constancy)과 공간적 매끄러움(spatial smoothness) 가정을 기반으로 한 에너지 최소화 방법이 사용되었다. 이후 FlowNet, PWC-Net과 같은 CNN 기반의 coarse-to-fine 전략이 등장했으나, 빠른 움직임 처리에 한계가 있었다. RAFT는 반복적인 full-field refinement를 통해 이를 극복했으며, 이후 GMA, MS-RAFT 등이 이를 확장하였다. 최근에는 FlowFormer, GMFlow와 같이 Transformer의 global dependency 모델링 능력을 활용한 연구들이 진행되었으나, 높은 계산 비용이 실용적 배포의 걸림돌이 되고 있다.

### State Space Models (SSMs)
SSM은 제어 이론에서 유래되었으며, 최근 Mamba는 입력에 따라 파라미터가 변하는 Selective mechanism을 도입하여 Transformer 수준의 성능을 내면서도 선형 복잡도(linear complexity)를 달성하였다. Vision Mamba, VMamba 등이 이미지 분류 및 세그멘테이션에 적용되었으나, Optical Flow 분야에서는 point cloud 기반의 scene flow 추정에 사용된 사례(FlowMamba) 외에는 일반적인 Optical Flow 아키텍처에 적용된 바가 없었다.

## 🛠️ Methodology

### 1. Preliminaries: State Space Models (SSMs)
Mamba의 기반이 되는 SSM은 입력 시퀀스 $x_t$를 은닉 상태 $h_t$를 통해 출력 $y_t$로 매핑하는 선형 상미분 방정식(ODE)으로 정의된다:
$$h_t = Ah_{t-1} + Bx_t$$
$$y_t = Ch_t$$
여기서 $A, B, C$는 시스템 파라미터이다. Mamba는 $B, C, \Delta$를 입력 $x_i$에 의존하게 만들어 동적인 특징 표현을 가능하게 한다:
$$B_i = S_B(x_i), \quad C_i = S_C(x_i), \quad \Delta_i = \text{Softplus}(S_\Delta(x_i))$$

### 2. Overall Architecture
MambaFlow의 전체 파이프라인은 다음과 같다:
**CNN Encoder $\rightarrow$ PolyMamba $\rightarrow$ Global Matching $\rightarrow$ PulseMamba $\rightarrow$ Final Flow**

### 3. PolyMamba (Feature Enhancement)
특징 맵 $F_1, F_2$를 입력받아 글로벌 컨텍스트를 강화한다.
- **Self-Mamba**: 각 프레임 내부의 의존성을 모델링한다. 양방향(bidirectional) 연산을 통해 특징을 강화한다.
  $$F_{self}^i = \text{Self-Forward}(F^i) + \text{Self-Backward}(F^i), \quad i \in \{1, 2\}$$
- **Cross-Mamba**: 두 프레임 간의 상호작용을 모델링한다. 한 쪽의 특징을-메인 브랜치로, 다른 쪽을-변조 브랜치(modulation branch)로 사용하여 파라미터 $[\Delta, B, C]$를 생성하고 Selective Scan을 수행한다.
- **MLP**: 각 블록 그룹 뒤에 가벼운 MLP를 배치하여 지역적-전역적 특징을 융합한다.

### 4. Global Matching (Initial Flow Estimation)
PolyMamba를 통해 강화된 특징 $F_q, F_v$를 사용하여 4D cost volume을 생성하고 초기 흐름 $V_{initial}$을 추정한다.
- **Matching Distribution ($M$)**: 
  $$M = \text{softmax}\left(\frac{F_q F_v^\top}{\sqrt{D}}\right)$$
- **Initial Flow**: 타겟 이미지의 픽셀 그리드 좌표 $G$에 $M$을 가중 평균하여 대응점을 찾고, 현재 좌표와의 차이를 통해 $V_{initial}$을 계산한다.

### 5. PulseMamba (Iterative Refinement)
초기 흐름을 정교하게 다듬는 모듈이다.
- **Attention Guidance Aggregator (AGA)**: 단순히 특징을 연결(concatenation)하는 대신, 움직임 특징 $M$, 컨텍스트 특징 $F_q$, 은닉 상태 $h$를 입력받아 공간 주의 지도(spatial attention maps) $A$를 생성하고 가중 합을 구한다.
  $$x_{AGA} = \sum_{f \in \{M, F_q, h\}} A^f \odot f$$
- **Autoregressive Refinement**: Mamba 레이어를 통해 은닉 상태를 업데이트하고 flow 증분 $\Delta V$를 예측하여 반복적으로 업데이트한다.
  $$h^{(i)} = \text{Mamba}(x_{AGA}^{(i-1)})$$
  $$V^{(i)} = V^{(i-1)} + \text{FlowHead}(h^{(i)})$$

## 📊 Results

### 실험 설정
- **데이터셋**: FlyingChairs, FlyingThings3D로 사전 학습 후 KITTI, HD1K, Sintel로 미세 조정(fine-tuning)하였다.
- **지표**: End-point-error (EPE), KITTI의 F1-all, 큰 움직임에 대한 $s_{40+}$ EPE를 사용하였다.
- **하드웨어**: NVIDIA GeForce RTX 3090 GPU 사용.

### 정량적 결과 (Sintel & KITTI)
- **Sintel (Clean)**: MambaFlow는 EPE 1.20을 달성하여 MS-RAFT+보다 정확도가 높으며, 추론 속도는 **116.5ms**로 MS-RAFT+(1108.2ms)보다 약 **9.51배 빠르다**.
- **SEA-RAFT(L) 비교**: 추가 데이터(TartanAir)를 사용한 SEA-RAFT(L)과 유사한 속도를 보이면서도, Sintel Clean에서 EPE를 약 8.4% 감소시켰다.
- **효율성**: 파라미터 수(20.5M)와 추론 시간 면에서 State-of-the-art (SOTA) 모델들과 비교해 매우 우수한 균형을 보여준다.

### 절제 연구 (Ablation Study)
- **PolyMamba 구성**: Cross-Mamba가 성능 향상에 가장 크게 기여하였으며, 이는 CNN 백본이 잡지 못하는 두 특징 간의 상호 관계를 모델링하기 때문이다.
- **PulseMamba 반복**: 반복 횟수가 증가함에 따라 EPE가 감소하며, 특히 unmatched 픽셀에서 큰 이득을 보였다.
- **AGA 효과**: 단순 연결보다 AGA를 통한 적응적 특징 융합이 일관되게 더 낮은 EPE를 기록하였다.

## 🧠 Insights & Discussion

### 강점
MambaFlow는 Mamba의 선형 복잡도와 글로벌 모델링 능력을 Optical Flow에 성공적으로 이식하였다. 특히 Transformer의 높은 정확도를 유지하면서도 CNN 수준의 빠른 추론 속도를 달성했다는 점이 고무적이다. 이는 자원이 제한된 엣지 디바이스(resource-constrained devices)에서의 실시간 Optical Flow 적용 가능성을 높인다.

### 한계 및 향후 과제
- **일반화 문제**: KITTI 데이터셋 결과에서 나타나듯, 학습 데이터와 테스트 데이터 간의 간극이 클 때 일반화 성능이 다소 떨어지는 경향이 있다.
- **미탐색 영역**: 모션 블러(motion blur), 안개(fog)와 같은 극한 환경이나 초고해상도 이미지에서의 성능은 아직 검증되지 않았다.
- **해결 방안**: 저자들은 TartanAir나 VIPER와 같은 대규모 데이터셋으로의 사전 학습을 통해 이 문제를 해결할 계획임을 명시하였다.

### 비판적 해석
본 연구는 Mamba를 단순한 대체제로 사용한 것이 아니라, PolyMamba와 PulseMamba라는 구체적인 구조적 설계를 통해 Optical Flow의 특성(프레임 간 대응점 찾기 및 반복적 정제)을 반영하려 노력하였다. 다만, Mamba의 다양한 변형 모델들이 계속 등장하고 있으므로, 최신 SSM 변형 아키텍처를 적용했을 때의 성능 향상 폭에 대한 추가 분석이 필요할 것으로 보인다.

## 📌 TL;DR

MambaFlow는 **Mamba(Selective SSM)를 기반으로 한 최초의 end-to-end Optical Flow 추정 프레임워크**이다. 특징 강화 단계의 **PolyMamba**와 정제 단계의 **PulseMamba(AGA 포함)**를 통해 Transformer 급의 전역 정보 캡처 능력을 갖추면서도 선형 복잡도를 달성하였다. 실험 결과, SOTA 모델 대비 **비슷하거나 더 높은 정확도를 유지하면서 추론 속도를 획기적으로 단축**시켰으며, 이는 실시간성 및 저전력 환경의 광학 흐름 추정에 매우 중요한 기여를 할 것으로 기대된다.