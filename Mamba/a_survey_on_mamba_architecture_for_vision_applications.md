# A Survey on Mamba Architecture for Vision Applications

Fady Ibrahim, Guangjun Liu, Guanghui Wang (2025)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 표준으로 자리 잡은 Vision Transformer (ViT) 아키텍처의 치명적인 한계점인 연산 복잡도 문제를 해결하고자 한다. Transformer의 핵심인 Attention 메커니즘은 입력 시퀀스 길이에 대해 이차 복잡도(Quadratic Complexity)를 가지며, 이는 고해상도 이미지나 긴 비디오 시퀀스를 처리할 때 메모리 사용량과 추론 시간의 급격한 증가를 초래하여 확장성(Scalability) 문제를 야기한다.

따라서 본 연구의 목표는 선형 복잡도(Linear Scalability)를 가지면서도 효율적인 전처리와 향상된 문맥 인식 능력을 제공하는 State-Space Models (SSMs) 기반의 Mamba 아키텍처를 분석하고, 이를 시각적 작업(Vision Tasks)에 어떻게 적용하고 최적화할 수 있는지를 종합적으로 검토하는 것이다. 특히 Vision Mamba (ViM)와 VideoMamba를 중심으로 이미지 및 비디오 이해를 위한 최신 아키텍처 혁신 사례들을 상세히 분석한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Mamba 아키텍처를 시각 도메인에 적용한 최신 연구들을 체계적으로 분류하고 분석한 점에 있다. 주요 기여 사항은 다음과 같다.

첫째, 1D 시퀀스 데이터에 최적화된 기존 Mamba를 비인과적(Non-causal) 특성을 가진 시각 데이터에 적합하도록 변형한 Bidirectional Scanning 및 Spatiotemporal Processing 메커니즘을 상세히 설명한다.

둘째, Position Embeddings, Cross-scan modules, Hierarchical designs와 같은 구조적 개선 사항들이 어떻게 글로벌 및 로컬 특징 추출 성능을 최적화하는지 분석한다.

셋째, 이미지 분류(Image Classification), 시맨틱 세그멘테이션(Semantic Segmentation), 객체 탐지(Object Detection), 인간 행동 인식(Human Action Recognition) 등 다양한 작업에 대해 여러 Mamba 변형 모델들의 정량적 성능을 비교 분석하여, 작업별 최적의 아키텍처 선택을 위한 실무적인 가이드를 제공한다.

## 📎 Related Works

논문은 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)가 컴퓨터 비전의 발전을 이끌었음을 언급한다. 특히 ViT는 복잡한 의존성을 캡처하는 능력이 뛰어나 state-of-the-art 성능을 달성했지만, 앞서 언급한 이차 복잡도 문제가 상용화 및 대규모 데이터 처리의 병목 현상이 되고 있음을 지적한다.

기존의 Mamba 관련 서베이 논문들이 텍스트, 추천 시스템, 의료 영상 등 광범위한 분야를 다루었다면, 본 논문은 오직 시각적 작업(Visual Tasks)에만 집중하여 차별성을 둔다. 또한, 단순히 모델을 나열하는 것이 아니라 ViM과 VideoMamba라는 핵심 모델을 기준으로 이들의 구조적 차이와 발전 방향을 심도 있게 다룬다.

## 🛠️ Methodology

### 1. Mamba 및 SSM의 기초 이론

Mamba의 기반이 되는 State-Space Models (SSMs)는 1D 입력 시퀀스 $x(t)$를 숨겨진 상태(Hidden State) $h(t)$를 통해 출력 $y(t)$로 매핑하는 시스템이다. 연속 시간 시스템은 다음과 같이 정의된다.

$$\frac{dh(t)}{dt} = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$

여기서 $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$이다. 이 연속 시스템을 이산화(Discretization)하기 위해 Zero Order Hold (ZOH) 방법을 사용하며, step size $\Delta$를 이용하여 다음과 같이 변환한다.

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

최종적으로 이산화된 시스템 방정식은 다음과 같이 표현된다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t, \quad y_t = Ch_t$$

### 2. Selective Scanning Mechanism (S6)

전통적인 SSM은 입력에 관계없이 고정된 파라미터를 사용하여 문맥 인식 능력이 떨어진다. Mamba는 $B, C, \Delta$ 파라미터를 입력 $x$의 함수로 만드는 Selective Scan Mechanism (S6)을 도입하여, 불필요한 정보는 버리고 중요한 정보는 유지하는 능력을 갖추게 한다.

$$B, C, \Delta = \text{Linear}(x)$$

### 3. Vision Mamba (ViM)의 구조

이미지 데이터는 텍스트와 달리 방향성이 없는 비인과적(Non-causal) 특성을 가진다. ViM은 이를 해결하기 위해 다음과 같은 전략을 사용한다.

- **Bidirectional Scanning**: 시퀀스를 정방향과 역방향으로 동시에 스캔하여 각 요소가 앞뒤 문맥을 모두 참조할 수 있게 한다.
- **Position Embeddings**: 1D 시퀀스 모델인 Mamba에 공간적 인식 능력을 부여하기 위해 위치 임베딩을 추가한다.
- **Gating Mechanism**: 정방향 및 역방향 출력 특징을 SiLU 활성화 함수를 통한 게이팅 메커니즘으로 조절하여 합산한다.

### 4. VideoMamba 및 시공간 처리

비디오 데이터의 3D 특성을 처리하기 위해 VideoMamba는 Spatiotemporal Scanning을 도입한다. Spatial-First, Temporal-First, 그리고 Spatiotemporal 스캔 방식을 통해 공간적 정보와 시간적 정보를 효율적으로 결합하며, 긴 의존성을 처리하기 위한 Linear-complexity operator를 사용한다.

## 📊 Results

본 논문은 다양한 벤치마크 데이터셋을 통해 Mamba 기반 모델들의 성능을 비교하였다.

### 1. 이미지 분류 (ImageNet-1k)

$\text{Top-1 Accuracy}$ 기준, **Spatial Mamba-S**가 $84.6\%$로 가장 높은 성능을 보였다. 이는 Structure-Aware State Fusion 덕분으로 분석된다. 반면, $\text{EfficientViM-M4}$는 $81.9\%$의 정확도를 가지면서도 파라미터 수($21.3\text{M}$)와 연산량($4.1\text{G FLOPS}$)이 매우 적어 자원 제한 환경에서 효율적임을 입증하였다.

### 2. 시맨틱 세그멘테이션 (ADE20K)

$\text{mIoU}$ 기준, **Vmamba-S**와 **Spatial Mamba-S**가 $50.6$으로 공동 최고 성능을 기록하였다. 다만, 이 모델들은 연산 비용($1028\text{G}$ 및 $992\text{G FLOPS}$)이 매우 높았다. $\text{MambaR}$는 $45.3$으로 상대적으로 낮은 성능을 보여, 레지스터 토큰 임베딩이 세그멘테이션 작업에는 덜 효과적일 수 있음을 시사한다.

### 3. 객체 탐지 (MS-COCO)

$AP_{bb}^{75}$ 지표에서 **Spatial Mamba-S**가 $54.2$로 가장 우수한 성능을 보였으며, **VSSD-S**가 $53.1$로 그 뒤를 이었다. $\text{EfficientViM-M4}$는 효율성은 극대화되었으나 성능($41.1$)은 크게 저하되는 트레이드오프를 보였다.

### 4. 인간 행동 인식 (Kinetics-400, SSv2)

비디오 작업에서는 **VideoMambaPro-S**가 Kinetics-400에서 $88.5\%$의 정확도를 달성하며 기존 VideoMamba-S($81.5\%$)보다 월등한 성능을 보였다. 이는 Residual SSM 및 Masked Backward Residual SSM 구성 요소가 복잡한 시공간 특징 추출에 기여했기 때문이다.

## 🧠 Insights & Discussion

### 강점 및 기회

Mamba 아키텍처는 Transformer의 성능을 유지하면서도 선형 복잡도를 달성함으로써, 고해상도 이미지와 초장거리 비디오 시퀀스 처리에서 압도적인 효율성을 보여준다. 특히 Selective Scanning과 Bidirectional 접근법은 시각 데이터의 특성을 잘 반영하고 있으며, Structure-Aware fusion과 같은 기법을 통해 공간적 의존성 캡처 능력을 지속적으로 향상시키고 있다.

### 한계 및 비판적 해석

첫째, **Artifacts 문제**이다. ViM의 특징 맵에서 아티팩트가 발견된다는 점이 지적되었으며, 이를 해결하기 위해 $\text{MambaR}$에서 Register Tokens를 도입하는 등의 보완책이 제시되었다.
둘째, **비디오 성능의 격차**이다. VideoMamba가 효율성 면에서는 뛰어나지만, 여전히 일부 벤치마크에서 Transformer 기반 모델의 성능에 미치지 못하는 경우가 존재한다. 특히 '역사적 쇠퇴(Historical Decay)'와 '요소 간 모순(Element Contradiction)' 문제가 해결해야 할 과제로 남아 있다.
셋째, **연산 비용의 불균형**이다. 일부 고성능 모델(Spatial Mamba 등)은 정확도는 높지만 FLOPS가 매우 높아, Mamba의 원래 취지인 '효율성'과 '성능' 사이의 상충 관계가 여전히 존재한다.

## 📌 TL;DR

본 논문은 Transformer의 이차 복잡도 문제를 해결하기 위해 선형 복잡도의 SSM 기반 Mamba 아키텍처를 시각 작업에 적용한 연구들을 종합적으로 분석한 서베이 보고서이다. **Vision Mamba (ViM)**와 **VideoMamba**를 중심으로 Bidirectional 및 Spatiotemporal Scanning과 같은 핵심 메커니즘을 설명하며, 특히 **Spatial Mamba-S**와 **VideoMambaPro-S**가 각 영역에서 뛰어난 성능을 보임을 확인하였다. 이 연구는 향후 네이티브 2D/3D Mamba 모델 개발 및 멀티모달 융합 연구의 중요한 기초 자료가 될 것이며, 특히 실시간 고해상도 영상 처리 시스템의 핵심 아키텍처로 자리 잡을 가능성이 높다.
