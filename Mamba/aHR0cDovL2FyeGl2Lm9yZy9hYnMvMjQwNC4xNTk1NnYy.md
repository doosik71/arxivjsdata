# A Survey on Visual Mamba

Hanwei Zhang, Ying Zhu, Dan Wang, Lijun Zhang, Tianxiang Chen, and Zi Ye (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 Transformer 아키텍처가 직면한 계산 효율성 문제를 해결하기 위한 대안으로 부상한 Mamba 모델들의 최신 연구 동향을 분석한다. Transformer의 핵심인 Self-attention 메커니즘은 이미지 크기에 따라 계산 복잡도가 이차적으로 증가하는 $\mathcal{O}(N^2)$의 특성을 가지고 있으며, 이는 고해상도 이미지 처리 시 막대한 계산 자원을 요구하고 엣지 디바이스나 실시간 시스템으로의 배포를 어렵게 만든다. 

따라서 본 연구의 목표는 선형 복잡도 $\mathcal{O}(L)$를 가지면서도 긴 시퀀스의 의존성을 효과적으로 캡처할 수 있는 Selective State Space Model(SSM)인 Mamba를 컴퓨터 비전 작업에 어떻게 적용하고 최적화할 수 있는지에 대한 포괄적인 분석 보고서를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 다음과 같다.

- **최초의 종합적 서베이**: 컴퓨터 비전 영역에서 Mamba 기술을 적용한 연구들을 체계적으로 정리한 첫 번째 종합 서베이 논문이다.
- **기술적 분류 체계 확립**: Naive Mamba 프레임워크를 기반으로, 이를 개선하기 위해 Convolution, Recurrence, Attention 등 타 아키텍처와 결합한 기법들을 분류하고 분석하였다.
- **응용 분야별 택소노미 제공**: 일반 비전 작업(High/Mid-level 및 Low-level), 의료 영상 처리, 원격 탐사(Remote Sensing) 등 다양한 도메인에서의 Mamba 적용 사례를 정리하고 각 작업의 특성에 따른 발전 방향을 제시하였다.

## 📎 Related Works

본 논문에서는 딥러닝 아키텍처의 진화 과정을 통해 Mamba의 등장 배경을 설명한다. 

- **기존 아키텍처의 한계**: MLP와 CNN은 국부적인 특징 추출에는 능숙하지만 전역적인 관계를 파악하는 데 한계가 있으며, RNN은 시퀀스 데이터 처리가 가능하나 vanishing gradient 문제와 긴 의존성 학습의 어려움이 있다.
- **Transformer의 명과 암**: Transformer는 Attention 메커니즘을 통해 전역적 특징 표현 능력을 극대화하였으나, 앞서 언급한 이차 복잡도의 계산 비용이 치명적인 약점으로 작용한다.
- **SSM의 진화**: State Space Model(SSM)은 본래 제어 이론에서 유래하였으며, 선형 복잡도를 가지지만 매개변수가 시간에 따라 변하지 않는 Linear Time Invariance(LTI) 특성 때문에 시퀀스 문맥 표현 능력이 제한적이었다. Mamba는 이러한 LTI의 제약을 깨고 입력 데이터에 따라 매개변수가 동적으로 변하는 'Selection Mechanism'을 도입하여 Transformer 수준의 표현력과 RNN 수준의 효율성을 동시에 달성하였다.

## 🛠️ Methodology

### 1. Mamba의 수학적 기초 (SSM Formulation)
Mamba의 근간이 되는 SSM은 1차원 시퀀스 $x(t)$를 숨겨진 상태 $h(t)$를 통해 $y(t)$로 매핑하는 선형 상미분 방정식으로 정의된다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A$는 진화 매개변수, $B$와 $C$는 투영 매개변수이다. 딥러닝 적용을 위해 Zero-Order Hold(ZOH) 가정을 사용하여 이를 이산화(Discretization)하며, 타임스케일 매개변수 $\Delta$를 통해 다음과 같이 변환된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

이산화된 시스템은 다음과 같은 재귀 형태로 표현된다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = C h_t$$

### 2. Selection Mechanism 및 Hardware-aware Design
Mamba의 핵심은 매개변수 $B, C, \Delta$가 입력 $x$에 의존하도록 설계한 **Selection Mechanism**이다. 이를 통해 모델은 입력 시퀀스에서 중요한 정보를 선택적으로 유지하거나 삭제할 수 있는 필터링 능력을 갖게 된다. 또한, GPU 메모리 I/O 효율을 극대화하기 위해 커널 퓨전(Kernel Fusion)과 병렬 스캔(Parallel Scan)을 결합한 **Hardware-aware algorithm**을 사용하여 계산 속도를 획기적으로 높였다.

### 3. Visual Mamba의 핵심 과제: Scanning Strategy
Mamba는 본래 1차원 시퀀스를 처리하도록 설계되었으나, 이미지는 2차원 데이터이다. 이를 해결하기 위해 본 논문은 다양한 스캔 전략을 소개한다.

- **ViM (Vision Mamba)**: 이미지를 패치 시퀀스로 변환한 후, 전방향(Forward)과 후방향(Backward)으로 동시에 스캔하는 **Bidirectional Scan**을 사용하여 2D 공간 정보를 캡처한다.
- **VMamba**: 이미지를 4가지 방향으로 스캔하여 공간적 도메인을 탐색하는 **Cross-Scan Module(CSM)**을 도입하여 비인과적(Non-causal) 이미지 데이터를 순서가 있는 패치 시퀀스로 변환한다.
- **기타 전략**: 국부적 정보를 강조하는 Local Scan, 공간적 연속성을 높인 Continuous 2D Scanning, 계산 효율을 극대화한 ES2D(Efficient 2D Scanning) 등이 제안되었다.

## 📊 Results

본 논문은 개별 실험 결과보다는 Mamba 기반 모델들이 다양한 비전 태스크에서 거둔 성과를 종합적으로 요약하여 제시한다.

### 1. 일반 비전 작업 (General Vision)
- **Backbone**: ViM, VMamba, PlainMamba 등이 Transformer 기반의 ViT와 경쟁 가능한 성능을 보이면서도 추론 속도와 메모리 효율성에서 우위를 점하고 있다.
- **High/Mid-level**: 객체 탐지(SSM-ViT), 참조 이미지 세그멘테이션(ReMamber), 비디오 이해(VideoMamba) 등에서 선형 복잡도의 이점을 활용하여 고해상도 및 긴 비디오 시퀀스를 효율적으로 처리한다.
- **Low-level**: 이미지 초해상도(MMA), 복원(MambaIR) 등의 작업에서 전역적 수용장(Global Receptive Field)과 계산 효율성의 트레이드-오프 문제를 해결하였다.

### 2. 특수 도메인 적용
- **의료 영상 (Medical Vision)**: 고해상도 의료 영상의 특성상 Transformer의 비용 부담이 컸으나, U-Mamba, SegMamba 등이 도입되면서 2D/3D 세그멘테이션 및 분류 작업에서 효율적인 전역 문맥 모델링을 가능케 하였다.
- **원격 탐사 (Remote Sensing)**: 매우 높은 해상도의 이미지(VHR)를 다루는 RS-Mamba 등이 Omnidirectional Selective Scan을 통해 다각도에서 이미지를 모델링하며 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 가능성
Mamba는 Transformer의 전역적 모델링 능력과 RNN의 선형 복잡도라는 두 마리 토끼를 잡은 아키텍처이다. 특히 데이터의 길이가 길어질수록(예: 고해상도 이미지, 긴 비디오, 3D 의료 영상) Transformer 대비 압도적인 효율성을 보여주며, 이는 실시간 시스템 및 자원 제한적 환경에서 매우 강력한 도구가 될 수 있다.

### 한계 및 미해결 과제
- **스캔 전략의 의존성**: 2D 데이터를 1D로 펼치는 스캔 방식에 따라 모델의 성능이 크게 좌우된다. 현재까지 제안된 다양한 스캔 방식들이 있으나, 이미지의 공간적 구조를 완벽하게 보존하는 최적의 스캔 방식에 대해서는 여전히 논의가 필요하다.
- **사전 학습 데이터**: Transformer가 대규모 데이터셋으로 사전 학습되어 강력한 성능을 내는 것처럼, Mamba 역시 대규모 데이터셋에서의 사전 학습 효율성과 일반화 성능에 대한 더 많은 검증이 필요하다.
- **해석 가능성**: 특히 의료 분야에서 모델이 왜 특정 영역을 세그멘테이션 했는지에 대한 해석 가능성(Interpretability) 확보가 중요한 과제로 남아 있다.

## 📌 TL;DR

본 논문은 Transformer의 이차 복잡도 문제를 해결하기 위해 등장한 **Mamba(Selective SSM)** 모델의 컴퓨터 비전 적용 현황을 분석한 최초의 종합 서베이이다. Mamba는 **Selection Mechanism**과 **Hardware-aware Scan**을 통해 선형 복잡도로 전역적 의존성을 학습하며, 특히 **Cross-Scan**과 같은 전략을 통해 2D 이미지 데이터를 효율적으로 처리한다. 이 연구는 고해상도 이미지 처리가 필수적인 **의료 영상 및 원격 탐사** 분야에서 Mamba가 Transformer를 대체할 강력한 대안이 될 수 있음을 시사한다.