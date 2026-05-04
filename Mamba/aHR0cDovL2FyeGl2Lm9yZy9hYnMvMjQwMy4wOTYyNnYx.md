# Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding

Guo Chen, Yifei Huang, Jilan Xu, Baoqi Pei, Zhe Chen, Zhiqi Li, Jiahao Wang, Kunchang Li, Tong Lu, and Limin Wang (2024)

## 🧩 Problem to Solve

비디오 이해(Video Understanding) 분야에서는 시공간적 역동성을 포착하기 위해 RNN, 3D CNN, 그리고 Transformer와 같은 다양한 아키텍처가 연구되어 왔다. 특히 Transformer 기반 모델들은 글로벌 컨텍스트 상호작용과 데이터 의존적 동적 계산 능력을 통해 뛰어난 성능을 보였으나, 시퀀스 길이에 따라 계산 복잡도가 이차적으로 증가하는 Quadratic Computational Complexity 문제가 존재한다. 이로 인해 프레임 수가 많은 긴 비디오(Long-form videos)를 처리할 때 효율성이 급격히 떨어지는 한계가 있다.

본 논문의 목표는 최근 자연어 처리(NLP) 분야에서 선형 시간 복잡도로 긴 시퀀스 모델링 능력을 입증한 State Space Model(SSM), 특히 Mamba 아키텍처가 비디오 이해 도메인에서 Transformer를 대체할 수 있는 실질적인 대안이 될 수 있는지 종합적으로 분석하는 것이다.

## ✨ Key Contributions

본 연구는 새로운 단일 모델을 제안하는 것보다 Mamba의 잠재력을 광범위하게 조사하는 데 집중하며, 다음과 같은 핵심 기여를 한다.

1.  **Video Mamba Suite 구축**: Mamba를 비디오 모델링에 적용하기 위해 4가지 역할(Role)을 정의하고, 이를 바탕으로 총 14개의 모델 및 모듈로 구성된 'Video Mamba Suite'를 구축하였다.
2.  **광범위한 벤치마크 평가**: 13개의 데이터셋과 12개의 비디오 이해 작업(Task)에 걸쳐 Mamba 기반 모델들의 성능을 검증하여 Transformer 대비 우위성과 효율성을 입증하였다.
3.  **DBM(Decomposed Bidirectionally Mamba) 블록 제안**: 기존 ViM(Vision Mamba) 블록의 구조를 개선하여, 입력 프로젝터를 분리하고 SSM 파라미터를 공유하는 DBM 블록을 제안하였으며, 특히 소규모 데이터셋에서 성능 향상을 확인하였다.
4.  **효율성-성능 트레이드오프 분석**: 프레임 수가 증가함에 따라 Mamba의 선형 복잡도가 Transformer의 이차 복잡도 대비 추론 속도 면에서 가지는 압도적인 이점을 정량적으로 분석하였다.

## 📎 Related Works

비디오 모델링은 초기 2D 네트워크 기반의 프레임 샘플링(TSN)에서 시작하여, 3D CNN을 통해 시공간 정보를 동시에 추출하는 방향으로 발전하였다. 이후 Transformer의 등장으로 글로벌 컨텍스트를 활용한 비디오 Transformer들이 주류가 되었으나, 앞서 언급한 계산 복잡도 문제로 인해 긴 비디오 처리에는 한계가 있었다.

최근에는 이를 해결하기 위해 RetNet이나 RWKV와 같은 선형 복잡도 모델들이 등장하였으며, 특히 SSM 기반의 S4와 Mamba는 긴 시퀀스 모델링에서 탁월한 성능을 보였다. 기존의 SSM 기반 비디오 연구는 주로 장기 비디오 분류(Long-term video classification)에 국한되어 있었으나, 본 논문은 이를 다양한 비디오 이해 작업으로 확장하여 Mamba의 범용성을 탐구한다.

## 🛠️ Methodology

### 1. State Space Model (SSM) 기초
SSM은 연속적인 시스템을 이산화하여 시퀀스를 처리한다. 입력 $x(t)$를 은닉 상태 $h(t)$를 통해 출력 $y(t)$로 변환하는 과정은 다음과 같은 상태 방정식으로 정의된다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$이다. 실제 구현을 위해 $\Delta$(timescale parameter)를 이용한 Zero-Order Hold 기법으로 이산화하며, 이산화된 식은 다음과 같다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = C h_t$$

최종 출력은 구조화된 컨볼루션 커널 $K = (CB, CAB, \dots, CA^{M-1}B)$를 이용한 글로벌 컨볼루션 과정 $y = x * K$를 통해 계산된다.

### 2. Mamba 블록 및 변형 구조
본 논문에서는 세 가지 형태의 블록을 비교 분석한다.

*   **Mamba Block**: Linear attention 연산자와 MLP 블록을 결합한 구조로, 하드웨어 최적화 알고리즘을 통해 효율적인 학습과 추론을 지원한다.
*   **ViM (Vision Mamba) Block**: Mamba의 단방향 스캔 한계를 극복하기 위해 Forward와 Backward 양방향 Selective Scanning 브랜치를 추가한 구조이다. 두 방향의 특징을 게이팅 레이어로 처리한 후 평균을 내어 출력한다.
*   **DBM (Decomposed Bidirectionally Mamba) Block**: 본 논문에서 제안한 구조로, ViM과 반대로 설계를 뒤집었다. 입력 시퀀스를 먼저 서로 다른 선형 레이어를 통해 Forward 특징 $x_f$와 Backward 특징 $x_b$로 분리한 뒤, **공유된 파라미터**를 가진 SSM 모듈을 통과시킨다. 이후 각각의 게이팅 레이어를 거쳐 Concatenation 하여 최종 출력을 생성한다.

### 3. Mamba의 4가지 역할 (4 Roles)
Mamba를 비디오 이해 시스템 내에서 다음과 같은 네 가지 역할로 정의하여 실험하였다.

1.  **Temporal Model**: 비디오의 시간적 흐름을 모델링하는 전체 모델 (예: ActionMamba).
2.  **Temporal Module**: 기존 모델의 시간적 어텐션을 대체하는 모듈 (예: TimeMamba의 시간 어댑터).
3.  **Multi-modal Interaction Network**: 텍스트와 비디오 간의 상호작용을 처리하는 네트워크 (예: UniVTG의 교차 모달리티 인코더).
4.  **Space-time Model**: 시공간 정보를 동시에 처리하는 통합 모델 (예: ViViM).

## 📊 Results

### 1. 정량적 성능 결과
*   **Temporal Action Localization**: HACS Segment 데이터셋에서 ActionMamba(DBM 적용)가 average mAP 44.56을 기록하며, Transformer 기반인 ActionFormer(43.34)를 유의미하게 앞질렀다.
*   **Temporal Action Segmentation**: GTEA 데이터셋에서 ASMamba가 ASFormer보다 우수한 성능을 보였으며, 이는 Mamba의 강력한 시간적 인코딩 능력을 시사한다.
*   **Dense & Paragraph Captioning**: PDVC 모델의 Deformable Attention을 DBM 기반 Mamba로 교체했을 때, 특히 YouCook2 데이터셋의 CIDEr 지표에서 성능 향상이 뚜렷하였다.
*   **Action Anticipation**: 인과적 추론(Causal inference) 작업에서 Mamba 블록이 Testra의 Causal self-attention보다 더 우수한 추론 능력을 보여주었다.
*   **Video Temporal Grounding (VTG)**: UniVTG의 Transformer를 Mamba로 대체한 결과, Qvhighlight 데이터셋에서 average mAP가 38.48에서 44.74로 크게 상승하였다.
*   **Zero-shot Retrieval & QA**: TimeMamba와 ViViM이 각각 TimeSformer와 ViT 대비 우수한 성능을 보였으며, 특히 EgoSchema의 초장기 비디오 QA에서 프레임 수가 증가할수록 TimeMamba의 성능 향상 폭이 더 컸다.

### 2. 효율성 분석 (Inference Speed)
프레임 수 증가에 따른 추론 속도 측정 결과, Mamba 계열 모델들의 선형 복잡도가 빛을 발하였다.
*   **TimeMamba vs TimeSformer**: 입력 프레임이 8192개를 넘어서는 시점부터 TimeMamba의 추론 속도가 TimeSformer를 추월하기 시작한다.
*   **ViViM vs ViT**: ViViM-T는 입력 프레임이 256개 이상일 때 Flash-attention을 적용한 ViT-T보다 더 빠른 속도를 보였으며, Flash-attention이 없는 경우에는 64개 프레임부터 더 효율적이었다.

## 🧠 Insights & Discussion

### 1. 강점 및 발견
*   **Linear Scalability**: Mamba의 가장 큰 강점은 긴 시퀀스에 대한 선형 복잡도이다. 이는 특히 수천 프레임에 달하는 초장기 비디오 이해 작업에서 Transformer의 Quadratic 복잡도 문제를 해결할 수 있는 실질적인 대안임을 입증하였다.
*   **DBM의 유효성**: DBM 블록은 방향성 편향(Directional bias)을 도입하고 동적 모델링 능력을 일부 조절함으로써, 특히 소규모 데이터셋에서 과적합을 방지하고 성능을 높이는 효과가 있었다.
*   **텍스트-비디오 융합 위치**: Multi-modal Mamba 실험 결과, 텍스트 토큰을 비디오 토큰의 **왼쪽(Left side)**에 배치했을 때 가장 좋은 성능이 나타났다. 이는 Mamba의 선형 스캔 메커니즘 특성상 텍스트 조건이 먼저 입력되는 것이 유리함을 시사한다.

### 2. 한계 및 비판적 해석
*   **Joint Space-Time Modeling의 위험성**: 실험 결과, 시공간을 통합하여 스캔하는 Space-time ViM 블록이 오히려 분리형(Divided) 구조보다 성능이 떨어지는 경우가 발견되었다. 이는 SSM의 스캔 방식이 사전 학습된 공간적 특징(Spatial feature)의 분포를 훼손할 수 있음을 의미하며, 비디오 모델링에서는 여전히 분리형 시공간 모델링이 더 안정적일 수 있다는 점을 시사한다.
*   **범용적 설계의 부재**: 본 논문은 Mamba를 '교체'하여 적용하는 방식을 취했으나, SSM의 특성을 극대화한 전용 비디오 아키텍처 설계에 대해서는 향후 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 Mamba(SSM)가 비디오 이해 분야에서 Transformer를 대체할 수 있는지 검증하기 위해 14개 모델과 12개 작업을 포함하는 **Video Mamba Suite**를 구축하였다. 실험 결과, Mamba는 대부분의 비디오 작업에서 Transformer와 대등하거나 더 우수한 성능을 보였으며, 특히 **프레임 수가 증가할수록 추론 속도 면에서 압도적인 효율성**을 나타냈다. 또한, 제안된 **DBM 블록**은 소규모 데이터셋에서 효과적임을 확인하였다. 이 연구는 향후 초장기 비디오 분석 및 효율적인 비디오 파운데이션 모델 구축에 중요한 기초 자료가 될 것으로 기대된다.