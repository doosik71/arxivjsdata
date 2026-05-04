# MambaEVT: Event Stream based Visual Object Tracking using State Space Model

Xiao Wang, Chao Wang, Shiao Wang, Xixi Wang, Zhicheng Zhao, Lin Zhu, Bo Jiang (2024)

## 🧩 Problem to Solve

본 논문은 이벤트 카메라(Event Camera) 기반의 시각적 객체 추적(Visual Object Tracking, VOT)에서 발생하는 두 가지 주요 문제를 해결하고자 한다.

첫째는 **높은 계산 복잡도(High Computational Complexity)**이다. 최근의 이벤트 기반 추적 알고리즘들은 Vision Transformer(ViT)를 백본 네트워크로 채택하고 있는데, Transformer의 Self-attention 메커니즘은 $O(N^2)$의 계산 복잡도를 가진다. 이는 실제 하드웨어에 추적 알고리즘을 배포할 때 매우 비효율적이며 성능 병목 현상을 야기한다.

둘째는 **정적 템플릿(Static Target Template)**의 한계이다. 기존의 Siamese 추적 프레임워크는 초기 프레임에서 추출한 정적 템플릿을 사용하여 타겟을 국지화한다. 이러한 방식은 단순한 시나리오에서는 잘 작동하지만, 장기 추적이나 타겟의 외형이 심하게 변하는 환경에서는 추적 성능이 급격히 저하되는 문제가 있다.

따라서 본 논문의 목표는 추적 정확도와 계산 비용 사이의 최적의 균형(trade-off)을 맞춘 새로운 이벤트 기반 시각적 추적 프레임워크를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 선형 복잡도를 가진 **State Space Model(SSM)**, 특히 Mamba 아키텍처를 추적 시스템의 백본과 템플릿 업데이트 모듈에 도입하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Mamba 기반 추적 프레임워크(MambaEVT) 제안**: Vision Mamba를 백본으로 사용하여 특성 추출, 상호작용 및 융합을 동시에 수행함으로써 계산 비용을 낮추면서도 높은 정확도를 달성하였다. 이는 이벤트 카메라를 이용한 최초의 Mamba 기반 추적 프레임워크이다.
2.  **Memory Mamba를 이용한 동적 템플릿 업데이트 전략**: 타겟의 외형 변화에 대응하기 위해 학습 가능한 Memory Mamba 네트워크를 제안하였다. 템플릿 라이브러리에서 샘플의 다양성을 고려하여 동적 템플릿을 생성함으로써 강건성을 높였다.
3.  **효율성 및 성능 검증**: EventVOT, VisEvent, FE240hz 등 대규모 데이터셋에서 실험을 통해 기존 Transformer 기반 모델보다 훨씬 적은 파라미터 수로 경쟁력 있는 성능을 보임을 입증하였다.

## 📎 Related Works

### 1. 이벤트 기반 시각적 객체 추적
기존 연구들은 RGB-Event 멀티모달 융합 방식(예: VisEvent)이나 이벤트 전용 방식(예: HDETrack)으로 나뉜다. 최근에는 Transformer 기반의 네트워크가 주류를 이루고 있으나, 본 논문은 Transformer의 높은 연산 비용 문제를 지적하며 순수 Mamba 스타일의 접근 방식을 택하여 차별점을 두었다.

### 2. 동적 템플릿 업데이트
외형 변화에 대응하기 위해 LSTM 기반의 메모리 네트워크나 고정 크기의 외부 메모리를 사용하는 Memformer, STARK 등이 제안되었다. 기존의 동적 업데이트 메커니즘은 주로 테스트 단계에서만 작동하는 비학습형 방식이 많았으나, 본 논문의 Memory Mamba는 학습 가능(trainable)한 모듈로 설계되어 훈련 데이터를 더 잘 활용할 수 있다.

### 3. State Space Model (SSM)
S4 모델에서 시작하여 선택적 스캔 메커니즘을 도입한 Mamba가 등장하였으며, 이를 시각 영역으로 확장한 Vim(Vision Mamba)과 VMamba가 제안되었다. 본 연구는 이러한 Vision Mamba의 효율적인 시퀀스 모델링 능력을 추적 작업에 적용하였다.

## 🛠️ Methodology

### 1. Preliminary: Mamba
Mamba의 기초가 되는 State Space Model은 연속 시스템에서 1차원 함수 $x(t) \in \mathbb{R}$를 은닉 상태 $h(t) \in \mathbb{R}^N$를 통해 $y(t) \in \mathbb{R}$로 매핑한다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

이를 컴퓨터 시스템에서 처리하기 위해 Zero-Order Hold (ZOH) 방법을 통해 이산화(discretization)하면 다음과 같이 표현된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

이산화된 시스템의 수식은 다음과 같다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

### 2. 전체 파이프라인
MambaEVT는 Siamese 추적 프레임워크를 따른다.
1.  **입력 표현**: 이벤트 스트림을 고정 길이의 이벤트 프레임으로 변환한다. 초기 템플릿 $E^0_z$와 탐색 영역 $E^t_x$를 추출하여 토큰화한다.
2.  **Mamba 기반 백본**: 정적 템플릿 $T^0_z$, 동적 특성 $E^t_d$, 탐색 영역 $S^t_x$를 결합하여 $E^f = [T^0_z, E^t_d, S^t_x]$를 구성한다. 이를 Vision Mamba(Vim) 블록에 통과시켜 특성을 추출하고 상호작용시킨다.
    $$E^l_f = \text{Vim}(E^{l-1}_f) + E^{l-1}_f, \quad l=\{1, 2, \dots, L\}$$
    방향성 간섭을 없애기 위해 순방향과 역방향 스캔을 모두 사용한다.
3.  **추적 헤드(Tracking Head)**: FCN(Fully Convolutional Network)으로 구성되며, 타겟 분류 점수 맵, 로컬 오프셋(local offset), 정규화된 바운딩 박스 크기를 출력한다.

### 3. 동적 템플릿 업데이트 전략
외형 변화를 해결하기 위해 **Memory Mamba**를 도입하였다.
-   **템플릿 라이브러리**: 단기 메모리(ST)와 장기 메모리(LT)로 구분하여 관리한다. ST는 큐(queue) 형태로 고정 간격 샘플링을 수행하며, LT는 유사도 기반 전략으로 장기 기억을 처리한다.
-   **다양성 측정**: 새로운 템플릿을 LT에 추가할 때, Gram 행렬 $G$의 행렬식(determinant)을 사용하여 다양성을 측정한다.
    $$G(z_1, \dots, z_n) = \begin{bmatrix} z_1 \star z_1 & \dots & z_1 \star z_n \\ \vdots & \ddots & \vdots \\ z_n \star z_1 & \dots & z_n \star z_n \end{bmatrix}$$
    여기서 $\star$는 피어슨 선형 상관 계수(Pearson linear correlation)를 의미한다.
-   **동적 템플릿 생성**: 메모리 라이브러리의 시퀀스 $Z = [z_1, z_2, \dots, z_m]$를 Memory Mamba 네트워크에 입력하고, 마지막 $N$개의 패치를 융합된 동적 템플릿 $E^d$로 사용한다.

### 4. 손실 함수
최종 손실 함수는 Focal loss, $L_1$ loss, GIoU loss의 가중치 합으로 정의된다.
$$\mathcal{L} = \lambda_1 L_1 + \lambda_2 L_{focal} + \lambda_3 L_{GIoU}$$
실험적으로 $\lambda_1=5, \lambda_2=1, \lambda_3=2$로 설정하였다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: EventVOT, VisEvent, FE240hz.
-   **지표**: Precision Rate (PR), Normalized Precision Rate (NPR), Success Rate (SR).
-   **구현**: PyTorch 기반, NVIDIA RTX 3090 GPU 사용. Vim-S 모델을 ImageNet-1K로 사전 학습 후 50 epoch 동안 훈련.

### 2. 주요 결과
-   **EventVOT**: MambaEVT는 SR 56.5, PR 56.7, NPR 65.5를 기록하였다. 특히 OSTrack(ViT 기반)과 비교했을 때 SR을 1.1pt 개선하면서도 파라미터 수를 **92.1M에서 29.3M으로 대폭 줄였다**.
-   **FE240hz**: SR 58.09, PR 91.97을 달성하였다. AQATrack 같은 모델보다 절대 성능은 약간 낮지만, 파라미터 효율성 면에서 압도적인 우위를 보였다.
-   **VisEvent**: MambaEVT-P(파라미터 공유 안 함) 버전이 SR 37.2, PR 51.8로 더 높은 성능을 보였으며, 이는 가변 길이 시퀀스 처리 시 더 많은 파라미터가 필요함을 시사한다.

### 3. 분석 (Ablation Study)
-   **구성 요소**: Memory Mamba(MM)와 Memory Library(ML)를 모두 적용했을 때 가장 높은 성능을 보였다.
-   **백본 비교**: VMamba-S가 성능은 가장 좋았으나, 연산 자원과 성능의 균형을 위해 Vim-S를 최종 선택하였다.
-   **메모리 용량**: LT 용량이 증가할수록 성능이 향상되었으며, LT=16, ST=6일 때 최적의 성능을 보였다.
-   **동적 템플릿 수**: 템플릿 수가 11개일 때까지 성능이 향상되다가 이후 감소하는 경향을 보였다.

## 🧠 Insights & Discussion

### 강점
MambaEVT는 기존의 Transformer 기반 추적 모델들이 가진 고비용 구조를 SSM으로 대체하여 **파라미터 효율성**을 극대화하였다. 특히 학습 가능한 Memory Mamba를 통해 동적 템플릿을 생성함으로써, 단순한 정적 템플릿의 한계를 극복하고 외형 변화에 강건하게 대응할 수 있음을 보여주었다.

### 한계 및 미해결 질문
1.  **추론 속도(FPS)**: 파라미터 수는 줄었으나 실제 실행 속도(FPS)는 상대적으로 낮게 나타났다. 이는 Mamba 모델의 구현 최적화가 더 필요함을 의미한다.
2.  **학습 전략**: Mamba는 본래 매우 긴 시퀀스 모델링에 강점이 있으나, 현재의 훈련 프로세스가 이러한 능력을 완전히 활용하지 못했을 가능성이 있다.

### 비판적 해석
본 논문은 효율성(Parameter efficiency)을 강조하지만, 실제 실시간 시스템에서 중요한 것은 파라미터 수보다는 추론 속도(Latency/FPS)이다. 저자들 스스로도 FPS가 낮음을 인정하고 있으며, 이는 향후 Spiking Neural Networks(SNN) 도입 등을 통해 해결해야 할 핵심 과제로 보인다.

## 📌 TL;DR

본 논문은 이벤트 카메라 기반 객체 추적에서 Transformer의 높은 계산 비용과 정적 템플릿의 한계를 해결하기 위해 **선형 복잡도의 Mamba(SSM) 백본과 Memory Mamba 기반 동적 템플릿 업데이트 전략**을 제안하였다. 실험 결과, 기존 SOTA 모델 대비 파라미터 수를 1/3 수준으로 줄이면서도 대등한 추적 성능을 달성하였다. 이 연구는 이벤트 기반 비전 시스템의 경량화 및 효율적 시퀀스 모델링 가능성을 제시하여 향후 임베디드 환경의 고속 추적 연구에 기여할 것으로 기대된다.