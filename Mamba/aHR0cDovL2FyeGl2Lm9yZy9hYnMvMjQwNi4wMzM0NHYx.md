# Audio Mamba: Bidirectional State Space Model for Audio Representation Learning

Mehmet Hamza Erol, Arda Senocak, Jiu Feng, Joon Son Chung (2024)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 오디오 분류 작업에서 기존의 표준 모델인 Audio Spectrogram Transformer (AST)가 가진 계산 효율성 문제이다. AST는 Self-attention 메커니즘에 의존하는데, 이는 입력 시퀀스 길이에 대해 이차 복잡도($O(n^2)$)의 계산 비용을 발생시킨다. 따라서 오디오 시퀀스의 길이가 길어질수록 메모리 사용량이 급격히 증가하고 추론 속도가 느려지는 한계가 있다.

본 논문의 목표는 Self-attention을 완전히 제거하고, 선형 복잡도($O(n)$)를 가지는 State Space Model (SSM), 특히 Mamba 아키텍처를 오디오 분류에 도입하여 성능은 유지하거나 향상시키면서 계산 효율성을 획기적으로 높이는 것이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 기여는 오디오 분류를 위해 설계된 최초의 Self-attention-free 모델인 **Audio Mamba (AuM)**를 제안한 것이다. 핵심 설계 아이디어는 다음과 같다.

1. **Bidirectional SSM의 도입**: Mamba의 기본 단방향 스캔 방식은 인과적(Causal) 데이터에는 적합하지만, 2D 형태의 스펙트로그램 데이터에서는 전역 문맥을 파악하기 위해 양방향 정보가 필요하다. 이를 위해 Forward 및 Backward 방향으로 모두 스캔하는 양방향 SSM 구조를 채택하였다.
2. **효율적인 전역 문맥 모델링**: Self-attention 없이도 SSM을 통해 선형 시간 복잡도로 긴 오디오 시퀀스의 전역적 특징을 캡처할 수 있도록 설계하였다.
3. **전략적 CLS 토큰 배치**: 양방향 스캔의 특성을 고려하여, 분류를 위한 $[CLS]$ 토큰을 시퀀스의 시작이나 끝이 아닌 **중앙(Middle)**에 배치함으로써 학습 성능을 최적화하였다.

## 📎 Related Works

기존의 오디오 분류는 CNN 기반 방법론에서 Transformer 기반의 AST로 패러다임이 전환되었다. AST는 Vision Transformer (ViT)의 원리를 오디오 스펙트로그램에 적용하여 우수한 성능을 보였으나, 앞서 언급한 이차 복잡도 문제가 항상 수반되었다.

최근 자연어 처리 및 비전 분야에서는 Mamba와 같은 SSM이 Transformer의 대안으로 부상하고 있다. 특히 Vision Mamba (ViM)는 양방향 SSM을 사용하여 이미지의 전역 문맥을 효율적으로 모델링할 수 있음을 보여주었다. 본 연구는 이러한 ViM의 통찰을 오디오 도메인으로 확장하여, AST의 패치화(Patchification) 구조와 Mamba의 효율적인 시퀀스 모델링을 결합하였다.

## 🛠️ Methodology

### 전체 파이프라인

AuM의 전체 흐름은 다음과 같다:

1. **입력 변환**: 오디오 파형을 스펙트로그램 $X \in \mathbb{R}^{F \times T}$ (주파수 $\times$ 시간)로 변환한다.
2. **Patchification**: 스펙트로그램을 $p \times p$ 크기의 정사각형 패치 $N$개로 나누고, 이를 1차원 벡터로 펼친 후 선형 투영(Linear Projection)을 통해 $D$-차원 임베딩 공간으로 매핑한다.
3. **토큰 구성**: 임베딩된 패치 시퀀스의 중앙에 학습 가능한 $[CLS]$ 토큰을 삽입하고, 위치 임베딩(Positional Embedding)을 더해 최종 토큰 시퀀스 $T \in \mathbb{R}^{(N+1) \times D}$를 생성한다.
4. **AuM Encoder**: $L$개의 AuM 블록을 거쳐 특징을 추출한다.
5. **분류**: 인코더를 통과한 중앙 $[CLS]$ 토큰의 표현을 분류 헤드(Classification Head)에 전달하여 최종 클래스를 예측한다.

### State Space Model (SSM) 및 Mamba

SSM은 연속적인 시스템을 모델링하는 선형 시불변 시스템으로, 다음과 같은 방정식으로 표현된다:
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$
여기서 $A \in \mathbb{R}^{N \times N}, B \in \mathbb{R}^{N \times D}, C \in \mathbb{R}^{D \times N}$이다. 딥러닝 적용을 위해 이를 이산화(Discretization)하면 다음과 같은 형태가 된다:
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

Mamba는 여기서 더 나아가 파라미터 $A, B, \Delta$를 입력 값에 따라 변하는 **시변(Time-variant)** 형태로 변환하여, 모델이 입력 데이터에 따라 필요한 정보를 선택적으로 업데이트할 수 있게 한다.

### AuM Block의 세부 구조

AuM은 Mamba의 단방향 스캔 한계를 극복하기 위해 다음과 같은 설계를 사용한다:

- **Forward Conv1D**: 먼저 입력 시퀀스에 1차원 합성곱을 적용하여 특징을 추출한다.
- **Bidirectional SSM**: 추출된 동일한 특징을 바탕으로 **Forward SSM**과 **Backward SSM**을 각각 적용한다. 이는 Transformer의 Self-attention처럼 시퀀스의 전체 문맥을 양방향에서 동시에 고려할 수 있게 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: AudioSet (Full/Balanced), VGGSound, VoxCeleb, Speech Commands V2, Epic-Sounds 등 6개 벤치마크를 사용하였다.
- **비교 대상**: AST-B/16 (Base model)과 AuM-B/16을 비교하였다.
- **지표**: 다중 레이블 데이터셋(AudioSet)은 mAP를, 단일 레이블 데이터셋은 Top-1 Accuracy(Acc)를 사용하였다.

### 주요 결과

1. **분류 성능**: Table 1에 따르면, AuM-B/16은 대부분의 데이터셋에서 AST-B/16과 비슷하거나 더 높은 성능을 보였다. 특히 AudioSet Balanced (mAP 13.28% vs 10.41%)와 VoxCeleb (Acc 42.58% vs 37.25%)에서 뚜렷한 향상을 보였다.
2. **계산 효율성**:
    - **메모리**: AuM은 시퀀스 길이에 따라 메모리 사용량이 선형적으로 증가한다. AST-B가 20초 분량의 오디오에서 메모리 부족(OoM)이 발생하는 반면, AuM-B는 80초 분량까지 처리 가능하다.
    - **속도**: 토큰 수가 4096개일 때, AuM은 AST보다 추론 속도가 약 1.6배 빠르며, 시퀀스 길이가 길어질수록 이 격차는 더 벌어진다.
3. **Ablation Study**:
    - **방향성**: 단방향 SSM보다 양방향 SSM이 전반적으로 우수한 성능을 보였다.
    - **Conv1D 구성**: 각 방향마다 별도의 Conv1D를 두는 것(Bi-Bi)보다, 하나의 Forward Conv1D 결과를 공유하여 양방향 SSM으로 스캔하는 것(Fo-Bi)이 더 효과적이었다.
    - **CLS 위치**: 양방향 구조에서는 $[CLS]$ 토큰을 중앙에 배치하는 것이 가장 성능이 좋았다. 특히 단방향 SSM에서 토큰을 앞에 배치하면 이후의 정보가 반영되지 않아 성능이 급격히 저하되는 현상이 관찰되었다.

## 🧠 Insights & Discussion

### 강점 및 의의

AuM은 오디오 분류 분야에서 Self-attention의 계산 비용 문제를 해결하면서도 성능 손실이 없음을 입증하였다. 특히 긴 오디오 시퀀스를 처리해야 하는 실제 환경에서 AuM의 선형 복잡도는 매우 강력한 이점이 된다. 또한 AST와 유사한 패치화 구조를 유지함으로써, 기존 Transformer 기반 오디오 모델들의 유연성을 그대로 계승하였다.

### 한계 및 논의사항

- **사전 학습 가중치의 부재**: AST는 ImageNet으로 사전 학습된 ViT 가중치를 사용할 수 있어 성능을 크게 높일 수 있으나, Mamba 계열은 이에 상응하는 대규모 사전 학습 가중치가 공개된 경우가 적어 공정한 비교에 제약이 있었다. 다만, AudioSet을 이용한 도메인 내 사전 학습(In-domain pre-training) 결과에서는 AuM이 AST를 압도하는 경향을 보여, 적절한 가중치가 제공된다면 더 높은 잠재력을 가졌음을 시사한다.
- **가정**: 본 연구는 스펙트로그램을 이미지처럼 패치화하여 처리하는 방식에 의존하고 있다.

## 📌 TL;DR

본 논문은 오디오 분류를 위해 Self-attention을 제거하고 양방향 State Space Model (SSM)을 도입한 **Audio Mamba (AuM)**를 제안한다. AuM은 계산 복잡도를 $O(n^2)$에서 $O(n)$으로 낮추어 긴 오디오 시퀀스 처리에 있어 압도적인 메모리 및 속도 효율성을 보이며, 성능 면에서도 기존의 AST 모델과 대등하거나 이를 상회한다. 이 연구는 향후 대규모 오디오 데이터셋을 활용한 자기지도학습(SSL)이나 멀티모달 학습에서 효율적인 오디오 백본 네트워크로 활용될 가능성이 매우 높다.
