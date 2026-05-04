# SR-Mamba: Effective Surgical Phase Recognition with State Space Model

Rui Cao, Jiangliu Wang, and Yun-Hui Liu (2024)

## 🧩 Problem to Solve

수술 단계 인식(Surgical Phase Recognition)은 컴퓨터 보조 중재(Computer-Assisted Interventions, CAI) 시스템의 효율성과 안전성을 높이는 데 매우 중요하다. 수술 비디오 분석에서 가장 핵심적인 도전 과제 중 하나는 비디오 내에 존재하는 장거리 시간적 관계(long-distance temporal relationships)를 모델링하는 것이다.

일반적으로 수술 비디오는 수 시간에 걸쳐 진행되므로, 전체 시퀀스를 end-to-end 방식으로 학습시키는 것이 매우 어렵다. 이로 인해 기존 연구들은 공간적 특징 추출기(Spatial Feature Extractor)를 먼저 학습시킨 후, 추출된 특징을 바탕으로 시간적 모델을 학습시키는 '2단계 학습(two-step training)' 방식을 주로 사용해 왔다. 하지만 이러한 방식은 학습 과정이 복잡하고 하이퍼파라미터 튜닝에 많은 비용이 발생한다는 단점이 있다. 따라서 본 논문의 목표는 긴 시퀀스를 효율적으로 처리하면서도 높은 정확도를 제공하는 새로운 모델인 SR-Mamba를 제안하여 수술 단계 인식의 성능을 높이고 학습 프로세스를 단순화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 선형 스케일링(linear scalability) 특성을 가진 State Space Model(SSM)의 최신 구현체인 Mamba를 수술 비디오 분석에 도입하는 것이다.

1. **Bidirectional Mamba Decoder**: 수술 비디오의 전후 맥락을 모두 파악하기 위해 양방향 Mamba 디코더를 설계하여 시간적 관계 모델링 능력을 극대화하였다.
2. **Single-step Training**: Mamba의 효율적인 연산 능력을 활용하여, 기존의 2단계 학습 방식에서 벗어나 공간적 특징 추출기와 시간적 모델을 동시에 학습시키는 단일 단계(single-step) end-to-end 학습 프레임워크를 제안하였다.
3. **Auxiliary Anticipation Task**: 단순한 단계 인식뿐만 아니라, 향후 진행될 단계를 예측하는 '작업 예측(workflow anticipation)'을 보조 작업(auxiliary task)으로 추가하여 인식 성능을 더욱 향상시켰다.

## 📎 Related Works

수술 워크플로우 인식 분야에서는 시간적 관계를 분석하기 위해 다음과 같은 모델들이 사용되어 왔다.

- **RNN/LSTM**: 초기 표준 아키텍처로 CNN-LSTM 구조가 사용되었으나, 매우 긴 시퀀스의 의존성을 학습하는 데 한계가 있다.
- **Temporal Convolutional Networks (TCN)**: LSTM을 대체하여 성능 향상을 가져왔으며, 다단계 TCN 구조 등이 제안되었다.
- **Transformers**: Self-attention 메커니즘을 통해 성능을 더욱 높였으나, 시퀀스 길이에 따라 연산 복잡도가 제곱으로 증가하는 문제가 있다.

최근 제안된 State Space Models (SSMs)와 특히 Mamba는 컨볼루션 기반의 연산 방식을 통해 Transformer와 유사한 성능을 내면서도 시퀀스 길이에 대해 선형적인 계산 효율성을 제공한다. SR-Mamba는 이러한 Mamba의 장점을 활용하여 기존 Transformer 기반 모델들의 연산 부담을 줄이면서도 장거리 의존성을 효과적으로 포착하고자 한다.

## 🛠️ Methodology

### State Space Models (SSM) 및 Mamba 배경

Mamba는 선형 상미분 방정식(Ordinary Differential Equation, ODE)을 사용하여 입력 시퀀스 $x(t)$를 은닉 상태 $h(t)$를 거쳐 출력 $y(t)$로 매핑한다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A$는 상태 행렬, $B$는 입력 행렬, $C$는 출력 행렬이다. 실제 계산을 위해 Zero-Order Hold (ZOH) 기법을 통해 이산화(discretization)하며, 다음과 같이 표현된다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

이때 이산화 파라미터는 $\bar{A} = \exp(A\Delta)$, $\bar{B} = (\bar{A}-I)^{-1}(\exp(A\Delta)-I)B\Delta$로 계산된다. Mamba는 $\Delta, B, C$를 입력의 함수로 정의하여 선택적(selective) 모델링을 가능하게 하며, 하드웨어 최적화된 Parallel Scan 알고리즘을 통해 빠른 학습과 추론을 지원한다.

### SR-Mamba 아키텍처

전체 파이프라인은 다음과 같은 흐름으로 구성된다.

1. **Spatial Feature Extraction**: 입력 RGB 비디오 시퀀스 $X$에 대해 경량화된 ResNet34를 사용하여 공간적 임베딩 $F$를 추출한다. $F \in \mathbb{R}^{T \times 512}$ 형태를 가진다. 이때 메모리 효율을 위해 ResNet34의 앞 두 레이어는 동결(freeze)하고 나머지는 학습 가능하다.
2. **Bidirectional Mamba Decoder**: 추출된 특징 $F$는 양방향 Mamba 디코더 $\Psi$로 입력된다.
    - **Forward & Backward Paths**: 입력 시퀀스를 정방향과 역방향 두 경로로 처리한다.
    - **Processing**: 각 경로에서 1-D Convolution을 적용하고, Linear Projection을 통해 SSM 파라미터($B, C, \Delta$)를 생성한다.
    - **Merging**: 정방향 출력 $y_{\text{forward}}$와 역방향 출력 $y_{\text{backward}}$를 SiLU 활성화 함수 기반의 게이팅 메커니즘($z'$)으로 결합하여 최종 출력 $y$를 생성한다.
3. **Output Generation**: 모델은 최종적으로 두 가지 결과를 출력한다.
    - $\hat{R}$: 수술 단계 인식(Workflow Recognition)
    - $\hat{A}$: 수술 단계 예측(Workflow Anticipation, 남은 시간 예측)

### 학습 및 최적화

손실 함수는 인식 작업을 위한 Cross-Entropy loss($L_r$)와 예측 작업을 위한 SmoothL1 loss($L_a$)의 가중 합으로 정의된다.

$$L_r = -\sum_{t=1}^T r_t \log(\hat{r}_t)$$
$$L_a = -\sum_{t=1}^T \text{SmoothL1}(\hat{a}_t, a_t)$$
$$L = \lambda_1 L_r + \lambda_2 L_a$$

시퀀스 길이가 최대 길이 $N_{\max} = 2048$을 초과하는 경우, 단계 전환점(phase transition)과 그 인접 프레임을 우선적으로 보존하고 나머지 구간을 균등 샘플링하는 전략을 사용하여 계산 부하를 관리한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80 (담낭 절제술 비디오 80개), CATARACTS (백내장 수술 비디오 50개).
- **지표**: Cholec80에서는 Accuracy(AC), Precision(PR), Recall(RE), Jaccard Index(JA)를 사용하였고, CATARACTS에서는 F1-score를 사용하였다.
- **환경**: NVIDIA RTX 3090 GPU 1대, ResNet34 backbone (ImageNet-1k 사전 학습).

### 정량적 결과

- **Cholec80**: SR-Mamba는 Accuracy 92.6% $\pm$ 8.6, Jaccard 81.5% $\pm$ 8.6를 기록하여 비교 대상 모델 중 가장 높은 성능을 보였다. 특히 도구 레이블(tool labels)을 사용한 MTRCNet-CL보다 AC 기준 3.4%p 높았으며, LoViT와 대등하거나 더 나은 성능을 보이면서도 파라미터 수(21.3M)는 매우 효율적이다.
- **CATARACTS**: 단계 레이블만을 사용한 방법들 중 가장 높은 평균 F1 score(0.892)를 달성하였다.

### 절제 실험 (Ablation Study)

- **시퀀스 길이**: 학습 시퀀스 길이가 512에서 2048로 증가함에 따라 AC와 JA가 모두 향상되어, 긴 시퀀스 모델링의 이점이 확인되었다.
- **보조 작업**: Anticipation loss를 추가했을 때 AC와 JA가 약 0.5%p 상승하였다.
- **아키텍처 비교**:
  - 단방향(Vanilla) Mamba보다 양방향(Bi-M) 구조가 훨씬 뛰어난 성능을 보여, 수술 비디오의 전역적 맥락 파악에 양방향성이 필수적임을 입증하였다.
  - 단일 단계 학습(Single-step)을 적용한 ResNet34+Bi-M 구조가 2단계 학습을 적용한 ResNet50+Bi-M보다 높은 성능을 보였다.

## 🧠 Insights & Discussion

본 논문의 결과는 강력한 시간적 모델링 능력이 공간적 특징 추출기의 복잡도를 보완할 수 있음을 시사한다. 대부분의 기존 SOTA 모델들이 ResNet50나 Vision Transformer(ViT) 같은 무거운 백본을 사용하는 반면, SR-Mamba는 더 가벼운 ResNet34를 사용하고도 더 높은 성능을 냈다. 이는 Bidirectional Mamba Decoder가 수술 비디오의 장거리 의존성을 매우 효과적으로 포착하고 있음을 의미한다.

또한, 기존의 2단계 학습 방식은 특징 추출 단계에서 정보 손실이 발생하거나 전체 최적화가 어려울 수 있는데, Mamba의 선형 복잡도 덕분에 가능한 '단일 단계 end-to-end 학습'이 실제로 더 높은 정확도와 단순한 파이프라인을 제공한다는 점이 고무적이다. 다만, 본 논문에서 제시한 샘플링 전략이 매우 긴 비디오의 모든 세부 사항을 완벽하게 보존하는지에 대해서는 추가적인 논의가 필요할 수 있다.

## 📌 TL;DR

SR-Mamba는 수술 단계 인식의 고질적인 문제인 '장거리 시간적 관계 모델링'을 해결하기 위해 **양방향 Mamba 디코더**와 **단일 단계(single-step) end-to-end 학습** 방식을 제안한 모델이다. 이 모델은 가벼운 ResNet34 백본과 예측 보조 작업을 결합하여 Cholec80 및 CATARACTS 데이터셋에서 SOTA 성능을 달성하였으며, 특히 연산 효율성을 유지하면서도 Transformer 수준의 성능을 낼 수 있음을 보여주었다. 이는 향후 초장거리 비디오 분석 분야에서 SSM 기반 모델들이 중요한 역할을 할 가능성을 제시한다.
