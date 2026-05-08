# Autoregressive Pretraining with Mamba in Vision

Sucheng Ren, Xianhang Li, Haoqin Tu, Feng Wang, Fangxun Shu, Lei Zhang, Jieru Mei, Linjie Yang, Peng Wang, Heng Wang, Alan Yuille, Cihang Xie (2024)

## 🧩 Problem to Solve

본 논문은 최근 컴퓨터 비전 분야에서 새로운 백본(backbone)으로 주목받고 있는 State Space Model(SSM) 기반의 Mamba 아키텍처가 가진 확장성(scalability) 및 전이 가능성(transferability)의 한계를 해결하고자 한다.

기존의 Vision Mamba(Vim)와 같은 모델들은 주로 지도 학습(supervised learning) 환경에서 훈련되었는데, 이러한 방식은 모델의 크기를 키웠을 때 성능이 정체되거나(performance plateauing), 매우 큰 모델 사이즈에서는 훈련이 붕괴(training collapse)되는 현상이 발생한다. 따라서 저자들은 거대 모델에서도 안정적으로 성능을 확장할 수 있도록, 자연어 처리(NLP) 분야에서 검증된 자기지도 학습 방식인 자기회귀 사전학습(autoregressive pretraining)을 Mamba 아키텍처에 도입하여 시각적 표현 학습 능력을 강화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 단방향 재귀 구조(unidirectional recurrent structure)가 자기회귀 모델링의 특성과 완벽하게 일치한다는 점을 활용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **ARM (Autoregressive Mamba) 제안**: Mamba 아키텍처에 최적화된 자기회귀 사전학습 프레임워크를 제안하여, 지도 학습 대비 높은 정확도와 강력한 확장성을 확보하였다.
2. **클러스터 기반 예측 단위(Cluster-based Prediction Unit)**: 단순히 픽셀이나 패치 단위로 예측하는 대신, 공간적으로 인접한 패치들을 그룹화한 '클러스터' 단위를 도입하여 학습 효율과 성능을 극대화하였다.
3. **MambaMLP 아키텍처 설계**: Mamba를 Token Mixer로, MLP를 Channel Mixer로 사용하는 새로운 블록 구조를 제안하였으며, 사전학습(단방향 스캔)과 미세조정(다방향 스캔) 단계에 따라 스캔 전략을 다르게 적용하여 효율성을 높였다.
4. **최대 규모의 Vision Mamba 구현**: ARM을 통해 지금까지 가장 큰 규모의 Vision Mamba 모델인 ARM-H를 성공적으로 훈련시켰으며, ImageNet에서 85.5%(384x384 입력 시)라는 최고 성능을 달성하였다.

## 📎 Related Works

**State Space Model (SSM) 및 Mamba**
SSM은 Transformer의 연산 복잡도를 선형적으로 줄이면서도 긴 시퀀스 모델링이 가능한 대안으로 제시되었다. 특히 Mamba는 선택적 스캐닝(selective scanning) 기법을 통해 기존 SSM의 한계를 극복하고 NLP에서 Transformer와 대등한 성능을 보였다.

**Vision Mamba의 기존 접근법**
Vim은 양방향 스캔을 통해 시각 데이터의 방향성 문제를 해결하려 했으며, VMamba는 2D 컨볼루션과 CrossScan 모듈을 결합한 계층적 구조를 사용한다. 그러나 이러한 모델들은 대부분 지도 학습에 의존하여 확장성 문제에 직면해 있다.

**자기지도 시각 표현 학습**
MAE와 같은 마스크 이미지 모델링(MIM)이나 대조 학습(Contrastive Learning)이 널리 사용되어 왔으나, 본 논문은 NLP의 표준인 자기회귀 학습을 Mamba에 적용한 첫 번째 연구라는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Mamba 기초 이론

Mamba는 연속적인 상태 공간 방정식에서 시작하며, 다음과 같은 선형 상미분 방정식(ODE)으로 정의된다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$
여기서 $x(t)$는 입력, $h(t)$는 은닉 상태, $y(t)$는 출력이다. 이를 이산화(discretization)하면 다음과 같은 재귀 형태로 변환된다.
$$h'_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = \bar{C}h_t$$
이 구조는 입력을 순차적으로 처리하므로, 다음 토큰을 예측하는 자기회귀 모델링의 단방향 특성과 정확히 일치한다.

### 2. 자기회귀 사전학습 (ARM)

**예측 단위 (Prediction Unit)**
이미지를 1D 시퀀스로 변환할 때, 저자들은 다음 세 가지 단위를 비교 분석하였다.

- **Pixel-based**: 개별 픽셀을 예측. 연산량이 너무 많아 저해상도 이미지에서만 가능하다.
- **Patch-based**: $16 \times 16$ 패치 단위로 예측.
- **Cluster-based (제안)**: 인접한 패치들을 묶어 더 큰 클러스터($64 \times 64$ 크기가 최적)를 형성하여 예측 단위로 사용한다.

**예측 순서 (Prediction Order)**
2D 이미지를 1D 시퀀스로 나열하는 방법으로 Row-first, Column-first, Random 순서를 실험하였으며, 그 결과 **Row-first and Forward** (행 우선 정방향) 방식이 가장 효율적이고 안정적임을 확인하였다.

**손실 함수 (Loss Function)**
클러스터 $c_i$에 대해, 이전 클러스터들의 시퀀스를 통해 다음 클러스터를 예측하며 평균 제곱 오차(MSE) 손실을 최소화한다.
$$L^{ARM} = \sum_{i=1}^{n-1} \| f([c_1, \dots, c_i]) - c_{i+1} \|^2$$
여기서 $f(\cdot)$는 Mamba 모델이며, 타겟은 정규화된 픽셀 값이다.

### 3. MambaMLP 아키텍처

본 논문은 Transformer의 구조를 참고하여 **MambaMLP** 블록을 설계하였다.

- **Token Mixer**: Mamba 레이어가 담당한다.
- **Channel Mixer**: SwiGLU 기반의 MLP 레이어가 담당한다.

특히 학습 단계에 따라 Mamba의 스캔 방식을 다르게 적용한다.

- **사전학습 단계**: 자기회귀 특성에 맞춰 **1-scan (단방향)** 구조를 사용하여 훈련 속도를 극대화한다.
- **미세조정(Finetuning) 단계**: 전역 정보를 더 잘 파악하기 위해 **4-scan (다방향)** 구조로 변경하여 성능을 높인다.

## 📊 Results

### 1. ImageNet-1K 성능 비교

ARM은 다양한 모델 사이즈에서 지도 학습 기반 모델보다 우수한 성능을 보였다.

- **Base-size**: ARM-B는 83.2%의 정확도를 달성하여, 지도 학습 기반의 MambaMLP-B(81.2%)보다 2.0% 높다.
- **Huge-size**: ARM-H는 85.0% (입력 크기 $384 \times 384$ 시 85.5%)를 기록하며, 기존의 모든 Vision Mamba 변체들을 능가하였다. 특히 지도 학습 기반의 Vim-H는 이 규모에서 훈련 붕괴(collapsed)가 발생했으나, ARM은 성공적으로 학습되었다.

### 2. 강건성(Robustness) 및 일반화 능력

Out-of-domain 데이터셋(ImageNet-A, R, S 등)에 대한 실험 결과, ARM은 지도 학습 모델보다 월등한 강건성을 보였다. 예를 들어 ARM-B는 Vim-B 대비 ImageNet-A에서 4.4%의 성능 향상을 보였다.

### 3. 효율성 분석

사전학습 전략별 훈련 비용을 비교한 결과, ARM은 MAE나 대조 학습보다 훨씬 빠른 훈련 속도를 보였다. Base 모델 기준, ARM의 훈련 시간은 약 34시간으로, 다른 전략 대비 2배에서 10배까지 빠르다. 이는 Mamba의 단방향 재귀 구조를 그대로 활용했기 때문이다.

### 4. 어블레이션 연구 (Ablation Study)

- **클러스터 크기**: $64 \times 64$ 크기의 클러스터를 사용할 때 성능이 가장 높았다.
- **예측 순서**: 정해진 순서(Row/Column first) 간의 차이는 적었으나, 랜덤 순서(Random permutation)를 사용할 경우 성능이 크게 하락하였다.
- **예측 타겟**: 정규화된 픽셀(Normed Pixel)을 MSE 손실로 학습하는 것이 dVAE의 이산 토큰을 사용하는 것보다 성능이 좋았다.

## 🧠 Insights & Discussion

본 연구는 Mamba 아키텍처가 가진 본질적인 단방향성이 자기회귀 학습과 매우 잘 어울린다는 점을 실험적으로 입증하였다. 가장 주목할 점은 **사전학습이 Mamba의 확장성(Scaling) 문제를 해결하는 핵심 열쇠**가 되었다는 것이다. 지도 학습으로는 불가능했던 Huge-size 모델의 안정적인 훈련이 자기회귀 사전학습을 통해 가능해졌다.

또한, 단순한 아키텍처 변경보다 **데이터를 어떻게 시퀀스로 구성하느냐(예측 단위 및 순서)**가 성능에 큰 영향을 미친다는 점을 밝혔다. 특히 클러스터 단위의 예측이 픽셀이나 패치 단위보다 효율적이라는 점은 시각 데이터의 지역적 상관관계를 Mamba가 더 잘 학습하도록 돕는 장치가 된다.

다만, 미세조정 단계에서 4-scan 구조로 변경하여 성능을 높였는데, 이는 사전학습 시의 단방향 제약이 추론 시에는 오히려 정보 손실이 될 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 Mamba 아키텍처를 위한 자기회귀 사전학습 방법론인 **ARM**을 제안한다. 클러스터 기반의 예측 단위와 MambaMLP 구조를 통해 훈련 효율을 극대화하였으며, 이를 통해 지도 학습에서 발생하던 모델 확장성 및 훈련 붕괴 문제를 해결하였다. 결과적으로 최대 85.5%의 ImageNet 정확도를 가진 거대 Mamba 모델을 구현하였으며, 이는 향후 Vision Mamba 모델들의 확장 및 실용적 적용에 중요한 기반이 될 것으로 보인다.
