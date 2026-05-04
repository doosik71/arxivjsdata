# U-Shape Mamba: State Space Model for faster diffusion

Alex Ergasti, Filippo Botti, Tomaso Fontanini, Claudio Ferrari, Massimo Bertozzi, Andrea Prati (2025)

## 🧩 Problem to Solve

최근 이미지 생성 분야에서 Diffusion Model은 매우 높은 품질의 결과를 보여주고 있으나, 막대한 계산 비용이 여전히 큰 과제로 남아 있다. 특히 Latent Diffusion Model(LDM)이나 DiT, U-ViT와 같은 모델들은 Transformer 기반의 백본을 사용하여 확장성을 높였으나, 하드웨어 요구 사항이 매우 높고 연산 복잡도가 시퀀스 길이의 제곱에 비례하여 증가하는 문제가 있다.

본 논문의 목표는 이러한 계산 오버헤드를 획기적으로 줄이면서도 고품질의 이미지 생성 능력을 유지하는 효율적인 확산 모델을 개발하는 것이다. 특히 연구 커뮤니티가 더 제한적인 하드웨어 환경에서도 고성능 생성 모델을 사용할 수 있도록 하여 접근성을 높이고 탄소 발자국을 줄이는 것을 지향한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba 기반의 State Space Model(SSM)을 U-Net의 계층적 구조(Hierarchical Structure)와 결합한 **U-Shape Mamba (USM)** 아키텍처를 제안하는 것이다.

주요 기여 사항은 다음과 같다:
- **Mamba 기반의 U-Net 백본 설계**: 기존의 Convolutional layer나 Transformer block 대신 Mamba 블록을 사용하며, 인코더에서 시퀀스 길이를 점진적으로 줄이고 디코더에서 이를 다시 복구하는 U-Shape 구조를 도입하였다.
- **연산 효율성의 극대화**: 기존의 Mamba 기반 확산 모델인 Zigma와 비교하여 GFlops를 약 3분의 1 수준으로 낮추었으며, 메모리 사용량 감소와 추론 속도 향상을 달성하였다.
- **생성 품질 향상**: 연산량은 크게 줄었음에도 불구하고 AFHQ, CelebAHQ, COCO 데이터셋에서 Zigma보다 낮은 FID(Frechet Inception Distance)를 기록하며 더 우수한 이미지 품질을 입증하였다.

## 📎 Related Works

**Diffusion Models 및 Flow Matching**
기존의 확산 모델은 노이즈를 제거하는 역과정(Backward process)을 학습하며, 이 과정에서 많은 단계(Step)가 필요하여 효율성이 떨어진다. 이를 해결하기 위해 최근에는 노이즈 분포와 데이터 분포 사이의 직선 경로를 학습하는 Flow Matching(또는 Rectified Flow) 방식이 제안되었으며, 본 논문에서도 이를 채택하여 학습 및 추론 속도를 높였다.

**Mamba (State Space Models)**
Transformer의 이차 복잡도($O(N^2)$) 문제를 해결하기 위해 선형 복잡도($O(N)$)를 갖는 SSM이 주목받았으며, 특히 Mamba는 SSM 파라미터를 입력에 의존하게 만들어(Selective Scan) Transformer에 필적하는 성능을 내면서도 훨씬 적은 자원을 사용한다.

**Zigma**
Zigma는 Mamba를 확산 모델의 백본으로 적용하여 DiT나 U-ViT 수준의 품질을 내면서 GFlops를 절반으로 줄인 모델이다. 본 논문의 USM은 Zigma를 베이스라인으로 삼아, 계층적 구조를 통해 연산 효율을 한 단계 더 발전시킨 모델이다.

## 🛠️ Methodology

### 1. Mamba (State Space Model)
Mamba는 연속적인 상태 공간 방정식을 이산화하여 시퀀스를 처리한다. 기본 시스템 방정식은 다음과 같다:
$$\begin{aligned} h'(t) &= Ah(t) + Bx(t) \\ y(t) &= Ch(t) + Dx(t) \end{aligned}$$
여기서 $A, B, C, D$는 학습 가능한 행렬이다. 이를 딥러닝에 적용하기 위해 Zero-Order Holder(ZOH) 규칙을 사용하여 이산화하면 다음과 같은 RNN 형태로 변환된다:
$$\begin{aligned} h_k &= \bar{A}h_{k-1} + \bar{B}x_k \\ y_k &= Ch_k + Dx_k \end{aligned}$$
Mamba의 핵심은 $B, C, \Delta$ 행렬이 고정된 것이 아니라 입력 $x$에 따라 결정되는 input-dependent 구조라는 점이며, 이는 선형 fully-connected layer를 통해 구현된다:
$$B = \text{Lin}_B(x), \quad C = \text{Lin}_C(x), \quad \Delta = \text{Lin}_\Delta(x)$$

### 2. Rectified Flow (Flow Matching)
본 논문은 데이터 $x$와 가우시안 노이즈 $\epsilon$ 사이를 선형 보간하는 단순한 경로를 학습한다:
$$z(t) = tx + (1-t)\epsilon, \quad t \in [0, 1]$$
모델 $v_\theta$는 이 경로의 도함수(속도)인 $v = \frac{d}{dt}z(t) = x - \epsilon$를 예측하도록 학습된다. 손실 함수는 다음과 같은 MSE(Mean Squared Error)를 사용한다:
$$L(\theta) = w(t)\|v_\theta(z(t), t) - (x - \epsilon)\|^2$$
이때 $w(t)$는 logit-normal 분포를 따르는 가중치 함수를 사용하여 특정 궤적 영역의 학습 우선순위를 조절한다.

### 3. U-Shape Mamba (USM) Architecture
전체 시스템은 VAE Encoder $\rightarrow$ USM Core $\rightarrow$ VAE Decoder 구조를 가진다.

- **U-Shape 구조**: 총 25개의 블록으로 구성된다 (Encoder 12개 $\rightarrow$ Bottleneck 1개 $\rightarrow$ Decoder 12개).
- **인코더 (Encoder)**: 매 3개의 블록마다 Downsampling 연산(Stride 2인 Convolution layer)을 수행하여 시퀀스 길이를 $\frac{1}{4}$로 줄인다. 최종적으로 bottleneck에서는 원래 길이의 $\frac{1}{64}$까지 압축된 특징 맵 $\phi_{\text{middle}}$을 얻는다.
- **디코더 (Decoder)**: 인코더의 역순으로 Transposed Convolution(Up Conv)을 통해 공간 해상도를 점진적으로 복원한다.
- **Skip Connections**: 정보 손실을 막기 위해 인코더의 특징 $\phi_{\text{enc}}$를 디코더의 특징 $\phi_{\text{dec}}$와 결합(Concatenate)한 후, Linear projection을 통해 차원을 다시 줄여 연결한다.
- **Main Block 구성**: `AdaLN` $\rightarrow$ `Mamba Block` $\rightarrow$ `Optional Cross-Attention` 순으로 구성된다. AdaLN은 타임스텝 $t$에 따라 시퀀스를 스케일링하고 시프트하며, Cross-Attention은 텍스트 조건($w$)이 주어질 때 사용된다.
- **Scan Configuration**: Zigma의 방식을 따라 8가지의 서로 다른 스캔 방향을 순환하며 적용하여 다양한 방향의 공간 정보를 학습하도록 설계하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: AFHQ (동물 얼굴), CelebAHQ (연예인 얼굴), COCO (텍스트-이미지 생성).
- **평가 지표**: FID (낮을수록 고품질), GFlops, 메모리 점유율, 추론 시간(Step당 평균 시간).
- **비교 대상**: Zigma.

### 정량적 결과
- **연산 효율성**: 텍스트 조건이 없을 때 USM은 20.66 GFlops를 사용하여 Zigma(64.12 GFlops)의 약 1/3 수준의 연산량만 필요로 한다. 텍스트 조건 포함 시에도 USM(40.84)이 Zigma(105.94)보다 월등히 효율적이다.
- **속도 및 메모리**: 추론 속도는 Zigma 대비 텍스트 미포함 시 73%, 포함 시 81% 더 빠르며, 메모리 사용량 역시 약 30~39% 감소하였다.
- **생성 품질 (FID)**: 
    - AFHQ: USM(16.87) vs Zigma(32.17) $\rightarrow$ **15.3 포인트 개선**
    - CelebAHQ: USM(21.80) vs Zigma(22.54) $\rightarrow$ **0.74 포인트 개선**
    - COCO: USM(39.10) vs Zigma(41.80) $\rightarrow$ **2.7 포인트 개선**

### 절제 연구 (Ablation Study)
Skip Connection의 유무를 비교한 결과, 이를 제거했을 때 AFHQ 데이터셋의 FID가 $16.87 \rightarrow 22.84$로 크게 상승(품질 저하)하였다. 이는 U-Shape 구조에서 세부적인 공간 정보를 유지하는 데 Skip Connection이 필수적임을 시사한다.

## 🧠 Insights & Discussion

본 연구는 Mamba의 선형 복잡도라는 이점과 U-Net의 계층적 특징 추출 능력을 결합함으로써, 생성 모델의 고질적인 문제인 계산 비용 문제를 효과적으로 해결하였다. 특히 단순히 모델을 가볍게 만든 것이 아니라, 오히려 FID 점수를 낮춰 품질을 향상시켰다는 점이 고무적이다.

**강점 및 통찰**:
- **효율적인 해상도 제어**: 모든 층에서 동일한 시퀀스 길이를 유지하는 대신, 점진적으로 줄였다가 늘리는 U-Shape 구조가 Mamba 모델에서도 매우 효과적으로 작동함을 확인하였다.
- **현실적 적용 가능성**: GFlops와 메모리 사용량의 획기적인 감소는 고가의 GPU 없이도 고품질 이미지 생성이 가능함을 의미하며, 이는 AI 모델의 지속 가능성(Carbon Footprint 감소) 측면에서도 긍정적이다.

**한계 및 논의**:
- 본 논문에서는 VAE Encoder의 설정을 LDM의 설정을 그대로 따랐으므로, VAE 자체의 효율성을 개선한다면 전체 파이프라인의 속도를 더 높일 수 있을 것이다.
- 다양한 해상도(예: 1024px 이상)에서의 확장성(Scalability)에 대한 추가 분석이 있다면 더 완벽한 검증이 될 수 있을 것으로 보인다.

## 📌 TL;DR

**U-Shape Mamba (USM)**는 Mamba의 선형 연산 효율성과 U-Net의 계층적 구조를 결합한 새로운 확산 모델 백본이다. 기존 Mamba 기반 모델인 Zigma 대비 **연산량(GFlops)을 1/3로 줄이고 추론 속도를 70~80% 향상**시켰음에도 불구하고, **FID 지표에서 더 우수한 이미지 품질**을 달성하였다. 이 연구는 고성능 이미지 생성 모델의 하드웨어 진입 장벽을 낮추고 효율적인 생성 AI 설계를 위한 중요한 방향성을 제시한다.