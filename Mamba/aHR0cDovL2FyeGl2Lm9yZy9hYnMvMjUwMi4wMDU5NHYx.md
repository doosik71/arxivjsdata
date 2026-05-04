# Fast Vision Mamba: Pooling Spatial Dimensions for Accelerated Processing

Saarthak Kapse, Robin Betz, Srinivasan Sivanandan (2025)

## 🧩 Problem to Solve

본 논문은 Vision Mamba(Vim)와 같은 State Space Model(SSM) 기반의 시각 모델들이 고해상도 이미지 처리 시 겪는 계산 효율성 문제를 해결하고자 한다.

Mamba는 Vision Transformer(ViT)의 Self-Attention이 갖는 2차 복잡도(quadratic complexity)를 선형 복잡도(linear complexity)로 줄여 토큰 간 상호작용을 처리한다. 특히 Recurrent hidden state 프로세스를 Parallel scan 알고리즘으로 가속하여, 입력 토큰 수 $L$에 대해 순차적 단계(sequential steps)를 $\log(L)$ 수준의 병렬 단계로 단축한다.

그러나 시각 데이터의 경우, 이미지 해상도가 증가함에 따라 토큰의 수 $L$이 해상도에 대해 2차적으로 증가($L=h \times w$)한다. 결과적으로 Parallel scan을 사용하더라도 병렬 단계의 수가 해상도에 따라 2차적으로 증가하게 되어, 고해상도 이미지 처리 시 처리량(throughput)이 급격히 저하되는 문제가 발생한다. 따라서 본 연구의 목표는 모델의 성능을 유지하면서 SSM 블록의 recurrent 단계 수를 줄여, 계산 시간을 해상도에 대해 선형적으로 스케일링하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SSM 블록 진입 전, 2D 토큰 그리드의 한 차원에 대해 **평균 풀링(Average Pooling)**을 적용하여 recurrent 계산량을 획기적으로 줄이는 것이다.

1. **FastVim 아키텍처 제안**: 토큰의 수를 $h^2$에서 $h$로 압축하여 SSM scan의 병렬 단계를 $2 \times \log(h)$에서 $1 \times \log(h)$로 줄인다. 이를 통해 고해상도 이미지에서 최대 72.5%의 추론 속도 향상을 달성한다.
2. **차원 교차 풀링(Alternating Pooling)**: 특정 층에서 열(column) 방향으로 풀링하면 행(row) 내 토큰 간 상호작용이 제한된다. 이를 해결하기 위해 각 Mamba 블록마다 풀링하는 차원을 행과 열로 교차 적용함으로써, 모든 토큰이 여러 블록을 거치며 암시적으로 상호작용하도록 설계하였다.
3. **확장 모델 제안**: 불규칙한 그리드나 마스킹된 데이터에 대응하는 **FastMaskVim**과, 다채널 이미지 처리를 위한 per-channel tokenization 기반의 **FastChannelVim**을 제안하여 범용성을 높였다.

## 📎 Related Works

- **Vision Mamba (Vim, VMamba)**: SSM을 시각 데이터에 적용하여 ViT보다 효율적인 스케일링을 가능하게 하였다. 하지만 여전히 고해상도에서 토큰 수의 2차적 증가로 인한 병렬 단계 증가 문제가 존재한다.
- **Sparse Contextualization**: ViT에서 토큰 병합(Merging)이나 가지치기(Pruning)를 통해 효율성을 높이려는 시도가 있었다. Mamba에서도 Token fusion(Famba-V)이나 Pruning-aware alignment(Vim-prune) 등이 제안되었으나, FastVim은 파라미터 추가 없이 단순한 풀링만으로 유사하거나 더 나은 효율성을 달성한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조

FastVim의 기본 구조는 Vim을 기반으로 하며, 각 SSM 블록 내에서 다음과 같은 절차를 거친다.

- **입력**: 이미지 토큰이 Norm 및 Expansion 레이어를 통과한다.
- **Transpose**: 매 블록마다 토큰 그리드를 전치(transpose)하여 풀링 차원을 교차시킨다.
- **Pooling**: 1D Convolution 이후, 행 또는 열 방향으로 평균 풀링(Average Pooling)을 수행하여 시퀀스 길이를 $L=h \times w$에서 $h$(또는 $w$)로 압축한다.
- **SSM Scan**: 압축된 토큰들을 사용하여 Selective scan을 수행한다.
- **Repeat (Decompression)**: SSM의 출력을 다시 원래의 토큰 그리드 크기($h \times w$)로 복원(repeat)한다.
- **Skip Connection & Norm**: 복원된 출력에 skip connection($D x_t$)을 더하고 Norm 레이어를 통과시킨다.

### 2. 수학적 배경 및 방정식

Mamba의 핵심인 SSM은 다음과 같은 연속 시간 상태 방정식으로 정의된다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

이를 이산화(discretization)하여 discrete-time SSM으로 변환하면 다음과 같다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t, \quad y_t = Ch_t + Dx_t$$

여기서 Mamba는 $B, C, \Delta$를 입력 $x$에 의존하는 함수로 만들어 선택적 스캔(selective scan)을 가능하게 한다.

- $B = s_B(x), C = s_C(x), \Delta = \tau_\Delta(\text{Parameter} + s_\Delta(x))$

FastVim은 이 과정에서 $x$를 풀링하여 $\text{length}(x)$를 줄임으로써, $\log(L)$에 비례하는 parallel scan의 단계 수를 직접적으로 감소시킨다.

### 3. 변형 모델

- **FastMaskVim**: 마스킹된 토큰이 존재하는 불규칙한 그리드에서는 단순 평균 풀링 대신, 행 내의 토큰 합을 구한 뒤 전체 열의 수 $w$로 나누는 방식을 사용한다. 이는 마스킹으로 인한 정보 손실을 방지하기 위함이다.
- **FastChannelVim**: 각 채널별로 토큰화를 수행하는 per-channel tokenization 환경에서, 공간 차원에 대해서만 풀링을 적용하여 매우 긴 시퀀스를 효율적으로 처리한다.

## 📊 Results

### 1. 이미지 분류 (ImageNet-1k)

- **정량적 결과**: FastVim-T, S, B 모델 모두 baseline인 Vim 모델과 대등한 성능을 보였다 (예: FastVim-B 및 Vim-B w/ LN 모두 Top-1 Accuracy 82.6% 달성). 이는 파라미터 증가 없이 풀링만으로도 충분한 상호작용이 가능함을 시사한다.

### 2. 효율성 분석

- **FLOPs**: 해상도가 높아질수록 ViT는 2차적으로 증가하는 반면, Vim과 FastVim은 선형적으로 증가한다. FastVim은 Vim 대비 최대 38.5% 적은 FLOPs를 사용한다.
- **Throughput**: $2048 \times 2048$ 고해상도 이미지에서 Vim 대비 **72.5%의 속도 향상**을 기록하였다. 이는 SSM scan 시간이 FastVim에서 해상도 변화에 관계없이 거의 일정하게 유지되기 때문이다.

### 3. 자기지도학습 (MAE)

- **FastMaskVim**: MAE 사전학습을 통해 ImageNet-1k에서 **86.7%의 Top-1 Accuracy**를 달성하며, Mamba 기반 인코더 중 SOTA 성능을 기록하였다.

### 4. 기타 작업

- **Cell Imaging (JUMP-CP)**: per-channel tokenization을 적용한 FastChannelVim은 Transformer 기반의 ChannelViT보다 약 8.3% 높은 정확도를 보이며, 긴 시퀀스 모델링에서 Mamba의 우위를 입증하였다.
- **Semantic Segmentation (ADE20K)**: FastVim-B는 mIoU 47.8%를 기록하며 DeiT 및 Vim-T/S보다 우수한 성능을 보였다.
- **Object Detection (MSCOCO)**: FastVim-B는 $\text{AP}_{\text{box}}$ 50.0%를 달성하여 baseline들을 상회하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

FastVim은 계산 복잡도를 해상도에 대해 선형적으로 유지하면서도 성능 저하가 거의 없다는 점이 매우 강력하다. 특히 고해상도 이미지로 갈수록 Vim 및 ViT와의 속도 격차가 벌어지므로, 의료 영상이나 위성 이미지와 같은 거대 이미지 분석에 매우 적합한 구조이다.

### 2. Mamba vs Transformer

흥미로운 점은 동일한 풀링 기법을 ViT에 적용했을 때 성능이 급격히 하락했다는 것이다. 이는 Mamba의 1D Convolution과 SSM의 문맥화(contextualization) 능력이 Transformer의 Self-Attention과는 근본적으로 다르며, Mamba가 이러한 희소한 상호작용(sparse interaction)에 더 강건함을 의미한다.

### 3. 한계 및 비판적 해석

- **오버헤드 발생**: 논문의 분석에 따르면, 풀링 후 다시 원래 크기로 복원하는 `repeat` 연산과 `skip connection` 계산이 PyTorch 기본 구현에서 상당한 시간을 점유한다.
- **안정성 문제**: Vim-B 모델 학습 시 loss spike가 발생하는 불안정성이 관찰되었으며, 이를 해결하기 위해 SSM scan 이후 LayerNorm을 추가하는 것이 필수적이었다. 이는 Mamba 기반 대형 모델의 학습 안정성을 위한 추가적인 정규화 장치가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 Vision Mamba의 계산 병목인 recurrent 단계를 줄이기 위해 **공간 차원 풀링(Spatial Pooling)**과 **차원 교차 적용(Alternating Dimensions)** 전략을 제안한 FastVim을 소개한다. 이 방법은 모델 성능을 유지하면서 고해상도 이미지 추론 속도를 최대 72.5% 향상시켰으며, MAE 사전학습 시 Mamba 기반 모델 중 SOTA 성능(86.7%)을 달성하였다. 특히 매우 긴 시퀀스를 처리해야 하는 다채널 현미경 이미지 분석 등에서 Transformer 대비 압도적인 효율성과 정확도를 보여, 향후 거대 이미지 처리 및 비디오 도메인 연구에 중요한 기반이 될 것으로 보인다.
