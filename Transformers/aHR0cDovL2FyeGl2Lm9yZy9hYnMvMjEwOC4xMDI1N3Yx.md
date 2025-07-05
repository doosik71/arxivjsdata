# SwinIR: Image Restoration Using Swin Transformer
Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte

## 🧩 Problem to Solve
이미지 복원(초해상화, 노이즈 제거, JPEG 압축 아티팩트 제거 등)은 저품질 이미지로부터 고품질 이미지를 복원하는 것을 목표로 하는 오랜 저수준(low-level) 컴퓨터 비전 문제입니다. 기존의 최첨단 이미지 복원 방법들은 주로 CNN(Convolutional Neural Networks) 기반이었지만, CNN은 다음과 같은 한계가 있습니다:
*   **내용 독립적인 상호작용:** 동일한 컨볼루션 커널이 이미지의 다른 영역을 복원하는 데 사용되어 최적의 선택이 아닐 수 있습니다.
*   **장거리 의존성 모델링의 비효율성:** 컨볼루션의 지역적 처리 원칙으로 인해 장거리 의존성 모델링에 효과적이지 않습니다.

한편, 고수준(high-level) 비전 작업에서 인상적인 성능을 보여준 Transformer 모델들은 이미지 복원에 거의 시도되지 않았습니다. 기존 Vision Transformer 모델들은 이미지를 고정된 크기의 패치로 분할하여 독립적으로 처리하므로 다음과 같은 단점이 있습니다:
*   **경계 픽셀의 정보 활용 부족:** 패치 외부의 인접 픽셀을 활용하지 못합니다.
*   **패치 경계 아티팩트 발생:** 복원된 이미지에 패치 주변의 경계 아티팩트가 생길 수 있습니다.

## ✨ Key Contributions
*   **Swin Transformer 기반의 강력한 이미지 복원 모델 SwinIR 제안:** Swin Transformer를 기반으로 한 강력한 이미지 복원 모델인 SwinIR을 제안하며, 이는 저수준 비전 작업에서 Transformer의 잠재력을 보여줍니다.
*   **최첨단 성능 달성:** 이미지 초해상화(고전적, 경량, 실세계), 이미지 노이즈 제거(흑백, 컬러), JPEG 압축 아티팩트 제거 등 세 가지 대표적인 이미지 복원 작업에서 최첨단 방법을 능가하는 성능을 달성했습니다.
*   **효율적인 모델:** 기존 최첨단 방법 대비 최대 $0.14 \sim 0.45\text{dB}$의 PSNR 향상을 보였으며, 총 파라미터 수는 최대 $67\%$까지 감소시켰습니다. 이는 적은 파라미터로 더 나은 성능을 달성하는 SwinIR의 효율성을 입증합니다.
*   **CNN과 Transformer의 장점 통합:** Swin Transformer의 로컬 어텐션 메커니즘을 통해 CNN의 대규모 이미지 처리 장점을, Shifted Window 방식을 통해 Transformer의 장거리 의존성 모델링 장점을 통합했습니다.
*   **안정적인 학습과 빠른 수렴:** Transformer 기반 모델임에도 불구하고, 소규모 데이터셋에서도 CNN 기반 모델보다 더 나은 결과를 달성하며 더 빠르고 안정적으로 수렴함을 보여주었습니다.

## 📎 Related Works
*   **이미지 복원 (Image Restoration):**
    *   **전통적 모델 기반 방법:** BM3D [14], A+ [73] 등.
    *   **CNN 기반 방법:** SRCNN [18], DnCNN [90], ARCNN [17]을 필두로 잔차 블록(residual block) [40, 88], 조밀 연결(dense connection) [81, 97] 등 정교한 아키텍처 디자인을 통해 성능을 개선했습니다.
    *   **CNN 기반 어텐션 메커니즘:** 채널 어텐션(channel attention) [95, 63], 비지역 어텐션(non-local attention) [52, 61], 적응형 패치 통합(adaptive patch aggregation) [100] 등을 활용했습니다.
*   **비전 Transformer (Vision Transformer):**
    *   자연어 처리 분야의 Transformer [76]는 이미지 분류 [19], 객체 탐지 [6], 분할 [84] 등 컴퓨터 비전 분야에서 인상적인 성능을 보여왔습니다.
    *   **이미지 복원을 위한 Transformer:**
        *   IPT [9]: 표준 Transformer 기반으로 다양한 복원 문제에 적용 가능하나, $115.5\text{M}$ 이상의 많은 파라미터와 $1.1\text{M}$ 이상의 대규모 데이터셋, 멀티태스크 학습에 의존합니다.
        *   VSR-Transformer [5]: 비디오 SR에서 특징 융합을 위해 자기-어텐션(self-attention)을 사용하나, 이미지 특징은 여전히 CNN에서 추출합니다.
        *   IPT와 VSR-Transformer 모두 패치 단위 어텐션(patch-wise attention)을 사용하며, 이는 이미지 복원에 부적절할 수 있습니다.
        *   동시 연구로 Swin Transformer [56] 기반의 U-Shaped 아키텍처 [82]가 제안되었습니다.
    *   **Swin Transformer [56]:** 로컬 어텐션 메커니즘과 Shifted Window 방식을 통해 CNN의 장점(대규모 이미지 처리)과 Transformer의 장점(장거리 의존성 모델링)을 모두 통합하여 이미지 분류에서 뛰어난 성능을 보였습니다.

## 🛠️ Methodology
SwinIR은 세 가지 주요 모듈로 구성됩니다: Shallow Feature Extraction, Deep Feature Extraction, High-Quality Image Reconstruction.

1.  **Shallow Feature Extraction (저수준 특징 추출):**
    *   저품질 입력 이미지 $I_{LQ} \in \mathbb{R}^{H \times W \times C_{in}}$로부터 $3 \times 3$ 컨볼루션 레이어 $H_{SF}(\cdot)$를 사용하여 저수준 특징 $F_0 \in \mathbb{R}^{H \times W \times C}$를 추출합니다.
    *   $F_0 = H_{SF}(I_{LQ})$
    *   추출된 $F_0$는 저주파수 정보를 보존하고, 안정적인 학습을 제공하기 위해 재구성 모듈로 직접 전달됩니다.

2.  **Deep Feature Extraction (고수준 특징 추출):**
    *   주로 $K$개의 Residual Swin Transformer Blocks (RSTB)와 마지막 $3 \times 3$ 컨볼루션 레이어로 구성됩니다.
    *   각 RSTB는 여러 Swin Transformer Layer (STL)를 포함하여 로컬 어텐션과 교차-윈도우 상호작용을 가능하게 합니다.
    *   특징 추출 과정은 다음과 같습니다:
        *   $F_i = H_{RSTB_i}(F_{i-1}), \quad i=1, 2, \dots, K$
        *   $F_{DF} = H_{CONV}(F_K)$
    *   잔차 블록 끝에 컨볼루션 레이어를 추가하여 특징을 강화하고 Transformer 기반 네트워크에 컨볼루션 연산의 귀납적 편향(inductive bias)을 제공합니다.

3.  **High-Quality Image Reconstruction (고품질 이미지 재구성):**
    *   재구성 모듈 $H_{REC}(\cdot)$은 저수준 특징 $F_0$와 고수준 특징 $F_{DF}$를 융합하여 고품질 이미지 $I_{RHQ}$를 재구성합니다.
    *   $I_{RHQ} = H_{REC}(F_0 + F_{DF})$
    *   긴 스킵 연결(long skip connection)을 통해 저주파수 정보가 재구성 모듈로 직접 전달되어, 고수준 특징 추출 모듈이 고주파수 정보 복원에 집중할 수 있도록 돕고 학습을 안정화합니다.
    *   이미지 초해상화(업샘플링 필요)의 경우 서브픽셀 컨볼루션 레이어(sub-pixel convolution layer)를 사용하고, 노이즈 제거 및 JPEG 아티팩트 제거(업샘플링 불필요)의 경우 단일 컨볼루션 레이어를 사용합니다.
    *   고품질 이미지 대신 저품질 이미지와 고품질 이미지 간의 잔차를 학습하는 잔차 학습(residual learning) 방식을 채택합니다.
    *   $I_{RHQ} = H_{SwinIR}(I_{LQ}) + I_{LQ}$

*   **손실 함수 (Loss Function):**
    *   **이미지 초해상화:** $L_1$ 픽셀 손실 $L = \left\|I_{RHQ} - I_{HQ}\right\|_1$을 최적화하며, 실세계 이미지 SR의 경우 픽셀 손실, GAN 손실, 지각 손실(perceptual loss)을 조합합니다.
    *   **이미지 노이즈 제거 및 JPEG 압축 아티팩트 제거:** Charbonnier 손실 $L = \sqrt{\left\|I_{RHQ} - I_{HQ}\right\|^2 + \epsilon^2}$를 사용하며, $\epsilon$은 $10^{-3}$으로 설정됩니다.

*   **Residual Swin Transformer Block (RSTB):**
    *   Swin Transformer Layer (STL)와 컨볼루션 레이어를 포함하는 잔차 블록입니다.
    *   $L$개의 Swin Transformer Layer를 통해 중간 특징을 추출한 후, 잔차 연결 전에 컨볼루션 레이어를 추가합니다.
    *   $F_{i,j} = H_{STL_{i,j}}(F_{i,j-1}), \quad j=1, 2, \dots, L$
    *   $F_{i,out} = H_{CONV_i}(F_{i,L}) + F_{i,0}$
    *   이 설계는 SwinIR의 변환 불변성(translational equivariance)을 향상시키고, 다른 블록에서 재구성 모듈로의 정체성 기반 연결(identity-based connection)을 제공하여 다양한 수준의 특징을 통합할 수 있도록 합니다.

*   **Swin Transformer Layer (STL):**
    *   표준 멀티-헤드 자기-어텐션(multi-head self-attention) 기반이며, 주요 차이점은 **로컬 어텐션(local attention)**과 **Shifted Window 메커니즘**입니다.
    *   입력을 겹치지 않는 $M \times M$ 로컬 윈도우로 분할한 후 각 윈도우에 대해 자기-어텐션을 계산합니다.
    *   어텐션 함수: $\text{Attention}(Q,K,V) = \text{SoftMax}(QK^T/\sqrt{d}+B)V$ (여기서 $B$는 학습 가능한 상대 위치 인코딩입니다.)
    *   멀티-레이어 퍼셉트론(MLP)과 LayerNorm (LN), 잔차 연결이 사용됩니다.
    *   로컬 윈도우 간의 연결을 가능하게 하기 위해 일반 윈도우 분할과 Shifted Window 분할을 교대로 사용합니다.

## 📊 Results
*   **전반적 성능:** SwinIR은 다양한 이미지 복원 작업에서 최첨단 성능을 달성했으며, 기존 방법 대비 PSNR이 최대 $0.14 \sim 0.45\text{dB}$ 향상되었고, 파라미터 수는 최대 $67\%$까지 감소했습니다.
*   **고전적 이미지 초해상화:**
    *   Set5, Set14, BSD100, Urban100, Manga109 등 거의 모든 벤치마크 데이터셋에서 최상의 성능을 기록했습니다.
    *   특히 Manga109 (x4)에서 최대 $0.26\text{dB}$의 PSNR 이득을 보였습니다.
    *   ImageNet을 학습에 사용하고 훨씬 많은 파라미터를 가진 IPT보다 적은 파라미터로 더 높은 정확도를 달성했습니다.
    *   SwinIR의 파라미터 수(11.8M)는 IPT(115.5M)보다 훨씬 적고, 최첨단 CNN 기반 모델(15.4~44.3M)과 비교해도 적거나 비슷합니다.
    *   시각적 비교에서 SwinIR은 높은 주파수 디테일을 복원하고 블러링 아티팩트를 줄여 선명하고 자연스러운 결과물을 생성했습니다.
*   **경량 이미지 초해상화:**
    *   작은 크기의 SwinIR 모델(878K 파라미터)은 경쟁 모델 대비 최대 $0.53\text{dB}$의 PSNR 마진으로 뛰어난 성능을 보였으며, 파라미터 및 FLOPs 수는 유사하여 높은 효율성을 입증했습니다.
*   **실세계 이미지 초해상화:**
    *   BSRGAN과 동일한 열화 모델로 재학습한 SwinIR은 시각적으로 만족스러운 이미지와 선명한 가장자리를 생성하여 ESRGAN, RealSR, BSRGAN, Real-ESRGAN 등 다른 모델들을 능가했습니다.
*   **JPEG 압축 아티팩트 제거:**
    *   Classic5 및 LIVE1 데이터셋에서 기존 최첨단 모델 대비 최소 $0.07 \sim 0.11\text{dB}$의 평균 PSNR 이득을 보였습니다.
    *   DRUNet보다 적은 파라미터 수(11.5M vs 32.7M)에도 불구하고 더 나은 성능을 달성했습니다.
*   **이미지 노이즈 제거 (흑백 및 컬러):**
    *   모든 비교 모델(BM3D, DnCNN, DRUNet 등)보다 우수한 성능을 보였으며, 특히 Urban100 데이터셋에서 DRUNet을 최대 $0.3\text{dB}$ 능가했습니다.
    *   DRUNet보다 적은 파라미터 수(12.0M vs 32.7M)로 높은 효율성을 입증했습니다.
    *   시각적 결과에서 SwinIR은 심한 노이즈를 효과적으로 제거하면서 고주파수 이미지 디테일을 보존하여 더 선명한 가장자리와 자연스러운 질감을 만들어냈습니다.
*   **제거 연구 (Ablation Study):**
    *   채널 수, RSTB 수, RSTB 내의 STL 수는 모델 성능에 양의 상관관계를 보였습니다.
    *   SwinIR은 RCAN보다 다양한 패치 크기에서 더 나은 성능을 보였고, 패치 크기가 클수록 PSNR 이득이 커졌습니다.
    *   SwinIR은 학습 이미지 수가 적을 때도 CNN 기반 모델보다 우수한 성능을 보였으며, RCAN보다 빠르고 더 잘 수렴하는 것을 확인했습니다.
    *   RSTB 내의 잔차 연결은 PSNR을 $0.16\text{dB}$ 향상시키는 중요한 요소였고, $3 \times 3$ 컨볼루션 레이어가 $1 \times 1$ 컨볼루션보다 성능에 더 기여했습니다.

## 🧠 Insights & Discussion
*   SwinIR은 Swin Transformer의 계층적 구조와 Shifted Window 메커니즘을 성공적으로 활용하여 이미지 복원과 같은 저수준 비전 작업에서 뛰어난 성능을 달성했습니다.
*   이는 기존 CNN 기반 모델의 한계(내용 독립성, 장거리 의존성 모델링 부족)와 기존 Vision Transformer의 단점(패치 아티팩트, 높은 파라미터 수)을 효과적으로 극복했습니다.
*   SwinIR의 Shallow Feature Extraction과 Deep Feature Extraction 모듈의 조합, 그리고 긴 스킵 연결은 저주파수 정보를 효율적으로 보존하면서 고주파수 디테일 복원에 집중할 수 있도록 하여 학습 안정성과 성능 향상에 기여합니다.
*   RSTB 내부의 잔차 연결과 컨볼루션 레이어는 Transformer 기반 네트워크에 CNN의 귀납적 편향을 부여하고, 다양한 수준의 특징을 효과적으로 통합하는 데 필수적임을 실험을 통해 입증했습니다.
*   일반적으로 Transformer 모델이 대규모 데이터셋과 느린 수렴을 필요로 한다는 통념과는 달리, SwinIR은 소규모 데이터셋에서도 뛰어난 성능을 보였고, 기존 CNN 모델보다 빠르고 안정적으로 수렴하는 효율성을 보여주었습니다.
*   제안된 SwinIR은 높은 효율성과 일반화 능력을 바탕으로 다양한 이미지 복원 작업에서 새로운 최첨단 성능을 제시하며, 향후 이미지 디블러링(deblurring) 및 디레이닝(deraining)과 같은 다른 복원 작업으로 확장될 가능성을 보여줍니다.

## 📌 TL;DR
*   **문제:** 기존 CNN 기반 이미지 복원 모델은 내용 독립적인 처리와 장거리 의존성 모델링에 한계가 있으며, 기존 Transformer 모델은 파라미터가 많고 패치 아티팩트 문제가 있었습니다.
*   **방법:** SwinIR은 Swin Transformer를 활용하여 이미지 복원을 위한 가볍고 강력한 모델을 제안합니다. 이 모델은 저주파수 특징을 위한 Shallow Feature Extraction (얕은 컨볼루션), 로컬 및 교차-윈도우 어텐션을 위한 Residual Swin Transformer Blocks (RSTB) 기반의 Deep Feature Extraction, 그리고 이 특징들을 잔차 학습으로 결합하는 재구성 모듈로 구성됩니다.
*   **핵심 결과:** SwinIR은 다양한 이미지 복원 작업(초해상화, 노이즈 제거, JPEG 아티팩트 제거)에서 기존 CNN 및 Transformer 기반 모델보다 훨씬 적은 파라미터로도 일관되게 최첨단 성능을 달성하며, 빠른 수렴과 뛰어난 효율성 및 일반화 능력을 입증했습니다.