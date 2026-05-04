# A residual dense vision transformer for medical image super-resolution with segmentation-based perceptual loss fine-tuning

Jin Zhu, Yang Guang, Pietro Lio (2023)

## 🧩 Problem to Solve

의료 영상 분야에서 고해상도 영상은 조기 진단, 수술 가이드, 질병 모니터링 등에 필수적이다. 그러나 실제 의료 환경에서 고해상도 영상을 획득하려면 스캔 시간 증가, 환자의 움직임으로 인한 아티팩트, 방사선 조사량 제한, 그리고 하드웨어 비용 등의 제약이 따른다. 이를 해결하기 위해 저해상도(LR) 영상으로부터 고해상도(HR) 영상을 복원하는 단일 이미지 초해상도(Single-Image Super-Resolution, SISR) 기술이 대안으로 제시되고 있다.

최근 일반 영상 처리 분야에서는 Vision Transformer(ViT)가 뛰어난 성능을 보이고 있으나, 이를 의료 영상 SR 작업에 적용하는 데에는 다음과 같은 문제점이 존재한다.
1. **데이터 부족 및 민감성**: 의료 영상 데이터셋은 상대적으로 크기가 작으며, 진단에 중요한 민감한 정보와 구조를 보존해야 한다는 특수성이 있다.
2. **평가 지표의 한계**: PSNR이나 SSIM 같은 전통적인 신호 충실도(Signal Fidelity) 지표는 의료 영상의 실제 진단적 가치나 기계적 인지 성능을 충분히 반영하지 못한다.
3. **효율성 문제**: ViT는 연산 비용이 매우 높기 때문에, 의료 영상 처리를 위한 효율적인 아키텍처 설계가 필요하다.

본 논문의 목표는 Residual Dense 연결과 Local Feature Fusion을 도입하여 효율적인 ViT 기반의 의료 영상 SR 모델을 제안하고, 의료 영상 분할(Segmentation) 작업의 사전 지식을 활용한 Perceptual Loss를 통해 복원 품질을 최적화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.
1. **Residual Dense Swin Transformer (RDST) 제안**: 기존의 Swin Transformer에 CNN에서 검증된 Residual Dense Connection과 Local Feature Fusion(LFF) 메커니즘을 결합하여, 파라미터 수를 획기적으로 줄이면서도 복원 성능을 높인 새로운 백본 네트워크를 설계하였다.
2. **Segmentation-based Perceptual Loss 도입**: 사전 학습된 segmentation U-Net의 특성 맵(Feature Map) 간의 거리를 이용한 새로운 Perceptual Loss를 제안하였다. 이를 통해 의료 영상의 구조적 특성을 보존하는 방향으로 모델을 미세 조정(Fine-tuning)할 수 있게 하였다.
3. **효율성 및 성능 입증**: 4가지 공개 의료 영상 데이터셋(OASIS, BraTS, ACDC, COVID)에서 실험한 결과, SOTA 모델인 SwinIR 대비 파라미터 수는 38% 수준(RDST-E의 경우 20%)에 불과하면서도 평균 PSNR을 향상시켰으며, 하위 작업인 분할(Segmentation) 정확도 또한 높였다.

## 📎 Related Works

### 기존 SR 네트워크
- **CNN 기반**: SRCNN, VDSR부터 EDSR, RDN과 같이 Dense Connection 및 Residual Learning을 활용한 모델들이 발전해 왔다. 특히 RCAN과 HAN은 Attention 메커니즘을 통해 정보 추출 능력을 높였다.
- **Transformer 기반**: SwinIR과 같은 모델이 Shifted-window attention을 통해 연산 효율성을 높이며 자연어 처리의 성공을 이미지 복원 작업으로 확장시켰다.
- **의료 영상 SR**: 초기에는 보간법(Interpolation)에 의존했으나, 이후 GAN 및 CNN 기반 방법론으로 발전하였다. 하지만 대부분의 연구가 자연 영상 기반의 모델을 단순 적용하거나, 의료 영상 특유의 구조적 보존 문제를 완전히 해결하지 못했다.

### 기존 접근 방식과의 차별점
본 연구는 단순한 아키텍처 변경에 그치지 않고, **의료 영상 분석의 하위 작업(Segmentation)을 상위 작업(SR)의 학습 과정에 통합**시켰다는 점에서 차별화된다. 기존의 Perceptual Loss가 주로 VGG-Net과 같은 자연 영상 분류 모델을 사용했다면, 본 논문은 의료 영상 전용 분할 모델을 사용하여 도메인 특화된 사전 지식을 활용한다.

## 🛠️ Methodology

### 1. RDST 네트워크 구조
전체 파이프라인은 (1) 얕은 특징 추출 헤드(H), (2) 심층 특징 추출 본체($R_n$), (3) 업샘플러(U)로 구성된다. 수학적으로 다음과 같이 표현된다.

$$F_{lr} = H(I_{lr})$$
$$F_{d} = F_{lr} + C_{k \times k}(R_n(F_{lr}))$$
$$I_{sr} = U_s(F_{d})$$

여기서 $s$는 확대 배율이며, $C_{k \times k}$는 $k \times k$ 커널을 가진 합성곱 층이다.

#### A. Dense STL Block (DSTB)
가장 기본 단위인 Swin Transformer Layer(STL) 두 개와 MLP 기반의 병목(Bottleneck) 층으로 구성된다. 입력 특징 맵을 $\text{cat}$ 연산을 통해 연결하여 특징 재사용성을 높이며, 연산량 감소를 위해 특징 차원을 $d$에서 $g$로 압축한다.

$$F_{d+g}^{i} = \text{cat}[F_{d}^{i-1}, B_{d \rightarrow g}(S^2(F_{d}^{i-1}))]$$

#### B. Residual Dense Swin Transformer Block (RDSTB)
3개의 DSTB를 순차적으로 쌓고, 마지막에 Local Feature Fusion(LFF)을 위한 $3 \times 3$ 합성곱 층을 배치한다. LFF는 $(3 \times g + d)$ 차원의 특징 맵을 다시 $d$ 차원으로 압축하여 정보 흐름을 안정화하고 연산 비용을 제어한다.

$$F_{d}^{i} = F_{d}^{i-1} + B_{3 \times g+d \rightarrow d}(D_3(F_{d}^{i-1}))$$

### 2. Segmentation U-Net 기반 Perceptual Loss
학습은 두 단계로 진행된다. 1단계에서는 픽셀 단위의 $L_1$ 손실 함수를 사용하여 기본 학습을 수행하며, 2단계에서는 사전 학습된 U-Net을 이용해 미세 조정을 수행한다.

#### A. 손실 함수의 정의
사전 학습된 U-Net의 특정 층에서의 특징 맵 차이를 이용해 다음과 같이 정의한다.
- **Fidelity-focused Loss ($L^{E(i)}$)**: 인코더의 $i$번째 블록 특징 맵 간의 $L_1$ 거리.
  $$L^{E(i)} = L_1(U[E^i](G(I_{lr})) - U[E^i](I_{hr}))$$
- **Machine Perception Loss ($L^{HRL}$)**: 예측된 분할 라벨 간의 유사도를 Dice 계수로 측정.
  $$L^{HRL} = 1 - \text{Dice}(U(I_{sr}), U(I_{hr}))$$

#### B. 최종 학습 목표
최종 손실 함수는 $L_1$ 손실과 Perceptual Loss의 가중 합으로 구성된다.
$$L^{SR} = \alpha L_1 + \lambda L^U$$

## 📊 Results

### 실험 설정
- **데이터셋**: OASIS(뇌 MR), BraTS(뇌 MR 다중 모달), ACDC(심장 MR), COVID(흉부 CT).
- **비교 대상**: EDSR, RDN, RCAN, HAN (CNN 기반), SwinIR (ViT 기반).
- **평가 지표**: PSNR, SSIM 및 하위 분할 작업의 Dice Coefficient.

### 주요 결과
1. **정량적 성능**: RDST는 7가지 모달리티 중 6가지에서 최고의 PSNR을 달성하였다. SwinIR 대비 파라미터 수는 38% 수준임에도 불구하고 평균 +0.09 dB의 PSNR 향상을 보였다.
2. **분할 정확도**: RDST의 결과물로 분할 작업을 수행했을 때, 15개 대상 영역 중 8개 영역에서 가장 높은 Dice 계수를 기록하여, 단순 수치 향상이 아닌 진단적 유용성이 증가했음을 증명하였다.
3. **효율성**: RDST-E(Lite 버전)는 SwinIR 파라미터의 20%만 사용하며 추론 속도는 46% 더 빠르면서도 유사한 성능을 유지하였다.
4. **Perceptual Loss 효과**: $L^{E(1)}$을 사용한 미세 조정은 SOTA 모델들의 PSNR을 평균 +0.14 dB 향상시켰으며, $L^{HRL}$은 하위 분할 작업의 Dice 계수를 평균 +0.0023 향상시켰다.

## 🧠 Insights & Discussion

### 1. ViT vs CNN
실험을 통해 Vision Transformer가 CNN보다 엣지(Edge) 부분의 복원 능력이 뛰어나다는 것을 확인하였다. 특히 self-attention 메커니즘이 파라미터를 더 효율적으로 활성화하여, 더 적은 연산량과 파라미터로도 더 깊은 네트워크의 효과를 낼 수 있음을 시사한다.

### 2. 충실도-인간 인지-기계 인지의 트레이드오프
본 논문은 매우 중요한 통찰을 제시한다.
- **PSNR/SSIM**: 이미지의 충실도(Fidelity)를 나타내지만, 인간의 시각적 만족도나 기계의 분석 성능과 반드시 일치하지 않는다.
- **FID**: 인간의 시각적 인지 품질을 대변하지만, 의료 영상에서는 그 의미가 불분명하다.
- **Dice Coefficient**: 기계적 인지(Machine Perception) 성능을 나타낸다.
결과적으로, $L^{E(1)}$은 충실도를 높이는 데 유리하고, $L^{HRL}$은 하위 분석 작업(기계 인지) 성능을 높이는 데 유리하다. 따라서 사용 목적에 따라 손실 함수를 선택해 모델을 맞춤형으로 구축해야 한다.

### 3. 한계점
- **시뮬레이션 데이터**: 본 연구는 bicubic downsampling과 가우시안 노이즈를 통한 시뮬레이션 데이터로 수행되었다. 실제 임상 환경의 복잡한 저해상도 특성을 완전히 반영하지 못했을 가능성이 있다.
- **과적합(Overfitting)**: 데이터셋 규모가 작은 ACDC의 경우, 모델 크기가 큰 RDST보다 RDST-E가 더 좋은 성능을 보였는데, 이는 의료 영상 SR에서 과적합 방지가 매우 중요함을 보여준다.

## 📌 TL;DR

본 논문은 의료 영상 초해상도(SR)를 위해 **Residual Dense 연결과 Local Feature Fusion을 결합한 효율적인 ViT 구조(RDST)**를 제안하고, **사전 학습된 분할 모델의 지식을 활용한 Perceptual Loss**를 통해 복원 품질을 최적화하였다. 결과적으로 SwinIR 대비 파라미터를 62% 줄이면서도 PSNR과 하위 진단 작업(분할) 성능을 동시에 향상시켰다. 이 연구는 향후 의료 영상의 재구성, 합성 및 다양한 저수준(low-level) 분석 작업의 효율적인 백본으로 활용될 가능성이 높다.