# Global and Local Mamba Network for Multi-Modality Medical Image Super-Resolution

Zexin Ji, Beiji Zou, Xiaoyan Kui, Sébastien Thureau, Su Ruan (2025)

## 🧩 Problem to Solve

본 논문은 다중 모달리티(Multi-modality) 의료 영상의 초해상도(Super-Resolution, SR) 구현 시 발생하는 효율성과 정확성의 트레이드-오프 문제를 해결하고자 한다. 의료 영상에서 고해상도 이미지는 정확한 진단에 필수적이지만, 획득 시간이 길고 비용이 많이 든다. 이를 해결하기 위해 저해상도(LR) 이미지와 고해상도 참조(Reference, Ref) 이미지를 결합하는 다중 모달리티 SR 기법이 사용된다.

기존 방식들은 다음과 같은 한계를 가진다:

1. **표현 능력과 계산 비용의 충돌**: Convolutional Neural Networks(CNNs)는 국소적(Local) 특징 추출에는 효율적이나 장거리 의존성(Long-range dependency) 모델링 능력이 부족하며, Vision Transformer(ViT)는 전역적(Global) 컨텍스트 캡처 능력이 뛰어나지만 토큰 수에 따라 계산 복잡도가 제곱으로 증가한다.
2. **모달리티 내 정보의 활용 부족**: 기존 방법들은 LR 이미지와 Ref 이미지를 동일하게 처리한다. 하지만 LR 이미지는 세부 디테일은 부족해도 전역적인 구조 정보가 중요하며, Ref 이미지는 정밀한 국소적 텍스처 정보를 제공한다는 서로 다른 역할이 존재함에도 이를 충분히 활용하지 못한다.
3. **모달리티 간 의존성 활용 미흡**: 서로 다른 모달리티 간의 유사성, 차이점, 상호 보완적 관계를 적응적으로 융합하는 메커니즘이 부족하여 정보 손실이 발생한다.

따라서 본 논문의 목표는 Mamba 아키텍처를 도입하여 선형 계산 복잡도로 전역 의존성을 모델링하고, LR과 Ref 이미지의 특성에 맞는 개별 브랜치를 통해 정밀한 의료 영상 SR을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Global 및 Local Mamba를 분리한 듀얼 브랜치 네트워크(GLMamba)**를 설계하여, 입력 이미지의 특성에 따라 전역적 구조와 국소적 디테일을 동시에 최적화하는 것이다.

주요 기여 사항은 다음과 같다:

- **GLMamba 네트워크**: LR 이미지용 Global Mamba 브랜치와 Ref 이미지용 Local Mamba 브랜치를 구성하여 각 모달리티의 역할(전역 구조 보완 vs 국소 디테일 제공)을 최적화하였다.
- **Deform Block 및 Modulator**: 픽셀 수준의 유연한 샘플링을 위한 Deformable Convolution 기반 블록과, 이를 Mamba 특징과 결합하여 정밀도를 높이는 Modulator를 제안하였다.
- **Multi-Modality Feature Fusion Block**: 모달리티 간의 유사성(Similarity), 차이점(Difference), 상호 보완성(Complementarity)을 모두 고려하여 특징을 적응적으로 융합하는 블록을 설계하였다.
- **Contrastive Edge Loss (CELoss)**: Laplacian 연산자를 활용하여 의료 영상의 경계선 텍스처와 대비를 강화하는 손실 함수를 도입하였다.

## 📎 Related Works

### 1. Single-Modality Super-Resolution

CNN 기반 방법들은 강력한 특징 추출 능력을 가지나 수용 영역(Receptive field)이 제한적이라는 단점이 있다. 이후 Transformer 기반 방법들이 장거리 의존성을 캡처하며 등장하였으나, 높은 계산 비용과 대규모 데이터셋 요구라는 한계가 있었다. 최근 Mamba(State Space Model)가 선형 복잡도로 이를 해결하며 의료 영상 SR에 적용되기 시작하였다.

### 2. Multi-Modality Super-Resolution

다양한 대조도(Contrast) 이미지를 사용하는 방식(MINet, SANet 등)이 제안되었다. 이들은 참조 이미지의 보조 정보를 활용해 SR 성능을 높이지만, 여전히 높은 계산 비용으로 인해 임상 적용에 제약이 있다.

### 3. Mamba 및 State Space Models (SSMs)

Mamba는 Selective Scan State Space model (S6)을 통해 입력 데이터에 따라 동적으로 정보를 통합한다. 이는 CNN의 국소성과 Transformer의 전역성을 결합하면서도 계산 효율성을 극대화한 모델이다.

## 🛠️ Methodology

### 1. 전체 파이프라인

GLMamba는 두 개의 브랜치로 구성된다.

- **LR 브랜치**: $\text{LR Image} \rightarrow \text{Upsampling} \rightarrow \text{Global Mamba} \rightarrow \text{Deform Block} \rightarrow \text{Modulator}$.
- **Ref 브랜치**: $\text{Ref Image} \rightarrow \text{Local Mamba} \rightarrow \text{Deform Block} \rightarrow \text{Modulator}$.
이후 두 브랜치의 특징은 **Multi-Modality Feature Fusion Block**에서 융합되어 최종 SR 이미지($S^R$)와 재구성된 참조 이미지($Rec^{Ref}$)가 생성된다.

### 2. Global and Local Mamba

- **Global Mamba**: LR 이미지는 누락된 디테일을 보완하기 위해 전체적인 맥락이 중요하다. 따라서 이미지 전체를 4방향(좌$\rightarrow$우, 상$\rightarrow$하, 우$\rightarrow$좌, 하$\rightarrow$상)으로 스캔하여 전역적 의존성을 모델링하는 2D Selective Scan (SS2D)을 적용한다.
- **Local Mamba**: Ref 이미지는 이미 고해상도이므로 세밀한 국소 정보가 중요하다. 이를 위해 이미지를 **4개의 사분면(Quadrant)**으로 나누어 각 영역 내에서 독립적으로 패치 간의 상관관계를 학습함으로써 국소적 디테일을 보존한다.

### 3. Deform Block 및 Modulator

**Deform Block**은 고정된 샘플링 패턴 대신 이미지 내용에 따라 동적으로 오프셋($\Delta p_k$)과 스칼라 값($\Delta m_k$)을 학습하여 유연하게 특징을 추출한다. 수식은 다음과 같다:
$$Y(p) = \sum_{k=1}^{K} w_k \cdot X(p + p_k + \Delta p_k) \cdot \Delta m_k$$
**Modulator**는 Deform Block에서 추출된 픽셀 수준의 특징과 Mamba의 패치 수준 특징을 결합한다. Sigmoid 함수를 통해 Deformable 특징을 기반으로 Mamba 특징을 선택적으로 강화함으로써 미세 구조에 대한 인지 능력을 높인다.

### 4. Multi-Modality Feature Fusion Block

세 가지 관점에서 특징을 융합한다:

- **Difference (차이점)**: $F_{diffuse} = F_{LR\uparrow} - F_{Ref}$ (중복 제거 및 핵심 차이 강조)
- **Similarity (유사성)**: $F_{simfuse} = F_{LR\uparrow} \otimes F_{Ref}$ (일관된 특징 강화)
- **Complementarity (상호 보완성)**: Softmax 기반의 가중치 맵($w_{LR\uparrow}, w_{Ref}$)을 생성하여 두 특징을 가중 합산한다:
  $$F_{comfuse} = (F_{LR\uparrow} \otimes w_{LR\uparrow}) \oplus (F_{Ref} \otimes w_{Ref})$$
최종적으로 이 세 가지 특징($F_{diffuse}, F_{simfuse}, F_{comfuse}$)은 Global Max Pooling과 Fully Connected layer를 통해 계산된 적응적 가중치로 가중 합산되어 $F_{fuse}$가 된다.

### 5. Objective Function

전체 손실 함수는 픽셀 단위의 $L_1$ 손실과 경계선 강화를 위한 **Contrastive Edge Loss (CELoss)**의 합으로 정의된다:
$$\text{Loss} = \alpha L_{SR1} + \beta L_{Ref1} + \gamma L_{CELoss}$$
여기서 $L_{CELoss}$는 세 가지 Laplacian 커널($E_1$: 수평/수직 엣지, $E_2$: 대각선 엣지, $E_3$: 국소 대비)을 사용하여 고주파 세부 정보를 복원한다:
$$L_{CELoss} = \frac{1}{3} \sum_{i=1}^{3} (E_i \odot S^R - E_i \odot HR)^2$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: BraTS2021 (T1 $\rightarrow$ T2 SR), IXI (PD $\rightarrow$ T2 SR).
- **지표**: PSNR, SSIM, Dice Score (분할 작업용).
- **구현**: PyTorch, NVIDIA Tesla V100, Adam optimizer, 채널 수 96.

### 2. 주요 결과

- **정량적 성능**: BraTS2021 및 IXI 데이터셋 모두에서 2$\times$ 및 4$\times$ 업샘플링 시 기존 방법들(SRCNN, MINet, SANet, WavTrans 등)보다 높은 PSNR과 SSIM을 기록하였다. 특히 4$\times$ SR에서 성능 향상이 두드러졌다.
- **효율성**: 다른 다중 모달리티 SR 방법들과 비교했을 때, 파라미터 수(1.187M)와 연산량(32.27G FLOPS)이 현저히 낮아 매우 경량화된 모델임을 입증하였다.
- **다운스트림 작업(Tumor Segmentation)**: SR 결과물을 SwinUnet 분할 모델에 입력했을 때, 타 방법론보다 높은 Dice Score를 기록하여, 본 모델이 생성한 고해상도 이미지가 실제 임상 진단(종양 분할)에 더 유용함을 보여주었다.

## 🧠 Insights & Discussion

### 1. 강점

- **역할 기반 설계**: LR과 Ref 이미지의 특성을 구분하여 Global/Local Mamba로 처리한 접근 방식이 매우 효율적이다.
- **계산 효율성**: Mamba의 선형 복잡도를 활용하여 Transformer의 성능을 내면서도 연산 비용을 획기적으로 낮췄다.
- **엣지 보존**: CELoss를 통해 의료 영상에서 중요한 해부학적 경계선을 뚜렷하게 복원하여 분할 성능까지 향상시킨 점이 인상적이다.

### 2. 한계 및 향후 과제

- **2D 처리의 한계**: 본 논문은 2D 슬라이스 단위로 처리하였다. 실제 MRI는 3D 데이터이므로, 3D 공간 정보를 활용하는 방향으로 확장한다면 성능이 더 향상될 가능성이 크다.
- **정렬(Alignment) 가정**: 두 모달리티가 이미 정렬되어 있다는 전제하에 작동한다. 하지만 실제 임상 데이터는 정렬되지 않은 경우가 많으므로, 정렬과 SR을 동시에 수행하는 멀티태스크 프레임워크 연구가 필요하다.

## 📌 TL;DR

본 논문은 의료 영상 SR을 위해 **Global Mamba(LR용)와 Local Mamba(Ref용)**를 결합한 **GLMamba** 네트워크를 제안한다. Deformable 블록과 적응적 모달리티 융합 블록, 그리고 경계선 강화 손실 함수(CELoss)를 통해 **계산 효율성(선형 복잡도)과 복원 정밀도를 동시에 달성**하였으며, 이는 최종적으로 종양 분할 정확도 향상으로 이어졌다. 이 연구는 특히 자원이 제한된 의료 환경에서 고해상도 영상을 빠르게 얻기 위한 효율적인 딥러닝 구조를 제시했다는 점에서 가치가 있다.
