# SEGSRNet for Stereo-Endoscopic Image Super-Resolution and Surgical Instrument Segmentation

Mansoor Hayat, Supavadee Aramvith, Titipat Achakulvisut (2024)

## 🧩 Problem to Solve

본 논문은 로봇 수술 및 의료 영상 분야에서 빈번하게 발생하는 저해상도 스테레오 내시경 이미지의 문제점을 해결하고자 한다. 구체적으로, 이미지의 해상도가 낮을 경우 수술 도구(Surgical Instruments)를 정밀하게 식별하고 분할(Segmentation)하는 것이 어려워지며, 이는 수술의 정확도와 환자의 치료 결과에 직접적인 영향을 미칠 수 있다.

따라서 본 연구의 목표는 최첨단 초해상도(Super-Resolution, SR) 기술을 세그멘테이션 단계 이전에 적용함으로써 이미지의 선명도를 높이고, 이를 통해 최종적인 수술 도구 분할의 정확도를 향상시키는 하이브리드 프레임워크인 SEGSRNet을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'초해상도(SR)를 통한 입력 데이터의 품질 향상이 후속 작업인 세그멘테이션의 성능을 끌어올린다'**는 직관에 기반한다. 이를 위해 다음과 같은 설계를 도입하였다.

1. **SR-Segmentation 파이프라인**: 단순한 분할 모델이 아니라, SR 모듈을 통해 고해상도 이미지를 먼저 생성하고 이를 세그멘테이션 모델의 입력으로 사용하는 구조를 설계하였다.
2. **스테레오 특징 최적화**: 스테레오 이미지의 특성을 활용하기 위해 cross-view 정보 학습에 능한 biPAM(bi-Directional Parallax Attention Map) 네트워크와 다중 스케일 어텐션 메커니즘을 통합하였다.
3. **효율적인 분할 구조**: ResNet18 기반의 인코더와 공간 피라미드 풀링(Spatial Pyramid Pooling, SPP) 블록을 결합한 SPP-LinkNet-34를 사용하여 다중 스케일 특징 추출 능력을 강화하였다.

## 📎 Related Works

논문에서는 스테레오 이미지 SR을 위해 뷰 일관성(View Consistency)을 유지하는 것이 중요함을 언급하며, DCSSR의 Parallax Attention Module과 iPASSR의 biPAM과 같은 기존 연구들을 소개한다. 또한, 의료 영상 분할 분야에서 널리 사용되는 CNN 기반의 U-Net 및 TernausNet 등의 아키텍처를 언급한다.

기존의 접근 방식들이 SR 또는 세그멘테이션 중 어느 한 쪽의 성능 향상에 집중했다면, 본 논문은 이 두 가지를 통합하여 저해상도 환경에서도 정밀한 도구 식별이 가능하도록 하는 시스템적 접근을 시도했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

SEGSRNet은 크게 **Super-Resolution(SR) 파이프라인**과 **Segmentation 파이프라인**의 두 단계로 구성된다.

### 1. Super-Resolution Part

SR 모듈은 저해상도(LR) 이미지를 고해상도(HR) 이미지로 복원하며, 다음과 같은 세부 구성 요소를 가진다.

* **특징 추출 및 정제 (Feature Extraction and Refinement)**:
  * **CCSB (Combined Channel and Spatial Attention Block)**: 채널 어텐션(CAB)과 공간 어텐션(SAB)을 결합하여 중요 영역과 특징 맵을 강화한다.
  * **ASPP (Atrous Spatial Pyramid Pooling)** 및 **RDB (Residual Dense Blocks)**: 특징 계층을 심화하여 고수준의 특징을 추출한다.
* **Cross-View Feature Interaction Module (biPAM)**: 스테레오 이미지의 좌우 뷰 간 상호작용을 학습한다.
  * 특징 텐서 $F_U, F_V$를 생성하기 위해 다음과 같은 연산을 수행한다.
    $$F_{X(h, w, c)} = F_{X(h, w, c)} - \frac{1}{W} \sum_{i=1}^{W} F_{X(h, i, c)} \quad \text{for } X \in \{U, V\}$$
  * 어텐션 맵 $M_{R \to L}$과 $M_{L \to R}$을 사용하여 교차 뷰 상호작용을 수행한다.
    $$F_{X \to Y} = M_{X \to Y} \otimes F_X \quad \text{for } (X, Y) \in \{(R, L)\}$$
  * **폐색(Occlusion) 처리**: 유효 마스크 $V_L, V_R$을 계산하여 가려진 영역을 대상 뷰의 특징으로 채운다.
    $$F_{X \to Y} = V_Y \cdot F_{Y \to X} + (1 - V_Y) \cdot F_Y \quad \text{for } (X, Y) \in \{(R, L)\}$$
* **재구성 블록 (Reconstruction Block)**: 정제 블록, RDB, 채널 어텐션 층(CALayer)을 거쳐 최종적으로 Sub-pixel layer를 통해 고해상도 이미지를 복원한다.

### 2. Segmentation Part

SR을 통해 생성된 고해상도 이미지는 **SPP-LinkNet-34** 모델로 전달된다.

* **구조**: 인코더-디코더 구조를 가지며, 인코더로는 경량화된 ResNet18을 사용한다.
* **SPP (Spatial Pyramid Pooling) 블록**: 다운샘플링 과정에서 손실된 공간 정보를 효율적으로 회복하고, 다양한 스케일의 입력을 처리하여 분할 정확도를 높인다.
* **작업 종류**: 이진 분할(Binary), 부분 분할(Parts), 유형 분할(Type) 세 가지 작업을 수행한다.

### 3. 학습 절차 및 설정

* **데이터셋**: SR 학습 및 평가에는 MICCAI 2018 데이터셋을, 세그멘테이션 학습 및 평가에는 EndoVis 2017 데이터셋을 사용하였다.
* **최적화**: Pytorch 2.0, Nvidia 3090Ti GPU 환경에서 Adam optimizer를 사용하였으며, 학습률은 $3 \times 10^{-4}$, 총 100 epoch 동안 학습하였다. 파라미터 초기화는 Xavier initialization을 적용하였다.

## 📊 Results

### 1. 초해상도(SR) 성능 평가

$\times 2$ 및 $\times 4$ 확대 배율에서 PSNR과 SSIM 지표를 통해 측정하였다.

* **결과**: SEGSRNet은 Bicubic, SRCNN, VDSR 및 최신 모델인 DCSSRNet 등과 비교했을 때 두 데이터셋(MICCAI 2018, EndoVis 2017) 모두에서 가장 높은 PSNR과 SSIM 수치를 기록하였다. 특히 $\times 4$ 배율에서도 높은 복원 성능을 보였다.

### 2. 세그멘테이션 성능 평가

EndoVis 2017 데이터셋에 대해 10-fold 교차 검증을 통해 IoU와 Dice Score를 측정하였다.

* **이진 분할(Binary Segmentation)**: IoU $83.65\%$, Dice Score $89.80\%$를 달성하여 U-Net 대비 약 $9.81\%$ 향상된 성능을 보였다.
* **부분 분할(Parts Segmentation)**: U-Net 대비 약 $27.60\%$ 향상된 IoU 성능을 보였다.
* **유형 분할(Type Segmentation)**: 다른 작업에 비해 성능 향상 폭이 낮았으며, 이는 모델이 전역적 문맥 정보(Global contextual information)에 치중하여 픽셀 단위의 세밀한 구분에는 한계가 있었기 때문으로 분석된다.

## 🧠 Insights & Discussion

**강점**:
본 연구는 단순히 세그멘테이션 모델의 구조를 변경하는 것이 아니라, 전처리에 해당하는 SR 단계를 고도화함으로써 후속 작업의 입력 품질을 근본적으로 개선하였다. 특히 스테레오 이미지의 특성을 반영한 biPAM과 SPP-LinkNet-34의 조합이 의료 영상의 특수한 환경(저해상도, 도구의 복잡한 형태)에서 효과적임을 입증하였다.

**한계 및 비판적 해석**:

1. **유형 분할의 성능 저하**: 논문에서도 언급했듯이 Type segmentation에서 낮은 성능을 보였다. 이는 SR이 이미지의 전반적인 선명도는 높이지만, 서로 다른 도구를 구분 짓는 결정적인 미세 특징(Fine-grained features)까지 완벽하게 복원하지 못했거나, 세그멘테이션 모델이 이를 충분히 학습하지 못했음을 시사한다.
2. **실시간성 검토**: SR 모델과 세그멘테이션 모델을 순차적으로 적용하는 파이프라인 특성상, 연산량이 증가하여 실제 수술 중 실시간(Real-time) 적용 가능성에 대한 정밀한 지연 시간(Latency) 분석이 부족하다.

## 📌 TL;DR

SEGSRNet은 **'초해상도(SR) $\to$ 세그멘테이션'** 순서의 하이브리드 프레임워크를 통해 저해상도 스테레오 내시경 영상에서 수술 도구 분할 정확도를 높인 연구이다. biPAM 기반의 SR 모듈과 SPP-LinkNet-34 분할 모델을 결합하여 기존 U-Net 등보다 월등한 IoU 성능을 보였으며, 이는 향후 로봇 수술의 정밀도 향상과 환자 안전 개선에 기여할 가능성이 크다.
