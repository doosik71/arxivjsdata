# A Comprehensive Review of U-Net and Its Variants: Advances and Applications in Medical Image Segmentation

Wang Jiangtao, Nur Intan Raihana Ruhaiyem, Fu Panpan (2024)

## 🧩 Problem to Solve

의료 영상 분석에서 병변과 주변 조직 사이의 대조도(contrast)가 낮고 경계가 흐릿하며, 동일한 질병 내에서도 병변의 모양과 가장자리가 매우 다양하게 나타나는 특성이 있다. 이러한 특성은 정밀한 의료 영상 분할(Medical Image Segmentation)을 어렵게 만드는 핵심 요인이 된다. 정확한 병변 분할은 환자의 상태 평가와 치료 계획 수립을 위한 필수적인 전제 조건이다. 따라서 본 논문은 의료 영상 분할 분야에서 가장 널리 사용되는 U-Net 아키텍처와 그 변형 모델들을 체계적으로 분석하여, 각 모델의 구조적 개선점과 적용 가능성을 제시하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 U-Net 및 그 변형 모델들을 구조적 수정 관점에서 분류하고, 의료 영상 모달리티(Imaging Modality)별로 적합한 모델을 체계적으로 정리한 것이다. 특히 다음의 네 가지 핵심 개선 메커니즘을 중심으로 분석을 수행하였다.

1.  **Jump-connection (Skip-connection) 메커니즘**: 정보 손실을 방지하고 수렴 속도를 높이는 구조.
2.  **Residual-connection 메커니즘**: 기울기 소실(Vanishing Gradient) 문제를 해결하고 네트워크 깊이를 확장하는 구조.
3.  **3D U-Net**: 3차원 볼륨 데이터의 공간적 구조 정보를 직접 처리하는 확장 구조.
4.  **Transformer 메커니즘**: 전역적 문맥 정보(Global Context)를 포착하여 CNN의 국소적 수용역(Receptive Field) 한계를 극복하는 구조.

## 📎 Related Works

의료 영상 분할의 기초가 된 Fully Convolutional Networks (FCN)는 완전 연결 계층(Fully Connected Layer)을 컨볼루션 계층으로 대체하여 픽셀 수준의 시맨틱 분할을 가능하게 한 선구적인 모델이다. 이후 Ronneberger 등이 제안한 U-Net은 인코더(Encoder)를 통한 특징 추출과 디코더(Decoder)를 통한 해상도 복원, 그리고 그 사이를 잇는 Skip-connection을 도입하여 적은 양의 데이터로도 높은 성능을 낼 수 있음을 입증하였다. 기존의 접근 방식들은 주로 국소적인 특징 추출에 집중하였으나, 최근 연구들은 전역적 의존성 파악과 3차원 데이터 처리, 그리고 라벨링 데이터 부족 문제를 해결하기 위한 도메인 적응(Domain Adaptation) 방향으로 발전하고 있다.

## 🛠️ Methodology

본 논문은 U-Net의 변형 모델들을 크게 네 가지 범주로 나누어 분석한다.

### 1. 구조적 변형 모델
*   **인코더 최적화**: Y-Net, $\Psi$-Net과 같이 인코더의 수를 늘리거나 병렬 구조를 도입하여 복잡한 타겟의 특징 추출 능력을 강화한다. $\Psi$-Net의 경우 3개의 병렬 인코더와 Self-attention을 통해 전역 특징을 캡처한다.
*   **다중 U-Net 네트워크**: W-Net(중첩 구조), Triple U-Net(RGB, H-channel, 분할 브랜치의 병렬 구성)과 같이 여러 개의 U-Net을 직렬 또는 병렬로 연결하여 다중 스케일 정보와 다양한 모달리티 특징을 통합한다.
*   **3D U-Net 및 V-Net**: 3D 컨볼루션 커널을 사용하여 볼륨 데이터의 공간적 상관관계를 직접 학습한다. V-Net은 여기에 Residual unit을 추가하여 특징 추출 효율을 높였다.
*   **Transformer 결합**: UNETR, TransUNet과 같이 인코더에 Transformer를 도입하여 전역적 문맥을 학습하고, 디코더의 CNN 구조를 통해 국소적 세부 정보를 복원한다.

### 2. 도메인 적응 및 준지도 학습
데이터 부족 문제를 해결하기 위해 다음과 같은 기법들이 분석되었다.
*   **SCP-Net**: Self-aware 및 Cross-sample Prototype Network를 통해 일관성 학습(Consistency Learning)을 수행하여 라벨 노이즈 영향을 줄인다.
*   **UVCGAN**: CycleGAN 구조에 ViT(Vision Transformer) 병목 구간을 추가하여 비쌍방향(Unpaired) 이미지 변환 성능을 높였다.
*   **DDSP**: Inter-channel similarity feature alignment (IFA)와 시맨틱 일관성 손실을 통해 소스 도메인과 타겟 도메인의 특징을 정밀하게 정렬한다.

### 3. 성능 향상 전략
*   **데이터 증강**: 기하학적 변환(회전, 이동)뿐만 아니라 MixUp, CutMix와 같은 고급 이미지 혼합 기법 및 AutoAugment를 통한 자동 최적화를 사용한다.
*   **네트워크 모듈 개선**: MultiRes block, Dense connection, Attention Gate 등을 도입하여 특징 재사용성을 높이고 불필요한 정보를 억제한다.

### 4. 평가 지표 (Evaluation Metrics)
모델의 성능을 측정하기 위해 사용되는 주요 수식은 다음과 같다.
*   **Dice Similarity Coefficient (DSC)**:
    $$DSC = \frac{2 \times \text{Area}_{\text{pre}} \cap \text{Area}_{\text{tru}}}{\text{Area}_{\text{pre}} + \text{Area}_{\text{tru}}}$$
*   **Jaccard Index (JI)**:
    $$JI = \frac{\text{Area}_{\text{pre}} \cap \text{Area}_{\text{tru}}}{\text{Area}_{\text{pre}} \cup \text{Area}_{\text{tru}}}$$
*   **Mean Intersection over Union (MIoU)**: 각 클래스별 IoU의 평균값으로 계산한다.
    $$MIoU = \frac{1}{m} \sum_{i=1}^{m} IoU_i$$
*   **Hausdorff Distance (HD)**: 예측된 윤곽선과 실제 윤곽선 사이의 최대 거리를 측정하여 경계 정확도를 평가한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 다양한 문헌의 결과를 종합한 비교 분석표를 제시한다.

*   **모달리티별 적합 모델**: CT 데이터(LIDC-IDRI, LiTS)에는 3D U-Net, V-Net, DenseNet+U-Net이, MRI 데이터(BraTS)에는 Attention U-Net, nnU-Net, UNETR 등이 효과적인 것으로 나타났다.
*   **정량적 성능**:
    *   **Swin-UNet**은 ACDC 데이터셋에서 $DSC = 79.13\%$를 기록하였다.
    *   **BRAU-Net++**는 ISIC-2018 및 CVC-ClinicDB 데이터셋에서 각각 $90.1\%$, $92.94\%$의 높은 DSC를 달성하며 SOTA 성능을 보였다.
    *   **UNETR**는 BTCV 데이터셋에서 $DSC = 83.28\%$ 수준의 성능을 보이며 3D 이미지의 전역 특징 포착 능력을 입증하였다.
*   **효율성**: EMCAD 프레임워크는 기존 방법 대비 파라미터 수를 $79.4\%$, FLOPs를 $80.3\%$ 감소시키면서도 높은 정확도를 유지하였다.

## 🧠 Insights & Discussion

### 강점 및 한계 분석
분석 결과, U-Net 변형 모델들은 다음과 같은 공통적인 한계점을 가진다.
1.  **전역 특징 포착의 한계**: CNN 기반 모델은 수용역의 제한으로 인해 전역적 문맥 파악이 어렵다 (예: H-DenseUNet).
2.  **높은 계산 비용**: 고해상도 의료 영상과 깊은 네트워크 구조로 인해 VRAM 소모가 매우 크다 (예: V-Net, TransUNet).
3.  **일반화 능력 부족**: 특정 데이터셋에서 학습된 모델이 다른 도메인의 데이터셋으로 전이될 때 성능이 급격히 저하된다.
4.  **경계 국소화 능력 저하**: 특히 Transformer 기반 모델은 시퀀스 관계에 집중하여 픽셀 간의 세밀한 공간 관계를 놓치는 경향이 있어 경계선이 부정확하다.

### 비판적 해석 및 제언
저자는 이러한 문제를 해결하기 위해 다음과 같은 전략을 제안한다. 전역 특징 포착을 위해서는 Transformer의 Self-attention을 결합하되, 경계 정확도를 높이기 위해 Boundary-aware 모듈이나 다중 스케일 특징 융합 전략을 병행해야 한다. 또한, 계산 효율성을 위해 Depthwise separable convolution이나 모델 가지치기(Pruning), 양자화(Quantization) 도입이 필수적이다. 특히 임상 적용을 위해서는 모델의 결정 과정을 시각화하는 '해석 가능성(Interpretability)' 연구가 향후 핵심 과제가 될 것이다.

## 📌 TL;DR

본 논문은 의료 영상 분할의 표준인 U-Net과 그 변형 모델들을 구조적/기능적 관점에서 종합적으로 분석한 리뷰 논문이다. Jump/Residual connection, 3D 확장, Transformer 결합이라는 4대 핵심 메커니즘이 어떻게 성능을 향상시키는지 체계적으로 정리하였으며, 각 의료 영상 모달리티에 적합한 모델 맵을 제공한다. 전역 특징 포착과 계산 효율성, 경계 정밀도 사이의 트레이드-오프 관계를 명시하고, 향후 SAM(Segment Anything Model)의 의료 분야 최적화 및 모델 해석 가능성 연구의 중요성을 강조한다.