# MGFI-Net: A Multi-Grained Feature Integration Network for Enhanced Medical Image Segmentation

Yucheng Zeng (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation, MIS)은 임상 응용 분야에서 매우 중요한 역할을 하지만, 노이즈가 많은 배경, 낮은 대비(low contrast), 그리고 복잡한 해부학적 구조로 인해 관심 영역(Region of Interest, ROI)을 정확하게 획정하는 데 어려움이 있다. 기존의 분할 모델들은 주로 국소적 특징(local features) 추출에 집중하여 넓은 문맥 정보(broader contextual cues)와 세부적인 국소 정보를 효과적으로 통합하지 못하는 한계가 있다. 이로 인해 다중 입도(multi-grained) 정보의 통합이 부족해지며, 결과적으로 복잡한 형태의 객체를 분할할 때 경계선이 모호해지거나 불완전한 분할 결과가 나타나는 문제가 발생한다. 본 논문의 목표는 다중 입도 특징 통합과 적응형 경계 보존 메커니즘을 통해 의료 영상의 복잡한 형태와 노이즈 환경에서도 정밀한 분할 성능을 달성하는 MGFI-Net을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 스케일의 특징 간 계층적 관계를 활용하여 최적의 정보를 선택적으로 추출하고, 변형 가능한 컨볼루션(Deformable Convolution)을 통해 객체의 기하학적 변형에 유연하게 대응하는 것이다. 구체적으로는 다음과 같은 두 가지 핵심 모듈을 설계하였다.

1.  **Multi-Grained Feature Integration (MGFI) Module**: 국소적 세부 사항과 광범위한 문맥 정보를 동시에 캡처하기 위해 다중 분기(multi-branch) 구조를 도입하여 다양한 입도의 특징을 통합한다.
2.  **Adaptive Edge (AE) Module**: 고정된 수용 영역(receptive field)의 한계를 극복하기 위해 Deformable Convolution을 사용하여 커널의 모양과 위치를 동적으로 조정함으로써, 복잡하거나 불규칙한 경계선을 정밀하게 보존한다.

## 📎 Related Works

기존의 의료 영상 분할 연구들은 주로 CNN 기반의 모델들을 활용해 왔다. 대표적으로 U-Net은 skip connection을 통해 세부 정보를 보존하며 널리 사용되었으나, 광범위한 문맥 정보를 캡처하는 능력이 부족하다는 단점이 있다. 이를 개선하기 위해 Attention U-Net은 주의 집중 메커니즘을 통해 관련 영역에 집중하도록 설계되었고, CE-Net은 dilated convolution과 context encoder를 통해 다중 스케일 특징 추출을 시도하였다. 또한 KiU-Net은 overcomplete convolutional 구조를 통해 디테일 보존 능력을 높였으나, 계산 복잡도가 매우 높아 효율성이 떨어진다는 문제가 있다. MGFI-Net은 이러한 기존 모델들이 가진 문맥 정보 통합의 부족함과 경계선 모호성, 그리고 계산 효율성 간의 트레이드-오프 문제를 동시에 해결하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조
MGFI-Net은 **Encoder $\rightarrow$ MGFI Module $\rightarrow$ Decoder $\rightarrow$ AE Module** 순으로 구성된 파이프라인을 가진다. Encoder는 다중 스케일 정보를 추출하고, Decoder는 해상도를 복원하며, 그 사이의 MGFI 모듈과 끝단의 AE 모듈이 각각 문맥 강화와 경계 정밀화를 담당한다.

### Multi-Grained Feature Integration (MGFI) Module
MGFI 모듈은 상단(Upper section)과 하단(Lower section)의 두 부분으로 나뉜다.

1.  **상단 섹션**: 입력 특징 맵 $F_{in}$에 대해 overlapping downsampling을 수행하여 $F_{overlap}$을 생성한다. 이후 depthwise separable convolution을 적용하여 국소 특징 $F_{dw}$를 추출한다. depthwise convolution의 연산은 다음과 같이 정의된다.
    $$y_{depthwise}^{(m)}(p_0) = \sum_{k=1}^{K} w_k^{(m)} \cdot x^{(m)}(p_0 + p_k)$$
    추출된 $F_{dw}$와 $F_{overlap}$은 채널 방향으로 융합된 후, $3 \times 3$ convolution, BatchNorm, ReLU를 거쳐 $F_{concat}$이 된다.
    $$F_{concat} = \text{ReLU}(\text{BatchNorm}(\text{Conv3}(F_{overlap} + F_{dw})))$$
    이후 residual connection을 통해 공간적 세부 사항과 연속성을 보존한다.

2.  **하단 섹션**: 융합된 특징 맵을 세 가지 병렬 분기로 처리하여 다중 입도 특징을 추출한다.
    -   **Deformable Convolution 분기**: 비정형 모양과 비강체 변형(non-rigid deformation)을 처리한다.
    -   **Atrous Convolution 분기**: 추가 연산 비용 없이 수용 영역을 확장하여 광범위한 문맥을 캡처한다.
    -   **Standard $3 \times 3$ Convolution 분기**: 중요한 국소 공간 세부 사항을 보존한다.
    최종적으로 세 분기의 출력($F_{deform}, F_{atrous}, F_{standard}$)을 연결(concatenation)하고 $1 \times 1$ convolution을 통해 압축하여 최종 특징 맵 $F_{final}$을 생성한다.
    $$F_{final} = \text{CONV}(\text{Concat}(F_{deform}, F_{atrous}, F_{standard}))$$

### Adaptive Edge (AE) Module
AE 모듈은 네트워크의 마지막 단계에서 경계선을 정밀하게 다듬는 역할을 한다. $3 \times 3$ convolution으로 오프셋(offset)을 생성하고, 이를 Deformable Convolution에 적용하여 샘플링 위치를 동적으로 조정한다.
$$y(p_0) = \sum_{k=1}^{K} w_k \cdot x(p_0 + p_k + \Delta p_k)$$
여기서 $\Delta p_k$는 학습된 오프셋을 의미한다. 최종적으로 $1 \times 1$ convolution을 통해 단일 채널의 엣지 특징 맵을 생성하여 경계 감독(edge supervision)에 활용한다.

### 손실 함수 (Loss Function)
영역 분할과 경계 정밀도를 동시에 잡기 위해 Cross-Entropy Loss, Dice Loss, Boundary Loss를 결합한 하이브리드 손실 함수 $L_{hybrid}$를 사용한다.
$$L_{hybrid} = L_{Cross-entropy} + L_{Dice} + \lambda \times L_{Boundary}$$
-   **Cross-Entropy Loss**: 픽셀 단위 분류 오차를 측정한다.
-   **Dice Loss**: 클래스 불균형 문제를 해결하여 작은 타겟 영역의 분할 성능을 높인다.
-   **Boundary Loss**: Canny 엣지 검출기로 생성한 ground truth 경계와 모델의 예측 경계를 비교하여 경계선을 정밀화한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: CVC-ClinicDB (폴립), 2018 Data Science Bowl (핵), ISIC-2018 (피부 병변)의 세 가지 공개 데이터셋을 사용하였다.
-   **비교 모델**: U-Net, UNet++, Attention U-Net, CE-Net, KiU-Net, MedFormer.
-   **평가 지표**: Precision, Recall, Accuracy, Dice, IoU 및 효율성 지표(FLOPs, Params, FPS).
-   **구현 세부사항**: ResNet-34를 Encoder로 사용하였으며, NVIDIA L20 GPU 환경에서 PyTorch 1.13.0를 이용하여 학습하였다.

### 주요 결과
1.  **분할 정확도**: 모든 데이터셋에서 MGFI-Net이 SOTA 모델들보다 우수한 성능을 보였다. 특히 CVC-ClinicDB에서 Accuracy 0.9911, Dice 0.9497, IoU 0.9050를 기록하며 가장 높은 정밀도를 달성하였다.
2.  **시각적 분석**: 시각적 결과에서 MGFI-Net은 배경 노이즈를 효과적으로 제거하고, 다른 모델(UNet++, CE-Net 등)이 놓치기 쉬운 복잡한 경계선이나 좁은 영역을 정확하게 분할하는 모습을 보였다.
3.  **효율성 분석**: CVC-ClinicDB 기준, MGFI-Net은 Dice 94.97%라는 높은 정확도를 유지하면서도 FPS 101.45를 기록하여 추론 속도 면에서 매우 효율적임을 입증하였다. 특히 MedFormer와 비교했을 때 훨씬 적은 FLOPs와 Params로 유사하거나 더 높은 성능을 냈다.
4.  **Ablation Study**: MGFI 모듈의 상/하단 섹션을 제거하거나 AE 모듈을 제거했을 때 Dice와 IoU 점수가 유의미하게 하락함을 확인하였다. 이는 다중 입도 특징 통합과 적응형 경계 보존이 모델 성능 향상의 핵심 요소임을 뒷받침한다.

## 🧠 Insights & Discussion

MGFI-Net의 가장 큰 강점은 **정확도와 효율성의 균형**이다. 많은 고성능 모델들이 계산 복잡도를 높여 성능을 올리는 반면, 본 모델은 Deformable Convolution과 Depthwise Separable Convolution 같은 효율적인 연산 구조를 적재적소에 배치하여 실시간 의료 영상 분할 가능성을 보여주었다. 특히, 고정된 필터 크기를 사용하는 기존 CNN의 한계를 AE 모듈의 동적 샘플링으로 해결하여 의료 영상 특유의 불규칙한 형태를 잘 잡아낸 점이 인상적이다.

다만, 본 논문에서 제시된 실험은 주로 2D 이미지 데이터셋에 국한되어 있다. 실제 의료 현장에서는 3D 데이터(MRI, CT 등)가 많이 사용되므로, 제안된 MGFI-Net 구조를 3D로 확장했을 때의 연산 비용 증가 문제와 성능 유지 여부가 향후 중요한 연구 과제가 될 것이다. 또한, $\lambda$와 같은 하이퍼파라미터가 경계 손실과 지역 손실 간의 균형에 어떤 영향을 미치는지에 대한 더 상세한 분석이 추가되었다면 더 완벽한 보고서가 되었을 것이다.

## 📌 TL;DR

MGFI-Net은 의료 영상의 노이즈와 복잡한 경계선 문제를 해결하기 위해 **다중 입도 특징 통합(MGFI) 모듈**과 **적응형 경계(AE) 모듈**을 제안한 모델이다. 이 모델은 다양한 스케일의 문맥 정보와 유연한 경계 특징을 효과적으로 결합하여, 폴립, 핵, 피부 병변 분할 작업에서 SOTA 모델들보다 높은 정확도와 뛰어난 추론 속도(FPS)를 동시에 달성하였다. 이 연구는 고정밀 실시간 의료 영상 분할 시스템 구축을 위한 효율적인 아키텍처 설계 방향을 제시하며, 향후 MRI나 CT와 같은 3D 의료 영상 분야로의 확장 가능성이 높다.