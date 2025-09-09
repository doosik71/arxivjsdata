# A Comprehensive Analysis of Mamba for 3D Volumetric Medical Image Segmentation
Chaohan Wang, Yutong Xie, Qi Chen, Yuyin Zhou, Qi Wu

## 🧩 Problem to Solve

체적 의료 영상 분할(Volumetric medical image segmentation)은 임상 응용 분야에서 매우 중요하지만, 기존 딥러닝 모델들은 다음과 같은 한계에 직면해 있습니다.

*   **CNN (예: U-Net)의 한계**: 수용 필드(receptive field)가 제한적이고 지역성 편향(locality bias)이 있어 장거리 종속성(long-range dependency) 및 전역적 문맥(global context)을 효과적으로 캡처하기 어렵습니다.
*   **Transformer의 한계**: 동적 자체 주의 메커니즘을 통해 장거리 종속성을 모델링할 수 있지만, 고해상도 3D 데이터 처리 시 극심한 계산 복잡도와 메모리 요구 사항(Out of Memory, OOM)으로 인해 비실용적입니다.
*   **Mamba의 잠재력**: 선택적 상태 공간 모델(Selective State Space Models, SSM) 기반의 Mamba는 Transformer보다 계산 효율성이 높아 장거리 종속성 모델링을 위한 유망한 대안으로 부상했지만, 3D 의료 영상 분할에서의 광범위한 기능과 잠재적 이점에 대한 포괄적인 분석이 부족합니다.

이 연구는 Mamba의 3D 의료 영상 분할 능력에 대한 심층적인 조사를 통해 다음 세 가지 핵심 질문에 답하고자 합니다: Mamba가 Transformer를 대체할 수 있는가? 다중 스케일(multi-scale) 표현 학습을 향상시킬 수 있는가? 복잡한 스캔(scanning) 전략이 Mamba의 잠재력을 최대한 발휘하는 데 필수적인가?

## ✨ Key Contributions

*   **Mamba의 역할에 대한 포괄적 분석**: 3가지 대규모 공용 벤치마크 데이터셋(AMOS, TotalSegmentator, BraTS)을 사용하여 3D 의료 영상 분할에서 Mamba의 역할과 과제를 철저히 분석하고, 향후 연구를 위한 통찰력과 기반을 제공했습니다.
*   **Mamba 전용 기능 제안**: 1D Depthwise Convolution (DWConv)을 3D DWConv으로 대체, 다중 스케일 Mamba 블록(MSv4) 개발, 3D 데이터에 특화된 Tri-scan 전략 설계 등 Mamba의 기능을 탐색하고 향상시키기 위한 태스크별 접근 방식을 제안했습니다.
*   **새로운 벤치마크 모델 UlikeMamba$_{3d}$MT 제시**: 검증된 전략(3D DWConv, MSv4, Tri-scan)을 통합하여 3D 의료 영상 분할을 위한 새로운 Mamba 기반 네트워크인 UlikeMamba$_{3d}$MT를 구축했습니다. 이 모델은 nnUNet, CoTr, UNETR, SwinUNETR, U-Mamba와 같은 선도적인 CNN 및 Transformer 기반 모델, 그리고 기존 Mamba 기반 모델보다 경쟁력 있는 정확도와 우수한 계산 효율성을 제공하며 새로운 벤치마크를 수립했습니다.
*   **Transformer 대체 가능성 입증**: U-모양 Mamba 기반 네트워크(UlikeMamba)가 3D Depthwise Convolution으로 강화되었을 때 U-모양 Transformer 기반 네트워크(UlikeTrans)를 일관되게 능가하며 정확도와 계산 효율성을 모두 향상시켰습니다.
*   **다중 스케일 표현 학습 강화**: 제안된 다중 스케일 Mamba 블록(MSv4)이 복잡한 분할 작업, 특히 TotalSegmentator 데이터셋에서 미세한 디테일과 전역적 문맥을 모두 캡처하는 데 우수한 성능을 보였습니다.
*   **스캔 전략의 필요성 분석**: 단순한 스캔 방법으로도 충분한 경우가 많음을 발견했으며, Tri-scan 접근 방식이 가장 어려운 시나리오에서 현저한 이점을 제공함을 입증했습니다.

## 📎 Related Works

*   **Transformer 기반 분할 네트워크**:
    *   **TransUNet [16]**: CNN 기반 지역 특징 추출과 Transformer 기반 전역 종속성 모델링을 결합한 U-Net 아키텍처.
    *   **TransHRNet [17]**: 병렬 다중 해상도 전략으로 계산 복잡도 해결.
    *   **UCTNet [18]**: CNN 기반 불확실성 맵으로 식별된 불확실한 영역에 Transformer를 선택적으로 적용.
    *   **kMaXU [19]**: CNN 및 Transformer 인코더를 통합하고 다중 스케일 k-Means Mask Transformer 블록으로 클래스 불균형 해결.
    *   **공통적 한계**: 고해상도 3D 의료 영상 처리 시 높은 계산 복잡성 유지.

*   **Mamba 기반 분할 네트워크**:
    *   **Mamba [5]**: Transformer에 비해 우수한 메모리 효율성과 계산 속도로 장거리 종속성 캡처.
    *   **하이브리드 모델 (Mamba + CNN)**:
        *   **SegMamba [9]**: 인코더에 다중 방향 Mamba 모듈 사용.
        *   **U-Mamba [8]**: 인코더 및 디코더 모두에서 Mamba와 CNN 결합.
        *   **LKM-UNet [20], Polyp-Mamba [21], H-vmunet [22], EM-Net [23], Tri-Plane Mamba [24]** 등 다양한 변형 연구.
    *   **순수 Mamba 기반 모델**:
        *   **VM-UNet [25]**: 전체 아키텍처에 VSS 블록 채택.
        *   **Swin-UMamba [10]**: 인코더의 CNN 블록을 Visual State-Space (VSS) 블록으로 대체.
        *   **UD-Mamba [26], Mamba-UNet [27]** 등 순수 Mamba 기반 인코더-디코더 구조 연구.
    *   **기존 연구의 한계**: 주로 Mamba의 실현 가능성을 검증하는 데 초점을 맞추었으며, 그 영향과 잠재적 장점에 대한 포괄적인 분석은 부족했습니다.

## 🛠️ Methodology

이 연구는 3가지 주요 분석을 통해 3D 의료 영상 분할에서 Mamba의 능력을 종합적으로 탐구합니다.

1.  **Mamba vs. Transformer (UlikeMamba vs. UlikeTrans)**:
    *   **목표**: 3D 의료 영상 분할에서 Transformer 기반 아키텍처를 Mamba 기반 네트워크로 대체할 수 있는지 평가합니다.
    *   **모델 디자인**: U자형 인코더-디코더 구조를 따르는 UlikeMamba(Mamba 기반)와 UlikeTrans(Transformer 기반)를 설계했습니다 (Fig. 1).
    *   **Mamba 레이어 개선**: 기존 Mamba의 1D Depthwise Convolution (DWConv) [15]을 3D DWConv로 대체하여 체적 데이터의 3D 공간 일관성을 보존하고 지역 특징을 더 잘 캡처하도록 했습니다.
    *   **Transformer 레이어**: UlikeTrans에서는 3D 체적 영상의 높은 계산 복잡성을 줄이기 위해 Spatial-Reduction Attention (SRA) [28]을 적용했습니다.
    *   **학습 설정**: nnUNet [3] 프레임워크를 사용하고 AdamW [29] 옵티마이저로 1000 epoch 동안 학습시켰습니다.
    *   **평가 지표**: Dice 계수(정확도) 및 FLOPs(계산 복잡도)를 사용했습니다.

2.  **Mamba의 다중 스케일 모델링 잠재력 (Multi-Scale Modeling)**:
    *   **목표**: Mamba의 장거리 종속성 모델링 능력이 다중 스케일 표현 학습을 향상시킬 수 있는지 탐구합니다.
    *   **다중 스케일 모델링 스킴 (Fig. 3)**:
        *   **MSv1**: 3x3x3 및 7x7x7 커널을 가진 두 개의 병렬 컨볼루션 레이어에서 특징을 추출한 후, Mamba 또는 Transformer 레이어를 거쳐 요소별 합산(element-wise summation)으로 통합합니다.
        *   **MSv2**: 3x3x3 및 7x7x7 컨볼루션 출력부를 연결(concatenate)한 후, Mamba 또는 Transformer 레이어를 거칩니다.
        *   **MSv3**: 3x3x3, 5x5x5, 7x7x7 커널을 가진 세 개의 병렬 컨볼루션 레이어 출력부를 연결한 후, Mamba 또는 Transformer 레이어를 거칩니다.
        *   **MSv4 (Mamba 전용)**: 세 개의 3D DWConv 레이어(3x3x3, 5x5x5, 7x7x7)로 다중 스케일 특징을 추출하고 연결한 후, Mamba의 상태 공간 모델(SSM)로 처리합니다.
    *   **적용**: UlikeTrans$_{SRA}$ 및 UlikeMamba$_{3d}$ 모델의 인코더 단계(ES1-ES4)에 MSv1, MSv2, MSv3 스킴을 적용했으며, MSv4는 UlikeMamba$_{3d}$에만 적용했습니다.

3.  **다중 스캔 전략 vs. 단일 스캔 전략 (Multi-scan Strategy vs. Single-scan Strategy)**:
    *   **목표**: 3D 의료 영상 분할에서 복잡한 스캔 전략의 필요성을 평가합니다.
    *   **스캔 전략 (UlikeMamba$_{3d}$에 적용) (Fig. 4)**:
        *   **Single-scan (forward)**: 체적 특징을 평탄화하고 단일 축을 따라 순방향으로 순차 처리합니다.
        *   **Dual-scan (forward+backward)**: 동일 축을 따라 순방향 및 역방향으로 두 번 스캔하고, 두 스캔의 특징을 병합합니다.
        *   **Dual-scan (forward+random)**: 표준 순방향 스캔과 무작위 순서 스캔을 결합하여 병합합니다.
        *   **Tri-scan (left-right, up-down, front-back)**: 좌-우, 상-하, 전-후의 세 방향으로 스캔하고, 각 스캔을 별도의 SSM 레이어를 통해 처리한 후, 출력을 병합하여 3D 볼륨의 통합된 표현을 생성합니다.

4.  **최종 통합 모델 (UlikeMamba$_{3d}$MT)**:
    *   앞선 분석에서 검증된 최적의 전략(Mamba 레이어의 3D DWConv, MSv4 다중 스케일 전략, Tri-scan)을 단일 모델에 통합했습니다.
    *   이 모델의 성능을 nnUNet, CoTr, UNETR, SwinUNETR, U-Mamba 등과 같은 최첨단 기준 모델들과 비교하여 검증했습니다.

## 📊 Results

*   **분석 1: Mamba vs. Transformer**:
    *   바닐라 UlikeTrans (순수 자기 주의)는 OOM 문제를 겪었습니다.
    *   UlikeMamba$_{1d}$ (1D DWConv 사용)는 UlikeTrans$_{SRA}$와 유사한 Dice 점수를 달성하면서도 더 적은 파라미터와 FLOPs (44.88 GFLOPs vs. 64.47 GFLOPs)를 기록하여 Mamba의 효율성을 입증했습니다.
    *   제안된 UlikeMamba$_{3d}$ (3D DWConv 사용)는 모든 데이터셋에서 UlikeMamba$_{1d}$ 및 UlikeTrans$_{SRA}$를 일관되게 능가하며 평균 Dice 87.45를 달성했고, FLOPs는 46.03 GFLOPs로 유지되었습니다 (Table 1). 이는 3D DWConv가 Mamba의 성능을 크게 향상시킴을 보여줍니다.

*   **분석 2: Mamba의 다중 스케일 모델링 잠재력**:
    *   모든 다중 스케일 스킴은 UlikeTrans$_{SRA}$와 UlikeMamba$_{3d}$ 모두에서 성능 향상을 가져왔지만, UlikeMamba$_{3d}$는 더 높은 정확도와 낮은 계산 비용을 유지하며 일관되게 우수한 성능을 보였습니다 (Table 2).
    *   특히, UlikeMamba$_{3d}$에 적용된 MSv4 스킴은 가장 높은 평균 Dice 점수(88.01)와 62.23 GFLOPs를 기록하며 가장 효율적인 다중 스케일 전략임을 입증했습니다.
    *   TotalSegmentator와 같은 복잡한 데이터셋(117개 클래스)에서 다중 스케일 전략으로 인한 성능 향상이 가장 두드러졌습니다.

*   **분석 3: 다중 스캔 전략 vs. 단일 스캔 전략**:
    *   Tri-scan 방식이 모든 데이터셋에서 가장 높은 평균 Dice 점수(87.93)를 달성했지만, 가장 높은 계산 비용(26.38M 파라미터, 53.09G FLOPs)이 발생했습니다 (Table 3).
    *   Dual-scan (forward+backward) 및 Dual-scan (forward+random)은 Single-scan에 비해 미미한 성능 향상만을 보였습니다.
    *   Single-scan은 가장 낮은 계산 요구 사항(24.30M 파라미터, 46.03G FLOPs)으로도 경쟁력 있는 성능(평균 Dice 87.45)을 제공했습니다.
    *   Tri-scan의 이점은 TotalSegmentator와 같이 복잡한 공간 관계 캡처가 필수적인 작업에서 가장 명확하게 나타났습니다.

*   **최첨단 모델과의 비교 (UlikeMamba$_{3d}$MT)**:
    *   제안된 UlikeMamba$_{3d}$MT (3D DWConv, MSv4, Tri-scan 통합)는 AMOS (89.95) 및 BraTS (90.60) 데이터셋에서 경쟁력 있는 Dice 점수를 달성했습니다 (Fig. 6).
    *   nnUNet, CoTr, UNETR, SwinUNETR, U-Mamba 등 기존 최첨단 모델들보다 더 낮은 계산 비용(93.09 GFLOPs)으로 우수한 성능을 보이며 새로운 벤치마크를 수립했습니다.

## 🧠 Insights & Discussion

*   **Mamba의 효율성과 효과성**: Mamba는 3D 의료 영상 분할에서 Transformer를 대체할 수 있는 효과적이고 계산 효율적인 대안임을 입증했습니다. 특히 3D Depthwise Convolution을 통합할 때 성능이 크게 향상되며, 고해상도 체적 데이터 처리 시 Transformer의 OOM 문제를 회피할 수 있습니다.
*   **3D 공간 일관성의 중요성**: 바닐라 Mamba의 1D DWConv 대신 3D DWConv를 사용하는 것이 체적 데이터의 3D 공간 일관성을 유지하고 지역 특징을 효과적으로 캡처하는 데 필수적이며, 이는 전반적인 분할 정확도를 높이는 데 기여합니다.
*   **다중 스케일 표현 학습의 역할**: Mamba 기반 모델, 특히 MSv4와 같은 다중 스케일 전략을 사용할 때, 다양한 크기의 해부학적 구조를 포함하는 복잡한 분할 작업에서 미세한 디테일과 전역적 문맥 정보를 효과적으로 캡처할 수 있습니다. Mamba는 본질적으로 장거리 종속성 모델링에 효율적이므로, Transformer에 비해 다중 스케일 전략으로부터 얻는 *상대적인* 성능 향상은 작을 수 있습니다.
*   **스캔 전략의 실용성**: 3D 의료 데이터의 강한 구조적 사전 정보(structural priors)로 인해 단일 스캔과 같은 단순한 스캔 전략으로도 많은 응용 분야에서 충분한 성능을 제공할 수 있습니다. 그러나 가장 까다로운 시나리오에서는 Tri-scan과 같은 포괄적인 다방향 스캔이 공간적 관계를 더 완벽하게 캡처하여 성능을 크게 향상시킬 수 있습니다. Dual-scan 방법은 제한적인 이점을 제공했습니다.
*   **UlikeMamba$_{3d}$MT의 변혁적 잠재력**: 제안된 UlikeMamba$_{3d}$MT는 Mamba의 효율성, 3D DWConv, 다중 스케일 모델링, Tri-scan 전략을 성공적으로 통합하여 3D 의료 영상 분할 분야에서 새로운 성능 기준을 제시하며 Mamba가 이 분야의 선도적인 아키텍처가 될 수 있음을 시사합니다.

*   **한계점**:
    *   Mamba의 우수성에 대한 검증은 주로 경험적 결과에 의존하며, 수학적 유도를 통한 심층적인 이론적 분석이 부족합니다.
    *   현재 Mamba 기반 아키텍처는 여전히 컨볼루션 커널을 통합하고 있습니다. 향후 연구에서는 고해상도 3D 의료 영상 분할을 위한 완전히 컨볼루션 없는(convolution-free) 순수 Mamba 기반 아키텍처 개발을 목표로 할 수 있습니다.

## 📌 TL;DR

**문제**: 3D 의료 영상 분할은 임상적으로 중요하지만, CNN은 전역 문맥 파악에, Transformer는 고해상도 데이터 처리 시 계산 복잡도와 OOM 문제로 어려움을 겪습니다. Mamba는 효율적인 대안으로 부상했으나, 그 능력에 대한 포괄적 분석이 부족했습니다.

**해결책**: 본 연구는 Mamba의 3D 의료 영상 분할 능력을 심층적으로 분석하기 위해 Mamba와 Transformer를 비교하고, 다중 스케일 모델링 기법을 평가하며, 다양한 스캔 전략의 필요성을 탐구했습니다. 특히, 3D DWConv를 Mamba에 도입하고, 다중 스케일 Mamba 블록(MSv4)과 Tri-scan 전략을 설계하여 Mamba의 성능을 극대화했습니다.

**핵심 결과**:
1.  **Mamba의 Transformer 능가**: 3D DWConv로 강화된 Mamba 기반 UlikeMamba는 Transformer 기반 UlikeTrans보다 정확도와 계산 효율성 면에서 우수합니다.
2.  **다중 스케일 학습 강화**: Mamba 기반 모델, 특히 MSv4는 복잡한 분할 작업에서 미세한 디테일과 전역적 문맥을 효과적으로 캡처합니다.
3.  **스캔 전략의 효과**: 단순한 스캔 방법으로도 충분한 경우가 많지만, Tri-scan은 가장 어려운 시나리오에서 포괄적인 공간 관계 캡처를 통해 상당한 성능 향상을 제공합니다.
4.  **새로운 벤치마크 수립**: 이러한 발전 사항들을 통합한 UlikeMamba$_{3d}$MT 모델은 기존의 최첨단 모델들(nnUNet, CoTr, U-Mamba 등)을 뛰어넘는 경쟁력 있는 정확도와 탁월한 계산 효율성을 제공하며 3D 의료 영상 분할의 새로운 벤치마크를 제시했습니다.

결론적으로, Mamba는 3D 의료 영상 분할을 위한 효율적이고 정확한 접근 방식을 위한 변혁적인 동력으로 자리매김할 수 있습니다.