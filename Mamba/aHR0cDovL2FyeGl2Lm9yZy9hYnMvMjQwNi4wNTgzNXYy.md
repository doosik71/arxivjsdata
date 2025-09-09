# Mamba YOLO: A Simple Baseline for Object Detection with State Space Model
Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu, Hongbo Li

## 🧩 Problem to Solve
최근 딥러닝 기술의 발전으로 YOLO 시리즈는 실시간 객체 탐지기의 새로운 벤치마크를 제시했습니다. Transformer 기반 구조는 모델의 수용 필드를 크게 확장하고 상당한 성능 향상을 달성하며 이 분야에서 가장 강력한 솔루션으로 부상했습니다. 그러나 self-attention 메커니즘의 이차 복잡성은 모델의 계산 부담을 증가시키는 단점이 있습니다. 본 논문은 이러한 이차 복잡성 문제를 해결하고, 동시에 YOLO 모델의 실시간 요구 사항을 충족하며, 사전 학습 없이도 효율적으로 훈련될 수 있는 객체 탐지 모델을 제안하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **선형 복잡성을 가진 SSM 백본 제안:** self-attention의 이차 복잡성을 해결하기 위해 선형 복잡성의 State Space Model (SSM)을 도입한 ODMamba 백본을 제안합니다. 이 모델은 사전 학습 없이도 훈련이 간단합니다.
*   **ODMamba 매크로 구조 설계:** 실시간 요구 사항을 위해 ODMamba의 매크로 구조를 설계하고 최적의 스테이지 비율 및 스케일링 크기를 결정했습니다.
*   **RG Block 설계:** SSM의 수용 필드 부족 및 약한 이미지 지역화와 같은 시퀀스 모델링의 잠재적 한계를 해결하기 위해 다중 브랜치 구조를 사용하여 채널 차원을 모델링하는 Residual Gated (RG) Block을 설계했습니다. 이는 지역 이미지 의존성을 더 정확하고 중요하게 포착합니다.

## 📎 Related Works
*   **실시간 객체 탐지기:** YOLOv6, YOLOv7, YOLOv8과 같은 YOLO 시리즈 모델들은 DarkNet, E-ELAN, CSPDarknet53, C2f와 같은 백본 및 Neck 구조 개선을 통해 성능을 향상시켜 왔습니다. 특히 Gold YOLO는 Gather-and-Distribute (GD) 메커니즘을 도입하여 SOTA를 달성했습니다.
*   **End-to-End 객체 탐지기:** DETR, Deformable DETR, DINO, RT-DETR과 같은 Transformer 기반 모델들은 앵커 생성 및 비최대 억제와 같은 수작업 구성 요소를 우회하여 강력한 전역 모델링 기능을 제공하지만, 대규모 데이터셋 사전 학습, 높은 계산 비용, 작은 객체 탐지 등의 문제가 있습니다.
*   **Vision State Space Models:** S4, Mamba와 같은 Structured Self-Modulation (SSM) 기반 모델들은 선형 시간 복잡성과 긴 시퀀스 모델링 능력으로 주목받고 있습니다. Vision Mamba, VMamba (Cross-Scan 모듈 도입), LocalMamba 등은 비전 분야에 SSM을 적용하려는 시도를 보였습니다.

## 🛠️ Methodology
Mamba YOLO는 SSM 기반의 ODMamba 백본과 Neck 부분으로 구성됩니다.

1.  **Mamba YOLO 전체 아키텍처:**
    *   **Simple Stem:** 기존 ViT의 패치 분할 대신, 스트라이드 2, 커널 크기 3의 두 가지 컨볼루션을 사용하여 초기 피처 맵을 생성합니다.
    *   **Vision Clue Merge:** SS2D의 선택적 동작을 방해하지 않도록, 정규화를 제거하고 차원 맵을 분할하며, 여분의 피처 맵을 채널 차원에 추가한 후 $4\times$ 압축 포인트와이즈 컨볼루션(pointwise convolution)을 사용하여 다운샘플링합니다.
    *   **ODMamba Backbone:** Simple Stem, ODSSBlock, Vision Clue Merge 모듈로 구성됩니다.
    *   **Neck:** PAFPN 디자인을 따르며, C2f 모듈 대신 ODSSBlock을 사용하여 풍부한 그라디언트 흐름과 피처 융합을 처리합니다.
    *   **Decoupled Head:** Neck에서 얻은 $\{P_3, P_4, P_5\}$ 피처를 기반으로 탐지 결과를 출력합니다.

2.  **ODSSBlock (Object Detection State Space Block):** Mamba YOLO의 핵심 모듈입니다.
    *   입력 피처 $Z_{l-3}$는 ConvModule을 통과하여 $Z_{l-2}$가 됩니다.
    *   **SS2D (Selective Scan 2D):** `LayerNorm` 후 $Z_{l-2}$에 적용됩니다. 입력 이미지를 네 가지 대칭 방향(상하, 하상, 좌우, 우좌)으로 스캔하여 확장(Scan Expansion)합니다. 이 확장된 시퀀스는 Mamba의 핵심인 S6 블록을 통해 처리된 후, Scan Merge를 통해 병합되어 전역 피처를 추출합니다. SSM의 선형 시간 복잡성을 유지하면서 전역적인 의존성을 모델링합니다.
    *   **RG Block (Residual Gated Block):** `LayerNorm` 후 $Z_{l-1}$에 적용됩니다.
        *   입력 $f'_A$와 $f'_B$로부터 두 개의 브랜치 $R^{l-1}_{\text{local}}$과 $R^{l-1}_{\text{global}}$을 생성합니다.
        *   $R^{l-1}_{\text{local}}$ 브랜치에는 `Depth-wise Convolution (DWConv)`을 위치 인코딩 모듈로 사용하고 잔차 연결을 통해 효율적인 그래디언트 흐름을 유도합니다.
        *   로컬 정보는 $Y(x) = \Phi(\text{DWConv}(x) \oplus x)$ ($x$는 활성화 함수 포함)를 통해 처리됩니다.
        *   $R^{l-1}_{\text{global}}$과 $Y(R^{l-1}_{\text{local}})$을 곱하여 정보 융합($R^l_{\text{fusion}} = R^{l-1}_{\text{global}} \odot Y(R^{l-1}_{\text{local}})$)을 수행한 후, 선형 레이어와 원래 입력 $f'_A$의 잔차 연결을 통해 최종 출력 $f_{\text{RG}}$를 생성합니다.
        *   게이팅 메커니즘과 깊이 분리 컨볼루션 잔차 연결을 통해 SSM의 이미지 지역화 능력과 수용 필드 부족 문제를 보완하고 로컬 및 전역 특징을 효과적으로 포착합니다.

## 📊 Results
*   **MSCOCO 벤치마크 성능:** Mamba YOLO는 MSCOCO 데이터셋에서 SOTA 객체 탐지기들과 비교하여 성능-효율성 측면에서 뛰어난 트레이드오프를 보여줍니다.
    *   **Mamba YOLO-T (Tiny 버전):** PPYOLOE-S 또는 YOLO-MS-XS 대비 `mAP`가 1.1% / 1.5% 향상되었으며, GPU 추론 지연 시간은 0.9ms / 0.2ms 감소했습니다. YOLOv8-S와 유사한 정확도에서 매개변수 수를 48%, FLOPs를 53% 줄이고 GPU 추론 지연 시간을 0.4ms 단축했습니다.
    *   **Mamba YOLO-B (Base 버전):** Gold-YOLO-M 대비 `mAP`가 3.7% 더 높았으며, PPYOLOE-M 대비 매개변수 수를 18%, FLOPs를 9% 줄이고 GPU 추론 지연 시간을 1.8ms 단축했습니다.
    *   **Mamba YOLO-L (Large 버전):** 최고 성능의 Gold-YOLO-L과 비교하여 `mAP` 0.3% 향상 및 매개변수 수 0.9% 감소를 달성했습니다.
*   **효율성:** DINO-R50과 비교 시, 입력 이미지 해상도 증가에 따른 GPU 메모리 및 FLOPs 증가가 DINO는 이차 함수적 경향을 보이는 반면, Mamba YOLO는 선형적 증가를 유지하며 효율성에서 우위를 보였습니다.
*   **Ablation Study (모듈별):** ODSSBlock, RG Block, Vision Clue Merge가 각각 모델의 정확도 향상에 기여함을 확인했습니다. 특히 Vision Clue Merge는 SSM을 위해 더 많은 시각적 단서를 보존하여 성능을 높였습니다.
*   **Ablation Study (RG Block 구조):** 게이팅 메커니즘과 깊이 분리 컨볼루션 잔차 연결을 통합한 RG Block이 다른 MLP 변형들(Original, Convolutional, Res-Convolutional, Gated MLP)보다 가장 높은 정확도 향상을 가져왔습니다.
*   **Ablation Study (ODSSBlock 반복 수 및 Feature Map):** 백본에서 ODSSBlock의 반복 수는 `[3,6,6,3]`이 가장 적절한 구성이었으며, Neck 부분에 ODSSBlock을 사용하는 것이 정확도를 높이는 데 효과적이었습니다. 출력 Feature Map으로 `{P_3, P_4, P_5}`를 사용하는 것이 정확도와 복잡성 사이의 균형을 가장 잘 이루었습니다.
*   **시각화:** 복잡한 배경, 심하게 겹치거나 가려진 객체에 대해서도 정확한 탐지 결과를 보여주었습니다.

## 🧠 Insights & Discussion
Mamba YOLO는 YOLO의 실시간 객체 탐지 성능과 Transformer의 전역 모델링 능력을 결합하려는 효과적인 시도를 보여줍니다. Transformer의 주요 병목인 self-attention의 이차 복잡성을 선형 복잡성을 가진 SSM으로 대체하여 계산 효율성을 크게 향상시켰습니다. 특히, 대규모 데이터셋 사전 학습 없이도 경쟁력 있는 성능을 달성할 수 있다는 점은 실제 적용 가능성을 높입니다.

ODSSBlock과 RG Block의 설계는 SSM이 가지는 시퀀스 모델링의 잠재적 한계(수용 필드 부족, 약한 이미지 지역화)를 보완하는 중요한 기여를 합니다. RG Block의 게이팅 메커니즘과 깊이 분리 컨볼루션 잔차 연결은 계층적 구조 내에서 중요한 특징을 효과적으로 전파하고 로컬 및 전역 특징을 모두 포착하는 데 기여합니다.

이 연구는 Mamba 아키텍처를 실시간 객체 탐지 작업에 적용한 첫 번째 탐색이며, YOLO 시리즈에 새로운 기준점을 제시함으로써 해당 분야 연구자들에게 새로운 아이디어를 제공할 것으로 기대됩니다.

## 📌 TL;DR
Transformer의 이차 복잡성 문제를 해결하면서 YOLO의 실시간 객체 탐지 성능을 향상시키기 위해, 본 논문은 선형 복잡성을 가진 State Space Model (SSM) 기반의 **Mamba YOLO**를 제안합니다. Mamba YOLO는 핵심 모듈인 **ODSSBlock** (Selective Scan 2D (SS2D)와 Residual Gated (RG) Block 포함)을 백본과 Neck에 적용하여 전역 및 로컬 특징 포착 능력을 강화합니다. 그 결과, Mamba YOLO는 MSCOCO 벤치마크에서 기존 SOTA 모델 대비 더 나은 `mAP`와 효율성을 달성하며, 대규모 사전 학습 없이도 경쟁력 있는 성능을 보였습니다.