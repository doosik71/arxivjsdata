# Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation

Zifu Wan, Pingping Zhang, Yuhao Wang, Silong Yong, Simon Stepputtis, Katia Sycara, Yaqi Xie (2024)

## 🧩 Problem to Solve

본 연구는 저조도 환경, 과다 노출, 혹은 태양광 반사나 화재와 같은 가혹한 조건에서 기존의 RGB 기반 시각 모델이 겪는 인식 성능 저하 문제를 해결하고자 한다. 이를 위해 열화상(Thermal)이나 깊이(Depth)와 같은 추가적인 모달리티(X-modality)를 결합하는 multi-modal semantic segmentation을 수행한다.

Multi-modal 접근 방식의 핵심은 서로 다른 모달리티가 제공하는 보완적인 정보를 효과적으로 정렬하고 융합하는 것이지만, 기존 방법론들은 다음과 같은 한계를 가진다.
1. **CNN 기반 방식**: 연산 복잡도는 낮으나, 커널 크기에 제한된 작은 수용역(receptive field)으로 인해 지역적 편향(local reductive bias) 문제가 발생한다.
2. **Vision Transformer(ViT) 기반 방식**: 전역 수용역(global receptive field)을 제공하여 시각적 모델링 능력이 뛰어나지만, self-attention 메커니즘의 특성상 입력 크기에 대해 이차 복잡도(quadratic complexity)를 가지므로 효율성이 떨어진다.

따라서 본 논문의 목표는 전역 수용역을 유지하면서도 선형 복잡도(linear complexity)로 동작하는 State Space Model(SSM), 특히 Mamba를 활용하여 효율적이고 강력한 multi-modal semantic segmentation 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **Mamba의 선형 복잡도와 전역 수용역 특성을 multi-modal 융합 구조에 이식**하는 것이다. 주요 기여 사항은 다음과 같다.

1. **최초의 Mamba 적용**: Multi-modal semantic segmentation 분야에 State Space Model(특히 Mamba)을 최초로 성공적으로 적용하였다.
2. **Mamba 기반 융합 메커니즘**: 서로 다른 모달리티 간의 정보 교환을 위한 **Cross Mamba Block (CroMB)**과 통합된 특징에서 핵심 정보를 선택하는 **Concat Mamba Block (ConMB)**을 제안하였다.
3. **Channel-Aware Decoder**: 전역 공간 문맥뿐만 아니라 채널 간의 관계를 효과적으로 모델링하기 위해 채널 어텐션이 통합된 **Channel-Aware Visual State Space Block (CAVSSB)** 기반의 디코더를 설계하였다.

## 📎 Related Works

### 관련 연구 및 한계
- **Multi-Modal Semantic Segmentation**: RGB-T(Thermal) 및 RGB-D(Depth) 분야에서 Encoder-Decoder 구조, Dense connection, Dilated convolution 등이 사용되었다. 최근에는 Transformer 기반의 CMX나 CMNeXt 등이 우수한 성능을 보였으나, 앞서 언급한 이차 복잡도 문제가 효율성을 저해한다.
- **State Space Models (SSM)**: S4와 Mamba는 긴 시퀀스 모델링에서 Transformer를 능가하는 효율성을 보였다. 최근 Vision Mamba(ViM)나 VMamba 등이 이미지 분류 및 의료 영상 분할에 적용되었으나, 대부분은 기존 모듈을 단순 교체하는 수준이며 multi-modal 태스크에 특화된 설계는 부족했다.

### 차별점
Sigma는 단순히 Mamba를 백본으로 사용하는 것에 그치지 않고, 모달리티 간의 상호작용을 유도하는 **Cross-modal matrix 교환 메커니즘**과 **전방-후방 스캔(Forward-Inverse Scan)**을 통한 특징 강화 구조를 설계하여 multi-modal 데이터의 특성을 반영하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
Sigma는 **Siamese Mamba Encoder $\rightarrow$ Fusion Module $\rightarrow$ Channel-Aware Mamba Decoder**의 순차적 구조로 이루어져 있다. 두 개의 인코더 브랜치는 가중치를 공유(weight-sharing)하여 연산 효율을 높이며, 각 레벨에서 추출된 특징은 융합 모듈을 거쳐 디코더로 전달된다.

### 2. Siamese Mamba Encoder
인코더는 **Visual State Space Block (VSSB)**을 계층적으로 쌓아 멀티스케일 특징을 추출한다.
- **VSSB 구조**: Linear projection, Depth-wise Convolution, 그리고 **Selective Scan 2D (SS2D)** 모듈로 구성된다.
- **SS2D**: 이미지를 4가지 방향(좌상$\rightarrow$우하, 우하$\rightarrow$좌상, 우상$\rightarrow$좌하, 좌하$\rightarrow$우상)으로 평탄화하여 시퀀스로 처리함으로써, 2D 이미지의 전역적인 공간 정보를 효율적으로 캡처한다.

### 3. Fusion Module
융합 모듈은 두 단계로 구성된다.

#### (1) Cross Mamba Block (CroMB)
서로 다른 모달리티 간의 정보를 교환하는 단계이다. Mamba의 상태 공간 방정식에서 출력 $y$를 결정하는 행렬 $C$를 서로 교차하여 적용한다.
- **핵심 원리**: RGB 모달리티의 은닉 상태(hidden state) $h_{rgb}$에 Thermal 모달리티의 행렬 $C_x$를 곱하여 출력 $y_{rgb}$를 생성한다.
$$y_t^{rgb} = C_x h_t^{rgb} + D_{rgb} x_t^{rgb}$$
$$y_t^x = C_{rgb} h_t^x + D_x x_t^x$$
이를 통해 한 모달리티의 특징이 다른 모달리티의 가이드에 따라 재구성되어 상호 보완적인 정보가 강화된다.

#### (2) Concat Mamba Block (ConMB)
CroMB를 통해 강화된 특징들을 최종적으로 통합하는 단계이다.
- **절차**: 두 모달리티의 특징을 채널 방향이 아닌 시퀀스 길이 방향으로 연결(concatenate)한다.
- **전방 및 후방 스캔**: 연결된 시퀀스 $S_{Concat}$와 이를 역순으로 배치한 $S_{Inverse}$를 모두 SSM으로 처리한 후, 역순 결과를 다시 뒤집어 합산한다.
- **최종 출력**: 합산된 결과에서 각 모달리티 영역을 분리한 후, 스케일링 파라미터를 곱하고 채널 차원으로 연결하여 최종 융합 특징 $F_{fuse}$를 생성한다.

### 4. Channel-Aware Mamba Decoder
디코더는 **Channel-Aware Visual State Space Block (CAVSSB)**을 사용하여 최종 예측 맵을 생성한다.
- **구조**: VSSB가 전역 공간 문맥을 캡처한다면, 이에 추가적으로 **Channel Attention** (Average Pooling과 Max Pooling 사용)을 결합하여 중요한 채널 정보를 선택적으로 강조한다.
- 이를 통해 공간적 정보와 채널 정보를 동시에 고려하는 robust한 디코딩이 가능하다.

## 📊 Results

### 실험 설정
- **데이터셋**: RGB-T (MFNet, PST900), RGB-D (NYU Depth V2, SUN RGB-D)
- **지표**: mIoU (mean Intersection over Union)
- **비교 대상**: RTFNet, FuseSeg, CMX, CMNeXt 등 CNN 및 Transformer 기반 SOTA 모델

### 주요 결과
1. **정량적 성능 (RGB-T)**:
   - MFNet 데이터셋에서 Sigma-Base는 가장 높은 성능을 보였으며, Sigma-Tiny 모델조차 많은 기존 모델보다 적은 파라미터와 FLOPs로 더 우수한 mIoU를 기록하였다.
   - PST900 데이터셋에서도 타 모델 대비 2% 이상의 성능 향상을 달성하였다.
2. **정량적 성능 (RGB-D)**:
   - NYU Depth V2에서 Sigma-Small 모델은 CMNeXt보다 약 49.8M 적은 파라미터를 사용하면서도 더 높은 성능을 보여, 효율성과 정확도의 최적의 균형을 증명하였다.
3. **정성적 분석**:
   - 저조도나 그림자가 심한 환경에서 RGB 단독 모델이 놓치는 객체(예: 소파 옆의 둥근 의자)를 깊이/열화상 정보를 통해 정확하게 분할하는 것을 확인하였다.
   - RGB의 색상 구분 능력과 Thermal의 질감 차별화 능력이 결합되어 경계선 묘사가 정교해졌다.
4. **복잡도 분석**:
   - Transformer의 self-attention 기반 융합(ConSA)은 입력 시퀀스 길이에 따라 연산량이 기하급수적으로 증가하지만, Sigma의 ConMB는 선형적으로 증가하여 압도적인 효율성을 보였다.

## 🧠 Insights & Discussion

### 강점
- **효율적인 전역 모델링**: Mamba를 통해 ViT 수준의 전역 수용역을 확보하면서도 연산 비용을 획기적으로 낮추었다. 특히 고해상도 이미지 처리 시 Transformer 대비 압도적인 이점을 가진다.
- **특화된 융합 구조**: 단순한 연결이나 합산이 아니라, SSM의 시스템 행렬($C$)을 교환하는 방식(CroMB)과 양방향 스캔(ConMB)을 도입하여 multi-modal 데이터의 특성을 잘 활용하였다.

### 한계 및 향후 과제
- **모달리티 확장성**: 현재는 두 개의 모달리티(RGB-X)에 집중하고 있으나, Mamba의 긴 시퀀스 처리 능력을 활용하면 LiDAR 등 더 많은 모달리티를 통합하는 연구로 확장 가능하다.
- **메모리 사용량**: 이미지의 전역 정보를 얻기 위해 4방향 스캔을 수행하는데, 이로 인해 메모리 사용량이 4배로 증가한다. 이는 엣지 디바이스 배포 시 제약 사항이 될 수 있으며, 향후 1D SSM이나 positional encoder 도입을 통해 최적화할 필요가 있다.

## 📌 TL;DR

본 논문은 **Mamba(Selective SSM)를 multi-modal semantic segmentation에 최초로 적용한 Sigma 네트워크**를 제안한다. Siamese 인코더와 전역 수용역을 갖는 Mamba 기반의 융합 모듈(CroMB, ConMB), 그리고 채널 어텐션이 결합된 디코더를 통해 **선형 복잡도로 높은 정확도**를 달성하였다. 실험 결과, RGB-T 및 RGB-D 데이터셋 모두에서 기존 Transformer 기반 SOTA 모델보다 **더 적은 연산량으로 더 높은 성능**을 기록하여, 향후 효율적인 multi-modal 인지 시스템 구축에 중요한 가능성을 제시하였다.