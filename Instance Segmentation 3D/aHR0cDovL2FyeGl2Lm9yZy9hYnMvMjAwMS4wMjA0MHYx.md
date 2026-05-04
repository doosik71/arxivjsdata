# Robust Semantic Segmentation of Brain Tumor Regions from 3D MRIs

Andriy Myronenko, Ali Hatamizadeh (2020)

## 🧩 Problem to Solve

본 논문은 3D MRI 영상을 이용한 뇌종양 영역의 자동 분할(Semantic Segmentation) 문제를 다룬다. 뇌종양 분할은 질병의 진단 및 치료 계획 수립에 필수적인 기초 비전 작업이다. 특히, 다양한 MRI 모달리티(T1, T1c, T2, FLAIR)를 활용하여 종양의 서로 다른 특성과 확산 영역을 정밀하게 구분해야 하는 복잡성이 존재한다.

논문의 구체적인 목표는 BraTS 2019(Multimodal Brain Tumor Segmentation Challenge) 데이터셋을 활용하여, 뇌종양의 세 가지 중첩된 하위 영역인 Whole Tumor (WT), Tumor Core (TC), 그리고 Enhancing Tumor (ET)를 정확하게 분할하는 강건한 딥러닝 모델을 구축하고 최적의 방법론(Best Practices)을 탐색하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 검증된 3D Encoder-Decoder 아키텍처를 기반으로, 성능을 극대화하기 위한 최적의 구성 요소(Architecture design choices)와 상호 보완적인 손실 함수(Complementary loss functions)의 조합을 탐색하는 것이다. 특히 단순한 모델 구조에 그치지 않고, 정규화 방법, 학습 전략, 그리고 세 가지 서로 다른 특성의 손실 함수를 결합하여 분할 정확도를 높이고 모델의 강건성을 확보하고자 하였다.

## 📎 Related Works

이전 연구인 BraTS 2018의 상위 모델들은 주로 딥러닝 기반의 접근 방식을 취했다. 
- **Myronenko [16]**: 보조 작업(Secondary task)을 위한 별도의 디코더를 추가하여 네트워크에 추가적인 구조적 제약을 가하는 방식을 탐구하였다.
- **Isensee et al. [11]**: 일반적인 U-Net 아키텍처에 소수의 수정만 가해도 경쟁력 있는 성능을 낼 수 있음을 보였다.
- **McKinly et al. [13]**: U-Net 구조 내부에 Dilated Convolution이 적용된 DenseNet 구조를 삽입한 모델을 제안하였다.
- **Zhou et al. [19]**: 멀티 스케일 컨텍스트 정보를 활용하고, 공유된 백본을 통해 세 가지 종양 영역을 계층적으로 분할하며 Attention 블록을 추가한 앙상블 모델을 제안하였다.

본 논문은 저자의 이전 연구([16])를 계승하지만, 보조 작업 디코더 대신 다양한 아키텍처 설계 선택지와 보완적인 손실 함수 조합을 탐구한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
본 모델은 3D MRI 영상을 입력으로 받아 세 개의 채널(WT, TC, ET)을 출력하는 Encoder-Decoder 기반의 CNN 구조를 가진다. 입력 데이터는 T1, T1c, T2, FLAIR의 4개 모달리티가 결합된 4채널 3D 영상이다.

### 주요 구성 요소
1. **Encoder**: 
   - ResNet 블록을 기본 단위로 사용하며, 각 블록은 두 개의 Convolution, Normalization, ReLU 및 Additive Identity Skip Connection으로 구성된다.
   - Strided Convolution을 통해 공간 해상도를 단계적으로 2배씩 줄이고 특징 맵(Feature map)의 크기를 2배씩 늘린다.
   - 최종 Encoder Endpoint의 크기는 입력 영상보다 8배 작게 설정하여 공간적 정보를 최대한 보존하였다.

2. **Decoder**:
   - Encoder와 대칭적인 구조를 가지며, 각 레벨마다 하나의 블록이 배치된다.
   - $1 \times 1 \times 1$ Convolution으로 채널 수를 줄인 후, 3D Bilinear Upsampling을 통해 공간 해상도를 2배로 확장한다.
   - 이후 Encoder의 동일 레벨 출력값을 더하는 Skip Connection을 적용하고 ResNet 블록을 통과시킨다.
   - 최종 출력단은 $1 \times 1 \times 1$ Convolution과 Sigmoid 함수를 거쳐 3개의 채널(종양 하위 영역)을 생성한다.

### 손실 함수 (Loss Functions)
모델은 세 가지 손실 함수의 합으로 구성된 Hybrid Loss를 사용한다:
$$L = L_{dice} + L_{focal} + L_{acl}$$

- **Soft Dice Loss ($L_{dice}$)**: 예측 마스크 $p_{pred}$와 정답 마스크 $p_{true}$ 간의 중첩도를 측정한다.
$$L_{dice} = 1 - \frac{2 * \sum p_{true} * p_{pred}}{\sum p_{true}^2 + \sum p_{pred}^2 + \epsilon}$$
- **Focal Loss ($L_{focal}$)**: 클래스 불균형 문제를 해결하기 위해 어려운 샘플에 더 큰 가중치를 부여한다. ($\gamma=2$ 적용)
$$L_{focal} = -\frac{1}{N} \sum (1 - p_{pred})^\gamma p_{true} \log(p_{pred} + \epsilon)$$
- **Supervised Active Contour Loss ($L_{acl}$)**: 3D로 확장된 능동 윤곽선 손실로, 체적 항(Volumetric term)과 길이 항(Length term)으로 구성된다.
$$L_{acl} = L_{vol} + L_{length}$$
  - $L_{vol}$: 전경(foreground)과 배경(background)의 에너지를 이용하여 영역 내부와 외부의 차이를 최소화한다.
  - $L_{length}$: 예측 마스크의 경계면 길이를 최소화하여 부드러운 분할 결과를 유도한다.
$$L_{length} = \sum \sqrt{|\nabla p_{pred,x}|^2 + |\nabla p_{pred,y}|^2 + |\nabla p_{pred,z}|^2} + \epsilon$$

### 학습 및 최적화 절차
- **Optimizer**: Adam을 사용하며, 초기 학습률 $\alpha_0 = 1e-4$에서 시작하여 다음과 같은 스케줄러를 통해 점진적으로 감소시킨다.
$$\alpha = \alpha_0 * (1 - e^{N_e})^{0.9}$$ (여기서 $N_e$는 총 에폭 수인 300을 의미한다)
- **정규화 및 증강**: L2 norm 정규화($1e-5$)와 초기 Encoder Convolution 이후 Spatial Dropout(0.2)을 적용하였다. 입력 영상은 zero mean, unit std로 정규화하며, 랜덤 강도 시프트/스케일링 및 축 방향 미러 플립(probability 0.5)을 적용하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: BraTS 2019 (훈련 335케이스, 검증 125케이스).
- **입력 크기**: $240 \times 240 \times 155$ 영상에서 $160 \times 192 \times 128$ 크기로 랜덤 크롭하여 사용.
- **평가 지표**: Dice score 및 Hausdorff distance.
- **하드웨어**: NVIDIA Tesla V100 32GB GPU. 8장의 GPU를 사용하는 DGX-1 서버에서 데이터 병렬화(Data Parallelism)를 통해 학습 시간을 8시간으로 단축하였다.

### 정량적 결과
검증 데이터셋(Validation)과 테스트 데이터셋(Testing)에 대한 결과는 다음과 같다.

- **Validation (Single Model, Batch 8)**:
  - Dice: ET(0.800), WT(0.894), TC(0.834)
  - Hausdorff (mm): ET(3.92), WT(15.89), TC(6.562)

- **Testing (Ensemble)**:
  - Dice: ET(0.826), WT(0.882), TC(0.837)
  - Hausdorff (mm): ET(2.203), WT(4.713), TC(3.968)

### 효율성
- 단일 GPU 기준 에폭당 10분, 총 300 에폭 학습에 2일이 소요된다.
- 추론 시간(Inference time)은 단일 V100 GPU에서 샘플당 0.4초이다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **정규화 함수의 영향**: Group Normalization(GN)과 Instance Normalization(IN)은 유사한 성능을 보였으나, Batch Normalization(BN)은 성능이 현저히 낮았다. 이는 3D 영상의 특성상 배치 크기(최대 16)가 너무 작아 BN이 제대로 작동하지 않았기 때문으로 분석된다.
- **네트워크 설계**: 네트워크의 깊이(Depth)를 더 깊게 하는 것보다 너비(Width, 필터 수)를 늘리는 것이 결과 향상에 일관적인 효과가 있었다.
- **데이터 증강**: 랜덤 히스토그램 매칭, 어파인 변환, 회전, 랜덤 필터링 등의 정교한 증강 기법을 시도했으나, 유의미한 성능 향상은 관찰되지 않았다.

### 한계 및 비판적 해석
본 논문은 특정 챌린지의 성능 향상을 위한 'Best Practice' 탐색에 집중하고 있어, 제안된 하이브리드 손실 함수 각각이 정확히 어떤 기여를 했는지에 대한 개별 절제 실험(Ablation Study)이 부족하다. 또한, 앙상블 결과는 제시되어 있으나 앙상블의 구체적인 구성(어떤 모델들을 조합했는지)에 대한 설명이 명시되지 않았다.

## 📌 TL;DR

본 논문은 3D MRI 뇌종양 분할을 위해 ResNet 기반의 Encoder-Decoder 구조에 Soft Dice, Focal, 그리고 3D Active Contour Loss를 결합한 하이브리드 손실 함수를 적용한 방법론을 제안한다. 실험을 통해 3D 의료 영상 분할에서는 Batch Normalization보다 Group/Instance Normalization이 효과적이며, 모델의 깊이보다 너비를 확장하는 것이 성능 향상에 유리함을 입증하였다. 이 연구는 BraTS 챌린지와 같은 고정밀 의료 영상 분할 작업에서 안정적인 성능을 내기 위한 실무적인 가이드라인을 제공한다.