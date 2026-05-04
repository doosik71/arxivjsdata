# Hybrid Swin Deformable Attention U-Net for Medical Image Segmentation

Lichao Wang, Jiahao Huang, Xiaodan Xing, Guang Yang (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 임상 진단에서 매우 중요한 작업이지만, 최근의 딥러닝 모델들은 다음과 같은 두 가지 핵심적인 문제에 직면해 있다.

첫째, 모델의 해석 가능성(Interpretability) 부족이다. 많은 하이브리드 모델들이 높은 성능을 보이지만, 모델이 어떤 근거로 특정 영역을 분할했는지에 대한 설명이 부족하여 실제 임상 환경에서 의료진이 모델의 결정을 신뢰하고 적용하는 데 한계가 있다.

둘째, 기존의 Multi-Head Self-Attention (MSA) 메커니즘의 경직성이다. 일반적인 MSA나 Swin Transformer의 Shifted Window MSA(SMSA)는 정해진 윈도우 내의 모든 패치에 대해 어텐션을 계산한다. 그러나 의료 영상 속의 장기나 병변(예: 좌심실 벽)은 형태가 매우 불규칙하고 변형이 심하기 때문에, 고정된 윈도우 방식은 불필요한 영역까지 계산하는 연산 낭비(Redundancy)를 초래하고 이는 해석의 모호함으로 이어진다.

따라서 본 논문의 목표는 정밀한 분할 성능을 유지하면서도, 모델이 실제로 어디에 집중하고 있는지를 시각적으로 명확하게 보여줄 수 있는 해석 가능한 하이브리드 U-Net 구조를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Swin Deformable MSA (SDMSA)** 모듈과 **Parallel Convolution** 브랜치를 결합한 하이브리드 블록을 설계하여 U-Net 구조에 적용하는 것이다.

1. **SDAH-UNet 제안**: Swin Deformable Attention을 도입한 하이브리드 U-Net 아키텍처를 제안하여 해부학적 구조 및 병변 분할 작업에서 최첨단(SOTA) 성능을 달성하였다.
2. **SDAPC 블록 설계**: SDMSA와 Parallel Convolution이 병렬로 구성된 SDAPC(Swin Deformable MSA with Parallel Convolution) 블록을 통해, 전역적인 형태 구조(Holistic shape)와 세부적인 질감 특징(Detailed texture)을 동시에 포착하도록 하였다.
3. **시각적 해석 가능성 제공**: Deformable Attention의 샘플링 포인트(Deformation points)를 통해 모델이 타겟 영역에 어떻게 집중하는지를 직접적으로 시각화함으로써, 기존의 Grad-CAM이나 일반 어텐션 맵보다 더 정밀한 설명력을 제공하였다.

## 📎 Related Works

최근 의료 영상 분할 분야에서는 CNN의 국부적 특징 추출 능력과 Transformer의 전역적 문맥 파악 능력을 결합한 하이브리드 모델 연구가 활발하다. 

- **기존 접근 방식의 한계**: Grad-CAM이나 Layer-Wise Relevance Propagation과 같은 사후 해석 방법(Post-hoc interpretation)은 객체의 경계를 정밀하게 캡처하지 못하며, 모델 내부의 의사결정 과정을 내재적으로 보여주지 못한다는 단점이 있다. 
- **Attention 메커니즘의 도입**: MSA 기반 모델들은 어텐션 스코어 맵을 통해 어느 정도의 해석 가능성을 제공하지만, 앞서 언급한 바와 같이 고정된 수용역(Receptive field)으로 인해 불필요한 연산이 발생하고 정밀한 타겟팅이 어렵다.
- **차별점**: 본 논문은 Deformable Attention을 Swin Transformer의 윈도우 구조에 통합하여, 데이터의 특성에 따라 샘플링 위치를 유연하게 조정함으로써 연산 효율성을 높이고 해석의 정밀도를 극대화하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (SDAH-UNet)
SDAH-UNet은 대칭적인 U-자형 구조를 가진 엔드-투-엔드 모델이다.
- **Encoder**: 3개의 SDAPC 블록으로 구성되며, 시작 부분에 Convolutional Embedding 모듈이 위치한다.
- **Decoder**: 3개의 SDAPC 블록으로 구성되며, 마지막 부분에 DeConvolutional Expanding 모듈이 위치한다.
- **Bottleneck**: 인코더와 디코더 사이에 1개의 SDAPC 블록이 추가로 배치되어 저해상도 특징을 처리한다.

### 2. 세부 구성 요소
- **Convolutional Embedding & Expanding**: 인코더의 시작에서는 4개의 연속적인 소형 커널 합성곱 층, GELU 활성화 함수, Layer Normalization을 사용하여 픽셀 수준의 공간 정보를 인코딩한다. 디코더의 끝에서는 Deconvolution 층을 통해 해상도를 복원한다.
- **SDAPC Block**: 두 개의 병렬 경로로 구성된다.
    - **첫 번째 경로**: $\text{DwConv} \rightarrow \text{LN} \rightarrow \text{FC}_1 \rightarrow \text{GELU} \rightarrow \text{FC}_2$ 순으로 연산하며 잔차 연결(Residual connection)을 더한다.
    - **두 번째 경로**: SDMSA 층과 DwConv 층이 병렬로 배치되어 있으며, 두 출력의 결합(Concatenation) 후 최종 FC 층을 통과하고 다시 잔차 연결을 더한다.

### 3. Swin Deformable Multi-Head Self-Attention (SDMSA)
SDMSA는 고정된 윈도우 내에서 샘플링 지점을 유동적으로 변경하여 타겟에 집중한다.

- **작동 원리**: 입력 특징 맵 $F$를 윈도우 단위로 나누고 여러 개의 헤드로 분리한다. 각 윈도우의 쿼리 $q_i^j$를 기반으로 $\text{CNN}_{offset}$을 통해 오프셋 $\Delta p_i^j$를 학습한다.
- **샘플링**: 균일하게 분포된 기준점 $p_i^j$에 학습된 오프셋을 더해 변형된 샘플링 지점 $p_i^j + \Delta p_i^j$를 생성하고, Bilinear Interpolation $\phi(\cdot)$을 통해 특징을 추출한다.
- **수식 설명**:
    - 쿼리와 오프셋 생성:
      $$q_i^j = F_i^j W_{Q_i^j}, \quad \Delta p_i^j = \text{CNN}_{offset}(q_i^j)$$
    - 변형된 특징 추출:
      $$\hat{F}_i^j = \phi(F_i^j; p_i^j + \Delta p_i^j)$$
    - 키(Key)와 밸류(Value) 계산 및 최종 어텐션 결과 $Z_i^j$ 도출:
      $$k_i^j = \hat{F}_i^j W_{K_i^j}, \quad v_i^j = \hat{F}_i^j W_{V_i^j}$$
      $$Z_i^j = \text{SoftMax}(q_i^j {k_i^j}^\top / \sqrt{d} + b_i^j) v_i^j$$
      (여기서 $b_i^j$는 고정된 위치 편향(Positional bias)이다.)

## 📊 Results

### 1. 실험 설정
- **데이터셋**: ACDC (심장 해부학적 분할, 단일 MRI) 및 BraTS2020 (뇌종양 병변 분할, 다중 모달리티 MRI)
- **지표**: Dice Similarity Coefficient (DSC), Hausdorff95 (HD95)
- **손실 함수**: $\text{Dice Loss} (L_{dice}) + \text{Cross-Entropy Loss} (L_{CE})$
- **비교 모델**: UNet, nnUNet, SwinUNet, TransUNet, UNet-2022

### 2. 정량적 결과
- **ACDC 데이터셋**: SDAH-UNet은 평균 DSC에서 가장 높은 성능을 보였으며, 특히 Myocardium(MYO)과 Left Ventricle(LV) 분할에서 매우 강한 성능을 나타냈다.
- **BraTS2020 데이터셋**: Whole Tumor(WT), Tumor Core(TC), Enhancing Tumor(ET) 모든 영역에서 기존 SOTA 모델들보다 높은 DSC를 기록하였다.
- **복잡도**: 파라미터 수는 $\text{CNN}_{offset}$과 병렬 구조로 인해 약간 증가했으나, FLOPs(연산량)는 다른 Swin 기반 모델들과 비슷하거나 오히려 낮은 수준을 유지하여 효율성을 입증하였다.

### 3. 해석 가능성 및 절제 연구(Ablation Study)
- **시각적 분석**: Deformation points를 시각화한 결과, 모델이 단순한 영역 표시를 넘어 실제 종양의 경계와 고밀도 정보 영역에 정밀하게 집중하고 있음을 확인하였다. 이는 Grad-CAM보다 훨씬 명확한 근거를 제공한다.
- **절제 연구**:
    - SDAPC 블록의 개수를 늘릴수록 성능이 향상되었으며, 특히 첫 번째 블록을 SDAPC로 교체했을 때 DSC가 약 1.2% 상승하였다.
    - SDMSA 브랜치와 Conv 브랜치 중 하나만 사용할 때보다 두 브랜치를 모두 사용하는 Dual-branch 구조에서 성능이 약 2% 더 높게 나타나, 두 메커니즘의 상호보완적 역할이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 흔히 발생하는 '블랙박스' 문제를 SDMSA라는 구조적 장치를 통해 해결하려 했다는 점에서 큰 강점을 가진다. 

특히, **Deformation points**라는 개념을 단순한 성능 향상 도구가 아니라, 모델이 "어디를 보고 있는가"와 "어떻게 그곳에 집중하게 되었는가"를 설명하는 시각적 증거로 활용한 점이 매우 독창적이다. 이는 의료 현장에서 AI의 판단 근거를 요구하는 임상적 요구사항을 기술적으로 잘 풀어낸 사례라고 평가할 수 있다.

다만, 본 논문에서는 모델의 파라미터 수 증가가 미미하다고 언급하였으나, 실제 추론 시 Bilinear Interpolation 과정이 추가됨에 따라 발생하는 실제 연산 지연 시간(Latency)에 대한 상세한 분석은 부족하다. 또한, 다양한 해상도의 입력 이미지에 대한 강건성 테스트가 추가된다면 더 완벽한 검증이 될 것으로 보인다.

## 📌 TL;DR

SDAH-UNet은 Swin Transformer의 윈도우 구조에 **Deformable Attention**과 **병렬 합성곱**을 결합하여, 의료 영상의 불규칙한 형태를 정밀하게 분할하고 그 과정을 시각적으로 설명할 수 있는 모델이다. ACDC와 BraTS2020 데이터셋에서 SOTA 성능을 달성했으며, 특히 샘플링 포인트의 변형을 통해 모델의 집중 영역을 명확히 보여줌으로써 의료 AI의 해석 가능성 문제를 효과적으로 개선하였다. 이 연구는 향후 높은 신뢰성이 요구되는 정밀 의료 진단 시스템의 핵심 구조로 활용될 가능성이 크다.