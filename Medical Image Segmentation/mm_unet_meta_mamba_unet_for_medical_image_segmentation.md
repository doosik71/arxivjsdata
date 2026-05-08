# MM-UNet: Meta Mamba UNet for Medical Image Segmentation

Bin Xie, Yan Yan, and Gady Agam (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에 State Space Models(SSMs), 특히 Mamba를 적용할 때 발생하는 두 가지 핵심적인 한계점을 해결하고자 한다.

첫째는 **공간적 불연속성(Spatial Discontinuities)** 문제이다. SSM은 본래 1차원 시퀀스 데이터를 처리하도록 설계되었기 때문에, 3차원 의료 영상을 1차원으로 펼치는(flattening) 과정에서 인접한 행이나 층 사이의 관계가 단절되는 현상이 발생한다. 이는 모델이 공간적 구조를 정확하게 추론하는 것을 방해한다.

둘째는 **고분산 데이터(High-variance Data) 적합 문제**이다. 의료 영상은 장기와 조직에 따라 픽셀 강도(intensity)의 변화가 매우 크며, 분석 결과 SSM은 평균에서 크게 벗어난 고분산 데이터를 적합시키는 데 어려움이 있음이 확인되었다. 이는 정밀한 분할 성능을 저하시키는 원인이 된다.

따라서 본 연구의 목표는 이러한 SSM의 내재적 한계를 극복할 수 있는 통합 U-자형 구조인 MM-UNet을 제안하여, 연산 효율성을 유지하면서도 높은 분할 정확도를 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SSM의 취약점을 보완하기 위해 **하이브리드 모듈 설계**와 **양방향 스캔 전략**을 도입하는 것이다.

1. **MM-UNet 아키텍처 제안**: 기존의 Mamba 기반 분할 모델들을 통합적으로 표현할 수 있는 유연한 U-자형 인코더-디코더 구조를 설계하였다.
2. **하이브리드 모듈(Hybrid Module) 설계**: SSM이 고분산 데이터에 취약하다는 점에 착안하여, 두 개의 CNN 레이어를 배치한 후 그 뒤에 SSM을 위치시키고 이를 전체적으로 Residual Connection 내부에 배치하였다. 이는 Residual Connection 내부의 피처 맵이 더 낮은 분산을 가지므로 SSM의 학습 안정성과 성능을 높인다는 직관에 기반한다.
3. **양방향 스캔 순서(Bi-directional Scan Order) 전략**: 단방향 스캔 시 발생하는 불연속성 문제를 해결하기 위해, 정방향(DHW)과 역방향(flip(DHW)) 스캔을 동시에 사용하는 전략을 도입하여 공간적 일관성을 확보하였다.

## 📎 Related Works

의료 영상 분할 분야에서는 전통적으로 CNN 기반의 U-Net과 그 변형 모델들이 주류를 이루었으며, 최근에는 글로벌 문맥(global context)을 잘 포착하는 Vision Transformers(ViTs)가 도입되었다. 하지만 CNN은 수용 영역(receptive field)의 국소성으로 인해 장거리 의존성(long-range dependencies) 포착에 한계가 있고, ViTs는 어텐션 메커니즘의 이차 시간 복잡도($O(N^2)$)로 인해 계산 비용이 매우 높다는 단점이 있다.

최근 등장한 SSM 및 Mamba는 선형 시간 복잡도로 장거리 시퀀스를 모델링할 수 있어 대안으로 주목받고 있으며, 이를 의료 영상에 적용한 U-Mamba, VM-UNet, SegMamba 등의 연구가 진행되었다. 그러나 기존 연구들은 Mamba를 단순히 기존 아키텍처에 통합하는 데 집중했을 뿐, 의료 영상의 3차원 특성과 데이터 분산 특성에 따른 SSM의 내재적 한계를 심층적으로 분석하고 해결하려는 시도는 부족했다.

## 🛠️ Methodology

### 1. State Space Sequence Models (SSMs) 기반

SSM은 1차원 입력 신호 $x(t)$를 잠재 상태 $h(t)$를 통해 출력 신호 $y(t)$로 매핑하며, 기본 방정식은 다음과 같다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A, B, C$는 학습 가능한 파라미터이다. 이 연속 시스템은 Zero-Order Hold(ZOH) 방법을 통해 이산화되어 다음과 같이 표현된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

이산화된 상태에서 시스템은 다음과 같은 반복적인 형태로 계산된다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = C h_t$$

Mamba는 여기에 입력 의존적 선택 메커니즘(selection mechanism)을 추가하여 정보 필터링 능력을 강화하고, 하드웨어 최적화 알고리즘을 통해 선형 시간 복잡도로 계산을 수행한다.

### 2. MM-UNet 아키텍처

MM-UNet은 대칭적인 U-자형 구조를 가지며, Stem 모듈, 인코더(EncMetaBlock), 보틀넥(MetaBlock), 디코더(DecMetaBlock)로 구성된다.

- **Meta-Block의 유연성**: 각 블록은 CNN 기반, 순수 Mamba 기반, 또는 하이브리드 기반으로 교체 가능하도록 설계되었다.
- **하이브리드 모듈 구조**: 실험 결과, 두 개의 연속적인 CNN 레이어 뒤에 Mamba 모듈을 배치하고, 이를 Residual Connection 내부에 포함시킨 구조가 가장 우수한 성능을 보였다. 이는 CNN이 일종의 전처리기 역할을 하여 SSM에 입력되는 데이터의 분산을 낮춰주기 때문이다.
- **배치 전략**: Mamba 모듈을 인코더와 보틀넥에 배치하는 것이 디코더에 배치하는 것보다 성능 향상에 더 효과적임이 밝혀졌다.

### 3. 스캔 전략 (Scan Design)

3차원 데이터를 1차원으로 펼칠 때 발생하는 불연속성을 줄이기 위해 **1D BiScan** 전략을 사용한다.

- **DHW 스캔**: Depth $\to$ Height $\to$ Width 순으로 스캔한다.
- **flip(DHW) 스캔**: DHW의 역순으로 스캔하여, 정방향 스캔에서 발생한 불연속 지점을 보완한다.
- **추론 시 최적화**: 추론 단계에서는 축(axial), 관상(coronal), 시상(sagittal) 방향으로 8번의 플립(flip)을 수행하여 예측값을 평균 내고, 중앙 영역의 가중치를 높이기 위해 3D 가우시안 필터를 곱하는 후처리를 적용한다.

### 4. 학습 절차 및 손실 함수

- **프레임워크**: nnUNet 프레임워크를 기반으로 하며, SGD 옵티마이저와 다항식 감쇠(polynomial decay) 학습률 전략을 사용한다.
- **손실 함수**: Cross-Entropy Loss와 Dice Loss를 결합하여 사용한다.
- **Deep Supervision**: 디코더의 마지막 3단계에서 보조 손실(auxiliary losses)을 적용하며, 해상도가 낮아질수록 가중치를 절반씩 줄여 합산한다.

$$L = w_1 \cdot L_1 + w_2 \cdot L_2 + w_3 \cdot L_3$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: AMOS2022 (복부 CT 장기 분할, 16개 구조) 및 Synapse (복부 CT, 13개 장기) 데이터셋을 사용하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC)를 사용하였다.
- **비교 대상**: nnUNet, 3D UX-Net (CNN 기반), UNETR, SwinUNETR, nnFormer (Transformer 기반), VMUNet, UMamba, SwinUMamba (Mamba 기반) 모델들과 비교하였다.

### 2. 정량적 결과

- **AMOS2022**: MM-UNet은 **91.0%의 Dice score**를 기록하여, nnUNet 대비 3.2%, 3D UX-Net 대비 1.0% 높은 성능을 보였다. 특히 모든 Mamba 기반 모델들보다 우수한 성적을 거두었다.
- **Synapse**: MM-UNet은 **87.1%의 Dice score**를 달성하여 SOTA 성능을 기록하였다. 특히 간(liver)과 비장(spleen)과 같은 대형 장기 분할에서 SSM의 장거리 의존성 포착 능력이 빛을 발하며 nnFormer(1.5%↑)와 nnUNet(6.9%↑)을 크게 앞섰다.

### 3. 정성적 및 분석적 결과

- **Attention Map 분석**: $\text{QK}^T$ 값을 시각화한 결과, SSM이 매우 적은 파라미터만으로도 1차원으로 펼쳐진 데이터에서 이미지의 패턴을 효과적으로 포착함을 확인하였다.
- **스캔 전략 비교**: 단순 DHW 스캔보다 양방향 스캔(BiScan)이 성능을 향상시켰으며, 3개 이상의 스캔 쌍을 추가하는 것은 중복 정보로 인해 이득이 없거나 오히려 성능을 저하시켰다.

## 🧠 Insights & Discussion

본 논문은 SSM을 비전 작업, 특히 3차원 의료 영상 분할에 적용할 때 단순히 모델을 쌓는 것이 아니라, 데이터의 특성(고분산)과 구조적 특성(공간적 불연속성)을 먼저 분석하고 이를 아키텍처 설계에 반영했다는 점에서 강점이 있다.

특히 **"Residual Connection 내부의 피처 맵이 낮은 분산을 가진다"**는 관찰을 통해 Mamba 모듈의 최적 위치를 찾아낸 점과, 복잡한 스캔 방식보다 단순한 **양방향 스캔(Bi-directional scan)**이 가장 효율적임을 실험적으로 증명한 점이 인상적이다.

다만, 논문에서 제시된 하이브리드 구조가 구체적으로 어떤 빈도의 분산 감소를 유도하는지에 대한 정밀한 통계적 분석보다는 시각화된 분포도(Fig. 3)에 의존하고 있다는 점이 한계로 보일 수 있다. 또한, MRI나 초음파와 같이 CT와는 다른 특성을 가진 다른 모달리티에서도 동일한 분산 및 불연속성 문제가 발생하는지에 대해서는 향후 연구 과제로 남겨두었다.

## 📌 TL;DR

MM-UNet은 SSM(Mamba)의 고분산 데이터 적합 능력 부족과 3D 데이터의 1D 평탄화 시 발생하는 불연속성 문제를 해결한 의료 영상 분할 모델이다. 이를 위해 **CNN-Mamba 하이브리드 모듈을 Residual Connection 내부에 배치**하고 **양방향 스캔 전략**을 도입하였다. 결과적으로 AMOS2022(91.0%)와 Synapse(87.1%) 데이터셋에서 SOTA 성능을 달성하였으며, 이는 SSM 기반 모델의 최적 설계 방향을 제시한 연구로 평가된다.
