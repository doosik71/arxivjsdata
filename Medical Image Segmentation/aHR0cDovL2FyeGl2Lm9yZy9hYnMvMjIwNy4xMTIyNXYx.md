# Large-Kernel Attention for 3D Medical Image Segmentation

Hao Li, Yang Nan, Javier Del Ser and Guang Yang (2022)

## 🧩 Problem to Solve

본 논문은 MRI 및 CT와 같은 3D 의료 영상에서 여러 장기와 종양을 자동으로 분할(segmentation)하는 문제를 해결하고자 한다. 3D 의료 영상 분할은 암의 진단과 치료 계획 수립에 필수적이지만, 다음과 같은 기술적 난제들이 존재한다.

첫째, 장기들이 서로 겹쳐 있거나 복잡하게 연결되어 있으며, 해부학적 변이가 심하고 대비(contrast)가 낮아 경계 구분이 어렵다. 둘째, 종양의 경우 모양, 위치, 외형이 매우 다양하며, 전체 영상에서 종양이 차지하는 부피가 작아 배경 복셀(voxel)의 영향력이 지배적이기 때문에 정확한 분할이 어렵다.

기존의 Convolutional Neural Networks(CNN) 기반 방법들은 국소적인 특징 추출에는 능숙하지만, 장거리 의존성(long-range dependence)을 학습하는 데 한계가 있다. 반면, Self-attention 메커니즘은 전역적인 정보를 포착할 수 있으나, 3D 데이터에 적용할 경우 연산 복잡도가 입력 크기의 제곱에 비례하여 급격히 증가하며, 의료 영상의 구조적 세부 사항이나 채널 적응성(channel adaptation)을 간과하는 경향이 있다. 따라서 본 논문의 목표는 연산 효율성을 유지하면서도 국소적 문맥과 전역적 의존성을 동시에 포착할 수 있는 새로운 어텐션 모듈을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **분해된 대형 커널(Decomposed Large-Kernel, LK) 어텐션 모듈**의 제안이다. 이 모듈의 중심 아이디어는 CNN의 국소적 문맥 정보 추출 능력과 Self-attention의 장거리 의존성 학습 능력을 결합하는 것이다.

특히, 매우 큰 커널을 사용하는 컨볼루션을 그대로 적용할 경우 발생하는 막대한 연산 비용 문제를 해결하기 위해, 하나의 큰 커널 컨볼루션을 세 개의 작은 연산(Depth-wise, Depth-wise Dilated, $1 \times 1 \times 1$ Convolution)으로 분해하여 파라미터 수와 연산량을 획기적으로 줄였다. 이를 통해 3D 의료 영상의 복잡한 구조를 효율적으로 학습하고, 중요한 특징은 증폭시키고 노이즈는 억제하는 적응적 특징 선택이 가능하게 하였다.

## 📎 Related Works

논문에서는 다장기 분할(Multi-organ Segmentation)과 종양 분할(Tumor Segmentation) 분야의 기존 연구들을 검토한다.

1. **다장기 분할**: 3D FCN 및 U-Net 계열의 네트워크들이 주로 사용되었으며, 최근에는 coarse-to-fine 방식이나 반지도 학습(semi-supervised learning) 등을 통해 정확도를 높이려는 시도가 있었다. 그러나 컨볼루션 층의 국소성으로 인해 장거리 공간 관계를 학습하는 데 어려움이 있었다.
2. **종양 분할**: BraTS 챌린지를 중심으로 비대칭 U-Net이나 nnU-Net과 같은 프레임워크가 제안되었다. 일부 연구에서는 Transformer 기반의 UNETR 등을 도입하여 전역 정보를 수집하려 했으나, 앞서 언급한 연산 비용과 3D 구조 정보 손실이라는 한계가 존재했다.

본 연구는 이러한 기존 접근 방식과 달리, Self-attention의 전역적 시야를 확보하면서도 CNN의 효율성과 구조적 특성을 유지하는 LK 어텐션 모듈을 통해 차별점을 둔다.

## 🛠️ Methodology

### 1. LK Attention Module

LK 어텐션 모듈은 입력 특징 맵에서 중요한 영역을 식별하여 어텐션 맵을 생성하고, 이를 원래의 특징 맵에 적용하는 구조이다.

#### 컨볼루션 분해 (Convolutional Decomposition)
$K \times K \times K$ 크기의 대형 커널 컨볼루션을 직접 수행하는 대신, 다음과 같이 세 단계로 분해하여 연산 효율을 높인다.
- $(2d-1) \times (2d-1) \times (2d-1)$ 크기의 **Depth-wise (DW) Convolution**
- $\frac{K}{d} \times \frac{K}{d} \times \frac{K}{d}$ 크기의 **Depth-wise Dilated (DWD) Convolution** (dilation rate $d$ 적용)
- $1 \times 1 \times 1$ **Convolution**

이 분해 방식의 파라미터 수($N_{PRM}$)와 연산량(FLOPs)은 다음과 같이 정의된다.
- 원본 LK 컨볼루션 파라미터 수:
$$N_{PRM,O} = C \times (C \times (K \times K \times K) + 1)$$
- 분해된 LK 컨볼루션 파라미터 수:
$$N_{PRM,D} = C \times ((2d-1)^3 + (\frac{K}{d})^3 + C + 3)$$
여기서 $C$는 채널 수이다. 논문에서는 $K=21$일 때 $d=3$을 적용함으로써 파라미터 수를 획기적으로 줄이면서도 성능을 유지함을 보였다.

#### 전체 연산 흐름
LK 어텐션 모듈의 전체 과정은 다음과 같은 방정식으로 표현된다.
1. 어텐션 맵 $A$ 생성:
$$A = \sigma_{\text{sigmoid}}(\text{Conv}_{1\times1\times1}(\text{Conv}_{DW}(\text{Conv}_{DWD}(\sigma_{lReLU}(\text{GN}(\text{Input}))))))$$
2. 최종 출력 계산:
$$\text{Output} = A \otimes (\sigma_{lReLU}(\text{GN}(\text{Input}))) + \sigma_{lReLU}(\text{GN}(\text{Input}))$$
여기서 $\text{GN}$은 Group Normalization, $\sigma_{lReLU}$는 Leaky ReLU 활성화 함수, $\otimes$는 요소별 곱셈(element-wise multiplication)을 의미한다.

### 2. LK Attention-based U-Net 아키텍처

제안된 모듈은 3D U-Net 구조에 통합되었다.
- **Encoder**: 6개의 스케일로 구성된 컨볼루션 블록을 통해 특징을 추출한다. 각 블록은 $3 \times 3 \times 3$ 커널의 컨볼루션 2층, GN, lReLU로 구성되며, stride-2 컨볼루션을 통해 다운샘플링을 수행한다.
- **Decoder**: Encoder와 대칭 구조를 가지며, $4 \times 4 \times 4$ Transposed Convolution으로 업샘플링을 수행한다.
- **LK Attention 적용**: 업샘플링된 특징 맵에 LK 어텐션 모듈을 배치한다. 모든 층에 적용하는 'Full' 방식과 디코더의 중간 층에만 적용하는 'Mid' 방식이 제안되었으며, 실험 결과 **Mid-type**이 효율성과 성능 면에서 가장 우수함을 확인하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - **CT-ORG**: 6개 장기(간, 방광, 폐, 신장, 뼈, 뇌)가 포함된 140개의 CT 영상.
    - **BraTS 2020**: 뇌종양 분할을 위한 다중 모달리티 MRI 영상.
- **평가 지표**: Dice Score(공간적 중첩도 측정)와 95% Hausdorff Distance (HD95, 경계 간의 거리 측정)를 사용하였다.
- **비교 대상**: 기본 U-Net, nnU-Net, CBAM, H2NF-Net 등 SOTA 모델들과 비교하였다.

### 주요 결과
1. **정량적 성능**: Mid-type LK attention-based U-Net이 대부분의 지표에서 SOTA 성능을 달성하였다. 특히 CT-ORG 데이터셋의 폐(lungs) 분할과 BraTS 2020의 종양 하위 영역(ET, TC) 분할에서 뚜렷한 향상을 보였다.
2. **연산 효율성**: 분해된 LK 컨볼루션은 원본 LK 컨볼루션 대비 파라미터 수를 약 $0.02\% \sim 0.17\%$ 수준으로 극적으로 낮추면서도, 성능 하락은 거의 없거나 오히려 향상되는 결과를 보였다.
3. **어텐션 위치 분석**: LK 어텐션 모듈을 디코더의 중간 단계(Mid)에 배치했을 때, 파라미터 수는 Full-type의 약 1/6 수준이면서 성능은 최적으로 나타났다.

## 🧠 Insights & Discussion

본 논문의 결과는 3D 의료 영상 분할에서 전역적인 문맥 정보를 포착하는 것이 매우 중요하며, 이를 위해 대형 커널을 사용하되 효율적인 분해 전략을 사용하는 것이 유효함을 입증한다.

**강점 및 해석**:
- **적응적 특징 선택**: LK 어텐션 맵을 시각화한 결과, 모델이 불필요한 배경 노이즈를 억제하고 실제 장기 및 종양 영역에 집중하고 있음을 확인하였다. 이는 모델의 예측 결과에 대해 국소적인 설명 가능성(local explanation)을 제공한다는 점에서 임상적 가치가 크다.
- **장거리 의존성 해결**: 큰 커널 커버리지를 통해 멀리 떨어진 복셀 간의 상관관계를 학습함으로써, 형태가 불규칙한 종양이나 큰 장기의 경계를 더 정확하게 잡아낼 수 있었다.

**한계 및 비판적 논의**:
- **특정 장기의 성능 저하**: 방광(bladder)과 같은 작은 장기의 경우, 어텐션 메커니즘 적용 시 오히려 Dice score가 약간 감소하는 경향이 나타났다. 이는 어텐션이 상대적으로 크고 뚜렷한 장기에 집중되면서 작은 장기에 대한 계산 자원이 분산되었기 때문으로 분석된다.
- **해상도 문제**: 전처리 과정에서 GPU 메모리 제한으로 인해 리샘플링(resampling)을 수행하였으며, 이로 인해 최종 결과물의 해상도가 Ground Truth보다 낮아지는 현상이 발생하였다.
- **데이터 품질 의존성**: 일부 MRI 모달리티(T2)의 블러링(blurring) 현상이 있을 때 분할 정확도가 떨어지는 모습이 관찰되어, 데이터의 무결성이 성능에 큰 영향을 미침을 시사한다.

## 📌 TL;DR

본 논문은 3D 의료 영상 분할의 연산 비용 문제를 해결하면서 전역적 문맥을 학습하기 위해, **분해된 대형 커널(Large-Kernel) 어텐션 모듈**을 제안하고 이를 U-Net에 통합하였다. 제안된 방법은 파라미터 수를 획기적으로 줄이면서도 다장기 및 뇌종양 분할 작업에서 SOTA 성능을 달성하였으며, 특히 모델의 예측 근거를 시각화할 수 있는 설명 가능성을 제공한다. 이 연구는 향후 고해상도 3D 의료 영상 분석 및 정밀 의료를 위한 딥러닝 아키텍처 설계에 중요한 기초 자료가 될 것으로 보인다.