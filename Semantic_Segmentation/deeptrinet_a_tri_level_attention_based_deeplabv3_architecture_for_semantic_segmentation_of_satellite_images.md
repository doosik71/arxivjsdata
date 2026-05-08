# DeepTriNet: A Tri-Level Attention Based DeepLabv3+ Architecture for Semantic Segmentation of Satellite Images

Tareque Bashar Ovi et al. (2023)

## 🧩 Problem to Solve

본 논문은 위성 이미지의 semantic segmentation(의미론적 분할) 과정에서 발생하는 성능 저하 문제를 해결하고자 한다. 위성 이미지 내의 소규모 객체를 인식하는 것은 매우 까다로운 과제인데, 이는 기존의 딥러닝 네트워크들이 하위 레벨(low-level)의 특징을 간과하거나, 서로 다른 특징 맵(feature map)들이 보유한 정보량의 차이를 적절히 처리하지 못하기 때문이다.

따라서 본 연구의 목표는 인코더(encoder)와 디코더(decoder) 사이의 의미론적 정보 격차(semantic information gap)를 줄여 위성 이미지의 세밀한 분할 성능을 높이는 새로운 네트워크 구조인 DeepTriNet을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 기존의 DeepLabv3+ 아키텍처에 두 가지 핵심 Attention 메커니즘을 통합하여 하이브리드 구조를 설계한 것이다.

첫째, **Tri-Level Attention Unit (TAU)**을 도입하였다. 이는 채널(Channel), 공간(Spatial), 픽셀(Pixel)이라는 세 가지 수준의 추상화를 결합한 자기 지도(self-supervised) attention 기법으로, 관련 정보의 일반화 성능을 높여준다.

둘째, 디코더 부분에 **Squeeze-and-Excitation Networks (SENets)**를 통합하였다. 이를 통해 채널 간의 상호 의존성을 명시적으로 모델링하고, 각 채널의 특징 출력을 적응적으로 재교정(re-calibrate)하여 중요한 특징에 더 많은 가중치를 부여하도록 설계하였다.

## 📎 Related Works

최근 위성 이미지 분할 분야에서는 CNN 기반의 다양한 접근 방식이 시도되었다. FastFCN, ResNet-50, FCN-8, SegNet, U-Net, 그리고 DeepLabv3+ 등이 대표적이다. 기존 연구들은 주로 GID-2 데이터셋의 일부 클래스나 LandCover.ai 데이터셋의 단일 클래스(예: 수체 추출)에 집중하는 경향이 있었다.

본 논문은 기존 연구들이 단순한 end-to-end 딥러닝 모델을 사용한 것과 달리, 자기 지도 학습 기반의 Attention 메커니즘을 적용하여 15개의 클래스를 가진 GID-2 데이터셋과 4개 클래스의 LandCover.ai 데이터셋 모두에서 성능을 검증했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

DeepTriNet은 기본적으로 DeepLabv3+의 인코더-디코더 구조를 따른다. 인코더는 이미지에서 필수 정보를 추출하고, 디코더는 이를 바탕으로 원래 해상도에 가까운 결과물을 재구성한다. 여기에 TAU와 SENets가 추가되어 특징 추출 및 정제 과정을 강화한다.

### 주요 구성 요소 및 역할

1. **Tri-Level Attention Unit (TAU)**:
   TAU는 세 가지 수준의 attention을 통해 특징의 관련성을 높인다.
   - **Channel Attention (CA)**: 넓은 관점에서 정보량이 많은 채널을 강조한다.
   - **Spatial Attention (SA)**: 관심 영역(ROI)이 포함된 국소적인 공간 영역에 집중한다.
   - **Pixel Attention (PA)**: 가장 낮은 수준에서 각 픽셀의 특징 관련성을 평가한다.
   TAU는 DeepLabv3+의 **Atrous Spatial Pyramid Pooling (ASPP)** 섹션에 통합되어, 각 합성곱(convolution) 연산 이후에 배치되어 정제된 특징을 추출한다.

2. **Squeeze-and-Excitation Networks (SENets)**:
   SENets는 채널 간의 관계를 모델링하여 중요도가 낮은 채널은 억제하고 중요한 채널은 강화한다. 특징 맵을 단일 수치로 압축(Squeeze)하여 전역적인 정보를 파악한 뒤, 2층 신경망을 통해 각 채널에 적용할 가중치를 생성(Excitation)하여 원래 특징 맵에 곱하는 방식으로 작동한다. 이는 디코더 부분에 배치되어 최종 출력의 정밀도를 높인다.

### 학습 및 전처리 절차

고해상도 위성 이미지의 픽셀 손실을 방지하기 위해 **Grid and Patch** 방식을 사용한다. 전체 이미지를 $256 \times 256 \times 3$ 크기의 784개 서브 이미지(패치)로 분할하여 처리하며, 추론 후에는 이를 다시 결합하여 고해상도 이미지를 재구성한다.

입력 데이터는 다음과 같은 수식을 통해 $-1$에서 $1$ 사이의 값으로 정규화된다.
$$I^n_{i,j} = \frac{I_{i,j}}{127.5} - 1$$
여기서 $I_{i,j}$는 원본 이미지의 픽셀 값이며, $I^n_{i,j}$는 정규화된 픽셀 값이다.

## 📊 Results

### 실험 설정

- **데이터셋**: LandCover.ai (4 클래스: 건물, 숲, 물, 도로) 및 GID-2 (15 클래스: 정밀 토지 피복 분류)
- **측정 지표**: Accuracy, Precision, Recall, IoU (Intersection over Union)

### 정량적 결과

DeepTriNet의 성능 결과는 다음과 같다.

| 데이터셋 | Accuracy | IoU | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **LandCover.ai (4 class)** | $98\%$ | $80\%$ | $88\%$ | $79\%$ |
| **GID-2 (15 class)** | $77\%$ | $58\%$ | $68\%$ | $55\%$ |

### 결과 분석

LandCover.ai 데이터셋에서 DeepTriNet은 기존의 DeepLabv3+, U-Net, SegNet보다 우수한 정확도를 보였다. 단, 수체(water) 하나만을 분류한 특정 연구보다는 정확도가 낮았으나, 이는 다중 클래스 분류라는 더 어려운 작업임을 고려해야 한다. GID-2 데이터셋의 경우 클래스 수가 15개로 많고 데이터의 다양성이 높음에도 불구하고 유의미한 성능을 달성하였다. 학습 곡선 분석 결과, 과적합(overfitting) 없이 안정적으로 수렴하는 양상을 보였다.

## 🧠 Insights & Discussion

본 논문은 TAU와 SENets라는 두 가지 attention 기법을 전략적으로 배치함으로써 위성 이미지의 복잡한 특징을 효과적으로 포착할 수 있음을 입증하였다. 특히 TAU가 ASPP 단계에서 특징을 정제하고, SENets가 디코더에서 채널 가중치를 조정하는 구조는 semantic gap을 줄이는 데 기여한 것으로 판단된다.

다만, 논문에서 언급되었듯이 본 모델은 높은 성능을 내기 위해 **상당한 양의 학습 데이터와 계산 자원(Computing Power)**을 필요로 한다는 한계가 있다. 또한, 전처리 과정에서 이미지를 패치 단위로 나누어 처리하므로, 패치 경계 부분에서의 연속성을 완벽하게 보장하는 방법에 대한 심도 있는 논의는 부족한 편이다.

비판적으로 해석하자면, GID-2 데이터셋에서의 결과($\text{IoU } 58\%$)는 LandCover.ai에 비해 상대적으로 낮다. 이는 클래스 수가 증가함에 따라 모델이 각 클래스의 미세한 차이를 구분하는 데 여전히 어려움이 있음을 시사하며, 향후 더 강력한 backbone 네트워크나 정교한 손실 함수(loss function) 도입이 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 DeepLabv3+ 아키텍처에 **Tri-Level Attention Unit (TAU)**과 **Squeeze-and-Excitation Networks (SENets)**를 통합한 **DeepTriNet**을 제안하여 위성 이미지의 semantic segmentation 성능을 향상시켰다. 특히 LandCover.ai 데이터셋에서 높은 정확도($98\%$)를 기록하며 다중 클래스 분류 능력을 입증하였다. 이 연구는 정밀한 토지 이용 및 피복 지도 작성, 천연자원 관리 및 도시 변화 탐지 분야에 실제적으로 기여할 가능성이 크다.
