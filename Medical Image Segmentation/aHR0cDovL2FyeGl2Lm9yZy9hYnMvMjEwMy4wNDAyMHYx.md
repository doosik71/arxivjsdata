# NeRD: Neural Representation of Distribution for Medical Image Segmentation

Hang Zhang, Rongguang Wang, Jinwei Zhang, Chao Li, Gufeng Yang, Pascal Spincemaille, Thanh D. Nguyen, and Yi Wang (2021)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 널리 사용되는 Convolutional Neural Networks(CNNs), 특히 U-Net 구조가 겪는 **특징 분포 변화(Feature Distribution Shifting)** 문제를 해결하고자 한다.

CNN의 핵심은 공간 불변성(Spatial Invariance)을 가진 필터를 공유하여 파라미터를 줄이고 일반화 능력을 높이는 것이다. 그러나 실제 구현 과정에서 사용되는 다음과 같은 연산들이 공간 불변성을 깨뜨리고 특징 분포를 왜곡시킨다.
1. **Padding**: 합성곱 계층의 패딩 연산은 특징 맵에 아티팩트(Artefacts)를 생성하며, 이미지의 경계와 중심 간의 특징 분포를 다르게 만든다.
2. **Pooling**: Max-pooling이나 Strided Convolution과 같은 다운샘플링 연산은 샘플링 이론을 무시하게 되어 공간 불변성 속성을 파괴한다.

이러한 문제는 특히 뇌 병변 분할과 같은 정밀한 작업에서, 이미지 경계나 뇌실(Ventricle) 근처(이미지 중심부)에 위치한 병변들이 오분류되는 결과(과분할 또는 미검출)로 이어진다. 따라서 본 논문의 목표는 이러한 공간적 위치에 따른 특징 분포의 변화를 보상하여 다시 공간 불변성을 회복하는 통합 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Neural Implicit Representation (NIR)** 기술을 응용하여, 이미지의 좌표(Coordinate)를 해당 위치의 특징 분포(Feature Distribution)로 매핑하는 함수를 학습하는 것이다.

기존의 네트워크가 모든 위치에서 동일한 특징 분포를 가정하는 것과 달리, NeRD(Neural Representation of Distribution)는 픽셀의 좌표 정보를 입력받아 그 위치에 적합한 분포 파라미터를 동적으로 추정한다. 이를 통해 패딩이나 풀링으로 인해 발생한 공간적 왜곡을 보정하고, 위치에 최적화된 정규화 또는 분류를 수행함으로써 분할 성능을 향상시킨다.

## 📎 Related Works

### 뇌 병변 분할 (Brain Lesion Segmentation)
다양한 자동화 접근 방식(예: 2.5D stacked DenseNet, Folded Attention Network, Geometric Loss, Boundary Loss 등)이 제안되었으며, 이들은 주로 문맥 정보 획득이나 데이터 불균형 문제를 해결하는 데 집중하였다. 그러나 이러한 연구들은 CNN 내부에서 발생하는 특징 분포의 공간적 변화(Feature Distribution Shifting) 문제는 고려하지 않았다.

### Neural Implicit Representation (NIR)
최근 그래픽스 분야에서 3D 형상이나 외관을 표현하기 위해 좌표를 값으로 매핑하는 연속 함수로 표현하는 방식(예: DeepSDF, NeRF)이 발전하였다. 본 논문은 이러한 NIR의 개념을 이미지 특징 분포 매핑에 처음으로 적용하였다.

### 메타 학습 (Meta Learning)
네트워크의 가중치를 다른 네트워크가 예측하게 하는 전략이다. 본 연구는 이미지 좌표를 기반으로 네트워크 가중치를 동적으로 추정한다는 점에서 일종의 메타 학습 전략을 사용하고 있다.

## 🛠️ Methodology

### 전체 파이프라인
NeRD 프레임워크는 크게 세 단계로 구성된다.
1. **특징 추출**: U-Net과 같은 Encoder-Decoder 구조를 통해 최종 특징 맵 $X \in \mathbb{R}^{H \times W \times C}$를 생성한다.
2. **분포 추정**: Offset Generator가 픽셀별 위치 벡터 $v$를 제공하면, 이를 MLP 기반의 Distribution Calibrator가 받아 해당 위치의 분포 파라미터를 생성한다.
3. **분류**: 추정된 분포 파라미터를 이용하여 특징 벡터를 정규화하거나 분류기를 적용한 후, 최종 Sigmoid 함수를 통해 분할 맵을 얻는다.

### 주요 구성 요소 및 수식 설명

#### 1. 위치 벡터 (Position Vector)
각 픽셀 $v$의 위치는 이미지의 상, 우, 하, 좌 경계로부터의 거리로 정의된 4차원 벡터로 표현된다.
$$v = (d_t, d_r, d_b, d_l)$$

#### 2. NeRDm (Mean-Variance Estimator)
특징 분포를 가우시안 분포 $x_v \sim \mathcal{N}(\mu, \Sigma)$로 가정한다. 계산 효율성을 위해 공분산 행렬 $\Sigma$를 벡터 $\sigma$로 단순화하고, MLP $f_\Theta$를 통해 $\mu$와 $\sigma$를 추정한다.
$$f_\Theta : (v) \to (\mu, \sigma)$$

수치적 안정성을 위해 실제로는 $1/\sigma$와 $-\mu/\sigma$를 직접 추정하며, 최종 출력 $s_v$는 다음과 같이 계산된다.
$$s_v = w^T \left( x_v \frac{1}{\sigma} - \frac{\mu}{\sigma} \right)$$
여기서 $w$는 MLP 분류기의 가중치이며, 이는 특징 벡터 $x_v$에 대해 Z-score 정규화를 수행한 후 선형 분류를 적용하는 것과 같다.

#### 3. NeRDc (Pixel-aligned Classifier)
분포 파라미터를 추정하는 대신, 각 픽셀 위치마다 고유한 선형 분류기 가중치 $w_v$를 직접 추정하는 방식이다.
$$f_\Theta : (v) \to (w_v)$$
$$s_v = x_v^T w_v$$
이 방식은 위치별로 최적화된 전용 분류기를 갖는 것과 같은 효과를 준다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - White Matter Hyperintensities (WMH): 60개의 3D 스캔 (T1, FLAIR 모달리티).
    - Left Atrial (LA) segmentation: 60명의 환자로부터 얻은 154개의 3D LGE-MRIs.
- **비교 대상**: Baseline U-Net, NeRDc, NeRDm.
- **네트워크 용량**: 필터 수에 따라 Low-capacity (256)와 High-capacity (512) 두 그룹으로 나누어 실험하였다.
- **평가 지표**: 
    - WMH: Dice, LDice (Lesion-wise Dice), LTPR (True Positive Rate), LFPR (False Positive Rate).
    - LA: Dice, Jaccard, HD (Hausdorff Distance), 95HD, ASD (Average Surface Distance).

### 주요 결과
1. **WMH 분할**: 
    - Low-capacity 그룹에서 NeRDc는 U-Net 대비 LDice가 2.8% 상승하고 LFPR이 3% 감소하는 등 성능 향상을 보였다.
    - 특히 **Low-capacity NeRDc의 성능이 High-capacity U-Net의 성능과 대등하거나 더 뛰어난 결과**를 보였다.
2. **LA 분할**: 
    - Low-capacity 그룹에서 NeRDm이 HD 지표를 4.5만큼 크게 낮추며 유의미한 개선을 보였다.
    - High-capacity 그룹에서는 NeRDc가 모든 지표에서 가장 우수한 성능을 기록하였다.
3. **정성적 분석**: 
    - Vanilla U-Net은 이미지 경계나 중심부(뇌실 근처)에서 병변을 놓치거나(Missing) 잘못 검출(Over-segmenting)하는 경향이 있었으나, NeRD 모듈을 적용한 모델들은 이러한 오류를 효과적으로 억제하였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 CNN의 고질적인 문제인 패딩과 풀링으로 인한 공간 불변성 파괴 문제를 "좌표 기반의 특징 분포 추정"이라는 새로운 관점에서 접근하였다. 특히 모델의 파라미터 수를 늘려 용량을 키우는 대신, NeRD 모듈을 통해 위치 정보를 효율적으로 활용함으로써 **적은 연산 자원(Low-capacity)으로도 더 큰 모델(High-capacity) 이상의 성능을 낼 수 있음**을 입증하였다.

### 한계 및 논의사항
- **가정의 단순함**: 특징 분포가 단순히 이미지 경계로부터의 거리(4D 벡터)에 의존한다고 가정하였다. 하지만 실제 특징 분포는 이미지의 국소적 콘텐츠(Context)에 따라서도 변할 수 있으므로, 좌표 정보 외에 지역적 특징을 함께 고려하는 방식에 대한 논의가 필요하다.
- **일반화 가능성**: 뇌 병변과 좌심방 분할이라는 두 가지 작업에서 효과를 보였으나, 다양한 다른 의료 영상 도메인에서도 동일한 분포 시프트 문제가 발생하는지, 그리고 NeRD가 보편적인 해결책이 될 수 있는지는 추가 검증이 필요하다.

## 📌 TL;DR

이 논문은 CNN의 패딩 및 풀링 연산으로 인해 발생하는 **특징 분포 시프트(Feature Distribution Shifting)** 문제를 해결하기 위해, 이미지 좌표를 특징 분포 파라미터로 매핑하는 **NeRD(Neural Representation of Distribution)** 모듈을 제안하였다. NeRD는 위치별로 최적화된 정규화 및 분류를 가능하게 하여, 특히 이미지 경계 및 중심부에서의 분할 정확도를 높였다. 결과적으로 적은 파라미터만으로도 대형 네트워크 수준의 성능을 달성하였으며, 이는 향후 효율적인 의료 영상 분석 모델 설계에 중요한 기여를 할 것으로 보인다.