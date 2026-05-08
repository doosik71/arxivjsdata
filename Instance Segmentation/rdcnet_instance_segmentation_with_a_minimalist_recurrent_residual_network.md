# RDCNet: Instance segmentation with a minimalist recurrent residual network

Raphael Ortiz, Gustavo de Medeiros, Antoine H.F.M. Peters, Prisca Liberali, and Markus Rempfler (2020)

## 🧩 Problem to Solve

본 논문은 생물학적 이미지 분석에서 매우 중요한 단계인 인스턴스 분할(Instance Segmentation) 문제를 해결하고자 한다. 특히 세포나 핵과 같이 동일한 클래스의 객체들이 밀집해 있는 환경에서 각 객체를 정확하게 구분하고 경계를 획정하는 것이 핵심이다.

기존의 딥러닝 기반 인스턴스 분할 방법들은 다음과 같은 한계를 가지고 있다. 첫째, Mask-RCNN과 같은 제안 기반(Proposal-driven) 방식은 계산 복잡도가 높으며, StarDist와 같이 고정된 형태(star-convex polyhedron)로 객체를 근사하는 방식은 객체의 모양이 둥글지 않을 경우 정확도가 떨어지는 제약이 있다. 둘째, 세그멘테이션 후 검출하는 방식이나 임베딩(Embedding) 기반 방식들은 종종 실제 분할 작업이 아닌 대리 작업(Surrogate tasks)을 통해 학습되므로 완전한 end-to-end 학습이 어렵다. 셋째, 일반적인 Fully Convolutional Network(FCN) 기반의 임베딩 방법은 이동 불변성(Translation Invariance)으로 인해 텍스처에 과도하게 의존하게 되어, 타일링(Tiling) 처리 시 문제가 발생하거나 모델 구조가 지나치게 복잡해지는 경향이 있다.

따라서 본 연구의 목표는 계산 효율성이 높으면서도(Light-weight), 해석 가능하며, 다양한 생물학적 이미지 모달리티에 범용적으로 적용 가능한 minimalist recurrent network를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **RDCNet(Recurrent Dilated Convolutional Network)**이라는 최소주의적 구조의 재귀 네트워크를 설계한 것이다. 주요 기여 사항은 다음과 같다.

1. **sSDC(shared Stacked Dilated Convolution) 레이어 도입**: 공유 가중치를 가진 적층 팽창 컨볼루션(Stacked Dilated Convolution)을 사용하여 파라미터 수를 획기적으로 줄이면서도, 초기 반복 단계에서 넓은 수용 영역(Receptive Field)을 확보하여 전역적인 컨텍스트를 효율적으로 학습하게 한다.
2. **재귀적 잔차 구조(Recurrent Residual Structure)**: 입력 이미지와 이전 단계의 예측값을 함께 입력으로 받아 출력을 반복적으로 정밀화(Refine)하는 구조를 채택하였다. 이를 통해 모델의 복잡도를 낮추면서도 성능을 높였으며, 각 반복 단계의 중간 예측값을 시각화함으로써 모델의 동작을 해석할 수 있게 하였다.
3. **Semi-convolutional Embedding 및 분할 손실 함수 결합**: 픽셀 좌표를 직접 활용하는 semi-convolutional 레이어를 통해 임베딩을 생성하고, 이를 대리 손실 함수가 아닌 실제 인스턴스 분할 손실 함수(Soft Jaccard Loss)로 최적화하여 학습의 효율성과 정확도를 높였다.

## 📎 Related Works

논문에서는 기존의 인스턴스 분할 접근 방식을 크게 두 가지 클래스로 구분하여 설명한다.

- **Proposal-driven methods**: Mask-RCNN과 같이 바운딩 박스를 먼저 예측한 후 내부를 분할하는 방식이다. StarDist는 이를 확장하여 별 모양의 다면체(Star-convex polyhedron)를 예측함으로써 강건함을 높였으나, 객체의 모양이 정형화되어 있지 않은 경우 경계 정확도가 떨어진다는 한계가 있다.
- **Segment-then-detect methods**: 먼저 전체 영역을 분할한 후 인스턴스를 구분하는 방식이다. 초기에는 배경, 전경, 경계를 구분하는 대리 작업에 의존하였으며, 최근에는 각 픽셀에 임베딩 벡터를 할당하여 유사도에 따라 인스턴스를 구분하는 임베딩 방법론이 제안되었다. 하지만 많은 임베딩 방법들이 여전히 대리 손실 함수를 사용하며, FCN의 특성상 텍스처 의존도가 높아 모델이 무거워지는 문제가 있다.

RDCNet은 이러한 한계를 극복하기 위해 구조적으로는 recurrent 구조를 통해 효율성을 꾀하고, 학습 측면에서는 semi-convolutional 임베딩과 직접적인 분할 손실 함수를 결합하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

RDCNet은 학습 가능한 다운샘플링(Downsampling) 레이어와 업샘플링(Upsampling) 레이어 사이에 하나의 얕은 재귀 블록(Recurrent Block)이 위치한 샌드위치 구조를 가진다. 3D 데이터의 경우, Z축의 이방성(Anisotropy)을 고려하여 XY축과 Z축의 스트라이드(Stride)를 다르게 설정한다.

### 재귀적 정밀화 과정 (Recurrent Refinement)

네트워크는 다음과 같은 잔차 형식의 수식을 통해 $i$번째 반복의 출력 $Y_i$를 생성한다.

$$Y_i = f_\theta(X, Y_{i-1}) + Y_{i-1}$$

여기서 $X$는 입력 이미지, $Y_{i-1}$은 이전 단계의 출력, $f_\theta$는 재귀 블록의 변환 함수이다. $Y_0$는 0으로 초기화된다.

### 핵심 구성 요소: sSDC (shared Stacked Dilated Convolution)

재귀 블록의 핵심인 sSDC는 다음과 같이 구성된다.

- **Grouped Convolutions**: 커널 크기 3의 그룹 컨볼루션을 사용한다.
- **Stacked Dilated Convolutions**: 각 그룹은 공유 가중치를 가진 팽창 컨볼루션들을 적층하여 사용함으로써, 추가적인 파라미터 증가 없이 수용 영역을 확장한다.
- **Point-wise ($1 \times 1$) Convolution**: 그룹 간의 정보를 섞어주고 채널 수를 조정하기 위해 사용된다.
- **Activation**: 잔차 경로를 명확히 유지하기 위해 첫 레이어를 제외한 모든 컨볼루션 앞에 Leaky ReLU pre-activation을 적용한다.

### 출력 및 손실 함수

최종 출력은 두 개의 브랜치로 나뉜다.

1. **Semantic Branch**: Softmax 활성화 함수를 통해 전경(Foreground)과 배경(Background) 확률을 예측한다.
2. **Instance Branch**: Additive semi-convolutional 레이어를 통해 인스턴스 임베딩을 생성한다. 이는 컨볼루션 출력에 픽셀/복셀 좌표를 더하는 방식이다.

학습 시에는 임베딩 $y$를 기반으로 픽셀 $u$가 인스턴스 $k$에 속할 확률 $P(u=k)$를 다음과 같이 계산한다.

$$P(u=k) = \exp\left( -\frac{\|y_u - \hat{y}_k\|^2}{2\sigma^2} \right)$$

여기서 $\hat{y}_k$는 인스턴스 $k$의 실제 마스크 영역 내 임베딩들의 평균값(Centroid)이며, $\sigma$는 대역폭(Bandwidth) 파라미터이다. $\sigma$는 물리적 의미를 갖는 마진(margin) 파라미터로 재정의될 수 있다.

$$\text{margin} = \sigma \sqrt{-2 \ln 0.5}$$

최종적으로 전경 예측과 인스턴스 임베딩 예측 모두에 대해 **Soft Jaccard Loss**를 사용하여 최적화한다.

### 추론 및 후처리 (Post-processing)

추론 단계에서는 정답 중심점($\hat{y}_k$)을 알 수 없으므로, 임베딩 공간에서 **Hough voting** 기법을 사용하여 중심점을 추정한다. 임베딩 값들을 히스토그램으로 빈(bin)에 담고, 마진 파라미터와 관련된 윈도우 크기 내에서 지역 최댓값(Local maxima)을 중심점으로 선택한다. 이후 전경 픽셀들을 임베딩 공간에서 가장 가까운 중심점에 할당함으로써 최종 인스턴스 라벨을 부여한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CVPPP2017 (식물 잎), MoNuSeg (조직학적 핵), 3D-ORG (3D 오가노이드 핵) 세 가지 서로 다른 모달리티의 데이터셋을 사용하였다.
- **비교 대상**: U-Net 기반 베이스라인(동일한 ESJ 손실 및 semi-conv 사용) 및 기존 SOTA 방법론(StarDist, Mask-RCNN 등)과 비교하였다.
- **지표**: Instance Precision, Recall, F1-score (IoU threshold 0.5), Symmetric Best Dice (SBD), Aggregated Jaccard Index (AJI)를 사용하였다.

### 주요 결과

1. **정량적 성능**:
    - **CVPPP2017**과 **3D-ORG** 데이터셋에서 SOTA 성능을 달성하였다.
    - **MoNuSeg**에서는 최상위권 성능은 아니지만, 사람 간 일치도(Inter-human agreement)보다 높은 성능을 보였으며, 도메인 특화 조정 없이도 매우 경쟁력 있는 결과를 냈다.
2. **효율성**:
    - U-Net 베이스라인 대비 파라미터 수가 약 $30\times$ 적음에도 불구하고 모든 지표에서 더 나은 성능을 보였다.
    - **메모리 점유율**: 3D 패치(32x256x256) 예측 시 U-Net은 8.8 GB의 VRAM을 사용한 반면, RDCNet은 3.8 GB만을 사용하여 3D 데이터 처리에 매우 유리함을 입증하였다.
3. **하이퍼파라미터 분석**:
    - **Dilation Rates**: 팽창률이 높을수록 F1-score가 증가하는 경향을 보였다(단, 3D-ORG에서는 패치 크기 제한으로 인해 너무 큰 팽창률은 오히려 성능을 저하시켰다).
    - **Iterations**: 반복 횟수가 증가함에 따라 성능이 향상되다가 약 5회 정도에서 정체(Plateau)되는 양상을 보였다. 5회 반복 시 U-Net 베이스라인 대비 연산량(MACs)을 68% 수준으로 낮추면서도 효율적인 성능을 낼 수 있었다.

## 🧠 Insights & Discussion

### 강점 및 해석

RDCNet의 가장 큰 강점은 **최소주의적 설계**임에도 불구하고 강력한 성능을 낸다는 점이다. 특히 sSDC와 재귀적 구조의 결합은 모델의 파라미터 수를 획기적으로 줄이면서도 수용 영역을 넓히는 효과를 주었다. 또한, 임베딩 기반 방식의 고질적인 문제인 텍스처 의존성을 semi-convolutional 레이어를 통해 해결함으로써 공간적 정보를 직접적으로 활용하게 하였다.

### 물리적 근거 기반의 하이퍼파라미터

본 논문은 $\sigma$나 $\text{margin}$과 같은 하이퍼파라미터를 단순히 수치적으로 튜닝하는 것이 아니라, 실제 객체의 크기나 밀도와 같은 물리적 특성과 연결하여 설정할 수 있음을 보여주었다. 이는 모델의 설정 과정을 보다 체계적이고 해석 가능하게 만든다.

### 한계 및 비판적 논의

MoNuSeg 데이터셋에서 절대적인 SOTA를 달성하지 못한 점은 아쉬우나, 이는 도메인 특화 모델들이 사용한 복잡한 제약 조건들을 제외하고 범용적인 구조만으로 접근했기 때문으로 해석된다. 또한, 반복 횟수가 증가함에 따라 추론 시간이 선형적으로 증가하므로, 실시간성이 극도로 중요한 작업에서는 최적의 반복 횟수($i=5$)를 찾는 것이 필수적이다.

## 📌 TL;DR

본 논문은 생물학적 이미지의 인스턴스 분할을 위해 **RDCNet**이라는 경량 재귀 네트워크를 제안하였다. 이 모델은 **sSDC**를 통해 파라미터 효율성을 극대화하고, **semi-convolutional 임베딩**과 **Soft Jaccard Loss**를 결합하여 end-to-end 학습을 구현하였다. 실험 결과, U-Net 대비 파라미터를 $\approx 30$배 줄이면서도 3개 데이터셋 중 2개에서 SOTA를 달성하였으며, 특히 메모리 효율성이 뛰어나 3D 볼륨 데이터 처리에 매우 적합함을 입증하였다. 향후 이 구조는 시계열(Temporal) 차원을 포함한 4D 데이터 분석으로 확장될 가능성이 높다.
