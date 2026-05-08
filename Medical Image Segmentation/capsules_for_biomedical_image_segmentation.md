# Capsules for Biomedical Image Segmentation

Rodney LaLonde, Ziyue Xu, Ismail Irmakci, Sanjay Jain, Ulas Bagci (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에 캡슐 네트워크(Capsule Network)를 처음으로 적용하여 해결하고자 한다. 특히, 병변이 있는 폐(pathological lungs)의 CT 스캔 영상과 인체 허벅지의 근육 및 지방 조직 MRI 영상 분할을 주요 대상으로 한다.

의료 영상 분할에서 해결해야 할 핵심 문제는 다음과 같다.

1. **CNN의 구조적 한계**: 기존의 Convolutional Neural Networks(CNN)는 뉴런의 출력이 스칼라(scalar) 값이며 가산적(additive)인 특성을 가진다. 이로 인해 커널 내 뉴런 간의 공간적 관계를 무시하게 되며, 특징 맵(feature map)은 단순히 특징의 존재 여부만을 나타낼 뿐, 특징의 정확한 위치, 포즈(pose), 변형 등의 상세 정보를 보존하지 못한다.
2. **의료 데이터의 복잡성**: 병리적 폐 영상의 경우, 고도의 클래스 내 변동성(intra-class variation), 노이즈, 아티팩트 및 비정상적 구조가 많아 경계선을 정확히 획정하는 것이 매우 어렵다.
3. **캡슐 네트워크의 계산 비용**: 기존의 Capsule Network(CapsNet)는 벡터 표현을 저장하고 Dynamic Routing을 수행하는 과정에서 메모리와 계산 비용이 기하급수적으로 증가하여, 고해상도 의료 영상(예: $512 \times 512$)에 적용하는 것이 사실상 불가능했다.

따라서 본 논문의 목표는 계산 효율성을 높인 캡슐 네트워크 구조를 설계하여, 적은 파라미터로도 고해상도 의료 영상에서 기존 CNN 기반 모델보다 우수한 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 캡슐 네트워크의 표현력은 유지하면서 계산 부담을 획기적으로 줄이는 **SegCaps** 아키텍처를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Locally-constrained Routing**: 모든 자식 캡슐이 모든 부모 캡슐로 라우팅되는 기존 방식 대신, 정의된 국소적 커널(spatially-local kernel) 내에서만 라우팅이 일어나도록 제한하여 메모리 사용량을 줄였다.
2. **Transformation Matrix Sharing**: 동일한 캡슐 타입 내의 모든 그리드 멤버가 변환 행렬(transformation matrix)을 공유하도록 설계하여 학습해야 할 파라미터 수를 크게 감소시켰다.
3. **Deconvolutional Capsules**: 국소적 라우팅으로 인해 손실된 전역적 문맥 정보(global context)를 보완하기 위해, Transposed Convolution을 이용한 '디컨볼루션 캡슐' 개념을 도입하여 딥 인코더-디코더(Deep Encoder-Decoder) 구조를 구현하였다.
4. **Masked Reconstruction Regularization**: 분할 작업의 정규화를 위해, 긍정 클래스(target class)의 픽셀만을 마스킹하여 재구성하는 손실 함수를 도입함으로써 네트워크가 입력 데이터의 분포를 더 잘 학습하도록 유도하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들을 분석하고 차별점을 제시한다.

- **CNN 기반 분할 모델**: U-Net, FCN 및 이를 확장한 Tiramisu(DenseNet 기반 U-Net) 등이 의료 영상 분할의 표준으로 사용되고 있다. 하지만 이들은 앞서 언급한 스칼라 활성화 함수의 한계로 인해 공간적 관계 보존 능력이 떨어진다.
- **기존 캡슐 네트워크**: Sabour et al. (2017)이 제안한 CapsNet은 MNIST, CIFAR10과 같은 작은 이미지의 분류 작업에서 뛰어난 성능을 보였으나, 모든 캡슐 간의 완전 연결(fully-connected) 라우팅 방식으로 인해 고해상도 이미지 처리에는 부적합하다.
- **차별점**: SegCaps는 캡슐 네트워크를 '분류'가 아닌 '분할' 작업에 처음으로 적용하였으며, 국소적 라우팅과 변환 행렬 공유를 통해 기존 캡슐 네트워크가 처리하지 못했던 고해상도($512 \times 512$) 이미지 처리를 가능하게 하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인

SegCaps는 딥 인코더-디코더 구조를 가진다. 입력 영상은 먼저 2D Convolutional layer를 거쳐 16개의 특징 맵으로 변환되며, 이후 Convolutional Capsule layer와 Deconvolutional Capsule layer가 교대로 배치되어 특징 추출과 해상도 복원을 수행한다.

### 2. Convolutional Capsules 및 Routing

레이어 $\ell$의 자식 캡슐 $C$가 레이어 $\ell+1$의 부모 캡슐 $P$로 전달되는 과정은 다음과 같다.

**예측 벡터(Prediction Vector) 생성**:
부모 캡슐 타입 $t^{\ell+1}_j$에 대해, 학습된 변환 행렬 $M_{t^{\ell+1}_j}$와 국소 커널 내의 자식 캡슐 출력 $U_{x,y|t^\ell_i}$를 곱하여 예측 벡터 $\hat{u}$를 생성한다.
$$\hat{u}_{x,y|t^\ell_i} = M_{t^{\ell+1}_j} \cdot U_{x,y|t^\ell_i}$$
여기서 $M$은 공간적 위치 $(x,y)$에 관계없이 공유된다.

**Dynamic Routing**:
부모 캡슐 $p_{x,y}$는 예측 벡터들의 가중치 합으로 계산된다.
$$p_{x,y} = \sum_{n} r_{t^\ell_i|x,y} \hat{u}_{x,y|t^\ell_i}$$
라우팅 계수 $r$은 다음과 같은 Softmax 함수로 결정된다.
$$r_{t^\ell_i|x,y} = \frac{\exp(b_{t^\ell_i|x,y})}{\sum_{t^{\ell+1}_j} \exp(b_{t^\ell_i|x,y})}$$
최종 출력 벡터 $v_{x,y}$는 비선형 Squashing 함수를 통해 생성된다.
$$v_{x,y} = \frac{\|p_{x,y}\|^2}{1+\|p_{x,y}\|^2} \frac{p_{x,y}}{\|p_{x,y}\|}$$

### 3. Deconvolutional Capsules

디코더 단계에서 해상도를 높이기 위해 Transposed Convolution과 유사한 방식을 사용한다. Fractional striding을 통해 캡슐 그리드의 높이와 너비를 업샘플링한 후, 위와 동일한 국소 라우팅 과정을 거쳐 부모 캡슐을 생성한다.

### 4. Reconstruction Regularization

네트워크가 가장 지배적인 특징에만 매몰되는 '모드 붕괴(mode collapse)'를 방지하기 위해 재구성 손실을 추가한다. 긍정 클래스에 해당하는 픽셀만을 마스킹하여 입력 영상을 재구성하며, 다음과 같은 MSE 손실 함수를 사용한다.
$$L_R = \gamma \sum_{x,y} \|R_{x,y} - O^r_{x,y}\|$$
여기서 $R_{x,y}$는 마스킹된 타겟 픽셀, $O^r_{x,y}$는 재구성 네트워크의 출력이다. 전체 손실 함수는 이 재구성 손실($L_R$)과 가중치 기반의 이진 교차 엔트로피(Weighted BCE) 손실의 합으로 구성된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**:
  - 폐 CT: LIDC-IDRI, LTRC, UHG (임상), JHU-TBS, JHU-TB (전임상/마우스) 총 1,960개 스캔.
  - 허벅지 MRI: BLSA 데이터셋 150개 스캔 (3가지 대비 영상).
- **비교 대상**: U-Net, Tiramisu, P-HNN.
- **평가 지표**: Dice Similarity Coefficient (Dice), Hausdorff Distance (HD).

### 2. 정량적 결과

- **폐 분할**: 모든 데이터셋에서 SegCaps가 Dice score와 HD 면에서 SOTA(State-of-the-art) 모델들을 능가하였다. 특히 전임상(마우스) 데이터셋인 JHU-TBS에서 U-Net(90.38%)보다 높은 93.35%의 Dice score를 기록하였다.
- **MRI 분할**: 허벅지 근육 및 지방 조직 분할에서도 U-Net과 대등하거나 더 우수한 성능을 보였으며, 이전 SOTA 방법론(Irmakci et al., 2018)을 크게 앞질렀다.
- **파라미터 효율성**: SegCaps는 매우 적은 수의 파라미터만으로 높은 성능을 냈다. 구체적으로 U-Net 파라미터의 4.6%, P-HNN의 9.5%, Tiramisu의 14.9% 수준만 사용하였다.

### 3. 정성적 결과 및 일반화 능력

- **분할 품질**: CNN 기반 모델들이 자주 겪는 과분할(over-segmentation) 및 분할 누수(segmentation-leakage) 현상이 SegCaps에서는 현저히 적게 나타났다.
- **어핀 불변성(Affine Equivariance)**: 자연 이미지(PASCAL VOC)를 이용해 단 한 장의 이미지로 과적합시킨 후, 회전 및 반전된 이미지에 대해 테스트한 결과, U-Net은 실패한 반면 SegCaps는 강건하게 대응하여 캡슐 네트워크의 포즈 일반화 능력을 입증하였다.

## 🧠 Insights & Discussion

**강점 및 분석**:

- **파라미터 효율성**: 동일한 파라미터 수로 CNN 모델을 축소하여 비교했을 때, SegCaps가 더 높은 성능을 보였다. 이는 캡슐의 벡터 표현 방식이 CNN의 스칼라 방식보다 파라미터당 정보 밀도가 훨씬 높음을 시사한다.
- **구조적 필요성**: 단순 3층 캡슐 구조(Base-Caps)보다 디컨볼루션 캡슐을 이용한 인코더-디코더 구조가 성능을 비약적으로 향상시켰다. 이는 의료 영상 분할과 같이 전역적 문맥 정보가 필수적인 작업에서 딥 구조의 중요성을 보여준다.
- **정규화 효과**: 재구성 손실($L_R$)을 추가했을 때 성능이 향상되었으며, 이는 모델이 단순한 판별적 특징뿐만 아니라 데이터의 전반적인 분포를 학습하게 함으로써 일반화 성능을 높였기 때문으로 분석된다.

**한계 및 논의**:

- **라우팅 반복 횟수**: 실험 결과 3회의 반복(iteration)이 가장 최적이었으나, 이는 여전히 계산 비용을 증가시키는 요인이다. 논문에서도 언급되었듯이, 더 효율적인 라우팅 메커니즘에 대한 연구가 필요하다.
- **데이터 의존성**: 전임상 데이터의 경우 해부학적 변동성이 매우 커서 자동 분할이 극도로 어려우나, 본 연구를 통해 딥러닝 기반의 전임상 분할 가능성을 처음으로 제시하였다.

## 📌 TL;DR

본 논문은 캡슐 네트워크를 의료 영상 분할에 최초로 적용한 **SegCaps**를 제안한다. **국소적 라우팅(Locally-constrained Routing)**과 **변환 행렬 공유**, 그리고 **디컨볼루션 캡슐**을 통해 기존 캡슐 네트워크의 메모리 문제를 해결하고 고해상도 이미지 처리를 가능케 하였다. 실험 결과, SegCaps는 U-Net 등 기존 SOTA 모델보다 **약 5% 미만의 파라미터만 사용하고도 더 높은 분할 정확도(Dice, HD)를 달성**하였으며, 특히 입력 영상의 회전 및 반전에 대해 매우 강건한 일반화 능력을 보였다. 이 연구는 캡슐 네트워크가 의료 영상 분석에서 매우 효율적인 대안이 될 수 있음을 입증하였다.
