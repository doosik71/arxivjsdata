# CapsNet for Medical Image Segmentation

Minh Tran, Viet-Khoa Vo-Ho, Kyle Quinn, Hien Nguyen, Khoa Luu, and Ngan Le (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 기존의 Convolutional Neural Networks(CNNs)가 가진 구조적 한계를 해결하고자 한다. CNN은 비정형 데이터에서 특징을 자동으로 추출하는 능력이 뛰어나지만, 다음과 같은 치명적인 문제점을 가지고 있다.

첫째, CNN은 이미지의 회전(Rotation) 및 아핀 변환(Affine Transformation)에 매우 민감하다. CNN의 일반화 성능은 이러한 변형이 포함된 대규모 레이블 데이터셋에 의존하는데, 의료 영상 분야에서는 데이터 획득 비용이 매우 높고 엄격한 개인정보 보호 규정으로 인해 대량의 주석(Annotation) 데이터를 확보하는 것이 어렵다.

둘째, CNN의 Pooling 계층은 특징의 존재 여부는 알려주지만, 특징들 간의 상대적인 위치 및 기하학적 관계(Positional information/Spatial relation)를 손실시키는 경향이 있다. 이로 인해 입력 이미지의 방향이나 크기가 조금만 달라져도 네트워크의 성능이 급격히 저하되는 문제가 발생한다.

따라서 본 연구의 목표는 CNN의 이러한 한계를 극복하기 위해 Capsule Network(CapsNet)를 의료 영상 분할에 도입하고, 2D 및 3D 볼륨 데이터에 최적화된 다양한 CapsNet 아키텍처를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 뉴런의 스칼라 출력(Scalar output)을 벡터 출력(Vector output), 즉 '캡슐(Capsule)'로 대체하여 객체의 계층적 포즈 관계(Hierarchical pose relationships)를 보존하는 것이다.

CapsNet의 중심적인 직관은 이미지 내의 개별 파트(Part)와 전체(Whole) 사이의 관계를 학습함으로써, 단순한 특징의 존재 여부가 아닌 객체의 기하학적 속성(위치, 방향, 크기 등)을 인코딩하는 것이다. 이를 통해 데이터 증강(Data Augmentation)에 과도하게 의존하지 않고도 입력의 변형에 대해 더 높은 강건성(Robustness)을 확보하고, 적은 양의 데이터로도 일반화 성능을 높일 수 있다.

## 📎 Related Works

논문에서는 CapsNet의 성능을 개선하기 위한 기존 연구들을 두 가지 그룹으로 분류하여 설명한다.

1. **라우팅 메커니즘 개선 연구**: 기본 CapsNet의 Dynamic Routing이 가진 높은 계산 복잡도를 해결하려는 시도들이다. EM Routing은 Expectation-Maximization 알고리즘을 통해 결합 계수를 업데이트하며, Straight-through attentive routing은 어텐션 모듈을 통해 복잡도를 줄인다. 또한, 벡터 대신 행렬이나 텐서를 사용하는 방식 등이 제안되었다.
2. **네트워크 아키텍처 확장 연구**: CNN 계층과 캡슐 계층을 결합한 형태, unsupervised capsule autoencoder, Aff-CapsNets, 그리고 Memory-augmented CapsNet 등이 있다. 특히 DECAPS는 Inverted Dynamic Routing(IDR)을 통해 성능을 높여 흉부 방사선 전문의 수준의 성능을 달성한 바 있다.

본 논문은 이러한 기초 연구들을 바탕으로, 의료 영상 분할이라는 구체적인 Task에 맞게 UNet 구조와 캡슐 네트워크를 결합하거나, 자가 지도 학습(Self-supervised learning)을 도입하여 데이터 부족 문제를 해결하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. CapsNet의 기본 원리

CapsNet은 특징 추출 후 이를 벡터 형태의 캡슐로 표현한다. 각 캡슐의 벡터 길이는 해당 엔티티가 존재할 확률을 나타내며, 벡터의 방향은 엔티티의 속성(Instantiation parameters)을 나타낸다.

**핵심 구성 요소 및 절차:**

- **Non-shared transformation module**: 하위 캡슐 $u_i$를 상위 캡슐 $\hat{u}_{j|i}$로 변환하여 예측 투표(Vote)를 생성한다.
  $$\hat{u}_{j|i} = W_{ij} u_i$$
- **Dynamic routing layer**: 투표 결과와 상위 캡슐의 출력이 일치하는지 확인하여 결합 계수 $c_{ij}$를 업데이트하는 반복 과정이다.
- **Squashing function**: 벡터의 길이를 $[0, 1)$ 범위로 압축하여 존재 확률을 표현한다.
  $$v_j = \text{squash}(s_j) = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \frac{s_j}{\|s_j\|}$$
- **Loss functions**:
  - **Margin Loss**: 클래스 존재 여부에 따라 정의되는 손실 함수이다.
    $$L_m = \sum_{k} \max(0, m^+ - \|v_k\|)^2 + \sum_{k} \mathbb{1}_{\{k \neq \text{class}\}} \max(0, \|v_k\| - m^-)^2$$
  - **Reconstruction Loss**: 캡슐 표현으로부터 원본 이미지를 복원하여 정규화(Regularization)하는 손실 함수이다.

### 2. 의료 영상 분할을 위한 특화 아키텍처

#### 2D-SegCaps

UNet 기반 구조로, 인코더와 디코더 경로 모두에 캡슐 블록을 배치한다.

- **Convolutional Capsule Encoder**: 모든 공간 위치에서 가중치 행렬 $W$를 공유함으로써 파라미터 수를 줄이고, 국소적인 영역(Kernel size) 내의 캡슐들만 상위 캡슐로 라우팅하도록 제한한다.
- **Deconvolutional Capsule Decoder**: 전치(Transpose) 연산을 통해 해상도를 복원한다.
- **Reconstruction Regularization**: 양성 클래스 픽셀만을 대상으로 MSE 손실을 계산하여 학습을 안정화한다.

#### 3D-SegCaps 및 3D-UCaps

3D-SegCaps는 2D-SegCaps를 볼륨 데이터로 확장하여 시간적/공간적 관계를 함께 고려한다. **3D-UCaps**는 여기서 한 단계 더 나아가, 인코더에는 표현력 좋은 캡슐 계층을 사용하고, 디코더에는 계산 비용이 낮고 세밀한 복원이 가능한 일반 Deconvolution 계층을 사용하는 하이브리드 구조를 제안한다.

#### SS-3DCapsNet (Self-supervised 3D CapsNet)

레이블 부족 문제를 해결하기 위해 자가 지도 학습(SSL)을 도입했다.

- **Pretext Task**: 원본 이미지에 노이즈 추가, 블러링, 채널 제거, 패치 스와핑 등의 변형을 가한 후, 이를 다시 복원하는 Task를 수행하여 네트워크가 데이터의 일반적인 특징을 학습하게 한다.
- **Downstream Task**: Pre-training 된 네트워크를 실제 의료 영상 분할 데이터로 미세 조정(Fine-tuning)한다.

## 📊 Results

### 실험 설정

- **데이터셋**: iSeg (영아 뇌 MRI), Cardiac (심장 MRI), Hippocampus (해마 MRI) 등 소규모 데이터셋을 사용하였다.
- **지표**: Dice Score를 주 지표로 사용하였다.
- **비교 대상**: 2D-SegCaps, 3D-SegCaps, 3D-UCaps, SS-3DCapsNet 및 일반 3D-UNet.

### 주요 결과

1. **정량적 성능**: iSeg 데이터셋에서 SS-3DCapsNet이 평균 Dice Score 92.39%로 가장 높은 성능을 보였으며, 3D-UCaps(92.08%)가 그 뒤를 이었다. Cardiac 및 Hippocampus 데이터셋에서도 SS-3DCapsNet이 가장 우수하거나 대등한 성능을 기록하였다.
2. **회전 등변성(Rotation Equivariance)**: 실험 결과, CapsNet 계열의 모델들이 일반 CNN(3D-UNet)보다 이미지 회전 시 성능 저하가 훨씬 적게 나타났다. 이는 캡슐이 포즈 정보를 내재적으로 학습하기 때문이다.
3. **강건성(Robustness)**: 다양한 아티팩트(Artifact)가 포함된 이미지에서도 CapsNet 기반 모델들이 CNN보다 더 안정적인 성능을 유지함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 CapsNet이 특히 **소규모 데이터셋** 환경에서 CNN보다 강력한 성능을 발휘함을 입증하였다. 이는 의료 영상 분야에서 가장 고질적인 문제인 '데이터 부족'을 해결할 수 있는 가능성을 보여준다.

**강점 및 기여:**

- 단순한 특징 검출을 넘어 파트-전체 관계를 모델링함으로써 아핀 변환에 대한 강건성을 확보하였다.
- 자가 지도 학습(SSL)을 캡슐 네트워크에 결합하여 레이블 의존도를 낮추는 효과적인 파이프라인을 제시하였다.

**한계 및 논의:**

- **계산 복잡도**: Dynamic Routing 과정에서의 반복 연산으로 인해 CNN 대비 학습 및 추론 시간이 상당히 길다.
- **성능 격차**: 소규모 데이터에서는 우수하지만, 대규모 데이터셋에서의 SOTA 모델들과의 격차를 완전히 좁혔는지는 추가 검증이 필요하다.
- **결론**: 저자는 향후 연구 방향으로 CapsNet의 표현력과 CNN의 효율성을 결합한 하이브리드 아키텍처 연구가 유망할 것이라고 제안한다.

## 📌 TL;DR

본 논문은 CNN이 가진 기하학적 정보 손실 및 데이터 의존성 문제를 해결하기 위해 **Capsule Network를 의료 영상 분할에 적용**한 연구이다. 특히 3D 볼륨 데이터 처리를 위한 **3D-UCaps**와 데이터 부족 문제를 극복하기 위한 자가 지도 학습 기반의 **SS-3DCapsNet**을 제안하였다. 실험 결과, 제안된 방법론은 적은 데이터로도 일반화 성능이 뛰어나며, 이미지 회전 및 변형에 대해 기존 CNN보다 훨씬 강건한 성능을 보임을 입증하였다. 이는 향후 고비용의 의료 데이터 레이블링 문제를 완화하고 정밀한 의료 영상 분석을 가능케 하는 중요한 기초 연구가 될 것으로 보인다.
