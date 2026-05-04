# CapsNet for Medical Image Segmentation

Minh Tran, Viet-Khoa Vo-Ho, Kyle Quinn, Hien Nguyen, Khoa Luu, and Ngan Le (2022)

## 🧩 Problem to Solve

본 연구는 의료 영상 분할(Medical Image Segmentation) 분야에서 기존 Convolutional Neural Networks(CNNs)가 가진 구조적 한계를 해결하고자 한다. CNN은 비정형 데이터에서 특징을 자동으로 추출하는 능력이 뛰어나지만, 다음과 같은 치명적인 문제점을 가지고 있다.

첫째, CNN은 회전(Rotation) 및 아핀 변환(Affine Transformation)에 매우 민감하다. 이는 CNN이 객체의 기하학적 관계를 무시하기 때문이며, 이를 극복하기 위해 방대한 양의 라벨링된 데이터셋을 통한 데이터 증강(Data Augmentation)에 의존해야 한다. 그러나 의료 데이터의 경우, 전문의의 어노테이션 비용이 매우 높고 엄격한 개인정보 보호 규정으로 인해 대규모 데이터 확보가 어렵다.

둘째, CNN의 풀링 레이어(Pooling Layer)는 공간적 정보(Positional Information)를 손실시키는 경향이 있어, 객체의 부분과 전체 사이의 계층적 관계(Part-whole relationship)를 보존하지 못한다. 결과적으로 입력 이미지의 방향이나 크기가 조금만 달라져도 성능이 급격히 저하되는 문제가 발생한다.

따라서 본 논문의 목표는 벡터 출력을 통해 객체의 속성과 공간적 관계를 보존하는 Capsule Network(CapsNet)를 의료 영상 분할 작업에 적용하고, 이를 2D 및 3D 볼륨 데이터로 확장하여 CNN 대비 강건성과 효율성을 분석하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 뉴런의 스칼라 출력(Scalar output)을 벡터 출력(Vector output), 즉 캡슐(Capsule)로 대체하여 객체의 '포즈(Pose)' 정보를 보존하는 것이다.

중심적인 설계 직관은 각 캡슐이 특정 시각적 엔티티(Visual entity)를 학습하고, 이들의 관계를 **Dynamic Routing**이라는 메커니즘을 통해 상위 계층으로 전달함으로써, 단순한 특징의 존재 여부가 아닌 객체의 기하학적 구조와 상대적 위치 관계를 학습하도록 하는 것이다. 이를 통해 데이터 효율성을 높이고, 변형된 입력에 대해 더 높은 강건성(Robustness)을 확보하고자 한다.

## 📎 Related Works

논문에서는 CapsNet의 성능을 개선하기 위한 기존 연구들을 두 가지 방향으로 분류하여 설명한다.

1. **라우팅 메커니즘 개선**: 기존의 Dynamic Routing의 높은 계산 복잡도를 줄이기 위해 Expectation-Maximization(EM) Routing, Straight-through attentive routing, Consistent dynamic routing 등이 제안되었다. 또한, 벡터 대신 행렬이나 텐서를 사용하여 엔티티를 표현함으로써 파라미터 수를 줄이려는 시도가 있었다.
2. **네트워크 아키텍처 확장**: Convolutional 레이어와 Capsule 레이어를 결합한 하이브리드 구조, Unsupervised capsule autoencoder, Aff-CapsNets, Memory-augmented CapsNet 등이 연구되었다. 특히 DECAPS는 Inverted Dynamic Routing(IDR)을 통해 흉부 방사선 전문의 수준의 성능을 달성한 바 있다.

기존 CNN 기반 방식은 데이터 증강을 통해 불변성(Invariance)을 억지로 학습시키려 하지만, CapsNet은 학습된 가중치 내에 불변하는 부분-전체 공간 관계를 직접 인코딩함으로써 근본적으로 다른 접근 방식을 취한다.

## 🛠️ Methodology

### 1. CapsNet의 기본 원리

CapsNet은 특징 추출을 위해 다음과 같은 구성 요소를 사용한다.

- **Non-shared transformation module**: 하위 캡슐의 출력을 변환 행렬과 곱하여 상위 캡슐에 대한 '투표(Vote)' 벡터를 생성한다.
- **Dynamic Routing**: 일종의 합의 메커니즘으로, 하위 캡슐의 예측 벡터와 상위 캡슐의 실제 출력 벡터 간의 유사성(Agreement)을 측정하여 가중치를 업데이트한다.
- **Squashing Function**: 캡슐 벡터의 길이를 $[0, 1)$ 범위로 압축하여 해당 엔티티의 존재 확률로 해석하게 한다. 수식은 다음과 같다.
$$v_j = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \frac{s_j}{\|s_j\|}$$
- **Loss Functions**: 예측값과 정답 간의 거리를 측정하는 **Margin Loss**와, 학습된 캡슐 표현으로부터 원본 이미지를 복원하여 정규화하는 **Reconstruction Loss**를 함께 사용한다.

### 2. 의료 영상 분할을 위한 아키텍처 확장

#### 2D-SegCaps

UNet 구조를 기반으로 인코더와 디코더 경로 모두에 캡슐 블록을 배치한 아키텍처이다.

- **Convolutional Capsule Encoder**: 모든 공간 위치에서 변환 행렬을 공유함으로써 파라미터 수를 줄이며, 국소적인 커널 범위 내에서만 라우팅을 수행하여 계산 복잡도를 낮췄다.
- **Deconvolutional Capsule Decoder**: 인코더의 역연산을 통해 세그멘테이션 맵을 생성한다.
- **손실 함수**: Margin Loss, Weighted Cross Entropy Loss, 그리고 MSE 기반의 Reconstruction Regularization Loss의 가중 합으로 정의된다.

#### 3D-SegCaps 및 3D-UCaps

- **3D-SegCaps**: 2D-SegCaps를 3D 볼륨 데이터(MRI, CT 등)로 확장하여 시간적/공간적 관계를 캡슐에 통합하였다.
- **3D-UCaps**: 3D-SegCaps의 계산 비용 문제를 해결하기 위해, 특징 추출(Encoder)에는 캡슐 레이어를 사용하고, 세밀한 복원이 필요한 디코더(Decoder) 경로에는 일반적인 Deconvolutional 레이어를 사용하는 하이브리드 구조를 채택하였다.

#### SS-3DCapsNet (Self-Supervised 3D CapsNet)

라벨링된 데이터 부족 문제를 해결하기 위해 자기지도학습(Self-supervised Learning)을 도입하였다.

- **Pretext Task**: 원본 이미지에 노이즈 추가, 블러링, 채널 삭제, 패치 스와핑 등 6가지 변형을 가한 후, 이를 다시 원본으로 복원하는 Task를 수행하여 네트워크가 데이터의 본질적인 표현(Representation)을 학습하게 한다.
- **Downstream Task**: 사전 학습된 네트워크를 실제 의료 영상 분할 작업에 맞게 미세 조정(Fine-tuning)한다.

## 📊 Results

### 실험 설정

- **데이터셋**: iSeg (영유아 뇌 MRI), Cardiac (심장 MRI), Hippocampus (해마 MRI) 등 소규모 의료 데이터셋을 사용하였다.
- **평가 지표**: Dice Score를 주요 지표로 사용하였다.
- **비교 대상**: 2D-SegCaps, 3D-SegCaps, 3D-UCaps, SS-3DCapsNet 및 일반 3D-UNet.

### 주요 결과

1. **정량적 성능**: iSeg 데이터셋에서 SS-3DCapsNet이 가장 높은 Average Dice Score(92.39%)를 기록하며 기존 CapsNet 기반 모델들을 소폭 상회하였다.
2. **데이터 효율성**: CapsNet 계열의 모델들은 특히 데이터셋의 크기가 작은 경우 CNN 기반 모델보다 우수한 성능을 보였다.
3. **강건성 분석**:
    - **회전 등변성(Rotation Equivariance)**: 3D-UCaps 등의 모델이 3D-UNet보다 회전 변형이 가해진 입력에 대해 더 안정적인 Dice Score를 유지함을 확인하였다.
    - **아티팩트 내성**: 다양한 이미지 아티팩트가 포함된 상황에서도 CapsNet 구조가 CNN보다 상대적으로 더 높은 성능을 유지하였다.

## 🧠 Insights & Discussion

본 논문은 CapsNet이 의료 영상 분할에서 가지는 명확한 강점과 한계를 동시에 제시한다.

**강점**:

- **적은 데이터로의 일반화**: 부분-전체 관계를 학습하는 특성 덕분에 대규모 데이터 증강 없이도 미지의 변형(Unseen variations)에 대해 더 잘 일반화한다.
- **해석 가능성**: 캡슐의 각 차원이 회전, 크기, 두께와 같은 구체적인 시각적 속성을 인코딩하므로 CNN보다 모델의 내부 동작을 해석하기 유리하다.
- **기하학적 강건성**: 아핀 변환 및 회전에 대해 CNN보다 구조적으로 더 강건한 특성을 보인다.

**한계 및 비판적 해석**:

- **계산 복잡도**: Dynamic Routing 과정에서의 반복적인 계산으로 인해 학습 및 추론 시간이 CNN보다 훨씬 오래 걸린다.
- **성능 격차**: 여전히 최신 SOTA CNN 기반 모델(예: 대규모 데이터로 학습된 모델)의 절대적인 성능에는 미치지 못하는 경우가 많다.
- **하이브리드 구조의 필요성**: 3D-UCaps의 결과에서 알 수 있듯이, 모든 레이어를 캡슐로 구성하기보다 인코더에는 캡슐을, 디코더에는 CNN 구조를 사용하는 것이 효율성과 성능 면에서 더 합리적일 수 있다.

## 📌 TL;DR

본 연구는 CNN의 공간 정보 손실 및 데이터 의존성 문제를 해결하기 위해 **Capsule Network를 의료 영상 분할에 적용**하고, 이를 3D 및 자기지도학습(SSL) 구조로 확장한 연구이다. 특히 **SS-3DCapsNet**을 통해 라벨링된 데이터가 부족한 의료 환경에서도 강건한 성능을 낼 수 있음을 입증하였다. 이 연구는 향후 계산 복잡도를 낮춘 **CapsNet-CNN 하이브리드 아키텍처**가 의료 영상 분석의 새로운 방향성이 될 수 있음을 시사한다.
