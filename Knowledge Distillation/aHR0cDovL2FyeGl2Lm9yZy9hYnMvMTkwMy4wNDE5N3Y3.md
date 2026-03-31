# Structured Knowledge Distillation for Dense Prediction

Yifan Liu, Changyong Shu, Jingdong Wang, Chunhua Shen (2020)

## 🧩 Problem to Solve

컴퓨터 비전 분야의 Dense Prediction(밀집 예측) task는 입력 이미지로부터 복잡한 출력 구조를 학습하는 근본적인 문제군을 의미한다. 여기에는 Semantic Segmentation, Depth Estimation, Object Detection 등이 포함된다. 이러한 task들은 각 픽셀에 범주 레이블을 할당하거나 특정 값을 회귀하여 구조화된 출력을 생성해야 하므로, 이미지 수준 예측 문제보다 훨씬 어렵다. 결과적으로 만족스러운 정확도를 달성하기 위해서는 대규모 네트워크가 요구된다.

그러나 동시에 제한된 연산 자원을 가진 엣지 디바이스에서 컴퓨팅을 가능하게 하기 위해 경량 모델이 필요하다. 기존의 Dense Prediction task를 위한 지식 증류(Knowledge Distillation) 전략은 이미지 분류를 위한 증류 방식을 직접 차용하여 각 픽셀에 대해 개별적으로 지식 증류를 수행하는 경우가 많았다. 이러한 픽셀 단위 증류 방식은 Dense Prediction의 중요한 특성인 '구조 정보(structure information)'를 무시하여 차선(sub-optimal)의 성능을 초래한다.

이 논문의 목표는 Dense Prediction이 본질적으로 구조화된 예측 문제라는 사실을 고려하여, 대규모 네트워크(교사 네트워크, teacher network)로부터 경량 네트워크(학생 네트워크, compact network)로 구조화된 지식(structured knowledge)을 효과적으로 전이하는 새로운 지식 증류 전략을 제안하는 것이다. 궁극적으로 경량 모델이 만족스러운 정확도를 유지하면서 제한된 연산 환경에서 동작할 수 있도록 돕는 것이 이 연구의 목표이다.

## ✨ Key Contributions

이 논문의 핵심 기여는 Dense Prediction task의 구조화된 특성을 고려하여, 대규모 네트워크에서 경량 네트워크로 구조 정보를 전이하는 새로운 지식 증류 전략을 제안하는 데 있다. 이는 기존 픽셀 단위 증류의 한계를 극복하고, 경량 모델의 성능을 크게 향상시킨다.

중심적인 직관 및 설계 아이디어는 다음과 같다.

- Dense Prediction은 구조화된 예측 문제이므로, 픽셀 단위 정보뿐만 아니라 픽셀 간의 관계 및 전체적인 구조 정보를 증류하는 것이 필수적이다.
- 이를 위해 두 가지 보완적인 구조화된 증류 방식을 제안한다.
  - **Pair-wise Distillation**: 공간적 위치 간의 쌍별(pair-wise) 유사성을 정적 어피니티 그래프(static affinity graph)를 구축하여 증류한다. 이는 짧은 범위 및 긴 범위의 구조 정보를 동시에 포착한다.
  - **Holistic Distillation**: 적대적 학습(adversarial training)을 사용하여 교사 및 학생 네트워크 출력 간의 고차(high-order) 및 전체적(holistic) 일관성을 증류한다. 이는 전체 출력 구조의 품질을 평가하는 판별자(discriminator)를 통해 이루어진다.

이러한 구조화된 지식 증류 방식을 통해 학생 네트워크는 교사 네트워크의 풍부한 구조적 이해를 효과적으로 학습하여, Semantic Segmentation, Depth Estimation, Object Detection과 같은 다양한 Dense Prediction task에서 경량 네트워크의 성능을 추가적인 추론 시간 및 파라미터 없이 크게 향상시킬 수 있다.

## 📎 Related Works

논문은 Dense Prediction의 세 가지 주요 분야(Semantic Segmentation, Depth Estimation, Object Detection) 및 Knowledge Distillation, Adversarial Learning과 관련된 기존 연구들을 소개하고, 제안하는 방식과의 차별점을 명시한다.

**Semantic Segmentation**:

- **기존 접근 방식**: 초기 FCNs [1] 이후, ResNet [21], DenseNet [22], HRNet과 같은 강력한 백본 네트워크, Dilated Convolutions [2], Multi-path Refine Networks (RefineNet) [4], Pyramid Pooling Modules (PSPNet) [3] 등을 통해 정확도 향상을 이루었다. Lin et al. [24]은 딥 모델과 구조화된 출력 학습을 결합하기도 했다.
- **경량 네트워크**: 실시간 및 모바일 애플리케이션을 위해 ENet [6], SQ [26], ESPNet [15], MobileNet [28], ShuffleNet [29], ICNet [7] 등 경량 네트워크 설계에 대한 연구가 활발히 진행되었다. 이들은 주로 컨볼루션 연산 가속화 기술(예: 인수분해 기법)을 활용한다.

**Depth Estimation**:

- **기존 접근 방식**: 단일 이미지에서 깊이 추정은 본질적으로 난해한 문제(ill-posed problem)이다. 초기에는 수제 특징(hand-crafted features)에 의존했으나, Eigen et al. [31] 이후 딥러닝 모델이 지배적이 되었다 [32], [33], [34], [35]. Fei [36]는 의미론적으로 정보화된 기하학적 손실을, Yin et al. [37]은 가상 법선(virtual normal) 손실을 도입하여 기하학적 정보를 활용했다.
- **경량 네트워크**: 효율적인 백본으로 인코더를 대체하여 계산 비용을 줄이는 시도가 있었으나, 경량 네트워크의 용량 한계로 인해 성능 저하를 겪는 경우가 많았다 [10], [37], [38].
- **차별점**: FastDepth [10]와 같은 밀접한 선행 연구는 경량 깊이 네트워크 훈련에 프루닝(pruning)을 적용했지만, 본 연구는 구조화된 지식 증류에 초점을 맞춘다.

**Object Detection**:

- **기존 접근 방식**: 초기 연구 [39], [40]는 제안(proposal) 예측 후 경계 상자(bounding box)의 위치를 개선하는 방식으로 좋은 성능을 달성했다. YOLO [8], SSD [9]와 같은 One-stage 방식은 효율성을 높였다. RetinaNet [41]은 Focal Loss를 제안하여 One-stage 방식의 성능을 Two-stage 방식과 유사하게 만들었다. 최근에는 Anchor-free 방식 (FCOS [5])이 주목받고 있으며, 이는 Object Detection을 Dense Prediction 문제로 해결한다.
- **차별점**: 본 연구는 FCOS 프레임워크에 구조화된 지식 증류를 적용한다.

**Knowledge Distillation**:

- **기존 접근 방식**: 힌튼(Hinton) et al. [18]이 제안한 지식 증류는 대규모 모델(교사)에서 경량 모델(학생)로 지식을 전이하여 경량 네트워크의 성능을 향상시킨다. 이미지 분류에서는 교사 모델의 클래스 확률(soft targets)을 사용하거나 중간 특징 맵(intermediate feature maps)을 전이한다 [19], [44].
- **Dense Prediction을 위한 지식 증류의 한계**:
  - MIMIC [11]: 객체 탐지 네트워크를 위해 픽셀 수준에서 특징 맵을 정렬했지만, 픽셀 간의 *구조 정보는 활용하지 않았다*.
  - Xie et al. [12]: Semantic Segmentation에 지식 증류를 적용하여 각 픽셀의 클래스 확률(본 논문의 픽셀 단위 증류와 유사)과 각 로컬 패치의 중앙-주변 차이(local relation)를 증류했다.
- **본 연구의 차별점**: Xie et al. [12]의 "local relation"은 본 논문의 Pair-wise Distillation의 특수한 경우로 볼 수 있다. 본 연구는 정적 어피니티 그래프를 구축하여 *다른 위치 간의 관계*를 전이하는 Pair-wise Distillation과 *고차 정보를 포착하는 전체적 지식(holistic knowledge)*을 전이하는 Holistic Distillation에 중점을 둔다. 이는 기존 방식들이 간과했던 중요하고 넓은 범위의 구조 정보를 효과적으로 증류한다.

**Adversarial Learning (GANs)**:

- **기존 접근 방식**: GANs [49]는 텍스트 및 이미지 생성에 널리 연구되었으며, Conditional GANs [51]는 이미지-투-이미지 변환(예: 스타일 전이 [52], 이미지 색칠 [53])에 성공적으로 적용되었다. 적대적 학습 아이디어는 포즈 추정 [54] 및 Semantic Segmentation [55]에도 활용되었다.
- **Semantic Segmentation에서의 GAN 한계**: Luc et al. [55]의 Semantic Segmentation 연구에서는 생성자의 연속적인 출력과 이산적인 실제 레이블 간의 불일치로 인해 판별자의 성공이 제한되는 문제가 있었다.
- **본 연구의 차별점**: 본 연구에서 사용된 GAN은 이러한 문제를 겪지 않는다. 판별자를 위한 "Ground Truth"는 교사 네트워크의 로짓(logits)이며, 이는 *실수 값(real valued)*이다. 따라서 생성자(학생 네트워크)의 연속적인 출력과의 불일치 문제가 발생하지 않는다. 본 연구는 적대적 학습을 통해 교사 네트워크와 경량 네트워크가 생성하는 출력 맵 간의 정렬을 촉진하며, Ground Truth로 계산되는 task loss는 선택 사항이므로 레이블 없는 데이터에도 적용 가능성이 있다.

## 🛠️ Methodology

이 논문은 Dense Prediction task를 위한 구조화된 지식 증류 방법을 제안한다. 전체 파이프라인은 기존의 픽셀 단위 증류와 함께 두 가지 새로운 구조화된 증류 방식(Pair-wise Distillation 및 Holistic Distillation)을 포함한다.

### 전체 파이프라인 및 시스템 구조

본 논문의 증류 프레임워크는 Figure 2에 요약되어 있다. 훈련 과정에서 대규모의 "교사 네트워크(Teacher Net, $T$)"는 고정된 상태로 유지된다. "학생 네트워크(Student Net, $S$)"는 경량 아키텍처를 가지며, 교사 네트워크의 지식을 증류하기 위해 최적화된다. Holistic Distillation을 위해 "판별자 네트워크(Discriminator Net, $D$)"도 함께 최적화된다. 학생 네트워크는 기존의 task-specific loss(예: Semantic Segmentation의 교차 엔트로피 손실)와 제안하는 세 가지 증류 항(term)을 결합한 목적 함수로 훈련된다.

### 각 주요 구성 요소 및 역할

1. **Pixel-wise Distillation (픽셀 단위 증류)**
    - **역할**: 이미지 분류에서 차용한 기본적인 지식 증류 전략이다. 각 픽셀에 대해 학생 네트워크의 클래스 확률 분포를 교사 네트워크의 클래스 확률 분포에 맞추도록 학습한다. 이는 Semantic Segmentation 문제를 개별 픽셀 레이블링 문제의 집합으로 간주한다.
    - **수학적 설명**: 교사 모델이 생성하는 클래스 확률을 "소프트 타겟(soft targets)"으로 사용하여 학생 네트워크를 훈련한다. 손실 함수는 다음과 같다.
        $$ L_{pi}(S) = \frac{1}{W' \times H'} \sum_{i \in R} KL(q_s^i \Vert q_t^i) $$
        여기서 $q_s^i$는 학생 네트워크 $S$가 생성한 $i$-번째 픽셀의 클래스 확률을, $q_t^i$는 교사 네트워크 $T$가 생성한 $i$-번째 픽셀의 클래스 확률을 나타낸다. $KL(\cdot)$은 두 확률 분포 간의 Kullback-Leibler (KL) Divergence이며, $R=\{1,2,...,W' \times H'\}$는 모든 픽셀을 나타낸다. $W'$와 $H'$는 특징 맵의 공간 해상도이다.

2. **Structured Knowledge Distillation (구조화된 지식 증류)**
    기존 픽셀 단위 증류와 더불어, 구조화된 지식을 전이하기 위한 두 가지 방식이 제안된다.

    a.  **Pair-wise Distillation (쌍별 증류)**
        -   **역할**: 공간적 레이블링의 연속성을 강화하는 데 널리 사용되는 Markov Random Field 프레임워크에서 영감을 받았다. 공간적 위치 간의 쌍별 관계, 특히 쌍별 유사성(pair-wise similarities)을 전이한다.
        -   **어피니티 그래프 구축**: 공간적 쌍별 관계를 나타내기 위해 정적 어피니티 그래프를 구축한다.
            -   **노드(Nodes)**: 각 노드는 공간적 위치를 나타낸다. 논문은 각 노드의 "세분성(granularity)" $\beta$와 "연결 범위(connection range)" $\alpha$를 제어한다. Figure 3에 제시된 바와 같이, $\beta \times \beta$ 크기의 공간적 로컬 패치 내 픽셀들을 평균 풀링(average pooling)하여 노드의 특징 $1 \times C$를 집계한다.
            -   **연결(Connections)**: 두 노드 사이의 연결은 유사성을 나타낸다. 각 노드에 대해 공간적 거리(여기서는 Chebyshev 거리 사용)에 따라 상위 $\alpha$개의 가까운 노드와의 유사성만 고려한다.
            -   **유사성 계산**: 두 노드 $i$와 $j$의 집계된 특징 $f_i$와 $f_j$로부터 유사성 $a_{ij}$는 다음과 같이 계산된다.
                $$ a_{ij} = f_i^T f_j / (\Vert f_i \Vert_2 \Vert f_j \Vert_2) $$
        -   **수학적 설명**: 학생 네트워크 $S$와 교사 네트워크 $T$가 생성한 $i$-번째 노드와 $j$-번째 노드 사이의 유사성 $a_s^{ij}$ 및 $a_t^{ij}$ 간의 제곱 차이(squared difference)를 최소화한다. 손실 함수는 다음과 같다.
            $$ L_{pa}(S) = \frac{\beta}{W' \times H' \times \alpha} \sum_{i \in R'} \sum_{j \in \alpha} (a_s^{ij} - a_t^{ij})^2 $$
            여기서 $R'=\{1,2,...,\frac{W' \times H'}{\beta^2}\}$ (단, 논문 텍스트에는 $\frac{W' \times H'}{\beta}$로 표기되어 있으나, $\beta \times \beta$ 패치 집계 및 Table 2의 실험 결과로 미루어 $\beta^2$가 더 합리적이다.)는 모든 노드를 나타낸다.

    b.  **Holistic Distillation (전체적 증류)**
        -   **역할**: 픽셀 단위 및 쌍별 증류에서 포착되지 않는 고차(higher-order) 일관성을 정렬하는 것을 목표로 한다. 조건부 생성적 적대 신경망(Conditional Generative Adversarial Learning, cGAN) [51]을 활용한다.
        -   **학생 네트워크 ($S$)**: 입력 RGB 이미지 $I$에 조건화된 생성자(generator)로 동작하여 "가짜(fake)" 세그멘테이션 맵 $Q_s$를 예측한다.
        -   **교사 네트워크 ($T$)**: 교사가 예측한 세그멘테이션 맵 $Q_t$는 "실제(real)" 샘플로 간주된다. 학생 네트워크는 $Q_t$와 유사한 $Q_s$를 생성하도록 장려된다.
        -   **판별자 네트워크 ($D$)**: GAN의 판별자로 작동하는 임베딩 네트워크이다. 입력 이미지 $I$와 세그멘테이션 맵 $Q$ (학생 또는 교사의 출력)를 함께 받아 전체적인 임베딩 점수(holistic embedding score)를 생성한다.
            -   **아키텍처**: 5개의 컨볼루션 블록(ReLU, BN 포함, 마지막 블록 제외)으로 구성된 완전 컨볼루션 네트워크(FCN)이다. 마지막 세 블록 사이에 두 개의 Self-attention 모듈 [58]이 삽입되어 구조 정보를 포착한다. 입력 직후 배치 정규화(Batch Normalization, BN) 층을 추가하여 세그멘테이션 맵과 RGB 채널 간의 스케일 차이를 정규화한다.
            -   **역할**: $D$는 교사로부터 나온 출력 맵에 대해 높은 점수를, 학생으로부터 나온 출력 맵에 대해 낮은 점수를 생성하도록 훈련된다. 이를 통해 $D$는 세그멘테이션 맵의 품질을 평가하는 지식을 인코딩하게 되며, 학생은 $D$의 평가 하에 더 높은 점수를 얻도록 학습되어 성능이 향상된다.
        -   **Wasserstein Distance**: 두 분포가 겹치지 않을 때 불안정한 기울기 문제가 발생하는 KL 또는 JS Divergence 대신, 두 분포($p_s(Q_s)$와 $p_t(Q_t)$) 간의 차이를 측정하는 부드러운 척도인 Wasserstein Distance [57]를 사용한다. Lipschitz 제약은 기울기 패널티(gradient penalty) [57]를 적용하여 강제된다.
        -   **수학적 설명**: 판별자를 위한 Wasserstein Distance 목적 함수는 다음과 같다.
            $$ L_{ho}(S,D) = E_{Q_s \sim p_s(Q_s)}[D(Q_s|I)] - E_{Q_t \sim p_t(Q_t)}[D(Q_t|I)] $$
            여기서 $E[\cdot]$은 기댓값 연산자이며, $D(\cdot)$는 판별자 네트워크이다.

### 훈련 목표, 손실 함수 및 추론 절차

**전체 목적 함수**: 학생 네트워크 $S$와 판별자 네트워크 $D$를 위한 전체 목적 함수는 표준 다중 클래스 교차 엔트로피 손실 $L_{mc}(S)$와 증류 항들의 합으로 구성된다.
$$ L(S,D) = L_{mc}(S) + \lambda_1 (L_{pi}(S) + L_{pa}(S)) - \lambda_2 L_{ho}(S,D) $$
여기서 $\lambda_1$과 $\lambda_2$는 각각 $10$과 $0.1$로 설정되어 손실 값의 범위를 비교 가능하게 만든다.

**학습 절차**: 목적 함수는 $S$의 파라미터에 대해서는 최소화하고, $D$의 파라미터에 대해서는 최대화하는 방식으로 다음 두 단계를 반복하여 구현된다.

1. **판별자 $D$ 훈련**: $L_{ho}(S,D)$를 $D$의 파라미터에 대해 최소화한다. $D$는 교사 네트워크로부터의 실제 샘플($Q_t$)에는 높은 임베딩 점수를, 학생 네트워크로부터의 가짜 샘플($Q_s$)에는 낮은 임베딩 점수를 부여하는 것을 목표로 한다.
2. **경량 Segmentation 네트워크 $S$ 훈련**: 판별자 $D$가 고정된 상태에서, $S$는 다음과 같은 손실 함수를 최소화한다.
    $$ L_{mc}(S) + \lambda_1 (L_{pi}(S) + L_{pa}(S)) - \lambda_2 L_{s_{ho}}(S) $$
    여기서 $L_{s_{ho}}(S) = E_{Q_s \sim p_s(Q_s)}[D(Q_s|I)]$ 이다. 학생 $S$는 $D$의 평가 하에 더 높은 점수를 얻는 것을 목표로 하여, 교사 네트워크의 전체적 출력 품질을 모방하게 된다.

### 기타 Dense Prediction Task로의 확장

- **Object Detection**: FCOS [5] 프레임워크를 따라 각 픽셀에 대해 클래스 $c^*$와 경계 상자 위치를 나타내는 4D 벡터 $t^*=(l,t,r,b)$를 예측한다. 증류 항은 정규화(regularization)로 사용된다.
- **Depth Estimation**: 연속적인 깊이 값을 $C$개의 이산적인 범주로 나누어 분류 문제로 해결한다 [59]. 추론 시에는 [37]과 같이 소프트 가중 합(soft weighted sum)을 적용한다.
  - Pair-wise Distillation은 중간 특징 맵에 직접 적용될 수 있다.
  - Holistic Distillation은 깊이 맵을 입력으로 사용한다. 깊이 맵은 연속적인 맵이므로 Ground Truth를 GAN의 실제 샘플로 사용할 수도 있지만, 레이블 없는 데이터에 방법론을 적용하기 위해 교사 네트워크의 깊이 맵을 실제 샘플로 사용한다. Depth Estimation task에서는 픽셀 단위 증류가 정확도를 거의 향상시키지 못하는 것으로 나타나, 구조화된 지식 증류만 사용된다.

## 📊 Results

이 논문은 Semantic Segmentation, Depth Estimation, Object Detection의 세 가지 Dense Prediction task에서 제안하는 Structured Knowledge Distillation 방법의 효과를 검증한다.

### Semantic Segmentation

**데이터셋, 작업, 기준선, 지표**:

- **데이터셋**: Cityscapes [60], CamVid [61], ADE20K [62].
- **작업**: 픽셀 단위 클래스 분류.
- **기준선**: ESPNet-C [15], ESPNet [15], ResNet18(0.5), ResNet18(1.0), MobileNetV2Plus [16] 등 다양한 경량 네트워크.
- **교사 네트워크**: ResNet101 백본을 사용하는 PSPNet [3].
- **지표**: mIoU (mean Intersection over Union), class IoU, Pixel Accuracy, 모델 크기 (#Params), 연산 복잡도 (FLOPs).

**주요 정량적 또는 정성적 결과**:

1. **증류 기법별 효과 (Table 1)**:
    - Pixel-wise (PI), Pair-wise (PA), Holistic (HO) 증류 기법이 학생 네트워크의 성능을 점진적으로 향상시킨다.
    - ImageNet 사전 학습이 없는 작은 ResNet18(0.5)의 경우, 모든 증류 기법 적용 시 mIoU가 55.37%에서 62.35%로 **6.26%p** 크게 향상되었다.
    - ImageNet 사전 학습이 있는 ResNet18(1.0)의 경우에도 mIoU가 71.10%에서 74.08%로 **2.9%p** 향상되었다. 증류의 효과는 더 작은 네트워크와 사전 학습이 없는 네트워크에서 더 두드러진다.

2. **Pair-wise Distillation의 어피니티 그래프 분석 (Table 2)**:
    - 연결 범위($\alpha$)를 늘리면 증류 성능이 향상되며, 완전히 연결된 그래프(fully connected graph)가 가장 좋은 성능(71.37% mIoU)을 보였다.
    - 노드 세분성($\beta$): $\beta=2 \times 2$ 패치를 사용하여 노드를 구성하는 것이 픽셀 단위 노드($\beta=1 \times 1$)보다 약간 더 나은 성능(71.78% mIoU vs 71.37% mIoU)을 보이면서 연결 수를 크게 줄일 수 있어 효율성과 정확도 간의 좋은 균형점을 제공한다. 이는 작은 로컬 패치로 노드를 정의하는 것이 다른 위치 간의 더 안정적인 상관관계를 형성할 수 있음을 시사한다.
    - 다중 레벨의 어피니티 그래프를 융합하는 전략은 추가적인 계산 비용으로 인해 미미한 성능 향상만 보였다.

3. **Holistic Distillation의 적대적 학습 분석 (Table 3, 4, 5, Fig. 5, 6)**:
    - 판별자 아키텍처: Self-attention 레이어를 추가하면 mIoU가 향상된다. 2개의 Self-attention 블록(A2L4)이 성능, 안정성, 계산 비용 측면에서 적절한 균형을 제공한다. 판별자가 깊을수록(더 많은 컨볼루션 블록) 적대적 학습에 도움이 된다.
    - Self-attention 레이어는 판별자가 구조를 더 잘 포착하도록 돕고, 이는 학생 네트워크가 트럭, 버스, 기차, 오토바이와 같은 구조화된 객체(structured objects)의 정확도를 크게 향상시킨다(Table 4, Fig. 6).
    - 잘 훈련된 판별자가 평가한 결과, Holistic Distillation을 적용한 학생 네트워크의 출력 맵은 교사 네트워크와 유사한 임베딩 점수 분포를 보였으며, 이는 GAN이 전체적 구조 지식 증류에 효과적임을 나타낸다(Table 5, Fig. 5).

4. **다른 증류 방법과의 비교 (Table 6)**:
    - 본 논문의 Pair-wise Distillation (PA)은 MIMIC [11], Attention Transfer [44]와 같은 특징 기반 증류 방법과 Xie et al. [12]의 Local Pair-wise Distillation보다 우수한 성능을 보인다. 이는 개별 픽셀의 특징 정렬을 넘어 구조화된 지식을 전이하고, 전체적인 구조 정보를 포착하는 완전히 연결된 쌍별 증류의 효과를 입증한다.

5. **최종 Semantic Segmentation 결과 (Table 7, 8, 9, Fig. 7, 8, 9, 10, 11)**:
    - **Cityscapes**: ESPNet-C, ESPNet, ResNet18(0.5), ResNet18(1.0), MobileNetV2Plus 등 여러 경량 네트워크에서 mIoU가 크게 향상된다. 특히 사전 학습이 없는 네트워크(ResNet18(0.5), ESPNet-C)에서 **7.3%p, 6.6%p**와 같은 매우 큰 성능 향상을 보인다. MobileNetV2Plus를 사용했을 때, MD (Enhanced) [12]보다 높은 세그멘테이션 품질(74.5% vs 71.9% on validation)을 달성한다. 구조화된 증류는 버스(+17.23%p), 트럭(+10.03%p)과 같은 구조화된 객체 클래스에서 특히 유의미한 개선을 가져왔다(Figure 7, 8).
    - **CamVid**: 학생 네트워크의 mIoU가 향상되었고, 추가적인 레이블 없는 데이터(Cityscapes에서 샘플링)를 활용했을 때 ESPNet-C 및 ESPNet에서 mIoU가 각각 **13.5%p, 12.6%p** 크게 향상되었다(Table 8, Figure 10). 이는 레이블 없는 데이터에서도 증류 방식이 지식을 전이할 수 있음을 보여준다.
    - **ADE20K**: ESPNet에서 mIoU가 **3.78%p** 향상되었고, MobileNetV2와 ResNet18에서도 각각 **3.74%p, 2.78%p** mIoU가 향상되었다(Table 9, Figure 11).

### Depth Estimation

**데이터셋, 작업, 기준선, 지표**:

- **데이터셋**: NYUD-V2 [60].
- **작업**: 단안 깊이 추정.
- **기준선**: MobileNetV2 백본을 사용하는 [37] 모델.
- **교사 네트워크**: ResNext101 백본을 사용하는 [37] 모델.
- **지표**: 평균 절대 상대 오차 (rel), 평균 log10 오차 (log10), RMSE (rms), 임계값 정확도 ($\delta_i < 1.25^i$).

**주요 정량적 또는 정성적 결과**:

1. **증류 기법별 효과 (Table 10)**:
    - 깊이 맵은 실수 값이므로, Semantic Segmentation과 달리 픽셀 단위 증류는 깊이 추정에서 정확도를 거의 향상시키지 못했다. 따라서 Depth Estimation에서는 Structured Knowledge Distillation만 사용되었다.
    - 구조화된 지식 증류(PA+HO)는 baseline의 rel 오차를 0.181에서 0.173으로 감소시켰다.
    - 추가적인 레이블 없는 데이터(NYUD-V2의 비디오 시퀀스에서 샘플링된 30K 이미지)를 사용했을 때 rel 오차는 0.160으로 더 감소하여, 레이블 없는 데이터 활용 가능성을 확인시켜준다.

2. **최종 Depth Estimation 결과 (Table 11)**:
    - MobileNetV2 백본을 사용하는 강한 기준선(baseline) 모델의 rel 오차를 0.135에서 0.130으로 개선했다. 이는 모든 평가 지표(rel, log10, rms, $\delta_1, \delta_2, \delta_3$)에서 성능 향상을 보여주며, 실시간 깊이 모델에 대한 구조화된 지식 증류의 효과를 입증한다.

### Object Detection

**데이터셋, 작업, 기준선, 지표**:

- **데이터셋**: COCO [70] (minival, test-dev).
- **작업**: 객체 경계 상자 예측 및 클래스 분류.
- **기준선**: MobileNetV2 백본을 사용하는 FCOS [5] 기반 모델 (c128-MNV2, c256-MNV2).
- **교사 네트워크**: ResNeXt-32x8d-101-FPN 백본을 사용하는 FCOS [5].
- **지표**: mAP (average precision over multiple IoU thresholds), AP50, AP75, APs (small), APm (medium), APl (large).

**주요 정량적 또는 정성적 결과**:

1. **다른 증류 방법과의 비교 (Table 12)**:
    - MIMIC [11] (픽셀 수준 특징 정렬)은 baseline mAP를 0.4%p 향상시켰다.
    - 본 논문의 Pair-wise Distillation (PA)만으로 mAP를 0.9%p 향상시켰다.
    - 모든 증류 항(PI+PA+HO)을 결합했을 때 mAP는 32.1%로 **1.1%p** 가장 크게 향상되었다.
    - 특히 AP75, APs (small objects), APl (large objects) 지표에서 더욱 큰 개선을 보여, 증류 방법이 작은 객체 및 높은 IoU 임계값에서의 감지 정확도에 효과적임을 나타낸다.

2. **최종 Object Detection 결과 (Table 13, 14, Fig. 12)**:
    - c128-MNV2 모델의 mAP는 **0.9%p** 향상되었고, c256-MNV2 모델의 mAP는 **0.8%p** 향상되었다.
    - COCO test-dev 데이터셋에서 MobileNetV2-FPN 학생 네트워크는 Structured Knowledge Distillation을 통해 FCOS baseline (31.4% AP)보다 높은 **34.1% AP**를 달성했다.
    - 이러한 성능 향상은 추론 시간에 추가적인 비용 없이 이루어져, 경량 모델의 효율성을 유지하면서 정확도를 높일 수 있음을 보여준다.
    - 정성적 결과(Figure 12)에서는 본 증류 방법을 적용한 검출기가 '사람'이나 '새'와 같은 작고 가려진(occluded) 객체를 더 잘 탐지하는 것을 볼 수 있다.

## 🧠 Insights & Discussion

**논문에서 뒷받침되는 강점**:

- **구조화된 지식 증류의 중요성**: 논문은 Dense Prediction task에서 픽셀 단위 정보뿐만 아니라 픽셀 간의 구조적 관계를 증류하는 것이 필수적임을 설득력 있게 보여준다. 이는 기존 지식 증류 방법의 한계를 명확히 지적하고 이를 극복하는 새로운 방향을 제시한다.
- **보완적인 증류 기법**: Pair-wise Distillation과 Holistic Distillation이라는 두 가지 구조화된 증류 기법은 서로 다른 수준의 구조 정보를 포착하여 상호 보완적으로 작동한다. Pair-wise 증류는 지역적 및 장거리 공간 관계를, Holistic 증류는 고차원적이고 전체적인 특징을 효과적으로 전이한다.
- **광범위한 적용 가능성**: 제안된 방법은 Semantic Segmentation, Depth Estimation, Object Detection이라는 세 가지 주요 Dense Prediction task에서 모두 강력한 성능 향상을 입증했다. 이는 방법론의 일반성과 견고성을 보여주는 중요한 증거이다.
- **효율성 유지**: 경량 학생 네트워크의 성능을 크게 향상시키면서도 추론 시에는 추가적인 연산 비용(FLOPs)이나 파라미터(Params)를 요구하지 않는다. 이는 모바일 또는 엣지 디바이스와 같은 자원 제약적인 환경에 딥러닝 모델을 배포하는 데 매우 실용적인 이점이다.
- **레이블 없는 데이터 활용**: 특히 Holistic Distillation은 레이블이 없는 데이터에서도 교사 네트워크의 지식을 전이할 수 있음을 보여주었다. 이는 고비용의 데이터 라벨링 과정을 완화하고, 대규모 레이블 없는 데이터셋을 활용하여 모델을 더욱 강화할 수 있는 가능성을 제시한다.
- **심층적인 분석 및 검증**: 어피니티 그래프의 파라미터($\alpha, \beta$) 변화, 판별자 아키텍처의 영향, 그리고 기존 증류 방법들과의 비교 등 포괄적인 ablation study를 통해 제안된 기법들의 효과를 명확하게 분석하고 검증한다.

**한계, 가정 또는 미해결 질문**:

- **정적 어피니티 그래프의 유연성**: Pair-wise Distillation에서 사용되는 어피니티 그래프는 정적(static)으로 구축된다. 실제 이미지 콘텐츠에 따라 동적으로 변화하는 관계를 더 잘 포착할 수 있는 유연한 그래프 구조가 잠재적으로 더 나은 성능을 가져올 수 있을지에 대한 질문이 남는다.
- **교사 네트워크의 의존성**: 지식 증류의 본질적인 한계로, 학생 네트워크의 성능은 교사 네트워크의 품질에 크게 의존한다. 만약 교사 네트워크 자체에 구조적 이해의 한계가 있다면, 이러한 한계가 학생 네트워크로 전파될 수 있다. 논문은 잘 훈련된 교사 네트워크가 존재함을 가정한다.
- **훈련 시의 계산 복잡성**: 추론 시 추가 비용이 없다는 것은 큰 장점이지만, 적대적 학습(Holistic Distillation)과 어피니티 그래프 구축(Pair-wise Distillation)은 훈련 과정의 계산 복잡도를 증가시킨다. 훈련 효율성을 추가적으로 개선할 수 있는 방안에 대한 논의는 제한적이다.
- **하이퍼파라미터 민감도**: 손실 함수에서 증류 항들의 가중치($\lambda_1, \lambda_2$)는 경험적으로 설정되었다. 이러한 하이퍼파라미터가 다른 Dense Prediction task나 다양한 학생/교사 네트워크 아키텍처에 대해 얼마나 민감하게 작용하는지에 대한 심층적인 분석은 부족하다.
- **판별자 디자인의 일반화**: 판별자 아키텍처(예: Self-attention 레이어 수)는 Semantic Segmentation task에서 최적화되었다. 이 디자인이 Depth Estimation이나 Object Detection과 같은 다른 Dense Prediction task에서도 최적으로 작동하는지, 아니면 각 task에 맞게 추가적인 미세 조정이 필요한지에 대한 일반화 문제는 여전히 존재한다.

**논문에 근거한 간략한 비판적 해석 및 논의사항**:
이 논문은 Dense Prediction의 구조적 특성을 지식 증류 과정에 성공적으로 통합한 선구적인 연구이다. 기존 픽셀 단위 증류의 한계를 명확히 짚어내고, 이를 쌍별 관계 및 전체적 일관성 전이를 통해 극복함으로써 경량 모델의 성능을 비약적으로 향상시켰다. 특히 Self-attention을 활용한 판별자가 구조화된 객체 인식에 결정적인 기여를 한다는 점과, 레이블 없는 데이터를 효과적으로 활용할 수 있다는 점은 이 연구의 실제 적용 가능성과 확장성을 높이는 중요한 발견이다. 비록 훈련 시 계산 비용 증가와 정적 그래프 구조의 유연성 부족이라는 한계가 있지만, 추론 시 비용 증가 없이 성능을 높였다는 점은 매우 가치 있는 기여이다. 이 연구는 효율적인 딥러닝 모델 개발 방향에 중요한 이정표를 제시하며, 향후 더 정교한 동적 구조 모델링이나 훈련 효율성 개선을 위한 연구의 발판이 될 것으로 기대된다.

## 📌 TL;DR

이 논문은 Dense Prediction task(Semantic Segmentation, Depth Estimation, Object Detection)를 위한 경량 네트워크 훈련 시 "구조화된 지식 증류(Structured Knowledge Distillation)" 방법을 제안한다. 기존 픽셀 단위 지식 증류가 Dense Prediction의 구조적 특성을 간과하여 성능이 제한적이라는 문제점을 해결하기 위해, 논문은 두 가지 핵심 증류 방식을 도입한다. 첫째, **Pair-wise Distillation**은 공간적 위치 간의 쌍별 유사성을 정적 어피니티 그래프를 구축하여 교사 네트워크로부터 학생 네트워크로 전이한다. 둘째, **Holistic Distillation**은 조건부 적대적 학습을 활용하여 교사와 학생 네트워크의 출력 구조 간의 고차적이고 전체적인 일관성을 정렬한다. 이러한 구조화된 증류 방식은 픽셀 단위 증류와 결합되어 학생 네트워크의 학습을 보조한다. 실험 결과, 이 방법은 Semantic Segmentation, Depth Estimation, Object Detection의 세 가지 task에서 다양한 경량 네트워크의 성능을 추론 시 추가적인 FLOPs나 파라미터 없이 크게 향상시켰으며 (예: Semantic Segmentation에서 최대 7.3%p mIoU 개선), 특히 구조화된 객체 인식에 효과적이고 레이블 없는 데이터도 활용할 수 있음을 입증했다. 이 연구는 경량 Dense Prediction 모델의 정확도와 효율성을 동시에 높이는 중요한 길을 제시한다.
