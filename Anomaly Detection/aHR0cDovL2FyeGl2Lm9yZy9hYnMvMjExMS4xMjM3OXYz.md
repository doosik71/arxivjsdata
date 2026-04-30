# Efficient Anomaly Detection Using Self-Supervised Multi-Cue Tasks

Loïc Jézéquel, Ngoc-Son Vu, Jean Beaudet, and Aymeric Histace (2022)

## 🧩 Problem to Solve

본 논문은 이미지 데이터에서 정상을 정의하고 그 외의 데이터를 이상치로 판별하는 Anomaly Detection (AD) 문제를 다룬다. 특히, 기존의 Self-Supervised Learning (SSL) 기반 AD 방법론들이 가진 다음과 같은 세 가지 주요 한계점을 해결하고자 한다.

첫째, 기존 방법들은 기하학적 변환(geometric transformations)에 의존하여 세밀한 특징(finer features)을 포착하지 못하며, 이로 인해 세밀한 차이를 구분해야 하는 Fine-grained 문제에서 성능이 떨어진다. 둘째, 이상치의 종류(anomaly type)에 따라 성능 편차가 심해 범용성이 부족하다. 셋째, 추론 단계에서 수많은 변환을 적용해야 하므로 추론 시간이 매우 길어 실시간 적용이 어렵다는 점이다.

따라서 본 연구의 목표는 구조(Structure), 색상(Colorimetry), 질감(Texture)이라는 세 가지 상보적인 시각적 단서(visual cues)를 활용하는 효율적인 SSL 기반 AD 프레임워크를 설계하여, 정밀한 이상치 탐지 성능을 확보하고 추론 효율성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 시각적 특성을 학습하는 세 가지 보조 작업(pretext tasks)을 동시에 해결함으로써, 이미지의 풍부한 표현(representation)을 학습하는 것이다.

1. **Multi-Cue 학습 체계**: 구조적 단서를 위한 Piece-wise jigsaw puzzle, 색상 정보를 위한 Tint rotation, 질감 정보를 위한 Partial re-colorization 작업을 도입하여 상보적인 특징을 추출한다.
2. **객체 중심의 학습 (Attention Mechanism)**: 배경의 다양성으로 인해 발생하는 노이즈를 줄이기 위해, Attention mechanism을 도입하여 객체가 포함된 영역에 더 집중하여 학습하도록 설계하였다.
3. **GMM 기반 색상 복원**: 단순 회귀 방식의 색상 복원이 가진 다봉성(multi-modality) 문제를 해결하기 위해 Gaussian Mixture Model (GMM)과 Expectation-Maximization (EM) 알고리즘을 적용하여 정밀한 질감 및 색상 표현을 학습한다.
4. **효율적인 추론 및 융합**: 다수의 변환을 적용하던 기존 방식과 달리, 최적화된 OOD (Out-of-Distribution) 탐지 함수와 Median 기반의 score fusion을 통해 추론 속도를 높이고 안정성을 확보하였다.

## 📎 Related Works

기존의 AD 방법론은 크게 고전적 방법, 딥러닝 기반 방법, SSL 기반 방법으로 나뉜다.

- **고전적 방법**: OC-SVM, SVDD, Isolation Forest 등이 있으며, 주로 저차원 데이터에서 효과적이지만 고차원 이미지 데이터에서는 한계가 명확하다.
- **딥러닝 기반 방법**: OC-NN과 같은 One-class Neural Network나 GAN을 이용한 재구성 오차(reconstruction error) 기반 방법들이 제안되었다. 하지만 이들은 이상치 클래스의 경계가 불분명한 경우 성능이 저하되는 경향이 있다.
- **SSL 기반 AD**: GeoTrans나 MHRot과 같이 이미지 회전이나 기하학적 변환을 분류하는 작업을 통해 정상 데이터를 학습한다. 그러나 GeoTrans의 경우 추론 시 72개의 변환을 적용해야 하므로 매우 느리며, 세밀한(fine-grained) 이상치 탐지 능력이 부족하다는 한계가 있다.

본 논문은 이러한 기존 SSL 방식에서 한 걸음 나아가, 단순한 기하학적 변환을 넘어 색상과 질감이라는 다각도 단서를 통합하고, 추론 효율성을 극대화했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

전체 시스템은 판별적(Discriminative) 작업과 생성적(Generative) 작업으로 구성된 두 개의 브랜치(U-branch, L-branch) 구조를 가진다.

### 1. Discriminative Branch (U-branch)
이 브랜치는 하나의 공유 Encoder $\phi$를 사용하며, 두 가지 작업을 수행한다.

**A. Piece-wise Puzzle Task**
기존의 전체 퍼즐 정답(permutation index)을 맞추는 방식 대신, 각 조각(piece)의 원래 위치를 예측하는 방식으로 개선하였다.
- **학습**: 이미지를 $n$개의 조각으로 나누고 셔플한 뒤, 각 조각 $i$에 대해 원래 위치 $\Pi^i$를 예측한다. 손실 함수로는 Cross-Entropy loss $L_{CE}$를 사용한다.
$$L_{pzl}(\Pi(I)) = \frac{1}{n} \sum_{i=1}^{n} L_{CE}(\phi \circ f_i(\Pi(I)); \Pi^i)$$
- **추론**: 각 조각의 예측 결과에 대해 OOD score를 계산하고 그 평균을 이상치 점수로 사용한다.

**B. Tint Rotation Task**
HSV 색 공간의 Hue 채널에 오프셋 $\theta$를 더하는 변환을 통해 색상 정보를 학습한다.
- **학습**: 예측된 각도 $\Theta$를 기반으로 원본 이미지와 예측 이미지 사이의 RGB 공간에서의 $L_1$ 오차를 최소화한다.
$$L_{tint}(\gamma(I, \theta)) = \mathbb{E}_{\Theta|\gamma(I, \theta)} \left[ \frac{\|I - \gamma(I, \theta - \Theta)\|_1}{W \times H \times 255} \right]$$
- **추론**: Softmax 확률값과 각 각도별 $L_1$ 오차의 가중 합을 통해 점수를 산출한다.

**C. Intra-piece Task & Attention**
퍼즐 조각 내부에서 Tint rotation을 동시에 수행하며, 이때 객체 영역에 가중치를 두는 Attention map $P_{ij}$를 학습한다. 배경에 과적합되는 것을 방지하기 위해 Attention map의 분산을 조절하는 $L_{density}$를 추가한다.

### 2. Generative Branch (L-branch)
이미지의 질감을 학습하기 위해 Partial Re-colorization 작업을 수행한다.

- **구조**: UNet 아키텍처를 사용하여 L(luminance) 채널과 이미지 외곽 border $\alpha$의 색상 정보를 입력받아 중심부의 $(A, B)$ 색상 채널을 예측한다.
- **GMM 기반 밀도 추정**: 색상의 다봉성(multi-modality)을 처리하기 위해 각 픽셀의 색상 분포를 Gaussian Mixture Model로 모델링한다.
$$p(A_{ij}, B_{ij} | I^{part}) = \sum_{k=1}^{K} \pi_{ij}^{(k)} \mathcal{N}(A_{ij}, B_{ij}; \mu_{ij}^{(k)}, \Sigma_{ij}^{(k)})$$
- **학습**: EM 알고리즘을 통해 파라미터 $\pi, \mu, \Sigma$를 최적화하며, Cholesky decomposition을 사용하여 공분산 행렬 $\Sigma$의 양의 정정치(positive definite)를 보장한다.

### 3. OOD Detection 및 Fusion
- **OOD 함수**: Softmax truth(정답 클래스의 확률값)와 Mahalanobis distance를 실험하였으며, 최종 모델에서는 Softmax truth를 사용한다.
- **Fusion**: 각 작업(Puzzle, Tint, Colorization)에서 나온 OOD score들을 Median 함수를 통해 하나의 최종 이상치 점수로 통합한다.

## 📊 Results

### 실험 설정
- **데이터셋**:
    - Object Anomalies: F-MNIST, CIFAR-10, CIFAR-100 (One-vs-all 프로토콜)
    - Style Anomalies: CUB-200 (Birds), FounderType-200 (Fonts)
    - Local/Complex Anomalies: WMCA (Face anti-spoofing; 실물 얼굴 vs 가짜 얼굴 탐지)
- **지표**: AUROC, EER (Equal Error Rate), APCER @ 5% BPCER.
- **비교 대상**: ADGAN, GANomaly, GeoTrans, MHRot, PuzzleGeom, DROC-contrastive 등.

### 주요 결과
- **정량적 성과**: 
    - CIFAR-10에서 PuzzleGeom 대비 상대 오차가 36% 개선되었다.
    - WMCA(얼굴 위조 탐지) 데이터셋에서 AUROC 91.4%를 달성하며, SOTA 모델 및 심지어 일부 Semi-supervised 모델보다 높은 성능을 보였다.
    - 특히 WMCA에서 APCER @ 5% BPCER 수치를 33.8%(PuzzleGeom)에서 27.3%로 크게 낮추었다.
- **정성적 분석**: 각 작업(Puzzle, Tint, Colorization)이 서로 다른 유형의 이상치(구조적 결함, 색상 왜곡, 질감 이상)를 상보적으로 잘 잡아내는 것을 확인하였다.

## 🧠 Insights & Discussion

본 연구의 강점은 단일한 보조 작업에 의존하지 않고, 이미지의 세 가지 핵심 요소(구조, 색상, 질감)를 모두 고려한 Multi-cue 접근 방식을 취했다는 점이다. 특히 GMM을 이용한 색상 복원은 단순 회귀 모델이 가질 수 없는 색상의 다양성을 포착하여 Fine-grained 탐지 능력을 향상시켰다.

또한, Attention mechanism의 도입은 AD 모델이 흔히 겪는 '배경 노이즈' 문제를 효과적으로 억제하였다. 실험 결과, 배경이 다양하거나 객체가 작은 데이터셋일수록 Attention의 기여도가 높게 나타났다.

한계점으로는, 여전히 One-class setting에서 학습되므로 학습 데이터에 포함되지 않은 완전히 새로운 도메인의 정상 데이터가 들어올 경우 오탐지 가능성이 존재한다. 또한, 본 논문에서는 Softmax truth를 사용했으나, 분석 과정에서 Mahalanobis distance가 더 견고한(robust) 성능을 보인 점은 향후 개선 가능성을 시사한다.

## 📌 TL;DR

본 논문은 구조(Piece-wise Puzzle), 색상(Tint Rotation), 질감(GMM-based Colorization)이라는 세 가지 보조 작업을 결합한 효율적인 self-supervised anomaly detection 프레임워크를 제안한다. 이 방법은 특히 세밀한 차이를 구분해야 하는 Fine-grained 이상치 탐지와 얼굴 위조 탐지(Face anti-spoofing)에서 기존 SOTA 모델들을 크게 상회하는 성능을 보였으며, 추론 속도 또한 획기적으로 개선하였다. 이는 향후 정밀한 보안 시스템이나 산업용 결함 탐지 분야에 실용적으로 적용될 가능성이 매우 높다.