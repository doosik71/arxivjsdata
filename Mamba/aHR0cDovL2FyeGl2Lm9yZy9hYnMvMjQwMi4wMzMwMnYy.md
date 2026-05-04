# Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining

Jiarun Liu, Hao Yang, Hong-Yu Zhou, Yan Xi, Lequan Yu, Yizhou Yu, Yong Liang, Guangming Shi, Shaoting Zhang, Hairong Zheng, and Shanshan Wang (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)에서 정확한 결과를 얻기 위해서는 국소적 특징(Local features)부터 전역적 의존성(Global dependencies)에 이르는 다중 스케일 정보의 통합이 필수적이다. 그러나 기존의 방법론들은 다음과 같은 한계점을 가진다.

첫째, 합성곱 신경망(CNN)은 수용 영역(Receptive field)이 국소적으로 제한되어 있어 이미지 내 먼 거리의 정보를 캡처하는 전역적 문맥 모델링 능력이 부족하다. 둘째, 비전 트랜스포머(ViT)는 전역적 의존성을 모델링할 수 있으나, 어텐션 메커니즘의 계산 복잡도가 시퀀스 길이의 제곱에 비례하는 Quadratic complexity 문제를 겪으며, 특히 고해상도 의료 영상 처리 시 메모리 소모와 계산 부담이 매우 크다. 또한 ViT는 데이터 집약적인 특성으로 인해 데이터셋이 제한적인 의료 분야에서 오버피팅(Overfitting)에 취약하다.

최근 Mamba 기반의 상태 공간 모델(State Space Models, SSMs)이 선형 복잡도로 긴 시퀀스를 효율적으로 모델링하며 대안으로 떠올랐으나, 기존의 Mamba 기반 의료 영상 모델들은 대부분 처음부터 학습(Train from scratch)되는 방식을 취하고 있다. CNN과 ViT에서 이미 그 효과가 입증된 대규모 데이터셋 기반의 사전 학습(Pretraining)을 Mamba 모델에 어떻게 적용하고 활용할 것인가가 본 논문의 핵심 연구 문제이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **ImageNet으로 사전 학습된 Mamba 기반의 백본을 의료 영상 분할 네트워크의 인코더로 통합**하여, 데이터 효율성을 높이고 전역적 모델링 능력을 극대화하는 것이다. 주요 기여 사항은 다음과 같다.

1. **사전 학습의 영향력 검증**: Mamba 기반 네트워크에서 ImageNet 사전 학습이 의료 영상 분할 성능 향상에 결정적인 역할을 한다는 것을 최초로 실험적으로 입증하였다.
2. **Swin-UMamba 아키텍처 제안**: 사전 학습된 비전 모델의 능력을 통합할 수 있도록 설계된 새로운 Mamba 기반 UNet 구조를 제안하였다.
3. **효율적인 변형 모델 Swin-UMamba$\dagger$ 제안**: 디코더 부분까지 Mamba 블록으로 대체하여 파라미터 수와 연산량(FLOPs)을 획기적으로 줄이면서도 경쟁력 있는 성능을 유지하는 경량화 모델을 제시하였다.

## 📎 Related Works

기존의 의료 영상 분할 연구는 크게 세 가지 방향으로 진행되었다.

1. **CNN 기반 접근 방식**: U-Net, nnU-Net, SegResNet 등이 대표적이며, 국소적 특징 추출에는 뛰어나지만 전역적 문맥 파악에 한계가 있다.
2. **Transformer 기반 접근 방식**: UNETR, Swin-UNETR, nnFormer 등이 전역적 의존성을 모델링하기 위해 도입되었으나, 높은 계산 복잡도와 많은 양의 데이터를 요구하는 단점이 있다.
3. **Mamba 기반 접근 방식**: 최근 U-Mamba, SegMamba 등이 제안되어 선형 복잡도로 전역 정보를 처리하려는 시도가 있었다. 하지만 이들은 대부분 사전 학습을 활용하지 않고 의료 데이터로만 학습되었기에, 소규모 데이터셋에서의 일반화 성능과 학습 안정성 측면에서 한계가 존재한다.

Swin-UMamba는 이러한 기존 Mamba 기반 모델들과 달리, 일반 비전 도메인에서 학습된 VMamba-Tiny의 가중치를 전이 학습(Transfer Learning)함으로써 초기화 성능을 높이고 오버피팅을 방지한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
Swin-UMamba는 전형적인 U-자형(U-shaped) 아키텍처를 따르며, 크게 세 부분으로 구성된다:
1. **Mamba-based Encoder**: ImageNet으로 사전 학습된 VMamba-Tiny 구조를 활용하여 다중 스케일 특징을 추출한다.
2. **Decoder**: 인코더에서 추출된 고수준 세맨틱 정보와 스킵 연결(Skip connection)을 통해 전달된 저수준 디테일을 결합하여 최종 분할 맵을 생성한다.
3. **Skip Connections**: 인코더와 디코더 사이의 정보 격차를 줄이기 위해 사용된다.

### Mamba-based VSS Block 및 SS2D
본 모델의 기본 단위는 **Visual State Space (VSS) block**이다. 2D 이미지를 1D 시퀀스로 단순 변환할 때 발생하는 수용 영역 제한 문제를 해결하기 위해 **2D-selective-scan (SS2D)** 메커니즘을 도입하였다. SS2D는 이미지 패치를 네 가지 방향으로 전개하여 네 개의 서로 다른 시퀀스를 생성하고, 각각을 SSM으로 처리한 후 다시 병합한다.

입력 특징 $z$에 대해 SS2D의 출력 $\bar{z}$는 다음과 같이 정의된다.
$$z^v = \text{expand}(z, v)$$
$$\bar{z}^v = \text{S6}(z^v)$$
$$\bar{z} = \text{merge}(\bar{z}^1, \bar{z}^2, \bar{z}^3, \bar{z}^4)$$
여기서 $v \in \{1, 2, 3, 4\}$는 네 가지 스캔 방향을 의미하며, $\text{S6}$는 각 요소가 압축된 은닉 상태를 통해 이전 샘플들과 상호작용하게 하는 핵심 SSM 연산자이다.

### Encoder 및 사전 학습 통합
인코더는 총 5단계(Stage)로 구성된다.
- **Stage 1 (Stem)**: $7 \times 7$ 커널과 stride 2를 가진 Conv 레이어를 통해 $2\times$ 다운샘플링을 수행한다.
- **Stage 2**: $2 \times 2$ 패치 임베딩 레이어를 통해 해상도를 원래 이미지의 $1/4$로 유지한다.
- **Stage 3-5**: 패치 병합(Patch Merging) 레이어로 $2\times$ 다운샘플링을 수행하며, 각 단계마다 여러 개의 VSS 블록이 배치되어 특징을 추출한다.
- **사전 학습 적용**: VSS 블록과 패치 병합 레이어의 가중치는 ImageNet으로 사전 학습된 VMamba-Tiny 모델에서 가져와 초기화한다. 단, 패치 임베딩 블록은 입력 채널 및 패치 크기 차이로 인해 제외된다.

### Decoder 구조 (Swin-UMamba vs Swin-UMamba$\dagger$)
1. **Swin-UMamba (CNN-based Decoder)**:
   - 전치 합성곱(Transpose Convolution)을 통한 업샘플링을 수행한다.
   - 스킵 연결된 특징 $z'_l$과 이전 단계의 특징 $z_{l+1}$을 결합(Cat)한 후, 잔차 연결(Residual connection)이 포함된 두 개의 Conv 블록($\text{Res}^{(1)}_l, \text{Res}^{(2)}_l$)을 통과시킨다.
   - 깊은 감독(Deep supervision)을 위해 각 스케일마다 세그멘테이션 헤드($\text{Conv}_l$)를 배치한다.
   - 수식:
     $$\hat{z}^l = \text{Res}^{(2)}_l(\text{Cat}(z_{l+1}, \text{Res}^{(1)}_l(z'_l)))$$
     $$z^l = \text{DeConv}_l(\hat{z}^l), \quad y^l = \text{Conv}_l(\hat{z}^l)$$

2. **Swin-UMamba$\dagger$ (Mamba-based Decoder)**:
   - 무거운 CNN 기반 디코더를 대체하여 **Patch Expanding** 레이어와 **VSS 블록**을 사용한다.
   - 입력 이미지의 직접적인 스킵 연결을 제거하고, 4x 패치 임베딩 레이어를 도입하여 구조를 단순화하였다.
   - 이를 통해 파라미터 수를 60M에서 28M로, FLOPs를 68.0G에서 18.9G로 대폭 낮추었다.

## 📊 Results

### 실험 설정
- **데이터셋**: AbdomenMRI (복부 장기 분할), Endoscopy (내시경 도구 분할), Microscopy (세포 분할).
- **지표**: Dice Similarity Coefficient (DSC), Normalized Surface Distance (NSD), F1 score.
- **비교 모델**: nnU-Net, SegResNet (CNN), UNETR, Swin-UNETR, nnFormer (ViT), U-Mamba (Mamba).

### 주요 결과
1. **성능 우위**: 모든 데이터셋에서 Swin-UMamba와 Swin-UMamba$\dagger$가 기존 CNN, ViT, Mamba 기반 모델들을 능가하였다. 특히 AbdomenMRI에서 U-Mamba\_Enc 대비 평균 2.72%의 성능 향상을 보였다.
2. **사전 학습의 중요성**:
   - AbdomenMRI에서 사전 학습을 제외했을 때 DSC가 $0.7760 \rightarrow 0.7054$로 급격히 하락하였다.
   - Swin-UMamba$\dagger$의 경우, 사전 학습 없이는 기본 설정에서 수렴(Convergence)조차 제대로 되지 않는 현상이 발견되었다.
   - Endoscopy 데이터셋에서는 사전 학습 적용 시 DSC가 약 12.84% 향상되어, 소규모 데이터셋일수록 사전 학습의 효과가 극대화됨을 확인하였다.
3. **효율성**: Swin-UMamba$\dagger$는 Swin-UMamba보다 파라미터와 연산량이 훨씬 적음에도 불구하고, Endoscopy 및 Microscopy 데이터셋에서 오히려 더 높거나 대등한 성능을 보였다. 이는 적은 파라미터가 소규모 데이터셋에서 오버피팅을 방지하는 데 도움이 되었음을 시사한다.

## 🧠 Insights & Discussion

**강점 및 분석**:
본 연구는 Mamba 모델이 비전 태스크에서 강력한 성능을 보임에도 불구하고, 정작 의료 영상 분야에서는 사전 학습의 이점을 활용하지 못하고 있었다는 점을 정확히 짚어냈다. 실험 결과는 Mamba-based 모델 역시 CNN이나 ViT와 마찬가지로, 대규모 일반 데이터셋(ImageNet)에서 학습된 가중치를 통해 강건한 초기값을 가질 때 의료 영상과 같은 특수 도메인에서도 빠르게 수렴하고 높은 성능을 낼 수 있음을 보여준다.

**한계 및 논의**:
- **디코더의 역할**: AbdomenMRI에서는 CNN 기반 디코더가 더 좋은 성능을 냈으나, 다른 데이터셋에서는 Mamba 기반 디코더가 유리했다. 이는 데이터셋의 해상도와 타겟 객체의 특성에 따라 최적의 디코더 구조가 다를 수 있음을 의미한다.
- **하이퍼파라미터**: 저자들은 SOTA 달성보다 사전 학습의 영향력 분석에 집중했으므로, 추가적인 튜닝을 통해 성능을 더 끌어올릴 여지가 남아 있다.
- **데이터 의존성**: 사전 학습이 매우 효과적이었지만, 이는 결국 ImageNet이라는 거대 데이터셋에 의존하는 결과이다. 의료 영상 전용 대규모 데이터셋으로 사전 학습을 진행했을 때의 효과에 대한 추가 연구가 필요하다.

## 📌 TL;DR

Swin-UMamba는 **ImageNet으로 사전 학습된 VMamba 백본을 UNet 구조의 인코더로 결합**한 의료 영상 분할 모델이다. 실험을 통해 **Mamba 기반 모델에서도 사전 학습이 성능 향상과 학습 안정성에 결정적인 역할**을 한다는 것을 입증하였으며, CNN 및 ViT 기반 모델보다 우수한 성능을 기록하였다. 특히 Mamba 기반 디코더를 적용한 $\text{Swin-UMamba}\dagger$는 연산량을 획기적으로 줄이면서도 높은 효율성을 보여, 향후 자원이 제한된 의료 현장으로의 배포 가능성을 높였다.