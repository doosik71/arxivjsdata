# From Pixel to Patch: Synthesize Context-aware Features for Zero-shot Semantic Segmentation

Zhangxuan Gu, Siyuan Zhou, Li Niu, Zihan Zhao, Liqing Zhang (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Zero-shot Semantic Segmentation (ZSSS)**이다. 일반적인 Semantic Segmentation은 모든 픽셀에 대해 정밀한 레이블링이 필요한 매우 노동 집약적인 작업이다. ZSSS의 목표는 훈련 단계에서 보지 못한 **Unseen categories**에 대해, 해당 카테고리의 **Category-level semantic representations** (예: word embedding)만을 이용하여 픽셀 단위의 세그멘테이션을 수행하는 것이다.

이 문제의 중요성은 데이터 어노테이션 비용을 획기적으로 줄이는 데 있으며, 특히 기존의 Zero-shot Learning 연구들이 주로 이미지 분류(Classification)에 집중되었던 것과 달리, 픽셀 수준의 정밀도가 요구되는 세그멘테이션 작업으로 이를 확장하는 것이 핵심이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **Context-aware feature Generation Network (CaGNet)**를 통해 Unseen categories에 대한 시각적 특징(Visual features)을 합성하여 분류기를 미세 조정(Finetuning)하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Context-aware Feature Synthesis**: 단순한 랜덤 벡터 대신, 픽셀 주변의 문맥 정보(Contextual information)를 인코딩한 **Contextual latent code**를 생성기의 입력으로 사용하여, 합성된 특징의 다양성을 높이고 Mode collapse 문제를 완화하였다.
2. **From Pixel to Patch**: 기존의 픽셀 단위 특징 생성에서 더 나아가, 픽셀 간의 관계(Inter-pixel relationship)를 고려할 수 있도록 **Patch-wise feature generation** 및 Finetuning 방식을 제안하였다.
3. **Semantic Layout Modeling**: Patch-wise 특징을 생성하기 위해 **PixelCNN**을 수정하여, 타당한 카테고리 배치(Semantic layout)를 가진 Category patch를 먼저 생성하고 이를 기반으로 특징 패치를 합성하는 파이프라인을 구축하였다.
4. **Adaptive Context Selection**: 각 픽셀마다 적절한 문맥 스케일(Small, Middle, Large)을 자동으로 결정하는 **Context selector**를 도입하여 특징 추출 능력을 향상시켰다.

## 📎 Related Works

논문은 ZSSS 접근 방식을 크게 두 그룹으로 나눈다.

1. **Visual-to-Semantic Mapping**: 시각적 특징을 Word embedding 공간으로 투영하는 방식 (예: SPNet).
2. **Semantic-to-Visual Mapping**: Word embedding으로부터 시각적 특징을 생성하는 방식 (예: ZS3Net, CSRL).

본 연구는 두 번째 그룹에 속하며, 특히 **ZS3Net**을 확장하였다. ZS3Net은 Word embedding과 랜덤 벡터를 사용하여 특징을 생성하지만, 다음과 같은 한계가 있다.

- **Mode Collapse**: 랜덤 벡터를 사용함에 따라 생성된 특징이 몇 가지 모드로 수렴하여 다양성이 부족하다.
- **Limited Finetuning**: 픽셀 단위 특징만 생성하므로, 분류기의 마지막 $1 \times 1$ Convolution layer만 미세 조정할 수 있다는 제약이 있다.

CaGNet은 랜덤 벡터를 문맥 기반의 Latent code로 대체하고, 패치 단위 생성을 통해 더 깊은 층의 분류기를 학습시킴으로써 이러한 한계를 극복한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

CaGNet은 **Segmentation Backbone ($E$)**, **Contextual Module ($CM$)**, **Feature Generator ($G$)**, **Discriminator ($D$)**, 그리고 **Classifier ($C$)**로 구성된다. 기본 백본으로는 Deeplabv2를 사용한다.

### 2. Contextual Module (CM) 및 Latent Code

CM은 백본의 출력 특징 맵 $F^n$으로부터 픽셀별 문맥 정보를 추출한다.

- **Context Selector**: 세 가지 다른 스케일의 Dilated Convolution을 통해 문맥 맵 $\hat{F}_0^n, \hat{F}_1^n, \hat{F}_2^n$을 생성하고, 학습 가능한 가중치 맵 $A^n$을 통해 각 픽셀에 최적화된 스케일을 선택적으로 결합한다.
- **Contextual Latent Code**: 결합된 문맥 정보를 통해 가우시안 분포의 파라미터인 $\mu_{z_{n,i}}$와 $\sigma_{z_{n,i}}$를 예측하고, 다음과 같이 latent code $z_{n,i}$를 샘플링한다.
    $$z_{n,i} = \mu_{z_{n,i}} + \epsilon \sigma_{z_{n,i}}, \quad \epsilon \sim \mathcal{N}(0,1)$$
- **KL Divergence Loss**: 샘플링된 분포가 표준 정규 분포 $\mathcal{N}(0,1)$에 가깝도록 강제하여 추론 시 확률적 샘플링이 가능하게 한다.
    $$\mathcal{L}_{KL} = D_{KL}[\mathcal{N}(\mu_{z_{n,i}}, \sigma_{z_{n,i}}) || \mathcal{N}(0,1)]$$

### 3. Pixel-wise Feature Generation

생성기 $G$는 Word embedding $w_{s_{n,i}}$와 latent code $z_{n,i}$를 입력받아 가짜 특징 $\tilde{x}_{s_{n,i}}$를 생성한다. 학습을 위해 다음의 손실 함수들을 사용한다.

- **Reconstruction Loss**: 실제 특징 $x_{s_{n,i}}$와의 L2 거리를 최소화한다.
    $$\mathcal{L}_{REC} = \sum_{n,i} ||x_{s_{n,i}} - \tilde{x}_{s_{n,i}}||_2^2$$
- **Classification Loss**: 분류기 $C$의 교차 엔트로피 손실이다.
    $$\mathcal{L}_{CLS} = -\sum_{n,i} y_{s_{n,i}} \log(C(x_{s_{n,i}}))$$
- **Adversarial Loss**: 생성된 특징이 실제 특징과 구분이 불가능하도록 하는 GAN 손실이다.
    $$\mathcal{L}_{ADV} = \sum_{n,i} (D(x_{s_{n,i}}))^2 + (1 - D(\tilde{x}_{s_{n,i}}))^2$$

### 4. Patch-wise Feature Generation

픽셀 간 관계를 학습하기 위해 $3 \times 3$ 패치 단위로 특징을 생성한다.

- **Category Patch Generation**: PixelCNN을 수정하여 Word embedding을 기반으로 타당한 카테고리 배치(Category patch)를 생성한다. 이는 $\log p(c) = \sum_{t=1}^{k^2} \log p(c_t | w_{c_1}, \dots, w_{c_{t-1}})$ 식을 통해 자동회귀적으로 생성된다.
- **Feature Patch Synthesis**: 생성된 카테고리 패치의 각 픽셀에 동일한 $z$를 적용하고, $G$를 통해 특징 패치를 생성하여 분류기의 $3 \times 3$ Conv layer를 학습시킨다.

### 5. 학습 및 최적화 절차

학습은 두 단계로 진행된다.

1. **Training**: Seen categories 데이터만을 사용하여 $E, CM, G, D, C$ 전체를 최적화한다.
2. **Finetuning**: $E$와 $CM$을 고정한 상태에서, $G$가 생성한 Seen/Unseen 특징들을 사용하여 분류기 $C$를 미세 조정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Pascal-VOC 2012, Pascal-Context, COCO-stuff.
- **지표**: mIoU, hIoU (Seen과 Unseen의 균형을 맞추기 위해 hIoU를 핵심 지표로 사용).
- **비교 대상**: SPNet, ZS3Net, CSRL, Hu et al.

### 2. 정량적 결과

실험 결과, CaGNet(pa, 패치 기반)이 CaGNet(pi, 픽셀 기반) 및 기존 베이스라인들을 상당 부분 능가하였다.

- **Pascal-VOC**에서 CaGNet(pa)는 hIoU $0.4326$을 달성하여 ZS3Net ($0.2874$) 대비 큰 성능 향상을 보였다.
- 특히 **Unseen categories**의 mIoU가 크게 상승하여, 문맥 기반 특징 생성이 Unseen 클래스 인식에 매우 효과적임을 입증하였다.
- **Patch-wise vs Pixel-wise**: 모든 데이터셋에서 CaGNet(pa)가 CaGNet(pi)보다 우수한 성능을 보였으며, 이는 픽셀 간의 지역적 관계를 고려한 학습이 유효함을 의미한다.

### 3. 정성적 분석 및 다양성 평가

- **특징 다양성**: Unseen 특징 간의 평균 유클리드 거리를 측정한 결과, ZS3Net보다 CaGNet이 실제(Real) 특징의 다양성에 훨씬 근접한 결과를 보였으며, 이는 Mode collapse가 성공적으로 억제되었음을 시사한다.
- **Context Selector**: 시각화 결과, 얼굴이나 작은 객체 같은 변별력 있는 지역은 Small-scale 문맥을, 배경 등은 Large-scale 문맥을 선택하는 적응적 특성이 확인되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰

- **문맥의 중요성**: 단순히 랜덤 노이즈를 사용하는 대신, 픽셀의 주변 정보를 Latent code로 활용함으로써 시각적 특징의 생성 품질과 다양성을 동시에 확보하였다.
- **구조적 제약의 효과**: PixelCNN을 이용해 생성한 Category patch에 '카테고리 수 제한(최대 3개)' 및 '일부 픽셀 고정'과 같은 제약을 가했을 때 더 현실적인 패치가 생성되어 성능이 향상되었다.

### 2. 한계 및 논의사항

- **패치 크기의 제한**: 실험 결과 패치 크기 $k$가 커질수록 ($3 \rightarrow 5 \rightarrow 7$) 성능이 하락하였다. 이는 Unseen 객체의 정확한 형태나 포즈를 예측하여 큰 패치를 생성하는 것이 매우 어렵기 때문이며, 또한 패치가 커지면 내부 픽셀들이 동일한 문맥 코드를 공유한다는 가정이 깨지기 때문이다.
- **계산 비용**: Contextual Module의 추가로 인해 ZS3Net 대비 추론 속도가 약간 느려졌으나, 실용적인 범위 내에 있다.

## 📌 TL;DR

본 논문은 Zero-shot Semantic Segmentation에서 Unseen 클래스를 인식하기 위해 **문맥 정보를 반영한 특징 생성 네트워크(CaGNet)**를 제안한다. 랜덤 벡터 대신 **Contextual latent code**를 사용하여 특징의 다양성을 높여 Mode collapse를 해결하였으며, **PixelCNN 기반의 패치 생성** 방식을 도입해 픽셀 간 관계를 학습함으로써 세그멘테이션 성능을 크게 향상시켰다. 이 연구는 향후 더 큰 규모의 객체 형태와 포즈를 고려한 특징 합성 연구로 확장될 가능성이 높다.
