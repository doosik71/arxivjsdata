# DA-TRANSUNET: INTEGRATING SPATIAL AND CHANNEL DUAL ATTENTION WITH TRANSFORMER U-NET FOR MEDICAL IMAGE SEGMENTATION

Guanqun Sun, Yizhi Pan, Weikun Kong, Zichang Xu, Jianhua Ma, Teeradaj Racharak, Le-Minh Nguyen, Junyi Xin (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 질병의 정량화 및 치료 평가를 위해 매우 중요하지만, 기존의 자동화된 분할 모델들은 몇 가지 한계를 가지고 있다. 전통적인 U-Net 구조와 그 변형 모델들은 국소적인 특징 추출에는 능숙하지만, 이미지의 내재적인 위치(Position) 및 채널(Channel) 특징을 충분히 활용하지 못하는 경향이 있다.

최근 이를 해결하기 위해 Transformer를 통합한 모델들이 등장하였으나, 이들 역시 두 가지 주요 문제에 직면해 있다. 첫째, Transformer는 글로벌 컨텍스트를 파악하는 데는 뛰어나지만, 이미지 특유의 공간적 위치와 채널 간의 상관관계를 고려하는 내장 메커니즘이 부족하다. 둘째, 성능 향상을 위해 단순히 Transformer 블록을 과도하게 쌓는 방식은 파라미터 수를 급격히 증가시켜 계산 복잡도를 높이는 반면, 실제 성능 향상은 미미한 경우가 많다. 따라서 본 논문은 파라미터 효율성을 유지하면서도 위치 및 채널 특징을 정밀하게 추출할 수 있는 새로운 프레임워크를 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer 기반의 U-Net 구조에 공간적 및 채널적 주의 집중 메커니즘을 결합한 **Dual Attention Block (DA-Block)**을 전략적으로 배치하는 것이다. 

단순히 Transformer 층을 늘리는 대신, 정밀하게 설계된 DA-Block을 Transformer 층 이전의 임베딩 단계와 인코더-디코더를 잇는 Skip Connection 층에 배치함으로써, 이미지의 위치 및 채널 특성을 최적화하여 추출한다. 이를 통해 Transformer가 더 정교한 글로벌 특징을 추출할 수 있도록 돕고, Skip Connection을 통해 전달되는 특징 맵에서 불필요한 정보를 필터링하여 디코더의 복원 능력을 향상시키는 구조적 설계를 제안한다.

## 📎 Related Works

의료 영상 분할 분야에서는 U-Net을 기반으로 한 다양한 연구가 진행되었다. ResUnet은 잔차 연결(Residual connection)을, UNet++는 Skip Connection의 구조적 개선을 통해 성능을 높였으며, Attention U-net은 주의 집중 메커니즘을 도입하여 특정 영역의 국소화 능력을 개선하였다. 

최근에는 Vision Transformer (ViT)를 U-Net에 결합한 TransUNet이나 Swin-Unet과 같은 모델들이 등장하여 글로벌 컨텍스트 추출 능력을 입증하였다. 그러나 이러한 모델들은 여전히 이미지 특유의 위치 및 채널 정보를 충분히 활용하지 못하거나, 과도한 Transformer 블록 사용으로 인한 계산 복잡도 증가라는 한계를 가지고 있다. 본 연구는 이러한 기존 모델들과 달리, DA-Block을 통해 위치와 채널 특징을 명시적으로 추출하고 이를 Transformer 및 Skip Connection과 결합함으로써 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
DA-TransUNet은 전형적인 U-자형 인코더-디코더 구조를 따르지만, 인코더 내부에 DA-Block과 Transformer를 통합하고 Skip Connection 경로에 DA-Block을 추가 배치한 형태이다.

### Dual Attention Block (DA-Block)
DA-Block은 위치 특징을 추출하는 **Position Attention Module (PAM)**과 채널 특징을 추출하는 **Channel Attention Module (CAM)**로 구성된다.

1. **Position Attention Module (PAM)**: 
특징 맵 내의 임의의 두 위치 사이의 공간적 의존성을 포착한다. 입력 특징 $A \in \mathbb{R}^{C \times H \times W}$를 컨볼루션 층을 통해 $B, C, D \in \mathbb{R}^{C \times H \times W}$로 변환한 뒤, $B$와 $C$의 행렬 곱과 softmax를 통해 공간 주의 맵 $S \in \mathbb{R}^{N \times N}$ (여기서 $N=H \times W$)를 계산한다.
$$S_{ji} = \frac{\exp(B_i \cdot C_j)}{\sum_{i=1}^N \exp(B_i \cdot C_j)}$$
최종 출력 $E_j$는 다음과 같이 계산된다.
$$E_j = \alpha \sum_{i=1}^N (s_{ji}D_i) + A_j$$
여기서 $\alpha$는 학습 가능한 파라미터이다.

2. **Channel Attention Module (CAM)**: 
채널 간의 상관관계를 추출한다. 입력 $A$를 $\mathbb{R}^{C \times N}$으로 변형한 후, $A$와 그 전치 행렬의 곱에 softmax를 적용하여 채널 주의 맵 $X \in \mathbb{R}^{C \times C}$를 생성한다.
$$x_{ji} = \frac{\exp(A_i \cdot A_j)}{\sum_{i=1}^C \exp(A_i \cdot A_j)}$$
최종 출력 $E_j$는 다음과 같다.
$$E_j = \beta \sum_{i=1}^N (x_{ji}A_i) + A_j$$
여기서 $\beta$는 학습 가능한 파라미터이다.

3. **DA-Block 통합**:
PAM과 CAM 경로를 병렬로 구성하며, 각 경로의 출력 $\hat{\alpha}_1, \hat{\alpha}_2$를 합산한 후 컨볼루션 층을 통해 최종 출력을 얻는다.
$$\text{output} = \text{Conv}(\hat{\alpha}_1 + \hat{\alpha}_2)$$

### Encoder 및 Skip Connection
- **Encoder**: 3개의 컨볼루션 블록 $\rightarrow$ DA-Block $\rightarrow$ 임베딩 층 $\rightarrow$ Transformer 층 순으로 구성된다. DA-Block이 Transformer 앞에 위치하여 위치 및 채널 정보를 먼저 정제함으로써 Transformer의 글로벌 특징 추출 효율을 극대화한다.
- **Skip Connection**: 인코더와 디코더 사이의 세 가지 스케일 층에 DA-Block을 배치한다. 이는 인코더에서 전달되는 특징 중 불필요한 정보를 필터링하고 유의미한 특징만을 디코더에 전달하여 세밀한 이미지 복원을 돕는다.

### Decoder
디코더는 전통적인 CNN 기반의 업샘플링 구조를 사용한다. Skip Connection을 통해 전달된 정제된 특징 맵과 디코더의 특징 맵을 융합(Fusion)하고, 업샘플링 컨볼루션 블록을 거쳐 최종적으로 Segmentation Head를 통해 원본 해상도로 복원한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Synapse (복부 장기), CVC-ClinicDB (폴립), Chest X-ray, Kvasir-SEG (폴립), Kvasir-Instrument (내시경 도구), 2018 ISIC-Task (피부 병변) 등 총 6개의 데이터셋을 사용하였다.
- **평가 지표**: Dice Coefficient (DSC), Intersection over Union (IoU), Hausdorff Distance (HD)를 사용하였다.
- **구현 세부사항**: PyTorch 프레임워크와 NVIDIA RTX 3090 GPU를 사용하였으며, Adam 및 SGD 옵티마이저를 채택하였다. 손실 함수로는 $\frac{1}{2} \text{BCE} + \frac{1}{2} \text{DiceLoss}$ (또는 Synapse의 경우 Cross-Entropy)를 사용하였다.

### 주요 결과
- **Synapse 데이터셋**: DA-TransUNet은 평균 DSC $79.80\%$, HD $23.48 \text{mm}$를 기록하며 비교 모델 중 가장 높은 DSC 성능을 보였다. 특히 TransUNet 대비 DSC가 $2.32\%$ 향상되었으며, 췌장(Pancreas) 분할에서는 $5.73\%$라는 큰 폭의 향상을 보였다.
- **기타 데이터셋**: CVC-ClinicDB, Chest X-ray, ISIC2018, Kvasir-Instrument, Kvasir-SEG의 5개 데이터셋 모두에서 TransUNet보다 높은 IoU와 Dice 성능을 기록하였으며, 대부분의 데이터셋에서 SOTA(State-of-the-art) 성능을 달성하였다.
- **추론 속도**: 이미지 한 장당 분할 시간은 $35.98 \text{ms}$로, TransUNet($33.58 \text{ms}$)과 큰 차이가 없으면서 성능은 더 우수함을 입증하였다.

### 절제 연구 (Ablation Study)
DA-Block의 배치에 따른 성능 변화를 분석한 결과, 인코더에만 추가했을 때보다 Skip Connection에 추가했을 때, 그리고 두 곳 모두에 추가했을 때 성능이 단계적으로 향상됨을 확인하였다. 특히 Skip Connection의 모든 층에 DA-Block을 적용했을 때 DSC $79.80\%$로 최적의 성능이 나타났다.

## 🧠 Insights & Discussion

본 논문은 Transformer의 글로벌 특징 추출 능력과 DA-Block의 이미지 특화 특징(위치, 채널) 추출 능력이 상호 보완 관계에 있음을 보여주었다. 특히 Transformer 이전에 DA-Block을 배치함으로써 Transformer가 더 정제된 데이터를 입력받아 글로벌 컨텍스트를 더 정확하게 파악할 수 있게 되었다는 점이 핵심적인 통찰이다. 또한 Skip Connection에서의 DA-Block은 단순한 특징 전달을 넘어 '특징 정제' 역할을 수행하여 디코더의 복원 정확도를 높였다.

다만, 본 모델은 다음과 같은 한계를 가진다. 첫째, DA-Block의 도입으로 인해 연산 복잡도가 증가하여 실시간성이나 자원 제한 환경에서는 부담이 될 수 있다. 둘째, 디코더 부분은 기존 U-Net의 구조를 그대로 유지하고 있어, 디코더 자체를 최적화한다면 추가적인 성능 향상의 여지가 남아 있다.

## 📌 TL;DR

본 연구는 Transformer U-Net 구조에 공간(PAM) 및 채널(CAM) 주의 집중 메커니즘을 결합한 **DA-TransUNet**을 제안하였다. DA-Block을 인코더의 Transformer 이전 단계와 Skip Connection 경로에 전략적으로 배치함으로써, 글로벌 컨텍스트와 이미지 특유의 세부 특징을 동시에 효율적으로 포착하였다. 실험 결과, 6개의 의료 영상 데이터셋에서 기존 모델들을 능가하는 성능을 보였으며, 이는 향후 정밀한 자동 의료 진단 시스템 구축에 기여할 가능성이 높다.