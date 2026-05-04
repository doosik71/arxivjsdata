# A MUTUAL INCLUSION MECHANISM FOR PRECISE BOUNDARY SEGMENTATION IN MEDICAL IMAGES

Yizhi Pan, Junyi Xin, Tianhua Yang, Teeradaj Racharak, Le-Minh Nguyen, Guanqun Sun (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 질병의 정량화, 예후 평가 및 치료 결과 측정에 있어 핵심적인 역할을 한다. 그러나 기존의 딥러닝 기반 분할 방법들은 글로벌 특징(Global features)과 로컬 특징(Local features)의 심층적인 통합이 부족하며, 특히 의료 영상에서 매우 중요한 이상 영역의 정밀한 경계(Boundary) 디테일을 포착하는 데 한계가 있다.

의료 영상 데이터는 일반적으로 샘플 크기가 작고, 장기나 병변의 경계를 매우 정밀하게 묘사해야 한다는 특성이 있다. 일반적인 이미지 분할 모델과 달리, 의료 영상 분할은 병변 부위와 경계 디테일에 특화된 주의(Attention)가 필요하며, 이를 위해 채널(Channel) 정보와 위치(Position) 정보를 효과적으로 결합하는 메커니즘이 필수적이다. 본 논문의 목표는 이러한 한계를 극복하여 의료 영상의 정밀한 경계 분할을 가능하게 하는 새로운 모델인 MIPC-Net을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 방사선 전문의(Radiologists)의 작업 패턴에서 영감을 얻은 '상호 포함(Mutual Inclusion)' 메커니즘이다. 단순히 Transformer 모듈을 쌓는 방식이 아니라, 위치 특징을 추출할 때 채널 정보에 집중하고, 반대로 채널 특징을 추출할 때 위치 정보에 집중하도록 설계하여 두 정보가 서로를 보완하며 강화하도록 하였다.

주요 기여 사항은 다음과 같다:
1. **MIPC-Block 제안**: 위치 주의(Position Attention)와 채널 주의(Channel Attention)를 상호 포함 구조로 결합하여 의료 영상의 경계 분할 정밀도를 높였다.
2. **GL-MIPC-Residue 도입**: 인코더와 디코더의 통합을 강화하는 글로벌 잔차 연결(Global Residual Connection)을 통해 특징 추출 과정에서 손실된 유효 정보를 복원하고 불필요한 정보를 필터링한다.
3. **성능 검증**: Synapse, ISIC2018-Task, Segpc 등 세 가지 공개 데이터셋에서 기존 SOTA(State-of-the-Art) 모델들을 능가하는 성능을 입증하였으며, 특히 Hausdorff Distance(HD) 지표에서 괄목할 만한 감소를 달성하였다.

## 📎 Related Works

### 1. U-structure 기반 모델 통합
U-Net은 의료 영상 분할의 표준 구조로 자리 잡았으며, 이를 최적화하기 위한 다양한 변형 모델이 제안되었다. UNet++와 Unet3++는 복잡한 Skip-connection을 통해 특징 전달을 강화했고, ResUnet은 잔차 학습을 도입하였다. 최근에는 TransUNet과 Swin-Unet처럼 Transformer를 결합하여 글로벌 문맥(Global context)을 캡처하려는 시도가 이어지고 있다. 하지만 이러한 모델들은 전반적인 분할 겹침(Overlap) 성능에 집중할 뿐, 경계 디테일을 정밀하게 복원하는 데에는 여전히 한계가 있다.

### 2. Attention 모듈의 활용
채널 주의와 공간/위치 주의를 결합한 Dual Attention 메커니즘이 제안되어 왔다. SA-UNet, AA-TransUNet, DA-TransUNet 등이 대표적이다. 그러나 기존 방식들은 단순히 두 주의 메커니즘을 병렬로 배치하거나 단순 결합하는 수준에 그쳐, 의료 영상의 특성에 맞춘 심층적인 특징 추출에는 미흡한 점이 있다. 본 논문은 이를 해결하기 위해 두 주의 메커니즘이 서로를 가이드하는 상호 포함 방식을 제안하여 차별성을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (MIPC-Net)
MIPC-Net은 기본적으로 U-자형 인코더-디코더 구조를 따른다. 전체 파이프라인은 다음과 같은 세 가지 핵심 구성 요소로 이루어진다:
- **Encoder**: CNN 블록, MIPC-Block, Embedding layer, Transformer layer로 구성된다. CNN이 추출한 특징을 MIPC-Block이 정제하고, Transformer가 글로벌 정보를 캡처한다.
- **GL-MIPC-Skip-Connections**: 인코더와 디코더 사이의 시맨틱 간극을 줄이기 위해 DA-Block(Dual Attention Block)과 GL-MIPC-Residue를 사용하여 특징을 정제하고 복원한다.
- **Decoder**: 업샘플링 컨볼루션 블록과 특징 융합(Feature fusion) 과정을 통해 원본 해상도의 예측 맵을 재구성한다.

### 2. MIPC-Block (Mutual Inclusion of Position and Channel)
MIPC-Block은 위치와 채널 정보를 상호 보완적으로 추출하는 구조로, 크게 세 부분(Part A, B, C)으로 나뉜다.

- **Part A (Position-Dominant)**: 위치 정보를 주축으로 하되 채널 정보가 보조한다.
  - $\beta_1 = \text{FC}(\text{ChannelPool}(\text{Input}))$ : 채널 간 상관관계를 학습한다.
  - $\beta_2 = \text{PAM}(\text{Input})$ : Position Attention Module(PAM)을 통해 위치 특징을 추출한다.
  - 최종 출력: $\beta = \text{Sigmoid}(\beta_1) \cdot \beta_2$

- **Part C (Channel-Dominant)**: 채널 정보를 주축으로 하되 위치 정보가 보조한다.
  - $\alpha_1 = \text{Conv}(\text{PositionPool}(\text{Input}))$ : 공간 차원에서 위치 상관관계를 학습한다.
  - $\alpha_2 = \text{CAM}(\text{Input})$ : Channel Attention Module(CAM)을 통해 상세 채널 특징을 추출한다.
  - 최종 출력: $\alpha = \text{Sigmoid}(\alpha_1) \cdot \alpha_2$

- **Part B (Residual Part)**: 정보 손실을 최소화하기 위한 잔차 경로이다.
  - $\omega_1 = \text{Conv}(\text{Part A's Input})$, $\omega_2 = \text{Conv}(\text{Part C's Input})$
  - $\omega = \text{Conv}(\omega_1 \cdot \omega_2)$

최종 출력은 세 파트의 합을 잔차 네트워크에 통과시켜 얻는다:
$$\text{Output} = \text{Residual}(\alpha + \beta + \omega)$$

### 3. GL-MIPC-Skip-Connections
단순한 Skip-connection의 정보 손실 문제를 해결하기 위해 두 가지 전략을 사용한다.
- **DA-Skip-Connections**: 모든 Skip-connection에 Dual Attention Block(DA-Block)을 배치하여 전송되는 특징 중 불필요한 중복 정보를 필터링한다.
- **GL-MIPC-Residue**: 디코더에서 업샘플링된 특징을 MIPC-Block으로 정제한 후, 이를 다시 Skip-connection에 통합하여 인코더-디코더 간의 통합성을 높이고 소실된 유효 정보를 복원한다.

### 4. 학습 절차 및 손실 함수
모델은 PyTorch 프레임워크를 사용하여 NVIDIA RTX 3090 GPU에서 학습되었다. 최적화 알고리즘으로는 SGD를 사용하였으며, 손실 함수는 클래스 불균형 문제를 해결하기 위해 Cross-Entropy Loss와 Dice Loss를 결합하여 사용하였다.
$$\text{Loss} = \frac{1}{2} \times \text{Cross-Entropy Loss} + \frac{1}{2} \times \text{DiceLoss}$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 
  - **Synapse**: 8개 복부 장기에 대한 3D CT 스캔 이미지 (다중 클래스 분할).
  - **ISIC2018-Task**: 피부 병변 분할 이미지 (2D 단일 클래스).
  - **Segpc**: 현미경 이미지 내 세포 분할 (2D 단일 클래스).
- **평가 지표**: Dice Coefficient (DSC), Hausdorff Distance (HD), Accuracy (AC), Precision (PR), Specificity (SP). 특히 경계 분할 능력을 평가하기 위해 HD를 주요 지표로 활용하였다.

### 2. 정량적 결과 및 분석
- **Synapse 데이터셋**: MIPC-Net은 평균 DSC 80.00%, 평균 HD 19.32mm를 기록하여 비교 대상인 12개 SOTA 모델 중 최고 성능을 보였다. 특히 TransUNet 대비 DSC는 2.52% 향상되었고, HD는 12.37mm 감소하였다. 췌장, 비장, 위와 같이 경계가 불분명한 장기에서 높은 성능 향상을 보였다.
- **ISIC 2018 및 Segpc 데이터셋**: 두 데이터셋 모두에서 AC, PR, SP, Dice 지표 전반에 걸쳐 우수한 성능을 기록하였다. Segpc의 경우 Dice 지표에서 0.8675를 기록하며 기존 모델들보다 세포 간 분리 능력이 탁월함을 입증하였다.

### 3. 절제 연구 (Ablation Study)
- **상호 포함 메커니즘의 효과**: 단순 결합 모델(PC-Net)보다 상호 포함 메커니즘을 적용한 MIPC-Net의 DSC가 0.91% 높고 HD가 4.02mm 낮아, 두 정보의 시너지 효과가 입증되었다.
- **GL-MIPC-Residue의 영향**: 특히 첫 번째 Skip-connection 층에 GL-MIPC-Residue를 적용했을 때 가장 높은 성능(DSC 80.00%, HD 19.32mm)을 보였는데, 이는 얕은 층의 저수준 공간 정보가 경계 묘사에 결정적이기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과
본 논문은 위치와 채널 주의 메커니즘을 독립적으로 사용하는 대신, 서로가 서로를 가이드하는 '상호 포함' 방식을 통해 의료 영상의 복잡한 경계선을 매우 정밀하게 포착할 수 있음을 보여주었다. 특히 HD 지표의 상당한 감소는 실제 임상에서 중요한 '정밀한 경계 획정' 능력이 크게 개선되었음을 의미한다. 또한, 계산 효율성이 TransUNet과 유사한 수준(38.51ms)이면서 성능은 더 높다는 점이 실용적이다.

### 2. 한계 및 미해결 과제
- **계산 복잡도**: MIPC-Block과 DA-Block의 추가로 인해 모델의 파라미터와 연산량이 증가하였다. 이는 자원이 제한된 환경이나 실시간 처리가 필요한 환경에서는 제약이 될 수 있다.
- **통합 방식의 한계**: 현재 구조는 Vision Transformer와 주의 메커니즘을 병렬적으로 결합한 형태이다. 두 구조 간의 더 깊은 수준의 융합(Deep integration)이 이루어진다면 추가적인 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 의료 영상의 정밀한 경계 분할을 위해 위치와 채널 주의 메커니즘을 상호 보완적으로 결합한 **MIPC-Net**을 제안한다. 방사선 전문의의 진단 패턴을 모사한 **Mutual Inclusion** 메커니즘과 인코더-디코더 통합을 강화하는 **GL-MIPC-Residue**를 통해, 기존 SOTA 모델들보다 월등한 경계 묘사 성능을 달성하였다. 특히 Synapse 데이터셋에서 HD 지표를 획기적으로 낮춤으로써 의료 영상 분할의 정밀도를 한 단계 높였으며, 이는 향후 정밀 진단 및 치료 계획 수립에 기여할 가능성이 크다.