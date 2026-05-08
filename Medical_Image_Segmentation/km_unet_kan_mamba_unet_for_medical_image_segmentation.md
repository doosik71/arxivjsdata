# KM-UNet: KAN Mamba UNet for medical image segmentation

Yibo Zhang, Jingwen Zhao, Xiang Liu, Xian Tang, Yunyu Shi, Lina Wei, Guyue Zhang (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 컴퓨터 보조 진단 및 영상 유도 수술 시스템에서 매우 중요한 과제이다. 기존의 CNN 기반 방법론들은 수용 영역(Receptive Field)의 한계로 인해 장거리 의존성(Long-range dependencies)을 모델링하는 데 어려움을 겪는다. 이를 해결하기 위해 등장한 Transformer 기반 모델들은 전역적 문맥(Global context) 파악에는 능숙하지만, 연산 복잡도가 입력 크기의 제곱에 비례하는 Quadratic Complexity 문제를 가지고 있어 데이터가 제한적이거나 실시간 처리가 필요한 의료 환경에서 적용하기에 부담이 크다. 또한, 기존의 딥러닝 모델들은 대부분 '블랙박스(Black-box)' 특성을 지니고 있어, 진단 결과에 대한 신뢰성과 해석 가능성(Interpretability)이 요구되는 의료 분야에서 결정적인 한계를 보인다.

본 논문의 목표는 이러한 연산 효율성, 전역적 모델링 능력, 그리고 모델의 해석 가능성이라는 세 가지 문제를 동시에 해결하는 새로운 U-shaped 네트워크 구조인 KM-UNet을 제안하는 것이다.

## ✨ Key Contributions

KM-UNet의 핵심 아이디어는 Kolmogorov-Arnold Networks (KANs)의 해석 가능성과 State-Space Models (SSMs, 특히 Mamba)의 효율적인 장거리 모델링 능력을 U-Net 구조에 통합하는 것이다.

1. **KM-UNet 아키텍처 제안**: KAN과 SSM을 융합하여 효율적인 특성 표현과 확장 가능한 장거리 모델링 사이의 균형을 맞춘 의료 영상 분할 프레임워크를 최초로 제시하였다.
2. **SEM (Selective-Scan Efficient Multi-scale) Attention 모듈 도입**: SSM의 개념을 통합하여 공간 전반에 걸쳐 다중 스케일 정보를 학습할 수 있는 새로운 어텐션 메커니즘을 설계하였다.
3. **새로운 스캐닝 전략**: 외부 레이어에서 중심으로 회전하며 스캔하는 전략을 통해, 기존의 단순 스캐닝 방식으로는 포착하기 어려운 특징들을 효과적으로 추출하여 분할 정확도를 높였다.

## 📎 Related Works

기존의 의료 영상 분할은 U-Net의 Encoder-Decoder 구조와 Skip Connection을 통해 다중 스케일 특성을 캡처하는 방식이 주를 이루었다. 이후 U-Net++, 3D U-Net, V-Net 등이 등장하며 성능을 개선하였고, Transformer를 결합한 TransUNet이나 Swin-UNet 등이 전역적 문맥 파악 능력을 보여주었으나 높은 연산 비용이 문제로 지적되었다. 최근에는 Linear Complexity를 가진 State-Space Models (SSMs) 기반의 U-Mamba와 SegMamba가 등장하여 CNN과 SSM을 결합함으로써 효율성을 높이려 시도하였다.

본 논문은 여기서 더 나아가, 기존 SSM 기반 모델들이 여전히 가지고 있는 '블랙박스' 특성을 해결하기 위해 KAN을 도입한다. KAN은 전통적인 MLP의 선형 가중치 행렬을 학습 가능한 활성화 함수로 대체함으로써 모델의 투명성을 높이고 비선형 관계 모델링 능력을 강화한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

KM-UNet은 크게 **Convolution Phase**, **SEM Attention Module Phase**, **Tok-KAN Phase**의 세 단계로 구성된 3단계 Encoder-Decoder 구조를 가진다. Encoder에서는 SEM 모듈과 Patch Merging을 통해 해상도를 줄이며 특징을 추출하고, Decoder에서는 SEM 모듈과 Patch Expansion을 통해 원래 크기로 복원한다. Encoder와 Decoder 사이에는 단순 덧셈(Simple Addition) 방식의 Skip Connection이 적용되어 순수 SSM 모델의 성능을 강조한다.

### Selective-Scan Efficient Multi-scale (SEM) Attention Module

SEM 모듈은 특성 추출(Feature Extraction)과 어텐션(Attention)의 두 부분으로 나뉜다.

1. **특성 추출 (Feature Extraction)**:
    입력 특성 맵을 4가지 방향(좌상$\rightarrow$우하, 우상$\rightarrow$좌하, 우하$\rightarrow$좌상, 좌하$\rightarrow$우상)으로 전개하여 시퀀스로 변환한다. 각 방향의 스캔 결과는 입력값에 따라 파라미터를 동적으로 조정하는 S6 블록을 통과하며, 이후 Re-weight 연산을 통해 다시 원래 크기의 이미지로 병합된다.
    각 스캔 방향에 대한 표현식은 다음과 같다.
    $$x^{(dir)} = \text{Scan}(x, \text{direction})$$
    최종 특성 통합 과정은 다음과 같다.
    $$x' = \sum_{dir} \text{ReWeight}(\text{S6}(x^{(dir)}))$$

2. **다중 스케일 어텐션 (Multi-Scale Attention)**:
    채널 차원을 배치 차원으로 재구성(Reshape)한 후, 두 개의 병렬 컨볼루션 서브 네트워크를 사용하여 단거리 및 장거리 공간 의존성을 동시에 캡처한다. $1\times1$ 컨볼루션은 국소적 채널 상호작용을, $3\times3$ 컨볼루션은 더 넓은 공간적 관계를 모델링한다.
    $$X_{cross} = \text{Conv}_{1\times1}(X) + \text{Conv}_{3\times3}(X)$$
    최종적으로 $Y = \text{Softmax}(X_{cross})$를 통해 다중 스케일 정보가 통합된 특성 맵을 생성한다.

### Integration of KAN (Tok-KAN Phase)

KM-UNet은 모델의 병목(Bottleneck) 레이어에 KAN을 통합하여 해석 가능성과 비선형 모델링 능력을 높였다. KAN은 MLP의 선형 변환 행렬을 학습 가능한 파라미터화된 활성화 함수로 대체한다.

전통적인 MLP의 구조가 다음과 같다면:
$$\text{MLP}(z) = (W_{L-1} \circ \sigma \circ W_{L-2} \circ \dots \circ W_1 \circ \sigma \circ W_0)Z$$
KAN의 구조는 다음과 같이 표현된다:
$$\text{KAN}(z) = (\phi_{L-1} \circ \phi_{L-2} \circ \dots \circ \phi_1 \circ \phi_0)Z$$
여기서 각 $\phi_i$는 학습 가능한 활성화 함수이다. KM-UNet의 KAN 레이어는 다음과 같은 형태로 처리된다.
$$z' = \text{LN}(Z + \text{DwConv}(\Phi(Z)))$$
여기서 $\text{LN}$은 Layer Normalization을, $\text{DwConv}$는 Depth-wise Convolution을 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**: BUSI (유방 초음파), GlaS (위장관), CVC-ClinicDB (폴립), ISIC17 및 ISIC18 (피부 병변) 총 5종의 이종 데이터셋을 사용하였다.
- **구현 세부사항**: PyTorch 기반, NVIDIA RTX 4090 GPU 사용. Adam 옵티마이저, Binary Cross-Entropy(BCE)와 Dice Loss의 결합 손실 함수를 사용하였으며, Cosine Annealing 학습률 스케줄러(초기 $1e-4$, 최소 $1e-5$)를 적용하였다.
- **평가 지표**: Mean Intersection over Union (mIoU)와 Dice Similarity Coefficient (DSC/F1 Score)를 사용하였다.

### 정량적 결과

KM-UNet은 대부분의 데이터셋에서 SOTA 모델들보다 우수한 성능을 보였다.

- **정확도**: 평균 IoU $81.17\%$, F1 score $89.20\%$를 기록하며 U-Net, U-Net++, U-Mamba 등을 상회하였다.
- **효율성**: 파라미터 수는 $7.35\text{M}$, 연산량은 $17.66\text{ Gflops}$로, 특히 U-Mamba($86.3\text{M}$ params, $2087\text{ Gflops}$)에 비해 압도적으로 효율적이면서도 더 높은 정확도를 달성하였다.

### 정성적 결과 및 해석 가능성

KAN 레이어를 적용했을 때와 적용하지 않았을 때의 어텐션 맵을 비교한 결과, KAN을 통합한 모델이 타겟 경계선을 훨씬 더 정밀하게 찾아내며 Ground Truth 마스크와 높은 일치도를 보였다. 이는 KAN이 모델의 투명성을 높이고 주요 영역에 대한 집중력을 향상시킴을 시사한다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **효율성(SSM)과 해석 가능성(KAN)의 성공적인 결합**에 있다. 기존의 SSM 기반 모델들이 성능은 좋으나 내부 동작을 알 수 없었던 점을 KAN의 학습 가능한 활성화 함수 구조로 보완하였다. 또한, SEM 모듈의 4방향 스캐닝 전략은 의료 영상의 복잡한 해부학적 구조를 다각도에서 포착함으로써 단순한 CNN이나 Transformer보다 정교한 경계 추출이 가능함을 입증하였다.

다만, 논문에서 언급되었듯이 SEM 모듈의 연산 복잡도가 여전히 개선의 여지가 있다는 점이 한계로 남는다. 또한, 현재는 2D 이미지 분할에 집중되어 있으나, 향후 3D 의료 영상(CT, MRI)으로의 확장 가능성에 대해서는 구체적인 실험적 검증이 더 필요할 것으로 보인다.

## 📌 TL;DR

KM-UNet은 **KAN(Kolmogorov-Arnold Networks)의 해석 가능성**과 **SSM(Mamba)의 선형 연산 효율성**을 U-Net 구조에 통합한 새로운 의료 영상 분할 모델이다. 제안된 **SEM 어텐션 모듈**과 **다방향 스캐닝 전략**을 통해 기존 SOTA 모델들보다 적은 파라미터($7.35\text{M}$)와 연산량으로 더 높은 분할 정확도(평균 IoU $81.17\%$)를 달성하였다. 이 연구는 효율적이면서도 신뢰할 수 있는(해석 가능한) 의료 AI 시스템 구축을 위한 중요한 베이스라인을 제공한다.
