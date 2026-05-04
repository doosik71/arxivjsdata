# TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba

Xiaowen Ma, Zhenliang Ni, Xinghao Chen (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 Mamba 모델이 가지는 선형 복잡도(linear complexity)의 이점에도 불구하고, 경량화된 Mamba 기반 백본(backbone)들이 기존의 Convolution이나 Transformer 기반 모델들에 비해 경쟁력 있는 성능을 보여주지 못하는 문제를 해결하고자 한다.

기존의 Vision Mamba 연구들은 주로 이미지 도메인에서 스캐닝 경로(scanning path)를 수정하는 방식에 집중했으나, 저자들은 이러한 접근법만으로는 Mamba의 잠재력을 완전히 활용할 수 없다고 판단한다. 특히, Mamba 블록이 하이브리드 구조 내에서 주로 저주파(low-frequency) 정보만을 모델링하고 고주파(high-frequency) 정보를 무시하는 경향이 있어, 이로 인해 세밀한 인식 능력이 저하되고 연산 효율성이 떨어진다는 점을 핵심 문제로 지적한다. 따라서 본 연구의 목표는 주파수 분리(frequency decoupling)를 통해 Mamba의 효율성을 극대화하고 고주파 세부 정보를 보존하는 경량 하이브리드 Vision Mamba인 TinyViM을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **주파수 분리(Frequency Decoupling)**를 통해 각 주파수 성분에 최적화된 연산 장치를 배치하는 것이다.

1. **Mamba의 저주파 선호성 확인**: 스펙트럼 분석 및 정량적 분석을 통해, Convolution-Mamba 하이브리드 구조에서 Mamba 블록이 주로 저주파 성분을 캡처하고 고주파 성분을 억제한다는 것을 증명하였다.
2. **Laplace Mixer 제안**: Laplace 피라미드 구조를 이용해 특징 맵을 고주파와 저주파로 분리하고, 저주파 성분만 Mamba 블록($SS2D$)에 입력하여 전역 문맥을 캡처하게 하며, 고주파 성분은 재매개변수화된 $3 \times 3$ Depth-wise Convolution($RepDW$-$3$)으로 처리하여 효율성과 표현력을 동시에 높였다.
3. **Frequency Ramp Inception 설계**: 네트워크의 얕은 층에서는 고주파 세부 정보가 더 필요하고, 깊은 층에서는 저주파 전역 정보가 더 중요하다는 직관에 기반하여, 층이 깊어질수록 저주파 브랜치에 할당되는 채널 비율($\alpha$)을 점진적으로 높이는 구조를 도입하였다.
4. **TinyViM 아키텍처 구축**: 위 요소들을 결합하여 이미지 분류, 객체 검출, 세그멘테이션 등 다양한 하위 작업에서 기존 경량 모델 및 타 Mamba 기반 모델 대비 우수한 성능과 월등히 높은 처리량(throughput)을 달성하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계

- **Efficient CNNs**: MobileNet 시리즈 등은 Depth-wise separable Convolution을 통해 효율성을 높였으나, 수용 영역(receptive field)의 제한으로 인해 전역적인 상호작용을 학습하는 데 한계가 있다.
- **Transformers**: 전역 어텐션을 통해 문제를 해결하려 했으나, 입력 해상도에 따른 이차 복잡도(quadratic complexity)로 인해 실시간 배포 시나리오에서 계산 비용이 너무 크다.
- **Vision Mamba (Vim, VMamba 등)**: SSM(State Space Model)의 선형 복잡도를 활용해 전역 문맥을 캡처하려 했으며, 최근 EfficientVMamba와 같이 dilated-based scanning을 통해 효율성을 높이려는 시도가 있었다. 하지만 이러한 방법들은 여전히 처리량이 낮고, 경량 백본으로서 CNN/Transformer 기반 모델 대비 경쟁력이 부족하다.

### TinyViM의 차별점

TinyViM은 단순히 스캔 경로를 수정하는 대신, **주파수 도메인에서의 분석**을 통해 Mamba가 저주파 모델링에 특화되어 있다는 점을 발견하고 이를 구조적으로 활용한다. 저주파 성분의 해상도를 낮추어 Mamba에 입력함으로써 연산량을 획기적으로 줄이는 동시에, 고주파 성분은 CNN으로 보존하는 전략을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조

TinyViM은 4단계(stage)로 구성된 멀티스케일 백본 구조를 가진다. 각 단계는 국부적 특징을 추출하는 **Local Block**과 전역 문맥을 캡처하는 **TinyViM Block**으로 구성된다.

1. **Local Block**: $3 \times 3$ Reparameterized Convolution과 FFN(Feed-Forward Network)을 통해 로컬 특징을 추출하고 채널을 믹싱한다.
2. **TinyViM Block**: 핵심 구성 요소인 **Laplace Mixer**와 FFN으로 구성되며, 여기서 주파수 분리와 전역 모델링이 수행된다.

### Laplace Mixer 상세 설계

입력 특징 맵 $X \in \mathbb{R}^{H \times W \times D}$가 주어졌을 때, 처리 과정은 다음과 같다.

1. **채널 분할**: 파티션 계수 $\alpha$를 사용하여 $X$를 저주파 입력 $X_l \in \mathbb{R}^{H \times W \times \alpha D}$와 고주파 입력 $X_h \in \mathbb{R}^{H \times W \times (1-\alpha)D}$로 나눈다.
2. **저주파 브랜치 처리**:
    - Average Pooling을 통해 해상도를 낮춘 저주파 성분 $X_{ll} \in \mathbb{R}^{\hat{H} \times \hat{W} \times \alpha D}$를 생성한다.
    - 이 $X_{ll}$을 $SS2D$ (2D Selective Scanning) 블록에 입력하여 전역 문맥을 캡처한 $\hat{X}_{ll}$을 얻는다.
    - 또한, $X_{ll}$을 Nearest Neighbor Upsampling하여 복원한 뒤, 원래의 $X_l$에서 빼줌으로써 $X_l$ 내의 고주파 성분 $X_{lh}$를 추출한다.
    $$\text{Equation: } X_{ll} = \text{Pool}(X_l), \quad X_{lh} = X_l - \text{Upsample}(X_{ll})$$
3. **고주파 브랜치 처리**:
    - 추출된 $X_{lh}$와 처음에 분리한 $X_h$를 채널 방향으로 결합하여 전체 고주파 입력 $X_{hh}$를 만든다.
    - 이를 재매개변수화된 $3 \times 3$ Depth-wise Convolution($Rep^3$)에 통과시켜 고주파 성분을 강화한 $\hat{X}_{hh}$를 얻는다.
    $$\text{Equation: } \hat{X}_{hh} = \text{Rep}^3(X_{hh})$$
4. **통합**: $\hat{X}_{ll}$과 $\hat{X}_{hh}$를 요소별로 합산한 후, $1 \times 1$ Convolution을 통해 최종적으로 융합한다.

### Frequency Ramp Inception

네트워크 깊이에 따라 고주파와 저주파의 중요도가 다르다는 점을 반영하여 $\alpha$ 값을 단계별로 다르게 설정한다.

- **설정 값**: Stage 1(0.25) $\to$ Stage 2(0.5) $\to$ Stage 3(0.5) $\to$ Stage 4(0.75)
- 초반부에는 고주파 브랜치에 더 많은 채널을 할당하여 세부 특징을 잡고, 후반부에는 저주파 브랜치 비중을 높여 전역 정보를 더 많이 수집하도록 유도한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ImageNet-1K (분류), MS-COCO 2017 (검출 및 인스턴스 분할), ADE20K (시맨틱 세그멘테이션).
- **비교 지표**: Top-1 Accuracy, GMACs, Throughput (images/s), mAP, mIoU.
- **하드웨어**: Nvidia V100 GPU에서 측정.

### 주요 결과

1. **이미지 분류 (ImageNet-1K)**:
    - **TinyViM-S**: Top-1 정확도 79.2%를 달성하여 SwiftFormer-S(78.5%)보다 높으며, 처리량은 2574 im/s로 매우 빠르다.
    - **EfficientVMamba-T**와 비교 시, 정확도는 2.7%p 높으면서 처리량은 약 2배 더 높다.
    - TinyViM-B(81.2%)와 TinyViM-L(83.3%) 역시 유사 규모의 CNN, ViT, Mamba 모델들을 능가하는 성능을 보였다.

2. **객체 검출 및 인스턴스 분할 (MS-COCO)**:
    - Mask R-CNN 프레임워크를 사용한 결과, TinyViM-B는 EfficientVMamba-S보다 $\text{AP}_{\text{box}}$에서 3.0, $\text{AP}_{\text{mask}}$에서 2.0 더 높은 성능을 보였으며 처리량 또한 1.7배 높았다.

3. **시맨틱 세그멘테이션 (ADE20K)**:
    - Semantic FPN을 디코더로 사용했을 때, TinyViM은 SwiftFormer 및 FasterViT 대비 더 높은 mIoU를 기록하며 우수한 성능을 입증하였다.

4. **Ablation Study**:
    - **주파수 분리 효과**: Mamba에 저주파만 입력했을 때($\alpha$ 기반 분리), 정확도는 거의 유지(79.1% $\to$ 79.0%)하면서 처리량이 1.5배 증가함을 확인하였다.
    - **Ramp Inception**: $\alpha$를 고정(0.5)하거나 모든 채널을 분리하는 것보다, 단계별로 $\alpha$를 조절하는 방식이 성능과 효율성의 최적의 균형을 제공함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

TinyViM의 가장 큰 성과는 Mamba 모델의 고질적인 문제인 **낮은 처리량(Throughput)**을 획기적으로 개선했다는 점이다. 이는 단순히 알고리즘적 최적화가 아니라, "Mamba는 저주파 모델링에 특화되어 있다"는 스펙트럼 분석 기반의 통찰을 통해, Mamba에 입력되는 토큰의 수를 물리적으로 줄이면서도(Pooling 이용) 정보 손실을 CNN으로 보완하는 구조적 설계를 통해 달성되었다.

### 한계 및 논의

- **하드웨어 의존성**: 논문에서는 V100에서 측정하였으나, A100 GPU에서는 Mamba의 Triton 커널 등 하드웨어 최적화 덕분에 효율성이 더욱 극대화될 것이라고 언급한다. 이는 모델의 성능이 소프트웨어 최적화 상태에 따라 크게 달라질 수 있음을 시사한다.
- **가정**: 본 모델은 Mamba가 저주파를 선호한다는 분석 결과를 바탕으로 설계되었는데, 이러한 경향성이 모든 종류의 이미지 데이터셋이나 모든 규모의 모델에서 동일하게 나타나는지에 대한 일반화 가능성 검증이 더 필요할 수 있다.

## 📌 TL;DR

TinyViM은 Mamba 블록이 저주파 성분을 주로 모델링한다는 점에 착안하여, **Laplace Mixer**를 통해 저주파는 Mamba가, 고주파는 CNN이 처리하도록 분리한 하이브리드 백본이다. 특히 층별로 주파수 비중을 조절하는 **Frequency Ramp Inception**을 도입하여 효율성을 극대화하였다. 결과적으로 기존 Mamba 기반 모델 대비 **처리량을 2~3배 높이면서도**, 유사 규모의 CNN/Transformer 모델보다 뛰어난 정확도를 달성하여 실시간 비전 애플리케이션에 매우 적합한 구조를 제시하였다.
