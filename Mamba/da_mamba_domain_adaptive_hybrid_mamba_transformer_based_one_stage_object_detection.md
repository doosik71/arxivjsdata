# DA-Mamba: Domain Adaptive Hybrid Mamba-Transformer Based One-Stage Object Detection

A. Enes Doruk, Hasan F. Ates (2025)

## 🧩 Problem to Solve

본 논문은 객체 탐지(Object Detection) 모델에서 발생하는 **Domain Shift** 문제를 해결하고자 한다. Domain Shift란 학습 데이터(Source Domain)와 실제 테스트 데이터(Target Domain) 간의 데이터 분포가 달라 성능이 급격히 저하되는 현상을 의미한다.

기존의 접근 방식들은 다음과 같은 한계를 가진다:

- **2D CNN 기반 모델**: Receptive Field(수용 영역)가 제한적이어서 long-range dependency(장거리 의존성)를 포착하는 능력이 부족하며, 이는 공간적 분포 변화가 큰 타겟 도메인으로의 적응을 어렵게 만든다.
- **Transformer 기반 모델**: Self-attention 메커니즘을 통해 전역적 관계를 잘 포착하여 도메인 정렬(Alignment)에 유리하지만, 시퀀스 길이에 따른 연산 복잡도가 이차 함수적으로 증가($O(N^2)$)하여 실제 배포 및 실시간 탐지 작업에 적용하기에 계산 비용이 너무 높다.
- **Hybrid 구조의 노이즈**: CNN과 Transformer/Mamba를 혼합한 구조에서는 국소적 특징 추출과 전역적 특징 추출 간의 불일치로 인해 도메인 적응 과정에서 특징 노이즈(Feature Noise)가 발생할 가능성이 크다.

따라서 본 연구의 목표는 Mamba 아키텍처의 선형 연산 복잡도와 전역 모델링 능력을 활용하여, 효율적이면서도 강력한 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 기반의 단일 단계(One-stage) 객체 탐지 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 State-Space Model(SSM)과 Transformer의 Attention 메커니즘을 결합한 하이브리드 구조를 통해 계산 효율성과 도메인 적응 능력을 동시에 확보하는 것이다. 주요 기여 사항은 다음과 같다:

1. **HDAMT (Hybrid Domain-Adaptive Mamba-Transformer) 백본**: Mamba 블록을 사용하여 공간 및 채널 차원에서 도메인 적응형 스캐닝을 수행하고, Cross-attention을 통해 소스-우세(Source-dominant) 및 타겟-우세(Target-dominant) 특징을 추출함으로써 두 도메인 간의 소프트 정렬(Soft Alignment)을 구현한다.
2. **Margin ReLU 가이드 엔트로피 지식 증류 (EKD)**: 하이브리드 구조에서 발생하는 노이즈를 억제하기 위해 Margin ReLU를 적용하여 불확실한 활성화 값을 제거하고, 엔트로피 기반의 지식 증류를 통해 소스와 타겟 간의 양방향 특징 전이를 촉진한다.
3. **엔트로피 가이드 랜덤 다층 섭동 (ERMP)**: Cross-attention이 소스 도메인에 과적합(Overfitting)되는 것을 방지하기 위해, 엔트로피 민감 게이팅 메커니즘과 확률적 섭동(Stochastic Perturbation)을 결합하여 모델의 일반화 성능을 높인다.

## 📎 Related Works

### 1. Domain Adaptive Object Detection (DAOD)

기존의 UDA 기반 객체 탐지는 크게 적대적 학습(Adversarial Training), 자가 학습(Self-training), 이미지-투-이미지 변환(Image-to-Image Translation)으로 나뉜다.

- **Two-stage 모델**: Faster R-CNN 기반 모델들은 Region Proposal 단계에서 인스턴스 수준의 정렬이 가능하여 성능이 좋지만, 추론 속도가 느리고 배포가 어렵다.
- **One-stage 모델**: SSD나 RetinaNet 기반 모델들은 속도가 빠르지만, 명시적인 인스턴스 수준의 표현이 부족하여 도메인 적응이 더 어렵다.

### 2. Transformer-based Detection

DETR과 같은 모델들은 전역 문맥 모델링 능력이 뛰어나 DA-DETR 등에서 도메인 불변 영역을 식별하는 데 활용되었다. 그러나 앞서 언급한 이차 복잡도 문제로 인해 자원 제한 환경에서의 실용성이 떨어진다.

### 3. Vision Mamba

최근 State Space Models(SSMs)를 기반으로 한 Mamba 아키텍처가 선형 복잡도로 전역 수용 영역을 확보할 수 있는 대안으로 제시되었다. Vision Mamba, VMamba 등이 제안되었으나, 이를 **도메인 적응(Domain Adaptation)** 문제에 적용한 연구는 본 논문이 처음이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 모델은 **SSD (Single Shot MultiBox Detector)** 프레임워크에 **HDAMT 백본**을 통합한 구조이다. 전체 파이프라인은 다음과 같이 구성된다:

- **백본**: 6단계 구조로, 앞의 2단계는 Convolutional Layer를 통해 저수준 특징을 추출하고, 나머지 4단계는 HDAMT 블록을 통해 전역적 특징을 추출한다.
- **특징 분기**: 학습 시 소스 특징($Z_s$), 타겟 특징($Z_t$), 그리고 Cross-attention으로 생성된 소스-우세 특징($Z_{s \to t}$)와 타겟-우세 특징($Z_{t \to s}$)의 4가지 분기를 운용한다.

### 2. HDAMT 블록 상세

HDAMT 블록은 다음 세 가지 브랜치로 구성된다:

- **DA Spatial SSM**: $3 \times 3$ Depthwise Convolution을 적용한 후 공간적 코사인 유사도(Spatial Cosine Similarity)를 계산하여 공간적 일관성을 높인다.
- **DA Channel SSM**: 채널을 4개 세그먼트로 나누어 소스와 타겟 도메인 간에 일부 채널을 선택적으로 교환(Channel Swapping)함으로써 도메인 불변 상관관계를 학습하게 한다.
- **Transformation Branch**: 위 두 SSM의 출력과 결합되어 최종 특징 표현을 형성한다.

### 3. 적대적 정렬 (Adversarial Alignment)

도메인 간의 분포 차이를 줄이기 위해 Gradient Reversal Layer(GRL)를 포함한 판별기(Discriminator)를 사용한다.

- **Local Alignment**: 3단계 특징에 픽셀 수준 판별기를 적용하여 Cross-Entropy Loss로 학습한다.
- **Global Alignment**: 6단계 특징에 시맨틱 수준 판별기를 적용하며, 클래스 불균형 문제를 해결하기 위해 **Focal Loss**를 사용한다.
- **손실 함수**:
$$L_{adv} = \sum_{i} CE(\hat{y}_{local}^i, y_{local}^i) + FL(\hat{y}_{global}, y_{global})$$

### 4. Margin ReLU 가이드 엔트로피 지식 증류 (EKD)

하이브리드 구조의 노이즈를 억제하기 위해 **Margin ReLU ($\sigma_m$)**를 도입한다. Margin 값은 Batch Normalization 통계량의 가우시안 분포 꼬리 부분을 이용하여 적응적으로 결정된다.

- **엔트로피 손실**:
$$H_{t \to ts} = -\sum_{k} \sigma_m(Z_{t \to s}(k)) \log(\sigma_m(Z_t(k)))$$
$$H_{t \to st} = -\sum_{k} \sigma_m(Z_t(k)) \log(\sigma_m(Z_{s \to t}(k)))$$
최종 엔트로피 손실 $L_{entropy}$는 2, 4, 6단계에서 계산된 값들의 평균으로 정의된다.

### 5. 엔트로피 가이드 랜덤 다층 섭동 (ERMP)

소스 도메인에 대한 과적합을 방지하기 위해, 학습 중 무작위로 특정 단계(2, 4, 6단계 중 하나)를 선택해 타겟 특징에 섭동을 가한다.

- **게이팅 어텐션**:
$$\text{Attn}_{gating} = (1 - \sigma(\gamma)) \cdot H_{t \to st} + \sigma(\gamma) \cdot H_{t \to ts}$$
- **특징 업데이트**:
$$\tilde{Z}_t = Z_t + \alpha \cdot (\text{Attn}_{gating} - Z_t)$$
여기서 $\alpha$는 섭동 강도를 조절하는 파라미터이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 소스 도메인으로 Pascal VOC를 사용하고, 타겟 도메인으로 Clipart1k, Watercolor2k, Comic2k를 사용한다.
- **지표**: mAP (mean Average Precision)를 측정한다.
- **비교 대상**: CNN 기반(TFD, I3Net 등), Transformer 기반(DA-DETR) 모델들과 비교한다.

### 2. 정량적 결과

DA-Mamba-B(Base 모델)는 모든 데이터셋에서 SOTA 성능을 달성하였다.

- **Clipart1k**: DA-Mamba-B는 **45.3% mAP**를 기록하여, CNN 기반 최우수 모델인 TFD(41.2%)와 Transformer 기반 DA-DETR(41.3%)을 모두 능가하였다.
- **Watercolor2k**: DA-Mamba-B는 **57.8% mAP**를 달성하여 TFD(55.0%) 대비 2.8%p 향상되었다.
- **Comic2k**: DA-Mamba-B는 **37.1% mAP**를 기록하여 타 모델들보다 우수한 성능을 보였다.

### 3. 모델 복잡도 및 효율성

| 모델 | Parameters (M) | FLOPs (G) | Inference Time (ms) |
| :--- | :---: | :---: | :---: |
| DA-Mamba-S | 62.3 | 8.3 | 30 |
| DA-Mamba-B | 91.4 | 12.1 | 48 |

### 4. 분석 결과

- **Ablation Study**: HDAMT 블록이 가장 큰 성능 향상을 가져왔으며, EKD $\rightarrow$ ERMP $\rightarrow$ ADL 순으로 추가적인 성능 이득이 확인되었다.
- **STD 특징의 효과**: 단순히 소스-타겟 특징($Z_s, Z_t$)을 사용하는 것보다, Cross-attention으로 생성된 소스-타겟 우세 특징($Z_{s \to t}, Z_{t \to s}$)을 사용할 때 mAP가 **42.6%에서 45.3%로 크게 상승**하였다. 이는 도메인 편향을 억제한 소프트 표현이 적응에 더 유리함을 시사한다.

## 🧠 Insights & Discussion

### 강점

- **효율적인 전역 모델링**: Mamba의 선형 복잡도를 활용하여 Transformer의 성능(전역 문맥 포착)을 유지하면서도 연산 비용을 획기적으로 줄였다.
- **노이즈 제어**: 하이브리드 구조의 고질적 문제인 특징 노이즈를 Margin ReLU와 엔트로피 기반의 지식 증류로 영리하게 해결하였다.
- **과적합 방지**: ERMP 모듈을 통해 특정 도메인(소스)에 매몰되지 않고 타겟 도메인으로의 일반화 능력을 강화하였다.

### 한계 및 논의사항

- **데이터셋 특성**: 실험에 사용된 데이터셋(Clipart, Watercolor 등)은 스타일 변환이 주를 이루는 합성 데이터셋이다. 실제 센서 데이터의 변화(예: 주간 $\to$ 야간, 맑음 $\to$ 비)와 같은 더 복잡한 Real-world Domain Shift 상황에서도 동일한 성능 향상이 있을지는 추가 검증이 필요하다.
- **Mamba 아키텍처의 의존성**: 본 연구는 MambaVision의 구조를 기반으로 하고 있어, Mamba 자체의 성능 개선이나 구조 변화에 따라 결과가 민감하게 반응할 수 있다.

## 📌 TL;DR

본 논문은 Mamba의 선형 연산 효율성과 전역 모델링 능력을 Unsupervised Domain Adaptation(UDA) 객체 탐지에 최초로 적용한 **DA-Mamba**를 제안한다. 하이브리드 Mamba-Transformer 백본(HDAMT), 노이즈 억제를 위한 Margin ReLU 기반 지식 증류(EKD), 그리고 과적합 방지를 위한 랜덤 섭동(ERMP)을 통해 기존 CNN 및 Transformer 기반 모델보다 뛰어난 성능과 효율성을 동시에 달성하였다. 이는 자원이 제한된 환경에서 고성능의 도메인 적응형 객체 탐지 시스템을 구축하는 데 중요한 기반이 될 것으로 보인다.
