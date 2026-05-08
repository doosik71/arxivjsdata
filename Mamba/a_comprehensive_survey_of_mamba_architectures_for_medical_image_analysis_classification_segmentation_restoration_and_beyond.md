# A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond

Shubhi Bansal et al. (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석(Medical Image Analysis) 분야에서 기존의 딥러닝 아키텍처들이 가진 한계점을 해결하기 위해 Mamba 아키텍처의 적용 현황과 가능성을 분석하는 것을 목표로 한다.

구체적으로 해결하고자 하는 문제는 다음과 같다:

1. **CNN(Convolutional Neural Networks)의 한계**: CNN은 지역적 특징 추출에는 뛰어나지만, 의료 영상의 복잡한 3D 구조나 장거리 의존성(Long-range dependencies)을 모델링하는 데 한계가 있다.
2. **Transformer의 한계**: Transformer는 전역적 문맥 파악 능력이 뛰어나지만, Attention 메커니즘의 계산 복잡도가 시퀀스 길이의 제곱에 비례하는 $O(L^2)$의 Quadratic complexity를 가진다. 이는 고해상도 의료 영상 데이터를 처리할 때 막대한 메모리와 계산 자원을 요구한다.
3. **의료 영상의 특수성**: 의료 데이터는 공간적, 시간적 관계가 매우 복잡하며, 데이터셋의 규모가 제한적인 경우가 많아 효율적이면서도 강력한 표현 학습 능력을 갖춘 모델이 필요하다.

따라서 본 논문의 목표는 선형 시간 복잡도($O(L)$)를 가지면서도 Transformer에 필적하는 성능을 보이는 Mamba(Selective State Space Model) 아키텍처를 의료 영상 분석의 다양한 태스크(분류, 분할, 복원, 등록 등)에 어떻게 적용할 수 있는지 체계적으로 조사하고 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba 아키텍처를 의료 영상 분석 관점에서 종합적으로 정리한 최신 서베이 보고서를 제공하는 것이며, 주요 내용은 다음과 같다:

- **Mamba의 진화 과정 분석**: SSM(State Space Models)에서 시작하여 S4, S5를 거쳐 Mamba(S6)로 진화하는 이론적 배경과 구조적 차이점을 상세히 설명한다.
- **아키텍처 분류 체계(Taxonomy) 제시**: 의료 영상에 적용된 Mamba 모델을 Pure Mamba, U-Net 변형 모델, Hybrid 아키텍처(CNN, Transformer, GNN 결합형)로 분류하여 분석한다.
- **스캐닝 메커니즘(Scanning Mechanisms) 분석**: 1D 시퀀스 모델인 Mamba를 2D/3D 의료 영상에 적용하기 위한 다양한 스캐닝 방식(Bidirectional, SS2D, Local scan 등)을 체계화한다.
- **다양한 의료 태스크 적용 사례 정리**: Image Segmentation, Classification, Restoration, Registration 등 주요 태스크별 최신 모델과 성능 지표를 정리하여 제공한다.
- **실무적 리소스 제공**: 의료 영상 분석에 사용되는 주요 데이터셋 목록과 Mamba 관련 논문들의 GitHub 저장소를 정리하여 향후 연구의 가이드라인을 제시한다.

## 📎 Related Works

논문에서는 Mamba의 근간이 되는 State Space Models(SSMs)의 발전 과정을 관련 연구로 다룬다.

1. **SSM의 발전 경로**:
   - **S4 (Structured State Space Sequence Models)**: 효율적인 계산을 위해 bilinear discretization을 도입하고 HiPPO 행렬을 통해 장거리 의존성 문제를 해결하였다.
   - **S5 (Simplified State Space Layers)**: MIMO(Multiple Input Multiple Output) 접근 방식과 학습 가능한 타임 스케일 파라미터 $\Delta$를 도입하여 S4를 개선하였다.
   - **S6 (Mamba)**: 입력 데이터에 따라 파라미터가 변하는 Selection mechanism을 도입하여 하드웨어 최적화(SRAM/HBM 활용)를 통해 계산 효율성과 성능을 동시에 잡았다.

2. **기존 서베이와의 차별점**:
   - 기존의 Mamba 서베이들은 일반적인 비전(Vision) 도메인이나 광범위한 프레임워크에 집중되어 있었다.
   - 본 논문은 **의료 영상 도메인**에 특화되어, 의료 데이터셋 분석, 구체적인 의료 태스크별 실험 결과, 그리고 의료 환경에서의 실제 적용 가능성을 심층적으로 다룬다는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 특정 알고리즘을 제안하기보다, Mamba의 핵심 작동 원리와 의료 영상에 적용되는 구조적 방법론을 설명한다.

### 1. State Space Models (SSM)의 기본 원리

SSM은 1차원 입력 시퀀스 $u(t)$를 $N$차원 잠재 상태 $x(t)$로 변환하고, 이를 다시 1차원 출력 $y(t)$로 투영하는 시스템이다.

$$ \dot{x}(t) = Ax(t) + Bu(t) $$
$$ y(t) = Cx(t) + Du(t) $$

여기서 $A, B, C, D$는 시스템 파라미터이다. 디지털 컴퓨터에서 처리하기 위해 스텝 사이즈 $\Delta$를 이용한 이산화(Discretization) 과정이 필요하며, 이를 통해 순환(Recurrent) 형태나 합성곱(Convolutional) 형태로 변환하여 계산할 수 있다.

### 2. Mamba (S6)의 핵심: Selection Mechanism

Mamba의 가장 큰 특징은 파라미터 $B, C$와 스텝 사이즈 $\Delta$가 고정된 값이 아니라, **입력 $x$에 대한 함수**로 정의된다는 점이다.

- **Selective Scan**: 입력 데이터에 따라 어떤 정보를 유지하고 어떤 정보를 버릴지를 동적으로 결정한다.
- **Hardware-aware Algorithm**: GPU의 SRAM을 효율적으로 사용하여 중간 계산 결과를 저장함으로써, 계산 복잡도를 선형 시간 $O(L)$으로 유지하면서 빠르게 처리한다.

### 3. 의료 영상 적용을 위한 주요 구조 및 최적화

- **Scanning Mechanism**: Mamba는 본래 1D 시퀀스 모델이므로, 2D/3D 영상을 처리하기 위해 영상을 특정 방향으로 훑는 스캐닝 기법이 필수적이다.
  - **Bidirectional Scan**: 전방 및 후방으로 스캔하여 문맥을 파악한다.
  - **SS2D (Selective Scan 2D)**: 상$\to$하, 하$\to$상, 좌$\to$우, 우$\to$좌의 4방향 스캔 후 결과를 병합한다.
- **Hybrid Architecture**:
  - **CNN + Mamba**: CNN의 지역적 특징 추출 능력과 Mamba의 전역적 의존성 모델링 능력을 결합한다 (예: U-Mamba).
  - **U-Net Variants**: U-Net의 Encoder/Decoder 혹은 Skip connection 부분에 Mamba 블록을 삽입하여 성능을 높인다 (예: VM-UNet).
- **Learning Strategies**: 데이터가 부족한 의료 분야의 특성을 고려하여 Weakly Supervised, Semi-Supervised, Self-Supervised Learning 기법을 Mamba와 결합하여 적용한다.

## 📊 Results

본 논문은 다양한 벤치마크 데이터셋을 통한 Mamba 기반 모델들의 성능을 정량적으로 비교 분석한다.

### 1. Medical Image Segmentation (영상 분할)

- **지표**: Dice Similarity Coefficient (DSC), Hausdorff Distance (HD95), Accuracy (ACC) 등이 사용되었다.
- **결과**:
  - **SegMamba**는 3D CT 스캔에서 Transformer 기반 U-Net보다 우수한 성능을 보였다.
  - **UltraLight VM-UNet**은 매우 적은 파라미터(0.049M)만으로도 피부 병변 분할에서 경쟁력 있는 성능을 입증하여 경량화 가능성을 보여주었다.
  - **LKM-UNet**은 2D/3D 복부 장기 분할에서 기존 Mamba 기반 모델들보다 뛰어난 성능을 기록하였다.

### 2. Medical Image Classification (영상 분류)

- **지표**: Accuracy (Acc), F1-score, AUC 등이 사용되었다.
- **결과**:
  - **nnMamba**는 알츠하이머 예측(ADNI 데이터셋)에서 높은 정확도를 보였다.
  - **Vim4Path**는 조직 병리 영상(Whole Slide Images) 분류에서 ViT보다 적은 계산량으로도 대등하거나 더 나은 성능을 보였으며, 병리학자의 진단 워크플로우를 더 잘 모사하는 것으로 나타났다.

### 3. Medical Image Registration (영상 등록)

- **결과**: **VMambaMorph**가 기존의 MambaMorph보다 Dice 계수와 HD95 지표에서 더 우수한 정렬 성능을 보였으며, 계산 시간과 메모리 사용량 측면에서도 효율적임이 확인되었다.

## 🧠 Insights & Discussion

### 강점 (Strengths)

- **효율성**: Transformer의 $O(L^2)$ 복잡도를 $O(L)$로 낮추어, 고해상도 3D 의료 영상이나 긴 시퀀스의 데이터를 처리하는 데 압도적인 이점이 있다.
- **표현력**: CNN의 지역적 특징과 Transformer의 전역적 특징을 모두 잡을 수 있는 잠재력을 가지고 있으며, 특히 하이브리드 구조를 통해 의료 영상의 특수성을 잘 반영할 수 있다.

### 한계 및 미해결 과제 (Limitations)

- **공간 정보 손실**: 2D/3D 데이터를 1D 시퀀스로 변환하여 스캔하는 과정에서 본질적인 공간적 구조 정보가 일부 손실될 수 있다.
- **이론적 근거 부족**: NLP 분야에서는 Mamba의 작동 원리가 잘 알려져 있으나, 비전(Vision) 및 의료 영상 태스크에서 왜 잘 작동하는지에 대한 이론적 분석이 아직 부족하다.
- **인과성(Causality) 문제**: Mamba의 스캔 메커니즘은 본래 인과적(Causal)인 시퀀스 처리를 위해 설계되었으나, 영상 데이터는 비인과적(Non-causal) 특성을 가지므로 이를 완벽하게 적응시키는 것이 어렵다.

### 비판적 해석

Mamba는 분명히 계산 효율성 측면에서 혁신적이지만, 단순히 "Transformer를 대체한다"는 관점보다는 "어떤 태스크에서 어떤 스캐닝 전략을 사용해야 최적인가"에 대한 연구가 더 필요하다. 또한, 최근 등장한 **Mamba-2**나 **xLSTM**과 같은 모델들이 기존 Mamba의 한계(병렬 처리 효율, 메모리 갱신 문제 등)를 어떻게 해결하고 의료 영상에 적용될 수 있을지가 향후 핵심 쟁점이 될 것이다.

## 📌 TL;DR

본 논문은 의료 영상 분석 분야에서 Transformer의 고비용 계산 문제를 해결할 대안으로 떠오른 **Mamba(Selective SSM)** 아키텍처를 종합적으로 분석한 서베이 논문이다. Mamba는 **선형 시간 복잡도 $O(L)$**를 유지하면서도 강력한 전역 문맥 파악 능력을 갖추어, 고해상도 의료 영상의 분할, 분류, 복원 및 등록 태스크에서 기존 CNN 및 Transformer 기반 모델보다 효율적이고 우수한 성능을 보임을 입증하였다. 특히 다양한 **스캐닝 메커니즘**과 **하이브리드 구조**를 통해 의료 영상의 특성을 반영하는 방법론들을 제시하였으며, 이는 향후 실시간 의료 진단 시스템이나 초고해상도 의료 영상 분석 연구에 중요한 기초 자료가 될 것으로 평가된다.
