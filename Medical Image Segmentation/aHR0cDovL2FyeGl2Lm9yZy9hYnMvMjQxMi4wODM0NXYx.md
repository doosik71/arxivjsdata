# ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement

Mengqi Lei, Haochen Wu, Xinhua Lv, Xin Wang (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 발생하는 두 가지 주요 도전 과제를 해결하고자 한다.

첫 번째는 **모호한 경계(Ambiguous Boundaries)** 문제이다. 의료 영상에서는 병변 조직과 주변 정상 조직 사이의 전이 영역으로 인해 전경(Foreground)과 배경(Background)의 경계가 불분명한 'Soft Boundary' 현상이 빈번하게 나타난다. 특히 조명 조건이 좋지 않거나 대비(Contrast)가 낮은 환경에서는 이러한 경계의 구분이 더욱 어려워져 분할 정확도가 저하된다.

두 번째는 **공동 발생(Co-occurrence)** 현상이다. 의료 영상에서는 특정 크기의 병변이 동시에 여러 개 나타나거나, 큰 병변 주변에 작은 병변이 함께 존재하는 등 일정한 규칙성을 가진 공동 발생 패턴이 자주 관찰된다. 기존 딥러닝 모델들은 병변 자체의 특징보다는 이러한 주변 맥락(Contextual associations)에 과도하게 의존하여 학습하는 경향이 있으며, 이로 인해 병변이 단독으로 나타날 경우 오작동하거나 잘못된 예측을 수행하는 문제가 발생한다.

결과적으로 본 논문의 목표는 다양한 조명 및 대비 환경에서도 강건하게 작동하며, 공동 발생 현상으로 인한 오학습을 방지하여 정확한 분할 성능을 제공하는 일반적인 의료 영상 분할 프레임워크인 **ConDSeg**를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 전경과 배경의 대비(Contrast)를 명시적으로 학습하고 이를 통해 특징을 강화하여 모호한 경계를 극복하고 공동 발생의 영향을 최소화하는 것이다. 이를 위해 다음과 같은 설계를 도입하였다.

1.  **Consistency Reinforcement (CR)**: 인코더가 다양한 조명 및 대비 환경에서도 고품질의 특징을 추출할 수 있도록, 원본 이미지와 강하게 증강된 이미지 간의 예측 일관성을 극대화하는 사전 학습 전략을 제안한다.
2.  **Semantic Information Decoupling (SID)**: 특징 맵을 전경, 배경, 그리고 불확실성(Uncertainty) 영역으로 분리하여, 학습 과정에서 불확실성 영역을 점진적으로 줄여나가는 메커니즘을 구현한다.
3.  **Contrast-Driven Feature Aggregation (CDFA)**: SID에서 분리된 전경과 배경 특징을 활용하여 다중 레벨 특징 융합을 가이드하고 주요 특징을 강화함으로써, 대상 객체와 복잡한 배경 간의 구별 능력을 높인다.
4.  **Size-Aware Decoder (SA-Decoder)**: 디코더의 스케일 특이성(Scale singularity) 문제를 해결하기 위해 크기별로 특화된 세 개의 디코더를 구성하여, 다양한 크기의 객체를 개별적으로 위치시키고 공동 발생 특징에 의한 간섭을 방지한다.

## 📎 Related Works

기존의 의료 영상 분할 연구들은 주로 U-Net 및 그 변형 구조(U-Net++, ResUNet 등)를 기반으로 인코더-디코더 구조를 개선하거나 Skip Connection을 통해 정보 손실을 줄이는 방향으로 진행되었다. 최근에는 Transformer 기반의 인코더(Swin-Unet, TransUNet 등)를 도입하여 글로벌 문맥 정보를 활용하려는 시도가 많았다.

특히 경계 모호성 문제를 해결하기 위해 경계 예측 전용 디코더를 추가하거나(SFA), 경계 민감 손실 함수를 사용하는 방식(BCNet, CFA-Net) 등이 제안되었다. 하지만 이러한 방법들은 명시적인 경계 감독(Supervision)에 의존할 뿐, 모델이 스스로 모호한 영역의 불확실성을 줄이는 근본적인 능력은 강화하지 못하며, 가혹한 환경에서의 강건성이 부족하다는 한계가 있다.

또한, 공동 발생 현상은 기존 연구에서 간과되는 경우가 많았다. 본 논문은 기존 모델들이 병변 자체의 특징보다 맥락적 관계에 의존함으로써 단독 객체 검출에 실패하는 점을 지적하며, 이를 해결하기 위한 구조적 분리(Size-Aware Decoder)와 대비 기반 강화(CDFA)의 필요성을 강조하며 차별점을 둔다.

## 🛠️ Methodology

ConDSeg는 크게 두 단계(Two-stage)의 학습 과정을 거치는 아키텍처이다.

### 1. Consistency Reinforcement (CR)
첫 번째 단계에서는 인코더의 강건성을 확보하기 위해 $Net_0$(인코더와 단순 예측 헤드)를 학습시킨다. 입력 이미지 $X$와 강하게 증강된 이미지 $Aug(X)$를 각각 입력하여 두 개의 마스크 $M_1, M_2$를 생성한다.

$$M_1 = Net_0(X), \quad M_2 = Net_0(Aug(X))$$

이때 $M_1$과 $M_2$가 정답(GT)과 유사해야 할 뿐만 아니라, 서로 간의 일관성을 가져야 한다. 이를 위해 다음과 같은 **Consistency Loss** ($L_{cons}$)를 제안한다. 이 함수는 확률 분포 기반의 KL/JS divergence 대신 픽셀 수준의 이진화(Binarization)를 통한 BCE Loss를 사용하여 수치적 불안정성을 피한다.

$$B(M, t) = \begin{cases} 1 & \text{if } M \ge t \\ 0 & \text{otherwise} \end{cases}$$
$$L_{cons}(M_1, M_2) = \frac{1}{2} \left( L_{BCE}(B(M_2, t), M_1) + L_{BCE}(B(M_1, t), M_2) \right)$$

최종 1단계 손실 함수는 다음과 같다.
$$L_{stage1} = L_{mask1} + L_{mask2} + L_{cons}$$

### 2. Semantic Information Decoupling (SID)
인코더의 최상위 특징 맵 $f_4$를 입력받아 전경($f_{fg}$), 배경($f_{bg}$), 불확실성($f_{uc}$)의 세 가지 특징 맵으로 분리한다. 각 맵은 보조 헤드를 통해 마스크 $M_{fg}, M_{bg}, M_{uc}$로 예측된다.

세 마스크는 상호 배타적이고 합이 1이 되는 보완적 관계여야 한다. 이를 위해 다음과 같은 **Complementarity Loss** ($L_{compl}$)를 설계하였다.

$$L_{compl} = \frac{1}{N} \sum_{i=1}^N (M_{fg,i} \cdot M_{bg,i} + M_{fg,i} \cdot M_{uc,i} + M_{bg,i} \cdot M_{uc,i})$$

또한, 작은 객체의 예측 정확도를 높이기 위해 면적 비율에 기반한 동적 페널티 항 $\beta_1, \beta_2$를 BCE Dice Loss에 곱하여 학습의 안정성을 높였다.

### 3. Contrast-Driven Feature Aggregation (CDFA)
SID에서 추출된 $f_{fg}$와 $f_{bg}$를 가이드로 사용하여 다중 레벨 특징을 융합한다. $K \times K$ 윈도우 내에서 주변 특징을 집계하며, 전경과 배경의 대비 정보를 통해 어텐션 가중치를 생성한다.

입력 특징 맵 $F$에 대해, 전경 및 배경 어텐션 가중치 $A_{fg}, A_{bg}$를 생성하고 다음과 같이 가중치를 적용한 값 $\tilde{V}$를 계산한다.

$$\tilde{V}_{i,j}^\Delta = \text{Softmax}(\hat{A}_{fg,i,j}) \otimes (\text{Softmax}(\hat{A}_{bg,i,j}) \otimes V_{i,j}^\Delta)$$

여기서 $\otimes$는 행렬 곱셈을 의미하며, 이를 통해 전경과 배경의 대비 정보가 반영된 강화된 특징을 얻게 된다.

### 4. Size-Aware Decoder (SA-Decoder)
디코더를 소형($Decoder_s$), 중형($Decoder_m$), 대형($Decoder_l$) 세 가지로 분리하여 설계하였다. 각 디코더는 CDFA의 서로 다른 레벨에서 출력된 특징 맵을 입력받아 해당 크기의 객체를 독립적으로 예측한다. 최종 마스크는 이 세 디코더의 출력을 융합하여 생성한다. 이를 통해 특정 크기의 객체가 다른 객체와 함께 나타나는 공동 발생 패턴에 의존하지 않고, 크기별 특성에 맞게 객체를 위치시킬 수 있다.

### 5. 전체 학습 절차 및 손실 함수
2단계 학습에서는 인코더의 학습률을 낮게 설정하여 미세 조정(Fine-tuning)하며, 전체 네트워크를 다음과 같은 손실 함수로 최적화한다.

$$L_{stage2} = L_{mask} + \beta_1 L_{fg} + \beta_2 L_{bg} + L_{compl}$$

## 📊 Results

### 실험 설정
- **데이터셋**: Kvasir-SEG, Kvasir-Sessile, GlaS, ISIC-2016, ISIC-2017 (총 5개, 3가지 모달리티 포함)
- **평가 지표**: mIoU, mDSC, Recall, Precision
- **구현 세부사항**: ResNet-50 인코더, Adam 옵티마이저, 이미지 크기 $256 \times 256$

### 주요 결과
1.  **정량적 성능**: 모든 데이터셋에서 기존 SOTA 모델(TGANet, XBoundFormer 등)보다 우수한 성능을 기록하였다. 특히 Kvasir-SEG 데이터셋에서 mIoU 84.6, mDSC 90.5를 달성하였다.
2.  **수렴 속도**: CR 전략을 통한 2단계 학습을 진행했을 때, 단일 단계 학습보다 훨씬 빠른 수렴 속도와 더 높은 최종 성능을 보였다.
3.  **소거 연구(Ablation Study)**:
    - CR 전략은 인코더의 강건성을 유의미하게 향상시켰다.
    - SID와 CDFA 모듈을 추가했을 때 mIoU와 mDSC가 크게 상승하였으며, SA-Decoder까지 포함된 전체 구조에서 최적의 성능이 나타났다.
    - CDFA의 윈도우 크기 $K=3$일 때 가장 좋은 성능을 보였다.
    - 이진화 임계값 $t=0.5$에서 최적의 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상의 특수한 문제인 'Soft Boundary'와 'Co-occurrence'를 구조적/전략적으로 해결하려 했다는 점에서 강점이 있다. 단순히 레이어를 쌓는 것이 아니라, **Consistency Reinforcement $\rightarrow$ Semantic Decoupling $\rightarrow$ Contrast-Driven Aggregation $\rightarrow$ Size-Aware Decoding**으로 이어지는 논리적인 파이프라인을 구축하였다.

특히, 불확실성 영역($f_{uc}$)을 명시적으로 정의하고 이를 줄여나가는 $L_{compl}$ 설계는 모델이 모호한 경계 영역에서 스스로 확신을 갖도록 유도하는 효과적인 방법이다. 또한, SA-Decoder를 통해 스케일별 예측을 분리한 점은 맥락 의존적 오학습(Co-occurrence bias)을 방지하는 실질적인 해결책이 되었음을 Grad-CAM 시각화를 통해 입증하였다.

다만, 2단계 학습 과정은 추가적인 학습 시간을 요구하며, 인코더와 디코더 간의 학습률 차이를 설정하는 하이퍼파라미터 튜닝이 성능에 영향을 미칠 수 있다는 점이 고려되어야 한다. 또한, 3D 데이터셋(Synapse)으로 확장 실험을 진행했으나, 기본적으로 2D 슬라이스 기반으로 처리했다는 점은 완전한 3D 공간 정보를 활용하는 방식과는 차이가 있다.

## 📌 TL;DR

본 논문은 의료 영상의 모호한 경계와 공동 발생 문제를 해결하기 위한 **ConDSeg** 프레임워크를 제안한다. 강건한 특징 추출을 위한 **일관성 강화(CR)** 전략, 전경/배경/불확실성을 분리하는 **SID** 모듈, 대비 정보를 이용해 특징을 융합하는 **CDFA**, 그리고 크기별로 객체를 예측하는 **SA-Decoder**가 핵심이다. 5개의 의료 데이터셋에서 SOTA 성능을 달성하며 범용성을 입증하였으며, 특히 병변의 크기나 주변 환경에 구애받지 않는 정밀한 분할이 가능하다는 점이 중요하다. 향후 다양한 의료 영상 모달리티에 적용되어 실시간 진단 보조 도구로서의 활용 가능성이 높다.