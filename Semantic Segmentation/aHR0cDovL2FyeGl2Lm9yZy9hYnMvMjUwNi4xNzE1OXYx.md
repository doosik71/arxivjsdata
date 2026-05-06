# Co-Seg++: Mutual Prompt-Guided Collaborative Learning for Versatile Medical Segmentation

Qing Xu, Yuxiang Luo, Wenting Duan, Zhen Chen (2025)

## 🧩 Problem to Solve

의료 영상 분석에서는 장기나 조직을 구분하는 **Semantic Segmentation**과 개별 객체(예: 세포 핵, 치아)를 분리하는 **Instance Segmentation**을 동시에 수행해야 하는 경우가 많다. 예를 들어, 조직 병리 이미지에서는 조직 영역을 먼저 구분하고 그 내부의 개별 핵을 분리해야 하며, 치과용 CBCT 영상에서는 턱뼈와 같은 해부학적 구조와 개별 치아를 동시에 식별해야 한다.

기존 연구들은 이러한 서로 다른 세그멘테이션 작업을 독립적으로 처리하거나, 단순하게 인코더만 공유하는 방식을 채택해 왔다. 그러나 이러한 방식은 두 작업 사이의 근본적인 상호 의존성(Interdependencies)을 간과한다. 조직의 특성이 핵의 위치를 결정하고, 핵의 분포가 조직의 종류를 암시하는 상호 보완적인 관계가 있음에도 불구하고, 독립적인 최적화는 중복된 특징 추출과 낮은 일관성으로 인해 최적의 성능을 내지 못하는 한계가 있다. 본 논문의 목표는 semantic과 instance 세그멘테이션이 서로를 강화할 수 있는 협력적 학습 프레임워크인 **Co-Seg++**를 제안하여 다목적 의료 영상 세그멘테이션(Versatile Medical Segmentation) 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 semantic과 instance 세그멘테이션 간의 **폐쇄 루프 양방향 상호작용(Closed-loop bidirectional interaction)**을 통해 두 작업을 공동으로 최적화하는 **Co-segmentation Paradigm**을 도입하는 것이다.

이를 위해 다음과 같은 설계를 제안한다:

1. **Spatio-Temporal Prompt Encoder (STP-Encoder)**: 이미지 임베딩과 영역 마스크로부터 공간적·시간적 제약 조건을 추출하여 두 작업의 디코딩을 가이드하는 사전 공간 제약(Prior spatial constraints)을 생성한다.
2. **Multi-Task Collaborative Decoder (MTC-Decoder)**: 교차 가이드(Cross-guidance) 메커니즘을 통해 두 작업의 문맥적 일관성을 강화하고, 최종적으로 semantic과 instance 마스크를 공동으로 예측한다.
3. **상호 최적화 전략**: 두 작업의 확률 분포를 정렬하고 상호 정보를 활용함으로써, 한 작업의 예측 결과가 다른 작업의 정확도를 높이는 상호 보완적 학습 구조를 구축한다.

## 📎 Related Works

### 기존 연구 및 한계

- **전통적 세그멘테이션**: U-Net 기반의 CNN, Vision Transformer(ViT), 그리고 최근의 Mamba 모델들이 제안되었다. 하지만 대부분 특정 작업(Semantic 또는 Instance)에 특화되어 설계되었다.
- **Versatile Segmentation 접근법**: TIAToolbox와 같은 앙상블 방식이나 Cerberus와 같은 인코더 공유 방식이 존재한다. 인코더 공유 방식은 효율성은 높지만, 디코더 단계에서 작업 간의 상호작용이 부족하여 문맥적 지식을 충분히 활용하지 못한다.
- **Prompt-based Segmentation**: SAM(Segment Anything Model)과 그 파생 모델들이 프롬프트를 통해 유연한 세그멘테이션을 가능하게 했으나, 이는 주로 단일 작업의 성능 향상에 집중되어 있으며 서로 다른 세그멘테이션 작업 간의 상호 보완성을 활용하는 구조는 아니다.

### 차별점

Co-Seg++는 단순한 특징 공유를 넘어, 두 작업이 서로에게 프롬프트를 제공하는 협력적 구조를 통해 상호 최적화를 달성한다는 점에서 기존의 독립적 또는 인코더 공유 방식과 차별화된다.

## 🛠️ Methodology

### 1. Co-Segmentation Paradigm

본 논문은 semantic segmentation 결과 $y_1$과 instance segmentation 결과 $y_2$가 상호 의존적이라는 가정하에, 다음과 같은 결합 확률 모델을 정의한다:

$$p(y_1, y_2 | x, \theta_1, \theta_2) = p(y_1 | x, \theta_1) \cdot p(y_2 | y_1, x, \theta_2) = p(y_2 | x, \theta_2) \cdot p(y_1 | y_2, x, \theta_1)$$

이는 $y_1$을 알면 $y_2$를 예측하는 데 도움이 되고, 반대로 $y_2$를 알면 $y_1$의 예측 정확도가 높아짐을 수학적으로 의미한다. 이를 위해 그래디언트 계산 시 각 작업의 직접적인 손실뿐만 아니라 상호 정보(Mutual Information)와 작업 간 의존성을 반영한 항을 추가하여 공동 최적화를 수행한다.

### 2. Spatio-Temporal Prompt Encoder (STP-Encoder)

STP-Encoder는 이미지 특징과 마스크 로짓을 입력받아 사전 공간 제약 $c_i$를 생성한다.

- **Temporal Branch**: 이미지 임베딩 $h$를 입력으로 하여 $\text{Linear}_{\text{down}} \rightarrow \text{Conv1D} \rightarrow \text{SSM(State Space Model)} \rightarrow \text{Linear}_{\text{up}}$ 과정을 거쳐 시간적(순차적) 특징 $t$를 추출한다.
$$t = \text{Linear}_{\text{up}}(\text{LN}(\text{SSM}(\text{Conv1D}(\text{Linear}_{\text{down}}(h)))))$$
- **Spatial Branch**: 작업별 마스크 로짓 $m_i$를 $\text{Conv2D} \rightarrow \text{GELU} \rightarrow \text{LN} \rightarrow \text{Conv2D} \rightarrow \text{Self-Attention}$ 과정을 통해 공간적 특징 $s_i$를 추출한다.
$$s_i = \text{SA}(\text{Conv2D}_2(\text{GELU}(\text{LN}(\text{Conv2D}_1(m_i)))))$$
- **Integration**: Cross-Attention을 통해 $s_i$와 $t$를 결합하여 최종 제약 조건 $c_i$를 생성한다.
$$c_i = \text{CrossAttention}(Q(s_i), K(t), V(t))$$

### 3. Multi-Task Collaborative Decoder (MTC-Decoder)

MTC-Decoder는 양방향 상호작용을 통해 마스크를 정교화한다.

- **Cross-Guidance**: 각 작업의 쿼리 $q_i$는 자신의 셀프 어텐션 이후, 상대 작업의 제약 조건 $c_j$와 이미지 임베딩 $h$의 합($h \oplus c_j$)을 이용해 Cross-Attention을 수행한다.
$$q'_i = \text{CrossAttention}(Q(\text{SA}(q_i)), K(h \oplus c_j), V(h \oplus c_j))$$
이후 역방향 Cross-Attention을 통해 작업 특화 임베딩 $g_i$를 생성한다.
- **Spatial Consistency Constraint**: 두 작업의 예측 확률 분포 $y^{\text{prob}}_i$와 $y^{\text{prob}}_j$ 사이의 일관성을 위해 KL-Divergence 기반의 손실 함수 $L_{SCC}$를 도입한다.
$$L_{SCC} = E_{x \sim y^{\text{prob}}_i(x)} [\log_2 y^{\text{prob}}_i(x) - \log_2 y^{\text{prob}}_j(x)]$$

### 4. Training 및 Loss Function

전체 프레임워크는 **Hiera ViT**를 공유 인코더로 사용하며, 효율적인 튜닝을 위해 **Adapter**를 삽입하였다. 학습 프로세스는 두 번의 Forward pass로 구성된다:

1. **Forward 1**: 프롬프트 없이 초기 이진 마스크 $y^{\text{bin}}$을 생성하고, 이를 STP-Encoder에 전달하여 제약 조건 $c_i, c_j$를 생성한다.
2. **Forward 2**: 생성된 제약 조건을 MTC-Decoder에 입력하여 최종 semantic 및 instance 마스크를 생성한다.

**최종 손실 함수**:
$$L_{\text{CoSeg}} = \lambda_1(L_{SCC} + L_{BM}) + L_{\text{sem\_SEG}} + L_{\text{ins\_SEG}}$$
여기서 $L_{BM}$은 이진 마스크의 BCE 손실이며, $L_{\text{sem\_SEG}}$는 Dice + CE 손실, $L_{\text{ins\_SEG}}$는 Dice + Focal + MSE + MSGE 손실의 조합이다.

## 📊 Results

### 실험 설정

- **데이터셋**: 조직 병리(PUMA, GlaS, CRAG) 및 치과 CBCT(ToothFairy2).
- **평가 지표**: Semantic(Dice, mIoU, HD), Instance(F1, AJI), Panoptic(PQ).

### 주요 결과

- **Semantic Segmentation**: 모든 데이터셋에서 SOTA 성능을 달성하였다. 특히 PUMA 데이터셋의 Stroma(기질) 세그멘테이션에서 기존 최고 모델인 Zig-RiR 대비 Dice 기준 5.54%의 큰 향상을 보였다.
- **Instance Segmentation**: 조직 내 다양한 세포 핵 종류 및 치아 분리 작업에서 일관되게 높은 성능을 보였다. PUMA의 Tumor instance 세그멘테이션에서 PathoSAM 대비 F1 score가 3.71% 향상되었다.
- **Panoptic Segmentation**: semantic과 instance 능력을 동시에 요구하는 파놉틱 작업에서도 우수한 성능을 보였으며, 특히 CBCT 치아 세그멘테이션에서 PQ가 최대 3.68% 증가하였다.

### Ablation Study

STP-Encoder(P), MTC-Decoder(D), Co-segmentation Paradigm(C)의 기여도를 분석한 결과:

- 세 가지 구성 요소가 모두 포함되었을 때 최적의 성능(Average PQ 54.25%)을 보였다.
- 특히 **Co-segmentation Paradigm(C)** 단독 도입 시 PQ가 평균 1.61% 향상되어 가장 지배적인 기여를 함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 의료 영상에서 서로 다른 세그멘테이션 작업이 단순한 병렬 관계가 아니라, 강력한 **상호 보완적 관계**에 있음을 입증하였다. 특히 조직 영역(Semantic)과 개별 핵(Instance)의 관계를 프롬프트 기반의 협력 학습으로 풀어낸 점이 인상적이다.

**강점**:

- 단순한 아키텍처 공유를 넘어 확률론적 근거를 바탕으로 한 협력 학습 프레임워크를 제안하였다.
- Hiera ViT와 Adapter를 사용하여 파라미터 효율성을 유지하면서도 높은 성능을 달성하였다.
- 다양한 모달리티(병리 이미지, CBCT)에서 범용적인 성능 향상을 검증하였다.

**한계 및 논의**:

- 두 번의 Forward pass를 거쳐야 하므로, 단일 pass 모델에 비해 추론 시간이 증가할 가능성이 있다.
- 논문에서 구체적인 추론 시간(Inference time)이나 FPS에 대한 정량적 분석은 명시되지 않았다.

## 📌 TL;DR

Co-Seg++는 의료 영상의 semantic 및 instance 세그멘테이션이 서로를 가이드할 수 있도록 설계된 **상호 프롬프트 기반 협력 학습 프레임워크**이다. STP-Encoder와 MTC-Decoder를 통해 두 작업 간의 공간적·문맥적 의존성을 학습함으로써, 기존의 독립적 모델이나 단순 인코더 공유 모델보다 뛰어난 성능을 보였다. 이 연구는 다목적 의료 영상 분석에서 작업 간 시너지를 활용하는 새로운 패러다임을 제시하였으며, 향후 정밀 의료 진단 및 자동 분석 시스템의 정확도를 높이는 데 중요한 역할을 할 것으로 기대된다.
