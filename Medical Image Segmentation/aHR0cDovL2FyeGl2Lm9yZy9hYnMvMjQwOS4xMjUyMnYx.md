# Prompting Segment Anything Model with Domain-Adaptive Prototype for Generalizable Medical Image Segmentation

Zhikai Wei, Wenhui Dong, Peilin Zhou, Yuliang Gu, Zhou Zhao, and Yongchao Xu (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 발생하는 **단일 소스 도메인 일반화(Single-source Domain Generalization, SDG)** 문제를 해결하고자 한다. 

의료 영상 데이터는 촬영 장비의 특성, 운영자의 숙련도, 환자의 방사선 노출량 및 촬영 시간 등 다양한 요인으로 인해 데이터 분포의 차이, 즉 **도메인 시프트(Domain Shift)**가 빈번하게 발생한다. 이러한 분포 변화는 모델의 성능 저하를 야기하며, 특히 단 하나의 소스 도메인 데이터만으로 학습하여 본 적 없는 타겟 도메인(Unseen target domains)에서도 높은 성능을 유지해야 하는 SDG 설정은 매우 도전적인 과제이다.

논문의 목표는 거대 모델인 **Segment Anything Model (SAM)**의 강력한 분할 능력을 활용하면서, 의료 영상의 특성에 맞게 도메인 적응형 프롬프트를 자동으로 생성하고 인코더를 효율적으로 미세 조정(Fine-tuning)하여 일반화 능력을 극대화하는 **DAPSAM (Domain-Adaptive Prompt SAM)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM의 수동적인 프롬프트 입력 방식에서 벗어나, 소스 도메인에서 학습된 지식을 저장하는 **메모리 뱅크(Memory Bank)**를 통해 인스턴스별 최적의 프롬프트를 자동으로 생성하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Prototype-based Prompt Generator (PPG) 제안**: 소스 도메인 이미지들로부터 학습된 프로토타입을 저장하는 파라미터화된 메모리 뱅크를 도입하였다. 이를 통해 타겟 이미지에 대해 도메인-약결합(domain-weakly-correlated) 및 인스턴스-강결합(instance-strongly-correlated) 특성을 가진 적응형 프롬프트를 생성하여 일반화 성능을 높였다.
2.  **Generalized Adapter 설계**: SAM의 인코더를 효율적으로 튜닝하기 위해, 저수준 특징(Low-level features)을 중간 특징에 통합하고 채널 어텐션 필터(Channel Attention Filter)를 통해 불필요한 정보를 제거하는 새로운 어댑터 구조를 제안하였다.
3.  **SOTA 성능 입증**: 서로 다른 모달리티를 가진 두 가지 SDG 의료 영상 분할 벤치마크에서 기존의 CNN 기반 및 SAM 기반 방법론들을 뛰어넘는 성능을 달성하였다.

## 📎 Related Works

기존의 도메인 일반화 연구는 주로 이미지 수준의 스타일 증강(Style Augmentation)이나 특징 수준의 불변성 학습에 집중하였다. **MixStyle**이나 **CSDG**, **CCSDG**와 같은 방법들이 대표적이며, 최근에는 Vision Transformer(ViT) 기반 모델들이 Out-of-distribution(OOD) 일반화에 더 강건하다는 점이 밝혀졌다.

특히 **SAM**은 방대한 데이터를 통해 사전 학습되어 뛰어난 일반화 능력을 갖추고 있어 의료 영상 분야에도 적용되고 있다. **DeSAM**은 SAM의 디코더를 수정하여 마스크 생성과 프롬프트 임베딩을 분리하였으나, 인코더의 적응 능력을 충분히 활용하지 못했다는 한계가 있다. 또한, SAM의 성능은 프롬프트(점, 박스 등)의 설계에 크게 의존하는데, 이는 인간의 주관적인 판단과 상호작용이 필요하다는 실무적인 어려움이 있다. DAPSAM은 이러한 수동 프롬프트의 한계를 극복하고, 학습된 프로토타입을 통해 자동으로 적응형 프롬프트를 생성한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
DAPSAM은 고정된(Frozen) SAM 인코더에 **Generalized Adapter**를 추가하여 특징 추출 능력을 강화하고, 추출된 임베딩을 기반으로 **Prototype-based Prompt Generator (PPG)**를 통해 프롬프트를 생성하여 마스크 디코더(Mask Decoder)에 전달하는 구조를 가진다. 디코더는 전체 학습 가능(Fully trainable) 상태로 설정된다.

### 2. Generalized Adapter
의료 영상에서는 장기의 경계나 병변 부위와 같은 저수준 특징(Low-level features)이 매우 중요하다. 이를 위해 다음과 같은 절차를 거친다.

- **특징 융합**: ViT의 Patch Embedding 층에서 나온 초기 임베딩 $e_0$로부터 선형 층을 통해 저수준 특징 $F_{low}$를 추출하고, 이를 각 레이어의 중간 특징 $F$와 더해 융합 특징 $F_{fuse}$를 생성한다.
- **채널 필터링**: 일반화에 방해가 되는 중복 정보를 제거하기 위해 Global Average Pooling(GAP)과 Global Max Pooling(GMP)을 결합한 채널 어텐션 메커니즘을 적용한다.
$$F_{filtered} = F_{fuse} \otimes \sigma(GAP(F_{fuse}) + GMP(F_{fuse}))$$
여기서 $\sigma$는 시그모이드 함수이며, $\otimes$는 원소별 곱셈을 의미한다.
- **어댑터 적용**: 필터링된 특징은 기존의 Vanilla Adapter 구조(Down-projection $\rightarrow$ GELU $\rightarrow$ Up-projection)를 통해 최종 특징 $F'$로 변환된다.
$$F' = F + \text{MLP}_{up}(\text{GELU}(\text{MLP}_{down}(F_{filtered})))$$

### 3. Prototype-based Prompt Generator (PPG)
학습된 지식을 저장하고 활용하기 위해 파라미터화된 메모리 뱅크 $M \in \mathbb{R}^{N \times C}$를 도입한다.

- **인스턴스 프로토타입 추출**: 이미지 임베딩 $e_i$로부터 GAP와 GMP를 통해 인스턴스 수준의 프로토타입 $p_i$를 생성한다.
$$p_i = GAP(e_i) + GMP(e_i)$$
- **도메인 적응형 프로토타입 생성**: 메모리 뱅크에 저장된 프로토타입 $m_j$들과 $p_i$ 사이의 코사인 유사도 기반 가중치 $w_{i,j}$를 계산하여 적응형 프로토타입 $bp_i$를 생성한다.
$$bp_i = \sum_{j=1}^{N} w_{i,j} m_j, \quad w_{i,j} = \frac{\exp(\text{Sim}(p_i, m_j))}{\sum_{k=1}^{N} \exp(\text{Sim}(p_i, m_k))}$$
- **프롬프트 생성**: $bp_i$와 원래 임베딩 $e_i$ 사이의 유사도 맵 $A_i$를 생성하고, 이를 $[bp_i, A_i, e_i]$ 형태로 결합한 뒤 $1 \times 1$ 컨볼루션을 통해 최종 프롬프트를 생성한다.
$$\text{Prompt}_i = \text{Conv}_{1 \times 1}([bp_i, A_i, e_i])$$

### 4. 학습 목표 (Training Objective)
소스 도메인에서 모델을 학습시키기 위해 Cross Entropy Loss($L_{CE}$)와 Dice Loss($L_{Dice}$)를 결합한 손실 함수를 사용한다.
$$L = (1-\lambda)L_{CE} + \lambda L_{Dice}$$
여기서 $\lambda$는 두 손실의 균형을 맞추는 가중치(본 논문에서는 0.8로 설정)이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 
    - **Prostate dataset**: 6개 도메인(A~F)의 MRI 영상. Leave-one-out 방식으로 평가.
    - **RIGA+ dataset**: 5개 도메인의 안저(Fundus) 영상. 2개 도메인(BinRushed, Magrabia)을 소스로 사용하고 나머지 3개를 타겟으로 평가.
- **평가 지표**: Dice Similarity Coefficient (DSC).
- **구현 세부사항**: SAM의 ViT-B 버전 사용, Adapter Rank = 4, Optimizer = AdamW, Learning Rate = $5 \times 10^{-4}$.

### 2. 정량적 결과
- **Prostate 데이터셋**: DAPSAM은 평균 DSC **81.31%**를 기록하여, Baseline(78.87%) 및 DeSAM [grid](79.02%)보다 우수한 성능을 보였다. 특히 기존 CNN 기반 SOTA 모델들보다 유의미하게 높은 성능을 달성하였다.
- **RIGA+ 데이터셋**: 소스 도메인이 BinRushed일 때 **87.87%**, Magrabia일 때 **88.15%**의 평균 DSC를 기록하며, SAMed나 CCSDG 등 기존 방법론들을 앞질렀다.

### 3. 소거 연구 (Ablation Study)
- **구성 요소의 영향**: 저수준 특징 통합(LLFI) $\rightarrow$ 필터링(Filter) $\rightarrow$ PPG 순으로 추가할 때 성능이 단계적으로 향상됨을 확인하였다. 특히 PPG 모듈은 Baseline 대비 약 1.44%의 성능 향상을 가져왔다.
- **메모리 뱅크 크기 ($N$)**: $N=256$일 때 최적의 성능을 보였다. $N$이 너무 작으면 정보를 충분히 학습하지 못하고, 너무 크면 소스 도메인에 과적합(Overfitting)되어 일반화 성능이 떨어진다.

## 🧠 Insights & Discussion

**강점**:
본 연구는 SAM의 강력한 제로샷(Zero-shot) 능력을 유지하면서도, 의료 영상 특유의 도메인 시프트 문제를 해결하기 위한 효율적인 튜닝 전략을 제시하였다. 특히 사람이 직접 지정해야 하는 프롬프트의 의존성을 제거하고, 메모리 뱅크를 통해 데이터 기반의 적응형 프롬프트를 생성함으로써 실용성을 높였다. 또한, 저수준 특징의 통합과 채널 필터링을 통해 의료 영상 분할에 필수적인 경계 정보를 보존하고 노이즈를 제거한 점이 주효했다.

**한계 및 논의**:
메모리 뱅크의 크기 $N$에 따라 성능 편차가 발생하는 점은 하이퍼파라미터 최적화에 대한 부담이 있음을 시사한다. 또한, 본 논문에서는 단일 소스 도메인(SDG) 설정만 다루었으나, 실제 의료 현장에서는 여러 기관의 데이터를 동시에 사용할 수 있는 다중 소스(Multi-source) 설정에서의 확장 가능성에 대한 논의가 추가된다면 더 가치 있을 것이다.

**비판적 해석**:
제안된 방법론은 SAM의 거대한 파라미터를 고정하고 어댑터와 소규모 메모리 뱅크만 학습시키므로 연산 효율성이 뛰어나다. 하지만 결과적으로 SAM의 인코더가 생성하는 특징 공간이 타겟 도메인에서도 어느 정도 유효하다는 가정에 기반하고 있다. 만약 소스 도메인과 타겟 도메인의 간극이 극도로 클 경우, 단순한 프로토타입 보간만으로 일반화가 가능할지에 대해서는 추가적인 검증이 필요해 보인다.

## 📌 TL;DR

**요약**: DAPSAM은 SAM을 의료 영상 분할에 적용하기 위해 **저수준 특징을 통합한 일반화 어댑터**와 **메모리 뱅크 기반의 적응형 프롬프트 생성기**를 도입한 프레임워크이다. 이를 통해 수동 프롬프트 없이도 소스 도메인의 지식을 활용해 본 적 없는 타겟 도메인의 영상을 효과적으로 분할할 수 있다.

**의의**: 본 연구는 거대 모델의 효율적인 미세 조정(Parameter-efficient fine-tuning)과 도메인 일반화 기술을 결합하여, 데이터 부족과 도메인 시프트가 심한 의료 AI 분야에서 실질적인 성능 향상을 이끌어낼 수 있는 방향성을 제시하였다.