# CRISP: Contrastive Residual Injection and Semantic Prompting for Continual Video Instance Segmentation

Baichen Liu, Qi Lyu, Xudongong Wang, Jiahua Dong, Lianqing Liu, Zhi Han (2025)

## 🧩 Problem to Solve

본 논문은 **Continual Video Instance Segmentation (CVIS)** 분야에서 발생하는 고질적인 문제들을 해결하고자 한다. CVIS는 새로운 객체 카테고리를 학습하는 가소성(Plasticity)과 기존에 학습한 지식을 유지하는 안정성(Stability)을 동시에 확보해야 하며, 동시에 비디오 프레임 전반에 걸쳐 시간적 일관성(Temporal Consistency)을 유지해야 하는 매우 어려운 과제이다.

저자들은 기존의 Continual Learning 방법론을 비디오 도메인에 적용했을 때 다음과 같은 세 가지 수준의 **의미론적 혼란(Semantic Confusion)**이 발생한다는 점을 지적한다.

1. **Instance-wise Confusion**: 동일한 카테고리에 속한 서로 다른 두 객체를 하나의 인스턴스로 잘못 병합하여 인식하는 문제이다.
2. **Category-wise Confusion**: 동일한 비디오 내의 인스턴스가 서로 다른 카테고리로 잘못 분류되거나, 클래스 간 임베딩 공간이 겹쳐 발생하는 오분류 문제이다.
3. **Task-wise Confusion**: 새로운 태스크를 위한 쿼리(Query)를 초기화할 때 기존 태스크의 쿼리에 지나치게 의존함으로써, 새로운 카테고리를 학습하는 능력이 저하되고 의미론적 표류(Semantic Drift)가 발생하는 문제이다.

결과적으로 본 논문의 목표는 이러한 세 가지 수준의 혼란을 억제하여, 치명적 망각(Catastrophic Forgetting)을 방지하고 비디오 인스턴스 분할 및 추적 성능을 향상시키는 **CRISP** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 각 수준의 혼란을 해결하기 위해 서로 다른 보완적 메커니즘을 도입하는 것이다.

- **카테고리 수준(Category-wise)**: CLIP의 텍스트 인코더를 활용한 **Adaptive Residual Semantic Prompt (ARSP)** 프레임워크를 통해 클래스별 세부 정보를 주입하고, Contrastive Learning 기반의 **Instance Semantic Consistency Loss**를 통해 쿼리와 프롬프트 간의 정렬을 강화한다.
- **태스크 수준(Task-wise)**: 기존 쿼리 공간의 주성분을 분석하는 **PCA-guided Initialization** 전략을 도입하여, 새로운 태스크의 쿼리가 기존 지식을 계승하면서도 충분한 변별력을 갖도록 초기화한다.
- **인스턴스 수준(Instance-wise)**: 인스턴스 간의 구분 가능성을 높이기 위해 쿼리 간의 상관관계를 모델링하는 **Instance Correlation Loss**를 제안하여 추적 성능과 인스턴스 분별력을 높인다.

## 📎 Related Works

### 기존 연구 및 한계

- **Instance Segmentation**: Mask R-CNN과 같은 다단계 파이프라인에서 Mask2Former와 같은 Transformer 기반의 end-to-end 모델로 발전하였다.
- **Video Instance Segmentation (VIS)**: ISTR, IFC 등이 제안되었으나, 이들은 고정된 클래스 세트를 가정하며 Continual Learning 환경에서의 망각 문제는 다루지 않았다.
- **Continual Image Segmentation**: Knowledge Distillation이나 Pseudo-labeling 기반 방법들이 제안되었으나, 비디오 데이터의 특성인 시간적 특징 표류(Temporal Feature Drift)와 프레임 간 인스턴스 연관성 불일치 문제를 해결하지 못한다.
- **Prompt-tuning**: L2P, DualPrompt 등 프롬프트 풀을 이용해 망각을 방지하는 연구가 있었으나, 비디오의 고차원 데이터 특성과 인스턴스 추적의 복잡성을 충분히 반영하지 못했다.

### 차별점

CRISP는 단순한 이미지 기반의 Continual Learning을 넘어, 비디오의 **시간적 일관성**과 **인스턴스 식별력**을 유지하기 위해 쿼리 공간의 상관관계 분석(PCA)과 세밀한 세만틱 프롬프트 주입(ARSP)을 결합했다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 구조 (Overview)

CRISP는 Mask2Former for Video를 기반으로 하며, 학습 단계에서만 **Residual Semantic Prompt**를 주입하고 추론 단계에서는 이를 제거하여 속도를 유지한다. 시스템은 크게 ARSP 모듈, PCA 기반 초기화, 그리고 두 가지 보조 손실 함수($L_{ISC}, L_{IC}$)로 구성된다.

### 2. Adaptive Residual Semantic Prompt (ARSP)

카테고리 수준의 혼란을 해결하기 위한 모듈이다.

- **Prompt Generator**: CLIP 텍스트 인코더를 통해 각 클래스에 해당하는 학습 가능한 세만틱 잔차 프롬프트 풀 $P_t \in \mathbb{R}^{c_t \times d}$를 생성한다.
- **Query-Prompt Matching**: 현재 태스크의 객체 쿼리 $Q_t$와 프롬프트 $P_t$ 간의 코사인 유사도를 계산하여 최적의 프롬프트를 매칭한다.
    $$S_{i,j} = \frac{Q_t P_t^T}{\|Q_t\|_F \|P_t^T\|_F}$$
- **Hierarchical Injection**: Transformer 디코더의 Self-Attention 연산 시, Value($V$) 값에 매칭된 잔차 프롬프트 $P_m$을 더해주는 방식으로 정보를 주입한다.
    $$O_l = \text{Softmax}\left(\frac{Q_l (K_l)^T}{\sqrt{d_k}}\right)(V_l + P_{lm})$$

### 3. Instance Semantic Consistency Loss ($L_{ISC}$)

추론 시에는 프롬프트를 사용하지 않으므로, 학습 중에 쿼리가 프롬프트의 세만틱 정보를 충분히 학습하도록 Contrastive Learning을 적용한다. 쿼리와 매칭된 프롬프트 간의 유사도는 높이고, 다른 클래스의 프롬프트와는 멀어지게 설계한다.
$$L_{ISC} = \frac{1}{N_q^t} \sum_{i=1}^{N_q^t} \log \left( 1 + \frac{\sum_{j=1}^{c_t} I_{ij} \exp(S_{ij})}{\exp(S_{i, a_i})} \right)$$
여기서 $a_i$는 매칭된 프롬프트의 인덱스이며, $I_{ij}$는 $j \neq a_i$일 때 1인 지시함수이다.

### 4. PCA-guided Initialization

태스크 간의 혼란을 막기 위해, 새로운 태스크의 쿼리를 단순 복제하는 대신 기존 태스크 쿼리들의 주성분(Principal Components)을 추출하여 초기화한다.

1. 기존 쿼리 $Q_o$에 대해 PCA를 수행하여 주성분 벡터들을 추출한다.
2. 고유값이 큰 상위 $c_t$개의 벡터를 선택한다.
3. 기존 쿼리 매니폴드에 맞게 정렬(Alignment) 과정을 거쳐 현재 태스크의 초기 쿼리 $Q_t$로 설정한다.

### 5. Instance Correlation Loss ($L_{IC}$)

인스턴스 간의 변별력을 높이기 위해 쿼리 간의 내적 행렬(Inner Product Matrix)이 초기 태스크의 쿼리 상관관계 구조를 유지하도록 강제한다.
$$L_{IC} = \text{MSE}(\hat{Q}_t \hat{Q}_t^T, \hat{Q}_0 \hat{Q}_0^T)$$
여기서 $\hat{Q}$는 정규화된 쿼리 벡터이며, 이를 통해 쿼리들이 서로 충분히 구분되면서도 필요한 유사성을 유지하게 한다.

### 6. 최종 손실 함수

최종 목적 함수는 세그멘테이션 손실($L_{Seg}$)과 제안된 두 보조 손실의 합으로 정의된다.
$$L_{tol} = L_{Seg} + \lambda_{ISC} L_{ISC} + \lambda_{IC} L_{IC}$$
(논문에서는 $\lambda_{ISC} = \lambda_{IC} = 3$으로 설정하였다.)

## 📊 Results

### 실험 설정

- **데이터셋**: YouTube-VIS-2019 및 YouTube-VIS-2021.
- **시나리오**:
  - 20-4 (초기 20 클래스, 단계당 4 클래스 추가, 총 6단계)
  - 20-2 (초기 20 클래스, 단계당 2 클래스 추가, 총 11단계)
  - 20-5 (초기 20 클래스, 단계당 5 클래스 추가, 총 5단계)
  - 10-10 (초기 10 클래스, 단계당 10 클래스 추가, 총 4단계)
- **평가 지표**: mAP, $AP_{50}, AP_{75}, AR1, AR10$ 및 망각률을 측정하는 Forgetting Ratio (FR).

### 주요 결과

- **정량적 성능**: YouTube-VIS-2019의 20-4 시나리오에서 mAP 28.1, FR 1.93을 기록하며 ECLIPSE(mAP 25.03) 및 CoMBO(mAP 15.8) 등 기존 방법론을 유의미하게 상회하였다.
- **장단기 시나리오 분석**:
  - **장기 시나리오(Long-term)**: 단계가 많아질수록 CRISP의 성능 우위가 뚜렷하게 나타난다. 이는 PCA 초기화와 ARSP가 누적되는 망각을 효과적으로 억제함을 보여준다.
  - **단기 시나리오(Short-term)**: 10-10 시나리오에서는 CoMBO가 일부 지표에서 우세했으나, 이는 데이터 양이 많고 태스크 간격이 짧아 단순 적응이 유리했기 때문으로 분석된다.
- **Ablation Study**: PCA-guided Initialization(PI)을 제거했을 때 성능 하락이 가장 컸으며, 이는 태스크 간 변별력을 확보하는 것이 CVIS에서 매우 중요함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 CVIS의 문제를 단순히 '망각'이라는 관점이 아니라, 인스턴스/카테고리/태스크라는 세 가지 층위의 '의미론적 혼란'으로 세분화하여 접근한 점이 매우 인상적이다. 특히 PCA를 이용한 쿼리 초기화는 딥러닝 모델이 새로운 태스크를 학습할 때 겪는 '초기값 설정' 문제를 수학적인 주성분 분석으로 해결하여, 학습의 안정성과 변별력을 동시에 잡았다.

### 한계 및 논의사항

1. **추론 효율성**: 학습 시에는 ARSP를 통해 강력한 가이드를 제공하지만, 추론 시에는 이를 제거한다. 이 과정에서 학습-추론 간의 괴리(Gap)가 발생할 가능성이 있으며, 이를 완전히 해결했는지에 대한 심도 있는 분석이 필요하다.
2. **데이터 의존성**: CLIP 텍스트 인코더를 사용하여 프롬프트를 생성하므로, CLIP이 사전에 학습하지 못한 특수한 도메인의 객체에 대해서는 성능이 저하될 가능성이 있다.
3. **범용성**: 현재는 Video Instance Segmentation에 국한되어 실험이 진행되었다. 저자들이 언급했듯이 Panoptic Segmentation과 같은 더 복잡한 시나리오로의 확장이 필요하다.

## 📌 TL;DR

**CRISP**는 Continual Video Instance Segmentation에서 발생하는 세 가지 수준의 의미론적 혼란(인스턴스, 카테고리, 태스크)을 해결하기 위한 프레임워크이다. **PCA 기반 쿼리 초기화**로 태스크 간 표류를 막고, **CLIP 기반의 잔차 세만틱 프롬프트(ARSP)**와 **상관관계 손실 함수**를 통해 클래스 및 인스턴스 식별력을 극대화하였다. 특히 장기적인 지속 학습 시나리오에서 기존 SOTA 모델들을 뛰어넘는 성능을 보였으며, 비디오 도메인의 특성을 반영한 Continual Learning의 효과적인 방향성을 제시하였다.
