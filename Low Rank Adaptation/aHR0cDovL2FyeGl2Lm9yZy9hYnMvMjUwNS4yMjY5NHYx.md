# MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning

Dacao Zhang, Kun Zhang, Shimao Chu, Le Wu, Xin Li, Si Wei (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)의 효율적인 미세 조정(Fine-tuning)을 위한 Parameter-Efficient Fine-Tuning (PEFT) 방법론, 특히 Low-Rank Adaptation (LoRA)의 한계를 해결하고자 한다. 

기존의 LoRA는 고정된 intrinsic rank $r$을 사용하는데, 이는 다양한 태스크가 서로 다른 복잡도를 가지고 있어 각 태스크마다 최적의 rank가 다르다는 점을 간과한다. 예를 들어, 어떤 태스크는 매우 낮은 rank에서도 좋은 성능을 보이지만, 다른 태스크는 더 높은 rank가 필요할 수 있다. 이를 해결하기 위해 각 태스크별로 개별적인 LoRA 모듈을 학습시키거나 여러 개의 LoRA를 병렬로 사용하는 방법이 제안되었으나, 이는 파라미터 수의 증가와 계산 비용의 상승을 초래하여 LoRA 본연의 효율성 목적에 어긋난다. 

따라서 본 연구의 목표는 다중 태스크 학습(Multi-Task Learning, MTL) 시나리오에서 각 태스크의 특성에 맞게 rank를 동적으로 조정함으로써, 파라미터 효율성을 유지하면서도 성능을 극대화하는 적응형 메커니즘을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LoRA 모듈 내의 **서로 다른 rank 자체를 '전문가(Expert)'로 취급**하는 Mixture of Experts (MoE) 구조를 도입하는 것이다. 

중심적인 설계 직관은 다음과 같다. 하나의 큰 LoRA 행렬을 준비하고, 태스크에 따라 이 행렬의 일부(특정 rank까지의 슬라이스)만을 사용하도록 선택하는 것이다. 이렇게 하면 낮은 rank의 전문가들은 여러 태스크 간의 공통 정보를 공유하고, 높은 rank의 전문가들은 각 태스크의 고유하고 세부적인 정보를 학습할 수 있게 된다. 이를 위해 태스크 임베딩을 기반으로 적절한 rank 전문가를 선택하는 **Adaptive Rank Selector**를 제안하였다.

## 📎 Related Works

### 기존 연구 및 한계
1. **PEFT 및 LoRA 변형**: LoRA 외에도 AdaLoRA, DyLoRA, SoRA 등이 제안되었다. 이들은 rank를 동적으로 조정하거나 중요도에 따라 파라미터 예산을 할당하지만, 주로 단일 태스크 시나리오에 집중되어 있어 다중 태스크 환경에서의 태스크 간 관계나 차이점을 충분히 고려하지 않는다.
2. **Multi-Task Learning (MTL) 접근법**: HyperFormer와 같은 하이퍼네트워크 기반 방식이나 Prompt Tuning 기반 방식이 존재한다. 그러나 이들은 추론 시 추가적인 지연 시간(latency)이 발생하거나, 2단계 학습 과정이 필요하여 효율성이 떨어진다.
3. **Parallel LoRA 전략**: MultiLoRA, MixLoRA, MoELoRA 등 여러 개의 LoRA 모듈을 병렬로 배치하는 방식이 제안되었다. 하지만 이는 전체 파라미터 수를 크게 증가시켜 LoRA의 원래 목적과 배치되며, 태스크별 최적 rank 할당 문제를 완전히 해결하지 못한다.

### MoRE의 차별점
MoRE는 병렬 모듈을 사용하는 대신, 단일 LoRA 모듈 내에서 rank를 계층적으로 전문가화하여 공유 정보와 특화 정보를 동시에 관리한다. 또한, Contrastive Learning (CL)을 통해 태스크 임베딩의 품질을 높여 정교한 rank 선택을 가능하게 한다.

## 🛠️ Methodology

### 전체 파이프라인
MoRE는 입력 데이터가 들어오면 해당 태스크의 임베딩을 참조하여 최적의 rank 전문가를 선택하고, 선택된 rank만큼의 LoRA 가중치만을 사용하여 연산을 수행하는 구조이다.

### 주요 구성 요소 및 상세 설명

#### 1. Task Embedding 및 CL-based Optimization
각 태스크 $T_t$를 대표하는 벡터 $e_t$를 학습한다. 감독 신호가 없는 태스크 임베딩의 품질을 높이기 위해 **Contrastive Learning (CL)**을 도입한다. 동일한 태스크의 샘플 표현 $h_i$와 해당 태스크 임베딩 $e_t$ 사이의 유사도는 최대화하고, 다른 태스크 임베딩 $e_k$와의 유사도는 최소화하도록 다음과 같은 손실 함수를 사용한다.

$$L_{con} = \frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(h_i, e_t)/\tau)}{\sum_{k=1}^{T} \exp(\text{sim}(h_i, e_k)/\tau)} \right]$$

#### 2. Adaptive Rank Selector
태스크 임베딩 $e_t$를 입력으로 받아 어떤 rank 전문가를 사용할지 결정하는 게이팅 네트워크 $G(\cdot)$이다.

- **확률 분포 계산**: 
  $$p_t = G(e_t) = \text{softmax}(W_g e_t + b_g)$$
- **Rank 선택**: 가장 확률이 높은 index를 선택하여 rank $r_t$를 결정한다.
  $$r_t = \text{arg max } p_t$$
- **전문가 구성**: 선택된 $r_t$에 따라 LoRA 행렬 $A, B$를 슬라이싱하여 사용한다.
  $$A_t = A[:r_t, :], \quad B_t = B[:, :r_t]$$

#### 3. 학습 절차 및 최적화
- **Straight-Through Estimator (STE)**: $\text{arg max}$ 연산은 미분이 불가능하므로, STE를 사용하여 기울기가 게이팅 네트워크로 전달되도록 한다.
  $$\text{Ste}(p_t) = p_t + \text{sg}[\text{one\_hot}(p_t) - p_t]$$
- **Linear Scaling**: 낮은 rank 부분은 여러 태스크가 공유하므로 업데이트가 빈번하게 일어난다. 이를 안정화하기 위해 가중치에 $\frac{r_t}{|T|}$ (여기서 $|T|$는 전체 태스크 수)를 곱하는 선형 스케일링을 적용한다.
- **최종 손실 함수**: 생성 손실 $L_{gen}$과 대조 학습 손실 $L_{con}$을 결합한다.
  $$L = L_{gen} + \lambda L_{con}$$

#### 4. Balanced Dataset Sampling
데이터셋 간의 심한 불균형(예: MNLI vs RTE)을 해결하기 위해, 데이터셋 크기에 반비례하는 샘플링 가중치 $\phi_t$를 부여하여 각 데이터셋이 학습 과정에 균형 있게 기여하도록 한다.

## 📊 Results

### 실험 설정
- **데이터셋**: GLUE benchmark, 상식 추론(Commonsense Reasoning: BoolQ, PIQA, OBQA, ARC), Few-shot 도메인 전이 태스크.
- **모델**: LLaMA2-7B 및 T5-base.
- **비교 대상**: Full FT, Adapter, Prompt Tuning, LoRA, MultiLoRA, MixLoRA, MoELoRA 등.

### 주요 결과
1. **정량적 성능**: GLUE 벤치마크와 상식 추론 태스크 모두에서 MoRE가 기존 PEFT 및 다중 태스크 베이스라인보다 우수한 성능을 보였다. 특히 LLaMA2-7B 모델에서 성능 향상이 두드러졌다.
2. **Few-shot 전이 성능**: 적은 양의 데이터만으로도 도메인 전이가 효과적으로 이루어짐을 확인하였으며, 이는 MoRE가 태스크 간 공유 정보와 특화 정보를 효율적으로 분리했음을 시사한다.
3. **학습 속도**: LoRA 대비 약 $1.6\times$ 정도 느리지만, MixLoRA($4.2\times$)나 MoELoRA($3.6\times$)보다 훨씬 빠른 학습 속도를 기록하여 효율적인 트레이드-오프를 달성하였다.
4. **파라미터 효율성**: 추론 단계에서는 태스크-전문가 매핑이 고정되므로, 사실상 vanilla LoRA와 동일한 파라미터 수로 추론이 가능하다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **Rank 할당의 적절성**: 분석 결과, 대부분의 태스크가 낮은 rank(1, 2, 3)를 주로 사용하며, 특정 태스크(예: MRPC)는 특정 전문가(Expert 4)에 의존하는 경향을 보였다. 이는 MoRE가 태스크의 복잡도에 따라 자원을 유연하게 할당하고 있음을 증명한다.
- **태스크 임베딩의 유효성**: PCA 시각화 결과, 유사한 태스크(MRPC, QNLI)는 가깝게 뭉치고 성격이 다른 태스크(STSB, CoLA)는 멀리 떨어져 있어, CL 기반의 임베딩 학습이 태스크 특성을 정확히 포착했음을 알 수 있다.

### 한계 및 비판적 해석
- **모델 확장성**: GPU 자원 제한으로 인해 13B 이상의 초거대 모델에 대한 검증이 이루어지지 않았다.
- **추론 지연 시간**: MoE 구조 특성상 가중치를 원래 모델에 병합(Merge)할 수 없으므로, 일반 LoRA보다는 추론 시 지연 시간이 발생할 수밖에 없다.
- **생성 태스크 검증 부족**: NLG(자연어 생성) 태스크에 대한 초기 실험은 수행하였으나, 보다 상세한 분석과 실험이 추가로 필요하다.

## 📌 TL;DR

본 논문은 LoRA의 고정된 rank 문제를 해결하기 위해 **단일 LoRA 모듈 내의 다양한 rank를 전문가(Expert)로 활용하는 MoRE**를 제안한다. 태스크 임베딩과 적응형 선택기를 통해 각 태스크에 최적화된 rank를 동적으로 할당하며, Contrastive Learning과 균형 샘플링 전략으로 학습 안정성을 높였다. 결과적으로 MoRE는 적은 파라미터 증가만으로 다중 태스크 환경에서 기존 LoRA 변형 모델들보다 뛰어난 성능과 학습 효율성을 보여주었으며, 이는 향후 LLM의 효율적인 다중 태스크 적응 연구에 중요한 방향성을 제시한다.