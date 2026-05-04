# ElaLoRA: Elastic & Learnable Low-Rank Adaptation for Efficient Model Fine-Tuning

Huandong Chang, Zicheng Ma, Mingyuan Ma, Zhenting Qi, Andrew Sabot, Hong Jiang, H. T. Kung (2025)

## 🧩 Problem to Solve

최근 대규모 사전 학습 모델(Pre-trained Language Models, PLMs)의 크기가 급격히 증가함에 따라, 이를 특정 도메인에 맞게 조정하기 위한 파인튜닝(Fine-tuning) 과정에서 막대한 메모리와 계산 비용이 발생하고 있다. 이를 해결하기 위해 LoRA(Low-Rank Adaptation)와 같은 매개변수 효율적 파인튜닝(Parameter-Efficient Fine-Tuning, PEFT) 기법이 널리 사용되고 있다.

그러나 기존의 LoRA는 모든 레이어에 동일한 고정 랭크(Fixed Rank)를 할당하는 방식을 사용한다. 이는 모델의 각 레이어가 작업 적응에 기여하는 중요도가 서로 다름에도 불구하고 이를 고려하지 못한다는 한계가 있다. 즉, 더 많은 용량이 필요한 레이어에서는 언더피팅(Underfitting)이 발생하고, 중요도가 낮은 레이어에서는 불필요한 매개변수가 낭비되는 비효율성이 존재한다. 기존의 동적 랭크 할당 방법들(AdaLoRA, SaLoRA 등)은 주로 랭크의 가지치기(Pruning)에만 집중하거나, 반대로 랭크 확장(Expansion)에만 치중하여 학습 과정 중에 랭크를 유연하게 조절하는 능력이 부족하였다.

본 논문의 목표는 학습 과정에서 각 레이어의 중요도에 따라 랭크를 동적으로 가지치기하고 동시에 확장할 수 있는 Elastic & Learnable한 LoRA 프레임워크인 ElaLoRA를 제안하는 것이다.

## ✨ Key Contributions

ElaLoRA의 핵심 아이디어는 그래디언트(Gradient) 기반의 중요도 점수를 활용하여, 계산 자원을 가장 중요한 레이어에 동적으로 재배치하는 것이다. 주요 기여 사항은 다음과 같다.

1. **동적 랭크 재할당 메커니즘**: 파인튜닝 과정에서 랭크의 가지치기(Pruning)와 확장(Expansion)을 동시에 수행하는 최초의 방법론을 제시하였다.
2. **원칙적인 중요도 측정**: 손실 함수에 대한 가중치의 민감도를 측정하여 각 랭크의 기여도를 정량화하고, 이를 통해 효율적인 랭크 할당을 수행한다.
3. **범용적 성능 입증**: 자연어 이해(NLU), 자연어 생성(NLG), 그리고 컴퓨터 비전(Vision) 작업에 걸쳐 다양한 파라미터 예산 하에서 기존 PEFT 방법론보다 우수한 성능을 보임을 실험적으로 증명하였다.
4. **랭크 분포 분석**: 어떤 레이어가 작업 적응에 더 중요한 역할을 하는지 분석하여, 제안하는 적응형 랭크 할당 전략의 이론적 타당성을 검증하였다.

## 📎 Related Works

기존의 PEFT 연구들은 주로 업데이트해야 할 파라미터 수를 줄이는 방향으로 발전해 왔다.

- **LoRA**: 가중치 업데이트 행렬을 두 개의 저차원 행렬의 곱으로 분해하여 파라미터를 줄였으나, 고정 랭크 사용으로 인해 레이어별 중요도를 반영하지 못한다.
- **SVD 기반 적응형 방법 (AdaLoRA, SaLoRA)**: 특이값 분해(SVD)를 통해 중요도가 낮은 특이값을 제거하는 가지치기 방식을 사용한다. 하지만 초기 랭크를 높게 설정해야 하므로 계산 효율성이 떨어지는 문제가 있다.
- **랭크 확장 방법 (IncreLoRA)**: 최소 랭크에서 시작하여 휴리스틱하게 랭크를 늘려가지만, 학습 초기 샘플들이 충분히 학습되지 못할 위험이 있다.
- **기타 방법**: DoRA는 랭크-1 성분으로 분해하여 가지치기를 수행하며, DyLoRA는 미리 정의된 분포에서 랭크를 샘플링하는 방식을 취한다.

ElaLoRA는 이러한 기존 연구들과 달리, 학습 도중 중요도에 따라 랭크를 줄이는 것과 늘리는 것을 동시에 수행함으로써 자원 효율성과 모델 성능 사이의 최적의 균형을 찾는다.

## 🛠️ Methodology

ElaLoRA는 SVD 기반의 파라미터화, 중요도 점수 계산, 그리고 동적 랭크 학습 알고리즘의 세 가지 핵심 구성 요소로 이루어져 있다.

### 1. SVD 기반 Low-Rank Adaptation

사전 학습된 가중치 행렬 $W$의 업데이트를 다음과 같이 정의한다.
$$W = W^{(0)} + \Delta = W^{(0)} + P\Lambda Q$$
여기서 $P \in \mathbb{R}^{d_1 \times r}$와 $Q \in \mathbb{R}^{r \times d_2}$는 각각 좌측 및 우측 특이 벡터(Singular Vectors)이며, $\Lambda \in \mathbb{R}^{r \times r}$는 특이값(Singular Values)을 담은 대각 행렬이다. 수치적 안정성과 SVD의 성질을 유지하기 위해 다음과 같은 직교성 제약 조건(Orthogonality Constraints)을 부여한다.
$$R(P,Q) = \|P^\top P - I\|_F^2 + \|QQ^\top - I\|_F^2$$

### 2. 중요도 점수 계산 (Importance Score Computation)

각 랭크의 중요도는 손실 함수 $L$에 대한 가중치 $w$의 민감도를 기반으로 계산한다. 일차 테일러 전개(First-order Taylor expansion)를 사용하여 가중치 제거 시의 영향을 근사한다.
$$s(w) = \left| w \cdot \frac{\partial L}{\partial w} \right|$$
특정 랭크 $i$의 전체 중요도 $S_i$는 해당 랭크의 특이값 $\lambda_i$와 이에 대응하는 특이 벡터 $P, Q$의 성분들을 합산하여 구한다.
$$S_i = s(\lambda_i) + \frac{1}{d_1} \sum_{j=1}^{d_1} s(P_{ji}) + \frac{1}{d_2} \sum_{j=1}^{d_2} s(Q_{ij})$$
노이즈를 줄이기 위해 지수 이동 평균(EMA)을 적용하여 최종 중요도 점수 $s^{(t)}$를 산출한다.
$$\bar{I}^{(t)} = \beta_1 \bar{I}^{(t-1)} + (1-\beta_1)I^{(t)}$$
$$\bar{U}^{(t)} = \beta_2 \bar{U}^{(t-1)} + (1-\beta_2)|I^{(t)} - \bar{I}^{(t)}|$$
$$s^{(t)} = \bar{I}^{(t)} \cdot \bar{U}^{(t)}$$

### 3. 동적 랭크 학습 (Dynamic Rank Learning)

학습 과정은 **Warm-up $\rightarrow$ Dynamic Rank Adjustment $\rightarrow$ Stabilization**의 3단계로 진행된다.

- **Warm-up**: 초기 단계에서는 랭크를 고정하여 모델이 기본 표현을 학습하게 한다.
- **Dynamic Rank Adjustment**: 정해진 간격마다 다음 과정을 수행한다.
    1. 각 가중치 행렬에서 가장 중요도가 낮은 $k$개의 랭크를 식별한다.
    2. 모든 행렬에서 식별된 $k \times N$개의 랭크를 중요도 순으로 정렬한다.
    3. 전체에서 가장 중요도가 낮은 $b^{(i)}$개의 랭크를 제거(Prune)한다.
    4. 상대적으로 중요도가 높은 행렬(즉, 가장 낮은 랭크조차 중요도가 높은 경우)에 $b^{(i)}$개의 랭크를 추가(Expand)한다. 추가된 랭크는 Gram-Schmidt 직교화 방식으로 초기화한다.
- **Stabilization**: 최종 단계에서는 랭크 업데이트를 중단하여 모델이 안정적으로 수렴하도록 한다.

### 4. 동적 랭크 스케줄러 (Dynamic Rank Scheduler)

학습이 진행됨에 따라 랭크 조정의 공격성을 점진적으로 줄이기 위해 3차 다항식 보간법(Cubic polynomial interpolation)을 사용한 스케줄러를 도입한다.
$$P = \frac{\text{currentstep} - \text{initialwarmup}}{\text{totalstep} - \text{finalstabilization} - \text{initialwarmup}}$$
$$b^{(t)} = \text{round}(b \times (1 - P^3))$$
이를 통해 학습 초기에는 적극적으로 랭크를 탐색하고, 후반부에는 안정적인 수렴을 도모한다.

## 📊 Results

### 실험 설정

- **모델 및 데이터셋**: NLU(GLUE benchmark, DeBERTaV3-base), NLG(XSum dataset, BART-base), Vision(VTAB-1k subset, ViT-B/16).
- **비교 대상**: LoRA, AdaLoRA, DoRA, DyLoRA 등.
- **평가 지표**: Accuracy, F1 score, MCC, Pearson correlation, ROUGE-1/2/L 등.

### 주요 결과

1. **NLU (GLUE)**: ElaLoRA는 모든 랭크 설정($r=2, 4, 10$)에서 가장 높은 평균 성능을 기록하였다. 특히 **ElaLoRA ($r=2$)가 AdaLoRA ($r=4$)보다 높은 성능**을 보였는데, 이는 적은 파라미터로도 효율적인 랭크 배치가 가능함을 시사한다.
2. **NLG (XSum)**: BART-base 모델 실험에서 ElaLoRA는 LoRA와 AdaLoRA를 일관되게 능가하였으며, $r=6$ 설정에서 가장 우수한 ROUGE 점수를 달성하였다.
3. **Vision (VTAB-1k)**: ViT-B/16 모델을 사용하여 자연(Natural), 특수(Specialized), 구조(Structured) 작업 모두에서 LoRA와 AdaLoRA보다 높은 정확도를 보였으며, 평균 정확도 64.88%를 기록하였다.

## 🧠 Insights & Discussion

### 랭크 분포 분석

ElaLoRA가 최종적으로 할당한 랭크의 분포를 분석한 결과, **중간 피드포워드 레이어(Intermediate feed-forward layers)**에 가장 많은 랭크가 할당되었고, 최종 프로젝션 레이어에는 적은 랭크가 할당되었다.
실제로 중간 피드포워드 레이어의 랭크를 제거했을 때 성능이 급격히 하락한 반면, 최종 프로젝션 레이어를 제거했을 때는 영향이 적었다. 이는 작업 적응에 있어 특정 레이어가 결정적인 역할을 한다는 것을 의미하며, ElaLoRA의 적응형 할당 전략이 실제 모델의 요구사항을 정확히 반영하고 있음을 보여준다.

### 중요도 점수 분석

중요도 점수의 분포를 분석했을 때, ElaLoRA는 AdaLoRA나 고정 랭크 설정보다 중요도 분포의 밀도가 오른쪽(높은 값)으로 이동하는 경향을 보였다. 이는 ElaLoRA가 선택한 파라미터들이 손실 함수에 더 큰 영향을 미치도록 최적화되었음을 의미한다.

### 한계 및 논의

본 논문은 다양한 벤치마크에서 우수한 성능을 보였으나, 사전 학습된 모델이 가진 기존의 편향(Bias)을 제거하는 기능은 포함하고 있지 않다. 또한, 랭크 조정 간격($t_{adjust}$)이나 스케줄러 파라미터 $b$와 같은 하이퍼파라미터에 대한 민감도 분석이 추가적으로 필요할 수 있다.

## 📌 TL;DR

ElaLoRA는 파인튜닝 과정에서 각 레이어의 중요도를 실시간으로 측정하여 **랭크를 동시에 가지치기(Pruning)하고 확장(Expansion)하는 동적 랭크 재할당 프레임워크**이다.

이 연구는 고정된 랭크를 사용하는 기존 LoRA의 한계를 극복하고, 한정된 파라미터 예산 내에서 최적의 효율을 낼 수 있는 랭크 분포를 스스로 찾아낸다. 특히 NLU, NLG, Vision 등 다양한 도메인에서 기존 PEFT 방법론들을 압도하는 성능을 보였으며, 이는 자원이 제한된 환경에서 대규모 모델을 효율적으로 미세 조정하는 데 매우 중요한 역할을 할 가능성이 크다.
