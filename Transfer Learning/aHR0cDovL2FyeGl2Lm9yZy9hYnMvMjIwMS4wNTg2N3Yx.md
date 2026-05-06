# Transferability in Deep Learning: A Survey

Junguang Jiang, Yang Shu, Jianmin Wang, Mingsheng Long (2022)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델의 가장 큰 병목 현상 중 하나인 **데이터 효율성(Data Efficiency)** 문제를 해결하고자 한다. 현재의 주류 딥러닝 알고리즘은 높은 성능을 내기 위해 수백만에서 수조 개에 이르는 대규모 데이터셋을 필요로 한다. 하지만 새로운 태스크나 도메인에 진입할 때마다 이러한 방대한 양의 데이터를 수집하고 레이블링하는 것은 비용 면에서 매우 비효율적이며, 때로는 불가능에 가깝다.

반면, 인간은 이전의 학습 경험에서 얻은 관련 지식을 새로운 문제에 적용하는 **전이 능력(Transferability)**을 가지고 있어 극소수의 샘플만으로도 새로운 작업을 수행할 수 있다. 따라서 본 논문의 목표는 딥러닝의 전 과정(Lifecycle)에서 전이 능력을 어떻게 획득하고 활용할 수 있는지에 대해 통합적이고 체계적인 관점을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 파편화되어 있던 딥러닝의 여러 연구 분야(Pre-training, Task Adaptation, Domain Adaptation 등)를 **전이 능력(Transferability)**이라는 하나의 핵심 키워드로 연결하여 통합적인 프레임워크를 제시한 것이다.

1. **딥러닝 라이프사이클의 정의**: 딥러닝의 적용 과정을 '전이 가능한 지식을 획득하는 **Pre-training(사전 학습)** 단계'와 '획득한 지식을 재사용하는 **Adaptation(적응)** 단계'의 두 단계로 구조화하여 분석한다.
2. **포괄적인 방법론 분석**: 모델 아키텍처의 영향부터 지도/비지도 사전 학습, 태스크 적응, 도메인 적응에 이르기까지 전이 능력을 향상시키기 위한 핵심 원리와 최신 기법들을 상세히 리뷰한다.
3. **이론적 기반 제시**: 특히 도메인 적응(Domain Adaptation) 분야에서 $H\Delta H$-Divergence 및 Disparity Discrepancy와 같은 수학적 이론이 어떻게 실제 알고리즘으로 구현되는지를 체계적으로 설명한다.
4. **오픈소스 라이브러리 및 벤치마크 제공**: 연구자들이 전이 능력을 공정하게 평가할 수 있도록 `TLlib`라는 오픈소스 라이브러리를 구현하고, 대규모 데이터셋 기반의 벤치마크 결과를 제시한다.

## 📎 Related Works

논문은 전이 능력을 일반화(Generalizability), 해석 가능성(Interpretability), 강건성(Robustness)과 함께 딥러닝과 인간 학습 사이의 간극을 줄이기 위한 핵심 속성으로 정의한다.

기존 연구들은 주로 특정 영역에 국한되어 있었다. 예를 들어, Domain Adaptation은 분포 변화(Distribution Shift) 해결에만 집중했고, Continual Learning은 지속적 학습에서의 망각 방지에만 치중했다. 또한, 전통적인 CNN이나 RNN은 지역적 연결성(Local Connectivity)이라는 강한 **귀납적 편향(Inductive Bias)**을 가지고 있어 데이터 효율성은 높지만, 대규모 데이터가 주어진 상황에서는 오히려 표현력과 전이 능력을 제한하는 한계가 있었다. 본 논문은 이러한 개별 연구들을 '전이 능력의 획득과 활용'이라는 관점에서 통합하여 차별성을 둔다.

## 🛠️ Methodology

본 논문은 딥러닝의 라이프사이클을 크게 Pre-training과 Adaptation으로 나누어 설명한다.

### 1. Pre-Training (지식 획득 단계)

이 단계의 목표는 다양한 다운스트림 태스크에 적용 가능한 **일반적 전이 능력(Generic Transferability)**을 확보하는 것이다.

* **모델 아키텍처**: ResNet은 잔차 학습(Residual Learning)을 통해 층을 깊게 쌓아 용량을 키웠으며, Transformer와 ViT는 강한 귀납적 편향을 제거하여 대규모 데이터로부터 더 풍부한 전이 지식을 학습할 수 있게 한다.
* **Supervised Pre-Training**: 대규모 레이블 데이터(예: ImageNet)를 사용하여 특징 추출기를 학습시킨다.
  * **Meta-Learning**: '학습하는 법을 학습'하는 것으로, 소수의 데이터로 빠르게 적응할 수 있는 초기 파라미터 $\phi$를 찾는 것을 목표로 한다. 최적화 식은 다음과 같은 이단계(Bi-level) 최적화 구조를 가진다.
        $$\phi^* = \arg \max_{\phi} \sum_{i=1}^{n} \log P(\theta_i(\phi) | D_{ts}^i), \text{ where } \theta_i(\phi) = \arg \max_{\theta} \log P(\theta | D_{tr}^i, \phi)$$
  * **Causal Learning**: 분포 변화에 강건한 인과 관계를 학습하여 OOD(Out-of-Distribution) 일반화 능력을 높인다. Invariant Risk Minimization (IRM)은 모든 환경 $e \in E_{tr}$에서 최적인 분류기 $h$를 찾는 제약 조건 최적화 문제를 푼다.
* **Unsupervised Pre-Training**: 레이블 없는 데이터를 활용한다.
  * **Generative Learning**: 데이터 분포 $P(X)$를 학습한다. GPT와 같은 Autoregressive 모델은 다음 토큰을 예측하고, BERT와 같은 Autoencoding 모델은 마스킹된 토큰을 복원하는 방식을 사용한다.
  * **Contrastive Learning**: 동일 샘플의 서로 다른 뷰(View) 간 거리는 좁히고, 서로 다른 샘플 간 거리는 멀게 하여 변별력 있는 표현을 학습한다. (예: SimCLR, MoCo)

### 2. Adaptation (지식 활용 단계)

이 단계의 목표는 사전 학습된 지식을 특정 태스크나 도메인에 맞게 최적화하는 **특수 전이 능력(Specific Transferability)**을 구현하는 것이다.

* **Task Adaptation**: 사전 학습된 모델 $h_{\theta_0}$를 타겟 데이터 $\hat{T}$에 맞게 조정한다.
  * **Catastrophic Forgetting 방지**: 새로운 태스크를 배울 때 기존 지식을 잃는 문제를 해결하기 위해 EWC(Elastic Weight Consolidation)와 같은 규제화(Regularization) 기법을 사용한다.
        $$\Omega(\theta) = \sum_{j} \frac{1}{2} F_j \| \theta_j - \theta_0^j \|^2$$
        여기서 $F$는 Fisher 정보 행렬로, 중요한 파라미터가 변하지 않도록 강제한다.
  * **Negative Transfer 방지**: 관련 없는 지식이 전이되어 성능이 떨어지는 현상을 막기 위해 BSS(Batch Spectral Shrinkage) 등을 통해 불필요한 스펙트럼 성분을 억제한다.
  * **Parameter Efficiency**: 모든 파라미터를 튜닝하는 대신, Adapter Tuning이나 Diff Pruning을 통해 극소수의 파라미터 $\delta_{task}$만 학습하여 저장 비용을 줄인다.
* **Domain Adaptation (UDA)**: 레이블이 있는 소스 도메인 $S$에서 레이블이 없는 타겟 도메인 $T$로 전이한다.
  * **Statistics Matching**: MMD(Maximum Mean Discrepancy) 등을 사용하여 두 도메인의 통계적 분포를 직접 맞춘다.
  * **Domain Adversarial Learning**: 도메인 판별기(Discriminator)와 특징 추출기(Generator)가 서로 경쟁하게 하여, 판별기가 도메인을 구분하지 못하도록(즉, 도메인 불변한 특징을 추출하도록) 학습한다. DANN의 목적 함수는 다음과 같다.
        $$\min_{\psi, h} \mathbb{E}_{(x_s, y_s) \sim \hat{S}} L_{CE}(h(z_s), y_s) + \lambda L_{DANN}(\psi)$$
  * **Hypothesis Adversarial Learning**: 두 분류기 간의 출력 차이(Discrepancy)를 이용해 타겟 도메인의 샘플을 소스 도메인의 분포 안으로 끌어들인다.

## 📊 Results

### 1. 실험 설정 및 데이터셋

* **NLP**: GLUE 벤치마크(9개 태스크)를 통해 다양한 언어 이해 작업에서의 전이 성능을 측정하였다.
* **CV**: ImageNet-R, ImageNet-Sketch, DomainNet 등을 사용하여 도메인 간 전이 성능을 평가하였다.
* **지표**: Accuracy, F1-score, mIoU 등을 사용하였다.

### 2. 주요 결과

* **Pre-training 효과**: ViT와 같은 대용량 모델이 Transformer 아키텍처와 결합했을 때, 전통적인 ResNet보다 훨씬 강력한 전이 능력을 보였다. 특히 unsupervised pre-training(예: MAE)이 downstream task에서 높은 성능을 기록했다.
* **Task Adaptation**: 단순 Fine-tuning보다 Regularization Tuning(예: SMART)이 데이터가 적은 상황에서 더 안정적인 성능을 보였으며, Adapter Tuning은 매우 적은 파라미터 업데이트(약 3.6%)만으로도 Full Fine-tuning에 근접한 성능을 낼 수 있음을 확인했다.
* **Domain Adaptation**: 이론적 기반의 MDD(Margin Disparity Discrepancy)가 DomainNet 및 ImageNet-scale 데이터셋에서 기존 DANN이나 MCD보다 우수한 성능을 보였으며, 특히 대규모 데이터셋으로 갈수록 parametric distance 기반 방법들이 더 유리한 경향을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰

본 논문은 전이 능력을 단순히 '기법'의 나열이 아닌 '라이프사이클'의 관점에서 분석함으로써, 사전 학습 단계의 결정이 적응 단계에 어떤 영향을 미치는지 명확히 설명했다. 특히, **"전이 능력 vs 변별력(Transferability vs Discriminability)"**의 딜레마를 지적한 점이 인상적이다. 도메인 불변 특징을 너무 강하게 추구하면(전이 능력 $\uparrow$), 클래스 간의 구분 능력이 저하될 수 있다(변별력 $\downarrow$)는 점을 분석하고 이를 해결하기 위한 BSP 등의 방법을 제시하였다.

### 2. 한계 및 미해결 질문

* **데이터 규모의 영향**: 많은 Meta-learning이나 Causal learning 방법론들이 여전히 소규모 데이터셋에서 검증되었으며, 이들이 초거대 모델 및 데이터셋 환경에서도 동일한 효율성을 보일지는 미지수이다.
* **휴리스틱한 설계**: 비지도 사전 학습의 태스크 설계(예: 마스킹 비율, 대조 학습의 augmentation 방식)가 여전히 경험적인 휴리스틱에 의존하고 있으며, 이에 대한 이론적 분석이 부족하다.

### 3. 비판적 해석

논문은 광범위한 방법론을 다루고 있어 백과사전식 구성에 가깝다. 하지만 각 방법론 간의 상충 관계(Trade-off)를 표(Table 2, 4, 5)로 명확히 정리하여, 사용자가 자신의 상황(데이터 양, 계산 자원, 도메인 차이 등)에 맞는 최적의 전략을 선택할 수 있는 가이드를 제공했다는 점에서 학술적 가치가 높다.

## 📌 TL;DR

본 논문은 딥러닝의 **데이터 효율성** 문제를 해결하기 위한 **전이 능력(Transferability)**을 중심으로, `사전 학습(Pre-training) $\rightarrow$ 적응(Adaptation) $\rightarrow$ 평가(Evaluation)`로 이어지는 통합적인 분석 프레임워크를 제시한 서베이 논문이다. 다양한 아키텍처, 학습 전략 및 도메인 적응 이론을 집대성하였으며, 실질적인 구현을 돕기 위한 `TLlib` 라이브러리를 함께 제공하였다. 이 연구는 향후 적은 데이터로도 고성능을 내는 효율적인 딥러닝 모델 설계 및 도메인 확장 연구에 핵심적인 기초 자료로 활용될 가능성이 매우 크다.
