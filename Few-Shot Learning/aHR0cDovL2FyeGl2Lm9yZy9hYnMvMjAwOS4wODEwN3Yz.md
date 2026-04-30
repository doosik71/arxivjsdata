# Few-Shot Unsupervised Continual Learning through Meta-Examples

Alessia Bertugli, Stefano Vincenzi, Simone Calderara, Andrea Passerini (2020)

## 🧩 Problem to Solve

본 논문은 현실 세계의 데이터가 가진 특성인 **데이터의 희소성(few), 레이블의 부재(unlabeled), 그리고 스트림 형태의 순차적 유입(stream)**이라는 세 가지 제약 조건을 동시에 해결하고자 한다. 기존의 딥러닝 솔루션들은 대개 지도 학습 기반이거나, 데이터가 균형 있게 분포되어 있다는 가정하에 설계되어 있어 실제 환경에서의 적용 범위가 제한적이다. 특히, 시간이 지남에 따라 진화하는 온라인 스트리밍 데이터의 경우 **치명적 망각(Catastrophic Forgetting)** 문제와 새로운 작업에 대한 **일반화(Generalization)** 능력이 동시에 요구된다.

따라서 본 연구의 목표는 레이블이 없고 클래스 간 불균형이 존재하는 상황에서도 새로운 작업을 빠르게 학습하고 이전의 지식을 유지할 수 있는 **FUSION(Few-shot UnSupervIsed cONtinual learning)** 설정과 이를 해결하기 위한 **MEML(Meta-Example Meta-Learning)** 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Self-attention 메커니즘을 이용해 각 태스크의 가장 핵심적인 특징을 응축한 'Meta-Example'을 생성하고, 이를 통해 학습 효율성과 일반화 성능을 극대화**하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **FUSION 설정 제안**: 비지도 학습, 퓨샷 학습, 지속 학습이 결합된 복합적인 설정(Unsupervised Meta-Continual Learning)을 정의하고, 불균형한 태스크(Unbalanced Tasks)가 일반화 성능에 미치는 영향을 분석하였다.
2.  **Meta-Example 기반의 단일 Inner Loop 업데이트**: MAML과 같은 기존 메타 학습 방식의 다중 Inner Loop 업데이트 대신, Attention을 통해 생성된 단일 Meta-Example에 대해서만 업데이트를 수행함으로써 학습 시간과 메모리 사용량을 획기적으로 줄였다.
3.  **불균형 태스크의 효용성 입증**: 데이터를 강제로 균형 있게 맞추는 것보다, 클러스터링을 통해 생성된 자연스러운 불균형 태스크를 유지하는 것이 특징의 다양성을 보존하여 결과적으로 더 나은 일반화 성능을 보임을 실험적으로 증명하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개하며 본 연구와의 차별점을 제시한다.

-   **지속 학습(Continual Learning)**: 치명적 망각을 방지하기 위해 Replay Buffer, 네트워크 확장, 정규화(Regularization), 증류(Distillation) 등의 기법이 사용되었다. 그러나 대부분은 지도 학습 기반이다.
-   **메타 학습(Meta-Learning)**: MAML과 같이 새로운 작업에 빠르게 적응하는 능력을 학습한다. 최근 비지도 메타 학습(CACTUs 등)이 제안되었으나, 이는 주로 독립 동일 분포(i.i.d) 데이터를 가정하며 지속 학습의 관점에서는 다루어지지 않았다.
-   **메타-지속 학습(Meta-Continual Learning)**: "잊지 않는 법을 학습(learn how not to forget)"하려는 시도가 있었으며, OML(Meta-learning representations for continual learning)과 같은 모델이 제안되었다. MEML은 이러한 구조를 계승하되, 비지도 환경과 불균형 태스크, 그리고 Attention 기반의 Meta-Example 업데이트라는 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 (FUSION)
전체 시스템은 네 가지 단계로 구성된다.
1.  **Embedding Learning**: DeepCluster 또는 ACAI를 사용하여 원본 데이터를 적절한 임베딩 공간으로 투영한다.
2.  **Unsupervised Task Construction**: 학습된 임베딩 공간에 K-means 클러스터링을 적용하여 각 클러스터를 하나의 클래스로 간주하고, 불균형한 태스크 분포 $p(T)$를 생성한다.
3.  **Meta-Continual Training**: 제안된 MEML 알고리즘을 통해 모델을 학습시킨다.
4.  **Meta-Continual Test**: 학습된 표현(Representation)은 고정하고, 새로운 클래스에 대해 예측 층만 미세 조정(Fine-tuning)하여 성능을 평가한다.

### 2. 네트워크 구조
모델은 크게 두 부분으로 나뉜다.
-   **Feature Extraction Network (FEN)**: 입력 데이터에서 특징 벡터를 추출하는 네트워크 ($\theta$로 파라미터화).
-   **Classification Network (CLN)**: 추출된 특징을 바탕으로 클래스를 분류하는 네트워크 ($W$로 파라미터화).

### 3. MEML 학습 절차 및 방정식

#### 3.1 Inner Loop: Meta-Example 생성 및 업데이트
각 태스크 $T_i = (S_{cluster}, S_{query})$가 샘플링되면, $S_{cluster}$에 속한 모든 샘플 $X_{0:K}$를 FEN에 통과시켜 특징 벡터 $R_{0:K}$를 얻는다.

이후, Attention 함수 $f_\rho$를 통해 각 샘플의 중요도인 어텐션 계수 $\alpha$를 계산한다.
$$\alpha_{0:K} = \text{Softmax}[f_\rho(R_{0:K})]$$

이 계수를 이용하여 해당 클래스의 핵심 특징을 대표하는 **Meta-Example ($ME$)**을 생성한다.
$$ME = \sum_{k=0}^K [R_k * \alpha_k]$$

최종적으로, 이 단일 $ME$를 사용하여 CLN의 파라미터 $W$와 Attention 파라미터 $\rho$를 업데이트한다.
$$\psi \leftarrow \psi - \alpha \nabla_\psi \ell_i(f_\psi(ME), Y_0)$$
여기서 $\psi = \{W, \rho\}$이며, $\alpha$는 Inner loop 학습률이다.

#### 3.2 Outer Loop: 일반화 및 망각 방지
Outer loop에서는 전체 네트워크 파라미터 $\phi = \{\theta, W, \rho\}$를 업데이트한다. 이때, 현재 클래스의 데이터와 전체 경로에서 무작위로 샘플링된 데이터의 앙상블인 $S_{query}$를 입력으로 사용하여 손실 함수를 계산한다.
$$\phi \leftarrow \phi - \beta \nabla_\phi \ell_i(f_\phi(X_{0:Q}), Y_{0:Q})$$
여기서 $\beta$는 Outer loop 학습률이다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: Omniglot, Mini-ImageNet, SlimageNet64 (추가 실험), Cifar100 및 Cub (OoD 실험).
-   **비교 대상**: OML(기존 메타-지속 학습), MAML, CACTUs, 그리고 지도 학습 기반의 Oracle 모델.
-   **측정 지표**: Accuracy.

### 2. 주요 결과
-   **불균형 태스크의 우수성**: 데이터를 강제로 균형 있게 맞춘(Balanced) 설정보다, 자연스럽게 불균형한(Unbalanced) 설정에서 일반화 성능이 더 높게 나타났다. 이는 소규모 클러스터가 제공하는 데이터의 다양성이 일반화에 필수적임을 시사한다.
-   **Meta-Example의 효과**: 단일 $ME$ 업데이트 방식이 다중 업데이트나 단순 평균(mean) 방식보다 일관되게 높은 성능을 보였으며, 특히 퓨샷 설정에서 강점을 보였다.
-   **효율성**: MEML은 단일 Inner loop 업데이트를 수행하므로, OML 대비 학습 시간이 대폭 단축되었으며 GPU 메모리 사용량도 약 1/3 수준으로 감소하였다.
-   **클러스터 수의 영향**: 실제 클래스 수보다 더 많은 수의 클러스터를 설정했을 때(Mini-ImageNet의 경우 256개) 더 높은 성능을 보였다. 이는 미리 정의된 클래스 구분보다 데이터 자체의 구조적 정보를 활용하는 것이 더 유리함을 의미한다.
-   **OoD 일반화**: Mini-ImageNet에서 학습하고 Cifar100에서 테스트하는 등 분포 외(Out-of-Distribution) 작업에서도 비지도 접근 방식이 지도 학습 기반보다 더 강건한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 통찰
본 연구는 비지도 환경에서 데이터의 **'다양성(Variety)'**이 **'균형(Balance)'**보다 훨씬 중요하다는 점을 발견하였다. 특히, 큰 클러스터는 일반적인 특징을 학습하게 하고, 작은 클러스터는 희귀하지만 중요한 정보를 제공함으로써 모델이 더 넓은 범위의 데이터에 적응할 수 있게 한다. 또한, Attention 기반의 Meta-Example 생성은 계산 비용을 줄이면서도 각 태스크의 가장 대표성 있는 특징을 효과적으로 추출하여 학습의 효율성을 높였다.

### 한계 및 비판적 해석
-   **클러스터 수 결정 문제**: 실험을 통해 최적의 클러스터 수를 찾았으나, 실제 환경에서 정답 클래스 수를 모르는 상태에서 최적의 $K$값을 결정하는 명확한 기준이나 자동화된 방법론은 제시되지 않았다.
-   **가정 사항**: 임베딩 학습 단계에서 DeepCluster나 ACAI 같은 사전 학습 모델에 의존하고 있는데, 이 단계의 성능이 전체 파이프라인의 상한선(Upper bound)을 결정할 가능성이 크다.

## 📌 TL;DR

본 논문은 레이블이 없고 클래스 불균형이 존재하는 지속 학습 환경(FUSION)을 정의하고, 이를 해결하기 위한 **MEML** 알고리즘을 제안하였다. 핵심은 **Self-attention을 통해 각 클래스의 대표 특징인 'Meta-Example'을 생성하여 단 한 번의 업데이트만으로 빠르게 적응**하는 것이며, 데이터를 강제로 균형 맞추기보다 **불균형한 상태 그대로의 다양성을 활용**하는 것이 일반화 성능에 더 유리함을 입증하였다. 이 연구는 추후 레이블이 없는 대규모 스트리밍 데이터셋에서의 효율적인 적응형 모델 설계에 중요한 기반이 될 것으로 보인다.