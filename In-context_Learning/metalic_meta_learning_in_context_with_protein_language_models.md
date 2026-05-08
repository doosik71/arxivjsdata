# METALIC: META-LEARNING IN-CONTEXT WITH PROTEIN LANGUAGE MODELS

Jacob Beck, Shikha Surana, Manus McAuliffe, Oliver Bent, Thomas D. Barrett, Juan Jose Garau Luis, & Paul Duckworth (2025)

## 🧩 Problem to Solve

본 논문은 단백질의 생물물리적 및 기능적 특성인 **Fitness**(적합도)를 예측하는 문제를 다룬다. 단백질 Fitness 예측은 신약 개발, 의료 연구, 농업 등 다양한 분야에서 매우 중요하지만, 실험실 내($in\ vitro$) 주석 데이터가 매우 희소하다는 치명적인 한계가 있다.

기존의 단백질 언어 모델(Protein Language Models, PLMs)은 일반적으로 대규모의 일반 단백질 서열 데이터로 사전 학습된 후, 특정 Fitness 예측 작업에 대해 Fine-tuning되거나 Zero-shot으로 적용된다. 그러나 데이터가 극도로 부족한 경우, 모델은 단백질 서열의 Likelihood(우도)와 Fitness 점수 사이에 강한 상관관계가 있다는 단순한 가정에 의존하게 된다. 이러한 접근 방식은 데이터가 없는 환경에서 일반화 능력이 떨어지며, 데이터가 적을 때는 적절한 회귀(Regression) 모델을 학습시키기 어렵다는 문제가 있다.

따라서 본 논문의 목표는 **다양한 Fitness 예측 작업의 분포를 통한 Meta-learning을 도입하여, 데이터가 매우 적거나 없는(low-data settings) 환경에서도 새로운 Fitness 예측 작업에 효과적으로 적응하는 모델을 구축하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **In-context Meta-learning과 Fine-tuning을 결합**하는 것이다.

1. **In-context Meta-learning 도입**: 특정 작업의 데이터가 부족하더라도, 다른 유사한 Fitness 예측 작업들로부터 '어떻게 PLM 임베딩을 사용하여 Fitness를 예측할 것인가'에 대한 지식을 학습한다.
2. **효율적인 적응 파이프라인**: 계산 비용이 매우 높은 Higher-order gradients(고차 미분)를 사용하는 기존의 Gradient-based meta-learning과 달리, In-context learning을 통해 빠르게 적응하고 이후에 단순한 Fine-tuning을 수행하는 구조를 제안한다.
3. **데이터 희소성 극복**: Meta-training 단계에서 학습된 Prior를 통해, Zero-shot 환경에서도 SOTA(State-of-the-art) 성능을 달성하였으며, Few-shot 환경에서는 기존 SOTA 모델 대비 파라미터 수를 18배나 줄이면서도 강력한 성능을 유지하였다.

## 📎 Related Works

### Meta-Learning

Meta-learning은 작업의 분포를 통해 학습하여 새로운 작업에 빠르게 적응하는 것을 목표로 한다.

- **Gradient-based meta-learning**: MAML이나 Reptile처럼 파라미터 초기화를 학습하여 몇 번의 Gradient step만으로 적응하는 방식이다. 하지만 대형 모델에서는 Meta-gradient 계산 비용이 매우 크다.
- **In-context meta-learning**: 모델이 컨텍스트 내의 데이터를 조건으로 사용하여 적응하는 방식이다. 계산 효율적이지만, Gradient 기반 방식보다 Out-of-distribution(OOD) 작업에 대한 일반화 성능이 떨어지는 경향이 있다.

### PLM-based Fitness Prediction

기존 방식들은 PLM이 학습한 서열의 Likelihood가 Fitness와 상관관계가 높다는 가정에 기반한다. 특히 Masked Language Model의 경우, 각 아미노산이 Fitness에 독립적으로 기여한다는 가정을 사용하기도 한다.

### In-Context PLMs

ProteinNPT와 같은 모델들이 In-context 데이터를 사용하지만, 이들은 컨텍스트를 사용하는 '방법' 자체를 Meta-learn 하지는 않았다. 본 논문은 이러한 아키텍처를 활용하되, Meta-learning 단계를 추가하여 컨텍스트 활용 능력을 극대화했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

Metalic은 크게 **PLM 임베딩 $\rightarrow$ Axial Attention $\rightarrow$ MLP**의 구조를 가진다.

1. **Embedding**: 사전 학습된 ESM2-8M 모델의 세 번째 레이어를 사용하여 단백질 서열을 residue-level 임베딩으로 변환한다.
2. **Concatenation**: Support set의 서열 임베딩과 Fitness 점수(선형 층을 통해 투영됨)를 결합한다. Query set의 경우, Fitness 값 대신 학습 가능한 단일 임베딩을 사용한다.
3. **Axial Attention**: 계산 복잡도를 $O(K^2 L^2)$에서 $O(K^2 + L^2)$로 줄이기 위해, 서열 방향(row)과 서열 간 방향(column)으로 나누어 Self-attention을 적용하는 Axial Attention 블록을 사용한다. 여기서 $K$는 shot 수, $L$은 단백질 길이이다.
4. **Prediction**: Mean-pooling된 서열 임베딩과 Fitness 임베딩을 MLP에 통과시켜 최종 예측값 $v^{(Q)}_i$를 생성한다.

### Meta-Training 및 손실 함수

본 모델은 절대적인 수치 예측보다 상대적인 순위 예측에 집중하는 **Preference-based objective**를 사용한다. 두 서열 $x^{(Q)}_i$와 $x^{(Q)}_j$ 중 어느 것이 더 높은 Fitness를 가졌는지를 이진 분류 문제로 정의한다.

두 서열의 예측값 차이를 Sigmoid 함수에 통과시켜 확률을 계산한다:
$$p(y^{(Q)}_i > y^{(Q)}_j) = \sigma(v^{(Q)}_i - v^{(Q)}_j)$$

이에 따른 손실 함수는 다음과 같이 정의된다:
$$L(\theta, D^{(Q)}_T, D^{(S)}_T) = -\sum_{i=1}^{N^{(Q)}} \sum_{j \neq i}^{N^{(Q)}} I(y^{(Q)}_i > y^{(Q)}_j) \log \sigma(v^{(Q)}_i - v^{(Q)}_j)$$
여기서 $I$는 지시 함수(Indicator function)이다. Meta-learning의 최종 목표는 전체 작업 분포 $\mathcal{D}$에 대해 이 손실의 기댓값을 최소화하는 파라미터 $\theta$를 찾는 것이다:
$$J(\theta, \mathcal{D}) = -\mathbb{E}_{D_T \in \mathcal{D}} \mathbb{E}_{(D^{(S)}_T, D^{(Q)}_T) \in D_T} L(\theta, D^{(Q)}_T, D^{(S)}_T)$$

### Inference-time Fine-tuning

추론 단계에서 일반화 능력을 높이기 위해 Fine-tuning을 수행한다. 이때 Support set의 라벨이 입력으로 이미 들어가 있기 때문에, 단순히 학습하면 모델이 데이터를 암기(Memorization)하는 문제가 발생한다.

이를 해결하기 위해 **Sub-sampling** 전략을 사용한다. 전체 Support set $D^{(S)}_T$를 더 작은 규모의 가상 Support set $D^{(S')}_T$와 Query set $D^{(Q')}_T$로 나누어 학습함으로써, 모델이 보지 못한 데이터에 대해 일반화하도록 유도한다. 학습 후에는 다시 전체 Support set을 컨텍스트로 넣어 추론한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ProteinGym 벤치마크 (단일 변이 single-mutant 121개, 다중 변이 multi-mutant 68개 작업).
- **지표**: Spearman rank correlation ($\rho$).
- **설정**: Zero-shot ($N^{(S)}=0$), Few-shot ($N^{(S)}=16, 128$).

### 주요 결과

1. **Zero-shot 성능**: 단일 변이 작업에서 Metalic은 $\rho \approx 0.48$을 기록하며 VESPA, ESM1-v 등 기존의 거대 모델들을 제치고 SOTA를 달성하였다. 특히 베이스라인인 ESM2-8M ($\rho \approx 0.12$) 대비 비약적인 상승을 보여 Meta-learning의 효과를 입증했다.
2. **Few-shot 성능**: 16-shot 설정에서 가장 강력한 성능을 보였으며, 128-shot에서는 PoET 등과 대등한 성능을 보였다. 특히 SOTA급 성능을 내면서도 파라미터 수는 훨씬 적어(약 18배 차이) 효율성이 매우 높다.
3. **다중 변이(Multi-mutant) 성능**: 다중 변이 작업에서는 SOTA에 도달하지 못했으나, 학습 데이터(작업 수)가 증가함에 따라 성능이 선형적으로 증가하는 경향을 보여 향후 데이터 확충 시 개선 가능성을 제시하였다.
4. **Ablation Study**: Meta-training을 제거했을 때 성능이 가장 크게 하락($\rho \approx -0.046$ in zero-shot)하여, 단순한 아키텍처나 임베딩보다 **Meta-learning 과정 자체가 성능의 핵심**임을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 발견

- **비지도 In-context 학습의 창발**: Zero-shot 설정에서도 Query set 내의 다른 단백질 서열 정보를 활용하여 성능이 향상됨을 확인하였다. Attention map 분석 결과, 모델이 학습 과정에서 특정 유용한 단백질에 주의를 기울이는 '수직선' 형태의 패턴이 나타났으며, 이는 Unsupervised in-context adaptation이 일어나고 있음을 시사한다.
- **계산 효율성**: 고비용의 Meta-gradient를 계산하지 않고도, In-context learning과 단순 Fine-tuning의 조합만으로 Gradient-based meta-learning(Reptile 등)보다 우수한 혹은 대등한 성능을 낼 수 있음을 보였다.

### 한계 및 비판적 해석

- **다중 변이 데이터의 부족**: 단일 변이에서는 압도적이지만 다중 변이에서 SOTA를 달성하지 못한 점은 Meta-learning 모델의 특성상 학습 데이터(작업의 수)에 매우 의존적임을 보여준다.
- **보조 모델 의존성**: ESM-IF1과 같은 구조 기반 모델의 예측값을 추가했을 때(Metalic-AuxIF) 성능이 크게 향상되는 것은, 여전히 PLM 임베딩만으로는 단백질의 3차원 구조적 특성을 완벽히 캡처하기 어렵다는 점을 시사한다.

## 📌 TL;DR

본 논문은 단백질 Fitness 예측의 데이터 희소성 문제를 해결하기 위해 **In-context Meta-learning과 Fine-tuning을 결합한 Metalic**을 제안한다. 이 모델은 거대한 파라미터 수에 의존하는 대신, 다양한 작업 분포를 통해 '학습하는 방법'을 배워 데이터가 극도로 적은 환경에서 탁월한 적응력을 보인다. 특히 Zero-shot 단일 변이 예측에서 SOTA를 달성하였으며, 추론 시의 효율적인 Fine-tuning 전략을 통해 계산 비용과 성능의 최적 균형을 찾았다. 이 연구는 향후 단백질 설계 및 엔지니어링 분야에서 데이터 부족 문제를 해결하는 핵심적인 방법론이 될 가능성이 높다.
