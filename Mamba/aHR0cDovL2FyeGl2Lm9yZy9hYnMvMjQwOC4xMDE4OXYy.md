# Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models

Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico Kolter, and Albert Gu (2024/2025)

## 🧩 Problem to Solve

현대 언어 모델의 주류인 Transformer 아키텍처는 강력한 성능을 보여주지만, Self-attention 메커니즘의 연산 복잡도가 시퀀스 길이의 제곱에 비례하는 Quadratic-time 복잡도($O(T^2)$)를 가진다는 치명적인 단점이 있다. 이를 해결하기 위해 Mamba와 같은 State Space Models (SSMs)를 포함한 Subquadratic 아키텍처들이 제안되었으나, 이들은 일반적으로 최상위 Transformer 모델들에 비해 훨씬 적은 계산 자원과 데이터로 사전 학습되었다.

결과적으로, 이미 막대한 계산 자원이 투입되어 학습된 Transformer의 지식을 효율적인 Subquadratic 모델로 전이할 수 있다면, 추론 비용은 낮추면서도 높은 성능을 유지하는 모델을 구축할 수 있을 것이라는 점이 이 연구의 핵심 목표이다. 즉, Quadratic-time 모델의 지식을 Subquadratic 모델로 증류(Distillation)하여 효율성과 성능의 간극을 메우고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Transformer의 지식을 SSM으로 전이하기 위한 3단계 점진적 증류 프레임워크인 **MOHAWK** (Matrix Orientation, Hidden-State Alignment, Weight-Transfer and Knowledge Distillation)를 제안한 것이다.

이 방법론의 중심적인 직관은 Transformer의 Attention과 SSM 모두를 토큰 시퀀스에 대해 서로 다른 형태의 **Mixing Matrix(혼합 행렬)**를 적용하는 'Sequence Transformation'으로 볼 수 있다는 점이다. 따라서 단순한 최종 출력값의 모방이 아니라, 행렬 수준의 정렬부터 블록 수준의 정렬, 그리고 최종 모델 수준의 정렬로 이어지는 세밀한 단계별 증류를 통해 아키텍처 간의 구조적 차이를 극복하고 지식을 효율적으로 전이할 수 있다.

## 📎 Related Works

기존의 지식 증류(Knowledge Distillation) 연구는 주로 모델 압축(Compression), 즉 큰 Transformer를 더 작은 Transformer로 만드는 데 집중해 왔다. 일부 연구에서 Attention 행렬을 일치시키거나 블록의 출력을 정렬하는 시도가 있었으나, 이는 주로 동일 아키텍처 내에서의 최적화였으며 서로 다른 아키텍처(Transformer $\rightarrow$ SSM) 간의 전이와는 차이가 있다.

또한, Transformer를 RNN이나 Linear Attention으로 변환하려는 시도가 있었으나, 본 논문은 Mamba-2와 같이 보다 표현력이 뛰어난 최신 SSM 구조를 활용하며, 단순히 가중치를 전이하는 수준을 넘어 3단계의 정밀한 정렬 과정을 거친다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. Sequence Transformation 및 Matrix Mixer의 관점
논문은 시퀀스 모델의 동작을 $Y = MX$ 형태의 행렬 곱으로 정의하며, 여기서 $M$을 **Matrix Mixer**라고 부른다. Transformer의 Self-attention은 $\text{Softmax}(QK^\top)$라는 행렬을 사용하는 Mixer이며, Mamba-2와 같은 SSM은 구조화된 상태 공간 이중성(Structured State Space Duality, SSD)을 통해 Semi-separable matrix 형태의 Mixer를 구현한다.

### 2. MOHAWK의 3단계 증류 절차

#### Stage 1: Matrix Orientation (행렬 방향 정렬)
가장 미세한 단계로, 학생 모델(SSM)의 Mixer 행렬이 교사 모델(Transformer)의 Attention 행렬을 최대한 가깝게 모방하도록 학습시킨다.
$$ \min_{\theta} \| \text{TeacherMixer}(u) - \text{StudentMixer}_{\theta}(u) \|_F $$
여기서 $\| \cdot \|_F$는 Frobenius norm을 의미한다. 이 단계에서는 다른 구성 요소들을 고정하거나 identity function으로 설정하여, 오직 행렬 간의 거리만을 최소화하는 데 집중한다.

#### Stage 2: Hidden-State Alignment (은닉 상태 정렬)
행렬 수준의 정렬 이후, 각 블록(Block)의 최종 출력값(Hidden state)을 일치시키는 단계이다.
$$ \min_{\theta} \| \text{AttnBlock}(u) - \text{StudentMixerBlock}_{\theta}(u) \|_2 $$
단순 행렬 모방을 넘어, Gating 메커니즘 등을 포함한 블록 전체의 기능적 출력을 정렬함으로써 교사 모델의 분포를 더 잘 학습하게 한다.

#### Stage 3: Weight-Transfer and Knowledge Distillation (가중치 전이 및 지식 증류)
마지막으로, Mixer를 제외한 나머지 가중치(MLP, Embedding, Layer Norm 등)를 교사 모델에서 학생 모델로 직접 전이한다. 이후 전체 모델을 end-to-end로 학습시키며 교사 모델의 Logits 분포를 따르도록 하는 Knowledge Distillation을 수행한다.
$$ \min_{\theta} L_{CE}(\text{TeacherModel}(x), \text{StudentModel}_{\theta}(x)) $$
특히, 지식이 주로 MLP 블록에 저장되어 있다는 가설에 따라 MLP 가중치를 동결(Freeze)한 채 학습을 진행해도 성능이 유지됨을 확인하였다.

### 3. Phi-Mamba 아키텍처
본 연구는 Phi-1.5(1.3B)를 교사 모델로, Mamba-2 기반의 **Phi-Mamba**를 학생 모델로 설정하였다. Mamba-2의 구조를 수정하여 Transformer의 Multi-head 구조와 대응하도록 변경하였으며, 불필요한 비선형 활성화 함수나 정규화 층을 제거하여 증류 효율을 높였다. 또한, 일부 Attention 층을 유지하는 **Hybrid Phi-Mamba** 버전도 제안하였다.

## 📊 Results

### 1. 정량적 성능 평가
Phi-Mamba는 단 30억(3B) 개의 토큰만을 사용하여 증류되었으며, 이는 기존에 스크래치(scratch)부터 학습된 Mamba-2(315B 토큰) 대비 1% 미만의 데이터만 사용한 결과이다. 그럼에도 불구하고 Winogrande, ARC-C 등 주요 벤치마크에서 기존의 모든 오픈소스 비-Transformer 모델들을 압도하는 성능을 보였다.

- **Phi-Mamba (1.5B)**: Winogrande 71.7% (Mamba-2의 60.9% 대비 대폭 상승)
- **Hybrid Phi-Mamba (1.5B)**: 평균 성능 66.0% 달성 (교사 모델인 Phi-1.5의 67.2%에 근접)

### 2. 분석 및 어블레이션 연구
- **단계별 중요성**: Stage 3(최종 증류)만 수행했을 때보다 Stage 1 $\rightarrow$ 2 $\rightarrow$ 3의 순차적 과정을 거쳤을 때 성능이 비약적으로 향상됨을 확인하였다.
- **행렬 표현력**: Mamba-2의 SSD 구조가 Linear Attention이나 Toeplitz 구조보다 Transformer의 Attention 행렬을 훨씬 더 정밀하게 근사(Approximation)할 수 있음을 Frobenius distance 측정을 통해 증명하였다.
- **하이브리드 구성**: Attention 층을 균일하게 배치한 하이브리드 모델이 성능이 가장 좋았으며, Attention 층의 개수가 많을수록 성능이 향상되는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 Transformer의 'Quadratic Knowledge'가 충분히 표현력이 있는 Mixer(예: Mamba-2)를 통해 'Subquadratic Model'로 전이될 수 있음을 입증하였다. 특히, 모델의 학습 가능성(Trainability)과 증류 가능성(Distillability)은 서로 다른 속성일 수 있으며, 적절한 정렬 과정을 거친다면 매우 적은 데이터로도 강력한 SSM 모델을 구축할 수 있다는 점이 고무적이다.

다만, 하이브리드 모델의 경우 여전히 일부 Attention 층이 남아있어 완전히 $O(T)$의 복잡도를 달성하지는 못한다는 점과, 증류 과정에서 발생하는 Loss spike와 같은 학습 불안정성 문제가 언급되었다. 또한, 교사 모델의 MLP 지식을 그대로 활용했다는 점은 SSM 자체가 독립적으로 학습되었을 때보다 전이 학습되었을 때 더 효율적일 수 있음을 시사한다.

## 📌 TL;DR

이 논문은 Transformer의 강력한 성능(Quadratic Knowledge)을 Mamba-2와 같은 효율적인 SSM으로 전이하기 위한 3단계 증류 프레임워크 **MOHAWK**를 제안한다. 행렬 정렬 $\rightarrow$ 은닉 상태 정렬 $\rightarrow$ 최종 Logits 증류 순으로 진행되는 이 방법론을 통해, **Phi-Mamba**는 기존 SSM 모델들이 사용한 데이터의 1% 미만만을 사용하여 더 높은 성능을 달성하였다. 이는 향후 거대 언어 모델의 추론 비용을 획기적으로 줄이면서도 성능을 유지하는 새로운 모델 구축 경로를 제시한다.