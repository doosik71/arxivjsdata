# Learning Mamba as a Continual Learner: Meta-learning State Space Models for Efficient Continual Learning

Chongyang Zhao, Dong Gong (2024)

## 🧩 Problem to Solve

본 논문은 비정상 데이터 스트림(non-stationary data stream)으로부터 효율적으로 학습하면서도 과거의 데이터를 모두 저장하거나 재계산하지 않는 Continual Learning (CL)의 효율성 문제를 해결하고자 한다. 최근에는 CL을 시퀀스 예측(Sequence Prediction, SP) 문제로 정의하고, 효율적인 Continual Learner를 메타 학습시키는 Meta-Continual Learning (MCL) 프레임워크가 제안되었다.

기존의 MCL에서는 강력한 시퀀스 모델링 능력을 가진 Transformer가 주로 사용되었다. 하지만 Transformer는 모든 과거 표현을 저장하기 위해 선형적으로 증가하는 Key-Value (KV) 캐시에 의존한다. 이는 모든 데이터를 저장하지 않아야 한다는 CL의 핵심 목표와 정면으로 충돌하며, 메모리와 계산 비용을 증가시켜 효율성을 제한한다. 반면, Linear Transformer나 Performer와 같은 Attention-free 모델들은 고정된 크기의 hidden state를 사용하여 효율적이지만, MCL 작업에서의 표현력(expressive power)이 부족하여 성능이 낮다는 한계가 있었다.

따라서 본 논문의 목표는 Transformer 수준의 강력한 성능을 유지하면서도, 고정된 크기의 상태 공간을 사용하여 메모리 효율성을 확보한 새로운 메타 학습 기반 Continual Learner를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Selective State Space Model (SSM)인 Mamba를 MCL 프레임워크에 도입하여 **MambaCL**을 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1.  **Mamba 기반의 MCL 구현**: Mamba의 고정된 크기 hidden state와 선형 시간 복잡도를 활용하여, KV 캐시 없이도 효율적으로 데이터를 압축하고 학습하는 Continual Learner를 설계하였다.
2.  **Selectivity Regularization 도입**: Mamba는 Transformer와 달리 명시적인 Attention 메커니즘이 없어 메타 학습 시 수렴이 어렵고 학습 속도가 느리다. 이를 해결하기 위해 SSM, Linear Transformer, 그리고 Vanilla Transformer 사이의 수학적 관계를 이용하여, 모델의 선택적 행동(selectivity behavior)을 가이드하는 정규화 항을 도입하였다.
3.  **다양한 일반화 시나리오 검증**: 단순한 MCL 설정을 넘어, 학습 시 보지 못한 긴 시퀀스 길이, 심한 도메인 시프트(Domain Shift), 노이즈 섞인 입력 등 실제 환경에 가까운 시나리오에서 Mamba의 강건성(Robustness)을 입증하였다.

## 📎 Related Works

**1. Continual Learning (CL)**
기존 CL 연구는 재현(Replay), 정규화(Regularization), 아키텍처 기반 방법 등을 통해 치명적 망각(Catastrophic Forgetting)을 방지하려 했다. 하지만 재현 기반 방법은 메모리 및 개인정보 보호 제약이 있으며, 정규화 방법은 모델 용량의 한계가 존재한다.

**2. Meta-Continual Learning (MCL)**
MCL은 여러 CL 에피소드를 통해 일반화된 학습 능력을 갖춘 모델을 메타 학습시키는 접근 방식이다. 특히 최근 연구들은 CL을 시퀀스 모델링 문제로 재정의하여, 데이터 스트림을 컨텍스트로 사용하여 새로운 쿼리를 예측하는 방식을 취하고 있다.

**3. Transformer 및 Attention-free 모델**
Transformer는 Attention 메커니즘을 통해 우수한 성능을 보이지만 $O(N^2)$의 복잡도를 가진다. 이를 해결하기 위한 Linear Transformer 등은 $O(N)$의 복잡도를 가지며 고정된 상태 크기를 갖지만, MCL에서의 성능은 Transformer에 비해 현저히 낮았다. Mamba는 Selective SSM을 통해 이 두 가지(효율성과 성능)의 간극을 메울 수 있는 대안으로 제시되었다.

## 🛠️ Methodology

### 전체 파이프라인
MambaCL은 메타 학습된 Mamba 모델 $f_\theta(\cdot)$가 온라인 데이터 스트림을 처리하며 hidden state를 업데이트하고, 이를 통해 테스트 샘플에 대한 예측을 수행하는 구조이다. 전체 과정은 다음과 같은 시퀀스 예측 문제로 공식화된다:
$$(x^{train}_1, y^{train}_1, \dots, x^{train}_T, y^{train}_T, x^{test}_k) \rightarrow y^{test}_k$$

### Mamba 및 SSM의 적용
Mamba는 입력에 따라 변하는 파라미터를 가진 Selective SSM을 사용한다. 기본적으로 SSM은 다음과 같은 상태 방정식으로 표현된다:
$$h_t = A^t h_{t-1} + B^t z_t, \quad u_t = C^t h_t + D z_t$$
여기서 $h_t$는 고정된 크기의 hidden state이며, $A^t, B^t, C^t$는 입력 토큰 $z_t$에 따라 동적으로 생성되는 파라미터이다. MambaCL은 각 입력 차원별로 독립적인 SSM을 적용하여 정보를 압축하고 저장함으로써 Transformer의 KV 캐시 없이도 컨텍스트를 유지한다.

### 학습 목표 및 손실 함수
모델의 파라미터 $\theta$는 다음과 같은 메타 학습 목적 함수를 통해 최적화된다:
$$\min_\theta \mathbb{E}_{(D^{train}, D^{test}) \sim P(X,Y)} \sum_{(x^{test}, y^{test}) \in D^{test}} \ell(f_\theta((D^{train}, x^{test})), y^{test})$$

### Selectivity Regularization
Mamba의 학습 안정성을 높이기 위해, 쿼리 토큰과 관련된 이전 토큰들 사이의 연관성을 강제하는 정규화 항 $\ell_{slct}$를 추가한다. 
- **연관성 지표**: 정답 레이블이 같은 샘플들 간의 관계를 나타내는 ground truth 벡터 $p$를 정의한다.
- **Mamba의 연관성**: Mamba의 파라미터 $C^t$와 $B^t$ 사이의 관계를 통해 모델이 내부적으로 생성하는 연관성 패턴 $q^{Mamba}$를 유도한다.
- **손실 함수**: KL Divergence를 사용하여 모델의 예측 패턴과 실제 정답 패턴 사이의 차이를 최소화한다.
$$\ell_{slct}((x,y)) = KL(p_{idx((x,y))}, q^*_{idx((x,y))})$$
최종 손실 함수는 $\mathcal{L} = \mathcal{L}_{MCL} + \lambda \ell_{slct}$ 형태로 구성된다.

## 📊 Results

### 실험 설정
- **데이터셋**: Cifar-100, ImageNet-1K, Celeb, Omniglot (일반 분류), CUB-200, Stanford Dogs/Cars (세밀 분류), DomainNet (도메인 시프트), Sine wave/Rotation (회귀).
- **비교 대상**: OML, Vanilla Transformer, Linear Transformer, Performer.
- **평가 지표**: 분류 정확도(%) 및 회귀 오차(MSE).

### 주요 결과
1.  **분류 성능**: 일반 이미지 분류 및 세밀 분류(Fine-grained) 작업 모두에서 MambaCL은 Linear Transformer와 Performer를 압도하며, Vanilla Transformer와 대등하거나 더 높은 성능을 보였다. 특히 세밀 분류 작업(Table 3)에서 Mamba의 우위가 뚜렷하게 나타났다.
2.  **효율성**: MambaCL은 Transformer보다 훨씬 적은 파라미터 수를 가지며(Mamba 5.4M vs TF 9.2M), 추론 속도는 약 2.6배 더 빠르다(858 ep/s vs 325 ep/s).
3.  **일반화 능력 (Generalization)**:
    - **시퀀스 길이**: 학습 시보다 더 긴 시퀀스(더 많은 태스크나 샷 수)가 들어왔을 때, Transformer는 성능이 급격히 하락(Meta-overfitting)하는 반면, Mamba는 상대적으로 완만한 성능 저하를 보이며 강건함을 입증했다.
    - **도메인 시프트**: DomainNet 데이터셋 실험에서 Mamba는 학습 데이터와 분포가 크게 다른 도메인(예: Quickdraw)에서도 Transformer보다 높은 적응력을 보였다.
    - **노이즈 강건성**: 입력 임베딩에 가우시안 노이즈를 추가했을 때, Transformer는 성능이 급락했으나 Mamba는 높은 정확도를 유지했다.

## 🧠 Insights & Discussion

**1. Transformer의 Meta-overfitting과 Mamba의 강점**
시각화 분석 결과, Transformer는 학습 시의 시퀀스 길이나 특정 위치에 대한 패턴 편향(Positional Bias)을 학습하는 경향이 있어, 테스트 시 길이가 달라지면 제대로 작동하지 않는 '메타 오버피팅' 문제가 발생한다. 반면 Mamba는 재귀적으로 업데이트되는 latent state를 통해 전역적인 시퀀스 정보를 누적하므로, 특정 위치가 아닌 콘텐츠 중심의 연관성을 학습하여 일반화 능력이 더 뛰어나다.

**2. 전역 정보 캡처 능력**
세밀 분류(Fine-grained) 작업에서의 우수한 성적은 Mamba가 시퀀스 전체의 미세한 차이를 포착하여 상태에 저장하는 능력이 뛰어남을 시사한다. Transformer의 Attention은 국소적인 독립 표현에 의존하는 경향이 있는 반면, Mamba는 선택적 SSM을 통해 필요한 정보를 효율적으로 압축하고 유지한다.

**3. 한계 및 확장 가능성**
본 연구는 주로 온라인 CL 설정에 집중하였으며, 오프라인 설정이나 더 거대한 규모의 파운데이션 모델로의 확장 가능성에 대해서는 추가 연구가 필요하다. 또한 Mamba에 Mixture-of-Experts (MoE)를 결합했을 때 성능이 더욱 향상됨을 확인하여, 향후 모델 용량 확장 전략에 대한 단서를 제공하였다.

## 📌 TL;DR

본 논문은 메모리 효율성이 낮은 Transformer 기반의 Meta-Continual Learning을 개선하기 위해, 고정 크기 상태 공간을 사용하는 **MambaCL**을 제안한다. Mamba의 Selective SSM을 CL의 시퀀스 예측 문제에 접목하고, 학습 안정화를 위한 **Selectivity Regularization**을 도입하였다. 실험 결과, MambaCL은 기존 Attention-free 모델보다 월등히 뛰어나며, Transformer보다 적은 자원으로 동등 이상의 성능을 낼 뿐만 아니라, 시퀀스 길이 변화, 도메인 시프트, 노이즈 등 극한의 환경에서 훨씬 강력한 일반화 능력을 보여주어 효율적인 Continual Learning의 새로운 방향성을 제시하였다.