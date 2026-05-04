# From Sparse to Soft Mixture of Experts

Joan Puigcerver, Carlos Riquelme, Basil Mustafa, Neil Houlsby (2024)

## 🧩 Problem to Solve

본 논문은 기존의 Sparse Mixture of Experts (MoEs) 아키텍처가 모델의 용량을 효율적으로 확장할 수 있음에도 불구하고 가진 여러 구조적 한계점을 해결하고자 한다. Sparse MoE는 각 토큰을 특정 전문가(Expert)에게 할당하는 이산적(discrete) 최적화 문제를 다루는데, 이 과정에서 다음과 같은 문제들이 발생한다.

첫째, **학습의 불안정성(Training instability)**과 **토큰 드롭핑(Token dropping)** 문제이다. 특정 전문가에게 토큰이 몰리거나, 일부 토큰이 어떤 전문가에게도 할당되지 못해 정보가 손실되는 현상이 발생한다. 둘째, 전문가의 수를 무분별하게 확장하기 어렵다는 점과 파인튜닝(Finetuning) 시 효율성이 떨어진다는 점이다. 셋째, 하드웨어 가속기에서 속도를 저하시키는 `top-k` 정렬 작업 및 비결정론적(non-deterministic) 라우팅 문제가 존재한다.

결과적으로 본 연구의 목표는 MoE의 이점인 '낮은 추론 비용으로의 모델 용량 확장'은 유지하면서, 위에서 언급한 이산적 라우팅의 문제점들을 해결할 수 있는 완전 미분 가능한(fully-differentiable) 새로운 MoE 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 **Hard Assignment(이산적 할당)** 방식을 **Soft Assignment(부드러운 할당)** 방식으로 전환하는 것이다.

기존 MoE가 "어떤 토큰을 어떤 전문가에게 보낼 것인가"를 결정했다면, Soft MoE는 모든 입력 토큰들의 가중 합(weighted combination)을 통해 '슬롯(slot)'을 생성하고, 이 슬롯을 전문가에게 전달한다. 즉, 개별 토큰을 전문가에게 직접 매핑하는 대신, 토큰들을 섞어(mixing) 전문가가 처리할 수 있는 입력 형태를 만드는 방식을 취한다. 이를 통해 라우팅 과정이 완전히 연속적이고 미분 가능해지며, 토큰 드롭핑이나 전문가 불균형 문제에서 완전히 자유로워진다.

## 📎 Related Works

논문에서는 기존의 Sparse MoE 라우팅 알고리즘들을 다음과 같이 분류하고 그 한계를 지적한다.

- **Tokens Choice (Top-k):** 각 토큰이 가장 높은 점수를 가진 $k$개의 전문가를 선택한다. 이 방식은 토큰 드롭핑과 전문가 불균형 문제에 취약하다.
- **Experts Choice:** 각 전문가가 가장 점수가 높은 $C$개의 토큰을 선택한다. 토큰 드롭핑 문제는 여전하며, 일부 토큰이 여러 전문가에 의해 중복 선택되는 불균형이 발생한다.
- **기타 최적화 기법:** Linear Programming, Optimal Transport, 강화학습 등을 이용한 할당 방식이 제안되었으나, 대부분 이산적인 성격을 띠어 미분 불가능하거나 계산 비용이 매우 높다는 한계가 있다.

Soft MoE는 이러한 이산적 할당 방식에서 벗어나 모든 토큰의 정보를 가중 평균하여 전달하는 연속적인 매핑을 사용함으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인

Soft MoE 레이어는 입력 토큰들을 섞어 슬롯을 만들고, 이를 전문가에게 통과시킨 뒤 다시 토큰 형태로 복원하는 과정을 거친다.

### 2. 상세 동작 및 방정식

입력 토큰 시퀀스를 $X \in \mathbb{R}^{m \times d}$ (여기서 $m$은 토큰 수, $d$는 차원)라고 할 때, 과정은 다음과 같다.

**가. Dispatch Weights (분배 가중치) 계산**
학습 가능한 파라미터 $\Phi \in \mathbb{R}^{d \times (n \cdot p)}$를 이용하여 입력 토큰과 슬롯 간의 로짓(logits)을 계산하고, 슬롯 방향(열 방향)으로 Softmax를 적용하여 분배 가중치 $D$를 구한다.
$$D_{ij} = \frac{\exp((X\Phi)_{ij})}{\sum_{i'=1}^{m} \exp((X\Phi)_{i'j})}$$

**나. Input Slots 생성**
위에서 구한 $D$를 이용하여 모든 입력 토큰의 가중 합으로 입력 슬롯 $\tilde{X}$를 생성한다.
$$\tilde{X} = D^\top X$$
이렇게 생성된 $\tilde{X}$의 각 행(슬롯)은 해당 전문가 $f_i$에게 입력으로 전달된다.

**다. Expert 처리**
각 슬롯은 할당된 전문가 함수 $f$ (일반적으로 MLP)를 통해 처리되어 출력 슬롯 $\tilde{Y}$가 된다.
$$\tilde{Y}_i = f_{\lfloor i/p \rfloor}(\tilde{X}_i)$$

**라. Combine Weights (결합 가중치) 및 최종 출력**
출력 슬롯들을 다시 원래 토큰 형태로 복원하기 위해, 토큰 방향(행 방향)으로 Softmax를 적용한 결합 가중치 $C$를 계산하고 가중 합을 구한다.
$$C_{ij} = \frac{\exp((X\Phi)_{ij})}{\sum_{j'=1}^{n \cdot p} \exp((X\Phi)_{ij'})}, \quad Y = C\tilde{Y}$$

### 3. 학습 안정화를 위한 L2 Normalization

모델 차원 $d$가 커질수록 Softmax 결과가 One-hot 벡터로 수렴하며 학습이 불안정해지는 현상이 발생한다. 이를 방지하기 위해 $X$와 $\Phi$에 대해 $L_2$ 정규화를 수행하고 학습 가능한 스칼라 값을 곱해주는 방식을 사용한다.
$$\text{normalized\_logits} = \text{l2\_normalize}(X) \cdot (\text{scale} \cdot \text{l2\_normalize}(\Phi))$$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** JFT-4B (사전 학습), ImageNet-1k (10-shot 및 파인튜닝 평가).
- **비교 대상:** Dense ViT, Sparse MoE (Tokens Choice, Experts Choice).
- **측정 지표:** JFT-4B Precision-at-1, ImageNet 10-shot Accuracy, 추론 시간(ms/img) 및 FLOPs.

### 2. 주요 결과

- **훈련 효율성 (Pareto Frontier):** 동일한 훈련 비용(FLOPs 또는 TPU-days) 대비 Soft MoE가 Dense ViT 및 다른 Sparse MoE보다 월등히 높은 성능을 보였다.
- **추론 속도 및 성능:**
  - **Soft MoE B/16** 모델은 ViT H/14와 유사하거나 더 높은 성능을 내면서도 추론 속도는 **5.7배 더 빠르며**, FLOPs 기준으로는 **10.4배 더 효율적**이다.
  - **Soft MoE L/16** 모델은 모든 ViT 모델보다 뛰어난 성능을 보이면서도 ViT H/14보다 추론 속도가 약 2배 빠르다.
- **확장성 (Scalability):** 전문가의 수 $n$을 대폭 늘려도 전체 슬롯 수만 일정하다면 시간 복잡도가 일정하게 유지된다. 이는 `top-k` 연산이 없는 Soft MoE의 구조적 이점으로, 수천 개의 전문가를 사용해도 추론 속도 저하가 거의 없다.
- **범용성:** 이미지-텍스트 대조 학습(Contrastive Learning) 실험에서도 Soft MoE-L/16이 ViT-L/16보다 Zero-shot 성능이 우수함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

Soft MoE는 기존 Sparse MoE의 고질적인 문제였던 이산적 라우팅을 연속적 믹싱으로 대체함으로써 **학습 안정성**과 **하드웨어 효율성**을 동시에 잡았다. 특히 `top-k` 연산을 제거함으로써 GPU/TPU 가속기에서 최적의 처리량을 낼 수 있으며, 전문가 수를 늘려 모델 용량을 확장해도 추론 비용이 증가하지 않는다는 점이 매우 강력한 강점이다.

### 한계 및 미해결 과제

- **Auto-regressive Decoding의 어려움:** 모든 토큰을 섞어 슬롯을 만드는 구조 특성상, 미래 토큰이 과거 토큰에 영향을 주지 않아야 하는 인과적(causal) 마스킹을 적용하기 어렵다. 따라서 LLM과 같은 생성형 모델에 그대로 적용하기에는 한계가 있으며, 이는 향후 연구 과제로 남겨두었다.
- **메모리 소비:** 추론 시간 복잡도는 일정하지만, 전문가 수를 늘리면 모델 파라미터가 증가하므로 메모리 요구량이 크게 늘어난다.

### 비판적 해석

본 논문은 "슬롯"이라는 개념을 통해 가상 토큰을 생성하고 이를 전문가에게 배분함으로써, 사실상 전역적인 정보(global information)를 전문가가 처리하도록 유도한다. 이는 기존 MoE가 토큰의 지역적 특성에 의존해 전문가를 선택하던 방식에서 벗어나, 데이터의 전체적인 분포를 전문가들이 나누어 학습하게 함으로써 더 높은 효율성을 달성한 것으로 해석된다.

## 📌 TL;DR

본 논문은 Sparse MoE의 이산적 라우팅 문제를 해결하기 위해 토큰들을 부드럽게 섞어 전문가에게 전달하는 **Soft MoE**를 제안한다. 이를 통해 토큰 드롭핑과 학습 불안정성을 제거하고, 완전 미분 가능한 구조를 구축하였다. 실험 결과, Soft MoE는 Dense ViT보다 훨씬 적은 추론 비용으로 더 높은 성능을 달성하며, 특히 전문가 수를 대폭 늘려도 추론 속도가 유지되는 뛰어난 확장성을 보여주어 향후 초거대 비전 모델 설계에 중요한 이정표를 제시한다.
