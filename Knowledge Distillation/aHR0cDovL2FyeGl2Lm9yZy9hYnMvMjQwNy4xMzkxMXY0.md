# Continual Distillation Learning: Knowledge Distillation in Prompt-based Continual Learning

Qifan Zhang, Yunhui Guo, Yu Xiang (2025)

## 🧩 Problem to Solve

본 논문은 프롬프트 기반 지속 학습(Prompt-based Continual Learning, CL) 모델의 성능을 향상시키기 위해 지식 증류(Knowledge Distillation, KD)를 결합한 **지속 증류 학습(Continual Distillation Learning, CDL)**이라는 새로운 문제 정의와 해결 방안을 제시한다.

프롬프트 기반 CL에서는 Vision Transformer(ViT)를 백본으로 사용하며, 일반적으로 백본의 크기가 클수록 성능이 향상되는 경향이 있다. 그러나 거대 모델은 추론 시 계산 비용이 매우 높다는 단점이 있다. 따라서 거대 ViT(Teacher)의 지식을 작은 ViT(Student)로 전이하여 추론 효율성을 높이면서도 높은 성능을 유지하는 것이 본 연구의 핵심 목표이다.

특히, 기존의 지식 증류 방식들을 프롬프트 기반 CL 환경에 그대로 적용했을 때 발생하는 성능 저하 문제를 해결하고자 하며, 이를 통해 작은 모델이 거대 모델의 성능에 근접하도록 만드는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **CDL 문제 정의**: 지속 학습 환경에서 거대 모델(Teacher)로부터 작은 모델(Student)로 지식을 전이하는 '지속 증류 학습(Continual Distillation Learning)'이라는 새로운 연구 영역을 정의하고 실험적으로 분석하였다.
2. **Distillation Information Forgetting 발견**: 기존의 KD 방법들이 프롬프트 기반 CL에서 효과적이지 않은 이유가, 각 태스크마다 프롬프트를 새로 선택하는 메커니즘으로 인해 이전 태스크에서 증류된 지식이 소실되는 '증류 정보 망각(Distillation Information Forgetting)' 현상 때문임을 밝혀냈다.
3. **KDP(Knowledge Distillation based on Prompts) 제안**: 증류 정보 망각 문제를 해결하기 위해, 모든 태스크에서 전역적으로 공유되는 **KD 프롬프트(KD Prompts)**를 도입하여 태스크 간 지식 전이가 지속적으로 이루어지도록 설계한 KDP 방법을 제안하였다.

## 📎 Related Works

### 지속 학습 (Continual Learning)

지속 학습의 주된 목적은 새로운 태스크를 학습할 때 이전 지식을 잊어버리는 치명적 망각(Catastrophic Forgetting)을 방지하는 것이다. 기존 연구들은 함수 정규화(Function Regularization), 가중치 정규화(Weight Regularization), 아키텍처 기반 접근법, 그리고 메모리 버퍼를 사용하는 리플레이 기반(Replay-based) 방식 등으로 발전해 왔다.

### 프롬프트 기반 지속 학습 (Prompt-based CL)

최근에는 사전 학습된 ViT 백본을 동결(Freeze)하고, 학습 가능한 프롬프트를 최적화하는 방식이 SOTA 성능을 보이고 있다. 대표적으로 L2P, DualPrompt, CODA-Prompt 등이 있으며, 이들은 백본을 고정함으로써 망각 문제를 완화한다. 본 논문은 이러한 프롬프트 기반 방식에서 모델 크기에 따른 성능 차이가 뚜렷하다는 점에 주목하였다.

### 지식 증류 (Knowledge Distillation)

지식 증류는 일반적으로 Teacher 모델의 로짓(Logit)이나 중간 특징(Feature)을 Student 모델이 학습하도록 하여 성능을 끌어올리는 기법이다. 그러나 기존 KD는 주로 오프라인 데이터셋을 기반으로 하며, 태스크가 순차적으로 들어오는 지속 학습 환경에서의 KD, 특히 프롬프트 기반 모델에서의 KD는 그동안 충분히 연구되지 않았다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (Teacher-Student Model)

본 논문에서 제안하는 CDL 프레임워크는 거대 ViT를 Teacher로, 작은 ViT를 Student로 설정한다. 두 모델 모두 L2P, DualPrompt, CODA-Prompt와 같은 프롬프트 기반 CL 모델을 사용한다. 학습 절차는 다음과 같다.

- 새로운 태스크의 데이터가 들어오면, 먼저 Teacher 모델을 학습시킨다.
- 이후 Teacher 모델을 동결한 상태에서, Teacher가 제공하는 소프트 라벨(Soft labels)이나 내부 토큰(Internal tokens)을 이용하여 Student 모델을 학습시킨다.

### 2. 증류 정보 망각 (Distillation Information Forgetting)

프롬프트 기반 CL은 쿼리-키(Query-Key) 메커니즘을 통해 입력 이미지에 맞는 프롬프트를 선택한다. 만약 Student 모델이 태스크 A에서 특정 프롬프트 집합 $P_A$를 통해 Teacher의 지식을 배웠더라도, 태스크 B에서 전혀 다른 프롬프트 집합 $P_B$를 선택하게 되면, $P_A$에 저장되었던 Teacher의 지식은 활용되지 못하고 버려지게 된다. 백본은 동결되어 있으므로 지식은 오직 프롬프트에만 저장되는데, 이 선택적 메커니즘이 지식의 연속적인 전이를 방해하는 것이다.

### 3. KDP (Knowledge Distillation based on Prompts)

KDP는 위 문제를 해결하기 위해 **KD 프롬프트**라는 새로운 개념을 도입한다.

**핵심 설계:**

- **KD 프롬프트**: 기존 CL 프롬프트 풀(Pool)과 독립적으로, 모든 태스크가 공유하는 전역 프롬프트이다. 쿼리-키 선택 과정 없이 항상 삽입되므로, 태스크가 바뀌어도 Teacher로부터 배운 지식이 유지된다.
- **Prefix-Tuning 적용**: KD 프롬프트 $P^{kd}$는 $\text{Key}(h_K)$와 $\text{Value}(h_V)$ 부분에 각각 삽입되어 MSA(Multi-head Self-Attention) 레이어의 동작을 가이드한다. 수식은 다음과 같다.
$$f_{Pre-T}(P^{cl}, h, P^{kd}) = \text{MSA}(h_Q, [P^{cl}_K; h_K, P^{kd}_K], [P^{cl}_V; h_V, P^{kd}_V])$$
- **KD 토큰 및 분류기**: DeiT의 방식을 차용하여, Student 모델의 첫 번째 레이어에 KD 토큰을 추가하고, 이를 위한 별도의 KD 분류기를 둔다.

**손실 함수 (Loss Function):**
Student 모델의 최종 손실 함수 $L^S$는 다음과 같이 구성된다.
$$L^S = (1-\alpha)L(g^S_\phi(f^S_b(x; P^{kd}_{1:n})), y) + \alpha L^{KD}(k^S_\phi(f^S_b(x; P^{kd}_{1:n})), g^T_\phi(f^T_b(x))) + \lambda L^{pool}$$

- 첫 번째 항: Student 분류기의 일반적인 분류 손실(Cross-Entropy)이다.
- 두 번째 항: Teacher와 Student의 확률 분포 간의 KL 발산(KL Divergence)을 이용한 증류 손실이다.
$$L^{KD} = \tau^2 \sum_i p^T_i \log \left( \frac{p^T_i}{p^S_i} \right)$$
- 세 번째 항: 사용된 프롬프트 기반 CL 방법(L2P, CODA-Prompt 등)의 프롬프트 풀 최적화 손실이다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-100, ImageNet-R (각 10개 태스크로 분할)
- **비교 모델**: 기본 Student 모델 ($\emptyset$), 기존 KD(Logit/Feature 기반), DeiT, 그리고 제안하는 KDP.
- **백본 조합**: ViT-Large $\rightarrow$ ViT-Base, ViT-Base $\rightarrow$ ViT-Small.
- **평가 지표**: 평균 정확도(Avg. Acc), 망각률(Forgetting Rate).

### 주요 결과

1. **정량적 성능**: Table 1 기준, CODA-Prompt를 사용할 때 ViT-Base에서 ViT-Small로 증류한 경우, KDP는 CIFAR-100에서 84.31%의 정확도를 기록하며 기존 KD 방식들보다 우수한 성능을 보였다. 또한 망각률 역시 가장 낮게 나타났다.
2. **일반화 능력**: KDP는 L2P, DualPrompt, CODA-Prompt 등 다양한 프롬프트 기반 CL 모델에 결합했을 때 모두 성능 향상을 가져왔으며, 특히 ViT-Small 기반 모델의 성능을 거대 모델 수준으로 끌어올리는 효과를 보였다.
3. **절제 실험(Ablation Study)**:
    - **KD 프롬프트 vs KD 분류기**: 두 구성 요소가 모두 존재할 때 최적의 성능을 보였으며, KD 프롬프트만 있을 때보다 KD 분류기(KD 토큰)를 함께 사용할 때 더 효과적이었다.
    - **백본 동결 여부**: ViT 백본의 일부를 해제(Unfreeze)하여 학습시켰을 때, 오히려 치명적 망각이 심해져 성능이 급격히 하락하였다 (Avg. Acc $66.98\% \rightarrow 29.98\%$). 이는 백본을 동결하는 프롬프트 기반 CL의 철학이 옳음을 입증한다.
    - **삽입 위치**: 추가 학습 가능 파라미터를 기존 CL 프롬프트 풀에 넣는 것보다, 별도의 KD 프롬프트로 분리하여 삽입하는 것이 훨씬 효과적이었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 프롬프트 기반 CL의 핵심인 '선택적 프롬프트' 메커니즘이 일반적인 학습에는 유리하지만, 지식 증류 관점에서는 '정보의 단절'을 야기한다는 점을 정확히 짚어냈다. 이를 해결하기 위해 전역적으로 공유되는 KD 프롬프트를 도입함으로써, Teacher의 지식을 저장하는 전용 통로를 만들어 준 것이 주효했다. 이는 학습 가능한 파라미터의 양을 단순히 늘린 것이 아니라, **지식의 저장 위치와 접근 방식을 분리**함으로써 망각 문제를 해결한 영리한 설계이다.

### 한계 및 논의사항

- **학습 오버헤드**: Teacher와 Student 모델을 동시에 유지하며 학습시켜야 하므로, 학습 시간과 메모리 소모가 증가한다는 점이 명시적 한계로 언급되었다.
- **가정**: 본 연구는 Teacher 모델이 이미 최적화되었다고 가정하며, 지속적으로 업데이트되는 Teacher로부터 Student가 얼마나 빠르게 적응할 수 있는지에 대한 실시간성 분석은 부족하다.
- **비판적 해석**: 제안된 KDP 방식은 결국 추가적인 프롬프트를 도입하는 것이므로, 추론 시의 파라미터 증가가 미미하더라도 레이어마다 삽입되는 프롬프트의 수가 늘어남에 따라 발생할 수 있는 아주 작은 연산 오버헤드에 대한 분석이 추가되었다면 더 완벽했을 것이다.

## 📌 TL;DR

본 논문은 거대 ViT의 지식을 작은 ViT로 전이하여 프롬프트 기반 지속 학습 모델의 효율성과 성능을 동시에 잡으려는 **지속 증류 학습(CDL)** 문제를 제안한다. 기존 KD 방식이 프롬프트 선택 과정에서 지식을 잊어버리는 **'증류 정보 망각'** 문제를 겪는다는 것을 발견하고, 이를 해결하기 위해 전역적으로 공유되는 **KD 프롬프트(KDP)**와 KD 분류기를 도입하였다. 실험 결과, KDP는 다양한 프롬프트 기반 CL 프레임워크에서 SOTA 성능을 달성하였으며, 특히 작은 모델의 성능을 획기적으로 향상시켜 실제 엣지 디바이스 배포 가능성을 높였다.
