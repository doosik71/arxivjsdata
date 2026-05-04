# Multi-Head Mixture-of-Experts

Xun Wu, Shaohan Huang, Wenhui Wang, Furu Wei (2024)

## 🧩 Problem to Solve

본 논문은 Sparse Mixture of Experts (SMoE) 구조가 가진 두 가지 핵심적인 한계점을 해결하고자 한다.

첫째는 **낮은 Expert Activation** 문제이다. SMoE는 이론적으로 모델의 용량을 확장하면서 계산 비용을 일정하게 유지할 수 있지만, 실제 학습 및 추론 과정에서는 소수의 Expert만이 활성화되고 대다수는 사용되지 않는 'Dead Experts' 현상이 발생한다. 이는 모델의 표현 능력을 충분히 활용하지 못하게 하며, 특히 복잡한 태스크에서 더 많은 수의 Expert를 학습시키는 것을 방해한다.

둘째는 **개별 토큰 내의 미세한 분석 능력 부족(Lack of fine-grained analytical capabilities)**이다. 현재의 토큰화 방식은 하나의 토큰이 단일한 의미만을 갖는다고 가정한다. 그러나 언어 데이터에서는 다의어(Polysemous words)나 서로 다른 언어에서 형태는 같으나 의미가 다른 False Cognates가 존재하며, 시각 데이터에서는 하나의 패치(Patch) 내에 매우 다양하고 세밀한 정보가 포함되어 있다. 기존 SMoE는 하나의 토큰을 하나의 대표적인 Expert 세트에 할당하므로, 이러한 복합적인 의미 체계를 세밀하게 포착하는 데 한계가 있다.

따라서 본 논문의 목표는 Expert의 활성도를 높이면서, 토큰 수준에서 더 세밀한 의미 분석이 가능한 새로운 MoE 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문은 **Multi-Head Mixture-of-Experts (MH-MoE)**라는 구조를 제안한다. 핵심 아이디어는 Multi-head attention의 개념을 MoE의 Expert 라우팅 과정에 도입하는 것이다.

입력 토큰을 그대로 Expert에게 보내는 대신, **Multi-head mechanism을 통해 하나의 토큰을 여러 개의 sub-token으로 분할**한다. 분할된 sub-token들은 각각 서로 다른 Expert들에게 병렬적으로 할당되어 처리되며, 처리 후에는 다시 원래의 토큰 형태로 통합된다.

이러한 설계를 통해 MH-MoE는 다음과 같은 효과를 얻는다:
1. **Expert 활성화 증대**: 하나의 토큰이 여러 sub-token으로 나뉘어 더 많은 Expert에게 분배되므로, 더 많은 Expert가 학습 및 추론에 참여하게 된다.
2. **세밀한 의미 포착**: 각 head가 서로 다른 표현 공간(Representation space)을 담당하는 Expert로 sub-token을 보냄으로써, 하나의 토큰이 가진 다중 의미나 시각적 세부 사항을 동시에 분석할 수 있다.
3. **유연한 통합**: 기존 SMoE 프레임워크와 독립적으로 구현되어 있어, 기존 모델에 쉽게 통합하여 성능을 높일 수 있다.

## 📎 Related Works

본 논문은 **Sparse Mixture of Experts (SMoE)** 연구를 기반으로 한다. SMoE는 모든 파라미터를 사용하는 Dense 모델과 달리, 라우터(Router)를 통해 입력 토큰에 적합한 일부 Expert만을 선택적으로 활성화하여 효율성을 극대화하는 방식이다. GShard나 Mixtral 8x7B와 같은 모델들이 대표적인 예시이다.

기존의 SMoE는 주로 **Top-k routing** 방식을 사용하는데, 이는 $N$개의 Expert 중 가장 높은 점수를 받은 $k$개만을 선택하는 방식이다. 하지만 $k$값을 높이면 성능은 향상될 수 있으나 계산 효율성이 급격히 떨어진다는 한계가 있다. MH-MoE는 $k$값을 직접적으로 높이는 대신, 입력 데이터의 차원을 분할하여 처리하는 방식을 통해 계산 복잡도를 유지하면서도 실질적인 Expert 활용도를 높였다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인
MH-MoE 레이어는 다음과 같은 순서로 동작한다:
**입력 토큰 $\rightarrow$ Multi-head Layer $\rightarrow$ Token-Splitting-Merging (TSM) $\rightarrow$ Router $\rightarrow$ Experts $\rightarrow$ Merge Layer $\rightarrow$ 출력 토큰**

### 상세 구성 요소 및 절차

**1. Multi-head Layer 및 Token Splitting**
입력 시퀀스 $X \in \mathbb{R}^{l \times d}$ (여기서 $l$은 토큰 수, $d$는 차원)가 들어오면, 먼저 학습 가능한 행렬 $W_{head} \in \mathbb{R}^{d \times d}$를 통해 투영된다:
$$\hat{X} = X \cdot W_{head}^\top$$
이후 $\hat{X}$의 각 토큰은 $h$개의 sub-token으로 분할되어 $\ddot{X} \in \mathbb{R}^{(l \times h) \times \frac{d}{h}}$ 형태가 된다. 즉, 하나의 토큰 $x_i$가 $h$개의 작은 벡터 $x_{ij} \in \mathbb{R}^{\frac{d}{h}}$로 쪼개지는 과정이다.

**2. Routing 및 Expert 처리**
분할된 각 sub-token $x_{ij}$는 독립적으로 라우팅된다. $p$번째 Expert로 갈 확률(Gating value)은 다음과 같이 계산된다:
$$g(f_{FFN}^p) = \frac{\exp(x_{ij} \cdot e_p)}{\sum_{\xi=0}^{N} \exp(x_{ij} \cdot e_\xi)}$$
여기서 $e_p \in \mathbb{R}^{\frac{d}{h}}$는 각 Expert의 학습 가능한 임베딩이다. Top-k 라우팅을 통해 선택된 Expert 집합 $\Phi$에 의해 다음과 같이 처리된다:
$$o_{ij} = x_{ij} + \sum_{p \in \Phi} g(f_{FFN}^p) \cdot f_{FFN}^p(x_{ij})$$

**3. Token Merging 및 Merge Layer**
처리된 sub-token $o_{ij}$들은 다시 원래의 토큰 순서대로 재배열되어 $\bar{X} \in \mathbb{R}^{l \times d}$ 형태로 병합된다. 마지막으로, 여러 Expert로부터 수집된 다양한 정보를 효과적으로 통합하기 위해 Merge Layer($W_{merge} \in \mathbb{R}^{d \times d}$)를 통과시킨다:
$$\check{X} = \bar{X} \cdot W_{merge}^\top$$

### 훈련 목표 및 손실 함수
MH-MoE는 태스크 손실 함수($L_{task}$)와 Expert 간의 부하 균형을 맞추기 위한 **Load Balancing Loss**($L_{balance}$)를 함께 최소화한다.
$$L = L_{task} + \alpha L_{balance}$$
여기서 $L_{balance}$는 특정 Expert에만 토큰이 쏠리는 현상을 방지하며, 모든 Expert가 고르게 학습되도록 유도한다.

### 계산 복잡도 조절
본 논문은 MH-MoE가 추가적인 연산 비용을 초래하지 않도록 하이퍼파라미터 $\beta$를 도입하여 Expert 내부의 hidden dimension을 조정함으로써, 전체 파라미터 수와 계산량을 기존 SMoE와 유사하거나 오히려 더 낮게 유지하였다.

## 📊 Results

### 실험 설정
- **데이터셋 및 태스크**: 
    1. 영어 중심 언어 모델링 (RedPajama 데이터셋, GPT 태스크)
    2. 다국어 언어 모델링 (Multilingual Wikipedia, GPT 태스크)
    3. 마스크드 멀티모달 모델링 (이미지 및 텍스트 데이터, Masked modeling 태스크)
- **비교 대상**: Dense 모델, X-MoE (기존 SMoE 구현체)
- **평가 지표**: Perplexity (PPL), Zero-shot Accuracy, VQA score, BLEU@4 등

### 주요 결과
**1. 정량적 성능 (Upstream & Downstream)**
- **Perplexity**: 모든 태스크에서 MH-MoE가 가장 낮은 PPL을 기록하였다. 특히 Expert 수를 8개에서 32개로 늘렸을 때, MH-MoE의 성능 향상 폭이 X-MoE보다 훨씬 컸다.
- **언어 이해 (English/Multilingual)**: LLM Evaluation Harness 및 XNLI 벤치마크에서 X-MoE 대비 평균 0.6~1.5 포인트의 성능 향상을 보였다.
- **멀티모달 (Vision-Language)**: VQAv2, NLVR2, COCO Captioning 등에서 Dense 및 X-MoE를 큰 차이로 앞섰다. 특히 VQAv2 test-dev에서는 X-MoE 대비 1.69 포인트 상승하였다.

**2. Expert 활성화 분석**
- X-MoE의 Expert 활성화 비율이 8.33%에 불과했던 반면, MH-MoE는 **90.71%**라는 매우 높은 활성화 비율을 기록하여 'Dead Experts' 문제를 사실상 해결했음을 입증하였다.

**3. 확장성 (Scalability)**
- Expert 수를 8개에서 256개까지 확장했을 때, X-MoE는 64개 부근에서 성능 포화 상태에 이르렀으나, MH-MoE는 256개까지 지속적으로 성능이 향상되는 모습을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석
MH-MoE의 성능 향상은 단순히 파라미터를 늘린 결과가 아니라, **토큰의 세분화된 처리**에서 기인한다.
- **언어적 관점**: 분석 결과, 다의어(Polysemous)나 False Cognates 토큰의 경우 일반 토큰보다 더 다양한 Expert로 sub-token들을 분배하는 경향이 확인되었다. 이는 모델이 하나의 토큰이 가진 여러 의미 층위를 서로 다른 Expert를 통해 병렬적으로 분석하고 있음을 시사한다.
- **시각적 관점**: 이미지의 텍스처가 복잡하고 의미가 풍부한 영역(High-frequency regions)일수록 더 다양한 Expert가 할당되는 것이 확인되었다.

### 한계 및 논의사항
- **Head 수($h$)의 최적점**: 실험 결과 $h$가 6까지는 성능이 상승하지만, 그 이상(8, 12 등)으로 늘리면 오히려 성능이 하락하는 경향을 보였다. 이는 토큰을 너무 잘게 쪼갤 경우 원래의 의미적 정보가 훼손될 수 있음을 의미하며, 적절한 분할 수준을 결정하는 것이 중요함을 시사한다.
- **단일 MLP 층의 충분성**: MLP 층을 여러 겹 쌓는 것보다 단일 층만으로도 토큰의 분할과 통합이 충분히 이루어짐을 확인하였다.

### 비판적 해석
본 논문은 SMoE의 고질적인 문제인 Expert 활용도 저하를 매우 단순하고 직관적인 '분할 후 통합' 방식으로 해결하였다. 특히 계산 복잡도를 늘리지 않으면서도 실질적인 연산 효율을 높였다는 점이 인상적이다. 다만, $h$ 값에 따른 성능 변동이 존재하므로, 다양한 도메인이나 더 큰 규모의 모델에서도 동일한 최적 $h$ 값이 유지될지는 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 SMoE의 낮은 Expert 활성화율과 토큰의 다중 의미 포착 능력을 개선하기 위해, 입력 토큰을 여러 sub-token으로 나누어 서로 다른 Expert에 분배하고 다시 합치는 **Multi-Head Mixture-of-Experts (MH-MoE)**를 제안한다. 이를 통해 **Expert 활성도를 8%에서 90% 수준으로 획기적으로 높였으며**, 언어 및 시각 데이터의 미세한 의미를 더 잘 포착하여 성능을 향상시켰다. 특히 모델 규모 확장(Scaling-up) 시 기존 SMoE보다 더 높은 성능 상한선을 보여주어, 향후 초거대 MoE 모델의 효율적인 학습 및 설계에 중요한 기여를 할 가능성이 높다.