# Mixture of Diverse Size Experts

Manxi Sun, Wei Liu, Jian Luan, Pengzhi Gao, and Bin Wang (2024)

## 🧩 Problem to Solve

현재의 Sparsely-Activated Mixture-of-Experts (MoE) 아키텍처는 모델의 파라미터를 확장하면서도 계산 비용을 효율적으로 유지할 수 있다는 장점으로 인해 대규모 언어 모델(LLM)에서 널리 사용되고 있다. 그러나 기존 MoE 설계의 치명적인 한계점은 모든 Expert가 동일한 크기(homogeneous size)를 가진다는 점이다.

언어 모델이 다음 토큰을 생성할 때, 모든 토큰이 동일한 난이도를 가지지 않는다. 어떤 토큰은 매우 쉽게 예측 가능하지만, 어떤 토큰은 훨씬 더 복잡한 추론과 많은 양의 지식을 필요로 한다. 모든 Expert의 크기가 동일하면, 토큰의 난이도에 따라 적절한 용량(capacity)의 Expert를 선택할 수 없게 되며, 이는 모델의 성능 향상을 가로막는 병목 현상으로 작용한다.

따라서 본 논문의 목표는 Expert들의 크기를 다양하게 설계하여, 각 토큰의 예측 난이도에 최적화된 크기의 Expert가 할당될 수 있도록 하는 **Mixture of Diverse Size Experts (MoDSE)** 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Expert의 파라미터 크기를 다양화하여 토큰별 예측 난이도에 맞게 계산 자원을 적응적으로 할당**하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Diverse Size Experts 설계**: 기존의 동일 크기 Expert 구조에서 벗어나, 서로 다른 hidden layer 차원을 가진 Expert들을 구성함으로써 토큰의 난이도에 따른 적절한 용량의 Expert를 제공한다.
2. **Expert-Pair Allocation 전략**: Expert들의 크기가 달라짐에 따라 발생할 수 있는 GPU 간의 연산 및 메모리 불균형(load imbalance) 문제를 해결하기 위해, 큰 Expert와 작은 Expert를 한 쌍으로 묶어 동일한 GPU에 배치하는 전략을 제안한다.
3. **토큰 라우팅 분석**: 난이도가 높은 토큰(Cross Entropy loss가 높은 토큰)일수록 더 큰 크기의 Expert로 라우팅되는 경향이 있음을 정량적으로 분석하여, 제안 방법론의 직관을 입증하였다.

## 📎 Related Works

### 기존 MoE 및 FFN 설계

- **DeepSeekMoE**: Fine-Grained Expert Segmentation을 통해 Expert를 더 작게 쪼개어 전문성을 높이고, Shared Expert Isolation을 통해 공통 지식을 처리하는 방식을 제안하였다.
- **HyperMoE**: HyperNetworks를 도입하여 Expert 간의 지식 전이를 촉진하였다.
- **DeLighT & Apple OpenELM**: 레이어별(layer-wise) 또는 블록별로 scaling을 다르게 하여 파라미터를 효율적으로 할당하였다.

### 차별점

기존 연구들이 주로 Expert의 개수를 늘리거나, 레이어 간의 크기를 조정하거나, Expert를 세분화하는 데 집중했다면, MoDSE는 **단일 MoE 레이어 내부에서 Expert 개별의 크기를 다르게 설정**함으로써 토큰 난이도에 따른 적응적 용량 할당이라는 관점에서 접근하였다.

## 🛠️ Methodology

### 1. 전체 구조 및 Diverse Size Experts

MoDSE는 Transformer의 Dense FFN 레이어를 MoE 레이어로 대체하는 구조를 가진다. 기존 MoE는 모든 Expert $E_i(\cdot)$가 동일한 hidden dimension $h$를 가지지만, MoDSE는 각 Expert $\hat{E}_i(\cdot)$가 서로 다른 hidden dimension $\hat{h}_i$를 가진다.

전체 파라미터 규모를 유지하기 위해, Expert들을 쌍(pair)으로 묶어 관리한다. $N$개의 Expert가 있을 때, $n = N/2$개의 쌍 $(i_1^k, i_2^k)$을 생성하며, 각 쌍의 평균 크기가 기존의 $h$와 같도록 설계한다.

$$ \hat{h}_{i_1^k} + \hat{h}_{i_2^k} = 2 \times h, \quad k \in \{1, \dots, n\} $$

이렇게 함으로써 모델의 전체 파라미터 수는 기존 vanilla MoE와 동일하게 유지하면서, 일부 Expert는 $h$보다 크고 일부는 작게 구성할 수 있다.

### 2. 추론 및 라우팅 절차

입력 $x$에 대해 gating network $G(x)$가 top-k개의 Expert를 선택하며, 최종 출력 $\hat{y}$는 다음과 같이 계산된다.

$$ \hat{y} = \sum_{i=1}^{N} \hat{G}_i(x) \hat{E}_i(x) $$

여기서 gating score $H(x)_i$는 학습 가능한 가중치 $W_g$와 노이즈 성분을 포함하여 계산되며, Softmax와 KeepTopK 함수를 통해 최종 게이트 값 $\hat{G}_i(x)$가 결정된다.

### 3. Load Balance 전략

Expert의 크기가 다르면 큰 Expert가 배치된 GPU에 연산 부하가 집중되는 문제가 발생한다. 이를 해결하기 위해 두 가지 전략을 사용한다.

- **Expert-Pair Allocation**: 앞서 정의한 Expert 쌍(큰 것 하나, 작은 것 하나)을 동일한 GPU 노드에 배치한다. 이렇게 하면 GPU당 할당되는 총 파라미터 양이 일정하게 유지되어 하드웨어 수준의 밸런스를 맞출 수 있다.
- **Auxiliary Load Balance Loss**: 라우팅 붕괴(routing collapse)를 막고 모든 Expert가 균등하게 선택되도록 Switch Transformers의 보조 손실 함수 $L_a$를 사용한다.

$$ L_a = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i $$

여기서 $f_i$는 실제로 Expert $i$에 할당된 토큰의 비율이고, $P_i$는 라우터가 Expert $i$에 할당한 확률의 평균값이다.

## 📊 Results

### 실험 설정

- **모델 규모**: $300\text{M} \times 8$ 및 $700\text{M} \times 8$ (여기서 8은 Expert의 수)
- **데이터셋**: 영어 및 중국어가 포함된 100B 토큰의 대규모 데이터셋으로 사전 학습.
- **평가 지표**: MMLU, GSM8K, MATH, AGIEval 등 다양한 downstream task에서 few-shot in-context learning으로 평가.

### 주요 결과

1. **정량적 성능 향상**: $700\text{M} \times 8$ 모델 기준, MoDSE는 모든 벤치마크에서 baseline(동일 크기 MoE)보다 우수한 성능을 보였다. 특히 MATH(0.8 $\rightarrow$ 2.6)나 SIQA(42.9 $\rightarrow$ 60.9)에서 큰 폭의 향상이 관찰되었다.
2. **학습 수렴 속도**: MoDSE는 baseline보다 더 빠르게 수렴하며, 전체 학습 과정 동안 더 낮은 Cross Entropy(CE) loss 값을 유지하였다.
3. **추론 효율성**: Expert-pair allocation 전략 덕분에 baseline과 MoDSE 간의 추론 시간 차이는 거의 없었으며, 매우 유사한 수준의 효율성을 보였다.

### 토큰 라우팅 분석

- **난이도별 성능**: CE loss가 높은 '어려운 토큰'일수록 MoDSE 적용 시 loss 감소 폭이 더 크게 나타났다. 이는 MoDSE가 어려운 토큰을 처리하는 능력을 유의미하게 향상시켰음을 의미한다.
- **Expert 선택 경향**: 분석 결과, 예측 난이도가 높은 토큰들은 통계적으로 더 큰 크기의 Expert를 선택하는 경향이 뚜렷했다. (예: 큰 Expert 그룹으로 라우팅된 토큰 수가 작은 Expert 그룹보다 2배 이상 많음)

## 🧠 Insights & Discussion

### 강점 및 통찰

본 연구는 **"모든 토큰에 동일한 계산 자원을 투입하는 것은 비효율적"**이라는 직관을 아키텍처 수준에서 성공적으로 구현하였다. 특히 파라미터 총량을 고정한 상태에서 크기만 다양화했음에도 성능이 향상되었다는 점은, 모델의 성능이 단순히 파라미터의 총합이 아니라 **자원의 적절한 배치(allocation)**에 달려 있음을 시사한다.

### 한계 및 비판적 해석

- **확장성(Scalability) 검증 부족**: 실험이 $700\text{M}$ 규모의 소형 모델에서만 진행되었다. 초거대 모델(Billion 단위 이상)에서도 동일한 효과가 나타날지, 혹은 Expert 크기의 다양성이 너무 커질 경우 오히려 학습 불안정성이 생기지 않을지에 대한 검증이 필요하다.
- **데이터 및 토크나이저 비공개**: 자체 학습 데이터와 토크나이저를 사용하였으므로, 다른 공개 데이터셋에서도 동일한 경향성이 나타나는지 재현 가능성에 대한 의문이 남는다.
- **Expert 크기 결정 방식**: 본 논문에서는 Expert 크기를 임의의 비율(예: 4.5, 4.0, 3.0 등)로 설정하였다. 이러한 크기 분포를 하이퍼파라미터로 설정하는 대신, 학습 과정에서 최적의 크기를 찾아가는 학습 가능한 구조로 발전시킨다면 더 강력한 모델이 될 수 있을 것이다.

## 📌 TL;DR

MoDSE는 모든 Expert의 크기가 동일한 기존 MoE의 한계를 극복하기 위해, **Expert들의 크기를 다양하게 구성하여 토큰의 난이도에 맞게 최적의 용량을 할당**하는 새로운 구조를 제안한다. GPU 간 부하 불균형은 **Expert-pair allocation** 전략으로 해결하였으며, 실험을 통해 **어려운 토큰일수록 큰 Expert가 처리**하며 전체적인 모델 성능과 수렴 속도가 향상됨을 입증하였다. 이 연구는 향후 효율적인 LLM 설계를 위해 파라미터의 단순 확장이 아닌 '적응적 할당'의 중요성을 제시하였다는 점에서 가치가 크다.
