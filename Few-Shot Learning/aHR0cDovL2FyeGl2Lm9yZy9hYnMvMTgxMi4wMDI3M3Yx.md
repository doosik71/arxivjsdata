# Cross-Modulation Networks For Few-Shot Learning

Hugo Prol, Vincent Dumoulin, and Luis Herranz (2018)

## 🧩 Problem to Solve

본 논문은 적은 양의 학습 데이터만으로 새로운 클래스를 인식해야 하는 Few-Shot Learning(FSL) 문제를 다룬다. 기존의 많은 FSL 접근 방식, 특히 Metric Learning 기반의 방법들은 서포트 세트(support set)와 쿼리 예제(query example)를 각각 독립적으로 임베딩 공간으로 투영한 뒤, 마지막 단계에서 두 벡터 간의 유사도를 계산하여 예측을 수행한다.

저자들은 이러한 방식이 정보의 결합을 예측 파이프라인의 매우 후반 단계에서만 수행한다는 점에 주목하였다. 즉, 추상화 수준이 낮은 중간 단계의 특징(intermediate representations)들이 분류에 유용한 정보를 담고 있음에도 불구하고, 이를 활용하지 못하고 최종 임베딩 단계에서만 비교가 이루어지는 것은 지나치게 제한적인 제약일 수 있다는 것이 문제의 핵심이다. 따라서 본 논문의 목표는 특징 추출 과정의 다양한 추상화 수준에서 서포트 예제와 쿼리 예제가 서로 상호작용할 수 있도록 하여, 더욱 강건한 메트릭 공간을 생성하는 아키텍처를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Cross-Modulation** 메커니즘을 통해 특징 추출 과정 전반에 걸쳐 서포트 예제와 쿼리 예제의 정보를 결합하는 것이다. 이를 위해 Feature-wise Linear Modulation (FiLM) 레이어를 도입하여, 한 예제의 정보가 다른 예제의 특징 맵을 변조(modulation)하도록 설계하였다. 결과적으로 네트워크는 고정된 특징 추출기가 아니라, 비교 대상인 상대 예제에 따라 동적으로 특징을 조정하는 적응형 특징 추출기로 동작하게 된다.

## 📎 Related Works

기존의 Few-Shot Learning 연구는 크게 네 가지 방향으로 구분된다:
1. **Optimization-based**: 적은 데이터로도 빠르게 적응할 수 있는 초기화를 학습하는 방식 (예: MAML).
2. **Metric Learning**: 예제들을 비교할 수 있는 임베딩 공간을 학습하는 방식 (예: Matching Networks, Prototypical Networks).
3. **Optimization Algorithm Learning**: 최종 모델을 생성하는 최적화 알고리즘 자체를 학습하는 방식.
4. **Memory/Attention-based**: 외부 메모리나 어텐션 메커니즘을 통해 새로운 문제에 적응하는 방식.

본 연구는 특히 Metric Learning 아키텍처를 확장한다. 기존의 Tadam이나 일부 어텐션 기반 방식에서도 특징 변조(feature-wise modulation)가 사용되었으나, 본 논문의 방식은 다음과 같은 차별점을 가진다:
- **Pairwise Interaction**: 서포트와 쿼리 예제가 쌍(pair)을 이루어 서로를 변조하는 상호작용을 수행한다.
- **Multi-level Interaction**: 고수준 특징(high-level features)만 사용하는 것이 아니라, 네트워크의 여러 추상화 단계(여러 레이어)에서 국소적으로 변조가 일어난다.

## 🛠️ Methodology

### 1. 기본 프레임워크: Matching Networks
본 논문은 Matching Networks를 베이스라인으로 사용한다. 쿼리 예제 $x^*$에 대해 클래스 $c$일 확률은 다음과 같이 정의된다:

$$p(y^* = c | x^*, S) = \sum_{i=1}^{NK} h(x^*, x_i) \mathbb{1}_{y_i=c}$$

여기서 $h(x^*, x_i)$는 쿼리 예제와 서포트 예제 간의 코사인 유사도에 대한 소프트맥스(softmax) 값이다:

$$h(x^*, x_i) = \frac{\exp(\text{cosine}(f(x^*), f(x_i)))}{\sum_{j=1}^{NK} \exp(\text{cosine}(f(x^*), f(x_j)))}$$

여기서 $f(\cdot)$는 4개의 컨볼루션 블록(3x3 conv $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ 2x2 max pooling)으로 구성된 특징 추출기이다.

### 2. Cross-Modulation via FiLM
FiLM(Feature-wise Linear Modulation)은 입력 특징 맵에 아핀 변환(affine transformation)을 적용하여 표현을 변조하는 기법이다. 입력 $x \in \mathbb{R}^{H \times W \times C}$에 대해 FiLM은 다음과 같이 계산된다:

$$\text{FiLM}(x) = \gamma_z \odot x + \beta_z$$

여기서 $\gamma_z, \beta_z \in \mathbb{R}^C$는 조건 입력 $z$로부터 생성된 파라미터이며, $\odot$은 아다마르 곱(Hadamard product)을 의미한다.

본 논문에서는 이를 구체화하여 다음과 같은 수식을 사용한다:

$$\text{FiLM}(x) = (1 + \gamma_0 \gamma_z) \odot x + \beta_0 \beta_z$$

이때 $\gamma_0, \beta_0$는 학습 가능한 파라미터이며, 변조의 희소성(sparsity)을 강제하기 위해 $L^1$ 정규화 페널티를 부여한다.

### 3. FiLM Generator 및 네트워크 구조
변조 파라미터 $\gamma_z, \beta_z$를 생성하는 **FiLM Generator $G(x_1, x_2)$**는 다음과 같이 구성된다:

$$G(x_1, x_2) = \phi([x_1, x_2])W + b$$

- $[x_1, x_2]$: 서포트 예제와 쿼리 예제의 채널 방향 결합(channel-wise concatenation).
- $\phi$: Global Average Pooling(GAP)과 ReLU 함수.
- $W, b$: 학습 가능한 가중치 행렬과 편향.

전체 네트워크는 두 개의 공유된 임베딩 함수(support용, query용)를 가지며, 첫 번째 컨볼루션 블록은 일반적인 방식으로 처리하고, 나머지 **2, 3, 4번째 블록에 Cross-Modulation 메커니즘을 적용**한다. (첫 번째 블록은 연산 오버헤드 대비 성능 향상이 적어 제외되었다.)

## 📊 Results

### 1. 실험 설정
- **데이터셋**: miniImageNet.
- **평가 설정**: 5-way 1-shot 및 5-way 5-shot.
- **최적화**: Adam optimizer, 초기 학습률 0.001 (매 $10^5$ 에피소드마다 절반으로 감소).
- **지표**: 1,000개 에피소드에 대한 평균 정확도 및 95% 신뢰 구간.

### 2. 정량적 결과
Table 1에 따르면, Cross-Modulation Networks는 다음과 같은 성능을 보였다:
- **5-way 1-shot**: $50.94 \pm 0.61\%$의 정확도를 기록하여, 베이스라인인 Matching Networks($49.39 \pm 0.62\%$) 대비 $1.55\%$ 향상되었으며, 다른 SOTA 모델들과 대등한 수준에 도달하였다.
- **5-way 5-shot**: $66.65 \pm 0.67\%$의 정확도를 기록하여 소폭 향상되었다.

### 3. 분석 결과
- **Modulation의 유효성 검증**: 변조 메커니즘에 무작위 노이즈를 섞었을 때, 모든 블록의 변조를 왜곡시키면 정확도가 $7.97\%$ 급락하는 것을 확인하였다(Table 2). 이는 모델이 실제로 변조 메커니즘을 활용하여 학습했음을 의미한다.
- **추상화 수준별 기여도**: 블록별 소거 실험 결과, 모델은 모든 수준에서 변조를 활용하지만 특히 **2번째와 4번째 블록**의 변조에 더 크게 의존하는 경향을 보였다.
- **Self-modulation vs Cross-modulation**: 가중치 행렬 $W$를 분석한 결과, 자기 자신의 특징으로 변조하는 Self-modulation의 영향력이 더 크지만, 상대 예제의 정보를 이용하는 Cross-modulation 역시 유의미하게 작용하고 있음이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 Metric Learning 기반 FSL 모델이 가진 '지나치게 늦은 정보 결합' 문제를 지적하고, 이를 해결하기 위해 특징 추출 과정 전반에 걸쳐 상호작용을 허용하는 구조를 제안하였다. 

**강점**:
- 단순한 임베딩 비교를 넘어, 쿼리와 서포트 간의 관계를 동적으로 반영하는 적응형 특징 추출 방식을 제안하였다.
- Ablation study와 가중치 분석을 통해 제안한 메커니즘이 실제로 어떻게 작동하는지(어느 레이어에서, 어떤 방식으로) 분석하여 설득력을 높였다.

**한계 및 논의사항**:
- **연산 효율성**: 모든 서포트-쿼리 쌍에 대해 Cartesian product를 생성하여 FiLM Generator에 입력해야 하므로, 데이터 수가 늘어날수록 연산 비용이 급격히 증가하는 scaling 문제가 존재한다.
- **데이터 Regime에 따른 효과**: 1-shot에서는 효과가 뚜렷하지만 5-shot에서는 향상 폭이 적다. 이는 데이터가 많아질수록 개별 쌍의 상호작용보다는 클래스의 대표값(prototype)을 찾는 것이 더 중요해지기 때문일 수 있다.
- **범용성**: Matching Networks 외에 Prototypical Networks 등 다른 메트릭 기반 모델에 적용했을 때의 결과는 명시되지 않았다.

## 📌 TL;DR

본 연구는 Few-Shot Learning에서 서포트 세트와 쿼리 예제가 특징 추출 과정의 여러 단계에서 서로의 특징을 변조할 수 있도록 하는 **Cross-Modulation Networks**를 제안한다. FiLM 레이어를 통해 구현된 이 메커니즘은 5-way 1-shot miniImageNet 데이터셋에서 Matching Networks의 성능을 SOTA 수준으로 끌어올렸으며, 분석을 통해 다양한 추상화 단계에서의 상호작용이 분류 성능 향상에 기여함을 입증하였다. 이 연구는 향후 FSL 모델에서 단순한 거리 계산을 넘어선 정교한 상호작용 설계의 중요성을 시사한다.