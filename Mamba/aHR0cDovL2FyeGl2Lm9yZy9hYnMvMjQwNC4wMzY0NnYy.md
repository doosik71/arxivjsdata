# Locating and Editing Factual Associations in Mamba

Arnab Sen Sharma, David Atkinson, and David Bau (2024)

## 🧩 Problem to Solve

본 연구는 State Space Model(SSM) 기반의 언어 모델인 Mamba에서 사실적 지식의 회상(factual recall)이 어떻게 이루어지는지 그 내부 메커니즘을 분석하는 것을 목표로 한다. 기존의 autoregressive transformer 언어 모델(LM) 연구들에 따르면, 사실적 지식의 회상이 모델 내의 특정 모듈과 특정 토큰 위치에 국한되어 나타나는 '국소성(locality)' 현상이 발견되었다.

Mamba는 Transformer와는 완전히 다른 아키텍처(Attention과 MLP 대신 Convolution과 SSM 사용)를 가지고 있음에도 불구하고, Transformer에서 발견된 이러한 국소적 특성이 Mamba에서도 유사하게 나타나는지 확인하고자 한다. 이는 최신 신경망 아키텍처가 진화함에 따라, 기존 Transformer를 위해 개발된 해석 가능성(interpretability) 분석 도구들이 다른 아키텍처에도 일반화되어 적용될 수 있는지를 검증하는 방법론적 도전 과제를 포함하고 있다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba의 사실적 회상 메커니즘이 Transformer와 매우 유사한 국소적 패턴을 보인다는 것을 입증한 점이다. 주요 기여 사항은 다음과 같다.

- **사실 회상의 국소성 확인**: Activation patching을 통해 Mamba에서도 특정 레이어와 토큰 위치(특히 subject의 마지막 토큰과 prompt의 마지막 토큰)가 사실 회상에 결정적인 역할을 함을 밝혔다.
- **모델 편집 가능성 증명**: ROME(Rank One Model Editing) 기법을 Mamba에 적용하여, 특정 가중치 행렬을 수정함으로써 새로운 사실을 성공적으로 삽입할 수 있음을 보였다.
- **정보 흐름 분석**: Transformer의 attention knock-out 기법을 Mamba에 맞게 변형하여 적용함으로써, 사실 정보가 모델 내부에서 어떻게 흐르는지 분석하였다.
- **아키텍처 간 공통점 발견**: Mamba와 Transformer가 구조적으로 매우 다름에도 불구하고, 사실 회상이라는 작업에 있어서는 매우 유사한 계산 패턴을 공유한다는 통찰을 제공하였다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구들을 바탕으로 한다.

- **Transformer의 사실 회상 분석**: Meng et al. (2022a,b)의 ROME 연구와 Geva et al. (2023)의 attention 분석, Hernandez et al. (2023)의 관계 임베딩 선형성 연구 등이 기반이 되었다. 이들 연구는 Transformer의 MLP 모듈이 사실적 지식을 저장하고, attention 모듈이 이를 추출한다는 점을 시사했다.
- **Mamba 및 SSM**: Gu & Dao (2023)가 제안한 Mamba는 선형 시간 복잡도로 시퀀스를 모델링하며 Transformer에 경쟁하는 성능을 보인다. 그러나 Mamba 내부에서 지식이 어떻게 표현되고 회상되는지에 대한 해석적 연구는 지금까지 부족한 상태였다.
- **차별점**: 기존 연구들이 주로 Transformer 구조에 집중한 반면, 본 연구는 recurrent 성격의 SSM 아키텍처에서도 동일한 해석 도구들이 작동하는지를 직접적으로 비교 분석했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### Mamba 아키텍처 및 수식

Mamba는 `MambaBlock`을 층층이 쌓은 구조이며, 각 블록은 다음과 같이 계산된다.
잔차 연결(residual connection)을 포함한 최종 상태 $h^{(\ell)}_i$는 다음과 같다.
$$h^{(\ell)}_i = h^{(\ell-1)}_i + o^{(\ell)}_i$$
여기서 $o^{(\ell)}_i$는 $\ell$번째 블록의 출력이며, 다음과 같이 계산된다.
$$o^{(\ell)}_i = W^{(\ell)}_o (s^{(\ell)}_i \otimes g^{(\ell)}_i)$$
$\otimes$는 요소별 곱셈(Hadamard product)을 의미한다. 여기서 $s^{(\ell)}_i$는 Conv1D와 selective-SSM 연산을 거쳐 과거 토큰의 정보를 가져오는 경로이며, $g^{(\ell)}_i$는 정보 흐름을 조절하는 gating 메커니즘이다.
$$g^{(\ell)}_i = \text{SiLU}(W^{(\ell)}_g h^{(\ell-1)}_i)$$

### 분석 방법론

1. **Activation Patching (Causal Tracing)**:
   - **Clean Run ($G$)**: 정답을 맞히는 프롬프트(예: "Michael Jordan professionally played")를 입력하여 모든 내부 상태를 캐싱한다.
   - **Corrupted Run ($G^*$)**: 주어를 다른 주어(예: "Pelé")로 바꾸어 오답이 나오게 한다.
   - **Patched Run ($G^*[\leftarrow h^{(\ell)}_i]$)**: $G^*$를 실행하되, 특정 위치의 상태 $h^{(\ell)}_i$만 $G$에서 캐싱한 값으로 교체한다.
   - **Indirect Effect (IE)**: 상태 복구가 정답 확률을 얼마나 높였는지 측정한다.
     $$IE_{h^{(\ell)}_i} = \frac{p^*[\leftarrow h^{(\ell)}_i](o) - p^*(o)}{p(o) - p^*(o)}$$

2. **ROME (Rank One Model Editing)**:
   - 특정 선형 변환을 연상 메모리로 간주하고, 랭크-1 업데이트를 통해 특정 키 $k^*$가 새로운 값 $v^*$로 매핑되도록 가중치를 수정한다.
   - Mamba에서는 $W_a, W_g, W_o$ 행렬을 대상으로 편집을 시도하였다.

3. **Linearity of Relation Embedding (LRE)**:
   - 사실 관계 추출 과정을 1차 테일러 급수로 근사하여, 입력-출력 관계가 선형적인지 확인한다.
     $$F(s, r) \approx \beta J \rho s + b$$

4. **Attention Knock-out (Mamba adaptation)**:
   - Mamba는 attention이 없으므로, 특정 토큰 $k$에서 미래 토큰으로 흐르는 정보를 차단하기 위해 $a^{(\ell)}_k$ 값을 평균값으로 대체하는 mean-ablation 기법을 사용하였다.

## 📊 Results

### 1. 사실 회상의 국소성 (Localization)

- **결과**: Mamba에서도 Transformer와 유사하게 **'early site'**(주어의 마지막 토큰 위치의 초기-중간 레이어)와 **'late site'**(프롬프트 마지막 토큰 위치의 후기 레이어)에서 높은 IE가 관찰되었다.
- **구성 요소별 역할**: $W_o$ 행렬이 두 위치 모두에서 가장 강력한 인과적 효과를 보였으며, $s^{(\ell)}_i$ (SSM 출력)는 주로 late site에서만 높은 IE를 보였다. 이는 Transformer의 attention 모듈 동작과 유사하다.

### 2. 모델 편집 (ROME)

- **결과**: $W_a, W_g, W_o$ 모두에서 사실 편집이 가능했다.
- **최적의 위치**: 실험적으로 $W_o$를 수정했을 때 가장 높은 점수($S$)와 일반화 성능($PS$)을 보였다. 특히 $W_o$는 early-mid 레이어에서 사실 회상을 매개하는 핵심 역할을 수행함이 확인되었다.

### 3. 관계 임베딩의 선형성 (LRE)

- **결과**: 26개의 사실 관계 중 약 10개 정도만이 50% 이상의 faithfulness(충실도)를 보였다.
- **특징**: 정답의 범위(range)가 넓은 관계일수록 선형 근사가 어려웠으며, 이는 Pythia(Transformer) 모델에서도 동일하게 나타난 현상이다.

### 4. 정보 흐름 분석 (Knock-out)

- **관계 정보**: 초기-중간 레이어에서 관계(relation)를 정의하는 토큰들의 정보 흐름을 차단하면 정답 확률이 최대 50%까지 하락했다.
- **주어 정보**: 레이어 43-48 구간에서 주어 정보의 흐름을 차단했을 때 확률이 크게 떨어졌으며, 이는 해당 구간의 $s_i$ 상태들이 사실 회상에 결정적임을 시사한다.

## 🧠 Insights & Discussion

본 연구는 Mamba와 Transformer라는 서로 다른 두 아키텍처가 사실적 지식을 처리하는 방식에서 놀라운 유사성을 공유하고 있음을 보여준다.

**강점 및 해석**:

- **태스크 유도적 국소성**: 저자들은 이러한 유사성이 특정 아키텍처의 특성이 아니라, **'autoregressive language modeling'이라는 작업 자체에서 기인**한 것이라고 추론한다. 텍스트를 순차적으로 처리해야 하는 제약 조건 하에서, 주어의 끝부분은 주어를 완전히 인식하고 관련 지식을 인출하기 위한 자연스러운 '정보 병목(bottleneck)' 지점이 되며, 이에 따라 모델들이 공통적으로 국소적 계산 패턴을 형성하게 된다는 것이다.

**한계 및 비판적 해석**:

- **정밀도 문제**: Mamba의 Conv1D와 non-linearity(SiLU) 때문에 Transformer의 attention edge를 끊는 것만큼 정밀하게 특정 토큰 간의 정보 흐름을 차단하는 것이 어려웠다. 이로 인해 mean-ablation이라는 다소 거친 방법을 사용해야 했으며, 이는 분석의 정밀도를 일부 떨어뜨릴 수 있다.
- **모델 규모**: Mamba-2.8b라는 단일 모델에 대해 실험이 진행되었으므로, 모델 크기에 따른 일반화 가능성에 대해서는 추가 연구가 필요하다.

## 📌 TL;DR

이 논문은 Mamba 모델의 사실 회상 메커니즘을 분석하여, **Mamba가 Transformer와 유사하게 특정 토큰 위치와 레이어에 지식 회상 과정이 국소화되어 있음**을 밝혔다. 특히 ROME를 통해 $W_o$ 가중치를 수정함으로써 사실 편집이 가능함을 보였으며, 이는 사실 회상의 국소적 패턴이 아키텍처보다는 autoregressive 학습 태스크 자체의 특성일 가능성이 높음을 시사한다. 향후 새로운 언어 모델 아키텍처를 설계하더라도 유사한 국소적 지식 처리 패턴이 나타날 가능성이 크며, 기존의 Transformer 해석 도구들이 범용적으로 사용될 수 있음을 입증한 연구이다.
