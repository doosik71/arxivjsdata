# DELTA-LORA: FINE-TUNING HIGH-RANK PARAMETERS WITH THE DELTA OF LOW-RANK MATRICES

Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong & Lei Zhang (2023)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)을 특정 하위 태스크에 적응시키기 위한 효율적인 파라미터 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 방법론을 다룬다. 

현대적인 LLM은 수십억 개의 파라미터를 가지고 있어 모든 파라미터를 업데이트하는 전수 미세 조정(Full Fine-tuning)을 수행할 경우, 엄청난 양의 GPU 메모리가 요구된다. 특히 AdamW와 같은 최적화 알고리즘은 그래디언트와 모멘텀을 저장하기 위해 모델 파라미터 크기의 2~3배에 달하는 추가 메모리를 사용하므로, 많은 연구 기관과 기업이 이를 수행하기 어렵다.

이를 해결하기 위해 LoRA(Low-Rank Adaptation)와 같은 저차원 행렬 분해 방식의 PEFT가 제안되었으나, 이러한 방법들은 전체 파라미터의 극히 일부(보통 1% 미만)만을 학습시킨다. 이로 인해 전수 미세 조정과 비교했을 때 성능 격차가 발생하는 한계가 있으며, 이는 저차원 행렬의 증분 업데이트만으로는 하위 태스크의 복잡한 표현(Representation)을 충분히 학습하기에 부족하기 때문이다. 따라서 본 논문의 목표는 LoRA의 낮은 메모리 비용을 유지하면서도, 전수 미세 조정에 근접하는 학습 능력을 갖춘 새로운 PEFT 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **사전 학습된 가중치 행렬 $W$와 저차원 행렬 $A, B$를 동시에 업데이트**하는 것이다. 구체적인 설계 아이디어는 다음과 같다.

1.  **Delta-based Update**: 사전 학습된 가중치 $W$를 업데이트하기 위해 $W$의 그래디언트를 직접 계산하여 저장하는 대신, 두 저차원 행렬의 곱 $AB$의 변화량(Delta), 즉 $\Delta AB = A^{(t+1)}B^{(t+1)} - A^{(t)}B^{(t)}$를 이용하여 $W$를 갱신한다.
2.  **Memory Efficiency**: $W$의 그래디언트를 계산하거나 최적화 도구(Optimizer)의 모멘텀을 저장할 필요가 없으므로, 메모리 요구량은 기존 LoRA와 거의 동일하게 유지된다.
3.  **Dropout Removal**: 저차원 경로에 존재하던 Dropout 레이어를 제거하여, $W$의 그래디언트와 $AB$의 그래디언트가 수학적으로 동일하게 유지되도록 설계하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 PEFT 연구들을 언급하며 차별점을 제시한다.

*   **Adapter & Prompt-Tuning/Prefix-Tuning**: 모델 레이어 사이에 가벼운 파라미터를 추가하거나 입력 프롬프트를 최적화하는 방식이다. 하지만 이러한 방법들은 추론 단계에서 추가적인 연산 오버헤드를 발생시킨다는 단점이 있다.
*   **LoRA (Low-Rank Adaptation)**: $\Delta W = AB$ 형태로 가중치 업데이트를 모델링하여 추론 오버헤드가 없고 메모리 효율적이다. 그러나 앞서 언급했듯 전수 미세 조정과의 성능 격차가 존재한다.
*   **DyLoRA & AdaLoRA**: LoRA의 랭크($r$)를 동적으로 조절하거나 중요도에 따라 파라미터 예산을 할당하는 방식이다. 하지만 여전히 $W$ 자체는 고정(Frozen)된 상태로 유지된다.
*   **Q-LoRA**: 4-bit 양자화를 통해 메모리 사용량을 극단적으로 줄인 방법이다.

**차별점**: 기존의 모든 LoRA 계열 방법론은 $W$를 고정하고 $A, B$만을 학습시키지만, Delta-LoRA는 $W$를 직접 업데이트 프로세스에 포함시켜 모델의 표현 학습 능력을 극대화한다.

## 🛠️ Methodology

### 1. 전체 구조 및 원리
Delta-LoRA의 기본 구조는 LoRA와 유사하게 사전 학습된 가중치 $W \in \mathbb{R}^{c \times d}$와 두 개의 저차원 행렬 $A \in \mathbb{R}^{c \times r}$, $B \in \mathbb{R}^{r \times d}$를 사용한다. 입력 $x$에 대한 순전파 과정은 다음과 같다.

$$h = Wx + \frac{\alpha}{r}ABx$$

여기서 $\alpha$는 스케일링 계수, $r$은 랭크이다. Delta-LoRA는 여기서 $A, B$뿐만 아니라 $W$도 함께 업데이트한다.

### 2. $W$의 업데이트 메커니즘 (The Delta of Low-Rank Matrices)
본 논문은 Dropout이 없는 경우, 손실 함수 $L$에 대한 $W$의 그래디언트 $g_W$와 $AB$의 그래디언트 $g_{AB}$가 동일하다는 수학적 성질에 주목한다.

$$g_W = \frac{\partial L}{\partial W} = \frac{\partial L}{\partial h_{i+1}} \cdot h_i^\top$$
$$g_{AB} = \frac{\partial L}{\partial AB} = \frac{\partial L}{\partial h_{i+1}} \cdot h_i^\top$$
$$\implies g_W = g_{AB}$$

그러나 $g_{AB}$를 직접 계산하는 것은 $W$의 그래디언트를 계산하는 것만큼 비용이 많이 든다. 이에 대한 대안으로, 최적화 과정에서 $A$와 $B$가 업데이트되면서 발생하는 **곱의 차이(Delta)**를 $W$의 업데이트 방향으로 사용한다.

구체적인 업데이트 식은 다음과 같다.
1.  $A$와 $B$는 기존의 AdamW 최적화 알고리즘을 통해 업데이트된다.
2.  $W$는 다음과 같은 규칙으로 업데이트된다:
$$W^{(t+1)} = W^{(t)} + \lambda \cdot \frac{\alpha}{r} \cdot (A^{(t+1)}B^{(t+1)} - A^{(t)}B^{(t)})$$

여기서 $\lambda$는 $W$의 업데이트 비율을 조절하는 하이퍼파라미터이며, $K$번의 반복 학습(iteration) 이후부터 이 업데이트를 시작한다.

### 3. Dropout 제거의 필요성
기존 LoRA는 $A, B$ 앞에 Dropout 레이어를 배치한다. 하지만 Delta-LoRA의 분석에 따르면, Dropout이 존재할 경우 $W$의 그래디언트와 $AB$의 그래디언트 사이에 불일치가 발생한다.

$$g_{AB} = \frac{\partial L}{\partial h_{i+1}} \cdot \text{Drop}(h_i)^\top \neq g_W$$

따라서 $W$의 업데이트 방향을 정확하게 유도하기 위해 Delta-LoRA에서는 저차원 경로의 Dropout을 제거하였다.

## 📊 Results

### 1. 실험 설정
*   **모델**: RoBERTa-base, GPT2-Medium, BART-Large
*   **데이터셋**: 
    *   NLU: GLUE benchmark (8개 태스크)
    *   NLG (Data-to-Text): E2E Challenge, WebNLG Challenge 2017
    *   Summarization: XSum dataset
*   **비교 대상**: Full Fine-Tuning, LoRA, DyLoRA, AdaLoRA 및 일부 파라미터만 튜닝한 Fine-Tuning 변형 모델들.

### 2. 주요 결과
*   **NLG 성능**: E2E Challenge 데이터셋에서 Delta-LoRA는 BLEU, ROUGE-L 등 모든 지표에서 LoRA 및 AdaLoRA보다 뛰어난 성능을 보였으며, 일부 전수 미세 조정(Full FT) 결과와 대등하거나 상회하는 결과를 얻었다.
*   **NLU 성능**: GLUE 벤치마크의 8개 모든 태스크에서 기존 PEFT 방법론들을 앞질렀다. 특히 데이터 양이 적은 SST-2, CoLA, RTE 태스크에서 눈에 띄는 향상을 보였는데, 이는 더 많은 파라미터를 최적화함으로써 강건한 표현을 학습했기 때문으로 분석된다.
*   **요약 성능**: XSum 데이터셋에서도 Rouge-1, 2, L, Sum 모든 지표에서 LoRA 및 AdaLoRA보다 높은 성능을 기록하였다.

### 3. 분석 및 통찰
*   **파라미터 민감도**: 업데이트 비율 $\lambda$가 2일 때 최적의 성능을 보였으며, 시작 단계 $K$가 500 이상일 때 안정적인 성능 향상이 관찰되었다.
*   **코사인 유사도 분석**: 전수 미세 조정된 가중치와 원래 가중치 사이의 코사인 유사도를 측정한 결과, LoRA는 유사도가 매우 높은 반면(변화가 적음), Delta-LoRA는 유사도가 낮게 나타났다. 이는 Delta-LoRA가 $W$를 실질적으로 더 많이 수정하여 더 나은 표현력을 획득했음을 시사한다.

## 🧠 Insights & Discussion

### 강점
Delta-LoRA는 LoRA의 가장 큰 약점인 '제한된 학습 용량'을 해결하면서도, PEFT의 핵심 가치인 '메모리 효율성'을 그대로 유지했다는 점이 매우 고무적이다. $W$의 그래디언트를 직접 저장하지 않고 저차원 행렬의 변화량을 대리물(surrogate)로 사용한 점이 영리한 설계이다.

### 한계 및 비판적 해석
1.  **수학적 근사치**: 부록 A.1에서 언급되었듯이, $\Delta AB$가 실제 $W$의 최적 업데이트 방향($\Delta W$)과 완전히 일치하지는 않는다. 최적화 도구(AdamW)의 모멘텀과 Weight Decay가 개입하기 때문이다. 즉, 이 방법은 이론적으로 완벽한 그래디언트 하강법이라기보다 유효한 방향으로 유도하는 근사 방식에 가깝다.
2.  **하이퍼파라미터 의존성**: $\lambda$ (업데이트 비율)와 $K$ (시작 시점)라는 새로운 하이퍼파라미터가 도입되었다. 태스크마다 최적의 값이 다를 수 있어, 이에 대한 자동화된 설정 방법이 제시되지 않은 점은 아쉽다.
3.  **Dropout 제거의 트레이드오프**: 과적합(Overfitting) 방지를 위해 사용되던 Dropout을 제거했으므로, 데이터셋이 매우 작거나 모델이 너무 클 경우 과적합 위험이 증가할 수 있다.

## 📌 TL;DR

Delta-LoRA는 저차원 행렬 $A, B$의 업데이트 차이($\Delta AB$)를 이용하여 사전 학습된 가중치 $W$를 함께 업데이트하는 새로운 PEFT 방법론이다. 이를 통해 LoRA 수준의 낮은 메모리 비용을 유지하면서도, 전수 미세 조정에 가까운 높은 표현 학습 능력을 확보하여 다양한 NLP 태스크(NLU, NLG, 요약)에서 SOTA 성능을 달성하였다. 이 연구는 향후 초거대 모델의 효율적인 적응 학습 분야에서 중요한 기반이 될 가능성이 높다.