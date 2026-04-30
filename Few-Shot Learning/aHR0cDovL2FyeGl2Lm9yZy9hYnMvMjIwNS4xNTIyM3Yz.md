# Prompting ELECTRA: Few-Shot Learning with Discriminative Pre-Trained Models

Mengzhou Xia, Mikel Artetxe, Jingfei Du, Danqi Chen, Ves Stoyanov (2022)

## 🧩 Problem to Solve

본 논문은 Masked Language Model(MLM) 기반의 모델들이 텍스트 인필링(text infilling) 방식으로 구성된 prompt-based few-shot learning에서 거둔 성공을 discriminative pre-trained model, 특히 ELECTRA에 적용하고자 한다. 

기존의 BERT나 RoBERTa와 같은 MLM들은 특정 토큰을 `[MASK]`로 치환하고 이를 예측하는 방식으로 학습되기에, 다운스트림 태스크를 '빈칸 채우기' 형태로 변환하는 prompt-based learning 패러다임에 자연스럽게 부합한다. 반면, ELECTRA와 같은 판별 모델(discriminative model)은 토큰의 생성 여부를 판별하도록 설계되어 있어, 기존의 prompt-based learning 프레임워크를 그대로 적용하기 어려웠다.

본 연구의 목표는 ELECTRA의 판별적 특성을 유지하면서 prompt-based few-shot learning이 가능하도록 방법론을 제안하고, 이것이 기존 MLM 기반의 접근 방식보다 더 효과적인 few-shot learner가 될 수 있음을 입증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 ELECTRA의 사전 학습 목표인 '토큰의 원본 여부 판별(distinguishing if a token is generated or original)'을 prompt-based prediction에 그대로 정렬시키는 것이다.

1.  **Discriminative Prompting**: 타겟 옵션(target options) 중 정답 토큰은 '원본(original)'으로, 오답 토큰은 '생성된 것(replaced/generated)'으로 판별하도록 ELECTRA의 discriminator head를 재사용하여 학습시킨다.
2.  **Multi-token Extension**: MLM이 multi-token 옵션을 처리하기 위해 복잡한 autoregressive decoding이나 별도의 휴리스틱이 필요한 것과 달리, ELECTRA는 판별 헤드를 통해 multi-token span의 표현을 집계(aggregation)함으로써 추가 연산 오버헤드 없이 자연스럽게 확장할 수 있다.
3.  **Few-shot Efficiency**: ELECTRA가 MLM보다 적은 계산 자원으로 학습되었음에도 불구하고, few-shot 설정에서 BERT 및 RoBERTa보다 우수한 성능을 보임을 실험적으로 증명하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 한계점을 언급한다.

-   **Prompting Masked Language Models**: BERT, RoBERTa와 같은 MLM들은 prompt-based fine-tuning을 통해 적은 예시만으로도 높은 성능을 낼 수 있다. 하지만 multi-token 옵션이 포함된 태스크의 경우, MLM은 pre-training 목표에서 벗어난 방식(예: multi-class hinge loss)을 사용하거나 추론 시 배치 처리가 불가능한 pseudo-autoregressive decoding 방식을 사용해야 한다는 효율성 및 정렬 문제가 존재한다.
-   **Discriminative Pre-trained Models (ELECTRA)**: ELECTRA는 generator가 생성한 토큰과 원본 토큰을 구분하는 discriminator를 학습시킨다. 이는 full-shot 설정에서는 매우 강력하지만, few-shot learning의 prompt 패러다임에는 적합하지 않은 구조였다.

본 논문은 ELECTRA의 discriminator head를 그대로 활용함으로써, MLM의 구조적 한계를 극복하고 pre-training과 fine-tuning 간의 목표 정렬을 달성하여 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
본 방법론은 입력 문장 $x$, 레이블 집합 $Y$, 템플릿 $T$ 및 매핑 함수 $M$을 사용하여 ELECTRA를 prompt 기반으로 학습시킨다.

### 1. Single-token Target Words 처리
감성 분석과 같은 단일 토큰 예측 태스크의 경우, 다음과 같은 프롬프트를 구성한다.
$$T(x, y) = x \text{ It was } M(y)$$
여기서 $M(y)$는 레이블 $y$에 대응하는 단어이다. 모델은 정답 단어가 포함된 프롬프트에서는 해당 단어를 '원본'으로, 오답 단어가 포함된 프롬프트에서는 '생성된 토큰'으로 판별하도록 학습된다. 이때 사용되는 손실 함수는 다음과 같다.

$$\mathcal{L} = -\log H(c(M(y))) - \sum_{y' \in Y \setminus \{y\}} \log(1 - H(c(M(y'))))$$

-   $c(\cdot)$: 문맥화된 임베딩(contextualized embedding)
-   $H$: ELECTRA의 discriminator head
-   $M(y)$: 정답 타겟 단어
-   $M(y')$: 오답 타겟 단어

추론 단계에서는 각 옵션에 대해 모델을 통과시켜 가장 높은 '원본' 확률을 가진 단어를 선택한다.

### 2. Multi-token Target Options 처리
COPA와 같은 다지선다형 태스크에서는 옵션 $M(y)$가 여러 토큰으로 구성된다. 본 논문은 autoregressive decoding 없이 다음과 같은 세 가지 집계 방식을 제안한다.

-   **Representation Averaging (rep)**: 옵션 내 모든 토큰의 hidden representation을 평균 낸 후 discriminator head에 입력한다.
    $$H \left( \frac{1}{|y|} \sum_{j} c(y_j) \right)$$
-   **Probability Averaging (prob)**: 각 토큰별로 판별 확률을 계산한 후 그 평균값을 사용한다.
    $$\frac{1}{|y|} \sum_{j} H(c(y_j))$$
-   **[CLS] Token**: 단순히 `[CLS]` 토큰의 확률값을 사용한다.
    $$H(c([CLS]))$$

## 📊 Results

### 실험 설정
-   **모델**: BERT-base/large, RoBERTa-base/large, ELECTRA-base/large.
-   **데이터셋**: 
    -   Single-token: SST-2, SST-5, MR, MNLI, RTE, QNLI, SNLI, AGNews, BoolQ.
    -   Multi-token: COPA, StoryCloze, HellaSwag, PIQA.
-   **지표**: Accuracy.
-   **설정**: Few-shot (레이블당 16개 또는 32개 예시), Zero-shot, Full-shot.

### 주요 결과
1.  **Single-token 태스크**: ELECTRA-base는 prompt-based few-shot 설정에서 BERT보다 평균 10.2점, RoBERTa보다 3.1점 높은 성능을 보였다. 특히 Zero-shot 성능에서도 ELECTRA가 명확한 우위를 점했다.
2.  **Multi-token 태스크**: ELECTRA는 RoBERTa 및 PET(MLM 기반 prompt-tuning) 방식보다 전반적으로 더 높고 안정적인 성능을 기록하였다. 특히 ELECTRA-large 모델은 COPA 등의 태스크에서 매우 강력한 성능을 보였다.
3.  **데이터 양에 따른 변화**: 데이터 수($K$)가 적을수록 ELECTRA와 RoBERTa의 성능 격차가 커졌으며, $K \ge 256$이 되는 시점에서 두 모델의 성능이 수렴하는 경향을 보였다. 이는 ELECTRA의 판별적 사전 학습 방식이 few-shot 환경에 매우 적합함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 분석: Antonym(반의어) 문제
본 논문의 가장 흥미로운 분석은 ELECTRA가 왜 MLM보다 few-shot 성능이 좋은가에 대한 이유이다.

RoBERTa와 같은 MLM의 zero-shot 출력 분포를 분석한 결과, 정답이 'terrible'인 경우 모델이 반의어인 'great'에도 높은 확률을 부여하는 경향이 발견되었다. 이는 MLM이 문맥상 유사한 위치에 올 수 있는 단어들을 유사하게 처리하기 때문이다.

반면 ELECTRA는 사전 학습 과정에서 generator가 원본 단어를 반의어로 교체하여 discriminator에게 제공했을 가능성이 크다. 즉, discriminator는 **'great'와 'terrible'과 같은 반의어를 명확히 구분하도록 훈련**되었으며, 이것이 downstream 태스크에서 정교한 zero-shot/few-shot 예측 능력으로 이어진다고 해석할 수 있다.

### 한계점
-   **계산 효율성**: MLM은 단 한 번의 forward pass로 모든 타겟 단어의 확률을 구할 수 있지만, 제안된 ELECTRA 방식은 타겟 옵션의 개수 $|Y|$만큼 forward pass를 수행해야 하므로 추론 속도가 느리다.
-   **적용 범위**: 옵션의 집합이 제한된 판별 태스크(Discriminative tasks)에만 적용 가능하며, 생성 태스크에는 적용할 수 없다.

## 📌 TL;DR

본 연구는 ELECTRA의 discriminator head를 재사용하여 discriminative pre-trained model을 위한 prompt-based few-shot learning 방법론을 제안하였다. 실험 결과, ELECTRA는 단일 토큰 및 멀티 토큰 예측 태스크 모두에서 BERT 및 RoBERTa를 능가하는 성능을 보였으며, 특히 데이터가 매우 적은 few-shot 설정에서 강력한 효율성을 입증하였다. 이는 ELECTRA의 사전 학습 과정이 반의어와 같은 세밀한 개념 차이를 구분하는 능력을 길러주어, prompt 기반의 태스크 수행에 더 유리하게 작용했기 때문으로 분석된다. 이 연구는 판별 모델이 few-shot learning에서도 매우 유용한 도구가 될 수 있음을 시사한다.