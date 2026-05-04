# ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis

James P. Beno (2025)

## 🧩 Problem to Solve

본 연구는 텍스트의 감정 분석(Sentiment Analysis) 작업에서 양방향 인코더(Bidirectional Encoder) 기반의 모델과 대규모 언어 모델(LLM)을 결합하여 성능을 최적화하고 비용 효율적인 솔루션을 찾는 것을 목표로 한다. 구체적으로는 긍정, 부정, 중립의 3진 분류(three-way sentiment classification) 문제를 다룬다.

감정 분석 분야에서 BERT, RoBERTa, ELECTRA와 같은 양방향 트랜스포머는 파인튜닝(Fine-tuning) 시 매우 뛰어난 성능을 보이며, GPT-4o와 같은 LLM은 제로샷(Zero-shot) 학습 능력이 탁월하다는 특징이 있다. 본 논문은 이 두 가지 서로 다른 강점을 가진 모델들이 협력했을 때, 단일 모델을 사용할 때보다 더 나은 성능을 낼 수 있는지, 그리고 그 과정에서 비용 효율성을 어떻게 확보할 수 있는지를 탐구한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 파인튜닝된 소형 인코더 모델(ELECTRA)의 예측 결과를 LLM(GPT-4o 시리즈)의 프롬프트에 컨텍스트(Context)로 제공함으로써 LLM의 분류 성능을 보조하게 하는 것이다.

주요 기여 사항은 다음과 같다:

- **새로운 협업 구조 제안**: 파인튜닝된 양방향 인코더가 GPT 모델의 감정 분류 작업을 보조하는 새로운 협업 파이프라인을 제안하였다.
- **비용 효율적 성능 향상 입증**: 파인튜닝되지 않은 GPT 모델의 프롬프트를 파인튜닝된 인코더의 예측값으로 보강했을 때, 성능이 유의미하게 향상되며 비용 대비 성능 비율(Cost/Performance ratio)이 가장 낮음을 증명하였다.
- **프롬프트 구성 요소 분석**: 예측 라벨, 확률값, 유사 사례(Few-shot examples) 등 인코더의 어떤 출력 형식이 LLM의 판단에 가장 효과적인지에 대한 가이드라인을 제시하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **MLM 및 ELECTRA**: BERT와 RoBERTa는 양방향 인코딩을 통해 텍스트의 전역적 표현을 학습하지만, 마스킹된 토큰(약 15%)에 대해서만 학습이 이루어지므로 효율성이 떨어진다. ELECTRA는 이를 개선하여 '치환된 토큰 탐지(Replaced Token Detection)' 방식을 통해 모든 토큰에서 학습함으로써 더 적은 계산 자원으로 동등하거나 더 높은 성능을 달성하였다.
- **GPT 모델**: 초기 자기회귀(Autoregressive) 모델들은 감정 분석에서 양방향 모델에 밀렸으나, GPT-4 등 거대 모델의 등장은 제로샷/퓨샷 성능을 비약적으로 향상시켰다. 하지만 감정의 미묘한 뉘앙스나 맥락적 세부 사항을 포착하는 데는 여전히 한계가 있다는 지적이 있다.

### 기존 협업 방식과의 차별점

기존 연구들은 GPT를 사용하여 소수 클래스의 데이터를 증강(Augmentation)한 후 RoBERTa를 학습시키거나, 특정 단계(예: Aspect Extraction)만 GPT가 수행하고 점수 매기기는 RoBERTa가 수행하는 파이프라인 방식을 사용하였다. 또한, RoBERTa의 확신도가 낮을 때만 LLM을 호출하는 결정 트리(Decision Tree) 방식(CMBS 접근법)이 제안된 바 있다. 반면, 본 연구는 **"Show-Me-Your-Answers (SMYA)"** 방식으로, 항상 ELECTRA의 예측값을 GPT에게 전달하고 GPT가 이를 참고하여 최종 결정을 내리게 함으로써 LLM이 인코더의 제안을 수용하거나 거부하도록 설계하였다.

## 🛠️ Methodology

### 전체 시스템 구조

전체 파이프라인은 `Fine-tuned ELECTRA` $\rightarrow$ `Prompt Augmentation` $\rightarrow$ `GPT Model` 순으로 진행된다. 먼저 ELECTRA 모델을 감정 분석 데이터셋으로 파인튜닝하며, 추론 시 이 모델의 출력을 GPT의 입력 프롬프트에 삽입한다.

### 주요 구성 요소 및 역할

1. **ELECTRA 인코더**: 텍스트를 입력받아 감정 라벨을 예측한다.
    - **구조**: Mean pooling layer $\rightarrow$ 2개의 은닉층(차원 1024) $\rightarrow$ 최종 출력층(차원 3).
    - **활성화 함수**: Swish GLU를 사용하여 비선형성을 확보하였다.
2. **GPT 모델**: ELECTRA의 예측값이 포함된 프롬프트를 바탕으로 최종 감정을 분류한다.
    - 대상 모델: GPT-4o, GPT-4o-mini.
3. **DSPy (Declarative Self-improving Python)**: 프롬프트를 체계적으로 프로그래밍하고 최적화하기 위해 사용되었다.

### 프롬프트 보강 시나리오

연구진은 ELECTRA의 출력을 GPT에게 전달하는 여러 형식을 실험하였다:

- **Predicted Label**: ELECTRA가 예측한 클래스 라벨만 제공.
- **Probabilities**: 각 클래스별 예측 확률값(%)을 제공.
- **Top Examples**: ELECTRA Large 모델의 임베딩 공간에서 코사인 유사도가 높은 상위 5개 예시와 라벨을 제공.
- **Balanced Examples**: 클래스별로 상위 2개씩, 총 6개의 예시를 제공하여 특정 클래스로의 편향을 방지.
- **All of the Above**: 위 모든 정보를 통합하여 제공.

### 훈련 및 학습 절차

- **데이터셋**: SST-3와 DynaSent R1, R2를 통합한 Merged Dataset을 사용하였다.
- **ELECTRA 학습**: AdamW 옵티마이저와 CosineAnnealingWarmRestarts 스케줄러를 사용하여 모든 레이어를 파인튜닝하였다.
- **GPT 학습**: OpenAI API를 통해 세 가지 템플릿(Minimal, Prompt, Prompt with Label)으로 파인튜닝을 진행하였다.
  - $FT\text{-}M$: 시스템 역할 외에 프롬프트 없음.
  - $FT$: 기본 DSPy 프롬프트 사용.
  - $FT\text{-}L$: 프롬프트에 ELECTRA의 예측 라벨을 포함.

## 📊 Results

### 실험 설정

- **데이터셋**: SST-3, DynaSent R1, DynaSent R2 (통합 데이터셋).
- **평가 지표**: Macro average F1 score, Cost per F1 point (파인튜닝 비용 / F1 점수).

### 주요 정량적 결과

1. **베이스라인**: 제로샷 GPT-4o (79.97)와 GPT-4o-mini (79.41)가 파인튜닝 전의 ELECTRA 모델들보다 월등히 높은 성능을 보였다.
2. **협업 효과 (Non-FT GPT)**: 파인튜닝되지 않은 GPT-4o-mini에 ELECTRA Base FT의 예측 라벨을 추가했을 때, F1 점수가 $79.41 \rightarrow 82.50$으로 크게 향상되었다.
3. **최고 성능**: GPT-4o FT-M(Minimal fine-tune) 모델이 86.99로 가장 높은 성능을 기록하였으며, GPT-4o-mini FT(86.70)가 근소한 차이로 뒤를 이었다.
4. **비용 효율성**: **ELECTRA Base FT + GPT-4o-mini (Non-FT)** 조합이 F1 포인트당 \$0.12로 가장 낮은 비용 효율을 보였다. 또한, GPT-4o-mini FT는 GPT-4o FT 대비 비용을 76% 절감하면서 거의 유사한 성능을 냈다.

### 주요 발견

- **파인튜닝된 GPT의 역설**: GPT 모델을 파인튜닝하면, ELECTRA의 예측값을 추가하는 것이 오히려 성능을 저하시켰다. (예: GPT-4o-mini FT $86.70 \rightarrow 81.06$으로 하락).
- **추가 정보의 무용성**: 확률값이나 유사 사례(Few-shot)를 추가하는 것은 단순 라벨 제공보다 성능 향상에 거의 기여하지 못했다. (단, DynaSent R2와 같은 어려운 데이터셋에서는 소폭의 향상이 관찰됨).

## 🧠 Insights & Discussion

### 비판적 해석 및 분석

본 연구의 가장 흥미로운 점은 **GPT의 '비판적 사고' 능력과 '파인튜닝' 사이의 트레이드-오프**이다. 기본 GPT 모델은 입력된 정보를 비판적으로 평가하고 종합하는 능력이 있어, ELECTRA의 예측이 틀렸을 때 이를 거부하고 자신의 판단을 내릴 수 있다. 그러나 특정 작업(감정 분류)을 위해 파인튜닝되면, 모델은 텍스트에서 라벨로 직접 매핑하는 최적화 경로를 학습하게 되어, 외부 제안(ELECTRA의 예측)을 필터링하는 비판적 사고 과정을 생략하게 되는 것으로 해석된다.

이를 해결하기 위해 학습 에폭(Epoch)을 1에서 5로 늘렸을 때, GPT가 ELECTRA의 예측이 맞는지 틀린지를 더 잘 구별하기 시작하며 성능이 일부 회복되었다. 하지만 이는 비용을 6~7배 증가시키므로 효율적이지 않다.

### 강점 및 한계

- **강점**: 실제 프로젝트에서 선택할 수 있는 세 가지 명확한 경로(최저 비용: ELECTRA Base FT + GPT-4o-mini / 고성능: GPT-4o-mini FT / 로컬 전용: ELECTRA Large FT)를 제시하였다.
- **한계**: 비용 계산 시 추론 시 발생하는 API 호출 비용은 제외하고 파인튜닝 비용만 고려하였다. 또한, OpenAI의 폐쇄형 모델만 사용하였으며, Llama 3와 같은 오픈소스 LLM과의 결합 가능성은 검토되지 않았다.

## 📌 TL;DR

본 논문은 파인튜닝된 ELECTRA 모델의 예측값을 GPT-4o의 프롬프트에 제공하는 협업 구조를 제안한다. **파인튜닝되지 않은 GPT-4o-mini에 ELECTRA Base FT의 예측값을 추가하는 것이 비용 대비 성능(F1 포인트당 \$0.12)이 가장 우수함**을 밝혀냈다. 반면, GPT를 직접 파인튜닝할 경우 외부 모델의 보조가 오히려 성능을 떨어뜨리는 경향이 있으며, 비용 효율적인 고성능 솔루션으로는 **GPT-4o-mini를 직접 파인튜닝**하는 것이 가장 권장된다. 이 연구는 제한된 리소스를 가진 환경에서 감정 분석 성능을 극대화할 수 있는 실용적인 가이드라인을 제공한다.
