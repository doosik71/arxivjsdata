# SlimPajama-DC: Understanding Data Combinations for LLM Training

Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, Eric Xing (2024)

## 🧩 Problem to Solve

현대 거대 언어 모델(LLM)의 성공은 단순히 데이터의 양이 아니라 데이터의 '다양성'에 달려 있다. 하지만 여러 출처(Web, Wikipedia, GitHub, Books 등)에서 수집된 데이터를 단순히 합쳐서 사용할 경우, 서로 다른 소스 간에 중복된 데이터가 존재하게 되어 학습 효율이 떨어지고 모델이 특정 패턴에 과적합(Overfitting)될 위험이 있다.

본 논문은 특히 다음과 같은 문제들을 해결하고자 한다:

1. **중복 제거 전략의 영향 분석**: 개별 데이터셋 내부에서만 중복을 제거하는 Local Deduplication과 전체 통합 데이터셋에서 중복을 제거하는 Global Deduplication이 모델 성능에 미치는 영향을 분석한다.
2. **데이터 조합 및 비율의 최적화**: 서로 다른 도메인의 데이터 조합과 그 구성 비율이 모델의 일반화 능력과 특정 작업의 전문성 사이의 트레이드오프(Trade-off)에 어떤 영향을 주는지 탐구한다.

결과적으로, 본 연구의 목표는 SlimPajama 데이터셋을 활용하여 LLM 학습을 위한 데이터 조합의 최적 관행(Best Practices)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 데이터 중심의 분석을 통해 LLM 학습 효율을 높이는 방법을 제시한 것이다.

1. **Global Deduplication의 중요성 입증**: 여러 소스의 데이터를 통합할 때, 소스 간의 중복을 제거하는 Global Deduplication이 Local Deduplication보다 모델 성능 향상에 훨씬 유리함을 실험적으로 보여주었다.
2. **데이터 다양성의 영향 분석**: 7가지 서로 다른 데이터 조합 설정(DC-1 $\sim$ DC-7)을 구축하여, 데이터 소스가 다양해질수록 전반적인 벤치마크 성능이 향상됨을 확인하였다.
3. **RRGS(Risk of Random Guessing Score) 지표 제안**: 소형 모델(1.3B)이 MMLU와 같은 어려운 벤치마크에서 단순히 무작위로 추측하여 점수를 얻는 문제를 해결하기 위해, 무작위 추측 위험도를 측정하는 새로운 지표인 RRGS를 도입하였다.
4. **PTWD(Progressive Training on Weight Decay) 전략**: 대규모 배치(Large Batch Size) 학습 시 발생하는 일반화 간극(Generalization Gap)을 줄이기 위해 가중치 감쇠(Weight Decay)를 단계적으로 적용하는 학습 전략을 제안하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 본 연구와의 차별점을 설명한다.

- **RedPajama & SlimPajama**: RedPajama는 LLaMA 학습 데이터셋을 재현하려는 시도로 1.2T 토큰을 보유하고 있다. SlimPajama는 이를 기반으로 엄격한 중복 제거를 거쳐 627B 토큰으로 정제한 데이터셋이다. 본 논문은 이 SlimPajama를 사용하여 데이터 조합의 영향을 분석하는 'SlimPajama-DC' 연구를 수행하였다.
- **The Pile**: 22개의 다양한 고품질 서브셋을 결합하여 일반적인 지식과 하위 작업 일반화 능력을 높이려 한 연구이다. 본 연구는 이와 유사하게 다양성을 추구하지만, 중복 제거 방식과 도메인 가중치에 더 집중한다.
- **Data Selection & Pruning**: Importance Sampling이나 DSIR과 같은 데이터 선택 프레임워크가 존재하며, 최근에는 데이터 프루닝(Pruning)을 통해 전력 법칙(Power Law) 스케일링을 극복하려는 시도가 있다.
- **Generalization Gap in Large Batch Training**: 배치 사이즈가 커지면 학습 속도는 빨라지지만, 모델이 손실 함수(Loss Landscape)의 날카로운 최소값(Sharp Minima)에 빠져 일반화 능력이 떨어지는 현상이 보고되었다. 본 연구는 이를 해결하기 위해 PTWD라는 새로운 접근 방식을 제안한다.

## 🛠️ Methodology

### 1. 데이터 전처리 파이프라인

SlimPajama-DC는 다음과 같은 엄격한 전처리 과정을 거친다.

- **Low-length Document Filtering**: 구두점 및 공백 등을 제거한 후 200자 미만의 문서를 필터링하여 메타데이터만 포함된 저품질 데이터를 제거한다.
- **Global Deduplication**: MinHashLSH 알고리즘을 사용하여 전체 데이터셋에서 중복을 제거한다. Jaccard 유사도 임계값을 $0.8$로 설정하고, 소문자로 변환된 13-gram을 사용하여 문서 시그니처를 생성한다.

### 2. 모델 아키텍처 및 학습 설정

- **Cerebras-GPT Architecture**: GPT-3와 유사한 자기회귀(Autoregressive) 트랜스포머 디코더 구조를 사용하지만, 모든 블록에 Dense Attention을 적용한다.
- **ALiBi (Attention with Linear Biases)**: 위치 임베딩을 더하는 대신, 쿼리-키 어텐션 점수에 거리에 따른 편향(Bias)을 직접 적용하여 입력 길이 외삽(Extrapolation) 능력을 높인다.
- **SwiGLU 활성화 함수**: 다음과 같은 수식의 SwiGLU를 사용하여 비선형성을 구현한다.
  $$\text{SwiGLU}(x,W,V,b,c,\beta) = \text{Swish}_\beta(xW+b) \otimes (xV+c)$$
- **학습 세부 사항**: AdamW 옵티마이저를 사용하며, $\beta_1=0.9, \beta_2=0.95$, Weight Decay $0.1$을 적용한다. 모든 모델은 bf16 혼합 정밀도로 학습되었다.

### 3. 데이터 조합 설정 (Configurations)

총 7가지 조합(DC-1 $\sim$ DC-7)을 통해 데이터 다양성과 비율의 영향을 분석한다.

- **DC-1**: SlimPajama CommonCrawl 전용.
- **DC-6**: 가장 다양한 도메인(CC, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange)을 모두 포함한 조합.
- **DC-7**: 비교군으로서 RefinedWeb 데이터셋을 사용.

### 4. RRGS (Risk of Random Guessing Score)

MMLU 벤치마크에서 1.3B 모델의 점수가 단순 추측인지 확인하기 위해 다음 수식을 제안한다.
$$RRGS = 1 - \frac{1}{N} \sum_{i=1}^N (|s_i - 0.25|)$$
여기서 $s_i$는 MMLU의 각 서브 아이템 점수이며, $0.25$는 4지선다형 문제의 무작위 추측 기댓값이다. 이 값이 낮을수록 모델의 예측이 무작위 추측에서 벗어나 실제 능력을 반영하고 있음을 의미한다.

### 5. PTWD (Progressive Training on Weight Decay)

7B 모델의 대규모 배치 학습 시 과적합을 방지하기 위해 가중치 감쇠를 3단계로 적용한다.

1. **Phase 1**: $WD = 0$으로 설정하여 모델이 빠르게 수렴하게 한다.
2. **Phase 2**: $WD = 0.5$로 높여 과적합을 강하게 억제한다.
3. **Phase 3**: $WD = 0.1$로 조정하여 표준적인 LLM 학습 상태로 마무리한다.

## 📊 Results

### 1. 벤치마크 평가 결과

- **종합 성능**: DC-6(최대 다양성 조합)가 SlimPajama 조합 중 가장 높은 평균 정확도를 기록하였다.
- **비교 분석**: DC-1 $\sim$ DC-6의 결과는 동일한 토큰 수(330B)로 학습된 RedPajama-1.3B보다 전반적으로 우수한 성능을 보였다. 이는 Global Deduplication의 효과와 데이터 다양성의 중요성을 입증한다.
- **특수 도메인 성능**: DC-1(CommonCrawl 전용)은 ARC 및 MMLU에서 높은 점수를 보였으나, TruthfulQA에서는 최하위를 기록하여 다양성 부족으로 인한 한계를 보였다.

### 2. 학습 손실(Training Loss) 분석

- **손실과 성능의 비상관성**: 가장 높은 성능을 보인 DC-7의 학습 손실이 오히려 가장 높게 나타났다. 이는 낮은 Training Loss가 반드시 높은 모델 성능으로 이어지지 않음을 시사한다.
- **코드 데이터의 영향**: 코드 데이터 비중이 높은 DC-3가 가장 낮은 학습 손실을 기록하였다. 이는 코드 데이터가 언어 데이터보다 예측 가능성이 높아 손실 값을 낮추는 경향이 있음을 보여준다.

### 3. LBS-7B 모델 결과

- **학습 효율**: LBS-7B 모델은 LLaMA나 MPT-7B보다 훨씬 높은 처리량(Throughput)과 MFU(Model FLOPs Utilization)를 달성하였으며, 총 GPU 사용 시간을 크게 단축하였다.
- **Instruction Tuning**: Pre-trained LBS-7B 모델에 ShareGPT 데이터로 Instruction Tuning을 적용한 결과, MMLU와 TruthfulQA 성능이 대폭 향상되었다.

## 🧠 Insights & Discussion

- **강점**: 본 논문은 데이터 중복 제거 방식(Global vs Local)과 도메인 믹스가 LLM 성능에 미치는 영향을 정교한 실험으로 입증하였다. 특히 RRGS라는 새로운 지표를 통해 소형 모델 평가의 신뢰성을 높였다.
- **데이터 다양성 vs 전문성**: 특정 데이터셋(예: CommonCrawl 전용)에 집중하면 일부 벤치마크에서 높은 성적을 낼 수 있으나, 전반적인 일반화 능력과 상식(TruthfulQA 등)을 위해서는 다양한 소스의 결합이 필수적이다.
- **학습 전략의 유연성**: PTWD 전략은 대규모 배치 학습에서 발생하는 일반화 간극 문제를 해결할 수 있는 실용적인 방법론을 제시하였다.
- **한계 및 논의**: 학습 손실(Loss)이 성능의 절대적인 지표가 될 수 없다는 점은 데이터 구성 시 Loss curve에만 의존하는 기존 방식에 경종을 울린다. 또한, 본 연구는 1.3B와 7B 모델에 집중되어 있어, 더 거대한 모델에서도 동일한 데이터 비율 최적화가 작동할지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 LLM 학습 시 **Global Deduplication**을 적용하고 **데이터 소스의 다양성을 극대화**하는 것이 모델의 전반적인 성능 향상에 결정적임을 밝혔다. 또한, 소형 모델의 무작위 추측 위험을 측정하는 **RRGS 지표**와 대규모 배치 학습의 과적합을 방지하는 **PTWD 전략**을 제안하였다. 이 연구는 데이터의 양보다 '질'과 '조합'이 중요함을 입증하며, 향후 효율적인 LLM 프리트레이닝을 위한 데이터 중심 가이드를 제공한다.
