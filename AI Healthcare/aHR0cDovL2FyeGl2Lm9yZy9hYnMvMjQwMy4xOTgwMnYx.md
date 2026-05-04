# Developing Healthcare Language Model Embedding Spaces

Niall Taylor, Dan Schofield, Andrey Kormilitzin, Dan W Joyce, Alejo Nevado-Holgado (2024)

## 🧩 Problem to Solve

본 논문은 일반적인 말뭉치로 학습된 사전 학습 언어 모델(Pre-trained Language Models, PLMs)이 의료 분야와 같은 특정 도메인(Out-of-domain) 데이터셋에서 성능이 급격히 저하되는 문제를 해결하고자 한다. 특히, 의료 텍스트는 크게 '생물 의학 텍스트(Biomedical text)'와 '임상 텍스트(Clinical text)'로 나뉘는데, 임상 텍스트는 전문 용어의 약어 사용이 빈번하고 문법과 구문이 비정형적인 특성이 있어 일반 모델이 이해하기 매우 어렵다.

또한, 영국 NHS(National Health Service)의 임상 데이터는 미국 중심의 오픈소스 의료 데이터셋과도 언어적 패턴 및 관용구에서 상당한 차이가 존재한다. 따라서 본 연구의 목표는 제한된 컴퓨팅 자원(단일 GPU) 환경에서 소형 LLM(BERT-like 모델)을 의료 도메인에 효과적으로 적응시켜, 특히 문서 수준(Document-level)의 임베딩 공간을 최적화함으로써 다운스트림 분류 작업의 성능을 높이는 방법을 탐색하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 소형 LLM을 의료 도메인에 정렬시키기 위한 세 가지 사전 학습 전략을 비교 분석하고, 특히 문서 수준의 표현력을 높이기 위한 Contrastive Learning의 효용성을 입증한 것이다.

1. **사전 학습 방법론 비교**: 전통적인 Masked Language Modeling(MLM), 비지도 학습 기반의 DeCLUTR, 그리고 의료 데이터의 메타데이터(Note Category)를 활용한 새로운 사전 학습 목적 함수를 제안하고 비교하였다.
2. **의료 도메인 특화 임베딩 공간 구축**: 단순한 토큰 레벨의 학습을 넘어, 문서 전체를 대표하는 임베딩 공간을 구축하여 적은 양의 레이블 데이터만으로도 높은 분류 성능을 낼 수 있는 효율적인 방법론을 제시하였다.
3. **자원 효율적 적응 기법**: 고가의 GPU 클러스터 없이 단일 GPU만으로도 의료 도메인 특화 모델을 구축할 수 있음을 보여줌으로써, 개인정보 보호가 중요한 의료 현장에서의 로컬 배포 가능성을 시사하였다.

## 📎 Related Works

기존의 BERT, RoBERTa와 같은 모델들은 MLM 방식을 통해 토큰 레벨의 문맥 표현에는 뛰어나지만, 문서 전체의 의미를 담는 임베딩 생성에는 한계가 있다. 이를 해결하기 위해 Sentence-BERT, SimCSE, DeCLUTR 등의 Contrastive Learning 기반 접근법이 제안되었으며, 의료 분야에서는 BioBERT, ClinicalBERT 등이 도메인 적응을 시도하였다.

그러나 기존 의료 LLM들은 주로 미국 데이터셋에 치중되어 있어 영국 NHS와 같은 지역적 특성이 강한 데이터에는 성능이 떨어지는 한계가 있다. 본 연구는 이러한 지역적 도메인 차이를 인지하고, 단순히 모델 크기를 키우는 대신 사전 학습 목적 함수(Objective function)를 변경하여 임베딩 공간의 정렬(Alignment)과 균일성(Uniformity)을 개선하는 방향으로 차별화를 두었다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 연구는 RoBERTa-base를 기본 모델로 사용하며, 이를 의료 데이터셋으로 추가 사전 학습(Continued Pre-training)시킨 후, 문서 분류를 위한 다운스트림 작업에 적용하는 구조를 가진다. 문서 수준의 표현을 얻기 위해 마지막 트랜스포머 층의 모든 토큰 임베딩을 평균 내는 Mean Pooling 함수 $g$를 사용한다.

### 2. 세 가지 사전 학습 전략

#### (1) Masked Language Modeling (MLM)

가장 표준적인 방법으로, 입력 토큰의 일부를 `[MASK]` 토큰으로 대체하고 이를 예측하도록 학습한다. 손실 함수 $L_{mlm}$은 다음과 같다.
$$L_{mlm}(X,Y) = -\sum_{n=1}^{N} W_n \left( \sum_{i=1}^{|V|} Y_{ni} \ln(f_{lm}(X)_n) \right)$$
여기서 $W_n$은 마스크된 토큰인 경우에만 1이 되어, 마스크된 위치의 예측 정확도만을 계산한다.

#### (2) DeCLUTR (Contrastive Learning)

비지도 학습 기반의 Contrastive Learning으로, 동일 문서 내의 서로 다른 두 스팬(Span)을 추출하여 이를 '양성 쌍(Positive pair)'으로, 다른 문서의 스팬을 '음성 쌍(Negative pair)'으로 정의한다. InfoNCE 손실 함수를 사용하여 양성 쌍의 코사인 유사도는 높이고 음성 쌍의 유사도는 낮춘다.
$$P(s_i, s_j; \tau) = \frac{\exp(s_i \cdot s_j / \tau)}{\sum_{k \neq i,j} \exp(s_i \cdot z_k / \tau)}$$
$$L_{InfoNCE} = -\frac{1}{2} \left[ \frac{1}{P} \sum_{i,j=0}^{P} \log P(s_i, s_j; \tau) + \frac{1}{P} \sum_{i,j=0}^{P} \log P(s_j, s_i; \tau) \right]$$
여기서 $\tau$는 학습 가능한 온도(Temperature) 파라미터이다.

#### (3) Note Category 기반 사전 학습

의료 데이터에 포함된 메타데이터(작성자의 직역, 부서 등)인 'Note Category'를 예측하는 분류 작업을 사전 학습에 도입하였다. 이는 BERT의 Next Sentence Prediction(NSP)을 대체하는 개념으로, 문서 임베딩 $e$를 분류 헤드 $f_{head}$에 통과시켜 정답 클래스를 예측하는 Cross-Entropy 손실 함수 $L_{note}$를 사용한다.
$$L_{note} = -\sum_{c=1}^{N} y_c \log(p_c)$$

### 3. 최종 학습 목적 함수

본 연구는 MLM과 위에서 언급한 Contrastive/Category 손실을 결합하여 함께 최적화한다.
$$L = L_{MLM} + w(L_{contrastive})$$
여기서 $w$는 Note Category 손실이 전체 학습을 지배하지 않도록 조절하는 가중치(예: 0.1)이다.

### 4. 다운스트림 적응 및 분석

학습된 모델 상단에 2개 층의 MLP 분류기를 추가하여 파인튜닝을 진행한다. 특히 모델의 가중치를 고정한 채 분류기만 학습시키는 'Frozen' 설정과 전체를 학습시키는 'Fine-tuned' 설정을 나누어 임베딩 자체의 유용성을 평가하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MIMIC-III(미국 ICU), OHFT(영국 정신건강), PSIR(영국 환자 안전 사고 보고서)
- **작업**: ICD-9 Triage(M-Tri), Referral Team(O-Tri), Incident Category(P-Cat), Severity(P-Sev) 등 문서 분류
- **지표**: $F_1$ macro score

### 2. 주요 결과

- **도메인 적응의 중요성**: 모든 데이터셋에서 도메인 사전 학습을 거친 모델이 일반 RoBERTa-base 모델보다 우수한 성능을 보였다.
- **Contrastive Learning의 우위**: 모델 가중치를 고정한 **Frozen 설정에서 DeCLUTR 기반 모델이 가장 높은 성능**을 기록하였다. 이는 DeCLUTR이 다운스트림 작업에 즉시 활용 가능한 고품질의 문서 임베딩 공간을 생성했음을 의미한다.
- **Few-shot 성능**: 레이블 데이터가 매우 적은 상황(클래스당 16~200개 샘플)에서 Contrastive Learning 모델이 MLM 모델보다 훨씬 빠르게 성능이 향상되는 양상을 보였다(그림 3 참조).
- **임베딩 공간 분석**:
  - **Cosine Similarity**: Contrastive 모델들이 클래스 내/외부 간의 유사도 분리가 가장 뚜렷하게 나타났다.
  - **Uniformity & Alignment**: DeCLUTR 모델은 임베딩 공간의 다양성(Low Uniformity)을 유지하면서도 클래스 내 응집력(High Alignment)을 확보하는 특성을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 발견

본 연구는 소형 모델에서도 사전 학습 목적 함수를 적절히 설정하면 거대 모델 못지않은 도메인 특화 성능을 낼 수 있음을 보여주었다. 특히, 레이블이 없는 상태에서 임베딩 공간을 최적화하는 DeCLUTR 방식이 의료 문서 분류라는 실제 작업에서 매우 효율적이라는 점을 입증하였다. 또한, 단일 GPU로도 몇 시간 내에 도메인 적응이 가능하다는 점에서 실용성이 매우 높다.

### 2. 한계 및 비판적 해석

- **DeCLUTR 샘플링 문제**: DeCLUTR은 최소 문서 길이 조건을 만족하는 샘플만 사용하므로, 짧은 문서가 많은 데이터셋에서는 학습 데이터의 일부가 누락되는 편향이 발생한다.
- **메타데이터의 제한적 효과**: Note Category 기반 학습은 임베딩 공간의 군집화(Clustering) 지표는 개선시켰으나, 실제 분류 성능(F1 score) 향상으로 이어지지는 않았다. 이는 메타데이터가 제공하는 정보가 다운스트림 분류 작업의 결정 경계와 완전히 일치하지 않기 때문으로 해석된다.
- **토큰 레벨 성능**: 문서 수준의 성능은 향상되었으나, NER과 같은 토큰 레벨 작업에서는 MLM 모델과 큰 차이가 없었다. 이는 Contrastive Loss가 문서 수준의 표현력은 높이지만 개별 토큰의 세밀한 의미 표현을 크게 변화시키지는 않음을 시사한다.

## 📌 TL;DR

본 논문은 의료 도메인, 특히 영국 NHS 데이터에 특화된 소형 LLM 임베딩 공간을 구축하는 방법을 연구하였다. **결론적으로, 비지도 Contrastive Learning(DeCLUTR) 방식이 일반적인 MLM보다 문서 수준의 임베딩 품질을 크게 향상시키며, 특히 레이블 데이터가 부족한 Few-shot 상황과 모델을 고정해서 사용하는 Frozen 설정에서 압도적인 효율성을 보였다.** 이 연구는 개인정보 보호와 자원 제한이 엄격한 의료 환경에서 효율적으로 도메인 특화 모델을 배포할 수 있는 가이드라인을 제공한다.
