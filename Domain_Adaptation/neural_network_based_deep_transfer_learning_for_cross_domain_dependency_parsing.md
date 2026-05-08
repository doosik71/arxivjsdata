# Neural Network based Deep Transfer Learning for Cross-domain Dependency Parsing

Zhentao Xia, Likai Wang, Weiguang Qu, Junsheng Zhou, Yanhui Gu (2019)

## 🧩 Problem to Solve

본 논문은 서로 다른 도메인 간의 의존 구문 분석(Cross-domain Dependency Parsing) 문제를 해결하고자 한다. 의존 구문 분석은 의미역 결정(Semantic Role Labeling), 관계 추출(Relation Extraction), 기계 번역(Machine Translation) 등 다양한 자연어 처리 시스템의 핵심 구성 요소이다.

일반적인 구문 분석기는 학습 데이터와 테스트 데이터가 동일한 도메인에서 추출된 In-domain 설정에 집중되어 있으나, 실제 환경에서는 웹 데이터의 급증으로 인해 도메인이 다른 데이터에 대한 분석 성능이 저하되는 문제가 발생한다. 본 연구의 목표는 소스 도메인(Source Domain)에서 학습된 의존 구문 분석기를 세 가지 서로 다른 타겟 도메인(상품 리뷰, 상품 블로그, 웹 소설)에 효과적으로 적응(Adaptation)시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 Stack-Pointer Network(STACKPTR) 구조에 **Self-attention 메커니즘**과 **신경망 기반의 딥 전이 학습(Deep Transfer Learning)**을 결합하는 것이다.

1. **Contextual Representation 강화**: Self-attention을 통해 단어의 의미를 문맥적으로 더 잘 포착하여 표현 벡터의 질을 높였다.
2. **도메인 적응을 위한 전이 학습**: 타겟 도메인의 학습 데이터가 부족한 문제를 해결하기 위해, 소스 도메인에서 사전 학습된 네트워크의 일부 구조와 파라미터를 타겟 도메인 모델의 일부로 재사용하고 미세 조정(Fine-tuning)하는 전략을 사용하였다.

## 📎 Related Works

논문은 의존 구문 분석의 두 가지 주류 접근 방식인 그래프 기반(Graph-based) 알고리즘과 전이 기반(Transition-based) 알고리즘을 언급한다. 본 연구의 기반이 되는 STACKPTR은 포인터 네트워크(Pointer Network)를 백본으로 하며, 트리 구조의 헤드 단어 순서를 유지하기 위해 내부 스택을 사용하는 구조이다.

기존 연구들이 주로 In-domain 설정에 집중했다는 한계를 지적하며, 본 논문은 이를 도메인 적응 문제로 모델링하여 전이 학습을 통해 해결하려 한다. 특히 소스 도메인의 방대한 지식을 타겟 도메인으로 전이함으로써, 타겟 도메인의 적은 데이터량으로도 효율적인 학습이 가능하도록 설계하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

시스템은 크게 네 가지 구성 요소로 이루어져 있다: **Token Representation Layer $\rightarrow$ Self-attention Layer $\rightarrow$ Stack-Pointer Network $\rightarrow$ Domain Adaptation**.

### 2. 주요 구성 요소 및 역할

#### (1) Token Representation

입력 문장 $S=\{w_1, w_2, \dots, w_n\}$에 대해 다음과 같은 세 가지 임베딩을 결합(Concatenate)하여 입력 벡터 $X=\{x_1, x_2, \dots, x_n\}$를 생성한다.

- **Word-level Embedding**: 사전 학습된 Glove 모델을 사용하여 단어를 벡터로 변환한다.
- **Character-level Embedding**: CNN을 사용하여 단어의 문자 시퀀스를 인코딩하여 형태적 정보를 포착한다.
- **POS Embedding**: 품사(Part-of-Speech) 정보를 추가하여 문맥 정보를 보강한다.

#### (2) Stack-Pointer Networks (STACKPTR)

포인터 네트워크를 백본으로 하며, 내부 스택을 통해 트리의 상단-하단(Top-down) 구조를 유지한다.

- **인코딩**: BiLSTM을 사용하여 각 단어를 은닉 상태 $e_i$로 인코딩한다.
- **디코딩**: Top-down, Depth-first 전이 시스템을 구현한다. 시간 단계 $t$에서 스택 최상단 단어의 인코딩 상태 $e_i$를 받아 디코더 은닉 상태 $d_t$를 생성하고, Biaffine Attention 메커니즘을 통해 어텐션 벡터 $a_t$를 계산한다.
- **수식**:
  $$s_{it} = \text{score}(d_t, s_i)$$
  $$a_t = \text{softmax}(s_t)$$
- **동작**: $a_t$에서 가장 높은 점수를 가진 위치 $p$를 반환하여 의존 아크 $w_i \to w_p$를 생성하고, $w_p$를 스택에 푸시한다. 자신을 가리키면 모든 자식 노드를 찾은 것으로 간주하여 팝(pop)한다.

#### (3) Self-Attention Layer

단어 벡터가 문맥적 의미를 포착할 수 있도록 Multi-head Attention을 적용한다.

- **연산 과정**: Query($Q$), Key($K$), Value($V$) 행렬을 사용하여 다음과 같이 계산한다.
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_w}}\right)V$$
  $$\text{MultiHead}(Q, K, V) = W_O [h_{\text{head}1}; \dots; h_{\text{head}r}]$$
  여기서 $r$은 헤드의 수이며, $W_O$ 등은 학습 가능한 선형 변환 파라미터이다.

#### (4) Domain Adaptation (Deep Transfer Learning)

타겟 도메인의 데이터 부족 문제를 해결하기 위해 다음과 같은 전이 학습 전략을 취한다.

- **파라미터 재사용**: 소스 도메인에서 학습된 Encoder와 Decoder의 파라미터, 그리고 Self-attention 파라미터를 유지(Retain)한다.
- **부분적 재학습**: 소스 도메인에서 학습된 Biaffine Attention 메커니즘의 파라미터는 버리고, 타겟 도메인에 맞게 새로운 Biaffine Attention Score를 학습시킨다.
- **미세 조정**: 이후 전체 네트워크의 파라미터를 타겟 도메인 데이터로 Fine-tune 한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 소스 도메인은 Balanced Corpus(BC)를 사용하였고, 타겟 도메인은 상품 리뷰(PC), 상품 블로그(PB), 웹 소설(ZX) 세 가지를 사용하였다.
- **평가 지표**: Labeled Attachment Score (LAS, 헤드와 레이블이 모두 정확하게 예측된 단어의 비율)를 사용하며, 세 타겟 도메인의 평균 LAS로 최종 성능을 측정한다.
- **하이퍼파라미터**: Word Embedding 차원 300, RNN 은닉 유닛 256, Attention Head 수 4, Adam Optimizer 사용.

### 2. 실험 결과 (Development Set 기준)

| Model | PC LAS | PB LAS | ZX LAS | Average LAS |
| :--- | :---: | :---: | :---: | :---: |
| STACKPTR (Baseline) | 61.1 | 74.8 | 74.6 | 70.2 |
| STACKPTR + Multi-head Attention | 60.9 | 75.5 | 75.1 | 70.5 |
| STACKPTR + Transfer Learning | 61.9 | 75.7 | 75.4 | 71 |
| **STACKPTR + Attn + Transfer** | **62.6** | **76.9** | **76.3** | **71.9** |

- **결과 분석**: Self-attention과 전이 학습을 모두 적용했을 때 가장 높은 성능(평균 71.9 LAS)을 보였다. 특히 전이 학습 단독 적용 시보다 두 기법을 함께 사용했을 때 성능 향상 폭이 더 컸다. 이는 Multi-head attention이 도메인 간 교차 분석에 필요한 다양한 특징을 포착하고, 전이 학습이 적은 데이터 환경에서 학습 효율을 높였기 때문이다.

## 🧠 Insights & Discussion

본 논문은 구조적 제약이 있는 STACKPTR에 문맥 표현력을 높이는 Self-attention과 도메인 간 지식을 전이하는 Transfer Learning을 결합하여 유의미한 성능 향상을 이끌어냈다. 특히 중국어의 경우 도메인이 다르더라도 기본적인 의미론적 이해(Semantic understanding)는 유사하다는 가정을 통해 전이 학습의 정당성을 확보하였다.

하지만, 실험 결과에서 상품 리뷰(PC) 도메인의 LAS가 다른 도메인(PB, ZX)에 비해 현저히 낮게 나타났다. 이는 상품 리뷰 데이터가 구어체 성격이 강하거나 문장 구조가 비정형적일 가능성이 높음을 시사하며, 이러한 특수한 도메인에 대해서는 추가적인 개선이 필요함을 보여준다. 또한, 본 논문에서 제시한 'Deep Transfer Learning'은 엄밀히 말하면 사전 학습된 가중치를 이용한 초기화 및 미세 조정(Fine-tuning)에 가까우며, 보다 정교한 도메인 적응 기법(예: Adversarial training)에 대한 탐색은 이루어지지 않았다.

## 📌 TL;DR

본 연구는 소스 도메인의 지식을 타겟 도메인으로 전이하여 데이터 부족 문제를 해결하는 **Cross-domain Dependency Parser**를 제안한다. **STACKPTR**을 기반으로 **Multi-head Self-attention**을 통해 문맥 표현을 강화하고, 소스 도메인의 네트워크 파라미터를 재사용하는 **전이 학습** 전략을 통해 타겟 도메인에서의 구문 분석 성능을 향상시켰다. 이 접근법은 저자원 도메인의 언어 분석 시스템을 구축할 때 유용한 방법론이 될 수 있다.
