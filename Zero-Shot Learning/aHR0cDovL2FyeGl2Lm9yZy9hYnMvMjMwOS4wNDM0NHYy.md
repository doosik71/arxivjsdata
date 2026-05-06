# ZERO-SHOT ROBUSTIFICATION OF ZERO-SHOT MODELS

Dyah Adila, Changho Shin, Linrong Cai, Frederic Sala (2024)

## 🧩 Problem to Solve

본 논문은 Zero-shot inference 모델이 대규모 사전 학습 데이터로부터 상속받은 편향(inherited biases) 또는 가짜 상관관계(spurious correlations)로 인해 발생하는 성능 저하 문제를 해결하고자 한다. 예를 들어, 시각-언어 모델이 '물새(waterbird)'를 인식할 때 새의 특징이 아닌 '물 배경'이라는 가짜 상관관계에 의존하여, 육지에 있는 물새를 잘못 분류하는 경우가 이에 해당한다.

이러한 문제는 특히 데이터의 희귀 슬라이스(rare data slices)에서 심화되며, 결과적으로 Worst-group accuracy를 크게 떨어뜨린다. 기존의 해결책인 Fine-tuning은 레이블링된 데이터가 필요하며, 이는 추가 학습 없이 즉시 사용 가능하다는 Zero-shot 모델의 핵심적인 장점을 훼손한다. 따라서 본 연구의 목표는 레이블 데이터, 추가 학습, 그리고 수동적인 개념 식별 없이도 Zero-shot 모델의 강건성(robustness)을 향상시키는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 언어 모델(Language Model, LM)이 보유한 지식을 활용하여 모델 스스로 혹은 다른 모델을 개선하는 '실행 가능한 통찰(actionable insights)'을 추출하는 것이다.

ROBOSHOT은 태스크 설명(task description)만을 이용하여 LM으로부터 긍정적(helpful) 통찰과 부정적(harmful) 통찰을 얻고, 이를 임베딩 공간으로 투영하여 유해한 성분은 제거하고 유익한 성분은 증폭시킨다. 즉, 모델의 가중치를 수정하는 대신 입력 및 클래스 임베딩을 조정하여 가짜 상관관계를 중화하고 실제 유용한 특성을 강조하는 방식으로 강건성을 확보한다.

## 📎 Related Works

기존의 Zero-shot 강건성 향상 연구들은 주로 레이블 데이터를 이용한 Fine-tuning이나 Contrastive loss를 통한 어댑터 학습에 의존하였다. 그러나 이는 Zero-shot의 이점을 포기하게 만든다는 한계가 있다. 또한, 단어 임베딩의 디바이아싱(debiasing) 연구들은 성별과 같은 특정 개념을 수동으로 지정해야 하므로 다양한 태스크에 범용적으로 적용하기 어렵다.

ROBOSHOT은 다음과 같은 점에서 기존 연구와 차별화된다. 첫째, 어떠한 레이블 데이터나 추가 학습 없이 완전히 Zero-shot 방식으로 작동한다. 둘째, 사람이 직접 디바이아싱 벡터를 정의하는 대신, LM을 통해 유해/유익 개념을 자동으로 추출함으로써 자동화를 달성하였다. 셋째, 단순히 유해 성분을 제거하는 것을 넘어 유익한 성분을 증폭시키는 메커니즘을 도입하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

ROBOSHOT은 입력 임베딩 $x$와 클래스 임베딩 $c$를 수정하여 가짜 상관관계를 제거하고 유용한 특징을 강화하는 시스템이다. 전체 과정은 'LM을 통한 통찰 추출 $\rightarrow$ 임베딩 공간의 부분공간(subspace) 정의 $\rightarrow$ 벡터 투영 및 수정' 순으로 진행된다.

### 주요 구성 요소 및 절차

1. **Insight Representations 추출**:
   - LM에게 태스크 설명을 제공하고 "두 클래스 간의 편향된/가짜 차이점은 무엇인가?"(harmful)와 "두 클래스의 실제 특성은 무엇인가?"(helpful)를 질문한다.
   - 얻어진 텍스트 통찰 $\{s_1, s_2\}$를 텍스트 인코더 $g$를 통해 임베딩하고, 그 차이를 통해 통찰 벡터 $v$를 생성한다.
   - $$v = \frac{g(s_1) - g(s_2)}{\|g(s_1) - g(s_2)\|}$$
   - 다수의 통찰 벡터가 추출될 경우, 표준 행렬 분해 기법(QR decomposition 등)을 사용하여 유해 및 유익 성분을 위한 직교 기저(orthogonal basis)를 구성한다.

2. **Harmful Component 제거 (Vector Rejection)**:
   - 입력 임베딩 $x$에서 유해 통찰 벡터 $v_j$ 방향의 성분을 제거한다.
   - $$x \leftarrow x - \frac{\langle x, v_j \rangle}{\|v_j\|^2} v_j$$
   - 이후 $\|x\|$로 다시 정규화(renormalize)하여 크기를 맞춘다.

3. **Helpful Component 증폭 (Vector Addition)**:
   - 제거 단계 이후, 유익 통찰 벡터 $u_k$ 방향의 성분을 더하여 해당 특성을 강조한다.
   - $$x \leftarrow x + \frac{\langle x, u_k \rangle}{\|u_k\|^2} u_k$$

### Label-Free Adaptation (LFA)

추가적인 강건성 향상을 위해, 레이블이 없는 훈련 세트와 아주 적은 양의 검증 데이터(약 100개)만 사용하여 투영 행렬 $\Pi$를 학습하는 LFA 변형 모델을 제안한다. 손실 함수 $\mathcal{L}_{LFA}$는 $\Pi x$와 유해 통찰 $v$의 내적은 최소화하고, 유익 통찰 $u$와의 내적은 최대화하도록 설계되었다.
$$\mathcal{L}_{LFA}(\Pi x, u, v) = \frac{1}{|S|} \sum_{j=1}^{S} \langle \Pi x, v_j \rangle - \frac{1}{|R|} \sum_{k=1}^{R} \langle \Pi x, u_k \rangle$$

### 이론적 분석

논문은 임베딩이 유해($z_s$), 유익($z_r$), 무해($z_b$) 성분의 혼합으로 구성되어 있다고 가정한다. Theorem 4.1과 4.2를 통해, ROBOSHOT 적용 후 유해 성분의 계수 $A_s$는 감소하고 유익 성분의 계수 $A_r$은 증가함을 수학적으로 증명하였다. 특히, 통찰 벡터의 노이즈 $\sigma_{insight}$가 작을수록, 즉 LM이 더 정확한 통찰을 제공할수록 강건성 향상 폭이 커짐을 보였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 시각-언어 태스크(Waterbirds, CelebA, PACS, VLCS, CXR14)와 NLP 태스크(CivilComments, HateXplain, Amazon, Gender Bias) 총 9개 데이터셋을 사용하였다.
- **측정 지표**: 전체 평균 정확도(AVG), 최악 그룹 정확도(WG, Worst-group Accuracy), 그리고 둘 사이의 차이인 Gap을 측정하였다.
- **비교 대상**: Vanilla Zero-shot (ZS), 그룹 정보가 포함된 프롬프트를 사용하는 Group Prompt ZS, LM에 직접 질문하는 Direct Prompting (ChatGPT, BART-MNLI) 등과 비교하였다.

### 주요 결과

- **강건성 향상**: ROBOSHOT은 9개 데이터셋에서 평균적으로 Worst-group accuracy를 $15.98\%$ 향상시켰으며, AVG 성능은 거의 유지하거나 소폭 향상시키는 결과를 보였다.
- **범용성**: CLIP (ViT-B-32, ViT-L-14), ALIGN, AltCLIP 등 다양한 사전 학습 모델과 호환됨을 확인하였다.
- **NLP 적용**: 텍스트 분류에서도 유해 성분 제거를 통해 BERT 및 Ada 임베딩 모델의 WG 성능을 유의미하게 개선하였으며, 일부 사례에서는 최신 LLM(ChatGPT)의 직접 프롬프팅 성능에 근접하는 결과를 얻었다.
- **LM 용량 영향**: ChatGPT뿐만 아니라 Flan-T5, GPT2, LLaMA와 같은 다양한 용량의 LM을 사용해 통찰을 추출했을 때도 Vanilla ZS보다 우수한 성능을 보였으며, 이는 LM의 능력이 높을수록 성능이 더 좋아지는 상관관계를 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석

ROBOSHOT의 가장 큰 강점은 사전 학습 모델의 가중치를 건드리지 않고도, 오직 임베딩 공간에서의 기하학적 조작만으로 강건성을 확보했다는 점이다. 마진 분석(Margin Analysis) 결과, 유해 성분 제거는 가짜 상관관계가 있는 데이터($D_{sp}$)의 마진을 증가시켜 오류를 줄이는 효과가 있으며, 이후 유익 성분 추가 단계가 전반적인 마진을 다시 끌어올려 성능 저하를 방지함을 확인하였다.

### 한계 및 논의사항

- **모델 간 차이**: ALIGN 모델의 Waterbirds 태스크에서는 성능 향상이 미비했는데, 이는 ALIGN의 텍스트 임베딩 공간에서 유해 통찰과 유익 통찰 벡터가 충분히 구분되지 않고 붕괴(collapse)되어 있었기 때문으로 분석된다. 이는 모델의 임베딩 공간 특성에 따라 ROBOSHOT의 효과가 달라질 수 있음을 시사한다.
- **가정의 유효성**: 본 논문은 임베딩이 유해, 유익, 무해 성분의 선형 결합으로 이루어져 있다고 가정한다. 저자들은 구글 이미지 검색 결과의 평균 임베딩을 이용한 실험을 통해 무해 성분이 평균화 과정에서 상쇄될 수 있음을 보여주었으나, 실제 복잡한 데이터셋에서 이 가정이 완벽하게 성립하는지에 대해서는 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 레이블 데이터나 추가 학습 없이, 언어 모델(LM)에서 추출한 태스크 통찰을 이용해 Zero-shot 모델의 임베딩을 수정함으로써 가짜 상관관계를 제거하고 강건성을 높이는 **ROBOSHOT**을 제안한다. 유해 성분은 투영을 통해 제거하고 유익 성분은 증폭시키는 단순한 벡터 연산만으로 9개 이미지 및 NLP 데이터셋에서 Worst-group accuracy를 평균 $15.98\%$ 향상시켰다. 이 연구는 거대 모델의 편향을 해결하기 위해 데이터 수집이나 Fine-tuning 없이도 임베딩 수준의 조작만으로 충분한 효과를 거둘 수 있음을 입증하여, 향후 Zero-shot 모델의 배포 및 최적화 과정에서 중요한 역할을 할 가능성이 높다.
