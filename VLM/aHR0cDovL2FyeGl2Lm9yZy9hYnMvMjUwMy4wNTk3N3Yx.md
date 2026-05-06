# IS YOUR VIDEO LANGUAGE MODEL A RELIABLE JUDGE?

Ming Liu, Wensheng Zhang (2025)

## 🧩 Problem to Solve

최근 비디오 언어 모델(Video Language Models, VLMs)의 활용도가 높아짐에 따라, 이들의 성능을 평가하는 강건하고 확장 가능한 방법의 필요성이 증대되었다. 전통적으로는 인간 전문가가 평가를 수행했으나, 이는 주관성에 따른 일관성 부족과 확장성의 한계라는 문제가 있다. 이를 해결하기 위해 VLM을 평가자로 사용하는 'VLM-as-a-judge' 방식이 제안되었지만, VLM 평가자의 신뢰성(reliability)에 대한 연구는 여전히 부족한 상태이다.

본 논문은 개별 VLM이 모델의 내재적 편향(bias)이나 환각(hallucination), 혹은 비디오 콘텐츠에 대한 이해 부족으로 인해 신뢰할 수 없는 평가 결과를 내놓을 수 있다는 점에 주목한다. 연구의 목표는 현재 VLM들이 평가자로서 얼마나 신뢰할 수 있는지 체계적으로 분석하고, 여러 모델의 의견을 종합하는 '집단 지성(collective thought)' 접근 방식이 이러한 신뢰성 문제를 해결할 수 있는지 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여 및 직관은 다음과 같다.

- **VLM 평가자의 신뢰성 분석**: 현재의 VLM들이 평가자로서 갖는 한계를 분석하고, 특히 성능이 낮은 모델이 높은 모델을 평가할 때 발생하는 불일치 문제를 규명하였다.
- **집단 지성 접근법의 한계 증명**: 단순히 여러 VLM의 평가를 집계하는 방식이 항상 신뢰성을 높이는 것은 아니며, 신뢰할 수 없는 모델이 포함될 경우 오히려 노이즈가 유입되어 최종 평가 결과의 품질이 저하될 수 있음을 보였다.
- **이해 능력과 평가 능력의 분리**: 성능이 낮은 모델(Video-LLaVA)을 파인튜닝하여 비디오 이해 능력을 높였음에도 불구하고, 평가자로서의 신뢰성은 크게 개선되지 않았음을 발견하였다. 이는 '콘텐츠 이해 능력'과 '비판적 평가 능력'이 서로 다른 차원의 능력임을 시사한다.

## 📎 Related Works

- **Video Language Models**: VideoChat, LLaMA-VID 등 비디오 백본과 LLM을 결합한 다양한 모델들이 제안되었으며, 최근에는 더 긴 비디오를 처리하거나 특정 도메인에 최적화된 모델들이 개발되고 있다.
- **Model Evaluation**: 기존에는 BLEU, METEOR, ROUGE, CIDER와 같은 텍스트 유사도 기반 지표를 사용했으나, 이는 복잡하고 주관적인 답변의 미묘한 차이를 포착하지 못한다. 이에 따라 LLM-as-a-judge와 같은 자동화된 평가 방식이 대두되었다.
- **Collective Decision-Making**: 집단 지성(Collective Intelligence)과 '군중의 지혜(Wisdom of Crowds)' 이론에 기반하여, 다양한 관점을 통합하면 개인의 편향을 줄이고 의사결정 정확도를 높일 수 있다는 이론적 배경을 가지고 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 연구는 '분석 후 평가(Analyze-then-Judge)' 프레임워크를 사용하며, 전체 과정은 다음과 같은 단계로 구성된다.

1. **데이터 수집**: CVRR-ES 데이터셋에서 비디오-질문 쌍$(v, t)$을 추출하고, 평가 대상인 VLM Candidate 모델들로부터 응답 $r_{ij}$를 생성한다.
2. **개별 평가**: 개별 VLM Judge들이 각 응답에 대해 1~5점 척도로 점수를 부여한다.
3. **기준점 설정 (Reference)**: 가장 신뢰할 수 있는 기준으로 'LLM Agent-Debate' 방식을 사용한다. 이는 참조 답변(Reference response)을 가진 여러 LLM 에이전트들이 토론을 통해 합의된 점수를 도출하는 방식이다.
4. **신뢰성 측정**: VLM 평가자의 점수와 LLM Agent-Debate의 점수 간의 일치도를 측정한다.

### 2. 신뢰성 측정 지표: Weighted Cohen's Kappa

단순 일치도가 아닌, 범주 간의 거리(점수 차이)를 고려하기 위해 Weighted Cohen's Kappa($\kappa$)를 사용한다.

$$ \kappa = 1 - \frac{\sum_{\alpha, \beta} w_{\alpha\beta} O_{\alpha\beta}}{\sum_{\alpha, \beta} w_{\alpha\beta} E_{\alpha\beta}} $$

여기서 $O_{\alpha\beta}$는 관측된 빈도, $E_{\alpha\beta}$는 기대 빈도이다. 가중치 $w_{\alpha\beta}$는 다음과 같은 quadratic weighting scheme을 따른다.

$$ w_{\alpha\beta} = 1 - \left( \frac{\alpha - \beta}{k-1} \right)^2 $$

($k$는 가능한 등급의 수인 5를 의미하며, $\alpha, \beta$는 1~5 사이의 정수이다.)

### 3. 집단 평가 방법론 (Collective & Mixture)

- **Collective Thought Judge**: GPT-4o와 같은 고급 모델($M^J_a$)이 여러 VLM 평가자들의 리뷰($R_1, \dots, R_q$)와 비디오-질문-응답 쌍을 모두 입력받아 최종 평가 $A$를 생성한다.
  $$ A = M^J_a(r_{i,j}, R_1, R_2, \dots, R_q) $$
- **Mixture of Judges**: 모든 모델을 사용하는 대신, 각 시각적 차원(Visual Dimension)별로 Weighted Cohen's Kappa가 특정 임계값 $\theta$ 이상이거나 상위 $k$개에 해당하는 신뢰할 수 있는 모델 집합 $M'_J$만 선택하여 최종 평가를 수행한다.
  $$ M'_J = \{ M^J_e \mid \kappa_{d,e} \ge \theta \} $$

## 📊 Results

### 1. 실험 설정

- **대상 모델**:
  - Candidates: Video-LLaVA, LLaMA-VID, GPT-4o mini, Video-ChatGPT, mPLUG-Owl-Video
  - Judges: Video-LLaVA, LLaMA-VID, GPT-4o mini, InternVL2, GPT-4o
- **데이터셋**: CVRR-ES (2,400개의 QA 쌍, 11가지 시각적 차원 포함).
- **지표**: 1~5점 척도 점수 및 Weighted Cohen's Kappa.

### 2. 주요 결과

- **VLM 평가자의 과대평가 경향**: Video-LLaVA와 LLaMA-VID는 거의 모든 차원에서 Candidate 모델들에게 매우 높은 점수(약 4.0)를 부여하는 경향을 보였다. 반면, LLM Agent-Debate는 훨씬 엄격하고 낮은 점수를 부여했다.
- **신뢰성 격차**: $\kappa$ 값 분석 결과, GPT-4o가 Agent-Debate와 가장 높은 일치도를 보였으며, Video-LLaVA와 LLaMA-VID는 일치도가 매우 낮거나 심지어 음수 값을 기록했다.
- **집단 지성의 역설**: 모든 평가자를 포함한 Collective Thought 방식은 GPT-4o 단독 평가보다 신뢰도가 낮았다. 이는 신뢰할 수 없는 모델들이 생성한 노이즈가 GPT-4o의 판단까지 방해했음을 의미한다.
- **Mixture 방식의 한계**: 신뢰도 기반으로 평가자를 선별한 Mixture of Judges 방식 역시 GPT-4o 단독 평가보다 뛰어난 성능을 보이지 못했다.

## 🧠 Insights & Discussion

- **이해 능력 $\neq$ 평가 능력**: Video-LLaVA를 파인튜닝하여 비디오 이해력을 높였음에도 불구하고, 평가 점수 분포는 여전히 상향 편향되었으며 $\kappa$ 값의 개선은 미미했다. 이는 모델이 "무엇이 일어났는가"를 아는 것과 "이 응답이 얼마나 정확한가"를 판단하는 비판적 분석 능력이 서로 다른 영역임을 시사한다.
- **Weak-to-Strong 평가의 불가능성**: 성능이 낮은 VLM(Weak)이 높은 VLM(Strong)을 평가하는 것은 불가능에 가깝다. 낮은 모델은 상위 모델의 정교한 응답을 구별해낼 비판적 추론 능력이 부족하기 때문이다.
- **집단 지성의 취약성**: "군중의 지혜"가 작동하려면 개별 구성원의 판단이 독립적이고 어느 정도의 정확성을 가져야 한다. 하지만 VLM 평가자들의 경우 공통적인 편향(과대평가)과 심한 노이즈를 가지고 있어, 단순 집계가 오히려 성능을 저하시키는 결과로 이어졌다.

## 📌 TL;DR

본 논문은 VLM을 평가자로 사용할 때의 신뢰성 문제를 다룬다. 분석 결과, 많은 VLM들이 후보 모델들의 성능을 과대평가하는 경향이 있으며, 단순히 여러 모델의 의견을 모으는 집단 지성 방식은 신뢰할 수 없는 모델의 노이즈로 인해 오히려 역효과를 낼 수 있음을 밝혀냈다. 특히 비디오 이해 능력을 높이는 것만으로는 평가자로서의 신뢰성을 확보할 수 없으므로, 향후 VLM 평가를 위해서는 단순 이해를 넘어선 **비판적 분석 능력(critical analysis ability)**에 특화된 훈련이 필요함을 시사한다.
