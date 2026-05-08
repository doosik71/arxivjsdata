# Imitation Learning in the Deep Learning Era: A Novel Taxonomy and Recent Advances

Iason Chrysomallis, Georgios Chalkiadakis (2025)

## 🧩 Problem to Solve

본 논문은 딥러닝 시대의 Imitation Learning (IL, 모방 학습) 분야에서 발생하는 정의의 모호함과 기존 분류 체계의 한계를 해결하고자 한다. 구체적으로는 다음과 같은 문제들에 집중한다.

첫째, 전문가(Expert)가 제공하는 데이터의 형태가 매우 다양함에도 불구하고, 이를 체계적으로 구분하여 학습 전략과 연결 짓는 기준이 부족하다. 전문가 데이터는 완전한 state-action 궤적부터 단순한 상태 관찰(observation), 또는 텍스트 지시어에 이르기까지 범위가 매우 넓다.

둘째, 에이전트가 전문가의 행동을 얼마나 정확하게 복제해야 하는지에 대한 목표 설정이 불분명하다. 단순한 모방을 넘어 전문가의 성능을 초과하거나, 환경의 제약 조건에 적응해야 하는 실무적인 요구사항이 증가하고 있다.

셋째, 기존의 IL 분류법들은 최근의 연구 트렌드, 특히 Implicit Imitation이나 Adversarial-based 방법론의 급격한 발전을 충분히 반영하지 못하고 있다. 따라서 본 논문은 최신 연구 흐름을 반영한 새로운 Taxonomy(분류 체계)를 제안하고, 이를 바탕으로 최신 연구 동향을 분석하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥러닝 기반 모방 학습의 최신 연구들을 체계적으로 정리하기 위해 새로운 분류 체계를 제안하고, 이를 통해 각 방법론의 기술적 특성과 한계를 분석한 것이다.

가장 중심적인 아이디어는 전문가 데이터의 가용성과 학습 목적에 따라 IL을 **Explicit Imitation**, **Implicit Imitation**, 그리고 **Inverse Reinforcement Learning (IRL)**의 세 가지 상위 카테고리로 구분하는 것이다. 특히, 행동 정보($a$) 없이 상태 전이($s \rightarrow s'$)만을 관찰하여 학습하는 Implicit Imitation의 비중을 높게 설정하여 최신 트렌드를 반영하였다. 또한, 단순히 알고리즘을 나열하는 것이 아니라 Covariate Shift, 전문가의 Suboptimality, Multi-modality와 같은 핵심 난제들을 중심으로 각 논문이 이를 어떻게 해결하려 했는지 비판적으로 검토한다.

## 📎 Related Works

논문은 기존의 IL 관련 서베이 연구들이 특정 도메인(예: 로보틱스)에 국한되거나, 초기 IL 방법론들에 치중되어 있음을 지적한다. 특히, Knowledge Distillation(KD)과의 차이점을 명확히 정의하며 관련 연구의 범위를 한정한다.

KD는 주로 교사 모델의 Soft Label(확률 분포)을 모방하는 반면, IL은 전문가가 선택한 특정 행동(Hard Label)을 모방하는 것에 집중한다. 또한, 기존의 Taxonomy들이 환경의 역할이나 단순 알고리즘 종류에 따라 분류했다면, 본 논문은 데이터의 가용성(Explicit vs Implicit)을 최상위 기준으로 삼아 실무적인 데이터 수집 제약 사항을 더 잘 반영하도록 설계되었다.

## 🛠️ Methodology

본 논문은 제안하는 Taxonomy에 따라 IL의 방법론을 다음과 같이 구조화하여 설명한다.

### 1. Explicit Imitation

전문가가 상태($s$)와 행동($a$) 정보를 모두 제공하는 설정이다.

* **Behavioral Cloning (BC):** 전문가의 데이터를 지도 학습(Supervised Learning) 방식으로 직접 매핑하는 가장 기초적인 방법이다. 주요 문제는 에이전트가 훈련 데이터에 없는 상태에 진입했을 때 오류가 누적되는 Covariate Shift이다.
* **Adversarial Methods:** GAIL(Generative Adversarial Imitation Learning)이 대표적이다. Generator(에이전트 정책)와 Discriminator(전문가와 에이전트 구분기)가 서로 경쟁하며 학습한다. Discriminator의 출력값 $\mathcal{D}(s, a)$를 이용해 다음과 같은 보상 함수를 정의한다.
$$r(s, a) = -\log(1 - \mathcal{D}(s, a))$$
이 보상은 에이전트가 전문가의 행동 분포에 가까워지도록 유도한다.

### 2. Implicit Imitation

전문가의 행동($a$) 정보 없이 상태 전이($s, s'$)만을 관찰하는 설정이다.

* **Model-Based:** Inverse Dynamics Model을 먼저 학습하여 상태 전이를 일으킨 행동을 추론한 뒤, 이를 바탕으로 BC를 수행한다. (예: BCO)
* **Model-Free:** 행동 추론 과정 없이 직접 정책을 학습한다. 최근에는 Diffusion Model을 Discriminator로 사용하여 더 밀도 높은 보상 신호를 생성하는 DiffAIL과 같은 방법이 등장했다. 또한 DIIQN과 같이 DRL과 IL을 결합하여 전문가의 가이드라인을 따르되 환경의 보상을 통해 전문가를 능가하도록 설계된 구조가 있다.

### 3. Inverse Reinforcement Learning (IRL)

행동의 복제가 아닌, 전문가가 최적화하고자 하는 잠재적인 보상 함수 $\mathcal{R}$을 추론하는 것이 목표이다. 보상 함수가 복원되면 표준 RL 알고리즘을 통해 정책을 도출할 수 있다. 최근에는 데이터의 정렬(Data Alignment)보다는 작업의 의도(Task Alignment)를 파악하기 위해 궤적 간의 랭킹(Ranking) 정보를 사용하는 방식이 제안되었다.

## 📊 Results

본 논문은 특정 실험 결과보다는 다수의 논문을 분석한 종합적인 리뷰 결과를 제시한다.

* **정량적 성과:** DIIQN과 같은 최신 implicit 방법론들이 표준 DRL보다 학습 속도가 훨씬 빠르며, 일부 설정에서는 전문가의 성능을 능가함을 확인하였다.
* **정성적 분석:** Adversarial 방법론은 Covariate Shift 해결에 효과적이지만 학습 불안정성(Gradient Explosion)이 크며, 이를 해결하기 위해 보상 값을 클리핑하는 CREDO 등의 기법이 유효함을 분석하였다.
* **적용 도메인:** 자율주행, 로보틱스, 헬스케어, 텍스트 생성 등 다양한 도메인에서 IL이 적용되고 있으며, 특히 고차원 시각 데이터나 텍스트 시퀀스 데이터로의 확장성이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 딥러닝 기반 IL의 현재 위치와 미래 과제에 대해 다음과 같은 통찰을 제공한다.

**강점 및 성과:**
최신 IL 연구들은 단순한 모방을 넘어 전문가의 Suboptimality(불완전함)를 처리하는 방향으로 진화하고 있다. 중요도 샘플링(Importance Weighting)이나 Negative Learning을 통해 노이즈가 섞인 데이터에서도 최적의 정책을 추출하려는 시도가 돋보인다.

**한계 및 미해결 질문:**

1. **안전성(Safety):** 특히 자율주행이나 의료 분야에서 학습 과정 중의 안전성을 보장하는 메커니즘이 여전히 부족하다.
2. **데이터 효율성:** 고품질의 전문가 데이터를 수집하는 비용이 매우 높음에도 불구하고, 적은 데이터로 일반화 성능을 높이는 방법론에 대한 연구가 더 필요하다.
3. **다중 에이전트(Multi-agent):** 단일 에이전트 모방에 비해 다중 에이전트 환경에서의 상호작용을 모방하는 연구는 상대적으로 미진한 상태이다.
4. **평가 표준화:** 각 논문이 서로 다른 벤치마크와 환경을 사용하고 있어, 객관적인 성능 비교가 어렵다는 점이 지적된다.

## 📌 TL;DR

본 논문은 딥러닝 시대의 모방 학습(IL)을 **Explicit**, **Implicit**, **IRL**로 구분하는 새로운 분류 체계를 제안하고, 최신 연구들을 심도 있게 분석한 서베이 논문이다. 특히 행동 정보가 없는 Implicit Imitation의 중요성을 강조하며, Covariate Shift와 전문가의 불완전성 문제를 해결하려는 최신 기법들을 정리하였다. 이 연구는 향후 더 안전하고 데이터 효율적인 모방 학습 시스템을 설계하는 데 필요한 이론적 토대와 벤치마크 방향성을 제시한다는 점에서 중요한 가치를 가진다.
