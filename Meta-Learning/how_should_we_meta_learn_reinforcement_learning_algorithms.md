# How Should We Meta-Learn Reinforcement Learning Algorithms?

Alexander D. Goldie, Zilin Wang, Jakob N. Foerster, Shimon Whiteson (2025)

## 🧩 Problem to Solve

강화학습(Reinforcement Learning, RL) 알고리즘의 개선은 전통적으로 인간의 직관에 의존한 수동 설계(Manual Design)를 통해 이루어져 왔다. 하지만 이러한 방식은 인간의 직관이라는 한계에 갇혀 있으며, 돌파구를 찾는 과정이 매우 번거롭다. 이에 대한 대안으로 데이터로부터 알고리즘 자체를 학습하는 메타 학습(Meta-Learning) 패러다임이 주목받고 있다. 특히 RL 알고리즘은 불안정성이 높고 많은 경우 지도 학습이나 비지도 학습의 알고리즘을 차용한 상태로 사용되어 RL 환경에 최적화되지 않은 경우가 많기에, 메타 학습의 잠재력이 매우 크다.

그럼에도 불구하고, 지금까지는 서로 다른 메타 학습 알고리즘(예: 블랙박스 함수 최적화를 위한 진화 전략, 코드 제안을 위한 LLM 활용 등) 간의 직접적인 비교 연구가 매우 부족했다. 따라서 어떤 메타 학습 방식이 특정 RL 설정에 적합한지, 그리고 각 방식의 장단점(성능, 해석 가능성, 샘플 비용, 학습 시간 등)이 무엇인지에 대한 명확한 기준이 부재한 상태이다. 본 논문의 목표는 다양한 RL 파이프라인 구성 요소에 적용되는 여러 메타 학습 알고리즘을 경험적으로 비교 분석하여, 향후 효율적인 RL 알고리즘 설계를 위한 가이드라인을 제시하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 RL 알고리즘을 학습시키기 위한 서로 다른 메타 학습 방법론들을 체계적으로 비교 분석하여 각 접근 방식의 Trade-off를 규명한 것이다. 단순히 새로운 알고리즘을 제안하는 것이 아니라, '알고리즘을 어떻게 학습시킬 것인가'라는 메타 관점에서의 분석을 수행하였다.

주요 직관은 메타 학습 알고리즘의 선택이 학습된 알고리즘의 일반화 성능(Generalisation), 샘플 효율성, 해석 가능성 및 확장성(Scalability)에 결정적인 영향을 미친다는 점이다. 이를 위해 블랙박스 학습, 증류(Distillation), 상징적 발견(Symbolic Discovery), 그리고 LLM 기반 제안이라는 네 가지 경로를 설정하고, 이를 다양한 RL 구성 요소(Optimiser, Drift function 등)에 적용하여 그 효용성을 검증하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 토대로 한다.

1. **Learned Algorithms**: Learned Policy Gradient (LPG), Learned Policy Optimisation (LPO), 그리고 Optimisation for Plasticity, Exploration and Nonstationarity (OPEN)와 같이 RL의 특정 업데이트 규칙이나 최적화 도구를 신경망으로 대체하려는 시도들이 있었다. 기존 연구들은 주로 제안된 알고리즘의 성능을 수동 설계된 베이스라인과 비교하는 데 집중했으나, 본 논문은 알고리즘을 '학습시키는 방법' 자체의 비교에 집중한다는 점에서 차별점을 가진다.
2. **Distillation**: 교사 모델(Teacher)의 지식을 학생 모델(Student)에게 전달하는 증류 기법이 정책(Policy)이나 데이터셋 수준에서 연구되어 왔다. 본 연구는 이를 알고리즘 수준으로 확장하여, 블랙박스 형태의 학습된 알고리즘을 더 작은 네트워크나 상징적 함수(Symbolic function)로 증류했을 때 일반화 성능이 향상되는지 분석한다.
3. **LLM-based Discovery**: 최근 LLM을 이용해 코드를 생성하고 알고리즘을 발견하려는 시도(예: DiscoPOP)가 증가하고 있다. 본 논문은 이러한 LLM 기반 접근법이 기존의 진화 전략이나 상징적 최적화보다 샘플 효율적인지, 그리고 실제 RL 환경에서 실용적인지를 평가한다.

## 🛠️ Methodology

### 1. 메타 학습 알고리즘 (Meta-Learning Algorithms)

본 논문에서는 알고리즘을 학습시키기 위한 다섯 가지 접근 방식을 정의한다.

- **Black-Box Meta-Learning**: 알고리즘을 신경망으로 표현한다. 학습에는 Meta-gradients(BPTT 사용) 또는 Evolution Strategies(ES)를 사용한다. ES는 적합도 함수 $F(\cdot)$를 최대화하기 위해 다음과 같은 자연 구배(Natural Gradient) 추정치를 사용하여 파라미터 $\tilde{\theta}$를 업데이트한다.
    $$\nabla_{\theta} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} [F(\theta + \sigma\epsilon)] = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} [\epsilon F(\theta + \sigma\epsilon)]$$
- **Black-Box Distillation**: 학습된 블랙박스 교사 모델을 다른 신경망(학생 모델)으로 증류한다. 동일한 크기(Same-Size) 또는 더 작은 크기(Smaller)의 네트워크를 사용하며, 환경 샘플링 대신 합성 데이터를 이용한 $L^2$ 회귀(Regression)를 통해 출력을 일치시킨다.
- **Symbolic Discovery**: 추상 구문 트리(AST)를 진화시켜 해석 가능한 수학적 함수를 직접 찾는다. 다만, RL 환경에서의 시뮬레이션 비용이 너무 커 본 논문의 경험적 분석에서는 제외하고 이론적 배경으로만 다룬다.
- **Symbolic Distillation**: 블랙박스 알고리즘을 상징적 프로그램으로 증류한다. PySR 라이브러리를 사용하여 교사 모델의 출력과 $L^2$ 손실이 가장 낮은 함수를 탐색한다.
- **LLM Proposal**: LLM(GPT o3-mini)에 알고리즘의 입력 정보와 이전 성능 데이터를 제공하여 코드를 제안하게 한다. DiscoPOP 프레임워크를 기반으로 하며, 기존의 수동 설계 알고리즘에서 시작하는 Warm-start 방식을 사용한다.

### 2. 분석 대상이 된 학습된 알고리즘 (Meta-Learned Algorithms)

메타 학습 알고리즘을 적용할 대상(Target)으로 다음 네 가지를 선정하였다.

- **LPO (Learned Policy Optimisation)**: PPO의 Mirror Drift 함수를 대체한다. 입력값은 정책 비율 $r$과 어드밴티지 $A$의 변형된 형태인 $\mathbf{x} = [(1-r), (1-r)^2, (1-r)A, \dots]$ 등을 사용하며, Mirror Learning 조건(비음수성, $r=1$에서 0 및 기울기 0)을 만족해야 한다.
- **LPG (Learned Policy Gradient)**: Actor-Critic의 업데이트 규칙 자체를 학습하며, Backward-LSTM 구조를 사용하여 고정 길이의 롤아웃 데이터를 처리한다.
- **OPEN (Optimisation for Plasticity, Exploration and Non-stationarity)**: RL 최적화의 난제인 가소성 상실(Plasticity loss), 탐색(Exploration), 비정상성(Non-stationarity)을 해결하기 위해 설계된 최적화 도구이다. 뉴런의 휴면 상태(Dormancy), 네트워크 깊이, 학습 진행도 등을 입력으로 받는다.
- **No Features**: OPEN의 단순화 버전으로, 일반적인 최적화 도구의 입력(파라미터, 그라디언트, 모멘텀)만 사용한다.

## 📊 Results

실험은 Ant(Brax), MinAtar 등 다양한 환경에서 수행되었으며, 내부 분포(In-distribution) 성능과 외부 분포(Out-of-distribution) 일반화 성능을 모두 측정하였다.

### 1. 알고리즘별 성능 분석

- **LPO**: 모든 증류 방식이 비슷한 성능을 보였으며, 특히 Same-size distillation이 일반화 성능을 소폭 향상시켰다. LLM 제안 방식은 내부 성능은 낮았으나, OOD 일반화 성능은 가장 뛰어났다. 이는 LLM이 PPO와 유사한, 검증된 형태의 함수를 제안했기 때문으로 분석된다.
- **No Features**: LLM 제안 방식이 내부 성능과 일반화 성능 모두에서 압도적으로 강했다. 반면, 블랙박스 학습이 실패한 경우 증류를 통한 성능 개선은 불가능했다.
- **Feed-Forward OPEN**: 입력 변수가 많아짐에 따라 LLM과 상징적 증류가 완전히 실패하였다. LLM은 제공된 입력 특징들을 적절히 활용하는 코드를 생성하지 못했으며, 상징적 증류는 고차원 공간 탐색에 어려움을 겪어 단순한 상수 함수로 수렴하는 경향을 보였다.
- **Recurrent LPG**: 증류(Same-size, Smaller 모두)를 적용했을 때 OOD 일반화 성능이 향상되었다. 이는 증류 과정이 일종의 정규화(Regularization) 역할을 하여 분산을 줄였기 때문으로 해석된다.
- **Recurrent OPEN**: 긴 롤아웃(Long rollout) 특성상 증류가 매우 어려웠으며 성능이 낮게 나타났다. LLM은 Adam과 유사한 단순한 최적화 도구를 제안하여 OOD에서 좋은 성능을 냈으나, 이는 하이퍼파라미터 튜닝에 크게 의존한 결과였다.

### 2. 종합 지표 비교

| 접근 방식 | 샘플 비용 | 학습 시간 | 테스트 시간 | 해석 가능성 | 확장성 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Black-Box Learning | 높음 | 느림 | 느림 | 나쁨 | 최상 |
| Same-Size Distillation | 추가 없음 | 느림 | 느림 | 나쁨 | 좋음 |
| Smaller Distillation | 추가 없음 | 중간 | 중간 | 나쁨 | 좋음 |
| Symbolic Distillation | 추가 없음 | 중-느림 | 빠름 | 중간 | 나쁨 |
| LLM Proposal | 낮음 | 빠름 | 빠름 | 좋음 | 중간 |

## 🧠 Insights & Discussion

본 연구는 메타 학습 알고리즘 선택에 있어 다음과 같은 통찰을 제공한다.

첫째, **LLM 기반 제안의 효용성과 한계**이다. LLM은 매우 샘플 효율적이며 일반화 능력이 뛰어난 알고리즘을 빠르게 찾을 수 있다. 하지만 이는 입력 변수가 적고 이해하기 쉬우며, 검색을 시작할 수 있는 강력한 Warm-start 알고리즘이 존재할 때만 가능하다. 고차원 입력 특징을 모두 활용해야 하는 복잡한 알고리즘(예: OPEN)의 경우 LLM의 성능은 급격히 저하된다.

둘째, **증류의 정규화 효과**이다. 블랙박스 알고리즘을 동일한 크기의 네트워크로 증류하는 것은 추가적인 샘플 비용 없이 일반화 성능을 높일 수 있는 "저렴한" 방법이다. 다만, 교사 모델 자체가 성능이 낮다면 증류는 아무런 도움이 되지 않는다.

셋째, **상징적 표현의 확장성 문제**이다. 상징적 증류는 해석 가능성이라는 큰 장점이 있지만, 입력 변수의 개수가 증가함에 따라 탐색 공간이 기하급수적으로 커져 실질적으로 사용하기 어렵다.

결과적으로, 다수의 특징(Feature)을 사용해야 하는 확장성 있는 알고리즘을 개발하기 위해서는 여전히 **블랙박스 메타 학습**이 유일하고 가장 실용적인 방법이다.

## 📌 TL;DR

본 논문은 RL 알고리즘을 학습시키기 위한 다양한 메타 학습 방법론(블랙박스, 증류, 상징적 발견, LLM)을 체계적으로 비교하였다. 분석 결과, **단순한 구조에서는 LLM 제안 방식**이 가장 효율적이며, **복잡한 고차원 특징을 다룰 때는 블랙박스 학습**이 필수적이고, **일반화 성능을 약간 높이고 싶을 때는 동일 크기의 블랙박스 증류**가 효과적임을 밝혀냈다. 이 가이드라인은 향후 연구자들이 불필요한 실험 비용을 줄이고 목적에 맞는 최적의 메타 학습 전략을 선택하는 데 기여할 것으로 기대된다.
