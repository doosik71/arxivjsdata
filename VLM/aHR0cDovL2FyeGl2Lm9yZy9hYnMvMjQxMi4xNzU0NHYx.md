# Retention Score: Quantifying Jailbreak Risks for Vision Language Models

Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho (2025)

## 🧩 Problem to Solve

본 논문은 시각-언어 모델(Vision-Language Models, VLMs)의 보안 취약점, 특히 모델의 안전 가이드라인을 우회하여 유해한 출력을 유도하는 '제일브레이크(Jailbreak)' 공격에 대한 복원력을 정량적으로 평가하는 문제를 다룬다. VLMs는 컴퓨터 비전과 대규모 언어 모델(LLMs)을 결합하여 다중 모달 능력을 향상시켰으나, 이로 인해 시각적 공간과 텍스트 공간이라는 두 가지 공격 경로가 생성되어 공격 표면이 확대되었다.

기존의 적대적 공격 기반 평가 방식은 새로운 더 강력한 공격 기법이 발견되면 이전의 평가 결과가 무효화될 수 있다는 비일관성 문제가 있으며, 최적화 기반의 공격 생성 방식은 계산 비용과 시간이 매우 많이 소요된다는 단점이 있다. 따라서 본 연구의 목표는 특정 공격 기법에 의존하지 않고(Attack-agnostic), 모델의 복원력을 수학적으로 보증할 수 있는 '인증된 복원력(Certified Robustness)' 지표인 Retention Score를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 생성 모델(Generative Model)을 사용하여 입력 데이터 주변의 세만틱 공간을 탐색하고, 유해성과 비유해성 점수 간의 마진(Margin)을 계산함으로써 모델의 견고함을 측정하는 것이다. 주요 기여 사항은 다음과 같다.

- **Retention Score 제안**: 시각적 요소에 대한 Retention-I와 텍스트 요소에 대한 Retention-T로 구성된 다중 모달 복원력 측정 지표를 설계하였다.
- **복원력 인증(Robustness Certification)**: Retention Score가 $\ell_2$-norm 범위 내의 섭동(Perturbation)에 대해 모델의 안전성을 보장하는 인증서 역할을 할 수 있음을 수학적으로 증명하였다.
- **VLM 취약성 분석**: 시각적 모듈의 통합이 오히려 모델의 제일브레이크 저항력을 약화시킬 수 있음을 발견하였으며, 이는 단순 LLM보다 VLM이 더 취약할 수 있음을 시사한다.
- **효율성 및 범용성 입증**: 기존의 최적화 기반 공격 방식보다 계산 시간을 최대 30배 단축하였으며, 블랙박스 API 모델(GPT-4V, Gemini Pro Vision)에도 적용 가능한 평가 프레임워크를 제공하였다.

## 📎 Related Works

논문에서는 VLMs의 정렬(Alignment)과 적대적 공격에 관한 기존 연구들을 소개한다.

- **VLM 정렬**: RLHF(Reinforcement Learning from Human Feedback)와 지침 튜닝(Instruction Tuning)을 통해 모델이 인간의 가치에 부합하고 유해한 콘텐츠를 생성하지 않도록 하는 연구들이 진행되어 왔다.
- **제일브레이크 공격**: 단일 시각적 적대적 예제만으로도 정렬된 VLM을 무력화할 수 있다는 연구(Qi et al. 2023a)나, 텍스트 영역에서의 GCG, AutoDAN과 같은 정교한 프롬프트 공격 기법들이 제시되었다.
- **복원력 평가 지표**: CLEVER Score나 GREAT Score와 같이 립시츠 상수(Lipschitz constant) 등을 이용해 지역적/전역적 복원력을 측정하려는 시도가 있었으나, VLM의 다중 모달 환경에서 조건부 복원력을 평가하는 방식은 미비하였다.

본 연구는 특정 공격 시나리오를 생성하여 성공률을 측정하는 기존 방식과 달리, 입력 분포 주변의 마진을 측정함으로써 '공격에 무관한(Attack-agnostic)' 인증 지표를 제공한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

Retention Score의 전체 흐름은 다음과 같다.

1. **샘플 생성**: 조건부 확산 모델(Conditional Diffusion Model)을 사용하여 원본 이미지나 텍스트와 세만틱하게 유사한 합성 샘플들을 대량으로 생성한다.
2. **VLM 추론**: 생성된 샘플들을 VLM에 입력하여 응답을 얻는다.
3. **유해성 판단**: VLM의 응답을 유해성 판단 분류기(Toxicity Judgment Classifier, 예: Perspective API 또는 Llama-70B)에 통과시켜 유해성 점수를 획득한다.
4. **점수 계산**: 비유해성 점수와 유해성 점수의 차이(마진)를 기반으로 Retention Score를 산출한다.

### 2. 세부 구성 요소 및 방정식

#### A. 유해성 판단 모델 및 공간 정의

VLM과 판단 분류기가 결합된 전체 시스템을 $M: \mathbb{R}^d \times \Lambda \to \Pi_2$로 정의한다. 여기서 $M^{nt}$는 비유해성(non-toxic) 확률, $M^t$는 유해성(toxic) 확률을 나타낸다. 텍스트의 경우 이산적인 특성 때문에 세만틱 인코더 $s$와 디코더 $\psi$를 도입하여 연속적인 벡터 공간 $\mathbb{R}^k$로 변환하여 처리한다.

#### B. Retention-Image Score (Retention-I)

이미지 $I$와 텍스트 프롬프트 집합 $X=\{T_1, \dots, T_m\}$이 주어졌을 때, 확산 모델 $G^I(z|I)$를 통해 유사 이미지를 생성한다. 개별 샘플에 대한 지역 점수 함수 $g^I$는 다음과 같이 정의된다.

$$g^I(M, G^I(z|I), T) = \sqrt{\frac{\pi}{2}} \cdot \{M^{nt}(G^I(z|I), T) - M^t(G^I(z|I), T)\}_+$$

여기서 $\{\cdot\}_+ = \max\{\cdot, 0\}$이다. 최종 Retention-I Score는 모든 프롬프트와 생성된 샘플들에 대한 평균값으로 계산된다.

$$R^I(M, I, X) = \frac{1}{m \cdot n} \sum_{j=1}^{m} \sum_{i=1}^{n} g^I(M, G^I(z_i|I), T_j)$$

#### C. Retention-Text Score (Retention-T)

텍스트의 경우, 패러프레이징 확산 모델 $G^T(z|T)$를 사용하여 세만틱하게 유사한 텍스트를 생성한다. 지역 점수 함수 $g^T$는 다음과 같다.

$$g^T(M, I, s(G^T(z|T))) = \sqrt{\frac{\pi}{2}} \cdot \{M^{nt}(I, \psi(s(G^T(z|T)))) - M^t(I, \psi(s(G^T(z|T))))\}_+$$

최종 Retention-T Score $R^T$는 동일하게 평균을 내어 산출하며, 이는 텍스트 섭동에 대한 복원력을 의미한다.

### 3. 복원력 인증 (Robustness Certification)

논문은 Theorem 1을 통해 Retention Score의 수학적 의미를 부여한다.

- **핵심 정리**: 이미지 섭동 $\delta^I$ 또는 텍스트 세만틱 섭동 $\delta^T$의 크기가 각각 $R^I$ 또는 $R^T$보다 작다면($\|\delta\|_2 < R$), 모델 $M$은 해당 입력에 대해 반드시 비유해성 결과($M^{nt} \geq 0.5$)를 유지한다.
- 이는 Retention Score가 단순한 통계치가 아니라, 모델이 안전하게 유지되는 **최소 섭동 반지름의 하한선**임을 보증하는 인증서(Certificate) 역할을 함을 의미한다.

## 📊 Results

### 1. 실험 설정

- **대상 모델**: MiniGPT-4, LLaVA, InstructBLIP 및 이들의 베이스 LLM, 그리고 블랙박스 API인 GPT-4V, Gemini Pro Vision.
- **데이터셋**: RealToxicityPrompts(이미지 공격용), AdvBench Harmful Behaviours(텍스트 공격용).
- **측정 지표**: Retention Score 및 공격 성공률(ASR, Attack Success Rate).

### 2. 주요 결과

- **공격 성공률과의 상관관계**: 이미지 및 텍스트 공격 모두에서 Retention Score가 높을수록 ASR이 낮게 나타났다. 특히 이미지 공격에서 MiniGPT-4 > InstructBLIP > LLaVA 순으로 복원력이 높았으며, 이는 실제 ASR 순위와 일치한다.
- **블랙박스 API 평가**: Gemini Pro Vision의 경우, 내부 보안 설정(None, Few, Some, Most)에 따라 Retention-I Score가 뚜렷하게 변화함을 확인하였다. GPT-4V의 복원력은 Gemini의 'Some'과 'Most' 설정 사이의 수준인 것으로 나타났다.
- **시각적 모듈 통합의 영향**: 매우 흥미로운 점은, 단순 LLM에 시각적 모듈을 추가하여 VLM을 만들었을 때 LLaVA와 InstructBLIP의 경우 Retention-T Score가 감소하고 ASR이 크게 증가했다는 것이다. 이는 시각적 인터페이스의 추가가 모델의 전반적인 안전 가드레일을 약화시킬 수 있음을 보여준다.
- **계산 효율성**: 기존의 최적화 기반 적대적 공격 방식과 비교했을 때, Retention Score 측정 방식은 동일 샘플당 실행 시간을 약 2배에서 30배까지 단축시켰다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 논문은 VLM의 안전성을 평가함에 있어 '특정 공격 기법의 성공 여부'라는 단편적인 측정에서 벗어나, '입력 공간의 어느 정도 범위까지 안전한가'라는 수학적 보증(Certification)의 개념을 도입하였다. 특히, 블랙박스 모델에 대해서도 생성 모델을 이용해 복원력을 측정할 수 있다는 점은 실무적으로 매우 유용한 도구가 될 수 있다.

### 한계 및 비판적 해석

- **$\ell_2$-norm 중심의 분석**: 본 논문의 이론적 보증은 $\ell_2$-norm 기반의 섭동에 집중되어 있다. 그러나 텍스트 공격의 경우 단어의 변경이나 교체와 같은 $\ell_0$-norm 기반의 변화가 더 치명적일 수 있는데, 이에 대한 분석이 부족하다는 점이 한계로 명시되어 있다.
- **생성 모델의 의존성**: Retention Score의 정확도는 사용된 확산 모델이 원본 데이터 주변의 세만틱 공간을 얼마나 잘 샘플링하느냐에 달려 있다. 만약 생성 모델이 특정 편향을 가지고 있다면 복원력 측정 결과에 왜곡이 생길 가능성이 있다.
- **시각 모듈의 취약성 기전**: 시각 모듈 추가가 왜 안전성을 저하시키는지에 대해 LLaVA의 동적 튜닝 구조 등을 언급하였으나, 구체적인 내부 메커니즘에 대한 심층 분석보다는 현상 관찰에 그친 측면이 있다.

## 📌 TL;DR

본 논문은 VLM의 제일브레이크 위험을 정량화하기 위해, 생성 모델을 이용해 입력 주변의 안전 마진을 측정하는 **Retention Score**를 제안한다. 이 지표는 수학적으로 $\ell_2$-norm 범위 내의 복원력을 보증하는 인증서 역할을 하며, 기존 공격 기반 평가보다 훨씬 빠르게 모델의 견고함을 순위화할 수 있다. 특히, 시각적 모듈의 통합이 오히려 모델의 안전성을 해칠 수 있다는 점을 밝혀내어, 향후 안전한 다중 모달 모델 설계의 중요성을 시사한다.
