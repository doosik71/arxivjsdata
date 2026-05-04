# GPT Adoption and the Impact of Disclosure Policies

Cathy Yang, David Restrepo Amariles, Leo Allen, Aurore Troussel (2025)

## 🧩 Problem to Solve

본 연구는 조직 내에서 생성형 사전 학습 트랜스포머(Generative Pre-trained Transformers, GPT) 특히 ChatGPT와 같은 대규모 언어 모델(LLM)의 도입 과정에서 발생하는 '그림자 도입(shadow adoption)', 즉 사용자가 조직에 알리지 않고 AI 도구를 사용하는 현상을 분석한다. 전문 서비스 기업(컨설팅, 감사, 법률 등)의 경우, AI 사용으로 인한 기밀 유출이나 잘못된 정보 생성(hallucination)과 같은 법적·운영적 리스크가 존재하며, 이는 관리자와 실무자 간의 이해관계 상충을 야기한다.

연구의 핵심 문제는 대리인 이론(Agency Theory)의 관점에서 볼 때, 실무자(Agent)가 GPT를 사용하여 생산성을 높이더라도 이를 관리자(Principal)에게 공개하지 않음으로써 발생하는 정보 비대칭성(Information Asymmetry)과 도덕적 해이(Moral Hazard)가 조직 전체의 GPT 도입과 효율적인 리스크 관리를 어떻게 저해하는가 하는 점이다. 따라서 본 논문은 공개 정책(Disclosure Policy)이 이러한 정보 비대칭을 해소하고 관리자와 실무자 간의 인센티브 정렬(Incentive Alignment)에 어떤 영향을 미치는지 규명하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 중심적인 직관은 GPT 도입의 저해 요인이 단순한 기술적 거부감이 아니라, 조직 내 위계 구조에서 발생하는 대리인 비용(Agency Costs)에 있다는 점이다. 주요 기여 사항은 다음과 같다.

1. **대리인 이론의 적용**: 기존의 AI 도입 연구가 주로 개인의 생산성이나 기술 수용 모델(TAM)에 집중한 것과 달리, 관리자와 실무자 간의 주체-대리인 관계를 통해 조직적 차원의 도입 메커니즘을 분석하였다.
2. **공개 정책의 이중성 규명**: 공개 정책이 정보 비대칭을 줄여 리스크 우려를 완화할 수 있지만, 동시에 관리자가 실무자의 노력을 저평가하게 만들어 실무자의 GPT 사용 인센티브를 낮추는 상충 관계(Trade-off)가 있음을 실험적으로 입증하였다.
3. **경험 많은 관리자의 역할 발견**: 관리자의 숙련도(Experience)가 GPT 기반 결과물의 가치를 더 정확하게 인식하게 하며, 이것이 인센티브 정렬을 이끄는 핵심 변수임을 제시하였다.

## 📎 Related Works

논문은 다음과 같은 기존 연구들을 검토하고 차별점을 제시한다.

- **기술 수용 모델(TAM) 및 알고리즘 혐오**: 기존 문헌은 지각된 유용성(Perceived Usefulness)과 리스크가 기술 채택을 결정한다고 설명하며, 특히 주관적인 작업이나 숙련된 전문가일수록 알고리즘 도입에 저항하는 경향이 있다고 언급한다.
- **AI 투명성 연구**: 알고리즘 투명성(Transparency)과 공개(Disclosure)가 신뢰와 책임감을 높여 도입을 촉진한다는 연구와, 반대로 정보 과부하로 인해 신뢰를 낮춘다는 연구가 대립하고 있다.
- **기존 연구의 한계**: 대다수의 GPT 도입 연구는 개인 수준의 분석에 그쳤으며, 기업 내의 위계 구조(Corporate Hierarchies)와 그로 인한 인센티브 구조의 변화를 간과하였다. 본 연구는 이를 보완하기 위해 대리인 이론을 도입하여 조직 내 역학 관계를 분석한다.

## 🛠️ Methodology

### 실험 설계 (Survey Experiment)

본 연구는 실제 컨설팅 펌의 중급 관리자 92명을 대상으로 설문 실험을 진행하였다. 실험은 관리자가 주니어 분석가가 작성한 제안서 초안(Research Brief)을 평가하는 시나리오로 구성되었다.

- **콘텐츠 생성**: 두 명의 분석가가 LLM 없이 작성한 'No-GPT deck'과 ChatGPT 3.5를 사용하여 수정한 'Human-GPT deck'을 각각 준비하였다.
- **처치 변수(Treatment)**: 관리자 집단을 두 그룹으로 나누어, 한 그룹에는 콘텐츠의 생성 출처를 알리는 '공개(Disclosure)' 조건을, 다른 그룹에는 알리지 않는 '비공개(No-Disclosure)' 조건을 부여하였다.
- **측정 항목**: 콘텐츠 품질(Deck Quality), 투입된 추정 시간(Perceived Effort), GPT 사용 허용 의사(SecGPTAuth) 등을 측정하였다.

### 분석 모델 및 방정식

본 연구는 OLS(Ordinary Least Squares) 회귀 분석을 사용하여 인센티브 변화를 측정하였다.

**1. 실무자의 GPT 사용 인센티브 분석 (Equation 1)**
관리자가 인식하는 품질($Y$)과 노력을 분석하는 모델이다.
$$Y_{i,c,v} = \beta_0 + \beta_1 \times \text{Human-GPT}_{c,v} + \beta_2 \times \text{Human-GPT}_{c,v} \times \text{Disclosure}_i + \eta_c + \gamma_i + \epsilon_{i,c,v}$$

- $\beta_1$: 비공개 상태에서 GPT 사용 시 인식되는 품질/노력의 변화량이다.
- $\beta_2$: 공개 정책이 적용되었을 때, GPT 사용에 따른 평가 변화가 어떻게 수정되는지를 나타낸다.

**2. 관리자의 GPT 도입 의사 결정 분석 (Equation 2)**
관리자가 특정 섹션($s$)에 대해 GPT 사용을 승인할 확률($SecGPTAuth$)을 분석하는 모델이다.
$$\text{SecGPTAuth}_{i,s} = \gamma_0 + \gamma_1 \times \text{IndGPTRisk}_i + \gamma_2 \times \text{IndGPTUsefulness}_i + \gamma_3 \times \text{Disclosure}_i + \gamma_4 \times \text{IndGPTRisk}_i \times \text{Disclosure}_i + \gamma_5 \times \text{IndGPTUsefulness}_i \times \text{Disclosure}_i + \gamma_6 \times X_i + \gamma_s + \epsilon_{i,s}$$

- $\text{IndGPTRisk}$와 $\text{IndGPTUsefulness}$는 각각 지각된 리스크와 유용성을 의미하며, 이들이 공개 정책($\text{Disclosure}$)과 상호작용하여 도입 의사에 어떤 영향을 주는지 분석한다.

## 📊 Results

### 1. 정보 비대칭성 (Information Asymmetry)

- **비공개 조건**: 관리자들은 출처가 공개되지 않았을 때 Human-GPT deck과 No-GPT deck을 구분하지 못하였다. 이는 GPT 생성물이 인간의 텍스트와 매우 유사하여 정보 비대칭이 강하게 발생함을 의미한다.
- **공개 조건**: 공개 정책이 시행되면 정보 비대칭이 유의미하게 감소하였다.

### 2. 인센티브 정렬 (Incentive Alignment)

- **비공개 시 (Non-Disclosure)**: 실무자는 GPT를 사용했을 때 더 높은 품질 평가를 받았으나($\beta_1 > 0$), 관리자는 리스크 우려로 인해 전반적으로 GPT 도입을 꺼리는 경향을 보였다. 즉, 실무자는 '그림자 도입'을 통해 이득을 얻지만, 관리자와의 인센티브는 정렬되지 않은 상태이다.
- **공개 시 (Disclosure)**:
  - **품질 평가**: GPT 사용을 공개했을 때, 비공개 시 얻었던 품질 평가의 이점이 사라졌다.
  - **노력 저평가**: 특히 관리자들은 GPT 사용이 공개된 경우, 실무자가 투입한 노력을 유의미하게 낮게 평가하였다($\beta_2 < 0$).
  - **리스크 완화**: 다만, 공개 정책은 리스크에 민감한 관리자들이 GPT 도입을 허용하게 만드는 긍정적인 효과($\gamma_4 > 0$)가 있었다.

### 3. 관리자 숙련도의 영향 (Subsample Analysis)

- **경험 많은 관리자**: 숙련도가 높은 관리자들은 GPT를 사용한 결과물의 품질 향상을 더 잘 인식하였으며, 공개 조건에서도 실무자의 기여도를 덜 폄하하는 경향을 보였다. 이는 경험이 풍부한 관리자가 존재할 때 인센티브 정렬이 더 잘 이루어짐을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 발견

본 연구는 단순한 '투명성' 제고가 반드시 AI 도입을 촉진하지 않는다는 점을 밝혀냈다. 공개 정책은 관리자의 리스크 우려를 줄여주지만, 동시에 실무자의 성과와 노력을 깎아내리는 '인센티브의 역설'을 초래한다. 이는 전문 서비스 기업에서 '노력'이 곧 '가치'로 환산되는 보상 구조 때문에 발생하는 현상이다.

### 한계 및 비판적 해석

- **실험적 설정**: 본 연구는 가상의 RFP 시나리오와 설문 기반의 실험이므로, 실제 장기적인 고용 관계나 복잡한 보상 체계가 작동하는 현실의 역동성을 완전히 반영하지 못했을 가능성이 있다.
- **GPT 버전의 제한**: ChatGPT 3.5를 사용하였으나, 최신 모델(GPT-4o 등)의 경우 인간과의 구분이 더 어려워지거나 혹은 더 명확해질 수 있어 결과가 달라질 수 있다.

### 논의 사항

결국 공개 정책만으로는 부족하며, GPT 도입으로 인해 절약된 시간이 어떻게 재분배되어야 하는지, 그리고 AI를 활용한 성과를 어떻게 공정하게 측정할 것인지에 대한 새로운 '보상 체계(Salary Schemes)'와 '리스크 분담 프레임워크'가 병행되어야 함을 시사한다.

## 📌 TL;DR

본 논문은 컨설팅 펌 내에서 GPT 사용을 둘러싼 관리자(주체)와 실무자(대리인) 간의 갈등을 대리인 이론으로 분석하였다. 실험 결과, **공개 정책(Disclosure Policy)은 정보 비대칭과 리스크 우려를 줄여주지만, 역설적으로 관리자가 실무자의 노력을 저평가하게 만들어 실무자의 도입 의지를 꺾는 부작용**을 낳는다. 이를 해결하기 위해 단순한 공개 의무화를 넘어 **$\text{披露} \rightarrow \text{리스크 분담} \rightarrow \text{모니터링} \rightarrow \text{성과 보상}$으로 이어지는 종합적인 기업 AI 정책(Corporate AI Policy)의 수립**이 필수적임을 강조한다.
