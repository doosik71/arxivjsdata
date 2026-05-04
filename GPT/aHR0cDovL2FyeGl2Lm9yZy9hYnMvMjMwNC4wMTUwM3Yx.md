# GPT-4 to GPT-3.5: 'Hold My Scalpel' – A Look at the Competency of OpenAI’s GPT on the Plastic Surgery In-Service Training Exam

Jonathan D. Freedman MD, PhD, Ian A. Nappier AB (2023)

## 🧩 Problem to Solve

본 연구는 OpenAI의 대규모 언어 모델(LLM)인 GPT-3.5와 GPT-4가 성형외과 전문의 수련 과정에서 매우 중요한 지표로 활용되는 PSITE(Plastic Surgery In-Service Training Exam) 시험에서 어느 정도의 역량을 발휘하는지 평가하고자 한다.

PSITE는 성형외과 레지던트의 숙련도를 측정하는 핵심 벤치마크이며, 이 시험의 점수는 전문의 자격 취득을 위한 필기 시험 합격 여부와 높은 상관관계를 가진다. 특히, 기존의 많은 AI 평가 연구들이 시뮬레이션된 테스트나 연습 문제만을 사용한 것과 달리, 본 연구는 실제 임상 사례(clinical vignettes)가 포함된 실제 PSITE 기출 문제를 사용하여 모델의 실질적인 전문 지식 수준을 측정하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 실제 성형외과 전문의 시험인 PSITE를 통해 GPT-3.5와 GPT-4의 성능 차이를 정량적으로 분석하고, GPT-4가 전문 의료 분야에서 비약적인 성능 향상을 이루었음을 증명한 것이다. 특히 텍스트 기반의 입력만으로도 GPT-4가 숙련된 레지던트 수준의 성적을 거둘 수 있음을 보여주었으며, 향후 시각적 정보가 포함된 multimodal 입력이 가능해질 경우 '초인적인(superhuman)' 성능을 낼 가능성을 제시하였다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구 및 방법론을 참고하였다.

- **GPT-4 Technical Report**: OpenAI가 공개한 기술 보고서의 평가 방법론을 벤치마킹하여 MCQ(Multiple Choice Question) 시험 평가 절차를 설계하였다.
- **USMLE 평가 연구**: 의료 면허 시험(USMLE)에 대한 GPT 성능 평가 방식과 유사한 접근법을 채택하였다.
- **MELD (Memorization effects Levenshtein detector)**: Nori et al. (2023)이 제안한 방법론으로, LLM이 훈련 데이터셋에 포함된 문제를 단순히 암기하여 답변하는 '데이터 누수(Data Leakage)' 현상을 탐지하기 위해 사용되었다.

## 🛠️ Methodology

### 1. 시험 문제 분류

2021년과 2022년 PSITE 문제를 다음의 5가지 유형으로 분류하였다.

- **Text Only**: 문제와 정답 모두 텍스트로만 구성된 경우.
- **Supplemental Figure**: 텍스트 외에 추가 이미지나 도표가 포함된 경우.
- **Table Answers**: 정답이 표 형태로 제시된 경우.
- **Embedded Answers**: 정답이 이미지나 도표 내에 직접 포함된 경우.
- **Stem Table**: 문제의 줄기(stem) 부분에 표가 포함된 경우.

### 2. 평가 파이프라인 (GPT-3.5)

OpenAI의 평가 방법론을 복제하여 Python 코드를 통해 다음과 같은 절차로 진행하였다.

- **Few-shot Learning**: 모델에게 5개의 PSITE 예시 문제와 정답, 공식 설명을 먼저 제공하여 문맥을 학습시킨다.
- **2단계 프롬프트 구조**:
    1. 첫 번째 요청: 문제와 선택지를 제공하고, 먼저 정답에 대한 **설명(explanation)**을 작성하도록 유도한다.
    2. 두 번째 요청: 작성된 설명을 바탕으로 최종적인 **정답 알파벳(예: "A", "B")**만을 추출한다.
- **파라미터 설정**: $t=0.3, n=1, \text{max\_tokens}=512$ (설명 생성 시) 및 $t=0.0, n=1$ (정답 추출 시) 설정을 통해 결과의 일관성을 유지하였다.

### 3. GPT-4 평가 방식

연구 당시 GPT-4 API에 대한 접근이 제한적이었으므로, ChatGPT 인터페이스를 통해 평가를 진행하였다. GPT-3.5가 틀린 문제들을 순차적으로 입력하여 GPT-4의 정답 여부를 확인하는 방식으로 성능 향상분을 추정하였다.

### 4. 데이터 검증 (MELD Method)

모델이 문제를 암기했는지 확인하기 위해 MELD 방법을 사용하였다. 이는 문제 텍스트를 두 부분으로 나누고, 앞부분만 입력했을 때 모델이 뒷부분을 정확히 생성하는지 확인하는 방식이다. Levenshtein distance ratio 임계값을 $0.95$로 설정하여 일치 여부를 판단하였다.

## 📊 Results

### 1. 문제 구성 분포

- **2022년 시험**: Text Only(81.2%), Supplemental Figure(14.0%) 순으로 구성되었다.
- **2021년 시험**: Text Only(84.8%), Supplemental Figure(10.0%) 순으로 구성되었다.

### 2. 모델 성능 비교 (정량적 결과)

GPT-4는 모든 섹션에서 GPT-3.5 대비 압도적인 성능 향상을 보였다.

- **2022 PSITE**:
  - 정답률: $54\% \rightarrow 75\%$로 상승.
  - 백분위 점수(Percentile): **8th $\rightarrow$ 88th percentile**로 급격히 상승.
- **2021 PSITE**:
  - 정답률: $51\% \rightarrow 78\%$로 상승.
  - 백분위 점수(Percentile): **3rd $\rightarrow$ 97th percentile**로 상승.

### 3. 섹션별 성능 (2022년 기준)

- **Core Principles**: $57\% \rightarrow 82\%$ (가장 높은 정답률)
- **Hand and Lower Extremity**: $56\% \rightarrow 80\%$
- **Comprehensive**: $55\% \rightarrow 76\%$
- **Craniomaxillofacial**: $59\% \rightarrow 71\%$
- **Breast and Cosmetic**: $43\% \rightarrow 68\%$ (가장 낮은 정답률이나 향상폭은 큼)

### 4. 데이터 누수 검증

MELD 분석 결과, 평가 대상인 500개의 문제 중 임계값 $0.95$를 넘는 문제가 $0\%$였다. 이는 모델이 시험 문제를 암기한 것이 아니라 실제 추론을 통해 답을 냈음을 시사한다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석

GPT-4는 텍스트 전용 모드만으로도 전체 레지던트 중 상위 $12\% \sim 3\%$에 해당하는 성적을 거두었다. 이는 단순 지식 검색을 넘어 임상 시나리오를 분석하고 최적의 수술적 접근법이나 약물을 선택하는 능력이 전문가 수준에 도달했음을 의미한다.

### 2. 한계 및 가정

- **시각 정보의 부재**: PSITE 문제의 약 $13\% \sim 16\%$가 시각적 요소(이미지, 도표)를 포함하고 있으나, 본 실험에서는 이를 텍스트로만 처리하거나 제외하였다. GPT-4의 multimodal 능력을 활용한다면 성적이 더 상승할 가능성이 매우 높다.
- **평가 방식의 차이**: GPT-3.5는 API를 통해 표준화된 파이프라인으로 평가했으나, GPT-4는 ChatGPT 인터페이스를 통해 일부 문제만 평가했다는 방법론적 차이가 존재한다.

### 3. 비판적 논의

본 연구는 GPT-4가 매우 높은 점수를 기록했음을 보여주지만, 실제 수술실에서의 술기나 환자와의 상호작용 같은 실무적 역량까지 대체할 수 있는지는 알 수 없다. 다만, 지식 기반의 의사결정 보조 도구로서의 가능성은 충분히 입증되었다.

## 📌 TL;DR

본 연구는 실제 성형외과 전문의 시험(PSITE)을 통해 GPT-3.5와 GPT-4의 역량을 비교 분석하였다. GPT-3.5는 하위권의 성적을 기록한 반면, **GPT-4는 상위 $12\% \sim 3\%$에 해당하는 놀라운 성능 향상**을 보이며 전문의 수준의 지식 능력을 입증하였다. 데이터 누수 검증을 통해 이 결과가 암기가 아닌 추론의 결과임을 확인하였으며, 향후 multimodal 기능이 통합된다면 의료 분야에서 초인적인 성능을 발휘할 가능성이 크다.
