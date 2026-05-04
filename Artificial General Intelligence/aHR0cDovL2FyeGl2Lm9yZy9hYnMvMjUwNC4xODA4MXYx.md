# Hype and Adoption of Generative Artificial Intelligence Applications

Vinh Truong (2024)

## 🧩 Problem to Solve

본 연구는 생성형 인공지능(Generative AI, 이하 GAI)의 등장에 따른 사회적 수용 과정과 이에 대한 대중의 심리적, 정서적 반응을 분석하고자 한다. GAI는 단순히 정보를 검색하는 도구를 넘어 새로운 콘텐츠를 생성하는 능력을 갖추고 있어, 기존의 기술 도입과는 다른 양상을 보이며 특히 일자리 대체에 대한 불안감과 같은 복잡한 정서적 반응을 유발한다.

기존 연구들은 특정 시점의 대중 정서를 단편적으로 보여주는 스냅샷(snapshot) 방식의 분석에 그쳤거나, 이론적 배경 없이 가공되지 않은 데이터만을 제시하는 한계가 있었다. 또한, 정보 탐색자의 관점에서 기술 수용을 분석했을 뿐, 정보를 직접 생성하는 '콘텐츠 제작자'의 관점에서의 분석은 부족했다. 따라서 본 논문의 목표는 GAI의 수용 과정이 이론적인 기술 하이프 사이클(Hype Cycle)과 정서적 변화 곡선(Change Curve)을 따르는지 실증적으로 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GAI의 수용 과정이 이성적 기대치와 정서적 반응이라는 두 가지 경로로 동시에 진행된다는 '이중 단계 프로세스(dual-stage process)' 가설을 제시한 것이다.

1. **이론적 통합**: 기술의 성숙도와 대중의 기대치를 다루는 `Gartner Hype Cycle`과, 변화에 따른 인간의 정서적 단계 변화를 다루는 `Kübler-Ross Change Curve`를 결합하여 GAI 수용 과정을 분석하는 프레임워크를 구축하였다.
2. **종단적 분석(Longitudinal Approach)**: 단일 시점의 분석이 아닌, 시간의 흐름에 따른 정서 및 감정의 변화를 추적함으로써 기술 수용의 동적인 진화 과정을 포착하였다.
3. **세밀한 감정 분석**: 단순한 긍정/부정 분류를 넘어 28가지의 세분화된 감정 카테고리를 사용하여 대중의 심리 상태를 정밀하게 분석하였다.

## 📎 Related Works

논문에서는 기술 수용 및 변화 관리를 설명하는 여러 모델을 소개하며 기존 연구와의 차별점을 제시한다.

- **S-curve**: 기술 발전이 초기에는 느리게, 이후 급격하게 진행되다가 포화 상태에 이르는 선형적 성숙 과정을 설명하지만, 사용자의 심리적 반응을 반영하지 못한다.
- **Gartner Hype Cycle**: 기술 트리거(Technology Trigger)부터 생산성 안정기(Plateau of Productivity)까지 5단계를 통해 기대치와 실망감이 교차하는 과정을 설명한다. 본 논문은 이를 GAI에 적용하여 실증적으로 검증하고자 한다.
- **변화 관리 모델**: Tuckman의 집단 발달 모델, Bridges의 전환 모델, Satir 변화 모델 등이 언급되지만, 이들은 주로 구조적 단계나 인지적 전환에 집중하며 정서적 기반을 간과하는 경향이 있다.
- **Kübler-Ross Change Curve**: 원래 말기 환자의 슬픔의 단계를 설명하기 위해 설계되었으나, 본 연구에서는 이를 기술 도입에 따른 정서적 저항과 수용 과정(부정 $\rightarrow$ 분노 $\rightarrow$ 타협 $\rightarrow$ 우울 $\rightarrow$ 수용)을 분석하는 도구로 활용한다.

## 🛠️ Methodology

본 연구는 연역적 연구 방법을 채택하여 가설을 세우고, 소셜 미디어 데이터를 통해 이를 검증하는 파이프라인을 구축하였다.

### 1. 데이터 수집 및 전처리

- **데이터 소스**: X(구 Twitter)의 Public API를 사용하여 "ChatGPT", "Bing AI", "Microsoft Office Copilot"과 같은 키워드가 포함된 트윗을 수집하였다.
- **전처리**: 중복 데이터, 무관한 해시태그, 빈 필드를 제거하여 데이터의 품질을 확보하였으며, 각 트윗의 타임스탬프를 활용해 시계열 분석이 가능하게 하였다.

### 2. 정서 분석 (Sentiment Analysis)

- **사용 도구**: `VandeSentiment` (VADER)
- **분석 방법**: 텍스트를 단순히 긍정/부정으로 나누지 않고, $-1$ (매우 부정적)에서 $+1$ (매우 긍정적) 사이의 연속적인 수치인 **Compound Score**를 산출하였다. 이 점수를 시간축으로 플로팅하여 Gartner Hype Cycle의 단계와 비교하였다.

### 3. 감정 분석 (Emotion Analysis)

- **사용 도구**: `EmoRoBERTa` (RoBERTa 및 BERT 기반의 모델)
- **분석 방법**: 텍스트를 28가지의 세분화된 감정 카테고리로 분류하였다.
- **절차**:
  - 일별 각 감정 카테고리별 총점(Total emotion scores)을 계산한다.
  - 이를 시계열 곡선으로 시각화하여 Kübler-Ross Change Curve의 7단계(Shock $\rightarrow$ Denial $\rightarrow$ Frustration $\rightarrow$ Depression $\rightarrow$ Experimentation $\rightarrow$ Decision $\rightarrow$ Integration)와 매핑한다.

## 📊 Results

### 1. 정서 분석 결과 (Sentiment Analysis)

ChatGPT 공개 후 100일간의 정서 점수를 분석한 결과는 다음과 같다.

- **기대치의 정점(Peak of Inflated Expectations)**: 초기 30일 동안 점수가 꾸준히 상승하여 약 $0.37$로 정점에 도달하였다. 이는 새로운 기술에 대한 초기 낙관주의를 반영한다.
- **환멸의 계곡(Trough of Disillusionment)**: 정점 이후 약 한 달 뒤, 점수가 $0.24$까지 하락하였다. 이는 사용자들이 실제 사용 과정에서 기술적 한계나 윤리적 문제에 직면하며 기대치가 조정되는 단계이다.
- **계몽의 언덕 및 생산성 안정기(Slope of Enlightenment & Plateau of Productivity)**: 이후 점수가 다시 완만하게 상승하여 100일 시점에는 약 $0.27$에서 안정화되었다. 이는 기술의 가치를 현실적으로 인식하고 수용하는 단계에 진입했음을 시사한다.

### 2. 감정 분석 결과 (Emotional Analysis)

28가지 감정의 추이를 분석하여 Kübler-Ross 곡선과의 일치성을 확인하였다.

- **Shock 단계**: Surprise, Joy, Admiration, Pride가 초기에 지배적이었으며, 특히 Surprise는 가장 가파른 하락 곡선을 보였다.
- **Denial 단계**: Disgust, Fear, Relief 등이 나타났으며, 특히 Disgust(혐오)와 Fear(공포)의 상승은 윤리적 우려와 일자리 불안감을 반영한다.
- **Frustration 단계**: 2개월 차에 Annoyance, Anger, Confusion 등이 나타나며 기술적 한계와 씨름하는 모습이 관찰되었다.
- **Sadness 단계**: 3개월 차에 Sadness, Disappointment, Grief 등이 정점을 찍으며 정서적 저점이 형성되었다.
- **Decision 및 Experiment 단계**: Gratitude, Excitement, Realization, Curiosity 등이 상승하며 기술을 도구로서 수용하려는 시도가 나타났다.
- **Integration 단계**: Neutrality(중립)가 가장 가파른 상승 곡선을 보이며, GAI가 디지털 환경의 일상적인 부분으로 통합되었음을 보여준다.

## 🧠 Insights & Discussion

본 연구는 GAI 수용 과정에서 **인지적 판단(Sentiment)**과 **정서적 반응(Emotion)**이 서로 다른 궤적을 그리면서도 상호 보완적으로 작용함을 밝혀냈다.

- **인지와 정서의 괴리**: 대중의 정서 점수(Sentiment)는 전반적으로 양수(+)를 유지하며 낙관적이었으나, 세부 감정 분석(Emotion)에서는 공포, 혐오, 우울과 같은 강한 부정적 정서가 순차적으로 나타났다. 이는 초기 낙관주의가 내면의 충격과 불안감을 가리고 있었음을 의미한다.
- **비선형적 적응**: 기술 수용은 단순히 기능적 유용성을 깨닫는 선형적 과정이 아니라, 정서적 저항과 회복을 거치는 복잡한 심리적 과정이다.
- **실무적 시사점**: 기업과 정책 입안자들은 기술의 기능적 도입뿐만 아니라, 사용자가 겪는 정서적 단계(특히 환멸의 계곡과 우울 단계)를 이해하고 이를 완화할 수 있는 공감 기반의 전략(예: 교육 리소스 제공, 심리적 지지)을 세워야 한다.

**한계점**:

- 특정 산업군(의료, 예술 등)의 맥락을 고려하지 않고 전반적인 소셜 미디어 데이터를 분석했다는 점이 한계로 지적된다. 산업별로 GAI가 주는 충격과 수용 속도가 다를 수 있으므로 이에 대한 세부 연구가 필요하다.

## 📌 TL;DR

본 논문은 GAI의 수용 과정이 **Gartner Hype Cycle(이성적 기대)**과 **Kübler-Ross Change Curve(정서적 적응)**라는 두 가지 모델을 동시에 따른다는 가설을 소셜 미디어 데이터 분석을 통해 실증적으로 검증하였다. 분석 결과, 대중은 '낙관 $\rightarrow$ 환멸 $\rightarrow$ 안정'의 정서 흐름과 '충격 $\rightarrow$ 부정 $\rightarrow$ 분노 $\rightarrow$ 우울 $\rightarrow$ 수용'의 감정 흐름을 거쳐 GAI를 통합하는 과정을 보였다. 이 연구는 향후 기업이 GAI를 도입할 때 기술적 통합뿐만 아니라 사용자의 정서적 적응을 관리하는 것이 필수적임을 시사한다.
