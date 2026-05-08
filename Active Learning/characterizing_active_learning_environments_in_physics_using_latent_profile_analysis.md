# Characterizing active learning environments in physics using latent profile analysis

Kelley Commeford, Eric Brewe, and Adrienne Traxler (2021)

## 🧩 Problem to Solve

물리학 교육에서 '능동적 학습(Active Learning)'은 기존의 수동적 강의 방식보다 학습 효과가 높다는 수많은 연구 결과가 있다. 그러나 '능동적 학습'이라는 용어는 매우 광범위한 우산 용어(umbrella term)로 사용되고 있으며, 그 내부에는 서로 다른 이념적 토대와 기계적 차이를 가진 다양한 교수법들이 존재한다.

현재 대부분의 연구는 단순히 '능동적 학습'과 '수동적 강의'를 비교하는 수준에 머물러 있다. 이러한 접근 방식으로는 서로 다른 능동적 학습 스타일들이 구체적으로 어떤 메커니즘을 통해 차별화되는지, 그리고 어떤 특정 요소가 실제 학습 이득을 이끌어내는지 정확히 짚어내기 어렵다. 따라서 본 논문의 목표는 물리학의 다양한 능동적 학습 교수법들을 구분 짓는 고유한 특성을 정량적으로 분석하고, 이를 분류할 수 있는 체계를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 COPUS(Classroom Observation Protocol for Undergraduate STEM) 관찰 데이터에 **잠재 프로파일 분석(Latent Profile Analysis, LPA)**을 적용하여, 명시적인 레이블 없이도 관찰된 행동 패턴만으로 서로 다른 물리학 교수법들을 분류할 수 있는지 검증하는 것이다. 즉, 수업 시간 동안 교수자와 학생이 수행하는 활동의 비율(Profile)을 변수로 사용하여, 데이터 내에 숨겨진 잠재적 그룹(Latent Profiles)을 찾아내고 이것이 실제 적용된 교수법과 일치하는지 확인하는 '개념 증명(Proof of Concept)' 연구를 수행하였다.

## 📎 Related Works

본 논문은 Stains et al. [7]의 연구를 기반으로 한다. Stains et al.은 다양한 STEM 분야의 COPUS 데이터를 LPA를 통해 분석하여 수업 환경을 '교조적(Didactic, 주로 강의)', '상호작용적 강의(Interactive lecture, 하이브리드)', '학생 중심(Student centered, 능동적)'의 세 가지 범주로 분류하였다.

본 연구는 다음과 같은 점에서 기존 연구와 차별화된다:

1. **분석 범위의 구체화**: 단순히 '능동적'인지 '아닌지'를 구분하는 것을 넘어, 능동적 학습 범주 내부의 세부적인 교수법 간 차이를 분석하는 데 집중한다.
2. **샘플링 방식의 차이**: 무작위 표본 추출이 아닌, 특정 능동적 학습 교수법이 고충실도(High-fidelity)로 구현된 사이트를 의도적으로 선택하는 목적적 샘플링(Purposeful sampling)을 사용하였다.
3. **대상 한정**: 물리학 교육 환경으로 범위를 제한하여 더 정밀한 분류 체계를 시도하였다.

## 🛠️ Methodology

### 1. 데이터 수집 및 COPUS 프로파일 생성

연구진은 6가지 주요 물리학 교수법(Tutorials in Introductory Physics, ISLE, Modeling Instruction, Peer Instruction, Context-Rich Problems, SCALE-UP)이 적용된 환경에서 COPUS 관찰을 수행하였다. COPUS는 2분 간격으로 교수자의 활동(12개 코드)과 학생의 활동(13개 코드)이 발생했는지 여부를 기록하는 프로토콜이다.

본 연구에서는 '바 차트(Bar chart)' 방식을 사용하여 COPUS 프로파일을 생성하였다. 각 코드의 발생 횟수를 합산한 후, 전체 관찰 간격 수로 나누어 해당 활동이 수업 시간에서 차지하는 비율을 계산한다.
$$ \text{COPUS Profile Value} = \frac{\text{Total marks for a specific code}}{\text{Total number of intervals}} $$
이는 특정 활동이 수업 시간의 몇 퍼센트 동안 지속되었는지를 나타내며, 단순 발생 빈도보다 수업 시간의 활용 방식을 더 명확하게 보여준다.

### 2. 잠재 프로파일 분석 (Latent Profile Analysis, LPA)

LPA는 가우시안 혼합 모델(Gaussian Mixture Modeling)을 사용하여 관찰된 변수들을 바탕으로 데이터 내에 숨겨진 그룹을 찾아내는 방법이다.

- **작동 원리**: LPA는 관찰된 데이터 분포가 여러 개의 서로 다른 가우시안 분포(프로파일)의 조합으로 설명될 수 있다고 가정한다. 초기 가우시안 분포를 설정한 후, 각 데이터 포인트가 특정 프로파일에 속할 확률을 계산하고, 기대값이 최대화될 때까지 반복적으로 업데이트하여 수렴시킨다.
- **모델 설정**: 본 연구에서는 `TidyLPA` R 패키지를 사용하였으며, 데이터셋의 크기가 상대적으로 작아 모델의 수렴 가능성을 고려하여 **Model 1(프로파일 간 동일 분산, 공분산 0)**을 선택하였다.
  - **동일 분산(Equal Variance)**: 모든 프로파일의 가우시안 분포 너비가 같다고 가정한다.
  - **공분산 0(Zero Covariance)**: 입력 변수(COPUS 코드들) 간의 상호 의존성이 없다고 가정한다.

### 3. 분석 절차

2개에서 8개 사이의 프로파일 수에 대해 모델을 실행하였으며, 모델의 적합도는 **BIC(Bayesian Information Criterion)**를 통해 평가하였다. BIC 값이 낮을수록 모델의 적합도가 높다고 판단한다.

## 📊 Results

### 1. 정량적 결과 및 모델 선택

BIC 분석 결과, 프로파일 수가 2개에서 3개로 갈 때는 약간 증가했다가, 5개까지 급격히 감소하는 양상을 보였다. 5, 6, 7개 프로파일 모델은 BIC 값이 거의 비슷하게 나타났으나, 가장 적은 수의 프로파일로 효율적인 분류가 가능했던 **5-프로파일 솔루션**을 주요 결과로 채택하였다.

### 2. 프로파일 분류 결과

- **2-프로파일 솔루션**: 수업을 크게 '상호작용적 강의 스타일(Interactive lecture-like)'과 '그 외(Other)'로 성공적으로 분리하였다.
  - 교수자의 강의($I.Lec$)와 학생의 청취($S.L$) 활동이 두 그룹을 가르는 가장 큰 차이점이었다.
- **5-프로파일 솔루션**: 다음과 같이 5가지 범주로 세분화하여 분류하였다.
  - **Profile 1 (Interactive lecture-like)**: Peer Instruction, SCALE-UP, 일부 강의 구성 요소. 클릭커 질문($I.CQ$)과 후속 토론($I.Fup$) 비중이 높았다.
  - **Profile 2 (Modeling Instruction)**: Modeling Instruction 전용. 학생 발표($S.SP$)와 전체 학급 토론($S.WC$)이 유일하게 나타난 그룹이다.
  - **Profile 3 (ISLE labs)**: ISLE 실험 수업.
  - **Profile 4 (CRP labs)**: Context-Rich Problems 실험 수업. 학생들의 예측 활동($S.Prd$)이 ISLE 실험과 구분되는 핵심 특징이었다.
  - **Profile 5 (Recitation-like)**: Tutorials, ISLE recitation 등. 그룹 워크시트 활동($S.WG$)의 비중이 압도적으로 높았다.

## 🧠 Insights & Discussion

### 1. 유사 교수법 간의 차별점 발견

본 연구는 외견상 유사해 보이는 교수법들의 내부적 차이를 드러냈다.

- **Modeling Instruction vs SCALE-UP**: 두 방법 모두 스튜디오 형식의 통합 환경을 제공하지만, SCALE-UP은 여전히 강의가 주요 정보 전달 수단인 반면, Modeling Instruction은 학생들의 탐구와 전체 토론을 통해 정보를 습득한다는 점이 COPUS 프로파일(강의 비중 및 학생 발표 유무)을 통해 명확히 구분되었다.
- **실험(Lab) vs 연습(Recitation)**: 두 활동 모두 소그룹 활동이 많아 혼동될 수 있으나, 연습 세션은 워크시트 기반 활동($S.WG$)이 주를 이루는 반면, 실험 세션은 '기타 그룹 활동'($S.OG$)으로 분류되어 구분되었다.

### 2. 연구의 한계 및 비판적 해석

- **데이터셋의 규모**: 데이터셋이 작아 공분산(Covariance) 모델을 적용하지 못하고 Zero Covariance 모델을 사용해야 했다. 변수 간의 상관관계(예: 강의와 판서의 동시 발생)를 분석하지 못한 점은 한계이다.
- **관찰의 빈도**: 단기간의 방문 관찰로 인해 일일 변동성(Day-to-day fluctuations)을 완전히 배제할 수 없다. 저자들은 향후 연구에서 여러 날의 관찰 데이터를 합친 '통합 COPUS 프로파일'을 사용할 것을 제안한다.
- **관찰 도구의 해상도**: COPUS가 실험과 연습 세션의 세부 활동 차이를 정밀하게 구분하지 못하는 경향이 있으므로, LOPUS(실험 전용 프로토콜)와의 병행 사용이 필요해 보인다.

## 📌 TL;DR

본 논문은 물리학의 다양한 능동적 학습 교수법들이 실제로 어떻게 다르게 운영되는지 정량적으로 분석하기 위해, **COPUS 관찰 데이터에 잠재 프로파일 분석(LPA)을 적용**한 연구이다. 분석 결과, 단순히 '능동적'이라는 분류를 넘어 **강의형-모델링-ISLE실험-CRP실험-연습형**의 5가지 고유한 행동 프로파일로 교수법들을 구분할 수 있음을 증명하였다. 이는 향후 특정 능동적 학습 메커니즘이 학습 성과에 미치는 영향을 정밀하게 분석할 수 있는 방법론적 토대를 제공한다.
