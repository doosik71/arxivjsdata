# Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on Keyword Spotting

Clara Rosina Fernandez, Severine Guillaume, Guillaume Wisniewski (2025)

## 🧩 Problem to Solve

본 논문은 wav2vec2 및 HuBERT와 같은 사전 학습된(pretrained) 음성 표현(speech representations)에서 나타나는 **Anisotropy(비등방성)** 현상이 실제 다운스트림 태스크의 성능에 어떠한 영향을 미치는지 분석하는 것을 목표로 한다.

Anisotropy란 벡터 공간 내의 표현들이 균일하게 분포하지 않고 좁은 원뿔(narrow cone) 형태의 특정 방향으로 집중되는 현상을 의미한다. 이로 인해 무작위로 선택된 두 임베딩 사이의 코사인 유사도가 매우 높게 나타나는 경향이 있으며, 이는 이론적으로 모델의 표현력(expressiveness)을 제한하고 유용성을 저하시킬 가능성이 있다.

특히 본 연구는 계산 문서 언어학(computational documentary linguistics) 관점에서 **Keyword Spotting(KWS)** 태스크를 사례 연구로 설정하였다. 저자원 언어의 경우 전체 전사(transcription) 데이터가 부족하므로, 전사 없이 음성 표현의 유사도만으로 특정 단어를 찾을 수 있는 능력이 매우 중요하다. 따라서 본 논문의 핵심 목표는 표현 공간의 Anisotropy가 심함에도 불구하고, 이러한 표현들이 실제로 단어 식별 및 검색에 유효하게 사용될 수 있는지 검증하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 표현 공간의 기하학적 특성(Anisotropy)과 실제 태스크 성능 사이의 괴리를 측정하는 것이다. 단순히 임베딩 간의 평균 거리를 측정하는 기존 연구에서 벗어나, **Dynamic Time Warping(DTW)** 알고리즘을 이용한 Keyword Spotting이라는 구체적인 활용 사례를 통해 Anisotropy의 실질적인 영향을 평가하였다.

연구의 핵심 결과는 wav2vec2의 표현이 강한 Anisotropy를 보임에도 불구하고, 음성 신호의 고수준 음성학적 구조(high-level phonetic structures)를 효과적으로 캡처하며, 이는 화자 간의 변이(speaker variation)에 강건한(robust) 특성을 가진다는 점을 밝혀낸 것이다.

## 📎 Related Works

최근 Transformer 기반 모델(BERT, GPT-2, T5 등)의 표현 공간을 분석한 여러 연구에서 Anisotropy 현상이 공통적으로 관찰되었다. 특히 텍스트 표현에서 시작된 이 논의는 최근 음성 표현 영역으로 확장되었으며, Transformer 구조 자체에 내재된 특성이라는 주장이 제기되었다.

기존 연구들은 주로 무작위로 선택된 두 프레임 간의 거리 분포를 측정하는 데 그쳤으며, 이러한 기하학적 특성이 실제 다운스트림 태스크에서 어떤 결과로 이어지는지에 대한 구체적인 분석은 부족했다. 본 논문은 이러한 한계를 극복하기 위해 KWS라는 실질적인 응용 분야를 도입하여 Anisotropy의 실질적인 영향력을 평가했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Anisotropy 측정 및 분석
본 논문은 표현 공간의 Anisotropy를 정량화하기 위해 다음과 같은 측정 지표 $A$를 사용한다.

$$A = E_{i \neq j} (1 - \cos(x_i, x_j))$$

여기서 $x_i, x_j$는 코퍼스에서 무작위로 추출된 두 개의 오디오 표현이며, $E[\cdot]$는 모든 쌍에 대한 기대값을 의미한다. $A$ 값이 1에 가까울수록 표현들이 좁은 영역에 집중된 Anisotropic한 상태이며, 0에 가까울수록 공간 전체에 균일하게 분포된 Isotropic한 상태임을 나타낸다.

또한, 특정 차원이 비정상적으로 큰 크기와 분산을 가져 코사인 유사도 계산을 지배하는 **Rogue dimensions**의 존재를 분석하였다.

### 2. Keyword Spotting 파이프라인
본 연구는 전사 없이 쿼리 오디오와 코퍼스 내 오디오의 유사도를 비교하여 단어를 찾는 시스템을 구축하였다.

- **시스템 구조**: 쿼리 시퀀스와 타겟 레코딩 시퀀스 사이의 최적 정렬을 찾기 위해 **Subsequence Dynamic Time Warping (DTW)** 알고리즘을 사용한다.
- **유사도 측정**: DTW의 기초 유사도 지표로 코사인 유사도(cosine similarity)를 사용한다.
- **비교 대상 표현 (Representations)**:
    1. **MFCC (Baseline)**: 전통적인 음성 특징 추출 방식으로, 13개의 MFCC 계수를 사용한다.
    2. **XLSR-53 Word**: 단어 구간만 추출하여 XLSR-53 모델로 인코딩한 표현이다. (비문맥적)
    3. **XLSR-53 Contextual**: 전체 문장을 인코딩한 후, 해당 단어에 해당하는 벡터만 추출한 표현이다. (문맥적)

### 3. 평가 지표
검색 성능을 측정하기 위해 $\text{Precision@k}$와 $\text{Recall@k}$를 사용한다.
- $\text{Precision@k}$: 상위 $k$개의 검색 결과 중 타겟 단어를 포함한 문장의 비율이다.
- $\text{Recall@k}$: 전체 타겟 문장 중 상위 $k$개 결과 내에 포함된 비율이다.

## 📊 Results

### 1. Anisotropy 검증 결과
XLSR-53 모델을 분석한 결과, Anisotropy 지표 $A$는 0.46으로 측정되어 강한 Anisotropy가 존재함이 확인되었다. 특히 모델의 후반부 레이어로 갈수록 Anisotropy가 심해지며, 특정 레이어(예: 레이어 22, 23)에서는 **Rogue dimensions**의 영향이 극도로 높게 나타났다.

### 2. Keyword Spotting 성능 분석
실험 결과, 표현 공간의 Anisotropy에도 불구하고 XLSR-53의 표현은 매우 효과적으로 단어를 식별하였다.

- **XLSR-53 vs MFCC**: $\text{Precision@k}$가 $k$가 증가함에 따라 감소하는 속도가 XLSR-53이 MFCC보다 훨씬 느렸다. 이는 XLSR-53이 물리적인 음향 특성보다 더 추상적이고 언어적인 정보를 잘 포착함을 의미한다.
- **문맥의 영향**: Contextual Representation이 Word Representation보다 성능이 우수했다. 특히 $\text{Precision@1}$에서 Contextual 표현은 쿼리가 추출된 원본 레코딩을 항상 정확히 찾아냈으나, Word 표현은 그렇지 못했다.
- **레이어별 특성**: 성능은 레이어에 따라 크게 달라졌으며, 특히 레이어 15, 16, 17에서 성능 저하가 관찰되었다. 하지만 전반적으로 깊은 레이어로 갈수록 동일 단어 간의 거리가 짧아지고 분산이 줄어드는 경향을 보였다.

| 지표 | $k$ | MFCC | XLSR-53 | MFCC (Recall) | XLSR-53 (Recall) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Precision | 1 | 100.0% | 100.0% | 5.7% | 5.7% |
| | 10 | 16.8% | 54.7% | 8.8% | 31.2% |
| | 100 | 6.4% | 12.1% | 29.8% | 64.8% |

*(위 표는 논문의 Table 2 내용을 요약한 것이며, XLSR-53은 최적 레이어 기준임)*

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 **"표현 공간이 Anisotropic 하다고 해서 반드시 그 표현이 무용한 것은 아니다"**라는 중요한 통찰을 제공한다. 코사인 유사도의 절대적인 수치들이 모두 높게 나타나 해석이 어려울 수 있지만, 상대적인 거리 차이는 여전히 유의미하며 이를 통해 음성학적 구조를 식별할 수 있음을 입증하였다.

특히 wav2vec2가 단순히 입력 신호를 복원하는 loss로 학습되었음에도 불구하고, 화자의 목소리, 마이크 특성, 주변 소음과 같은 물리적 요인을 배제하고 **추상적인 언어 정보(invariant speech representations)**를 학습했다는 점이 고무적이다.

### 한계 및 논의사항
- **레이어별 성능 편차**: 특정 레이어(15-17)에서 성능이 급격히 떨어지는 이유에 대해 논문 내에서 명확한 설명을 제시하지 못하였다. 이는 향후 연구에서 분석되어야 할 부분이다.
- **Recall의 한계**: $\text{Precision@k}$는 높았으나 $\text{Recall@k}$는 $k=100$에서도 65% 수준에 머물렀다. 이는 동일한 단어라 하더라도 발음의 변이나 환경에 따라 여전히 표현 공간에서 멀리 떨어져 있는 경우가 존재함을 시사한다.

## 📌 TL;DR

본 논문은 wav2vec2와 같은 사전 학습된 음성 모델에서 나타나는 **Anisotropy(표현의 좁은 분포)** 현상이 실제 **Keyword Spotting(단어 검색)** 성능을 저하시키는지 분석하였다. 실험 결과, 표현 공간의 기하학적 불균형에도 불구하고 DTW 기반의 유사도 측정은 매우 효과적이었으며, 특히 MFCC보다 뛰어난 강건함을 보였다. 이는 사전 학습된 모델이 화자 독립적인 고수준 음성학적 특징을 잘 포착하고 있음을 의미하며, 전사가 불가능한 저자원 언어의 문서화 작업에 실질적으로 활용될 가능성이 높음을 시사한다.