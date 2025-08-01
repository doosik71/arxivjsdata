# Helsinki Speech Challenge 2024
Martin Ludvigsen, Elli Karvonen, Markus Juvonen, and Samuli Siltanen

## 🧩 Problem to Solve
이 논문은 헬싱키 음성 챌린지 2024(HSC2024)를 소개하며, 실제 환경에서 녹음된 손상된 음성 데이터를 활용하여 음성 향상(Speech Enhancement) 및 오디오 역권선(Audio Deconvolution) 알고리즘을 개발하는 것을 목표로 합니다.
주요 해결 과제는 다음과 같습니다:
- **실제 데이터 활용:** 음성 처리 분야에서 흔히 사용되는 합성 훈련 데이터에 대한 의존성을 벗어나, 실제 세계의 복잡한 조건(필터링 및 잔향)에 의해 손상된 대규모 실측 데이터를 제공하여 현실적인 음성 향상 기술 개발을 촉진합니다.
- **연구 분야 간 격차 해소:** 응용 수학적 도구를 사용하는 역문제(Inverse Problems) 분야와 최근 기계 학습에 크게 의존하는 음성 향상(Speech Enhancement) 분야 간의 연결을 모색합니다.
- **다운스트림 태스크 유용성 입증:** 음성 향상 기술이 음성 인식(Speech Recognition)과 같은 다운스트림 태스크에 유용함을 시연하고, 역으로 음성 인식 모델을 음성 향상 알고리즘의 성능을 정량화하는 지표로 활용하는 방법을 제안합니다.

## ✨ Key Contributions
- **실제 데이터 기반 챌린지 제안:** 실제 세계의 필터링 및 잔향 효과를 포함하는 독특한 음성 데이터셋을 기반으로 한 음성 향상 챌린지 HSC2024를 시작했습니다.
- **실측 음성 데이터셋 구축:** 스피커와 마이크를 이용한 '필터 실험' 및 '잔향 실험' 설정을 통해 정제된 음성과 손상된 음성 쌍을 포함하는 대규모 실측 데이터셋을 구축하여 공개했습니다. 이 데이터셋은 기존의 합성 데이터셋이 포착하기 어려운 비선형적 효과와 실제 녹음 환경의 아티팩트를 포함합니다.
- **음성 인식을 통한 성능 측정 제안:** Mozilla DeepSpeech 모델과 문자 오류율(CER)을 사용하여 음성 향상 알고리즘의 성능을 객관적으로 정량화하는 혁신적인 접근 방식을 제시하고, 이는 음성 향상의 실용적인 다운스트림 적용 가능성을 보여줍니다.
- **다양한 난이도 수준의 데이터 제공:** 데이터는 필터링, 잔향, 그리고 이 둘이 결합된 총 12가지 난이도 수준으로 나뉘어 제공되어, 참가자들이 점진적으로 어려운 문제에 도전할 수 있도록 했습니다.
- **비선형성 분석 및 강조:** 실제 녹음에서 스피커의 비선형성, 고조파 왜곡, 진동 공명 등 LTI(Linear Time-Invariant) 모델로 설명하기 어려운 비선형 효과가 관찰됨을 입증하여, 실제 오디오 손상이 단순한 선형 모델을 넘어섬을 강조했습니다.

## 📎 Related Works
- **음성 향상 및 역문제:** 일반적인 음성 향상 방법론 및 선형/비선형 역문제 연구(Mueller & Siltanen [1], O’Shaughnessy [2]).
- **데이터 기반 챌린지:** 2021 헬싱키 디블러 챌린지(Helsinki Deblur Challenge) [3]에서 영감을 받아 다운스트림 태스크를 통한 평가 방식 채택.
- **음성 품질 평가 지표:** PESQ(Perceptual Evaluation of Speech Quality) [4] 및 STOI(Short-Time Objective Intelligibility) [5]와 같은 전통적인 음성 품질 지표 언급.
- **음성 인식 모델:** Mozilla DeepSpeech [6]를 성능 평가를 위한 핵심 음성 인식 모델로 채택.
- **음성 데이터셋:** LibriSpeech [7], WSJ0-2mix [8]와 같은 기존의 대규모 음성 데이터셋을 언급하며, 이들이 주로 합성 데이터를 사용한다는 점을 지적하고 본 챌린지의 실측 데이터와의 차별성 강조.
- **텍스트-투-스피치(TTS) 모델:** OpenAI의 tts-1 [9] 모델을 사용하여 챌린지의 깨끗한(clean) 음성 데이터를 생성.

## 🛠️ Methodology
1.  **클린 데이터 생성:**
    *   OpenAI의 tts-1 텍스트-투-스피치 모델을 사용하여 Project Gutenberg의 책에서 발췌한 텍스트로부터 고품질의 노이즈 없는 음성 데이터를 생성했습니다.
    *   생성된 음성은 16kHz로 다운샘플링하고, 오디오 레벨을 정규화하며, 패딩을 추가했습니다.
    *   초기 데이터셋 중 DeepSpeech 모델로 측정했을 때 CER이 가장 나빴던 5%의 샘플을 제거하여 데이터 품질과 DeepSpeech의 적합성을 확보했습니다.
2.  **오염된 데이터 녹음 (실제 환경):**
    *   **녹음 장비:** 실제와 유사한 녹음 환경을 시뮬레이션하고 실제 노이즈를 도입하기 위해 비교적 저가형 장비(Genelec 스피커, Røde 라발리에 마이크, Zoom H4N 필드 레코더)를 사용했습니다.
    *   **필터 실험 (Task 1):** 방음 튜브 내부에 스피커와 마이크를 배치하고, 스피커와 마이크 사이에 다양한 흡음재(폼, 종이 타월, 버블 랩 등)를 점진적으로 추가하여 주파수 감쇠(필터링) 수준을 조절하며 데이터를 녹음했습니다.
    *   **잔향 실험 (Task 2):** 길고 밀폐된 지하 복도에서 스피커와 마이크 사이의 거리를 변화시키며 음성 신호가 벽에 부딪혀 발생하는 심한 잔향 효과를 포착했습니다.
    *   **결합 실험 (Task 3):** Task 2의 잔향 데이터를 필터 실험 설정(Task 1의 특정 필터 재료 사용)을 통해 재녹음하여 필터링과 잔향이 결합된 복합적인 손상을 생성했습니다.
    *   녹음 과정에서 자연스럽게 발생하는 지연, 팝/크랙 소리, 주변 소음, 클리핑과 같은 실제 아티팩트와 비선형성을 의도적으로 포함했습니다.
3.  **임펄스 응답 측정:**
    *   시스템의 임펄스 응답을 추정하기 위해 스윕 사인파, 가우시안 노이즈, 짧은 가우시안 노이즈 버스트를 녹음했습니다. 이를 통해 비선형 공명 현상(특히 필터 실험에서 50-250Hz 대역)을 관찰했습니다.
4.  **성능 평가 지표:**
    *   **문자 오류율 (Character Error Rate, CER):** 정량적 평가 지표로 사용되며, $CER = \frac{S+D+I}{N}$ 공식으로 계산됩니다. 여기서 $S, D, I$는 각각 대체, 삭제, 삽입 문자 수이며, $N$은 총 문자 수입니다.
    *   **전처리:** CER 계산 전 공백 및 포맷 제거, 소문자 변환, 영국식-미국식 영어 변환을 통해 사소한 차이로 인한 오류를 방지했습니다.
    *   **음성 인식 모델:** Mozilla DeepSpeech v0.9.3 모델을 사용하여 음성 데이터를 텍스트로 변환하고, 이를 정답 텍스트와 비교하여 CER을 산출했습니다.
5.  **챌린지 규칙 및 채점:**
    *   참가 팀은 각 레벨에서 평균 CER이 0.3 미만이거나(노이즈 데이터보다 낮아야 함) 통과해야 다음 레벨로 진출할 수 있습니다.
    *   정상 작동 확인(Sanity Check): 복구된 오디오가 원본 스피커의 목소리임을 확인합니다.
    *   가장 많은 레벨을 통과한 팀이 우승하며, 동점일 경우 통과한 모든 레벨의 평균 CER로 순위를 결정합니다.

## 📊 Results
-   **클린 데이터 품질:** DeepSpeech 모델로 평가한 클린 데이터의 중앙값 CER은 0이었고, 전처리된 데이터 세트의 평균 CER은 0.005에서 0.009 사이로 매우 낮아, 고품질의 음성임을 확인했습니다.
-   **녹음 데이터 손상 수준:** 녹음된 데이터의 평균 CER은 0.0419(T1L1)에서 1.00(T3L2)까지 크게 증가하여, 난이도 수준에 따른 데이터 손상도 증가를 명확히 보여주었습니다. 이는 DeepSpeech 모델이 음성 품질 저하에 민감하게 반응함을 입증합니다.
-   **스펙트로그램 분석:** Figures 10과 11의 스펙트로그램 비교는 클린 데이터와 녹음 데이터 간의 시각적인 차이를 명확히 보여주며, 주파수 감쇠(필터링) 및 시간 분산(잔향) 효과를 시각적으로 확인시켜 줍니다.
-   **임펄스 응답의 비선형성:** 임펄스 응답 측정 결과(Figure 12), 특히 필터 실험에서 50Hz에서 250Hz 사이의 주파수 대역에서 두드러지는 비선형 공명(스피커의 저주파 재현 한계, 고조파 왜곡, 진동 공명)이 관찰되었습니다. 이는 실제 오디오 손상이 단순한 선형 시불변(LTI) 모델만으로는 충분히 설명되지 않음을 시사합니다.

## 🧠 Insights & Discussion
-   **실세계 데이터의 중요성:** 이 챌린지는 합성 데이터가 포착할 수 없는 실제 녹음 환경의 복잡성(아날로그 및 디지털 신호 처리, 비선형 효과)을 반영한 실측 데이터의 중요성을 강조합니다. 실제 데이터는 음성 향상 연구에 더 현실적이고 도전적인 과제를 제시합니다.
-   **음성 인식 모델의 유용성:** DeepSpeech와 CER을 음성 향상 평가 지표로 사용하는 것은 음성 품질의 주관적인 평가 문제를 우회하고, 음성 인식이 음성 향상의 중요한 다운스트림 태스크임을 고려할 때 매우 실용적입니다. 이는 음성 향상 알고리즘이 실제 응용 시나리오에서 얼마나 효과적인지 직접적으로 보여줍니다.
-   **LTI 모델의 한계:** 실제 임펄스 응답에서 관찰된 비선형 효과는 선형 시불변 모델만으로는 실제 오디오 손상을 정확하게 모델링하기 어렵다는 중요한 통찰을 제공합니다. 이는 음성 향상 알고리즘이 이러한 비선형성을 다룰 수 있어야 함을 시사합니다.
-   **데이터 기반 접근법의 필요성:** 복잡한 음성 신호에 대한 효과적인 모델 기반 사전 지식(prior)이 부족하므로, 비선형 효과를 효과적으로 다루기 위해 대규모 데이터셋을 활용하는 데이터 기반 접근법에 대한 의존성이 더욱 커지고 있음을 강조합니다.
-   **향후 연구 촉진:** 챌린지 종료 후에도 이 데이터셋이 음성 향상 및 역문제 분야의 지속적인 연구에 활용되기를 기대합니다.

## 📌 TL;DR
-   **문제:** 실세계의 복잡한 음성 신호 손상(필터링, 잔향, 비선형성) 문제를 해결하고, 기존의 합성 데이터 의존성에서 벗어나 실제 데이터를 활용한 음성 향상 기술 개발을 촉진합니다.
-   **제안 방법:** 헬싱키 음성 챌린지 2024(HSC2024)를 제안, 실제 녹음 환경에서 생성된 정제된 음성-손상된 음성 쌍 데이터셋 제공. 음성 향상 알고리즘의 성능 평가를 위해 Mozilla DeepSpeech 모델 기반의 문자 오류율(CER)을 정량적 지표로 사용합니다.
-   **핵심 결과:** 실제 데이터는 비선형적 손상을 포함하여 선형 시불변(LTI) 모델로만 설명하기 어렵고, DeepSpeech 모델은 음성 품질 저하에 민감하여 음성 향상 알고리즘의 효과적인 평가 지표가 될 수 있음을 입증합니다.