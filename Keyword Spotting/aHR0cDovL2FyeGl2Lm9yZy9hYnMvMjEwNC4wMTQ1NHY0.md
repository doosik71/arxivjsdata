# Few-Shot Keyword Spotting in Any Language
Mark Mazumder, Colby Banbury, Josh Meyer, Pete Warden, Vijay Janapa Reddi

## 🧩 Problem to Solve
키워드 스포팅(KWS) 모델 학습은 각 키워드마다 수천 개의 다양한 화자 및 악센트 샘플을 수동으로 수집하고 선별해야 하는데, 이는 자원 부족 언어(low-resource languages)에서는 실질적으로 불가능한 요구사항입니다. 본 논문은 이 데이터 요구사항을 단 5개의 훈련 예제로 완화하여, 모든 언어에서 KWS를 가능하게 하는 것을 목표로 합니다.

## ✨ Key Contributions
*   다국어 임베딩 표현을 통해 22개 언어에서 유망한 5-샷 키워드 스포팅(KWS) 정확도를 달성했습니다.
*   다국어 임베딩이 정확도를 향상시키고 새로운 언어에도 일반화됨을 보였습니다.
*   자원 부족 환경에서 분류 및 스트리밍 정확도 평가에 크라우드소싱 데이터의 가치를 강조했습니다.
*   코드와 모델을 오픈 소스로 공개하고 Colab 환경을 제공하여 재현성 및 확장을 용이하게 했습니다.

## 📎 Related Works
*   기존 키워드 스포팅(KWS) 연구들은 심층 신경망(DNN), 컨볼루션 신경망(CNN), 장단기 기억 신경망(LSTM) 등을 사용했지만, 대상 키워드에 대해 수천 개의 샘플을 필요로 했습니다. 본 논문은 단 5개의 샘플만을 요구하여 데이터 부족 언어에서 KWS를 가능하게 합니다.
*   자원 부족 언어를 위한 KWS 기존 방식으로는 다국어 병목 특징(multilingual bottleneck features), 동적 시간 왜곡(DTW), 오토인코더 등을 활용하는 연구([10], [11])가 있었으나, 이들은 키워드당 약 30개의 샘플과 몇 시간의 전사되지 않은 데이터를 필요로 했습니다.
*   소수 샷 학습(few-shot learning) 분야에서는 3개의 예제로 영어와 한국어에서 높은 KWS 정확도를 보인 연구([12]), 영어에서 제로 샷(zero-shot) 접근 방식을 시도한 연구([13]), 5개 언어에서 소수 샷 임베딩을 평가한 연구([14]), 10개 언어에서 음성 용어 검색을 수행한 연구([15]) 등이 있습니다. 본 논문은 더 간단한 임베딩 방식을 사용하고, 훨씬 더 많은 수의 언어(22개)와 화자에 대해 단 5개의 훈련 예제로 소수 샷 성능을 평가합니다.
*   음성 합성을 통한 KWS 접근 방식([16])도 있었으나, 본 연구는 합성할 데이터가 부족한 언어를 대상으로 하므로 직접적인 비교 대상이 아닙니다.

## 🛠️ Methodology
이 연구는 다국어 임베딩 모델 학습, 전이 학습 수행, 대규모 키워드 데이터셋 자동 추출의 세 가지 주요 단계로 구성됩니다.

*   **다국어 임베딩 모델 (그림 1a)**
    *   **아키텍처**: TensorFlow Lite Micro의 마이크로프론트엔드 스펙트로그램(49x40x1)을 입력으로 사용합니다. 약 1,100만 개의 파라미터를 가진 EfficientNet-B0 기반의 분류기이며, 글로벌 평균 풀링 레이어, ReLU 활성화 함수를 사용하는 두 개의 2048-유닛 완전 연결 레이어, 1024-유닛 SELU 활성화 레이어(최종 전 계층), 그리고 761개 카테고리의 소프트맥스 출력으로 구성됩니다. 이 분류기의 최종 전 계층 출력을 임베딩 표현으로 재사용합니다. SELU 활성화 함수는 자체 정규화 특성 때문에 선택되었습니다.
    *   **훈련 데이터**: 9개 언어(표 1 참조)에서 760개의 빈번한 단어를 대상으로 총 140만 개의 샘플로 모델을 훈련시켰습니다. 짧은 단어와 불용어를 제외하기 위해 문자 길이가 3 이상인 단어만 선택했습니다. 각 추출 샘플은 1초 길이로 침묵으로 패딩되었습니다.
    *   **데이터 증강**: 모든 훈련 샘플의 10%는 Google Speech Commands 데이터셋의 배경 소음을 사용한 배경 소음 카테고리로 구성됩니다. 키워드 샘플은 무작위 100ms 시간 이동, 10% SNR로 다중화된 배경 소음, 그리고 SpecAugment [21]로 증강되었습니다.

*   **5-샷 전이 학습 (그림 1b)**
    *   **미세 조정**: 임베딩 레이어의 출력 특징 벡터에 대해 3-클래스 소프트맥스 레이어(대상, 알 수 없음, 배경 카테고리)를 미세 조정하기 위해 단 5개의 대상 샘플을 사용합니다.
    *   **비대상 샘플**: 9개 임베딩 언어에서 미리 계산된 5,000개의 "알 수 없음" 발화 뱅크에서 128개의 비대상 샘플을 가져옵니다. 임베딩 모델이 보지 못했던 새로운 언어(예: 웨일스어)에서 KWS 모델을 훈련할 때도 이 뱅크의 비대상 샘플이 사용되므로, 해당 언어에서 추가적인 비대상 샘플을 수집할 필요가 없습니다.
    *   **레이어 고정**: 미세 조정 시 임베딩 레이어의 가중치는 고정되며, 소프트맥스 레이어만 업데이트됩니다.
    *   **훈련 샘플 구성**: 총 256개의 훈련 샘플 중 약 45%는 대상 카테고리(5개의 대상 예제를 증강), 45%는 미리 계산된 비대상 단어 집합에서 추출된 부정 샘플, 10%는 배경 소음으로 구성됩니다.

*   **데이터셋 생성**
    *   **소스**: Common Voice [2]에서 키워드를 추출했습니다.
    *   **정렬**: Montreal Forced Aligner [3]를 사용하여 Common Voice의 각 `<audio,transcript>` 쌍에 대한 단어 수준 정렬을 추정했습니다.
    *   **추출**: 정렬 타이밍을 사용하여 키워드 추출을 자동화했으며, 총 22개 언어에서 3,126개의 키워드에 걸쳐 4,383,489개의 샘플을 추출했습니다.

## 📊 Results
본 연구는 5-샷 키워드 스포팅(KWS) 모델에 대한 분류 및 스트리밍 정확도 평가 결과를 제시합니다.

*   **임베딩 모델 정확도 (표 1)**: 다국어 임베딩 모델은 9개 훈련 언어의 검증 데이터셋에서 전체 분류 정확도 79.81%를 달성했습니다. 언어별로는 카탈루냐어가 87.63%로 가장 높았고, 네덜란드어가 72.60%로 가장 낮았습니다.
*   **도메인 간극 (표 2)**: Common Voice 추출 데이터셋과 Google Speech Commands(GSC) 수동 기록 데이터셋 간의 "left" 키워드에 대한 `tinyconv` 모델 교차 비교 결과, Common Voice 추출 데이터로 훈련된 모델이 GSC 테스트셋에서 GSC 데이터로 훈련된 모델보다 성능이 낮아(78.07% vs 90.49%) 도메인 간극이 있음을 시사했습니다.
*   **단일 언어 vs. 다국어 임베딩 (그림 2a vs. 2b)**: 다국어 임베딩을 사용했을 때 KWS 분류 정확도가 단일 언어 임베딩보다 크게 향상되었습니다. 20개 미학습 단어를 대상으로 한 5-샷 KWS 모델에서 평균 $F_1$ 점수는 단일 언어 임베딩의 0.58에서 다국어 임베딩의 0.75로 증가했습니다(임계값 0.8 기준). 이는 언어 간 일반화 가능한 특징이 각 언어의 정확도에 긍정적인 영향을 미칠 수 있음을 보여줍니다.
*   **임베딩 외부 언어 분류 (그림 2c)**: 임베딩 모델 훈련 시 관찰되지 않은 13개 새로운 언어에 대해서도 다국어 임베딩 모델이 잘 일반화됨을 보였습니다. 평균 $F_1$ 점수는 임계값 0.8에서 0.65를 기록했습니다.
*   **5-샷 스트리밍 정확도 (그림 3)**:
    *   **키워드 스포팅 (웨이크-워드 모의실험, 그림 3a/b)**: 22개 언어의 440개 키워드에 걸쳐 평균 87.4%의 스트리밍 KWS 정확도와 4.3%의 오인식률(FAR)을 달성했습니다(임계값 0.8 기준). 주변 오디오(context)로 패딩된 임베딩 표현을 사용해도 성능 저하는 관찰되지 않았습니다.
    *   **키워드 검색 (연속 음성, 그림 3c/d)**: 임베딩 모델을 침묵 및 주변 오디오(context)로 패딩된 샘플 모두를 사용하여 훈련했을 때 정확도가 크게 향상되었습니다. 임계값 0.8에서 평균 TPR 77.2% 및 FPR 2.3%를 달성하여, 키워드가 연속된 발화에서 연음 현상(coarticulation effects) 속에서도 식별될 수 있음을 보여주었습니다.

## 🧠 Insights & Discussion
*   **함의 및 시사점**
    *   본 연구는 자동으로 생성된 데이터셋으로 사전 훈련된 임베딩 표현을 사용하여 22개 언어에서 임의의 키워드에 대한 5-샷 키워드 스포팅(KWS)이 가능함을 입증했습니다.
    *   훈련되지 않은 언어에 대해서도 임베딩이 일반화될 수 있음을 보여주었으며, 이는 자원 부족 언어에 대한 음성 기반 명령 인터페이스를 신속하게 개발할 수 있는 기반을 마련합니다. 자원 부족 언어의 사용자가 단 5개의 대상 키워드 예제를 녹음하는 것만으로 강력한 다중 화자 KWS 모델을 얻을 수 있습니다.
    *   주변 오디오 컨텍스트를 포함하여 임베딩을 훈련함으로써 연속 음성에서의 키워드 검색 정확도가 향상됨을 확인했습니다.

*   **한계 및 향후 연구**
    *   자동 추출된 키워드와 수동으로 기록된 키워드 간의 도메인 간극(domain gap)에 대한 추가적인 탐색이 필요합니다.
    *   향후 연구에서는 단어 수준 정렬이 비실용적인 언어에 대한 지원을 모색할 것입니다.
    *   현재의 미세 조정 방식은 간단하며, 문헌의 자연스러운 확장(예: Prototypical Networks [24])을 적용할 계획입니다.
    *   온-디바이스 배포를 위해 지식 증류(knowledge distillation)를 통해 더 작은 임베딩 표현을 개발하는 것을 목표로 합니다.
    *   Google Speech Commands(GSC)로 훈련된 최신 모델과의 추가 비교가 이루어질 것입니다.
    *   자동화된 KWS 평가에서는 5개의 훈련 샘플이 강제 정렬 추출에서 무작위로 선택되며 수동으로 검사되지 않으므로, 일부 모델의 성능은 추출 오류나 원본 Common Voice 녹음의 오류로 인해 부정적인 영향을 받을 수 있다는 한계가 있습니다.

## 📌 TL;DR
**문제**: 키워드 스포팅(KWS) 모델은 수천 개의 훈련 샘플을 필요로 하여 자원 부족 언어에서는 실현하기 어렵습니다.
**방법**: Common Voice 데이터셋과 강제 정렬(forced alignment)을 통해 9개 언어에서 140만 개의 샘플을 자동으로 추출하여 다국어 임베딩 모델을 훈련합니다. 새로운 키워드를 스포팅하기 위해 단 5개의 대상 키워드 예제와 128개의 비대상 샘플을 사용하여 이 임베딩 모델을 미세 조정합니다(임베딩 가중치는 고정).
**주요 결과**: 임베딩 모델이 보지 못했던 9개 언어의 새로운 키워드 180개에 대해 평균 $F_1$ 점수 0.75를 달성했으며, 임베딩 모델이 전혀 보지 못했던 13개 새로운 언어의 키워드 260개에 대해서는 평균 $F_1$ 점수 0.65를 달성했습니다. 또한, 22개 언어에서 평균 87.4%의 스트리밍 KWS 정확도(4.3% 오인식률)와 연속 음성에서의 유망한 키워드 검색 결과를 보였습니다.