# ASR-free CNN-DTW keyword spotting using multilingual bottleneck features for almost zero-resource languages

Raghav Menon, Herman Kamper, Emre Yilmaz, John Quinn, Thomas Niesler (2018)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 자원이 매우 부족한(almost zero-resource) 언어 환경에서 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템 없이 효율적인 키워드 검출(Keyword Spotting, KWS)을 수행하는 것이다.

일반적인 KWS 시스템은 ASR을 통해 생성된 래티스(lattice)를 탐색하여 키워드의 존재 여부를 판단한다. 하지만 최신 ASR 시스템은 막대한 양의 전사 데이터(transcribed speech)를 필요로 하며, 특히 아프리카 일부 지역과 같이 언어 자원이 부족한 환경에서는 숙련된 전사 작업자를 찾기 어려워 시스템 구축에 큰 제약이 따른다. 이러한 문제는 인도주의적 구호 활동을 위한 라디오 브라우징 시스템의 신속한 배포를 방해하는 주요 원인이 된다.

따라서 본 연구의 목표는 대규모 전사 데이터 없이, 소량의 고립된 키워드(isolated keywords) 예시와 전사되지 않은 대량의 도메인 데이터만을 활용하여 성능이 우수하면서도 추론 속도가 빠른 KWS 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Dynamic Time Warping(DTW)의 정밀함과 Convolutional Neural Network(CNN)의 속도 효율성을 결합**하고, 이를 **다국어 Bottleneck Features(BNFs)로 강화**하는 것이다.

구체적인 설계 아이디어는 다음과 같다.
1. **DTW-based Supervision**: 소량의 키워드 예시를 사용하여 전사되지 않은 데이터에 대해 DTW 템플릿 매칭을 수행하고, 여기서 얻은 유사도 점수를 CNN의 학습 타겟(Target)으로 사용한다. 즉, CNN이 DTW의 동작을 모방하도록 학습시켜, 추론 시에는 정렬(alignment) 과정 없이 빠르게 키워드를 검출하게 한다.
2. **Multilingual BNFs의 도입**: 대상 언어의 데이터가 부족하므로, 자원이 풍부한 다른 다국어 데이터로 학습된 TDNN(Time Delay Neural Network)의 중간층인 Bottleneck layer에서 특징을 추출하여 사용한다. 이를 통해 다양한 언어에 공통적으로 존재하는 음향적 특성을 활용하여 저자원 언어의 KWS 성능을 높이고자 하였다.

## 📎 Related Works

기존의 ASR-free KWS 방식으로는 쿼리를 오디오 형태로 제공하는 Query-by-Example(QbyE) 방식이 주로 사용되었다.
- **DTW 기반 접근**: DTW는 적은 데이터로도 작동 가능하지만, 모든 검색 구간에 대해 반복적인 정렬을 수행해야 하므로 계산 비용이 매우 높고 속도가 느리다는 한계가 있다.
- **신경망 기반 임베딩**: RNN, Autoencoding Encoder-Decoder, Siamese CNN 등을 사용하여 가변 길이의 음성을 고정 차원 벡터로 변환해 비교하는 방식이 제안되었다. 그러나 이러한 방식들은 대부분 대량의 훈련 데이터를 필요로 한다는 점에서 저자원 환경에 적용하기 어렵다.

본 논문은 이전 연구에서 제안된 CNN-DTW 구조를 계승하되, 특징 추출 단계에서 MFCC(Mel-Frequency Cepstral Coefficients) 대신 다국어 BNFs를 적용함으로써 데이터 부족 문제를 해결하고 성능을 최적화했다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인
시스템은 크게 두 단계로 구성된다. 먼저 DTW를 이용해 CNN을 학습시키기 위한 지도 신호(supervisory signal)를 생성하고, 이후 이 신호를 타겟으로 하여 CNN 모델을 학습시킨다. 학습이 완료된 CNN은 새로운 입력 음성에서 정렬 과정 없이 즉각적으로 키워드 존재 확률을 출력한다.

### 2. DTW를 이용한 타겟 생성
소량의 고립된 키워드 템플릿과 전사되지 않은 음성 데이터 $U$ 사이의 유사도를 계산한다. 키워드 $K$의 $i$번째 예시를 $k_i$, $U$ 내의 연속적인 세그먼트를 $u_p$라고 할 때, 각 프레임 $j$에 대한 점수 $c_j$는 다음과 같이 계산된다.

$$c = \min_{i \in 1...N} \left[ \min_{u_p \in U} \text{DTW}\{k_i, u_p\} \right]$$

여기서 $\text{DTW}\{k_i, u_p\}$는 두 특징 시퀀스 간의 정렬 비용이다. 계산된 비용 $c \in [0, 2]$는 $0$일 때 완벽한 일치를 의미하며, 이를 $\text{sigmoid}$ 함수 등을 통해 $y \in [0, 1]$ 범위의 타겟 벡터로 변환하여 CNN 학습에 사용한다.

### 3. CNN 아키텍처 및 학습
CNN은 DTW의 결과값을 예측하는 회귀 모델처럼 동작하며, Summed Cross-Entropy 손실 함수를 사용하여 학습된다.
- **구조**: 3개의 Convolutional layer (필터 수 64, 128, 256) $\rightarrow$ Max pooling $\rightarrow$ 3개의 Dense layer (500, 100, 300 units).
- **추론**: 입력 특징을 CNN에 통과시켜 출력된 값을 기반으로 임계값을 적용해 키워드 존재 여부를 판단한다.

### 4. 특징 추출기 (Feature Extractors)
본 논문은 다음 세 가지 특징 추출 방식을 비교 분석하였다.
- **Multilingual BNFs**: 여러 언어로 학습된 TDNN의 중간 병목층(Bottleneck layer)에서 추출한 특징이다.
    - **2-language TDNN**: 네덜란드어와 프리지안어(Frisian)로 학습된 11층 모델.
    - **10-language TDNN**: GlobalPhone 코퍼스의 10개 다양한 언어로 학습된 6층 모델.
- **Stacked Denoising Autoencoder (SAE)**: 전사되지 않은 5개 언어 데이터로 학습된 비지도 학습 모델로, 입력 데이터를 재구성하는 과정에서 학습된 내부 표현을 특징으로 사용한다.
- **MFCC**: 전통적인 음성 특징 추출 방식으로, 비교를 위한 베이스라인으로 사용되었다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: South African Broadcast News (SABN) 코퍼스와 40개 키워드에 대한 고립 발화 데이터(1,920개)를 사용하였다.
- **지표**: ROC 곡선 하단 면적인 $\text{AUC}$와 $\text{Equal Error Rate (EER)}$를 측정 지표로 사용하였다.

### 2. 주요 결과
- **특징 추출기 성능 (Table 2)**: 10개 언어로 학습된 BNFs가 가장 높은 $\text{AUC}(0.7725)$와 가장 낮은 $\text{EER}(0.2884)$를 기록하며 MFCC보다 우수한 성능을 보였다. 반면, 2개 언어 BNFs와 SAE는 MFCC보다 성능이 낮게 나타났다.
- **시스템별 성능 비교 (Table 3)**:
    - BNFs를 사용했을 때 CNN-DTW 시스템의 $\text{AUC}$는 $0.7422$로, MFCC 기반($0.6285$) 대비 **절대치 기준 10.9% 향상**되었다.
    - CNN-DTW는 자신이 모방하고자 하는 대상인 DTW-KS($\text{AUC } 0.7699$)에 근접하는 성능을 보였다.
- **효율성**: 추론 속도 면에서 CNN-DTW는 GPU(GTX 1080) 기준 약 5분이 소요된 반면, DTW-KS는 CPU(20-core) 기준 900분이 소요되어 압도적인 속도 이점을 가짐을 확인하였다.

## 🧠 Insights & Discussion

본 연구를 통해 도출된 주요 통찰은 다음과 같다.
첫째, **언어적 다양성의 중요성**이다. 2개 언어보다 10개 언어로 학습된 BNFs가 훨씬 우수한 성능을 보였다는 점은, 더 다양하고 많은 언어 데이터를 통해 학습된 모델이 타겟 언어에 공통적으로 적용 가능한 더 범용적인 음향 특징을 추출할 수 있음을 시사한다.
둘째, **지도 학습 기반 특징의 우위**이다. 비지도 학습 방식인 SAE보다 지도 학습 방식인 BNFs의 성능이 높게 나타난 것은, 음소(phone) 단위의 명시적인 타겟을 가지고 학습한 특징이 KWS 작업에 더 유용함을 보여준다.
셋째, **지식 증류(Knowledge Distillation)의 효과**이다. 매우 느리지만 정밀한 DTW를 '교사(Teacher)'로 삼아 CNN을 '학생(Student)'으로 학습시킴으로써, 정확도는 유지하면서 추론 속도를 획기적으로 개선할 수 있었다.

다만, 특정 키워드(예: 'health')에서는 성능 향상이 미미했다는 점은 키워드의 음향적 특성에 따라 BNFs의 효과가 다를 수 있음을 의미하며, 이는 향후 연구에서 다루어야 할 과제로 남는다.

## 📌 TL;DR

본 논문은 전사 데이터가 거의 없는 저자원 언어 환경에서 **다국어 Bottleneck Features(BNFs)**와 **CNN-DTW** 구조를 결합한 키워드 검출 시스템을 제안하였다. 10개 언어로 학습된 BNFs를 사용하여 MFCC 대비 $\text{AUC}$를 10.9% 향상시켰으며, DTW의 정밀함을 CNN으로 전이시켜 추론 속도를 획기적으로(900분 $\rightarrow$ 5분) 단축하였다. 이 연구는 자원이 부족한 지역의 인도주의적 모니터링 시스템 구축에 있어 실질적이고 효율적인 대안을 제시한다.