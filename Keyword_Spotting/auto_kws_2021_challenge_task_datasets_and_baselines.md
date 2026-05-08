# Auto-KWS 2021 Challenge: Task, Datasets, and Baselines

Jingsong Wang, Yuxuan He, Chunyu Zhao, Qijie Shao, Wei-Wei Tu, Tom Ko, Hung-yi Lee, Lei Xie (2021)

## 🧩 Problem to Solve

본 논문은 사용자가 직접 지정한 키워드와 특정 화자의 목소리로만 기기를 깨울 수 있는 **Customized Keyword Spotting (KWS)** 문제를 해결하기 위한 AutoML(Automated Machine Learning) 챌린지를 제안한다.

일반적인 KWS 시스템은 미리 정의된 고정 키워드(예: "OK Google", "Hey Siri")를 사용하지만, 실제 응용 분야에서는 사용자의 개인화된 요구에 따라 키워드와 화자를 자유롭게 설정할 수 있는 유연성과 보안성이 필요하다. 특히, 사용자가 어떤 언어나 억양을 사용하더라도 시스템이 이를 자동으로 학습하고 인식해야 한다는 점이 핵심이다.

논문의 목표는 인간의 개입 없이 등록(Enrollment) 과정과 예측(Prediction) 과정을 자동으로 수행하는 AutoML 솔루션을 구축하고, 이를 실제 환경에서 수집된 데이터셋을 통해 평가하는 것이다.

## ✨ Key Contributions

본 연구의 주요 기여는 다음과 같다.

- **Customized KWS 문제 정의**: 특정 화자가 지정한 키워드로만 작동하는 개인화된 깨우기 단어 감지 문제를 정의하고, 이를 위한 챌린지 프레임워크를 설계하였다.
- **실제 환경 데이터셋 구축**: 다양한 언어, 방언, 억양이 포함된 실제 환경의 오디오 데이터셋을 구축하여 시스템의 실용적 성능을 평가할 수 있게 하였다.
- **AutoML 기반 코드 경쟁 방식 도입**: 단순히 예측 결과물을 제출하는 것이 아니라, 제한된 시간과 자원 내에서 동작하는 전체 파이프라인 코드를 제출하게 함으로써 자동화된 머신러닝 솔루션을 독려하였다.
- **기준 시스템(Baselines) 제시**: 참여자들이 참고할 수 있도록 서로 다른 접근 방식을 가진 두 가지 베이스라인 시스템을 구현하고 그 성능을 분석하였다.

## 📎 Related Works

논문에서는 KWS와 관련된 기존의 여러 접근 방식을 소개한다.

- **전통적 KWS 및 딥러닝**: DNN 기반의 KWS가 기존 방식을 대체하고 있으며, 최근에는 더 복잡한 네트워크 구조가 도입되고 있다.
- **Query by Example (QbE)**: 테스트 오디오 세그먼트를 템플릿과 비교하여 결정하는 방식으로, 모델 재학습 없이 새로운 키워드에 대응할 수 있는 유연성을 제공한다.
- **Speaker Verification (SV)**: 화자 인증 시스템은 보안을 강화하지만, 기존 시스템들은 KWS와 SV를 별개의 파이프라인으로 처리하여 유연성이 떨어지는 한계가 있었다.
- **Self-Supervised Learning (SSL)**: wav2vec 2.0과 같은 사전 학습된 모델들이 음성 표현(Representation) 추출에서 뛰어난 성능을 보이고 있다.

본 챌린지는 이러한 개별 기술들을 통합하여, 저자원(Low-resource) 상황에서 Meta-learning이나 Few-shot learning과 같은 AutoML 기술을 통해 개인화된 KWS 문제를 해결하고자 한다.

## 🛠️ Methodology

### 1. 시스템 프로토콜 및 평가 지표

각 태스크는 $T = (D_e, D_{te}, L, B_T, B_S)$로 정의된다. 여기서 $D_e$는 등록 데이터, $D_{te}$는 레이블이 없는 테스트 데이터, $L$은 손실 함수, $B_T$는 시간 예산, $B_S$는 공간 예산을 의미한다.

평가 지표로는 Miss Rate (MR)와 False Alarm Rate (FAR)를 결합한 Wake-up Score $S_i$를 사용한다.
$$S_i = MR + \alpha \times FAR$$
여기서 $\alpha$는 페널티 계수로 9로 설정되었다. 최종 점수는 모든 화자의 $S_i$ 평균값으로 계산된다.

### 2. 데이터셋 구성

- **데이터 수집**: 중국어(표준어 및 방언)와 영어 사용자의 음성 데이터를 수집하였으며, 총 165명의 화자로부터 206개의 오디오를 확보하였다.
- **데이터 증강**: 테스트 데이터의 다양성을 높이기 위해 오디오 스플라이싱(Splicing), MUSAN 노이즈 추가(SNR 5dB~25dB), RIR-NOISES 추가, 볼륨 섭동(Perturbation) 등의 기법을 적용하였다.

### 3. 베이스라인 시스템

#### Baseline Method 1: QbE + SV 결합 방식

이 방식은 Query by Example(QbE) 시스템으로 1차 필터링을 수행하고, Speaker Verification(SV) 시스템으로 최종 검증하는 파이프라인을 가진다.

1. **QbE 모듈**:
   - **BNF Extractor**: MFCC 특징을 입력으로 하는 TDNN 기반의 Bottleneck Feature (BNF) 추출기를 사용한다.
   - **SLN-DTW**: 추출된 BNF를 템플릿 평균값과 Segmental Local Normalized DTW(SLN-DTW)를 통해 비교한다.
   - **판단**: SLN-DTW 점수가 임계값 $\gamma_1 = 0.80$보다 크면 SV 모듈로 전달한다.
2. **SV 모듈**:
   - **X-Vector**: Kaldi 툴킷을 사용하여 훈련된 X-Vector 모델을 통해 화자 벡터를 추출한다.
   - **검증**: 등록된 화자의 벡터와 입력 음성의 벡터 간 코사인 거리(Cosine Distance)를 계산하여 $\gamma_2 = 0.83$보다 크면 최종적으로 깨우기(Wake-up)로 판단한다.

#### Baseline Method 2: Pretrained Model + DTW 방식

사전 학습된 모델을 통해 특징을 추출하고 단순 거리 측정 방식을 사용하는 간단한 파이프라인이다.

1. **특징 추출**: wav2vec 2.0 사전 학습 모델의 Quantization 모듈을 사용하여 256차원의 음성 표현을 추출한다.
2. **판별**: Dynamic Time Warping (DTW)를 사용하여 테스트 오디오와 등록 오디오 세그먼트를 비교하며, 최소 점수를 해당 오디오의 점수로 취한다.
3. **판단**: 계산된 점수가 임계값(0.45~0.6 사이에서 결정)보다 크면 깨우기 상태로 판단한다.

## 📊 Results

실험은 Google Cloud VM (Nvidia Tesla P100 GPU) 환경에서 수행되었으며, 결과는 다음과 같다.

| Phase | Baseline | Average Score | Average Miss Rate | Average FA Rate | Compute Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Feedback** | Baseline 1 | 0.859 | 0.443 | 0.046 | 0:54:07 |
| | Baseline 2 | 1.695 | 0.481 | 0.135 | 1:24:07 |
| **Final** | Baseline 1 | 0.742 | 0.531 | 0.023 | 1:57:15 |
| | Baseline 2 | 1.086 | 0.691 | 0.044 | 3:01:32 |

- **정량적 결과**: Baseline 1이 Baseline 2보다 Miss Rate, False Alarm Rate, 그리고 최종 Average Score 모든 면에서 우수한 성능을 보였다.
- **효율성**: Baseline 2의 파이프라인이 구조적으로는 더 단순함에도 불구하고, 실제 실행 시간(Compute Time)은 Baseline 1이 더 빨랐다.
- **종합 평가**: 두 베이스라인 모두 성능이 상대적으로 낮게 나타났는데, 이는 Customized KWS 문제가 매우 어려운 과제임을 시사하며 향후 개선 여지가 많음을 보여준다.

## 🧠 Insights & Discussion

**강점 및 시사점**
본 연구는 단순히 알고리즘의 정확도만을 측정하는 것이 아니라, 하드웨어 자원과 시간 제한을 둔 '코드 경쟁' 형식을 도입하여 실제 배포 가능한 AutoML 솔루션의 필요성을 강조하였다. 또한, 다양한 언어와 억양이 포함된 데이터셋을 통해 실용적인 벤치마크를 제공하였다.

**한계 및 비판적 해석**
제시된 베이스라인들의 성능이 매우 낮다는 점은, 기존의 단순한 QbE나 사전 학습 모델-DTW 조합만으로는 개인화된 KWS 문제를 해결하기 어렵다는 것을 의미한다. 특히, 등록 데이터가 매우 적은 상황(Few-shot)에서 화자와 키워드의 특성을 동시에 효율적으로 학습하는 방법론에 대한 연구가 부족함을 알 수 있다.

또한, Baseline 1에서 사용된 BNF 추출기가 Magicdata라는 특정 데이터셋으로 훈련되었는데, 이것이 챌린지의 다양한 언어/억양 데이터셋에 얼마나 일반화될 수 있는지에 대한 상세 분석이 부족하다.

## 📌 TL;DR

본 논문은 사용자 지정 키워드와 화자를 인식하는 **Customized KWS를 위한 AutoML 챌린지**를 제안한다. 다양한 언어와 실제 환경 노이즈가 포함된 데이터셋을 구축하였으며, **BNF+SV 결합 방식(Baseline 1)**이 **wav2vec 2.0+DTW 방식(Baseline 2)**보다 성능과 효율성 면에서 우수함을 확인하였다. 전반적인 베이스라인 성능이 낮게 측정되어, 향후 Few-shot learning이나 Meta-learning 기반의 고도화된 AutoML 접근법이 필수적임을 시사한다.
