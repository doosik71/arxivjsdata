# DOES SINGLE-CHANNEL SPEECH ENHANCEMENT IMPROVE KEYWORD SPOTTING ACCURACY? A CASE STUDY

Avamarie Brueggeman, Takuya Higuchi, Masood Delfarah, Stephen Shum, Vineet Garg (2024)

## 🧩 Problem to Solve

본 논문은 단일 채널 Speech Enhancement(SE, 음성 향상) 기술이 Keyword Spotting(KWS, 키워드 검출)의 정확도를 실제로 향상시키는지에 대해 분석한다. 일반적으로 SE는 자동 음성 인식(ASR)의 성능을 높이기 위해 널리 연구되어 왔으나, KWS 분야에서의 효과, 특히 단일 채널 마이크로폰 신호만을 사용하는 환경에서의 효용성은 충분히 연구되지 않았다. 

배경 소음은 KWS의 성능을 저하시키는 핵심 요인이며, 이를 해결하기 위해 SE를 전처리기(Frontend)로 사용하는 방안이 고려될 수 있다. 하지만 단일 채널 SE의 경우 비선형 처리 과정에서 발생하는 Artifact(왜곡)가 후속 작업인 KWS 성능에 부정적인 영향을 미칠 수 있다는 우려가 존재한다. 따라서 본 연구의 목표는 다양한 훈련 조건(Clean vs Noisy training)과 최적화 기법(Joint Training, Audio Injection)을 적용하여 단일 채널 SE가 KWS 정확도에 미치는 영향을 종합적으로 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 단일 채널 SE와 KWS 백엔드 모델 간의 상호작용을 체계적으로 분석한 점에 있다. 주요 설계 아이디어는 다음과 같다.

1. **백엔드 훈련 조건에 따른 성능 분석**: KWS 모델이 깨끗한 음성(Clean speech)으로 훈련되었을 때와 소음이 섞인 음성(Noisy speech)으로 훈련되었을 때, SE의 적용 효과가 어떻게 달라지는지 대조 실험을 수행하였다.
2. **Joint Training의 도입**: 단순히 SE 모델을 고정하여 사용하는 것이 아니라, SE 모델과 KWS 모델을 동시에 학습시켜 KWS 작업에 최적화된 음성 향상을 유도하였다.
3. **Audio Injection 및 Soft-switching 탐색**: SE 과정에서 발생하는 왜곡을 줄이기 위해 향상된 신호와 원래의 소음 신호를 가중 평균하여 사용하는 Audio Injection 기법을 적용하였으며, 나아가 신경망이 각 발화별로 최적의 가중치를 예측하도록 하는 Soft-switching 구조를 실험하였다.

## 📎 Related Works

기존의 KWS 소음 강건성 연구들은 주로 KWS 모델 자체를 소음 데이터로 훈련시키는 방식에 집중해 왔으며, 명시적인 SE 전처리를 수행하는 경우는 드물었다. 특히 멀티 채널 SE를 이용한 KWS 성능 향상은 여러 연구에서 보고되었으나, 단일 채널 SE는 공간 정보의 부재와 비선형 처리로 인한 왜곡 문제로 인해 더 까다로운 과제로 남아 있다.

또한, KWS는 ASR과 달리 처리해야 할 키워드의 길이가 매우 짧고, 웨이크워드(Wake-word) 검출과 같은 실시간 스트리밍 처리가 필수적이라는 특성이 있다. 본 논문은 이러한 제약 조건을 고려하여 최신 시간 도메인 SE 모델인 Conv-TasNet과 고성능 KWS 백엔드인 BC-ResNet-8을 사용하여 기존 연구들보다 진보된 아키텍처 기반의 분석을 수행했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
본 연구의 전체 파이프라인은 $\text{Noisy Speech} \rightarrow \text{SE Model (Frontend)} \rightarrow \text{KWS Model (Backend)}$ 순으로 구성된다.

### 주요 구성 요소 및 역할
1. **Speech Enhancement (Frontend)**: **Conv-TasNet**을 사용한다. 이 모델은 인코더를 통해 시간 도메인 신호를 2차원 표현으로 변환하고, 세퍼레이터(Separator)가 마스크를 예측하여 적용한 뒤, 디코더를 통해 다시 1차원 시간 도메인 신호로 복원하는 구조이다. 실시간 스트리밍 적용을 위해 Causal 모델을 사용하였다.
2. **Keyword Spotting (Backend)**: **BC-ResNet-8** 모델을 사용한다. 이는 최신 KWS 어쿠스틱 모델로, 입력 신호로부터 키워드 존재 여부를 판별한다.

### 훈련 목표 및 손실 함수
- **SE 단독 훈련**: Signal-to-Distortion Ratio (SDR)를 손실 함수로 사용하여 훈련한다.
- **Joint Training**: SE와 KWS를 동시에 학습시키며, 다음과 같은 결합 손실 함수를 사용한다.
  $$L = L_{CE} + \beta \cdot L_{SDR}$$
  여기서 $L_{CE}$는 KWS의 Cross-Entropy 손실이며, $\beta$는 하이퍼파라미터로 $0.01$로 설정되었다.

### Audio Injection 및 Soft-switching
SE 과정의 왜곡을 완화하기 위해 향상된 신호 $x'$와 원래의 소음 신호 $x$를 가중 합산하여 $x''$를 생성한다.
$$x'' = \alpha x' + (1 - \alpha) x$$
- **Fixed $\alpha$**: 모든 발화에 대해 동일한 $\alpha$ 값을 적용한다.
- **Soft-switching**: BLSTM 기반의 별도 신경망을 통해 각 발화마다 최적의 $\alpha$ 값을 예측하도록 학습시킨다.

## 📊 Results

### 실험 설정
- **데이터셋**: Google Speech Command (GSC) v2 데이터셋을 사용하며, WHAM! 노이즈 데이터셋을 결합하여 $0 \sim 15\text{dB}$ 범위의 SNR을 가진 소음 데이터를 생성하였다.
- **비교 대상**: 
    - M1: Clean 데이터로만 훈련된 KWS 모델.
    - M2: Noisy 데이터로 훈련된 KWS 모델.
- **평가 지표**: 테스트 세트에서의 정확도(Accuracy, %)를 측정하였다.

### 주요 결과
1. **백엔드 모델의 훈련 상태에 따른 영향**:
    - **M1 (Clean-trained)**: SE를 적용하고 Joint Training을 수행했을 때, 소음 환경에서의 정확도가 $94.3\%$에서 $95.4\%$로 향상되었다. 즉, 백엔드가 깨끗한 데이터로만 학습된 경우 SE가 효과적이다.
    - **M2 (Noisy-trained)**: SE를 적용하더라도 기본 Baseline($96.0\%$)보다 성능이 낮거나 거의 차이가 없었다. 이미 소음에 강건하게 훈련된 모델에게는 SE가 오히려 방해가 되거나 무의미함을 시사한다.
2. **Audio Injection 및 Soft-switching 효과**:
    - $\alpha$ 값을 조정하여 원래 신호를 일부 섞어주는 방식이 SE 단독 적용($\alpha=1$)보다 성능이 좋았다.
    - 하지만 Soft-switching 모델을 통해 $\alpha$를 동적으로 예측하게 하더라도, 소음 데이터로 훈련된 M2의 Baseline 성능을 넘어서지는 못했다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 SE 성능 지표인 SDR의 향상이 반드시 다운스트림 태스크인 KWS의 정확도 향상으로 이어지지 않는다는 점을 실증적으로 보여주었다. 특히 백엔드 모델이 이미 소음 데이터에 노출되어 강건성을 획득한 경우, SE 전처리기가 도입하는 왜곡(Artifact)이 소음 그 자체보다 더 해로울 수 있다는 통찰을 제공한다.

### 한계 및 논의 사항
1. **제한된 문맥(Limited Context)**: Conv-TasNet은 긴 수용 영역(Receptive field)을 가질 수 있으나, GSC 데이터셋의 발화 길이가 1초로 매우 짧아 SE 모델이 충분한 문맥 정보를 학습하고 활용하는 데 한계가 있었을 가능성이 크다.
2. **데이터 부족**: Joint Training 시 검증 세트에서는 성능 향상이 보였으나 테스트 세트에서 일반화되지 않은 점은, Joint Training을 위해 필요한 데이터 양이 부족했음을 시사한다.
3. **시뮬레이션 데이터의 한계**: 실제 환경의 소음이 아닌 시뮬레이션된 소음을 사용했으므로, 실제 도메인에서의 SE-KWS 결합 효과는 다를 수 있다.
4. **소음 종류의 단순함**: 본 실험은 비음성 소음(Non-speech noise)만을 다루었으나, 실제 환경에서는 다른 사람의 음성(Speech noise)이 더 큰 문제가 될 수 있으며 이에 대한 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 단일 채널 음성 향상(SE)이 키워드 검출(KWS) 성능을 개선하는지 분석하였다. 실험 결과, **KWS 모델이 깨끗한 데이터로 학습된 경우**에는 SE가 도움이 되지만, **KWS 모델이 이미 소음 데이터로 학습되어 강건성을 갖춘 경우**에는 SE가 성능을 오히려 떨어뜨리거나 이득이 없음을 확인하였다. 이는 KWS와 같은 짧은 신호 처리 작업에서는 SE 전처리보다 소음 강건 훈련(Noise-robust training)이 더 효과적일 수 있음을 시사하며, 향후 더 방대한 실제 데이터셋을 통한 연구의 필요성을 제기한다.