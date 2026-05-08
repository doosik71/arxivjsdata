# Understanding temporally weakly supervised training: A case study for keyword spotting

Heinrich Dinkel, Weiji Zhuang, Zhiyong Yan, Yongqing Wang, Junbo Zhang and Yujun Wang (2023)

## 🧩 Problem to Solve

본 논문은 Keyword Spotting (KWS) 모델 학습 시 요구되는 강력한 지도 학습(Strong Supervision)의 의존성 문제를 해결하고자 한다. 일반적으로 DNN 기반의 KWS 학습에는 발화된 키워드의 정확한 시작점과 끝점이라는 정밀한 시간적 위치 정보가 필요하다. 이러한 정보는 주로 자동 음성 인식(ASR) 시스템의 forced alignment를 통해 획득하지만, 오디오에 노이즈가 심할 경우 alignment가 어긋나 감지 정확도가 떨어지는 한계가 있다. 또한, 사람이 직접 정밀하게 라벨링하는 작업은 비용과 시간이 매우 많이 소요된다.

따라서 본 연구의 목표는 정밀한 시간 정보 없이 coarse한 수준의 라벨(예: 5초 이내에 키워드가 존재함)만 사용하는 temporally weakly supervised learning(시간적 약지도 학습)이 KWS 모델의 성능에 미치는 영향을 탐구하고, 이것이 실제 환경에서 강력한 지도 학습의 대안이 될 수 있는지 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 KWS 모델을 학습시킬 때 정확한 위치 정보를 제공하는 것이 오히려 모델의 일반화 능력을 제한할 수 있다는 관점에서 출발한다. 연구진은 모델이 단순히 타겟 라벨을 찾는 것을 넘어, 노이즈 속에서 타겟 라벨의 위치를 스스로 찾아내도록(localize) 유도하는 약지도 학습 방식이 더 효과적일 수 있다고 주장한다.

주요 기여 사항은 다음과 같다:

- KWS 분야에서 시간적 약지도 학습(temporally weakly supervised training)이 미치는 영향에 대한 체계적인 연구 수행.
- 강지도 학습(strongly supervised) 기반의 KWS와 약지도 학습 기반 KWS의 성능 비교 분석.
- 약지도 학습 기반 KWS 모델을 성공적으로 학습시키기 위한 실질적인 방법론(예: random cropping) 제시.

## 📎 Related Works

기존의 오디오 관련 분야인 audio tagging, voice activity detection (VAD), sound event detection (SED) 등에서는 약지도 학습이 이미 성공적으로 적용되어 왔다. 하지만 KWS 분야에서는 그동안 부정확한 라벨 경계(imperfect label boundaries)를 모델 성능을 저하시키는 주요 장애물로 간주하여, 이를 해결하기 위한 max-pooling loss나 sequence-to-sequence training 등의 연구가 진행되었다.

본 논문은 기존 접근 방식과 달리, 라벨의 불확실성을 장애물이 아닌 모델의 로컬라이제이션 능력을 키우는 기회로 활용하며, 특히 노이즈가 심한 환경에서 forced alignment의 한계를 극복할 수 있는 대안으로서 약지도 학습의 가능성을 제시한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 연구는 강지도 학습 베이스라인과 제안하는 End-to-End (E2E) 약지도 학습 모델을 비교한다. 전체적인 흐름은 데이터셋 준비, 모델 학습, 그리고 깨끗한 테스트 세트를 이용한 평가 순으로 진행된다.

### 데이터셋 구성

연구진은 Google Speech Commands V1 (GSCV1) 데이터셋과 Audioset 노이즈 데이터셋을 사용하여 세 가지 유형의 학습 데이터를 구축하였다.

1. **Clean Dataset**: 기본 베이스라인으로, 노이즈가 없는 깨끗한 키워드 데이터.
2. **Weak-ts**: 타겟 키워드를 $t$초($t \in \{3, 5, 7\}$) 길이의 노이즈 시퀀스 내에 겹치지 않게 무작위로 삽입한 데이터.
3. **Weak-SNR-ts**: 키워드 샘플을 특정 SNR(0, 5, 10 dB) 비율로 노이즈와 섞어 생성한 데이터.

### 모델 아키텍처 및 학습 절차

- **Strongly Labeled Baseline**: Kaldi 툴킷의 KWS 레시피를 따르는 LF-MMI 방식과 TDNN(Time-Delay Neural Network)을 사용한다. Oracle(정확한 타임스탬프 사용)과 Force-align(FA, 예측된 위치 사용) 두 가지 설정으로 비교한다.
- **Proposed E2E Approach**: TC-ResNet8 모델을 백엔드로 사용한다. 이 모델은 파라미터 수가 66,000개로 매우 적어 과적합 가능성이 낮고 학습 속도가 빠르다.
- **Feature Extraction**: 모든 오디오는 16 kHz로 샘플링되며, 32ms 윈도우와 10ms 시프트 값을 가진 64 bins의 log-Mel spectrograms (LMS)를 특징량으로 사용한다.

### 학습 목표 및 손실 함수

E2E 모델의 학습 목표는 표준 categorical cross-entropy loss를 최소화하는 것이다.

- **Optimizer**: Adam ($\text{learning rate} = 0.001$)
- **Batch Size**: 64
- **Epochs**: 최대 200 epoch
- **Evaluation**: 검증 세트에서 최고 성능을 보인 상위 4개 체크포인트의 가중치를 평균(weight-averaging)하여 테스트 세트에서 평가한다.

## 📊 Results

### 실험 설정

- **데이터셋**: GSCV1 (11개 클래스 서브셋)
- **지표**: Accuracy(정확도) 및 mAP(mean Average Precision)

### 주요 결과

1. **약지도 학습의 효과 (Table 1)**:
   - 깨끗한 데이터에서는 강지도(Oracle/FA)와 제안 방식(E2E)이 유사한 성능을 보인다.
   - 노이즈가 추가될수록 강지도 FA 방식의 성능은 급격히 하락($96.21\% \rightarrow 87.21\%$)하지만, 제안하는 E2E 방식은 입력 길이가 3s에서 7s로 늘어나도 성능 하락이 매우 적어(97.03% $\rightarrow$ 96.68%) 안정적인 성능을 유지한다. 이는 E2E 모델이 명시적 감독 없이도 타겟 키워드를 자동으로 로컬라이즈할 수 있음을 시사한다.

2. **Train-Test 길이 불일치 해결 (Table 2)**:
   - 학습 데이터(3~7s)와 테스트 데이터(1s)의 길이 차이로 인해 mAP가 하락하는 현상이 발견되었다. 이를 해결하기 위해 학습 중 **random cropping (1s)**을 적용한 결과, mAP가 크게 향상되었으며 이는 테스트 환경과의 정합성을 높이고 데이터 증강 효과를 가져온 결과로 해석된다.

3. **가산 노이즈(Additive Noise) 환경 (Table 3)**:
   - SNR이 낮아질수록(노이즈가 심해질수록) 전체적인 성능은 하락한다.
   - 하지만 SNR 0dB 이상의 환경에서 타겟 키워드가 전체 음성의 약 15%($1/7$초) 이상 존재한다면, 약지도 학습 방식이 강지도 Oracle 베이스라인보다 더 높은 성능을 낼 수 있음을 확인하였다.

4. **Ablation Study (Table 4)**:
   - 타겟 샘플에만 약지도 라벨을 적용하고 non-target 샘플은 깨끗한 데이터를 사용했을 때, 샘플 길이가 길어질수록 성능이 급격히 저하되었다. 이는 모델이 노이즈 샘플을 '거부'하는 법을 배우기 위해서는 non-target 데이터에서도 적절한 노이즈 샘플(Audioset 등)이 제공되어야 함을 의미한다.

## 🧠 Insights & Discussion

본 연구는 시간적 약지도 학습이 일종의 **Label Smoothing** 역할을 수행한다는 중요한 통찰을 제공한다. 약지도 라벨링된 샘플을 random cropping 할 경우, 타겟 키워드가 포함되지 않은 'False Positive' 샘플이 생성될 확률이 높다. 모델은 이러한 샘플들을 non-target 데이터(노이즈가 포함된)와 함께 학습하면서, 결과적으로 더 강건한 결정 경계를 형성하게 된다.

강점으로는 정밀한 라벨링 없이도 충분히 높은 성능을 낼 수 있음을 입증하여 데이터 구축 비용을 획기적으로 줄일 수 있다는 점이다. 다만, 한계점으로는 강지도 FA 방식과 약지도 방식의 라벨 획득 과정(자동 vs 수동)이 다르기 때문에 완전히 동일 선상에서 비교하기 어렵다는 점이 있다.

비판적으로 해석하자면, 본 연구의 결과는 인위적으로 생성된 데이터셋(Audioset 노이즈 삽입)에 기반하고 있어, 실제 야생(in-the-wild) 환경의 복잡한 노이즈에서도 동일한 효율성이 보장될지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 KWS 모델 학습 시 정밀한 시간적 위치 정보 없이 coarse한 라벨만 사용하는 **시간적 약지도 학습(temporally weakly supervised learning)**의 효용성을 입증하였다. 실험 결과, 약지도 학습은 깨끗한 데이터에서 강지도 학습과 대등한 성능을 보일 뿐만 아니라, **강한 노이즈가 존재하는 환경에서는 오히려 강지도 학습(특히 forced alignment 기반)보다 더 우수한 성능**을 나타냈다. 특히 학습 시 **random cropping**을 적용하는 것이 성능 최적화에 필수적이며, 타겟 키워드가 입력 데이터의 최소 15% 이상을 차지할 때 효과적이다. 이 연구는 향후 대규모의 정밀 라벨링 없이도 고성능 KWS 모델을 구축할 수 있는 실질적인 방향성을 제시한다.
