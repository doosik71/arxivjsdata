# NATURAL LANGUAGE GENERATION FOR ELECTRONIC HEALTH RECORDS

Scott Lee (2018/2019)

## 🧩 Problem to Solve

본 논문은 전자 건강 기록(Electronic Health Records, EHRs) 내의 비정형 텍스트, 특히 응급실(Emergency Department, ED)의 주소(Chief Complaints, CC)를 합성하여 생성하는 문제를 다룬다.

기존의 합성 EHR 생성 방법들은 주로 수치형이나 범주형 같은 이산적 변수(discrete variables)를 생성하는 데 집중되어 있었으며, 자유 형식의 텍스트를 생성하는 기능은 결여되어 있었다. EHR 데이터는 환자의 개인 식별 정보(Personally Identifiable Information, PII)를 포함하고 있어 HIPAA와 같은 규제로 인해 연구 목적으로 외부와 공유하기가 매우 어렵다. 텍스트 데이터를 비식별화(de-identification)하기 위해서는 수동 검토가 필요한 경우가 많아 비용과 시간이 많이 소요된다는 문제가 있다.

따라서 본 연구의 목표는 EHR의 이산적 변수들을 입력으로 하여, 임상적으로 유효하면서도 개인 식별 정보가 제거된 현실적인 합성 주소(synthetic chief complaints)를 생성하는 딥러닝 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기계 번역(Machine Translation)이나 이미지 캡셔닝(Image Captioning)에서 사용되는 **Encoder-Decoder 모델** 구조를 EHR 텍스트 생성에 적용하는 것이다.

기존의 이미지 캡셔닝이 '이미지(비텍스트) $\rightarrow$ 텍스트'의 관계를 학습하듯, 본 모델은 '환자의 이산적 특성(비텍스트) $\rightarrow$ 주소 텍스트'의 관계를 학습하도록 설계되었다. 이를 통해 단순한 데이터 복제가 아닌, 입력된 환자 정보에 부합하는 자연어 문장을 생성함으로써 개인 정보 유출 위험을 낮추고 데이터의 유용성을 유지하고자 하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급한다.

- **Generative Adversarial Networks (GANs):** Choi et al. (2017)은 GAN을 사용하여 환자의 이산적 변수들을 합성하는 모델을 제안하였다. 하지만 이 모델은 텍스트 데이터를 생성하지 못한다는 한계가 있다.
- **Encoder-Decoder Models:** Bahdanau et al. (2014) 등의 기계 번역 연구와 Vinyals et al. (2015)의 이미지 캡셔닝 연구가 기반이 되었다. 이들은 서로 다른 도메인의 데이터를 입력받아 타겟 시퀀스를 생성하는 데 성공하였으며, 본 논문은 이 프레임워크를 의료 데이터에 이식하였다.

## 🛠️ Methodology

### 1. 데이터 구조 및 전처리

- **입력 데이터 ($R$):** 연령대, 성별, 퇴원 진단 코드(CCS 코드), 도착 수단, 처분(disposition), 병원 코드, 월, 연도 등의 변수를 사용한다.
- **전처리:** 모든 변수를 정수 형태로 리코딩한 후 sparse binary vector(원-핫 벡터 확장형)로 변환하여 403차원의 벡터 $R$을 생성한다.
- **텍스트 데이터 ($S$):** 주소(Chief Complaint) 텍스트에서 빈도가 10회 미만인 단어와 길이가 18단어를 초과하는 기록을 제거하여 노이즈를 줄이고 계산 복잡도를 낮춘다.

### 2. 모델 아키텍처

모델은 다음과 같은 구조를 가진다.

- **Encoder:** 단순한 Feedforward neural network로 구성되며, 입력된 sparse vector $R$을 128차원의 dense vector $z_{R}$으로 압축한다.
  $$z_{R} = f_{enc}(R)$$
- **Decoder:** 단일 레이어의 LSTM(Long Short-Term Memory) 네트워크를 사용한다.
  - 입력: 텍스트의 word embedding $x_t$와 Encoder에서 생성된 $z_{R}$을 사용한다.
  - 출력: 각 타임스텝 $t$에서 다음 단어의 확률 분포 $\hat{y}_t$를 Softmax 함수를 통해 예측한다.
  $$\hat{y}_t = \text{Softmax}(W_{out} h_t)$$

### 3. 학습 및 추론 절차

- **학습:** Encoder 부분은 오토인코더(autoencoder)로 사전 학습(pretrain)한 후, 전체 모델을 end-to-end로 학습시킨다. 손실 함수로는 Categorical Cross-Entropy를 사용한다.
  $$\mathcal{L}(\theta, R, S) = -\sum_{t=1}^{N} \log P(y_t | y_{<t}, R; \theta)$$
- **추론 (텍스트 생성):** 다음 세 가지 샘플링 기법을 비교 분석한다.
  - **Greedy Sampling:** 매 단계에서 가장 확률이 높은 단어를 선택한다.
  - **Probabilistic Sampling:** 확률 분포에 따라 단어를 선택하며, Temperature 파라미터를 통해 다양성을 조절한다.
  - **Beam Search:** 상위 $k$개의 후보 시퀀스를 유지하며 최적의 경로를 탐색한다.

## 📊 Results

### 1. 텍스트 품질 평가 (Translation Metrics)

- **지표:** BLEU, ROUGE(단문 특성에 맞게 수정됨), CIDEr, 그리고 Word Embedding 기반의 Cosine Similarity를 사용하였다.
- **결과:** **Greedy sampling**이 가장 높은 점수를 기록하였다. Beam search는 $k$ 값을 높여도 품질이 크게 향상되지 않았으며, Probabilistic sampling은 가장 낮은 성능을 보였다.

### 2. 역학적 타당성 (Epidemiological Validity)

- **단어 분포:** 성별에 따른 'preg(임신)' 단어의 출현 빈도나 연령대별 'overdose(과다복용)' 빈도가 실제 데이터와 매우 유사하게 생성됨을 확인하였다.
- **Odds Ratio (OR):** 고령자 집단에서 'fall(낙상)' 단어가 나타날 확률이 젊은 층보다 훨씬 높게 나타나는 경향이 합성 데이터에서도 유지되었다.
- **진단 예측 능력:** 합성된 주소 텍스트를 통해 진단 코드(CCS code)를 예측하는 GRU 분류기를 학습시킨 결과, 실제 텍스트를 사용했을 때보다 **합성 텍스트를 사용했을 때 F1-score가 약 5%p 향상**되었다. 이는 모델이 학습 과정에서 저빈도 노이즈를 제거하는 'denoising' 효과를 가졌음을 시사한다.

### 3. 개인 식별 정보(PII) 제거

- **평가 방법:** word2vec의 nearest neighbors를 이용해 실제 데이터에 존재하는 의사 이름 84개를 추출한 뒤, 합성 텍스트에 이 이름들이 포함되었는지 확인하였다.
- **결과:** 실제 데이터에서는 224건의 이름이 발견되었으나, **합성 데이터에서는 단 한 건의 이름도 발견되지 않았다 (0건)**.

## 🧠 Insights & Discussion

### 강점

본 모델은 의료 데이터의 이산적 특성을 자연어 문장으로 효과적으로 변환할 수 있음을 입증하였다. 특히, 저빈도 단어와 오타를 자동으로 제거함으로써 임상적으로 더 명확하고 정제된(denoised) 텍스트를 생성하는 부수적인 효과를 거두었다. 또한, PII를 완전히 제거하면서도 데이터의 통계적 특성을 유지한다는 점에서 데이터 공유 가능성을 높였다.

### 한계 및 비판적 해석

- **언어적 다양성 감소:** 모델이 확률이 높은 '전형적인(canonical)' 단어를 선택하는 경향이 있어, 실제 데이터가 가진 언어적 다양성이 훼손된다. 이는 새로운 질병의 징후를 포착해야 하는 **신드롬 감시(Syndromic Surveillance)** 목적의 연구에는 부적합할 수 있다.
- **연관성 증폭:** 특정 변수와 단어 간의 관계가 실제보다 더 강하게 나타나는 경향(예: 낙상과 고령자의 관계 증폭)이 발견되었다. 이는 가설 설정이나 탐색적 데이터 분석(EDA) 단계에서 왜곡된 결론을 내리게 할 위험이 있다.
- **복잡한 텍스트 생성의 어려움:** 본 연구는 비교적 짧은 '주소(Chief Complaint)'에 집중했으나, 더 길고 구조가 복잡한 '트리아제 노트(Triage notes)' 등을 생성하기 위해서는 더 발전된 아키텍처가 필요할 것이다.

## 📌 TL;DR

본 논문은 Encoder-Decoder 구조를 활용하여 EHR의 이산적 변수들로부터 현실적인 합성 주소(Chief Complaint) 텍스트를 생성하는 모델을 제안하였다. 제안된 모델은 개인 식별 정보(PII)를 완벽히 제거하면서도 실제 의료 데이터의 역학적 특성을 잘 보존하며, 오히려 텍스트의 노이즈를 제거하여 진단 예측 성능을 높이는 효과를 보였다. 이 연구는 향후 GAN과 결합하여 완전히 합성된(fully-synthetic) EHR 데이터셋을 구축함으로써, 의료 데이터 공유 및 머신러닝 연구의 가속화를 가능하게 할 것으로 기대된다.
