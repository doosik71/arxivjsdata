# A Study of Enhancement, Augmentation, and Autoencoder Methods for Domain Adaptation in Distant Speech Recognition
Hao Tang, Wei-Ning Hsu, Franc ̧ois Grondin, James Glass

## 🧩 Problem to Solve
근거리 음성(close-talking speech)으로 학습된 음성 인식기(speech recognizer)가 원거리 음성(distant speech)에 대해 적용될 때 발생하는 성능 저하(최대 40% 절대적인 단어 오류율(WER) 증가) 문제를 해결하고자 합니다. 이 연구는 이 도메인 불일치를 줄이고, 원거리 음성 인식의 성능 저하 원인을 규명하며, 기존의 근거리 음성 인식기를 원거리 음성으로 적응시키는 다양한 방법론들을 도메인 적응 관점에서 체계적으로 비교하고 정량화하는 것을 목표로 합니다.

## ✨ Key Contributions
*   근거리-원거리 음성 인식 도메인 적응을 위한 여러 접근 방식(음성 향상, 다중 조건 훈련, 데이터 증강, 오토인코더)을 통제된 환경(AMI 데이터셋)에서 비교 및 평가했습니다.
*   두 도메인 간의 성능 격차를 정량화하고 특성을 분석했습니다.
*   다중 조건 훈련(Multi-condition training)이 가장 좋은 결과를 제공하지만, 모든 도메인에서 레이블된 데이터가 필요하다는 가장 엄격한 요구 사항이 있음을 확인했습니다.
*   음성 향상(Speech enhancement)이 두 번째로 효과적이며, 레이블되지 않은 병렬 데이터만으로도 상당한 WER 개선을 달성했습니다.
*   시뮬레이션된 잔향(reverberation)을 사용한 데이터 증강보다 오토인코더(FHVAE) 기반의 비지도 도메인 적응(unsupervised domain adaptation)이 더 나은 성능을 보여, 독립적인 레이블 없는 데이터만으로도 유망한 접근 방식임을 입증했습니다.
*   AMI 데이터셋에서 근거리 및 원거리 음성 간의 불일치가 잔향보다는 교차 통화(cross-talking)와 같은 다른 요인에 기인할 가능성이 높다는 시사점을 제시했습니다.

## 📎 Related Works
*   **일반적인 도메인 적응:** Domain adaptation 이론 [1, 2].
*   **화자 적응:** 화자 불일치 처리 기법 [3, 4].
*   **잡음 강인 음성 인식:** 음성 향상 기술 [6, 8, 14, 23, 24, 25, 26] 및 다중 조건 훈련 [7, 10, 27, 28].
*   **원거리 음성 인식:** 원거리 음성 인식의 어려움과 해결책 [9, 10, 11, 12, 13, 17, 18, 19].
*   **데이터 증강:** 이미지 분류 및 음성 인식 분야에서의 데이터 증강 적용 [15, 16, 29, 30, 31, 32].
*   **변분 오토인코더:** Factorized Hierarchical Variational Autoencoder (FHVAE)를 이용한 비지도 표현 학습 [33, 40].
*   **아키텍처 및 훈련:** Time-Delay Neural Networks (TDNNs) [34, 35], Long Short-Term Memory (LSTM) [38], Adam optimizer [39].
*   **잔향 시뮬레이션:** 이미지 방법(image method)을 이용한 잔향 생성 [37].

## 🛠️ Methodology
AMI (Affective Media Interaction) 데이터셋을 사용하여 근거리 마이크(IHM, Independent Headset Microphone)와 단일 원거리 마이크(SDM, Single Distant Microphone) 녹음을 비교했습니다.

1.  **특징 추출:** 80차원 로그 멜 필터뱅크(log Mel filterbank) 특징을 사용했습니다.
2.  **음향 모델:** 8개 계층의 시간 지연 신경망(TDNN)을 음향 모델로 사용했으며, 각 계층당 1000개의 은닉 유닛을 가집니다. SGD(Stochastic Gradient Descent)와 교차 엔트로피 손실로 훈련했습니다.
3.  **평가 지표:** 단어 오류율(WER)을 사용하여 성능을 측정했습니다.

다양한 도메인 적응 접근 방식을 평가했습니다:

*   **음성 향상 (Speech Enhancement):**
    *   TDNN 기반의 인코더-디코더 구조를 사용하여 SDM 특징으로부터 IHM 특징을 예측하도록 훈련했습니다.
    *   훈련된 모델은 SDM 음성을 향상(enhance)시킨 후, 이 향상된 데이터를 IHM으로 훈련된 음성 인식기의 입력으로 사용했습니다.
    *   병렬 데이터($\{(x_i, \tilde{x}_i)\}$)를 사용하여 $\tilde{x}_i$로부터 $x_i$를 근사하도록 목표($\min \|x_i - T(\tilde{x}_i)\|^2_2$)를 설정했습니다.

*   **다중 조건 훈련 (Multi-condition Training):**
    *   IHM과 SDM 데이터를 결합하여 단일 TDNN 음성 인식기를 훈련했습니다.

*   **데이터 증강 (Data Augmentation):**
    *   이미지 방법(image method) [37]을 사용하여 다양한 크기의 방에 대한 룸 임펄스 응답(RIRs, Room Impulse Responses)을 시뮬레이션했습니다.
    *   IHM 음성에 이 RIR을 무작위로 적용하여 시뮬레이션된 잔향이 포함된 IHM-r 데이터를 생성했습니다.
    *   TDNN을 IHM과 IHM-r 데이터를 결합하여 훈련했습니다.

*   **비지도 도메인 적응 (Unsupervised Domain Adaptation) - 오토인코더:**
    *   Factorized Hierarchical Variational Autoencoder (FHVAE) [33]를 사용하여 음성의 언어적 내용($z_1$)과 도메인 의존적 잡음 요인($z_2$)을 분리했습니다.
    *   FHVAE는 두 개의 인코더(공유 분포 및 도메인 특정 분포)와 하나의 디코더로 구성됩니다.
    *   훈련 후, 언어적 내용을 나타내는 공유 분포 인코더의 출력($\mu_1$, 선택적으로 $\log\sigma_1$)을 새로운 특징으로 추출했습니다.
    *   이 FHVAE 생성 특징을 사용하여 새로운 TDNN 음성 인식기를 훈련했습니다.

## 📊 Results
*   **기준선 (Baseline):** IHM 모델을 SDM에 적용했을 때 WER이 27.4%에서 70.3%로 크게 증가했습니다. SDM 모델을 SDM에 적용하면 49.7%였습니다.
*   **다중 조건 훈련 (IHM + SDM):** IHM에서 27.2%, SDM에서 45.3%의 WER을 달성하여 SDM 성능을 가장 크게 개선했습니다.
*   **데이터 증강 (IHM + IHM-r):** IHM+IHM-r로 훈련했을 때 SDM WER이 70.3%에서 63.3%로 소폭 개선되었습니다. 시뮬레이션된 잔향만으로는 전체 성능 저하를 설명하지 못했습니다.
*   **음성 향상 (Speech Enhancement):** SDM-e (향상된 SDM)에 대한 WER이 70.3%에서 54.2%로 크게 감소했습니다.
*   **비지도 도메인 적응 (FHVAE):** FHVAE에서 추출된 특징($\mu_1$)으로 훈련한 모델은 SDM에서 61.8%의 WER을 달성했습니다. 이는 시뮬레이션된 잔향을 사용한 데이터 증강보다 더 나은 결과입니다. 로그 분산($\log\sigma_1$)을 특징에 포함시키는 것은 오히려 성능을 저하시켰습니다(72.9%).

## 🧠 Insights & Discussion
*   **도메인 불일치 특성:** 근거리 및 원거리 음성 간의 성능 격차가 매우 크며, AMI 데이터셋에서는 잔향(reverberation)이 성능 저하의 주된 원인이 아닐 수 있다는 강력한 시사점을 얻었습니다. 교차 통화(cross-talking) 등 다른 요인들이 더 큰 영향을 미칠 수 있습니다.
*   **데이터 요구 사항과 성능:**
    *   **다중 조건 훈련**은 가장 뛰어난 성능을 보였지만, 두 도메인 모두에 레이블된 데이터가 필요하다는 가장 높은 데이터 요구 사항을 가집니다.
    *   **음성 향상**은 다음으로 좋은 성능을 보였고, 레이블되지 않은 병렬 데이터만으로도 가능하여 데이터 수집 비용이 비교적 적게 듭니다.
    *   **데이터 증강**은 생성 과정이 타겟 도메인의 조건을 충분히 반영할 경우 다중 조건 훈련에 필적할 잠재력이 있지만, 본 연구에서는 시뮬레이션된 잔향만으로는 한계가 있었습니다.
    *   **비지도 도메인 적응(FHVAE)**은 독립적인 레이블 없는 데이터만으로도 시뮬레이션된 잔향 데이터 증강보다 더 나은 결과를 달성하여 매우 유망한 접근 방식임을 보여주었습니다.

## 📌 TL;DR
**문제:** 근거리 음성으로 학습된 인식기는 원거리 음성에서 40% 이상의 큰 WER 저하를 보입니다.
**방법:** 이 연구는 음성 향상, 다중 조건 훈련, 데이터 증강 (시뮬레이션된 잔향), 그리고 FHVAE 기반의 비지도 도메인 적응 등 네 가지 주요 도메인 적응 방법을 AMI 데이터셋에서 체계적으로 비교 분석했습니다.
**핵심 발견:**
*   **다중 조건 훈련**이 SDM에서 45.3% WER로 가장 좋은 성능을 보였으나, 레이블된 타겟 도메인 데이터가 필요합니다.
*   **음성 향상**은 SDM에서 54.2% WER로 두 번째로 효과적이며, 레이블 없는 병렬 데이터만으로 가능합니다.
*   **FHVAE 기반 비지도 적응**은 SDM에서 61.8% WER을 달성하여, 독립적인 레이블 없는 데이터만으로도 시뮬레이션된 잔향 증강보다 유망한 결과를 보였습니다.
*   AMI 데이터셋의 성능 저하는 잔향보다는 **다른 요인(예: 교차 통화)** 때문일 가능성이 높습니다.