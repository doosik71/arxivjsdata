# LSTM-Autoencoder based Anomaly Detection for Indoor Air Quality Time Series Data

Yuanyuan Wei, Julian Jang-Jaccard, Wen Xu, Fariza Sabrina, Seyit Camtepe, Mikael Boulic

## 🧩 Problem to Solve

실내 공기질(IAQ) 데이터의 이상 탐지는 인간의 건강 및 웰빙과 밀접하게 연관되어 중요한 연구 분야입니다. 그러나 기존의 통계 및 얕은(shallow) 머신러닝 기반 이상 탐지 방법들은 여러 데이터 포인트 간의 상관관계(즉, 장기 의존성)를 포함하는 이상 징후를 효과적으로 탐지하지 못하는 한계가 있습니다. 특히, 실내 이산화탄소($\text{CO}_2$) 농도는 건강에 직접적인 영향을 미치며, 대규모 실시간 모니터링은 비용 문제와 데이터 품질 변동성으로 인해 이상 탐지가 더욱 어렵습니다.

## ✨ Key Contributions

- **하이브리드 딥러닝 모델 제안:** 시계열 데이터의 장기 의존성을 학습하는 Long Short-Term Memory (LSTM) 네트워크와 재구성 손실(reconstruction loss)을 기반으로 최적의 임계값을 식별하는 Autoencoder (AE)를 결합한 하이브리드 모델을 제안했습니다.
- **실제 데이터셋 적용 및 검증:** 뉴질랜드 학교에서 실제 배포된 SKOol MOnitoring BOx (SKOMOBO)를 통해 수집된 DunedinCO$_2$ 시계열 데이터셋에 제안 모델을 적용하고 검증했습니다.
- **우수한 성능 입증:** 포괄적인 평가 기준에 따라 다른 유사 모델들과 비교 실험한 결과, 99%를 초과하는 매우 높은 이상 탐지 정확도(99.50%)를 달성했습니다.

## 📎 Related Works

- **기존 통계 및 얕은 머신러닝 방법:**
  - Ottosen et al. [9]: k-Nearest Neighbour (kNN) 및 AutoRegressive Integrated Moving Average (ARIMA)를 사용하여 점(point) 및 문맥(contextual) 이상 징후를 개별적으로 탐지하고, k-means로 클러스터링.
  - Wei et al. [7]: Mean and Standard Deviation (MSD)으로 노이즈 데이터를 제거한 후 k-means를 적용하는 MSD-Kmeans 모델을 $\text{PM}_{10}$ 데이터에 사용.
  - Li et al. [10]: 클러스터링 기반 Fuzzy C-means를 사용하여 다변량 시계열 데이터의 이상을 탐지했으나, 고차원 데이터 구조 파악에 한계가 있음.
- **RNN/LSTM 기반 예측 및 이상 탐지:**
  - Sharma et al. [12]: Multi-Layer Perceptron (MLP), eXtream Gradient Boosting Regression (XGBR) 및 LSTM-wF (forget gate 없는 LSTM)로 IAQ를 예측하고 추정.
  - Mumtaz et al. [14]: IoT 센서 기반 LSTM 모델로 실내 공기 오염 물질 농도를 예측하고 이상 탐지 시 경고.
  - Wu et al. [8]: 산업용 IoT 시계열 이상 탐지를 위해 Stacked LSTM으로 예측 오차를 얻은 후 Gaussian Naive Bayes로 이상을 탐지하는 LSTM-Gauss-NBayes 접근법 제안.
- **LSTM-Autoencoder 변형 모델:**
  - Park et al. [18]: 로봇 보조 급식 시스템의 멀티모달 이상 탐지를 위해 LSTM-VAE (Variational Autoencoder) 하이브리드 모델 사용.
  - Liu et al. [19]: 여러 시간 스케일에 걸쳐 발생하는 이상을 식별하기 위해 VAE와 LSTM을 결합한 VAE-LSTM 모델 제안.
  - Trinh et al. [23]: LSTM-AE로 특징을 추출하고 재구성 오차를 계산한 후 Isolation Forest로 이상을 탐지하는 하이브리드 모델 제안.
  - Yin et al. [27]: Convolutional Recurrent Autoencoder 기반의 시계열 이상 탐지 (본 논문의 모델과 유사).

## 🛠️ Methodology

제안하는 LSTM-Autoencoder 모델은 시계열 데이터의 장기 의존성을 파악하고 재구성 오차를 통해 이상 징후를 탐지합니다.

1. **입력 시퀀스 데이터 준비 (Phase 1: To sequence):**
   - 원본 시계열 데이터셋을 고정된 $T$ 길이의 시간 윈도우 시퀀스 $[X_1, X_2, ..., X_n]$으로 변환합니다.
   - 각 시퀀스 $X$는 $T$개의 타임스텝 데이터를 포함하며, 본 연구에서는 10분 간격으로 수집된 $\text{CO}_2$ 샘플 10개를 $10 \times 1$ 크기의 2차원 배열로 재구성합니다.
2. **LSTM 인코더 (Phase 2: LSTM-AE training):**
   - 입력 시퀀스 $[x_1, ..., x_t]$는 LSTM 네트워크로 구성된 인코더에 입력됩니다.
   - 인코더는 여러 LSTM 셀을 통해 각 샘플의 장기 의존성을 학습하고, 고차원 입력 벡터를 저차원 잠재 공간(latent space) 표현으로 압축합니다.
   - 최종 LSTM 셀의 출력은 인코딩된 특징 벡터(예: $1 \times 16$)로 표현됩니다.
   - `RepeatVector` 레이어는 이 인코딩된 특징 벡터를 타임스텝 수($T$)만큼 복제하여 디코더로 전달합니다.
3. **LSTM 디코더 (Phase 2: LSTM-AE training):**
   - 디코더는 인코더의 `RepeatVector`에서 나온 복제된 인코딩 특징 벡터를 입력으로 받아, LSTM 네트워크를 통해 원본 시퀀스와 유사한 출력 $\hat{X}$를 재구성합니다.
   - `TimeDistributed Dense` 레이어를 통해 재구성된 출력이 원본 입력과 동일한 형태(예: $10 \times 1$)를 가지도록 합니다.
4. **이상 탐지 (Phase 3: Threshold setting & Phase 4: Anomaly detection on testing set):**
   - **재구성 손실 계산:** 모델은 출력 $\hat{X}_i$와 입력 $X_i$ 간의 차이인 재구성 손실을 최소화하도록 훈련됩니다. 본 연구에서는 Mean Absolute Error (MAE) 손실 함수를 사용합니다:
     $$L(MAE) = \frac{\sum_{i=1}^{n} |x_i - \hat{x}_i|}{n}$$
     여기서 $x_i$는 원본 입력, $\hat{x}_i$는 디코더의 출력, $n$은 샘플 수입니다.
   - **임계값 설정:** 훈련 세트(정상 범위 $\text{CO}_2$ 데이터만 포함)에서 계산된 모든 재구성 손실 중 최댓값을 이상 탐지 임계값 $\eta$로 설정합니다.
   - **이상 탐지:** 테스트 세트의 각 데이터 포인트에 대해 재구성 손실을 계산하고, 이 손실 값이 임계값 $\eta$보다 크면 해당 데이터 포인트를 이상 징후로 분류합니다.
     $$X'_i = \begin{cases} \text{anomalies} & \text{if } ltest_{arr}[i] > \eta \\ \text{normal} & \text{otherwise} \end{cases}$$
     여기서 $X'_i$는 재구성된 시계열의 데이터 포인트, $ltest_{arr}[i]$는 테스트 세트의 $i$번째 데이터 포인트에 대한 재구성 손실입니다.

## 📊 Results

- **훈련/검증 손실:** 훈련 및 검증 손실은 약 8 epoch 후에 안정화되어, 과적합이나 과소적합 없이 모델이 잘 학습되었음을 보여줍니다 (평균 손실률 약 0.07%).
- **모델 아키텍처 영향:**
  - 1개 히든 레이어(16개 LSTM 유닛) 모델이 F1-score 94.68%로 가장 높은 성능을 보였습니다.
  - 2개, 3개 히든 레이어 모델은 각각 93.48%, 93.31%의 F1-score를 기록했습니다.
- **시간 슬라이딩 윈도우 길이 영향:**
  - 시간 윈도우 길이 10에서 가장 높은 TPR (True Positive Rate)과 F1-score를 달성했습니다 (AUC-ROC 95.0%).
  - 윈도우 길이가 15와 20일 때 성능이 가장 낮았고, 25를 초과하면서 성능이 점진적으로 감소했습니다.
- **최종 모델 성능 (시간 윈도우 길이 10, 1개 히든 레이어, 16개 유닛):**
  - 전체 테스트 샘플 42,787개 중 정상 샘플 40,697개, 비정상 샘플 2,100개였습니다.
  - **정확도(Accuracy): 99.50%**
  - **정밀도(Precision): 100%** (정상을 비정상으로 오분류한 FP가 0개)
  - **재현율(Recall): 89.90%** (비정상 2,100개 중 1,888개 정확히 탐지)
  - **F1-score: 94.68%**
  - **AUC-ROC: 94.8%**
- **다른 유사 모델과의 비교:** DunedinCO$_2$ 데이터셋에서 제안 모델은 정확도(99.50%)와 정밀도(100%) 모두에서 다른 유사 모델들(Yin et al. [27], Nguyen et al. [28] 등)보다 우수한 성능을 보였습니다. 특히, 정밀도 100%는 정상 데이터를 이상으로 오탐지하는 경우가 없음을 의미합니다.

## 🧠 Insights & Discussion

- **장기 의존성 학습의 효율성:** LSTM과 Autoencoder의 결합 모델이 시계열 IAQ 데이터의 복잡한 장기 의존성을 성공적으로 학습하고, 이를 통해 기존 통계 및 얕은 머신러닝 모델의 한계를 극복했음을 보여줍니다.
- **높은 신뢰성의 이상 탐지:** 100%의 정밀도는 제안 모델이 오탐지(False Positive)를 전혀 발생시키지 않아, 실제 시스템에서 불필요한 경보를 최소화할 수 있는 강력한 장점을 가집니다. 이는 특히 $\text{CO}_2$와 같이 인간 건강에 중요한 요소에 대한 모니터링에서 매우 중요합니다.
- **재현율 개선의 필요성:** 비록 높은 전반적 성능을 보였지만, 89.90%의 재현율은 여전히 일부 실제 비정상 데이터를 정상으로 오분류(False Negative)하는 경우가 있음을 시사합니다. 향후 이 부분을 개선하기 위한 연구가 필요할 수 있습니다.
- **모델 아키텍처 및 하이퍼파라미터 최적화:** 히든 레이어 수와 시간 슬라이딩 윈도우 길이가 모델 성능에 미치는 영향을 체계적으로 분석하여, 데이터 특성에 맞는 최적의 모델 구성을 찾는 것이 중요함을 확인했습니다. 특히, 이 데이터셋에서는 10분 길이의 시간 윈도우가 가장 효과적이었습니다.
- **실제 적용 가능성:** 실제 환경에서 수집된 대규모 데이터를 사용한 실험은 제안 모델이 실제 환경에서 IAQ 이상 탐지 시스템으로 활용될 수 있는 잠재력이 높음을 입증합니다.
- **향후 연구:** 본 모델을 DDoS 공격 탐지와 같은 다른 시계열 분석 기반 이상 탐지 분야에 적용할 계획을 가지고 있습니다.

## 📌 TL;DR

**문제:** 실내 공기질(IAQ) 시계열 데이터에서 장기 의존성을 포함하는 이상 징후를 기존 방법으로는 효과적으로 탐지하기 어렵습니다.
**방법:** 시계열 데이터의 장기 의존성을 학습하는 LSTM 네트워크와 재구성 손실을 기반으로 이상 탐지 임계값을 설정하는 Autoencoder를 결합한 하이브리드 LSTM-Autoencoder 모델을 제안했습니다.
**결과:** 뉴질랜드 학교에서 수집된 실제 $\text{CO}_2$ 데이터셋에 적용한 결과, 99.50%의 높은 정확도, 100%의 정밀도, 94.68%의 F1-score를 달성하여 다른 유사 모델들보다 우수한 이상 탐지 성능을 입증했습니다.
