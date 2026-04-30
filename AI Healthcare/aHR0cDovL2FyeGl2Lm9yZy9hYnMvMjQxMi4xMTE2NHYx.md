# Missing data imputation for noisy time-series data and applications in healthcare

Lien P. Le, Xuan-Hien Nguyen Thi, Thu Nguyen, Michael A. Riegler, Pål Halvorsen, and Binh T. Nguyen (2024)

## 🧩 Problem to Solve

본 논문은 헬스케어 분야의 시계열 데이터에서 빈번하게 발생하는 노이즈(noise)와 결측치(missing values) 문제를 해결하고자 한다. 환자의 활동량을 모니터링하기 위해 사용되는 액티그래피(actigraphy)와 같은 웨어러블 기기 데이터는 센서 오류, 데이터 수집 중단 또는 사용자의 비협조로 인해 데이터가 누락되거나 불완전한 경우가 많다. 이러한 결측치를 적절히 채우는 Imputation 과정은 임상 연구의 정확한 분석을 위해 매우 필수적이다. 따라서 본 연구의 목표는 다양한 결측률(10%~80%) 환경에서 전통적인 통계 기반 방법과 최신 딥러닝 기반 Imputation 방법들의 성능을 비교 분석하고, 특히 Imputation 과정이 데이터의 노이즈를 제거하는 Denoising 효과를 제공하는지 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

첫째, 노이즈가 포함된 시계열 데이터에 대해 MICE-RF와 최신 딥러닝 기반 Imputation 기법들(SAITS, BRITS, Transformer)의 성능을 정량적으로 비교 분석하였다.

둘째, 특히 단변량 시계열 데이터에서 MICE-RF가 특정 결측률 이하에서 최신 딥러닝 기법들보다 우수한 성능을 보일 수 있음을 입증하였다.

셋째, 시계열 데이터를 특정 주기($T_{period}$)를 가진 행렬 형태로 재구성(reshape)하여 MICE-RF에 적용함으로써, 과거와 미래의 시점 정보를 동시에 활용하는 전략을 제시하였다. 이는 LSTM의 예측 전략과 유사한 맥락에서 시계열의 시간적 의존성을 효과적으로 포착하는 방법이다.

넷째, Imputation 알고리즘을 적용한 결과가 원본 데이터(Ground Truth)를 사용했을 때보다 하위 작업인 분류(Classification) 성능을 향상시키는 경우가 있음을 발견하였으며, 이를 통해 Imputation이 단순한 값 채우기를 넘어 Denoising 효과를 제공한다는 점을 시사하였다.

## 📎 Related Works

기존의 결측치 처리 방식으로는 결측치를 직접 처리하는 방법이나 단순 보간법이 사용되었다. 특히 Last Observation Carried Forward (LOCF)나 선형 보간법(linear interpolation)은 구현이 간단하여 자주 쓰이지만, 데이터에 편향(bias)을 일으키거나 부정확한 값을 생성할 가능성이 크다는 한계가 있다.

또한 K-Nearest Neighbors (KNN)나 MICE와 같은 고전적 통계 방법들이 널리 사용되어 왔으며, 최근에는 복잡한 패턴 모델링 능력을 갖춘 Generative Adversarial Networks (GANs)나 BRITS와 같은 딥러닝 기반 방법론들이 제안되었다. 본 논문은 이러한 기존 방법들 중 최신 딥러닝 모델들(SAITS, BRITS, Transformer)을 대조군으로 설정하여, 전통적인 MICE 방식에 Random Forest를 결합한 MICE-RF의 효율성을 재조명하였다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 및 MICE-RF의 구조

본 연구에서는 MICE-RF를 포함한 네 가지 알고리즘을 비교한다. 특히 MICE-RF의 경우, 단변량 시계열 데이터의 계절성(seasonality)을 활용하기 위해 데이터를 행렬 형태로 재구성하는 전처리를 수행한다.

시계열 데이터 $x^{(i)}$가 길이 $T_i$를 가진 벡터일 때, 이를 다음과 같은 행렬 $X^{(i)}$로 변환한다.

$$
X^{(i)} = \begin{pmatrix} 
x^{(i)}_1 & x^{(i)}_2 & \dots & x^{(i)}_{T_{period}} \\
x^{(i)}_{T_{period}+1} & x^{(i)}_{T_{period}+2} & \dots & x^{(i)}_{2T_{period}} \\
\vdots & \vdots & \ddots & \vdots \\
x^{(i)}_{(k-1)T_{period}+1} & x^{(i)}_{(k-1)T_{period}+2} & \dots & x^{(i)}_{T_i}
\end{pmatrix}
$$

여기서 $T_{period}$는 시계열의 주기이자 $T_i$의 약수여야 하며, $k = T_i / T_{period}$이다. 이렇게 재구성된 행렬에 MICE-RF를 적용하면, 각 열이 서로 다른 시점의 특성이 되고, 행 간의 관계를 통해 과거와 미래의 값을 참조하여 현재의 결측치를 추론할 수 있게 된다. Imputation 완료 후에는 다시 원래의 벡터 형태로 복원한다.

### 비교 대상 딥러닝 모델

1. **SAITS (Self-Attention-Based Imputation for Time Series):**
   Self-Attention 메커니즘을 통해 복잡한 시간적 관계를 포착한다. Diagonally Masked Self-Attention (DMSA) 블록과 Weighted Combination Block으로 구성된다. 학습 시에는 마스킹된 값을 맞추는 Masked Imputation Task (MIT)와 관측된 값을 복원하는 Observed Reconstruction Task (ORT)를 동시에 최적화하여 일관성을 유지한다.

2. **BRITS (Bidirectional Recurrent Imputation for Time Series):**
   양방향 RNN(LSTM)을 사용하여 전방향과 후방향 모두에서 시간적 관계를 학습한다. 결측치를 학습 가능한 파라미터로 취급하며, 양방향 예측값이 일치하도록 하는 일관성 제약(consistency constraint)을 손실 함수에 포함하여 학습한다.

3. **Transformer:**
   Self-Attention을 통해 국소적(local) 및 장기적(long-range) 의존성을 동시에 파악한다. 문맥 인식 벡터(context-aware vector)를 생성하여 결측치를 예측하며, 다양한 시뮬레이션 결측 패턴으로 학습하여 강건성을 확보한다.

## 📊 Results

### 실험 설정
- **데이터셋:** Psykose (조현병 예측, 단변량), Depresjon (우울증 예측, 단변량), HTAD (가정 내 활동 예측, 다변량) 세 가지를 사용하였다.
- **평가 지표:** Imputation 정확도는 Mean Absolute Error (MAE)로 측정하였으며, 실제 활용 가능성을 평가하기 위해 하위 분류 작업의 F1-score, AUC, MCC를 측정하였다.
- **비교 기준:** 결측률을 10%에서 80%까지 10% 단위로 증가시키며 실험하였으며, 각 데이터셋의 원본 논문에서 사용한 분류기(Logistic Regression, AdaBoost, KNN)를 베이스라인으로 사용하였다.

### 주요 결과
1. **MAE 관점:** Psykose와 Depresjon과 같은 단변량 데이터셋에서는 결측률 60% 미만일 때 MICE-RF가 가장 낮은 MAE를 기록하였다. 반면, 주기성이 없는 다변량 데이터셋인 HTAD에서는 SAITS가 가장 낮은 MAE를 유지하며 강건함을 보였다.
2. **분류 성능(Downstream Task) 관점:** 흥미롭게도 MAE가 가장 낮다고 해서 반드시 분류 성능이 가장 좋은 것은 아니었다. HTAD 데이터셋의 경우 MICE-RF가 MAE는 높았으나 하위 분류 작업에서는 가장 좋은 성적을 거두었다.
3. **Denoising 효과 발견:** Psykose 데이터셋의 결과(Table 1)를 보면, 결측률 10%~30% 구간에서 MICE-RF로 Imputation한 데이터의 F1-score(0.853~0.863)가 원본 데이터의 베이스라인(0.848)보다 높게 나타났다. 이는 Imputation 과정이 데이터의 노이즈를 제거하여 분류 모델이 더 깨끗한 특징을 학습하게 했음을 의미한다.

## 🧠 Insights & Discussion

본 논문은 통계 기반의 MICE-RF와 딥러닝 모델들의 트레이드오프를 명확히 보여준다. Random Forest의 앙상블 특성상 MICE-RF는 과적합(overfitting)에 강하며, 특히 데이터가 부족하거나 노이즈가 많은 환경에서 안정적인 성능을 보인다. 반면, 딥러닝 모델들은 유연성이 높지만 데이터가 희소할 때 과적합 위험이 크며, 하이퍼파라미터(레이어 수, $n_{steps}$ 등)에 매우 민감하게 반응한다.

또한, MICE-RF의 연산 비용 문제는 한계점으로 지적된다. 반복적인 Imputation 과정과 Random Forest의 특성상, 결측치가 많고 시계열이 길어질수록 계산 시간이 급격히 증가한다. 특히 $T_{period}$를 작게 설정할수록 생성되는 서브 시퀀스가 많아져 연산 부담이 커진다.

결론적으로, 시계열 데이터의 특성(단변량/다변량, 주기성 유무)에 따라 적절한 Imputation 전략을 선택해야 하며, 단순한 수치적 오차(MAE)보다는 최종 목적지인 분석/분류 작업의 성능을 기준으로 모델을 평가하는 것이 중요함을 시사한다.

## 📌 TL;DR

본 연구는 노이즈가 섞인 헬스케어 시계열 데이터의 결측치 처리를 위해 MICE-RF와 최신 딥러닝 기법들을 비교 분석하였다. 주기성이 있는 단변량 데이터에서는 MICE-RF가, 주기성이 없는 다변량 데이터에서는 SAITS와 같은 딥러닝 모델이 효과적임을 확인하였다. 특히, Imputation 결과가 원본 데이터보다 분류 성능을 향상시키는 현상을 통해 Imputation이 **Denoising(노이즈 제거)** 역할을 수행할 수 있음을 입증하였다. 이는 향후 불완전한 의료 데이터를 정제하고 분석하는 파이프라인 설계에 있어 중요한 근거가 될 수 있다.