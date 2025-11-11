# RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series

Qingsong Wen, Jingkun Gao, Xiaomin Song, Liang Sun, Huan Xu, Shenghuo Zhu

## 🧩 Problem to Solve

시계열 데이터를 추세(trend), 계절성(seasonality), 잔차(remainder) 성분으로 분해하는 것은 이상 감지(anomaly detection) 및 예측(forecasting)에 매우 중요하지만, 기존 방법론들은 다음과 같은 실제 데이터의 복잡한 특성들을 제대로 다루지 못했습니다:

- 계절성 변동 및 이동(shift), 추세 및 잔차의 급격한 변화를 처리하는 능력 부족.
- 이상치(anomalies)를 포함한 데이터에 대한 견고성(robustness) 부족.
- 긴 계절성 주기(long seasonality period)를 가진 시계열 데이터에 적용하기 어려움 (예: IoT 데이터에서 1분 간격으로 수집된 하루 주기 데이터, $T=1440$).
- STL은 긴 주기와 높은 노이즈에 유연성이 떨어지고, 계절성 이동/변동 시 정확한 추출에 실패합니다.
- X-13-ARIMA-SEATS와 같은 방법은 소규모/중규모 데이터에만 적합하며, 긴 주기 데이터에는 확장성이 없습니다.
- STR은 급격한 추세 변화를 따라가지 못합니다.

## ✨ Key Contributions

- 복잡한 실제 시계열 데이터의 주요 과제(계절성 변동 및 이동, 급격한 추세 및 잔차 변화, 이상치)를 해결하는 새롭고 일반적인 시계열 분해 알고리즘 **RobustSTL**을 제안합니다.
- 희소 정규화(sparse regularization)를 포함하는 최소 절대 편차(Least Absolute Deviations, LAD) 손실을 사용하여 회귀 문제를 풀어 추세 성분을 견고하게 추출합니다.
- 추출된 추세를 바탕으로 비국소 계절 필터링(non-local seasonal filtering)을 적용하여 계절성 변동과 이동을 극복하고 계절성 성분을 견고하게 추출합니다.
- 긴 계절성 주기를 가진 시계열 데이터에도 효과적으로 적용 가능합니다.
- 합성 데이터 및 실제 데이터 실험을 통해 기존 최첨단 분해 알고리즘보다 우수한 성능을 입증했습니다.

## 📎 Related Works

- **고전적 방법 (X-11, X-11-ARIMA, X-13-ARIMA-SEATS):** 초기에는 견고성을 향상시켰으나, 긴 계절성 주기나 느리게 변하는 계절성에는 확장성이 제한됩니다.
- **TBATS (Trigonometric Exponential Smoothing State Space model with Box-Cox transformation, ARMA errors, Trend and Seasonal Components):** 복잡하고 비정수적인 계절성을 처리하지만, 긴 계절성 주기에 대한 높은 계산 비용 문제를 안고 있습니다.
- **Hodrick-Prescott 필터:** 간단하고 계산 비용이 낮지만, 추세와 긴 주기 계절성을 분해하지 못하며, 급격한 추세 변화에 취약합니다.
- **STR (Seasonal-Trend decomposition procedure based on Regression):** 추세, 계절성, 잔차의 동시 추출을 시도하며 계절성 이동에 유연하고, $L_1$-norm 정규화를 사용해 이상치에 견고하지만, 추세의 급격한 변화를 따라가지 못합니다.
- **SSA (Singular Spectrum Analysis):** 모델 프리(model-free) 방식이며 짧은 시계열에 잘 작동하지만, 강한 가정으로 인해 일부 실제 데이터셋에는 적용하기 어렵습니다.

## 🛠️ Methodology

RobustSTL 알고리즘은 다음의 4단계를 반복하여 추세, 계절성, 잔차를 추출합니다:

1. **노이즈 제거 (Noise Removal):**

   - 입력 시계열 $y_t$에 에지 보존(edge-preserving) 특성을 가진 양방향 필터링(bilateral filtering)을 적용하여 노이즈를 제거하고 $y'_t$를 얻습니다.
   - $$ y'_{t} = \sum_{j \in J} w\_{tj} y_j $$
   - 필터 가중치 $w_{tj}$는 시간적 거리와 값의 유사성을 모두 고려한 두 가우시안 함수의 곱으로 정의됩니다.
   - $$ w\_{tj} = \frac{1}{z} e^{-\frac{|j-t|^2}{2\delta^2_d}} e^{-\frac{|y_j-y_t|^2}{2\delta^2_i}} $$

2. **추세 추출 (Trend Extraction):**

   - 노이즈 제거된 $y'_t$에 계절 차분(seasonal difference) $\nabla_T y'_t = y'_t - y'_{t-T}$를 적용하여 계절성 효과를 완화합니다.
   - 다음 목적 함수를 최소화하여 추세의 1차 차분 $\nabla\tau_t$를 추출합니다.
   - $$ \min*{\nabla\tau} \sum*{t=T+1}^N |g*t - \sum*{i=0}^{T-1} \nabla\tau*{t-i}| + \lambda_1 \sum*{t=2}^N |\nabla\tau*t| + \lambda_2 \sum*{t=3}^N |\nabla^2\tau_t| $$
     - 첫 번째 항은 이상치에 견고한 LAD ($L_1$-norm) 손실을 사용하여 경험적 오류를 측정합니다.
     - 두 번째 항과 세 번째 항은 각각 1차 및 2차 차분 연산자 제약 조건으로, 추세가 느리게 변화하거나 급격한 수준 변화를 보일 수 있으며, 조각별 선형(piecewise linear)임을 가정합니다.
   - 이 문제는 선형 프로그래밍으로 변환하여 효율적으로 해결되며, 이를 통해 상대 추세 $\tilde{\tau}^r_t$를 얻습니다.
   - $y''_{t} = y'_{t} - \tilde{\tau}^r_t$로 신호를 업데이트합니다.

3. **계절성 추출 (Seasonality Extraction):**

   - $y''_{t}$에 비국소 계절 필터링(non-local seasonal filtering)을 적용하여 계절성 $s_t$를 추출합니다.
   - $$ \tilde{s}_t = \sum_{(t',j) \in \Omega} w_t^{(t',j)} y''\_j $$
   - 가중치 $w_t^{(t',j)}$는 현재 시점 $t$와 이전 $K$개 계절 주기 이웃($t-kT$) 내의 $2H+1$개 지점 $j$ 간의 시간적 거리 및 값 유사성에 따라 결정됩니다. 이는 계절성 이동에 적응하고 이상치에 견고하게 합니다.

4. **최종 조정 (Final Adjustment):**
   - 한 주기 내의 모든 계절성 성분의 합이 0이 되도록 조정합니다.
   - $\hat{\tau}_1 = \frac{1}{\lfloor N/T \rfloor} \sum_{t=1}^{\lfloor N/T \rfloor} \tilde{s}_t$를 계산합니다.
   - 추세 및 계절성 추정치를 다음과 같이 업데이트합니다: $\hat{s}_t = \tilde{s}_t - \hat{\tau}_1$, $\hat{\tau}_t = \tilde{\tau}^r_t + \hat{\tau}_1$.
   - 잔차는 $\hat{r}_t = y_t - \hat{s}_t - \hat{\tau}_t$로 얻습니다.

- 이러한 1~4단계는 추정치의 정확도가 수렴할 때까지 반복됩니다.

## 📊 Results

- **합성 데이터:**
  - RobustSTL은 급격한 추세 변화, 계절성 이동, 이상치 및 노이즈가 포함된 합성 데이터에서 추세, 계절성, 잔차를 성공적으로 분리했습니다.
  - 다른 비교 알고리즘(Standard STL, TBATS, STR)에 비해 추세 및 계절성 성분 모두에서 훨씬 낮은 평균 제곱 오차(MSE)와 평균 절대 오차(MAE)를 달성했습니다 (예: 추세 MSE에서 RobustSTL은 0.0530, STR은 1.6761).
- **실제 데이터 (슈퍼마켓 매출, 파일 교환 횟수):**
  - RobustSTL은 실제 데이터셋에서도 부드러운 계절성 신호를 추출하며, 변화하는 패턴과 계절성 이동에 잘 적응했습니다.
  - 추출된 추세 신호는 급격한 변화를 신속하게 포착하고 이상치에 강건했습니다. 스파이크 이상치는 잔차 신호에 잘 보존되었습니다.
  - Standard STL과 STR은 급격한 추세 변화를 따라가지 못했고, TBATS는 추세가 이상치에 영향을 받는 경향을 보였습니다.
- **계산 효율성:** RobustSTL은 $L_1$-norm 정규화를 이용한 최적화 문제로 효율적으로 해결될 수 있어 TBATS 및 STR보다 훨씬 빠른 계산 속도를 보였습니다. (STR은 긴 계절성 주기에서 계산 시간이 길어 일부 결과가 보고되지 않음).

## 🧠 Insights & Discussion

- RobustSTL은 추세의 급격한 변화, 불규칙한 계절성(이동 및 변동 포함), 스파이크/딥 이상치를 효과적이고 효율적으로 처리합니다. 이는 특히 긴 계절성 주기를 가진 실제 복잡한 시계열 데이터에 중요합니다.
- LAD 손실과 희소 정규화를 사용하여 추세의 급격한 변화를 보존하면서 견고하게 추세를 추출하는 것이 핵심입니다.
- 비국소 계절 필터링은 계절성 이동에 적응하고 이상치에 강건하게 대응하는 데 효과적입니다.
- 반복적인 접근 방식은 분해의 정확도를 향상시킵니다.
- 알고리즘의 효율성은 대규모 IoT 데이터에 적용 가능하게 합니다.
- 향후 연구는 시계열 분해를 이상 감지와 직접 통합하여 더욱 견고하고 정확한 감지 결과를 제공하는 방향으로 진행될 수 있습니다.

## 📌 TL;DR

**문제:** 기존 시계열 분해 방법들은 계절성 이동, 급격한 추세 변화, 이상치, 긴 계절성 주기와 같은 실제 데이터의 복잡한 특성들을 처리하는 데 한계가 있습니다.
**방법:** RobustSTL은 양방향 필터링을 통한 노이즈 제거, LAD 손실과 $L_1$-norm 희소 정규화를 사용한 견고한 추세 추출 (선형 프로그래밍으로 해결), 비국소 계절 필터링을 통한 유연하고 견고한 계절성 추출, 그리고 최종 조정 단계를 반복하는 알고리즘을 제안합니다.
**결과:** RobustSTL은 합성 및 실제 데이터셋에서 기존 최첨단 방법(STL, TBATS, STR)보다 우수한 성능을 보였으며, 긴 주기, 급격한 변화 및 이상치를 포함하는 시계열을 정확하고 효율적으로 분해합니다.
