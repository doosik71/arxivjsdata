# Time Series Forecasting With Deep Learning: A Survey

Bryan Lim, Stefan Zohren

## 🧩 Problem to Solve

이 논문은 다양한 도메인의 시계열 데이터셋을 다루기 위해 개발된 수많은 딥 러닝 아키텍처를 종합적으로 검토하는 것을 목표로 합니다. 특히, 단일 예측 및 다중 예측(multi-horizon forecasting)에 사용되는 일반적인 인코더 및 디코더 설계, 시간 정보를 통합하는 방법, 통계 모델과 신경망 구성 요소를 결합한 하이브리드 모델의 최근 개발 동향, 그리고 딥 러닝이 시계열 데이터 기반 의사 결정을 지원하는 방법에 대해 설명합니다.

## ✨ Key Contributions

- **시계열 딥 러닝 아키텍처 포괄적 분류:** CNN, RNN (특히 LSTM), 어텐션 메커니즘을 포함한 주요 인코더 설계와 예측 출력 (점 추정, 확률론적 예측) 및 손실 함수를 체계적으로 설명합니다.
- **다중 예측 방법론 분석:** 반복적(iterative) 방법과 직접적(direct) 방법을 구분하여 다중 예측 문제에 대한 딥 러닝 접근 방식을 제시합니다.
- **하이브리드 딥 러닝 모델 조명:** 통계 모델의 도메인 지식과 딥 러닝의 유연성을 결합한 하이브리드 모델(예: ES-RNN, Deep State Space Models)이 순수 방법론보다 뛰어난 성능을 보이는 이유와 방식을 설명합니다.
- **의사 결정 지원을 위한 딥 러닝:** 시계열 데이터에서 모델의 예측을 해석하고(interpretability) 반사실적 예측(counterfactual prediction)을 생성하여 의사 결정 과정을 지원하는 딥 러닝의 역할을 강조합니다.
- **미래 연구 방향 제시:** 연속 시간(continuous-time) 모델 및 계층적(hierarchical) 모델과 같은 딥 러닝 시계열 예측 분야의 유망한 미래 연구 방향을 제시합니다.

## 📎 Related Works

- **전통적인 시계열 모델:** AR (Autoregressive) \[6], 지수 평활 (Exponential Smoothing) \[7,8], 구조적 시계열 모델 (Structural Time Series Models) \[9].
- **머신 러닝 모델:** 커널 회귀 (Kernel Regression) \[19], 서포트 벡터 회귀 (Support Vector Regression) \[20], 가우시안 프로세스 (Gaussian Processes) \[21,22].
- **딥 러닝 기반 시계열 모델:**
  - **CNN 기반:** WaveNet (dilated convolutions) \[32], TCN (Temporal Convolutional Network) \[33].
  - **RNN 기반:** LSTM (Long Short-Term Memory) \[44], DeepAR \[37], Deep State Space Models \[38], Recurrent Neural Filters \[39].
  - **어텐션 기반:** Transformer 아키텍처 \[49,53,54], Temporal Attention Learning \[52].
- **하이브리드 모델:** ES-RNN (M4 대회 우승) \[64], Deep Hybrid Model for Weather Forecasting \[66].
- **모델 해석 가능성 (Interpretability):** LIME (Local Interpretable Model-Agnostic Explanations) \[71], SHAP (Shapley Additive Explanations) \[72], Saliency Maps \[73,74], Influence Functions \[75].
- **반사실적 예측 및 인과 추론 (Counterfactual Predictions & Causal Inference):** IPTW (Inverse-Probability-of-Treatment-Weighting) 기반 접근법 \[81], G-computation 프레임워크 확장 \[82], 도메인 적대적 학습 (Domain Adversarial Training) \[83].

## 🛠️ Methodology

이 논문은 시계열 예측을 위한 딥 러닝 방법론을 인코더-디코더 프레임워크를 중심으로 분류하고 설명합니다.

1. **시계열 딥 러닝의 기본 구성 요소:**
   - **인코더 ($g_{enc}$):** 과거의 시계열 정보 ($y_{t-k:t}, x_{t-k:t}, s$)를 잠재 변수 $z_t$로 인코딩합니다.
     - **CNN (Convolutional Neural Networks):** 인과적 컨볼루션(causal convolutions) 및 확장 컨볼루션(dilated convolutions)을 사용하여 시간 불변적(time-invariant)이고 지역적인(local) 패턴을 학습합니다. 수용 필드(receptive field) 크기를 통해 과거 정보를 통합합니다.
       $$ (W\*h)^{(l,t,d*l)} = \sum*{\tau=0}^{\lfloor k/d*l \rfloor} W^{(l,\tau)}h^l*{t-d_l\tau} $$
     - **RNN (Recurrent Neural Networks):** 내부 메모리 상태 $z_t$를 재귀적으로 업데이트하여 과거 정보를 요약합니다. LSTM은 게이트 메커니즘을 통해 장기 의존성 문제를 해결합니다.
       $$ z*t = \nu(z*{t-1}, y_t, x_t, s) $$
     - **어텐션 메커니즘 (Attention Mechanisms):** 동적으로 생성된 가중치 $\alpha(\kappa_t, q_\tau)$를 사용하여 과거의 중요한 시점에 직접적으로 초점을 맞춥니다.
       $$ h*t = \sum*{\tau=0}^{k} \alpha(\kappa*t, q*\tau)v\_{t-\tau} $$
   - **디코더 ($g_{dec}$):** 인코딩된 잠재 변수 $z_t$를 사용하여 최종 예측 $\hat{y}_{t+1}$을 생성합니다.
     - **출력 및 손실 함수:**
       - **점 추정 (Point Estimates):** 이진 분류를 위한 이진 교차 엔트로피 손실 ($L_{classification}$), 연속 값 회귀를 위한 평균 제곱 오차 손실 ($L_{regression}$).
       - **확률론적 예측 (Probabilistic Outputs):** 가우시안 분포의 평균 $\mu(t,\tau)$ 및 분산 $\zeta(t,\tau)^2$과 같은 예측 분포의 파라미터를 출력하여 불확실성을 모델링합니다.
         $$ y\_{t+\tau} \sim \mathcal{N}(\mu(t,\tau), \zeta(t,\tau)^2) $$
2. **다중 예측 모델:**
   - **반복적 방법 (Iterative Methods):** 자기회귀(autoregressive) 방식으로 예측된 표본을 다음 예측 단계의 입력으로 재귀적으로 사용하여 다중 예측을 생성합니다.
   - **직접적 방법 (Direct Methods):** 시퀀스-투-시퀀스(sequence-to-sequence) 아키텍처를 사용하여 모든 가용한 입력으로부터 직접적으로 전체 예측 구간의 값을 생성합니다.
3. **하이브리드 모델 (Hybrid Models):**
   - **비확률론적 하이브리드:** 통계 모델의 예측 방정식에 딥 러닝 출력을 결합하여 파라미터를 동적으로 학습합니다 (예: ES-RNN).
   - **확률론적 하이브리드:** 딥 러닝을 사용하여 가우시안 프로세스나 선형 상태 공간 모델과 같은 확률론적 생성 모델의 파라미터를 생성합니다 (예: Deep State Space Models).
4. **의사 결정 지원:**
   - **해석 가능성 (Interpretability):** LIME, SHAP, Saliency Map과 같은 사후 해석(post-hoc) 기법과 어텐션 가중치를 활용한 내재적 해석(inherent interpretability)을 통해 모델 예측의 근거를 설명합니다.
   - **반사실적 예측 및 인과 추론 (Counterfactual Predictions & Causal Inference):** 시간 의존적 교란 변수(time-dependent confounding)를 조정하는 딥 러닝 방법을 사용하여 다양한 행동이 시계열 궤적에 미치는 영향을 평가하고 시나리오 분석을 수행합니다.

## 📊 Results

이 서베이 논문은 새로운 실험 결과를 제시하기보다는 기존 연구들의 핵심 성과를 종합합니다.

- **다양한 아키텍처의 효과 입증:** CNN, RNN, 어텐션 기반 모델들이 다양한 시계열 데이터셋에서 효과적으로 시간 정보를 통합하고 예측 성능을 개선했음을 보여줍니다.
- **하이브리드 모델의 우수성:** M4 시계열 예측 대회에서 ES-RNN과 같은 하이브리드 딥 러닝 모델이 순수 통계 또는 순수 머신러닝 모델보다 우수한 예측 성능을 달성했음을 강조합니다. 이는 도메인 지식과 딥 러닝의 결합이 갖는 시너지를 입증합니다.
- **불확실성 정량화:** 딥 러닝 모델이 확률론적 예측을 통해 예측의 불확실성을 정량화할 수 있음을 보여주며, 이는 의사 결정자들이 위험을 관리하는 데 유용합니다.
- **의사 결정 지원 강화:** 어텐션 메커니즘을 통한 모델의 해석 가능성과 반사실적 예측 기법이 시계열 데이터 기반의 의사 결정 과정을 효과적으로 지원할 수 있음을 입증합니다.

## 🧠 Insights & Discussion

- **딥 러닝의 강력함:** 딥 러닝은 수동적인 피처 엔지니어링 없이도 시계열 데이터의 복잡한 시간적 역학을 데이터 기반 방식으로 학습할 수 있는 강력한 도구입니다. 이는 방대한 데이터와 컴퓨팅 자원이 가용할 때 특히 두드러집니다.
- **하이브리드 모델의 실용성:** 딥 러닝의 유연성과 통계 모델의 도메인 지식을 결합한 하이브리드 접근 방식은 과적합 위험을 줄이고, 정지성(stationarity) 및 비정지성(non-stationarity) 구성 요소를 분리하며, 낮은 데이터 체제(low data regimes)에서도 우수한 성능을 발휘하여 실제 응용에서 매우 유용합니다. M4 대회의 결과는 이를 명확히 보여줍니다.
- **예측을 넘어선 가치:** 딥 러닝은 단순히 미래 값을 예측하는 것을 넘어, 모델의 예측 근거를 설명하는 해석 가능성(interpretability)과 다양한 가상 시나리오에 대한 영향을 평가하는 반사실적 예측(counterfactual prediction)을 제공하여 의사 결정 지원 시스템으로서의 가치를 확장합니다.
- **남아있는 과제:**
  - **불규칙한 시계열 데이터:** 딥 러닝 모델은 일반적으로 정기적인 간격으로 이산화된 시계열 데이터를 요구하므로, 관측치가 누락되거나 불규칙한 간격으로 발생하는 데이터셋에 적용하기 어렵습니다. 뉴럴 상미분 방정식(Neural Ordinary Differential Equations)과 같은 연속 시간 모델이 유망한 해결책으로 제시됩니다.
  - **계층적 시계열 모델링:** 시계열 데이터는 종종 논리적 그룹화가 있는 계층적 구조를 가집니다 (예: 지역별 제품 판매량). 이러한 계층 구조를 명시적으로 고려하는 아키텍처 개발은 예측 성능을 더욱 향상시킬 수 있는 중요한 연구 방향입니다.

## 📌 TL;DR

이 서베이 논문은 시계열 예측을 위한 딥 러닝의 현재 상태를 종합적으로 검토합니다. **문제**는 다양한 딥 러닝 아키텍처, 하이브리드 모델, 의사 결정 지원 메커니즘을 정리하고 분석하는 것입니다. **제안된 방법**은 CNN, RNN, 어텐션 기반 인코더와 점/확률론적 디코더를 포함한 핵심 빌딩 블록, 반복적/직접적 다중 예측 전략, 통계 모델과 신경망을 결합한 하이브리드 접근 방식, 그리고 해석 가능성 및 반사실적 예측을 통한 의사 결정 지원 도구를 분류하고 설명하는 것입니다. **주요 결과**는 딥 러닝이 강력한 데이터 기반 예측을 제공하며, 하이브리드 모델이 도메인 지식과의 결합을 통해 성능을 향상시키고, 어텐션 메커니즘이 장기 의존성 학습과 해석 가능성을 높인다는 것입니다. 향후 과제로는 불규칙한 데이터 처리를 위한 연속 시간 모델과 계층적 시계열 모델링이 남아있습니다.
