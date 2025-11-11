# ANOMALYBERT: SELF-SUPERVISED TRANSFORMER FOR TIMESERIES ANOMALY DETECTION USING DATA DEGRADATION SCHEME

Yungi Jeong, Eunseok Yang, Jung Hyun Ryu, Imseong Park, Myungjoo Kang

## 🧩 Problem to Solve

다변량 시계열 데이터(예: 센서 값, 네트워크 데이터)에서 기계적 결함으로 인해 발생하는 이상 징후를 감지하는 것은 중요한 문제이지만, 훈련 데이터에 이상치 레이블이 없는 경우가 많아 어렵습니다. 특히, 시간적 맥락과 변수 간의 상호 관계를 동시에 이해하여 예상치 못한 동작과 여러 변수와 관련된 이상 징후를 효과적으로 감지하는 것이 난제입니다.

## ✨ Key Contributions

- **자기 지도 학습 기반의 시계열 이상 감지 모델 제안**: BERT에서 영감을 받아 시계열 데이터에 적합한 데이터 변형 기법(data degradation scheme)을 활용한 Transformer 기반의 자기 지도 학습 모델 `AnomalyBERT`를 제안합니다.
- **새로운 데이터 변형 기법 도입**: 입력 데이터의 일부를 4가지 유형의 합성 이상치(Soft replacement, Uniform replacement, Length adjustment, Peak noise) 중 하나로 대체하는 기법을 제안합니다. 이 기법은 기존의 정의된 5가지 유형의 이상치(global, contextual, shapelet, seasonal, trend)를 효과적으로 포괄합니다.
- **Transformer 아키텍처 개선**: 시계열 데이터의 시간적 정보를 반영하기 위해 1D 상대 위치 바이어스(relative position bias)를 Transformer의 Multi-Head Self-Attention (MSA) 모듈에 적용합니다.
- **최첨단 성능 달성**: 5가지 실제 벤치마크 데이터셋(SWaT, WADI, MSL, SMAP, SMD)에서 기존 최첨단 방법들을 뛰어넘는 F1-score 성능을 달성하여 복잡한 시계열 내 이상치 감지 능력을 입증합니다.

## 📎 Related Works

- **시계열 이상 감지**: LOF, DAGMM(GMM + DNN), LSTM-VAE(LSTM + VAE), OmniAnomaly(GRU + VAE), GDN(Graph Neural Network), Anomaly Transformer 등 RNN, 그래프 신경망, Transformer를 활용한 다양한 딥러닝 기반 접근법들이 연구되었습니다.
- **Transformer 및 변형 모델**: NLP 분야에서 성공을 거둔 Transformer (Vaswani et al., 2017)와 그 변형 모델들(BERT, T5, SpanBERT, XLNet, BART)에서 아이디어를 얻었습니다. 특히 BERT의 마스크드 언어 모델링(MLM)은 본 논문의 데이터 변형 기법에 영감을 주었습니다. Vision Transformer (ViT)와 같이 컴퓨터 비전 분야에도 Transformer가 성공적으로 적용되고 있습니다.

## 🛠️ Methodology

`AnomalyBERT`는 선형 임베딩 계층(linear embedding layer), Transformer 본체(Transformer body), 예측 블록(prediction block)의 세 부분으로 구성됩니다.

1. **선형 임베딩 계층**: 입력 다변량 시계열 윈도우 $X = x_{t_{0}:t_{1}} \in \mathbb{R}^{N \times D}$의 각 데이터 패치 $x_{t:t+p}$를 임베딩 특징 $f_i$로 투영합니다.
2. **Transformer 본체**: 모든 임베딩 특징 $\{f_i\}_{1 \le i \le M}$를 입력받아 잠재 특징(latent features) $\{h_i\}_{1 \le i \le M}$를 생성합니다. 이 Transformer는 Multi-Head Self-Attention (MSA) 모듈과 MLP 블록으로 구성된 계층을 포함하며, 위치 정보를 주입하기 위해 정현파 위치 인코딩(sinusoidal positional encodings) 대신 1D 상대 위치 바이어스(relative position bias) $B = [b_{i,j}] \in \mathbb{R}^{M \times M}$를 사용합니다.
   $$ \text{Attention}(Q, K, V) = \text{SoftMax}\left( \frac{QK^{T}}{\sqrt{d}} + B \right)V $$
    여기서 $b_{i,j} = \hat{b}_{j-i}$는 학습 가능한 바이어스 테이블 $\hat{B}$에서 가져옵니다.
3. **데이터 변형 기법(Data Degradation Scheme)**: 훈련 단계에서 레이블이 없는 데이터를 활용하기 위해, 입력 윈도우 $X$의 무작위 구간 $[t'_{0}, t'_{1}]$을 다음 4가지 합성 이상치 중 하나로 대체하여 degraded input을 생성합니다.
   - **Soft replacement**: 윈도우 외부의 시퀀스와 가중 합으로 대체.
   - **Uniform replacement**: 상수 시퀀스로 대체.
   - **Length adjustment**: 시퀀스 길이를 늘리거나 줄임.
   - **Peak noise**: 단일 피크 값 추가.
4. **훈련**: 모델은 degraded input을 받아 이진 교차 엔트로피 손실(binary cross entropy loss)을 사용하여 degraded 부분을 정상/이상으로 분류하도록 훈련됩니다.
   $$ L = - \frac{1}{N} \sum*{t=t*{0}}^{t*{1}} \left( \mathbf{1}*{[t'_{0},t'_{1}]}(t) \cdot \log a*t + (1 - \mathbf{1}*{[t'_{0},t'_{1}]}(t)) \cdot \log(1 - a*t) \right) $$
   여기서 $\mathbf{1}*{[t'_{0},t'_{1}]}(t)$는 합성 레이블의 역할을 합니다.

## 📊 Results

- **벤치마크 성능**: AnomalyBERT는 SWaT, WADI, MSL, SMAP, SMD 5개 벤치마크 데이터셋에서 F1-score 기준으로 모든 기존 방법을 능가하는 성능을 보였습니다. 특히 MSL 및 SMAP 데이터셋에서 뛰어난 성능을 발휘했습니다.
- **합성 이상치의 영향**:
  - Soft replacement는 기존의 5가지 전형적인 이상치 유형을 모두 커버하는 강력한 효과를 보였습니다.
  - Uniform replacement와 Peak noise는 Soft replacement를 보완하여 성능 향상에 기여합니다.
  - Length adjustment는 SWaT 데이터셋에서는 성능을 향상시켰지만, WADI 데이터셋에서는 오히려 저하시켜 데이터셋의 특성(예: 다양한 주파수 포함 여부)에 따라 그 영향이 달라짐을 보여주었습니다.
- **시각화**: t-SNE를 사용한 시각화 결과, AnomalyBERT는 비정상 데이터를 정상 데이터와 효과적으로 분리하는 잠재 특징을 학습함을 보여주었습니다.

## 🧠 Insights & Discussion

- AnomalyBERT는 자기 지도 학습과 새로운 데이터 변형 기법을 통해 레이블이 없는 복잡한 시계열 데이터에서 이상 징후를 성공적으로 감지할 수 있음을 입증했습니다.
- Transformer 아키텍처에 1D 상대 위치 바이어스를 적용한 것이 시계열 데이터의 시간적 맥락을 효과적으로 포착하는 데 중요함을 시사합니다.
- 다양한 유형의 합성 이상치를 혼합하여 사용하는 것이 단일 유형만 사용하는 것보다 모델의 감지 능력을 향상시킬 수 있음을 보여줍니다. 이는 합성 이상치가 실제 이상 징후의 다양한 측면을 포괄적으로 모방하는 데 도움이 됩니다.
- 데이터셋의 특성(예: 주파수 분포)에 따라 특정 유형의 합성 이상치(예: Length adjustment)의 적용 여부를 조절해야 할 필요성이 있습니다.
- 향후 연구에서는 실제 이상 징후를 더 자연스럽게 모방하거나 데이터 특성에 맞는 적절한 이상치 유형을 혼합하는 등 변형 알고리즘을 개선하여 모델 성능을 더욱 향상시킬 잠재력이 있습니다.

## 📌 TL;DR

AnomalyBERT는 레이블 없는 시계열 이상 감지를 위해 데이터 변형 기법을 활용한 자기 지도 Transformer 모델입니다. 4가지 유형의 합성 이상치를 도입하여 입력 데이터의 일부를 대체하고, 1D 상대 위치 바이어스를 갖춘 Transformer로 이상 여부를 분류합니다. 5가지 실제 벤치마크에서 기존 SOTA를 뛰어넘는 성능을 달성했으며, 이 자기 지도 학습 방식이 복잡한 시계열의 이상 징후를 효과적으로 학습하고 감지함을 보여줍니다.
