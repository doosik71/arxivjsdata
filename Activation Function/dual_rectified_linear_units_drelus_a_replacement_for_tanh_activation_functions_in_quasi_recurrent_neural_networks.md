# Dual Rectified Linear Units (DReLUs): A Replacement for Tanh Activation Functions in Quasi-Recurrent Neural Networks

Frédéric Godin, Jonas Degrave, Joni Dambre, Wesley De Neve

## 🧩 Problem to Solve

최신 신경망에서 Rectified Linear Unit (ReLU)은 학습 속도 향상, 기울기 소실 문제 완화, 희소 활성화 유도 등의 이점으로 널리 사용됩니다. 그러나 ReLU는 양수 이미지(unbounded positive image)만 가지며 음수 값을 표현할 수 없어 순환 신경망(RNN)의 은닉 상태(hidden state) 업데이트에 사용하기 어렵습니다. 이는 RNN에서 은닉 상태가 지수적으로 커지거나(간단한 RNN의 경우), 특정 값을 정확히 빼는 것이 불가능하여 표현력이 제한되는(QRNN의 경우) 문제를 야기합니다. 반면, `tanh`와 `sigmoid`와 같은 활성화 함수는 유계(bounded)이므로 RNN에서 주로 사용되지만, 기울기 소실 문제에 취약하여 깊은 네트워크 학습을 어렵게 합니다. 이 논문은 `tanh`의 장점(양수 및 음수 이미지)과 ReLU의 장점(기울기 소실 완화, 희소성)을 모두 가지면서 QRNN에서 `tanh` 활성화 함수를 대체할 수 있는 새로운 활성화 함수를 제시하는 것을 목표로 합니다.

## ✨ Key Contributions

- **Dual Rectified Linear Unit (DReLU) 소개:** `tanh` 활성화 함수를 성공적으로 대체할 수 있는 새로운 신경망 구성 요소인 DReLU를 제안합니다. DReLU는 QRNN에서 사용될 때 양수 및 음수 이미지를 모두 가지면서도 기울기 소실에 덜 민감합니다.
- **QRNN 실험 재현 및 확장:** 기존 Quasi-Recurrent Neural Network (QRNN) 논문(Bradbury et al., 2017)의 감성 분류 및 단어 수준 언어 모델링 실험을 독립적으로 재현하고, 추가적으로 문자 수준 언어 모델링 작업을 평가합니다.
- **깊은 스택 학습 능력 입증:** DReLU 기반 QRNN 레이어를 최대 8개까지 쉽게 쌓을 수 있으며, 이는 스킵 연결(skip connections) 없이도 가능함을 보여줍니다. 이를 통해 문자 수준 언어 모델링에서 기존 LSTM 기반의 얕은 아키텍처보다 뛰어난 최신(state-of-the-art) 결과를 달성합니다.
- **Dual Exponential Linear Unit (DELU) 소개:** DReLU의 지수 확장 버전인 DELU도 함께 제시합니다.

## 📎 Related Works

- **Rectified Linear Units (ReLUs):** Nair & Hinton (2010)이 처음 소개했으며, 빠른 실행, 기울기 소실 완화, 희소성 유도 등의 장점으로 ResNets (He et al., 2016) 및 DenseNets (Huang et al., 2017)와 같은 성공적인 피드포워드 신경망 아키텍처에 널리 사용되었습니다. 그러나 RNN에서는 문제점이 있습니다.
- **Parametric ReLUs (PReLUs):** He et al. (2015)이 음수 영역에서 작은 값으로 입력 값을 곱하여 ReLU의 "죽은 뉴런" 문제를 해결했습니다.
- **Exponential Linear Units (ELUs):** Clevert et al. (2016)이 제안한 활성화 함수로, ReLU보다 개선된 성능을 보이며 바이어스 변화(bias shift)를 자연스럽게 해결하고 노이즈에 강합니다.
- **Recurrent Neural Networks (RNNs) 및 Long Short-Term Memory (LSTMs):** Elman (1990)과 Hochreiter & Schmidhuber (1997)에 의해 제안되었으며, 가변 길이 시퀀스 모델링에 강점을 가집니다. 일반적으로 `sigmoid`나 `tanh`와 같은 유계 활성화 함수를 사용합니다.
- **Quasi-Recurrent Neural Networks (QRNNs):** Bradbury et al. (2017)이 제안한 하이브리드 RNN으로, CNN과 LSTM 기반의 단순화된 순환 단계를 결합하여 계산 속도를 크게 향상시켰습니다. QRNN은 은닉-은닉 행렬 곱셈을 제거하여 무한정 성장하는 은닉 상태 문제를 완화합니다.

## 🛠️ Methodology

이 논문은 QRNN의 `tanh` 활성화 함수를 DReLU로 대체하는 방법을 제안합니다.

1. **ReLU의 RNN 적용 문제점 분석:**

   - **간단한 RNN:** $h_t = g(W \cdot h_{t-1} + U \cdot x_t + b)$ 에서 $g$가 ReLU와 같은 비유계 함수일 경우, 가중치 행렬의 가장 큰 고유값이 1보다 크면 은닉 상태 $h_t$가 지수적으로 커져 폭주하는 기울기(exploding gradients) 문제를 야기합니다.
   - **LSTM:** 셀 상태(cell state)와 은닉 상태(hidden state)가 이전 상태에 행렬 곱셈을 통해 의존하므로, 비유계 활성화 함수를 사용하면 유사한 문제가 발생할 수 있습니다.
   - **QRNN의 셀 상태 문제:** QRNN은 은닉-은닉 행렬 곱셈을 제거하여 위 문제를 부분적으로 해결했지만, 후보 셀 상태 $\tilde{c}_t$에 ReLU를 적용하면 오직 양수 값만 더해질 수 있으므로 이전 셀 상태 $c_{t-1}$에서 값을 뺄 수 없게 되어 표현력이 제한됩니다. `tanh`는 음수 이미지도 가지므로 이 문제가 없습니다.

2. **Dual Rectified Linear Unit (DReLU) 정의:**

   - `tanh`의 양수 및 음수 이미지 특성과 ReLU의 기울기 소실 완화 및 희소성 특성을 모두 가지기 위해 두 개의 일반 ReLU 출력을 빼는 방식의 DReLU를 제안합니다.
   - 수학적 정의: $f_{DReL}(a,b) = \max(0,a) - \max(0,b)$
   - **특징:**
     - 양수 및 음수 이미지를 모두 가집니다.
     - 정확히 0이 될 수 있습니다. 이는 희소성을 유도하고 노이즈에 강하게 만듭니다.
     - ReLU와 유사하게 활성화될 때 기울기의 크기를 증폭시키거나 감소시키지 않아 기울기 소실에 덜 민감합니다.
     - 계산이 빠릅니다.
   - QRNN의 후보 셀 상태 $\tilde{c}_t$ 계산에 DReLU를 적용합니다:
     $\tilde{c}_t = \max(0, U_{c,1} \cdot x_{t-n+1:t}) - \max(0, U_{c,2} \cdot x_{t-n+1:t})$

3. **Dual Exponential Linear Unit (DELU) 정의:**
   - ELU의 개선된 성능을 DReLU에 적용한 것으로, DReLU의 부드러운 버전으로 볼 수 있습니다.
   - 수학적 정의: $f_{DEL}(a,b) = f_{EL}(a) - f_{EL}(b)$ ($f_{EL}$은 ELU 함수)

## 📊 Results

- **감성 분류 (IMDb 데이터셋):**

  - 밀집 연결(Dense Connected)된 LSTM 및 QRNN(`tanh`)과 비교했을 때, DReLU/DELU 기반 QRNN은 유사한 정확도를 보입니다 (91.0-91.2%).
  - 스킵 연결이 없는 경우, DReLU 기반 QRNN(91.0%)은 `tanh` 기반 QRNN(90.5%)보다 높은 정확도를 달성하며, `tanh` 기반 QRNN이 스킵 연결 없이 정확도가 크게 저하되는 문제를 DReLU는 완화합니다.
  - DReLU 기반 QRNN은 LSTM보다 2.5배 빠른 속도를 보입니다.

- **단어 수준 언어 모델링 (Penn Treebank):**

  - DReLU/DELU 기반 QRNN은 `tanh` 기반 QRNN(80.0 test perplexity)을 능가하는 성능(78.4 및 78.5 test perplexity)을 보입니다.
  - 단일 ReLU를 사용한 QRNN은 훨씬 낮은 성능(85.3 test perplexity)을 보여 DReLU가 `tanh`의 우수한 대체제임을 입증합니다.
  - DReLU 기반 QRNN은 Zaremba et al. (2014)의 중간 크기 LSTM(82.7 test perplexity)보다 우수합니다.
  - DReLU는 `tanh`보다 훨씬 더 많은 희소 활성화를 유도합니다 (DReLU: 53.90% 거의 0, `tanh`: 10.02% 거의 0).

- **문자 수준 언어 모델링 (Penn Treebank, enwik8/Hutter Prize):**
  - **Penn Treebank:** DReLU/DELU 기반 QRNN은 `tanh` 기반 QRNN보다 우수한 BPC (Bits-Per-Character) 점수를 달성합니다. 8개 층의 DReLU/DELU-QRNN (500 은닉 유닛)은 1.21 BPC를 기록하며, 유사한 파라미터 수를 가진 HyperLSTM을 능가하는 새로운 최신 결과입니다.
  - **Hutter Prize (enwik8):** 더 도전적인 데이터셋에서 8개 층의 DReLU-QRNN (1000 은닉 유닛)은 1.25 BPC를 달성하며, ByteNet (1.31) 및 Recurrent Highway Networks (1.27) 등 복잡한 아키텍처를 능가하는 최신 성능을 보입니다. 4개 층의 DReLU-QRNN도 HyperLSTM과 유사하거나 더 나은 성능을 보입니다.

## 🧠 Insights & Discussion

- **`tanh`의 성공적인 대체:** DReLU 및 DELU는 QRNN에서 `tanh` 활성화 함수를 성공적으로 대체할 수 있는 유효한 드롭인(drop-in) 솔루션입니다. 이들은 `tanh`와 같이 양수 및 음수 활성화를 모두 가지면서도, ReLU와 같이 활성화될 때 기울기의 크기를 감소시키지 않아 기울기 소실 문제에 덜 민감합니다.
- **희소성과 노이즈 강건성:** DReLU는 정확히 0이 될 수 있어 `tanh`보다 더 많은 희소 활성화를 유도합니다. 이는 대규모 신경망의 효율적인 학습을 가능하게 하며, 노이즈에 대한 강건성을 높입니다.
- **깊은 네트워크 학습:** DReLU의 기울기 소실 완화 특성은 스킵 연결 없이도 QRNN 레이어를 깊게 쌓는 것을 가능하게 하며, 이는 복잡한 시퀀스 모델링 작업에서 최신 성능을 달성하는 데 기여합니다.
- **성능 및 효율성:** DReLU 기반 QRNN은 LSTM과 유사하거나 더 나은 성능을 달성하면서도 훨씬 빠른 학습 속도를 제공합니다. 이는 특히 `tanh` 기반 QRNN이 스킵 연결 없이는 성능이 저하되는 상황에서 두드러집니다.
- **단일 ReLU의 한계:** 후보 셀 상태 $\tilde{c}_t$에 단일 ReLU를 사용하는 것은 QRNN의 표현력을 심각하게 제한하며, DReLU가 `tanh`를 대체하는 데 있어 훨씬 우월한 선택임을 실험적으로 보여줍니다.

## 📌 TL;DR

**문제:** QRNN에서 `tanh` 활성화 함수는 기울기 소실에 취약하며, ReLU는 양수 값만 출력하여 표현력이 제한되어 `tanh`를 직접 대체하기 어렵습니다.
**방법:** 본 논문은 양수 및 음수 이미지(unbounded positive and negative image)를 모두 가지며, 정확히 0이 될 수 있고, 기울기 소실에 덜 민감한 새로운 활성화 함수인 Dual Rectified Linear Unit (DReLU)과 그 지수 확장인 Dual Exponential Linear Unit (DELU)을 제안합니다. 이들은 QRNN의 후보 셀 상태 $\tilde{c}_t$ 계산에 `tanh`를 대체하여 사용됩니다.
**결과:** DReLU/DELU 기반 QRNN은 감성 분류, 단어 수준 언어 모델링, 문자 수준 언어 모델링 등 세 가지 NLP 태스크에서 `tanh` 기반 QRNN 및 LSTM과 유사하거나 더 나은 성능을 달성했습니다. 특히 DReLU는 깊은 QRNN 스택(최대 8개 층)을 가능하게 하며, 문자 수준 언어 모델링에서 최신(state-of-the-art) 결과를 기록했습니다. 이는 DReLU가 `tanh`의 장점과 ReLU의 장점을 결합하여 깊은 순환 신경망 학습의 효율성과 성능을 개선할 수 있음을 입증합니다.
