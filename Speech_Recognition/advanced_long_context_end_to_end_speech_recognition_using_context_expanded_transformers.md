# Advanced Long-context End-to-end Speech Recognition Using Context-expanded Transformers

Takaaki Hori, Niko Moritz, Chiori Hori, Jonathan Le Roux (2021)

## 🧩 Problem to Solve

본 논문은 강의나 대화와 같이 긴 오디오 녹음물을 처리해야 하는 end-to-end 자동 음성 인식(ASR) 시스템의 효율성과 정확도를 높이는 것을 목표로 한다.

대부분의 end-to-end ASR 모델은 개별 발화(independent utterances)를 인식하도록 설계되어 있다. 그러나 실제 환경에서는 화자 정보나 대화 주제와 같은 여러 발화에 걸친 문맥적 정보(contextual information)가 인식 성능을 향상시키는 데 매우 중요하다. 저자들은 이전 연구에서 여러 개의 연속된 발화를 동시에 입력받아 마지막 발화의 출력 시퀀스를 예측하는 context-expanded Transformer를 제안하여 성능 향상을 입증한 바 있다.

그럼에도 불구하고 기존의 context-expanded Transformer는 다음과 같은 세 가지 주요 문제점을 가지고 있다.

1. **아키텍처의 한계**: 최신 모델인 Conformer와 같은 고성능 아키텍처에서도 동일한 문맥 확장 효과가 나타나는지 확인되지 않았다.
2. **높은 계산 복잡도**: self-attention 메커니즘의 특성상 입력 시퀀스 길이에 따라 계산 복잡도가 제곱으로 증가하므로, 여러 발화를 입력으로 사용하는 문맥 확장 방식은 디코딩 시간을 크게 증가시킨다.
3. **실시간성 부족**: 현재의 방식은 스트리밍 ASR(streaming ASR)에 적용할 수 없어 온라인 애플리케이션으로의 활용이 어렵다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 문맥 확장 프레임워크를 유지하면서, 모델 아키텍처의 고도화와 디코딩 프로세스의 최적화를 통해 정확도, 속도, 실시간성을 동시에 확보하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Conformer 아키텍처 도입**: vanilla Transformer 대신 합성곱(convolution) 모듈이 추가된 Conformer를 적용하여 인식 정확도를 극대화하였다.
2. **Activation Recycling 기법 제안**: 슬라이딩 윈도우 방식의 디코딩 시, 이전 발화들에 대해 이미 계산된 hidden activation vector를 캐싱하여 재사용함으로써 계산량을 획기적으로 줄였다.
3. **Triggered Attention을 통한 스트리밍 구현**: CTC를 활용해 토큰 방출 타이밍을 예측하고 attention decoder를 활성화하는 triggered attention 기법을 도입하여, 긴 문맥을 활용하면서도 낮은 지연 시간의 스트리밍 인식을 가능하게 하였다.

## 📎 Related Works

본 연구는 저자들의 이전 연구인 context-expanded Transformer[19]를 기반으로 한다. 기존의 long-form ASR 연구들은 주로 attention 기반의 encoder-decoder나 RNN-T 모델에 의존해 왔으며, 특히 RNN-T는 추론 단계에서의 확장성에 집중하였다.

계산 효율성을 위한 activation recycling의 경우, 자연어 처리 분야의 Transformer-XL[29]에서 유사한 개념이 도입된 바 있다. 그러나 본 논문의 방식은 다음과 같은 차별점을 가진다.

- **단위의 차이**: Transformer-XL은 고정 크기의 텍스트 블록을 사용하지만, 본 모델은 ASR의 기본 처리 단위인 '발화(utterance)'를 기준으로 활성화를 재사용한다.
- **제약 조건**: Transformer-XL과 달리, encoder에서 발화 간의 backward self-attention을 금지하고, decoder에서 발화 내 source attention만을 적용하는 제약을 두어 ASR 특성에 맞게 최적화하였다.

## 🛠️ Methodology

### 1. Context-expanded Transformer 및 Conformer

기본적인 구조는 여러 개의 인접한 발화 $X_{v:u} = (X_v, \dots, X_{u-1}, X_u)$를 입력으로 받아 마지막 발화 $X_u$에 대한 최적의 토큰 시퀀스 $\hat{Y}_u$를 찾는 것이다.

목표 함수는 다음과 같이 정의된다:
$$\hat{Y}_u = \text{argmax}_{Y_u \in V^*} p(Y_u | Y_{v:u-1}, X_{v:u})$$

**Conformer 확장**:
기본 Transformer의 encoder 블록을 Conformer의 'sandwich structure'로 대체한다. 이 구조는 다음과 같은 순서로 구성된다:
$$\text{Half-step FFN} \rightarrow \text{MHSA} \rightarrow \text{Convolution Module} \rightarrow \text{Half-step FFN}$$
여기서 convolution module은 point-wise convolution, GLU activation, depth-wise convolution, batch normalization, Swish activation 등으로 구성되어 국소적인 특징 추출 능력을 높인다.

### 2. Activation Recycling

입력 시퀀스가 길어짐에 따라 발생하는 $O(N^2)$의 복잡도를 해결하기 위해, 이전 발화 $X_{v:u-1}$의 hidden activation $H^n_{v:u-1}$을 메모리에 저장해두고 재사용한다.

새로운 발화 $X_u$를 처리할 때, query는 현재 발화 $H^n_u$로 제한하고 key와 value는 캐싱된 $H^n_{v:u-1}$과 현재의 $H^n_u$를 모두 사용한다. 이를 통해 계산 복잡도는 $O(|H^n_{v:u}|^2)$에서 $O(|H^n_u| \times |H^n_{v:u}|)$로 감소한다. 이 기법을 적용하기 위해 위치 정보에 독립적인 **relative positional encoding**을 사용하며, 훈련 시 미래 발화의 정보를 참조하지 않도록 backward self-attention을 마스킹 처리한다.

### 3. Triggered Attention 및 스트리밍

실시간 인식을 위해 CTC를 사용하여 토큰 생성 시점을 예측하고 attention decoder를 트리거하는 방식을 사용한다. 훈련 단계에서 self-attention과 source-attention 레이어에 시간적 제약을 두는 마스킹을 적용하여 미래 문맥에 접근할 수 없는 상황을 시뮬레이션한다.

### 4. 학습 및 손실 함수

본 모델은 CTC와 Attention의 장점을 결합한 joint CTC-attention loss를 사용하여 학습한다:
$$L_u = -\alpha \log p^{trs}(Y^*_u | Y^*_{v:u-1}, X_{v:u}) - (1-\alpha) \log p^{ctc}(Y^*_u | X_{v:u})$$
여기서 $\alpha$는 두 손실 함수의 균형을 맞추는 스케일링 인자이다.

## 📊 Results

### 실험 설정

- **데이터셋**: HKUST(중국어, 200시간), Switchboard(영어, 300시간) 전화 대화 코퍼스.
- **지표**: CER(Character Error Rate), WER(Word Error Rate), RTF(Real-Time Factor).
- **비교 대상**: Baseline Transformer/Conformer, 문맥 확장 모델, ESPnet Conformer 등.

### 주요 결과

1. **인식 정확도**: 문맥 확장(Context-Ex.)을 적용했을 때, Transformer와 Conformer 모두에서 5%~13.5%의 상대적 에러 감소가 나타났다. 특히 **Context-Ex. Conformer**는 HKUST에서 **17.3% CER**, Switchboard에서 **12.0%/6.3% WER**을 기록하며 SOTA 성능을 달성하였다.
2. **디코딩 속도 (RTF)**: 문맥 확장 모델은 입력 길이가 길어 RTF가 baseline 대비 약 3배 증가하지만, **Activation recycling**을 적용하면 에러 증가 없이 디코딩 시간을 절반 이하로 단축할 수 있다.
3. **스트리밍 성능**: 스트리밍 환경에서 baseline 모델은 미래 정보의 부재로 에러가 크게 증가하는 경향이 있으나, 문맥 확장 모델은 이전 발화의 정보를 이미 가지고 있으므로 에러 증가폭(%inc)이 상대적으로 적게 나타났다.

## 🧠 Insights & Discussion

본 논문은 문맥 정보가 ASR의 정확도뿐만 아니라 스트리밍 환경에서의 견고함(robustness)에도 기여한다는 점을 시사한다. 스트리밍 ASR은 본질적으로 미래 정보를 사용할 수 없다는 제약이 있는데, 이때 이전 발화들로부터 축적된 문맥 정보가 미래 정보의 공백을 어느 정도 메워주는 역할을 한다고 해석할 수 있다.

또한, 계산 복잡도 문제를 해결하기 위해 제안된 activation recycling이 단순한 속도 향상을 넘어, 실제 서비스 가능한 수준의 RTF를 달성하게 함으로써 long-context 모델의 실용성을 입증하였다.

다만, 본 논문에서는 20~25초 정도의 제한된 윈도우 크기 내에서 문맥 확장을 수행하였으므로, 훨씬 더 긴 장기 의존성(long-term dependency)을 처리해야 하는 경우에 대한 분석은 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 긴 오디오 인식 성능을 높이기 위해 **문맥 확장(Context-expansion)** 기법을 **Conformer** 아키텍처에 적용하고, **Activation Recycling**과 **Triggered Attention**을 통해 계산 효율성과 스트리밍 가능성을 동시에 확보하였다. 결과적으로 HKUST 및 Switchboard 데이터셋에서 SOTA 수준의 정확도를 달성하였으며, 디코딩 시간을 50% 이상 단축하여 실용적인 long-context E2E ASR 시스템을 구축하였다.
