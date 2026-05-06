# Efficient conformer-based speech recognition with linear attention

Shengqiang Li, Menglong Xu, Xiao-Lei Zhang (2021)

## 🧩 Problem to Solve

본 논문은 최근 음성 인식 분야에서 뛰어난 성능을 보이는 Conformer 모델의 계산 복잡도와 모델 크기 문제를 해결하고자 한다. Conformer는 CNN과 Transformer를 결합하여 지역적 및 전역적 의존성을 동시에 모델링할 수 있다는 장점이 있으나, 핵심 구성 요소인 Dot-product self-attention의 계산 복잡도가 입력 특징(input feature) 길이 $T$에 대해 제곱 비례($O(T^2)$)한다는 치명적인 단점이 있다.

이러한 제곱 복잡도는 긴 음성 입력 시 과도한 GPU 메모리 사용과 학습 시간 증가를 초래한다. 또한, 모델의 파라미터 수가 많아 실제 배포 환경에서의 효율성이 떨어진다는 점이 문제로 제기된다. 따라서 본 연구의 목표는 Conformer의 성능을 최대한 유지하면서 계산 복잡도를 선형 수준($O(T)$)으로 낮추고, 모델 파라미터 수를 획기적으로 줄인 효율적인 음성 인식 모델인 LAC(Linear Attention based Conformer)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Conformer의 연산 병목 지점인 Self-attention과 Feed-forward 네트워크를 효율적인 구조로 교체하는 것이다.

1. **Multi-head Linear Self-Attention (MHLSA) 도입**: 기존의 Dot-product attention에서 행렬 곱셈의 순서를 변경함으로써 계산 복잡도를 $O(T^2)$에서 $O(T)$로 낮추었다. 이는 상관관계 행렬(correlation matrix)을 명시적으로 계산하지 않고도 유사한 기능을 수행하도록 설계되었다.
2. **Low-rank Matrix Factorization 기반 LFFN 설계**: Feed-forward 모듈의 거대한 가중치 행렬을 두 개의 작은 행렬의 곱으로 분해하는 Low-rank matrix factorization을 적용하여, 성능 저하를 최소화하면서 파라미터 수를 약 50% 감소시켰다.
3. **CTC objective와의 결합**: LAC 모델의 학습 및 추론 단계에서 Connectionist Temporal Classification (CTC) 목적 함수를 함께 사용하여 전체적인 인식 성능을 향상시켰다.

## 📎 Related Works

논문은 기존의 End-to-End 음성 인식 접근 방식을 CTC, RNN Transducer, Attention-based encoder-decoder의 세 가지로 분류하며, 특히 Transformer 구조의 효율성을 높이려는 시도들을 언급한다.

* **계산 복잡도 감소 시도**: Fujita 등은 Lightweight Dynamic Convolution을 사용하여 attention을 대체하였고, Performer(FAVOR+ 방식)는 정규 직교 랜덤 특징(positive orthogonal random features)을 통해 선형 복잡도를 달성하였다. 또한, Local Dense Synthesizer Attention (LDSA)은 attention 가중치의 길이를 제한하여 복잡도를 줄였다.
* **모델 경량화 시도**: QuartzNet은 1D separable convolution을 사용하여 모델 크기를 줄였으며, Low-rank transformer는 행렬 분해를 통해 압축을 시도하였다. ContextNet은 Squeeze-and-excitation 모듈을 통해 전역 문맥 정보를 효율적으로 통합하였다.

LAC는 이러한 선행 연구들에서 영감을 얻어, 특히 객체 검출 분야에서 제안된 Efficient Attention을 Multi-head 버전으로 확장하여 Conformer에 적용했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

LAC는 CNN frontend, Encoder 스택, Decoder 스택으로 구성된다. CNN frontend는 입력 음성 특징을 추출하며, 이후 여러 층의 LAC encoder block을 통과한다.

### 1. CNN Frontend 및 위치 인코딩

입력 특징은 두 개의 CNN 레이어와 임베딩 레이어를 거쳐 $X_{emb} \in \mathbb{R}^{T \times d_e}$가 된다. 기존 Conformer는 Relative position encoding을 사용하지만, LAC의 MHLSA는 상관관계 행렬을 계산하지 않으므로 절대 위치 인코딩(Absolute position encoding)을 사용한다. 위치 인코딩 $P$는 다음과 같이 사인 및 코사인 함수로 정의된다.

$$p_{i,2j} = \sin\left(\frac{i}{10000^{2j/d_e}}\right), \quad p_{i,2j+1} = \cos\left(\frac{i}{10000^{2j/d_e}}\right)$$

최종적으로 입력값은 $X_{pos} = X_{emb} + P$가 되어 인코더로 전달된다.

### 2. LAC Encoder Block

각 인코더 블록은 두 개의 Low-rank Feed Forward (LFFN) 모듈이 MHLSA 모듈과 Convolution 모듈을 감싸고 있는 형태이다. 연산 순서는 다음과 같다.

1. $\tilde{X}_i = X_i + \frac{1}{2} \text{LFFN}(X_i)$
2. $X'_i = \tilde{X}_i + \text{MHLSA}(\tilde{X}_i)$
3. $X''_i = X'_i + \text{Conv}(X'_i)$
4. $Y_i = \text{Layernorm}(X''_i + \frac{1}{2} \text{LFFN}(X''_i))$

### 3. Low-rank Feed Forward Module (LFFN)

기존의 FFN은 두 개의 선형 변환 $W_1 \in \mathbb{R}^{d \times d_{ff}}$와 $W_2 \in \mathbb{R}^{d_{ff} \times d}$를 사용한다. LAC는 이를 Low-rank 분해를 통해 다음과 같이 구현한다.

$$\text{LFFN}(X) = \text{Dropout}(\text{Swish}(XE_1D_1))E_2D_2$$

여기서 $E_1 \in \mathbb{R}^{d \times d_{bn}}$, $D_1 \in \mathbb{R}^{d_{bn} \times d_{ff}}$이며, $d_{bn}$은 병목(bottleneck) 차원이다. 이 방식은 파라미터 수를 $d \times d_{ff}$에서 $d_{bn} \times (d + d_{ff})$로 획기적으로 줄인다.

### 4. Multi-head Linear Self-Attention (MHLSA)

기존의 Dot-product attention은 $\text{softmax}(QK^T/\sqrt{d_k})V$ 순으로 계산되어 $O(T^2)$의 복잡도를 가진다. 반면 MHLSA는 행렬 곱셈의 결합 법칙을 이용하여 순서를 변경한다. $h$번째 헤드의 출력은 다음과 같다.

$$\text{LinearAtt}(Q_h, K_h, V_h) = \sigma_{row}(Q_h d_k^{1/4}) (\sigma_{col}(K_h d_k^{1/4})^T V_h)$$

여기서 $\sigma_{row}$와 $\sigma_{col}$은 각각 행과 열 방향으로 적용되는 Softmax 함수이다. 이렇게 하면 $(K^TV)$를 먼저 계산하여 $O(T d_k^2)$의 선형 복잡도로 연산이 가능하다.

## 📊 Results

### 실험 설정

* **데이터셋**: AISHELL-1 (중국어, 170시간), LibriSpeech (영어, 970시간).
* **특징 추출**: 80-channel log-mel filterbank coefficients.
* **모델 설정**: 인코더 12블록, 디코더 6블록, Attention 헤드 4개, Hidden dimension 256, FFN output dimension 2048.
* **학습**: Adam optimizer, SpecAugment 적용, CTC weight 0.3 (학습) / 0.6 (디코딩).

### 주요 결과

1. **인식 성능**:
    * **AISHELL-1**: CER 5.02% (Test set)를 달성하여 Conformer(4.88%)와 매우 근소한 차이를 보이며, 다른 7개 모델보다 우수한 성능을 보였다.
    * **LibriSpeech**: 'dev-clean' 2.1%, 'test-other' 2.3%의 WER을 기록하여 Conformer와 경쟁 가능한 수준의 성능을 입증하였다.
2. **효율성**:
    * **파라미터 수**: Conformer 대비 약 50% 수준으로 감소하였다 (AISHELL-1 기준 45.15M $\rightarrow$ 22.83M).
    * **학습 속도**: Conformer보다 AISHELL-1에서는 1.23배, LibriSpeech에서는 1.18배 빨라졌다.
3. **Ablation Study**:
    * 병목 차원 $d_{bn}$이 100일 때 모델 크기와 성능 사이의 최적의 트레이드오프가 나타났다.
    * LFFN을 일반 FFN으로 교체하면 파라미터가 급증하고, MHLSA를 MHSA로 교체하면 복잡도가 다시 $O(T^2)$로 증가함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Conformer의 강력한 성능을 유지하면서도 실용적인 효율성을 확보하는 방법을 제시하였다. 특히 MHLSA를 통해 $T$가 매우 긴 음성 데이터에서도 메모리 부족 문제 없이 효율적인 연산이 가능함을 보였다. 또한 LFFN을 통한 파라미터 압축이 성능에 미치는 영향이 매우 적다는 점은, 음성 인식 모델의 FFN 층에 상당한 중복성(redundancy)이 존재함을 시사한다.

다만, absolute position encoding을 사용함으로써 얻는 이득이 relative position encoding을 포기함으로써 잃는 정보보다 큰지에 대한 심층적인 분석은 부족하다. 또한, 제안된 선형 attention이 모든 데이터셋에서 일관되게 dot-product attention과 동일한 수준의 성능을 낼 수 있는지에 대한 추가 검증이 필요할 수 있다.

## 📌 TL;DR

본 연구는 Conformer의 계산 복잡도와 모델 크기를 줄이기 위해 **Multi-head Linear Self-Attention (MHLSA)**와 **Low-rank Feed Forward (LFFN)** 모듈을 제안한 **LAC** 모델을 개발하였다. 이를 통해 계산 복잡도를 $O(T^2)$에서 $O(T)$로 낮추고 파라미터 수를 50% 절감했음에도 불구하고, AISHELL-1과 LibriSpeech 데이터셋에서 Conformer에 근접하는 높은 성능을 유지하였다. 이 연구는 실시간 음성 인식 시스템이나 리소스가 제한된 환경에서의 Conformer 적용 가능성을 크게 높였다는 점에서 가치가 있다.
