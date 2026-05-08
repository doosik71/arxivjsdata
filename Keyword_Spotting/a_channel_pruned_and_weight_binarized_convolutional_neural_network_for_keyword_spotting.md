# A Channel-Pruned and Weight-Binarized Convolutional Neural Network for Keyword Spotting

Jiancheng Lyu and Spencer Sheen (2019)

## 🧩 Problem to Solve

본 논문은 모바일 폰과 같이 자원이 제한된 플랫폼에서 동작해야 하는 Keyword Spotting(KWS, 키워드 검출) 작업의 효율성을 높이기 위해, 합성곱 신경망(Convolutional Neural Network, CNN)의 복잡도를 줄이는 문제를 다룬다.

신경망의 파라미터 수를 줄이면서도 모델의 성능(정확도)을 유지하는 것은 매우 중요하며, 특히 실시간으로 동작해야 하는 음성 인식 기반의 KWS 시스템에서는 계산 비용과 메모리 사용량을 최소화하는 것이 필수적이다. 따라서 본 연구의 목표는 Channel Pruning(채널 가지치기)과 Weight Binarization(가중치 이진화)을 결합하여, 성능 손실을 최소화하면서 모델의 크기를 획기적으로 줄인 슬림한 CNN을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 채널 수 감소를 위한 **RGSM(Relaxed Group-wise Splitting Method)**과 가중치의 정밀도를 1비트로 낮추는 **Weight Binarization**을 통합한 3단계 훈련 절차를 제안한 것이다.

단순히 가중치에 페널티를 부여하는 방식보다 효율적인 변수 분할 방식(Variable Splitting Method)을 그룹 희소성(Group Sparsity) 문제에 적용하여, 모델 성능의 하락을 $0.25\%$ 이내로 억제하면서도 $50\%$ 이상의 채널 희소성(Channel Sparsity)을 달성하였다.

## 📎 Related Works

논문에서는 채널 가지치기를 위해 통계학에서 널리 사용되는 **Group Lasso (GL)** 페널티를 언급한다. GL은 그룹 단위로 가중치를 0으로 만들어 구조적 희소성을 유도하는 방법이다. 하지만 저자들은 KWS를 위한 CNN 학습에 GL을 직접적으로 적용(Gradient Descent 기반의 직접 최적화)했을 때, 희소성을 구현하는 효율이 매우 떨어진다는 점을 발견하였다.

또한, 가중치 양자화의 일종인 **BinaryConnect (BC)**와 그 개선 버전인 **Blended version**을 소개한다. 이는 가중치를 $\pm 1$의 이진 값으로 표현하여 연산 속도를 높이는 기법이다. 본 논문은 이러한 기존의 개별적인 최적화 기법들을 유기적으로 결합하여 단계별로 적용하는 전략을 취함으로써 기존 방식과의 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 파이프라인 (Three-Stage Procedure)

모델 최적화는 다음의 세 단계로 진행된다.

- **Stage I (Channel Pruning):** RGSM을 사용하여 적절한 채널 희소성을 확보한다. 이 과정에서 정확도가 다소 하락할 수 있다.
- **Stage II (Float Weight Retraining):** Stage I에서 제거된 채널을 제외하고, 남은 채널의 가중치(32-bit float)를 재학습시켜 손실된 정확도를 회복한다.
- **Stage III (Weight Binarization):** Stage II의 가중치를 초기값(Warm start)으로 사용하여 가중치를 1비트 이진 값으로 변환하고 최적화한다.

### 2. RGSM (Relaxed Group-wise Splitting Method)

채널 가지치기를 위해 Group Lasso 페널티를 사용하며, 이를 효율적으로 풀기 위해 다음의 라그랑지안 함수(Lagrangian function)를 교대로 최소화한다.

$$L_{\beta}(u, w) = \ell(w) + \mu P(u) + \frac{\beta}{2} \|w - u\|_2^2$$

여기서 $\ell(w)$는 손실 함수, $P(u)$는 Group Lasso 페널티, $\beta$는 정규화 파라미터이다.

- **$u$-업데이트:** Group Lasso의 근접 연산자(Proximal Operator)를 사용하여 닫힌 형태(Closed form)로 계산한다.
$$\text{Prox}_{\text{GL}, \lambda}(w_g) := w_g \frac{\max(\|w_g\| - \lambda, 0)}{\|w_g\|}$$
- **$w$-업데이트:** 확률적 경사 하강법(SGD)을 사용하여 업데이트한다.
$$w_{t+1} = w_t - \eta \nabla \ell(w_t) - \eta \beta(w_t - u_t)$$

### 3. Weight Binarization

가중치를 $\pm 1$로 투영하기 위해 다음과 같은 투영 연산자 $\text{proj}_{Q, a}(w)$를 정의한다.

$$\text{proj}_{Q, a}(w) = \frac{\sum_{j=1}^D |w_j|}{D} \text{sgn}(w)$$

여기서 $\text{sgn}(w)$는 각 원소의 부호를 결정하며, 앞의 계수는 가중치들의 절대값 평균을 곱해 크기를 조정하는 역할이다. 학습 시에는 가중치 정체(Weight stagnation) 현상을 막기 위해 Floating weight와 Binary weight를 섞어서 업데이트하는 Blended version의 BinaryConnect 알고리즘을 사용한다.

### 4. 네트워크 구조

대상 모델은 2개의 Convolution layer와 1개의 Fully Connected layer로 구성된다. 입력은 윈도우 푸리에 변환을 거친 스펙트로그램(Spectrogram) 이미지이며, 두 번째 Conv layer의 64개 채널을 대상으로 가지치기를 수행한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업:** 12개 클래스(silence, unknown, yes, no, up, down, left, right, on, off, stop, go)의 키워드 분류.
- **지표:** 검증 정확도(Validation Accuracy) 및 채널 희소성(Channel Sparsity).
- **비교 대상:** Original Audio-CNN, GL 기반 가지치기, RGSM 기반 가지치기, GSBC 기반 가지치기.

### 주요 결과

1. **Stage I (가지치기 효율):** RGSM은 GL이나 GSBC보다 훨씬 효율적으로 희소성을 생성하였다. $\lambda=0.05$일 때 $56.3\%$의 희소성을 보였으며, $\lambda=0.04$일 때 $51.6\%$의 희소성과 $76.6\%$의 정확도를 기록하였다 (Table 1).
2. **Stage II (정확도 회복):** 마스킹 레이어를 통해 가지치기된 채널을 고정한 후 재학습한 결과, 정확도가 원래 모델 수준인 $87.9\% \sim 89.2\%$까지 회복되었다 (Table 2).
3. **Stage III (이진화 결과):** 최종적으로 가중치를 이진화했을 때, 원래 모델($88.5\%$)과 비교하여 매우 적은 차이인 $87\% \sim 88.3\%$의 정확도를 유지하였다 (Table 3).

| 모델 | 채널 희소성 | 최종 정확도 (Stage III) |
| :--- | :---: | :---: |
| Original Audio-CNN | $0\%$ | $88.50\%$ |
| RGSM + Binarization ($\lambda=0.04$) | $51.6\%$ | $88.3\%$ |
| RGSM + Binarization ($\lambda=0.05$) | $56.3\%$ | $87.0\%$ |

## 🧠 Insights & Discussion

본 연구는 단순한 가중치 양자화나 가지치기 단독 적용보다, **'구조적 가지치기 $\rightarrow$ 정밀 튜닝 $\rightarrow$ 이진화'**로 이어지는 단계적 접근법이 모델 압축에 매우 효과적임을 입증하였다. 특히 RGSM이 일반적인 Group Lasso보다 딥러닝 모델의 구조적 희소성을 유도하는 데 더 적합하다는 점을 실험적으로 보여주었다.

하드웨어 구현 측면에서는 MacBook Air CPU 환경에서 float precision 가중치 모델만으로도 약 $28.87\%$의 속도 향상을 확인하였다. 다만, 저자들은 현재의 마스킹 레이어 구현 방식이 요소별 텐서 곱셈(element-wise multiplication)을 사용하므로, 모바일 환경에서 더 효율적으로 구현할 방법이 필요하다고 언급한다. 또한 향후 연구로 $\ell_0$ 페널티 적용 가능성과 더 큰 네트워크로의 확장 가능성을 제시한다.

## 📌 TL;DR

이 논문은 KWS를 위한 CNN 모델을 압축하기 위해 **RGSM 기반의 채널 가지치기**와 **가중치 이진화**를 결합한 3단계 학습 프레임워크를 제안한다. 이를 통해 **채널의 $50\%$ 이상을 제거하고 가중치를 1비트로 줄였음에도 불구하고, 정확도 손실을 $0.25\%$ 이내로 유지**하는 성과를 거두었다. 이는 저사양 임베디드 기기에서 고성능 음성 인식 모델을 구현하는 데 중요한 방법론이 될 수 있다.
