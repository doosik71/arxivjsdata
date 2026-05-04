# FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation

Liqun Ma, Mingjie Sun, Zhiqiang Shen (2024)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(Large Language Models, LLMs)의 막대한 파라미터 규모로 인해 발생하는 저장 공간 및 계산 비용의 문제를 해결하고자 한다. 기존의 양자화 기술은 이를 완화하기 위해 사용되어 왔으며, 특히 파라미터를 $\{-1, 1\}$로 표현하는 이진화(Binarization)는 압축률과 효율성을 극대화할 수 있는 가장 극단적인 형태의 양자화 방식이다.

그러나 완전한 이진화는 정확도 손실이 매우 크다는 단점이 있다. 이를 해결하기 위해 일부 중요한 파라미터만 풀 정밀도(Full-precision)로 유지하거나(PB-LLM, BiLLM), $\{-1, 0, 1\}$의 삼진법을 사용하는(BitNet b1.58) 방식이 제안되었으나, 이는 여전히 추가적인 저장 공간을 필요로 하거나 하드웨어 구현 시 오버헤드를 발생시킨다. 따라서 본 연구의 목표는 기존의 풀 정밀도 모델(FP16 또는 BF16)에 근접하는 성능을 가지면서, 처음부터(from scratch) 학습 가능한 완전 이진화된 거대 언어 모델(Fully Binarized LLM)을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 처음으로 대규모 이진 언어 모델을 처음부터 학습시켜 풀 정밀도 모델과 경쟁 가능한 성능을 달성했다는 점이다. 이를 가능하게 한 중심적인 아이디어는 다음과 같다.

1.  **Autoregressive Distillation (AD) 도입**: 이진화된 모델의 불안정한 학습을 해결하기 위해, 풀 정밀도 교사 모델(Teacher Model)의 예측 확률 분포를 학습하는 자기회귀 증류(Autoregressive Distillation) 손실 함수를 제안하였다.
2.  **FBI-Linear 모듈 설계**: 학습 가능한 컬럼 단위 스케일 파라미터($\alpha, \beta$)를 도입하여 이진화로 인한 정보 손실을 보완하고 표현력을 높였다.
3.  **학습 패러다임의 전환**: 기존 연구들이 사전 학습된 모델을 이진화하는 방식에 집중한 것과 달리, 무작위 초기화 상태에서 처음부터 학습하는 것이 가능하며 오히려 더 안정적일 수 있음을 입증하였다.

## 📎 Related Works

기존의 신경망 이진화 연구는 주로 이미지 분류 작업의 BNN(Binarized Neural Networks)에서 시작되었으며, $\text{sign}$ 함수의 비미분성 문제를 해결하기 위해 Straight-Through Estimator (STE)를 사용하는 방식이 일반적이다.

LLM 분야에서의 이진화 접근 방식은 다음과 같이 구분된다.
- **부분 이진화 (Partial Binarization)**: PB-LLM이나 BiLLM은 중요 파라미터(Salient parameters)를 풀 정밀도로 남겨두어 성능 저하를 막지만, 저장 공간 효율성이 떨어진다.
- **근사 이진화 (Near-one-bit)**: BitNet b1.58은 파라미터를 $\{-1, 0, 1\}$로 제한하여 효율을 높였으나, 완전한 1-bit 모델은 아니다.
- **양자화 인식 학습 (QAT)**: BitNet이나 OneBit는 QAT를 통해 이진화를 시도하였으나, 본 논문은 더 간소화된 AD 기반의 학습 절차를 통해 더 나은 성능을 달성하였다.

## 🛠️ Methodology

### 전체 시스템 구조
FBI-LLM은 LLaMA 아키텍처를 기반으로 하며, 모델 내의 모든 Linear 레이어를 **FBI-Linear**로 대체한다. 단, 출력 토큰 분포에 직접적인 영향을 주는 Causal Head, 시맨틱 정보를 담고 있는 Embedding 레이어, 그리고 활성화 값을 스케일링하는 Layer Norm은 정밀도 유지를 위해 풀 정밀도(16-bit)로 유지한다.

### FBI-Linear 모듈
FBI-Linear는 풀 정밀도 가중치 $W_f \in \mathbb{R}^{m \times n}$를 다음과 같이 이진화한다.
$$W_b = \text{sign}(W_f)$$
여기서 $\text{sign}(\cdot)$ 함수는 $W_{f_{ij}} > 0$이면 $1$, 그렇지 않으면 $-1$을 반환한다. 이진화된 가중치 $W_b$의 표현력을 높이기 위해 학습 가능한 컬럼 단위 스케일 벡터 $\alpha \in \mathbb{R}^n$와 $\beta \in \mathbb{R}^n$를 도입하여 최종 가중치를 다음과 같이 계산한다.
$$f_{W_{b \cdot,j}} = \alpha_j W_{b \cdot,j} + \beta_j$$
여기서 $f_{W_{b \cdot,j}}$는 스케일링 된 $j$번째 컬럼을 의미한다. $\alpha$와 $\beta$는 가중치의 평균과 편차를 기반으로 초기화되어 학습 수렴 속도를 높인다.

### Autoregressive Distillation (AD)
표준적인 언어 모델 학습은 정답 토큰에 대한 Cross-Entropy 손실을 사용하지만, FBI-LLM은 풀 정밀도 교사 모델($p^T$)의 확률 분포를 학생 모델($p^S$)이 따라가도록 하는 AD 손실 함수를 사용한다.
$$L = -\frac{1}{n} \sum_{i=1}^{n} p^T(x_{i+1}) \cdot \log p^S(x_{i+1})$$
교사 모델이 제공하는 Soft Label은 단순한 정답(Hard Label)보다 풍부한 정보(다른 가능성 있는 단어들 간의 관계 등)를 담고 있어, 표현력이 제한적인 이진 모델이 더 효율적으로 학습하도록 돕는다.

### 학습 절차 및 최적화
$\text{sign}$ 함수는 미분이 불가능하므로, 역전파 시에는 **Straight-Through Estimator (STE)**를 사용하여 gradient를 그대로 전달한다.
$$\frac{\partial L}{\partial W_f} = \frac{\partial L}{\partial W_b}$$
또한 학습 중 발생하는 Loss Spike(손실 급증) 문제는 PaLM의 방식과 유사하게, 문제가 발생한 데이터 청크를 건너뛰고 이전 체크포인트로 복구하는 방식으로 해결하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: Amber 데이터셋 (1.26조 개의 토큰)
- **모델 규모**: 130M, 1.3B, 7B 파라미터
- **교사 모델**: LLaMA2-7B
- **평가 지표**: Perplexity (PPL) 및 Zero-shot Accuracy (BoolQ, PIQA, HellaSwag, Winogrande, ARC, OpenbookQA)

### 주요 결과
1.  **성능 우위**: FBI-LLM은 유사한 규모의 다른 이진화 모델(BiLLM, OneBit, BitNet)보다 낮은 Perplexity와 높은 제로샷 정확도를 보였다. 특히 1.3B 모델은 일부 7B 규모의 이진 모델(BiLLM-LLaMA2)과 대등하거나 더 나은 성능을 보였다.
2.  **AD의 효과**: 표준 자기회귀 손실(Normal Autoregressive Loss)만 사용했을 때보다 AD를 사용했을 때 모든 지표에서 일관되게 높은 성능을 기록하였다.
3.  **압축 효율**: 7B 모델 기준, 풀 정밀도 LLaMA 대비 약 90%의 저장 공간을 절감하였다. 추가된 스케일 파라미터($\alpha, \beta$)가 차지하는 비중은 매우 미미하여 전체 저장 효율에 영향을 주지 않았다.

## 🧠 Insights & Discussion

### 사전 학습 가중치의 불필요성
연구팀은 사전 학습된 가중치에서 시작하는 것(Continue training)과 무작위 초기화에서 시작하는 것(Training from scratch)을 비교 분석하였다. 분석 결과, Flip-Flop Ratio(가중치의 부호가 바뀌는 비율)와 Gradient Norm 측면에서 무작위 초기화 방식이 오히려 더 안정적인 학습 곡선을 보였다. 이는 이진 모델과 풀 정밀도 모델이 시맨틱을 인코딩하는 파라미터 공간의 패턴이 서로 다르기 때문에, 억지로 기존 가중치를 이입하는 것보다 처음부터 이진 구조에 맞게 학습하는 것이 더 효율적임을 시사한다.

### 학습 불안정성
이진 모델은 풀 정밀도 모델보다 표현력이 제한적이어서 학습 중 Loss Spike가 발생할 확률이 더 높았다. (7B 모델 약 6%, 1.3B 모델 약 15%). 이는 데이터 청크 스킵 전략으로 해결 가능하며, 충분한 데이터가 제공된다면 성능이 계속해서 향상될 가능성이 높음을 확인하였다.

### 비판적 해석
본 연구는 이론적인 저장 공간의 이점을 입증하였으나, 실제 추론 속도 향상을 위해서는 이진 연산을 직접 지원하는 전용 하드웨어 가속기가 필요하다는 한계가 있다. 또한, 활성화 값(Activation)은 여전히 16-bit로 유지하고 있어, 완전한 연산 효율화를 위해서는 활성화 값의 이진화 연구가 병행되어야 할 것이다.

## 📌 TL;DR

본 논문은 **Autoregressive Distillation(AD)**을 통해 풀 정밀도 모델에 근접한 성능을 내는 **완전 이진화(1-bit) LLM인 FBI-LLM**을 제안하였다. 특히, 사전 학습된 가중치 없이 처음부터 학습하는 것이 가능하며 오히려 더 안정적이라는 점을 발견하였다. 이 연구는 LLM의 저장 공간을 획기적으로(최대 90%) 줄이면서도 성능을 유지할 수 있음을 보여주었으며, 향후 1-bit 전용 하드웨어 설계 및 효율적인 모델 배포를 위한 중요한 기반을 마련하였다.