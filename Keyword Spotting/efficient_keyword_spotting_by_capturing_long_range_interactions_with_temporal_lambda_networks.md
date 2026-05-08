# EFFICIENT KEYWORD SPOTTING BY CAPTURING LONG-RANGE INTERACTIONS WITH TEMPORAL LAMBDA NETWORKS

Biel Tura, Santiago Escuder, Ferran Diego, Carlos Segura, Jordi Luque (2021)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 제한된 자원을 가진 소형 임베디드 기기(Small-footprint devices)에서 효율적으로 동작하는 키워드 검출(Keyword Spotting, KWS) 모델을 구축하는 것이다.

최근 Attention 메커니즘 기반의 모델들은 음성 인식 분야에서 전례 없는 성능을 보여주었으나, 이러한 모델들은 계산 복잡도가 매우 높고 메모리 사용량이 많아 스마트 홈, IoT 기기 등 자원이 한정된 환경에 배포하기에는 부적합하다. 반면, 기존의 Convolutional Neural Networks(CNN)는 계산 효율성은 높지만 음성 신호의 장기 의존성(Long-range dependencies)을 포착하는 능력이 부족하다는 한계가 있다.

따라서 본 논문의 목표는 Transformer와 같은 Attention 메커니즘의 장점인 장거리 상호작용 포착 능력을 유지하면서도, 계산 비용을 획기적으로 줄여 저사양 기기에서도 구동 가능한 효율적인 KWS 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Attention 메커니즘의 대안으로 제안된 **Lambda Networks**를 음성 도메인, 특히 KWS 작업에 처음으로 적용한 것이다.

구체적인 설계 전략은 다음과 같다.

1. **Lambda Layer 도입**: 고비용의 Attention Map을 계산하는 대신, 선형 함수인 'Lambda'를 통해 장거리 상호작용을 효율적으로 캡처한다.
2. **Temporal LambdaResNet 구조**: 기존의 ResNet 기반 구조에서 Residual Block 내의 두 번째 Convolutional layer를 Temporal Lambda layer로 교체한 $\text{LambdaResNet18}$ 아키텍처를 제안한다.
3. **1D Temporal Convolution 활용**: 2D Convolution 대신 1D Convolution을 사용하여 모델의 복잡도를 더욱 낮추고 연산 속도를 향상시켰다.

## 📎 Related Works

논문에서 언급하는 기존 접근 방식과 그 한계는 다음과 같다.

- **CNN 및 MLP 기반 방식**: early-stage의 KWS는 MLP나 CNN을 사용하여 효율성을 높였으나, 시간적 맥락을 충분히 활용하지 못하는 경우가 많았다.
- **RNN 기반 방식**: 장기 의존성을 학습할 수 있으나, 실시간 음성 인식 시스템에 통합하기에 구조적 어려움이 있다.
- **Attention 및 Transformer 기반 방식 (KWT 등)**: 최신 SOTA 성능을 보여주지만, 파라미터 수가 매우 많고 계산 복잡도가 높아 임베디드 기기 적용이 어렵다.
- **Residual Convolutional Networks (ResNet15 등)**: 현재 KWS에서 널리 쓰이는 효율적인 구조이나, 여전히 국소적인 특징(Local features) 추출에 집중되어 있어 매우 긴 시간적 관계를 포착하는 데 한계가 있다.

본 연구는 이러한 Trade-off(정확도 vs 계산 자원)를 해결하기 위해 Attention의 성능과 CNN의 효율성을 절충한 Lambda framework를 도입함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. Input Preprocessing

입력 데이터는 40개의 Mel filterbank 에너지를 사용하여 생성된 log-scaled Mel spectrogram을 사용한다. 분석 윈도우는 $20\text{ms}$, 스트라이드는 $10\text{ms}$이다. 특이한 점은 이를 일반적인 2D 이미지 형태로 처리하지 않고, 각 주파수 밴드를 독립적인 1D 입력 채널로 간주하여 처리한다는 것이다.

### 2. Temporal Lambda Layer

Lambda layer는 Attention 메커니즘의 연산 비용을 줄이면서 유사한 효과를 내기 위해 설계되었다.

#### Self-Attention과의 비교

기본적인 Self-attention은 Query($Q$), Key($K$), Value($V$)를 이용하여 다음과 같이 Attention Map을 생성하고 가중 합을 구한다.
$$A = \sigma\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
여기서 $\sigma(\cdot)$는 softmax 연산이다. 이 과정은 입력 길이 $n$에 대해 $O(n^2)$의 시간 및 공간 복잡도를 가지며, 특히 Multi-head 구조에서는 head 수 $h$만큼 비용이 배가된다.

#### Lambda Operator

Lambda layer는 Attention Map을 직접 계산하지 않고, 다음과 같은 **Contextual Lambda function** $\lambda_n$을 정의한다.
$$\lambda_n = \underbrace{\sigma(K)^T V}_{\lambda_c} + \underbrace{E_n^T V}_{\lambda_{pn}}$$

- **Content Lambda ($\lambda_c$)**: 문맥의 내용만을 기반으로 쿼리를 어떻게 변환할지를 인코딩한다.
- **Position Lambda ($\lambda_{pn}$)**: 학습 가능한 positional embedding $E_n$을 통해 쿼리의 상대적 위치에 따른 변환을 인코딩한다.

최종 출력 $y_n$은 이 Lambda 함수를 쿼리 $q_n$에 적용하여 얻는다.
$$y_n = \lambda_n^T q_n = (\lambda_c + \lambda_{pn})^T q_n$$

또한, 연산량을 더 줄이기 위해 **Multi-query Lambda** 방식을 사용한다. 이는 하나의 Lambda 함수를 공유하고 여러 개의 쿼리 $q_1, \dots, q_h$를 적용하여 출력을 이어붙이는 방식으로, 계산 복잡도를 $h$배만큼 감소시킨다. 더 나아가, 위치 임베딩의 범위를 국소 영역 $r$로 제한하는 **Lambda Convolution**을 통해 시간 복잡도를 선형($O(n)$)으로 낮추었다.

### 3. Model Architecture ($\text{LambdaResNet18}$)

제안된 $\text{LambdaResNet18}$은 총 18개의 레이어로 구성된다.

- **구조**: 초기 1D Conv layer $\rightarrow$ 4개의 Residual Stage (각 Stage는 2개의 Residual Lambda Block으로 구성) $\rightarrow$ Average Pooling $\rightarrow$ Fully Connected Layer $\rightarrow$ Softmax.
- **Residual Lambda Block**: 첫 번째 레이어는 $3 \times 1$ 1D Convolution을 수행하고, 두 번째 레이어는 Temporal Lambda layer를 수행한 뒤 Shortcut connection을 통해 더해진다.
- **채널 구성**: $\{16, 24, 36, 48, 60\}$ 순으로 확장되며, 각 블록의 첫 Conv layer에서 stride 2를 사용하여 시간축 해상도를 줄인다.
- **하이퍼파라미터**: Multi-query approach ($h=4$), $d_k=16$, $d_v=d/h$, local context size $r=23$을 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands v2 (10, 20, 35개 키워드 서브태스크).
- **데이터 증강**: 배경 소음 추가, Clip distortion, Cropping, Pitch shift, Temporal shift/stretch, Volume 조절 등을 확률적으로 적용하였다.
- **학습 설정**: Cross-Entropy loss, SGD (momentum 0.9), Cosine learning rate decay, $L_2$ weight decay ($10^{-3}$)를 사용하였다.

### 정량적 결과 (Table 3 기반)

$\text{LambdaResNet18}$은 기존 모델들과 비교하여 다음과 같은 결과를 보였다.

| 모델 | 35 Keywords Acc (%) | 파라미터 수 | 연산량 (FLOPS) |
| :--- | :---: | :---: | :---: |
| $\text{ResNet15}$ | $\mathbf{95.9}$ | $237\text{K}$ | $894\text{M}$ |
| $\text{TC-ResNet14}$ | $91.3$ | $137\text{K}$ | $6.1\text{M}$ |
| $\text{LambdaResNet18}$ | $93.1$ | $\mathbf{89\text{K}}$ | $\mathbf{3.3\text{M}}$ |
| $\text{LambdaResNet18-2}$ | $94.2$ | $270\text{K}$ | $8.4\text{M}$ |

- **정확도 및 효율성**: $\text{ResNet15}$가 가장 높은 정확도를 보였으나, $\text{LambdaResNet18}$은 $\text{ResNet15}$보다 파라미터 수는 약 $2.6\times$ 적고, 연산 속도는 $100\times$ 이상 빠르다.
- **경량 모델 비교**: 1D Conv 기반의 $\text{TC-ResNet14}$보다 모든 서브태스크에서 평균 $1.5\%$ 더 높은 정확도를 기록하면서도, 파라미터 수는 더 적고 연산 속도는 $2\times$ 더 빠르다.
- **Transformer 대비**: $\text{Keyword Transformer (KWT)}$ 모델들보다 정확도는 약간 낮으나, 모델 크기는 $85\%$ 이상 작으며 계산 복잡도 면에서 압도적인 우위를 점한다.

## 🧠 Insights & Discussion

본 논문은 Attention 메커니즘이 주는 성능 이점을 유지하면서 계산 비용을 획기적으로 낮출 수 있는 Lambda layer의 가능성을 KWS 작업에서 입증하였다. 특히 1D Convolution과 Lambda layer의 조합이 임베디드 기기 환경에서 최적의 Trade-off를 제공함을 보여주었다.

**강점**:

- Attention Map 계산을 생략함으로써 $O(n^2)$의 복잡도를 효과적으로 제어하였다.
- 1D 신호 처리 관점에서 주파수 밴드를 독립 채널로 처리하여 연산량을 극단적으로 낮추었다.

**한계 및 논의사항**:

- **정확도 격차**: 절대적인 정확도 면에서는 여전히 고비용의 $\text{ResNet15}$나 $\text{KWT}$에 밀리는 경향이 있다. 이는 Lambda layer가 Attention을 완벽하게 대체하기보다는 '근사'하는 방식이기 때문으로 해석된다.
- **미해결 과제**: 저자는 음성학적 유사성(Phonetic similarity) 기반의 커스텀 손실 함수를 적용할 경우 성능이 더 향상될 수 있음을 시사하며, 이를 향후 연구 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 Attention의 고비용 문제를 해결하기 위해 **Lambda layer**를 도입한 **$\text{LambdaResNet18}$** 아키텍처를 제안한다. 이 모델은 1D Convolution과 Lambda Operator를 결합하여 장거리 상호작용을 효율적으로 포착하며, 기존 SOTA 모델(KWT) 대비 **85% 더 가볍고**, $\text{ResNet15}$ 대비 **100배 이상 빠른** 추론 속도를 구현하였다. 결과적으로 저사양 임베디드 기기에서 구동 가능한 고효율 KWS 모델의 새로운 기준을 제시하였다.
