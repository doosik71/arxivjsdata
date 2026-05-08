# GPT-PPG: A GPT-based Foundation Model for Photoplethysmography Signals

Zhaoliang Chen, Cheng Ding, Saurabh Kataria, Runze Yan, Minxiao Wang, Randall Lee, and Xiao Hu (2025)

## 🧩 Problem to Solve

본 논문은 광혈류측정(Photoplethysmography, PPG) 신호를 분석하기 위한 GPT 기반의 파운데이션 모델(Foundation Model)을 제안한다. PPG 신호는 비침습적이고 웨어러블 기기를 통해 쉽게 수집할 수 있어 임상적으로 매우 유용하지만, 실제 적용에 있어 다음과 같은 문제점들이 존재한다.

첫째, PPG 신호는 노이즈와 움직임으로 인한 아티팩트(motion artifacts)에 매우 취약하며, 데이터셋 간의 변동성이 크다. 둘째, 임상 데이터의 특성상 정답 레이블(labeled data)이 포함된 학습 데이터를 대량으로 확보하는 것이 매우 어렵다.

따라서 본 연구의 목표는 대규모의 레이블 없는 PPG 데이터를 통해 사전 학습된 파운데이션 모델을 구축함으로써, 적은 양의 레이블 데이터만으로도 심방세동(Atrial Fibrillation, AF) 감지, 심박수(Heart Rate, HR) 및 호흡수(Respiration Rate, RR) 추정, 혈압(Blood Pressure, BP) 예측과 같은 다양한 하위 작업(downstream tasks)에서 높은 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 자연어 처리 분야에서 검증된 GPT(Generative Pre-trained Transformer)의 생성적 사전 학습 패러다임을 연속적인 시계열 데이터인 PPG 신호에 적용하는 것이다. 주요 기여 사항은 다음과 같다.

- **PPG 전용 GPT 아키텍처 설계**: 연속적인 PPG 신호의 특성에 맞게 GPT 구조를 조정하고, 19M부터 1B 파라미터까지 네 가지 규모의 모델을 구현하였다.
- **대규모 데이터셋 사전 학습**: 2억 개 이상의 30초 PPG 샘플이 포함된 방대한 데이터셋을 사용하여 모델의 일반화 능력을 극대화하였다.
- **Mixed-Objective Fine-tuning 프레임워크**: 하위 작업 수행 시 목표 손실 함수($L_o$)와 신호 모델링 손실 함수($L_m$)를 결합하여, 하위 데이터셋의 분포에 효율적으로 적응하게 하는 학습 방법을 제안하였다.
- **생성적 능력의 활용**: 별도의 미세 조정(fine-tuning) 없이도 신호의 누락된 부분을 복원하는 신호 디노이징(denoising) 작업에서 뛰어난 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 시계열 데이터 모델링은 주로 Informer나 Autoformer와 같이 Attention 메커니즘을 활용하여 장기 의존성을 캡처하는 방식이었으며, 최근에는 TimeGPT나 MOMENT와 같은 시계열 파운데이션 모델들이 등장하였다. 생체 신호 분야에서는 HeartBeit(ECG 분석)나 BIOT(EEG 임베딩)와 같이 트랜스포머 기반의 모델들이 제안되었다.

PPG 분석에 있어서는 기존에 CNN-RNN 기반의 회귀 모델이나 GAN 기반의 데이터 증강 기법들이 사용되어 특정 작업(예: AF 감지)에서 성공을 거두었다. 또한, SiamQuality와 같은 대조 학습(contrastive learning) 기반의 파운데이션 모델 시도가 있었으나, 본 논문은 GPT의 생성적(generative) 특성을 이용하여 특징 추출뿐만 아니라 신호 복원과 같은 생성 작업까지 수행할 수 있다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 데이터 준비 및 전처리

UCSF 의료 센터에서 수집된 24,100명 이상의 환자로부터 얻은 260만 시간의 PPG 신호를 사용한다.

- **세그먼트 및 리샘플링**: 전체 데이터를 겹치지 않는 30초 단위의 스트립으로 나누고, 40Hz로 리샘플링한다.
- **정규화**: 각 샘플 $X \in \mathbb{R}^{1200}$를 Min-Max 정규화를 통해 $[0, 1]$ 범위로 변환한다.
- **패칭(Patching)**: 정규화된 신호를 1초 단위(40개 샘플)의 패치 30개로 나누어 $X = \{x_1, x_2, \dots, x_{30}\}$ 형태로 구성한다.

### 2. 모델 아키텍처 및 사전 학습

모델은 GPT-19M, 85M, 345M, 1B의 네 가지 규모로 구축되었다.

- **입력 구조**: 각 패치를 선형 층을 통해 $d$ 차원의 벡터로 매핑한다. 문장의 시작을 알리는 `[SOS]` 토큰 대신 학습 가능한 벡터 $h_s$를 입력 시퀀스 앞에 추가하여 $\{h_s, h_1, h_2, \dots, h_{29}\}$ 형태로 트랜스포머 디코더에 입력한다. ($h_{30}$은 정답 패치 $x_{31}$이 없으므로 입력에서 제외된다.)
- **구성 요소**: RMSNorm과 Rotary Positional Embedding을 사용하여 시퀀스의 상대적 위치 정보를 캡처한다.
- **손실 함수 (Logit-Laplace Distribution Loss)**:
  연속적인 PPG 신호를 예측할 때 MSE(Mean Squared Error)를 사용하면 모델이 단순히 평균값(약 0.5)만 예측하는 모델 붕괴(model collapse) 현상이 발생한다. 이를 해결하기 위해 $(0, 1)$ 구간에서 정의된 Logit-Laplace 분포 기반의 손실 함수를 사용한다. 확률 밀도 함수(PDF)는 다음과 같다.
  $$f(x; \mu, b) = \frac{1}{2bx(1-x)} \exp\left(-\frac{|\text{logit}(x)-\mu|}{b}\right)$$
  여기서 $x$는 실제 값이며, 모델은 위치 파라미터 $\mu$와 척도 파라미터 $b$를 예측한다. 이 손실 함수의 최소화는 $L1$ 거리 손실을 최소화하는 것과 유사하며, sigmoid 변환을 통해 출력 범위를 $(0, 1)$로 제한하여 수치적 안정성을 확보한다.

### 3. Mixed-Objective Fine-tuning 프레임워크

사전 학습된 모델을 하위 작업에 적응시키기 위해 다음과 같은 결합 손실 함수를 사용한다.
$$L(y, y', X, X') = L_o(y, y') + \lambda L_m(X, X')$$
여기서 $L_o$는 하위 작업의 목표 손실(회귀는 MSE, 분류는 Cross Entropy)이고, $L_m$은 사전 학습 때와 동일한 Logit-Laplace 신호 모델링 손실이다. $\lambda$ 값은 학습이 진행됨에 따라 서서히 0으로 감소(annealing)시킨다.

- **특징 추출 및 예측**:
  1. **Attention Pooling**: 트랜스포머 디코더에서 나온 패치 표현 $H \in \mathbb{R}^{N \times D}$를 학습 가능한 가중치 $w$를 이용해 가중합하여 시퀀스 레벨 특징 $h \in \mathbb{R}^D$를 생성한다.
  2. **Gated MLP**: 생성된 $h$를 $\text{SiLU}(h^\top W_2 \odot h^\top W_3)W_1$ 형태의 Gated MLP에 통과시켜 최종 예측값 $y'$를 도출한다.

### 4. 프레임워크 확장 (Extensions)

- **Bidirectional Feature Extraction**: GPT의 단방향 어텐션 한계를 극복하기 위해 인과적 마스크(causal mask)를 제거하고, 무작위로 마스킹된 패치를 복원하도록 학습시켜 양방향 정보를 활용한다.
- **Fallback Method**: 모델의 가중치를 고정한 채(Freeze GPT) 예측 헤드만 학습시킬 때 사용하는 효율적인 방법이다. 신호 모델링 손실로부터 얻은 가능도(likelihood) $L(x)$를 이용하여, 신뢰도가 낮을 때 학습 가능한 파라미터 $y_{\text{fallback}}$에 의존하게 한다.
  $$\hat{y} = L(x)P(x) + \frac{y_{\text{fallback}}}{\max(L(x), \delta)}$$
- **Test-time Domain Adaptation (Personalization)**: 테스트 데이터의 일부(5~10%)를 사용하여 레이블 없이 신호 모델링 손실($L_m$)만으로 모델을 미세 조정함으로써, 피험자 개인의 신호 분포에 맞게 모델을 최적화한다.

## 📊 Results

### 1. 하위 작업 성능 (Downstream Tasks)

다양한 벤치마크에서 GPT-PPG는 기존 SOTA 방법들과 대등하거나 이를 능가하는 성능을 보였다.

- **심방세동(AF) 감지**: Stanford 데이터셋에서 GPT-1B 모델이 F1-score 0.847을 기록하며 매우 높은 성능을 보였다.
- **심박수(HR) 추정**: WESAD, DaLiA, IEEE 데이터셋에서 MAE 기준으로 SOTA 수준에 근접하였으며, 특히 GPT-1B 모델이 가장 우수한 성능을 보였다.
- **호흡수(RR) 추정**: BIDMC 데이터셋에서 기존 방법들을 크게 상회하는 MAE(GPT-1B: 0.93)를 기록하였다.
- **혈압(BP) 추정**: PulseDB 데이터셋에서 SBP와 DBP 모두에서 타 모델 대비 낮은 MAE를 달성하였다.

### 2. 효율성 및 일반화 분석

- **양방향 특징 추출의 효과**: 85M 모델에 양방향 추출 방식을 적용했을 때, 심박수 추정 작업에서 1B 모델과 대등하거나 더 좋은 성능을 보였다.
- **Parameter-Efficient Fine-tuning**: GPT 층을 고정하고 Fallback Method를 적용했을 때, 단순 고정 방식보다 성능이 향상됨을 확인하였다. 이는 사전 학습된 표현력이 유효함을 시사한다.
- **개인화(Personalization)**: 테스트 데이터의 10%만 사용하여 적응시켰을 때, 심박수 추정 MAE가 유의미하게 감소하였다.
- **생성 작업 성능**: 신호의 최대 50%가 마스킹된 상황에서도 원래 신호를 매우 유사하게 복원해내는 능력을 보였다.

### 3. 스케일링 분석 (Scaling Analysis)

모델 크기가 19M $\to$ 85M $\to$ 345M $\to$ 1B로 커짐에 따라 모든 작업에서 성능이 꾸준히 향상되었다. 특히 19M에서 85M로 넘어갈 때 성능 향상 폭이 가장 컸으며, 85M 모델이 계산 효율성과 성능 사이의 최적의 균형점(sweet spot)인 것으로 분석되었다.

## 🧠 Insights & Discussion

**강점 및 의의**
본 연구는 대규모 레이블 없는 데이터로 사전 학습된 GPT 기반 파운데이션 모델이 PPG 분석의 데이터 부족 문제를 효과적으로 해결할 수 있음을 입증하였다. 특히, 단순한 분류/회귀를 넘어 신호 복원이라는 생성적 능력을 갖추었다는 점과, $\lambda$를 활용한 Mixed-Objective 학습을 통해 하위 데이터셋의 분포 차이를 극복하려 한 점이 돋보인다.

**한계 및 비판적 해석**

1. **분포 민감성**: 모델이 사전 학습 데이터와 테스트 데이터의 분포 차이에 매우 민감하게 반응한다. 이는 OOD(Out-of-Distribution) 일반화 능력이 아직 부족함을 의미하며, 본문에서도 서로 다른 데이터셋 간의 교차 검증 시 성능이 크게 하락하는 모습이 관찰되었다.
2. **계산 비용**: 1B 규모의 모델은 성능은 좋으나 엣지 기기(wearables) 배포에는 현실적으로 어려움이 있다. 지식 증류(distillation) 등의 경량화 연구가 필수적이다.
3. **아키텍처 비교 부족**: 동일 데이터셋으로 사전 학습된 Encoder-only 모델(예: BERT style)과의 직접적인 비교가 이루어지지 않아, 왜 반드시 GPT의 생성적 구조여야 하는지에 대한 정밀한 분석이 다소 부족하다.
4. **패치 기반 가정**: 패치 내의 데이터 포인트들이 독립적으로 예측 가능하다는 가정을 사용하였는데, 이는 신호의 특성에 따라 유효하지 않을 수 있다.

## 📌 TL;DR

본 논문은 2억 개 이상의 샘플로 사전 학습된 **GPT 기반의 PPG 파운데이션 모델(GPT-PPG)**을 제안한다. 이 모델은 심방세동 감지, 심박수/호흡수/혈압 추정 등 다양한 심혈관 분석 작업에서 SOTA급 성능을 보이며, 특히 **Logit-Laplace 손실 함수**를 도입해 연속 신호의 생성적 학습을 가능케 했다. 또한, **Mixed-Objective Fine-tuning**과 **Fallback Method**를 통해 효율적인 하위 작업 적응 방법을 제시하였으며, 추가 학습 없이도 신호 디노이징이 가능한 생성 능력을 입증하였다. 향후 웨어러블 헬스케어 기기에서 적은 데이터로도 정밀한 진단을 가능케 하는 핵심 모델로 활용될 가능성이 높다.
