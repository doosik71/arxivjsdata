# ARST: Auto-Regressive Surgical Transformer for Phase Recognition from Laparoscopic Videos

Xiaoyang Zou, Wenyong Liu, Junchen Wang, Rong Tao, and Guoyan Zheng (2022)

## 🧩 Problem to Solve

본 논문은 복강경 수술 영상(Laparoscopic Videos)을 이용한 **온라인 수술 단계 인식(On-line Surgical Phase Recognition)** 문제를 해결하고자 한다. 수술 워크플로우 분석은 컴퓨터 보조 중재(CAI) 시스템에서 수술의 표준화와 품질 평가를 위해 필수적이며, 정확한 단계 인식은 수술 중 실시간 피드백 제공, 이상 상황 알림, 신입 외과의의 교육 등 수술의 안전성과 지능 수준을 높이는 데 매우 중요하다.

그러나 수술 영상 기반의 단계 인식은 다음과 같은 이유로 매우 어렵다.

1. **시각적 유사성**: 서로 다른 단계의 프레임들이 시각적으로 매우 유사하여 단계 간 상관관계를 모델링하는 것이 까다롭다.
2. **Hard Frames의 존재**: 빠른 카메라 움직임, 가스 발생, 카메라의 시야 이탈 등으로 인해 인식하기 어려운 프레임들이 빈번하게 발생한다.

따라서 본 논문의 목표는 이전 단계의 예측 결과를 현재 단계 예측에 반영하는 **Auto-Regressive(자기회귀)** 방식을 도입하여 단계 간 상관관계를 효율적으로 캡처하고, 추론 시 발생하는 예측의 불안정성(frequently jumped predictions)을 해결하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 자기회귀 특성을 수술 단계 인식에 도입하여, 각 프레임의 단계 예측을 이전 프레임들의 예측 결과에 조건부로 의존하게 만드는 것이다. 주요 기여 사항은 다음과 같다.

1. **ARST(Auto-Regressive Surgical Transformer) 제안**: Transformer 기반의 자기회귀 프레임워크를 통해 단계 예측의 조건부 확률 분포를 모델링함으로써, 단계 간의 상관관계를 암시적으로 캡처한다.
2. **Banded Causal Mask 도입**: 모든 과거 프레임을 참조하는 대신, 특정 윈도우 크기($W$) 내의 이전 프레임들만 참조하도록 제한하여 노이즈를 줄이고 온라인 인식의 효율성을 높였다.
3. **Consistency Constraint Inference (CCI) 전략**: 추론 단계에서 급격한 단계 변동(jumped predictions)을 억제하기 위해, 단계 전환이 감지되었을 때 일정 기간($n=10$ 프레임) 동안 동일한 예측이 유지되는지 확인하는 제약 조건을 도입하여 예측의 일관성과 신뢰성을 높였다.

## 📎 Related Works

기존의 수술 단계 인식 연구는 다음과 같은 흐름으로 발전해 왔다.

- **초기 접근 방식**: 다양한 수술 중 기록되는 다차원 상태 신호(state signals)에 의존하였다.
- **딥러닝 기반 방식**: CNN을 이용한 프레임 단위 인식에서 시작하여, temporal dependency를 모델링하기 위해 LSTM(EndoLSTM)이나 TCN(TeCNO)과 같은 구조가 도입되었다.
- **Transformer 기반 방식**: 최근에는 Self-attention 메커니즘을 활용한 모델(Opera, Trans-SVNet 등)이 제안되어 우수한 성능을 보였다.

**기존 방식과의 차별점**: 기존의 Transformer 기반 모델들은 주로 Attention dependency 모델링에 집중했을 뿐, Transformer의 핵심 특징 중 하나인 **Auto-regression(자기회귀)** 특성을 활용하여 이전 예측값을 현재 예측의 입력으로 사용하는 구조를 설계하지 않았다. ARST는 이 점을 개선하여 단계 전환의 패턴을 능동적으로 학습한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

ARST는 프레임 레벨의 특징 추출기(Feature Extractor)와 Transformer 기반의 Encoder-Decoder 구조로 구성된다.

### 2. Feature Extractor (특징 추출기)

영상에서 공간적, 시간적 특징을 모두 추출하여 Encoder의 입력 임베딩으로 사용한다.

- **Spatial Features**: ResNet-50을 사용하여 각 프레임에서 512차원의 공간 특징 $Z_t$를 추출한다.
- **Temporal Features**: 추출된 공간 특징을 기반으로 2단계 Causal TCN인 **TeCNO**를 통해 시간적 문맥이 반영된 512차원의 특징 $F_t$를 추출한다.

### 3. Auto-Regressive Surgical Transformer (ARST)

경량화된 1층 Encoder-Decoder Transformer 구조를 사용한다.

- **Encoder**: 프레임 특징 $F_{1:t}$를 입력으로 받으며, Masked Multi-head Attention과 Feed-forward 층으로 구성된다.
- **Decoder**: 이전 프레임들의 예측 결과인 shifted outputs $\hat{y}_{0:t-1}$을 입력으로 받는다. Encoder의 출력과 Decoder의 입력 간의 Cross-attention을 통해 최종 확률 $p_{1:t}$를 출력한다.
- **학습 절차**: 학습 시에는 속도 향상과 정답 근접을 위해 **Teacher Forcing** 전략을 사용하여, 예측값이 아닌 실제 정답(Ground Truth) $y_{0:T-1}$을 Decoder의 입력으로 넣어 병렬 학습을 수행한다.

### 4. 주요 방정식 및 메커니즘

#### (1) Banded Causal Mask

과거의 너무 먼 정보는 현재 결정에 노이즈가 될 수 있다는 직관에 따라, 윈도우 너비 $W$ 만큼의 범위만 허용하는 Banded Causal Mask $M_{bc}$를 도입한다.
$$\text{Attention}(Q, K, V) = \text{Softmax}(M_{bc} \circ \frac{QK^T}{\sqrt{d}})V$$
여기서 $M_{bc}$는 너비 $W$ 외부의 요소를 $-\infty$로 설정하여 Softmax 결과가 0이 되게 만든다.

#### (2) 자기회귀 예측 (Auto-regressive Prediction)

전체 비디오의 단계 예측 확률은 베이즈 정리에 따라 다음과 같은 조건부 확률의 곱으로 분해된다.
$$p(\hat{y}_{1:T} | F_{1:T}) = \prod_{t=1}^{T} p(\hat{y}_t | \hat{y}_{0:t-1}, F_{1:t})$$

#### (3) Phase Embedding 및 Positional Encoding

예측된 단계 $\hat{y}$를 512차원 벡터 $E$로 변환할 때, 단순 One-hot 인코딩 대신 512차원을 $c$개의 세그먼트로 나누어 특정 세그먼트만 1로 채우는 방식을 사용하여 특징 공간 내 단계 간 거리를 넓혔다. 또한, 위치 정보를 제공하기 위해 Sine/Cosine 함수 기반의 Positional Encoding(PE)을 추가한다.
$$PE(t, 2i) = \sin\left(\frac{t}{10000^{2i/512}}\right), \quad PE(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/512}}\right)$$

### 5. Consistency Constrained Inference (CCI)

추론 시 발생하는 잦은 단계 점프를 방지하기 위한 전략이다.

- **작동 원리**: $t$ 시점에서 단계 전환($P_t \neq P_{t-1}$)이 감지되면, 즉시 반영하지 않고 다음 $n$개(예: 10개)의 프레임에 대해 이전 단계 $P_{t-1}$을 Decoder에 계속 입력하며 예측을 수행한다.
- **판단 기준**: 만약 다음 $n$개 프레임의 예측 결과가 모두 새로운 단계 $P_t$와 일치할 때만 실제 단계 전환이 일어난 것으로 간주한다. 그렇지 않으면 노이즈로 판단하여 $P_t$를 $P_{t-1}$로 수정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Cholec80 (담낭 절제술 영상 데이터셋, 80개 영상)
- **평가 지표**: Accuracy, Precision, Recall, Jaccard Index
- **비교 대상**: ResNet-50, SV-RCNet, TeCNO, Trans-SVNet

### 2. 주요 결과

- **정량적 성능**: TeCNO 특징($F_S = T$)을 사용했을 때, Accuracy 89.27%, Jaccard Index 76.10%를 기록하며 비교 대상 중 가장 높은 성능을 보였다.
- **Ablation Study**:
  - **Mask Width ($W$)**: $W=5$일 때 최적의 성능을 보였다. $W$가 너무 크면 노이즈에 취약해지고, 너무 작으면 예측이 불안정해지는 경향이 있다.
  - **AR 및 CCI 효과**: Auto-regression(AR)만 추가했을 때보다 CCI 전략을 함께 사용했을 때 Jaccard 지수가 각각 4.4%(공간 특징 사용 시) 및 3.6%(시간 특징 사용 시) 상승하였다.
- **정성적 결과**: 시각화 결과, ARST는 다른 모델들에 비해 예측 결과가 훨씬 매끄러우며(smoother), 특히 CCI 도입 후 예측 단계가 갑자기 튀는 현상이 거의 사라졌다.
- **추론 속도**: 프레임당 평균 15.15ms가 소요되어, 약 **66 fps**의 실시간 추론 속도를 달성하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 단순한 특징 추출을 넘어, **이전의 예측 결과가 현재의 결과에 영향을 주는 자기회귀 구조**를 수술 단계 인식에 성공적으로 적용하였다. 특히, 딥러닝 모델이 흔히 겪는 '불안정한 예측(jumping)' 문제를 해결하기 위해 사후 처리(post-processing)가 아닌 추론 프로세스 자체에 제약 조건(CCI)을 통합한 점이 매우 실용적이다.

### 한계 및 논의사항

1. **특징 추출기 의존성**: ARST 구조 자체는 뛰어나지만, 입력으로 들어가는 TeCNO나 ResNet-50 같은 특징 추출기의 성능에 따라 최종 결과가 크게 좌우된다.
2. **CCI의 하이퍼파라미터 $n$**: 단계 전환을 확정 짓는 윈도우 크기 $n=10$이 경험적으로 설정되었는데, 수술 종류나 환자마다 단계 전환 속도가 다를 수 있으므로 이에 대한 적응적 설정(adaptive setting)이 필요할 수 있다.
3. **데이터셋 제약**: Cholec80이라는 단일 데이터셋에서 검증되었으므로, 다른 종류의 수술 영상에서도 동일한 일반화 성능을 보일지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 복강경 수술 영상의 단계 인식을 위해 **자기회귀 기반의 Transformer 모델(ARST)**을 제안하였다. 이전 단계의 예측치를 현재 예측의 조건으로 사용하는 구조와 윈도우 기반의 **Banded Causal Mask**, 그리고 예측의 일관성을 강제하는 **CCI 전략**을 통해 SOTA 성능을 달성하고 실시간성(66 fps)을 확보하였다. 이 연구는 향후 문맥 인지형 컴퓨터 보조 수술 시스템 개발에 중요한 기여를 할 것으로 기대된다.
