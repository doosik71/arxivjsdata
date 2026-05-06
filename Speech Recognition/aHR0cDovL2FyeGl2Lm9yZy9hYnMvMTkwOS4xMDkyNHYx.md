# Understanding Semantics from Speech Through Pre-training

Pengwei Wang, Liangchen Wei, Yong Cao, Jinghui Xie, Yuji Cao, Zaiqing Nie (2019)

## 🧩 Problem to Solve

본 논문은 음성 신호에서 직접 의미를 추론하는 End-to-end Spoken Language Understanding (SLU) 시스템의 성능 향상을 목표로 한다. 전통적인 SLU 방식은 Automatic Speech Recognition (ASR)을 통해 음성을 텍스트로 변환한 뒤, Natural Language Understanding (NLU) 모듈을 통해 의미를 분석하는 파이프라인 구조를 가진다. 그러나 이 방식은 ASR 단계에서 발생한 오류가 NLU 단계로 전이되는 Error Propagation 문제가 발생하여 전체 시스템의 정확도 상한선이 제한되는 치명적인 단점이 있다.

이를 해결하기 위해 제안된 End-to-end SLU 방식은 중간 텍스트 표현 없이 오디오 특징에서 직접 의미를 추론함으로써 오류 전이 문제를 제거한다. 하지만 End-to-end SLU 시스템에서 Acoustic Model (AM) 컴포넌트는 ASR 데이터를 통해 사전 학습(Pre-training)이 가능하지만, 실제 의미를 학습하는 SLU 컴포넌트는 태스크 특화된 소량의 레이블링 된 데이터에만 의존해야 한다는 한계가 있다. 따라서 본 논문은 대규모 비정형 오디오 데이터를 활용하여 SLU 컴포넌트가 의미적 특징(Semantic features)을 사전에 학습할 수 있도록 하는 대규모 비지도 사전 학습 방법을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 End-to-end SLU 시스템의 SLU 컴포넌트를 위해 최초로 대규모 비지도 사전 학습(Large-scale unsupervised pre-training)을 도입했다는 점이다. 특히, 텍스트와는 다른 음성 특징인 Phoneme Posterior Sequence의 특성을 반영하여 다음과 같은 설계 아이디어를 제시하였다.

1. **BERT-PLM 제안**: Permutation Language Modeling (PLM) 목적 함수를 기반으로 하되, BERT 네트워크의 완전한 양방향 문맥 정보(Full bi-directional context)를 활용할 수 있는 새로운 사전 학습 모델인 BERT-PLM을 제안하였다.
2. **계산 효율성 및 문맥 정보 최적화**: 기존 XLNET의 마스킹 전략이 가지는 불완전한 문맥 정보 활용 문제를 해결하기 위해, 순열(Permutation)을 조합(Combination)의 관점으로 재해석하여 효율적인 회귀 목적 함수로 변환하였다.
3. **음성 특화 처리**: 의미 없는 무음 구간인 "SIL" (Silence) 포네마가 사전 학습의 목적 함수를 방해하는 문제를 해결하기 위해, 타겟 시퀀스에서 주요 "SIL" 슬라이스를 제외하는 전략을 도입하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **End-to-End SLU**: 기존 연구들은 오디오 인코더를 강화하거나 SincNet 등을 사용하여 성능을 높이려 했다. 일부 연구에서는 AM 컴포넌트를 ASR 타겟으로 사전 학습하여 성능을 개선했으나, 이는 음향적 특징을 잡는 것일 뿐 SLU 컴포넌트가 필요로 하는 의미적 특징을 학습하는 것과는 거리가 멀다.
- **NLU 사전 학습**: BERT, GPT, XLNET 등 텍스트 기반의 대규모 사전 학습 모델들이 NLU 분야에서 혁신적인 성능 향상을 가져왔다. 하지만 이러한 모델들은 유니그램(Unigram) 형태의 텍스트 시퀀스를 가정하므로, 분포 벡터 형태인 Phoneme Posterior Sequence에 그대로 적용하기 어렵다.

### 차별점

본 논문은 사전 학습의 대상을 AM 컴포넌트가 아닌 **SLU 컴포넌트**로 확장하였다는 점이 가장 큰 차별점이다. 또한, 텍스트 모델인 BERT의 `[MASK]` 토큰 방식이 가지는 도메인 간 불일치 문제와 XLNET의 불완전한 문맥 활용 문제를 해결한 BERT-PLM 구조를 통해 음성 도메인에 최적화된 사전 학습 체계를 구축하였다.

## 🛠️ Methodology

### 전체 시스템 구조

전체 모델은 **Acoustic Model (AM) 컴포넌트**와 **SLU 컴포넌트**로 구성된다. AM 컴포넌트는 raw audio를 입력받아 포네마 분포인 Phoneme Posterior를 출력하며, SLU 컴포넌트는 이 분포 시퀀스를 입력받아 최종 의미(Semantic meaning)를 분류한다.

### 1. Acoustic Model (AM) 컴포넌트

본 연구에서는 두 가지 AM 구현체를 사용하여 SLU 사전 학습의 강건성을 검증하였다.

- **SincNet**: Raw audio를 처리하기 위해 SincNet 레이어를 사용하고, 이후 Convolution 및 Bi-directional GRU 레이어를 통해 포네마 및 단어 레벨의 로짓을 출력한다.
- **DFSMN**: CTC Loss를 사용하여 학습된 모델로, 다른 구조에서도 SLU 사전 학습이 효과적인지 확인하기 위해 사용되었다.

### 2. SLU 컴포넌트

SLU 컴포넌트는 Transformer Encoder 네트워크를 사용한다. AM에서 출력된 포네마 분포 $h_{phoneme}$는 다음과 같은 방식으로 토큰 임베딩 $v_{token}$으로 변환된다.

$$v_{token} = E_{phoneme} \cdot h_{phoneme}$$

여기서 $E_{phoneme}$은 포네마 임베딩 행렬이다. 이는 포네마 분포를 가중치로 하여 포네마 임베딩들을 가중 평균(Weighted pooling)하는 것과 같다. 이후 Relative Positional Encoding을 적용한 Transformer 블록을 거치며, 최종적으로 Pooling Network를 통해 분류를 위한 벡터 표현을 생성한다.

### 3. BERT-PLM 사전 학습

#### Permutation Language Modeling (PLM)

기존의 PLM은 타겟 시퀀스 $x$의 순열 $z$에 대해, 특정 절단 지점 $c$ 이후의 시퀀스를 이전 시퀀스에 기반해 예측하는 것을 목표로 한다.

$$\log p_{\theta}(x_{z_{>c}} | x_{z_{\leq c}}) = \sum_{t=c+1}^{|z|} \log p_{\theta}(x_{z_t} | x_{z_{<t}})$$

#### BERT-PLM의 제안 및 증명

기존 XLNET 방식은 병렬 계산을 위해 마스킹을 사용하므로 각 시점 $t$에서 전체 문맥을 볼 수 없는 문제가 있다. 본 논문은 순열의 기대값이 조합(Combination)의 기대값과 같다는 것을 증명하여, 순열 대신 조합을 샘플링하는 방식으로 문제를 전환하였다.

$$\mathbb{E}_{z \sim Z_T} [\log p_{\theta}(x_{z_{>c}} | x_{z_{\leq c}})] = \mathbb{E}_{|x| \sim [1, T-c]} [\log p_{\theta}(x | \hat{x})]$$

여기서 $\hat{x}$는 문맥 서브시퀀스(Context subsequence)의 조합이다. 이를 통해 모델은 $\hat{x}$에 대해 완전한 양방향 Transformer Encoder를 사용할 수 있게 되어, 모든 시점에서 전체 문맥 정보를 활용할 수 있다.

#### 구현 세부 사항 및 "SIL" 처리

- **입력 처리**: 타겟 서브시퀀스 $x$ 부분을 마스킹하고, 해당 위치의 임베딩을 학습 가능한 벡터 $w$로 대체하여 위치 정보는 유지하되 정답 정보는 차단한다.
- **SIL 포네마 제외**: 무음 구간인 "SIL"을 예측하는 것은 의미가 없으므로, 사전 학습의 타겟 서브시퀀스 $x$를 샘플링할 때 주요 "SIL" 타임 슬라이스를 제외한다.
- **학습 절차**: 대규모 비정형 오디오 데이터로 사전 학습을 진행한 후, AM 컴포넌트의 가중치를 고정(Frozen)한 상태에서 태스크 특화 데이터로 파인튜닝(Fine-tuning)한다. 이때 사전 학습의 PLM 손실 함수를 함께 사용하여 Regularization 효과를 얻는다.

## 📊 Results

### 실험 설정

- **데이터셋**: 사전 학습을 위해 VoxCeleb, LibriSpeech, Common Voice 등 4,000시간 이상의 오디오 데이터를 사용하였다. 성능 평가는 Fluent Speech Commands(영어)와 In-House 데이터셋(중국어)에서 수행되었다.
- **비교 대상 (Baselines)**:
  - **ASR + NLU**: 전통적인 파이프라인 방식.
  - **Non-Pretraining**: 사전 학습 없이 학습한 End-to-end 모델.
  - **SOTA**: Lugosch et al. (2019)이 제안한 최신 End-to-end 모델.

### 주요 결과

1. **오류율 감소**: Fluent Speech Command 데이터셋에서 BERT-PLM은 $1.05\%$의 에러율을 기록하여, SOTA 모델($1.2\%$) 대비 약 $12.5\%$의 에러 감소율(Error Reduction)을 보였으며, Non-Pretraining 모델($1.95\%$)보다는 압도적으로 우수한 성능을 보였다.
2. **F1 Score 향상**: In-House 데이터셋에서도 Non-Pretraining 대비 Macro-F1 ($78.30\% \rightarrow 79.36\%$) 및 Micro-F1 ($89.38\% \rightarrow 90.45\%$) 점수가 모두 향상되었다.
3. **데이터 양에 따른 효과**: 파인튜닝 데이터의 양이 적을수록 사전 학습의 이득이 크게 나타났으며, 데이터가 많아질수록 그 차이가 줄어드는 경향을 보였다 (Figure 4). 이는 레이블링 된 데이터가 부족한 상황에서 사전 학습이 매우 효과적임을 시사한다.
4. **마스킹 비율 분석**: 타겟 프레임의 마스킹 비율을 $15\%$로 설정했을 때 가장 좋은 성능을 보였으며, 너무 낮으면(5~10%) 과적합(Overfitting)이 발생하고, 너무 높으면(20%) 학습 난이도가 지나치게 높아져 성능이 하락하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 음성 인식의 중간 결과물인 Phoneme Posterior의 특성을 정확히 파악하여, 텍스트 기반의 BERT/XLNET 구조를 음성 도메인에 맞게 변형한 점이 매우 뛰어나다. 특히 순열 문제를 조합 문제로 치환하여 완전한 양방향 문맥을 활용하게 한 이론적 접근과 "SIL" 포네마 처리를 통한 실용적 접근이 조화를 이루어 성능 향상을 이끌어냈다.

### 한계 및 논의사항

- **AM 의존성**: SLU 컴포넌트의 입력이 AM의 출력에 의존하므로, AM의 성능이 극도로 낮을 경우 사전 학습의 효과가 제한될 가능성이 있다. 비록 본 논문에서 다양한 AM 모델을 사용해 강건성을 입증했으나, AM-SLU 간의 공동 최적화(Joint Optimization) 가능성에 대해서는 다루지 않았다.
- **계산 자원**: 4,000시간 이상의 대규모 데이터를 학습시키기 위해 상당한 GPU 자원이 소모되었을 것으로 보이며, 이에 대한 효율적인 학습 방법론에 대한 추가 논의가 필요하다.
- **언어 확장성**: 영어와 중국어 두 언어에서 검증하였으나, 포네마 구조가 매우 다른 다른 언어에서도 동일한 수준의 효과가 나타날지는 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 End-to-end SLU 시스템의 SLU 컴포넌트를 위한 **최초의 대규모 비지도 사전 학습 방법론인 BERT-PLM**을 제안한다. Phoneme Posterior Sequence에 최적화된 조합 기반의 Permutation Language Modeling 목적 함수를 통해 완전한 양방향 문맥 정보를 학습하며, 무음 구간(SIL) 처리 전략을 통해 학습 효율을 높였다. 실험 결과, 기존 SOTA 모델 대비 에러율을 $12.5\%$ 감소시켰으며, 특히 레이블 데이터가 부족한 환경에서 탁월한 성능 향상을 보였다. 이는 향후 데이터 수집이 어려운 특수 도메인의 음성 명령 시스템 구축에 매우 중요한 역할을 할 것으로 기대된다.
