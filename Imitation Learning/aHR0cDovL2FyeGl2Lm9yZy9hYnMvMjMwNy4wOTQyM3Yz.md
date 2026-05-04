# Scaling Laws for Imitation Learning in Single-Agent Games

Jens Tuyls, Dhruv Madeka, Kari Torkkola, Dean P. Foster, Karthik Narasimhan, Sham Kakade (2024)

## 🧩 Problem to Solve

본 논문은 단일 에이전트 게임 환경에서 Imitation Learning (IL), 특히 Behavioral Cloning (BC)을 수행할 때 모델의 크기와 데이터의 양을 확장하는 것(Scaling up)이 성능에 어떤 영향을 미치는지 분석하는 것을 목표로 한다.

기존의 IL 연구들은 학습된 정책이 전문가의 행동을 완전히 복구하지 못하거나, 훈련 데이터의 가짜 상관관계에 의존하는 Causal Confusion 문제로 인해 실제 환경에서 전문가보다 훨씬 낮은 성능을 보이는 한계가 있었다. 하지만 자연어 처리(NLP) 분야의 대규모 언어 모델(LLM)에서 모델과 데이터의 규모를 키움으로써 비약적인 성능 향상을 이룬 Scaling Law 사례와 달리, IL 분야에서는 모델 및 데이터 크기가 성능에 미치는 영향에 대한 심도 있는 연구가 부족했다.

따라서 본 연구는 compute budget (FLOPs)에 따른 모델 크기와 데이터 크기의 최적 조합을 찾고, 이를 통해 성능 향상을 예측할 수 있는 Scaling Law가 IL에서도 성립하는지 검증하고자 한다. 특히, 단순한 Atari 게임뿐만 아니라 부분 관측성(Partial Observability)과 장기 의존성이 매우 강해 AI에게 매우 어려운 과제인 NetHack 게임을 통해 그 효용성을 증명하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여 사항은 다음과 같다.

1. **IL Scaling Law의 발견**: 단일 에이전트 게임에서 BC의 Cross-entropy loss와 Mean return이 compute budget (FLOPs)에 따라 매끄럽게 변화하며, 이를 Power Law(거듭제곱 법칙) 형태로 설명할 수 있음을 보였다.
2. **손실 함수와 성능의 상관관계 규명**: 학습 시의 Cross-entropy loss와 환경 내 실제 Return 사이의 강한 상관관계를 확인하였으며, 이를 통해 loss의 감소가 곧 에이전트의 성능 향상으로 이어짐을 정량적으로 증명하였다.
3. **NetHack SOTA 달성 및 예측 가능성 입증**: 도출된 Scaling Law를 사용하여 NetHack 에이전트의 최적 모델 크기와 데이터 양을 예측하였으며, 실제로 이를 통해 구현한 에이전트가 기존 offline 설정의 SOTA(State-of-the-art) 성능을 약 1.7배 상회하는 성과를 거두었다.
4. **부분 관측성의 영향 분석**: NetHack과 같은 복잡한 환경에서 Context length의 확장과 정보의 일치(Information Parity, 예: 인벤토리 정보 포함)가 BC 성능 향상에 필수적임을 밝혔다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 배경으로 한다.

- **Scaling Laws**: Kaplan et al. (2020)과 Hoffmann et al. (2022)은 LLM에서 모델 크기, 데이터셋 크기, compute budget 사이의 관계를 Power Law로 정의하였다. 본 논문은 이러한 직관을 IL 영역으로 확장하였다.
- **Reinforcement Learning (RL) Scaling**: Hilton et al. (2023)은 RL에서의 Scaling Law를 연구하였으나, IL에 대해서는 다루지 않았다.
- **NetHack Challenge**: NetHack은 절차적으로 생성되는 맵과 부분 관측성 때문에 매우 어려운 환경이다. 기존의 neural agent들은 전문가 시스템인 AutoAscend에 비해 턱없이 낮은 성능을 보였으며, 본 논문은 이를 '규모의 확장'이라는 관점에서 해결하려 시도하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 학습 목표

본 연구에서는 전문가의 궤적 데이터 $\mathcal{D}$를 이용하여 전문가 정책 $\pi$를 복제하는 Behavioral Cloning (BC)을 사용한다. 학습자는 다음과 같은 Cross-entropy loss를 최소화하도록 최적화된다.

$$L(\theta) = -\mathbb{E}_{(h_t, a_t) \sim \mathcal{D}} [\log \pi_\theta(a_t | h_t)]$$

여기서 $h_t$는 과거의 상태와 행동의 이력(history)을 포함한다.

### Scaling 분석 방법론

Compute budget ($C$, FLOPs)에 따른 최적의 모델 크기($N$)와 데이터 크기($D$)를 찾기 위해 두 가지 접근 방식을 사용한다.

**1. isoFLOP profiles**
특정 FLOPs 예산 내에서 모델 크기를 다양하게 변화시키며 검증 손실(Validation Loss)을 측정한다. 각 FLOPs 지점마다 손실이 최소가 되는 최적의 $N$과 $D$를 찾고, 이를 바탕으로 다음과 같은 Power Law 회귀식을 도출한다.

$$N_{opt} = a_N C^{\alpha} + b_N, \quad D_{opt} = a_D C^{\beta} + b_D, \quad L_{opt} = a_L C^{\gamma} + b_L$$

**2. Parametric fit**
모든 데이터 포인트를 사용하여 다음과 같은 이차 형식의 함수로 loss를 근사한다.

$$\log \hat{L}(N, D) = \beta_0 + \beta_N \log N + \beta_D \log D + \beta_{N^2} (\log N)^2 + \beta_{ND} \log N \log D + \beta_{D^2} (\log D)^2$$

이후 Lagrange multipliers를 사용하여 $C \approx 6ND$ (Transformer 기준) 제약 조건 하에서 $L$을 최소화하는 $N_{opt}$와 $D_{opt}$를 수학적으로 유도한다.

### 아키텍처 및 계산량 정의

- **Atari**: CNN 기반 에이전트를 사용하며, CNN 채널 수와 최종 linear layer의 너비를 확장하여 모델 크기를 조절하였다.
- **NetHack**: Transformer 기반 에이전트를 사용하며, hidden size와 layer 수를 확장하였다. Compute budget 계산을 위해 $\text{FLOPs} \approx 6ND$ 공식을 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: Atari 8종 게임 및 NetHack (NLD-AA-L 데이터셋, 약 55B 샘플).
- **지표**: Validation Loss 및 Mean Return.

### 주요 결과

1. **Scaling Law 성립**: Atari와 NetHack 모든 게임에서 FLOPs가 증가함에 따라 $L_{opt}$가 매끄럽게 감소하는 Power Law 경향이 관찰되었다.
2. **Loss-Return 상관관계**: 최적 loss $L$과 평균 return $R$ 사이의 관계가 다음과 같은 역거듭제곱 법칙(inverse power law)으로 설명됨을 확인하였다.
   $$R = (a_{RL} (1/L)^{\delta} + b_{RL})^{-1}$$
   이는 loss를 줄이는 것이 실제 환경 성능 향상으로 직접 연결됨을 의미한다.
3. **NetHack 성능 향상**: Scaling Law를 통해 예측된 최적 모델(138M parameters)과 데이터(40B samples)를 사용하여 학습시킨 결과, 기존 offline SOTA 모델(diff History LM)의 점수인 4504점을 크게 상회하는 **7784점**을 기록하였다 (약 1.7배 향상).
4. **RL 확장**: IMPALA 알고리즘을 이용한 RL 설정에서도 모델 크기와 환경 상호작용 횟수가 FLOPs에 따라 Power Law 형태로 스케일링됨을 확인하였다.

## 🧠 Insights & Discussion

### 부분 관측성(Partial Observability)의 중요성

연구진은 NetHack에서 성능 정체 구간이 발생하는 이유를 분석하며 두 가지 핵심 요소를 발견하였다.

- **Context Length**: Context length를 128에서 4096까지 확장했을 때 loss가 유의미하게 감소하였다. 이는 전문가 정책(AutoAscend)이 매우 긴 이력을 참조하므로, 학습자 또한 긴 context를 가져야 함을 시사한다.
- **Information Parity**: 기존 연구에서 제외되었던 '인벤토리' 정보를 추가했을 때 성능이 향상되었다. 학습자가 전문가가 사용하는 정보와 동일한 정보에 접근할 수 있어야(Information Parity) 온전한 복제가 가능하다는 것을 보여준다.

### 비판적 해석 및 한계

- **Reward Density**: 본 연구는 보상이 비교적 조밀한(dense) 환경에서 수행되었다. 보상이 매우 희소한(sparse) 환경(예: Montezuma's Revenge)에서도 동일한 Scaling Law가 성립할지는 미지수이며, 이 경우 다른 대리 지표(proxy metrics)가 필요할 수 있다.
- **데이터 가용성**: 본 연구에서는 시뮬레이터를 통해 대량의 전문가 데이터를 생성할 수 있었으나, 실제 인간 데이터에 의존해야 하는 경우 데이터 확보가 병목 지점이 될 수 있다.
- **전문가 품질**: 전문가의 수준이 높을수록 Scaling Law가 수렴하는 상한선(ceiling)이 높아지지만, Scaling Law라는 경향성 자체는 전문가의 수준과 관계없이 일정하게 나타나는 특징을 보였다.

## 📌 TL;DR

본 논문은 단일 에이전트 게임의 Imitation Learning에서 모델 크기와 데이터 양을 확장했을 때 성능이 Power Law를 따라 예측 가능하게 향상된다는 것을 입증하였다. 특히 NetHack이라는 고난도 환경에서 최적의 스케일을 예측하여 적용함으로써 기존 offline SOTA 성능을 1.7배 끌어올렸으며, 이를 통해 IL에서도 LLM과 유사한 Scaling Law 전략이 유효함을 보여주었다. 이 연구는 향후 복잡한 환경의 에이전트를 학습시킬 때 무작정 모델을 키우기보다, compute budget에 따른 최적의 모델-데이터 조합을 계산하여 효율적으로 학습시킬 수 있는 이론적 근거를 제공한다.
