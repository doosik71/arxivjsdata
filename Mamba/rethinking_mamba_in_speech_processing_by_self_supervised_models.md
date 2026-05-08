# Rethinking Mamba in Speech Processing by Self-Supervised Models

Xiangyu Zhang, Jianbo Ma, Mostafa Shahin, Beena Ahmed, Julien Epps (2024)

## 🧩 Problem to Solve

최근 Mamba 기반 모델은 컴퓨터 비전, 자연어 처리, 그리고 음성 처리 분야에서 뛰어난 성능을 보여주고 있다. 그러나 음성 처리의 세부 작업(task)에 따라 Mamba 모델의 성능 편차가 크게 나타나는 현상이 관찰되었다. 구체적으로, 음성 향상(Speech Enhancement)이나 스펙트럼 재구성(Spectrum Reconstruction)과 같은 작업에서는 Mamba 모델 단독으로도 우수한 성능을 내지만, 자동 음성 인식(Automatic Speech Recognition, ASR)과 같은 작업에서는 Attention 기반 모델의 성능을 넘어서기 위해 추가적인 모듈(예: Decoder)이 필수적이다.

본 논문은 이러한 성능 차이의 원인을 규명하고자 하며, Mamba 기반 모델이 음성 처리의 "재구성(Reconstruction)" 작업에는 최적화되어 있으나, ASR과 같은 "분류(Classification)" 작업에서는 내부적으로 재구성 단계가 선행되어야 하며 이를 위해 추가 모듈이 필요하다는 가설을 세우고 이를 검증하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 정보 이론(Information Theory)의 관점에서 Mamba 모델의 내부 표현(Representation)을 분석하여, 모델의 성능과 상호 정보량(Mutual Information, MI)의 변화 패턴 사이의 상관관계를 밝혀낸 것이다.

연구진은 입력 데이터 $X$와 모델의 중간 레이어 특징 $T_i$ 사이의 상호 정보량 $I(X; T_i)$를 측정함으로써, 성능이 좋은 모델은 특징이 '감소 후 증가'하는 재구성 패턴을 보이는 반면, 성능이 낮은 모델은 특징이 '지속적으로 감소'하는 경향을 보인다는 점을 입증하였다. 이를 통해 Mamba 모델이 음성 인식과 같은 복잡한 작업에서 성공하기 위해서는 정보의 압축 후 다시 복원하는 재구성 프로세스가 필요함을 이론적으로 제시하였다.

## 📎 Related Works

기존의 Transformer 기반 모델은 뛰어난 성능을 보이지만, Self-attention 메커니즘의 계산 복잡도로 인해 긴 시퀀스를 처리하는 데 한계가 있다. 이를 해결하기 위해 Structured State Space Models (S4)가 제안되었으며, Mamba는 여기에 시변(time-varying) 메커니즘을 통합하여 선형 시간 복잡도로 시퀀스를 모델링하며 다양한 분야에서 성과를 거두었다.

음성 처리 분야에서도 Mamba를 적용한 사례가 늘고 있다. 특히 음성 향상이나 SSAST(Self-Supervised Audio Spectrogram Transformer) 프레임워크를 Mamba로 대체한 SSAMBA 모델 등은 매우 효율적인 성능을 보였다. 그러나 ASR 작업에서는 단순히 Mamba를 사용하는 것만으로는 부족하며, Conformer와 같은 추가적인 구조나 디코더가 결합되었을 때 비로소 Attention 기반 모델을 상회하는 결과가 도출되었다. 본 논문은 이러한 기존 현상을 단순한 관찰을 넘어 정보 이론적으로 분석했다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. 상호 정보량 추정 (Mutual Information Estimation)

본 논문은 고차원 연속 확률 변수 간의 상호 정보량을 측정하기 위해 MINE(Mutual Information Neural Estimation) 기법을 사용한다. 입력 지역 특징(local features) $X \in \mathbb{R}^{L \times D}$와 모델의 $i$번째 레이어 특징 $T_i \in \mathbb{R}^{L \times D}$ 사이의 상호 정보량은 다음과 같이 정의된다.

$$I_i(X; T_i) = H(X) - H(X|T_i) = D_{KL}(P(X, T_i) \parallel P(X) \otimes P(T_i))$$

여기서 $D_{KL}$은 KL-divergence를 의미한다. 하지만 실제 계산의 어려움으로 인해 MINE에서는 다음과 같은 통계 네트워크 $\psi_\theta$를 통해 $I$를 추정한다.

$$I_\Theta(X; T_i) = \sup_{\theta \in \Theta} \mathbb{E}_{P_{X, T_i}}[\psi_\theta] - \log(\mathbb{E}_{P_X \times P_{T_i}}[e^{\psi_\theta}])$$

분석을 위해 LibriSpeech 데이터셋에서 1,000개의 샘플을 무작위로 추출하여 각 레이어별 평균 상호 정보량 $\bar{I}_i(X; T_i)$를 계산하였다.

### 2. 분석 대상 및 실험 설계

가설 검증을 위해 세 가지 서로 다른 성격의 모델을 분석하였다.

- **ConBiMamba**: ASR 작업 수행 모델로, Decoder 유무에 따른 MI 패턴 변화를 관찰한다.
- **SSAMBA**: 스펙트럼 패치 재구성 작업 모델로, 기본적으로 높은 성능을 보이는 모델의 MI 패턴을 분석한다.
- **Mamba-HuBERT**: Transformer 레이어를 ConBiMamba로 대체하여 학습시킨 자기지도학습 모델이다. HuBERT의 특성상 분류 작업에 가깝기 때문에, 단독 사용 시와 downstream 모델(Conformer) 결합 시의 성능 및 MI 변화를 비교한다.

## 📊 Results

### 1. 음성 인식 작업 (ConBiMamba)

- **결과**: Decoder가 없는 경우 Word Error Rate (WER)가 상승하며 성능이 저하되었다. 이때 MI 패턴은 레이어가 깊어질수록 지속적으로 감소하는 경향을 보였다. 반면, Decoder를 추가했을 때 성능이 크게 향상되었으며, MI 패턴은 초기에 감소하다가 이후 다시 증가하는 'V'자 형태의 재구성 패턴을 나타냈다.
- **의미**: Mamba가 ASR에서 성공하려면 정보를 다시 복원하는 과정이 필요함을 시사한다.

### 2. 스펙트럼 재구성 작업 (SSAMBA)

- **결과**: SSAMBA-base 및 SSAMBA-tiny 모델 모두에서 MI가 처음에는 감소하다가 이후 뚜렷하게 증가하는 패턴이 관찰되었다. 특히 최종 레이어에서의 MI 증가폭이 ASR 모델보다 훨씬 컸다.
- **의미**: 재구성 작업의 본질 자체가 입력 정보의 복원을 포함하고 있으므로, Mamba가 단독으로도 매우 효율적으로 작동함을 보여준다.

### 3. Mamba-HuBERT 분석

- **결과**: Mamba-HuBERT를 단독으로 fine-tuning 했을 때는 MI가 지속적으로 감소하며 HuBERT(Transformer 기반)보다 낮은 성능을 보였다. 그러나 뒤에 Conformer 모델을 추가로 연결했을 때, MI 패턴이 '감소 후 증가'하는 재구성 패턴으로 변하며 표준 HuBERT + Conformer 조합에 근접하는 성능을 달성하였다.
- **정량적 지표 (WER)**: Mamba-HuBERT (Layer 9 특징 기반) 단독일 때는 Test Clean 기준 12.32%였으나, Conformer 추가 시 9.2%까지 낮아져 HuBERT + Conformer(9.3%)와 유사한 수준이 되었다.

## 🧠 Insights & Discussion

본 연구는 Mamba 모델이 가진 근본적인 특성이 **"정보의 압축과 복원(Reconstruction)"**에 최적화되어 있다는 통찰을 제공한다. 정보 이론의 'Information Bottleneck' 관점에서 볼 때, 신경망은 불필요한 정보를 제거하여 압축하는 과정을 거친다. 하지만 음성 인식과 같은 복잡한 분류 작업에서는 단순히 정보를 압축하는 것만으로는 부족하며, 특정 수준까지 압축된 표현을 다시 유의미한 형태로 재구성하는 단계가 필수적이다.

Mamba 모델 단독으로는 이 재구성 단계가 부족하여 MI가 계속 감소하는 경향을 보이지만, Decoder나 Conformer와 같은 추가 모듈이 결합되면 이들이 재구성 역할을 수행하여 MI를 다시 끌어올리고, 결과적으로 높은 성능을 내게 된다. 이는 Mamba를 음성 처리의 다양한 태스크에 적용할 때, 단순한 레이어 대체보다는 태스크의 성격(재구성 vs 분류)에 맞는 아키텍처 설계가 중요함을 시사한다.

## 📌 TL;DR

이 논문은 Mamba 모델이 음성 처리의 **재구성(Reconstruction) 작업에는 강하지만, 분류(Classification) 작업에는 약하다**는 가설을 세우고, 이를 **상호 정보량(Mutual Information) 분석**을 통해 증명하였다. 분석 결과, 성능이 좋은 Mamba 기반 모델은 레이어 깊이에 따라 정보량이 '감소 후 증가'하는 패턴을 보였으며, ASR과 같은 작업에서 성능을 높이려면 Decoder나 Conformer 같은 추가 모듈이 이 재구성 역할을 수행해야 함을 밝혔다. 이는 향후 Mamba 기반 음성 모델 설계 시 단순 구조 대체가 아닌, 정보 흐름을 고려한 아키텍처 설계가 필요함을 시사하는 중요한 연구이다.
