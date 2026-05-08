# ATTENTION-FREE KEYWORD SPOTTING

Mashrur M. Morshed & Ahmad Omar Ahsan (2022)

## 🧩 Problem to Solve

본 논문은 키워드 검출(Keyword Spotting, KWS) 문제에서 최근 주류가 된 self-attention 메커니즘의 필수성에 대해 의문을 제기한다. KWS는 제한된 어휘의 음성 키워드를 식별하는 작업으로, 주로 항상 켜져 있는(always-on) 엣지 디바이스에서 동작해야 하므로 높은 정확도와 동시에 극도로 높은 연산 효율성이 요구된다.

최근 Vision Transformer(ViT)의 성공 이후, KWT(Keyword Transformer)나 AST(Audio Spectrogram Transformer)와 같이 self-attention 기반의 모델들이 KWS 분야에서도 뛰어난 성능을 보였다. 그러나 self-attention은 연산 복잡도가 높고 파라미터 수가 증가하는 경향이 있다. 따라서 본 연구의 목표는 self-attention 없이도 경쟁력 있는 성능을 낼 수 있는 효율적인 MLP 기반의 아키텍처를 제안하여, KWS 작업에서 self-attention이 정말로 대체 불가능한 요소인지 확인하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 컴퓨터 비전 분야에서 Transformer의 대안으로 제시된 gated MLP(gMLP) 구조를 KWS 작업에 도입하는 것이다.

1. **KW-MLP 제안**: self-attention을 완전히 제거하고 gated-MLP 블록을 사용하여 메모리 효율성을 극대화한 KW-MLP 아키텍처를 제안한다.
2. **파라미터 효율성 입증**: 0.5M 개 미만의 파라미터만으로 기존의 self-attention 기반 모델(KWT 등)과 대등한 성능을 달성함으로써, KWS 작업에서 MLP만으로도 충분한 표현력을 가질 수 있음을 보였다.
3. **모델 경량화**: 지식 증류(Knowledge Distillation)를 통해 최저 0.213M 개의 파라미터를 가진 초경량 모델을 설계하고, 여전히 높은 수준의 정확도를 유지함을 입증하였다.

## 📎 Related Works

### 기존 키워드 검출 연구

초기 KWS 접근 방식은 주로 Convolutional Neural Networks(CNN)를 사용했으며, 특히 depth-wise separable convolution을 이용한 경량 모델들이 제안되었다. 이후 attention 메커니즘을 결합한 RNN 기반 모델(MHAtt-RNN)이나 ResNet 변형 모델들이 등장하였다. 최근에는 ViT의 영향을 받은 KWT와 AST가 등장하여 state-of-the-art(SOTA) 성능을 기록하였으나, 이들은 상대적으로 많은 파라미터를 필요로 하거나 대규모 데이터셋(ImageNet, Audioset)의 사전 학습에 의존하는 경향이 있다.

### MLP 기반 비전 연구

최근 비전 분야에서는 self-attention의 필요성에 의문을 제기하는 연구들이 진행되었다. MLP-Mixer는 토큰 믹싱과 채널 믹싱만을 사용하여 경쟁력 있는 성능을 보였으며, ResMLP와 gMLP는 단순한 선형 투영(linear projection)과 게이팅(gating) 메커니즘만으로 Transformer에 근접한 성능을 달성하였다. 본 논문은 이러한 gMLP의 가능성을 오디오 도메인의 KWS 작업으로 확장한 것이다.

## 🛠️ Methodology

### 전체 파이프라인

KW-MLP의 입력은 Mel-frequency cepstrum coefficients(MFCC)이다. 입력 데이터 $X \in \mathbb{R}^{F \times T}$ (여기서 $F$는 주파수 빈, $T$는 시간 단계)를 $F \times 1$ 크기의 패치로 나누어 총 $T$개의 패치를 생성한다. 이후 이 패치들을 평탄화(flatten)하여 $X_0 \in \mathbb{R}^{T \times F}$로 만들고, 선형 투영 행렬 $P_0 \in \mathbb{R}^{F \times d}$를 통해 고차원 임베딩 공간으로 매핑한다.

$$X_E = X_0 P_0$$

이렇게 생성된 임베딩 $X_E$는 $L$개의 연속적이고 동일한 gated-MLP(gMLP) 블록을 통과하게 된다.

### gMLP 블록 구조 및 방정식

각 gMLP 블록은 임베딩 차원과 시간 차원 간의 투영을 반복 수행한다. 상세 과정은 다음과 같다.

1. **채널 투영 및 활성화**: 입력 $X_{in}$을 행렬 $U \in \mathbb{R}^{d \times D}$를 통해 투영하고 GELU 활성화 함수 $\sigma$를 적용한다.
    $$Z = \sigma(X_{in} U)$$

2. **Temporal Gating Unit (TGU)**: $Z$를 잔차(residual) 성분 $Z_r$과 게이트(gate) 성분 $Z_g \in \mathbb{R}^{T \times D/2}$로 분할한다. 게이트 성분 $Z_g$는 시간축에 대한 선형 투영 행렬 $G \in \mathbb{R}^{T \times T}$를 통과한 후, 잔차 성분 $Z_r$과 원소별 곱(element-wise product, $\odot$)을 수행한다.
    $$\tilde{Z} = g(Z) = Z_r \odot (Z_g^T G)^T$$
    실제 구현 시 시간 투영 연산은 `Conv1D(T, T, 1)` 레이어를 통해 효율적으로 처리된다.

3. **최종 투영 및 잔차 연결**: $\tilde{Z}$를 다시 원래의 임베딩 차원 $d$로 투영하는 행렬 $V$를 곱하고, 입력 $X_{in}$과 더해준다.
    $$X_{out} = \tilde{Z} V \oplus X_{in}$$

본 논문에서는 원본 gMLP와 달리 LayerNorm을 스킵 연결(skip-connection) 직전, 즉 두 번째 임베딩 투영 이후에 배치하는 것이 수렴 속도와 최적화 면에서 더 유리함을 발견하였다.

### 훈련 설정 및 지식 증류

- **하이퍼파라미터**: $L=12$, 임베딩 차원 $d=64$, 투영 차원 $D=256$을 기본으로 사용하며, 입력 MFCC 크기는 $40 \times 98$이다.
- **지식 증류(Knowledge Distillation)**: 깊이가 얕은 모델($L=10, 8, 6$)의 성능을 높이기 위해 $L=12$ 모델을 교사(Teacher) 모델로 사용하는 KD 기법을 적용하였다. 이때 온도 파라미터가 코사인 어닐링 규칙에 따라 감소하는 annealed KD 방식을 채택하였다.

## 📊 Results

### 실험 환경

- **데이터셋**: Google Speech Commands V2-12 및 V2-35 벤치마크를 사용하였다.
- **지표**: 정확도(Accuracy)와 파라미터 수(Parameters)를 주요 지표로 측정하였다.
- **비교 대상**: Att-RNN, Res-15, MHAtt-RNN, AST-S, AST-P, KWT-1/2/3 등이 비교 대상이 되었다.

### 주요 결과

- **정확도**: KW-MLP ($L=12$)는 V2-12에서 97.63%, V2-35에서 97.56%의 정확도를 기록하였다. 이는 self-attention 기반의 KWT-1(97.72% / 96.85%)과 대등하거나 오히려 능가하는 수준이다.
- **효율성**: KW-MLP의 파라미터 수는 0.424M 개로, KWT-1(0.607M)보다 적으며 AST-S(87M)에 비해서는 압도적으로 적다.
- **경량 모델**: KD를 적용한 가장 작은 모델($L=6$)은 단 0.213M 개의 파라미터만으로도 V2-12와 V2-35에서 각각 97.12%, 97.17%라는 높은 정확도를 달성하였다.

| Method | V2-12 Acc | V2-35 Acc | Params (M) |
| :--- | :---: | :---: | :---: |
| KWT-1 | 97.72 | 96.85 | 0.607 |
| **KW-MLP** | **97.63** | **97.56** | **0.424** |
| **KW-MLP (L=6, KD)** | **97.12** | **97.17** | **0.213** |

## 🧠 Insights & Discussion

### 시간 투영 행렬의 시각화

저자들은 Temporal Gating Unit의 가중치 행렬 $G$를 시각화하여 흥미로운 점을 발견하였다. 학습된 $G$ 행렬이 대각 행렬(diagonal), 단위 행렬(identity), 또는 토플리츠 행렬(Toeplitz matrix)과 유사한 형태를 띠고 있었다. 이는 KW-MLP가 KWS 작업에 필수적인 **이동 불변성(shift-invariance)**을 부분적으로 학습했음을 시사한다. 즉, 키워드가 1초의 오디오 클립 내 어느 시점에서 발생하더라도 동일하게 인식할 수 있는 능력을 MLP 구조가 스스로 획득한 것으로 해석할 수 있다.

### 강점 및 한계

- **강점**: self-attention 없이도 매우 적은 파라미터로 높은 성능을 내며, 복잡한 런타임 데이터 증강(resampling, time-shifting 등) 없이 Spectral Augmentation만으로도 빠른 학습이 가능하다.
- **한계**: 학습 속도를 높이기 위해 다양한 데이터 증강 기법을 충분히 탐색하지 않았다. 이로 인해 데이터 수가 상대적으로 적은 V2-12 작업에서 V2-35 작업보다 일반화 성능이 다소 떨어지는 모습이 관찰되었다.

## 📌 TL;DR

본 논문은 KWS 작업에서 self-attention을 제거하고 gated MLP 기반의 **KW-MLP**를 제안하여, 파라미터 수를 획기적으로 줄이면서도 기존 Transformer 기반 모델과 대등한 성능을 달성하였다. 특히 $L=6$ 모델은 0.213M 개의 파라미터만으로 높은 정확도를 보여, 자원 제한적인 엣지 디바이스 환경에서 KWS를 구현하는 데 매우 유용한 대안이 될 수 있음을 입증하였다.
