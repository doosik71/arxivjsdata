# Text-Aware Adapter for Few-Shot Keyword Spotting

Youngmoon Jung, Jinyoung Lee, Seungjin Lee, Myunghun Jung, Yong-Hyeok Lee, Hoon-Young Cho (2024)

## 🧩 Problem to Solve

본 논문은 텍스트 등록 기반의 유연한 키워드 스포팅(Text-enrolled Flexible KWS, 이하 TF-KWS) 시스템에서 특정 키워드에 대한 인식 성능을 향상시키는 문제를 다룬다. TF-KWS는 사용자가 음성 샘플을 직접 녹음할 필요 없이 텍스트 입력만으로 키워드를 등록할 수 있어 편의성이 높지만, 특정 키워드에 최적화되어 대량의 데이터로 학습된 키워드 전용 모델에 비해 성능이 떨어진다는 한계가 있다.

특히, 특정 키워드에 대해 대량의 음성 데이터를 수집하는 것은 현실적으로 어렵기 때문에, 매우 적은 수의 음성 샘플만으로도 모델을 최적화할 수 있는 Few-shot learning 관점의 접근이 필요하다. 따라서 본 연구의 목표는 사전 학습된 TF-KWS 모델을 기반으로, 한정된 음성 데이터를 사용하여 특정 타겟 키워드에 빠르게 적응(Adaptation)시킬 수 있는 효율적인 전이 학습 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 학습된 Acoustic Encoder의 일부 파라미터만을 미세 조정하는 **Text-Aware Adapter (TA-adapter)**를 도입하는 것이다.

TA-adapter의 중심적인 직관은 텍스트 인코더에서 생성된 Text Embedding (TE)을 타겟 키워드의 대표 벡터로 활용하여, 이를 통해 Acoustic Encoder가 해당 키워드의 특성을 더 잘 추출하도록 조건화(Conditioning)하는 것이다. 이를 위해 본 논문은 계산 효율성이 높고 오버피팅 위험이 적은 가벼운 어댑터 구조를 제안하며, 핵심 구성 요소인 TCFM, FW-adapter, TE classifier를 통해 적은 파라미터 수정만으로도 높은 성능 향상을 달성한다.

## 📎 Related Works

기존의 유연한 KWS 연구들은 주로 Meta-learning이나 Self-supervised learning (SSL) 모델을 결합하여 Few-shot 성능을 높이려 했으며, 주로 사전 학습된 인코더 뒤에 키워드 전용 분류 층을 추가하는 방식을 사용했다.

또한, 컴퓨터 비전과 NLP 분야에서 제안된 Adapter 방식은 모델의 중간 레이어에 작은 모듈을 삽입하여 고정된 사전 학습 모델을 특정 태스크에 적응시키는 기법으로, 파라미터 효율성이 높다는 장점이 있다. 본 논문은 이러한 Adapter 개념을 TF-KWS에 적용하였다.

특히, 기존의 AdaKWS는 Adaptive Instance Normalization (AdaIN)을 사용하여 키워드 정보를 주입하였으나, 이는 모든 인코더를 처음부터 함께 학습시켜야 하므로 많은 양의 데이터가 필요하고 파라미터 수가 많아 Few-shot 환경에는 부적합하다는 한계가 있다. 본 논문의 TA-adapter는 텍스트 인코더를 동결하고 Learnable Activation Function (LAF)을 사용하여 훨씬 적은 파라미터로 조건화를 수행함으로써 이 문제를 해결한다.

## 🛠️ Methodology

### 전체 시스템 구조

TA-adapter는 사전 학습된 ECAPA-TDNN 기반의 Acoustic Encoder와 Text Encoder를 기반으로 하며, 특정 키워드에 적응하기 위해 다음의 세 가지 구성 요소를 추가하거나 수정한다.

### 1. Text-conditioned Feature Modulation (TCFM)

TCFM은 Text Embedding (TE)의 정보를 Acoustic Encoder에 주입하여 타겟 키워드에 최적화된 특징을 추출하도록 돕는다. 기존 ECAPA-TDNN의 ReLU 활성화 함수를 **Learnable Activation Function (LAF)**으로 대체하여 구현한다.

LAF는 여러 개의 기본 활성화 함수 $\{A_1, \dots, A_a\}$의 가중치 합으로 정의되며, 가중치는 TE에 의해 결정된다.
$$s = \text{softmax}(TE \cdot w + b), \quad y = \text{LAF}(h|TE) = \sum_{i=1}^{a} s_i A_i(h)$$
여기서 $TE$는 텍스트 임베딩, $w$와 $b$는 학습 가능한 파라미터, $h$는 입력 특징값이다.

기존의 AdaIN 방식이 채널별 스케일과 바이어스를 직접 생성하여 큰 파라미터 비용이 드는 반면, TCFM은 TE를 통해 활성화 함수의 조합 비중만을 조절하므로 매우 효율적이다. 본 연구에서는 ELU, ReLU, Swish 등 6가지 기본 함수를 사용하였다.

### 2. Feature Weight Adapter (FW-adapter)

FW-adapter는 사전 학습된 모델이 이미 핵심적인 특징들을 학습했다는 가정하에, 저수준 특징이 고수준 특징으로 집계되는 방식만을 미세 조정한다. 이를 위해 Acoustic Encoder 내부의 **Batch Normalization (BN)** 레이어와 **Squeeze-and-Excitation (SE)** 모듈의 파라미터만을 선택적으로 업데이트한다. 이는 적은 데이터로도 오버피팅을 방지하면서 도메인 적응을 가능하게 한다.

### 3. Text Embedding Classifier

타겟 키워드 $k$에 대한 최종 판별을 위해 TE classifier를 사용한다. 새로운 가중치 벡터를 처음부터 학습하는 대신, 사전 학습 단계에서 이미 키워드의 대표 벡터로 학습된 **TE를 분류기의 가중치 $\theta_k$로 고정**하여 사용한다.

최종 출력 확률 $p(k|x)$는 Acoustic Embedding (AE)과 TE 간의 코사인 유사도 $S(\cdot, \cdot)$를 기반으로 다음과 같이 계산된다.
$$p(k|x) = \sigma(\theta_k \cdot AE) = \sigma(S(TE, AE))$$
여기서 $\sigma$는 시그모이드 함수이며, AE와 TE는 모두 $L_2$-normalized 상태이다. 학습은 Binary Cross-Entropy loss를 사용하여 수행된다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands (GSC) V2 데이터셋의 35개 키워드 사용 (사전 학습 시 본 키워드 10개, 보지 못한 키워드 25개).
- **시나리오**: 5-shot, 10-shot, 15-shot의 저자원 환경 설정.
- **평가 지표**: Average Precision (AP) 및 Equal Error Rate (EER).
- **환경**: 잡음(MUSAN) 및 잔향(OpenSLR RIR)이 추가된 실제 환경 모사.

### 주요 결과

1. **Ablation Study (FW-adapter 및 TE classifier)**:
   - 15-shot 환경에서 단순한 전체 미세 조정(FT 15-shot)은 사전 학습 모델(PT)보다 성능이 낮게 나타났다(67.04% vs 72.02%). 이는 데이터 부족으로 인한 오버피팅 때문이다.
   - 반면, BN과 SE 모듈만을 조정하는 FW-adapter를 적용했을 때 AP가 84.33%까지 상승하여, 파라미터 효율적인 튜닝의 중요성을 입증하였다.

2. **TCFM의 효과 및 위치**:
   - TCFM을 인코더의 하위 레이어(G0~G3)에 적용하면 오히려 성능이 하락하였으나, 상위 레이어(G4, G5)에 적용했을 때 성능이 크게 향상되었다.
   - 최종적으로 FW-adapter와 TCFM(G4, G5 적용)을 결합했을 때, 사전 학습 모델의 AP를 72.02%에서 **87.22%**로 끌어올렸다.

3. **베이스라인 비교**:
   - 5-shot 시나리오에서 TA-adapter는 AP 87.63% / EER 5.09%를 기록하며, AdaMS, RPL, AdaKWS 등 기존 방식들을 압도하였다. 특히 데이터가 적을수록(5-shot) 타 모델과의 성능 격차가 더 커지는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 TF-KWS에서 적은 양의 데이터로 성능을 극대화하기 위해 '무엇을 학습시키고 무엇을 고정할 것인가'에 대한 명확한 전략을 제시하였다.

- **상호 보완적 구조**: TCFM은 텍스트 정보를 통해 특징 추출의 '방향'을 조건화하고, FW-adapter는 특징의 '가중치'를 미세 조정함으로써 두 모듈이 시너지 효과를 낸다.
- **레이어별 역할 차이**: 하위 레이어는 일반적인 음성 특징을 추출하므로 고정하는 것이 유리하고, 상위 레이어는 키워드 특화 정보를 처리하므로 TCFM을 통한 조건화가 효과적이라는 점을 실험적으로 확인하였다.
- **파라미터 효율성**: 전체 파라미터의 단 0.14%만 추가/수정함으로써 성능을 크게 향상시켰으며, 이는 모듈형 설계 덕분에 언제든지 원래의 사전 학습 모델로 복구가 가능하다는 실용적인 이점을 제공한다.

다만, 본 연구는 텍스트 기반 임베딩에 의존하고 있으며, 텍스트와 음성 간의 정렬(Alignment)이 완벽하지 않은 경우의 한계에 대해서는 구체적으로 다루지 않았다.

## 📌 TL;DR

본 논문은 적은 수의 음성 샘플로 특정 키워드 인식 성능을 높이는 **TA-adapter**를 제안한다. 이 방법은 텍스트 임베딩(TE)을 활용해 활성화 함수를 동적으로 조절하는 **TCFM**과, BN/SE 레이어만 튜닝하는 **FW-adapter**를 결합하여 파라미터 증가를 최소화(0.14%)하면서도 Few-shot KWS 성능을 획기적으로 개선하였다. 향후 TTS를 이용한 합성 데이터를 활용해 Zero-shot 전이 학습으로 확장할 가능성을 제시하였다.
