# Efficient Conformer with Prob-Sparse Attention Mechanism for End-to-End Speech Recognition

Xiong Wang, Sining Sun, Lei Xie, Long Ma (2021)

## 🧩 Problem to Solve

본 논문은 종단간(End-to-End, E2E) 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템에서 핵심적인 역할을 하는 Conformer 모델의 연산 효율성 문제를 해결하고자 한다. Transformer와 Conformer의 핵심인 Self-attention 메커니즘은 전역적인 정보를 캡처하는 능력이 뛰어나지만, 입력 시퀀스의 길이 $T$에 대해 시간 및 메모리 복잡도가 $O(T^2)$로 증가하는 치명적인 단점이 있다. 특히 음성 신호는 텍스트와 달리 프레임 단위의 길이가 매우 길기 때문에, 긴 발화(Utterance)를 디코딩할 때 연산 비용이 급격히 상승한다.

또한, 저자들은 음성 신호가 매우 구조화되어 있어 특정 입력에 대해 불필요한 프레임이 존재하는 '정보 중복성(Information Redundancy)' 문제가 있음을 지적한다. 즉, 모든 쿼리(Query)에 대해 모든 키(Key)와의 어텐션 점수를 계산하는 기존의 방식은 많은 중복 계산을 포함하고 있어 비효율적이다. 따라서 본 연구의 목표는 인식 정확도를 유지하면서도 추론 속도를 높이고 메모리 사용량을 줄이기 위해 Conformer에 Prob-Sparse Attention 메커니즘을 도입하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모든 쿼리를 계산하는 대신, 정보량이 많아 '중요한' 일부 쿼리만을 선택적으로 계산하는 Prob-Sparse Self-attention을 Conformer에 적용하는 것이다.

1. **Prob-Sparse Attention 도입**: 쿼리의 어텐션 점수 분포가 균등 분포(Uniform Distribution)에서 얼마나 떨어져 있는지를 측정하여, 분포가 불균등한(즉, 특정 키에 강하게 집중된) 유효한 쿼리만을 선택해 연산한다.
2. **K-L Divergence 기반 희소성 측정**: Kullback-Leibler(K-L) divergence를 사용하여 각 쿼리의 유효성을 정량화하고, 이를 통해 계산량을 $O(uT)$ 수준으로 낮춘다(여기서 $u$는 선택된 유효 쿼리의 수이다).
3. **Sparsity Measurement Sharing**: 각 레이어마다 희소성 측정을 반복하는 대신, 여러 레이어에 걸쳐 측정값을 공유함으로써 추가적인 연산 오버헤드를 더욱 줄이는 전략을 제안한다.

## 📎 Related Works

최근 E2E ASR에서는 Transducer와 Attention-based Encoder-Decoder(AED) 프레임워크가 주로 사용되며, 그 중심에는 Transformer 아키텍처가 있다. Transformer는 LSTM과 같은 RNN 기반 모델보다 장거리 전역 문맥 모델링 능력이 뛰어나고 계산 효율적이다. 하지만 Transformer의 Self-attention만으로는 국소적 정보(Local information) 캡처 능력이 부족하다는 한계가 있다. 이를 해결하기 위해 Convolution 모듈을 결합한 Conformer가 제안되었으며, 이는 현재 ASR 분야에서 최첨단(State-of-the-art) 성능을 보이고 있다.

기존의 연산 효율화 연구로는 긴 문서를 처리하기 위해 Local windowed attention과 Global attention을 결합한 Longformer, 그리고 긴 히스토리 문맥을 메모리 뱅크로 변환한 Emformer 등이 있다. 그러나 이러한 접근 방식들은 연산 복잡도는 줄였을지 모르나, 데이터 자체에 내재된 '계산 중복성' 문제는 고려하지 않았다는 점에서 본 논문의 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. Conformer Transducer 구조

본 논문은 Conformer Transducer 모델을 기반으로 한다. 이 모델은 크게 세 가지 구성 요소로 이루어져 있다.

- **Encoder**: 음성 특징 $\mathbf{x}$를 고차원 표현 $\mathbf{h}^{enc}$로 변환한다. $\mathbf{h}^{enc} = \text{Encoder}(\mathbf{x})$.
- **Prediction Network**: 이전 레이블 $\mathbf{y}_{u-1}$을 입력받아 $\mathbf{h}^{pred}_u$를 생성하며, Embedding 레이어와 $N$개의 LSTM 레이어로 구성된다.
- **Joint Network**: Encoder의 출력과 Prediction Network의 출력을 결합하여 최종 레이블의 확률 $P(k|t, u)$를 예측하는 Softmax 레이어로 구성된다.

Conformer Encoder의 블록은 Feed-forward module $\rightarrow$ Multi-head Self-attention (MHSA) $\rightarrow$ Convolution module $\rightarrow$ Feed-forward module 순으로 구성되며, 각 단계는 LayerNorm과 잔차 연결(Residual connection)을 포함한다.

### 2. Prob-Sparse Self-Attention 메커니즘

기존의 Scaled Dot-product Attention은 다음과 같이 정의된다.
$$\text{A}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d}}\right)\mathbf{V}$$
여기서 $i$번째 쿼리 $\mathbf{q}_i$에 대한 어텐션 점수 분포를 $P$라고 할 때, 이 분포가 균등 분포 $U$와 유사하면 해당 쿼리는 단순히 모든 값의 평균을 내는 것과 같아져 정보 가치가 낮다. 이를 측정하기 위해 K-L divergence를 사용한 희소성 측정치 $M_{sparse}$를 다음과 같이 정의한다.
$$M_{sparse}(\mathbf{q}_i, \mathbf{K}) = \ln \sum_{j=1}^L e^{\mathbf{q}_i \mathbf{k}_j^T / \sqrt{d}} - \frac{1}{L} \sum_{j=1}^L \mathbf{q}_i \mathbf{k}_j^T / \sqrt{d} - \ln L$$
(상수를 제외한 식 (10) 참조)

### 3. 효율적인 구현 및 추론 절차

모든 쿼리에 대해 $M_{sparse}$를 계산하는 것은 여전히 $O(L^2)$의 비용이 들므로, 본 논문은 샘플링 기법을 통해 이를 근사한다. 무작위로 샘플링된 키 행렬 $\tilde{\mathbf{K}}$를 사용하여 $\bar{M}_{sparse}$를 계산하고, 상위 $L_{sparse}$개의 쿼리만 선택한다.

- **쿼리 선택**: $\bar{M}_{sparse}$ 값이 큰 상위 $L_{sparse} = r_{sparse} L$개의 쿼리 인덱스 집합 $I_{sparse}$를 추출한다.
- **최종 연산**:
$$\text{SA}(\mathbf{q}_i, \mathbf{K}, \mathbf{V}) = \begin{cases} \sum_{j=1}^L p(k_j|\mathbf{q}_i)\mathbf{v}_j, & \text{if } i \in I_{sparse} \\ \mathbf{v}_i, & \text{else} \end{cases}$$
즉, 선택되지 않은 쿼리는 복잡한 어텐션 계산 없이 자신의 Value 값을 그대로 출력한다.

### 4. Sparsity Measurement Sharing

추가적인 연산 비용을 줄이기 위해, $M_{sparse}$ 값을 $N_{share}$ 레이어마다 한 번씩 계산하고 이를 다음 $N_{share}-1$ 레이어에서 공유하는 전략을 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: AISHELL-1 (중국어), LibriSpeech (영어)
- **특징**: 80-dim log mel-filterbanks + 3-dim pitch features
- **모델 설정**: 16개의 Conformer blocks, Attention dimension 256, 4 heads, Convolution kernel size 3.

### 2. 주요 결과 및 분석

- **인식 성능 (CER/WER)**:
  - AISHELL-1 데이터셋에서 Prob-sparse attention을 처음부터 학습(scratch)시킨 경우(E0)는 baseline(B0)보다 성능이 떨어졌다.
  - 하지만 baseline 모델(B0)에서 초기화한 후 Prob-sparse attention으로 재학습(E1, $r_{sparse}=0.5$)시킨 결과, CER이 $6.7\%$에서 $6.5\%$로 오히려 개선되었다. 이는 음성 데이터의 중복성을 제거하는 것이 모델 성능에 긍정적인 영향을 줄 수 있음을 시사한다.
  - 무작위 쿼리 선택(E2) 시 CER이 $8.2\%$로 급증하여, 유효한 쿼리를 선택하는 Prob-sparse 방식의 정당성이 입증되었다.
- **Sparse Rate ($r_{sparse}$)의 영향**:
  - $r_{sparse} < 0.35$일 때는 정보 손실로 인해 성능이 저하되지만, $0.35 \sim 0.5$ 구간에서는 baseline보다 우수한 성능을 보였다.
- **Sparsity Sharing 효과**:
  - $N_{share}=4$ 또는 $8$로 설정하여 측정값을 공유해도 CER의 증가가 거의 없거나 무시할 만한 수준이었으며, 이는 계산량을 획기적으로 줄일 수 있음을 보여준다.
- **효율성 (Efficiency)**:
  - 추론 과정에서 문장 길이에 따라 **추론 속도는 $8\% \sim 45\%$ 향상**되었으며, **메모리 사용량은 $15\% \sim 45\%$ 감소**하였다. 특히 문장의 길이가 길어질수록 효율성 이득이 더 커지는 경향을 보였다.

## 🧠 Insights & Discussion

본 연구는 Conformer의 Self-attention이 가진 연산 복잡도 문제를 '정보의 희소성'이라는 관점에서 접근하여 효과적으로 해결하였다. 특히 단순히 계산량을 줄이는 것에 그치지 않고, K-L divergence라는 통계적 근거를 통해 유효한 정보를 보존하면서 불필요한 계산을 제거했다는 점이 강점이다.

다만, 실험 결과에서 나타나듯 Prob-sparse attention을 처음부터 학습시키는 것보다 이미 학습된 모델에서 초기화하여 사용하는 것이 훨씬 성능이 좋았다. 이는 유효한 쿼리를 판별하기 위한 $M_{sparse}$ 측정치가 어느 정도 학습된 모델의 가중치를 기반으로 해야 정확하게 작동한다는 것을 의미하며, 이는 실질적인 적용 시 초기 학습 전략이 매우 중요하다는 제약 사항을 시사한다.

결론적으로, 음성 신호의 고유한 특성인 정보 중복성을 활용하여 모델의 경량화를 달성했으며, 이는 실시간 음성 인식 시스템이나 온디바이스(On-device) 환경에서 매우 유용한 최적화 기법이 될 수 있을 것으로 판단된다.

## 📌 TL;DR

본 논문은 Conformer 기반 E2E 음성 인식 모델의 $O(T^2)$ 연산 복잡도를 해결하기 위해, K-L divergence 기반의 **Prob-Sparse Attention** 메커니즘을 제안하였다. 중요도가 낮은 중복 쿼리를 계산에서 제외하고 상위 유효 쿼리만을 처리함으로써, **인식 정확도를 유지(또는 소폭 향상)하면서도 추론 속도를 최대 $45\%$ 높이고 메모리 사용량을 최대 $45\%$ 절감**하였다. 이 연구는 특히 긴 음성 시퀀스를 처리해야 하는 실제 ASR 시스템의 효율성을 극대화하는 데 중요한 기여를 한다.
