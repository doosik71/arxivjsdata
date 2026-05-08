# Tree-constrained Pointer Generator with Graph Neural Network Encodings for Contextual Speech Recognition

Guangzhi Sun, Chao Zhang, Philip C. Woodland (2022)

## 🧩 Problem to Solve

종단간(End-to-End) 자동 음성 인식(ASR) 시스템에서 가장 까다로운 문제 중 하나는 빈도가 매우 낮은 '롱테일(long-tail)' 단어들을 정확하게 인식하는 것이다. 이러한 단어들은 전체 단어 오류율(Word Error Rate, WER)에 미치는 영향은 작을 수 있으나, 명사나 고유 명사와 같이 정보 가치가 높은 핵심 단어인 경우가 많아 후속 이해 작업(downstream understanding tasks)에서 매우 중요하다.

본 논문은 사용자 연락처, 재생 목록, 또는 프레젠테이션 슬라이드와 같은 외부 컨텍스트 지식을 활용하여 이러한 희귀 단어들의 인식률을 높이는 Contextual Biasing 문제를 해결하고자 한다. 특히, 대규모 biasing list를 효율적으로 처리하면서도, 현재 디코딩 단계에서 향후 어떤 단어가 나타날지에 대한 정보를 미리 반영하여 biasing 적용 여부를 더 정확하게 결정하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Prefix-tree 구조의 biasing list를 Graph Neural Network(GNN), 구체적으로는 Tree-RNN을 사용하여 인코딩하는 것**이다.

기존의 Tree-constrained Pointer Generator(TCPGen)는 각 노드의 단편적인 정보만을 사용했지만, 본 논문은 Tree-RNN을 통해 특정 노드의 표현형(representation)에 해당 노드로부터 시작되는 모든 하위 가지(branch)들의 정보를 재귀적으로 통합한다. 이를 통해 디코딩 과정에서 **Lookahead(미리 보기)** 기능을 구현함으로써, 현재 단계에서 biasing 단어가 생성될 확률($P^{gen}$)을 더 정밀하게 예측할 수 있게 한다.

## 📎 Related Works

기존의 Contextual Biasing 접근 방식은 크게 세 가지로 나뉜다.

1. **Shallow Fusion (SF):** 가중 유한 상태 트랜스듀서(WFST)를 사용하여 점수 수준에서 보간하는 방식이다.
2. **Deep Context:** 신경망 내부에 컨텍스트를 직접 임베딩하는 방식으로, 고정된 구문 접두어에 의존하지 않지만 biasing list가 커질수록 메모리 사용량이 급증하는 한계가 있다.
3. **Pointer Generator 및 TCPGen:** 신경망 지름길(neural shortcuts)을 통해 출력 분포를 직접 수정하는 방식이다. 특히 TCPGen은 symbolic prefix-tree search를 결합하여 효율성을 높였다.

본 논문은 TCPGen의 구조를 계승하되, 단순한 단어 조각(wordpiece) 임베딩 대신 GNN 기반의 노드 인코딩을 도입함으로써 기존 TCPGen이 가졌던 '미래 정보 부족' 문제를 해결하고 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 TCPGen 구조

TCPGen은 ASR 모델(AED 또는 RNN-T)의 출력 분포와 biasing list 기반의 Pointer Generator 분포를 보간하여 최종 출력을 결정한다.

- **TCPGen 분포 ($P^{ptr}$):** 현재 상태의 쿼리 벡터 $q_i$와 prefix-tree에서 유효한 노드들의 키 벡터 $K$ 사이의 Scaled Dot-Product Attention을 통해 계산된다.
$$P^{ptr}(y_i|y_{1:i-1}, x_{1:T}) = \text{Softmax}(\text{Mask}(q_i K^T / \sqrt{d}))$$
- **최종 출력 분포:** 모델의 기본 분포 $P^{mdl}$와 TCPGen 분포 $P^{ptr}$를 생성 확률 $\hat{P}^{gen}_i$를 가중치로 하여 보간한다.
$$P(y_i) = P^{mdl}(y_i)(1 - \hat{P}^{gen}_i) + P^{ptr}(y_i)P^{gen}_i$$

### 2. GNN(Tree-RNN) 인코딩

본 논문의 핵심인 Tree-RNN은 prefix-tree의 각 노드 $n_j$에 대해 다음과 같이 표현형 $h^{tree}_{n_j}$를 계산한다.
$$h^{tree}_{n_j} = f(W_1 y_j + \sum_{k=1:K} W_2 h^{tree}_{n_k})$$
여기서 $y_j$는 해당 노드의 wordpiece 임베딩이며, $\sum W_2 h^{tree}_{n_k}$는 모든 자식 노드들의 정보를 합산한 것이다. $f(\cdot)$는 ReLU 활성화 함수를 사용한다. 이 과정은 잎 노드(leaf node)부터 루트 노드까지 재귀적으로 수행된다.

### 3. Lookahead 기능의 구현

이렇게 생성된 노드 표현형은 TCPGen의 Key($k_j$)와 Value($v_j$) 벡터로 사용된다.
$$k_j = W^K h^{tree}_{n_j}, \quad v_j = W^V h^{tree}_{n_j}$$
결과적으로, 특정 노드(예: 'Tur')에 도달했을 때, 이미 그 노드의 표현형 안에 하위 노드들(예: 'ner')의 정보가 포함되어 있으므로, 모델은 'Turner'라는 전체 단어가 나타날 것임을 미리 인지하고 $P^{gen}$을 높게 설정할 수 있다.

### 4. 학습 및 추론 절차

- **학습:** $W_1, W_2, W^K, W^V$ 등의 파라미터는 ASR 시스템과 함께 역전파(back-propagation)를 통해 공동 최적화된다.
- **추론:** biasing list가 고정되어 있다면 GNN 인코딩은 디코딩 시작 전 오프라인으로 미리 계산할 수 있으므로, 추론 시의 시간 및 공간 복잡도 증가량은 무시할 수 있는 수준이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Librispeech (시뮬레이션 기반 biasing) 및 AMI (슬라이드 OCR 기반 실제 컨텍스트 biasing).
- **모델:** Conformer 기반의 AED 및 RNN-T.
- **평가 지표:** WER 및 희귀 단어 오류율(R-WER). AMI 데이터의 경우 슬라이드 희귀 단어 오류율($R^s$-WER)을 측정하였다.

### 2. 정량적 결과

- **Librispeech:** AED 모델에서 GNN 인코딩 적용 시, 기존 TCPGen 대비 R-WER의 상대적 개선 폭이 test-clean 세트 기준 37%에서 50%로 크게 향상되었다. RNN-T에서도 유사한 경향이 확인되었다.
- **OOV 단어 인식:** 학습 데이터에 없었던 OOV 단어에 대해 AED 모델의 WER이 $73\% \to 40\%(\text{TCPGen}) \to 33\%(\text{GNN})$로 감소하여 zero-shot learning 능력이 입증되었다.
- **AMI (Visual-grounded):** 슬라이드에서 추출한 biasing list를 사용했을 때, GNN 인코딩을 적용한 TCPGen이 가장 낮은 $R^s$-WER을 기록하였다. 특히 Shallow Fusion(SF)과 결합했을 때 성능이 더욱 향상되었다.

### 3. 정성적 결과 및 분석

- **Heatmap 분석:** $P^{gen}_i$의 히트맵 분석 결과, GNN 인코딩을 사용했을 때 biasing 단어의 앞부분 wordpiece 단계에서부터 생성 확률이 이미 높게 나타났다. 이는 Lookahead 기능이 실제로 작동하여 biasing 적용 시점을 더 정확하게 예측하고 있음을 시사한다.

## 🧠 Insights & Discussion

**강점:**
본 논문은 단순한 임베딩의 나열이 아니라, 데이터의 구조(Tree)를 신경망 아키텍처(GNN)에 직접 반영함으로써 ASR의 고질적인 문제인 희귀 단어 인식 성능을 효과적으로 높였다. 특히, 추론 시 오프라인 계산이 가능하다는 점에서 실용성이 매우 높다.

**한계 및 가정:**

- 실험에서 사용된 biasing list의 크기가 수천 단어 수준으로 제한적이며, 수만 단어 이상의 극단적인 대규모 리스트에서도 동일한 효율성이 유지되는지에 대한 분석은 부족하다.
- AMI 실험에서 OCR 도구(Tesseract 4)의 성능에 의존하여 biasing list를 생성하므로, OCR 자체의 오류가 ASR 결과에 미치는 영향이 존재할 수 있다.

**비판적 해석:**
본 연구는 Tree-RNN이라는 비교적 고전적인 GNN 구조를 사용하였다. 저자들도 결론에서 언급했듯이, 향후 Graph Convolutional Networks(GCN)나 Graph Attention Networks(GAT)와 같은 더 발전된 GNN 구조를 도입한다면, 노드 간의 관계를 더 유연하게 학습하여 성능을 추가로 끌어올릴 수 있을 것으로 판단된다.

## 📌 TL;DR

본 논문은 ASR의 희귀 단어 인식 성능을 높이기 위해 **Prefix-tree 구조의 biasing list를 Tree-RNN으로 인코딩하는 TCPGen 확장 모델**을 제안한다. 이를 통해 디코딩 중 미래의 단어 정보를 미리 파악하는 **Lookahead 기능**을 구현하였으며, 그 결과 Librispeech와 AMI 데이터셋 모두에서 희귀 단어 오류율(R-WER)을 유의미하게 낮추었다. 이 기술은 실시간 추론 비용 증가가 거의 없으면서도 컨텍스트 기반 음성 인식의 정확도를 높일 수 있어, 실제 서비스 적용 가능성이 매우 높다.
