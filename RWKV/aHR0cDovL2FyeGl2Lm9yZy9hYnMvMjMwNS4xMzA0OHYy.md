# RWKV: Reinventing RNNs for the Transformer Era
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Xingjian Du, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemysław Kazienko, Jan Kocoń, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanisław Woźniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, Rui-Jie Zhu

## 🧩 Problem to Solve
Transformer는 자연어 처리(NLP) 분야에 혁신을 가져왔지만, 시퀀스 길이에 따라 메모리 및 계산 복잡도가 $O(T^2)$으로 증가하는 문제점을 가지고 있습니다. 반면, 순환 신경망(RNN)은 메모리 및 계산 요구사항이 선형적($O(T)$)이지만, 병렬화 및 확장성 한계로 인해 Transformer와 동등한 성능을 달성하는 데 어려움을 겪습니다. 이 논문은 Transformer의 효율적인 병렬 훈련과 RNN의 효율적인 추론을 결합하여, 계산 효율성과 모델 성능 사이의 균형을 맞추는 것을 목표로 합니다.

## ✨ Key Contributions
*   **RWKV (Receptance Weighted Key Value) 아키텍처 도입**: RNN과 Transformer의 장점을 결합하고 각 모델의 주요 단점을 완화하는 새로운 모델 아키텍처를 제안했습니다.
*   **선형 어텐션 메커니즘**: 전통적인 점곱(dot-product) 어텐션을 효과적인 채널 지향 어텐션으로 대체하는 변형된 선형 어텐션 메커니즘을 활용하여 최저 계산 및 메모리 복잡도를 달성했습니다. 이를 통해 모델을 Transformer 또는 RNN 형태로 공식화할 수 있습니다.
*   **대규모 모델 훈련 및 확장성 입증**: 최대 140억 개 파라미터를 가진 모델을 훈련하여, RWKV가 유사한 크기의 Transformer와 동등한 성능을 보임을 입증했습니다. 이는 훈련된 가장 큰 RNN입니다.
*   **스케일링 법칙 준수**: RWKV가 Transformer와 동일한 스케일링 법칙을 따른다는 것을 실험적으로 보여주었습니다.
*   **사전 학습된 모델 공개**: 1억 6,900만 개부터 140억 개 파라미터까지 다양한 크기의 사전 학습된 모델을 공개했습니다.

## 📎 Related Works
*   **순환 신경망 (RNNs)**: LSTM (Hochreiter and Schmidhuber, 1997) 및 GRU (Chung et al., 2014)와 같은 아키텍처는 효율적인 메모리 사용이 가능하지만, 기울기 소실 문제와 훈련 시 시간 차원 병렬화의 어려움을 겪습니다.
*   **트랜스포머 (Transformers)**: Vaswani et al. (2017)에 의해 소개되었으며, 병렬 훈련과 장거리 의존성 포착에 능하지만, 자가 어텐션 메커니즘의 $O(T^2)$ 복잡도로 인해 긴 시퀀스에 대한 계산 및 메모리 부담이 큽니다.
*   **효율적인 Transformer 변형**: Longformer (Beltagy et al., 2020), Reformer (Kitaev et al., 2020), Performer (Choromanski et al., 2020), Linear Transformers (Katharopoulos et al., 2020), FlashAttention (Dao et al., 2022a) 등이 Transformer의 확장성 문제를 개선하려는 시도들입니다.
*   **Attention Free Transformer (AFT)**: Zhai et al. (2021)은 점곱 자가 어텐션을 대체하는 효율적인 계산 방식을 제안했으며, RWKV는 이 아이디어에서 영감을 받았습니다.
*   **State Space Models (SSM)**: Gu et al. (2021) 및 그 변형들 (Dao et al., 2022b; Poli et al., 2023)은 긴 시퀀스를 효율적으로 모델링하는 데 상당한 진전을 보여주었습니다.

## 🛠️ Methodology
1.  **RWKV 아키텍처 설계**:
    *   **핵심 요소**: Receptance ($R$), Weight ($W$, 학습 가능한 위치 가중치 감쇠 벡터), Key ($K$), Value ($V$). 이 네 가지 요소는 각 타임스텝에서 곱셈 방식으로 상호작용합니다.
    *   **잔차 블록 스택**: 모델은 시계열 혼합 (time-mixing) 및 채널 혼합 (channel-mixing) 서브-블록으로 구성된 잔차 블록을 쌓아 만들어집니다.
    *   **토큰 시프트**: 각 블록에서 현재 입력 $x_t$와 이전 입력 $x_{t-1}$의 선형 보간을 통해 $R, K, V$ (시계열 혼합) 및 $R', K'$ (채널 혼합) 벡터를 생성합니다. 예를 들어, $r_t = W_r \cdot (\mu_r \odot x_t + (1-\mu_r) \odot x_{t-1})$ 와 같이 계산됩니다.
2.  **WKV 연산자**:
    *   AFT에서 영감을 받아 $W$를 채널별 벡터로 처리하고 상대 위치에 의해 수정합니다.
    *   WKV 연산은 다음과 같은 재귀적 공식으로 정의됩니다:
        $$ \text{wkv}_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \odot v_i + e^{u+k_t} \odot v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}} $$
        여기서 $w \in (\mathbb{R}_{\geq 0})^d$는 학습 가능한 시간 감쇠 벡터이며, $u$는 현재 토큰에 대한 가중치 벡터입니다.
3.  **출력 게이팅 (Output Gating)**: 시계열 혼합 및 채널 혼합 블록 모두에서 수용도 벡터의 시그모이드 ($\sigma(r)$)를 사용하여 출력을 게이팅합니다.
4.  **Transformer와 유사한 훈련**:
    *   **시간 병렬 모드**: Transformer와 유사하게, RWKV는 매트릭스 곱셈 $W_\lambda$ ($\lambda \in \{r, k, v, o\}$)을 포함하여 훈련 시 효율적으로 병렬화될 수 있습니다.
    *   시퀀스 배치를 처리하는 시간 복잡도는 $O(BTd^2)$이며, WKV 업데이트는 $O(BTd)$입니다.
5.  **RNN과 유사한 추론**:
    *   **시간 순차 모드**: RWKV는 추론 시 자동회귀 디코딩을 위해 재귀적으로 공식화될 수 있어, 상수 계산 및 메모리 복잡도 $O(D)$를 가집니다 (Table 1 참조).
6.  **추가 최적화**:
    *   **커스텀 CUDA 커널**: WKV 계산의 순차적 특성으로 인한 비효율성을 개선하기 위해 개발되었습니다.
    *   **작은 초기 임베딩 (Small Init Embedding)**: 임베딩 행렬을 작은 값으로 초기화하고 LayerNorm을 적용하여 훈련 과정을 가속화하고 안정화합니다.
    *   **커스텀 초기화 (Custom Initialization)**: 대부분의 가중치를 0으로 초기화하고 선형 레이어에는 바이어스를 사용하지 않음으로써 명확한 정보 흐름을 확립합니다.

## 📊 Results
*   **모델 스케일링**: 1억 6,900만 개부터 140억 개 파라미터까지 총 6개의 RWKV 모델을 훈련했으며, 모두 Pile 데이터셋(3,300억 토큰)에서 1 epoch 동안 학습되었습니다.
*   **성능 비교**: ARC, BoolQ, COPA, HellaSwag 등 12개 NLP 벤치마크에서 RWKV는 Pythia, OPT, BLOOM과 같은 유사한 크기의 Transformer 모델들과 동등한 평균 성능을 보였습니다 (Figure 1).
*   **스케일링 법칙**: RWKV는 Transformer와 동일한 일반적인 형태의 스케일링 법칙을 따르며, 손실(loss) 대 계산량(compute) 그래프에서 $r^2$ 값이 0.994로 매우 높은 적합도를 보였습니다 (Figure 4).
*   **장기 컨텍스트 처리**: 점진적으로 증가하는 시퀀스 길이로 파인튜닝 (최대 8192 토큰)했을 때, Pile 데이터셋에서 테스트 손실이 지속적으로 감소하여 RWKV가 장기 컨텍스트 정보를 효과적으로 활용할 수 있음을 입증했습니다 (Figure 6).
*   **Long-Range Arena (LRA) 벤치마크**: RWKV는 5개 데이터셋에서 S4 모델 다음으로 두 번째로 좋은 성능을 보였으며, 특히 자연어 및 컴퓨터 코드 처리 관련 문제에서는 S4와 거의 동등한 성능을 나타냈습니다.
*   **추론 효율성**: Transformer와 달리 RWKV는 텍스트 생성 시 선형적인 시간 스케일링을 보이며 (Figure 7), 메모리 요구사항도 효율적입니다 (Figure 13).

## 🧠 Insights & Discussion
*   **선형 어텐션의 함의**: RWKV의 선형 어텐션은 뛰어난 효율성을 제공하지만, 매우 긴 컨텍스트에서 미세한 정보를 정확히 회상해야 하는 특정 작업에서는 표준 Transformer의 $O(T^2)$ 어텐션에 비해 정보 유실의 잠재적 한계를 가질 수 있습니다. 이는 정보가 단일 벡터 표현으로 통합되기 때문입니다.
*   **프롬프트 엔지니어링의 중요성**: RWKV는 RNN의 특성상 과거 정보를 "되돌아볼" 수 없으므로, 표준 Transformer 모델보다 프롬프트 엔지니어링에 훨씬 민감합니다. 프롬프트의 정보 순서를 신중하게 조정함으로써 성능이 크게 향상될 수 있음이 확인되었습니다 (예: RTE 작업에서 F1 Macro가 44.2%에서 74.8%로 증가).
*   **새로운 모델링 패러다임**: RWKV는 시퀀스 데이터에서 복잡한 관계를 모델링하기 위한 확장 가능하고 효율적인 아키텍처의 새로운 길을 열었으며, 수십억 개의 파라미터를 가진 사전 학습된 모델로 이러한 주장을 입증한 최초의 사례입니다.
*   **향후 연구 방향**:
    *   모델의 표현력을 높이기 위해 시간 감쇠 공식을 개선하고 초기 모델 상태를 탐색할 수 있습니다.
    *   병렬 스캔을 적용하여 $wkv_t$ 계산의 비용을 $O(B \log(T) d)$로 더욱 줄일 수 있습니다.
    *   인코더-디코더 아키텍처에 RWKV 메커니즘을 적용하여 교차 어텐션 메커니즘을 대체할 가능성이 있습니다.
    *   RWKV의 내부 상태를 활용하여 모델의 해석 가능성, 시퀀스 데이터 예측 및 안전성을 향상시킬 수 있습니다.
*   **윤리적 고려사항**: 낮은 추론 비용 덕분에 RWKV는 소비자 및 엣지 하드웨어에 배포하기에 적합하여 AI 민주화에 기여할 수 있습니다. 그러나 훈련 데이터에 존재하는 편향이나 유해 콘텐츠를 재생산할 가능성은 여전히 존재합니다.

## 📌 TL;DR
RWKV는 Transformer의 병렬 훈련 효율성과 RNN의 추론 효율성을 결합한 새로운 아키텍처입니다. 선형 어텐션 메커니즘을 통해 $O(T)$의 계산 및 메모리 복잡도를 달성하며, 최대 140억 개의 파라미터로 확장하여 Transformer와 동등한 성능을 보입니다. 장기 컨텍스트 처리 능력이 뛰어나고 효율적인 추론을 제공하지만, 프롬프트 엔지니어링에 대한 민감도가 높습니다.