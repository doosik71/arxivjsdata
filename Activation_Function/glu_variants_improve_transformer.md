# GLU Variants Improve Transformer

Noam Shazeer

## 🧩 Problem to Solve

Transformer 모델의 Feed-Forward Network (FFN) 서브레이어는 일반적으로 ReLU 또는 GELU/Swish와 같은 활성화 함수를 사용합니다. 이 논문은 이러한 표준 활성화 함수를 Gated Linear Unit (GLU)의 변형으로 대체함으로써 Transformer의 성능을 향상시킬 수 있는지 탐구합니다.

## ✨ Key Contributions

- **새로운 FFN 아키텍처 제안**: Transformer의 FFN 서브레이어에 GLU 및 그 변형(ReGLU, GEGLU, SwiGLU 등)을 활용한 아키텍처를 제안했습니다.
- **성능 향상 입증**: 제안된 GLU 변형 FFN이 기존 ReLU, GELU, Swish 기반 FFN보다 사전 학습 퍼플렉시티(perplexity) 및 다양한 다운스트림 언어 이해 태스크(GLUE, SuperGLUE, SQuAD)에서 일관되게 더 나은 성능을 보임을 입증했습니다.
- **효율성 유지**: 기존 FFN과 동일한 수준의 매개변수 및 계산량을 유지하면서 성능 향상을 달성했습니다. 이는 GLU 변형 FFN이 세 개의 가중치 행렬을 사용하므로, 은닉 유닛 수를 $2/3$으로 줄여 매개변수 수를 맞추는 방식으로 이루어졌습니다.
- **간단한 구현**: 제안된 아키텍처는 구현이 간단하며, 명백한 계산상의 단점이 없습니다.

## 📎 Related Works

- **Transformer (Vaswani et al., 2017)**: Multi-head attention과 Position-wise Feed-Forward Networks (FFN)를 기반으로 하는 시퀀스-투-시퀀스 모델. 이 논문의 기반 아키텍처입니다.
- **Gated Linear Units (GLU) (Dauphin et al., 2016)**: 두 개의 선형 변환 중 하나에 시그모이드 함수를 적용한 후 성분별 곱을 수행하는 신경망 레이어. 이 논문은 GLU의 다양한 변형을 탐구합니다.
- **ReLU (Glorot et al., 2011)**: Transformer FFN에 널리 사용되는 활성화 함수입니다.
- **GELU (Hendrycks and Gimpel, 2016) & Swish (Ramachandran et al., 2017)**: ReLU를 대체하는 새로운 활성화 함수로, Transformer FFN에도 적용될 수 있습니다.
- **T5 (Raffel et al., 2019)**: 전이 학습(transfer-learning) 설정을 위해 이 논문에서 사용된 코드 베이스, 모델 아키텍처 및 훈련 태스크의 원천입니다.

## 🛠️ Methodology

1. **GLU 및 변형 정의**:
   - **GLU**: $\sigma(xW+b) \otimes (xV+c)$
   - **ReGLU**: $\text{max}(0, xW+b) \otimes (xV+c)$
   - **GEGLU**: $\text{GELU}(xW+b) \otimes (xV+c)$
   - **SwiGLU**: $\text{Swish}_{\beta}(xW+b) \otimes (xV+c)$
   - 실험에서는 편향(bias) 항을 생략하고, 각 활성화 함수에 따라 새로운 FFN 레이어를 구성합니다. 예를 들어, $\text{FFN}_{\text{GEGLU}}(x, W, V, W_2) = (\text{GELU}(xW) \otimes xV)W_2$ 입니다.
2. **매개변수 및 계산량 정규화**: GLU 변형 FFN은 세 개의 가중치 행렬($W, V, W_2$)을 사용하는 반면, 기존 FFN은 두 개의 행렬($W_1, W_2$)을 사용합니다. 공정한 비교를 위해 GLU 변형 FFN의 은닉 유닛 수($d_{\text{ff}}$)를 기존 FFN의 $2/3$으로 줄여 매개변수 및 계산량을 동일하게 유지합니다 (예: $3072 \rightarrow 2048$).
3. **실험 설정**:
   - **모델**: T5 모델 아키텍처(인코더-디코더 Transformer, $d_{\text{model}}=768$, 12개 레이어)를 사용합니다.
   - **사전 학습**: C4 데이터셋에서 스팬 필링(span-filling) 목적 함수로 524,288 스텝 동안 사전 학습을 수행합니다. Adafactor 최적화기 및 역제곱근 학습률 스케줄을 사용합니다. [Raffel et al., 2019]와 달리 사전 학습 중 드롭아웃을 사용하지 않습니다. 모델 품질 지표로 홀드아웃 세트의 로그-퍼플렉시티를 측정합니다.
   - **미세 조정**: 사전 학습된 모델을 SQuAD, GLUE, SuperGLUE 벤치마크의 언어 이해 태스크 혼합 데이터셋으로 131,072 스텝 동안 미세 조정합니다. 미세 조정 시에는 드롭아웃 0.1을 사용합니다.

## 📊 Results

- **사전 학습 퍼플렉시티**:
  - GEGLU와 SwiGLU 변형이 가장 낮은 퍼플렉시티를 기록하며, 기존 ReLU (1.677) 대비 GEGLU (1.633) 및 SwiGLU (1.636)가 더 우수한 성능을 보였습니다.
- **다운스트림 태스크**:
  - **GLUE, SuperGLUE, SQuAD**: GLU 변형 FFN 모델들은 대부분의 태스크에서 기존 ReLU, GELU, Swish 기반 FFN 모델보다 더 높은 점수를 달성했습니다. 특히, FFN$_{\text{ReGLU}}$, FFN$_{\text{SwiGLU}}$, FFN$_{\text{GEGLU}}$ 등이 여러 태스크에서 최상위 성능을 기록했습니다.
  - [Raffel et al., 2019]의 결과보다 모든 모델에서 전반적으로 더 높은 점수를 달성했는데, 이는 사전 학습 중 드롭아웃을 사용하지 않은 효과로 분석되었습니다.

## 🧠 Insights & Discussion

- **성능 개선의 유용성**: GLU 계열 활성화 함수를 Transformer의 FFN에 도입하는 것이 사전 학습 및 다양한 다운스트림 태스크에서 모델 성능을 향상시키는 효과적인 방법임을 보여주었습니다.
- **효율적인 대안**: 매개변수 및 계산량의 증가 없이 성능을 개선했다는 점에서, GLU 변형 FFN은 기존 Transformer FFN의 효율적이고 강력한 대안이 될 수 있습니다.
- **이론적 배경 부족**: 저자는 GLU 변형이 왜 더 잘 작동하는지에 대한 이론적 설명을 제공하지 않았습니다. 이는 게이팅 메커니즘이 신경망 학습에 미치는 영향에 대한 추가적인 연구 필요성을 시사합니다.
- **사전 학습 드롭아웃의 영향**: 사전 학습 중 드롭아웃을 생략한 것이 전반적인 모델 성능 향상에 기여했다는 결과는, 대규모 언어 모델의 사전 학습 전략에 있어 정규화 기법의 적용 방식에 대한 추가적인 탐구가 필요함을 암시합니다.

## 📌 TL;DR

이 논문은 Transformer의 Feed-Forward Network (FFN)에서 ReLU/GELU 대신 Gated Linear Unit (GLU) 및 그 변형(ReGLU, GEGLU, SwiGLU)을 사용하도록 제안합니다. 매개변수 및 계산량을 기존과 동일하게 유지하면서, GEGLU와 SwiGLU를 포함한 GLU 변형 FFN은 사전 학습 퍼플렉시티와 GLUE, SuperGLUE, SQuAD 등의 다양한 다운스트림 태스크에서 기존 FFN보다 더 나은 성능을 달성했습니다. 이는 간단한 변경으로 Transformer의 성능을 효과적으로 개선할 수 있음을 보여줍니다.
