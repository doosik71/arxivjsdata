# A Study on ReLU and Softmax in Transformer

Kai Shen, Junliang Guo, Xu Tan, Siliang Tang, Rui Wang, Jiang Bian

## 🧩 Problem to Solve

Transformer 아키텍처는 Self-Attention Network(SAN)와 Feed-Forward Network(FFN)로 구성되며, 이들은 이전 연구에서 키-값 메모리로 간주될 수 있다고 제안되었습니다. 하지만 FFN은 ReLU를, SAN 및 전통적인 키-값 메모리는 Softmax를 활성화 함수로 사용한다는 중요한 차이가 존재합니다. 이러한 활성화 함수의 차이로 인해 FFN과 키-값 메모리가 완전히 동등하지 않으며, 이 차이가 모델 성능, 특히 긴 시퀀스 처리 능력에 어떤 영향을 미치는지 명확히 밝혀지지 않았습니다. 본 논문은 ReLU와 Softmax의 특성 차이를 심층적으로 분석하고, 트랜스포머 아키텍처에서 두 활성화 함수의 역할과 관계를 재정립하는 것을 목표로 합니다.

## ✨ Key Contributions

- **FFN과 키-값 메모리 등가성 재확립:** Softmax 기반 FFN에 Layer Normalization을 추가하면 ReLU 기반 FFN과 동등한 성능을 보임을 발견하여, FFN과 키-값 메모리의 등가성을 재정립했습니다.
- **ReLU의 대규모 슬롯/긴 시퀀스 처리 능력 입증:** ReLU는 Softmax에 비해 대규모 키-값 슬롯(FFN) 및 긴 시퀀스(SAN) 처리에 더 뛰어난 능력을 가짐을 정량적으로 분석하고 입증했습니다. Softmax는 요소에 대한 지수 정규화로 인해 분포가 과도하게 중앙 집중화되어 많은 문맥 정보를 활용하지 못하는 반면, ReLU는 이러한 제한이 없습니다.
- **ReLU 기반 SAN의 안정화:** SAN에서 Softmax를 ReLU로 직접 대체할 경우 발생하는 분산 폭발 문제를 해결하기 위해 분산 감소 계수와 정규화 손실 함수를 제안했습니다.
- **완전 ReLU Transformer (ReLUFormer) 제안:** FFN과 SAN 구성 요소 모두에 ReLU 활성화 함수를 적용한 ReLUFormer 아키텍처를 제안했습니다. ReLUFormer는 FFN을 전역 키-값 메모리로, SAN을 지역 메모리로 통합하는 키-값 메모리 네트워크로 간주될 수 있음을 주장했습니다.
- **장문 시퀀스 처리 성능 향상:** ReLUFormer가 단문 번역에서 기존 Transformer보다 우수하고, 특히 문서 번역과 같은 장문 시퀀스 처리 작업에서 Softmax 기반 모델들을 크게 능가함을 실험을 통해 입증했습니다.

## 📎 Related Works

- **Transformer (Vaswani et al., 2017):** 현대 NLP의 기반이 되는 주의(attention) 메커니즘을 사용하는 모델.
- **FFN/SAN과 키-값 메모리의 관계 (Geva et al., 2020; Sukhbaatar et al., 2019; Dai et al., 2021; Yao et al., 2022):** Transformer의 FFN과 SAN이 키-값 메모리 네트워크와 유사한 형태로 지식을 저장하고 검색한다는 개념.
- **외부 메모리 통합 (Lample et al., 2019; Sukhbaatar et al., 2015; Wu et al., 2022; Dai et al., 2019):** Transformer에 외부 또는 영속적인 메모리 메커니즘을 통합하여 모델의 지식 저장 및 장기 의존성 처리 능력을 향상시키는 연구.
- **Layer Normalization (Ba et al., 2016):** 신경망 훈련의 안정성과 성능 향상을 위한 정규화 기법.
- **희소 어텐션(Sparse Attention) 활성화 함수 (Martins & Astudillo, 2016 - Sparsemax; Peters et al., 2019 - 1.5Entmax; Zhang et al., 2021 - Rectified Linear Attention (ReLA)):** Softmax의 대안으로, 어텐션 가중치를 희소하게 만들어 효율성을 높이거나 특정 특성을 부여하는 활성화 함수들.
- **ReLU 초기화 (He et al., 2015):** ReLU 활성화 함수를 사용하는 딥러닝 모델의 가중치 초기화 방법론.

## 🛠️ Methodology

1. **FFN과 키-값 메모리 간의 연결 재조명:**

   - **활성화 함수 교체:** FFN의 ReLU를 Softmax로 대체했을 때 성능 저하(BLEU 점수 34.22 → 33.08)를 관찰했습니다. 이는 Softmax의 출력 분산이 ReLU보다 훨씬 작아 이전 레이어의 잔차에 압도되기 때문으로 분석되었습니다.
   - **Layer Normalization 도입:** Softmax 기반 FFN 출력에 Layer Normalization (LN)을 추가하여 분산을 조절했습니다. 수정된 FFN은 다음과 같습니다: $H = \text{LN}(\text{Softmax}(X \cdot K^T) \cdot V)$. 이로써 FFN과 키-값 메모리의 등가성을 확립했습니다.
   - **메모리 슬롯 크기 확장:** FFN의 히든 차원($d_h$)을 다양한 크기로 변화시키며 ReLU, Softmax, Softmax+LN의 성능을 비교했습니다. ReLU는 슬롯 수가 많을 때 Softmax+LN보다 뛰어난 성능을 보였습니다.
   - **정량적 분석:** Softmax의 점수 분포가 상위 $p\%$ 요소에 과도하게 집중되며(상위 $0.2\%$가 $85\%$ 이상의 점수 차지), 값 공간의 이방성(anisotropy)이 높아 다양한 지식 저장에 비효율적임을 분석했습니다. ReLU는 이러한 문제를 완화했습니다.

2. **Self-Attention Network (SAN)에서 ReLU 활용 (ReLUFormer):**
   - **분산 폭발 문제:** SAN에서 Softmax를 ReLU로 직접 대체 시 모델이 수렴하지 못하는 문제를 발견했습니다. 이는 ReLU 기반 SAN 출력 $h_i$의 분산이 시퀀스 길이 $n$에 따라 $N(0, n/2)$로 동적으로 증가하기 때문입니다.
   - **분산 감소 계수 도입:** SAN의 출력을 $\gamma / \sqrt{n/2}$로 스케일링하여 분산을 안정화했습니다. 수정된 SAN 공식은 다음과 같습니다:
     $$h_i = \sum_{j=1}^{n} \text{ReLU}(q_i^T k_j) \cdot \frac{\gamma}{\sqrt{n/2}} \cdot v_j$$
   - **정규화 손실 함수 도입:** ReLU 기반 SAN의 가중치 분포가 너무 희소하여 문맥 정보를 충분히 활용하지 못하는 문제를 해결하기 위해 두 가지 정규화 손실을 제안했습니다.
     - **정규화 손실:** $\log(\sum_{i=1}^{n} s_i)$ - 가중치의 합을 1에 가깝게 유도하여 비제로 요소 수를 증가시킵니다.
     - **엔트로피-마진 정규화:** $\max(H(s) - C, 0)$ - 가중치 분포의 엔트로피 $H(s) = -\sum_{i=1}^{n} s_i \log(s_i)$를 상한 $C$ (실험에서 $C = 0.7 \log(n)$)보다 작게 유지하여 정보성 있는 분포를 유도합니다.
   - **ReLUFormer 아키텍처:** 위 모든 기법을 통합하여 FFN과 SAN 모두에 ReLU를 사용하는 완전 ReLU Transformer인 ReLUFormer를 제안했습니다. 디코더의 인과적 및 교차 어텐션에도 ReLU를 적용하기 위한 방안을 제시했습니다.

## 📊 Results

- **FFN 활성화 함수 비교:**

  - IWSLT14 De-En 번역 태스크에서 FFN의 ReLU를 Softmax로 대체 시 BLEU 점수가 34.22에서 33.08로 하락했습니다.
  - Softmax FFN에 Layer Normalization을 추가하자 BLEU 점수가 34.21로 회복되어 ReLU FFN과 유사한 성능을 보였습니다.
  - 메모리 슬롯($d_h$)이 3072, 4096과 같이 커질수록 ReLU가 Softmax+LN보다 우수한 성능을 나타내, ReLU의 대규모 값 처리 능력을 입증했습니다.
  - Softmax는 상위 $0.2\%$ 요소가 $85\%$ 이상의 점수를 차지하는 고도로 집중된 분포를 보였고, 이방성 점수가 높아 값 공간의 붕괴를 시사했습니다. ReLU는 이러한 과도한 집중 문제를 완화했습니다.

- **SAN 활성화 함수 비교 (ReLUFormer):**
  - SAN에서 Softmax를 ReLU로 직접 대체했을 때는 모델이 수렴하지 않았습니다.
  - 분산 감소 계수만 적용했을 때는 BLEU 33.19를 달성했지만, 기존 Softmax (34.22)에 미치지 못했습니다.
  - 분산 감소 계수와 정규화 손실을 모두 적용한 ReLUFormer는 IWSLT14 De-En에서 34.56, WMT14 En-De에서 27.64 BLEU를 달성하여 Vanilla Transformer 및 Sparsemax, 1.5Entmax, ReLA와 같은 다른 희소 어텐션 모델들보다 우수한 성능을 보였습니다.
  - 추론 속도 면에서 ReLUFormer는 Vanilla Transformer와 유사하며, Sparsemax 및 1.5Entmax보다 약 1.7배 빨랐습니다.
  - **문서 번역 (장문 시퀀스) 실험:** Europarl7 데이터셋에서 긴 시퀀스(512, 1024, 2048)에 대해 ReLUFormer가 Vanilla Transformer와 Sparsemax보다 지속적으로 우수한 성능을 보였습니다. 특히 1024 길이 시퀀스에서 ReLUFormer는 Softmax Transformer 대비 BLEU 1.15점 향상을 달성했습니다. Sparsemax는 장문 시퀀스에서 훈련에 실패했습니다.
  - 정량적 분석 및 어텐션 맵 시각화를 통해, ReLU는 Softmax보다 덜 중앙 집중적인 분포를 보이며, 더 먼 상관관계를 포착하고 노이즈가 적은 어텐션 맵을 생성하여 장문 시퀀스 모델링에 유리함을 확인했습니다.

## 🧠 Insights & Discussion

- **ReLU와 Softmax의 근본적인 차이:**
  - **분산:** ReLU는 Softmax보다 큰 분산을 가지며 더 풍부한 표현력을 제공합니다. SAN에서 ReLU의 출력 분산은 시퀀스 길이에 비례하여 동적으로 변하며 분산 폭발로 이어질 수 있습니다.
  - **정규화:** Softmax는 지수 정규화로 인해 가중치 분포가 과도하게 중앙 집중화됩니다. 이는 많은 키-값 슬롯이나 긴 시퀀스를 처리할 때 더 넓은 문맥 정보 활용을 제한하여 성능 저하를 초래합니다. ReLU는 이러한 제한이 없어 더 넓은 문맥을 통합할 수 있습니다.
- **FFN과 키-값 메모리의 등가성:** Softmax 기반 키-값 메모리에 Layer Normalization이 추가되면 FFN과 등가성을 갖습니다. LN은 Softmax로 인한 작은 분산 및 과도한 집중 문제를 완화하여 성능을 향상시킵니다.
- **ReLU의 강점:** ReLU는 FFN의 대규모 키-값 슬롯과 SAN의 긴 시퀀스를 효과적으로 처리하는 데 강점을 보입니다. 이는 덜 중앙 집중적인 특성 덕분으로, 시퀀스가 길어질수록 더 많은 토큰의 문맥 정보를 통합할 수 있게 합니다.
- **Transformer의 메모리 네트워크 관점:** 본 연구는 Transformer를 FFN이 전역 메모리 역할을 하고 SAN이 지역 메모리 역할을 하는 통합된 메모리 네트워크로 볼 수 있다는 관점을 강화합니다.
- **한계 및 향후 연구:** 현재 연구는 기계 번역 태스크에만 초점을 맞추고 있으며, SAN의 $O(N^2)$ 복잡도는 여전히 존재합니다. 향후 언어 모델링, 텍스트 요약 등 다양한 태스크로의 확장과 더 효율적인 어텐션 메커니즘을 통한 Latency 개선 연구가 필요합니다.

## 📌 TL;DR

본 논문은 트랜스포머 아키텍처 내 ReLU와 Softmax 활성화 함수의 차이점을 심층 분석했습니다. FFN과 SAN에서 Softmax가 대규모 메모리 슬롯이나 긴 시퀀스에서 과도한 가중치 집중으로 문맥 활용을 제한하는 문제를 발견했습니다. 이를 해결하기 위해 FFN에 Layer Normalization을 추가하여 FFN과 키-값 메모리의 등가성을 확립하고, SAN에 ReLU를 적용할 때 발생하는 분산 폭발을 분산 감소 계수와 정규화 손실로 안정화하여 **ReLUFormer**를 제안했습니다. ReLUFormer는 단문 번역에서 기존 Transformer보다 우수하며, 특히 장문 문서 번역에서 Softmax 기반 모델들을 크게 능가하여, ReLU가 더 넓은 문맥을 효과적으로 활용할 수 있음을 입증했습니다.
