# Longformer: The Long-Document Transformer

Iz Beltagy, Matthew E. Peters, Arman Cohan

## 🧩 Problem to Solve

기존 트랜스포머(Transformer) 모델은 셀프 어텐션(self-attention) 메커니즘이 시퀀스 길이 $n$에 대해 $O(n^2)$의 시간 및 메모리 복잡도를 가지므로, 수천 토큰 이상의 긴 문서를 효율적으로 처리하는 데 어려움이 있습니다. 이로 인해 긴 문서를 처리해야 하는 자연어 처리(NLP) 태스크에서는 문서를 분할하거나 축소해야 했고, 이 과정에서 중요한 맥락 정보가 손실되거나 복잡한 아키텍처가 필요하다는 한계가 있었습니다.

## ✨ Key Contributions

- **선형 확장성 어텐션 메커니즘 개발**: 시퀀스 길이에 대해 선형적으로 확장되는 어텐션 메커니즘을 도입하여 긴 문서 처리를 가능하게 했습니다.
- **지역 및 전역 어텐션 결합**: 고정 크기 슬라이딩 윈도우 어텐션(sliding window attention)과 태스크에 특화된 전역 어텐션(global attention)을 결합하여 컨텍스트 정보를 효과적으로 포착합니다.
- **사전 학습 및 파인튜닝 성공**: 대부분의 선행 연구가 자기회귀 언어 모델링에 집중했던 것과 달리, Longformer를 사전 학습하고 이를 다양한 다운스트림 태스크에 파인튜닝하여 긴 문서 NLP 태스크에서 RoBERTa를 꾸준히 능가함을 입증했습니다.
- **최첨단(SOTA) 성능 달성**: 문자 단위 언어 모델링(character-level language modeling)에서 `text8` 및 `enwik8` 데이터셋에서 SOTA 결과를 달성했습니다. 또한, WikiHop 및 TriviaQA와 같은 QA 태스크에서 새로운 SOTA를 기록했습니다.
- **LED(Longformer-Encoder-Decoder) 소개**: 긴 문서 생성(generative) 시퀀스-투-시퀀스(sequence-to-sequence) 태스크를 지원하는 Longformer 변형 모델을 도입하고, arXiv 요약 데이터셋에서 효과를 입증했습니다.

## 📎 Related Works

- **긴 문서 트랜스포머**: 시퀀스 길이 제한을 해결하기 위한 두 가지 주요 접근 방식이 있었습니다.
  - **좌-우(Left-to-Right) 접근 방식**: 문서를 청크 단위로 처리하며, 주로 자기회귀 언어 모델링에 사용되었지만 양방향 컨텍스트가 필요한 전이 학습(transfer learning)에는 부적합했습니다 (예: Transformer-XL, Adaptive Span).
  - **희소 어텐션(Sparse Attention) 패턴**: 전체 어텐션 행렬 계산을 피하고 특정 패턴의 어텐션만 계산합니다. Sparse Transformer와 유사한 접근 방식을 사용하지만, Longformer는 더 유연한 CUDA 커널과 태스크에 특화된 전역 어텐션 패턴을 추가했습니다.
- **태스크별 긴 문서 모델**: 기존 BERT-스타일 모델의 512 토큰 제한을 우회하기 위해 문서를 자르거나, 청크로 분할 후 결합하거나, 두 단계 모델을 사용하는 등의 복잡한 접근 방식이 있었습니다. Longformer는 이러한 방식에서 발생하는 정보 손실이나 계단식 오류 없이 전체 시퀀스를 한 번에 처리할 수 있게 합니다.
- **동시대 연구**: ETC, GMAT, BigBird와 같은 동시대 연구들도 로컬(local) + 전역(global) 어텐션 아이디어를 탐구했으나, Longformer는 사전 학습 및 파인튜닝 맥락에서의 광범위한 평가와 유연한 아키텍처를 강점으로 가집니다.

## 🛠️ Methodology

Longformer의 핵심은 시퀀스 길이에 대해 선형적으로 확장되는 효율적인 어텐션 패턴을 사용하는 것입니다.

1. **어텐션 패턴($O(n)$ 복잡도)**:
   - **슬라이딩 윈도우 어텐션(Sliding Window Attention)**: 각 토큰은 고정 크기 $w$의 윈도우 내의 토큰에만 어텐션합니다. 여러 레이어를 쌓으면 CNN처럼 전체 입력에 접근하는 큰 수용 필드(receptive field)를 형성할 수 있습니다. 계산 복잡도는 $O(n \times w)$입니다.
   - **확장 슬라이딩 윈도우(Dilated Sliding Window)**: 윈도우 내에 $d$ 크기의 간격을 두어 계산량 증가 없이 수용 필드를 더욱 확장합니다. 각 어텐션 헤드마다 다른 확장 설정(dilation configuration)을 사용하여 지역 및 장거리 컨텍스트를 동시에 포착할 수 있습니다.
   - **전역 어텐션(Global Attention)**: 슬라이딩 윈도우 어텐션으로는 태스크별 표현을 학습하기에 유연성이 부족하므로, 특정 입력 위치에 "전역 어텐션"을 추가합니다. 전역 어텐션이 지정된 토큰은 시퀀스 내의 모든 토큰에 어텐션하고, 모든 토큰 또한 해당 전역 토큰에 어텐션합니다. ([CLS] 토큰, 질문 토큰 등) 전역 토큰의 수가 $n$에 비해 적기 때문에 전체 복잡도는 여전히 $O(n)$입니다.
2. **전역 어텐션을 위한 선형 투영(Linear Projections)**: 슬라이딩 윈도우 어텐션($Q_s, K_s, V_s$)과 전역 어텐션($Q_g, K_g, V_g$)을 위해 각각 별도의 선형 투영(linear projection)을 사용합니다. 이는 각 어텐션 유형을 모델링하는 데 유연성을 제공하며, 성능 향상에 필수적입니다. 초기화 시 $Q_g, K_g, V_g$는 $Q_s, K_s, V_s$의 값과 동일하게 설정됩니다.
3. **구현**: 희소 어텐션 패턴은 PyTorch/TensorFlow 같은 기존 딥러닝 라이브러리에서 직접 지원하지 않는 밴드 행렬 곱셈(banded matrix multiplication)을 필요로 합니다. 이를 위해 `Longformer-chunk` (비확장 윈도우에 최적화된 벡터화 구현)와 `Longformer-cuda` (TVM을 사용하여 구현된 확장 슬라이딩 윈도우를 지원하는 맞춤형 CUDA 커널)를 개발하여 효율성을 확보했습니다.
4. **사전 학습 및 파인튜닝**:
   - **사전 학습**: RoBERTa의 사전 학습된 체크포인트에서 시작하여 마스크 언어 모델링(MLM) 목표로 Longformer를 추가 사전 학습합니다.
   - **위치 임베딩(Position Embeddings)**: RoBERTa의 최대 512 토큰 위치 임베딩을 4,096 토큰까지 확장하기 위해, 기존 512개의 임베딩을 반복 복사하여 초기화합니다. 이는 RoBERTa의 지역 컨텍스트 편향을 활용하면서 새로운 위치 임베딩 학습을 가속화합니다.
5. **Longformer-Encoder-Decoder (LED)**: 시퀀스-투-시퀀스 태스크를 위해 인코더-디코더 아키텍처를 도입합니다. 인코더에는 Longformer의 효율적인 어텐션 패턴을 사용하고, 디코더는 인코딩된 토큰과 이전 디코딩 위치에 대해 전체 셀프 어텐션을 사용합니다. BART 모델에서 파라미터를 초기화하고 위치 임베딩을 16K 토큰까지 확장합니다.

## 📊 Results

- **문자 단위 언어 모델링**:
  - `text8` 및 `enwik8` 데이터셋에서 소규모 모델(small models)을 사용하여 각각 BPC 1.10, 1.00으로 새로운 SOTA를 달성했습니다.
  - 대규모 모델(large models)의 경우 `enwik8`에서 Transformer-XL과 동등하거나 Sparse Transformer와 일치하는 성능을 보였습니다.
- **사전 학습 및 파인튜닝**:
  - RoBERTa에서 이어서 사전 학습된 Longformer는 4,096 토큰의 긴 시퀀스를 처리할 수 있으며, RoBERTa에 비해 BPC가 크게 향상되었습니다.
  - **QA, 상호참조 해결, 문서 분류**: WikiHop, TriviaQA, HotpotQA (QA), OntoNotes (상호참조), IMDB, Hyperpartisan news (분류) 등 다양한 긴 문서 NLP 태스크에서 RoBERTa baseline을 꾸준히 능가했습니다. 특히 WikiHop과 Hyperpartisan news와 같이 긴 컨텍스트가 필수적인 태스크에서 큰 성능 향상을 보였습니다.
  - **WikiHop 및 TriviaQA SOTA**: Longformer-large 모델은 WikiHop과 TriviaQA에서 각각 3.6점, 4점 향상된 F1 점수로 새로운 SOTA를 달성했습니다.
- **LED 요약**: arXiv 요약 데이터셋에서 Longformer-Encoder-Decoder (LED)는 추가 사전 학습 없이 BART로부터 초기화되었음에도 불구하고, 긴 입력(16,384 토큰)을 처리하는 능력으로 BigBird를 약간 능가하며 SOTA를 달성했습니다. 입력 시퀀스 길이가 길수록 ROUGE 점수가 유의미하게 향상됨을 확인했습니다.
- **제거 연구(Ablation Study)**: Longformer의 성능 향상은 긴 시퀀스, 전역 어텐션, 전역 어텐션을 위한 별도 선형 투영, MLM 사전 학습, 그리고 더 긴 훈련 기간 덕분임을 확인했습니다.

## 🧠 Insights & Discussion

- **효율적인 긴 문서 처리**: Longformer는 기존 트랜스포머의 $O(n^2)$ 복잡도 한계를 극복하여 긴 문서를 청킹(chunking)하거나 단축할 필요 없이 효율적으로 처리할 수 있습니다. 이는 정보 손실을 방지하고 복잡한 태스크별 아키텍처의 필요성을 줄여줍니다.
- **유연한 어텐션 디자인**: 지역 윈도우 어텐션과 태스크에 특화된 전역 어텐션의 결합은 Longformer가 다양한 NLP 태스크의 요구사항에 맞춰 유연하게 컨텍스트를 학습할 수 있도록 합니다. 특히 전역 어텐션이 특정 토큰에 집중함으로써 중요한 정보를 쉽게 통합할 수 있게 합니다.
- **전이 학습의 확장**: Longformer는 긴 문서 NLP 태스크에 대한 전이 학습 패러다임을 성공적으로 확장했습니다. RoBERTa와 같은 기존 모델의 사전 학습 가중치를 활용하면서도, 긴 시퀀스 처리 능력을 추가하여 광범위한 다운스트림 태스크에서 성능 향상을 이끌어냈습니다.
- **LED의 잠재력**: Longformer-Encoder-Decoder (LED)는 긴 문서 요약과 같은 시퀀스-투-시퀀스 생성 태스크에서 뛰어난 성능을 보여주며, 인코더-디코더 모델에서도 효율적인 긴 시퀀스 처리가 가능함을 입증했습니다.
- **제한 및 향후 연구**: 더 많은 사전 학습 목표 탐색(특히 LED를 위해), 처리 가능한 시퀀스 길이 증가, Longformer 모델이 혜택을 받을 수 있는 다른 태스크 탐색 등이 향후 연구로 제안됩니다.

## 📌 TL;DR

Longformer는 트랜스포머의 셀프 어텐션 $O(n^2)$ 복잡도로 인한 긴 문서 처리 한계를 해결하기 위해, 시퀀스 길이에 선형적으로 비례하는(O(n)) 효율적인 어텐션 메커니즘을 제안합니다. 이 메커니즘은 지역 윈도우 어텐션과 태스크별 전역 어텐션을 결합하며, RoBERTa 기반의 사전 학습 및 파인튜닝을 통해 문자 단위 언어 모델링 및 다양한 긴 문서 NLP 다운스트림 태스크(QA, 분류)에서 SOTA 성능을 달성했습니다. 또한, 시퀀스-투-시퀀스 태스크를 위한 Longformer-Encoder-Decoder(LED)를 도입하여 긴 문서 요약에서 SOTA를 기록했습니다. Longformer는 긴 문서 처리 시 정보 손실 없이 간단하고 효과적인 모델링을 가능하게 합니다.
