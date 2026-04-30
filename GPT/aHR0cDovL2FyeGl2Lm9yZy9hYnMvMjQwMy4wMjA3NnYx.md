# VTG-GPT: Tuning-Free Zero-Shot Video Temporal Grounding with GPT

Yifang Xu, Yunzhuo Sun, Zien Xie, Benxiang Zhai, and Sidan Du (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Video Temporal Grounding (VTG) 분야에서 발생하는 데이터 의존성 및 인간의 주관적 편향(human bias) 문제이다. VTG는 자연어 쿼리가 주어졌을 때, 비정형 비디오 내에서 해당 쿼리와 가장 관련이 깊은 시간적 구간(start and end timestamp)을 찾아내는 작업이다.

기존의 지도 학습(supervised learning) 기반 VTG 모델들은 방대한 양의 비디오-텍스트 쌍 데이터셋을 필요로 하며, 이는 막대한 계산 비용과 인건비를 발생시킨다. 또한, 사람이 직접 작성한 ground-truth 쿼리에는 오타(misspelling)나 비디오 내용과 일치하지 않는 잘못된 설명(incorrect descriptions)과 같은 편향이 포함되어 있어, 모델이 잘못된 정보를 학습하거나 오작동하게 만드는 원인이 된다. 한편, 기존의 제로샷(zero-shot) 방식들은 주로 raw 비디오 프레임의 특징(feature)을 직접 사용하는데, 이는 텍스트에 비해 비디오 데이터가 가진 중복 정보(redundant information)가 너무 많아 정확한 세부 내용 인식에 어려움을 겪는 한계가 있다.

따라서 본 논문의 목표는 별도의 훈련이나 미세 조정(fine-tuning) 없이도, 대규모 언어 모델(LLM)과 다중 모달 모델(LMM)을 활용하여 인간의 편향을 제거하고 비디오의 중복 정보를 줄임으로써 높은 성능을 내는 tuning-free 제로샷 VTG 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비디오와 쿼리 모두를 '언어적 영역(linguistic domain)'으로 변환하여 분석함으로써 모델의 부담을 줄이고 정확도를 높이는 것이다.

첫째, LLM인 Baichuan2를 사용하여 원본 쿼리의 오타를 수정하고, 문맥에 맞게 재구성하며, 다양한 표현의 쿼리를 생성함으로써 인간이 작성한 쿼리에 내재된 편향을 제거하는 Query Debiasing 전략을 제안한다.

둘째, LMM인 MiniGPT-v2를 활용하여 비디오의 각 프레임을 상세한 텍스트 캡션으로 변환한다. 이는 비디오의 raw 픽셀 데이터에서 발생하는 불필요한 배경 정보나 중복 데이터를 제거하고, 모델이 핵심적인 시각적 내용에만 집중할 수 있도록 돕는다.

셋째, 이렇게 정제된 '디바이아스된 쿼리'와 '이미지 캡션' 사이의 유사도를 계산하여 시간적 구간을 예측하는 proposal generator와 length-aware scoring 기반의 후처리 과정을 설계하여 훈련 없이도 높은 성능의 구간 추출을 가능하게 하였다.

## 📎 Related Works

기존의 VTG 연구는 크게 네 가지 방향으로 나뉜다.
1. **Fully-supervised (FS)**: Moment-DETR와 같이 대량의 데이터를 통해 시각-텍스트 특징을 정렬하는 모델들이다. 성능은 높으나 데이터 구축 비용이 매우 크다.
2. **Weakly-supervised (WS)**: 비디오-쿼리 쌍만 사용하고 정확한 타임스탬프 없이 학습하는 방식이다.
3. **Unsupervised (US)**: 클러스터링을 통해 유사 쿼리를 생성하거나 CLIP과 같은 사전 학습 모델을 사용하여 유사도를 측정한다. 하지만 비디오-쿼리 쌍의 불일치로 인한 편향 문제가 존재한다.
4. **Zero-shot (ZS)**: CLIP이나 BLIP-2 같은 동결된 비전-언어 모델을 활용한다. 하지만 raw 프레임의 중복 정보로 인해 성능 향상에 한계가 있다.

본 논문의 VTG-GPT는 위 방법들과 달리 어떠한 훈련이나 어댑터 설계 없이 순수하게 추론(inference)만으로 동작하며, 특히 시각 정보를 텍스트 캡션으로 변환하여 처리한다는 점에서 기존의 제로샷 방식과 차별화된다.

## 🛠️ Methodology

VTG-GPT의 전체 파이프라인은 Query Debiasing $\rightarrow$ Image Captioning $\rightarrow$ Proposal Generation $\rightarrow$ Post-processing의 4단계로 구성된다.

### 1. Query Debiasing
사람이 작성한 쿼리 $T$의 편향을 제거하기 위해 Baichuan2를 사용한다. 과정은 다음과 같다.
- **오타 및 문법 수정**: 원본 쿼리 $T$를 문법적으로 정확한 $T^c$로 수정한다.
- **의도 유지 및 재구성**: 수정된 쿼리 $T^c$를 기반으로, 동일한 의도를 유지하면서 표현만 다르게 재구성한다.
- **다양성 확보**: 최종적으로 의미는 같지만 구문적으로 서로 다른 5개의 디바이아스된 쿼리 $Q$를 생성하여 모델이 특정 표현에 의존하지 않게 한다.

### 2. Image Captioning
비디오 $V$의 시각적 중복성을 줄이기 위해 MiniGPT-v2를 사용하여 각 프레임을 텍스트 캡션 $C$로 변환한다. 구체적으로 `[image caption] Please describe the content of this image in detail.`라는 프롬프트를 통해 각 프레임의 핵심 내용을 텍스트로 추출한다. 이를 통해 배경 잡음을 제거하고 의미론적 유사도 측정의 정확도를 높인다.

### 3. Proposal Generation
이제 모든 입력이 텍스트 영역으로 변환되었으므로, Sentence-BERT를 사용하여 디바이아스된 쿼리 $Q$와 이미지 캡션 $C$ 사이의 코사인 유사도 $S_s$를 계산한다.

$$S_s = \cos(f_q, f_c) = \frac{f_q \cdot f_c}{\|f_q\|\|f_c\|}$$

여기서 $f_q$와 $f_c$는 각각 쿼리와 캡션의 정규화된 풀링 특징 벡터이다. 이후 dynamic threshold $\theta$를 설정하여 구간을 생성한다. $\theta$는 유사도 $S_s$의 히스토그램에서 상위 $k$개의 빈(bin)에 해당하는 값으로 결정된다. 유사도가 $\theta$를 초과하는 프레임을 시작점으로 잡고, $\lambda$개 이상의 연속된 프레임이 $\theta$보다 낮아지면 해당 구간을 종료점으로 설정하여 temporal proposals $P$를 생성한다.

### 4. Post-Processing
생성된 다수의 proposal 중 최적의 구간을 선택하기 위해 length-aware scoring을 적용한다. 단순 유사도 평균 대신, 구간의 길이와 유사도를 동시에 고려하는 최종 점수 $S_f$를 계산한다.

$$S_f = \alpha \times S_l + (1 - \alpha) \times S_s$$

여기서 $S_l = L_p / L_n$이며, $L_p$는 해당 proposal 내에서 $\theta$를 초과하는 프레임 수, $L_n$은 비디오 전체에서 $\theta$를 초과하는 총 프레임 수이다. 마지막으로 IoU 임계값 $\mu$를 이용한 Non-Maximum Suppression (NMS)을 통해 중복되는 구간을 제거하고 최종 구간 $Seg$를 도출한다.

## 📊 Results

### 실험 설정
- **데이터셋**: QVHighlights, Charades-STA, ActivityNet-Captions 세 가지 데이터셋을 사용하였다.
- **측정 지표**: Recall-1 at IoU thresholds (R1@m), mean Average Precision (mAP), mean IoU (mIoU)를 사용하였다.
- **구현 세부사항**: MiniGPT-v2와 Baichuan2-7B-Chat를 사용하였으며, 유사도 모델로는 Sentence-BERT를 채택하였다. 모든 과정은 훈련 없이 추론으로만 진행되었다.

### 주요 결과
- **SOTA 비교**: QVHighlights 데이터셋에서 이전의 SOTA 제로샷 모델(Diwan et al.)보다 R1@0.7에서 +7.49, mAP@0.5에서 +7.23의 큰 폭의 성능 향상을 보였다.
- **지도 학습 모델과의 비교**: 놀랍게도 Fully-supervised 방식인 Moment-DETR와 경쟁 가능한 수준의 성능을 달성하였으며, 일부 지표에서는 이를 능가하였다.
- **데이터셋별 성능**: Charades-STA에서도 제로샷 SOTA(Luo et al.)를 앞섰으나, ActivityNet-Captions에서는 약간 낮은 성능을 보였는데, 이는 긴 비디오의 경우 계산 자원 한계로 인해 다운샘플링 비율을 높게 잡았기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구의 가장 큰 성과는 학습 데이터 없이 LLM과 LMM의 추론 능력만으로 지도 학습 모델에 근접한 성능을 낸 것이다. 특히 Query Debiasing의 효과가 컸는데, 사람이 작성한 쿼리의 오타를 수정하고 다양한 표현으로 확장함으로써 텍스트 인코더가 의미론적 정보를 더 풍부하게 포착할 수 있게 하였다. 또한, 비디오를 직접 처리하지 않고 캡션으로 변환하여 처리한 것이 시각적 노이즈를 획기적으로 줄이는 역할을 하였다.

### 한계 및 비판적 해석
1. **계산 효율성 문제**: 모든 프레임을 LMM(MiniGPT-v2)을 통해 캡션으로 변환하는 과정은 매우 많은 추론 시간을 소요할 가능성이 크다. 실제 서비스 적용 시에는 이 병목 현상을 해결할 효율적인 모델이 필요하다.
2. **시간적 맥락의 손실**: 프레임 단위의 캡션을 생성한 뒤 유사도를 계산하는 방식은, 비디오의 '흐름'이나 '동적인 변화'와 같은 시간적 맥락(temporal context)을 완전히 반영하지 못하고 정적인 이미지들의 집합으로 취급한다는 한계가 있다.
3. **쿼리 개수의 임계점**: 실험 결과 디바이아스된 쿼리를 5개까지 늘릴 때는 성능이 향상되었으나, 그 이상에서는 오히려 성능이 하락하였다. 이는 과도한 재구성이 원본 쿼리의 의도에서 벗어난 내용을 생성하기 때문으로 보이며, 최적의 재구성 횟수를 결정하는 기준에 대한 추가 연구가 필요하다.

## 📌 TL;DR

VTG-GPT는 훈련이나 미세 조정이 전혀 필요 없는 **tuning-free 제로샷 비디오 시간적 구간 추출(VTG)** 프레임워크이다. LLM(Baichuan2)으로 쿼리의 편향을 제거하고, LMM(MiniGPT-v2)으로 비디오 프레임을 텍스트 캡션으로 변환하여, 모든 문제를 **텍스트 도메인에서의 유사도 측정 문제**로 치환하여 해결한다. 실험 결과, 기존 제로샷 모델들을 압도하고 지도 학습 모델과 대등한 성능을 보여주었으며, 이는 LLM/LMM의 강력한 일반화 능력을 VTG 작업에 성공적으로 적용했음을 입증한다. 향후 비디오 기반 GPT(Video-based GPT)를 도입한다면 시간적 맥락 파악 능력이 더욱 향상될 것으로 기대된다.