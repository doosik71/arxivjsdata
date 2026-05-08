# Zero-shot Video Moment Retrieval With Off-the-Shelf Models

Anuj Diwan, Puyuan Peng, Raymond J. Mooney (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Video Moment Retrieval (VMR) 분야에서 발생하는 데이터 수집의 고비용 문제와 대규모 사전 학습 모델의 파인튜닝(fine-tuning)에 필요한 막대한 컴퓨팅 자원 문제이다. VMR은 비디오와 자연어 쿼리가 주어졌을 때, 비디오 내에서 쿼리와 관련된 특정 시간 구간(moment)을 찾아내는 작업이다.

기존의 VMR 모델들은 고품질의 인간 주석(human-annotated) 데이터셋에 크게 의존하며, 최신 Transformer 기반 모델들을 특정 작업에 맞게 최적화하기 위해서는 매우 높은 연산 비용이 발생한다. 특히, CLIP이나 GPT-3와 같이 모델 가중치가 API 뒤에 숨겨져 있거나 규모가 너무 커서 일반적인 연구자가 직접 파인튜닝하기 어려운 경우가 많다. 따라서 본 연구의 목표는 추가적인 파인튜닝 없이, 공개된 기존 모델(off-the-shelf models)만을 활용하여 VMR 작업을 수행하는 효율적인 Zero-shot 접근 방식을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 VMR 과정을 **'모먼트 제안(Moment Proposal) $\rightarrow$ 모먼트-쿼리 매칭(Moment-Query Matching) $\rightarrow$ 후처리(Post-processing)'**라는 3단계 파이프라인으로 분해하고, 각 단계에서 훈련되지 않은 기존의 공개 모델들을 조합하여 사용하는 것이다.

중심적인 직관은 VMR을 위한 전용 데이터셋으로 모델을 학습시키는 대신, 이미 이미지-텍스트 정렬(alignment)이 잘 되어 있는 CLIP이나 비디오 캡셔닝 모델을 그대로 재사용함으로써 데이터 부족 문제를 우회할 수 있다는 점이다. 특히, 비디오의 장면 전환을 감지하는 Shot Detection 기술을 제안 단계에 도입하여, 고정된 윈도우 방식보다 더 의미 있는 후보 구간을 추출하고자 하였다.

## 📎 Related Works

### 기존 VMR 접근 방식

최근의 SOTA(State-of-the-Art) 모델인 Moment-DETR나 UMT는 세그먼트 제안과 신뢰도 점수 계산을 동시에 수행하는 엔드 투 엔드(end-to-end) 딥러닝 네트워크를 설계하고, 이를 VMR 데이터셋으로 학습시킨다. 반면 MCN, CAL, XML과 같은 초기 모델들은 제안 단계와 점수 계산 단계를 분리한 파이프라인 방식을 사용하였으나, 이 역시 각 구성 요소를 VMR 데이터셋으로 학습시켜야 했다는 한계가 있다.

### Zero-shot Transfer 및 CLIP

CLIP과 같은 Foundation Model의 등장으로, 새로운 작업에 대해 학습 데이터 없이 혹은 매우 적은 데이터만으로 지식을 전이하는 Zero-shot/Few-shot 학습이 가능해졌다. CLIP은 이미지-텍스트 매칭 능력이 뛰어나 다양한 시각-언어 작업에 활용되고 있으며, 본 논문은 이를 VMR의 매칭 단계에 적용하여 차별화를 꾀하였다.

## 🛠️ Methodology

본 논문이 제안하는 제로샷 파이프라인은 다음과 같은 세 단계로 구성된다.

### 1. Generating Moment Proposals (모먼트 제안)

비디오 $V$에서 쿼리 $Q$와 관련 있을 법한 $K$개의 후보 구간 $\{v_1, v_2, \dots, v_K\}$을 생성한다.

- **ShotDetect**: `PySceneDetect` 툴킷의 content-aware detector를 사용하여 장면 전환(shot transition)을 감지한다. 인접한 프레임 간의 색상 변화가 임계값 $\lambda$보다 클 때 장면이 바뀌었다고 판단하며, 이를 통해 서로 겹치지 않는 의미 있는 세그먼트들을 추출한다.
- **SlidingWindow (Baseline)**: 비교를 위해 15초 길이의 윈도우를 10초 간격으로 생성하는 단순한 방식도 함께 제안한다.

### 2. Moment-Query Matching (모먼트-쿼리 매칭)

추출된 각 세그먼트 $v_k$와 쿼리 $Q$ 사이의 유사도 점수 $s_k \in [0, 1]$를 계산한다. 두 가지 방법을 탐색하였다.

- **VideoCaptioning**:
    1. UniVL 모델을 사용하여 각 세그먼트 $v_k$에 대한 자연어 캡션 $l_k$를 생성한다.
    2. MPNet 모델 $E$를 사용하여 캡션 $E(l_k)$와 쿼리 $E(Q)$를 임베딩 공간으로 투영한다.
    3. 두 임베딩 간의 코사인 유사도를 계산하여 $s_k$를 산출한다.
- **CLIP**:
    1. 세그먼트 $v_k$에서 1fps 비율로 $M_k$개의 프레임을 샘플링한다.
    2. CLIP의 Image Encoder와 Text Encoder를 사용하여 각 프레임 $f_i^{(k)}$와 쿼리 $Q$를 임베딩한다.
    3. 각 프레임과 쿼리 사이의 코사인 유사도 $\{s_1^{(k)}, \dots, s_{M_k}^{(k)}\}$를 구한 후, 집계 함수 $f$ (본 논문에서는 $\max$ 사용)를 통해 세그먼트의 최종 점수를 결정한다.
    $$s_k = \max(s_1^{(k)}, s_2^{(k)}, \dots, s_{M_k}^{(k)})$$

### 3. Post-processing (후처리)

계산된 점수를 바탕으로 최종 구간을 결정하기 위해 **SimpleWatershed** 알고리즘을 사용한다.

- 유사도 점수 임계값 $\gamma$를 설정한다.
- 연속된 세그먼트들 $\{v_i, \dots, v_j\}$의 점수가 모두 $\gamma$ 이상인 경우, 이들을 하나의 단일 세그먼트로 병합한다.
- 병합된 세그먼트의 점수는 해당 구간 내의 최대 점수 $\max(s_i, \dots, s_j)$로 설정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: QVHighlights의 filtered validation set (val-filt)을 사용하였다.
- **평가 지표**: mAP (IoU 0.5, 0.75 및 평균), Recall@1 (IoU 0.5, 0.7)을 사용하였다.
- **비교 대상**:
  - VMR-Supervised: MCN, CAL, XML, Moment-DETR (saliency loss 제외)
  - VMR-Supervised + Saliency: XML+, Moment-DETR, UMT
  - Zero-Shot: 기존 CLIP+Watershed 베이스라인

### 주요 결과

1. **제로샷 성능 향상**: 제안된 `ShotDetect + CLIP + SimpleWatershed` 조합은 기존의 제로샷 베이스라인보다 모든 지표에서 최소 2.5배, 최대 5배(R1@0.7 기준) 높은 성능을 보였다.
2. **지도 학습 모델과의 비교**: 놀랍게도 본 연구의 제로샷 접근 방식이 saliency-score 없이 학습된 모든 VMR-Supervised 모델들의 Recall 지표를 능가하였으며, mAP 지표에서도 Moment-DETR (w/o saliency)와 경쟁 가능한 수준임을 보여주었다.
3. **Saliency 기반 모델과의 격차**: 하지만 프레임 레벨의 saliency score를 사용하여 학습된 모델(UMT w/ PT 등)과는 여전히 상당한 성능 격차가 존재하였다.
4. **구간 길이에 따른 분석**:
    - 모든 모델이 짧은 구간(10초 미만) 탐색에 어려움을 겪지만, 본 제안 방식은 지도 학습 모델들보다 **짧은 구간에서 훨씬 더 높은 mAP**를 기록하였다.
    - 반면, 긴 구간(30초 초과)에서는 지도 학습 모델들이 더 우세하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 추가 학습 없이 공개 모델들의 조합만으로도 강력한 VMR 베이스라인을 구축할 수 있음을 입증하였다. 특히 짧은 구간에서 성능이 좋게 나타난 점은 Shot Detection을 통한 제안 단계가 지도 학습 모델들이 갖지 못한 유용한 귀납적 편향(inductive bias)을 제공했기 때문으로 해석된다.

### 한계 및 비판적 해석

가장 큰 한계는 프레임 레벨의 정밀한 주석(saliency score)이 제공된 모델과의 성능 차이이다. 이는 단순한 이미지-텍스트 매칭만으로는 비디오 내의 '중요도'를 완전히 파악하기 어렵다는 것을 의미한다. 또한, CLIP을 QVHighlights 데이터로 파인튜닝하려는 시도가 있었으나, 유의미한 성능 향상이 없었다는 점은 제로샷 성능이 이미 모델의 잠재 능력을 거의 다 활용하고 있거나, 혹은 현재의 파인튜닝 전략이 부족했음을 시사한다.

### Oracular Bounds의 의미

완벽한 매칭 모델이 있다고 가정했을 때의 상한선(Oracular Bound)을 분석한 결과, Postprocessing bound가 매우 높게 나타났다. 이는 제안-매칭-후처리 파이프라인 자체가 이론적으로는 매우 강력하며, 앞으로 매칭 모델의 정밀도만 높인다면 SOTA 모델을 충분히 뛰어넘을 가능성이 있음을 보여준다.

## 📌 TL;DR

본 논문은 **Shot Detection $\rightarrow$ CLIP Matching $\rightarrow$ SimpleWatershed**로 이어지는 3단계 파이프라인을 통해, 추가 학습이 전혀 필요 없는 **Zero-shot Video Moment Retrieval** 방법을 제안하였다. 이 방법은 기존 제로샷 모델 대비 성능을 최대 5배 향상시켰으며, 일부 지도 학습 모델보다 뛰어난 Recall 성능을 보였고, 특히 짧은 비디오 구간 탐색에서 강점을 보였다. 이는 향후 데이터 수집 비용을 줄이면서도 고성능 VMR 시스템을 구축하는 데 중요한 기초 연구가 될 것으로 보인다.
